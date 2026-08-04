// Verifies SSIM fwd/bwd (spec 08 §3–§7, Phase 3 Task 2) — fixtures/ssim_basic
// (24×24×3, valid map 14×14, ssim=0.585063, v_img1 = d(1-ssim)/d(img1) with upstream≡1 — an f64
// golden locked down by six self-verifications, task-1-report §2).
//
// The comparison contract: ssim, ssim_map, and v_img1 all use the meta.json tolerances (never
// hardcode them in tests — the 07 open question #8 procedure). Neither fwd nor bwd uses atomics, so
// **bit-exact determinism** across two Metal runs (memcmp) is a gate (08 §7). Boundary coverage:
//  - fwd 24×24: the map is a single 14×14 tile, yet the 26×26 apron exceeds the image (24×24) —
//    **partial-tile apron boundary masking is triggered by the fixture itself** (08 §5 correction).
//  - bwd 24×24: grid 2×2 (measured in input pixels) — the right and bottom partial tiles plus the
//    zero-padded scatter of the coefficient map (14×14) (the apron permanently covers the region
//    outside the map on all four sides) are triggered together.
//  - ssim_partial_tile_boundary: a 40×40 synthetic case (map 30×30 → fwd grid 2×2) covering
//    multiple tiles, embedding invariance (bit-exact), and a cross-check against the CPU reference.
#include "TestHarness.h"
#include "TestUtils.h" // shared_metal_backend
#include "KernelConstants.h" // kSsimWindowSize (defined once, shared with the kernel)
#include "splatx/CpuKernels.h"
#include "splatx/Json.h"
#include "splatx/KernelBackend.h"
#include "splatx/SXTensor.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace splatx;

namespace {

std::string fx(const char* rel) { return sxtest::fixture_dir() + "/ssim_basic/" + rel; }

struct SsimFixture {
    std::map<std::string, double> meta;
    SXTensor img1, img2;          // inputs
    SXTensor ssim, ssim_map;      // fwd golden
    SXTensor v_img1;              // bwd golden (autograd, upstream≡1, λ not applied)
};

SsimFixture load_fixture() {
    SsimFixture f;
    f.meta = json_parse_flat(fx("meta.json"));
    f.img1 = SXTensor::load(fx("img1.sxt"));
    f.img2 = SXTensor::load(fx("img2.sxt"));
    f.ssim = SXTensor::load(fx("ssim.sxt"));
    f.ssim_map = SXTensor::load(fx("ssim_map.sxt"));
    f.v_img1 = SXTensor::load(fx("v_img1.sxt"));
    return f;
}

SsimInputs make_inputs(const SsimFixture& f, bool need_coeffs) {
    SsimInputs in;
    in.img1 = f.img1.f32();
    in.img2 = f.img2.f32();
    in.I = static_cast<int64_t>(f.meta.at("I"));
    in.H = static_cast<int64_t>(f.meta.at("height"));
    in.W = static_cast<int64_t>(f.meta.at("width"));
    in.C = static_cast<int64_t>(f.meta.at("C"));
    in.need_coeffs = need_coeffs;
    return in;
}

// Compares fwd against the golden — both the map and the scalar use the meta tolerances (the scalar
// goes through the same path as a 1-element array).
// max rel is INFO for the record (not a gate) — it is the evidence behind measured tolerance
// adjustments (the D1 / 07 procedure).
void compare_fwd_golden(const char* tag, const SsimFwdOutputs& out, const SsimFixture& f) {
    double max_rel = 0.0;
    for (std::size_t i = 0; i < out.ssim_map.size() && i < f.ssim_map.count(); i++) {
        const double e = f.ssim_map.f32()[i];
        if (e != 0.0)
            max_rel = std::max(max_rel, std::fabs(out.ssim_map[i] - e) / std::fabs(e));
    }
    std::printf("  [%s] ssim=%.9f (golden %.9f), map max rel=%.3g\n", tag, out.ssim,
                static_cast<double>(f.ssim.f32()[0]), max_rel);
    SX_CHECK(out.ssim_map.size() == f.ssim_map.count());
    SX_ALL_CLOSE(out.ssim_map.data(), f.ssim_map.f32(), f.ssim_map.count(), f.meta, "ssim_map");
    const float s = static_cast<float>(out.ssim);
    SX_ALL_CLOSE(&s, f.ssim.f32(), 1, f.meta, "ssim");
}

} // namespace

// (a) The CPU reference fwd vs the golden — a sequential 2-pass separable conv (the same f32
// arithmetic as the kernel).
SX_TEST(ssim_cpu_fwd_golden) {
    const SsimFixture f = load_fixture();
    auto cpu = make_cpu_backend();
    const SsimFwdOutputs out = cpu->ssim_fwd(make_inputs(f, false));
    compare_fwd_golden("cpu", out, f);
    SX_CHECK(out.coeff_mu.empty()); // need_coeffs=false — the coefficient maps are not allocated
}

// (b) Metal fwd vs the golden, plus bit-exact determinism (memcmp across two runs — the map, the
// three coefficient maps, and the scalar).
// At 24×24 the map is a single 14×14 tile while the apron is 26×26 > the image — partial-tile
// boundary masking is already triggered by this fixture (08 §5 correction; if the masking
// regresses, OOB garbage makes the result diverge from the golden).
SX_TEST(ssim_metal_fwd_golden_and_determinism) {
    const SsimFixture f = load_fixture();
    auto& metal = sxtest::shared_metal_backend();
    const SsimInputs in = make_inputs(f, true);
    const SsimFwdOutputs a = metal.ssim_fwd(in);
    compare_fwd_golden("metal", a, f);
    SX_CHECK(a.coeff_mu.size() == f.ssim_map.count());
    SX_CHECK(a.coeff_sigma.size() == f.ssim_map.count());
    SX_CHECK(a.coeff_cross.size() == f.ssim_map.count());

    const SsimFwdOutputs b = metal.ssim_fwd(in);
    SX_CHECK(std::memcmp(a.ssim_map.data(), b.ssim_map.data(),
                         a.ssim_map.size() * sizeof(float)) == 0);
    SX_CHECK(std::memcmp(a.coeff_mu.data(), b.coeff_mu.data(),
                         a.coeff_mu.size() * sizeof(float)) == 0);
    SX_CHECK(std::memcmp(a.coeff_sigma.data(), b.coeff_sigma.data(),
                         a.coeff_sigma.size() * sizeof(float)) == 0);
    SX_CHECK(std::memcmp(a.coeff_cross.data(), b.coeff_cross.data(),
                         a.coeff_cross.size() * sizeof(float)) == 0);
    SX_CHECK(a.ssim == b.ssim); // f64 summation in index order — bit-identical input gives a bit-identical sum
}

// (c) Metal bwd vs the golden (v_img1), plus bit-exact determinism — the full chain exactly as the
// real training path runs it, Metal fwd (need_coeffs) → Metal bwd (which also verifies production
// of the three coefficient maps). v_loss=1 (the golden convention — upstream≡1; combining λ=0.2 is
// the Trainer's job / Task 3). The 24×24 input gives grid 2×2 — bwd partial tiles and the
// zero-padded scatter are triggered.
SX_TEST(ssim_metal_bwd_golden_and_determinism) {
    const SsimFixture f = load_fixture();
    auto& metal = sxtest::shared_metal_backend();
    const SsimInputs in = make_inputs(f, true);
    const SsimFwdOutputs fwd = metal.ssim_fwd(in);

    SsimGradsIn g;
    g.coeff_mu = fwd.coeff_mu.data();
    g.coeff_sigma = fwd.coeff_sigma.data();
    g.coeff_cross = fwd.coeff_cross.data();
    g.v_loss = 1.f;
    const std::vector<float> a = metal.ssim_bwd(in, g);
    SX_CHECK(a.size() == f.v_img1.count());
    SX_ALL_CLOSE(a.data(), f.v_img1.f32(), f.v_img1.count(), f.meta, "v_img1");

    const std::vector<float> b = metal.ssim_bwd(in, g);
    SX_CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

// (d) Partial-tile and multi-tile boundaries — a 40×40 synthetic case (map 30×30 → fwd grid 2×2:
// an interior full tile plus right and bottom partial tiles, with OOB masking of the right and
// bottom apron).
//  (1) **Embedding invariance (bit-exact)**: img[0:24,0:24] is identical to the fixture, and the
//      rest is a deterministic pattern. A valid map[p] depends only on the contents of the
//      in-bounds window [p,p+10]² (the rationale in the 08 §5 correction), so map[0:14,0:14] MUST
//      be **bit-identical** to the standalone 24×24 run — an empirical assertion that the
//      masked zero-loads of the apron in the 24×24 run never leak into the valid map.
//  (2) The full map is cross-checked against the CPU reference (the same f32 arithmetic — the gate
//      reuses the meta ssim_map/ssim tolerances as their single source of truth: the same kernel
//      and the same error class, honoring the ban on hardcoding tolerances in tests).
SX_TEST(ssim_partial_tile_boundary) {
    const SsimFixture f = load_fixture();
    auto& metal = sxtest::shared_metal_backend();
    const int64_t HE = 40, WE = 40, C = 3;
    const int64_t H0 = static_cast<int64_t>(f.meta.at("height"));
    const int64_t W0 = static_cast<int64_t>(f.meta.at("width"));

    // The synthetic input: the top-left 24×24 is the fixture bytes verbatim, the rest is a
    // deterministic pattern in [0,1).
    std::vector<float> img1e(static_cast<std::size_t>(HE * WE * C));
    std::vector<float> img2e(img1e.size());
    for (int64_t y = 0; y < HE; y++)
        for (int64_t x = 0; x < WE; x++)
            for (int64_t c = 0; c < C; c++) {
                const std::size_t off = static_cast<std::size_t>((y * WE + x) * C + c);
                if (y < H0 && x < W0) {
                    const std::size_t src = static_cast<std::size_t>((y * W0 + x) * C + c);
                    img1e[off] = f.img1.f32()[src];
                    img2e[off] = f.img2.f32()[src];
                } else {
                    img1e[off] = static_cast<float>((x * 37 + y * 17 + c * 11) % 101) / 101.f;
                    img2e[off] = static_cast<float>((x * 23 + y * 29 + c * 7) % 97) / 97.f;
                }
            }

    SsimInputs ine;
    ine.img1 = img1e.data();
    ine.img2 = img2e.data();
    ine.I = 1; ine.H = HE; ine.W = WE; ine.C = C;
    const SsimFwdOutputs me = metal.ssim_fwd(ine);

    // (1) Embedding invariance — compared bit-for-bit against the map of the standalone 24×24 run
    // (submap [0:14,0:14]).
    const SsimFwdOutputs m24 = metal.ssim_fwd(make_inputs(f, false));
    const int64_t mh0 = H0 - (kSsimWindowSize - 1), mw0 = W0 - (kSsimWindowSize - 1); // 14
    const int64_t mwe = WE - (kSsimWindowSize - 1);                                   // 30
    std::size_t sub_mismatch = 0;
    for (int64_t my = 0; my < mh0; my++)
        for (int64_t mx = 0; mx < mw0; mx++)
            for (int64_t c = 0; c < C; c++) {
                const float* pe = &me.ssim_map[static_cast<std::size_t>((my * mwe + mx) * C + c)];
                const float* p0 = &m24.ssim_map[static_cast<std::size_t>((my * mw0 + mx) * C + c)];
                if (std::memcmp(pe, p0, sizeof(float)) != 0) sub_mismatch++;
            }
    SX_CHECK(sub_mismatch == 0);

    // (2) The full map and the scalar — cross-checked against the CPU reference (reusing the meta
    // tolerances as their single source of truth).
    auto cpu = make_cpu_backend();
    const SsimFwdOutputs ce = cpu->ssim_fwd(ine);
    SX_CHECK(me.ssim_map.size() == ce.ssim_map.size());
    SX_ALL_CLOSE(me.ssim_map.data(), ce.ssim_map.data(), ce.ssim_map.size(), f.meta, "ssim_map");
    const float ms = static_cast<float>(me.ssim), cs = static_cast<float>(ce.ssim);
    SX_ALL_CLOSE(&ms, &cs, 1, f.meta, "ssim");
    std::printf("  INFO 40x40 map bitwise mismatch (metal vs cpu, for the record): %zu/%zu\n",
                sxtest::count_bitwise_mismatch(me.ssim_map.data(), ce.ssim_map.data(),
                                               ce.ssim_map.size()),
                ce.ssim_map.size());
}
