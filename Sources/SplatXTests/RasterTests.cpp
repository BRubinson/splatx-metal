// Verifies rasterize forward (the forward section of spec 05 plus the supplement and the 2nd
// correction, Phase 1c Task 3) — fixtures/raster_small/{nobg,bg} (I=1, 32×32, 2×2 tiles — the
// hand-check, the MAX_ALPHA clamp, exclusive early termination, the empty-tile background
// (07 Correction 4), and the permanently invalid g5), raster_std (I=2, 160×120 — **real
// verification of the per-image pointer advance**, with partial tile rows), and raster_dense (a
// single 16×16 tile, n_isects=640 — 3 iterations of the batch (CTA 256) loop, with
// max last_id=639 ≥ 512 = real blending in the 3rd batch).
//
// The comparison contract (task-2 report §3 / 07 §4): render_colors and render_alphas use the
// meta.json tolerances (never hardcode them in tests), and last_ids is **exact** (the boundary
// flake policy = 07 open question #5 — start with exact and report if a flake is measured. Because
// the golden is f64 while the kernel is f32, a pixel whose next_T approaches 1e-4 to within about
// 1e-7 relative could in theory flip — raster_small is immune by design, with a margin of ≥4×).
// fwd accumulates sequentially per pixel (no atomics) → bit-exact determinism across two Metal runs
// (memcmp) is the gate.
// The masks path has no fixture (07 Correction 6) — it is unverified in v1.
#include "TestHarness.h"
#include "TestUtils.h" // shared_metal_backend / SX_EXACT_I32 / count_bitwise_mismatch
#include "splatx/CpuKernels.h"
#include "splatx/Json.h"
#include "splatx/KernelBackend.h"
#include "splatx/SXTensor.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace splatx;

namespace {

const char* const kRasterFixtures[] = {"raster_small/nobg", "raster_small/bg", "raster_std",
                                       "raster_dense"};

std::string fxr(const std::string& dir, const char* rel) {
    return sxtest::fixture_dir() + "/" + dir + "/" + rel;
}

struct RasterFixture {
    std::map<std::string, double> meta;
    SXTensor means2d, conics, colors, opacities, flatten_ids, isect_offsets; // inputs
    SXTensor backgrounds;                                                    // has_background only
    SXTensor render_colors, render_alphas, last_ids;                         // fwd golden
    SXTensor v_render_colors, v_render_alphas;                               // injected bwd grads
    SXTensor v_means2d, v_conics, v_colors, v_opacities;                     // bwd golden (autograd)
    bool has_bg = false;
};

RasterFixture load_fixture(const std::string& dir) {
    RasterFixture f;
    f.meta = json_parse_flat(fxr(dir, "meta.json"));
    f.means2d = SXTensor::load(fxr(dir, "means2d.sxt"));
    f.conics = SXTensor::load(fxr(dir, "conics.sxt"));
    f.colors = SXTensor::load(fxr(dir, "colors.sxt"));
    f.opacities = SXTensor::load(fxr(dir, "opacities.sxt"));
    f.flatten_ids = SXTensor::load(fxr(dir, "flatten_ids.sxt"));
    f.isect_offsets = SXTensor::load(fxr(dir, "isect_offsets.sxt"));
    f.has_bg = f.meta.at("has_background") != 0.0;
    if (f.has_bg) f.backgrounds = SXTensor::load(fxr(dir, "backgrounds.sxt"));
    f.render_colors = SXTensor::load(fxr(dir, "render_colors.sxt"));
    f.render_alphas = SXTensor::load(fxr(dir, "render_alphas.sxt"));
    f.last_ids = SXTensor::load(fxr(dir, "last_ids.sxt"));
    f.v_render_colors = SXTensor::load(fxr(dir, "v_render_colors.sxt"));
    f.v_render_alphas = SXTensor::load(fxr(dir, "v_render_alphas.sxt"));
    f.v_means2d = SXTensor::load(fxr(dir, "v_means2d.sxt"));
    f.v_conics = SXTensor::load(fxr(dir, "v_conics.sxt"));
    f.v_colors = SXTensor::load(fxr(dir, "v_colors.sxt"));
    f.v_opacities = SXTensor::load(fxr(dir, "v_opacities.sxt"));
    return f;
}

RasterizeFwdInputs make_inputs(const RasterFixture& f) {
    RasterizeFwdInputs in;
    in.means2d = f.means2d.f32();
    in.conics = f.conics.f32();
    in.colors = f.colors.f32();
    in.opacities = f.opacities.f32();
    if (f.has_bg) in.backgrounds = f.backgrounds.f32();
    in.isect_offsets = f.isect_offsets.i32();
    in.flatten_ids = f.flatten_ids.i32();
    in.I = static_cast<int64_t>(f.meta.at("I"));
    in.N = static_cast<int64_t>(f.meta.at("N"));
    in.n_isects = static_cast<int64_t>(f.meta.at("n_isects"));
    in.image_width = static_cast<uint32_t>(f.meta.at("width"));
    in.image_height = static_cast<uint32_t>(f.meta.at("height"));
    in.tile_size = static_cast<uint32_t>(f.meta.at("tile_size"));
    in.tile_width = static_cast<uint32_t>(f.meta.at("tile_width"));
    in.tile_height = static_cast<uint32_t>(f.meta.at("tile_height"));
    return in;
}

// The additional bwd inputs — the fwd outputs are injected as the fixture's **golden**
// render_alphas and last_ids (this blocks error propagation from the Metal fwd — 07 / task-2 §4).
// The upstream gradients are the stored bytes verbatim.
RasterizeBwdGradsIn make_bwd_grads_in(const RasterFixture& f) {
    RasterizeBwdGradsIn g;
    g.render_alphas = f.render_alphas.f32();
    g.last_ids = f.last_ids.i32();
    g.v_render_colors = f.v_render_colors.f32();
    g.v_render_alphas = f.v_render_alphas.f32();
    return g;
}

// Compares bwd against the golden — all four tensors use the meta tolerances (rtol/atol plus
// rel_l2; never hardcode them in tests).
void compare_bwd_golden(const char* tag, const RasterizeBwdGrads& out, const RasterFixture& f) {
    std::printf("  [%s]\n", tag);
    SX_CHECK(out.v_means2d.size() == f.v_means2d.count());
    SX_CHECK(out.v_conics.size() == f.v_conics.count());
    SX_CHECK(out.v_colors.size() == f.v_colors.count());
    SX_CHECK(out.v_opacities.size() == f.v_opacities.count());
    SX_ALL_CLOSE(out.v_means2d.data(), f.v_means2d.f32(), f.v_means2d.count(), f.meta,
                 "v_means2d");
    SX_ALL_CLOSE(out.v_conics.data(), f.v_conics.f32(), f.v_conics.count(), f.meta, "v_conics");
    SX_ALL_CLOSE(out.v_colors.data(), f.v_colors.f32(), f.v_colors.count(), f.meta, "v_colors");
    SX_ALL_CLOSE(out.v_opacities.data(), f.v_opacities.f32(), f.v_opacities.count(), f.meta,
                 "v_opacities");
}

// Compares against the golden — both render outputs are toleranced (meta is their single source of
// truth) and last_ids is exact (07 §4).
void compare_golden(const char* tag, const RasterizeFwdOutputs& out, const RasterFixture& f) {
    std::printf("  [%s]\n", tag);
    SX_CHECK(out.render_colors.size() == f.render_colors.count());
    SX_CHECK(out.render_alphas.size() == f.render_alphas.count());
    SX_ALL_CLOSE(out.render_colors.data(), f.render_colors.f32(), f.render_colors.count(),
                 f.meta, "render_colors");
    SX_ALL_CLOSE(out.render_alphas.data(), f.render_alphas.f32(), f.render_alphas.count(),
                 f.meta, "render_alphas");
    SX_EXACT_I32("last_ids", out.last_ids, f.last_ids.i32(), f.last_ids.count());
}

} // namespace

// (1) The CPU reference vs all four goldens — with and without a background, the I=2 pointer
// advance, and the full 3-iteration batch loop path.
SX_TEST(raster_fwd_cpu_matches_golden) {
    for (const char* dir : kRasterFixtures) {
        const RasterFixture f = load_fixture(dir);
        compare_golden(dir, rasterize_fwd_cpu(make_inputs(f)), f);
    }
}

// (2) Metal vs all four goldens — drives both kRastHasBackground specialization variants end to
// end.
SX_TEST(raster_fwd_metal_matches_golden) {
    for (const char* dir : kRasterFixtures) {
        const RasterFixture f = load_fixture(dir);
        compare_golden(dir, sxtest::shared_metal_backend().rasterize_fwd(make_inputs(f)), f);
    }
}

// (3) Bit-exact Metal determinism — accumulation is sequential per pixel (no atomics), so exact
//     equality across two runs is the contract (spec 05 / 07). raster_std (80 tiles × 2 images) and
//     raster_dense (a single tile with 3 batch iterations) assert memcmp 0 even under varying
//     threadgroup scheduling. raster_small/bg covers determinism for the
//     kRastHasBackground=true specialized pipeline (a Minor carry-over from 1c Task 3).
SX_TEST(raster_fwd_metal_bitwise_deterministic) {
    for (const char* dir : {"raster_small/bg", "raster_std", "raster_dense"}) {
        const RasterFixture f = load_fixture(dir);
        const RasterizeFwdInputs in = make_inputs(f);
        const RasterizeFwdOutputs a = sxtest::shared_metal_backend().rasterize_fwd(in);
        const RasterizeFwdOutputs b = sxtest::shared_metal_backend().rasterize_fwd(in);
        SX_CHECK(a.render_colors.size() == b.render_colors.size());
        SX_CHECK(std::memcmp(a.render_colors.data(), b.render_colors.data(),
                             a.render_colors.size() * sizeof(float)) == 0);
        SX_CHECK(std::memcmp(a.render_alphas.data(), b.render_alphas.data(),
                             a.render_alphas.size() * sizeof(float)) == 0);
        SX_CHECK(std::memcmp(a.last_ids.data(), b.last_ids.data(),
                             a.last_ids.size() * sizeof(int32_t)) == 0);
    }
}

// (4) CPU ↔ Metal cross-check — two independent implementations of the same f32 semantics (the only
//     difference is the ulp of the exp implementation).
//     last_ids is exact and both render outputs use the meta tolerances; the bitwise mismatch count
//     is recorded (not a gate).
SX_TEST(raster_fwd_cpu_metal_cross) {
    for (const char* dir : kRasterFixtures) {
        const RasterFixture f = load_fixture(dir);
        const RasterizeFwdInputs in = make_inputs(f);
        const RasterizeFwdOutputs cpu = rasterize_fwd_cpu(in);
        const RasterizeFwdOutputs mtl = sxtest::shared_metal_backend().rasterize_fwd(in);
        std::printf("  [%s]\n", dir);
        SX_EXACT_I32("last_ids cpu-vs-metal", mtl.last_ids, cpu.last_ids.data(),
                     cpu.last_ids.size());
        SX_ALL_CLOSE(mtl.render_colors.data(), cpu.render_colors.data(),
                     cpu.render_colors.size(), f.meta, "render_colors");
        SX_ALL_CLOSE(mtl.render_alphas.data(), cpu.render_alphas.data(),
                     cpu.render_alphas.size(), f.meta, "render_alphas");
        std::printf("    bitwise mismatch (for the record): colors %zu/%zu, alphas %zu/%zu\n",
                    sxtest::count_bitwise_mismatch(mtl.render_colors.data(),
                                                   cpu.render_colors.data(),
                                                   cpu.render_colors.size()),
                    cpu.render_colors.size(),
                    sxtest::count_bitwise_mismatch(mtl.render_alphas.data(),
                                                   cpu.render_alphas.data(),
                                                   cpu.render_alphas.size()),
                    cpu.render_alphas.size());
    }
}

// (5) raster_small semantics — per-pixel comparison against the meta hand_check (the authoritative
//     hand-computed verification, task-2 §3.1) plus the background relation. Covered: the MAX_ALPHA
//     clamp (g0), exclusive early termination (g2 is not included → last=1), last not being updated
//     when an invalid is skipped (g5 → last=5), the empty-tile background (07 Correction 4 —
//     determines that the output write happens outside the batch loop, for every pixel), and the
//     invariance of alphas and last_ids between the bg and nobg variants.
SX_TEST(raster_fwd_small_semantics) {
    const RasterFixture fn = load_fixture("raster_small/nobg");
    const RasterFixture fb = load_fixture("raster_small/bg");
    auto& metal = sxtest::shared_metal_backend();
    const RasterizeFwdOutputs on = metal.rasterize_fwd(make_inputs(fn));
    const RasterizeFwdOutputs ob = metal.rasterize_fwd(make_inputs(fb));
    const uint32_t W = static_cast<uint32_t>(fn.meta.at("width"));

    // The 3 meta hand_check pixels — render_alphas (f32 proximity) and last_ids (exact)
    const struct {
        const char* key;
        uint32_t x, y;
    } px[] = {{"pixel_x8_y8", 8, 8}, {"pixel_x24_y8", 24, 8}, {"pixel_x24_y24_empty_tile", 24, 24}};
    for (const auto& p : px) {
        const size_t pix = static_cast<size_t>(p.y) * W + p.x;
        const std::string base = std::string("hand_check.") + p.key + ".";
        const double ra_want = fn.meta.at(base + "render_alphas");
        const int li_want = static_cast<int>(fn.meta.at(base + "last_ids"));
        for (const RasterizeFwdOutputs* o : {&on, &ob}) {
            SX_CHECK_NEAR(o->render_alphas[pix], ra_want, 1e-6, 1e-7); // golden f64 vs f32
            SX_CHECK(o->last_ids[pix] == li_want);
        }
    }

    // Pixel (24,24) in the empty tile (1,1): nobg gives exactly 0, and bg gives the background
    // verbatim (T=1 → fma(1,bg,0)=bg, bit-identical) — this determines that the kernel's output
    // write sits outside the batch loop (for every in-bounds pixel).
    const size_t empty_pix = 24 * W + 24;
    for (size_t k = 0; k < 3; k++) {
        SX_CHECK(on.render_colors[empty_pix * 3 + k] == 0.0f);
        SX_CHECK(ob.render_colors[empty_pix * 3 + k] == fb.backgrounds.f32()[k]);
    }

    // The invariants across the two variants (task-2 §3.1 — the inputs and seed are identical, and
    // bg is irrelevant to the alpha path): render_alphas is bit-identical, last_ids is exact, and
    // rc_bg == rc_nobg + (1-ra)·bg (compared in f64).
    SX_CHECK(std::memcmp(on.render_alphas.data(), ob.render_alphas.data(),
                         on.render_alphas.size() * sizeof(float)) == 0);
    SX_EXACT_I32("last_ids nobg-vs-bg", ob.last_ids, on.last_ids.data(), on.last_ids.size());
    const float* bg = fb.backgrounds.f32();
    for (size_t pix = 0; pix < on.render_alphas.size(); pix++) {
        const double T = 1.0 - static_cast<double>(on.render_alphas[pix]);
        for (size_t k = 0; k < 3; k++) {
            const double want = static_cast<double>(on.render_colors[pix * 3 + k]) +
                                T * static_cast<double>(bg[k]);
            SX_CHECK_NEAR(ob.render_colors[pix * 3 + k], want, 1e-6, 1e-7);
        }
    }
}

// ===== backward (the backward section of spec 05 plus supplements 1, 2, and 3, Phase 1c Task 4) =====
// There is no CPU bwd reference — comparison is directly against the autograd f64 golden (the 1a
// rule). The fwd outputs used as inputs are the injected golden render_alphas and last_ids (this
// blocks fwd error propagation). Because this is an atomic accumulation path it is not bit-exact
// deterministic — the contract is the meta tolerances plus two-run self-consistency rel_l2 (07 §4).

// (6) Metal bwd vs all four goldens — nobg/bg (the v_alpha background term), raster_std (I=2 — real
//     verification of the per-image pointer advance in bwd, mixed with bin_final truncation), and
//     raster_dense (a single tile with 640 isects — 8 simdgroups concentrated on the same Gaussian
//     = maximum atomic contention, plus 3 batch iterations traversed in reverse).
//     absgrad is off (the default) — the contract is that v_means2d_abs is an empty vector.
SX_TEST(raster_bwd_metal_matches_golden) {
    for (const char* dir : kRasterFixtures) {
        const RasterFixture f = load_fixture(dir);
        const RasterizeBwdGrads out =
            sxtest::shared_metal_backend().rasterize_bwd(make_inputs(f), make_bwd_grads_in(f));
        compare_bwd_golden(dir, out, f);
        SX_CHECK(out.v_means2d_abs.empty()); // the default path, with absgrad off
    }
}

// (7) Self-consistency of the atomic path — the two-run rel_l2 gate (task-2 §3:
//     ‖run1−run2‖₂/‖golden‖₂ <= 0.1·rel_l2). This measures the stability of a path that cannot be
//     bit-reproduced because the atomicAdd order differs from run to run (raster_dense has the
//     highest contention).
SX_TEST(raster_bwd_metal_self_consistency) {
    for (const char* dir : kRasterFixtures) {
        const RasterFixture f = load_fixture(dir);
        const RasterizeFwdInputs in = make_inputs(f);
        const RasterizeBwdGradsIn g = make_bwd_grads_in(f);
        auto& metal = sxtest::shared_metal_backend();
        const RasterizeBwdGrads a = metal.rasterize_bwd(in, g);
        const RasterizeBwdGrads b = metal.rasterize_bwd(in, g);
        const struct {
            const char* name;
            const std::vector<float>&r1, &r2;
            const SXTensor& golden;
        } rows[] = {
            {"v_means2d", a.v_means2d, b.v_means2d, f.v_means2d},
            {"v_conics", a.v_conics, b.v_conics, f.v_conics},
            {"v_colors", a.v_colors, b.v_colors, f.v_colors},
            {"v_opacities", a.v_opacities, b.v_opacities, f.v_opacities},
        };
        std::printf("  [%s]\n", dir);
        for (const auto& r : rows) {
            const double gate =
                0.1 * f.meta.at(std::string("tolerances.") + r.name + ".rel_l2");
            const double rel =
                sxtest::rel_l2(r.r1.data(), r.r2.data(), r.golden.count(), r.golden.f32());
            std::printf("    %-11s self-consistency rel_l2 %.3g (gate %.3g)\n", r.name, rel,
                        gate);
            SX_CHECK(rel <= gate);
        }
    }
}

// (8) absgrad (kRastHasAbsGrad) — verification is off in v1 (there is no autograd golden, 07 open
//     question #3): the on variant only confirms that the specialized pipeline compiles and
//     dispatches successfully, and the numerics are covered instead by the triangle-inequality
//     property (|Σv_xy| <= Σ|v_xy| — with slack for atomic rounding ulp) together with the fact
//     that all four main gradients still match the golden (absgrad is a separate accumulation only;
//     the main path is unchanged).
SX_TEST(raster_bwd_metal_absgrad_on_compiles) {
    const RasterFixture f = load_fixture("raster_small/nobg");
    RasterizeBwdGradsIn g = make_bwd_grads_in(f);
    g.absgrad = true;
    const RasterizeBwdGrads out =
        sxtest::shared_metal_backend().rasterize_bwd(make_inputs(f), g);
    compare_bwd_golden("raster_small/nobg absgrad=on", out, f);
    SX_CHECK(out.v_means2d_abs.size() == f.v_means2d.count());
    for (size_t i = 0; i < out.v_means2d_abs.size(); i++) {
        const double lhs = std::fabs(out.v_means2d[i]);
        const double rhs = out.v_means2d_abs[i];
        SX_CHECK(lhs <= rhs + 1e-6 + 1e-5 * rhs); // |sum| <= sum of absolutes (+ rounding slack)
        SX_CHECK(rhs >= 0.0);
    }
}

// (9) Built-in zero-initialization (D8-1) and the n_isects==0 skip — ① two consecutive raster_dense
//     calls both match the golden (if encode_zero_fill were missing, the second call would fail at
//     double the value or with garbage), ② a synthetic n_isects==0 input yields all zeros after the
//     dispatch is skipped (the launcher at Bwd.cu:365-369 — zeros_like semantics).
SX_TEST(raster_bwd_metal_zero_init_and_empty) {
    const RasterFixture f = load_fixture("raster_dense");
    const RasterizeFwdInputs in = make_inputs(f);
    const RasterizeBwdGradsIn g = make_bwd_grads_in(f);
    auto& metal = sxtest::shared_metal_backend();
    (void)metal.rasterize_bwd(in, g); // the first call (discarded)
    compare_bwd_golden("raster_dense 2nd consecutive call", metal.rasterize_bwd(in, g), f);

    // n_isects == 0 — an empty range: isect_offsets is all zeros and flatten_ids may be nullptr.
    const std::vector<float> zf(32 * 32 * 3, 0.5f); // dummy inputs (the kernel never touches them)
    const std::vector<int32_t> zi(4, 0);
    RasterizeFwdInputs ein;
    ein.means2d = zf.data();
    ein.conics = zf.data();
    ein.colors = zf.data();
    ein.opacities = zf.data();
    ein.isect_offsets = zi.data();
    ein.I = 1;
    ein.N = 2;
    ein.n_isects = 0;
    ein.image_width = 32;
    ein.image_height = 32;
    ein.tile_width = 2;
    ein.tile_height = 2;
    RasterizeBwdGradsIn eg;
    eg.render_alphas = zf.data();
    eg.last_ids = zi.data(); // undersized, but never touched (the dispatch is skipped)
    eg.v_render_colors = zf.data();
    eg.v_render_alphas = zf.data();
    const RasterizeBwdGrads eout = metal.rasterize_bwd(ein, eg);
    SX_CHECK(eout.v_means2d.size() == 4 && eout.v_conics.size() == 6 &&
             eout.v_colors.size() == 6 && eout.v_opacities.size() == 2);
    for (const std::vector<float>* v :
         {&eout.v_means2d, &eout.v_conics, &eout.v_colors, &eout.v_opacities})
        for (float x : *v) SX_CHECK(x == 0.0f);
}
