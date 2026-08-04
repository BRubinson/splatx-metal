// Rng(Philox4x32-10) + mcmc_relocation / mcmc_perturb_positions verification (Phase 1c Task 5,
// spec 06).
//
// RNG (written from scratch — RngPhilox.h cites the literature):
// (1) the literature KAT anchors + unit tests of the conversion at the boundary values
//     (x∈{0,0xFFFFFFFF}) (06 Correction 5)
// (2) the philox integer stage bit-identical between CPU and Metal (contract) + a rel gate on the
//     normal conversion + determinism/stream separation
// relocation/perturb (derived from gsplat):
// (3)(4) the fixtures/relocation_basic golden (CPU/Metal — including the bit-identical binoms LUT)
// (5) the fixtures/perturb_basic golden (CPU+Metal) + two Metal runs bit-identical (the kernel's
//     determinism contract)
#include "TestHarness.h"
#include "TestUtils.h" // shared_metal_backend / count_bitwise_mismatch
#include "splatx/CpuKernels.h"
#include "splatx/Json.h"
#include "splatx/KernelBackend.h"
#include "splatx/SXTensor.h"
#include "RngPhilox.h" // targets the single-source core (KAT/boundaries — same source as the kernel)

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace splatx;

namespace {

std::string fx(const char* dir, const char* rel) {
    return sxtest::fixture_dir() + "/" + dir + "/" + rel;
}

} // namespace

// (1) The Philox4x32-10 literature KAT + uniform/Box-Muller boundary values (06 Correction 5).
// KAT source: the three "philox4x32 10" sets from Random123 tests/kat_vectors (D. E. Shaw
// Research) — quoted test data, not derived code. The only external anchor for this from-scratch
// implementation: 4 counter words + 2 key words → 4 expected output words. The all-0 / all-1
// counters are themselves the boundary counter cases.
SX_TEST(rng_philox_kat_and_boundary) {
    struct Kat {
        uint32_t c[4], k[2], e[4];
    };
    const Kat kats[] = {
        {{0u, 0u, 0u, 0u},
         {0u, 0u},
         {0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}},
        {{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu},
         {0xffffffffu, 0xffffffffu},
         {0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}},
        {{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u},
         {0xa4093822u, 0x299f31d0u},
         {0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}},
    };
    for (const Kat& kat : kats) {
        const SXPhilox4 r =
            sx_philox4x32_10(kat.c[0], kat.c[1], kat.c[2], kat.c[3], kat.k[0], kat.k[1]);
        for (int j = 0; j < 4; j++) SX_CHECK(r.v[j] == kat.e[j]);
    }

    // Boundaries of the uniform conversion — u ∈ (0,1] over the whole range (the basis for why
    // log(0) cannot happen, 06 Correction 5):
    // x=0 → 2^-32 (the minimum, > 0), x=0xFFFFFFFF → exactly 1.0 (float(x) rounds to 2^32).
    const uint32_t edges[] = {0u, 1u, 0x80000000u, 0xFFFFFFFEu, 0xFFFFFFFFu};
    for (uint32_t x : edges) {
        const float u = sx_u32_to_uniform(x);
        SX_CHECK(u > 0.0f && u <= 1.0f);
    }
    SX_CHECK(sx_u32_to_uniform(0u) == 1.0f / 4294967296.0f);
    SX_CHECK(sx_u32_to_uniform(0xFFFFFFFFu) == 1.0f);

    // The 4 boundary combinations of Box-Muller — no NaN/Inf. x0=0xFFFFFFFF gives u1=1 → r=0 →
    // z=±0 exactly (the log(1)=0 path); x0=0 gives u1=2^-32 → r=√(64·ln2)≈6.66 (the maximum
    // amplitude path).
    const uint32_t ext[] = {0u, 0xFFFFFFFFu};
    for (uint32_t a : ext)
        for (uint32_t b : ext) {
            const SXNormal2 z = sx_box_muller(a, b);
            SX_CHECK(std::isfinite(z.z0) && std::isfinite(z.z1));
        }
    SX_CHECK(sx_box_muller(0xFFFFFFFFu, 0u).z0 == 0.0f); // exact value on the r=0 path
    const float r_max = sx_box_muller(0u, 0u).z0;        // u1=2^-32, u2=2^-32 → cos≈1
    SX_CHECK(r_max > 6.6f && r_max < 6.7f);              // √(-2·ln 2^-32) = 6.6604

    // Drive the CPU reference end to end — an odd-count tail + every element finite (even when the
    // boundary values never show up in the stream, the conversion itself guarantees (0,1], so
    // finiteness is structural; this is a regression scan).
    RngNormalInputs in;
    in.seed = 0xDEADBEEFCAFEF00DULL;
    in.offset = 7;
    in.count = 1021; // not a multiple of 4 — the last thread writes partially
    const std::vector<float> z = rng_normal_f32_cpu(in);
    SX_CHECK(z.size() == 1021);
    std::size_t nonfinite = 0;
    for (float v : z)
        if (!std::isfinite(v)) nonfinite++;
    SX_CHECK(nonfinite == 0);
}

// (2) rng CPU↔Metal — the philox integer stage bit-identical (contract) / a rel gate on the normal
// conversion / determinism (rerunning the same (seed,offset) is bit-identical) / stream separation
// (offset+1 differs) / moments.
SX_TEST(rng_normal_metal_matches_cpu) {
    KernelBackend& mtl = sxtest::shared_metal_backend();
    RngNormalInputs in;
    in.seed = 0x853c49e6748fea9bULL;   // an arbitrary 64-bit value (drives the lo/hi split path)
    in.offset = 0x20260711FFFFFFFFULL; // offset_lo=0xFFFFFFFF — a word boundary
    in.count = (1 << 18) + 3;          // 262147 — not a multiple of 4 (tail thread writes partially)

    // The philox integer stage — bit-identical (the single source of truth RngPhilox.h is compiled
    // for both sides)
    const std::vector<uint32_t> cu = rng_philox_u32_cpu(in);
    const std::vector<uint32_t> mu = mtl.rng_philox_u32(in);
    SX_CHECK(cu.size() == mu.size());
    SX_CHECK(std::memcmp(cu.data(), mu.data(), cu.size() * sizeof(uint32_t)) == 0);

    // The normal conversion — zero NaN/Inf + a mixed tolerance (a self-comparison gate rather than
    // a golden fixture: 10^2 of headroom over the 1–2 ulp deviation between precise::log/sin/cos
    // and libm — atol 1e-6, rtol 1e-5)
    const std::vector<float> cn = rng_normal_f32_cpu(in);
    const std::vector<float> mn = mtl.rng_normal_f32(in);
    SX_CHECK(cn.size() == mn.size());
    std::size_t bad = 0, nonfinite = 0;
    double worst = 0.0;
    for (std::size_t i = 0; i < cn.size(); i++) {
        if (!std::isfinite(mn[i])) nonfinite++;
        const double d = std::fabs(static_cast<double>(mn[i]) - cn[i]);
        const double tol = 1e-6 + 1e-5 * std::fabs(static_cast<double>(cn[i]));
        if (!(d <= tol)) bad++;
        if (d > worst) worst = d;
    }
    std::printf("  INFO rng normal metal vs cpu: worst abs diff %.3g, bitwise mismatch %zu/%zu\n",
                worst, sxtest::count_bitwise_mismatch(mn.data(), cn.data(), cn.size()),
                cn.size());
    SX_CHECK(nonfinite == 0);
    SX_CHECK(bad == 0);

    // Determinism — rerunning the same (seed, offset) is bit-identical (the contract of being
    // counter-based and stateless)
    const std::vector<float> mn2 = mtl.rng_normal_f32(in);
    SX_CHECK(std::memcmp(mn.data(), mn2.data(), mn.size() * sizeof(float)) == 0);

    // Stream separation — offset+1 (which crosses the offset_hi carry boundary) is a different
    // sequence
    RngNormalInputs in2 = in;
    in2.offset += 1;
    const std::vector<float> other = mtl.rng_normal_f32(in2);
    SX_CHECK(sxtest::count_bitwise_mismatch(other.data(), mn.data(), mn.size()) > 0);

    // Moment control (a fixed seed, so the values are deterministic) — standard normal:
    // |mean| < 5σ/√n ≈ 0.0098, |var-1| < 5·√(2/n) ≈ 0.014
    double sum = 0.0, sum2 = 0.0;
    for (float v : mn) {
        sum += v;
        sum2 += static_cast<double>(v) * v;
    }
    const double mean = sum / static_cast<double>(mn.size());
    const double var = sum2 / static_cast<double>(mn.size()) - mean * mean;
    std::printf("  INFO rng moments: mean %.5f, var %.5f (n=%zu)\n", mean, var, mn.size());
    SX_CHECK(std::fabs(mean) < 0.0098);
    SX_CHECK(std::fabs(var - 1.0) < 0.014);
}

// (3) mcmc_relocation CPU vs the relocation_basic golden — includes the bit-identical binoms LUT
// (contract).
SX_TEST(mcmc_relocation_cpu_matches_golden) {
    auto meta = json_parse_flat(fx("relocation_basic", "meta.json"));
    SXTensor opacities = SXTensor::load(fx("relocation_basic", "opacities.sxt"));
    SXTensor scales = SXTensor::load(fx("relocation_basic", "scales.sxt"));
    SXTensor ratios = SXTensor::load(fx("relocation_basic", "ratios.sxt"));
    SXTensor binoms = SXTensor::load(fx("relocation_basic", "binoms.sxt"));
    SXTensor g_op = SXTensor::load(fx("relocation_basic", "new_opacities.sxt"));
    SXTensor g_sc = SXTensor::load(fx("relocation_basic", "new_scales.sxt"));
    const int n_max = static_cast<int>(meta.at("n_max"));

    // The host LUT (Pascal in u64 → float) is bit-identical to the golden LUT (math.comb → float32)
    // (06 §kernel 2)
    const std::vector<float> lut = make_binoms_lut(n_max);
    SX_CHECK(lut.size() == binoms.count());
    SX_CHECK(sxtest::count_bitwise_mismatch(lut.data(), binoms.f32(), lut.size()) == 0);

    RelocationInputs in;
    in.opacities = opacities.f32();
    in.scales = scales.f32();
    in.ratios = ratios.i32();
    in.binoms = lut.data(); // the production path — confirmed bit-identical to the golden above
    in.N = opacities.shape[0];
    in.n_max = n_max;
    in.min_opacity = static_cast<float>(meta.at("min_opacity"));
    const RelocationOutputs out = mcmc_relocation_cpu(in);

    SX_ALL_CLOSE(out.new_opacities.data(), g_op.f32(), out.new_opacities.size(), meta,
                 "new_opacities");
    SX_ALL_CLOSE(out.new_scales.data(), g_sc.f32(), out.new_scales.size(), meta, "new_scales");

    // fixture contract — asserts that the clamp paths fire (boundary indices: the generator's
    // docstring): [0]/[3] the floor (min_opacity), [1] the ceiling (1-FLT_EPSILON — o=1.0, ratio=1)
    SX_CHECK(out.new_opacities[0] == in.min_opacity);
    SX_CHECK(out.new_opacities[3] == in.min_opacity);
    SX_CHECK(out.new_opacities[1] == 1.0f - FLT_EPSILON);
}

// (4) mcmc_relocation Metal vs the golden — also records whether it is bit-identical to CPU
// (precise::pow vs powf).
SX_TEST(mcmc_relocation_metal_matches_golden) {
    auto meta = json_parse_flat(fx("relocation_basic", "meta.json"));
    SXTensor opacities = SXTensor::load(fx("relocation_basic", "opacities.sxt"));
    SXTensor scales = SXTensor::load(fx("relocation_basic", "scales.sxt"));
    SXTensor ratios = SXTensor::load(fx("relocation_basic", "ratios.sxt"));
    SXTensor g_op = SXTensor::load(fx("relocation_basic", "new_opacities.sxt"));
    SXTensor g_sc = SXTensor::load(fx("relocation_basic", "new_scales.sxt"));

    RelocationInputs in;
    const std::vector<float> lut = make_binoms_lut(static_cast<int>(meta.at("n_max")));
    in.opacities = opacities.f32();
    in.scales = scales.f32();
    in.ratios = ratios.i32();
    in.binoms = lut.data();
    in.N = opacities.shape[0];
    in.n_max = static_cast<int>(meta.at("n_max"));
    in.min_opacity = static_cast<float>(meta.at("min_opacity"));

    const RelocationOutputs mtl = sxtest::shared_metal_backend().mcmc_relocation(in);
    SX_ALL_CLOSE(mtl.new_opacities.data(), g_op.f32(), mtl.new_opacities.size(), meta,
                 "new_opacities");
    SX_ALL_CLOSE(mtl.new_scales.data(), g_sc.f32(), mtl.new_scales.size(), meta, "new_scales");

    const RelocationOutputs cpu = mcmc_relocation_cpu(in);
    std::printf("  INFO relocation metal vs cpu bitwise mismatches: new_opacities %zu/%zu, "
                "new_scales %zu/%zu\n",
                sxtest::count_bitwise_mismatch(mtl.new_opacities.data(),
                                               cpu.new_opacities.data(),
                                               cpu.new_opacities.size()),
                cpu.new_opacities.size(),
                sxtest::count_bitwise_mismatch(mtl.new_scales.data(), cpu.new_scales.data(),
                                               cpu.new_scales.size()),
                cpu.new_scales.size());
    // Metal takes the same clamp path (independent of the precise::pow deviation — the clamp is
    // pinned to a constant)
    SX_CHECK(mtl.new_opacities[0] == in.min_opacity);
    SX_CHECK(mtl.new_opacities[1] == 1.0f - FLT_EPSILON);
}

// (5) mcmc_perturb_positions CPU+Metal vs the perturb_basic golden + Metal determinism
// (bit-identical). noise is an input buffer, so the kernel is fully deterministic — a memcmp of two
// runs on the same input is the contract (spec 06).
SX_TEST(mcmc_perturb_matches_golden) {
    auto meta = json_parse_flat(fx("perturb_basic", "meta.json"));
    SXTensor positions = SXTensor::load(fx("perturb_basic", "positions.sxt"));
    SXTensor quats = SXTensor::load(fx("perturb_basic", "quats.sxt"));
    SXTensor scales_log = SXTensor::load(fx("perturb_basic", "scales_log.sxt"));
    SXTensor opacities_logit = SXTensor::load(fx("perturb_basic", "opacities_logit.sxt"));
    SXTensor noise = SXTensor::load(fx("perturb_basic", "noise.sxt"));
    SXTensor g_out = SXTensor::load(fx("perturb_basic", "positions_out.sxt"));

    PerturbInputs in;
    in.quats = quats.f32();
    in.scales_log = scales_log.f32();
    in.opacities_logit = opacities_logit.f32();
    in.noise = noise.f32();
    in.N = positions.shape[0];
    in.noise_scale = static_cast<float>(meta.at("noise_scale"));
    in.t = static_cast<float>(meta.at("t"));
    in.k = static_cast<float>(meta.at("k"));

    // CPU vs the golden (in-place — copy the input)
    std::vector<float> cpu_pos(positions.f32(), positions.f32() + positions.count());
    in.positions = cpu_pos.data();
    mcmc_perturb_positions_cpu(in);
    SX_ALL_CLOSE(cpu_pos.data(), g_out.f32(), cpu_pos.size(), meta, "positions_out");

    // Metal vs the golden
    std::vector<float> mtl_pos(positions.f32(), positions.f32() + positions.count());
    in.positions = mtl_pos.data();
    sxtest::shared_metal_backend().mcmc_perturb_positions(in);
    SX_ALL_CLOSE(mtl_pos.data(), g_out.f32(), mtl_pos.size(), meta, "positions_out");

    // Metal determinism — rerunning the same input is bit-identical (noise is an input buffer → a
    // pure map kernel)
    std::vector<float> mtl_pos2(positions.f32(), positions.f32() + positions.count());
    in.positions = mtl_pos2.data();
    sxtest::shared_metal_backend().mcmc_perturb_positions(in);
    SX_CHECK(std::memcmp(mtl_pos.data(), mtl_pos2.data(), mtl_pos.size() * sizeof(float)) == 0);

    std::printf("  INFO perturb metal vs cpu bitwise mismatches: %zu/%zu\n",
                sxtest::count_bitwise_mismatch(mtl_pos.data(), cpu_pos.data(), cpu_pos.size()),
                cpu_pos.size());

    // fixture contract — [1] has gate 0 (density≈1): positions unchanged (bit-identical to the
    // input, on CPU and Metal alike — with w=0 the addition is Σ·0, so +0.0 preserves the value)
    for (int j = 0; j < 3; j++) {
        SX_CHECK(std::memcmp(&cpu_pos[3 + j], &positions.f32()[3 + j], 4) == 0);
        SX_CHECK(std::memcmp(&mtl_pos[3 + j], &positions.f32()[3 + j], 4) == 0);
    }
}
