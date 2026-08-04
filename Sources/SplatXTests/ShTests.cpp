// Verifies sh_fwd/sh_bwd — fixtures/sh_k16 (B=2, N=64, K=16: deg0..3 fwd plus deg3 bwd) and
// fixtures/sh_k25 (B=1, N=32, K=25: deg4 fwd). The task-3 authoritative contract, spec 02 plus
// dispositions #5, #6, and #7.
// Five scenarios (brief Step 1): (1) CPU sh_fwd vs the golden (2) Metal fwd deg0..3 vs the golden
// (3) Metal fwd deg4 vs sh_k25 (4) Metal bwd vs the golden (v_coeffs/v_dirs/v_means, plus two-run
// self-consistency, plus the kShComputeVDirs=false variant) (5) bit-exact determinism on the
// contention-free B=1 path.
// meta.json is the single source of truth for tolerances (never hardcode them in tests — the 07
// open question #8 procedure).
// Masked-out fwd outputs are unwritten by contract (disposition #5 — do not zero-fill them): the
// mask slots of the vector read back from Metal hold undefined values, so the comparison runs
// under the mask (mask_period=D); the CPU can be compared in full as a by-product of its
// zero-initialized vector (the golden also stores 0 in the mask slots — the task-3 fixture
// convention).
#include "TestHarness.h"
#include "TestUtils.h" // shared_metal_backend / rel_l2 / count_bitwise_mismatch (shared in Task 1)
#include "splatx/CpuKernels.h"
#include "splatx/Json.h"
#include "splatx/KernelBackend.h"
#include "splatx/SXTensor.h"

#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace splatx;

namespace {

std::string fx16(const char* rel) { return sxtest::fixture_dir() + "/sh_k16/" + rel; }
std::string fx25(const char* rel) { return sxtest::fixture_dir() + "/sh_k25/" + rel; }

// The shared sh_k16 loader — all three inputs plus meta (B=2, N=64, K=16, D=3).
struct ShK16 {
    SXTensor dirs, coeffs, masks;
    std::map<std::string, double> meta;
    ShK16()
        : dirs(SXTensor::load(fx16("dirs.sxt"))),
          coeffs(SXTensor::load(fx16("coeffs.sxt"))),
          masks(SXTensor::load(fx16("masks.sxt"))),
          meta(json_parse_flat(fx16("meta.json"))) {}
    ShInputs inputs(uint32_t deg) const {
        ShInputs in;
        in.dirs = dirs.f32();
        in.coeffs = coeffs.f32();
        in.masks = masks.u8();
        in.degrees_to_use = deg;
        in.B = dirs.shape[0];
        in.N = dirs.shape[1];
        in.K = coeffs.shape[1];
        in.D = static_cast<uint32_t>(coeffs.shape[2]);
        return in;
    }
};

// The sh_k25 inputs (B=1, N=32, K=25, deg4, no masks — the kShHasMasks=false path).
ShInputs k25_inputs(const SXTensor& dirs, const SXTensor& coeffs) {
    ShInputs in;
    in.dirs = dirs.f32();
    in.coeffs = coeffs.f32();
    in.masks = nullptr;
    in.degrees_to_use = 4;
    in.B = dirs.shape[0];
    in.N = dirs.shape[1];
    in.K = coeffs.shape[1];
    in.D = static_cast<uint32_t>(coeffs.shape[2]);
    return in;
}

} // namespace

// (1) CPU sh_fwd vs the golden — deg 0..3 (sh_k16, the mask path) and deg 4 (sh_k25, no masks).
// The CPU uses a zero-initialized vector and skips masked elements, so mask slot 0 == golden 0:
// comparing everything also verifies the skip semantics. deg4 checks the 15 hardcoded 4th-order
// constants against typos (disposition #7 — deg4 is fwd only).
SX_TEST(sh_fwd_cpu_matches_golden) {
    ShK16 f;
    for (uint32_t deg = 0; deg <= 3; deg++) {
        const std::string name = "colors_deg" + std::to_string(deg);
        SXTensor golden = SXTensor::load(fx16((name + ".sxt").c_str()));
        ShOutputs o = sh_fwd_cpu(f.inputs(deg));
        SX_ALL_CLOSE(o.colors.data(), golden.f32(), o.colors.size(), f.meta, name);
    }
    auto meta25 = json_parse_flat(fx25("meta.json"));
    SXTensor dirs = SXTensor::load(fx25("dirs.sxt"));
    SXTensor coeffs = SXTensor::load(fx25("coeffs.sxt"));
    SXTensor golden = SXTensor::load(fx25("colors_deg4.sxt"));
    ShOutputs o = sh_fwd_cpu(k25_inputs(dirs, coeffs));
    SX_ALL_CLOSE(o.colors.data(), golden.f32(), o.colors.size(), meta25, "colors_deg4");
}

// (2) Metal sh_fwd deg0..3 vs the golden — the kShHasMasks=true pipeline. The comparison runs under
// the mask (unwritten slots are excluded — disposition #5). The Metal vs CPU bitwise mismatch count
// is recorded as INFO (valid slots only).
SX_TEST(sh_fwd_metal_matches_golden_k16) {
    ShK16 f;
    const size_t D = static_cast<size_t>(f.coeffs.shape[2]);
    for (uint32_t deg = 0; deg <= 3; deg++) {
        const std::string name = "colors_deg" + std::to_string(deg);
        SXTensor golden = SXTensor::load(fx16((name + ".sxt").c_str()));
        ShOutputs mtl = sxtest::shared_metal_backend().sh_fwd(f.inputs(deg));
        SX_ALL_CLOSE_MASKED(mtl.colors.data(), golden.f32(), mtl.colors.size(), f.meta, name,
                            f.masks.u8(), D);
        ShOutputs cpu = sh_fwd_cpu(f.inputs(deg));
        std::printf("  INFO deg%u metal vs cpu bitwise mismatches (valid slots only): %zu/%zu\n",
                    deg,
                    sxtest::count_bitwise_mismatch(mtl.colors.data(), cpu.colors.data(),
                                           mtl.colors.size(), f.masks.u8(), D),
                    mtl.colors.size());
    }
}

// (3) Metal sh_fwd deg4 vs the sh_k25 golden — the kShHasMasks=false pipeline (everything is
// written, the generic K=25 path). fwd is a pure map — this also confirms the two runs are
// bit-identical.
SX_TEST(sh_fwd_metal_matches_golden_k25_deg4) {
    auto meta25 = json_parse_flat(fx25("meta.json"));
    SXTensor dirs = SXTensor::load(fx25("dirs.sxt"));
    SXTensor coeffs = SXTensor::load(fx25("coeffs.sxt"));
    SXTensor golden = SXTensor::load(fx25("colors_deg4.sxt"));
    ShInputs in = k25_inputs(dirs, coeffs);
    ShOutputs o1 = sxtest::shared_metal_backend().sh_fwd(in);
    SX_ALL_CLOSE(o1.colors.data(), golden.f32(), o1.colors.size(), meta25, "colors_deg4");
    ShOutputs o2 = sxtest::shared_metal_backend().sh_fwd(in);
    SX_CHECK(std::memcmp(o1.colors.data(), o2.colors.data(),
                         o1.colors.size() * sizeof(float)) == 0);
}

// (4) Metal sh_bwd (deg3, B=2) vs the autograd golden — v_coeffs (atomic accumulation across the
// batch), v_dirs, and v_means (the reduction summing over the B axis). The v_colors input is
// injected from the fixture golden (the stored mask-multiplied values) — chaining the Metal fwd
// output is forbidden (D7 blocks error propagation). Includes the two-run self-consistency gate
// (07) and the kShComputeVDirs=false variant (which computes only v_coeffs).
SX_TEST(sh_bwd_metal_matches_golden) {
    ShK16 f;
    SXTensor v_colors = SXTensor::load(fx16("v_colors.sxt"));
    SXTensor g_coeffs = SXTensor::load(fx16("v_coeffs.sxt"));
    SXTensor g_dirs = SXTensor::load(fx16("v_dirs.sxt"));
    SXTensor g_means = SXTensor::load(fx16("v_means.sxt"));
    ShInputs in = f.inputs(static_cast<uint32_t>(f.meta.at("degrees_to_use_bwd")));
    in.compute_v_dirs = true;

    ShGrads r1 = sxtest::shared_metal_backend().sh_bwd(in, v_colors.f32());
    SX_ALL_CLOSE(r1.v_coeffs.data(), g_coeffs.f32(), r1.v_coeffs.size(), f.meta, "v_coeffs");
    SX_ALL_CLOSE(r1.v_dirs.data(), g_dirs.f32(), r1.v_dirs.size(), f.meta, "v_dirs");

    // The v_dirs slots of masked-out elements are exactly 0 (the kernel skips them, so the
    // zero-initialized buffer is preserved, D8-1).
    const size_t BN = static_cast<size_t>(in.B * in.N);
    size_t masked = 0, nonzero = 0;
    for (size_t e = 0; e < BN; e++) {
        if (f.masks.u8()[e] != 0) continue;
        masked++;
        for (int j = 0; j < 3; j++)
            if (r1.v_dirs[e * 3 + j] != 0.0f) nonzero++;
    }
    SX_CHECK(masked > 0);  // the fixture contract: the masks are mixed (88/128 valid)
    SX_CHECK(nonzero == 0);

    // v_means = v_dirs.sum(over the B axis) — the gradient reduction for the dirs = means - campos
    // chain (spec 02).
    const size_t N3 = static_cast<size_t>(in.N) * 3;
    std::vector<float> v_means(N3);
    for (size_t i = 0; i < N3; i++) v_means[i] = r1.v_dirs[i] + r1.v_dirs[N3 + i];
    SX_ALL_CLOSE(v_means.data(), g_means.f32(), v_means.size(), f.meta, "v_means");

    // Two-run self-consistency: ‖r1−r2‖₂/‖golden‖₂ ≤ 0.1·rel_l2 (07 — the atomic contention gate).
    ShGrads r2 = sxtest::shared_metal_backend().sh_bwd(in, v_colors.f32());
    const double gate_c = 0.1 * f.meta.at("tolerances.v_coeffs.rel_l2");
    const double gate_d = 0.1 * f.meta.at("tolerances.v_dirs.rel_l2");
    const double sc_c =
        sxtest::rel_l2(r1.v_coeffs.data(), r2.v_coeffs.data(), r1.v_coeffs.size(), g_coeffs.f32());
    const double sc_d =
        sxtest::rel_l2(r1.v_dirs.data(), r2.v_dirs.data(), r1.v_dirs.size(), g_dirs.f32());
    std::printf("  INFO self-consistency rel_l2: v_coeffs %.3g (gate %.3g), v_dirs %.3g (gate %.3g)\n",
                sc_c, gate_c, sc_d, gate_d);
    SX_CHECK(sc_c <= gate_c);
    SX_CHECK(sc_d <= gate_d);

    // The kShComputeVDirs=false variant — a pipeline in which the v_dirs argument and output do not
    // exist at all (D8-5).
    in.compute_v_dirs = false;
    ShGrads r3 = sxtest::shared_metal_backend().sh_bwd(in, v_colors.f32());
    SX_ALL_CLOSE(r3.v_coeffs.data(), g_coeffs.f32(), r3.v_coeffs.size(), f.meta, "v_coeffs");
    SX_CHECK(r3.v_dirs.empty());
}

// (5) Bit-exact determinism on the contention-free B=1 path — the batch 0 slice of sh_k16 (with
// B=1 there is exactly one writer per v_coeffs slot, so the result is deterministic even with
// atomics, spec 02). v_dirs still has 3-thread atomic contention across the channels of the same
// element, so its order is not guaranteed — the gate is self-consistency within the meta
// tolerance, and bit equality is recorded as INFO (the same convention as Task 6 c2). With
// kShComputeVDirs=false there is no contention at all — every output is bit-exact deterministic.
SX_TEST(sh_bwd_metal_b1_bit_deterministic) {
    ShK16 f;
    SXTensor v_colors = SXTensor::load(fx16("v_colors.sxt"));
    ShInputs in = f.inputs(3);
    in.B = 1; // the batch 0 slice — the first N elements of dirs, masks, and v_colors are batch 0 as is

    in.compute_v_dirs = true;
    ShGrads r1 = sxtest::shared_metal_backend().sh_bwd(in, v_colors.f32());
    ShGrads r2 = sxtest::shared_metal_backend().sh_bwd(in, v_colors.f32());
    ShGrads r3 = sxtest::shared_metal_backend().sh_bwd(in, v_colors.f32());
    SX_CHECK(std::memcmp(r1.v_coeffs.data(), r2.v_coeffs.data(),
                         r1.v_coeffs.size() * sizeof(float)) == 0);
    SX_CHECK(std::memcmp(r1.v_coeffs.data(), r3.v_coeffs.data(),
                         r1.v_coeffs.size() * sizeof(float)) == 0);
    std::printf("  INFO v_dirs bitwise mismatches across runs (channel atomic, not gated): "
                "run2 %zu/%zu, run3 %zu/%zu\n",
                sxtest::count_bitwise_mismatch(r1.v_dirs.data(), r2.v_dirs.data(), r1.v_dirs.size()),
                r1.v_dirs.size(),
                sxtest::count_bitwise_mismatch(r1.v_dirs.data(), r3.v_dirs.data(), r1.v_dirs.size()),
                r1.v_dirs.size());
    SX_ALL_CLOSE(r2.v_dirs.data(), r1.v_dirs.data(), r1.v_dirs.size(), f.meta, "v_dirs");
    SX_ALL_CLOSE(r3.v_dirs.data(), r1.v_dirs.data(), r1.v_dirs.size(), f.meta, "v_dirs");

    in.compute_v_dirs = false;
    ShGrads n1 = sxtest::shared_metal_backend().sh_bwd(in, v_colors.f32());
    ShGrads n2 = sxtest::shared_metal_backend().sh_bwd(in, v_colors.f32());
    SX_CHECK(std::memcmp(n1.v_coeffs.data(), n2.v_coeffs.data(),
                         n1.v_coeffs.size() * sizeof(float)) == 0);
}
