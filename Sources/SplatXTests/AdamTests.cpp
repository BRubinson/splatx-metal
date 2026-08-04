// fused_adam verification — fixtures/adam_basic (the task-3 fixture of record: N=1000, D=3,
// seed 21). Four cases: (1) CPU vs golden, (2) Metal vs golden, (3) a record of Metal vs CPU
// agreement (including the denormal boundary), (4) the valid mask variant (CPU/Metal vs
// *_out_masked + bit-identical masked slots).
// meta.json is the single source of truth for tolerances (never hardcode them in the test — the
// procedure of 07 open question #8).
#include "TestHarness.h"
#include "TestUtils.h" // shared_metal_backend / count_bitwise_mismatch (shared out in Task 1)
#include "splatx/CpuKernels.h"
#include "splatx/Json.h"
#include "splatx/KernelBackend.h"
#include "splatx/SXTensor.h"

#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace splatx;

namespace {

std::string fx(const char* rel) { return sxtest::fixture_dir() + "/adam_basic/" + rel; }

// adam is in-place — copy the fixture inputs so that input and expected stay separate.
struct AdamState {
    std::vector<float> param, exp_avg, exp_avg_sq;
};

AdamState load_state() {
    SXTensor p = SXTensor::load(fx("param.sxt"));
    SXTensor m = SXTensor::load(fx("exp_avg.sxt"));
    SXTensor v = SXTensor::load(fx("exp_avg_sq.sxt"));
    return {std::vector<float>(p.f32(), p.f32() + p.count()),
            std::vector<float>(m.f32(), m.f32() + m.count()),
            std::vector<float>(v.f32(), v.f32() + v.count())};
}

AdamInputs make_inputs(AdamState& s, const SXTensor& grad, const uint8_t* valid,
                       int64_t n_gauss, const std::map<std::string, double>& meta) {
    AdamInputs in;
    in.param = s.param.data();
    in.grad = grad.f32();
    in.exp_avg = s.exp_avg.data();
    in.exp_avg_sq = s.exp_avg_sq.data();
    in.valid = valid;
    in.n_elems = static_cast<int64_t>(s.param.size());
    in.n_gauss = n_gauss;
    in.lr = static_cast<float>(meta.at("lr"));
    in.b1 = static_cast<float>(meta.at("b1"));
    in.b2 = static_cast<float>(meta.at("b2"));
    in.eps = static_cast<float>(meta.at("eps"));
    return in;
}

} // namespace

// (1) CPU reference vs the adam_basic golden — the kAdamHasValidMask=false path (valid is ignored,
// every element is updated).
SX_TEST(fused_adam_cpu_matches_golden) {
    auto meta = json_parse_flat(fx("meta.json"));
    SXTensor grad = SXTensor::load(fx("grad.sxt"));
    SXTensor g_param = SXTensor::load(fx("param_out.sxt"));
    SXTensor g_m = SXTensor::load(fx("exp_avg_out.sxt"));
    SXTensor g_v = SXTensor::load(fx("exp_avg_sq_out.sxt"));

    AdamState s = load_state();
    AdamInputs in = make_inputs(s, grad, nullptr, g_param.shape[0], meta);
    fused_adam_cpu(in);

    SX_ALL_CLOSE(s.param.data(), g_param.f32(), s.param.size(), meta, "param_out");
    SX_ALL_CLOSE(s.exp_avg.data(), g_m.f32(), s.exp_avg.size(), meta, "exp_avg_out");
    SX_ALL_CLOSE(s.exp_avg_sq.data(), g_v.f32(), s.exp_avg_sq.size(), meta, "exp_avg_sq_out");
}

// (2) Metal vs the adam_basic golden — the first kernel driven all the way through MetalBackend
// (the kAdamHasValidMask=false pipeline).
SX_TEST(fused_adam_metal_matches_golden) {
    auto meta = json_parse_flat(fx("meta.json"));
    SXTensor grad = SXTensor::load(fx("grad.sxt"));
    SXTensor g_param = SXTensor::load(fx("param_out.sxt"));
    SXTensor g_m = SXTensor::load(fx("exp_avg_out.sxt"));
    SXTensor g_v = SXTensor::load(fx("exp_avg_sq_out.sxt"));

    AdamState s = load_state();
    AdamInputs in = make_inputs(s, grad, nullptr, g_param.shape[0], meta);
    sxtest::shared_metal_backend().fused_adam(in);

    SX_ALL_CLOSE(s.param.data(), g_param.f32(), s.param.size(), meta, "param_out");
    SX_ALL_CLOSE(s.exp_avg.data(), g_m.f32(), s.exp_avg.size(), meta, "exp_avg_out");
    SX_ALL_CLOSE(s.exp_avg_sq.data(), g_v.f32(), s.exp_avg_sq.size(), meta, "exp_avg_sq_out");
}

// (3) Metal vs CPU — records whether they agree exactly (denormal boundary rows [0..2]:
//     exp_avg_sq={0,1e-40,1e-30}, grad=0 — Apple GPU FTZ, measured on device; 06 open question 3).
//     The gate is "within the meta.json tolerances".
SX_TEST(fused_adam_metal_matches_cpu) {
    auto meta = json_parse_flat(fx("meta.json"));
    SXTensor grad = SXTensor::load(fx("grad.sxt"));
    const int64_t N = SXTensor::load(fx("valid.sxt")).shape[0];

    AdamState cpu = load_state();
    AdamInputs cin = make_inputs(cpu, grad, nullptr, N, meta);
    fused_adam_cpu(cin);

    AdamState mtl = load_state();
    AdamInputs min = make_inputs(mtl, grad, nullptr, N, meta);
    sxtest::shared_metal_backend().fused_adam(min);

    // Bitwise comparison — recorded to document whether they agree (Metal vs CPU; FTZ/denormal
    // measured on device).
    const size_t d_param =
        sxtest::count_bitwise_mismatch(mtl.param.data(), cpu.param.data(), mtl.param.size());
    const size_t d_m =
        sxtest::count_bitwise_mismatch(mtl.exp_avg.data(), cpu.exp_avg.data(), mtl.exp_avg.size());
    const size_t d_v = sxtest::count_bitwise_mismatch(mtl.exp_avg_sq.data(),
                                                      cpu.exp_avg_sq.data(), mtl.exp_avg_sq.size());
    std::printf("  INFO metal vs cpu bitwise mismatches: param %zu/%zu, exp_avg %zu/%zu, "
                "exp_avg_sq %zu/%zu\n",
                d_param, mtl.param.size(), d_m, mtl.exp_avg.size(), d_v, mtl.exp_avg_sq.size());
    for (size_t i = 0; i < 3; i++) // denormal boundary rows (prints column 0 only, as the sample)
        std::printf("  INFO denormal row %zu: exp_avg_sq cpu=%.9g metal=%.9g, param cpu=%.9g "
                    "metal=%.9g\n",
                    i, static_cast<double>(cpu.exp_avg_sq[i * 3]),
                    static_cast<double>(mtl.exp_avg_sq[i * 3]),
                    static_cast<double>(cpu.param[i * 3]), static_cast<double>(mtl.param[i * 3]));

    // Gate: within the meta.json tolerances against the CPU reference (an exact match passes
    // automatically).
    SX_ALL_CLOSE(mtl.param.data(), cpu.param.data(), cpu.param.size(), meta, "param_out");
    SX_ALL_CLOSE(mtl.exp_avg.data(), cpu.exp_avg.data(), cpu.exp_avg.size(), meta, "exp_avg_out");
    SX_ALL_CLOSE(mtl.exp_avg_sq.data(), cpu.exp_avg_sq.data(), cpu.exp_avg_sq.size(), meta,
                 "exp_avg_sq_out");
}

// (4) The valid mask variant — the kAdamHasValidMask=true pipeline. For a Gaussian with valid=0,
//     param and both moments stay exactly as they came in (bit-identical — not even decayed).
SX_TEST(fused_adam_valid_mask) {
    auto meta = json_parse_flat(fx("meta.json"));
    SXTensor grad = SXTensor::load(fx("grad.sxt"));
    SXTensor valid = SXTensor::load(fx("valid.sxt"));
    SXTensor g_param = SXTensor::load(fx("param_out_masked.sxt"));
    SXTensor g_m = SXTensor::load(fx("exp_avg_out_masked.sxt"));
    SXTensor g_v = SXTensor::load(fx("exp_avg_sq_out_masked.sxt"));
    const int64_t N = valid.shape[0];
    const int64_t D = g_param.shape[1];

    AdamState cpu = load_state();
    AdamInputs cin = make_inputs(cpu, grad, valid.u8(), N, meta);
    fused_adam_cpu(cin);
    SX_ALL_CLOSE(cpu.param.data(), g_param.f32(), cpu.param.size(), meta, "param_out_masked");
    SX_ALL_CLOSE(cpu.exp_avg.data(), g_m.f32(), cpu.exp_avg.size(), meta, "exp_avg_out_masked");
    SX_ALL_CLOSE(cpu.exp_avg_sq.data(), g_v.f32(), cpu.exp_avg_sq.size(), meta,
                 "exp_avg_sq_out_masked");

    AdamState mtl = load_state();
    AdamInputs min = make_inputs(mtl, grad, valid.u8(), N, meta);
    sxtest::shared_metal_backend().fused_adam(min);
    SX_ALL_CLOSE(mtl.param.data(), g_param.f32(), mtl.param.size(), meta, "param_out_masked");
    SX_ALL_CLOSE(mtl.exp_avg.data(), g_m.f32(), mtl.exp_avg.size(), meta, "exp_avg_out_masked");
    SX_ALL_CLOSE(mtl.exp_avg_sq.data(), g_v.f32(), mtl.exp_avg_sq.size(), meta,
                 "exp_avg_sq_out_masked");

    // A valid=0 slot must be bit-identical to the input (on CPU and Metal alike — skip semantics).
    AdamState input = load_state();
    size_t masked = 0, violations = 0;
    for (int64_t g = 0; g < N; g++) {
        if (valid.u8()[g] != 0) continue;
        masked++;
        for (int64_t j = 0; j < D; j++) {
            const size_t k = static_cast<size_t>(g * D + j);
            if (std::memcmp(&cpu.param[k], &input.param[k], 4) != 0 ||
                std::memcmp(&cpu.exp_avg[k], &input.exp_avg[k], 4) != 0 ||
                std::memcmp(&cpu.exp_avg_sq[k], &input.exp_avg_sq[k], 4) != 0 ||
                std::memcmp(&mtl.param[k], &input.param[k], 4) != 0 ||
                std::memcmp(&mtl.exp_avg[k], &input.exp_avg[k], 4) != 0 ||
                std::memcmp(&mtl.exp_avg_sq[k], &input.exp_avg_sq[k], 4) != 0)
                violations++;
        }
    }
    SX_CHECK(masked > 0);      // fixture contract: [3..5]=0 forced — masked slots exist
    SX_CHECK(violations == 0); // a skipped Gaussian touches none of the tensors
}

// Regression: the assert_all_close vacuous-pass guard — zero compared elements (everything masked
// out, or an empty tensor) and mask_period==0 must not pass silently; they have to take the FAIL
// path. The intentional failures the guard records are rolled back after judging whether it fired
// (the 3 FAIL lines printed below are intended).
SX_TEST(assert_all_close_vacuous_pass_guard) {
    const std::map<std::string, double> meta{{"tolerances.t.rtol", 1e-6},
                                             {"tolerances.t.atol", 1e-7}};
    const float a[2] = {1.f, 2.f}, e[2] = {1.f, 2.f};
    auto& reg = sxtest::Registry::shared();
    std::printf("  INFO the next 3 FAIL lines are intentional (guard verification, rolled back)\n");

    // (1) everything masked out → compared == 0 → expect 1 FAIL
    const unsigned char mask_all_zero[2] = {0, 0};
    int before = reg.failures;
    SX_ALL_CLOSE_MASKED(a, e, 2, meta, "t", mask_all_zero, 1);
    bool fired = (reg.failures == before + 1);
    reg.failures = before; // no-op if it never fired — the SX_CHECK below records the real failure
    SX_CHECK(fired);

    // (2) count == 0 (empty tensor) → expect 1 FAIL
    before = reg.failures;
    SX_ALL_CLOSE(a, e, static_cast<size_t>(0), meta, "t");
    fired = (reg.failures == before + 1);
    reg.failures = before;
    SX_CHECK(fired);

    // (3) mask != nullptr && mask_period == 0 → expect 1 FAIL before the division
    const unsigned char mask_one[1] = {1};
    before = reg.failures;
    SX_ALL_CLOSE_MASKED(a, e, 2, meta, "t", mask_one, 0);
    fired = (reg.failures == before + 1);
    reg.failures = before;
    SX_CHECK(fired);
}
