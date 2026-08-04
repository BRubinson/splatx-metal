#pragma once
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace sxtest {

struct Registry {
    static Registry& shared() { static Registry r; return r; }
    std::vector<std::pair<std::string, std::function<void()>>> tests;
    int failures = 0;
};

struct Register {
    Register(const char* name, std::function<void()> fn) {
        Registry::shared().tests.emplace_back(name, std::move(fn));
    }
};

// The list of names excluded by a build condition because they are host-only (macOS) tests
// (Phase 6 Task 3 — MetallibTests.cpp is excluded on iOS). Empty on a macOS build. This is the
// single source of truth for the runner report stating "N host-only tests excluded" — the excluded
// translation unit registers itself (this prevents name drift: registration and exclusion sit
// adjacent in the two branches of the same #if, in the same file).
inline std::vector<std::string>& host_only_excluded() {
    static std::vector<std::string> v;
    return v;
}

struct RegisterHostOnlyExcluded {
    explicit RegisterHostOnlyExcluded(const char* name) { host_only_excluded().push_back(name); }
};

// Per-test report callback — for the iOS app runner (the splatx_tests_c.h C surface adapts it).
// The CLI does not supply one.
using TestReportFn = std::function<void(const char* name, bool passed, int failed_checks)>;

// Re-runnable runner: failures are computed against a snapshot taken on entry (so that repeated
// runs in the iOS app do not accumulate and contaminate the count from a previous run). The
// single-run CLI behavior is the same as before.
inline int run_all(const TestReportFn& report = {}) {
    auto& r = Registry::shared();
    const int start_failures = r.failures;
    for (auto& [name, fn] : r.tests) {
        std::printf("TEST %s\n", name.c_str());
        const int before = r.failures;
        try {
            fn();
        } catch (const std::exception& e) {
            std::printf("  FAIL (exception): %s\n", e.what());
            r.failures++;
        } catch (...) {
            std::printf("  FAIL (exception): <unknown>\n");
            r.failures++;
        }
        if (report) report(name.c_str(), r.failures == before, r.failures - before);
    }
    // State the host-only exclusions explicitly (the iOS runner — on macOS the list is empty, so
    // nothing is printed and the gate string is unchanged)
    if (!host_only_excluded().empty()) {
        std::printf("EXCLUDED %zu host-only test(s):", host_only_excluded().size());
        for (const auto& n : host_only_excluded()) std::printf(" %s", n.c_str());
        std::printf("\n");
    }
    const int failures = r.failures - start_failures;
    if (failures) { std::printf("FAILED (%d checks)\n", failures); return 1; }
    std::printf("OK (%zu tests)\n", r.tests.size());
    return 0;
}

// ===== assertAllClose — the single source of truth for golden fixture comparison (Tasks 4–7) =====
// meta   = the result of splatx::json_parse_flat("<fixture>/meta.json").
// tensor = looked up in meta under the "tolerances.<tensor>.rtol"/".atol" keys (required — a
//          missing key is treated as a failure; tolerances MUST NOT be hardcoded in tests,
//          meta.json is their single source of truth — the 07 open question #8 procedure).
//          ".rel_l2" is optional — when present, ||a-e||₂/||e||₂ <= rel_l2 is checked in addition
//          (for bwd gradients).
// mask   = when not nullptr, elements with mask[i / mask_period] == 0 are excluded from the
//          comparison (the radii valid gate in proj, and similar — 07 Correction 3;
//          mask_period = element count / mask element count).
// NaN always counts as a violation. Zero compared elements (everything masked out, or an empty
// tensor) is a FAIL, not a vacuous pass — so that a regression such as everything being culled is
// never passed silently.
inline void assert_all_close(const char* file, int line, const float* actual,
                             const float* expected, std::size_t count,
                             const std::map<std::string, double>& meta,
                             const std::string& tensor,
                             const unsigned char* mask = nullptr, std::size_t mask_period = 1) {
    auto& reg = Registry::shared();
    if (mask && mask_period == 0) { // avoids integer-division UB (unused when mask == nullptr — check skipped)
        std::printf("  FAIL %s:%d: assert_all_close(%s) mask_period must be > 0\n", file, line,
                    tensor.c_str());
        reg.failures++;
        return;
    }
    const std::string base = "tolerances." + tensor + ".";
    const auto rit = meta.find(base + "rtol");
    const auto ait = meta.find(base + "atol");
    if (rit == meta.end() || ait == meta.end()) {
        std::printf("  FAIL %s:%d: meta.json missing tolerances for '%s'\n", file, line,
                    tensor.c_str());
        reg.failures++;
        return;
    }
    const double rtol = rit->second, atol = ait->second;
    std::size_t bad = 0, compared = 0, worst_i = 0;
    double worst_excess = -1.0, diff2 = 0.0, ref2 = 0.0;
    for (std::size_t i = 0; i < count; i++) {
        if (mask && mask[i / mask_period] == 0) continue;
        compared++;
        const double a = actual[i], e = expected[i];
        const double d = std::fabs(a - e);
        diff2 += d * d;
        ref2 += e * e;
        const double tol = atol + rtol * std::fabs(e);
        if (!(d <= tol)) { // includes NaN
            bad++;
            const double excess = d - tol;
            if (!(excess <= worst_excess)) { worst_excess = excess; worst_i = i; }
        }
    }
    if (compared == 0) { // no vacuous pass (the rel_l2 check would be 0/0 and meaningless — return now)
        std::printf("  FAIL %s:%d: assert_all_close(%s) compared 0 elements (all masked or empty)\n",
                    file, line, tensor.c_str());
        reg.failures++;
        return;
    }
    if (bad) {
        std::printf("  FAIL %s:%d: %s allclose violated %zu/%zu (rtol=%g atol=%g), "
                    "worst [%zu]: %.9g vs %.9g\n",
                    file, line, tensor.c_str(), bad, compared, rtol, atol, worst_i,
                    static_cast<double>(actual[worst_i]), static_cast<double>(expected[worst_i]));
        reg.failures++;
    }
    if (const auto lit = meta.find(base + "rel_l2"); lit != meta.end()) {
        const double rel = ref2 > 0.0 ? std::sqrt(diff2 / ref2) : std::sqrt(diff2);
        if (!(rel <= lit->second)) {
            std::printf("  FAIL %s:%d: %s rel_l2 %g > %g\n", file, line, tensor.c_str(), rel,
                        lit->second);
            reg.failures++;
        }
    }
}

} // namespace sxtest

#define SX_ALL_CLOSE(actual, expected, count, meta, tensor) \
    ::sxtest::assert_all_close(__FILE__, __LINE__, (actual), (expected), (count), (meta), (tensor))
#define SX_ALL_CLOSE_MASKED(actual, expected, count, meta, tensor, mask, mask_period)       \
    ::sxtest::assert_all_close(__FILE__, __LINE__, (actual), (expected), (count), (meta),   \
                               (tensor), (mask), (mask_period))

#define SX_TEST(name)                                        \
    static void sx_test_##name();                            \
    static ::sxtest::Register sx_reg_##name(#name, sx_test_##name); \
    static void sx_test_##name()

#define SX_CHECK(cond)                                                     \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
            ::sxtest::Registry::shared().failures++;                       \
        }                                                                  \
    } while (0)

#define SX_CHECK_NEAR(a, b, rtol, atol)                                          \
    do {                                                                         \
        double _a = (a), _b = (b);                                               \
        double _tol = (atol) + (rtol) * std::fabs(_b);                           \
        if (std::fabs(_a - _b) > _tol) {                                         \
            std::printf("  FAIL %s:%d: |%g - %g| > %g\n", __FILE__, __LINE__,    \
                        _a, _b, _tol);                                           \
            ::sxtest::Registry::shared().failures++;                             \
        }                                                                        \
    } while (0)
