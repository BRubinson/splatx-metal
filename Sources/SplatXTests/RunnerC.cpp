// splatx_tests_c.h implementation — the test runner C surface (Phase 6 Task 3).
// The macOS CLI (splatx-core-tests main.cpp) and the SplatXBench iOS app share these functions
// (the runner logic is defined once — TestHarness.h run_all).
#include "splatx_tests_c.h"

#include "TestHarness.h"
#include "TestPaths.h"

#include <cstdint>
#include <cstdlib> // setenv (POSIX — common to macOS and iOS)

extern "C" int32_t sx_tests_run_all(const char* fixture_dir, const char* kernel_src_dir,
                                    const char* metallib_path, sx_tests_report_cb cb, void* user,
                                    int32_t* out_total, int32_t* out_failed) {
    // Path injection (TestPaths.h priority 1) — NULL clears the injection, i.e. falls back to the
    // environment variable and then the macro (when the CLI passes NULL, the previous behavior is
    // preserved exactly).
    sxtest::set_test_paths(fixture_dir, kernel_src_dir);
    // Explicit metallib — this uses the same channel (an environment variable) as the TestUtils.h
    // shared_metal_backend switch (the backend is a function-local static, so it must be set
    // before the first test runs).
    if (metallib_path && *metallib_path) setenv("SPLATX_TEST_METALLIB", metallib_path, 1);

    int32_t failed_tests = 0;
    const int rc = sxtest::run_all([&](const char* name, bool passed, int failed_checks) {
        if (!passed) failed_tests++;
        if (cb) cb(name, passed ? 1 : 0, static_cast<int32_t>(failed_checks), user);
    });
    if (out_total) *out_total = static_cast<int32_t>(sxtest::Registry::shared().tests.size());
    if (out_failed) *out_failed = failed_tests;
    return static_cast<int32_t>(rc);
}

extern "C" int32_t sx_tests_run_prefix(const char* prefix, sx_tests_report_cb cb, void* user,
                                       int32_t* out_total, int32_t* out_failed) {
    sxtest::set_test_paths(nullptr, nullptr);
    int32_t failed_tests = 0, total = 0;
    const int rc = sxtest::run_all(
        [&](const char* name, bool passed, int failed_checks) {
            total++;
            if (!passed) failed_tests++;
            if (cb) cb(name, passed ? 1 : 0, static_cast<int32_t>(failed_checks), user);
        },
        prefix);
    if (out_total) *out_total = total;
    if (out_failed) *out_failed = failed_tests;
    return static_cast<int32_t>(rc);
}

extern "C" int32_t sx_tests_host_only_excluded_count(void) {
    return static_cast<int32_t>(sxtest::host_only_excluded().size());
}

extern "C" const char* sx_tests_host_only_excluded_name(int32_t index) {
    const auto& v = sxtest::host_only_excluded();
    if (index < 0 || static_cast<size_t>(index) >= v.size()) return nullptr;
    return v[static_cast<size_t>(index)].c_str();
}
