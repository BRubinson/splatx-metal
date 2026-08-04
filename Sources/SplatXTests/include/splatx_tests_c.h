// splatx test runner C surface (Phase 6 Task 3 — shared single source of truth for the
// SplatXBench iOS app and the macOS CLI).
// The test registry (TestHarness.h) is C++, so Swift cannot import it directly; this minimal C
// wrapper exists for that reason (same pattern as splatx_train_c.h — the header is C only, so
// the Swift ClangImporter can parse it).
// The macOS CLI (splatx-core-tests main.cpp) calls the same functions — never duplicate the
// runner logic.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Report callback invoked as each test finishes (in registration order, called synchronously on
// the runner thread).
//   name:          test name (valid only during the callback — copy it to retain)
//   passed:        1 = passed, 0 = failed
//   failed_checks: number of checks that failed in this test (including exit via exception —
//                  at least 1)
typedef void (*sx_tests_report_cb)(const char* name, int32_t passed, int32_t failed_checks,
                                   void* user);

// Runs every registered test (TestHarness.h run_all — the detailed log goes to stdout via printf).
//   fixture_dir/kernel_src_dir: path injection (TestPaths.h set_test_paths — NULL clears the
//     injection and falls back to the environment variable, then the build-time macro. The iOS
//     app passes its bundle paths)
//   metallib_path: if not NULL and not the empty string, it is set as the SPLATX_TEST_METALLIB
//     environment variable — shared_metal_backend is then created with this explicit metallib
//     path (a load failure fails loudly, with no fallback — the TestUtils.h convention). NULL
//     keeps the existing environment variable / source compilation.
//   cb/user: per-test report callback (NULL allowed — the CLI uses stdout only)
//   out_total/out_failed: NULL allowed — number of tests run / number of tests failed
// Returns: 0 = all passed, 1 = there were failures (the CLI exit code convention — same as the
// previous run_all).
// Re-runnable — the failure counters are computed against a snapshot taken on each call (for the
// iOS app's repeated runs).
int32_t sx_tests_run_all(const char* fixture_dir, const char* kernel_src_dir,
                         const char* metallib_path, sx_tests_report_cb cb, void* user,
                         int32_t* out_total, int32_t* out_failed);

// The host-only (non-iOS) test list — the names of tests whose translation unit is disabled by a
// build condition (MetallibTests.cpp: it runs std::system and the build script, so it can neither
// compile nor run on iOS).
// On a macOS build this returns 0 (everything is registered and run). This is the basis for the
// runner report stating "N host-only tests excluded" (Phase 6 Task 1 carry-over).
int32_t sx_tests_host_only_excluded_count(void);
// Returns NULL if index is outside [0, count). The returned string has static lifetime (no copy
// needed).
const char* sx_tests_host_only_excluded_name(int32_t index);

#ifdef __cplusplus
}
#endif
