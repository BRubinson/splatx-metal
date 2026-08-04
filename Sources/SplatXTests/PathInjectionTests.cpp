// Verifies the TestPaths.h path resolution priority (Phase 6 Task 1 — iOS runtime injection).
// setter injection > environment variable (SPLATX_FIXTURE_DIR/SPLATX_KERNEL_SRC_DIR, an empty
// value counts as unset) > build-time macro. The entry state is restored via RAII so that fixture
// loading in other tests is unaffected (including the SX_CHECK failure and exception exit paths).
#include "TestHarness.h"
#include "TestPaths.h"

#include <cstdlib> // std::getenv / setenv / unsetenv (POSIX — common to macOS and iOS)
#include <string>

namespace {

using sxtest::fixture_dir;
using sxtest::kernel_src_dir;

// Captures both environment variables plus the injection globals on entry, and restores them on
// destruction.
struct RestorePathState {
    bool had_env_fx, had_env_ks;
    std::string env_fx, env_ks, inj_fx, inj_ks;
    RestorePathState() {
        const char* fx = std::getenv("SPLATX_FIXTURE_DIR");
        const char* ks = std::getenv("SPLATX_KERNEL_SRC_DIR");
        had_env_fx = fx != nullptr;
        had_env_ks = ks != nullptr;
        env_fx = fx ? fx : "";
        env_ks = ks ? ks : "";
        inj_fx = sxtest::TestPaths::injected().fixture_dir;
        inj_ks = sxtest::TestPaths::injected().kernel_src_dir;
    }
    ~RestorePathState() {
        if (had_env_fx) setenv("SPLATX_FIXTURE_DIR", env_fx.c_str(), 1);
        else unsetenv("SPLATX_FIXTURE_DIR");
        if (had_env_ks) setenv("SPLATX_KERNEL_SRC_DIR", env_ks.c_str(), 1);
        else unsetenv("SPLATX_KERNEL_SRC_DIR");
        sxtest::TestPaths::injected().fixture_dir = inj_fx;
        sxtest::TestPaths::injected().kernel_src_dir = inj_ks;
    }
};

SX_TEST(test_paths_override_priority) {
    RestorePathState restore;

    // (3) Macro fallback — nothing injected and no environment variable set (the current default
    // behavior on macOS).
    unsetenv("SPLATX_FIXTURE_DIR");
    unsetenv("SPLATX_KERNEL_SRC_DIR");
    sxtest::set_test_paths(nullptr, nullptr);
    SX_CHECK(fixture_dir() == FIXTURE_DIR);
    SX_CHECK(kernel_src_dir() == KERNEL_SRC_DIR);

    // (2) Environment variable override (following the precedent set by SPLATX_TEST_METALLIB).
    setenv("SPLATX_FIXTURE_DIR", "/env/fixtures", 1);
    setenv("SPLATX_KERNEL_SRC_DIR", "/env/kernel-src", 1);
    SX_CHECK(fixture_dir() == "/env/fixtures");
    SX_CHECK(kernel_src_dir() == "/env/kernel-src");

    // An empty environment variable counts as unset — it falls back to the macro (a guard against
    // an accidental export VAR=).
    setenv("SPLATX_FIXTURE_DIR", "", 1);
    SX_CHECK(fixture_dir() == FIXTURE_DIR);
    setenv("SPLATX_FIXTURE_DIR", "/env/fixtures", 1);

    // (1) setter injection takes priority over the environment variable (the iOS app path).
    sxtest::set_test_paths("/app/bundle/fixtures", "/app/bundle/kernel-src");
    SX_CHECK(fixture_dir() == "/app/bundle/fixtures");
    SX_CHECK(kernel_src_dir() == "/app/bundle/kernel-src");

    // Partial injection: nullptr clears the injection for that path — it falls back to the next
    // priority (the environment variable).
    sxtest::set_test_paths("/app/bundle/fixtures", nullptr);
    SX_CHECK(fixture_dir() == "/app/bundle/fixtures");
    SX_CHECK(kernel_src_dir() == "/env/kernel-src");
}

} // namespace
