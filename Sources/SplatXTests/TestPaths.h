#pragma once
// The single source of truth for test path resolution (Phase 6 Task 1 — iOS runtime injection).
//
// The FIXTURE_DIR/KERNEL_SRC_DIR build-time macros (Package.swift cxxSettings .define, based on
// Context.packageDirectory) are a macOS development convenience — inside an iOS app bundle those
// absolute paths do not exist on the device, so a runtime override is required. The priority
// order (following the precedent set by SPLATX_TEST_METALLIB — the same pattern as
// shared_metal_backend in TestUtils.h):
//   1) set_test_paths() injection — embedders such as the iOS app (an in-process function call)
//   2) the SPLATX_FIXTURE_DIR / SPLATX_KERNEL_SRC_DIR environment variables — CLI/CI (an empty
//      value counts as unset)
//   3) the build-time macro       — the default (macOS development, the current behavior as is)
//
// The principle is that test logic is left unmodified — this header handles path resolution only.
// Consumers MUST NOT use the macros directly; use only fixture_dir()/kernel_src_dir().
#include <cstdlib> // std::getenv
#include <filesystem> // tmp_path — temp_directory_path (a temp path compatible with the iOS sandbox)
#include <string>

#ifndef FIXTURE_DIR
#error "FIXTURE_DIR must be defined (Package.swift cxxSettings)"
#endif
#ifndef KERNEL_SRC_DIR
#error "KERNEL_SRC_DIR must be defined (Package.swift cxxSettings)"
#endif

namespace sxtest {

// The runtime injection state — exactly one global (the permitted limit). An empty string means
// nothing was injected (fall back to the next priority).
// A function-local static inside an inline function — C++17 inline semantics guarantee a single
// instance (the same technique as shared_metal_backend).
struct TestPaths {
    std::string fixture_dir;
    std::string kernel_src_dir;
    static TestPaths& injected() {
        static TestPaths p;
        return p;
    }
};

// The setter for embedders — call it once before running tests on a host where setting environment
// variables is awkward (the iOS app).
// nullptr clears the injection for that path (explicit initialization — the previous value is not
// retained).
inline void set_test_paths(const char* fixture_dir, const char* kernel_src_dir) {
    TestPaths::injected().fixture_dir = fixture_dir ? fixture_dir : "";
    TestPaths::injected().kernel_src_dir = kernel_src_dir ? kernel_src_dir : "";
}

// The shared resolution: injection > environment variable (only when non-empty) > macro.
inline std::string resolve_test_path(const std::string& injected, const char* env_name,
                                     const char* macro_value) {
    if (!injected.empty()) return injected;
    if (const char* env = std::getenv(env_name); env && *env) return env;
    return macro_value;
}

// The fixtures root — the only place the FIXTURE_DIR macro is consumed.
inline std::string fixture_dir() {
    return resolve_test_path(TestPaths::injected().fixture_dir, "SPLATX_FIXTURE_DIR",
                             FIXTURE_DIR);
}

// The kernel source root — the only place the KERNEL_SRC_DIR macro is consumed (within the test
// target).
inline std::string kernel_src_dir() {
    return resolve_test_path(TestPaths::injected().kernel_src_dir, "SPLATX_KERNEL_SRC_DIR",
                             KERNEL_SRC_DIR);
}

// Temporary file paths for tests (Phase 6 Task 3) — replaces a hardcoded "/tmp/<name>". The iOS
// sandbox cannot write to "/tmp", so this resolves through temp_directory_path() (macOS: TMPDIR,
// iOS: the container tmp — both are writable). The test logic is unchanged; only path resolution
// was lifted into this header.
inline std::string tmp_path(const std::string& name) {
    return (std::filesystem::temp_directory_path() / name).string();
}

} // namespace sxtest
