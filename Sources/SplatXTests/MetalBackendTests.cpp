#include <exception>
#include <string>

#include "TestHarness.h"
#include "TestPaths.h" // kernel_src_dir() — single source of truth for paths (runtime injection)
#include "splatx/MetalBackend.h"

SX_TEST(metal_backend_create_and_name) {
    auto backend = splatx::make_metal_backend(nullptr, sxtest::kernel_src_dir());
    SX_CHECK(backend != nullptr);
    // The backward-compatible signature is source-compile only — name() reports which layer was
    // selected (Phase 2 Task 2).
    SX_CHECK(std::string(backend->name()) == "metal(source)");
}

SX_TEST(metal_backend_missing_dir_throws) {
    bool threw = false;
    try {
        splatx::make_metal_backend(nullptr, "/nonexistent/splatx-kernel-src");
    } catch (const std::exception&) {
        threw = true;
    }
    SX_CHECK(threw);
}
