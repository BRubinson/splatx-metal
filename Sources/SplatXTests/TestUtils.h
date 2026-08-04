#pragma once
// The single source of truth for shared test helpers (Phase 1b Task 1 — delegated from the
// Phase 1a final review).
// metal_backend(), rel_l2, and count_bitwise_mismatch had been triplicated across AdamTests,
// ProjectionMetalTests, and ShTests; they were lifted here. Tasks 3–7 reuse this header (no
// file-local redefinitions). Tolerance comparison (assert_all_close) is owned by TestHarness.h.
#include "TestHarness.h"
#include "TestPaths.h" // fixture_dir()/kernel_src_dir() — the single source of truth for path resolution (runtime injection)
#include "splatx/KernelBackend.h"
#include "splatx/MetalBackend.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib> // std::getenv — the SPLATX_TEST_METALLIB override
#include <cstring>
#include <memory>
#include <vector>

namespace sxtest {

// A shared MetalBackend — every test translation unit uses the same instance (library compilation
// and the pipeline cache happen once for the whole process). A function-local static inside an
// inline function is guaranteed to be a single instance by C++17 inline semantics. A creation
// failure surfaces as an exception in the first test that uses it.
//
// The SPLATX_TEST_METALLIB environment variable: when set, the backend is created from the
// offline metallib at that path — the switch that drives the entire suite (22 kernel entries plus
// every function constant specialization) through the metallib path end to end. Because the
// explicit-path convention applies (a failure throws immediately, with no fallback), a wrong path
// fails loudly. The default (unset) is source compilation — the default CI path (determinism).
// How to run it:
//   tools/build_metallib.sh && SPLATX_TEST_METALLIB=build/splatx.metallib swift run splatx-core-tests
inline splatx::KernelBackend& shared_metal_backend() {
    static std::unique_ptr<splatx::KernelBackend> b = [] {
        if (const char* lib = std::getenv("SPLATX_TEST_METALLIB")) {
            std::printf("shared_metal_backend: SPLATX_TEST_METALLIB=%s (metallib path verification)\n",
                        lib);
            splatx::MetalBackendDesc desc;
            desc.metallib_path = lib;
            desc.search_exe_dir = false; // explicit path only — on a typo it throws, with no fallback
            return splatx::make_metal_backend(nullptr, desc);
        }
        return splatx::make_metal_backend(nullptr, kernel_src_dir());
    }();
    return *b;
}

// Two-run self-consistency: ‖a−b‖₂/‖ref‖₂ (the 07 gate — ref is the golden fixture).
// When ‖ref‖₂ == 0 it returns the absolute norm ‖a−b‖₂.
inline double rel_l2(const float* a, const float* b, std::size_t n, const float* ref) {
    double d2 = 0.0, r2 = 0.0;
    for (std::size_t i = 0; i < n; i++) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        d2 += d * d;
        r2 += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    return r2 > 0.0 ? std::sqrt(d2 / r2) : std::sqrt(d2);
}

// The bitwise mismatch count — for the record only (not a gate; it measures the fast-math ulp/FTZ
// deviation on device).
// When mask != nullptr, slots with mask[i / period] == 0 are excluded from the comparison (the
// unwritten contract — D3 / disposition #5).
inline std::size_t count_bitwise_mismatch(const float* a, const float* b, std::size_t n,
                                          const unsigned char* mask = nullptr,
                                          std::size_t period = 1) {
    std::size_t bad = 0;
    for (std::size_t i = 0; i < n; i++) {
        if (mask && mask[i / period] == 0) continue;
        if (std::memcmp(a + i, b + i, sizeof(float)) != 0) bad++;
    }
    return bad;
}

// ===== Exact integer assertions — the single source of truth for 1b integer output checks =====
// For radii (int32), isect_ids (u64 keys), flatten_ids/tile_offsets (int32), and the like, exact
// equality with no tolerance is the contract (D9-1 / 07 §4). count == 0 is a FAIL, not a vacuous
// pass (the same policy as assert_all_close — an empty comparison is never passed silently).
// actual_count is the real element count of the actual buffer — if it differs from count (the
// golden size), this FAILs immediately, before comparing any element (this prevents an OOB read at
// the golden size when a regression has made actual shorter — 1b final review carry-over). Most
// call sites hold a std::vector, so they use the vector overload below (the size is passed
// automatically).
inline void assert_exact_i32(const char* file, int line, const char* tag,
                             const std::int32_t* actual, std::size_t actual_count,
                             const std::int32_t* expected, std::size_t count) {
    auto& reg = Registry::shared();
    if (count == 0) {
        std::printf("  FAIL %s:%d: assert_exact_i32(%s) compared 0 elements\n", file, line, tag);
        reg.failures++;
        return;
    }
    if (actual_count != count) {
        std::printf("  FAIL %s:%d: %s actual size %zu != expected count %zu (size guard)\n",
                    file, line, tag, actual_count, count);
        reg.failures++;
        return;
    }
    std::size_t bad = 0, first = 0;
    for (std::size_t i = 0; i < count; i++)
        if (actual[i] != expected[i]) {
            if (bad == 0) first = i;
            bad++;
        }
    if (bad) {
        std::printf("  FAIL %s:%d: %s exact mismatch %zu/%zu (first [%zu]: %d vs %d)\n", file,
                    line, tag, bad, count, first, actual[first], expected[first]);
        reg.failures++;
    }
}

inline void assert_exact_i32(const char* file, int line, const char* tag,
                             const std::vector<std::int32_t>& actual,
                             const std::int32_t* expected, std::size_t count) {
    assert_exact_i32(file, line, tag, actual.data(), actual.size(), expected, count);
}

inline void assert_exact_u64(const char* file, int line, const char* tag,
                             const std::uint64_t* actual, std::size_t actual_count,
                             const std::uint64_t* expected, std::size_t count) {
    auto& reg = Registry::shared();
    if (count == 0) {
        std::printf("  FAIL %s:%d: assert_exact_u64(%s) compared 0 elements\n", file, line, tag);
        reg.failures++;
        return;
    }
    if (actual_count != count) {
        std::printf("  FAIL %s:%d: %s actual size %zu != expected count %zu (size guard)\n",
                    file, line, tag, actual_count, count);
        reg.failures++;
        return;
    }
    std::size_t bad = 0, first = 0;
    for (std::size_t i = 0; i < count; i++)
        if (actual[i] != expected[i]) {
            if (bad == 0) first = i;
            bad++;
        }
    if (bad) {
        std::printf("  FAIL %s:%d: %s exact mismatch %zu/%zu (first [%zu]: 0x%016llx vs "
                    "0x%016llx)\n",
                    file, line, tag, bad, count, first,
                    static_cast<unsigned long long>(actual[first]),
                    static_cast<unsigned long long>(expected[first]));
        reg.failures++;
    }
}

inline void assert_exact_u64(const char* file, int line, const char* tag,
                             const std::vector<std::uint64_t>& actual,
                             const std::uint64_t* expected, std::size_t count) {
    assert_exact_u64(file, line, tag, actual.data(), actual.size(), expected, count);
}

} // namespace sxtest

// Two argument forms (overloads): (vector, expected_ptr, count) — preferred, the actual size is
// automatic / (ptr, actual_count, expected_ptr, count) — for non-vector call sites such as
// reinterpret casts.
#define SX_EXACT_I32(tag, ...) \
    ::sxtest::assert_exact_i32(__FILE__, __LINE__, (tag), __VA_ARGS__)
#define SX_EXACT_U64(tag, ...) \
    ::sxtest::assert_exact_u64(__FILE__, __LINE__, (tag), __VA_ARGS__)
