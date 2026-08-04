#pragma once
#include <memory>
#include <string>
#include "splatx/KernelBackend.h"

namespace splatx {

// Layered kernel library loading descriptor (Phase 2 Task 2 — the v0.1 relocatable deployment
// convention).
//
// Priority (tried from the top):
//   (a) metallib_path   — an explicit path to an offline-compiled .metallib (the output of
//                         tools/build_metallib.sh). Loaded with newLibrary(URL:). An empty
//                         string skips this layer.
//                         **On failure it throws immediately** — this is an explicit user
//                         intent, so it MUST NOT fall back to the automatic layers (this blocks
//                         the trap where a typo in the path is silently replaced by a stale
//                         kernel or a source compile — Phase 2 Task 2 review).
//   (b) search_exe_dir  — automatic search for "splatx.metallib" next to the executable
//                         (_NSGetExecutablePath). This is the canonical path by which a deployed
//                         binary runs without a source tree.
//                         Absence is normal (fall through to the next layer); a load failure
//                         warns on stderr and then falls through to the next layer.
//   (c) kernel_src_dir  — the directory of .metal sources + shared headers
//                         (Sources/SplatXKernelSrc). The shared headers are placed first and the
//                         *.metal files are concatenated in name order, then compiled at runtime
//                         with newLibrary(source:) (the existing development-path fallback;
//                         the concatenated-compile convention: no duplicate symbols
//                         library-wide — D5). An empty string skips this layer.
//
// If every automatic layer ((b)/(c)) fails, a std::runtime_error accumulating the per-layer
// reasons is thrown.
// Which layer was selected is reported by KernelBackend::name():
//   "metal(metallib)" — (a)/(b), "metal(source)" — (c).
// function constants are kept unspecialized in the metallib — they are specialized at pipeline
// creation, so the kernel behavior is identical on both paths (MetallibTests gates this by
// comparing against the goldens and the source path).
struct MetalBackendDesc {
    std::string metallib_path;  // (a) explicit .metallib — an empty string skips this layer
    bool search_exe_dir = true; // (b) auto-search for splatx.metallib next to the executable
    std::string kernel_src_dir; // (c) source runtime-compile fallback — empty string skips it
};

// Creates the metal-cpp based KernelBackend (layered loading).
// device_ptr: a host-owned MTL::Device* (it is retained). If nullptr, one is created internally
//             with MTL::CreateSystemDefaultDevice().
// Failure (every loading layer failed, or no device) raises std::runtime_error.
std::unique_ptr<KernelBackend> make_metal_backend(void* device_ptr,
                                                  const MetalBackendDesc& desc);

// Backward-compatible signature — source compilation only (the reference path for tests and
// existing call sites).
// Identical to MetalBackendDesc{ .search_exe_dir = false, .kernel_src_dir = kernel_src_dir }:
// it never takes the metallib layers, so it deterministically resolves to the source path
// regardless of whether an artifact sits next to the executable.
std::unique_ptr<KernelBackend> make_metal_backend(void* device_ptr,
                                                  const std::string& kernel_src_dir);

} // namespace splatx
