// Derived from ArthurBrussee/brush (Apache-2.0), commit 3b80985 — the sort_isect host sort loop (a structural port of lib.rs:16–125).
// Derived from nerfstudio-project/gsplat (Apache-2.0), commit 77ab983 — the isect_tiles host orchestration (a structural port of Intersect.cpp).
// Reimplemented for Metal/C++.
#include "splatx/MetalBackend.h"
#include "splatx/DepthSupervision.h"
#include "splatx/Validate.h" // shared single source of truth for entry validation (throws as the CPU reference does)

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <mach-o/dyld.h> // _NSGetExecutablePath — auto-discovery of an exe-adjacent metallib (layer b)

// The 3 host-shared headers out of the 4 shared headers — they double as the dual host/device
// compilation check and as the single source of truth for setBytes.
#include "FunctionConstants.h"
#include "KernelConstants.h"
#include "KernelParams.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>  // stderr fallback warnings (layered loading — no silent fallback)
#include <cstdlib> // getenv — the SPLATX_KERNEL_TIMING probe gate
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace splatx {
namespace {

namespace fs = std::filesystem;

struct AutoreleasePoolGuard {
    NS::AutoreleasePool* pool;
    AutoreleasePoolGuard() : pool(NS::AutoreleasePool::alloc()->init()) {}
    ~AutoreleasePoolGuard() { pool->release(); }
    AutoreleasePoolGuard(const AutoreleasePoolGuard&) = delete;
    AutoreleasePoolGuard& operator=(const AutoreleasePoolGuard&) = delete;
};

std::string ns_error_suffix(NS::Error* error) {
    if (error && error->localizedDescription() && error->localizedDescription()->utf8String())
        return std::string(": ") + error->localizedDescription()->utf8String();
    return {};
}

[[noreturn]] void throw_ns_error(const std::string& what, NS::Error* error) {
    throw std::runtime_error(what + ns_error_suffix(error));
}

// [Step breakdown probe — taming-bwd Task 1] Gated by the SPLATX_KERNEL_TIMING environment variable.
// When it is set, commit_and_wait writes the GPU time per command buffer (the GPUStartTime/GPUEndTime
// difference) to stderr as a single line "[ktime] <label> <ms>" — solely for measuring the GPU time
// share of each kernel group (the methodology of record is docs/bench/2026-07-21-step-breakdown.md).
// It is not a deployment contract.
// The default (unset) path costs nothing beyond one cached bool branch. As with the path environment
// variables, an empty string counts as unset, and "0" is also read as off.
bool kernel_timing_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("SPLATX_KERNEL_TIMING");
        return v && *v && std::strcmp(v, "0") != 0;
    }();
    return enabled;
}

// The executable's directory (_NSGetExecutablePath) — the anchor for layer (b) splatx.metallib
// auto-discovery. On a resolution failure it returns an empty path (the caller accumulates
// diagnostics). Symlinks and relative components are normalized with weakly_canonical.
fs::path executable_dir() {
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size); // writes the required buffer size into size (returns -1)
    if (size == 0) return {};
    std::vector<char> buf(static_cast<std::size_t>(size) + 1, '\0');
    if (_NSGetExecutablePath(buf.data(), &size) != 0) return {};
    std::error_code ec;
    fs::path p = fs::weakly_canonical(fs::path(buf.data()), ec);
    if (ec) p = fs::path(buf.data());
    return p.parent_path();
}

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("MetalBackend: cannot read " + p.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Strips local includes (#include "...") — in the concatenated source the loader inserts the shared
// headers once at the top, so a local include inside a file would be a duplicate. (System includes
// (<...>) are kept — a duplicate metal_stdlib include is harmless.)
std::string strip_local_includes(const std::string& src) {
    std::istringstream in(src);
    std::ostringstream out;
    std::string line;
    while (std::getline(in, line)) {
        const auto first = line.find_first_not_of(" \t");
        if (first != std::string::npos && line.compare(first, 10, "#include \"") == 0) continue;
        out << line << '\n';
    }
    return out.str();
}

// KernelSrc directory → a single MSL source string.
// It puts the shared headers first (in a fixed dependency order) and concatenates the *.metal files
// by name. Every .metal compiles into one library, so duplicate symbols are prohibited (D5).
std::string assemble_library_source(const std::string& dir) {
    const fs::path root(dir);
    std::error_code ec;
    if (!fs::is_directory(root, ec))
        throw std::runtime_error("MetalBackend: kernel source directory not found: " + dir);

    static const char* const kSharedHeaders[] = {
        "KernelConstants.h", "FunctionConstants.h", "KernelParams.h", "KernelCommon.h",
        "RngPhilox.h"};

    std::ostringstream src;
    src << "#include <metal_stdlib>\n";
    for (const char* name : kSharedHeaders) {
        const fs::path p = root / name;
        if (!fs::exists(p, ec)) // never miss one silently — diagnose before the kernels fail on an undefined symbol
            throw std::runtime_error("MetalBackend: required shared header missing: " + p.string());
        src << "// ==== " << name << " ====\n" << strip_local_includes(read_file(p)) << '\n';
    }

    std::vector<fs::path> metal_files;
    for (const auto& entry : fs::directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().extension() == ".metal")
            metal_files.push_back(entry.path());
    }
    if (metal_files.empty())
        throw std::runtime_error("MetalBackend: no .metal sources in " + dir);
    std::sort(metal_files.begin(), metal_files.end(), [](const fs::path& a, const fs::path& b) {
        return a.filename().string() < b.filename().string();
    });
    for (const auto& p : metal_files)
        src << "// ==== " << p.filename().string() << " ====\n"
            << strip_local_includes(read_file(p)) << '\n';
    return src.str();
}

class MetalTrainStepExecutor; // the D11 resident step executor (defined below — it reaches into MetalBackend)

class MetalBackend final : public KernelBackend {
    friend class MetalTrainStepExecutor; // shares device_/queue_/pipeline()/helpers (same TU)

public:
    // Layered loading (the MetalBackend.h convention): (a) an explicit metallib → (b) an
    // exe-adjacent splatx.metallib → (c) compiling the source in kernel_src_dir.
    // (a) is the user's explicit intent — on failure there is no fallback, it throws immediately
    // (this blocks the trap where a typo in the path is silently replaced by a stale kernel or a
    // source compile — Phase 2 Task 2 review, Important 1).
    // A failure in the automatic layers (b)/(c) warns on stderr, accumulates diagnostics, and moves
    // to the next layer; if every layer fails it throws with the accumulated diagnostics. The chosen
    // layer is reflected in name(): "metal(metallib)" / "metal(source)".
    MetalBackend(void* device_ptr, const MetalBackendDesc& desc) {
        AutoreleasePoolGuard pool;
        if (device_ptr) {
            device_ = NS::RetainPtr(static_cast<MTL::Device*>(device_ptr));
        } else {
            device_ = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
        }
        if (!device_) throw std::runtime_error("MetalBackend: no Metal device available");

        std::string diag; // accumulates the failure reason per layer — the throw message when every layer fails

        // (a) An explicit metallib path — failure = throw immediately (it does not fall back to the automatic layers)
        if (!desc.metallib_path.empty()) {
            if (!try_load_metallib(desc.metallib_path, "explicit metallib", diag))
                throw std::runtime_error("MetalBackend: explicit metallib load failed:" + diag);
        }

        // (b) Auto-discovery of an exe-adjacent splatx.metallib (the reference path for relocatable deployment)
        if (!library_ && desc.search_exe_dir) {
            const fs::path dir = executable_dir();
            if (dir.empty()) {
                diag += "\n  [exe-adjacent] cannot resolve executable path";
            } else {
                const fs::path p = dir / "splatx.metallib";
                std::error_code ec;
                if (fs::is_regular_file(p, ec)) {
                    if (!try_load_metallib(p.string(), "exe-adjacent metallib", diag))
                        std::fprintf(stderr,
                                     "MetalBackend: exe-adjacent metallib load failed (%s) — "
                                     "falling back to source\n",
                                     p.string().c_str());
                } else {
                    // Absence during auto-discovery is a normal scenario (a development tree) — it
                    // accumulates a diagnostic without warning.
                    diag += "\n  [exe-adjacent] not found: " + p.string();
                }
            }
        }

        // (c) Runtime source compilation fallback (the existing development path)
        if (!library_) {
            if (desc.kernel_src_dir.empty()) {
                diag += "\n  [source] kernel_src_dir not given";
            } else {
                try {
                    const std::string source = assemble_library_source(desc.kernel_src_dir);
                    NS::Error* error = nullptr;
                    auto options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
                    // D2: keeps the Metal default (fast-math) — parity points are stated explicitly
                    // as precise:: inside the kernels.
                    NS::String* nssource =
                        NS::String::string(source.c_str(), NS::UTF8StringEncoding);
                    library_ =
                        NS::TransferPtr(device_->newLibrary(nssource, options.get(), &error));
                    if (!library_)
                        throw_ns_error("kernel library compile failed", error);
                    name_ = "metal(source)";
                } catch (const std::exception& e) {
                    diag += std::string("\n  [source] ") + e.what();
                }
            }
        }

        if (!library_)
            throw std::runtime_error("MetalBackend: all library loading layers failed:" + diag);

        queue_ = NS::TransferPtr(device_->newCommandQueue());
        if (!queue_) throw std::runtime_error("MetalBackend: newCommandQueue failed");
    }

    const char* name() const override { return name_.c_str(); }

    ProjectionOutputs projection_ewa_fwd(const ProjectionInputs&) override {
        throw std::runtime_error("projection_ewa_fwd: unimplemented on metal backend");
    }

    // Spec 01 projection_ewa_3dgs_fused_fwd — a pure map kernel (no atomics, no barriers; fwd takes
    // the same path regardless of C — kProjSingleCamera is bwd-only).
    // function constants: kProjUseCovars (covars6 vs quat/scale), kProjHasOpacities, kProjCalcComp.
    // An argument specialized to false is removed from the signature and needs no binding (D8-5).
    // Culled slot convention (D3): the kernel writes 0 only into the 2 radii slots — the culled slots
    // of means2d/depths/conics are unwritten (garbage, matching gsplat at::empty), so those slots
    // hold undefined values in the returned vectors as well. Do not read them, and do not zero-fill
    // them — that would diverge from the original and break the goldens. compensations is the one
    // exception: gsplat allocates it with at::zeros (Projection.cpp:411), so it is zero-initialized
    // identically here.
    ProjectionFusedOutputs projection_fused_fwd(const ProjectionFusedInputs& in) override {
        validate_projection_fused_inputs(in, "projection_fused_fwd");

        const std::size_t N = static_cast<std::size_t>(in.N);
        const std::size_t C = static_cast<std::size_t>(in.C);
        const std::size_t CN = C * N;
        ProjectionFusedOutputs out;
        out.radii.resize(CN * 2);
        out.means2d.resize(CN * 2);
        out.depths.resize(CN);
        out.conics.resize(CN * 3);
        if (in.calc_compensations) out.compensations.resize(CN);
        if (CN == 0) return out; // D8-6 dispatch skip

        const bool use_covars = in.covars != nullptr;
        const bool has_opacities = in.opacities != nullptr;

        ProjectionParams P;
        P.N = static_cast<uint>(N);
        P.C = static_cast<uint>(C);
        P.image_width = static_cast<uint>(in.width);
        P.image_height = static_cast<uint>(in.height);
        P.eps2d = in.eps2d;
        P.near_plane = in.near_plane;
        P.far_plane = in.far_plane;
        P.radius_clip = in.radius_clip;
        P.camera_model = in.camera_model;
        P.has_opacities = has_opacities ? 1u : 0u;
        P.calc_compensations = in.calc_compensations ? 1u : 0u;

        AutoreleasePoolGuard pool;
        MTL::ComputePipelineState* pso =
            pipeline("projection_ewa_3dgs_fused_fwd", {{kFCProjUseCovars, use_covars},
                                                       {kFCProjHasOpacities, has_opacities},
                                                       {kFCProjCalcComp, in.calc_compensations}});

        auto means = upload_buffer(in.means, N * 3 * sizeof(float));
        NS::SharedPtr<MTL::Buffer> covars, quats, scales, opacities;
        if (use_covars) {
            covars = upload_buffer(in.covars, N * 6 * sizeof(float));
        } else {
            quats = upload_buffer(in.quats, N * 4 * sizeof(float));
            scales = upload_buffer(in.scales, N * 3 * sizeof(float));
        }
        if (has_opacities) opacities = upload_buffer(in.opacities, N * sizeof(float));
        auto viewmats = upload_buffer(in.viewmats, C * 16 * sizeof(float));
        auto Ks = upload_buffer(in.Ks, C * 9 * sizeof(float));

        auto radii = upload_buffer(nullptr, CN * 2 * sizeof(int32_t));
        auto means2d = upload_buffer(nullptr, CN * 2 * sizeof(float));
        auto depths = upload_buffer(nullptr, CN * sizeof(float));
        auto conics = upload_buffer(nullptr, CN * 3 * sizeof(float));
        NS::SharedPtr<MTL::Buffer> compensations;
        if (in.calc_compensations) {
            compensations = upload_buffer(nullptr, CN * sizeof(float));
            std::memset(compensations->contents(), 0, CN * sizeof(float)); // matching at::zeros
        }

        MTL::CommandBuffer* cmd = queue_->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(means.get(), 0, 0);
        if (use_covars) {
            enc->setBuffer(covars.get(), 0, 1);
        } else {
            enc->setBuffer(quats.get(), 0, 2);
            enc->setBuffer(scales.get(), 0, 3);
        }
        if (has_opacities) enc->setBuffer(opacities.get(), 0, 4);
        enc->setBuffer(viewmats.get(), 0, 5);
        enc->setBuffer(Ks.get(), 0, 6);
        enc->setBuffer(radii.get(), 0, 7);
        enc->setBuffer(means2d.get(), 0, 8);
        enc->setBuffer(depths.get(), 0, 9);
        enc->setBuffer(conics.get(), 0, 10);
        if (in.calc_compensations) enc->setBuffer(compensations.get(), 0, 11);
        enc->setBytes(&P, sizeof(ProjectionParams), 12);
        // Spec 01: dispatchThreads(C*N), threadgroup 256 (1D)
        enc->dispatchThreads(MTL::Size::Make(CN, 1, 1), MTL::Size::Make(256, 1, 1));
        enc->endEncoding();
        commit_and_wait(cmd, "projection_ewa_3dgs_fused_fwd");

        std::memcpy(out.radii.data(), radii->contents(), CN * 2 * sizeof(int32_t));
        std::memcpy(out.means2d.data(), means2d->contents(), CN * 2 * sizeof(float));
        std::memcpy(out.depths.data(), depths->contents(), CN * sizeof(float));
        std::memcpy(out.conics.data(), conics->contents(), CN * 3 * sizeof(float));
        if (in.calc_compensations)
            std::memcpy(out.compensations.data(), compensations->contents(), CN * sizeof(float));
        return out;
    }

    // Spec 01 projection_ewa_3dgs_fused_bwd — a rematerialization VJP. The gradient output buffers
    // MUST be zero-filled through encode_zero_fill (D8-1) in the same command buffer before the
    // kernel accumulates into them.
    // function constants: kProjUseCovars / kProjCalcComp (= v_compensations != nullptr — the same
    // gate as CUDA's nullptr) / kProjSingleCamera (a non-atomic store at C==1, bit-exact
    // determinism) / kProjViewmatsGrad (g.viewmats_grad — outputs v_viewmats [C,16], golden-verified
    // in pose Task 1; B8 reduction strategy 2: a cid-uniform simdgroup uses simd_sum + a lane0
    // atomic).
    // bwd does not take opacities (spec 01 Correction 1 — in.opacities is ignored).
    ProjectionFusedGrads projection_fused_bwd(const ProjectionFusedInputs& in,
                                              const ProjectionFusedGradsIn& g) override {
        validate_projection_fused_inputs(in, "projection_fused_bwd");
        validate_projection_fused_grads_in(g, "projection_fused_bwd");
        const bool calc_comp = g.v_compensations != nullptr;

        const std::size_t N = static_cast<std::size_t>(in.N);
        const std::size_t C = static_cast<std::size_t>(in.C);
        const std::size_t CN = C * N;
        const bool use_covars = in.covars != nullptr;
        ProjectionFusedGrads out;
        out.v_means.assign(N * 3, 0.f);
        if (use_covars) {
            out.v_covars.assign(N * 6, 0.f);
        } else {
            out.v_quats.assign(N * 4, 0.f);
            out.v_scales.assign(N * 3, 0.f);
        }
        if (g.viewmats_grad) out.v_viewmats.assign(C * 16, 0.f);
        if (CN == 0) return out; // D8-6 dispatch skip (preserving zeros_like semantics)

        ProjectionParams P;
        P.N = static_cast<uint>(N);
        P.C = static_cast<uint>(C);
        P.image_width = static_cast<uint>(in.width);
        P.image_height = static_cast<uint>(in.height);
        P.eps2d = in.eps2d;
        P.near_plane = in.near_plane;
        P.far_plane = in.far_plane;
        P.radius_clip = in.radius_clip;
        P.camera_model = in.camera_model;
        P.has_opacities = 0u; // unused in bwd (Correction 1)
        P.calc_compensations = calc_comp ? 1u : 0u;

        AutoreleasePoolGuard pool;
        MTL::ComputePipelineState* pso =
            pipeline("projection_ewa_3dgs_fused_bwd",
                     {{kFCProjUseCovars, use_covars},
                      {kFCProjCalcComp, calc_comp},
                      {kFCProjSingleCamera, C == 1},
                      {kFCProjViewmatsGrad, g.viewmats_grad}});

        auto means = upload_buffer(in.means, N * 3 * sizeof(float));
        NS::SharedPtr<MTL::Buffer> covars, quats, scales;
        if (use_covars) {
            covars = upload_buffer(in.covars, N * 6 * sizeof(float));
        } else {
            quats = upload_buffer(in.quats, N * 4 * sizeof(float));
            scales = upload_buffer(in.scales, N * 3 * sizeof(float));
        }
        auto viewmats = upload_buffer(in.viewmats, C * 16 * sizeof(float));
        auto Ks = upload_buffer(in.Ks, C * 9 * sizeof(float));
        auto radii = upload_buffer(g.radii, CN * 2 * sizeof(int32_t));
        auto conics = upload_buffer(g.conics, CN * 3 * sizeof(float));
        NS::SharedPtr<MTL::Buffer> comps, v_comps;
        if (calc_comp) {
            comps = upload_buffer(g.compensations, CN * sizeof(float));
            v_comps = upload_buffer(g.v_compensations, CN * sizeof(float));
        }
        auto v_means2d = upload_buffer(g.v_means2d, CN * 2 * sizeof(float));
        auto v_depths = upload_buffer(g.v_depths, CN * sizeof(float));
        auto v_conics = upload_buffer(g.v_conics, CN * 3 * sizeof(float));

        auto v_means = upload_buffer(nullptr, N * 3 * sizeof(float));
        NS::SharedPtr<MTL::Buffer> v_covars, v_quats, v_scales, v_viewmats;
        if (use_covars) {
            v_covars = upload_buffer(nullptr, N * 6 * sizeof(float));
        } else {
            v_quats = upload_buffer(nullptr, N * 4 * sizeof(float));
            v_scales = upload_buffer(nullptr, N * 3 * sizeof(float));
        }
        if (g.viewmats_grad) v_viewmats = upload_buffer(nullptr, C * 16 * sizeof(float));

        MTL::CommandBuffer* cmd = queue_->commandBuffer();
        // D8-1: the gradient accumulation buffers MUST be zero-filled through encode_zero_fill
        // (encoding order blit → compute; unallocated (nullptr) entries are skipped by the helper).
        MTL::Buffer* const grad_bufs[] = {v_means.get(), v_covars.get(), v_quats.get(),
                                          v_scales.get(), v_viewmats.get()};
        encode_zero_fill(cmd, grad_bufs, 5);

        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(means.get(), 0, 0);
        if (use_covars) {
            enc->setBuffer(covars.get(), 0, 1);
        } else {
            enc->setBuffer(quats.get(), 0, 2);
            enc->setBuffer(scales.get(), 0, 3);
        }
        enc->setBuffer(viewmats.get(), 0, 4);
        enc->setBuffer(Ks.get(), 0, 5);
        enc->setBuffer(radii.get(), 0, 6);
        enc->setBuffer(conics.get(), 0, 7);
        if (calc_comp) {
            enc->setBuffer(comps.get(), 0, 8);
            enc->setBuffer(v_comps.get(), 0, 12);
        }
        enc->setBuffer(v_means2d.get(), 0, 9);
        enc->setBuffer(v_depths.get(), 0, 10);
        enc->setBuffer(v_conics.get(), 0, 11);
        enc->setBuffer(v_means.get(), 0, 13);
        if (use_covars) {
            enc->setBuffer(v_covars.get(), 0, 14);
        } else {
            enc->setBuffer(v_quats.get(), 0, 15);
            enc->setBuffer(v_scales.get(), 0, 16);
        }
        if (g.viewmats_grad) enc->setBuffer(v_viewmats.get(), 0, 17);
        enc->setBytes(&P, sizeof(ProjectionParams), 18);
        // Spec 01: dispatchThreads(C*N), threadgroup 256 (1D) — the same as fwd
        enc->dispatchThreads(MTL::Size::Make(CN, 1, 1), MTL::Size::Make(256, 1, 1));
        enc->endEncoding();
        commit_and_wait(cmd, "projection_ewa_3dgs_fused_bwd");

        std::memcpy(out.v_means.data(), v_means->contents(), N * 3 * sizeof(float));
        if (use_covars) {
            std::memcpy(out.v_covars.data(), v_covars->contents(), N * 6 * sizeof(float));
        } else {
            std::memcpy(out.v_quats.data(), v_quats->contents(), N * 4 * sizeof(float));
            std::memcpy(out.v_scales.data(), v_scales->contents(), N * 3 * sizeof(float));
        }
        if (g.viewmats_grad)
            std::memcpy(out.v_viewmats.data(), v_viewmats->contents(), C * 16 * sizeof(float));
        return out;
    }

    // Spec 02 sh_fwd (generic) — one thread = (batch, gaussian, channel), a pure map (no atomics).
    // Specialized on kShHasMasks (masks nullptr → the false variant, where the argument itself is
    // removed — D8-5).
    // colors for masked-out elements are unwritten (the disposition #5 standalone contract —
    // zero-filling is prohibited): those slots of the returned vector hold undefined values (matching
    // gsplat at::empty, in the same family as the D3 culled slot convention).
    // The harness compares under the mask. The `+0.5`, clamp, and relu_mask are host post-ops (owned
    // by Phase 1c).
    ShOutputs sh_fwd(const ShInputs& in) override {
        validate_sh_inputs(in, "sh_fwd");
        const std::size_t BN = static_cast<std::size_t>(in.B) * static_cast<std::size_t>(in.N);
        const std::size_t total = BN * in.D;
        ShOutputs out;
        out.colors.resize(total);
        if (total == 0) return out; // D8-6 dispatch skip

        SHParams P;
        P.degrees_to_use = in.degrees_to_use;
        P.B = static_cast<uint>(in.B);
        P.N = static_cast<uint>(in.N);
        P.K = static_cast<uint>(in.K);
        P.D = in.D;

        const bool has_masks = in.masks != nullptr;
        AutoreleasePoolGuard pool;
        MTL::ComputePipelineState* pso = pipeline("sh_fwd", {{kFCShHasMasks, has_masks}});

        auto dirs = upload_buffer(in.dirs, BN * 3 * sizeof(float));
        auto coeffs = upload_buffer(
            in.coeffs, static_cast<std::size_t>(in.N) * static_cast<std::size_t>(in.K) * in.D *
                           sizeof(float));
        NS::SharedPtr<MTL::Buffer> masks;
        if (has_masks) masks = upload_buffer(in.masks, BN);
        auto colors = upload_buffer(nullptr, total * sizeof(float));

        MTL::CommandBuffer* cmd = queue_->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(dirs.get(), 0, 0);
        enc->setBuffer(coeffs.get(), 0, 1);
        if (has_masks) enc->setBuffer(masks.get(), 0, 2);
        enc->setBuffer(colors.get(), 0, 3);
        enc->setBytes(&P, sizeof(SHParams), 4);
        // Spec 02: dispatchThreads(B*N*D), threadgroup 256 (1D)
        enc->dispatchThreads(MTL::Size::Make(total, 1, 1), MTL::Size::Make(256, 1, 1));
        enc->endEncoding();
        commit_and_wait(cmd, "sh_fwd");

        std::memcpy(out.colors.data(), colors->contents(), total * sizeof(float));
        return out;
    }

    // Spec 02 sh_bwd — a generic 1:1 atomic (v_coeffs: contention across batches, none at B==1;
    // v_dirs: contention across the D channels). The gradient buffers MUST be zero-filled through
    // encode_zero_fill (D8-1) in the same command buffer before the kernel accumulates (v_coeffs
    // matches CUDA's at::zeros).
    // function constants: kShHasMasks / kShComputeVDirs (at false there is no v_dirs argument and no
    // v_dirs output at all).
    ShGrads sh_bwd(const ShInputs& in, const float* v_colors) override {
        validate_sh_inputs(in, "sh_bwd");
        if (!v_colors) throw std::invalid_argument("sh_bwd: null v_colors");
        const std::size_t BN = static_cast<std::size_t>(in.B) * static_cast<std::size_t>(in.N);
        const std::size_t total = BN * in.D;
        const std::size_t coeffs_count =
            static_cast<std::size_t>(in.N) * static_cast<std::size_t>(in.K) * in.D;
        ShGrads out;
        out.v_coeffs.assign(coeffs_count, 0.f);
        if (in.compute_v_dirs) out.v_dirs.assign(BN * 3, 0.f);
        if (total == 0) return out; // D8-6 dispatch skip (preserving zeros semantics)

        SHParams P;
        P.degrees_to_use = in.degrees_to_use;
        P.B = static_cast<uint>(in.B);
        P.N = static_cast<uint>(in.N);
        P.K = static_cast<uint>(in.K);
        P.D = in.D;

        const bool has_masks = in.masks != nullptr;
        AutoreleasePoolGuard pool;
        MTL::ComputePipelineState* pso =
            pipeline("sh_bwd", {{kFCShHasMasks, has_masks},
                                {kFCShComputeVDirs, in.compute_v_dirs}});

        auto dirs = upload_buffer(in.dirs, BN * 3 * sizeof(float));
        auto coeffs = upload_buffer(in.coeffs, coeffs_count * sizeof(float));
        NS::SharedPtr<MTL::Buffer> masks;
        if (has_masks) masks = upload_buffer(in.masks, BN);
        auto v_colors_buf = upload_buffer(v_colors, total * sizeof(float));
        auto v_coeffs = upload_buffer(nullptr, coeffs_count * sizeof(float));
        NS::SharedPtr<MTL::Buffer> v_dirs;
        if (in.compute_v_dirs) v_dirs = upload_buffer(nullptr, BN * 3 * sizeof(float));

        MTL::CommandBuffer* cmd = queue_->commandBuffer();
        // D8-1: the gradient accumulation buffers MUST be zero-filled through encode_zero_fill
        // (encoding order blit → compute; unallocated (nullptr) entries are skipped by the helper).
        MTL::Buffer* const grad_bufs[] = {v_coeffs.get(), v_dirs.get()};
        encode_zero_fill(cmd, grad_bufs, 2);

        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(dirs.get(), 0, 0);
        enc->setBuffer(coeffs.get(), 0, 1);
        if (has_masks) enc->setBuffer(masks.get(), 0, 2);
        enc->setBuffer(v_colors_buf.get(), 0, 3);
        enc->setBuffer(v_coeffs.get(), 0, 4);
        if (in.compute_v_dirs) enc->setBuffer(v_dirs.get(), 0, 5);
        enc->setBytes(&P, sizeof(SHParams), 6);
        // Spec 02: dispatchThreads(B*N*D), threadgroup 256 (1D) — the same as fwd
        enc->dispatchThreads(MTL::Size::Make(total, 1, 1), MTL::Size::Make(256, 1, 1));
        enc->endEncoding();
        commit_and_wait(cmd, "sh_bwd");

        std::memcpy(out.v_coeffs.data(), v_coeffs->contents(), coeffs_count * sizeof(float));
        if (in.compute_v_dirs)
            std::memcpy(out.v_dirs.data(), v_dirs->contents(), BN * 3 * sizeof(float));
        return out;
    }

    // Spec 03 intersect_tile — 2 passes (count → GPU scan → emit; matching
    // Intersect.cpp:230–296).
    // Path selection: if both conics and opacities are present it specializes kIsectAccuTile=true
    // (the IntersectTile.cu:288 gate — the validator guarantees both-or-neither); otherwise AABB
    // (false). Task 4: the cumsum moved from the host CPU to a GPU 3-dispatch scan
    // (encode_prefix_scan) — encoded into the same command buffer and encoder as count (spec 03
    // §Dispatch sequence summary; the output contract is unchanged).
    // The n_isects readback is the same synchronization point as the original (Intersect.cpp:259
    // .item<int64_t>(), D9-3) — it reads the Apple Silicon UMA storageModeShared buffer directly (no
    // blit, spec 03 §Kernel 2).
    // Host skips: at n_elements==0 it returns without dispatching (Intersect.cpp:236–270 →
    // n_isects=0), and at n_isects==0 it omits the emit (Intersect.cpp:275). The u64 keys live in a
    // device ulong buffer (D3).
    IsectTilesOutputs isect_tiles(const IsectTilesInputs& in) override {
        validate_isect_tiles_inputs(in, "isect_tiles");
        const std::size_t n_elements = static_cast<std::size_t>(in.I * in.N);
        IsectTilesOutputs out;
        out.tile_n_bits =
            bits_for_count(static_cast<int64_t>(in.tile_width) * in.tile_height);
        out.image_n_bits = bits_for_count(in.I); // IntersectTile.cu:507–508
        out.tiles_per_gauss.assign(n_elements, 0);
        if (n_elements == 0) return out; // D8-6 dispatch skip

        const bool accu = in.conics != nullptr && in.opacities != nullptr; // the :288 gate

        IsectTileParams P;
        P.n_elements = static_cast<uint>(n_elements);
        P.I = static_cast<uint>(in.I);
        P.N = static_cast<uint>(in.N);
        P.tile_size = in.tile_size;
        P.tile_width = in.tile_width;
        P.tile_height = in.tile_height;
        P.tile_n_bits = out.tile_n_bits;
        P.image_n_bits = out.image_n_bits;

        AutoreleasePoolGuard pool;
        auto means2d = upload_buffer(in.means2d, n_elements * 2 * sizeof(float));
        auto radii = upload_buffer(in.radii, n_elements * 2 * sizeof(int32_t));
        NS::SharedPtr<MTL::Buffer> conics, opacities; // AccuTile only — shared by both passes
        if (accu) {
            conics = upload_buffer(in.conics, n_elements * 3 * sizeof(float));
            opacities = upload_buffer(in.opacities, n_elements * sizeof(float));
        }
        auto tiles_per_gauss = upload_buffer(nullptr, n_elements * sizeof(int32_t));
        const uint32_t num_groups =
            (P.n_elements + kScanBlockSize - 1) / kScanBlockSize;
        auto cum_buf = upload_buffer(nullptr, n_elements * sizeof(int64_t));
        auto block_sums = upload_buffer(nullptr, num_groups * sizeof(int64_t));

        { // Pass 1: count — dispatchThreads(n_elements), threadgroup 256 (IntersectTile.cu:524–525)
          // + the 3-dispatch GPU prefix scan encoded back to back into the same encoder (Task 4 — it
          // replaces the host cumsum).
            MTL::ComputePipelineState* pso =
                pipeline("isect_tile_count", {{kFCIsectAccuTile, accu}});
            MTL::CommandBuffer* cmd = queue_->commandBuffer();
            MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(pso);
            enc->setBuffer(means2d.get(), 0, 0);
            enc->setBuffer(radii.get(), 0, 1);
            if (accu) { // specializing to false removes the argument itself — no binding needed (D8-5)
                enc->setBuffer(conics.get(), 0, 2);
                enc->setBuffer(opacities.get(), 0, 3);
            }
            enc->setBuffer(tiles_per_gauss.get(), 0, 4);
            enc->setBytes(&P, sizeof(IsectTileParams), 5);
            enc->dispatchThreads(MTL::Size::Make(n_elements, 1, 1), MTL::Size::Make(256, 1, 1));
            enc->memoryBarrier(MTL::BarrierScopeBuffers); // makes the count writes visible to the scan reads
            encode_prefix_scan(enc, tiles_per_gauss.get(), cum_buf.get(), block_sums.get(),
                               P.n_elements, num_groups);
            enc->endEncoding();
            commit_and_wait(cmd, "isect_tile_count+prefix_scan");
        }
        std::memcpy(out.tiles_per_gauss.data(), tiles_per_gauss->contents(),
                    n_elements * sizeof(int32_t));

        // n_isects = cum[-1], read back on the host (the D9-3 sync point — the only place the CPU
        // intervenes; it reads the UMA shared buffer directly)
        out.n_isects =
            static_cast<const int64_t*>(cum_buf->contents())[n_elements - 1];
        if (out.n_isects == 0) return out; // emit skipped (Intersect.cpp:275)
        out.isect_ids.resize(static_cast<std::size_t>(out.n_isects));
        out.flatten_ids.resize(static_cast<std::size_t>(out.n_isects));

        auto depths = upload_buffer(in.depths, n_elements * sizeof(float));
        // cum_buf is already device-resident (the scan output) — emit consumes it directly, with no re-upload.
        auto isect_ids = upload_buffer(nullptr, out.isect_ids.size() * sizeof(uint64_t));
        auto flatten_ids = upload_buffer(nullptr, out.flatten_ids.size() * sizeof(int32_t));

        { // Pass 2: emit — writes only within its own range [cum[idx-1], cum[idx]) (no atomics needed)
            MTL::ComputePipelineState* pso =
                pipeline("isect_tile_emit", {{kFCIsectAccuTile, accu}});
            MTL::CommandBuffer* cmd = queue_->commandBuffer();
            MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
            enc->setComputePipelineState(pso);
            enc->setBuffer(means2d.get(), 0, 0);
            enc->setBuffer(radii.get(), 0, 1);
            enc->setBuffer(depths.get(), 0, 2);
            if (accu) { // the same gate as count (buffers 3–4 are indices reserved by Task 3)
                enc->setBuffer(conics.get(), 0, 3);
                enc->setBuffer(opacities.get(), 0, 4);
            }
            enc->setBuffer(cum_buf.get(), 0, 5);
            enc->setBuffer(isect_ids.get(), 0, 6);
            enc->setBuffer(flatten_ids.get(), 0, 7);
            enc->setBytes(&P, sizeof(IsectTileParams), 8);
            enc->dispatchThreads(MTL::Size::Make(n_elements, 1, 1), MTL::Size::Make(256, 1, 1));
            enc->endEncoding();
            commit_and_wait(cmd, "isect_tile_emit");
        }
        std::memcpy(out.isect_ids.data(), isect_ids->contents(),
                    out.isect_ids.size() * sizeof(uint64_t));
        std::memcpy(out.flatten_ids.data(), flatten_ids->contents(),
                    out.flatten_ids.size() * sizeof(int32_t));
        return out;
    }

    // Spec 03 §Kernel 2 GPU prefix scan — int32 counts → an int64 inclusive cumsum (matching
    // at::cumsum with int64 stated explicitly, Intersect.cpp:258). Integer addition is independent
    // of association order → exact agreement with the CPU cumsum_i64 is the contract (ScanTests). It
    // runs the same encode_prefix_scan used inside isect_tiles, in a standalone command buffer —
    // exposed so synthetic arrays can be verified directly.
    std::vector<int64_t> prefix_scan_i64(const int32_t* v, int64_t n) override {
        if (n < 0) throw std::invalid_argument("prefix_scan_i64: negative n");
        if (n > 0 && !v) throw std::invalid_argument("prefix_scan_i64: null input");
        if (n > INT32_MAX) throw std::invalid_argument("prefix_scan_i64: n must fit in int32");
        std::vector<int64_t> out(static_cast<std::size_t>(n));
        if (n == 0) return out; // D8-6 dispatch skip

        const uint32_t n_u = static_cast<uint32_t>(n);
        const uint32_t num_groups = (n_u + kScanBlockSize - 1) / kScanBlockSize;

        AutoreleasePoolGuard pool;
        auto in_buf = upload_buffer(v, out.size() * sizeof(int32_t));
        auto out_buf = upload_buffer(nullptr, out.size() * sizeof(int64_t));
        auto block_sums = upload_buffer(nullptr, num_groups * sizeof(int64_t));

        MTL::CommandBuffer* cmd = queue_->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        encode_prefix_scan(enc, in_buf.get(), out_buf.get(), block_sums.get(), n_u, num_groups);
        enc->endEncoding();
        commit_and_wait(cmd, "prefix_scan_i64");

        std::memcpy(out.data(), out_buf->contents(), out.size() * sizeof(int64_t));
        return out;
    }

    // Spec 04 GPU radix sort (Task 7 — it replaces Task 6's delegation to the CPU; the interface and
    // the sort contract are unchanged).
    // An MSL port of the brush (3b80985) reduce-then-scan LSD radix (the 5 kernels of
    // RadixSort.metal): per pass count → reduce → scan (a single threadgroup) → scan_add → scatter,
    // with the pass count = ceil(key_n_bits / 4) — sorting only the valid bits saves passes (e.g.
    // isect_std at 40 bits → 10 passes, matching the cub end_bit convention). Zero high bits are the
    // caller's contract — the GPU does not validate (the CPU reference sort_isect_cpu substitutes a
    // throw for that validation — the same pattern as the spec 03 §Kernel 3 convention).
    // Stability guarantee: bin-major counts (global offsets ordered by (digit, workgroup)) × the EPT
    // iteration order within a workgroup × the local 2-bit split ranking (in thread index order) —
    // the spec 04 §brush kernel formulas.
    // OneSweep and decoupled-lookback are prohibited (Metal does not guarantee forward progress
    // between threadgroups) — the global scan is decomposed into 3 stages, reduce → single-group
    // scan → scan_add (spec 04 §Basis for the prohibition).
    //
    // The host loop corresponds structurally to brush lib.rs:16–125. Differences (spec 04
    // Corrections 5 and 6 + the D9 decision):
    //  - The readback + exact-size path: n is already on the host, so num_wgs/num_reduce_wgs are
    //    computed and dispatched exactly (the same configuration as the brush deployment path — it
    //    avoids the unproven combination of a capacity dispatch). num_keys is passed through params
    //    (setBytes), so a [1] GPU buffer is unnecessary.
    //  - Double-buffer reuse with a swap (brush allocates anew each pass — Correction 5; reuse is
    //    more efficient).
    //  - The reduced buffer is a multiple of BLOCK_SIZE with a max(1) guard (lib.rs:76 — a premise of
    //    the scan chunk gate).
    // Every pass is encoded into one command buffer, and memoryBarrier(BarrierScopeBuffers) between
    // dispatches makes write → read visibility explicit (the same pattern as encode_prefix_scan).
    // Keys are u64 (device ulong — the D3 unsigned contract) and values are i32 (bit-preserving — the
    // GPU treats them as uint).
    SortIsectOutputs sort_isect(const SortIsectInputs& in) override {
        validate_sort_isect_inputs(in, "sort_isect");
        SortIsectOutputs out;
        const std::size_t n = static_cast<std::size_t>(in.n);
        out.keys_sorted.resize(n);
        out.values_sorted.resize(n);
        if (n == 0) return out; // D8-6 dispatch skip (the upstream n_isects==0 contract)

        const uint32_t num_passes =
            (in.key_n_bits + kRadixBitsPerPass - 1) / kRadixBitsPerPass; // ceil, 1..16
        const uint32_t num_wgs =
            static_cast<uint32_t>((n + kRadixBlockSize - 1) / kRadixBlockSize);
        const uint32_t num_reduce_wgs =
            kRadixBinCount * ((num_wgs + kRadixBlockSize - 1) / kRadixBlockSize);
        // scan walks in BLOCK_SIZE chunks → allocate a multiple of BLOCK_SIZE (+ a max(1) guard
        // against n=0 — a port of the brush lib.rs:76 guard; n>0 here, but the convention is kept)
        const uint32_t reduced_size =
            std::max(1u, (num_reduce_wgs + kRadixBlockSize - 1) / kRadixBlockSize) *
            kRadixBlockSize;

        RadixSortParams P;
        P.shift = 0;
        P.num_keys = static_cast<uint>(n);        // the validator guarantees n <= INT32_MAX
        P.num_wgs = num_wgs;
        P.num_reduce_wgs = num_reduce_wgs;

        AutoreleasePoolGuard pool;
        MTL::ComputePipelineState* pso_count = pipeline("radix_sort_count");
        MTL::ComputePipelineState* pso_reduce = pipeline("radix_sort_reduce");
        MTL::ComputePipelineState* pso_scan = pipeline("radix_sort_scan");
        MTL::ComputePipelineState* pso_scan_add = pipeline("radix_sort_scan_add");
        MTL::ComputePipelineState* pso_scatter = pipeline("radix_sort_scatter");

        auto keys_a = upload_buffer(in.keys, n * sizeof(uint64_t));
        auto vals_a = upload_buffer(in.values, n * sizeof(int32_t));
        auto keys_b = upload_buffer(nullptr, n * sizeof(uint64_t));
        auto vals_b = upload_buffer(nullptr, n * sizeof(int32_t));
        auto counts = upload_buffer(nullptr,
                                    static_cast<std::size_t>(kRadixBinCount) * num_wgs *
                                        sizeof(uint32_t)); // bin-major [16 * num_wgs]
        auto reduced = upload_buffer(nullptr, reduced_size * sizeof(uint32_t));

        MTL::Buffer* cur_keys = keys_a.get();
        MTL::Buffer* cur_vals = vals_a.get();
        MTL::Buffer* dst_keys = keys_b.get();
        MTL::Buffer* dst_vals = vals_b.get();

        const MTL::Size tg = MTL::Size::Make(kRadixWG, 1, 1);
        MTL::CommandBuffer* cmd = queue_->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        for (uint32_t pass = 0; pass < num_passes; pass++) {
            P.shift = pass * kRadixBitsPerPass;

            // 1) count — a histogram of BLOCK_SIZE keys per workgroup, output bin-major (the premise of stability)
            enc->setComputePipelineState(pso_count);
            enc->setBuffer(cur_keys, 0, 0);
            enc->setBuffer(counts.get(), 0, 1);
            enc->setBytes(&P, sizeof(RadixSortParams), 2);
            enc->dispatchThreadgroups(MTL::Size::Make(num_wgs, 1, 1), tg);
            enc->memoryBarrier(MTL::BarrierScopeBuffers);

            // 2) reduce — sums counts in BLOCK_SIZE chunks → reduced[num_reduce_wgs]
            enc->setComputePipelineState(pso_reduce);
            enc->setBuffer(counts.get(), 0, 0);
            enc->setBuffer(reduced.get(), 0, 1);
            enc->setBytes(&P, sizeof(RadixSortParams), 2);
            enc->dispatchThreadgroups(MTL::Size::Make(num_reduce_wgs, 1, 1), tg);
            enc->memoryBarrier(MTL::BarrierScopeBuffers);

            // 3) scan — MUST be a single threadgroup (this is what removes the forward-progress assumption)
            enc->setComputePipelineState(pso_scan);
            enc->setBuffer(reduced.get(), 0, 0);
            enc->setBytes(&P, sizeof(RadixSortParams), 1);
            enc->dispatchThreadgroups(MTL::Size::Make(1, 1, 1), tg);
            enc->memoryBarrier(MTL::BarrierScopeBuffers);

            // 4) scan_add — a local scan of counts + the global base from reduced → the global exclusive offset
            enc->setComputePipelineState(pso_scan_add);
            enc->setBuffer(reduced.get(), 0, 0);
            enc->setBuffer(counts.get(), 0, 1);
            enc->setBytes(&P, sizeof(RadixSortParams), 2);
            enc->dispatchThreadgroups(MTL::Size::Make(num_reduce_wgs, 1, 1), tg);
            enc->memoryBarrier(MTL::BarrierScopeBuffers);

            // 5) scatter — a local stable sort (2-bit split ×2) + the global scatter
            enc->setComputePipelineState(pso_scatter);
            enc->setBuffer(cur_keys, 0, 0);
            enc->setBuffer(cur_vals, 0, 1);
            enc->setBuffer(counts.get(), 0, 2);
            enc->setBuffer(dst_keys, 0, 3);
            enc->setBuffer(dst_vals, 0, 4);
            enc->setBytes(&P, sizeof(RadixSortParams), 5);
            enc->dispatchThreadgroups(MTL::Size::Make(num_wgs, 1, 1), tg);
            if (pass + 1 < num_passes) enc->memoryBarrier(MTL::BarrierScopeBuffers);

            std::swap(cur_keys, dst_keys); // double-buffer swap — the next pass's input is this pass's output
            std::swap(cur_vals, dst_vals);
        }
        enc->endEncoding();
        commit_and_wait(cmd, "radix_sort");

        std::memcpy(out.keys_sorted.data(), cur_keys->contents(), n * sizeof(uint64_t));
        std::memcpy(out.values_sorted.data(), cur_vals->contents(), n * sizeof(int32_t));
        return out;
    }

    // Spec 03 §Kernel 4 isect_offset_encode — the per-tile start offsets
    // offsets[I*tile_height*tile_width] derived from the sorted isect_ids (u64)
    // (IntersectTile.cu:925–988, launch at :990–1026).
    // At n_isects==0 it zero-fills without dispatching (matching :1004–1008 offsets.fill_(0)). Only
    // boundary threads write, the written ranges do not overlap, and every range is written exactly
    // once (spec 03 second verification) — so neither a zero-fill nor atomics are needed.
    std::vector<int32_t> isect_offset_encode(const IsectOffsetInputs& in) override {
        validate_isect_offset_inputs(in, "isect_offset_encode");
        const int64_t n_tiles = static_cast<int64_t>(in.tile_width) * in.tile_height;
        std::vector<int32_t> offsets(static_cast<std::size_t>(in.I * n_tiles), 0);
        if (in.n_isects == 0 || offsets.empty()) return offsets; // host zero-fill / I==0

        IsectOffsetParams P;
        P.n_isects = static_cast<uint>(in.n_isects);
        P.I = static_cast<uint>(in.I);
        P.n_tiles = static_cast<uint>(n_tiles);
        P.tile_n_bits = bits_for_count(n_tiles); // the same single source of truth as the isect_tiles encoding (spec 03 §1)

        AutoreleasePoolGuard pool;
        MTL::ComputePipelineState* pso = pipeline("isect_offset_encode");
        auto ids = upload_buffer(in.isect_ids,
                                 static_cast<std::size_t>(in.n_isects) * sizeof(uint64_t));
        auto off_buf = upload_buffer(nullptr, offsets.size() * sizeof(int32_t));

        MTL::CommandBuffer* cmd = queue_->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(ids.get(), 0, 0);
        enc->setBuffer(off_buf.get(), 0, 1);
        enc->setBytes(&P, sizeof(IsectOffsetParams), 2);
        // Spec 03: dispatchThreads(n_isects), threadgroup 256 (IntersectTile.cu:1001–1002)
        enc->dispatchThreads(MTL::Size::Make(static_cast<NS::UInteger>(in.n_isects), 1, 1),
                             MTL::Size::Make(256, 1, 1));
        enc->endEncoding();
        commit_and_wait(cmd, "isect_offset_encode");

        std::memcpy(offsets.data(), off_buf->contents(), offsets.size() * sizeof(int32_t));
        return offsets;
    }

    // Spec 05 rasterize forward — 1 threadgroup per tile (256 threads = 256 pixels), grid
    // I*tile_h*tile_w (dispatchThreadgroups — the tile grid is integral, so non-uniform is not
    // needed).
    // function constants: kRastHasBackground / kRastHasMasks (nullptr → the false variant, where the
    // argument itself is removed — D8-5). The kernel writes an output for every in-bounds pixel
    // (including the background of empty tiles, Fwd.cu:274-296), so the outputs need no
    // zero-initialization, and the dispatch MUST NOT be skipped even at n_isects==0 — the D8-6 skip
    // applies only when the pixel or tile count is 0. There are no atomics → bit-exact determinism
    // (RasterTests memcmp).
    RasterizeFwdOutputs rasterize_fwd(const RasterizeFwdInputs& in) override {
        validate_rasterize_fwd_inputs(in, "rasterize_fwd");
        const std::size_t I = static_cast<std::size_t>(in.I);
        const std::size_t IN = I * static_cast<std::size_t>(in.N);
        const std::size_t HW = static_cast<std::size_t>(in.image_height) * in.image_width;
        const std::size_t n_tiles = static_cast<std::size_t>(in.tile_width) * in.tile_height;
        RasterizeFwdOutputs out;
        out.render_colors.resize(I * HW * kRastCdim);
        out.render_alphas.resize(I * HW);
        out.last_ids.resize(I * HW);
        const bool has_depth = in.depths != nullptr;
        if (has_depth) out.render_depths.resize(I * HW);
        if (I == 0) return out; // D8-6 dispatch skip (the validator guarantees HW/n_tiles > 0)

        const bool has_bg = in.backgrounds != nullptr;
        const bool has_masks = in.masks != nullptr;

        RasterizeParams P;
        P.n_isects = static_cast<uint>(in.n_isects);
        P.image_width = in.image_width;
        P.image_height = in.image_height;
        P.I = static_cast<uint>(in.I);
        P.tile_width = in.tile_width;
        P.tile_height = in.tile_height;

        AutoreleasePoolGuard pool;
        MTL::ComputePipelineState* pso =
            pipeline("rasterize_to_pixels_3dgs_fwd", {{kFCRastHasBackground, has_bg},
                                                      {kFCRastHasMasks, has_masks},
                                                      {kFCRastHasDepth, has_depth}});

        // An empty tensor (N==0 / n_isects==0) cannot use a length-0 buffer (newBuffer returns
        // nullptr) — substitute a minimal dummy the kernel never reads (the range is empty, so
        // flatten_ids and the gaussian tensors are never accessed at all).
        auto up = [&](const void* p, std::size_t bytes) {
            return bytes ? upload_buffer(p, bytes) : upload_buffer(nullptr, 4);
        };
        auto means2d = up(in.means2d, IN * 2 * sizeof(float));
        auto conics = up(in.conics, IN * 3 * sizeof(float));
        auto colors = up(in.colors, IN * kRastCdim * sizeof(float));
        auto opacities = up(in.opacities, IN * sizeof(float));
        NS::SharedPtr<MTL::Buffer> backgrounds, masks;
        if (has_bg) backgrounds = upload_buffer(in.backgrounds, I * kRastCdim * sizeof(float));
        if (has_masks) masks = upload_buffer(in.masks, I * n_tiles);
        auto isect_offsets = upload_buffer(in.isect_offsets, I * n_tiles * sizeof(int32_t));
        auto flatten_ids =
            up(in.flatten_ids, static_cast<std::size_t>(in.n_isects) * sizeof(int32_t));
        auto render_colors = upload_buffer(nullptr, out.render_colors.size() * sizeof(float));
        auto render_alphas = upload_buffer(nullptr, out.render_alphas.size() * sizeof(float));
        auto last_ids = upload_buffer(nullptr, out.last_ids.size() * sizeof(int32_t));
        NS::SharedPtr<MTL::Buffer> depths, render_depths;
        if (has_depth) {
            depths = up(in.depths, IN * sizeof(float));
            render_depths = upload_buffer(nullptr, out.render_depths.size() * sizeof(float));
        }

        MTL::CommandBuffer* cmd = queue_->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(means2d.get(), 0, 0);
        enc->setBuffer(conics.get(), 0, 1);
        enc->setBuffer(colors.get(), 0, 2);
        enc->setBuffer(opacities.get(), 0, 3);
        if (has_bg) enc->setBuffer(backgrounds.get(), 0, 4);
        if (has_masks) enc->setBuffer(masks.get(), 0, 5);
        enc->setBuffer(isect_offsets.get(), 0, 6);
        enc->setBuffer(flatten_ids.get(), 0, 7);
        enc->setBuffer(render_colors.get(), 0, 8);
        enc->setBuffer(render_alphas.get(), 0, 9);
        enc->setBuffer(last_ids.get(), 0, 10);
        enc->setBytes(&P, sizeof(RasterizeParams), 11);
        if (has_depth) {
            enc->setBuffer(depths.get(), 0, 12);
            enc->setBuffer(render_depths.get(), 0, 13);
        }
        // Spec 05: 1 threadgroup per tile — grid {I*tile_h*tile_w, 1, 1}, threads {256, 1, 1}
        enc->dispatchThreadgroups(MTL::Size::Make(I * n_tiles, 1, 1),
                                  MTL::Size::Make(kRastCtaSize, 1, 1));
        enc->endEncoding();
        commit_and_wait(cmd, "rasterize_to_pixels_3dgs_fwd");

        std::memcpy(out.render_colors.data(), render_colors->contents(),
                    out.render_colors.size() * sizeof(float));
        std::memcpy(out.render_alphas.data(), render_alphas->contents(),
                    out.render_alphas.size() * sizeof(float));
        std::memcpy(out.last_ids.data(), last_ids->contents(),
                    out.last_ids.size() * sizeof(int32_t));
        if (has_depth)
            std::memcpy(out.render_depths.data(), render_depths->contents(),
                        out.render_depths.size() * sizeof(float));
        return out;
    }

    // Spec 05 rasterize backward — 1 threadgroup per tile, **1 thread per pixel, 2D {16,16,1}**
    // (launcher Bwd.cu:361 — different from forward's 1D {256,1,1}). It walks back-to-front, reduces
    // across the simdgroup, and does an atomicAdd from lane 0 — the atomic order is
    // non-deterministic, so there is no bit reproduction (verification uses the meta tolerances plus
    // two-run self-consistency rel_l2, 07 §4).
    // function constants: kRastHasBackground/kRastHasMasks (shared with fwd) + kRastHasAbsGrad (the
    // v_means2d_abs output — matching CUDA's nullptr gate, D8-5). Zero-initialization of the gradient
    // buffers is **built into** the same command buffer via encode_zero_fill (D8-1 — mandatory
    // because the kernel only accumulates).
    // At n_isects==0 the dispatch is skipped (launcher Bwd.cu:365-369) — it returns 0 under
    // zeros_like semantics.
    RasterizeBwdGrads rasterize_bwd(const RasterizeFwdInputs& in,
                                    const RasterizeBwdGradsIn& g) override {
        validate_rasterize_bwd_inputs(in, g, "rasterize_bwd");
        const std::size_t I = static_cast<std::size_t>(in.I);
        const std::size_t IN = I * static_cast<std::size_t>(in.N);
        const std::size_t HW = static_cast<std::size_t>(in.image_height) * in.image_width;
        const std::size_t n_tiles = static_cast<std::size_t>(in.tile_width) * in.tile_height;
        RasterizeBwdGrads out;
        out.v_means2d.assign(IN * 2, 0.f);
        out.v_conics.assign(IN * 3, 0.f);
        out.v_colors.assign(IN * kRastCdim, 0.f);
        out.v_opacities.assign(IN, 0.f);
        if (g.absgrad) out.v_means2d_abs.assign(IN * 2, 0.f);
        const bool has_depth = in.depths != nullptr;
        if (has_depth) out.v_depths.assign(IN, 0.f);
        if (I == 0 || in.n_isects == 0) return out; // dispatch skip (Bwd.cu:365-369 / D8-6)

        const bool has_bg = in.backgrounds != nullptr;
        const bool has_masks = in.masks != nullptr;

        RasterizeParams P;
        P.n_isects = static_cast<uint>(in.n_isects);
        P.image_width = in.image_width;
        P.image_height = in.image_height;
        P.I = static_cast<uint>(in.I);
        P.tile_width = in.tile_width;
        P.tile_height = in.tile_height;

        AutoreleasePoolGuard pool;
        MTL::ComputePipelineState* pso =
            pipeline("rasterize_to_pixels_3dgs_bwd", {{kFCRastHasBackground, has_bg},
                                                      {kFCRastHasMasks, has_masks},
                                                      {kFCRastHasAbsGrad, g.absgrad},
                                                      {kFCRastHasDepth, has_depth}});

        // The shared inputs (buffers 0-7) — uploaded identically to rasterize_fwd. n_isects>0 is
        // guaranteed at this point, so the gaussian tensors are non-empty (the validator + the
        // flatten_ids reference contract).
        auto means2d = upload_buffer(in.means2d, IN * 2 * sizeof(float));
        auto conics = upload_buffer(in.conics, IN * 3 * sizeof(float));
        auto colors = upload_buffer(in.colors, IN * kRastCdim * sizeof(float));
        auto opacities = upload_buffer(in.opacities, IN * sizeof(float));
        NS::SharedPtr<MTL::Buffer> backgrounds, masks;
        if (has_bg) backgrounds = upload_buffer(in.backgrounds, I * kRastCdim * sizeof(float));
        if (has_masks) masks = upload_buffer(in.masks, I * n_tiles);
        auto isect_offsets = upload_buffer(in.isect_offsets, I * n_tiles * sizeof(int32_t));
        auto flatten_ids = upload_buffer(in.flatten_ids,
                                         static_cast<std::size_t>(in.n_isects) * sizeof(int32_t));
        // The fwd outputs + the upstream gradients (buffers 8-11)
        auto render_alphas = upload_buffer(g.render_alphas, I * HW * sizeof(float));
        auto last_ids = upload_buffer(g.last_ids, I * HW * sizeof(int32_t));
        auto v_render_colors =
            upload_buffer(g.v_render_colors, I * HW * kRastCdim * sizeof(float));
        auto v_render_alphas = upload_buffer(g.v_render_alphas, I * HW * sizeof(float));
        // The gradient outputs (buffers 12-16) — allocated uninitialized, then encode_zero_fill (D8-1)
        auto v_means2d = upload_buffer(nullptr, out.v_means2d.size() * sizeof(float));
        auto v_conics = upload_buffer(nullptr, out.v_conics.size() * sizeof(float));
        auto v_colors = upload_buffer(nullptr, out.v_colors.size() * sizeof(float));
        auto v_opacities = upload_buffer(nullptr, out.v_opacities.size() * sizeof(float));
        NS::SharedPtr<MTL::Buffer> v_means2d_abs;
        if (g.absgrad)
            v_means2d_abs = upload_buffer(nullptr, out.v_means2d_abs.size() * sizeof(float));
        NS::SharedPtr<MTL::Buffer> depths, v_render_depths, v_depths;
        if (has_depth) {
            depths = upload_buffer(in.depths, IN * sizeof(float));
            v_render_depths = upload_buffer(g.v_render_depths, I * HW * sizeof(float));
            v_depths = upload_buffer(nullptr, out.v_depths.size() * sizeof(float));
        }

        MTL::CommandBuffer* cmd = queue_->commandBuffer();
        // D8-1: the gradient accumulation buffers MUST be zero-filled through encode_zero_fill
        // (encoding order blit → compute; unallocated (nullptr) entries are skipped by the helper).
        MTL::Buffer* const grad_bufs[] = {v_means2d.get(),   v_conics.get(),
                                          v_colors.get(),    v_opacities.get(),
                                          v_means2d_abs.get(), v_depths.get()};
        encode_zero_fill(cmd, grad_bufs, 6);

        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(means2d.get(), 0, 0);
        enc->setBuffer(conics.get(), 0, 1);
        enc->setBuffer(colors.get(), 0, 2);
        enc->setBuffer(opacities.get(), 0, 3);
        if (has_bg) enc->setBuffer(backgrounds.get(), 0, 4);
        if (has_masks) enc->setBuffer(masks.get(), 0, 5);
        enc->setBuffer(isect_offsets.get(), 0, 6);
        enc->setBuffer(flatten_ids.get(), 0, 7);
        enc->setBuffer(render_alphas.get(), 0, 8);
        enc->setBuffer(last_ids.get(), 0, 9);
        enc->setBuffer(v_render_colors.get(), 0, 10);
        enc->setBuffer(v_render_alphas.get(), 0, 11);
        enc->setBuffer(v_means2d.get(), 0, 12);
        enc->setBuffer(v_conics.get(), 0, 13);
        enc->setBuffer(v_colors.get(), 0, 14);
        enc->setBuffer(v_opacities.get(), 0, 15);
        if (g.absgrad) enc->setBuffer(v_means2d_abs.get(), 0, 16);
        enc->setBytes(&P, sizeof(RasterizeParams), 17);
        if (has_depth) {
            enc->setBuffer(depths.get(), 0, 18);
            enc->setBuffer(v_render_depths.get(), 0, 19);
            enc->setBuffer(v_depths.get(), 0, 20);
        }
        // Spec 05: grid {I*tile_h*tile_w, 1, 1} × threads {tile_size, tile_size, 1}, 2D
        enc->dispatchThreadgroups(MTL::Size::Make(I * n_tiles, 1, 1),
                                  MTL::Size::Make(kRastTileSize, kRastTileSize, 1));
        enc->endEncoding();
        commit_and_wait(cmd, "rasterize_to_pixels_3dgs_bwd");

        std::memcpy(out.v_means2d.data(), v_means2d->contents(),
                    out.v_means2d.size() * sizeof(float));
        std::memcpy(out.v_conics.data(), v_conics->contents(),
                    out.v_conics.size() * sizeof(float));
        std::memcpy(out.v_colors.data(), v_colors->contents(),
                    out.v_colors.size() * sizeof(float));
        std::memcpy(out.v_opacities.data(), v_opacities->contents(),
                    out.v_opacities.size() * sizeof(float));
        if (g.absgrad)
            std::memcpy(out.v_means2d_abs.data(), v_means2d_abs->contents(),
                        out.v_means2d_abs.size() * sizeof(float));
        if (has_depth)
            std::memcpy(out.v_depths.data(), v_depths->contents(),
                        out.v_depths.size() * sizeof(float));
        return out;
    }

    // Spec 06 fused_adam — in-place (it updates param/exp_avg/exp_avg_sq and reads them back to the
    // host). Two pipelines from the kAdamHasValidMask function constant (valid nullptr → the false
    // variant, where the argument itself is removed, so no dummy bind is needed — D8-5).
    void fused_adam(const AdamInputs& in) override {
        if (in.n_elems == 0) return; // D8-6 dispatch skip
        validate_adam_inputs(in, "fused_adam");

        const bool has_valid = in.valid != nullptr;
        const std::size_t bytes = static_cast<std::size_t>(in.n_elems) * sizeof(float);

        AdamParams P;
        P.N = static_cast<uint>(in.n_gauss);
        P.D = static_cast<uint>(in.n_elems / in.n_gauss);
        P.lr = in.lr; P.b1 = in.b1; P.b2 = in.b2; P.eps = in.eps;

        AutoreleasePoolGuard pool;
        MTL::ComputePipelineState* pso = pipeline("fused_adam", {{kFCAdamHasValidMask, has_valid}});

        auto param      = upload_buffer(in.param, bytes);
        auto grad       = upload_buffer(in.grad, bytes);
        auto exp_avg    = upload_buffer(in.exp_avg, bytes);
        auto exp_avg_sq = upload_buffer(in.exp_avg_sq, bytes);
        NS::SharedPtr<MTL::Buffer> valid;
        if (has_valid) valid = upload_buffer(in.valid, static_cast<std::size_t>(in.n_gauss));

        MTL::CommandBuffer* cmd = queue_->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(param.get(), 0, 0);
        enc->setBuffer(grad.get(), 0, 1);
        enc->setBuffer(exp_avg.get(), 0, 2);
        enc->setBuffer(exp_avg_sq.get(), 0, 3);
        if (has_valid) enc->setBuffer(valid.get(), 0, 4);
        enc->setBytes(&P, sizeof(AdamParams), 5);
        // Spec 06: dispatchThreads(N*D), threadgroup 256 (it presupposes non-uniform threadgroup support).
        enc->dispatchThreads(MTL::Size::Make(static_cast<NS::UInteger>(in.n_elems), 1, 1),
                             MTL::Size::Make(256, 1, 1));
        enc->endEncoding();
        commit_and_wait(cmd, "fused_adam");

        std::memcpy(in.param, param->contents(), bytes);
        std::memcpy(in.exp_avg, exp_avg->contents(), bytes);
        std::memcpy(in.exp_avg_sq, exp_avg_sq->contents(), bytes);
    }

    // Spec 06 rng_normal_f32 — Philox4x32-10 → Box-Muller (the single source of truth is Rng.metal
    // and RngPhilox.h). Four elements per thread — dispatchThreads(ceil(count/4), tg 256). Stateless
    // and counter-based.
    std::vector<float> rng_normal_f32(const RngNormalInputs& in) override {
        validate_rng_inputs(in, "rng_normal_f32");
        std::vector<float> out(static_cast<std::size_t>(in.count));
        if (in.count == 0) return out; // D8-6 dispatch skip
        run_rng("rng_normal_f32", in, out.data(), out.size() * sizeof(float));
        return out;
    }

    // Exposes the philox integer stage — the verification surface for the bit-identical CPU↔Metal
    // contract (McmcTests only).
    std::vector<uint32_t> rng_philox_u32(const RngNormalInputs& in) override {
        validate_rng_inputs(in, "rng_philox_u32");
        std::vector<uint32_t> out(static_cast<std::size_t>(in.count));
        if (in.count == 0) return out; // D8-6 dispatch skip
        run_rng("rng_philox_u32", in, out.data(), out.size() * sizeof(uint32_t));
        return out;
    }

    // Spec 06 mcmc_relocation — one thread = one Gaussian, a pure map (no atomics, no barriers).
    // The binoms LUT is a constant buffer (10404B @ n_max=51). The ratios range (1..n_max) is the
    // host clamp contract — Metal does not validate, exactly as the upstream original does not (the
    // CPU reference promotes that validation to a throw).
    RelocationOutputs mcmc_relocation(const RelocationInputs& in) override {
        validate_relocation_inputs(in, "mcmc_relocation");
        const std::size_t N = static_cast<std::size_t>(in.N);
        RelocationOutputs out;
        out.new_opacities.resize(N);
        out.new_scales.resize(N * 3);
        if (N == 0) return out; // D8-6 dispatch skip (RelocationCUDA.cu:101–105)

        RelocParams P;
        P.N = static_cast<uint>(N);
        P.n_max = in.n_max;
        P.min_opacity = in.min_opacity;

        AutoreleasePoolGuard pool;
        MTL::ComputePipelineState* pso = pipeline("mcmc_relocation");

        const std::size_t lut_count =
            static_cast<std::size_t>(in.n_max) * static_cast<std::size_t>(in.n_max);
        auto opacities = upload_buffer(in.opacities, N * sizeof(float));
        auto scales = upload_buffer(in.scales, N * 3 * sizeof(float));
        auto ratios = upload_buffer(in.ratios, N * sizeof(int32_t));
        auto binoms = upload_buffer(in.binoms, lut_count * sizeof(float));
        auto new_opacities = upload_buffer(nullptr, N * sizeof(float));
        auto new_scales = upload_buffer(nullptr, N * 3 * sizeof(float));

        MTL::CommandBuffer* cmd = queue_->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(opacities.get(), 0, 0);
        enc->setBuffer(scales.get(), 0, 1);
        enc->setBuffer(ratios.get(), 0, 2);
        enc->setBuffer(binoms.get(), 0, 3);
        enc->setBytes(&P, sizeof(RelocParams), 4);
        enc->setBuffer(new_opacities.get(), 0, 5);
        enc->setBuffer(new_scales.get(), 0, 6);
        // Spec 06: dispatchThreads(N), threadgroup 256 (RelocationCUDA.cu:97–98)
        enc->dispatchThreads(MTL::Size::Make(N, 1, 1), MTL::Size::Make(256, 1, 1));
        enc->endEncoding();
        commit_and_wait(cmd, "mcmc_relocation");

        std::memcpy(out.new_opacities.data(), new_opacities->contents(), N * sizeof(float));
        std::memcpy(out.new_scales.data(), new_scales->contents(), N * 3 * sizeof(float));
        return out;
    }

    // Spec 06 mcmc_perturb_positions — positions in place (each thread touches only its own element,
    // so there is no hazard). The noise arrives as an input buffer — the kernel is fully
    // deterministic (bit-identical across two runs is the contract, McmcTests memcmp).
    void mcmc_perturb_positions(const PerturbInputs& in) override {
        if (in.N == 0) return; // D8-6 dispatch skip (MCMCPerturbCUDA.cu:74–78)
        validate_perturb_inputs(in, "mcmc_perturb_positions");
        const std::size_t N = static_cast<std::size_t>(in.N);

        PerturbParams P;
        P.N = static_cast<uint>(N);
        P.noise_scale = in.noise_scale;
        P.t = in.t;
        P.k = in.k;

        AutoreleasePoolGuard pool;
        MTL::ComputePipelineState* pso = pipeline("mcmc_perturb_positions");

        auto positions = upload_buffer(in.positions, N * 3 * sizeof(float));
        auto quats = upload_buffer(in.quats, N * 4 * sizeof(float));
        auto scales_log = upload_buffer(in.scales_log, N * 3 * sizeof(float));
        auto opacities_logit = upload_buffer(in.opacities_logit, N * sizeof(float));
        auto noise = upload_buffer(in.noise, N * 3 * sizeof(float));

        MTL::CommandBuffer* cmd = queue_->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(positions.get(), 0, 0);
        enc->setBuffer(quats.get(), 0, 1);
        enc->setBuffer(scales_log.get(), 0, 2);
        enc->setBuffer(opacities_logit.get(), 0, 3);
        enc->setBuffer(noise.get(), 0, 4);
        enc->setBytes(&P, sizeof(PerturbParams), 5);
        // Spec 06: dispatchThreads(N), threadgroup 256 (MCMCPerturbCUDA.cu:82–83)
        enc->dispatchThreads(MTL::Size::Make(N, 1, 1), MTL::Size::Make(256, 1, 1));
        enc->endEncoding();
        commit_and_wait(cmd, "mcmc_perturb_positions");

        std::memcpy(in.positions, positions->contents(), N * 3 * sizeof(float));
    }

    // Spec 08 ssim forward — threadgroup (16,16) = one thread per map pixel, grid
    // (ceil(map_w/16), ceil(map_h/16), I·C). A separable 11-tap convolution in 2 passes + a 26×26
    // apron threadgroup cache (the apron of a partial tile is boundary-masked — the header comment of
    // Ssim.metal / 08 §5 Correction). There are no atomics → bit-exact determinism (a two-run memcmp
    // gate). Every map element is written exactly once, so no zero-fill is needed. The ssim scalar is
    // a **host f64 sum in index order** (the single source of truth is ssim_reduce_mean_f64 — the
    // 08 §5 baseline; device atomic float is prohibited; a partial_sums kernel reduction is to be
    // revisited in device-resident Task 5 — 08 open question 1). kSsimNeedCoeffs specialization: at
    // false the 3 coefficient map arguments are removed entirely — no binding needed (D8-5).
    SsimFwdOutputs ssim_fwd(const SsimInputs& in) override {
        validate_ssim_inputs(in, "ssim_fwd");
        const std::size_t H = static_cast<std::size_t>(in.H);
        const std::size_t W = static_cast<std::size_t>(in.W);
        const std::size_t C = static_cast<std::size_t>(in.C);
        const std::size_t map_h = H - (kSsimWindowSize - 1);
        const std::size_t map_w = W - (kSsimWindowSize - 1);
        const std::size_t in_count = static_cast<std::size_t>(in.I) * H * W * C;
        const std::size_t map_count = static_cast<std::size_t>(in.I) * map_h * map_w * C;

        SsimFwdOutputs out;
        out.ssim_map.resize(map_count);
        if (in.need_coeffs) {
            out.coeff_mu.resize(map_count);
            out.coeff_sigma.resize(map_count);
            out.coeff_cross.resize(map_count);
        }

        SsimParams P;
        P.I = static_cast<uint>(in.I);
        P.H = static_cast<uint>(H);
        P.W = static_cast<uint>(W);
        P.C = static_cast<uint>(C);
        P.map_h = static_cast<uint>(map_h);
        P.map_w = static_cast<uint>(map_w);
        P.G = 0.f; // unused in fwd

        AutoreleasePoolGuard pool;
        MTL::ComputePipelineState* pso =
            pipeline("ssim_fwd", {{kFCSsimNeedCoeffs, in.need_coeffs}});

        auto img1 = upload_buffer(in.img1, in_count * sizeof(float));
        auto img2 = upload_buffer(in.img2, in_count * sizeof(float));
        auto ssim_map = upload_buffer(nullptr, map_count * sizeof(float));
        NS::SharedPtr<MTL::Buffer> c_mu, c_sigma, c_cross;
        if (in.need_coeffs) {
            c_mu = upload_buffer(nullptr, map_count * sizeof(float));
            c_sigma = upload_buffer(nullptr, map_count * sizeof(float));
            c_cross = upload_buffer(nullptr, map_count * sizeof(float));
        }

        MTL::CommandBuffer* cmd = queue_->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(img1.get(), 0, 0);
        enc->setBuffer(img2.get(), 0, 1);
        enc->setBuffer(ssim_map.get(), 0, 2);
        if (in.need_coeffs) {
            enc->setBuffer(c_mu.get(), 0, 3);
            enc->setBuffer(c_sigma.get(), 0, 4);
            enc->setBuffer(c_cross.get(), 0, 5);
        }
        enc->setBytes(&P, sizeof(SsimParams), 6);
        // Spec 08 §5: dispatchThreadgroups — an integral grid of map tiles, threadgroup (16,16,1)
        enc->dispatchThreadgroups(
            MTL::Size::Make((map_w + kSsimTileSize - 1) / kSsimTileSize,
                            (map_h + kSsimTileSize - 1) / kSsimTileSize,
                            static_cast<NS::UInteger>(in.I * in.C)),
            MTL::Size::Make(kSsimTileSize, kSsimTileSize, 1));
        enc->endEncoding();
        commit_and_wait(cmd, "ssim_fwd");

        std::memcpy(out.ssim_map.data(), ssim_map->contents(), map_count * sizeof(float));
        if (in.need_coeffs) {
            std::memcpy(out.coeff_mu.data(), c_mu->contents(), map_count * sizeof(float));
            std::memcpy(out.coeff_sigma.data(), c_sigma->contents(), map_count * sizeof(float));
            std::memcpy(out.coeff_cross.data(), c_cross->contents(), map_count * sizeof(float));
        }
        out.ssim = ssim_reduce_mean_f64(out.ssim_map.data(), map_count);
        return out;
    }

    // Spec 08 ssim backward — one thread per input pixel, grid (ceil(W/16), ceil(H/16), I·C).
    // It loads the 3 coefficient maps (the fwd need_coeffs outputs) into a zero-padded apron, does a
    // scatter convolution, and combines pointwise (08 §4-(3)). G = -v_loss/M is computed on the host
    // in f64 and cast to f32 (the λ combination is the Trainer's responsibility — Task 3; the golden
    // fixtures use v_loss=1). Every pixel of v_img1 is written → no zero-fill and no atomics are
    // needed, and it is bit-exact deterministic (a two-run memcmp gate).
    std::vector<float> ssim_bwd(const SsimInputs& in, const SsimGradsIn& g) override {
        validate_ssim_inputs(in, "ssim_bwd");
        validate_ssim_grads_in(g, "ssim_bwd");
        const std::size_t H = static_cast<std::size_t>(in.H);
        const std::size_t W = static_cast<std::size_t>(in.W);
        const std::size_t C = static_cast<std::size_t>(in.C);
        const std::size_t map_h = H - (kSsimWindowSize - 1);
        const std::size_t map_w = W - (kSsimWindowSize - 1);
        const std::size_t in_count = static_cast<std::size_t>(in.I) * H * W * C;
        const std::size_t map_count = static_cast<std::size_t>(in.I) * map_h * map_w * C;

        SsimParams P;
        P.I = static_cast<uint>(in.I);
        P.H = static_cast<uint>(H);
        P.W = static_cast<uint>(W);
        P.C = static_cast<uint>(C);
        P.map_h = static_cast<uint>(map_h);
        P.map_w = static_cast<uint>(map_w);
        P.G = static_cast<float>(-static_cast<double>(g.v_loss) /
                                 static_cast<double>(map_count)); // -v_loss/M (f64 → f32)

        AutoreleasePoolGuard pool;
        MTL::ComputePipelineState* pso = pipeline("ssim_bwd");

        auto img1 = upload_buffer(in.img1, in_count * sizeof(float));
        auto img2 = upload_buffer(in.img2, in_count * sizeof(float));
        auto c_mu = upload_buffer(g.coeff_mu, map_count * sizeof(float));
        auto c_sigma = upload_buffer(g.coeff_sigma, map_count * sizeof(float));
        auto c_cross = upload_buffer(g.coeff_cross, map_count * sizeof(float));
        auto v_img1 = upload_buffer(nullptr, in_count * sizeof(float));

        MTL::CommandBuffer* cmd = queue_->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(img1.get(), 0, 0);
        enc->setBuffer(img2.get(), 0, 1);
        enc->setBuffer(c_mu.get(), 0, 2);
        enc->setBuffer(c_sigma.get(), 0, 3);
        enc->setBuffer(c_cross.get(), 0, 4);
        enc->setBuffer(v_img1.get(), 0, 5);
        enc->setBytes(&P, sizeof(SsimParams), 6);
        // Spec 08 §5: dispatchThreadgroups — an integral grid of input pixel tiles, threadgroup (16,16,1)
        enc->dispatchThreadgroups(
            MTL::Size::Make((W + kSsimTileSize - 1) / kSsimTileSize,
                            (H + kSsimTileSize - 1) / kSsimTileSize,
                            static_cast<NS::UInteger>(in.I * in.C)),
            MTL::Size::Make(kSsimTileSize, kSsimTileSize, 1));
        enc->endEncoding();
        commit_and_wait(cmd, "ssim_bwd");

        std::vector<float> out(in_count);
        std::memcpy(out.data(), v_img1->contents(), in_count * sizeof(float));
        return out;
    }

    // The D11 resident step executor factory (Phase 3 Task 5) — defined below MetalTrainStepExecutor.
    std::unique_ptr<TrainStepExecutor>
    make_train_step_executor(const TrainStepExecutorDesc& d) override;

private:
    // The metallib load shared by layers (a) and (b) — newLibrary(URL:). On success it sets
    // library_/name_; on failure it accumulates the reason into diag and returns false. What happens
    // after a failure is the caller's responsibility — (a) an explicit path throws immediately, while
    // (b) auto-discovery warns on stderr and moves to the next layer.
    bool try_load_metallib(const std::string& path, const char* layer, std::string& diag) {
        std::error_code ec;
        if (!fs::is_regular_file(path, ec)) {
            diag += "\n  [" + std::string(layer) + "] not a readable file: " + path;
            return false;
        }
        NS::Error* error = nullptr;
        NS::URL* url = NS::URL::fileURLWithPath(
            NS::String::string(path.c_str(), NS::UTF8StringEncoding));
        auto lib = NS::TransferPtr(device_->newLibrary(url, &error));
        if (!lib) {
            diag += "\n  [" + std::string(layer) + "] newLibrary(URL) failed for " + path +
                    ns_error_suffix(error);
            return false;
        }
        library_ = lib;
        name_ = "metal(metallib)";
        return true;
    }

    // The execution shared by rng_normal_f32 and rng_philox_u32 — only the output element type
    // differs (f32 and u32 are both 4B); the dispatch and the parameters are identical. It assumes
    // count > 0 (the caller applies the D8-6 skip).
    void run_rng(const char* fn_name, const RngNormalInputs& in, void* dst,
                 std::size_t bytes) {
        RngParams P;
        P.seed_lo = static_cast<uint>(in.seed & 0xFFFFFFFFu);
        P.seed_hi = static_cast<uint>(in.seed >> 32);
        P.offset_lo = static_cast<uint>(in.offset & 0xFFFFFFFFu);
        P.offset_hi = static_cast<uint>(in.offset >> 32);
        P.count = static_cast<uint>(in.count);

        AutoreleasePoolGuard pool;
        MTL::ComputePipelineState* pso = pipeline(fn_name);
        auto out_buf = upload_buffer(nullptr, bytes);

        MTL::CommandBuffer* cmd = queue_->commandBuffer();
        MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
        enc->setComputePipelineState(pso);
        enc->setBuffer(out_buf.get(), 0, 0);
        enc->setBytes(&P, sizeof(RngParams), 1);
        // Spec 06: dispatchThreads(ceil(count/4)), threadgroup 256 — 4 elements per thread
        const NS::UInteger n_threads = (static_cast<NS::UInteger>(in.count) + 3) / 4;
        enc->dispatchThreads(MTL::Size::Make(n_threads, 1, 1), MTL::Size::Make(256, 1, 1));
        enc->endEncoding();
        commit_and_wait(cmd, fn_name);

        std::memcpy(dst, out_buf->contents(), bytes);
    }

    // Spec 03 §Kernel 2 reduce-then-scan, encoded as 3 dispatches (PrefixScan.metal):
    //   scan_partial (an in-block scan + the block total) → scan_blocks_single_group (an exclusive
    //   scan of the totals with a carry loop in a single threadgroup — no upper bound on the block
    //   count) → scan_add_offsets (the addition).
    // They are encoded sequentially into the same encoder — memoryBarrier(BarrierScopeBuffers)
    // between dispatches makes write → read visibility explicit (the barrier points of spec 03
    // §Kernel 2). The caller owns creating and ending the encoder.
    void encode_prefix_scan(MTL::ComputeCommandEncoder* enc, MTL::Buffer* in_i32,
                            MTL::Buffer* out_i64, MTL::Buffer* block_sums, uint32_t n,
                            uint32_t num_groups) {
        const MTL::Size tg = MTL::Size::Make(kScanThreadgroupSize, 1, 1);

        // A: an inclusive scan within a block (kScanBlockSize) — a full threadgroup dispatch (the tail is guarded by n)
        enc->setComputePipelineState(pipeline("scan_partial"));
        enc->setBuffer(in_i32, 0, 0);
        enc->setBuffer(out_i64, 0, 1);
        enc->setBuffer(block_sums, 0, 2);
        enc->setBytes(&n, sizeof(uint32_t), 3);
        enc->dispatchThreadgroups(MTL::Size::Make(num_groups, 1, 1), tg);
        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // B: an exclusive scan of block_sums — a single threadgroup
        enc->setComputePipelineState(pipeline("scan_blocks_single_group"));
        enc->setBuffer(block_sums, 0, 0);
        enc->setBytes(&num_groups, sizeof(uint32_t), 1);
        enc->dispatchThreadgroups(MTL::Size::Make(1, 1, 1), tg);
        enc->memoryBarrier(MTL::BarrierScopeBuffers);

        // C: broadcasts and adds the block offsets — one thread = one element
        enc->setComputePipelineState(pipeline("scan_add_offsets"));
        enc->setBuffer(out_i64, 0, 0);
        enc->setBuffer(block_sums, 0, 1);
        enc->setBytes(&n, sizeof(uint32_t), 2);
        enc->dispatchThreads(MTL::Size::Make(n, 1, 1), tg);
    }

    // (kernel name, the set of function constant values) — the values are a list of (index, bool)
    // pairs (from the FunctionConstants.h bands).
    using FunctionConstantSet = std::vector<std::pair<uint32_t, bool>>;

    // The pipeline cache. It creates on first request and enforces threadExecutionWidth == 32 (D8-3):
    // this is the defense line for the simdgroup-width-32 assumption (specs 01/05) — a violation
    // throws immediately.
    MTL::ComputePipelineState* pipeline(const std::string& fn_name,
                                        const FunctionConstantSet& fcs = {}) {
        FunctionConstantSet sorted = fcs;
        std::sort(sorted.begin(), sorted.end());
        std::string key = fn_name;
        for (const auto& [idx, val] : sorted)
            key += "|" + std::to_string(idx) + "=" + (val ? "1" : "0");
        if (auto it = pipelines_.find(key); it != pipelines_.end()) return it->second.get();

        AutoreleasePoolGuard pool;
        NS::Error* error = nullptr;
        NS::String* nsname = NS::String::string(fn_name.c_str(), NS::UTF8StringEncoding);
        NS::SharedPtr<MTL::Function> fn;
        if (sorted.empty()) {
            fn = NS::TransferPtr(library_->newFunction(nsname));
            if (!fn) throw std::runtime_error("MetalBackend: kernel function not found: " + fn_name);
        } else {
            auto values = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
            for (const auto& [idx, val] : sorted) {
                const bool v = val;
                values->setConstantValue(&v, MTL::DataTypeBool, static_cast<NS::UInteger>(idx));
            }
            fn = NS::TransferPtr(library_->newFunction(nsname, values.get(), &error));
            if (!fn) throw_ns_error("MetalBackend: function specialization failed: " + fn_name, error);
        }
        auto pso = NS::TransferPtr(device_->newComputePipelineState(fn.get(), &error));
        if (!pso) throw_ns_error("MetalBackend: pipeline creation failed: " + fn_name, error);
        if (pso->threadExecutionWidth() != 32)
            throw std::runtime_error(
                "MetalBackend: threadExecutionWidth != 32 for " + fn_name + " (got " +
                std::to_string(pso->threadExecutionWidth()) + ") — D8-3 violation");
        MTL::ComputePipelineState* raw = pso.get();
        pipelines_.emplace(std::move(key), std::move(pso));
        return raw;
    }

    // Host memory → a StorageModeShared MTL::Buffer (the reference path for uploading SXT tensor
    // payloads — reused by Tasks 5–7). With data == nullptr it only allocates an uninitialized output
    // buffer (bwd outputs go through encode_zero_fill — D8-1). On Apple Silicon UMA, shared is the
    // minimum-copy path.
    // oom_hint: text appended to the end of an allocation failure message (it points at the
    // --no-resident escape hatch for the D11 resident buffers — Phase 3.5 Task 1; the default
    // nullptr leaves the existing message unchanged).
    NS::SharedPtr<MTL::Buffer> upload_buffer(const void* data, std::size_t bytes,
                                             const char* oom_hint = nullptr) {
        auto buf = NS::TransferPtr(device_->newBuffer(static_cast<NS::UInteger>(bytes),
                                                      MTL::ResourceStorageModeShared));
        if (!buf)
            throw std::runtime_error("MetalBackend: buffer allocation failed (" +
                                     std::to_string(bytes) + " bytes)" +
                                     (oom_hint ? oom_hint : ""));
        if (data) std::memcpy(buf->contents(), data, bytes);
        return buf;
    }

    // commit + wait for completion + propagate GPU errors (the same convention as the Phase 0
    // cb.error propagation).
    // [Step breakdown probe — taming-bwd Task 1] When SPLATX_KERNEL_TIMING is set, it writes the
    // command buffer's GPU time to stderr per label (see the kernel_timing_enabled comment).
    // On the host round-trip path there is exactly one command buffer per backend call (this helper
    // is the only commit point), so these records alone give the per-kernel-group breakdown; on the
    // resident path the whole-of-cb1/cb2 GPU time is recorded and used to cross-check the totals.
    // The default path is unchanged in meaning.
    static void commit_and_wait(MTL::CommandBuffer* cmd, const char* what) {
        cmd->commit();
        cmd->waitUntilCompleted();
        if (cmd->error())
            throw_ns_error(std::string("MetalBackend: ") + what + " command buffer failed",
                           cmd->error());
        if (kernel_timing_enabled())
            std::fprintf(stderr, "[ktime] %s %.4f\n", what,
                         (cmd->GPUEndTime() - cmd->GPUStartTime()) * 1000.0);
    }

    // D8-1: the zero-initialization helper for gradient and accumulation output buffers — every bwd
    // encoding MUST go through this helper before encoding the kernel (it guarantees the blit →
    // compute order within the same command buffer).
    // nullptr entries are skipped (optional buffers).
    static void encode_zero_fill(MTL::CommandBuffer* cmd, MTL::Buffer* const* buffers,
                                 std::size_t count) {
        if (count == 0) return;
        MTL::BlitCommandEncoder* blit = cmd->blitCommandEncoder();
        for (std::size_t i = 0; i < count; ++i) {
            if (!buffers[i]) continue;
            blit->fillBuffer(buffers[i], NS::Range::Make(0, buffers[i]->length()), 0);
        }
        blit->endEncoding();
    }

    NS::SharedPtr<MTL::Device> device_;
    NS::SharedPtr<MTL::Library> library_;
    NS::SharedPtr<MTL::CommandQueue> queue_;
    std::unordered_map<std::string, NS::SharedPtr<MTL::ComputePipelineState>> pipelines_;
    std::string name_; // records the loading layer that was selected — "metal(metallib)" / "metal(source)"
};

// ===== The D11 device-resident train step executor (Phase 3 Task 5 — 00-index §7) =====
// It keeps the 6 parameter groups + the 12 Adam moment arrays + the step intermediates resident in
// MTL buffers and encodes the D7 step into **2 command buffers** (the sync points are the single
// n_isects readback + the host loss summation):
//   CB1: activation (exp/sigmoid) → proj fwd → (the AA product) → SH prep (valid/dirs) → coeffs
//        assembly → sh fwd → the SH post-op → isect count → prefix scan  ── the n_isects readback
//   CB2: isect emit → radix sort → offset encode → raster fwd → (ssim fwd/bwd)
//        → the L1 (+ combined) gradient → [the blit zero-fill of the gradient buffers is at the head
//        of CB2 — D8-1] → raster bwd → the relu_mask product → sh bwd → (the AA chain) → proj bwd
//        → v_means += v_dirs → the activation gradient chain + reg → the v_coeffs split
//        → fused_adam ×6 → (noise: rng+perturb)
// The kernel bindings and dispatch configuration are item-for-item identical to the individual
// MetalBackend methods above (the verification surface — retained permanently by D11) — the
// cross-check gate is the host vs resident step comparison test in TrainerTests.
// Every buffer is preallocated at the arena_rows (cap_max) scale (consistent with the MCMC arena —
// no reallocation); only the isect and image-sized buffers grow and are reused based on capacity.
class MetalTrainStepExecutor final : public TrainStepExecutor {
public:
    MetalTrainStepExecutor(MetalBackend& mb, const TrainStepExecutorDesc& d)
        : mb_(mb), d_(d) {
        if (d.arena_rows < 1)
            throw std::invalid_argument("TrainStepExecutor: arena_rows must be >= 1");
        if (d.sh_degree < 0 || d.sh_degree > 3)
            throw std::invalid_argument("TrainStepExecutor: sh_degree must be in [0,3]");
        if (!(d.ssim_weight >= 0.0f && d.ssim_weight <= 1.0f))
            throw std::invalid_argument("TrainStepExecutor: ssim_weight must be in [0,1]");
        K_ = static_cast<int64_t>(d.sh_degree + 1) * (d.sh_degree + 1);
        const int64_t rows = d.arena_rows;
        if (rows * K_ * 3 > INT32_MAX)
            throw std::invalid_argument("TrainStepExecutor: arena_rows*K*3 exceeds int32");
        dims_ = {3, 4, 3, 1, 3, (K_ - 1) * 3};

        AutoreleasePoolGuard pool;
        for (int g = 0; g < 6; g++) {
            if (dims_[g] == 0) continue; // shN at sh_degree 0 — no buffer (adam is skipped)
            const std::size_t bytes = static_cast<std::size_t>(rows) * dims_[g] * sizeof(float);
            p_[g] = alloc(bytes);
            m1_[g] = alloc(bytes);
            m2_[g] = alloc(bytes);
        }
        scales_lin_ = alloc(rows * 3 * sizeof(float));
        opacs_act_ = alloc(rows * sizeof(float));
        radii_ = alloc(rows * 2 * sizeof(int32_t));
        means2d_ = alloc(rows * 2 * sizeof(float));
        depths_ = alloc(rows * sizeof(float));
        conics_ = alloc(rows * 3 * sizeof(float));
        valid_ = alloc(rows);
        dirs_ = alloc(rows * 3 * sizeof(float));
        coeffs_ = alloc(rows * K_ * 3 * sizeof(float));
        colors_ = alloc(rows * 3 * sizeof(float));
        relu_mask_ = alloc(rows * 3);
        tiles_per_gauss_ = alloc(rows * sizeof(int32_t));
        cum_ = alloc(rows * sizeof(int64_t));
        block_sums_ = alloc(((rows + kScanBlockSize - 1) / kScanBlockSize) * sizeof(int64_t));
        v_means2d_ = alloc(rows * 2 * sizeof(float));
        v_conics_ = alloc(rows * 3 * sizeof(float));
        v_colors_ = alloc(rows * 3 * sizeof(float));
        v_opacities_ = alloc(rows * sizeof(float));
        v_coeffs_ = alloc(rows * K_ * 3 * sizeof(float));
        v_dirs_ = alloc(rows * 3 * sizeof(float));
        v_means_ = alloc(rows * 3 * sizeof(float));
        v_quats_ = alloc(rows * 4 * sizeof(float));
        v_scales_ = alloc(rows * 3 * sizeof(float));
        v_sh0_ = alloc(rows * 3 * sizeof(float));
        if (dims_[5] > 0) v_shN_ = alloc(rows * dims_[5] * sizeof(float));
        v_depths_zero_ = alloc(rows * sizeof(float));
        std::memset(v_depths_zero_->contents(), 0, rows * sizeof(float)); // RGB-only steps
        v_depths_ = alloc(rows * sizeof(float));
        noise_ = alloc(rows * 3 * sizeof(float));
        if (d.antialiased) {
            comps_ = alloc(rows * sizeof(float));
            opacs_aa_ = alloc(rows * sizeof(float));
            v_comp_ = alloc(rows * sizeof(float));
        }
        viewmat_buf_ = alloc(16 * sizeof(float));
        K_buf_ = alloc(9 * sizeof(float));
        ensure_isect_capacity(1 << 16); // the minimum capacity — afterwards it grows 1.5x and is reused
    }

    // host → device full synchronization (rows rows — the first upload covers the whole arena, which
    // establishes the zero tail).
    void upload_params(const ResidentParamSync& s) override { sync_params(s, true); }
    // device → host — the partial readback for MCMC refine, eval, and export (rows = count suffices).
    void download_params(const ResidentParamSync& s) override { sync_params(s, false); }

    ResidentStepOutputs step(const ResidentStepInputs& in) override {
        // --- Entry validation (throws in the same family as the host path validators) ---
        if (in.count < 1 || in.count > d_.arena_rows)
            throw std::invalid_argument("TrainStepExecutor::step: count out of [1, arena_rows]");
        if (in.width < 1 || in.height < 1)
            throw std::invalid_argument("TrainStepExecutor::step: width/height must be positive");
        if (!in.viewmat || !in.K)
            throw std::invalid_argument("TrainStepExecutor::step: null viewmat/K");
        if (!in.forward_only && !in.gt_rgb)
            throw std::invalid_argument("TrainStepExecutor::step: null gt_rgb");
        const bool use_ssim = d_.ssim_weight > 0.0f && !in.forward_only;
        if (in.gt_depth && (in.depth_width < 1 || in.depth_height < 1))
            throw std::invalid_argument("TrainStepExecutor::step: gt_depth needs positive dims");
        if (!(in.depth_weight >= 0.0f && std::isfinite(in.depth_weight)))
            throw std::invalid_argument("TrainStepExecutor::step: depth_weight must be >= 0");
        // The depth channel runs for a supervised training step or an expected-depth render; the
        // RGB-only kernel variants are untouched otherwise.
        const bool use_depth_loss = !in.forward_only && in.gt_depth && in.depth_weight > 0.0f &&
                                    in.depth_valid > 0;
        const bool run_depth = use_depth_loss || in.want_depth;
        if (use_ssim && (in.height < static_cast<int>(kSsimWindowSize) ||
                         in.width < static_cast<int>(kSsimWindowSize)))
            throw std::invalid_argument( // matches validate_ssim_inputs (the valid map would be empty)
                "TrainStepExecutor::step: ssim requires H, W >= 11");

        const uint32_t N = static_cast<uint32_t>(in.count);
        const uint32_t W = static_cast<uint32_t>(in.width);
        const uint32_t H = static_cast<uint32_t>(in.height);
        const uint32_t tw = (W + kRastTileSize - 1) / kRastTileSize;
        const uint32_t th = (H + kRastTileSize - 1) / kRastTileSize;
        const std::size_t HW = static_cast<std::size_t>(W) * H;
        const std::size_t n_px = HW * 3;
        const bool aa = d_.antialiased;

        AutoreleasePoolGuard pool;
        ensure_image_capacity(HW, static_cast<std::size_t>(tw) * th, use_ssim);
        if (run_depth) ensure_depth_image_capacity();
        const std::size_t depth_px =
            use_depth_loss ? static_cast<std::size_t>(in.depth_width) * in.depth_height : 0;
        if (use_depth_loss) {
            ensure_depth_capacity(depth_px);
            std::memcpy(gt_depth_->contents(), in.gt_depth, depth_px * sizeof(float));
        }
        std::memcpy(viewmat_buf_->contents(), in.viewmat, 16 * sizeof(float));
        std::memcpy(K_buf_->contents(), in.K, 9 * sizeof(float));
        if (!in.forward_only)
            std::memcpy(gt_->contents(), in.gt_rgb, n_px * sizeof(float));

        MTL::Buffer* raster_opacs = aa ? opacs_aa_.get() : opacs_act_.get();

        ProjectionParams PJ;
        PJ.N = N;
        PJ.C = 1;
        PJ.image_width = W;
        PJ.image_height = H;
        PJ.eps2d = d_.eps2d;
        PJ.near_plane = d_.near_plane;
        PJ.far_plane = d_.far_plane;
        PJ.radius_clip = d_.radius_clip;
        PJ.camera_model = 0; // PINHOLE (the Phase 1 scope — the same as Trainer)
        PJ.has_opacities = 1u;
        PJ.calc_compensations = aa ? 1u : 0u;

        IsectTileParams PI;
        PI.n_elements = N;
        PI.I = 1;
        PI.N = N;
        PI.tile_size = kRastTileSize;
        PI.tile_width = tw;
        PI.tile_height = th;
        PI.tile_n_bits = bits_for_count(static_cast<int64_t>(tw) * th);
        PI.image_n_bits = bits_for_count(1);

        const MTL::Size tg256 = MTL::Size::Make(256, 1, 1);
        const uint32_t scan_groups = (N + kScanBlockSize - 1) / kScanBlockSize;

        // ---------- CB1: activation → proj fwd → SH → isect count → scan ----------
        {
            MTL::CommandBuffer* cmd = mb_.queue_->commandBuffer();
            if (aa) { // culled slots get comp = 0 (matching gsplat at::zeros — the opacity_aa = 0 gate)
                MTL::Buffer* const fills[] = {comps_.get()};
                const std::size_t bytes[] = {static_cast<std::size_t>(N) * sizeof(float)};
                fill_prefix(cmd, fills, bytes, 1);
            }
            MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();

            elt(enc, "train_exp", {p_[2].get(), scales_lin_.get()}, N * 3);
            elt(enc, "train_sigmoid", {p_[3].get(), opacs_act_.get()}, N);

            // proj fwd — the same bindings as MetalBackend::projection_fused_fwd (buffer 1, covars, is absent)
            enc->setComputePipelineState(
                mb_.pipeline("projection_ewa_3dgs_fused_fwd",
                             {{kFCProjUseCovars, false},
                              {kFCProjHasOpacities, true},
                              {kFCProjCalcComp, aa}}));
            enc->setBuffer(p_[0].get(), 0, 0);       // means
            enc->setBuffer(p_[1].get(), 0, 2);       // quats
            enc->setBuffer(scales_lin_.get(), 0, 3); // linear scales
            enc->setBuffer(opacs_act_.get(), 0, 4);  // post-sigmoid (fwd only)
            enc->setBuffer(viewmat_buf_.get(), 0, 5);
            enc->setBuffer(K_buf_.get(), 0, 6);
            enc->setBuffer(radii_.get(), 0, 7);
            enc->setBuffer(means2d_.get(), 0, 8);
            enc->setBuffer(depths_.get(), 0, 9);
            enc->setBuffer(conics_.get(), 0, 10);
            if (aa) enc->setBuffer(comps_.get(), 0, 11);
            enc->setBytes(&PJ, sizeof(ProjectionParams), 12);
            dispatch(enc, N);

            if (aa) // opacity_aa = σ(op)·comp (the AA block in Trainer.cpp — shared by isect and raster)
                elt(enc, "train_ew_mul", {opacs_act_.get(), comps_.get(), opacs_aa_.get()}, N);

            { // SH prep: valid = (radii>0).all(-1), dirs = means − campos
                TrainPrepShParams P;
                P.N = N;
                // campos = -Rᵀt — the same expression as Trainer.cpp forward_pass (f32)
                const float* vm = in.viewmat;
                P.campos_x = -(vm[0] * vm[3] + vm[4] * vm[7] + vm[8] * vm[11]);
                P.campos_y = -(vm[1] * vm[3] + vm[5] * vm[7] + vm[9] * vm[11]);
                P.campos_z = -(vm[2] * vm[3] + vm[6] * vm[7] + vm[10] * vm[11]);
                enc->setComputePipelineState(mb_.pipeline("train_prep_sh"));
                enc->setBuffer(radii_.get(), 0, 0);
                enc->setBuffer(p_[0].get(), 0, 1);
                enc->setBuffer(dirs_.get(), 0, 2);
                enc->setBuffer(valid_.get(), 0, 3);
                enc->setBytes(&P, sizeof(P), 4);
                dispatch(enc, N);
            }
            { // Assembles coeffs [N,K,3] — at K==1 shN is never accessed (sh0 is bound as a dummy)
                TrainCoeffsParams P{N, static_cast<uint32_t>(K_)};
                enc->setComputePipelineState(mb_.pipeline("train_assemble_coeffs"));
                enc->setBuffer(p_[4].get(), 0, 0);
                enc->setBuffer((dims_[5] > 0 ? p_[5] : p_[4]).get(), 0, 1);
                enc->setBuffer(coeffs_.get(), 0, 2);
                enc->setBytes(&P, sizeof(P), 3);
                dispatch(enc, static_cast<std::size_t>(N) * K_ * 3);
            }
            { // sh fwd — the same bindings as MetalBackend::sh_fwd
                SHParams P;
                P.degrees_to_use = in.degrees_to_use;
                P.B = 1;
                P.N = N;
                P.K = static_cast<uint32_t>(K_);
                P.D = 3;
                enc->setComputePipelineState(mb_.pipeline("sh_fwd", {{kFCShHasMasks, true}}));
                enc->setBuffer(dirs_.get(), 0, 0);
                enc->setBuffer(coeffs_.get(), 0, 1);
                enc->setBuffer(valid_.get(), 0, 2);
                enc->setBuffer(colors_.get(), 0, 3);
                enc->setBytes(&P, sizeof(SHParams), 4);
                dispatch(enc, static_cast<std::size_t>(N) * 3);
            }
            elt(enc, "train_sh_post_op", {colors_.get(), relu_mask_.get()}, N * 3);

            // isect count + prefix scan — identical to MetalBackend::isect_tiles pass 1
            enc->setComputePipelineState(mb_.pipeline("isect_tile_count", {{kFCIsectAccuTile, true}}));
            enc->setBuffer(means2d_.get(), 0, 0);
            enc->setBuffer(radii_.get(), 0, 1);
            enc->setBuffer(conics_.get(), 0, 2);
            enc->setBuffer(raster_opacs, 0, 3);
            enc->setBuffer(tiles_per_gauss_.get(), 0, 4);
            enc->setBytes(&PI, sizeof(IsectTileParams), 5);
            dispatch(enc, N);
            mb_.encode_prefix_scan(enc, tiles_per_gauss_.get(), cum_.get(), block_sums_.get(),
                                   N, scan_groups);
            enc->endEncoding();
            MetalBackend::commit_and_wait(cmd, "resident_step_cb1");
        }

        // ★ The D9-3 sync point: the n_isects readback (a direct UMA shared read — the only forced
        // synchronization per step)
        const int64_t n_isects =
            static_cast<const int64_t*>(cum_->contents())[static_cast<std::size_t>(N) - 1];
        // A cast guard for the uint32 parameters (PR/PO) — out of range it throws explicitly instead
        // of truncating silently (Validate.h — it writes down the contract for a range that cannot be
        // reached, Phase 3.5 Task 1).
        const uint32_t n_isects_u32 =
            validate_n_isects_readback(n_isects, "TrainStepExecutor::step");
        ensure_isect_capacity(static_cast<std::size_t>(n_isects));

        RasterizeParams PR;
        PR.n_isects = n_isects_u32;
        PR.image_width = W;
        PR.image_height = H;
        PR.I = 1;
        PR.tile_width = tw;
        PR.tile_height = th;

        const std::size_t map_h = use_ssim ? H - (kSsimWindowSize - 1) : 0;
        const std::size_t map_w = use_ssim ? W - (kSsimWindowSize - 1) : 0;
        const std::size_t map_count = use_ssim ? map_h * map_w * 3 : 0;

        // ---------- CB2: emit → sort → offset → raster → loss grad → the bwd chain → adam ----------
        int cur = 0; // the final position of the sort double buffer (the keys_/vals_ index)
        {
            MTL::CommandBuffer* cmd = mb_.queue_->commandBuffer();

            // D8-1: the gradient accumulation buffers are zero-filled by a single blit at the head of
            // CB2 (only the count-row prefix — equivalent to the exact-size buffer fill on the host
            // path). At n_isects==0 offsets is zero-filled too (isect_offset_encode is skipped —
            // matching the host fill_(0)).
            if (!in.forward_only) {
                MTL::Buffer* fills[16];
                std::size_t ranges[16];
                std::size_t nf = 0;
                auto add = [&](MTL::Buffer* b, std::size_t bytes) {
                    fills[nf] = b;
                    ranges[nf++] = bytes;
                };
                const std::size_t Ns = N;
                add(v_means2d_.get(), Ns * 2 * sizeof(float));
                add(v_conics_.get(), Ns * 3 * sizeof(float));
                add(v_colors_.get(), Ns * 3 * sizeof(float));
                add(v_opacities_.get(), Ns * sizeof(float));
                add(v_coeffs_.get(), Ns * K_ * 3 * sizeof(float));
                add(v_dirs_.get(), Ns * 3 * sizeof(float));
                add(v_means_.get(), Ns * 3 * sizeof(float));
                add(v_quats_.get(), Ns * 4 * sizeof(float));
                add(v_scales_.get(), Ns * 3 * sizeof(float));
                if (use_depth_loss) add(v_depths_.get(), Ns * sizeof(float));
                if (n_isects == 0)
                    add(offsets_.get(), static_cast<std::size_t>(tw) * th * sizeof(int32_t));
                fill_prefix(cmd, fills, ranges, nf);
            } else if (n_isects == 0) {
                MTL::Buffer* const fills[] = {offsets_.get()};
                const std::size_t bytes[] = {static_cast<std::size_t>(tw) * th *
                                             sizeof(int32_t)};
                fill_prefix(cmd, fills, bytes, 1);
            }

            // [Step breakdown probe — taming-bwd Task 1] When SPLATX_KERNEL_TIMING is set, CB2 of the
            // training step is split into sequential command buffers at the kernel-group boundaries
            // so the GPU time of each segment is recorded. The encoded content and its order are
            // unchanged (each block resets the pipeline and every binding, so swapping encoders does
            // not change the meaning; the same in-order queue plus tracked buffers guarantee ordering
            // and visibility across CB boundaries), and the segment CBs are committed back to back
            // without waiting — they create no GPU idle. Only the last CB is waited on with the usual
            // commit_and_wait. The default (unset) path stays the previous single CB2. This is probe
            // instrumentation only — not a deployment contract.
            const bool split_timing = kernel_timing_enabled() && !in.forward_only;
            std::vector<std::pair<MTL::CommandBuffer*, const char*>> seg_cbs;
            if (split_timing) { // splits the leading gradient zero-fill blit into its own segment
                seg_cbs.emplace_back(cmd, "res2_grad_zfill");
                cmd->commit();
                cmd = mb_.queue_->commandBuffer();
            }

            MTL::ComputeCommandEncoder* enc = cmd->computeCommandEncoder();
            auto seg_cut = [&](const char* label) { // a segment boundary — split_timing only
                if (!split_timing) return;
                enc->endEncoding();
                seg_cbs.emplace_back(cmd, label);
                cmd->commit();
                cmd = mb_.queue_->commandBuffer();
                enc = cmd->computeCommandEncoder();
            };

            if (n_isects > 0) {
                // emit — the same bindings as MetalBackend::isect_tiles pass 2
                enc->setComputePipelineState(
                    mb_.pipeline("isect_tile_emit", {{kFCIsectAccuTile, true}}));
                enc->setBuffer(means2d_.get(), 0, 0);
                enc->setBuffer(radii_.get(), 0, 1);
                enc->setBuffer(depths_.get(), 0, 2);
                enc->setBuffer(conics_.get(), 0, 3);
                enc->setBuffer(raster_opacs, 0, 4);
                enc->setBuffer(cum_.get(), 0, 5);
                enc->setBuffer(keys_[0].get(), 0, 6);
                enc->setBuffer(vals_[0].get(), 0, 7);
                enc->setBytes(&PI, sizeof(IsectTileParams), 8);
                dispatch(enc, N);

                cur = encode_radix_sort(enc, static_cast<std::size_t>(n_isects),
                                        32 + PI.tile_n_bits + PI.image_n_bits);

                IsectOffsetParams PO;
                PO.n_isects = n_isects_u32;
                PO.I = 1;
                PO.n_tiles = tw * th;
                PO.tile_n_bits = bits_for_count(static_cast<int64_t>(tw) * th);
                enc->setComputePipelineState(mb_.pipeline("isect_offset_encode"));
                enc->setBuffer(keys_[cur].get(), 0, 0);
                enc->setBuffer(offsets_.get(), 0, 1);
                enc->setBytes(&PO, sizeof(IsectOffsetParams), 2);
                dispatch(enc, static_cast<std::size_t>(n_isects));
            }
            seg_cut("res2_isect_sort"); // emit + radix sort + offset encode

            // raster fwd — the same bindings as MetalBackend::rasterize_fwd (no bg, no masks)
            enc->setComputePipelineState(
                mb_.pipeline("rasterize_to_pixels_3dgs_fwd", {{kFCRastHasBackground, false},
                                                              {kFCRastHasMasks, false},
                                                              {kFCRastHasDepth, run_depth}}));
            if (run_depth) {
                enc->setBuffer(depths_.get(), 0, 12);
                enc->setBuffer(render_depths_.get(), 0, 13);
            }
            enc->setBuffer(means2d_.get(), 0, 0);
            enc->setBuffer(conics_.get(), 0, 1);
            enc->setBuffer(colors_.get(), 0, 2);
            enc->setBuffer(raster_opacs, 0, 3);
            enc->setBuffer(offsets_.get(), 0, 6);
            enc->setBuffer(vals_[cur].get(), 0, 7); // at n_isects==0 the range is empty — never accessed
            enc->setBuffer(render_colors_.get(), 0, 8);
            enc->setBuffer(render_alphas_.get(), 0, 9);
            enc->setBuffer(last_ids_.get(), 0, 10);
            enc->setBytes(&PR, sizeof(RasterizeParams), 11);
            enc->dispatchThreadgroups(MTL::Size::Make(static_cast<std::size_t>(tw) * th, 1, 1),
                                      MTL::Size::Make(kRastCtaSize, 1, 1));
            enc->memoryBarrier(MTL::BarrierScopeBuffers);
            seg_cut("res2_raster_fwd");

            if (!in.forward_only) {
                MTL::Buffer* loss_img = render_colors_.get();

                SsimParams PS;
                if (use_ssim) {
                    PS.I = 1;
                    PS.H = H;
                    PS.W = W;
                    PS.C = 3;
                    PS.map_h = static_cast<uint32_t>(map_h);
                    PS.map_w = static_cast<uint32_t>(map_w);
                    PS.G = 0.f;
                    // ssim fwd (need_coeffs) — the same bindings as MetalBackend::ssim_fwd
                    enc->setComputePipelineState(
                        mb_.pipeline("ssim_fwd", {{kFCSsimNeedCoeffs, true}}));
                    enc->setBuffer(loss_img, 0, 0);
                    enc->setBuffer(gt_.get(), 0, 1);
                    enc->setBuffer(ssim_map_.get(), 0, 2);
                    enc->setBuffer(c_mu_.get(), 0, 3);
                    enc->setBuffer(c_sigma_.get(), 0, 4);
                    enc->setBuffer(c_cross_.get(), 0, 5);
                    enc->setBytes(&PS, sizeof(SsimParams), 6);
                    enc->dispatchThreadgroups(
                        MTL::Size::Make((map_w + kSsimTileSize - 1) / kSsimTileSize,
                                        (map_h + kSsimTileSize - 1) / kSsimTileSize, 3),
                        MTL::Size::Make(kSsimTileSize, kSsimTileSize, 1));
                    enc->memoryBarrier(MTL::BarrierScopeBuffers);
                    seg_cut("res2_ssim_fwd");
                    // ssim bwd — v_dssim (the kernel absorbs G = -λ/M; identical to MetalBackend::ssim_bwd)
                    PS.G = static_cast<float>(-static_cast<double>(d_.ssim_weight) /
                                              static_cast<double>(map_count));
                    enc->setComputePipelineState(mb_.pipeline("ssim_bwd"));
                    enc->setBuffer(loss_img, 0, 0);
                    enc->setBuffer(gt_.get(), 0, 1);
                    enc->setBuffer(c_mu_.get(), 0, 2);
                    enc->setBuffer(c_sigma_.get(), 0, 3);
                    enc->setBuffer(c_cross_.get(), 0, 4);
                    enc->setBuffer(v_dssim_.get(), 0, 5);
                    enc->setBytes(&PS, sizeof(SsimParams), 6);
                    enc->dispatchThreadgroups(
                        MTL::Size::Make((W + kSsimTileSize - 1) / kSsimTileSize,
                                        (H + kSsimTileSize - 1) / kSsimTileSize, 3),
                        MTL::Size::Make(kSsimTileSize, kSsimTileSize, 1));
                    enc->memoryBarrier(MTL::BarrierScopeBuffers);
                    seg_cut("res2_ssim_bwd");
                }
                { // L1 grad (+ the SSIM combination): v_render = w_l1·v_L1 (+ v_dssim).
                    TrainL1GradParams P;
                    P.n = static_cast<uint32_t>(n_px);
                    P.has_dssim = use_ssim ? 1u : 0u;
                    P.g = 1.0f / static_cast<float>(n_px); // the same f32 g as l1_loss_and_grad
                    P.w_l1 = 1.0f - d_.ssim_weight;
                    enc->setComputePipelineState(mb_.pipeline("train_l1_grad"));
                    enc->setBuffer(loss_img, 0, 0);
                    enc->setBuffer(gt_.get(), 0, 1);
                    enc->setBuffer((use_ssim ? v_dssim_.get() : loss_img), 0, 2);
                    enc->setBuffer(v_render_.get(), 0, 3);
                    enc->setBytes(&P, sizeof(P), 4);
                    dispatch(enc, n_px);
                }
                if (use_depth_loss) {
                    TrainDepthGradParams P;
                    P.width = W;
                    P.height = H;
                    P.depth_width = static_cast<uint32_t>(in.depth_width);
                    P.depth_height = static_cast<uint32_t>(in.depth_height);
                    P.g = in.depth_weight / static_cast<float>(in.depth_valid);
                    enc->setComputePipelineState(mb_.pipeline("train_depth_grad"));
                    enc->setBuffer(render_depths_.get(), 0, 0);
                    enc->setBuffer(render_alphas_.get(), 0, 1);
                    enc->setBuffer(gt_depth_.get(), 0, 2);
                    enc->setBuffer(v_render_depths_.get(), 0, 3);
                    enc->setBuffer(v_render_alphas_.get(), 0, 4);
                    enc->setBytes(&P, sizeof(P), 5);
                    dispatch(enc, HW);
                }
                seg_cut("res2_loss_grad"); // train_l1_grad (+ the SSIM combination, + depth)

                if (n_isects > 0) {
                    // raster bwd — the same bindings as MetalBackend::rasterize_bwd (absgrad off).
                    // At n_isects==0 it is skipped (zeros_like semantics — the gradients stay in the
                    // blit zero-filled state from above).
                    enc->setComputePipelineState(
                        mb_.pipeline("rasterize_to_pixels_3dgs_bwd",
                                     {{kFCRastHasBackground, false},
                                      {kFCRastHasMasks, false},
                                      {kFCRastHasAbsGrad, false},
                                      {kFCRastHasDepth, use_depth_loss}}));
                    enc->setBuffer(means2d_.get(), 0, 0);
                    enc->setBuffer(conics_.get(), 0, 1);
                    enc->setBuffer(colors_.get(), 0, 2);
                    enc->setBuffer(raster_opacs, 0, 3);
                    enc->setBuffer(offsets_.get(), 0, 6);
                    enc->setBuffer(vals_[cur].get(), 0, 7);
                    enc->setBuffer(render_alphas_.get(), 0, 8);
                    enc->setBuffer(last_ids_.get(), 0, 9);
                    enc->setBuffer(v_render_.get(), 0, 10);
                    enc->setBuffer(use_depth_loss ? v_render_alphas_.get()
                                                  : v_render_alphas_zero_.get(),
                                   0, 11);
                    if (use_depth_loss) {
                        enc->setBuffer(depths_.get(), 0, 18);
                        enc->setBuffer(v_render_depths_.get(), 0, 19);
                        enc->setBuffer(v_depths_.get(), 0, 20);
                    }
                    enc->setBuffer(v_means2d_.get(), 0, 12);
                    enc->setBuffer(v_conics_.get(), 0, 13);
                    enc->setBuffer(v_colors_.get(), 0, 14);
                    enc->setBuffer(v_opacities_.get(), 0, 15);
                    enc->setBytes(&PR, sizeof(RasterizeParams), 17);
                    enc->dispatchThreadgroups(
                        MTL::Size::Make(static_cast<std::size_t>(tw) * th, 1, 1),
                        MTL::Size::Make(kRastTileSize, kRastTileSize, 1));
                    enc->memoryBarrier(MTL::BarrierScopeBuffers);
                }
                seg_cut("res2_raster_bwd");

                elt(enc, "train_relu_mask_apply", {v_colors_.get(), relu_mask_.get()}, N * 3);

                { // sh bwd — the same bindings as MetalBackend::sh_bwd (masks + v_dirs)
                    SHParams P;
                    P.degrees_to_use = in.degrees_to_use;
                    P.B = 1;
                    P.N = N;
                    P.K = static_cast<uint32_t>(K_);
                    P.D = 3;
                    enc->setComputePipelineState(mb_.pipeline(
                        "sh_bwd", {{kFCShHasMasks, true}, {kFCShComputeVDirs, true}}));
                    enc->setBuffer(dirs_.get(), 0, 0);
                    enc->setBuffer(coeffs_.get(), 0, 1);
                    enc->setBuffer(valid_.get(), 0, 2);
                    enc->setBuffer(v_colors_.get(), 0, 3);
                    enc->setBuffer(v_coeffs_.get(), 0, 4);
                    enc->setBuffer(v_dirs_.get(), 0, 5);
                    enc->setBytes(&P, sizeof(SHParams), 6);
                    dispatch(enc, static_cast<std::size_t>(N) * 3);
                }
                seg_cut("res2_sh_bwd"); // train_relu_mask_apply + sh_bwd

                if (aa) { // the AA product-rule chain (before proj bwd — the same point as Trainer.cpp)
                    enc->setComputePipelineState(mb_.pipeline("train_aa_chain"));
                    TrainEltParams P{N};
                    enc->setBuffer(v_opacities_.get(), 0, 0);
                    enc->setBuffer(opacs_act_.get(), 0, 1);
                    enc->setBuffer(comps_.get(), 0, 2);
                    enc->setBuffer(v_comp_.get(), 0, 3);
                    enc->setBytes(&P, sizeof(P), 4);
                    dispatch(enc, N);
                }

                // proj bwd — the same bindings as MetalBackend::projection_fused_bwd
                // (C==1 → the kProjSingleCamera non-atomic store). The viewmats gradient is unused on
                // the resident path — always specialized to false (it exists only in the standalone
                // proj bwd API reached through the host).
                PJ.has_opacities = 0u; // unused in bwd (01 Correction 1)
                enc->setComputePipelineState(
                    mb_.pipeline("projection_ewa_3dgs_fused_bwd",
                                 {{kFCProjUseCovars, false},
                                  {kFCProjCalcComp, aa},
                                  {kFCProjSingleCamera, true},
                                  {kFCProjViewmatsGrad, false}}));
                enc->setBuffer(p_[0].get(), 0, 0);
                enc->setBuffer(p_[1].get(), 0, 2);
                enc->setBuffer(scales_lin_.get(), 0, 3);
                enc->setBuffer(viewmat_buf_.get(), 0, 4);
                enc->setBuffer(K_buf_.get(), 0, 5);
                enc->setBuffer(radii_.get(), 0, 6);
                enc->setBuffer(conics_.get(), 0, 7);
                if (aa) {
                    enc->setBuffer(comps_.get(), 0, 8);
                    enc->setBuffer(v_comp_.get(), 0, 12);
                }
                enc->setBuffer(v_means2d_.get(), 0, 9);
                enc->setBuffer(use_depth_loss ? v_depths_.get() : v_depths_zero_.get(), 0, 10);
                enc->setBuffer(v_conics_.get(), 0, 11);
                enc->setBuffer(v_means_.get(), 0, 13);
                enc->setBuffer(v_quats_.get(), 0, 15);
                enc->setBuffer(v_scales_.get(), 0, 16);
                enc->setBytes(&PJ, sizeof(ProjectionParams), 18);
                dispatch(enc, N);
                seg_cut("res2_proj_bwd"); // (includes train_aa_chain when aa) + proj bwd

                // v_means += v_dirs (folding in sh_bwd — D6), the activation gradient chain + the reg addition
                elt(enc, "train_add_inplace", {v_means_.get(), v_dirs_.get()}, N * 3);
                {
                    TrainGradChainParams P{N * 3, in.scale_reg_w};
                    enc->setComputePipelineState(mb_.pipeline("train_scale_grad_chain"));
                    enc->setBuffer(v_scales_.get(), 0, 0);
                    enc->setBuffer(scales_lin_.get(), 0, 1);
                    enc->setBytes(&P, sizeof(P), 2);
                    dispatch(enc, static_cast<std::size_t>(N) * 3);
                }
                {
                    TrainGradChainParams P{N, in.opacity_reg_w};
                    enc->setComputePipelineState(mb_.pipeline("train_opacity_grad_chain"));
                    enc->setBuffer(v_opacities_.get(), 0, 0);
                    enc->setBuffer(opacs_act_.get(), 0, 1);
                    enc->setBytes(&P, sizeof(P), 2);
                    dispatch(enc, N);
                }
                { // Splits v_coeffs → v_sh0/v_shN (at K==1 v_shN is never accessed — bound as a dummy)
                    TrainCoeffsParams P{N, static_cast<uint32_t>(K_)};
                    enc->setComputePipelineState(mb_.pipeline("train_split_v_coeffs"));
                    enc->setBuffer(v_coeffs_.get(), 0, 0);
                    enc->setBuffer(v_sh0_.get(), 0, 1);
                    enc->setBuffer((v_shN_ ? v_shN_ : v_sh0_).get(), 0, 2);
                    enc->setBytes(&P, sizeof(P), 3);
                    dispatch(enc, static_cast<std::size_t>(N) * K_ * 3);
                }
                seg_cut("res2_grad_chains"); // add_inplace + scale/opacity chain + split_v_coeffs

                // fused_adam ×6 — in D3 order with the valid mask (identical to MetalBackend::fused_adam)
                MTL::Buffer* const grads[6] = {v_means_.get(),  v_quats_.get(),
                                               v_scales_.get(), v_opacities_.get(),
                                               v_sh0_.get(),    v_shN_.get()};
                for (int g = 0; g < 6; g++) {
                    if (dims_[g] == 0) continue; // shN at sh_degree 0 (D8-6)
                    AdamParams P;
                    P.N = N;
                    P.D = static_cast<uint32_t>(dims_[g]);
                    P.lr = in.lr[g];
                    P.b1 = d_.b1;
                    P.b2 = d_.b2;
                    P.eps = d_.eps;
                    enc->setComputePipelineState(
                        mb_.pipeline("fused_adam", {{kFCAdamHasValidMask, true}}));
                    enc->setBuffer(p_[g].get(), 0, 0);
                    enc->setBuffer(grads[g], 0, 1);
                    enc->setBuffer(m1_[g].get(), 0, 2);
                    enc->setBuffer(m2_[g].get(), 0, 3);
                    enc->setBuffer(valid_.get(), 0, 4);
                    enc->setBytes(&P, sizeof(AdamParams), 5);
                    dispatch(enc, static_cast<std::size_t>(N) * dims_[g]);
                }
                seg_cut("res2_adam"); // fused_adam ×6

                if (in.inject_noise) { // MCMC noise (non-refine steps — the Trainer gates this)
                    RngParams P;
                    P.seed_lo = static_cast<uint32_t>(in.noise_seed & 0xFFFFFFFFu);
                    P.seed_hi = static_cast<uint32_t>(in.noise_seed >> 32);
                    P.offset_lo = static_cast<uint32_t>(in.noise_offset & 0xFFFFFFFFu);
                    P.offset_hi = static_cast<uint32_t>(in.noise_offset >> 32);
                    P.count = N * 3;
                    enc->setComputePipelineState(mb_.pipeline("rng_normal_f32"));
                    enc->setBuffer(noise_.get(), 0, 0);
                    enc->setBytes(&P, sizeof(RngParams), 1);
                    dispatch(enc, (static_cast<std::size_t>(N) * 3 + 3) / 4);

                    PerturbParams PP;
                    PP.N = N;
                    PP.noise_scale = in.noise_scale;
                    PP.t = 0.005f; // DEFAULT_MCMC_OPACITY_T/K (the same as the PerturbInputs defaults)
                    PP.k = 100.f;
                    enc->setComputePipelineState(mb_.pipeline("mcmc_perturb_positions"));
                    enc->setBuffer(p_[0].get(), 0, 0); // positions in place (the resident means)
                    enc->setBuffer(p_[1].get(), 0, 1);
                    enc->setBuffer(p_[2].get(), 0, 2);
                    enc->setBuffer(p_[3].get(), 0, 3);
                    enc->setBuffer(noise_.get(), 0, 4);
                    enc->setBytes(&PP, sizeof(PerturbParams), 5);
                    dispatch(enc, N);
                }
            }
            enc->endEncoding();
            // Under split_timing the last CB holds only the noise (rng+perturb) tail — hence the separate label.
            MetalBackend::commit_and_wait(cmd, split_timing ? "res2_tail_noise"
                                                            : "resident_step_cb2");
            // Reports the per-segment GPU time — at this point waiting on the last CB has guaranteed
            // that everything completed (an in-order queue). Segment errors are propagated per label.
            for (const auto& [scmd, label] : seg_cbs) {
                if (scmd->error())
                    throw_ns_error(std::string("MetalBackend: resident segment ") + label +
                                       " command buffer failed",
                                   scmd->error());
                std::fprintf(stderr, "[ktime] %s %.4f\n", label,
                             (scmd->GPUEndTime() - scmd->GPUStartTime()) * 1000.0);
            }
        }

        // ---------- loss (host f64 summation — the same expression as the l1_loss_and_grad/ssim conventions) ----------
        ResidentStepOutputs out;
        out.n_isects = n_isects;
        const float* render = static_cast<const float*>(render_colors_->contents());
        if (!in.forward_only) {
            const float* loss_src = render;
            double acc = 0.0;
            for (std::size_t i = 0; i < n_px; i++) {
                const float dv = loss_src[i] - in.gt_rgb[i];
                acc += std::fabs(static_cast<double>(dv));
            }
            const float l1_loss = static_cast<float>(acc / static_cast<double>(n_px));
            if (use_ssim) {
                const double ssim = ssim_reduce_mean_f64(
                    static_cast<const float*>(ssim_map_->contents()), map_count);
                out.loss = (1.0f - d_.ssim_weight) * l1_loss +
                           d_.ssim_weight * (1.0f - static_cast<float>(ssim));
            } else {
                out.loss = l1_loss;
            }
        }
        if (in.want_render) out.render_colors.assign(render, render + n_px); // keeps the raw render
        if (run_depth) {
            const float* acc = static_cast<const float*>(render_depths_->contents());
            const float* alpha = static_cast<const float*>(render_alphas_->contents());
            if (use_depth_loss)
                out.depth_loss = depth_l1_loss(acc, alpha, in.gt_depth, static_cast<int>(W),
                                               static_cast<int>(H), in.depth_width,
                                               in.depth_height, in.depth_valid);
            if (in.want_render || in.want_depth) {
                out.render_depths.resize(HW);
                for (std::size_t i = 0; i < HW; i++)
                    out.render_depths[i] = expected_depth(acc[i], alpha[i]);
            }
        }
        return out;
    }

private:
    // The single allocation path for the resident buffers (the ctor parameter, moment, and
    // intermediate buffers plus every ensure_* growth reallocation go through this function) — on
    // failure it appends a pointer to the --no-resident escape hatch to the message
    // (Phase 3.5 Task 1: the wording only; there is no fallback logic).
    NS::SharedPtr<MTL::Buffer> alloc(std::size_t bytes) {
        return mb_.upload_buffer(nullptr, bytes,
                                 "; device-resident executor buffer — if out of memory, "
                                 "rerun with --no-resident to use the host round-trip path");
    }

    // Shared encoding for the small elementwise kernels (the TrainEltParams contract kernels — the
    // buffer order equals the kernel declaration order).
    void elt(MTL::ComputeCommandEncoder* enc, const char* fn,
             std::initializer_list<MTL::Buffer*> bufs, std::size_t n) {
        enc->setComputePipelineState(mb_.pipeline(fn));
        uint32_t slot = 0;
        for (MTL::Buffer* b : bufs) enc->setBuffer(b, 0, slot++);
        TrainEltParams P{static_cast<uint32_t>(n)};
        enc->setBytes(&P, sizeof(P), slot);
        dispatch(enc, n);
    }

    // dispatchThreads(n, tg 256) + a write → read visibility barrier (the sequential single-encoder
    // chaining convention — the same pattern as encode_prefix_scan and the sort).
    static void dispatch(MTL::ComputeCommandEncoder* enc, std::size_t n) {
        enc->dispatchThreads(MTL::Size::Make(n, 1, 1), MTL::Size::Make(256, 1, 1));
        enc->memoryBarrier(MTL::BarrierScopeBuffers);
    }

    // blit fill 0 — only the count-row prefix range (equivalent to the exact-size buffer zero-fill on the host path).
    static void fill_prefix(MTL::CommandBuffer* cmd, MTL::Buffer* const* bufs,
                            const std::size_t* bytes, std::size_t n) {
        MTL::BlitCommandEncoder* blit = cmd->blitCommandEncoder();
        for (std::size_t i = 0; i < n; i++)
            blit->fillBuffer(bufs[i], NS::Range::Make(0, bytes[i]), 0);
        blit->endEncoding();
    }

    // Encodes the GPU radix sort — the same 5-kernel loop as MetalBackend::sort_isect (the double
    // buffer reuses the resident keys_/vals_). It returns the index of the buffer holding the final
    // sorted result.
    int encode_radix_sort(MTL::ComputeCommandEncoder* enc, std::size_t n, uint32_t key_n_bits) {
        const uint32_t num_passes = (key_n_bits + kRadixBitsPerPass - 1) / kRadixBitsPerPass;
        const uint32_t num_wgs = static_cast<uint32_t>((n + kRadixBlockSize - 1) / kRadixBlockSize);
        const uint32_t num_reduce_wgs =
            kRadixBinCount * ((num_wgs + kRadixBlockSize - 1) / kRadixBlockSize);

        RadixSortParams P;
        P.shift = 0;
        P.num_keys = static_cast<uint32_t>(n);
        P.num_wgs = num_wgs;
        P.num_reduce_wgs = num_reduce_wgs;

        const MTL::Size tg = MTL::Size::Make(kRadixWG, 1, 1);
        int cur = 0;
        enc->memoryBarrier(MTL::BarrierScopeBuffers); // the emit writes → the count reads
        for (uint32_t pass = 0; pass < num_passes; pass++) {
            P.shift = pass * kRadixBitsPerPass;
            enc->setComputePipelineState(mb_.pipeline("radix_sort_count"));
            enc->setBuffer(keys_[cur].get(), 0, 0);
            enc->setBuffer(counts_.get(), 0, 1);
            enc->setBytes(&P, sizeof(P), 2);
            enc->dispatchThreadgroups(MTL::Size::Make(num_wgs, 1, 1), tg);
            enc->memoryBarrier(MTL::BarrierScopeBuffers);

            enc->setComputePipelineState(mb_.pipeline("radix_sort_reduce"));
            enc->setBuffer(counts_.get(), 0, 0);
            enc->setBuffer(reduced_.get(), 0, 1);
            enc->setBytes(&P, sizeof(P), 2);
            enc->dispatchThreadgroups(MTL::Size::Make(num_reduce_wgs, 1, 1), tg);
            enc->memoryBarrier(MTL::BarrierScopeBuffers);

            enc->setComputePipelineState(mb_.pipeline("radix_sort_scan"));
            enc->setBuffer(reduced_.get(), 0, 0);
            enc->setBytes(&P, sizeof(P), 1);
            enc->dispatchThreadgroups(MTL::Size::Make(1, 1, 1), tg);
            enc->memoryBarrier(MTL::BarrierScopeBuffers);

            enc->setComputePipelineState(mb_.pipeline("radix_sort_scan_add"));
            enc->setBuffer(reduced_.get(), 0, 0);
            enc->setBuffer(counts_.get(), 0, 1);
            enc->setBytes(&P, sizeof(P), 2);
            enc->dispatchThreadgroups(MTL::Size::Make(num_reduce_wgs, 1, 1), tg);
            enc->memoryBarrier(MTL::BarrierScopeBuffers);

            enc->setComputePipelineState(mb_.pipeline("radix_sort_scatter"));
            enc->setBuffer(keys_[cur].get(), 0, 0);
            enc->setBuffer(vals_[cur].get(), 0, 1);
            enc->setBuffer(counts_.get(), 0, 2);
            enc->setBuffer(keys_[1 - cur].get(), 0, 3);
            enc->setBuffer(vals_[1 - cur].get(), 0, 4);
            enc->setBytes(&P, sizeof(P), 5);
            enc->dispatchThreadgroups(MTL::Size::Make(num_wgs, 1, 1), tg);
            enc->memoryBarrier(MTL::BarrierScopeBuffers);
            cur = 1 - cur;
        }
        return cur;
    }

    // Ensures the capacity of the isect-sized buffers (1.5x growth with reuse — retained across steps)
    void ensure_isect_capacity(std::size_t n) {
        if (n <= isect_cap_ && keys_[0]) return;
        std::size_t cap = std::max<std::size_t>(isect_cap_, 1024);
        while (cap < n) cap += cap / 2;
        const uint32_t num_wgs =
            static_cast<uint32_t>((cap + kRadixBlockSize - 1) / kRadixBlockSize);
        const uint32_t num_reduce_wgs =
            kRadixBinCount * ((num_wgs + kRadixBlockSize - 1) / kRadixBlockSize);
        const uint32_t reduced_size =
            std::max(1u, (num_reduce_wgs + kRadixBlockSize - 1) / kRadixBlockSize) *
            kRadixBlockSize;
        keys_[0] = alloc(cap * sizeof(uint64_t));
        keys_[1] = alloc(cap * sizeof(uint64_t));
        vals_[0] = alloc(cap * sizeof(int32_t));
        vals_[1] = alloc(cap * sizeof(int32_t));
        counts_ = alloc(static_cast<std::size_t>(kRadixBinCount) * num_wgs * sizeof(uint32_t));
        reduced_ = alloc(reduced_size * sizeof(uint32_t));
        isect_cap_ = cap;
    }

    // Ensures the capacity of the image- and tile-sized buffers (growth with reuse).
    // v_render_alphas is always 0 (neither L1 nor SSIM uses alpha — matching the v_render_alphas zero
    // buffer in Trainer.cpp).
    void ensure_image_capacity(std::size_t hw, std::size_t n_tiles, bool need_ssim) {
        if (hw > image_cap_) {
            std::size_t cap = std::max<std::size_t>(hw, 1);
            gt_ = alloc(cap * 3 * sizeof(float));
            render_colors_ = alloc(cap * 3 * sizeof(float));
            render_alphas_ = alloc(cap * sizeof(float));
            last_ids_ = alloc(cap * sizeof(int32_t));
            v_render_ = alloc(cap * 3 * sizeof(float));
            v_render_alphas_zero_ = alloc(cap * sizeof(float));
            std::memset(v_render_alphas_zero_->contents(), 0, cap * sizeof(float));
            image_cap_ = cap;
            ssim_cap_ = 0; // the ssim maps are image-sized too — reallocated alongside
            depth_image_cap_ = 0;
        }
        if (need_ssim && image_cap_ > ssim_cap_) {
            ssim_map_ = alloc(image_cap_ * 3 * sizeof(float));
            c_mu_ = alloc(image_cap_ * 3 * sizeof(float));
            c_sigma_ = alloc(image_cap_ * 3 * sizeof(float));
            c_cross_ = alloc(image_cap_ * 3 * sizeof(float));
            v_dssim_ = alloc(image_cap_ * 3 * sizeof(float));
            ssim_cap_ = image_cap_;
        }
        if (n_tiles > tiles_cap_) {
            offsets_ = alloc(n_tiles * sizeof(int32_t));
            tiles_cap_ = n_tiles;
        }
    }

    // The depth channel's image-sized buffers, allocated only once a step runs the channel.
    void ensure_depth_image_capacity() {
        if (depth_image_cap_ >= image_cap_ && render_depths_) return;
        render_depths_ = alloc(image_cap_ * sizeof(float));
        v_render_depths_ = alloc(image_cap_ * sizeof(float));
        v_render_alphas_ = alloc(image_cap_ * sizeof(float));
        depth_image_cap_ = image_cap_;
    }

    void ensure_depth_capacity(std::size_t n) {
        if (n <= depth_cap_ && gt_depth_) return;
        gt_depth_ = alloc(n * sizeof(float));
        depth_cap_ = n;
    }

    void sync_params(const ResidentParamSync& s, bool to_device) {
        if (s.rows < 1 || s.rows > d_.arena_rows)
            throw std::invalid_argument("TrainStepExecutor: sync rows out of [1, arena_rows]");
        for (int g = 0; g < 6; g++) {
            if (dims_[g] == 0) continue;
            if (!s.params[g] || !s.exp_avg[g] || !s.exp_avg_sq[g])
                throw std::invalid_argument("TrainStepExecutor: null sync pointer");
            const std::size_t bytes =
                static_cast<std::size_t>(s.rows) * dims_[g] * sizeof(float);
            if (to_device) {
                std::memcpy(p_[g]->contents(), s.params[g], bytes);
                std::memcpy(m1_[g]->contents(), s.exp_avg[g], bytes);
                std::memcpy(m2_[g]->contents(), s.exp_avg_sq[g], bytes);
            } else {
                std::memcpy(s.params[g], p_[g]->contents(), bytes);
                std::memcpy(s.exp_avg[g], m1_[g]->contents(), bytes);
                std::memcpy(s.exp_avg_sq[g], m2_[g]->contents(), bytes);
            }
        }
    }

    MetalBackend& mb_; // not owned — the Trainer keeps the executor only within the backend's lifetime
    TrainStepExecutorDesc d_;
    int64_t K_ = 1;
    std::array<int64_t, 6> dims_{}; // elements per row in D3 group order {3,4,3,1,3,(K-1)*3}

    // Parameters and moments (arena-resident)
    NS::SharedPtr<MTL::Buffer> p_[6], m1_[6], m2_[6];
    // fwd intermediates (arena-resident)
    NS::SharedPtr<MTL::Buffer> scales_lin_, opacs_act_, comps_, opacs_aa_;
    NS::SharedPtr<MTL::Buffer> radii_, means2d_, depths_, conics_;
    NS::SharedPtr<MTL::Buffer> valid_, dirs_, coeffs_, colors_, relu_mask_;
    NS::SharedPtr<MTL::Buffer> tiles_per_gauss_, cum_, block_sums_;
    // bwd gradients (arena-resident)
    NS::SharedPtr<MTL::Buffer> v_means2d_, v_conics_, v_colors_, v_opacities_, v_coeffs_,
        v_dirs_, v_means_, v_quats_, v_scales_, v_sh0_, v_shN_, v_comp_, v_depths_zero_, v_depths_;
    NS::SharedPtr<MTL::Buffer> noise_, viewmat_buf_, K_buf_;
    // isect- and image-sized (capacity-based growth with reuse)
    NS::SharedPtr<MTL::Buffer> keys_[2], vals_[2], counts_, reduced_;
    NS::SharedPtr<MTL::Buffer> gt_, render_colors_, render_alphas_, last_ids_, v_render_,
        v_render_alphas_zero_, offsets_, render_depths_, v_render_depths_, v_render_alphas_,
        gt_depth_;
    NS::SharedPtr<MTL::Buffer> ssim_map_, c_mu_, c_sigma_, c_cross_, v_dssim_;
    std::size_t isect_cap_ = 0, image_cap_ = 0, ssim_cap_ = 0, tiles_cap_ = 0, depth_cap_ = 0,
                depth_image_cap_ = 0;
};

std::unique_ptr<TrainStepExecutor>
MetalBackend::make_train_step_executor(const TrainStepExecutorDesc& d) {
    return std::make_unique<MetalTrainStepExecutor>(*this, d);
}

} // namespace

std::unique_ptr<KernelBackend> make_metal_backend(void* device_ptr,
                                                  const MetalBackendDesc& desc) {
    return std::make_unique<MetalBackend>(device_ptr, desc);
}

// Backward compatibility — source compilation only (the metallib layers are disabled:
// search_exe_dir=false so that tests take the source path deterministically, regardless of whether an
// exe-adjacent artifact exists).
std::unique_ptr<KernelBackend> make_metal_backend(void* device_ptr,
                                                  const std::string& kernel_src_dir) {
    MetalBackendDesc desc;
    desc.search_exe_dir = false;
    desc.kernel_src_dir = kernel_src_dir;
    return std::make_unique<MetalBackend>(device_ptr, desc);
}

} // namespace splatx
