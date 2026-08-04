// Derived from nerfstudio-project/gsplat (Apache-2.0), commit 77ab983.
// Reimplemented for Metal/C++.
#pragma once
#include <cstdint>
#include <vector>

namespace splatx {

struct ProjectionInputs {
    const float* means;    // [N,3] world
    const float* covars;   // [N,3,3] world, full 3x3 row-major
    const float* viewmats; // [C,4,4] world→cam row-major
    const float* Ks;       // [C,3,3]
    int64_t N = 0, C = 0;
    int width = 0, height = 0;
    float eps2d = 0.3f, near_plane = 0.01f, far_plane = 1e10f;
    bool calc_compensations = false;
};
struct ProjectionOutputs {
    std::vector<int32_t> radii;         // [C,N,2]
    std::vector<float> means2d;         // [C,N,2]
    std::vector<float> depths;          // [C,N]
    std::vector<float> conics;          // [C,N,3] (inverse covariance, upper-triangular: a, b, c)
    std::vector<float> compensations;   // [C,N] (filled only when calc_compensations)
};
ProjectionOutputs projection_ewa_fwd_cpu(const ProjectionInputs& in);

// Spec 06 fused_adam CPU reference (a reimplementation of AdamCUDA.cu:34–71) — the same formulas as
// the Metal kernel: all arithmetic in f32, no bias correction, eps added outside sqrt(v), and
// Gaussians with valid[g]==0 skip the parameter and both moments entirely (the moments are not
// even decayed). in-place.
struct AdamInputs; // defined in splatx/KernelBackend.h (single source of truth for the kernel interface)
void fused_adam_cpu(const AdamInputs& in);

// Spec 01 projection_ewa_3dgs_fused_fwd CPU reference (ProjectionEWA3DGSFused.cu:38–219, merged
// with Corrections 1–6 of the verification notes) — the same formulas and culling semantics as the
// Metal kernel (ProjectionEWA.metal).
// Unlike projection_ewa_fwd_cpu (above, which follows the _torch_impl.py semantics), this follows
// the CUDA kernel semantics: covars [N,6] upper-triangular (triu) and the quat+scale path both,
// det<=0 culling, opacity-aware extend, radius_clip, image-boundary culling on equality, and on a
// cull only the two radii slots are set to 0 (the other outputs are unwritten — on CPU the vectors
// are zero-initialized, so the culled slots stay 0; harness comparisons are made only under the
// valid=(radii>0) mask — D3 / 07 Correction 3).
// PINHOLE (camera_model=0) only — anything else throws (Phase 1 disposition #3).
struct ProjectionFusedInputs;  // defined in splatx/KernelBackend.h
struct ProjectionFusedOutputs;
ProjectionFusedOutputs projection_fused_fwd_cpu(const ProjectionFusedInputs& in);

// Spec 02 sh_fwd CPU reference (a reimplementation of the semantics of
// SphericalHarmonicsCUDA.cu:42–140 sh_coeffs_to_color_fast + the generic fwd kernel at :434–468) —
// the same formulas, constants (deg 0–4), and mask-skip semantics as the Metal kernel
// (SphericalHarmonics.metal). All arithmetic in f32; dirs normalization uses 1/sqrt (corresponding
// to CUDA rsqrtf / Metal precise::rsqrt — D2); unguarded at dir=0 (NaN propagates — disposition
// #6). Masked-out elements are unwritten — on CPU they stay 0 as a by-product of the
// zero-initialized vectors (disposition #5; the harness compares under the mask). There is no
// backward CPU reference — bwd is verified by comparing directly against the autograd golden
// fixture (an additional controller decision in 00-index §6).
struct ShInputs;  // defined in splatx/KernelBackend.h
struct ShOutputs;
ShOutputs sh_fwd_cpu(const ShInputs& in);

// ===== Spec 03 isect_tiles shared host helpers =====

// Spec 03 bits_for_count (a reimplementation of gsplat MathUtils.h:23–33) — the number of bits
// needed to index 0..count-1. Every site — key encoding (image_n_bits/tile_n_bits), offset
// unpacking, and the sort bit width — MUST use this single source of truth for the pack/unpack
// round trip to be guaranteed (spec 03 §1).
// Examples: count=1 → 0, count=8 → 3, count=8160 → 13.
inline std::uint32_t bits_for_count(std::int64_t count) {
    if (count <= 1) return 0u;
    std::uint32_t bits = 0;
    for (std::uint64_t v = static_cast<std::uint64_t>(count) - 1u; v != 0; v >>= 1) bits++;
    return bits;
}

// tiles_per_gauss(int32) →
// cum_tiles_per_gauss(int64) inclusive cumsum — corresponds to the upstream at::cumsum (int64
// explicitly, Intersect.cpp:258; the kernel reads it as int64). The Task 3 reference: shared by
// both the CPU and Metal backends; in Task 4 only the Metal side is replaced by a GPU 3-dispatch
// scan (reduce-then-scan).
inline std::vector<std::int64_t> cumsum_i64(const std::int32_t* v, std::size_t n) {
    std::vector<std::int64_t> cum(n);
    std::int64_t running = 0;
    for (std::size_t i = 0; i < n; i++) {
        running += v[i];
        cum[i] = running;
    }
    return cum;
}

// Spec 03 intersect_tile CPU reference (a reimplementation of the AccuTile helpers at
// IntersectTile.cu:83–207 + the kernel body at :213–464 + the 2-pass orchestration at
// Intersect.cpp:230–296) — the same formulas and emit order as the Metal kernel
// (IntersectTile.metal). The path gate is exactly the upstream one (:288): if both conics and
// opacities are given, AccuTile (SNUGBOX + the ellipse-line sweep, SpeedySplat arXiv:2412.00578 —
// the opacity-aware level-set t uses logf per disposition #11, and the t<=0 early return is
// approved deviation #13); if both are nullptr, the AABB fallback.
// **This function is the producer of the AccuTile goldens** — torch _isect_tiles has no AccuTile
// (07 Correction 2), so fixtures/isect_accutile is generated with
// `splatx-core-tests --dump-accutile` and committed. The cumsum is the int64 host one (cumsum_i64).
// CUDA's `assert(depth_f >= 0.f)` (IntersectTile.cu:283 — a negative value collapses the unsigned
// key order) is promoted to a validating throw on CPU, while the Metal kernel has no assert (the
// substitute rule of spec 03 §kernel 3).
struct IsectTilesInputs; // defined in splatx/KernelBackend.h
struct IsectTilesOutputs;
IsectTilesOutputs isect_tiles_cpu(const IsectTilesInputs& in);

// Spec 03 sort contract CPU reference (corresponding to cub::DeviceRadixSort::SortPairs,
// Intersect.cpp:299–324) — an index-based std::stable_sort, u64 unsigned key comparison, and ties
// preserve the input (emit) order. The golden equivalence criterion: np.lexsort((lo,hi)) (task-2
// report — the *_sorted goldens).
// A key with any bit set above key_n_bits (= 32 + tile_n_bits + image_n_bits) throws — the
// validation promotion of the encoding contract that lets the GPU radix (Task 7) sort only the low
// key_n_bits and still be equivalent (the same pattern as the substitute depth-validation rule of
// spec 03 §kernel 3; the Metal radix performs no check).
struct SortIsectInputs; // defined in splatx/KernelBackend.h
struct SortIsectOutputs;
SortIsectOutputs sort_isect_cpu(const SortIsectInputs& in);

// Spec 03 isect_offset_encode CPU reference (a direct port of intersect_offset_kernel at
// IntersectTile.cu:925–988 + the host zero fill for n_isects==0 at :1004–1008) — the same formulas
// as the Metal kernel (IntersectTile.metal). It builds the per-tile start offsets
// [I*tile_height*tile_width] (int32) from the sorted isect_ids.
// The tile mask is the 64-bit ((uint64_t)1 << tile_n_bits) - 1 — guarding against the UB of the
// upstream `(1 << tile_n_bits)` int shift at tile_n_bits=32 (spec 03 §kernel 4). tile_n_bits is
// recomputed with bits_for_count (the single source of truth above) — guaranteeing the pack/unpack
// round trip with the isect_tiles encoding.
struct IsectOffsetInputs; // defined in splatx/KernelBackend.h
std::vector<std::int32_t> isect_offset_encode_cpu(const IsectOffsetInputs& in);

// Spec 05 rasterize forward CPU reference (a direct port of
// RasterizeToPixels3DGSSerialBatchFwd.cu:41-297 + RasterizeToPixels3DGSDevice.cuh:44-55 — the same
// semantics as the Task 2 scalar oracle, reimplemented in C++ f32) — the same formulas, gates, and
// early termination as the Metal kernel (Rasterize3DGSFwd.metal).
// It is a sequential per-pixel loop (no cooperative load, no barriers), hence deterministic. All
// arithmetic in f32: sigma/alpha/next_T follow the CUDA expressions exactly, vis =
// std::exp(-sigma) (corresponding to Metal precise::exp — D2), and the weighted color sum and the
// background compositing use std::fma (corresponding to Metal metal::fma — the fast-math lesson of
// 1a Task 4). The early termination next_T <= 1e-4 is **exclusive** (the triggering Gaussian is not
// included — Fwd.cu:246-251), and last_ids is the absolute isect index (0 when there is no
// contribution). The background follows the final write expression pix_out + T·bg
// (Fwd.cu:286-293 — written even for in-bounds pixels of an empty tile, equivalent to 07
// Correction 4). The masks path is implemented only (not verified in v1).
struct RasterizeFwdInputs;  // defined in splatx/KernelBackend.h
struct RasterizeFwdOutputs;
RasterizeFwdOutputs rasterize_fwd_cpu(const RasterizeFwdInputs& in);

// ===== Spec 06 MCMC (Phase 1c Task 5) =====

// rng_philox_u32 / rng_normal_f32 CPU reference — uses the single-source-of-truth core in
// RngPhilox.h (the same source that Rng.metal uses, compiled for the host). The philox integer
// stages and the uniform conversion are contractually **bit-identical** with Metal, while
// Box-Muller uses std::log/sqrt/sin/cos (corresponding to Metal precise:: — a rel gate,
// 06 Correction 5). Written from scratch (based on the literature) — see the citations in the
// RngPhilox.h header.
struct RngNormalInputs; // defined in splatx/KernelBackend.h
std::vector<std::uint32_t> rng_philox_u32_cpu(const RngNormalInputs& in);
std::vector<float> rng_normal_f32_cpu(const RngNormalInputs& in);

// Host precomputation of the binoms LUT (a reimplementation of gsplat strategy/mcmc.py:95–100) —
// [n_max, n_max] row-major, lut[n*n_max + k] = C(n,k) (for k <= n, otherwise 0 — corresponding to
// torch.zeros).
// Pascal's triangle accumulated in u64 integers (C(50,25) ≈ 1.26e14 < 2^63 — exact) and then cast
// to float — bit-identical with the upstream math.comb → float32 tensor (the fp32 representation
// loss is reproduced identically as well, 06 §kernel 2).
std::vector<float> make_binoms_lut(int n_max = 51);

// mcmc_relocation CPU reference (a reimplementation of RelocationCUDA.cu:35–79) — the same formulas
// as the Metal kernel (AdamMCMC.metal): the clamp comes before the scale computation, denom_sum is
// accumulated sequentially in ascending (i,k) order (remaining gap 2 of the verification notes —
// the fixture goldens use the same order), and pow **forces the float path**
// (std::pow(float,float) — 06 Correction 2: the double-promotion trap of unqualified pow).
// ratios[idx] ∉ [1, n_max] throws — the validation promotion of the host clamp contract
// (relocation.py:61–62) (the same pattern as the high bits of the sort_isect_cpu keys; the Metal
// kernel performs no check, exactly as the upstream original).
struct RelocationInputs;  // defined in splatx/KernelBackend.h
struct RelocationOutputs;
RelocationOutputs mcmc_relocation_cpu(const RelocationInputs& in);

// mcmc_perturb_positions CPU reference (a reimplementation of MCMCPerturbCUDA.cu:27–61 +
// Utils.cuh:228–311) — the same formulas as the Metal kernel: quat normalization by 1/sqrt (of the
// sum of squares in x→y→z→w order — corresponding to KernelCommon.h quat_to_rotmat),
// Σ = (R·S)·(R·S)ᵀ, the sigmoid gate w = σ(-k(σ(op)-t))·noise_scale, Σ·n (not Cholesky), and an
// in-place addition to positions.
// The covariance composition can round differently from the fma-contraction family of the Metal
// float3x3 product, so it is compared under a tolerance (gated by the fixture meta — the
// fma/precise policy of spec 06).
struct PerturbInputs;     // defined in splatx/KernelBackend.h
void mcmc_perturb_positions_cpu(const PerturbInputs& in);

// ===== Spec 08 SSIM (Phase 3 Task 2) =====

// The single source of truth for the ssim scalar reduction — the map is summed **sequentially in
// ascending index order in f64** and then averaged arithmetically (the 08 §5 baseline: the loss
// scalar is a host readback point anyway (D7) — device atomic float is forbidden, determinism comes
// first). Shared by both the CPU and Metal backends. n == 0 is forbidden (the validator guarantees
// H,W >= 11).
inline double ssim_reduce_mean_f64(const float* map, std::size_t n) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; i++) sum += static_cast<double>(map[i]);
    return sum / static_cast<double>(n);
}

// ssim forward CPU reference — written from scratch (the standard formulas of Wang et al. 2004 +
// consistency with the gsplat fused_ssim call contract, spec 08 §3; not derived from any code).
// The same f32 formulas as the Metal kernel (Ssim.metal), the same separable 2-pass
// (horizontal→vertical), the same fixed tap order (kSsimWindow is the single source of truth for
// those constants), and std::fma accumulation (paired with metal::fma — the lesson of 1a Task 4).
// With need_coeffs it also computes the three coefficient maps that bwd takes as input
// (Dμ/Dσ/Dc — 08 §4-(1,2)).
// There is no ssim backward CPU reference — it is compared directly against the autograd golden
// (ssim_basic v_img1) (the same rule as raster/sh bwd, decided in 00-index §6).
struct SsimInputs;      // defined in splatx/KernelBackend.h
struct SsimFwdOutputs;
SsimFwdOutputs ssim_fwd_cpu(const SsimInputs& in);

} // namespace splatx
