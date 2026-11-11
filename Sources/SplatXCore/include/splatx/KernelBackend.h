#pragma once
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>
#include "splatx/CpuKernels.h"

namespace splatx {

struct RegionMask;

// ===== Phase 1 kernel interface structs =====
// The field names are exactly the gsplat kernel argument names (D4; docs/kernel-specs/01, 02, 03,
// 06).
// Together with Sources/SplatXKernelSrc/KernelParams.h (the host/device shared parameters), this is
// the single source of truth — when a struct field changes, update both files together.

// Spec 06 fused_adam — in-place update, no bias correction (reproducing the upstream original
// exactly).
struct AdamInputs {
    float*         param      = nullptr; // [n_elems]
    const float*   grad       = nullptr; // [n_elems] (kernel argument name param_grad)
    float*         exp_avg    = nullptr; // [n_elems]
    float*         exp_avg_sq = nullptr; // [n_elems]
    const uint8_t* valid      = nullptr; // [n_gauss] 0/1 (D3 uchar contract), nullptr allowed
    int64_t n_elems = 0;                 // N*D (total number of parameter scalars)
    int64_t n_gauss = 0;                 // N (AdamParams.D = n_elems / n_gauss)
    float lr = 0.f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
};

// Spec 01 projection_ewa_3dgs_fused_fwd/bwd shared inputs.
// covars != nullptr → the covars (6-triu) path; nullptr → the quats/scales path (kProjUseCovars).
struct ProjectionFusedInputs {
    const float* means     = nullptr; // [N,3] world
    const float* covars    = nullptr; // [N,6] upper-triangular (c00,c01,c02,c11,c12,c22), optional
    const float* quats     = nullptr; // [N,4] (w,x,y,z)
    const float* scales    = nullptr; // [N,3] linear (after the exp activation)
    const float* opacities = nullptr; // [N] post-sigmoid, nullptr allowed (fwd only, unused in
                                      //  bwd — 01 Correction 1)
    const float* viewmats  = nullptr; // [C,16] world→cam row-major
    const float* Ks        = nullptr; // [C,9] row-major
    int64_t N = 0, C = 0;
    int width = 0, height = 0;
    float eps2d = 0.3f, near_plane = 0.01f, far_plane = 1e10f, radius_clip = 0.f;
    uint32_t camera_model = 0;        // 0=PINHOLE, 1=ORTHO, 2=FISHEYE (Phase 1 verifies PINHOLE only)
    bool calc_compensations = false;
};

struct ProjectionFusedOutputs {
    std::vector<int32_t> radii;       // [C,N,2] (<=0 = culled; the other outputs of a culled slot
                                      //  are unwritten — D3)
    std::vector<float> means2d;       // [C,N,2]
    std::vector<float> depths;        // [C,N]
    std::vector<float> conics;        // [C,N,3] inverse covariance, upper-triangular (a,b,c)
    std::vector<float> compensations; // [C,N] only when calc_compensations
};

// bwd additional inputs: the three fwd outputs + the upstream gradients.
struct ProjectionFusedGradsIn {
    const int32_t* radii         = nullptr; // [C,N,2] fwd output (the culling gate)
    const float* conics          = nullptr; // [C,N,3] fwd output
    const float* compensations   = nullptr; // [C,N] fwd output, only when calc_compensations
    const float* v_means2d       = nullptr; // [C,N,2]
    const float* v_depths        = nullptr; // [C,N] (Phase 1: a zero buffer)
    const float* v_conics        = nullptr; // [C,N,3]
    const float* v_compensations = nullptr; // [C,N] optional
    bool viewmats_grad = false;             // whether v_viewmats is computed (kProjViewmatsGrad)
};

struct ProjectionFusedGrads {
    std::vector<float> v_means;    // [N,3]
    std::vector<float> v_covars;   // [N,6] only on the covars path
    std::vector<float> v_quats;    // [N,4] only on the quat/scale path
    std::vector<float> v_scales;   // [N,3] only on the quat/scale path
    std::vector<float> v_viewmats; // [C,16] only when viewmats_grad
};

// Spec 02 sh_fwd/sh_bwd (standalone — the fused assemble is out of Phase 1 scope, D10-4).
struct ShInputs {
    const float* dirs    = nullptr; // [B*N,3] unnormalized (the kernel normalizes with rsqrt;
                                    //  unguarded at dir=0 — same as the original)
    const float* coeffs  = nullptr; // [N,K,D]
    const uint8_t* masks = nullptr; // [B*N] 0/1 (D3 uchar contract), nullptr allowed (kShHasMasks)
    uint32_t degrees_to_use = 0;    // 0..4
    int64_t B = 1, N = 0, K = 1;
    uint32_t D = 3;                 // channel count (RGB=3)
    bool compute_v_dirs = false;    // bwd only (kShComputeVDirs)
};

struct ShOutputs {
    std::vector<float> colors; // [B*N,D] (+0.5/clamp/relu_mask are host post-ops — outside the kernel)
};

struct ShGrads {
    std::vector<float> v_coeffs; // [N,K,D] (always accumulated in fp32)
    std::vector<float> v_dirs;   // [B*N,3] only when compute_v_dirs
};

// Spec 03 intersect_tile inputs — dense only (disposition #14: packed/tile_mask are excluded,
// n_elements=I*N).
// The path is selected by the presence of conics/opacities, exactly as in the upstream original
// (IntersectTile.cu:288 `conics != nullptr && opacities != nullptr`): giving both selects AccuTile
// (kIsectAccuTile=true — SNUGBOX + sweep); giving neither falls back to AABB. Giving only one is
// rejected by the validator (a combination the original never produces).
// tile_n_bits/image_n_bits are not supplied by the caller — the backend computes them with
// bits_for_count (the single source of truth in CpuKernels.h) and exposes them in Outputs
// (pack/unpack consistency).
struct IsectTilesInputs {
    const float*   means2d   = nullptr; // [I,N,2] proj fwd output
    const int32_t* radii     = nullptr; // [I,N,2] int32 (<=0 = the culling gate — D3)
    const float*   depths    = nullptr; // [I,N] (non-culled slots are >= 0 — the premise of key
                                        //  monotonicity)
    const float*   conics    = nullptr; // [I,N,3] Σ⁻¹ upper-triangular (AccuTile only, stride 12B)
    const float*   opacities = nullptr; // [I,N] post-sigmoid (AccuTile only — the level-set t)
    int64_t I = 0, N = 0;
    uint32_t tile_size = 0, tile_width = 0, tile_height = 0;
};

// isect_ids/flatten_ids are the raw outputs in emit order (before sorting) — the sort (Tasks 6/7)
// and the offset encode (Task 4) are downstream operations. The emit order convention
// (deterministic): ascending element idx (= image → Gaussian) → ascending tile y → ascending
// tile x (the AABB loop order at IntersectTile.cu:435–461).
struct IsectTilesOutputs {
    std::vector<int32_t>  tiles_per_gauss; // [I,N] 0 where culled (radii<=0)
    std::vector<uint64_t> isect_ids;       // [n_isects] 64-bit keys — treated as unsigned (D3 ulong contract)
    std::vector<int32_t>  flatten_ids;     // [n_isects] global flattened index (IntersectTile.cu:459)
    int64_t n_isects = 0;                  // = cum_tiles_per_gauss[-1] (host readback, D9-3)
    uint32_t tile_n_bits = 0;              // bits_for_count(tile_width*tile_height)
    uint32_t image_n_bits = 0;             // bits_for_count(I) — effective sort bits = 32+tile+image
};

// Spec 03 §sort contract — sorts (u64 key, i32 value) pairs by ascending unsigned key, **stably**
// (corresponding to cub::DeviceRadixSort::SortPairs, Intersect.cpp:299–324). The keys are treated
// as ulong (D3 — an int64 signed comparison collapses the order once bit63 is set, in the extreme
// case image_n_bits+tile_n_bits==32).
// stable = ties preserve the input (emit) order — the premise for bit-exact agreement with the
// isect_ties sort goldens (task-2 report; np.lexsort == std::stable_sort as the equivalence
// criterion).
// key_n_bits = the effective sort bit width (32 + tile_n_bits + image_n_bits in the isect pipeline)
// — every bit above it MUST be 0 (the encoding contract; the CPU reference throws on a violation,
// while the GPU radix performs no check — the same pattern as the substitute depth-validation rule
// of spec 03 §kernel 3). The CPU compares the full u64 (equivalent under the premise that the high
// bits are 0), and the Task 7 GPU radix uses it to cut the number of passes (sorting only the low
// key_n_bits).
struct SortIsectInputs {
    const uint64_t* keys   = nullptr; // [n] isect_ids — raw, in emit order (isect_tiles output)
    const int32_t*  values = nullptr; // [n] flatten_ids
    int64_t n = 0;                    // n_isects (validator: int32 range — Task 7 GPU params uint)
    uint32_t key_n_bits = 64;         // default 64 = all bits effective (the safe default)
};

struct SortIsectOutputs {
    std::vector<uint64_t> keys_sorted;   // [n] — isect_ids_sorted (input to offset_encode)
    std::vector<int32_t>  values_sorted; // [n] — flatten_ids_sorted (consumed by the rasterizer)
};

// Spec 03 isect_offset_encode inputs — builds the per-tile start offsets
// offsets[I*tile_height*tile_width] (int32) from the **already sorted** isect_ids (u64, downstream
// of the Task 6/7 sort). tile_n_bits is not taken as an argument — the backend recomputes it with
// bits_for_count (the single source of truth) (pack/unpack consistency with the isect_tiles
// encoding, spec 03 §1). Consumption contract: the intersection range of tile t (a flattened index)
// = [offsets[t], t+1 < I*n_tiles ? offsets[t+1] : n_isects) — traversed by the rasterizer (1c).
struct IsectOffsetInputs {
    const uint64_t* isect_ids = nullptr; // [n_isects] sorted 64-bit keys (D3 ulong contract)
    int64_t n_isects = 0;
    int64_t I = 0;                       // image count (offset table [I, tile_h, tile_w])
    uint32_t tile_width = 0, tile_height = 0;
};

// Spec 05 rasterize_to_pixels_3dgs_fwd inputs — the Phase 1 instantiation is the single
// TILE=16/CTA=256/PPT=1/CDIM=3 combination (the tile_size==16 combination of the gsplat launcher at
// Fwd.cu:401-412; tile_size != 16 is rejected by the validator). Dense only (packed is excluded —
// the same as 1b disposition #14): because flatten_ids is an [I*N] global flattened index, the
// gaussian inputs are addressed through the whole flattened array with no per-image pointer advance
// (spec 05 supplement 1). isect_offsets, the three outputs, backgrounds, and masks do advance per
// image (Fwd.cu:92-103).
struct RasterizeFwdInputs {
    const float*   means2d       = nullptr; // [I,N,2]
    const float*   conics        = nullptr; // [I,N,3] Σ⁻¹ upper-triangular (A,B,C)
    const float*   colors        = nullptr; // [I,N,3] (CDIM=3)
    const float*   opacities     = nullptr; // [I,N] post-sigmoid
    const float*   backgrounds   = nullptr; // [I,3] optional (kRastHasBackground)
    const uint8_t* masks         = nullptr; // [I,TH,TW] 0/1 (D3 uchar; kRastHasMasks —
                                            //  not verified in v1, 07 Correction 6: no fixture)
    const int32_t* isect_offsets = nullptr; // [I,TH,TW] global cumulative (isect_offset_encode output)
    const int32_t* flatten_ids   = nullptr; // [n_isects] sorted (sort_isect output)
    int64_t I = 0, N = 0, n_isects = 0;
    uint32_t image_width = 0, image_height = 0;
    uint32_t tile_size = 16, tile_width = 0, tile_height = 0;
    // Optional expected-depth channel (kRastHasDepth): per-Gaussian camera z (projection depths),
    // composited like a colour. nullptr renders RGB only, through the unchanged kernel variant.
    const float*   depths        = nullptr; // [I,N]
};

// The three outputs (the final writes at Fwd.cu:274-296). last_ids is not a gaussian id but the
// **absolute index into the sorted isect list** (batch_start + t, Fwd.cu:261), and 0 when there is
// no contribution (a dual-purpose encoding — because of the MAX_ALPHA=0.99 clamp the first gaussian
// cannot cross the threshold, so contamination is impossible; spec 05 second verification).
// bwd (Task 4) consumes render_alphas/last_ids as inputs.
struct RasterizeFwdOutputs {
    std::vector<float>   render_colors; // [I,H,W,3]
    std::vector<float>   render_alphas; // [I,H,W,1] = 1 - T_final
    std::vector<int32_t> last_ids;      // [I,H,W]
    // Σ w·z per pixel, only when RasterizeFwdInputs.depths is set (else empty). The expected
    // depth is render_depths / render_alphas (gsplat RGB+ED normalises outside the kernel).
    std::vector<float>   render_depths; // [I,H,W]
};

// Spec 05 rasterize_to_pixels_3dgs_bwd additional inputs (Bwd.cu:62-68) — the two fwd outputs +
// the upstream gradients. Tests inject the fixture **golden** render_alphas/last_ids as the fwd
// outputs (blocking error propagation from the Metal fwd — 07 / task-2 §4). absgrad selects whether
// to output v_means2d_abs, a separate accumulation of |v_xy| (the AbsGS densification heuristic,
// Bwd.cu:269-272) — the kRastHasAbsGrad function constant. It is off for v1 verification (07 open
// question #3 — there is no autograd golden).
struct RasterizeBwdGradsIn {
    const float*   render_alphas   = nullptr; // [I,H,W,1] fwd output (for restoring T_final)
    const int32_t* last_ids        = nullptr; // [I,H,W] fwd output (absolute isect index)
    const float*   v_render_colors = nullptr; // [I,H,W,3]
    const float*   v_render_alphas = nullptr; // [I,H,W,1]
    bool absgrad = false;                     // kRastHasAbsGrad
    // Required exactly when RasterizeFwdInputs.depths is set — the upstream gradient of
    // render_depths (the Σ w·z channel, not the normalised expected depth).
    const float*   v_render_depths = nullptr; // [I,H,W]
};

// Gradient outputs — all [I,N,·] flattened, with **no per-image pointer advance** (flatten_ids is
// an [I*N] global flattened index — spec 05 supplement 1). If n_isects==0 they are all 0 (the
// launcher skip at Bwd.cu:365-369 — zeros_like semantics). This is an atomic accumulation path, so
// it is not bit-exact deterministic — verification uses the meta tolerances + two-run
// self-consistency rel_l2 (07 §4).
struct RasterizeBwdGrads {
    std::vector<float> v_means2d;     // [I,N,2]
    std::vector<float> v_conics;      // [I,N,3]
    std::vector<float> v_colors;      // [I,N,3]
    std::vector<float> v_opacities;   // [I,N]
    std::vector<float> v_means2d_abs; // [I,N,2] only when absgrad (otherwise an empty vector)
    std::vector<float> v_depths;      // [I,N] only with the depth channel (feeds proj bwd v_depths)
};

// Spec 06 rng_normal_f32 / rng_philox_u32 — the Philox4x32-10 counter-based RNG (approved as
// disposition #24 in 00-index §6.1; written from scratch — see the literature comments in
// RngPhilox.h).
// Stateless: counter = (thread index, offset_lo, offset_hi, 0), key = seed — reproducible from
// (seed, offset, index) alone. offset serves as the training step counter (separating the stream
// per step).
struct RngNormalInputs {
    uint64_t seed = 0;   // key (seed_lo, seed_hi)
    uint64_t offset = 0; // the high 2 words of the counter — the step counter
    int64_t count = 0;   // number of output elements (uint32 range — RngParams.count)
};

// Spec 06 mcmc_relocation (MCMC Eq.9, RelocationCUDA.cu:35–79) — when a dead Gaussian is
// teleported, this produces the new opacity/scale that preserve the render invariant after it is
// duplicated n times. Preprocessing the inputs (the sigmoid/exp activations, clamping ratios to
// 1..n_max and casting to int) is the host's responsibility (relocation.py:55–66 — the kernel has
// no guard; only the CPU reference promotes a range violation to a validating throw). binoms is the
// LUT precomputed by make_binoms_lut().
struct RelocationInputs {
    const float*   opacities = nullptr; // [N] post-sigmoid
    const float*   scales    = nullptr; // [N,3] linear (after the exp activation)
    const int32_t* ratios    = nullptr; // [N] duplication count, pre-clamped to 1..n_max (int32)
    const float*   binoms    = nullptr; // [n_max*n_max] row-major LUT (float)
    int64_t N = 0;
    int32_t n_max = 51;                 // strategy/mcmc.py:95
    float min_opacity = 0.005f;         // strategy/mcmc.py:88
};

struct RelocationOutputs {
    std::vector<float> new_opacities; // [N]
    std::vector<float> new_scales;    // [N,3]
};

// Spec 06 mcmc_perturb_positions (MCMCPerturbCUDA.cu:27–61) — positions in-place.
// noise is an **input buffer** (the kernel does not generate random numbers — the original also
// pre-generates it with torch.randn_like, ops.py:438; with the noise injected, the kernel is fully
// deterministic). The producer is rng_normal_f32.
struct PerturbInputs {
    float*       positions       = nullptr; // [N,3] modified in-place
    const float* quats           = nullptr; // [N,4] wxyz, pre-activation (the kernel normalizes)
    const float* scales_log      = nullptr; // [N,3] log-scale (the kernel applies exp)
    const float* opacities_logit = nullptr; // [N] logit (the kernel applies sigmoid)
    const float* noise           = nullptr; // [N,3] standard normal
    int64_t N = 0;
    float noise_scale = 0.f; // lr * noise_lr (strategy/mcmc.py:190)
    float t = 0.005f;        // DEFAULT_MCMC_OPACITY_T (strategy/ops.py:35)
    float k = 100.f;         // DEFAULT_MCMC_OPACITY_K (strategy/ops.py:36)
};

// Spec 08 ssim_fwd/ssim_bwd shared inputs — an independently designed implementation of the gsplat
// fused_ssim call contract (padding=valid, a fixed 11×11 Gaussian with σ=1.5, C1=1e-4/C2=9e-4, and
// the gradient for img1 only — v_img2 is not implemented, 08 §2). The image layout is [I,H,W,C]
// (the render_colors D6 convention — an NCHW transposed buffer is forbidden; this is handled by the
// kernel indexing). The Phase 3 verification scope is I=1, C=3 (the indexing expressions are the
// general form — D3).
struct SsimInputs {
    const float* img1 = nullptr; // [I,H,W,C] the render side — what is differentiated
    const float* img2 = nullptr; // [I,H,W,C] GT
    int64_t I = 0, H = 0, W = 0, C = 0; // H, W >= 11 required (the validator throws — with an empty
                                        // map the mean is undefined; gsplat gives NaN, which cannot
                                        // occur under training conditions)
    bool need_coeffs = false;    // training mode — also outputs the three coefficient maps
                                 //  (Dμ/Dσ/Dc) that bwd takes as input
};

// fwd outputs — ssim_map[p] corresponds to window **start** p (center p+5), with size
// [I,H-10,W-10,C] (channel-last).
// The ssim scalar = the arithmetic mean of the map (M = I·C·(H-10)·(W-10)) — for determinism
// **the host sums the map sequentially in ascending index order in f64** (ssim_reduce_mean_f64 is
// the single source of truth; the 08 §5 baseline — device atomic float accumulation is forbidden).
// loss = 1 - ssim, and the λ=0.2 combination is owned by Trainer (Task 3).
struct SsimFwdOutputs {
    double ssim = 0.0;              // the f64 host-summed scalar
    std::vector<float> ssim_map;    // [I,H',W',C]
    std::vector<float> coeff_mu;    // [I,H',W',C] Dμ — only when need_coeffs (otherwise empty)
    std::vector<float> coeff_sigma; // [I,H',W',C] Dσ
    std::vector<float> coeff_cross; // [I,H',W',C] Dc
};

// bwd additional inputs — the three coefficient maps saved by fwd (need_coeffs=true) + the uniform
// upstream scalar.
// v_img1(q) = (-v_loss/M)·[ (w⊛Dμ)(q) + 2·x(q)·(w⊛Dσ)(q) + y(q)·(w⊛Dc)(q) ] (08 §4-(3);
// ⊛ = the adjoint of the valid correlation = a zero-padded full conv of the coefficient maps).
// v_loss = d(total)/d(1-ssim) — the golden fixture uses 1 (pure d(1-ssim)/d(img1)), and the λ=0.2
// scaling is owned by the caller (Trainer).
struct SsimGradsIn {
    const float* coeff_mu    = nullptr; // [I,H',W',C] fwd output
    const float* coeff_sigma = nullptr;
    const float* coeff_cross = nullptr;
    float v_loss = 1.f;
};

// ===== D11 device-resident train step (Phase 3 Task 5 — 00-index §7) =====
// An executor that keeps the 6 parameter groups + the 12 Adam moment buffers + the intermediate
// buffers device-resident and encodes the D7 step into a small number of command buffers.
// **The per-kernel methods are retained permanently as the verification surface** — this API is an
// addition alongside them, and the reference path for the fixtures and for parity is the host
// vector path (decided in D11).
//
// Correctness contract (the resident comparison test in TrainerTests is the gate):
// - Every kernel it assembles is one of the existing verified kernels + the small elementwise
//   kernels of TrainStep.metal (written from scratch — the formulas and their order reproduce the
//   Trainer.cpp reference).
// - The sources of deviation from the host path are (1) libm ↔ precise:: in the exp/sigmoid
//   activations (the same contract as the perturb kernel — a tolerance) and (2) the accumulation
//   order of the atomic bwd kernels (at the level of the existing self-consistency).
//   Inside the resident path, the whole forward is bit-exact deterministic.
// - Loss reporting keeps the host f64 summation convention: render/ssim_map are read directly from
//   the UMA shared buffers and summed with the same expressions as l1_loss_and_grad and
//   ssim_reduce_mean_f64 (the gradients come from GPU kernels).
struct TrainStepExecutorDesc {
    int64_t arena_rows = 0; // parameter row count (MCMC: cap_max — never reallocated between steps)
    int sh_degree = 3;      // K_total = (sh_degree+1)^2
    // Fixed projection/adam/loss hyperparameters (corresponding to TrainConfig — invariant per step)
    float eps2d = 0.3f, near_plane = 0.01f, far_plane = 1e10f, radius_clip = 0.f;
    float b1 = 0.9f, b2 = 0.999f, eps = 1e-15f;
    float ssim_weight = 0.2f; // λ — 0 bypasses the SSIM kernels and buffers entirely
    bool antialiased = false; // kProjCalcComp + the opacity_aa chain (the same point as Trainer.cpp)
};

// The resident parameter synchronization view — the group order is the D3 reference:
//   0 means[3], 1 quats[4], 2 scales[3] (log), 3 opacities[1] (logit), 4 sh0[3],
//   5 shN[(K-1)*3] (dim 0 when sh_degree is 0 — nullptr allowed).
// rows: the number of rows to synchronize. **The first upload covers the whole arena** (establishing
// the zero tail on the device — the MCMC arena contract); afterwards count rows are enough (on both
// device and host, everything beyond count rows is untouched).
struct ResidentParamSync {
    float* params[6] = {};
    float* exp_avg[6] = {};
    float* exp_avg_sq[6] = {};
    int64_t rows = 0;
};

struct ResidentStepInputs {
    const float* viewmat = nullptr; // [16] world→cam row-major
    const float* K = nullptr;       // [9]
    int width = 0, height = 0;
    const float* gt_rgb = nullptr;  // [H,W,3] — not needed when forward_only (nullptr allowed)
    int64_t count = 0;              // the current alive N (<= arena_rows)
    uint32_t degrees_to_use = 0;
    float lr[6] = {};               // D3 group order (the means lr already reflects the schedule)
    // Regularization weights (the w of the host add_*_reg_grad_inplace — reg/denominator when
    // reg > 0, otherwise 0)
    float opacity_reg_w = 0.f, scale_reg_w = 0.f;
    bool forward_only = false;      // for render() — skips loss, bwd, and adam entirely
    bool want_render = false;       // whether to read render_colors back
    // MCMC noise injection (false on refine steps, which are host-owned — Trainer decides)
    bool inject_noise = false;
    uint64_t noise_seed = 0, noise_offset = 0;
    float noise_scale = 0.f;
    // Depth supervision: with gt_depth set and depth_weight > 0 (training steps only) the
    // inverse-depth L1 joins the loss; want_depth alone renders the expected depth (eval).
    const float* gt_depth = nullptr; // [depth_height,depth_width] metres, 0 = invalid
    int depth_width = 0, depth_height = 0;
    int64_t depth_valid = 0;         // valid sidecar samples over the view (the loss denominator)
    float depth_weight = 0.f;
    bool want_depth = false;
    // Region mask (RegionMask.h; training steps only): with mask set the L1, SSIM and depth terms
    // are masked (kTrainHasMask) and normalised by mask_count (L1), mask_ssim_count (SSIM) and
    // depth_valid (the depth-valid ∩ in-mask count). nullptr = unmasked, the previous kernels.
    const RegionMask* mask = nullptr;
    int64_t mask_count = 0;         // M: in-mask view pixels
    int64_t mask_ssim_count = 0;     // Ms: in-mask valid SSIM map pixels
    // Shape regulariser (train_scale_shape_reg): flatten_weight/N and needle_weight/N, 0 = off.
    float flatten_w = 0.f, needle_w = 0.f, needle_ratio = 5.f;
};

struct ResidentStepOutputs {
    float loss = 0.f;     // 0 when forward_only
    int64_t n_isects = 0; // the D9-3 sync point readback value
    std::vector<float> render_colors; // filled only when want_render (otherwise an empty vector)
    float depth_loss = 0.f;           // the unweighted inverse-depth L1 (0 when not supervised)
    std::vector<float> render_depths; // [H,W] expected depth (0 below kDepthMinAlpha), only when
                                      // the depth channel ran and want_render or want_depth
};

class TrainStepExecutor {
public:
    virtual ~TrainStepExecutor() = default;
    virtual void upload_params(const ResidentParamSync&) = 0;   // host → device
    virtual void download_params(const ResidentParamSync&) = 0; // device → host (writes back)
    virtual ResidentStepOutputs step(const ResidentStepInputs&) = 0;
};

// The kernel invocation abstraction layer. Phase 1: MetalBackend; from v2 on: CudaBackend.
// Every backend MUST pass the numerical-agreement tests against the CPU reference.
class KernelBackend {
public:
    virtual ~KernelBackend() = default;
    virtual const char* name() const = 0;
    virtual ProjectionOutputs projection_ewa_fwd(const ProjectionInputs& in) = 0;

    // ===== Phase 1 extensions — the default implementation throws; each backend
    // implements only what it needs =====
    virtual void fused_adam(const AdamInputs&) { // in-place
        throw std::runtime_error("fused_adam: unimplemented");
    }
    virtual ProjectionFusedOutputs projection_fused_fwd(const ProjectionFusedInputs&) {
        throw std::runtime_error("projection_fused_fwd: unimplemented");
    }
    virtual ProjectionFusedGrads projection_fused_bwd(const ProjectionFusedInputs&,
                                                      const ProjectionFusedGradsIn&) {
        throw std::runtime_error("projection_fused_bwd: unimplemented");
    }
    virtual ShOutputs sh_fwd(const ShInputs&) {
        throw std::runtime_error("sh_fwd: unimplemented");
    }
    virtual ShGrads sh_bwd(const ShInputs&, const float* /*v_colors [B*N,D]*/) {
        throw std::runtime_error("sh_bwd: unimplemented");
    }
    virtual IsectTilesOutputs isect_tiles(const IsectTilesInputs&) {
        throw std::runtime_error("isect_tiles: unimplemented");
    }
    // The prefix scan primitive (spec 03 §kernel 2) — tiles_per_gauss(int32) → inclusive
    // cumsum(int64), corresponding to at::cumsum (int64 explicitly, Intersect.cpp:258). It is
    // exposed so that each backend's implementation of the cumsum inside isect_tiles can be
    // verified directly (ScanTests — synthetic large arrays).
    // CPU: delegates to the host cumsum_i64; Metal: a 3-dispatch reduce-then-scan (Task 4).
    virtual std::vector<int64_t> prefix_scan_i64(const int32_t* /*in [n]*/, int64_t /*n*/) {
        throw std::runtime_error("prefix_scan_i64: unimplemented");
    }
    // Spec 03 §sort (the replacement for cub::DeviceRadixSort::SortPairs) — CPU: an index-based
    // std::stable_sort (Task 6); Metal: a GPU radix sort (Task 7 — the spec 04 brush port,
    // RadixSort.metal with 5 kernels, pass count = ceil(key_n_bits/4)). The sort contract is
    // common to both backends.
    virtual SortIsectOutputs sort_isect(const SortIsectInputs&) {
        throw std::runtime_error("sort_isect: unimplemented");
    }
    // Spec 03 §kernel 4 — returns offsets[I*tile_height*tile_width] (int32).
    // If n_isects==0 they are all 0 (corresponding to the upstream offsets.fill_(0),
    // IntersectTile.cu:1004–1008).
    virtual std::vector<int32_t> isect_offset_encode(const IsectOffsetInputs&) {
        throw std::runtime_error("isect_offset_encode: unimplemented");
    }
    // Spec 05 rasterize forward — one threadgroup per tile, front-to-back alpha blending.
    // The accumulation is sequential per pixel (no atomics), so bit-exact determinism is the
    // contract on both CPU and Metal (memcmp across two runs).
    // Note: the dispatch MUST NOT be skipped even when n_isects==0 — the kernel writes the output
    // of every in-bounds pixel (including the background of empty tiles) (Fwd.cu:274-296; the
    // 07 Correction 4 path).
    virtual RasterizeFwdOutputs rasterize_fwd(const RasterizeFwdInputs&) {
        throw std::runtime_error("rasterize_fwd: unimplemented");
    }
    // Spec 05 rasterize backward — a back-to-front reverse traversal (recomputing T) + a simdgroup
    // reduction + an atomicAdd from lane 0. There is no CPU reference (direct comparison against
    // the autograd golden — the 1a rule): Metal only.
    // Zero-initialization of the gradient buffers is built into the backend (encode_zero_fill,
    // D8-1).
    virtual RasterizeBwdGrads rasterize_bwd(const RasterizeFwdInputs&,
                                            const RasterizeBwdGradsIn&) {
        throw std::runtime_error("rasterize_bwd: unimplemented");
    }
    // Spec 06 rng_normal_f32 — Philox4x32-10 → uniform (0,1] → Box-Muller standard normal [count].
    // The philox integer stages and the uniform conversion are contractually bit-identical between
    // CPU and Metal (RngPhilox.h is the single source of truth); only the normal transform
    // (log/sqrt/sin/cos) goes through the transcendental rel gate (06 Correction 5).
    virtual std::vector<float> rng_normal_f32(const RngNormalInputs&) {
        throw std::runtime_error("rng_normal_f32: unimplemented");
    }
    // Exposes the philox integer stages as [count]×u32 — the verification surface for the
    // bit-identical contract (McmcTests only; it uses the same (tid, offset, seed) mapping as
    // rng_normal_f32, so the streams correspond 1:1).
    virtual std::vector<uint32_t> rng_philox_u32(const RngNormalInputs&) {
        throw std::runtime_error("rng_philox_u32: unimplemented");
    }
    // Spec 06 mcmc_relocation — returns new_opacities/new_scales (host post-processing — the
    // logit/log back-conversion and the parameter rearrangement are owned by Trainer,
    // strategy/ops.py:337–347).
    virtual RelocationOutputs mcmc_relocation(const RelocationInputs&) {
        throw std::runtime_error("mcmc_relocation: unimplemented");
    }
    // Spec 06 mcmc_perturb_positions — positions in-place (noise is an input buffer —
    // deterministic).
    virtual void mcmc_perturb_positions(const PerturbInputs&) {
        throw std::runtime_error("mcmc_perturb_positions: unimplemented");
    }
    // Spec 08 ssim forward — a separable 11-tap conv in 2 passes (with a threadgroup apron cache),
    // one thread per map pixel. No atomics → bit-exact determinism is the contract (memcmp across
    // two runs). The CPU reference (ssim_fwd_cpu) is a sequential implementation with the same f32
    // formulas and accumulation order (fma pairs) — the golden is fixtures/ssim_basic (generated
    // in f64).
    virtual SsimFwdOutputs ssim_fwd(const SsimInputs&) {
        throw std::runtime_error("ssim_fwd: unimplemented");
    }
    // Spec 08 ssim backward — a zero-padded scatter conv (the adjoint) over the three coefficient
    // maps + a pointwise combination, one thread per input pixel (v_img1 [I,H,W,C] is written for
    // every pixel — no zero-fill and no atomics needed, bit-exact deterministic). There is no CPU
    // reference — it is compared directly against the autograd golden (ssim_basic v_img1) (the
    // raster bwd rule).
    virtual std::vector<float> ssim_bwd(const SsimInputs&, const SsimGradsIn&) {
        throw std::runtime_error("ssim_bwd: unimplemented");
    }
    // The D11 device-resident step executor (Phase 3 Task 5) — a backend that does not support it
    // returns nullptr (not a throw): Trainer then falls back to the host round-trip path (the CPU
    // backend, for instance).
    virtual std::unique_ptr<TrainStepExecutor>
    make_train_step_executor(const TrainStepExecutorDesc&) {
        return nullptr;
    }
};

std::unique_ptr<KernelBackend> make_cpu_backend();

} // namespace splatx
