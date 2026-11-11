// Trainer — Gaussian initialization (init_splats), L1 loss/grad, activation helpers, and the
// training loop (Task 7).
// (The initialization recipe reproduces the formulas of the 3DGS/gsplat convention — it is not
// derived from their code. The training recipe values (LR and schedules) follow gsplat
// examples/simple_trainer.py — derivation notice in docs/provenance.md.)
#pragma once
#include "splatx/AdamState.h"
#include "splatx/Checkpoint.h"
#include "splatx/DepthSupervision.h"
#include "splatx/KernelBackend.h"
#include "splatx/RegionMask.h"
#include "splatx/SplatCloud.h"
#include "splatx/TrainingInput.h"
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace splatx {

// The SH degree-0 basis constant C0 = 1/(2·√π). RGB(0..1) → SH DC inverse transform:
// sh0 = (c - 0.5) / C0 (the inverse of the render-side post-op "+0.5" — the 02 SH spec, matching
// the gsplat rgb_to_sh formula).
inline constexpr float kShC0 = 0.28209479177387814f;

// ===== Initialization =====
// Builds a SplatCloud from TrainingInput.seed_points/seed_rgb:
// - Subsample: if M > max_points, a uniform stride (pick(i) = i·M/N, N = min(M, max_points)).
//   (The max_points cap is about controlling memory and training scale — since Phase 6 round 2 the
//   KNN uses a uniform grid, so the cap is not a premise of the KNN complexity. Default 50k —
//   CLI --max-points.)
// - means = the selected points as they are, sh0 = (rgb/255 - 0.5)/C0, shN = 0,
//   quats = identity (w=1,x=y=z=0), opacities = logit(init_opacity) (stored in logit-space — the
//   SplatCloud convention; gsplat init_opa is 0.1, revisited as 0.5 in the Phase 2 MCMC preset).
// - scales = **log(the mean 3-NN distance × init_scale)** (isotropic, all 3 components equal):
//   the self-excluded 3-NN (knn3_mean_dists — below, an **exact** uniform-grid algorithm), with
//   neighbor count k = min(3, N-1); a mean distance of 0 (duplicate points) or N==1 is clamped to
//   1e-6 before the multiplication and the log.
//   init_scale corresponds to gsplat simple_trainer.py:323 `log(dist_avg·init_scale)` — it is a
//   **multiplier** applied to the 3-NN-based distance, not a fixed scale (the MCMC preset uses
//   0.1 — see below).
// Input violations (empty seed, rgb size mismatch, opacity ∉ (0,1), init_scale <= 0,
// sh_degree ∉ [0,3], max_points < 1) throw.
struct InitOptions {
    int64_t max_points = 50000;
    float init_opacity = 0.1f; // gsplat simple_trainer.py:167 init_opa
    float init_scale = 1.0f;   // gsplat simple_trainer.py:169 init_scale — 3-NN distance multiplier
    int sh_degree = 3;
};

SplatCloud init_splats(const TrainingInput& input, const InitOptions& opts);

// Computes the self-excluded 3-NN mean distance for every point in one batch (Phase 6 round 2
// Task 1 — replacing the O(N²) bottleneck that took 187s on A17 at 138k points). pts = [n,3]
// row-major, returns [n] (element i = the mean Euclidean distance to point i's k = min(3, n-1)
// neighbors; 0 when n == 1 — clamping is the caller's responsibility).
//
// A uniform-grid based **exact** algorithm (not an approximation — the distance values are
// bit-identical with brute force):
// - Cell size: over the d active axes of the per-axis 1–99% quantile box (robust to outliers),
//   (volume × target density / n)^(1/d) — the target is 0.5 points per cell (measured as the
//   fastest on garden — see the Trainer.cpp comments).
//   Points outside the quantile box are clamped into the boundary cells.
// - For each point: candidates are collected starting from its own cell (ring 0) and expanding by
//   Chebyshev ring r, continuing to expand while there are fewer than 3 candidates. Once ring r
//   completes, it stops if the third-ranked squared distance d3 < (r·cell)²·(1-1e-6).
// - Correctness argument: cell assignment uses the coordinates clamped into the grid box — the
//   clamp is a per-axis non-expanding projection (a contraction), so two points whose cell
//   Chebyshev distance is q have a **real distance ≥ (q-1)·cell**.
//   Hence an unscanned point in a ring > r has a real distance ≥ r·cell, and since the 1e-6 margin
//   absorbs the relative rounding error of the f32 squared distance (≤ ~5e-7), it cannot displace
//   d3 — the top 3 distance values are settled.
//   For ties (equal distance values), the mean is unchanged whichever neighbor is picked, and
//   candidate selection is fixed to the lexicographically smallest 3 by (squared distance, index)
//   — determinism independent of scan order.
// - Termination guarantee: once r reaches the ring that covers the whole grid (r_cover), it
//   terminates with every point scanned — equivalent to the brute-force fallback (there is no
//   infinite loop). If every point has identical coordinates (a fully degenerate bbox) it returns
//   all zeros early, and small inputs of n ≤ 256 use brute force without building a grid (both are
//   exact and give the same values).
// - The grid always covers the quantile box (there is no per-axis dims cap — if the total cell
//   count would exceed 4n, only the cells are enlarged). If coverage broke under extreme
//   anisotropy (1e6×1×1e-3, for example), it would cause mass boundary clamping and runaway ring
//   expansion (review fix 1 — the shell enumeration is also clipped to the grid's real bounds).
// A null pts (while n > 0), n < 0, or n > INT32_MAX/4 (the int32 index width of the 4n total-cell
// bound) throws.
std::vector<float> knn3_mean_dists(const float* pts, int64_t n);

// ===== L1 loss + upstream grad (host — performance-irrelevant, D11) =====
// loss = mean(|render - gt|) (the sum accumulates in double), n_elems = H·W·3 (I·H·W·3 when I>1).
// v_render_colors[i] = sign(render[i] - gt[i]) / n_elems — fed straight into
// RasterizeBwdGradsIn.v_render_colors of raster bwd. **sign(0) = 0** (the torch.sign convention —
// adopting 0 as the subgradient of |x|). null, or n_elems<=0, throws.
struct L1LossResult {
    float loss = 0.f;
    std::vector<float> v_render_colors; // [n_elems]
};

L1LossResult l1_loss_and_grad(const float* render, const float* gt, int64_t n_elems);

// The masked L1 (RegionMask.h): loss = Σ m(p)·|r−g| / (3M), v = sign(r−g)·m(p)/(3M) over a
// view_w × view_h RGB view, with M = mask_view_count. With an all-ones mask it equals
// l1_loss_and_grad bit for bit; M = 0 gives loss 0 and a zero gradient.
L1LossResult l1_loss_and_grad_masked(const float* render, const float* gt, int view_w, int view_h,
                                     const RegionMask& mask, int64_t mask_count);

// ===== PSNR + train/test split (Phase 2 Task 5 — the v0.1 gate measurement tools) =====
// psnr = 10·log10(1/mse), mse = mean((a-b)²) over all pixels and channels (accumulated in double)
// — this presumes 0..1 normalized images. Identical to the gsplat evaluation definition
// (simple_trainer.py:611 torchmetrics PeakSignalNoiseRatio(data_range=1.0) — the MSE is the mean
// over every element of the image). mse == 0 gives +inf.
// **It does not clamp** — the caller clamps render to [0,1] before calling
// (corresponding to simple_trainer.py:1240 `torch.clamp(colors, 0, 1)`). null, or n_elems<=0,
// throws.
float psnr(const float* a, const float* b, int64_t n_elems);
// The PSNR over the in-mask pixels of an RGB view (all three channels, f64 sum). NaN when the mask
// has no in-mask pixel; with an all-ones mask it equals psnr.
float psnr_masked(const float* a, const float* b, int view_w, int view_h, const RegionMask& mask);

// ===== color_correct (eval only — cc_psnr; spec 2026-07-15-color-correct-eval §3) =====
// An exact port of color_correct_affine from gsplat gsplat/color_correct.py:98–149 (commit
// 77ab983, derivation notice in docs/provenance.md). A per-channel (RGB) independent scalar
// affine — there is no mixing between channels:
//   var_c = max(mean(ref_c²) − mean(ref_c)², 1e-8)                  (zero-denominator guard)
//   a_c   = (mean(ref_c·img_c) − mean(ref_c)·mean(img_c)) / var_c   (fits a·ref+b = img)
//   b_c   = mean(img_c) − a_c·mean(ref_c)   ← **computed with a_c before it is replaced**
//                                             (the upstream operation order)
//   a_c   = |a_c| < 1e-8 ? 1 : a_c                                  (divide-by-zero guard)
//   out_c = clip((img_c − b_c) / a_c, 0, 1)                         (inverse mapping → aligned to ref)
// The guard constant 1e-8 and the [0,1] clip are the same as the reference. The layout is the same
// as psnr (interleaved RGB, n_elems = H·W·3, channel stride 3). The per-channel accumulation, a/b,
// and the inverse mapping are in double (consistent with the psnr policy), and the output is f32.
// out is caller-owned [n_elems] — a separate buffer is presumed (in-place is not contracted).
// A null img/ref/out, n_elems <= 0, or n_elems % 3 != 0 throws.
void color_correct_affine(const float* img, const float* ref, int64_t n_elems, float* out);

// ===== eval SSIM (eval only — ssim/cc_ssim; spec 2026-07-15-perceptual-eval §2–§3) =====
// A fwd-only scalar under the same convention as the training loss (valid padding — the map is
// [h-10,w-10,3], an 11×11 Gaussian with σ=1.5, C1=1e-4/C2=9e-4 with KernelConstants as the single
// source of truth, and the scalar = the mean of the map summed sequentially in f64 — the
// ssim_reduce_mean_f64 rule). The engine is chosen through the backend argument
// (training-speed Task 1 — spec 2026-07-21-train-speed-quickwins §②):
//   nullptr (default) → ssim_fwd_cpu (CpuKernels.h) — the previous path unchanged (the CPU oracle,
//     bit-exact deterministic)
//   non-null          → backend->ssim_fwd(need_coeffs=false) — the GPU when it is Metal (removing
//     the CPU-SSIM dominant term from the eval tick). It is the same golden-verified kernel with
//     the same f64 summation rule, so the convention is unchanged; numerically it may deviate from
//     the CPU at the level of the f32 conv accumulation order and precise:: rounding (repeated
//     calls are bit-exact deterministic within each engine) — gated by the comparison test
//     eval_ssim_backend_matches_cpu_oracle.
// The layout is the same as psnr (interleaved RGB [h,w,3], channel-last). It returns the mean SSIM
// over the valid map (double).
// **If h < 11 || w < 11 it returns a quiet NaN (it does not throw)** — the valid map is empty, so
// no measurement is possible. This guard is **before** the backend call (the backend validator
// (Validate.h) throws on the same condition, but eval is an observation rather than a contract, so
// it only reports NaN — an intentional difference; the caller (run_eval) excludes NaN views from
// the average).
// A null img/ref throws (consistent with the psnr/cc practice).
double eval_ssim(const float* img, const float* ref, int32_t h, int32_t w,
                 KernelBackend* backend = nullptr);

// train/test split (the MipNeRF360 evaluation convention — gsplat
// examples/datasets/colmap.py:456–461):
//   test  = { i : i % test_every == 0 },  train = { i : i % test_every != 0 }
// test_every defaults to 8 (colmap.py:128 / simple_trainer.py:98). test_every <= 0 turns the split
// off — everything is train and test is left empty. The indices presume the images are **sorted by
// image name** (colmap.py:194–199 — the gsplat Parser sorts by name and then indexes; the sorting
// is the caller's (TrainC's) responsibility). Both lists are ascending.
// Note: test_every == 1 leaves train empty, exactly as in the upstream original — the caller must
// reject it.
struct TrainTestSplit {
    std::vector<int64_t> train, test;
};
TrainTestSplit make_train_test_split(int64_t n, int test_every);

// ===== Activation helpers (the SplatCloud pre-activation internal representation ↔ kernel inputs)
// The kernel input convention (established in 1a — KernelBackend.h): proj/isect/raster take
// **linear scales (exp)** and **post-sigmoid opacities**. The internal representation is
// log/logit-space (SplatCloud.h).
// These use the host std::exp/log (the activation sites inside the Metal kernels use precise:: —
// a surface separate from D2).
float sigmoid_f32(float x); // 1/(1+exp(-x)) — the same formula as KernelCommon.h sigmoid_standard
float logit_f32(float p);   // the inverse of sigmoid: log(p/(1-p)); p outside (0,1) throws

std::vector<float> activate_scales(const float* log_scales, int64_t n);    // exp
std::vector<float> inv_activate_scales(const float* scales, int64_t n);    // log (scale <= 0 throw)
std::vector<float> activate_opacities(const float* logits, int64_t n);     // sigmoid
std::vector<float> inv_activate_opacities(const float* opacs, int64_t n);  // logit

// ===== MCMC densification strategy options (Phase 2 Task 3 — design §5) =====
// Derivation notice: reproduces the MCMCStrategy defaults of gsplat strategy/mcmc.py (commit
// 77ab983) — cap_max, noise_lr, the refine periods, and min_opacity = mcmc.py:82–88. Schedules are
// stored as **ratios**, never as absolute iterations (the design §4 discipline — preventing resume
// collapse): the gsplat defaults 500/25000/100 are converted to ratios against a total of 30k
// iterations. The actual boundaries are lround(total_iters·ratio), and every is max(1,·).
// Determinism: the seed fixes both multinomial (mt19937_64 — its output is fixed by the standard)
// and the noise (the Philox key, offset=iter) — the same seed with the same call order reproduces
// bit-exactly (McmcStrategyTests).
struct McmcOptions {
    bool enabled = false;
    int64_t cap_max = 1'000'000;                    // mcmc.py:82 — the fixed arena size (§5.2)
    float min_opacity = 0.005f;                     // mcmc.py:88 — dead = sigmoid(op) <= this value
    float refine_start_ratio = 500.0f / 30000.0f;   // mcmc.py:84 (from iter > start)
    float refine_stop_ratio = 25000.0f / 30000.0f;  // mcmc.py:85 (until iter < stop)
    float refine_every_ratio = 100.0f / 30000.0f;   // mcmc.py:87
    float noise_lr = 5e5f;                          // mcmc.py:83 — noise_scale = lr_means·this value
    uint64_t seed = 42;                             // seed for multinomial + the perturb noise
    // noise_injection_stop_iter (mcmc.py:56/86, default -1 = unlimited) is not exposed — with the
    // default kept, its gate (mcmc.py:179–185) is always true, so there is no behavioral difference
    // (judged Minor in the Task 3 review).
};

// ===== Training loop (Task 7 — the D7 kernel order, host round-trip chaining D11) =====
// Derivation notice: the LR and schedule values reproduce the recipe of gsplat
// examples/simple_trainer.py (commit 77ab983) (design document §4). Schedules are **ratio-based
// (progress fraction)**, never absolute iterations — preventing resume collapse.
struct TrainConfig {
    int total_iters = 300;      // schedule normalization denominator
                                //  (progress fraction p = iter/total_iters)
    float scene_scale = 1.0f;   // scene_scale_from_cameras() result — the means lr multiplier
    // Learning rates (design §4 — the gsplat simple_trainer recipe)
    float lr_means = 1.6e-4f;   // × scene_scale, 0.01^p exponential decay in the progress fraction p
    float lr_scales = 5e-3f;
    float lr_quats = 1e-3f;
    float lr_opacities = 5e-2f;
    float lr_sh0 = 2.5e-3f;
    float lr_shN = 2.5e-3f / 20.0f;
    float adam_b1 = 0.9f, adam_b2 = 0.999f, adam_eps = 1e-15f; // eps 1e-15 (design §4)
    // Gradual SH degree increase: the gsplat "+1 every 1000 steps out of 30000 steps" = an interval
    // of 1/30 of the progress fraction (ratio-based)
    float sh_degree_interval_ratio = 1.0f / 30.0f;
    // projection parameters (the gsplat defaults)
    float eps2d = 0.3f, near_plane = 0.01f, far_plane = 1e10f, radius_clip = 0.0f;
    // Regularization loss weights (Phase 2 Task 4 — gsplat simple_trainer.py:207–210 defaults to
    // 0.0, and at :993–996 loss += reg·mean(...) is added only when `> 0.0`; the MCMC preset sets
    // 0.01).
    // The loss definitions are at losses.py:675–700: opacity_reg_loss = sigmoid(opacities).mean()
    // ([N] — denominator N), scale_reg_loss = exp(log_scales).mean() ([N,3] over **every element**
    // — denominator N·3).
    // splatx does not recompute the scalar loss; it adds the analytic derivative directly to the
    // gradients just before adam (helpers below).
    float opacity_reg = 0.0f;
    float scale_reg = 0.0f;
    // The SSIM loss weight λ (Phase 3 Task 3 — reproducing gsplat simple_trainer.py:171
    // ssim_lambda=0.2 and :961 `torch.lerp(l1loss, 1-ssimloss, λ)` = (1-λ)·L1 + λ·(1-SSIM)):
    //   loss     = (1-λ)·L1 + λ·(1-ssim)     (the reported loss — excluding the reg terms keeps
    //                                         the P2 Task 4 decision)
    //   v_render = (1-λ)·v_L1 + λ·v_DSSIM    (λ·v_DSSIM is absorbed into the kernel G of
    //                                         ssim_bwd(v_loss=λ))
    // 0.0 bypasses the SSIM path (the fwd/bwd kernels) entirely — bit-identical to the previous
    // pure-L1 behavior (backward compatible).
    // > 0 requires H, W >= 11 (validate_ssim_inputs — the valid map must be non-empty).
    // A value ∉ [0,1] throws from the ctor.
    float ssim_weight = 0.2f;
    // Anti-aliasing (the Mip-Splatting 2D compensation, Phase 3 Task 4 — reproducing gsplat
    // rendering.py rasterize_mode="antialiased"): when true, proj fwd runs with
    // calc_compensations=true and outputs comp = √(det Σ′ / det(Σ′ + eps2d·I)) (kProjCalcComp —
    // verified in 1a); the host then forms opacity_aa = sigmoid(op)·comp, and **both isect and
    // raster** consume opacity_aa. The point of application is exactly the upstream one:
    // rendering.py:864–865 `opacities = opacities * compensations` sits **before** isect_tiles
    // (:884–896, the opacities= argument) — so the AccuTile level set is likewise based on the
    // post-AA opacity. bwd is the host product-rule chain (01 spec gap #1 — host-owned):
    // v_comp = v_op_aa·sigmoid(op) → injected into proj bwd as v_compensations, and
    // v_opacity = v_op_aa·comp → the existing σ′ chain. Default false — consistent with the gsplat
    // default "classic" (:438, the published bench condition). Standard PLY does not record whether
    // AA was used (see the viewer note in the README).
    bool antialiased = false;
    // device-resident train_step (Phase 3 Task 5 — D11 / 00-index §7): when true, the parameters,
    // moments, and intermediate buffers are kept device-resident and the step is encoded into 2
    // command buffers (the sync points = the n_isects readback + the host f64 loss summation).
    // If the backend does not support it (CPU, for instance), it automatically falls back to the
    // host round-trip path. false = the previous host round-trip chaining (the verification
    // surface — the CLI --no-resident escape hatch). Correctness contract: the two paths assemble
    // identically — the only sources of deviation are libm ↔ metal precise:: in the exp/sigmoid
    // activations (a tolerance) and the accumulation order of the atomic bwd kernels (the resident
    // comparison test in TrainerTests is the gate). The forward of render()/step() on the resident
    // path shares the resident kernel activations, so bit-exact consistency within the path (a
    // self-render loss of 0, and the like) is preserved.
    bool resident = true;
    // Depth supervision weight (DepthSupervision.h): a step given a DepthTarget adds
    // depth_weight·(inverse-depth L1 over the expected depth) to the loss. 0 = off — the RGB-only
    // kernels and the previous trajectory, bit for bit in the forward. Must be finite and >= 0.
    float depth_weight = 0.0f;
    // Shape regulariser (train_scale_shape_reg / add_shape_reg_grad_inplace): with the linear
    // scales sorted s1 ≥ s2 ≥ s3 per Gaussian, loss += (flatten_weight/N)·Σ s3 (flatten toward a
    // surface disc) + (needle_weight/N)·Σ max(0, s1/s2 − needle_ratio) (the hinge reads s1/s2, never
    // s1/s3, so it does not fight the flatten term). Added to the gradients only, never to the
    // reported loss. Weights 0 = off (the previous trajectory); finite, >= 0; needle_ratio >= 1.
    float flatten_weight = 0.0f;
    float needle_weight = 0.0f;
    float needle_ratio = 5.0f;
    // Whether the dataset carries region masks (RegionMask.h). The mask itself arrives per step;
    // this only enters the checkpoint fingerprint (TrainC sets it from the sidecars).
    bool has_mask = false;
    // MCMC densification (Phase 2 Task 3 — with enabled=false this is the same fixed-N training
    // as 1c)
    McmcOptions mcmc;
};

// The MCMC preset (reproducing the "mcmc" config of gsplat examples/simple_trainer.py:1579–1587):
// init_opa=0.5, init_scale=0.1 (the 3-NN distance multiplier), opacity_reg=scale_reg=0.01, and
// strategy=MCMCStrategy → cfg.mcmc.enabled=true. All other fields (cap_max, lr, and the schedules)
// are unchanged.
void apply_mcmc_presets(InitOptions& io, TrainConfig& cfg);

// scene_scale = the **maximum distance from the centroid of the camera centers × 1.1** (the gsplat
// formula — the Parser scene_scale of datasets/colmap.py × the 1.1 factor of simple_trainer.py;
// the reference for means LR consistency).
// campos = -Rᵀt (viewmat is world→cam row-major). With fewer than 2 cameras, or in the degenerate
// case (maximum distance 0), it returns 1.0.
float scene_scale_from_cameras(const std::vector<TrainingCamera>& cams);

// Schedule queries (ratio-based): means lr = lr_means·scene_scale·0.01^(iter/total_iters),
// SH degree = min(iter / max(1, total·ratio), sh_degree).
float means_lr_at(const TrainConfig& cfg, int iter);
uint32_t sh_degree_at(const TrainConfig& cfg, int iter, int sh_degree);

// The activation gradient chain (task-6 handover 3 — host-owned): converts the linear/post-sigmoid
// gradients the kernels return back into gradients of the internal representation (log/logit).
// A size mismatch is the caller's responsibility (it throws, for debugging).
// v_log_scale = v_scale·exp(log_s) = v_scale·scale_linear
void chain_scale_grads_inplace(std::vector<float>& v_scales, const float* scales_linear);
// v_logit = v_opacity·σ·(1−σ)
void chain_opacity_grads_inplace(std::vector<float>& v_opacities, const float* opacs_act);

// The AA compensation gradient chain (Phase 3 Task 4 — the product rule for
// opacity_aa = sigmoid(op)·comp):
// The input v_opacities = d loss/d opacity_aa from raster bwd (raster and isect consume
// opacity_aa).
//   returns   v_comp[i]      = v_op_aa[i]·opacs_act[i]   (for injecting into proj bwd as
//                                                         v_compensations)
//   in-place  v_opacities[i] = v_op_aa[i]·comp[i]        (d loss/d post-sigmoid opacity — after
//                                                         which chain_opacity_grads_inplace applies
//                                                         the σ′ chain:
//                                                         v_logit = v_op_aa·comp·σ(1−σ))
// For culled slots both are 0, because comp = 0 (proj fwd zero-initializes it — corresponding to
// gsplat at::zeros) and v_op_aa = 0 (raster bwd zero-fill). Matching sizes are the caller's
// responsibility.
std::vector<float> chain_aa_opacity_grads_inplace(std::vector<float>& v_opacities,
                                                  const float* opacs_act,
                                                  const float* compensations);

// ===== Adding the regularization gradients (Phase 2 Task 4 — just before adam, at the same point
// as the activation gradient chain) =====
// **Adds** the analytic derivative of gsplat simple_trainer.py:993–996 `loss += reg·mean(...)` to
// the upstream gradients (matching autograd accumulation — it does not overwrite). The denominator
// is the element count of the torch .mean():
//   opacity (losses.py:675–686, opacities [N] 1D — simple_trainer.py:332):
//     v_logit[i]    += reg·σᵢ(1−σᵢ)/N            (N = v_opacities.size())
//   scale   (losses.py:689–700, log_scales [N,3] — the mean is over **every element**, so the
//            denominator is N·3):
//     v_log_s[i,k]  += reg·exp(log_s[i,k])/(N·3)  (N·3 = v_scales.size())
// opacs_act/scales_lin are the sigmoid/exp activation values the forward already computed (reusing
// σ' = σ(1−σ) and d exp/d log_s = exp). reg <= 0 is a no-op (the same gate as the gsplat `> 0.0`).
void add_opacity_reg_grad_inplace(std::vector<float>& v_opacities, const float* opacs_act,
                                  float reg);
void add_scale_reg_grad_inplace(std::vector<float>& v_scales, const float* scales_lin,
                                float reg);
// The shape regulariser's host formula (TrainConfig.flatten_weight; train_scale_shape_reg is the
// resident twin), over N = v_scales.size()/3 Gaussians: flatten_w = flatten_weight/N and
// needle_w = needle_weight/N, each a no-op at <= 0.
void add_shape_reg_grad_inplace(std::vector<float>& v_scales, const float* scales_lin,
                                float flatten_w, float needle_w, float needle_ratio);

struct TrainStepResult {
    // The reported loss = (1-λ)·L1 + λ·(1-ssim) (λ = ssim_weight; with 0 it is pure L1 — the same
    // as before).
    // Directly comparable with gsplat simple_trainer.py:961 loss.item() (the reg terms are not
    // included — P2 Task 4).
    float loss = 0.0f;
    int64_t n_isects = 0;
    // MCMC refine observations (the Task 6 bench log — on a non-refine step, refined=false and the
    // counts are 0)
    bool mcmc_refined = false;
    int64_t n_relocated = 0; // number of teleported dead slots (McmcRefineResult.relocated_dead)
    int64_t n_added = 0;     // number of newly added slots (McmcRefineResult.added_src)
    // [H,W,3] — the last render (the input for --save-preview).
    std::vector<float> render_colors;
    // The unweighted inverse-depth L1 of this step (0 when the step was not depth-supervised) —
    // reported beside, never inside, loss.
    float depth_loss = 0.0f;
};

// A forward render with the expected-depth channel (eval of the held-out depth error).
struct DepthRender {
    std::vector<float> rgb;   // [H,W,3] — the same values as Trainer::render
    std::vector<float> depth; // [H,W] expected depth in metres, 0 below kDepthMinAlpha
};

// The D7 assembly of one training step (I=C=1, B=1, PINHOLE; calc_compensations = cfg.antialiased
// — the default false is classic mode, true is AA: isect and raster consume
// opacity_aa = sigmoid(op)·comp):
//   activation (exp/sigmoid) → proj fwd → (antialiased) compute opacity_aa (host)
//   → SH fwd (dirs=means−campos, masks=(radii>0).all(-1))
//   → host post-op (+0.5, clamp to [0,∞) + storing the per-channel relu_mask — pre_clamp > 0)
//   → isect (AccuTile — opacity_aa) → sort → offset → raster fwd (opacity_aa)
//   → L1 loss/grad (+ when ssim_weight > 0, ssim_fwd(need_coeffs)→ssim_bwd(v_loss=λ) combine into
//     v_render = (1-λ)·v_L1 + λ·v_DSSIM — Phase 3 Task 3)
//   → raster bwd
//   → multiply v_colors by relu_mask (blocking the gradient of channels where the clamp fired —
//     corresponding to gsplat rendering.py; since this blocks the sh_bwd input, it masks both
//     v_coeffs and v_dirs (→v_means))
//   → sh bwd (v_dirs→v_means accumulation) → (antialiased) the AA product-rule chain (host —
//     v_comp = v_op_aa·σ, v_opacity = v_op_aa·comp) → proj bwd (injecting v_compensations)
//   → the activation gradient chain (host)
//   → adding the regularization gradients (opacity_reg/scale_reg > 0 — just before adam, Task 4)
//   → fused_adam ×6 parameter groups (D3 order: means→quats→scales→opacities→sh0(→shN))
//   → (mcmc.enabled) MCMC: on a refine step, relocate→add; the noise is applied on every step
//     (gsplat simple_trainer.py:1133–1169 — the step_post_backward order after optimizer.step()).
// adam skips culled slots through the valid mask.
//
// MCMC derivation notice (Phase 2 Task 3): the orchestration procedure is a reimplementation of
// gsplat strategy/mcmc.py:139–239 (_relocate_gs/_add_new_gs/step_post_backward) +
// strategy/ops.py:301–401 (relocate/sample_add — multinomial, bincount, the index flow, and the
// Adam state surgery rules).
// The cap_max fixed arena: when mcmc.enabled, the SplatCloud vectors and the Adam moments are
// preallocated once at cap_max rows and only count (= alive) grows (reallocation is forbidden —
// avoiding the gsplat #444 OOM, design §5.2). Kernel dispatches and the grad buffers are sized by
// count (the arena tail is untouched).
class Trainer {
public:
    // The Adam moments (public — for verifying the MCMC state surgery (McmcStrategyTests), and as
    // the future checkpoint surface).
    // Defined in splatx/AdamState.h (split into its own header — it is an alias, so the existing
    // spelling stays valid).
    using AdamState = splatx::AdamState;
    enum class ParamGroup { Means, Quats, Scales, Opacities, Sh0, ShN };

    Trainer(KernelBackend& backend, SplatCloud cloud, const TrainConfig& cfg);
    // gt_rgb = [H,W,3] f32 (ImageF32.rgb — matching cam.width/height is validated by the caller).
    // iter is 0-based.
    // With cfg.resident and backend support, this takes the resident step path (otherwise the
    // previous host round-trip D7 assembly).
    // depth: the view's LiDAR target — it supervises the step only when cfg.depth_weight > 0.
    // mask: the view's region mask (RegionMask.h) — it masks the L1, SSIM and depth terms; its
    // view_count/ssim_count are used when >= 0, else computed. nullptr = unmasked.
    TrainStepResult step(const TrainingCamera& cam, const float* gt_rgb, int iter,
                         const DepthTarget* depth = nullptr, const RegionMask* mask = nullptr);
    // Forward-only render with the expected depth (full sh_degree).
    DepthRender render_with_depth(const TrainingCamera& cam);
    // Forward-only render (eval/preview — full sh_degree). Returns [H,W,3].
    // When resident is active it uses the resident forward — the same kernel and activation path as
    // the forward of step() (preserving bit-exact consistency for a self-rendered GT).
    std::vector<float> render(const TrainingCamera& cam);
    // Parameter access — on the resident path, if the device is authoritative, the host mirror is
    // synchronized first (a lazy download; observations such as count are valid after the value
    // copy). The reason this became non-const: the synchronization side effect.
    const SplatCloud& cloud() {
        sync_host_from_device();
        return cloud_;
    }
    // Lightweight observation for callers that only need the splat count (no download — count is
    // host-owned)
    int64_t splat_count() const { return cloud_.count; }

    // ===== The MCMC strategy surface (driven by step() — the public members below are for direct
    // calls from tests and tools) =====
    // With mcmc.enabled=false, calling refine/inject/adam_state throws (blocking uninitialized
    // misuse).
    struct McmcRefineResult {
        std::vector<int64_t> relocated_dead; // teleported dead slots (ops.py:321 dead_indices)
        std::vector<int64_t> relocation_src; // the matching sources (global sampled_idxs — the
                                             //  rows whose Adam state is reset to 0)
        std::vector<int64_t> added_src;      // the add sources (the origin of each new slot j —
                                             //  the sources themselves are not reset)
    };
    // The refine schedule decision (mcmc.py:159–163):
    // iter > start && iter < stop && iter % every == 0
    // (the boundaries are ratio × total_iters — see the McmcOptions comments).
    bool mcmc_refine_due(int iter) const;
    // relocate (ops.py:302–356) → add (ops.py:360–401). Runs immediately, without the schedule
    // decision.
    McmcRefineResult mcmc_refine();
    // Noise injection on every step (mcmc.py:179–193): noise = rng_normal(seed, offset=iter),
    // noise_scale = means_lr_at(iter)·noise_lr. positions in-place.
    void mcmc_inject_noise(int iter);
    AdamState& adam_state(ParamGroup g); // state surgery verification surface (McmcStrategyTests)

    // ===== Checkpoints (Phase 6 round 2 Task 2 — design §7.5/§8.1; the format and modes are in
    // Checkpoint.h) =====
    // make_checkpoint: call it at a **step-boundary safe point** (after step() returns — there is
    // no partial mid-step state). On the resident path it first recovers the values through the
    // existing lazy synchronization state machine (sync_host_from_device — downloading the
    // parameters + the Adam moments, with Home staying on the device) and then takes the snapshot —
    // it is read-only, so it has no effect on the subsequent training trajectory, and the GPU drain
    // is already guaranteed by the time executor step() returns (the waitUntilCompleted sync point
    // inside the step — the "save after the drain" of §7.5).
    // iter = the number of completed steps (= the iter at which the resume loop re-enters).
    // factor/test_split = the dataset fingerprint (owned by TrainC/CLI — the convention is 0 when
    // the library is used directly). params and the moments copy only count rows (the arena tail is
    // not included), and with mcmc the serialized mt19937_64 state is included (the RNG audit in
    // Checkpoint.h).
    TrainerCheckpoint make_checkpoint(int iter, int factor = 0, int test_split = 0);
    // restore_checkpoint: call it right after construction — it verifies the config fingerprint
    // (verify_checkpoint_config: an sh_degree mismatch = throw, §8.1; the other fields also raise
    // explicit errors) and then restores the parameters, the Adam moments, and the MCMC RNG state.
    // count is reset to the value in ck (exceeding the arena throws), and the moment arena tail is
    // re-established as 0. The resident state is reset — the next resident step re-uploads the
    // whole arena (re-establishing the zero tail on the device). If the resume loop continues
    // calling step(cam, gt, iter) from ck.iter, the ratio-based schedules (lr, SH promotion, and
    // the refine boundaries) continue naturally (see the TrainConfig comments).
    void restore_checkpoint(const TrainerCheckpoint& ck, int factor = 0, int test_split = 0);

    struct Forward; // fwd intermediates (defined in Trainer.cpp — declared public because the
                    // cpp helpers reference it)

private:
    Forward forward_pass(const TrainingCamera& cam, uint32_t degrees_to_use,
                         bool with_depth = false);
    void adam_step(std::vector<float>& param, const std::vector<float>& grad, AdamState& st,
                   const uint8_t* valid, float lr);
    void mcmc_relocate(McmcRefineResult& out);
    void mcmc_add(McmcRefineResult& out);
    // The fingerprint derived from cfg_ (shared by make/restore — factor/test_split are the
    // caller-owned dataset fingerprint)
    CheckpointConfig checkpoint_config(int factor, int test_split) const;
    // Resets exp_avg/exp_avg_sq of the sampled/new rows to 0 across every parameter group
    // (ops.py:345–347).
    void mcmc_zero_adam_rows(const int64_t* rows, size_t n_rows);

    // ===== The device-resident path (Phase 3 Task 5 — D11) =====
    // The state machine that tracks where the parameters and moments are authoritative:
    // device_authoritative_ = the last update happened on the device (a resident step). When the
    // host reads the parameters, a lazy download runs (host_fresh_); when the host mutates them
    // (MCMC refine/noise/adam_state), it downloads and then sets device_authoritative_ = false —
    // the next resident step re-uploads. The first upload covers the whole arena (establishing the
    // zero tail); afterwards only count rows.
    bool resident_ready();          // lazy creation of the executor (pinned false if unsupported)
    void resident_upload();         // host → device (Home transitions)
    void sync_host_from_device();   // lazy download (for reading — Home is unchanged)
    void sync_host_for_mutation();  // download + move Home to host (required before mutating)
    ResidentParamSync make_sync(int64_t rows); // pointer view of the 6 parameter/moment groups
    TrainStepResult step_resident(const TrainingCamera& cam, const float* gt_rgb, int iter,
                                  const DepthTarget* depth, const RegionMask* mask,
                                  int64_t mask_count, int64_t mask_ssim_count,
                                  int64_t depth_valid);

    KernelBackend& backend_;
    SplatCloud cloud_; // with mcmc.enabled, a cap_max fixed arena (count = alive, the vectors
                       //  are cap-sized)
    TrainConfig cfg_;
    int64_t arena_rows_ = 0; // row count of the parameter vectors (mcmc: cap_max, otherwise the
                             //  initial count)
    AdamState means_st_, quats_st_, scales_st_, opac_st_, sh0_st_, shN_st_;
    std::mt19937_64 mcmc_rng_;  // multinomial only (the perturb noise is Philox — backend-owned)
    std::vector<float> binoms_; // [51·51] Eq.9 LUT (precomputed when mcmc.enabled)

    // device-resident state (used only when cfg_.resident && the backend supports it)
    std::unique_ptr<TrainStepExecutor> exec_;
    bool exec_unavailable_ = false;    // creation failed / unsupported — never retried
                                       //  (pinned to the host fallback)
    bool exec_uploaded_once_ = false;  // whether the first upload (the whole arena) completed
    bool device_authoritative_ = false;
    bool host_fresh_ = true;           // the host mirror matches the device (the lazy download cache)
};

} // namespace splatx
