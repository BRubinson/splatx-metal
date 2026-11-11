// splatx train C API — the pure C surface the Swift CLI (splatx-cli) calls. Original work
// (not derived).
// SplatXCore is a C++ module (its public headers use std::vector and friends), so Swift cannot
// import it directly; this minimal wrapper target exists for that reason (the header is C only —
// the Swift ClangImporter parses it as-is).
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Runs training from a COLMAP bundle (data_dir: sparse/0 + images[_factor]).
//   factor:        1|2|4|8 — selects the images_N folder and scales K/width/height by 1/factor.
//                  If images_N is absent, falls back to downscaling the images/ originals by
//                  factor at load time (Integration Task 2 — iOS capture app bundles have no
//                  pre-resized folder. The convention is TrainingInput.h resolve_images_dir /
//                  ImageIO.h downscaled_dim)
//   iters:         number of training iterations (images are chosen round-robin — iter % n_cameras)
//   max_points:    init_splats subsample cap (premise: KNN is O(N²))
//   sh_degree:     0..3
//   out_ply:       path the INRIA PLY is saved to at the end
//   preview_png:   NULL allowed — after training, saves a render from the eval_image camera as PNG
//   eval_image:    camera index for the preview render (out-of-range is clamped)
//   kernel_src_dir: NULL uses the build-time KERNEL_SRC_DIR macro value — a SwiftPM build gets the
//                  package path (Sources/SplatXKernelSrc), an XCFramework distribution build gets
//                  an **intentionally invalid path** (build-machine paths are forbidden —
//                  build_xcframework.sh). If the fallback to that invalid path actually fires, it
//                  fails loudly instead of silently substituting something else (non-zero return +
//                  err_buf diagnostic). The source-compile fallback layer fires only when the
//                  automatic search for splatx.metallib next to the executable finds nothing; it
//                  is not a fallback for an explicit metallib_path failure. XCFramework consumers
//                  MUST inject either metallib_path or a valid kernel_src_dir at runtime.
//   metallib_path: NULL allowed — explicit path to an offline-compiled .metallib
//                  (the tools/build_metallib.sh output). When given, only this path is used and a
//                  load failure is an immediate error (non-zero return — no automatic layer
//                  fallback, the MetalBackend.h convention). NULL searches for splatx.metallib
//                  next to the executable, then compiles from kernel_src_dir, in that order.
//   log_every:     progress log period (in iterations, to stdout)
//   mcmc:          non-zero turns MCMC densification on and applies the gsplat mcmc preset
//                  (init_opa=0.5, init_scale=0.1, opacity_reg=scale_reg=0.01 —
//                  simple_trainer.py:1579–1587 reproduction; the refine/noise schedules use the
//                  default ratios)
//   cap_max:       upper bound on the Gaussian count when mcmc is non-zero (the fixed arena size,
//                  requires >= the initial N). Ignored when mcmc is 0.
//   test_split:    0 = split off (every camera is train — the previous behavior). K > 0 splits off
//                  the cameras whose name-sorted index satisfies i % K == 0 as test, and training
//                  cycles through train only (the MipNeRF360 convention K=8 — gsplat
//                  datasets/colmap.py:456–461). K == 1 is rejected (throw) because train would be
//                  empty. Cameras are indexed after sorting by image name, whether or not
//                  test_split is used (the gsplat colmap.py:194–199 convention).
//   eval_every:    K > 0 renders every test view every K iterations and at the end of training,
//                  logging the mean PSNR (the render is clamped to [0,1] before psnr — matches
//                  gsplat simple_trainer.py:1240). Requires test_split > 0 (otherwise throw).
//                  Even at 0, one final eval still runs at the end when test_split > 0.
//   ssim_weight:   SSIM loss weight λ ∈ [0,1] (outside that range: throw) — loss = (1-λ)·L1 +
//                  λ·(1-SSIM), and the gradients combine the same way (reproduction of gsplat
//                  ssim_lambda 0.2 — simple_trainer.py:171/:961). 0 bypasses the SSIM path
//                  entirely (pure L1, the previous behavior). > 0 requires the training images to
//                  have H/W >= 11 (the SSIM valid map).
//   resident:      non-zero selects the device-resident train_step (Phase 3 Task 5 — on is the
//                  recommended default): parameters, moments, and intermediate buffers stay
//                  resident on the GPU, with 2 command buffers per step (the sync points are the
//                  n_isects readback and the host-side loss summation). MCMC refine and eval stay
//                  on the host (partial readback/writeback of the resident buffers). 0 selects the
//                  previous host round-trip chaining (a verification surface — the CLI
//                  --no-resident escape hatch). The only deviations between the two paths are the
//                  libm↔metal precise:: tolerance of the exp/sigmoid activations and the atomic
//                  bwd accumulation order (TrainerTests gates the comparison).
//   antialiased:   non-zero enables anti-aliasing (the Mip-Splatting 2D compensation —
//                  reproduction of gsplat rasterize_mode="antialiased"): projection outputs the
//                  compensation factor comp = sqrt(det Σ' / det(Σ' + eps2d·I)) and opacity·comp is
//                  what enters isect/raster (rendering.py:864–896 — the AccuTile level set also
//                  uses the post-AA opacity, and the bwd product-rule chain is included).
//                  0 = classic (the gsplat default and the condition of the published benches).
//                  Note: a standard PLY does not store whether AA was used (see the README viewer
//                  note).
//   resume_checkpoint: NULL allowed — resumes training from a checkpoint file (Phase 6 round 2
//                  Task 2, design §7.5/§8.1). After loading, the config **fingerprint is
//                  verified**: if sh_degree, factor, test_split, total_iters (= iters), mcmc
//                  on-off, cap_max, mcmc seed, scene_scale, ssim_weight, or antialiased differs
//                  from the current request, it is rejected with an error naming the field
//                  (sh_degree follows the §8.1 color-contamination lesson — there is no relaxed
//                  path). On success init_splats (KNN) is skipped, params, the Adam moments, and
//                  the MCMC RNG are restored, and training resumes from ck.iter — view selection
//                  (iter % n_train), the ratio-based schedules, and the noise (Philox offset=iter)
//                  are all iter-based, so the stream is identical to a continuous run.
//                  max_points does not apply to a resume (it is an init-only cap).
//   save_checkpoint_path: NULL allowed — the path checkpoints are saved to. When given, one save
//                  happens at the end of training, plus a save at a step boundary every K
//                  iterations when checkpoint_every > 0. Writes are atomic (temp + rename — the
//                  partial file is never exposed at the final path). The CLI/C API saves are
//                  **fp32 lossless** (fp16 half precision is the iOS resign-active surface —
//                  Checkpoint.h, consumed by the Task 5 Swift API).
//   checkpoint_every: K > 0 saves every K iterations (save_checkpoint_path is required —
//                  otherwise throw). 0 = a single save at the end only (when a path is given).
//                  Negative values throw.
//   auto_cap:      non-zero computes cap_max automatically (Phase 6 round 2 Task 3 — design §7.2;
//                  mcmc must be non-zero, otherwise throw). The cap_max argument is ignored and
//                  the value is determined after the COLMAP load and factor scaling by the
//                  MemoryGovernor.h formula: (budget×4/5 − the fixed components (GT cache,
//                  images, isect, and base)) / resident bytes-per-splat, clamped to [100k, 4M].
//                  The basis of the computation is printed as a single "auto-cap: ..." log line.
//                  If the init subsample cap max_points exceeds the computed cap, it is lowered to
//                  the cap (a pre-emptive guard for the contract where Trainer throws when the
//                  initial N > cap — announced in the log). When combined with resume_checkpoint,
//                  the computed cap must match the cap_max in the checkpoint fingerprint (a
//                  mismatch is explicitly rejected by fingerprint verification — the arena size
//                  contract).
//   memory_budget_bytes: the auto_cap budget (bytes). <= 0 uses the platform fallback — on macOS
//                  a conservative 50% of sysctl hw.memsize (see the rationale comment in
//                  MemoryGovernor.cpp). iOS apps must inject os_proc_available_memory() (the
//                  first-class path — no fallback).
//   thermal_mode:  the thermal response mode (Phase 6 round 2 Task 4, design §7.4 — the
//                  authoritative policy contract is SplatXCore ThermalGovernor.h):
//                    0 = performance: no intervention (state-transition observation logs only)
//                    1 = balanced (the recommended default): on entering critical, submission
//                        stops at a step boundary (a safe point) — pause, an idle wait on a cv —
//                        and resumes (not an abort) after 5s continuously at serious or below
//                        (hysteresis). serious itself triggers no intervention — it is the normal
//                        operating state under sustained GPU load (consistent with §7.4 "stop
//                        **above** serious" — the A17 bake-off measured evidence is in
//                        ThermalGovernor.h). After 10s continuously at critical (the clock stays
//                        valid during a pause), a partial export (the currently alive state to
//                        out_ply — reusing the existing export filter path) and a checkpoint save
//                        (when save_checkpoint_path is given) are performed, followed by a
//                        graceful abort — **return 2** (eval and the preview are skipped — this
//                        avoids sustained load).
//                    2 = lowheat: balanced plus a 100ms idle inserted between steps plus the pause
//                        threshold lowered to serious (the resume threshold is fair — the previous
//                        balanced threshold).
//                  Injection through sx_thermal_set_state is the first-class source of the state
//                  (the app's thermal observer — see below). The macOS CLI has no injection
//                  source, so the state is fixed at nominal = no action. Any other value is
//                  rejected. The hidden test path (the injection schedule and the shortened
//                  timings) is documented in the TrainC.cpp SPLATX_THERMAL_* comments (for driving
//                  the CLI E2E end to end without inducing real-device heat (serious/critical)).
// The PLY save goes through the export filter (PlyIO.h filter_cloud_for_export — drops NaN/Inf
// rows and keeps sigmoid(opacity) > 0.005, with statistics logged. The convention for comparing
// against gsplat exporter.py is in that header's comments).
// Returns 0 on success. A thermal graceful abort returns 2 and an sx_train_request_stop
// cooperative stop returns 3 (neither is an error — the partial export and the checkpoint have
// been written, and err_buf is not filled in).
// On failure it returns 1 and writes a NUL-terminated message into err_buf.
// (This 25-argument form is a **compatibility wrapper** over sx_train_run_args — the Task 5
// extension fields exist only on the struct form.)
int32_t sx_train_run(const char* data_dir,
                     int32_t factor,
                     int32_t iters,
                     int64_t max_points,
                     int32_t sh_degree,
                     const char* out_ply,
                     const char* preview_png,
                     int32_t eval_image,
                     const char* kernel_src_dir,
                     const char* metallib_path,
                     int32_t log_every,
                     int32_t mcmc,
                     int64_t cap_max,
                     int32_t test_split,
                     int32_t eval_every,
                     float ssim_weight,
                     int32_t antialiased,
                     int32_t resident,
                     const char* resume_checkpoint,
                     const char* save_checkpoint_path,
                     int32_t checkpoint_every,
                     int32_t auto_cap,
                     int64_t memory_budget_bytes,
                     int32_t thermal_mode,
                     char* err_buf,
                     int32_t err_buf_len);

// ===== Task 5: struct argument form + progress callback + extended resume (the surface the Swift
// SplatXTrainer consumes) =====
// The existing 25-argument sx_train_run is retained as a thin wrapper over this struct form
// (compatibility first — the conclusion of the Task 4 carry-over "consider a struct": the existing
// call sites (the CLI and SplatXBench) stay unchanged, and extensions accumulate only on the
// struct form).

// A progress tick — handed to the callback at every step boundary (on the training-loop thread).
// This is the first-class progress stream, used instead of parsing the stdout log (the SplatXBench
// approach). App-layer metrics such as memory/thermal observation are joined into the tick by the
// caller — this library does not subscribe to platform APIs, on principle (ThermalGovernor.h).
typedef struct SxTrainProgress {
    int32_t iter;          // number of completed iterations (including this step — 1..run_until)
    int32_t run_until;     // the ending iteration of this run (reflects an extended resume)
    int32_t total_iters;   // the ratio-schedule denominator (can be smaller than run_until on an
                           // extended resume)
    float loss;            // the loss reported for this step ((1-λ)L1 + λ(1-SSIM)); computed when
                           // iter % log_every is 0 or 1, on eval ticks and on the first and last
                           // step, else the last computed value
    float psnr;            // mean PSNR(dB) if an eval_every evaluation ran on this tick, else NaN
    double step_ms;        // duration of this step (ms)
    int64_t n_isects;
    int64_t splat_count;   // the current N (a lightweight observation — no download)
    int32_t thermal_state; // the governor's current state 0..3 (the last injected value)
    // ---- Depth supervision extension (at the end of the struct — existing offsets unchanged) ----
    float depth_rmse;      // held-out expected-depth RMSE (m) on an eval tick when test views have
                           // depth sidecars, else NaN (DepthSupervision.h convention)
    // ---- Region mask extension (at the end of the struct — existing offsets unchanged) ----
    float psnr_masked;       // held-out PSNR (dB) over in-mask pixels on an eval tick when test
                             // views have masks/<stem>.png, else NaN (RegionMask.h)
    float depth_rmse_masked; // held-out depth RMSE (m) over depth-valid in-mask samples, else NaN
} SxTrainProgress;

// Called synchronously on the training-loop thread — a slow callback delays training by exactly
// that much. Do not re-enter sx_train_* inside the callback (sx_thermal_set_state is allowed — its
// contract is "any thread").
typedef void (*SxTrainProgressFn)(const SxTrainProgress* progress, void* user_data);

// ===== App round 3 Task 1: the in-memory snapshot callback (the input for live rendering without
// going through a file — spec 2026-07-20-trainer-app-live-stream §2. This comment is the
// authoritative contract) =====
// A pure output surface of the same family as the live PLY export — the training algorithm is
// unchanged. The contract:
//   - Enabled when snapshot_cb != NULL && snapshot_every > 0 (**both** — the off combinations are
//     valid input). struct form only — it does not exist on the 25-argument sx_train_run wrapper
//     (same as live_ply).
//   - Timing: at a step boundary, in the same phase as the live PLY hook — when the completed iter
//     is a multiple of snapshot_every ((it+1) % every == 0, after the MCMC refine has been
//     applied). While parked (a cooling pause or a hold) there are no steps, so no callback fires
//     either.
//   - Filter: identical to the final export and the live PLY (PlyIO.h filter_cloud_for_export —
//     drops NaN/Inf rows and keeps sigmoid(opacity) > 0.005). count = the number of rows that pass
//     the filter.
//   - On a tick where count == 0 the array pointers may be NULL — consumers must gate on count.
//   - Domain: **raw values** with no activation applied (identical to what a PLY records — scales
//     in log-space, opacities in logit-space, quats w-first and unnormalized, sh0/shN raw SH
//     coefficients). Applying the activations is the consumer's responsibility.
//   - Pointer lifetime: valid **only for the duration of the callback** (an internal filtered copy
//     — freed after it returns). Copy inside the callback to retain the data.
//   - Thread: called synchronously on the training-loop thread — a slow callback delays training
//     by exactly that much. Do not re-enter sx_train_* inside the callback (sx_thermal_set_state
//     is the only exception — same as SxTrainProgressFn).
//   - Failure is non-fatal: on an exception from cloud() or the filter, a warning goes to stderr
//     and training continues (the live PLY precedent — not reflected in the return code or in
//     stats).
//   - Cost: on the resident path every snapshot triggers a **full download** of the parameters
//     (cloud() → sync_host_from_device — a read, so Home is unchanged; the Trainer.cpp location
//     state machine). When it coincides with the live PLY on the same tick, the host_fresh_ cache
//     makes the two share a single download (the TrainC.cpp hook shares even the filtered copy
//     once).
typedef struct SxSplatSnapshot {
    int32_t iter;            // the completed iteration (1..run_until)
    int32_t sh_degree;       // 0..3 — K = (sh_degree+1)^2 - 1
    int64_t count;           // number of rows that pass the export filter (N)
    const float* means;      // [N,3]
    const float* quats;      // [N,4] w-first, raw and unnormalized
    const float* scales;     // [N,3] raw, log-space
    const float* opacities;  // [N]   raw, logit-space
    const float* sh0;        // [N,3] raw coefficients
    const float* shN;        // [N,K,3] coefficient-first RGB interleaved, NULL when degree is 0
} SxSplatSnapshot;

typedef void (*SxSnapshotFn)(const SxSplatSnapshot* snap, void* user);

// End-of-run summary (filled into args.stats — NULL allowed). Valid only for return codes 0/2/3.
typedef struct SxTrainStats {
    int32_t completed_iters;  // number of completed steps (reflects a thermal abort or a stop —
                              // the checkpoint iter)
    int32_t run_until;        // the ending iteration that was targeted
    int32_t total_iters;      // the ratio-schedule denominator
    int64_t exported_splats;  // N that passed the export filter (the actual row count in the out
                              // file)
    float final_eval_psnr;    // mean PSNR(dB) of the final eval — NaN if it did not run
                              // (test_split 0, or an interrupted run)
    double train_seconds;     // wall clock of the training loop (excludes load and export)
    int32_t checkpoint_saved; // whether the final save to save_checkpoint_path was performed (0|1)
    // ---- color_correct Task 2 extension (at the end of the struct — the offsets of the existing
    //      fields are unchanged, so zero-init consumers stay safe) ----
    // Mean cc PSNR(dB) of the final eval: per view, the render is matched to the GT with
    // color_correct_affine (Trainer.h — per-channel affine appearance alignment), then the
    // arithmetic mean of the resulting PSNRs. It does not replace the raw value (final_eval_psnr)
    // — the two are **reported side by side** (a fair metric for an appearance-neutral model —
    // spec 2026-07-15-color-correct-eval §1). NaN if it did not run (test_split 0, or an
    // interrupted run) — same convention.
    float final_cc_psnr;
    // ---- Perceptual-metric extension (at the end of the struct — the offsets of the existing
    //      fields are unchanged, so zero-init consumers stay safe. The final_cc_psnr
    //      precedent) ----
    // Mean SSIM / cc SSIM of the final eval (valid padding, 11×11 σ=1.5 — the same convention as
    // the training loss, spec 2026-07-15-perceptual-eval §2). Views with h<11||w<11 are excluded
    // from the mean; NaN when there are zero valid views or when it did not run (test_split 0, or
    // an interrupted run).
    float final_ssim;
    float final_cc_ssim;
    // ---- Depth supervision extension (at the end of the struct — existing offsets unchanged) ----
    // Expected-depth RMSE (metres) of the final eval, pooled over the held-out views' valid sidecar
    // samples (DepthSupervision.h mask). NaN when it did not run or no test view has a sidecar —
    // reported for every run with sidecars, depth-supervised or not.
    float final_depth_rmse;
    // ---- Region mask extension (at the end of the struct — existing offsets unchanged) ----
    // The final eval over the region masks (RegionMask.h): the mean over held-out views with an
    // in-mask pixel of their in-mask PSNR (dB), and the depth RMSE (m) pooled over the depth-valid
    // in-mask samples. NaN when it did not run or no test view has a mask.
    float final_psnr_masked;
    float final_depth_rmse_masked;
    int32_t mask_views; // views (train + test) with a masks/<stem>.png sidecar
} SxTrainStats;

// All of the sx_train_run arguments plus the Task 5 extensions. **Zero/NULL-initialize first**,
// then fill in only the fields you need (in Swift, the `SxTrainArgs()` default construction is a
// zero-init). The field meanings are the same as in the sx_train_run argument comments — only the
// extension fields are commented below.
typedef struct SxTrainArgs {
    const char* data_dir;
    int32_t factor;
    int32_t iters;
    int64_t max_points;
    int32_t sh_degree;
    const char* out_path;
    const char* preview_png;          // NULL allowed
    int32_t eval_image;
    const char* kernel_src_dir;       // NULL allowed
    const char* metallib_path;        // NULL allowed
    int32_t log_every;
    int32_t mcmc;
    int64_t cap_max;
    int32_t test_split;
    int32_t eval_every;
    float ssim_weight;
    int32_t antialiased;
    int32_t resident;
    const char* resume_checkpoint;    // NULL allowed
    const char* save_checkpoint_path; // NULL allowed
    int32_t checkpoint_every;
    int32_t auto_cap;
    int64_t memory_budget_bytes;
    int32_t thermal_mode;
    // ---- Task 5 extensions ----
    // Extended resume (Task 2 carry-over — TrainerBackend.resume(checkpoint,
    // additionalIterations)):
    // 0 = the previous behavior (ck.iter → iters (= the fingerprint's total_iters)). K > 0 requires
    // resume_checkpoint and runs **K more iters** from ck.iter (run_until = ck.iter + K — which may
    // exceed total_iters). In that case the iters argument is **ignored** and the checkpoint's
    // value remains authoritative as the ratio-schedule denominator (total_iters) (fingerprint
    // verification also passes against that value — the semantics are "run the same run longer").
    //
    // Rationale for the semantics (extend with a fixed denominator vs. reset the denominator — of
    // the two options in the Task 2 carry-over, the former was adopted):
    // 1) The denominator of the schedule ratios (design §4) is the definition of "the planned run".
    //    If the denominator is reset to ck.iter+K, the progress fraction p at the resume point
    //    moves backwards, the lr rises again (a 0.01^p jump), and the MCMC refine window (default
    //    stop 25k/30k) reopens — so the trajectory and the quality change with every extension.
    // 2) With a fixed denominator, in the p = iter/total > 1 region the lr **keeps decaying
    //    monotonically** as 0.01^p, the SH promotion is saturated, and refine does not run because
    //    it is outside the stop ratio — **pure refinement** with no growth. This is exactly
    //    consistent with the stated intent of the iOS capture app's TrainingConfig.enhance
    //    ("additional iters past densification — no NEW gaussians, pure refinement").
    // 3) It preserves the fixed-total_iters contract of the fingerprint (Checkpoint.h — "if the
    //    denominator differs, the meaning of iter differs") — the reset option would require
    //    removing total_iters from the fingerprint, which weakens the safety of ordinary resume.
    int32_t resume_extend_iters;
    // Save precision for save_checkpoint_path: 0 = fp32 lossless (the previous behavior — the
    // canonical choice for the CLI and desktop), non-zero = fp16 half precision (the iOS surface —
    // half the size and time; the lossiness contract is Checkpoint.h).
    int32_t checkpoint_fp16;
    SxTrainProgressFn progress;       // NULL allowed — called at every step boundary
    void* progress_user;              // passed through to progress unchanged
    SxTrainStats* stats;              // NULL allowed — receives the end-of-run summary (valid for
                                      // returns 0/2/3)
    // ---- App round 2 Task 1 extension: live PLY export (at the end of the struct — the offsets
    //      of the existing fields are unchanged, so zero-init consumers stay safe. spec
    //      2026-07-20-trainer-app-live-polish §2) ----
    // Periodically refreshes the current model as a PLY for a live preview during training.
    // Enabled **only when both** live_ply_path (NULL/empty string = off) and live_ply_every
    // (<= 0 = off) are valid.
    //   - Timing: at a step boundary (after step() returns — the same block as the governor and
    //     checkpoint polling). Written when the completed iter is a multiple of live_ply_every
    //     (the same phase as checkpoint_every).
    //   - Filter: identical to the final export (PlyIO.h filter_cloud_for_export — drops NaN/Inf
    //     rows and keeps sigmoid(opacity) > 0.005).
    //   - Atomicity: written to "<live_ply_path>.tmp" and then renamed — the reader (the app
    //     viewer) never sees a partial file (an existing saved copy is also replaced atomically).
    //   - Format: **always PLY** (no automatic .spz extension detection — this surface is consumed
    //     directly by the viewer).
    //   - Failure is **non-fatal**: a warning to stderr, the .tmp is cleaned up, and training
    //     continues (the live preview is a secondary artifact — not a reason to stop training. Not
    //     reflected in the return code or in stats).
    //   - Cost: on the resident path every export triggers a **full download** of the parameters
    //     (cloud() → sync_host_from_device — a read, so Home is unchanged; the Trainer.cpp
    //     location state machine). Do not set the period too short (the app uses 250).
    //   - The core does not delete the live file when training ends (that is the app's
    //     responsibility — spec §3).
    // This extension is struct-form only — it does not exist on the 25-argument sx_train_run
    // wrapper (the contract at the top of this header).
    const char* live_ply_path;        // NULL/empty = off
    int32_t live_ply_every;           // <= 0 = off
    // ---- App round 3 Task 1 extension: the in-memory snapshot callback (at the end of the struct
    //      — the offsets of the existing fields are unchanged, so zero-init consumers stay safe.
    //      The authoritative contract is the SxSplatSnapshot/SxSnapshotFn comments) ----
    SxSnapshotFn snapshot_cb;         // NULL = off
    void* snapshot_user;              // passed through to snapshot_cb unchanged
    int32_t snapshot_every;           // <= 0 = off (the app uses 50)
    // ---- Depth supervision extension (at the end of the struct — zero-init means off) ----
    // Weight of the confidence-masked inverse-depth L1 over the expected depth, applied on train
    // views that have a depth sidecar (depths/<stem>.png beside images/; DepthSupervision.h).
    // 0 = off (RGB-only training, unchanged). Must be finite and >= 0.
    float depth_weight;
    // ---- Shape regulariser extension (at the end of the struct — zero-init means off) ----
    // Per Gaussian with linear scales s1 ≥ s2 ≥ s3 (TrainConfig in Trainer.h): flatten_weight pulls
    // s3 down (loss (w/N)·Σ s3), needle_weight is a hinge on s1/s2 above needle_ratio (loss
    // (w/N)·Σ max(0, s1/s2 − ratio)). Weights finite and >= 0, 0 = off; needle_ratio >= 1, and 0
    // reads as 5. Region masks need no argument: masks/<stem>.png sidecars beside images/ switch
    // them on (RegionMask.h).
    float flatten_weight;
    float needle_weight;
    float needle_ratio;
} SxTrainArgs;

// The struct-argument entry point itself — the return/error contract is identical to sx_train_run
// (0 success / 1 failure + err_buf / 2 thermal graceful abort / 3 cooperative stop). A NULL args
// returns 1.
int32_t sx_train_run_args(const SxTrainArgs* args, char* err_buf, int32_t err_buf_len);

// ===== Task 5: training control (process-global — the same "exactly one concurrent training run"
// premise as sx_thermal_set_state) =====

// Requests a cooperative stop — an in-flight sx_train_run(_args) halts training at the next step
// boundary (a safe point), performs the same wind-up as a thermal abort (a checkpoint save (when
// save_checkpoint_path is given) plus a partial export, skipping eval and the preview), and then
// returns **3**. Even while parked in a pause (hold or thermal) it wakes the loop immediately and
// stops (ThermalGovernor::request_external_stop — the wake surface from the Task 4 carry-over).
// A no-op when there is no active run — unlike the thermal state, it is **not carried over** to
// the next run (every run gets a new governor). Swift consumers call this from the Task
// cancellation handler (SplatXTrainer).
// **Non-blocking** (Task 5 review 2): this call returns immediately even while
// sx_train_pause_checkpoint is holding the control mutex during a save (several seconds) — the
// request flag is written first and then delivered with a try-lock, and if the try-lock fails the
// mutex holder (the pause path) delivers it on the way out of its critical section. When it
// overlaps a pause save, the order is that the stop is delivered **after the save completes** (the
// checkpoint is not thrown away — even for a cancellation during resign-active, preserving the
// material to resume from takes priority).
void sx_train_request_stop(void);

// resign-active pause-and-drain (the §7.5 3-second contract — Task 5 pauseAndDrain):
//   1) Sets the governor's external hold latch — independent of the thermal state and mode, and a
//      thermal transition cannot release it (no resume race — the "external hold latch" comment in
//      ThermalGovernor.h, Task 4 carry-over ②)
//   2) Waits until the training thread parks at a step-boundary safe point (bounded by timeout_ms
//      — by the time a step returns the GPU drain is already guaranteed: the waitUntilCompleted
//      sync point inside the step, Trainer.h)
//   3) Saves the checkpoint **on the calling thread** (non-zero fp16 is recommended — half the
//      size and time, Checkpoint.h)
// Returns (two failures the caller must respond to differently — Task 5 review 1):
//    0 success — the hold is **kept** (release it with sx_train_resume_after_pause when returning
//      to the foreground; until it is released the loop stays parked = no background GPU
//      submission)
//    1 save failed (err_buf — the hold has been released)
//   -1 no active training loop (no run, a section outside the loop such as load or init, or the
//      loop ended while waiting) — this means there is no training state to save yet or any more,
//      so it **can be ignored** (the hold has been released)
//   -2 park timeout (a long step or an eval in progress — no safe point reached within timeout_ms)
//      — the loop is **still running** and no checkpoint was saved (the hold has been released —
//      background GPU submission may continue). The caller must retry or report it (MUST NOT be
//      ignored — this is the scenario §7.5 exists to prevent).
int32_t sx_train_pause_checkpoint(const char* path, int32_t fp16, int32_t timeout_ms,
                                  char* err_buf, int32_t err_buf_len);

// Releases the hold latch — training resumes on the return to foreground (the thermal pause
// condition stays independent: even after the release, if the state is at or above the pause
// threshold (balanced: critical) the thermal policy takes over). A no-op when there is no active
// run.
void sx_train_resume_after_pause(void);

// Observation: 1 when the active run's training loop is parked at a safe point (a thermal pause or
// a hold), 0 when it is running, and -1 when there is no active run. This is what Swift consumers
// synthesize the coolingDown tick from.
int32_t sx_train_paused(void);

// Observation: the last injected thermal state (the sx_thermal_set_state value, initially
// 0=nominal).
int32_t sx_thermal_get_state(void);

// ===== Task 5: checkpoint header query (the parameter tensors are not loaded — only a few KB are
// read) =====
// The reference path for **deriving the resume arguments from the checkpoint file** (Swift resume
// reads the fingerprint fields from the file rather than from what the app remembers —
// structurally blocking fingerprint-mismatch accidents). 0 on success, 1 + err_buf on failure.
typedef struct SxCheckpointInfo {
    int32_t iter;          // number of completed steps (= the iter a resume starts from)
    int32_t total_iters;   // the ratio-schedule denominator (part of the fingerprint)
    int32_t sh_degree;
    int32_t factor;
    int32_t test_split;
    int32_t mcmc_enabled;  // 0|1
    int32_t antialiased;   // 0|1
    int64_t cap_max;       // 0 when mcmc is off
    uint64_t mcmc_seed;    // 0 when mcmc is off
    float scene_scale;
    float ssim_weight;
    int64_t alive_count;   // number of saved splats
    int32_t fp16;          // the save precision (0|1)
} SxCheckpointInfo;

int32_t sx_checkpoint_info(const char* path, SxCheckpointInfo* out, char* err_buf,
                           int32_t err_buf_len);

// Injects the thermal state (design §7.4 — both a simulation hook and the first-class path for the
// app's observer). The app (Swift) calls it from its Foundation ProcessInfo.thermalState observer
// (NSNotification) — this library does not subscribe to the thermal API directly (the
// ThermalGovernor.h principle). **Callable from any thread**: because sx_train_run is blocking,
// having a separate thread or observer call it is the normal path (this is why it is an injection
// function rather than a callback — the existing C surface is a pure blocking-function style with
// no handles or callbacks, and for observation events the call direction is app→library, so
// function injection is the natural fit).
// Process-global: the last injected value is retained, takes effect immediately in the governor of
// an in-flight sx_train_run, and an injection before training starts carries over as the initial
// state of the next run (the premise of exactly one concurrent training run — a structure shared
// by the CLI and the app). state: 0=nominal 1=fair 2=serious 3=critical (1:1 with
// ProcessInfo.ThermalState). Out-of-range values are ignored (robustness of the observation path —
// no throw).
void sx_thermal_set_state(int32_t state);

// The standalone surface for computing cap_max automatically (design §7.2 — the entry point the
// iOS capture app's TrainerBackend uses for its UI (the Quality slider range and so on) before
// training starts; sx_train_run's auto_cap runs the same formula internally). The basis for the
// formula and the constants is the SplatXCore MemoryGovernor.h header comment (measured against
// the current implementation: fp32 Adam plus the resident executor — fp16 resident moments are
// outside the premise).
//   budget_bytes: available memory (bytes) — on iOS, injecting os_proc_available_memory() is the
//                 first-class path. <= 0 uses the macOS fallback (a conservative 50% of sysctl
//                 hw.memsize).
//   image_w/h:    the training resolution (after factor is applied)
//   n_gt_images:  the number of resident GT views (the GT cache holds every train+test view — the
//                 total camera count)
//   sh_degree:    0..3
//   gt_cache_on:  non-zero assumes the GT cache for all views stays resident (the TrainC default —
//                 on when the estimated total is at or below the GT cache limit. The limit
//                 contract is sx_gt_cache_limit_bytes)
//   out_budget_insufficient: NULL allowed (Task 4 carry-over ④ — the budget-shortfall signal).
//                 When non-NULL, writes 1 if the value computed before clamping is below the lower
//                 bound (100k) and 0 otherwise — so when the return value is the clamped lower
//                 bound 100000, the caller (deciding on a UI warning or on refusing to train) can
//                 tell "a legitimately computed 100k" apart from "the lower bound forced by an
//                 insufficient budget".
// Returns: the suggested cap_max (clamped to [100000, 4000000]). Argument violations return -1.
int64_t sx_suggest_cap_max(int64_t budget_bytes,
                           int32_t image_w,
                           int32_t image_h,
                           int32_t n_gt_images,
                           int32_t sh_degree,
                           int32_t gt_cache_on,
                           int32_t* out_budget_insufficient);

// The downsample factor policy (design §7.3). image_w/h are the **source resolution** (factor 1).
//   ios 0 (macOS): the smallest factor (1|2|4|8) that satisfies the 1600px long-edge limit (for
//                  oversized inputs that exceed it even at /8, the best available value 8).
//   ios non-zero:  returns the fixed constant for starting at /4 (the iOS capture app measurement
//                  '/4 dense > /2 sparse' plus the history of /2 diverging — opening /2 up is
//                  allowed only after a position-LR recalibration, the suggest_factor comment in
//                  MemoryGovernor.h).
// Returns: 1|2|4|8. Argument violations return -1.
int32_t sx_suggest_factor(int32_t image_w, int32_t image_h, int32_t ios);

// The GT cache limit (Training-speed Task 2 — 2026-07-21 diagnosis: with the previous fixed 4 GiB,
// real factor 2 usage (an estimated total of 9.7 GB on garden) exceeded the limit and degenerated
// into a decode on every step). A pure function — the caller queries the physical RAM with sysctl
// hw.memsize (HW_MEMSIZE) and injects it (for unit testability — the TrainC training path is the
// only real consumer, the sx_suggest_* pattern).
//   physical_ram_bytes: physical RAM in bytes. 0 (an anomalous value such as a sysctl failure)
//                       falls back to the 4 GiB lower bound.
//   ios 0 (macOS): max(4 GiB, 50% of physical RAM) — 12 GiB on a 24 GB machine (which keeps the
//                  cache on for the factor 2 garden 9.7 GB). The 50% has the same conservative
//                  basis as default_memory_budget_bytes (Apple Silicon UMA — a safety line that
//                  excludes the GPU working set and the OS's share, MemoryGovernor.cpp).
//   ios non-zero:  a fixed 4 GiB (conservative for jetsam — the previous behavior is kept).
// Returns: the limit in bytes to compare against the estimated GT cache total
// (n_views × W×H × 12 B). No throw and no error.
uint64_t sx_gt_cache_limit_bytes(uint64_t physical_ram_bytes, int32_t ios);

// PLY ↔ SPZ format conversion (SPZ Task 3 — the backend of the splatx convert subcommand).
// The format is decided by the extension convention for both input and output (the same as
// sx_train_run --out — case-insensitive):
//   .spz → SPZ v3 (Niantic spz), anything else → INRIA PLY (including no extension).
// Filter policy (this is a format conversion, not a re-export):
//   - .spz output: the filter built into the save_spz contract applies (NaN/Inf rows make the
//     quantized packing UB — SpzIO.h. The statistics go to the stdout log). Rows near the
//     threshold can be judged differently after quantization, so double filtering is avoided — the
//     PLY output path applies **no re-filtering** (for SPZ input, load_spz guarantees finiteness;
//     for PLY input, the original rows are preserved).
//   - SPZ→SPZ: the antialiased flag in the input header is preserved. PLY input carries no flag
//     information, so false is written; SPZ→PLY loses the flag (a standard PLY has no such field —
//     a warning is logged).
// opacity saturation: for SPZ input, byte 255/0 is clamped to the finite ±ln(509) by load_spz
// (SpzIO.h).
// Returns 0 on success. On failure it returns non-zero and writes a NUL-terminated message into
// err_buf.
int32_t sx_convert_run(const char* in_path,
                       const char* out_path,
                       char* err_buf,
                       int32_t err_buf_len);

// Forward-renders a trained PLY from a COLMAP camera to a PNG (no training — read_inria_ply →
// Trainer(cloud) → one Trainer::render). factor determines the render resolution (it scales the
// camera intrinsics and dims by 1/factor, the same convention as sx_train_run). cam_index is an
// index into the name-sorted cameras (out-of-range is clamped). With metallib_path NULL it searches
// for splatx.metallib next to the executable — and fails when there is none (no source-compile
// fallback: sx_render_ply does not set kernel_src_dir, unlike sx_train_run).
// Intended for rendering different PLYs (for example an iOS output vs. a macOS one) from the same
// camera and resolution for a visual or PSNR comparison. Returns 0 on success. On failure, non-zero
// plus a NUL-terminated message in err_buf.
int32_t sx_render_ply(const char* ply_path,
                      const char* data_dir,
                      int32_t factor,
                      int32_t cam_index,
                      const char* metallib_path,
                      const char* out_png,
                      char* err_buf,
                      int32_t err_buf_len);

// ===== Cinema Task 2: the session-style render C API (sx_render_session_*) — this comment is the
// authoritative contract =====
// The final-render surface for video (spec 2026-07-26-cinema-campath "render and export
// pipeline"): re-renders a sequence of frames from arbitrary cameras with the training reference
// renderer (Trainer::render — f32 full SH, exact per-tile sorting, AA applied). Unlike
// sx_render_ply (one-shot — every call loads the PLY and creates a backend, takes only a COLMAP
// camera index, and outputs a PNG), it amortizes the load and the backend creation into a single
// open and takes an arbitrary viewmat, intrinsics, and resolution per frame. The numerical training
// path is untouched — a session owns only its own backend/Trainer instances and is independent of
// the global training state (the sx_train_* controls and the thermal governor) (forward-only —
// loss, bwd, and adam are skipped entirely).
//
// Thread contract (stated explicitly — there is no enforcement mechanism; enforcing exclusion is
// the app's responsibility):
//   - A session is **owned by a single thread**: call open/frame/close from the same thread, or
//     have the caller fully serialize them with external synchronization. Concurrent frame calls
//     on the same handle are forbidden.
//   - **MUST NOT run concurrently with sx_train_run(_args)** — the same process-global GPU
//     discipline as the premise of exactly one concurrent training run (Swift consumers should use
//     the same static exclusion gate as training).
//   - frame is a **blocking** call with a GPU round trip — it MUST NOT be called directly on a
//     Swift concurrency executor; the dedicated-Thread rule applies (the same as sx_train_run_args
//     — not minutes, but at 4K 0.1~0.24s per frame × hundreds of frames creates the same problem).

// An opaque session handle — the internals (the PLY cloud + KernelBackend + Trainer) are owned by
// TrainC.cpp.
typedef struct SxRenderSession SxRenderSession;

// **Zero/NULL-initialize first**, then fill in only the fields you need (the same convention as
// SxTrainArgs — in Swift the `SxRenderSessionArgs()` default construction is a zero-init).
typedef struct SxRenderSessionArgs {
    // Required — the path of the INRIA PLY produced by training. Loaded by read_inria_ply as-is
    // (no re-filtering — the same as sx_render_ply: the premise is an artifact to which the export
    // filter was already applied at the end of training).
    // Zero rows (an empty cloud) is an open error (the Trainer contract — count >= 1).
    const char* ply_path;
    // NULL allowed — an explicit path to an offline-compiled .metallib. When given, only this path
    // is used and a load failure **returns NULL immediately (no automatic layer fallback)** —
    // exactly the MetalBackend.h (a) contract (the same as sx_render_ply and sx_train_run. For app
    // consumers, injecting the BundlePaths metallib here is the reference path). NULL searches for
    // splatx.metallib next to the executable, then compiles from kernel_src_dir, in that order (the
    // same three layers as sx_train_run — unlike sx_render_ply, which has no kernel_src_dir, the
    // source fallback does exist here).
    const char* metallib_path;
    // NULL allowed — the source-compile fallback path. NULL uses the build-time KERNEL_SRC_DIR
    // macro value (the same invalid-path convention for XCFramework as in the sx_train_run argument
    // comments — it fails loudly).
    const char* kernel_src_dir;
    // Non-zero enables anti-aliasing (TrainConfig.antialiased — the same meaning as the
    // sx_train_run antialiased argument). A standard PLY does not store whether AA was used, so
    // **the caller must pass the setting used at training time** for the render to be consistent
    // with training (the app passes the antialiased value recorded in its RunRecord).
    int32_t antialiased;
    // 0 = derive it from the PLY (the degree inferred from the f_rest count, as-is — the ordinary
    // path).
    // 1..3 = an upper bound on the render SH degree: **exceeding the PLY degree is an open error**
    // (the higher-band coefficients do not exist); below it, the higher-band coefficients are
    // truncated session-locally (the file is unchanged — for a fast low-band preview). Any other
    // value (negative, or 4+) is an open error.
    int32_t sh_degree;
} SxRenderSessionArgs;

// Opens a session — performs the PLY load, make_metal_backend, and Trainer creation exactly
// **once**. On success it returns a handle (it must be released with close — otherwise it leaks);
// on failure it returns NULL and writes a NUL-terminated message into err_buf (the existing err_buf
// convention — there is no separate last-error function).
SxRenderSession* sx_render_session_open(const SxRenderSessionArgs* args,
                                        char* err_buf,
                                        int32_t err_buf_len);

// Renders one frame (through Trainer::render — a resident forward, the same kernel path as the
// training eval and sx_render_ply).
//   viewmat:  [16] world→cam 4x4 **row-major** — the TrainingCamera.viewmat/gsplat convention
//             (the COLMAP camera coordinate system: +X right, +Y down, +Z forward). The contents
//             are not validated — non-finite values end up as culling or a NaN render (the NaN→0
//             of the u8 conversion is the guard).
//   fx/fy/cx/cy: intrinsics in pixels (K = [[fx,0,cx],[0,fy,cy],[0,0,1]] — PINHOLE).
//             fx/fy must be positive and cx/cy finite (a violation returns -1).
//   width/height: the output resolution (>= 1). **May differ from frame to frame** (the internal
//             image buffers grow as needed — safe for session reuse).
//   out_rgba: a caller-owned buffer of **width*height*4 bytes** — RGBA8 interleaved, row-major
//             with the origin at the top left (the same order as the render_colors [H,W,3]
//             convention). The conversion uses the same quantization as save_png_rgb_f32: NaN→0,
//             clamp to [0,1], round(v·255) (the sx_render_ply PNG pixel comparison in
//             RenderSessionTests gates this identity). alpha is fixed at 255. There is no gamma
//             conversion — the render values are in the same sRGB encoding space as the training
//             GT (tag them as sRGB when consuming them through CVPixelBuffer/AVAssetWriter).
// Returns: 0 success / -1 argument violation (NULL, dimensions, intrinsics) / -2 render failure (a
// backend exception — err_buf). Determinism: the same camera in the same session → bit-identical
// output (the forward bit-exact determinism contract).
int32_t sx_render_session_frame(SxRenderSession* session,
                                const float* viewmat,
                                float fx,
                                float fy,
                                float cx,
                                float cy,
                                int32_t width,
                                int32_t height,
                                uint8_t* out_rgba,
                                char* err_buf,
                                int32_t err_buf_len);

// Closes a session — destroys the backend/Trainer (returning the GPU resources). NULL is ignored
// (no-op).
void sx_render_session_close(SxRenderSession* session);

#ifdef __cplusplus
}
#endif
