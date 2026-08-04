// Checkpoint — serialization of a Trainer state snapshot (Phase 6 round 2 Task 2 — design
// §7.5/§8.1). Written from scratch.
//
// Purpose: saving **within 3 seconds** after the GPU drain on resign-active (the §7.5 contract —
// the fp16 half-precision mode is what makes that possible), and a consistent resume on return or
// relaunch. Resume enforces **fingerprint verification** of the config — in particular an sh_degree
// mismatch is rejected unconditionally (§8.1: the lesson from the iOS capture app, where continuing
// training from a checkpoint with a different degree misaligned the SH coefficient arrays and
// contaminated the colors).
//
// Format (a bespoke binary, fixed little-endian — every Apple target is LE):
//   magic "SXCK" + version(u32) + flags(u32, bit0 = fp16)
//   config fingerprint (written field by field — struct padding is not written)
//   iter(i32) + alive_count(i64)
//   mcmc RNG state (u64 length + the mt19937_64 text serialization — length 0 if mcmc is off)
//   params ×6 (D3 group order) → Adam moments ×6 (exp_avg, exp_avg_sq per group)
// Every tensor size is derived from alive_count·dim(sh_degree) — there are no redundant size fields.
// The write is atomic and durable: write to temp (path + ".tmp") → fflush+fsync → std::rename →
// fsync the directory (best-effort). The atomicity of rename keeps the partial file from ever being
// exposed at the final path, and the data fsync before the rename removes the window in which a
// power loss leaves a zero-length or partial file because "only the rename was journaled and the
// contents never landed" (review fix 2 — iOS resign-active = the battery-exhaustion scenario is
// exactly that window).
//
// fp16 mode (flags bit0):
// - params and exp_avg: direct IEEE 754 half conversion (round-to-nearest-even). Values beyond
//   half's maximum finite value 65504 are clamped to ±65504 (preventing inf contamination when a
//   diverged state is saved); NaN is preserved.
//   The relative error bound is 2^-11 ≈ 4.88e-4 (in the normal range) — the measurements are in the
//   CheckpointTests comments.
// - exp_avg_sq: **sqrt compression** — sqrt(v) is stored as half instead of v and squared back on
//   load. Rationale: v = EMA(g²) commonly lands in the 1e-16..1e-6 range, but half's smallest
//   subnormal is 5.96e-8, so a direct conversion underflows en masse → 0. fused_adam has no bias
//   correction and computes step = -lr·m/(√v+eps) with eps = 1e-15 (TrainConfig), so an element
//   whose v collapsed to 0 risks a runaway step on the order of m/eps ~ 1e7·lr. sqrt compression
//   lowers the representable floor to v ≈ 8.9e-16 (= (2.98e-8)²) (effectively the dead-moment
//   region) and keeps the relative error at 2x (≈ 9.8e-4).
// - paired zeroing: on load, any element whose exp_avg_sq underflowed to 0 has exp_avg reset to 0
//   as well — this structurally blocks the m/eps runaway of the (m ≠ 0, v = 0) combination. Given
//   the EMA decay rates (b1 0.9 / b2 0.999), m/√v can spread by up to ~√(1-b1²/…) ≈ 3x, so the
//   boundary where only m survives really does exist. Zeroing both has the same semantics as the
//   moment reset in MCMC relocate (the gradient re-accumulates).
//
// fp32 mode is lossless (a bit-exact round trip — gated by CheckpointTests). The CLI and C API save
// as fp32 (the reference format for resuming desktop training); fp16 is the iOS resign-active
// surface (consumed by Tasks 4/5).
//
// fp16 measurements (M4, CheckpointTests — Phase 6 round 2 Task 2):
// - Size: a synthetic 1M splat sh_degree 3 model (matching garden factor 4 at cap 1M) is 337.6 MB —
//   50% of the fp32 675 MB (177 half-precision values per splat = 354 B). The small smoke file
//   ratio is 0.513 (including the header).
// - Loss: round-trip rel_l2 — params 2.5e-4 / exp_avg 2.7e-4 / exp_avg_sq 5.2e-4 (consistent with
//   the prediction from half's 4.88e-4 relative error bound and the 2x of sqrt compression).
// - Time: the 1M snapshot (make_checkpoint) 580 ms + the save (including fsync) 1560 ms =
//   **2140 ms < 3000 ms** (inside the §7.5 contract — measured on macOS; iOS 500k on a real device
//   is Task 6).
#pragma once
#include "splatx/SplatCloud.h"
#include <cstdint>
#include <string>
#include <vector>

namespace splatx {

// The config fingerprint — what resume safety verification checks (verify_checkpoint_config). It
// holds only the fields that change the meaning of resuming training:
// - sh_degree: the shape of the parameter arrays (§8.1 — a mismatch is rejected unconditionally)
// - cap_max/mcmc_enabled/mcmc_seed: arena size, strategy, and the noise (Philox key) / multinomial
//   seeds
// - factor: dataset resolution (owned by TrainC — the convention is 0 when the library is used
//   directly)
// - test_split: the train/test split convention (i % K == 0 → test) — on a mismatch the training
//   view stream after resume (the it % n_train traversal) diverges and already-trained views leak
//   into test, contaminating the evaluation (review fix 1 — like factor, a dataset fingerprint
//   owned by TrainC/CLI, 0 when the library is used directly)
// - total_iters: the denominator of the ratio-based schedules (lr decay, SH promotion, refine
//   boundaries) — on a mismatch the meaning of iter itself changes, so "resume from iter" no longer
//   holds
// - scene_scale: the means lr multiplier (derived from the cameras — a mismatch signals a different
//   dataset)
// - ssim_weight/antialiased: the loss and render definitions
// (The two reserved uint8s in the serialization are not fingerprint fields — the first byte was
//  formerly the bilateral_grid fingerprint slot; after that feature was removed (the 2026-07-17
//  cleanup) a fixed 0 is written and a non-zero value is rejected on load. See Checkpoint.cpp.)
struct CheckpointConfig {
    int32_t sh_degree = 3;
    int32_t factor = 0;
    int32_t test_split = 0;
    int32_t total_iters = 0;
    uint8_t mcmc_enabled = 0;
    uint8_t antialiased = 0;
    int64_t cap_max = 0;      // 0 if mcmc is off
    uint64_t mcmc_seed = 0;   // 0 if mcmc is off
    float scene_scale = 1.0f;
    float ssim_weight = 0.0f;
};

// Adam moment snapshot (corresponds to Trainer::AdamState — defined independently to avoid a
// circular include)
struct CheckpointMoments {
    std::vector<float> exp_avg, exp_avg_sq; // [count·dim] (0 when the moments are unallocated)
};

// Trainer state snapshot — the authoritative state at a step boundary. params and the moments cover
// only alive_count rows (the arena tail is not included — the tail is structurally 0 and is
// re-established on resume).
struct TrainerCheckpoint {
    CheckpointConfig config;
    int32_t iter = 0;              // completed step count = the iter to resume at (0-based re-entry point)
    SplatCloud cloud;              // count = alive_count
    CheckpointMoments moments[6];  // D3 group order: means/quats/scales/opacities/sh0/shN
    // MCMC RNG audit (Task 2 brief): the noise is Philox(seed, offset=iter) — it is **stateless**,
    // so saving iter is enough; but multinomial (mt19937_64) is a **stateful** engine that advances
    // on every refine, so its state must be serialized. The text representation of mt19937_64 is
    // specified by the ISO standard as the sequence of state values — cross-platform reproducible.
    // Empty string if mcmc is off.
    std::string mcmc_rng_state;
};

// Fingerprint verification — saved (the checkpoint) vs current (the resume request). A mismatch
// raises a std::runtime_error naming the field and both values. For sh_degree the message includes
// the §8.1 rationale (rejected unconditionally — there is no relaxation path).
void verify_checkpoint_config(const CheckpointConfig& saved, const CheckpointConfig& current);

// File serialization. save is atomic (temp + rename); failure throws (the temp file removal is
// attempted). load validates the magic, version, and size consistency — a violation or a truncated
// file throws.
void save_checkpoint_file(const TrainerCheckpoint& ck, const std::string& path, bool fp16);
TrainerCheckpoint load_checkpoint_file(const std::string& path);

// Reads the header only (fingerprint + iter + alive_count) — the parameter and moment tensors are
// not loaded (only a few KB are read from a 337 MB 1M fp16 file). Purpose: the Task 5 Swift resume
// treats the checkpoint **file as authoritative** and derives the fingerprint arguments from it
// (sh_degree/factor/test_split/cap_max/...) — this structurally blocks the accident of the app
// remembering the config separately and drifting out of sync (the C surface is sx_checkpoint_info).
// The magic, version, and range validation share the same prefix as load_checkpoint_file (tensor
// size consistency is validated only in the full load).
struct CheckpointHeader {
    CheckpointConfig config;
    int32_t iter = 0;
    int64_t alive_count = 0;
    bool fp16 = false;
    uint32_t version = 0;
};
CheckpointHeader load_checkpoint_header(const std::string& path);

} // namespace splatx
