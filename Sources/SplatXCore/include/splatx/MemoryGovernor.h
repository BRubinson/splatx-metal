#pragma once
// Memory governance (Phase 6 round 2 Task 3 — design §7.2 automatic cap_max sizing + §7.3
// resolution policy).
//
// cap_max is not a fixed value; it is computed from a device memory budget formula (§7.2). For the
// budget (available memory), **caller injection is the first-class path**: the iOS app passes
// os_proc_available_memory() (the headroom up to the jetsam limit — the real ceiling of the app
// process). The macOS CLI fallback is default_memory_budget_bytes() (a conservative estimate based
// on sysctl hw.memsize — rationale in the implementation comments).
//
// ── Premise (important) ──
// The constants in the formula are **measured against the current implementation**: fp32
// parameters + fp32 Adam moments + the device-resident executor (on by default). The "iOS degree 2
// + fp16 moments ≈ 380B/splat" of design §7.1 holds only once resident fp16 moments are
// implemented (out of scope — a follow-up), and MUST NOT be pre-applied to this formula.
// When fp16 moments land, only resident_bytes_per_splat needs updating (the structure of the
// formula is unchanged).
//
// ── Bytes/splat tally (code-derived — itemized in the resident_bytes_per_splat implementation
//    comments) ──
// K = (sh_degree+1)², P = 11+3K (parameter floats per row: means3+quats4+scales3+opac1+
// sh0·3+shN·3(K-1)). Resident per splat:
//   host arena (Trainer ctor, mcmc on):        cloud 4P + Adam(exp_avg+exp_avg_sq) 8P = 12P
//   device params+moments (resident executor): p_/m1_/m2_, 3 copies = 12P
//   device fwd intermediate buffers:           100 + 12K
//   device bwd grad buffers:                   92 + 24K
//   total: 24P + 192 + 36K = **456 + 108K** B/splat
//   (deg 0/1/2/3 = 564 / 888 / 1428 / 2184 B. AA on adds +12 B — off by default, absorbed by the
//    safety margin.)
//
// ── Cross-check against measurements (garden factor 4 = 1297×840, 185 views, deg3,
//    resident+SSIM, /usr/bin/time -l) ──
//   cap 1M: RSS 3.86 GB / footprint 4.81 GB   (docs/bench/2026-07-12-v01-final-bench.md)
//   cap 2M: RSS 5.16 GB / footprint 7.22 GB   (docs/bench/2026-07-12-phase3.5-regate.md)
//   Increment: footprint (7.22−4.81)/1M — the "GB" in the bench tables is the **GiB conversion**
//   of the time -l byte counts (7,755,961,768 B = 7.22 GiB), so the increment is 2.41 GiB/1M ≈
//   **2.59 KB/splat** — **~18.6% of headroom** over the 2.184 KB code tally (isect growth +
//   allocator slack; Task 4 carry-over ③ correction — the previous "2.41 KB, ~10%" came from
//   dividing GiB by a decimal GB by mistake). The RSS increment of 1.3 KB/splat **under-counts**
//   because of compressed memory (the moment arena has many zero pages, and so on) — the formula
//   uses the code tally, which is consistent with footprint.
//   Absolute-value hand-check: the cap 2M model = 2184×2M + the fixed components (below)
//   ≈ 7.2 GB ≈ footprint 7.22 GB.
//
// ── Fixed components independent of the splat count (suggest_cap_max deducts these from the
//    budget up front) ──
//   GT cache:      n_gt_images × W×H × 12 B (ImageF32 3ch fp32 — TrainC keeps every train+test
//                  view resident, decoded once per view; if off, a single scratch image)
//   image buffers: W×H × 108 B (render/GT/grad 6ch 48 + the 5 SSIM maps 60 — assumes the default
//                  λ=0.2 is on; with λ=0 this over-estimates, but in the conservative direction)
//   isect:         kIsectsPerPixel × W×H × 24 B (radix key/value double buffers 2×8+2×4 —
//                  proportional to n_isects and therefore resolution dependent. garden f4
//                  steady-state measured 0.88 isect/px (n_isects 957,884 @1M, from the log in
//                  docs/bench/2026-07-11-garden-7k-bench.md) × 1.5x capacity growth slack +
//                  headroom for scene and convergence-stage variation → coefficient 4)
//   base:          256 MiB (process + Metal library + COLMAP/decode transient buffers — observed
//                  as the residual after subtracting the N and GT components from the 0.43 GB
//                  peak RSS of a small mcmc-off run)
//
// Formula: cap = clamp((budget×4/5 − fixed) / bytes_per_splat, 100k, 4M)
//   (4/5 = an 80% safety margin — it absorbs AA, the λ=0 deviation, and allocator slack. Integer
//    arithmetic budget/5*4 — for determinism of the hand-check tests.)
#include <cstdint>

namespace splatx {

// ===== Policy constants (§7.2/§7.3 — the single source of truth tests and logs reference) =====
inline constexpr int64_t kCapSafetyNumer = 4;  // safety margin 4/5 = 80% of the budget
inline constexpr int64_t kCapSafetyDenom = 5;
inline constexpr int64_t kCapMinClamp = 100'000;    // lower bound — below this, training is pointless
inline constexpr int64_t kCapMaxClamp = 4'000'000;  // upper bound — top of the §7.2 macOS 24GB band
inline constexpr int64_t kGtBytesPerPixel = 12;     // ImageF32 3ch fp32
inline constexpr int64_t kImageBufBytesPerPixel = 108; // 48 (common) + 60 (the 5 SSIM maps)
inline constexpr int64_t kIsectsPerPixel = 4;       // measured 0.88/px + growth slack headroom (header comment)
inline constexpr int64_t kIsectBytesPerEntry = 24;  // keys 2×8 + vals 2×4 (radix double buffers)
inline constexpr int64_t kBaseOverheadBytes = 256ll << 20; // process/Metal/decode transient buffers

// §7.3 resolution policy constants
inline constexpr int kMacLongSideMaxPx = 1600; // macOS: 1600px cap on the long side
inline constexpr int kIosStartFactor = 4;      // iOS: fixed /4 start (see the suggest_factor comment)

// Breakdown of the sizing rationale — the caller's log ("auto-cap: ...") and the hand-check tests
// verify it item by item.
struct CapMaxSuggestion {
    int64_t cap_max = 0;         // the final suggestion (after clamping)
    int64_t raw_cap = 0;         // before clamping: (usable−fixed)/bytes_per_splat — may be negative
    int64_t bytes_per_splat = 0; // resident_bytes_per_splat(sh_degree)
    int64_t usable_bytes = 0;    // budget × 4/5 (integer arithmetic budget/5*4)
    int64_t gt_cache_bytes = 0;  // fixed-component breakdown ↓
    int64_t image_bytes = 0;
    int64_t isect_bytes = 0;
    int64_t base_bytes = 0;
    bool clamped_min = false;
    bool clamped_max = false;
    int64_t fixed_bytes() const {
        return gt_cache_bytes + image_bytes + isect_bytes + base_bytes;
    }
};

// Resident bytes per splat (the code tally from the header comment — closed form 456 + 108K,
// K=(sh_degree+1)²).
// sh_degree ∉ [0,3] throws.
int64_t resident_bytes_per_splat(int sh_degree);

// The macOS budget fallback — a conservative estimate based on sysctl hw.memsize (rationale in the
// implementation comments). Do not use it on iOS — there the app injects
// os_proc_available_memory() directly into suggest_cap_max (the first-class path).
int64_t default_memory_budget_bytes();

// Automatic cap_max sizing (§7.2 — the formula in the header comment). image_w/h are the
// **training resolution** (after factor is applied), and n_gt_images is the number of resident GT
// views (the TrainC GT cache holds every train+test view — the total camera count).
// budget_bytes <= 0, image_w·h < 1, n_gt_images < 1, or sh_degree ∉ [0,3] throws.
CapMaxSuggestion suggest_cap_max(int64_t budget_bytes, int image_w, int image_h,
                                 int n_gt_images, int sh_degree, bool gt_cache_on);

// The §7.3 resolution policy platform axis.
enum class ResolutionPlatform { macOS, iOS };

// Suggests the downsample factor (§7.3). image_w/h are the **source resolution** (factor 1).
// - macOS: a 1600px cap on the long side — picks the smallest factor (1|2|4|8) that satisfies it
//   (the test is long ≤ 1600×f — consistent with the round(orig/f) convention of the downsample
//   folders). For an oversized input that still exceeds the cap at /8, the best available value 8
//   is returned (the COLMAP images_N folder convention goes up to 8 — the sx_train_run contract).
// - iOS: a **fixed /4 start constant** (kIosStartFactor). Measured in the iOS capture app: "under
//   the memory budget, /4 dense beats /2 sparse", plus the **history of divergence at /2** (§7.3) —
//   opening up higher resolutions (/2 and above) is permitted only after position-LR
//   recalibration (recalibrating the pixel scale per resolution) is complete.
//   Until then iOS returns 4 regardless of the resolution.
// image_w/h < 1 throws.
int suggest_factor(int image_w, int image_h, ResolutionPlatform platform);

} // namespace splatx
