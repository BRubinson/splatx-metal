// Memory governance implementation (Phase 6 round 2 Task 3 — design §7.2/§7.3). The basis for the
// formulas and constants is the MemoryGovernor.h header comment (the bytes/splat tally, the garden
// measured cross-check, and the fixed-component breakdown).
#include "splatx/MemoryGovernor.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include <sys/sysctl.h>

namespace splatx {

int64_t resident_bytes_per_splat(int sh_degree) {
    if (sh_degree < 0 || sh_degree > 3)
        throw std::invalid_argument("resident_bytes_per_splat: sh_degree must be in [0,3]");
    // Tally taken from the code (K=(d+1)², P=11+3K — source per item):
    //   host arena 12P B (Trainer.cpp ctor: the 6 cloud vectors 4P + AdamState exp_avg/exp_avg_sq 8P)
    //   device p_/m1_/m2_ 12P B (MetalBackend.cpp MetalTrainStepExecutor ctor)
    //   device fwd 100+12K B (scales_lin 12 + opacs_act 4 + radii 8 + means2d 8 + depths 4
    //     + conics 12 + valid 1 + dirs 12 + colors 12 + relu_mask 3 + tiles_per_gauss 4
    //     + cum 8 + noise 12 = 100; coeffs 12K. block_sums is 8/1024 B per row — ignored)
    //   device bwd 92+24K B (v_means2d 8 + v_conics 12 + v_colors 12 + v_opacities 4
    //     + v_dirs 12 + v_means 12 + v_quats 16 + v_scales 12 + v_sh0 12 + v_depths_zero 4
    //     = 104; v_coeffs 12K + v_shN 12(K−1) = 24K−12 → total 92+24K)
    //   grand total 24P + 192 + 36K = 456 + 108K (deg 0/1/2/3 = 564/888/1428/2184 B)
    // AA on adds +12 B (comps/opacs_aa/v_comp — off by default, absorbed by the 4/5 safety margin).
    // Measured cross-check: the footprint increment 2.41 KB/splat (cap 1M→2M) ⊇ 2184 B + isect growth + slack.
    const int64_t K = static_cast<int64_t>(sh_degree + 1) * (sh_degree + 1);
    return 456 + 108 * K;
}

int64_t default_memory_budget_bytes() {
    // macOS fallback — a conservative 50% of physical RAM (hw.memsize). Basis:
    // - On Apple Silicon UMA the GPU working set draws from the same pool
    //   (recommendedMaxWorkingSetBytes is roughly 60–75% of physical). Once the OS, the file cache,
    //   the compressor, and other apps take their share, the amount a training process can hold
    //   resident without swapping or page compression sits safely at about half.
    // - Measured on a 24GB M4: at cap 2M the 7.22 GB footprint runs to completion with room to spare
    //   (60% of the 12GB budget) — docs/bench/2026-07-12-phase3.5-regate.md.
    // iOS does not use this fallback — the app injects os_proc_available_memory() (the headroom left
    // under the jetsam limit) into suggest_cap_max (MemoryGovernor.h — the first-class path).
    uint64_t memsize = 0;
    size_t len = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0) != 0 || memsize == 0)
        return 4ll << 30; // sysctl failure (abnormal) — half of an assumed 8GB machine, the minimum conservative value
    return static_cast<int64_t>(memsize / 2);
}

CapMaxSuggestion suggest_cap_max(int64_t budget_bytes, int image_w, int image_h,
                                 int n_gt_images, int sh_degree, bool gt_cache_on) {
    if (budget_bytes < 1)
        throw std::invalid_argument("suggest_cap_max: budget_bytes must be >= 1");
    if (image_w < 1 || image_h < 1)
        throw std::invalid_argument("suggest_cap_max: image_w/image_h must be >= 1");
    if (n_gt_images < 1)
        throw std::invalid_argument("suggest_cap_max: n_gt_images must be >= 1");

    CapMaxSuggestion s;
    s.bytes_per_splat = resident_bytes_per_splat(sh_degree); // includes sh_degree validation
    const int64_t hw = static_cast<int64_t>(image_w) * image_h;

    // Fixed components (the breakdown in the header comment — independent of the splat count,
    // deducted from the budget up front)
    s.gt_cache_bytes = (gt_cache_on ? static_cast<int64_t>(n_gt_images) : 1) * hw *
                       kGtBytesPerPixel;
    s.image_bytes = hw * kImageBufBytesPerPixel;
    s.isect_bytes = kIsectsPerPixel * hw * kIsectBytesPerEntry;
    s.base_bytes = kBaseOverheadBytes;

    // 4/5 safety margin — integer arithmetic (budget/5*4: no overflow, and deterministic for the
    // test hand-check)
    s.usable_bytes = budget_bytes / kCapSafetyDenom * kCapSafetyNumer;

    s.raw_cap = (s.usable_bytes - s.fixed_bytes()) / s.bytes_per_splat;
    s.cap_max = std::clamp(s.raw_cap, kCapMinClamp, kCapMaxClamp);
    s.clamped_min = s.raw_cap < kCapMinClamp;
    s.clamped_max = s.raw_cap > kCapMaxClamp;
    return s;
}

int suggest_factor(int image_w, int image_h, ResolutionPlatform platform) {
    if (image_w < 1 || image_h < 1)
        throw std::invalid_argument("suggest_factor: image_w/image_h must be >= 1");
    if (platform == ResolutionPlatform::iOS) {
        // §7.3 — iOS is fixed to start at /4. Measured in the iOS capture app: under the memory
        // budget, "/4 dense beats /2 sparse", plus a history of divergence at /2. Opening up /2 or
        // finer is allowed only after the position-LR recalibration is complete (the suggest_factor
        // comment in MemoryGovernor.h — this returns a constant regardless of resolution).
        return kIosStartFactor;
    }
    // macOS — the smallest factor that satisfies the 1600px long-side cap (the long ≤ 1600×f test is
    // consistent with the round(orig/f) common practice of the downsample folders and avoids floating
    // point). garden 5187×3361 → 4 (the bench of record).
    const int64_t long_side = std::max(image_w, image_h);
    for (int f : {1, 2, 4, 8})
        if (long_side <= static_cast<int64_t>(kMacLongSideMaxPx) * f) return f;
    return 8; // still over the cap even at /8 — the best the COLMAP images_N folder convention allows (header comment)
}

} // namespace splatx
