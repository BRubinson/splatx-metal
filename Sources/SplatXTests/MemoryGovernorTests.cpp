// Memory governance tests (Phase 6 round 2 Task 3 — design §7.2/§7.3). Pure CPU — no backend and
// no fixture needed. Every expected value is hardcoded from a **hand-check** (never reuse the
// implementation's arithmetic — that would be tautological).
// The basis for the constants is the MemoryGovernor.h header comment (the bytes/splat tally taken
// from the code + the comparison against the garden footprint).
#include "TestHarness.h"

#include "splatx/MemoryGovernor.h"

#include <stdexcept>

namespace {

template <typename Fn>
bool throws_invalid(Fn&& fn) {
    try {
        fn();
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    }
}

} // namespace

SX_TEST(memory_governor_cap_formula) {
    // Closed form for bytes/splat: 456 + 108K (K=(d+1)² — the sum of the per-item tally in
    // MemoryGovernor.h):
    // deg3 hand-check: P=59 → host 12P=708 + device moments 12P=708 + fwd 100+12·16=292
    //           + bwd 92+24·16=476 = 2184
    SX_CHECK(splatx::resident_bytes_per_splat(0) == 564);
    SX_CHECK(splatx::resident_bytes_per_splat(1) == 888);
    SX_CHECK(splatx::resident_bytes_per_splat(2) == 1428);
    SX_CHECK(splatx::resident_bytes_per_splat(3) == 2184);

    // Hand-check of an injected budget — the garden factor 4 shape (1297×840=1,089,480 px,
    // 185 views, deg3, cache on) with 10 GiB injected. By hand:
    //   usable = 10,737,418,240 / 5 × 4 = 8,589,934,592
    //   gt     = 185 × 1,089,480 × 12   = 2,418,645,600
    //   image  = 1,089,480 × 108        =   117,663,840
    //   isect  = 4 × 1,089,480 × 24     =   104,590,080
    //   base   = 256 MiB                =   268,435,456
    //   cap    = (8,589,934,592 − 2,909,334,976) / 2184 = 2,601,007 (remainder 328)
    const auto s = splatx::suggest_cap_max(10ll << 30, 1297, 840, 185, 3, true);
    SX_CHECK(s.gt_cache_bytes == 2418645600ll);
    SX_CHECK(s.image_bytes == 117663840ll);
    SX_CHECK(s.isect_bytes == 104590080ll);
    SX_CHECK(s.base_bytes == 268435456ll);
    SX_CHECK(s.usable_bytes == 8589934592ll);
    SX_CHECK(s.bytes_per_splat == 2184);
    SX_CHECK(s.raw_cap == 2601007ll);
    SX_CHECK(s.cap_max == 2601007ll);
    SX_CHECK(!s.clamped_min && !s.clamped_max);

    // GT cache off — only one scratch image stays resident: gt = 1×1,089,480×12 = 13,073,760
    //   cap = (8,589,934,592 − 503,763,136) / 2184 = 3,702,459 (remainder 1,000)
    const auto off = splatx::suggest_cap_max(10ll << 30, 1297, 840, 185, 3, false);
    SX_CHECK(off.gt_cache_bytes == 13073760ll);
    SX_CHECK(off.cap_max == 3702459ll);

    // Reproduces the bands of design §7.2 (the same value the fallback would produce is injected,
    // which keeps this deterministic):
    //   macOS 24GB (hw.memsize/2 = 12,884,901,888) garden f4 deg3 → 3,387,631 ∈ [3M, 4M]
    //   iPhone 8GB (assuming os_proc_available_memory 3 GiB) 4032×3024//4=1008×756,
    //   120 views, deg2 → 739,312 ∈ [0.5M, 1M]
    const auto mac = splatx::suggest_cap_max(12884901888ll, 1297, 840, 185, 3, true);
    SX_CHECK(mac.cap_max == 3387631ll);
    const auto ios = splatx::suggest_cap_max(3ll << 30, 1008, 756, 120, 2, true);
    SX_CHECK(ios.cap_max == 739312ll);

    // macOS fallback smoke test — positive, derived from sysctl hw.memsize (the exact value is
    // machine-dependent, so it is not pinned)
    SX_CHECK(splatx::default_memory_budget_bytes() > 0);
}

SX_TEST(memory_governor_cap_clamps) {
    // Floor: a 1 GiB budget against the garden f4 shape — usable(819 MiB) < fixed(2.9 GB) → raw
    // goes negative → clamped to 100k (the caller announces the budget shortfall in the log —
    // TrainC auto-cap)
    const auto lo = splatx::suggest_cap_max(1ll << 30, 1297, 840, 185, 3, true);
    SX_CHECK(lo.raw_cap < 0);
    SX_CHECK(lo.cap_max == splatx::kCapMinClamp);
    SX_CHECK(lo.clamped_min && !lo.clamped_max);

    // Ceiling: a 64 GiB budget + small images → clamped to 4M
    const auto hi = splatx::suggest_cap_max(64ll << 30, 640, 480, 10, 3, true);
    SX_CHECK(hi.cap_max == splatx::kCapMaxClamp);
    SX_CHECK(hi.clamped_max && !hi.clamped_min);

    // Precise hand-check of the boundary where neither clamp fires — a budget solved backwards so
    // that cap lands on exactly 150,000:
    //   100×100, 1 view, cache off, deg3: fixed = 120,000 + 1,080,000 + 960,000 + 268,435,456
    //   = 270,595,456. target usable = fixed + 2184×150,000 = 598,195,456 → budget = ×5/4
    //   = 747,744,320 (divides by 5 exactly, so the integer arithmetic is lossless)
    const auto mid = splatx::suggest_cap_max(747744320ll, 100, 100, 1, 3, false);
    SX_CHECK(mid.usable_bytes == 598195456ll);
    SX_CHECK(mid.cap_max == 150000ll);
    SX_CHECK(!mid.clamped_min && !mid.clamped_max);

    // Argument violations — the throw contract (the C surface converts them to -1: the
    // sx_suggest_* delegation in TrainC.cpp)
    SX_CHECK(throws_invalid([] { splatx::suggest_cap_max(0, 100, 100, 1, 3, false); }));
    SX_CHECK(throws_invalid([] { splatx::suggest_cap_max(1 << 20, 0, 100, 1, 3, false); }));
    SX_CHECK(throws_invalid([] { splatx::suggest_cap_max(1 << 20, 100, 0, 1, 3, false); }));
    SX_CHECK(throws_invalid([] { splatx::suggest_cap_max(1 << 20, 100, 100, 0, 3, false); }));
    SX_CHECK(throws_invalid([] { splatx::suggest_cap_max(1 << 20, 100, 100, 1, 4, false); }));
    SX_CHECK(throws_invalid([] { splatx::suggest_cap_max(1 << 20, 100, 100, 1, -1, false); }));
    SX_CHECK(throws_invalid([] { splatx::resident_bytes_per_splat(4); }));
}

SX_TEST(memory_governor_factor_policy) {
    using P = splatx::ResolutionPlatform;
    // macOS — the smallest factor that satisfies the 1600px cap on the long side (long ≤ 1600×f,
    // §7.3)
    SX_CHECK(splatx::suggest_factor(1600, 1000, P::macOS) == 1);
    SX_CHECK(splatx::suggest_factor(1601, 1000, P::macOS) == 2);
    SX_CHECK(splatx::suggest_factor(1000, 1601, P::macOS) == 2); // height is the long side
    SX_CHECK(splatx::suggest_factor(3200, 2000, P::macOS) == 2);
    SX_CHECK(splatx::suggest_factor(3201, 2000, P::macOS) == 4);
    SX_CHECK(splatx::suggest_factor(5187, 3361, P::macOS) == 4); // garden source res → f4 (bench of record)
    SX_CHECK(splatx::suggest_factor(6400, 4000, P::macOS) == 4);
    SX_CHECK(splatx::suggest_factor(6401, 4000, P::macOS) == 8);
    SX_CHECK(splatx::suggest_factor(12800, 8000, P::macOS) == 8);
    SX_CHECK(splatx::suggest_factor(99999, 3000, P::macOS) == 8); // over even at /8 — best is 8

    // iOS — pinned to a /4 start regardless of resolution (§7.3: '/4 dense > /2 sparse' measured on
    // device, plus the history of /2 diverging. Opening /2 up happens only after position-LR
    // recalibration — the suggest_factor comment in MemoryGovernor.h)
    SX_CHECK(splatx::kIosStartFactor == 4);
    SX_CHECK(splatx::suggest_factor(640, 480, P::iOS) == 4);
    SX_CHECK(splatx::suggest_factor(5187, 3361, P::iOS) == 4);
    SX_CHECK(splatx::suggest_factor(12800, 9600, P::iOS) == 4);

    SX_CHECK(throws_invalid([] { splatx::suggest_factor(0, 100, P::macOS); }));
    SX_CHECK(throws_invalid([] { splatx::suggest_factor(100, 0, P::iOS); }));
}
