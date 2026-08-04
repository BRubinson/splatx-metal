// GT cache limit tests (training-speed Task 2 — the pure function sx_gt_cache_limit_bytes). Pure
// CPU — no backend and no fixture needed. Every expected value is hardcoded from a **hand-check**
// (never reuse the implementation's arithmetic — that would be tautological; the MemoryGovernorTests
// rule). The authoritative contract for the policy is the splatx_train_c.h header comment:
// macOS (ios=0) max(4 GiB, 50% of physical RAM) / iOS (ios≠0) fixed 4 GiB (jetsam-conservative).
// Physical RAM is injected by the caller (TrainC) from sysctl hw.memsize — here the value is
// injected directly so that both branches are verified deterministically.
#include "TestHarness.h"

#include "splatx_train_c.h"

#include <cstdint>

SX_TEST(gt_cache_limit_macos_ram_proportional) {
    // 4 GiB floor (hand-check: 4×2^30 = 4,294,967,296)
    // - RAM 0 (guards against the outlier value sysctl returns on failure) → the floor
    SX_CHECK(sx_gt_cache_limit_bytes(0, 0) == 4294967296ull);
    // - RAM 4 GiB: 50% = 2 GiB < 4 GiB → stays at the floor
    SX_CHECK(sx_gt_cache_limit_bytes(4294967296ull, 0) == 4294967296ull);
    // - RAM 8 GiB (8,589,934,592): 50% = exactly 4 GiB — stays at the floor on the boundary (an
    //   8GB device behaves identically to the previous fixed 4 GiB = no regression)
    SX_CHECK(sx_gt_cache_limit_bytes(8589934592ull, 0) == 4294967296ull);
    // - the boundary just past the floor: RAM 8 GiB + 2 B → 50% = 4 GiB + 1 B
    SX_CHECK(sx_gt_cache_limit_bytes(8589934594ull, 0) == 4294967297ull);

    // The RAM-proportional range (hand-check: RAM/2)
    // - 16 GiB (17,179,869,184) → 8 GiB (8,589,934,592)
    SX_CHECK(sx_gt_cache_limit_bytes(17179869184ull, 0) == 8589934592ull);
    // - 24 GiB (25,769,803,776) → 12 GiB (12,884,901,888) — the target machine of design §③
    SX_CHECK(sx_gt_cache_limit_bytes(25769803776ull, 0) == 12884901888ull);

    // Guard at the top end: UINT64_MAX → the /2 comes first, so no wrap: 9,223,372,036,854,775,807
    SX_CHECK(sx_gt_cache_limit_bytes(18446744073709551615ull, 0) == 9223372036854775807ull);
}

SX_TEST(gt_cache_limit_ios_fixed_4gib) {
    // The iOS branch is fixed at 4 GiB regardless of RAM (keeps the previous behavior — jetsam
    // conservatism)
    SX_CHECK(sx_gt_cache_limit_bytes(0, 1) == 4294967296ull);
    SX_CHECK(sx_gt_cache_limit_bytes(8589934592ull, 1) == 4294967296ull);
    SX_CHECK(sx_gt_cache_limit_bytes(25769803776ull, 1) == 4294967296ull);
    SX_CHECK(sx_gt_cache_limit_bytes(18446744073709551615ull, 1) == 4294967296ull);
    // Any nonzero ios flag means iOS (the same convention as the ios argument of sx_suggest_factor)
    SX_CHECK(sx_gt_cache_limit_bytes(25769803776ull, -1) == 4294967296ull);
}

SX_TEST(gt_cache_limit_garden_factor2_target) {
    // Consistent with the design goal (2026-07-21 training-speed diagnosis §③): garden factor 2 on
    // a 24 GB machine. Hand-check of the total-size estimate (the TrainC formula
    // n_views × W×H × 12 B, downscaled_dim(5187,2)=2594, downscaled_dim(3361,2)=1681):
    // 185 × 2594×1681 × 12 = 185 × 4,360,514 × 12
    // = 9,680,341,080 B (≈9.0 GiB — the "9.7 GB" of the diagnosis).
    const uint64_t garden_f2_estimate = 9680341080ull;
    // Under the previous fixed 4 GiB this degraded to off (9.68 GB > 4.29 GB — decoding every step),
    SX_CHECK(garden_f2_estimate > 4294967296ull);
    // whereas with the adaptive limit (24 GiB machine → 12 GiB) it turns on — the very goal of this
    // task.
    SX_CHECK(garden_f2_estimate <= sx_gt_cache_limit_bytes(25769803776ull, 0));
    // iOS stays off (confirms that jetsam conservatism is retained)
    SX_CHECK(garden_f2_estimate > sx_gt_cache_limit_bytes(25769803776ull, 1));
}
