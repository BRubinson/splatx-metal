// Phase 2 Task 3 — the MCMC strategy (host orchestration) + Adam state surgery.
// Derivation basis: gsplat strategy/mcmc.py:139–239 + strategy/ops.py:301–401 (commit 77ab983).
// For the Eq.9/perturb kernels themselves the golden gates in McmcTests
// (relocation_basic/perturb_basic) are the reference — what is verified here is the index flow
// (dead/sampled/new), the state surgery convention, arena growth, and determinism. (a)–(c) use the
// CPU backend (pure verification of the host logic); (d) uses the Metal backend (end-to-end
// determinism of rng_normal→perturb→relocation).
#include "TestHarness.h"
#include "TestUtils.h" // shared_metal_backend / count_bitwise_mismatch
#include "splatx/CpuKernels.h"
#include "splatx/KernelBackend.h"
#include "splatx/Trainer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using namespace splatx;

namespace {

// A small cloud in which every component differs from row to row — sh_degree 1 (an shN row =
// 3·3 = 9, so the copy path is covered).
// opacity is given as a post-sigmoid value (storing it in logit space is the internal convention).
SplatCloud make_cloud(const std::vector<float>& op_act) {
    const int64_t n = static_cast<int64_t>(op_act.size());
    SplatCloud c;
    c.count = n;
    c.sh_degree = 1;
    c.means.resize(n * 3);
    c.scales.resize(n * 3);
    c.quats.assign(n * 4, 0.0f);
    c.opacities.resize(n);
    c.sh0.resize(n * 3);
    c.shN.resize(n * 9);
    for (int64_t i = 0; i < n; i++) {
        for (int k = 0; k < 3; k++) {
            c.means[i * 3 + k] = static_cast<float>(i) + 0.1f * static_cast<float>(k);
            c.scales[i * 3 + k] =
                std::log(0.1f + 0.01f * static_cast<float>(i) + 0.005f * static_cast<float>(k));
            c.sh0[i * 3 + k] = 0.05f * static_cast<float>(i) + 0.01f * static_cast<float>(k);
        }
        c.quats[i * 4 + 0] = 1.0f;
        c.quats[i * 4 + 1] = 0.01f * static_cast<float>(i);
        c.opacities[i] = logit_f32(op_act[static_cast<size_t>(i)]);
        for (int k = 0; k < 9; k++)
            c.shN[i * 9 + k] = -0.02f * static_cast<float>(i) - 0.001f * static_cast<float>(k);
    }
    return c;
}

constexpr Trainer::ParamGroup kGroups[6] = {
    Trainer::ParamGroup::Means, Trainer::ParamGroup::Quats,     Trainer::ParamGroup::Scales,
    Trainer::ParamGroup::Sh0,   Trainer::ParamGroup::Opacities, Trainer::ParamGroup::ShN};
// Elements per row as make_cloud lays them out (same order as kGroups): means/scales/sh0 3,
// quats 4, opac 1, shN 9
constexpr int64_t kDims[6] = {3, 4, 3, 3, 1, 9};

} // namespace

// (a) The relocation index flow + Eq.9 — with a single alive (0.9) and 3 dead (<=0.005), the
// sampled source is deterministic regardless of the RNG (all 0) and ratio = bincount+1 = 4.
// Checks that the dead Gaussians really do teleport and that opacity and scale on both the source
// and the dead rows come out at the Eq.9 values (the CPU kernel that already passes the
// relocation_basic golden is reused as the expected-value generator — what is under verification
// is the strategy's gather/scatter index flow).
SX_TEST(mcmc_strategy_relocates_dead_to_eq9) {
    auto cpu = make_cpu_backend();
    SplatCloud c = make_cloud({0.9f, 0.004f, 0.002f, 0.0049f});
    const SplatCloud before = c;
    TrainConfig cfg;
    cfg.total_iters = 100;
    cfg.mcmc.enabled = true;
    cfg.mcmc.cap_max = 4; // = N → suppresses add (isolates relocation: ceil(1.05·4)=5 > cap)
    Trainer tr(*cpu, std::move(c), cfg);
    const Trainer::McmcRefineResult res = tr.mcmc_refine();

    SX_CHECK((res.relocated_dead == std::vector<int64_t>{1, 2, 3}));
    SX_CHECK((res.relocation_src == std::vector<int64_t>{0, 0, 0}));
    SX_CHECK(res.added_src.empty());
    const SplatCloud& after = tr.cloud();
    SX_CHECK(after.count == 4);

    // The closed form of the Eq.9 opacity (RelocationCUDA.cu:57–62): 1 - (1-o)^(1/4) — in the
    // range where no clamp fires
    const double o0 = sigmoid_f32(before.opacities[0]);
    const double expect_op = 1.0 - std::pow(1.0 - o0, 0.25);
    for (int i = 0; i < 4; i++)
        SX_CHECK_NEAR(sigmoid_f32(after.opacities[i]), expect_op, 1e-5, 1e-6);

    // scale — the expected value is produced by calling the CPU kernel directly (source 0's
    // post-sigmoid/linear values, ratio=4)
    const std::vector<float> lut = make_binoms_lut(51);
    const float src_op = sigmoid_f32(before.opacities[0]);
    float src_sc[3];
    for (int k = 0; k < 3; k++) src_sc[k] = std::exp(before.scales[k]);
    const int32_t ratio = 4;
    RelocationInputs rin;
    rin.opacities = &src_op;
    rin.scales = src_sc;
    rin.ratios = &ratio;
    rin.binoms = lut.data();
    rin.N = 1;
    const RelocationOutputs expect = mcmc_relocation_cpu(rin);
    SX_CHECK_NEAR(sigmoid_f32(after.opacities[0]), expect.new_opacities[0], 1e-6, 1e-7);
    for (int k = 0; k < 3; k++)
        SX_CHECK_NEAR(after.scales[k], std::log(expect.new_scales[k]), 1e-6, 1e-6);

    // A dead row = a bitwise copy of the **updated** source row (ops.py:337–343 — the copy happens
    // after the update, so the dead rows receive the new opacity/scale too; means/quats/sh stay
    // exactly as the source had them)
    for (int i = 1; i < 4; i++) {
        SX_CHECK(std::memcmp(&after.opacities[i], &after.opacities[0], 4) == 0);
        SX_CHECK(std::memcmp(&after.scales[i * 3], &after.scales[0], 12) == 0);
        SX_CHECK(std::memcmp(&after.means[i * 3], &after.means[0], 12) == 0);
        SX_CHECK(std::memcmp(&after.quats[i * 4], &after.quats[0], 16) == 0);
        SX_CHECK(std::memcmp(&after.sh0[i * 3], &after.sh0[0], 12) == 0);
        SX_CHECK(std::memcmp(&after.shN[i * 9], &after.shN[0], 36) == 0);
    }
    // The source's means/quats/sh are unchanged (relocation updates opacity and scale only)
    SX_CHECK(std::memcmp(&after.means[0], &before.means[0], 12) == 0);
    SX_CHECK(std::memcmp(&after.quats[0], &before.quats[0], 16) == 0);
    SX_CHECK(std::memcmp(&after.sh0[0], &before.sh0[0], 12) == 0);
    SX_CHECK(std::memcmp(&after.shN[0], &before.shN[0], 36) == 0);
}

// (b) The index convention of the Adam state surgery — compared line by line against the upstream
// gsplat original (ops.py):
// relocate resets **only the sampled source** to 0 (the dead rows are kept — the optimizer_fn at
// ops.py:345–347 does nothing but v[sampled_idxs]=0 and never touches dead_indices), while add
// zeroes **only the new rows** (the source is kept — ops.py:389–391 is nothing but a cat of
// zeros). That the two operations follow different conventions is the criterion being judged.
SX_TEST(mcmc_adam_surgery_indices_follow_gsplat) {
    auto cpu = make_cpu_backend();

    // — Scenario A: relocate (source row 0 reset / dead rows 1..3 kept)
    {
        TrainConfig cfg;
        cfg.total_iters = 100;
        cfg.mcmc.enabled = true;
        cfg.mcmc.cap_max = 4;
        Trainer tr(*cpu, make_cloud({0.9f, 0.004f, 0.002f, 0.0049f}), cfg);
        for (auto g : kGroups) {
            Trainer::AdamState& st = tr.adam_state(g);
            std::fill(st.exp_avg.begin(), st.exp_avg.end(), 7.0f);
            std::fill(st.exp_avg_sq.begin(), st.exp_avg_sq.end(), 3.0f);
        }
        const Trainer::McmcRefineResult res = tr.mcmc_refine();
        SX_CHECK((res.relocation_src == std::vector<int64_t>{0, 0, 0}));
        for (int gi = 0; gi < 6; gi++) {
            Trainer::AdamState& st = tr.adam_state(kGroups[gi]);
            const int64_t d = kDims[gi];
            for (int64_t k = 0; k < d; k++) { // source row 0: both moments reset to 0
                SX_CHECK(st.exp_avg[k] == 0.0f);
                SX_CHECK(st.exp_avg_sq[k] == 0.0f);
            }
            for (int64_t row = 1; row < 4; row++) // dead rows: not reset, per the original (kept)
                for (int64_t k = 0; k < d; k++) {
                    SX_CHECK(st.exp_avg[row * d + k] == 7.0f);
                    SX_CHECK(st.exp_avg_sq[row * d + k] == 3.0f);
                }
        }
    }

    // — Scenario B: add (source kept / new rows 0) + the parameter copy and Eq.9 with ratio=2
    {
        TrainConfig cfg;
        cfg.total_iters = 100;
        cfg.mcmc.enabled = true;
        cfg.mcmc.cap_max = 3; // ceil(1.05·2) = 3 → n_add = 1
        Trainer tr(*cpu, make_cloud({0.7f, 0.3f}), cfg);
        const SplatCloud snap = tr.cloud(); // snapshot after the arena grows, before refine
        for (auto g : kGroups) {
            Trainer::AdamState& st = tr.adam_state(g);
            std::fill(st.exp_avg.begin(), st.exp_avg.end(), 7.0f);
            std::fill(st.exp_avg_sq.begin(), st.exp_avg_sq.end(), 3.0f);
        }
        const Trainer::McmcRefineResult res = tr.mcmc_refine();
        SX_CHECK(res.relocated_dead.empty()); // no dead rows → relocate is a no-op
        SX_CHECK(res.added_src.size() == 1);
        SX_CHECK(tr.cloud().count == 3);
        const int64_t s = res.added_src.empty() ? 0 : res.added_src[0];
        SX_CHECK(s == 0 || s == 1);

        // ratio = bincount+1 = 2 → new_op = 1 - (1-o)^(1/2) — applied to source and new row alike
        const SplatCloud& after = tr.cloud();
        const double o_s = sigmoid_f32(snap.opacities[s]);
        SX_CHECK_NEAR(sigmoid_f32(after.opacities[s]), 1.0 - std::sqrt(1.0 - o_s), 1e-5, 1e-6);
        SX_CHECK(std::memcmp(&after.opacities[2], &after.opacities[s], 4) == 0);
        SX_CHECK(std::memcmp(&after.scales[2 * 3], &after.scales[s * 3], 12) == 0);
        SX_CHECK(std::memcmp(&after.means[2 * 3], &snap.means[s * 3], 12) == 0);
        SX_CHECK(std::memcmp(&after.quats[2 * 4], &snap.quats[s * 4], 16) == 0);
        SX_CHECK(std::memcmp(&after.sh0[2 * 3], &snap.sh0[s * 3], 12) == 0);
        SX_CHECK(std::memcmp(&after.shN[2 * 9], &snap.shN[s * 9], 36) == 0);
        const int64_t oth = 1 - s; // the non-source row is entirely unchanged
        SX_CHECK(std::memcmp(&after.opacities[oth], &snap.opacities[oth], 4) == 0);
        SX_CHECK(std::memcmp(&after.scales[oth * 3], &snap.scales[oth * 3], 12) == 0);

        for (int gi = 0; gi < 6; gi++) {
            Trainer::AdamState& st = tr.adam_state(kGroups[gi]);
            const int64_t d = kDims[gi];
            for (int64_t row = 0; row < 2; row++) // existing rows (source included): sentinel kept
                for (int64_t k = 0; k < d; k++) {
                    SX_CHECK(st.exp_avg[row * d + k] == 7.0f);
                    SX_CHECK(st.exp_avg_sq[row * d + k] == 3.0f);
                }
            for (int64_t k = 0; k < d; k++) { // new row 2: explicitly 0 (verifies that the arena
                SX_CHECK(st.exp_avg[2 * d + k] == 0.0f); // tail the test dirtied is reset too)
                SX_CHECK(st.exp_avg_sq[2 * d + k] == 0.0f);
            }
        }
    }
}

// (c) The alive_count growth trajectory (integer arithmetic for ceil(1.05N) + the cap_max ceiling)
// + no arena reallocation + the refine schedule (ratio-based) + the constructor/misuse guards.
SX_TEST(mcmc_growth_trajectory_capped_and_schedule) {
    auto cpu = make_cpu_backend();
    TrainConfig cfg;
    cfg.total_iters = 3000;
    cfg.mcmc.enabled = true;
    cfg.mcmc.cap_max = 24;
    Trainer tr(*cpu, make_cloud(std::vector<float>(16, 0.5f)), cfg);

    // The arena contract (design §5.2): the vectors are preallocated once at cap rows
    SX_CHECK(tr.cloud().means.size() == 24u * 3u);
    SX_CHECK(tr.cloud().opacities.size() == 24u);
    SX_CHECK(tr.cloud().shN.size() == 24u * 9u);
    const float* arena0 = tr.cloud().means.data();

    // Hand-check of the trajectory: ceil(21N/20) = (21N+19)/20 →
    // 16→17→18→19→20→21→23(22.05↑)→24(cap)→24
    const int64_t expect[9] = {17, 18, 19, 20, 21, 23, 24, 24, 24};
    for (int i = 0; i < 9; i++) {
        const int64_t pre = tr.cloud().count;
        const Trainer::McmcRefineResult r = tr.mcmc_refine();
        SX_CHECK(tr.cloud().count == expect[i]);
        SX_CHECK(tr.cloud().count <= 24);
        SX_CHECK(static_cast<int64_t>(r.added_src.size()) == expect[i] - pre);
        if (pre == 24) SX_CHECK(r.added_src.empty()); // once cap is reached, n_add = 0
    }
    SX_CHECK(tr.cloud().means.data() == arena0); // no realloc while growing (avoids gsplat #444)

    // The refine schedule (ratios → iter boundaries): total 3000 → start 50, stop 2500, every 10
    // (the gsplat defaults 500/25000/100 @ 30k converted to ratios — the strict inequalities of
    // mcmc.py:159–163)
    SX_CHECK(!tr.mcmc_refine_due(0));
    SX_CHECK(!tr.mcmc_refine_due(50));   // iter > start — the boundary is excluded
    SX_CHECK(!tr.mcmc_refine_due(55));   // not a multiple of every
    SX_CHECK(tr.mcmc_refine_due(60));
    SX_CHECK(tr.mcmc_refine_due(2490));
    SX_CHECK(!tr.mcmc_refine_due(2500)); // iter < stop — the boundary is excluded
    SX_CHECK(!tr.mcmc_refine_due(2510));

    // Constructor guard: cap_max < the initial count → throw
    bool threw = false;
    TrainConfig bad = cfg;
    bad.mcmc.cap_max = 8;
    try {
        Trainer t2(*cpu, make_cloud(std::vector<float>(16, 0.5f)), bad);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    SX_CHECK(threw);

    // mcmc off: refine_due is always false, and refine/adam_state throw to block misuse
    TrainConfig off;
    off.total_iters = 100;
    Trainer t3(*cpu, make_cloud({0.5f, 0.5f}), off);
    SX_CHECK(!t3.mcmc_refine_due(10));
    threw = false;
    try {
        (void)t3.mcmc_refine();
    } catch (const std::logic_error&) {
        threw = true;
    }
    SX_CHECK(threw);
}

// (d) Determinism — two runs with the same seed and the same call order (refine + noise on every
// step) leave every parameter bit-identical (Metal end to end: multinomial=mt19937_64 on the host
// + Philox rng_normal(offset=iter) + the relocation/perturb kernels). Non-vacuity guard: confirms
// that relocation, growth, and noise actually happened.
SX_TEST(mcmc_determinism_fixed_seed_bitwise) {
    KernelBackend& mtl = sxtest::shared_metal_backend();
    const std::vector<float> ops = {0.9f,  0.5f, 0.003f, 0.2f,  0.7f,  0.004f,
                                    0.35f, 0.6f, 0.15f,  0.8f, 0.45f, 0.25f};
    TrainConfig cfg;
    cfg.total_iters = 300;
    cfg.mcmc.enabled = true;
    cfg.mcmc.cap_max = 20;
    cfg.mcmc.seed = 20260711;

    auto run = [](Trainer& t) {
        int64_t relocated = 0;
        for (int it = 0; it < 12; it++) {
            if (it % 3 == 0)
                relocated += static_cast<int64_t>(t.mcmc_refine().relocated_dead.size());
            t.mcmc_inject_noise(it);
        }
        return relocated;
    };
    Trainer a(mtl, make_cloud(ops), cfg);
    Trainer b(mtl, make_cloud(ops), cfg);
    const int64_t ra = run(a), rb = run(b);

    SX_CHECK(ra == rb);
    SX_CHECK(ra >= 2);              // the first refine relocates the dead (idx 2, 5) — non-vacuous
    SX_CHECK(a.cloud().count == b.cloud().count);
    SX_CHECK(a.cloud().count > 12); // drives the add path (12→13→14→15→16)

    auto same = [](const std::vector<float>& x, const std::vector<float>& y) {
        return x.size() == y.size() &&
               (x.empty() || std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0);
    };
    SX_CHECK(same(a.cloud().means, b.cloud().means));
    SX_CHECK(same(a.cloud().scales, b.cloud().scales));
    SX_CHECK(same(a.cloud().quats, b.cloud().quats));
    SX_CHECK(same(a.cloud().opacities, b.cloud().opacities));
    SX_CHECK(same(a.cloud().sh0, b.cloud().sh0));
    SX_CHECK(same(a.cloud().shN, b.cloud().shN));

    // The noise injection actually lands — means differ from the initial cloud (non-vacuous)
    const SplatCloud init = make_cloud(ops);
    SX_CHECK(sxtest::count_bitwise_mismatch(a.cloud().means.data(), init.means.data(),
                                            static_cast<size_t>(init.count) * 3) > 0);
}
