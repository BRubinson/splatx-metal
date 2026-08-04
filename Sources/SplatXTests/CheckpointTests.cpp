// Phase 6 round 2 Task 2 — incremental checkpoints (design §7.5/§8.1; the basis for the format and
// the modes is Checkpoint.h).
// (a) an fp32 file round trip is bit-lossless + config fingerprint verification + corrupt files
//     are rejected
// (b) resume continuity: "20 steps straight through" vs "10 steps→save→load→10 steps" — the
//     forward of the first resumed step is **bit-identical** (the deterministic region) + an rel_l2
//     gate on the steps after it (the atomic bwd basis — see the comment below)
// (c) the fp16 quality gate (the round-trip rel error, measured) + the paired-zeroing rule + the
//     save time of a synthetic 1M state (recorded against the §7.5 3-second contract)
#include "TestHarness.h"
#include "TestPaths.h"
#include "TestUtils.h"
#include "splatx/Checkpoint.h"
#include "splatx/Trainer.h"

#include <TargetConditionals.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace splatx;

namespace {

// The dim of each group in D3 order (identical to group_dim in Checkpoint.cpp — a test-local
// hand-check)
int64_t ckt_dim(int g, int sh_degree) {
    constexpr int64_t fixed[5] = {3, 4, 3, 1, 3};
    if (g < 5) return fixed[g];
    return (static_cast<int64_t>(sh_degree) + 1) * (sh_degree + 1) * 3 - 3;
}

const std::vector<float>* ckt_group(const SplatCloud& c, int g) {
    const std::vector<float>* v[6] = {&c.means, &c.quats, &c.scales,
                                      &c.opacities, &c.sh0, &c.shN};
    return v[g];
}

std::vector<float>* ckt_group(SplatCloud& c, int g) {
    return const_cast<std::vector<float>*>(ckt_group(static_cast<const SplatCloud&>(c), g));
}

// A deterministic synthetic checkpoint (for the file round trip — no GPU needed)
TrainerCheckpoint make_synthetic_ck(int64_t n, int sh_degree, bool with_rng) {
    TrainerCheckpoint ck;
    ck.config.sh_degree = sh_degree;
    ck.config.factor = 4;
    ck.config.test_split = 8; // the split convention in the fingerprint (review fix 1)
    ck.config.total_iters = 30000;
    ck.config.mcmc_enabled = with_rng ? 1 : 0;
    ck.config.antialiased = 1;
    ck.config.cap_max = with_rng ? n * 2 : 0;
    ck.config.mcmc_seed = with_rng ? 42 : 0;
    ck.config.scene_scale = 4.5f;
    ck.config.ssim_weight = 0.2f;
    ck.iter = 7000;
    ck.cloud.count = n;
    ck.cloud.sh_degree = sh_degree;
    std::mt19937 rng(20260712);
    std::uniform_real_distribution<float> u(-3.0f, 3.0f);
    std::uniform_real_distribution<float> upos(0.0f, 1.0f);
    for (int g = 0; g < 6; g++) {
        const size_t cnt = static_cast<size_t>(n) * ckt_dim(g, sh_degree);
        std::vector<float>* v = ckt_group(ck.cloud, g);
        v->resize(cnt);
        for (float& x : *v) x = u(rng);
        ck.moments[g].exp_avg.resize(cnt);
        ck.moments[g].exp_avg_sq.resize(cnt);
        for (float& x : ck.moments[g].exp_avg) x = 1e-3f * u(rng);
        // exp_avg_sq: approximates the real distribution — log-uniform over 1e-12..1e-4 (the
        // region the fp16 sqrt compression targets)
        for (float& x : ck.moments[g].exp_avg_sq)
            x = std::pow(10.0f, -12.0f + 8.0f * upos(rng));
    }
    if (with_rng) {
        std::mt19937_64 eng(42);
        for (int i = 0; i < 5; i++) (void)eng(); // mid-stream state (advanced, then serialized)
        std::ostringstream oss;
        oss << eng;
        ck.mcmc_rng_state = oss.str();
    }
    return ck;
}

// Per-group bit-identical check (the fp32 lossless contract)
void check_ck_bitwise_equal(const TrainerCheckpoint& a, const TrainerCheckpoint& b) {
    SX_CHECK(a.cloud.count == b.cloud.count);
    SX_CHECK(a.cloud.sh_degree == b.cloud.sh_degree);
    SX_CHECK(a.iter == b.iter);
    for (int g = 0; g < 6; g++) {
        const std::vector<float>*pa = ckt_group(a.cloud, g), *pb = ckt_group(b.cloud, g);
        SX_CHECK(pa->size() == pb->size());
        if (!pa->empty())
            SX_CHECK(std::memcmp(pa->data(), pb->data(), pa->size() * sizeof(float)) == 0);
        SX_CHECK(a.moments[g].exp_avg == b.moments[g].exp_avg);       // vector == is bitwise (f32,
        SX_CHECK(a.moments[g].exp_avg_sq == b.moments[g].exp_avg_sq); //  assuming no NaN)
    }
    SX_CHECK(a.mcmc_rng_state == b.mcmc_rng_state);
}

int64_t file_size(const std::string& path) {
    std::error_code ec;
    const auto sz = std::filesystem::file_size(path, ec);
    return ec ? -1 : static_cast<int64_t>(sz);
}

} // namespace

// ===== (a) fp32 round trip is bit-lossless + fingerprint verification + corruption rejected =====
SX_TEST(checkpoint_file_fp32_roundtrip_and_fingerprint) {
    const TrainerCheckpoint ck = make_synthetic_ck(7, 2, /*with_rng=*/true);
    const std::string path = sxtest::tmp_path("splatx_ck_test.sxck");
    save_checkpoint_file(ck, path, /*fp16=*/false);
    SX_CHECK(!std::filesystem::exists(path + ".tmp")); // atomic write — no temp left behind
    const TrainerCheckpoint back = load_checkpoint_file(path);

    // Every config field + iter + params/moments + the RNG state are bit-identical (the fp32
    // lossless contract)
    SX_CHECK(back.config.sh_degree == 2 && back.config.factor == 4);
    SX_CHECK(back.config.test_split == 8);
    SX_CHECK(back.config.total_iters == 30000 && back.config.mcmc_enabled == 1);
    SX_CHECK(back.config.antialiased == 1 && back.config.cap_max == 14);
    SX_CHECK(back.config.mcmc_seed == 42);
    SX_CHECK(back.config.scene_scale == 4.5f && back.config.ssim_weight == 0.2f);
    check_ck_bitwise_equal(ck, back);

    // The restored RNG stream is equivalent: the next 3 outputs match the engine at save time
    // exactly (the ISO text state contract)
    {
        std::mt19937_64 orig(42);
        for (int i = 0; i < 5; i++) (void)orig();
        std::mt19937_64 restored;
        std::istringstream iss(back.mcmc_rng_state);
        iss >> restored;
        SX_CHECK(static_cast<bool>(iss));
        for (int i = 0; i < 3; i++) SX_CHECK(orig() == restored());
    }

    // Fingerprint verification: a match passes, a mismatch in any field throws explicitly (for
    // sh_degree the §8.1 message is checked)
    const CheckpointConfig good = ck.config;
    bool threw = false;
    try { verify_checkpoint_config(ck.config, good); } catch (...) { threw = true; }
    SX_CHECK(!threw);
    auto expect_reject = [&](CheckpointConfig cur, const char* field) {
        bool caught = false;
        try {
            verify_checkpoint_config(ck.config, cur);
        } catch (const std::runtime_error& e) {
            caught = std::string(e.what()).find(field) != std::string::npos;
        }
        SX_CHECK(caught); // the field name appears in the message (the explicit-error contract)
    };
    { CheckpointConfig c = good; c.sh_degree = 3;    expect_reject(c, "sh_degree"); }
    { CheckpointConfig c = good; c.factor = 8;       expect_reject(c, "factor"); }
    { CheckpointConfig c = good; c.test_split = 0;   expect_reject(c, "test_split"); }
    { CheckpointConfig c = good; c.total_iters = 1;  expect_reject(c, "total_iters"); }
    { CheckpointConfig c = good; c.mcmc_enabled = 0; expect_reject(c, "mcmc"); }
    { CheckpointConfig c = good; c.cap_max = 99;     expect_reject(c, "cap_max"); }
    { CheckpointConfig c = good; c.mcmc_seed = 7;    expect_reject(c, "seed"); }
    { CheckpointConfig c = good; c.scene_scale = 1;  expect_reject(c, "scene_scale"); }
    { CheckpointConfig c = good; c.ssim_weight = 0;  expect_reject(c, "ssim_weight"); }
    { CheckpointConfig c = good; c.antialiased = 0;  expect_reject(c, "antialiased"); }
    // The sh_degree message carries the §8.1 basis (the authoritative wording for enforcing a
    // fixed degree)
    {
        CheckpointConfig c = good;
        c.sh_degree = 0;
        bool has_basis = false;
        try { verify_checkpoint_config(ck.config, c); }
        catch (const std::runtime_error& e) {
            has_basis = std::string(e.what()).find("8.1") != std::string::npos;
        }
        SX_CHECK(has_basis);
    }

    // Corruption is rejected: a damaged magic / a version mismatch / truncation — all throw
    auto expect_load_throw = [&](const std::string& p) {
        bool caught = false;
        try { (void)load_checkpoint_file(p); } catch (const std::exception&) { caught = true; }
        SX_CHECK(caught);
    };
    std::vector<char> bytes;
    {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        SX_CHECK(f != nullptr);
        bytes.resize(static_cast<size_t>(file_size(path)));
        SX_CHECK(std::fread(bytes.data(), 1, bytes.size(), f) == bytes.size());
        std::fclose(f);
    }
    const std::string bad = sxtest::tmp_path("splatx_ck_test_bad.sxck");
    auto write_bad = [&](const std::vector<char>& b) {
        std::FILE* f = std::fopen(bad.c_str(), "wb");
        SX_CHECK(f != nullptr);
        SX_CHECK(std::fwrite(b.data(), 1, b.size(), f) == b.size());
        std::fclose(f);
    };
    { std::vector<char> b = bytes; b[0] = 'X'; write_bad(b); expect_load_throw(bad); }
    { std::vector<char> b = bytes; b[4] = 99;  write_bad(b); expect_load_throw(bad); } // version
    { std::vector<char> b = bytes; b.resize(b.size() / 2); write_bad(b); expect_load_throw(bad); }
    { std::vector<char> b = bytes; b.push_back(0); write_bad(b); expect_load_throw(bad); } // extra
    expect_load_throw(sxtest::tmp_path("splatx_ck_missing.sxck")); // missing file
    std::remove(bad.c_str());
    std::remove(path.c_str());
}

// ===== (b) resume continuity — fp32, MCMC on (drives the RNG state and the refine index flow) =====
// Determinism due diligence (the basis for the "bitwise vs gate" judgment in the brief):
// - An fp32 checkpoint is lossless and the resident download/upload is a memcpy — so the resumed
//   state is **bit-identical** to the state of the straight-through run at the step 10 boundary.
//   The whole forward has no atomics and is therefore bit-deterministic (TrainerTests (p) ① — the
//   precedent for a memcmp gate). The n_isects/loss/render of the first resumed step are
//   consequently required to match the straight-through run bit for bit.
// - The bwd side (raster/sh/proj), by contrast, accumulates through atomicAdd in an order that
//   differs from run to run, so **even two runs of the same path are not bit-identical in the
//   parameters** (which is why the (p) ② double-run uses an rel_l2 1e-5 gate). The comparison N
//   steps after resuming therefore reuses the existing comparison gates rather than bitwise
//   equality: parameters rel_l2 ≤ 1e-4 (the (n)/(o) precedent — includes margin for 10 steps of
//   accumulation), moments ≤ 1e-3 ((o)).
// - RNG: the noise is Philox(seed, offset=iter) and so continues from iter alone, while multinomial
//   restores the mt19937_64 state — with cap 64 the add draws keep going in the resumed region too
//   (n_added > 0 on every step), so a state mismatch would split the source indices and the rel_l2
//   gate would catch it (a non-vacuity assertion is included).
SX_TEST(trainer_checkpoint_resume_continuity_fp32) {
    KernelBackend& b = sxtest::shared_metal_backend();
    TrainingCamera cam;
    for (int i = 0; i < 16; i++) cam.viewmat[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    const float K[9] = {70.0f, 0.0f, 32.0f, 0.0f, 70.0f, 32.0f, 0.0f, 0.0f, 1.0f};
    for (int i = 0; i < 9; i++) cam.K[i] = K[i];
    cam.width = 64;
    cam.height = 64;

    // The same family of initial state as TrainerTests (o) — perturbation + 2 dead slots (which
    // makes relocate non-vacuous)
    SplatCloud init;
    init.count = 16;
    init.sh_degree = 0;
    init.means.resize(16 * 3);
    init.scales.assign(16 * 3, std::log(0.13f));
    init.quats.assign(16 * 4, 0.0f);
    init.opacities.assign(16, logit_f32(0.85f));
    init.sh0.resize(16 * 3);
    init.shN.clear();
    for (int i = 0; i < 16; i++) {
        init.means[i * 3 + 0] = -0.6f + 0.4f * static_cast<float>(i % 4);
        init.means[i * 3 + 1] = -0.6f + 0.4f * static_cast<float>(i / 4);
        init.means[i * 3 + 2] = 3.0f;
        init.quats[i * 4 + 0] = 1.0f;
        for (int k = 0; k < 3; k++)
            init.sh0[i * 3 + k] = (0.2f + 0.04f * static_cast<float>((i + k) % 16)) / kShC0;
    }
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto& v : init.means) v += 0.05f * u(rng);
    for (auto& v : init.scales) v += 0.30f * u(rng);
    for (auto& v : init.quats) v += 0.05f * u(rng);
    for (auto& v : init.opacities) v += 0.70f * u(rng);
    for (auto& v : init.sh0) v += 0.60f * u(rng);
    init.opacities[2] = logit_f32(0.001f); // dead (clearly separated from min_opacity 0.005)
    init.opacities[5] = logit_f32(0.001f);

    TrainConfig cfg;
    cfg.total_iters = 100;
    cfg.lr_means = 2e-3f;
    cfg.lr_scales = 2e-2f;
    cfg.lr_quats = 5e-3f;
    cfg.lr_opacities = 5e-2f;
    cfg.lr_sh0 = 2e-2f;
    cfg.ssim_weight = 0.0f;
    cfg.opacity_reg = 0.01f;
    cfg.scale_reg = 0.01f;
    cfg.mcmc.enabled = true;
    cfg.mcmc.cap_max = 64; // add draws continue in the resumed region (10..19) — non-vacuous RNG
    cfg.mcmc.seed = 777;

    const std::vector<float> gt(static_cast<size_t>(cam.width) * cam.height * 3, 0.35f);

    // A: the straight-through run — a checkpoint at the step boundary after 0..9 (10..19 keep going)
    Trainer a(b, init, cfg);
    for (int it = 0; it < 10; it++) (void)a.step(cam, gt.data(), it);
    const TrainerCheckpoint ck = a.make_checkpoint(10);
    SX_CHECK(ck.iter == 10 && ck.cloud.count == a.splat_count());
    SX_CHECK(!ck.mcmc_rng_state.empty()); // mcmc on — state serialization is non-vacuous

    // Goes through a file round trip (exactly the production path — it hits the disk)
    const std::string path = sxtest::tmp_path("splatx_ck_resume.sxck");
    save_checkpoint_file(ck, path, /*fp16=*/false);
    const TrainerCheckpoint loaded = load_checkpoint_file(path);
    std::remove(path.c_str());
    check_ck_bitwise_equal(ck, loaded); // the fp32 round trip is lossless (premise of continuity)

    // B: resume — construct from ck.cloud + restore (the same flow as TrainC resume)
    Trainer c(b, loaded.cloud, cfg);
    c.restore_checkpoint(loaded);
    SX_CHECK(c.splat_count() == a.splat_count());
    { // Right after restore the parameters are bit-identical (snapshot = resumed state). The
      // Trainer cloud is sized to the cap arena, so only the leading count·dim portion is
      // compared (the checkpoint holds count rows — the tail is structurally 0).
        const SplatCloud& ca = c.cloud();
        for (int g = 0; g < 6; g++) {
            const std::vector<float>* pa = ckt_group(ck.cloud, g);
            const std::vector<float>* pb = ckt_group(ca, g);
            SX_CHECK(pb->size() >= pa->size());
            if (!pa->empty())
                SX_CHECK(std::memcmp(pa->data(), pb->data(), pa->size() * sizeof(float)) == 0);
        }
    }

    // 10..19: A (straight-through) and B (resumed) in parallel — the first step is bitwise, the
    // rest use the comparison gates (basis above)
    bool any_refined = false;
    int64_t total_added = 0;
    for (int it = 10; it < 20; it++) {
        const TrainStepResult ra = a.step(cam, gt.data(), it);
        const TrainStepResult rc = c.step(cam, gt.data(), it);
        SX_CHECK(ra.n_isects > 0);
        SX_CHECK(ra.n_isects == rc.n_isects); // the forward assembly matches exactly
        SX_CHECK(ra.mcmc_refined == rc.mcmc_refined);
        SX_CHECK(ra.n_relocated == rc.n_relocated); // the multinomial index flow (RNG restored)
        SX_CHECK(ra.n_added == rc.n_added);
        SX_CHECK(a.splat_count() == c.splat_count());
        any_refined = any_refined || ra.mcmc_refined;
        total_added += ra.n_added;
        if (it == 10) {
            // The forward of the first resumed step — the bit-deterministic region (before any
            // atomics): loss and render are bit-identical
            SX_CHECK(ra.loss == rc.loss);
            SX_CHECK(ra.render_colors.size() == rc.render_colors.size());
            SX_CHECK(std::memcmp(ra.render_colors.data(), rc.render_colors.data(),
                                 ra.render_colors.size() * sizeof(float)) == 0);
        }
    }
    SX_CHECK(any_refined);     // non-vacuous — refine really happens in the resumed region
    SX_CHECK(total_added > 0); // non-vacuous — the add draws really consume the RNG state

    // Comparison of parameters and moments after 20 steps — the only deviation is the atomic bwd
    // accumulation order (the basis for the gate is above).
    // Only the count rows are compared (the arena tail is structurally 0 on both sides — this keeps
    // comparison noise out).
    const SplatCloud& fa = a.cloud();
    const SplatCloud& fc = c.cloud();
    SX_CHECK(fa.count == fc.count);
    double worst = 0.0;
    for (int g = 0; g < 6; g++) {
        const std::vector<float>* pa = ckt_group(fa, g);
        const std::vector<float>* pc = ckt_group(fc, g);
        SX_CHECK(pa->size() == pc->size());
        const size_t cnt = static_cast<size_t>(fa.count) * ckt_dim(g, fa.sh_degree);
        if (cnt == 0) continue;
        worst = std::max(worst, sxtest::rel_l2(pc->data(), pa->data(), cnt, pa->data()));
    }
    std::printf("  [resume fp32] 20-step param rel_l2 worst = %.3g (gate 1e-4)\n", worst);
    SX_CHECK(worst <= 1e-4);
    Trainer::AdamState& ma = a.adam_state(Trainer::ParamGroup::Means);
    Trainer::AdamState& mc = c.adam_state(Trainer::ParamGroup::Means);
    SX_CHECK(ma.exp_avg.size() == mc.exp_avg.size());
    const double m_l2 =
        sxtest::rel_l2(mc.exp_avg.data(), ma.exp_avg.data(), ma.exp_avg.size(), ma.exp_avg.data());
    std::printf("  [resume fp32] means exp_avg rel_l2 = %.3g (gate 1e-3)\n", m_l2);
    SX_CHECK(m_l2 <= 1e-3);

    // A resume that violates the fingerprint is rejected (end to end — the restore path): a Trainer
    // differing only in sh_degree throws
    {
        SplatCloud wrong = loaded.cloud;
        wrong.sh_degree = 1;
        wrong.shN.assign(static_cast<size_t>(wrong.count) * wrong.sh_rest_coeffs() * 3, 0.0f);
        Trainer w(b, wrong, cfg);
        bool caught = false;
        try { w.restore_checkpoint(loaded); }
        catch (const std::runtime_error&) { caught = true; }
        SX_CHECK(caught);
    }
}

// ===== (c) fp16 — the round-trip quality gate + paired zeroing + the save time of a synthetic 1M
// state (compared against the 3-second contract) =====
SX_TEST(trainer_checkpoint_fp16_quality_and_save_time) {
    // --- 1) Measure the fp16 round-trip error on a real training state (a 10-step Metal smoke) ---
    KernelBackend& b = sxtest::shared_metal_backend();
    TrainingCamera cam;
    for (int i = 0; i < 16; i++) cam.viewmat[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    const float K[9] = {70.0f, 0.0f, 32.0f, 0.0f, 70.0f, 32.0f, 0.0f, 0.0f, 1.0f};
    for (int i = 0; i < 9; i++) cam.K[i] = K[i];
    cam.width = 64;
    cam.height = 64;

    SplatCloud init;
    init.count = 16;
    init.sh_degree = 0;
    init.means.resize(16 * 3);
    init.scales.assign(16 * 3, std::log(0.13f));
    init.quats.assign(16 * 4, 0.0f);
    init.opacities.assign(16, logit_f32(0.85f));
    init.sh0.resize(16 * 3);
    init.shN.clear();
    for (int i = 0; i < 16; i++) {
        init.means[i * 3 + 0] = -0.6f + 0.4f * static_cast<float>(i % 4);
        init.means[i * 3 + 1] = -0.6f + 0.4f * static_cast<float>(i / 4);
        init.means[i * 3 + 2] = 3.0f;
        init.quats[i * 4 + 0] = 1.0f;
        for (int k = 0; k < 3; k++)
            init.sh0[i * 3 + k] = (0.2f + 0.04f * static_cast<float>((i + k) % 16)) / kShC0;
    }
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto& v : init.means) v += 0.05f * u(rng);
    for (auto& v : init.scales) v += 0.30f * u(rng);
    for (auto& v : init.quats) v += 0.05f * u(rng);
    for (auto& v : init.opacities) v += 0.70f * u(rng);
    for (auto& v : init.sh0) v += 0.60f * u(rng);

    TrainConfig cfg;
    cfg.total_iters = 100;
    cfg.lr_means = 2e-3f;
    cfg.lr_scales = 2e-2f;
    cfg.lr_quats = 5e-3f;
    cfg.lr_opacities = 5e-2f;
    cfg.lr_sh0 = 2e-2f;
    cfg.ssim_weight = 0.2f; // the default combined loss (the SSIM path included)

    const std::vector<float> gt(static_cast<size_t>(cam.width) * cam.height * 3, 0.35f);
    Trainer tr(b, init, cfg);
    for (int it = 0; it < 10; it++) (void)tr.step(cam, gt.data(), it);
    const TrainerCheckpoint ck = tr.make_checkpoint(10);

    const std::string p32 = sxtest::tmp_path("splatx_ck_f32.sxck");
    const std::string p16 = sxtest::tmp_path("splatx_ck_f16.sxck");
    save_checkpoint_file(ck, p32, false);
    save_checkpoint_file(ck, p16, true);
    const int64_t s32 = file_size(p32), s16 = file_size(p16);
    std::printf("  [fp16] file size %lld B vs fp32 %lld B (ratio %.3f)\n",
                static_cast<long long>(s16), static_cast<long long>(s32),
                static_cast<double>(s16) / static_cast<double>(s32));
    SX_CHECK(s32 > 0 && s16 > 0);
    SX_CHECK(s16 < (s32 * 6) / 10); // half precision — under 60% incl. header (measured ~0.50)

    // Measured round-trip rel_l2 — in the normal range of half the relative error is bounded by
    // 2^-11 ≈ 4.88e-4, so rel_l2 (a root-mean-square) comes out below that. Measured (M4):
    // params 2.5e-4 / exp_avg 2.7e-4 / exp_avg_sq 5.2e-4 (sqrt compression — consistent with the
    // predicted 2× error, bound ~9.8e-4). The gates are 1e-3 for params/exp_avg, and 4e-3 for
    // exp_avg_sq, which leaves room for small values erased by underflow.
    const TrainerCheckpoint b16 = load_checkpoint_file(p16);
    double worst_p = 0.0, worst_m1 = 0.0, worst_m2 = 0.0;
    for (int g = 0; g < 6; g++) {
        const std::vector<float>* pa = ckt_group(ck.cloud, g);
        const std::vector<float>* pb = ckt_group(b16.cloud, g);
        SX_CHECK(pa->size() == pb->size());
        if (!pa->empty())
            worst_p = std::max(worst_p,
                               sxtest::rel_l2(pb->data(), pa->data(), pa->size(), pa->data()));
        const CheckpointMoments &ma = ck.moments[g], &mb = b16.moments[g];
        if (!ma.exp_avg.empty()) {
            worst_m1 = std::max(worst_m1, sxtest::rel_l2(mb.exp_avg.data(), ma.exp_avg.data(),
                                                         ma.exp_avg.size(), ma.exp_avg.data()));
            worst_m2 = std::max(worst_m2,
                                sxtest::rel_l2(mb.exp_avg_sq.data(), ma.exp_avg_sq.data(),
                                               ma.exp_avg_sq.size(), ma.exp_avg_sq.data()));
        }
    }
    std::printf("  [fp16] roundtrip rel_l2: params %.3g, exp_avg %.3g, exp_avg_sq %.3g\n",
                worst_p, worst_m1, worst_m2);
    SX_CHECK(worst_p <= 1e-3);
    SX_CHECK(worst_m1 <= 1e-3);
    SX_CHECK(worst_m2 <= 4e-3);

    // --- 2) The paired-zeroing rule (basis in Checkpoint.h — blocks the m/eps blow-up) ---
    // v = 1e-16: sqrt = 1e-8 < half the smallest half subnormal (2.98e-8) → underflows to 0 → m is
    // forced to 0 as well.
    // v = 1e-9: sqrt = 3.16e-5 is representable → m and v are preserved approximately (the m/√v
    // step size stays continuous).
    {
        TrainerCheckpoint tiny = make_synthetic_ck(2, 0, false);
        tiny.moments[0].exp_avg = {1.0f, 1e-3f, 5e-5f, 0.0f, 0.25f, -0.5f};
        tiny.moments[0].exp_avg_sq = {1e-16f, 1e-4f, 1e-9f, 0.0f, 0.01f, 0.09f};
        const std::string pt = sxtest::tmp_path("splatx_ck_pair.sxck");
        save_checkpoint_file(tiny, pt, true);
        const TrainerCheckpoint tb = load_checkpoint_file(pt);
        std::remove(pt.c_str());
        SX_CHECK(tb.moments[0].exp_avg_sq[0] == 0.0f); // underflow
        SX_CHECK(tb.moments[0].exp_avg[0] == 0.0f);    // paired zeroing (m was 1.0, but v=0 → 0)
        SX_CHECK_NEAR(tb.moments[0].exp_avg_sq[2], 1e-9, 2e-3, 0); // sqrt compression keeps this
        SX_CHECK_NEAR(tb.moments[0].exp_avg[2], 5e-5, 2e-3, 0);
        SX_CHECK_NEAR(tb.moments[0].exp_avg_sq[5], 0.09, 1e-3, 0);
    }

    // --- 3) The fp16 resume quality gate: 10 steps in parallel with an fp32 resume — the loss does
    // not diverge and the two stay close ---
    // The fp16 perturbation (~5e-4 rel) can amplify over 10 steps, so the gate is on the **loss
    // trajectory** rather than on a parameter comparison (a quality contract): every step is finite
    // and the final loss is within |Δ| ≤ 2e-3 + 15%·loss32 of the fp32 resume. Measured
    // |Δ| = 1.65e-3 (the fp16 side is in fact lower — the trajectories merely branch while both
    // converge; loss32 ≈ 1.45e-2, first measured on M4) — the gate carries roughly a 2.5× margin
    // over the measurement (absorbing the run-to-run deviation of the atomic bwd).
    {
        const TrainerCheckpoint l32 = load_checkpoint_file(p32);
        Trainer t32(b, l32.cloud, cfg);
        t32.restore_checkpoint(l32);
        Trainer t16(b, b16.cloud, cfg);
        t16.restore_checkpoint(b16);
        float loss32 = 0.0f, loss16 = 0.0f;
        for (int it = 10; it < 20; it++) {
            loss32 = t32.step(cam, gt.data(), it).loss;
            loss16 = t16.step(cam, gt.data(), it).loss;
            SX_CHECK(std::isfinite(loss32) && std::isfinite(loss16));
        }
        std::printf("  [fp16] resumed 10-step final loss: fp32 %.6f vs fp16 %.6f (|d| %.2e)\n",
                    static_cast<double>(loss32), static_cast<double>(loss16),
                    std::fabs(static_cast<double>(loss32) - loss16));
        SX_CHECK(std::fabs(static_cast<double>(loss32) - loss16) <=
                 2e-3 + 0.15 * std::fabs(static_cast<double>(loss32)));
    }
    std::remove(p32.c_str());
    std::remove(p16.c_str());

    // --- 4) fp16 save time of a synthetic large state — recorded against the §7.5 "within 3
    // seconds" contract ---
    // Corresponds to garden factor 4 with cap 1M: sh_degree 3, an mcmc arena of 1M (the brief
    // permits a synthetic state here).
    // Measured (M4): snapshot 580 ms + save (fsync included) 1560 ms = **2140 ms**, file 337.6 MB —
    // inside the 3-second contract (worst-case size, since no moment is 0). The iOS runner shrinks
    // this to 200k in consideration of the memory ceiling (of the runner app) — measuring 500k on a
    // real device is Task 6's responsibility. The gate is 10 seconds and catches only pathological
    // regressions (margin for device-to-device variation); the measured value is recorded through
    // printf.
#if TARGET_OS_IPHONE
    const int64_t big_n = 200'000;
#else
    const int64_t big_n = 1'000'000;
#endif
    {
        const std::unique_ptr<KernelBackend> cpu = make_cpu_backend(); // times serialization only
        SplatCloud big;
        big.count = big_n;
        big.sh_degree = 3;
        auto fill = [](std::vector<float>& v, size_t n, uint32_t salt) {
            v.resize(n);
            for (size_t i = 0; i < n; i++) { // a cheap deterministic fill (values do not affect time)
                const uint32_t h = (static_cast<uint32_t>(i) * 2654435761u) ^ salt;
                v[i] = static_cast<float>(h & 0xFFFF) * (1.0f / 65536.0f) - 0.5f;
            }
        };
        const size_t n = static_cast<size_t>(big_n);
        fill(big.means, n * 3, 1);
        fill(big.scales, n * 3, 2);
        fill(big.quats, n * 4, 3);
        fill(big.opacities, n, 4);
        fill(big.sh0, n * 3, 5);
        fill(big.shN, n * 45, 6);
        TrainConfig bcfg;
        bcfg.total_iters = 30000;
        bcfg.mcmc.enabled = true;
        bcfg.mcmc.cap_max = big_n;
        Trainer btr(*cpu, std::move(big), bcfg);
        constexpr Trainer::ParamGroup groups[6] = {
            Trainer::ParamGroup::Means, Trainer::ParamGroup::Quats,
            Trainer::ParamGroup::Scales, Trainer::ParamGroup::Opacities,
            Trainer::ParamGroup::Sh0, Trainer::ParamGroup::ShN};
        for (int g = 0; g < 6; g++) { // fill the moments too (no zero compression — worst-case size)
            Trainer::AdamState& st = btr.adam_state(groups[g]);
            fill(st.exp_avg, st.exp_avg.size(), 100 + static_cast<uint32_t>(g));
            for (size_t i = 0; i < st.exp_avg_sq.size(); i++)
                st.exp_avg_sq[i] = 1e-8f + 1e-6f * static_cast<float>(i % 97);
        }
        const std::string pbig = sxtest::tmp_path("splatx_ck_big.sxck");
        const auto t0 = std::chrono::steady_clock::now();
        const TrainerCheckpoint bck = btr.make_checkpoint(7000, 4);
        const auto t1 = std::chrono::steady_clock::now();
        save_checkpoint_file(bck, pbig, true);
        const auto t2 = std::chrono::steady_clock::now();
        const double make_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double save_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        const int64_t sz = file_size(pbig);
        std::remove(pbig.c_str());
        std::printf("  [fp16 1M] N=%lld snapshot %.0f ms + save %.0f ms = %.0f ms, "
                    "file %.1f MB (vs the §7.5 contract's 3000 ms)\n",
                    static_cast<long long>(big_n), make_ms, save_ms, make_ms + save_ms,
                    static_cast<double>(sz) / (1024.0 * 1024.0));
        SX_CHECK(sz > 0);
        SX_CHECK(make_ms + save_ms < 10000.0); // pathological regression guard (measured in the log)
    }
}

// ===== (d) Header-only load (Phase 6 round 2 Task 5 — reference path of sx_checkpoint_info) =====
// load_checkpoint_header reads only the fingerprint, iter, and alive_count (the parameter tensors
// are not loaded) — it is what the Swift resume derives its resume arguments from. It runs through
// the same prefix parser as the full load (read_header is the single source of truth), so only
// value consistency is compared here, plus the rejection of a truncated file.
SX_TEST(checkpoint_header_load) {
    const TrainerCheckpoint ck = make_synthetic_ck(5, 1, /*with_rng=*/true);
    const std::string path = sxtest::tmp_path("splatx_ck_header_test.sxck");
    for (const bool fp16 : {false, true}) {
        save_checkpoint_file(ck, path, fp16);
        const CheckpointHeader h = load_checkpoint_header(path);
        SX_CHECK(h.fp16 == fp16);
        SX_CHECK(h.iter == ck.iter);
        SX_CHECK(h.alive_count == ck.cloud.count);
        SX_CHECK(h.config.sh_degree == ck.config.sh_degree);
        SX_CHECK(h.config.factor == ck.config.factor);
        SX_CHECK(h.config.test_split == ck.config.test_split);
        SX_CHECK(h.config.total_iters == ck.config.total_iters);
        SX_CHECK(h.config.mcmc_enabled == ck.config.mcmc_enabled);
        SX_CHECK(h.config.antialiased == ck.config.antialiased);
        SX_CHECK(h.config.cap_max == ck.config.cap_max);
        SX_CHECK(h.config.mcmc_seed == ck.config.mcmc_seed);
        SX_CHECK(h.config.scene_scale == ck.config.scene_scale);
        SX_CHECK(h.config.ssim_weight == ck.config.ssim_weight);
    }
    // A truncated file (cut in the middle of the header) is rejected — only 8 bytes after the magic
    {
        std::vector<char> whole;
        {
            std::FILE* f = std::fopen(path.c_str(), "rb");
            SX_CHECK(f != nullptr);
            char buf[64];
            const size_t got = std::fread(buf, 1, sizeof(buf), f);
            std::fclose(f);
            SX_CHECK(got >= 16);
            whole.assign(buf, buf + 12); // only through magic(4) + version(4) + flags(4)
        }
        const std::string trunc = sxtest::tmp_path("splatx_ck_header_trunc.sxck");
        std::FILE* f = std::fopen(trunc.c_str(), "wb");
        SX_CHECK(f != nullptr);
        std::fwrite(whole.data(), 1, whole.size(), f);
        std::fclose(f);
        bool threw = false;
        try { (void)load_checkpoint_header(trunc); } catch (const std::exception&) { threw = true; }
        SX_CHECK(threw);
        std::filesystem::remove(trunc);
    }
    std::filesystem::remove(path);
}

// ===== (e) Reserved byte rejection (the 2026-07-17 big cleanup) — the old bilateral_grid slot ====
// The v1 offset contract (invariant): magic(4)+version(4)+flags(4)+int32×4(16)+uint8×2(2) →
// offset 30.
// A file with it off (= reserved 0) round-trips with no regression; a file patched to a nonzero
// value is rejected explicitly (a silent, wrong restore MUST NOT happen).
SX_TEST(checkpoint_reserved_byte_zero_roundtrip_and_reject) {
    const TrainerCheckpoint ck = make_synthetic_ck(5, 1, /*with_rng=*/false);
    const std::string path = sxtest::tmp_path("splatx_ck_reserved.sxck");
    save_checkpoint_file(ck, path, /*fp16=*/false);
    { // the reserved-0 path round-trips with no regression + offset 30 in the file really is 0
        const TrainerCheckpoint back = load_checkpoint_file(path);
        check_ck_bitwise_equal(ck, back);
        std::FILE* f = std::fopen(path.c_str(), "rb");
        SX_CHECK(f != nullptr);
        SX_CHECK(std::fseek(f, 30, SEEK_SET) == 0);
        SX_CHECK(std::fgetc(f) == 0);
        std::fclose(f);
    }
    { // patch offset 30 to 1 → an explicit "no longer supported" rejection
        std::FILE* f = std::fopen(path.c_str(), "r+b");
        SX_CHECK(f != nullptr);
        SX_CHECK(std::fseek(f, 30, SEEK_SET) == 0);
        SX_CHECK(std::fputc(1, f) == 1);
        std::fclose(f);
        bool caught = false;
        try { (void)load_checkpoint_file(path); }
        catch (const std::runtime_error& e) {
            caught = std::string(e.what()).find("no longer supported") != std::string::npos;
        }
        SX_CHECK(caught);
        bool caught_hdr = false; // the header-only loader shares the parser — same rejection
        try { (void)load_checkpoint_header(path); }
        catch (const std::runtime_error&) { caught_hdr = true; }
        SX_CHECK(caught_hdr);
    }
    std::remove(path.c_str());
}
