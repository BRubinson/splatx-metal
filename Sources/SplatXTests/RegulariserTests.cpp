// Shape regulariser tests (flatten on s3, the needle hinge on s1/s2, train_scale_shape_reg against
// the host twin, the checkpoint fingerprint). Run alone with sx_tests_run_prefix("regulariser_") —
// the Swift RegionMaskTests entry.
#include "TestHarness.h"
#include "TestUtils.h"
#include "splatx/Checkpoint.h"
#include "splatx/KernelBackend.h"
#include "splatx/Trainer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace splatx;

namespace {

TrainingCamera make_camera(int w, int h, float f) {
    TrainingCamera cam;
    for (int i = 0; i < 16; i++) cam.viewmat[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    const float K[9] = {f, 0.0f, 0.5f * w, 0.0f, f, 0.5f * h, 0.0f, 0.0f, 1.0f};
    for (int i = 0; i < 9; i++) cam.K[i] = K[i];
    cam.width = w;
    cam.height = h;
    return cam;
}

// The shape loss the regulariser differentiates, over log scales [N,3].
double shape_loss(const std::vector<double>& log_s, double fw, double nw, double ratio) {
    double l = 0.0;
    for (size_t g = 0; g < log_s.size() / 3; g++) {
        double s[3] = {std::exp(log_s[g * 3]), std::exp(log_s[g * 3 + 1]), std::exp(log_s[g * 3 + 2])};
        std::sort(s, s + 3, [](double a, double b) { return a > b; });
        l += fw * s[2] + nw * std::max(0.0, s[0] / s[1] - ratio);
    }
    return l;
}

// The analytic gradient (add_shape_reg_grad_inplace) against central differences of shape_loss.
double worst_fd_error(const std::vector<float>& log_s, float fw, float nw, float ratio,
                      std::vector<float>& analytic) {
    const std::vector<float> lin = activate_scales(log_s.data(), static_cast<int64_t>(log_s.size()));
    analytic.assign(log_s.size(), 0.0f);
    add_shape_reg_grad_inplace(analytic, lin.data(), fw, nw, ratio);
    double worst = 0.0;
    for (size_t i = 0; i < log_s.size(); i++) {
        std::vector<double> p(log_s.begin(), log_s.end()), m = p;
        const double h = 1e-4;
        p[i] += h;
        m[i] -= h;
        const double fd = (shape_loss(p, fw, nw, ratio) - shape_loss(m, fw, nw, ratio)) / (2 * h);
        worst = std::max(worst, std::fabs(fd - analytic[i]) /
                                    std::max({std::fabs(fd), std::fabs(1.0 * analytic[i]), 1e-9}));
    }
    return worst;
}

SplatCloud make_cloud(float needle_stretch) {
    SplatCloud c;
    c.count = 16;
    c.sh_degree = 0;
    c.means.resize(48);
    c.scales.assign(48, std::log(0.13f));
    c.quats.assign(64, 0.0f);
    c.opacities.assign(16, logit_f32(0.7f));
    c.sh0.assign(48, 0.1f);
    for (int i = 0; i < 16; i++) {
        c.means[i * 3 + 0] = -0.6f + 0.4f * static_cast<float>(i % 4);
        c.means[i * 3 + 1] = -0.6f + 0.4f * static_cast<float>(i / 4);
        c.means[i * 3 + 2] = 2.6f + 0.05f * static_cast<float>(i % 5);
        c.quats[i * 4] = 1.0f;
        c.scales[i * 3 + i % 3] += needle_stretch; // one long axis per Gaussian
    }
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto& v : c.scales) v += 0.30f * u(rng);
    for (auto& v : c.quats) v += 0.05f * u(rng);
    for (auto& v : c.opacities) v += 0.70f * u(rng);
    for (auto& v : c.sh0) v += 0.60f * u(rng);
    return c;
}

double cloud_rel_l2(SplatCloud& a, SplatCloud& b) {
    const std::pair<const std::vector<float>*, const std::vector<float>*> groups[] = {
        {&a.means, &b.means}, {&a.quats, &b.quats}, {&a.scales, &b.scales},
        {&a.opacities, &b.opacities}, {&a.sh0, &b.sh0}};
    double worst = 0.0;
    for (const auto& [pa, pb] : groups)
        worst = std::max(worst, sxtest::rel_l2(pb->data(), pa->data(), pa->size(), pa->data()));
    return worst;
}

} // namespace

// Flatten: d/dlog s3 = w·s3 on the smallest axis only, matching central finite differences.
SX_TEST(regulariser_flatten_finite_difference) {
    std::mt19937 rng(9);
    std::uniform_real_distribution<float> u(-3.0f, -0.5f);
    std::vector<float> log_s(8 * 3);
    for (auto& v : log_s) v = u(rng);
    std::vector<float> g;
    const double worst = worst_fd_error(log_s, 0.25f, 0.0f, 5.0f, g);
    bool only_min = true;
    for (size_t k = 0; k < 8; k++) {
        const float* s = &log_s[k * 3];
        const size_t imin = static_cast<size_t>(std::min_element(s, s + 3) - s);
        for (size_t a = 0; a < 3; a++) only_min &= (a == imin) == (g[k * 3 + a] != 0.0f);
    }
    std::printf("  [flatten] 8 Gaussians: worst rel error vs finite differences %.2e, gradient on "
                "s3 only %s\n", worst, only_min ? "yes" : "no");
    SX_CHECK(worst <= 1e-5);
    SX_CHECK(only_min);
}

// Needle: zero below the ratio, finite differences above it, and a 10:10:1 disc pays nothing.
SX_TEST(regulariser_needle_hinge) {
    // s1/s2 = 3 (below 5), 8 (above), and a 10:10:1 disc.
    const std::vector<float> log_s = {std::log(0.3f),  std::log(0.1f),  std::log(0.05f),
                                      std::log(0.02f), std::log(0.16f), std::log(0.01f),
                                      std::log(0.1f),  std::log(0.1f),  std::log(0.01f)};
    std::vector<float> g;
    const double worst = worst_fd_error(log_s, 0.0f, 0.5f, 5.0f, g);
    SX_CHECK(worst <= 1e-5);
    SX_CHECK(g[0] == 0.0f && g[1] == 0.0f && g[2] == 0.0f); // below the ratio
    SX_CHECK(g[4] > 0.0f && g[3] < 0.0f && g[5] == 0.0f);  // s1 = axis 1 pulled in, s2 = axis 0 out
    SX_CHECK_NEAR(g[4], 0.5 * 8.0, 1e-5, 0);
    SX_CHECK(g[6] == 0.0f && g[7] == 0.0f && g[8] == 0.0f); // the disc
    std::vector<float> gf;
    (void)worst_fd_error(log_s, 0.25f, 0.5f, 5.0f, gf);
    SX_CHECK(gf[6] == 0.0f && gf[7] == 0.0f && gf[8] > 0.0f); // flatten only touches its thin axis
    std::printf("  [needle] worst rel error vs finite differences %.2e; r=3 zero, r=8 d/dlog s1 "
                "%.4f, disc 10:10:1 zero\n", worst, static_cast<double>(g[4]));
}

// train_scale_shape_reg (resident) reproduces add_shape_reg_grad_inplace (host): three steps on
// both paths agree, and the regulariser visibly changes the scales against a weights-0 run.
SX_TEST(regulariser_resident_matches_host) {
    KernelBackend& b = sxtest::shared_metal_backend();
    const TrainingCamera cam = make_camera(64, 64, 70.0f);
    const SplatCloud c = make_cloud(1.2f);
    TrainConfig cfg;
    cfg.total_iters = 100;
    cfg.ssim_weight = 0.2f;
    cfg.flatten_weight = 10.0f;
    cfg.needle_weight = 1.0f;
    cfg.needle_ratio = 2.0f;
    TrainConfig cfg_host = cfg;
    cfg_host.resident = false;
    TrainConfig cfg_off = cfg;
    cfg_off.flatten_weight = 0.0f;
    cfg_off.needle_weight = 0.0f;
    Trainer th(b, c, cfg_host), tr(b, c, cfg), toff(b, c, cfg_off);
    const std::vector<float> gt(static_cast<size_t>(cam.width) * cam.height * 3, 0.35f);
    for (int it = 0; it < 3; it++) {
        (void)th.step(cam, gt.data(), it);
        (void)tr.step(cam, gt.data(), it);
        (void)toff.step(cam, gt.data(), it);
    }
    SplatCloud ch = th.cloud(), cr = tr.cloud(), coff = toff.cloud();
    const double worst = cloud_rel_l2(ch, cr);
    const double effect = sxtest::rel_l2(cr.scales.data(), coff.scales.data(), cr.scales.size(),
                                         coff.scales.data());
    std::printf("  [shape reg resident vs host] param rel_l2 worst %.3g (gate 1e-4); scales vs "
                "weights-0 run %.3g\n", worst, effect);
    SX_CHECK(worst <= 1e-4);
    SX_CHECK(effect > 1e-3);
}

// Weights 0 are a no-op, and the checkpoint fingerprint carries the three weights and the mask bit
// (version 2), rejecting a resume that changes any of them.
SX_TEST(regulariser_zero_noop_and_checkpoint_fingerprint) {
    std::vector<float> v = {0.1f, -0.2f, 0.3f, 0.4f, 0.5f, -0.6f};
    const std::vector<float> before = v;
    const std::vector<float> lin = {0.5f, 0.01f, 0.02f, 0.3f, 0.3f, 0.001f};
    add_shape_reg_grad_inplace(v, lin.data(), 0.0f, 0.0f, 5.0f);
    SX_CHECK(std::memcmp(v.data(), before.data(), v.size() * sizeof(float)) == 0);
    const TrainConfig defaults;
    SX_CHECK(defaults.flatten_weight == 0.0f && defaults.needle_weight == 0.0f &&
             defaults.needle_ratio == 5.0f && !defaults.has_mask);

    KernelBackend& b = sxtest::shared_metal_backend();
    TrainConfig cfg;
    cfg.flatten_weight = 10.0f;
    cfg.needle_weight = 0.01f;
    cfg.needle_ratio = 5.0f;
    cfg.has_mask = true;
    Trainer tr(b, make_cloud(0.0f), cfg);
    const std::string path = sxtest::tmp_path("splatx_ck_shape_reg.sxck");
    save_checkpoint_file(tr.make_checkpoint(0, 1, 0), path, false);
    const CheckpointHeader h = load_checkpoint_header(path);
    SX_CHECK(h.version == 2);
    SX_CHECK(h.config.flatten_weight == 10.0f && h.config.needle_weight == 0.01f &&
             h.config.needle_ratio == 5.0f && h.config.has_mask == 1);
    const TrainerCheckpoint ck = load_checkpoint_file(path);
    Trainer same(b, ck.cloud, cfg);
    same.restore_checkpoint(ck, 1, 0); // identical fingerprint restores
    const auto rejects = [&](TrainConfig other) {
        Trainer t(b, ck.cloud, other);
        try { t.restore_checkpoint(ck, 1, 0); }
        catch (const std::runtime_error&) { return true; }
        return false;
    };
    TrainConfig k = cfg;
    k.flatten_weight = 5.0f;
    SX_CHECK(rejects(k));
    k = cfg;
    k.needle_weight = 0.0f;
    SX_CHECK(rejects(k));
    k = cfg;
    k.needle_ratio = 4.0f;
    SX_CHECK(rejects(k));
    k = cfg;
    k.has_mask = false;
    SX_CHECK(rejects(k));
    std::remove(path.c_str());
}
