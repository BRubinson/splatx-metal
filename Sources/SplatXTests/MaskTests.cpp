// Region mask tests (the masks/<stem>.png sidecar, the masked L1/SSIM/depth terms and the masked
// held-out metrics). Run alone with sx_tests_run_prefix("mask_") — the Swift RegionMaskTests entry.
#include "TestHarness.h"
#include "TestUtils.h"
#include "KernelConstants.h"
#include "splatx/CpuKernels.h"
#include "splatx/DepthSupervision.h"
#include "splatx/ImageIO.h"
#include "splatx/KernelBackend.h"
#include "splatx/RegionMask.h"
#include "splatx/Trainer.h"
#include "splatx/TrainingInput.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
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

// A mask whose sidecar columns x < zero_cols are 0 and the rest 1.
RegionMask make_mask(int w, int h, int zero_cols) {
    RegionMask m;
    m.width = w;
    m.height = h;
    m.m.assign(static_cast<size_t>(w) * h, 1.0f);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < std::min(zero_cols, w); x++) m.m[static_cast<size_t>(y) * w + x] = 0.0f;
    return m;
}

// The 4×4 anisotropic, rotated grid the depth tests train on (an isotropic cloud has a zero
// quaternion gradient that Adam turns into lr-sized steps of atomic-order noise).
SplatCloud make_grid_cloud() {
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
    }
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto& v : c.scales) v += 0.30f * u(rng);
    for (auto& v : c.quats) v += 0.05f * u(rng);
    for (auto& v : c.opacities) v += 0.70f * u(rng);
    for (auto& v : c.sh0) v += 0.60f * u(rng);
    return c;
}

DepthTarget make_depth(int w, int h, float depth_m) {
    DepthTarget t;
    t.width = w;
    t.height = h;
    t.depth_m.assign(static_cast<size_t>(w) * h, depth_m);
    for (size_t i = 0; i < t.depth_m.size(); i += 5) t.depth_m[i] = 0.0f;
    return t;
}

bool bitwise_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

bool all_finite(const std::vector<float>& v) {
    return std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
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

bool cloud_bitwise_equal(const SplatCloud& a, const SplatCloud& b) {
    return bitwise_equal(a.means, b.means) && bitwise_equal(a.quats, b.quats) &&
           bitwise_equal(a.scales, b.scales) && bitwise_equal(a.opacities, b.opacities) &&
           bitwise_equal(a.sh0, b.sh0);
}

// A render/gt pair with |render − gt| ≥ 0.05 everywhere (no L1 kink within a finite-difference step).
void make_images(int w, int h, std::vector<float>& render, std::vector<float>& gt) {
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> u(0.2f, 0.8f);
    render.resize(static_cast<size_t>(w) * h * 3);
    gt.resize(render.size());
    for (size_t i = 0; i < render.size(); i++) {
        gt[i] = u(rng);
        float r = u(rng);
        if (std::fabs(r - gt[i]) < 0.05f) r = gt[i] + (r >= gt[i] ? 0.05f : -0.05f);
        render[i] = r;
    }
}

// The masked L1+SSIM total and its gradient over the render (the Trainer host formula).
struct MaskedLoss {
    double loss = 0.0;
    std::vector<float> grad;
};

MaskedLoss masked_l1_ssim(KernelBackend& b, const std::vector<float>& render,
                          const std::vector<float>& gt, int w, int h, const RegionMask& mask,
                          float lam, bool want_grad) {
    const int64_t M = mask_view_count(mask, w, h);
    const int64_t Ms = mask_ssim_count(mask, w, h);
    double l1 = 0.0;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            if (!(mask.at(x, y, w, h) > 0.0f)) continue;
            for (int c = 0; c < 3; c++) {
                const size_t i = (static_cast<size_t>(y) * w + x) * 3 + c;
                l1 += std::fabs(static_cast<double>(render[i]) - gt[i]);
            }
        }
    l1 /= static_cast<double>(3 * M);
    SsimInputs si;
    si.img1 = render.data();
    si.img2 = gt.data();
    si.I = 1;
    si.H = h;
    si.W = w;
    si.C = 3;
    si.need_coeffs = true;
    SsimFwdOutputs so = b.ssim_fwd(si);
    const double s = masked_ssim_mean(so.ssim_map.data(), mask, w, h, 3, Ms);
    MaskedLoss out;
    out.loss = (1.0 - lam) * l1 + lam * (1.0 - s);
    if (!want_grad) return out;
    L1LossResult lg = l1_loss_and_grad_masked(render.data(), gt.data(), w, h, mask, M);
    mask_ssim_coeffs_inplace(so.coeff_mu.data(), so.coeff_sigma.data(), so.coeff_cross.data(),
                             mask, w, h, 3);
    SsimGradsIn sgi;
    sgi.coeff_mu = so.coeff_mu.data();
    sgi.coeff_sigma = so.coeff_sigma.data();
    sgi.coeff_cross = so.coeff_cross.data();
    sgi.v_loss = static_cast<float>(static_cast<double>(lam) *
                                    static_cast<double>(so.ssim_map.size()) /
                                    (3.0 * static_cast<double>(Ms)));
    const std::vector<float> v_dssim = b.ssim_bwd(si, sgi);
    out.grad.resize(render.size());
    for (size_t i = 0; i < render.size(); i++)
        out.grad[i] = (1.0f - lam) * lg.v_render_colors[i] + v_dssim[i];
    return out;
}

} // namespace

// An all-ones mask reproduces the unmasked L1 and SSIM loss and gradients bit for bit.
SX_TEST(mask_all_ones_matches_unmasked_l1_ssim_bitwise) {
    KernelBackend& b = sxtest::shared_metal_backend();
    const int w = 40, h = 30;
    std::vector<float> render, gt;
    make_images(w, h, render, gt);
    const RegionMask ones = make_mask(10, 8, 0);
    const int64_t M = mask_view_count(ones, w, h), Ms = mask_ssim_count(ones, w, h);
    SX_CHECK(M == w * h);
    SX_CHECK(Ms == (w - 10) * (h - 10));

    const L1LossResult a = l1_loss_and_grad(render.data(), gt.data(), w * h * 3);
    const L1LossResult m = l1_loss_and_grad_masked(render.data(), gt.data(), w, h, ones, M);
    SX_CHECK(a.loss == m.loss);
    SX_CHECK(bitwise_equal(a.v_render_colors, m.v_render_colors));

    const float lam = 0.2f;
    SsimInputs si;
    si.img1 = render.data();
    si.img2 = gt.data();
    si.I = 1;
    si.H = h;
    si.W = w;
    si.C = 3;
    si.need_coeffs = true;
    SsimFwdOutputs so = b.ssim_fwd(si);
    SsimGradsIn g;
    g.coeff_mu = so.coeff_mu.data();
    g.coeff_sigma = so.coeff_sigma.data();
    g.coeff_cross = so.coeff_cross.data();
    g.v_loss = lam;
    const std::vector<float> v_plain = b.ssim_bwd(si, g);
    SX_CHECK(masked_ssim_mean(so.ssim_map.data(), ones, w, h, 3, Ms) == so.ssim);
    mask_ssim_coeffs_inplace(so.coeff_mu.data(), so.coeff_sigma.data(), so.coeff_cross.data(),
                             ones, w, h, 3);
    g.v_loss = static_cast<float>(static_cast<double>(lam) *
                                  static_cast<double>(so.ssim_map.size()) /
                                  (3.0 * static_cast<double>(Ms)));
    SX_CHECK(g.v_loss == lam);
    const std::vector<float> v_masked = b.ssim_bwd(si, g);
    SX_CHECK(bitwise_equal(v_plain, v_masked));
    std::printf("  [all-ones] %dx%d: L1 loss %.7f, %zu L1 and %zu SSIM gradient values bit-equal\n",
                w, h, static_cast<double>(m.loss), m.v_render_colors.size(), v_masked.size());
}

// Ten resident steps with an all-ones mask (L1 + SSIM + depth) against the unmasked steps: the
// first step's loss and render are bit-equal, and the parameters stay within the unmasked run's
// own run-to-run spread (raster bwd accumulates atomically, so no two resident runs are
// guaranteed bit-identical).
SX_TEST(mask_all_ones_resident_steps_match_unmasked) {
    KernelBackend& b = sxtest::shared_metal_backend();
    const TrainingCamera cam = make_camera(64, 64, 70.0f);
    const SplatCloud c = make_grid_cloud();
    TrainConfig cfg;
    cfg.total_iters = 100;
    cfg.ssim_weight = 0.2f;
    cfg.depth_weight = 0.5f;
    cfg.lr_means = 2e-3f;
    Trainer plain_a(b, c, cfg), plain_b(b, c, cfg), masked(b, c, cfg);
    DepthTarget depth = make_depth(16, 16, 3.0f);
    depth.view_valid = depth_valid_count(depth, cam.width, cam.height);
    const RegionMask ones = make_mask(16, 16, 0);
    const std::vector<float> gt(static_cast<size_t>(cam.width) * cam.height * 3, 0.35f);
    bool first_equal = false;
    for (int it = 0; it < 10; it++) {
        const TrainStepResult ra = plain_a.step(cam, gt.data(), it, &depth);
        (void)plain_b.step(cam, gt.data(), it, &depth);
        const TrainStepResult rm = masked.step(cam, gt.data(), it, &depth, &ones);
        if (it == 0)
            first_equal = ra.loss == rm.loss && ra.depth_loss == rm.depth_loss &&
                          bitwise_equal(ra.render_colors, rm.render_colors);
    }
    SX_CHECK(first_equal);
    SplatCloud ca = plain_a.cloud(), cb = plain_b.cloud(), cm = masked.cloud();
    const double self = cloud_rel_l2(ca, cb);
    const double diff = cloud_rel_l2(ca, cm);
    const bool bit_ab = cloud_bitwise_equal(ca, cb), bit_am = cloud_bitwise_equal(ca, cm);
    std::printf("  [all-ones resident] 10 steps: unmasked self-spread rel_l2 %.3g (bit-equal %s), "
                "masked vs unmasked %.3g (bit-equal %s)\n",
                self, bit_ab ? "yes" : "no", diff, bit_am ? "yes" : "no");
    if (bit_ab) SX_CHECK(bit_am); // a deterministic run must be reproduced exactly
    SX_CHECK(diff <= std::max(10.0 * self, 1e-6));
}

// A left-half-zero mask: the L1 gradient is exactly 0 on the masked-out half, the L1+SSIM gradient
// is exactly 0 beyond the SSIM half-window (5 px) from the boundary, and a resident training step
// leaves a Gaussian drawn only there bit-for-bit unchanged while one in the region moves.
SX_TEST(mask_left_half_zero_gradient) {
    KernelBackend& b = sxtest::shared_metal_backend();
    const int w = 32, h = 24;
    std::vector<float> render, gt;
    make_images(w, h, render, gt);
    const RegionMask left = make_mask(8, 6, 4); // view columns x < 16 are out
    const int64_t M = mask_view_count(left, w, h);
    SX_CHECK(M == (w / 2) * h);
    const L1LossResult l1 = l1_loss_and_grad_masked(render.data(), gt.data(), w, h, left, M);
    const MaskedLoss ml = masked_l1_ssim(b, render, gt, w, h, left, 0.2f, true);
    bool l1_left_zero = true, total_left_zero = true, right_nonzero = false;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            for (int c = 0; c < 3; c++) {
                const size_t i = (static_cast<size_t>(y) * w + x) * 3 + c;
                if (x < w / 2 && l1.v_render_colors[i] != 0.0f) l1_left_zero = false;
                if (x < w / 2 - 5 && ml.grad[i] != 0.0f) total_left_zero = false;
                if (x >= w / 2 && ml.grad[i] != 0.0f) right_nonzero = true;
            }
    SX_CHECK(l1_left_zero);
    SX_CHECK(total_left_zero);
    SX_CHECK(right_nonzero);

    // End to end on the resident path: two small Gaussians, one at u ≈ 10 (out), one at u ≈ 50 (in).
    const TrainingCamera cam = make_camera(64, 64, 70.0f);
    SplatCloud c;
    c.count = 2;
    c.sh_degree = 0;
    c.means = {(10.0f - 32.0f) * 2.6f / 70.0f, 0.0f, 2.6f, (50.0f - 32.0f) * 2.6f / 70.0f, 0.0f, 2.6f};
    c.scales = {std::log(0.03f), std::log(0.02f), std::log(0.025f),
                std::log(0.03f), std::log(0.02f), std::log(0.025f)};
    c.quats = {0.98f, 0.1f, 0.05f, 0.02f, 0.98f, 0.1f, 0.05f, 0.02f};
    c.opacities = {logit_f32(0.6f), logit_f32(0.6f)};
    c.sh0 = {0.2f, -0.1f, 0.3f, 0.2f, -0.1f, 0.3f};
    TrainConfig cfg;
    cfg.total_iters = 100;
    cfg.ssim_weight = 0.2f;
    Trainer tr(b, c, cfg);
    const RegionMask half = make_mask(16, 16, 8);
    const std::vector<float> gt_img(static_cast<size_t>(64) * 64 * 3, 0.8f);
    for (int it = 0; it < 5; it++) (void)tr.step(cam, gt_img.data(), it, nullptr, &half);
    const SplatCloud& after = tr.cloud();
    bool out_unchanged = true, in_moved = false;
    for (int k = 0; k < 3; k++) {
        out_unchanged &= after.means[k] == c.means[k] && after.scales[k] == c.scales[k] &&
                         after.sh0[k] == c.sh0[k];
        in_moved |= after.sh0[3 + k] != c.sh0[3 + k];
    }
    out_unchanged &= after.opacities[0] == c.opacities[0];
    for (int k = 0; k < 4; k++) out_unchanged &= after.quats[k] == c.quats[k];
    SX_CHECK(out_unchanged);
    SX_CHECK(in_moved);
    std::printf("  [left-half] L1 zero on x<%d, L1+SSIM zero on x<%d; resident: out-of-region "
                "Gaussian unchanged %s, in-region moved %s\n",
                w / 2, w / 2 - 5, out_unchanged ? "yes" : "no", in_moved ? "yes" : "no");
}

// The masked L1+SSIM gradient matches central finite differences of the masked loss, inside the
// region, across the boundary band and outside it.
SX_TEST(mask_l1_ssim_gradient_matches_finite_differences) {
    KernelBackend& b = sxtest::shared_metal_backend();
    const int w = 32, h = 24;
    std::vector<float> render, gt;
    make_images(w, h, render, gt);
    const RegionMask left = make_mask(8, 6, 4);
    const float lam = 0.2f;
    const MaskedLoss base = masked_l1_ssim(b, render, gt, w, h, left, lam, true);
    const int pts[][3] = {{2, 3, 0}, {8, 12, 1},  {12, 5, 2}, {14, 18, 0}, {16, 6, 1},
                          {18, 9, 2}, {20, 15, 0}, {24, 11, 1}, {28, 20, 2}, {30, 2, 0}};
    double worst = 0.0;
    for (const auto& p : pts) {
        const size_t i = (static_cast<size_t>(p[1]) * w + p[0]) * 3 + p[2];
        const float hstep = 1e-2f;
        std::vector<float> rp = render, rm = render;
        rp[i] += hstep;
        rm[i] -= hstep;
        const double fd = (masked_l1_ssim(b, rp, gt, w, h, left, lam, false).loss -
                           masked_l1_ssim(b, rm, gt, w, h, left, lam, false).loss) /
                          (static_cast<double>(rp[i]) - rm[i]);
        const double an = base.grad[i];
        const double err = std::fabs(fd - an);
        const double rel = err / std::max({std::fabs(fd), std::fabs(an), 1e-12});
        std::printf("  [fd] px (%2d,%2d,c%d): analytic %+.6e fd %+.6e rel %.2e\n", p[0], p[1], p[2],
                    an, fd, rel);
        if (std::fabs(an) < 1e-9 && std::fabs(fd) < 1e-7) continue; // both zero (out of region)
        worst = std::max(worst, rel);
    }
    SX_CHECK(worst <= 2e-2);
}

// The resident masked step (train_l1_grad / train_ssim_mask_coeffs / train_depth_grad under
// kTrainHasMask) reproduces the host masked step.
SX_TEST(mask_resident_matches_host_step) {
    KernelBackend& b = sxtest::shared_metal_backend();
    const TrainingCamera cam = make_camera(64, 64, 70.0f);
    const SplatCloud c = make_grid_cloud();
    TrainConfig cfg;
    cfg.total_iters = 100;
    cfg.ssim_weight = 0.2f;
    cfg.depth_weight = 0.5f;
    cfg.lr_means = 2e-3f;
    TrainConfig cfg_host = cfg;
    cfg_host.resident = false;
    Trainer th(b, c, cfg_host);
    Trainer tr(b, c, cfg);
    const DepthTarget depth = make_depth(16, 16, 3.0f);
    RegionMask mask = make_mask(16, 16, 6);
    cache_mask_counts(mask, cam.width, cam.height);
    const std::vector<float> gt(static_cast<size_t>(cam.width) * cam.height * 3, 0.35f);
    for (int it = 0; it < 3; it++) {
        const TrainStepResult rh = th.step(cam, gt.data(), it, &depth, &mask);
        const TrainStepResult rr = tr.step(cam, gt.data(), it, &depth, &mask);
        std::printf("  [masked resident vs host] it %d: loss host %.8f resident %.8f | depth %.7f %.7f\n",
                    it, static_cast<double>(rh.loss), static_cast<double>(rr.loss),
                    static_cast<double>(rh.depth_loss), static_cast<double>(rr.depth_loss));
        SX_CHECK(rh.loss > 0.0f && rh.depth_loss > 0.0f);
        SX_CHECK_NEAR(rr.loss, rh.loss, 1e-5, 0);
        SX_CHECK_NEAR(rr.depth_loss, rh.depth_loss, 1e-4, 1e-7);
    }
    SplatCloud ch = th.cloud(), cr = tr.cloud();
    const double worst = cloud_rel_l2(ch, cr);
    std::printf("  [masked resident vs host] param rel_l2 worst = %.3g (gate 1e-4)\n", worst);
    SX_CHECK(worst <= 1e-4);
}

// Under a mask the depth term counts depth-valid ∩ in-mask samples and has no gradient outside.
SX_TEST(mask_depth_counts_intersection) {
    const int vw = 32, vh = 24;
    DepthTarget t = make_depth(8, 6, 2.0f);
    const RegionMask left = make_mask(8, 6, 3);
    int64_t want = 0;
    for (int y = 0; y < vh; y++)
        for (int x = 0; x < vw; x++) {
            const unsigned sx = sx_depth_sidecar_coord(x, vw, 8), sy = sx_depth_sidecar_coord(y, vh, 6);
            if (t.depth_m[sy * 8 + sx] > 0.0f && left.m[sy * 8 + sx] > 0.0f) want++;
        }
    const int64_t got = depth_valid_count(t, vw, vh, &left);
    SX_CHECK(got == want);
    SX_CHECK(got < depth_valid_count(t, vw, vh));
    std::vector<float> acc(static_cast<size_t>(vw) * vh, 2.4f), alpha(acc.size(), 0.8f);
    const DepthLossResult dl = depth_loss_and_grad(acc.data(), alpha.data(), t, vw, vh, got, 1.0f, &left);
    bool outside_zero = true, inside_nonzero = false;
    for (int y = 0; y < vh; y++)
        for (int x = 0; x < vw; x++) {
            const size_t i = static_cast<size_t>(y) * vw + x;
            const bool in = left.at(x, y, vw, vh) > 0.0f;
            if (!in && (dl.v_render_depths[i] != 0.0f || dl.v_render_alphas[i] != 0.0f))
                outside_zero = false;
            if (in && dl.v_render_depths[i] != 0.0f) inside_nonzero = true;
        }
    SX_CHECK(outside_zero && inside_nonzero);
    // ED = 3 m everywhere against 2 m: L = |1/3 − 1/2| over the counted samples.
    SX_CHECK_NEAR(dl.loss, 1.0 / 6.0, 1e-6, 0);
    DepthErrorSum all, masked;
    std::vector<float> ed(acc.size(), 3.0f);
    accumulate_depth_error(ed.data(), t, vw, vh, all);
    accumulate_depth_error(ed.data(), t, vw, vh, masked, &left);
    SX_CHECK(masked.n == want && all.n == depth_valid_count(t, vw, vh));
    std::printf("  [depth ∩ mask] %lld of %lld depth-valid samples in-mask\n",
                static_cast<long long>(got), static_cast<long long>(all.n));
}

// An all-zero mask: no gradient and no NaN on either path, and the resident step leaves every
// parameter bit-for-bit unchanged.
SX_TEST(mask_all_zero_no_gradient_no_nan) {
    KernelBackend& b = sxtest::shared_metal_backend();
    const int w = 32, h = 24;
    std::vector<float> render, gt;
    make_images(w, h, render, gt);
    const RegionMask zero = make_mask(8, 6, 8);
    SX_CHECK(mask_view_count(zero, w, h) == 0 && mask_ssim_count(zero, w, h) == 0);
    const L1LossResult l1 = l1_loss_and_grad_masked(render.data(), gt.data(), w, h, zero, 0);
    SX_CHECK(l1.loss == 0.0f);
    SX_CHECK(std::all_of(l1.v_render_colors.begin(), l1.v_render_colors.end(),
                         [](float v) { return v == 0.0f; }));
    SX_CHECK(masked_ssim_mean(nullptr, zero, w, h, 3, 0) == 1.0);

    const TrainingCamera cam = make_camera(64, 64, 70.0f);
    const SplatCloud c = make_grid_cloud();
    TrainConfig cfg;
    cfg.total_iters = 100;
    cfg.ssim_weight = 0.2f;
    cfg.depth_weight = 0.5f;
    const DepthTarget depth = make_depth(16, 16, 3.0f);
    const RegionMask none = make_mask(16, 16, 16);
    const std::vector<float> gt_img(static_cast<size_t>(64) * 64 * 3, 0.35f);
    for (const bool resident : {true, false}) {
        TrainConfig k = cfg;
        k.resident = resident;
        Trainer tr(b, c, k);
        bool finite = true;
        for (int it = 0; it < 3; it++) {
            const TrainStepResult r = tr.step(cam, gt_img.data(), it, &depth, &none);
            finite &= std::isfinite(r.loss) && r.loss == 0.0f && r.depth_loss == 0.0f;
        }
        const SplatCloud& after = tr.cloud();
        const bool unchanged = cloud_bitwise_equal(after, c);
        std::printf("  [all-zero %s] loss 0 and finite %s, parameters unchanged %s\n",
                    resident ? "resident" : "host", finite ? "yes" : "no", unchanged ? "yes" : "no");
        SX_CHECK(finite);
        SX_CHECK(unchanged);
        SX_CHECK(all_finite(after.means) && all_finite(after.scales));
    }
}

// The masked PSNR equals the PSNR under an all-ones mask and ignores the masked-out half.
SX_TEST(mask_psnr_all_ones_and_ignores_masked_out) {
    const int w = 32, h = 24;
    std::vector<float> render, gt;
    make_images(w, h, render, gt);
    const RegionMask ones = make_mask(8, 6, 0);
    const float p = psnr(render.data(), gt.data(), static_cast<int64_t>(render.size()));
    const float pm = psnr_masked(render.data(), gt.data(), w, h, ones);
    SX_CHECK_NEAR(pm, p, 1e-6, 0);
    const RegionMask left = make_mask(8, 6, 4);
    const float before = psnr_masked(render.data(), gt.data(), w, h, left);
    std::vector<float> corrupted = render;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w / 2; x++)
            for (int c = 0; c < 3; c++) corrupted[(static_cast<size_t>(y) * w + x) * 3 + c] = 9.0f;
    const float after = psnr_masked(corrupted.data(), gt.data(), w, h, left);
    SX_CHECK(before == after);
    SX_CHECK(psnr(corrupted.data(), gt.data(), static_cast<int64_t>(render.size())) < p - 10.0f);
    SX_CHECK(std::isnan(psnr_masked(render.data(), gt.data(), w, h, make_mask(8, 6, 8))));
    std::printf("  [masked PSNR] all-ones %.4f = PSNR %.4f; left-masked %.4f before and after "
                "corrupting the masked-out half\n",
                static_cast<double>(pm), static_cast<double>(p), static_cast<double>(after));
}

// Mask and depth sidecars of one view must share dims (the loader and the step both throw).
SX_TEST(mask_dims_mismatch_depth_throws) {
    const RegionMask m = make_mask(16, 12, 0);
    const DepthTarget d = make_depth(32, 24, 2.0f);
    bool thrown = false;
    try { check_mask_matches_depth(m, d); }
    catch (const std::runtime_error&) { thrown = true; }
    SX_CHECK(thrown);
    check_mask_matches_depth(m, make_depth(16, 12, 2.0f)); // equal dims pass

    KernelBackend& b = sxtest::shared_metal_backend();
    const TrainingCamera cam = make_camera(64, 48, 60.0f);
    TrainConfig cfg;
    cfg.depth_weight = 0.5f;
    Trainer tr(b, make_grid_cloud(), cfg);
    const std::vector<float> gt(static_cast<size_t>(64) * 48 * 3, 0.5f);
    bool step_thrown = false;
    try { (void)tr.step(cam, gt.data(), 0, &d, &m); }
    catch (const std::runtime_error&) { step_thrown = true; }
    SX_CHECK(step_thrown);
}

// masks/<stem>.png round-trips (any nonzero code is in-region), training_input_from_colmap
// resolves it beside images/, and a 256×192 sidecar samples a 1920×1440 view nearest-neighbour.
SX_TEST(mask_png_round_trip_and_nearest_sampling) {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "splatx-region-mask-test";
    fs::remove_all(root);
    fs::create_directories(root / "images");
    fs::create_directories(root / "masks");
    ImageU16 png;
    png.width = 256;
    png.height = 192;
    png.data.resize(256 * 192);
    for (int y = 0; y < 192; y++)
        for (int x = 0; x < 256; x++)
            png.data[y * 256 + x] = ((x / 7 + y / 5) % 3 == 0) ? 0 : ((x + y) % 2 ? 255 : 1);
    save_png_gray8(png, (root / "masks" / "view_000.png").string());
    const RegionMask m = load_region_mask((root / "masks" / "view_000.png").string());
    SX_CHECK(m.width == 256 && m.height == 192);
    bool codes_ok = true;
    for (size_t i = 0; i < m.m.size(); i++)
        codes_ok &= m.m[i] == (png.data[i] != 0 ? 1.0f : 0.0f);
    SX_CHECK(codes_ok);

    const int vw = 1920, vh = 1440;
    bool sample_ok = true;
    int64_t want = 0;
    for (int y = 0; y < vh; y++)
        for (int x = 0; x < vw; x++) {
            const int sx = static_cast<int>(std::floor((x + 0.5) * 256.0 / vw));
            const int sy = static_cast<int>(std::floor((y + 0.5) * 192.0 / vh));
            const float v = m.m[static_cast<size_t>(sy) * 256 + sx];
            if (v > 0.0f) want++;
            if ((x % 97 == 0 || y % 89 == 0) && m.at(x, y, vw, vh) != v) sample_ok = false;
        }
    SX_CHECK(sample_ok);
    SX_CHECK(mask_view_count(m, vw, vh) == want);

    ColmapBundle bundle;
    bundle.cameras.push_back(ColmapCamera{1, 1, 8, 6, {5.0, 5.0, 4.0, 3.0}, 5.0f, 5.0f, 4.0f, 3.0f});
    bundle.images.push_back(ColmapImage{1, {1, 0, 0, 0}, {0, 0, 0}, 1, "view_000.jpg"});
    bundle.images.push_back(ColmapImage{2, {1, 0, 0, 0}, {0, 0, 0}, 1, "view_001.jpg"});
    const TrainingInput ti = training_input_from_colmap(bundle, (root / "images").string());
    SX_CHECK(ti.cameras.size() == 2);
    SX_CHECK(ti.cameras[0].mask_path == (root / "masks" / "view_000.png").string());
    SX_CHECK(ti.cameras[1].mask_path.empty());
    std::printf("  [mask sidecar] 256x192 -> 1920x1440: %lld in-mask view pixels\n",
                static_cast<long long>(want));
    fs::remove_all(root);
}
