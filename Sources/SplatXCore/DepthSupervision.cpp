// Depth supervision — the contract is in DepthSupervision.h. Written from scratch.
#include "splatx/DepthSupervision.h"
#include "splatx/ImageIO.h"
#include "splatx/RegionMask.h"
#include "KernelConstants.h"
#include <cmath>
#include <stdexcept>

namespace splatx {

namespace {

inline float sidecar_sample(const float* gt, int x, int y, int view_w, int view_h, int depth_w,
                            int depth_h) {
    const unsigned sx = sx_depth_sidecar_coord(static_cast<unsigned>(x),
                                               static_cast<unsigned>(view_w),
                                               static_cast<unsigned>(depth_w));
    const unsigned sy = sx_depth_sidecar_coord(static_cast<unsigned>(y),
                                               static_cast<unsigned>(view_h),
                                               static_cast<unsigned>(depth_h));
    return gt[static_cast<size_t>(sy) * depth_w + sx];
}

inline bool outside(const RegionMask* mask, int x, int y, int view_w, int view_h) {
    return mask && !(mask->at(x, y, view_w, view_h) > 0.0f);
}

} // namespace

DepthTarget load_depth_sidecar(const std::string& depth_path, const std::string& conf_path) {
    const ImageU16 mm = load_png_gray_raw(depth_path);
    DepthTarget t;
    t.width = mm.width;
    t.height = mm.height;
    t.depth_m.resize(mm.data.size());
    for (size_t i = 0; i < mm.data.size(); i++) t.depth_m[i] = static_cast<float>(mm.data[i]) * 1e-3f;
    if (!conf_path.empty()) {
        const ImageU16 conf = load_png_gray_raw(conf_path);
        if (conf.width != mm.width || conf.height != mm.height)
            throw std::runtime_error("depth sidecar: confidence " + conf_path + " dims " +
                                     std::to_string(conf.width) + "x" +
                                     std::to_string(conf.height) + " != depth " +
                                     std::to_string(mm.width) + "x" + std::to_string(mm.height));
        for (size_t i = 0; i < conf.data.size(); i++)
            if (conf.data[i] < kDepthMinConfidence) t.depth_m[i] = 0.0f;
    }
    return t;
}

int64_t depth_valid_count(const DepthTarget& t, int view_w, int view_h, const RegionMask* mask) {
    if (t.empty() || view_w < 1 || view_h < 1) return 0;
    if (mask && mask->empty()) mask = nullptr;
    int64_t n = 0;
    for (int y = 0; y < view_h; y++)
        for (int x = 0; x < view_w; x++)
            if (sidecar_sample(t.depth_m.data(), x, y, view_w, view_h, t.width, t.height) > 0.0f &&
                !outside(mask, x, y, view_w, view_h))
                n++;
    return n;
}

float expected_depth(float acc, float alpha) { return alpha >= kDepthMinAlpha ? acc / alpha : 0.0f; }

float depth_l1_loss(const float* acc, const float* alpha, const float* gt, int view_w, int view_h,
                    int depth_w, int depth_h, int64_t n_valid, const RegionMask* mask) {
    if (n_valid <= 0) return 0.0f;
    if (mask && mask->empty()) mask = nullptr;
    double sum = 0.0;
    for (int y = 0; y < view_h; y++) {
        for (int x = 0; x < view_w; x++) {
            const size_t i = static_cast<size_t>(y) * view_w + x;
            const float g = sidecar_sample(gt, x, y, view_w, view_h, depth_w, depth_h);
            if (!(g > 0.0f) || !(alpha[i] >= kDepthMinAlpha)) continue;
            if (outside(mask, x, y, view_w, view_h)) continue;
            const float ed = acc[i] / alpha[i];
            if (!(ed >= kDepthMinZ)) continue;
            sum += std::fabs(static_cast<double>(1.0f / ed - 1.0f / g));
        }
    }
    return static_cast<float>(sum / static_cast<double>(n_valid));
}

DepthLossResult depth_loss_and_grad(const float* acc, const float* alpha, const DepthTarget& t,
                                    int view_w, int view_h, int64_t n_valid, float weight,
                                    const RegionMask* mask) {
    if (!acc || !alpha || t.empty())
        throw std::invalid_argument("depth_loss_and_grad: null render or empty target");
    if (mask && mask->empty()) mask = nullptr;
    const size_t hw = static_cast<size_t>(view_w) * view_h;
    DepthLossResult r;
    r.v_render_depths.assign(hw, 0.0f);
    r.v_render_alphas.assign(hw, 0.0f);
    if (n_valid <= 0) return r;
    r.loss = depth_l1_loss(acc, alpha, t.depth_m.data(), view_w, view_h, t.width, t.height, n_valid,
                           mask);
    const float g = weight / static_cast<float>(n_valid);
    for (int y = 0; y < view_h; y++) {
        for (int x = 0; x < view_w; x++) {
            const size_t i = static_cast<size_t>(y) * view_w + x;
            const float gt = sidecar_sample(t.depth_m.data(), x, y, view_w, view_h, t.width, t.height);
            const float a = alpha[i];
            if (!(gt > 0.0f) || !(a >= kDepthMinAlpha)) continue;
            if (outside(mask, x, y, view_w, view_h)) continue;
            const float ed = acc[i] / a;
            if (!(ed >= kDepthMinZ)) continue;
            const float d = 1.0f / ed - 1.0f / gt;
            const float s = d > 0.0f ? 1.0f : (d < 0.0f ? -1.0f : 0.0f);
            const float v_acc = g * s * (-1.0f / (ed * ed)) / a;
            r.v_render_depths[i] = v_acc;
            r.v_render_alphas[i] = -v_acc * ed;
        }
    }
    return r;
}

void accumulate_depth_error(const float* ed, const DepthTarget& t, int view_w, int view_h,
                            DepthErrorSum& sum, const RegionMask* mask) {
    if (!ed || t.empty()) return;
    if (mask && mask->empty()) mask = nullptr;
    for (int y = 0; y < view_h; y++) {
        for (int x = 0; x < view_w; x++) {
            const float gt = sidecar_sample(t.depth_m.data(), x, y, view_w, view_h, t.width, t.height);
            const float e = ed[static_cast<size_t>(y) * view_w + x];
            if (!(gt > 0.0f) || !(e >= kDepthMinZ)) continue;
            if (outside(mask, x, y, view_w, view_h)) continue;
            const double d = static_cast<double>(e) - gt;
            sum.sq += d * d;
            sum.n++;
        }
    }
}

} // namespace splatx
