// Depth supervision — the LiDAR depth sidecar, the expected depth and the confidence-masked
// inverse-depth L1. Written from scratch over the gsplat RGB+ED convention (the rasterizer
// composites Σ w·z beside the colours and the expected depth is that sum divided by alpha).
//
// Sidecar convention (beside images/ in the COLMAP folder): depths/<stem>.png is a 16-bit
// greyscale PNG of camera-z depth in millimetres (0 = no sample) and depths/<stem>.conf.png an
// optional 8-bit PNG of ARConfidenceLevel codes (0 low, 1 medium, 2 high). The sidecar may be
// smaller than the view (the ARKit 256×192 depth map); view pixels sample it nearest-neighbour
// (sx_depth_sidecar_coord), so it must share the view's aspect and orientation.
//
// Loss over view pixels p whose sidecar sample is valid:
//   ED_p = acc_p / α_p,   L = (1/M)·Σ |1/ED_p − 1/gt_p|
// M counts the valid sidecar samples over the whole view (independent of the render); pixels with
// α_p < kDepthMinAlpha or ED_p < kDepthMinZ contribute neither loss nor gradient. The total
// training loss adds depth_weight·L to the RGB loss.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace splatx {

// Samples whose confidence code is below this are dropped at load time.
inline constexpr uint16_t kDepthMinConfidence = 1;

struct DepthTarget {
    int width = 0, height = 0;
    std::vector<float> depth_m; // [H,W] metres, 0 = invalid
    int64_t view_valid = -1;    // depth_valid_count for the view, cached by the loader; < 0 = unknown
    bool empty() const { return depth_m.empty(); }
};

// Reads the depth PNG (millimetres) and, when conf_path is non-empty, masks samples below
// kDepthMinConfidence. A confidence image must match the depth dims. Throws on failure.
DepthTarget load_depth_sidecar(const std::string& depth_path, const std::string& conf_path);

// Number of view pixels whose nearest sidecar sample is valid — the loss denominator M.
int64_t depth_valid_count(const DepthTarget& t, int view_w, int view_h);

// The expected depth of one pixel (0 below kDepthMinAlpha).
float expected_depth(float acc, float alpha);

// The unweighted loss L (f64 sum) — the reporting path shared by the host and resident steps.
float depth_l1_loss(const float* acc, const float* alpha, const float* gt, int view_w, int view_h,
                    int depth_w, int depth_h, int64_t n_valid);

struct DepthLossResult {
    float loss = 0.f;                     // unweighted L
    std::vector<float> v_render_depths;   // [H,W] d(weight·L)/d acc
    std::vector<float> v_render_alphas;   // [H,W] d(weight·L)/d α (through the normalisation)
};

// The host formula (train_depth_grad reproduces it on the resident path).
DepthLossResult depth_loss_and_grad(const float* acc, const float* alpha, const DepthTarget& t,
                                    int view_w, int view_h, int64_t n_valid, float weight);

// Squared error of the expected depth against the sidecar over the same mask, for the held-out
// RMSE (metres). ed is the per-pixel expected depth (expected_depth()).
struct DepthErrorSum {
    double sq = 0.0;
    int64_t n = 0;
};
void accumulate_depth_error(const float* ed, const DepthTarget& t, int view_w, int view_h,
                            DepthErrorSum& sum);

} // namespace splatx
