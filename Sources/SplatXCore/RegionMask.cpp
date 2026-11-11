// Region mask — the contract is in RegionMask.h. Written from scratch.
#include "splatx/RegionMask.h"
#include "splatx/DepthSupervision.h"
#include "splatx/ImageIO.h"
#include "KernelConstants.h"
#include <cmath>
#include <stdexcept>

namespace splatx {

namespace {

constexpr int kSsimHalf = static_cast<int>(kSsimWindowSize - 1) / 2; // map pixel q ↔ view centre q + 5

} // namespace

SidecarIndex make_sidecar_index(int side_w, int side_h, int view_w, int view_h) {
    SidecarIndex ix;
    if (side_w < 1 || side_h < 1 || view_w < 1 || view_h < 1) return ix;
    ix.view_w = view_w;
    ix.view_h = view_h;
    ix.row.resize(static_cast<size_t>(view_h));
    ix.col.resize(static_cast<size_t>(view_w));
    for (int y = 0; y < view_h; y++)
        ix.row[static_cast<size_t>(y)] =
            sx_depth_sidecar_coord(static_cast<unsigned>(y), static_cast<unsigned>(view_h),
                                   static_cast<unsigned>(side_h)) *
            static_cast<uint32_t>(side_w);
    for (int x = 0; x < view_w; x++)
        ix.col[static_cast<size_t>(x)] = sx_depth_sidecar_coord(
            static_cast<unsigned>(x), static_cast<unsigned>(view_w), static_cast<unsigned>(side_w));
    return ix;
}

float RegionMask::at_uncached(int x, int y, int view_w, int view_h) const {
    const unsigned sx = sx_depth_sidecar_coord(static_cast<unsigned>(x),
                                               static_cast<unsigned>(view_w),
                                               static_cast<unsigned>(width));
    const unsigned sy = sx_depth_sidecar_coord(static_cast<unsigned>(y),
                                               static_cast<unsigned>(view_h),
                                               static_cast<unsigned>(height));
    return m[static_cast<size_t>(sy) * width + sx];
}

const SidecarIndex& mask_index(const RegionMask& mask, int view_w, int view_h,
                               SidecarIndex& scratch) {
    if (mask.index.matches(view_w, view_h)) return mask.index;
    scratch = make_sidecar_index(mask.width, mask.height, view_w, view_h);
    return scratch;
}

RegionMask load_region_mask(const std::string& path) {
    const ImageU16 raw = load_png_gray_raw(path);
    RegionMask r;
    r.width = raw.width;
    r.height = raw.height;
    r.m.resize(raw.data.size());
    for (size_t i = 0; i < raw.data.size(); i++) r.m[i] = raw.data[i] != 0 ? 1.0f : 0.0f;
    return r;
}

void check_mask_matches_depth(const RegionMask& mask, const DepthTarget& depth) {
    if (mask.empty() || depth.empty()) return;
    if (mask.width != depth.width || mask.height != depth.height)
        throw std::runtime_error("region mask: dims " + std::to_string(mask.width) + "x" +
                                 std::to_string(mask.height) + " != depth sidecar " +
                                 std::to_string(depth.width) + "x" +
                                 std::to_string(depth.height));
}

int64_t mask_view_count(const RegionMask& mask, int view_w, int view_h) {
    if (mask.empty() || view_w < 1 || view_h < 1) return 0;
    int64_t n = 0;
    for (int y = 0; y < view_h; y++)
        for (int x = 0; x < view_w; x++)
            if (mask.at(x, y, view_w, view_h) > 0.0f) n++;
    return n;
}

int64_t mask_ssim_count(const RegionMask& mask, int view_w, int view_h) {
    const int map_w = view_w - 2 * kSsimHalf, map_h = view_h - 2 * kSsimHalf;
    if (mask.empty() || map_w < 1 || map_h < 1) return 0;
    int64_t n = 0;
    for (int y = 0; y < map_h; y++)
        for (int x = 0; x < map_w; x++)
            if (mask.at(x + kSsimHalf, y + kSsimHalf, view_w, view_h) > 0.0f) n++;
    return n;
}

void cache_mask_counts(RegionMask& mask, int view_w, int view_h) {
    mask.index = mask.empty() ? SidecarIndex{}
                              : make_sidecar_index(mask.width, mask.height, view_w, view_h);
    mask.view_count = mask_view_count(mask, view_w, view_h);
    mask.ssim_count = mask_ssim_count(mask, view_w, view_h);
}

void mask_ssim_coeffs_inplace(float* coeff_mu, float* coeff_sigma, float* coeff_cross,
                              const RegionMask& mask, int view_w, int view_h, int channels) {
    const int map_w = view_w - 2 * kSsimHalf, map_h = view_h - 2 * kSsimHalf;
    if (map_w < 1 || map_h < 1) return;
    for (int y = 0; y < map_h; y++) {
        for (int x = 0; x < map_w; x++) {
            const float mv = mask.at(x + kSsimHalf, y + kSsimHalf, view_w, view_h);
            const size_t base = (static_cast<size_t>(y) * map_w + x) * channels;
            for (int c = 0; c < channels; c++) {
                coeff_mu[base + c] *= mv;
                coeff_sigma[base + c] *= mv;
                coeff_cross[base + c] *= mv;
            }
        }
    }
}

double masked_l1_sum(const float* render, const float* gt, const RegionMask& mask, int view_w,
                     int view_h) {
    if (mask.empty() || view_w < 1 || view_h < 1) return 0.0;
    SidecarIndex scratch;
    const SidecarIndex& ix = mask_index(mask, view_w, view_h, scratch);
    const uint32_t* col = ix.col.data();
    const float* mv = mask.m.data();
    double acc = 0.0;
    for (int y = 0; y < view_h; y++) {
        const float* mrow = mv + ix.row[static_cast<size_t>(y)];
        const float* r = render + static_cast<size_t>(y) * view_w * 3;
        const float* g = gt + static_cast<size_t>(y) * view_w * 3;
        for (int x = 0; x < view_w; x++) {
            if (!(mrow[col[x]] > 0.0f)) continue;
            for (int c = 0; c < 3; c++)
                acc += std::fabs(static_cast<double>(r[3 * x + c] - g[3 * x + c]));
        }
    }
    return acc;
}

double masked_ssim_mean(const float* ssim_map, const RegionMask& mask, int view_w, int view_h,
                        int channels, int64_t ssim_count) {
    const int map_w = view_w - 2 * kSsimHalf, map_h = view_h - 2 * kSsimHalf;
    if (ssim_count <= 0 || map_w < 1 || map_h < 1) return 1.0;
    SidecarIndex scratch;
    const SidecarIndex& ix = mask_index(mask, view_w, view_h, scratch);
    const uint32_t* col = ix.col.data() + kSsimHalf;
    const float* mv = mask.m.data();
    double acc = 0.0;
    for (int y = 0; y < map_h; y++) {
        const float* mrow = mv + ix.row[static_cast<size_t>(y + kSsimHalf)];
        const float* srow = ssim_map + static_cast<size_t>(y) * map_w * channels;
        for (int x = 0; x < map_w; x++) {
            if (!(mrow[col[x]] > 0.0f)) continue;
            const float* s = srow + static_cast<size_t>(x) * channels;
            for (int c = 0; c < channels; c++) acc += static_cast<double>(s[c]);
        }
    }
    return acc / static_cast<double>(static_cast<int64_t>(channels) * ssim_count);
}

} // namespace splatx
