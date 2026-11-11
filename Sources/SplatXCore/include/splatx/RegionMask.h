// Region mask — the per-view masks/<stem>.png sidecar that confines the photometric and depth
// losses to a region of the scene. Written from scratch.
//
// Sidecar convention (beside images/ in the COLMAP folder, like depths/): masks/<stem>.png is an
// 8-bit greyscale PNG, nonzero = in-region. It may be smaller than the view (the ARKit 256×192
// grid); view pixels sample it nearest-neighbour through sx_depth_sidecar_coord, exactly as the
// depth sidecar is sampled, so it must share the view's aspect and orientation. When a view also
// has a depth sidecar the two must have the same dims. A view without a mask file is unmasked.
//
// Masked losses (m(p) ∈ {0,1}, M = Σ m over the view's pixels):
//   L1   = (1/(3M))·Σ_p m(p)·Σ_c |r − g|
//   SSIM = (1/(3·Ms))·Σ_q m(centre(q))·S(q)   over the valid map, Ms = in-mask map pixels
//   depth: a sample counts only when it is depth-valid and in-mask (DepthSupervision.h)
// With an all-ones mask every term reduces to the unmasked one bit for bit. A view with M = 0
// contributes no photometric or depth gradient.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace splatx {

struct DepthTarget;

struct RegionMask {
    int width = 0, height = 0;
    std::vector<float> m;    // [H,W] row-major, 1 = in-region, 0 = out
    int64_t view_count = -1; // M for the view, cached by the loader; < 0 = unknown
    int64_t ssim_count = -1; // Ms for the view, cached by the loader; < 0 = unknown
    bool empty() const { return m.empty(); }
    // The mask value of view pixel (x, y) for a view_w × view_h view.
    float at(int x, int y, int view_w, int view_h) const;
};

// Reads masks/<stem>.png (8-bit or 16-bit grey; channel 0 of anything wider). Throws on failure.
RegionMask load_region_mask(const std::string& path);

// Throws when a view's mask and depth sidecar disagree in dims.
void check_mask_matches_depth(const RegionMask& mask, const DepthTarget& depth);

// M: in-mask view pixels.
int64_t mask_view_count(const RegionMask& mask, int view_w, int view_h);
// Ms: valid SSIM map pixels ((H−10)·(W−10)) whose window centre is in-mask.
int64_t mask_ssim_count(const RegionMask& mask, int view_w, int view_h);
// Fills view_count and ssim_count for the view.
void cache_mask_counts(RegionMask& mask, int view_w, int view_h);

// Multiplies the three SSIM coefficient maps ([map_h,map_w,channels]) by m(centre) — the masked
// dL/dS of each map pixel. The resident twin is train_ssim_mask_coeffs.
void mask_ssim_coeffs_inplace(float* coeff_mu, float* coeff_sigma, float* coeff_cross,
                              const RegionMask& mask, int view_w, int view_h, int channels);

// Σ m(centre(q))·S(q) / (channels·Ms), summed in f64 in index order (the ssim_reduce_mean_f64
// rule). 1 when Ms = 0 (no SSIM term).
double masked_ssim_mean(const float* ssim_map, const RegionMask& mask, int view_w, int view_h,
                        int channels, int64_t ssim_count);

} // namespace splatx
