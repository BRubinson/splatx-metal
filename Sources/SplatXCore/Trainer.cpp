// Trainer — Gaussian initialization (init_splats), L1 loss/grad, activation helpers, the training
// loop (Task 7), and the MCMC densification strategy (Phase 2 Task 3).
// (The initialization recipe reproduces the formulas of 3DGS/gsplat common practice — it is not
// derived from their code. The LR and schedule numbers of the training loop reproduce the gsplat
// examples/simple_trainer.py recipe, and the MCMC orchestration is a reimplementation of the
// procedure in gsplat strategy/mcmc.py:139–239 + strategy/ops.py:301–401 — derivation notice in
// docs/provenance.md. The D7 order assembly, the activation gradient chain, and the fixed cap_max
// arena are written from scratch.)
#include "splatx/Trainer.h"
#include "splatx/CpuKernels.h" // eval_ssim — reuses ssim_fwd_cpu (perceptual metrics Task 1)
#include "splatx/KernelBackend.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace splatx {

namespace {

// Upper bound on the MCMC Eq.9 ratio = the binoms LUT dimension (gsplat strategy/mcmc.py:95
// n_max=51).
constexpr int kMcmcNMax = 51;

// ===== Shared 3-NN convention (documented at knn3_mean_dists in Trainer.h — the record of the
// correctness argument) =====

constexpr float kKnnInf = std::numeric_limits<float>::infinity();

// Inserts into the ascending top 3 (squared distances) — ties break by ascending index (the 3
// smallest under lexicographic (d2, j)). The same (d2, j) set is determined regardless of scan
// order — deterministic (the argument is in Trainer.h).
inline void knn3_insert(float d2, int32_t j, float best_d2[3], int32_t best_j[3]) {
    if (d2 < best_d2[0] || (d2 == best_d2[0] && j < best_j[0])) {
        best_d2[2] = best_d2[1]; best_j[2] = best_j[1];
        best_d2[1] = best_d2[0]; best_j[1] = best_j[0];
        best_d2[0] = d2; best_j[0] = j;
    } else if (d2 < best_d2[1] || (d2 == best_d2[1] && j < best_j[1])) {
        best_d2[2] = best_d2[1]; best_j[2] = best_j[1];
        best_d2[1] = d2; best_j[1] = j;
    } else if (d2 < best_d2[2] || (d2 == best_d2[2] && j < best_j[2])) {
        best_d2[2] = d2; best_j[2] = j;
    }
}

// Mean of the sqrt sum over the ascending top 3 — brute-force and grid sum in the same order, so
// they are bit-identical.
inline float knn3_mean_from_best(const float best_d2[3]) {
    float sum = 0.0f;
    int k = 0;
    for (int t = 0; t < 3; t++) {
        if (best_d2[t] == kKnnInf) break;
        sum += std::sqrt(best_d2[t]);
        k++;
    }
    return k > 0 ? sum / static_cast<float>(k) : 0.0f;
}

// Mean 3-NN distance of point i (brute-force, excluding itself) — the small-n path (the same
// arithmetic as the grid). Neighbor count k = min(3, n-1); if k == 0 (n == 1) it returns 0 — the
// caller's 1e-6 clamp handles that.
float knn3_mean_dist_brute(const float* pts, int64_t n, int64_t i) {
    float best_d2[3] = {kKnnInf, kKnnInf, kKnnInf};
    int32_t best_j[3] = {-1, -1, -1};
    const float xi = pts[i * 3 + 0], yi = pts[i * 3 + 1], zi = pts[i * 3 + 2];
    for (int64_t j = 0; j < n; j++) {
        if (j == i) continue; // excludes itself by index — coincident points stay as distance-0 neighbors
        const float dx = pts[j * 3 + 0] - xi;
        const float dy = pts[j * 3 + 1] - yi;
        const float dz = pts[j * 3 + 2] - zi;
        knn3_insert(dx * dx + dy * dy + dz * dz, static_cast<int32_t>(j), best_d2, best_j);
    }
    return knn3_mean_from_best(best_d2);
}

} // namespace

std::vector<float> knn3_mean_dists(const float* pts, int64_t n) {
    if (n < 0 || (n > 0 && !pts))
        throw std::invalid_argument("knn3_mean_dists: null pts or n < 0");
    // ncells ≤ 4n (the cap below) must fit an int32 cell index → n ≤ INT32_MAX/4
    if (n > std::numeric_limits<int32_t>::max() / 4)
        throw std::invalid_argument("knn3_mean_dists: n exceeds int32 grid limit (4n cells)");
    std::vector<float> out(static_cast<size_t>(n), 0.0f);
    if (n <= 1) return out;

    // For small inputs brute-force is cheaper (it avoids the grid construction overhead) — the
    // results are identical (both are exact).
    constexpr int64_t kBruteMaxN = 256;
    if (n <= kBruteMaxN) {
        for (int64_t i = 0; i < n; i++) out[i] = knn3_mean_dist_brute(pts, n, i);
        return out;
    }

    // ---- 1) Grid box: the 1–99% quantiles per axis (so outliers do not contaminate the cell size —
    // points outside the box are absorbed by clamping to the boundary cell, and correctness is
    // preserved by the clamp-contraction argument. Documented in Trainer.h) ----
    double lo[3], hi[3];
    {
        std::vector<float> axis(static_cast<size_t>(n));
        const int64_t q_lo = n / 100, q_hi = n - 1 - n / 100; // the 1% / 99% ranks
        for (int a = 0; a < 3; a++) {
            for (int64_t i = 0; i < n; i++) axis[i] = pts[i * 3 + a];
            std::nth_element(axis.begin(), axis.begin() + q_lo, axis.end());
            lo[a] = axis[q_lo];
            std::nth_element(axis.begin(), axis.begin() + q_hi, axis.end());
            hi[a] = axis[q_hi];
        }
    }
    // If the quantile box is degenerate on every axis (most points share a coordinate), fall back to
    // the full bbox
    if (!(hi[0] > lo[0] || hi[1] > lo[1] || hi[2] > lo[2])) {
        for (int a = 0; a < 3; a++) { lo[a] = kKnnInf; hi[a] = -kKnnInf; }
        for (int64_t i = 0; i < n; i++) {
            for (int a = 0; a < 3; a++) {
                const double v = pts[i * 3 + a];
                lo[a] = std::min(lo[a], v);
                hi[a] = std::max(hi[a], v);
            }
        }
        // If the full bbox is degenerate too → all points share a coordinate → all distances are 0
        // (early return — avoids O(N²) cells)
        if (!(hi[0] > lo[0] || hi[1] > lo[1] || hi[2] > lo[2])) return out;
    }

    // ---- 2) Cell edge: (volume × target density / n)^(1/d) over the d active axes — the target is
    // 0.5 points per cell (measured on garden 138k at t∈{8,2,1,0.5,0.25}: 941/388/269/196/199 ms —
    // the data is clustered, so cells must be subdivided finely to reduce the candidate count in
    // dense cells; 0.25 hits the 4n cap below and gains nothing).
    // The total cell count is capped at 4n (it prevents dims from exploding on quasi-planar or skewed
    // distributions — if it does not fit, the cell grows; the natural cell count 2n at t=0.5 is
    // within the cap). **No per-axis cap is imposed** — the grid MUST always cover the quantile box
    // (coverage preservation), which blocks the pathology where, under extreme anisotropy (e.g.
    // 1e6×1×1e-3), most points clamp into boundary cells and the ring expansion explodes (review fix
    // 1: a cap firing counts as "does not fit", so the cell size keeps growing. Measured at n=750:
    // 60s+ → milliseconds) ----
    const double ext[3] = {hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]};
    double prod = 1.0;
    int d_active = 0;
    for (int a = 0; a < 3; a++) {
        if (ext[a] > 0.0) { prod *= ext[a]; d_active++; }
    }
    constexpr double kTargetPerCell = 0.5;
    double cell = std::pow(prod * kTargetPerCell / static_cast<double>(n),
                           1.0 / static_cast<double>(d_active));
    const double max_cells = 4.0 * static_cast<double>(n);
    int64_t dims[3];
    for (;;) {
        double da[3], dprod = 1.0; // kept in double (with an anisotropic initial cell one axis can
        for (int a = 0; a < 3; a++) { // reach 1e6+ cells — the fit test runs before the int64 cast)
            da[a] = ext[a] > 0.0 ? std::floor(ext[a] / cell) + 1.0 : 1.0;
            dprod *= da[a];
        }
        if (dprod <= max_cells) { // it fits — only now cast (each axis ≤ dprod ≤ 4n)
            for (int a = 0; a < 3; a++) dims[a] = static_cast<int64_t>(da[a]);
            break;
        }
        cell *= 1.2599210498948732; // 2^(1/3) — doubles the cell volume each time (geometric convergence, finite termination)
    }
    const int64_t nx = dims[0], ny = dims[1], nz = dims[2];
    const int64_t ncells = nx * ny * nz; // <= 4n <= the int32 range (guarded by n <= INT32_MAX/4)

    // ---- 3) Cell assignment (with clamping) + counting sort (CSR buckets — ascending index within
    // a cell) ----
    auto cell_coord = [&](int64_t i, int a) -> int64_t {
        const double v = std::floor((static_cast<double>(pts[i * 3 + a]) - lo[a]) / cell);
        return std::clamp(static_cast<int64_t>(v), int64_t{0}, dims[a] - 1);
    };
    std::vector<int32_t> cell_of(static_cast<size_t>(n));
    std::vector<int32_t> bucket_start(static_cast<size_t>(ncells) + 1, 0);
    for (int64_t i = 0; i < n; i++) {
        const int64_t c = (cell_coord(i, 2) * ny + cell_coord(i, 1)) * nx + cell_coord(i, 0);
        cell_of[i] = static_cast<int32_t>(c);
        bucket_start[c + 1]++;
    }
    for (int64_t c = 0; c < ncells; c++) bucket_start[c + 1] += bucket_start[c];
    std::vector<int32_t> bucket_pts(static_cast<size_t>(n));
    {
        std::vector<int32_t> cursor(bucket_start.begin(), bucket_start.end() - 1);
        for (int64_t i = 0; i < n; i++)
            bucket_pts[cursor[cell_of[i]]++] = static_cast<int32_t>(i); // stable (ascending i)
    }

    // ---- 4) Query: expand rings r = 0, 1, ... (the termination and correctness arguments are in
    // Trainer.h) ----
    const int64_t kmin = std::min<int64_t>(3, n - 1);
    for (int64_t i = 0; i < n; i++) {
        const float xi = pts[i * 3 + 0], yi = pts[i * 3 + 1], zi = pts[i * 3 + 2];
        const int64_t ci[3] = {cell_coord(i, 0), cell_coord(i, 1), cell_coord(i, 2)};
        float best_d2[3] = {kKnnInf, kKnnInf, kKnnInf};
        int32_t best_j[3] = {-1, -1, -1};

        auto scan_cell = [&](int64_t cx, int64_t cy, int64_t cz) {
            const int64_t c = (cz * ny + cy) * nx + cx;
            for (int32_t t = bucket_start[c]; t < bucket_start[c + 1]; t++) {
                const int32_t j = bucket_pts[t];
                if (j == static_cast<int32_t>(i)) continue; // excludes itself, by index
                const float dx = pts[j * 3 + 0] - xi;
                const float dy = pts[j * 3 + 1] - yi;
                const float dz = pts[j * 3 + 2] - zi;
                knn3_insert(dx * dx + dy * dy + dz * dz, j, best_d2, best_j);
            }
        };

        // The ring that covers the whole grid — reaching it means every point has been scanned
        // (equivalent to the brute-force fallback)
        int64_t r_cover = 0;
        for (int a = 0; a < 3; a++)
            r_cover = std::max(r_cover, std::max(ci[a], dims[a] - 1 - ci[a]));

        for (int64_t r = 0;; r++) {
            if (r == 0) {
                scan_cell(ci[0], ci[1], ci[2]);
            } else {
                // Chebyshev shell (a non-overlapping decomposition into 6 faces): both z faces in
                // full, both y faces over the z interior, both x faces over the y and z interiors.
                // The enumeration range is clipped to the real grid bounds — cells outside are never
                // iterated, so the cost per ring is proportional to the number of in-bounds cells
                // (review fix 1: on an anisotropically shrunk grid there is no (2r+1)² idle spinning
                // even when r_cover is large)
                const int64_t xlo = std::max(ci[0] - r, int64_t{0});
                const int64_t xhi = std::min(ci[0] + r, nx - 1);
                const int64_t ylo = std::max(ci[1] - r, int64_t{0});
                const int64_t yhi = std::min(ci[1] + r, ny - 1);
                const int64_t izlo = std::max(ci[2] - r + 1, int64_t{0}); // z interior
                const int64_t izhi = std::min(ci[2] + r - 1, nz - 1);
                const int64_t iylo = std::max(ci[1] - r + 1, int64_t{0}); // y interior
                const int64_t iyhi = std::min(ci[1] + r - 1, ny - 1);
                for (int64_t dz : {-r, r}) {
                    const int64_t cz = ci[2] + dz;
                    if (cz < 0 || cz >= nz) continue;
                    for (int64_t cy = ylo; cy <= yhi; cy++)
                        for (int64_t cx = xlo; cx <= xhi; cx++)
                            scan_cell(cx, cy, cz);
                }
                for (int64_t dy : {-r, r}) {
                    const int64_t cy = ci[1] + dy;
                    if (cy < 0 || cy >= ny) continue;
                    for (int64_t cz = izlo; cz <= izhi; cz++)
                        for (int64_t cx = xlo; cx <= xhi; cx++)
                            scan_cell(cx, cy, cz);
                }
                for (int64_t dx : {-r, r}) {
                    const int64_t cx = ci[0] + dx;
                    if (cx < 0 || cx >= nx) continue;
                    for (int64_t cz = izlo; cz <= izhi; cz++)
                        for (int64_t cy = iylo; cy <= iyhi; cy++)
                            scan_cell(cx, cy, cz);
                }
            }
            if (best_d2[kmin - 1] != kKnnInf) {
                // Lower bound on the true distance beyond ring r is (r·cell)² — with a 1e-6 f32
                // rounding margin (the argument is in Trainer.h)
                const double bound = static_cast<double>(r) * cell;
                if (static_cast<double>(best_d2[kmin - 1]) < bound * bound * (1.0 - 1e-6))
                    break;
            }
            if (r >= r_cover) break; // every cell has been scanned
        }
        out[i] = knn3_mean_from_best(best_d2);
    }
    return out;
}

SplatCloud init_splats(const TrainingInput& input, const InitOptions& opts) {
    const int64_t M = static_cast<int64_t>(input.seed_points.size()) / 3;
    if (M < 1 || input.seed_points.size() % 3 != 0)
        throw std::invalid_argument("init_splats: seed_points must be non-empty [M,3]");
    if (input.seed_rgb.size() != input.seed_points.size())
        throw std::invalid_argument("init_splats: seed_rgb size != seed_points size");
    if (opts.max_points < 1)
        throw std::invalid_argument("init_splats: max_points must be >= 1");
    if (!(opts.init_opacity > 0.0f && opts.init_opacity < 1.0f))
        throw std::invalid_argument("init_splats: init_opacity must be in (0,1)");
    if (!(opts.init_scale > 0.0f)) // rejects NaN too — the log domain (a 0 multiplier gives log(0) = -inf)
        throw std::invalid_argument("init_splats: init_scale must be > 0");
    if (opts.sh_degree < 0 || opts.sh_degree > 3)
        throw std::invalid_argument("init_splats: sh_degree must be in [0,3]");

    // Uniform-stride subsample — pick(i) = i·M/N (int64 multiply: N <= max_points, so no overflow);
    // at N == M it is the identity. The O(N²) KNN presupposes this cap (default 50k — brief).
    const int64_t N = std::min(M, opts.max_points);

    SplatCloud c;
    c.count = N;
    c.sh_degree = opts.sh_degree;
    c.means.resize(N * 3);
    c.scales.resize(N * 3);
    c.quats.assign(N * 4, 0.0f);
    c.opacities.assign(N, logit_f32(opts.init_opacity)); // logit-space (the SplatCloud convention)
    c.sh0.resize(N * 3);
    c.shN.assign(static_cast<size_t>(N) * c.sh_rest_coeffs() * 3, 0.0f); // higher bands are 0

    constexpr float kInv255 = 1.0f / 255.0f;
    for (int64_t i = 0; i < N; i++) {
        const int64_t src = i * M / N;
        for (int k = 0; k < 3; k++) {
            c.means[i * 3 + k] = input.seed_points[src * 3 + k];
            // sh0 = (rgb01 - 0.5)/C0 — the inverse SH DC transform (the inverse of the render post-op +0.5)
            c.sh0[i * 3 + k] =
                (static_cast<float>(input.seed_rgb[src * 3 + k]) * kInv255 - 0.5f) / kShC0;
        }
        c.quats[i * 4 + 0] = 1.0f; // unit quat (w-first)
    }

    // scales = log(mean 3-NN distance × init_scale), isotropic so all 3 components are equal —
    // init_scale is the **multiplier** in gsplat simple_trainer.py:323 `log(dist_avg·init_scale)`
    // (it stays 3-NN based; the MCMC preset is 0.1). A distance of 0 (all neighbors coincident) and
    // N==1 are clamped to 1e-6 before the multiply — this blocks log(0) = -inf (the brief's guard; at
    // init_scale = 1.0 it is bit-identical to 1c). The 3-NN values are computed in bulk on a uniform
    // grid (knn3_mean_dists — an exact algorithm, bit-identical to brute-force. Phase 6 round 2
    // Task 1).
    const std::vector<float> knn_means = knn3_mean_dists(c.means.data(), N);
    for (int64_t i = 0; i < N; i++) {
        const float mean_dist = std::max(knn_means[i], 1e-6f);
        const float log_s = std::log(mean_dist * opts.init_scale);
        c.scales[i * 3 + 0] = log_s;
        c.scales[i * 3 + 1] = log_s;
        c.scales[i * 3 + 2] = log_s;
    }
    return c;
}

L1LossResult l1_loss_and_grad(const float* render, const float* gt, int64_t n_elems) {
    if (!render || !gt || n_elems <= 0)
        throw std::invalid_argument("l1_loss_and_grad: null buffers or n_elems <= 0");
    L1LossResult out;
    out.v_render_colors.resize(n_elems);
    const float g = 1.0f / static_cast<float>(n_elems);
    double acc = 0.0; // only the summation accumulates in double (prevents f32 accumulation error over H·W·3 ~ 1e7 elements)
    for (int64_t i = 0; i < n_elems; i++) {
        const float d = render[i] - gt[i]; // the difference itself stays f32 (the same width as the kernel pipeline)
        acc += std::fabs(static_cast<double>(d));
        // sign(0) = 0 (the torch.sign convention — it adopts subgradient 0 for |x|)
        out.v_render_colors[i] = d > 0.0f ? g : (d < 0.0f ? -g : 0.0f);
    }
    out.loss = static_cast<float>(acc / static_cast<double>(n_elems));
    return out;
}

L1LossResult l1_loss_and_grad_masked(const float* render, const float* gt, int view_w, int view_h,
                                     const RegionMask& mask, int64_t mask_count) {
    if (!render || !gt || view_w < 1 || view_h < 1 || mask.empty())
        throw std::invalid_argument("l1_loss_and_grad_masked: null buffers, empty view or mask");
    L1LossResult out;
    const int64_t n_elems = static_cast<int64_t>(view_w) * view_h * 3;
    out.v_render_colors.assign(static_cast<size_t>(n_elems), 0.0f);
    if (mask_count <= 0) return out;
    const float g = 1.0f / static_cast<float>(3 * mask_count);
    double acc = 0.0;
    for (int y = 0; y < view_h; y++) {
        for (int x = 0; x < view_w; x++) {
            const float m = mask.at(x, y, view_w, view_h);
            if (!(m > 0.0f)) continue;
            const size_t p = (static_cast<size_t>(y) * view_w + x) * 3;
            for (size_t c = 0; c < 3; c++) {
                const float d = render[p + c] - gt[p + c];
                acc += std::fabs(static_cast<double>(d));
                out.v_render_colors[p + c] = (d > 0.0f ? g : (d < 0.0f ? -g : 0.0f)) * m;
            }
        }
    }
    out.loss = static_cast<float>(acc / static_cast<double>(3 * mask_count));
    return out;
}

float psnr_masked(const float* a, const float* b, int view_w, int view_h, const RegionMask& mask) {
    if (!a || !b || view_w < 1 || view_h < 1 || mask.empty())
        throw std::invalid_argument("psnr_masked: null buffers, empty view or mask");
    double acc = 0.0;
    int64_t n = 0;
    for (int y = 0; y < view_h; y++) {
        for (int x = 0; x < view_w; x++) {
            if (!(mask.at(x, y, view_w, view_h) > 0.0f)) continue;
            const size_t p = (static_cast<size_t>(y) * view_w + x) * 3;
            for (size_t c = 0; c < 3; c++) {
                const double d = static_cast<double>(a[p + c]) - static_cast<double>(b[p + c]);
                acc += d * d;
            }
            n += 3;
        }
    }
    if (n == 0) return std::numeric_limits<float>::quiet_NaN();
    const double mse = acc / static_cast<double>(n);
    if (mse <= 0.0) return std::numeric_limits<float>::infinity();
    return static_cast<float>(10.0 * std::log10(1.0 / mse));
}

// ===== PSNR + split (Phase 2 Task 5 — the basis for the definitions is the Trainer.h comment) =====

float psnr(const float* a, const float* b, int64_t n_elems) {
    if (!a || !b || n_elems <= 0)
        throw std::invalid_argument("psnr: null buffers or n_elems <= 0");
    double acc = 0.0; // the sum of squares accumulates in double (the same policy as l1_loss_and_grad)
    for (int64_t i = 0; i < n_elems; i++) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        acc += d * d;
    }
    const double mse = acc / static_cast<double>(n_elems);
    if (mse <= 0.0) return std::numeric_limits<float>::infinity(); // identical images
    return static_cast<float>(10.0 * std::log10(1.0 / mse));
}

// A port of gsplat color_correct.py:98–149 — the basis for the formula, the guards, and the operation
// order is the Trainer.h comment.
void color_correct_affine(const float* img, const float* ref, int64_t n_elems, float* out) {
    if (!img || !ref || !out || n_elems <= 0 || n_elems % 3 != 0)
        throw std::invalid_argument(
            "color_correct_affine: null buffers, n_elems <= 0, or n_elems % 3 != 0");
    const double n_px = static_cast<double>(n_elems / 3); // pixel count (samples per channel)
    for (int c = 0; c < 3; c++) {
        // Four moments per channel — accumulated in double (the same policy as
        // psnr/l1_loss_and_grad); mean = sum/n as a division (matching torch .mean() — it avoids the
        // extra rounding of a reciprocal multiply)
        double ref_sum = 0.0, img_sum = 0.0, ref_img_sum = 0.0, ref_ref_sum = 0.0;
        for (int64_t i = c; i < n_elems; i += 3) {
            const double r = static_cast<double>(ref[i]);
            const double m = static_cast<double>(img[i]);
            ref_sum += r;
            img_sum += m;
            ref_img_sum += r * m;
            ref_ref_sum += r * r;
        }
        const double ref_mean = ref_sum / n_px;
        const double img_mean = img_sum / n_px;
        const double var_ref = std::max(ref_ref_sum / n_px - ref_mean * ref_mean, 1e-8);
        double a = (ref_img_sum / n_px - ref_mean * img_mean) / var_ref;
        const double b = img_mean - a * ref_mean; // uses a from **before** the replacement (the original order)
        if (std::fabs(a) < 1e-8) a = 1.0;         // divide-by-zero guard (same as the reference)
        for (int64_t i = c; i < n_elems; i += 3) {
            const double v = (static_cast<double>(img[i]) - b) / a;
            out[i] = static_cast<float>(std::min(std::max(v, 0.0), 1.0)); // clip [0,1]
        }
    }
}

// SSIM scalar for eval only — fwd-only, with the same convention as the training loss (valid
// padding, 11×11 σ=1.5, C1/C2 defined once in KernelConstants). If a backend is present it goes
// through the golden-verified backend->ssim_fwd (on Metal that means the GPU — training-speed Task 1:
// it removes the CPU-SSIM dominant term of the eval tick); otherwise it uses the previous
// ssim_fwd_cpu (the oracle is retained — the reference for the cross-check tests). On both paths the
// scalar reduction is the single source of truth ssim_reduce_mean_f64, a host f64 sequential sum
// (08 §5 — device atomic float is prohibited).
// At h<11||w<11 the valid map is empty and measurement is impossible — the backend and training
// validators throw on the same condition (Validate.h), but eval is observation rather than contract,
// so it merely reports NaN before calling the backend.
double eval_ssim(const float* img, const float* ref, int32_t h, int32_t w,
                 KernelBackend* backend) {
    if (!img || !ref) throw std::invalid_argument("eval_ssim: null buffer");
    if (h < 11 || w < 11) return std::numeric_limits<double>::quiet_NaN();
    SsimInputs si;
    si.img1 = img;
    si.img2 = ref;
    si.I = 1;
    si.H = h;
    si.W = w;
    si.C = 3;
    si.need_coeffs = false; // fwd-only — the 3 coefficient maps are not produced (bwd does not need them)
    return backend ? backend->ssim_fwd(si).ssim : ssim_fwd_cpu(si).ssim;
}

TrainTestSplit make_train_test_split(int64_t n, int test_every) {
    TrainTestSplit s;
    if (n < 0) throw std::invalid_argument("make_train_test_split: negative n");
    for (int64_t i = 0; i < n; i++) {
        // gsplat colmap.py:456–461 — test_every <= 0 turns the split off (everything is train)
        (test_every > 0 && i % test_every == 0 ? s.test : s.train).push_back(i);
    }
    return s;
}

float sigmoid_f32(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float logit_f32(float p) {
    if (!(p > 0.0f && p < 1.0f))
        throw std::invalid_argument("logit_f32: p must be in (0,1)");
    return std::log(p / (1.0f - p));
}

std::vector<float> activate_scales(const float* log_scales, int64_t n) {
    std::vector<float> out(n);
    for (int64_t i = 0; i < n; i++) out[i] = std::exp(log_scales[i]);
    return out;
}

std::vector<float> inv_activate_scales(const float* scales, int64_t n) {
    std::vector<float> out(n);
    for (int64_t i = 0; i < n; i++) {
        // Guard (task-6 handover, Minor): scale <= 0 or NaN is outside the log domain — throw instead
        // of propagating -inf/NaN
        if (!(scales[i] > 0.0f))
            throw std::invalid_argument("inv_activate_scales: scales must be > 0 (log domain)");
        out[i] = std::log(scales[i]);
    }
    return out;
}

std::vector<float> activate_opacities(const float* logits, int64_t n) {
    std::vector<float> out(n);
    for (int64_t i = 0; i < n; i++) out[i] = sigmoid_f32(logits[i]);
    return out;
}

std::vector<float> inv_activate_opacities(const float* opacs, int64_t n) {
    std::vector<float> out(n);
    for (int64_t i = 0; i < n; i++) out[i] = logit_f32(opacs[i]);
    return out;
}

// ===== Training loop (Task 7 — D7 order, host round-trip chaining D11) =====

float scene_scale_from_cameras(const std::vector<TrainingCamera>& cams) {
    const size_t n = cams.size();
    if (n < 2) return 1.0f; // a single camera — the spread is undefined, so default to 1
    // campos = -Rᵀt (viewmat world→cam row-major: cam = R·world + t)
    std::vector<double> pos(n * 3);
    for (size_t c = 0; c < n; c++) {
        const float* vm = cams[c].viewmat;
        for (int i = 0; i < 3; i++)
            pos[c * 3 + i] = -(static_cast<double>(vm[0 * 4 + i]) * vm[0 * 4 + 3] +
                               static_cast<double>(vm[1 * 4 + i]) * vm[1 * 4 + 3] +
                               static_cast<double>(vm[2 * 4 + i]) * vm[2 * 4 + 3]);
    }
    double cx = 0, cy = 0, cz = 0;
    for (size_t c = 0; c < n; c++) { cx += pos[c * 3]; cy += pos[c * 3 + 1]; cz += pos[c * 3 + 2]; }
    cx /= static_cast<double>(n); cy /= static_cast<double>(n); cz /= static_cast<double>(n);
    // The gsplat formula: **the maximum distance from the centroid × 1.1** (scene_scale = max dist in
    // the datasets/colmap.py Parser × the 1.1 multiplier in simple_trainer.py) — this makes the means
    // LR consistent with gsplat.
    double max_dist = 0;
    for (size_t c = 0; c < n; c++) {
        const double dx = pos[c * 3] - cx, dy = pos[c * 3 + 1] - cy, dz = pos[c * 3 + 2] - cz;
        max_dist = std::max(max_dist, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    return max_dist > 0.0 ? static_cast<float>(max_dist * 1.1) : 1.0f; // guards the degenerate case (all centers coincide)
}

float means_lr_at(const TrainConfig& cfg, int iter) {
    const float p = cfg.total_iters > 0
                        ? static_cast<float>(iter) / static_cast<float>(cfg.total_iters)
                        : 0.0f;
    return cfg.lr_means * cfg.scene_scale * std::pow(0.01f, p); // 1% at progress fraction 1 (design §4)
}

uint32_t sh_degree_at(const TrainConfig& cfg, int iter, int sh_degree) {
    if (sh_degree <= 0 || iter < 0) return 0;
    const int interval = std::max(
        1, static_cast<int>(static_cast<float>(cfg.total_iters) * cfg.sh_degree_interval_ratio));
    return static_cast<uint32_t>(std::min(iter / interval, sh_degree));
}

void chain_scale_grads_inplace(std::vector<float>& v_scales, const float* scales_linear) {
    for (size_t i = 0; i < v_scales.size(); i++) v_scales[i] *= scales_linear[i];
}

void chain_opacity_grads_inplace(std::vector<float>& v_opacities, const float* opacs_act) {
    for (size_t i = 0; i < v_opacities.size(); i++) {
        const float s = opacs_act[i];
        v_opacities[i] *= s * (1.0f - s);
    }
}

// AA product-rule chain (Phase 3 Task 4): opacity_aa = σ·comp — the analytic derivative matching
// autograd. v_comp = v_op_aa·σ (returned for injection into proj bwd), and v_opacities becomes
// v_op_aa·comp in place (on a post-sigmoid basis — chain_opacity_grads_inplace then multiplies by σ′
// to complete the logit basis: d loss/d logit = v_op_aa·comp·σ(1−σ), d loss/d comp = v_op_aa·σ —
// the chain formula from the brief).
std::vector<float> chain_aa_opacity_grads_inplace(std::vector<float>& v_opacities,
                                                  const float* opacs_act,
                                                  const float* compensations) {
    std::vector<float> v_comp(v_opacities.size());
    for (size_t i = 0; i < v_opacities.size(); i++) {
        const float v_op_aa = v_opacities[i];
        v_comp[i] = v_op_aa * opacs_act[i];
        v_opacities[i] = v_op_aa * compensations[i];
    }
    return v_comp;
}

// ===== Regularization gradients (Phase 2 Task 4 — the analytic derivative of gsplat
// losses.py:675–700) =====
// d/dlogitᵢ [reg·meanₙ(σ(logit))] = reg·σᵢ(1−σᵢ)/N — opacities is 1D [N] (denominator N).
void add_opacity_reg_grad_inplace(std::vector<float>& v_opacities, const float* opacs_act,
                                  float reg) {
    if (!(reg > 0.0f) || v_opacities.empty()) return; // the gsplat `> 0.0` gate (:993)
    const float w = reg / static_cast<float>(v_opacities.size());
    for (size_t i = 0; i < v_opacities.size(); i++) {
        const float s = opacs_act[i];
        v_opacities[i] += w * s * (1.0f - s);
    }
}

// d/dlog_s[i,k] [reg·mean_{N·3}(exp(log_s))] = reg·exp(log_s[i,k])/(N·3) — torch .mean() averages
// over **every element** of [N,3], so the denominator is N·3 (= v_scales.size()), not N
// (losses.py:689–700).
void add_scale_reg_grad_inplace(std::vector<float>& v_scales, const float* scales_lin,
                                float reg) {
    if (!(reg > 0.0f) || v_scales.empty()) return; // the gsplat `> 0.0` gate (:995)
    const float w = reg / static_cast<float>(v_scales.size());
    for (size_t i = 0; i < v_scales.size(); i++) v_scales[i] += w * scales_lin[i];
}

void add_shape_reg_grad_inplace(std::vector<float>& v_scales, const float* scales_lin,
                                float flatten_w, float needle_w, float needle_ratio) {
    if (!(flatten_w > 0.0f) && !(needle_w > 0.0f)) return;
    const size_t n = v_scales.size() / 3;
    for (size_t g = 0; g < n; g++) {
        const float s[3] = {scales_lin[g * 3], scales_lin[g * 3 + 1], scales_lin[g * 3 + 2]};
        size_t i1 = 0, i2 = 1, i3 = 2; // s[i1] ≥ s[i2] ≥ s[i3], ties keep axis order
        if (s[i1] < s[i2]) std::swap(i1, i2);
        if (s[i2] < s[i3]) std::swap(i2, i3);
        if (s[i1] < s[i2]) std::swap(i1, i2);
        if (flatten_w > 0.0f) v_scales[g * 3 + i3] += flatten_w * s[i3];
        if (needle_w > 0.0f && s[i2] > 0.0f) {
            const float r = s[i1] / s[i2];
            if (r > needle_ratio) {
                v_scales[g * 3 + i1] += needle_w * r;
                v_scales[g * 3 + i2] -= needle_w * r;
            }
        }
    }
}

// MCMC presets (the "mcmc" config in gsplat examples/simple_trainer.py:1579–1587):
// reproduces Config(init_opa=0.5, init_scale=0.1, opacity_reg=0.01, scale_reg=0.01,
// strategy=MCMCStrategy(...)). No other field is touched.
void apply_mcmc_presets(InitOptions& io, TrainConfig& cfg) {
    io.init_opacity = 0.5f; // :1583 (logit(0.5) = 0 — storing in logit-space is init_splats' responsibility)
    io.init_scale = 0.1f;   // :1584 — a multiplier on the mean 3-NN distance (not a fixed scale)
    cfg.opacity_reg = 0.01f; // :1585
    cfg.scale_reg = 0.01f;   // :1586
    cfg.mcmc.enabled = true; // :1587 matching strategy=MCMCStrategy
}

// Forward intermediates — bwd MUST consume the same buffers as fwd (activations, flatten_ids,
// tile_offsets) for the gradients to agree (the D7 order contract). This struct owns the vectors —
// it is the single source of truth behind the downstream pointer references.
struct Trainer::Forward {
    std::vector<float> scales_lin;  // [N,3] exp(scales)
    std::vector<float> opacs_act;   // [N] sigmoid(opacities)
    std::vector<uint8_t> valid;     // [N] (radii>0).all(-1) — shared by sh masks and adam valid
    std::vector<float> dirs;        // [N,3] means − campos
    std::vector<float> coeffs;      // [N,K,3] assembled from sh0+shN
    std::vector<float> colors;      // [N,3] after the SH post-op (+0.5, clamp)
    std::vector<uint8_t> relu_mask; // [N,3] per-channel pre_clamp > 0 — masks v_colors in bwd
    std::vector<float> opacs_aa;    // [N] sigmoid(op)·comp — only when antialiased (empty otherwise)
    ProjectionFusedOutputs proj;
    IsectTilesOutputs isect;
    SortIsectOutputs sorted;
    std::vector<int32_t> offsets;   // [1,TH,TW]
    RasterizeFwdOutputs raster;
    uint32_t tile_width = 0, tile_height = 0;
    uint32_t degrees_to_use = 0;
    int64_t K_total = 1;            // (sh_degree+1)^2

    // The opacity that isect and raster consume — with AA it is opacity_aa (gsplat
    // rendering.py:864–865 runs before isect_tiles (:884–896): both the AccuTile level set and
    // raster work from the post-AA opacity).
    const std::vector<float>& raster_opacs() const {
        return opacs_aa.empty() ? opacs_act : opacs_aa;
    }
};

namespace {

ProjectionFusedInputs make_proj_inputs(const SplatCloud& c, const TrainingCamera& cam,
                                       const TrainConfig& cfg,
                                       const std::vector<float>& scales_lin,
                                       const std::vector<float>& opacs_act,
                                       const float* viewmat) {
    ProjectionFusedInputs p;
    p.means = c.means.data();
    p.quats = c.quats.data();   // pre-activation (the kernel normalizes internally — 01 Correction 6)
    p.scales = scales_lin.data();
    p.opacities = opacs_act.data(); // fwd-only opacity-aware radius (bwd ignores it — 01 Correction 1)
    p.viewmats = viewmat;           // the injection point for refined poses (nullptr is not allowed — the caller guarantees this)
    p.Ks = cam.K;
    p.N = c.count;
    p.C = 1;
    p.width = cam.width;
    p.height = cam.height;
    p.eps2d = cfg.eps2d;
    p.near_plane = cfg.near_plane;
    p.far_plane = cfg.far_plane;
    p.radius_clip = cfg.radius_clip;
    p.camera_model = 0;              // PINHOLE (Phase 1)
    // With antialiased it outputs comp (matching gsplat rendering.py:834/:856 calc_compensations=
    // (rasterize_mode == "antialiased")). false = classic mode (D10-1 — the previous behavior).
    p.calc_compensations = cfg.antialiased;
    return p;
}

ShInputs make_sh_inputs_impl(const std::vector<float>& dirs, const std::vector<float>& coeffs,
                             const std::vector<uint8_t>& valid, uint32_t degrees_to_use,
                             int64_t N, int64_t K) {
    ShInputs s;
    s.dirs = dirs.data();
    s.coeffs = coeffs.data();
    s.masks = valid.data(); // D7: masks = (radii>0).all(-1)
    s.degrees_to_use = degrees_to_use;
    s.B = 1;
    s.N = N;
    s.K = K;
    s.D = 3;
    return s;
}

RasterizeFwdInputs make_raster_inputs(const Trainer::Forward& f, const TrainingCamera& cam,
                                      int64_t N, bool with_depth = false) {
    RasterizeFwdInputs r;
    if (with_depth) r.depths = f.proj.depths.data(); // the projection's camera z, per Gaussian
    r.means2d = f.proj.means2d.data();
    r.conics = f.proj.conics.data();
    r.colors = f.colors.data();
    r.opacities = f.raster_opacs().data(); // with AA, opacity_aa (after applying gsplat :864–865)
    r.backgrounds = nullptr; // no background (black) — the gsplat default
    r.isect_offsets = f.offsets.data();
    r.flatten_ids = f.sorted.values_sorted.data();
    r.I = 1;
    r.N = N;
    r.n_isects = f.isect.n_isects;
    r.image_width = static_cast<uint32_t>(cam.width);
    r.image_height = static_cast<uint32_t>(cam.height);
    r.tile_size = 16;
    r.tile_width = f.tile_width;
    r.tile_height = f.tile_height;
    return r;
}

} // namespace

Trainer::Trainer(KernelBackend& backend, SplatCloud cloud, const TrainConfig& cfg)
    : backend_(backend), cloud_(std::move(cloud)), cfg_(cfg) {
    if (cloud_.count < 1) throw std::invalid_argument("Trainer: empty cloud");
    if (cfg_.total_iters < 1) throw std::invalid_argument("Trainer: total_iters must be >= 1");
    // λ ∈ [0,1] (rejects NaN — a negated-form guard). gsplat uses ssim_lambda=0.2; 0 keeps backward
    // compatibility with pure L1.
    if (!(cfg_.ssim_weight >= 0.0f && cfg_.ssim_weight <= 1.0f))
        throw std::invalid_argument("Trainer: ssim_weight must be in [0,1]");
    if (!(cfg_.depth_weight >= 0.0f && std::isfinite(cfg_.depth_weight)))
        throw std::invalid_argument("Trainer: depth_weight must be finite and >= 0");
    if (!(cfg_.flatten_weight >= 0.0f && std::isfinite(cfg_.flatten_weight)))
        throw std::invalid_argument("Trainer: flatten_weight must be finite and >= 0");
    if (!(cfg_.needle_weight >= 0.0f && std::isfinite(cfg_.needle_weight)))
        throw std::invalid_argument("Trainer: needle_weight must be finite and >= 0");
    if (!(cfg_.needle_ratio >= 1.0f && std::isfinite(cfg_.needle_ratio)))
        throw std::invalid_argument("Trainer: needle_ratio must be finite and >= 1");
    arena_rows_ = cloud_.count; // mcmc off — the same fixed N as 1c
    if (!cfg_.mcmc.enabled) return;

    const McmcOptions& m = cfg_.mcmc;
    if (m.cap_max < cloud_.count)
        throw std::invalid_argument("Trainer: mcmc.cap_max must be >= initial splat count");
    if (!(m.min_opacity > 0.0f && m.min_opacity < 1.0f))
        throw std::invalid_argument("Trainer: mcmc.min_opacity must be in (0,1)"); // the logit domain
    if (!(m.refine_every_ratio > 0.0f))
        throw std::invalid_argument("Trainer: mcmc.refine_every_ratio must be > 0");

    // Fixed cap_max arena (design §5.2 — it avoids the gsplat #444 OOM structure): the vectors are
    // preallocated once at cap rows and MUST NOT be reallocated afterwards — only count (= alive)
    // grows. The tail slots are 0 (the untouched contract: the kernel dispatches, the gradients, and
    // the adam n_elems all work from count).
    arena_rows_ = m.cap_max;
    const size_t cap = static_cast<size_t>(m.cap_max);
    cloud_.means.resize(cap * 3, 0.0f);
    cloud_.scales.resize(cap * 3, 0.0f);
    cloud_.quats.resize(cap * 4, 0.0f);
    cloud_.opacities.resize(cap, 0.0f);
    cloud_.sh0.resize(cap * 3, 0.0f);
    cloud_.shN.resize(cap * static_cast<size_t>(cloud_.sh_rest_coeffs()) * 3, 0.0f);
    // The Adam moments are preallocated as an arena too — so state surgery is defined even for a
    // refine that happens before the first adam step
    const auto alloc = [](AdamState& st, size_t n) {
        st.exp_avg.assign(n, 0.0f);
        st.exp_avg_sq.assign(n, 0.0f);
    };
    alloc(means_st_, cloud_.means.size());
    alloc(quats_st_, cloud_.quats.size());
    alloc(scales_st_, cloud_.scales.size());
    alloc(opac_st_, cloud_.opacities.size());
    alloc(sh0_st_, cloud_.sh0.size());
    alloc(shN_st_, cloud_.shN.size());

    binoms_ = make_binoms_lut(kMcmcNMax); // the Eq.9 LUT (mcmc.py:95–100 — the McmcTests bit contract)
    mcmc_rng_.seed(m.seed);
}

Trainer::Forward Trainer::forward_pass(const TrainingCamera& cam, uint32_t degrees_to_use,
                                       bool with_depth) {
    const int64_t N = cloud_.count;
    if (cam.width < 1 || cam.height < 1)
        throw std::invalid_argument("Trainer: camera width/height must be positive");
    const float* vm = cam.viewmat;

    Forward f;
    f.degrees_to_use = degrees_to_use;
    f.K_total = static_cast<int64_t>(cloud_.sh_degree + 1) * (cloud_.sh_degree + 1);

    // (host) Activation — the kernel input convention: linear scales (exp) and post-sigmoid opacities
    f.scales_lin = activate_scales(cloud_.scales.data(), N * 3);
    f.opacs_act = activate_opacities(cloud_.opacities.data(), N);

    // [01] projection fwd (with antialiased, calc_compensations=true — comp is output alongside)
    const ProjectionFusedInputs pin =
        make_proj_inputs(cloud_, cam, cfg_, f.scales_lin, f.opacs_act, vm);
    f.proj = backend_.projection_fused_fwd(pin);

    // (host) Applies the AA compensation (Phase 3 Task 4 — gsplat rendering.py:864–865 `opacities =
    // opacities * compensations` runs **before** isect_tiles (:884–896)): opacity_aa then feeds both
    // isect (the AccuTile level set) and raster. For the opacity-aware radius inside proj fwd, the
    // kernel applies the same product itself (kProjCalcComp — ProjectionEWA.metal:138, same as the
    // upstream original). Culled slots have comp = 0 (zero-initialized — matching at::zeros) →
    // opacity_aa = 0.
    if (cfg_.antialiased) {
        f.opacs_aa.resize(static_cast<size_t>(N));
        for (int64_t i = 0; i < N; i++)
            f.opacs_aa[i] = f.opacs_act[i] * f.proj.compensations[i];
    }

    // (host) valid = (radii>0).all(-1) — shared by the sh masks, the adam valid, and the culling gate (D6/D7)
    f.valid.resize(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; i++)
        f.valid[i] = (f.proj.radii[i * 2] > 0 && f.proj.radii[i * 2 + 1] > 0) ? 1 : 0;

    // (host) dirs = means − campos, campos = -Rᵀt
    float campos[3];
    for (int i = 0; i < 3; i++)
        campos[i] = -(vm[0 * 4 + i] * vm[0 * 4 + 3] + vm[1 * 4 + i] * vm[1 * 4 + 3] +
                      vm[2 * 4 + i] * vm[2 * 4 + 3]);
    f.dirs.resize(static_cast<size_t>(N) * 3);
    for (int64_t i = 0; i < N; i++)
        for (int k = 0; k < 3; k++) f.dirs[i * 3 + k] = cloud_.means[i * 3 + k] - campos[k];

    // (host) Assembles coeffs [N,K,3] — k=0 is sh0, k>=1 is shN (the inverse of SplatCloud's split storage)
    f.coeffs.resize(static_cast<size_t>(N) * f.K_total * 3);
    for (int64_t i = 0; i < N; i++) {
        for (int c = 0; c < 3; c++) f.coeffs[(i * f.K_total) * 3 + c] = cloud_.sh0[i * 3 + c];
        for (int64_t k = 1; k < f.K_total; k++)
            for (int c = 0; c < 3; c++)
                f.coeffs[(i * f.K_total + k) * 3 + c] =
                    cloud_.shN[(i * (f.K_total - 1) + (k - 1)) * 3 + c];
    }

    // [02] SH fwd + (host) post-op: +0.5, a [0,∞) clamp, and storing a per-channel relu_mask
    // (pre_clamp > 0 — the backward semantics of gsplat rendering.py clamp_min(0): a channel where
    // the clamp fires has subgradient 0. The convention of keeping the post-op outside the kernel is
    // D6/D7). Masked-out slots (valid=0) are unwritten by SH (garbage — disposition #5), but raster
    // never reads culled Gaussians, and a relu_mask built from garbage/NaN is harmless because
    // v_colors for that slot is 0 (for NaN, pre > 0 == false → mask 0).
    {
        const ShInputs sin =
            make_sh_inputs_impl(f.dirs, f.coeffs, f.valid, degrees_to_use, N, f.K_total);
        ShOutputs sh = backend_.sh_fwd(sin);
        f.colors = std::move(sh.colors);
        f.relu_mask.resize(f.colors.size());
        for (size_t i = 0; i < f.colors.size(); i++) {
            const float pre = f.colors[i] + 0.5f;
            f.relu_mask[i] = pre > 0.0f ? 1 : 0;
            f.colors[i] = std::max(pre, 0.0f);
        }
    }

    // [03] isect (AccuTile — conics+opacities, the gsplat production default)
    f.tile_width = static_cast<uint32_t>((cam.width + 15) / 16);
    f.tile_height = static_cast<uint32_t>((cam.height + 15) / 16);
    {
        IsectTilesInputs iin;
        iin.means2d = f.proj.means2d.data();
        iin.radii = f.proj.radii.data();
        iin.depths = f.proj.depths.data();
        iin.conics = f.proj.conics.data();
        iin.opacities = f.raster_opacs().data(); // with AA, opacity_aa — the level set is AA-based too (:896)
        iin.I = 1;
        iin.N = N;
        iin.tile_size = 16;
        iin.tile_width = f.tile_width;
        iin.tile_height = f.tile_height;
        f.isect = backend_.isect_tiles(iin);
    }

    // [04] sort (stable — the valid bits are 32 + tile + image)
    {
        SortIsectInputs sin;
        sin.keys = f.isect.isect_ids.data();
        sin.values = f.isect.flatten_ids.data();
        sin.n = f.isect.n_isects;
        sin.key_n_bits = 32 + f.isect.tile_n_bits + f.isect.image_n_bits;
        f.sorted = backend_.sort_isect(sin);
    }

    // [03] isect_offset_encode
    {
        IsectOffsetInputs oin;
        oin.isect_ids = f.sorted.keys_sorted.data();
        oin.n_isects = f.isect.n_isects;
        oin.I = 1;
        oin.tile_width = f.tile_width;
        oin.tile_height = f.tile_height;
        f.offsets = backend_.isect_offset_encode(oin);
    }

    // [05] rasterize fwd
    f.raster = backend_.rasterize_fwd(make_raster_inputs(f, cam, N, with_depth));
    return f;
}

// ===== The device-resident path (Phase 3 Task 5 — D11/00-index §7) =====
// It keeps the parameters, moments, and intermediate buffers resident on the GPU and delegates the
// D7 step to the backend executor. The assembly order and the formulas are identical to the host
// step() below (the TrainStep.metal kernels reproduce the host computation — the cross-check gate is
// TrainerTests). What stays on the host: the schedules (lr and sh degree — step arguments), the MCMC
// refine (low frequency — see the synchronization convention below), eval/logging, and the GT upload.
//
// Synchronization convention (the state machine of where the authoritative copy lives):
// - Before a resident step, upload if device_authoritative_ is not set (the first time uploads the
//   whole arena — establishing the zero tail; afterwards only count rows). After the step,
//   host_fresh_ = false (the host mirror is stale).
// - Reads (cloud()/observing loss) download lazily (sync_host_from_device — Home is unchanged).
// - Mutations (mcmc_refine/mcmc_inject_noise/adam_state) download and then move Home to the host
//   (sync_host_for_mutation) — the next resident step re-uploads. Noise injection on a refine step is
//   host-owned as well (it preserves the adam → refine → noise order — reproducing gsplat
//   step_post_backward). Noise on a non-refine step is encoded on the resident path (the same
//   rng_normal/perturb kernels — Philox is deterministic from (seed, offset=iter) alone, so the
//   stream is bit-identical to the host path).

bool Trainer::resident_ready() {
    if (!cfg_.resident) return false;
    if (exec_) return true;
    if (exec_unavailable_) return false;
    TrainStepExecutorDesc d;
    d.arena_rows = arena_rows_;
    d.sh_degree = cloud_.sh_degree;
    d.eps2d = cfg_.eps2d;
    d.near_plane = cfg_.near_plane;
    d.far_plane = cfg_.far_plane;
    d.radius_clip = cfg_.radius_clip;
    d.b1 = cfg_.adam_b1;
    d.b2 = cfg_.adam_b2;
    d.eps = cfg_.adam_eps;
    d.ssim_weight = cfg_.ssim_weight;
    d.antialiased = cfg_.antialiased;
    exec_ = backend_.make_train_step_executor(d);
    if (!exec_) exec_unavailable_ = true; // an unsupported backend (CPU, etc.) — pinned to the host fallback
    return exec_ != nullptr;
}

ResidentParamSync Trainer::make_sync(int64_t rows) {
    std::vector<float>* params[6] = {&cloud_.means, &cloud_.quats,     &cloud_.scales,
                                     &cloud_.opacities, &cloud_.sh0,   &cloud_.shN};
    AdamState* sts[6] = {&means_st_, &quats_st_, &scales_st_, &opac_st_, &sh0_st_, &shN_st_};
    ResidentParamSync s;
    s.rows = rows;
    for (int g = 0; g < 6; g++) {
        if (params[g]->empty()) continue; // shN at sh_degree 0 — the group does not exist
        // Lazy zero-allocation of the moments (the same meaning as the lazy assign in the host adam_step)
        if (sts[g]->exp_avg.empty()) {
            sts[g]->exp_avg.assign(params[g]->size(), 0.0f);
            sts[g]->exp_avg_sq.assign(params[g]->size(), 0.0f);
        }
        s.params[g] = params[g]->data();
        s.exp_avg[g] = sts[g]->exp_avg.data();
        s.exp_avg_sq[g] = sts[g]->exp_avg_sq.data();
    }
    return s;
}

void Trainer::resident_upload() {
    // The first upload covers the whole arena (it establishes the zero tail on the device — the MCMC
    // arena contract); afterwards only count rows (both device and host leave everything past count
    // untouched, so that is sufficient — growth from a refine is covered by the new count).
    const int64_t rows = exec_uploaded_once_ ? cloud_.count : arena_rows_;
    exec_->upload_params(make_sync(rows));
    exec_uploaded_once_ = true;
    device_authoritative_ = true;
    host_fresh_ = true;
}

void Trainer::sync_host_from_device() {
    if (!device_authoritative_ || host_fresh_) return;
    exec_->download_params(make_sync(cloud_.count));
    host_fresh_ = true;
}

void Trainer::sync_host_for_mutation() {
    sync_host_from_device();
    device_authoritative_ = false; // host mutation follows — the next resident step re-uploads
}

TrainStepResult Trainer::step_resident(const TrainingCamera& cam, const float* gt_rgb, int iter,
                                       const DepthTarget* depth, const RegionMask* mask,
                                       int64_t mask_count, int64_t mask_ssim_count,
                                       int64_t depth_valid, bool want_loss) {
    if (cam.width < 1 || cam.height < 1)
        throw std::invalid_argument("Trainer: camera width/height must be positive");
    if (!device_authoritative_) resident_upload();

    ResidentStepInputs in;
    in.viewmat = cam.viewmat;
    in.K = cam.K;
    in.width = cam.width;
    in.height = cam.height;
    in.gt_rgb = gt_rgb;
    in.count = cloud_.count;
    in.degrees_to_use = sh_degree_at(cfg_, iter, cloud_.sh_degree);
    in.lr[0] = means_lr_at(cfg_, iter); // D3 group order (the same lr as the 6 host adam calls)
    in.lr[1] = cfg_.lr_quats;
    in.lr[2] = cfg_.lr_scales;
    in.lr[3] = cfg_.lr_opacities;
    in.lr[4] = cfg_.lr_sh0;
    in.lr[5] = cfg_.lr_shN;
    // Regularization weights — reproduces the w expression of add_*_reg_grad_inplace (the > 0 gate +
    // the mean denominators N and N·3)
    in.opacity_reg_w = cfg_.opacity_reg > 0.0f
                           ? cfg_.opacity_reg / static_cast<float>(cloud_.count)
                           : 0.0f;
    in.scale_reg_w = cfg_.scale_reg > 0.0f
                         ? cfg_.scale_reg / static_cast<float>(cloud_.count * 3)
                         : 0.0f;
    in.want_loss = want_loss;
    in.want_render = true; // upholds the TrainStepResult.render_colors contract (the input for --save-preview)
    if (depth) {
        in.gt_depth = depth->depth_m.data();
        in.depth_width = depth->width;
        in.depth_height = depth->height;
        in.depth_valid = depth_valid;
        in.depth_weight = cfg_.depth_weight;
    }
    if (mask) {
        in.mask = mask;
        in.mask_count = mask_count;
        in.mask_ssim_count = mask_ssim_count;
    }
    // Shape regulariser — the w expression of add_shape_reg_grad_inplace (weight/N, 0 = off)
    in.flatten_w = cfg_.flatten_weight > 0.0f
                       ? cfg_.flatten_weight / static_cast<float>(cloud_.count)
                       : 0.0f;
    in.needle_w = cfg_.needle_weight > 0.0f
                      ? cfg_.needle_weight / static_cast<float>(cloud_.count)
                      : 0.0f;
    in.needle_ratio = cfg_.needle_ratio;
    const bool refine_due = cfg_.mcmc.enabled && mcmc_refine_due(iter);
    if (cfg_.mcmc.enabled && !refine_due) {
        in.inject_noise = true;
        in.noise_seed = cfg_.mcmc.seed;
        in.noise_offset = static_cast<uint64_t>(iter);
        in.noise_scale = means_lr_at(cfg_, iter) * cfg_.mcmc.noise_lr; // mcmc.py:190
    }

    ResidentStepOutputs ro = exec_->step(in);
    host_fresh_ = false; // adam (+noise) updated the device parameters — the mirror is stale

    TrainStepResult r;
    if (want_loss) {
        last_loss_ = ro.loss;
        last_depth_loss_ = ro.depth_loss;
    }
    r.loss = last_loss_;
    r.depth_loss = last_depth_loss_;
    r.loss_fresh = want_loss;
    r.n_isects = ro.n_isects;
    r.render_colors = std::move(ro.render_colors);
    if (refine_due) {
        // Refine step: reuses the host orchestration — mcmc_refine()/mcmc_inject_noise() reclaim the
        // parameters and moments through sync_host_for_mutation() and move Home to the host.
        const McmcRefineResult rr = mcmc_refine();
        r.mcmc_refined = true;
        r.n_relocated = static_cast<int64_t>(rr.relocated_dead.size());
        r.n_added = static_cast<int64_t>(rr.added_src.size());
        mcmc_inject_noise(iter); // noise after refine (the same order as the host path, reflecting the new count)
    }
    return r;
}

TrainStepResult Trainer::step(const TrainingCamera& cam, const float* gt_rgb, int iter,
                              const DepthTarget* depth, const RegionMask* mask,
                              bool want_loss) {
    if (!gt_rgb) throw std::invalid_argument("Trainer::step: null gt_rgb");
    if (depth && depth->empty()) depth = nullptr;
    if (cfg_.depth_weight <= 0.0f) depth = nullptr; // off — the RGB-only step, unchanged
    if (mask && mask->empty()) mask = nullptr;      // no mask — the unmasked step, unchanged
    if (mask && depth) check_mask_matches_depth(*mask, *depth);
    const int64_t mask_count =
        mask ? (mask->view_count >= 0 ? mask->view_count
                                      : mask_view_count(*mask, cam.width, cam.height))
             : 0;
    const int64_t mask_ssim =
        mask ? (mask->ssim_count >= 0 ? mask->ssim_count
                                      : mask_ssim_count(*mask, cam.width, cam.height))
             : 0;
    const int64_t depth_valid =
        depth ? (depth->view_valid >= 0 ? depth->view_valid
                                        : depth_valid_count(*depth, cam.width, cam.height, mask))
              : 0;
    if (resident_ready())
        return step_resident(cam, gt_rgb, iter, depth, mask, mask_count, mask_ssim, depth_valid,
                             want_loss);
    const int64_t N = cloud_.count;
    const bool use_depth = depth && depth_valid > 0;

    Forward f = forward_pass(cam, sh_degree_at(cfg_, iter, cloud_.sh_degree), use_depth);

    const float* loss_img = f.raster.render_colors.data();

    // (host) L1 loss + upstream grad. v_render_alphas is a zero buffer (neither L1 nor SSIM uses alpha).
    const int64_t hw = static_cast<int64_t>(cam.width) * cam.height;
    L1LossResult l1 = mask ? l1_loss_and_grad_masked(loss_img, gt_rgb, cam.width, cam.height,
                                                     *mask, mask_count)
                           : l1_loss_and_grad(loss_img, gt_rgb, hw * 3);
    float loss = l1.loss;

    // (kernel + host) Combines the SSIM loss (Phase 3 Task 3 — reproducing gsplat
    // simple_trainer.py:961 `torch.lerp(l1loss, 1-ssimloss, λ)` with λ=ssim_lambda=0.2):
    //   loss     = (1-λ)·L1 + λ·(1-ssim)
    //   v_render = (1-λ)·v_L1 + λ·v_DSSIM   — the λ·v_DSSIM term is absorbed by ssim_bwd(v_loss=λ)
    //   into the kernel's G = -λ/M and returned as is (the disposition of 08 open question 4 — it is
    //   not combined afterwards on the host).
    // img1 = render (the differentiated input), img2 = GT (v_img2 is unimplemented — 08 §2; the
    // gradient goes to render only, confirmed in Task 1). render goes in without a [0,1] clamp — the
    // same condition as the gsplat loss path (task-1-report §4). At λ=0 the SSIM kernels are bypassed
    // entirely — bit-identical to the previous pure-L1 behavior.
    if (cfg_.ssim_weight > 0.0f) {
        const float lam = cfg_.ssim_weight;
        SsimInputs si;
        // [1,H,W,3] channel-last (the D6 convention — no transpose).
        si.img1 = loss_img;
        si.img2 = gt_rgb;
        si.I = 1;
        si.H = cam.height;
        si.W = cam.width;
        si.C = 3;
        si.need_coeffs = true; // training mode — also outputs the 3 coefficient maps (Dμ/Dσ/Dc) that bwd takes as input
        SsimFwdOutputs so = backend_.ssim_fwd(si); // H/W < 11 makes the validator throw
        double ssim = so.ssim;
        float v_loss = lam; // upstream d(total)/d(1-ssim) = λ
        if (mask) {
            // Masked: dL/dS(q) × m(centre), and ssim_bwd's G = -v_loss/map_count becomes -λ/(3·Ms).
            mask_ssim_coeffs_inplace(so.coeff_mu.data(), so.coeff_sigma.data(),
                                     so.coeff_cross.data(), *mask, cam.width, cam.height, 3);
            const double map_count = static_cast<double>(so.ssim_map.size());
            v_loss = mask_ssim > 0 ? static_cast<float>(static_cast<double>(lam) * map_count /
                                                        (3.0 * static_cast<double>(mask_ssim)))
                                   : 0.0f;
            ssim = masked_ssim_mean(so.ssim_map.data(), *mask, cam.width, cam.height, 3, mask_ssim);
        }
        SsimGradsIn sgi;
        sgi.coeff_mu = so.coeff_mu.data();
        sgi.coeff_sigma = so.coeff_sigma.data();
        sgi.coeff_cross = so.coeff_cross.data();
        sgi.v_loss = v_loss;
        const std::vector<float> v_dssim = backend_.ssim_bwd(si, sgi);
        const float w_l1 = 1.0f - lam;
        for (size_t i = 0; i < l1.v_render_colors.size(); i++)
            l1.v_render_colors[i] = w_l1 * l1.v_render_colors[i] + v_dssim[i];
        loss = w_l1 * l1.loss + lam * (1.0f - static_cast<float>(ssim));
    }
    std::vector<float> v_render_alphas(static_cast<size_t>(hw), 0.0f);
    // Depth supervision: the inverse-depth L1 over the expected depth enters through the Σwz
    // channel and, by the α normalisation, through v_render_alphas.
    DepthLossResult dl;
    if (use_depth) {
        dl = depth_loss_and_grad(f.raster.render_depths.data(), f.raster.render_alphas.data(),
                                 *depth, cam.width, cam.height, depth_valid, cfg_.depth_weight,
                                 mask);
        v_render_alphas = dl.v_render_alphas;
    }

    // [05] rasterize bwd — reuses the same buffers as fwd (activations, flatten_ids, offsets) (the D7
    // contract). v_render_colors is the total gradient combined above (at λ=0 it is pure v_L1).
    RasterizeBwdGradsIn gin;
    gin.render_alphas = f.raster.render_alphas.data();
    gin.last_ids = f.raster.last_ids.data();
    gin.v_render_colors = l1.v_render_colors.data();
    gin.v_render_alphas = v_render_alphas.data();
    if (use_depth) gin.v_render_depths = dl.v_render_depths.data();
    RasterizeBwdGrads rg = backend_.rasterize_bwd(make_raster_inputs(f, cam, N, use_depth), gin);

    // [02] SH bwd — applies relu_mask: channels where the clamp fired (pre_clamp <= 0) have their
    // gradient blocked (torch clamp subgradient 0). Because the block happens at the **input stage**
    // of sh_bwd, both v_coeffs and v_dirs (→ accumulated into v_means) are masked automatically.
    if (rg.v_colors.size() != f.relu_mask.size())
        throw std::logic_error("Trainer::step: v_colors/relu_mask size mismatch");
    for (size_t i = 0; i < rg.v_colors.size(); i++)
        if (!f.relu_mask[i]) rg.v_colors[i] = 0.0f;
    ShInputs sin = make_sh_inputs_impl(f.dirs, f.coeffs, f.valid, f.degrees_to_use, N, f.K_total);
    sin.compute_v_dirs = true;
    const ShGrads sg = backend_.sh_bwd(sin, rg.v_colors.data());

    // [01] projection bwd (v_depths is a zero buffer unless the step is depth-supervised). It MUST
    // be called with the same viewmat as fwd (cam.viewmat) for the gradients to be consistent.
    const std::vector<float> v_depths =
        use_depth ? rg.v_depths : std::vector<float>(static_cast<size_t>(N), 0.0f);
    const ProjectionFusedInputs pin =
        make_proj_inputs(cloud_, cam, cfg_, f.scales_lin, f.opacs_act, cam.viewmat);
    ProjectionFusedGradsIn pgin;
    pgin.radii = f.proj.radii.data();
    pgin.conics = f.proj.conics.data();
    pgin.v_means2d = rg.v_means2d.data();
    pgin.v_depths = v_depths.data();
    pgin.v_conics = rg.v_conics.data();
    // (host) AA bwd product-rule chain (Phase 3 Task 4 — 01 spec gap #1 is host-owned):
    // rg.v_opacities from raster bwd is d loss/d opacity_aa (fwd consumed opacity_aa).
    //   v_comp       = v_op_aa·σ(op)  → injected into proj bwd as v_compensations (kProjCalcComp —
    //                                   it activates the tolerance verification path exclusive to
    //                                   the 1a c2 fixture)
    //   v_opacities  = v_op_aa·comp   (in place — on a post-sigmoid basis, connected to the σ′ chain
    //                                   below: d/d logit = v_op_aa·comp·σ(1−σ))
    // In gsplat the :864–865 multiply is in the autograd graph and therefore automatic — here the
    // analytic derivative reproduces it.
    std::vector<float> v_comp;
    if (cfg_.antialiased) {
        v_comp = chain_aa_opacity_grads_inplace(rg.v_opacities, f.opacs_act.data(),
                                                f.proj.compensations.data());
        pgin.compensations = f.proj.compensations.data();
        pgin.v_compensations = v_comp.data();
    }
    ProjectionFusedGrads pg = backend_.projection_fused_bwd(pin, pgin);

    // (host) v_means accumulation: folds in v_dirs from sh_bwd (dirs = means − campos → ∂dirs/∂means = I)
    for (int64_t i = 0; i < N * 3; i++) pg.v_means[i] += sg.v_dirs[i];

    // (host) Activation gradient chain — task-6 handover 3
    chain_scale_grads_inplace(pg.v_scales, f.scales_lin.data());
    chain_opacity_grads_inplace(rg.v_opacities, f.opacs_act.data());

    // (host) Adds the regularization gradients (Phase 2 Task 4 — gsplat simple_trainer.py:993–996):
    // the analytic derivative of loss += reg·mean(sigmoid(op)) + reg·mean(exp(scales)) is added
    // directly to the internal-representation (logit/log) gradients right before adam — this comes
    // after the chain, so σ′/exp are not applied again (it is already on a pre-activation basis). The
    // MCMC preset sets 0.01; at the default 0.0 it is a no-op (the same gate as the upstream
    // original). The reported loss (r.loss) is the combined loss (1-λ)·L1 + λ·DSSIM (Phase 3 Task 3 —
    // at λ=0 it is L1), and the reg terms are NOT included in the reported loss — they show up only
    // as a training signal (gradients) (the Phase 2 Task 4 decision stands).
    add_opacity_reg_grad_inplace(rg.v_opacities, f.opacs_act.data(), cfg_.opacity_reg);
    add_scale_reg_grad_inplace(pg.v_scales, f.scales_lin.data(), cfg_.scale_reg);
    add_shape_reg_grad_inplace(
        pg.v_scales, f.scales_lin.data(),
        cfg_.flatten_weight > 0.0f ? cfg_.flatten_weight / static_cast<float>(N) : 0.0f,
        cfg_.needle_weight > 0.0f ? cfg_.needle_weight / static_cast<float>(N) : 0.0f,
        cfg_.needle_ratio);

    // (host) Splits v_coeffs [N,K,3] → v_sh0 [N,3] / v_shN [N,K-1,3] (the inverse of the coeffs assembly)
    std::vector<float> v_sh0(static_cast<size_t>(N) * 3);
    std::vector<float> v_shN;
    if (f.K_total > 1) v_shN.resize(static_cast<size_t>(N) * (f.K_total - 1) * 3);
    for (int64_t i = 0; i < N; i++) {
        for (int c = 0; c < 3; c++) v_sh0[i * 3 + c] = sg.v_coeffs[(i * f.K_total) * 3 + c];
        for (int64_t k = 1; k < f.K_total; k++)
            for (int c = 0; c < 3; c++)
                v_shN[(i * (f.K_total - 1) + (k - 1)) * 3 + c] =
                    sg.v_coeffs[(i * f.K_total + k) * 3 + c];
    }

    // [06] fused_adam ×6 — in D3 parameter order. The valid mask makes culled slots skip everything,
    // moments included. sh0 and shN keep separate lrs just as gsplat does (2.5e-3 vs /20) —
    // SplatCloud's split storage is the single source of truth.
    const uint8_t* valid = f.valid.data();
    adam_step(cloud_.means, pg.v_means, means_st_, valid, means_lr_at(cfg_, iter));
    adam_step(cloud_.quats, pg.v_quats, quats_st_, valid, cfg_.lr_quats);
    adam_step(cloud_.scales, pg.v_scales, scales_st_, valid, cfg_.lr_scales);
    adam_step(cloud_.opacities, rg.v_opacities, opac_st_, valid, cfg_.lr_opacities);
    adam_step(cloud_.sh0, v_sh0, sh0_st_, valid, cfg_.lr_sh0);
    if (f.K_total > 1) adam_step(cloud_.shN, v_shN, shN_st_, valid, cfg_.lr_shN);

    // [07] MCMC (Phase 2 Task 3) — reproduces the order of gsplat simple_trainer.py:1133–1169:
    // strategy.step_post_backward runs **after** optimizer.step(). The refine (relocate→add) happens
    // only on scheduled steps, while noise is injected every step (mcmc.py:139–193). A count change
    // takes effect from the next step's fwd. The lr in noise_scale is the means lr used by this
    // step's adam — gsplat calls get_last_lr after scheduler.step() (one step of decay ahead, a
    // ×0.01^(1/total) deviation), a minor difference (documented in task-3-report).
    TrainStepResult r;
    if (cfg_.mcmc.enabled) {
        if (mcmc_refine_due(iter)) {
            const McmcRefineResult rr = mcmc_refine();
            r.mcmc_refined = true;
            r.n_relocated = static_cast<int64_t>(rr.relocated_dead.size());
            r.n_added = static_cast<int64_t>(rr.added_src.size());
        }
        mcmc_inject_noise(iter);
    }

    r.loss = loss; // the combined loss (see the TrainStepResult comment — at λ=0 it is l1.loss unchanged)
    r.n_isects = f.isect.n_isects;
    r.render_colors = std::move(f.raster.render_colors);
    r.depth_loss = dl.loss;
    last_loss_ = r.loss;
    last_depth_loss_ = r.depth_loss;
    return r;
}

DepthRender Trainer::render_with_depth(const TrainingCamera& cam) {
    DepthRender out;
    if (resident_ready()) {
        if (cam.width < 1 || cam.height < 1)
            throw std::invalid_argument("Trainer: camera width/height must be positive");
        if (!device_authoritative_) resident_upload();
        ResidentStepInputs in;
        in.viewmat = cam.viewmat;
        in.K = cam.K;
        in.width = cam.width;
        in.height = cam.height;
        in.count = cloud_.count;
        in.degrees_to_use = static_cast<uint32_t>(cloud_.sh_degree);
        in.forward_only = true;
        in.want_render = true;
        in.want_depth = true;
        ResidentStepOutputs ro = exec_->step(in);
        out.rgb = std::move(ro.render_colors);
        out.depth = std::move(ro.render_depths);
        return out;
    }
    Forward f = forward_pass(cam, static_cast<uint32_t>(cloud_.sh_degree), true);
    out.depth.resize(f.raster.render_alphas.size());
    for (size_t i = 0; i < out.depth.size(); i++)
        out.depth[i] = expected_depth(f.raster.render_depths[i], f.raster.render_alphas[i]);
    out.rgb = std::move(f.raster.render_colors);
    return out;
}

std::vector<float> Trainer::render(const TrainingCamera& cam) {
    if (resident_ready()) {
        // Resident forward — the same kernels and activation path as the forward in step() (so a
        // self-rendered GT is bit-consistent). The parameters are only read, so Home does not move
        // (it uploads host → device only when needed).
        if (cam.width < 1 || cam.height < 1)
            throw std::invalid_argument("Trainer: camera width/height must be positive");
        if (!device_authoritative_) resident_upload();
        ResidentStepInputs in;
        in.viewmat = cam.viewmat;
        in.K = cam.K;
        in.width = cam.width;
        in.height = cam.height;
        in.count = cloud_.count;
        in.degrees_to_use = static_cast<uint32_t>(cloud_.sh_degree); // eval uses the full degree
        in.forward_only = true;
        in.want_render = true;
        return std::move(exec_->step(in).render_colors);
    }
    Forward f = forward_pass(cam, static_cast<uint32_t>(cloud_.sh_degree));
    return std::move(f.raster.render_colors);
}

void Trainer::adam_step(std::vector<float>& param, const std::vector<float>& grad, AdamState& st,
                        const uint8_t* valid, float lr) {
    const int64_t n_elems = static_cast<int64_t>(grad.size()); // = count·D (only alive slots are updated)
    if (n_elems == 0) return; // D8-6 (shN at sh_degree 0, etc.)
    // The arena contract: param has arena_rows_ rows (cap_max under mcmc), grad has count (= alive)
    // rows. With mcmc off, arena_rows_ == count, which is equivalent to 1c's grad.size() ==
    // param.size() check.
    const int64_t D = static_cast<int64_t>(param.size()) / arena_rows_;
    if (grad.size() != static_cast<size_t>(cloud_.count * D))
        throw std::logic_error("Trainer::adam_step: grad/param size mismatch");
    if (st.exp_avg.empty()) {
        st.exp_avg.assign(param.size(), 0.0f);
        st.exp_avg_sq.assign(param.size(), 0.0f);
    }
    AdamInputs a;
    a.param = param.data();
    a.grad = grad.data();
    a.exp_avg = st.exp_avg.data();
    a.exp_avg_sq = st.exp_avg_sq.data();
    a.valid = valid;
    a.n_elems = n_elems;
    a.n_gauss = cloud_.count;
    a.lr = lr;
    a.b1 = cfg_.adam_b1;
    a.b2 = cfg_.adam_b2;
    a.eps = cfg_.adam_eps;
    backend_.fused_adam(a);
}

// ===== MCMC strategy (Phase 2 Task 3) =====
// Derivation notice: a reimplementation of the procedure in gsplat strategy/mcmc.py:139–239 +
// strategy/ops.py:301–401 (commit 77ab983). The Eq.9 numerics are computed by the mcmc_relocation
// kernel (spec 06 — it passes the golden gate); this block only handles the index flow
// (dead/sampled/new), the parameter scatter, and the Adam state surgery.

namespace {

// Matches torch.multinomial(replacement=True) (ops.py:62–92) — reproducing host determinism.
// mt19937_64 output is fixed by the ISO standard → bit-reproducible across platforms.
// uniform_real_distribution is implementation-defined (libc++ and libstdc++ differ) and is therefore
// prohibited — use u = (rng() >> 11)·2^-53 ∈ [0,1) (53 bits of precision).
// The cumulative sum is double and the search is upper_bound: an element with weight 0 leaves the cdf
// equal to the previous entry and can never be selected (consistent with torch.multinomial semantics).
std::vector<int64_t> multinomial_with_replacement(const float* weights, int64_t n_weights,
                                                  int64_t n_samples, std::mt19937_64& rng) {
    std::vector<double> cdf(static_cast<size_t>(n_weights));
    double acc = 0.0;
    for (int64_t i = 0; i < n_weights; i++) {
        const float w = weights[i];
        if (!(w >= 0.0f)) // rejects NaN too
            throw std::invalid_argument("multinomial: weights must be >= 0 and finite");
        acc += static_cast<double>(w);
        cdf[static_cast<size_t>(i)] = acc;
    }
    if (!(acc > 0.0)) throw std::invalid_argument("multinomial: weight sum must be > 0");
    std::vector<int64_t> out(static_cast<size_t>(n_samples));
    for (int64_t j = 0; j < n_samples; j++) {
        const double u = static_cast<double>(rng() >> 11) * 0x1.0p-53 * acc; // [0, acc)
        const int64_t idx = std::upper_bound(cdf.begin(), cdf.end(), u) - cdf.begin();
        out[static_cast<size_t>(j)] = std::min(idx, n_weights - 1); // guards the floating-point boundary
    }
    return out;
}

// SoA row copy (dst row ← src row) — the arena equivalent of ops.py:342
// `p[dead_indices] = p[sampled_idxs]` / ops.py:386 `cat([p, p[sampled_idxs]])`.
inline void copy_param_row(std::vector<float>& v, int64_t dim, int64_t dst, int64_t src) {
    for (int64_t k = 0; k < dim; k++) v[dst * dim + k] = v[src * dim + k];
}

// Gathers the relocation kernel inputs from a list of source indices and runs it (shared by relocate
// and add — ops.py:329–335 == ops.py:373–379, the same preprocessing):
// opacities = a post-sigmoid gather, scales = an exp(log_s) gather, and
// ratios = bincount(sampled)[sampled] + 1 clamped to [1, n_max] then cast to int32
// (relocation.py:61–62).
RelocationOutputs run_relocation_for(KernelBackend& backend, const SplatCloud& cloud,
                                     const std::vector<float>& op_act,
                                     const std::vector<int64_t>& sampled,
                                     const std::vector<float>& binoms, float min_opacity) {
    const int64_t n = static_cast<int64_t>(sampled.size());
    std::unordered_map<int64_t, int32_t> counts;
    for (int64_t s : sampled) counts[s]++;
    std::vector<float> g_op(static_cast<size_t>(n));
    std::vector<float> g_sc(static_cast<size_t>(n) * 3);
    std::vector<int32_t> ratios(static_cast<size_t>(n));
    for (int64_t j = 0; j < n; j++) {
        const int64_t s = sampled[static_cast<size_t>(j)];
        g_op[static_cast<size_t>(j)] = op_act[static_cast<size_t>(s)];
        for (int k = 0; k < 3; k++)
            g_sc[static_cast<size_t>(j) * 3 + k] = std::exp(cloud.scales[s * 3 + k]);
        ratios[static_cast<size_t>(j)] =
            std::min(counts[s] + 1, kMcmcNMax); // the lower bound of 1 is trivially met (its own count >= 1)
    }
    RelocationInputs rin;
    rin.opacities = g_op.data();
    rin.scales = g_sc.data();
    rin.ratios = ratios.data();
    rin.binoms = binoms.data();
    rin.N = n;
    rin.n_max = kMcmcNMax;
    rin.min_opacity = min_opacity;
    return backend.mcmc_relocation(rin);
}

} // namespace

bool Trainer::mcmc_refine_due(int iter) const {
    if (!cfg_.mcmc.enabled) return false;
    const McmcOptions& m = cfg_.mcmc;
    const double total = static_cast<double>(cfg_.total_iters);
    const int start = static_cast<int>(std::lround(total * m.refine_start_ratio));
    const int stop = static_cast<int>(std::lround(total * m.refine_stop_ratio));
    const int every = std::max(1, static_cast<int>(std::lround(total * m.refine_every_ratio)));
    // mcmc.py:159–163 — strict inequalities on both sides (no refine on the start/stop boundary steps)
    return iter > start && iter < stop && iter % every == 0;
}

Trainer::McmcRefineResult Trainer::mcmc_refine() {
    if (!cfg_.mcmc.enabled) throw std::logic_error("Trainer::mcmc_refine: mcmc disabled");
    sync_host_for_mutation(); // resident: reclaims parameters + moments, then makes the host authoritative
    McmcRefineResult out;
    mcmc_relocate(out); // mcmc.py:164–167 (_relocate_gs)
    mcmc_add(out);      // mcmc.py:169–175 (_add_new_gs)
    return out;
}

// relocate (ops.py:302–356): teleports dead Gaussians to source positions drawn from the alive set
// with replacement, in proportion to opacity.
void Trainer::mcmc_relocate(McmcRefineResult& out) {
    const int64_t N = cloud_.count;
    const std::vector<float> op_act = activate_opacities(cloud_.opacities.data(), N);

    // (1) dead = sigmoid(op) <= min_opacity (mcmc.py:203–204), alive = the rest (ops.py:321–322)
    std::vector<int64_t> dead, alive;
    for (int64_t i = 0; i < N; i++)
        (op_act[static_cast<size_t>(i)] <= cfg_.mcmc.min_opacity ? dead : alive).push_back(i);
    if (dead.empty()) return;
    if (alive.empty()) return; // total wipeout — the upstream original is undefined here (an empty multinomial), so guard and skip

    // (2) Draws from alive with replacement in proportion to opacity → maps back to global indices (ops.py:326–328)
    std::vector<float> probs(alive.size());
    for (size_t j = 0; j < alive.size(); j++) probs[j] = op_act[static_cast<size_t>(alive[j])];
    const std::vector<int64_t> local = multinomial_with_replacement(
        probs.data(), static_cast<int64_t>(alive.size()), static_cast<int64_t>(dead.size()),
        mcmc_rng_);
    std::vector<int64_t> sampled(dead.size());
    for (size_t j = 0; j < dead.size(); j++) sampled[j] = alive[static_cast<size_t>(local[j])];

    // (3)(4) ratios (bincount+1, clamped) → the relocation kernel (ops.py:329–335)
    const RelocationOutputs ro = run_relocation_for(backend_, cloud_, op_act, sampled, binoms_,
                                                    cfg_.mcmc.min_opacity);

    // (5) Parameter surgery (ops.py:337–343): the source opacity/scale ← the new values (converted
    // back to the internal logit/log representation), and **only then** dead row = a copy of the
    // source row (for every parameter — the copy happens after the update, so dead gets the new
    // values too). Duplicated sources receive identical (op, ratio, scale) inputs, so the new values
    // are identical as well — duplicate writes are harmless (same as the upstream original).
    for (size_t j = 0; j < sampled.size(); j++) {
        const int64_t s = sampled[j];
        cloud_.opacities[s] = logit_f32(ro.new_opacities[j]); // ∈ [min_op, 1-eps] ⊂ (0,1)
        for (int k = 0; k < 3; k++) {
            const float ns = ro.new_scales[j * 3 + k];
            if (!(ns > 0.0f)) // log-domain guard (the kernel contract says positive — a violation fails loudly)
                throw std::logic_error("Trainer::mcmc_relocate: non-positive new_scale");
            cloud_.scales[s * 3 + k] = std::log(ns);
        }
    }
    const int64_t K3 = static_cast<int64_t>(cloud_.sh_rest_coeffs()) * 3;
    const std::pair<std::vector<float>*, int64_t> params[] = {
        {&cloud_.means, 3},     {&cloud_.quats, 4}, {&cloud_.scales, 3},
        {&cloud_.opacities, 1}, {&cloud_.sh0, 3},   {&cloud_.shN, K3}};
    for (const auto& [v, dim] : params)
        for (size_t j = 0; j < dead.size(); j++) copy_param_row(*v, dim, dead[j], sampled[j]);

    // (6) Adam state surgery (ops.py:345–347): **only the sampled rows are reset to 0 — the upstream
    // original does not reset the dead rows** (confirmed against the original: optimizer_fn does only
    // `v[sampled_idxs] = 0` and never touches dead_indices). This is counter-intuitive, but it is the
    // gsplat reproduction contract — do not "fix" it, the goldens encode this behavior.
    mcmc_zero_adam_rows(sampled.data(), sampled.size());

    out.relocated_dead = std::move(dead);
    out.relocation_src = std::move(sampled);
}

// add (ops.py:360–401 + mcmc.py:219–239): 5% growth — a new slot is an Eq.9 split copy of a sampled
// source.
void Trainer::mcmc_add(McmcRefineResult& out) {
    const int64_t N = cloud_.count;
    // Growth target (mcmc.py:226–228). Instead of the original's int(1.05·N) truncation, this uses
    // the ceil from plan Task 3 / design §5.2 — it prevents stagnation at small N. Integer arithmetic
    // ceil(21N/20) = (21N+19)/20 (excludes floating-point error).
    const int64_t target = std::min(cfg_.mcmc.cap_max, (N * 21 + 19) / 20);
    const int64_t n_add = target - N;
    if (n_add <= 0) return;

    // (ops.py:369–372) probs = the opacity of the **entire** current set — unlike relocate, dead is
    // not excluded (relocate runs first, so dead has usually already been revived). The new sources
    // use global indices directly.
    const std::vector<float> op_act = activate_opacities(cloud_.opacities.data(), N);
    const std::vector<int64_t> sampled =
        multinomial_with_replacement(op_act.data(), N, n_add, mcmc_rng_);

    // (ops.py:373–379) ratios → the relocation kernel (the same preprocessing as relocate)
    const RelocationOutputs ro = run_relocation_for(backend_, cloud_, op_act, sampled, binoms_,
                                                    cfg_.mcmc.min_opacity);

    // (ops.py:381–387) The source opacity/scale ← the new values, and the new slot (N+j) = a copy of
    // the updated source row (for every parameter). Instead of a cat, it writes into the arena tail —
    // only count grows.
    for (size_t j = 0; j < sampled.size(); j++) {
        const int64_t s = sampled[j];
        cloud_.opacities[s] = logit_f32(ro.new_opacities[j]);
        for (int k = 0; k < 3; k++) {
            const float ns = ro.new_scales[j * 3 + k];
            if (!(ns > 0.0f))
                throw std::logic_error("Trainer::mcmc_add: non-positive new_scale");
            cloud_.scales[s * 3 + k] = std::log(ns);
        }
    }
    const int64_t K3 = static_cast<int64_t>(cloud_.sh_rest_coeffs()) * 3;
    const std::pair<std::vector<float>*, int64_t> params[] = {
        {&cloud_.means, 3},     {&cloud_.quats, 4}, {&cloud_.scales, 3},
        {&cloud_.opacities, 1}, {&cloud_.sh0, 3},   {&cloud_.shN, K3}};
    for (const auto& [v, dim] : params)
        for (int64_t j = 0; j < n_add; j++)
            copy_param_row(*v, dim, N + j, sampled[static_cast<size_t>(j)]);

    // (ops.py:389–391) The Adam state of a new row = 0. **The source rows are NOT reset** — the
    // convention differs from relocate (confirmed against the original: the optimizer_fn of
    // sample_add only cats zeros and never resets sampled_idxs). This asymmetry is intentional
    // reproduction of the gsplat original — do not "fix" it, the goldens encode this behavior. New
    // arena rows are structurally 0 already, but they are reset explicitly (guarding against an
    // externally touched state and against a future shrink).
    std::vector<int64_t> new_rows(static_cast<size_t>(n_add));
    for (int64_t j = 0; j < n_add; j++) new_rows[static_cast<size_t>(j)] = N + j;
    mcmc_zero_adam_rows(new_rows.data(), new_rows.size());

    cloud_.count = N + n_add;
    out.added_src = sampled;
}

// noise injection (mcmc.py:179–193 + ops.py:463–511): every step adds covariance- and
// opacity-gated noise to means. The noise comes from Philox(seed, offset=iter) — a separate stream
// per step, reproducible (spec 06).
void Trainer::mcmc_inject_noise(int iter) {
    if (!cfg_.mcmc.enabled) throw std::logic_error("Trainer::mcmc_inject_noise: mcmc disabled");
    sync_host_for_mutation(); // resident: reclaims before mutating means (the next resident step re-uploads)
    const int64_t N = cloud_.count;
    RngNormalInputs rin;
    rin.seed = cfg_.mcmc.seed;
    rin.offset = static_cast<uint64_t>(iter);
    rin.count = N * 3;
    const std::vector<float> noise = backend_.rng_normal_f32(rin);

    PerturbInputs pin;
    pin.positions = cloud_.means.data();
    pin.quats = cloud_.quats.data();
    pin.scales_log = cloud_.scales.data();
    pin.opacities_logit = cloud_.opacities.data();
    pin.noise = noise.data();
    pin.N = N;
    pin.noise_scale = means_lr_at(cfg_, iter) * cfg_.mcmc.noise_lr; // mcmc.py:190
    // t/k = the PerturbInputs defaults 0.005/100 (ops.py:35–36 DEFAULT_MCMC_OPACITY_T/K)
    backend_.mcmc_perturb_positions(pin);
}

void Trainer::mcmc_zero_adam_rows(const int64_t* rows, size_t n_rows) {
    const int64_t K3 = static_cast<int64_t>(cloud_.sh_rest_coeffs()) * 3;
    const std::pair<AdamState*, int64_t> groups[] = {{&means_st_, 3}, {&quats_st_, 4},
                                                     {&scales_st_, 3}, {&opac_st_, 1},
                                                     {&sh0_st_, 3},   {&shN_st_, K3}};
    for (const auto& [st, dim] : groups) {
        if (st->exp_avg.empty() || dim == 0) continue; // shN at sh_degree 0, etc.
        for (size_t r = 0; r < n_rows; r++)
            for (int64_t k = 0; k < dim; k++) {
                st->exp_avg[rows[r] * dim + k] = 0.0f;
                st->exp_avg_sq[rows[r] * dim + k] = 0.0f;
            }
    }
}

// ===== Checkpoints (Phase 6 round 2 Task 2 — design §7.5/§8.1; the contract is in
// Trainer.h/Checkpoint.h) =====

namespace {
// Column count in D3 group order (the same definition as group_dim in Checkpoint.cpp — the count·dim
// serialization contract)
inline int64_t ck_group_dim(int g, int sh_degree) {
    constexpr int64_t fixed[5] = {3, 4, 3, 1, 3};
    if (g < 5) return fixed[g];
    return (static_cast<int64_t>(sh_degree) + 1) * (sh_degree + 1) * 3 - 3; // (K-1)·3
}
} // namespace

CheckpointConfig Trainer::checkpoint_config(int factor, int test_split) const {
    CheckpointConfig c;
    c.sh_degree = cloud_.sh_degree;
    c.factor = factor;
    c.test_split = test_split;
    c.total_iters = cfg_.total_iters;
    c.mcmc_enabled = cfg_.mcmc.enabled ? 1 : 0;
    c.antialiased = cfg_.antialiased ? 1 : 0;
    c.cap_max = cfg_.mcmc.enabled ? cfg_.mcmc.cap_max : 0;
    c.mcmc_seed = cfg_.mcmc.enabled ? cfg_.mcmc.seed : 0;
    c.scene_scale = cfg_.scene_scale;
    c.ssim_weight = cfg_.ssim_weight;
    c.flatten_weight = cfg_.flatten_weight;
    c.needle_weight = cfg_.needle_weight;
    c.needle_ratio = cfg_.needle_ratio;
    c.has_mask = cfg_.has_mask ? 1 : 0;
    return c;
}

TrainerCheckpoint Trainer::make_checkpoint(int iter, int factor, int test_split) {
    if (iter < 0) throw std::invalid_argument("Trainer::make_checkpoint: iter must be >= 0");
    // resident: lazily downloads the parameters and moments (read-only — Home stays on the device, so
    // the subsequent step trajectory is unchanged). On the host path, or right after a refine
    // (Home=host), this is a no-op.
    sync_host_from_device();

    TrainerCheckpoint ck;
    ck.config = checkpoint_config(factor, test_split);
    ck.iter = iter;
    const int64_t n = cloud_.count;
    ck.cloud.count = n;
    ck.cloud.sh_degree = cloud_.sh_degree;
    const std::vector<float>* params[6] = {&cloud_.means, &cloud_.quats,     &cloud_.scales,
                                           &cloud_.opacities, &cloud_.sh0,   &cloud_.shN};
    std::vector<float>* out_params[6] = {&ck.cloud.means, &ck.cloud.quats, &ck.cloud.scales,
                                         &ck.cloud.opacities, &ck.cloud.sh0, &ck.cloud.shN};
    const AdamState* sts[6] = {&means_st_, &quats_st_, &scales_st_, &opac_st_, &sh0_st_, &shN_st_};
    for (int g = 0; g < 6; g++) {
        const size_t cnt = static_cast<size_t>(n) * ck_group_dim(g, cloud_.sh_degree);
        out_params[g]->assign(params[g]->begin(),
                              params[g]->begin() + static_cast<int64_t>(cnt)); // count rows only
        CheckpointMoments& m = ck.moments[g];
        if (sts[g]->exp_avg.empty()) {
            // The moments are lazily unallocated (before the first step, etc.) — snapshot zeros (the
            // same meaning as the lazy zero-init)
            m.exp_avg.assign(cnt, 0.0f);
            m.exp_avg_sq.assign(cnt, 0.0f);
        } else {
            m.exp_avg.assign(sts[g]->exp_avg.begin(),
                             sts[g]->exp_avg.begin() + static_cast<int64_t>(cnt));
            m.exp_avg_sq.assign(sts[g]->exp_avg_sq.begin(),
                                sts[g]->exp_avg_sq.begin() + static_cast<int64_t>(cnt));
        }
    }
    if (cfg_.mcmc.enabled) {
        // The multinomial engine is stateful — serialize it as text (the RNG audit comment in
        // Checkpoint.h). operator<< is const — it does not move the engine's stream position (the
        // snapshot does not interfere). The classic locale is pinned (review fix 4): it blocks a
        // global locale's thousands separators and the like from contaminating the numeric text and
        // breaking the restore (iss >> engine) — save and restore stay symmetric.
        std::ostringstream oss;
        oss.imbue(std::locale::classic());
        oss << mcmc_rng_;
        ck.mcmc_rng_state = oss.str();
    }
    return ck;
}

void Trainer::restore_checkpoint(const TrainerCheckpoint& ck, int factor, int test_split) {
    // Fingerprint verification — sh_degree is taken from cloud_ (if the cloud passed to the ctor and
    // ck disagree, it is caught here). TrainC/CLI also pre-verify using a fingerprint built from the
    // user-requested values (defense in depth — TrainC.cpp).
    verify_checkpoint_config(ck.config, checkpoint_config(factor, test_split));
    const int64_t n = ck.cloud.count;
    if (n < 1) throw std::invalid_argument("Trainer::restore_checkpoint: empty checkpoint");
    if (n > arena_rows_)
        throw std::invalid_argument(
            "Trainer::restore_checkpoint: checkpoint count exceeds arena rows (cap_max)");
    // mcmc off: the arena is a fixed count (from the ctor cloud) — it MUST match ck.count exactly for
    // the vector and dispatch size contracts to hold (the intended flow: pass ck.cloud straight into
    // the ctor — the TrainC resume path)
    if (!cfg_.mcmc.enabled && n != arena_rows_)
        throw std::invalid_argument(
            "Trainer::restore_checkpoint: count != ctor cloud count (mcmc off fixed-N)");

    std::vector<float>* params[6] = {&cloud_.means, &cloud_.quats,     &cloud_.scales,
                                     &cloud_.opacities, &cloud_.sh0,   &cloud_.shN};
    const std::vector<float>* in_params[6] = {&ck.cloud.means, &ck.cloud.quats,
                                              &ck.cloud.scales, &ck.cloud.opacities,
                                              &ck.cloud.sh0, &ck.cloud.shN};
    AdamState* sts[6] = {&means_st_, &quats_st_, &scales_st_, &opac_st_, &sh0_st_, &shN_st_};
    for (int g = 0; g < 6; g++) { // pre-verifies the size contract (prevents a partial restore — every group is checked before anything is written)
        const size_t cnt = static_cast<size_t>(n) * ck_group_dim(g, cloud_.sh_degree);
        if (in_params[g]->size() != cnt || ck.moments[g].exp_avg.size() != cnt ||
            ck.moments[g].exp_avg_sq.size() != cnt)
            throw std::invalid_argument(
                "Trainer::restore_checkpoint: group size != count*dim (corrupt checkpoint)");
    }
    cloud_.count = n;
    for (int g = 0; g < 6; g++) {
        std::copy(in_params[g]->begin(), in_params[g]->end(), params[g]->begin());
        // Moments: re-establish zeros across the whole arena, then restore count rows (it clears
        // leftover state in the tail — the untouched contract)
        sts[g]->exp_avg.assign(params[g]->size(), 0.0f);
        sts[g]->exp_avg_sq.assign(params[g]->size(), 0.0f);
        std::copy(ck.moments[g].exp_avg.begin(), ck.moments[g].exp_avg.end(),
                  sts[g]->exp_avg.begin());
        std::copy(ck.moments[g].exp_avg_sq.begin(), ck.moments[g].exp_avg_sq.end(),
                  sts[g]->exp_avg_sq.begin());
    }
    if (cfg_.mcmc.enabled) {
        if (ck.mcmc_rng_state.empty()) // the fingerprint says mcmc is on but the state is absent = corruption
            throw std::invalid_argument("Trainer::restore_checkpoint: missing mcmc rng state");
        std::istringstream iss(ck.mcmc_rng_state);
        iss.imbue(std::locale::classic()); // symmetric with the save side (review fix 4)
        iss >> mcmc_rng_;
        if (!iss)
            throw std::invalid_argument("Trainer::restore_checkpoint: corrupt mcmc rng state");
    }
    // Resets the resident state — the next resident step re-establishes the device zero tail through
    // a first upload (of the whole arena)
    exec_uploaded_once_ = false;
    device_authoritative_ = false;
    host_fresh_ = true;
}

Trainer::AdamState& Trainer::adam_state(ParamGroup g) {
    if (!cfg_.mcmc.enabled) throw std::logic_error("Trainer::adam_state: mcmc disabled");
    sync_host_for_mutation(); // the returned reference allows mutation — conservatively make the host authoritative
    switch (g) {
    case ParamGroup::Means: return means_st_;
    case ParamGroup::Quats: return quats_st_;
    case ParamGroup::Scales: return scales_st_;
    case ParamGroup::Opacities: return opac_st_;
    case ParamGroup::Sh0: return sh0_st_;
    case ParamGroup::ShN: return shN_st_;
    }
    throw std::logic_error("Trainer::adam_state: invalid group");
}

} // namespace splatx
