// Phase 1c Task 6 — the ImageIO PNG round-trip, the init_splats KNN hand-check, and the L1 loss and
// grad hand-check.
// Phase 2 Task 5 — the PSNR hand-check and the train-test split convention.
// Integration Task 2 — the factor downscale fallback when images_{factor} is absent
// (resolve_images_dir plus load_image_rgb_f32_downscaled).
#include "TestHarness.h"
#include "TestPaths.h" // tmp_path() — a temp path compatible with the iOS sandbox
#include "splatx/ImageIO.h"
#include "splatx/Json.h"     // perceptual metrics Task 1 — the ssim_basic meta tolerances (json_parse_flat)
#include "splatx/SXTensor.h" // perceptual metrics Task 1 — loading the ssim_basic fixture
#include "splatx/Trainer.h"
#include "splatx_train_c.h" // color correction Task 2 — the SxTrainStats.final_cc_psnr C API integration assertion
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib> // std::getenv — drives SPLATX_TEST_METALLIB end to end (the E2E backend is created per run)
#include <cstring> // std::memcmp — eval_ssim determinism (bit-identical)
#include <filesystem>
#include <fstream> // writing the synthetic COLMAP txt bundle (color correction Task 2 E2E)
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace splatx;

// ===== (a) PNG save → load round-trip (within the 8-bit quantization error) =====
// The JPG path shares the same code (CGImageSource detects the format automatically) — there is no
// JPG fixture in the repository, so it is verified through the PNG round-trip (the task-6 brief).
// Using W != H (5x3) also catches a flipped row stride.
SX_TEST(image_png_roundtrip_8bit_quantization) {
    const int W = 5, H = 3;
    ImageF32 a;
    a.width = W;
    a.height = H;
    a.rgb.resize(static_cast<size_t>(W) * H * 3);
    for (size_t i = 0; i < a.rgb.size(); i++) // a 0..1 gradient (every pixel differs — position sensitive)
        a.rgb[i] = static_cast<float>(i) / static_cast<float>(a.rgb.size() - 1);
    a.rgb[0] = 0.0f;
    a.rgb[1] = 1.0f;
    a.rgb[2] = 17.0f / 255.0f; // exactly on the 8-bit grid — quantization is expected to be lossless

    const std::string path_s = sxtest::tmp_path("splatx_test_img.png"); // iOS sandbox compatible (TestPaths.h)
    const char* path = path_s.c_str();
    save_png_rgb_f32(a, path);
    ImageF32 b = load_image_rgb_f32(path);
    std::remove(path);

    SX_CHECK(b.width == W && b.height == H);
    SX_CHECK(b.rgb.size() == a.rgb.size());
    // The 8-bit quantization limit: round(v·255)/255 → |a-b| <= 0.5/255 (plus f32 slack)
    const float tol = 0.5f / 255.0f + 1e-6f;
    size_t bad = 0;
    for (size_t i = 0; i < a.rgb.size() && i < b.rgb.size(); i++)
        if (!(std::fabs(a.rgb[i] - b.rgb[i]) <= tol)) bad++;
    SX_CHECK(bad == 0);
    // Values exactly on the grid are bit-identical (k/255 → k → k/255)
    SX_CHECK(b.rgb[0] == 0.0f);
    SX_CHECK(b.rgb[1] == 1.0f);
    SX_CHECK(b.rgb[2] == 17.0f / 255.0f);

    // Out-of-range values are clamped to [0,1] before saving
    ImageF32 c;
    c.width = 1;
    c.height = 1;
    c.rgb = {-0.5f, 1.5f, 0.25f};
    save_png_rgb_f32(c, path);
    ImageF32 d = load_image_rgb_f32(path);
    std::remove(path);
    SX_CHECK(d.rgb.size() == 3);
    if (d.rgb.size() == 3) {
        SX_CHECK(d.rgb[0] == 0.0f);
        SX_CHECK(d.rgb[1] == 1.0f);
        SX_CHECK_NEAR(d.rgb[2], 0.25f, 0, tol);
    }

    // Failure paths: loading a nonexistent file, and saving with a size mismatch → throw
    bool threw = false;
    try { (void)load_image_rgb_f32(sxtest::tmp_path("splatx_test_img_missing.png")); } catch (...) { threw = true; }
    SX_CHECK(threw);
    ImageF32 e;
    e.width = 2;
    e.height = 2;
    e.rgb = {0.f, 0.f, 0.f}; // 2*2*3 != 3
    threw = false;
    try { save_png_rgb_f32(e, path); } catch (...) { threw = true; }
    SX_CHECK(threw);
}

// ===== (a2) The fallback when images_{factor} is absent — Integration Task 2 =====
// Using a synthetic bundle (images/ only, 101x67 — odd dimensions: 101/4 = 25.25 → 25 rounds down,
// 67/4 = 16.75 → 17 rounds up), this covers ① the dimension convention
// downscaled_dim == round(orig/factor) (including values measured from the garden images_4),
// ② the fallback decision (images_N absent → images/ plus the flag, present → no fallback, neither
// present → throw), ③ the exact dimensions of the downscaled load plus its content (the top stays
// white and the bottom black — verifying vertical flipping and scaling), and ④ intrinsics
// consistency (the camera dimensions scaled with the same formula as the TrainC factor block ==
//    the loader output — the condition for passing the GT dims guard).
SX_TEST(image_factor_fallback_downscale) {
    namespace fs = std::filesystem;

    // The dimension convention — the single source of truth shared by the TrainC intrinsics scaling
    // and the fallback loader
    SX_CHECK(downscaled_dim(101, 4) == 25); // 25.25 → 25
    SX_CHECK(downscaled_dim(67, 4) == 17);  // 16.75 → 17
    SX_CHECK(downscaled_dim(5187, 4) == 1297); // measured from garden source→images_4 (the ImageMagick convention)
    SX_CHECK(downscaled_dim(3361, 4) == 840);
    SX_CHECK(downscaled_dim(1920, 2) == 960);

    const fs::path data = sxtest::tmp_path("splatx_fallback_data");
    std::error_code ec;
    fs::remove_all(data, ec);
    fs::create_directories(data / "images");
    const int W = 101, H = 67, F = 4;
    ImageF32 src;
    src.width = W;
    src.height = H;
    src.rgb.assign(static_cast<size_t>(W) * H * 3, 0.0f);
    for (int y = 0; y < H / 2; y++) // the top half white, the bottom half black
        for (int x = 0; x < W * 3; x++) src.rgb[static_cast<size_t>(y) * W * 3 + x] = 1.0f;
    const std::string img_path = (data / "images" / "0001.png").string();
    save_png_rgb_f32(src, img_path);

    // The fallback decision: images_4 absent → images/ plus downscale_on_load
    const ImagesDirChoice fb = resolve_images_dir(data.string(), F);
    SX_CHECK(fb.downscale_on_load);
    SX_CHECK(fb.dir == (data / "images").string());
    // factor 1 always uses the images/ source (this is not a fallback)
    const ImagesDirChoice f1 = resolve_images_dir(data.string(), 1);
    SX_CHECK(!f1.downscale_on_load && f1.dir == (data / "images").string());

    // The downscaled load: the exact target dimensions (including the rounding of odd dimensions) —
    // matching the intrinsics scaling formula
    const int w1 = downscaled_dim(W, F), h1 = downscaled_dim(H, F); // the TrainC factor block formula
    ImageF32 dn = load_image_rgb_f32_downscaled(img_path, F);
    SX_CHECK(dn.width == 25 && dn.height == 17);
    SX_CHECK(dn.width == w1 && dn.height == h1); // the condition for passing the GT dims guard
    SX_CHECK(dn.rgb.size() == static_cast<size_t>(dn.width) * dn.height * 3);
    // The content: rows far from the boundary (y ≈ 33/67·17 ≈ 8.5) keep their original color — with
    // 0.02 of slack for resampling ringing
    if (dn.width == 25 && dn.height == 17) {
        for (int x = 0; x < 25 * 3; x++) {
            SX_CHECK_NEAR(dn.rgb[x], 1.0f, 0, 0.02);                          // row 0 = white
            SX_CHECK_NEAR(dn.rgb[static_cast<size_t>(16) * 25 * 3 + x], 0.0f, 0, 0.02); // row 16 = black
        }
    }
    // A factor 1 downscaled load is equivalent to the default load (in dimensions)
    ImageF32 same = load_image_rgb_f32_downscaled(img_path, 1);
    SX_CHECK(same.width == W && same.height == H);

    // No fallback when images_4 exists — the existing folder choice is kept (an absolute condition)
    fs::create_directories(data / "images_4");
    const ImagesDirChoice pre = resolve_images_dir(data.string(), F);
    SX_CHECK(!pre.downscale_on_load);
    SX_CHECK(pre.dir == (data / "images_4").string());

    // If neither folder exists, this is an explicit error (it is not deferred to the first decode)
    bool threw = false;
    try {
        (void)resolve_images_dir((data / "no_such_bundle").string(), F);
    } catch (const std::exception&) {
        threw = true;
    }
    SX_CHECK(threw);
    // The loader explicitly rejects an excessive factor (target dimension < 1) — a 3x2 image with
    // factor 8
    ImageF32 tiny;
    tiny.width = 3;
    tiny.height = 2;
    tiny.rgb.assign(3 * 2 * 3, 0.5f);
    const std::string tiny_path = (data / "images" / "tiny.png").string();
    save_png_rgb_f32(tiny, tiny_path);
    threw = false;
    try {
        (void)load_image_rgb_f32_downscaled(tiny_path, 8);
    } catch (const std::exception&) {
        threw = true;
    }
    SX_CHECK(threw);
    fs::remove_all(data, ec);
}

// ===== (b) init_splats — a 4-point KNN hand-check, the field conventions, and the subsample and
// zero-distance guards =====
SX_TEST(init_splats_knn_hand_check) {
    TrainingInput ti;
    // Hand-computed coordinates: the pairwise distances are d(0,1)=1, d(0,2)=2, d(0,3)=4, d(1,2)=√5,
    // d(1,3)=√17, d(2,3)=√20
    ti.seed_points = {0, 0, 0,  1, 0, 0,  0, 2, 0,  0, 0, 4};
    ti.seed_rgb = {255, 0, 128,  0, 0, 0,  255, 255, 255,  10, 20, 30};
    InitOptions o;
    o.max_points = 100; // M=4 < cap → no subsampling
    o.init_opacity = 0.1f;
    o.sh_degree = 2;
    SplatCloud c = init_splats(ti, o);

    SX_CHECK(c.count == 4);
    SX_CHECK(c.sh_degree == 2 && c.sh_rest_coeffs() == 8);
    SX_CHECK(c.means.size() == 12 && c.scales.size() == 12 && c.quats.size() == 16);
    SX_CHECK(c.opacities.size() == 4 && c.sh0.size() == 12);
    SX_CHECK(c.shN.size() == 4u * 8u * 3u);

    // means = the seed verbatim (bit-identical)
    for (size_t i = 0; i < 12; i++) SX_CHECK(c.means[i] == ti.seed_points[i]);
    // quats = identity (w-first), shN = 0
    for (int i = 0; i < 4; i++) {
        SX_CHECK(c.quats[i * 4 + 0] == 1.0f && c.quats[i * 4 + 1] == 0.0f &&
                 c.quats[i * 4 + 2] == 0.0f && c.quats[i * 4 + 3] == 0.0f);
    }
    for (float v : c.shN) SX_CHECK(v == 0.0f);
    // opacities = logit(0.1) = ln(0.1/0.9) = -ln 9 = -2.1972246 (stored in logit space)
    for (int i = 0; i < 4; i++) SX_CHECK_NEAR(c.opacities[i], -2.1972246f, 1e-6, 0);
    // sh0 = (rgb/255 - 0.5)/C0: 255 → 0.5/C0 = 1.7724539, 0 → -1.7724539,
    // 128 → (128/255 - 0.5)/C0 = 0.0069509
    SX_CHECK_NEAR(c.sh0[0], 1.7724539f, 1e-6, 0);
    SX_CHECK_NEAR(c.sh0[1], -1.7724539f, 1e-6, 0);
    SX_CHECK_NEAR(c.sh0[2], (128.0f / 255.0f - 0.5f) / kShC0, 1e-6, 0);
    SX_CHECK_NEAR(c.sh0[3], -1.7724539f, 1e-6, 0); // p1 rgb (0,0,0)

    // scales = log(the mean 3-NN distance), isotropic so all three components are equal — the
    // expected value is derived from the closed-form distances (hand-computed above)
    const double exp_mean[4] = {
        (1.0 + 2.0 + 4.0) / 3.0,                              // p0: 7/3
        (1.0 + std::sqrt(5.0) + std::sqrt(17.0)) / 3.0,       // p1
        (2.0 + std::sqrt(5.0) + std::sqrt(20.0)) / 3.0,       // p2
        (4.0 + std::sqrt(17.0) + std::sqrt(20.0)) / 3.0,      // p3
    };
    for (int i = 0; i < 4; i++) {
        SX_CHECK_NEAR(c.scales[i * 3 + 0], std::log(exp_mean[i]), 1e-6, 0);
        SX_CHECK(c.scales[i * 3 + 0] == c.scales[i * 3 + 1]);
        SX_CHECK(c.scales[i * 3 + 1] == c.scales[i * 3 + 2]);
    }

    // Subsampling: M=5 with max_points=2 → an even stride picks {0, 2} (i·M/N)
    TrainingInput ts;
    for (int i = 0; i < 5; i++) {
        ts.seed_points.insert(ts.seed_points.end(),
                              {static_cast<float>(10 * i), 0.f, 0.f});
        ts.seed_rgb.insert(ts.seed_rgb.end(), {static_cast<uint8_t>(i), 0, 0});
    }
    InitOptions o2;
    o2.max_points = 2;
    SplatCloud s = init_splats(ts, o2);
    SX_CHECK(s.count == 2);
    SX_CHECK(s.means.size() == 6 && s.means[0] == 0.0f && s.means[3] == 20.0f);
    SX_CHECK_NEAR(s.sh0[3], (2.0f / 255.0f - 0.5f) / kShC0, 1e-6, 0); // rgb travels with the point (seed 2)

    // The zero-distance guard: all points coincide → mean distance 0 → clamped to 1e-6 → log(1e-6)
    TrainingInput tz;
    for (int i = 0; i < 4; i++) {
        tz.seed_points.insert(tz.seed_points.end(), {1.f, 2.f, 3.f});
        tz.seed_rgb.insert(tz.seed_rgb.end(), {0, 0, 0});
    }
    SplatCloud z = init_splats(tz, o);
    for (float v : z.scales) SX_CHECK_NEAR(v, std::log(1e-6), 1e-6, 0);

    // Invalid inputs throw: an empty seed, an rgb size mismatch, or an out-of-range opacity
    auto throws = [](TrainingInput in, InitOptions op) {
        try { (void)init_splats(in, op); } catch (...) { return true; }
        return false;
    };
    SX_CHECK(throws(TrainingInput{}, o));
    TrainingInput tm = ti;
    tm.seed_rgb.pop_back();
    SX_CHECK(throws(tm, o));
    InitOptions ob = o;
    ob.init_opacity = 1.0f; // logit(1) diverges
    SX_CHECK(throws(ti, ob));
}

// ===== (b') The knn3_mean_dists grid compared against the brute-force reference (Phase 6 2nd round
// Task 1) =====
// The grid is an **exact** algorithm (the argument is in Trainer.h), so the rel error is 0 — because
// the arithmetic is identical (squared distances → summing sqrt in ascending order), bit equality is
// required as well. The reference implementation below uses the same formulas as the O(N²) original
// from 1c (test-local — it preserves the reference for a path that was removed from production).
namespace {
std::vector<float> knn3_ref_brute(const std::vector<float>& pts) {
    const int64_t n = static_cast<int64_t>(pts.size()) / 3;
    constexpr float kInf = std::numeric_limits<float>::infinity();
    std::vector<float> out(n, 0.0f);
    for (int64_t i = 0; i < n; i++) {
        float best[3] = {kInf, kInf, kInf};
        const float xi = pts[i * 3 + 0], yi = pts[i * 3 + 1], zi = pts[i * 3 + 2];
        for (int64_t j = 0; j < n; j++) {
            if (j == i) continue;
            const float dx = pts[j * 3 + 0] - xi;
            const float dy = pts[j * 3 + 1] - yi;
            const float dz = pts[j * 3 + 2] - zi;
            const float d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < best[0]) {
                best[2] = best[1]; best[1] = best[0]; best[0] = d2;
            } else if (d2 < best[1]) {
                best[2] = best[1]; best[1] = d2;
            } else if (d2 < best[2]) {
                best[2] = d2;
            }
        }
        float sum = 0.0f;
        int k = 0;
        for (float d2 : best) {
            if (d2 == kInf) break;
            sum += std::sqrt(d2);
            k++;
        }
        out[i] = k > 0 ? sum / static_cast<float>(k) : 0.0f;
    }
    return out;
}

// Compares every point bit-for-bit and confirms the init_scale (log(dist·s)) rel error is 0
void check_knn_grid_exact(const std::vector<float>& pts, const char* label) {
    const std::vector<float> ref = knn3_ref_brute(pts);
    const std::vector<float> got =
        splatx::knn3_mean_dists(pts.data(), static_cast<int64_t>(pts.size()) / 3);
    SX_CHECK(got.size() == ref.size());
    size_t bad = 0;
    double max_rel = 0.0;
    for (size_t i = 0; i < ref.size() && i < got.size(); i++) {
        if (got[i] != ref[i]) bad++; // bit-identical (an exact algorithm plus the same arithmetic order)
        const double lr = std::log(std::max(ref[i], 1e-6f) * 0.1f); // the MCMC preset multiplier
        const double lg = std::log(std::max(got[i], 1e-6f) * 0.1f);
        if (lr != 0.0) max_rel = std::max(max_rel, std::fabs(lg - lr) / std::fabs(lr));
    }
    if (bad != 0 || max_rel != 0.0)
        std::printf("  [knn %s] mismatch %zu / rel err %.3e\n", label, bad, max_rel);
    SX_CHECK(bad == 0);
    SX_CHECK(max_rel == 0.0); // gate (b): the init_scale rel error against the exact KNN = 0
}
} // namespace

SX_TEST(knn3_grid_matches_bruteforce) {
    std::mt19937 rng(20260712);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    // (1) A uniform cube of 1500 points (the grid path — above the brute-force threshold of 256)
    std::vector<float> uniform(1500 * 3);
    for (float& v : uniform) v = u01(rng) * 10.0f - 5.0f;
    check_knn_grid_exact(uniform, "uniform");

    // (2) A cluster plus distant outliers (the shape of a COLMAP seed — the path that clamps points
    // outside the quantile box to boundary cells)
    std::vector<float> skewed;
    for (int i = 0; i < 1200; i++)
        for (int a = 0; a < 3; a++) skewed.push_back(u01(rng));
    for (int i = 0; i < 30; i++) // ~2.4% outliers, coordinates ±1e3
        for (int a = 0; a < 3; a++) skewed.push_back((u01(rng) - 0.5f) * 2e3f);
    check_knn_grid_exact(skewed, "skewed");

    // (3) Planar (a degenerate z axis) and (4) linear (one active axis with sparse spacing — many
    // ring expansions are required)
    std::vector<float> planar(1000 * 3);
    for (int i = 0; i < 1000; i++) {
        planar[i * 3 + 0] = u01(rng) * 4.0f;
        planar[i * 3 + 1] = u01(rng) * 4.0f;
        planar[i * 3 + 2] = 7.5f;
    }
    check_knn_grid_exact(planar, "planar");
    std::vector<float> line(800 * 3, 0.0f);
    for (int i = 0; i < 800; i++)
        line[i * 3 + 0] = static_cast<float>(i) * (1.0f + 0.01f * static_cast<float>(i));
    check_knn_grid_exact(line, "line");

    // (5) Independence from input order: reversing (2) still keeps each point's value travelling with
    // it (a deterministic tiebreak)
    const int64_t ns = static_cast<int64_t>(skewed.size()) / 3;
    std::vector<float> rev(skewed.size());
    for (int64_t i = 0; i < ns; i++)
        for (int a = 0; a < 3; a++) rev[(ns - 1 - i) * 3 + a] = skewed[i * 3 + a];
    const std::vector<float> fwd_out = splatx::knn3_mean_dists(skewed.data(), ns);
    const std::vector<float> rev_out = splatx::knn3_mean_dists(rev.data(), ns);
    size_t moved = 0;
    for (int64_t i = 0; i < ns; i++)
        if (fwd_out[i] != rev_out[ns - 1 - i]) moved++;
    SX_CHECK(moved == 0);
}

// ===== (b'') knn3_mean_dists edge cases — duplicate coordinates, N<4, distant clusters, and ring
// expansion =====
SX_TEST(knn3_grid_edge_cases) {
    // Every point at the same coordinate (a fully degenerate bbox — the grid early-return path): all
    // distances are 0
    std::vector<float> same(600 * 3);
    for (int i = 0; i < 600; i++) {
        same[i * 3 + 0] = 1.f; same[i * 3 + 1] = 2.f; same[i * 3 + 2] = 3.f;
    }
    const std::vector<float> z = splatx::knn3_mean_dists(same.data(), 600);
    for (float v : z) SX_CHECK(v == 0.0f);

    // N = 1 → {0} (no neighbors), N = 2 → their mutual distance, N = 3 → the mean over k = 2
    const std::vector<float> p1 = {0.f, 0.f, 0.f};
    SX_CHECK(splatx::knn3_mean_dists(p1.data(), 1) == std::vector<float>{0.0f});
    const std::vector<float> p2 = {0.f, 0.f, 0.f, 3.f, 4.f, 0.f}; // d = 5
    const std::vector<float> o2 = splatx::knn3_mean_dists(p2.data(), 2);
    SX_CHECK(o2[0] == 5.0f && o2[1] == 5.0f);
    const std::vector<float> p3 = {0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 2.f, 0.f};
    const std::vector<float> o3 = splatx::knn3_mean_dists(p3.data(), 3);
    // p0: (1+2)/2, p1: (1+√5)/2, p2: (2+√5)/2
    SX_CHECK_NEAR(o3[0], 1.5f, 1e-6, 0);
    SX_CHECK_NEAR(o3[1], (1.0 + std::sqrt(5.0)) / 2.0, 1e-6, 0);
    SX_CHECK_NEAR(o3[2], (2.0 + std::sqrt(5.0)) / 2.0, 1e-6, 0);

    // Mostly duplicates plus a few scattered points (a degenerate quantile box → the full-bbox
    // fallback path) — compared against brute force
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::vector<float> dup(700 * 3, 0.5f);
    for (int i = 690; i < 700; i++)
        for (int a = 0; a < 3; a++) dup[i * 3 + a] = u01(rng) * 5.0f;
    check_knn_grid_exact(dup, "mostly-dup");

    // Two distant clusters (many empty cells between them — the boundary for ring expansion and
    // early termination) — compared against brute force
    std::vector<float> clusters;
    for (int i = 0; i < 600; i++)
        for (int a = 0; a < 3; a++) clusters.push_back(u01(rng));
    for (int i = 0; i < 600; i++)
        for (int a = 0; a < 3; a++)
            clusters.push_back(u01(rng) + (a == 0 ? 100.0f : 0.0f));
    check_knn_grid_exact(clusters, "two-clusters");

    // Invalid input: null pts with n > 0
    bool threw = false;
    try { (void)splatx::knn3_mean_dists(nullptr, 4); } catch (...) { threw = true; }
    SX_CHECK(threw);
}

// ===== (b''') The extreme anisotropy performance regression (review fix 1) =====
// A 1e6 × 1 × 1e-3 scale distribution: if the per-axis dims cap breaks grid coverage, most points get
// clamped into boundary cells and ring expansion explodes to r in the thousands — over 60 seconds at
// n=750 (measured by the reviewer before the fix; brute force took <1 ms). After the fix (preserving
// coverage plus clipping the shell boundary) it must be on the order of milliseconds.
SX_TEST(knn3_grid_extreme_anisotropy_perf) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    const int64_t n = 750;
    std::vector<float> pts(static_cast<size_t>(n) * 3);
    for (int64_t i = 0; i < n; i++) {
        pts[i * 3 + 0] = u01(rng) * 1e6f;
        pts[i * 3 + 1] = u01(rng);
        pts[i * 3 + 2] = u01(rng) * 1e-3f;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<float> got = splatx::knn3_mean_dists(pts.data(), n);
    const double grid_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
            .count();
    std::printf("  [knn anisotropic] n=%lld grid %.2f ms\n", static_cast<long long>(n),
                grid_ms);
    // Correctness: bit-identical against brute force (the same contract as the other scenes)
    const std::vector<float> ref = knn3_ref_brute(pts);
    size_t bad = 0;
    for (size_t i = 0; i < ref.size(); i++)
        if (got[i] != ref[i]) bad++;
    SX_CHECK(bad == 0);
    // The timing assertion (loose — an absolute upper bound): over 60 seconds before the fix vs ~ms
    // after. 2 seconds leaves generous margin for debug builds and slow machines (to prevent flakes)
    // — it only catches a recurrence of the pathology.
    SX_CHECK(grid_ms < 2000.0);
}

// ===== (c) The L1 loss/grad hand-check, the sign(0) convention, and the activation helpers =====
SX_TEST(l1_loss_grad_and_activation_utils) {
    // A 1x2 pixel RGB image (n_elems = H·W·3 = 6). diff = {+0.25, 0, -0.1, -0.5, +0.5, 0}
    const float r[6] = {0.5f, 0.2f, 0.9f, 0.0f, 1.0f, 0.3f};
    const float g[6] = {0.25f, 0.2f, 1.0f, 0.5f, 0.5f, 0.3f};
    L1LossResult res = l1_loss_and_grad(r, g, 6);
    // loss = (0.25 + 0 + 0.1 + 0.5 + 0.5 + 0)/6 = 0.225 (hand-check)
    SX_CHECK_NEAR(res.loss, 0.225f, 1e-6, 0);
    SX_CHECK(res.v_render_colors.size() == 6);
    // grad = sign(r-g)/6. **The sign(0) = 0 convention** (matching torch.sign) — indices 1 and 5 are
    // exactly 0
    const float s = 1.0f / 6.0f;
    const float expected[6] = {+s, 0.0f, -s, -s, +s, 0.0f};
    for (int i = 0; i < 6; i++) SX_CHECK(res.v_render_colors[i] == expected[i]);

    bool threw = false;
    try { (void)l1_loss_and_grad(r, g, 0); } catch (...) { threw = true; }
    SX_CHECK(threw);

    // The activation helpers — scales: the exp/log round-trip, opacities: the sigmoid/logit round-trip
    const float logs[3] = {0.0f, -1.0f, 2.0f};
    std::vector<float> lin = activate_scales(logs, 3);
    SX_CHECK(lin[0] == 1.0f); // exp(0) = 1 exactly
    SX_CHECK_NEAR(lin[1], std::exp(-1.0), 1e-6, 0);
    SX_CHECK_NEAR(lin[2], std::exp(2.0), 1e-6, 0);
    std::vector<float> logs2 = inv_activate_scales(lin.data(), 3);
    for (int i = 0; i < 3; i++) SX_CHECK_NEAR(logs2[i], logs[i], 1e-6, 1e-7);

    const float logits[3] = {0.0f, -2.1972246f, 3.0f};
    std::vector<float> op = activate_opacities(logits, 3);
    SX_CHECK(op[0] == 0.5f);                 // σ(0) = 0.5 exactly
    SX_CHECK_NEAR(op[1], 0.1f, 1e-6, 0);     // σ(logit(0.1)) = 0.1
    SX_CHECK_NEAR(op[2], 1.0 / (1.0 + std::exp(-3.0)), 1e-6, 0);
    std::vector<float> logits2 = inv_activate_opacities(op.data(), 3);
    for (int i = 0; i < 3; i++) SX_CHECK_NEAR(logits2[i], logits[i], 1e-5, 1e-6);

    // The scalar helper hand-check, plus a re-confirmation of the direction of the kernel input
    // convention (internal logit/log → linear for the kernel)
    SX_CHECK_NEAR(logit_f32(0.1f), -2.1972246f, 1e-6, 0); // ln(1/9)
    SX_CHECK_NEAR(sigmoid_f32(-2.1972246f), 0.1f, 1e-6, 0);
    threw = false;
    try { (void)logit_f32(0.0f); } catch (...) { threw = true; } // p ∈ (0,1) is enforced
    SX_CHECK(threw);
    threw = false;
    try { (void)logit_f32(1.0f); } catch (...) { threw = true; }
    SX_CHECK(threw);
}

// ===== Task 7 — Trainer E2E (D7 assembly and host round-trip chaining, D11) =====
// Only the two tests below use the Metal backend (raster bwd is Metal-only — the 1a rule).
#include "TestUtils.h" // shared_metal_backend
#include <random>

namespace {

// A synthetic single-camera scene — the camera at the origin looking down +z, 64×64, f=70.
TrainingCamera make_smoke_camera() {
    TrainingCamera cam;
    for (int i = 0; i < 16; i++) cam.viewmat[i] = (i % 5 == 0) ? 1.0f : 0.0f; // identity
    const float K[9] = {70.0f, 0.0f, 32.0f, 0.0f, 70.0f, 32.0f, 0.0f, 0.0f, 1.0f};
    for (int i = 0; i < 9; i++) cam.K[i] = K[i];
    cam.width = 64;
    cam.height = 64;
    return cam;
}

// The GT scene: a 4×4 grid of Gaussians on the z=3 plane (covering screen pixels 18..46), isotropic
// scale 0.13, opacity 0.85, and colors from an index-based palette. sh_degree defaults to 0 (the
// smoke test uses DC only — the higher bands are covered by the garden smoke test). When
// sh_degree >= 1, shN is filled with a small-amplitude deterministic pattern (a different value per
// band and channel) — this drives the coeffs assembly/disassembly and the shN adam group in the
// Task 5 device-resident comparison.
SplatCloud make_smoke_gt_cloud(int sh_degree = 0) {
    SplatCloud c;
    c.count = 16;
    c.sh_degree = sh_degree;
    c.means.resize(16 * 3);
    c.scales.assign(16 * 3, std::log(0.13f));
    c.quats.assign(16 * 4, 0.0f);
    c.opacities.assign(16, logit_f32(0.85f));
    c.sh0.resize(16 * 3);
    c.shN.resize(static_cast<size_t>(16) * c.sh_rest_coeffs() * 3); // deg 0 → an empty vector
    for (size_t j = 0; j < c.shN.size(); j++)
        c.shN[j] = 0.03f * static_cast<float>(static_cast<int>(j % 7) - 3);
    for (int i = 0; i < 16; i++) {
        const int gx = i % 4, gy = i / 4;
        c.means[i * 3 + 0] = -0.6f + 0.4f * static_cast<float>(gx);
        c.means[i * 3 + 1] = -0.6f + 0.4f * static_cast<float>(gy);
        c.means[i * 3 + 2] = 3.0f;
        c.quats[i * 4 + 0] = 1.0f;
        // The palette: each channel cycles through 0.15..0.9 — sh0 = (rgb01 − 0.5)/C0
        const float rgb[3] = {0.15f + 0.05f * static_cast<float>(i),
                              0.9f - 0.04f * static_cast<float>(i),
                              0.2f + 0.03f * static_cast<float>((i * 7) % 16)};
        for (int k = 0; k < 3; k++) c.sh0[i * 3 + k] = (rgb[k] - 0.5f) / kShC0;
    }
    return c;
}

} // namespace

// (d) The self-reconstruction smoke test — taking the GT render (a forward pass through the same
// pipeline) as ground truth, it asserts the L1 loss descends over 100 iterations from a
// deterministically perturbed initialization (the brief: at most half the starting value, and
// approximately monotone).
// The LR is raised for the smoke test (the default recipe is scaled for 30k iterations — that is the
// 300-iteration garden smoke test's job).
// This is the final consistency test of the signs and chains across the whole pipeline (including
// the activation gradients and the v_dirs reduction).
SX_TEST(trainer_self_reconstruction_loss_halves) {
    KernelBackend& b = sxtest::shared_metal_backend();
    const TrainingCamera cam = make_smoke_camera();
    const SplatCloud gt = make_smoke_gt_cloud();

    TrainConfig cfg;
    cfg.total_iters = 100;
    cfg.scene_scale = 1.0f; // a single camera — scene_scale_from_cameras also gives 1.0 (see the test below)
    cfg.lr_means = 2e-3f;
    cfg.lr_scales = 2e-2f;
    cfg.lr_quats = 5e-3f;
    cfg.lr_opacities = 5e-2f;
    cfg.lr_sh0 = 2e-2f;
    cfg.ssim_weight = 0.0f; // pins the regression to pure L1 (the default is 0.2 — the SSIM-on variant is a separate test)

    // The GT render — a forward pass through the same pipeline (self-reconstruction: the render is
    // itself the ground truth image)
    Trainer gt_renderer(b, gt, cfg);
    const std::vector<float> gt_img = gt_renderer.render(cam);
    SX_CHECK(gt_img.size() == static_cast<size_t>(cam.width) * cam.height * 3);

    // The perturbed initialization (a deterministic seed — the mt19937 algorithm is fixed by the
    // standard). quats are perturbed component-wise before normalization (the kernel normalizes
    // internally — this exercises the v_quats path end to end; a sign error makes it diverge)
    SplatCloud init = gt;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto& v : init.means) v += 0.05f * u(rng);
    for (auto& v : init.scales) v += 0.30f * u(rng);
    for (auto& v : init.quats) v += 0.05f * u(rng);
    for (auto& v : init.opacities) v += 0.70f * u(rng);
    for (auto& v : init.sh0) v += 0.60f * u(rng);

    Trainer tr(b, init, cfg);
    std::vector<float> losses;
    losses.reserve(cfg.total_iters);
    for (int it = 0; it < cfg.total_iters; it++) {
        const TrainStepResult r = tr.step(cam, gt_img.data(), it);
        SX_CHECK(std::isfinite(r.loss));
        SX_CHECK(r.n_isects > 0); // never silently passes an everything-culled regression
        losses.push_back(r.loss);
    }

    auto avg = [&](size_t from, size_t to) {
        double s = 0;
        for (size_t i = from; i < to; i++) s += losses[i];
        return s / static_cast<double>(to - from);
    };
    const double head10 = avg(0, 10), tail10 = avg(losses.size() - 10, losses.size());
    const double tail5 = avg(losses.size() - 5, losses.size());
    std::printf("  loss[0]=%.6f loss[50]=%.6f tail5=%.6f (head10=%.6f tail10=%.6f)\n",
                static_cast<double>(losses.front()), static_cast<double>(losses[50]), tail5,
                head10, tail10);
    SX_CHECK(losses.front() > 0.0f);          // did the perturbation actually produce a loss
    SX_CHECK(tail5 <= 0.5 * losses.front());  // at most half the starting value (the brief)
    SX_CHECK(tail10 < head10);                // approximately monotone descent (window averages)
}

// (e) relu_mask — blocks the gradient of channels where the SH post-op clamp fires (Phase 2 Task 1,
// corresponding to the clamp_min backward in gsplat rendering.py). For g0 the red channel has
// pre_clamp = C0·(-4)+0.5 < 0 → the clamp fires → v_colors is masked → v_sh0 is exactly 0 → the adam
// step leaves it bit-unchanged. green and blue have pre_clamp > 0 → the gradient passes through →
// the parameters move. **Channel selectivity** is the criterion — this catches regressions in both
// directions: a mask that blocks everything (no channel moves) and a missing mask (every channel
// moves).
SX_TEST(trainer_relu_mask_blocks_clamped_channel_grad) {
    KernelBackend& b = sxtest::shared_metal_backend();
    const TrainingCamera cam = make_smoke_camera();

    SplatCloud c;
    c.count = 1;
    c.sh_degree = 0;
    c.means = {0.0f, 0.0f, 3.0f};
    c.scales.assign(3, std::log(0.3f)); // the projected σ ≈ 70·0.3/3 = 7px — guarantees pixel coverage
    c.quats = {1.0f, 0.0f, 0.0f, 0.0f};
    c.opacities = {logit_f32(0.9f)};
    c.sh0 = {-4.0f, 0.5f, 0.8f};
    c.shN.clear();

    // Self-verification of the premise: red is the only channel where the clamp fires
    // (pre_clamp = C0·sh0 + 0.5)
    SX_CHECK(kShC0 * c.sh0[0] + 0.5f < 0.0f);
    SX_CHECK(kShC0 * c.sh0[1] + 0.5f > 0.0f && kShC0 * c.sh0[2] + 0.5f > 0.0f);

    TrainConfig cfg;
    cfg.total_iters = 10;
    cfg.ssim_weight = 0.0f; // pins the regression to pure L1 (the masking also applies to the combined grad — that is a separate test)
    const SplatCloud init = c;
    Trainer tr(b, std::move(c), cfg);
    const std::vector<float> gt(static_cast<size_t>(cam.width) * cam.height * 3, 0.3f);
    const TrainStepResult r = tr.step(cam, gt.data(), 0);
    SX_CHECK(r.n_isects > 0); // the Gaussian really takes part in rasterization (meaningless if everything is culled)

    const SplatCloud& after = tr.cloud();
    SX_CHECK(after.sh0[0] == init.sh0[0]); // the clamped channel: the grad is exactly 0 → adam leaves it bit-unchanged
    SX_CHECK(after.sh0[1] != init.sh0[1]); // a passing channel: it really moves (confirms the grad path survives)
    SX_CHECK(after.sh0[2] != init.sh0[2]);
}

// (f) A hand-check of the schedules, the activation gradient chain, and the guards — the Trainer's
// auxiliary surface (no GPU needed; compared against closed forms).
SX_TEST(trainer_schedules_chains_and_guards) {
    // scene_scale (the gsplat formula — the maximum distance from the centroid × 1.1):
    // R=I, t=±(1,0,0) → campos = ∓(1,0,0), centroid 0, maximum distance 1 → 1.1.
    // Three asymmetric cameras (campos x = -1, +1, +3 → centroid +1, maximum distance 2 vs a mean
    // distance of 4/3) verify that max is distinguished from mean (2.2 ≠ 1.4667).
    TrainingCamera a = make_smoke_camera(), b = make_smoke_camera(), c = make_smoke_camera();
    a.viewmat[3] = 1.0f;  // t = (+1,0,0) → campos (-1,0,0)
    b.viewmat[3] = -1.0f; // t = (-1,0,0) → campos (+1,0,0)
    c.viewmat[3] = -3.0f; // t = (-3,0,0) → campos (+3,0,0)
    SX_CHECK_NEAR(scene_scale_from_cameras({a, b}), 1.1f, 1e-6, 0);
    SX_CHECK_NEAR(scene_scale_from_cameras({a, b, c}), 2.2f, 1e-6, 0);
    SX_CHECK(scene_scale_from_cameras({a}) == 1.0f);      // a single camera → the 1.0 fallback
    SX_CHECK(scene_scale_from_cameras({a, a}) == 1.0f);   // degenerate (coincident centers) → the 1.0 fallback

    // The exponential decay of the means lr (ratio-based): lr·scene·0.01^(iter/total)
    TrainConfig cfg;
    cfg.total_iters = 100;
    cfg.scene_scale = 2.0f;
    cfg.lr_means = 1.6e-4f;
    SX_CHECK_NEAR(means_lr_at(cfg, 0), 3.2e-4, 1e-6, 0);           // the start = lr·scene
    SX_CHECK_NEAR(means_lr_at(cfg, 50), 3.2e-4 * 0.1, 1e-6, 0);    // halfway = ×0.01^0.5
    SX_CHECK_NEAR(means_lr_at(cfg, 100), 3.2e-4 * 0.01, 1e-6, 0);  // the end = ×1%

    // The SH degree schedule (ratio-based — the gsplat interval of 1/30): total 300 → interval 10
    TrainConfig scfg;
    scfg.total_iters = 300;
    SX_CHECK(sh_degree_at(scfg, 0, 3) == 0);
    SX_CHECK(sh_degree_at(scfg, 9, 3) == 0);
    SX_CHECK(sh_degree_at(scfg, 10, 3) == 1);
    SX_CHECK(sh_degree_at(scfg, 25, 3) == 2);
    SX_CHECK(sh_degree_at(scfg, 299, 3) == 3); // the cap = sh_degree
    SX_CHECK(sh_degree_at(scfg, 299, 0) == 0); // sh_degree 0 → always 0
    // Compared at the authoritative gsplat scale (simple_trainer.py:910 min(step//1000, 3),
    // interval=:165): total 30000 → interval 1000 — the promotion boundaries match the upstream
    // original iteration for iteration (the Task 4 ratio boundaries)
    TrainConfig gcfg;
    gcfg.total_iters = 30000;
    SX_CHECK(sh_degree_at(gcfg, 999, 3) == 0);
    SX_CHECK(sh_degree_at(gcfg, 1000, 3) == 1);
    SX_CHECK(sh_degree_at(gcfg, 2999, 3) == 2);
    SX_CHECK(sh_degree_at(gcfg, 3000, 3) == 3);
    SX_CHECK(sh_degree_at(gcfg, 29999, 3) == 3); // the upper bound holds

    // The activation gradient chain (task-6 handover 3): v_log_scale = v·scale_lin,
    // v_logit = v·σ(1−σ)
    std::vector<float> v_s = {2.0f, -3.0f};
    const float lin[2] = {0.5f, 2.0f};
    chain_scale_grads_inplace(v_s, lin);
    SX_CHECK(v_s[0] == 1.0f && v_s[1] == -6.0f);
    std::vector<float> v_o = {4.0f};
    const float act[1] = {0.25f}; // σ(1−σ) = 0.1875
    chain_opacity_grads_inplace(v_o, act);
    SX_CHECK(v_o[0] == 4.0f * 0.1875f);

    // The inv_activate_scales guard (a Minor task-6 handover item): scale <= 0 → throw (the domain
    // of log)
    bool threw = false;
    const float bad0[1] = {0.0f};
    try { (void)inv_activate_scales(bad0, 1); } catch (const std::invalid_argument&) { threw = true; }
    SX_CHECK(threw);
    threw = false;
    const float badn[1] = {-1.0f};
    try { (void)inv_activate_scales(badn, 1); } catch (const std::invalid_argument&) { threw = true; }
    SX_CHECK(threw);
    const float ok[1] = {2.0f};
    SX_CHECK_NEAR(inv_activate_scales(ok, 1)[0], std::log(2.0), 1e-6, 0);
}

// ===== Phase 2 Task 4 — the MCMC regularization gradients, the presets, and the CLI surface =====

// (g) A hand-check of the regularization gradients (comparing the denominators against gsplat
// losses.py:675–700), the mcmc presets (simple_trainer.py:1579–1587), and the meaning of the
// init_scale multiplier (simple_trainer.py:323). No GPU needed.
SX_TEST(trainer_reg_grads_and_mcmc_preset_hand_check) {
    // opacity_reg: opacities is 1D [N] (simple_trainer.py:332) → the mean denominator is N = 2.
    // σ'(logit) = σ(1−σ): σ=0.5 → 0.25, σ=0.25 → 0.1875. It is **added** to the existing gradient
    // (autograd accumulation).
    std::vector<float> v_o = {0.1f, -0.2f};
    const float act[2] = {0.5f, 0.25f};
    add_opacity_reg_grad_inplace(v_o, act, 0.01f);
    SX_CHECK_NEAR(v_o[0], 0.1f + 0.01f * 0.25f / 2.0f, 1e-7, 0);
    SX_CHECK_NEAR(v_o[1], -0.2f + 0.01f * 0.1875f / 2.0f, 1e-7, 0);

    // scale_reg: log_scales is [N,3] (simple_trainer.py:323) → torch .mean() averages over **every
    // element**, so the denominator is N·3 = 6 (losses.py:689–700). Had the denominator been N (=2),
    // the first element would be 0.005 — a factor of 3 away from the expected 0.01/6 below, so a
    // regression is caught unambiguously.
    std::vector<float> v_s = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    const float lin[6] = {1.0f, 2.0f, 3.0f, 0.5f, 4.0f, 6.0f};
    add_scale_reg_grad_inplace(v_s, lin, 0.01f);
    SX_CHECK_NEAR(v_s[0], 0.01 / 6.0, 1e-7, 0);
    SX_CHECK_NEAR(v_s[2], 0.01 * 3.0 / 6.0, 1e-7, 0);
    SX_CHECK_NEAR(v_s[5], 1.0 + 0.01 * 6.0 / 6.0, 1e-7, 0);

    // The gate: reg <= 0 → no-op (the same `> 0.0` gate as gsplat simple_trainer.py:993/:995)
    std::vector<float> v_g = {0.25f};
    add_opacity_reg_grad_inplace(v_g, act, 0.0f);
    add_scale_reg_grad_inplace(v_g, lin, -1.0f);
    SX_CHECK(v_g[0] == 0.25f);

    // The mcmc presets (simple_trainer.py:1579–1587): they change only the 4 designated fields plus
    // turning the strategy on; everything else is unchanged
    InitOptions io;
    TrainConfig cfg;
    cfg.lr_means = 9.0f; // a sentinel that watches for unintended changes
    apply_mcmc_presets(io, cfg);
    SX_CHECK(io.init_opacity == 0.5f); // :1583
    SX_CHECK(io.init_scale == 0.1f);   // :1584
    SX_CHECK(cfg.opacity_reg == 0.01f && cfg.scale_reg == 0.01f); // :1585–1586
    SX_CHECK(cfg.mcmc.enabled);        // :1587 strategy=MCMCStrategy
    SX_CHECK(cfg.lr_means == 9.0f && cfg.mcmc.cap_max == 1'000'000);
    SX_CHECK(io.max_points == 50000 && io.sh_degree == 3);

    // init_scale is a **multiplier** on the 3-NN distance (log(dist·init_scale)) — for the same input
    // the scales shift by exactly +log(0.1), while means, quats, and the rest are unchanged.
    // opacities = logit(0.5) = 0 (exactly).
    TrainingInput ti;
    ti.seed_points = {0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 4};
    ti.seed_rgb = {255, 0, 128, 0, 0, 0, 255, 255, 255, 10, 20, 30};
    InitOptions base;
    base.max_points = 100;
    const SplatCloud plain = init_splats(ti, base);
    InitOptions preset = base;
    TrainConfig unused;
    apply_mcmc_presets(preset, unused);
    const SplatCloud mc = init_splats(ti, preset);
    SX_CHECK(mc.count == plain.count);
    for (size_t i = 0; i < mc.scales.size(); i++)
        SX_CHECK_NEAR(mc.scales[i], plain.scales[i] + std::log(0.1f), 1e-6, 1e-6);
    for (float v : mc.opacities) SX_CHECK(v == 0.0f);
    for (size_t i = 0; i < mc.means.size(); i++) SX_CHECK(mc.means[i] == plain.means[i]);

    // init_scale <= 0 or NaN → throw (the domain of log)
    for (float bad : {0.0f, -0.5f, std::numeric_limits<float>::quiet_NaN()}) {
        InitOptions ib = base;
        ib.init_scale = bad;
        bool threw = false;
        try { (void)init_splats(ti, ib); } catch (const std::invalid_argument&) { threw = true; }
        SX_CHECK(threw);
    }
}

// (h) An E2E test of where the regularization gradients are integrated into step() (immediately
// before adam) — because gt is the model's own render (bit-identical), the L1 gradient is exactly 0
// (the sign(0)=0 convention) → the only training signal is the reg terms. After one step, only the
// opacity and scale parameters descend (reg pulls σ and exp down), while means, quats, and sh0 have
// gradient 0 → adam moves them by 0 → they are bit-unchanged. The control group (reg=0) leaves every
// parameter bit-unchanged — verifying the gate.
SX_TEST(trainer_reg_grads_apply_before_adam_e2e) {
    KernelBackend& b = sxtest::shared_metal_backend();
    const TrainingCamera cam = make_smoke_camera();

    SplatCloud c;
    c.count = 1;
    c.sh_degree = 0;
    c.means = {0.0f, 0.0f, 3.0f};
    c.scales.assign(3, std::log(0.3f)); // the projected σ ≈ 7px — guarantees valid (radii>0)
    c.quats = {1.0f, 0.0f, 0.0f, 0.0f};
    c.opacities = {logit_f32(0.9f)};
    c.sh0 = {0.1f, 0.2f, 0.3f}; // pre_clamp > 0 in every channel — excludes relu_mask interference
    c.shN.clear();

    TrainConfig cfg;
    cfg.total_iters = 10;
    cfg.opacity_reg = 0.01f;
    cfg.scale_reg = 0.01f;
    cfg.ssim_weight = 0.0f; // pinned to pure L1 — the premise that "the only training signal is reg" (the L1 grad is exactly 0)

    Trainer renderer(b, c, cfg);
    const std::vector<float> gt = renderer.render(cam); // the same pipeline → a bit-identical render

    const SplatCloud init = c;
    Trainer tr(b, c, cfg);
    const TrainStepResult r = tr.step(cam, gt.data(), 0);
    SX_CHECK(r.n_isects > 0);
    SX_CHECK(r.loss == 0.0f); // self-verification of the self-render consistency (the premise that there is no L1 signal)

    const SplatCloud& after = tr.cloud();
    SX_CHECK(after.opacities[0] < init.opacities[0]); // the logit descends from reg alone
    for (int k = 0; k < 3; k++) SX_CHECK(after.scales[k] < init.scales[k]); // log_s descends
    for (int k = 0; k < 3; k++) SX_CHECK(after.means[k] == init.means[k]); // grad 0 → unchanged
    for (int k = 0; k < 4; k++) SX_CHECK(after.quats[k] == init.quats[k]);
    for (int k = 0; k < 3; k++) SX_CHECK(after.sh0[k] == init.sh0[k]);

    // The control group: reg = 0 → no training signal at all → every parameter is bit-unchanged
    // (catching regressions in both the gate and the addition point)
    TrainConfig cfg0 = cfg;
    cfg0.opacity_reg = 0.0f;
    cfg0.scale_reg = 0.0f;
    Trainer tr0(b, init, cfg0);
    (void)tr0.step(cam, gt.data(), 0);
    const SplatCloud& same = tr0.cloud();
    SX_CHECK(same.opacities[0] == init.opacities[0]);
    for (int k = 0; k < 3; k++) SX_CHECK(same.scales[k] == init.scales[k]);
}

// (i) The self-reconstruction smoke test — the MCMC-on variant (Task 4 brief, item 4): with a small
// cap (16→24), reg 0.01 matching the preset, and the default ratio-based schedule (100 total
// iterations → start 2 / stop 83 / every 1), it checks that convergence holds even while refine
// (growth via add) and the per-step noise are driven on the real training path.
// The LR and perturbation are the same as in (d) — and the criteria are the same as with mcmc off
// (at most half, and approximately monotone).
SX_TEST(trainer_self_reconstruction_mcmc_on_converges) {
    KernelBackend& b = sxtest::shared_metal_backend();
    const TrainingCamera cam = make_smoke_camera();
    const SplatCloud gt = make_smoke_gt_cloud();

    TrainConfig cfg;
    cfg.total_iters = 100;
    cfg.scene_scale = 1.0f;
    cfg.lr_means = 2e-3f;
    cfg.lr_scales = 2e-2f;
    cfg.lr_quats = 5e-3f;
    cfg.lr_opacities = 5e-2f;
    cfg.lr_sh0 = 2e-2f;
    cfg.ssim_weight = 0.0f; // pinned to pure L1 — isolates the mcmc on/off difference alone (the same conditions as (d))

    Trainer gt_renderer(b, gt, cfg); // the gt render is independent of mcmc (forward only)
    const std::vector<float> gt_img = gt_renderer.render(cam);

    // MCMC on plus the preset regularization (the init presets are not applied here — the perturbed
    // initialization must match (d) so that only the mcmc on/off difference is isolated. The
    // init_opa and init_scale presets themselves are verified by (g).)
    cfg.opacity_reg = 0.01f;
    cfg.scale_reg = 0.01f;
    cfg.mcmc.enabled = true;
    cfg.mcmc.cap_max = 24; // a small arena — the growth trajectory 16→…→24 (the cap)
    cfg.mcmc.seed = 20260711;

    SplatCloud init = gt;
    std::mt19937 rng(12345); // the same deterministic perturbation as (d)
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto& v : init.means) v += 0.05f * u(rng);
    for (auto& v : init.scales) v += 0.30f * u(rng);
    for (auto& v : init.quats) v += 0.05f * u(rng);
    for (auto& v : init.opacities) v += 0.70f * u(rng);
    for (auto& v : init.sh0) v += 0.60f * u(rng);

    Trainer tr(b, init, cfg);
    std::vector<float> losses;
    losses.reserve(cfg.total_iters);
    for (int it = 0; it < cfg.total_iters; it++) {
        const TrainStepResult r = tr.step(cam, gt_img.data(), it);
        SX_CHECK(std::isfinite(r.loss));
        SX_CHECK(r.n_isects > 0);
        SX_CHECK(tr.cloud().count <= 24); // MUST NOT exceed the cap (the arena upper bound)
        losses.push_back(r.loss);
    }

    SX_CHECK(tr.cloud().count == 24); // the refine schedule really runs — it grows to the cap (not vacuous)

    auto avg = [&](size_t from, size_t to) {
        double s = 0;
        for (size_t i = from; i < to; i++) s += losses[i];
        return s / static_cast<double>(to - from);
    };
    const double head10 = avg(0, 10), tail10 = avg(losses.size() - 10, losses.size());
    const double tail5 = avg(losses.size() - 5, losses.size());
    std::printf("  [mcmc on] loss[0]=%.6f tail5=%.6f (head10=%.6f tail10=%.6f) N=%lld\n",
                static_cast<double>(losses.front()), tail5, head10, tail10,
                static_cast<long long>(tr.cloud().count));
    SX_CHECK(losses.front() > 0.0f);
    SX_CHECK(tail5 <= 0.5 * losses.front()); // convergence holds with mcmc on too (brief item 4)
    SX_CHECK(tail10 < head10);               // approximately monotone descent
}

// ===== Phase 3 Task 3 — SSIM loss integration (loss = (1-λ)·L1 + λ·(1-SSIM), the gsplat lerp) =====

// (j) A hand-check of the combined loss, the λ=0 gate, and the guards — it compares the loss
// reported by step() against an independent computation (l1_loss_and_grad plus backend.ssim_fwd —
// a render through the same pipeline is bit-identical, the same premise as (h)) to confirm it is
// exactly (1-λ)·L1 + λ·(1-ssim). λ=0 bypasses the SSIM path entirely — the reported loss is
// bit-identical to pure L1 (backward compatibility). For λ>0, that the gradient path is genuinely
// combined is observed through parameter divergence (after one step, sh0 differs from the λ=0 run).
// Guards for H/W < 11 and λ ∉ [0,1].
SX_TEST(trainer_ssim_loss_combination_and_gate) {
    KernelBackend& b = sxtest::shared_metal_backend();
    const TrainingCamera cam = make_smoke_camera(); // 64×64 >= 11 — SSIM valid OK
    const SplatCloud gt_cloud = make_smoke_gt_cloud();

    TrainConfig cfg;
    cfg.total_iters = 10;
    SX_CHECK(cfg.ssim_weight == 0.2f); // the default = the gsplat ssim_lambda of 0.2 (brief item 1)

    // The independent expected value: a render through the same pipeline (bit-identical), with L1
    // and SSIM computed outside step()
    Trainer renderer(b, gt_cloud, cfg);
    const std::vector<float> render = renderer.render(cam);
    const std::vector<float> gt(render.size(), 0.35f); // a constant GT — guarantees L1 > 0 and ssim < 1
    const L1LossResult l1 =
        l1_loss_and_grad(render.data(), gt.data(), static_cast<int64_t>(render.size()));
    SsimInputs si;
    si.img1 = render.data();
    si.img2 = gt.data();
    si.I = 1;
    si.H = cam.height;
    si.W = cam.width;
    si.C = 3;
    const SsimFwdOutputs so = b.ssim_fwd(si);
    SX_CHECK(l1.loss > 0.0f && so.ssim < 1.0);       // self-verification that this is not degenerate
    const float lam = cfg.ssim_weight;
    const float expected =
        (1.0f - lam) * l1.loss + lam * (1.0f - static_cast<float>(so.ssim));

    Trainer tr(b, gt_cloud, cfg); // λ = 0.2 (the default)
    const TrainStepResult r = tr.step(cam, gt.data(), 0);
    SX_CHECK(r.n_isects > 0);
    SX_CHECK_NEAR(r.loss, expected, 1e-7, 0); // the same f32 combination formula as inside step — effectively bit-level

    // The λ=0 gate: the reported loss is bit-identical to pure L1 (the SSIM kernel is never called —
    // backward compatible with the previous behavior)
    TrainConfig cfg0 = cfg;
    cfg0.ssim_weight = 0.0f;
    Trainer tr0(b, gt_cloud, cfg0);
    const TrainStepResult r0 = tr0.step(cam, gt.data(), 0);
    SX_CHECK(r0.loss == l1.loss);

    // Demonstrating that the gradients are combined: after one step, λ=0.2 and λ=0 MUST give
    // different sh0 (v_DSSIM really reaches adam — if only the loss were combined while the gradient
    // stayed pure L1, this catches it)
    bool sh0_diverged = false;
    for (size_t i = 0; i < tr.cloud().sh0.size() && !sh0_diverged; i++)
        sh0_diverged = tr.cloud().sh0[i] != tr0.cloud().sh0[i];
    SX_CHECK(sh0_diverged);

    // Guard 1: λ > 0 with H/W < 11 → the ssim_fwd validator throws (a tiny image — the caution in
    // the brief)
    TrainingCamera tiny = make_smoke_camera();
    tiny.width = 8;
    tiny.height = 8;
    tiny.K[2] = 4.0f; // keeps cx/cy centered (this is the throw path, so the render content is irrelevant)
    tiny.K[5] = 4.0f;
    Trainer tr_tiny(b, gt_cloud, cfg);
    const std::vector<float> tiny_gt(static_cast<size_t>(8 * 8 * 3), 0.35f);
    bool threw = false;
    try { (void)tr_tiny.step(tiny, tiny_gt.data(), 0); }
    catch (const std::invalid_argument&) { threw = true; }
    SX_CHECK(threw);
    // With λ=0 the same tiny image can still be trained as before (demonstrating the SSIM path is
    // bypassed)
    Trainer tr_tiny0(b, gt_cloud, cfg0);
    SX_CHECK(std::isfinite(tr_tiny0.step(tiny, tiny_gt.data(), 0).loss));

    // Guard 2: ssim_weight ∉ [0,1] (including NaN) → the ctor throws
    for (float bad : {-0.1f, 1.5f, std::numeric_limits<float>::quiet_NaN()}) {
        TrainConfig cb = cfg;
        cb.ssim_weight = bad;
        threw = false;
        try { Trainer t(b, gt_cloud, cb); }
        catch (const std::invalid_argument&) { threw = true; }
        SX_CHECK(threw);
    }
}

// (k) The self-reconstruction smoke test — the SSIM-on variant (brief item 2): with the same LR and
// perturbation as (d), it checks that training on the combined loss with ssim_weight 0.2 (the
// default) still converges. The criteria are also the same as (d) (at most half the starting value
// measured on the combined loss, and approximately monotone) — demonstrating that the SSIM gradient
// path (the ssim_fwd → bwd → raster bwd chain) runs without diverging in a real training loop.
SX_TEST(trainer_self_reconstruction_ssim_on_converges) {
    KernelBackend& b = sxtest::shared_metal_backend();
    const TrainingCamera cam = make_smoke_camera();
    const SplatCloud gt = make_smoke_gt_cloud();

    TrainConfig cfg;
    cfg.total_iters = 100;
    cfg.scene_scale = 1.0f;
    cfg.lr_means = 2e-3f;
    cfg.lr_scales = 2e-2f;
    cfg.lr_quats = 5e-3f;
    cfg.lr_opacities = 5e-2f;
    cfg.lr_sh0 = 2e-2f;
    cfg.ssim_weight = 0.2f; // the default, but stated explicitly — it is the subject of this test

    Trainer gt_renderer(b, gt, cfg);
    const std::vector<float> gt_img = gt_renderer.render(cam);

    SplatCloud init = gt;
    std::mt19937 rng(12345); // the same deterministic perturbation as (d)
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto& v : init.means) v += 0.05f * u(rng);
    for (auto& v : init.scales) v += 0.30f * u(rng);
    for (auto& v : init.quats) v += 0.05f * u(rng);
    for (auto& v : init.opacities) v += 0.70f * u(rng);
    for (auto& v : init.sh0) v += 0.60f * u(rng);

    Trainer tr(b, init, cfg);
    std::vector<float> losses;
    losses.reserve(cfg.total_iters);
    for (int it = 0; it < cfg.total_iters; it++) {
        const TrainStepResult r = tr.step(cam, gt_img.data(), it);
        SX_CHECK(std::isfinite(r.loss));
        SX_CHECK(r.n_isects > 0);
        losses.push_back(r.loss);
    }

    auto avg = [&](size_t from, size_t to) {
        double s = 0;
        for (size_t i = from; i < to; i++) s += losses[i];
        return s / static_cast<double>(to - from);
    };
    const double head10 = avg(0, 10), tail10 = avg(losses.size() - 10, losses.size());
    const double tail5 = avg(losses.size() - 5, losses.size());
    std::printf("  [ssim on] loss[0]=%.6f tail5=%.6f (head10=%.6f tail10=%.6f)\n",
                static_cast<double>(losses.front()), tail5, head10, tail10);
    SX_CHECK(losses.front() > 0.0f);
    SX_CHECK(tail5 <= 0.5 * losses.front()); // convergence holds on the combined loss too
    SX_CHECK(tail10 < head10);               // approximately monotone descent
}

// ===== Phase 3 Task 4 — anti-aliasing (Mip compensation) training integration =====

// (l) A hand-check of the AA gradient chain plus the equivalence of the application point (brief
// item 2). The authoritative gsplat source: rendering.py :864–865
// `opacities = opacities * compensations` sits **before** isect_tiles (:884–896) — so both isect
// (the AccuTile level set) and raster consume opacity_aa = σ(op)·comp.
// ① A hand-computed product-rule chain (all values exactly representable in binary — compared at the
//    bit level): v_comp = v_op_aa·σ, v_op = v_op_aa·comp, then the σ′ chain gives
//    v_logit = v_op_aa·comp·σ(1−σ).
// ② Equivalence of the application point: render(aa on, opacity o) ≈ render(classic, opacity
//    σ⁻¹(σ(o)·comp)) — matching the classic render of a pre-multiplied opacity is what proves that
//    the proj radius (multiplied inside the kernel), the isect level set, and raster all use
//    opacity_aa (if even one of those points used the non-AA opacity, the two renders would
//    diverge). The only error is the logit/sigmoid round-trip (~1e-7).
// ③ Not vacuous: comp < 1 (the eps2d blur — always, by definition) → the aa-on render really does
//    differ from classic.
// ④ The zero-gradient fixed point: gt = the model's own render (aa on) → the L1 gradient is exactly
//    0 → after one step every parameter is bit-unchanged (the AA chain preserves 0 as 0 — this
//    guards against a regression where multiplying by comp=0 in a culled slot contaminates with
//    NaN).
SX_TEST(trainer_aa_grad_chain_and_apply_point_equivalence) {
    // ① The hand computation — every value is exactly representable in binary (0.5, 0.25, 0.75,
    // 0.1875, and so on) → asserted as bit-identical
    std::vector<float> v_op = {0.5f, -0.25f};
    const float act[2] = {0.5f, 0.25f};   // σ(logit)
    const float comp[2] = {0.5f, 0.75f};  // the compensation factors
    const std::vector<float> v_comp = chain_aa_opacity_grads_inplace(v_op, act, comp);
    SX_CHECK(v_comp.size() == 2);
    SX_CHECK(v_comp[0] == 0.25f);      // v_op_aa·σ = 0.5·0.5
    SX_CHECK(v_comp[1] == -0.0625f);   // -0.25·0.25
    SX_CHECK(v_op[0] == 0.25f);        // v_op_aa·comp = 0.5·0.5
    SX_CHECK(v_op[1] == -0.1875f);     // -0.25·0.75
    chain_opacity_grads_inplace(v_op, act); // σ′ = σ(1−σ) = {0.25, 0.1875}
    SX_CHECK(v_op[0] == 0.0625f);      // v_op_aa·comp·σ(1−σ) = 0.5·0.5·0.25
    SX_CHECK(v_op[1] == -0.03515625f); // -0.25·0.75·0.1875 (the d/d logit formula in the brief)
    // Reproducing a culled slot (comp = 0, v_op_aa = 0): both are exactly 0 — no NaN and no contamination
    std::vector<float> v_cull = {0.0f};
    const float act1[1] = {0.9f}, comp0[1] = {0.0f};
    const std::vector<float> vc0 = chain_aa_opacity_grads_inplace(v_cull, act1, comp0);
    SX_CHECK(vc0[0] == 0.0f && v_cull[0] == 0.0f);

    // ② Equivalence of the application point (Metal E2E)
    KernelBackend& b = sxtest::shared_metal_backend();
    const TrainingCamera cam = make_smoke_camera();
    const SplatCloud gt = make_smoke_gt_cloud();
    const int64_t N = gt.count;

    TrainConfig cfg_aa;
    cfg_aa.total_iters = 10;
    cfg_aa.ssim_weight = 0.0f;
    cfg_aa.antialiased = true;
    TrainConfig cfg_classic = cfg_aa;
    cfg_classic.antialiased = false;

    // Computes comp independently through a direct proj fwd call (using the same input convention as
    // the internal Trainer path — linear scales and post-sigmoid opacities, with eps2d and the rest
    // matched to the TrainConfig defaults)
    const std::vector<float> scales_lin = activate_scales(gt.scales.data(), N * 3);
    const std::vector<float> opacs_act = activate_opacities(gt.opacities.data(), N);
    ProjectionFusedInputs pin;
    pin.means = gt.means.data();
    pin.quats = gt.quats.data();
    pin.scales = scales_lin.data();
    pin.opacities = opacs_act.data();
    pin.viewmats = cam.viewmat;
    pin.Ks = cam.K;
    pin.N = N;
    pin.C = 1;
    pin.width = cam.width;
    pin.height = cam.height;
    pin.eps2d = cfg_aa.eps2d;
    pin.near_plane = cfg_aa.near_plane;
    pin.far_plane = cfg_aa.far_plane;
    pin.radius_clip = cfg_aa.radius_clip;
    pin.calc_compensations = true;
    const ProjectionFusedOutputs po = b.projection_fused_fwd(pin);
    SX_CHECK(po.compensations.size() == static_cast<size_t>(N));
    for (int64_t i = 0; i < N; i++) { // self-verification of the premise that every slot is valid, plus comp ∈ (0,1)
        SX_CHECK(po.radii[i * 2] > 0 && po.radii[i * 2 + 1] > 0);
        SX_CHECK(po.compensations[i] > 0.0f && po.compensations[i] < 1.0f);
    }

    // The classic arm: opacity is pre-multiplied as σ⁻¹(σ(op)·comp) (the same effective opacity as AA)
    SplatCloud pre = gt;
    for (int64_t i = 0; i < N; i++)
        pre.opacities[i] = logit_f32(opacs_act[i] * po.compensations[i]);

    Trainer tr_aa(b, gt, cfg_aa);
    Trainer tr_pre(b, pre, cfg_classic);
    Trainer tr_plain(b, gt, cfg_classic);
    const std::vector<float> img_aa = tr_aa.render(cam);
    const std::vector<float> img_pre = tr_pre.render(cam);
    const std::vector<float> img_plain = tr_plain.render(cam);
    SX_CHECK(img_aa.size() == img_pre.size() && img_aa.size() == img_plain.size());
    float max_diff_pre = 0.0f, max_diff_plain = 0.0f;
    for (size_t i = 0; i < img_aa.size(); i++) {
        max_diff_pre = std::max(max_diff_pre, std::fabs(img_aa[i] - img_pre[i]));
        max_diff_plain = std::max(max_diff_plain, std::fabs(img_aa[i] - img_plain[i]));
    }
    std::printf("  [aa] max|aa - premul| = %.3g, max|aa - classic| = %.3g\n",
                static_cast<double>(max_diff_pre), static_cast<double>(max_diff_plain));
    SX_CHECK(max_diff_pre <= 5e-6f);   // equivalent (only the logit/sigmoid round-trip error)
    SX_CHECK(max_diff_plain > 1e-3f);  // ③ not vacuous — comp < 1 really shows up in the render

    // ④ The zero-gradient fixed point (aa on): gt = the model's own render → after one step every
    // parameter is bit-unchanged
    const TrainStepResult r0 = tr_aa.step(cam, img_aa.data(), 0);
    SX_CHECK(r0.loss == 0.0f);
    SX_CHECK(r0.n_isects > 0);
    const SplatCloud& after = tr_aa.cloud();
    for (size_t i = 0; i < gt.means.size(); i++) SX_CHECK(after.means[i] == gt.means[i]);
    for (size_t i = 0; i < gt.scales.size(); i++) SX_CHECK(after.scales[i] == gt.scales[i]);
    for (size_t i = 0; i < gt.opacities.size(); i++)
        SX_CHECK(after.opacities[i] == gt.opacities[i]);
    // The default gate: antialiased defaults to false (the gsplat default is "classic" — the
    // conditions of the published bench)
    SX_CHECK(TrainConfig{}.antialiased == false);
}

// (m) The self-reconstruction smoke test — the AA-on variant (brief item 2): with the same LR and
// perturbation as (d) and antialiased=true (pure L1, so only the AA path is isolated), it checks
// that the whole chain — fwd (opacity_aa) and bwd (v_comp → injected into proj bwd via
// kProjCalcComp → contributions to v_scales, v_quats, and v_means, plus
// v_logit = v_op_aa·comp·σ′) — still converges in a real training loop. The criteria are the same
// as (d).
SX_TEST(trainer_self_reconstruction_aa_on_converges) {
    KernelBackend& b = sxtest::shared_metal_backend();
    const TrainingCamera cam = make_smoke_camera();
    const SplatCloud gt = make_smoke_gt_cloud();

    TrainConfig cfg;
    cfg.total_iters = 100;
    cfg.scene_scale = 1.0f;
    cfg.lr_means = 2e-3f;
    cfg.lr_scales = 2e-2f;
    cfg.lr_quats = 5e-3f;
    cfg.lr_opacities = 5e-2f;
    cfg.lr_sh0 = 2e-2f;
    cfg.ssim_weight = 0.0f;  // pure L1 — isolates the AA on/off difference alone (the same conditions as (d))
    cfg.antialiased = true;  // the subject of this test itself (the gt render uses the same cfg — an
                             // AA-pipeline self-reconstruction: the ground truth is itself an
                             // opacity_aa render)

    Trainer gt_renderer(b, gt, cfg);
    const std::vector<float> gt_img = gt_renderer.render(cam);

    SplatCloud init = gt;
    std::mt19937 rng(12345); // the same deterministic perturbation as (d)
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto& v : init.means) v += 0.05f * u(rng);
    for (auto& v : init.scales) v += 0.30f * u(rng);
    for (auto& v : init.quats) v += 0.05f * u(rng);
    for (auto& v : init.opacities) v += 0.70f * u(rng);
    for (auto& v : init.sh0) v += 0.60f * u(rng);

    Trainer tr(b, init, cfg);
    std::vector<float> losses;
    losses.reserve(cfg.total_iters);
    for (int it = 0; it < cfg.total_iters; it++) {
        const TrainStepResult r = tr.step(cam, gt_img.data(), it);
        SX_CHECK(std::isfinite(r.loss));
        SX_CHECK(r.n_isects > 0);
        losses.push_back(r.loss);
    }

    auto avg = [&](size_t from, size_t to) {
        double s = 0;
        for (size_t i = from; i < to; i++) s += losses[i];
        return s / static_cast<double>(to - from);
    };
    const double head10 = avg(0, 10), tail10 = avg(losses.size() - 10, losses.size());
    const double tail5 = avg(losses.size() - 5, losses.size());
    std::printf("  [aa on] loss[0]=%.6f tail5=%.6f (head10=%.6f tail10=%.6f)\n",
                static_cast<double>(losses.front()), tail5, head10, tail10);
    SX_CHECK(losses.front() > 0.0f);
    SX_CHECK(tail5 <= 0.5 * losses.front()); // convergence holds with AA on too (brief item 2)
    SX_CHECK(tail10 < head10);               // approximately monotone descent
}

// ===== Phase 2 Task 5 — the PSNR hand-check (10·log10(1/mse), normalized to 0..1, MSE over every
// element) =====
SX_TEST(psnr_manual_formula) {
    const int64_t n = 4 * 5 * 3; // an even element count of the form W·H·C
    // A constant difference of 0.1 → mse = 0.01 → exactly 20 dB
    std::vector<float> a(static_cast<size_t>(n), 0.5f), b(static_cast<size_t>(n), 0.6f);
    SX_CHECK_NEAR(psnr(a.data(), b.data(), n), 20.0, 0, 1e-4);
    // A constant difference of 0.5 → mse = 0.25 → 10·log10(4) = 6.0205999…
    std::vector<float> c(static_cast<size_t>(n), 0.25f), d(static_cast<size_t>(n), 0.75f);
    SX_CHECK_NEAR(psnr(c.data(), d.data(), n), 6.02059991, 0, 1e-4);
    // Mixed (only half differ, by 0.2) → mse = 0.5·0.04 = 0.02 → 10·log10(50) = 16.9897000…
    // — this drives the fact that the MSE averages over **every pixel and channel** (averaging over
    // only the differing pixels would give 13.98)
    std::vector<float> e(static_cast<size_t>(n), 0.5f), g(static_cast<size_t>(n), 0.5f);
    for (int64_t i = 0; i < n / 2; i++) g[static_cast<size_t>(i)] = 0.7f;
    SX_CHECK_NEAR(psnr(e.data(), g.data(), n), 16.98970004, 0, 1e-4);
    // Identical images → +inf (mse == 0)
    SX_CHECK(std::isinf(psnr(a.data(), a.data(), n)));
    // Symmetry
    SX_CHECK(psnr(a.data(), b.data(), n) == psnr(b.data(), a.data(), n));
    // Failure paths: null, or a non-positive element count → throw
    bool threw = false;
    try { (void)psnr(nullptr, b.data(), n); } catch (...) { threw = true; }
    SX_CHECK(threw);
    threw = false;
    try { (void)psnr(a.data(), b.data(), 0); } catch (...) { threw = true; }
    SX_CHECK(threw);
}

// ===== color_correct Task 1 — numerical comparison against gsplat color_correct.py:98–149 (a pin)
// =====
// The expected values were produced by actually running the reference: the outputs obtained by
// feeding the f32 inputs below, unchanged, into color_correct_affine in
// research/gsplat/gsplat/color_correct.py (torch 2.9.1, f64) (the generation script and the measured
// error are in internal measurement records). The per-channel fit is non-trivial, with
// a ≈ (1.0209, 0.8063, 1.0755) and b ≈ (0.1087, 0.0978, −0.0136), and the clip fires on 2 elements
// (18→0, 19→1) — so this drives the formula, the guards, and the clip all at once.
SX_TEST(color_correct_affine_gsplat_pin) {
    // 12 pixels (4×3) of interleaved RGB. img is ref distorted with a per-channel gain and offset
    // (R 1.2/+0.05, G 0.8/+0.10, B 1.1/−0.03) plus a ±0.01 perturbation plus a [0,1] clip — imitating
    // the shape of real eval usage.
    const std::vector<float> ref = {
        0.10f, 0.20f, 0.30f, 0.90f, 0.80f, 0.70f, 0.50f, 0.40f, 0.60f,
        0.20f, 0.90f, 0.10f, 0.70f, 0.30f, 0.80f, 0.40f, 0.60f, 0.50f,
        0.00f, 1.00f, 0.20f, 0.80f, 0.10f, 0.90f, 0.30f, 0.50f, 0.40f,
        0.60f, 0.70f, 0.00f, 1.00f, 0.00f, 0.60f, 0.50f, 0.30f, 0.90f,
    };
    const std::vector<float> img = {
        0.18f, 0.26f, 0.30f, 1.00f, 0.75f, 0.75f, 0.64f, 0.41f, 0.62f,
        0.30f, 0.83f, 0.09f, 0.88f, 0.34f, 0.85f, 0.53f, 0.57f, 0.51f,
        0.06f, 0.91f, 0.20f, 1.00f, 0.18f, 0.96f, 0.42f, 0.51f, 0.42f,
        0.76f, 0.65f, 0.00f, 1.00f, 0.11f, 0.64f, 0.66f, 0.33f, 0.95f,
    };
    const std::vector<double> expected = {
        0.069827857058457454, 0.20118202856794473, 0.29156792876805337,
        0.87303354170276992, 0.80889252828758951, 0.70998177058043166,
        0.52040663867339654, 0.38721585878432574, 0.5891066619550982,
        0.18737015806824855, 0.90811054642859279, 0.096308122990731423,
        0.75549124069297879, 0.30040008367054782, 0.80296264894803093,
        0.41265951968377307, 0.5856519320279322, 0.48682770683492088,
        0.0, 1.0, 0.19858706425568146,
        0.87303354170276992, 0.10196401042694136, 0.90524154864729911,
        0.30491242988609457, 0.51123839994137976, 0.4031449329303543,
        0.63794893968318767, 0.68486995016893559, 0.012625349086164859,
        0.87303354170276992, 0.015148216832363604, 0.6077028154602544,
        0.53999706109762191, 0.28799784064332234, 0.89594347189472101,
    };
    const int64_t n = static_cast<int64_t>(img.size());
    std::vector<float> out(img.size(), -1.0f);
    color_correct_affine(img.data(), ref.data(), n, out.data());
    double max_err = 0.0;
    for (size_t i = 0; i < out.size(); i++) {
        max_err = std::max(max_err, std::fabs(static_cast<double>(out[i]) - expected[i]));
        // atol 1e-6 — the measured max |out − torch_f64| = 2.85e-8 (dominated by the ≤½ulp rounding
        // of the f32 output; the printf below reports the measured value on every run). A margin of
        // about 35× — this is not an arbitrary relaxation.
        SX_CHECK_NEAR(out[i], expected[i], 0.0, 1e-6);
    }
    std::printf("  [cc pin] max |out - torch_f64| = %.3g\n", max_err);
    // The elements where the clip fires land exactly on the boundary values (the reference
    // pre-clip: 18 → −0.0477, 19 → 1.0073)
    SX_CHECK(out[18] == 0.0f);
    SX_CHECK(out[19] == 1.0f);
}

// ===== color_correct Task 1 — identity: img == ref → cc == img, bit-identical =====
// Between the same buffer, the four accumulators are bit-identical in pairs → a = x/x = exactly 1,
// b = m − 1·m = 0 → out = clip(img) = img (the clip is a no-op because the input is in [0,1]). This
// assumes var > 1e-8 (so the values must have spread).
SX_TEST(color_correct_affine_identity) {
    std::vector<float> im(36);
    for (size_t i = 0; i < im.size(); i++) // a 0..1 gradient — ensures per-channel spread
        im[i] = static_cast<float>(i) / static_cast<float>(im.size() - 1);
    std::vector<float> out(im.size(), -1.0f);
    color_correct_affine(im.data(), im.data(), static_cast<int64_t>(im.size()), out.data());
    SX_CHECK(std::memcmp(out.data(), im.data(), im.size() * sizeof(float)) == 0);
}

// ===== color_correct Task 1 — the guard and clip boundaries (dyadic values → exact arithmetic
// assertions) =====
SX_TEST(color_correct_affine_guards_and_clip) {
    // (1) A constant ref → var = 0 exactly → the 1e-8 clamp fires. Taking img as dyadic values
    //     ({0.25, 0.75}, mean 0.5) makes the covariance exactly 0 too → a_orig = 0/1e-8 = 0 → the
    //     |a| < 1e-8 substitution gives a = 1, b = img_mean = 0.5 → out = clip(img − 0.5), exactly
    //     {0, 0.25}. If either guard were missing, it would diverge with 0/0 = NaN or a division by
    //     zero.
    {
        const std::vector<float> ref(12, 0.5f);
        const std::vector<float> img = {0.25f, 0.25f, 0.25f, 0.75f, 0.75f, 0.75f,
                                        0.25f, 0.25f, 0.25f, 0.75f, 0.75f, 0.75f};
        std::vector<float> out(img.size(), -1.0f);
        color_correct_affine(img.data(), ref.data(), 12, out.data());
        for (size_t i = 0; i < out.size(); i++) {
            SX_CHECK(std::isfinite(out[i]));
            SX_CHECK(out[i] >= 0.0f && out[i] <= 1.0f);
            SX_CHECK(out[i] == (img[i] == 0.25f ? 0.0f : 0.25f)); // exact arithmetic
        }
    }
    // (2) The clip: on the normal path for a (≈0.5), the inverse mapping (img − 0.25)/0.5 falls
    //     outside [0,1] → it is clipped to exactly the boundary. Per channel, ref {0, 0.5, 1} and
    //     img {0.5, 0, 1} → pre-clip {0.5, −0.5, 1.5} → out {0.5, 0, 1}.
    {
        const std::vector<float> ref = {0.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.5f, 1.0f, 1.0f, 1.0f};
        const std::vector<float> img = {0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
        std::vector<float> out(img.size(), -1.0f);
        color_correct_affine(img.data(), ref.data(), 9, out.data());
        for (int k = 0; k < 3; k++) {
            SX_CHECK_NEAR(out[static_cast<size_t>(k)], 0.5, 0, 1e-6); // only the 1/3 rounding is involved
            SX_CHECK(out[static_cast<size_t>(3 + k)] == 0.0f);        // the lower clip is exact
            SX_CHECK(out[static_cast<size_t>(6 + k)] == 1.0f);        // the upper clip is exact
        }
    }
}

// ===== color_correct Task 1 — determinism and the failure paths =====
SX_TEST(color_correct_affine_determinism_and_validation) {
    // Determinism: sequential double accumulation (no parallelism and no atomics) — repeated calls
    // are bit-identical
    std::vector<float> img(36), ref(36);
    for (size_t i = 0; i < img.size(); i++) {
        img[i] = static_cast<float>((i * 7 + 3) % 36) / 35.0f;
        ref[i] = static_cast<float>((i * 11 + 5) % 36) / 35.0f;
    }
    std::vector<float> o1(img.size(), -1.0f), o2(img.size(), -2.0f);
    color_correct_affine(img.data(), ref.data(), 36, o1.data());
    color_correct_affine(img.data(), ref.data(), 36, o2.data());
    SX_CHECK(std::memcmp(o1.data(), o2.data(), o1.size() * sizeof(float)) == 0);
    // Failure paths: a null img/ref/out, n_elems <= 0, or a value that is not a multiple of 3 →
    // throw (consistent with the psnr convention)
    auto throws = [&](const float* a, const float* b, int64_t n, float* o) {
        try {
            color_correct_affine(a, b, n, o);
        } catch (...) { return true; }
        return false;
    };
    SX_CHECK(throws(nullptr, ref.data(), 36, o1.data()));
    SX_CHECK(throws(img.data(), nullptr, 36, o1.data()));
    SX_CHECK(throws(img.data(), ref.data(), 36, nullptr));
    SX_CHECK(throws(img.data(), ref.data(), 0, o1.data()));
    SX_CHECK(throws(img.data(), ref.data(), -3, o1.data()));
    SX_CHECK(throws(img.data(), ref.data(), 35, o1.data())); // not a multiple of 3
}

// ===== color_correct Task 2 — (c) recovering an appearance offset: cc_psnr > raw_psnr =====
// This reproduces the cc data flow of run_eval exactly (spec §4:
// psnr(color_correct_affine(render, gt), gt)): when the "render" is a per-channel gain and offset
// distortion of the GT (an exposure / white-balance gauge), the raw PSNR penalizes that gauge, but
// the affine alignment removes it, so the cc PSNR must recover substantially — this asserts the
// premise property of appearance-neutral model evaluation (cc eval).
SX_TEST(color_correct_psnr_recovers_appearance_offset) {
    const int W = 8, H = 6;
    const int64_t n = static_cast<int64_t>(W) * H * 3;
    std::vector<float> gt(static_cast<size_t>(n)), render(gt.size());
    // GT: a deterministic pattern in the range [0.1, 0.9] — ensuring per-channel spread (the region
    // where the var and a guards do not fire)
    for (size_t i = 0; i < gt.size(); i++)
        gt[i] = 0.1f + 0.8f * static_cast<float>((i * 13 + 7) % 144) / 143.0f;
    // render = gain_c·gt + offset_c (per channel). The values are chosen to stay inside [0,1] — a
    // pure gauge case in which both the [0,1] clamp in eval and the cc clip are no-ops
    // (R [0.13,0.77] / G [0.075,0.915] / B [0.17,0.89]) → after alignment, the GT must be recovered
    // down to f32 rounding.
    const float gain[3] = {0.8f, 1.05f, 0.9f};
    const float offset[3] = {0.05f, -0.03f, 0.08f};
    for (size_t i = 0; i < render.size(); i++)
        render[i] = gain[i % 3] * gt[i] + offset[i % 3];
    const float raw = psnr(render.data(), gt.data(), n);
    std::vector<float> cc(render.size(), -1.0f);
    color_correct_affine(render.data(), gt.data(), n, cc.data());
    const float cc_psnr = psnr(cc.data(), gt.data(), n);
    std::printf("  [cc recovery] raw %.3f dB -> cc %.3f dB\n", static_cast<double>(raw),
                static_cast<double>(cc_psnr));
    SX_CHECK(std::isfinite(raw) && raw > 0.0f);
    // The gauge-removal assertion — not merely cc > raw, but by a large margin (an exact affine gauge
    // is recovered almost completely: measured raw ≈ 24 dB → cc ≈ 100+ dB. +20 dB is a conservative
    // lower bound).
    SX_CHECK(cc_psnr > raw + 20.0f);
}

// ===== Perceptual metrics Task 1 — eval_ssim (wiring that reuses ssim_fwd_cpu; spec 2026-07-15
// §2–§3) =====

namespace {
// The ssim_basic fixture path (the fx() idiom from SsimTests.cpp — it uses only fixture_dir(); never
// hardcode the path)
std::string ssim_fx(const char* rel) { return sxtest::fixture_dir() + "/ssim_basic/" + rel; }
} // namespace

// (1) Identity plus direction: ssim(x,x) is mathematically exactly 1 (μ1=μ2, σ1²=σ2²=σ12 — both
//     factors are (2ab+C)/(a²+b²+C) = 1). Only the rounding of the f32 kernel arithmetic is
//     involved — the measured |1-s| is around 1e-8, and 1e-6 is a conservative upper bound. The
//     distorted image (img2 = img1 composed with a gain, an offset, a sine, and noise — see the
//     meta lineage) MUST come out lower.
SX_TEST(eval_ssim_identity_and_direction) {
    const SXTensor img1 = SXTensor::load(ssim_fx("img1.sxt")); // [1,24,24,3] f32
    const SXTensor img2 = SXTensor::load(ssim_fx("img2.sxt")); // a distortion of img1 (synthesized GT)
    const double s_self = eval_ssim(img1.f32(), img1.f32(), 24, 24);
    const double s_dist = eval_ssim(img2.f32(), img1.f32(), 24, 24);
    std::printf("  [eval_ssim] self=%.12f (|1-s|=%.3g), distort=%.9f\n", s_self,
                std::fabs(1.0 - s_self), s_dist);
    SX_CHECK(std::fabs(s_self - 1.0) <= 1e-6);
    SX_CHECK(s_dist < s_self);
}

// (2) Cross-check against the existing golden — no new fixture needed: compared against the fwd
//     golden scalar of ssim_basic (ssim.sxt, a single f32 element) using the meta.json tolerances
//     (atol 1e-6 / rtol 5e-5). meta is the single source of truth for tolerances (never hardcode
//     them).
SX_TEST(eval_ssim_matches_ssim_basic_golden) {
    const std::map<std::string, double> meta = json_parse_flat(ssim_fx("meta.json"));
    const SXTensor img1 = SXTensor::load(ssim_fx("img1.sxt"));
    const SXTensor img2 = SXTensor::load(ssim_fx("img2.sxt"));
    const SXTensor golden = SXTensor::load(ssim_fx("ssim.sxt"));
    const double s = eval_ssim(img1.f32(), img2.f32(), static_cast<int32_t>(meta.at("height")),
                               static_cast<int32_t>(meta.at("width")));
    std::printf("  [eval_ssim golden] %.9f (golden %.9f)\n", s,
                static_cast<double>(golden.f32()[0]));
    const float sf = static_cast<float>(s);
    SX_ALL_CLOSE(&sf, golden.f32(), 1, meta, "ssim");
}

// (3) The small-image NaN convention: h<11 || w<11 makes the valid map empty → a quiet NaN (not a
//     throw — deliberately different from the training validator; the authoritative statement is the
//     Trainer.h comment). Exactly at the 11×11 boundary the 1×1 map is finite.
SX_TEST(eval_ssim_small_image_nan) {
    std::vector<float> a(24 * 24 * 3), b(a.size());
    for (size_t i = 0; i < a.size(); i++) { // a deterministic pattern in the range [0,1]
        a[i] = static_cast<float>((i * 13 + 7) % 144) / 143.0f;
        b[i] = static_cast<float>((i * 7 + 3) % 144) / 143.0f;
    }
    SX_CHECK(std::isnan(eval_ssim(a.data(), b.data(), 10, 24))); // h < 11
    SX_CHECK(std::isnan(eval_ssim(a.data(), b.data(), 24, 10))); // w < 11
    const double s_min = eval_ssim(a.data(), b.data(), 11, 11);  // exactly at the boundary — a 1×1 valid map
    std::printf("  [eval_ssim 11x11] %.9f\n", s_min);
    SX_CHECK(std::isfinite(s_min));
}

// (4) Determinism: a sequential f32 conv (ssim_fwd_cpu) plus a sequential f64 summation
//     (ssim_reduce_mean_f64) — no parallelism and no atomics, so two calls are bit-identical as
//     doubles.
SX_TEST(eval_ssim_determinism) {
    const SXTensor img1 = SXTensor::load(ssim_fx("img1.sxt"));
    const SXTensor img2 = SXTensor::load(ssim_fx("img2.sxt"));
    const double s1 = eval_ssim(img1.f32(), img2.f32(), 24, 24);
    const double s2 = eval_ssim(img1.f32(), img2.f32(), 24, 24);
    SX_CHECK(std::memcmp(&s1, &s2, sizeof(double)) == 0);
}

// (5) Failure paths: a null img/ref → throw (consistent with the psnr and cc conventions — the idiom
// from the color_correct determinism test)
SX_TEST(eval_ssim_null_throws) {
    std::vector<float> buf(11 * 11 * 3, 0.5f);
    auto throws = [&](const float* a, const float* b) {
        try {
            (void)eval_ssim(a, b, 11, 11);
        } catch (...) { return true; }
        return false;
    };
    SX_CHECK(throws(nullptr, buf.data()));
    SX_CHECK(throws(buf.data(), nullptr));
}

// ===== Training speed Task 1 — eval_ssim backend routing (moving the eval SSIM to the GPU; spec
// 2026-07-21 §②) =====

// (6) GPU vs the CPU oracle: on the same synthetic input (rectangular plus multiple tiles — guarding
//     against H/W transposition and tile boundary regressions), eval_ssim(backend=Metal) agrees
//     approximately with the previous CPU path (ssim_fwd_cpu), plus repeated Metal calls are
//     bit-exact deterministic (the contract that ssim_fwd uses no atomics, plus f64 summation in
//     index order).
//
//     Why the tolerance is hardcoded here (this is not a golden fixture — meta.json is untouched):
//     this comparison verifies the deviation between two engines (the CPU sequential 2-pass conv vs
//     the Metal threadgroup-apron conv) on synthetic input. SSIM fwd has no transcendental
//     functions, so the only sources of deviation are (1) the accumulation order of the f32 conv and
//     (2) the final rounding of the libm↔metal precise:: division; and because the scalar is
//     summed in f64 in ascending index order on both sides (ssim_reduce_mean_f64, defined once), the
//     per-element deviations of the map are attenuated in the mean. The upper bound is borrowed from
//     the ssim scalar class in the ssim_basic meta.json (atol 1e-6 + rtol 5e-5 — the class
//     demonstrated by the Metal↔f64 golden comparison) — there is no basis for relaxing an
//     engine-to-engine comparison beyond that.
SX_TEST(eval_ssim_backend_matches_cpu_oracle) {
    const int32_t H = 36, W = 48; // a 26×38 valid map — a 3×2 grid of 16×16 tiles (including partial tiles)
    std::vector<float> render(static_cast<size_t>(H) * W * 3), gt(render.size());
    for (size_t i = 0; i < gt.size(); i++) { // a deterministic pattern — a GT and its distortion (SSIM ∈ (0,1))
        gt[i] = static_cast<float>((i * 31 + 11) % 257) / 256.0f;
        const float noise =
            0.08f * static_cast<float>(static_cast<int>((i * 7) % 13) - 6) / 6.0f;
        render[i] = std::clamp(0.9f * gt[i] + 0.05f + noise, 0.0f, 1.0f); // reproduces the run_eval clamp
    }
    const double cpu = eval_ssim(render.data(), gt.data(), H, W); // omitting the backend = the CPU oracle
    KernelBackend& metal = sxtest::shared_metal_backend();
    const double gpu = eval_ssim(render.data(), gt.data(), H, W, &metal);
    std::printf("  [eval_ssim backend] cpu=%.12f gpu=%.12f |d|=%.3g\n", cpu, gpu,
                std::fabs(gpu - cpu));
    SX_CHECK(std::isfinite(gpu) && gpu > 0.0 && gpu < 1.0);
    SX_CHECK(std::fabs(gpu - cpu) <= 1e-6 + 5e-5 * std::fabs(cpu));
    const double gpu2 = eval_ssim(render.data(), gt.data(), H, W, &metal);
    SX_CHECK(std::memcmp(&gpu, &gpu2, sizeof(double)) == 0); // bit-exact determinism across calls
}

// (7) The routing convention: the h/w<11 NaN guard runs **before** the backend is called (the
//     backend validator throws under the same condition — the authoritative statement is the
//     Trainer.h comment: eval is observation, not a contract), and going through CpuBackend
//     delegates to ssim_fwd_cpu (CpuBackend.cpp), so it is bit-identical to the oracle as a double —
//     the identity proof that routing itself does not change the numerical conventions.
SX_TEST(eval_ssim_backend_nan_guard_and_cpu_identity) {
    std::vector<float> a(24 * 24 * 3), b(a.size());
    for (size_t i = 0; i < a.size(); i++) { // the same deterministic pattern as eval_ssim_small_image_nan
        a[i] = static_cast<float>((i * 13 + 7) % 144) / 143.0f;
        b[i] = static_cast<float>((i * 7 + 3) % 144) / 143.0f;
    }
    KernelBackend& metal = sxtest::shared_metal_backend();
    SX_CHECK(std::isnan(eval_ssim(a.data(), b.data(), 10, 24, &metal))); // h<11 — not a throw
    SX_CHECK(std::isnan(eval_ssim(a.data(), b.data(), 24, 10, &metal))); // w<11 — not a throw
    const auto cpu_backend = make_cpu_backend();
    const double oracle = eval_ssim(a.data(), b.data(), 24, 24);
    const double routed = eval_ssim(a.data(), b.data(), 24, 24, cpu_backend.get());
    SX_CHECK(std::memcmp(&oracle, &routed, sizeof(double)) == 0);
}

// ===== Perceptual metrics Task 2 — direction: cc removes the appearance gauge, so SSIM recovers
// =====
// This reproduces the ccSSIM data flow of run_eval exactly
// (eval_ssim(color_correct_affine(render, gt), gt)): when the "render" is a per-channel gain and
// offset distortion of the GT (the same gauge as the cc recovery test — the region where the clamp
// and clip are no-ops), the affine alignment removes the gauge down to f32 rounding, so
// eval_ssim(cc, gt) must exceed raw and must approach 1. The argument order is the same as in
// run_eval (img=render|cc, ref=GT — the Trainer.h convention; never swap them, because the f32
// operation order depends on it).
SX_TEST(eval_ssim_cc_recovers_appearance_offset) {
    const int W = 24, H = 16; // ≥ 11×11 (a 6×14 valid map) — the NaN convention does not fire
    const int64_t n = static_cast<int64_t>(W) * H * 3;
    std::vector<float> gt(static_cast<size_t>(n)), render(gt.size());
    // GT: the same deterministic pattern as the cc recovery test, in the range [0.1, 0.9]
    for (size_t i = 0; i < gt.size(); i++)
        gt[i] = 0.1f + 0.8f * static_cast<float>((i * 13 + 7) % 144) / 143.0f;
    const float gain[3] = {0.8f, 1.05f, 0.9f};
    const float offset[3] = {0.05f, -0.03f, 0.08f};
    for (size_t i = 0; i < render.size(); i++)
        render[i] = gain[i % 3] * gt[i] + offset[i % 3];
    const double raw_ssim = eval_ssim(render.data(), gt.data(), H, W);
    std::vector<float> cc(render.size(), -1.0f);
    color_correct_affine(render.data(), gt.data(), n, cc.data());
    const double cc_ssim = eval_ssim(cc.data(), gt.data(), H, W);
    std::printf("  [ssim cc recovery] raw %.9f -> cc %.9f\n", raw_ssim, cc_ssim);
    SX_CHECK(std::isfinite(raw_ssim) && raw_ssim > 0.0);
    SX_CHECK(cc_ssim > raw_ssim); // direction — removing the gauge improves SSIM
    // An exact affine gauge is recovered almost completely — the cc SSIM must approach 1 (measured:
    // raw 0.987244730, cc 1.000000022 — f32 kernel rounding can push it slightly above 1, so an
    // upper-bound assertion is forbidden. 0.99 is a lower bound that excludes raw while letting cc
    // through — the path is bit-exact deterministic, so there is no flake).
    SX_CHECK(cc_ssim > 0.99);
}

// ===== Phase 2 Task 5 — the train/test split convention (authoritative source: gsplat
// colmap.py:456–461) =====
// test = { i : i % test_every == 0 }, train = the rest — the MipNeRF360 convention of test_every 8.
SX_TEST(train_test_split_mipnerf360_convention) {
    const TrainTestSplit s = make_train_test_split(20, 8);
    SX_CHECK(s.test == (std::vector<int64_t>{0, 8, 16}));
    SX_CHECK(s.train.size() == 17);
    for (int64_t t : s.train) SX_CHECK(t % 8 != 0); // train contains no test index
    // Disjointness and completeness: train ∪ test = {0..19}, each exactly once (sorting the union
    // gives the identity sequence)
    std::vector<int64_t> all = s.train;
    all.insert(all.end(), s.test.begin(), s.test.end());
    std::sort(all.begin(), all.end());
    SX_CHECK(all.size() == 20);
    for (int64_t i = 0; i < static_cast<int64_t>(all.size()); i++)
        SX_CHECK(all[static_cast<size_t>(i)] == i);
    // Both are in ascending order (a premise of TrainC iteration and eval determinism)
    SX_CHECK(std::is_sorted(s.train.begin(), s.train.end()));
    SX_CHECK(std::is_sorted(s.test.begin(), s.test.end()));

    // split off (test_every <= 0): everything is train
    const TrainTestSplit off = make_train_test_split(5, 0);
    SX_CHECK(off.train.size() == 5 && off.test.empty());

    // The boundary: test_every == 1 → everything is test (the upstream numpy predicate verbatim —
    // the caller (TrainC) rejects it)
    const TrainTestSplit one = make_train_test_split(3, 1);
    SX_CHECK(one.train.empty() && one.test.size() == 3);

    // n < test_every: test = {0} only (index 0 is always test — the same as gsplat)
    const TrainTestSplit few = make_train_test_split(4, 8);
    SX_CHECK(few.test.size() == 1 && few.test[0] == 0 && few.train.size() == 3);

    // n == 0: both are empty (not a throw — checking whether cameras exist is the caller's
    // responsibility)
    const TrainTestSplit empty = make_train_test_split(0, 8);
    SX_CHECK(empty.train.empty() && empty.test.empty());
}

// ===== Phase 3 Task 5 — the device-resident train_step (D11 / 00-index §7) =====

namespace {

// The shared initial state for the device-resident comparison — the same perturbation as (d) (a
// deterministic seed). When sh_degree >= 1, shN is perturbed as well (consumed after sh0 — at deg 0,
// shN is empty, so the stream is exactly identical to the existing one).
SplatCloud make_perturbed_cloud(int sh_degree = 0) {
    SplatCloud init = make_smoke_gt_cloud(sh_degree);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto& v : init.means) v += 0.05f * u(rng);
    for (auto& v : init.scales) v += 0.30f * u(rng);
    for (auto& v : init.quats) v += 0.05f * u(rng);
    for (auto& v : init.opacities) v += 0.70f * u(rng);
    for (auto& v : init.sh0) v += 0.60f * u(rng);
    for (auto& v : init.shN) v += 0.10f * u(rng);
    return init;
}

// Asserts an upper bound on the rel_l2 of every parameter group between two Trainers (ref = the
// first argument — the host or first run).
void check_clouds_close(Trainer& a, Trainer& b, double gate, const char* tag) {
    const SplatCloud& ca = a.cloud();
    const SplatCloud& cb = b.cloud();
    SX_CHECK(ca.count == cb.count);
    const std::pair<const std::vector<float>*, const std::vector<float>*> groups[] = {
        {&ca.means, &cb.means},         {&ca.quats, &cb.quats}, {&ca.scales, &cb.scales},
        {&ca.opacities, &cb.opacities}, {&ca.sh0, &cb.sh0},     {&ca.shN, &cb.shN}};
    double worst = 0.0;
    for (const auto& [pa, pb] : groups) {
        SX_CHECK(pa->size() == pb->size());
        if (pa->empty()) continue;
        worst = std::max(worst, sxtest::rel_l2(pb->data(), pa->data(), pa->size(), pa->data()));
    }
    std::printf("  [%s] param rel_l2 worst = %.3g (gate %.1g)\n", tag, worst, gate);
    SX_CHECK(worst <= gate);
}

// The shared runner that contrasts steps on the host round-trip path against the device-resident
// path — for the gates, see the comment in the body of (n): exact equality of n_isects per step,
// loss |Δ| ≤ 1e-6+1e-4·|loss|, a first-step forward render rel_l2 ≤ 1e-5 (before any atomics — only
// the activation deviation), and rel_l2 ≤ 1e-4 across every parameter group after the steps.
void run_host_resident_contrast(KernelBackend& b, const TrainingCamera& cam,
                                const SplatCloud& init, const TrainConfig& cfg, int steps,
                                const char* tag) {
    TrainConfig cfg_host = cfg;
    cfg_host.resident = false; // the verification surface (host round-trip chaining — the previous path)
    Trainer tr_h(b, init, cfg_host);
    Trainer tr_r(b, init, cfg);

    const std::vector<float> gt(static_cast<size_t>(cam.width) * cam.height * 3, 0.35f);
    for (int it = 0; it < steps; it++) {
        const TrainStepResult rh = tr_h.step(cam, gt.data(), it);
        const TrainStepResult rr = tr_r.step(cam, gt.data(), it);
        SX_CHECK(rh.n_isects > 0);
        SX_CHECK(rh.n_isects == rr.n_isects); // exact equality of the forward assembly (the tile intersection set)
        const double dl = std::fabs(static_cast<double>(rh.loss) - rr.loss);
        std::printf("  [%s] it %d: loss host %.7f resident %.7f (|d|=%.2e), n_isects %lld\n",
                    tag, it, static_cast<double>(rh.loss), static_cast<double>(rr.loss), dl,
                    static_cast<long long>(rh.n_isects));
        SX_CHECK(dl <= 1e-6 + 1e-4 * std::fabs(static_cast<double>(rh.loss)));
        if (it == 0) {
            // Comparing the first-step forward render — this segment precedes any atomics, so only
            // the activation deviation remains
            SX_CHECK(rh.render_colors.size() == rr.render_colors.size());
            const double rl2 = sxtest::rel_l2(rr.render_colors.data(), rh.render_colors.data(),
                                              rh.render_colors.size(), rh.render_colors.data());
            std::printf("  [%s] it 0 forward render rel_l2 = %.3g\n", tag, rl2);
            SX_CHECK(rl2 <= 1e-5);
        }
    }
    check_clouds_close(tr_h, tr_r, 1e-4, tag);
}

} // namespace

// (n) Step comparison — the host round-trip path vs the device-resident path over N=5 steps, from
// the same initial state and against the same GT.
// The assembly uses the same kernels, so there are only two sources of deviation: (1) the libm↔metal
// precise:: difference in the exp/sigmoid activations (the same contract as the perturb kernel — a
// tolerance), and (2) the accumulation order of the atomic bwd kernels (raster/sh/proj) (around the
// existing self-consistency gate of 1e-4) (the brief: "anything going through atomics gets a
// tolerance").
// The integrity of the forward assembly is pinned down separately by **exact equality** of n_isects
// per step (a different tile intersection set is caught immediately — because radii is an integer
// boundary, this is effectively insensitive to a 1 ulp activation deviation).
SX_TEST(trainer_resident_vs_host_step_contrast) {
    KernelBackend& b = sxtest::shared_metal_backend();
    const TrainingCamera cam = make_smoke_camera();
    const SplatCloud init = make_perturbed_cloud();

    TrainConfig cfg;
    cfg.total_iters = 100;
    cfg.lr_means = 2e-3f;
    cfg.lr_scales = 2e-2f;
    cfg.lr_quats = 5e-3f;
    cfg.lr_opacities = 5e-2f;
    cfg.lr_sh0 = 2e-2f;
    cfg.ssim_weight = 0.2f;  // includes the SSIM path (the default — comparing the combined fwd/bwd grads too)
    cfg.opacity_reg = 0.01f; // includes the reg-addition kernel path
    cfg.scale_reg = 0.01f;
    SX_CHECK(TrainConfig{}.resident); // the default gate — resident is on by default (the brief)

    run_host_resident_contrast(b, cam, init, cfg, 5, "host vs resident, 5 steps");
}

// (n-2) A step comparison variant — sh_degree 1 (K=4): this puts the shN branch of the coeffs
// assembly and disassembly (train_assemble_coeffs / train_split_v_coeffs) and the device-resident
// path of the shN adam group under the comparison gate. total 100 → an sh degree interval of 3
// (sh_degree_at) — 6 steps drive through the promotion boundary (iter 3, degrees_to_use 0→1): the
// upper-band gradients start flowing from the 3rd step (verifying that the schedule is threaded
// through).
SX_TEST(trainer_resident_vs_host_contrast_sh_degree1) {
    KernelBackend& b = sxtest::shared_metal_backend();
    const TrainingCamera cam = make_smoke_camera();
    const SplatCloud init = make_perturbed_cloud(1); // K_total = 4, the shN [16,3,3] perturbation pattern

    TrainConfig cfg;
    cfg.total_iters = 100;
    cfg.lr_means = 2e-3f;
    cfg.lr_scales = 2e-2f;
    cfg.lr_quats = 5e-3f;
    cfg.lr_opacities = 5e-2f;
    cfg.lr_sh0 = 2e-2f;
    cfg.lr_shN = 1e-3f;      // raised above the default (1.25e-4) — makes the shN movement clearly visible in the comparison
    cfg.ssim_weight = 0.2f;
    cfg.opacity_reg = 0.01f;
    cfg.scale_reg = 0.01f;
    // Self-verification: the promotion boundary is at iter 3 as intended (0..2 are DC only — the shN
    // gradient is 0)
    SX_CHECK(sh_degree_at(cfg, 2, 1) == 0 && sh_degree_at(cfg, 3, 1) == 1);

    Trainer probe(b, init, cfg); // for the not-vacuous check — re-runs under the same conditions as the comparison runner
    run_host_resident_contrast(b, cam, init, cfg, 6, "sh_degree 1, 6 steps");

    // Not vacuous: on the steps with degrees_to_use >= 1, the shN gradient really flowed and moved
    // the parameters
    const std::vector<float> gt(static_cast<size_t>(cam.width) * cam.height * 3, 0.35f);
    for (int it = 0; it < 6; it++) (void)probe.step(cam, gt.data(), it);
    bool shn_moved = false;
    const SplatCloud& after = probe.cloud();
    for (size_t j = 0; j < after.shN.size() && !shn_moved; j++)
        shn_moved = after.shN[j] != init.shN[j];
    SX_CHECK(shn_moved);
}

// (n-3) A step comparison variant — antialiased=true: contrasts the host and device-resident paths
// for the AA chain (opacity_aa = the σ·comp product → consumed by isect and raster, with v_comp
// injected in bwd plus the train_aa_chain kernel).
// λ=0.2 and reg are kept — the maximal assembly path, with AA, SSIM, and reg all overlapping.
SX_TEST(trainer_resident_vs_host_contrast_antialiased) {
    KernelBackend& b = sxtest::shared_metal_backend();
    const TrainingCamera cam = make_smoke_camera();
    const SplatCloud init = make_perturbed_cloud();

    TrainConfig cfg;
    cfg.total_iters = 100;
    cfg.lr_means = 2e-3f;
    cfg.lr_scales = 2e-2f;
    cfg.lr_quats = 5e-3f;
    cfg.lr_opacities = 5e-2f;
    cfg.lr_sh0 = 2e-2f;
    cfg.ssim_weight = 0.2f;
    cfg.opacity_reg = 0.01f;
    cfg.scale_reg = 0.01f;
    cfg.antialiased = true; // the subject of this variant itself (kProjCalcComp plus the AA product-rule chain)

    run_host_resident_contrast(b, cam, init, cfg, 5, "antialiased, 5 steps");
}

// (o) A comparison that goes through MCMC refine — checks that refine (host orchestration plus
// device-resident↔host synchronization) and the device-resident noise encoding on non-refine steps
// keep the same trajectory as the host path.
// Two dead slots are planted (σ=0.001 << the min_opacity of 0.005) so relocate is not vacuous, and
// a cap of 24 makes the add growth not vacuous. Under the refine schedule (100 total → start 2 /
// stop 83 / every 1), refine runs from iter 3 — so the noise is driven on both sides: encoded
// device-resident on iters 0..2, and host-owned from 3 onward. The multinomial (mt19937_64 on the
// host) and Philox (offset=iter) streams are identical on both paths — exact equality of the index
// flow is the expected result.
SX_TEST(trainer_resident_mcmc_refine_contrast) {
    KernelBackend& b = sxtest::shared_metal_backend();
    const TrainingCamera cam = make_smoke_camera();
    SplatCloud init = make_perturbed_cloud();
    init.opacities[2] = logit_f32(0.001f); // dead (clearly separated from the 0.005 boundary — a stable classification)
    init.opacities[5] = logit_f32(0.001f);

    TrainConfig cfg;
    cfg.total_iters = 100;
    cfg.lr_means = 2e-3f;
    cfg.lr_scales = 2e-2f;
    cfg.lr_quats = 5e-3f;
    cfg.lr_opacities = 5e-2f;
    cfg.lr_sh0 = 2e-2f;
    cfg.ssim_weight = 0.0f; // pure L1 — isolates the MCMC path (the SSIM combination is covered by (n))
    cfg.opacity_reg = 0.01f;
    cfg.scale_reg = 0.01f;
    cfg.mcmc.enabled = true;
    cfg.mcmc.cap_max = 24;
    cfg.mcmc.seed = 777;

    TrainConfig cfg_host = cfg;
    cfg_host.resident = false;
    Trainer tr_h(b, init, cfg_host);
    Trainer tr_r(b, init, cfg);

    const std::vector<float> gt(static_cast<size_t>(cam.width) * cam.height * 3, 0.35f);
    bool any_refined = false;
    int64_t total_added = 0;
    for (int it = 0; it < 8; it++) {
        const TrainStepResult rh = tr_h.step(cam, gt.data(), it);
        const TrainStepResult rr = tr_r.step(cam, gt.data(), it);
        SX_CHECK(rh.n_isects == rr.n_isects);
        SX_CHECK(rh.mcmc_refined == rr.mcmc_refined);
        SX_CHECK(rh.n_relocated == rr.n_relocated); // exact equality of the index flow (multinomial)
        SX_CHECK(rh.n_added == rr.n_added);
        SX_CHECK(tr_h.splat_count() == tr_r.splat_count());
        any_refined = any_refined || rh.mcmc_refined;
        total_added += rh.n_added;
    }
    SX_CHECK(any_refined);      // not vacuous — refine really ran
    SX_CHECK(total_added > 0);  // drives the growth via add
    SX_CHECK(tr_h.splat_count() > init.count);
    check_clouds_close(tr_h, tr_r, 1e-4, "mcmc refine, 8 steps");
    // The Adam moments are compared through the synchronization as well (the consistency of the
    // round trip between the refine state surgery and the device-resident adam update)
    Trainer::AdamState& mh = tr_h.adam_state(Trainer::ParamGroup::Means);
    Trainer::AdamState& mr = tr_r.adam_state(Trainer::ParamGroup::Means);
    SX_CHECK(mh.exp_avg.size() == mr.exp_avg.size());
    const double m_l2 = sxtest::rel_l2(mr.exp_avg.data(), mh.exp_avg.data(),
                                       mh.exp_avg.size(), mh.exp_avg.data());
    std::printf("  means exp_avg rel_l2 = %.3g\n", m_l2);
    SX_CHECK(m_l2 <= 1e-3); // the moments are a first-order accumulation of the grads — the atomic deviation hits them more directly than the parameters
}

// (p) Device-resident determinism plus render()/step() forward consistency — the contract of the
// device-resident path itself:
// ① Two runs from the same initial state reproduce the first-step forward **bit-identically** (the
//    entire forward segment — activations, proj, SH, isect, scan, sort, offset, and raster — uses no
//    atomics), and the loss is bit-identical too.
// ② After 5 steps the parameters agree at the self-consistency level (1e-5) (the only deviation is
//    the accumulation order of the atomic bwd kernels).
// ③ Whether render() takes the same device-resident path as the forward in step() — using its own
//    render as GT, the L1 loss of one step is exactly 0 (if the activations differed between the
//    paths, the bits would diverge and the 0 would break immediately).
SX_TEST(trainer_resident_determinism_and_render_consistency) {
    KernelBackend& b = sxtest::shared_metal_backend();
    const TrainingCamera cam = make_smoke_camera();
    const SplatCloud init = make_perturbed_cloud();

    TrainConfig cfg;
    cfg.total_iters = 100;
    cfg.lr_means = 2e-3f;
    cfg.lr_scales = 2e-2f;
    cfg.lr_quats = 5e-3f;
    cfg.lr_opacities = 5e-2f;
    cfg.lr_sh0 = 2e-2f;
    cfg.ssim_weight = 0.2f;

    Trainer a(b, init, cfg);
    Trainer c(b, init, cfg);
    const std::vector<float> gt(static_cast<size_t>(cam.width) * cam.height * 3, 0.35f);
    for (int it = 0; it < 5; it++) {
        const TrainStepResult ra = a.step(cam, gt.data(), it);
        const TrainStepResult rc = c.step(cam, gt.data(), it);
        SX_CHECK(ra.n_isects == rc.n_isects);
        if (it == 0) { // ① bit-exact forward determinism (before any atomics — memcmp)
            SX_CHECK(ra.render_colors.size() == rc.render_colors.size());
            SX_CHECK(std::memcmp(ra.render_colors.data(), rc.render_colors.data(),
                                 ra.render_colors.size() * sizeof(float)) == 0);
            SX_CHECK(ra.loss == rc.loss); // the loss is an f64 host summation over bit-identical input — bit-identical
        }
    }
    check_clouds_close(a, c, 1e-5, "resident double-run"); // ② at the self-consistency level

    // ③ render()/step() forward consistency — the self-render fixed point (ssim_weight 0: L1 only,
    // sign(0)=0)
    TrainConfig cfg0 = cfg;
    cfg0.ssim_weight = 0.0f;
    Trainer tr(b, init, cfg0);
    const std::vector<float> self_img = tr.render(cam); // the device-resident forward (full sh_degree=0)
    const TrainStepResult r0 = tr.step(cam, self_img.data(), 0);
    SX_CHECK(r0.n_isects > 0);
    SX_CHECK(r0.loss == 0.0f); // if the paths diverged, the activation bits would differ and this would fail immediately
}

// ===== color_correct Task 2 — (a)(b) C API integration: SxTrainStats.final_cc_psnr =====
// (Extended by perceptual metrics Task 2: final_ssim and final_cc_ssim are asserted alongside under
//  the same (a)(b) contract — the e2e image is 64×48 ≥ 11×11, so the SSIM is finite, and with the
//  split off the NaN convention is identical)
// An sx_train_run_args E2E test (a small synthetic COLMAP txt bundle — a C++ port of
// Tests/SplatXMetalTests/SyntheticColmap.swift, with 3 gradient PNG views whose phases differ per
// view):
//   (a) test_split 2 → both final_eval_psnr and final_cc_psnr are finite (>0 dB)
//       plus final_ssim is finite in (0,1] and final_cc_ssim is finite
//   (b) test_split 0 → everything is NaN (the not-performed NaN convention — the new fields follow
//       the same contract as the existing ones)
// A new backend is created for each run (TrainC make_metal_backend), so when SPLATX_TEST_METALLIB is
// set, that path is injected as metallib_path unchanged in order to drive the same path end to end
// under the metallib gate too (when unset, kernel_src_dir() source compilation is used — on iOS the
// injected path is resolved, TestPaths.h).
namespace {

// The synthetic COLMAP txt bundle: root/sparse/0/{cameras,images,points3D}.txt plus
// root/images/*.png (factor 1 → images/). What is verified is the stats wiring contract, not the
// quality of training convergence, so arbitrary content is sufficient (the same as the
// SyntheticColmap.swift comment). It returns the root to be used as data_dir.
std::string make_synthetic_colmap_txt_bundle(const std::string& name) {
    namespace fs = std::filesystem;
    const int W = 64, H = 48, kCams = 3;
    const fs::path root = sxtest::tmp_path(name);
    std::error_code ec;
    fs::remove_all(root, ec); // removes leftovers from a previous run (prevents contamination from mixed bin/txt)
    fs::create_directories(root / "sparse/0");
    fs::create_directories(root / "images");
    const auto write_file = [](const fs::path& p, const std::string& s) {
        std::ofstream f(p, std::ios::binary);
        f << s;
        if (!f) throw std::runtime_error("test: write failed " + p.string());
    };

    // cameras.txt — a single shared PINHOLE camera: fx=fy=50, principal point at the center (the
    // points fall inside the frustum of every view)
    write_file(root / "sparse/0/cameras.txt", "1 PINHOLE 64 48 50 50 32 24\n");

    // images.txt — identity rotation plus a translation along x, with the second line (POINTS2D)
    // empty (the parser contract)
    std::string imgs;
    for (int i = 0; i < kCams; i++) {
        char line[128];
        std::snprintf(line, sizeof(line), "%d 1 0 0 0 %.6f 0 0 1 img%03d.png\n\n", i + 1,
                      0.15 * (i - (kCams - 1) / 2.0), i);
        imgs += line;
    }
    write_file(root / "sparse/0/images.txt", imgs);

    // points3D.txt — a deterministic scatter in front of the cameras (z 3..5) (an LCG — the same
    // coefficients as SyntheticColmap.swift)
    uint64_t state = 0x51475321ull;
    const auto uniform = [&state](double lo, double hi) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return lo + (hi - lo) * (static_cast<double>((state >> 33) % 1000000ull) / 1e6);
    };
    std::string pts;
    for (int i = 0; i < 60; i++) {
        char line[160];
        std::snprintf(line, sizeof(line), "%d %.6f %.6f %.6f %d %d %d 0\n", i + 1,
                      uniform(-1.2, 1.2), uniform(-0.9, 0.9), uniform(3.0, 5.0),
                      static_cast<int>(uniform(30, 225)), static_cast<int>(uniform(30, 225)),
                      static_cast<int>(uniform(30, 225)));
        pts += line;
    }
    write_file(root / "sparse/0/points3D.txt", pts);

    // RGB gradient PNGs whose phase differs per view (the dims match the camera — passing the TrainC
    // dims guard)
    for (int i = 0; i < kCams; i++) {
        ImageF32 img;
        img.width = W;
        img.height = H;
        img.rgb.resize(static_cast<size_t>(W) * H * 3);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                const size_t o = (static_cast<size_t>(y) * W + x) * 3;
                img.rgb[o + 0] = static_cast<float>((x * 4 + i * 40) % 256) / 255.0f;
                img.rgb[o + 1] = static_cast<float>((y * 5 + i * 20) % 256) / 255.0f;
                img.rgb[o + 2] = static_cast<float>((x + y + i * 60) % 256) / 255.0f;
            }
        char nm[32];
        std::snprintf(nm, sizeof(nm), "img%03d.png", i);
        save_png_rgb_f32(img, (root / "images" / nm).string());
    }
    return root.string();
}

} // namespace

SX_TEST(train_c_stats_final_cc_psnr) {
    const std::string root = make_synthetic_colmap_txt_bundle("splatx_cc_e2e_bundle");
    const std::string ksd = sxtest::kernel_src_dir();
    const char* lib = std::getenv("SPLATX_TEST_METALLIB");

    const auto run = [&](int32_t test_split, const char* out_name, SxTrainStats& st) {
        const std::string out = sxtest::tmp_path(out_name);
        SxTrainArgs a{}; // 0/NULL initialization, then only the needed fields (the header contract;
                         // thermal_mode 0 = performance, no intervention — the same determinism
                         // rationale as SplatXTrainerTests.swift)
        a.data_dir = root.c_str();
        a.factor = 1;
        a.iters = 6;
        a.max_points = 200;
        a.sh_degree = 1;
        a.out_path = out.c_str();
        a.log_every = 1000; // suppresses stdout noise (observation goes through stats)
        a.test_split = test_split;
        a.resident = 1;
        if (lib && *lib)
            a.metallib_path = lib;
        else
            a.kernel_src_dir = ksd.c_str();
        a.stats = &st;
        char err[512] = {};
        const int32_t rc = sx_train_run_args(&a, err, sizeof(err));
        if (rc != 0) std::printf("  sx_train_run_args rc=%d err=%s\n", rc, err);
        SX_CHECK(rc == 0);
    };

    // (a) test_split 2 (test = {0, 2}, train = {1}) — the final eval runs → both raw and cc are valid
    SxTrainStats with_split{};
    run(2, "splatx_cc_e2e_split.ply", with_split);
    std::printf("  [cc stats] split: final_eval_psnr=%.3f final_cc_psnr=%.3f "
                "final_ssim=%.4f final_cc_ssim=%.4f\n",
                static_cast<double>(with_split.final_eval_psnr),
                static_cast<double>(with_split.final_cc_psnr),
                static_cast<double>(with_split.final_ssim),
                static_cast<double>(with_split.final_cc_ssim));
    SX_CHECK(std::isfinite(with_split.final_eval_psnr) && with_split.final_eval_psnr > 0.0f);
    SX_CHECK(std::isfinite(with_split.final_cc_psnr) && with_split.final_cc_psnr > 0.0f);
    // Perceptual metrics Task 2 — 64×48 ≥ 11×11, so it is finite. SSIM ∈ (0,1] (identical images give
    // exactly 1; a negative value is mathematically possible, but for a trained render vs GT anything
    // at or below 0 signals a wiring error)
    SX_CHECK(std::isfinite(with_split.final_ssim) && with_split.final_ssim > 0.0f &&
             with_split.final_ssim <= 1.0f);
    SX_CHECK(std::isfinite(with_split.final_cc_ssim));

    // (b) test_split 0 — the eval does not run → the new fields are NaN, just like the existing ones
    SxTrainStats no_split{};
    run(0, "splatx_cc_e2e_nosplit.ply", no_split);
    SX_CHECK(std::isnan(no_split.final_eval_psnr)); // the existing convention (the baseline)
    SX_CHECK(std::isnan(no_split.final_cc_psnr));   // a new field — the same convention
    SX_CHECK(std::isnan(no_split.final_ssim));      // perceptual metrics Task 2 — the same convention
    SX_CHECK(std::isnan(no_split.final_cc_ssim));
}
