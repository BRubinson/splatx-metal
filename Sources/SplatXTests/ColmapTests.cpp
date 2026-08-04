#include "TestHarness.h"
#include "TestPaths.h" // fixture_dir() — single source of truth for paths (runtime injection)
#include "splatx/ColmapLoader.h"
#include "splatx/ImageIO.h"        // integration Task 2 — the factor downscale fallback smoke test
#include "splatx/TrainingInput.h"  // resolve_images_dir
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splatx;
namespace fs = std::filesystem;

SX_TEST(colmap_mini_bundle_parses) {
    ColmapBundle b = load_colmap_bundle(sxtest::fixture_dir() + "/colmap_mini/sparse/0");

    SX_CHECK(b.cameras.size() == 2);
    SX_CHECK(b.cameras[0].model_id == 1 && b.cameras[0].width == 1920);
    SX_CHECK_NEAR(b.cameras[0].fx, 1500.0f, 0, 1e-4);
    SX_CHECK_NEAR(b.cameras[0].fy, 1510.0f, 0, 1e-4);
    SX_CHECK_NEAR(b.cameras[1].fx, 500.0f, 0, 1e-4);  // SIMPLE_PINHOLE: fx=fy=f
    SX_CHECK_NEAR(b.cameras[1].fy, 500.0f, 0, 1e-4);

    SX_CHECK(b.images.size() == 2);
    SX_CHECK(b.images[0].name == "frame_000.jpg");
    SX_CHECK_NEAR(b.images[1].tvec[0], 0.2, 0, 1e-12);
    SX_CHECK_NEAR(b.images[0].qvec[0], 1.0, 0, 1e-12);

    SX_CHECK(b.points.size() == 3);
    SX_CHECK_NEAR(b.points[2].xyz[2], 5.0, 0, 1e-12);
    SX_CHECK(b.points[0].rgb[0] == 255);
}

// ===== txt support (iOS capture app integration Task 1) =====

namespace {

void write_file(const fs::path& path, const std::string& content) {
    std::ofstream f(path, std::ios::binary);
    f << content;
}

// A fresh tmp bundle directory (removes leftovers from a previous run — a surviving cameras.bin or
// similar would contaminate detection)
fs::path fresh_bundle_dir(const std::string& name) {
    const fs::path dir = sxtest::tmp_path(name);
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    return dir;
}

std::string fmt17(double v) { // shortest round trip (%.17g — bit-identical when reparsed by strtod)
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

std::string model_name(int model_id) {
    switch (model_id) {
        case 0: return "SIMPLE_PINHOLE";
        case 1: return "PINHOLE";
        default: throw std::runtime_error("test: model name not mapped " + std::to_string(model_id));
    }
}

// Returns the message load_colmap_bundle threw (an empty string if it did not throw)
std::string colmap_throw_msg(const std::string& dir) {
    try {
        (void)load_colmap_bundle(dir);
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return {};
}

bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

} // namespace

// ① A bin fixture → txt conversion pair loads to exactly the same result (camera parameters,
// poses, and points alike). The conversion happens inside the test: the result of the bin load is
// serialized into standard COLMAP txt (%.17g — a lossless double round trip), while comments,
// blank lines, real POINTS2D data, and tracks are mixed in so that the parser's accepted range is
// verified at the same time.
SX_TEST(colmap_txt_matches_bin) {
    const ColmapBundle bin = load_colmap_bundle(sxtest::fixture_dir() + "/colmap_mini/sparse/0");
    const fs::path dir = fresh_bundle_dir("colmap_txt_pair");

    std::string cams = "# Camera list with one line of data per camera:\n"
                       "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n\n";
    for (const auto& c : bin.cameras) {
        cams += std::to_string(c.id) + " " + model_name(c.model_id) + "\t" +
                std::to_string(c.width) + " " + std::to_string(c.height);
        for (double p : c.params) cams += "  " + fmt17(p); // verifies that runs of spaces are ok
        cams += "\n";
    }
    write_file(dir / "cameras.txt", cams);

    std::string imgs = "# Image list with two lines of data per image:\n";
    for (size_t i = 0; i < bin.images.size(); i++) {
        const auto& im = bin.images[i];
        imgs += std::to_string(im.id);
        for (double q : im.qvec) imgs += " " + fmt17(q);
        for (double t : im.tvec) imgs += " " + fmt17(t);
        imgs += " " + std::to_string(im.camera_id) + " " + im.name + "\n";
        // Second line: the first image carries real POINTS2D data (verifies the skip), the rest are
        // blank lines (the SplatXKit style)
        imgs += (i == 0) ? "10.0 20.0 -1 30.0 40.0 7\n" : "\n";
        if (i == 0) imgs += "# interleaved comment between image blocks\n";
    }
    write_file(dir / "images.txt", imgs);

    std::string pts = "# 3D point list with one line of data per point:\n";
    for (size_t i = 0; i < bin.points.size(); i++) {
        const auto& p = bin.points[i];
        pts += std::to_string(i + 7); // fixture point3D_id 7..9 (unused by the loader — checks only)
        for (double x : p.xyz) pts += " " + fmt17(x);
        for (int k = 0; k < 3; k++) pts += " " + std::to_string(int(p.rgb[k]));
        pts += " 0.5";
        if (i == 0) pts += " 1 0"; // one track pair (verifies the skip)
        pts += "\n";
    }
    write_file(dir / "points3D.txt", pts);

    const ColmapBundle txt = load_colmap_bundle(dir.string());

    SX_CHECK(txt.cameras.size() == bin.cameras.size());
    for (size_t i = 0; i < bin.cameras.size() && i < txt.cameras.size(); i++) {
        const auto &a = txt.cameras[i], &e = bin.cameras[i];
        SX_CHECK(a.id == e.id && a.model_id == e.model_id);
        SX_CHECK(a.width == e.width && a.height == e.height);
        SX_CHECK(a.params == e.params); // doubles match exactly (%.17g round trip)
        SX_CHECK(a.fx == e.fx && a.fy == e.fy && a.cx == e.cx && a.cy == e.cy);
    }
    SX_CHECK(txt.images.size() == bin.images.size());
    for (size_t i = 0; i < bin.images.size() && i < txt.images.size(); i++) {
        const auto &a = txt.images[i], &e = bin.images[i];
        SX_CHECK(a.id == e.id && a.camera_id == e.camera_id && a.name == e.name);
        for (int k = 0; k < 4; k++) SX_CHECK(a.qvec[k] == e.qvec[k]);
        for (int k = 0; k < 3; k++) SX_CHECK(a.tvec[k] == e.tvec[k]);
    }
    SX_CHECK(txt.points.size() == bin.points.size());
    for (size_t i = 0; i < bin.points.size() && i < txt.points.size(); i++) {
        const auto &a = txt.points[i], &e = bin.points[i];
        for (int k = 0; k < 3; k++) SX_CHECK(a.xyz[k] == e.xyz[k]);
        for (int k = 0; k < 3; k++) SX_CHECK(a.rgb[k] == e.rgb[k]);
    }
}

// ② Error paths: corrupt input is rejected (an explicit error carrying the file and line number) +
// an empty points3D is accepted + an error when neither file exists.
SX_TEST(colmap_txt_rejects_corrupt) {
    const std::string kValidCam = "1 PINHOLE 100 80 50.0 50.0 50.0 40.0\n";
    const std::string kValidImg = "1 1 0 0 0 0.5 0 2 1 a.jpg\n\n";
    const std::string kValidPts = "7 0 0 1 255 0 0 -1\n";

    auto make = [&](const std::string& name, const std::string& cams, const std::string& imgs,
                    const std::string& pts) {
        const fs::path dir = fresh_bundle_dir(name);
        write_file(dir / "cameras.txt", cams);
        write_file(dir / "images.txt", imgs);
        write_file(dir / "points3D.txt", pts);
        return dir.string();
    };

    // An unsupported (unknown) camera model — an explicit error on line 5, after 2 comment lines
    // and 1 blank line
    {
        const std::string msg = colmap_throw_msg(
            make("colmap_txt_bad_model",
                 "# c1\n# c2\n\n" + kValidCam + "2 FISHEYE_X 10 10 1 1 1\n", kValidImg, kValidPts));
        SX_CHECK(contains(msg, "unknown camera model FISHEYE_X"));
        SX_CHECK(contains(msg, "cameras.txt:5:")); // includes the file and line number
    }
    // A non-numeric token
    {
        const std::string msg = colmap_throw_msg(
            make("colmap_txt_bad_width", "1 PINHOLE abc 80 50 50 50 40\n", kValidImg, kValidPts));
        SX_CHECK(contains(msg, "invalid WIDTH 'abc'") && contains(msg, "cameras.txt:1:"));
    }
    // The number of params does not match the model
    {
        const std::string msg = colmap_throw_msg(
            make("colmap_txt_bad_nparams", "1 PINHOLE 100 80 50 50 50\n", kValidImg, kValidPts));
        SX_CHECK(contains(msg, "PINHOLE expects 4 params, got 3"));
    }
    // Too few tokens on an images.txt pose line
    {
        const std::string msg = colmap_throw_msg(
            make("colmap_txt_bad_pose", kValidCam, "1 1 0 0 0 0 0 0 1\n\n", kValidPts));
        SX_CHECK(contains(msg, "expected 10 tokens") && contains(msg, "images.txt:1:"));
    }
    // The POINTS2D line of the last image in images.txt is missing (truncated)
    {
        const std::string msg = colmap_throw_msg(
            make("colmap_txt_truncated_img", kValidCam, "1 1 0 0 0 0.5 0 2 1 a.jpg", kValidPts));
        SX_CHECK(contains(msg, "missing POINTS2D line"));
    }
    // points3D.txt: too few tokens / RGB out of range / an odd number of track tokens
    {
        const std::string msg = colmap_throw_msg(
            make("colmap_txt_bad_pt", kValidCam, kValidImg, "7 0 0 1 255 0 0\n"));
        SX_CHECK(contains(msg, "expected POINT3D_ID") && contains(msg, "points3D.txt:1:"));
    }
    {
        const std::string msg = colmap_throw_msg(
            make("colmap_txt_bad_rgb", kValidCam, kValidImg, "7 0 0 1 300 0 0 -1\n"));
        SX_CHECK(contains(msg, "RGB out of range"));
    }
    {
        const std::string msg = colmap_throw_msg(
            make("colmap_txt_bad_track", kValidCam, kValidImg, "7 0 0 1 255 0 0 -1 3\n"));
        SX_CHECK(contains(msg, "TRACK[]"));
    }
    // An empty points3D.txt (comments only) is accepted — the seed can be injected separately
    {
        const ColmapBundle b = load_colmap_bundle(
            make("colmap_txt_empty_pts", kValidCam, kValidImg, "# no points\n\n"));
        SX_CHECK(b.points.empty());
        SX_CHECK(b.cameras.size() == 1 && b.images.size() == 1);
        SX_CHECK(b.images[0].name == "a.jpg");
    }
    // Neither cameras.bin nor cameras.txt exists → an explicit error
    {
        const fs::path dir = fresh_bundle_dir("colmap_txt_neither");
        const std::string msg = colmap_throw_msg(dir.string());
        SX_CHECK(contains(msg, "cannot open") && contains(msg, "cameras.{bin,txt}"));
    }
}

// pose Task 6 — distortion camera models are rejected explicitly (this prevents silent breakage).
// splatx-metal only ever does pure PINHOLE projection, so derive_pinhole — which previously
// dropped k1 silently on SIMPLE_RADIAL and corrupted the intrinsics to 0 on RADIAL/OPENCV/FISHEYE
// and the like — is guarded with an explicit error (the model name + guidance to undistort).
// This point is shared by bin and txt, so the txt path serves as the representative case, checked
// together with PINHOLE showing no regression.
SX_TEST(colmap_rejects_distortion_models) {
    const std::string kValidImg = "1 1 0 0 0 0.5 0 2 1 a.jpg\n\n";
    const std::string kValidPts = "7 0 0 1 255 0 0 -1\n";

    auto reject_msg = [&](const std::string& name, const std::string& cams) {
        const fs::path dir = fresh_bundle_dir(name);
        write_file(dir / "cameras.txt", cams);
        write_file(dir / "images.txt", kValidImg);
        write_file(dir / "points3D.txt", kValidPts);
        return colmap_throw_msg(dir.string());
    };

    // SIMPLE_RADIAL (4 params: f cx cy k1) — previously the load succeeded, silently dropping k1.
    {
        const std::string msg =
            reject_msg("colmap_dist_simple_radial", "1 SIMPLE_RADIAL 100 80 50 50 40 0.01\n");
        SX_CHECK(contains(msg, "SIMPLE_RADIAL"));           // (a) the model name
        SX_CHECK(contains(msg, "unsupported camera model"));
        SX_CHECK(contains(msg, "undistort"));               // (b) guidance to undistort
    }
    // OPENCV (8 params) — previously the intrinsics were silently corrupted to 0.
    {
        const std::string msg =
            reject_msg("colmap_dist_opencv", "1 OPENCV 100 80 50 50 50 40 0.01 0.001 0 0\n");
        SX_CHECK(contains(msg, "OPENCV") && !contains(msg, "OPENCV_FISHEYE")); // exactly OPENCV
        SX_CHECK(contains(msg, "undistort"));
    }
    // No regression: pure PINHOLE still loads successfully, exactly as before (it is not rejected).
    {
        const fs::path dir = fresh_bundle_dir("colmap_dist_pinhole_ok");
        write_file(dir / "cameras.txt", "1 PINHOLE 100 80 50.0 50.0 50.0 40.0\n");
        write_file(dir / "images.txt", kValidImg);
        write_file(dir / "points3D.txt", kValidPts);
        const ColmapBundle b = load_colmap_bundle(dir.string());
        SX_CHECK(b.cameras.size() == 1 && b.cameras[0].model_id == 1);
        SX_CHECK_NEAR(b.cameras[0].fx, 50.0f, 0, 1e-4);
    }
}

namespace {

// Finds a real capture bundle from the iOS capture app (LiDAR mode —
// <bundle>/sparse/0/cameras.txt exists).
// Priority: the environment variable SPLATX_CAPTURE_BUNDLE (names the bundle root directly) →
// ~/Documents/SplatX/Captures/<LiDAR_*> (mirrors the path the app exposes through Files) →
// ~/Documents/SplatX/CapturePulls/<*>/<LiDAR_*> (the customary directory for copies pulled off the
// device, read-only).
// Returns every candidate in lexicographic order (integration Task 2 — the fallback smoke test
// picks the bundle whose dimensions are consistent).
std::vector<std::string> find_capture_bundles() {
    auto is_bundle = [](const fs::path& d) {
        std::error_code ec;
        return fs::exists(d / "sparse" / "0" / "cameras.txt", ec);
    };
    if (const char* env = std::getenv("SPLATX_CAPTURE_BUNDLE"); env && *env)
        if (is_bundle(env)) return {env};
    const char* home = std::getenv("HOME");
    if (!home || !*home) return {};
    auto is_lidar_dir = [](const fs::directory_entry& e) {
        std::error_code ec;
        return e.is_directory(ec) && e.path().filename().string().rfind("LiDAR", 0) == 0;
    };
    std::vector<std::string> found;
    const fs::path roots[] = {fs::path(home) / "Documents/SplatX/Captures",
                              fs::path(home) / "Documents/SplatX/CapturePulls"};
    for (const auto& root : roots) {
        std::error_code ec;
        for (fs::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
            if (is_lidar_dir(*it) && is_bundle(it->path())) {
                found.push_back(it->path().string());
                continue;
            }
            std::error_code ec2;
            if (!it->is_directory(ec2)) continue;
            for (fs::directory_iterator it2(it->path(), ec2), end2; !ec2 && it2 != end2;
                 it2.increment(ec2))
                if (is_lidar_dir(*it2) && is_bundle(it2->path())) found.push_back(it2->path().string());
        }
    }
    std::sort(found.begin(), found.end()); // deterministic order
    return found;
}

std::string find_capture_bundle() { // previous contract: first bundle lexicographically (fixed)
    const std::vector<std::string> found = find_capture_bundles();
    return found.empty() ? std::string{} : found.front();
}

} // namespace

// ③ Smoke test on a real iOS capture app bundle — it loads, and the camera/frame counts check
// out. When no real bundle exists, a synthetic one in exactly the format SplatXKit's
// ColmapModelWriter emits (fixed 9 decimal places, 2 lines per image — the second line blank,
// ERROR -1 + an empty track) stands in. Which of the two was used is printed to the log.
SX_TEST(colmap_capture_bundle_smoke) {
    std::string bundle = find_capture_bundle();
    size_t expected_images = 0;
    size_t expected_cameras = 1; // a LiDAR capture has a single PINHOLE (LidarColmapAssembler)
    if (!bundle.empty()) {
        std::error_code ec;
        for (fs::directory_iterator it(fs::path(bundle) / "images", ec), end; !ec && it != end;
             it.increment(ec)) {
            const std::string ext = it->path().extension().string();
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".JPG") expected_images++;
        }
        std::printf("  (smoke: real capture bundle %s, images/=%zu)\n", bundle.c_str(),
                    expected_images);
    } else {
        const fs::path dir = fresh_bundle_dir("colmap_capture_synth");
        fs::create_directories(dir / "sparse" / "0");
        // Replicates the byte format of SplatXKit's ColmapModelWriter (header comments included)
        write_file(dir / "sparse/0/cameras.txt",
                   "# Camera list with one line of data per camera:\n"
                   "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
                   "# Number of cameras: 1\n"
                   "1 PINHOLE 3840 2160 2630.486816406 2630.486816406 1927.362548828 "
                   "1080.766479492\n");
        write_file(dir / "sparse/0/images.txt",
                   "# Image list with two lines of data per image:\n"
                   "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
                   "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
                   "# Number of images: 2\n"
                   "1 0.199346713 -0.681712678 0.692767474 -0.124907751 0.004545137 "
                   "-0.013350213 0.007937477 1 0001.jpg\n\n"
                   "2 0.151923488 -0.667671467 0.708769770 -0.169644925 0.000084247 "
                   "-0.000010037 -0.000024466 1 0002.jpg\n\n");
        write_file(dir / "sparse/0/points3D.txt",
                   "# 3D point list with one line of data per point:\n"
                   "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
                   "# Number of points: 2\n"
                   "1 0.055174768 0.073448002 -0.410558969 160 194 191 -1.000000000\n"
                   "2 0.055137429 0.069971517 -0.408418536 166 200 196 -1.000000000\n");
        bundle = dir.string();
        expected_images = 2;
        std::printf("  (smoke: synthetic SplatXKit-format bundle — real capture not found)\n");
    }

    const ColmapBundle b = load_colmap_bundle(bundle + "/sparse/0");
    SX_CHECK(b.cameras.size() == expected_cameras);
    SX_CHECK(!b.cameras.empty() && b.cameras[0].model_id == 1); // PINHOLE
    SX_CHECK(!b.cameras.empty() && b.cameras[0].fx > 0.0f && b.cameras[0].width > 0);
    SX_CHECK(!b.images.empty());
    if (expected_images > 0) SX_CHECK(b.images.size() == expected_images);
    for (const auto& im : b.images) SX_CHECK(im.camera_id == 1 && !im.name.empty());
}

// ④ integration Task 2 — smoke test of the factor 4 fallback load on a real capture bundle. A
// capture bundle has no pre-resized images_4 folder, so resolve_images_dir must engage the fallback
// (images/ + downscale_on_load); and on a bundle whose source and camera dimensions are consistent,
// loading the first image downscaled by factor 4 must land exactly on the camera dimensions given
// by the same formula the intrinsics scale uses (downscaled_dim) — the condition for getting
// through the TrainC GT dims guard. Older EXIF-rotated bundles (where cameras.txt disagrees with
// the orientation the source decodes to — the concerns reported in task-1) are skipped: that
// mismatch is an error path the guard catches exactly as before, because even on the fallback path
// the loader derives its target from the native dimensions (that the guard is still in place is
// confirmed through the skip log).
// When no real bundle exists, a synthetic one (images/ PNG + txt, with odd downscaled dimensions)
// stands in — printed to the log.
SX_TEST(capture_bundle_factor4_fallback_smoke) {
    const int F = 4;
    std::string bundle;
    for (const std::string& cand : find_capture_bundles()) {
        const ColmapBundle cb = load_colmap_bundle(cand + "/sparse/0");
        if (cb.cameras.empty() || cb.images.empty()) continue;
        const splatx::ImageF32 raw =
            splatx::load_image_rgb_f32(cand + "/images/" + cb.images[0].name);
        if (raw.width == static_cast<int>(cb.cameras[0].width) &&
            raw.height == static_cast<int>(cb.cameras[0].height)) {
            bundle = cand;
            break;
        }
        std::printf("  (skip %s — original %dx%d != cameras.txt %lldx%lld, an old bundle with EXIF rotation)\n",
                    cand.c_str(), raw.width, raw.height,
                    static_cast<long long>(cb.cameras[0].width),
                    static_cast<long long>(cb.cameras[0].height));
    }
    if (!bundle.empty()) {
        std::printf("  (smoke: real capture bundle %s, factor %d fallback)\n", bundle.c_str(), F);
    } else {
        // Synthetic bundle — 321x243: 321/4 = 80.25 → 80, 243/4 = 60.75 → 61 (drives odd rounding)
        const fs::path dir = fresh_bundle_dir("colmap_capture_fallback_synth");
        fs::create_directories(dir / "sparse" / "0");
        fs::create_directories(dir / "images");
        splatx::ImageF32 img;
        img.width = 321;
        img.height = 243;
        img.rgb.assign(static_cast<size_t>(321) * 243 * 3, 0.5f);
        splatx::save_png_rgb_f32(img, (dir / "images" / "0001.png").string());
        write_file(dir / "sparse/0/cameras.txt", "1 PINHOLE 321 243 200.0 200.0 160.5 121.5\n");
        write_file(dir / "sparse/0/images.txt", "1 1 0 0 0 0 0 0 1 0001.png\n\n");
        write_file(dir / "sparse/0/points3D.txt", "# no points\n");
        bundle = dir.string();
        std::printf("  (smoke: synthetic bundle — real capture not found)\n");
    }

    const splatx::ImagesDirChoice ch = splatx::resolve_images_dir(bundle, F);
    SX_CHECK(ch.downscale_on_load); // no images_4 — the fallback engages
    SX_CHECK(ch.dir == bundle + "/images");

    const ColmapBundle b = load_colmap_bundle(bundle + "/sparse/0");
    const splatx::TrainingInput ti = splatx::training_input_from_colmap(b, ch.dir);
    SX_CHECK(!ti.cameras.empty());
    if (ti.cameras.empty()) return;
    const splatx::TrainingCamera& cam = ti.cameras[0]; // source resolution dims (before scaling)
    const int w1 = splatx::downscaled_dim(cam.width, F);  // same formula as the TrainC factor block
    const int h1 = splatx::downscaled_dim(cam.height, F);
    const splatx::ImageF32 dn = splatx::load_image_rgb_f32_downscaled(cam.image_path, F);
    std::printf("  (fallback load: %dx%d -> %dx%d, target %dx%d)\n", cam.width, cam.height,
                dn.width, dn.height, w1, h1);
    SX_CHECK(dn.width == w1 && dn.height == h1); // the condition for passing the GT dims guard
    SX_CHECK(dn.rgb.size() == static_cast<size_t>(w1) * h1 * 3);
    // Pixel range smoke test — 0..1 (the f32 conversion path is sound)
    size_t out_of_range = 0;
    for (float v : dn.rgb)
        if (!(v >= 0.0f && v <= 1.0f)) out_of_range++;
    SX_CHECK(out_of_range == 0);
}
