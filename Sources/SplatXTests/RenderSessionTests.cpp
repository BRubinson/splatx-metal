// Tests for the session-based render C API (Cinema Task 2) — the authoritative contract is the
// sx_render_session_* header comment in splatx_train_c.h. The verification axes:
//   (1) Pixel parity — the u8 output of a session frame matches **exactly** a direct call to
//       Trainer::render (f32) with the same PLY, camera, and default config, put through the same
//       quantization (the forward bit-exact determinism contract — 00-index. Even though the
//       backend is a separate instance, the f32 results are bit-identical, so exact u8 equality is
//       a well-founded expectation).
//   (2) Cross-check against sx_render_ply — pixels are compared only under the metallib gate
//       (sx_render_ply has no source-compilation fallback, so it cannot create a backend under the
//       default gate — the header contract).
//       The default-gate branch instead asserts the contract "(a) an explicit metallib failure is
//       an immediate error, with no fallback" on both surfaces (the test count is the same under
//       both gates — following the precedent of the SPLATX_CAPTURE_BUNDLE branch).
//   (3) The session reuse contract — multiple frames, resolution changes, and determinism (the same
//       camera gives bit-identical output).
//   (4) That the AA and sh_degree arguments take effect, and the error paths for invalid arguments.
// A new backend is created for each session, so when SPLATX_TEST_METALLIB is set, that path is
// injected as metallib_path in order to drive the same path end to end under the metallib gate too
// (when unset, kernel_src_dir() source compilation is used — following the precedent of
// train_c_stats_final_cc_psnr).
#include "TestHarness.h"
#include "TestPaths.h"
#include "TestUtils.h" // shared_metal_backend — for the direct comparison
#include "splatx_train_c.h"

#include "splatx/ColmapLoader.h"
#include "splatx/ImageIO.h"
#include "splatx/PlyIO.h"
#include "splatx/Trainer.h"
#include "splatx/TrainingInput.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace splatx;

namespace {

// A synthetic render scene — of the same family as make_smoke_gt_cloud in TrainerTests (a 4×4 grid
// on the z=3 plane, isotropic scale 0.13, opacity 0.85). Parameterized by sh_degree, with
// **non-zero shN in every band** (the 0.03·((j%7)−3) pattern) — this is what makes sh_degree
// truncation actually change the pixels (the basis for the sh_degree contract test).
SplatCloud make_session_cloud(int sh_degree) {
    SplatCloud c;
    c.count = 16;
    c.sh_degree = sh_degree;
    c.means.resize(16 * 3);
    c.scales.assign(16 * 3, std::log(0.13f));
    c.quats.assign(16 * 4, 0.0f);
    c.opacities.assign(16, std::log(0.85f / 0.15f)); // logit(0.85)
    c.sh0.resize(16 * 3);
    c.shN.resize(static_cast<size_t>(16) * c.sh_rest_coeffs() * 3);
    for (size_t j = 0; j < c.shN.size(); j++)
        c.shN[j] = 0.03f * static_cast<float>(static_cast<int>(j % 7) - 3);
    constexpr float kC0 = 0.28209479177387814f; // the SH DC normalization constant (Y_0^0)
    for (int i = 0; i < 16; i++) {
        const int gx = i % 4, gy = i / 4;
        c.means[i * 3 + 0] = -0.6f + 0.4f * static_cast<float>(gx);
        c.means[i * 3 + 1] = -0.6f + 0.4f * static_cast<float>(gy);
        c.means[i * 3 + 2] = 3.0f;
        c.quats[i * 4 + 0] = 1.0f;
        const float rgb[3] = {0.15f + 0.05f * static_cast<float>(i),
                              0.9f - 0.04f * static_cast<float>(i),
                              0.2f + 0.03f * static_cast<float>((i * 7) % 16)};
        for (int k = 0; k < 3; k++) c.sh0[i * 3 + k] = (rgb[k] - 0.5f) / kC0;
    }
    return c;
}

std::string write_session_ply(const std::string& name, int sh_degree, bool pad_sh_full) {
    const std::string path = sxtest::tmp_path(name);
    write_inria_ply(make_session_cloud(sh_degree), path, pad_sh_full);
    return path;
}

// A PINHOLE camera with identity rotation, a (tx,ty) translation, and the principal point at the
// center (world→cam, row-major — the translation components are [3] and [7]). At f=50 the z=3 grid
// fits inside a 64×48 frame (±0.6·50/3 = ±10px).
TrainingCamera make_camera(int w, int h, float f, float tx = 0.0f, float ty = 0.0f) {
    TrainingCamera cam;
    for (int i = 0; i < 16; i++) cam.viewmat[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    cam.viewmat[3] = tx;
    cam.viewmat[7] = ty;
    const float K[9] = {f,    0.0f, static_cast<float>(w) * 0.5f,
                        0.0f, f,    static_cast<float>(h) * 0.5f,
                        0.0f, 0.0f, 1.0f};
    std::memcpy(cam.K, K, sizeof(K));
    cam.width = w;
    cam.height = h;
    return cam;
}

// f32 [H,W,3] → RGBA8 — reproduces the quantization formula from the contract (identical to
// save_png_rgb_f32 and frame: NaN→0, clamp to [0,1], round(v·255), alpha 255). This is the expected
// value generator for the direct comparison.
std::vector<uint8_t> to_rgba8(const std::vector<float>& rgb, int w, int h) {
    const size_t n_px = static_cast<size_t>(w) * static_cast<size_t>(h);
    std::vector<uint8_t> out(n_px * 4);
    for (size_t p = 0; p < n_px; p++) {
        for (int c = 0; c < 3; c++) {
            float v = rgb[p * 3 + static_cast<size_t>(c)];
            if (!(v > 0.0f)) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            out[p * 4 + static_cast<size_t>(c)] = static_cast<uint8_t>(lrintf(v * 255.0f));
        }
        out[p * 4 + 3] = 255;
    }
    return out;
}

// Assembles the session arguments — including the metallib gate branch. The holder owns the string
// lifetimes (the pointers in args are valid only while the holder is alive).
struct SessionArgsHolder {
    std::string ply;
    std::string ksd = sxtest::kernel_src_dir();
    const char* lib = std::getenv("SPLATX_TEST_METALLIB");
    explicit SessionArgsHolder(std::string ply_path) : ply(std::move(ply_path)) {}
    SxRenderSessionArgs args(int antialiased = 0, int sh_degree = 0) const {
        SxRenderSessionArgs a{}; // the 0/NULL initialization convention (the header contract)
        a.ply_path = ply.c_str();
        if (lib && *lib)
            a.metallib_path = lib;
        else
            a.kernel_src_dir = ksd.c_str();
        a.antialiased = antialiased;
        a.sh_degree = sh_degree;
        return a;
    }
};

int32_t frame_with(SxRenderSession* s, const TrainingCamera& cam, std::vector<uint8_t>& out,
                   char* err, int32_t err_len) {
    out.assign(static_cast<size_t>(cam.width) * cam.height * 4, 0);
    return sx_render_session_frame(s, cam.viewmat, cam.K[0], cam.K[4], cam.K[2], cam.K[5],
                                   cam.width, cam.height, out.data(), err, err_len);
}

size_t count_nonzero_rgb(const std::vector<uint8_t>& rgba) {
    size_t n = 0;
    for (size_t i = 0; i < rgba.size(); i++)
        if (i % 4 != 3 && rgba[i] != 0) n++;
    return n;
}

// A synthetic COLMAP txt bundle (for the sx_render_ply cross-check — of the same family as the
// bundle builder in train_c_stats; rendering never opens the GT images, so images/ is not created
// and factor is 1). The camera: PINHOLE 64×48, f=50, principal point at the center = the same
// parameters as make_camera(64,48,50), with identity rotation and tvec 0.
std::string make_render_colmap_bundle(const std::string& name) {
    namespace fs = std::filesystem;
    const fs::path root = sxtest::tmp_path(name);
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root / "sparse/0");
    const auto write_file = [](const fs::path& p, const std::string& s) {
        std::ofstream f(p, std::ios::binary);
        f << s;
        if (!f) throw std::runtime_error("test: write failed " + p.string());
    };
    write_file(root / "sparse/0/cameras.txt", "1 PINHOLE 64 48 50 50 32 24\n");
    // IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME (plus an empty POINTS2D line). After sorting by
    // image name, index 0 = img000 (tvec 0 — facing the grid head-on).
    write_file(root / "sparse/0/images.txt", "1 1 0 0 0 0 0 0 1 img000.png\n\n"
                                             "2 1 0 0 0 0.2 0 0 1 img001.png\n\n");
    std::string pts;
    for (int i = 0; i < 8; i++)
        pts += std::to_string(i + 1) + " 0.1 0.1 " + std::to_string(3.0 + 0.2 * i) +
               " 128 128 128 0\n";
    write_file(root / "sparse/0/points3D.txt", pts);
    return root.string();
}

} // namespace

// (1) Pixel parity — a session frame vs a direct call to Trainer::render put through the same
// quantization, with exact u8 equality.
// The direct comparison uses the default TrainConfig (it must match what the session assembles
// internally — if the session assembly diverges, this test breaks: it is the contract gate).
SX_TEST(render_session_matches_trainer_render) {
    const SessionArgsHolder h(write_session_ply("render_session_parity.ply", 3, true));
    char err[512] = {};
    const SxRenderSessionArgs a = h.args();
    SxRenderSession* s = sx_render_session_open(&a, err, sizeof err);
    if (!s) std::printf("  open err=%s\n", err);
    SX_CHECK(s != nullptr);
    if (!s) return;

    const TrainingCamera cam = make_camera(64, 48, 50.0f);
    std::vector<uint8_t> got;
    const int32_t rc = frame_with(s, cam, got, err, sizeof err);
    if (rc != 0) std::printf("  frame rc=%d err=%s\n", rc, err);
    SX_CHECK(rc == 0);

    KernelBackend& b = sxtest::shared_metal_backend();
    SplatCloud cloud = read_inria_ply(h.ply);
    TrainConfig cfg; // defaults — the same assembly as the session (AA=0) and sx_render_ply
    Trainer tr(b, std::move(cloud), cfg);
    const std::vector<uint8_t> want = to_rgba8(tr.render(cam), cam.width, cam.height);

    SX_CHECK(got.size() == want.size());
    size_t bad = 0, first = 0;
    for (size_t i = 0; i < got.size() && i < want.size(); i++)
        if (got[i] != want[i]) {
            if (bad == 0) first = i;
            bad++;
        }
    if (bad)
        std::printf("  parity mismatch %zu/%zu bytes (first [%zu]: %d vs %d)\n", bad,
                    want.size(), first, got[first], want[first]);
    SX_CHECK(bad == 0);
    // Guards against a vacuous pass — if everything were background (0), the match above would be
    // meaningless (this guards against an everything-culled regression)
    SX_CHECK(count_nonzero_rgb(got) > 0);
    for (size_t p = 3; p < got.size(); p += 4) // the contract that alpha is fixed at 255
        if (got[p] != 255) {
            SX_CHECK(got[p] == 255);
            break;
        }
    sx_render_session_close(s);
}

// (3) Session reuse — multiple frames (different cameras give different output), determinism (the
// same camera gives bit-identical output), resolution changes (growth of the internal buffers), and
// the errors for invalid frame arguments.
SX_TEST(render_session_multi_frame_reuse) {
    const SessionArgsHolder h(write_session_ply("render_session_reuse.ply", 3, true));
    char err[512] = {};
    const SxRenderSessionArgs a = h.args();
    SxRenderSession* s = sx_render_session_open(&a, err, sizeof err);
    if (!s) std::printf("  open err=%s\n", err);
    SX_CHECK(s != nullptr);
    if (!s) return;

    const TrainingCamera camA = make_camera(64, 48, 50.0f);
    const TrainingCamera camB = make_camera(64, 48, 50.0f, 0.3f, 0.0f);
    const TrainingCamera camC = make_camera(64, 48, 50.0f, 0.0f, 0.3f);
    std::vector<uint8_t> fA, fB, fC, fA2;
    SX_CHECK(frame_with(s, camA, fA, err, sizeof err) == 0);
    SX_CHECK(frame_with(s, camB, fB, err, sizeof err) == 0);
    SX_CHECK(frame_with(s, camC, fC, err, sizeof err) == 0);
    SX_CHECK(count_nonzero_rgb(fA) > 0);
    // Different cameras → different output (the observable surface for "no crash" and "no state
    // contamination between frames")
    SX_CHECK(fA != fB);
    SX_CHECK(fA != fC);
    SX_CHECK(fB != fC);
    // Determinism — the same session with the same camera is bit-identical (the header contract)
    SX_CHECK(frame_with(s, camA, fA2, err, sizeof err) == 0);
    SX_CHECK(fA == fA2);

    // Resolution changes — enlarging (which grows the buffers) then shrinking, both succeeding with
    // non-empty content
    const TrainingCamera big = make_camera(96, 80, 75.0f);
    const TrainingCamera small = make_camera(32, 24, 25.0f);
    std::vector<uint8_t> fBig, fSmall;
    SX_CHECK(frame_with(s, big, fBig, err, sizeof err) == 0);
    SX_CHECK(count_nonzero_rgb(fBig) > 0);
    SX_CHECK(frame_with(s, small, fSmall, err, sizeof err) == 0);
    SX_CHECK(count_nonzero_rgb(fSmall) > 0);

    // Invalid frame arguments — -1 plus err_buf (the session MUST remain valid)
    std::vector<uint8_t> buf(static_cast<size_t>(64) * 48 * 4);
    err[0] = '\0';
    SX_CHECK(sx_render_session_frame(nullptr, camA.viewmat, 50, 50, 32, 24, 64, 48,
                                     buf.data(), err, sizeof err) == -1);
    SX_CHECK(err[0] != '\0');
    SX_CHECK(sx_render_session_frame(s, nullptr, 50, 50, 32, 24, 64, 48, buf.data(), err,
                                     sizeof err) == -1);
    SX_CHECK(sx_render_session_frame(s, camA.viewmat, 50, 50, 32, 24, 64, 48, nullptr, err,
                                     sizeof err) == -1);
    SX_CHECK(sx_render_session_frame(s, camA.viewmat, 50, 50, 32, 24, 0, 48, buf.data(), err,
                                     sizeof err) == -1);
    SX_CHECK(sx_render_session_frame(s, camA.viewmat, -50, 50, 32, 24, 64, 48, buf.data(),
                                     err, sizeof err) == -1);
    // A normal frame still works after a violation (the error does not contaminate the session)
    std::vector<uint8_t> fA3;
    SX_CHECK(frame_with(s, camA, fA3, err, sizeof err) == 0);
    SX_CHECK(fA == fA3);
    sx_render_session_close(s);
}

// (4a) The AA flag takes effect — with the same PLY and camera, antialiased 0 and 1 produce
// different output (the comp < 1 opacity compensation — the observable surface proving the
// TrainConfig.antialiased chain is wired end to end).
SX_TEST(render_session_antialiased_flag) {
    const SessionArgsHolder h(write_session_ply("render_session_aa.ply", 3, true));
    char err[512] = {};
    const TrainingCamera cam = make_camera(64, 48, 50.0f);

    const SxRenderSessionArgs a0 = h.args(/*antialiased=*/0);
    SxRenderSession* s0 = sx_render_session_open(&a0, err, sizeof err);
    if (!s0) std::printf("  open(aa=0) err=%s\n", err);
    SX_CHECK(s0 != nullptr);
    const SxRenderSessionArgs a1 = h.args(/*antialiased=*/1);
    SxRenderSession* s1 = sx_render_session_open(&a1, err, sizeof err);
    if (!s1) std::printf("  open(aa=1) err=%s\n", err);
    SX_CHECK(s1 != nullptr);
    if (!s0 || !s1) {
        sx_render_session_close(s0);
        sx_render_session_close(s1);
        return;
    }
    std::vector<uint8_t> classic, aa;
    SX_CHECK(frame_with(s0, cam, classic, err, sizeof err) == 0);
    SX_CHECK(frame_with(s1, cam, aa, err, sizeof err) == 0);
    SX_CHECK(count_nonzero_rgb(classic) > 0);
    SX_CHECK(count_nonzero_rgb(aa) > 0);
    SX_CHECK(classic != aa);
    sx_render_session_close(s0);
    sx_render_session_close(s1);
}

// (4b) The sh_degree contract — 0 derives it from the PLY, a lower value truncates (different
// output), an equal value does not truncate (identical output), and a higher or out-of-range value
// is an open error.
SX_TEST(render_session_sh_degree_contract) {
    const SessionArgsHolder h(write_session_ply("render_session_deg.ply", 3, true));
    char err[512] = {};
    const TrainingCamera cam = make_camera(64, 48, 50.0f);

    const SxRenderSessionArgs a_derive = h.args(0, /*sh_degree=*/0);
    SxRenderSession* s_derive = sx_render_session_open(&a_derive, err, sizeof err);
    if (!s_derive) std::printf("  open(derive) err=%s\n", err);
    SX_CHECK(s_derive != nullptr);
    const SxRenderSessionArgs a_low = h.args(0, /*sh_degree=*/1);
    SxRenderSession* s_low = sx_render_session_open(&a_low, err, sizeof err);
    if (!s_low) std::printf("  open(deg1) err=%s\n", err);
    SX_CHECK(s_low != nullptr);
    const SxRenderSessionArgs a_full = h.args(0, /*sh_degree=*/3);
    SxRenderSession* s_full = sx_render_session_open(&a_full, err, sizeof err);
    if (!s_full) std::printf("  open(deg3) err=%s\n", err);
    SX_CHECK(s_full != nullptr);
    if (s_derive && s_low && s_full) {
        std::vector<uint8_t> img_derive, img_low, img_full;
        SX_CHECK(frame_with(s_derive, cam, img_derive, err, sizeof err) == 0);
        SX_CHECK(frame_with(s_low, cam, img_low, err, sizeof err) == 0);
        SX_CHECK(frame_with(s_full, cam, img_full, err, sizeof err) == 0);
        SX_CHECK(count_nonzero_rgb(img_derive) > 0);
        SX_CHECK(img_low != img_derive);   // truncating the (non-zero) upper bands → the color changes
        SX_CHECK(img_full == img_derive);  // specifying the PLY degree exactly = the no-truncation path
    }
    sx_render_session_close(s_derive);
    sx_render_session_close(s_low);
    sx_render_session_close(s_full);

    // Higher than the PLY degree — requesting sh_degree 3 against an unpadded deg1 PLY
    // (pad_sh_full=false, so read infers deg1) is an open error (it fails fast before the backend is
    // created — no GPU is used)
    const SessionArgsHolder h1(write_session_ply("render_session_deg1.ply", 1, false));
    const SxRenderSessionArgs a_over = h1.args(0, /*sh_degree=*/3);
    err[0] = '\0';
    SX_CHECK(sx_render_session_open(&a_over, err, sizeof err) == nullptr);
    SX_CHECK(err[0] != '\0');
    // Out of range (4 or more, or negative) — rejected before the ply is even read
    const SxRenderSessionArgs a_range = h1.args(0, /*sh_degree=*/4);
    err[0] = '\0';
    SX_CHECK(sx_render_session_open(&a_range, err, sizeof err) == nullptr);
    SX_CHECK(err[0] != '\0');
}

// (4c) Invalid open arguments — NULL, a missing field, or a nonexistent file gives NULL plus
// err_buf (a fail-fast that uses no GPU).
SX_TEST(render_session_invalid_open_args) {
    char err[512] = {};
    SX_CHECK(sx_render_session_open(nullptr, err, sizeof err) == nullptr);
    SX_CHECK(err[0] != '\0');

    SxRenderSessionArgs a{}; // ply_path is missing
    err[0] = '\0';
    SX_CHECK(sx_render_session_open(&a, err, sizeof err) == nullptr);
    SX_CHECK(err[0] != '\0');

    const std::string missing = sxtest::tmp_path("render_session_no_such.ply");
    std::filesystem::remove(missing);
    a.ply_path = missing.c_str();
    err[0] = '\0';
    SX_CHECK(sx_render_session_open(&a, err, sizeof err) == nullptr);
    SX_CHECK(err[0] != '\0');

    sx_render_session_close(nullptr); // the no-op contract (no crash)
}

// (2) The sx_render_ply cross-check — the rationale for the branch is in the comment at the top of
// this file (under the default gate, sx_render_ply cannot create a backend, so a pixel comparison
// is impossible).
SX_TEST(render_session_matches_sx_render_ply) {
    const std::string ply = write_session_ply("render_session_xcheck.ply", 3, true);
    const std::string root = make_render_colmap_bundle("render_session_xcheck_bundle");
    char err[512] = {};
    const char* lib = std::getenv("SPLATX_TEST_METALLIB");

    if (lib && *lib) {
        // The metallib gate: render the same PLY through the same camera on both APIs and assert
        // exact pixel equality. The camera is reproduced with the same convention as TrainC (load →
        // sort by image name → index).
        const ColmapBundle bundle = load_colmap_bundle(root + "/sparse/0");
        TrainingInput ti = training_input_from_colmap(bundle, root + "/images");
        std::sort(ti.cameras.begin(), ti.cameras.end(),
                  [](const TrainingCamera& x, const TrainingCamera& y) {
                      return x.image_path < y.image_path;
                  });
        SX_CHECK(ti.cameras.size() == 2);
        const TrainingCamera cam = ti.cameras[0]; // img000 — identity rotation, tvec 0

        const std::string png = sxtest::tmp_path("render_session_xcheck.png");
        const int32_t rc =
            sx_render_ply(ply.c_str(), root.c_str(), 1, 0, lib, png.c_str(), err, sizeof err);
        if (rc != 0) std::printf("  sx_render_ply rc=%d err=%s\n", rc, err);
        SX_CHECK(rc == 0);

        SessionArgsHolder h(ply);
        const SxRenderSessionArgs a = h.args();
        SxRenderSession* s = sx_render_session_open(&a, err, sizeof err);
        if (!s) std::printf("  open err=%s\n", err);
        SX_CHECK(s != nullptr);
        if (rc != 0 || !s) {
            sx_render_session_close(s);
            return;
        }
        std::vector<uint8_t> got;
        SX_CHECK(frame_with(s, cam, got, err, sizeof err) == 0);
        sx_render_session_close(s);

        // The PNG 8-bit round-trip is lossless: load returns u8/255, so round(v·255) recovers the
        // original u8 for every value (this holds across all of 0..255 — the relative error of
        // ~1e-7 is far below 0.5).
        const ImageF32 img = load_image_rgb_f32(png);
        SX_CHECK(img.width == cam.width && img.height == cam.height);
        const std::vector<uint8_t> want = to_rgba8(img.rgb, img.width, img.height);
        size_t bad = 0, first = 0;
        for (size_t i = 0; i < got.size() && i < want.size(); i++)
            if (got[i] != want[i]) {
                if (bad == 0) first = i;
                bad++;
            }
        if (bad)
            std::printf("  xcheck mismatch %zu/%zu bytes (first [%zu]: %d vs %d)\n", bad,
                        want.size(), first, got[first], want[first]);
        SX_CHECK(got.size() == want.size());
        SX_CHECK(bad == 0);
        SX_CHECK(count_nonzero_rgb(got) > 0); // guards against a vacuous pass
    } else {
        // The default gate: asserts that both surfaces are consistent with the contract "(a) an
        // explicit metallib failure is an immediate error, with no fallback" (the session does not
        // fall back on an explicit failure, even when a valid kernel_src_dir is available).
        const std::string bogus = sxtest::tmp_path("render_session_no_such.metallib");
        std::filesystem::remove(bogus);
        const std::string png = sxtest::tmp_path("render_session_xcheck_fail.png");
        const int32_t rc = sx_render_ply(ply.c_str(), root.c_str(), 1, 0, bogus.c_str(),
                                         png.c_str(), err, sizeof err);
        SX_CHECK(rc != 0);
        SX_CHECK(err[0] != '\0');

        const std::string ksd = sxtest::kernel_src_dir();
        SxRenderSessionArgs a{};
        a.ply_path = ply.c_str();
        a.metallib_path = bogus.c_str();
        a.kernel_src_dir = ksd.c_str(); // a valid fallback exists — contract (a) still takes priority
        err[0] = '\0';
        SX_CHECK(sx_render_session_open(&a, err, sizeof err) == nullptr);
        SX_CHECK(err[0] != '\0');
    }
}
