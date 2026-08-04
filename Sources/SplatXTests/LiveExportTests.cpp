// app round 2 Task 1 — live PLY export C API integration (SxTrainArgs.live_ply_path/
// live_ply_every). The authoritative contract is the comment at the end of SxTrainArgs in
// splatx_train_c.h (step boundary, shared export filter, tmp+rename atomicity, non-fatal).
// Only the wiring contract is asserted here: after training on a small synthetic bundle, the live
// file exists, the PLY vertex count is > 0, and no ".tmp" leftover remains.
// The synthetic COLMAP txt bundle is a local copy of the make_synthetic_colmap_txt_bundle pattern
// in TrainerTests.cpp (separate translation units — it sits in an anonymous namespace and cannot be
// shared. 64×48, 3 views, gradients; arbitrary content is sufficient).
// The zero-init of SxTrainArgs leaves ssim_weight = 0 — the SSIM path is bypassed entirely (so the
// H,W >= 11 constraint does not apply). A Metal device is required — there is no skip
// (the same premise as the existing integration tests).
#include "TestHarness.h"
#include "TestPaths.h" // tmp_path()/kernel_src_dir() — never hardcode paths (iOS sandbox)
#include "splatx/ImageIO.h"
#include "splatx/PlyIO.h" // read_inria_ply — asserts the vertex count of the live file
#include "splatx_train_c.h"
#include <cmath> // std::isfinite — asserts the first row of raw snapshot values is readable
#include <cstdint>
#include <cstdio>
#include <cstdlib> // std::getenv — passes SPLATX_TEST_METALLIB through (backend built per run)
#include <filesystem>
#include <fstream>
#include <string>

using namespace splatx;

namespace {

// Synthetic COLMAP txt bundle: root/sparse/0/{cameras,images,points3D}.txt + root/images/*.png
// (factor 1 → images/). What is under verification is the live export wiring, so arbitrary content
// is sufficient.
std::string make_live_export_bundle(const std::string& name) {
    namespace fs = std::filesystem;
    const int W = 64, H = 48, kCams = 3;
    const fs::path root = sxtest::tmp_path(name);
    std::error_code ec;
    fs::remove_all(root, ec); // remove leftovers from a previous run
    fs::create_directories(root / "sparse/0");
    fs::create_directories(root / "images");
    const auto write_file = [](const fs::path& p, const std::string& s) {
        std::ofstream f(p, std::ios::binary);
        f << s;
        if (!f) throw std::runtime_error("test: write failed " + p.string());
    };

    // cameras.txt — one shared PINHOLE: fx=fy=50, principal point at the center (the points fall
    // inside every view's frustum)
    write_file(root / "sparse/0/cameras.txt", "1 PINHOLE 64 48 50 50 32 24\n");

    // images.txt — identity rotation + translation along x; the second line (POINTS2D) is blank
    // (the parser contract)
    std::string imgs;
    for (int i = 0; i < kCams; i++) {
        char line[128];
        std::snprintf(line, sizeof(line), "%d 1 0 0 0 %.6f 0 0 1 img%03d.png\n\n", i + 1,
                      0.15 * (i - (kCams - 1) / 2.0), i);
        imgs += line;
    }
    write_file(root / "sparse/0/images.txt", imgs);

    // points3D.txt — a deterministic scatter in front of the cameras (z 3..5) (LCG — the same
    // coefficients as TrainerTests)
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

    // RGB gradient PNGs with a different phase per view (dims match the camera — passes the TrainC
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

SX_TEST(train_c_live_ply_export) {
    namespace fs = std::filesystem;
    const std::string root = make_live_export_bundle("splatx_live_export_bundle");
    const std::string ksd = sxtest::kernel_src_dir();
    const char* lib = std::getenv("SPLATX_TEST_METALLIB");

    const std::string out = sxtest::tmp_path("splatx_live_export_out.ply");
    const std::string live = sxtest::tmp_path("splatx_live_export_live.ply");
    const std::string live_tmp = live + ".tmp";
    std::error_code ec;
    fs::remove(live, ec); // clear leftovers so the "exists" assertion sees this run's output
    fs::remove(live_tmp, ec);

    SxTrainArgs a{}; // 0/NULL init, then only the needed fields (header contract — ssim_weight 0)
    a.data_dir = root.c_str();
    a.factor = 1;
    a.iters = 40; // live_ply_every 10 → 4 updates, at completed iters 10/20/30/40
    a.max_points = 200;
    a.sh_degree = 1;
    a.out_path = out.c_str();
    a.log_every = 1000; // suppress stdout noise
    a.resident = 1;
    a.live_ply_path = live.c_str();
    a.live_ply_every = 10;
    if (lib && *lib)
        a.metallib_path = lib;
    else
        a.kernel_src_dir = ksd.c_str();
    char err[512] = {};
    const int32_t rc = sx_train_run_args(&a, err, sizeof(err));
    if (rc != 0) std::printf("  sx_train_run_args rc=%d err=%s\n", rc, err);
    SX_CHECK(rc == 0);

    // The live file exists and the vertex count is > 0 (read_inria_ply — via parsing the header's
    // element vertex)
    SX_CHECK(fs::exists(live));
    if (fs::exists(live)) {
        const SplatCloud cloud = read_inria_ply(live);
        std::printf("  [live export] N=%lld\n", static_cast<long long>(cloud.count));
        SX_CHECK(cloud.count > 0);
    }
    // Atomicity: after a successful rename no ".tmp" leftover remains (the failure path is under
    // the same cleanup contract)
    SX_CHECK(!fs::exists(live_tmp));
}

// app round 3 Task 1 — the in-memory snapshot callback (SxTrainArgs.snapshot_cb/user/every). The
// authoritative contract is the SxSplatSnapshot comment in splatx_train_c.h. Only the wiring
// contract is asserted: the number of deliveries (iters/every), the iter phase
// ((it+1) % every == 0 + monotonically increasing), count > 0, sh_degree consistent with the
// training argument, shN non-NULL when deg > 0, and a finite first row of raw values (the filter
// removes NaN/Inf rows, so every row that passes is finite).
// snapshot_cb is a C function pointer — a capture-less lambda plus the user pointer carries the
// result struct out.
namespace {

struct SnapshotProbe {
    int calls = 0;
    int32_t last_iter = 0;      // doubles as the phase check (vs. previous iter in the callback)
    int64_t last_count = 0;
    int32_t last_sh_degree = -1;
    bool phase_ok = true;       // each iter is a multiple of every, and increases monotonically
    bool pointers_ok = true;    // required pointers non-NULL + shN convention + finite first row
};

} // namespace

SX_TEST(train_c_snapshot_callback) {
    const std::string root = make_live_export_bundle("splatx_snapshot_cb_bundle");
    const std::string ksd = sxtest::kernel_src_dir();
    const char* lib = std::getenv("SPLATX_TEST_METALLIB");
    const std::string out = sxtest::tmp_path("splatx_snapshot_cb_out.ply");

    SnapshotProbe probe;
    SxTrainArgs a{}; // 0/NULL init, then only the needed fields (header contract — ssim_weight 0)
    a.data_dir = root.c_str();
    a.factor = 1;
    a.iters = 40; // snapshot_every 10 → 4 deliveries, at completed iters 10/20/30/40
    a.max_points = 200;
    a.sh_degree = 1; // deg > 0 — so the shN non-NULL convention is actually exercised
    a.out_path = out.c_str();
    a.log_every = 1000; // suppress stdout noise
    a.resident = 1;
    a.snapshot_every = 10;
    a.snapshot_user = &probe;
    a.snapshot_cb = [](const SxSplatSnapshot* s, void* user) {
        auto* p = static_cast<SnapshotProbe*>(user);
        p->calls++;
        // Phase: a multiple of every(=10) + monotonically increasing (no captures allowed — every
        // is kept in sync with the test value by hand)
        if (s->iter % 10 != 0 || s->iter <= p->last_iter) p->phase_ok = false;
        p->last_iter = s->iter;
        p->last_count = s->count;
        p->last_sh_degree = s->sh_degree;
        if (!s->means || !s->quats || !s->scales || !s->opacities || !s->sh0 ||
            (s->sh_degree > 0 && !s->shN)) {
            p->pointers_ok = false;
        } else if (s->count > 0 &&
                   !(std::isfinite(s->means[0]) && std::isfinite(s->quats[0]) &&
                     std::isfinite(s->scales[0]) && std::isfinite(s->opacities[0]) &&
                     std::isfinite(s->sh0[0]) &&
                     (s->sh_degree == 0 || std::isfinite(s->shN[0])))) {
            p->pointers_ok = false; // an unreadable first row in the raw domain = a wiring bug
        }
    };
    if (lib && *lib)
        a.metallib_path = lib;
    else
        a.kernel_src_dir = ksd.c_str();
    char err[512] = {};
    const int32_t rc = sx_train_run_args(&a, err, sizeof(err));
    if (rc != 0) std::printf("  sx_train_run_args rc=%d err=%s\n", rc, err);
    SX_CHECK(rc == 0);

    std::printf("  [snapshot cb] calls=%d last_iter=%d last_count=%lld sh_degree=%d\n",
                probe.calls, probe.last_iter, static_cast<long long>(probe.last_count),
                probe.last_sh_degree);
    SX_CHECK(probe.calls == 4);            // 40 iters / every 10 (the last iter included)
    SX_CHECK(probe.last_iter == 40);
    SX_CHECK(probe.last_count > 0);        // rows pass the filter even in tiny synthetic training
    SX_CHECK(probe.last_sh_degree == 1);   // consistent with the training argument
    SX_CHECK(probe.phase_ok);
    SX_CHECK(probe.pointers_ok);
}
