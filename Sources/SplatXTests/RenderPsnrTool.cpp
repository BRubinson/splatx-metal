// The PLY render comparison subcommand (SPZ Task 3 — `splatx-core-tests --render-psnr`).
// Forward-renders two PLY files through the same single COLMAP camera view and prints the
// render-to-render PSNR — used to quantify the SPZ round-trip quantization loss (the measurement
// tool behind docs/bench/2026-07-12-spz-roundtrip.md).
// This is a subcommand harness, not a test — there is no registration macro (SX_TEST), the same
// pattern as --dump-accutile (IsectTests). The CLI has no "load a PLY, then render" combination
// (rendering is train-only), so it lives in the test harness.
// The camera pipeline follows the same convention as TrainC (sx_train_run): sorted by image name
// → factor scaling (the per-axis round(orig/factor) ratio) → index selection. Rendering goes
// through Trainer::render (forward-only, full sh_degree — the same path as --save-preview), then a
// [0,1] clamp, then splatx::psnr.
#include "TestUtils.h" // shared_metal_backend (including the SPLATX_TEST_METALLIB override)
#include "splatx/ColmapLoader.h"
#include "splatx/PlyIO.h"
#include "splatx/Trainer.h"
#include "splatx/TrainingInput.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace sxtest {

namespace {

// Renders a single PLY (the Trainer is destroyed after rendering — this prevents the peak memory
// caused by holding both clouds at once)
std::vector<float> render_ply(const std::string& ply_path, const splatx::TrainingCamera& cam) {
    splatx::SplatCloud cloud = splatx::read_inria_ply(ply_path);
    std::printf("render: %s (N=%lld, sh_degree %d)\n", ply_path.c_str(),
                static_cast<long long>(cloud.count), cloud.sh_degree);
    std::fflush(stdout);
    splatx::TrainConfig cfg; // defaults — render() is forward-only, so LR and schedules are irrelevant
    splatx::Trainer trainer(shared_metal_backend(), std::move(cloud), cfg);
    std::vector<float> img = trainer.render(cam);
    // [0,1] clamp — the psnr contract (Trainer.h: the caller clamps, gsplat simple_trainer.py:1240)
    for (float& v : img) v = std::clamp(v, 0.0f, 1.0f);
    return img;
}

} // namespace

int render_psnr_compare(const std::string& ply_a, const std::string& ply_b,
                        const std::string& data_dir, int factor, int cam_index) {
    try {
        if (factor != 1 && factor != 2 && factor != 4 && factor != 8) {
            std::printf("render-psnr: factor must be 1|2|4|8\n");
            return 1;
        }
        // Load the COLMAP cameras, sort by image name, and apply factor scaling — the same
        // procedure as TrainC (so that the index matches the camera number of a training run)
        const splatx::ColmapBundle bundle = splatx::load_colmap_bundle(data_dir + "/sparse/0");
        const std::string images_dir =
            data_dir + (factor == 1 ? "/images" : "/images_" + std::to_string(factor));
        splatx::TrainingInput ti = splatx::training_input_from_colmap(bundle, images_dir);
        if (ti.cameras.empty()) {
            std::printf("render-psnr: no cameras in COLMAP bundle\n");
            return 1;
        }
        std::sort(ti.cameras.begin(), ti.cameras.end(),
                  [](const splatx::TrainingCamera& a, const splatx::TrainingCamera& b) {
                      return a.image_path < b.image_path;
                  });
        const size_t idx = static_cast<size_t>(std::clamp<int64_t>(
            cam_index, 0, static_cast<int64_t>(ti.cameras.size()) - 1));
        splatx::TrainingCamera cam = ti.cameras[idx];
        if (factor > 1) {
            const int w0 = cam.width, h0 = cam.height;
            const int w1 = static_cast<int>(std::lround(static_cast<double>(w0) / factor));
            const int h1 = static_cast<int>(std::lround(static_cast<double>(h0) / factor));
            const double sx = static_cast<double>(w1) / w0, sy = static_cast<double>(h1) / h0;
            cam.K[0] = static_cast<float>(cam.K[0] * sx); // fx
            cam.K[2] = static_cast<float>(cam.K[2] * sx); // cx
            cam.K[4] = static_cast<float>(cam.K[4] * sy); // fy
            cam.K[5] = static_cast<float>(cam.K[5] * sy); // cy
            cam.width = w1;
            cam.height = h1;
        }
        std::printf("render-psnr: camera %zu/%zu (%dx%d, factor %d)\n", idx, ti.cameras.size(),
                    cam.width, cam.height, factor);
        std::fflush(stdout);

        const std::vector<float> img_a = render_ply(ply_a, cam);
        const std::vector<float> img_b = render_ply(ply_b, cam);
        const double v = splatx::psnr(img_a.data(), img_b.data(),
                                      static_cast<int64_t>(img_a.size()));
        std::printf("render-psnr: %.3f dB (%s vs %s, camera %zu)\n", v, ply_a.c_str(),
                    ply_b.c_str(), idx);
        return 0;
    } catch (const std::exception& e) {
        std::printf("render-psnr failed: %s\n", e.what());
        return 1;
    }
}

} // namespace sxtest
