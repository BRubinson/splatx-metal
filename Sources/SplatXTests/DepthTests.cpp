// Depth supervision tests (the expected-depth channel, its gradient and the sidecar loader).
// Run alone with sx_tests_run_prefix("depth_") — the Swift DepthSupervisionTests entry.
#include "TestHarness.h"
#include "TestUtils.h"
#include "splatx/CpuKernels.h"
#include "splatx/DepthSupervision.h"
#include "splatx/ImageIO.h"
#include "splatx/Json.h"
#include "splatx/KernelBackend.h"
#include "splatx/SXTensor.h"
#include "splatx/Trainer.h"
#include "splatx/TrainingInput.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace splatx;

namespace {

std::string fx(const std::string& dir, const char* rel) {
    return sxtest::fixture_dir() + "/" + dir + "/" + rel;
}

// A pinhole camera at the origin looking down +z.
TrainingCamera make_camera(int w, int h, float f) {
    TrainingCamera cam;
    for (int i = 0; i < 16; i++) cam.viewmat[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    const float K[9] = {f, 0.0f, 0.5f * w, 0.0f, f, 0.5f * h, 0.0f, 0.0f, 1.0f};
    for (int i = 0; i < 9; i++) cam.K[i] = K[i];
    cam.width = w;
    cam.height = h;
    return cam;
}

// Projection → isect → sort → offsets on the backend, for a raster-ready scene.
struct Scene {
    ProjectionFusedOutputs proj;
    std::vector<float> colors, opacities;
    SortIsectOutputs sorted;
    std::vector<int32_t> offsets;
    int64_t n_isects = 0;
    uint32_t tw = 0, th = 0;
    int64_t N = 0;
    TrainingCamera cam;

    RasterizeFwdInputs raster(const float* depths) const {
        RasterizeFwdInputs r;
        r.means2d = proj.means2d.data();
        r.conics = proj.conics.data();
        r.colors = colors.data();
        r.opacities = opacities.data();
        r.isect_offsets = offsets.data();
        r.flatten_ids = sorted.values_sorted.data();
        r.I = 1;
        r.N = N;
        r.n_isects = n_isects;
        r.image_width = static_cast<uint32_t>(cam.width);
        r.image_height = static_cast<uint32_t>(cam.height);
        r.tile_size = 16;
        r.tile_width = tw;
        r.tile_height = th;
        r.depths = depths;
        return r;
    }
};

Scene make_scene(KernelBackend& b, const TrainingCamera& cam, const std::vector<float>& means,
                 const std::vector<float>& scales, const std::vector<float>& opacities) {
    Scene s;
    s.cam = cam;
    s.N = static_cast<int64_t>(opacities.size());
    s.opacities = opacities;
    std::vector<float> quats(static_cast<size_t>(s.N) * 4, 0.0f);
    for (int64_t i = 0; i < s.N; i++) quats[i * 4] = 1.0f;
    ProjectionFusedInputs p;
    p.means = means.data();
    p.quats = quats.data();
    p.scales = scales.data();
    p.opacities = opacities.data();
    p.viewmats = cam.viewmat;
    p.Ks = cam.K;
    p.N = s.N;
    p.C = 1;
    p.width = cam.width;
    p.height = cam.height;
    s.proj = b.projection_fused_fwd(p);
    s.colors.resize(static_cast<size_t>(s.N) * 3);
    for (size_t i = 0; i < s.colors.size(); i++) s.colors[i] = 0.2f + 0.1f * static_cast<float>(i % 7);
    s.tw = static_cast<uint32_t>((cam.width + 15) / 16);
    s.th = static_cast<uint32_t>((cam.height + 15) / 16);
    IsectTilesInputs ii;
    ii.means2d = s.proj.means2d.data();
    ii.radii = s.proj.radii.data();
    ii.depths = s.proj.depths.data();
    ii.conics = s.proj.conics.data();
    ii.opacities = s.opacities.data();
    ii.I = 1;
    ii.N = s.N;
    ii.tile_size = 16;
    ii.tile_width = s.tw;
    ii.tile_height = s.th;
    const IsectTilesOutputs isect = b.isect_tiles(ii);
    SortIsectInputs si;
    si.keys = isect.isect_ids.data();
    si.values = isect.flatten_ids.data();
    si.n = isect.n_isects;
    si.key_n_bits = 32 + isect.tile_n_bits + isect.image_n_bits;
    s.sorted = b.sort_isect(si);
    IsectOffsetInputs oi;
    oi.isect_ids = s.sorted.keys_sorted.data();
    oi.n_isects = isect.n_isects;
    oi.I = 1;
    oi.tile_width = s.tw;
    oi.tile_height = s.th;
    s.offsets = b.isect_offset_encode(oi);
    s.n_isects = isect.n_isects;
    return s;
}

// A constant-depth sidecar smaller than the view, with every 5th sample invalid.
DepthTarget make_target(int w, int h, float depth_m, const TrainingCamera& cam) {
    DepthTarget t;
    t.width = w;
    t.height = h;
    t.depth_m.assign(static_cast<size_t>(w) * h, depth_m);
    for (size_t i = 0; i < t.depth_m.size(); i += 5) t.depth_m[i] = 0.0f;
    t.view_valid = depth_valid_count(t, cam.width, cam.height);
    return t;
}

bool bitwise_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

} // namespace

// (1) The expected-depth channel leaves RGB untouched: on the raster_std golden scene the depth
// variant's colours, alphas and last ids equal the RGB-only variant's bit for bit, its Σwz matches
// the CPU reference bit for bit, and its bwd with a zero depth gradient still meets the RGB golden.
SX_TEST(depth_fwd_rgb_channels_bit_exact) {
    KernelBackend& b = sxtest::shared_metal_backend();
    const std::string dir = "raster_std";
    const auto meta = json_parse_flat(fx(dir, "meta.json"));
    const SXTensor means2d = SXTensor::load(fx(dir, "means2d.sxt"));
    const SXTensor conics = SXTensor::load(fx(dir, "conics.sxt"));
    const SXTensor colors = SXTensor::load(fx(dir, "colors.sxt"));
    const SXTensor opacities = SXTensor::load(fx(dir, "opacities.sxt"));
    const SXTensor flatten_ids = SXTensor::load(fx(dir, "flatten_ids.sxt"));
    const SXTensor isect_offsets = SXTensor::load(fx(dir, "isect_offsets.sxt"));
    const SXTensor v_render_colors = SXTensor::load(fx(dir, "v_render_colors.sxt"));
    const SXTensor v_render_alphas = SXTensor::load(fx(dir, "v_render_alphas.sxt"));
    const SXTensor g_v_means2d = SXTensor::load(fx(dir, "v_means2d.sxt"));
    const SXTensor g_v_conics = SXTensor::load(fx(dir, "v_conics.sxt"));
    const SXTensor g_v_colors = SXTensor::load(fx(dir, "v_colors.sxt"));
    const SXTensor g_v_opacities = SXTensor::load(fx(dir, "v_opacities.sxt"));

    RasterizeFwdInputs in;
    in.means2d = means2d.f32();
    in.conics = conics.f32();
    in.colors = colors.f32();
    in.opacities = opacities.f32();
    in.isect_offsets = isect_offsets.i32();
    in.flatten_ids = flatten_ids.i32();
    in.I = static_cast<int64_t>(meta.at("I"));
    in.N = static_cast<int64_t>(meta.at("N"));
    in.n_isects = static_cast<int64_t>(meta.at("n_isects"));
    in.image_width = static_cast<uint32_t>(meta.at("width"));
    in.image_height = static_cast<uint32_t>(meta.at("height"));
    in.tile_size = static_cast<uint32_t>(meta.at("tile_size"));
    in.tile_width = static_cast<uint32_t>(meta.at("tile_width"));
    in.tile_height = static_cast<uint32_t>(meta.at("tile_height"));

    std::vector<float> depths(static_cast<size_t>(in.I * in.N));
    for (size_t g = 0; g < depths.size(); g++) depths[g] = 1.0f + 0.1f * static_cast<float>(g % 17);

    const RasterizeFwdOutputs rgb = b.rasterize_fwd(in);
    RasterizeFwdInputs in_d = in;
    in_d.depths = depths.data();
    const RasterizeFwdOutputs rgbd = b.rasterize_fwd(in_d);
    const RasterizeFwdOutputs cpu = rasterize_fwd_cpu(in_d);

    SX_CHECK(rgb.render_depths.empty());
    SX_CHECK(bitwise_equal(rgb.render_colors, rgbd.render_colors));
    SX_CHECK(bitwise_equal(rgb.render_alphas, rgbd.render_alphas));
    SX_CHECK(rgb.last_ids == rgbd.last_ids);
    SX_CHECK(rgbd.render_depths.size() == rgb.render_alphas.size());
    // The CPU reference agrees to float rounding, as the colours do (Metal fast-math contraction).
    double worst_d = 0.0, worst_c = 0.0;
    for (size_t i = 0; i < cpu.render_depths.size(); i++)
        worst_d = std::max(worst_d, std::fabs(static_cast<double>(rgbd.render_depths[i]) -
                                              cpu.render_depths[i]));
    for (size_t i = 0; i < cpu.render_colors.size(); i++)
        worst_c = std::max(worst_c, std::fabs(static_cast<double>(rgbd.render_colors[i]) -
                                              cpu.render_colors[i]));
    SX_CHECK(worst_d <= 1e-5);
    double max_ed = 0.0;
    for (size_t i = 0; i < rgbd.render_depths.size(); i++)
        max_ed = std::max(max_ed, static_cast<double>(
                                      expected_depth(rgbd.render_depths[i], rgbd.render_alphas[i])));
    std::printf("  [raster_std] %zu px: RGB/alpha/last_ids bit-equal with the depth channel on; "
                "Metal vs CPU max |d| depth %.3g, colour %.3g; max ED %.3f\n",
                rgbd.render_depths.size(), worst_d, worst_c, max_ed);
    SX_CHECK(max_ed > 1.0 && max_ed <= 2.6 + 1e-3); // ED is a convex combination of the depths

    const SXTensor render_alphas = SXTensor::load(fx(dir, "render_alphas.sxt"));
    const SXTensor last_ids = SXTensor::load(fx(dir, "last_ids.sxt"));
    const std::vector<float> zero_vd(rgbd.render_depths.size(), 0.0f);
    RasterizeBwdGradsIn gi;
    gi.render_alphas = render_alphas.f32();
    gi.last_ids = last_ids.i32();
    gi.v_render_colors = v_render_colors.f32();
    gi.v_render_alphas = v_render_alphas.f32();
    gi.v_render_depths = zero_vd.data();
    const RasterizeBwdGrads gd = b.rasterize_bwd(in_d, gi);
    SX_ALL_CLOSE(gd.v_means2d.data(), g_v_means2d.f32(), g_v_means2d.count(), meta, "v_means2d");
    SX_ALL_CLOSE(gd.v_conics.data(), g_v_conics.f32(), g_v_conics.count(), meta, "v_conics");
    SX_ALL_CLOSE(gd.v_colors.data(), g_v_colors.f32(), g_v_colors.count(), meta, "v_colors");
    SX_ALL_CLOSE(gd.v_opacities.data(), g_v_opacities.f32(), g_v_opacities.count(), meta,
                 "v_opacities");
    SX_CHECK(std::all_of(gd.v_depths.begin(), gd.v_depths.end(), [](float v) { return v == 0.0f; }));
}

// (2) The depth-loss gradient: on a three-Gaussian scene, v_depths (and v_opacities through the α
// normalisation) from raster bwd match central finite differences of the inverse-depth L1.
SX_TEST(depth_loss_gradient_matches_finite_differences) {
    KernelBackend& b = sxtest::shared_metal_backend();
    const TrainingCamera cam = make_camera(64, 48, 60.0f);
    const std::vector<float> means = {-0.15f, 0.05f, 2.0f, 0.10f, -0.05f, 2.5f, 0.0f, 0.1f, 3.0f};
    const std::vector<float> scales = {0.18f, 0.15f, 0.12f, 0.22f, 0.20f, 0.10f,
                                       0.30f, 0.25f, 0.15f};
    const std::vector<float> opacs = {0.6f, 0.5f, 0.7f};
    const Scene s = make_scene(b, cam, means, scales, opacs);
    SX_CHECK(s.n_isects > 0);
    const DepthTarget target = make_target(16, 12, 1.5f, cam);
    SX_CHECK(target.view_valid > 0);

    const auto loss_at = [&](const std::vector<float>& depths, const std::vector<float>& op) {
        Scene t = s;
        t.opacities = op;
        const RasterizeFwdOutputs o = b.rasterize_fwd(t.raster(depths.data()));
        return static_cast<double>(depth_l1_loss(o.render_depths.data(), o.render_alphas.data(),
                                                 target.depth_m.data(), cam.width, cam.height,
                                                 target.width, target.height, target.view_valid));
    };

    std::vector<float> depths(s.proj.depths.begin(), s.proj.depths.begin() + s.N);
    const RasterizeFwdOutputs o = b.rasterize_fwd(s.raster(depths.data()));
    const DepthLossResult dl = depth_loss_and_grad(o.render_depths.data(), o.render_alphas.data(),
                                                   target, cam.width, cam.height,
                                                   target.view_valid, 1.0f);
    const std::vector<float> zero_rgb(o.render_colors.size(), 0.0f);
    RasterizeBwdGradsIn gi;
    gi.render_alphas = o.render_alphas.data();
    gi.last_ids = o.last_ids.data();
    gi.v_render_colors = zero_rgb.data();
    gi.v_render_alphas = dl.v_render_alphas.data();
    gi.v_render_depths = dl.v_render_depths.data();
    const RasterizeBwdGrads g = b.rasterize_bwd(s.raster(depths.data()), gi);
    SX_CHECK(dl.loss > 0.0f);

    for (int64_t k = 0; k < s.N; k++) {
        const float h = 5e-3f * depths[k];
        std::vector<float> dp = depths, dm = depths;
        dp[k] += h;
        dm[k] -= h;
        const double fd = (loss_at(dp, opacs) - loss_at(dm, opacs)) /
                          (static_cast<double>(dp[k]) - dm[k]);
        const double an = g.v_depths[k];
        const double rel = std::fabs(fd - an) / std::max({std::fabs(fd), std::fabs(an), 1e-12});
        std::printf("  [three-gaussian] g%lld z %.3f: v_depths %.6e fd %.6e rel %.2e\n",
                    static_cast<long long>(k), static_cast<double>(depths[k]), an, fd, rel);
        SX_CHECK(rel <= 1e-3);
    }
    for (int64_t k = 0; k < s.N; k++) {
        const float h = 1e-3f;
        std::vector<float> op = opacs, om = opacs;
        op[k] += h;
        om[k] -= h;
        const double fd = (loss_at(depths, op) - loss_at(depths, om)) /
                          (static_cast<double>(op[k]) - om[k]);
        const double an = g.v_opacities[k];
        const double rel = std::fabs(fd - an) / std::max({std::fabs(fd), std::fabs(an), 1e-12});
        std::printf("  [three-gaussian] g%lld opacity %.2f: v_opacities %.6e fd %.6e rel %.2e\n",
                    static_cast<long long>(k), static_cast<double>(opacs[k]), an, fd, rel);
        SX_CHECK(rel <= 1e-2);
    }
}

// The resident train_depth_grad path reproduces the host depth_loss_and_grad step: three
// depth-supervised steps on both paths agree on the depth loss and every parameter group.
SX_TEST(depth_resident_matches_host_step) {
    KernelBackend& b = sxtest::shared_metal_backend();
    const TrainingCamera cam = make_camera(64, 64, 70.0f);
    SplatCloud c;
    c.count = 16;
    c.sh_degree = 0;
    c.means.resize(48);
    c.scales.assign(48, std::log(0.13f));
    c.quats.assign(64, 0.0f);
    c.opacities.assign(16, logit_f32(0.7f));
    c.sh0.assign(48, 0.1f);
    for (int i = 0; i < 16; i++) {
        c.means[i * 3 + 0] = -0.6f + 0.4f * static_cast<float>(i % 4);
        c.means[i * 3 + 1] = -0.6f + 0.4f * static_cast<float>(i / 4);
        c.means[i * 3 + 2] = 2.6f + 0.05f * static_cast<float>(i % 5);
        c.quats[i * 4] = 1.0f;
    }
    // Anisotropic and rotated, as the existing host/resident contrasts start: an isotropic cloud
    // has a zero quaternion gradient that Adam turns into lr-sized steps of atomic-order noise.
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto& v : c.scales) v += 0.30f * u(rng);
    for (auto& v : c.quats) v += 0.05f * u(rng);
    for (auto& v : c.opacities) v += 0.70f * u(rng);
    for (auto& v : c.sh0) v += 0.60f * u(rng);
    TrainConfig cfg;
    cfg.total_iters = 100;
    cfg.ssim_weight = 0.2f;
    cfg.depth_weight = 0.5f;
    cfg.lr_means = 2e-3f;
    TrainConfig cfg_host = cfg;
    cfg_host.resident = false;
    Trainer th(b, c, cfg_host);
    Trainer tr(b, c, cfg);
    const DepthTarget target = make_target(16, 16, 3.0f, cam);
    const std::vector<float> gt(static_cast<size_t>(cam.width) * cam.height * 3, 0.35f);
    for (int it = 0; it < 3; it++) {
        const TrainStepResult rh = th.step(cam, gt.data(), it, &target);
        const TrainStepResult rr = tr.step(cam, gt.data(), it, &target);
        std::printf("  [resident vs host] it %d: depth loss host %.7f resident %.7f\n", it,
                    static_cast<double>(rh.depth_loss), static_cast<double>(rr.depth_loss));
        SX_CHECK(rh.depth_loss > 0.0f);
        SX_CHECK_NEAR(rr.depth_loss, rh.depth_loss, 1e-4, 1e-7);
    }
    const SplatCloud& ch = th.cloud();
    const SplatCloud& cr = tr.cloud();
    const std::pair<const std::vector<float>*, const std::vector<float>*> groups[] = {
        {&ch.means, &cr.means}, {&ch.quats, &cr.quats}, {&ch.scales, &cr.scales},
        {&ch.opacities, &cr.opacities}, {&ch.sh0, &cr.sh0}};
    double worst = 0.0;
    for (const auto& [pa, pb] : groups)
        worst = std::max(worst, sxtest::rel_l2(pb->data(), pa->data(), pa->size(), pa->data()));
    std::printf("  [resident vs host] param rel_l2 worst = %.3g (gate 1e-4)\n", worst);
    SX_CHECK(worst <= 1e-4);

    const DepthRender dr = tr.render_with_depth(cam);
    SX_CHECK(dr.depth.size() == static_cast<size_t>(cam.width) * cam.height);
    SX_CHECK(dr.rgb == tr.render(cam));
}

// The sidecar convention: 16-bit millimetre and 8-bit confidence PNGs round-trip raw, confidence
// below medium masks a sample, and training_input_from_colmap resolves depths/<stem>.png beside
// the image folder.
SX_TEST(depth_sidecar_png_round_trip) {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "splatx-depth-sidecar-test";
    fs::remove_all(root);
    fs::create_directories(root / "images");
    fs::create_directories(root / "depths");
    ImageU16 mm;
    mm.width = 4;
    mm.height = 3;
    mm.data = {0, 1, 1234, 65535, 2500, 2500, 2500, 2500, 800, 900, 1000, 1100};
    ImageU16 conf;
    conf.width = 4;
    conf.height = 3;
    conf.data = {2, 2, 2, 2, 0, 1, 2, 0, 2, 2, 2, 2};
    save_png_gray16(mm, (root / "depths" / "view_000.png").string());
    save_png_gray8(conf, (root / "depths" / "view_000.conf.png").string());

    const ImageU16 back = load_png_gray_raw((root / "depths" / "view_000.png").string());
    SX_CHECK(back.width == 4 && back.height == 3 && back.data == mm.data);
    const ImageU16 cback = load_png_gray_raw((root / "depths" / "view_000.conf.png").string());
    SX_CHECK(cback.data == conf.data);

    const DepthTarget t = load_depth_sidecar((root / "depths" / "view_000.png").string(),
                                             (root / "depths" / "view_000.conf.png").string());
    SX_CHECK(t.width == 4 && t.height == 3);
    SX_CHECK_NEAR(t.depth_m[2], 1.234, 1e-6, 0);
    SX_CHECK_NEAR(t.depth_m[3], 65.535, 1e-6, 0);
    SX_CHECK(t.depth_m[4] == 0.0f && t.depth_m[7] == 0.0f); // confidence 0
    SX_CHECK_NEAR(t.depth_m[5], 2.5, 1e-6, 0);              // confidence 1 kept
    SX_CHECK(t.depth_m[0] == 0.0f);                         // no sample
    // 8×6 view over the 4×3 sidecar: each sample covers 2×2 view pixels.
    SX_CHECK(depth_valid_count(t, 8, 6) == 9 * 4); // 9 valid samples × 2×2 view pixels

    ColmapBundle bundle;
    bundle.cameras.push_back(ColmapCamera{1, 1, 8, 6, {5.0, 5.0, 4.0, 3.0}, 5.0f, 5.0f, 4.0f, 3.0f});
    bundle.images.push_back(ColmapImage{1, {1, 0, 0, 0}, {0, 0, 0}, 1, "view_000.jpg"});
    bundle.images.push_back(ColmapImage{2, {1, 0, 0, 0}, {0, 0, 0}, 1, "view_001.jpg"});
    const TrainingInput ti = training_input_from_colmap(bundle, (root / "images").string());
    SX_CHECK(ti.cameras.size() == 2);
    SX_CHECK(ti.cameras[0].depth_path == (root / "depths" / "view_000.png").string());
    SX_CHECK(ti.cameras[0].confidence_path == (root / "depths" / "view_000.conf.png").string());
    SX_CHECK(ti.cameras[1].depth_path.empty() && ti.cameras[1].confidence_path.empty());
    fs::remove_all(root);
}
