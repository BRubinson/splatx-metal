#include "TestHarness.h"
#include "TestPaths.h" // fixture_dir() — the single source of truth for path resolution (runtime injection)
#include "splatx/PlyIO.h"
#include "splatx/TrainingInput.h"
#include "splatx/ColmapLoader.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>

using namespace splatx;

static SplatCloud make_cloud(int64_t n, int degree) {
    SplatCloud c;
    c.count = n;
    c.sh_degree = degree;
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    auto fill = [&](std::vector<float>& v, size_t len) { v.resize(len); for (auto& x : v) x = d(rng); };
    fill(c.means, n * 3); fill(c.scales, n * 3); fill(c.quats, n * 4);
    fill(c.opacities, n); fill(c.sh0, n * 3);
    fill(c.shN, n * c.sh_rest_coeffs() * 3);
    return c;
}

SX_TEST(ply_roundtrip_degree3) {
    SplatCloud a = make_cloud(50, 3);
    const std::string path_s = sxtest::tmp_path("splatx_test_d3.ply"); // iOS sandbox compatible (TestPaths.h)
    const char* path = path_s.c_str();
    write_inria_ply(a, path);
    SplatCloud b = read_inria_ply(path);
    std::remove(path);

    SX_CHECK(b.count == 50);
    SX_CHECK(b.sh_degree == 3);          // 45 f_rest entries → degree 3 is inferred
    SX_CHECK(b.sh_rest_coeffs() == 15);
    for (size_t i = 0; i < a.means.size(); i++) SX_CHECK(a.means[i] == b.means[i]);
    for (size_t i = 0; i < a.shN.size(); i++) SX_CHECK(a.shN[i] == b.shN[i]);
    for (size_t i = 0; i < a.quats.size(); i++) SX_CHECK(a.quats[i] == b.quats[i]);
    for (size_t i = 0; i < a.opacities.size(); i++) SX_CHECK(a.opacities[i] == b.opacities[i]);
}

SX_TEST(ply_roundtrip_degree2) {
    // The default write zero-pads f_rest to 45 entries (design §8.1 — PlyIO.h), so the
    // degree-preserving round-trip is verified through the explicit pad_sh_full=false path (the
    // raw 24-entry layout contract).
    SplatCloud a = make_cloud(10, 2);    // f_rest 8*3=24 entries
    const std::string path_s = sxtest::tmp_path("splatx_test_d2.ply"); // iOS sandbox compatible (TestPaths.h)
    const char* path = path_s.c_str();
    write_inria_ply(a, path, /*pad_sh_full=*/false);
    SplatCloud b = read_inria_ply(path);
    std::remove(path);
    SX_CHECK(b.sh_degree == 2);
    for (size_t i = 0; i < a.shN.size(); i++) SX_CHECK(a.shN[i] == b.shN[i]);

    // The round-trip of the default (padding) write: 45 entries → deg3 is inferred, the original
    // coefficients are preserved in the lower 8 positions per channel, and the upper 7 (k >= 8) are
    // exactly 0.
    write_inria_ply(a, path);
    SplatCloud p = read_inria_ply(path);
    std::remove(path);
    SX_CHECK(p.sh_degree == 3 && p.sh_rest_coeffs() == 15);
    for (int64_t i = 0; i < 10; i++)
        for (int k = 0; k < 15; k++)
            for (int ch = 0; ch < 3; ch++) {
                const float got = p.shN[(i * 15 + k) * 3 + ch];
                const float want = k < 8 ? a.shN[(i * 8 + k) * 3 + ch] : 0.0f;
                SX_CHECK(got == want);
            }
}

// ===== SH zero-padding — the property set viewers require (design §8.1) =====
// An audit of MetalSplatter SplatIO (tag 1.0.1 = the iOS capture app's pin,
// SplatPLYSceneReader.swift):
// the vertex element requires x/y/z, f_dc_0..2, scale_0..2, opacity, and rot_0..3 as floats, and if
// f_rest_0 is present then all 45 of f_rest_0..44 are required as floats (a fixed loop — a deg1
// file with 9 entries is rejected with "No property named f_rest_9", and 24 entries is rejected at
// f_rest_24 → the minimum padding is 45).
// deg0 is accepted through the branch where f_rest is absent — this also pins down that no
// unnecessary padding is emitted.
SX_TEST(ply_export_sh_padding_viewer_property_set) {
    const std::string path_s = sxtest::tmp_path("splatx_test_pad.ply"); // iOS sandbox compatible (TestPaths.h)
    const char* path = path_s.c_str();
    const auto read_header = [&]() {
        FILE* f = std::fopen(path, "rb");
        char buf[8192] = {0};
        std::fread(buf, 1, sizeof(buf) - 1, f);
        std::fclose(f);
        std::string h(buf);
        return h.substr(0, h.find("end_header"));
    };
    const auto has_prop = [](const std::string& h, const std::string& n) {
        return h.find("property float " + n + "\n") != std::string::npos;
    };

    // deg1 (K=3) default write → satisfies the full SplatIO required set
    SplatCloud a = make_cloud(4, 1);
    write_inria_ply(a, path);
    {
        const std::string h = read_header();
        for (const char* n : {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2",
                              "rot_3"})
            SX_CHECK(has_prop(h, n));
        for (int i = 0; i < 45; i++) SX_CHECK(has_prop(h, "f_rest_" + std::to_string(i)));
        SX_CHECK(!has_prop(h, "f_rest_45")); // exactly 45 entries
    }
    // The round-trip: deg3 is inferred, the original coefficients (the lower 3 per channel) are
    // preserved, and the padding is 0
    SplatCloud b = read_inria_ply(path);
    std::remove(path);
    SX_CHECK(b.count == 4 && b.sh_degree == 3);
    for (int64_t i = 0; i < 4; i++)
        for (int k = 0; k < 15; k++)
            for (int ch = 0; ch < 3; ch++) {
                const float got = b.shN[(i * 15 + k) * 3 + ch];
                const float want = k < 3 ? a.shN[(i * 3 + k) * 3 + ch] : 0.0f;
                SX_CHECK(got == want);
            }

    // deg0 (K=0): 0 f_rest entries as is (accepted by the SplatIO SH0 branch — this avoids 180 B
    // of padding per row)
    SplatCloud z = make_cloud(2, 0);
    write_inria_ply(z, path);
    {
        const std::string h = read_header();
        SX_CHECK(h.find("f_rest_") == std::string::npos);
    }
    SplatCloud zr = read_inria_ply(path);
    std::remove(path);
    SX_CHECK(zr.count == 2 && zr.sh_degree == 0);
}

SX_TEST(ply_header_field_order_is_inria) {
    SplatCloud a = make_cloud(2, 1);
    const std::string path_s = sxtest::tmp_path("splatx_test_hdr.ply"); // iOS sandbox compatible (TestPaths.h)
    const char* path = path_s.c_str();
    write_inria_ply(a, path);
    FILE* f = std::fopen(path, "rb");
    char buf[4096] = {0};
    std::fread(buf, 1, sizeof(buf) - 1, f);
    std::fclose(f);
    std::remove(path);
    std::string h(buf);
    SX_CHECK(h.find("format binary_little_endian 1.0") != std::string::npos);
    // The order: f_rest_0 follows f_dc_2, and the last property is rot_3.
    // The default write zero-pads f_rest to 45 entries even for deg1 (design §8.1) — opacity
    // follows f_rest_44.
    SX_CHECK(h.find("property float f_dc_2\nproperty float f_rest_0") != std::string::npos);
    SX_CHECK(h.find("property float f_rest_44\nproperty float opacity") != std::string::npos);
    SX_CHECK(h.find("property float rot_3\nend_header") != std::string::npos);
}

SX_TEST(ply_non_float_property_throws) {
    // A header containing a uchar property MUST throw immediately, rather than silently corrupting
    // the data
    const std::string path_s = sxtest::tmp_path("splatx_test_uchar.ply"); // iOS sandbox compatible (TestPaths.h)
    const char* path = path_s.c_str();
    FILE* f = std::fopen(path, "wb");
    const char* header =
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 1\n"
        "property float x\n"
        "property uchar red\n"
        "end_header\n";
    std::fwrite(header, 1, std::strlen(header), f);
    std::fclose(f);

    bool threw = false;
    try { read_inria_ply(path); } catch (...) { threw = true; }
    std::remove(path);
    SX_CHECK(threw);
}

// ===== The export filter (Phase 2 Task 5 — the PlyIO.h convention: compared against gsplat
// export_splats:629–651) =====
// Mixes dead (low-opacity), NaN, Inf, and huge-but-finite scale rows into a synthetic cloud to
// verify both classification and preservation.
SX_TEST(export_filter_low_opacity_nonfinite_scale_policy) {
    constexpr float kInf = std::numeric_limits<float>::infinity();
    constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

    SplatCloud c = make_cloud(6, 1); // K = 3
    for (auto& o : c.opacities) o = 2.0f;      // sigmoid(2) ≈ 0.881 — unambiguously alive
    c.opacities[1] = -7.0f;                    // dead: sigmoid(-7) ≈ 9.1e-4 <= 0.005
    c.means[2 * 3 + 1] = kNaN;                 // nonfinite (means)
    c.shN[(3 * 3 + 0) * 3 + 2] = kInf;         // nonfinite (shN — the first coefficient of row 3)
    c.scales[4 * 3 + 0] = 30.0f;               // huge but finite (exp(30) ≈ 1e13) — preserved
    c.opacities[5] = kNaN;                     // opacity NaN — classified nonfinite, not low-opacity

    // Imitates the MCMC cap_max arena: even when the vector tail extends past the count rows and
    // holds garbage (NaN), it MUST be ignored
    c.means.resize(c.means.size() + 3, kNaN);
    c.opacities.resize(c.opacities.size() + 1, kNaN);

    ExportFilterStats st;
    const SplatCloud f = filter_cloud_for_export(c, 0.005f, &st);
    SX_CHECK(st.input == 6);
    SX_CHECK(st.removed_nonfinite == 3);   // rows 2, 3, and 5
    SX_CHECK(st.removed_low_opacity == 1); // row 1
    SX_CHECK(st.kept == 2 && f.count == 2);
    SX_CHECK(f.sh_degree == 1);
    // The surviving rows = {0, 4}, with order preserved and values untransformed (there is no scale
    // clamp — the gsplat standard ply convention; the [-20,20] in exporter.py is exclusive to chunk
    // quantization in the compressed format, so not applying it is pinned down as the convention)
    for (int k = 0; k < 3; k++) SX_CHECK(f.means[0 * 3 + k] == c.means[0 * 3 + k]);
    for (int k = 0; k < 3; k++) SX_CHECK(f.means[1 * 3 + k] == c.means[4 * 3 + k]);
    SX_CHECK(f.scales[1 * 3 + 0] == 30.0f);
    for (int k = 0; k < 4; k++) SX_CHECK(f.quats[1 * 4 + k] == c.quats[4 * 4 + k]);
    for (int k = 0; k < 9; k++) SX_CHECK(f.shN[1 * 9 + k] == c.shN[4 * 9 + k]);
    // The filter output passes a PLY round-trip (the distribution surface contract — the default
    // write zero-pads f_rest to 45 entries, so a deg1 input is inferred as deg3. Design §8.1 /
    // PlyIO.h)
    const std::string path_s = sxtest::tmp_path("splatx_test_export_filter.ply"); // iOS sandbox compatible (TestPaths.h)
    const char* path = path_s.c_str();
    write_inria_ply(f, path);
    const SplatCloud g = read_inria_ply(path);
    std::remove(path);
    SX_CHECK(g.count == 2 && g.sh_degree == 3);

    // The boundary: sigmoid(0) == 0.5 exactly — the comparison is a strict `>`, so a row exactly at
    // the threshold is removed (dead is the complement of `<=`, the same predicate as gsplat
    // exporter.py:236)
    SplatCloud b = make_cloud(2, 0);
    b.opacities[0] = 0.0f; // sigmoid == min_opacity → removed
    b.opacities[1] = 0.1f; // sigmoid > 0.5 → kept
    ExportFilterStats st2;
    const SplatCloud fb = filter_cloud_for_export(b, 0.5f, &st2);
    SX_CHECK(st2.removed_low_opacity == 1 && fb.count == 1);
    SX_CHECK(fb.opacities[0] == 0.1f);

    // With min_opacity <= 0 the opacity filter is naturally inert (sigmoid > 0 always holds) — only
    // NaN and Inf are applied
    ExportFilterStats st3;
    const SplatCloud fc = filter_cloud_for_export(c, 0.0f, &st3);
    SX_CHECK(st3.removed_low_opacity == 0 && st3.removed_nonfinite == 3 && fc.count == 3);
}

SX_TEST(training_input_from_colmap_builds_cameras) {
    ColmapBundle b = load_colmap_bundle(sxtest::fixture_dir() + "/colmap_mini/sparse/0");
    TrainingInput ti = training_input_from_colmap(b, "/data/images");

    SX_CHECK(ti.cameras.size() == 2);
    SX_CHECK(ti.cameras[0].image_path == "/data/images/frame_000.jpg");
    SX_CHECK(ti.cameras[0].width == 1920 && ti.cameras[0].height == 1440);
    SX_CHECK_NEAR(ti.cameras[0].K[0], 1500.0f, 0, 1e-4);  // fx
    SX_CHECK_NEAR(ti.cameras[0].K[5], 720.0f, 0, 1e-4);   // cy
    // The identity qvec (1,0,0,0) → rotation = I, translation = tvec
    SX_CHECK_NEAR(ti.cameras[0].viewmat[0], 1.0f, 0, 1e-6);
    SX_CHECK_NEAR(ti.cameras[0].viewmat[5], 1.0f, 0, 1e-6);
    SX_CHECK_NEAR(ti.cameras[0].viewmat[3], 0.1f, 0, 1e-6);   // t.x
    SX_CHECK_NEAR(ti.cameras[0].viewmat[15], 1.0f, 0, 1e-6);
    SX_CHECK(ti.seed_points.size() == 9 && ti.seed_rgb.size() == 9);
    SX_CHECK_NEAR(ti.seed_points[8], 5.0f, 0, 1e-6);          // the z of the 3rd point
}
