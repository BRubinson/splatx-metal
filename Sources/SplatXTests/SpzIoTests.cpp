// SPZ Task 2/3 — tests for SpzIO (save_spz/load_spz). The reload in the Task 2 tests uses the
// vendored spz::loadSpz directly (load_spz did not exist at the time) — it is kept because it also
// demonstrates the coordinate system empirically (a raw load flips y and z). The Task 3 tests go
// through load_spz. The authoritative contract for the encoding conventions is the SpzIO.h comment.
//
// ===== Measured basis for the tolerances (the quantization resolution of ThirdParty/spz v2.1.0
// load-spz.cc) =====
// positions: 24-bit fixed point, fractionalBits=12 (packGaussians:270) — resolution 2^-12 ≈
//            2.441e-4, rounding error ≤ 2^-13 ≈ 1.221e-4 → atol 1.3e-4
// scales(log): uint8 (s+10)·16 (:291) — step 1/16, error ≤ 1/32 = 0.03125 → atol 0.032
//              (representable range [-10, 5.9375] — the test values are constructed within it)
// quats: smallest-three, 9 bits of magnitude plus sign per component (:245) — step (1/√2)/511 ≈
//        1.384e-3, per-component error ≤ 6.92e-4; the largest component is reconstructed as
//        sqrt(1-Σq²), so the propagated error is ≤ ~3e-3
//        → atol 4e-3 (compared after normalization and sign canonicalization (q~-q))
// opacity: uint8 in sigmoid space (:301) — error ≤ 1/510 ≈ 1.96e-3 → compared in sigmoid space
//          with atol 2e-3 (in logit space the error near saturation is unbounded — byte 255 →
//          logit +inf)
// sh0: uint8 with colorScale=0.15 (:45,:306) — step 1/38.25 ≈ 2.614e-2, error ≤ 1.31e-2
//      → atol 1.4e-2 (representable range about ±3.33)
// shN band 1 (coefficients 0..2): quantizeSH 5-bit buckets (:312,:318) — error ≤ (0.5+4)/128 ≈
//      3.52e-2 → atol 3.6e-2
// shN bands 2 and 3 (coefficients 3..14): 4-bit buckets (:313,:323) — error ≤ (0.5+8)/128 ≈
//      6.64e-2 → atol 6.7e-2 (the quantizeSH representable range is [-1, 127/128] — the test
//      values are constructed at ±0.7)
#include "TestHarness.h"
#include "TestPaths.h" // tmp_path() — a temp path compatible with the iOS sandbox
#include "load-spz.h"
#include "splatx/PlyIO.h" // the Task 3 round-trip chain (write/read_inria_ply)
#include "splatx/SpzIO.h"

#include <cmath>
#include <cstdio>
#include <limits>

using namespace splatx;

namespace {

double sig(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// Normalizes the expected value (w-first, denormalized input allowed); since q and -q are the same
// rotation, the sign is flipped when the dot product with the loaded value (xyzw) is < 0 before
// comparing components (spz packing canonicalizes to a positive largest component in the RUB frame,
// so which sign comes back is frame-dependent — canonicalizing by the dot product is robust).
void check_quat_near(const float* expect_wfirst, const float* loaded_xyzw, double atol) {
    double e[4] = {expect_wfirst[0], expect_wfirst[1], expect_wfirst[2], expect_wfirst[3]};
    const double n = std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + e[3] * e[3]);
    for (double& v : e) v /= n;
    double a[4] = {loaded_xyzw[3], loaded_xyzw[0], loaded_xyzw[1], loaded_xyzw[2]}; // → w-first
    const double dot = e[0] * a[0] + e[1] * a[1] + e[2] * a[2] + e[3] * a[3];
    if (dot < 0) for (double& v : a) v = -v;
    for (int k = 0; k < 4; k++) SX_CHECK_NEAR(a[k], e[k], 0.0, atol);
}

} // namespace

SX_TEST(spz_save_roundtrip_quantization_tolerances) {
    SplatCloud c;
    c.count = 4;
    c.sh_degree = 3; // K = 15
    // Every value is constructed within the spz representable range (see the rationale in the
    // header comment)
    c.means = {0.5f,  -1.25f, 2.0f,   //
               -3.5f, 0.75f,  -0.5f,  //
               10.0f, -20.0f, 30.0f,  //
               0.03f, 0.4f,   -0.9f};
    c.scales = {-4.0f, -2.5f, -1.0f, -6.75f, 0.5f,  2.0f,
                -9.5f, 5.5f,  -3.3f, -0.1f,  1.25f, -5.0f};
    c.quats = {1.0f,  0.0f, 0.0f,  0.0f,          // identity
               0.5f,  -0.5f, 0.5f, -0.5f,         // all four components tied
               2.0f,  1.0f, -1.0f, 0.5f,          // denormalized (spz normalizes internally)
               -0.2f, 0.4f, -0.7f, 0.5f};         // negative largest component (the sign canonicalization path)
    c.opacities = {2.0f, -1.0f, 0.25f, 1.5f}; // sigmoid 0.881/0.269/0.562/0.818 — all alive
    c.sh0 = {0.3f, -0.8f, 1.2f, -1.5f, 0.05f, 0.6f, 2.0f, -2.2f, 0.0f, 1.0f, -0.4f, 0.9f};
    c.shN.resize(4 * 15 * 3);
    for (size_t j = 0; j < c.shN.size(); j++)
        c.shN[j] = 0.7f * std::sin(0.37f * static_cast<float>(j) + 0.1f);

    const std::string path_s = sxtest::tmp_path("splatx_test_roundtrip.spz"); // iOS sandbox compatible (TestPaths.h)
    const char* path = path_s.c_str();
    save_spz(c, path);

    // to=RDF: restores the file (RUB) back to the training world (RDF) — symmetric with save's
    // from=RDF (the coordinate system section of SpzIO.h)
    spz::UnpackOptions to_rdf;
    to_rdf.to = spz::CoordinateSystem::RDF;
    const spz::GaussianCloud g = spz::loadSpz(path, to_rdf);
    SX_CHECK(g.numPoints == 4);
    SX_CHECK(g.shDegree == 3);
    SX_CHECK(!g.antialiased); // off by default

    for (size_t i = 0; i < c.means.size(); i++)
        SX_CHECK_NEAR(g.positions[i], c.means[i], 0.0, 1.3e-4);
    for (size_t i = 0; i < c.scales.size(); i++)
        SX_CHECK_NEAR(g.scales[i], c.scales[i], 0.0, 0.032);
    for (int64_t i = 0; i < 4; i++)
        check_quat_near(&c.quats[i * 4], &g.rotations[i * 4], 4e-3);
    for (size_t i = 0; i < c.opacities.size(); i++)
        SX_CHECK_NEAR(sig(g.alphas[i]), sig(c.opacities[i]), 0.0, 2e-3);
    for (size_t i = 0; i < c.sh0.size(); i++)
        SX_CHECK_NEAR(g.colors[i], c.sh0[i], 0.0, 1.4e-2);
    // The sh layout is identical ([N,K,3], coefficient outer and channel inner) — compared with no
    // index transformation, using per-band tolerances
    SX_CHECK(g.sh.size() == c.shN.size());
    for (int64_t i = 0; i < 4; i++)
        for (int k = 0; k < 15; k++)
            for (int ch = 0; ch < 3; ch++) {
                const size_t j = (static_cast<size_t>(i) * 15 + k) * 3 + ch;
                SX_CHECK_NEAR(g.sh[j], c.shN[j], 0.0, k < 3 ? 3.6e-2 : 6.7e-2);
            }

    // Confirms the authoritative coordinate system: a raw load (to=UNSPECIFIED → no transformation)
    // gives the file contents verbatim = RUB. If save really packed as RDF→RUB (180° about the x
    // axis = flipping the sign of y and z), then y and z must come back flipped.
    const spz::GaussianCloud raw = spz::loadSpz(path, {});
    for (int64_t i = 0; i < 4; i++) {
        SX_CHECK_NEAR(raw.positions[i * 3 + 0], c.means[i * 3 + 0], 0.0, 1.3e-4);
        SX_CHECK_NEAR(raw.positions[i * 3 + 1], -c.means[i * 3 + 1], 0.0, 1.3e-4);
        SX_CHECK_NEAR(raw.positions[i * 3 + 2], -c.means[i * 3 + 2], 0.0, 1.3e-4);
    }
    std::remove(path);
}

SX_TEST(spz_save_antialiased_flag_roundtrip) {
    SplatCloud c;
    c.count = 1;
    c.sh_degree = 0; // K = 0 — no shN (the spz sh array is empty too)
    c.means = {0.1f, 0.2f, 0.3f};
    c.scales = {-2.0f, -2.0f, -2.0f};
    c.quats = {1.0f, 0.0f, 0.0f, 0.0f};
    c.opacities = {1.0f};
    c.sh0 = {0.5f, -0.5f, 0.0f};

    const std::string path_s = sxtest::tmp_path("splatx_test_aa.spz"); // iOS sandbox compatible (TestPaths.h)
    const char* path = path_s.c_str();
    SpzSaveOptions aa_on;
    aa_on.antialiased = true; // TrainConfig.antialiased → SPZ header flags bit0
    save_spz(c, path, aa_on);
    const spz::GaussianCloud on = spz::loadSpz(path, {});
    SX_CHECK(on.numPoints == 1 && on.shDegree == 0);
    SX_CHECK(on.antialiased);

    save_spz(c, path); // default options — antialiased=false
    const spz::GaussianCloud off = spz::loadSpz(path, {});
    SX_CHECK(!off.antialiased);
    std::remove(path);
}

SX_TEST(spz_save_applies_export_filter) {
    constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

    SplatCloud c;
    c.count = 4;
    c.sh_degree = 1; // K = 3
    c.means = {1.0f, -2.0f, 3.0f, 4.0f, 5.0f, 6.0f, -1.0f, 0.5f, 2.5f, 7.0f, -8.0f, 9.0f};
    c.scales.assign(12, -2.0f);
    c.quats = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
    c.opacities = {2.0f, 2.0f, -7.0f, 2.0f}; // row 2: sigmoid(-7) ≈ 9.1e-4 <= 0.005 → dead
    c.sh0.assign(12, 0.25f);
    c.shN.assign(4 * 3 * 3, 0.1f);
    c.means[1 * 3 + 2] = kNaN; // row 1: nonfinite — without the filter the round→int32 cast is UB

    const std::string path_s = sxtest::tmp_path("splatx_test_filter.spz"); // iOS sandbox compatible (TestPaths.h)
    const char* path = path_s.c_str();
    ExportFilterStats st;
    SpzSaveOptions so;
    so.filter_stats = &st;
    save_spz(c, path, so); // the filter is built in (the SpzIO.h contract — it shares PlyIO's filter_cloud_for_export)
    SX_CHECK(st.input == 4);
    SX_CHECK(st.removed_nonfinite == 1);
    SX_CHECK(st.removed_low_opacity == 1);
    SX_CHECK(st.kept == 2);

    spz::UnpackOptions to_rdf;
    to_rdf.to = spz::CoordinateSystem::RDF;
    const spz::GaussianCloud g = spz::loadSpz(path, to_rdf);
    std::remove(path);
    SX_CHECK(g.numPoints == 2); // the surviving rows = {0, 3} (order preserved)
    for (int k = 0; k < 3; k++) {
        SX_CHECK_NEAR(g.positions[0 * 3 + k], c.means[0 * 3 + k], 0.0, 1.3e-4);
        SX_CHECK_NEAR(g.positions[1 * 3 + k], c.means[3 * 3 + k], 0.0, 1.3e-4);
    }
    for (float v : g.positions) SX_CHECK(std::isfinite(v));
    for (float v : g.alphas) SX_CHECK(std::isfinite(v));

    // The sh_degree range guard: spz supports only 0..3 — 4 throws immediately (this prevents
    // silently producing a corrupt file)
    SplatCloud d;
    d.count = 0;
    d.sh_degree = 4;
    bool threw = false;
    try { save_spz(d, path); } catch (const std::exception&) { threw = true; }
    SX_CHECK(threw);
}

// ===== Task 3 — load_spz (import) =====

// The full PLY→SPZ→PLY chain: write_inria_ply → read_inria_ply (lossless) → save_spz
// (quantization) → load_spz (the inverse RDF transform, the w-first reordering, and an untransformed
// sh copy) → write_inria_ply → read_inria_ply. The final result is compared against the original
// using the Task 2 quantization tolerances at the top of this file — the PLY round-trip is a raw
// float dump, so the only source of loss is the single SPZ packing.
SX_TEST(spz_load_roundtrip_ply_spz_ply_tolerances) {
    SplatCloud c;
    c.count = 4;
    c.sh_degree = 3; // K = 15
    c.means = {0.5f,  -1.25f, 2.0f,   //
               -3.5f, 0.75f,  -0.5f,  //
               10.0f, -20.0f, 30.0f,  //
               0.03f, 0.4f,   -0.9f};
    c.scales = {-4.0f, -2.5f, -1.0f, -6.75f, 0.5f,  2.0f,
                -9.5f, 5.5f,  -3.3f, -0.1f,  1.25f, -5.0f};
    c.quats = {1.0f,  0.0f, 0.0f,  0.0f,          // identity
               0.5f,  -0.5f, 0.5f, -0.5f,         // all four components tied
               2.0f,  1.0f, -1.0f, 0.5f,          // denormalized (spz normalizes internally)
               -0.2f, 0.4f, -0.7f, 0.5f};         // negative largest component (the sign canonicalization path)
    c.opacities = {2.0f, -1.0f, 0.25f, 1.5f};     // all alive, and within the non-saturating range
    c.sh0 = {0.3f, -0.8f, 1.2f, -1.5f, 0.05f, 0.6f, 2.0f, -2.2f, 0.0f, 1.0f, -0.4f, 0.9f};
    c.shN.resize(4 * 15 * 3);
    for (size_t j = 0; j < c.shN.size(); j++)
        c.shN[j] = 0.7f * std::sin(0.37f * static_cast<float>(j) + 0.1f);

    const std::string ply_a_s = sxtest::tmp_path("splatx_test_rt_a.ply"); // iOS sandbox compatible (TestPaths.h)
    const char* ply_a = ply_a_s.c_str();
    const std::string spz_p_s = sxtest::tmp_path("splatx_test_rt.spz"); // iOS sandbox compatible (TestPaths.h)
    const char* spz_p = spz_p_s.c_str();
    const std::string ply_b_s = sxtest::tmp_path("splatx_test_rt_b.ply"); // iOS sandbox compatible (TestPaths.h)
    const char* ply_b = ply_b_s.c_str();
    write_inria_ply(c, ply_a);
    const SplatCloud from_ply = read_inria_ply(ply_a); // includes the f_rest transpose round-trip (lossless)
    save_spz(from_ply, spz_p);
    bool aa = true; // the default save is false — confirms the out-param is actually written
    const SplatCloud from_spz = load_spz(spz_p, &aa);
    SX_CHECK(!aa);
    write_inria_ply(from_spz, ply_b);
    const SplatCloud r = read_inria_ply(ply_b);
    std::remove(ply_a);
    std::remove(spz_p);
    std::remove(ply_b);

    SX_CHECK(r.count == 4);
    SX_CHECK(r.sh_degree == 3);
    for (size_t i = 0; i < c.means.size(); i++)
        SX_CHECK_NEAR(r.means[i], c.means[i], 0.0, 1.3e-4);
    for (size_t i = 0; i < c.scales.size(); i++)
        SX_CHECK_NEAR(r.scales[i], c.scales[i], 0.0, 0.032);
    for (int64_t i = 0; i < 4; i++) {
        // r.quats is w-first — check_quat_near takes its loaded-value argument as xyzw, so reorder
        // before passing it
        const float xyzw[4] = {r.quats[i * 4 + 1], r.quats[i * 4 + 2], r.quats[i * 4 + 3],
                               r.quats[i * 4 + 0]};
        check_quat_near(&c.quats[i * 4], xyzw, 4e-3);
    }
    for (size_t i = 0; i < c.opacities.size(); i++)
        SX_CHECK_NEAR(sig(r.opacities[i]), sig(c.opacities[i]), 0.0, 2e-3);
    for (size_t i = 0; i < c.sh0.size(); i++)
        SX_CHECK_NEAR(r.sh0[i], c.sh0[i], 0.0, 1.4e-2);
    SX_CHECK(r.shN.size() == c.shN.size());
    for (int64_t i = 0; i < 4; i++)
        for (int k = 0; k < 15; k++)
            for (int ch = 0; ch < 3; ch++) {
                const size_t j = (static_cast<size_t>(i) * 15 + k) * 3 + ch;
                SX_CHECK_NEAR(r.shN[j], c.shN[j], 0.0, k < 3 ? 3.6e-2 : 6.7e-2);
            }
    // The contract that no NaN or Inf is present (SpzIO.h) — every field is finite
    for (float v : r.means) SX_CHECK(std::isfinite(v));
    for (float v : r.opacities) SX_CHECK(std::isfinite(v));
}

// Correctness of the SPZ conversion of a zero-padded PLY (a follow-up to design §8.1 — the
// sx_convert_run PLY→SPZ path): for a deg1 output, the default write zero-pads f_rest to 45
// entries → read infers deg3 (with the higher coefficients 0) → save_spz packs it with shDegree 3.
// The original coefficients (band 1) are preserved within the 5-bit quantization tolerance, and the
// padding coefficients (bands 2 and 3) round-trip to exactly 0.0 because quantizeSH(0) = byte 128
// (load-spz.cc:77–83; bucket rounding also keeps 128 — the flipSh sign multiplication of the RUB
// transform is harmless too, since it is ±0).
SX_TEST(spz_convert_accepts_zero_padded_ply) {
    SplatCloud c;
    c.count = 2;
    c.sh_degree = 1; // K = 3
    c.means = {0.5f, -1.25f, 2.0f, -3.5f, 0.75f, -0.5f};
    c.scales = {-4.0f, -2.5f, -1.0f, -6.75f, 0.5f, 2.0f};
    c.quats = {1.0f, 0.0f, 0.0f, 0.0f, 0.5f, -0.5f, 0.5f, -0.5f};
    c.opacities = {2.0f, -1.0f}; // all alive (the built-in export filter does not trigger)
    c.sh0 = {0.3f, -0.8f, 1.2f, -1.5f, 0.05f, 0.6f};
    c.shN.resize(2 * 3 * 3);
    for (size_t j = 0; j < c.shN.size(); j++)
        c.shN[j] = 0.6f * std::sin(0.41f * static_cast<float>(j) + 0.2f);

    const std::string ply_s = sxtest::tmp_path("splatx_test_padded.ply"); // iOS sandbox compatible (TestPaths.h)
    const std::string spz_s = sxtest::tmp_path("splatx_test_padded.spz"); // iOS sandbox compatible (TestPaths.h)
    write_inria_ply(c, ply_s); // the default is zero-padding (45 f_rest entries)
    const SplatCloud padded = read_inria_ply(ply_s);
    SX_CHECK(padded.sh_degree == 3); // 45 entries → deg3 is inferred (the PlyIO.h contract)
    save_spz(padded, spz_s);
    const SplatCloud r = load_spz(spz_s);
    std::remove(ply_s.c_str());
    std::remove(spz_s.c_str());

    SX_CHECK(r.count == 2 && r.sh_degree == 3);
    SX_CHECK(r.shN.size() == static_cast<size_t>(2) * 15 * 3);
    for (int64_t i = 0; i < 2; i++)
        for (int k = 0; k < 15; k++)
            for (int ch = 0; ch < 3; ch++) {
                const float got = r.shN[(i * 15 + k) * 3 + ch];
                if (k < 3) // band 1 original coefficients — the 5-bit quantization atol (rationale at the top of the file)
                    SX_CHECK_NEAR(got, c.shN[(i * 3 + k) * 3 + ch], 0.0, 3.6e-2);
                else       // padding coefficients — they round-trip to exactly 0
                    SX_CHECK(got == 0.0f);
            }
}

// Degree inference: the SPZ header shDegree field → SplatCloud.sh_degree (plus the shN size
// K = (d+1)²-1), and whether the antialiased out-param recovers header flags bit0 (alternating
// on/off per degree).
SX_TEST(spz_load_infers_sh_degree) {
    const std::string path_s = sxtest::tmp_path("splatx_test_degree.spz"); // iOS sandbox compatible (TestPaths.h)
    const char* path = path_s.c_str();
    for (int deg = 0; deg <= 3; deg++) {
        SplatCloud c;
        c.count = 2;
        c.sh_degree = deg;
        const int K = c.sh_rest_coeffs();
        c.means = {0.1f, 0.2f, 0.3f, -0.4f, 0.5f, -0.6f};
        c.scales.assign(6, -2.0f);
        c.quats = {1, 0, 0, 0, 1, 0, 0, 0};
        c.opacities = {1.0f, -1.0f};
        c.sh0 = {0.5f, -0.5f, 0.0f, 0.25f, 0.1f, -0.3f};
        c.shN.assign(static_cast<size_t>(2) * K * 3, 0.2f);

        SpzSaveOptions so;
        so.antialiased = (deg % 2 == 1);
        save_spz(c, path, so);
        bool aa = false;
        const SplatCloud r = load_spz(path, &aa);
        SX_CHECK(r.sh_degree == deg);
        SX_CHECK(r.count == 2);
        SX_CHECK(r.shN.size() == static_cast<size_t>(2) * K * 3);
        SX_CHECK(aa == (deg % 2 == 1));
    }
    // The call form that does not recover antialiased (the nullptr default) — usage symmetric with
    // read_inria_ply
    const SplatCloud r = load_spz(path);
    SX_CHECK(r.sh_degree == 3);
    std::remove(path);
}

// Failure paths plus the opacity saturation clamp (the Task 2 carry-over policy — SpzIO.h
// kSpzOpacityLogitLimit): corrupt, non-SPZ, and missing files throw, and the ±inf produced by the
// saturated bytes 255 and 0 is normalized to the finite ±ln(509).
SX_TEST(spz_load_rejects_corrupt_and_clamps_saturated_opacity) {
    // (1) Non-SPZ bytes (not gzip) → throw
    const std::string bad_s = sxtest::tmp_path("splatx_test_bad.spz"); // iOS sandbox compatible (TestPaths.h)
    const char* bad = bad_s.c_str();
    {
        FILE* f = std::fopen(bad, "wb");
        const char junk[] = "this is not an spz file at all — just junk bytes";
        std::fwrite(junk, 1, sizeof(junk), f);
        std::fclose(f);
    }
    bool threw = false;
    try { load_spz(bad); } catch (const std::exception&) { threw = true; }
    SX_CHECK(threw);
    std::remove(bad);

    // (2) Missing file → throw
    threw = false;
    try { load_spz(sxtest::tmp_path("splatx_test_nonexistent.spz")); } catch (const std::exception&) {
        threw = true;
    }
    SX_CHECK(threw);

    // (3) The opacity saturation round-trip: logit 8 → sigmoid 0.99966 → byte 255 → unpack +inf →
    //     clamped to +ln(509). logit -12 → sigmoid 6.1e-6 → byte 0 → unpack -inf → clamped to
    //     -ln(509) (min_opacity 0 bypasses the export filter's removal of low-opacity rows —
    //     sigmoid > 0 always holds for a finite logit). The middle value 0.5 confirms saturation
    //     does not trigger.
    SplatCloud c;
    c.count = 3;
    c.sh_degree = 0;
    c.means = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, -0.1f, -0.2f, -0.3f};
    c.scales.assign(9, -2.0f);
    c.quats = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
    c.opacities = {8.0f, -12.0f, 0.5f};
    c.sh0.assign(9, 0.25f);

    const std::string path_s = sxtest::tmp_path("splatx_test_saturate.spz"); // iOS sandbox compatible (TestPaths.h)
    const char* path = path_s.c_str();
    SpzSaveOptions so;
    so.min_opacity = 0.0f;
    save_spz(c, path, so);
    const SplatCloud r = load_spz(path);
    std::remove(path);
    SX_CHECK(r.count == 3);
    for (float v : r.opacities) SX_CHECK(std::isfinite(v)); // the contract that no NaN or Inf is present
    SX_CHECK_NEAR(r.opacities[0], kSpzOpacityLogitLimit, 0.0, 1e-6);  // +inf → +ln(509)
    SX_CHECK_NEAR(r.opacities[1], -kSpzOpacityLogitLimit, 0.0, 1e-6); // -inf → -ln(509)
    SX_CHECK_NEAR(sig(r.opacities[2]), sig(0.5), 0.0, 2e-3); // the clamp does not trigger for a non-saturated value
}
