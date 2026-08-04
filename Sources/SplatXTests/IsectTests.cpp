// isect_tiles verification (spec 03) —
// The AABB path (kIsectAccuTile=false):
//   fixtures/isect_std  (I=2, N=128, tile 16, 10×8=80 tiles, n_isects=5097, a chained golden),
//   fixtures/isect_ties (I=1 → the image_n_bits=0 edge, tied keys and radii=0 culling,
//   n_isects=53).
// The AccuTile path (kIsectAccuTile=true, Task 5):
//   fixtures/isect_accutile — the golden producer is the C++ CPU reference (isect_tiles_cpu, in
//   dump_accutile_fixture below; torch _isect_tiles has no AccuTile — 07 Correction 2). The same
//   code both generates and checks the golden, which is a self-confirmation risk → the defenses are
//   (a) the assertion AccuTile ⊆ AABB (AABB is the baseline already compared against the torch
//   golden in isect_std — an element-wise ≤ on the counts + the (flatten,key) pairs being a
//   subset), (b) an independent geometric oracle unrelated to the golden (the minimum of the convex
//   quadratic form over the tile box, min q ≤ t) + symmetry property assertions
//   (isect_tiles_accutile_properties).
// The exact-integer contract (D9-1 / 07 §4): tiles_per_gauss/isect_ids/flatten_ids/n_isects are all
// SX_EXACT — no tolerance (meta.json tolerances: {}). SXT does not support int64, so the golden
// isect_ids are stored split into lo/hi int32 (D3) — lo is reinterpreted as uint32 and reassembled
// as (hi<<32)|lo (the contract from the task-2 report).
// Bit-exact determinism: memcmp of two Metal runs. The emit order convention (AABB: element idx →
// tile y → x; AccuTile: element idx → sweep axis u → orthogonal axis v) is pinned by exact equality
// with the raw, pre-sort golden.
#include "TestHarness.h"
#include "TestUtils.h" // shared_metal_backend / SX_EXACT_I32 / SX_EXACT_U64
#include "KernelConstants.h" // kAlphaThreshold/kGaussianExtend (level set t — one source w/ kernel)
#include "splatx/CpuKernels.h"
#include "splatx/Json.h"
#include "splatx/KernelBackend.h"
#include "splatx/SXTensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace splatx;

namespace {

std::string fx(const std::string& dir, const char* rel) {
    return sxtest::fixture_dir() + "/" + dir + "/" + rel;
}

struct IsectFixture {
    std::map<std::string, double> meta;
    SXTensor means2d, radii, depths;              // inputs
    SXTensor conics, opacities;                    // AccuTile inputs (isect_accutile only)
    SXTensor tiles_per_gauss, lo, hi, flatten_ids; // pass1/pass2 (raw emit) goldens
    std::vector<uint64_t> golden_ids;              // reassembled (uint32)hi << 32 | (uint32)lo
};

IsectFixture load_fixture(const std::string& dir) {
    IsectFixture f;
    f.meta = json_parse_flat(fx(dir, "meta.json"));
    f.means2d = SXTensor::load(fx(dir, "means2d.sxt"));
    f.radii = SXTensor::load(fx(dir, "radii.sxt"));
    f.depths = SXTensor::load(fx(dir, "depths.sxt"));
    f.tiles_per_gauss = SXTensor::load(fx(dir, "tiles_per_gauss.sxt"));
    f.lo = SXTensor::load(fx(dir, "isect_ids_lo.sxt"));
    f.hi = SXTensor::load(fx(dir, "isect_ids_hi.sxt"));
    f.flatten_ids = SXTensor::load(fx(dir, "flatten_ids.sxt"));
    f.golden_ids.resize(f.lo.count());
    for (size_t i = 0; i < f.golden_ids.size(); i++)
        f.golden_ids[i] =
            (static_cast<uint64_t>(static_cast<uint32_t>(f.hi.i32()[i])) << 32) |
            static_cast<uint64_t>(static_cast<uint32_t>(f.lo.i32()[i]));
    return f;
}

IsectFixture load_accutile_fixture() {
    IsectFixture f = load_fixture("isect_accutile");
    f.conics = SXTensor::load(fx("isect_accutile", "conics.sxt"));
    f.opacities = SXTensor::load(fx("isect_accutile", "opacities.sxt"));
    return f;
}

IsectTilesInputs make_inputs(const IsectFixture& f, bool accutile = false) {
    IsectTilesInputs in;
    in.means2d = f.means2d.f32();
    in.radii = f.radii.i32();
    in.depths = f.depths.f32();
    if (accutile) { // the AccuTile gate = passing conics+opacities together (IntersectTile.cu:288)
        in.conics = f.conics.f32();
        in.opacities = f.opacities.f32();
    }
    in.I = static_cast<int64_t>(f.meta.at("I"));
    in.N = static_cast<int64_t>(f.meta.at("N"));
    in.tile_size = static_cast<uint32_t>(f.meta.at("tile_size"));
    in.tile_width = static_cast<uint32_t>(f.meta.at("tile_width"));
    in.tile_height = static_cast<uint32_t>(f.meta.at("tile_height"));
    return in;
}

// Exact equality with the golden (integers/bits) — counts, keys, flatten, n_isects, and the bit
// widths alike.
void compare_golden(const std::string& tag, const IsectTilesOutputs& out,
                    const IsectFixture& f) {
    SX_CHECK(out.n_isects == static_cast<int64_t>(f.meta.at("n_isects")));
    SX_CHECK(out.tile_n_bits == static_cast<uint32_t>(f.meta.at("tile_n_bits")));
    SX_CHECK(out.image_n_bits == static_cast<uint32_t>(f.meta.at("image_n_bits")));
    SX_EXACT_I32((tag + " tiles_per_gauss").c_str(), out.tiles_per_gauss,
                 f.tiles_per_gauss.i32(), f.tiles_per_gauss.count());
    SX_EXACT_U64((tag + " isect_ids").c_str(), out.isect_ids, f.golden_ids.data(),
                 f.golden_ids.size());
    SX_EXACT_I32((tag + " flatten_ids").c_str(), out.flatten_ids, f.flatten_ids.i32(),
                 f.flatten_ids.count());
}

} // namespace

// (1) The CPU reference vs the isect_std golden — includes the bits_for_count boundary values
// (the examples of spec 03 §1).
SX_TEST(isect_tiles_cpu_matches_golden) {
    SX_CHECK(bits_for_count(1) == 0);    // MathUtils.h:25 — count<=1 → 0
    SX_CHECK(bits_for_count(8) == 3);    // a case a floor(log2)+1 scheme overestimates as 4
    SX_CHECK(bits_for_count(8160) == 13);
    SX_CHECK(bits_for_count(80) == 7);   // isect_std tile_n_bits
    SX_CHECK(bits_for_count(2) == 1);    // isect_std image_n_bits

    const IsectFixture f = load_fixture("isect_std");
    const IsectTilesOutputs out = isect_tiles_cpu(make_inputs(f));
    compare_golden("cpu", out, f);
}

// (2) Metal vs the isect_std golden — drives the isect_tile_count/emit two passes (the cumsum runs
// on the host CPU).
SX_TEST(isect_tiles_metal_matches_golden) {
    const IsectFixture f = load_fixture("isect_std");
    const IsectTilesOutputs out = sxtest::shared_metal_backend().isect_tiles(make_inputs(f));
    compare_golden("metal", out, f);
}

// (3) Metal bit-exact determinism — memcmp of two runs on identical input (an integer pipeline —
// it must come out exactly the same).
SX_TEST(isect_tiles_metal_bitwise_deterministic) {
    const IsectFixture f = load_fixture("isect_std");
    const IsectTilesInputs in = make_inputs(f);
    const IsectTilesOutputs a = sxtest::shared_metal_backend().isect_tiles(in);
    const IsectTilesOutputs b = sxtest::shared_metal_backend().isect_tiles(in);
    SX_CHECK(a.n_isects == b.n_isects);
    SX_CHECK(a.tiles_per_gauss.size() == b.tiles_per_gauss.size());
    SX_CHECK(a.isect_ids.size() == b.isect_ids.size());
    SX_CHECK(std::memcmp(a.tiles_per_gauss.data(), b.tiles_per_gauss.data(),
                         a.tiles_per_gauss.size() * sizeof(int32_t)) == 0);
    SX_CHECK(std::memcmp(a.isect_ids.data(), b.isect_ids.data(),
                         a.isect_ids.size() * sizeof(uint64_t)) == 0);
    SX_CHECK(std::memcmp(a.flatten_ids.data(), b.flatten_ids.data(),
                         a.flatten_ids.size() * sizeof(int32_t)) == 0);
}

// (4) isect_ties — the emit order matches (the raw, pre-sort order inside a group of tied keys is
//     the upstream premise of the stable sort contract — the task-2 report). Covers the I=1 →
//     image_n_bits=0 encoding edge + radii=0 culling (g17).
SX_TEST(isect_tiles_ties_emit_order) {
    const IsectFixture f = load_fixture("isect_ties");
    const IsectTilesInputs in = make_inputs(f);

    // fixture contract: culled slots exist (g17 radii=0 → tiles_per_gauss 0)
    size_t culled = 0;
    for (size_t i = 0; i < f.tiles_per_gauss.count(); i++)
        if (f.tiles_per_gauss.i32()[i] == 0) culled++;
    SX_CHECK(culled > 0);

    const IsectTilesOutputs cpu = isect_tiles_cpu(in);
    compare_golden("cpu(ties)", cpu, f);

    const IsectTilesOutputs mtl = sxtest::shared_metal_backend().isect_tiles(in);
    compare_golden("metal(ties)", mtl, f);
}

// ===== AccuTile (Task 5) =====

namespace {

// Helper that runs AccuTile on a single Gaussian (I=1, N=1, depth=1) — synthetic input for the
// property tests.
IsectTilesOutputs run_accutile_single(bool metal, const float conic[3], float opacity,
                                      float mean_x, float mean_y, int32_t radius,
                                      uint32_t tile_size, uint32_t TW, uint32_t TH) {
    const float means2d[2] = {mean_x, mean_y};
    const int32_t radii[2] = {radius, radius}; // in AccuTile radii is only the culling gate
    const float depths[1] = {1.0f};
    const float opac[1] = {opacity};
    IsectTilesInputs in;
    in.means2d = means2d;
    in.radii = radii;
    in.depths = depths;
    in.conics = conic;
    in.opacities = opac;
    in.I = 1;
    in.N = 1;
    in.tile_size = tile_size;
    in.tile_width = TW;
    in.tile_height = TH;
    return metal ? sxtest::shared_metal_backend().isect_tiles(in) : isect_tiles_cpu(in);
}

// Recovers the set of (tx, ty) tiles from the emit keys — assumes I=1 (image bits 0).
std::set<std::pair<int, int>> tile_set(const IsectTilesOutputs& o, uint32_t TW) {
    std::set<std::pair<int, int>> s;
    for (const uint64_t key : o.isect_ids) {
        const uint32_t tile_id =
            static_cast<uint32_t>((key >> 32) & ((1ull << o.tile_n_bits) - 1));
        s.emplace(static_cast<int>(tile_id % TW), static_cast<int>(tile_id / TW));
    }
    return s;
}

// The independent geometric oracle (self-confirmation defense b — a closed form unrelated to the
// sweep implementation): the minimum of the quadratic form q(d) = A·dx² + 2B·dx·dy + C·dy² over the
// closed tile rectangle. q is convex, so the minimum over the box is attained either at the
// interior critical point (= 0 when the center lies inside the box) or on one of the 4 edges. On
// the edge x=xe, ∂q/∂dy = 0 → dy* = -B·xe/C, clamped to the edge's interval (the y=ye edges are
// symmetric). The arithmetic is double — headroom against the kernel's f32 is secured by the choice
// of case parameters (no ties on tile boundaries).
double min_q_over_tile(double A, double B, double C, double mx, double my, int tx, int ty,
                       double ts) {
    const double x0 = tx * ts - mx, x1 = (tx + 1) * ts - mx;
    const double y0 = ty * ts - my, y1 = (ty + 1) * ts - my;
    if (x0 <= 0.0 && 0.0 <= x1 && y0 <= 0.0 && 0.0 <= y1) return 0.0;
    const auto q = [&](double dx, double dy) { return A * dx * dx + 2.0 * B * dx * dy + C * dy * dy; };
    const auto clampd = [](double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    };
    double m = q(x0, clampd(-B * x0 / C, y0, y1));
    m = std::min(m, q(x1, clampd(-B * x1 / C, y0, y1)));
    m = std::min(m, q(clampd(-B * y0 / A, x0, x1), y0));
    m = std::min(m, q(clampd(-B * y1 / A, x0, x1), y1));
    return m;
}

// The oracle's tile set: {(tx,ty) : min q ≤ t}. t is the same f32 level set the kernel uses
// (IntersectTile.cu:303 — fminf(3.33², 2·log(opacity/α_thresh)), dispositions #11/#13).
std::set<std::pair<int, int>> oracle_tiles(const float conic[3], float opacity, float mx,
                                           float my, uint32_t ts, uint32_t TW, uint32_t TH) {
    std::set<std::pair<int, int>> s;
    const float t = std::fmin(kGaussianExtend * kGaussianExtend,
                              2.0f * std::log(opacity / kAlphaThreshold));
    if (t <= 0.f) return s; // the same gate as disposition #13
    for (uint32_t ty = 0; ty < TH; ty++)
        for (uint32_t tx = 0; tx < TW; tx++)
            if (min_q_over_tile(conic[0], conic[1], conic[2], mx, my, static_cast<int>(tx),
                                static_cast<int>(ty), ts) <= static_cast<double>(t))
                s.emplace(static_cast<int>(tx), static_cast<int>(ty));
    return s;
}

// Exact equality of the CPU and Metal outputs (shared by the AccuTile property cases).
void compare_outputs_exact(const std::string& tag, const IsectTilesOutputs& a,
                           const IsectTilesOutputs& b) {
    SX_CHECK(a.n_isects == b.n_isects);
    SX_CHECK(a.tiles_per_gauss.size() == b.tiles_per_gauss.size());
    SX_CHECK(a.isect_ids.size() == b.isect_ids.size());
    SX_EXACT_I32((tag + " tiles_per_gauss").c_str(), a.tiles_per_gauss,
                 b.tiles_per_gauss.data(), b.tiles_per_gauss.size());
    if (!a.isect_ids.empty() && a.isect_ids.size() == b.isect_ids.size()) {
        SX_EXACT_U64((tag + " isect_ids").c_str(), a.isect_ids, b.isect_ids.data(),
                     b.isect_ids.size());
        SX_EXACT_I32((tag + " flatten_ids").c_str(), a.flatten_ids, b.flatten_ids.data(),
                     b.flatten_ids.size());
    }
}

} // namespace

// (5) CPU AccuTile vs the committed golden + self-confirmation defense (a): AccuTile ⊆ AABB.
//     The golden producer is the CPU reference itself (--dump-accutile), so this comparison alone
//     does nothing but pin regressions — self-confirmation is broken by asserting the inclusion
//     against the AABB path (already compared with the torch golden in isect_std) as the baseline:
//     an element-wise ≤ on the tile counts + the (flatten_id, key) pairs being a subset.
SX_TEST(isect_tiles_cpu_accutile_matches_golden) {
    const IsectFixture f = load_accutile_fixture();
    const IsectTilesInputs in = make_inputs(f, /*accutile=*/true);
    const IsectTilesOutputs accu = isect_tiles_cpu(in);
    compare_golden("cpu(accutile)", accu, f);

    IsectTilesInputs aabb_in = in; // same input, only conics/opacities removed → the AABB fallback
    aabb_in.conics = nullptr;
    aabb_in.opacities = nullptr;
    const IsectTilesOutputs aabb = isect_tiles_cpu(aabb_in);
    SX_CHECK(aabb.n_isects == static_cast<int64_t>(f.meta.at("n_isects_aabb")));
    SX_CHECK(accu.n_isects < aabb.n_isects); // a real cut from a quality-lossless optimization

    // (a-1) Element-wise inclusion of the tile counts: the AccuTile count ≤ the AABB count
    //       (t ≤ 3.33² — the tight ellipse ⊆ the radius-budget rectangle).
    size_t count_violations = 0;
    for (size_t i = 0; i < accu.tiles_per_gauss.size(); i++)
        if (accu.tiles_per_gauss[i] > aabb.tiles_per_gauss[i]) count_violations++;
    SX_CHECK(count_violations == 0);

    // (a-2) Subset at intersection granularity: every (flatten_id, key) pair AccuTile emits must
    //       also be present in the AABB emit (keys of the same element share their iid/depth bits,
    //       so this effectively compares tiles only).
    std::set<std::pair<int32_t, uint64_t>> aabb_pairs;
    for (size_t k = 0; k < aabb.isect_ids.size(); k++)
        aabb_pairs.emplace(aabb.flatten_ids[k], aabb.isect_ids[k]);
    size_t missing = 0;
    for (size_t k = 0; k < accu.isect_ids.size(); k++)
        if (aabb_pairs.find({accu.flatten_ids[k], accu.isect_ids[k]}) == aabb_pairs.end())
            missing++;
    SX_CHECK(missing == 0);

    // The special slots (disposition #13 — an approved deviation): opacity exactly == α_thresh →
    // t=0, and opacity < α_thresh → t<0. Both pass the radii gate (they are not culled) while only
    // AccuTile must produce 0 tiles — AABB does not use opacity, so it stays >0 (this pins the
    // guard as specific to AccuTile).
    const size_t t0 = static_cast<size_t>(f.meta.at("t_zero_idx"));
    const size_t tn = static_cast<size_t>(f.meta.at("t_neg_idx"));
    SX_CHECK(f.radii.i32()[t0 * 2] > 0 && f.radii.i32()[t0 * 2 + 1] > 0);
    SX_CHECK(f.radii.i32()[tn * 2] > 0 && f.radii.i32()[tn * 2 + 1] > 0);
    SX_CHECK(accu.tiles_per_gauss[t0] == 0);
    SX_CHECK(accu.tiles_per_gauss[tn] == 0);
    SX_CHECK(aabb.tiles_per_gauss[t0] > 0);
    SX_CHECK(aabb.tiles_per_gauss[tn] > 0);
}

// (6) Metal AccuTile (kIsectAccuTile=true) vs the committed golden — exact integer equality with
// the CPU golden.
SX_TEST(isect_tiles_metal_accutile_matches_golden) {
    const IsectFixture f = load_accutile_fixture();
    const IsectTilesOutputs out =
        sxtest::shared_metal_backend().isect_tiles(make_inputs(f, /*accutile=*/true));
    compare_golden("metal(accutile)", out, f);
}

// (7) Metal AccuTile bit-exact determinism — memcmp of two runs on identical input.
SX_TEST(isect_tiles_metal_accutile_bitwise_deterministic) {
    const IsectFixture f = load_accutile_fixture();
    const IsectTilesInputs in = make_inputs(f, /*accutile=*/true);
    const IsectTilesOutputs a = sxtest::shared_metal_backend().isect_tiles(in);
    const IsectTilesOutputs b = sxtest::shared_metal_backend().isect_tiles(in);
    SX_CHECK(a.n_isects == b.n_isects);
    SX_CHECK(a.tiles_per_gauss.size() == b.tiles_per_gauss.size());
    SX_CHECK(a.isect_ids.size() == b.isect_ids.size());
    SX_CHECK(std::memcmp(a.tiles_per_gauss.data(), b.tiles_per_gauss.data(),
                         a.tiles_per_gauss.size() * sizeof(int32_t)) == 0);
    SX_CHECK(std::memcmp(a.isect_ids.data(), b.isect_ids.data(),
                         a.isect_ids.size() * sizeof(uint64_t)) == 0);
    SX_CHECK(std::memcmp(a.flatten_ids.data(), b.flatten_ids.data(),
                         a.flatten_ids.size() * sizeof(int32_t)) == 0);
}

// (8) AccuTile property assertions (self-confirmation defense b — synthetic input unrelated to the
//     golden, verified on CPU and Metal at once):
//     P1 an isotropic Gaussian (exactly at the grid center) — reflection-symmetric tile selection
//        in x and y + agreement with the oracle (isY=false),
//     P2 an axis-aligned anisotropic one (wide in x) — the isY=true path (the short y axis becomes
//        the outer sweep loop) + agreement with the oracle,
//     P2' rotation (B≠0) combined with isY=true — the quadrant that P2/P3 alone leave unverified
//         (a review finding: a hypothetical sign bug in the isY-swapped frame stays inside the
//         rect, so it passes ⊆AABB, and if CPU and Metal share it, it passes the golden comparison
//         too — the oracle is the only external judge) + an assertion that isY=true really is the
//         branch taken,
//     P3 a 45°-rotated anisotropic one — agreement with the oracle + a real cut against AABB (the
//        SNUGBOX tight-bbox effect),
//     P4 the t<=0 guard (disposition #13) — opacity == α_thresh (t=0) and opacity < α_thresh (t<0)
//        → 0 tiles, while the same conic with opacity 0.9 gives >0 (pinning that the gate comes
//        from the t computation).
SX_TEST(isect_tiles_accutile_properties) {
    const uint32_t TS = 16, TW = 8, TH = 8; // a 128×128 image, 8×8 = 64 tiles

    // --- P1: isotropic σ=13.7, mean=(64,64) exactly at the grid center, opacity 0.9 ---
    {
        const float s = 13.7f;
        const float conic[3] = {1.f / (s * s), 0.f, 1.f / (s * s)};
        const IsectTilesOutputs cpu = run_accutile_single(false, conic, 0.9f, 64.f, 64.f,
                                                          46 /*ceil(3.33σ)*/, TS, TW, TH);
        const IsectTilesOutputs mtl =
            run_accutile_single(true, conic, 0.9f, 64.f, 64.f, 46, TS, TW, TH);
        compare_outputs_exact("P1 cpu vs metal", cpu, mtl);
        const auto set = tile_set(cpu, TW);
        SX_CHECK(!set.empty() && set.size() < static_cast<size_t>(TW) * TH);
        SX_CHECK(set == oracle_tiles(conic, 0.9f, 64.f, 64.f, TS, TW, TH));
        for (const auto& [tx, ty] : set) { // symmetric Gaussian → symmetric tiles (a spec property)
            SX_CHECK(set.count({static_cast<int>(TW) - 1 - tx, ty}) == 1);
            SX_CHECK(set.count({tx, static_cast<int>(TH) - 1 - ty}) == 1);
        }
    }

    // --- P2: axis-aligned anisotropic σx=25, σy=6 → x_span > y_span → the isY=true sweep path ---
    {
        const float conic[3] = {1.f / 625.f, 0.f, 1.f / 36.f};
        const IsectTilesOutputs cpu = run_accutile_single(false, conic, 0.8f, 63.4f, 66.1f,
                                                          84 /*ceil(3.33·25)*/, TS, TW, TH);
        const IsectTilesOutputs mtl =
            run_accutile_single(true, conic, 0.8f, 63.4f, 66.1f, 84, TS, TW, TH);
        compare_outputs_exact("P2 cpu vs metal", cpu, mtl);
        SX_CHECK(tile_set(cpu, TW) == oracle_tiles(conic, 0.8f, 63.4f, 66.1f, TS, TW, TH));
        SX_CHECK(cpu.n_isects > 0);
    }

    // --- P2': rotation (B≠0) + isY=true — σx=25/σy=6 rotated by 15°. The only oracle case in
    //     which the sweep frame swap (IntersectTile.cu:127–135) and the signs of the B≠0
    //     chord/argmin/argmax are exercised at the same time. ---
    {
        // conic = Σ⁻¹, Σ = R(15°)·diag(25², 6²)·R(15°)ᵀ (built in double, then cast to f32;
        // det Σ = 25²·6² = 22500). Σ00≈585.5, Σ01≈147.25, Σ11≈75.46.
        const double th = 15.0 * M_PI / 180.0, cth = std::cos(th), sth = std::sin(th);
        const double s1 = 625.0, s2 = 36.0, det = s1 * s2;
        const double S00 = cth * cth * s1 + sth * sth * s2;
        const double S01 = cth * sth * (s1 - s2);
        const double S11 = sth * sth * s1 + cth * cth * s2;
        const float conic[3] = {static_cast<float>(S11 / det), static_cast<float>(-S01 / det),
                                static_cast<float>(S00 / det)};
        const float opac = 0.8f, mx = 63.4f, my = 66.1f;
        SX_CHECK(conic[1] != 0.f); // confirms the rotation (B ≠ 0)

        // Whether this case really takes the isY=true branch — the rect spans are recomputed here
        // (reproducing the formulas of IntersectTile.cu:306–308, 318–327, 337 in double; the spans
        // are 4 vs 8, so this is insensitive to f32/f64 differences).
        {
            const double A = conic[0], B = conic[1], C = conic[2];
            const double t =
                std::fmin(static_cast<double>(kGaussianExtend) * kGaussianExtend,
                          2.0 * std::log(static_cast<double>(opac) / kAlphaThreshold));
            const double disc = B * B - A * C;
            const double x_extent = std::sqrt(-t / disc * C); // ≈ 78.9
            const double y_extent = std::sqrt(-t / disc * A); // ≈ 28.3
            const int rminx = std::max(0, std::min((int)TW, (int)((mx - x_extent) / TS)));
            const int rminy = std::max(0, std::min((int)TH, (int)((my - y_extent) / TS)));
            const int rmaxx = std::max(0, std::min((int)TW, (int)((mx + x_extent) / TS + 1.0)));
            const int rmaxy = std::max(0, std::min((int)TH, (int)((my + y_extent) / TS + 1.0)));
            SX_CHECK((rmaxy - rminy) < (rmaxx - rminx)); // y_span < x_span → isY=true (:337)
        }

        const IsectTilesOutputs cpu = run_accutile_single(false, conic, opac, mx, my,
                                                          81 /*ceil(3.33·√Σ00)*/, TS, TW, TH);
        const IsectTilesOutputs mtl =
            run_accutile_single(true, conic, opac, mx, my, 81, TS, TW, TH);
        compare_outputs_exact("P2' cpu vs metal", cpu, mtl);
        SX_CHECK(tile_set(cpu, TW) == oracle_tiles(conic, opac, mx, my, TS, TW, TH));
        SX_CHECK(cpu.n_isects > 0);
    }

    // --- P3: 45°-rotated anisotropic (σ1=30, σ2=4) — Σ=[[458,442],[442,458]], conic=Σ⁻¹
    //     (det=14400). AABB (radii=ceil(3.33·√458)=72) covers effectively the whole grid, while
    //     AccuTile covers only the diagonal band. ---
    {
        const float conic[3] = {458.f / 14400.f, -442.f / 14400.f, 458.f / 14400.f};
        const IsectTilesOutputs cpu =
            run_accutile_single(false, conic, 0.95f, 64.3f, 63.7f, 72, TS, TW, TH);
        const IsectTilesOutputs mtl =
            run_accutile_single(true, conic, 0.95f, 64.3f, 63.7f, 72, TS, TW, TH);
        compare_outputs_exact("P3 cpu vs metal", cpu, mtl);
        SX_CHECK(tile_set(cpu, TW) == oracle_tiles(conic, 0.95f, 64.3f, 63.7f, TS, TW, TH));
        // A real cut against the AABB baseline (the same input, only conics/opacities removed)
        const float means2d[2] = {64.3f, 63.7f};
        const int32_t radii2[2] = {72, 72};
        const float depths1[1] = {1.0f};
        IsectTilesInputs aabb_in;
        aabb_in.means2d = means2d;
        aabb_in.radii = radii2;
        aabb_in.depths = depths1;
        aabb_in.I = 1;
        aabb_in.N = 1;
        aabb_in.tile_size = TS;
        aabb_in.tile_width = TW;
        aabb_in.tile_height = TH;
        const IsectTilesOutputs aabb = isect_tiles_cpu(aabb_in);
        SX_CHECK(cpu.n_isects > 0 && cpu.n_isects < aabb.n_isects);
    }

    // --- P4: the t<=0 guard (disposition #13 — an approved deviation): it passes the radii gate
    //     yet yields 0 tiles ---
    {
        const float s = 13.7f;
        const float conic[3] = {1.f / (s * s), 0.f, 1.f / (s * s)};
        const IsectTilesOutputs t_zero =
            run_accutile_single(false, conic, kAlphaThreshold, 64.f, 64.f, 46, TS, TW, TH);
        SX_CHECK(t_zero.n_isects == 0 && t_zero.tiles_per_gauss[0] == 0);
        const IsectTilesOutputs t_neg =
            run_accutile_single(false, conic, 0.001f, 64.f, 64.f, 46, TS, TW, TH);
        SX_CHECK(t_neg.n_isects == 0 && t_neg.tiles_per_gauss[0] == 0);
        const IsectTilesOutputs mtl_zero =
            run_accutile_single(true, conic, kAlphaThreshold, 64.f, 64.f, 46, TS, TW, TH);
        SX_CHECK(mtl_zero.n_isects == 0 && mtl_zero.tiles_per_gauss[0] == 0);
        const IsectTilesOutputs mtl_neg =
            run_accutile_single(true, conic, 0.001f, 64.f, 64.f, 46, TS, TW, TH);
        SX_CHECK(mtl_neg.n_isects == 0 && mtl_neg.tiles_per_gauss[0] == 0);
        const IsectTilesOutputs alive =
            run_accutile_single(false, conic, 0.9f, 64.f, 64.f, 46, TS, TW, TH);
        SX_CHECK(alive.n_isects > 0); // same conic/radii, only opacity differs → the gate is t
    }
}

// ===== Sorting + the full chain (Task 6) =====

namespace {

// The sort/offset goldens (the downstream goldens of isect_std/isect_ties — the task-2 convention:
// *_sorted rests on the equivalence np.lexsort stable == std::stable_sort, and offsets on
// _isect_offset_encode).
struct SortedGolden {
    std::vector<uint64_t> keys; // isect_ids_sorted — reassembled (uint32)hi << 32 | (uint32)lo (D3)
    SXTensor flatten_sorted;    // flatten_ids_sorted
    SXTensor offsets;           // [I, tile_h, tile_w] i32
};

SortedGolden load_sorted_golden(const std::string& dir) {
    SortedGolden g;
    const SXTensor lo = SXTensor::load(fx(dir, "isect_ids_sorted_lo.sxt"));
    const SXTensor hi = SXTensor::load(fx(dir, "isect_ids_sorted_hi.sxt"));
    g.flatten_sorted = SXTensor::load(fx(dir, "flatten_ids_sorted.sxt"));
    g.offsets = SXTensor::load(fx(dir, "offsets.sxt"));
    g.keys.resize(lo.count());
    for (size_t i = 0; i < g.keys.size(); i++)
        g.keys[i] = (static_cast<uint64_t>(static_cast<uint32_t>(hi.i32()[i])) << 32) |
                    static_cast<uint64_t>(static_cast<uint32_t>(lo.i32()[i]));
    return g;
}

// The full pipeline chain (the D7 convention: chaining the outputs of a Metal fwd run is forbidden
// — **inject the goldens**. Every stage runs golden input → golden comparison, which blocks error
// propagation between kernels):
//   inject the proj_fully_fused fwd golden outputs (means2d/radii/depths)
//   → isect_tiles (AABB; on Metal the scan runs on the GPU internally — Task 4) → exact equality
//     with the raw emit golden
//   → sort_isect → exact equality with the sorted goldens (*_sorted)
//   → isect_offset_encode → exact equality with the offsets golden.
void run_full_chain(const std::string& tag, KernelBackend& b) {
    const IsectFixture f = load_fixture("isect_std");

    // The chaining premise turned into a test: the isect_std inputs == byte-identical copies of the
    // proj_fully_fused fwd goldens (the task-2 convention — the generator's cmp is pinned on the
    // consumer side as well). What gets injected is the proj golden buffer itself.
    const std::string proj = sxtest::fixture_dir() + "/proj_fully_fused";
    const SXTensor p_means2d = SXTensor::load(proj + "/means2d.sxt"); // [C,N,2] f32
    const SXTensor p_radii = SXTensor::load(proj + "/radii.sxt");     // [C,N,2] i32
    const SXTensor p_depths = SXTensor::load(proj + "/depths.sxt");   // [C,N] f32
    SX_CHECK(p_means2d.data == f.means2d.data);
    SX_CHECK(p_radii.data == f.radii.data);
    SX_CHECK(p_depths.data == f.depths.data);

    IsectTilesInputs in = make_inputs(f);
    in.means2d = p_means2d.f32(); // injects the proj fwd golden output (equality checked above)
    in.radii = p_radii.i32();
    in.depths = p_depths.f32();

    // Stage 1: isect_tiles — the raw emit golden (tiles_per_gauss/isect_ids/flatten_ids/n_isects)
    const IsectTilesOutputs isect = b.isect_tiles(in);
    compare_golden(tag + " chain", isect, f);

    // Stage 2: sort_isect — the sorted golden (stable + ulong keys, valid bits = 32+tile+image)
    SortIsectInputs sin;
    sin.keys = isect.isect_ids.data();
    sin.values = isect.flatten_ids.data();
    sin.n = isect.n_isects;
    sin.key_n_bits = 32 + isect.tile_n_bits + isect.image_n_bits;
    const SortIsectOutputs sorted = b.sort_isect(sin);
    const SortedGolden g = load_sorted_golden("isect_std");
    SX_EXACT_U64((tag + " chain keys_sorted").c_str(), sorted.keys_sorted,
                 g.keys.data(), g.keys.size());
    SX_EXACT_I32((tag + " chain flatten_sorted").c_str(), sorted.values_sorted,
                 g.flatten_sorted.i32(), g.flatten_sorted.count());

    // Stage 3: isect_offset_encode — the offsets golden (the sort output is consumed as is)
    IsectOffsetInputs oin;
    oin.isect_ids = sorted.keys_sorted.data();
    oin.n_isects = isect.n_isects;
    oin.I = in.I;
    oin.tile_width = in.tile_width;
    oin.tile_height = in.tile_height;
    const std::vector<int32_t> offsets = b.isect_offset_encode(oin);
    SX_CHECK(offsets.size() == g.offsets.count());
    SX_EXACT_I32((tag + " chain offsets").c_str(), offsets, g.offsets.i32(),
                 g.offsets.count());
}

} // namespace

// (9) CPU sort_isect vs the isect_ties sorted golden — a head-on verification of the stable
//     contract. On top of the golden comparison there are two independent checks unrelated to the
//     golden: (i) the sequence of values per key preserves the emit order (across every tied group
//     — a non-stable sort violates it), (ii) the keys are in unsigned ascending order. Edges: n=0
//     gives empty input and output; high bits outside key_n_bits throw on validation (the check is
//     promoted on CPU — the Task 7 GPU radix does not validate).
SX_TEST(sort_isect_cpu_stable_matches_golden) {
    const IsectFixture f = load_fixture("isect_ties");
    const SortedGolden g = load_sorted_golden("isect_ties");
    const std::unique_ptr<KernelBackend> cpu = make_cpu_backend();

    SortIsectInputs in;
    in.keys = f.golden_ids.data(); // the raw emit-order golden is the sort input
    in.values = f.flatten_ids.i32();
    in.n = static_cast<int64_t>(f.golden_ids.size());
    in.key_n_bits = 32 + static_cast<uint32_t>(f.meta.at("tile_n_bits")) +
                    static_cast<uint32_t>(f.meta.at("image_n_bits")); // 32+4+0 = 36
    const SortIsectOutputs out = cpu->sort_isect(in);
    SX_EXACT_U64("cpu sort keys(ties)", out.keys_sorted, g.keys.data(), g.keys.size());
    SX_EXACT_I32("cpu sort flatten(ties)", out.values_sorted, g.flatten_sorted.i32(),
                 g.flatten_sorted.count());

    // (i) A golden-independent check of stability: the list of values for key k in the sorted
    //     output == the list of values for key k in the input (emit order) — for every key and
    //     every tied group (tied groups [12,12,12,14]).
    std::map<uint64_t, std::vector<int32_t>> emit_order;
    for (size_t i = 0; i < f.golden_ids.size(); i++)
        emit_order[f.golden_ids[i]].push_back(f.flatten_ids.i32()[i]);
    size_t max_group = 0;
    for (const auto& [key, vals] : emit_order) max_group = std::max(max_group, vals.size());
    SX_CHECK(max_group == 14); // fixture contract: largest tied group (12 clones + g12/g13, task-2)
    std::map<uint64_t, size_t> cursor;
    size_t stable_violations = 0;
    for (size_t i = 0; i < out.keys_sorted.size(); i++) {
        const std::vector<int32_t>& seq = emit_order.at(out.keys_sorted[i]);
        if (out.values_sorted[i] != seq[cursor[out.keys_sorted[i]]++]) stable_violations++;
    }
    SX_CHECK(stable_violations == 0);

    // (ii) Postcondition: the keys are in unsigned ascending order
    size_t order_violations = 0;
    for (size_t i = 1; i < out.keys_sorted.size(); i++)
        if (out.keys_sorted[i - 1] > out.keys_sorted[i]) order_violations++;
    SX_CHECK(order_violations == 0);

    // n=0 — empty input and output (the dispatch/sort skip contract)
    const SortIsectOutputs empty = cpu->sort_isect(SortIsectInputs{});
    SX_CHECK(empty.keys_sorted.empty() && empty.values_sorted.empty());

    // With key_n_bits=32 (depth bits only) the tile bits of the ties keys (>= bit 32) fall outside
    // the valid width → throw
    bool threw = false;
    SortIsectInputs bad = in;
    bad.key_n_bits = 32;
    try {
        cpu->sort_isect(bad);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    SX_CHECK(threw);
}

// (10) The full CPU chain — inject the proj fwd goldens → isect → sort → offset_encode, with every
//      stage exactly equal to the isect_std golden in integers.
SX_TEST(isect_chain_cpu_matches_golden) {
    const std::unique_ptr<KernelBackend> cpu = make_cpu_backend();
    run_full_chain("cpu", *cpu);
}

// (11) The full Metal chain — isect (GPU count/emit + GPU scan) → sort (GPU radix — Task 7, the 5
//      kernels of RadixSort.metal) → offset_encode (GPU). The D7 golden-injection convention is
//      the same. Exact equality with the sorted golden is what covers chain regressions of the GPU
//      radix (verification (a) of the brief).
SX_TEST(isect_chain_metal_matches_golden) {
    run_full_chain("metal", sxtest::shared_metal_backend());
}

// ===== The golden producer (Task 5): --dump-accutile <out_dir> =====
// Generates the AccuTile golden fixture — the golden producer is the CPU reference
// (isect_tiles_cpu). torch _isect_tiles models only the AABB fallback, so a cross comparison is
// impossible (07 Correction 2 — recorded in the meta note).
// Input chaining: byte-identical copies of means2d/radii/depths/conics from the proj_fully_fused
// fwd goldens (the same convention as isect_std) + opacities synthesized with an LCG (fixed seed).
// Two special slots are injected: opacity exactly == α_thresh (t=0) and opacity < α_thresh (t<0) —
// covering the disposition #13 guard.
// The self-confirmation defenses are handled by tests (5) and (8) (the AABB inclusion + the
// independent geometric oracle).
namespace sxtest {

int dump_accutile_fixture(const std::string& out_dir) {
    const std::string proj = sxtest::fixture_dir() + "/proj_fully_fused";
    const SXTensor means2d = SXTensor::load(proj + "/means2d.sxt"); // [C,N,2] f32
    const SXTensor radii = SXTensor::load(proj + "/radii.sxt");     // [C,N,2] i32
    const SXTensor depths = SXTensor::load(proj + "/depths.sxt");   // [C,N] f32
    const SXTensor conics = SXTensor::load(proj + "/conics.sxt");   // [C,N,3] f32
    const auto pmeta = json_parse_flat(proj + "/meta.json");
    const int64_t I = static_cast<int64_t>(pmeta.at("C")); // number of images = number of cameras
    const int64_t N = static_cast<int64_t>(pmeta.at("N"));
    const uint32_t width = static_cast<uint32_t>(pmeta.at("width"));
    const uint32_t height = static_cast<uint32_t>(pmeta.at("height"));
    const uint32_t tile_size = 16; // the same configuration as isect_std (10×8 = 80 tiles)
    const uint32_t tile_width = (width + tile_size - 1) / tile_size;
    const uint32_t tile_height = (height + tile_size - 1) / tile_size;
    const size_t n_elements = static_cast<size_t>(I * N);

    // Synthesizes opacities — an LCG (the Numerical Recipes constants, with a fixed seed). The
    // single source of truth for reproducibility is the committed opacities.sxt file itself (this
    // code records how it was generated).
    std::vector<float> opacities(n_elements);
    uint32_t s = 0x5EEDACC0u;
    for (auto& o : opacities) {
        s = s * 1664525u + 1013904223u;
        o = 0.02f + 0.97f * (static_cast<float>(s >> 8) * (1.0f / 16777216.0f)); // [0.02, 0.99)
    }
    // The special slots: the first two among the non-culled (radii>0) slots — they have to pass the
    // radii gate for the t<=0 guard to actually run.
    int64_t t_zero_idx = -1, t_neg_idx = -1;
    for (size_t i = 0; i < n_elements && t_neg_idx < 0; i++) {
        if (radii.i32()[i * 2] <= 0 || radii.i32()[i * 2 + 1] <= 0) continue;
        if (t_zero_idx < 0) {
            t_zero_idx = static_cast<int64_t>(i);
            opacities[i] = kAlphaThreshold; // exact → opacity/α_thresh == 1 → log = 0 → t = 0
        } else {
            t_neg_idx = static_cast<int64_t>(i);
            opacities[i] = 0.001f; // < α_thresh(≈0.00392) → t < 0
        }
    }
    if (t_zero_idx < 0 || t_neg_idx < 0) {
        std::fprintf(stderr, "dump-accutile: no non-culled slots for special opacities\n");
        return 1;
    }

    IsectTilesInputs in;
    in.means2d = means2d.f32();
    in.radii = radii.i32();
    in.depths = depths.f32();
    in.conics = conics.f32();
    in.opacities = opacities.data();
    in.I = I;
    in.N = N;
    in.tile_size = tile_size;
    in.tile_width = tile_width;
    in.tile_height = tile_height;
    const IsectTilesOutputs accu = isect_tiles_cpu(in);

    IsectTilesInputs aabb_in = in; // the AABB baseline on the same input — measures the cut and
                                   // backs the inclusion check
    aabb_in.conics = nullptr;
    aabb_in.opacities = nullptr;
    const IsectTilesOutputs aabb = isect_tiles_cpu(aabb_in);

    std::filesystem::create_directories(out_dir);
    // Inputs (byte-identical copies of proj_fully_fused + the synthesized opacities)
    means2d.save(out_dir + "/means2d.sxt");
    radii.save(out_dir + "/radii.sxt");
    depths.save(out_dir + "/depths.sxt");
    conics.save(out_dir + "/conics.sxt");
    SXTensor op_t;
    op_t.dtype = 0; // f32
    op_t.shape = {I, N};
    op_t.data.resize(n_elements * sizeof(float));
    std::memcpy(op_t.data.data(), opacities.data(), op_t.data.size());
    op_t.save(out_dir + "/opacities.sxt");
    // The goldens (raw, in emit order — the sort/offset goldens are not in Task 6/7's scope here:
    // this fixture pins only the AccuTile enumeration contract. The lo/hi split is D3 — SXT has no
    // int64 support)
    SXTensor tpg_t;
    tpg_t.dtype = 1; // i32
    tpg_t.shape = {I, N};
    tpg_t.data.resize(n_elements * sizeof(int32_t));
    std::memcpy(tpg_t.data.data(), accu.tiles_per_gauss.data(), tpg_t.data.size());
    tpg_t.save(out_dir + "/tiles_per_gauss.sxt");
    const size_t M = static_cast<size_t>(accu.n_isects);
    std::vector<int32_t> lo(M), hi(M);
    for (size_t k = 0; k < M; k++) {
        lo[k] = static_cast<int32_t>(static_cast<uint32_t>(accu.isect_ids[k] & 0xFFFFFFFFull));
        hi[k] = static_cast<int32_t>(static_cast<uint32_t>(accu.isect_ids[k] >> 32));
    }
    SXTensor key_t;
    key_t.dtype = 1;
    key_t.shape = {static_cast<int64_t>(M)};
    key_t.data.resize(M * sizeof(int32_t));
    std::memcpy(key_t.data.data(), lo.data(), key_t.data.size());
    key_t.save(out_dir + "/isect_ids_lo.sxt");
    std::memcpy(key_t.data.data(), hi.data(), key_t.data.size());
    key_t.save(out_dir + "/isect_ids_hi.sxt");
    std::memcpy(key_t.data.data(), accu.flatten_ids.data(), key_t.data.size());
    key_t.save(out_dir + "/flatten_ids.sxt");

    const double reduction =
        100.0 * (1.0 - static_cast<double>(accu.n_isects) / static_cast<double>(aabb.n_isects));
    std::FILE* mf = std::fopen((out_dir + "/meta.json").c_str(), "w");
    if (!mf) {
        std::fprintf(stderr, "dump-accutile: cannot write meta.json\n");
        return 1;
    }
    std::fprintf(mf,
        "{\n"
        "  \"I\": %lld,\n"
        "  \"N\": %lld,\n"
        "  \"chained_from\": \"proj_fully_fused (a byte-identical copy of the means2d/radii/depths/conics fwd goldens, seed 7) — the same chaining convention as isect_std\",\n"
        "  \"compare\": \"exact\",\n"
        "  \"golden_source\": \"C++ CPU reference isect_tiles_cpu (AccuTile) — splatx-core-tests --dump-accutile. Reimplementation of gsplat@77ab983:IntersectTile.cu:83-207,288-372\",\n"
        "  \"height\": %u,\n"
        "  \"image_n_bits\": %u,\n"
        "  \"key_encoding\": \"isect_id(u64) = image_id << (32+tile_n_bits) | tile_id << 32 | f32bits(depth); tile_id = tile_y*tile_width + tile_x; SXT does not support int64 → stored split into lo/hi int32 (D3).\",\n"
        "  \"n_isects\": %lld,\n"
        "  \"n_isects_aabb\": %lld,\n"
        "  \"n_tiles\": %u,\n"
        "  \"note\": \"The golden producer = the CPU reference — no torch cross-check is possible (torch _isect_tiles has no AccuTile, 07 Correction 2). Defenses against the self-confirmation risk of generating and verifying the golden with the same code: (a) the AccuTile ⊆ AABB assertion (per-element tile count ≤ plus (flatten,key) pair subset — AABB is the baseline already checked against the torch golden in isect_std), (b) an independent geometric oracle (the box minimum of a convex quadratic form) and symmetry property tests (IsectTests isect_tiles_accutile_properties). Measured AccuTile reduction rate: n_isects %lld vs AABB %lld (%.1f%% reduction). opacities is LCG-synthesized (seed 0x5EEDACC0) over [0.02,0.99) plus 2 special slots (t_zero_idx: opacity==the exact α_thresh value → t=0, t_neg_idx: 0.001 < α_thresh → t<0 — covers the approved deviation of disposition #13). Emit order: element idx → sweep axis u → orthogonal axis v (the isY swap frame). All tensors are compared exact (int/bit).\",\n"
        "  \"opacity_seed\": %u,\n"
        "  \"t_neg_idx\": %lld,\n"
        "  \"t_zero_idx\": %lld,\n"
        "  \"tensors\": {\n"
        "    \"inputs\": [\"means2d\", \"radii\", \"depths\", \"conics\", \"opacities\"],\n"
        "    \"outputs\": [\"tiles_per_gauss\", \"isect_ids_lo\", \"isect_ids_hi\", \"flatten_ids\"]\n"
        "  },\n"
        "  \"tile_height\": %u,\n"
        "  \"tile_n_bits\": %u,\n"
        "  \"tile_size\": %u,\n"
        "  \"tile_width\": %u,\n"
        "  \"tolerances\": {},\n"
        "  \"width\": %u\n"
        "}\n",
        static_cast<long long>(I), static_cast<long long>(N), height, accu.image_n_bits,
        static_cast<long long>(accu.n_isects), static_cast<long long>(aabb.n_isects),
        tile_width * tile_height, static_cast<long long>(accu.n_isects),
        static_cast<long long>(aabb.n_isects), reduction, 0x5EEDACC0u,
        static_cast<long long>(t_neg_idx), static_cast<long long>(t_zero_idx), tile_height,
        accu.tile_n_bits, tile_size, tile_width, width);
    std::fclose(mf);

    std::printf("wrote %s: accutile n_isects=%lld, aabb n_isects=%lld (%.1f%% reduction), "
                "t_zero_idx=%lld t_neg_idx=%lld\n",
                out_dir.c_str(), static_cast<long long>(accu.n_isects),
                static_cast<long long>(aabb.n_isects), reduction,
                static_cast<long long>(t_zero_idx), static_cast<long long>(t_neg_idx));
    return 0;
}

} // namespace sxtest
