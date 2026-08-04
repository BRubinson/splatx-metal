// Niantic SPZ(v3) writer/reader — the SplatCloud ↔ spz::GaussianCloud mapping + saveSpz/loadSpz.
// The authoritative record for the encoding conventions and the coordinate-system reasoning is the
// SpzIO.h header comment (from an audit of the spz v2.1.0 source).
#include "splatx/SpzIO.h"

#include "load-spz.h" // ThirdParty/spz v2.1.0 (SplatXCore depends on the spz target)

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace splatx {

void save_spz(const SplatCloud& cloud, const std::string& path, const SpzSaveOptions& opts) {
    // spz supports SH degree 0..3 only (load-spz.cc dimForDegree:58–72 — out of that range spz
    // silently serializes an empty PackedGaussians and leaves a corrupt file → throw immediately here).
    if (cloud.sh_degree < 0 || cloud.sh_degree > 3)
        throw std::invalid_argument("spz: unsupported sh_degree " +
                                    std::to_string(cloud.sh_degree) + " (0..3 only)");

    // Export filter — shares PlyIO's filter_cloud_for_export (the single source of truth; do not
    // reimplement it). The basis is in SpzIO.h: SPZ packing quantizes (round → int32 cast), so a
    // NaN/Inf row is UB.
    ExportFilterStats st;
    const SplatCloud f = filter_cloud_for_export(cloud, opts.min_opacity, &st);
    if (opts.filter_stats) *opts.filter_stats = st;

    if (f.count > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument("spz: count exceeds int32 numPoints limit");
    // Note: the spz reader rejects files above 10M points (load-spz.cc:549 maxPointsToRead) — within
    // the v0.1 cap of 2M it never gates, so no write-side limit is imposed.

    const int64_t N = f.count;

    spz::GaussianCloud g;
    g.numPoints = static_cast<int32_t>(N);
    g.shDegree = f.sh_degree; // K = sh_rest_coeffs() == spz dimForDegree (0/3/8/15 match)
    g.antialiased = opts.antialiased; // → SPZ header flags bit0 (tells the viewer to apply mip-splat compensation)
    g.positions = f.means;   // raw xyz — the RDF→RUB flip is performed by PackOptions.from below
    g.scales = f.scales;     // log-scale as-is (the same convention on both sides — splat-types.h:85)
    g.alphas = f.opacities;  // logit as-is (splat-types.h:87 "before sigmoid activation")
    g.colors = f.sh0;        // the f_dc SH0 coefficients as-is (splat-types.h:88)

    // quats: SplatCloud w-first [w,x,y,z] → spz xyzw (only the position of w is rearranged — the
    // SpzIO.h table). Normalization and sign canonicalization are performed internally by spz's
    // packQuaternionSmallestThree (load-spz.cc:218).
    g.rotations.resize(static_cast<size_t>(N) * 4);
    for (int64_t i = 0; i < N; i++) {
        g.rotations[i * 4 + 0] = f.quats[i * 4 + 1]; // x
        g.rotations[i * 4 + 1] = f.quats[i * 4 + 2]; // y
        g.rotations[i * 4 + 2] = f.quats[i * 4 + 3]; // z
        g.rotations[i * 4 + 3] = f.quats[i * 4 + 0]; // w
    }

    // shN [N,K,3], coefficient-outer and channel-inner == the spz sh layout (splat-types.h:107–114)
    // — no conversion. (The PLY f_rest channel-planar transpose belongs to the PlyIO writer alone —
    // PlyIO.cpp:39–41)
    g.sh = f.shN;

    // Coordinate system: the training world (= the PLY raw dump coordinate system) = what spz names
    // RDF → spz packs RDF→RUB. The full reasoning is in the SpzIO.h comment.
    spz::PackOptions po;
    po.from = spz::CoordinateSystem::RDF;
    if (!spz::saveSpz(g, po, path))
        throw std::runtime_error("spz: cannot write " + path);
}

SplatCloud load_spz(const std::string& path, bool* antialiased) {
    // to=RDF: file (RUB) → training world (RDF) — symmetric with save_spz's PackOptions.from=RDF
    // (the coordinate-system section of SpzIO.h; unpackGaussians:529 performs the RUB→to inverse).
    spz::UnpackOptions uo;
    uo.to = spz::CoordinateSystem::RDF;
    spz::GaussianCloud g = spz::loadSpz(path, uo);

    // The vendored spz returns an empty cloud on every failure (indistinguishable from a legitimately
    // empty file — the basis is in SpzIO.h)
    if (g.numPoints <= 0)
        throw std::runtime_error("spz: cannot load " + path +
                                 " (missing, corrupt, unsupported version, or empty)");
    if (g.shDegree < 0 || g.shDegree > 3) // guard — a range deserialize:566 already rejects
        throw std::runtime_error("spz: unsupported sh_degree " + std::to_string(g.shDegree) +
                                 " in " + path);

    const size_t N = static_cast<size_t>(g.numPoints);
    const size_t K = static_cast<size_t>((g.shDegree + 1) * (g.shDegree + 1) - 1);
    // Verifies the SoA size contract (a defensive recheck of the sizes unpack already guarantees —
    // prevents a silent OOB)
    if (g.positions.size() != N * 3 || g.scales.size() != N * 3 || g.rotations.size() != N * 4 ||
        g.alphas.size() != N || g.colors.size() != N * 3 || g.sh.size() != N * K * 3)
        throw std::runtime_error("spz: field size mismatch in " + path);

    if (antialiased) *antialiased = g.antialiased; // recovers SPZ header flags bit0

    SplatCloud c;
    c.count = g.numPoints;
    c.sh_degree = g.shDegree; // degree inference = the SPZ header shDegree field (the K derivation match is verified above)
    c.means = std::move(g.positions);  // raw xyz — the RUB→RDF flip is performed by UnpackOptions above
    c.scales = std::move(g.scales);    // log-scale as-is (the same convention on both sides)
    c.sh0 = std::move(g.colors);       // the f_dc SH0 coefficients as-is
    c.shN = std::move(g.sh);           // identical [N,K,3] layout — no conversion (the SpzIO.h table)

    // quats: spz xyzw (w-last) → SplatCloud w-first (the exact inverse of the save_spz rearrangement)
    c.quats.resize(N * 4);
    for (size_t i = 0; i < N; i++) {
        c.quats[i * 4 + 0] = g.rotations[i * 4 + 3]; // w
        c.quats[i * 4 + 1] = g.rotations[i * 4 + 0]; // x
        c.quats[i * 4 + 2] = g.rotations[i * 4 + 1]; // y
        c.quats[i * 4 + 3] = g.rotations[i * 4 + 2]; // z
    }

    // Finite clamp on opacity (Task 2 carry-over — the kSpzOpacityLogitLimit rationale comment in
    // SpzIO.h): unpacking the saturated bytes 255/0 yields ±inf (invSigmoid(1)/invSigmoid(0) —
    // load-spz.cc:518). SplatCloud's contract is the absence of NaN/Inf — clamp to ±ln(509) (finite
    // byte values never reach this range).
    c.opacities.resize(N);
    for (size_t i = 0; i < N; i++)
        c.opacities[i] =
            std::clamp(g.alphas[i], -kSpzOpacityLogitLimit, kSpzOpacityLogitLimit);
    return c;
}

} // namespace splatx
