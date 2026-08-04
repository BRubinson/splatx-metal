// Niantic SPZ writer/reader (SPZ v3 — vendored spz v2.1.0), sharing the export filter
// (SPZ Task 2 writer / Task 3 reader). Structurally symmetric with PlyIO.
#pragma once
#include "splatx/PlyIO.h" // ExportFilterStats + filter_cloud_for_export (single source of truth for the export filter)
#include "splatx/SplatCloud.h"
#include <string>

namespace splatx {

// ===== The single source of truth for the SplatCloud ↔ spz::GaussianCloud encoding convention ==
// Audited against the ThirdParty/spz v2.1.0 sources (splat-types.h / load-spz.cc — line numbers
// refer to the vendored copy):
//
//   Field      SplatCloud (gsplat convention)  spz GaussianCloud                  Mapping
//   ---------- ------------------------------- ---------------------------------- ----------
//   means      [N,3] raw float                 positions raw float                as-is
//              (packing: 24-bit fixed point, fractionalBits=12 — load-spz.cc:270,
//               resolution 2^-12)
//   scales     [N,3] log-space                 log-space (splat-types.h:85)       as-is
//              (packing: uint8 (s+10)·16 — load-spz.cc:291, representable range [-10, 5.9375])
//   quats      [N,4] w-first, unnormalized     **xyzw (w-last)** (splat-types.h:86; the
//              allowed                         reordering is reconfirmed by
//              convertCoordinates:145–150, which flips only [0..2] and preserves [3]=w)
//              (packing: v3 = smallest-three, 9 bits + sign per component, normalization is
//               performed inside spz — load-spz.cc:216–255; pinning v3 is a user decision, and
//               deserialize:572 uses S3 only for v>=3)
//   opacities  [N] logit-space                 logit-space "before sigmoid        as-is
//                                              activation" (splat-types.h:87)
//              (packing: sigmoid→uint8 — load-spz.cc:301; sigmoid>254.5/255 (logit>~6.23)
//               saturates to 255 → +inf on reload. Inherent to the format — the writer does
//               not clamp it)
//   sh0        [N,3] f_dc SH0 coefficients     colors = SH DC coefficients        as-is
//                                              (splat-types.h:88)
//              (packing: uint8, colorScale=0.15 — load-spz.cc:45,306, step 1/38.25)
//   shN        [N,K,3] coefficient-outer,      sh uses the same layout            as-is
//              channel-inner                   (splat-types.h:107–114 "channel is
//                                              the inner (fastest varying) axis")
//              NOTE: differs from the channel-planar [N,3,K] of PLY f_rest — the transpose is
//                PlyIO writer only (PlyIO.cpp:39–41). The spz path copies without conversion.
//              (packing: band 1 at 5 bits, bands 2 and 3 at 4 bits, bucket quantization —
//               load-spz.cc:312–327)
//
// Coordinate system (the reasoning — also recorded in task-2-report):
//   The contents of an spz file are defined as RUB (packGaussians:263 converts
//   PackOptions.from→RUB, and unpackGaussians:529 applies the inverse RUB→UnpackOptions.to).
//   spz treats PLY as RDF (loadSplatFromPly:842 RDF→to, saveSplatToPly:856 from→RDF). The splatx
//   PLY path is a raw dump of the training world coordinates (the INRIA/gsplat convention) = RDF
//   in spz's naming. Therefore save_spz MUST pack with from=RDF so that a convention-conforming
//   viewer (which assumes RUB) renders it in the **same pose** as the PLY.
//   RDF→RUB is a 180° rotation about the x axis (sign flip of y and z), and the SH side is handled
//   by spz's flipSh sign table — an axis sign flip needs no general Wigner rotation (the design
//   §8.3 issue does not arise in this case).
//   Round-trip verification: spz::loadSpz(to=RDF) restores the original, while a raw
//   (UNSPECIFIED) load has y and z flipped (SpzIoTests.cpp).
struct SpzSaveOptions {
    bool antialiased = false;      // TrainConfig.antialiased → SPZ header flags bit0
                                   // (load-spz.cc:129 FlagAntialiased, written at :538)
    float min_opacity = 0.005f;    // export filter threshold (same default as the PlyIO.h convention)
    ExportFilterStats* filter_stats = nullptr; // optional — retrieves the filter stats (for logging)
};

// Saves a SplatCloud to an SPZ v3 file. The export filter (filter_cloud_for_export) is applied
// internally — unlike PLY (a raw float dump), SPZ packing quantizes through a round()→int32 cast,
// so a NaN/Inf row leads to UB. The filter is therefore part of the save_spz contract, not a
// caller policy.
// Throws if sh_degree ∉ [0,3] or the write fails.
void save_spz(const SplatCloud& cloud, const std::string& path, const SpzSaveOptions& opts = {});

// ===== import (Task 3 — the exact inverse of save_spz) =====
// The finite-clamp limit on opacity: SPZ opacity is packed as uint8 in sigmoid space (table
// above), so byte 255 represents all of sigmoid ∈ (254.5/255, 1] and byte 0 represents all of
// [0, 0.5/255). spz unpack returns invSigmoid(byte/255), so byte 255 → invSigmoid(1) = +inf and
// byte 0 → invSigmoid(0) = -inf (load-spz.cc:518 — a Task 2 carry-over concern). The SplatCloud
// contract is the absence of NaN/Inf (on retraining the export filter removes the row, losing
// data), so load_spz clamps the logit to ±ln(509) — ln(509) = invSigmoid(254.5/255) ≈ 6.2324 is
// the lower bound of the interval that byte 255 represents (= the finite representative with the
// smallest representation error between the byte-254 value ln(254) ≈ 5.537 and +inf).
// The unpacked values of the finite bytes (1..254) satisfy |logit| <= ln(509), so the clamp fires
// only on the saturated bytes.
// Fields other than opacity unpack as an affine transform of the byte, so they cannot produce a
// non-finite value.
inline constexpr float kSpzOpacityLogitLimit = 6.2324481f; // ln(509) = invSigmoid(254.5/255)

// SPZ file → SplatCloud (an API symmetric with read_inria_ply). The reverse direction of the
// encoding convention table:
//   - spz::loadSpz + UnpackOptions.to = RDF — file (RUB) → training world (RDF), exactly
//     symmetric with save's PackOptions.from = RDF (coordinate system section above).
//   - quats: spz xyzw (w-last) → SplatCloud w-first reordering (the inverse of save).
//   - shN: the layout is identical ([N,K,3] coefficient-outer, channel-inner) — copied without
//     conversion.
//   - sh_degree: inferred from the SPZ header shDegree field (spz deserialize rejects anything
//     outside 0..3).
//   - opacities: finite clamp to ±kSpzOpacityLogitLimit (rationale above).
// If antialiased != nullptr, SPZ header flags bit0 is retrieved (information SplatCloud does not
// carry — for preserving the flag across an SPZ→SPZ conversion).
// Failure throws. Note: the vendored spz C++ API returns an empty GaussianCloud on every failure
// (missing file, corruption, unsupported version, or more than 10M points), which is
// indistinguishable from a legitimately empty file (load-spz.cc:548–593 return {}) — so load_spz
// rejects numPoints == 0 outright (an empty SPZ is the export-filter-wiped-everything warning
// case and is therefore meaningless as an import target; the choice is to not let a corrupt file
// pass silently as an empty cloud).
SplatCloud load_spz(const std::string& path, bool* antialiased = nullptr);

} // namespace splatx
