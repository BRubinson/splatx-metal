// Writer/reader for the standard INRIA 3DGS PLY layout + the export filter (Phase 2 Task 5).
#pragma once
#include "splatx/SplatCloud.h"
#include <cstdint>
#include <string>

namespace splatx {

// pad_sh_full (default true): if 0 < K < 15, f_rest is zero-padded out to 45 entries (design §8.1 —
// "keep PLY standard compatibility by zero-padding f_rest on export"). Basis, audited on
// 2026-07-13 against research/MetalSplatter tag 1.0.1 (= the iOS capture app's pin):
// SplatIO's SplatPLYSceneReader.swift, once f_rest_0 is present, requires **all 45** of
// f_rest_0..44 as floats (a fixed sphericalHarmonicsCount=45 loop — deg1's 9 entries fail with
// "No property named f_rest_9", and deg2's 24 entries are likewise rejected at f_rest_24).
// So the minimum padding for viewer compatibility is the full 45.
// deg0 (K=0) is accepted by the f_rest-absent branch, so it is not padded (saving 180 B per row).
// read_inria_ply infers a padded file as deg3 (with the higher coefficients zero) — use false only
// for raw dumps that must preserve the degree (the render result is identical: higher-band
// coefficients of 0 contribute nothing).
void write_inria_ply(const SplatCloud& cloud, const std::string& path, bool pad_sh_full = true);
SplatCloud read_inria_ply(const std::string& path); // parses property names, infers degree from f_rest count

// ===== export filter (design §5.4 — deployment-quality cleanup before the PLY is saved) =====
// Checked against the measured conventions of gsplat exporter.py (commit 77ab983):
//   (1) NaN/Inf filter: export_splats:629–651 — if **any parameter** of a row (means/scales/
//       quats/opacity/sh0/shN) is NaN or Inf, the row is removed. Common to every format
//       including the standard "ply" format (always on).
//   (2) Low-opacity filter: **absent** from the upstream standard "ply" path; only the compressed
//       format (splat2ply_bytes_compressed:236) applies `sigmoid(op) > 1/255`. Per design §5.4
//       (addressing gsplat #822), splatx applies it to the standard PLY as well, but with a
//       threshold of 0.005 — the same value as the MCMC dead criterion (mcmc.py:88 min_opacity)
//       and DefaultStrategy prune_opa (default.py:54). The predicate is the same strict
//       `sigmoid(op) > min_opacity` as the original (dead = the complement of `<=` — consistent
//       with Trainer::mcmc_relocate). min_opacity <= 0 is naturally a no-op (sigmoid > 0 always
//       holds).
//   (3) Scale clamp: **omitted** — confirmed that the standard "ply" path of export_splats has no
//       maximum-scale clamp (the [-20,20] at exporter.py:310–311 is used only to compute the
//       chunk quantization bounds of the compressed format). Among the huge-scale artifacts of
//       the #785 kind, the non-finite values are removed by (1), and finite huge scales are
//       preserved exactly as the upstream convention prescribes (documented in task-5-report).
// Only the first count rows of the input are read (the MCMC cap_max arena tail is untouched —
// only the alive rows are saved).
struct ExportFilterStats {
    int64_t input = 0;               // input row count (= cloud.count)
    int64_t removed_nonfinite = 0;   // NaN/Inf rows (rows also below the opacity threshold land here)
    int64_t removed_low_opacity = 0; // finite but sigmoid(op) <= min_opacity
    int64_t kept = 0;                // output row count
};
SplatCloud filter_cloud_for_export(const SplatCloud& cloud, float min_opacity = 0.005f,
                                   ExportFilterStats* stats = nullptr);

} // namespace splatx
