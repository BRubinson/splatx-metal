#pragma once
// The shared single source of truth for kernel-entry input validation (Phase 1b Task 1 —
// delegated by the Phase 1a final review).
// The CPU reference (CpuKernels.cpp) and MetalBackend share the same functions and therefore throw
// with identical semantics (this removes the duplicated separate implementations). `what` is the
// exception message prefix — pass the caller's entry point name ("fused_adam_cpu" / "fused_adam" /
// "sh_fwd", and so on).
// New validators for Tasks 3–7 (isect/scan/sort) go into this header as well (do not redefine
// them elsewhere).
#include <cstdint>
#include <stdexcept>
#include <string>

#include "KernelConstants.h" // kRastTileSize (the Phase 1 rasterize instantiation — single
                             // source of truth shared with the kernel)
#include "splatx/KernelBackend.h"

namespace splatx {

// ===== General-purpose primitives =====

[[noreturn]] inline void throw_invalid(const char* what, const std::string& msg) {
    throw std::invalid_argument(std::string(what) + ": " + msg);
}

inline void validate_nonnull(const void* p, const char* what, const char* name) {
    if (!p) throw_invalid(what, std::string("null ") + name);
}

inline void validate_range_u32(std::uint32_t v, std::uint32_t lo, std::uint32_t hi,
                               const char* what, const char* name) {
    if (v < lo || v > hi)
        throw_invalid(what, std::string(name) + " must be " + std::to_string(lo) + ".." +
                                std::to_string(hi));
}

// ===== Per-kernel composite validators (promoted from the three duplicates of Phase 1a) =====

// Spec 06 fused_adam — the n_elems==0 dispatch skip (D8-6) is decided by the caller beforehand.
inline void validate_adam_inputs(const AdamInputs& in, const char* what) {
    if (!in.param || !in.grad || !in.exp_avg || !in.exp_avg_sq)
        throw_invalid(what, "null tensor");
    if (in.n_elems < 0 || in.n_gauss <= 0 || in.n_elems % in.n_gauss != 0)
        throw_invalid(what, "n_elems must be a positive multiple of n_gauss");
}

// Spec 01 projection_fused fwd/bwd shared inputs — PINHOLE only (Phase 1, disposition #3;
// an unsupported camera_model raises runtime_error, not invalid_argument — the existing contract
// is retained).
inline void validate_projection_fused_inputs(const ProjectionFusedInputs& in,
                                             const char* what) {
    if (in.camera_model != 0)
        throw std::runtime_error(std::string(what) +
                                 ": only PINHOLE (camera_model=0) supported in Phase 1");
    if (!in.means || !in.viewmats || !in.Ks)
        throw_invalid(what, "null means/viewmats/Ks");
    if (!in.covars && (!in.quats || !in.scales))
        throw_invalid(what, "need covars (6-triu) or quats+scales");
    if (in.N < 0 || in.C < 0)
        throw_invalid(what, "negative N or C");
}

// Spec 01 bwd additional inputs — the three golden fwd outputs + the upstream v_*.
// v_compensations != nullptr is the kProjCalcComp gate (the same nullptr gate as CUDA), and in
// that case the fwd output compensations must be supplied together with it.
inline void validate_projection_fused_grads_in(const ProjectionFusedGradsIn& g,
                                               const char* what) {
    if (!g.radii || !g.conics || !g.v_means2d || !g.v_depths || !g.v_conics)
        throw_invalid(what, "null radii/conics/v_means2d/v_depths/v_conics");
    if (g.v_compensations && !g.compensations)
        throw_invalid(what, "v_compensations requires compensations (fwd output)");
}

// Spec 02 sh_fwd/sh_bwd — (degree+1)^2 <= K corresponds to the input validation in gsplat
// SphericalHarmonics.cpp:48-54.
inline void validate_sh_inputs(const ShInputs& in, const char* what) {
    if (!in.dirs || !in.coeffs)
        throw_invalid(what, "null dirs/coeffs");
    if (in.B < 0 || in.N < 0)
        throw_invalid(what, "negative B or N");
    if (in.K < 1 || in.D == 0)
        throw_invalid(what, "K and D must be positive");
    if (in.degrees_to_use > 4)
        throw_invalid(what, "degrees_to_use must be 0..4");
    const std::int64_t bands = static_cast<std::int64_t>(in.degrees_to_use) + 1;
    if (bands * bands > in.K)
        throw_invalid(what, "(degrees_to_use+1)^2 must be <= K");
}

// Spec 03 isect_tiles — the key bit budget image_n_bits + tile_n_bits <= 32 corresponds to the
// TORCH_CHECK at IntersectTile.cu:513–522. flatten_ids is written as an int32 cast
// (IntersectTile.cu:459), so n_elements = I*N must also fit in int32 range.
inline void validate_isect_tiles_inputs(const IsectTilesInputs& in, const char* what) {
    if (!in.means2d || !in.radii || !in.depths)
        throw_invalid(what, "null means2d/radii/depths");
    // The AccuTile path gate requires both inputs present, same as the upstream original
    // (IntersectTile.cu:288) — a combination with only one of them is a state the original never
    // produces, so it is rejected.
    if ((in.conics != nullptr) != (in.opacities != nullptr))
        throw_invalid(what, "conics and opacities must be given together (AccuTile) or both null (AABB)");
    if (in.I < 0 || in.N < 0)
        throw_invalid(what, "negative I or N");
    if (in.tile_size == 0 || in.tile_width == 0 || in.tile_height == 0)
        throw_invalid(what, "tile_size/tile_width/tile_height must be positive");
    const std::uint32_t tile_n_bits =
        bits_for_count(static_cast<std::int64_t>(in.tile_width) * in.tile_height);
    const std::uint32_t image_n_bits = bits_for_count(in.I);
    if (image_n_bits + tile_n_bits > 32)
        throw_invalid(what, "image_n_bits + tile_n_bits must be <= 32");
    if (in.N > 0 && in.I > INT32_MAX / in.N)
        throw_invalid(what, "I*N must fit in int32 (flatten_ids)");
}

// Spec 03 sort — n is confined to int32 range by the upstream flatten_ids (int32) contract and as
// a guard for the Task 7 GPU parameters (uint). key_n_bits is the effective sort bit width — in
// the isect pipeline it is 32 + tile_n_bits + image_n_bits ∈ [32, 64], but as a primitive 1..64 is
// allowed. Checking that the high bits of the keys are 0 is O(n), so it is owned by the CPU
// reference rather than by the validator (sort_isect_cpu — the validation-promotion rule).
inline void validate_sort_isect_inputs(const SortIsectInputs& in, const char* what) {
    if (in.n < 0) throw_invalid(what, "negative n");
    if (in.n > 0 && (!in.keys || !in.values)) throw_invalid(what, "null keys/values");
    if (in.n > INT32_MAX) throw_invalid(what, "n must fit in int32 (flatten_ids/GPU params)");
    validate_range_u32(in.key_n_bits, 1, 64, what, "key_n_bits");
}

// Spec 03 isect_offset_encode — the unpack (tile_n_bits) uses the same bits_for_count single
// source of truth as the isect_tiles encoding, so the bit budget validation is identical as well
// (corresponding to IntersectTile.cu:513–522). The offsets values and the boundary thread index
// are written as int32 (IntersectTile.cu:964, 985), so n_isects must also fit in int32 range, and
// I*n_tiles is the guard line for IsectOffsetParams (uint) and for the offsets table size.
inline void validate_isect_offset_inputs(const IsectOffsetInputs& in, const char* what) {
    if (in.n_isects < 0) throw_invalid(what, "negative n_isects");
    if (in.n_isects > 0 && !in.isect_ids) throw_invalid(what, "null isect_ids");
    if (in.I < 0) throw_invalid(what, "negative I");
    if (in.n_isects > 0 && in.I == 0)
        throw_invalid(what, "n_isects > 0 requires I > 0 (keys encode image ids)");
    if (in.tile_width == 0 || in.tile_height == 0)
        throw_invalid(what, "tile_width/tile_height must be positive");
    const std::int64_t n_tiles = static_cast<std::int64_t>(in.tile_width) * in.tile_height;
    if (bits_for_count(in.I) + bits_for_count(n_tiles) > 32)
        throw_invalid(what, "image_n_bits + tile_n_bits must be <= 32");
    if (in.n_isects > INT32_MAX)
        throw_invalid(what, "n_isects must fit in int32 (offsets values)");
    if (in.I > 0 && n_tiles > INT32_MAX / in.I)
        throw_invalid(what, "I*n_tiles must fit in int32 (offsets table)");
}

// Spec 05 rasterize fwd — Phase 1 ports only the tile_size==16 instantiation of the gsplat launcher
// (Fwd.cu:401-412; the tile_size==4 variant is CDIM>=32 only and is not ported — unsupported values
// throw, corresponding to the original AT_ERROR).
// The tile grid must cover the image for every pixel to be written (upstream this is guaranteed by
// the surrounding orchestration — here it is promoted to validation). n_isects/last_ids are written
// as int32 (Fwd.cu:293), and I*n_tiles guards RasterizeParams (uint) and the offsets table (the
// same budget as isect_offset_encode).
inline void validate_rasterize_fwd_inputs(const RasterizeFwdInputs& in, const char* what) {
    if (!in.means2d || !in.conics || !in.colors || !in.opacities || !in.isect_offsets)
        throw_invalid(what, "null means2d/conics/colors/opacities/isect_offsets");
    if (in.n_isects > 0 && !in.flatten_ids)
        throw_invalid(what, "null flatten_ids");
    if (in.I < 0 || in.N < 0 || in.n_isects < 0)
        throw_invalid(what, "negative I/N/n_isects");
    if (in.n_isects > 0 && in.N == 0)
        throw_invalid(what, "n_isects > 0 requires N > 0 (flatten_ids reference gaussians)");
    if (in.tile_size != kRastTileSize)
        throw_invalid(what, "tile_size must be 16 (Phase 1 rasterize instantiation)");
    if (in.image_width == 0 || in.image_height == 0)
        throw_invalid(what, "image_width/image_height must be positive");
    if (in.tile_width == 0 || in.tile_height == 0)
        throw_invalid(what, "tile_width/tile_height must be positive");
    if (static_cast<std::uint64_t>(in.tile_width) * in.tile_size < in.image_width ||
        static_cast<std::uint64_t>(in.tile_height) * in.tile_size < in.image_height)
        throw_invalid(what, "tile grid must cover the image");
    if (in.n_isects > INT32_MAX)
        throw_invalid(what, "n_isects must fit in int32 (last_ids)");
    const std::int64_t n_tiles = static_cast<std::int64_t>(in.tile_width) * in.tile_height;
    if (in.I > 0 && n_tiles > INT32_MAX / in.I)
        throw_invalid(what, "I*n_tiles must fit in int32 (offsets table)");
}

// Spec 05 rasterize bwd — reuses the shared fwd input validation and additionally requires the two
// fwd outputs and the two upstream gradients (Bwd.cu:62-68). absgrad is an output option, so it is
// not validated (kRastHasAbsGrad specialization).
inline void validate_rasterize_bwd_inputs(const RasterizeFwdInputs& in,
                                          const RasterizeBwdGradsIn& g, const char* what) {
    validate_rasterize_fwd_inputs(in, what);
    if (!g.render_alphas || !g.last_ids)
        throw_invalid(what, "null render_alphas/last_ids (fwd outputs)");
    if (!g.v_render_colors || !g.v_render_alphas)
        throw_invalid(what, "null v_render_colors/v_render_alphas");
}

// The n_isects readback guard for the D11 resident executor (Phase 3.5 Task 1 — carried over from
// the Phase 3 final review).
// It validates the int64 prefix scan readback before the cast into
// RasterizeParams/IsectOffsetParams.n_isects (uint32), turning an over-the-limit value into an
// explicit throw instead of a silent truncation. The limit is **int32**, not uint32 — the offsets
// values (IntersectTile.cu:964), last_ids (Fwd.cu:293), and flatten_ids are written as int32, so
// this is the same boundary as the host-path validators (validate_isect_offset_inputs /
// validate_rasterize_fwd_inputs) (path parity; it covers the uint32 truncation concern as well).
// This exists purely to write the contract down for an unreachable range: at INT32_MAX isects the
// resident keys/vals buffers alone are ~48 GiB, so the allocation-failure throw in
// ensure_isect_capacity always fires first.
inline std::uint32_t validate_n_isects_readback(std::int64_t n_isects, const char* what) {
    if (n_isects < 0) throw_invalid(what, "negative n_isects (prefix scan readback)");
    if (n_isects > INT32_MAX)
        throw_invalid(what, "n_isects must fit in int32 (offsets/last_ids — host-path parity)");
    return static_cast<std::uint32_t>(n_isects);
}

// Spec 06 rng_normal_f32/rng_philox_u32 — count is confined to the RngParams.count (uint) range.
// The count==0 skip is decided by the caller (the same convention as D8-6).
inline void validate_rng_inputs(const RngNormalInputs& in, const char* what) {
    if (in.count < 0) throw_invalid(what, "negative count");
    if (in.count > static_cast<std::int64_t>(UINT32_MAX))
        throw_invalid(what, "count must fit in uint32 (RngParams.count)");
}

// Spec 06 mcmc_relocation — checking the ratios value range (1..n_max) is O(N), so it is owned by
// the CPU reference rather than by the validator (the same validation-promotion pattern as the high
// bits of the sort_isect keys — Metal performs no check, exactly as the upstream original).
// The binoms size (n_max*n_max) is a pointer contract and cannot be validated — it is the caller's
// responsibility.
inline void validate_relocation_inputs(const RelocationInputs& in, const char* what) {
    if (!in.opacities || !in.scales || !in.ratios || !in.binoms)
        throw_invalid(what, "null opacities/scales/ratios/binoms");
    if (in.N < 0) throw_invalid(what, "negative N");
    if (in.n_max < 1) throw_invalid(what, "n_max must be >= 1");
    if (in.N > static_cast<std::int64_t>(UINT32_MAX))
        throw_invalid(what, "N exceeds uint32 (RelocParams.N)");
}

// Spec 08 ssim fwd/bwd shared inputs — H, W >= 11 is the host guard that keeps the valid map
// non-empty (so the mean is defined) (the 08 §5 host contract; gsplat yields NaN in that case — it
// cannot occur under training conditions, so the original is unguarded and this is a validation
// promotion). I, C > 0 for the same reason (M = I·C·(H-10)·(W-10) > 0). I·C guards the dispatch
// grid z axis (uint).
inline void validate_ssim_inputs(const SsimInputs& in, const char* what) {
    if (!in.img1 || !in.img2) throw_invalid(what, "null img1/img2");
    if (in.I <= 0 || in.C <= 0)
        throw_invalid(what, "I and C must be positive (ssim mean undefined on empty batch)");
    if (in.H < 11 || in.W < 11)
        throw_invalid(what, "H and W must be >= 11 (valid map would be empty)");
    if (in.H > UINT32_MAX || in.W > UINT32_MAX)
        throw_invalid(what, "H/W exceed uint32 (SsimParams)");
    if (in.I > static_cast<std::int64_t>(UINT32_MAX) / in.C)
        throw_invalid(what, "I*C exceeds uint32 (dispatch grid z)");
}

// Spec 08 ssim bwd additional inputs — the three coefficient maps saved by fwd (need_coeffs=true)
// are required (08 §5 kernel 2).
inline void validate_ssim_grads_in(const SsimGradsIn& g, const char* what) {
    if (!g.coeff_mu || !g.coeff_sigma || !g.coeff_cross)
        throw_invalid(what, "null coeff maps (run ssim_fwd with need_coeffs=true)");
}

// Spec 06 mcmc_perturb_positions — N ≤ uint32 corresponds to the upstream TORCH_CHECK
// (MCMCPerturbCUDA.cu:79). noise is a required input buffer (there is no RNG inside the kernel —
// 06 §3).
inline void validate_perturb_inputs(const PerturbInputs& in, const char* what) {
    if (!in.positions || !in.quats || !in.scales_log || !in.opacities_logit || !in.noise)
        throw_invalid(what, "null positions/quats/scales_log/opacities_logit/noise");
    if (in.N < 0) throw_invalid(what, "negative N");
    if (in.N > static_cast<std::int64_t>(UINT32_MAX))
        throw_invalid(what, "N exceeds uint32 (MCMCPerturbCUDA.cu:79)");
}

} // namespace splatx
