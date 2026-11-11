// splatx-metal — shared kernel constants (shared by host C++ and device MSL)
// Derived from nerfstudio-project/gsplat (Apache-2.0), commit 77ab983 (Common.h).
// Reimplemented for Metal/C++.
//
// D4: constants are PascalCase with a `k` prefix and all live in this one file
//     (docs/kernel-specs/00-index.md §2).
// D5-2: program-scope constants use internal linkage only (constant constexpr) — prevents link collisions.
#ifndef SPLATX_KERNEL_CONSTANTS_H
#define SPLATX_KERNEL_CONSTANTS_H

#ifdef __METAL_VERSION__
#define SX_KCONST constant constexpr float
#else
#define SX_KCONST inline constexpr float
#endif

SX_KCONST kAlphaThreshold         = 1.f / 255.f; // gsplat Common.h:97 ALPHA_THRESHOLD
SX_KCONST kGaussianExtend         = 3.33f;       // gsplat Common.h:99 GAUSSIAN_EXTEND (opacity-aware radius)
SX_KCONST kMinCompensation        = 0.005f;      // gsplat Common.h:110 MIN_COMPENSATION
SX_KCONST kMaxAlpha               = 0.99f;       // raster alpha upper bound (spec 05)
SX_KCONST kTransmittanceThreshold = 1e-4f;       // raster next_T early-termination threshold (spec 05)
SX_KCONST kMinOneMinusAlpha       = 1e-6f;       // raster bwd 1/(1-α) lower bound (spec 05)

// spec 08 SSIM constants (gsplat losses.py:142–143 — assumes a dynamic range of 1).
SX_KCONST kSsimC1 = 1e-4f; // (0.01)²
SX_KCONST kSsimC2 = 9e-4f; // (0.03)²

// spec 08 §3 1D Gaussian window taps (11 taps, σ=1.5) — **the single source of truth for these
// constants** (identical to window_taps_f32 in fixtures/ssim_basic meta.json): hardcoded values
// obtained by computing g(i)=exp(-(i-5)²/4.5) and w=g/Σg in f64 and then casting to f32 —
// no runtime transcendental functions (D2, the same contract as the relocation binoms LUT).
// The host CPU reference (ssim_fwd_cpu) and Ssim.metal share this single source of truth.
SX_KCONST kSsimWindow[11] = {
    0.001028380123898387f,  0.0075987582094967365f, 0.036000773310661316f,
    0.10936068743467331f,   0.21300554275512695f,   0.26601171493530273f,
    0.21300554275512695f,   0.10936068743467331f,   0.036000773310661316f,
    0.0075987582094967365f, 0.001028380123898387f};

// Depth supervision (expected depth ED = Σwz / α against a LiDAR sidecar): a pixel enters the
// inverse-depth loss and the depth RMSE only with α >= kDepthMinAlpha and ED >= kDepthMinZ (metres).
SX_KCONST kDepthMinAlpha = 0.01f;
SX_KCONST kDepthMinZ     = 1e-3f;

#undef SX_KCONST

// spec 03 §Kernel 2 prefix scan block configuration — the host (num_groups computation and grid
// size) and the device (block/thread indexing) MUST share the same values (single source of truth).
#ifdef __METAL_VERSION__
#define SX_KCONST_UINT constant constexpr unsigned int
#else
#define SX_KCONST_UINT inline constexpr unsigned int
#endif

SX_KCONST_UINT kScanThreadgroupSize = 256; // scan_partial/scan_blocks threadgroup size
SX_KCONST_UINT kScanElemsPerThread  = 4;   // elements accumulated serially per scan_partial thread
SX_KCONST_UINT kScanBlockSize       = kScanThreadgroupSize * kScanElemsPerThread; // 1024

// spec 04 radix sort constants — corresponding to brush kernels.rs:7–20 (WG/BITS_PER_PASS/
// BIN_COUNT/ELEMENTS_PER_THREAD/BLOCK_SIZE). The host (num_wgs/num_reduce_wgs/dispatch sizes)
// and the device (indexing) MUST share the same values (single source of truth).
// Caution: the packed 8-bit counter trick in scatter is coupled to the WG=256 premise (spec 04
// Verification Notes reinforcement — the carry/borrow cancels only for ranks and partial sums
// ≤ 256). If you change kRadixWG, you MUST re-examine the trick's premise.
SX_KCONST_UINT kRadixWG             = 256; // threadgroup size shared by all 5 radix kernels
SX_KCONST_UINT kRadixBitsPerPass    = 4;   // 4-bit digit LSD
SX_KCONST_UINT kRadixBinCount       = 1u << kRadixBitsPerPass; // 16
SX_KCONST_UINT kRadixElemsPerThread = 4;   // EPT
SX_KCONST_UINT kRadixBlockSize      = kRadixWG * kRadixElemsPerThread; // 1024 (keys per workgroup)
// brush MAX_SUBGROUPS=32 is padding for a variable subgroup width of 8–64 — Apple is fixed at 32
// (D8-3: pipeline() enforces threadExecutionWidth==32), so this is settled at 8
// (spec 04 §Adoption strategy, rationale 3).
SX_KCONST_UINT kRadixSimdWidth      = 32;
SX_KCONST_UINT kRadixNumSimdgroups  = kRadixWG / kRadixSimdWidth; // 8

// spec 05 rasterize fwd/bwd compile-time instances (only the tile_size==16 combination of the
// gsplat Fwd.cu:401-412 launcher — Phase 1). TILE=16/CTA=256 → BATCH_SIZE=256,
// PPT=16·16/256=1, ROW_STRIDE=16.
// CDIM sizes the local arrays (pix_out/buffer/v_rgb_local), so it MUST be a compile-time constant
// (spec 05 §CDIM) — Phase 1 has the single RGB=3 variant. The host (dispatch grid and buffer
// sizes) and the device (threadgroup array sizes) MUST share the same values (single source of truth).
SX_KCONST_UINT kRastTileSize = 16;  // pixels along one tile edge
SX_KCONST_UINT kRastCtaSize  = 256; // threadgroup size = BATCH_SIZE (cooperative load unit)
SX_KCONST_UINT kRastCdim     = 3;   // number of color channels (RGB)

// spec 08 §5 SSIM kernel tile configuration — the host (grid computation) and the device
// (threadgroup array sizes) MUST share the same values (single source of truth).
// 26×26 apron: tile 16 + window 11 - 1.
SX_KCONST_UINT kSsimWindowSize = 11;                                  // 11×11, σ=1.5 (fixed — 08 §2-5)
SX_KCONST_UINT kSsimTileSize   = 16;                                  // fwd/bwd threadgroup edge (16×16 = 256)
SX_KCONST_UINT kSsimApron      = kSsimTileSize + kSsimWindowSize - 1; // 26

#undef SX_KCONST_UINT

// Nearest-neighbour lookup of a view pixel in the depth sidecar grid: floor((x + 0.5)·side/view),
// shared by the host loss, the eval RMSE and train_depth_grad.
inline unsigned int sx_depth_sidecar_coord(unsigned int x, unsigned int view, unsigned int side) {
    const unsigned int s = ((2u * x + 1u) * side) / (2u * view);
    return s < side ? s : side - 1u;
}

#endif // SPLATX_KERNEL_CONSTANTS_H
