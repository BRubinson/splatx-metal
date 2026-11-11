// splatx-metal — the single source of truth for the kernel parameter structs
// (shared by host C++ and device MSL)
// Derived from nerfstudio-project/gsplat (Apache-2.0), commit 77ab983.
// Reimplemented for Metal/C++.
//
// D4: `<area>Params` in PascalCase. D5-4: every TU includes this single definition (ODR agreement).
// The fields are 1:1 with the gsplat kernel arguments (docs/kernel-specs/01, 02, 06 — exactly the
// blocks in the spec bodies).
// Every field is a 4-byte scalar → identical host/device layout (no padding, static_assert below).
#ifndef SPLATX_KERNEL_PARAMS_H
#define SPLATX_KERNEL_PARAMS_H

#ifndef __METAL_VERSION__
#include <cstdint>
typedef std::uint32_t uint; // same type as uint in Darwin sys/types.h — redeclaration allowed
#endif

// spec 01 "Constant/parameter block" (passed via setBytes)
struct ProjectionParams {
    uint  N;                  // number of Gaussians (Phase 1: B=1)
    uint  C;                  // number of cameras
    uint  image_width;
    uint  image_height;
    float eps2d;              // default 0.3
    float near_plane;         // default 0.01
    float far_plane;          // default 1e10
    float radius_clip;        // default 0.0
    uint  camera_model;       // 0=PINHOLE, 1=ORTHO, 2=FISHEYE
    uint  has_opacities;      // fwd only
    uint  calc_compensations; // fwd: emit the compensation term / bwd: use v_compensations
};

// spec 02 "Proposed kernel signature"
struct SHParams {
    uint degrees_to_use; // 0..4 (Phase 1 verification covers 0..3)
    uint B;              // flattened batch*camera count
    uint N;              // number of Gaussians
    uint K;              // number of SH bands ((deg+1)^2 <= K)
    uint D;              // number of channels (RGB=3)
};

// spec 06 fused_adam (AdamCUDA.cu:34–71) — no bias correction (exactly as in the upstream original)
struct AdamParams {
    uint  N;   // number of Gaussians
    uint  D;   // elements per Gaussian (param length = N*D)
    float lr;
    float b1;
    float b2;
    float eps;
};

// spec 03 "Parameter block" — isect_tile_count/emit (corresponding to the IntersectTile.cu:233–237
// launch arguments). dense only (disposition #14 — packed/tile_mask excluded), single device
// (key_start/key_end removed).
struct IsectTileParams {
    uint n_elements;   // I * N (dense)
    uint I;            // number of images
    uint N;            // Gaussians per image
    uint tile_size;
    uint tile_width;
    uint tile_height;
    uint tile_n_bits;  // bits_for_count(tile_width*tile_height) — computed on the host (spec 03 §1)
    uint image_n_bits; // bits_for_count(I) — for validation (image_n_bits + tile_n_bits <= 32)
};

// spec 03 §Kernel 4 isect_offset_encode (corresponding to the launch_intersect_offset_kernel
// launch arguments, IntersectTile.cu:990–1026). tile_n_bits is computed on the host with
// bits_for_count — the same single source of truth as the isect_tiles key encoding
// (pack/unpack round-trip, spec 03 §1).
struct IsectOffsetParams {
    uint n_isects;
    uint I;           // number of images (offsets = [I, tile_height, tile_width] flattened)
    uint n_tiles;     // tile_width * tile_height
    uint tile_n_bits; // bits_for_count(n_tiles)
};

// spec 05 rasterize fwd/bwd (corresponding to the RasterizeToPixels3DGSSerialBatchFwd.cu:43-62
// launch arguments — N/packed are omitted because the kernel body does not use them (spec 05
// second verification pass), and block_offset is removed because there is a single device).
// TILE_SIZE/CTA_SIZE/CDIM are compile-time constants (KernelConstants.h kRast*) — they are not
// params. Dispatch: one threadgroup = one tile, grid = I*tile_height*tile_width.
struct RasterizeParams {
    uint n_isects;     // length of flatten_ids (the range_end sentinel of the last tile)
    uint image_width;
    uint image_height;
    uint I;            // number of images (per-image pointer advance — Fwd.cu:92-103)
    uint tile_width;   // grid_w
    uint tile_height;  // grid_h
};

// spec 04 radix sort (shared by the 5 kernels of RadixSort.metal, setBytes per pass). Difference
// from brush lib.rs (spec 04 Correction 6 + decision D9): brush passes num_keys as a GPU buffer
// [1], but this code adopts the **readback + exact-size path** (n is already on the host when
// sort_isect is entered) — it avoids capacity dispatch (an unproven combination) and instead has
// the host fix num_keys/num_wgs/num_reduce_wgs and pass them as params. The kernel's
// group_id >= num_wgs gate is retained for parity with the original (it never fires under
// exact-size dispatch). num_keys ≤ INT32_MAX is guaranteed by validate_sort_isect_inputs.
struct RadixSortParams {
    uint shift;          // pass * kRadixBitsPerPass (0..60 — u64 keys)
    uint num_keys;       // n (exact size — gates the partially occupied last workgroup)
    uint num_wgs;        // ceil(num_keys / kRadixBlockSize) — number of count/scatter groups
    uint num_reduce_wgs; // kRadixBinCount * ceil(num_wgs / kRadixBlockSize) — reduce/scan_add
};

// spec 06 mcmc_relocation (corresponding to the RelocationCUDA.cu:35–79 launch arguments) — the
// binoms LUT is an [n_max, n_max] row-major constant buffer (precomputed on the host by
// make_binoms_lut, n_max=51).
struct RelocParams {
    uint  N;           // number of Gaussians (samples)
    int   n_max;       // one edge of the LUT — kept as the original int argument (strategy/mcmc.py:95 n_max=51)
    float min_opacity; // default 0.005 (strategy/mcmc.py:88) — the clamp precedes the scale computation
};

// spec 06 mcmc_perturb_positions (corresponding to the MCMCPerturbCUDA.cu:63–97 launch arguments).
struct PerturbParams {
    uint  N;
    float noise_scale; // lr * noise_lr (strategy/mcmc.py:190, noise_lr default 5e5)
    float t;           // DEFAULT_MCMC_OPACITY_T = 0.005 (strategy/ops.py:35)
    float k;           // DEFAULT_MCMC_OPACITY_K = 100.0 (strategy/ops.py:36)
};

// spec 08 ssim_fwd/ssim_bwd (Ssim.metal — designed in-house, consistent with the gsplat fused_ssim
// call contract). map_h/map_w = H-10/W-10 (valid map — 08 §3). G is a bwd-only uniform scalar
// = -v_loss/M (M = I·C·map_h·map_w; the host computes it in f64 and casts to f32 — 08 §4-(3)).
// fwd does not use G (0).
struct SsimParams {
    uint  I;     // number of images (grid z = I*C)
    uint  H;     // input height
    uint  W;     // input width
    uint  C;     // number of channels (layout [I,H,W,C] channel-last — the D6 render_colors convention)
    uint  map_h; // H - 10
    uint  map_w; // W - 10
    float G;     // bwd: -v_loss/M
};

// ===== parameters of the small device-resident train_step elementwise kernels (Phase 3 Task 5) =====
// TrainStep.metal — written in-house (the D7 host operations of Trainer.cpp turned into kernels —
// Trainer.cpp is the single source of truth for the formulas and their order; the kernels
// reproduce the same expressions).

// Shared by train_exp / train_sigmoid / train_ew_mul / train_sh_post_op / train_relu_mask_apply /
// train_add_inplace / train_aa_chain — only the element count is needed.
struct TrainEltParams {
    uint n; // total element count
};

// train_prep_sh — one thread per Gaussian: valid = (radii>0).all(-1), dirs = means - campos.
struct TrainPrepShParams {
    uint  N;
    float campos_x; // the host computes it as -Rᵀt (same expression as campos in Trainer.cpp)
    float campos_y;
    float campos_z;
};

// train_assemble_coeffs / train_split_v_coeffs — coeffs [N,K,3] ↔ sh0 [N,3] + shN [N,K-1,3].
struct TrainCoeffsParams {
    uint N;
    uint K; // K_total = (sh_degree+1)^2 (>= 1)
};

// train_l1_grad — v_render = w_l1·sign(render-gt)·g (+ v_dssim if has_dssim).
// g = 1/n (f32 — same as the host l1_loss_and_grad), sign(0) = 0 (the torch.sign convention).
struct TrainL1GradParams {
    uint  n;         // H*W*3
    uint  has_dssim; // if 1, add v_dssim (ssim_weight > 0)
    float g;         // 1/n
    float w_l1;      // 1 - ssim_weight
};

// train_scale_grad_chain / train_opacity_grad_chain — the activation gradient chain plus the
// regularization addition. reg_w = reg > 0 ? reg / the mean denominator : 0 (the same expression
// as w in the host add_*_reg_grad_inplace, including the > 0 gate — with 0 nothing is added).
struct TrainGradChainParams {
    uint  n;
    float reg_w;
};

// train_depth_grad — the confidence-masked inverse-depth L1 over the expected depth (reproduces
// the host depth_loss_and_grad). g = depth_weight / (count of valid sidecar samples over the view).
struct TrainDepthGradParams {
    uint  width;        // view
    uint  height;
    uint  depth_width;  // sidecar
    uint  depth_height;
    float g;
};

// The region mask (kTrainHasMask — RegionMask.h): a [mask_height,mask_width] 0/1 grid sampled
// nearest-neighbour for each view pixel through sx_depth_sidecar_coord. map_w/map_h are the valid
// SSIM map dims (train_ssim_mask_coeffs; the map pixel (x, y) is centred on view pixel (x+5, y+5)).
struct TrainMaskParams {
    uint width;       // view
    uint height;
    uint mask_width;  // sidecar
    uint mask_height;
    uint map_w;       // W - 10
    uint map_h;       // H - 10
};

// train_scale_shape_reg — per Gaussian, with linear scales sorted s1 ≥ s2 ≥ s3:
//   d/dlog s3 += flatten_w·s3;   r = s1/s2 > needle_ratio → d/dlog s1 += needle_w·r, d/dlog s2 −= needle_w·r
// flatten_w = flatten_weight/N and needle_w = needle_weight/N (computed on the host).
struct TrainShapeRegParams {
    uint  n;
    float flatten_w;
    float needle_w;
    float needle_ratio;
};

#ifndef __METAL_VERSION__
static_assert(sizeof(TrainMaskParams) == 24, "TrainMaskParams: host/device layout mismatch");
static_assert(sizeof(TrainShapeRegParams) == 16,
              "TrainShapeRegParams: host/device layout mismatch");
static_assert(sizeof(TrainEltParams) == 4, "TrainEltParams: host/device layout mismatch");
static_assert(sizeof(TrainPrepShParams) == 16, "TrainPrepShParams: host/device layout mismatch");
static_assert(sizeof(TrainCoeffsParams) == 8, "TrainCoeffsParams: host/device layout mismatch");
static_assert(sizeof(TrainL1GradParams) == 16, "TrainL1GradParams: host/device layout mismatch");
static_assert(sizeof(TrainGradChainParams) == 8,
              "TrainGradChainParams: host/device layout mismatch");
static_assert(sizeof(TrainDepthGradParams) == 20,
              "TrainDepthGradParams: host/device layout mismatch");
#endif

// spec 06 rng_normal_f32 / rng_philox_u32 (Rng.metal — written in-house, literature citation in
// RngPhilox.h). counter = (tid, offset_lo, offset_hi, 0), key = (seed_lo, seed_hi) — stateless
// reproduction. The u64 seed/offset are split into lo/hi (keeping the convention that every field
// is a 4-byte scalar — identical layout).
struct RngParams {
    uint seed_lo;
    uint seed_hi;
    uint offset_lo; // used as the training step counter (a separate stream per step)
    uint offset_hi;
    uint count;     // total output elements — 4 per thread; the last thread writes partially
};

#ifndef __METAL_VERSION__
static_assert(sizeof(ProjectionParams) == 44, "ProjectionParams: host/device layout mismatch");
static_assert(sizeof(SHParams) == 20, "SHParams: host/device layout mismatch");
static_assert(sizeof(AdamParams) == 24, "AdamParams: host/device layout mismatch");
static_assert(sizeof(IsectTileParams) == 32, "IsectTileParams: host/device layout mismatch");
static_assert(sizeof(IsectOffsetParams) == 16, "IsectOffsetParams: host/device layout mismatch");
static_assert(sizeof(RasterizeParams) == 24, "RasterizeParams: host/device layout mismatch");
static_assert(sizeof(RadixSortParams) == 16, "RadixSortParams: host/device layout mismatch");
static_assert(sizeof(RelocParams) == 12, "RelocParams: host/device layout mismatch");
static_assert(sizeof(PerturbParams) == 16, "PerturbParams: host/device layout mismatch");
static_assert(sizeof(RngParams) == 20, "RngParams: host/device layout mismatch");
static_assert(sizeof(SsimParams) == 28, "SsimParams: host/device layout mismatch");
#endif

#endif // SPLATX_KERNEL_PARAMS_H
