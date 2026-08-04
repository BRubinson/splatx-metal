# 05 — Tile rasterize forward/backward (3DGS)

CUDA → Metal porting spec. Extracts gsplat's tile-based alpha-blending rasterizer (forward) and its gradient kernel (backward) at code-level precision.

## Original files and commit (77ab983)

- gsplat commit: `77ab983ffe43420b2131669cb35776b883ca4c3c` ("[NV] uint64 pixel mask, sparse-bwd robustness fixes, CUDA 12.8 build, ortho camera (#1023)")
- License: Apache-2.0 (a derivation notice is required)

| Role | File |
|---|---|
| Forward kernel + launcher | `research/gsplat/gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchFwd.cu` |
| Backward kernel + launcher | `research/gsplat/gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchBwd.cu` |
| Shared header for the per-pixel blending formulas (`GaussianWeight`, `eval_gaussian_weight`, the blend fwd/bwd device functions) | `research/gsplat/gsplat/cuda/csrc/RasterizeToPixels3DGSDevice.cuh` |
| Constant definitions (`ALPHA_THRESHOLD`, `MAX_ALPHA`, `TRANSMITTANCE_THRESHOLD`, `MIN_ONE_MINUS_ALPHA`) | `research/gsplat/gsplat/cuda/include/Common.h:97-114` |
| `warpSum` reduction | `research/gsplat/gsplat/cuda/include/Utils.cuh:152-200` |
| CDIM dispatch list (`GSPLAT_NUM_CHANNELS`) | `research/gsplat/gsplat/cuda/csrc/Config.h:71-72` |

Constant values (Common.h:97-114):

```c
#define ALPHA_THRESHOLD         (1.f / 255.f)   // ≈ 0.003921568627
#define MAX_ALPHA               0.99f
#define TRANSMITTANCE_THRESHOLD 1e-4f           // = (1 - MAX_ALPHA)^2
#define MIN_ONE_MINUS_ALPHA     1e-6f           // lower bound of the 1/(1-alpha) denominator in bwd
```

The rationale comment at Common.h:100-104: "MAX_ALPHA and TRANSMITTANCE_THRESHOLD are chosen so that the equivalent of a maximal opacity Gaussian has to be rasterized twice to reach the threshold, without getting the transmittance too small for numerical stability of the backward pass. i.e. TRANSMITTANCE_THRESHOLD = (1 - MAX_ALPHA)^2"

## Algorithm overview

- The image is split into `tile_size × tile_size` tiles, and the inputs are `isect_offsets` (the per-tile start offset, [I, tile_h, tile_w]) and `flatten_ids` (depth-sorted flattened Gaussian indices, [n_isects]) produced by the preceding stages (intersect/sort). The sort is front-to-back: the `range_start` end is the Gaussian closest to the camera (blended first).
- **Forward**: one threadgroup (= CTA) handles one tile. Gaussians are cooperatively loaded into shared memory `BATCH_SIZE = CTA_SIZE` at a time, then each thread alpha-blends front-to-back for its own pixel. When the transmittance `T` drops to `TRANSMITTANCE_THRESHOLD` or below, that pixel terminates. Outputs: `render_colors`, `render_alphas = 1 - T`, and `last_ids` (the isect index of the last contributing Gaussian).
- **Backward**: the same tiling, one thread per pixel. It walks the Gaussians **back-to-front** (the reverse of the sort order), recomputing `T` from `T_final` by division, and backpropagates `v_render_colors`/`v_render_alphas` into `v_means2d`/`v_conics`/`v_colors`/`v_opacities` (+ the optional `v_means2d_abs`). After a 32-lane warp reduction, only lane 0 performs a global atomicAdd.

### Shared Gaussian weight evaluation (RasterizeToPixels3DGSDevice.cuh:44-55)

conic = (A, B, C) = the (xx, xy, yy) components of the inverse 2D covariance. delta = (dx, dy) = mean2d − pixel_center.

```
sigma = 0.5f * (conic.x * dx*dx + conic.z * dy*dy) + conic.y * dx*dy   // :47
vis   = __expf(-sigma)                                                  // :48
alpha = min(MAX_ALPHA, opac * vis)                                      // :49  ← the 0.99 clamp
valid = !(sigma < 0.f || alpha < ALPHA_THRESHOLD)                       // :53  ← the 1/255 cutoff
```

If `valid == false`, that (Gaussian, pixel) pair is skipped in both forward and backward. `vis = exp(-sigma)` is reused directly in backward.

## Formulas in detail — forward

Original: `RasterizeToPixels3DGSSerialBatchFwd.cu:41-297`, template `<CDIM, TILE_SIZE, CTA_SIZE>`.

### Compile-time constants (Fwd.cu:65-71)

```c
BATCH_SIZE        = CTA_SIZE
PIXELS_PER_THREAD = TILE_SIZE * TILE_SIZE / CTA_SIZE   // PPT below
ROW_STRIDE        = CTA_SIZE / TILE_SIZE
TILE_MASK         = TILE_SIZE - 1
TILE_SHIFT        = ctz(TILE_SIZE)                      // 16 -> 4
ALL_DONE          = (1u << PPT) - 1u                    // the full done_mask bitmask
```

The combinations the launcher actually instantiates (Fwd.cu:401-412):
- `tile_size == 16` → `<TILE=16, CTA=256>` → **PPT = 1**, ROW_STRIDE = 16
- `tile_size == 4`  → `<TILE=4,  CTA=16>`  → **PPT = 1** (to avoid the bwd shared-memory limit at CDIM ≥ 32, comment at Fwd.cu:390-400)

The rationale comment for choosing PPT=1 (Fwd.cu:396-400): "CTA=64 (PPT=4) keeps a pix_out[PPT][CDIM] accumulator per thread, whose register footprint scales with 4*CDIM and spills to local memory at high channel counts (~10x slower forward at CDIM=128). PPT=1 makes register pressure scale with CDIM alone; it is parity at CDIM=3."

### Tile/pixel address computation (Fwd.cu:77-119)

```
linear_block_index = blockIdx.x + block_offset          // block_offset is for multi-GPU; 0 on a single device
image_id    = linear_block_index / (tile_width * tile_height)
tile_linear = linear_block_index % (tile_width * tile_height)
tile_x = tile_linear % tile_width;  tile_y = tile_linear / tile_width
tile_id = tile_y * tile_width + tile_x

thread_x = tid & TILE_MASK;  thread_y = tid >> TILE_SHIFT
out_x = tile_x * TILE_SIZE + thread_x
px    = (float)out_x + 0.5f                              // pixel-center correction (+0.5)
// p in [0, PPT):
out_y[p]  = tile_y * TILE_SIZE + thread_y + p * ROW_STRIDE
py[p]     = (float)out_y[p] + 0.5f
pix_id[p] = out_y[p] * image_width + out_x
```

Per-image pointer advance: `isect_offsets += image_id * grid_h * grid_w`, `render_colors += image_id * H * W * CDIM`, `render_alphas`/`last_ids += image_id * H * W`, `backgrounds += image_id * CDIM`, `masks += image_id * grid_h * grid_w` (Fwd.cu:92-103).

### Boundary and mask handling (Fwd.cu:121-159)

- OOB threads do not return; they only set their `done_mask` bits — because `__syncthreads_count` requires every thread in the block to arrive (comment at Fwd.cu:121-133).
  - `done_mask = (out_x >= image_width) ? ALL_DONE : 0`, then `done_mask |= (1u << p)` for every p with `out_y[p] >= image_height`.
- If `masks != nullptr && !masks[tile_id]` (a condition uniform across the whole tile): for every in-bounds pixel write `render_colors = backgrounds` (0 if absent), `render_alphas = 0`, and `last_ids = 0`, then **the whole block returns** (Fwd.cu:141-159). This is safe because the return is uniform and happens before the first barrier.

### Gaussian range (Fwd.cu:164-168)

```
range_start = isect_offsets[tile_id]
range_end   = (image_id == I-1 && tile_id == the last tile) ? n_isects : isect_offsets[tile_id + 1]
num_batches = ceil_div(range_end - range_start, BATCH_SIZE)
```

Note: at an image boundary `isect_offsets[tile_id + 1]` reads the first offset of the next image, which is correct because the offsets are cumulative across the whole [I, tile_h, tile_w].

### Shared memory layout (Fwd.cu:170-173)

The `extern __shared__` dynamic array is manually partitioned into three regions (a total of `CTA_SIZE * (4 + 12 + 12)` bytes, launcher Fwd.cu:346):

```
int32_t id_batch[BATCH_SIZE]           // the flatten_ids[idx] value (the flattened Gaussian index g)
vec3    xy_opacity_batch[BATCH_SIZE]   // {mean.x, mean.y, opacity} — glm vec3, 12-byte stride
vec3    conic_batch[BATCH_SIZE]        // {A, B, C}
```

Note: forward does not stage `colors` into shared memory; it reads them directly from global only at the moment a contribution is confirmed (Fwd.cu:255).

### Batch loop + alpha blending (Fwd.cu:179-272)

Initial state: `T[p] = 1.0f`, `cur_idx[p] = 0u`, `pix_out[p][k] = 0.f` (register arrays).

```
for b in [0, num_batches):                                  // #pragma unroll 1 (comment at Fwd.cu:190-194: avoids register pressure)
    // cooperative load: one Gaussian per thread
    batch_start = range_start + BATCH_SIZE * b
    idx = batch_start + tid
    if idx < range_end:
        g = flatten_ids[idx]
        id_batch[tid] = g
        xy_opacity_batch[tid] = {means2d[g].x, means2d[g].y, opacities[g]}
        conic_batch[tid] = conics[g]
    __syncthreads()          // replaced by __syncwarp() when CTA_SIZE <= 32 (Fwd.cu:212-219)

    batch_size = min(BATCH_SIZE, range_end - batch_start)
    for t in [0, batch_size) while done_mask != ALL_DONE:   // front-to-back
        conic = conic_batch[t];  xy_opac = xy_opacity_batch[t]
        opac = xy_opac.z;  dx = xy_opac.x - px
        for p in [0, PPT):                                   // unrolled
            if done_mask & (1u << p): continue
            dy = xy_opac.y - py[p]
            gw = eval_gaussian_weight(conic, dx, dy, opac)   // sigma/vis/alpha/valid
            if !gw.valid: continue
            alpha = gw.alpha
            next_T = T[p] * (1.0f - alpha)
            if next_T <= TRANSMITTANCE_THRESHOLD:            // 1e-4, early termination
                done_mask |= (1u << p); continue             // "exclusive": this Gaussian is not included
            g = id_batch[t]
            vis = alpha * T[p]                               // this Gaussian's contribution weight for the pixel
            for k in [0, CDIM): pix_out[p][k] += colors[g*CDIM + k] * vis
            cur_idx[p] = batch_start + t                     // absolute isect index in flatten_ids terms
            T[p] = next_T

    // break out of the batch loop once the whole tile is done (barrier and count in one)
    if __syncthreads_count(done_mask == ALL_DONE) >= BATCH_SIZE: break   // Fwd.cu:268
```

Important details:
- **The T ≤ 1e-4 early termination is exclusive**: the Gaussian that pushed it past the threshold is itself not folded into the color, and neither `T` nor `cur_idx` is updated (Fwd.cu:246-251).
- Since it starts at `T = 1`, the first Gaussian would need `alpha ≥ 0.9999` to hit the threshold, which the `MAX_ALPHA = 0.99` clamp makes impossible → `cur_idx` can never be polluted on a pixel with no contribution.
- `cur_idx` (→ `last_ids`) is not a Gaussian id but **the absolute index into the sorted intersection list** (`batch_start + t`). Its initial value is 0.

### Final writes (Fwd.cu:274-296)

For in-bounds pixels:

```
render_alphas[pix_id[p]] = 1.0f - T[p]
render_colors[pix_id[p]*CDIM + k] = backgrounds == nullptr
    ? pix_out[p][k]
    : pix_out[p][k] + T[p] * backgrounds[k]
last_ids[pix_id[p]] = (int32_t)cur_idx[p]
```

### Rationale for keeping transmittance in float (original comments)

- Fwd.cu:176-178: "transmittance is gonna be used in the backward pass which requires a high numerical precision so we use double for it. However double make bwd 1.5x slower so we stick with float for now." (The first sentence of the comment is a historical leftover; the actual code is `float T[PPT]`.)
- Fwd.cu:281-285: "Here T is the transmittance AFTER the last gaussian in this pixel. We (should) store double precision as T would be used in backward pass and it can be very small and causing large diff in gradients with float32. However, double precision makes the backward pass 1.5x slower so we stick with float for now."
- Numerical safeguards back up keeping float: (a) `TRANSMITTANCE_THRESHOLD = 1e-4 = (1-MAX_ALPHA)^2` terminates before T becomes small enough to break float precision (Common.h:100-106), and (b) backward's `1/(1-alpha)` has the lower bound `MIN_ONE_MINUS_ALPHA = 1e-6` (Common.h:112-114).
- **Implication for Metal**: Apple GPUs have no device-side double, so keeping float is also the only option available. Thanks to the threshold design above, the upstream original concludes that float is sufficient.

## Formulas in detail — backward

Original: `RasterizeToPixels3DGSSerialBatchBwd.cu:41-320`, template `<CDIM, scalar_t=float>`. Uses `cooperative_groups`.

### The render equation and its gradients (derivation basis)

What forward computes:

```
render_color[k] = Σ_i c_i[k] · α_i · T_i  +  T_final · bg[k],   T_i = Π_{j<i} (1 - α_j)
render_alpha    = 1 - T_final
α_i = min(MAX_ALPHA, opac_i · exp(-σ_i)),  σ_i = 0.5(A dx² + C dy²) + B dx dy
```

Backpropagation (exactly in the form the code takes):

```
∂L/∂c_i[k]  = α_i T_i · v_render_c[k]                                    → v_rgb_local[k] = fac * v_render_c[k],  fac = alpha * T
∂L/∂α_i     = Σ_k (c_i[k]·T_i − S_i[k]/(1−α_i)) · v_render_c[k]
              + T_final/(1−α_i) · v_render_a
              − T_final/(1−α_i) · Σ_k bg[k]·v_render_c[k]                 (S_i[k] = Σ_{j>i} c_j[k] α_j T_j = buffer[k])
∂α/∂σ       = −opac·vis            (when the clamp is inactive)
∂α/∂opac    = vis
∂σ/∂conic   = (0.5 dx², dx·dy, 0.5 dy²)
∂σ/∂mean2d  = (A dx + B dy, B dx + C dy)                                  (delta = mean − pixel, so the sign is unchanged)
```

### Block/pixel addressing (Bwd.cu:77-117)

- The thread configuration differs from forward: `threads = {tile_size, tile_size, 1}`, a **2D threadgroup**, exactly one thread per pixel (launcher Bwd.cu:361).
- Decompose `linear_block_index` into (image, tile_y, tile_x) (Bwd.cu:79-83):
  ```
  image_id = lbi / (tile_width * tile_height)
  tile_iy  = (lbi / tile_width) % tile_height     // block_index.y
  tile_ix  = lbi % tile_width                     // block_index.z
  tile_id  = tile_iy * tile_width + tile_ix
  i = tile_iy * tile_size + thread_index.y        // pixel row
  j = tile_ix * tile_size + thread_index.x        // pixel column
  px = j + 0.5f;  py = i + 0.5f
  pix_id = min(i * image_width + j, image_width * image_height - 1)   // OOB threads clamp to the last pixel (Bwd.cu:114)
  inside = (i < image_height && j < image_width)
  ```
- A masked tile returns immediately (Bwd.cu:106-109) — unlike forward it writes nothing (this assumes the gradient buffers were zero-initialized beforehand).
- OOB threads (`!inside`) do not return; they are kept around as cooperative-load workers (comment at Bwd.cu:116-117: "keep not rasterizing threads around for reading data").

### Shared memory layout (Bwd.cu:128-132)

Forward's three regions + **rgbs_batch** (a total of `tile_size² * (4 + 12 + 12 + 4*CDIM)` bytes, launcher Bwd.cu:384-385):

```
int32_t id_batch[block_size]
vec3    xy_opacity_batch[block_size]
vec3    conic_batch[block_size]
float   rgbs_batch[block_size * CDIM]     // backward caches the colors in shared memory too
```

### Per-pixel initial state (Bwd.cu:134-149)

```
T_final   = 1.0f - render_alphas[pix_id]        // restores forward's final T
T         = T_final                              // the starting value of the back-to-front walk
buffer[k] = 0.f                                  // accumulates "contributions from behind (farther than) the current Gaussian"
bin_final = inside ? last_ids[pix_id] : 0        // the isect index of the last contributor to this pixel
v_render_c[k] = v_render_colors[pix_id*CDIM + k]
v_render_a    = v_render_alphas[pix_id]
```

Per-warp precomputation (Bwd.cu:153-155):

```
warp = cg::tiled_partition<32>(block)
warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>())   // max within the warp
```

### The reverse batch loop (Bwd.cu:156-319)

```
for b in [0, num_batches):
    block.sync()                                   // (barrier 1) guarantees the previous batch's shared reads are done
    batch_end  = range_end - 1 - block_size * b    // int32 (can be negative, comment at Bwd.cu:165)
    batch_size = min(block_size, batch_end + 1 - range_start)
    idx = batch_end - tr                           // tr = thread_rank; slot 0 = the back-most one in the batch
    if idx >= range_start:
        g = flatten_ids[idx]
        id_batch[tr] = g
        xy_opacity_batch[tr] = {means2d[g].x, means2d[g].y, opacities[g]}
        conic_batch[tr] = conics[g]
        for k: rgbs_batch[tr*CDIM + k] = colors[g*CDIM + k]
    block.sync()                                   // (barrier 2) load complete

    // ascending t = descending absolute index (batch_end - t) = back-to-front progress
    for t in [max(0, batch_end - warp_bin_final), batch_size):    // everything past the warp's last contributor is skipped wholesale (Bwd.cu:187)
        valid = inside
        if batch_end - t > bin_final: valid = 0    // invalid if behind this pixel's last contributor (Bwd.cu:190-193)
        if valid:
            conic = conic_batch[t];  xy_opac = xy_opacity_batch[t]
            opac = xy_opac.z;  delta = {xy_opac.x - px, xy_opac.y - py}
            gw = eval_gaussian_weight(conic, delta.x, delta.y, opac)
            vis = gw.vis;  alpha = gw.alpha
            if !gw.valid: valid = false
        if !warp.any(valid): continue              // skip when the whole warp is invalid (Bwd.cu:216-219)

        // lane-local gradients; invalid lanes stay 0 (Bwd.cu:220-224)
        v_rgb_local[CDIM] = {0}; v_conic_local = {0,0,0}
        v_xy_local = {0,0}; v_xy_abs_local = {0,0}; v_opacity_local = 0
        if valid:
            ra = 1.0f / fmaxf(MIN_ONE_MINUS_ALPHA, 1.0f - alpha)   // the 1e-6 lower bound (Bwd.cu:229)
            T *= ra                                 // recompute T: restores T at the moment this Gaussian was blended
            fac = alpha * T                         // == forward's vis weight
            for k: v_rgb_local[k] = fac * v_render_c[k]

            v_alpha = Σ_k (rgbs_batch[t*CDIM+k] * T - buffer[k] * ra) * v_render_c[k]   // Bwd.cu:240-244
            v_alpha += T_final * ra * v_render_a                                        // Bwd.cu:246
            if backgrounds != nullptr:
                v_alpha += -T_final * ra * Σ_k backgrounds[k] * v_render_c[k]           // Bwd.cu:248-257

            if opac * vis <= MAX_ALPHA:             // when the clamp is active (opac*vis > 0.99) the σ/opac gradients are cut off (Bwd.cu:259)
                v_sigma = -opac * vis * v_alpha
                v_conic_local = {0.5f * v_sigma * delta.x * delta.x,
                                 v_sigma * delta.x * delta.y,
                                 0.5f * v_sigma * delta.y * delta.y}
                v_xy_local = {v_sigma * (conic.x * delta.x + conic.y * delta.y),
                              v_sigma * (conic.y * delta.x + conic.z * delta.y)}
                if v_means2d_abs != nullptr:        // absgrad (Bwd.cu:269-272)
                    v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)}
                v_opacity_local = vis * v_alpha

            for k: buffer[k] += rgbs_batch[t*CDIM+k] * fac        // update the accumulation of contributions from behind (Bwd.cu:276-280)

        // warp reduction: sums the gradients of 32 pixels for the same Gaussian t (Bwd.cu:282-289)
        warpSum<CDIM>(v_rgb_local, warp)            // cg::reduce(plus) per channel
        warpSum(v_conic_local, warp)                // per vec3 component
        warpSum(v_xy_local, warp)                   // per vec2 component
        if v_means2d_abs != nullptr: warpSum(v_xy_abs_local, warp)
        warpSum(v_opacity_local, warp)

        if warp.thread_rank() == 0:                 // only lane 0 does the atomic scatter (Bwd.cu:290-317)
            g = id_batch[t]
            for k: atomicAdd_system(v_colors + g*CDIM + k, v_rgb_local[k])
            atomicAdd_system(&v_conics[g].x/.y/.z, v_conic_local.x/y/z)     // 3 adds
            atomicAdd_system(&v_means2d[g].x/.y,   v_xy_local.x/y)          // 2 adds
            if v_means2d_abs != nullptr:
                atomicAdd_system(&v_means2d_abs[g].x/.y, v_xy_abs_local.x/y) // 2 adds
            atomicAdd_system(v_opacities + g, v_opacity_local)               // 1 add
```

Detailed notes:
- **Recomputing T**: forward's T is not stored; it is recovered from `T_final` by `T *= 1/(1-α_i)`. The `MIN_ONE_MINUS_ALPHA = 1e-6` lower bound prevents the gradient from blowing up as α→1. The division order is the exact reverse of forward's multiplication order, so the error stays controlled even in float (see the rationale for keeping float above).
- **`buffer[k]`**: while walking back-to-front, the "sum of `c·α·T` over the Gaussians already processed (= the ones behind)". The second term of `v_alpha`, `−buffer[k]·ra·v_render_c[k]`, corresponds to the `∂T_j/∂α_i` chain.
- **atomicAdd sites**: `CDIM + 3 + 2 (+2) + 1` float atomicAdds from one warp lane per Gaussian. `atomicAdd_system` exists because of this fork's multi-GPU support (system scope); on single-device Metal, device scope is sufficient.
- **absgrad**: when `v_means2d_abs` is not nullptr, `|v_xy|` is accumulated separately — an extra output for the densification heuristic (AbsGS).
- The launcher skips the kernel when `n_isects == 0` (Bwd.cu:365-369).
- The `warp.any(valid)` skip and the `max(0, batch_end - warp_bin_final)` start point are pure performance optimizations and do not affect the results.

## Thread and memory configuration

| Item | Forward | Backward |
|---|---|---|
| grid | `{I * tile_h * tile_w, 1, 1}` — one block per tile | same |
| threads | `{CTA_SIZE, 1, 1}` 1D. tile 16 → 256, tile 4 → 16 | `{tile_size, tile_size, 1}` 2D. tile 16 → 256 |
| PPT | 1 (both instantiated combinations) | 1 (structurally, pixel = thread) |
| shared/threadgroup | `CTA * (4+12+12)` = **7,168 B** at 16/256 | `ts² * (4+12+12+4·CDIM)`; **10,240 B** at 16² with CDIM=3 |
| Barriers | 2 per batch: `__syncthreads` after the load (`__syncwarp` when CTA≤32), `__syncthreads_count` at the end of the batch | 2 per batch: `block.sync()` at the top of the loop, `block.sync()` after the load |
| warp ops | none | `cg::reduce(greater)` once (up front), `warp.any`, `cg::reduce(plus)` many times |
| atomic | none | float atomicAdd from lane 0 (CDIM+6 or CDIM+8 per Gaussian) |

Because the backward shared memory scales with CDIM, the `tile_size=4` path exists to avoid the shared-memory limit at CDIM ≥ 32 (comment at Fwd.cu:390-395).

## MSL porting design

### Premises (confirmed by measurement on device)

- simdgroup width 32 maps 1:1 to a CUDA warp (measured on M4 and A17 Pro). Assert `threadExecutionWidth == 32` when creating the pipeline.
- device `atomic_float` is natively supported → no CAS fallback is needed. `memory_order_relaxed` is sufficient (pure accumulation, read only after the kernel completes).
- A threadgroup memory budget of 32KB (32,768 B).

### Batch-size sizing (the 32KB budget)

If vec3 is kept as MSL `packed_float3` (12 B):

| Kernel | Formula | TILE=16 (batch 256) | Condition to fit in 32KB |
|---|---|---|---|
| Forward | `256·(4+12+12)` + ~32 B of ballot scratch | **7,200 B** | always has headroom |
| Backward | `256·(28 + 4·CDIM)` | CDIM=3: **10,240 B** | `CDIM ≤ 25` (`28+4C ≤ 128`) |

- With `float3` (16 B alignment) it is 9,216 B for fwd and 12,288 B for bwd at CDIM=3 — still with headroom. If consistency comes first, the default is `packed_float3`, which matches the CUDA stride.
- CDIM > 25 (e.g. 32): either add a tile=4/batch=16 variant as gsplat does (`16·(28+4C)` → C ≤ 505), or decouple the batch size from the threadgroup size and extend the load into a strided loop. Not needed in Phase 1 (CDIM=3 RGB).
- On Apple GPUs threadgroup memory affects occupancy, so at 7.2KB for fwd and 10.2KB for bwd multiple tiles can be resident.

### CDIM and option handling

- `CDIM` is the size of local arrays (`pix_out[CDIM]`, `buffer[CDIM]`, `v_rgb_local[CDIM]`), so it must be a **compile-time constant**. MSL function constants cannot be used for array sizes, so gsplat's template dispatch is mapped onto pipeline variants driven by `MTLCompileOptions.preprocessorMacros` (`-D CDIM=3` and so on). Phase 1 has a single CDIM=3 variant.
- Option buffers (`backgrounds`, `masks`, `v_means2d_abs`) cannot be null-checked in MSL, so branch on **function constants**:
  ```msl
  constant bool kHasBackground [[function_constant(0)]];
  constant bool kHasMasks      [[function_constant(1)]];
  constant bool kHasAbsGrad    [[function_constant(2)]];
  ```

### Proposed forward kernel signature

```msl
// Rasterize3DGSFwd.metal — CDIM, TILE_SIZE(16), and CTA_SIZE(256) are preprocessor macros
struct RasterizeParams {
    uint  n_isects;
    uint  image_width, image_height;
    uint  num_images;               // I
    uint  tile_width, tile_height;  // grid_w, grid_h
};

kernel void rasterize_to_pixels_3dgs_fwd(
    constant RasterizeParams&    p             [[buffer(0)]],
    device const float2*         means2d       [[buffer(1)]],   // [I*N, 2] flattened
    device const packed_float3*  conics        [[buffer(2)]],
    device const float*          colors        [[buffer(3)]],   // [I*N, CDIM]
    device const float*          opacities     [[buffer(4)]],
    device const float*          backgrounds   [[buffer(5)]],   // kHasBackground
    device const bool*           masks         [[buffer(6)]],   // kHasMasks
    device const int*            isect_offsets [[buffer(7)]],   // [I, tile_h, tile_w]
    device const int*            flatten_ids   [[buffer(8)]],   // [n_isects]
    device float*                render_colors [[buffer(9)]],
    device float*                render_alphas [[buffer(10)]],
    device int*                  last_ids      [[buffer(11)]],
    uint  block_id  [[threadgroup_position_in_grid]],           // = linear_block_index (block_offset is unnecessary)
    uint  tid       [[thread_index_in_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_id   [[simdgroup_index_in_threadgroup]]);
```

- Dispatch: `dispatchThreadgroups(MTLSize(I*tile_h*tile_w, 1, 1), threadsPerThreadgroup: (256,1,1))`.
- Threadgroup memory sizes are compile-time, so declare them statically in the kernel body (CUDA's manual `extern __shared__` partitioning is unnecessary):
  ```msl
  threadgroup int           id_batch[CTA_SIZE];
  threadgroup packed_float3 xy_opacity_batch[CTA_SIZE];
  threadgroup packed_float3 conic_batch[CTA_SIZE];
  threadgroup uchar         sg_done[CTA_SIZE / 32];   // replaces __syncthreads_count
  ```

**Barrier placement (forward)** — keeps the same count as CUDA (2 per batch):

1. Immediately after the cooperative load: `threadgroup_barrier(mem_flags::mem_threadgroup)` (the `__syncthreads` of CUDA Fwd.cu:212-219; the TILE=4/CTA=16 variant uses `simdgroup_barrier(mem_flags::mem_threadgroup)`).
2. Replacement for the end-of-batch `__syncthreads_count(done) >= BATCH_SIZE` (Fwd.cu:268) — a simd ballot plus a threadgroup tally:
   ```msl
   bool lane_done = (done_mask == ALL_DONE);
   if (simd_lane == 0) sg_done[simd_id] = (uchar)simd_all(lane_done);
   threadgroup_barrier(mem_flags::mem_threadgroup);   // this barrier also re-synchronizes before the next batch load
   bool all_done = true;
   for (uint s = 0; s < CTA_SIZE / 32; ++s) all_done = all_done && sg_done[s];
   if (all_done) break;
   ```
   Because the next iteration's shared writes come after this barrier, the race against the previous batch's shared reads (the processing loop) is blocked as well — the same role `__syncthreads_count` plays as a barrier in CUDA.
- `__expf(-sigma)` → `metal::exp(-sigma)` (fast-math by default). Compare against the golden fixture on a tolerance basis, not in ULPs.
- The remaining logic (the OOB done_mask, writing the background for a masked tile then returning uniformly, the exclusive early termination, and writing last_ids) is a 1:1 direct port.

### Proposed backward kernel signature

```msl
// Rasterize3DGSBwd.metal — CDIM is a preprocessor macro, TILE_SIZE fixed at 16
kernel void rasterize_to_pixels_3dgs_bwd(
    constant RasterizeParams&    p               [[buffer(0)]],
    device const float2*         means2d         [[buffer(1)]],
    device const packed_float3*  conics          [[buffer(2)]],
    device const float*          colors          [[buffer(3)]],
    device const float*          opacities       [[buffer(4)]],
    device const float*          backgrounds     [[buffer(5)]],   // kHasBackground
    device const bool*           masks           [[buffer(6)]],   // kHasMasks
    device const int*            tile_offsets    [[buffer(7)]],
    device const int*            flatten_ids     [[buffer(8)]],
    device const float*          render_alphas   [[buffer(9)]],
    device const int*            last_ids        [[buffer(10)]],
    device const float*          v_render_colors [[buffer(11)]],
    device const float*          v_render_alphas [[buffer(12)]],
    device atomic_float*         v_means2d_abs   [[buffer(13)]],  // kHasAbsGrad, [.., 2] flattened
    device atomic_float*         v_means2d       [[buffer(14)]],  // [.., 2] flattened
    device atomic_float*         v_conics        [[buffer(15)]],  // [.., 3] flattened
    device atomic_float*         v_colors        [[buffer(16)]],  // [.., CDIM] flattened
    device atomic_float*         v_opacities     [[buffer(17)]],
    uint   block_id   [[threadgroup_position_in_grid]],
    uint2  local_xy   [[thread_position_in_threadgroup]],          // (x=j offset, y=i offset)
    uint   tr         [[thread_index_in_threadgroup]],             // = local_xy.y*16 + local_xy.x
    uint   simd_lane  [[thread_index_in_simdgroup]]);
```

- Dispatch: `dispatchThreadgroups(MTLSize(I*tile_h*tile_w, 1, 1), threadsPerThreadgroup: (16,16,1))`.
- The gradient output buffers are zero-initialized on the CPU side before the kernel (the same premise as gsplat). Bind the gradient buffers as flattened float arrays rather than vec2/vec3 structures so that `atomic_float` can be accessed element-wise.
- The 2D threadgroup linearizes as `x + y*width`, so on a 16-wide tile one simdgroup = two consecutive rows — the same layout as the CUDA warp. The lane composition of `warp_bin_final` is preserved 1:1.

**CUDA → MSL mapping table (backward)**

| CUDA (Bwd.cu line) | MSL |
|---|---|
| `block.sync()` (:159, :184) | `threadgroup_barrier(mem_flags::mem_threadgroup)` — 2 per batch, same positions |
| `cg::tiled_partition<32>(block)` (:154) | implicit simdgroup (assert width 32) |
| `cg::reduce(warp, bin_final, greater<int>)` (:155) | `simd_max(bin_final)` |
| `warp.any(valid)` (:216) | `simd_any(valid)` |
| `warpSum(float/vec2/vec3)` (Utils.cuh:154-191) | per-component `simd_sum(x)` |
| `warp.thread_rank() == 0` (:290) | `simd_lane == 0` (or `simd_is_first()`) |
| `atomicAdd_system(ptr, v)` (:297-316) | `atomic_fetch_add_explicit(ptr, v, memory_order_relaxed)` — device scope is sufficient |

**atomic sites (lane 0 only, per Gaussian)**: `v_colors` CDIM times, `v_conics` 3 times, `v_means2d` 2 times, `v_means2d_abs` 2 times (optional), and `v_opacities` once. Because this sits inside a divergent loop with no barriers, the `continue` (the warp.any skip) can be ported directly — the key condition that permits the direct port is that there is no threadgroup barrier inside the processing loop.

**Barrier placement (backward)**: once at the top of the batch loop (guaranteeing the previous batch's shared reads are complete) + once after the cooperative load. Barriers inside the processing loop are forbidden (divergent control flow).

### Simplifications related to multi-image / multi-device

- `block_offset` (multi-GPU chunking) and the system scope of `atomicAdd_system` are removed on single-device Metal. `linear_block_index = threadgroup_position_in_grid.x`.
- The `packed` flag is in fact unused in the kernel body (addressing is already indirected through flatten_ids) — it can be omitted from the MSL signature.
- If `n_isects == 0`, skip the backward dispatch (handled at the encoder level, the same as launcher Bwd.cu:365-369).

## Open questions

1. **exp precision**: the error characteristics of CUDA `__expf` (fast approx) and MSL `metal::exp`/`fast::exp` differ. At what level to set the golden fixture tolerances (fwd colors and bwd gradients separately) — in bwd especially, `vis` multiplies into v_sigma so the error can be amplified.
2. **Non-determinism of atomic float accumulation**: the atomicAdd order varies from run to run, so the v_* results are not bit-reproducible. Verification against the CPU reference has to be relative-error based, and the threshold tolerance needs to be set per fixture.
3. **Threadgroup layout choice**: `packed_float3` (12 B, matching the CUDA stride) vs. `float3` (16 B alignment) vs. SoA (three float arrays) — decide after measuring bank conflicts and load efficiency on Apple GPUs. No effect on consistency.
4. **Value of a PPT>1 variant**: the CUDA side reverted to PPT=1 because of register spills (Fwd.cu:396-400), but PPT=4 (CTA 64) may be advantageous given the Apple GPU register file and occupancy characteristics. Phase 1 ports PPT=1 directly and judges later by benchmark.
5. **The tile=4 / CDIM≥32 path**: not needed in Phase 1 (CDIM=3). When CDIM is extended, whether to add a tile=4 variant or to go with a strided load that decouples the batch size from the threadgroup size is undecided.
6. **bwd's thread layout differing from fwd (1D vs 2D)**: whether to keep the two kernels' thread configurations different as in the upstream original, or to unify both to 1D (or both to 2D) in MSL. Unifying does not change the numerical results, but if the simdgroup-to-row mapping changes, the atomic contention pattern can change.
7. **Documenting the dtype and meaning of last_ids**: `last_ids` is not a Gaussian id but the absolute index into the sorted isect list (int32). Whether to support large scenes where n_isects exceeds the int32 range is decided together with the upstream pipeline (the intersect spec).
8. **Derivation notice**: this spec and the MSL implementation that follows are derivatives of gsplat (Apache-2.0) — the procedure for including the original copyright notices (UC Regents, Nerfstudio Team, NVIDIA) in NOTICE and in file headers still needs to be decided.

## Verification notes (adversarial verifier)

The originals (`RasterizeToPixels3DGSSerialBatchFwd.cu`, `...Bwd.cu`, `RasterizeToPixels3DGSDevice.cuh`, `Common.h`, `Utils.cuh`, `Config.h`, commit 77ab983) were re-read independently and every formula, sign, index, clamp, constant, and line citation in the body was checked against them. **No formula, sign, index, or clamp errors were found.** The following were confirmed individually: `sigma`/`vis`/`alpha`/`valid` (Device.cuh:47-53), the exclusive early termination `next_T <= TRANSMITTANCE_THRESHOLD` (Fwd.cu:246-251), `cur_idx = batch_start + t` (Fwd.cu:261), the final write expressions (Fwd.cu:286-293), `ra`/`T *= ra`/`fac` (Bwd.cu:229-232), the signs of the three `v_alpha` terms (Bwd.cu:240-257), the clamp gate `opac * vis <= MAX_ALPHA` (Bwd.cu:259), `v_sigma`/`v_conic_local`/`v_xy_local`/`v_opacity_local` (Bwd.cu:261-273), the position of the `buffer` update and the use of `fac` (Bwd.cu:276-280), and the warpSum→atomic order (Bwd.cu:282-317). What follows are not corrections but **supplements** (making omissions explicit and clarifying to prevent misreading).

### Supplement 1 — the missing per-image pointer advance in backward (required for consistency at I > 1)

The body states only forward's pointer advance (Fwd.cu:92-103). Backward has the same advance (Bwd.cu:90-102), and omitting it for multiple images (I>1) makes the results wrong:

```
tile_offsets    += image_id * tile_height * tile_width      // Bwd.cu:90
render_alphas   += image_id * image_height * image_width    // Bwd.cu:91
last_ids        += image_id * image_height * image_width    // Bwd.cu:92
v_render_colors += image_id * image_height * image_width * CDIM   // Bwd.cu:93
v_render_alphas += image_id * image_height * image_width    // Bwd.cu:94
backgrounds     += image_id * CDIM                (when not nullptr, Bwd.cu:95-98)
masks           += image_id * tile_height * tile_width  (when not nullptr, Bwd.cu:99-102)
```

By contrast, **the gradient output buffers (`v_means2d`, `v_conics`, `v_colors`, `v_opacities`, `v_means2d_abs`) and the Gaussian inputs (`means2d`, `conics`, `colors`, `opacities`) do not advance per image** — because `g = flatten_ids[idx]` is already a flattened index over all of [I*N] (or nnz) (Bwd.cu:171, 292; the same at Fwd.cu:202). Every `pix_id` access in the body's pseudocode must be read on the premise that it uses the pointers after this advance.

### Supplement 2 — the source lines for backward's range_start/range_end

The backward section of the body implicitly reuses forward's range computation; the source in the original is Bwd.cu:122-126, and the logic is identical to forward (Fwd.cu:164-168): `range_start = tile_offsets[tile_id]`, `range_end = (image_id == I-1 && tile_id == tile_width*tile_height - 1) ? n_isects : tile_offsets[tile_id + 1]`, `num_batches = ceil_div(range_end - range_start, block_size)`. The only difference from forward (compile-time `BATCH_SIZE`) is that `block_size = block.size()` is a runtime value in backward (`tile_size²`, Bwd.cu:125).

### Supplement 3 — clarifying the unit of the atomicAdd count (to prevent misreading)

The body's "CDIM+6 (or +8) per Gaussian" is **per simdgroup (warp)**. tile 16 → 256 threads = 8 simdgroups, and `warp.thread_rank() == 0` (Bwd.cu:290) means **lane 0 of each warp**, so the total number of atomicAdds one tile performs for one Gaussian is at most **8 × (CDIM+6 or CDIM+8)** (warps where `warp.any(valid)` is false are excluded by the continue, Bwd.cu:216-219). In Metal too, each simdgroup independently runs `simd_sum` and then a lane 0 scatter — "optimizing" this into a single scatter per threadgroup would not diverge from the original (it is mathematically identical), but it would put the barrier needed for the threadgroup reduction inside a divergent loop, which makes it **a forbidden pattern instead**. Keep the per-warp structure exactly as it is.

### Supplement 4 — `__launch_bounds__(CTA_SIZE)` (Fwd.cu:42)

The forward kernel carries a `__launch_bounds__(CTA_SIZE)` register hint (performance-related, no effect on numerics). The MSL counterpart is to specify `maxTotalThreadsPerThreadgroup = 256` when creating the pipeline (or `[[max_total_threads_per_threadgroup(256)]]`) — this makes the compiler allocate registers on the premise that 256 threads are resident.

### Supplement 5 — the simdgroup premise of the tile=4 backward variant (attached to open question 5)

Instantiating backward at tile=4 gives a threadgroup of 4×4=16 threads, smaller than the simdgroup width of 32. The CUDA original also becomes an edge case here, applying `cg::tiled_partition<32>` (Bwd.cu:154) to a 16-thread block. In Metal a 16-thread threadgroup forms one partial simdgroup and `simd_sum`/`simd_max`/`simd_any` tally only the active lanes, so a direct port stays consistent; but the body's "assert `threadExecutionWidth == 32` when creating the pipeline" has to be relaxed for this variant, to be judged against the threadgroup size (16) instead. No effect on Phase 1 (tile 16, CDIM=3).

### Minor line-citation refinements

- The exact location of the warpSum overloads: `Utils.cuh:153-191` (float* DIM :153-161, float :163-167, vec4 :169-176, vec3 :178-184, vec2 :186-191). The body's ":152-200" and ":154-191" are adequate in practice, but the exact ranges are as above.
- The exact starting line of the transmittance comment: Fwd.cu:175-178 (the body's 176-178 is valid with respect to the quoted sentences).
- `GSPLAT_NUM_CHANNELS`: Config.h:71-73 (`#ifndef` at :71, the list at :72).

**Conclusion: verification passes — no formula, constant, or structural errors. Read together with the five supplements above, this document is sufficient as the sole reference for the Metal implementation.**

### Second independent re-verification (adversarial verifier, second pass)

The five original files (`RasterizeToPixels3DGSSerialBatchFwd.cu`, `...Bwd.cu`, `RasterizeToPixels3DGSDevice.cuh`, `include/Common.h`, `include/Utils.cuh`, `csrc/Config.h`) were read again from the beginning and both the body and the first-pass verification notes above were re-checked against them. Results:

- **The CUDA extraction (formulas, signs, indices, clamps, constants, and line citations): 0 errors.** Everything the first-pass notes confirmed, plus the following independently confirmed:
  - `vec3 = glm::vec<3, float>` (Common.h:64-65) — the type basis for the body's "12-byte stride" claim. Consistent with the fwd shmem formula `CTA_SIZE*(4+12+12)` (Fwd.cu:346) and the bwd shmem formula `ts²*(4+12+12+4·CDIM)` (Bwd.cu:384-385).
  - The `packed`/`N` parameters are unused anywhere in the fwd (Fwd.cu:45,43) and bwd (Bwd.cu:44,46) kernel bodies — this confirms the body's claim that they "can be omitted from the MSL signature".
  - The three `v_alpha` terms re-derived from the render equation: `∂render_color[k]/∂α_i = c_i[k]·T_i − S_i[k]/(1−α_i)`, `∂render_alpha/∂α_i = T_final/(1−α_i)`, `∂(T_final·bg[k])/∂α_i = −T_final·bg[k]/(1−α_i)` — the terms and signs match the code form at Bwd.cu:243, :246, and :256 exactly.
  - The result-invariance of the `warp.any(valid)` continue (Bwd.cu:216-219) and of the `t = max(0, batch_end - warp_bin_final)` start point (Bwd.cu:187): in both cases the code path confirms that no `T`/`buffer` update and no non-zero gradient contribution occurs on the lane/warp in question — the body's "pure performance optimization" claim is justified.
  - The forward early-termination invariant re-confirmed: at `T=1`, `next_T = 1−α ≤ 1e-4` would require `α ≥ 0.9999`, which the `MAX_ALPHA = 0.99` clamp makes impossible → the `cur_idx` initial value of 0 surviving uncorrupted as `last_ids = 0` is a valid encoding of "no contributing Gaussian" (Fwd.cu:186, 246-251, 293).
  - Supplements 1–5 of the first-pass notes and the line-citation refinements were all re-confirmed to match the original (Bwd.cu:90-102, :122-126, :290; Fwd.cu:42; Utils.cuh:153-191; Config.h:71-73).

- **One correction (a defect internal to the body's MSL design section, unrelated to the CUDA extraction)**: in the body's forward snippet, `threadgroup uchar sg_done[CTA_SIZE / 32];` gives `16/32 = 0` in the TILE=4/CTA=16 variant — **a zero-size array, which does not compile / is UB**. Correction: either size the array as `max(CTA_SIZE / 32, 1)` (`((CTA_SIZE + 31) / 32)` in the preprocessor), or — since for CTA ≤ 32 variants the whole threadgroup is a single (partial) simdgroup — drop the `sg_done` tally entirely and replace it with `if (simd_all(lane_done)) break;` + `simdgroup_barrier`, which is the correct approach (equivalent to CUDA's `__syncthreads_count` working at CTA=16 as well). No effect on Phase 1 (TILE=16/CTA=256) — 256/32 = 8 is fine.

**Second-pass conclusion: the CUDA extraction passes the consistency verification. The one correction (the sg_done size above) applies only to the tile=4 variant and does not affect the Phase 1 implementation path.**
