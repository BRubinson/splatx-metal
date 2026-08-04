# 03 — Tile intersection (intersect_tile) + offset encoding (intersect_offset)

## Upstream files and commit (77ab983)

- gsplat commit: `77ab983ffe43420b2131669cb35776b883ca4c3c` (Apache-2.0, a derivation notice is required)
- Kernel bodies: `research/gsplat/gsplat/cuda/csrc/IntersectTile.cu`
  - `row_key_column_range` (59–77), `accutile_ellipse_intersection` (83–94), `accutile_process_tiles` (96–207)
  - `intersect_tile_kernel` (213–464), `launch_intersect_tile_kernel` (466–569)
  - the multi-GPU path `intersect_tile_kernels_privateuseone` (686–923) — where cub::DeviceScan is used
  - `intersect_offset_kernel` (925–988), `launch_intersect_offset_kernel` (990–1026)
  - `radix_sort_double_buffer` (1078–1121), `segmented_radix_sort_double_buffer` (1125–1176)
- Host orchestration (2-pass): `research/gsplat/gsplat/cuda/csrc/Intersect.cpp` — `intersect_tile` (170–329), `intersect_offset` (547–561)
- Bit-width helper: `research/gsplat/gsplat/cuda/csrc/MathUtils.h:23–33` (`bits_for_count`)
- Constants: `research/gsplat/gsplat/cuda/include/Common.h:97–99`
  - `ALPHA_THRESHOLD = (1.f / 255.f)` (Common.h:97)
  - `GAUSSIAN_EXTEND = 3.33f` (Common.h:99)
- Python API: `research/gsplat/gsplat/cuda/_wrapper.py` — `isect_tiles` (1036–), `isect_offset_encode` (1168–1187)

This operation **has no backward**. Intersection enumeration, sorting, and offset encoding are non-differentiable, and the Python side wraps them in `@torch.no_grad()` as well (around _wrapper.py:1035). Everything under "Formula detail" below is therefore forward.

## Algorithm overview

For each projected Gaussian (2D mean `means2d`, conic `Σ⁻¹`, per-axis radii `radii`, depth `depths`), enumerate the screen tiles it covers, build a 64-bit key that can be sorted in render order (image → tile → ascending depth), and then build a per-tile start-offset table. The full pipeline (single device, Intersect.cpp:170–329):

```
Pass 1 (count)  : intersect_tile_kernel(cum_tiles_per_gauss = nullptr)
                  → tiles_per_gauss[e] : int32, tiles covered by each element     (Intersect.cpp:238–256)
Prefix sum      : cum_tiles_per_gauss = cumsum(tiles_per_gauss, int64)            (Intersect.cpp:258)
                  n_isects = cum_tiles_per_gauss[-1]  ← host readback (sync point) (Intersect.cpp:259)
Alloc           : isect_ids[n_isects] int64, flatten_ids[n_isects] int32          (Intersect.cpp:273–274)
Pass 2 (emit)   : intersect_tile_kernel(cum_tiles_per_gauss passed in)
                  → each element writes keys into its range [cum[e-1], cum[e])    (Intersect.cpp:277–296)
Sort            : radix sort pairs (isect_ids keys, flatten_ids values)           (Intersect.cpp:299–324)
Offset encode   : intersect_offset_kernel → offsets[I, tile_h, tile_w]            (Intersect.cpp:547–561)
```

count and emit are the **same kernel** (`intersect_tile_kernel`); `first_pass` branches on whether `cum_tiles_per_gauss == nullptr` (IntersectTile.cu:248). The tile enumeration formula is completely identical in both passes, so emit can never write more than count reported.

There are two paths for the tile decision:
- **AccuTile path** (`conics != nullptr && opacities != nullptr`, IntersectTile.cu:288–373): SNUGBOX + AccuTile from SpeedySplat (arXiv:2412.00578). It computes the actual intersections between the tight, opacity-aware ellipse level set and the tile boundary lines, enumerating far fewer tiles than an AABB. This is the default path for 3DGS training.
- **AABB fallback path** (conics/opacities not supplied, e.g. 2DGS, IntersectTile.cu:374–463): the overlap between the per-axis radius rectangle and the tile grid.

## Formula detail (all forward)

### 0. Inputs and common indexing

`intersect_tile_kernel<scalar_t>` (IntersectTile.cu:213–464). The element index is 1D:

```
idx = this_grid().thread_rank();  if (idx >= count) return;  idx += offset;   // 246–253
// single device: offset = 0, count = n_elements
// dense:  n_elements = I * N,  image id: iid = idx / N                       // 268
// packed: n_elements = nnz,    iid = image_ids[idx]                          // 268
```

Input layout (all contiguous memory):
- `means2d`: scalar_t `[.., N, 2]` or `[nnz, 2]` → `mean2d = {(float)means2d[2*idx], (float)means2d[2*idx+1]}` (266)
- `radii`: **int32** `[.., N, 2]` — `radius_x = radii[idx*2]`, `radius_y = radii[idx*2+1]` (255–256)
- `depths`: scalar_t `[.., N]`, `conics`: **float** `[.., N, 3]` (stride 12B, the triu of Σ⁻¹), `opacities`: **float** `[.., N]`

Culling (identical in both passes, lines 255–264):

```
if (radius_x <= 0 || radius_y <= 0) { if (first_pass) tiles_per_gauss[idx] = 0; return; }
```

Note: `gaussian_ids` in the kernel signature is only declared and is **never used** in the body (lines 224–225). `flatten_ids` always receives the global flattened index `idx` (line 459).

### 1. 64-bit key encoding

The encoding is prepared in the emit pass only (lines 271–286):

```
iid_enc      = iid << (32 + tile_n_bits);                       // 273
depth_f      = (float)depths[idx];                              // 279  forced narrowing scalar_t → float
assert(depth_f >= 0.f);                                         // 284  a negative value breaks the unsigned ordering
depth_id_enc = (int64_t)__float_as_uint(depth_f);               // 285  bit reinterpretation, zero-extended into the low 32 bits
```

Key write (identical on AccuTile line 197 and AABB line 457):

```
isect_ids[cur_idx] = iid_enc | (tile_id << 32) | depth_id_enc;
```

Bit layout (verbatim upstream comment at IntersectTile.cu:455–456: "e.g. tile_n_bits = 22: image id (10 bits) | tile id (22 bits) | depth (32 bits)"):

```
bit 63 ──────────────────────────────────────────────── bit 0
[ image_id : image_n_bits ][ tile_id : tile_n_bits ][ depth bits : 32 ]
   << (32 + tile_n_bits)        << 32                 __float_as_uint
```

- Why depth is narrowed to float: when scalar_t is double, a 32-bit reinterpretation reads only half of it and monotonicity breaks (comment at lines 276–279). **Only for non-negative floats** does an unsigned comparison of the IEEE-754 bit pattern agree monotonically with a value comparison.
- `bits_for_count` (MathUtils.h:23–33):

```cpp
inline uint32_t bits_for_count(int64_t count) {
    if (count <= 1) return 0u;
    return (uint32_t)std::bit_width((uint64_t)count - 1u);   // bits needed to index 0..count-1
}
// e.g. count=1 → 0,  count=8 → 3 (a floor(log2)+1 scheme overestimates it as 4),  count=8160 → 13
```

- `image_n_bits = bits_for_count(I)`, `tile_n_bits = bits_for_count(tile_width * tile_height)` (IntersectTile.cu:507–508). Constraint: `image_n_bits + tile_n_bits <= 32` (TORCH_CHECK, 513–522). Every pack/unpack site must use the same helper for the round trip to be guaranteed (comment at MathUtils.h:13–20).
- Sorting: `cub::DeviceRadixSort::SortPairs` over the bit range `[0, 32 + tile_n_bits + image_n_bits)` (IntersectTile.cu:1096–1104). The segmented variant sorts only the low `[0, 32 + tile_n_bits)` as per-image segments, because the image axis is already contiguous (1148–1159). DoubleBuffer reduces auxiliary memory from O(N+P) to O(P) (1076–1077, 1093–1095), and after sorting the selector identifies which buffer holds the result (1105–1120).

### 2. AccuTile path — the opacity-aware level set and SNUGBOX (288–335)

The conic `(A, B, C)` is the triu of Σ⁻¹ (293–295). The quadratic form is `q(dx,dy) = A·dx² + 2B·dx·dy + C·dy²`.

```
disc = B*B - A*C;              // 298   = -det(Σ⁻¹) < 0 (when positive definite)
```

**opacity-aware level set** (300–303): `alpha = opacity · exp(-0.5·q) ≥ ALPHA_THRESHOLD` ⇔ `q ≤ 2·ln(opacity / ALPHA_THRESHOLD)`:

```
t = fminf(GAUSSIAN_EXTEND * GAUSSIAN_EXTEND, 2.0f * __logf(opacity / ALPHA_THRESHOLD));
//        = min(3.33² = 11.0889f, 2·ln(255·opacity))       — __logf is the CUDA fast-math log
```

The boundary ellipse is `{d : q(d) = t}`. The lower the opacity, the smaller t becomes and the smaller the ellipse. The upper bound `GAUSSIAN_EXTEND²` matches the radius budget used in projection (comment at 303).

Note: if `opacity < ALPHA_THRESHOLD` then `t < 0`, which makes the sqrt argument of the extent negative → NaN. The upstream original assumes that low-alpha Gaussians are filtered out upstream (in projection), since radii is computed with the same t budget.

**SNUGBOX — the tight AABB of the ellipse** (305–316):

```
neg_t_over_disc = -t / disc;                       // > 0
x_extent = sqrtf(neg_t_over_disc * C);             // = sqrt(t·C / (A·C - B²))
y_extent = sqrtf(neg_t_over_disc * A);
bbox_min = {mean.x - x_extent, mean.y - y_extent};
bbox_max = {mean.x + x_extent, mean.y + y_extent};
```

Derivation: the x-extent of `q(d)=t` follows from substituting the Lagrange condition `∂q/∂dy = 0 → dy = -B·dx/C`, giving `dx² = t·C/(AC-B²) = -t·C/disc`.

**Tangency coordinates** (313–316) — the "other axis" coordinate at the points where the ellipse touches the four sides of the AABB:

```
Bx_over_C   = B * x_extent / C;
By_over_A   = B * y_extent / A;
bbox_argmin = {mean.y + Bx_over_C,  mean.x + By_over_A};
// .x = y coordinate of the leftmost point (x = bbox_min.x)   (dy = -B·dx/C, dx = -x_extent)
// .y = x coordinate of the topmost point  (y = bbox_min.y)   (dx = -B·dy/A, dy = -y_extent)
bbox_argmax = {mean.y - Bx_over_C,  mean.x - By_over_A};
// .x = y coordinate of the rightmost point, .y = x coordinate of the bottommost point
```

**Tile rectangle clamp** (318–324) — C's `(int)` cast truncates toward zero, but the `max(0, ·)` clamp makes the negative case harmless:

```
rect_min.x = max(0, min((int)tile_width,  (int)(bbox_min.x / tile_size_f)));
rect_min.y = max(0, min((int)tile_height, (int)(bbox_min.y / tile_size_f)));
rect_max.x = max(0, min((int)tile_width,  (int)(bbox_max.x / tile_size_f + 1.f)));   // exclusive
rect_max.y = max(0, min((int)tile_height, (int)(bbox_max.y / tile_size_f + 1.f)));
if ((rect_max.y - rect_min.y) * (rect_max.x - rect_min.x) == 0) { count 0; return; }  // 326–335
```

**Sweep axis selection** (337): `isY = y_span < x_span` — the shorter of the tile rows/columns becomes the outer loop (u), minimizing the number of ellipse-line intersection computations. When `isY`, `accutile_process_tiles` swaps x↔y throughout rect/bbox/argmin/argmax right at entry (127–135), so the rest of the logic is uniform with u = the sweep axis and v = the orthogonal axis.

### 3. AccuTile path — the ellipse/tile-line intersection sweep (`accutile_process_tiles`, 96–207)

**Ellipse/vertical-line intersection** (`accutile_ellipse_intersection`, 83–94). In the sweep frame, the v coordinates of the two intersections between the line `u = coord` and the level set `q = t`:

```
p_u = isY ? p.y : p.x;   p_v = isY ? p.x : p.y;   coeff = isY ? A : C;
h         = coord - p_u;
sqrt_term = sqrtf(disc * h * h + t * coeff);
return { (-B*h - sqrt_term)/coeff + p_v,          // v_lo
         (-B*h + sqrt_term)/coeff + p_v };        // v_hi
```

Derivation (!isY, h=dx, w=dy): `C·w² + 2B·h·w + (A·h² - t) = 0 → w = (-B·h ± sqrt((B²-AC)h² + t·C)) / C`. Non-negativity of the discriminant: if `|h| ≤ extent_u` then `disc·h² + t·coeff ≥ disc·extent_u² + t·coeff = 0`. This condition is guaranteed by the caller's guards (`bbox_min.x <= min_line`, `max_line <= bbox_max.x`).

**Strip sweep** (137–206). `BLOCK = (float)tile_size`:

```
tiles_count = 0
intersect_max_line = {bbox_max.y, bbox_min.y}          // 142  neutral: v_max for the min candidate, v_min for the max candidate
min_line = rect_min.x * BLOCK                          // 144
intersect_min_line = (bbox_min.x <= min_line) ? ellipse_intersection(min_line)   // 145–152
                                              : intersect_max_line               // line is left of the ellipse → neutral
for u in [rect_min.x, rect_max.x):                     // 155, #pragma unroll 1
    max_line = min_line + BLOCK
    if (max_line <= bbox_max.x)                        // 158  update only when inside the right edge of the ellipse
        intersect_max_line = ellipse_intersection(max_line)
    // v range of this strip: the bbox extremum if the tangency point lies inside the strip, otherwise the extremum of the two boundary-line intersections
    ellipse_min = (min_line <= bbox_argmin.y && bbox_argmin.y < max_line)         // 163–170
                  ? bbox_min.y : min(intersect_min_line.x, intersect_max_line.x)
    ellipse_max = (min_line <= bbox_argmax.y && bbox_argmax.y < max_line)         // 172–179
                  ? bbox_max.y : max(intersect_min_line.y, intersect_max_line.y)
    min_tile_v = max(rect_min.y, min(rect_max.y, (int)(ellipse_min / BLOCK)))     // 181
    max_tile_v = min(rect_max.y, max(rect_min.y, (int)(ellipse_max / BLOCK + 1))) // 182
    for v in [min_tile_v, max_tile_v):                 // 185, #pragma unroll 1
        tile_id  = isY ? u * tile_width + v : v * tile_width + u                  // 187
        tile_key = image_id * n_tiles + tile_id                                   // 188
        if (tile_key < key_start || tile_key >= key_end) continue                 // 189–192  always passes on a single device
        ++tiles_count
        if (emit):                                                                // 195–200
            isect_ids[*cur_idx]   = iid_enc | (tile_id << 32) | depth_id_enc
            flatten_ids[*cur_idx] = (int32_t)flatten_idx        // = the global idx
            ++(*cur_idx)
    intersect_min_line = intersect_max_line;  min_line = max_line                 // 203–204
```

- In the last strip, if `max_line > bbox_max.x`, `intersect_max_line` retains the previous boundary-line value (= `intersect_min_line`) — conservatively safe, because if the extremum lies in that strip the argmin/argmax branch covers it.
- emit order: a deterministic order of ascending u then ascending v (in the sweep frame). The `cur_idx` starting point is `first_pass ? 0 : (idx == 0 ? 0 : cum_tiles_per_gauss[idx-1])` (338) — the exclusive conversion of the inclusive cumsum.
- At the end of the count pass, `tiles_per_gauss[idx] = (int32_t)tiles_count` (369–372).

### 4. AABB fallback path (374–463)

Used when conics/opacities are not supplied (2DGS and the like). The intersection between the per-axis radius rectangle and the tile grid:

```
tile_radius_x = radius_x / (float)tile_size;  tile_radius_y = radius_y / (float)tile_size;   // 377–378
tile_x = mean.x / tile_size;  tile_y = mean.y / tile_size;                                    // 379–380
tile_min.x = min(max(0, (int32_t)floor(tile_x - tile_radius_x)), tile_width);   // 384  inclusive
tile_min.y = min(max(0, (int32_t)floor(tile_y - tile_radius_y)), tile_height);
tile_max.x = min(max(0, (int32_t)ceil (tile_x + tile_radius_x)), tile_width);   // 386  exclusive
tile_max.y = min(max(0, (int32_t)ceil (tile_y + tile_radius_y)), tile_height);
```

Per-row key-window clipping, `row_key_column_range` (59–77) — a helper that clips the column range to the global tile-key interval `[key_start, key_end)` owned by each device under multi-GPU. On a single device (`key_start=0, key_end=I*n_tiles`) it is the identity: `col_start=rect_min.x, col_end=rect_max.x, done=false`:

```
row_key_offset = image_id * n_tiles + row * tile_width;
done      = row_key_offset + rect_min.x >= key_end;
col_start = min(rect_max.x, max(rect_min.x, key_start - row_key_offset));
col_end   = max(rect_min.x, min(rect_max.x, key_end   - row_key_offset));
```

count pass (400–432): sums `range.end - range.start` per row (with per-element checks when a mask is present, 414–428). emit pass (434–462): the same traversal, writing keys with `tile_id = i * tile_width + j` (457–459). `tile_mask` (optional, `[I, tile_h, tile_w]` bool) is used only on the sparse path and is applied identically in both passes, preserving count/emit consistency (388–398).

### 5. The 2-pass orchestration and prefix sum (Intersect.cpp:230–296)

```
Pass 1: launch_intersect_tile_kernel(..., cum_tiles_per_gauss = nullopt,
                                     tiles_per_gauss as output, isect_ids/flatten_ids = nullopt)   // 238–256
cum_tiles_per_gauss = at::cumsum(tiles_per_gauss.view({-1}), 0, at::kLong);   // 258  int64 stated explicitly (the kernel reads it as int64)
n_isects = cum_tiles_per_gauss[-1].item<int64_t>();                           // 259  ← synchronous GPU→CPU readback
isect_ids   = at::empty({n_isects}, kLong);  flatten_ids = at::empty({n_isects}, kInt);  // 273–274
Pass 2: launch_intersect_tile_kernel(..., cum_tiles_per_gauss passed in,
                                     tiles_per_gauss = nullopt, isect_ids/flatten_ids as output)   // 277–296
```

Offsets for the segmented sort (optional, 260–265): `offsets = cat({0}, cumsum(sum(tiles_per_gauss, -1)))` — the exclusive boundaries of the per-image intersection counts.

**Where cub::DeviceScan is used**: on the standard single-GPU path the prefix sum is `at::cumsum` (Intersect.cpp:258); cub appears only on the multi-GPU `privateuseone` path:

```cpp
CUB_WRAPPER_ASYNC(stream, cub::DeviceScan::InclusiveSum,
                  device_counts[device_id], device_cumsum[device_id], n_elements);   // IntersectTile.cu:796–798
```

That same multi-GPU path also uses `cub::DeviceMergeSort::SortPairs` (910–917) and the `atomicAdd_system` inside `add_counts_kernel` for summing per-device counts (670–684). **The single-device path contains no atomics at all.**

### 6. isect_offset_encode — `intersect_offset_kernel` (925–988)

Builds the per-tile start offsets `offsets[I, tile_height, tile_width]` (int32) from the sorted `isect_ids`. One intersection per thread. Example from the upstream comment (936–939): `ids = [1,1,1,3,3], n_tiles = 6 → counts [0,3,0,2,0,0], cumsum [0,3,3,5,5,5], offsets [0,0,3,3,5,5]`.

```
isect_id_curr = isect_ids[idx] >> 32;                       // 947  strip the 32 depth bits
iid_curr      = isect_id_curr >> tile_n_bits;               // 948
tid_curr      = isect_id_curr & ((1 << tile_n_bits) - 1);   // 949  ⚠ int shift: UB when tile_n_bits=32 (outside the range used in practice)
id_curr       = iid_curr * n_tiles + tid_curr;              // 950  flattened tile index

if (idx == 0)                                               // 952–959
    for i in [0, id_curr]:            offsets[i] = 0;       //  fills up to and including the first valid tile
if (idx == n_isects - 1)                                    // 960–967
    for i in [id_curr+1, I*n_tiles):  offsets[i] = n_isects;//  fills the tail
if (idx > 0):                                               // 969–987
    isect_id_prev = isect_ids[idx-1] >> 32;
    if (isect_id_prev == isect_id_curr) return;             //  same (image,tile) → not a boundary
    id_prev = (isect_id_prev >> tile_n_bits) * n_tiles + (isect_id_prev & ((1 << tile_n_bits) - 1));
    for i in [id_prev+1, id_curr]:    offsets[i] = idx;     //  writes the boundary, empty tile ranges included
```

- When `n_isects == 0`, the host calls `offsets.fill_(0)` without launching the kernel (1004–1008).
- Consumption contract: the intersection range of tile `t` (flattened index) is `[offsets[t], t+1 < I*n_tiles ? offsets[t+1] : n_isects)` — the rasterizer walks the `flatten_ids` of that range in ascending depth order.
- Load characteristics: because a boundary thread fills empty tile ranges in a loop, a single thread writes a long range when intersections are sparse (worst case O(I·n_tiles)). The upstream original accepts this.

## Thread and memory configuration

| Kernel | Grid | Block | Work per thread | shared mem | atomics |
|---|---|---|---|---|---|
| `intersect_tile_kernel` (count/emit) | `ceil(n_elements / 256)` | **256** (IntersectTile.cu:524–525) | full tile enumeration for one Gaussian (× image) | none | none |
| `intersect_offset_kernel` | `ceil(n_isects / 256)` | **256** (1001–1002) | one intersection; fills the range if it is a boundary | none | none |
| `add_counts_kernel` (multi-GPU only) | `ceil(n_elements/256)` | 256 | one `atomicAdd_system` | none | yes |

- When `n_elements == 0`, the launch itself is skipped (527–531).
- Divergence: `intersect_tile_kernel` runs a variable-length double loop proportional to the Gaussian's size → large load imbalance within a warp. The upstream original accepts this without sorting or reordering. `#pragma unroll 1` (154, 184) suppresses unrolling (to avoid register pressure).
- Memory access: all inputs are contiguous, coalesced reads; the outputs `isect_ids/flatten_ids` are contiguous per-Gaussian range writes.

## MSL porting design

Assumes a single device (common to M4 / A17 Pro: simdgroup 32, threadgroup memory 32KB). Multi-GPU-only logic is removed — the `key_start/key_end` window (always the identity `[0, I*n_tiles)`), the `done` early exit in `row_key_column_range`, `add_counts_kernel`, and `atomicAdd_system`. **This kernel family needs neither atomics nor threadgroup barriers** (the prefix sum aside).

### Parameter block

```metal
struct IsectTileParams {
    uint  n_elements;    // I*N (dense) or nnz (packed)
    uint  I;             // number of images
    uint  N;             // dense: Gaussians per image, packed: 0
    uint  packed;        // 0/1
    uint  tile_size;
    uint  tile_width;
    uint  tile_height;
    uint  tile_n_bits;   // bits_for_count(tile_width*tile_height)  — computed on the CPU
    uint  image_n_bits;  // bits_for_count(I)                        — for validation
};
// constants: GAUSSIAN_EXTEND = 3.33f, ALPHA_THRESHOLD = 1.0f/255.0f (function constant or #define)
// computed on the CPU: bits_for_count(c) = (c <= 1) ? 0 : (64 - __builtin_clzll((uint64_t)c - 1))
//                      validate image_n_bits + tile_n_bits <= 32 before dispatching
```

### Kernel 1: `isect_tile_count`

```metal
kernel void isect_tile_count(
    device const float*  means2d          [[buffer(0)]],  // [n_elements*2], stride 8B
    device const int*    radii            [[buffer(1)]],  // [n_elements*2] int32
    device const float*  conics           [[buffer(2)]],  // [n_elements*3], stride 12B — do not load as float3 (16B alignment); access element-wise
    device const float*  opacities        [[buffer(3)]],  // [n_elements]
    device const long*   image_ids        [[buffer(4)]],  // packed mode only (unbound for dense)
    device int*          tiles_per_gauss  [[buffer(5)]],  // [n_elements] output
    constant IsectTileParams& p           [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
```

- threadgroup size **256** (`MTLSize(256,1,1)`), grid `ceil(n_elements/256)` or the exact size via `dispatchThreads`. Keep the early return `if (gid >= p.n_elements) return;`.
- The body is a direct translation of IntersectTile.cu:255–372. There is no emit branch (only `tiles_per_gauss[gid] = count`).
- `__logf(x)` → depending on the precision policy, either `metal::log(x)` (preferred, matches the CPU reference) or `metal::fast::log(x)` (matching the CUDA fast-math approximation). See open question 1.
- `#pragma unroll 1` → `#pragma clang loop unroll(disable)`.

### Kernel 2: prefix sum — replacing cub::DeviceScan / at::cumsum (2-level scan)

`tiles_per_gauss` int32 → `cum_tiles_per_gauss` int64 (keep int64 as in the upstream original; n_isects can approach the int32 limit). A 3-dispatch reduce-then-scan:

```metal
// Dispatch A: partial sums per group.  256 threads × ELEMS(=4) = 1024 elements/group
kernel void scan_partial(
    device const int*  in          [[buffer(0)]],   // tiles_per_gauss
    device long*       out         [[buffer(1)]],   // result of the in-group inclusive scan
    device long*       block_sums  [[buffer(2)]],   // [num_groups]
    constant uint&     n           [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid_group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]])
// inside: thread-serial(ELEMS) → simd_prefix_inclusive_sum → simdgroup sums into threadgroup memory (8 of them, 32B)
//         → threadgroup_barrier(mem_flags::mem_threadgroup) → simdgroup 0 scans the 8 sums → barrier → add the offsets
// threadgroup memory: long[8] = 64B (negligible against the 32KB limit)

// Dispatch B: exclusive scan of block_sums.  A single threadgroup handles all of it in a carry-carrying loop
kernel void scan_blocks_single_group(device long* block_sums, constant uint& num_groups, ...)
// num_groups ≤ a few tens of thousands (5M Gaussians / 1024 ≈ 4883) → a single-group loop suffices; no 3-level recursion needed

// Dispatch C: broadcast-add the group offsets
kernel void scan_add_offsets(device long* out, device const long* block_sums, constant uint& n, ...)
```

- Barrier points: two `threadgroup_barrier` calls inside Dispatch A. Between dispatches, command-encoder ordering guarantees it (sequential dispatches within the same encoder, plus `memoryBarrier(scope: .buffers)` or a separate encoder if needed).
- `n_isects` readback: blit `cum[-1]` into a storageModeShared buffer and then `waitUntilCompleted` — the same sync point as the upstream original's `.item<int64_t>()` (Intersect.cpp:259). On Apple Silicon UMA the blit can be skipped by reading the shared buffer directly.
- Porting reference: brush's (Apache-2.0) `crates/brush-prefix-sum` is a WGSL implementation with the same purpose — permitted as a reference for cross-checking the algorithm.

### Kernel 3: `isect_tile_emit`

```metal
kernel void isect_tile_emit(
    device const float*  means2d              [[buffer(0)]],
    device const int*    radii                [[buffer(1)]],
    device const float*  depths               [[buffer(2)]],  // unnecessary for count, bound for emit only
    device const float*  conics               [[buffer(3)]],
    device const float*  opacities            [[buffer(4)]],
    device const long*   image_ids            [[buffer(5)]],  // packed mode only
    device const long*   cum_tiles_per_gauss  [[buffer(6)]],
    device long*         isect_ids            [[buffer(7)]],  // [n_isects]
    device int*          flatten_ids          [[buffer(8)]],  // [n_isects]
    constant IsectTileParams& p               [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
```

- threadgroup 256, the same grid as count. The tile enumeration formula MUST be **exactly the same code** as in count (share it as an inline function by templating `accutile_process_tiles`; rather than a function constant or a nullable device pointer, split emit and count into **two kernel variants** to avoid MSL's null-buffer binding problem).
- Key assembly: `iid_enc = (long)iid << (32 + p.tile_n_bits);` `depth_id_enc = (long)as_type<uint>((float)depths[gid]);` — `__float_as_uint` ↔ `as_type<uint>`. MSL has no `assert(depth_f >= 0)`, so replace it with a separate validation kernel in debug builds or with CPU reference verification.
- `cur_idx = (gid == 0) ? 0 : cum_tiles_per_gauss[gid - 1];` — each thread writes only into its own range, so no atomic is needed.

### Sorting (replacing cub::DeviceRadixSort — out of scope here, only the contract is stated)

- Required contract: sort `(int64 key, int32 value)` pairs ascending by the low `32 + tile_n_bits + image_n_bits` bits of the key, and **stable** (cub radix is stable — preserving the pre-sort order of equal keys is required for bit-identical golden fixtures).
- Treat the key as **ulong (uint64)**. cub special-cases the sign bit of int64 signed keys, and in the extreme case `image_n_bits + tile_n_bits == 32` bit63 can be set. An unsigned sort matches the encoding intent exactly.
- Implementation candidates: (a) an in-house LSD radix sort at 8 bits/pass — sorting only the significant bits to cut the pass count (e.g. 13+0+32 = 45 bits → 6 passes); (b) porting brush's `crates/brush-sort` (WGSL GPU radix sort, Apache-2.0). To be decided in a separate spec document.

### Kernel 4: `isect_offset_encode`

```metal
kernel void isect_offset_encode(
    device const long* isect_ids  [[buffer(0)]],   // already sorted, [n_isects]
    device int*        offsets    [[buffer(1)]],   // [I * tile_height * tile_width]
    constant IsectOffsetParams& p [[buffer(2)]],   // { uint n_isects; uint I; uint n_tiles; uint tile_n_bits; }
    uint gid [[thread_position_in_grid]])
```

- threadgroup 256, grid `ceil(n_isects/256)`. A direct translation of IntersectTile.cu:925–988. No atomics or barriers are needed (only boundary threads write, and their ranges do not overlap — the idx==0 thread covers `[0, id_curr]`, boundary threads cover `(id_prev, id_curr]`, and the last thread covers `(id_curr, I*n_tiles)`).
- Mask in 64 bits: `tid = (ulong)isect_id & (((ulong)1 << p.tile_n_bits) - 1);` — guards against the int UB of the upstream original's `(1 << tile_n_bits)`.
- When `n_isects == 0`, skip the dispatch and fill with zeros on the CPU (or with a fill kernel) — the same as the upstream original's 1004–1008.

### Dispatch sequence summary (one encoder preferred)

```
[isect_tile_count] → [scan_partial] → [scan_blocks] → [scan_add_offsets]
→ (CPU: n_isects readback, allocate isect_ids/flatten_ids)    ← the only CPU sync point
→ [isect_tile_emit] → [radix sort passes…] → [isect_offset_encode]
```

## Open questions

1. **`__logf` precision policy**: CUDA's `__logf` is a fast-math approximation, whereas the CPU golden reference will use a precise log. A difference in `t` can change whether a boundary tile is included (which changes the count itself). Decide whether the golden fixture comparison demands "an exact match of the tile set" or "one boundary tile of slack". Recommended: use `metal::log` in MSL and standardize the CPU reference on `logf` as well, securing a near-bit-level match.
2. **The n_isects readback sync point**: the upstream original also synchronizes with `.item()` (Intersect.cpp:259). Decide whether Metal permits one `waitUntilCompleted` per training step, or goes sync-free by preallocating a worst case (e.g. the previous frame's n_isects × a slack factor) plus a GPU-side clamp.
3. **opacity < ALPHA_THRESHOLD → t < 0 → sqrt(NaN)**: the upstream original does not guard against this, on the premise of upstream culling. Decide whether MSL adds a `t <= 0` early exit (count 0) or leaves it as-is to match the original's behavior. (Adding it leaves results unchanged for valid inputs — adding the guard is preferred, but the fixture policy needs confirmation.)
4. **Choice of sort kernel implementation**: an in-house LSD radix vs porting brush's `brush-sort`. Includes optimizing the pass count against the number of significant bits (`32 + tile_n_bits + image_n_bits`), guaranteeing stability, and handling ulong keys — decide whether this is split into a separate spec (e.g. 04-radix-sort).
5. **Phase 1 scope for the packed mode and the tile_mask (sparse) path**: the default training path is dense + AccuTile. Decide whether to exclude packed (the `image_ids` indirection) and `tile_mask`/`intersect_tile_sparse` (Intersect.cpp:563–689) from the initial port. (The kernel's `gaussian_ids` argument is unused even in the upstream original — it is fine to settle on removing it in MSL.)
6. **Load imbalance in the `offsets` fill**: in scenes with many empty tiles, the boundary thread's range-fill loop grows long. Keep the upstream approach vs the alternative of a per-tile binary search (thread = tile, lower_bound over `isect_ids`) — keep it if reproduction of the original takes priority, and revisit after profiling.
7. **When the AABB path is needed for 2DGS**: if Phase 1 is 3DGS-only, decide whether to implement the AABB fallback (no conics) now or to go with AccuTile first. The fallback branches automatically on whether the fixture supplies conics/opacities, so keeping the interface identical is preferred.
8. **int64 arithmetic performance**: 64-bit integer arithmetic is slower than 32-bit on Apple GPUs. Key assembly and comparison are unavoidable, but for single-image training (I=1, image_n_bits=0) there is room to replace the (tile_id << 32 | depth) structure with a 2-pass sort over a 32-bit (tile_id, depth) pair — keep this separate from the correctness spec and review it in the performance stage.

## Verification notes (adversarial verifier)

The CUDA originals (all of `IntersectTile.cu`, `Intersect.cpp:150–349, 520–689`, `MathUtils.h`, `Common.h:85–109`, and the decorator/function locations in `_wrapper.py`) were re-read independently, and the spec's formulas, signs, indices, clamps, constants, and line-number evidence were compared item by item.

**Conclusion: verification passed. Zero formula errors.** Items compared — the 2-pass structure with the cumsum and readback (Intersect.cpp:236–296), the key encoding (`iid << (32+tile_n_bits)`, `tile_id << 32`, `__float_as_uint` zero-extend; IntersectTile.cu:273–285, 197, 457), `bits_for_count` (MathUtils.h:23–33, with the example values 8→3 and 8160→13 recomputed and matching), the AccuTile level set `t = fminf(3.33², 2·__logf(opacity/ALPHA_THRESHOLD))` (303) and the SNUGBOX extents and argmin/argmax signs (306–316, re-derived via Lagrange and matching), the root formula and non-negative-discriminant guard of `accutile_ellipse_intersection` (83–94, 145–161), the strip sweep's neutral-value initialization, argmin/argmax branches, and `min_tile_v/max_tile_v` clamps (142–182), `tile_id = isY ? u*tile_width+v : v*tile_width+u` (187), the ordering that puts the key-window skip **before** `++tiles_count` (189–193), the AABB path's floor/ceil clamps (384–387) and `row_key_column_range` (59–77), the three boundary-fill ranges of `intersect_offset_kernel` (952–987, with the example [0,0,3,3,5,5] recomputed and matching), the 256 threads / ceil grid (524–525, 1001–1002), the cub usage sites (DeviceScan only under privateuseone, 796–798; radix bit ranges 1096–1104/1148–1159), and the constants (Common.h:97, 99).

Corrections and supplements (precision improvements, not errors):

1. **Line-evidence correction**: `assert(depth_f >= 0.f)` is at IntersectTile.cu:**283** (spec §1 cites 284 — line 284 is the comment on the following line). The `depth_id_enc` assignment at 285 is correct.
2. **Supplement on the radii load type**: the upstream original **loads the int32 buffer into float local variables** — `const float radius_x = radii[idx * 2];` (IntersectTile.cu:255–256). The culling comparison (257) and the AABB divisions (377–378) are therefore float operations. Since int32 values are exactly representable in float, reading them as int in MSL and doing the comparison/division there gives identical results — either is fine for a direct translation.
3. **Supplement on the host skip conditions**: reflect the upstream original's two skips in the MSL dispatch sequence (§Dispatch sequence summary) — skip Pass 1 when `n_elements == 0` and treat `n_isects = 0` (Intersect.cpp:236–270), and skip Pass 2 (emit) and the sort when `n_isects == 0` (Intersect.cpp:275, 299). The zero-fill fallback for offset encode is already reflected in spec §Kernel 4 (IntersectTile.cu:1004–1008).
4. **Supplement on `(int)` truncation**: the `(int)` casts in `min_tile_v/max_tile_v` (IntersectTile.cu:181–182) also truncate toward zero, exactly like the rect clamp in §2, but the `max(rect_min.y, ·)/min(rect_max.y, ·)` clamps (rect values ≥ 0) leave the negative case unchanged — porting it on a floor basis gives the same result after clamping.

No omissions or errors beyond those 4 items. All six items of the extraction brief (the 2-pass structure, the AccuTile formulas, the 64-bit key and bits_for_count, isect_offset_encode, replacing cub::DeviceScan with a 2-level scan, and the thread configuration) were confirmed covered. This document is sufficient as the sole reference for the Metal implementation.

### Second verification (independent re-verification)

The four upstream files (all of `IntersectTile.cu` 1–1177, `Intersect.cpp:140–369, 540–689`, all of `MathUtils.h`, and `Common.h:85–109`) and the `_wrapper.py` locations were re-read independently of the first verification above, and every formula, sign, index, clamp, constant, and line-number reference in the spec body and the first-pass verification notes was re-compared item by item.

**Conclusion: verification passed, no corrections. Zero new errors.** All four corrections in the first-pass verification notes were also confirmed to match the upstream original (the assert is indeed at IntersectTile.cu:283; radii is indeed loaded into float locals at IntersectTile.cu:255–256; the host skips are indeed at Intersect.cpp:236/275/299; and the harmlessness of the (int) truncation under clamping was confirmed by numerical recomputation down to the negative-coordinate cases — truncation with bbox_min<0 and the `(int)(x/tile+1.f)` boundary).

Non-obvious items additionally confirmed by recomputation:
- SNUGBOX argmin/argmax signs: re-deriving the tangency condition at the leftmost point (dx=-x_extent), `2B·dx+2C·dy=0 → dy=+B·x_extent/C` — the sign of `bbox_argmin.x = mean.y + Bx_over_C` (315) matches. The topmost point matches by symmetry.
- The non-negative-discriminant guard: `disc·h² + t·coeff ≥ 0 ⇔ |h| ≤ extent_u` — case analysis confirms that the initial `min_line` call is guaranteed by the combination of the `bbox_min.x ≤ min_line` guard (145) and the early exit with rect_max.x=0 when bbox_max.x < 0 (328–335), while the in-loop `max_line` calls are guaranteed by the guard at 158. Because `max_line` increases monotonically, once the guard at 158 turns off it never turns back on — the spec §3 statement that it "retains the previous boundary-line value (= intersect_min_line)" is accurate (since 203 resynchronizes it on every iteration).
- Non-overlap of the `intersect_offset_kernel` write ranges: idx==0 covers `[0, id_curr₀]`, boundaries cover `[id_prevₖ+1, id_currₖ]`, and the last thread covers `[id_curr₊1, I·n_tiles)` — confirmed that every range is written exactly once, including the n_isects==1 case (where the first thread is also the last thread).
- `bits_for_count` boundary values: count=1→0, 8→3, 8160→13 recomputed and matching (MathUtils.h:25–31). The reading of `isY = y_span < x_span` (337) as "the shorter axis becomes the outer loop" matches (traced through the swap at 127–135).
- Two harmless line-number notations (no correction needed): in spec §0's "246–253", line 246 is a comment line (the code is 247–253); `@torch.no_grad()` is at _wrapper.py:1034 (within the tolerance of the spec's "around 1035").

This document (including the first-pass corrections) is sufficient as the sole reference for the Metal implementation.
