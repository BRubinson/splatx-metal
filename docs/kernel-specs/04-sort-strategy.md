# 04. GPU sorting strategy — MSL porting spec for the isect key sort

Gaussian Splatting rasterization requires an intersection list sorted in (image, tile, depth) order.
This document covers (1) how gsplat uses `cub::DeviceRadixSort`, (2) brush's (Apache-2.0)
WGSL/CubeCL reduce-then-scan radix sort, (3) the MSL multi-pass radix sort design spec
built on that evidence, and (4) the tile-local sorting alternative with its A/B bench plan.

---

## Upstream files and commits

### gsplat (Apache-2.0, pinned at commit `77ab983ffe43420b2131669cb35776b883ca4c3c`)

| File | Contents |
|---|---|
| `research/gsplat/gsplat/cuda/csrc/IntersectTile.cu` | isect key generation (intersect_tile_kernel), `radix_sort_double_buffer` (lines 1078–1121), `segmented_radix_sort_double_buffer` (lines 1125–1176), `cub::DeviceMergeSort` for multi-GPU (lines 899–920), `intersect_offset_kernel` (lines 925–988) |
| `research/gsplat/gsplat/cuda/csrc/Intersect.cpp` | host orchestration: bit-width computation (lines 213–228), building the segmented offsets (lines 260–265), the sort branch (lines 299–324) |
| `research/gsplat/gsplat/cuda/csrc/MathUtils.h` | `bits_for_count` (lines 23–33) |
| `research/gsplat/gsplat/cuda/csrc/SparseTileLayout.cu` | bit-bounded sorting of pixel keys, `launch_sort_tile_keys` (lines 104–129) — a secondary case |

### brush (Apache-2.0, commit `3b80985709e2ec04fd6c8622a40e36473647a8e0`)

| File | Contents |
|---|---|
| `research/brush/crates/brush-sort/src/kernels.rs` | the five CubeCL radix sort kernels (count/reduce/scan/scan_add/scatter) |
| `research/brush/crates/brush-sort/src/lib.rs` | the `radix_argsort` host loop (pass orchestration, lines 16–125) |
| `research/brush/crates/brush-render/src/render.rs` | the two-stage sorting strategy: DepthSort (lines 176–183) + Tile sort (lines 227–229) |
| `research/brush/crates/brush-render/src/kernels/map_gaussians.rs` | emits (tile_id, compact_gid) pairs + sentinel padding |
| `research/brush/crates/brush-render/src/get_tile_offset.rs` | sorted result → per-tile [start, end) offsets |
| `research/brush/crates/brush-sort/README.md` | provenance: compute-shader-101 PR#31 ← FidelityFX Parallel Sort (the reduce-then-scan structure) |

License notice: both repositories are Apache-2.0, and the derivation notice is retained when porting.
The OpenSplat / msplat / splat-apple / gsplat-mlx code was not read (complying with the prohibited list).

---

## Algorithm overview

### gsplat: a single 64-bit key + a vendor-library radix sort

1. Run `intersect_tile_kernel` in 2 passes: the first counts the tiles per Gaussian (`tiles_per_gauss`),
   then after a `cumsum` the second emits `(isect_ids: int64, flatten_ids: int32)` pairs.
2. Key encoding (high bits → low bits): `image_id | tile_id | float_bits(depth)`.
3. `cub::DeviceRadixSort::SortPairs` sorts **only the significant bit range of the key**
   (`begin_bit=0`, `end_bit = 32 + tile_n_bits + image_n_bits`).
4. From the sorted `isect_ids`, `intersect_offset_kernel` produces the per-tile start offsets `[I, tile_h, tile_w]`.

The CUB radix sort is **stable**, so equal keys (identical image/tile/depth bits) keep their emission
order (= ascending gaussian index) → deterministic.

### brush: 32-bit keys × a two-stage stable radix argsort

brush never builds a 64-bit key. Instead it reaches the same result with **two stable sorts**:

- **Stage 1 (DepthSort)**: argsort the visible Gaussians by `depth` (the raw f32 bits, 32 bits)
  → `global_from_compact_gid`, a compact id permutation in ascending depth (render.rs:176–183).
- **Stage 2 (Tile sort)**: argsort the intersection list `(tile_id_from_isect, compact_gid_from_isect)`
  by `tile_id` alone (only as many bits as needed) (render.rs:227–229).
  Because the sort is stable and the values (compact_gid) are already in depth order,
  **each tile ends up in ascending depth order automatically**.

The sort itself is a **multi-pass LSD radix sort (4-bit digit, reduce-then-scan)** in the
FidelityFX Parallel Sort family: each pass runs the five kernels
`count → reduce → scan (single workgroup) → scan_add → scatter`,
and the pass count is `ceil(sorting_bits / 4)` (lib.rs:57).
It uses neither decoupled-lookback nor OneSweep, so it makes no forward-progress assumption
between workgroups — that is the crux of portability to wgpu (Metal included), and it is the
structure our MSL port follows.

---

## Formula detail

### Forward (sorting is a forward-only operation)

Sorting is non-differentiable, so **no backward exists**.
However, the backward kernel (rasterize backward) reuses the `flatten_ids` / `tile_offsets`
produced by the forward as-is and walks the splats within a tile **in reverse**, so the forward
and backward must **share the same sort-result buffers** for the alpha-compositing
recomputation to match. (The order contract — see below.)

#### gsplat key encoding (IntersectTile.cu)

```
// IntersectTile.cu:273           iid_enc = iid << (32 + tile_n_bits);
// IntersectTile.cu:279-285       depth_f = (float)depths[idx];  assert(depth_f >= 0.f);
//                                depth_id_enc = __float_as_uint(depth_f);   // low 32 bits, zero-extend
// IntersectTile.cu:197 (AccuTile path), 457 (AABB path):
isect_ids[cur_idx]   = iid_enc | (tile_id << 32) | depth_id_enc;   // int64
flatten_ids[cur_idx] = (int32_t)idx;                               // gaussian flat index
```

- Bit layout (e.g. `tile_n_bits = 22`, comment at IntersectTile.cu:455–456):
  `[ image id (image_n_bits) | tile id (tile_n_bits) | depth f32 bits (32) ]`
- The IEEE-754 bit ordering is monotonic only when depth is **non-negative**.
  That holds after near-plane culling, and the upstream original pins it with `assert(depth_f >= 0.f)`
  (IntersectTile.cu:283).
- `tile_id = i * tile_width + j` (row-major, IntersectTile.cu:454).

#### Bit-width computation (MathUtils.h:23–33, Intersect.cpp:213–228)

```
bits_for_count(count) = (count <= 1) ? 0 : bit_width(uint64(count) - 1)   // = ceil(log2(count))
image_n_bits = bits_for_count(I)
tile_n_bits  = bits_for_count(n_tiles),  n_tiles = tile_width * tile_height
constraint: image_n_bits + tile_n_bits <= 32   (TORCH_CHECK, Intersect.cpp:219-228)
```

Example: 1920×1080 with tile_size=16 → tile_width=120, tile_height=68, n_tiles=8160
→ `tile_n_bits = bit_width(8159) = 13`. With I=1 → `image_n_bits = 0`.

#### The three gsplat sort calls

**(a) Global radix sort — the default path** (IntersectTile.cu:1078–1121)

```
// keys: int64 isect_ids, values: int32 flatten_ids. DoubleBuffer cuts auxiliary memory from O(N+P) to O(P).
cub::DoubleBuffer<int64_t> d_keys(isect_ids, isect_ids_sorted);        // :1094
cub::DoubleBuffer<int32_t> d_values(flatten_ids, flatten_ids_sorted);  // :1095
cub::DeviceRadixSort::SortPairs(d_keys, d_values, n_isects,
    /*begin_bit=*/0,
    /*end_bit=*/32 + tile_n_bits + image_n_bits,                       // :1101-1102
    stream);
// d_keys.selector / d_values.selector identify which buffer holds the sorted result (:1105-1120)
```

The key optimization: `end_bit` restricts the sort to the significant bits (45, say) → fewer radix
passes than sorting all 64 (`passes = ceil((end_bit-begin_bit)/RADIX_BITS)`).

**(b) Segmented radix sort — per-image segments** (IntersectTile.cu:1125–1176)

```
// the image dimension is contiguous within isect_ids, so splitting into per-image segments
// lets image_n_bits be excluded from the sort (comment at :1145-1147).
cub::DeviceSegmentedRadixSort::SortPairs(d_keys, d_values, n_isects,
    /*n_segments=*/I,
    offsets, offsets + 1,          // int64 [I+1], cumulative per-image isect offsets
    /*begin_bit=*/0,
    /*end_bit=*/32 + tile_n_bits,  // :1157 — the image bits are omitted
    stream);
// building offsets (Intersect.cpp:260-265):
//   offsets = cat([0], cumsum(sum(tiles_per_gauss, -1)))
// forbidden for packed input (Intersect.cpp:211): TORCH_CHECK(!(packed && segmented))
```

**(c) Multi-GPU path — merge sort** (IntersectTile.cu:899–920)

```
// sorts each device's contiguous key-range chunk separately (a comparison sort with no bit limit)
cub::DeviceMergeSort::SortPairs(isect_ids + isect_offset, flatten_ids + isect_offset,
                                isect_count, ::cuda::std::less<int64_t>{});   // :912-916
```

**(Secondary) SparseTileLayout.cu:120** — pixel keys follow the same pattern:
a bit-bounded `DeviceRadixSort::SortPairs` call with
`end_bit = pos_bits + nbits(n_images * n_tiles)` (Intersect.cpp:752–753).
"Sort only the significant bits" is the convention throughout gsplat.

#### Post-sort processing: tile offsets (IntersectTile.cu:925–988)

```
// thread idx compares the high 32 bits (image|tile) of isect_ids[idx] and isect_ids[idx-1],
// and where they change it writes offsets[id_prev+1 .. id_curr] = idx (:969-987)
isect_id_curr = isect_ids[idx] >> 32;                       // strip the depth bits
iid_curr = isect_id_curr >> tile_n_bits;
tid_curr = isect_id_curr & ((1 << tile_n_bits) - 1);
id_curr  = iid_curr * n_tiles + tid_curr;
// idx==0 → offsets[0..id_curr] = 0,  idx==n_isects-1 → offsets[id_curr+1..] = n_isects
```

#### The brush two-stage strategy (render.rs)

```
// stage 1: depth argsort — all 32 bits (render.rs:180-181)
(_, global_from_compact_gid) = radix_argsort(depths[0..num_visible],
                                             global_from_presort_gid[0..num_visible], 32);
// the depth key reinterprets the f32 tensor as u32 (src: Tensor<u32> in kernels.rs).
// the bit ordering is monotonic on the premise depth >= 0 — the same assumption as gsplat.

// gather the intersection counts in depth order, then prefix sum (render.rs:184-186)
compact_counts = gather(intersect_counts, global_from_compact_gid);
cum_tiles_hit  = prefix_sum(compact_counts);

// emit (tile_id, compact_gid) (map_gaussians.rs:61-82)
//   tile_id = tx + ty * tile_bw                              (map_gaussians.rs:68)
//   sentinel_tile_id = tile_bw * tile_bh                     (map_gaussians.rs:56)
//   count mismatches are padded with the sentinel — it sorts after the valid tiles and is excluded from rasterization

// stage 2: tile argsort — only the bits needed (render.rs:227-229)
bits = u32::BITS - num_tiles.leading_zeros();   // = bit_width(num_tiles); the width including the sentinel
(tile_id_from_isect, compact_gid_from_isect) = radix_argsort(tile_id_from_isect,
                                                             compact_gid_from_isect, bits);
```

- Example: n_tiles=8160 → bits=13 → pass count `ceil(13/4)=4`. Stage 1's 32 bits takes 8 passes.
  That is 12 passes in total, but each pass moves only 32-bit keys/values — half the bandwidth
  of moving gsplat-style 64-bit keys.
- Bit-width difference from gsplat: gsplat uses `bits_for_count(n) = bit_width(n-1)`,
  brush uses `bit_width(n)` — brush is one bit wider when n is exactly a power of two
  (because the sentinel `n_tiles` must fit in the key space).
- **The crux of consistency is stability**: the stage-2 sort must be stable for the depth order within a tile to survive.

#### The brush radix sort kernel formulas (kernels.rs)

Constants (kernels.rs:7–20):

```
WG = 256 (workgroup thread count), BITS_PER_PASS = 4, BIN_COUNT = 16,
ELEMENTS_PER_THREAD = 4, BLOCK_SIZE = WG * ELEMENTS_PER_THREAD = 1024,
MAX_SUBGROUPS = 32   // padding to cover variable subgroup widths of 8~64 (Apple is fixed at 32)
num_wgs = ceil(num_keys / BLOCK_SIZE)
num_reduce_wgs = BIN_COUNT * ceil(num_wgs / BLOCK_SIZE)
```

**pass p (p = 0 .. ceil(bits/4)-1), shift = 4p:**

1. `sort_count` (kernels.rs:28–65): reads BLOCK_SIZE keys per workgroup, accumulates a
   threadgroup atomic histogram, and stores it **bin-major**:
   ```
   digit = (src[i] >> shift) & 0xF                       // :55
   counts[digit * num_wgs + group_id] = hist[digit]      // :63  (bin-major layout)
   ```
   Because it is bin-major, a linear exclusive scan of counts gives the global start offsets in
   "(ascending digit, ascending workgroup)" order → **the premise of a stable LSD**.

2. `sort_reduce` (kernels.rs:67–124): sums counts in BLOCK_SIZE chunks →
   `reduced[num_reduce_wgs]`. Hierarchical reduction with the subgroup `plane_sum` plus threadgroup partials.

3. `sort_scan` (kernels.rs:126–212): **a single workgroup** exclusive-scans all of reduced
   in a loop over BLOCK_SIZE chunks (carrying the inter-chunk carry in a threadgroup variable).
   Internals: a transposed lds layout (`col = idx/EPT, row = idx%EPT` :144–145) gives each thread
   4 contiguous elements to scan serially → `plane_inclusive_sum` → a scan of the subgroup partials.
   Because it is a single workgroup, there is no inter-workgroup synchronization or forward-progress assumption at all.

4. `sort_scan_add` (kernels.rs:214–300): locally scans each BLOCK_SIZE chunk of counts and adds
   `reduced[group_id]` (the chunk's global base), updating counts into global exclusive offsets.

5. `sort_scatter` (kernels.rs:302–443): processes BLOCK_SIZE keys per workgroup in EPT=4 iterations
   (WG=256 keys per iteration):
   - **Local stable sort**: splits the 4-bit digit into two 2-bit steps. The trick:
     ```
     bit_key = (digit >> bit_shift) & 3
     packed_input = 1 << (bit_key * 8)                    // four 8-bit counters inside one u32  :355
     sg_inclusive = plane_inclusive_sum(packed_input)     // one simd scan ranks all 4 bins at once :357
     total = the packed sum over the whole workgroup      // :388
     bin_offsets = (total<<8)+(total<<16)+(total<<24)     // byte lane b = Σ_{a<b} total_a :389
     key_offset = ((bin_offsets + exclusive_at_thread) >> (bit_key*8)) & 0xFF   // :392
     ```
     Within the same bit_key the rank follows thread index order → **stable**.
   - Local histogram (threadgroup atomic, :404) → inclusive scan (:407–423) →
     global scatter with `total_offset = bin_offset_cache[digit] + (UNIT_POS - scratch[digit-1])` (:425–431).
     Each iteration then does `bin_offset_cache[digit] += hist[digit]` (:436–439).
   - Out-of-range lanes are padded with `local_key = 0xFFFFFFFF`, and writes are gated on `total_offset < num_keys`
     (:343, :432).

   Iteration order (by index) × the stable local split × the bin-major global offsets = **stable overall**.

Host loop (lib.rs:57–123): each pass allocates `count_buf [num_wgs*16]` and `reduced_buf
[ceil(num_reduce_wgs/BLOCK)*BLOCK]`, and swaps the keys/values double buffer.
`num_keys` is passed as a GPU buffer (`num_keys_buf`), which makes indirect dispatch possible;
the dispatch count is based on buffer capacity (capacity dispatch), and the kernel exits early when `group_id >= num_wgs`.

### Backward

Sorting has no backward. But the following **order contract** is a premise of gradient correctness:

- The rasterize forward walks the tile range `[range_start, range_end)` front→back (ascending depth),
  and the backward walks the same `flatten_ids` back→front.
- The sort outputs (`flatten_ids`/`tile_offsets`) must therefore be reused across fwd/bwd, and must be
  identical on every run for the same input (determinism) for training reproducibility to hold.

---

## Thread and memory configuration

### gsplat (CUDA)

- Key generation `intersect_tile_kernel`: 256 threads/block, one thread per Gaussian (or per nnz)
  (IntersectTile.cu:524–525). The 2-pass structure (count → cumsum → emit) derives the scatter position
  from the prefix sum — **no global atomic slot allocation → deterministic emission order**.
- Sorting: delegated to CUB's internal configuration (tuned per device). With `cub::DoubleBuffer`, the
  temporaries are just one extra copy each of keys/values plus CUB temp storage
  (comment at IntersectTile.cu:1076–1077).
- `intersect_offset_kernel`: 256 threads/block, one thread per isect.

### brush (CubeCL → WGSL/SPIR-V/MSL)

- All sort kernels: 1-D workgroups of 256 threads. count/scatter use `num_wgs` groups,
  reduce/scan_add use `num_reduce_wgs` groups, and scan uses **1 group**.
- threadgroup (shared) memory (for scatter, kernels.rs:323–329):
  `lds_keys[256]`, `lds_values[256]`, and `lds_scratch[256]` at 1KB each, `bin_offset_cache[256]` 1KB
  (only 16 actually used), `local_histogram[16]` 64B (atomic u32), `partials[32]` 128B,
  and `chunk_total[1]` — about 4.2KB total ≪ 32KB.
- subgroup operations: `plane_sum`, `plane_inclusive_sum`, and `plane_exclusive_sum`
  (corresponding 1:1 with MSL's `simd_sum`/`simd_prefix_*`).
- Global atomics: **none inside the sort kernels** (threadgroup atomics only) → the result is deterministic.
  (brush's source of nondeterminism is not the sort but the atomic slot compaction in `project_forward` —
  our port removes it by replacing it with prefix-sum compaction.)

---

## MSL porting design

### Adopted strategy

**Adopt the brush-style two-stage 32-bit stable radix argsort as the default path**, and verify against
the gsplat golden fixture by equivalence of "the tile range plus the (depth, gaussian_id) order within the tile".

Rationale:

1. Metal has no vendor library equivalent to CUB/`DeviceRadixSort` (MPS sorting supports neither
   key-value pairs nor bit-range limits) → an in-house kernel is mandatory.
2. A 64-bit key sort takes 16 passes with a 4-bit digit (12 passes even at 45 significant bits) and moves 12B
   (key8+val4) per pass. The two-stage approach totals 8+4=12 passes × 8B, so the bandwidth is smaller
   and the key buffer is half the size.
3. Consistent with what was measured and settled on M4/A17: the simdgroup width is fixed at 32 → a single
   subgroup scan path; threadgroup memory is 32KB → 8× headroom.

### Why OneSweep / decoupled-lookback are prohibited

NVIDIA OneSweep and decoupled-lookback scan are structured so that a later workgroup spin-waits on an
earlier workgroup publishing its aggregate/prefix flag, which presupposes **a forward-progress guarantee
between scheduled threadgroups**. CUDA guarantees this (in practice), but:

- Metal / Apple GPUs **make no documented guarantee** about inter-threadgroup execution order,
  fair scheduling, or forward progress. If a spin-waiting group occupies a core and the producer group is
  never scheduled, it deadlocks → GPU hang / watchdog reset.
- The portability layers reach the same conclusion: this is exactly why the compute-shader-101/FidelityFX
  family that brush builds on adopted reduce-then-scan instead of lookback when targeting wgpu
  (including the Metal backend) (brush-sort README).

The global scan is therefore decomposed into three stages: **(a) reduce → (b) a single-threadgroup scan →
(c) scan_add**. A single threadgroup can always make progress, so it is safe, and the scan input size is
`num_reduce_wgs = 16 * ceil(num_wgs/1024)` — e.g. 4M intersections → num_wgs=4096 →
64 reduced entries → the single-group scan cost is negligible.

### Constants and parameters

```metal
// SortConstants.h — identical to brush (corresponds to kernels.rs:7-20)
constant constexpr uint WG                  = 256;  // threadgroup thread count
constant constexpr uint BITS_PER_PASS      = 4;
constant constexpr uint BIN_COUNT          = 16;
constant constexpr uint ELEMENTS_PER_THREAD = 4;
constant constexpr uint BLOCK_SIZE          = WG * ELEMENTS_PER_THREAD; // 1024
constant constexpr uint SIMD_WIDTH          = 32;   // fixed by M4/A17 measurement
constant constexpr uint NUM_SIMDGROUPS      = WG / SIMD_WIDTH; // 8

struct RadixSortParams {
    uint shift;           // pass * BITS_PER_PASS
    uint num_wgs;         // ceil(capacity / BLOCK_SIZE) — dispatch based on capacity
    uint num_reduce_wgs;  // BIN_COUNT * ceil(num_wgs / BLOCK_SIZE)
};
```

`num_keys` is taken not as a constant but as a **device buffer `[1]`** (the GPU computes the intersection
count → capacity dispatch plus an in-kernel gate, with no CPU readback; the brush lib.rs:50 approach).
Optionally, a micro kernel that fills `MTLDispatchThreadgroupsIndirectArguments` can switch this to an
indirect dispatch (see the open questions).

### The five kernel signatures (encoded once per pass)

```metal
// 1) histogram: BLOCK_SIZE keys per threadgroup, bin-major output
kernel void radix_sort_count(
    device const uint*      num_keys   [[buffer(0)]],   // [1]
    device const uint*      src_keys   [[buffer(1)]],   // [capacity]
    device uint*            counts     [[buffer(2)]],   // [BIN_COUNT * num_wgs] bin-major
    constant RadixSortParams& p        [[buffer(3)]],
    uint  gid  [[threadgroup_position_in_grid]],
    ushort lid [[thread_position_in_threadgroup]]);
// threadgroup: atomic_uint hist[BIN_COUNT];
// barriers: one after zero-initializing hist, one after accumulation (threadgroup_barrier(mem_threadgroup))
// atomic: atomic_fetch_add_explicit(&hist[digit], 1, memory_order_relaxed)
// output: counts[digit * num_wgs + gid] — bin-major, essential for keeping gsplat-style determinism

// 2) chunk sums: the reduce of reduce-then-scan
kernel void radix_sort_reduce(
    device const uint* num_keys, device const uint* counts,
    device uint* reduced,                                 // [align(num_reduce_wgs, BLOCK_SIZE)]
    constant RadixSortParams& p, uint gid, ushort lid,
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_id   [[simdgroup_index_in_threadgroup]]);
// threadgroup: uint partials[NUM_SIMDGROUPS];
// simd_sum → partials[simd_id] → barrier → simdgroup 0 totals them with simd_sum

// 3) global exclusive scan: MUST be dispatchThreadgroups(1)
kernel void radix_sort_scan(
    device const uint* num_keys, device uint* reduced,
    constant RadixSortParams& p, ushort lid, ushort simd_lane, ushort simd_id);
// threadgroup: uint partials[NUM_SIMDGROUPS]; uint lds[BLOCK_SIZE]; uint chunk_total;
// loop over BLOCK_SIZE chunks + carry (threadgroup variable). 3 barriers per chunk
// (after loading lds / after scanning partials / after rewriting lds). Uses simd_prefix_inclusive_sum.

// 4) propagate the scan result into counts
kernel void radix_sort_scan_add(
    device const uint* num_keys, device const uint* reduced, device uint* counts,
    constant RadixSortParams& p, uint gid, ushort lid, ushort simd_lane, ushort simd_id);
// counts[...] = local exclusive scan + reduced[gid]  (corresponds to kernels.rs:214-300)

// 5) local stable sort + global scatter
kernel void radix_sort_scatter(
    device const uint* num_keys,
    device const uint* src_keys,  device const uint* src_vals,
    device const uint* counts,                            // global exclusive offsets (bin-major)
    device uint*       dst_keys,  device uint*       dst_vals,
    constant RadixSortParams& p, uint gid, ushort lid, ushort simd_lane, ushort simd_id);
// threadgroup: uint lds_keys[WG]; uint lds_vals[WG]; uint lds_scratch[BIN_COUNT];
//              uint bin_offset_cache[BIN_COUNT]; atomic_uint hist[BIN_COUNT];
//              uint partials[NUM_SIMDGROUPS]; uint chunk_total;
//              about 2.3KB total ≪ 32KB
// EPT=4 iterations. Inside each: 2-bit split × 2 (packed 8-bit counters + simd_prefix_inclusive_sum,
//   exactly the formulas of kernels.rs:351-401), 2 barriers per split.
// End of each iteration: hist atomic accumulation → barrier → scratch inclusive scan → barrier →
//   global write (dst[bin_offset_cache[digit] + local_offset], gated on num_keys) →
//   bin_offset_cache += hist → barrier.
// Padding lanes: local_key = 0xFFFFFFFFu, naturally dropped by the write gate.
```

### Host orchestration (the KernelBackend layer)

```
radix_argsort(keys: SXTBuffer<u32>, vals: SXTBuffer<u32>, sorting_bits: u32):
    for pass in 0 ..< ceil(sorting_bits / 4):
        p.shift = pass * 4
        encode: count → reduce → scan(1 tg) → scan_add → scatter
        swap(cur, dst)   // double buffer; mind which buffer holds the final result when the pass count is odd
    return (cur_keys, cur_vals)
```

Temporary buffers: `counts [16 * num_wgs] u32`, `reduced [align(num_reduce_wgs, 1024)] u32`, and one
extra copy each of the keys/values double buffer. Encode every pass into a single `MTLCommandBuffer`
(the compute pass boundary is a sufficient barrier between passes).

### Pipeline integration (the two-stage strategy, mirroring brush render.rs)

```
1. project_forward: visibility decision + depth computation. Compaction uses the prefix sum of the
   visibility mask rather than atomic slots → deterministic from the presort id onward. (the gsplat 2-pass approach)
2. radix_argsort(depth_bits, presort_gid, 32)            // 8 passes
   - key = as_type<uint>(depth), premise depth >= 0 (port the assert into the CPU reference)
3. per-gaussian tile count (porting gsplat's intersect_tile first pass) → prefix_sum
4. map_gaussians: emit (tile_id, compact_gid). Slots are derived from the prefix sum (deterministic).
   Count mismatches are padded with sentinel_tile_id = n_tiles (the map_gaussians.rs:56 approach)
5. bits = bit_width(n_tiles)  // sentinel included. e.g. 8160 tiles → 13 → 4 passes
   radix_argsort(tile_id_from_isect, compact_gid_from_isect, bits)
6. tile_offsets: gsplat's intersect_offset_kernel or brush's get_tile_offsets approach
   ([start,end) pairs, with the sentinel excluded by a tid < n_tiles gate)
```

### Determinism guarantee checklist

| Site | Rule |
|---|---|
| Compaction / emission slots | No global atomic counters; derive slots from the prefix sum only |
| Each radix pass | bin-major counts + index-order local ranking = stable, bit-identical across runs |
| Global scan | Serial chunks in a single threadgroup — no floating point, fixed order |
| Tied keys | Two stable stages: within a tile, lexicographic by (depth bits, previous-stage order = gaussian index) |
| Verification | 100 runs on identical input → identical flatten_ids hash (CI test) |

### Equivalence verification rules against the gsplat golden fixture

The gsplat result (a stable radix over 64-bit keys) and our two-stage result must have
**identical within-tile ordering, namely (depth_bits asc, gaussian_index asc)**. Verification:

1. Generate the expected values in the C++ CPU reference with the gsplat approach (64-bit keys + `std::stable_sort`).
2. Cut the Metal two-stage result into per-tile ranges with tile_offsets and compare `flatten_ids` range by range.
3. Deliberately include a tie case with exactly equal depth bits in the fixture
   (three Gaussians of identical depth landing in the same tile).

---

## The tile-local sorting alternative (conceptual design — no external code consulted)

An alternative that skips the global radix sort and sorts each tile's segment independently. What follows
is this document author's conceptual design; no code from the prohibited-list repositories was consulted.

### Structure

```
1. per-gaussian tile count → a per-tile count histogram (tile_counts[n_tiles], pre-aggregated in the
   threadgroup + a device atomic add)
2. prefix sum of tile_counts → the tile segment start offsets (securing tile_offsets without a global sort)
3. scatter: each Gaussian writes a (depth_bits, gaussian_id, …) entry into its own tile segment. The slot
   within a segment comes from a per-tile atomic cursor — the order at this point is nondeterministic.
4. tile_local_sort kernel: one threadgroup = one tile. Load the segment into threadgroup memory and sort
   by the composite key (depth_bits, gaussian_id) — a comparison-based bitonic / odd-even merge sort.
   The gaussian_id tie-break in the composite key cancels the nondeterministic order from step 3, so the
   final result is deterministic.
```

### Constraints and risks

- **Capacity**: 32KB of threadgroup memory ÷ (key u64 + val u32 = 12B) ≈ 2,730 entries per tile.
  Tiles that exceed it need either (a) chunked sorting plus a merge pass, or (b) a fallback to the global
  radix for that tile alone — the point where complexity spikes.
- **Load imbalance**: if a few Gaussians with large splats create tiles of thousands of entries, that
  threadgroup dominates the overall latency. Bitonic is O(k log² k), which is unfavorable for large k.
- **Memory pattern of the atomic scatter**: the step-3 scatter is a random write — whether Apple TBDR's
  SLC absorbs it needs to be measured.
- Upside: the 12 global passes (12 re-movements of keys+values) disappear, and the isect data moves only
  twice in total (one scatter + one within the sort). It may be favorable early in training, when average
  tile occupancy is low (few splats).

### A/B bench plan

| Item | Contents |
|---|---|
| Variants | A: two-stage global radix (4-bit digit) / A8: the same structure with an 8-bit digit (BIN=256, half the passes) / B: tile-local bitonic / C: hybrid (one global tile-sort + a tile-local depth-sort) |
| Devices | M4 Air (macOS), iPhone 15 Pro A17 (iOS, sustained load) |
| Data | The golden fixture scene + 3 synthetic distributions (uniform / tile-concentrated hot spot / early-training-style few large splats), N = 0.25M/1M/3M Gaussians, resolutions 1080p/4K, intersection counts swept from 1M to 30M |
| Metrics | Sort-stage GPU ms (MTLCounterSampleBuffer), whole-frame ms, peak memory, determinism (100-run hash), and ms after clock throttling under 10 minutes of sustained load on iPhone |
| Decision | Default is A. If A8 beats A, promote the digit width. B/C are adopted only if they reach ≥1.3× on both devices and pass the determinism and capacity-fallback tests |
| Regression guard | Record the linear regression coefficient of sort ms against intersection count in CI; fail on a regression exceeding 10% |

---

## Open questions

1. **Digit width 4 vs 8**: an 8-bit digit (BIN=256) halves the pass count (32 bits: 8→4), but the
   scatter's local ranking grows to four 2-bit splits and threadgroup usage rises by about 3KB.
   Which one wins on Apple GPUs is decided by the A/A8 bench (the FidelityFX original uses 4-bit,
   CUB is in the 6–8 bit family).
2. **When to switch to indirect dispatch**: start with capacity dispatch (plus the in-kernel gate), and
   switch to `dispatchThreadgroups(indirectBuffer:)` if the idle-threadgroup overhead measures as
   significant from mid-training onward, when the intersection count fluctuates widely.
   The encoding overhead on iOS has not been measured.
3. **Segmented sort support**: if support matching gsplat's `DeviceSegmentedRadixSort`
   (multi-image batched training) becomes necessary — deferred, since the Phase 1 scope is I=1.
   Under the two-stage strategy it needs verifying whether concatenating the image bits above the
   tile key (`bits += image_n_bits`) is sufficient.
4. **Where to enforce the depth ≥ 0 invariant**: gsplat uses an `assert` (on device); MSL has no
   assert, so decide between the CPU reference and a debug-build validation kernel.
   (If a negative depth gets in, the float-bit ordering inverts — a silent image-quality bug.)
5. **brush's num_keys=capacity padding vs an exact n_isects readback**: the training loop has a
   loss readback anyway, which may hide the cost of an n_isects readback. Profiling is needed to see
   whether the capacity approach matters only on the render-only (viewer) path.
6. **The 67M+ intersection scale**: brush keeps a regression test for a scan chunk-boundary bug at
   around 67M keys (lib.rs:291–339). Our scan kernel must likewise allocate the `reduced` buffer aligned
   to a multiple of BLOCK_SIZE and port the same-scale test — confirm whether that scale is reachable
   at 3M Gaussians × 4K resolution.

---

## Verification notes (adversarial verifier)

The upstream originals (gsplat `77ab983`, brush `3b80985`) were re-compared independently. The key
encoding, the bit-width formulas, the three sort calls, the offset kernel, the structure of brush's five
kernels, the two-stage strategy, the sentinel handling, the constants, and the file:line citations were all
confirmed to match the upstream originals (bits_for_count MathUtils.h:23–33,
end_bit IntersectTile.cu:1102/1157, key emission 197/457, tile_id 454, assert 283,
offsets construction Intersect.cpp:263–264, brush constants kernels.rs:7–20, bin-major
kernels.rs:63, sentinel map_gaussians.rs:56, the bits computation render.rs:227,
the `tid < num_tiles` gate of get_tile_offsets at get_tile_offset.rs:34–52,
the atomic compaction in project_forward at project_forward.rs:122, and so on). The carry/borrow behavior
of the packed 8-bit counter trick (kernels.rs:355–392) was also recomputed down to the WG=256 boundary
cases, and the spec's formulas are accurate.

The following corrections and supplements are recorded:

### Correction 1 (correctness — risk of a missing barrier): `radix_sort_scan` takes 5 barriers per chunk, not 3

The spec body's "3 barriers per chunk (after loading lds / after scanning partials / after rewriting lds)"
is missing two relative to the upstream original. The chunk loop of brush's `sort_scan_kernel` contains
`sync_cube()` **5 times** (kernels.rs:152, 165, 190, 197, 207):

1. :152 — after the transposed lds load
2. :165 — **after the last lane of the simdgroup stores `partials`** (before subgroup 0 reads
   partials; omitting it races on the partials read)
3. :190 — after scanning partials and writing `chunk_total`
4. :197 — after rewriting `lds += base` (before re-reading the transposed positions)
5. :207 — after rewriting `reduced`, at the **end of the chunk** (before the next chunk overwrites
   lds/partials/chunk_total; omitting it creates a hazard across loop iterations)

All five points need `threadgroup_barrier(mem_flags::mem_threadgroup)` in the MSL port.

### Correction 2 (correctness — risk of a missing barrier): `radix_sort_scatter` takes 3 barriers per 2-bit split, not 2

The spec's "2 barriers per split" is insufficient. brush has `sync_cube()` **3 times** per split
iteration (the while bit_shift loop) (kernels.rs:361, 386, 396):

1. :361 — after storing `partials` (before the scan reads it)
2. :386 — after scanning partials and writing `chunk_total`
3. :396 — after the scattered writes into `lds_keys`/`lds_values` (before re-reading at the `UNIT_POS` position)

### Correction 3 (correctness — one missing barrier): separating the read and the update of `bin_offset_cache` at the end of a scatter iteration

Implementing the spec's end-of-iteration order — "scratch inclusive scan → barrier → global write
(`dst[bin_offset_cache[digit] + local_offset]`) → `bin_offset_cache += hist` → barrier" —
means that while lanes 0–15 update `bin_offset_cache` (+= hist), other threads may still be reading the
same array for their global writes (a race). brush **reads it into a register first and separates the two
with a barrier** (kernels.rs:425–426, 440):

```
global_offset = bin_offset_cache[key_index];   // :425  read ahead into a register
sync_cube();                                   // :426  ← the barrier missing from the spec
... total_offset = global_offset + local_offset; global write (:427-435)
bin_offset_cache[UNIT_POS] += hist[UNIT_POS];  // :436-439 (UNIT_POS < 16)
sync_cube();                                   // :440  end of the iteration
```

The MSL kernel must do the same: read the cache ahead → barrier → global write + cache update → barrier.

### Correction 4 (descriptive accuracy): the scan's carry is a thread register, not a threadgroup variable

The spec's "carry (threadgroup variable)" differs from the upstream original. In brush the only
threadgroup variable is `chunk_total[1]`; the carry is a **thread-local register**
(`let mut carry = 0u32;` kernels.rs:139). Every thread accumulates the same value with
`carry += chunk_total[0]` (kernels.rs:209) after the end-of-chunk barrier, so convergence is guaranteed.
The register approach is simpler because it needs no single-writer barrier — implement it with a register
in the MSL port as well.

### Correction 5 (descriptive accuracy): the brush host loop allocates fresh buffers every pass rather than "swapping" a double buffer

Contrary to the spec's "swaps the keys/values double buffer", brush's lib.rs **newly allocates**
`output_keys`/`output_values` every pass and reassigns them (lib.rs:105–106, 121–122;
`count_buf` and `reduced_buf` are also allocated anew per pass, lib.rs:58, 76–77). It relies on the
allocator pool. The MSL design's choice of "one extra copy each of keys/values plus a swap" is a valid
design decision (reuse is more efficient), but it was inaccurate as a description of brush. brush's
reduced buffer size also carries a `.max(1)` guard (lib.rs:76) — port that into MSL as an n=0 defense.

### Supplements (not errors)

- The `image_n_bits + tile_n_bits <= 32` TORCH_CHECK exists twice: at Intersect.cpp:219–228 and again at
  IntersectTile.cu:513–522 (a double defense at host and launcher).
- The `flatten_ids` write on the AccuTile path uses the helper parameter `flatten_idx`
  (IntersectTile.cu:198), but the caller passes `idx` (IntersectTile.cu:363), so the spec's
  `(int32_t)idx` notation is semantically accurate on both paths.
- The gsplat key emission site has a `key_start`/`key_end` filter for multi-GPU chunking
  (IntersectTile.cu:188–192). On a single device, `key_start=0` and
  `key_end=I*n_tiles` (IntersectTile.cu:509–510) let the whole range through — the Phase 1 port can omit
  the filter (or hold it constant over the whole range) and remain equivalent.
- The scatter's packed 8-bit counter trick is coupled to the **premise WG=256** (even when the per-bin
  local rank and partial sums reach 256, the carries/borrows of u32 add/subtract cancel, so the 8-bit lane
  extraction stays correct). Changing WG away from 256 requires re-examining this trick's premise.
- The two-stage cost comparison ("8+4=12 passes") is conservative: the 8 passes of stage 1 move an array
  of size `num_visible` (≤N), and only the 4 passes of stage 2 move an array of size `n_isects`, so the
  actual bandwidth gain is larger than the body states.
- Refinement of the "MPS sorting" footnote: MPSGraph does have `sort`/`argSort`, but stability is not
  documented and there is no bit-range limit or key-value pair sort. Since determinism (stability) is a
  requirement, the body's conclusion that an in-house kernel is mandatory stands.

No other formula, constant, or citation errors were found. With the corrections above applied, this
document is sufficient as the sole reference for the Metal implementation.

### Second re-verification (independent re-comparison)

The entire document, including the first-pass verification notes, was independently compared again against
the upstream originals (gsplat `77ab983`, brush `3b80985` — commit hashes re-confirmed with
`git rev-parse HEAD`).

**First-pass Corrections 1–5 were all settled against the upstream original (CONFIRMED):**

- Correction 1: `sort_scan_kernel` has exactly 5 `sync_cube()` calls — kernels.rs:152, 165, 190, 197, 207. Confirmed.
- Correction 2: 3 per scatter split iteration — kernels.rs:361, 386, 396. Confirmed.
- Correction 3: read `bin_offset_cache` ahead (:425) → `sync_cube()` (:426) → global write (:427–435) →
  cache update (:436–439) → `sync_cube()` (:440). Confirmed.
- Correction 4: `let mut carry = 0u32;` kernels.rs:139 (a thread register), `carry += chunk_total[0]` :209. Confirmed.
- Correction 5: fresh allocation per pass at lib.rs:58 (count_buf), :76–77 (reduced_buf, with the `.max(1)`
  guard at :76), :105–106 (output), :121–122 (reassignment). Confirmed.
- The supplement's WG=256 boundary argument for the packed 8-bit counter was recomputed as well — the lane
  value `pre_a + excl_a ≤ 256`, and equality holds only when "every element is in bin ≤ a", in which case
  no thread extracts a higher lane, so the carry never contaminates the result. Accurate.

**Additional Correction 6 (descriptive accuracy — brush's readback approach was described incorrectly):**

The body's "num_keys is passed as a GPU buffer … capacity dispatch with no CPU readback (the brush
lib.rs:50 approach)" and open question 5's "brush's num_keys=capacity padding" differ from brush's actual
shipping path. brush synchronously reads two atomic counts back to the CPU **before** the sort —
the render.rs:145 comment "Read both atomic counts in one transaction BEFORE the sort", with the
transaction at :146–167. It then sizes the sort inputs to the actual counts:
the depths/presort_gid slices `[0..num_visible_sz]` (render.rs:174–179) and
`buffer_size = max(num_intersections, 1)` (render.rs:210–212). The `max_n` held by `num_keys_buf` at
lib.rs:50 is therefore not a capacity but the **actual count (= the input tensor length)**.
It is true that the kernel gate (`group_id >= num_wgs` — kernels.rs:40, 78, 225, 316) and the indirect
dispatch mentioned in the brush-sort README *support* readback-free operation, but brush neither verified
nor operated that configuration. The MSL design of "capacity dispatch + kernel gate, no readback" is itself
valid given the kernel structure, but recognize that it is **a new combination, not one demonstrated in
brush**, and add separate tests for the num_keys < capacity case (a partially occupied last workgroup plus
entirely idle workgroups). Open question 5's comparison should be read as "our capacity approach (new) vs
the readback + exact-size approach brush actually uses (verified)".

**Two additional supplements (not errors):**

- The scatter's zero-initialization of `local_histogram` happens **at the start of every EPT iteration**
  (kernels.rs:338–341), and its visibility depends not on a dedicated barrier but on the `sync_cube()`
  inside the split loop (:361 and others). There is also one barrier immediately after the initial load of
  `bin_offset_cache` at kernel entry (:332–334) — neither is spelled out in the body's kernel signature
  comments. If the MSL port alters the split structure (e.g. the A8 variant's four 2-bit splits → a
  different ranking scheme), always confirm that at least one `threadgroup_barrier` remains between the
  zero-initialization and the atomic accumulation at the point corresponding to :404.
- The tile_id on the AccuTile path takes an axis-swapped form:
  `tile_id = isY ? (u * tile_width + v) : (v * tile_width + u)` (IntersectTile.cu:187,
  `isY = y_span < x_span` :337). The resulting value is the same row-major index
  (`row * tile_width + col`) as the AABB path cited in the body (:454), so the body's description is
  semantically accurate — just note when porting AccuTile that u/v swap their column/row roles depending on isY.

Second re-verification conclusion: apart from additional Correction 6 and the two supplements, there are no
formula, constant, sign, index, clamp, or citation errors. With the first- and second-pass corrections
applied, this document is sufficient as the sole reference for the Metal implementation.
