// IntersectTile.metal — isect_tile_count / isect_tile_emit / isect_offset_encode
// (spec 03, D4 naming).
// Derived from nerfstudio-project/gsplat (Apache-2.0), commit 77ab983.
// Reimplemented for Metal/C++.
// Original: gsplat/cuda/csrc/IntersectTile.cu:83–207 (the SNUGBOX+AccuTile helpers — SpeedySplat
//           arXiv:2412.00578), :213–464 (intersect_tile_kernel — count and emit are one kernel
//           that branches on the presence of cum_tiles_per_gauss; the MSL splits it into two
//           kernel variants to avoid binding a null buffer — spec 03 §Kernel 3); the 2-pass host
//           orchestration is gsplat/cuda/csrc/Intersect.cpp:230–296 (MetalBackend::isect_tiles),
//           gsplat/cuda/csrc/IntersectTile.cu:925–988 (intersect_offset_kernel →
//           isect_offset_encode; launch :990–1026, host-side 0 fill when n_isects==0 :1004–1008).
//
// The single-device premise (spec 03 §MSL porting design): the key_start/key_end window and
// row_key_column_range are identities and are removed; packed/tile_mask are excluded
// (disposition #14 — dense only, iid = idx / N).
// No atomics and no threadgroup barriers — emit writes only within its own range
// [cum[idx-1], cum[idx]).
// The AccuTile path (kIsectAccuTile=true — IntersectTile.cu:288–372): the producer of the golden is
// the CPU reference (isect_tiles_cpu; torch has no AccuTile — 07 Correction 2), and exact integer
// agreement is the contract.
// Parity points (D2, disposition #11): every division, sqrt, and log that determines a tile boundary
// is precise:: — matching the CPU IEEE `/`, std::sqrt, and logf. The expression forms are kept
// identical to the CPU side (CpuKernels.cpp accutile_*) so the fast-math FMA contraction decisions
// cannot diverge. The `t <= 0` early return is an approved deviation, disposition #13
// (see the accutile_element comment).
// The original assert(depth_f >= 0.f) (IntersectTile.cu:283) is absent from the MSL — the CPU
// reference (isect_tiles_cpu) replaces it with a validation throw (spec 03 §Kernel 3).
#include <metal_stdlib>
#include "KernelConstants.h"
#include "FunctionConstants.h"
#include "KernelParams.h"
#include "KernelCommon.h"
using namespace metal;

// The AABB tile rectangle (IntersectTile.cu:377–387) — both the count and emit passes MUST share
// this single formula (the invariant that emit never writes more than count — spec 03 §Algorithm
// overview).
// The divisions are precise::divide — bit-identity with the IEEE f32 division of the CPU reference
// is the premise for exact agreement of the integer tile boundaries (floor/ceil) (a D2 parity
// point). tmin is inclusive, tmax exclusive (:382).
struct TileRect {
    int2 tmin;
    int2 tmax;
};

static inline TileRect aabb_tile_rect(const float2 mean2d, const float radius_x,
                                      const float radius_y, constant IsectTileParams& p)
{
    const float tile_size_f   = (float)p.tile_size;
    const float tile_radius_x = precise::divide(radius_x, tile_size_f); // :377
    const float tile_radius_y = precise::divide(radius_y, tile_size_f); // :378
    const float tile_x        = precise::divide(mean2d.x, tile_size_f); // :379
    const float tile_y        = precise::divide(mean2d.y, tile_size_f); // :380
    TileRect r;
    r.tmin.x = min(max(0, (int)floor(tile_x - tile_radius_x)), (int)p.tile_width);  // :384
    r.tmin.y = min(max(0, (int)floor(tile_y - tile_radius_y)), (int)p.tile_height); // :385
    r.tmax.x = min(max(0, (int)ceil(tile_x + tile_radius_x)), (int)p.tile_width);   // :386
    r.tmax.y = min(max(0, (int)ceil(tile_y + tile_radius_y)), (int)p.tile_height);  // :387
    return r;
}

// ===== AccuTile — SNUGBOX + the ellipse-line sweep (SpeedySplat arXiv:2412.00578, a literal
// translation of the IntersectTile.cu:83–207 helpers + the :288–372 body) =====
// Identical to the CPU reference (CpuKernels.cpp accutile_*) down to the form of the expressions —
// the divisions, sqrt, and log are precise:: (a D2 parity point, disposition #11: matching the CPU
// IEEE `/`, std::sqrt, and logf). float min/max use fmin/fmax (the CUDA device
// min(float,float)=fminf semantics, matching the CPU std::fmin/std::fmax).

// Ellipse-vertical-line intersection (accutile_ellipse_intersection, IntersectTile.cu:83–94) — in
// the sweep frame, the v coordinates {v_lo, v_hi} of the two intersections of the line u = coord
// with the level set q = t. Non-negativity of the discriminant is guaranteed by the caller's guards
// (:145, :158 — |h| <= extent_u) (derivation in spec 03 §3).
static inline float2 accutile_ellipse_intersection(float A, float B, float C, float disc,
                                                   float t, float2 p, bool isY, float coord)
{
    const float p_u = isY ? p.y : p.x;   // :86
    const float p_v = isY ? p.x : p.y;   // :87
    const float coeff = isY ? A : C;     // :88
    const float h = coord - p_u;         // :90
    const float sqrt_term = precise::sqrt(disc * h * h + t * coeff); // :91
    return float2(precise::divide(-B * h - sqrt_term, coeff) + p_v,   // :93 v_lo
                  precise::divide(-B * h + sqrt_term, coeff) + p_v);  //     v_hi
}

// The strip sweep (accutile_process_tiles, IntersectTile.cu:96–207) — count and emit share this
// single formula (the invariant that emit never writes more than count). EMIT is a compile-time
// branch — it corresponds to the original's `isect_ids != nullptr` gate (:195) (the count
// specialization takes null pointers, and EMIT=false removes that code as dead). The key-window
// skip (:189–192) always passes on a single device and is removed.
template <bool EMIT>
static inline uint accutile_process_tiles(float A, float B, float C, float disc, float t,
                                          float2 p, float2 bbox_min, float2 bbox_max,
                                          float2 bbox_argmin, float2 bbox_argmax, int2 rect_min,
                                          int2 rect_max, float BLOCK, int tile_width, bool isY,
                                          ulong iid_enc, ulong depth_id_enc, uint flatten_idx,
                                          device ulong* isect_ids, device int* flatten_ids,
                                          thread long& cur_idx)
{
    if (isY) { // :127–135 sweep-frame swap — from here on u = the sweep axis (the shorter one), v = orthogonal
        rect_min    = int2(rect_min.y, rect_min.x);
        rect_max    = int2(rect_max.y, rect_max.x);
        bbox_min    = float2(bbox_min.y, bbox_min.x);
        bbox_max    = float2(bbox_max.y, bbox_max.x);
        bbox_argmin = float2(bbox_argmin.y, bbox_argmin.x);
        bbox_argmax = float2(bbox_argmax.y, bbox_argmax.x);
    }

    uint tiles_count = 0;                                // :137
    float2 intersect_min_line, intersect_max_line;
    float ellipse_min, ellipse_max;
    float min_line, max_line;

    intersect_max_line = float2(bbox_max.y, bbox_min.y); // :142 neutral values (v_max as the min candidate, v_min as the max)

    min_line = rect_min.x * BLOCK;                       // :144
    if (bbox_min.x <= min_line)                          // :145 the line is inside the left edge of the ellipse — valid
        intersect_min_line = accutile_ellipse_intersection(A, B, C, disc, t, p, isY, min_line);
    else
        intersect_min_line = intersect_max_line;         // :151 the line is outside the ellipse on the left → neutral

#pragma clang loop unroll(disable)
    for (int u = rect_min.x; u < rect_max.x; ++u) {      // :155 (#pragma unroll 1)
        max_line = min_line + BLOCK;                     // :157
        if (max_line <= bbox_max.x)                      // :158 update only when inside the right side of the ellipse
            intersect_max_line = accutile_ellipse_intersection(A, B, C, disc, t, p, isY, max_line);

        if (min_line <= bbox_argmin.y && bbox_argmin.y < max_line) // :163 tangent point inside the strip
            ellipse_min = bbox_min.y;                              // :165 → the bbox extremum
        else
            ellipse_min = fmin(intersect_min_line.x, intersect_max_line.x); // :169

        if (min_line <= bbox_argmax.y && bbox_argmax.y < max_line) // :172
            ellipse_max = bbox_max.y;                              // :174
        else
            ellipse_max = fmax(intersect_min_line.y, intersect_max_line.y); // :178

        const int min_tile_v =
            max(rect_min.y, min(rect_max.y, (int)(precise::divide(ellipse_min, BLOCK))));       // :181
        const int max_tile_v =
            min(rect_max.y, max(rect_min.y, (int)(precise::divide(ellipse_max, BLOCK) + 1.f))); // :182

#pragma clang loop unroll(disable)
        for (int v = min_tile_v; v < max_tile_v; ++v) {  // :185 (#pragma unroll 1)
            ++tiles_count;                               // :193
            if (EMIT) {                                  // :195 only the emit specialization writes
                const ulong tile_id = (ulong)(isY ? u * tile_width + v : v * tile_width + u); // :187
                isect_ids[cur_idx]   = iid_enc | (tile_id << 32) | depth_id_enc;              // :197
                flatten_ids[cur_idx] = (int)flatten_idx;                                      // :198
                ++cur_idx;                                                                    // :199
            }
        }

        intersect_min_line = intersect_max_line;         // :203
        min_line = max_line;                             // :204
    }
    return tiles_count;                                  // :206
}

// AccuTile element processing (IntersectTile.cu:288–367) — builds the level set, the SNUGBOX, and
// the rect from the conic and the opacity, then calls the sweep. The count specialization
// (EMIT=false) uses only the return value (corresponding to first_pass at :364).
template <bool EMIT>
static inline uint accutile_element(const device float* conic /*[3] stride 12B — element access*/,
                                    float opacity, float2 mean2d, constant IsectTileParams& p,
                                    ulong iid_enc, ulong depth_id_enc, uint flatten_idx,
                                    device ulong* isect_ids, device int* flatten_ids,
                                    thread long& cur_idx)
{
    const float A = conic[0], B = conic[1], C = conic[2]; // :293–295 the triu of Σ⁻¹
    const float disc = B * B - A * C;                     // :298 = -det(Σ⁻¹) < 0 (when positive definite)

    // The opacity-aware level set (:300–303): alpha = opacity·exp(-0.5q) >= α_thresh ⇔
    // q <= 2·ln(opacity/α_thresh), capped at GAUSSIAN_EXTEND² (the same radius budget as the
    // projection). CUDA __logf ↔ Metal precise::log ↔ CPU logf are unified (disposition #11).
    const float t = fmin(kGaussianExtend * kGaussianExtend,
                         2.0f * precise::log(precise::divide(opacity, kAlphaThreshold)));
    // Disposition #13 (an approved deviation): with opacity < α_thresh, t < 0 → the sqrt argument
    // goes negative → NaN. The upstream original is unguarded because it assumes the upstream
    // (projection) culling (caution in spec 03 §2) — here there is a `t <= 0` early return
    // (0 tiles). For valid inputs (opacity > α_thresh) the result is unchanged.
    if (t <= 0.f) return 0;

    // SNUGBOX — the tight AABB of the ellipse (:305–316). x_extent = sqrt(t·C/(AC-B²)) (a Lagrange derivation).
    const float neg_t_over_disc = precise::divide(-t, disc);     // :306 > 0
    const float x_extent = precise::sqrt(neg_t_over_disc * C);   // :307
    const float y_extent = precise::sqrt(neg_t_over_disc * A);   // :308
    const float2 bbox_min = float2(mean2d.x - x_extent, mean2d.y - y_extent); // :310
    const float2 bbox_max = float2(mean2d.x + x_extent, mean2d.y + y_extent); // :311
    const float Bx_over_C = precise::divide(B * x_extent, C);    // :313
    const float By_over_A = precise::divide(B * y_extent, A);    // :314
    // Tangent point coordinates (:315–316) — .x = the y of the leftmost/rightmost point,
    // .y = the x of the topmost/bottommost point
    const float2 bbox_argmin = float2(mean2d.y + Bx_over_C, mean2d.x + By_over_A);
    const float2 bbox_argmax = float2(mean2d.y - Bx_over_C, mean2d.x - By_over_A);

    // Tile rectangle clamp (:318–324) — the (int) cast truncates toward zero (different from the
    // floor/ceil of the AABB path — Task 3 review note). The max(0,·) clamp makes the negative case
    // harmless.
    const float tile_size_f = (float)p.tile_size;                // :318
    int2 rect_min, rect_max;
    rect_min.x = max(0, min((int)p.tile_width,
                            (int)(precise::divide(bbox_min.x, tile_size_f))));        // :319
    rect_min.y = max(0, min((int)p.tile_height,
                            (int)(precise::divide(bbox_min.y, tile_size_f))));        // :321
    rect_max.x = max(0, min((int)p.tile_width,
                            (int)(precise::divide(bbox_max.x, tile_size_f) + 1.f)));  // :323
    rect_max.y = max(0, min((int)p.tile_height,
                            (int)(precise::divide(bbox_max.y, tile_size_f) + 1.f)));  // :324

    const int y_span = rect_max.y - rect_min.y; // :326
    const int x_span = rect_max.x - rect_min.x; // :327
    if (y_span * x_span == 0) return 0;         // :328–335 empty rect → 0 tiles

    const bool isY = y_span < x_span;           // :337 the shorter axis becomes the outer sweep loop
    return accutile_process_tiles<EMIT>(A, B, C, disc, t, mean2d, bbox_min, bbox_max,
                                        bbox_argmin, bbox_argmax, rect_min, rect_max,
                                        tile_size_f, (int)p.tile_width, isY, iid_enc,
                                        depth_id_enc, flatten_idx, isect_ids, flatten_ids,
                                        cur_idx); // :340–367 (BLOCK = (float)tile_size, :125)
}

// Pass 1 (count) — intersect_tile_kernel first_pass (IntersectTile.cu:255–264, AccuTile
// :288–372 / AABB :400–432). One thread = one element (idx ∈ [0, I*N)).
// Dispatch: dispatchThreads(n_elements, tg 256) (:524–525).
kernel void isect_tile_count(
    device const float* means2d         [[buffer(0)]], // [n_elements*2]
    device const int*   radii           [[buffer(1)]], // [n_elements*2] int32
    device const float* conics          [[buffer(2), function_constant(kIsectAccuTile)]], // [n_elements*3] stride 12B
    device const float* opacities       [[buffer(3), function_constant(kIsectAccuTile)]], // [n_elements]
    device int*         tiles_per_gauss [[buffer(4)]], // [n_elements] output
    constant IsectTileParams& p         [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= p.n_elements) return;                  // :249 bounds guard
    const float radius_x = (float)radii[idx * 2];     // :255 int32 → float, loaded into a local
    const float radius_y = (float)radii[idx * 2 + 1]; // :256
    if (radius_x <= 0.f || radius_y <= 0.f) {         // :257–264 culling (count 0)
        tiles_per_gauss[idx] = 0;
        return;
    }
    const float2 mean2d = float2(means2d[2 * idx], means2d[2 * idx + 1]); // :266
    if (kIsectAccuTile) {
        // AccuTile count — first_pass writes only the sweep return value (:340–372; the emit
        // arguments are null and EMIT=false removes that code as dead).
        long cur_idx = 0; // :338 first_pass → 0 (unused)
        tiles_per_gauss[idx] = (int)accutile_element<false>(
            conics + idx * 3, opacities[idx], mean2d, p, 0ul, 0ul, idx,
            (device ulong*)nullptr, (device int*)nullptr, cur_idx);
        return;
    }
    const TileRect r = aabb_tile_rect(mean2d, radius_x, radius_y, p);
    // Single device with no mask: the sum of the per-row ranges = the closed-form rectangle area (:400–431)
    tiles_per_gauss[idx] = (r.tmax.y - r.tmin.y) * (r.tmax.x - r.tmin.x); // :430
}

// Pass 2 (emit) — intersect_tile_kernel !first_pass (IntersectTile.cu:266–286, AccuTile
// :288–367 / AABB :434–462). The emit order (deterministic): ascending element idx → (AccuTile) the
// sweep axis u → the orthogonal axis v / (AABB) ascending tile y → ascending x.
kernel void isect_tile_emit(
    device const float* means2d             [[buffer(0)]], // [n_elements*2]
    device const int*   radii               [[buffer(1)]], // [n_elements*2] int32
    device const float* depths              [[buffer(2)]], // [n_elements]
    device const float* conics              [[buffer(3), function_constant(kIsectAccuTile)]], // [n_elements*3] stride 12B
    device const float* opacities           [[buffer(4), function_constant(kIsectAccuTile)]], // [n_elements]
    device const long*  cum_tiles_per_gauss [[buffer(5)]], // [n_elements] int64 inclusive cumsum
    device ulong*       isect_ids           [[buffer(6)]], // [n_isects] u64 keys (D3: treated as unsigned)
    device int*         flatten_ids         [[buffer(7)]], // [n_isects]
    constant IsectTileParams& p             [[buffer(8)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= p.n_elements) return;                  // :249
    const float radius_x = (float)radii[idx * 2];     // :255
    const float radius_y = (float)radii[idx * 2 + 1]; // :256
    if (radius_x <= 0.f || radius_y <= 0.f) return;   // :257–264 (the emit pass writes nothing)

    const uint  iid          = idx / p.N;                          // :268 dense
    const ulong iid_enc      = (ulong)iid << (32 + p.tile_n_bits); // :273
    const float depth_f      = depths[idx];                        // :279 (the input is already f32)
    const ulong depth_id_enc = (ulong)as_type<uint>(depth_f);      // :285 matches __float_as_uint

    const float2 mean2d = float2(means2d[2 * idx], means2d[2 * idx + 1]); // :266
    long cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1]; // :338/:434 inclusive→exclusive
    if (kIsectAccuTile) {
        // AccuTile emit — the same accutile_element as count (writes only within its own range, no atomics needed).
        accutile_element<true>(conics + idx * 3, opacities[idx], mean2d, p, iid_enc,
                               depth_id_enc, idx, isect_ids, flatten_ids, cur_idx);
        return;
    }
    const TileRect r = aabb_tile_rect(mean2d, radius_x, radius_y, p);
    for (int i = r.tmin.y; i < r.tmax.y; ++i) {                   // :435
        for (int j = r.tmin.x; j < r.tmax.x; ++j) {               // :448
            const ulong tile_id  = (ulong)(i * (int)p.tile_width + j);       // :454
            isect_ids[cur_idx]   = iid_enc | (tile_id << 32) | depth_id_enc; // :457
            flatten_ids[cur_idx] = (int)idx;                                 // :459 global flattened index
            ++cur_idx;
        }
    }
}

// isect_offset_encode — a literal translation of intersect_offset_kernel
// (IntersectTile.cu:925–988). From the sorted isect_ids it builds the per-tile start offsets
// offsets[I*tile_height*tile_width].
// One thread = one intersection, dispatchThreads(n_isects, tg 256) (:1001–1002). No atomics and no
// barriers are needed — only boundary threads write, and the write ranges do not overlap
// (idx==0: [0, id_curr]; a boundary: (id_prev, id_curr]; the last: (id_curr, I*n_tiles) — spec 03
// second verification pass: every position is written exactly once → no zero-fill needed).
// The tile mask is the 64-bit ((ulong)1 << tile_n_bits) - 1 — guarding against the shift UB of the
// original `(1 << tile_n_bits)` (int, :949) at tile_n_bits=32 (spec 03 §Kernel 4, handed over from
// Task 3). For n_isects==0 the host fills 0 without dispatching (:1004–1008).
kernel void isect_offset_encode(
    device const ulong* isect_ids [[buffer(0)]], // [n_isects] sorted u64 keys (D3)
    device int*         offsets   [[buffer(1)]], // [I * tile_height * tile_width] output
    constant IsectOffsetParams& p [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= p.n_isects) return;                            // :944–946 bounds guard
    const ulong tile_mask = ((ulong)1 << p.tile_n_bits) - 1;  // :949 (64-bit — guards the int UB)
    const ulong isect_id_curr = isect_ids[idx] >> 32;         // :947 strips the 32 depth bits
    const ulong id_curr =                                     // :948–950 flattened tile index
        (isect_id_curr >> p.tile_n_bits) * p.n_tiles + (isect_id_curr & tile_mask);

    if (idx == 0)                                             // :952–959 fill 0 up to the first valid tile
        for (ulong i = 0; i <= id_curr; ++i) offsets[i] = 0;
    if (idx == p.n_isects - 1)                                // :960–967 fill the tail
        for (ulong i = id_curr + 1; i < (ulong)p.I * p.n_tiles; ++i)
            offsets[i] = (int)p.n_isects;
    if (idx > 0) {                                            // :969–987 boundary threads
        const ulong isect_id_prev = isect_ids[idx - 1] >> 32;
        if (isect_id_prev == isect_id_curr) return;           // :974 same (image,tile) — not a boundary
        const ulong id_prev =
            (isect_id_prev >> p.tile_n_bits) * p.n_tiles + (isect_id_prev & tile_mask);
        for (ulong i = id_prev + 1; i <= id_curr; ++i)        // :981–986 includes the empty-tile range
            offsets[i] = (int)idx;
    }
}
