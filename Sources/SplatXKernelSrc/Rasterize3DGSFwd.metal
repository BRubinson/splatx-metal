// Rasterize3DGSFwd.metal — rasterize_to_pixels_3dgs_fwd (spec 05 forward section +
// supplements 1 and 4 + the second correction, merged).
// Derived from nerfstudio-project/gsplat (Apache-2.0), commit 77ab983.
// Reimplemented for Metal/C++.
// Originals: RasterizeToPixels3DGSSerialBatchFwd.cu:41-297 (the kernel body),
//            RasterizeToPixels3DGSDevice.cuh:37-55 (eval_gaussian_weight — shared via KernelCommon.h),
//            Common.h:97-114 (the constants — shared via KernelConstants.h).
//
// Phase 1 instance: TILE=16 / CTA=256 → BATCH_SIZE=256, **PPT=1** (the tile_size==16 combination of
// the launcher Fwd.cu:401-412). Because PPT=1, the original's p-loop (PIXELS_PER_THREAD) and the
// done_mask bitmask collapse into one pixel per thread and a bool done (ALL_DONE=1, ROW_STRIDE
// irrelevant — numerically equivalent).
// The tile=4/CTA=16 variant (for CDIM>=32 only) is not ported — the host validator enforces
// tile_size==16.
//
// Structure (1:1 with CUDA):
//  - one threadgroup = one tile, 256 threads = 256 pixels. Grid = I*tile_h*tile_w.
//  - gaussians are cooperatively loaded into threadgroup memory BATCH_SIZE at a time (id, xy +
//    opacity, conic — colors are not placed in shared memory but read directly from device memory
//    once a contribution is confirmed, Fwd.cu:255).
//  - the barrier count matches CUDA (2 per batch): a threadgroup_barrier after the load
//    (corresponding to __syncthreads), and the sg_done aggregation barrier at the end of the batch
//    (corresponding to __syncthreads_count — it doubles as the resynchronization before the next
//    batch load).
//  - OOB threads MUST NOT return (they only set done) — they are needed as cooperative-load workers
//    and for barrier uniformity (Fwd.cu:121-133).
//  - the early termination next_T <= kTransmittanceThreshold(1e-4) is **exclusive** — the triggering
//    gaussian is not reflected in the color, in T, or in cur_idx (Fwd.cu:246-251).
//  - last_ids = the **absolute isect index** (batch_start + t) of the last contributing gaussian, and
//    0 when there is no contribution (for the first gaussian to cross the threshold at T=1 it would
//    need alpha>=0.9999, which the MAX_ALPHA=0.99 clamp makes impossible → the dual use of 0 in the
//    encoding is never polluted, Fwd.cu:186/261/293).
//  - no atomics, sequential accumulation per pixel → **bit-exact determinism** (gated by the two-run
//    memcmp in RasterTests).
//
// Parity points (D2 / the lesson of 1a Task 4): vis = precise::exp(-sigma) (KernelCommon.h), and the
// weighted color sum and the background composition use explicit metal::fma — paired with std::exp
// and std::fma in the CPU reference (rasterize_fwd_cpu).
//
// threadgroup memory (Fwd.cu:170-173 + the sg_done scratch): 256·(4+12+12) + 8 = 7,176 B
// (packed_float3 is 12 B — matching the CUDA glm vec3 stride, spec 05 §Batch size calculation).
// sg_done size 256/32 = 8 — second correction: correct on the TILE=16/CTA=256 path (a CTA<=32
// variant is forbidden).
#include <metal_stdlib>
#include "KernelConstants.h"
#include "FunctionConstants.h"
#include "KernelParams.h"
#include "KernelCommon.h"
using namespace metal;

static_assert(kRastTileSize * kRastTileSize == kRastCtaSize,
              "PPT must be 1 (Phase 1 instantiation)");
static_assert(kRastCtaSize / 32u >= 1u, "sg_done size must be >= 1 (second correction)");

// Corresponds to __launch_bounds__(CTA_SIZE) (supplement 4) — register allocation assuming 256 resident.
[[max_total_threads_per_threadgroup(256)]]
kernel void rasterize_to_pixels_3dgs_fwd(
    device const float2*        means2d       [[buffer(0)]],  // [I,N,2] flattened
    device const packed_float3* conics        [[buffer(1)]],  // [I,N,3] (A,B,C)
    device const float*         colors        [[buffer(2)]],  // [I,N,CDIM]
    device const float*         opacities     [[buffer(3)]],  // [I,N]
    device const float*         backgrounds   [[buffer(4), function_constant(kRastHasBackground)]], // [I,CDIM]
    device const uchar*         masks         [[buffer(5), function_constant(kRastHasMasks)]],      // [I,TH,TW] 0/1
    device const int*           isect_offsets [[buffer(6)]],  // [I,TH,TW] global cumulative
    device const int*           flatten_ids   [[buffer(7)]],  // [n_isects] already sorted
    device float*               render_colors [[buffer(8)]],  // [I,H,W,CDIM]
    device float*               render_alphas [[buffer(9)]],  // [I,H,W,1]
    device int*                 last_ids      [[buffer(10)]], // [I,H,W]
    constant RasterizeParams&   P             [[buffer(11)]],
    uint block_id  [[threadgroup_position_in_grid]],          // = linear_block_index (no block_offset needed)
    uint tid       [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    constexpr uint TILE_SIZE  = kRastTileSize;      // 16
    constexpr uint BATCH_SIZE = kRastCtaSize;       // 256 (= CTA_SIZE, Fwd.cu:65)
    constexpr uint CDIM       = kRastCdim;          // 3
    constexpr uint TILE_MASK  = TILE_SIZE - 1;      // 15
    constexpr uint TILE_SHIFT = 4;                  // ctz(16)

    // Tile/pixel addressing (Fwd.cu:77-119)
    const uint tiles_per_image = P.tile_width * P.tile_height;
    const uint image_id        = block_id / tiles_per_image;
    const uint tile_id         = block_id % tiles_per_image; // = tile_y*tile_width + tile_x
    const uint tile_x          = tile_id % P.tile_width;
    const uint tile_y          = tile_id / P.tile_width;

    const uint thread_x = tid & TILE_MASK;
    const uint thread_y = tid >> TILE_SHIFT;

    // Per-image pointer advance (Fwd.cu:92-103) — the 4 gaussian inputs are not advanced
    // (flatten_ids holds [I*N] global flattened indices — spec 05 supplement 1).
    isect_offsets += image_id * tiles_per_image;
    render_colors += image_id * P.image_height * P.image_width * CDIM;
    render_alphas += image_id * P.image_height * P.image_width;
    last_ids      += image_id * P.image_height * P.image_width;
    if (kRastHasBackground) backgrounds += image_id * CDIM;
    if (kRastHasMasks)      masks       += image_id * tiles_per_image;

    const uint out_x  = tile_x * TILE_SIZE + thread_x;
    const uint out_y  = tile_y * TILE_SIZE + thread_y; // PPT=1 — the p·ROW_STRIDE term is eliminated
    const float px    = (float)out_x + 0.5f;           // pixel-center correction (+0.5)
    const float py    = (float)out_y + 0.5f;
    const uint pix_id = out_y * P.image_width + out_x;

    // OOB threads do not return; they only set done — cooperative load and barrier uniformity (Fwd.cu:121-133).
    bool done = (out_x >= P.image_width) || (out_y >= P.image_height);
    const bool inside = !done;

    // Masked-out tile: write the background (0 if there is none), then return uniformly across the
    // threadgroup (Fwd.cu:141-159 — safe because it is a uniform branch before the first barrier.
    // An unverified path in v1 — there is no fixture, 07 Correction 6).
    if (kRastHasMasks && masks[tile_id] == 0) {
        if (inside) {
            for (uint k = 0; k < CDIM; ++k)
                render_colors[pix_id * CDIM + k] = kRastHasBackground ? backgrounds[k] : 0.0f;
            render_alphas[pix_id] = 0.0f;
            last_ids[pix_id]      = 0;
        }
        return;
    }

    // Gaussian range (Fwd.cu:164-168) — because offsets is an [I,TH,TW] global cumulative array, the
    // [tile_id + 1] read at an image boundary (the first offset of the next image) is also correct.
    const int range_start = isect_offsets[tile_id];
    const int range_end   = (image_id == P.I - 1u && tile_id == tiles_per_image - 1u)
                              ? (int)P.n_isects
                              : isect_offsets[tile_id + 1];
    const uint num_batches =
        (uint)(range_end - range_start + (int)BATCH_SIZE - 1) / BATCH_SIZE;

    // threadgroup batches (Fwd.cu:170-173) + the sg_done scratch — 7,176 B total
    threadgroup int           id_batch[BATCH_SIZE];         // the flatten_ids[idx] value (flattened g)
    threadgroup packed_float3 xy_opacity_batch[BATCH_SIZE]; // {mean.x, mean.y, opacity}
    threadgroup packed_float3 conic_batch[BATCH_SIZE];      // {A, B, C}
    threadgroup uchar         sg_done[kRastCtaSize / 32u];  // 8 — replaces __syncthreads_count

    float T        = 1.0f;              // transmittance (Fwd.cu:179-184 — spec 05 gives the rationale for float)
    uint  cur_idx  = 0u;                // absolute isect index of the last contribution, 0 if none (:186)
    float pix_out[CDIM] = {0.f, 0.f, 0.f};

    for (uint b = 0; b < num_batches; ++b) {
        // Cooperative load: one gaussian per thread (Fwd.cu:197-208)
        const int batch_start = range_start + (int)(BATCH_SIZE * b);
        const int idx         = batch_start + (int)tid;
        if (idx < range_end) {
            const int g          = flatten_ids[idx];
            id_batch[tid]        = g;
            const float2 xy      = means2d[g];
            xy_opacity_batch[tid] = packed_float3(xy.x, xy.y, opacities[g]);
            conic_batch[tid]      = conics[g];
        }
        // Wait for the load to complete (Fwd.cu:212-219 __syncthreads — CTA=256 > 32)
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // front-to-back blending (Fwd.cu:221-264) — PPT=1: once done, the batch processing is skipped
        const uint batch_size = min(BATCH_SIZE, (uint)(range_end - batch_start));
        for (uint t = 0; (t < batch_size) && !done; ++t) {
            const packed_float3 conic   = conic_batch[t];
            const packed_float3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const float dx   = xy_opac.x - px;
            const float dy   = xy_opac.y - py;
            const GaussianWeight gw = eval_gaussian_weight(conic, dx, dy, opac);
            if (!gw.valid) continue;

            const float alpha  = gw.alpha;
            const float next_T = T * (1.0f - alpha);
            if (next_T <= kTransmittanceThreshold) { // early termination — exclusive (Fwd.cu:246-251)
                done = true;
                continue;
            }

            const int g     = id_batch[t];
            const float vis = alpha * T; // this gaussian's contribution weight for the pixel (:254)
            for (uint k = 0; k < CDIM; ++k) // weighted-sum fma (paired with CPU std::fma — the 1a Task 4 lesson)
                pix_out[k] = metal::fma(colors[g * CDIM + k], vis, pix_out[k]);
            cur_idx = (uint)(batch_start + (int)t); // absolute isect index (:261)
            T       = next_T;
        }

        // Replaces __syncthreads_count(done) >= BATCH_SIZE (Fwd.cu:268) — a simd ballot plus a
        // threadgroup aggregation (spec 05 §Barrier placement 2). This barrier also serves as the
        // resynchronization before the next batch load (guaranteeing the shared reads of the
        // previous batch have completed).
        const bool sg_all = simd_all(done); // a uniform point — every lane is active
        if (simd_lane == 0) sg_done[simd_id] = (uchar)sg_all;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool all_done = true;
        for (uint s = 0; s < kRastCtaSize / 32u; ++s) all_done = all_done && (sg_done[s] != 0);
        if (all_done) break;
    }

    // Final write (Fwd.cu:274-296) — regardless of an empty range or early termination, every
    // in-bounds pixel is composited as pix_out + T·bg (including the empty-tile bg path of
    // 07 Correction 4).
    if (inside) {
        render_alphas[pix_id] = 1.0f - T;
        for (uint k = 0; k < CDIM; ++k)
            render_colors[pix_id * CDIM + k] =
                kRastHasBackground ? metal::fma(T, backgrounds[k], pix_out[k]) : pix_out[k];
        last_ids[pix_id] = (int)cur_idx;
    }
}
