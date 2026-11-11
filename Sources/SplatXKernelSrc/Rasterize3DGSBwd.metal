// Rasterize3DGSBwd.metal — rasterize_to_pixels_3dgs_bwd (spec 05 backward section +
// supplements 1, 2, and 3 + the second verification pass, merged).
// Derived from nerfstudio-project/gsplat (Apache-2.0), commit 77ab983.
// Reimplemented for Metal/C++.
// Originals: RasterizeToPixels3DGSSerialBatchBwd.cu:41-320 (the kernel body),
//            RasterizeToPixels3DGSDevice.cuh:37-55 (eval_gaussian_weight — shared via KernelCommon.h),
//            Utils.cuh:153-191 (warpSum → a per-component simd_sum),
//            Common.h:97-114 (the constants — shared via KernelConstants.h).
//
// Phase 1 instance: TILE fixed at 16, CDIM=3. Unlike the forward, the thread configuration is a
// **2D threadgroup {16,16,1}** — exactly one thread per pixel (launcher Bwd.cu:361).
// block_size = tile_size² = 256 (Bwd.cu:125 — a runtime value in the original, but with TILE fixed
// at 16 it is the compile-time constant kRastCtaSize).
// The 2D linearization x + y*width → on a 16-wide tile, one simdgroup = 2 consecutive rows = a 1:1
// match with the CUDA warp layout (threadExecutionWidth==32 is asserted at pipeline creation — D8-3).
//
// Structure (1:1 with CUDA):
//  - one threadgroup = one tile. Gaussians are cooperatively loaded **back-to-front** (reverse of
//    the sort order; slot 0 = the back-most within the batch) — the forward's 3 regions plus
//    rgbs_batch (bwd caches the colors too).
//  - T recomputation: instead of storing the forward's T, T is walked back from
//    T_final = 1 - render_alphas by `T *= 1/(1-α)` (the ra denominator is bounded below by
//    kMinOneMinusAlpha=1e-6, Bwd.cu:229) — the division order is the exact reverse of the forward
//    multiplication, so the error stays controlled even in float (spec 05).
//  - OOB threads MUST NOT return — pix_id is clamped to the last pixel (Bwd.cu:114) and the thread
//    is kept as a cooperative-load worker (Bwd.cu:116-117). Returning on a masked-out tile is
//    allowed because the condition is threadgroup-uniform (Bwd.cu:106-109 — every thread returns
//    together, before any barrier or simd operation).
//  - **alive flag (D8-2)**: inside the processing loop, per-pixel invalidity is expressed as a valid
//    gate plus a zero contribution, not as an early return or continue — every lane reaches
//    simd_sum. The only continue is on simd_any(valid) (Bwd.cu:216-219) — its result is
//    simdgroup-uniform, so it is consistent with the reductions.
//  - after the simdgroup reduction (warpSum → per-component simd_sum), **only lane 0** performs the
//    device atomicAdd (Bwd.cu:282-317). One tile = 8 simdgroups scattering independently — merging
//    them with a threadgroup reduction would put a barrier inside a divergent loop, which is a
//    forbidden pattern (spec 05 supplement 3, D8-4).
//  - the atomicAdd uses relaxed device scope (the original's atomicAdd_system is for multi-GPU —
//    unnecessary on a single device, spec 05 mapping table). The order is nondeterministic → there
//    is no bit-exact reproduction: verification is the meta tolerances plus the two-run
//    self-consistency rel_l2 (07 §4).
//  - 2 barriers per batch (at the same positions as Bwd.cu:159/:184): at the top of the loop
//    (guaranteeing the threadgroup reads of the previous batch have completed) and after the
//    cooperative load. Barriers inside the processing loop are forbidden (divergent control flow).
//  - absgrad (kRastHasAbsGrad): |v_xy| is accumulated separately into v_means2d_abs
//    (Bwd.cu:269-272, the AbsGS densification heuristic) — off for v1 verification
//    (07 open question #3).
//  - the gradient output buffers are zero-initialized by the host with encode_zero_fill in the same
//    command buffer (D8-1 built in). When n_isects==0 the host skips the dispatch
//    (launcher Bwd.cu:365-369).
//
// Per-image pointer advance (Bwd.cu:90-102 / supplement 1): only tile_offsets (= isect_offsets),
// render_alphas, last_ids, v_render_colors, and v_render_alphas, plus backgrounds/masks when present.
// **The 5 gradient outputs and the 4 gaussian inputs are not advanced** — g = flatten_ids[idx] is
// already an [I*N] global flattened index (Bwd.cu:171/:292).
//
// threadgroup memory (Bwd.cu:128-132, the launcher formula at :384-385): 256·(4+12+12+4·3) = 10,240 B
// (packed_float3 is 12 B — matching the CUDA glm vec3 stride, spec 05 §Batch size calculation).
#include <metal_stdlib>
#include "KernelConstants.h"
#include "FunctionConstants.h"
#include "KernelParams.h"
#include "KernelCommon.h"
using namespace metal;

static_assert(kRastTileSize * kRastTileSize == kRastCtaSize,
              "bwd block_size must equal tile_size^2 (Bwd.cu:125, Phase 1 instantiation)");

// Corresponds to __launch_bounds__ (the same intent as supplement 4) — register allocation assuming 256 resident.
[[max_total_threads_per_threadgroup(256)]]
kernel void rasterize_to_pixels_3dgs_bwd(
    // 0-7: the input section shared with the forward (the task-3 §6.4 buffer contract)
    device const float2*        means2d         [[buffer(0)]],  // [I,N,2] flattened
    device const packed_float3* conics          [[buffer(1)]],  // [I,N,3] (A,B,C)
    device const float*         colors          [[buffer(2)]],  // [I,N,CDIM]
    device const float*         opacities       [[buffer(3)]],  // [I,N]
    device const float*         backgrounds     [[buffer(4), function_constant(kRastHasBackground)]], // [I,CDIM]
    device const uchar*         masks           [[buffer(5), function_constant(kRastHasMasks)]],      // [I,TH,TW] 0/1
    device const int*           isect_offsets   [[buffer(6)]],  // [I,TH,TW] global cumulative
    device const int*           flatten_ids     [[buffer(7)]],  // [n_isects] already sorted
    // 8-11: the fwd outputs + the upstream gradients (Bwd.cu:62-68)
    device const float*         render_alphas   [[buffer(8)]],  // [I,H,W,1]
    device const int*           last_ids        [[buffer(9)]],  // [I,H,W]
    device const float*         v_render_colors [[buffer(10)]], // [I,H,W,CDIM]
    device const float*         v_render_alphas [[buffer(11)]], // [I,H,W,1]
    // 12-16: the gradient outputs (atomic accumulation — zero-initialized beforehand, D8-1)
    device atomic_float*        v_means2d       [[buffer(12)]], // [I,N,2] flattened
    device atomic_float*        v_conics        [[buffer(13)]], // [I,N,3] flattened
    device atomic_float*        v_colors        [[buffer(14)]], // [I,N,CDIM] flattened
    device atomic_float*        v_opacities     [[buffer(15)]], // [I,N]
    device atomic_float*        v_means2d_abs   [[buffer(16), function_constant(kRastHasAbsGrad)]], // [I,N,2]
    constant RasterizeParams&   P               [[buffer(17)]],
    // Expected-depth channel: routed exactly like a fourth colour channel (no background term).
    device const float*         depths          [[buffer(18), function_constant(kRastHasDepth)]], // [I,N]
    device const float*         v_render_depths [[buffer(19), function_constant(kRastHasDepth)]], // [I,H,W]
    device atomic_float*        v_depths        [[buffer(20), function_constant(kRastHasDepth)]], // [I,N]
    uint2 tg_pos    [[threadgroup_position_in_grid]],           // .x = linear_block_index
                                                                // (a 2D threadgroup, so the position
                                                                //  attribute ranks match — MSL rule)
    uint2 local_xy  [[thread_position_in_threadgroup]],         // (x = column offset, y = row offset)
    uint  tr        [[thread_index_in_threadgroup]],            // = local_xy.y*16 + local_xy.x (thread_rank)
    uint  simd_lane [[thread_index_in_simdgroup]])
{
    constexpr uint TILE_SIZE  = kRastTileSize; // 16
    constexpr uint BLOCK_SIZE = kRastCtaSize;  // 256 = block.size() (Bwd.cu:125)
    constexpr uint CDIM       = kRastCdim;     // 3
    const uint block_id       = tg_pos.x;      // the grid is 1D — y is always 0

    // Block/pixel addressing (Bwd.cu:77-88) — decomposes linear_block_index into (image, tile_y, tile_x)
    const uint tiles_per_image = P.tile_width * P.tile_height;
    const uint image_id        = block_id / tiles_per_image;
    const uint tile_id         = block_id % tiles_per_image; // = tile_iy*tile_width + tile_ix
    const uint tile_ix         = tile_id % P.tile_width;     // block_index.z
    const uint tile_iy         = tile_id / P.tile_width;     // block_index.y
    const uint i               = tile_iy * TILE_SIZE + local_xy.y; // pixel row (Bwd.cu:87)
    const uint j               = tile_ix * TILE_SIZE + local_xy.x; // pixel column (Bwd.cu:88)

    // Per-image pointer advance (Bwd.cu:90-102, supplement 1) — the 5 gradient outputs and the 4
    // gaussian inputs are not advanced (flatten_ids holds [I*N] global flattened indices).
    isect_offsets   += image_id * tiles_per_image;
    render_alphas   += image_id * P.image_height * P.image_width;
    last_ids        += image_id * P.image_height * P.image_width;
    v_render_colors += image_id * P.image_height * P.image_width * CDIM;
    v_render_alphas += image_id * P.image_height * P.image_width;
    if (kRastHasBackground) backgrounds += image_id * CDIM;
    if (kRastHasMasks)      masks       += image_id * tiles_per_image;
    if (kRastHasDepth)      v_render_depths += image_id * P.image_height * P.image_width;

    // Masked-out tile: return without writing anything (Bwd.cu:106-109 — this assumes the gradient
    // buffers were zero-initialized beforehand). The condition is threadgroup-uniform and this sits
    // before the first barrier or simd operation, so it does not violate D8-2 (the prohibition on
    // divergent early returns). An unverified path in v1 — there is no fixture (07 Correction 6).
    if (kRastHasMasks && masks[tile_id] == 0) return;

    const float px = (float)j + 0.5f; // pixel-center correction (Bwd.cu:111-112)
    const float py = (float)i + 0.5f;
    // OOB threads clamp to the last pixel (Bwd.cu:114) — they do not return and stay as load workers
    const uint pix_id = min(i * P.image_width + j, P.image_width * P.image_height - 1u);
    const bool inside = (i < P.image_height) && (j < P.image_width);

    // Gaussian range (Bwd.cu:122-126, supplement 2 — the same logic as the forward)
    const int range_start = isect_offsets[tile_id];
    const int range_end   = (image_id == P.I - 1u && tile_id == tiles_per_image - 1u)
                              ? (int)P.n_isects
                              : isect_offsets[tile_id + 1];
    const uint num_batches =
        (uint)(range_end - range_start + (int)BLOCK_SIZE - 1) / BLOCK_SIZE;

    // threadgroup batches (Bwd.cu:128-132) — the forward's 3 regions + rgbs_batch, 10,240 B total
    threadgroup int           id_batch[BLOCK_SIZE];         // the flatten_ids[idx] value (flattened g)
    threadgroup packed_float3 xy_opacity_batch[BLOCK_SIZE]; // {mean.x, mean.y, opacity}
    threadgroup packed_float3 conic_batch[BLOCK_SIZE];      // {A, B, C}
    threadgroup float         rgbs_batch[BLOCK_SIZE * CDIM]; // bwd caches the colors too

    // Per-pixel initial state (Bwd.cu:134-149) — OOB threads also read via the clamped pix_id (same
    // as the upstream original; valid is always false for them, so the values go unused).
    const float T_final = 1.0f - render_alphas[pix_id]; // restores the final T of the forward
    float T = T_final;                                  // the starting value of the back-to-front walk
    float buffer_acc[CDIM] = {0.f, 0.f, 0.f}; // S_i[k], contributions accumulated behind (farther than) this gaussian
    const int bin_final = inside ? last_ids[pix_id] : 0; // absolute isect index of the last contribution

    float v_render_c[CDIM];
    for (uint k = 0; k < CDIM; ++k) v_render_c[k] = v_render_colors[pix_id * CDIM + k];
    const float v_render_a = v_render_alphas[pix_id];
    const float v_render_d = kRastHasDepth ? v_render_depths[pix_id] : 0.f;
    float buffer_depth     = 0.f; // S_i for the depth channel

    // warp pre-reduction (Bwd.cu:153-155): cg::reduce(warp, bin_final, greater) → simd_max
    const int warp_bin_final = simd_max(bin_final);

    for (uint b = 0; b < num_batches; ++b) {
        // (barrier 1) guarantees the threadgroup reads of the previous batch have completed (Bwd.cu:159 block.sync)
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cooperative load: slot 0 = the back-most within the batch. batch_end - tr can go negative
        // — keep it int (Bwd.cu:161-182, the comment at :165)
        const int batch_end  = range_end - 1 - (int)(BLOCK_SIZE * b);
        const int batch_size = min((int)BLOCK_SIZE, batch_end + 1 - range_start);
        const int idx        = batch_end - (int)tr;
        if (idx >= range_start) {
            const int g  = flatten_ids[idx];
            id_batch[tr] = g;
            const float2 xy       = means2d[g];
            xy_opacity_batch[tr]  = packed_float3(xy.x, xy.y, opacities[g]);
            conic_batch[tr]       = conics[g];
            for (uint k = 0; k < CDIM; ++k)
                rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
        }
        // (barrier 2) the load is complete (Bwd.cu:184 block.sync). Barriers are forbidden in the processing loop below.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Ascending t = descending absolute index (batch_end - t) = back-to-front (Bwd.cu:187).
        // The start point max(0, batch_end - warp_bin_final) skips everything after the simdgroup's
        // last contributor outright — the value is simdgroup-uniform, so this is a safe, purely
        // performance optimization (spec 05 second verification pass).
        for (int t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            // alive flag (D8-2): a valid gate plus a zero contribution — every lane reaches simd_sum.
            bool valid = inside;
            if (batch_end - t > bin_final) valid = false; // behind the last contributor (Bwd.cu:190-193)
            float alpha = 0.f, opac = 0.f, vis = 0.f;
            float2 delta = float2(0.f);
            packed_float3 conic = packed_float3(0.f, 0.f, 0.f);
            if (valid) {
                conic = conic_batch[t];
                const packed_float3 xy_opac = xy_opacity_batch[t];
                opac  = xy_opac.z;
                delta = float2(xy_opac.x - px, xy_opac.y - py); // = mean2d − pixel_center
                const GaussianWeight gw = eval_gaussian_weight(conic, delta.x, delta.y, opac);
                vis   = gw.vis;
                alpha = gw.alpha;
                if (!gw.valid) valid = false; // sigma<0 ‖ alpha<1/255 — the same gate as fwd
            }
            // Skip when the whole simdgroup is invalid (Bwd.cu:216-219 warp.any) — the simd_any
            // result is simdgroup-uniform, so it is consistent with the simd_sum below (the only
            // continue that is allowed).
            if (!simd_any(valid)) continue;

            // lane-local gradients — invalid lanes keep 0 (Bwd.cu:220-225)
            float  v_rgb_local[CDIM] = {0.f, 0.f, 0.f};
            float3 v_conic_local     = float3(0.f);
            float2 v_xy_local        = float2(0.f);
            float2 v_xy_abs_local    = float2(0.f);
            float  v_opacity_local   = 0.f;
            float  v_depth_local     = 0.f;
            if (valid) {
                // T recomputation (Bwd.cu:229-232): ra = 1/(1-α), bounded below by kMinOneMinusAlpha=1e-6
                const float ra  = 1.0f / max(kMinOneMinusAlpha, 1.0f - alpha);
                T              *= ra;
                const float fac = alpha * T; // == the contribution weight of the forward (Fwd.cu:254 vis)
                for (uint k = 0; k < CDIM; ++k) v_rgb_local[k] = fac * v_render_c[k];

                // The 3 terms of v_alpha (Bwd.cu:239-257):
                //  ① Σ_k (c_i[k]·T_i − S_i[k]/(1−α_i))·v_render_c[k]
                //  ② + T_final/(1−α_i)·v_render_a
                //  ③ − T_final/(1−α_i)·Σ_k bg[k]·v_render_c[k]  (only when a background exists)
                float v_alpha = 0.f;
                for (uint k = 0; k < CDIM; ++k)
                    v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer_acc[k] * ra) *
                               v_render_c[k];
                v_alpha += T_final * ra * v_render_a;
                float depth_g = 0.f;
                if (kRastHasDepth) {
                    depth_g       = depths[id_batch[t]];
                    v_depth_local = fac * v_render_d;
                    v_alpha      += (depth_g * T - buffer_depth * ra) * v_render_d;
                }
                if (kRastHasBackground) {
                    float accum = 0.f;
                    for (uint k = 0; k < CDIM; ++k) accum += backgrounds[k] * v_render_c[k];
                    v_alpha += -T_final * ra * accum;
                }

                // Clamp gate (Bwd.cu:259) — when opac·vis > kMaxAlpha(0.99), α is a constant clamp,
                // so the σ and opac gradients are blocked. The boundary is **included** (<=) on the
                // passing side — identical to clamp_max autograd (confirmed in the task-2 review).
                if (opac * vis <= kMaxAlpha) {
                    const float v_sigma = -opac * vis * v_alpha; // ∂α/∂σ = −opac·vis
                    v_conic_local = float3(0.5f * v_sigma * delta.x * delta.x, // ∂σ/∂conic
                                           v_sigma * delta.x * delta.y,
                                           0.5f * v_sigma * delta.y * delta.y);
                    v_xy_local = float2(v_sigma * (conic.x * delta.x + conic.y * delta.y),
                                        v_sigma * (conic.y * delta.x + conic.z * delta.y));
                    if (kRastHasAbsGrad) // absgrad (Bwd.cu:269-272)
                        v_xy_abs_local = float2(abs(v_xy_local.x), abs(v_xy_local.y));
                    v_opacity_local = vis * v_alpha; // ∂α/∂opac = vis
                }

                // Update the accumulated behind-contribution S_i[k] += c_i[k]·α_i·T_i (Bwd.cu:276-280)
                for (uint k = 0; k < CDIM; ++k)
                    buffer_acc[k] += rgbs_batch[t * CDIM + k] * fac;
                if (kRastHasDepth) buffer_depth += depth_g * fac;
            }

            // simdgroup reduction (warpSum, Bwd.cu:282-289 / Utils.cuh:153-191) — a per-component
            // simd_sum. A threadgroup reduction is forbidden (it would put a barrier inside a
            // divergent loop — spec 05 supplement 3).
            for (uint k = 0; k < CDIM; ++k) v_rgb_local[k] = simd_sum(v_rgb_local[k]);
            v_conic_local = float3(simd_sum(v_conic_local.x), simd_sum(v_conic_local.y),
                                   simd_sum(v_conic_local.z));
            v_xy_local    = float2(simd_sum(v_xy_local.x), simd_sum(v_xy_local.y));
            if (kRastHasAbsGrad)
                v_xy_abs_local = float2(simd_sum(v_xy_abs_local.x), simd_sum(v_xy_abs_local.y));
            v_opacity_local = simd_sum(v_opacity_local);
            if (kRastHasDepth) v_depth_local = simd_sum(v_depth_local);

            // Only lane 0 does the atomic scatter (Bwd.cu:290-317) — CDIM+6 (+2 with absgrad) per
            // simdgroup; one tile = 8 independent simdgroups (supplement 3). relaxed device scope
            // (spec 05 mapping table).
            if (simd_lane == 0) {
                const int g = id_batch[t]; // [I*N] global flattened index (Bwd.cu:292)
                for (uint k = 0; k < CDIM; ++k)
                    atomic_fetch_add_explicit(v_colors + g * CDIM + k, v_rgb_local[k],
                                              memory_order_relaxed);
                atomic_fetch_add_explicit(v_conics + g * 3 + 0, v_conic_local.x,
                                          memory_order_relaxed);
                atomic_fetch_add_explicit(v_conics + g * 3 + 1, v_conic_local.y,
                                          memory_order_relaxed);
                atomic_fetch_add_explicit(v_conics + g * 3 + 2, v_conic_local.z,
                                          memory_order_relaxed);
                atomic_fetch_add_explicit(v_means2d + g * 2 + 0, v_xy_local.x,
                                          memory_order_relaxed);
                atomic_fetch_add_explicit(v_means2d + g * 2 + 1, v_xy_local.y,
                                          memory_order_relaxed);
                if (kRastHasAbsGrad) {
                    atomic_fetch_add_explicit(v_means2d_abs + g * 2 + 0, v_xy_abs_local.x,
                                              memory_order_relaxed);
                    atomic_fetch_add_explicit(v_means2d_abs + g * 2 + 1, v_xy_abs_local.y,
                                              memory_order_relaxed);
                }
                atomic_fetch_add_explicit(v_opacities + g, v_opacity_local,
                                          memory_order_relaxed);
                if (kRastHasDepth)
                    atomic_fetch_add_explicit(v_depths + g, v_depth_local, memory_order_relaxed);
            }
        }
    }
}
