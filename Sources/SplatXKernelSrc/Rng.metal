// Rng.metal — rng_philox_u32 / rng_normal_f32 (spec 06 §Noise producer, 00-index disposition #24).
// Written from scratch (from the literature) — Salmon et al., "Parallel Random Numbers: As Easy
// as 1, 2, 3" (SC'11). The core lives in RngPhilox.h as the single source of truth (the same
// source as the host CPU reference — the philox integer stage and the uniform conversion are
// bit-identical between CPU and Metal, and that is the contract).
//
// counter-based: no state buffer — reproduced from (seed, offset, tid) alone (spec 06 design
// rationale 2: reproducibility is guaranteed even when the thread count changes from step to
// step). One philox block per thread = 4×u32 → 4 output elements.
// Dispatch: dispatchThreads(ceil(count/4), tg 256). No atomics, no barriers, no threadgroup memory.
#include <metal_stdlib>
#include "KernelParams.h"
#include "RngPhilox.h"
using namespace metal;

// Emits the philox integer stage as-is — the verification surface for the bit-identity contract
// (before the uniform/normal conversions). The perturb path does not consume it (test-only —
// McmcTests memcmps it against the CPU reference).
kernel void rng_philox_u32(
    device uint*        out [[buffer(0)]], // [count]
    constant RngParams& P   [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    const uint base = tid * 4u;
    if (base >= P.count) return;
    const SXPhilox4 r = sx_philox4x32_10(tid, P.offset_lo, P.offset_hi, 0u,
                                         P.seed_lo, P.seed_hi);
    for (uint j = 0; j < 4u && base + j < P.count; ++j) out[base + j] = r.v[j];
}

// Standard normals [count] — philox 4×u32 → uniform (0,1] → 2 Box-Muller pairs.
// u1 = 0 is impossible (sx_u32_to_uniform — 06 Correction 5 blocks log(0)); RngPhilox.h pins
// log/sqrt/sin/cos to precise:: (D2 — close to CPU libm, subject to the rel gate).
kernel void rng_normal_f32(
    device float*       out [[buffer(0)]], // [count]
    constant RngParams& P   [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    const uint base = tid * 4u;
    if (base >= P.count) return;
    const SXPhilox4 r = sx_philox4x32_10(tid, P.offset_lo, P.offset_hi, 0u,
                                         P.seed_lo, P.seed_hi);
    const SXNormal2 a = sx_box_muller(r.v[0], r.v[1]);
    const SXNormal2 b = sx_box_muller(r.v[2], r.v[3]);
    const float z[4] = {a.z0, a.z1, b.z0, b.z1};
    for (uint j = 0; j < 4u && base + j < P.count; ++j) out[base + j] = z[j];
}
