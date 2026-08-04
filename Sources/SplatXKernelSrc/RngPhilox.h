// splatx-metal — Philox4x32-10 counter-based RNG core
// (the single source of truth shared by host C++ and device MSL)
//
// Written from scratch (from the literature) — not derived from any other project's code
// (00-index §6.1 controller decision, disposition #24):
//   J. K. Salmon, M. A. Moraes, R. O. Dror, D. E. Shaw,
//   "Parallel Random Numbers: As Easy as 1, 2, 3", SC'11 (doi:10.1145/2063384.2063405), §3.3.
//   The constants follow the paper's notation: multipliers 0xD2511F53 / 0xCD9E8D57, key schedule
//   (Weyl) 0x9E3779B9 / 0xBB67AE85, 10 rounds. Verification anchor: the 3 philox4x32×10 sets from
//   the Random123 kat_vectors (McmcTests.cpp — quoted test data, not derived code).
//
// Single-source-of-truth contract (spec 06 §Noise producer + Verification Notes Correction 5):
// - the philox integer stage (4×u32) is pure integer arithmetic — **bit-identical** between host
//   and device is the contract (the same source is compiled for both; verified by the
//   rng_philox_u32 kernel ↔ rng_philox_u32_cpu).
// - the uniform conversion sx_u32_to_uniform is also bit-identical, being an IEEE conversion plus
//   an exact multiplication by 2^-32.
//   u ∈ (0,1] is guaranteed — **u1 = 0 is impossible** (blocks log(0), Correction 5): you MUST use
//   the floating-point arithmetic (float(x)+1)·2⁻³² (the uint arithmetic x+1 wraps to 0 at
//   0xFFFFFFFF — forbidden).
// - only the log/sqrt/sin/cos of Box-Muller deviate as transcendental functions
//   (Metal precise:: ↔ host libm) — they are subject to the rel gate, not to bit-identity (D2).
#ifndef SPLATX_RNG_PHILOX_H
#define SPLATX_RNG_PHILOX_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
typedef ulong sx_rng_u64;
#define SX_RNG_LOG(x)  metal::precise::log(x)
#define SX_RNG_SQRT(x) metal::precise::sqrt(x)
#define SX_RNG_SIN(x)  metal::precise::sin(x)
#define SX_RNG_COS(x)  metal::precise::cos(x)
#else
#include <cmath>
#include <cstdint>
typedef std::uint32_t uint; // same type as uint in Darwin sys/types.h — redeclaration allowed (as in KernelParams.h)
typedef std::uint64_t sx_rng_u64;
#define SX_RNG_LOG(x)  std::log(x)
#define SX_RNG_SQRT(x) std::sqrt(x)
#define SX_RNG_SIN(x)  std::sin(x)
#define SX_RNG_COS(x)  std::cos(x)
#endif

struct SXPhilox4 {
    uint v[4];
};

// One Philox4x32-10 block: counter (c0,c1,c2,c3) × key (k0,k1) → 4×u32.
// The key of round r = (k0 + r·0x9E3779B9, k1 + r·0xBB67AE85), r = 0..9 (Salmon et al. §3.3).
static inline SXPhilox4 sx_philox4x32_10(uint c0, uint c1, uint c2, uint c3, uint k0, uint k1)
{
    for (int r = 0; r < 10; ++r) {
        if (r > 0) {
            k0 += 0x9E3779B9u; // Weyl increment (golden ratio, 2^32/φ)
            k1 += 0xBB67AE85u; // (√3−1)·2^31
        }
        const sx_rng_u64 p0 = (sx_rng_u64)0xD2511F53u * c0;
        const sx_rng_u64 p1 = (sx_rng_u64)0xCD9E8D57u * c2;
        const uint hi0 = (uint)(p0 >> 32), lo0 = (uint)p0;
        const uint hi1 = (uint)(p1 >> 32), lo1 = (uint)p1;
        const uint n0 = hi1 ^ c1 ^ k0;
        const uint n2 = hi0 ^ c3 ^ k1;
        c0 = n0;
        c1 = lo1;
        c2 = n2;
        c3 = lo0;
    }
    SXPhilox4 out;
    out.v[0] = c0; out.v[1] = c1; out.v[2] = c2; out.v[3] = c3;
    return out;
}

// u32 → uniform (0,1] — boundaries: x=0 → 2⁻³², x=0xFFFFFFFF → 1.0 (float(x) rounds to 2³² and
// then absorbs the +1.0f). u > 0 over the whole range, so log(u) is finite (Correction 5). The
// uint→float conversion and the multiplication by a power of two are IEEE-deterministic —
// bit-identical between host and device.
static inline float sx_u32_to_uniform(uint x)
{
    return ((float)x + 1.0f) * (1.0f / 4294967296.0f);
}

struct SXNormal2 {
    float z0, z1;
};

// Box-Muller: (x0,x1) → 2 standard normals. u1 = T(x0) ∈ (0,1] → -2·ln(u1) ∈ [0, 44.36], finite;
// with u1=1 (x0=0xFFFFFFFF), r=0 → z=±0 (NaN/Inf impossible — pinned by the boundary-value unit test).
static inline SXNormal2 sx_box_muller(uint x0, uint x1)
{
    const float u1 = sx_u32_to_uniform(x0);
    const float u2 = sx_u32_to_uniform(x1);
    const float r = SX_RNG_SQRT(-2.0f * SX_RNG_LOG(u1));
    const float theta = 6.28318530717958647692f * u2; // 2π (rounds to 6.2831855f in f32)
    SXNormal2 out;
    out.z0 = r * SX_RNG_COS(theta);
    out.z1 = r * SX_RNG_SIN(theta);
    return out;
}

#endif // SPLATX_RNG_PHILOX_H
