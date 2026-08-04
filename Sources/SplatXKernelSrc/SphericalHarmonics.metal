// SphericalHarmonics.metal — sh_fwd / sh_bwd (spec 02 body + the 6 Verification Notes
// reinforcements + the second re-verification pass, merged).
// Derived from nerfstudio-project/gsplat (Apache-2.0), commit 77ab983.
// Reimplemented for Metal/C++.
// Originals: (SphericalHarmonicsCUDA.cu:42–140 sh_coeffs_to_color_fast / 142–430
//  sh_coeffs_to_color_fast_vjp / 434–468 the generic fwd kernel / 710–762 the bwd kernel).
// The SH basis is the hardcoded expansion in the style of Sloan, "Efficient Spherical Harmonic
// Evaluation", JCGT 2013 (deg 0–4 are all ported — verification covers deg≤3 fwd+bwd and deg4 fwd
// — disposition #7).
//
// - one thread = one (batch, gaussian, channel). The indexing is exactly as in the upstream
//   original (offset=0, count=N).
// - dirs is an unnormalized input — the normalization happens inside the kernel with
//   precise::rsqrt (D2). Unguarded at dir=0 (rsqrt(0)=inf → NaN propagates, same as the
//   original — disposition #6).
// - colors of masked-out elements are unwritten (the standalone contract; do not zero-fill —
//   disposition #5).
// - the `+0.5` shift, the clamp, and relu_mask are a host post-op outside the kernel (owned by the
//   Phase 1c training loop).
// - fp32 only (D2 — fp16 coeffs are deferred to Phase 2), fused assemble excluded (disposition #8).
// - the weighted sum of the coefficients is an explicit left-associative metal::fma chain (the
//   lesson of the Task 4 fast-math distribution/reassociation — rounding of the same family as
//   CUDA nvcc's default FFMA contraction, with the formula unchanged).
// - bwd is a generic 1:1 atomic implementation (the D4 hot-path variant comes later): v_coeffs
//   contends across batches (with B==1 there is exactly one writer per slot — no contention,
//   bit-exact determinism), and v_dirs contends across the D channels.
#include <metal_stdlib>
#include "FunctionConstants.h"
#include "KernelParams.h"
using namespace metal;

// ===== forward device functions (SphericalHarmonicsCUDA.cu:42–140, 1:1) =====
// The coeffs and colors pointers are already shifted by the caller by gaussian_id*K*D / elem_id*D.
// With degree 0 the dir is never read (the normalization sits inside the degree>=1 branch — :54).
static inline void sh_coeffs_to_color_fast(
    const uint degree, const uint D, const uint c,
    const float3 dir,
    device const float* coeffs, // [K,D]
    device float* colors)       // [D]
{
    // degree 0 (:54)
    float result = 0.2820947917738781f * coeffs[c];
    if (degree >= 1) {
        // Internal normalization (:59-62) — precise::rsqrt (a D2 parity point), unguarded (disposition #6)
        const float inorm = precise::rsqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        const float x = dir.x * inorm;
        const float y = dir.y * inorm;
        const float z = dir.z * inorm;

        // degree 1 (:64-67) — signs/order (-y, +z, -x); coeffs index 1→y, 2→z, 3→x
        float acc = -y * coeffs[1 * D + c];
        acc = fma(z, coeffs[2 * D + c], acc);
        acc = fma(-x, coeffs[3 * D + c], acc);
        result = fma(0.48860251190292f, acc, result);
        if (degree >= 2) { // (:68-85)
            const float z2 = z * z;
            const float fTmp0B = -1.092548430592079f * z;
            const float fC1 = x * x - y * y;
            const float fS1 = 2.f * x * y;
            const float pSH6 = fma(0.9461746957575601f, z2, -0.3153915652525201f);
            const float pSH7 = fTmp0B * x;
            const float pSH5 = fTmp0B * y;
            const float pSH8 = 0.5462742152960395f * fC1;
            const float pSH4 = 0.5462742152960395f * fS1;
            acc = pSH4 * coeffs[4 * D + c];
            acc = fma(pSH5, coeffs[5 * D + c], acc);
            acc = fma(pSH6, coeffs[6 * D + c], acc);
            acc = fma(pSH7, coeffs[7 * D + c], acc);
            acc = fma(pSH8, coeffs[8 * D + c], acc);
            result += acc;
            if (degree >= 3) { // (:86-106)
                const float fTmp0C = fma(-2.285228997322329f, z2, 0.4570457994644658f);
                const float fTmp1B = 1.445305721320277f * z;
                const float fC2 = x * fC1 - y * fS1; // Chebyshev recurrence cos(2φ)→cos(3φ)
                const float fS2 = x * fS1 + y * fC1;
                const float pSH12 = z * fma(1.865881662950577f, z2, -1.119528997770346f);
                const float pSH13 = fTmp0C * x;
                const float pSH11 = fTmp0C * y;
                const float pSH14 = fTmp1B * fC1;
                const float pSH10 = fTmp1B * fS1;
                const float pSH15 = -0.5900435899266435f * fC2;
                const float pSH9 = -0.5900435899266435f * fS2;
                acc = pSH9 * coeffs[9 * D + c];
                acc = fma(pSH10, coeffs[10 * D + c], acc);
                acc = fma(pSH11, coeffs[11 * D + c], acc);
                acc = fma(pSH12, coeffs[12 * D + c], acc);
                acc = fma(pSH13, coeffs[13 * D + c], acc);
                acc = fma(pSH14, coeffs[14 * D + c], acc);
                acc = fma(pSH15, coeffs[15 * D + c], acc);
                result += acc;
                if (degree >= 4) { // (:108-134) — reachable only when K=25
                    const float fTmp0D = z * fma(-4.683325804901025f, z2, 2.007139630671868f);
                    const float fTmp1C = fma(3.31161143515146f, z2, -0.47308734787878f);
                    const float fTmp2B = -1.770130769779931f * z;
                    const float fC3 = x * fC2 - y * fS2;
                    const float fS3 = x * fS2 + y * fC2;
                    const float pSH20 =
                        fma(1.984313483298443f * z, pSH12, -1.006230589874905f * pSH6);
                    const float pSH21 = fTmp0D * x;
                    const float pSH19 = fTmp0D * y;
                    const float pSH22 = fTmp1C * fC1;
                    const float pSH18 = fTmp1C * fS1;
                    const float pSH23 = fTmp2B * fC2;
                    const float pSH17 = fTmp2B * fS2;
                    const float pSH24 = 0.6258357354491763f * fC3;
                    const float pSH16 = 0.6258357354491763f * fS3;
                    acc = pSH16 * coeffs[16 * D + c];
                    acc = fma(pSH17, coeffs[17 * D + c], acc);
                    acc = fma(pSH18, coeffs[18 * D + c], acc);
                    acc = fma(pSH19, coeffs[19 * D + c], acc);
                    acc = fma(pSH20, coeffs[20 * D + c], acc);
                    acc = fma(pSH21, coeffs[21 * D + c], acc);
                    acc = fma(pSH22, coeffs[22 * D + c], acc);
                    acc = fma(pSH23, coeffs[23 * D + c], acc);
                    acc = fma(pSH24, coeffs[24 * D + c], acc);
                    result += acc;
                }
            }
        }
    }
    colors[c] = result; // (:139)
}

// ===== backward device functions (SphericalHarmonicsCUDA.cu:142–430, 1:1) =====

// The normalization Jacobian — common to every degree exit point (:184-190, :246-252, :328-334,
// :422-428): tangential projection, then multiply by 1/‖dir_raw‖ (only once, at the very end —
// correctness pitfall checklist 3).
static inline float3 sh_dir_normalization_vjp(const float3 dir_n, const float3 v_dir_n,
                                              const float inorm)
{
    return (v_dir_n - dot(v_dir_n, dir_n) * dir_n) * inorm;
}

// Atomic accumulation into a v_coeffs element (corresponds to gpuAtomicAdd) — fp32, relaxed (spec 02 MSL table).
#define SX_SH_VADD(k, val) \
    atomic_fetch_add_explicit(&v_coeffs[(k) * D + c], (val), memory_order_relaxed)

// sh_coeffs_to_color_fast_vjp — CUDA's `v_dir != nullptr` branch is replaced by the
// kShComputeVDirs function constant (D8-5). v_coeffs MUST be zero-initialized before the call (D8-1).
// v_dir (the accumulated partials w.r.t. the normalized direction → tangential projection) is
// returned thread-locally — the global summation into v_dirs is done by the calling kernel
// (:756-761). The per-degree early-return structure is exactly as in the upstream original
// (this kernel has no simd reduction, so the D8-2 alive-flag transformation is unnecessary).
static inline void sh_coeffs_to_color_fast_vjp(
    const uint degree, const uint D, const uint c,
    const float3 dir,
    device const float* coeffs,    // [K,D]
    const float v_colors_local,    // v_colors[c] (fp32)
    device atomic_float* v_coeffs, // [K,D] fp32 accumulation
    thread float3& v_dir)
{
    // degree 0 (:159)
    SX_SH_VADD(0, 0.2820947917738781f * v_colors_local);
    if (degree < 1) return;

    const float inorm = precise::rsqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    const float x = dir.x * inorm;
    const float y = dir.y * inorm;
    const float z = dir.z * inorm;
    float v_x = 0.f, v_y = 0.f, v_z = 0.f;

    // degree 1 v_coeffs (:170-172) — the basis values are themselves the Jacobian
    SX_SH_VADD(1, -0.48860251190292f * y * v_colors_local);
    SX_SH_VADD(2, 0.48860251190292f * z * v_colors_local);
    SX_SH_VADD(3, -0.48860251190292f * x * v_colors_local);

    if (kShComputeVDirs) { // degree 1 v_dir contribution (:174-179)
        v_x += -0.48860251190292f * coeffs[3 * D + c] * v_colors_local;
        v_y += -0.48860251190292f * coeffs[1 * D + c] * v_colors_local;
        v_z += 0.48860251190292f * coeffs[2 * D + c] * v_colors_local;
    }
    if (degree < 2) { // (:180-193)
        if (kShComputeVDirs)
            v_dir = sh_dir_normalization_vjp(float3(x, y, z), float3(v_x, v_y, v_z), inorm);
        return;
    }

    // degree 2 basis (:195-208)
    const float z2 = z * z;
    const float fTmp0B = -1.092548430592079f * z;
    const float fC1 = x * x - y * y;
    const float fS1 = 2.f * x * y;
    const float pSH6 = fma(0.9461746957575601f, z2, -0.3153915652525201f);
    const float pSH7 = fTmp0B * x;
    const float pSH5 = fTmp0B * y;
    const float pSH8 = 0.5462742152960395f * fC1;
    const float pSH4 = 0.5462742152960395f * fS1;
    SX_SH_VADD(4, pSH4 * v_colors_local);
    SX_SH_VADD(5, pSH5 * v_colors_local);
    SX_SH_VADD(6, pSH6 * v_colors_local);
    SX_SH_VADD(7, pSH7 * v_colors_local);
    SX_SH_VADD(8, pSH8 * v_colors_local);

    // Intermediate partials referenced by later degree blocks — CUDA declares them uninitialized (:210).
    // With kShComputeVDirs=false the using blocks are removed as well, so the value does not matter
    // (the 0 is only to suppress warnings).
    float fC1_x = 0.f, fC1_y = 0.f, fS1_x = 0.f, fS1_y = 0.f, pSH6_z = 0.f;
    if (kShComputeVDirs) { // degree 2 partials (:211-240)
        const float fTmp0B_z = -1.092548430592079f;
        fC1_x = 2.f * x;
        fC1_y = -2.f * y;
        fS1_x = 2.f * y;
        fS1_y = 2.f * x;
        pSH6_z = 2.f * 0.9461746957575601f * z;
        const float pSH7_x = fTmp0B;
        const float pSH7_z = fTmp0B_z * x;
        const float pSH5_y = fTmp0B;
        const float pSH5_z = fTmp0B_z * y;
        const float pSH8_x = 0.5462742152960395f * fC1_x;
        const float pSH8_y = 0.5462742152960395f * fC1_y;
        const float pSH4_x = 0.5462742152960395f * fS1_x;
        const float pSH4_y = 0.5462742152960395f * fS1_y;

        float acc = pSH4_x * coeffs[4 * D + c];
        acc = fma(pSH8_x, coeffs[8 * D + c], acc);
        acc = fma(pSH7_x, coeffs[7 * D + c], acc);
        v_x = fma(v_colors_local, acc, v_x);
        acc = pSH4_y * coeffs[4 * D + c];
        acc = fma(pSH8_y, coeffs[8 * D + c], acc);
        acc = fma(pSH5_y, coeffs[5 * D + c], acc);
        v_y = fma(v_colors_local, acc, v_y);
        acc = pSH6_z * coeffs[6 * D + c];
        acc = fma(pSH7_z, coeffs[7 * D + c], acc);
        acc = fma(pSH5_z, coeffs[5 * D + c], acc);
        v_z = fma(v_colors_local, acc, v_z);
    }
    if (degree < 3) { // (:242-255)
        if (kShComputeVDirs)
            v_dir = sh_dir_normalization_vjp(float3(x, y, z), float3(v_x, v_y, v_z), inorm);
        return;
    }

    // degree 3 basis (:257-274)
    const float fTmp0C = fma(-2.285228997322329f, z2, 0.4570457994644658f);
    const float fTmp1B = 1.445305721320277f * z;
    const float fC2 = x * fC1 - y * fS1;
    const float fS2 = x * fS1 + y * fC1;
    const float pSH12 = z * fma(1.865881662950577f, z2, -1.119528997770346f);
    const float pSH13 = fTmp0C * x;
    const float pSH11 = fTmp0C * y;
    const float pSH14 = fTmp1B * fC1;
    const float pSH10 = fTmp1B * fS1;
    const float pSH15 = -0.5900435899266435f * fC2;
    const float pSH9 = -0.5900435899266435f * fS2;
    SX_SH_VADD(9, pSH9 * v_colors_local);
    SX_SH_VADD(10, pSH10 * v_colors_local);
    SX_SH_VADD(11, pSH11 * v_colors_local);
    SX_SH_VADD(12, pSH12 * v_colors_local);
    SX_SH_VADD(13, pSH13 * v_colors_local);
    SX_SH_VADD(14, pSH14 * v_colors_local);
    SX_SH_VADD(15, pSH15 * v_colors_local);

    float fC2_x = 0.f, fC2_y = 0.f, fS2_x = 0.f, fS2_y = 0.f, pSH12_z = 0.f;
    if (kShComputeVDirs) { // degree 3 partials (:276-321)
        const float fTmp0C_z = -2.285228997322329f * 2.f * z;
        const float fTmp1B_z = 1.445305721320277f;
        fC2_x = fC1 + x * fC1_x - y * fS1_x;
        fC2_y = x * fC1_y - fS1 - y * fS1_y;
        fS2_x = fS1 + x * fS1_x + y * fC1_x;
        fS2_y = x * fS1_y + fC1 + y * fC1_y;
        pSH12_z = fma(3.f * 1.865881662950577f, z2, -1.119528997770346f);
        const float pSH13_x = fTmp0C;
        const float pSH13_z = fTmp0C_z * x;
        const float pSH11_y = fTmp0C;
        const float pSH11_z = fTmp0C_z * y;
        const float pSH14_x = fTmp1B * fC1_x;
        const float pSH14_y = fTmp1B * fC1_y;
        const float pSH14_z = fTmp1B_z * fC1;
        const float pSH10_x = fTmp1B * fS1_x;
        const float pSH10_y = fTmp1B * fS1_y;
        const float pSH10_z = fTmp1B_z * fS1;
        const float pSH15_x = -0.5900435899266435f * fC2_x;
        const float pSH15_y = -0.5900435899266435f * fC2_y;
        const float pSH9_x = -0.5900435899266435f * fS2_x;
        const float pSH9_y = -0.5900435899266435f * fS2_y;

        float acc = pSH9_x * coeffs[9 * D + c]; // v_x term order (:302-307)
        acc = fma(pSH15_x, coeffs[15 * D + c], acc);
        acc = fma(pSH10_x, coeffs[10 * D + c], acc);
        acc = fma(pSH14_x, coeffs[14 * D + c], acc);
        acc = fma(pSH13_x, coeffs[13 * D + c], acc);
        v_x = fma(v_colors_local, acc, v_x);
        acc = pSH9_y * coeffs[9 * D + c]; // v_y term order (:309-314)
        acc = fma(pSH15_y, coeffs[15 * D + c], acc);
        acc = fma(pSH10_y, coeffs[10 * D + c], acc);
        acc = fma(pSH14_y, coeffs[14 * D + c], acc);
        acc = fma(pSH11_y, coeffs[11 * D + c], acc);
        v_y = fma(v_colors_local, acc, v_y);
        acc = pSH12_z * coeffs[12 * D + c]; // v_z term order (:316-321)
        acc = fma(pSH13_z, coeffs[13 * D + c], acc);
        acc = fma(pSH11_z, coeffs[11 * D + c], acc);
        acc = fma(pSH14_z, coeffs[14 * D + c], acc);
        acc = fma(pSH10_z, coeffs[10 * D + c], acc);
        v_z = fma(v_colors_local, acc, v_z);
    }
    if (degree < 4) { // (:324-337)
        if (kShComputeVDirs)
            v_dir = sh_dir_normalization_vjp(float3(x, y, z), float3(v_x, v_y, v_z), inorm);
        return;
    }

    // degree 4 basis (:339-361) — reached only when K=25
    const float fTmp0D = z * fma(-4.683325804901025f, z2, 2.007139630671868f);
    const float fTmp1C = fma(3.31161143515146f, z2, -0.47308734787878f);
    const float fTmp2B = -1.770130769779931f * z;
    const float fC3 = x * fC2 - y * fS2;
    const float fS3 = x * fS2 + y * fC2;
    // The upstream vjp writes `+ -1.006230589874905f` (:344) — mathematically identical to the `-` in the forward
    const float pSH20 = fma(1.984313483298443f * z, pSH12, -1.006230589874905f * pSH6);
    const float pSH21 = fTmp0D * x;
    const float pSH19 = fTmp0D * y;
    const float pSH22 = fTmp1C * fC1;
    const float pSH18 = fTmp1C * fS1;
    const float pSH23 = fTmp2B * fC2;
    const float pSH17 = fTmp2B * fS2;
    const float pSH24 = 0.6258357354491763f * fC3;
    const float pSH16 = 0.6258357354491763f * fS3;
    SX_SH_VADD(16, pSH16 * v_colors_local);
    SX_SH_VADD(17, pSH17 * v_colors_local);
    SX_SH_VADD(18, pSH18 * v_colors_local);
    SX_SH_VADD(19, pSH19 * v_colors_local);
    SX_SH_VADD(20, pSH20 * v_colors_local);
    SX_SH_VADD(21, pSH21 * v_colors_local);
    SX_SH_VADD(22, pSH22 * v_colors_local);
    SX_SH_VADD(23, pSH23 * v_colors_local);
    SX_SH_VADD(24, pSH24 * v_colors_local);

    if (kShComputeVDirs) { // degree 4 partials (:363-420)
        const float fTmp0D_z = fma(3.f * -4.683325804901025f, z2, 2.007139630671868f);
        const float fTmp1C_z = 2.f * 3.31161143515146f * z;
        const float fTmp2B_z = -1.770130769779931f;
        const float fC3_x = fC2 + x * fC2_x - y * fS2_x;
        const float fC3_y = x * fC2_y - fS2 - y * fS2_y;
        const float fS3_x = fS2 + y * fC2_x + x * fS2_x;
        const float fS3_y = x * fS2_y + fC2 + y * fC2_y;
        // pSH20_z product-rule expansion (:375 — correctness pitfall checklist 2)
        const float pSH20_z = fma(1.984313483298443f, fma(z, pSH12_z, pSH12),
                                  -1.006230589874905f * pSH6_z);
        const float pSH21_x = fTmp0D;
        const float pSH21_z = fTmp0D_z * x;
        const float pSH19_y = fTmp0D;
        const float pSH19_z = fTmp0D_z * y;
        const float pSH22_x = fTmp1C * fC1_x;
        const float pSH22_y = fTmp1C * fC1_y;
        const float pSH22_z = fTmp1C_z * fC1;
        const float pSH18_x = fTmp1C * fS1_x;
        const float pSH18_y = fTmp1C * fS1_y;
        const float pSH18_z = fTmp1C_z * fS1;
        const float pSH23_x = fTmp2B * fC2_x;
        const float pSH23_y = fTmp2B * fC2_y;
        const float pSH23_z = fTmp2B_z * fC2;
        const float pSH17_x = fTmp2B * fS2_x;
        const float pSH17_y = fTmp2B * fS2_y;
        const float pSH17_z = fTmp2B_z * fS2;
        const float pSH24_x = 0.6258357354491763f * fC3_x;
        const float pSH24_y = 0.6258357354491763f * fC3_y;
        const float pSH16_x = 0.6258357354491763f * fS3_x;
        const float pSH16_y = 0.6258357354491763f * fS3_y;

        float acc = pSH16_x * coeffs[16 * D + c]; // v_x term order (:397-404)
        acc = fma(pSH24_x, coeffs[24 * D + c], acc);
        acc = fma(pSH17_x, coeffs[17 * D + c], acc);
        acc = fma(pSH23_x, coeffs[23 * D + c], acc);
        acc = fma(pSH18_x, coeffs[18 * D + c], acc);
        acc = fma(pSH22_x, coeffs[22 * D + c], acc);
        acc = fma(pSH21_x, coeffs[21 * D + c], acc);
        v_x = fma(v_colors_local, acc, v_x);
        acc = pSH16_y * coeffs[16 * D + c]; // v_y term order (:405-412)
        acc = fma(pSH24_y, coeffs[24 * D + c], acc);
        acc = fma(pSH17_y, coeffs[17 * D + c], acc);
        acc = fma(pSH23_y, coeffs[23 * D + c], acc);
        acc = fma(pSH18_y, coeffs[18 * D + c], acc);
        acc = fma(pSH22_y, coeffs[22 * D + c], acc);
        acc = fma(pSH19_y, coeffs[19 * D + c], acc);
        v_y = fma(v_colors_local, acc, v_y);
        acc = pSH20_z * coeffs[20 * D + c]; // v_z term order (:413-420)
        acc = fma(pSH21_z, coeffs[21 * D + c], acc);
        acc = fma(pSH19_z, coeffs[19 * D + c], acc);
        acc = fma(pSH22_z, coeffs[22 * D + c], acc);
        acc = fma(pSH18_z, coeffs[18 * D + c], acc);
        acc = fma(pSH23_z, coeffs[23 * D + c], acc);
        acc = fma(pSH17_z, coeffs[17 * D + c], acc);
        v_z = fma(v_colors_local, acc, v_z);

        // (:422-428) — degree 4 is the final exit point, so finish unconditionally
        v_dir = sh_dir_normalization_vjp(float3(x, y, z), float3(v_x, v_y, v_z), inorm);
    }
}

#undef SX_SH_VADD

// ===== Kernels =====

// spherical_harmonics_fwd_kernel (generic, :434–468) — a pure map (no atomics, no barriers).
// Masked-out elements are unwritten (early return — disposition #5; the output buffer is
// uninitialized, matching at::empty).
// Dispatch: dispatchThreads(B*N*D, tg 256).
kernel void sh_fwd(
    device const packed_float3* dirs [[buffer(0)]], // [B*N] unnormalized
    device const float* coeffs       [[buffer(1)]], // [N,K,D]
    device const uchar* masks        [[buffer(2), function_constant(kShHasMasks)]], // [B*N] 0/1
    device float* colors             [[buffer(3)]], // [B*N,D]
    constant SHParams& P             [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= P.B * P.N * P.D) return; // defensive, since dispatchThreads is used (:452-455)
    const uint elem = idx / P.D;        // = batch_id * N + gaussian_id (offset=0, :456-459)
    const uint c = idx % P.D;
    const uint gaussian_id = elem % P.N;
    if (kShHasMasks && masks[elem] == 0) return; // masked out → colors unwritten (:461-464)
    sh_coeffs_to_color_fast(P.degrees_to_use, P.D, c, float3(dirs[elem]),
                            coeffs + gaussian_id * P.K * P.D, colors + elem * P.D);
}

// spherical_harmonics_bwd_kernel (:710–762) — a generic 1:1 atomic implementation (the first
// implementation; the D4 hot-path variant comes later).
// v_coeffs[gaussian]: coeffs is shared along the batch axis, so batches contend (with B==1 there
// is exactly one writer per slot — no contention, bit-exact determinism). v_dirs[elem]: the D
// channel threads of the same elem contend.
// Both buffers MUST be zero-initialized before the dispatch (D8-1 — host encode_zero_fill).
kernel void sh_bwd(
    device const packed_float3* dirs [[buffer(0)]],
    device const float* coeffs       [[buffer(1)]],
    device const uchar* masks        [[buffer(2), function_constant(kShHasMasks)]],
    device const float* v_colors     [[buffer(3)]], // [B*N,D]
    device atomic_float* v_coeffs    [[buffer(4)]], // [N,K,D] fp32 accumulation
    device atomic_float* v_dirs      [[buffer(5), function_constant(kShComputeVDirs)]], // [B*N,3]
    constant SHParams& P             [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= P.B * P.N * P.D) return;
    const uint elem = idx / P.D;
    const uint c = idx % P.D;
    const uint gaussian_id = elem % P.N;
    if (kShHasMasks && masks[elem] == 0) return; // no gradient contribution — v_* stays 0 (:739-742)
    float3 v_dir = float3(0.f);
    sh_coeffs_to_color_fast_vjp(P.degrees_to_use, P.D, c, float3(dirs[elem]),
                                coeffs + gaussian_id * P.K * P.D, v_colors[elem * P.D + c],
                                v_coeffs + gaussian_id * P.K * P.D, v_dir);
    if (kShComputeVDirs) { // sum the channel-local accumulation into the global buffer (:756-761)
        atomic_fetch_add_explicit(&v_dirs[elem * 3 + 0], v_dir.x, memory_order_relaxed);
        atomic_fetch_add_explicit(&v_dirs[elem * 3 + 1], v_dir.y, memory_order_relaxed);
        atomic_fetch_add_explicit(&v_dirs[elem * 3 + 2], v_dir.z, memory_order_relaxed);
    }
}
