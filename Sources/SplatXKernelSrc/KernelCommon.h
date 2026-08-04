// splatx-metal — shared device helpers (MSL-only header)
// Derived from nerfstudio-project/gsplat (Apache-2.0), commit 77ab983 (Utils.cuh).
// Reimplemented for Metal/C++.
//
// D5-1: every .metal links into a single metallib, so device helpers are defined in this header
// as static inline only (no duplicate symbols). Separate from the host C++ implementation (Math.cpp).
#ifndef SPLATX_KERNEL_COMMON_H
#define SPLATX_KERNEL_COMMON_H
#ifdef __METAL_VERSION__

#include <metal_stdlib>

// quat → rotation matrix. A 1:1 port of gsplat Utils.cuh:228–251.
// ProjectionEWA and AdamMCMC (mcmc_perturb_positions) share this single definition (D5-1, spec 06).
// - quat storage order = [w, x, y, z] (Utils.cuh:230, D3)
// - normalization: precise::rsqrt (matches the rsqrt_parse_safe device path, D2)
// - follows the x→y→z→w accumulation order of the sum of squares
//   (Utils.cuh:232, spec 01 Verification Correction 6 — for bit-exact comparison)
static inline metal::float3x3 quat_to_rotmat(const float4 quat)
{
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    // normalize
    float inv_norm  = metal::precise::rsqrt(x * x + y * y + z * z + w * w);
    x              *= inv_norm;
    y              *= inv_norm;
    z              *= inv_norm;
    w              *= inv_norm;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    // Identical to the glm mat3 constructor — built column by column (spec 01 F2)
    return metal::float3x3(
        float3(1.f - 2.f * (y2 + z2), 2.f * (xy + wz), 2.f * (xz - wy)), // 1st col
        float3(2.f * (xy - wz), 1.f - 2.f * (x2 + z2), 2.f * (yz + wx)), // 2nd col
        float3(2.f * (xz + wy), 2.f * (yz - wx), 1.f - 2.f * (x2 + y2))  // 3rd col
    );
}

// quat+scale → 3D covariance. A 1:1 port of gsplat Utils.cuh:285–302 (the covar output of
// quat_scale_to_covar_preci) — the preci output is omitted because the fused kernel does not
// use it (spec 01 F2).
// M = R_g·diag(s) (glm: column j = column j of R × s[j]), covar = M·Mᵀ.
static inline metal::float3x3 quat_scale_to_covar(const float4 quat, const float3 scale)
{
    const metal::float3x3 R = quat_to_rotmat(quat);
    const metal::float3x3 M =
        metal::float3x3(R[0] * scale.x, R[1] * scale.y, R[2] * scale.z);
    return M * metal::transpose(M);
}

// 2×2 inverse — a 1:1 match of the glm::inverse(mat2) analytic formula (multiply by the
// reciprocal 1/det, column-major component layout; spec 01 F6 / the MSL type mapping table).
// det > 0 is guaranteed by the caller (called after the add_blur det<=0 culling).
static inline metal::float2x2 mat2_inv(const metal::float2x2 M)
{
    const float rdet = 1.f / (M[0][0] * M[1][1] - M[1][0] * M[0][1]);
    return metal::float2x2(float2(M[1][1] * rdet, -M[0][1] * rdet),
                           float2(-M[1][0] * rdet, M[0][0] * rdet));
}

// Weight evaluation for a (gaussian, pixel) pair — a 1:1 port of gsplat
// RasterizeToPixels3DGSDevice.cuh:37-55 (GaussianWeight + eval_gaussian_weight).
// rasterize fwd and bwd share it (spec 05).
// - conic = the upper-triangular (triu) part of the inverse 2D covariance (A=xx, B=xy, C=yy),
//   delta = mean2d − pixel_center
// - vis: CUDA __expf → metal::precise::exp (a D2 parity point — close to the CPU std::exp golden,
//   00-index disposition #21)
// - if valid == false, both fwd and bwd skip this pair (Device.cuh:53 — sigma<0 or alpha<1/255)
struct GaussianWeight {
    float vis;   // precise::exp(-sigma)
    float alpha; // min(kMaxAlpha, opac * vis)
    bool  valid; // !(sigma < 0 || alpha < kAlphaThreshold)
};

static inline GaussianWeight eval_gaussian_weight(const packed_float3 conic, const float dx,
                                                  const float dy, const float opac)
{
    const float sigma =
        0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy; // Device.cuh:47
    const float vis   = metal::precise::exp(-sigma);                       // :48
    const float alpha = metal::min(kMaxAlpha, opac * vis);                 // :49
    GaussianWeight out;
    out.vis   = vis;
    out.alpha = alpha;
    out.valid = !(sigma < 0.f || alpha < kAlphaThreshold);                 // :53
    return out;
}

// Standard sigmoid — a 1:1 port of gsplat MCMCPerturbCUDA.cu:22–25 (sigmoid_standard).
// exp is precise:: (a D2 parity point — close to the CPU std::exp golden). For a large negative x,
// exp(-x) overflows f32 (inf) and 0 is returned — the same limiting behavior as CUDA expf and
// host expf (no NaN).
static inline float sigmoid_standard(float x)
{
    return 1.0f / (1.0f + metal::precise::exp(-x));
}

// Add further shared helpers to this file as static inline, in the porting task of the kernel
// that needs them (D5-1).

#endif // __METAL_VERSION__
#endif // SPLATX_KERNEL_COMMON_H
