// ProjectionEWA.metal — projection_ewa_3dgs_fused_fwd/bwd (spec 01 F0–F8 / B0–B8 +
// Verification Notes Corrections 1–6, merged).
// Derived from nerfstudio-project/gsplat (Apache-2.0), commit 77ab983.
// Reimplemented for Metal/C++.
// Original: (ProjectionEWA3DGSFused.cu:38–219 fwd / 376–638 bwd, Utils.cuh: posW2C(_VJP)/covarW2C(_VJP)/
//  quat_scale_to_covar_preci(+_vjp)/quat_to_rotmat_vjp/persp_proj(_vjp)/add_blur(_vjp)/
//  inverse_vjp, and the Common.h constants).
//
// Phase 1: PINHOLE only (disposition #3 — ORTHO/FISHEYE are not ported; the host validates
// camera_model).
// B is fixed at 1 (D3) — the indexing is (cid, gid), gid-fastest (F0/B0).
#include <metal_stdlib>
#include "KernelConstants.h"
#include "FunctionConstants.h"
#include "KernelParams.h"
#include "KernelCommon.h"
using namespace metal;

// Replaces the covars (nullptr) branch: kProjUseCovars=false ↔ the quat/scale path (D8-5).
// A derived function constant — it gates the quats/scales buffer arguments (the name is unique
// library-wide, D5-3).
constant bool kProjQuatScalePath = !kProjUseCovars;

// persp_proj — a 1:1 port of gsplat Utils.cuh:567–607 (PINHOLE, F4a).
// glm mat3x2 (3 columns × 2 rows) = MSL float3x2, with the J[col][row] indexing kept as is
// (spec 01 type mapping).
// mean2d uses the unclamped x·rz; only the 3rd column of J uses the FOV-clamped tx/ty (paired with
// the backward branch).
static inline void persp_proj(
    const float3 mean3d,
    const float3x3 cov3d,
    const float fx, const float fy, const float cx, const float cy,
    const uint width, const uint height,
    thread float2x2& cov2d, thread float2& mean2d)
{
    const float x = mean3d.x, y = mean3d.y, z = mean3d.z;

    const float tan_fovx  = 0.5f * (float)width / fx;
    const float tan_fovy  = 0.5f * (float)height / fy;
    const float lim_x_pos = ((float)width - cx) / fx + 0.3f * tan_fovx;
    const float lim_x_neg = cx / fx + 0.3f * tan_fovx;
    const float lim_y_pos = ((float)height - cy) / fy + 0.3f * tan_fovy;
    const float lim_y_neg = cy / fy + 0.3f * tan_fovy;

    const float rz  = 1.f / z;
    const float rz2 = rz * rz;
    const float tx  = z * metal::min(lim_x_pos, metal::max(-lim_x_neg, x * rz));
    const float ty  = z * metal::min(lim_y_pos, metal::max(-lim_y_neg, y * rz));

    const float3x2 J = float3x2(
        float2(fx * rz, 0.f),                     // 1st col
        float2(0.f, fy * rz),                     // 2nd col
        float2(-fx * tx * rz2, -fy * ty * rz2));  // 3rd col
    cov2d  = J * cov3d * metal::transpose(J);
    mean2d = float2(fx * x * rz + cx, fy * y * rz + cy);
}

// The spec 01 fwd kernel (ProjectionEWA3DGSFused.cu:38–219). A pure map — no atomics, no barriers,
// no threadgroup memory. On culling, only the 2 radii slots are written with 0; the other output
// buffers are left unwritten (D3 — do not zero-fill them).
// Dispatch: dispatchThreads(C*N, tg 256).
kernel void projection_ewa_3dgs_fused_fwd(
    device const float* means         [[buffer(0)]],  // [N,3]
    device const float* covars        [[buffer(1), function_constant(kProjUseCovars)]],   // [N,6] triu
    device const float* quats         [[buffer(2), function_constant(kProjQuatScalePath)]], // [N,4] wxyz
    device const float* scales        [[buffer(3), function_constant(kProjQuatScalePath)]], // [N,3]
    device const float* opacities     [[buffer(4), function_constant(kProjHasOpacities)]],  // [N]
    device const float* viewmats      [[buffer(5)]],  // [C,16] row-major
    device const float* Ks            [[buffer(6)]],  // [C,9] row-major
    device int*         radii         [[buffer(7)]],  // [C,N,2]
    device float*       means2d       [[buffer(8)]],  // [C,N,2]
    device float*       depths        [[buffer(9)]],  // [C,N]
    device float*       conics        [[buffer(10)]], // [C,N,3]
    device float*       compensations [[buffer(11), function_constant(kProjCalcComp)]],     // [C,N]
    constant ProjectionParams& P      [[buffer(12)]],
    uint idx [[thread_position_in_grid]])
{
    // F0: thread indexing (lines 66–81, B=1). The guard is defensive, since dispatchThreads is used.
    if (idx >= P.C * P.N) return;
    const uint cid = idx / P.N;
    const uint gid = idx % P.N;
    const uint gi  = idx; // global_idx = cid*N + gid

    device const float* mw = means + gid * 3;
    device const float* vm = viewmats + cid * 16;
    device const float* K  = Ks + cid * 9;

    // F1: row-major viewmats [R|t] → rebuilt in glm column-major (lines 83–95)
    const float3x3 R = float3x3(
        float3(vm[0], vm[4], vm[8]),   // 1st col
        float3(vm[1], vm[5], vm[9]),   // 2nd col
        float3(vm[2], vm[6], vm[10])); // 3rd col
    const float3 t = float3(vm[3], vm[7], vm[11]);
    const float3 mean_c = R * float3(mw[0], mw[1], mw[2]) + t; // posW2C
    // near/far culling (lines 100–105) — equality survives
    if (mean_c.z < P.near_plane || mean_c.z > P.far_plane) {
        radii[gi * 2] = 0;
        radii[gi * 2 + 1] = 0;
        return;
    }

    // F2: the source of the 3D covariance (lines 107–130) — the branch is removed by function constant specialization
    float3x3 covar;
    if (kProjUseCovars) {
        device const float* cv = covars + gid * 6; // [N,6] triu (c00,c01,c02,c11,c12,c22)
        covar = float3x3(float3(cv[0], cv[1], cv[2]),
                         float3(cv[1], cv[3], cv[4]),
                         float3(cv[2], cv[4], cv[5]));
    } else {
        covar = quat_scale_to_covar(
            float4(quats[gid * 4], quats[gid * 4 + 1], quats[gid * 4 + 2], quats[gid * 4 + 3]),
            float3(scales[gid * 3], scales[gid * 3 + 1], scales[gid * 3 + 2]));
    }
    // F3: covar_c = R·Σ·Rᵀ (covarW2C, Utils.cuh:116–123)
    const float3x3 covar_c = R * covar * metal::transpose(R);

    // F4a: PINHOLE persp_proj (the host validates camera_model — Phase 1 is PINHOLE only)
    float2x2 covar2d;
    float2 mean2d;
    persp_proj(mean_c, covar_c, K[0], K[4], K[2], K[5], P.image_width, P.image_height,
               covar2d, mean2d);

    // F5: add_blur (Utils.cuh:455–463) — the compensation is computed before the det<=0 culling
    // (the order of the upstream original)
    const float det_orig = covar2d[0][0] * covar2d[1][1] - covar2d[0][1] * covar2d[1][0];
    covar2d[0][0] += P.eps2d;
    covar2d[1][1] += P.eps2d;
    const float det = covar2d[0][0] * covar2d[1][1] - covar2d[0][1] * covar2d[1][0];
    const float compensation =
        metal::sqrt(metal::max(kMinCompensation * kMinCompensation, det_orig / det));
    if (det <= 0.f) {
        radii[gi * 2] = 0;
        radii[gi * 2 + 1] = 0;
        return;
    }

    // F6: conic = inverse(covar2d) (lines 160–161)
    const float2x2 covar2d_inv = mat2_inv(covar2d);

    // F7: the opacity-aware radius + culling (lines 163–204)
    float extend = kGaussianExtend;
    if (kProjHasOpacities) {
        float opacity = opacities[gid];
        if (kProjCalcComp) opacity *= compensation; // only in AA mode (lines 169–172)
        if (opacity < kAlphaThreshold) {
            radii[gi * 2] = 0;
            radii[gi * 2 + 1] = 0;
            return;
        }
        // opacity-aware bbox (arXiv:2402.00525 §B.2, line 180). __logf → precise::log (D2)
        extend = metal::min(kGaussianExtend,
                            metal::sqrt(2.0f * metal::precise::log(opacity / kAlphaThreshold)));
    }
    const float radius_x = metal::ceil(extend * metal::sqrt(covar2d[0][0]));
    const float radius_y = metal::ceil(extend * metal::sqrt(covar2d[1][1]));
    // radius_clip is an AND — it culls only when both are small (line 188)
    if (radius_x <= P.radius_clip && radius_y <= P.radius_clip) {
        radii[gi * 2] = 0;
        radii[gi * 2 + 1] = 0;
        return;
    }
    // Culling outside the image area (lines 196–199) — the boundary equality culls as well
    if (mean2d.x + radius_x <= 0.f || mean2d.x - radius_x >= (float)P.image_width ||
        mean2d.y + radius_y <= 0.f || mean2d.y - radius_y >= (float)P.image_height) {
        radii[gi * 2] = 0;
        radii[gi * 2 + 1] = 0;
        return;
    }

    // F8: outputs (lines 206–218)
    radii[gi * 2] = (int)radius_x;
    radii[gi * 2 + 1] = (int)radius_y;
    means2d[gi * 2] = mean2d.x;
    means2d[gi * 2 + 1] = mean2d.y;
    depths[gi] = mean_c.z;
    conics[gi * 3] = covar2d_inv[0][0];
    conics[gi * 3 + 1] = covar2d_inv[0][1];
    conics[gi * 3 + 2] = covar2d_inv[1][1];
    if (kProjCalcComp) compensations[gi] = compensation;
}

// ============================== backward (spec 01 B0–B8) ==============================
// VJP helpers for the projection only — only this kernel uses them, so they live in this TU rather
// than in KernelCommon.h (D5-1).

// inverse_vjp — Utils.cuh:448–453 (B1). P = M⁻¹ ⇒ dL/dM = -P·(dL/dP)·P.
// (-= is IEEE-identical to += -(…) — sign inversion is an exact operation.)
static inline void inverse_vjp(const float2x2 Minv, const float2x2 v_Minv,
                               thread float2x2& v_M)
{
    v_M -= Minv * v_Minv * Minv;
}

// add_blur_vjp — Utils.cuh:465–492 (B2). comp² = det(M)/det(M+eps2d·I) ⇒
// d(comp²)/dM = (1-comp²)·conic_blur - eps2d·det(conic_blur)·I.
// Caution (Correction 2): line 486 of the original, `v_compensation * 0.5 / (compensation + 1e-6)`,
// uses double literals and is the only promotion to fp64 in CUDA — MSL has no double, so this is
// computed in float (D2: reflected separately in the tolerances of the fixture that enables
// v_compensations). The fwd MIN_COMPENSATION clamp is ignored in the backward (exactly as in the
// upstream original).
static inline void add_blur_vjp(const float eps2d, const float2x2 conic_blur,
                                const float compensation, const float v_compensation,
                                thread float2x2& v_covar)
{
    const float det_conic_blur =
        conic_blur[0][0] * conic_blur[1][1] - conic_blur[0][1] * conic_blur[1][0];
    const float v_sqr_comp = v_compensation * 0.5f / (compensation + 1e-6f);
    const float one_minus_sqr_comp = 1.f - compensation * compensation;
    v_covar[0][0] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][0] - eps2d * det_conic_blur);
    v_covar[0][1] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][1]);
    v_covar[1][0] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][0]);
    v_covar[1][1] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][1] - eps2d * det_conic_blur);
}

// persp_proj_vjp — a 1:1 port of Utils.cuh:609–690 (B4a, PINHOLE). It recomputes (rematerializes)
// all of lim/rz/tx/ty/J from the fwd persp_proj, then runs the VJP along the 3 paths covar/mean/J.
// The v_J[col][row] indexing is 1:1 with glm mat3x2 (spec 01 type mapping). The FOV clamp branch
// (lines 670–685): unclamped lanes are routed through the d(tx)/dx = 1 path, clamped lanes through
// the d(tx)/dz = tx/z path.
static inline void persp_proj_vjp(
    const float3 mean3d,
    const float3x3 cov3d,
    const float fx, const float fy, const float cx, const float cy,
    const uint width, const uint height,
    const float2x2 v_cov2d, const float2 v_mean2d,
    thread float3& v_mean3d, thread float3x3& v_cov3d)
{
    const float x = mean3d.x, y = mean3d.y, z = mean3d.z;

    const float tan_fovx  = 0.5f * (float)width / fx;
    const float tan_fovy  = 0.5f * (float)height / fy;
    const float lim_x_pos = ((float)width - cx) / fx + 0.3f * tan_fovx;
    const float lim_x_neg = cx / fx + 0.3f * tan_fovx;
    const float lim_y_pos = ((float)height - cy) / fy + 0.3f * tan_fovy;
    const float lim_y_neg = cy / fy + 0.3f * tan_fovy;

    const float rz  = 1.f / z;
    const float rz2 = rz * rz;
    const float tx  = z * metal::min(lim_x_pos, metal::max(-lim_x_neg, x * rz));
    const float ty  = z * metal::min(lim_y_pos, metal::max(-lim_y_neg, y * rz));

    const float3x2 J = float3x2(
        float2(fx * rz, 0.f),                     // 1st col
        float2(0.f, fy * rz),                     // 2nd col
        float2(-fx * tx * rz2, -fy * ty * rz2));  // 3rd col

    // (1) the covar path: cov2d = J·V·Jᵀ ⇒ dL/dV = Jᵀ·G·J (line 654)
    v_cov3d += metal::transpose(J) * v_cov2d * J;

    // (2) the mean2d path — uses the unclamped x and y (lines 659–660)
    v_mean3d += float3(fx * rz * v_mean2d[0], fy * rz * v_mean2d[1],
                       -(fx * x * v_mean2d[0] + fy * y * v_mean2d[1]) * rz2);

    // (3) the J path: dL/dJ = G·J·Vᵀ + Gᵀ·J·V (line 667)
    const float rz3 = rz2 * rz;
    const float3x2 v_J =
        v_cov2d * J * metal::transpose(cov3d) + metal::transpose(v_cov2d) * J * cov3d;

    // (4) the FOV clamp branch (lines 670–685)
    if (x * rz <= lim_x_pos && x * rz >= -lim_x_neg) {
        v_mean3d.x += -fx * rz2 * v_J[2][0];
    } else {
        v_mean3d.z += -fx * rz3 * v_J[2][0] * tx;
    }
    if (y * rz <= lim_y_pos && y * rz >= -lim_y_neg) {
        v_mean3d.y += -fy * rz2 * v_J[2][1];
    } else {
        v_mean3d.z += -fy * rz3 * v_J[2][1] * ty;
    }
    // (5) derivatives of the J entries with respect to z (lines 686–689)
    v_mean3d.z += -fx * rz2 * v_J[0][0] - fy * rz2 * v_J[1][1] +
                  2.f * fx * tx * rz3 * v_J[2][0] + 2.f * fy * ty * rz3 * v_J[2][1];
}

// posW2C_VJP — Utils.cuh:94–113 (B6). pC = R·pW + t.
// glm::outerProduct(c, r): result[col j][row i] = c[i]·r[j] ⇒ column j = c·r[j] (spec 01 type mapping).
static inline void pos_w2c_vjp(const float3x3 R, const float3 pW, const float3 v_pC,
                               thread float3x3& v_R, thread float3& v_t, thread float3& v_pW)
{
    v_R += float3x3(v_pC * pW.x, v_pC * pW.y, v_pC * pW.z); // outerProduct(v_pC, pW)
    v_t += v_pC;
    v_pW += metal::transpose(R) * v_pC;
}

// covarW2C_VJP — Utils.cuh:128–146 (B6). covarC = R·covarW·Rᵀ.
static inline void covar_w2c_vjp(const float3x3 R, const float3x3 covarW,
                                 const float3x3 v_covarC,
                                 thread float3x3& v_R, thread float3x3& v_covarW)
{
    v_R += v_covarC * R * metal::transpose(covarW) +
           metal::transpose(v_covarC) * R * covarW;
    v_covarW += metal::transpose(R) * v_covarC * R;
}

// quat_to_rotmat_vjp — Utils.cuh:253–283 (B7). Includes the normalization VJP (tangent-plane
// projection × 1/‖q‖).
// The x→y→z→w accumulation order of the sum of squares plus precise::rsqrt — the same convention as
// quat_to_rotmat in KernelCommon.h (Correction 6, D2). The v_R indexing stays glm [col][row].
static inline void quat_to_rotmat_vjp(const float4 quat, const float3x3 v_R,
                                      thread float4& v_quat)
{
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    const float inv_norm = metal::precise::rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;
    const float4 v_quat_n = float4(
        2.f * (x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
               z * (v_R[0][1] - v_R[1][0])),
        2.f * (-2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
               z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])),
        2.f * (x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
               z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])),
        2.f * (x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
               2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])));
    const float4 quat_n = float4(w, x, y, z);
    v_quat += (v_quat_n - metal::dot(v_quat_n, quat_n) * quat_n) * inv_norm;
}

// quat_scale_to_covar_vjp — Utils.cuh:313–347 (B7). covar = M·Mᵀ, M = R_g·S ⇒
// dL/dM = (G+Gᵀ)·M, dL/dR_g = v_M·S, dL/ds_j = dot(col_j(R_g), col_j(v_M)).
// The addition order follows the explicit order of the original (a+b+c) — do not collapse it into
// dot() (this keeps the signs and indices easy to audit).
static inline void quat_scale_to_covar_vjp(const float4 quat, const float3 scale,
                                           const float3x3 R, const float3x3 v_covar,
                                           thread float4& v_quat, thread float3& v_scale)
{
    // M = R·S, v_M·S: multiplying by the diagonal S is column j × s[j] (glm column-major)
    const float3x3 M = float3x3(R[0] * scale.x, R[1] * scale.y, R[2] * scale.z);
    const float3x3 v_M = (v_covar + metal::transpose(v_covar)) * M;
    const float3x3 v_R_g = float3x3(v_M[0] * scale.x, v_M[1] * scale.y, v_M[2] * scale.z);
    quat_to_rotmat_vjp(quat, v_R_g, v_quat);
    v_scale[0] += R[0][0] * v_M[0][0] + R[0][1] * v_M[0][1] + R[0][2] * v_M[0][2];
    v_scale[1] += R[1][0] * v_M[1][0] + R[1][1] * v_M[1][1] + R[1][2] * v_M[1][2];
    v_scale[2] += R[2][0] * v_M[2][0] + R[2][1] * v_M[2][1] + R[2][2] * v_M[2][2];
}

// The spec 01 bwd kernel (ProjectionEWA3DGSFused.cu:376–638) — rematerialization: the intermediates
// (covar/mean_c/covar_c/J) are not stored; the fwd chain (B3) is recomputed and then the VJP is
// propagated in reverse order.
// - the bwd inputs do not include opacities (Correction 1). The pointer shift convention is exactly
//   as in Correction 3.
// - D8-2: early returns are forbidden — they are converted into an alive flag plus a zero
//   contribution. If you port the CUDA B0 early return as is, the simd_sum on the v_viewmats path
//   breaks under divergence (this MUST be followed).
// - the gradient output buffers are zero-initialized by the dispatch wrapper with encode_zero_fill
//   (D8-1) — the kernel only accumulates.
// - kProjSingleCamera (C==1): global thread ↔ gid is 1:1 → no reduction and no atomics are needed,
//   and the store is deterministic (spec 01 reduction strategy 1; a relaxed atomic store == a plain
//   store, with a single writer per slot).
// - C>1: a per-thread atomicAdd — there is no equivalent of labeled_partition, and gid duplication
//   within a warp happens only when gaussian_count<32 (Correction 4), so a reduction gains nothing.
// - kProjViewmatsGrad: enabled and golden-verified in pose Task 1 (v_viewmats [C,16]). Reduction
//   strategy 2: a cid-uniform simdgroup uses simd_sum + a lane-0 atomic (1/32 of the traffic), and
//   boundary simdgroups use a per-lane atomic. Culled lanes also reach this point without an early
//   return (contributing v_R/v_t = 0) → the simd_sum is safe from divergence (D8-2).
kernel void projection_ewa_3dgs_fused_bwd(
    device const float* means           [[buffer(0)]],  // [N,3]
    device const float* covars          [[buffer(1), function_constant(kProjUseCovars)]],    // [N,6] triu
    device const float* quats           [[buffer(2), function_constant(kProjQuatScalePath)]], // [N,4] wxyz
    device const float* scales          [[buffer(3), function_constant(kProjQuatScalePath)]], // [N,3]
    device const float* viewmats        [[buffer(4)]],  // [C,16] row-major
    device const float* Ks              [[buffer(5)]],  // [C,9] row-major
    device const int*   radii           [[buffer(6)]],  // [C,N,2] fwd output (the culling gate)
    device const float* conics          [[buffer(7)]],  // [C,N,3] fwd output
    device const float* compensations   [[buffer(8), function_constant(kProjCalcComp)]],  // [C,N]
    device const float* v_means2d       [[buffer(9)]],  // [C,N,2]
    device const float* v_depths        [[buffer(10)]], // [C,N]
    device const float* v_conics        [[buffer(11)]], // [C,N,3]
    device const float* v_compensations [[buffer(12), function_constant(kProjCalcComp)]], // [C,N]
    device atomic_float* v_means        [[buffer(13)]], // [N,3]  — zero-initialized beforehand (D8-1)
    device atomic_float* v_covars       [[buffer(14), function_constant(kProjUseCovars)]],    // [N,6]
    device atomic_float* v_quats        [[buffer(15), function_constant(kProjQuatScalePath)]], // [N,4]
    device atomic_float* v_scales       [[buffer(16), function_constant(kProjQuatScalePath)]], // [N,3]
    device atomic_float* v_viewmats     [[buffer(17), function_constant(kProjViewmatsGrad)]],  // [C,16]
    constant ProjectionParams& P        [[buffer(18)]],
    uint idx [[thread_position_in_grid]])
{
    // B0 (lines 411–425): indexing + the culling gate — an alive flag instead of an early return (D8-2).
    const uint CN = P.C * P.N;
    const bool in_grid = idx < CN;                 // out of bounds is defensive, since dispatchThreads is used
    const uint gi = in_grid ? idx : 0u;            // clamps the index of a dead lane (to keep reads safe)
    const uint cid = gi / P.N;
    const uint gid = gi % P.N;
    const bool alive = in_grid && radii[gi * 2] > 0 && radii[gi * 2 + 1] > 0;

    float3   v_mean  = float3(0.f);
    float3x3 v_covar = float3x3(0.f);
    float4   v_quat  = float4(0.f);
    float3   v_scale = float3(0.f);
    float3x3 v_R     = float3x3(0.f);
    float3   v_t     = float3(0.f);

    if (alive) {
        // B1 (lines 438–442): the conic → covar2d VJP. conics and v_* are pointer-shifted (Correction 3).
        // The 0.5 on the off-diagonal: the conic b represents two slots of a symmetric matrix — the
        // gradient is split evenly between them.
        device const float* cn  = conics + gi * 3;
        device const float* vcn = v_conics + gi * 3;
        const float2x2 covar2d_inv = float2x2(float2(cn[0], cn[1]), float2(cn[1], cn[2]));
        const float2x2 v_covar2d_inv =
            float2x2(float2(vcn[0], vcn[1] * .5f), float2(vcn[1] * .5f, vcn[2]));
        float2x2 v_covar2d = float2x2(0.f);
        inverse_vjp(covar2d_inv, v_covar2d_inv, v_covar2d);

        // B2 (lines 444–450): the compensation VJP — compensations and v_compensations are not
        // shifted but indexed directly by [global_idx] (Correction 3).
        if (kProjCalcComp)
            add_blur_vjp(P.eps2d, covar2d_inv, compensations[gi], v_compensations[gi],
                         v_covar2d);

        // B3 (lines 452–494): forward recomputation (F1–F3 rematerialization)
        device const float* mw = means + gid * 3;
        device const float* vm = viewmats + cid * 16;
        device const float* K  = Ks + cid * 9;
        const float3x3 R = float3x3(
            float3(vm[0], vm[4], vm[8]),   // 1st col
            float3(vm[1], vm[5], vm[9]),   // 2nd col
            float3(vm[2], vm[6], vm[10])); // 3rd col
        const float3 t = float3(vm[3], vm[7], vm[11]);

        float3x3 covar;
        float4 quat = float4(0.f);
        float3 scale = float3(0.f);
        if (kProjUseCovars) {
            device const float* cv = covars + gid * 6;
            covar = float3x3(float3(cv[0], cv[1], cv[2]),
                             float3(cv[1], cv[3], cv[4]),
                             float3(cv[2], cv[4], cv[5]));
        } else {
            quat = float4(quats[gid * 4], quats[gid * 4 + 1], quats[gid * 4 + 2],
                          quats[gid * 4 + 3]);
            scale = float3(scales[gid * 3], scales[gid * 3 + 1], scales[gid * 3 + 2]);
            covar = quat_scale_to_covar(quat, scale);
        }
        const float3 mean_w = float3(mw[0], mw[1], mw[2]);
        const float3 mean_c = R * mean_w + t;                    // posW2C
        const float3x3 covar_c = R * covar * metal::transpose(R); // covarW2C

        // B4a (lines 496–551): PINHOLE persp_proj_vjp → (v_mean_c, v_covar_c)
        float3x3 v_covar_c = float3x3(0.f);
        float3 v_mean_c = float3(0.f);
        persp_proj_vjp(mean_c, covar_c, K[0], K[4], K[2], K[5], P.image_width,
                       P.image_height, v_covar2d,
                       float2(v_means2d[gi * 2], v_means2d[gi * 2 + 1]), v_mean_c,
                       v_covar_c);

        // B5 (lines 553–554): depths was mean_c.z, so this is a simple addition.
        v_mean_c.z += v_depths[gi];

        // B6 (lines 556–562): the camera → world VJP → (v_mean, v_covar, v_R, v_t)
        pos_w2c_vjp(R, mean_w, v_mean_c, v_R, v_t, v_mean);
        covar_w2c_vjp(R, covar, v_covar_c, v_R, v_covar);

        // B7 (lines 596–602): the quat/scale path — a per-thread conversion of v_covar →
        // (v_quat, v_scale) (CUDA also does this per thread, before the warpSum). The covars path
        // packs v_covar into 6-triu in B8.
        if (!kProjUseCovars) {
            const float3x3 rotmat = quat_to_rotmat(quat); // the same normalization as fwd (F2 / Correction 6)
            quat_scale_to_covar_vjp(quat, scale, rotmat, v_covar, v_quat, v_scale);
        }
    }

    // B8 (lines 564–637): output accumulation.
    if (kProjSingleCamera) {
        // The C==1 non-atomic path: one writer per slot — bit-exact determinism. Dead lanes
        // contribute nothing (the buffer stays zero-initialized).
        if (alive) {
            atomic_store_explicit(v_means + gid * 3 + 0, v_mean.x, memory_order_relaxed);
            atomic_store_explicit(v_means + gid * 3 + 1, v_mean.y, memory_order_relaxed);
            atomic_store_explicit(v_means + gid * 3 + 2, v_mean.z, memory_order_relaxed);
            if (kProjUseCovars) {
                // Packed into 6-triu by merging the symmetric entries — the two off-diagonal slots are summed (lines 588–593)
                atomic_store_explicit(v_covars + gid * 6 + 0, v_covar[0][0], memory_order_relaxed);
                atomic_store_explicit(v_covars + gid * 6 + 1, v_covar[0][1] + v_covar[1][0],
                                      memory_order_relaxed);
                atomic_store_explicit(v_covars + gid * 6 + 2, v_covar[0][2] + v_covar[2][0],
                                      memory_order_relaxed);
                atomic_store_explicit(v_covars + gid * 6 + 3, v_covar[1][1], memory_order_relaxed);
                atomic_store_explicit(v_covars + gid * 6 + 4, v_covar[1][2] + v_covar[2][1],
                                      memory_order_relaxed);
                atomic_store_explicit(v_covars + gid * 6 + 5, v_covar[2][2], memory_order_relaxed);
            } else {
                atomic_store_explicit(v_quats + gid * 4 + 0, v_quat[0], memory_order_relaxed);
                atomic_store_explicit(v_quats + gid * 4 + 1, v_quat[1], memory_order_relaxed);
                atomic_store_explicit(v_quats + gid * 4 + 2, v_quat[2], memory_order_relaxed);
                atomic_store_explicit(v_quats + gid * 4 + 3, v_quat[3], memory_order_relaxed);
                atomic_store_explicit(v_scales + gid * 3 + 0, v_scale[0], memory_order_relaxed);
                atomic_store_explicit(v_scales + gid * 3 + 1, v_scale[1], memory_order_relaxed);
                atomic_store_explicit(v_scales + gid * 3 + 2, v_scale[2], memory_order_relaxed);
            }
        }
    } else {
        // The C>1 atomic path: as many threads as there are cameras accumulate into the same gid
        // (equivalent to lines 568–617).
        if (alive) {
            atomic_fetch_add_explicit(v_means + gid * 3 + 0, v_mean.x, memory_order_relaxed);
            atomic_fetch_add_explicit(v_means + gid * 3 + 1, v_mean.y, memory_order_relaxed);
            atomic_fetch_add_explicit(v_means + gid * 3 + 2, v_mean.z, memory_order_relaxed);
            if (kProjUseCovars) {
                atomic_fetch_add_explicit(v_covars + gid * 6 + 0, v_covar[0][0],
                                          memory_order_relaxed);
                atomic_fetch_add_explicit(v_covars + gid * 6 + 1, v_covar[0][1] + v_covar[1][0],
                                          memory_order_relaxed);
                atomic_fetch_add_explicit(v_covars + gid * 6 + 2, v_covar[0][2] + v_covar[2][0],
                                          memory_order_relaxed);
                atomic_fetch_add_explicit(v_covars + gid * 6 + 3, v_covar[1][1],
                                          memory_order_relaxed);
                atomic_fetch_add_explicit(v_covars + gid * 6 + 4, v_covar[1][2] + v_covar[2][1],
                                          memory_order_relaxed);
                atomic_fetch_add_explicit(v_covars + gid * 6 + 5, v_covar[2][2],
                                          memory_order_relaxed);
            } else {
                atomic_fetch_add_explicit(v_quats + gid * 4 + 0, v_quat[0], memory_order_relaxed);
                atomic_fetch_add_explicit(v_quats + gid * 4 + 1, v_quat[1], memory_order_relaxed);
                atomic_fetch_add_explicit(v_quats + gid * 4 + 2, v_quat[2], memory_order_relaxed);
                atomic_fetch_add_explicit(v_quats + gid * 4 + 3, v_quat[3], memory_order_relaxed);
                atomic_fetch_add_explicit(v_scales + gid * 3 + 0, v_scale[0], memory_order_relaxed);
                atomic_fetch_add_explicit(v_scales + gid * 3 + 1, v_scale[1], memory_order_relaxed);
                atomic_fetch_add_explicit(v_scales + gid * 3 + 2, v_scale[2], memory_order_relaxed);
            }
        }
    }
    if (kProjViewmatsGrad) {
        // v_viewmats (lines 618–637) — the camera pose gradient path. Retained only as a golden
        // verification surface (ProjectionMetalTests, a gsplat reproduction asset). Its connection
        // to the training loop (in-training pose refinement) was removed in the 2026-07-17 cleanup —
        // this gradient is computed for verification only (no trainer consumes it).
        // The resident path uses it together with kProjSingleCamera (C==1) — this block accumulates
        // atomically even for a single camera (after the buffer 0-fill), so it is consistent.
        // Spec 01 reduction strategy 2: if every lane in the warp has the same cid, use simd_sum +
        // a lane-0 atomic (1/32 of the traffic); boundary simdgroups use a per-lane atomic. Every
        // lane reaches this point (no early return — D8-2), and dead lanes contribute v_R/v_t = 0,
        // so the simd_sum is safe.
        const bool cid_uniform = metal::simd_all(cid == metal::simd_broadcast_first(cid));
        if (cid_uniform) {
            for (uint i = 0; i < 3; i++) {     // row
                for (uint j = 0; j < 3; j++) { // col — glm[col][row] → row-major i*4+j (B8)
                    const float r = metal::simd_sum(v_R[j][i]);
                    if (metal::simd_is_first())
                        atomic_fetch_add_explicit(v_viewmats + cid * 16 + i * 4 + j, r,
                                                  memory_order_relaxed);
                }
                const float tv = metal::simd_sum(v_t[i]);
                if (metal::simd_is_first())
                    atomic_fetch_add_explicit(v_viewmats + cid * 16 + i * 4 + 3, tv,
                                              memory_order_relaxed);
            }
        } else if (alive) {
            for (uint i = 0; i < 3; i++) {
                for (uint j = 0; j < 3; j++)
                    atomic_fetch_add_explicit(v_viewmats + cid * 16 + i * 4 + j, v_R[j][i],
                                              memory_order_relaxed);
                atomic_fetch_add_explicit(v_viewmats + cid * 16 + i * 4 + 3, v_t[i],
                                          memory_order_relaxed);
            }
        }
    }
}
