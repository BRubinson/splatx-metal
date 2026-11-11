// Derived from nerfstudio-project/gsplat (Apache-2.0), commit 77ab983.
// Reimplemented for Metal/C++.
// Reference: _world_to_cam, _persp_proj, and _fully_fused_projection in gsplat/cuda/_torch_impl.py
//       gsplat/cuda/csrc/AdamCUDA.cu:34–71 (fused_adam_cpu)
//       gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu:38–219 + Utils.cuh (projection_fused_fwd_cpu)
//       gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu:42–140 + 434–468 (sh_fwd_cpu)
//       gsplat/cuda/csrc/IntersectTile.cu:213–464 AABB path + Intersect.cpp:230–296 (isect_tiles_cpu)
//       gsplat/cuda/csrc/IntersectTile.cu:925–988, 1004–1008 (isect_offset_encode_cpu)
//       gsplat/cuda/csrc/Intersect.cpp:299–324 the sort contract only (sort_isect_cpu — the
//       implementation is our own std::stable_sort, not derived from cub::DeviceRadixSort::SortPairs)
//       gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchFwd.cu:41–297 +
//       RasterizeToPixels3DGSDevice.cuh:37–55 (rasterize_fwd_cpu)
//       gsplat/cuda/csrc/RelocationCUDA.cu:35–79 (mcmc_relocation_cpu) +
//       gsplat/strategy/mcmc.py:95–100 (make_binoms_lut)
//       gsplat/cuda/csrc/MCMCPerturbCUDA.cu:27–61 + Utils.cuh:228–311
//       (mcmc_perturb_positions_cpu)
// rng_*_cpu is not derived from gsplat — written from scratch (from the literature), see the
// citations in RngPhilox.h.
#include "splatx/CpuKernels.h"
#include "splatx/KernelBackend.h" // AdamInputs, ProjectionFusedInputs/Outputs
#include "splatx/Math.h"
#include "splatx/Validate.h" // shared single source of truth for entry validation (throws as MetalBackend does)
#include "KernelConstants.h" // kAlphaThreshold/kGaussianExtend/kMinCompensation (single source of truth with the kernels)
#include "RngPhilox.h"       // single source of truth for the Philox4x32-10 core (same source as Rng.metal — bit-identical is the contract)
#include <algorithm>
#include <cfloat> // FLT_EPSILON (the upper clamp in mcmc_relocation — the same constant as the kernel)
#include <cmath>
#include <cstring>
#include <numeric>
#include <string>
#include <utility>

namespace splatx {

ProjectionOutputs projection_ewa_fwd_cpu(const ProjectionInputs& in) {
    const int64_t N = in.N, C = in.C;
    ProjectionOutputs out;
    out.radii.assign(C * N * 2, 0);
    out.means2d.assign(C * N * 2, 0.0f);
    out.depths.assign(C * N, 0.0f);
    out.conics.assign(C * N * 3, 0.0f);
    out.compensations.assign(C * N, 0.0f);

    for (int64_t c = 0; c < C; c++) {
        const float* V = in.viewmats + c * 16; // world→cam
        Mat3 R = {{V[0], V[1], V[2], V[4], V[5], V[6], V[8], V[9], V[10]}};
        Vec3 t = {V[3], V[7], V[11]};
        const float* K = in.Ks + c * 9;
        const float fx = K[0], fy = K[4], cx = K[2], cy = K[5];
        const float tan_fovx = 0.5f * in.width / fx, tan_fovy = 0.5f * in.height / fy;
        const float lim_x_pos = (in.width - cx) / fx + 0.3f * tan_fovx;
        const float lim_x_neg = cx / fx + 0.3f * tan_fovx;
        const float lim_y_pos = (in.height - cy) / fy + 0.3f * tan_fovy;
        const float lim_y_neg = cy / fy + 0.3f * tan_fovy;

        for (int64_t n = 0; n < N; n++) {
            const float* mw = in.means + n * 3;
            // world → cam
            float mx = R.m[0] * mw[0] + R.m[1] * mw[1] + R.m[2] * mw[2] + t.x;
            float my = R.m[3] * mw[0] + R.m[4] * mw[1] + R.m[5] * mw[2] + t.y;
            float mz = R.m[6] * mw[0] + R.m[7] * mw[1] + R.m[8] * mw[2] + t.z;

            Mat3 Cw = {{in.covars[n * 9 + 0], in.covars[n * 9 + 1], in.covars[n * 9 + 2],
                        in.covars[n * 9 + 3], in.covars[n * 9 + 4], in.covars[n * 9 + 5],
                        in.covars[n * 9 + 6], in.covars[n * 9 + 7], in.covars[n * 9 + 8]}};
            Mat3 Cc = mat3_mul(mat3_mul(R, Cw), mat3_transpose(R));

            // _persp_proj: builds the Jacobian from tx,ty clamped to the field-of-view limits
            const float tz = mz, tz2 = mz * mz;
            float txc = tz * std::clamp(mx / tz, -lim_x_neg, lim_x_pos);
            float tyc = tz * std::clamp(my / tz, -lim_y_neg, lim_y_pos);
            // J = [[fx/tz, 0, -fx*tx/tz^2], [0, fy/tz, -fy*ty/tz^2]]
            const float J00 = fx / tz, J02 = -fx * txc / tz2;
            const float J11 = fy / tz, J12 = -fy * tyc / tz2;

            // cov2d = J * Cc * J^T (2x2)
            // computes (J*Cc) first, for rows 0 and 1
            float JC0[3] = {J00 * Cc.m[0] + J02 * Cc.m[6], J00 * Cc.m[1] + J02 * Cc.m[7], J00 * Cc.m[2] + J02 * Cc.m[8]};
            float JC1[3] = {J11 * Cc.m[3] + J12 * Cc.m[6], J11 * Cc.m[4] + J12 * Cc.m[7], J11 * Cc.m[5] + J12 * Cc.m[8]};
            float c00 = JC0[0] * J00 + JC0[2] * J02;
            float c01 = JC0[1] * J11 + JC0[2] * J12;
            float c10 = JC1[0] * J00 + JC1[2] * J02;
            float c11 = JC1[1] * J11 + JC1[2] * J12;

            const float det_orig = c00 * c11 - c01 * c10;
            c00 += in.eps2d; c11 += in.eps2d;
            float det = c00 * c11 - c01 * c10;
            det = std::max(det, 1e-10f);

            if (in.calc_compensations) {
                float ratio = det_orig / det;
                out.compensations[c * N + n] =
                    std::sqrt(std::max(ratio, kMinCompensation * kMinCompensation));
            }

            const size_t o2 = (c * N + n) * 2;
            out.means2d[o2 + 0] = (fx * mx + cx * mz) / mz; // K[:2,:3]·mean / z
            out.means2d[o2 + 1] = (fy * my + cy * mz) / mz;
            out.depths[c * N + n] = mz;

            const size_t o3 = (c * N + n) * 3;
            out.conics[o3 + 0] = c11 / det;
            out.conics[o3 + 1] = -(c01 + c10) / 2.0f / det;
            out.conics[o3 + 2] = c00 / det;

            float rx = std::ceil(3.33f * std::sqrt(c00));
            float ry = std::ceil(3.33f * std::sqrt(c11));
            bool valid = (mz > in.near_plane) && (mz < in.far_plane);
            if (valid) {
                bool inside = (out.means2d[o2] + rx > 0) && (out.means2d[o2] - rx < in.width) &&
                              (out.means2d[o2 + 1] + ry > 0) && (out.means2d[o2 + 1] - ry < in.height);
                if (inside) {
                    out.radii[o2 + 0] = static_cast<int32_t>(rx);
                    out.radii[o2 + 1] = static_cast<int32_t>(ry);
                }
            }
        }
    }
    return out;
}

// Spec 06 fused_adam (a reimplementation of AdamCUDA.cu:34–71) — the same formula as the Metal kernel
// (AdamMCMC.metal). All arithmetic is f32 (the upstream original also keeps float in its registers);
// there is no bias correction (D8-7 — improving on the original is prohibited); eps is added outside
// sqrt(v). Gaussians with valid[g]==0 skip param and both moments entirely (the moments are not even
// decayed).
void fused_adam_cpu(const AdamInputs& in) {
    if (in.n_elems == 0) return; // the same as the dispatch-skip convention (D8-6)
    validate_adam_inputs(in, "fused_adam_cpu");

    const int64_t D = in.n_elems / in.n_gauss;
    for (int64_t p = 0; p < in.n_elems; p++) {
        if (in.valid && in.valid[p / D] == 0) continue; // AdamCUDA.cu:56 — if not visible, skip everything
        const float g = in.grad[p];
        float m = in.exp_avg[p];
        float v = in.exp_avg_sq[p];
        m = in.b1 * m + (1.0f - in.b1) * g;
        v = in.b2 * v + (1.0f - in.b2) * g * g;
        const float step = -in.lr * m / (std::sqrt(v) + in.eps); // eps is outside the sqrt (06)
        in.param[p] += step;
        in.exp_avg[p] = m;
        in.exp_avg_sq[p] = v;
    }
}

// Spec 01 projection_ewa_3dgs_fused_fwd (a reimplementation of ProjectionEWA3DGSFused.cu:38–219, all
// arithmetic f32) — the CUDA kernel semantics exactly. The parity reference for the Metal kernel
// (ProjectionEWA.metal) (D1).
// Differences from projection_ewa_fwd_cpu (above): covars is [N,6] triu (D3) or quat/scale (F2);
// culling happens before the outputs are written (near/far → det<=0 → opacity → radius_clip → image
// bounds); mean2d = fx·x·rz + cx (persp_proj, Utils.cuh:606); and conic multiplies by the reciprocal
// 1/det (the glm::inverse approach).
// PINHOLE only (disposition #3 — ORTHO/FISHEYE are excluded from Phase 1 verification, so no code
// path exists for them either).
ProjectionFusedOutputs projection_fused_fwd_cpu(const ProjectionFusedInputs& in) {
    validate_projection_fused_inputs(in, "projection_fused_fwd");

    const int64_t N = in.N, C = in.C;
    const size_t CN = static_cast<size_t>(C * N);
    ProjectionFusedOutputs out;
    // The CUDA host uses at::empty (uninitialized; only compensations gets at::zeros —
    // Projection.cpp:395–411). In the CPU reference, std::vector zero-initialization leaves the culled
    // slots at 0 (by contract these are undefined values — comparison happens only under the valid
    // mask, D3). For radii, writing 0 on culling is the contract, so it agrees with zero-initialization.
    out.radii.assign(CN * 2, 0);
    out.means2d.assign(CN * 2, 0.0f);
    out.depths.assign(CN, 0.0f);
    out.conics.assign(CN * 3, 0.0f);
    if (in.calc_compensations) out.compensations.assign(CN, 0.0f);

    for (int64_t c = 0; c < C; c++) {
        // F1: loads row-major [R|t] from viewmats (ProjectionEWA3DGSFused.cu:83–95)
        const float* V = in.viewmats + c * 16;
        const Mat3 R = {{V[0], V[1], V[2], V[4], V[5], V[6], V[8], V[9], V[10]}};
        const Vec3 t = {V[3], V[7], V[11]};
        const float* K = in.Ks + c * 9;
        const float fx = K[0], fy = K[4], cx = K[2], cy = K[5];
        // F4a persp_proj preparation (Utils.cuh:588–595)
        const float tan_fovx = 0.5f * static_cast<float>(in.width) / fx;
        const float tan_fovy = 0.5f * static_cast<float>(in.height) / fy;
        const float lim_x_pos = (static_cast<float>(in.width) - cx) / fx + 0.3f * tan_fovx;
        const float lim_x_neg = cx / fx + 0.3f * tan_fovx;
        const float lim_y_pos = (static_cast<float>(in.height) - cy) / fy + 0.3f * tan_fovy;
        const float lim_y_neg = cy / fy + 0.3f * tan_fovy;

        for (int64_t n = 0; n < N; n++) {
            const size_t gi = static_cast<size_t>(c * N + n); // global_idx (B=1)
            // F1: mean_c = R·μ + t (posW2C, Utils.cuh:81–89)
            const float* mw = in.means + n * 3;
            const float mx = R.m[0] * mw[0] + R.m[1] * mw[1] + R.m[2] * mw[2] + t.x;
            const float my = R.m[3] * mw[0] + R.m[4] * mw[1] + R.m[5] * mw[2] + t.y;
            const float mz = R.m[6] * mw[0] + R.m[7] * mw[1] + R.m[8] * mw[2] + t.z;
            // near/far culling (lines 100–105) — equality survives
            if (mz < in.near_plane || mz > in.far_plane) continue;

            // F2: source of the 3D covariance (lines 107–130)
            Mat3 Cw;
            if (in.covars) {
                const float* cv = in.covars + n * 6; // [N,6] triu (c00,c01,c02,c11,c12,c22)
                Cw = {{cv[0], cv[1], cv[2], cv[1], cv[3], cv[4], cv[2], cv[4], cv[5]}};
            } else {
                Cw = quat_scale_to_covar(in.quats + n * 4, in.scales + n * 3); // reuses Math.h
            }
            // F3: covar_c = R·Σ·Rᵀ (covarW2C, Utils.cuh:116–123)
            const Mat3 Cc = mat3_mul(mat3_mul(R, Cw), mat3_transpose(R));

            // F4a: persp_proj (Utils.cuh:567–607)
            const float rz = 1.0f / mz, rz2 = rz * rz;
            const float txc = mz * std::min(lim_x_pos, std::max(-lim_x_neg, mx * rz));
            const float tyc = mz * std::min(lim_y_pos, std::max(-lim_y_neg, my * rz));
            const float J00 = fx * rz, J02 = -fx * txc * rz2;
            const float J11 = fy * rz, J12 = -fy * tyc * rz2;
            // covar2d = J·Σc·Jᵀ — only rows 0 and 1 of J are valid (J01=J10=0)
            const float JC0[3] = {J00 * Cc.m[0] + J02 * Cc.m[6], J00 * Cc.m[1] + J02 * Cc.m[7],
                                  J00 * Cc.m[2] + J02 * Cc.m[8]};
            const float JC1[3] = {J11 * Cc.m[3] + J12 * Cc.m[6], J11 * Cc.m[4] + J12 * Cc.m[7],
                                  J11 * Cc.m[5] + J12 * Cc.m[8]};
            float c00 = JC0[0] * J00 + JC0[2] * J02;
            const float c01 = JC0[1] * J11 + JC0[2] * J12; // row0,col1
            const float c10 = JC1[0] * J00 + JC1[2] * J02; // row1,col0
            float c11 = JC1[1] * J11 + JC1[2] * J12;
            const float m2x = fx * mx * rz + cx; // uses the unclamped x·rz (F4a caution)
            const float m2y = fy * my * rz + cy;

            // F5: add_blur (Utils.cuh:455–463) — compensation is computed before the det<=0 culling
            const float det_orig = c00 * c11 - c01 * c10;
            c00 += in.eps2d;
            c11 += in.eps2d;
            const float det = c00 * c11 - c01 * c10;
            const float compensation =
                std::sqrt(std::max(kMinCompensation * kMinCompensation, det_orig / det));
            if (det <= 0.0f) continue; // culling (lines 153–158)

            // F6: conic = inverse(covar2d) — the glm approach (multiply by the reciprocal 1/det, [0][1] = -c10·rdet)
            const float rdet = 1.0f / det;
            const float conic_a = c11 * rdet;
            const float conic_b = -c10 * rdet;
            const float conic_c = c00 * rdet;

            // F7: opacity-aware radius + culling (lines 163–204)
            float extend = kGaussianExtend;
            if (in.opacities) {
                float opacity = in.opacities[n];
                if (in.calc_compensations) opacity *= compensation; // only in AA mode
                if (opacity < kAlphaThreshold) continue;
                // D2: CUDA __logf ↔ CPU logf unified (Metal uses precise::log)
                extend = std::min(kGaussianExtend,
                                  std::sqrt(2.0f * std::log(opacity / kAlphaThreshold)));
            }
            const float radius_x = std::ceil(extend * std::sqrt(c00));
            const float radius_y = std::ceil(extend * std::sqrt(c11));
            // radius_clip is an AND — cull only when both are small (line 188)
            if (radius_x <= in.radius_clip && radius_y <= in.radius_clip) continue;
            // culling outside the image area (lines 196–199) — equality at the boundary culls too
            if (m2x + radius_x <= 0.0f || m2x - radius_x >= static_cast<float>(in.width) ||
                m2y + radius_y <= 0.0f || m2y - radius_y >= static_cast<float>(in.height))
                continue;

            // F8: outputs (lines 206–218)
            out.radii[gi * 2] = static_cast<int32_t>(radius_x);
            out.radii[gi * 2 + 1] = static_cast<int32_t>(radius_y);
            out.means2d[gi * 2] = m2x;
            out.means2d[gi * 2 + 1] = m2y;
            out.depths[gi] = mz;
            out.conics[gi * 3] = conic_a;
            out.conics[gi * 3 + 1] = conic_b;
            out.conics[gi * 3 + 2] = conic_c;
            if (in.calc_compensations) out.compensations[gi] = compensation;
        }
    }
    return out;
}

namespace {

// SH basis evaluation — a 1:1 port of sh_coeffs_to_color_fast (SphericalHarmonicsCUDA.cu:42–140).
// Hardcoded expansion in the style of Sloan, "Efficient Spherical Harmonic Evaluation", JCGT 2013
// (deg 0–4). All arithmetic f32. Normalization is 1/sqrt (matching CUDA rsqrtf / Metal
// precise::rsqrt — D2), unguarded at dir=0 (rsqrt(0)=inf → NaN propagates, same as the upstream
// original — disposition #6).
// At degree 0 it never reads dir at all (the normalization sits inside the degree>=1 branch — the
// original at :54).
// The coeffs/colors pointers arrive already shifted by gaussian_id*K*D / elem_id*D at the call site.
void sh_coeffs_to_color_fast_cpu(uint32_t degree, uint32_t D, uint32_t c, const float* dir,
                                 const float* coeffs, float* colors) {
    // degree 0 (:54)
    float result = 0.2820947917738781f * coeffs[c];
    if (degree >= 1) {
        const float inorm =
            1.0f / std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
        const float x = dir[0] * inorm;
        const float y = dir[1] * inorm;
        const float z = dir[2] * inorm;

        // degree 1 (:64-67) — signs/order (-y, +z, -x); coeffs indices 1→y, 2→z, 3→x
        result += 0.48860251190292f *
                  (-y * coeffs[1 * D + c] + z * coeffs[2 * D + c] - x * coeffs[3 * D + c]);
        if (degree >= 2) { // (:68-85)
            const float z2 = z * z;
            const float fTmp0B = -1.092548430592079f * z;
            const float fC1 = x * x - y * y;
            const float fS1 = 2.f * x * y;
            const float pSH6 = 0.9461746957575601f * z2 - 0.3153915652525201f;
            const float pSH7 = fTmp0B * x;
            const float pSH5 = fTmp0B * y;
            const float pSH8 = 0.5462742152960395f * fC1;
            const float pSH4 = 0.5462742152960395f * fS1;
            result += pSH4 * coeffs[4 * D + c] + pSH5 * coeffs[5 * D + c] +
                      pSH6 * coeffs[6 * D + c] + pSH7 * coeffs[7 * D + c] +
                      pSH8 * coeffs[8 * D + c];
            if (degree >= 3) { // (:86-106)
                const float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
                const float fTmp1B = 1.445305721320277f * z;
                const float fC2 = x * fC1 - y * fS1; // Chebyshev recurrence cos(2φ)→cos(3φ)
                const float fS2 = x * fS1 + y * fC1;
                const float pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
                const float pSH13 = fTmp0C * x;
                const float pSH11 = fTmp0C * y;
                const float pSH14 = fTmp1B * fC1;
                const float pSH10 = fTmp1B * fS1;
                const float pSH15 = -0.5900435899266435f * fC2;
                const float pSH9 = -0.5900435899266435f * fS2;
                result += pSH9 * coeffs[9 * D + c] + pSH10 * coeffs[10 * D + c] +
                          pSH11 * coeffs[11 * D + c] + pSH12 * coeffs[12 * D + c] +
                          pSH13 * coeffs[13 * D + c] + pSH14 * coeffs[14 * D + c] +
                          pSH15 * coeffs[15 * D + c];
                if (degree >= 4) { // (:108-134) — reachable only when K=25
                    const float fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
                    const float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
                    const float fTmp2B = -1.770130769779931f * z;
                    const float fC3 = x * fC2 - y * fS2;
                    const float fS3 = x * fS2 + y * fC2;
                    const float pSH20 =
                        1.984313483298443f * z * pSH12 - 1.006230589874905f * pSH6;
                    const float pSH21 = fTmp0D * x;
                    const float pSH19 = fTmp0D * y;
                    const float pSH22 = fTmp1C * fC1;
                    const float pSH18 = fTmp1C * fS1;
                    const float pSH23 = fTmp2B * fC2;
                    const float pSH17 = fTmp2B * fS2;
                    const float pSH24 = 0.6258357354491763f * fC3;
                    const float pSH16 = 0.6258357354491763f * fS3;
                    result += pSH16 * coeffs[16 * D + c] + pSH17 * coeffs[17 * D + c] +
                              pSH18 * coeffs[18 * D + c] + pSH19 * coeffs[19 * D + c] +
                              pSH20 * coeffs[20 * D + c] + pSH21 * coeffs[21 * D + c] +
                              pSH22 * coeffs[22 * D + c] + pSH23 * coeffs[23 * D + c] +
                              pSH24 * coeffs[24 * D + c];
                }
            }
        }
    }
    colors[c] = result; // (:139)
}

} // namespace

// Spec 02 sh_fwd CPU reference (the semantics of the generic fwd kernel :434–468, offset=0/count=N).
// Masked-out elements are unwritten (disposition #5) — as a by-product of the zero-initialized vector
// those slots stay 0.
ShOutputs sh_fwd_cpu(const ShInputs& in) {
    validate_sh_inputs(in, "sh_fwd_cpu");

    const size_t BN = static_cast<size_t>(in.B) * static_cast<size_t>(in.N);
    const size_t N = static_cast<size_t>(in.N);
    const size_t KD = static_cast<size_t>(in.K) * in.D;
    ShOutputs out;
    out.colors.assign(BN * in.D, 0.0f);
    for (size_t elem = 0; elem < BN; elem++) {
        if (in.masks && in.masks[elem] == 0) continue; // masked out → unwritten (:461-464)
        const size_t gaussian_id = elem % N;
        for (uint32_t c = 0; c < in.D; c++)
            sh_coeffs_to_color_fast_cpu(in.degrees_to_use, in.D, c, in.dirs + elem * 3,
                                        in.coeffs + gaussian_id * KD,
                                        out.colors.data() + elem * in.D);
    }
    return out;
}

namespace {

// AABB tile rectangle (IntersectTile.cu:377–387) — both the count and emit passes MUST share this
// single formula (the invariant that emit never writes beyond count — spec 03 §Algorithm overview).
// The division is IEEE f32 — the Metal side matches it with precise::divide (a D2 parity point: it
// presupposes exact agreement of the integer tile boundary floor/ceil). tmin inclusive, tmax
// exclusive (:382).
struct TileRect {
    int32_t min_x, min_y, max_x, max_y;
};

inline TileRect aabb_tile_rect(float mean_x, float mean_y, float radius_x, float radius_y,
                               float tile_size_f, int32_t tile_width, int32_t tile_height) {
    const float tile_radius_x = radius_x / tile_size_f; // :377
    const float tile_radius_y = radius_y / tile_size_f; // :378
    const float tile_x = mean_x / tile_size_f;          // :379
    const float tile_y = mean_y / tile_size_f;          // :380
    TileRect r;
    r.min_x = std::min(std::max(0, static_cast<int32_t>(std::floor(tile_x - tile_radius_x))),
                       tile_width);  // :384
    r.min_y = std::min(std::max(0, static_cast<int32_t>(std::floor(tile_y - tile_radius_y))),
                       tile_height); // :385
    r.max_x = std::min(std::max(0, static_cast<int32_t>(std::ceil(tile_x + tile_radius_x))),
                       tile_width);  // :386
    r.max_y = std::min(std::max(0, static_cast<int32_t>(std::ceil(tile_y + tile_radius_y))),
                       tile_height); // :387
    return r;
}

// ===== AccuTile — SNUGBOX + ellipse-line sweep (SpeedySplat arXiv:2412.00578, a direct port of the
// IntersectTile.cu:83–207 helpers + the :288–372 body) =====
// Keep this identical to the Metal kernel (IntersectTile.metal) down to the form of the expressions —
// division/sqrt/log determine the integer tile boundaries, so the Metal side matches with
// precise::divide/sqrt/log (a D2 parity point, disposition #11). float min/max use fmin/fmax to match
// the CUDA device min(float,float)=fminf semantics.

struct F2 { // matches CUDA float2 (expresses the sweep-frame swap exactly as the original does)
    float x, y;
};

// Ellipse-vertical-line intersection (accutile_ellipse_intersection, IntersectTile.cu:83–94) — in the
// sweep frame, the v coordinates {v_lo, v_hi} of the two intersections between the line u = coord and
// the level set q = t. Non-negativity of the discriminant is guaranteed by the caller's guards
// (:145, :158 — |h| <= extent_u) (derived in spec 03 §3).
inline F2 accutile_ellipse_intersection(float A, float B, float C, float disc, float t, F2 p,
                                        bool isY, float coord) {
    const float p_u = isY ? p.y : p.x;   // :86
    const float p_v = isY ? p.x : p.y;   // :87
    const float coeff = isY ? A : C;     // :88
    const float h = coord - p_u;         // :90
    const float sqrt_term = std::sqrt(disc * h * h + t * coeff); // :91
    return {(-B * h - sqrt_term) / coeff + p_v,   // :93 v_lo
            (-B * h + sqrt_term) / coeff + p_v};  //     v_hi
}

// Strip sweep (accutile_process_tiles, IntersectTile.cu:96–207) — both the count and emit passes
// share this single formula (the invariant that emit never writes beyond count). Whether it emits is
// decided by isect_ids != nullptr (:195 — the same gate as the original's first_pass ? nullptr :
// isect_ids). Single-device direct port: the key window skip (:189–192) always passes, so it is
// removed (disposition #14).
inline uint32_t accutile_process_tiles(float A, float B, float C, float disc, float t, F2 p,
                                       F2 bbox_min, F2 bbox_max, F2 bbox_argmin, F2 bbox_argmax,
                                       int32_t rect_min_x, int32_t rect_min_y,
                                       int32_t rect_max_x, int32_t rect_max_y, float BLOCK,
                                       int32_t tile_width, bool isY, uint64_t iid_enc,
                                       uint64_t depth_id_enc, int32_t flatten_idx,
                                       uint64_t* isect_ids, int32_t* flatten_ids,
                                       int64_t* cur_idx) {
    if (isY) { // :127–135 sweep-frame swap — afterwards u = sweep axis (the shorter one), v = orthogonal axis
        std::swap(rect_min_x, rect_min_y);
        std::swap(rect_max_x, rect_max_y);
        bbox_min = {bbox_min.y, bbox_min.x};
        bbox_max = {bbox_max.y, bbox_max.x};
        bbox_argmin = {bbox_argmin.y, bbox_argmin.x};
        bbox_argmax = {bbox_argmax.y, bbox_argmax.x};
    }

    uint32_t tiles_count = 0;                       // :137
    F2 intersect_min_line, intersect_max_line;
    float ellipse_min, ellipse_max;
    float min_line, max_line;

    intersect_max_line = {bbox_max.y, bbox_min.y};  // :142 neutral values (v_max as min candidate, v_min as max candidate)

    min_line = rect_min_x * BLOCK;                  // :144
    if (bbox_min.x <= min_line)                     // :145 the line is inside the ellipse's left bound — the intersection is valid
        intersect_min_line = accutile_ellipse_intersection(A, B, C, disc, t, p, isY, min_line);
    else
        intersect_min_line = intersect_max_line;    // :151 the line is left of the ellipse → neutral

    for (int32_t u = rect_min_x; u < rect_max_x; ++u) { // :155 (#pragma unroll 1)
        max_line = min_line + BLOCK;                    // :157
        if (max_line <= bbox_max.x)                     // :158 update only when inside the ellipse's right bound
            intersect_max_line = accutile_ellipse_intersection(A, B, C, disc, t, p, isY, max_line);

        if (min_line <= bbox_argmin.y && bbox_argmin.y < max_line) // :163 the tangent point lies inside the strip
            ellipse_min = bbox_min.y;                              // :165 → the bbox extremum
        else
            ellipse_min = std::fmin(intersect_min_line.x, intersect_max_line.x); // :169

        if (min_line <= bbox_argmax.y && bbox_argmax.y < max_line) // :172
            ellipse_max = bbox_max.y;                              // :174
        else
            ellipse_max = std::fmax(intersect_min_line.y, intersect_max_line.y); // :178

        const int32_t min_tile_v = std::max(
            rect_min_y, std::min(rect_max_y, static_cast<int32_t>(ellipse_min / BLOCK)));  // :181
        const int32_t max_tile_v = std::min(
            rect_max_y,
            std::max(rect_min_y, static_cast<int32_t>(ellipse_max / BLOCK + 1.f)));        // :182

        for (int32_t v = min_tile_v; v < max_tile_v; ++v) { // :185 (#pragma unroll 1)
            ++tiles_count;                                  // :193
            if (isect_ids != nullptr) {                     // :195 only the emit pass writes
                const uint64_t tile_id =
                    static_cast<uint64_t>(isY ? u * tile_width + v : v * tile_width + u); // :187
                isect_ids[*cur_idx] = iid_enc | (tile_id << 32) | depth_id_enc;           // :197
                flatten_ids[*cur_idx] = flatten_idx;                                      // :198
                ++(*cur_idx);                                                             // :199
            }
        }

        intersect_min_line = intersect_max_line; // :203
        min_line = max_line;                     // :204
    }
    return tiles_count;                          // :206
}

// AccuTile element processing (IntersectTile.cu:288–367) — builds the level set, the SNUGBOX, and the
// rect from conic/opacity, then calls the sweep. The count pass calls it with isect_ids=nullptr
// (:364), and the return value is the tile count.
inline uint32_t accutile_element(const float* conic /*[3]*/, float opacity, float mean_x,
                                 float mean_y, float tile_size_f, int32_t tile_width,
                                 int32_t tile_height, uint64_t iid_enc, uint64_t depth_id_enc,
                                 int32_t flatten_idx, uint64_t* isect_ids, int32_t* flatten_ids,
                                 int64_t* cur_idx) {
    const float A = conic[0], B = conic[1], C = conic[2]; // :293–295 Σ⁻¹ upper-triangular (triu)
    const float disc = B * B - A * C;                     // :298 = -det(Σ⁻¹) < 0 (if positive definite)

    // opacity-aware level set (:300–303): alpha = opacity·exp(-0.5q) >= α_thresh ⇔
    // q <= 2·ln(opacity/α_thresh), capped at GAUSSIAN_EXTEND² (the same radius budget as projection).
    // CUDA __logf ↔ CPU logf ↔ Metal precise::log unified (disposition #11).
    const float t = std::fmin(kGaussianExtend * kGaussianExtend,
                              2.0f * std::log(opacity / kAlphaThreshold));
    // Disposition #13 (an approved deviation): if opacity < α_thresh then t < 0 → a negative sqrt
    // argument → NaN. The upstream original is unguarded because it presupposes upstream (projection)
    // culling (spec 03 §2 caution) — here there is an early return at t <= 0 (0 tiles). For valid
    // input (opacity > α_thresh) the result is unchanged.
    if (t <= 0.f) return 0;

    // SNUGBOX — the tight AABB of the ellipse (:305–316). x_extent = sqrt(t·C/(AC-B²)) (Lagrange derivation).
    const float neg_t_over_disc = -t / disc;                 // :306 > 0
    const float x_extent = std::sqrt(neg_t_over_disc * C);   // :307
    const float y_extent = std::sqrt(neg_t_over_disc * A);   // :308
    const F2 bbox_min = {mean_x - x_extent, mean_y - y_extent}; // :310
    const F2 bbox_max = {mean_x + x_extent, mean_y + y_extent}; // :311
    const float Bx_over_C = B * x_extent / C;                // :313
    const float By_over_A = B * y_extent / A;                // :314
    // tangent point coordinates (:315–316) — .x = the y coordinate of the leftmost/rightmost point,
    // .y = the x coordinate of the topmost/bottommost point
    const F2 bbox_argmin = {mean_y + Bx_over_C, mean_x + By_over_A};
    const F2 bbox_argmax = {mean_y - Bx_over_C, mean_x - By_over_A};

    // Tile rectangle clamp (:318–324) — the (int) cast truncates toward zero (different from the
    // floor/ceil of the AABB path — Task 3 review note). The max(0,·) clamp makes the negative case
    // harmless.
    const int32_t rect_min_x = std::max(
        0, std::min(tile_width, static_cast<int32_t>(bbox_min.x / tile_size_f)));         // :319
    const int32_t rect_min_y = std::max(
        0, std::min(tile_height, static_cast<int32_t>(bbox_min.y / tile_size_f)));        // :321
    const int32_t rect_max_x = std::max(
        0, std::min(tile_width, static_cast<int32_t>(bbox_max.x / tile_size_f + 1.f)));   // :323
    const int32_t rect_max_y = std::max(
        0, std::min(tile_height, static_cast<int32_t>(bbox_max.y / tile_size_f + 1.f)));  // :324

    const int32_t y_span = rect_max_y - rect_min_y; // :326
    const int32_t x_span = rect_max_x - rect_min_x; // :327
    if (y_span * x_span == 0) return 0;             // :328–335 empty rect → 0 tiles

    const bool isY = y_span < x_span;               // :337 the shorter axis becomes the outer sweep loop
    return accutile_process_tiles(A, B, C, disc, t, {mean_x, mean_y}, bbox_min, bbox_max,
                                  bbox_argmin, bbox_argmax, rect_min_x, rect_min_y, rect_max_x,
                                  rect_max_y, tile_size_f, tile_width, isY, iid_enc,
                                  depth_id_enc, flatten_idx, isect_ids, flatten_ids,
                                  cur_idx);        // :340–367 (BLOCK = (float)tile_size, :125)
}

} // namespace

// Spec 03 intersect_tile — the AccuTile path (conics+opacities, IntersectTile.cu:83–207 + 288–372) /
// the AABB fallback path (:374–463) + the 2-pass host orchestration (Intersect.cpp:230–296). The path
// gate is exactly the original's conics != nullptr && opacities != nullptr (:288 — the validator
// guarantees both-or-neither). Single-device direct port: the key_start/key_end window and
// row_key_column_range are identities, so they are removed, and packed/tile_mask are excluded
// (disposition #14 — dense only).
// The AccuTile t<=0 early return is the approved deviation of disposition #13 (see the
// accutile_element comment).
IsectTilesOutputs isect_tiles_cpu(const IsectTilesInputs& in) {
    validate_isect_tiles_inputs(in, "isect_tiles_cpu");
    const int64_t n_elements = in.I * in.N; // dense: I*N (IntersectTile.cu:268 iid = idx/N)
    IsectTilesOutputs out;
    out.tile_n_bits = bits_for_count(static_cast<int64_t>(in.tile_width) * in.tile_height);
    out.image_n_bits = bits_for_count(in.I); // IntersectTile.cu:507–508
    out.tiles_per_gauss.assign(static_cast<size_t>(n_elements), 0);
    if (n_elements == 0) return out; // host skip — n_isects=0 (Intersect.cpp:236–270)

    const bool accu = in.conics != nullptr && in.opacities != nullptr; // :288
    const float tile_size_f = static_cast<float>(in.tile_size);
    const int32_t TW = static_cast<int32_t>(in.tile_width);
    const int32_t TH = static_cast<int32_t>(in.tile_height);

    // Pass 1 (count) — the culling of IntersectTile.cu:255–264 is common; AccuTile uses the sweep's
    // return value (:340–372), AABB uses the closed-form rectangle area (:400–432, single device and
    // no mask — :416).
    for (int64_t idx = 0; idx < n_elements; idx++) {
        const float radius_x = static_cast<float>(in.radii[idx * 2]);     // :255 float local variable
        const float radius_y = static_cast<float>(in.radii[idx * 2 + 1]); // :256
        if (radius_x <= 0.f || radius_y <= 0.f) continue;                 // :257–264 (count 0)
        if (accu) {
            out.tiles_per_gauss[static_cast<size_t>(idx)] = static_cast<int32_t>(
                accutile_element(in.conics + idx * 3, in.opacities[idx], in.means2d[idx * 2],
                                 in.means2d[idx * 2 + 1], tile_size_f, TW, TH,
                                 /*iid_enc=*/0, /*depth_id_enc=*/0,
                                 static_cast<int32_t>(idx), nullptr, nullptr,
                                 nullptr)); // :364 first_pass → nullptr (no writes)
            continue;
        }
        const TileRect r = aabb_tile_rect(in.means2d[idx * 2], in.means2d[idx * 2 + 1],
                                          radius_x, radius_y, tile_size_f, TW, TH);
        out.tiles_per_gauss[static_cast<size_t>(idx)] =
            (r.max_y - r.min_y) * (r.max_x - r.min_x); // :430
    }

    // Prefix sum (int64) + the n_isects readback — matching at::cumsum (Intersect.cpp:258–259, D9-3).
    const std::vector<int64_t> cum =
        cumsum_i64(out.tiles_per_gauss.data(), static_cast<size_t>(n_elements));
    out.n_isects = cum.back();
    if (out.n_isects == 0) return out; // emit skipped (Intersect.cpp:275)

    out.isect_ids.resize(static_cast<size_t>(out.n_isects));
    out.flatten_ids.resize(static_cast<size_t>(out.n_isects));

    // Pass 2 (emit) — key encoding preparation (IntersectTile.cu:266–286) + the AccuTile sweep emit
    // (:338–367) / the AABB emit (:434–462). Emit order (deterministic): ascending element idx →
    // (AccuTile) sweep axis u → orthogonal axis v / (AABB) ascending tile y → ascending x.
    for (int64_t idx = 0; idx < n_elements; idx++) {
        const float radius_x = static_cast<float>(in.radii[idx * 2]);     // :255
        const float radius_y = static_cast<float>(in.radii[idx * 2 + 1]); // :256
        if (radius_x <= 0.f || radius_y <= 0.f) continue;                 // :257–264

        const int64_t iid = idx / in.N;                                        // :268 dense
        const uint64_t iid_enc = static_cast<uint64_t>(iid) << (32 + out.tile_n_bits); // :273
        const float depth_f = in.depths[idx];                                  // :279
        // CUDA assert(depth_f >= 0.f) (:283) — a negative or NaN value breaks the depth monotonicity
        // of the unsigned key. The CPU reference promotes this to a validation throw; the Metal kernel
        // has no assert (spec 03 §Kernel 3).
        if (!(depth_f >= 0.f))
            throw std::invalid_argument("isect_tiles_cpu: depth must be >= 0 at idx " +
                                        std::to_string(idx));
        uint32_t depth_bits;
        std::memcpy(&depth_bits, &depth_f, sizeof(depth_bits)); // :285 matching __float_as_uint
        const uint64_t depth_id_enc = depth_bits;               // zero-extended into the low 32 bits

        int64_t cur_idx = (idx == 0) ? 0 : cum[static_cast<size_t>(idx) - 1]; // :338/:434
        if (accu) { // the same accutile_element as count — emit writes only within its own range
            accutile_element(in.conics + idx * 3, in.opacities[idx], in.means2d[idx * 2],
                             in.means2d[idx * 2 + 1], tile_size_f, TW, TH, iid_enc,
                             depth_id_enc, static_cast<int32_t>(idx), out.isect_ids.data(),
                             out.flatten_ids.data(), &cur_idx);
            continue;
        }
        const TileRect r = aabb_tile_rect(in.means2d[idx * 2], in.means2d[idx * 2 + 1],
                                          radius_x, radius_y, tile_size_f, TW, TH);
        for (int32_t i = r.min_y; i < r.max_y; i++) {                          // :435
            for (int32_t j = r.min_x; j < r.max_x; j++) {                      // :448
                const uint64_t tile_id = static_cast<uint64_t>(i * TW + j);    // :454
                out.isect_ids[static_cast<size_t>(cur_idx)] =
                    iid_enc | (tile_id << 32) | depth_id_enc;                  // :457
                out.flatten_ids[static_cast<size_t>(cur_idx)] =
                    static_cast<int32_t>(idx);                                 // :459
                ++cur_idx;
            }
        }
    }
    return out;
}

// Spec 03 sort contract — the CPU reference matching cub::DeviceRadixSort::SortPairs
// (Intersect.cpp:299–324, a stable radix sort). An index-based std::stable_sort: because the
// comparator looks only at the key (u64 unsigned), the stable semantics preserve the index (= emit)
// order of tied keys exactly — the criterion for bit-exact agreement with the isect_ties sort golden
// (np.lexsort) (task-2 report). Validating bits above key_n_bits is promoted to a throw on the CPU
// (the Metal GPU radix does not validate — Task 7): a violating key breaks the equivalence between
// the GPU radix (which sorts only the low key_n_bits) and a full u64 comparison, so encoding bugs are
// caught here.
SortIsectOutputs sort_isect_cpu(const SortIsectInputs& in) {
    validate_sort_isect_inputs(in, "sort_isect_cpu");
    SortIsectOutputs out;
    const size_t n = static_cast<size_t>(in.n);
    out.keys_sorted.resize(n);
    out.values_sorted.resize(n);
    if (n == 0) return out; // host skip — the same contract as n_isects==0 upstream

    if (in.key_n_bits < 64) { // at == 64 every bit is valid (avoids the shift-by-64 UB)
        for (size_t i = 0; i < n; i++)
            if (in.keys[i] >> in.key_n_bits)
                throw std::invalid_argument(
                    "sort_isect_cpu: key has bits above key_n_bits at idx " +
                    std::to_string(i));
    }

    std::vector<int32_t> order(n); // n <= INT32_MAX (validator) — an int32 index is safe
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return in.keys[a] < in.keys[b]; // unsigned key comparison only — for ties, stable preserves the original order
    });
    for (size_t i = 0; i < n; i++) {
        out.keys_sorted[i] = in.keys[order[i]];
        out.values_sorted[i] = in.values[order[i]];
    }
    return out;
}

// Spec 03 §Kernel 4 isect_offset_encode — a direct port of intersect_offset_kernel
// (IntersectTile.cu:925–988). It assumes sorted isect_ids (upstream Task 6/7) — only boundary
// elements write, and the written ranges do not overlap (idx==0: [0, id_curr], boundary:
// (id_prev, id_curr], last: (id_curr, I*n_tiles) — spec 03 second verification: every range is
// written exactly once). The example from the original comment (:936–939): ids=[1,1,1,3,3],
// n_tiles=6 → offsets [0,0,3,3,5,5].
std::vector<int32_t> isect_offset_encode_cpu(const IsectOffsetInputs& in) {
    validate_isect_offset_inputs(in, "isect_offset_encode_cpu");
    const int64_t n_tiles = static_cast<int64_t>(in.tile_width) * in.tile_height;
    const int64_t total = in.I * n_tiles;
    std::vector<int32_t> offsets(static_cast<size_t>(total), 0);
    if (in.n_isects == 0) return offsets; // host zero-fill — offsets.fill_(0) (:1004–1008)

    const uint32_t tile_n_bits = bits_for_count(n_tiles); // the same single source of truth as the encoding (spec 03 §1)
    // 64-bit mask — guards against the tile_n_bits=32 UB in the original's `(1 << tile_n_bits)` (int, :949).
    const uint64_t tile_mask = (static_cast<uint64_t>(1) << tile_n_bits) - 1;
    // The high 32 bits of the key (image|tile) → the flattened tile index iid*n_tiles + tid (:947–950).
    const auto flat_tile_id = [&](uint64_t key) {
        const uint64_t isect_id = key >> 32; // drops the 32 depth bits (:947)
        return static_cast<int64_t>(isect_id >> tile_n_bits) * n_tiles +
               static_cast<int64_t>(isect_id & tile_mask);
    };

    for (int64_t idx = 0; idx < in.n_isects; idx++) {
        const int64_t id_curr = flat_tile_id(in.isect_ids[idx]);
        if (idx == 0) // zero-fills up to and including the first valid tile (:952–959)
            for (int64_t i = 0; i <= id_curr; i++) offsets[static_cast<size_t>(i)] = 0;
        if (idx == in.n_isects - 1) // fills the tail (:960–967)
            for (int64_t i = id_curr + 1; i < total; i++)
                offsets[static_cast<size_t>(i)] = static_cast<int32_t>(in.n_isects);
        if (idx > 0) { // boundary thread — the same (image,tile) is not a boundary (:969–987)
            if ((in.isect_ids[idx - 1] >> 32) == (in.isect_ids[idx] >> 32)) continue; // :974
            const int64_t id_prev = flat_tile_id(in.isect_ids[idx - 1]);
            for (int64_t i = id_prev + 1; i <= id_curr; i++) // includes the empty tile range (:981–986)
                offsets[static_cast<size_t>(i)] = static_cast<int32_t>(idx);
        }
    }
    return offsets;
}

namespace {

// A direct port of gsplat RasterizeToPixels3DGSDevice.cuh:37-55 (paired with the Metal helper in
// KernelCommon.h) — all arithmetic f32. vis: CUDA __expf → std::exp(float) (matching Metal
// precise::exp, D2).
struct GaussianWeightCpu {
    float vis;   // exp(-sigma)
    float alpha; // min(kMaxAlpha, opac * vis)
    bool  valid; // !(sigma < 0 || alpha < kAlphaThreshold)
};

inline GaussianWeightCpu eval_gaussian_weight_cpu(const float* conic, float dx, float dy,
                                                  float opac) {
    const float sigma =
        0.5f * (conic[0] * dx * dx + conic[2] * dy * dy) + conic[1] * dx * dy; // Device.cuh:47
    const float vis   = std::exp(-sigma);                                      // :48
    const float alpha = std::min(kMaxAlpha, opac * vis);                       // :49
    return {vis, alpha, !(sigma < 0.f || alpha < kAlphaThreshold)};            // :53
}

} // namespace

// Spec 05 rasterize forward CPU reference (a direct port of Fwd.cu:41-297 — pixel-sequential; the
// cooperative load and batch loops are numerically equivalent and therefore elided: the batch split
// is only a load schedule, and the t traversal is identical to a front-to-back sequential pass over
// the whole range). For the structural correspondence with the Metal kernel see
// Rasterize3DGSFwd.metal.
RasterizeFwdOutputs rasterize_fwd_cpu(const RasterizeFwdInputs& in) {
    validate_rasterize_fwd_inputs(in, "rasterize_fwd_cpu");
    const size_t I = static_cast<size_t>(in.I);
    const size_t H = in.image_height, W = in.image_width;
    const size_t n_tiles = static_cast<size_t>(in.tile_width) * in.tile_height;
    constexpr size_t CDIM = kRastCdim; // 3 (the single Phase 1 variant)

    RasterizeFwdOutputs out;
    out.render_colors.resize(I * H * W * CDIM);
    out.render_alphas.resize(I * H * W);
    out.last_ids.resize(I * H * W);
    const bool has_depth = in.depths != nullptr;
    if (has_depth) out.render_depths.resize(I * H * W);

    for (size_t img = 0; img < I; img++) {
        // Per-image pointer advance (Fwd.cu:92-103) — the gaussian inputs (means2d/conics/colors/
        // opacities) are NOT advanced (flatten_ids is an [I*N] global index — spec 05 supplement 1).
        const int32_t* offsets = in.isect_offsets + img * n_tiles;
        const float* bg = in.backgrounds ? in.backgrounds + img * CDIM : nullptr;
        const uint8_t* masks = in.masks ? in.masks + img * n_tiles : nullptr;
        float* rc = out.render_colors.data() + img * H * W * CDIM;
        float* ra = out.render_alphas.data() + img * H * W;
        int32_t* li = out.last_ids.data() + img * H * W;
        float* rd = has_depth ? out.render_depths.data() + img * H * W : nullptr;

        for (size_t tile = 0; tile < n_tiles; tile++) {
            const size_t tile_x = tile % in.tile_width;
            const size_t tile_y = tile / in.tile_width;

            // Masked-out tile: writes the background (0 if absent) and skips blending
            // (Fwd.cu:141-159 — an unverified path in v1, no fixture).
            const bool masked = masks && masks[tile] == 0;

            const int32_t range_start = offsets[tile];
            const int32_t range_end =
                (img == I - 1 && tile == n_tiles - 1)
                    ? static_cast<int32_t>(in.n_isects)
                    : offsets[tile + 1]; // image boundaries are exact too because the accumulation is global (Fwd.cu:164-168)

            for (size_t yy = 0; yy < in.tile_size; yy++) {
                const size_t out_y = tile_y * in.tile_size + yy;
                if (out_y >= H) continue; // OOB (partial tile)
                for (size_t xx = 0; xx < in.tile_size; xx++) {
                    const size_t out_x = tile_x * in.tile_size + xx;
                    if (out_x >= W) continue;
                    const size_t pix = out_y * W + out_x;

                    if (masked) {
                        for (size_t k = 0; k < CDIM; k++)
                            rc[pix * CDIM + k] = bg ? bg[k] : 0.f;
                        ra[pix] = 0.f;
                        li[pix] = 0;
                        if (rd) rd[pix] = 0.f;
                        continue;
                    }

                    const float px = static_cast<float>(out_x) + 0.5f; // pixel center (+0.5)
                    const float py = static_cast<float>(out_y) + 0.5f;
                    float T = 1.0f;
                    uint32_t cur_idx = 0u; // no contribution = 0 (a dual-purpose encoding — Fwd.cu:186)
                    float pix_out[CDIM] = {0.f, 0.f, 0.f};
                    float pix_depth = 0.f;

                    // front-to-back alpha blending (Fwd.cu:223-263)
                    for (int32_t idx = range_start; idx < range_end; idx++) {
                        const int32_t g = in.flatten_ids[idx];
                        const float dx = in.means2d[2 * g] - px;
                        const float dy = in.means2d[2 * g + 1] - py;
                        const GaussianWeightCpu gw = eval_gaussian_weight_cpu(
                            in.conics + 3 * g, dx, dy, in.opacities[g]);
                        if (!gw.valid) continue;
                        const float next_T = T * (1.0f - gw.alpha);
                        if (next_T <= kTransmittanceThreshold)
                            break; // early termination — exclusive (the trigger is not included, Fwd.cu:246-251)
                        const float vis = gw.alpha * T;
                        for (size_t k = 0; k < CDIM; k++) // weighted-sum fma (paired with metal::fma in Metal)
                            pix_out[k] = std::fma(in.colors[g * CDIM + k], vis, pix_out[k]);
                        if (rd) pix_depth = std::fma(in.depths[g], vis, pix_depth);
                        cur_idx = static_cast<uint32_t>(idx); // absolute isect index (:261)
                        T = next_T;
                    }

                    // Final write (Fwd.cu:274-296) — the background is written even for pixels with an empty range (Correction 4)
                    ra[pix] = 1.0f - T;
                    for (size_t k = 0; k < CDIM; k++)
                        rc[pix * CDIM + k] = bg ? std::fma(T, bg[k], pix_out[k]) : pix_out[k];
                    li[pix] = static_cast<int32_t>(cur_idx);
                    if (rd) rd[pix] = pix_depth;
                }
            }
        }
    }
    return out;
}

// ===== Spec 06 MCMC (Phase 1c Task 5) =====

// Spec 06 rng_philox_u32 CPU reference — the same mapping as Rng.metal:
// for thread t, counter = (t, offset_lo, offset_hi, 0) and key = (seed_lo, seed_hi).
// It is pure integer arithmetic (the single source of truth is RngPhilox.h), so bit-identical with
// Metal is the contract.
std::vector<std::uint32_t> rng_philox_u32_cpu(const RngNormalInputs& in) {
    validate_rng_inputs(in, "rng_philox_u32_cpu");
    std::vector<std::uint32_t> out(static_cast<size_t>(in.count));
    const uint32_t seed_lo = static_cast<uint32_t>(in.seed);
    const uint32_t seed_hi = static_cast<uint32_t>(in.seed >> 32);
    const uint32_t off_lo = static_cast<uint32_t>(in.offset);
    const uint32_t off_hi = static_cast<uint32_t>(in.offset >> 32);
    const size_t n_threads = (out.size() + 3) / 4;
    for (size_t t = 0; t < n_threads; t++) {
        const SXPhilox4 r = sx_philox4x32_10(static_cast<uint32_t>(t), off_lo, off_hi, 0u,
                                             seed_lo, seed_hi);
        for (size_t j = 0; j < 4 && t * 4 + j < out.size(); j++) out[t * 4 + j] = r.v[j];
    }
    return out;
}

// Spec 06 rng_normal_f32 CPU reference — philox 4×u32 → uniform (0,1] → two Box-Muller pairs
// (RngPhilox.h — u1=0 is impossible, 06 Correction 5). Box-Muller uses the host libm (matching Metal
// precise:: — bit-identical through the integer and uniform stages; only the transcendentals are
// gated relatively).
std::vector<float> rng_normal_f32_cpu(const RngNormalInputs& in) {
    validate_rng_inputs(in, "rng_normal_f32_cpu");
    std::vector<float> out(static_cast<size_t>(in.count));
    const uint32_t seed_lo = static_cast<uint32_t>(in.seed);
    const uint32_t seed_hi = static_cast<uint32_t>(in.seed >> 32);
    const uint32_t off_lo = static_cast<uint32_t>(in.offset);
    const uint32_t off_hi = static_cast<uint32_t>(in.offset >> 32);
    const size_t n_threads = (out.size() + 3) / 4;
    for (size_t t = 0; t < n_threads; t++) {
        const SXPhilox4 r = sx_philox4x32_10(static_cast<uint32_t>(t), off_lo, off_hi, 0u,
                                             seed_lo, seed_hi);
        const SXNormal2 a = sx_box_muller(r.v[0], r.v[1]);
        const SXNormal2 b = sx_box_muller(r.v[2], r.v[3]);
        const float z[4] = {a.z0, a.z1, b.z0, b.z1};
        for (size_t j = 0; j < 4 && t * 4 + j < out.size(); j++) out[t * 4 + j] = z[j];
    }
    return out;
}

// Spec 06 binoms LUT (a reimplementation of strategy/mcmc.py:95–100) — Pascal's triangle accumulated
// in u64 integers (C(50,25) ≈ 1.26e14 < 2^63 — exact), then cast to float. Bit-identical to the
// original math.comb → float32 tensor (the RTNE cast loss is reproduced identically as well — 06
// §Kernel 2 "to float in the same way").
std::vector<float> make_binoms_lut(int n_max) {
    if (n_max < 1) throw std::invalid_argument("make_binoms_lut: n_max must be >= 1");
    std::vector<float> lut(static_cast<size_t>(n_max) * static_cast<size_t>(n_max), 0.0f);
    std::vector<uint64_t> row(static_cast<size_t>(n_max), 0); // updates the Pascal row in place
    for (int n = 0; n < n_max; n++) {
        for (int k = n; k >= 1; k--) row[k] += row[k - 1]; // C(n,k) = C(n-1,k-1) + C(n-1,k)
        row[0] = 1;
        for (int k = 0; k <= n; k++)
            lut[static_cast<size_t>(n) * n_max + k] = static_cast<float>(row[k]);
    }
    return lut;
}

// Spec 06 mcmc_relocation (a reimplementation of RelocationCUDA.cu:35–79) — the same formula as the
// Metal kernel (AdamMCMC.metal). All arithmetic f32; pow is forced onto the float path
// (std::pow(float,float) → powf — 06 Correction 2: avoids the unqualified pow(double) promotion
// trap); the clamp comes before the scale computation; and denom_sum accumulates sequentially in
// ascending (i,k) order (the same order as the golden fixture — Verification Notes, remaining gap 2).
RelocationOutputs mcmc_relocation_cpu(const RelocationInputs& in) {
    validate_relocation_inputs(in, "mcmc_relocation_cpu");
    const size_t N = static_cast<size_t>(in.N);
    RelocationOutputs out;
    out.new_opacities.resize(N);
    out.new_scales.resize(N * 3);
    for (size_t idx = 0; idx < N; idx++) {
        const int n_idx = in.ratios[idx];
        // A violation of the host clamp contract (relocation.py:61–62) throws — promoted validation (Metal does not validate)
        if (n_idx < 1 || n_idx > in.n_max)
            throw std::invalid_argument("mcmc_relocation_cpu: ratios[" + std::to_string(idx) +
                                        "] out of [1, n_max] (host clamp contract)");
        const float old_op = in.opacities[idx];

        // new opacity (:60–62) — the clamp comes before the scale computation (the min/max order also follows the original's fminf(fmaxf))
        float new_opacity =
            1.0f - std::pow(1.0f - old_op, 1.0f / static_cast<float>(n_idx));
        new_opacity = std::min(std::max(new_opacity, in.min_opacity), 1.0f - FLT_EPSILON);
        out.new_opacities[idx] = new_opacity;

        // new scale (:64–78) — the sign trick (k&1) is mathematically equivalent to pow(-1,k) (same as the kernel)
        float denom_sum = 0.0f;
        for (int i = 1; i <= n_idx; i++) {
            for (int k = 0; k <= i - 1; k++) {
                const float bin_coeff = in.binoms[static_cast<size_t>(i - 1) * in.n_max + k];
                const float sign = (k & 1) ? -1.0f : 1.0f;
                const float term = (sign / std::sqrt(static_cast<float>(k + 1))) *
                                   std::pow(new_opacity, static_cast<float>(k + 1));
                denom_sum += bin_coeff * term;
            }
        }
        const float coeff = old_op / denom_sum;
        for (int i = 0; i < 3; i++)
            out.new_scales[idx * 3 + i] = coeff * in.scales[idx * 3 + i];
    }
    return out;
}

// Spec 06 mcmc_perturb_positions (a reimplementation of MCMCPerturbCUDA.cu:27–61 +
// Utils.cuh:228–311) — the same formula as the Metal kernel. The quat normalization is 1/sqrt of the
// sum of squares in x→y→z→w order (matching precise::rsqrt in KernelCommon.h quat_to_rotmat); R is
// the row expansion of Utils.cuh:240–250; Σ = (R·S)·(R·S)ᵀ; sigmoid is 1/(1+exp(-x)) (:22–25); and
// the perturbation is Σ·n (not Cholesky). The covariance composition rounds differently from the
// Metal float3x3 product, so it is compared under a tolerance (the spec 06 fma/precise policy — the
// meta gate).
void mcmc_perturb_positions_cpu(const PerturbInputs& in) {
    if (in.N == 0) return; // the same as the dispatch-skip convention (D8-6 / MCMCPerturbCUDA.cu:74–78)
    validate_perturb_inputs(in, "mcmc_perturb_positions_cpu");
    for (int64_t idx = 0; idx < in.N; idx++) {
        const float* q = in.quats + idx * 4; // (w,x,y,z)
        float w = q[0], x = q[1], y = q[2], z = q[3];
        const float inv_norm = 1.0f / std::sqrt(x * x + y * y + z * z + w * w);
        x *= inv_norm; y *= inv_norm; z *= inv_norm; w *= inv_norm;
        const float x2 = x * x, y2 = y * y, z2 = z * z;
        const float xy = x * y, xz = x * z, yz = y * z;
        const float wx = w * x, wy = w * y, wz = w * z;
        // R row-major (Utils.cuh:240–250 — the same elements as the column layout in KernelCommon.h)
        const float R[3][3] = {
            {1.f - 2.f * (y2 + z2), 2.f * (xy - wz),       2.f * (xz + wy)},
            {2.f * (xy + wz),       1.f - 2.f * (x2 + z2), 2.f * (yz - wx)},
            {2.f * (xz - wy),       2.f * (yz + wx),       1.f - 2.f * (x2 + y2)},
        };
        const float s[3] = {std::exp(in.scales_log[idx * 3 + 0]),
                            std::exp(in.scales_log[idx * 3 + 1]),
                            std::exp(in.scales_log[idx * 3 + 2])};
        float M[3][3]; // M = R·diag(s)
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) M[i][j] = R[i][j] * s[j];
        float C[3][3]; // Σ = M·Mᵀ (symmetric)
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                C[i][j] = M[i][0] * M[j][0] + M[i][1] * M[j][1] + M[i][2] * M[j][2];

        const float density = 1.0f / (1.0f + std::exp(-in.opacities_logit[idx])); // :52
        const float gate = 1.0f / (1.0f + std::exp(in.k * (density - in.t)));     // :53
        const float wgt = gate * in.noise_scale;

        const float n0 = in.noise[idx * 3 + 0] * wgt;
        const float n1 = in.noise[idx * 3 + 1] * wgt;
        const float n2 = in.noise[idx * 3 + 2] * wgt;
        in.positions[idx * 3 + 0] += C[0][0] * n0 + C[0][1] * n1 + C[0][2] * n2; // Σ·n
        in.positions[idx * 3 + 1] += C[1][0] * n0 + C[1][1] * n1 + C[1][2] * n2;
        in.positions[idx * 3 + 2] += C[2][0] * n0 + C[2][1] * n1 + C[2][2] * n2;
    }
}

// ===== Spec 08 SSIM forward (Phase 3 Task 2) =====
// Written from scratch — the standard formula (Wang et al. 2004), consistent with the gsplat
// fused_ssim call contract (padding=valid, losses.py:185). Not derived from any code (fused-ssim was
// never cloned and never touched — 08 §1).
// The same f32 arithmetic as the Metal kernel (Ssim.metal): a separable 11-tap convolution in 2
// passes (horizontal → vertical), a fixed tap order (kSsimWindow is the constant single source of
// truth, KernelConstants.h), and std::fma accumulation (paired with metal::fma — the lesson of 1a
// Task 4). The ssim scalar comes from ssim_reduce_mean_f64 (host f64 summation in index order — the
// single source of truth).
SsimFwdOutputs ssim_fwd_cpu(const SsimInputs& in) {
    validate_ssim_inputs(in, "ssim_fwd_cpu");
    const std::size_t H = static_cast<std::size_t>(in.H);
    const std::size_t W = static_cast<std::size_t>(in.W);
    const std::size_t C = static_cast<std::size_t>(in.C);
    const std::size_t map_h = H - (kSsimWindowSize - 1); // H - 10
    const std::size_t map_w = W - (kSsimWindowSize - 1); // W - 10
    const std::size_t map_count = static_cast<std::size_t>(in.I) * map_h * map_w * C;

    SsimFwdOutputs out;
    out.ssim_map.resize(map_count);
    if (in.need_coeffs) {
        out.coeff_mu.resize(map_count);
        out.coeff_sigma.resize(map_count);
        out.coeff_cross.resize(map_count);
    }

    // Processed independently per (i, c) (08 §3) — the horizontal pass intermediates are 5 buffers of
    // [H, map_w] (matching kernel phase 1).
    std::vector<float> h_m1(H * map_w), h_m2(H * map_w), h_e11(H * map_w), h_e22(H * map_w),
        h_e12(H * map_w);
    for (int64_t i = 0; i < in.I; i++) {
        for (std::size_t c = 0; c < C; c++) {
            // horizontal 11 taps — all rows × valid columns
            for (std::size_t y = 0; y < H; y++) {
                for (std::size_t xw = 0; xw < map_w; xw++) {
                    float m1 = 0.f, m2 = 0.f, e11 = 0.f, e22 = 0.f, e12 = 0.f;
                    for (uint32_t u = 0; u < kSsimWindowSize; u++) {
                        const float w = kSsimWindow[u];
                        const std::size_t off =
                            (((static_cast<std::size_t>(i) * H + y) * W) + xw + u) * C + c;
                        const float x = in.img1[off];
                        const float g = in.img2[off];
                        m1  = std::fma(w, x, m1);
                        m2  = std::fma(w, g, m2);
                        e11 = std::fma(w * x, x, e11);
                        e22 = std::fma(w * g, g, e22);
                        e12 = std::fma(w * x, g, e12);
                    }
                    const std::size_t hoff = y * map_w + xw;
                    h_m1[hoff] = m1; h_m2[hoff] = m2;
                    h_e11[hoff] = e11; h_e22[hoff] = e22; h_e12[hoff] = e12;
                }
            }
            // vertical 11 taps + the SSIM map (+ the 3 coefficient maps — 08 §4-(1,2))
            for (std::size_t my = 0; my < map_h; my++) {
                for (std::size_t mx = 0; mx < map_w; mx++) {
                    float mu1 = 0.f, mu2 = 0.f, e11 = 0.f, e22 = 0.f, e12 = 0.f;
                    for (uint32_t v = 0; v < kSsimWindowSize; v++) {
                        const float w = kSsimWindow[v];
                        const std::size_t hoff = (my + v) * map_w + mx;
                        mu1 = std::fma(w, h_m1[hoff], mu1);
                        mu2 = std::fma(w, h_m2[hoff], mu2);
                        e11 = std::fma(w, h_e11[hoff], e11);
                        e22 = std::fma(w, h_e22[hoff], e22);
                        e12 = std::fma(w, h_e12[hoff], e12);
                    }
                    // σ = E[·] - μ·μ (08 §3 — the denominator is safe thanks to the C2 lower bound).
                    // Explicit fma — identical to the Metal kernel (it removes the arbitrariness of
                    // fast-math/FP_CONTRACT contraction).
                    const float sigma1  = std::fma(-mu1, mu1, e11);
                    const float sigma2  = std::fma(-mu2, mu2, e22);
                    const float sigma12 = std::fma(-mu1, mu2, e12);
                    const float A1 = 2.f * mu1 * mu2 + kSsimC1;
                    const float B1 = mu1 * mu1 + mu2 * mu2 + kSsimC1;
                    const float A2 = 2.f * sigma12 + kSsimC2;
                    const float B2 = sigma1 + sigma2 + kSsimC2;
                    const float denom = B1 * B2;
                    const float S = (A1 * A2) / denom;
                    const std::size_t moff =
                        (((static_cast<std::size_t>(i) * map_h + my) * map_w) + mx) * C + c;
                    out.ssim_map[moff] = S;
                    if (in.need_coeffs) {
                        const float dS_dmu1     = 2.f * (mu2 * A2 / denom - mu1 * S / B1);
                        const float dS_dsigma1  = -S / B2;
                        const float dS_dsigma12 = 2.f * A1 / denom;
                        out.coeff_mu[moff] =
                            dS_dmu1 - 2.f * mu1 * dS_dsigma1 - mu2 * dS_dsigma12;
                        out.coeff_sigma[moff] = dS_dsigma1;
                        out.coeff_cross[moff] = dS_dsigma12;
                    }
                }
            }
        }
    }
    out.ssim = ssim_reduce_mean_f64(out.ssim_map.data(), map_count);
    return out;
}

} // namespace splatx
