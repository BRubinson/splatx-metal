// AdamMCMC.metal — fused_adam / mcmc_relocation / mcmc_perturb_positions (spec 06).
// Derived from nerfstudio-project/gsplat (Apache-2.0), commit 77ab983.
// Reimplemented for Metal/C++.
// Originals: AdamCUDA.cu:34–71 (fused_adam), RelocationCUDA.cu:35–79 (mcmc_relocation),
//            MCMCPerturbCUDA.cu:27–61 + Utils.cuh:228–311 (mcmc_perturb_positions).
#include <metal_stdlib>
#include "KernelConstants.h"
#include "FunctionConstants.h"
#include "KernelParams.h"
#include "KernelCommon.h"
using namespace metal;

// spec 06 fused_adam (ported from AdamCUDA.cu:34–71) — one thread = one scalar element, [0, N*D).
// - no bias correction (exactly as the upstream original — D8-7; the step counter is not passed either)
// - eps is added outside sqrt(v), and sqrt is precise:: (a D2 parity point — close to the CPU golden)
// - valid is uchar [N] (D3). In a pipeline with kAdamHasValidMask=false the argument itself is
//   removed, so no binding is needed (D8-5 — function constant specialization in place of a
//   null-buffer branch).
// - no threadgroup memory, no atomics, no barriers. Dispatch: dispatchThreads(N*D, tg 256).
kernel void fused_adam(
    device float*        param      [[buffer(0)]], // [N*D] in-place
    device const float*  param_grad [[buffer(1)]], // [N*D]
    device float*        exp_avg    [[buffer(2)]], // [N*D] in-place
    device float*        exp_avg_sq [[buffer(3)]], // [N*D] in-place
    device const uchar*  valid      [[buffer(4), function_constant(kAdamHasValidMask)]], // [N]
    constant AdamParams& P          [[buffer(5)]],
    uint p_idx [[thread_position_in_grid]])
{
    const uint g_idx = p_idx / P.D;
    if (g_idx >= P.N) return;                           // AdamCUDA.cu:52 bounds guard
    if (kAdamHasValidMask && valid[g_idx] == 0) return; // :56 — if not visible, skip everything

    const float g = param_grad[p_idx];
    float m = exp_avg[p_idx];
    float v = exp_avg_sq[p_idx];
    // Explicit fma: blocks fast-math from distributing (1-b)*x into x - b*x, which produces a
    // cancellation error (measured at ~3e-5 rel). CUDA also contracts to FFMA by default, so the
    // rounding is of the same family as the upstream original.
    m = metal::fma(1.0f - P.b1, g, P.b1 * m);
    v = metal::fma(1.0f - P.b2, g * g, P.b2 * v);
    const float step = -P.lr * m / (metal::precise::sqrt(v) + P.eps);

    param[p_idx]     += step;
    exp_avg[p_idx]    = m;
    exp_avg_sq[p_idx] = v;
}

// spec 06 mcmc_relocation (ported from RelocationCUDA.cu:35–79) — MCMC Eq.9
// ("3D Gaussian Splatting as Markov Chain Monte Carlo", arXiv:2404.09591).
// One thread = one Gaussian. Preprocessing the inputs (opacity sigmoid, scale exp, and clamping
// ratios to 1..n_max) is the host's responsibility (relocation.py:55–66 — the kernel has no
// defensive clamp, for parity with the upstream original).
// - the opacity clamp comes **before** the scale computation (comment at RelocationCUDA.cu:57–62 —
//   an intentional deviation from the transparency invariant of Eq.9, for numerical stability at
//   near-zero opacity)
// - `pow(-1,k)` is forbidden (a negative base gives NaN under fast-math — the common policy of
//   spec 06): use the sign trick (k&1)
// - pow and sqrt are precise:: (D2 — close to the CPU golden), and denom_sum accumulates
//   sequentially in ascending (i,k) order (the same order as the CPU reference and the fixture
//   golden — Verification Notes, remaining gap 2)
// - binoms is an [n_max, n_max] row-major LUT — in the constant address space (broadcast cache,
//   51×51×4B = 10404B). No threadgroup memory, no atomics, no barriers. Dispatch: N, tg 256.
kernel void mcmc_relocation(
    device const float*   opacities     [[buffer(0)]], // [N] post-sigmoid
    device const float*   scales        [[buffer(1)]], // [N,3] linear (after the exp activation)
    device const int*     ratios        [[buffer(2)]], // [N] pre-clamped to 1..n_max
    constant float*       binoms        [[buffer(3)]], // [n_max*n_max] row-major LUT
    constant RelocParams& P             [[buffer(4)]],
    device float*         new_opacities [[buffer(5)]], // [N]
    device float*         new_scales    [[buffer(6)]], // [N,3]
    uint idx [[thread_position_in_grid]])
{
    if (idx >= P.N) return; // RelocationCUDA.cu:49 bounds guard

    const int n_idx = ratios[idx];
    const float old_op = opacities[idx];

    // New opacity (:60–62) — the clamp precedes the scale computation
    float new_opacity = 1.0f - metal::precise::pow(1.0f - old_op, 1.0f / float(n_idx));
    new_opacity = metal::min(metal::max(new_opacity, P.min_opacity), 1.0f - FLT_EPSILON);
    new_opacities[idx] = new_opacity;

    // New scale (:64–78) — denom_sum = Σ_{i=1..n} Σ_{k=0..i-1} C(i-1,k)·(-1)^k/√(k+1)·ô^{k+1}
    float denom_sum = 0.0f;
    for (int i = 1; i <= n_idx; ++i) {
        for (int k = 0; k <= i - 1; ++k) {
            const float bin_coeff = binoms[(i - 1) * P.n_max + k]; // LUT access (:69)
            const float sign = (k & 1) ? -1.0f : 1.0f; // replaces pow(-1.0f,k) — avoids NaN
            const float term = (sign / metal::precise::sqrt(float(k + 1)))
                             * metal::precise::pow(new_opacity, float(k + 1));
            denom_sum += bin_coeff * term;
        }
    }
    const float coeff = old_op / denom_sum; // :74
    for (int i = 0; i < 3; ++i)
        new_scales[idx * 3 + i] = coeff * scales[idx * 3 + i]; // the same coefficient on all 3 axes
}

// spec 06 mcmc_perturb_positions (ported from MCMCPerturbCUDA.cu:27–61) — SGLD-like noise injection.
// One thread = one Gaussian, positions in-place (each thread touches only its own elements — no hazard).
// - the kernel **does not generate** random numbers — noise [N,3] is an input buffer (produced by
//   rng_normal_f32; given the noise, the kernel is fully deterministic — the bit-exact comparison
//   path against the golden fixture, spec 06 design rationale 1)
// - the quat→R→Σ composition uses the shared helper quat_scale_to_covar in KernelCommon.h
//   (Utils.cuh:228–311 — a single definition shared with the EWA projection: precise::rsqrt
//   normalization, wxyz order, column-major)
// - `Σ·n` — multiplies by the covariance **itself** (not a Cholesky factor or a sqrt,
//   MCMCPerturbCUDA.cu:56)
// - exp and sigmoid are precise:: (D2). The covariance composition (the float3x3 product) rounds in
//   the fma-contraction family — compared against the sequential-sum CPU reference within tolerance
//   (the fixture meta gate). Dispatch: N, tg 256.
kernel void mcmc_perturb_positions(
    device float*           positions       [[buffer(0)]], // [N,3] in-place
    device const float*     quats           [[buffer(1)]], // [N,4] wxyz, pre-activation
    device const float*     scales_log      [[buffer(2)]], // [N,3] log-scale (the kernel applies exp)
    device const float*     opacities_logit [[buffer(3)]], // [N] logit (the kernel applies sigmoid)
    device const float*     noise           [[buffer(4)]], // [N,3] standard normal (input)
    constant PerturbParams& P               [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= P.N) return; // MCMCPerturbCUDA.cu:40 bounds guard

    const float4 quat = float4(quats[idx * 4 + 0], quats[idx * 4 + 1],
                               quats[idx * 4 + 2], quats[idx * 4 + 3]); // (w,x,y,z)
    const float3 scale = float3(metal::precise::exp(scales_log[idx * 3 + 0]),
                                metal::precise::exp(scales_log[idx * 3 + 1]),
                                metal::precise::exp(scales_log[idx * 3 + 2])); // :47 expf
    const metal::float3x3 covar = quat_scale_to_covar(quat, scale); // Σ = R·S·Sᵀ·Rᵀ (:49–50)

    const float density = sigmoid_standard(opacities_logit[idx]);            // :52
    const float w = sigmoid_standard(-P.k * (density - P.t)) * P.noise_scale; // :53 opacity gate

    const float3 n = float3(noise[idx * 3 + 0], noise[idx * 3 + 1],
                            noise[idx * 3 + 2]) * w;  // :55
    const float3 transformed = covar * n;              // :56 — Σ·n (not Cholesky)

    positions[idx * 3 + 0] += transformed.x; // :58–60 in-place addition
    positions[idx * 3 + 1] += transformed.y;
    positions[idx * 3 + 2] += transformed.z;
}
