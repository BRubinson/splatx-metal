// TrainStep.metal — the small elementwise kernels used only by the device-resident train_step
// (Phase 3 Task 5).
// Written in-house — these turn the D7 host-side step operations of Trainer.cpp (activations,
// SH prep, post-op, L1 grad, the activation gradient chain, the regularization addition, and the
// coeffs assembly/split) into kernels, and **Trainer.cpp is the single source of truth for the
// formulas and the order of operations** (the same f32 expressions are reproduced — the comparison
// gate is the resident comparison test in TrainerTests).
// All of them are pure maps (one thread per element, no atomics, no barriers, no threadgroup
// memory) → bit-exact determinism.
//
// Precision caution (report §Design decisions): the exp in train_exp/train_sigmoid is
// metal::precise::exp — it is not guaranteed to be bit-identical to the host std::exp (libm)
// (of the D2 family: the same contract as the perturb kernel). The host path vs. resident path
// comparison is therefore tolerance-based. **Within** the resident path it is deterministic
// (the same parameter bits → the same activation bits).
#include <metal_stdlib>
#include "KernelConstants.h"
#include "FunctionConstants.h"
#include "KernelParams.h"
using namespace metal;

// scales_lin = exp(log_scales) — corresponds to the host activate_scales.
kernel void train_exp(
    device const float* in  [[buffer(0)]],
    device float*       out [[buffer(1)]],
    constant TrainEltParams& P [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= P.n) return;
    out[i] = precise::exp(in[i]);
}

// opacs_act = sigmoid(logits) — corresponds to the host activate_opacities (1/(1+exp(-x))).
// The division is precise::divide — this keeps the fast-math reciprocal approximation from
// perturbing the activation values themselves (the same intent as the 1b Task 3 measured convention).
kernel void train_sigmoid(
    device const float* in  [[buffer(0)]],
    device float*       out [[buffer(1)]],
    constant TrainEltParams& P [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= P.n) return;
    out[i] = precise::divide(1.0f, 1.0f + precise::exp(-in[i]));
}

// out = a * b — AA opacity_aa = opacs_act · comp (corresponds to the AA block of Trainer.cpp forward_pass).
kernel void train_ew_mul(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant TrainEltParams& P [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= P.n) return;
    out[i] = a[i] * b[i];
}

// a += b — the v_means += v_dirs reduction (corresponds to the v_means accumulation in Trainer.cpp step()).
kernel void train_add_inplace(
    device float*       a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant TrainEltParams& P [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= P.n) return;
    a[i] += b[i];
}

// valid = (radii>0).all(-1), dirs = means − campos — prep of the SH fwd inputs (D6/D7).
kernel void train_prep_sh(
    device const int*   radii [[buffer(0)]], // [N,2]
    device const float* means [[buffer(1)]], // [N,3]
    device float*       dirs  [[buffer(2)]], // [N,3]
    device uchar*       valid [[buffer(3)]], // [N] (the D3 uchar contract)
    constant TrainPrepShParams& P [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= P.N) return;
    valid[i] = (radii[i * 2] > 0 && radii[i * 2 + 1] > 0) ? 1 : 0;
    dirs[i * 3 + 0] = means[i * 3 + 0] - P.campos_x;
    dirs[i * 3 + 1] = means[i * 3 + 1] - P.campos_y;
    dirs[i * 3 + 2] = means[i * 3 + 2] - P.campos_z;
}

// Assembles coeffs [N,K,3] — k=0 comes from sh0, k>=1 from shN (the same direction as the coeffs
// assembly gather in Trainer.cpp — no reversal).
// grid = N*K*3. With K==1 no thread reaches the shN branch (shN may be bound to a dummy buffer).
kernel void train_assemble_coeffs(
    device const float* sh0    [[buffer(0)]], // [N,3]
    device const float* shN    [[buffer(1)]], // [N,K-1,3] (not accessed when K==1)
    device float*       coeffs [[buffer(2)]], // [N,K,3]
    constant TrainCoeffsParams& P [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    const uint total = P.N * P.K * 3;
    if (idx >= total) return;
    const uint i = idx / (P.K * 3);
    const uint r = idx % (P.K * 3);
    const uint k = r / 3;
    const uint c = r % 3;
    coeffs[idx] = (k == 0) ? sh0[i * 3 + c] : shN[(i * (P.K - 1) + (k - 1)) * 3 + c];
}

// Splits v_coeffs [N,K,3] → v_sh0 [N,3] / v_shN [N,K-1,3] (the inverse of the assembly — Trainer.cpp step()).
kernel void train_split_v_coeffs(
    device const float* v_coeffs [[buffer(0)]],
    device float*       v_sh0    [[buffer(1)]],
    device float*       v_shN    [[buffer(2)]], // not accessed when K==1 (dummy bind allowed)
    constant TrainCoeffsParams& P [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    const uint total = P.N * P.K * 3;
    if (idx >= total) return;
    const uint i = idx / (P.K * 3);
    const uint r = idx % (P.K * 3);
    const uint k = r / 3;
    const uint c = r % 3;
    if (k == 0) v_sh0[i * 3 + c] = v_coeffs[idx];
    else        v_shN[(i * (P.K - 1) + (k - 1)) * 3 + c] = v_coeffs[idx];
}

// SH post-op: pre = colors + 0.5, relu_mask = pre > 0, colors = max(pre, 0) —
// reproduces the host std::max(pre, 0.0f) semantics as (pre < 0 ? 0 : pre) (with NaN, pre is kept —
// this matches the host even in how NaN propagates out of garbage in masked-out slots, and it is
// harmless because nothing downstream consumes it).
kernel void train_sh_post_op(
    device float* colors    [[buffer(0)]], // [N,3] in-place
    device uchar* relu_mask [[buffer(1)]], // [N,3]
    constant TrainEltParams& P [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= P.n) return;
    const float pre = colors[i] + 0.5f;
    relu_mask[i] = pre > 0.0f ? 1 : 0;
    colors[i] = (pre < 0.0f) ? 0.0f : pre;
}

// L1 upstream grad (+ combined SSIM grad) — reproduces the gradient part of the host
// l1_loss_and_grad plus the combining expression in step():
//   base = sign(render-gt)·g (sign(0)=0), out = has_dssim ? w_l1·base + v_dssim : base.
// (With λ=0, base passes through unchanged — bit-identical to the previous pure-L1 behavior.
// The loss scalar is owned by the host f64 summation.)
kernel void train_l1_grad(
    device const float* render  [[buffer(0)]],
    device const float* gt      [[buffer(1)]],
    device const float* v_dssim [[buffer(2)]], // not accessed when has_dssim=0 (dummy bind allowed)
    device float*       out     [[buffer(3)]],
    constant TrainL1GradParams& P [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= P.n) return;
    const float d = render[i] - gt[i];
    const float base = d > 0.0f ? P.g : (d < 0.0f ? -P.g : 0.0f);
    out[i] = P.has_dssim ? (P.w_l1 * base + v_dssim[i]) : base;
}

// Inverse-depth L1 gradient over the expected depth ED = acc/α (reproduces the host
// depth_loss_and_grad, the single source of truth for the formula): with r = 1/ED − 1/gt,
//   v_acc = g·sign(r)·(−1/ED²)/α,   v_α = −v_acc·ED.
// Pixels whose sidecar sample is 0 (no LiDAR or below the confidence floor), whose α is below
// kDepthMinAlpha or whose ED is below kDepthMinZ write 0 to both outputs.
kernel void train_depth_grad(
    device const float* render_depths   [[buffer(0)]], // [H,W] Σ w·z
    device const float* render_alphas   [[buffer(1)]], // [H,W]
    device const float* gt_depth        [[buffer(2)]], // [dh,dw] metres, 0 = invalid
    device float*       v_render_depths [[buffer(3)]], // [H,W]
    device float*       v_render_alphas [[buffer(4)]], // [H,W]
    constant TrainDepthGradParams& P [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= P.width * P.height) return;
    const uint x  = i % P.width;
    const uint y  = i / P.width;
    const uint sx = sx_depth_sidecar_coord(x, P.width, P.depth_width);
    const uint sy = sx_depth_sidecar_coord(y, P.height, P.depth_height);
    const float gt = gt_depth[sy * P.depth_width + sx];
    const float a  = render_alphas[i];
    float v_acc = 0.0f, v_a = 0.0f;
    if (gt > 0.0f && a >= kDepthMinAlpha) {
        const float ed = render_depths[i] / a;
        if (ed >= kDepthMinZ) {
            const float r = 1.0f / ed - 1.0f / gt;
            const float s = r > 0.0f ? 1.0f : (r < 0.0f ? -1.0f : 0.0f);
            v_acc = P.g * s * (-1.0f / (ed * ed)) / a;
            v_a   = -v_acc * ed;
        }
    }
    v_render_depths[i] = v_acc;
    v_render_alphas[i] = v_a;
}

// relu_mask multiply — blocks the gradient of channels where the clamp fired (host: if (!mask[i]) v[i] = 0).
kernel void train_relu_mask_apply(
    device float*       v_colors [[buffer(0)]], // [N,3] in-place
    device const uchar* mask     [[buffer(1)]], // [N,3]
    constant TrainEltParams& P [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= P.n) return;
    if (mask[i] == 0) v_colors[i] = 0.0f;
}

// The AA product-rule chain (reproduces the host chain_aa_opacity_grads_inplace):
//   v_comp = v_op_aa·σ, v_opacities = v_op_aa·comp (in-place).
kernel void train_aa_chain(
    device float*       v_opacities [[buffer(0)]], // in-place: d/d opacity_aa → toward d/d σ(op)
    device const float* opacs_act   [[buffer(1)]],
    device const float* comp        [[buffer(2)]],
    device float*       v_comp      [[buffer(3)]],
    constant TrainEltParams& P [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= P.n) return;
    const float v_op_aa = v_opacities[i];
    v_comp[i] = v_op_aa * opacs_act[i];
    v_opacities[i] = v_op_aa * comp[i];
}

// The scale gradient chain plus regularization (host chain_scale_grads_inplace →
// add_scale_reg_grad_inplace):
//   v *= scales_lin;  if reg_w > 0, v += reg_w·scales_lin (reg_w = reg/(N·3), computed on the host).
kernel void train_scale_grad_chain(
    device float*       v_scales   [[buffer(0)]], // [N,3] in-place
    device const float* scales_lin [[buffer(1)]],
    constant TrainGradChainParams& P [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= P.n) return;
    float v = v_scales[i] * scales_lin[i];
    if (P.reg_w > 0.0f) v += P.reg_w * scales_lin[i];
    v_scales[i] = v;
}

// The opacity gradient chain plus regularization (host chain_opacity_grads_inplace →
// add_opacity_reg_grad_inplace): σ′ = σ(1−σ), v *= σ′;  if reg_w > 0, v += reg_w·σ·(1−σ)
// (reg_w = reg/N).
kernel void train_opacity_grad_chain(
    device float*       v_opacities [[buffer(0)]], // [N] in-place
    device const float* opacs_act   [[buffer(1)]],
    constant TrainGradChainParams& P [[buffer(2)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= P.n) return;
    const float s = opacs_act[i];
    float v = v_opacities[i] * (s * (1.0f - s));
    if (P.reg_w > 0.0f) v += P.reg_w * s * (1.0f - s);
    v_opacities[i] = v;
}
