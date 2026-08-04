// Ssim.metal — ssim_fwd / ssim_bwd (spec 08 §3–§5, Phase 3 Task 2).
// Written in-house (not derived code) — the formulas are the well-known standard SSIM (Wang et al.,
// "Image Quality Assessment: From Error Visibility to Structural Similarity", IEEE TIP 2004) plus
// the gsplat fused_ssim call contract (losses.py:185 padding="valid", 11×11 σ=1.5,
// C1=1e-4/C2=9e-4, gradients for img1 only — the fused-ssim library was never cloned and its code
// was never touched). The analytic backward derivation (08 §4) is the authoritative formula, pinned
// by the fixture generator against autograd at rtol 1e-10.
//
// Structure (08 §5): a separable 11-tap conv in 2 passes (horizontal → vertical) + a threadgroup
// apron cache.
//  - fwd: threadgroup (16,16) = one thread per map pixel, grid (ceil(map_w/16), ceil(map_h/16), I·C)
//    — channels and images go on the z axis (each channel is independent, 08 §3).
//    apron 26×26 (= 16 + 11 - 1).
//  - bwd: one thread per input pixel, grid (ceil(W/16), ceil(H/16), I·C). The 3 coefficient maps
//    (Dμ/Dσ/Dc — the training-mode outputs of fwd, kSsimNeedCoeffs) are loaded into a zero-padded
//    apron and scattered by a conv (the adjoint of the valid correlation = a full conv; w is 180°
//    symmetric, so it takes the same forward tap loop form as fwd — 08 §4-(3)).
//
// **Apron boundary masking for partial tiles (08 §5 correction — Task 1 review, Important)**: the
// 26×26 apron of the partial tiles at the right and bottom of the grid extends past the image
// boundary (e.g. the 24×24 fixture — with map 14×14 a single tile already puts apron rows/columns
// 24 and 25 out of bounds). OOB coordinates load 0 — that 0 (or the garbage that would leak in
// without masking) never enters a valid map pixel: for a valid map pixel p (≤ map_h-1) the vertical
// pass covers apron rows p..p+10 ≤ H-1, and the columns consumed by the horizontal pass are
// likewise p..p+10 ≤ W-1 — it **always consumes only in-bounds ranges**, and the OOB slots are read
// only by the computation of threads outside the map range (which write nothing). The purpose of
// the masking is not to protect valid values but to block the OOB read of the device buffer itself
// (rationale in spec 08 §5/§9).
//
// D2 precision: the window taps come from the kSsimWindow f32 constants, the single source of truth
// (KernelConstants.h — no runtime transcendental functions); the accumulation is fp32 in a fixed
// tap order (u=0..10) with **explicit metal::fma** (the lesson of 1a Task 4 — paired with std::fma
// in the CPU reference); the divisions are plain float (the denominators are bounded below by
// C1/C2 — precise:: is unnecessary).
// No atomics and no simd reductions → both fwd and bwd are **bit-exact deterministic** (gated by a
// two-run memcmp).
// The ssim scalar reduction happens outside the kernel — the host sums the map sequentially in f64
// in ascending index order (08 §5 baseline proposal and open question 1 — device atomic float is
// forbidden).
#include <metal_stdlib>
#include "KernelConstants.h"
#include "FunctionConstants.h"
#include "KernelParams.h"
using namespace metal;

static_assert(kSsimTileSize * kSsimTileSize == 256, "ssim threadgroup must be 256 threads");
static_assert(kSsimApron == 26, "apron = tile + window - 1");

// [I,H,W,C] channel-last flattened offset (D6 — transposed buffers are forbidden; handled by indexing).
static inline ulong sx_ihwc(uint i, uint y, uint x, uint c, uint H, uint W, uint C)
{
    return ((((ulong)i * H + y) * W) + x) * C + c;
}

// ===== Kernel 1: ssim_fwd — the 5 moments → the SSIM map (+ the 3 training-mode coefficient maps) =====
//
// phase 0  cooperative load: the 26×26 apron tiles of img1/img2 → TG (boundary masking — header comment)
// phase 1  horizontal 11 taps: 26 apron rows × 16 valid columns → the 5 partial moments
//          (Σw·x, Σw·y, Σw·x², Σw·y², Σw·xy)
// phase 2  vertical 11 taps: one map pixel per thread — complete the moments → μ/σ → write S(p)
//          (+ Dμ/Dσ/Dc when kSsimNeedCoeffs — 08 §4-(1,2))
// TG memory: 26·26·2·4B = 5,408 + 26·16·5·4B = 8,320 → 13,728 B < 32 KB.
// Threads outside the map range in a partial tile skip only the write (every thread passes the
// barriers uniformly — there is no phase 3 reduction, so no alive flag is needed).
[[max_total_threads_per_threadgroup(256)]]
kernel void ssim_fwd(
    device const float* img1        [[buffer(0)]], // [I,H,W,C] render (the differentiated input)
    device const float* img2        [[buffer(1)]], // [I,H,W,C] GT
    device float*       ssim_map    [[buffer(2)]], // [I,map_h,map_w,C]
    device float*       coeff_mu    [[buffer(3), function_constant(kSsimNeedCoeffs)]], // Dμ
    device float*       coeff_sigma [[buffer(4), function_constant(kSsimNeedCoeffs)]], // Dσ
    device float*       coeff_cross [[buffer(5), function_constant(kSsimNeedCoeffs)]], // Dc
    constant SsimParams& P          [[buffer(6)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 lid    [[thread_position_in_threadgroup]])
{
    constexpr uint TILE  = kSsimTileSize;   // 16
    constexpr uint APRON = kSsimApron;      // 26
    constexpr uint WIN   = kSsimWindowSize; // 11

    threadgroup float tg_x[APRON * APRON];
    threadgroup float tg_y[APRON * APRON];
    threadgroup float h_m1[APRON * TILE]; // horizontal-pass partial moments — [apron row][valid column]
    threadgroup float h_m2[APRON * TILE];
    threadgroup float h_e11[APRON * TILE];
    threadgroup float h_e22[APRON * TILE];
    threadgroup float h_e12[APRON * TILE];

    const uint tid = lid.y * TILE + lid.x;
    const uint img = tg_pos.z / P.C; // image index
    const uint ch  = tg_pos.z % P.C; // channel index
    const uint oy  = tg_pos.y * TILE; // map tile origin = the window start coordinate (map[p] = start p)
    const uint ox  = tg_pos.x * TILE;

    // phase 0 — cooperative load (~5.3 elements per thread). Apron boundary masking for partial
    // tiles (header comment): OOB coordinates load 0 — valid map pixels consume only in-bounds
    // ranges, so no garbage leaks in.
    for (uint idx = tid; idx < APRON * APRON; idx += TILE * TILE) {
        const uint ay = idx / APRON, ax = idx % APRON;
        const uint gy = oy + ay, gx = ox + ax;
        float vx = 0.f, vy = 0.f;
        if (gy < P.H && gx < P.W) {
            const ulong off = sx_ihwc(img, gy, gx, ch, P.H, P.W, P.C);
            vx = img1[off];
            vy = img2[off];
        }
        tg_x[idx] = vx;
        tg_y[idx] = vy;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // phase 1 — horizontal 11 taps (26 rows × 16 columns = 416 items over 256 threads, fixed tap order u=0..10).
    for (uint idx = tid; idx < APRON * TILE; idx += TILE * TILE) {
        const uint r = idx / TILE, cx = idx % TILE;
        float m1 = 0.f, m2 = 0.f, e11 = 0.f, e22 = 0.f, e12 = 0.f;
        for (uint u = 0; u < WIN; u++) {
            const float w = kSsimWindow[u];
            const float x = tg_x[r * APRON + cx + u];
            const float y = tg_y[r * APRON + cx + u];
            m1  = fma(w, x, m1);
            m2  = fma(w, y, m2);
            e11 = fma(w * x, x, e11);
            e22 = fma(w * y, y, e22);
            e12 = fma(w * x, y, e12);
        }
        h_m1[idx] = m1;
        h_m2[idx] = m2;
        h_e11[idx] = e11;
        h_e22[idx] = e22;
        h_e12[idx] = e12;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // phase 2 — vertical 11 taps + SSIM (threads outside the map range skip the write — partial tiles).
    const uint my = oy + lid.y, mx = ox + lid.x;
    if (my >= P.map_h || mx >= P.map_w) return; // no barrier after this point — uniformity preserved
    float mu1 = 0.f, mu2 = 0.f, e11 = 0.f, e22 = 0.f, e12 = 0.f;
    for (uint v = 0; v < WIN; v++) {
        const float w = kSsimWindow[v];
        const uint idx = (lid.y + v) * TILE + lid.x;
        mu1 = fma(w, h_m1[idx], mu1);
        mu2 = fma(w, h_m2[idx], mu2);
        e11 = fma(w, h_e11[idx], e11);
        e22 = fma(w, h_e22[idx], e22);
        e12 = fma(w, h_e12[idx], e12);
    }
    // σ = E[·] - μ·μ (the same form as gsplat/fused-ssim — 08 §3: slightly negative values are
    // possible, and the denominator is bounded below by C2).
    // Explicit fma, exactly as in the CPU reference (removes the arbitrariness of fast-math
    // contraction — the lesson of 1a Task 4).
    const float sigma1  = fma(-mu1, mu1, e11);
    const float sigma2  = fma(-mu2, mu2, e22);
    const float sigma12 = fma(-mu1, mu2, e12);
    const float A1 = 2.f * mu1 * mu2 + kSsimC1;
    const float B1 = mu1 * mu1 + mu2 * mu2 + kSsimC1;
    const float A2 = 2.f * sigma12 + kSsimC2;
    const float B2 = sigma1 + sigma2 + kSsimC2;
    const float denom = B1 * B2;
    const float S = (A1 * A2) / denom;

    const ulong moff = sx_ihwc(img, my, mx, ch, P.map_h, P.map_w, P.C);
    ssim_map[moff] = S;

    if (kSsimNeedCoeffs) {
        // 08 §4-(1): ∂S/∂μ1, ∂S/∂σ1², ∂S/∂σ12 → §4-(2): the 3 coefficient maps of the raw-moment chain.
        const float dS_dmu1     = 2.f * (mu2 * A2 / denom - mu1 * S / B1);
        const float dS_dsigma1  = -S / B2;
        const float dS_dsigma12 = 2.f * A1 / denom;
        coeff_mu[moff]    = dS_dmu1 - 2.f * mu1 * dS_dsigma1 - mu2 * dS_dsigma12;
        coeff_sigma[moff] = dS_dsigma1;
        coeff_cross[moff] = dS_dsigma12;
    }
}

// ===== Kernel 2: ssim_bwd — scatter of the 3 coefficient maps (zero-padded full conv) + pointwise combination =====
//
// v_img1(q) = G·[ (w ⊛ Dμ)(q) + 2·x(q)·(w ⊛ Dσ)(q) + y(q)·(w ⊛ Dc)(q) ] — 08 §4-(3).
// ⊛ = the adjoint of the valid correlation = a full conv over the coefficient maps zero-padded by
// 10 on all four sides. w is 180° symmetric, so (w ⊛ D)(q) = Σ_s w(s)·D_pad(q - 10 + s) — it can be
// computed with the same forward tap loop as fwd (only the apron origin shifts by -10). One thread
// per input pixel — **every input pixel is written**, so no zero-fill and no atomics are needed,
// and it is subject to bit-exact determinism.
//
// Apron boundary masking (the dual of fwd): the coefficient-map apron always extends beyond the map
// range (including into negative coordinates) — **outside the map range, loading 0 is exactly the
// mathematical zero-padding** (unlike fwd's "the OOB values are never consumed", here the 0 itself
// is part of the derivation — 08 §4-(3) / §6, the bwd scatter row).
// TG memory: 26·26·3·4B = 8,112 + 26·16·3·4B = 4,992 → 13,104 B < 32 KB.
[[max_total_threads_per_threadgroup(256)]]
kernel void ssim_bwd(
    device const float* img1        [[buffer(0)]], // [I,H,W,C] x(q) — for the pointwise combination
    device const float* img2        [[buffer(1)]], // [I,H,W,C] y(q)
    device const float* coeff_mu    [[buffer(2)]], // [I,map_h,map_w,C] Dμ (a training-mode output of fwd)
    device const float* coeff_sigma [[buffer(3)]], // Dσ
    device const float* coeff_cross [[buffer(4)]], // Dc
    device float*       v_img1      [[buffer(5)]], // [I,H,W,C]
    constant SsimParams& P          [[buffer(6)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 lid    [[thread_position_in_threadgroup]])
{
    constexpr uint TILE  = kSsimTileSize;   // 16
    constexpr uint APRON = kSsimApron;      // 26
    constexpr uint WIN   = kSsimWindowSize; // 11
    constexpr int  PAD   = (int)WIN - 1;    // 10 — the zero-pad width of the full conv

    threadgroup float tg_cmu[APRON * APRON];
    threadgroup float tg_csg[APRON * APRON];
    threadgroup float tg_ccr[APRON * APRON];
    threadgroup float h_cmu[APRON * TILE];
    threadgroup float h_csg[APRON * TILE];
    threadgroup float h_ccr[APRON * TILE];

    const uint tid = lid.y * TILE + lid.x;
    const uint img = tg_pos.z / P.C;
    const uint ch  = tg_pos.z % P.C;
    const int  oy  = (int)(tg_pos.y * TILE); // input pixel tile origin
    const int  ox  = (int)(tg_pos.x * TILE);

    // phase 0 — cooperative load of the 3 coefficient maps, apron origin (oy-10, ox-10).
    // Outside the map range = 0 (zero-padding).
    for (uint idx = tid; idx < APRON * APRON; idx += TILE * TILE) {
        const int ay = (int)(idx / APRON), ax = (int)(idx % APRON);
        const int py = oy + ay - PAD, px = ox + ax - PAD; // map coordinates (the window start p)
        float cmu = 0.f, csg = 0.f, ccr = 0.f;
        if (py >= 0 && py < (int)P.map_h && px >= 0 && px < (int)P.map_w) {
            const ulong off = sx_ihwc(img, (uint)py, (uint)px, ch, P.map_h, P.map_w, P.C);
            cmu = coeff_mu[off];
            csg = coeff_sigma[off];
            ccr = coeff_cross[off];
        }
        tg_cmu[idx] = cmu;
        tg_csg[idx] = csg;
        tg_ccr[idx] = ccr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // phase 1 — horizontal 11 taps (the same loop form as fwd — w is symmetric, so the adjoint equals a forward conv).
    for (uint idx = tid; idx < APRON * TILE; idx += TILE * TILE) {
        const uint r = idx / TILE, cx = idx % TILE;
        float s1 = 0.f, s2 = 0.f, s3 = 0.f;
        for (uint u = 0; u < WIN; u++) {
            const float w = kSsimWindow[u];
            s1 = fma(w, tg_cmu[r * APRON + cx + u], s1);
            s2 = fma(w, tg_csg[r * APRON + cx + u], s2);
            s3 = fma(w, tg_ccr[r * APRON + cx + u], s3);
        }
        h_cmu[idx] = s1;
        h_csg[idx] = s2;
        h_ccr[idx] = s3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // phase 2 — vertical 11 taps + pointwise combination (threads outside the input bounds skip the write).
    const uint qy = (uint)oy + lid.y, qx = (uint)ox + lid.x;
    if (qy >= P.H || qx >= P.W) return; // no barrier after this point
    float c1 = 0.f, c2 = 0.f, c3 = 0.f;
    for (uint v = 0; v < WIN; v++) {
        const float w = kSsimWindow[v];
        const uint idx = (lid.y + v) * TILE + lid.x;
        c1 = fma(w, h_cmu[idx], c1);
        c2 = fma(w, h_csg[idx], c2);
        c3 = fma(w, h_ccr[idx], c3);
    }
    const ulong qoff = sx_ihwc(img, qy, qx, ch, P.H, P.W, P.C);
    const float x = img1[qoff];
    const float y = img2[qoff];
    v_img1[qoff] = P.G * (c1 + 2.f * x * c2 + y * c3);
}
