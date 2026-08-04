# 02. Spherical Harmonics forward / backward kernel spec

A code-level extraction document for the CUDA → Metal port. Every formula, constant, and sign is carried over from the upstream original verbatim.

## Upstream files and commit (77ab983)

Based on gsplat commit `77ab983ffe43420b2131669cb35776b883ca4c3c` (Apache-2.0, a derivation notice is required).

| File | Role |
|---|---|
| `research/gsplat/gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu` | device functions `sh_coeffs_to_color_fast` (:42-140) and `sh_coeffs_to_color_fast_vjp` (:142-430), two forward kernels (:434-534), the backward kernel (:710-762), the launchers (:536-829), and the fused `assemble_proj_features_kernel` (:913-1080) |
| `research/gsplat/gsplat/cuda/csrc/SphericalHarmonics.cpp` | input validation (:37-72), fwd/bwd op wrappers (:133-220), fused autograd (:341-545) |
| `research/gsplat/gsplat/cuda/csrc/SphericalHarmonics.h` | launcher signature declarations |
| `research/gsplat/gsplat/cuda/_wrapper.py` | the Python `spherical_harmonics` (:434-462), the autograd hook `RegisterSphericalHarmonics` (:465-495) |
| `research/gsplat/gsplat/rendering.py` | `_maybe_evaluate_sh` (:687-715): the actual call site of the dirs computation, the mask, and the clamp |

The SH basis evaluation is a hardcoded expansion in the style of Sloan, "Efficient Spherical Harmonic Evaluation", JCGT 2013 (comment at SphericalHarmonicsCUDA.cu:37-40).

## Algorithm overview

- Inputs: `dirs [..., N, 3]` (unnormalized view directions), `coeffs [N, K, D]` (SH coefficients; K = number of bands, D = number of channels, D=3 for RGB), and the optional `masks [..., N]` (bool).
- Output: `colors [..., N, D]`.
- `degrees_to_use` (`degree` from here on) is a runtime argument. Constraint: `(degree+1)^2 <= K` (SphericalHarmonics.cpp:48-54). With K=16 that means degree ≤ 3; the upstream original hardcodes support up to K=25 (degree 4).
- Coefficient indexing: `coeffs[k * D + c]`, with k = 0..K-1 and c = the channel. **coeffs is indexed by gaussian_id only**, so it is shared across the batch (camera) axis B — which is why the backward needs an atomic on v_coeffs (see below).
- The color returned is the raw SH sum; the `+0.5` shift and `clamp_min(·, 0)` are applied **outside the kernel** (in Python) or in the fused kernel's post-op.

### dirs computation (caller side, rendering.py:704-706)

```python
camtoworlds = torch.inverse(viewmats)                            # [..., C, 4, 4]
dirs = means[..., None, :, :] - camtoworlds[..., None, :3, 3]    # [..., C, N, 3]  = means - campos
masks = (radii > 0).all(dim=-1)                                  # [..., C, N]
```

- `dirs` is handed to the kernel **without normalization**. The normalization (`rsqrtf`) is performed inside the SH device function (SphericalHarmonicsCUDA.cu:59-62 comment: "SH normalizes internally"; :1027-1029 says the same).
- The fused kernel is identical: `dir = vec3(mp[0]-cx, mp[1]-cy, mp[2]-cz)` (SphericalHarmonicsCUDA.cu:1026-1029).

### clamp / shift handling (rendering.py:707-714)

```python
features = spherical_harmonics(sh_degree, dirs, features, masks=masks)  # [..., C, N, D]
if clamp:
    features = torch.clamp_min(features + 0.5, 0.0)   # same as the Inria CUDA backend
else:
    features = features + 0.5
```

In the fused kernel the same operation is folded inside the kernel as `SHPostOp` (SphericalHarmonicsCUDA.cu:837-857):

```cpp
enum SHPostOp : uint32_t { SH_POST_NONE = 0, SH_POST_SHIFT = 1, SH_POST_SHIFT_RELU = 2 };
// SHIFT:      v + 0.5
// SHIFT_RELU: s = v + 0.5; return s > 0 ? s : 0;
```

With `SHIFT_RELU` it also emits `relu_mask[ch] = dst[ch] > 0` (based on the post-clamp value) for the backward (:1053-1062). The backward routes `g_color = g_color.mul(relu_mask)` into the SH backward (SphericalHarmonics.cpp:485-489).

## Formula detail

### SH basis evaluation constants (all hardcoded values)

| degree | Identifier | Value |
|---|---|---|
| 0 | (no named coefficient) | `0.2820947917738781f` |
| 1 | (no named coefficient) | `0.48860251190292f` |
| 2 | `fTmp0B` coefficient | `-1.092548430592079f` |
| 2 | `pSH6` | `0.9461746957575601f`, `0.3153915652525201f` |
| 2 | `pSH4/pSH8` coefficient | `0.5462742152960395f` |
| 3 | `fTmp0C` | `-2.285228997322329f`, `0.4570457994644658f` |
| 3 | `fTmp1B` coefficient | `1.445305721320277f` |
| 3 | `pSH12` | `1.865881662950577f`, `1.119528997770346f` |
| 3 | `pSH9/pSH15` coefficient | `-0.5900435899266435f` |
| 4 | `fTmp0D` | `-4.683325804901025f`, `2.007139630671868f` |
| 4 | `fTmp1C` | `3.31161143515146f`, `0.47308734787878f` |
| 4 | `fTmp2B` coefficient | `-1.770130769779931f` |
| 4 | `pSH20` | `1.984313483298443f`, `-1.006230589874905f` |
| 4 | `pSH16/pSH24` coefficient | `0.6258357354491763f` |

### Forward: `sh_coeffs_to_color_fast` (SphericalHarmonicsCUDA.cu:42-140)

Call granularity: one (Gaussian element, channel c) pair. `opmath_t` is float for both fp16 and fp32. Coefficient reads always upcast to fp32 via `static_cast<opmath_t>(coeffs[k*D+c])`.

```cpp
// degree 0 (SphericalHarmonicsCUDA.cu:54)
opmath_t result = 0.2820947917738781f * coeffs[c];

if (degree >= 1) {
    // normalization — inside the kernel. Uses rsqrtf (:57-62, comment: sqrt is optimized under --use_fast_math too, but rsqrtf was chosen)
    float inorm = rsqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
    float x = dir.x * inorm, y = dir.y * inorm, z = dir.z * inorm;

    // degree 1 (:64-67) — mind the signs: the order is (-y, +z, -x)
    result += 0.48860251190292f * ( -y * coeffs[1*D+c]
                                    + z * coeffs[2*D+c]
                                    - x * coeffs[3*D+c] );
    if (degree >= 2) {                       // (:68-85)
        float z2     = z * z;
        float fTmp0B = -1.092548430592079f * z;
        float fC1    = x*x - y*y;
        float fS1    = 2.f * x * y;
        float pSH6   = 0.9461746957575601f * z2 - 0.3153915652525201f;
        float pSH7   = fTmp0B * x;
        float pSH5   = fTmp0B * y;
        float pSH8   = 0.5462742152960395f * fC1;
        float pSH4   = 0.5462742152960395f * fS1;
        result += pSH4*coeffs[4*D+c] + pSH5*coeffs[5*D+c] + pSH6*coeffs[6*D+c]
                + pSH7*coeffs[7*D+c] + pSH8*coeffs[8*D+c];

        if (degree >= 3) {                   // (:86-106)
            float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
            float fTmp1B = 1.445305721320277f * z;
            float fC2    = x*fC1 - y*fS1;    // cos(2φ)→cos(3φ) Chebyshev recurrence
            float fS2    = x*fS1 + y*fC1;
            float pSH12  = z * (1.865881662950577f * z2 - 1.119528997770346f);
            float pSH13  = fTmp0C * x;
            float pSH11  = fTmp0C * y;
            float pSH14  = fTmp1B * fC1;
            float pSH10  = fTmp1B * fS1;
            float pSH15  = -0.5900435899266435f * fC2;
            float pSH9   = -0.5900435899266435f * fS2;
            result += pSH9*coeffs[9*D+c] + pSH10*coeffs[10*D+c] + pSH11*coeffs[11*D+c]
                    + pSH12*coeffs[12*D+c] + pSH13*coeffs[13*D+c] + pSH14*coeffs[14*D+c]
                    + pSH15*coeffs[15*D+c];

            if (degree >= 4) {               // (:108-134) — reachable only when K=25
                float fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
                float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
                float fTmp2B = -1.770130769779931f * z;
                float fC3    = x*fC2 - y*fS2;
                float fS3    = x*fS2 + y*fC2;
                float pSH20  = 1.984313483298443f * z * pSH12 - 1.006230589874905f * pSH6;
                float pSH21  = fTmp0D * x;
                float pSH19  = fTmp0D * y;
                float pSH22  = fTmp1C * fC1;
                float pSH18  = fTmp1C * fS1;
                float pSH23  = fTmp2B * fC2;
                float pSH17  = fTmp2B * fS2;
                float pSH24  = 0.6258357354491763f * fC3;
                float pSH16  = 0.6258357354491763f * fS3;
                result += pSH16*coeffs[16*D+c] + ... + pSH24*coeffs[24*D+c];
            }
        }
    }
}
colors[c] = result;   // (:139)
```

Points to watch:

- **At degree 0 the kernel never reads dir at all** (the normalization code sits inside the `degree >= 1` branch).
- If `dir == (0,0,0)`, `rsqrtf(0) = +inf` → NaN propagation. The upstream original has no guard.
- The degree-1 signs and order (-y, +z, -x) follow the Inria 3DGS convention. A flipped sign here is the single most common mistake during verification.

### Forward kernel (generic): `spherical_harmonics_fwd_kernel` (SphericalHarmonicsCUDA.cu:434-468)

```cpp
idx = grid.thread_rank();                        // 1 thread = 1 (batch, gaussian, channel)
count = B * gaussian_count * D;  if (idx >= count) return;
local_elem_id = idx / D;
batch_id      = local_elem_id / gaussian_count;
gaussian_id   = local_elem_id % gaussian_count + gaussian_offset;   // on a single GPU, offset=0 and count=N
elem_id       = batch_id * N + gaussian_id;
c             = idx % D;
if (masks != nullptr && !masks[elem_id]) return; // masked out → nothing is written to colors!
sh_coeffs_to_color_fast(degrees_to_use, D, c, dirs[elem_id],
                        coeffs + gaussian_id * K * D, colors + elem_id * D);
```

- **colors for masked-out elements are unwritten** — the output tensor is `at::empty` (SphericalHarmonics.cpp:146), so those slots hold garbage. The structure works because the downstream rasterizer filters on radii>0.
- `gaussian_offset/gaussian_count` are for multi-GPU chunking (:664-707) — they can be removed in the Metal port (fixed at offset=0, count=N).

### Forward kernel (K=16, D=3 hot path): `spherical_harmonics_fwd_kernel_k16_3channel` (SphericalHarmonicsCUDA.cu:474-534)

- Condition: `K==16 && D==3 && wide_load_aligned` (launcher :561-565; the coeffs pointer must be `uint4` (16B) aligned, or `ushort4` (8B) aligned for fp16 with deg≤1). Falls back to generic when unmet.
- 1 thread = 1 (batch, gaussian). Loads 48 coefficients into registers with vector loads and then evaluates the three channels sequentially:

```cpp
// PACK_SIZE: DEGREE<=1 → 4;  DEGREE>=2 → fp32 4 / fp16 8   (:502)
// LOAD_COUNT: deg0→1, deg1→3, deg2→(fp32 7 / fp16 4), deg3→(fp32 12 / fp16 6)  (:503-506)
// N_COEFFS = LOAD_COUNT * PACK_SIZE  (deg0:4, deg1:12, deg2: 28/32, deg3: 48)
sh_coeffs_to_color_fast<opmath_t>(DEGREE, 3, 0, dirs[idx], sh_coeffs, out_color);
sh_coeffs_to_color_fast<opmath_t>(DEGREE, 3, 1, dirs[idx], sh_coeffs, out_color);
sh_coeffs_to_color_fast<opmath_t>(DEGREE, 3, 2, dirs[idx], sh_coeffs, out_color);
colors[idx*3 + {0,1,2}] = out_color[{0,1,2}];    // (:527-533)
```

- `__launch_bounds__(256, 4)` (:475); DEGREE is a template parameter (dispatched as 0/1/2/3, `deg = min(degrees_to_use, 3)` :584).

### Backward: `sh_coeffs_to_color_fast_vjp` (SphericalHarmonicsCUDA.cu:142-430)

For a single input `v_colors[c]` (fp32) it produces two kinds of gradient.

**(a) v_coeffs** — the basis values themselves are the Jacobian with respect to the coefficients:

```cpp
// every term: gpuAtomicAdd(&v_coeffs[k*D + c], pSH_k * v_colors_local);
// deg0: 0.2820947917738781f * v_colors_local                     (:159)
// deg1: -0.48860251190292f*y, +0.48860251190292f*z, -0.48860251190292f*x  (:170-172)
// deg2: pSH4..pSH8 * v_colors_local                              (:204-208)
// deg3: pSH9..pSH15 * v_colors_local                             (:268-274)
// deg4: pSH16..pSH24 * v_colors_local                            (:353-361)
```

`v_coeffs` is **always an fp32 buffer** (signature comment :151-152). Even with fp16 coeffs, accumulation happens in fp32 and the cast is done on the host.

**(b) v_dir** — accumulates the partial derivatives with respect to the normalized direction `(x,y,z)`, then applies the Jacobian of the normalization (the tangent projection) at the very end:

```cpp
// degree 1 contribution (:174-179)
v_x += -0.48860251190292f * coeffs[3*D+c] * v_colors_local;
v_y += -0.48860251190292f * coeffs[1*D+c] * v_colors_local;
v_z +=  0.48860251190292f * coeffs[2*D+c] * v_colors_local;

// degree 2 partial derivatives (:210-239)
fTmp0B_z = -1.092548430592079f;
fC1_x = 2*x;  fC1_y = -2*y;  fS1_x = 2*y;  fS1_y = 2*x;
pSH6_z = 2.f * 0.9461746957575601f * z;
pSH7_x = fTmp0B;          pSH7_z = fTmp0B_z * x;
pSH5_y = fTmp0B;          pSH5_z = fTmp0B_z * y;
pSH8_x = 0.5462742152960395f * fC1_x;   pSH8_y = 0.5462742152960395f * fC1_y;
pSH4_x = 0.5462742152960395f * fS1_x;   pSH4_y = 0.5462742152960395f * fS1_y;
v_x += v_colors_local * (pSH4_x*coeffs[4*D+c] + pSH8_x*coeffs[8*D+c] + pSH7_x*coeffs[7*D+c]);
v_y += v_colors_local * (pSH4_y*coeffs[4*D+c] + pSH8_y*coeffs[8*D+c] + pSH5_y*coeffs[5*D+c]);
v_z += v_colors_local * (pSH6_z*coeffs[6*D+c] + pSH7_z*coeffs[7*D+c] + pSH5_z*coeffs[5*D+c]);

// degree 3 partial derivatives (:276-321)
fTmp0C_z = -2.285228997322329f * 2.f * z;   fTmp1B_z = 1.445305721320277f;
fC2_x = fC1 + x*fC1_x - y*fS1_x;   fC2_y = x*fC1_y - fS1 - y*fS1_y;
fS2_x = fS1 + x*fS1_x + y*fC1_x;   fS2_y = x*fS1_y + fC1 + y*fC1_y;
pSH12_z = 3.f * 1.865881662950577f * z2 - 1.119528997770346f;
pSH13_x = fTmp0C;  pSH13_z = fTmp0C_z * x;
pSH11_y = fTmp0C;  pSH11_z = fTmp0C_z * y;
pSH14_x = fTmp1B * fC1_x;  pSH14_y = fTmp1B * fC1_y;  pSH14_z = fTmp1B_z * fC1;
pSH10_x = fTmp1B * fS1_x;  pSH10_y = fTmp1B * fS1_y;  pSH10_z = fTmp1B_z * fS1;
pSH15_x = -0.5900435899266435f * fC2_x;  pSH15_y = -0.5900435899266435f * fC2_y;
pSH9_x  = -0.5900435899266435f * fS2_x;  pSH9_y  = -0.5900435899266435f * fS2_y;
// v_x: the pSH9_x,15_x,10_x,14_x,13_x terms / v_y: the pSH9_y,15_y,10_y,14_y,11_y terms
// v_z: the pSH12_z,13_z,11_z,14_z,10_z terms  (each multiplied by its coeffs and accumulated, :302-321)

// degree 4 partial derivatives (:363-420)
fTmp0D_z = 3.f * -4.683325804901025f * z2 + 2.007139630671868f;
fTmp1C_z = 2.f * 3.31161143515146f * z;    fTmp2B_z = -1.770130769779931f;
fC3_x = fC2 + x*fC2_x - y*fS2_x;   fC3_y = x*fC2_y - fS2 - y*fS2_y;
fS3_x = fS2 + y*fC2_x + x*fS2_x;   fS3_y = x*fS2_y + fC2 + y*fC2_y;
pSH20_z = 1.984313483298443f * (pSH12 + z*pSH12_z) - 1.006230589874905f * pSH6_z;
// pSH21_x=fTmp0D, pSH21_z=fTmp0D_z*x, pSH19_y=fTmp0D, pSH19_z=fTmp0D_z*y,
// pSH22/18 = the x/y/z partials of fTmp1C·{fC1,fS1}, pSH23/17 = fTmp2B·{fC2,fS2}, pSH24/16 = 0.6258357354491763f·{fC3,fS3}_{x,y}
```

**Normalization Jacobian (identical at every degree's exit point, :184-190, :246-252, :328-334, :422-428):**

```cpp
vec3 dir_n   = vec3(x, y, z);           // the normalized direction
vec3 v_dir_n = vec3(v_x, v_y, v_z);     // grad with respect to the normalized direction
vec3 v_d = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;   // tangent projection × 1/||dir||
v_dir->{x,y,z} = v_d.{x,y,z};
```

- Per-degree early-return structure: `if (degree < k) { (after finishing v_dir) return; }` — equivalent to the forward's nested `if (degree >= k)`.
- `v_dir` accumulates thread-locally per channel and is summed globally with an atomic **in the calling kernel**.

### Backward kernel: `spherical_harmonics_bwd_kernel` (SphericalHarmonicsCUDA.cu:710-762)

```cpp
// same indexing as the generic forward (1 thread = 1 (batch, gaussian, channel))
if (masks != nullptr && !masks[elem_id]) return;   // no grad contribution (v_* are zero-initialized)
vec3 v_dir = {0,0,0};
sh_coeffs_to_color_fast_vjp(degrees_to_use, D, c, dirs[elem_id],
    coeffs   + gaussian_id * K * D,
    v_colors + elem_id * D,
    v_coeffs + gaussian_id * K * D,       // fp32, atomic accumulation
    v_dirs == nullptr ? nullptr : &v_dir);
if (v_dirs != nullptr) {
    gpuAtomicAdd(v_dirs + elem_id*3,     v_dir.x);   // (:756-761)
    gpuAtomicAdd(v_dirs + elem_id*3 + 1, v_dir.y);
    gpuAtomicAdd(v_dirs + elem_id*3 + 2, v_dir.z);
}
```

Why the atomics are needed:

- `v_coeffs[gaussian_id]`: coeffs is shared across the batch (camera) axis, so **threads of different batch_id contend for the same slot**. Within one elem the channels c occupy separate slots, so they never contend. With B==1 there is no v_coeffs contention at all.
- `v_dirs[elem_id]`: the **D channel threads** of the same elem (3 for RGB) contend for the same three elements.

Host wrapper (SphericalHarmonics.cpp:200-215):

```cpp
v_coeffs_accum = at::zeros(coeffs.sizes(), fp32);   // always accumulate in fp32 (precision)
v_dirs = compute_v_dirs ? at::zeros_like(dirs) : undefined;
launch_...bwd_kernel(...);
v_coeffs = (coeffs is fp32) ? v_coeffs_accum : v_coeffs_accum.to(coeffs.scalar_type());
```

`compute_v_dirs` is decided in the Python autograd hook from `ctx.needs_input_grad[1]` (the dirs input) (_wrapper.py:479-483).

### Reducing v_dirs → v_means / v_campos

On the standalone path, PyTorch autograd handles the broadcast of `dirs = means[...,None,:,:] - campos[...,None,:]` automatically. On the fused path it is explicit (SphericalHarmonics.cpp:481-521):

```cpp
dirs = means.unsqueeze(batch_ndim) - campos.unsqueeze(-2);   // [*b, C, N, 3]
if (color_post == SHIFT_RELU) g_color = g_color.mul(relu_mask);
launch_spherical_harmonics_bwd_kernel(..., g_color, v_coeffs_accum, v_dirs);
v_means  = v_dirs.sum(batch_ndim);   // sum over the C (camera) axis
v_campos = v_dirs.sum(-2).neg();     // sum over the N axis, sign-negated
```

## Thread and memory configuration

| Kernel | grid | block | Notes |
|---|---|---|---|
| fwd generic (:607-644) | `ceil(B*N*D / 256)` | 256 | 1D, no shared memory |
| fwd k16 (:565-606) | `ceil(B*N / 256)` | 256 | `__launch_bounds__(256,4)`, uint4/ushort4 wide loads |
| bwd (:783-829) | `ceil(B*N*D / 256)` | 256 | two atomic sites, no shared memory |
| fused assemble (:1124-1126) | `(ceil(N/64), B*C)` 2D | 256 | TILE_ROWS=64, dynamic shared: `64*cstride*sizeof(coeffT) + 64*3*4 + 64*width*4` bytes, `__syncthreads()` twice (:1008, :1072) |

Memory layout (all C-contiguous):

- `dirs [..., N, 3]` fp32 — reinterpreted in the kernel as `vec3` (glm, 12 bytes packed).
- `coeffs [N, K, D]` fp32 or fp16 — element `coeffs[n][k][c] = base + n*K*D + k*D + c`.
- `colors / v_colors [..., N, D]` fp32 (the dtype follows dirs, SphericalHarmonics.cpp:143-146).
- `v_coeffs [N, K, D]` fp32 (accumulation buffer), `v_dirs [..., N, 3]` fp32.
- `masks [..., N]` bool.

### How fp16 SH is supported (summary)

1. **Only coeffs** may be fp16 (`AT_DISPATCH_V2(..., at::kFloat, at::kHalf)` :614-643, :796-828). dirs/colors/v_colors/v_dirs are always fp32 (`opmath_t = at::opmath_type<scalar_t> = float`).
2. All arithmetic is fp32 — coefficients are upcast with `static_cast<opmath_t>()` the moment they are read.
3. The backward's v_coeffs is **accumulated atomically in fp32** and cast to half on the host (SphericalHarmonics.cpp:200-215). There is no half atomic inside the kernel.
4. Only the wide-load form of the k16 hot path varies with dtype: fp16 with deg≤1 → `ushort4` (4 halves), otherwise → `uint4` (4 fp32 / 8 halves).

## MSL porting design

### Proposed kernel signatures

```metal
// SplatXSH.metal — one source, specialized with function constants
#include <metal_stdlib>
using namespace metal;

struct SHParams {
    uint degrees_to_use;  // 0..4 (Phase 1 is 0..3)
    uint B;               // flattened batch*camera count
    uint N;               // number of Gaussians
    uint K;               // number of SH bands ((deg+1)^2 <= K)
    uint D;               // number of channels (RGB=3)
};

constant bool kHasMasks      [[function_constant(0)]];
constant bool kComputeVDirs  [[function_constant(1)]];
constant bool kCoeffsHalf    [[function_constant(2)]];  // or split out a dedicated fp16 kernel
constant bool kSingleBatch   [[function_constant(3)]];  // B==1 → v_coeffs can use non-atomic stores

kernel void sh_fwd(
    constant SHParams&           p       [[buffer(0)]],
    device const packed_float3* dirs     [[buffer(1)]],  // [B*N] — same layout as glm::vec3 (12B)
    device const float*         coeffs   [[buffer(2)]],  // [N,K,D] (device const half* when half)
    device const bool*          masks    [[buffer(3)]],  // unbound when kHasMasks=false
    device float*               colors   [[buffer(4)]],  // [B*N, D]
    uint gid [[thread_position_in_grid]])                // gid < B*N*D
{
    const uint elem = gid / p.D;          // same decomposition as CUDA
    const uint c    = gid % p.D;
    const uint gaussian_id = elem % p.N;  // offset/chunking removed (single device)
    if (kHasMasks && !masks[elem]) return;
    // port sh_coeffs_to_color_fast 1:1 as an inline function (rsqrtf → metal::rsqrt)
}

kernel void sh_bwd(
    constant SHParams&           p        [[buffer(0)]],
    device const packed_float3* dirs      [[buffer(1)]],
    device const float*         coeffs    [[buffer(2)]],
    device const bool*          masks     [[buffer(3)]],
    device const float*         v_colors  [[buffer(4)]],  // [B*N, D]
    device atomic_float*        v_coeffs  [[buffer(5)]],  // [N,K,D] fp32, zero-init required
    device atomic_float*        v_dirs    [[buffer(6)]],  // [B*N, 3] fp32, zero-init required
    uint gid [[thread_position_in_grid]])
{
    // port the vjp function 1:1. gpuAtomicAdd →
    //   atomic_fetch_add_explicit(&v_coeffs[k*D+c], val, memory_order_relaxed);
}
```

### Dispatch configuration

- **threadgroup size 256** (same as CUDA; simdgroup 32 × 8). Both fwd and bwd use `dispatchThreads(MTLSize(B*N*D,1,1), MTLSize(256,1,1))` — Apple GPUs support non-uniform threadgroups, so the exact thread count can be dispatched without the CUDA-style `if (idx >= count) return` guard. Keeping the guard is harmless.
- **No barriers needed**: the standalone fwd/bwd use no threadgroup memory. The only kernel that needs barriers is the fused assemble tile kernel (`threadgroup_barrier(mem_flags::mem_threadgroup)` twice when ported), and that may be deferred outside the Phase 1 scope (it can be replaced by pure SH plus a cat on the Python side).
- **Atomic sites (backward only)**:
  - `v_coeffs[gaussian_id*K*D + k*D + c]` — 1/3/5/7/9 `atomic_fetch_add_explicit(..., memory_order_relaxed)` calls depending on degree. M4/A17 have native atomic_float, so no CAS fallback is needed (settled by on-device measurement).
  - `v_dirs[elem*3 + {0,1,2}]` — 3 calls after the channel loop.
  - Under the `kSingleBatch` (B==1) specialization the v_coeffs contention disappears, so a plain load/add/store can replace it (only one thread touches a given slot). v_dirs still has cross-channel contention, so keep the atomic there — or remove that contention too with a D=3-only "1 thread = 1 Gaussian" variant (below).
- **D=3 hot-path variant (preferred, mirrors the k16 kernel)**: making 1 thread = 1 (batch, gaussian) and looping over the three channels (a) lets the forward reuse coeffs through float4/half4 vector loads, and (b) keeps the backward's v_dirs entirely in thread-local accumulation, leaving atomics only on v_coeffs (and none at all when B==1). CUDA's uint4/ushort4 map to MSL `float4`/`half4` loads, and because Metal buffers are 16B aligned by default the alignment fallback logic can be simplified — but check alignment when binding coeffs buffer offsets.

### 1:1 mapping table for the port

| CUDA | MSL |
|---|---|
| `rsqrtf(x)` | `rsqrt(x)` (fast-math on by default; prefer an `-fno-fast-math` comparison during golden verification) |
| `glm::vec3` (12B packed) | `packed_float3` (in buffers), `float3` for arithmetic |
| `glm::dot` | `dot` |
| `gpuAtomicAdd(float*)` | `atomic_fetch_add_explicit(device atomic_float*, v, memory_order_relaxed)` |
| upcast when reading `at::Half` coefficients | `float(coeffs_h[i])` |
| template DEGREE specialization | function constant `kDegree`, or a runtime argument plus a branch (branching is cheap on Apple GPUs; secure correctness with the runtime branch first, then specialize) |
| `__launch_bounds__(256,4)` | `maxTotalThreadsPerThreadgroup(256)` hint on the pipeline descriptor |

### Correctness pitfall checklist

1. The degree-1 signs (-y, +z, -x) and the coefficient order k=1 (the y term), k=2 (the z term), k=3 (the x term).
2. The product-rule expansion of `pSH20_z`: `1.984313483298443f * (pSH12 + z*pSH12_z) - 1.006230589874905f * pSH6_z`.
3. The v_dir tangent projection happens **exactly once, at the end**, including the multiplication by `inorm` (= 1/||dir_raw||).
4. Masked-out forward outputs are unwritten (garbage) in the upstream original — for the golden fixture comparison, the Metal port should prefer to **zero-init the output buffer and apply the mask when comparing**.
5. The v_coeffs and v_dirs buffers MUST be zero-initialized before the kernel runs (CUDA uses `at::zeros`).
6. The clamp (+0.5, max 0) is a separate operation outside the SH kernel — decide its position in the SXT pipeline explicitly. During training the relu mask (`post-clamp value > 0`) must be stored, otherwise the backward will not match.

## Open questions

1. **Semantics of the forward output for masked-out elements**: the upstream original uses `at::empty` and leaves them unwritten. Decide whether the Metal port switches to zero-fill, and whether the golden fixture excludes the masked region from the comparison.
2. **The dir == 0 vector**: `rsqrt(0) = inf` → NaN. Decide whether to leave it unguarded like the upstream original or add an epsilon guard (a guard can diverge from the golden values — prefer unguarded during the bit-exact verification stage).
3. **Scope of degree 4 (K=25) support**: the upstream original hardcodes it, but the k16 hot path and standard 3DGS training only go up to degree 3. The porting scope depends on whether the Phase 1 golden fixture includes deg 4.
4. **fast-math divergence**: the ULP difference between CUDA `--use_fast_math` + `rsqrtf` and Metal's default fast-math. What level to set the golden verification tolerance (absolute/relative) at — a 3-way comparison with the CPU reference is preferred.
5. **When to port the fused `assemble_proj_features`**: the forward combines SH + shift + relu_mask + depth/extra, and the backward multiplies by the relu mask and then reuses the SH bwd. The proposal is to port only the standalone SH in Phase 1 and defer the fused version to a later optimization stage — needs to be decided.
6. **Where the v_coeffs fp16 cast happens**: the upstream original runs a separate `.to(half)` cast kernel on the host. Decide whether Metal does it in a separate blit/compute pass, or whether SXT consumes the fp32 gradient directly.
7. **Whether the wide-load optimization pays off**: given Apple GPU memory coalescing, scalar loads may already be enough. Implement with scalar loads, measure bandwidth with Instruments (Metal System Trace), and then decide whether to introduce float4/half4 loads.

## Verification notes (adversarial verifier)

The upstream originals (SphericalHarmonicsCUDA.cu, SphericalHarmonics.cpp, _wrapper.py, rendering.py) were re-read independently and every formula, constant, sign, index, and clamp was compared 1:1. **All the core formulas match** — the 15 forward constants for degrees 0 through 4 (0.2820947917738781f through 0.6258357354491763f), the degree-1 signs and order (-y, +z, -x) with the coeffs indices (1→y, 2→z, 3→x), the Chebyshev recurrence (fC2/fS2/fC3/fS3), every vjp partial-derivative term (including fC1_x=2x, fC1_y=-2y, pSH12_z=3·1.865881662950577f·z2-1.119528997770346f, and the product-rule expansion of pSH20_z), the four normalization-Jacobian exit points (:184-190, :246-252, :328-334, :422-428), the atomic sites and the contention analysis (including the absence of v_coeffs contention when B==1), the k16 PACK_SIZE/LOAD_COUNT table, the fp16 handling (only coeffs may be half, immediately upcast to fp32, accumulated in fp32 atomics and cast on the host), and the v_means/v_campos reduction (sum(batch_ndim) / sum(-2).neg()) were all confirmed to match the upstream original.

Corrections and supplements found:

1. **[Supplement — missing semantics] The fused kernel writes 0 into masked-out color channels** (SphericalHarmonicsCUDA.cu:1064-1070: `else { for(ch) dst[ch] = 0; }`). This differs from the standalone forward's "unwritten (garbage)" semantics. In other words, the upstream original itself is inconsistent about masked-out output semantics between the two paths, which bears directly on the answer to open question 1: if the Metal port chooses zero-fill, it agrees with the fused path and differs only from the standalone path. Note also that the fused `relu_mask` is unwritten for masked-out rows (`at::empty` garbage; :1055-1061 sits inside the active branch), but this has no effect on the gradient because the SH bwd kernel early-returns on the same mask in the backward (:739-742).
2. **[Correction — comment citation] The upstream comment at :57-58 reads "so we use sqrt here", but the actual code uses `rsqrtf` (:59)** — a stale comment in the original that disagrees with its own code. This spec's formula (rsqrtf) matches the actual code and therefore remains valid. Stated here so that anyone re-checking the upstream comment is not confused.
3. **[Correction — citation location] The "SH normalizes internally" comment actually lives at :1026-1027 in the fused kernel** (the dirs computation section cites it as "the :59-62 comment", but :59-62 is the normalization code itself, not that comment). The fact itself — that normalization is performed inside the kernel — is confirmed by the code (:59-62, :164-167) and remains valid.
4. **[Supplement — missing file scope] The backward has its own separate multi-device launcher too**: `launch_spherical_harmonics_bwd_kernels` (SphericalHarmonicsCUDA.cu:1327-1393), with the privateuseone backend wrapper at SphericalHarmonics.cpp:589-630 (`spherical_harmonics_bwd_privateuseone`). For the same reason as the forward chunking (:664-707), the Metal port can ignore it (offset=0, count=N).
5. **[Confirmed — type evidence] `vec3` is defined as `using vec3 = glm::vec<3, float>`** (research/gsplat/gsplat/cuda/include/Common.h:65, 12 bytes) — this settles the basis for this spec's claim that it corresponds to `packed_float3`.
6. **[Minor — wording] "Loads 48 coefficients with vector loads" (the k16 section) is stated for DEGREE=3**; lower degrees load only what they need, per the N_COEFFS table (4/12/28~32) (the table in that same section is already accurate, so treat the table as the reference). Also, fp16 deg2 loads LOAD_COUNT 4 × PACK_SIZE 8 = 32 elements and uses only 27 (9 bands × 3 channels) — safe because a K=16 row holds 48 elements, so the over-read stays within the row (:505).

No errors beyond those 6 items. Zero constant, sign, index, or clamp errors. Together with the notes above, this document is sufficient as the sole reference for the Metal implementation.

### Second independent re-verification (2026-07-11)

After confirming that the commit hash `77ab983ffe43420b2131669cb35776b883ca4c3c` matches, the four upstream files (all 1394 lines of SphericalHarmonicsCUDA.cu, all 644 lines of SphericalHarmonics.cpp, _wrapper.py:425-505, rendering.py:680-724) and Common.h:64-66 were read again from the start, and the body plus all six first-pass verification notes above were re-checked in full.

- **Formulas**: all 15 forward constant values, every vjp partial-derivative term (including degree 4, e.g. fTmp0D_z=3·(-4.683325804901025f)·z2+2.007139630671868f and fS3_x=fS2+y·fC2_x+x·fS2_x), the four normalization-Jacobian exit points, and the degree-1 signs and coeffs indices — zero errors.
- **The upstream's pSH20/pSH20_z notation**: the vjp original writes `+ -1.006230589874905f` (SphericalHarmonicsCUDA.cu:344, :375) while the forward writes `- 1.006230589874905f` (:115) — mathematically identical, so this spec's `-` notation is valid.
- **Indexing / atomics**: the generic fwd/bwd thread decomposition (:456-460, :734-738), k16's `masks[idx]` (:490, the same semantics as the generic `masks[elem_id]`), the v_coeffs and v_dirs atomic sites (:159-361, :758-760), and the B==1 no-contention analysis — all match.
- **Host / Python side**: the (degree+1)²≤K validation (.cpp:48-54), `at::empty` colors (:146), fp32 accumulation → host cast (:203-215), the `at::empty` creation of relu_mask (:401) and `g_color.mul(relu_mask)` (:485-489), `v_means=v_dirs.sum(batch_ndim)` (:515) / `v_campos=v_dirs.sum(-2).neg()` (:519), `needs_input_grad[1]` (_wrapper.py:480), and dirs/clamp (rendering.py:704-714) — all match.
- **The six first-pass verification notes**: the upstream evidence for each item (:1064-1070 zero-fill, :57-59 stale comment, :1026-1027 comment location, :1327-1393/:589-630 multi-device paths, Common.h:65 vec3, and the safety of the fp16 deg2 32-load/27-use over-read) was independently confirmed — all six are accurate.
- Minor notation: the _wrapper.py line citations (:434-462, :465-495) are off by ±1 line from the actual definitions (433-461, 464-494), but this does not hinder identification.

**Verification passed, no new corrections.** Together with the six first-pass notes, this document is sufficient as the sole reference for the Metal implementation.
