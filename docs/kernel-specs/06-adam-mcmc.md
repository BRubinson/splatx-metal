# 06. Fused Adam + MCMC kernels (SelectiveAdam / Relocation / MCMC Perturb)

## Original files and commit (77ab983)

Based on gsplat commit `77ab983ffe43420b2131669cb35776b883ca4c3c` (2026-07-09).
All paths are under `research/gsplat/`. License: Apache-2.0 (a derivation notice is required).

| Component | File | Key location |
|---|---|---|
| fused Adam kernel | `gsplat/cuda/csrc/AdamCUDA.cu` | kernel lines 34–71, launch lines 73–120 |
| Adam op binding | `gsplat/cuda/csrc/Adam.cpp` | lines 37–62 (input validation) |
| SelectiveAdam (Python) | `gsplat/optimizers/selective_adam.py` | lines 21–101 |
| Adam Python wrapper | `gsplat/cuda/_wrapper.py` | lines 417–430 (`adam`) |
| relocation kernel | `gsplat/cuda/csrc/RelocationCUDA.cu` | kernel lines 35–79, launch lines 81–125 |
| relocation Python | `gsplat/relocation.py` | lines 25–67 (`compute_relocation`) |
| MCMC perturb kernel | `gsplat/cuda/csrc/MCMCPerturbCUDA.cu` | kernel lines 27–61, launch lines 63–97 |
| MCMC perturb binding | `gsplat/cuda/csrc/MCMCPerturb.cpp` | lines 23–64 (tensor contract) |
| Covariance composition helpers | `gsplat/cuda/include/Utils.cuh` | `quat_to_rotmat` lines 228–251, `quat_scale_to_covar_preci` lines 285–311, `rsqrt_parse_safe` lines 43–50 |
| Type definitions | `gsplat/cuda/include/Common.h` | lines 65–70 (`vec3/vec4/mat3` = glm float types) |
| MCMC strategy orchestration | `gsplat/strategy/mcmc.py` | lines 82–100 (constants and the binoms LUT), lines 139–193 (step_post_backward) |
| relocate/perturb call sites | `gsplat/strategy/ops.py` | lines 33–58 (constants), 301–356 (relocate), 359–401 (sample_add), 404–511 (perturb) |
| Where eps=1e-15 is used | `examples/simple_trainer.py` | lines 368–377, 1122–1135 |

All three kernels in this document are **optimizer/state manipulation operations** and run inside a `@torch.no_grad()` context. **There is no backward kernel** (they sit outside the autograd graph).

---

## Algorithm overview

### 1. Fused Adam (SelectiveAdam)

A "visible-only" Adam in the Taming-3DGS line. Differences from the standard `torch.optim.Adam`:

1. **The parameter update, the first/second moment updates, and the step application are fused into a single kernel** — one dispatch per tensor.
2. **Visibility mask**: Gaussians that did not contribute to the render on that step (the invisible ones) are **skipped entirely** for both parameters and moments (not even the moment decay is applied).
3. **No bias correction**: the `1-β₁ᵗ` and `1-β₂ᵗ` correction terms of the standard Adam are absent from the kernel. `selective_adam.py:78` keeps `state["step"]` but never passes it to the kernel → effectively unused.
4. eps takes the `sqrt(v) + eps` form (added **after** the sqrt in the denominator; not `sqrt(v + eps)`).

### 2. Relocation (MCMC Eq. 9)

An implementation of Eq. (9) of "3D Gaussian Splatting as Markov Chain Monte Carlo" (arXiv:2404.09591).
When a dead Gaussian (opacity ≤ min_opacity) is teleported to the position of a live Gaussian, it computes the new opacity and scale so that the render result (the transmittance invariant) is preserved even though one Gaussian is duplicated into `n` copies.

- **The binomial coefficients are not computed in the kernel; it reads a LUT precomputed on the host.**
  `strategy/mcmc.py:95–100`: with `n_max = 51` and `binoms[n][k] = math.comb(n, k)`, a `[51, 51]` float tensor is built and kept resident on the device. The kernel accesses it as `binoms[(i-1) * n_max + k]` (row-major) (`RelocationCUDA.cu:69`).
- The double loop runs at worst `Σ_{i=1..51} i = 1326` iterations per thread.

### 3. MCMC Perturb (noise injection)

Every step it adds covariance- and opacity-weighted Gaussian noise to `params["means"]` in place (an SGLD-like exploration term). What is fused: quat normalization → rotation matrix → covariance composition (`Σ = R·S·Sᵀ·Rᵀ`) → the opacity gate → the noise transform → the position addition, all in a single kernel.

**Random number generation (important)**: the kernel **does not generate** random numbers. It **receives the standard-normal noise `[N,3]` tensor as an input buffer**.

- gsplat: `strategy/ops.py:438` `noise = torch.randn_like(positions)` → generated in advance by PyTorch's default CUDA generator (**Philox4x32-10**, counter-based, of the cuRAND family) and then passed to the kernel.
- brush (Apache-2.0, for reference): `crates/brush-train/src/train.rs:376–380` creates the tensor with burn's `Distribution::Normal(0.0, 1.0)` → again a framework-side RNG, not an RNG inside the kernel.

In other words, in the upstream CUDA original there is no code that "calls curand inside a kernel". In the Metal port too, **keeping the noise buffer as an input** is decisively advantageous for golden fixture verification (given the noise, the kernel is fully deterministic). The noise producer is designed as a separate counter-based RNG kernel (see the MSL design below).

---

## Formulas in detail

### 1. Fused Adam — forward (no backward)

`AdamCUDA.cu:34–71`. Thread index and mask:

```
p_idx = cg::this_grid().thread_rank()        // global scalar element index, [0, N*D)
g_idx = p_idx / D                            // Gaussian index
if (g_idx >= N) return;                      // AdamCUDA.cu:52
if (valid != nullptr && !valid[g_idx]) return;  // AdamCUDA.cu:56 — if invisible, skip everything
```

The update expressions (`AdamCUDA.cu:61–70`; all arithmetic is in float32 registers — even with `scalar_t=double` the internals are float):

```
g   = param_grad[p_idx]
m   = exp_avg[p_idx];  v = exp_avg_sq[p_idx]
m   = b1 * m + (1.0f - b1) * g
v   = b2 * v + (1.0f - b2) * g * g
step = -lr * m / (sqrt(v) + eps)             // no bias correction; eps is outside the sqrt
param[p_idx]     += step
exp_avg[p_idx]    = m
exp_avg_sq[p_idx] = v
```

Parameter contract (`Adam.cpp:37–59`):
- `param, param_grad, exp_avg, exp_avg_sq`: identically shaped `[N, ...]`, contiguous.
- `valid`: optional `bool [N]` (`valid.dim()==1 && valid.size(0)==N` is enforced). If `nullptr`, everything is updated.
- `lr, b1, b2, eps`: scalars (cast double → float).

**Handling of eps=1e-15**: the kernel uses eps as given. The actual training configuration is at `examples/simple_trainer.py:368–375`:

```
lr    = lr * sqrt(BS)                        # BS = batch_size * world_size
eps   = 1e-15 / sqrt(BS)
betas = (1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999))
```

`1e-15` is representable in fp32 (the smallest normal is ≈ 1.18e-38), so there is no cast loss. However, if `v` becomes tiny (≲1e-38) and enters the denormal range, `sqrt(v)≈0` makes `step ≈ -lr * m / eps` (×1e15) blow up — that is the structure, and the upstream CUDA original has the same property (from a parity standpoint it is reproduced as-is).

The SelectiveAdam Python layer (`selective_adam.py:62–101`):
- Exactly one tensor per parameter group is enforced (the assert at line 70).
- `exp_avg/exp_avg_sq` are lazily zero-initialized (lines 79–84).
- It computes `M = param.numel() // N`, but the kernel's `D` is recomputed in the launcher (`AdamCUDA.cu:85–86`) from `param.size(0)` and `numel()`.
- Where the visibility mask comes from (`simple_trainer.py:1122–1130`): in packed mode a `gaussian_ids` scatter, otherwise `(info["radii"] > 0).all(-1).any(0)` → bool `[N]`.

### 2. Relocation — forward (no backward)

`RelocationCUDA.cu:35–79`. One Gaussian per thread (`idx = threadIdx.x + blockIdx.x * blockDim.x`; return if `idx >= N`).

Inputs (preprocessed in Python, `relocation.py:55–66`, `strategy/ops.py:326–335`):
- `opacities[idx]` ∈ (0,1): values sampled from `sigmoid(params["opacities"])`. `[N]`
- `scales[idx*3+i]`: **linear** scale = a sample of `exp(params["scales"])`. `[N,3]`
- `ratios[idx]` = `n_idx`: the number of copies this Gaussian is split into = `bincount(sampled_idxs)[sampled_idxs] + 1`, clamped on the host with `clamp_(min=1, max=n_max)` and then cast to `int32` (`relocation.py:61–62`).
- `binoms`: an `[n_max, n_max]` LUT, `n_max = 51`.
- `min_opacity = 0.005` (`strategy/mcmc.py:88`).

The new opacity (`RelocationCUDA.cu:57–62` — per the comment, the clamp is applied **before** the scale computation; this is an intentional deviation from the Eq. 9 transmittance invariant for numerical stability at near-zero opacity):

```
n_idx        = ratios[idx]
new_opacity  = 1.0f - powf(1.0f - opacities[idx], 1.0f / n_idx)
new_opacity  = fminf(fmaxf(new_opacity, min_opacity), 1.0f - FLT_EPSILON)
new_opacities[idx] = new_opacity
```

The new scale (`RelocationCUDA.cu:64–78`):

```
denom_sum = 0.0f
for i in 1..=n_idx:
    for k in 0..=(i-1):
        bin_coeff  = binoms[(i - 1) * n_max + k]          // LUT access
        term       = (pow(-1.0f, k) / sqrt(float(k + 1))) * pow(new_opacity, k + 1)
        denom_sum += bin_coeff * term

coeff = opacities[idx] / denom_sum
for i in 0..3:
    new_scales[idx * 3 + i] = coeff * scales[idx * 3 + i]  // the same coefficient on all three axes
```

Written as formulas (Eq. 9):

```
denom_sum = Σ_{i=1..n} Σ_{k=0..i-1}  C(i-1, k) · ((-1)^k / √(k+1)) · ô^{k+1}
ŝ_axis    = (o / denom_sum) · s_axis        (ô = new_opacity, o = old opacity)
```

Post-processing of the kernel results happens on the host (`strategy/ops.py:337–347`): `p[sampled] = logit(new_opacities)`, `p[sampled] = log(new_scales)`, `p[dead] = p[sampled]`, and the optimizer moments get `v[sampled_idxs] = 0`. This orchestration (including the multinomial sampling and the bincount) stays as tensor operations rather than a GPU kernel.

### 3. MCMC Perturb — forward (no backward)

`MCMCPerturbCUDA.cu:27–61`. One Gaussian per thread.

Tensor contract (`MCMCPerturb.cpp:23–52`; float32 and contiguous are enforced for all of them):
- `positions [N,3]` — **modified in place**
- `quats [N,4]` — **wxyz order, pre-activation (unnormalized)** — the kernel normalizes them
- `scales [N,3]` — log-scale (the kernel applies exp)
- `opacities [N]` — logit (the kernel applies sigmoid)
- `noise [N,3]` — standard normal, **generated externally and passed in**
- Scalars: `noise_scale`, `t`, `k`

Constants (`strategy/ops.py:35–36`, `strategy/mcmc.py:83,190`):

```
t = DEFAULT_MCMC_OPACITY_T = 0.005
k = DEFAULT_MCMC_OPACITY_K = 100.0
noise_scale = lr * noise_lr          # noise_lr defaults to 5e5, lr = the "means" learning rate
```

The kernel body (`MCMCPerturbCUDA.cu:39–60`):

```
quat  = make_vec4(quats + idx*4)                    // (w,x,y,z)
scale = exp(make_vec3(scales_log + idx*3))          // component-wise expf

// covariance composition — Utils.cuh:285–311 (covar only, preci=nullptr)
R = quat_to_rotmat(quat)                            // normalizes internally with rsqrt, Utils.cuh:228–251
S = diag(scale.x, scale.y, scale.z)
M = R * S
covar = M * transpose(M)                            // Σ = R S Sᵀ Rᵀ  (symmetric 3x3)

density = sigmoid(opacities_logit[idx])             // 1/(1+expf(-x)), MCMCPerturbCUDA.cu:22–25
w       = sigmoid(-k * (density - t)) * noise_scale // opacity gate: only the dead ones move much
n           = make_vec3(noise + idx*3) * w
transformed = covar * n                             // **multiplies by Σ itself (not a Cholesky/sqrt)**
positions[idx*3 + {0,1,2}] += transformed.{x,y,z}
```

The exact `quat_to_rotmat` expression (`Utils.cuh:228–251`, **column-major** — the "1st col" and so on in the comments are glm columns):

```
inv_norm = rsqrt(x²+y²+z²+w²)                       // rsqrt_parse_safe, Utils.cuh:43–50
(x,y,z,w) *= inv_norm
R = [ 1-2(y²+z²)   2(xy-wz)     2(xz+wy)  ]         // the matrix written out in row notation
    [ 2(xy+wz)     1-2(x²+z²)   2(yz-wx)  ]
    [ 2(xz-wy)     2(yz+wx)     1-2(x²+y²)]
// glm constructor argument order: col0=(1-2(y²+z²), 2(xy+wz), 2(xz-wy)), col1=..., col2=...
```

The PyTorch fallback (`strategy/ops.py:494–511`) is the tensor version of the same formulas: `noise = randn * sigmoid(-k(op - t)) * noise_scale`, `noise = einsum("bij,bj->bi", covars, noise)`, `means += noise`. The choice between the fallback and the fused path is made by the env var `GSPLAT_MCMC_BACKEND` (`ops.py:33–46`).

Invocation conditions (`strategy/mcmc.py:179–193`): unlike relocation, noise injection runs **every step** (until `noise_injection_stop_iter`). relocation and sample_add run when `refine_start_iter(500) < step < refine_stop_iter(25000)` and `step % refine_every(100) == 0`.

---

## Thread and memory configuration (upstream CUDA original)

| Kernel | Grid | Block | Parallel unit | shared mem | atomic |
|---|---|---|---|---|---|
| `adam_kernel` | `ceil(N*D/256)` | 256 | 1 scalar element/thread | 0 (`AdamCUDA.cu:92`) | none |
| `relocation_kernel` | `ceil(N/256)` | 256 | 1 Gaussian/thread | 0 (`RelocationCUDA.cu:99`) | none |
| `mcmc_perturb_positions_kernel` | `ceil(N/256)` | 256 | 1 Gaussian/thread | 0 | none |

- All three kernels are embarrassingly parallel: no inter-thread communication, no barriers, no atomics whatsoever. If `n_elements == 0` the launch is skipped (`AdamCUDA.cu:94–98`, `RelocationCUDA.cu:101–105`, `MCMCPerturbCUDA.cu:74–78`).
- `mcmc_perturb` checks `N ≤ UINT32_MAX` (`MCMCPerturbCUDA.cu:79`).
- Adam uses `AT_DISPATCH_FLOATING_TYPES` (float/double), but the internal registers are float so it is effectively fp32. relocation uses the same dispatch. perturb enforces float32 only.
- Memory access pattern: all coalesced sequential reads and writes. In relocation, every thread repeatedly accesses the same `binoms` LUT (broadcast-friendly).

---

## MSL porting design

### Common policy

- A single code path (common to M4 and A17 Pro): simdgroup 32, no threadgroup memory needed, no atomics needed, no barriers needed.
- Threadgroup size **256** (the same as CUDA, a multiple of 32). Both targets support non-uniform threadgroups → `dispatchThreads` can be used, but keep the boundary guard identical to the upstream CUDA original (parity and safety).
- **fast-math caution (mandatory)**: Metal compiles with fast-math by default.
  - Under fast-math `metal::pow(x, y)` is implemented as `exp2(y*log2(x))`, which gives **NaN for a negative base** → the original's `pow(-1.0f, k)` (`RelocationCUDA.cu:70`) MUST NOT be ported verbatim. Replace it with `float sign = (k & 1) ? -1.0f : 1.0f;` (mathematically identical, and safer than any bit-level match with the CUDA result).
  - For `sqrt/rsqrt/exp/pow` where golden fixture parity is required, use the `metal::precise::` namespace or build the pipeline with `MTLCompileOptions.mathMode = .safe`. Prefer being explicit inside the kernel: `precise::sqrt(v) + eps` (Adam), `precise::rsqrt` (quat normalization), `precise::exp` (scale, sigmoid), and `precise::pow` (relocation's `new_opacity^(k+1)` and `(1-o)^(1/n)`).
- No fp64 → float32 only (the upstream original is effectively fp32 too, so nothing is lost).
- `FLT_EPSILON` (1.1920929e-7f) is defined in `<metal_stdlib>` — use the `1.0f - FLT_EPSILON` clamp as-is.
- Both glm and MSL `float3x3` are **column-major**, and their constructors take columns → the glm constructor arguments of `quat_to_rotmat` can be transcribed 1:1 as `float3x3(float3(c0), float3(c1), float3(c2))`. `covar * n` means the same thing too (matrix–column-vector product).

### Kernel 1: `fused_adam`

```metal
// port of AdamCUDA.cu:34–71
constant bool kHasValidMask [[function_constant(0)]];

kernel void fused_adam(
    device float*        param        [[buffer(0)]],   // [N*D]
    device const float*  param_grad   [[buffer(1)]],   // [N*D]
    device float*        exp_avg      [[buffer(2)]],   // [N*D]
    device float*        exp_avg_sq   [[buffer(3)]],   // [N*D]
    device const uchar*  valid        [[buffer(4), function_constant(kHasValidMask)]], // [N] bool
    constant AdamParams& P            [[buffer(5)]],   // {uint N; uint D; float lr, b1, b2, eps;}
    uint p_idx [[thread_position_in_grid]])
{
    const uint g_idx = p_idx / P.D;
    if (g_idx >= P.N) return;
    if (kHasValidMask && valid[g_idx] == 0) return;

    float g = param_grad[p_idx];
    float m = exp_avg[p_idx];
    float v = exp_avg_sq[p_idx];
    m = P.b1 * m + (1.0f - P.b1) * g;
    v = P.b2 * v + (1.0f - P.b2) * g * g;
    float step = -P.lr * m / (metal::precise::sqrt(v) + P.eps);

    param[p_idx]     += step;
    exp_avg[p_idx]    = m;
    exp_avg_sq[p_idx] = v;
}
```

- Dispatch: `dispatchThreads(MTLSize(N*D,1,1), threadsPerThreadgroup: (256,1,1))`.
- The `valid == nullptr` branch cannot be expressed in MSL (no null buffers), so use **function constant specialization** (the `kHasValidMask=false` pipeline = plain Adam, `true` = SelectiveAdam). Cache both pipelines.
- `valid` is `uchar` (1 byte) — match it to the SXT bool tensor layout.
- Do not add bias correction (the upstream original has none). Do not pass a `step` counter to the kernel either.
- If `N*D == 0`, skip the encoding (on the host).

### Kernel 2: `mcmc_relocation`

```metal
// port of RelocationCUDA.cu:35–79
kernel void mcmc_relocation(
    device const float* opacities     [[buffer(0)]],   // [N] post-sigmoid
    device const float* scales        [[buffer(1)]],   // [N,3] linear (post-exp values)
    device const int*   ratios        [[buffer(2)]],   // [N] pre-clamped to 1..n_max
    constant float*     binoms        [[buffer(3)]],   // [n_max*n_max] row-major LUT
    constant RelocParams& P           [[buffer(4)]],   // {uint N; int n_max; float min_opacity;}
    device float*       new_opacities [[buffer(5)]],   // [N]
    device float*       new_scales    [[buffer(6)]],   // [N,3]
    uint idx [[thread_position_in_grid]])
{
    if (idx >= P.N) return;

    int n_idx = ratios[idx];
    float old_op = opacities[idx];

    float new_opacity = 1.0f - metal::precise::pow(1.0f - old_op, 1.0f / float(n_idx));
    new_opacity = metal::min(metal::max(new_opacity, P.min_opacity), 1.0f - FLT_EPSILON);
    new_opacities[idx] = new_opacity;

    float denom_sum = 0.0f;
    for (int i = 1; i <= n_idx; ++i) {
        for (int k = 0; k <= i - 1; ++k) {
            float bin_coeff = binoms[(i - 1) * P.n_max + k];
            float sign = (k & 1) ? -1.0f : 1.0f;               // replaces pow(-1,k) — avoids the fast-math NaN
            float term = (sign / metal::precise::sqrt(float(k + 1)))
                       * metal::precise::pow(new_opacity, float(k + 1));
            denom_sum += bin_coeff * term;
        }
    }
    float coeff = old_op / denom_sum;
    for (int i = 0; i < 3; ++i)
        new_scales[idx * 3 + i] = coeff * scales[idx * 3 + i];
}
```

- **LUT policy decided**: a host-precomputed LUT, the same as the upstream original. `n_max=51` → `51*51*4B = 10404B` — place it in the `constant` address space to exploit the broadcast cache (the CUDA original uses global, but constant is a strict improvement). Generate the LUT on the CPU as the counterpart of `math.comb` (fill an integer Pascal triangle, then cast to float; C(50,25) ≈ 1.26e14 is within the fp32 range but loses precision — because the upstream original also uses a float LUT, it must be built **as float in the same way** to preserve parity).
- The `ratios` clamp (1..51) and the int cast are **the host's responsibility**, as in the upstream original (`relocation.py:61–62`). Adding a defensive clamp in the kernel is harmless for parity, but the original does not have one.
- `pow(new_opacity, k+1)`: the base is guaranteed to be in (0,1] (after the clamp), so even fast pow produces no NaN, but precise is preferred for precision parity. Keep the accumulation order (descending or ascending k) exactly as in the original (ascending) — the floating-point summation order affects the result.

### Kernel 3: `mcmc_perturb_positions`

```metal
// port of MCMCPerturbCUDA.cu:27–61 + Utils.cuh:228–311
kernel void mcmc_perturb_positions(
    device float*        positions       [[buffer(0)]],  // [N,3] in-place
    device const float*  quats           [[buffer(1)]],  // [N,4] wxyz, unnormalized
    device const float*  scales_log      [[buffer(2)]],  // [N,3] log-scale
    device const float*  opacities_logit [[buffer(3)]],  // [N] logit
    device const float*  noise           [[buffer(4)]],  // [N,3] standard normal (input)
    constant PerturbParams& P            [[buffer(5)]],  // {uint N; float noise_scale, t, k;}
    uint idx [[thread_position_in_grid]])
{
    if (idx >= P.N) return;

    float4 q = float4(quats[idx*4+0], quats[idx*4+1], quats[idx*4+2], quats[idx*4+3]); // (w,x,y,z)
    float3 s = metal::precise::exp(float3(scales_log[idx*3+0], scales_log[idx*3+1], scales_log[idx*3+2]));

    float3x3 R = quat_to_rotmat(q);          // shared helper (shared with the projection kernel, precise::rsqrt normalization)
    float3x3 M = R * float3x3(float3(s.x,0,0), float3(0,s.y,0), float3(0,0,s.z));
    float3x3 covar = M * metal::transpose(M);

    float density = 1.0f / (1.0f + metal::precise::exp(-opacities_logit[idx]));
    float w = (1.0f / (1.0f + metal::precise::exp(P.k * (density - P.t)))) * P.noise_scale;
    // = sigmoid(-k*(density-t)) * noise_scale, MCMCPerturbCUDA.cu:53

    float3 n = float3(noise[idx*3+0], noise[idx*3+1], noise[idx*3+2]) * w;
    float3 transformed = covar * n;          // Σ·n — not a Cholesky

    positions[idx*3+0] += transformed.x;
    positions[idx*3+1] += transformed.y;
    positions[idx*3+2] += transformed.z;
}
```

- `quat_to_rotmat` MUST **share the same helper** with the EWA projection spec (a separate document) — including the normalization (rsqrt), the wxyz order, and the column composition exactly as in Utils.cuh:240–250.
- `covar` is a symmetric 3x3, but the upstream original uses a full mat3 product, so keep it identical (if it is optimized into a 6-element symmetric storage, only after parity verification).

### The noise producer: a counter-based RNG kernel (new design)

On the CUDA side the noise tensor is built with `torch.randn_like` (Philox4x32-10 + Box-Muller, of the cuRAND family) and handed to the kernel. Metal has no built-in device RNG, so it is designed as follows:

```metal
// proposal: philox4x32-10 (or PCG4D), counter-based — no state storage, guarantees parallel reproducibility
kernel void rng_normal_f32(
    device float*        out    [[buffer(0)]],   // [count]
    constant RngParams&  P      [[buffer(1)]],   // {uint64_t seed; uint64_t offset(step counter); uint count;}
    uint tid [[thread_position_in_grid]])
{
    // 4 uniforms per thread → 4 normals via Box-Muller, written to out[tid*4 .. tid*4+3]
    // counter = (tid, offset_lo, offset_hi, 0), key = (seed_lo, seed_hi)
    // philox4x32-10: 10 rounds, multiplier constants 0xD2511F53, 0xCD9E8D57,
    //                key increments (Weyl) 0x9E3779B9, 0xBB67AE85
    // u ∈ (0,1] conversion: the (x >> 9) | 0x3F800000 bit trick, or (x+1) * 2^-32
    // Box-Muller: r = sqrt(-2 ln u1), z0 = r cos(2π u2), z1 = r sin(2π u2)
}
```

Design rationale and rules:
1. **Keep the noise as an input buffer** — keeping the perturb kernel signature 1:1 with CUDA means that injecting the very noise tensor gsplat wrote into the golden fixture allows the perturb kernel to be compared **bit for bit**. The RNG kernel is verified independently by statistical tests (mean, variance, and moments) plus a bit-identical comparison against a CPU reference.
2. Why it must be counter-based (stateless): the thread count changes every step (N grows), and reproducibility comes from `(seed, step, element_index)` alone with no state buffer to manage. Philox4x32-10 is the same algorithm as the curand default, which makes it easy to check against the literature, while PCG4D has a shorter shader implementation — pick one on performance and code size (see the open questions).
3. There is **no need to reproduce torch's Philox sequence bit-identically** (the noise is a stochastic element; distributional agreement is enough for training parity). Verification is split in two: (1) the fixed-noise injection path, and (2) comparing the RNG kernel itself against a CPU reference.
4. Alternative: `MPSMatrixRandomPhilox`/MPSGraph `randomNormal` could be used, but an in-house kernel is preferred for the shared iOS/macOS KernelBackend abstraction and for command-buffer fusion (encoding RNG→perturb into the same command buffer).

### Dispatch and encoding summary

| MSL kernel | Grid | threadgroup | Pipeline variants | Notes |
|---|---|---|---|---|
| `fused_adam` | `N*D` threads | 256 | `kHasValidMask` ×2 | one dispatch per parameter tensor (means, quats, scales, opacities, and sh separately) |
| `mcmc_relocation` | `N_sampled` threads | 256 | 1 | only during refine; N is the number of samples (small) |
| `rng_normal_f32` | `ceil(N*3/4)` threads | 256 | 1 | immediately before perturb, in the same command buffer |
| `mcmc_perturb_positions` | `N` threads | 256 | 1 | every step; positions is in-place, so ordering against other encoders must be guaranteed |

- All four kernels use 0 threadgroup memory, 0 atomics, and 0 `threadgroup_barrier`s.
- perturb does a read-modify-write of positions in place, but each thread touches only its own element, so there is no hazard. However, when encoding Adam step → perturb in that order into the same command buffer, verify Metal's automatic hazard tracking (a manual fence is required if an untracked heap is used).
- The host orchestration (the dead mask, the multinomial sampling, the bincount, the logit/log conversions, and the parameter rearrangement) is tensor operations in the upstream original too, so in Phase 1 it stays as SXT tensor operations on the CPU.

---

## Open questions

1. **Final choice of RNG algorithm**: Philox4x32-10 (identical to curand, with a high ALU cost from its 10 rounds) vs. PCG4D/PCG3D (the shader convention, low cost). Both satisfy the counter-based requirement. Decide after measuring the cost of Philox's 10 rounds under the A17 Pro power profile. Also decide whether the `sin/cos` of Box-Muller may use fast-math (it affects only distribution quality).
2. **Project-wide fast-math policy**: this document marks the parity sites with `precise::`, but the project-wide `mathMode` policy (safe per pipeline vs. selective precise inside kernels) is still undecided. It must be decided together with the golden fixture tolerances (absolute and relative).
3. **fp32 denormal behavior on Apple GPUs**: on Adam's `sqrt(v) + eps` path, if the flush behavior differs between CUDA (FTZ by default) and Apple GPUs when `v` is denormal, the extreme cases can diverge. Put the boundary cases `v ∈ {0, 1e-40, 1e-30}` into a fixture and confirm by measurement.
4. **binoms LUT precision**: `C(50,25) ≈ 1.26e14` cannot be represented exactly in fp32 (it exceeds 24 bits). Because the upstream original uses a float LUT, reproducing the same loss establishes parity, but how often `ratios` actually reaches 51 in practice (usually single digits) and the impact of the loss are unverified.
5. **SelectiveAdam's `state["step"]`**: it is unused because the kernel does no bias correction, but whether to reserve a step counter in the KernelBackend API against the possibility that gsplat adds the correction later.
6. **valid mask encoding**: the contract for reading the SXT bool tensor's byte layout (assumed 1 byte per element) as `uchar` needs to be settled. Switching to a packed bitmask would gain bandwidth but would diverge from the upstream original.
7. **How far to fuse perturb and Adam into one command buffer**: whether to encode `RNG → Adam (five parameter tensors) → perturb` into a single command buffer every step, or to leave it to the training loop structure (the Phase 2 training loop design).
8. **Moving the relocation orchestration to the GPU**: whether to port `torch.multinomial` (which falls back to numpy above 2^24, `ops.py:61–75`) and `bincount` to Metal, or keep them on the CPU. gsplat did not turn them into kernels either, so the proposal is to leave them outside the Phase 1 scope.

---

## Verification notes (adversarial verifier)

On 2026-07-11, against commit `77ab983ffe43420b2131669cb35776b883ca4c3c` (re-confirmed with git, working tree clean), `AdamCUDA.cu` / `Adam.cpp` / `RelocationCUDA.cu` / `relocation.py` / `MCMCPerturbCUDA.cu` / `MCMCPerturb.cpp` / `Utils.cuh` / `Common.h` / `selective_adam.py` / `_wrapper.py` / `strategy/mcmc.py` / `strategy/ops.py` / `examples/simple_trainer.py` were re-read independently and checked exhaustively.

**Items that passed the comparison** (no errors):
- The signs and order of the Adam update expressions, `sqrt(v) + eps` (added outside the sqrt), and the absence of bias correction — all match the original text at `AdamCUDA.cu:61–70`.
- The thread indexing `p_idx / D`, the `g_idx >= N` guard (`AdamCUDA.cu:50–55`), the `valid` skip (`:56–59`), the grid/block of 256 (`:90–91`), and the `n_elements==0` skip (`:94–98`) all match.
- relocation's `powf(1-o, 1/n)` → the clamp order `fminf(fmaxf(·, min_opacity), 1-FLT_EPSILON)` (**before** the scale computation), the LUT index `(i-1)*n_max+k`, the double-loop bounds `i=1..n_idx` and `k=0..i-1`, `(-1)^k/√(k+1)·ô^{k+1}`, and `coeff = o/denom_sum` shared by all three axes — all match the original text at `RelocationCUDA.cu:54–78`. `n_max=51`, `min_opacity=0.005`, the host-side `ratios` clamp (`relocation.py:61–62`), and the float binoms LUT (`mcmc.py:95–100`) match.
- perturb's wxyz order, the `rsqrt` normalization, the glm column-major constructor expansion (`Utils.cuh:228–251` — both the row and column notations in the document were re-derived and match), `Σ = M·Mᵀ, M=R·S` (`Utils.cuh:293–299`), `w = sigmoid(-k(density-t))·noise_scale` (`MCMCPerturbCUDA.cu:52–53`), `covar * n` (not a Cholesky, `:56`), the in-place addition (`:58–60`), the float32 enforcement (`MCMCPerturb.cpp:41–45`), and `N ≤ UINT32_MAX` (`MCMCPerturbCUDA.cu:79`) all match.
- Constants: `t=0.005, k=100.0` (`ops.py:35–36`), `noise_lr=5e5` (`mcmc.py:83`), `noise_scale = lr*noise_lr` (`mcmc.py:190`), the `eps=1e-15/√BS`, `lr·√BS`, and betas expressions (`simple_trainer.py:370–373`), and the visibility mask computation (`simple_trainer.py:1122–1130`) all match.
- Random numbers: there is no curand call inside any kernel — `noise` is generated in advance by `torch.randn_like` (`ops.py:438`) and passed in as an input buffer. The document's "keep the noise as an input" design policy is consistent with the upstream original.
- All four items of the extraction instructions (the Adam formulas/threads/eps/mask, the Eq. 9 LUT approach, the perturb fusion + where the RNG comes from, and the MSL caveats) are present in the document — no coverage gaps.

**Correction 1 — gsplat CUDA is built with `-use_fast_math` by default (a correction to the parity framing).**
`gsplat/cuda/build.py:66` `FAST_MATH = os.getenv("FAST_MATH", "1") == "1"`, `build.py:175` `extra_cuda_cflags += ["-use_fast_math"]`. That is, as shipped, the CUDA kernels themselves are compiled with approximate intrinsics (`powf→__powf`, `expf→__expf`, `--prec-sqrt=false`, `--prec-div=false`, `--ftz=true`). Therefore:
- Statements in the body's MSL design to the effect of "using `precise::` gives bit-identical results to CUDA" (for example the Kernel 2 note "precise is preferred for precision parity") must be read as **bit-identity with the CUDA GPU output cannot be the goal**. The parity target is Phase 0's **C++ CPU golden reference**, and the preference for `precise::` remains valid for that purpose (staying close to the precise CPU reference).
- "CUDA (FTZ by default)" in open question 3 is corrected to mean that, unlike nvcc's own default (`--ftz=false`), **gsplat's build default is fast-math, so FTZ ends up true** (basis: build.py:66,175). If the golden fixture is a CUDA GPU output, its denormals are already flushed values.
- When designing the fixture tolerances, first settle "whether the fixture producer is the CPU reference or a fast-math CUDA GPU" (see the remaining gaps).

**Correction 2 — the unqualified `pow(float, int)` at `RelocationCUDA.cu:70` takes different precision paths on host and device.**
In device compilation the C++ overloads of the CUDA math headers lower it to the float path (equivalent to powf, or `__powf` under fast-math), but **if the same expression is moved into host C++ (the CPU golden reference) and compiled there**, the C++ `<cmath>` promotion rules compute it as `pow(double, double)` (double precision). `pow(-1.0f, k)` is harmless because it is ±1, but `pow(new_opacity, k+1)` becomes a source of ULP differences. Both the CPU reference and MSL MUST be **unified onto the float path** as `powf(x, float(k+1))` / `precise::pow(x, float(k+1))` (the Kernel 2 code in the body is already correct because it uses a float cast — this note warns about the trap when implementing the CPU reference).

**Correction 3 — `state["step"]` in `selective_adam.py` is not "kept"; it is initialized once and then left alone.**
It is merely lazily initialized to `torch.tensor(0.0)` at `selective_adam.py:78` and is never incremented anywhere in `step()` (no incrementing code exists; all of lines 62–101 were checked). Read "keeps `state["step"]` but" in item 3 of the body's overview as "only initializes it and never updates it" — this is fully consistent with the absence of bias correction and reinforces the conclusion that the Metal port need not implement a step counter.

**Correction 4 — the brush reference's line numbers, and a warning that its formulas differ.**
`Tensor::random(..., Distribution::Normal(0.0, 1.0), ...)` is at `crates/brush-train/src/train.rs:378–382` (the body's 376–380 is the range that includes the preceding comment). Also, brush's noise gate **uses different formulas** from gsplat: it weights by `(1 - opacity)^150` passed through `clamp(0,1)` (`train.rs:372–373`) and clamps the noise to `±median_scale` (`train.rs:391`); it is not a sigmoid gate with a covariance rotation structure. brush is valid only as a reference for the fact that the RNG is framework-side; do not use it as a reference for the perturb formulas.

**Remaining gaps (must be settled before implementation starts):**
1. Identify the producer of the Phase 0 golden fixture — if it is a CPU C++ reference output, close parity is achievable through the Metal `precise::` path; if it is a fast-math CUDA GPU output, the tolerances must reflect the `__powf/__expf` error (relative, on the order of 1e-5) and FTZ. This cannot be decided from this document alone.
2. relocation's `denom_sum` summation is a sequential ascending float accumulation in the upstream original — the CPU reference must be implemented in the same order for the fixture to be stable (already mentioned in the body's Kernel 2 note, but it also needs to be written down as a rule for the CPU-side implementation).

### Second independent re-verification (2026-07-11, adversarial verifier re-run)

At commit `77ab983ffe43420b2131669cb35776b883ca4c3c` (re-confirmed with git rev-parse, working tree clean), the entire document including the first-pass verification notes above was again checked exhaustively against the original. Scope of the re-check: `AdamCUDA.cu` (kernel 35–71, launch 73–120, `sqrt(v)+eps` at line 66, the valid skip at lines 56–59), `Adam.cpp` (adam from line 37, the valid dim/size checks), `RelocationCUDA.cu` (kernel 36–79, the clamp at 60–62, the LUT index at 69, `pow(-1.0f,k)` at 70, coeff at 74), `relocation.py` (clamp_/int at 61–62), `MCMCPerturbCUDA.cu` (sigmoid at 22–25, w at 53, `covar*n` at 56, the in-place update at 58–60, N≤uint32 at 79), `MCMCPerturb.cpp` (the float32 enforcement at 41–45), `Utils.cuh` (rsqrt_parse_safe at 43, quat_to_rotmat at 228, the column expansion re-derived, quat_scale_to_covar_preci at 285), `Common.h` (vec/mat at 64–70), `selective_adam.py` (the assert at 70, step never incremented at 78), `_wrapper.py` (adam at 417), `strategy/mcmc.py` (noise_lr at 83, min_opacity at 88, binoms at 95–100, the refine conditions at 159–163, noise_scale=lr*noise_lr at 190), `strategy/ops.py` (T/K at 35–36, the relocate preprocessing at 326–335, the post-processing at 337–347, randn_like at 438, the torch fallback at 494–511), `simple_trainer.py` (lr, eps, and betas at 370–373, visibility at 1122–1130), `build.py` (FAST_MATH at 66, -use_fast_math at 175), and brush `train.rs` (the gate at 372–373, Tensor::random at 378–382, the clamp at 391). **The formulas, signs, indices, clamps, constants, and thread configuration all match. Corrections 1–4 of the first-pass verification notes were also re-confirmed to be consistent with the original.** There are 0 references to curand/philox anywhere in the gsplat CUDA tree (grep) — re-confirming that "the kernels do not generate random numbers".

**Correction 5 — both uniform-conversion comments in the RNG sketch are defective if implemented literally (risk of NaN in Box-Muller).**
The `u ∈ (0,1] conversion: the (x >> 9) | 0x3F800000 bit trick, or (x+1) * 2^-32` comment in the body's "noise producer" code MUST NOT be ported as sketched:
- Reinterpreting `(x >> 9) | 0x3F800000` as a float gives a value in **[1.0, 2.0)**, not (0,1]. Appending the conventional `-1.0f` makes it [0,1), which allows `u1 = 0` → `-inf/NaN` in `sqrt(-2 ln u1)`. If (0,1] is wanted, flip it into the form `2.0f - as_type<float>((x >> 9) | 0x3F800000u)`.
- Computing `(x + 1) * 2^-32` in **uint32 arithmetic** wraps `x+1` to 0 at `x = 0xFFFFFFFF` → `u = 0`. It MUST be computed in floating-point arithmetic as `(float(x) + 1.0f) * 0x1p-32f` to guarantee (0, 1] (the result range including the rounding of float(x) was confirmed to be (0,1]).
Box-Muller's `ln` applies only to u1, so guaranteeing (0,1] on the u1 path is sufficient. When implementing the RNG kernel and writing the CPU reference, lock this conversion down with unit tests (boundary values x ∈ {0, 0xFFFFFFFF}).

**Correction 6 — a minor citation-range fix.** In open question 8, the `torch.multinomial` 2^24 fallback is cited as "ops.py:61–75", but the actual branch (`if num_elements <= 2**24`) is at `ops.py:79` (the function itself starts at 61–62). The content is accurate; only the cited range fails to include the branch line.

Second-pass re-verification conclusion: with Corrections 5–6 above applied, this document is sufficient as the sole reference for the Metal implementation of the three kernels.
