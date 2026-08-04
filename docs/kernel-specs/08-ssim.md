# 08 — SSIM loss forward/backward (fused-ssim semantics)

Phase 3 new-kernel spec. The formulas, conventions, and kernel design for implementing the SSIM
loss (`1 - SSIM`) of the gsplat training path in Metal. Unlike 01–07, this is not a direct port of
the original CUDA code but **an independently designed implementation of the semantic contract
confirmed at the gsplat call site** — the fused-ssim library is not cloned and its code is
untouched (§1 below).

- Golden fixture: `fixtures/ssim_basic` (`tools/gen_fixtures.py:gen_ssim_basic`).
- The generator verifies the backward derivation (§4) numerically against autograd (rtol 1e-10) —
  **these are the authoritative formulas for the Task 2 kernel implementation**.

## 1. Original files and commit (77ab983)

gsplat commit `77ab983ffe43420b2131669cb35776b883ca4c3c`, license Apache-2.0 (derivation notice —
the existing NOTICE/provenance procedure). All paths are under `research/gsplat/`.

| Role | File | Key location |
|---|---|---|
| 1D Gaussian window | `gsplat/losses.py` | lines 84–89 (`_gaussian_kernel_1d`: `exp(-((x-5)²)/(2·1.5²))`, normalized) |
| 2D window (outer product) | `gsplat/losses.py` | lines 92–107 (`create_ssim_window`: `[C,1,11,11]`) |
| SSIM map formula (torch reference) | `gsplat/losses.py` | lines 110–149 (`torch_ssim_loss` — but note the **same-padding fallback**: lines 129–131 `padding=pad`) |
| C1/C2 constants | `gsplat/losses.py` | lines 142–143 (`C1 = 0.01**2`, `C2 = 0.03**2`) |
| Loss entry point | `gsplat/losses.py` | lines 154–201 (`ssim_loss` = `1 - ssim`) |
| **fused_ssim call convention** | `gsplat/losses.py` | lines 184–185: if `window_size == 11 and not img2.requires_grad and not img1.is_cpu`, then `1.0 - fused_ssim(img1, img2, padding="valid")` |
| Training loss combination | `examples/simple_trainer.py` | line 171 (`ssim_lambda: float = 0.2`), lines 952–961 (mask, permute, and `torch.lerp`) |
| fused-ssim library | (not cloned — rahul-goel/fused-ssim, MIT) | Only the public README semantics are referenced: NCHW, only the first image is differentiable, padding ∈ {same (default), valid}, 11×11 σ=1.5 hardcoded |

## 2. Usage conventions decided (consistent with gsplat — Task 1 findings)

1. **Padding = `valid`.** The gsplat CUDA training path (the path that produced the published
   benches) passes `padding="valid"` to fused_ssim explicitly (losses.py:185 — it **overrides**
   fused-ssim's default of same). The SSIM map covers only the positions where the 11×11 window
   fits entirely: `[H-10, W-10]`, and its mean is the ssim scalar.
   Note: gsplat's own torch fallback (`torch_ssim_loss`) uses same padding (lines 129–131), so
   **the two paths disagree even inside gsplat** — the parity target is fixed to the fused_ssim
   path (valid).
2. **Loss combination**: `loss = torch.lerp(l1, 1-ssim, λ) = (1-λ)·l1 + λ·(1-ssim)`,
   `λ = ssim_lambda = 0.2` (simple_trainer.py:171, 961). That is, 0.8·L1 + 0.2·DSSIM.
3. **Layout**: the gsplat call is NCHW — the trainer transposes `[B,H,W,C]` with
   `permute(0,3,1,2)` before passing it (simple_trainer.py:959). splatx keeps the image buffer as
   `[I,H,W,C]` (the render_colors convention, D6) and handles it through kernel indexing — do not
   create a transposed buffer.
4. **Gradients go to img1 (render) only.** In fused-ssim only the first argument is differentiable
   (gsplat guards on `not img2.requires_grad` as well). The v_img2 path is not implemented.
5. **Window and constants**: 11×11 Gaussian σ=1.5 (1D normalization then outer product — the 2D
   sum is 1), C1=0.01²=1e-4, C2=0.03²=9e-4 (assumes a dynamic range of 1). `window_size != 11` is
   a fallback path in gsplat too — fixed at 11.
6. **Input range contract [0,1]**: losses.py asserts only in debug mode. splatx's render_colors has
   no upper clamp so it can exceed 1 (the SH post-op clamps only the lower bound), but the formula
   behaves identically — gsplat trains under the same condition. No host guard is added.
7. **Masks**: when masking, simple_trainer **multiplies both images** by the mask before SSIM
   (lines 952–953). Phase 3 fixes masks=None (the same disposition as 07 Correction 6) — follow
   this convention if it is introduced.

## 3. Formulas in detail — forward

Independent per channel c and per image b (subscripts omitted below). Inputs x = img1 (render),
y = img2 (GT), window `w(u,v) = w1d(u)·w1d(v)`, `(u,v) ∈ [0,10]²`, `Σ w = 1`.

**1D window taps (f32 — hardcoded in the kernel as the single source of truth, identical to the
fixture meta `window_taps_f32`):**

```
w[0]=w[10] = 0.001028380123898387
w[1]=w[9]  = 0.0075987582094967365
w[2]=w[8]  = 0.036000773310661316
w[3]=w[7]  = 0.10936068743467331
w[4]=w[6]  = 0.21300554275512695
w[5]      = 0.26601171493530273
```

(Computed in f64 as `g(i)=exp(-(i-5)²/4.5)`, `w=g/Σg`, then cast to f32. C++/MSL hardcode these
constants verbatim — the same "constant single source of truth" contract as the relocation binoms
LUT.)

**The five moment maps** (valid position p = (py,px), `py ∈ [0,H-11]`, `px ∈ [0,W-11]` — map[p] is
the window **start** p, center p+5):

```
μ1(p)  = Σ_u w(u)·x(p+u)          E11(p) = Σ_u w(u)·x(p+u)²
μ2(p)  = Σ_u w(u)·y(p+u)          E22(p) = Σ_u w(u)·y(p+u)²
                                  E12(p) = Σ_u w(u)·x(p+u)·y(p+u)
σ1²(p) = E11 - μ1²    σ2²(p) = E22 - μ2²    σ12(p) = E12 - μ1·μ2
```

(The variance uses the `E[x²]-μ²` form, the same as gsplat/fused-ssim — in flat regions f32
cancellation can make it slightly negative, but C2 gives the denominator a lower bound so it is
safe. The golden computes the same form in f64.)

**SSIM map and scalar**:

```
A1 = 2·μ1·μ2 + C1     B1 = μ1² + μ2² + C1
A2 = 2·σ12  + C2      B2 = σ1² + σ2² + C2
S(p) = (A1·A2) / (B1·B2)
ssim = (1/M)·Σ_{b,c,p} S(p),   M = I·C·(H-10)·(W-10)
loss = 1 - ssim                     (Trainer combination: 0.8·L1 + 0.2·loss — Task 3)
```

All products and sums are separable: the 2D convolution is exactly equivalent to a 2-pass
horizontal 11-tap → vertical 11-tap (a mathematical identity because w is an outer product — only
the accumulation order differs, and the f32 deviation is absorbed by the tolerance).

## 4. Formulas in detail — analytic backward derivation (authoritative for Task 2)

Derive `v_img1 = dL/dx` for L = 1 - ssim. dL/dS(p) = -1/M (uniform over all p; when the Trainer
multiplies by λ it becomes -λ/M — generalized as the uniform scalar G).

**(1) Partial derivatives at map pixel p** — viewing S as a function of (μ1, σ1², σ12) (μ2 and σ2²
are constants):

```
∂S/∂μ1  = 2·( μ2·A2/(B1·B2) - μ1·S/B1 )
∂S/∂σ1² = -S/B2
∂S/∂σ12 = 2·A1/(B1·B2)
```

(Derivation: ∂S/∂A1 = A2/(B1B2), ∂S/∂B1 = -S/B1, ∂A1/∂μ1 = 2μ2, ∂B1/∂μ1 = 2μ1,
∂A2/∂σ12 = 2, ∂B2/∂σ1² = 1.)

**(2) Raw-moment chain** — because σ1² = E11 - μ1² and σ12 = E12 - μ1·μ2, define **three
coefficient maps** that correct for the terms routed through μ1:

```
Dμ(p)  = ∂S/∂μ1 - 2·μ1·∂S/∂σ1² - μ2·∂S/∂σ12     (∂σ1²/∂μ1 = -2μ1, ∂σ12/∂μ1 = -μ2)
Dσ(p)  = ∂S/∂σ1²
Dc(p)  = ∂S/∂σ12
```

**(3) Scatter to input pixel q** — ∂μ1(p)/∂x(q) = w(q-p), ∂E11(p)/∂x(q) = 2·x(q)·w(q-p),
∂E12(p)/∂x(q) = y(q)·w(q-p) (only when q-p ∈ [0,10]²):

```
v_img1(q) = G·[ (w ⊛ Dμ)(q) + 2·x(q)·(w ⊛ Dσ)(q) + y(q)·(w ⊛ Dc)(q) ],   G = -λ_eff/M
```

Here `(w ⊛ D)(q) = Σ_{p: q-p∈[0,10]²} w(q-p)·D(p)` — the **adjoint of the valid correlation** = a
full convolution over the coefficient map zero-padded by 10 on every side (w is 180°-symmetric, so
correlation = convolution). The output size returns from the map (H-10)×(W-10) to the input H×W,
and **every input pixel receives a gradient** (even boundary pixels fall inside the receptive field
of at least one map pixel — the fixture asserts grad≠0 at every pixel). This scatter conv is also
the same separable 2-pass of 11 taps.

**Numerical verification**: self-check (d) in `gen_ssim_basic` computes exactly the formulas above
(the three coefficient maps + `conv_transpose2d`) in f64 and asserts agreement with the autograd
gradient at rtol 1e-10 / atol 1e-14. It additionally compares central finite differences at 4
coordinates (self-check e). The derivation itself is locked together with the golden.

**Note**: v_img2 (the GT side) exists as a symmetric structure but is unused in training and
unsupported by fused-ssim (§2-4) — excluded from the implementation.

## 5. MSL kernel design

Working name `Ssim.metal`. A new design (not derived — the formulas come from §3/§4, the publicly
known Wang et al. 2004 standard, plus the gsplat call contract). D2 precision policy: the window
taps are hardcoded as the §3 f32 constants (no transcendental functions), accumulation is fp32 in a
fixed tap order (i=0..10), and the divisions use plain float (C1/C2 give the denominators a lower
bound — precise is unnecessary). `exp/pow` are unused, so there is no precise:: site.

### Kernel 1: `ssim_fwd` — 2-pass separable conv + threadgroup cache

- Dispatch: threadgroup (16,16), grid = (ceil((W-10)/16), ceil((H-10)/16), I·C) — **one thread per
  map pixel**, with channel and image on the z axis (independent per channel — §3).
- Work per threadgroup: a 16×16 map tile. The required input apron is (16+10)×(16+10) = 26×26.

```
phase 0  cooperative load: the 26×26 tiles of x and y → TG memory (26·26·2·4B = 5,408B)
         (Correction — Task 2: partial-tile aprons MUST be boundary-masked — with
         grid=ceil((W-10)/16) the right/bottom partial tiles have a 26×26 apron that
         crosses the image boundary (triggered from this 24×24 fixture onward). OOB
         coordinates load 0 — a valid map pixel p consumes only the in-bounds window
         [p,p+10]², so 0/garbage never enters the valid map. See the §9 Task 2 record.)
barrier
phase 1  horizontal 11 taps: the five partial moments over 26 rows × 16 valid columns
         (Σw·x, Σw·y, Σw·x², Σw·y², Σw·xy) → TG memory (26·16·5·4B = 8,320B)
barrier
phase 2  vertical 11 taps: each thread completes the five moments of its own map pixel
         → μ/σ → writes S(p)
         (+ in training mode it also computes and writes the three coefficient maps
         Dμ/Dσ/Dc on the fly per §4-(1,2))
phase 3  reduction: simd_sum(S) → one value per simdgroup → TG sum → write
         partial_sums[tg_index]
         (Correction — Task 2/5 implementation: phase 3 and partial_sums are **not
         implemented**. The actual reduction reads ssim_map back to the host and
         averages it with `ssim_reduce_mean_f64` (f64 summation in ascending index
         order — the single source of truth) — §9 record 3. device-resident (Task 5)
         also keeps the map buffer summed in f64 on the host (loss is a per-step
         readback point) — the disposition of §8 question 1.)
```

- Total TG memory ≈ 13.7KB < 32KB. Work per thread: ~5.3 elements to load, ~1.6 rows' worth in
  phase 1 (26·16/256), and exactly 1 map pixel in phase 2. In partial tiles (right/bottom of the
  grid), threads outside the valid map range are handled as "zero contribution + alive flag"
  (D8-2 — an early return before simd_sum is forbidden).
- Outputs: `ssim_map [I,H',W',C]` (f32, channel-last — the same convention as render_colors; for
  debugging and fixture comparison), plus in training mode `coeff_mu/coeff_sigma/coeff_cross
  [I,H',W',C]` and `partial_sums [n_threadgroups]` (not implemented — see the phase 3 correction).
- **The ssim scalar reduction is forced to be deterministic**: the single source of truth for the
  reduction is one host-side sequential f64 summation, `ssim_reduce_mean_f64` (CpuKernels.h),
  shared by all three paths (CPU, Metal, and device-resident) (for the execution path and how this
  was decided, see the phase 3 correction and §9 record 3 — restated in Phase 3.5 Task 1, which
  cleaned up the partial_sums premise). **Device atomic float accumulation is prohibited** — the
  loss scalar is a host readback point anyway (D7), so host summation is the contract.

### Kernel 2: `ssim_bwd` — coefficient-map scatter (zero-padded full conv)

- Dispatch: threadgroup (16,16), grid = (ceil(W/16), ceil(H/16), I·C) — **one thread per input
  pixel** (the output is v_img1 [I,H,W,C]).
- Inputs: the three coefficient maps stored by fwd, img1/img2, and the uniform scalar G (the host
  computes -λ_eff/M and passes it as a parameter — the λ combination is decided in Task 3).
- The structure is dual to fwd: load the three coefficient maps through a 26×26 apron, but
  **everything outside the map range is 0** (zero-pad — §4-(3)); horizontal pass → TG intermediate
  (26·16·3·4B = 4,992B) → vertical pass → each thread combines `G·(c1 + 2·x(q)·c2 + y(q)·c3)` and
  writes v_img1. x(q)/y(q) are two direct device loads.
- TG memory ≈ the coefficient apron 26·26·3·4B = 8,112B + the intermediate 4,992B ≈ 12.8KB < 32KB.
- No atomics are needed (each pixel is written exclusively), so this is subject to **bit-exact
  determinism** — a two-run memcmp gate is possible.

### Alternatives (adoption deferred)

- **Recomputation**: do not store the coefficient maps and recompute the five moments in bwd — this
  saves memory (the three maps are 3·H'·W'·C·4B; ~12MB per image for the 1237×822 factor-4 garden)
  but performs the five convs twice. fused-ssim's public API also stores intermediates in training
  mode — **the storing approach is adopted**; recomputation is revisited if memory pressure is
  measured (open question 2).
- **2-dispatch separable** (a horizontal kernel + a vertical kernel with an intermediate device
  buffer): the fallback path if the single TG-cache kernel fails or needs debugging. The numerics
  are identical (only the accumulation order differs).

### Host contract (`KernelBackend::ssim_loss` — Task 2)

- Inputs: img1 (render_colors) and img2 (GT) — `[I,H,W,C]` f32, **H ≥ 11 && W ≥ 11 is required**
  (otherwise the map is empty and the mean is undefined — the host throws. gsplat produces NaN in
  this case, and it cannot occur under training conditions). C=3, I=1 is the Phase 3 verification
  range (the indexing expressions stay general — D3).
- Outputs: the ssim scalar (f64 summation) and v_img1 (training mode). The λ scale is owned by the
  Trainer (Task 3): `v_render = 0.8·v_L1 + 0.2·v_DSSIM` — either fold λ into the bwd kernel's G or
  combine afterwards on the host (decided in Task 3; the fixture golden is the pure
  `d(1-ssim)/d(img1)` with no λ applied).

## 6. Padding and boundary conventions — summary

| Item | Convention |
|---|---|
| fwd map domain | valid — `map[py,px]` = window start `(py,px)`, center `(py+5,px+5)`, size `(H-10)×(W-10)` |
| fwd input access | the windows a valid map consumes are always in-bounds (valid — there is no image-padding concept); however the kernel's cooperative apron load goes OOB in partial tiles — boundary-masked to load 0 (§5 correction) |
| Reduction denominator M | `I·C·(H-10)·(W-10)` — **the number of valid map elements** (not the total pixel count) |
| bwd scatter | a zero-padded full conv over the coefficient maps — every input pixel receives a gradient, boundary pixels in proportion to their partial receptive-field overlap |
| Minimum size | H, W ≥ 11 (host guard) |
| same mode | not implemented (semantics used only by the gsplat torch fallback — not a parity target, open question 3) |

## 7. The `ssim_basic` fixture and proposed tolerances

- **Tensors**: inputs `img1` (= the raster_small/bg render_colors cropped to [4:28,4:28], 24×24×3 —
  covers the 16 tile boundary plus partial tiles, and stays within the <2MiB total-fixtures gate)
  and `img2` (a synthetic GT — low-frequency lighting + contrast/brightness shift + seed 37 noise,
  clamped to [0,1]). Outputs `ssim` (scalar [1] = 0.585063) and `ssim_map [1,14,14,3]`. Gradient
  `v_img1 [1,24,24,3]` = d(1-ssim)/d(img1) — **upstream ≡ 1 (the authoritative training direction,
  dump_backward unused, no λ applied)**.
- **Self-checks (generator)**: (a) ssim(a,a)=1 (1e-12), (b) swapping the arguments is bit-identical
  (symmetry), (c) a pure-Python scalar oracle over 3 pixels × 3 channels (a non-vectorized
  reimplementation, 1e-12), (d) the §4 derivation ↔ autograd (rtol 1e-10), (e) central finite
  differences at 4 coordinates, and (f) non-degeneracy (map standard deviation > 0.005, grad≠0 at
  every pixel).
- **Tolerances** (meta.json is the single source of truth — updated from Task 2 measurements, the
  D1/07 procedure. See the §9 Task 2 record):

| Tensor | rtol | atol | rel_l2 | Basis |
|---|---|---|---|---|
| `ssim` (scalar) | 5e-5 | 1e-6 | — | a mean over 588 elements, f64 host summation — the starting value 1e-5 was violated by the M4 measurement of rel 1.33e-5 (the systematic bias of the map error does not cancel in the mean) → 5e-5 (~3.7x headroom) |
| `ssim_map` | 1e-3 | 1e-5 | — | f32 conv accumulation (121 taps, 2-pass) + the error-amplification bound from the denominator floor C2=9e-4 (≈1/B2 ≤ 1.1e3 · moment error ~1e-7) — the starting value 1e-4 was violated by the M4 measurement of max rel 3.93e-4 (exactly the predicted σ=E[x²]−μ² cancellation amplification; the f64 oracle agrees with the golden at rel ≤ 5.1e-8 — the formula is correct) → 1e-3 (~2.6x headroom) |
| `v_img1` | 1e-3 | 1e-6 | 1e-3 | a rational-function partial-derivative chain + the scatter conv — the same grade as the v_colors starting value (bwd has no atomics → it is deterministic, so self-consistency is expected to be bit-exact). The starting value passes — unchanged |

- **Comparison gate**: neither fwd nor bwd uses atomics → two Metal runs are gated on **bit-exact
  determinism** (memcmp), the same as raster fwd. Comparison against the golden uses the meta
  tolerances.

## 8. Open questions (Task 2 carry-over)

1. **Final form of the reduction** — ~~host f64 summation of partial_sums (default) vs. a second
   kernel doing a single TG sum. The loss is a host readback anyway, so the default comes first;
   revisit in device-resident (Task 5)~~
   **Disposition complete (Task 5)**: the device-resident executor also keeps the ssim_map buffer
   summed in f64 on the host (`ssim_reduce_mean_f64`) — the loss coincides with a per-step readback
   point, so a partial_sums kernel reduction is not adopted (the same path as §9 record 3).
2. **Storing the three coefficient maps vs. recomputation** — storing is adopted (§5 Alternatives).
   ~12MB per image in f32 at 1080p class — negligible at factor-4 training resolution. Switch to
   recomputation if memory pressure is measured.
3. **The same padding mode** — add it if matching the gsplat torch fallback becomes necessary (it
   is not a parity target today). Outside the v0.1 scope.
4. **Where λ is combined** — ~~include it in the bwd kernel's G parameter vs. combine afterwards on
   the host. To be decided in Task 3~~
   **Disposition complete (Task 3)**: absorbing it into the kernel's G is adopted — the Trainer
   calls `ssim_bwd(v_loss=λ)` and receives λ·v_DSSIM directly, and the host only sums it with
   `(1-λ)·v_L1` (see the §9 record).
5. **f16 intermediate maps** — storing the coefficient/moment maps as half risks violating D2 (half
   is prohibited on gradient paths), so they stay f32 for now. There is room to revisit the map
   alone (non-gradient) during the performance phase.

## 9. Verification notes

First edition (Phase 3 Task 1). There has been no separate adversarial verifier pass yet — instead
the backward derivation (§4) and the forward formulas (§3) are locked numerically by the six
self-checks in `gen_ssim_basic` (§7), and (d) in particular compares this document's derivation
verbatim against autograd. If you change the derivation, the generator assertions break with it
(document-golden synchronization is enforced). Corrections found during the Task 2 implementation
and measurements are appended to this section.

**Task 2 implementation and measurement record** (`Ssim.metal` + `ssim_fwd_cpu` +
`KernelBackend::ssim_fwd/ssim_bwd`):

1. **Correction to §5 Kernel 1 phase 0 (Important)**: the first edition's "the apron of a valid map
   tile is always in-bounds" is false for partial tiles — the 26×26 apron of a right/bottom partial
   tile crosses the image boundary (in the 24×24 fixture: even the single 14×14 map tile puts apron
   rows/columns 24 and 25 OOB). The kernel implements boundary masking in the cooperative load
   (OOB → load 0). The justification is that 0/garbage never enters the valid map (the consumption
   range [p,p+10]² ≤ H-1/W-1 of a valid map pixel p is always in-bounds), and the
   `ssim_partial_tile_boundary` test demonstrates it through bit-identical invariance under a 40×40
   embedding.
2. **Tolerances updated from measurements (the D1/07 procedure)**: the f64 oracle (same formulas,
   f64 conv) agrees with the golden at rel ≤ 5.1e-8 — the formulas are correct. The f32
   implementation (CPU and Metal agree) reaches map max rel 3.93e-4 and scalar rel 1.33e-5 (M4)
   through σ=E[x²]−μ² cancellation × 1/B2 amplification — exactly the error estimate made in §7.
   meta update: ssim_map rtol 1e-4→1e-3, ssim rtol 1e-5→5e-5 (v_img1 unchanged — the starting value
   passes, 0 golden violations in bwd).
3. **Scalar reduction decided (disposition of open question 1)**: kernel phase 3 (partial_sums
   simd_sum) was not implemented — the map is read back to the host (needed anyway for debugging
   and fixture comparison) and averaged with `ssim_reduce_mean_f64` (sequential f64 summation in
   ascending index order — the single source of truth shared by CPU and Metal). This matches the §5
   default ("host summation is the default") and guarantees determinism. A partial_sums kernel
   reduction is revisited in device-resident (Task 5) together with removing the map readback.
4. **Determinism measured**: both fwd (map + the three coefficient maps) and bwd (v_img1) are
   bit-identical across two Metal runs (the memcmp gate passes). The 14×14×3 sub-map of the 40×40
   embedding is also bit-identical to the standalone 24×24 run.
5. The single source of truth for the interface is `SsimInputs`/`SsimFwdOutputs`/`SsimGradsIn` in
   `KernelBackend.h` — the λ combination (open question 4) is generalized as the uniform scalar
   `SsimGradsIn.v_loss` (golden verification uses v_loss=1; the Trainer decides how λ_eff is passed
   in Task 3).

**Task 3 integration record** (`Trainer::step` — loss combination, consistent with gsplat):

1. **How λ is passed, decided (disposition of open question 4)**: `ssim_bwd(v_loss=λ)` — the
   kernel's G = -λ/M absorbs λ and returns λ·v_DSSIM directly. The host combination is
   `v_render = (1-λ)·v_L1 + λ·v_DSSIM` (the λ factor is already multiplied in, so only the sum
   remains), and the reported loss is `(1-λ)·L1 + λ·(1-ssim)` — a reproduction of gsplat's
   `torch.lerp(l1loss, 1-ssimloss, 0.2)` (simple_trainer.py:961, ssim_lambda=0.2).
2. **Gate**: with `TrainConfig.ssim_weight` (default 0.2) = 0 the SSIM fwd/bwd kernels are bypassed
   entirely — bit-identical to the pure L1 of Phase 2 and earlier (test
   `trainer_ssim_loss_combination_and_gate`). λ ∉ [0,1] throws in the Trainer ctor, and λ > 0 with
   H/W < 11 throws in the §5 validator (extremely small images).
3. **Input conventions**: img1 = render_colors ([1,H,W,3] channel-last as-is — no transpose, D6),
   img2 = GT. The render goes in without a [0,1] clamp — the same condition as the gsplat loss path
   (task-1-report §4). Gradients go to the render only (v_img2 is not implemented — §2-4).
