# 00 — Kernel Specs Index: document map, cross-cutting design decisions, and the Phase 1 breakdown

The top-level design document for Phase 1 (porting gsplat CUDA to Metal MSL). Synthesizing
specs 01–07 and the adversarial verification notes in each of them, it decides (1) a per-document
summary, (2) the cross-cutting design decisions that span all kernels (D1–D10), (3) the
Phase 1a/1b/1c breakdown, and (4) the disposition of the verifier's critical gaps.

- Target commits: gsplat `77ab983ffe43420b2131669cb35776b883ca4c3c` (Apache-2.0),
  brush `3b80985709e2ec04fd6c8622a40e36473647a8e0` (Apache-2.0). Derivation notices: `NOTICE` / `docs/provenance.md`.
- Device facts settled (a premise shared by every document): on both M4 Air and iPhone 15 Pro
  (A17 Pro), device `atomic_float` is native, the simdgroup width is 32 (1:1 with a CUDA warp),
  and threadgroup memory is 32KB. The kernels are designed as a single code path for both.
- Reading rule: **The body of each spec MUST be read merged with the "Verification Notes
  (Corrections 1–N)" at its end.** The corrections are not retroactively reflected in the body.
- Citation rule for the status/history sections: paths under `docs/` other than
  `docs/kernel-specs/…`, `docs/provenance.md`, and `docs/BENCHMARKS.md` (plan/spec ledgers,
  bench/probe records, status summaries) are internal working documents that are **not included
  in the public repository** — those citations are retained as factual history, not as links.
  The public benchmark summary is `docs/BENCHMARKS.md`.

---

## 1. Document map and summaries

| # | Document | Verdict | Kernels | backward | atomic | threadgroup mem |
|---|---|---|---|---|---|---|
| 01 | `01-projection-fwd-bwd.md` | PASS_WITH_CORRECTIONS (6 corrections) | `projection_ewa_3dgs_fused_fwd/bwd` | yes | bwd only (v_means/v_covars/v_quats/v_scales/v_viewmats) | none |
| 02 | `02-spherical-harmonics.md` | PASS (6 reinforcements) | `sh_fwd/sh_bwd` | yes | bwd only (v_coeffs/v_dirs) | none |
| 03 | `03-tile-intersect.md` | PASS (4 supplementary notes) | `isect_tile_count/emit`, the 3 prefix scan kernels, `isect_offset_encode` | no | none | a small amount, scan only |
| 04 | `04-sort-strategy.md` | PASS_WITH_CORRECTIONS (6 corrections) | `radix_sort_{count,reduce,scan,scan_add,scatter}` | no | threadgroup atomic only | ~4.2KB |
| 05 | `05-rasterize-fwd-bwd.md` | PASS_WITH_CORRECTIONS (5 supplements + 1 correction) | `rasterize_to_pixels_3dgs_fwd/bwd` | yes | bwd only (v_means2d/v_conics/v_colors/v_opacities) | fwd 7.2KB / bwd 10.2KB |
| 06 | `06-adam-mcmc.md` | PASS_WITH_CORRECTIONS (6 corrections) | `fused_adam`, `mcmc_relocation`, `mcmc_perturb_positions`, `rng_normal_f32` | no (no_grad) | none | none |
| 07 | `07-fixture-strategy.md` | PASS_WITH_CORRECTIONS (8 corrections) | (not a kernel — fixtures, tolerances, and gen_fixtures extensions) | — | — | — |

### 01 — EWA Projection fwd/bwd (fused)

A map kernel with one thread per Gaussian. fwd: world→cam transform, EWA projection
(PINHOLE/ORTHO/FISHEYE), `add_blur` (eps2d=0.3) + compensation, conic, opacity-aware radius
(GAUSSIAN_EXTEND=3.33), and multi-stage culling. bwd: recompute the intermediates
(rematerialization), then propagate the VJP backward, accumulating with a warp reduction plus
float atomicAdd. Key corrections: the bwd inputs have no `opacities` (Correction 1),
`add_blur_vjp` has exactly one promotion to fp64 (Utils.cuh:486, Correction 2), the pointer-shift
convention (Correction 3), the fisheye fwd/bwd `z+eps` asymmetry is ported exactly as in the
upstream original (Correction 5), and quat normalization uses `rsqrt_parse_safe` with the addition
order x→y→z→w (Correction 6). MSL design: when C==1 the Gaussian gradients are non-atomic stores
(`kSingleCamera`); v_viewmats requires simd_sum plus an alive flag.

### 02 — Spherical Harmonics fwd/bwd

Hardcoded SH evaluation in the Sloan form (deg 0–4). One thread = (batch, gaussian, channel).
dirs takes unnormalized input and normalizes it with an rsqrt inside the kernel (unguarded;
dir=0 → NaN, same as the upstream original). bwd: v_coeffs (basis × v_color, fp32 atomic
accumulation) and v_dir (accumulate the partial derivatives, then one tangent projection). When
B==1 there is no contention on v_coeffs. `+0.5`, clamp, and relu_mask live outside the kernel
(rendering.py). Reinforcement: the fused kernel zero-fills masked-out entries but the standalone
one leaves them unwritten — the upstream original is itself inconsistent here (Reinforcement 1).
Excluding the fused `assemble_proj_features` from the Phase 1 scope is proposed.

### 03 — Tile intersection + offset encoding

A 2-pass structure (count → cumsum → n_isects readback → emit); no backward. Two paths decide
tile membership: AccuTile (SNUGBOX + ellipse-line sweep, needs conics/opacities — the gsplat
production default) and an AABB fallback. The 64-bit key is
`[image_n_bits | tile_n_bits | depth f32 bits 32]`, with `bits_for_count = bit_width(n-1)` and
`image_n_bits + tile_n_bits <= 32`. In `isect_offset_encode` the boundary threads fill the empty
tile ranges (non-overlapping writes, no atomics needed). The prefix sum is replaced by a
reduce-then-scan 3-dispatch. Only the sort contract is stated here; the implementation is
delegated to 04. Supplementary notes: radii is loaded into a float local variable, the host skips
the dispatch when `n_elements==0`/`n_isects==0`, and the depth ≥ 0 invariant (assert at line 283).

### 04 — GPU sort strategy

A replacement for cub::DeviceRadixSort. **OneSweep/decoupled-lookback is forbidden** (Metal does
not guarantee forward progress between threadgroups) → brush's FidelityFX-style reduce-then-scan
LSD radix (4-bit digit, 5 kernels: count→reduce→scan (single workgroup)→scan_add→scatter, with
bin-major counts guaranteeing stability). Adopted strategy: build on a 32-bit stable radix argsort
(two stages or a 2-word split — see D9). Determinism checklist (no atomic slot allocation, slots
derived from the prefix sum, a 100-run hash CI test). Corrections 1–3: 5 barriers in scan, 3 per
scatter split, and splitting out the pre-read barrier for `bin_offset_cache` — **merging the
barrier corrections is mandatory**. Correction 6: the readback-free capacity dispatch is a new
combination brush has never demonstrated (a dedicated test is required).

### 05 — Tile rasterize fwd/bwd

One threadgroup = one tile (TILE=16, CTA=256, PPT=1, CDIM=3). fwd: a cooperative load of a
256-element batch → front-to-back alpha blending, `alpha = min(0.99, opac·exp(-sigma))`, exclusive
early termination at `next_T <= 1e-4`, outputs render_colors/render_alphas/last_ids (an absolute
isect index). bwd: one thread per pixel (16×16 2D), back-to-front, recomputing `T` from `T_final`
by `1/(1-α)` (lower bound 1e-6), and after a warp reduction only lane 0 does the atomicAdd.
Supplement 1: bwd also has a per-image pointer advance (required for I>1). Supplement 3: the
atomic count is per simdgroup — "optimizing" it into a threadgroup reduction turns it into a
barrier inside a divergent loop, which is a forbidden pattern. The one correction (`sg_done` size
0) applies only to the tile=4 variant — irrelevant to Phase 1.

### 06 — Fused Adam + MCMC (Relocation / Perturb)

All three kernels are no_grad and embarrassingly parallel (no barriers, atomics, or shared
memory). Adam: no bias correction, `sqrt(v)+eps` (the addition is outside the sqrt), and when the
valid mask rejects a row everything is skipped, moments included. Relocation (MCMC Eq.9): the
binoms `[51,51]` float LUT is precomputed on the host, and the opacity clamp comes before the
scale computation. Perturb: compose quat→R→Σ and then `Σ·n` (not a Cholesky factor); **noise is an
input buffer** (there is no RNG inside the kernel — given the noise, the kernel is fully
deterministic). Correction 1: gsplat CUDA is built with `-use_fast_math`, so the parity target
must be the CPU golden (D1). Correction 5: both sketches of the RNG uniform conversion, if
implemented literally, admit a Box-Muller NaN — boundary-value unit tests are mandatory.
`pow(-1,k)` is NaN under fast-math → replaced by the sign trick.

### 07 — Fixture strategy

The forward goldens come from `_torch_impl`/`_math`, pure PyTorch (f32 in → f64 math → f32 out);
the backward goldens come from `torch.autograd.grad` with a fixed-seed upstream (saved as and
injected from `v_<name>.sxt`). For the rasterizer, `_rasterize_to_pixels` depends on CUDA/nerfacc,
so **a self-contained `_rasterize_reference` is newly implemented** (Correction 4: the background
is applied in a global stage outside the tile loop — this prevents a wrong golden for empty tiles
with backgrounds). Tolerances: element-wise rtol/atol + an overall rel_l2 + two-run
self-consistency, with the values stored in meta.json (hardcoding is prohibited). Correction 2:
the torch `_isect_tiles` golden covers only the AABB path (there is no AccuTile golden).
Correction 3: proj fwd outputs are compared only under the `valid=(radii>0).all(-1)` mask.
Correction 5: the CUDA contract for covars is [N,6] triu — a covars6/v_covars6 conversion rule is
needed.

---

## 2. Cross-cutting design decisions (D1–D10)

### D1. Parity target decided — the goldens are CPU-produced

- The producer of the Phase 1 golden fixtures is decided to be the **CPU** (torch CPU f64 math →
  f32 storage, plus the Phase 0 C++ CPU reference). They are not CUDA GPU output.
- Rationale: gsplat CUDA is built with `-use_fast_math` by default (`build.py:66,175` →
  `__powf`/`__expf`, FTZ=true), so bit-level or near agreement with a CUDA GPU cannot be a target
  even in principle (06 verification Correction 1). The promotion to fp64 in the
  `v_compensations` chain (Utils.cuh:486) and the host/device difference in `rsqrt_parse_safe`
  (01 verification Corrections 2 and 6) force the same conclusion.
- Every precision choice in the Metal kernels (D2) is therefore judged against "closeness to the
  precise CPU golden". The tolerances start from the table in 07, are stored in meta.json, and
  are updated after measuring on device (the two-run noise floor).

### D2. Precision policy — fp32 only, `metal::precise::` at parity points only

- **fp32 by default, across the whole pipeline.** Do not use half (never for gradients or
  accumulators). The only fp16 candidate is SH coeffs storage (CUDA also allows half for coeffs
  only, upcasting to fp32 immediately on read), and it is deferred past Phase 2. Apple GPUs have
  no device double — the upstream original also does all of its internal math in float32, so
  nothing is lost.
- Compilation keeps the Metal default (fast-math), but at the parity points below the kernels use
  **`metal::precise::` explicitly**:

| Point | CUDA original | MSL | Rationale |
|---|---|---|---|
| quat normalization (proj F2/B7, perturb) | `rsqrt_parse_safe` | `precise::rsqrt` | 01 Correction 6, 06 |
| SH dirs normalization | `rsqrtf` | `precise::rsqrt` | 02 |
| the log in the opacity-aware extend (proj F7 line 180, isect t line 303) | `__logf` | `precise::log` (the CPU reference is unified on `logf`) | 03 open question 1 |
| raster `vis = exp(-sigma)` (fwd/bwd) | `__expf` | `precise::exp` | 05 open question 1, 07 |
| Adam `sqrt(v)+eps` | `sqrt` | `precise::sqrt` | 06 |
| Relocation `pow`/`sqrt`, Perturb `exp` (sigmoid/scale) | `powf`/`expf` | `precise::pow`/`precise::sqrt`/`precise::exp` | 06 |
| Relocation `pow(-1.0f, k)` | `powf` | **forbidden** → `float sign = (k & 1) ? -1.f : 1.f` | 06 (fast-math pow gives NaN for a negative base) |
| float division that feeds an integer output (the isect AABB/AccuTile tile bounds) | `/` (CUDA IEEE) | `precise::divide` | the Metal fast-math approximation flips floor/ceil and thereby changes the tile set itself (measured in 1b Task 3) |

- Everything else (the four arithmetic operations and FMA contraction) may use fast-math — the
  deviation is absorbed by the meta.json tolerances.
- Special case: `v_compensation * 0.5 / (compensation + 1e-6)` in `add_blur_vjp` is the only
  promotion to fp64 in CUDA (01 Correction 2) — MSL computes it in float, and this term is
  reflected separately in the tolerances of the fixtures where `v_compensations` is enabled.
- Honor the fp addition order: the quat norm sum of squares follows the source order x→y→z→w
  (01 Correction 6), and relocation `denom_sum` accumulates sequentially in ascending (i,k) —
  **the CPU reference is implemented in the same order too** (06 residual gap 2).

### D3. Unified buffer layout — SoA flattened arrays, exactly the gsplat contract

- Every device buffer is a **C-contiguous flattened scalar array** (float / int32 / uchar). AoS
  struct buffers are forbidden. Three-element fields with a 12B stride (conics, and the
  xy_opacity/conic layout in threadgroup memory) are handled only through `packed_float3`;
  loading them as `float3` (16B) buffers is forbidden.
- **The order of the five training parameters (this is the single source of truth)**:
  `means [N,3]` → `quats [N,4]` → `scales [N,3]` → `opacities [N]` → `coeffs [N,K,3]`. The Adam
  dispatches, the fixtures, and the API argument order all follow this order.
- **quat is [w,x,y,z]** (gsplat `Utils.cuh:230`). If a PLY/SXT loader uses a different order,
  normalize it to this order in the loader. scales are log-scale and opacities are logits in
  their raw parameter form, and where the activation happens per kernel follows each spec
  (projection takes linear scale / sigmoid opacity, while perturb takes log/logit and applies the
  activation internally).
- **The covars contract is [N,6] upper-triangular, the same as CUDA**: `(c00,c01,c02,c11,c12,c22)`.
  v_covars is 6-triu as well, with the off-diagonals being the sum of the symmetric components
  (`v01+v10`, and so on). The fixtures keep the existing [N,3,3] full form and additionally store
  the converted `covars_triu.sxt` / `v_covars_triu.sxt` (07 Correction 5).
- **A bool tensor is uchar, 1 byte per element (0/1)**. This is the common contract for Adam
  `valid` and the raster/SH `masks` (decided in 06 open question 6). A packed bitmask is
  forbidden — it differs from the upstream original.
- The int64 keys (`isect_ids`) live on device as a single `ulong[n_isects]` buffer (the 03
  contract: treat them as unsigned). SXT does not support int64, so fixtures store them split
  into lo/hi int32 (07). When the GPU sort needs 32-bit word access, reinterpret the ulong buffer
  as a little-endian `uint2` (x=lo=depth bits, y=hi=img|tile).
- Culled-slot convention: on culling, proj fwd writes 0 only to the two `radii` slots;
  means2d, depths, conics, and compensations are left **unwritten (garbage)** — this keeps the
  gsplat contract, and zero-filling them is forbidden. Downstream trusts only radii>0 (isect is
  the gate, and the test harness compares under the valid mask — 07 Correction 3).
- The B (batch) axis is fixed at 1 in Phase 1, and C (cameras) = I (images) = 1 by default. The
  indexing expressions are nevertheless ported exactly as the specs state them (extending B/C
  then needs only a parameter change, no code change).

### D4. MSL naming conventions

- **Kernel function names**: keep the upstream gsplat names, snake_case —
  `projection_ewa_3dgs_fused_fwd/bwd`, `sh_fwd/sh_bwd`, `isect_tile_count/isect_tile_emit`,
  `isect_offset_encode`, `rasterize_to_pixels_3dgs_fwd/bwd`, `fused_adam`, `mcmc_relocation`,
  `mcmc_perturb_positions`. New kernels use `<area>_<action>`: `scan_partial`,
  `scan_blocks_single_group`, `scan_add_offsets`, `radix_sort_count/reduce/scan/scan_add/scatter`,
  `rng_normal_f32`.
- **Buffer/tensor names**: exactly the gsplat kernel argument names (means2d, conics,
  flatten_ids, …). Gradients take the `v_` prefix. Fixture file names are `<argument name>.sxt` /
  `v_<argument name>.sxt`.
- **Parameter structs**: `<area>Params` in PascalCase — `ProjectionParams`, `SHParams`,
  `IsectTileParams`, `IsectOffsetParams`, `RadixSortParams`, `RasterizeParams`, `AdamParams`,
  `RelocParams`, `PerturbParams`, `RngParams`. Defined once in the shared host/device header.
- **Constants**: `k`-prefixed PascalCase — `kAlphaThreshold(1.f/255.f)`, `kGaussianExtend(3.33f)`,
  `kMinCompensation(0.005f)`, `kMaxAlpha(0.99f)`, `kTransmittanceThreshold(1e-4f)`,
  `kMinOneMinusAlpha(1e-6f)`. All of them in one place, `KernelConstants.h`.
- **function constants**: `k<Area><Flag>` — `kProjUseCovars`, `kProjHasOpacities`,
  `kProjCalcComp`, `kProjSingleCamera`, `kProjViewmatsGrad`, `kShHasMasks`, `kShComputeVDirs`,
  `kIsectAccuTile`, `kRastHasBackground`, `kRastHasMasks`, `kRastHasAbsGrad`,
  `kAdamHasValidMask`. Both the name and the index MUST be **unique library-wide** (D5) — an enum
  in the single header `FunctionConstants.h` assigns an index band per area (projection 0–7,
  SH 8–15, isect 16–23, raster 24–31, adam/mcmc 32–39).
- **File layout** (`Sources/SplatXKernelSrc/` — Correction 2026-07-18: the original text said `Sources/SplatXMetal/Kernels/`, which is stale; this is where the training kernels actually live. The latter holds only `Probe.metal`):
  `ProjectionEWA.metal`, `SphericalHarmonics.metal`, `IntersectTile.metal`, `PrefixScan.metal`,
  `RadixSort.metal`, `Rasterize3DGSFwd.metal`, `Rasterize3DGSBwd.metal`, `AdamMCMC.metal`,
  `Rng.metal` plus the shared headers `KernelConstants.h`, `KernelParams.h` (shared with the
  host), `KernelCommon.h` (device helpers), and `FunctionConstants.h`.
- Array-size constants such as CDIM cannot be function constants → they are handled as pipeline
  variants via a preprocessor macro (`-D CDIM=3`) (05). Phase 1 has the single variant CDIM=3.

### D5. No duplicate symbols — the single-metallib link constraint

Every `.metal` source is compiled and linked into one `SplatX.metallib`. The rules:

1. **Define device helper functions only as `static inline` in a header.** In particular
   `quat_to_rotmat` must be a **shared, identical definition** between ProjectionEWA and
   AdamMCMC (perturb) (required by 06), so keep exactly one copy in `KernelCommon.h`. `mat2_inv`,
   `quat_scale_to_covar`, `eval_gaussian_weight`, sigmoid, and the rest go in the same header.
2. Program-scope constants are `constant constexpr` (internal linkage) or macros only — a
   non-constexpr `constant float x = ...` placed in several translation units collides at link
   time.
3. Both the name and the index of a function constant MUST be unique library-wide (the enum bands
   in D4).
4. Kernel function names and parameter struct names MUST be unique library-wide. Each translation
   unit includes the single definition in `KernelParams.h` (ODR agreement).
5. Each `.metal` file header carries the Apache-2.0 derivation notice comment (the original
   gsplat / brush copyright) (05 open question 8 — together with the NOTICE procedure).

### D6. Intermediate buffer specifications between kernels (the pipeline contract)

| Buffer | dtype, shape | Producer | Consumer | Convention |
|---|---|---|---|---|
| `radii` | int32 [C,N,2] | proj fwd | isect count/emit, computing the valid mask, the proj bwd gate | radii≤0 = culled. The other proj outputs in a culled slot are garbage (D3) |
| `means2d` | f32 [C,N,2] | proj fwd | isect, raster fwd/bwd | pixel coordinates (the +0.5 pixel-center convention lives on the raster side) |
| `depths` | f32 [C,N] | proj fwd | isect emit (the low 32 bits of the key) | **the depth ≥ 0 invariant** (guaranteed by the near_plane>0 culling; checked by a debug verification kernel / on the CPU) |
| `conics` | f32 [C,N,3] (stride 12B) | proj fwd | isect (AccuTile), raster fwd/bwd | upper-triangular (a,b,c). `packed_float3` access only |
| `compensations` | f32 [C,N] | proj fwd (optional) | the host AA chain (D10-1) | OFF by default in Phase 1 |
| `colors` | f32 [C,N,3] | sh_fwd + host post-op (+0.5, clamp) | raster fwd/bwd | relu_mask (uchar [C,N,3]) is produced by the post-op and multiplied into v_colors just before sh_bwd |
| `tiles_per_gauss` | int32 [C*N] | isect_tile_count | scan_partial | 0 for culled Gaussians |
| `cum_tiles_per_gauss` | int64(long) [C*N] | the 3 scan kernels | isect_tile_emit; the last element = `n_isects` | **the (only) CPU synchronous readback point** — read directly from the shared buffer |
| `isect_ids` | ulong [n_isects] | isect_tile_emit | sort → isect_offset_encode | `[img_bits\|tile_bits\|depth f32 bits]`, unsigned sort, stable |
| `flatten_ids` | int32 [n_isects] | isect_tile_emit (→ reordered by the sort) | raster fwd/bwd | a global flattened Gaussian index. fwd and bwd **share the same buffer** (the ordering contract, 04) |
| `tile_offsets` | int32 [I,TH,TW] | isect_offset_encode | raster fwd/bwd | the range of tile t is `[off[t], t+1<I*n_tiles ? off[t+1] : n_isects)`. **Whatever the sort strategy, it converges on this contract (D9)** |
| `render_colors/alphas/last_ids` | f32 [I,H,W,3] / f32 [I,H,W,1] / int32 [I,H,W] | raster fwd | loss, raster bwd | last_ids = an absolute isect index (not a Gaussian id); 0 when there is no contribution |
| `v_render_colors/alphas` | f32 | loss (host) | raster bwd | upstream |
| `v_means2d/v_conics/v_colors/v_opacities` | f32 (atomic_float accumulation) | raster bwd | proj bwd / sh_bwd / the AA chain / adam | **MUST be zero-initialized before the dispatch (D8-1)** |
| `v_depths` | f32 [C,N] | (Phase 1: a zero buffer) | proj bwd | depth rendering is unused → 0 |
| `v_means/v_quats/v_scales/(v_viewmats)` | f32 (atomic_float accumulation) | proj bwd | adam | the contribution reduced from sh_bwd's `v_dirs` (`v_means += Σ_C v_dirs`) accumulates into the same v_means buffer — zero-initialize once per step |
| `v_coeffs / v_dirs` | f32 (atomic_float accumulation) | sh_bwd | adam / the v_means reduction | v_coeffs always accumulates in fp32 |
| `valid` | uchar [N] | host: `(radii>0).all(-1)` | fused_adam | the D3 bool contract |
| `noise` | f32 [N,3] | rng_normal_f32 | mcmc_perturb_positions | given the noise input, perturb is fully deterministic (the bit-comparison path for fixtures) |
| `binoms` | f32 [51*51] | a host LUT (integer Pascal's triangle → cast to float) | mcmc_relocation | the `constant` address space; keeps the same float loss as the upstream original |

### D7. Kernel execution order (one training step; I=C=1, B=1, PINHOLE, classic mode)

```
params: means, quats, scales, opacities(logit), coeffs
   │  (host: prepare the activated tensors — exp(scales), sigmoid(opacities))
   ▼
[01] projection_ewa_3dgs_fused_fwd ──► radii, means2d, depths, conics (,compensations)
   │
   ├─► (host: dirs = means − campos, masks = (radii>0).all(-1))
   │   [02] sh_fwd ──► colors_sh ──(host post-op: +0.5, clamp, store relu_mask)──► colors
   │
   ├─► [03] isect_tile_count ─► tiles_per_gauss
   │        └─► [03] scan_partial → scan_blocks_single_group → scan_add_offsets
   │              └─► ★CPU sync point: n_isects readback, allocate isect_ids/flatten_ids
   │   [03] isect_tile_emit ─► isect_ids, flatten_ids
   │        └─► [04] sort (stable; D9) ─► sorted isect_ids, flatten_ids
   │              └─► [03] isect_offset_encode ─► tile_offsets
   ▼
[05] rasterize_to_pixels_3dgs_fwd(means2d, conics, colors, opacities, tile_offsets, flatten_ids)
     ──► render_colors, render_alphas, last_ids
   │
   ▼ loss (host) ─► v_render_colors, v_render_alphas
   ▼ ★zero-initialize every v_* output buffer with a blit fill (D8-1)
[05] rasterize_to_pixels_3dgs_bwd ──► v_means2d, v_conics, v_colors, v_opacities
[02] sh_bwd(v_colors × relu_mask) ──► v_coeffs, v_dirs ──(reduce)──► v_means(+accumulate)
[01] projection_ewa_3dgs_fused_bwd(v_means2d, v_depths=0, v_conics) ──► v_means(+), v_quats, v_scales
   ▼
[06] fused_adam ×5 (valid mask; parameter order per D3)
[06] (refine steps only) mcmc_relocation + host reordering / (every step) rng_normal_f32 → mcmc_perturb_positions
```

- There is one sync point, the n_isects readback (plus the loss readback). Everything else is
  encoded sequentially within a single command buffer.
- forward and backward must share the same `flatten_ids`/`tile_offsets` for the gradients to
  agree (the 04 ordering contract).
- bwd kernel tests take **the golden fwd outputs from the fixture** as input, not the Metal fwd
  outputs (this blocks error propagation, 07).

### D8. Common implementation rules (enforced for every kernel)

1. **Zero-initialize gradient/accumulation buffers**: every bwd dispatch, and v_coeffs/v_dirs of
   sh_bwd, MUST get a blit `fillBuffer(0)` before encoding — this is **built into** the bwd
   encoding helper in MetalBackend so that omitting it is impossible (omitting it produces
   silently wrong results — the 05 verifier's warning).
2. **No early return before a simd reduction**: kernels that execute
   `simd_sum`/`simd_max`/`simd_any` (proj bwd, raster bwd) MUST NOT port CUDA's early return
   as-is; convert it into an **alive flag + zero contribution** structure (01 reduction strategy
   section 2 — compliance is mandatory).
3. **Assert `threadExecutionWidth == 32` when creating a pipeline** — the defensive line for the
   premise that a 2D threadgroup linearizes into contiguous simdgroup assignment (05) and for
   every simd-width assumption.
4. **Use the barrier counts obtained by merging the spec body with the verification-note
   corrections**: 5 per RadixSort scan chunk, 3 per scatter split plus the `bin_offset_cache`
   pre-read barrier, and the histogram zero-initialization / initial-load barriers (04
   Corrections 1–3, second-round reinforcement). threadgroup barriers inside divergent control
   flow are forbidden (the raster processing loop).
5. **No null buffers**: every CUDA nullptr branch is replaced by function constant specialization
   plus a dummy bind.
6. **Dispatch skip**: when `n_elements==0`/`n_isects==0`, the host omits the encoding (03
   supplementary note 3; the 05 and 06 launchers do the same).
7. **Never give in to the temptation to improve the upstream original**: the fisheye `z+eps`
   asymmetry (01 Correction 5), culled slots left unwritten (D3), the absence of Adam bias
   correction (06), and the rest of the original's "inconsistencies" are not fixed. This is
   intentional reproduction — do not "fix" it, because the goldens encode exactly that behavior.
   The only approved deviations: the `t <= 0` guard in isect (disposition #13 in the table below)
   and the `pow(-1,k)` sign trick (D2).

### D9. Sort strategy decided — the contract is sort-agnostic, the implementation is two-stage

The rasterizer's input contract is only `(tile_offsets, flatten_ids)` (D6), so the sort
implementation is replaceable. The decided path:

1. **Correctness path (comes first in Phase 1b)**: emit writes the ulong keys exactly per the
   gsplat contract, and the sort starts as a **CPU `std::stable_sort` (64-bit keys)** — pass the
   **exact integer match** verification against the `isect_ids` (lo/hi) and `flatten_ids`
   fixtures first (07 §4 recommendation).
2. **Production GPU path (later in Phase 1b)**: build a 32-bit stable argsort from the
   brush-style reduce-then-scan radix 5 kernels (04), then integrate it with the strategy adopted
   in 04 (two stages: depth argsort → tile argsort). Verification against the gsplat golden is by
   equivalence of "the tile ranges plus, within a tile, the order (depth_bits asc,
   gaussian_index asc)" (the 04 verification convention; a tied-depth fixture is included).
3. The readback-free capacity dispatch is **deferred** — Phase 1 dispatches at the exact size
   after the n_isects readback, same as the upstream original (the path brush has demonstrated,
   04 Correction 6). The capacity mode is revisited in the performance phase together with
   dedicated tests (`num_keys < capacity`: a partially occupied last workgroup plus entirely
   idle workgroups).
4. Determinism is the contract: global atomic slot allocation is forbidden, slots are derived
   from the prefix sum only. CI has a test that runs the same input 100 times and checks that the
   flatten_ids hash is identical.

### D10. Phase 1 scope exclusion list (synthesized from all documents)

1. **Anti-aliasing mode is OFF by default** (`calc_compensations=false`). When it is on, the
   `opacity ×= compensation` derivative chain is decided to be an **elementwise operation owned
   by the host (the training loop)** — do not put it in the projection or rasterize kernels
   (gsplat owns it in the Python layer too; this resolves 01 open question 1).
2. camera model: **PINHOLE only**. ORTHO/FISHEYE keep their code paths but are excluded from
   verification. When FISHEYE bwd resumes, a finite-difference comparison test is mandatory (the
   01 verifier's recommendation).
3. packed mode, `tile_mask`/sparse intersection, the `gaussian_ids` argument (unused in the
   upstream original as well — removal decided), segmented sort (I>1), B>1 batching, and the
   multi-GPU leftovers (`atomicAdd_system`, key window, block_offset).
4. SH: the fused `assemble_proj_features` is excluded (standalone only). All degree 0–4 device
   functions are ported, but verification covers deg≤3 fwd+bwd and deg4 fwd only (fixture
   `sh_fwd_k25`).
5. rasterize: the tile=4 / CDIM≥32 variants, the PPT>1 variant, and verification of
   `v_means2d_abs` (absgrad) (not included in the v1 fixtures — whether to substitute the
   property test `|v_means2d| ≤ v_means2d_abs` is a Phase 2 decision).
6. Moving the relocation orchestration (multinomial/bincount) onto the GPU — gsplat does it with
   tensor ops as well, so it stays on CPU/SXT.
7. SXT format v2 (int64/uint8 dtypes) — v1 avoids it with the lo/hi split and by not using masks.

---

## 3. Phase 1a / 1b / 1c breakdown (recommendation)

The baseline plan (1a: clearing the carry-overs + the skeleton + projection/SH, 1b:
intersect+sort, 1c: rasterize+Adam+E2E) is **valid** per the dependency analysis. Two adjustments
are recommended.

- **Adjustment 1 — move `fused_adam` forward into 1a**: it is a minimal kernel with no barriers,
  atomics, or shared memory, which makes it the ideal first kernel for validating the
  MetalBackend skeleton (the pipeline cache, function constant specialization, the SXT→MTLBuffer
  harness, and assertAllClose). It also lightens the E2E assembly burden in 1c. (Optional — if
  the controller is worried about 1a's scope bloating, keep the baseline plan.)
- **Adjustment 2 — rasterize kernel development can run in parallel with 1b**: the raster
  fixtures are produced by Python (`_rasterize_reference`), so they do not depend on the Metal
  isect/sort being finished. With two workstreams, run 1b (intersect+sort) and the first half of
  1c (standalone verification of rasterize fwd/bwd) in parallel; only the E2E assembly waits for
  1b.

### Phase 1a — clearing the carry-overs + the MetalBackend skeleton + projection/SH

| Item | Content | Dependencies |
|---|---|---|
| skeleton | Extend `KernelBackend` with the Phase 1 methods, the MetalBackend implementation, the pipeline cache (function constants and `-D CDIM`), the `threadExecutionWidth==32` assert (D8-3), the bwd zero-initialization helper (D8-1), and the four shared headers (D4/D5) | none (MetalContext/GpuTimer already exist) |
| fixture infrastructure | Extend `gen_fixtures.py`: the registry, `dump_backward`, meta.json (tolerances), and reflecting 07 Correction 3 (valid masking) and Correction 5 (covars6 conversion). Test harness `assertAllClose` + two-run self-consistency | none |
| `fused_adam` (Adjustment 1) | `kHasValidMask` ×2 pipelines, a denormal boundary fixture (v ∈ {0,1e-40,1e-30}) | skeleton |
| `projection_ewa_3dgs_fused_fwd/bwd` | PINHOLE, both the covars (6-triu) and quat/scale paths, verify the `kSingleCamera` non-atomic path first (bit-exact determinism) → then the C=2 atomic path. Merge Corrections 1–6 | skeleton, fixtures (`proj_fully_fused` extended + `proj_fused_qs`) |
| `sh_fwd/sh_bwd` | Port deg 0–4; verify deg≤3 fwd+bwd / deg4 fwd. The D=3 hot-path variant comes later (the first implementation is the generic 1:1 atomic one) | skeleton, fixtures (`sh_fwd_bwd_k16`, `sh_fwd_k25`) |
| exit criteria | All proj/SH/adam fixtures pass (meta.json tolerances), and the C=1 non-atomic path is confirmed bit-exact deterministic | |

### Phase 1b — intersect + prefix scan + sort

| Item | Content | Dependencies |
|---|---|---|
| `isect_tile_count/emit` | Implement both the AABB and AccuTile paths (`kIsectAccuTile`), add the `t<=0` guard (disposition #13), dense only | the 1a proj fwd output contract |
| the 3 prefix scan kernels | reduce-then-scan, int32→int64, the n_isects readback sync point (D6) | skeleton |
| `isect_offset_encode` | consumes the ulong keys, 64-bit masks (guarding against int UB) | emit |
| sort (correctness) | Integrate CPU `std::stable_sort` → exact integer match against the `isect_tiles_std` fixture (AABB path) | emit |
| AccuTile golden | A C++ CPU reference based on the formulas in spec 03 plus a dedicated fixture (resolves 07 Correction 2) | emit |
| sort (GPU) | Port the brush 5 kernels (merging the barrier corrections, D8-4), equivalence verification (the 04 convention) + a tied-key fixture + a 100-run determinism hash + a 67M-scale scan chunk boundary test | scan |
| exit criteria | The proj fwd → isect → sort → offset chain passes via fixture chaining, and the GPU sort is equivalent to the CPU sort | |

**Phase 1b status** (exit gate passed — branch `feature/phase1b-intersect-sort`, C++ 55 + Swift 3 all green):

- `isect_tile_count/emit`: Done — both paths, AABB (Task 3) and AccuTile (Task 5), with a CPU reference plus Metal, exact integer match against the goldens and bit-exact determinism, including the `t<=0` guard.
- the 3 prefix scan kernels: Done — Task 4, int32→int64 reduce-then-scan, compared against the CPU plus bit-exact determinism (the n_isects readback sync point is verified with host round-trip chaining — 1b is the correctness stage).
- `isect_offset_encode`: Done — Task 4, consumes the ulong keys with 64-bit masks; CPU and Metal both match the golden exactly.
- sort (correctness): Done — Task 6, the CPU `std::stable_sort` backend matches the `isect_std` sorted golden exactly in integers.
- AccuTile golden: Done — Task 5, a C++ CPU reference plus the `isect_accutile` fixture (resolves 07 Correction 2; includes the oracle rotation and isY cases).
- sort (GPU): Done (2 items remain as follow-ups) — Task 7, the brush 5 kernels ported + stable verification with the tied-key fixture (`isect_ties`) + 23 partial-occupancy cases + CPU equivalence at 3M/1M keys. Remaining: the determinism hash runs 20 times (100 is left as CI headroom), and the 67M-scale multi-chunk scan is unverified locally (open question 6 — stated in the `RadixSort.metal` comments).
- exit criteria: met — the proj fwd → isect (AABB/AccuTile) → scan → sort → offset chain is an **exact integer match** through fixture chaining (`tolerances: {}` for every fixture, 0 changes to the meta.json tolerances), and the GPU radix sort is equivalent to the CPU stable_sort (tie order preserved plus repeat determinism).

### Phase 1c — rasterize + MCMC + the training-loop E2E

| Item | Content | Dependencies |
|---|---|---|
| raster fixtures | Implement `_rasterize_reference` (**07 Correction 4 is mandatory**: the background is applied globally outside the tile loop), `raster_small` (hand-check) / `raster_std` (chaining) / `raster_dense` (2 batch-loop iterations + heavy atomic contention) | the 1a fixture infrastructure (Metal 1b not needed — Adjustment 2) |
| `rasterize_to_pixels_3dgs_fwd` | TILE=16/CTA=256/PPT=1/CDIM=3, `sg_done` aggregation (256/32=8), exclusive early termination, `precise::exp` | skeleton, fixtures |
| `rasterize_to_pixels_3dgs_bwd` | 2D (16,16) threads, alive flag (D8-2), simd reduction + lane0 atomic, per-image pointer advance (05 supplement 1) | fwd, fixtures |
| the 3 MCMC kernels + RNG | `mcmc_relocation` (the LUT in `constant`), `rng_normal_f32` (boundary-value unit tests — 06 Correction 5), `mcmc_perturb_positions` (bit comparison with injected noise) | skeleton |
| training-loop E2E | Assemble in the D7 order: the activated tensors, the SH post-op (+0.5/clamp/relu_mask — host elementwise), loss, the v_* zero-initialization orchestration, the valid mask, and the refine schedule. Smoke-test with `colmap_mini` (confirm the loss descends) | all of 1a+1b+1c |
| exit criteria | The raster fwd/bwd fixtures pass (including the self-consistency gate), and the E2E training step's loss trajectory stays within tolerance of the CPU reference | |

**Phase 1c status** (exit gate passed — branch `feature/phase1c-rasterize-e2e`, C++ 76 + Swift 3 all green; against the 5 DoD items):

1. rasterize fwd/bwd: Done — the four `_rasterize_reference` goldens (small nobg/bg, std, dense) pass within the meta tolerances. fwd is bit-exact deterministic (two runs memcmp identical) with `last_ids` exact (0 mismatches); bwd's two-run self-consistency is at most 3.77e-7 (265× headroom against the 1e-4 gate). Tolerances were adjusted in exactly two places, both v_colors rtol (std 5e-3 / dense 1e-2 — the root cause is the f32 quantization of render_alphas, the bwd input; the measured evidence is in the task-4 report §4).
2. MCMC kernels + RNG: Done — the relocation/perturb goldens pass (fused_adam came earlier, in 1a); with injected noise, two Metal runs of perturb are bit-identical; Philox4x32-10 is bit-identical between CPU and Metal at the u32 stage (memcmp 0), plus 3 sets of Random123 KATs and 0 NaN/Inf at the boundary values (x∈{0,0xFFFFFFFF}).
3. Training on the real garden data: Done — `splatx train` on garden images_8 (185 cameras) ran 300 iterations with loss 0.1679→0.0577 (34% of the starting value), producing a 7.1 MB PLY (30k splats) plus a preview PNG (a manual smoke test, recorded in the task-7 report §4).
4. Automated self-reconstruction smoke test: Done — the `trainer_self_reconstruction_loss_halves` CI test (100 iterations, tail5 = 3.8% of loss[0] — far beyond the "at most half" criterion). The table's "compare the loss trajectory against the CPU reference" is replaced by the self-reconstruction CI plus the garden measurement, because raster bwd is Metal-only (there is no backward CPU reference — the additional decision in §6).
5. License/provenance: maintained — rasterize and MCMC carry the gsplat derivation notice (file header + NOTICE + ledger), while Rng is written from scratch (citing Salmon et al. SC'11 in the literature; not derived from their code). Phase 2 (integrating the MCMC strategy → v0.1) can start.

### Phase 2 — integrating the MCMC strategy + the v0.1 preliminary gate: status

(Exit gate passed — branch `feature/phase2-mcmc-v01`, C++ 89 + Swift 3 all green,
**on both the default path and the `.metallib` path** — plan: `docs/superpowers/plans/2026-07-11-phase2-mcmc-v01.md`,
bench of record: `docs/bench/2026-07-11-garden-7k-bench.md`. Against the 5 DoD items:)

1. MCMC runs to completion on garden and improves PSNR over off: Done — 7000 iterations ran to
   completion (growth reached the 1M cap at iteration 1219, 248 refines, 464,946 relocations,
   900,000 adds), test PSNR **25.297 dB** vs 23.103 dB with MCMC off = **+2.194 dB**
   (the bench table in `docs/STATUS.md` plus the previews in `docs/bench/`).
2. Adam state surgery verified by tests: Done — Task 3 `McmcStrategyTests` (the moments of the
   slots targeted by relocation/add are reset to 0, compared against the gsplat ops.py
   state-surgery convention) plus the mcmc-on self-reconstruction CI.
3. All tests pass on the precompiled .metallib path: Done — `tools/build_metallib.sh`
   (relocatable, keeping the function constants unspecialized) plus `SPLATX_TEST_METALLIB`
   injection drives all 89 tests in the suite through metallib tier (a) (22 kernel entries + every
   specialization). An explicit path that fails throws — no fallback.
4. The PSNR evaluation system + the garden 7K preliminary bench: Done — psnr and the train/test
   split (identical to the gsplat definition and the MipNeRF360 convention, with a hand-check
   test) plus the `--eval-every` trajectory; the bench is recorded as a comparison table against
   gsplat's published numbers (-1.00 dB, with the differing conditions stated — the v0.1 DoD
   "-0.5 dB" gets its final verdict after the Phase 3 SSIM work).
5. relu_mask gradient semantics consistent: Done — Task 1, the SH post-op clamp mask is applied
   on both paths, v_colors → sh_bwd (v_coeffs and v_dirs → v_means), with a dedicated
   channel-blocking test.

**Phase 3 handover list** (the quality boosters — design §6 Phase 3):

1. **fused SSIM loss** (0.8·L1 + 0.2·DSSIM) — **resolving the leading suspected cause of the
   -1.00 dB garden gap is the top priority** (`docs/bench/2026-07-11-garden-7k-bench.md` §3,
   candidate #1). The final verdict on the v0.1 DoD comes after this measurement.
2. **anti-aliasing** (the antialiased mode — the compensations × opacity derivative chain; the
   D10-1 decision that the host owns it stands).
3. **Evaluate a DashGaussian-style resolution schedule** (reimplemented from the paper — the
   no-code-contact rule).
4. **The device-resident train_step performance task** (carried over from D11/§7) — the target is
   to resolve host round-trip chaining at 364 ms/iter@N=1M (plus a GT decode every iteration).
   The per-kernel methods are retained as the verification surface.
5. **The design §5.4 scale clamp footnote** — recorded: confirmed that a maximum-scale clamp on
   export does not exist in the gsplat standard ply path, so it is **omitted** (non-finite values
   are removed by the NaN/Inf filter, and finite huge scales are preserved — reflected in the
   design document's §5.4 footnote and the `PlyIO.h` header comment; details in the local SDD
   artifact task-5-report §1). "No clamp" is pinned by a test assertion
   — if one is ever introduced, that assertion is the signal.
6. **Result of re-examining the adam valid mask — decided to keep the current behavior**: splatx's
   fused_adam skips the moment and reg updates for culled slots (radii==0) (the 1c D3 convention —
   an intentional deviation from the all-rows update of gsplat's stock torch.optim.Adam,
   task-4-report §1). The Task 6 PSNR gate passed (+2.19 dB, within the range explainable by the
   differing conditions against gsplat), so the trigger condition for a re-examination is not met
   — keep the current behavior. Resume if the opacity distribution looks anomalous in the full
   30K run bench (Phase 3+).

### Phase 3 — the quality boosters + the final v0.1 verdict: status

(Exit gate passed — branch `feature/phase3-quality-v01`, C++ **102** + Swift 3 all green,
on both the default path and the `.metallib` path — plan:
`docs/superpowers/plans/2026-07-11-phase3-quality-v01.md`, bench of record:
`docs/bench/2026-07-12-v01-final-bench.md`. Against the handover list above:)

1. **fused SSIM loss**: Done — a new spec `08-ssim.md` (designed in-house — fused-ssim was not
   cloned; consistent with the gsplat call contract `padding="valid"`) plus `Ssim.metal` fwd/bwd
   (analytic derivation in §4, f64 oracle rel ≤ 5.1e-8, bit-exact determinism), combined in the
   Trainer as `(1-λ)·L1 + λ·(1-SSIM)` (λ defaults to 0.2 — reproducing gsplat's `ssim_lambda` and
   `torch.lerp`; at λ=0 the SSIM path is bypassed bit-exactly to pure L1). Measured on garden:
   +0.241 dB (25.297→25.538 — the comparison also mixes in the host→resident switch; resident
   equivalence is guaranteed by the fixtures).
2. **anti-aliasing**: Done — the 1a `kProjCalcComp` / v_compensations path is now actually used;
   opacity×comp stays host-owned (D10-1) and is multiplied **before isect** (consistent with
   gsplat rendering.py); with aa off the result is bit-identical to the previous behavior. The
   separate AA arm measured -0.239 dB — it is not a device for raising PSNR at training
   resolution (bench of record §3).
3. **device-resident train_step**: Done — the §7 D11 carry-over is resolved. A resident-buffer
   step executor (`MetalTrainStepExecutor` + the 13 elementwise kernels in `TrainStep.metal`),
   host↔resident parameter rel_l2 ~1e-6 and an exact match on MCMC refine (including the
   sh_degree≥1 and AA variants), 335→91 ms/iter (3.7×, on the common basis of N=1M with the GT
   cache; the bench steady state is 76.8 ms). The host round-trip path is retained as the
   `--no-resident` verification surface (the D11 contract — it is the reference path for fixture
   parity).
4. **Evaluating the DashGaussian resolution schedule**: Not started — excluded from the Phase 3 plan's scope and carried over as a v0.2 candidate. (Correction 2026-07-18: it was later fully implemented and verified but then **discarded** — three measurements on garden 7k showed no point where the speed and quality gates coexist; the branch was deleted without merging and quarantined under the `archive/res-schedule` tag. See the top of STATUS. It is no longer a v0.2 candidate.)
5. **Final v0.1 verdict**: 5 of the 6 DoD items pass, and ③ (-0.5 dB) **partially misses**
   (bicycle -0.175 ✓ / garden -0.762 ✗) — by the plan's rule this counts as a miss, so the
   **v0.1.0 tag is deferred**. The leading candidates for the garden gap are the difference in N
   (4.48M published vs the 1M cap) and schedule compression — the follow-up (Phase 3.5 / v0.2)
   separates the factors with a cap 2M arm plus a full 30K run bench (bench of record §4).

**Phase 3.5 — re-judging v0.1**: Complete — measurements from the cap 2M arm and the full 30K run
confirmed that N is the main factor in the garden gap (+0.477@7K and +0.457@30K, with schedule
compression costing 0), and DoD ③ was re-judged as **a partial miss at cap 1M / a pass on every
scene at cap 2M** (C++ 103 green); the user's decision on 2026-07-12 (option A) produced the
**v0.1.0 tag** — the option table is in the document of record
`docs/bench/2026-07-12-phase3.5-regate.md` §3, and the current status is in `docs/STATUS.md`.

**SPZ export/import**: Complete — vendoring Niantic spz v2.1.0 (`ThirdParty/spz`, pinned to
SPZ v3) + `save_spz`/`load_spz` (RDF symmetry and a finite clamp on saturated opacity; `SpzIO.h`
is the single source of truth) + automatic detection of `--out *.spz` / `splatx convert`;
measured 11.28× on garden with a ~34 dB round-trip
(`docs/bench/2026-07-12-spz-roundtrip.md`), C++ **110** on both the default and metallib paths +
Swift 3 green.

**Phase 6, round 1 (iOS packaging + on-device measurement)**: Complete — the iOS metallib
(`--ios`), the XCFramework, and the SplatXBench app carried the full suite on A17 Pro at
**109/109 with 0 recalibrations** (confirming the numerical behavior of the single-code design —
`docs/probes/2026-07-12-a17-full-suite.md`) plus an iPhone smoke run to completion at 24.048 dB
(`docs/bench/2026-07-12-iphone-smoke.md`), C++ **111** on both paths + Swift 3 green — the
round-2 handover (memory governance, pause-and-cool, checkpoints, TrainerBackend, and so on) is
in `docs/STATUS.md`.

**Phase 6, round 2 (governance, pause-and-cool, checkpoints, trainer API)**: Complete — KNN grid
init (A17 187.3 s→1.075 s, 174×), auto-cap (§7.2), pause-and-cool (§7.4), fp16 checkpoints
(§7.5), and SplatXTrainer (1:1 with TrainerBackend); the 3-run checklist ①–⑤ on iPhone 15 Pro is
all ✓, and a real thermal pause of 90.6 s with automatic resume resolved the "thermal pause"
clause of the exit criteria (document of record `docs/bench/2026-07-12-iphone-stage2.md`),
C++ **126** on both paths + Swift **7** green —
next: integrating the splatx-ios NativeBackend (internal integration note, not in the public repository).

**→ Integration complete (2026-07-13~14)**: the COLMAP txt loader (`16eceb6`) and the factor
fallback (`6594fc4`) were merged into the core, splatx-ios got the NativeBackend and switched its
default backend to native, the bake-off document of record is
(`docs/bench/2026-07-14-splatx-ios-bakeoff.md`), and iOS↔macOS platform parity is settled at
0.07–0.28 dB. After that: pose-opt (off by default), the thermal threshold fix
(serious→critical), the macOS app, and `splatx render`.
The current summary is at the top of `docs/STATUS.md`.

**Bilateral Grid (2026-07-15)**: Complete — **BilateralSlice fwd/bwd** was added to the covered
kernels (`Sources/SplatXKernelSrc/BilateralSlice.metal` — derived from gsplat
examples/lib_bilagrid.py@77ab983, "Bilateral Guided Radiance Field Processing", Wang et al.
SIGGRAPH 2024): fwd has a golden plus bit-exact determinism, bwd has an autograd golden plus
atomic-scatter self-consistency, an identity grid is a passthrough, and
**off is bit-identical to the previous behavior** (parity on both the host and resident paths).
A per-view 3×4 color affine grid is integrated between render and loss (off by default; PLY/SPZ
export stays appearance-neutral and does not bake in the grid), design document of record
`docs/superpowers/specs/2026-07-15-bilateral-grid-design.md` — C++ **162** green on both paths,
bench of record `docs/bench/2026-07-15-bilateral-grid.md` (positioned as an opt-in only for
captures with exposure drift; follow-up = color_correct eval).

---

## 4. Disposition table for the verifier's critical gaps

| # | Source | Gap | Disposition |
|---|---|---|---|
| 1 | 01 | Ownership of the AA opacity×compensation chain | **Decided**: owned by the host (the training loop) as an elementwise op; Phase 1 defaults to classic mode (D10-1) |
| 2 | 01 | Bit-identical v_compensations is impossible in principle (promotion to fp64) | **Decided**: the parity target is the CPU golden (D1), and a tolerance specific to this term is registered in meta.json (D2) |
| 3 | 01 | Risk of a typo when porting FISHEYE bwd | **Decided**: Phase 1 is PINHOLE only; when FISHEYE resumes a finite-difference test is mandatory (D10-2) |
| 4 | 01 | Divergent simd_sum is undefined | **Decided**: the alive-flag rule is enforced for every kernel (D8-2) |
| 5 | 02 | Semantics of masked-out fwd outputs | **Decided**: keep the standalone contract (unwritten), zero-filling is forbidden, and the harness compares under the mask. The v1 fixtures use masks=None |
| 6 | 02 | A guard for rsqrt(0) at dir==0 | **Decided**: unguarded (same as the upstream original); the fixtures guarantee norm≥0.5 |
| 7 | 02 | Porting scope for degree 4 | **Decided**: the device functions cover all of deg 0–4, verification covers deg≤3 fwd+bwd / deg4 fwd (D10-4) |
| 8 | 02 | Whether to include the fused assemble | **Decided**: excluded, standalone SH only (D10-4) |
| 9 | 02 | Where the SHIFT_RELU relu_mask goes | **Decided**: a host post-op outside the kernel — store relu_mask, then multiply it in just before sh_bwd (D6/D7) |
| 10 | 03 | The sort spec needs its own document | **Resolved**: 04 is written, and D9 settles the integration |
| 11 | 03 | Precision policy for `__logf` | **Decided**: `precise::log` with the CPU unified on `logf`, targeting an exact match of the tile set (if boundary flake appears, relax the policy in meta.json) (D2) |
| 12 | 03 | The n_isects readback sync point | **Decided**: keep the same sync point as the upstream original; the capacity mode is deferred (D9-3) |
| 13 | 03 | A guard for `t<0 → NaN` | **Decided**: add a `t<=0` early return (count 0) — the result for valid inputs is unchanged, and it is recorded as an approved deviation (D8-7) |
| 14 | 03 | Scope of packed/tile_mask | **Decided**: excluded from Phase 1; dense plus both tile paths (D10-3) |
| 15 | 04 | The capacity dispatch is unproven | **Decided**: Phase 1 uses the readback plus an exact-size dispatch (the demonstrated path). If capacity is adopted, a dedicated partial-occupancy test is required (D9-3) |
| 16 | 04 | Barrier placement is incomplete (from the body alone) | **Decided**: merging Corrections 1–3 plus the second-round reinforcement is mandatory (D8-4) |
| 17 | 04 | segmented sort (I>1) | **Deferred**: Phase 1 has I=1. Carried over as a redesign item if batch training is introduced |
| 18 | 04 | Tile-local alternatives B/C are unverified | **Deferred**: reviewed in the performance phase, per the A/B bench plan |
| 19 | 05 | The sg_done correction applies only to tile=4 | **Confirmed**: Phase 1 (TILE=16) proceeds exactly as the body states |
| 20 | 05 | Contiguous simdgroup assignment is a measured premise | **Decided**: assert `threadExecutionWidth==32` (D8-3) |
| 21 | 05 | exp precision and the atomic tolerance are undecided | **Resolved**: the 07 tolerance table + meta.json + two-run self-consistency. `precise::exp` (D2) |
| 22 | 05 | The premise that atomic buffers are zero-initialized | **Decided**: built into the encoding helper (D8-1) |
| 23 | 06 | The producer of the fixtures was undecided | **Decided**: the CPU golden (D1) — this decision is the premise for all of D2 |
| 24 | 06 | The RNG algorithm was undecided | **Proposed**: Philox4x32-10 (the curand family, easy to check against the literature). A CPU reference plus boundary-value unit tests (x∈{0,0xFFFFFFFF}) are mandatory before the perturb E2E (06 Correction 5). Replacing it with PCG4D is considered after performance measurements — **awaiting the controller's final approval** |
| 25 | 06 | Byte layout of the valid mask | **Decided**: uchar, 1 byte per element (D3) |
| 26 | 06 | Summation order of the relocation denom_sum | **Decided**: sequential accumulation in ascending (i,k) is written into the CPU reference implementation rules (D2) |
| 27 | 07 | No AccuTile golden exists | **Decided**: implement both paths — AABB uses the `_isect_tiles` golden, and AccuTile gets its own C++ CPU reference plus a dedicated fixture in 1b (matching the scope of spec 03) |
| 28 | 07 | The background bug in `_rasterize_reference` | **Decided**: Correction 4 (applying the background globally) MUST be reflected — it is the first check item in the 1c fixture work |
| 29 | 07 | The valid-masking harness for proj fwd | **Decided**: comparison under the `valid=(radii>0).all(-1)` mask MUST be implemented (1a) |
| 30 | 07 | The covars [N,6] triu conversion | **Decided**: adopt the covars6/v_covars6 conversion rule; the existing [N,3,3] fixtures stay for Phase 0 compatibility (D3) |

## 5. Items awaiting controller approval (what this document does not decide)

1. **Final RNG selection** (disposition #24) — Philox4x32-10 is proposed; absent objection it is
   decided when 1c starts.
2. **The procedure for correcting tolerances from measurements** — when meta.json is updated and
   who is responsible (07 open question 8). Proposal: measure the two-run noise floor at each
   Phase exit gate → open an update PR only when there is a violation.
3. **Whether to adopt Adjustment 1 (moving fused_adam forward into 1a)** — recommended, but
   keeping the baseline plan is acceptable.
4. **The NOTICE derivation-notice procedure** (05 open question 8) — the .metal file headers plus
   when NOTICE is updated.
5. **The boundary-flake policy** (07 open question 5 — radii ceil / last_ids near the threshold) —
   start with exact comparison; whether to introduce the `next_T ∈ threshold±ε` exclusion rule
   when flake occurs is decided after measurement.

---

## 6. Controller decisions (2026-07-11 — disposition of the §5 items awaiting approval)

1. **RNG = Philox4x32-10 approved.** The CPU reference plus the boundary-value unit tests (x∈{0,0xFFFFFFFF}) remain mandatory before the perturb E2E. PCG4D may be revisited after performance measurements.
2. **The tolerance-correction procedure is approved**: measure the two-run noise floor at each Phase exit gate → commit a meta.json update only when there is a violation. Record the reason for the update in the commit message.
3. **Adjustment 1 is adopted**: move `fused_adam` forward into Phase 1a — it is used as the first kernel for validating the MetalBackend skeleton.
4. **The NOTICE procedure is decided**: for gsplat-derived .metal files, the existing notice header format plus an entry in the provenance ledger is sufficient (NOTICE already covers gsplat derivation as a whole). In Phase 1b, where the first brush-derived file (RadixSort.metal) appears, add a brush entry to NOTICE once.
5. **The boundary-flake policy is approved**: start with exact comparison, and introduce the relaxation rule (excluding threshold±ε) only when flake is actually measured.

Additional decision: backward kernels are verified by comparing directly against the autograd golden fixtures (the per-kernel C++ CPU reference is mandatory for forward only — a backward CPU reference is added optionally, when debugging needs it).

---

## 7. D11 — Trainer chaining strategy (**decided** — at the start of 1c, 2026-07-11)

**Decided**: the Phase 1c Trainer (the training-loop E2E) is assembled with **host round-trip
chaining** — it calls the current `KernelBackend`'s per-kernel methods (host vector in and out)
in the D7 order. The exit criteria for 1c are correctness (golden/CPU parity, a descending loss);
the per-step round-trip cost is a non-goal. Rationale: for verification purposes the host
round-trip is the right thing — the fixture comparison happens in host memory, and error
propagation between kernels is blocked (D7). 1b passed its exit gate the same way, which
demonstrates it.

**The device-resident single-command-buffer `train_step` is carried over to the Phase 2
performance task.** (**Carry-over resolved — Phase 3 Task 5**: all three requirements below are
implemented and verified — see §3 "Phase 3 … status", item 3. The per-kernel methods and the
host round-trip path are retained as the verification surface, as announced.)
The requirements as specified at the time (kept as originally written):

- **device-resident buffer handles**: separate upload and download at explicit boundaries (no
  round-trip per call).
- **Chaining between kernels**: encode proj fwd → isect → sort → raster into a single or a few
  command buffers with no host round-trip (the sync points are the one n_isects readback plus
  loss — D7).
- **v_means accumulation (D6)**: an API where the sh_bwd reduction and proj bwd accumulate into
  the same v_means buffer (zero-initialized once per step — consistent with D8-1).

**The per-kernel methods are retained permanently as the verification surface** — the
device-resident API does not replace them but is added alongside, and the reference path for
fixture and parity tests remains the host vector path even after Phase 2.

---

## 8. Retroactive corrections (not reflected in the body — the footnote is newer)

**Correction 1 (2026-07-17)**: the two features described by pose-opt (off by default) at the end
of "→ Integration complete" in the §3 completion history and by the
"Bilateral Grid (2026-07-15)" block **were removed in the 2026-07-17 sweep of unproven features**
(a measured verdict — bilateral removal `67b81b7`, pose-opt removal `49902d0`). Those blocks are
preserved as a factual record up to the point of removal — the BilateralSlice fwd/bwd kernels,
their fixtures, and the off = bit-identical parity gate died with the code, while the projection
bwd `v_viewmats` golden and the standalone host proj bwd API are retained as the verification
surface. For the basis of the verdict (the bench trilogy and the pose-bottleneck probe), the
replacement (`splatx refine-poses`), and the gate numbers, see the removal block at the top of
`docs/STATUS.md`.

> Correction (2026-07-18): `splatx refine-poses` was removed to focus purely on training (the
> block at the top of STATUS). This cross-reference is historical; the portable design is in
> `docs/refine-poses-portable-design.md`.
