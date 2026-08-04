# 07 — Phase 1 fixture strategy (forward/backward golden generation, tolerances, gen_fixtures extension)

The golden fixture generation strategy for verifying on device the Metal kernels ported in Phase 1
(SH, EWA projection fused, isect_tiles, isect_offset_encode, rasterize_to_pixels fwd/bwd).
It extends the Phase 0 pipeline (loading the gsplat `_torch_impl` stub → SXT tensors → comparison in
the test harness) unchanged.

## Original files and commit (77ab983)

Golden sources (Apache-2.0, reading and porting permitted, a derivation notice is required):

| File | Role |
|---|---|
| `research/gsplat/gsplat/cuda/_torch_impl.py` | The forward golden reference implementation (pure PyTorch) |
| `research/gsplat/gsplat/cuda/_math.py` | Math reference such as `_quat_scale_to_covar_preci`, plus the custom `SafeNormalize` backward |
| `research/gsplat/gsplat/cuda/_constants.py` | `MAX_ALPHA=0.99`, `MIN_COMPENSATION=0.005`, `ALPHA_THRESHOLD=1/255`, `TRANSMITTANCE_THRESHOLD=1e-4` |
| `research/gsplat/gsplat/cuda/include/Common.h:97-114` | The same constants on the CUDA side: `ALPHA_THRESHOLD (1.f/255.f)`, `GAUSSIAN_EXTEND 3.33f`, `MAX_ALPHA 0.99f`, `TRANSMITTANCE_THRESHOLD 1e-4f`, `MIN_COMPENSATION 0.005f`, `MIN_ONE_MINUS_ALPHA 1e-6f` |
| `research/gsplat/gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu` | SH fwd (:435, k16 fast path :475) / bwd (:711) — for checking the gradient tensors and the atomic sites |
| `research/gsplat/gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu` | projection fused fwd (:38) / bwd (:377) — the gradient tensors (:399-408) and the radii gate (:422) |
| `research/gsplat/gsplat/cuda/csrc/IntersectTile.cu` | isect key encoding (:270-285, :455-457), the two-pass structure (:248, :338) |
| `research/gsplat/gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchFwd.cu` | raster fwd kernel (:42), batch loading (:196-220), blending (:224-264), output writes (:283-295) |
| `research/gsplat/gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchBwd.cu` | raster bwd kernel (:42), T reconstruction (:229), v_alpha (:236-257), the clamp gate (:259), the warp reduction (:282-289), atomics (:297-316) |
| `research/gsplat/gsplat/cuda/csrc/RasterizeToPixels3DGSDevice.cuh:45-56` | `eval_gaussian_weight` — the sigma/vis/alpha/valid definitions |

Existing infrastructure (splatx-metal):

| File | Role |
|---|---|
| `tools/gen_fixtures.py` | Stub package loading (`_load_gsplat_modules`, :23-50) + generation of two fixtures |
| `tools/sxt.py` | SXT writer — supports only dtype 0=float32 and 1=int32 (`SXT1` magic, ndim, shape u64, raw bytes) |
| `fixtures/math_quat_scale_to_covar/`, `fixtures/proj_fully_fused/` | The existing Phase 0 fixtures (seed 42 / 7) |
| `Sources/SplatXCore/SXTensor.cpp` | SXT loader (binary compatibility MUST be preserved) |

## Algorithm overview

The four axes of the fixture strategy:

1. **Forward goldens**: for each Phase 1 kernel, call the pure PyTorch function in
   `_torch_impl`/`_math` directly and dump the inputs and outputs as SXT. See the mapping table
   below.
2. **Backward goldens**: run the same forward function on `requires_grad=True` leaves, inject a
   fixed-seed `torch.randn` upstream gradient through
   `torch.autograd.grad(outputs, inputs, grad_outputs=...)`, and dump the input gradients. The
   upstream tensor itself is also stored in the fixture so that the Metal side consumes identical
   bytes.
3. **Tolerances**: the Metal backward uses device `atomic_float` accumulation (a non-deterministic
   accumulation order), so the comparison uses an element-wise mixed tolerance + a whole-tensor
   relative L2 gate + two-run self-consistency.
4. **gen_fixtures.py extension**: reuse the existing stub loading (`_load_gsplat_modules`) unchanged
   and add a generator registry, a shared `dump_backward` helper, and a self-contained
   differentiable reference rasterizer.

### Phase 1 kernel → forward golden mapping

| Metal kernel (CUDA original) | Golden function (file:line) | Signature | Returns |
|---|---|---|---|
| `spherical_harmonics_fwd` (SphericalHarmonicsCUDA.cu:435) | `_spherical_harmonics` (_torch_impl.py:1052-1067) | `(degrees_to_use: int, dirs: [..., N, 3], coeffs: [N, K, D])` | `colors [..., N, D]` |
| `projection_ewa_3dgs_fused_fwd` (ProjectionEWA3DGSFused.cu:38) | `_fully_fused_projection` (_torch_impl.py:262-352) | `(means [..., N,3], covars [..., N,3,3], viewmats [..., C,4,4], Ks [..., C,3,3], width, height, eps2d=0.3, near_plane=0.01, far_plane=1e10, calc_compensations, camera_model)` | `(radii int32 [...,C,N,2], means2d [...,C,N,2], depths [...,C,N], conics [...,C,N,3], compensations [...,C,N] or None)` |
| ↑ the quats/scales input path | composed with `_quat_scale_to_covar_preci` (_math.py:689-719) | `(quats [...,4], scales [...,3], compute_covar, compute_preci, triu)` | `(covars [...,3,3], precis)` |
| `isect_tiles` (IntersectTile.cu:214) | `_isect_tiles` (_torch_impl.py:355-451, `@torch.no_grad`) | `(means2d [..., N,2], radii [..., N,2], depths [..., N], tile_size, tile_width, tile_height, sort=True)` | `(tiles_per_gauss int32 [...,N], isect_ids int64 [n_isects], flatten_ids int32 [n_isects])` |
| `isect_offset_encode` (IntersectTile.cu:670-) | `_isect_offset_encode` (_torch_impl.py:454-481, `@torch.no_grad`) | `(isect_ids int64, I, tile_width, tile_height)` | `offsets int32 [I, tile_height, tile_width]` (a global flattened cumulative sum) |
| `rasterize_to_pixels_3dgs_fwd` (RasterizeToPixels...Fwd.cu:42) | **cannot be used directly** — see below | — | — |

**Key finding**: `_rasterize_to_pixels` (_torch_impl.py:814-924) **cannot be used directly** as a
golden. Internally it depends on `rasterize_to_indices_in_range` (the CUDA `_wrapper`, imported at
:849) and on `nerfacc` (:763-766, via `accumulate`), so it cannot run on a macOS CPU. Therefore a
**self-contained differentiable reference rasterizer `_rasterize_reference`** is newly implemented
in `gen_fixtures.py` (see the formula details). Its formulas follow exactly the sigma/alpha
expressions of `accumulate` (_torch_impl.py:787-793) and the gate semantics of the CUDA fwd kernel
(RasterizeToPixels3DGSSerialBatchFwd.cu:224-295, Device.cuh:45-56). The two `_isect_*` functions,
which need forward only, are used as goldens as-is.

### Fixture chaining

Chaining the fixtures in the order of the real pipeline naturally guarantees that the inputs are
"non-degenerate":

```
proj_fully_fused (existing seed=7) ──radii/means2d/depths──▶ isect_tiles_std
isect_tiles_std ──isect_ids──▶ isect_offset_encode (the same fixture directory)
proj conics/means2d + isect offsets/flatten_ids + random colors/opacities ──▶ raster_std
```

## Formulas in detail

### Precise spec for each forward golden

#### 1. `_spherical_harmonics` (_torch_impl.py:1052-1067)

```python
dirs = F.normalize(dirs, p=2, dim=-1)              # :1063 (eps=1e-12 built in)
num_bases = (degrees_to_use + 1) ** 2              # :1064
bases[..., :num_bases] = _eval_sh_bases_fast(num_bases, dirs)   # :1066, higher coefficients are 0
colors = (bases[..., None] * coeffs).sum(dim=-2)   # :1067  → [..., N, D]
```

- The coefficient constants live in `_eval_sh_bases_fast` (:968-1049): `0.2820947917738781`,
  `-0.48860251190292`, `-1.092548430592079`, `0.5462742152960395`, … they MUST match CUDA down to
  the last digit.
- Assertions: `coeffs.dim() == 3` (no batch), `coeffs.shape[0] == dirs.shape[-2]`, and
  `(degrees_to_use+1)**2 <= K` (:1058-1061).
- CUDA normalizes inside the kernel with `rsqrtf` (SphericalHarmonicsCUDA.cu:59-62, **no epsilon**),
  while torch uses `F.normalize` (eps=1e-12). → **the fixture's dirs guarantee norm ≥ 0.1**, which
  removes the difference.
- `masks` is fixed to `None` (everything valid) in the v1 fixture.

#### 2. `_fully_fused_projection` (_torch_impl.py:262-352)

```python
means_c, covars_c = _world_to_cam(means, covars, viewmats)        # :294 (:225-259)
means2d, covars2d = _persp_proj(means_c, covars_c, Ks, W, H)      # :301 (pinhole, :53-108)
det_orig = det(covars2d)                                          # :305-308
covars2d += eye(2) * eps2d                                        # :309, eps2d=0.3
det = det(covars2d).clamp(min=1e-10)                              # :311-315
compensations = sqrt(clamp(det_orig/det, min=MIN_COMPENSATION**2))# :317-320 (0.005^2)
conics = [c11/det, -(c01+c10)/2/det, c00/det]                     # :324-331
depths = means_c[..., 2]                                          # :333
radius_{x,y} = ceil(3.33 * sqrt(covars2d[..,i,i]))                # :335-336 (GAUSSIAN_EXTEND)
radius[~(near_plane < depth < far_plane)] = 0                     # :340-341
radius[~inside] = 0   # AABB-screen intersection test             # :343-349
radii = radius.int()                                              # :351
```

- The field-of-view clamp of `_persp_proj`: `lim_x_pos = (W-cx)/fx + 0.3*tan_fovx` and so on
  (:91-96) — `tx = tz * clamp(tx/tz, -lim_x_neg, lim_x_pos)`. Keep the signs and the 0.3 constant.
- The quats/scales input variant: `covars, _ = _quat_scale_to_covar_preci(quats, scales, True, False)`
  (_math.py:689-719; `R = _quat_to_rotmat(quats)` (:655-673) → `M = R * scales`,
  `covar = M Mᵀ`), then composed with the function above. This corresponds to the CUDA fused fwd
  (ProjectionEWA3DGSFused.cu:44-47) taking covars, quats, and scales all as optional.
- The `opacities`-based culling and `radius_clip` (:48, :56) that exist only in the CUDA fwd are
  absent from the torch golden → **the fixture verifies only the radius_clip=0, opacities=None
  path** (open question #2).

#### 3. `_isect_tiles` (_torch_impl.py:355-451)

```python
tile_mins = floor(means2d/tile_size - radii/tile_size).int()      # :385-387
tile_maxs = ceil (means2d/tile_size + radii/tile_size).int()      # :388
# clamp each axis to [0, tile_width] and [0, tile_height]         # :389-392
tiles_per_gauss = prod(tile_maxs - tile_mins) * (radii > 0).all() # :393-394
# 64-bit key packing (one per gaussian per covered tile):
depth_id = float32_bits(depths[img, g])                           # :417-421 (bit reinterpretation)
isect_ids_lo[i] = depth_id                                        # :433
isect_ids_hi[i] = (image_id << tile_n_bits) | tile_id             # :434, tile_id = y*tile_width+x
flatten_ids[i]  = image_id * N + gauss_id                         # :435 (global flattened index)
isect_ids = (hi << 32) | (lo & 0xFFFFFFFF)                        # :442-444
isect_ids, order = torch.sort(isect_ids); flatten_ids = flatten_ids[order]  # :446-448
```

- `tile_n_bits = _bits_for_count(tile_width*tile_height)` (:405; `_bits_for_count` :42-50 —
  `(count-1).bit_length()`, which prevents over-counting at powers of two). Assertion:
  `image_n_bits + tile_n_bits + 32 <= 64` (:406).
- The bit layout is identical to the CUDA packing: `isect_ids[cur_idx] = iid_enc | (tile_id << 32) | depth_id_enc`
  (IntersectTile.cu:455-457), `iid_enc = iid << (32 + tile_n_bits)` (:273),
  `depth_id_enc = __float_as_uint(depth_f)` (:285). Sorting on the depth bit pattern preserves
  order **only for positive depths** — this holds because projection culls with
  `near_plane=0.01 > 0`.
- On a sort tie (identical image/tile/depth bits), the `flatten_ids` order depends on the stability
  of the sort — with randn-based depths there are effectively no ties in the fixture (open question
  #4).

#### 4. `_isect_offset_encode` (_torch_impl.py:454-481)

```python
isect_ids_uq, counts = unique_consecutive(isect_ids >> 32)        # :470 (upper 32 bits = img|tile)
tile_counts[image_ids_uq, tile_y, tile_x] = counts                # :472-477
offsets = cumsum(tile_counts.flatten()) - tile_counts             # :479-480, exclusive prefix sum
return offsets.int()                                              # [I, tile_height, tile_width]
```

- offsets is **a global cumulative sum over the whole flattened [I, TH, TW]**. This is consistent
  with the CUDA fwd taking the range_end of the last tile as `n_isects` (Fwd.cu:164-167).

#### 5. `_rasterize_reference` — a new self-contained golden (implemented in gen_fixtures.py)

A differentiable reference that vectorizes the per-pixel serial semantics of the CUDA fwd kernel
over tiles. The gate constants and their order follow exactly:

- `sigma = 0.5*(conic.x*dx² + conic.z*dy²) + conic.y*dx*dy`, `vis = exp(-sigma)`,
  `alpha = min(MAX_ALPHA, opac*vis)`, `valid = !(sigma < 0 || alpha < ALPHA_THRESHOLD)`
  (Device.cuh:47-53; the corresponding torch expressions at _torch_impl.py:787-793).
- Pixel centers `px = j + 0.5, py = i + 0.5` (Fwd.cu:~118, Bwd.cu:111-112), `dx = mean.x - px`.
- The early termination is **exclusive**: if `next_T = T*(1-alpha) <= TRANSMITTANCE_THRESHOLD`, that
  Gaussian is **not included** and the pixel terminates (Fwd.cu:246-251).
- Outputs: `render_alphas = 1 - T` (:286), `render_colors = pix_out + T_final*bg` (:289-292), and
  `last_ids = cur_idx` = the **global flatten_ids index** of the last contributing Gaussian
  (`cur_idx[p] = batch_start + t`, :261; the initial value 0 if there is no contribution, :186).

Pseudocode (vectorized per tile — isomorphic to CUDA's CTA-per-tile parallelization):

```python
def _rasterize_reference(means2d, conics, colors, opacities, W, H, tile_size,
                         isect_offsets, flatten_ids, backgrounds=None):
    # called with every tensor upcast to f64 (it doubles as the backward golden)
    I, N = means2d.shape[:2]; TH, TW = isect_offsets.shape[-2:]; C = colors.shape[-1]
    offs = torch.cat([isect_offsets.reshape(-1).long(),
                      torch.tensor([len(flatten_ids)])])          # global cumulative + sentinel
    out_c  = torch.zeros(I, H, W, C, dtype=means2d.dtype)
    out_a  = torch.zeros(I, H, W, 1, dtype=means2d.dtype)
    last   = torch.zeros(I, H, W, dtype=torch.int32)              # empty range → stays 0
    for img, ty, tx in product(range(I), range(TH), range(TW)):
        t = (img*TH + ty)*TW + tx; rs, re = offs[t].item(), offs[t+1].item()
        if rs == re: continue                                     # T=1, out=0, last=0 unchanged
        gid  = flatten_ids[rs:re].long()                          # [R] ∈ [I*N)
        xy   = means2d.reshape(-1,2)[gid]; con = conics.reshape(-1,3)[gid]
        opac = opacities.reshape(-1)[gid]; rgb = colors.reshape(-1,C)[gid]
        # P valid pixels in the tile (clipped at the image boundary), pxy [P,2]
        dx = xy[None,:,0] - pxy[:,0:1]; dy = xy[None,:,1] - pxy[:,1:2]     # [P,R]
        sigma = 0.5*(con[:,0]*dx*dx + con[:,2]*dy*dy) + con[:,1]*dx*dy
        alpha = torch.clamp_max(opac * torch.exp(-sigma), MAX_ALPHA)       # 0.99
        skip  = (sigma.detach() < 0) | (alpha.detach() < ALPHA_THRESHOLD)  # hard gate (1/255)
        a_eff = torch.where(skip, torch.zeros((), dtype=alpha.dtype), alpha)
        one_m = 1.0 - a_eff
        T_ex  = torch.cat([one_m.new_ones(P,1), one_m.cumprod(1)[:, :-1]], 1)  # exclusive T
        next_T = T_ex * one_m
        dead  = (next_T.detach() <= TRANSMITTANCE_THRESHOLD).int().cummax(1).values.bool()
        w     = a_eff * T_ex * (~dead)                            # excludes the triggering gaussian itself
        out_c[img, ys, xs] = (w[...,None] * rgb[None]).sum(1)
        a_pix = w.sum(1)                                          # == 1 - T_final (mathematically equivalent)
        out_a[img, ys, xs, 0] = a_pix
        if backgrounds is not None:
            out_c[img, ys, xs] += (1.0 - a_pix)[:,None] * backgrounds[img]
        contrib = (w.detach() > 0)
        last[img, ys, xs] = torch.where(contrib.any(1),
                                        rs + last_true_index(contrib, dim=1), 0).int()
    return out_c, out_a, last
```

Caveats:

- The `skip`/`dead` masks MUST be **decided on `.detach()`ed values** — the CUDA backward treats
  these gates as hard cuts too (bwd re-evaluates `gw.valid` and contributes 0 if invalid,
  Bwd.cu:204-211; Gaussians after `bin_final = last_ids[pix_id]` are excluded, :140, :189-192).
- The autograd of `clamp_max(., MAX_ALPHA)` (grad 0 when clamped) is equivalent to the CUDA bwd gate
  `if(opac * vis <= MAX_ALPHA)` (Bwd.cu:259) — for a clamped Gaussian, v_conic/v_xy/v_opacity are 0
  and only v_rgb remains (the `fac = alpha*T` path at :270-281 is outside the gate).
- Because `1 - a_eff >= 1 - MAX_ALPHA = 0.01`, there is no division by zero in the cumprod backward.
  For the same reason, the CUDA bwd's `ra = 1/fmaxf(MIN_ONE_MINUS_ALPHA, 1-alpha)` (Bwd.cu:229,
  1e-6f) is a dead guard as long as MAX_ALPHA=0.99 — put it into the Metal port identically, but it
  cannot be verified by a fixture.

### Backward golden generation strategy (torch.autograd.grad)

#### The shared recipe

```python
def dump_backward(out_dir, fwd_fn, inputs_f32: dict, diff_names: list[str],
                  out_names: list[str], grad_seed: int,
                  upstream_masks: dict[str, Tensor] | None = None):
    # 1) build the leaves by upcasting to f64 the inputs created and saved in f32
    leaves = {k: v.double().clone().requires_grad_(True) if k in diff_names else v
              for k, v in inputs_f32.items()}
    outs = fwd_fn(**leaves)                              # f64 graph
    # 2) upstream gradient: fixed-seed randn in f32 → save → upcast to f64 and inject
    g = torch.Generator().manual_seed(grad_seed)
    v_outs = {}
    for name, o in zip(out_names, outs):
        v = torch.randn(o.shape, generator=g, dtype=torch.float32)
        if upstream_masks and name in upstream_masks:
            v = v * upstream_masks[name].to(v.dtype)     # zero the upstream of culled elements
        save_sxt(out_dir / f"v_{name}.sxt", v.numpy())   # the identical bytes Metal will consume
        v_outs[name] = v.double()
    grads = torch.autograd.grad(
        outputs=[outs[out_names.index(n)] for n in out_names],
        inputs=[leaves[k] for k in diff_names],
        grad_outputs=[v_outs[n] for n in out_names],
        allow_unused=True)
    # 3) cast the expected gradients to f32 and save (a zero tensor if absent)
    for k, gr in zip(diff_names, grads):
        arr = (gr if gr is not None else torch.zeros_like(leaves[k])).float().numpy()
        save_sxt(out_dir / f"v_{k}.sxt", arr)
```

Principles:

- **Inputs and upstream are created and saved in f32** (so Metal and the golden consume
  bit-identical inputs), **the golden computation runs in f64** (gradients close to the true
  values), and **saving casts to f32**.
- The upstream seed is derived as `fixture_seed + 1000` (for example proj seed 7 → grad seed 1007).
- The file naming convention uses the gsplat kernel argument names as-is: both upstream and expected
  gradients are `v_<tensor name>.sxt`. There is no name collision inside a fixture (the output names
  differ from the input names), and the roles are distinguished by the `grad_outputs` /
  `grad_inputs` lists in meta.json.

#### Backward spec per kernel

**SH backward** — golden: autograd through `_spherical_harmonics`.
leaves: `coeffs [N,K,3]`, `dirs [C,N,3]`. upstream: `v_colors [C,N,3]` (fixed-seed randn).
Expected gradients: `v_coeffs [N,K,3]` (summed over the batch C — CUDA accumulates with
gpuAtomicAdd, SphericalHarmonicsCUDA.cu:159-361) and `v_dirs [C,N,3]` (:756-760 atomicAdd; the
normalization backward `v_d = (v_dir_n - dot(v_dir_n, dir_n)·dir_n) * inorm`, :186 — mathematically
identical to the autograd of torch `F.normalize`). Coefficients of degrees higher than
`degrees_to_use` get grad 0 (torch leaves those bases at 0 as well, so they agree automatically,
_torch_impl.py:1065-1066).

**Projection fused backward** — golden: autograd through `_fully_fused_projection` (the covars
path) and through the composition `_quat_scale_to_covar_preci`∘`_fully_fused_projection` (the
quats/scales path).
leaves: `{means, covars}` or `{means, quats, scales}` (the viewmats variant is deferred to Phase 2).
upstream: `v_means2d [C,N,2]`, `v_depths [C,N]`, `v_conics [C,N,3]`, `v_compensations [C,N]`
(1:1 with ProjectionEWA3DGSFused.cu:399-402).
**Upstream masking is required**: the CUDA bwd skips (cid, gid) pairs with `radii <= 0`
(ProjectionEWA3DGSFused.cu:422-425), so all four upstream tensors are multiplied by the mask
`valid = (radii > 0).all(-1)` before being saved and injected. The torch golden still has a graph
connected to culled Gaussians, so without the masking the golden alone would carry extra gradients.
Expected gradients: `v_means [N,3]`, `v_covars [N,3,3]` (or `v_quats [N,4]`, `v_scales [N,3]`) —
CUDA accumulates with gpuAtomicAdd across the C cameras (:568-596), which is equivalent to the
autograd summation.
Note: the torch golden's `det.clamp(min=1e-10)` (:315) and the `clamp(min=MIN_COMPENSATION²)` of
`compensations` (:319) give grad 0 when the clamp is active — cross-check against spec 02 when
implementing the Metal bwd whether the CUDA vjp has the same gate.

**Rasterize backward** — golden: autograd through `_rasterize_reference`.
leaves: `means2d [I,N,2]`, `conics [I,N,3]`, `colors [I,N,3]`, `opacities [I,N]`
(plus a `backgrounds [I,3]` variant). upstream: `v_render_colors [I,H,W,3]`,
`v_render_alphas [I,H,W,1]` (1:1 with RasterizeToPixels3DGSSerialBatchBwd.cu:66-68).
Expected gradients: `v_means2d`, `v_conics`, `v_colors`, `v_opacities` (:70-74; atomic sites at
:297-316).
`v_means2d_abs` (:69, :312-313) is the per-pixel sum of `|v_xy_local|` in the golden, so it cannot
be obtained by autograd — add a manual accumulation to the reference implementation if needed (open
question #3); it is unverified in v1.
`last_ids`/`render_alphas` are reused directly as bwd inputs from the forward fixture (:135
`T_final = 1 - render_alphas[pix_id]`, :140 `bin_final = last_ids[pix_id]`) — that is, **the Metal
bwd test takes the fixture's golden fwd outputs, not the Metal fwd outputs, as its inputs**, which
blocks error propagation.

**The isect family**: `@torch.no_grad` — no backward. Only exact-match comparison of the integer
outputs.

#### Tolerances that account for atomic non-determinism

The Metal backward uses the native `atomic_float` addition of the M4/A17 Pro, so the accumulation
order varies from run to run. The deviation caused by the non-associativity of fp32 addition scales
with (number of contributions × value scale). Comparison rules:

Element-wise: `|metal - golden| <= atol + rtol * |golden|`, plus for the whole tensor
`‖metal - golden‖₂ / max(‖golden‖₂, 1e-12) <= rel_l2`.

| fixture / tensor | Kind | rtol | atol | rel_l2 | Basis |
|---|---|---|---|---|---|
| SH fwd `colors` | fwd | 1e-5 | 1e-6 | — | only the fp32 operation-order difference of the polynomial |
| proj fwd `means2d`/`depths` | fwd | 1e-5 | 1e-5 | — | an absolute error in pixel units is allowed |
| proj fwd `conics`/`compensations` | fwd | 1e-4 | 1e-6 | — | amplification from the det division |
| proj fwd `radii` | fwd | exact (int) | — | — | consider allowing ±1 if the ceil boundary flakes (open question #5) |
| isect `tiles_per_gauss`/`isect_ids`/`flatten_ids`/`offsets` | fwd | exact (int) | — | — | an integer algorithm |
| raster fwd `render_colors`/`render_alphas` | fwd | 1e-4 | 1e-5 | — | serial accumulation (deterministic), but the exp implementations differ — Metal should use `metal::precise::exp` |
| raster fwd `last_ids` | fwd | exact (int) | — | — | a policy is needed if the threshold boundary flakes (open question #5) |
| SH bwd `v_coeffs`/`v_dirs` | bwd | 1e-4 | 1e-6 | 1e-4 | at most B(=2) atomics per element |
| proj bwd `v_means`/`v_covars`/`v_quats`/`v_scales` | bwd | 2e-4 | 1e-6 | 1e-4 | at most C(=2) atomics per element |
| raster bwd `v_colors` | bwd | 1e-3 | 1e-6 | 1e-3 | on the order of hundreds of atomics per element (the pixel count) |
| raster bwd `v_means2d`/`v_conics`/`v_opacities` | bwd | 5e-3 | 1e-5 | 1e-3 | the T reconstruction by division (Bwd.cu:229-231) + a large number of atomics |

An additional gate: **two-run self-consistency** — run the same Metal bwd twice and check
`‖run1 - run2‖₂ / ‖golden‖₂ <= 0.1 * rel_l2`. When this value is large it distinguishes whether the
error comes from atomic noise or from an implementation bug. The tolerance values are stored in
meta.json and read by the tests (hardcoding is prohibited) — they can be adjusted after measurement
on device.

### Proposed fixture input sizes (small but non-degenerate)

| fixture directory | Size | Why it avoids degeneracy |
|---|---|---|
| `sh_fwd_bwd_k16` | C=2, N=64, K=16, D=3; dumps `colors_deg{d}.sxt` for each of `degrees_to_use ∈ {0,1,2,3}`, bwd at deg 3 | covers every degree branch (:989,:999,:1013,:1029) plus the k16 fast path (:475). dirs = normalize(randn)*(0.5+rand) guarantees norm∈[0.5,1.5] |
| `sh_fwd_k25` | C=1, N=32, K=25, deg 4, fwd only | the generic kernel path of the degree 4 branch (:1029-1049) |
| `proj_fully_fused` (existing, extended) | N=128, C=2, W=160, H=120, seed 7 kept; bwd dumps added (grad seed 1007) | keeps the existing golden. Includes some radii=0 culled Gaussians (±clip, behind the near plane) — verifies the upstream masking path |
| `proj_fused_qs` | the same size, with quats/scales as inputs (seed 11) | the composed quat→covar vjp (the quats/scales path of the CUDA fused kernel) |
| `isect_tiles_std` | chained from proj: I=2, N=128, tile_size=16, tile_width=10, tile_height=8 (W=160/16, ceil(120/16)) | 80 tiles → tile_n_bits=7, image_n_bits=1 (non-trivial bit packing). H=120 is not a multiple of 16 → a partial tile row at the bottom. Large-radii Gaussians cover multiple tiles, radii=0 covers 0 tiles |
| `raster_small` | I=1, W=32, H=32, tile 16 → 2×2 tiles, N=24 hand-placed, CDIM=3; one variant with `backgrounds` | the minimal hand-checkable case: (a) three overlapping Gaussians at the center with opac 0.99/0.995 → early termination + the MAX_ALPHA clamp, (b) a small one inside a single tile, (c) one straddling 4 tiles, (d) one half off-screen, (e) one empty tile (the empty-range path: out=0, last=0) |
| `raster_std` | chained: I=2, W=160, H=120, 10×8 tiles, N=128, CDIM=3, backgrounds=None; fwd+bwd | a realistic distribution. colors=rand[0,1], opacities=rand*0.93+0.05, with 8 of them fixed at 0.99 to guarantee saturated pixels |
| `raster_dense` | I=1, W=64, H=64, 4×4 tiles, N=1024 concentrated at the center, opac≤0.995; fwd+bwd | hundreds of intersections per tile → two or more threadgroup batch loops (exceeding BATCH_SIZE=256) + a large number of early terminations + the case where bwd atomics explode |

Data-dependent sizes such as n_isects are carried by the shape in the SXT header itself and are also
recorded in meta.json.

### The extended structure of gen_fixtures.py

Reuse the existing stub loading of `_load_gsplat_modules()` (:23-50) unchanged (the gsplat package
is not installed, which avoids the CUDA JIT). Proposed structure:

```python
# tools/gen_fixtures.py (skeleton after the extension)
GENERATORS: dict[str, Callable] = {}
def fixture(name):
    def deco(fn): GENERATORS[name] = fn; return fn
    return deco

# shared helpers
def dump_backward(...)            # the recipe above
def _rasterize_reference(...)     # the differentiable golden rasterizer (f64)
def load_fixture_sxt(dir, name)   # for chaining: reloads the SXT of an existing fixture
def write_meta(dir, scalars, roles, tols, seeds)
#   the meta.json schema:
#   { "golden_source": "gsplat@77ab983:_torch_impl.py",
#     "seeds": {"input": 7, "grad": 1007},
#     "scalars": {"width":160, "height":120, "tile_size":16, ...},
#     "tensors": {"inputs":[...], "outputs":[...],
#                 "grad_outputs":["v_render_colors",...],   # upstream (injected)
#                 "grad_inputs":["v_means2d",...]},         # expected grads
#     "tolerances": {"v_means2d": {"rtol":5e-3,"atol":1e-5,"rel_l2":1e-3}, ...} }

@fixture("sh_fwd_bwd_k16")
def gen_sh(math_mod, ti): ...
@fixture("isect_tiles_std")
def gen_isect(math_mod, ti):
    means2d = load_fixture_sxt("proj_fully_fused", "means2d"); ...   # chaining
    tiles_per_gauss, isect_ids, flatten_ids = ti._isect_tiles(...)
    # SXT has no int64 → store as the same lo/hi split as the torch impl (:399-400)
    save_sxt(out/"isect_ids_lo.sxt", (isect_ids & 0xFFFFFFFF).int().numpy())  # depth bits
    save_sxt(out/"isect_ids_hi.sxt", (isect_ids >> 32).int().numpy())         # img|tile
    offsets = ti._isect_offset_encode(isect_ids, I=2, tile_width=10, tile_height=8)
    ...

if __name__ == "__main__":
    names = sys.argv[1:] or list(GENERATORS)     # selective regeneration
    math_mod, ti = _load_gsplat_modules()
    for n in names: GENERATORS[n](math_mod, ti)
```

- The existing `gen_math_fixture`/`gen_projection_fixture` are only registered in the registry;
  their output paths and seeds stay unchanged (Phase 0 test compatibility).
- The int64 `isect_ids` is stored as a **lo/hi int32 split** rather than extending the SXT dtypes —
  the torch golden itself computes with lo/hi (:399-400, :433-434), and it is natural for the Metal
  kernel to sort on a (hi,lo) two-word key as well. `SXTensor.cpp` is unchanged.
- Guaranteeing determinism: every random number uses an explicit `torch.Generator().manual_seed(...)`,
  only CPU tensors are used, and the torch version is pinned by `tools/uv.lock`. Regenerating a
  fixture MUST produce identical bytes (reproducibility check: consider recording
  `shasum fixtures/**/*.sxt` in meta.json).

## Thread and memory configuration

The parallelization shapes of the original kernels that the fixture sizes must cover (so that the
configuration under verification is 1:1 with the real dispatch shape):

| Kernel | CUDA parallelization | Condition the fixture forces |
|---|---|---|
| SH fwd/bwd | 1D grid: `B*N*D` threads, 256/block (SphericalHarmonicsCUDA.cu launch, bwd :786-791) | B*N*D = 2*64*3 = 384 → more than one threadgroup, with boundary threads (idx≥count) present |
| proj fused fwd/bwd | 1D grid: `B*C*N` (ProjectionEWA3DGSFused.cu:66-76) | C=2 verifies the (cid,gid) indexing and includes the radii=0 skip path |
| isect_tiles | two-pass: pass1 counts → prefix sum → pass2 writes (IntersectTile.cu:248-onwards, `first_pass`), followed by a cub radix sort | a mix of multi-tile and zero-tile Gaussians, plus a partial tile row |
| raster fwd | CTA per tile: `(TILE=16, CTA=256)` → PIXELS_PER_THREAD=1 (Fwd.cu:65-67), staging 256 Gaussians per batch into shared memory (:171-174) | `raster_dense` produces a range > 256, verifying two or more batch loops plus the `__syncthreads_count` early exit (:270-273) |
| raster bwd | CTA per tile, back-to-front, with the start point truncated by `bin_final`/`warp_bin_final` (Bwd.cu:140,154-155,186-192) | a mix of saturated pixels (last_ids < range_end-1) and unsaturated pixels |

The Metal threadgroup memory budget (at tile 16, CTA 256, CDIM 3, limit 32KB):

- fwd: `id_batch` 256×4 + `xy_opacity_batch` 256×12 + `conic_batch` 256×12 = **7,168 B** (the same layout as Fwd.cu:170-174)
- bwd: the above + `rgbs_batch` 256×3×4 = **10,240 B** (Bwd.cu:129-132) — both fit within 32KB with headroom.

## MSL porting design (kernel signatures as the fixture contract)

The fixture tensors are the kernel buffer contract. The test harness operates in this order: SXT →
`MTLBuffer` (identical layout, f32/i32 as-is) → dispatch → readback → the comparison utility.

```metal
// 1) SH fwd — dispatch: 1D, threadsPerThreadgroup=256, grid=ceil(B*N*D/256)
kernel void sh_fwd(
    constant SHParams& p,                      // degrees_to_use, B, N, K, D
    device const packed_float3* dirs,          // [B,N] ← dirs.sxt
    device const float* coeffs,                // [N,K,D] ← coeffs.sxt
    device float* colors,                      // [B,N,D] → compared against colors_deg{d}.sxt
    uint idx [[thread_position_in_grid]]);

// 2) SH bwd — atomic sites: v_coeffs (B times per K*D), v_dirs (3 components, D times/thread)
kernel void sh_bwd(
    constant SHParams& p,
    device const packed_float3* dirs,
    device const float* coeffs,
    device const float* v_colors,              // ← v_colors.sxt (upstream)
    device atomic_float* v_coeffs,             // [N,K,D] → compared against v_coeffs.sxt
    device atomic_float* v_dirs,               // [B,N,3] → compared against v_dirs.sxt
    uint idx [[thread_position_in_grid]]);
// The same decomposition as CUDA (thread = (batch, gaussian, channel), SphericalHarmonicsCUDA.cu:727-741).
// The first implementation uses 1:1 atomics with no simdgroup reduction; optimize after the fixture passes.

// 3) proj fused fwd/bwd — 1D, 256/tg, grid=ceil(C*N/256). The bwd atomics:
//    per-camera atomic_fetch_add into v_means/v_quats/v_scales (corresponds to ProjectionEWA3DGSFused.cu:577-596)

// 4) isect_tiles — pass1: count the tiles per gaussian (written to a buffer) → prefix sum on the
//    CPU or in a separate kernel → pass2: write the (hi,lo) keys and flatten_ids → sort. The first
//    implementation prioritizes passing the fixture with GPU writes + **a CPU std::stable_sort
//    (64-bit keys)**, leaving a GPU radix sort for later. isect_ids_lo/hi.sxt and
//    flatten_ids.sxt are compared for exact equality.

// 5) raster fwd — dispatch: threadgroup=(16,16,1) = 256 threads/tile,
//    threadgroupsPerGrid=(TW, TH, I). PIXELS_PER_THREAD=1 (the same configuration as CUDA's (16,256)).
kernel void raster_fwd(
    constant RasterParams& p,                  // W,H,tile_size,TW,TH,I,N,n_isects,has_bg
    device const packed_float2* means2d,       // [I,N]
    device const packed_float3* conics,
    device const float* colors,                // [I,N,3]
    device const float* opacities,
    device const float* backgrounds,           // [I,3] or null
    device const int* isect_offsets,           // [I,TH,TW]
    device const int* flatten_ids,             // [n_isects]
    device float* render_colors,               // [I,H,W,3]
    device float* render_alphas,               // [I,H,W,1]
    device int* last_ids,                      // [I,H,W]
    uint3 tg_id [[threadgroup_position_in_grid]],
    uint2 tid  [[thread_position_in_threadgroup]]);
// threadgroup memory: int id_batch[256]; float3 xy_opac[256]; float3 conic_batch[256];
// barrier: threadgroup_barrier(mem_threadgroup) after the batch load (corresponds to the
// __syncthreads of Fwd.cu:210-220; CTA=256 > simd 32, so simdgroup_barrier is not enough).
// early-exit vote: __syncthreads_count(done)==256 (Fwd.cu:270) →
// implement with a threadgroup atomic counter + barrier, or simd_all → a threadgroup reduction.
// exp uses metal::precise::exp (fast::exp risks exceeding the fixture tolerance and differs from CUDA __expf).

// 6) raster bwd — the same dispatch. warpSum(Bwd.cu:282-289) → simd_sum (simdgroup width 32,
//    1:1 with the CUDA warp); one thread per simdgroup commits to the device atomic_float
//    (atomicAdd_system → atomic_fetch_add_explicit(relaxed), corresponds to Bwd.cu:297-316).
//    warp_bin_final(cg::reduce max, :155) → simd_max.
```

The comparison utilities (test harness, Swift/C++):

- `assertAllClose(golden, actual, rtol, atol, relL2)` — reports the number of violations, the max
  abs/rel error, and the first violating index. It uses the tolerances from meta.json.
- Integer tensors use `assertEqual` (the radii/last_ids/isect family).
- bwd tests do `run twice → self-consistency` and then compare against the golden.
- Among the bwd kernel inputs, the forward products (`render_alphas`, `last_ids`, `conics`, and so
  on) MUST be loaded from **the fixture's golden values** — so that Metal fwd error does not
  contaminate the bwd verification.

## Open questions

1. **Whether to extend the SXT dtypes** — int64 (`isect_ids`) and bool (`masks`, the sparse
   `tile_mask`) are absent from the current format. v1 avoids them with the lo/hi int32 split and by
   not using masks, but the Phase 2 sparse family (`_isect_tiles_sparse` and
   `_build_sparse_tile_layout` use uint64 bitmasks, _torch_impl.py:485-710) may require format v2
   (dtype 2=int64, 3=uint8).
2. **No golden for radius_clip / opacity culling** — the CUDA fused fwd's `radius_clip` (:56) and
   the `opacities`-based culling are absent from the `_fully_fused_projection` torch golden. If the
   Phase 1 Metal kernel implements them, the fixture covers only the radius_clip=0, opacities=None
   path — a decision is needed between adding an in-house reference and deferring to Phase 2.
3. **Verifying `v_means2d_abs`** — an absolute-value accumulation that autograd cannot produce
   (Bwd.cu:312-313). It is used by densification (the growth heuristic), so it will be needed
   eventually. Whether to add a manual accumulation to `_rasterize_reference`, or to substitute a
   unit test on the kernel itself (the property `|v_means2d| <= v_means2d_abs` element-wise).
4. **Tie policy for the sort** — cub radix sort pairs is stable. If the Metal implementation
   (especially once a GPU sort is introduced) is non-stable, the flatten_ids order can differ for
   identical (image, tile, depth-bits) keys. The fixture avoids ties by using randn depths, but the
   question is whether it is worth adding an explicit tie-case fixture (two Gaussians at the same
   depth) to verify "is the resulting render invariant under swapping the alpha order on a tie?"
5. **Boundary flake policy** — for `radii`'s `ceil(3.33*sqrt(...))` and for pixels near the
   `last_ids` threshold, a single-ULP fp difference can change the integer result. The current
   proposal is exact comparison, plus consideration of a rule that excludes a pixel on failure if
   its `next_T` is within `TRANSMITTANCE_THRESHOLD ± ε`. Undecided until measured.
6. **The fp16 path** — the CUDA SH kernel supports half coeffs (AT_DISPATCH fp16/fp32, with
   v_coeffs accumulated in fp32). Phase 1 assumes only fp32 is verified, but if half training is on
   the roadmap because of the iOS memory budget, a half fixture (stored as f32, cast to half inside
   the kernel) will be needed.
7. **Verifying `_rasterize_reference` itself** — the golden-of-the-golden problem. Which of the two
   becomes the formal procedure: the hand-check of `raster_small` (a spreadsheet-level 3-Gaussian
   case), or cross-verification against the Phase 0 C++ CPU reference.
8. **Correcting the tolerance values against measurements** — the rtol/atol in the table are
   theoretical estimates. The procedure for updating the meta.json values after measuring on
   M4/A17 Pro (the two-run noise floor) — when, and by whom — still needs to be decided.

## Verification notes (adversarial verifier)

The result of independently re-reading and cross-checking the originals (`_torch_impl.py`,
`_math.py`, `_constants.py`, `Common.h`, `SphericalHarmonicsCUDA.cu`, `ProjectionEWA3DGSFused.cu`,
`IntersectTile.cu`, `RasterizeToPixels3DGSSerialBatch{Fwd,Bwd}.cu`,
`RasterizeToPixels3DGSDevice.cuh`, `_wrapper.py`, `rendering.py`, `tools/gen_fixtures.py`,
`tools/sxt.py`). The body of the formulas (the SH coefficient constants, the projection
clamps and signs, the isect bit packing, the blending gate semantics, the tolerance design, and the
gen_fixtures extension structure) was confirmed to match the originals. The corrections and
supplements below must be applied.

### Correction 1 (error) — the location of the `isect_offset_encode` CUDA kernel

The spec table's "`isect_offset_encode` (IntersectTile.cu:670-)" is a misprint. **:670 is
`add_counts_kernel`** (a helper for the multi-GPU privateuseone path, :670-684). The actual kernel is
**`intersect_offset_kernel` (IntersectTile.cu:925-988)**, with the launcher
`launch_intersect_offset_kernel` (:990-). The semantic description itself (a global flattened
exclusive prefix, filling 0 before the first valid tile :952-959, filling `n_isects` after the last
tile :960-967, and `offsets.fill_(0)` when `n_isects==0` :1004-1008) is consistent with the torch
golden `_isect_offset_encode` — only the location citation is corrected.

### Correction 2 (major omission) — the torch golden does not cover the AccuTile path of isect_tiles

The CUDA `intersect_tile_kernel` (IntersectTile.cu:214) has **two paths**:

- **AccuTile/SNUGBOX ellipse intersection** (:288-373): when `conics != nullptr && opacities != nullptr`.
  An opacity-aware isocontour `t = min(GAUSSIAN_EXTEND², 2·ln(opacity/ALPHA_THRESHOLD))` (:303), a
  tight AABB, and a line sweep (`accutile_process_tiles`, :340).
- **The AABB fallback** (:374-463): when conics/opacities are not provided. **The torch
  `_isect_tiles` golden models only this path** (the floor/ceil AABB; _torch_impl.py:385-394 and
  IntersectTile.cu:384-387 are 1:1).

The problem: with `with_ut=False` (the default at rendering.py:263), the default gsplat 3DGS
pipeline calls `isect_tiles(..., conics=conics, opacities=opacities)` (rendering.py:884-897,
_wrapper.py:1049-1075) → **the real default path is AccuTile**, and AccuTile produces fewer
intersections than AABB (tiles_per_gauss, isect_ids, and offsets all differ). Therefore:

- This spec's fixture (`isect_tiles_std`) is **valid only when the Metal isect kernel is restricted
  to the AABB contract (conics/opacities not passed)**. State explicitly that the Phase 1 Metal
  kernel scope is the AABB path.
- Porting AccuTile requires an in-house reference implementation (plus a dedicated fixture) — in the
  same category as open question #2 (radius_clip/opacity culling), it is registered as **open
  question #9**: "No golden for the AccuTile path. Phase 1 is restricted to AABB and AccuTile is
  decided in Phase 2".

### Correction 3 (omission) — proj fwd output comparison is valid only under the valid mask

For a culled (cid,gid), the CUDA fused fwd **writes only radii=0 and returns early**
(ProjectionEWA3DGSFused.cu — near/far :100-105, `det<=0` :153-158, opacity culling :172-177,
radius_clip :188-193, the inside test :196-204). means2d/depths/conics/compensations are **not
written** at those positions (the buffer's initial values remain). The torch golden, by contrast,
computes values for every Gaussian and stores them in the fixture. The test harness therefore MUST
**restrict the comparison** of proj fwd's `means2d`/`depths`/`conics`/`compensations` **to the
elements where `valid = (radii > 0).all(-1)`** (the same mask as the backward upstream masking). Add
this condition to the tolerance table. Side notes:

- The `det <= 0` culling that exists only in CUDA (:153-158) is absent from the torch golden (torch
  proceeds after `det.clamp(min=1e-10)`, :315). With PSD covar inputs it never fires because det>0
  after the blur (eps2d=0.3), but assert `min(det_blur) > 0` during fixture generation to guarantee
  that the golden and CUDA behave identically.
- A difference at the boundary equality: CUDA culls `z < near_plane` (keeping z==near, :100), while
  torch keeps `z > near_plane` (culling z==near, :340). far has the same structure. With randn
  inputs this is measure-zero, so there is no actual harm.

### Correction 4 (error) — the `_rasterize_reference` pseudocode: empty tiles + backgrounds

The pseudocode's `if rs == re: continue  # T=1, out=0, last=0 unchanged` is **wrong when backgrounds
are present**. The CUDA fwd's output write (:274-296) is **outside** the batch loop, so
`pix_out + T·bg = bg` is written even for the in-bounds pixels of an empty-range tile (T=1). Fix:
move the background application out of the tile loop into a global step —

```python
# after the tile loop ends (when backgrounds is not None, for every pixel):
out_c = out_c + (1.0 - out_a) * backgrounds[:, None, None, :]
```

Delete the per-tile line `out_c[...] += (1.0 - a_pix)[:,None] * backgrounds[img]`. When the "(e)
empty tile" case of `raster_small` is combined with the backgrounds variant, the golden itself is
wrong without this fix. (render_alphas=0 and last_ids=0 do match CUDA on an empty tile — that part
is fine.)

### Correction 5 (omission) — the covars buffer layout contract ([N,3,3] vs [N,6] triu)

The CUDA fused kernel takes covars as **[.., N, 6] upper-triangular** (ProjectionEWA3DGSFused.cu:45,
mat3 reconstruction at :112-122) and writes v_covars as **6-triu** as well, where the off-diagonals
are the sum of the symmetric components: `v_covars[1] = v_covar[0][1]+v_covar[1][0]` and so on
(:588-593). The torch golden and the existing `proj_fully_fused` fixture store **[N,3,3] full**. If
the Metal kernel follows the CUDA contract (6-triu), the "fixture tensor = kernel buffer contract"
principle breaks, so state the conversion rules explicitly:

- Input: `covars6 = [c00, c01, c02, c11, c12, c22]` (extracted from the full matrix, or generated
  directly with `_quat_scale_to_covar_preci(..., triu=True)`, _math.py:705-709).
- Expected gradient: `v_covars6 = [v00, v01+v10, v02+v20, v11, v12+v21, v22]` (converted from the
  torch golden's [N,3,3] gradient).
- Prefer additionally storing `covars_triu.sxt` / `v_covars_triu.sxt` in the fixture (keep the
  existing `covars.sxt` for Phase 0 compatibility).

### Correction 6 (omission, one line) — the tile masks argument of raster

The CUDA raster fwd/bwd have a per-tile `masks` option (Fwd.cu:51; for a masked-off tile it writes
bg/0/0 and returns, :141-159. Bwd.cu:53; it simply returns, :106-109). As with SH, state explicitly
that **the v1 fixture fixes masks=None** (the golden `_rasterize_reference` does not model masks).

### Correction 7 (minor) — line-citation refinements

- The citation ":270-281" backing "for a clamped Gaussian only v_rgb remains" in Bwd.cu →
  **:232-237** (`fac = alpha*T` at :232, `v_rgb_local[k] = fac * v_render_c[k]` at :236 — outside
  the gate at :259). :270-281 is the v_xy_abs/v_opacity code (inside the gate) and the buffer
  update. The claim itself is correct.
- The `v_means2d_abs` declaration is at Bwd.cu **:70** (the spec says :69).
- The v_quats/v_scales atomics of proj bwd are at **:596-617** (the ":577-596" in the spec's MSL
  section covers only v_means (:568-580) and v_covars (:581-595)). Also, CUDA performs a
  `cg::labeled_partition(warp, gid)` warp pre-reduction before the atomics (:566-567) — if the first
  Metal implementation uses 1:1 atomics, the logical number of accumulations is the same, so the
  tolerance rationale remains valid.
- `eval_gaussian_weight` is at Device.cuh **:44-55** (the spec says :45-56).
- The pixel centers in Fwd.cu: px at **:108**, py at **:117** (the spec says "~118").
- The early-exit vote is at Fwd.cu **:268**, `__syncthreads_count(done_mask == ALL_DONE) >= BATCH_SIZE`
  (the spec says ":270" and "==256" — the count cannot exceed 256, so the meaning is the same).

### Correction 8 (minor) — the basis for the SH bwd atomic counts

The tolerance table's "at most B(=2) atomics per element" is accurate only for `v_coeffs` (B batch
threads per element, SphericalHarmonicsCUDA.cu:159-361). **`v_dirs` gets D(=3) per element** — the D
channel threads of the same (batch, gaussian) each atomicAdd into x/y/z (:756-761). The numerical
impact is negligible so the table values can stand; only the basis is corrected.

### Items confirmed as passing verification (no correction needed)

- SH: the signatures of `_spherical_harmonics`/`_eval_sh_bases_fast`, the coefficient constants, the
  line numbers of the degree branches, the F.normalize vs rsqrtf difference and the norm≥0.5
  guarantee strategy, the normalization vjp expression (identical at :186/:422-424), and the
  automatic agreement of grad 0 for higher degrees — all match the originals.
- Projection: every line number, sign, and clamp of the `_fully_fused_projection` formula block
  (det, eps2d, compensations, conics, radius, and the gates) matches. The conic order
  (covar2d_inv[0][0], [0][1], [1][1]) matches. The upstream masking strategy based on the radii gate
  (:422-425) is sound.
- isect: the bit packing (hi=(iid<<tile_n_bits)|tile_id, key=(hi<<32)|depth_bits) is identical to
  CUDA's (:273, :285, :455-457). The 2^n handling of `_bits_for_count` matches (corresponding to
  bits_for_count in MathUtils.h). The arithmetic 80 tiles → tile_n_bits=7, image_n_bits=1 was
  confirmed. The premise that positive depths are monotonic is pinned in CUDA too by
  `assert(depth_f >= 0.f)` (:283).
- raster: the finding that `_rasterize_to_pixels` cannot be used directly was confirmed (the
  rasterize_to_indices_in_range import at :849, nerfacc at :763-766). The exclusive early
  termination (:246-251), last_ids = the batch-global index (:261) with an initial value of 0
  (:186), the equivalence `w.sum == 1-T_final`, the equivalence between the autograd of the
  MAX_ALPHA clamp and the gate (:259), and the dead `ra` guard (1-alpha ≥ 0.01 > 1e-6) — all
  confirmed.
- The threadgroup memory computation (fwd 7,168 B / bwd 10,240 B) matches the CUDA shmem formulas
  (Fwd.cu:346, Bwd.cu:384-385). For the thread configuration, CUDA's bwd is 2D
  (tile_size, tile_size, 1) (:361) while fwd is 1D CTA=256 — unifying on (16,16,1) in MSL is sound
  because the linearization order (x-fastest) is isomorphic to the CUDA warp layout.
- The existing infrastructure: gen_fixtures.py (stub loading at :23-50, seeds 42/7,
  calc_compensations=True) and sxt.py (SXT1, dtype 0/1, shape u64) — both match the spec's
  description.
