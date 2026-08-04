# Kernel spec 01 — EWA Projection Forward/Backward (3DGS Fused)

CUDA → Metal porting spec. This document extracts gsplat's `projection_ewa_3dgs_fused_fwd_kernel` / `projection_ewa_3dgs_fused_bwd_kernel` at code-level precision. Upstream file:line references are given alongside so that every formula can be compared against the Phase 0 CPU reference (`Sources/SplatXCore/CpuKernels.cpp`, `projection_ewa_fwd_cpu`).

---

## Upstream files and commit

- Commit pinned: **gsplat `77ab983ffe43420b2131669cb35776b883ca4c3c`** (Apache-2.0, derivation notice: `NOTICE` / `docs/provenance.md`)
- Main files:
  - `research/gsplat/gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu`
    - fwd kernel: lines 38–219, fwd launcher: lines 221–294 (threads=256: line 250)
    - bwd kernel: lines 376–638, bwd launcher: lines 640–723 (threads=256: line 676)
  - `research/gsplat/gsplat/cuda/include/Utils.cuh` — every math helper and VJP lives here
    - `posW2C` 81–89, `posW2C_VJP` 94–113, `covarW2C` 116–123, `covarW2C_VJP` 128–146
    - the `warpSum` family 152–222
    - `quat_to_rotmat` 228–251, `quat_to_rotmat_vjp` 253–283
    - `quat_scale_to_covar_preci` 285–311, `quat_scale_to_covar_vjp` 313–347
    - `inverse_vjp` 448–453, `add_blur` 455–463, `add_blur_vjp` 465–492
    - `ortho_proj` 498–526, `ortho_proj_vjp` 528–565
    - `persp_proj` 567–607, `persp_proj_vjp` 609–690
    - `fisheye_proj` 692–731, `fisheye_proj_vjp` 733–841
  - `research/gsplat/gsplat/cuda/include/Common.h` — glm typedefs (64–70), `CameraModelType` (75–82), constants (97–114)
  - `research/gsplat/gsplat/cuda/csrc/Projection.cpp` — tensor allocation semantics (fwd outputs `at::empty`, only `compensations` uses `at::zeros`: lines 395–411; all bwd gradient outputs use `at::zeros_like`: lines 641–655)
- Reference only (not extracted): `ProjectionEWA3DGSPacked.cu` (a 2-pass variant for sparse output, outside the Phase 1 scope)
- **Note**: `Cameras.cuh` is not used by this fused kernel. The fused kernel includes only `Common.h`, `Projection.h`, and `Utils.cuh` (`ProjectionEWA3DGSFused.cu:29–31`), and the projection functions (`persp_proj` and friends) all live in `Utils.cuh`. `Cameras.cuh` is for the newer UT kernel path (`ProjectionUT3DGSFused.cu`).

### Constants (Common.h:97–114)

```c
#define ALPHA_THRESHOLD  (1.f / 255.f)   // ≈ 0.003921569
#define GAUSSIAN_EXTEND  3.33f           // number of standard deviations at which the Gaussian is truncated
#define MIN_COMPENSATION 0.005f          // lower bound of the anti-aliasing compensation term
```

### glm notation convention (important)

- `Common.h:64–70`: `vec2/vec3/vec4`, `mat2/mat3/mat4`, `mat3x2 = glm::mat<3,2,float>` (**3 columns × 2 rows**).
- glm is **column-major**: `M[col][row]`. Constructor arguments are in column order as well.
- MSL's `float3x3` is likewise column-major with `M[col][row]` indexing → **the index notation ports 1:1** (see the MSL porting section below).
- The inputs `viewmats`, `Ks`, and `covars` are **stored row-major**, so the kernel rearranges them into column-major on load.

---

## Algorithm overview

One thread per Gaussian (precisely, one thread per (batch, camera, gaussian) combination):

1. **Forward**: world-space 3D Gaussian (mean μ, covar Σ, or quat+scale) → transform into camera space → per-camera-model EWA projection (project Σ into 2D with the Jacobian J) → add the `eps2d` blur (plus the anti-aliasing compensation term) → conic (2D inverse covariance) → opacity-aware radius → culling → write `radii/means2d/depths/conics/compensations`.
2. **Backward**: take the upstream gradients `v_means2d / v_depths / v_conics / v_compensations` and propagate the VJP backwards through the forward chain → output `v_means / v_covars` (or `v_quats + v_scales`) `/ v_viewmats`. Intermediates (covar_c, mean_c, and so on) are not stored; the **forward is recomputed** (rematerialization). Output accumulation uses a warp reduction plus float atomicAdd.

Batch axis: the tensors are `[B, C, N, ...]` (B = batch, C = camera, N = Gaussian). splatx Phase 1 defaults to B=1, C=1, but the indexing expressions are ported as-is.

---

## Formula detail — Forward

Kernel: `projection_ewa_3dgs_fused_fwd_kernel` (`ProjectionEWA3DGSFused.cu:38–219`)

### F0. Thread indexing (lines 66–81)

```
idx  = global thread id;  count = B * C * gaussian_count;  if (idx >= count) return;
bid  = idx / (C * gaussian_count)                  // batch id
cid  = (idx / gaussian_count) % C                  // camera id
gid  = idx % gaussian_count + gaussian_offset      // gaussian id (on a single GPU, offset=0 and count=N)
global_idx = (bid * C + cid) * N + gid
means    += bid*N*3 + gid*3;   viewmats += bid*C*16 + cid*16;   Ks += bid*C*9 + cid*9;
```

- Adjacent threads = adjacent Gaussians of the same camera (gid is the fastest axis) → coalesced loads.
- **Output on culling**: writes only `radii[global_idx*2] = radii[global_idx*2+1] = 0` and returns. The remaining outputs (`means2d/depths/conics`) are **unwritten** (the host allocates them with `at::empty`, `Projection.cpp:395–404`; only `compensations` uses `at::zeros`, line 411). → When porting to Metal, leaving the other output buffers undefined (garbage) for slots with radii=0 is identical by convention.

### F1. world → camera (lines 83–105)

`viewmats` is row-major `[R|t; 0 0 0 1]` (world→cam). Rebuilt in glm column-major form:

```
R = mat3( viewmats[0], viewmats[4], viewmats[8],    // 1st col = (m00, m10, m20)
          viewmats[1], viewmats[5], viewmats[9],    // 2nd col
          viewmats[2], viewmats[6], viewmats[10] )  // 3rd col
t = vec3( viewmats[3], viewmats[7], viewmats[11] )
mean_c = R * glm::make_vec3(means) + t              // posW2C, Utils.cuh:81–89
```

**near/far culling** (lines 100–105):

```
if (mean_c.z < near_plane || mean_c.z > far_plane) { radii = 0; return; }
```

(Boundary inclusion: `<`, `>` — equality survives.)

### F2. Source of the 3D covariance (lines 107–130)

One of two paths:

- **covars given directly** (`covars != nullptr`, lines 109–123): `[B,N,6]` upper-triangular (triu) packing `(c00, c01, c02, c11, c12, c22)` → restore the symmetric mat3:
  ```
  covar = mat3( covars[0], covars[1], covars[2],
                covars[1], covars[3], covars[4],
                covars[2], covars[4], covars[5] )
  ```
  (Note: the Phase 0 CPU reference `CpuKernels.h:11` takes a full 3×3 row-major matrix — numerically identical because the matrix is symmetric; only the packing differs.)
- **quat + scale** (lines 124–130): `quat_scale_to_covar_preci(quat, scale, &covar, nullptr)` (`Utils.cuh:285–311`):
  ```
  R_g = quat_to_rotmat(quat)          // Utils.cuh:228–251, see B7 below. quat storage order = [w, x, y, z]
  S   = diag(scale[0], scale[1], scale[2])
  M   = R_g * S
  covar = M * Mᵀ                       // = R_g S S R_gᵀ
  ```
  `quat_to_rotmat` internally **normalizes** with `inv_norm = rsqrt(w²+x²+y²+z²)` and then builds the standard rotation matrix (column-major):
  ```
  R_g = [ 1-2(y²+z²)   2(xy-wz)     2(xz+wy)
          2(xy+wz)     1-2(x²+z²)   2(yz-wx)
          2(xz-wy)     2(yz+wx)     1-2(x²+y²) ]   // math notation (row, col). The glm constructor fills column by column
  ```

### F3. covar world → camera (lines 131–132, Utils.cuh:116–123)

```
covar_c = R * covar * Rᵀ              // covarW2C
```

### F4. Camera projection → (covar2d, mean2d) (lines 134–149)

`fx = Ks[0], fy = Ks[4], cx = Ks[2], cy = Ks[5]` (Ks is a row-major 3×3).

#### F4a. PINHOLE — `persp_proj` (Utils.cuh:567–607)

```
x = mean_c.x; y = mean_c.y; z = mean_c.z;
tan_fovx  = 0.5f * width  / fx;
tan_fovy  = 0.5f * height / fy;
lim_x_pos = (width  - cx) / fx + 0.3f * tan_fovx;
lim_x_neg =  cx           / fx + 0.3f * tan_fovx;
lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
lim_y_neg =  cy           / fy + 0.3f * tan_fovy;

rz  = 1.f / z;   rz2 = rz * rz;
tx  = z * min(lim_x_pos, max(-lim_x_neg, x * rz));   // FOV clamp (0.3·tan_fov margin)
ty  = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

// J: 2×3 in math notation (glm mat3x2 = 3 cols × 2 rows, J[col][row])
//        row0: [ fx*rz     0       -fx*tx*rz2 ]
//        row1: [ 0         fy*rz   -fy*ty*rz2 ]
J = mat3x2( fx*rz, 0.f,   0.f, fy*rz,   -fx*tx*rz2, -fy*ty*rz2 );

covar2d = J * covar_c * Jᵀ            // 2×2
mean2d  = vec2( fx * x * rz + cx,  fy * y * rz + cy );
```

**Note**: `mean2d` uses the unclamped `x*rz`, while only the 3rd column of `J` uses the clamped `tx, ty`. (This pairs with the branch in the backward — see B4.)

#### F4b. ORTHO — `ortho_proj` (Utils.cuh:498–526)

```
J = mat3x2( fx, 0,   0, fy,   0, 0 );
covar2d = J * covar_c * Jᵀ;           // = [[fx²·Σ00, fx·fy·Σ01], [fx·fy·Σ10, fy²·Σ11]]
mean2d  = vec2( fx * x + cx,  fy * y + cy );   // z unused
```

#### F4c. FISHEYE — `fisheye_proj` (Utils.cuh:692–731)

```
eps = 1e-7f;
xy_len = length(vec2(x, y)) + eps;
theta  = atan2(xy_len, z + eps);
mean2d = vec2( x * fx * theta / xy_len + cx,  y * fy * theta / xy_len + cy );

x2 = x*x + eps;  y2 = y*y;  xy = x*y;  x2y2 = x2 + y2;
x2y2z2_inv = 1.f / (x2y2 + z*z);
b = atan2(xy_len, z) / xy_len / x2y2;
a = z * x2y2z2_inv / x2y2;
J = mat3x2( fx*(x2*a + y2*b),  fy*xy*(a-b),        // 1st col
            fx*xy*(a-b),       fy*(y2*a + x2*b),   // 2nd col
            -fx*x*x2y2z2_inv,  -fy*y*x2y2z2_inv ); // 3rd col
covar2d = J * covar_c * Jᵀ;
```

(The enum's `FTHETA` and `LIDAR` do not appear in this kernel's switch — unsupported.)

### F5. eps2d blur + compensation term — `add_blur` (lines 151–158, Utils.cuh:455–463)

```
det_orig = covar2d[0][0]*covar2d[1][1] - covar2d[0][1]*covar2d[1][0];
covar2d[0][0] += eps2d;                    // in-place! from here on covar2d is the blurred value
covar2d[1][1] += eps2d;
det_blur = covar2d[0][0]*covar2d[1][1] - covar2d[0][1]*covar2d[1][0];
compensation = sqrtf( max(MIN_COMPENSATION * MIN_COMPENSATION, det_orig / det_blur) );
det = det_blur;
if (det <= 0.f) { radii = 0; return; }     // culling (lines 153–158)
```

- Default `eps2d = 0.3` (the Python API default; the CPU reference `CpuKernels.h:16` is 0.3 as well).
- The lower bound of `compensation` is the sqrt of `0.005² = 2.5e-5`, i.e. 0.005.

### F6. conic (lines 160–161)

```
covar2d_inv = glm::inverse(covar2d);       // inverse of the blurred covar2d (2×2 symmetric)
// output packing: conics = (covar2d_inv[0][0], covar2d_inv[0][1], covar2d_inv[1][1])  // (a, b, c) triu
```

glm's `inverse(mat2)` is the analytic formula `1/det * [[d,-b],[-c,a]]`. det was already confirmed `> 0` in F5.

### F7. opacity-aware radius + culling (lines 163–204)

```
extend = GAUSSIAN_EXTEND;                  // 3.33f
if (opacities != nullptr) {
    opacity = opacities[bid*N + gid];
    if (compensations != nullptr)          // only in anti-aliasing mode
        opacity *= compensation;
    if (opacity < ALPHA_THRESHOLD) { radii = 0; return; }      // lines 172–177
    // opacity-aware bbox — arXiv:2402.00525 §B.2
    extend = min(GAUSSIAN_EXTEND, sqrt(2.0f * __logf(opacity / ALPHA_THRESHOLD)));  // line 180
}
radius_x = ceilf(extend * sqrtf(covar2d[0][0]));   // sqrt of the blurred variance (line 185)
radius_y = ceilf(extend * sqrtf(covar2d[1][1]));   // (line 186)

if (radius_x <= radius_clip && radius_y <= radius_clip) { radii = 0; return; }   // culled only when both are small (line 188)

// cull outside the image area (lines 196–199) — culls with boundary equality included
if (mean2d.x + radius_x <= 0 || mean2d.x - radius_x >= image_width ||
    mean2d.y + radius_y <= 0 || mean2d.y - radius_y >= image_height) { radii = 0; return; }
```

- `__logf` is the CUDA fast-math natural log. In MSL it maps to `log()` (fast-math by default).
- The `radius_clip` condition is an **AND**: if either one is `> radius_clip`, the Gaussian survives.

### F8. Outputs (lines 206–218)

```
radii[gi*2] = (int32_t)radius_x;  radii[gi*2+1] = (int32_t)radius_y;
means2d[gi*2] = mean2d.x;  means2d[gi*2+1] = mean2d.y;
depths[gi]    = mean_c.z;                        // camera z (not the Euclidean distance)
conics[gi*3]  = covar2d_inv[0][0];  conics[gi*3+1] = covar2d_inv[0][1];  conics[gi*3+2] = covar2d_inv[1][1];
if (compensations != nullptr) compensations[gi] = compensation;
```

---

## Formula detail — Backward

Kernel: `projection_ewa_3dgs_fused_bwd_kernel` (`ProjectionEWA3DGSFused.cu:376–638`)

Inputs: all forward inputs + `radii, conics, compensations` out of the forward outputs + the upstream gradients `v_means2d[B,C,N,2], v_depths[B,C,N], v_conics[B,C,N,3], v_compensations[B,C,N] (optional)`.
Outputs (all **accumulated with atomicAdd into zero-initialized buffers**, `Projection.cpp:641–655`): `v_means[B,N,3]`, plus `v_covars[B,N,6]` on the covars path or `v_quats[B,N,4] + v_scales[B,N,3]` on the quat/scale path, and optionally `v_viewmats[B,C,16]`.

### B0. Indexing + early exit (lines 411–425)

Indexing is identical to the forward. Additionally:

```
if (radii[global_idx*2] <= 0 || radii[global_idx*2+1] <= 0) return;   // skip Gaussians culled in the forward
```

### B1. conic → covar2d VJP — `inverse_vjp` (lines 438–442, Utils.cuh:448–453)

Restores the forward's `covar2d_inv` from the conics output (it is not recomputed):

```
covar2d_inv   = mat2( conics[0], conics[1],  conics[1], conics[2] );
v_covar2d_inv = mat2( v_conics[0], v_conics[1] * .5f,  v_conics[1] * .5f, v_conics[2] );
//   ↑ 0.5 on the off-diagonal: conic's b stands for two slots of the symmetric matrix, so the gradient is split in half
v_covar2d = mat2(0.f);
v_covar2d += -covar2d_inv * v_covar2d_inv * covar2d_inv;    // P = M⁻¹ ⇒ dL/dM = -P·(dL/dP)·P
```

### B2. compensation VJP — `add_blur_vjp` (lines 444–450, Utils.cuh:465–492)

Only when `v_compensations` is present. `conic_blur = covar2d_inv` (the inverse of the blurred covar), and `compensation` is the value stored by the forward:

```
det_conic_blur     = conic_blur[0][0]*conic_blur[1][1] - conic_blur[0][1]*conic_blur[1][0];
v_sqr_comp         = v_compensation * 0.5 / (compensation + 1e-6);     // d(comp)/d(comp²) chain + divide-by-zero guard
one_minus_sqr_comp = 1 - compensation * compensation;
v_covar2d[0][0] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][0] - eps2d * det_conic_blur);
v_covar2d[0][1] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][1]);
v_covar2d[1][0] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][0]);
v_covar2d[1][1] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][1] - eps2d * det_conic_blur);
```

Derivation (verbatim from the comment, Utils.cuh:469–483): for `comp² = det(M) / det(M + eps2d·I)`,
`d(comp²)/dM = (1 - comp²)·conic_blur - eps2d·det(conic_blur)·I`.
**Note**: when the forward's `MIN_COMPENSATION` clamp is active (that is, the value hit the lower bound), the backward ignores the clamp and applies the formula above unchanged (exactly as in the upstream original — a subgradient mismatch at the clamp boundary is accepted).

### B3. Forward recomputation (lines 452–494)

Load `R, t`, restore `covar` (from covars or via `quat_scale_to_covar_preci`), `mean_c = R·μ + t`, `covar_c = R·covar·Rᵀ` — the same code as forward F1–F3, re-executed.

### B4. Projection VJP → (v_mean_c, v_covar_c) (lines 496–551)

Starts from `v_covar_c = mat3(0)`, `v_mean_c = vec3(0)`.

#### B4a. PINHOLE — `persp_proj_vjp` (Utils.cuh:609–690)

After recomputing all of `tan_fov*, lim_*, rz, rz2, tx, ty, J` from forward F4a:

```
// (1) covar path: cov2d = J·V·Jᵀ ⇒ dL/dV = Jᵀ·G·J
v_cov3d += Jᵀ * v_cov2d * J;                          // line 654

// (2) mean2d path (uses the unclamped x, y):
v_mean3d += vec3( fx * rz * v_mean2d[0],
                  fy * rz * v_mean2d[1],
                  -(fx * x * v_mean2d[0] + fy * y * v_mean2d[1]) * rz2 );   // lines 659–660

// (3) J path: dL/dJ = G·J·Vᵀ + Gᵀ·J·V   (mat3x2, v_J[col][row])
rz3 = rz2 * rz;
v_J = v_cov2d * J * cov3dᵀ + v_cov2dᵀ * J * cov3d;    // line 667

// (4) FOV clamp branch (lines 670–685): since tx = z·clamp(x/z)
if (x*rz <= lim_x_pos && x*rz >= -lim_x_neg)   // clamp inactive: d(tx)/dx = 1
    v_mean3d.x += -fx * rz2 * v_J[2][0];       // v_J[2][0] = dL/dJ_02 (row 0, col 2)
else                                           // clamp active:   d(tx)/dz = tx/z
    v_mean3d.z += -fx * rz3 * v_J[2][0] * tx;
if (y*rz <= lim_y_pos && y*rz >= -lim_y_neg)
    v_mean3d.y += -fy * rz2 * v_J[2][1];       // v_J[2][1] = dL/dJ_12
else
    v_mean3d.z += -fy * rz3 * v_J[2][1] * ty;

// (5) derivatives of the J elements with respect to z (lines 686–689):
v_mean3d.z += -fx * rz2 * v_J[0][0]            // dJ_00/dz = -fx·rz²
            - fy * rz2 * v_J[1][1]             // dJ_11/dz = -fy·rz²
            + 2.f * fx * tx * rz3 * v_J[2][0]  // dJ_02/dz = 2·fx·tx·rz³ (tx held fixed)
            + 2.f * fy * ty * rz3 * v_J[2][1];
```

Sign and index check: since `J_02 = -fx·tx·rz²`, `∂J_02/∂tx = -fx·rz²` and `∂tx/∂x = 1` (unclamped) → the term in `(4)`. When clamped, `tx = z·lim` is a constant × z → `∂tx/∂z = tx/z` → `-fx·rz²·(tx/z) = -fx·rz³·tx`. `(5)` combines `∂(rz)/∂z = -rz²` with `∂(tx·rz²)/∂z|_{tx fixed}`.

#### B4b. ORTHO — `ortho_proj_vjp` (Utils.cuh:528–565)

```
v_cov3d  += Jᵀ * v_cov2d * J;                 // J = [[fx,0,0],[0,fy,0]]
v_mean3d += vec3( fx * v_mean2d[0], fy * v_mean2d[1], 0.f );   // J is constant, so there is no v_J path
```

#### B4c. FISHEYE — `fisheye_proj_vjp` (Utils.cuh:733–841)

mean path: uses the forward's J (the a, b formulas) as-is, in the form `v_mean3d += Jᵀ·v_mean2d` (lines 763–767).
covar path: `v_cov3d += Jᵀ·v_cov2d·J` (line 781) plus `v_J = v_cov2d·J·cov3dᵀ + v_cov2dᵀ·J·cov3d` (line 783), then the partial derivatives of the J elements with respect to (x,y,z) are expanded analytically (lines 784–840). Intermediate terms:

```
l4 = x2y2z2²;  theta = atan2(len_xy, z);
E = -l4·x2y2·theta + x2y2z2·x2y2·len_xy·z
F = 3·l4·theta - 3·x2y2z2·len_xy·z - 2·x2y2·len_xy·z
A = x(3E + x2·F);  B = y(E + x2·F);  C = x(E + y2·F);  D = y(3E + y2·F)
S1 = x2 - y2 - z²;  S2 = y2 - x2 - z²;  inv1 = x2y2z2_inv²;  inv2 = inv1 / (x2y2²·len_xy)
dJ_dx00 = fx·A·inv2, dJ_dx01 = fx·B·inv2, dJ_dx02 = fx·S1·inv1,
dJ_dx10 = fy·B·inv2, dJ_dx11 = fy·C·inv2, dJ_dx12 = 2·fy·xy·inv1,
dJ_dy00 = dJ_dx01, dJ_dy01 = fx·C·inv2, dJ_dy02 = 2·fx·xy·inv1,
dJ_dy10 = dJ_dx11, dJ_dy11 = fy·D·inv2, dJ_dy12 = fy·S2·inv1,
dJ_dz00 = dJ_dx02, dJ_dz01 = dJ_dy02, dJ_dz02 = 2·fx·x·z·inv1,
dJ_dz10 = dJ_dx12, dJ_dz11 = dJ_dy12, dJ_dz12 = 2·fy·y·z·inv1
v_mean3d.{x,y,z} += Σ_{rc} dJ_d{x,y,z}rc · v_J[c][r]      // lines 820–840
```

### B5. depth contribution (lines 553–554)

```
v_mean_c.z += v_depths[0];        // depths was mean_c.z, so this is a plain add
```

### B6. camera → world VJP (lines 556–562)

```
v_mean = vec3(0);  v_covar = mat3(0);  v_R = mat3(0);  v_t = vec3(0);

// posW2C_VJP (Utils.cuh:94–113): pC = R·pW + t
v_R  += glm::outerProduct(v_mean_c, μ_world);   // v_R[j][i] = v_mean_c[i]·μ[j] ⇒ dL/dR_ij = v_pC_i·pW_j
v_t  += v_mean_c;
v_mean += Rᵀ * v_mean_c;

// covarW2C_VJP (Utils.cuh:128–146): covarC = R·covarW·Rᵀ
v_R     += v_covar_c * R * covarᵀ + v_covar_cᵀ * R * covar;   // covar is symmetric so ᵀ could be dropped, but the original is kept
v_covar += Rᵀ * v_covar_c * R;
```

### B7. covar → (quat, scale) chain (lines 596–617) — the quat/scale parameterization path

When `covars == nullptr`:

```
rotmat = quat_to_rotmat(quat);          // includes the same normalization as the forward (see F2)
v_quat = vec4(0);  v_scale = vec3(0);
quat_scale_to_covar_vjp(quat, scale, rotmat, v_covar, v_quat, v_scale);   // Utils.cuh:313–347
```

`quat_scale_to_covar_vjp` expanded:

```
S = diag(sx, sy, sz);   M = R_g * S;    // covar = M·Mᵀ
// D = M·Mᵀ, G = v_covar ⇒ dL/dM = (G + Gᵀ)·M
v_M = (v_covar + v_covarᵀ) * M;
v_R_g = v_M * S;                        // M = R_g·S ⇒ dL/dR_g = (dL/dM)·Sᵀ = v_M·S
quat_to_rotmat_vjp(quat, v_R_g, v_quat);
// the diagonal components of dL/dS = dot product of column j of R_g with column j of v_M (glm column index)
v_scale[0] += R[0][0]*v_M[0][0] + R[0][1]*v_M[0][1] + R[0][2]*v_M[0][2];  // dot(col0(R_g), col0(v_M))
v_scale[1] += R[1][0]*v_M[1][0] + R[1][1]*v_M[1][1] + R[1][2]*v_M[1][2];
v_scale[2] += R[2][0]*v_M[2][0] + R[2][1]*v_M[2][1] + R[2][2]*v_M[2][2];
```

`quat_to_rotmat_vjp` (Utils.cuh:253–283) — **includes the VJP of the normalization**:

```
w,x,y,z = quat[0..3];  inv_norm = rsqrt(x²+y²+z²+w²);  (w,x,y,z) *= inv_norm;   // computed with the normalized values
// v_R indices are glm [col][row]
v_quat_n = vec4(
  2·( x·(v_R[1][2] - v_R[2][1]) + y·(v_R[2][0] - v_R[0][2]) + z·(v_R[0][1] - v_R[1][0]) ),   // dL/dw
  2·( -2x·(v_R[1][1] + v_R[2][2]) + y·(v_R[0][1] + v_R[1][0]) + z·(v_R[0][2] + v_R[2][0]) + w·(v_R[1][2] - v_R[2][1]) ),  // dL/dx
  2·( x·(v_R[0][1] + v_R[1][0]) - 2y·(v_R[0][0] + v_R[2][2]) + z·(v_R[1][2] + v_R[2][1]) + w·(v_R[2][0] - v_R[0][2]) ),   // dL/dy
  2·( x·(v_R[0][2] + v_R[2][0]) + y·(v_R[1][2] + v_R[2][1]) - 2z·(v_R[0][0] + v_R[1][1]) + w·(v_R[0][1] - v_R[1][0]) )    // dL/dz
);
quat_n = vec4(w, x, y, z);                       // the normalized quaternion, in [w,x,y,z] order
v_quat += ( v_quat_n - dot(v_quat_n, quat_n) * quat_n ) * inv_norm;   // projection onto the unit-sphere tangent plane × 1/‖q‖
```

### B8. Output accumulation — warp reduction + atomicAdd (lines 564–637)

```
warp         = cg::tiled_partition<32>(block);
warp_group_g = cg::labeled_partition(warp, gid);     // groups the lanes that share the same gid

if (v_means != nullptr) {
    warpSum(v_mean, warp_group_g);                   // sum within the group (Utils.cuh:179–184)
    if (warp_group_g.thread_rank() == 0)
        for i in 0..2: gpuAtomicAdd(&v_means[bid*N*3 + gid*3 + i], v_mean[i]);   // float atomicAdd
}
if (v_covars != nullptr) {           // covars path: symmetric components merged and packed into 6 (lines 588–593)
    warpSum(v_covar, warp_group_g);
    if (leader) {
        atomicAdd(v_covars+0, v_covar[0][0]);
        atomicAdd(v_covars+1, v_covar[0][1] + v_covar[1][0]);   // sums the two off-diagonal slots!
        atomicAdd(v_covars+2, v_covar[0][2] + v_covar[2][0]);
        atomicAdd(v_covars+3, v_covar[1][1]);
        atomicAdd(v_covars+4, v_covar[1][2] + v_covar[2][1]);
        atomicAdd(v_covars+5, v_covar[2][2]);
    }
} else {                             // quat/scale path (lines 596–617): after performing B7
    warpSum(v_quat, warp_group_g);  warpSum(v_scale, warp_group_g);
    if (leader) { atomicAdd ×4 (v_quats), atomicAdd ×3 (v_scales); }
}
if (v_viewmats != nullptr) {         // (lines 618–637)
    warp_group_c = cg::labeled_partition(warp, cid); // lanes that share the same camera
    warpSum(v_R, warp_group_c);  warpSum(v_t, warp_group_c);
    if (leader)
        for i(row) in 0..2: { for j(col) in 0..2: atomicAdd_system(&v_viewmats[i*4+j], v_R[j][i]);  // glm[col][row] → row-major storage
                              atomicAdd_system(&v_viewmats[i*4+3], v_t[i]); }
}
```

What the reduction means:
- Because the thread layout is gid-fastest, **a gid repeats inside one warp only when `gaussian_count < 32` or at a boundary wrap** — so labeled_partition for the Gaussian gradients yields groups of size 1 in practice (it is there to guarantee correctness).
- By contrast, **cid is identical across nearly every lane of a warp** → the `warp_group_c` reduction is the key optimization that cuts v_viewmats atomic traffic to about 1/32.
- `atomicAdd_system` is for multi-GPU (NVLink) system-scope atomicity. On single-device Metal a device-scope atomic is sufficient.

---

## Thread and memory configuration

| Item | Forward | Backward |
|---|---|---|
| Parallelization unit | one thread per (bid, cid, gid), `B*C*N` total | same |
| Block / threadgroup | 256 threads (`ProjectionEWA3DGSFused.cu:250, 676`), 1D grid | same |
| shared/threadgroup memory | none | none |
| Barriers | none | none (the warp reduction is cg shuffle-based) |
| atomics | none (output slots are 1:1) | float `gpuAtomicAdd`(v_means/v_covars/v_quats/v_scales), `atomicAdd_system`(v_viewmats) |
| warp primitives | none | `cg::tiled_partition<32>` + `cg::labeled_partition` + `cg::reduce(plus)` |
| Gradient buffer initialization | — | `zeros_like` on the host is mandatory (`Projection.cpp:641–655`) |
| Precision | the kernel is templated on `scalar_t`, but all internal math is float32 glm (`Common.h:64`). Only loads/stores use scalar_t | same |
| fast-math points | `__logf` (line 180), `rsqrt` (quat normalization), `ceilf`, `sqrtf` | `rsqrt`, many divisions |

Summary of the numerical-stability tricks:
1. near/far culling avoids dividing by `z ≈ 0` (`rz = 1/z` is only reached after passing near_plane).
2. The FOV clamp on persp's `tx, ty` (margin 0.3·tan_fov) — prevents the Jacobian from blowing up for Gaussians thrown far off screen. The backward routes the gradient differently depending on whether the clamp is active.
3. `det_blur <= 0` culling — guarantees the conic inverse is safe.
4. Lower bound `MIN_COMPENSATION=0.005` on `compensation` (forward), and the backward denominator `compensation + 1e-6`.
5. fisheye's `eps = 1e-7` (added to xy_len, z+eps, and x2).
6. quat is `rsqrt`-normalized on every load — unnormalized quaternion inputs are allowed, and the VJP includes the normalization chain too.
7. The `radii <= 0` early exit (bwd) blocks garbage propagation from the slots the forward culled (whose outputs are unwritten).

---

## MSL porting design

### Type mapping (glm → simd/MSL)

| glm | MSL | Notes |
|---|---|---|
| `vec2/3/4` | `float2/3/4` | same |
| `mat2` | `float2x2` | both are column-major, `M[col][row]` — **the indices port as-is** |
| `mat3` | `float3x3` | Same. But MSL's `float3x3` has float3 columns (16B aligned), so its **register/memory layout is 48B** — keep buffer I/O as scalar arrays and assemble on load (the upstream original uses the same pattern) |
| `mat3x2` | `float3x2` | MSL `floatNxM` = N columns, M rows — the same convention as glm `mat3x2` (3 cols, 2 rows). The `J[2][0]` index carries over unchanged |
| `glm::transpose` | `transpose()` | same |
| `glm::inverse(mat2)` | Prefer a manual implementation: `mat2_inv(M) = (1/det) * float2x2(float2(M[1][1], -M[0][1]), float2(-M[1][0], M[0][0]))` — MSL has no built-in matrix `inverse()` |
| `glm::outerProduct(c, r)` | Manual: `float3x3(c*r.x, c*r.y, c*r.z)` (column j = c·r[j]) |
| `glm::dot`, `glm::length` | `dot`, `length` | same |
| `__logf` | `log` (fast-math by default) or `metal::fast::log` | compare the error against `precise::log` during verification |
| `rsqrt` | `rsqrt` | same |
| `gpuAtomicAdd(float*)` | `atomic_fetch_add_explicit((device atomic_float*), v, memory_order_relaxed)` | native atomic_float confirmed on M4/A17 Pro — no CAS fallback needed |
| `atomicAdd_system` | same as above (single device, so system scope is unnecessary) |
| `cg::reduce(warp, v, plus)` | `simd_sum(v)` | simdgroup width fixed at 32 — 1:1 with a CUDA warp |
| `cg::labeled_partition` | **no direct equivalent** — see the reduction strategy below |

Precision: everything is `float` (32-bit). The upstream original also does its internal math in float32, so nothing is lost. Do not use half (gradient precision). Use the Metal compiler's default fast-math, but set a ULP tolerance for the golden fixture comparison (the FMA-contraction differences caused by fast-math are of the same order that gsplat CUDA experiences).

### Constants / parameter block

```metal
// SplatXMetal/Kernels/ProjectionEWA.h (shared host header)
constant constexpr float kAlphaThreshold  = 1.f / 255.f;
constant constexpr float kGaussianExtend  = 3.33f;
constant constexpr float kMinCompensation = 0.005f;

struct ProjectionParams {          // passed via setBytes
    uint  N;                       // number of Gaussians (Phase 1: B=1)
    uint  C;                       // number of cameras
    uint  image_width, image_height;
    float eps2d;                   // default 0.3
    float near_plane, far_plane;   // defaults 0.01 / 1e10
    float radius_clip;             // default 0.0
    uint  camera_model;            // 0=PINHOLE, 1=ORTHO, 2=FISHEYE
    uint  has_opacities;           // forward only
    uint  calc_compensations;      // fwd: emit the compensation term / bwd: use v_compensations
};
```

### Forward kernel signature

```metal
kernel void projection_ewa_3dgs_fused_fwd(
    device const float*  means         [[buffer(0)]],   // [N,3]
    device const float*  covars        [[buffer(1)]],   // [N,6] triu (or unbound)
    device const float*  quats         [[buffer(2)]],   // [N,4] (w,x,y,z)
    device const float*  scales        [[buffer(3)]],   // [N,3]
    device const float*  opacities     [[buffer(4)]],   // [N]
    device const float*  viewmats      [[buffer(5)]],   // [C,16] row-major
    device const float*  Ks            [[buffer(6)]],   // [C,9]  row-major
    device int32_t*      radii         [[buffer(7)]],   // [C,N,2]
    device float*        means2d       [[buffer(8)]],   // [C,N,2]
    device float*        depths       [[buffer(9)]],   // [C,N]
    device float*        conics        [[buffer(10)]],  // [C,N,3]
    device float*        compensations [[buffer(11)]],  // [C,N] (optional)
    constant ProjectionParams& p       [[buffer(12)]],
    uint idx [[thread_position_in_grid]])
```

- Dispatch: `dispatchThreads(MTLSize(C*N, 1, 1), threadsPerThreadgroup: MTLSize(256, 1, 1))` — Apple GPUs support non-uniform threadgroups, so the guard `if (idx >= C*N) return;` is defensive only.
- covars vs quat/scale branch: the upstream original checks for nullptr. Metal has no null buffers, so prefer separate compilation via **function constants**:
  ```metal
  constant bool kUseCovars   [[function_constant(0)]];
  constant bool kHasOpacities[[function_constant(1)]];
  constant bool kCalcComp    [[function_constant(2)]];
  ```
  (Removes the branch and saves registers. Bind a dummy for the unused buffers.)
- For the camera_model switch, Phase 1 may pin PINHOLE alone via a function constant (that is the scope of the reference comparison).
- atomics / barriers / threadgroup memory: **none**. A pure map kernel.
- On culling, write 0 to the two `radii` slots only — keep the same convention as CUDA (the other output buffers stay unwritten).

### Backward kernel signature

```metal
kernel void projection_ewa_3dgs_fused_bwd(
    device const float*  means          [[buffer(0)]],
    device const float*  covars         [[buffer(1)]],
    device const float*  quats          [[buffer(2)]],
    device const float*  scales         [[buffer(3)]],
    device const float*  viewmats       [[buffer(4)]],
    device const float*  Ks             [[buffer(5)]],
    device const int32_t* radii         [[buffer(6)]],   // forward output
    device const float*  conics         [[buffer(7)]],   // forward output
    device const float*  compensations  [[buffer(8)]],   // forward output (optional)
    device const float*  v_means2d      [[buffer(9)]],
    device const float*  v_depths       [[buffer(10)]],
    device const float*  v_conics       [[buffer(11)]],
    device const float*  v_compensations[[buffer(12)]],  // optional
    device atomic_float* v_means        [[buffer(13)]],  // [N,3]  — MUST be zero-initialized beforehand
    device atomic_float* v_covars       [[buffer(14)]],  // [N,6]  (covars path)
    device atomic_float* v_quats        [[buffer(15)]],  // [N,4]  (quat/scale path)
    device atomic_float* v_scales       [[buffer(16)]],  // [N,3]
    device atomic_float* v_viewmats     [[buffer(17)]],  // [C,16] (optional)
    constant ProjectionParams& p        [[buffer(18)]],
    uint idx [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
```

- **The gradient buffers MUST be zero-initialized before encoding**, via a blit (fillBuffer=0) or a dedicated clear kernel — the kernel only accumulates (this corresponds to `zeros_like`).
- function constants: `kUseCovars`, `kCalcComp` (= whether v_compensations is present), `kViewmatsGrad`.

#### Reduction strategy (replacing labeled_partition)

1. **Gaussian gradients (v_mean / v_covar / v_quat / v_scale)**:
   - C == 1 (the Phase 1 training default): the global thread ↔ gid mapping is 1:1, so **neither a reduction nor an atomic is needed** — write with a plain store. Split this path out with the function constant `kSingleCamera` (fastest and deterministic → favorable for golden comparison).
   - C > 1: the same gid appears once per camera → perform `atomic_fetch_add_explicit(..., relaxed)` per lane directly. The upstream original's labeled_partition only pays off when a warp contains lanes with the same gid (which the layout makes almost never), and Metal has no equivalent, so a per-thread atomic is both correct and simpler.
2. **v_viewmats (12 floats per camera)**: every lane in a warp is likely to share the same cid → use a simdgroup reduction to cut the atomics to 1/32:
   ```metal
   bool cid_uniform = simd_all(cid == simd_broadcast_first(cid));
   if (cid_uniform) {
       // simd_sum v_R(9) and v_t(3) separately, then only lane 0 does the atomicAdd (12 times)
       for (...) comp = simd_sum(comp);
       if (simd_is_first()) atomic_fetch_add_explicit(...);
   } else {
       // boundary warp: per-lane atomicAdd (correctness first)
   }
   ```
   `simd_sum` is only safe when there are no inactive lanes — **instead of an early return, use a `bool alive` flag and a zero contribution**, or structure the kernel so that culled lanes also reach the reduction point carrying `v_R=0, v_t=0`. (The CUDA original uses an early return, but cg::reduce operates over the set of active lanes; MSL's simd_sum risks being undefined in a divergent state, so this transformation is mandatory.)
3. atomic memory order: `memory_order_relaxed` is sufficient everywhere (pure accumulation; the command-buffer boundary guarantees visibility).
4. threadgroup memory / `threadgroup_barrier`: **not needed**. Only simdgroup operations are used.

#### threadgroup size

- Start at 256 (same as CUDA). Under register pressure, drop to 128 and check occupancy (the backward uses many registers because of persp_proj_vjp plus the quat VJP).
- Keep it a multiple of 32 (the simdgroup width is fixed at 32).

### Determinism (reproducibility) notes

- The forward is fully deterministic.
- The backward's float atomicAdd order is nondeterministic → compare against the golden fixture **using tolerances** (for example relative 1e-5). The C=1 + `kSingleCamera` path (non-atomic stores) is bit-exact deterministic, so run the first round of verification on that path.

### Comparison points against the CPU reference (Phase 0)

- `projection_ewa_fwd_cpu` (`CpuKernels.h:26`) takes covars as a full 3×3 row-major input — numerically identical to CUDA's 6-element triu packing (the matrix is symmetric). When comparing, only the packing conversion needs care.
- Absent from the CPU reference: the opacity-aware radius (the opacities input), radius_clip, the quat/scale path, and ORTHO/FISHEYE. When verifying the forward Metal kernel, compare on two fronts: inside the CPU reference's scope (PINHOLE, covars given directly, opacity unused → extend fixed at 3.33) and outside it (the golden fixture's direct gsplat output).

---

## Open questions

1. **No opacity gradient in the backward**: this kernel does not output `v_opacities`. The opacity-aware radius (F7) is treated as a non-differentiable culling heuristic, and the opacity gradient arises only in the rasterization backward. In anti-aliasing mode the `opacity *= compensation` chain is differentiated in the Python layer through the `compensations` output — when Phase 1 builds a training loop without Python orchestration, it must decide where that chain is owned (rasterizer vs projection).
2. **Fixing the quat layout**: gsplat uses `[w,x,y,z]` (`Utils.cuh:230`). Confirm this matches the quaternion order of splatx's SXT tensors and PLY loader (if it does not, normalize in the loader).
3. **fast-math ULP budget**: how large a fixture tolerance to allow for `__logf` vs MSL `log`, glm `inverse` vs the manual mat2 inverse, and FMA-contraction differences — to be decided after measuring against the CUDA fp32 values of the Phase 0 golden fixture.
4. **Scope of FISHEYE bwd verification**: the analytic derivatives in fisheye_proj_vjp (the E/F/A/B/C/D terms) are long formulas with a high typo risk — decide whether Phase 1 ports PINHOLE first and defers fisheye to a separate milestone with a numerical-differentiation comparison test.
5. **Convention for unwritten `radii` slots**: when the forward culls, means2d/depths/conics stay unwritten (garbage). Re-confirm across the whole pipeline that the downstream kernels (intersect/rasterize) trust the radii>0 filter alone — adding a zero-fill in the Metal implementation would make debugging easier at no performance cost, but it would diverge from the gsplat convention.
6. **Whether to pin B=1**: can the upstream original's batch axis B (multi-batch training) be excluded from the Phase 1 scope (gaining simpler indexing)? — awaiting the controller's decision.

---

## Verification notes (adversarial verifier)

The upstream originals (`ProjectionEWA3DGSFused.cu`, `Utils.cuh`, `Common.h`, `Projection.cpp`, `rendering.py`, `CpuKernels.h`) were re-read independently, and every formula sign, index, clamp, constant, and file:line citation in this document was cross-checked exhaustively. The forward chain (F0–F8), the backward chain (B0–B8), the persp/ortho/fisheye VJP expansions (including the E/F/A/B/C/D terms), the four components of quat_to_rotmat_vjp, and the warp reduction / atomic packing (including the v_covars off-diagonal summation) were all confirmed to **match the upstream original**. The cited line numbers all match the measured source as well (within ±1 line: the bwd kernel signature begins exactly at line 377; line 376 is the template declaration). Constant and default verification: `ALPHA_THRESHOLD`/`GAUSSIAN_EXTEND`/`MIN_COMPENSATION` (`Common.h:97,99,110`), `eps2d=0.3`/`near=0.01`/`far=1e10`/`radius_clip=0.0` (`rendering.py:246–249`), and the CPU reference citations (`CpuKernels.h:11,16,26`) are all correct.

The corrections and supplements that were found are below.

### Correction 1 — error in the backward input list: `opacities` is not a bwd input

The phrase "Inputs: **all forward inputs** + …" at the top of the body section "Formula detail — Backward" is inaccurate. The bwd kernel signature takes only `means, covars, quats, scales, viewmats, Ks` and **does not take `opacities`** (`ProjectionEWA3DGSFused.cu:384–389`), because the opacity-aware radius is non-differentiable culling and therefore need not be recomputed in the backward. The MSL bwd signature in this document already excludes opacities correctly; read the prose as "all forward inputs except opacities".

### Correction 2 — missing fp64 promotion precision: the double literals in `add_blur_vjp`

`Utils.cuh:486`:

```c
float v_sqr_comp = v_compensation * 0.5 / (compensation + 1e-6);
```

`0.5` and `1e-6` are **double literals**, so by the C++ promotion rules this single line is **computed in fp64 and then truncated to float** on CUDA (nvcc fast-math does not demote doubles either). MSL has no double → computing in float can produce a ~1 ULP difference at this point, and it propagates downstream through the four components of `v_covar2d`. This was confirmed to be **the only** place in this kernel chain where a double promotion occurs (the rest — `.5f` (bwd kernel line 440), `2.0f` (fwd line 180), and so on — are all float literals; integer literals such as those in `1 - comp*comp` and `3 * l4 * theta` promote to float). → When sizing the ULP budget of open question 3, treat this term as its own tolerance item in fixtures where `v_compensations` is enabled.

### Correction 3 — spelling out the bwd pointer shifts (the premise behind the B1/B2 notation)

The `conics[0], conics[1], conics[2]` and `v_conics[0..2]` notation in B1 assumes the shifts below (`ProjectionEWA3DGSFused.cu:432–436`):

```
conics    += global_idx * 3;
v_means2d += global_idx * 2;
v_depths  += global_idx;        // → v_depths[0] in B5
v_conics  += global_idx * 3;
```

By contrast, B2's `compensations`/`v_compensations` are indexed directly as `[global_idx]` with no shift (lines 447–448). In B8, `v_viewmats` is written as `[i*4+j]` on the leader lane after `v_viewmats += bid*C*16 + cid*16` (line 625), and `v_means`/`v_covars`/`v_quats`/`v_scales` are likewise shifted by `bid*N*k + gid*k` on the leader (lines 573, 587, 607–608). Port the index computation to Metal following this convention.

### Correction 4 — tightening the gid-duplication condition in "What the reduction means"

In the body's "a gid repeats inside one warp only when `gaussian_count < 32` or at a boundary wrap", the second clause is inaccurate. In the gid-fastest layout, even when a warp wraps a camera/batch boundary, the lanes' gid sets are `[N-k, N)` and `[0, 32-k)`, which are disjoint; therefore **duplication occurs only when `gaussian_count < 32`** (including the case where multi-GPU chunk splitting shrinks `gaussian_count`, `ProjectionEWA3DGSFused.cu:331–332, 772–773`). The conclusion (groups of size 1 in practice, replaceable with a per-thread atomic) remains valid.

### Correction 5 — the fisheye fwd/bwd `z + eps` asymmetry (a property of the upstream original — port it as-is)

The formula transcription in this document is accurate, but the asymmetry itself is stated explicitly here: only `theta` in the forward `mean2d` uses `atan2(xy_len, z + eps)` (`Utils.cuh:711`), whereas `b` in the forward J (line 720), `b` on the bwd mean path (line 761), and `theta` in the bwd (line 769) all use **`atan2(len_xy, z)` without eps**. The bwd mean gradient is therefore not the strict derivative of the forward `mean2d` but an eps-ignoring approximation (the behavior of the gsplat original). This is intentional reproduction of the gsplat original. Do not "fix" it to make it consistent in the Metal port — leave it exactly as the original; the golden fixtures are generated against this behavior.

### Other findings (no document change needed)

- The `v_covars // [..., N, 3, 3]` comment in the upstream bwd launcher (`ProjectionEWA3DGSFused.cu:665`) is a comment error in the original itself. The actual allocation is `zeros_like(covars)` = `[..., N, 6]` (`Projection.cpp:645`), and the kernel comment (line 405) also says `[B, N, 6]` — this document's `[B,N,6]` is the correct entry.
- The Russian typo ("клампов") in the B4a commentary was fixed inline during this verification.
- No other errors were found in the formulas, signs, indices, clamp boundaries (including the handling of equality), constants, thread configuration, or atomic usage sites.

**Verdict: with the corrections above applied, this document is sufficient as the sole reference for the Metal implementation (PASS_WITH_CORRECTIONS).**

### Re-verification (second pass, independent)

The full originals (`ProjectionEWA3DGSFused.cu` lines 1–822, `Utils.cuh` lines 30–841, `Common.h` lines 55–115, `Projection.cpp` lines 380–669, `rendering.py` lines 246–249, and all of `CpuKernels.h`) were read again from the start, and the body sections F0–F8 and B0–B8, the MSL porting section, and Corrections 1–5 above were all re-checked exhaustively. **Corrections 1–5 were each independently re-confirmed to match the upstream original** (absence of opacities in the bwd signature: `ProjectionEWA3DGSFused.cu:384–389`; the double promotion: `Utils.cuh:486`; the pointer shifts: lines 432–436, 447–448, 573, 587, 607–608, 625; the disjoint-gid argument: for `N ≥ 32` the wrap intervals `[N-k,N)` and `[0,32-k)` are always disjoint; the fisheye eps asymmetry: `Utils.cuh:711` vs 720/761/769; the upstream comment error: `ProjectionEWA3DGSFused.cu:665` vs `Projection.cpp:645`, kernel comment line 405). The exhaustive formula comparison found no new sign, index, clamp, or constant errors. Two minor supplements are recorded below.

#### Correction 6 — the actual function name and addition order of the quat normalization (for bit-exact comparison)

1. The normalization in `quat_to_rotmat`/`quat_to_rotmat_vjp` calls `rsqrt_parse_safe(...)`, not `rsqrt(...)` (`Utils.cuh:232, 257`). Its definition (`Utils.cuh:43–50`) is `rsqrt(x)` under `__CUDA_ARCH__` (device) and `1.f / std::sqrt(x)` on the host fallback. **On the device path, the body's `rsqrt` notation and the MSL mapping (`rsqrt`) remain valid** — but fast-math `rsqrt` and `1/sqrt` differ in ULP, so include the small difference between the CPU reference (host-fallback semantics) and Metal `rsqrt` in the ULP budget of open question 3.
2. The addition order of the normalization's sum of squares is `x*x + y*y + z*z + w*w` (**x first, w last**; `Utils.cuh:232, 257`). The prose `rsqrt(w²+x²+y²+z²)` in body section F2 is mathematically equivalent, but FP addition is non-associative, so implement it in source order (x→y→z→w) if a bit-level golden comparison is the goal. (The transcription in B7, `rsqrt(x²+y²+z²+w²)`, already matches the source order.)

#### Details confirmed during re-verification (no document change needed)

- The glm `inverse(mat2)` formula, the proposed MSL `mat2_inv` code (column-major component placement), and the `glm::outerProduct` column convention (`result[j][i] = c[i]·r[j]`) are all accurate.
- In the fisheye VJP's `dL_dt{x,y,z}_raw` summation (lines 820–840), the `dJ_d*rc ↔ v_J[c][r]` pairing matches the body's Σ notation exactly.
- The four components of `quat_to_rotmat_vjp` (lines 262–279), `quat_scale_to_covar_vjp` (lines 338–346), `posW2C_VJP` (lines 110–112), `covarW2C_VJP` (lines 144–145), `add_blur` (lines 457–461), `add_blur_vjp` (lines 485–491), and the persp VJP's FOV branch (lines 670–685) and z contribution (lines 686–689) all match in sign and index.
- The two-argument overload `glm::atan(a, b)` has atan2 semantics — the body's `atan2` notation is valid.
- The `sqrt` on forward line 180 is an unsuffixed call (resolved to the float overload) — identical in meaning to `sqrtf`; the notational difference is harmless.

**Second-pass verdict: with Corrections 1–6 applied, this document is sufficient as the sole reference for the Metal implementation (PASS_WITH_CORRECTIONS maintained).**
