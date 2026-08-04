"""Generate golden fixtures from the gsplat reference implementation (_torch_impl/_math).

Importing the whole gsplat package could touch the CUDA JIT path, so only the
three pure-PyTorch modules from research/gsplat are loaded directly under a stub
package context. Reference commit: 77ab983 (see Global Constraints).

Structure (spec 07, "gen_fixtures.py extension structure"):
- FIXTURES registry: {name: generator function}. Selective regeneration via CLI
  arguments (`uv run gen_fixtures.py [name ...]`; no arguments regenerates all).
- dump_backward: injects a fixed-seed upstream (generated and stored in f32) into
  the forward golden (an f64 graph) via torch.autograd.grad and saves the expected
  input gradients as v_<name>.sxt.
- Tensor file names follow the gsplat kernel argument names verbatim
  (`<arg>.sxt` / `v_<arg>.sxt`, D4).
- Reproducibility: every random draw uses a fixed torch.Generator().manual_seed(...),
  CPU tensors only, and the torch version is pinned by tools/uv.lock → two runs
  must be byte-identical.
"""
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sxt import load_sxt, save_sxt

ROOT = Path(__file__).resolve().parent.parent
GSPLAT_CUDA = ROOT / "research" / "gsplat" / "gsplat" / "cuda"
FIXTURES_DIR = ROOT / "fixtures"

# fixture registry: {name: generator function(math_mod, ti)}
FIXTURES = {}


def fixture(name):
    def deco(fn):
        FIXTURES[name] = fn
        return fn
    return deco


def _load_gsplat_modules():
    pkg_g = types.ModuleType("gsplat"); pkg_g.__path__ = []
    pkg_c = types.ModuleType("gsplat.cuda"); pkg_c.__path__ = [str(GSPLAT_CUDA)]
    sys.modules["gsplat"] = pkg_g
    sys.modules["gsplat.cuda"] = pkg_c

    # Stub for gsplat._helper.assert_shape used by _math.py
    helper = types.ModuleType("gsplat._helper")
    helper.assert_shape = lambda *a, **k: None
    sys.modules["gsplat._helper"] = helper

    # Stubs for the _wrapper symbols _torch_impl.py uses only as type hints
    wrapper = types.ModuleType("gsplat.cuda._wrapper")
    wrapper.CameraModel = str
    wrapper.RollingShutterType = str
    sys.modules["gsplat.cuda._wrapper"] = wrapper

    def load(name):
        spec = importlib.util.spec_from_file_location(f"gsplat.cuda.{name}", GSPLAT_CUDA / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"gsplat.cuda.{name}"] = mod
        spec.loader.exec_module(mod)
        return mod

    load("_constants")
    math_mod = load("_math")
    ti = load("_torch_impl")
    return math_mod, ti


def _write_meta(out_dir, meta):
    # "compare" field (1c Task 1 item 6 — for documentation; not yet enforced by the harness):
    # empty tolerances = an exact (integer/bit) comparison contract for all tensors; otherwise meta-tolerance based.
    meta.setdefault("compare", "toleranced" if meta.get("tolerances") else "exact")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def _save_f32(out_dir, name, t):
    save_sxt(str(out_dir / f"{name}.sxt"), t.detach().numpy().astype(np.float32))


def dump_backward(outputs, inputs, out_dir, grad_seed, upstream_masks=None):
    """Backward golden dump of the common recipe in spec 07.

    outputs/inputs: {name: f64 graph tensor}. The upstream is generated with a fixed
    randn seed in f32, saved as v_<output>.sxt (after multiplying the mask), upcast
    to f64 and injected; the expected input gradients from torch.autograd.grad are
    cast to f32 and saved as v_<input>.sxt.
    upstream_masks: {output name: broadcastable mask} — zeroes the upstream of culled elements.
    """
    g = torch.Generator().manual_seed(grad_seed)
    v_outs = []
    for name, o in outputs.items():
        v = torch.randn(o.shape, generator=g, dtype=torch.float32)
        if upstream_masks and name in upstream_masks:
            v = v * upstream_masks[name].to(v.dtype)
        save_sxt(str(out_dir / f"v_{name}.sxt"), v.numpy().astype(np.float32))
        v_outs.append(v.double())
    grads = torch.autograd.grad(
        outputs=list(outputs.values()),
        inputs=list(inputs.values()),
        grad_outputs=v_outs,
        allow_unused=True,
    )
    result = {}
    for (name, inp), gr in zip(inputs.items(), grads):
        t = (gr if gr is not None else torch.zeros_like(inp)).detach()
        save_sxt(str(out_dir / f"v_{name}.sxt"), t.float().numpy().astype(np.float32))
        result[name] = t
    return result


@fixture("math_quat_scale_to_covar")
def gen_math_fixture(math_mod, ti):
    out = FIXTURES_DIR / "math_quat_scale_to_covar"
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(42)
    quats = torch.randn(64, 4, generator=g)                # unnormalized input (the reference normalizes internally)
    scales = torch.rand(64, 3, generator=g) * 0.5 + 0.01
    covars, _ = math_mod._quat_scale_to_covar_preci(quats, scales, compute_covar=True, compute_preci=False)
    save_sxt(str(out / "quats.sxt"), quats.numpy().astype(np.float32))
    save_sxt(str(out / "scales.sxt"), scales.numpy().astype(np.float32))
    save_sxt(str(out / "covars.sxt"), covars.numpy().astype(np.float32))
    print(f"wrote {out}")


@fixture("proj_fully_fused")
def gen_projection_fixture(math_mod, ti):
    out = FIXTURES_DIR / "proj_fully_fused"
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(7)
    N, C, W, H = 128, 2, 160, 120

    means = torch.randn(N, 3, generator=g) * 1.5 + torch.tensor([0.0, 0.0, 5.0])
    quats = torch.randn(N, 4, generator=g)
    scales = torch.rand(N, 3, generator=g) * 0.4 + 0.05
    covars, _ = math_mod._quat_scale_to_covar_preci(quats, scales, compute_covar=True, compute_preci=False)
    # The CUDA fused kernel contract is the [N,6] upper triangle (07 Correction 5) — the Phase 0 compatible covars.sxt is kept
    covars_triu, _ = math_mod._quat_scale_to_covar_preci(
        quats, scales, compute_covar=True, compute_preci=False, triu=True)

    viewmats = torch.eye(4).repeat(C, 1, 1)
    viewmats[1, :3, 3] = torch.tensor([0.3, -0.2, 0.5])   # the second camera is translated
    Ks = torch.tensor([[[300.0, 0.0, 80.0], [0.0, 300.0, 60.0], [0.0, 0.0, 1.0]]]).repeat(C, 1, 1)

    radii, means2d, depths, conics, compensations = ti._fully_fused_projection(
        means, covars, viewmats, Ks, W, H,
        eps2d=0.3, near_plane=0.01, far_plane=1e10,
        calc_compensations=True, camera_model="pinhole",
    )
    for name, t in [("means", means), ("covars", covars), ("covars_triu", covars_triu),
                    ("viewmats", viewmats), ("Ks", Ks),
                    ("means2d", means2d), ("depths", depths), ("conics", conics),
                    ("compensations", compensations)]:
        save_sxt(str(out / f"{name}.sxt"), t.numpy().astype(np.float32))
    save_sxt(str(out / "radii.sxt"), radii.numpy().astype(np.int32))
    _write_meta(out, {
        "golden_source": "gsplat@77ab983:_torch_impl.py:_fully_fused_projection",
        "N": N, "C": C, "width": W, "height": H,
        "eps2d": 0.3, "near_plane": 0.01, "far_plane": 1e10,
        "radius_clip": 0.0, "camera_model": 0, "calc_compensations": 1,
        "seeds": {"input": 7},
        "note": "The means2d/depths/conics/compensations comparison is restricted to the elements where valid=(radii>0).all(-1) (07 Correction 3). "
                "covars_triu is the [N,6] upper triangle (triu) [c00,c01,c02,c11,c12,c22] (07 Correction 5). radii is compared exact.",
        "tolerances": {
            "means2d": {"rtol": 1e-5, "atol": 1e-5},
            "depths": {"rtol": 1e-5, "atol": 1e-5},
            "conics": {"rtol": 1e-4, "atol": 1e-6},
            "compensations": {"rtol": 1e-4, "atol": 1e-6},
        },
    })
    print(f"wrote {out}")


@fixture("adam_basic")
def gen_adam_basic(math_mod, ti):
    """The 06-spec fused_adam golden — no bias correction, step = -lr*m/(sqrt(v)+eps).

    The expected values are reproduced in f32 arithmetic throughout, exactly like the
    kernel ((1-b1)/(1-b2) are f32 subtractions too — the same rounding as the kernel's
    `1.0f - P.b1`).
    *_out: every row updated, ignoring valid / *_out_masked: Gaussians with valid=0 are
    skipped entirely, moments included.
    """
    out = FIXTURES_DIR / "adam_basic"
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(21)
    N, D = 1000, 3
    lr, b1, b2, eps = 1e-2, 0.9, 0.999, 1e-8

    param = torch.randn(N, D, generator=g)
    grad = torch.randn(N, D, generator=g)
    exp_avg = torch.randn(N, D, generator=g) * 0.1
    exp_avg_sq = torch.rand(N, D, generator=g) * 0.01          # second moment v >= 0
    # Boundary values v ∈ {0, 1e-40 (denormal), 1e-30} — those rows keep grad=0 so v stays
    # tiny after the update (06 open question 3: for measuring Apple GPU denormal/FTZ
    # behavior; absorbed by atol)
    for row, edge in enumerate((0.0, 1e-40, 1e-30)):
        exp_avg_sq[row] = edge
        grad[row] = 0.0
    valid = (torch.rand(N, generator=g) > 0.25).to(torch.uint8)
    valid[:3] = 1                                              # boundary rows always take the update path
    valid[3:6] = 0                                             # guarantee the skip path
    assert 0 < int(valid.sum()) < N

    lr32, b1_32, b2_32, eps32 = (torch.tensor(x, dtype=torch.float32) for x in (lr, b1, b2, eps))
    one_m_b1 = 1.0 - b1_32                                     # f32 subtraction
    one_m_b2 = 1.0 - b2_32

    m2 = b1_32 * exp_avg + one_m_b1 * grad
    v2 = b2_32 * exp_avg_sq + one_m_b2 * grad * grad
    p2 = param + (-lr32) * m2 / (torch.sqrt(v2) + eps32)

    vmask = valid.bool()[:, None]                              # [N,1] → [N,D] broadcast
    for name, t in [("param", param), ("grad", grad), ("exp_avg", exp_avg),
                    ("exp_avg_sq", exp_avg_sq),
                    ("param_out", p2), ("exp_avg_out", m2), ("exp_avg_sq_out", v2),
                    ("param_out_masked", torch.where(vmask, p2, param)),
                    ("exp_avg_out_masked", torch.where(vmask, m2, exp_avg)),
                    ("exp_avg_sq_out_masked", torch.where(vmask, v2, exp_avg_sq))]:
        _save_f32(out, name, t)
    save_sxt(str(out / "valid.sxt"), valid.numpy().astype(np.uint8))

    tol = {"rtol": 1e-6, "atol": 1e-7}
    _write_meta(out, {
        "golden_source": "gsplat@77ab983:AdamCUDA.cu:34-71 (the formula reproduced by hand in f32)",
        "N": N, "D": D, "lr": lr, "b1": b1, "b2": b2, "eps": eps,
        "seeds": {"input": 21},
        "note": "No bias correction, and eps is added outside sqrt(v) (06). *_out = every row updated, ignoring valid; "
                "*_out_masked = for the Gaussians with valid[g]=0 the param and both moments stay exactly as given in the input. "
                "Rows exp_avg_sq[0..2] = the boundary values {0, 1e-40, 1e-30} (grad=0 on those rows).",
        "tolerances": {name: dict(tol) for name in
                       ("param_out", "exp_avg_out", "exp_avg_sq_out",
                        "param_out_masked", "exp_avg_out_masked", "exp_avg_sq_out_masked")},
    })
    print(f"wrote {out}")


@fixture("proj_fused_qs")
def gen_proj_fused_qs(math_mod, ti):
    """The quats/scales input path of projection fused (01/07) — two sets: C=1 (c1) and C=2 (c2).

    The golden is the composite graph _quat_scale_to_covar_preci ∘ _fully_fused_projection (07).
    The bwd upstream is stored and injected after multiplying the valid=(radii>0).all(-1)
    mask (07 Correction 3).
    v_compensations is injected only in c2 (the calc_comp variant) — c1 is equivalent to v_comp=0.
    opacities is saved only for the kernel argument contract — the golden takes the
    opacities=None path (the torch golden has no opacity-aware bbox, 07 open question #2).
    """
    base = FIXTURES_DIR / "proj_fused_qs"
    g = torch.Generator().manual_seed(11)
    N, W, H = 128, 160, 120

    means = torch.randn(N, 3, generator=g) * 1.5 + torch.tensor([0.0, 0.0, 5.0])
    quats = torch.randn(N, 4, generator=g)
    scales = torch.rand(N, 3, generator=g) * 0.4 + 0.05
    opacities = torch.rand(N, generator=g) * 0.49 + 0.5

    viewmats_all = torch.eye(4).repeat(2, 1, 1)
    viewmats_all[1, :3, 3] = torch.tensor([0.3, -0.2, 0.5])
    Ks_all = torch.tensor([[[300.0, 0.0, 80.0], [0.0, 300.0, 60.0], [0.0, 0.0, 1.0]]]).repeat(2, 1, 1)

    for variant, C, with_vcomp in (("c1", 1, False), ("c2", 2, True)):
        out = base / variant
        out.mkdir(parents=True, exist_ok=True)
        viewmats, Ks = viewmats_all[:C], Ks_all[:C]

        # ---- forward golden (f32) ----
        covars, _ = math_mod._quat_scale_to_covar_preci(quats, scales, compute_covar=True, compute_preci=False)
        radii, means2d, depths, conics, compensations = ti._fully_fused_projection(
            means, covars, viewmats, Ks, W, H,
            eps2d=0.3, near_plane=0.01, far_plane=1e10,
            calc_compensations=True, camera_model="pinhole",
        )
        # Side note of Correction 3: assert det > 0 after blur (the CUDA det<=0 culling does not fire = behavior matches the golden)
        means_c, covars_c = ti._world_to_cam(means, covars, viewmats)
        _, covars2d = ti._persp_proj(means_c, covars_c, Ks, W, H)
        blur = covars2d + torch.eye(2) * 0.3
        det_blur = blur[..., 0, 0] * blur[..., 1, 1] - blur[..., 0, 1] * blur[..., 1, 0]
        assert float(det_blur.min()) > 0.0, "det(covar2d+eps2d I) <= 0: the CUDA culling and the golden diverge"

        valid = (radii > 0).all(-1)                            # [C,N]
        assert 0 < int(valid.sum()) < C * N, "degenerate valid mask: a mix of culled and surviving Gaussians is required"

        for name, t in [("means", means), ("quats", quats), ("scales", scales),
                        ("opacities", opacities), ("viewmats", viewmats), ("Ks", Ks),
                        ("means2d", means2d), ("depths", depths), ("conics", conics),
                        ("compensations", compensations)]:
            _save_f32(out, name, t)
        save_sxt(str(out / "radii.sxt"), radii.numpy().astype(np.int32))

        # ---- backward golden (f64 graph; the upstream is stored in f32 then injected as f64) ----
        # viewmats also requires_grad — the v_viewmats golden (pose optimization Task 1, kProjViewmatsGrad).
        # The kernel writes v_viewmats only for the [R|t] 3×4 part (lines 624–637); since
        # _world_to_cam reads only [:3,:3]/[:3,3] of viewmats, the autograd grad of the bottom
        # row [0,0,0,1] (flat 12–15) is also exactly 0 → convention matches the kernel
        # (unwritten, zero-initialized).
        means64 = means.double().clone().requires_grad_(True)
        quats64 = quats.double().clone().requires_grad_(True)
        scales64 = scales.double().clone().requires_grad_(True)
        viewmats64 = viewmats.double().clone().requires_grad_(True)
        covars64, _ = math_mod._quat_scale_to_covar_preci(quats64, scales64, compute_covar=True, compute_preci=False)
        _, m2d64, d64, con64, comp64 = ti._fully_fused_projection(
            means64, covars64, viewmats64, Ks.double(), W, H,
            eps2d=0.3, near_plane=0.01, far_plane=1e10,
            calc_compensations=True, camera_model="pinhole",
        )
        outputs = {"means2d": m2d64, "depths": d64, "conics": con64}
        masks = {"means2d": valid[..., None], "depths": valid, "conics": valid[..., None]}
        if with_vcomp:
            outputs["compensations"] = comp64
            masks["compensations"] = valid
        dump_backward(outputs, {"means": means64, "quats": quats64, "scales": scales64,
                                "viewmats": viewmats64},
                      out, grad_seed=1011, upstream_masks=masks)

        _write_meta(out, {
            "golden_source": "gsplat@77ab983:_math.py:_quat_scale_to_covar_preci + _torch_impl.py:_fully_fused_projection",
            "N": N, "C": C, "width": W, "height": H,
            "eps2d": 0.3, "near_plane": 0.01, "far_plane": 1e10,
            "radius_clip": 0.0, "camera_model": 0, "calc_compensations": 1,
            "bwd_inject_v_compensations": 1 if with_vcomp else 0,
            "seeds": {"input": 11, "grad": 1011},
            "tensors": {
                "inputs": ["means", "quats", "scales", "opacities", "viewmats", "Ks"],
                "outputs": ["radii", "means2d", "depths", "conics", "compensations"],
                "grad_outputs": (["v_means2d", "v_depths", "v_conics", "v_compensations"]
                                 if with_vcomp else ["v_means2d", "v_depths", "v_conics"]),
                "grad_inputs": ["v_means", "v_quats", "v_scales", "v_viewmats"],
            },
            "note": "The fwd comparison is restricted to the elements where valid=(radii>0).all(-1), "
                    "and radii is exact (07 Correction 3). The upstream v_* are stored after multiplication "
                    "by the valid mask — inject them as they are. opacities is unused by the golden "
                    "(the golden takes the opacities=None path, 07 open question #2)."
                    + (" v_means atol 5e-5: summing along the C=2 camera axis cancels signs (contributions"
                       " of order ±72 → a sum of order 0.04), so the ~1e-5 per-camera rounding of the fp32"
                       " kernel survives — a term-by-term fp64 recomputation was confirmed to match the"
                       " golden (Task 6), and CUDA fp32 has the same limit." if with_vcomp else ""),
            "tolerances": {
                "means2d": {"rtol": 1e-5, "atol": 1e-5},
                "depths": {"rtol": 1e-5, "atol": 1e-5},
                "conics": {"rtol": 1e-4, "atol": 1e-6},
                "compensations": {"rtol": 1e-4, "atol": 1e-6},
                # C=2: sign cancellation across cameras raises the v_means absolute error floor (see note above, controller decision 2)
                "v_means": {"rtol": 2e-4, "atol": 5e-5 if C == 2 else 1e-6, "rel_l2": 1e-4},
                "v_quats": {"rtol": 2e-4, "atol": 1e-6, "rel_l2": 1e-4},
                "v_scales": {"rtol": 2e-4, "atol": 1e-6, "rel_l2": 1e-4},
                # v_viewmats [C,16]: simd_sum (over all Gaussians) + per-camera atomicAdd accumulation
                # (B8 reduction strategy 2) — a sign-cancelling sum of large-magnitude terms
                # (fx/z·mean_w scale ~10²∼10³, |g|max≈2.7e3). rtol/rel_l2 follow the same convention
                # as this kernel's three bwd gradients (2e-4/1e-4); atol is the atomic absolute floor.
                # Measured (M4, Metal↔golden): rel_l2≈1e-7, maxabs≈3.7e-4, maxrel(|g|>1)≈3e-6 —
                # raised with a safety multiple (atol 2e-3 ≈ 5.5×maxabs). The bottom row flat 12–15
                # is exactly 0 in both the golden and the kernel (only viewmats [:3,:3]/[:3,3] is
                # differentiated) → error 0.
                "v_viewmats": {"rtol": 2e-4, "atol": 2e-3, "rel_l2": 1e-4},
            },
        })
        print(f"wrote {out}")


def _stable_sort_keys(ids_np):
    """Ascending **stable** sort indices for u64 keys — np.lexsort((lo, hi)), the last key is primary.

    Lexicographic ascending order on the (hi, lo) uint32 pairs == unsigned ascending
    order on the u64 keys (the sort contract, 03: stable + ulong). The equivalence
    baseline is the CPU std::stable_sort(ulong key). torch.sort does not guarantee
    tie order (stability), so it is not used for sort goldens.
    """
    lo = (ids_np & 0xFFFFFFFF).astype(np.uint32)
    hi = (ids_np >> 32).astype(np.uint32)
    return np.lexsort((lo, hi))


def _run_isect_golden(out, ti, means2d, radii, depths, tile_size, tile_width, tile_height):
    """Bulk dump of the _isect_tiles(sort=False) → np.lexsort(stable) → _isect_offset_encode golden.

    Stored: the 3 inputs (means2d f32/radii i32/depths f32) + tiles_per_gauss +
    the raw emit-order isect_ids lo/hi and flatten_ids + the sorted lo/hi and
    flatten_ids + offsets. 64-bit keys: SXT does not support int64 → stored split
    into int32 lo (low 32 = the depth f32 bits) and hi (high 32 = image|tile) (D3).
    Returns: a dict for verification/meta.
    """
    I, N = means2d.shape[0], means2d.shape[1]
    tiles_per_gauss, isect_ids, flatten_ids = ti._isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height, sort=False)
    ids_np = isect_ids.numpy()
    n_isects = int(ids_np.shape[0])
    assert n_isects > 0, "n_isects == 0: degenerate fixture"
    assert (ids_np >= 0).all()
    # Valid depth > 0 → depth bits < 2^31 → the lo int32 cast is safe + unsigned comparison is monotonic (03 §1)
    # (this assertion only catches the sign bit = depth<0 — depth==+0.0 has bit pattern 0 and passes)
    assert ((ids_np & 0xFFFFFFFF) < 2**31).all(), "sign bit in the depth bits — a depth<0 slipped in"
    # hi (the high 32 bits, image|tile) is also stored split as int32 — if the sign bit is set,
    # the i32 cast folds it negative; harmless for the lo/hi reassembly convention (uint32
    # reinterpretation) but outside the fixture readability/debugging contract.
    # At the current fixture scale (image+tile <= 8 bits) this trips far earlier (1c Task 1 item 5).
    assert ((ids_np >> 32) < 2**31).all(), "sign bit in the hi bits — image|tile bit width exceeded"

    order = _stable_sort_keys(ids_np)
    ids_sorted = ids_np[order]
    flat_sorted = flatten_ids.numpy()[order]
    # Cross-check of key values: the sorted key values must equal torch.sort's (lexsort is authoritative for tie order)
    ref_sorted, _ = torch.sort(isect_ids)
    assert np.array_equal(ids_sorted, ref_sorted.numpy())

    offsets = ti._isect_offset_encode(
        torch.from_numpy(ids_sorted.copy()), I, tile_width, tile_height)

    # offsets self-verification: diff([offsets.flatten(), n_isects]) == the flattened per-tile counts
    tile_n_bits = ti._bits_for_count(tile_width * tile_height)
    image_n_bits = ti._bits_for_count(I)
    hi_sorted = ids_sorted >> 32
    flat_tile = ((hi_sorted >> tile_n_bits) * (tile_width * tile_height)
                 + (hi_sorted & ((1 << tile_n_bits) - 1)))
    counts = np.diff(np.append(offsets.numpy().astype(np.int64).reshape(-1), n_isects))
    ref_counts = np.bincount(flat_tile, minlength=I * tile_width * tile_height)
    assert np.array_equal(counts, ref_counts), "offsets exclusive prefix verification failed"

    _save_f32(out, "means2d", means2d)
    save_sxt(str(out / "radii.sxt"), radii.numpy().astype(np.int32))
    _save_f32(out, "depths", depths)
    save_sxt(str(out / "tiles_per_gauss.sxt"), tiles_per_gauss.numpy().astype(np.int32))
    save_sxt(str(out / "isect_ids_lo.sxt"), (ids_np & 0xFFFFFFFF).astype(np.int32))
    save_sxt(str(out / "isect_ids_hi.sxt"), (ids_np >> 32).astype(np.int32))
    save_sxt(str(out / "flatten_ids.sxt"), flatten_ids.numpy().astype(np.int32))
    save_sxt(str(out / "isect_ids_sorted_lo.sxt"), (ids_sorted & 0xFFFFFFFF).astype(np.int32))
    save_sxt(str(out / "isect_ids_sorted_hi.sxt"), (ids_sorted >> 32).astype(np.int32))
    save_sxt(str(out / "flatten_ids_sorted.sxt"), flat_sorted.astype(np.int32))
    save_sxt(str(out / "offsets.sxt"), offsets.numpy().astype(np.int32))
    return {
        "n_isects": n_isects, "tile_n_bits": tile_n_bits, "image_n_bits": image_n_bits,
        "tiles_per_gauss": tiles_per_gauss, "ids_sorted": ids_sorted,
        "flat_sorted": flat_sorted, "offsets": offsets,
    }


_ISECT_KEY_ENCODING = (
    "isect_id(u64) = image_id << (32+tile_n_bits) | tile_id << 32 | f32bits(depth); "
    "tile_id = tile_y*tile_width + tile_x; tile_n_bits = bits_for_count(tile_width*tile_height), "
    "image_n_bits = bits_for_count(I). SXT does not support int64 → stored split into int32 lo "
    "(low 32 bits = the depth f32 bits reinterpreted) and hi (high 32 bits = image|tile) (D3)."
)
_ISECT_SORT_CONTRACT = (
    "Ascending **stable** sort on the ulong(u64) key (the sort contract, 03). The golden is "
    "np.lexsort((lo,hi)) — the equivalence baseline is the CPU std::stable_sort(ulong key). "
    "torch.sort is not used (stability is not guaranteed)."
)

# fixtures/isect_accutile is produced by the **C++ CPU reference**, not this script (Task 5):
#   swift run splatx-core-tests --dump-accutile fixtures/isect_accutile
# torch _isect_tiles has no AccuTile (SNUGBOX + ellipse-line sweep) path (07 Correction 2),
# so the golden producer is isect_tiles_cpu. For the self-consistency defenses (the
# AccuTile ⊆ AABB assertion + the independent geometric oracle) see
# Sources/splatx-core-tests/IsectTests.cpp. Do not regenerate it here.


@fixture("isect_std")
def gen_isect_std(math_mod, ti):
    """The standard golden for the isect pipeline — chained from the proj_fully_fused fwd outputs (radii/means2d/depths).

    torch _isect_tiles models **only the AABB fallback path** (07 Correction 2) — it is
    not a golden for the AccuTile (conics/opacities) path. C=2 is kept to verify the
    image_n_bits=1 path (key bit 39). The 3 inputs are re-saved as byte-identical copies
    of the proj_fully_fused SXT files (a self-contained fixture).
    """
    src = FIXTURES_DIR / "proj_fully_fused"
    out = FIXTURES_DIR / "isect_std"
    out.mkdir(parents=True, exist_ok=True)
    for f in ("means2d", "radii", "depths"):
        if not (src / f"{f}.sxt").exists():
            raise SystemExit("isect_std chains from proj_fully_fused — generate proj_fully_fused first")
    src_meta = json.loads((src / "meta.json").read_text())
    W, H = src_meta["width"], src_meta["height"]
    tile_size = 16
    tile_width = -(-W // tile_size)    # 160/16 = 10
    tile_height = -(-H // tile_size)   # ceil(120/16) = 8 → a partial tile row at the bottom

    means2d = torch.from_numpy(load_sxt(str(src / "means2d.sxt")))   # [C,N,2] f32
    radii = torch.from_numpy(load_sxt(str(src / "radii.sxt")))       # [C,N,2] i32
    depths = torch.from_numpy(load_sxt(str(src / "depths.sxt")))     # [C,N]   f32
    C, N = depths.shape

    r = _run_isect_golden(out, ti, means2d, radii, depths, tile_size, tile_width, tile_height)

    # Non-degeneracy: a mix of culled (0-tile) and multi-tile Gaussians + both images contribute (the 07 size table)
    tpg = r["tiles_per_gauss"]
    assert bool((tpg == 0).any()) and bool((tpg > 1).any())
    assert bool((tpg.sum(dim=-1) > 0).all()), "intersections required per image (the image_n_bits path)"
    # Reaches the right and bottom (partial tile row) edges of the tile grid
    tid_sorted = (r["ids_sorted"] >> 32) & ((1 << r["tile_n_bits"]) - 1)
    assert int((tid_sorted % tile_width).max()) == tile_width - 1
    assert int((tid_sorted // tile_width).max()) == tile_height - 1
    # This fixture must contain no tied keys — stable-under-ties verification is isect_ties' job
    assert np.unique(r["ids_sorted"]).size == r["n_isects"], "unexpected tied keys — duplicates the role of isect_ties"

    _write_meta(out, {
        "golden_source": "gsplat@77ab983:_torch_impl.py:_isect_tiles(sort=False) + np.lexsort + _isect_offset_encode",
        "chained_from": "proj_fully_fused (a byte-identical copy of the radii/means2d/depths fwd goldens, seed 7)",
        "I": C, "N": N, "width": W, "height": H,
        "tile_size": tile_size, "tile_width": tile_width, "tile_height": tile_height,
        "n_tiles": tile_width * tile_height,
        "tile_n_bits": r["tile_n_bits"], "image_n_bits": r["image_n_bits"],
        "n_isects": r["n_isects"],
        "seeds": {"input": 7},
        "tensors": {
            "inputs": ["means2d", "radii", "depths"],
            "outputs": ["tiles_per_gauss", "isect_ids_lo", "isect_ids_hi", "flatten_ids",
                        "isect_ids_sorted_lo", "isect_ids_sorted_hi", "flatten_ids_sorted",
                        "offsets"],
        },
        "key_encoding": _ISECT_KEY_ENCODING,
        "sort_contract": _ISECT_SORT_CONTRACT,
        "note": "torch _isect_tiles covers only the AABB fallback path (07 Correction 2) — it is not "
                "an AccuTile (conics/opacities) golden. The Metal isect kernel is verified under the "
                "AABB contract (conics/opacities not passed). isect_ids_lo/hi and flatten_ids are the "
                "raw output in emit order (2-pass pass2), and *_sorted are the sorted goldens. The keys "
                "of this fixture contain no ties (asserted) — stable-sort verification under ties is "
                "isect_ties. All tensors are compared exact (int/bit).",
        "tolerances": {},
    })
    print(f"wrote {out} (n_isects={r['n_isects']})")


@fixture("isect_ties")
def gen_isect_ties(math_mod, ti):
    """A small fixture dedicated to tied keys — verifies stable-sort order preservation (the sort contract 03, 07 open question #4).

    Synthesized input (I=1, no random numbers): 12 clones at the same position with the
    same depth (each covering 4 tiles) + 2 ties at different positions + one shallow and
    one deep depth in the same tile + one identical depth in a different tile + one culled
    with radii=0. In emit order (image → Gaussian → tile y → x) the flatten_ids within a
    tie group are ascending — a stable sort must preserve this, and the flatten_ids_sorted
    golden pins that order bit-for-bit.
    I=1 → also covers the image_n_bits=0 path (the key-encoding edge of single-image training).
    """
    out = FIXTURES_DIR / "isect_ties"
    out.mkdir(parents=True, exist_ok=True)
    tile_size, tile_width, tile_height = 16, 4, 4      # conceptually W=H=64, n_tiles=16 → tile_n_bits=4
    n_clones = 12
    #        (x,    y,    rx, ry, depth)
    rows = [(16.0, 16.0, 8, 8, 2.0)] * n_clones        # g0..g11: 4 tiles each — (0,0),(1,0),(0,1),(1,1)
    rows += [
        (20.0, 20.0, 2, 2, 2.0),                       # g12: tile (1,1) only — tied with the clones (different position)
        (28.0, 28.0, 2, 2, 2.0),                       # g13: tile (1,1) only — tied
        (24.0, 24.0, 2, 2, 1.5),                       # g14: same tile, shallow depth → sorts before the group
        (24.0, 24.0, 2, 2, 7.25),                      # g15: same tile, deep depth → sorts after the group
        (52.0, 52.0, 3, 3, 2.0),                       # g16: tile (3,3) — same depth but separated by tile
        (40.0, 8.0, 0, 0, 3.0),                        # g17: radii=0 → culled (0 tiles)
    ]
    means2d = torch.tensor([[x, y] for x, y, *_ in rows], dtype=torch.float32)[None]   # [1,N,2]
    radii = torch.tensor([[rx, ry] for _, _, rx, ry, _ in rows], dtype=torch.int32)[None]
    depths = torch.tensor([d for *_, d in rows], dtype=torch.float32)[None]
    N = len(rows)

    r = _run_isect_golden(out, ti, means2d, radii, depths, tile_size, tile_width, tile_height)

    # Assert the expected structure: tile counts / intersection count / tie groups
    expected_tpg = [4] * n_clones + [1, 1, 1, 1, 1, 0]
    assert r["tiles_per_gauss"].reshape(-1).tolist() == expected_tpg
    assert r["n_isects"] == 4 * n_clones + 5 == 53
    uniq, starts, cnts = np.unique(r["ids_sorted"], return_index=True, return_counts=True)
    tie_sizes = sorted(int(c) for c in cnts if c > 1)
    assert tie_sizes == [12, 12, 12, 14], f"unexpected tie group sizes: {tie_sizes}"
    # The stable contract: within a tie group flatten_ids preserves emit order (ascending)
    tie_groups = []
    for u, s, c in zip(uniq, starts, cnts):
        seg = r["flat_sorted"][s:s + c]
        assert (np.diff(seg) > 0).all(), "emit order (ascending Gaussian index) broken within a tie group"
        if c > 1:
            tie_groups.append({"hi": int(u >> 32), "lo": int(u & 0xFFFFFFFF), "count": int(c)})

    _write_meta(out, {
        "golden_source": "gsplat@77ab983:_torch_impl.py:_isect_tiles(sort=False) + np.lexsort + _isect_offset_encode",
        "I": 1, "N": N, "width": tile_width * tile_size, "height": tile_height * tile_size,
        "tile_size": tile_size, "tile_width": tile_width, "tile_height": tile_height,
        "n_tiles": tile_width * tile_height,
        "tile_n_bits": r["tile_n_bits"], "image_n_bits": r["image_n_bits"],
        "n_isects": r["n_isects"],
        "seeds": {},
        "tensors": {
            "inputs": ["means2d", "radii", "depths"],
            "outputs": ["tiles_per_gauss", "isect_ids_lo", "isect_ids_hi", "flatten_ids",
                        "isect_ids_sorted_lo", "isect_ids_sorted_hi", "flatten_ids_sorted",
                        "offsets"],
        },
        "key_encoding": _ISECT_KEY_ENCODING,
        "sort_contract": _ISECT_SORT_CONTRACT,
        "tie_groups": tie_groups,
        "note": "Dedicated to tied keys (hand-synthesized input, no random numbers, AABB path — 07 "
                "Correction 2). Within a group sharing the same (image,tile,depth bits) key, "
                "flatten_ids_sorted preserves emit order (ascending Gaussian index) — a non-stable "
                "sort mismatches this golden bit-for-bit. I=1 → covers the image_n_bits=0 edge. "
                "All tensors are compared exact (int/bit).",
        "tolerances": {},
    })
    print(f"wrote {out} (n_isects={r['n_isects']}, tie_groups={[g['count'] for g in tie_groups]})")


@fixture("sh_k16")
def gen_sh_k16(math_mod, ti):
    """SH deg 0..3 fwd + deg 3 bwd (K=16). The dirs source is means/campos (dirs = means - campos).

    The radii gate: only Gaussians with masks=(radii>0).all(-1) are valid (the same
    convention as 07 Correction 3).
    The fwd goldens colors_deg{d} store the mask=0 slots as 0 — the CUDA/Metal kernels
    skip writing masked slots (assuming a zero-initialized output buffer), so a full
    comparison is possible.
    The bwd upstream v_colors is stored and injected after multiplying the mask →
    v_coeffs/v_dirs match the CUDA skip semantics. v_means = v_dirs.sum(over the B axis)
    (since dirs is a translation of means).
    """
    out = FIXTURES_DIR / "sh_k16"
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(13)
    B, N, K = 2, 64, 16

    means = torch.randn(N, 3, generator=g) * 1.5 + torch.tensor([0.0, 0.0, 5.0])
    campos = torch.tensor([[0.0, 0.0, 0.0], [0.3, -0.2, 0.5]])
    dirs = means[None, :, :] - campos[:, None, :]              # [B,N,3]
    assert float(dirs.norm(dim=-1).min()) > 0.1, "dirs norm too small (07: rsqrt/F.normalize difference)"
    coeffs = torch.randn(N, K, 3, generator=g) * 0.5

    radii = torch.randint(1, 30, (B, N, 2), generator=g, dtype=torch.int32)
    kill = torch.rand(B, N, generator=g) < 0.3
    radii[kill] = 0
    masks = (radii > 0).all(-1)                                # [B,N] bool
    assert 0 < int(masks.sum()) < B * N

    for name, t in [("means", means), ("campos", campos), ("dirs", dirs), ("coeffs", coeffs)]:
        _save_f32(out, name, t)
    save_sxt(str(out / "radii.sxt"), radii.numpy().astype(np.int32))
    save_sxt(str(out / "masks.sxt"), masks.numpy().astype(np.uint8))

    for deg in range(4):                                       # cover every degree branch (07)
        colors = ti._spherical_harmonics(deg, dirs, coeffs)
        colors = colors * masks[..., None]
        _save_f32(out, f"colors_deg{deg}", colors)

    # ---- backward golden (deg 3, f64) ----
    coeffs64 = coeffs.double().clone().requires_grad_(True)
    means64 = means.double().clone().requires_grad_(True)
    dirs64 = means64[None, :, :] - campos.double()[:, None, :]
    colors64 = ti._spherical_harmonics(3, dirs64, coeffs64)
    grads = dump_backward({"colors": colors64},
                          {"coeffs": coeffs64, "dirs": dirs64, "means": means64},
                          out, grad_seed=1013,
                          upstream_masks={"colors": masks[..., None]})
    assert torch.allclose(grads["means"], grads["dirs"].sum(0)), "v_means != v_dirs.sum(B)"

    tol_bwd = {"rtol": 1e-4, "atol": 1e-6, "rel_l2": 1e-4}
    _write_meta(out, {
        "golden_source": "gsplat@77ab983:_torch_impl.py:_spherical_harmonics",
        "B": B, "N": N, "K": K, "D": 3, "degrees_to_use_bwd": 3,
        "seeds": {"input": 13, "grad": 1013},
        "tensors": {
            "inputs": ["dirs", "coeffs", "masks"],
            "inputs_aux": ["means", "campos", "radii"],
            "outputs": ["colors_deg0", "colors_deg1", "colors_deg2", "colors_deg3"],
            "grad_outputs": ["v_colors"],
            "grad_inputs": ["v_coeffs", "v_dirs", "v_means"],
        },
        "note": "masks=(radii>0).all(-1) [B,N] uint8. In colors_deg{d} the mask=0 slots are 0 "
                "(the kernel skips writing them → the output buffer is zero-initialized and then "
                "compared in full). v_colors is stored after multiplication by the mask. "
                "v_means[N,3] = v_dirs[B,N,3].sum(over the B axis) (the chain gradient of dirs = means - campos).",
        "tolerances": {
            "colors_deg0": {"rtol": 1e-5, "atol": 1e-6},
            "colors_deg1": {"rtol": 1e-5, "atol": 1e-6},
            "colors_deg2": {"rtol": 1e-5, "atol": 1e-6},
            "colors_deg3": {"rtol": 1e-5, "atol": 1e-6},
            "v_coeffs": dict(tol_bwd),
            "v_dirs": dict(tol_bwd),
            "v_means": dict(tol_bwd),
        },
    })
    print(f"wrote {out}")


@fixture("sh_k25")
def gen_sh_k25(math_mod, ti):
    """SH deg 4 (K=25) fwd only — the generic kernel path (07). No mask (all valid)."""
    out = FIXTURES_DIR / "sh_k25"
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(17)
    B, N, K, deg = 1, 32, 25, 4

    raw = torch.randn(B, N, 3, generator=g)
    dirs = F.normalize(raw, p=2, dim=-1) * (0.5 + torch.rand(B, N, 1, generator=g))  # norm ∈ [0.5, 1.5)
    coeffs = torch.randn(N, K, 3, generator=g) * 0.5
    colors = ti._spherical_harmonics(deg, dirs, coeffs)

    for name, t in [("dirs", dirs), ("coeffs", coeffs), ("colors_deg4", colors)]:
        _save_f32(out, name, t)
    _write_meta(out, {
        "golden_source": "gsplat@77ab983:_torch_impl.py:_spherical_harmonics",
        "B": B, "N": N, "K": K, "D": 3, "degrees_to_use": deg,
        "seeds": {"input": 17},
        "tensors": {"inputs": ["dirs", "coeffs"], "outputs": ["colors_deg4"]},
        "tolerances": {"colors_deg4": {"rtol": 1e-5, "atol": 1e-6}},
    })
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# rasterize goldens (1c Task 2) — the original reference implementation `_rasterize_reference` + 3 fixtures
# ---------------------------------------------------------------------------

# gsplat@77ab983 Common.h:97-114 constants (spec 05)
_MAX_ALPHA = 0.99                      # alpha clamp upper bound
_ALPHA_THRESHOLD = 1.0 / 255.0         # valid gate lower bound
_TRANSMITTANCE_THRESHOLD = 1e-4        # early termination threshold (= (1-MAX_ALPHA)^2)

_RASTER_GOLDEN_SOURCE = (
    "gen_fixtures.py:_rasterize_reference (an original pure-torch f64 implementation — gsplat "
    "_rasterize_to_pixels depends on CUDA/nerfacc and cannot be used on macOS, 07). The formulas "
    "and constants = gsplat@77ab983 RasterizeToPixels3DGSDevice.cuh:44-55 + SerialBatchFwd.cu:179-296 + "
    "Common.h:97-114 (spec 05)."
)
_RASTER_NOTE = (
    "Render equation: sigma=0.5(A dx²+C dy²)+B dx dy (delta=mean-pixel_center, center=+0.5), "
    "alpha=min(0.99, opac·exp(-sigma)), valid=!(sigma<0 || alpha<1/255), "
    "early termination at next_T=T(1-alpha)<=1e-4 **exclusive** (the triggering Gaussian is not "
    "included, Fwd.cu:246-251). last_ids = the **absolute isect index** (relative to flatten_ids) "
    "of the last contributing Gaussian, or 0 when nothing contributed (a dual-purpose encoding — "
    "because MAX_ALPHA=0.99 the first Gaussian can never cross the threshold, so there is no "
    "contamination, Fwd.cu:186/261/293). The background is applied globally in one pass outside "
    "the tile loop: render_colors += (1-render_alphas)·bg (07 Correction 4 — bg is written to the "
    "in-bounds pixels of empty tiles as well). masks=None is fixed (07 Correction 6). "
    "flatten_ids is the **sorted golden** (the isect sort contract, 03), and isect_offsets is the "
    "global [I,TH,TW] cumulative sum. Comparison: render_colors/render_alphas with tolerances, "
    "last_ids **exact** (the flake policy at the threshold boundary is 07 open question #5 — to be "
    "decided after measurement). The Metal bwd test feeds this fixture's fwd goldens "
    "(render_alphas/last_ids) in as input to block fwd error propagation (07). "
    "bwd: inject v_render_colors/v_render_alphas → v_means2d/v_conics/v_colors/v_opacities "
    "(last_ids is not differentiated). A Gaussian with the MAX_ALPHA clamp active contributes 0 to "
    "v_conics/v_means2d/v_opacities and only v_colors survives (the clamp_max autograd ≡ the "
    "Bwd.cu:259 gate). Self-verification: (a) T decays monotonically and the surviving T>1e-4, "
    "(b) the render_alphas==1-T_final identity (f64 1e-12), (c) an all-pixel comparison against a "
    "serial per-pixel scalar oracle (a literal translation of the CUDA fwd), (d) the raster_small "
    "hand-check assertions."
)
_RASTER_SIZE_NOTE = (
    "The total fixtures size cap was relaxed to < 2MB by approval (the 1c Task 2 brief) — the "
    "raster image tensors [I,H,W,{3,1}] (render/v_render) dominate the size."
)
_RASTER_TOLERANCES = {
    # Starting values from the 07 tolerance table (adjustable after on-device measurement — the tests read the meta)
    "render_colors": {"rtol": 1e-4, "atol": 1e-5},
    "render_alphas": {"rtol": 1e-4, "atol": 1e-5},
    "v_colors": {"rtol": 1e-3, "atol": 1e-6, "rel_l2": 1e-3},
    "v_means2d": {"rtol": 5e-3, "atol": 1e-5, "rel_l2": 1e-3},
    "v_conics": {"rtol": 5e-3, "atol": 1e-5, "rel_l2": 1e-3},
    "v_opacities": {"rtol": 5e-3, "atol": 1e-5, "rel_l2": 1e-3},
}


def _raster_tolerances(v_colors_rtol=1e-3):
    """raster meta.json tolerances — only the v_colors rtol allows per-fixture measured calibration (07 #8).

    1c Task 4 measurement (M4, f64 comparison against an independent scalar bwd
    simulator): the render_alphas the bwd consumes is a tensor stored in f32, so
    T_final = 1 - ra_f32 carries a relative ~3e-4 quantization error at pixels near
    termination (T~1e-4), amplified in the sign-cancelling sums of v_colors — measured
    max per-element relative error vs the golden (internal f64 T): raster_std 2.52e-3
    → rtol 5e-3 (2.0x margin), raster_dense 6.79e-3 → rtol 1e-2 (1.5x margin; the
    deterministic component dominates — atomic jitter ~1e-4). A re-run with an f64
    chain + the exact T_final matches the golden to ~6e-8 (formula agreement confirmed).
    raster_small has 0 violations (max_rel 1.2e-4) — starting value kept. rel_l2 (1e-3)
    passes on every fixture (≤7.3e-5) and is unchanged.
    """
    tol = {k: dict(v) for k, v in _RASTER_TOLERANCES.items()}
    tol["v_colors"]["rtol"] = v_colors_rtol
    return tol
_RASTER_TENSORS = {
    "inputs": ["means2d", "conics", "colors", "opacities", "flatten_ids", "isect_offsets"],
    "outputs": ["render_colors", "render_alphas", "last_ids"],
    "grad_outputs": ["v_render_colors", "v_render_alphas"],
    "grad_inputs": ["v_means2d", "v_conics", "v_colors", "v_opacities"],
}


def _rasterize_reference(means2d, conics, colors, opacities, flatten_ids_sorted, offsets,
                         W, H, tile_size, backgrounds=None, masks=None):
    """Self-contained differentiable raster golden (07 §_rasterize_reference + Corrections 4/6, the formula section of 05).

    Tile-wise vectorized front-to-back alpha blending — isomorphic to the CUDA fwd
    kernel (CTA-per-tile). Assumes an f64 graph (doubles as the backward golden).
    The hard gates (valid/early termination) are decided on detached values so that
    autograd matches the gate semantics of the CUDA bwd (07 caveat).

    Inputs: means2d [I,N,2], conics [I,N,3], colors [I,N,C], opacities [I,N] (all f64),
          flatten_ids_sorted int32/64 [n_isects] (the sorted golden), offsets int32 [I,TH,TW],
          backgrounds [I,C] or None. masks is unsupported in v1 (07 Correction 6).
    Returns: (render_colors [I,H,W,C], render_alphas [I,H,W,1], last_ids int32 [I,H,W],
           aux={"T_final" [I,H,W] f64, "terminated" [I,H,W] bool} — for self-verification/statistics, detached)
    """
    assert masks is None, "v1 fixtures fix masks=None (07 Correction 6)"
    assert means2d.dtype == torch.float64, "the golden assumes an f64 graph (07)"
    I, TH, TW = offsets.shape
    C = colors.shape[-1]
    n_isects = int(flatten_ids_sorted.shape[0])
    offs = torch.cat([offsets.reshape(-1).to(torch.int64),
                      torch.tensor([n_isects], dtype=torch.int64)])   # global cumulative + sentinel
    flat_xy = means2d.reshape(-1, 2)
    flat_con = conics.reshape(-1, 3)
    flat_rgb = colors.reshape(-1, C)
    flat_op = opacities.reshape(-1)

    out_c = torch.zeros(I, H, W, C, dtype=torch.float64)
    out_a = torch.zeros(I, H, W, 1, dtype=torch.float64)
    last = torch.zeros(I, H, W, dtype=torch.int32)
    T_final_map = torch.ones(I, H, W, dtype=torch.float64)
    terminated = torch.zeros(I, H, W, dtype=torch.bool)

    for img in range(I):
        for ty in range(TH):
            for tx in range(TW):
                t = (img * TH + ty) * TW + tx
                rs, re = int(offs[t]), int(offs[t + 1])
                if rs == re:
                    continue          # empty tile: keep out=0/last=0 — the background is outside the loop (Correction 4)
                gid = flatten_ids_sorted[rs:re].to(torch.int64)
                xy, con = flat_xy[gid], flat_con[gid]
                opac, rgb = flat_op[gid], flat_rgb[gid]
                # in-bounds pixels within the tile (clipped at the image boundary — partial tiles)
                ys = torch.arange(ty * tile_size, min((ty + 1) * tile_size, H))
                xs = torch.arange(tx * tile_size, min((tx + 1) * tile_size, W))
                yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                yy, xx = yy.reshape(-1), xx.reshape(-1)
                px = xx.to(torch.float64) + 0.5           # pixel center (Fwd.cu:108/117)
                py = yy.to(torch.float64) + 0.5
                dx = xy[None, :, 0] - px[:, None]         # delta = mean - pixel (sign as-is)
                dy = xy[None, :, 1] - py[:, None]         # [P,R]
                sigma = (0.5 * (con[None, :, 0] * dx * dx + con[None, :, 2] * dy * dy)
                         + con[None, :, 1] * dx * dy)     # Device.cuh:47 (cross term +B dx dy)
                alpha = torch.clamp_max(opac[None, :] * torch.exp(-sigma), _MAX_ALPHA)
                # hard gates are decided on detached values (the CUDA bwd is also a hard cut — 07 caveat)
                skip = (sigma.detach() < 0) | (alpha.detach() < _ALPHA_THRESHOLD)
                a_eff = torch.where(skip, alpha.new_zeros(()), alpha)
                one_m = 1.0 - a_eff
                P, R = one_m.shape
                T_ex = torch.cat([one_m.new_ones(P, 1),
                                  one_m.cumprod(dim=1)[:, :-1]], dim=1)   # exclusive T
                next_T = T_ex * one_m
                dead = (next_T.detach() <= _TRANSMITTANCE_THRESHOLD).to(torch.int32) \
                    .cummax(dim=1).values.to(torch.bool)  # exclusive: the trigger itself is also excluded (Fwd.cu:246-251)
                w = a_eff * T_ex * (~dead).to(a_eff.dtype)
                # ---- self-verification (a): T decays monotonically + surviving states have T > threshold ----
                T_ex_d = T_ex.detach()
                assert bool((T_ex_d[:, 1:] <= T_ex_d[:, :-1] + 1e-15).all()), "T monotonicity violated"
                assert bool((T_ex_d[~dead] > _TRANSMITTANCE_THRESHOLD).all()), \
                    "T <= 1e-4 in a surviving state — exclusive early-termination semantics violated"
                out_c[img, yy, xx] = (w.unsqueeze(-1) * rgb[None]).sum(dim=1)
                a_pix = w.sum(dim=1)                      # == 1 - T_final (mathematically equivalent)
                out_a[img, yy, xx, 0] = a_pix
                contrib = w.detach() > 0
                last_rel = torch.where(contrib, torch.arange(R)[None, :],
                                       torch.tensor(-1)).max(dim=1).values
                last[img, yy, xx] = torch.where(contrib.any(dim=1), rs + last_rel,
                                                torch.tensor(0)).to(torch.int32)
                T_final_map[img, yy, xx] = torch.where(dead, one_m.new_ones(()),
                                                       one_m).detach().prod(dim=1)
                terminated[img, yy, xx] = dead[:, -1]
    # ---- self-verification (b): the render_alphas == 1 - T_final identity (f64) ----
    assert torch.allclose(out_a.detach()[..., 0], 1.0 - T_final_map, rtol=0.0, atol=1e-12), \
        "render_alphas != 1 - T_final"
    if backgrounds is not None:
        # Correction 4: the background is applied globally outside the tile loop — bg is written to empty tiles' in-bounds pixels as well
        out_c = out_c + (1.0 - out_a) * backgrounds[:, None, None, :]
    return out_c, out_a, last, {"T_final": T_final_map, "terminated": terminated}


def _rasterize_scalar_oracle(means2d, conics, colors, opacities, flatten_ids_sorted, offsets,
                             W, H, tile_size, backgrounds=None):
    """Serial per-pixel numpy/f64 oracle — a literal translation of the CUDA fwd (Fwd.cu:179-296) (a per-pixel sequential loop + break).

    The 'golden of the golden' for the vectorized reference (_rasterize_reference)
    (a defense for 07 open question #7): catches cumprod/cummax vectorization bugs
    (off-by-one, dead latch, sort consumption order) through an independent path.
    The background follows the CUDA formula per pixel, `pix + T_final*bg` — the
    comparison itself verifies that this is mathematically equivalent (within
    floating-point error) to the reference's global (1-alpha)·bg.
    """
    I, TH, TW = offsets.shape
    C = colors.shape[-1]
    xy = means2d.astype(np.float64).reshape(-1, 2)
    con = conics.astype(np.float64).reshape(-1, 3)
    rgb = colors.astype(np.float64).reshape(-1, C)
    op = opacities.astype(np.float64).reshape(-1)
    flat = flatten_ids_sorted.astype(np.int64)
    offs = np.append(offsets.astype(np.int64).reshape(-1), len(flat))
    bg = None if backgrounds is None else backgrounds.astype(np.float64)
    rc = np.zeros((I, H, W, C))
    ra = np.zeros((I, H, W, 1))
    last = np.zeros((I, H, W), dtype=np.int32)
    for img in range(I):
        for ty in range(TH):
            for tx in range(TW):
                t = (img * TH + ty) * TW + tx
                rs, re = int(offs[t]), int(offs[t + 1])
                for i in range(ty * tile_size, min((ty + 1) * tile_size, H)):
                    for j in range(tx * tile_size, min((tx + 1) * tile_size, W)):
                        px, py = j + 0.5, i + 0.5
                        T, cur, has = 1.0, 0, False
                        pix = [0.0] * C
                        for idx in range(rs, re):
                            g = int(flat[idx])
                            dxv, dyv = xy[g, 0] - px, xy[g, 1] - py
                            s = (0.5 * (con[g, 0] * dxv * dxv + con[g, 2] * dyv * dyv)
                                 + con[g, 1] * dxv * dyv)
                            a = min(_MAX_ALPHA, op[g] * math.exp(-s))
                            if s < 0.0 or a < _ALPHA_THRESHOLD:
                                continue
                            nT = T * (1.0 - a)
                            if nT <= _TRANSMITTANCE_THRESHOLD:
                                break                     # exclusive early termination
                            for k in range(C):
                                pix[k] += rgb[g, k] * a * T
                            cur, has, T = idx, True, nT
                        ra[img, i, j, 0] = 1.0 - T
                        for k in range(C):
                            rc[img, i, j, k] = pix[k] + (T * bg[img, k] if bg is not None else 0.0)
                        last[img, i, j] = cur if has else 0
    return rc, ra, last


def _isect_chain(ti, means2d, radii, depths, tile_size, tile_width, tile_height):
    """The isect chain for synthetic raster fixtures: _isect_tiles(sort=False) → stable sort → offsets.

    Asserts no tied keys — guarantees depth uniqueness so raster goldens do not depend
    on the tie order of the sort.
    Returns: (flatten_ids_sorted np.int32, offsets torch.int32 [I,TH,TW], tiles_per_gauss)
    """
    I = means2d.shape[0]
    tiles_per_gauss, isect_ids, flatten_ids = ti._isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height, sort=False)
    ids_np = isect_ids.numpy()
    assert ids_np.shape[0] > 0, "n_isects == 0: degenerate fixture"
    order = _stable_sort_keys(ids_np)
    ids_sorted = ids_np[order]
    assert np.unique(ids_sorted).size == ids_sorted.size, \
        "tied keys found — raster fixtures assume unique depths (tie verification is isect_ties' job)"
    flat_sorted = flatten_ids.numpy()[order].astype(np.int32)
    offsets = ti._isect_offset_encode(
        torch.from_numpy(ids_sorted.copy()), I, tile_width, tile_height)
    return flat_sorted, offsets, tiles_per_gauss


def _raster_golden(out, *, means2d, conics, colors, opacities, flat_np, offsets,
                   W, H, tile_size, backgrounds, grad_seed, aux_inputs=None):
    """Shared dump for raster fixtures: save the inputs → the f64 reference fwd (with
    self-verifications a/b built in) → the all-pixel scalar-oracle comparison
    (self-verification c) → save the fwd goldens + the bwd goldens (dump_backward).

    Returns (rc, ra, last, aux, grads) — rc/ra are f64 graph tensors, grads is the dict of expected f64 grads.
    """
    out.mkdir(parents=True, exist_ok=True)
    for name, t in [("means2d", means2d), ("conics", conics),
                    ("colors", colors), ("opacities", opacities)]:
        _save_f32(out, name, t)
    save_sxt(str(out / "flatten_ids.sxt"), flat_np.astype(np.int32))
    save_sxt(str(out / "isect_offsets.sxt"), offsets.numpy().astype(np.int32))
    if backgrounds is not None:
        _save_f32(out, "backgrounds", backgrounds)
    for name, t in (aux_inputs or {}).items():
        if t.dtype == torch.int32:
            save_sxt(str(out / f"{name}.sxt"), t.numpy().astype(np.int32))
        else:
            _save_f32(out, name, t)

    m64 = means2d.double().clone().requires_grad_(True)
    con64 = conics.double().clone().requires_grad_(True)
    col64 = colors.double().clone().requires_grad_(True)
    op64 = opacities.double().clone().requires_grad_(True)
    bg64 = None if backgrounds is None else backgrounds.double()
    rc, ra, last, aux = _rasterize_reference(
        m64, con64, col64, op64, torch.from_numpy(flat_np), offsets,
        W, H, tile_size, backgrounds=bg64)

    # ---- self-verification (c): all-pixel comparison against the serial per-pixel scalar oracle ----
    orc, ora, olast = _rasterize_scalar_oracle(
        means2d.numpy(), conics.numpy(), colors.numpy(), opacities.numpy(),
        flat_np, offsets.numpy(), W, H, tile_size,
        None if backgrounds is None else backgrounds.numpy())
    assert np.allclose(rc.detach().numpy(), orc, rtol=1e-9, atol=1e-11), "oracle render_colors mismatch"
    assert np.allclose(ra.detach().numpy(), ora, rtol=1e-9, atol=1e-11), "oracle render_alphas mismatch"
    assert np.array_equal(last.numpy(), olast), "oracle last_ids mismatch"

    _save_f32(out, "render_colors", rc)
    _save_f32(out, "render_alphas", ra)
    save_sxt(str(out / "last_ids.sxt"), last.numpy().astype(np.int32))
    grads = dump_backward({"render_colors": rc, "render_alphas": ra},
                          {"means2d": m64, "conics": con64, "colors": col64, "opacities": op64},
                          out, grad_seed)
    return rc, ra, last, aux, grads


def _raster_fd_check(out, *, means2d, conics, colors, opacities, flat_np, offsets,
                     W, H, tile_size, backgrounds, grads, n_probe=12, seed=97,
                     eps=1e-6, atol=1e-6, rtol=1e-5):
    """Finite-difference check of the bwd goldens (a self-verification extension) — the
    coordinate-wise central difference of L = Σ v_render_colors·rc + Σ v_render_alphas·ra
    vs the expected autograd grads. Independently confirms that the detach gates (skip/dead)
    and clamp_max did not block a legitimate grad path. Loads the upstream bytes stored in
    the fixture and reconstructs the same L as dump_backward. Assumes generic positions far
    from the gate boundaries (a boundary coordinate in the sample fails deterministically →
    redesign the input)."""
    v_rc = torch.from_numpy(load_sxt(str(out / "v_render_colors.sxt"))).double()
    v_ra = torch.from_numpy(load_sxt(str(out / "v_render_alphas.sxt"))).double()
    flat_t = torch.from_numpy(flat_np)
    bg64 = None if backgrounds is None else backgrounds.double()
    base = {"means2d": means2d.double(), "conics": conics.double(),
            "colors": colors.double(), "opacities": opacities.double()}

    def loss(tensors):
        rc, ra, _, _ = _rasterize_reference(
            tensors["means2d"], tensors["conics"], tensors["colors"], tensors["opacities"],
            flat_t, offsets, W, H, tile_size, backgrounds=bg64)
        return float((v_rc * rc).sum() + (v_ra * ra).sum())

    rng = np.random.default_rng(seed)
    for name, t in base.items():
        flat = t.reshape(-1)
        idxs = rng.choice(flat.numel(), size=min(n_probe, flat.numel()), replace=False)
        for i in idxs:
            i = int(i)
            plus = {k: v.clone() for k, v in base.items()}
            minus = {k: v.clone() for k, v in base.items()}
            plus[name].reshape(-1)[i] += eps
            minus[name].reshape(-1)[i] -= eps
            fd = (loss(plus) - loss(minus)) / (2.0 * eps)
            gd = float(grads[name].reshape(-1)[i])
            assert abs(fd - gd) <= atol + rtol * abs(gd), \
                f"FD check failed: {name}[{i}] fd={fd} grad={gd}"


@fixture("raster_small")
def gen_raster_small(math_mod, ti):
    """A hand-checkable small raster fixture — I=1, 32×32px, 2×2 tiles, N=8 hand-placed (no random numbers).

    Cases: (a) g0..g2 overlapping at the same center — the MAX_ALPHA clamp (g0) + exclusive
    early termination (g2 excluded), (b) g4 small within a single tile, (c) g3 straddling
    2 tiles vertically, (d) g6 half off-screen on the left, (e) tile (1,1) empty (the
    Correction 4 path in the background variant), (f) g5 always invalid with opac<1/255,
    (g) g7 anisotropic conic (a B≠0 cross-term sign). 2 variants: nobg / bg (identical
    inputs and the same grad seed 1031 → identical upstream bytes; v_colors must be
    identical between the variants — the bg term touches only the alpha path).
    """
    base = FIXTURES_DIR / "raster_small"
    I, W, H, tile_size = 1, 32, 32, 16
    TW, TH = 2, 2
    #        x      y     A     B     C    opac   color              depth  r
    rows = [
        (8.5,  8.5,  0.5,  0.0,  0.5,  0.995, (1.0, 0.0, 0.0),  1.0,  4),  # g0 (a) front, clamped
        (8.5,  8.5,  0.5,  0.0,  0.5,  0.95,  (0.0, 1.0, 0.0),  1.1,  4),  # g1 (a) middle
        (8.5,  8.5,  0.5,  0.0,  0.5,  0.95,  (0.0, 0.0, 1.0),  1.2,  4),  # g2 (a) back — excluded by early termination
        (8.0,  16.0, 0.2,  0.0,  0.2,  0.8,   (1.0, 1.0, 0.0),  2.0,  6),  # g3 (c) straddles 2 tiles vertically
        (24.5, 8.5,  1.0,  0.0,  1.0,  0.9,   (0.0, 1.0, 1.0),  1.5,  3),  # g4 (b) small, single tile
        (28.5, 12.5, 0.5,  0.0,  0.5,  0.003, (1.0, 0.0, 1.0),  1.3,  3),  # g5 (f) opac<1/255 always invalid
        (0.5,  24.5, 0.3,  0.0,  0.3,  0.7,   (0.2, 0.4, 0.6),  1.7,  4),  # g6 (d) left half off-screen
        (8.5,  24.5, 0.4,  0.15, 0.3,  0.85,  (0.9, 0.5, 0.1),  1.9,  5),  # g7 (g) anisotropic B≠0
    ]
    N = len(rows)
    means2d = torch.tensor([[r[0], r[1]] for r in rows], dtype=torch.float32)[None]
    conics = torch.tensor([[r[2], r[3], r[4]] for r in rows], dtype=torch.float32)[None]
    opacities = torch.tensor([r[5] for r in rows], dtype=torch.float32)[None]
    colors = torch.tensor([list(r[6]) for r in rows], dtype=torch.float32)[None]
    depths = torch.tensor([r[7] for r in rows], dtype=torch.float32)[None]
    radii = torch.tensor([[r[8], r[8]] for r in rows], dtype=torch.int32)[None].expand(1, N, 2).contiguous()

    flat_np, offsets, tpg = _isect_chain(ti, means2d, radii, depths, tile_size, TW, TH)
    # isect hand check: pins the per-tile depth sort result and the offsets structure bit-for-bit
    assert tpg.reshape(-1).tolist() == [1, 1, 1, 2, 1, 1, 1, 1]
    assert flat_np.tolist() == [0, 1, 2, 3, 5, 4, 6, 7, 3], f"unexpected flatten order: {flat_np.tolist()}"
    assert offsets.reshape(-1).tolist() == [0, 4, 6, 9]
    n_isects = len(flat_np)

    # ---- self-verification (d) expected values: scalar hand computation (f64-upcast arithmetic on the stored f32 values) ----
    f32 = lambda x: float(np.float32(x))
    # pixel (x=8,y=8) center (8.5,8.5) = the g0..g2 center → sigma=0 exactly, vis=1
    a0 = min(_MAX_ALPHA, f32(0.995))         # clamp active → 0.99
    T1 = 1.0 - a0
    a1 = min(_MAX_ALPHA, f32(0.95))
    w1 = a1 * T1
    T2 = T1 * (1.0 - a1)
    next_g2 = T2 * (1.0 - a1)                # g2's next_T
    assert next_g2 <= _TRANSMITTANCE_THRESHOLD * 0.5, "insufficient early-termination margin (f32 flake risk)"
    exp88_rgb = np.array([a0, w1, 0.0])      # w0*(1,0,0) + w1*(0,1,0)
    exp88_a = a0 + w1
    # pixel (x=24,y=8) center (24.5,8.5) = the g4 center; g5 (d=1.3) has opac 0.003<1/255 → invalid
    a4 = min(_MAX_ALPHA, f32(0.9))
    exp248_rgb = np.array([0.0, a4, a4])
    exp248_a = a4
    bg = torch.tensor([[0.25, 0.5, 0.75]], dtype=torch.float32)   # [I,3]
    bg64 = bg.double()[0].numpy()

    results = {}
    for variant, bgt in (("nobg", None), ("bg", bg)):
        out = base / variant
        rc, ra, last, aux, grads = _raster_golden(
            out, means2d=means2d, conics=conics, colors=colors, opacities=opacities,
            flat_np=flat_np, offsets=offsets, W=W, H=H, tile_size=tile_size,
            backgrounds=bgt, grad_seed=1031,
            aux_inputs={"radii": radii, "depths": depths})
        rc_d = rc.detach().numpy()
        # ---- self-verification (d): single-pixel hand-check assertions ----
        e88 = exp88_rgb + (1.0 - exp88_a) * bg64 if bgt is not None else exp88_rgb
        e248 = exp248_rgb + (1.0 - exp248_a) * bg64 if bgt is not None else exp248_rgb
        assert np.allclose(rc_d[0, 8, 8], e88, rtol=0.0, atol=1e-12), \
            f"hand check (8,8) mismatch: {rc_d[0, 8, 8]} != {e88}"
        assert abs(float(ra.detach()[0, 8, 8, 0]) - exp88_a) < 1e-12
        assert int(last[0, 8, 8]) == 1, "last_ids(8,8) != 1 (g1 is the last contributor)"
        assert np.allclose(rc_d[0, 8, 24], e248, rtol=0.0, atol=1e-12), \
            f"hand check (24,8) mismatch: {rc_d[0, 8, 24]} != {e248}"
        assert abs(float(ra.detach()[0, 8, 24, 0]) - exp248_a) < 1e-12
        assert int(last[0, 8, 24]) == 5, "last_ids(24,8) != 5 (g4 = absolute isect index 4+1)"
        # representative pixel (24,24) of the empty tile (1,1): the Correction 4 path
        if bgt is None:
            assert float(rc[0, 24, 24].detach().abs().max()) == 0.0
        else:
            assert torch.equal(rc[0, 24, 24].detach(), bg.double()[0]), "bg not written to the empty tile (Correction 4)"
        assert float(ra.detach()[0, 24, 24, 0]) == 0.0 and int(last[0, 24, 24]) == 0
        # g5 (always invalid) contributes to no grad
        for gname in ("means2d", "conics", "colors", "opacities"):
            assert float(grads[gname].reshape(N, -1)[5].abs().max()) == 0.0, f"g5 {gname} grad != 0"
        assert bool(aux["terminated"][0, 8, 8]), "no early termination at pixel (8,8)"
        # ---- finite-difference check of the bwd goldens (48 coordinates/variant) ----
        _raster_fd_check(out, means2d=means2d, conics=conics, colors=colors,
                         opacities=opacities, flat_np=flat_np, offsets=offsets,
                         W=W, H=H, tile_size=tile_size, backgrounds=bgt, grads=grads)
        results[variant] = (rc, ra, last, grads)

        hand_check = {
            "pixel_x8_y8": {
                "tile": [0, 0], "isect_range": [0, 4],
                "trace": "g0: alpha=min(0.99,0.995)=0.99 (clamped), next_T=0.01>1e-4 → w0=0.99; "
                         "g1: alpha=f32(0.95), next_T≈5.0e-4>1e-4 → w1=alpha·0.01; "
                         "g2: next_T≈2.5e-5<=1e-4 → exclusive termination (g2 not included); "
                         "g3: terminated before it is reached (note: alpha≈2.81e-3<1/255 makes it invalid anyway)",
                "render_colors": [float(v) for v in e88],
                "render_alphas": exp88_a, "last_ids": 1,
            },
            "pixel_x24_y8": {
                "tile": [1, 0], "isect_range": [4, 6],
                "trace": "g5 (depth 1.3) first: opac 0.003<1/255 → invalid, skipped (last not updated); "
                         "g4: alpha=f32(0.9), w=alpha → last=4+1=5",
                "render_colors": [float(v) for v in e248],
                "render_alphas": exp248_a, "last_ids": 5,
            },
            "pixel_x24_y24_empty_tile": {
                "tile": [1, 1], "isect_range": [9, 9],
                "render_colors": ([float(v) for v in bg64] if bgt is not None else [0.0, 0.0, 0.0]),
                "render_alphas": 0.0, "last_ids": 0,
                "trace": "empty tile — nobg: 0, bg: the background is applied globally (07 Correction 4)",
            },
        }
        _write_meta(out, {
            "golden_source": _RASTER_GOLDEN_SOURCE,
            "I": I, "N": N, "cdim": 3, "width": W, "height": H,
            "tile_size": tile_size, "tile_width": TW, "tile_height": TH,
            "n_isects": n_isects, "has_background": 1 if bgt is not None else 0,
            "masks": 0,
            "seeds": {"grad": 1031},
            "isect": {"offsets_flat": [0, 4, 6, 9], "flatten_ids": [0, 1, 2, 3, 5, 4, 6, 7, 3],
                      "note": "Hand-made depths give unique keys — per-tile front-to-back = "
                              "(0,0):[g0,g1,g2,g3] (1,0):[g5,g4] (0,1):[g6,g7,g3] (1,1):empty"},
            "cases": "(a) g0..g2 overlapping (clamp + early termination) (b) g4 small "
                     "(c) g3 straddling 2 tiles (d) g6 half off-screen (e) tile (1,1) empty "
                     "(f) g5 always invalid (g) g7 anisotropic B=0.15",
            "hand_check": hand_check,
            "tensors": {**_RASTER_TENSORS,
                        "inputs": _RASTER_TENSORS["inputs"] + (["backgrounds"] if bgt is not None else []),
                        "inputs_aux": ["radii", "depths"]},
            "note": _RASTER_NOTE + " " + _RASTER_SIZE_NOTE +
                    " The nobg and bg variants share the same input and grad seeds (byte-identical "
                    "upstream) — v_colors is identical between the variants, and only "
                    "v_means2d/v_conics/v_opacities diverge through the bg term (v_alpha).",
            "tolerances": _raster_tolerances(),  # raster_small: starting values kept (0 measured violations)
        })
        print(f"wrote {out} (n_isects={n_isects})")

    # ---- cross-variant relation assertions (a self-verification c extension) ----
    (rc_n, ra_n, last_n, g_n), (rc_b, ra_b, last_b, g_b) = results["nobg"], results["bg"]
    assert torch.equal(ra_n.detach(), ra_b.detach()) and torch.equal(last_n, last_b)
    assert torch.equal(rc_b.detach(),
                       rc_n.detach() + (1.0 - ra_n.detach()) * bg.double()[:, None, None, :]), \
        "background application relation violated: rc_bg != rc_nobg + (1-alpha)·bg"
    assert torch.allclose(g_n["colors"], g_b["colors"], rtol=0.0, atol=1e-14), \
        "v_colors differs in the bg variant (the bg term must touch only the alpha path)"


@fixture("raster_std")
def gen_raster_std(math_mod, ti):
    """A realistic-distribution raster fixture — chained from proj_fully_fused (means2d/conics)
    + isect_std (the sorted golden/offsets), I=2, 160×120, 10×8 tiles. colors/opacities are
    synthetic (seed 23) — synthetic values are used instead of the SH golden colors (the
    tensor lineage is meta.lineage). fwd + bwd."""
    proj = FIXTURES_DIR / "proj_fully_fused"
    isec = FIXTURES_DIR / "isect_std"
    out = FIXTURES_DIR / "raster_std"
    for d, f in ((proj, "means2d"), (proj, "conics"),
                 (isec, "flatten_ids_sorted"), (isec, "offsets")):
        if not (d / f"{f}.sxt").exists():
            raise SystemExit("raster_std chains from proj_fully_fused + isect_std — generate them first")
    im = json.loads((isec / "meta.json").read_text())
    W, H, tile_size = im["width"], im["height"], im["tile_size"]
    TW, TH, I = im["tile_width"], im["tile_height"], im["I"]

    means2d = torch.from_numpy(load_sxt(str(proj / "means2d.sxt")))       # [2,128,2] f32
    conics = torch.from_numpy(load_sxt(str(proj / "conics.sxt")))         # [2,128,3] f32
    flat_np = load_sxt(str(isec / "flatten_ids_sorted.sxt"))              # [n_isects] i32
    offsets = torch.from_numpy(load_sxt(str(isec / "offsets.sxt")))       # [2,8,10] i32
    N = means2d.shape[1]
    assert len(flat_np) == im["n_isects"]

    g = torch.Generator().manual_seed(23)
    colors = torch.rand(I, N, 3, generator=g)
    opacities = torch.rand(I, N, generator=g) * 0.93 + 0.05
    # The 8 most-intersected per image are fixed at opac=0.99 → guarantees saturated pixels
    # in both images (the 07 size table). With 4 per image min T_final≈2.3e-4 > 1e-4 and no
    # early termination occurs (measured) — with 8, 69/77 pixels terminate per image while
    # most do not (the mixed bwd bin_final truncation requirement).
    counts = np.bincount(flat_np.astype(np.int64), minlength=I * N)
    boost = np.concatenate([np.argsort(-counts[:N], kind="stable")[:8],
                            N + np.argsort(-counts[N:], kind="stable")[:8]])
    opacities.reshape(-1)[torch.from_numpy(boost)] = 0.99

    rc, ra, last, aux, grads = _raster_golden(
        out, means2d=means2d, conics=conics, colors=colors, opacities=opacities,
        flat_np=flat_np, offsets=offsets, W=W, H=H, tile_size=tile_size,
        backgrounds=None, grad_seed=1023)

    term = aux["terminated"]
    n_term = int(term.sum())
    assert 0 < n_term < I * H * W, "a mix of saturated and unsaturated pixels is required (the 07 thread table: bwd bin_final truncation)"
    for img in range(I):
        assert int(term[img].sum()) > 0, f"no saturated pixel in image {img}"
        assert float(ra[img].detach().max()) > 0.5, f"degenerate contribution in image {img}"
    assert int(last.max()) < len(flat_np)
    assert float(ra.detach().min()) >= 0.0 and float(ra.detach().max()) <= 1.0

    _write_meta(out, {
        "golden_source": _RASTER_GOLDEN_SOURCE,
        "I": I, "N": N, "cdim": 3, "width": W, "height": H,
        "tile_size": tile_size, "tile_width": TW, "tile_height": TH,
        "n_isects": im["n_isects"], "has_background": 0, "masks": 0,
        "seeds": {"input": 23, "grad": 1023},
        "lineage": {
            "means2d": "a byte-identical copy of proj_fully_fused/means2d.sxt (the fwd golden, seed 7)",
            "conics": "a byte-identical copy of proj_fully_fused/conics.sxt",
            "flatten_ids": "a byte-identical copy of isect_std/flatten_ids_sorted.sxt (the sorted golden)",
            "isect_offsets": "a byte-identical copy of isect_std/offsets.sxt",
            "colors": "synthetic rand[0,1) seed 23 (not chained from the SH golden — only the [I,N,3] shape lineage)",
            "opacities": "synthetic rand*0.93+0.05 seed 23 + the 8 most-intersected per image set to opac=0.99",
        },
        "opac_boost_flat_ids": sorted(int(i) for i in boost),
        "n_terminated_per_image": [int(term[i].sum()) for i in range(I)],
        "n_terminated_pixels": n_term,
        "tensors": _RASTER_TENSORS,
        "note": _RASTER_NOTE + " " + _RASTER_SIZE_NOTE,
        # v_colors rtol 5e-3 — the 1c Task 4 measured calibration (rationale in the _raster_tolerances docstring)
        "tolerances": _raster_tolerances(v_colors_rtol=5e-3),
    })
    print(f"wrote {out} (n_isects={im['n_isects']}, terminated={n_term}/{I*H*W})")


@fixture("raster_dense")
def gen_raster_dense(math_mod, ti):
    """A single-tile high-density raster fixture — I=1, 16×16px, 1×1 tiles, N=640 (every Gaussian intersects the single tile).

    n_isects=640 > 2·BATCH_SIZE(256) → 3 iterations of the cooperative batch load loop +
    atomic contention on the same Gaussian across the 8 simdgroups per threadgroup (bwd).
    The dense front cluster (idx 32..39, opac 0.9) creates early-terminated pixels, and the
    low-opacity distant pixels make full survival across all 640. idx 0..31 have opac<1/255
    and are always invalid — the valid-gate skip path across a whole batch range.
    idx 40 is a **non-PSD conic** (det = A·C − B² < 0, hyperbolic contours — Phase 2 Task 1):
    demonstrates the sigma<0 branch of the valid gate — at some pixels it is skipped by
    sigma<0 alone even though alpha is large enough (≥1/255), and it actually contributes
    at sigma>=0 pixels (asserted by the generator)."""
    out = FIXTURES_DIR / "raster_dense"
    I, W, H, tile_size = 1, 16, 16, 16
    TW = TH = 1
    N = 640
    g = torch.Generator().manual_seed(29)

    means2d = torch.rand(I, N, 2, generator=g) * 16.0
    ca = torch.rand(I, N, generator=g) * 0.45 + 0.15
    cc = torch.rand(I, N, generator=g) * 0.45 + 0.15
    cb = (torch.rand(I, N, generator=g) * 2.0 - 1.0) * 0.5 * torch.sqrt(ca * cc)  # det >= 0.75·A·C > 0
    conics = torch.stack([ca, cb, cc], dim=-1)
    colors = torch.rand(I, N, 3, generator=g)
    opacities = torch.rand(I, N, generator=g) * 0.11 + 0.01
    depths = torch.rand(I, N, generator=g) * 10.0 + 0.5
    opacities[0, :32] = torch.rand(32, generator=g) * 0.002 + 0.001   # < 1/255 always invalid
    opacities[0, 32:40] = 0.9                                          # dense front cluster
    means2d[0, 32:40] = torch.rand(8, 2, generator=g) * 3.0 + 2.5      # cluster near (4,4)
    depths[0, 32:40] = torch.rand(8, generator=g) * 0.05 + 0.01        # frontmost
    # The non-PSD conic case (Phase 2 Task 1 — demonstrates the sigma<0 valid-gate branch):
    # det = 0.3·0.3 − 0.5² = −0.16 < 0 (hyperbolic). At (11.5,11.5), away from the front
    # cluster; depth 0.2 (behind the cluster, in front of the rest — not hidden by early
    # termination); opac 0.8 (when the gate sees sigma<0 alpha is clamped to 0.99 ≥ 1/255 —
    # sigma is the only blocker).
    NON_PSD_ID = 40
    conics[0, NON_PSD_ID] = torch.tensor([0.3, 0.5, 0.3])
    opacities[0, NON_PSD_ID] = 0.8
    means2d[0, NON_PSD_ID] = torch.tensor([11.5, 11.5])
    depths[0, NON_PSD_ID] = 0.2
    radii = torch.randint(2, 7, (I, N, 2), generator=g, dtype=torch.int32)

    # ---- assert the sigma<0 gate fires (f64 — the same skip formula shared by
    # _rasterize_reference/_rasterize_scalar_oracle/the kernel (KernelCommon.h eval_gaussian_weight)) ----
    A40, B40, C40 = (float(v) for v in conics[0, NON_PSD_ID])
    assert A40 * C40 - B40 * B40 < 0.0, "non-PSD intent violated (det >= 0)"
    yy, xx = np.meshgrid(np.arange(H) + 0.5, np.arange(W) + 0.5, indexing="ij")
    dx40 = float(means2d[0, NON_PSD_ID, 0]) - xx
    dy40 = float(means2d[0, NON_PSD_ID, 1]) - yy
    sig40 = 0.5 * (A40 * dx40 * dx40 + C40 * dy40 * dy40) + B40 * dx40 * dy40
    alpha40 = np.minimum(_MAX_ALPHA, float(opacities[0, NON_PSD_ID]) * np.exp(-sig40))
    neg40 = sig40 < 0.0
    assert bool(neg40.any()), "no sigma<0 pixel — the non-PSD gate did not fire"
    assert bool((alpha40[neg40] >= _ALPHA_THRESHOLD).any()), \
        "every sigma<0 pixel also has alpha<1/255 — a pixel where sigma is the only blocker is required"
    assert bool(((~neg40) & (alpha40 >= _ALPHA_THRESHOLD)).any()), \
        "no valid contributing pixel for the non-PSD Gaussian (sigma>=0 ∧ alpha>=1/255)"

    flat_np, offsets, tpg = _isect_chain(ti, means2d, radii, depths, tile_size, TW, TH)
    assert tpg.reshape(-1).tolist() == [1] * N, "every Gaussian must intersect the single tile"
    n_isects = len(flat_np)
    assert n_isects == N == 640
    num_batches = -(-n_isects // 256)
    assert num_batches >= 2, "the 2+ cooperative batch load requirement is not met"

    rc, ra, last, aux, grads = _raster_golden(
        out, means2d=means2d, conics=conics, colors=colors, opacities=opacities,
        flat_np=flat_np, offsets=offsets, W=W, H=H, tile_size=tile_size,
        backgrounds=None, grad_seed=1029,
        aux_inputs={"radii": radii, "depths": depths})

    term = aux["terminated"]
    n_term = int(term.sum())
    assert 0 < n_term < H * W, "a mix of early-terminated and full-range surviving pixels is required"
    max_last = int(last.max())
    assert max_last >= 512, f"no contribution from the 3rd batch (isect>=512): max last_id={max_last}"
    # Always-invalid Gaussians (idx<32) contribute to no grad (the valid gate blocks the graph)
    for gname in ("means2d", "conics", "colors", "opacities"):
        assert float(grads[gname].reshape(N, -1)[:32].abs().max()) == 0.0, \
            f"{gname} grad appeared on an always-invalid Gaussian"
    # The non-PSD Gaussian actually contributes at sigma>=0 pixels (front, high opacity —
    # the grad confirms a partial skip, not a full skip)
    assert float(grads["colors"].reshape(N, -1)[NON_PSD_ID].abs().max()) > 0.0, \
        "no contribution from the non-PSD Gaussian (colors grad == 0)"

    _write_meta(out, {
        "golden_source": _RASTER_GOLDEN_SOURCE,
        "I": I, "N": N, "cdim": 3, "width": W, "height": H,
        "tile_size": tile_size, "tile_width": TW, "tile_height": TH,
        "n_isects": n_isects, "has_background": 0, "masks": 0,
        "seeds": {"input": 29, "grad": 1029},
        "batches_at_cta256": num_batches,
        "always_invalid_ids": [0, 32], "front_cluster_ids": [32, 40],
        "non_psd_conic_id": NON_PSD_ID,
        "n_terminated_pixels": n_term, "max_last_id": max_last,
        "tensors": {**_RASTER_TENSORS, "inputs_aux": ["radii", "depths"]},
        "note": _RASTER_NOTE + " " + _RASTER_SIZE_NOTE +
                " A single tile (tile_n_bits=0) — dedicated to 3 batch-loop iterations, the "
                "early-exit vote, and bwd atomic contention. The always_invalid_ids=[0,32) range "
                "has opac<1/255 and is therefore invalid at every pixel (all expected gradients "
                "are 0 — asserted by the generator). non_psd_conic_id=40 is a hyperbolic conic "
                "with det<0 — at some pixels it is skipped by sigma<0 alone even though "
                "alpha≥1/255 (the generator asserts that the gate fires, and asserts that pixels "
                "with sigma>=0 contribute by colors grad != 0).",
        # v_colors rtol 1e-2 — the 1c Task 4 measured calibration (rationale in the _raster_tolerances docstring)
        "tolerances": _raster_tolerances(v_colors_rtol=1e-2),
    })
    print(f"wrote {out} (n_isects={n_isects}, batches={num_batches}, "
          f"terminated={n_term}/{H*W}, max_last={max_last})")


@fixture("relocation_basic")
def gen_relocation_basic(math_mod, ti):
    """The 06-spec mcmc_relocation golden (MCMC Eq.9) — an f32 hand reproduction of the RelocationCUDA.cu:35-79 formula.

    There is no pure torch reference (relocation.py only calls the CUDA kernel), so the
    kernel formula is reproduced place-by-place in numpy float32 (the same methodology
    as adam_basic's "the formula reproduced by hand in f32"):
    - all arithmetic in float32; pow takes the float path (np.power(f32,f32) → powf —
      06 verification-note Correction 2)
    - the opacity clamp (min_opacity, 1-FLT_EPSILON) comes **before** the scale computation (:57-62)
    - denom_sum is accumulated sequentially in ascending (i,k) order — the same order as
      the CPU reference and the Metal kernel (verification-note remaining gap 2)
    - the sign uses the (k&1) trick — mathematically equal to pow(-1,k) (the golden
      counterpart of the fast-math NaN avoidance point)
    The binoms LUT follows strategy/mcmc.py:95-100 verbatim (math.comb → float32) —
    **bit identity** with the C++ make_binoms_lut() is the contract (McmcTests memcmp).

    Boundary cases (fixed indices; the tests assert the clamp paths):
    [0] opacity 1e-4 < min_opacity → ô lower clamp 0.005
    [1] opacity exactly 1.0 + ratio 1 → ô=1 → upper clamp 1-FLT_EPSILON (safe since denom=ô)
    [2] opacity 0.9999 + ratio 2 → the high-ô (≈0.99) path
    [3] opacity min_opacity + ratio 51 → the maximum double loop (1326 iterations) + lower clamp
    The tail 8 have ratios {1,2,5,10,20,30,40,51} — full range coverage.
    """
    out = FIXTURES_DIR / "relocation_basic"
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(35)
    N, n_max, min_opacity = 64, 51, 0.005

    opacities = torch.rand(N, generator=g) * 0.97 + 0.02          # (0.02, 0.99)
    opacities[0], opacities[1], opacities[2], opacities[3] = 1e-4, 1.0, 0.9999, min_opacity
    scales = torch.rand(N, 3, generator=g) * 0.5 + 0.01           # linear (after exp activation)
    ratios = torch.randint(1, n_max + 1, (N,), generator=g, dtype=torch.int32)
    ratios[0], ratios[1], ratios[2], ratios[3] = 3, 1, 2, 51
    for j, r in enumerate((1, 2, 5, 10, 20, 30, 40, 51)):
        ratios[N - 8 + j] = r

    binoms = torch.zeros((n_max, n_max))                          # mcmc.py:95-100 verbatim
    for n in range(n_max):
        for k in range(n + 1):
            binoms[n, k] = math.comb(n, k)

    op32 = opacities.numpy().astype(np.float32)
    sc32 = scales.numpy().astype(np.float32)
    bin32 = binoms.numpy().astype(np.float32)
    rat = ratios.numpy()
    one = np.float32(1.0)
    flt_eps = np.float32(np.finfo(np.float32).eps)                # FLT_EPSILON
    min_op32 = np.float32(min_opacity)
    new_op = np.empty(N, np.float32)
    new_sc = np.empty((N, 3), np.float32)
    for gi in range(N):
        n_idx = int(rat[gi])
        o = op32[gi]
        n_o = one - np.power(one - o, one / np.float32(n_idx))
        n_o = np.minimum(np.maximum(n_o, min_op32), one - flt_eps)
        new_op[gi] = n_o
        denom = np.float32(0.0)
        for i in range(1, n_idx + 1):
            for k in range(0, i):
                sign = np.float32(-1.0) if (k & 1) else np.float32(1.0)
                term = (sign / np.sqrt(np.float32(k + 1))) * np.power(n_o, np.float32(k + 1))
                denom = np.float32(denom + bin32[i - 1, k] * term)
        coeff = np.float32(o / denom)
        new_sc[gi] = coeff * sc32[gi]
    assert np.all(np.isfinite(new_op)) and np.all(np.isfinite(new_sc))
    assert new_op[0] == min_op32 and new_op[3] == min_op32        # lower clamp fires
    assert new_op[1] == one - flt_eps                             # upper clamp fires

    _save_f32(out, "opacities", opacities)
    _save_f32(out, "scales", scales)
    save_sxt(str(out / "ratios.sxt"), rat.astype(np.int32))
    save_sxt(str(out / "binoms.sxt"), bin32)
    save_sxt(str(out / "new_opacities.sxt"), new_op)
    save_sxt(str(out / "new_scales.sxt"), new_sc)
    _write_meta(out, {
        "golden_source": "gsplat@77ab983:RelocationCUDA.cu:35-79 (the formula reproduced by hand in f32) + "
                         "strategy/mcmc.py:95-100 (binoms LUT)",
        "N": N, "n_max": n_max, "min_opacity": min_opacity,
        "seeds": {"input": 35},
        "note": "The clamp (min_opacity, 1-FLT_EPSILON) comes before the scale computation. denom_sum "
                "is a sequential f32 accumulation in ascending (i,k) order — the same order as CPU and "
                "Metal. binoms is bit-identical to the C++ make_binoms_lut by contract (memcmp in the "
                "test). Boundaries: [0]/[3] hit the lower clamp, [1] hits the upper clamp (o=1.0, "
                "ratio=1), and [3] runs the maximum loop at ratio=51. ratios holds the values "
                "pre-clamped to 1..n_max (a host contract).",
        "tolerances": {
            # new_opacities: a single pow — only ULP-level precise::pow (Metal) / powf (CPU·golden) differences
            "new_opacities": {"rtol": 1e-6, "atol": 1e-7},
            # new_scales: denom_sum accumulates up to 1326 terms × pow — headroom for accumulation error
            "new_scales": {"rtol": 1e-4, "atol": 1e-6},
        },
    })
    print(f"wrote {out} (ratio range {int(rat.min())}..{int(rat.max())})")


@fixture("perturb_basic")
def gen_perturb_basic(math_mod, ti):
    """The 06-spec mcmc_perturb_positions golden — an f32 reproduction of the torch fallback
    formula (the semantics of strategy/ops.py:494-511):
    covar = _quat_scale_to_covar_preci(quats, exp(scales_log)) (_math.py — internal quat
    normalization), w = sigmoid(-k·(sigmoid(logit)-t))·noise_scale,
    positions += einsum("nij,nj->ni", covars, noise·w).

    noise is **stored as an input buffer** (the kernel generates no random numbers — 06 §3):
    with noise injected the kernel is fully deterministic, so bit-identical output across two
    Metal runs is the contract. The covariance composition product (einsum/float3x3 multiply/
    CPU sequential sum) differs per implementation in fma and accumulation order, so it is
    compared with meta tolerances rather than bit-exact (the 06 fma/precise policy).

    Boundary cases: [0] logit -12 (density≈0 → the gate's maximum ≈σ(0.5)) + a large-norm
    quat, [1] logit +12 (density≈1 → σ(-99.5)=1/(1+expf(99.5))=1/inf=0 — the f32 overflow
    extreme, identically 0 on every backend) + a small-norm quat (the normalization path).
    """
    out = FIXTURES_DIR / "perturb_basic"
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(36)
    N = 128
    noise_scale, t, k = 0.05, 0.005, 100.0

    positions = torch.randn(N, 3, generator=g) * 2.0
    quats = torch.randn(N, 4, generator=g)                        # unnormalized (the kernel normalizes)
    quats[0] = torch.tensor([10.0, 0.0, 0.0, 0.0])                # large norm
    quats[1] = torch.tensor([0.01, 0.02, -0.01, 0.005])           # small norm
    scales_log = torch.rand(N, 3, generator=g) * 3.0 - 3.0        # log-scale ∈ [-3, 0]
    opacities_logit = torch.randn(N, generator=g) * 3.0
    opacities_logit[0], opacities_logit[1] = -12.0, 12.0
    noise = torch.randn(N, 3, generator=g)

    covars, _ = math_mod._quat_scale_to_covar_preci(
        quats, torch.exp(scales_log), compute_covar=True, compute_preci=False)
    density = torch.sigmoid(opacities_logit)
    w = torch.sigmoid(-k * (density - t)) * noise_scale
    assert float(w[1]) == 0.0, "the density≈1 gate must be exactly 0 at the f32 extreme"
    delta = torch.einsum("nij,nj->ni", covars, noise * w[:, None])
    positions_out = positions + delta
    assert torch.isfinite(positions_out).all()
    assert float(delta.abs().max()) > 0.0

    for name, tt in [("positions", positions), ("quats", quats), ("scales_log", scales_log),
                     ("opacities_logit", opacities_logit), ("noise", noise),
                     ("positions_out", positions_out)]:
        _save_f32(out, name, tt)
    _write_meta(out, {
        "golden_source": "gsplat@77ab983:strategy/ops.py:494-511 (the torch fallback formula, f32) + "
                         "_math.py:_quat_scale_to_covar_preci",
        "N": N, "noise_scale": noise_scale, "t": t, "k": k,
        "seeds": {"input": 36},
        "note": "noise is an input buffer (the kernel is deterministic — the contract is bit-identical "
                "output across two Metal runs). The covariance composition product is compared with "
                "tolerances because of differences in fma and accumulation order. [1] has gate 0 "
                "(positions unchanged — bit-identical). wxyz is a pre-activation quat, scales_log is in "
                "log space, and opacities_logit is in logit space (the kernel applies exp/sigmoid).",
        "tolerances": {
            "positions_out": {"rtol": 1e-5, "atol": 1e-6},
        },
    })
    print(f"wrote {out} (max|delta|={float(delta.abs().max()):.4g})")


@fixture("ssim_basic")
def gen_ssim_basic(math_mod, ti):
    """The 08-spec SSIM loss golden — a pure torch f64 reproduction of the semantics of the
    gsplat training path (fused_ssim padding="valid") (the fused-ssim code was neither
    cloned nor touched — the call convention is losses.py:185).

    Conventions (as confirmed in gsplat@77ab983 — 08 spec §2):
    - window: 11×11 Gaussian σ=1.5, normalized (losses.py:84-107, separable outer product)
    - constants: C1=0.01², C2=0.03² (losses.py:144-145)
    - padding: **valid** — the map is (H-10)×(W-10) and its mean is the ssim scalar
      (losses.py:185 fused_ssim(..., padding="valid") — differs from the torch fallback's
      (:131) same. The parity target is the CUDA training path = valid)
    - loss combination: lerp(l1, 1-ssim, 0.2) = 0.8·l1 + 0.2·(1-ssim) (simple_trainer.py:171,961)
    - the gradient is w.r.t. img1 (the render) only (the losses.py:184 guard — GT must not require grad)

    fixture: img1 = the [4:28,4:28] crop of raster_small/bg render_colors (24×24×3 — covers
    16 tile boundaries + partial tiles, honoring the <2MiB total-size gate), img2 =
    synthesized GT (low-frequency illumination + a contrast/brightness shift + seed 37
    noise, clamped to [0,1]). The bwd golden is the **authoritative training direction**:
    v_img1 = d(1-ssim)/d(img1) (upstream ≡ the scalar 1 — dump_backward not used, λ not applied).

    Self-verification: (a) ssim(a,a)=1 (f64 1e-12), (b) symmetry bit-identical, (c) a pure-python
    scalar oracle at 3 pixels × 3 channels, (d) the 08-spec analytic bwd derivation (3 coefficient
    maps + conv_transpose) checked against autograd — numerical validation of the derivation,
    (e) central finite differences at 4 coordinates, (f) non-degeneracy (map variance / grad≠0
    at every pixel).
    """
    out = FIXTURES_DIR / "ssim_basic"
    out.mkdir(parents=True, exist_ok=True)
    src = FIXTURES_DIR / "raster_small" / "bg" / "render_colors.sxt"
    if not src.exists():
        raise SystemExit("ssim_basic chains from raster_small — generate it first")
    render = torch.from_numpy(load_sxt(str(src)))            # [1,32,32,3] f32
    img1 = render[:, 4:28, 4:28, :].contiguous()             # [1,24,24,3] f32 crop
    _, H, W, C = img1.shape
    WIN, SIGMA = 11, 1.5
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    # img2 = synthesized GT (f32 arithmetic — the stored values are the input contract)
    g = torch.Generator().manual_seed(37)
    ys = torch.arange(H, dtype=torch.float32)
    xs = torch.arange(W, dtype=torch.float32)
    illum = (0.06 * torch.sin(2.0 * math.pi * xs / 12.0)[None, None, :, None]
             * torch.cos(2.0 * math.pi * ys / 16.0)[None, :, None, None])
    noise = torch.randn(1, H, W, C, generator=g) * 0.03
    img2 = torch.clamp(0.88 * img1 + 0.06 + illum + noise, 0.0, 1.0)
    assert float(img1.min()) >= 0.0 and float(img1.max()) <= 1.0, "img1 violates the [0,1] contract"
    assert float(img2.min()) >= 0.0 and float(img2.max()) <= 1.0, "img2 violates the [0,1] contract"

    # ---- f64 golden (the 07 recipe: stored f32 values → f64 arithmetic) — transposed to NCHW (the gsplat convention) ----
    x = img1.double().permute(0, 3, 1, 2).contiguous().requires_grad_(True)   # render
    y = img2.double().permute(0, 3, 1, 2).contiguous()                        # GT

    xw = torch.arange(WIN, dtype=torch.float64)
    g1d = torch.exp(-((xw - WIN // 2) ** 2) / (2.0 * SIGMA ** 2))
    w1d = g1d / g1d.sum()                                     # normalized 1D (losses.py:84-89)
    assert abs(float(w1d.sum()) - 1.0) < 1e-15
    assert torch.equal(w1d, w1d.flip(0)), "window symmetry violated"
    window = (w1d[:, None] @ w1d[None, :])[None, None].expand(C, 1, WIN, WIN).contiguous()

    def ssim_map_fn(a, b):
        """valid-conv SSIM map [1,C,H-10,W-10] + intermediate maps (08 spec §3 symbols)."""
        mu1 = F.conv2d(a, window, groups=C)                   # padding=0 = valid
        mu2 = F.conv2d(b, window, groups=C)
        sigma1_sq = F.conv2d(a * a, window, groups=C) - mu1 * mu1
        sigma2_sq = F.conv2d(b * b, window, groups=C) - mu2 * mu2
        sigma12 = F.conv2d(a * b, window, groups=C) - mu1 * mu2
        A1, A2 = 2.0 * mu1 * mu2 + C1, 2.0 * sigma12 + C2
        B1, B2 = mu1 * mu1 + mu2 * mu2 + C1, sigma1_sq + sigma2_sq + C2
        return (A1 * A2) / (B1 * B2), (mu1, mu2, A1, A2, B1, B2)

    smap, mids = ssim_map_fn(x, y)
    M = smap.numel()                                          # 1·3·14·14 = 588
    (v_x,) = torch.autograd.grad(1.0 - smap.mean(), x)        # training direction: d(1-ssim)/d(render)
    smap_d = smap.detach()
    ssim_mean = float(smap_d.mean())

    # ---- self-verification (a): ssim(a,a) = 1 ----
    m_id, _ = ssim_map_fn(x.detach(), x.detach())
    assert torch.allclose(m_id, torch.ones_like(m_id), rtol=0.0, atol=1e-12), "ssim(a,a) != 1"
    # ---- self-verification (b): symmetry — the formula is bit-identical under argument exchange ----
    m_sw, _ = ssim_map_fn(y, x.detach())
    assert torch.equal(m_sw, smap_d), "ssim(a,b) != ssim(b,a) (bitwise)"
    # ---- self-verification (c): a pure-python scalar oracle (non-vectorized reimplementation) ----
    wg = [math.exp(-((i - WIN // 2) ** 2) / (2.0 * SIGMA ** 2)) for i in range(WIN)]
    wg = [v / sum(wg) for v in wg]
    xl, yl = x.detach()[0].numpy(), y[0].numpy()              # [C,H,W] f64
    hand_check = {}
    for (pi, pj) in ((0, 0), (7, 7), (13, 5)):
        for c in range(C):
            s_mu1 = s_mu2 = s_11 = s_22 = s_12 = 0.0
            for i in range(WIN):
                for j in range(WIN):
                    w = wg[i] * wg[j]
                    a_, b_ = float(xl[c, pi + i, pj + j]), float(yl[c, pi + i, pj + j])
                    s_mu1 += w * a_; s_mu2 += w * b_
                    s_11 += w * a_ * a_; s_22 += w * b_ * b_; s_12 += w * a_ * b_
            var1, var2 = s_11 - s_mu1 * s_mu1, s_22 - s_mu2 * s_mu2
            cov = s_12 - s_mu1 * s_mu2
            s_oracle = ((2 * s_mu1 * s_mu2 + C1) * (2 * cov + C2)) / \
                       ((s_mu1 * s_mu1 + s_mu2 * s_mu2 + C1) * (var1 + var2 + C2))
            s_conv = float(smap_d[0, c, pi, pj])
            assert abs(s_oracle - s_conv) < 1e-12 * max(1.0, abs(s_conv)), \
                f"scalar oracle mismatch map({pi},{pj}) ch{c}: {s_oracle} != {s_conv}"
            hand_check[f"map_y{pi}_x{pj}_ch{c}"] = s_conv
    # ---- self-verification (d): check the 08-spec analytic bwd derivation (the authoritative kernel formula, Task 2) ----
    mu1, mu2, A1, A2, B1, B2 = (t.detach() for t in mids)
    G = -1.0 / M                                              # dL/dS, L = 1 - mean(S)
    dS_dmu1 = 2.0 * (mu2 * A2 / (B1 * B2) - mu1 * smap_d / B1)
    dS_ds11 = -smap_d / B2                                    # ∂S/∂σ1²
    dS_ds12 = 2.0 * A1 / (B1 * B2)                            # ∂S/∂σ12
    D_mu = dS_dmu1 - 2.0 * mu1 * dS_ds11 - mu2 * dS_ds12      # raw-moment chain correction
    convT = lambda t: F.conv_transpose2d(t, window, groups=C) # the adjoint of valid corr = zero-padded full conv
    v_analytic = G * (convT(D_mu) + 2.0 * x.detach() * convT(dS_ds11) + y * convT(dS_ds12))
    assert torch.allclose(v_analytic, v_x, rtol=1e-10, atol=1e-14), \
        "the analytic bwd derivation (08 spec) != autograd"
    # ---- self-verification (e): central finite differences at 4 coordinates ----
    for (c, i, j) in ((0, 0, 0), (1, 12, 7), (2, 23, 23), (0, 5, 18)):
        h = 1e-6
        xp, xm = x.detach().clone(), x.detach().clone()
        xp[0, c, i, j] += h; xm[0, c, i, j] -= h
        lp = 1.0 - ssim_map_fn(xp, y)[0].mean()
        lm = 1.0 - ssim_map_fn(xm, y)[0].mean()
        fd = float((lp - lm) / (2 * h))
        gv = float(v_x[0, c, i, j])
        assert abs(fd - gv) < max(1e-4 * abs(gv), 1e-9), \
            f"FD mismatch ({c},{i},{j}): {fd} != {gv}"
    # ---- self-verification (f): non-degeneracy ----
    assert 0.0 < ssim_mean < 0.99, f"degenerate ssim: {ssim_mean}"
    assert float(smap_d.std()) > 0.005, "degenerate ssim map variance"
    assert bool((v_x != 0).all()), "a pixel the grad does not reach exists (valid receptive-field coverage violated)"

    # ---- save (layout follows the splatx image convention [I,H,W,C] — the kernel consumes render_colors directly) ----
    _save_f32(out, "img1", img1)
    _save_f32(out, "img2", img2)
    _save_f32(out, "ssim_map", smap_d.permute(0, 2, 3, 1).contiguous())
    save_sxt(str(out / "ssim.sxt"), np.array([ssim_mean], dtype=np.float32))
    _save_f32(out, "v_img1", v_x.permute(0, 2, 3, 1).contiguous())
    _write_meta(out, {
        "golden_source": "gsplat@77ab983:gsplat/losses.py:84-151 (the window and SSIM formulas) + :185 "
                         "(the fused_ssim padding=\"valid\" call convention) — an original pure-torch "
                         "f64 reproduction (the fused-ssim code was neither cloned nor touched)",
        "I": 1, "C": C, "height": H, "width": W,
        "window_size": WIN, "sigma": SIGMA, "C1": C1, "C2": C2,
        "padding": "valid", "map_height": H - WIN + 1, "map_width": W - WIN + 1,
        "window_taps_f32": [float(np.float32(v)) for v in w1d.tolist()],
        "seeds": {"input": 37},
        "lineage": {
            "img1": "the f32 crop [0, 4:28, 4:28, :] of raster_small/bg/render_colors.sxt "
                    "(element bytes preserved) — the render side of training",
            "img2": "synthesized GT: clamp(0.88·img1 + 0.06 + 0.06·sin(2πx/12)cos(2πy/16) "
                    "+ 0.03·randn(seed 37), 0, 1) — f32 arithmetic",
        },
        "gsplat_convention": {
            "loss": "lerp(l1, 1-ssim, λ) = (1-λ)·l1 + λ·(1-ssim), λ=ssim_lambda=0.2 "
                    "(simple_trainer.py:171,961)",
            "layout": "the gsplat call is NCHW (the trainer does [B,H,W,C]→permute(0,3,1,2)) — the "
                      "fixture is stored in the splatx image convention [I,H,W,C], and only the "
                      "golden computation transposes to NCHW",
            "grad": "only img1 (the render) is differentiated (the losses.py:184 guard) — v_img2 is not stored",
        },
        "hand_check": hand_check,
        "tensors": {"inputs": ["img1", "img2"], "outputs": ["ssim", "ssim_map"],
                    "grad_inputs": ["v_img1"]},
        "note": "The bwd upstream is fixed at the scalar 1 (loss = 1 - mean(ssim_map) — the "
                "authoritative training direction; dump_backward is not used and λ=0.2 is not "
                "applied — the Trainer scales it when combining). ssim_map covers the valid region "
                "only (map[p] = window start p, center p+5). Self-verification: ssim(a,a)=1 / "
                "symmetry is bit-identical / a Python scalar oracle at 9 points / the bwd "
                "derivation of spec 08 checked against autograd (rtol 1e-10) / finite differences "
                "at 4 coordinates / gradients reach every pixel.",
        "tolerances": {
            # Phase 3 Task 2 measured calibration (the D1/07 procedure — updated from the 08 spec §7
            # starting values, recorded in §9). The f64 oracle matches the golden to rel ≤ 5.1e-8
            # (formula exact) — the violations are pure f32 cancellation: the σ = E[x²]−μ²
            # cancellation × the 1/B2 (≤1/C2) amplification (exactly the 08 §7 self-estimate).
            # M4 measurements: map max rel 3.9e-4 → rtol 1e-3 (~2.6x margin, the 1e-4 starting
            # value violated); ssim scalar rel 1.33e-5 → rtol 5e-5 (~3.7x margin, the 1e-5
            # starting value violated). v_img1 passes at the starting values — unchanged.
            "ssim": {"rtol": 5e-5, "atol": 1e-6},
            "ssim_map": {"rtol": 1e-3, "atol": 1e-5},
            "v_img1": {"rtol": 1e-3, "atol": 1e-6, "rel_l2": 1e-3},
        },
    })
    print(f"wrote {out} (ssim={ssim_mean:.6f}, map "
          f"{H - WIN + 1}x{W - WIN + 1}, M={M})")


if __name__ == "__main__":
    names = sys.argv[1:] or list(FIXTURES)
    unknown = [n for n in names if n not in FIXTURES]
    if unknown:
        raise SystemExit(f"unknown fixture(s): {unknown}; available: {sorted(FIXTURES)}")
    math_mod, ti = _load_gsplat_modules()
    for n in names:
        FIXTURES[n](math_mod, ti)
