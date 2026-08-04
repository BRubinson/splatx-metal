# Benchmarks

All headline numbers claimed in the README, with the conditions they were
measured under and the exact commands to reproduce them. Numbers are quoted
verbatim from the project's internal measurement records (which are
append-only; results are never retro-edited) — dates below are the
measurement dates. Expect variance across chips, memory sizes, thermal
conditions, and macOS versions.

## Environment

Unless stated otherwise:

| Item | Value |
|---|---|
| Machine | MacBook Air, Apple M4, 24 GB unified memory (fanless) |
| OS | macOS 26.2, Xcode 26 toolchain |
| Build | `swift build -c release` + offline `tools/build_metallib.sh`, metallib injected via `--metallib` |
| Power | AC power, `caffeinate -is`, no competing GPU load |
| Dataset | Mip-NeRF 360 (`garden`, `bicycle`), `images_4` (factor 4, 1297×840 / 1237×822) |
| Loss | `0.8·L1 + 0.2·(1−SSIM)`, anti-aliasing off, `sh-degree 3`, MCMC, device-resident path |

gsplat reference numbers are taken from gsplat's published evaluation at the
pinned commit `77ab983` (`docs/source/tests/eval.rst`; measured by the gsplat
authors on TITAN RTX / A100 with unconstrained Gaussian growth).

## Quality — 7K iterations (measured 2026-07-12)

| Scene | cap | test PSNR @ 7K | gsplat-7k published | Δ | Wall clock | Peak RSS |
|---|---|---|---|---|---|---|
| garden | 2M | **26.015 dB** (24 views) | 26.30 dB (N = 4.48M) | −0.285 | 14.9 min | 5.16 GB |
| garden | 1M | 25.538 dB (24 views) | 26.30 dB (N = 4.48M) | −0.762 | 8.5 min | 3.86 GB |
| bicycle | 1M | **23.535 dB** (25 views) | 23.71 dB (N = 3.62M) | −0.175 | 8.3 min | 3.69 GB |

Note the conditions: gsplat's published numbers use unconstrained growth
(4.48M / 3.62M Gaussians); splatx-metal runs a fixed MCMC arena at 45% or
less of the reference splat count. The project's acceptance gate is −0.5 dB
against the published numbers.

Reproduce (garden, cap 2M; for bicycle swap `--data` and use
`--cap-max 1000000`):

```sh
tools/build_metallib.sh
swift run -c release splatx train \
  --data ~/data/360_v2/garden --factor 4 \
  --mcmc --cap-max 2000000 --iters 7000 \
  --sh-degree 3 --max-points 100000 \
  --test-split 8 --eval-every 1000 --log-every 50 \
  --metallib build/splatx.metallib --out /tmp/garden_7k.ply
```

## Quality — 30K iterations (measured 2026-07-12)

At 30K the ratio-based schedule expands to gsplat's default MCMC schedule
(refine start 500 / stop 25000 / every 100).

| Scene | cap | test PSNR @ 30K | Wall clock | Steady-state step |
|---|---|---|---|---|
| garden | 1M | 26.323 dB | 39.5 min | 101.8 ms |
| garden | 2M | **26.780 dB** | 65.7 min | 231.0 ms |
| bicycle | 1M | 23.752 dB | 31.9 min | 91.3 ms |

For scale (conditions do not match — unconstrained growth, different GPUs):
gsplat-30k default reports garden 27.32 dB at N = 5.84M and bicycle 25.22 dB
at N = 6.26M; gsplat's MCMC 30K seven-scene averages are 29.18 dB (cap 1M)
and 29.53 dB (cap 2M) on A100, published as averages only, so per-scene
comparison is not possible.

Reproduce: same command as above with `--iters 30000`.

## Training speed (measured 2026-07-12 … 2026-07-21)

| Metric | Value |
|---|---|
| Steady-state step, garden 7K, N = 1M | **76.8 ms/iter** |
| Steady-state step, garden 7K, N = 2M | 144.3 ms/iter |
| Device-resident vs host round-trip (N = 1M, matched conditions) | 335 → 91 ms/iter (**3.7×**) |
| End-to-end effect on garden 7K (host-era vs resident) | 42.5 min → 8.5 min (5.0×) |

The resident-vs-host comparison is also recorded in
[kernel-specs/00-index.md](kernel-specs/00-index.md) (Phase 3 achievement
notes). The host path remains available for verification:

```sh
swift run -c release splatx train ... --no-resident   # parity oracle path
```

## iPhone (measured 2026-07-12, iPhone 15 Pro / A17 Pro)

- Full C++ core suite on device: **109/109 pass** at the then-current
  baseline, with no tolerance re-calibration against the M4 gates.
- On-device smoke training to completion (garden factor 8, 648×420, N ≤
  500k): ~99 ms/iter, final 24.048 dB (train-view PSNR).

## SPZ round-trip (measured 2026-07-12)

Input: garden 7K cap-1M export (982,532 splats, sh-degree 3).

| Metric | Value |
|---|---|
| PLY → SPZ size | 243.7 MB → **21.6 MB** (11.28×) |
| Splat count after round-trip | 982,532 → 982,532 (lossless filter pass) |
| Render-to-render PSNR (original vs round-tripped, same camera) | 34.17 dB (cam 0), 35.40 dB (cam 92) |

The ~34 dB quantization floor sits ~9 dB above the scene reconstruction
quality itself — visually lossless for distribution. Note that SPZ→SPZ
re-conversion re-runs the export filter and can drop near-threshold rows;
PLY output does not re-filter.

```sh
swift run splatx convert --in /tmp/garden_7k.ply --out /tmp/garden_7k.spz
swift run splatx convert --in /tmp/garden_7k.spz --out /tmp/garden_7k_rt.ply
```

## Cinema video export (measured 2026-07-26)

Final-quality render sessions (`sx_render_session_*`, the same renderer used
for training), garden at 980k splats, offline metallib:

| Resolution | Frame time |
|---|---|
| 1080p | 69 ms/frame |
| 4K | 111 ms/frame |

## Viewer depth sort (measured 2026-07-21, synthetic)

The vendored MetalSplatter fork replaces the upstream depth sort with an
exact GPU radix sort: 34.4 ms → **2.0 ms** per sort at 0.5M splats
(synthetic benchmark; see `ThirdParty/MetalSplatter/VERSION.txt` for the
full deviation list).
