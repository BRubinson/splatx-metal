# Architecture

This document explains how splatx-metal is put together and, more
importantly, *why* the seams are where they are. The deepest sources of truth
are not this file: **contracts live in header comments**
(`Sources/SplatXCore/include/splatx/KernelBackend.h`, `Trainer.h`,
`Checkpoint.h`, `MemoryGovernor.h`, `ThermalGovernor.h`,
`Sources/SplatXTrainC/include/splatx_train_c.h`) and in the kernel design
documents ([docs/kernel-specs/00-index.md](docs/kernel-specs/00-index.md),
cross-kernel decisions D1–D11). When this file and a header disagree, the
header wins. Read the relevant header before changing behavior, and update
its comment when you do.

## The three layers and the C boundary

```text
SplatXCore    C++17 / metal-cpp — kernels, Trainer, Metal & CPU backends, IO
    │         public headers are C++  →  Swift's ClangImporter cannot import them
SplatXTrainC  pure C — splatx_train_c.h is the single contract surface
    │
SplatXMetal (Swift) · splatx CLI · apps/SplatXMetalTrainer
```

The C layer exists because of a language constraint, not taste: Swift cannot
import C++ headers, so everything Swift needs crosses one pure-C header. The
same pattern is replicated for the test runner (`splatx_tests_c.h`).

A consequence worth internalizing: **the training data flow completes
entirely inside C.** One blocking call in `TrainC.cpp` performs COLMAP
loading → image-name sorting → factor scaling → init or checkpoint restore →
backend creation → the training loop → export filtering → PLY/SPZ writing.
Swift only observes progress, cancellation, and statistics through callbacks.
Progress callbacks are invoked synchronously on the training thread, and the
blocking run must be hosted on a dedicated `Thread`, never on a Swift
concurrency executor.

## One training step

`Trainer::step` (`Sources/SplatXCore/Trainer.cpp`) branches on its first
line: `if (resident_ready()) return step_resident(...)`.

- **Resident path (default)** — the whole step is encoded GPU-side by
  `TrainStepExecutor`. The host keeps only scheduling (learning rate, SH
  degree), MCMC refinement, and evaluation. There are exactly **two** CPU
  sync points per step: the intersection-count readback and the loss
  readback.
- **Host round-trip path (`--no-resident`)** — the same assembly, explicitly
  chained on the host. It is retained **permanently as the parity oracle**
  for fixture verification, not as a performance option.

Only two deviations between the paths are tolerated: libm-vs-Metal
`precise::` differences in the `exp`/`sigmoid` activations, and atomic
backward accumulation order. Equivalence between the two paths is gated by
tests.

The 13 kernels in `TrainStep.metal` have their **formula and operation-order
canon in `Trainer.cpp`, not in the kernel source**. Changing only the kernel
forks it from the host path.

### Resident parameter "Home" state machine

When training runs resident, parameters live on the GPU. `Trainer.cpp` tracks
where the authoritative copy lives ("Home") with two flags
(`device_authoritative_` / `host_fresh_`; the comment block in `Trainer.cpp`
is the canon):

- **Reading** host-side → `sync_host_from_device()` (lazy download; Home
  stays on device).
- **Mutating** host-side → `sync_host_for_mutation()` is **mandatory**
  (download + transition Home to host, so the next resident step re-uploads).

Skipping the mutation sync means the next resident step consumes stale device
values and your mutation **silently evaporates**. Related: never use the full
parameter accessor for logging or monitoring — on the resident path it
triggers a full parameter download; use `splat_count()` when only the count
is needed.

## The two backends are not a GPU/CPU fallback pair

`CpuBackend` is a **forward-only reference oracle**. Its backward entry
points (`rasterize_bwd`, `sh_bwd`, `projection_fused_bwd`, `ssim_bwd`) are
deliberately unimplemented, so CPU training is impossible by design —
gradients are validated directly against autograd goldens, without a CPU
reference in between. Conversely, `MetalBackend::projection_ewa_fwd` throws
(a CPU-only Phase 0 remnant). Do not be misled by the pure-virtual
signatures: the two backends are complementary verification surfaces, not
substitutes.

## Dual kernel compilation (design decision D5)

Every `.metal` source must compile through **two** pipelines:

1. **Offline** (`tools/build_metallib.sh`): per-file `xcrun metal -c -I
   <SrcDir> -std=metal3.2` → `.air` → linked `splatx.metallib`. Separate
   translation units.
2. **Runtime** (`MetalBackend.cpp`, `assemble_library_source`): five shared
   headers are prepended once, each `.metal` file's local includes are
   stripped, and everything is concatenated in name order into a **single
   giant translation unit**.

Hence the D5 symbol rules: device helpers are `static inline` in headers
only; kernel names, struct names, and function-constant names/indices are
globally unique across the library. A symbol collision or a missing include
**breaks only one of the two paths** — which is why the test gate runs both
configurations, always.

Adding a shared header requires registering it in `kSharedHeaders[]` in
`MetalBackend.cpp` in dependency order; that array is the runtime path's only
include graph.

### metallib loading is three-tiered

The `MetalBackendDesc` contract (`MetalBackend.h`):

- (a) explicit `metallib_path` → **failure throws immediately, no fallback.**
  A path typo silently replaced by stale kernels would let tests pass against
  the wrong code; the loud failure is deliberate. Do not soften it.
- (b) auto-discovery of `splatx.metallib` next to the executable → absence is
  **normal** (development-tree scenario, no warning). The asymmetry with (a)
  is intended.
- (c) `kernel_src_dir` runtime source compilation — a development
  convenience, not a distribution contract. The XCFramework build
  deliberately overrides it with an invalid path so that reaching tier (c)
  in a deployed app fails loudly instead of compiling from a build-machine
  path. The XCFramework does not embed a metallib; consumers bundle one and
  inject it via `metallib_path`.

## Governors live outside the Trainer

`MemoryGovernor` and `ThermalGovernor` never appear in `Trainer` code. All
intervention happens at **step boundaries**, polled by the orchestration
layer (`TrainC.cpp`'s training loop). The library never subscribes to
platform APIs — thermal state is **injected** by the app layer
(`sx_thermal_set_state`); the macOS CLI has no injection source, so thermal
mode is observational there. This inversion is what makes the whole policy
unit-testable without heating a device (`SPLATX_THERMAL_INJECT` schedules
fake states on a timeline).

Two policy facts that look wrong but are measured decisions (rationale in the
governor headers): the thermal `serious` state is deliberately non-blocking
(pausing there doubled wall-clock in measurement), and thermal abort is
terminal — there is no recovery path, only graceful export + checkpoint.

## Kernel-authoring rules (the D-decisions)

The cross-kernel decisions D1–D11 are specified in
[docs/kernel-specs/00-index.md](docs/kernel-specs/00-index.md); the ones most
often violated by newcomers:

- **`KernelParams.h` uses 4-byte scalars only.** Host and device must agree
  on layout with zero padding; the `static_assert(sizeof)` battery is the
  only enforcement (MSL has none). 64-bit values are split lo/hi.
- **Every backward/accumulation buffer is zero-filled at encode time**
  (D8-1); Metal buffers do not start at zero. `v_means` is accumulated by
  both `sh_bwd` and the projection backward into the same buffer, so its
  zero-fill must happen **once per step**, not per kernel (D6).
- **Fast-math parity (D2)**: the build scripts pass no numerical flags;
  parity between the two compilation paths is achieved in **kernel source**
  (`metal::precise::` at parity points, `precise::divide` for tile-boundary
  divisions, no `pow(-1, k)` — negative bases are NaN under fast-math; use
  the sign trick). Do not add optimization flags to the build scripts.
- **No CUDA-style early returns in simd-reduction kernels** (D8-2) —
  diverging a `simd_sum` is undefined; restructure as an alive-flag with
  zero contribution.
- `pipeline()` throws if `threadExecutionWidth != 32` (D8-3). That is the
  guard for the simd-width-32 assumption baked into the kernels, not a bug.
- Buffer arguments specialized away by a `false` function constant
  **disappear from the signature** (D8-5) — flipping a gate requires fixing
  the host's `setBuffer` pairing.
- Culled slots (radii ≤ 0) hold **unwritten garbage** in
  means2d/depths/conics (D3): never read them, and never "clean them up"
  with a zero-fill — the reference behavior is reproduced deliberately
  (only `compensations` is zero-initialized).
- `rasterize_fwd` cannot be skipped even when `n_isects == 0` (empty-tile
  background must still be written); `rasterize_bwd` with `n_isects == 0` is
  all zeros.
- SSIM's scalar reduction happens on the host in f64 — no device atomic
  floats; window taps are compile-time constants.
- The Philox RNG's integer path and uniform conversion are **bit-identical
  between CPU and Metal by contract**; do not "optimize" the uniform
  conversion with uint arithmetic (wraps at `0xFFFFFFFF` → `log(0)` → NaN).

### Adding a kernel touches eight places, in order

① `FunctionConstants.h` (index inside the area band; names/indices globally
unique) → ② `KernelParams.h` (params struct + `static_assert`) → ③
`KernelBackend.h` (Inputs/Outputs + throwing default) — ② and ③ are a pair;
never change one alone → ④ the `.metal` file (new files are auto-collected)
→ ⑤ `CpuKernels.h/.cpp` (**forward only** — backward has no CPU reference by
policy) → ⑥ `CpuBackend.cpp` (one-line delegation) → ⑦ `MetalBackend.cpp`
(pipeline + encoding; `encode_zero_fill` for gradient accumulators) → ⑧
fixture generator registry + a `SplatXTests/<name>Tests.cpp` + `meta.json`
tolerances.

## Verification model

Three stages: gsplat `77ab983` (PyTorch/CUDA) generates golden tensors with
deterministic seeds → a float32 CPU reference is checked against the goldens
→ Metal kernels are cross-checked against both. Fixtures chain: the
projection forward golden feeds the intersection fixture byte-identically,
which feeds rasterization (`lineage` in each `meta.json`). Backward tests
consume golden forward outputs, not live kernel outputs, so forward error
cannot mask backward error.

**Determinism is a per-kernel contract**, and it decides what a test may
assert:

- Bit-deterministic (memcmp/exact): projection fwd, SH fwd, intersection
  count/emit, offset encode, prefix scan, radix sort, raster fwd, SSIM
  fwd/bwd, RNG, Adam/relocation/perturbation, the resident TrainStep
  kernels. Integer outputs always compare exactly.
- Atomic accumulation order is non-deterministic → dual-run self-consistency
  with `rel_l2` only: raster bwd, SH bwd (B > 1), projection bwd (C > 1 /
  `v_viewmats`).

Tolerances live only in `meta.json` (a missing key fails; empty comparisons
fail). Projection forward comparisons are valid only under the valid-slot
mask — the reference implementation never writes culled slots.

Swift tests cover wiring and contracts only; all numerical correctness is
the C++ suite's job. There is no skip mechanism in the C++ harness: without
a Metal device, GPU tests fail rather than skip — that is intentional.

## The macOS app

`apps/SplatXMetalTrainer` consumes the Swift layer (`SplatXMetal`) plus a
vendored, modified MetalSplatter fork (`ThirdParty/MetalSplatter`; the
deviation list lives in its `VERSION.txt`). The Xcode project is generated
by xcodegen and not checked in — after adding or removing source files, run
`xcodegen generate` or the target will silently not include them. Rendering
in the viewer and the Cinema video export go through the same C render
session API (`sx_render_session_*`) as the CLI renderer; training and export
are mutually exclusive by design (one training run per process is a
structural assumption, enforced by a static gate on the Swift side).
