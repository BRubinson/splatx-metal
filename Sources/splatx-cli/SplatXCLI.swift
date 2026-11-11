import ArgumentParser
import Foundation
import SplatXMetal
import SplatXTrainC

@main
struct SplatX: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "splatx",
        abstract: "splatx-metal: a Gaussian Splatting training framework for Apple Silicon",
        version: "0.1.0",
        subcommands: [Train.self, Convert.self, Render.self, Probe.self,
                      Info.self]
    )
}

struct Train: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Train 3DGS from COLMAP data (Phase 1c: correctness E2E — host round-trip chaining)")

    @Option(name: .long, help: "COLMAP data directory (sparse/0 + images[_N])")
    var data: String

    @Option(name: .long,
            help: "Image downsample factor (1|2|4|8 — selects the images_N folder; if it is absent, the images/ originals are downscaled by factor at load time)")
    var factor: Int = 1

    @Option(name: .long, help: "Number of training iterations")
    var iters: Int = 300

    @Option(name: .customLong("max-points"), help: "Upper bound on the initial Gaussian count (init_splats cap)")
    var maxPoints: Int = 50000

    @Option(name: .customLong("sh-degree"), help: "SH degree (0..3)")
    var shDegree: Int = 3

    @Option(name: .long,
            help: "Output path — a .spz extension writes SPZ v3 (Niantic spz, quantized compression); anything else INRIA PLY")
    var out: String

    @Option(name: .customLong("save-preview"), help: "Post-training render preview PNG path (optional)")
    var savePreview: String?

    @Option(name: .customLong("eval-image"), help: "Preview render camera index")
    var evalImage: Int = 0

    @Option(name: .customLong("log-every"), help: "Progress log interval (iters)")
    var logEvery: Int = 10

    @Option(name: .long,
            help: "Path to an offline-compiled .metallib (output of tools/build_metallib.sh — if omitted, falls back to splatx.metallib next to the executable, then to source compilation)")
    var metallib: String?

    @Flag(name: .long,
          help: "Enable MCMC densification and auto-apply the gsplat mcmc preset (init_opa 0.5, init_scale 0.1, opacity/scale reg 0.01)")
    var mcmc: Bool = false

    @Option(name: .customLong("cap-max"),
            help: "Upper bound on the Gaussian count (--mcmc only — fixed arena size, at least the initial N. Ignored when --auto-cap is given)")
    var capMax: Int = 1_000_000

    @Flag(name: .customLong("auto-cap"),
          help: "Auto-size cap_max (--mcmc only — design §7.2): (memory budget × 80% − fixed components (GT cache, images, isect)) / resident bytes-per-splat, clamped to [100k, 4M]. Ignores --cap-max and prints the sizing rationale in the auto-cap log")
    var autoCap: Bool = false

    @Option(name: .customLong("memory-budget-mb"),
            help: "Memory budget for --auto-cap (MiB). If omitted, the macOS fallback applies — a conservative 50% of sysctl hw.memsize (rationale in MemoryGovernor.cpp)")
    var memoryBudgetMb: Int?

    @Option(name: .customLong("test-split"),
            help: "If K > 0, split out cameras whose name-sorted index satisfies i % K == 0 as the test set (MipNeRF360 convention 8, 0=off)")
    var testSplit: Int = 0

    @Option(name: .customLong("eval-every"),
            help: "Log the mean PSNR over all test views every K iterations (requires --test-split, 0=once at the end only)")
    var evalEvery: Int = 0

    @Option(name: .customLong("ssim-weight"),
            help: "SSIM loss weight λ ∈ [0,1] — loss = (1-λ)·L1 + λ·(1-SSIM) (gsplat ssim_lambda 0.2, 0=pure L1)")
    var ssimWeight: Float = 0.2

    @Option(name: .customLong("depth-weight"),
            help: "Depth supervision weight w ≥ 0 — loss += w·(confidence-masked inverse-depth L1 of the expected depth) on views with a depths/<stem>.png sidecar (16-bit mm; optional depths/<stem>.conf.png ARConfidenceLevel). 0=off")
    var depthWeight: Float = 0

    @Flag(name: .long,
          help: "Anti-aliasing (Mip-Splatting 2D compensation) — trains with opacity multiplied by the projection compensation factor (reproduces gsplat rasterize_mode=antialiased; default off=classic — the published-benchmark condition. Standard PLY has no AA flag, so generic viewers render without compensation; .spz output records the antialiased flag in its header — see the README caveat)")
    var antialiased: Bool = false

    @Flag(name: .customLong("no-resident"),
          help: "Disable the device-resident train_step — train with the legacy host round-trip chaining (a verification-surface escape hatch; the resident step is on by default)")
    var noResident: Bool = false

    @Option(name: .long,
            help: "Resume training from a checkpoint — validates the config fingerprint (rejects on sh_degree/factor/test-split/iters/mcmc/scene_scale/ssim/AA mismatch; sh_degree is locked — design §8.1). Skips init_splats and resumes from the saved iter")
    var resume: String?

    @Option(name: .customLong("extend-iters"),
            help: "Extended resume (requires --resume): train K more iters from the checkpoint iter. --iters is ignored and the ratio-schedule denominator (total_iters) stays fixed at the checkpoint value — pure refinement outside the refine window (semantics rationale in splatx_train_c.h)")
    var extendIters: Int = 0

    @Option(name: .customLong("save-checkpoint"),
            help: "Checkpoint save path (fp32 lossless, atomic temp+rename) — once at the end of training plus every --checkpoint-every K")
    var saveCheckpoint: String?

    @Option(name: .customLong("checkpoint-every"),
            help: "If K > 0, save a checkpoint every K iterations (requires --save-checkpoint, 0=once at the end only)")
    var checkpointEvery: Int = 0

    @Option(name: .customLong("thermal-mode"),
            help: "Thermal response mode (design §7.4): performance=no intervention (observation logs only) | balanced=pause at a step boundary on critical, resume with hysteresis at serious or below; on sustained critical, partial export then graceful abort (exit 2) — serious itself is no-intervention (the normal operating state under sustained load) | lowheat=balanced + idling between steps + pause threshold lowered to serious (resume at fair). The macOS CLI has no thermal injection source, so observation is a no-op — state is injected by the app layer (sx_thermal_set_state)")
    var thermalMode: String = "balanced"

    func run() throws {
        let thermalModes: [String: Int32] = ["performance": 0, "balanced": 1, "lowheat": 2]
        guard let thermal = thermalModes[thermalMode] else {
            throw ValidationError("--thermal-mode must be one of performance|balanced|lowheat")
        }
        var err = [CChar](repeating: 0, count: 4096)
        // The struct-argument form (sx_train_run_args — Task 5): the CLI also uses an extension
        // field (--extend-iters), so it calls the struct form instead of the wrapper. The C string
        // lifetimes are guaranteed by the nested withCString scopes.
        let rc = data.withCString { dataPtr in
            out.withCString { outPtr in
                withOptionalCString(savePreview) { previewPtr in
                    withOptionalCString(metallib) { metallibPtr in
                        withOptionalCString(resume) { resumePtr in
                            withOptionalCString(saveCheckpoint) { ckPtr -> Int32 in
                                var a = SxTrainArgs()
                                a.data_dir = dataPtr
                                a.factor = Int32(factor)
                                a.iters = Int32(iters)
                                a.max_points = Int64(maxPoints)
                                a.sh_degree = Int32(shDegree)
                                a.out_path = outPtr
                                a.preview_png = previewPtr
                                a.eval_image = Int32(evalImage)
                                a.metallib_path = metallibPtr
                                a.log_every = Int32(logEvery)
                                a.mcmc = mcmc ? 1 : 0
                                a.cap_max = Int64(capMax)
                                a.test_split = Int32(testSplit)
                                a.eval_every = Int32(evalEvery)
                                a.ssim_weight = ssimWeight
                                a.depth_weight = depthWeight
                                a.antialiased = antialiased ? 1 : 0
                                a.resident = noResident ? 0 : 1
                                a.resume_checkpoint = resumePtr
                                a.save_checkpoint_path = ckPtr
                                a.checkpoint_every = Int32(checkpointEvery)
                                a.auto_cap = autoCap ? 1 : 0
                                a.memory_budget_bytes = Int64(memoryBudgetMb ?? 0) * 1_048_576
                                a.thermal_mode = thermal
                                a.resume_extend_iters = Int32(extendIters)
                                // CLI checkpoints are fp32-only (desktop resume is the
                                // reference path — Task 2 contract)
                                a.checkpoint_fp16 = 0
                                return sx_train_run_args(&a, &err, Int32(err.count))
                            }
                        }
                    }
                }
            }
        }
        if rc == 2 { // thermal graceful abort — not an error (partial export/checkpoint written)
            print("thermal abort: sustained critical — partial results saved to --out (graceful exit, exit 2)")
            throw ExitCode(2)
        }
        if rc == 3 { // external stop, graceful halt (no CLI injection source — surface symmetry)
            print("stop: external stop requested — partial results saved to --out (graceful exit, exit 3)")
            throw ExitCode(3)
        }
        if rc != 0 {
            let bytes = err.prefix(while: { $0 != 0 }).map { UInt8(bitPattern: $0) }
            throw ValidationError("train failed: \(String(decoding: bytes, as: UTF8.self))")
        }
    }

    private func withOptionalCString<R>(_ s: String?,
                                        _ body: (UnsafePointer<CChar>?) -> R) -> R {
        guard let s else { return body(nil) }
        return s.withCString { body($0) }
    }
}

struct Convert: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Convert between PLY and SPZ formats (extension auto-detected — .spz is SPZ v3, anything else INRIA PLY)")

    @Option(name: .customLong("in"),
            help: "Input path (.ply | .spz — extension case-insensitive, same convention as train --out)")
    var input: String

    @Option(name: .long,
            help: "Output path (.spz | .ply). SPZ output has the export filter built in (drops NaN/Inf and low-opacity rows — the quantized packing contract); PLY output is not re-filtered")
    var out: String

    func run() throws {
        var err = [CChar](repeating: 0, count: 4096)
        let rc = sx_convert_run(input, out, &err, Int32(err.count))
        if rc != 0 {
            let bytes = err.prefix(while: { $0 != 0 }).map { UInt8(bitPattern: $0) }
            throw ValidationError("convert failed: \(String(decoding: bytes, as: UTF8.self))")
        }
    }
}

struct Render: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Render a trained PLY from a COLMAP camera → PNG (no training — for visual/PSNR comparison of iOS↔macOS outputs)")

    @Option(name: .long, help: "Input PLY path (INRIA 3DGS PLY)")
    var ply: String

    @Option(name: .long, help: "COLMAP data directory (sparse/0 + images[_N]) — camera poses/intrinsics")
    var data: String

    @Option(name: .long, help: "Image downsample factor (1|2|4|8) — determines the render resolution (the PLY is resolution-independent; scales the camera dims)")
    var factor: Int = 1

    @Option(name: .customLong("cam"), help: "Render camera index (name-sorted order, clamped when out of range)")
    var cam: Int = 0

    @Option(name: .long, help: "Output PNG path")
    var out: String

    // Unlike train, render does not pass kernel_src_dir, so it has no source-compilation fallback
    // (the authoritative contract is the sx_render_ply section of splatx_train_c.h — it fails when
    // there is no metallib next to the executable).
    @Option(name: .long, help: "Path to an offline .metallib (if omitted, auto-discovers splatx.metallib next to the executable — fails when absent, no source-compilation fallback)")
    var metallib: String?

    func run() throws {
        var err = [CChar](repeating: 0, count: 4096)
        let rc = ply.withCString { plyPtr in
            data.withCString { dataPtr in
                out.withCString { outPtr -> Int32 in
                    withOptionalCString(metallib) { libPtr in
                        sx_render_ply(plyPtr, dataPtr, Int32(factor), Int32(cam),
                                      libPtr, outPtr, &err, Int32(err.count))
                    }
                }
            }
        }
        if rc != 0 {
            let bytes = err.prefix(while: { $0 != 0 }).map { UInt8(bitPattern: $0) }
            throw ValidationError("render failed: \(String(decoding: bytes, as: UTF8.self))")
        }
    }

    private func withOptionalCString<R>(_ s: String?,
                                        _ body: (UnsafePointer<CChar>?) -> R) -> R {
        guard let s else { return body(nil) }
        return s.withCString { body($0) }
    }
}

struct Probe: ParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Run the device GPU capability/sustained-load probes")

    @Option(name: .long, help: "Sustained-load measurement duration (seconds)")
    var seconds: Double = 10.0

    @Option(name: .long, help: "Report JSON output path")
    var json: String?

    func run() throws {
        let ctx = try MetalContext(device: nil)
        print("probe start: \(ctx.device.name), sustained load \(seconds)s...")
        let report = try runProbeSuite(ctx, sustainedSeconds: seconds)

        print("  atomic float           : \(report.atomicFloatWorks ? "supported" : "unsupported (fatal — CAS fallback required)")")
        print("  simdgroup width        : \(report.simdWidth)")
        print("  threadgroup memory     : \(report.maxThreadgroupMemoryBytes / 1024)KB")
        print("  recommended working set: \(report.recommendedMaxWorkingSetBytes / (1024 * 1024))MB")
        print("  sustained-load ratio   : \(String(format: "%.2f", report.sustainedRatio)) (1.0=clock held)")

        if let path = json {
            let enc = JSONEncoder()
            enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            try enc.encode(report).write(to: URL(fileURLWithPath: path))
            print("report saved: \(path)")
        }
    }
}

struct Info: ParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Print build/device info")

    func run() throws {
        let ctx = try MetalContext(device: nil)
        print("splatx 0.1.0")
        print("device: \(ctx.device.name)")
        print("OS    : \(ProcessInfo.processInfo.operatingSystemVersionString)")
    }
}
