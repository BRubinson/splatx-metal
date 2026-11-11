// SplatXTrainer — the Swift training API that maps 1:1 onto the iOS capture app's TrainerBackend
// protocol (Phase 6 round 2 Task 5). The training body goes through the pure C surface
// (SplatXTrainC sx_train_run_args) — the SplatXCore public headers are C++, so Swift cannot
// import them directly (the existing layering is preserved).
//
// TrainerBackend correspondence (the iOS capture app integration contract — the table is below):
//   train(_:config:progress:)  ↔ train(dataDirectory:options:progress:)
//   resume(_:additionalIterations:progress:) ↔ resume(checkpoint:dataDirectory:
//                                               additionalIterations:options:progress:)
//   TrainingProgress ↔ Progress (iter/loss/psnr/N/step ms/thermal/memory/coolingDown)
//   TrainingResult/TrainingMetrics ↔ RunResult
//   TrainingCheckpoint.stateURL ↔ the SXCK checkpoint file (query the header via checkpointInfo(at:))
//
// Cancellation semantics (decided by the brief review): on Task cancellation the run stops at a
// safe point (a step boundary), saves the checkpoint (when checkpointURL is set) plus a partial
// export, and then **returns a result with outcome == .cancelled** — it does not throw
// CancellationError. Rationale: throwing would lose the TrainingResult (the partial output and
// the resume checkpoint), and a cancelled run is precisely the case that most needs material to
// resume from. The iOS capture app's convention (TrainingRunModel catches CancellationError and
// treats it as .cancelled — the msplat precedent) is restored in the NativeBackend adapter with
// the single line `if result.outcome == .cancelled { throw CancellationError() }` (snippet in the
// integration document).
//
// Thermal observation: this layer subscribes to ProcessInfo.thermalState and injects it through
// sx_thermal_set_state (the library does not subscribe to platform APIs — ThermalGovernor.h. The
// observer is the app's responsibility, and SplatXTrainer is the authoritative implementation of
// that app layer). It injects the current state once at the start of a run and then observes
// transitions. Tests can substitute direct injection through sx_thermal_set_state.
import Foundation
import SplatXTrainC
import Synchronization

public enum SplatXTrainerError: Error, Sendable {
    /// Violation of the premise of exactly one concurrent training run (the process-global
    /// contract of the C surface — splatx_train_c.h).
    case anotherRunActive
    /// sx_train_run_args returned 1 — the err_buf message.
    case trainFailed(String)
    /// pauseAndDrain failed to save (the hold has been released).
    case pauseFailed(String)
    /// pauseAndDrain: no active training loop (no run / inside the load-and-init stretch / while
    /// finishing) — it means there is no training state to save yet or any more, so this is
    /// **safe to ignore**.
    case noActiveTrainingLoop(String)
    /// pauseAndDrain: the safe point was not reached within parkTimeout (a long step or an eval is
    /// in progress) — the loop is **still running** while the checkpoint was not saved (and the
    /// hold has been released — a §7.5 violation state in which background GPU submission can
    /// continue). Never ignore this — retry it or report it.
    case drainTimeout(String)
    /// Invalid argument (additionalIterations < 1, and so on).
    case invalidArgument(String)
    /// Failed to query the checkpoint file (magic, version, or truncation).
    case checkpointUnreadable(String)
}

public final class SplatXTrainer: Sendable {

    // MARK: - Types

    /// Thermal response mode — C thermal_mode 0|1|2 (the policy contract of record is
    /// ThermalGovernor.h).
    public enum ThermalMode: Int32, Sendable {
        case performance = 0 // no intervention (observation logs only)
        case balanced = 1    // pause at critical / hysteresis resume at serious or below /
                             // abort when critical is sustained (serious = no intervention)
        case lowheat = 2     // balanced + idle between steps + pause threshold lowered to
                             // serious (resume at fair)
    }

    /// Training options — 1:1 with the sx_train_run_args arguments (the contract of record for the
    /// field meanings is splatx_train_c.h).
    /// For the field correspondence with TrainingConfig see the per-field comments (the iOS
    /// capture app integration contract).
    public struct Options: Sendable {
        /// Total iterations (the ratio-based schedule denominator — TrainingConfig.iterations).
        public var iterations: Int = 7_000
        /// Image downsample factor 1|2|4|8 (the images_N folder convention —
        /// TrainingConfig.imageDownscale).
        public var factor: Int = 4
        /// SH degree 0..3 (TrainingConfig.shDegree).
        public var shDegree: Int = 3
        /// The seed subsample cap for init_splats (TrainingConfig.maxSeedPoints).
        public var maxPoints: Int = 120_000
        /// MCMC densification (automatically applies the gsplat mcmc preset).
        public var mcmc: Bool = true
        /// Upper bound on the Gaussian count — the fixed arena (TrainingConfig.targetSplatCount).
        /// Ignored when autoCap is set.
        public var capMax: Int = 500_000
        /// Per-device automatic cap_max estimation (MemoryGovernor §7.2 — capMax is ignored).
        public var autoCap: Bool = false
        /// The autoCap budget (bytes). On iOS, injecting os_proc_available_memory() is
        /// first-class — with <= 0 it falls back to the macOS sysctl (on iOS it MUST NOT be left
        /// uninjected — splatx_train_c.h).
        public var memoryBudgetBytes: Int64 = 0
        /// With K > 0, splits off the views whose name-sorted index satisfies i % K == 0 as test
        /// (evalHoldout ↔ 8/0).
        public var testSplit: Int = 0
        /// With K > 0, the mean PSNR over all test views every K iters (requires testSplit).
        public var evalEvery: Int = 0
        /// SSIM loss weight λ ∈ [0,1] (gsplat default 0.2).
        public var ssimWeight: Float = 0.2
        /// Weight of the confidence-masked inverse-depth L1 on views with a depth sidecar
        /// (depths/<stem>.png beside images/ — splatx_train_c.h SxTrainArgs.depth_weight).
        /// 0 = off.
        public var depthWeight: Float = 0
        /// Anti-aliasing (the Mip-Splatting compensation — default off = gsplat classic).
        public var antialiased: Bool = false
        /// device-resident train_step (default on — the reference path for performance).
        public var resident: Bool = true
        /// Thermal response mode (the TrainingConfig.thermalCeiling mapping is in the integration
        /// document).
        public var thermalMode: ThermalMode = .balanced
        /// The stdout progress log period (separate from the progress callback — the callback
        /// fires every step).
        public var logEvery: Int = 50
        /// Output path — a .spz extension means SPZ v3, anything else INRIA PLY.
        public var outputURL: URL
        /// Preview PNG after training (optional).
        public var previewURL: URL? = nil
        /// Checkpoint save path (once at completion/abort/cancellation, plus every checkpointEvery
        /// K iters).
        public var checkpointURL: URL? = nil
        /// With K > 0, saves every K iters (requires checkpointURL).
        public var checkpointEvery: Int = 0
        /// Checkpoint precision — true means fp16 half precision (recommended on iOS: half the
        /// size and half the time, Checkpoint.h).
        public var checkpointFP16: Bool = true
        /// The live PLY export path (app round 2 Task 1 — a live preview during training).
        /// nil = off; enabled **only when both this and livePlyEvery are valid**. At a step
        /// boundary it performs an atomic tmp+rename update using the same filter as the final
        /// export; a failure is non-fatal (it warns and training continues); the format is fixed
        /// to PLY — the authoritative contract is splatx_train_c.h SxTrainArgs.live_ply_path.
        public var livePlyURL: URL? = nil
        /// With K > 0, refreshes the current model at livePlyURL every K iters (<= 0 = off). On
        /// the resident path every export downloads all parameters — beware short periods (the
        /// app recommends 250).
        public var livePlyEvery: Int = 0
        /// With K > 0, delivers an in-memory snapshot to snapshotHandler every K iters
        /// (<= 0 = off, app round 3 Task 1). Same phase as live PLY (a step boundary) — on the
        /// resident path every snapshot downloads all parameters (when it coincides with livePly
        /// on the same tick, the two share a single download and filter pass).
        /// The app recommends 50.
        public var snapshotEvery: Int = 0
        /// The in-memory snapshot handler (the input for live rendering without going through a
        /// file). nil = off; enabled **only when both this and snapshotEvery are valid**.
        /// **Called synchronously on the training-loop thread** — a slow handler delays training
        /// by exactly that much, and inside it you MUST NOT re-enter this trainer surface
        /// (train/resume/pause and the rest). The buffers of SplatSnapshot are **invalid once the
        /// handler returns** — to keep them, copy inside the handler. The authoritative contract
        /// is splatx_train_c.h SxSplatSnapshot.
        public var snapshotHandler: (@Sendable (SplatSnapshot) -> Void)? = nil
        /// An explicit path to an offline-compiled .metallib (XCFramework consumers MUST inject
        /// it — a load failure is an immediate error, with no automatic fallback. With nil it
        /// searches next to the executable, then compiles the kernel sources).
        public var metallibURL: URL? = nil
        /// The kernel source fallback directory (nil = the build-time default — for a SwiftPM
        /// build, the package path).
        public var kernelSourceDirectory: URL? = nil

        public init(outputURL: URL) { self.outputURL = outputURL }
    }

    /// A progress tick — the input for TrainingProgress 1:1. The C callback (SxTrainProgress)
    /// joined with this layer's memory and cooling observations (one tick per step boundary, plus
    /// a synthesized coolingDown tick every 0.5s during a thermal pause).
    public struct Progress: Sendable {
        public let iteration: Int          // completed step count (1-based — TrainingProgress.iteration)
        public let runUntil: Int           // the iteration this run ends at (reflects extended resume)
        public let totalIterations: Int    // the ratio-based schedule denominator
        public let loss: Double
        public let psnr: Double?           // only when this tick performed an eval
        public let splatCount: Int
        public let stepMilliseconds: Double
        public let nIsects: Int
        public let elapsedSeconds: Double
        public let thermalState: Int       // 0 nominal .. 3 critical (observed by the governor)
        public let coolingDown: Bool       // synthesized tick while thermal-paused (TrainingProgress.coolingDown)
        public let memoryFootprintMB: Int  // task_vm_info.phys_footprint
        public let peakMemoryFootprintMB: Int
        public let availableMemoryMB: Int  // os_proc_available_memory (iOS — 0 on macOS)
        public let depthRMSE: Double?      // held-out expected-depth RMSE (m) on an eval tick with sidecars
    }

    /// An in-memory snapshot (app round 3 Task 1) — a pointer view of the current model during
    /// training, in the **raw domain**. Identical to what PLY records: scales in log-space,
    /// opacities in logit-space, quats w-first and unnormalized, and sh0/shN as raw SH
    /// coefficients (activation is the consumer's responsibility). The rows are the ones that pass
    /// the export filter (NaN/Inf removal plus sigmoid(opacity) > 0.005 — it shares the single
    /// source of truth for that filter with the final export and live PLY).
    /// **Every buffer is invalid once snapshotHandler returns** — to keep one, copy it inside the
    /// handler (the C contract of record: the pointer lifetimes of splatx_train_c.h
    /// SxSplatSnapshot).
    public struct SplatSnapshot {
        public let iter: Int                             // completed iteration (1-based)
        public let shDegree: Int                         // 0..3 — K = (shDegree+1)^2 - 1
        public let count: Int                            // row count N (passed the filter)
        public let means: UnsafeBufferPointer<Float>     // [N,3]
        public let quats: UnsafeBufferPointer<Float>     // [N,4] w-first, unnormalized
        public let scales: UnsafeBufferPointer<Float>    // [N,3] log-space
        public let opacities: UnsafeBufferPointer<Float> // [N] logit-space
        public let sh0: UnsafeBufferPointer<Float>       // [N,3]
        public let shN: UnsafeBufferPointer<Float>?      // [N,K,3] coefficient-first, nil at degree 0

        /// memberwise — production construction is owned by SnapshotRelay; this is public for
        /// consumer-side tests (so the app's pure-logic streamer tests can build a synthetic view).
        public init(iter: Int, shDegree: Int, count: Int,
                    means: UnsafeBufferPointer<Float>, quats: UnsafeBufferPointer<Float>,
                    scales: UnsafeBufferPointer<Float>,
                    opacities: UnsafeBufferPointer<Float>,
                    sh0: UnsafeBufferPointer<Float>, shN: UnsafeBufferPointer<Float>?) {
            self.iter = iter
            self.shDegree = shDegree
            self.count = count
            self.means = means
            self.quats = quats
            self.scales = scales
            self.opacities = opacities
            self.sh0 = sh0
            self.shN = shN
        }
    }

    /// The final result — the input for TrainingResult/TrainingMetrics.
    public struct RunResult: Sendable {
        public enum Outcome: String, Sendable {
            case completed       // reached run_until (a normal run to completion)
            case thermalAborted  // graceful abort on sustained critical (partial export/checkpoint kept)
            case cancelled       // cooperative stop (Task cancellation/requestStop — partial results kept)
        }
        public let outcome: Outcome
        public let outputURL: URL          // always written, a partial export included
        public let checkpointURL: URL?     // options.checkpointURL (when set, the final state is saved)
        public let iterationsCompleted: Int
        public let runUntil: Int
        public let totalIterations: Int    // the schedule denominator (differs from runUntil on extended resume)
        public let finalPSNR: Double?      // the final eval (when testSplit > 0 && the run completed)
        public let finalDepthRMSE: Double? // held-out expected-depth RMSE (m) — needs test views with sidecars
        public let trainSeconds: Double    // wall clock of the training loop
        public let wallClockSeconds: Double
        public let splatCount: Int         // N rows that passed the export filter
        public let fileSizeBytes: Int
        public let peakMemoryFootprintMB: Int
        public let minAvailableMemoryMB: Int // meaningful only on iOS (0 on macOS)
        public let peakThermalState: Int
    }

    /// The SXCK checkpoint header (sx_checkpoint_info — the parameter tensors are not loaded).
    /// Authoritative for the resume arguments: the fingerprint fields are derived from the file,
    /// not from what the app remembers (this blocks the §8.1 degree contamination incident).
    public struct CheckpointInfo: Sendable {
        public let iterationsDone: Int
        public let totalIterations: Int
        public let shDegree: Int
        public let factor: Int
        public let testSplit: Int
        public let mcmc: Bool
        public let capMax: Int
        public let antialiased: Bool
        public let ssimWeight: Float
        public let sceneScale: Float
        public let splatCount: Int
        public let fp16: Bool
    }

    // MARK: - State (process-global — consistent with the C surface premise of one concurrent run)

    private static let runActive = Mutex<Bool>(false)

    public init() {}

    // MARK: - train / resume

    /// Trains from a COLMAP bundle (dataDirectory: sparse/0/*.bin + images[_N]/).
    /// 1:1 with TrainerBackend.train — progress fires at every step boundary plus the synthesized
    /// cooling ticks, and the return value is a RunResult (on cancellation it returns
    /// outcome == .cancelled instead of throwing — the cancellation semantics in the file header
    /// comment).
    public func train(
        dataDirectory: URL, options: Options,
        progress: @escaping @Sendable (Progress) -> Void
    ) async throws -> RunResult {
        try await run(dataDirectory: dataDirectory, options: options,
                      resumeCheckpoint: nil, additionalIterations: 0, progress: progress)
    }

    /// **Extended resume** from a checkpoint — 1:1 with
    /// TrainerBackend.resume(checkpoint, additionalIterations). Trains additionalIterations more
    /// iterations starting from the checkpoint iter.
    ///
    /// Semantics (this resolves the Task 2 carry-over "total_iters fingerprint vs. extension" —
    /// the authoritative rationale is splatx_train_c.h SxTrainArgs.resume_extend_iters): the
    /// ratio-based schedule denominator (total_iters) stays **fixed** at the checkpoint value and
    /// only the run is extended beyond it — lr keeps decaying monotonically, SH promotion is
    /// saturated, and it is outside the MCMC refine window (stop at 5/6 by default), so this is
    /// pure refinement without growth. Consistent with the iOS capture app's
    /// TrainingConfig.enhance("additional iters past densification — pure refinement").
    ///
    /// The **model fingerprint fields of options are ignored** and the checkpoint header values
    /// are authoritative (iterations/shDegree/factor/testSplit/mcmc/capMax/ssimWeight/antialiased
    /// — the same rule as the msplat precedent, where resume overrides degree from the
    /// checkpoint). Only the environment fields are used:
    /// outputURL/previewURL/checkpointURL(+every/fp16)/evalEvery/logEvery/thermalMode/
    /// resident/metallibURL/kernelSourceDirectory/maxPoints (no effect — resume skips init).
    public func resume(
        checkpoint: URL, dataDirectory: URL, additionalIterations: Int, options: Options,
        progress: @escaping @Sendable (Progress) -> Void
    ) async throws -> RunResult {
        guard additionalIterations >= 1 else {
            throw SplatXTrainerError.invalidArgument("additionalIterations must be >= 1")
        }
        let info = try Self.checkpointInfo(at: checkpoint) // derive the fingerprint + verify readability early
        var o = options
        o.iterations = info.totalIterations // ignored by C on the extend path, but kept consistent
        o.shDegree = info.shDegree
        o.factor = info.factor
        o.testSplit = info.testSplit
        o.mcmc = info.mcmc
        o.capMax = info.capMax
        o.autoCap = false // the fingerprint cap is authoritative — the contract is that
                          // fingerprint verification rejects a re-estimated mismatch
        o.ssimWeight = info.ssimWeight
        o.antialiased = info.antialiased
        if info.testSplit <= 0 { o.evalEvery = 0 } // eval needs test views (matches C validation up front)
        return try await run(dataDirectory: dataDirectory, options: o,
                             resumeCheckpoint: checkpoint,
                             additionalIterations: additionalIterations, progress: progress)
    }

    // MARK: - The resign-active surface (§7.5 — stop submitting → drain → fp16 checkpoint, 3-second contract)

    /// **Pins** a training run in progress at a step-boundary safe point (a hold — a latch that a
    /// thermal transition cannot release), confirms the GPU drain (the step returning = the sync
    /// points inside the step have completed), and then saves the checkpoint on the calling
    /// thread. On success the hold is **kept** — the loop stays parked (no background GPU
    /// submission); release it with resumeAfterPause() when returning to the foreground.
    ///
    /// **Blocking** (waiting to park ≤ parkTimeout, plus the save time — measured ~2.1s for fp16
    /// at 1M, Checkpoint.h): prefer calling it from a non-main context together with a background
    /// task assertion in the resign-active handler.
    ///
    /// Distinguishing the failures (each calls for a different response — the authoritative
    /// contract is splatx_train_c.h sx_train_pause_checkpoint):
    /// - `.noActiveTrainingLoop`: no run / inside the load-and-init stretch / while finishing —
    ///   it means there is no training state to save, so this is **safe to ignore**.
    /// - `.drainTimeout`: the safe point was not reached within parkTimeout — the loop is still
    ///   running while the checkpoint was not saved and the hold has been released (**never
    ///   ignore this** — background GPU submission can continue. Retry, or call requestStop and
    ///   then report).
    /// - `.pauseFailed`: the save failed (disk and the like — the hold has been released).
    public func pauseAndDrain(checkpointURL: URL, fp16: Bool = true,
                              parkTimeout: TimeInterval = 3.0) throws {
        var err = [CChar](repeating: 0, count: 4096)
        let rc = checkpointURL.path.withCString { path in
            sx_train_pause_checkpoint(path, fp16 ? 1 : 0, Int32(parkTimeout * 1000),
                                      &err, Int32(err.count))
        }
        if rc == 0 { return }
        let msg = errBufString(err)
        switch rc {
        case -1: throw SplatXTrainerError.noActiveTrainingLoop(msg)
        case -2: throw SplatXTrainerError.drainTimeout(msg)
        default: throw SplatXTrainerError.pauseFailed(msg)
        }
    }

    /// Releases the pauseAndDrain hold — resumes training when returning to the foreground (the
    /// thermal pause condition stays independent: even after the release, if the state is at or
    /// above the pause threshold (critical for balanced), the thermal policy takes over). A no-op
    /// when there is no active run.
    public func resumeAfterPause() { sx_train_resume_after_pause() }

    /// Requests a cooperative stop (a UI cancel button and the like — the same path as Task
    /// cancellation). At the next step boundary it writes the checkpoint plus a partial export and
    /// then returns with outcome == .cancelled.
    public func requestStop() { sx_train_request_stop() }

    /// Whether the training loop is parked at a safe point (a thermal pause or a hold — the input
    /// for the cooling UI).
    public var isPaused: Bool { sx_train_paused() == 1 }

    // MARK: - Query surface

    /// Queries the SXCK checkpoint header (reads a few KB — the tensors are not loaded).
    public static func checkpointInfo(at url: URL) throws -> CheckpointInfo {
        var err = [CChar](repeating: 0, count: 4096)
        var info = SxCheckpointInfo()
        let rc = url.path.withCString { p in
            sx_checkpoint_info(p, &info, &err, Int32(err.count))
        }
        guard rc == 0 else {
            throw SplatXTrainerError.checkpointUnreadable(errBufString(err))
        }
        return CheckpointInfo(
            iterationsDone: Int(info.iter), totalIterations: Int(info.total_iters),
            shDegree: Int(info.sh_degree), factor: Int(info.factor),
            testSplit: Int(info.test_split), mcmc: info.mcmc_enabled != 0,
            capMax: Int(info.cap_max), antialiased: info.antialiased != 0,
            ssimWeight: info.ssim_weight, sceneScale: info.scene_scale,
            splatCount: Int(info.alive_count), fp16: info.fp16 != 0)
    }

    /// Per-device cap_max estimation (§7.2 — the input for the UI Quality slider range before
    /// training). budgetInsufficient = the lower bound (100k) was force-clamped — the input for
    /// deciding whether to warn about or refuse an insufficient budget (carry-over ④).
    public static func suggestedCapMax(
        budgetBytes: Int64, imageWidth: Int, imageHeight: Int, gtImageCount: Int,
        shDegree: Int, gtCacheOn: Bool = true
    ) -> (capMax: Int, budgetInsufficient: Bool)? {
        var short: Int32 = 0
        let cap = sx_suggest_cap_max(budgetBytes, Int32(imageWidth), Int32(imageHeight),
                                     Int32(gtImageCount), Int32(shDegree), gtCacheOn ? 1 : 0,
                                     &short)
        guard cap > 0 else { return nil }
        return (Int(cap), short != 0)
    }

    /// The downsample factor policy (§7.3 — based on the original (factor 1) resolution. On iOS
    /// the start is fixed at /4).
    public static func suggestedFactor(imageWidth: Int, imageHeight: Int) -> Int? {
        #if os(iOS) || os(tvOS) || os(visionOS)
        let ios: Int32 = 1
        #else
        let ios: Int32 = 0
        #endif
        let f = sx_suggest_factor(Int32(imageWidth), Int32(imageHeight), ios)
        return f > 0 ? Int(f) : nil
    }

    // MARK: - Internal execution

    private func run(
        dataDirectory: URL, options: Options, resumeCheckpoint: URL?,
        additionalIterations: Int,
        progress: @escaping @Sendable (Progress) -> Void
    ) async throws -> RunResult {
        // The gate for exactly one concurrent training run (consistent with the C global
        // contract — a violation is an explicit refusal, not a deferred wait)
        let acquired = Self.runActive.withLock { active -> Bool in
            if active { return false }
            active = true
            return true
        }
        guard acquired else { throw SplatXTrainerError.anotherRunActive }
        defer { Self.runActive.withLock { $0 = false } }

        try Task.checkCancellation() // cancelled before starting — do not open a run

        // The thermal observer (see the file header comment): inject the current state once and
        // observe transitions → inject into C
        sx_thermal_set_state(Int32(ProcessInfo.processInfo.thermalState.rawValue))
        let observer = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: nil
        ) { _ in
            sx_thermal_set_state(Int32(ProcessInfo.processInfo.thermalState.rawValue))
        }
        defer { NotificationCenter.default.removeObserver(observer) }

        let relay = ProgressRelay(callback: progress)

        // Synthesized cooling ticks (TrainingProgress.coolingDown): poll the park observation
        // (sx_train_paused) every 0.5s. The C callback only arrives at step boundaries, so the
        // "Cooling down…" ticks during a pause are synthesized here.
        let coolingTask = Task.detached(priority: .utility) { [relay] in
            while relay.stillRunning {
                if Task.isCancelled { break } // no cooling ticks after cancellation (stop ends it soon)
                if sx_train_paused() == 1 { relay.emitCooling() }
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
        }
        defer { coolingTask.cancel() }

        // C string lifetimes: kept for the whole blocking run (the thread closure holds a strong
        // reference to the pool). The pointer bundle is @unchecked Sendable — the pool owns the
        // lifetimes and there is exactly one consuming thread.
        let pool = CStringPool()
        let cp = CPointerArgs(
            dataDir: pool.add(dataDirectory.path),
            out: pool.add(options.outputURL.path),
            preview: pool.add(options.previewURL?.path),
            metallib: pool.add(options.metallibURL?.path),
            kernelSrc: pool.add(options.kernelSourceDirectory?.path),
            resume: pool.add(resumeCheckpoint?.path),
            saveCk: pool.add(options.checkpointURL?.path),
            livePly: pool.add(options.livePlyURL?.path))
        let opts = options
        // The snapshot relay (app round 3 Task 1): the C callback (on the training-loop thread)
        // calls the handler synchronously. Lifetime follows the same rule as ProgressRelay — a
        // strong reference from the thread closure plus withExtendedLifetime.
        let snapshotRelay = opts.snapshotHandler.map(SnapshotRelay.init)

        let (rc, errMsg, stats):
            (Int32, String, SxTrainStats) = await withTaskCancellationHandler {
            await withCheckedContinuation { cont in
                // A dedicated thread — sx_train_run_args blocks for minutes, so it MUST NOT
                // occupy the cooperative thread pool (the Swift concurrency executor).
                let thread = Thread {
                    var err = [CChar](repeating: 0, count: 4096)
                    var stats = SxTrainStats()
                    var a = SxTrainArgs()
                    a.data_dir = cp.dataDir
                    a.factor = Int32(opts.factor)
                    a.iters = Int32(opts.iterations)
                    a.max_points = Int64(opts.maxPoints)
                    a.sh_degree = Int32(opts.shDegree)
                    a.out_path = cp.out
                    a.preview_png = cp.preview
                    a.eval_image = 0
                    a.kernel_src_dir = cp.kernelSrc
                    a.metallib_path = cp.metallib
                    a.log_every = Int32(opts.logEvery)
                    a.mcmc = opts.mcmc ? 1 : 0
                    a.cap_max = Int64(opts.capMax)
                    a.test_split = Int32(opts.testSplit)
                    a.eval_every = Int32(opts.evalEvery)
                    a.ssim_weight = opts.ssimWeight
                    a.depth_weight = opts.depthWeight
                    a.antialiased = opts.antialiased ? 1 : 0
                    a.resident = opts.resident ? 1 : 0
                    a.resume_checkpoint = cp.resume
                    a.save_checkpoint_path = cp.saveCk
                    a.checkpoint_every = Int32(opts.checkpointEvery)
                    a.auto_cap = opts.autoCap ? 1 : 0
                    a.memory_budget_bytes = opts.memoryBudgetBytes
                    a.thermal_mode = opts.thermalMode.rawValue
                    a.resume_extend_iters = Int32(additionalIterations)
                    a.checkpoint_fp16 = opts.checkpointFP16 ? 1 : 0
                    a.live_ply_path = cp.livePly // nil = off (app round 2 Task 1)
                    a.live_ply_every = Int32(opts.livePlyEvery)
                    if let snapshotRelay { // in-memory snapshot (app round 3 Task 1 — cb+every pair)
                        a.snapshot_cb = { sPtr, user in // capture-free — @convention(c)
                            guard let sPtr, let user else { return }
                            Unmanaged<SnapshotRelay>.fromOpaque(user).takeUnretainedValue()
                                .emit(sPtr.pointee)
                        }
                        a.snapshot_user = Unmanaged.passUnretained(snapshotRelay).toOpaque()
                        a.snapshot_every = Int32(opts.snapshotEvery)
                    }
                    a.progress = { pPtr, user in // capture-free — @convention(c)
                        guard let pPtr, let user else { return }
                        Unmanaged<ProgressRelay>.fromOpaque(user).takeUnretainedValue()
                            .emit(pPtr.pointee)
                    }
                    a.progress_user = Unmanaged.passUnretained(relay).toOpaque()
                    let rc = withUnsafeMutablePointer(to: &stats) { sp -> Int32 in
                        a.stats = sp
                        return sx_train_run_args(&a, &err, Int32(err.count))
                    }
                    withExtendedLifetime(pool) {} // explicitly pin the C pointer lifetimes
                    withExtendedLifetime(relay) {}
                    withExtendedLifetime(snapshotRelay) {}
                    cont.resume(returning: (rc, errBufString(err), stats))
                }
                thread.name = "splatx.trainer"
                thread.qualityOfService = .userInitiated
                thread.stackSize = 4 << 20
                thread.start()
            }
        } onCancel: {
            // Cooperative cancellation — even while parked (in a cooling pause) it wakes
            // immediately and stops at a safe point. For the window before registration
            // (load/init), the relay re-delivers on the next tick (see the note inside emit).
            relay.noteCancelled()
            sx_train_request_stop()
        }
        relay.finish()

        if rc == 1 { throw SplatXTrainerError.trainFailed(errMsg) }

        let outcome: RunResult.Outcome = rc == 2 ? .thermalAborted
            : rc == 3 ? .cancelled : .completed
        let snapshot = relay.snapshot()
        let fileSize = (try? FileManager.default
            .attributesOfItem(atPath: options.outputURL.path)[.size] as? Int).flatMap { $0 } ?? 0
        return RunResult(
            outcome: outcome,
            outputURL: options.outputURL,
            checkpointURL: stats.checkpoint_saved != 0 ? options.checkpointURL : nil,
            iterationsCompleted: Int(stats.completed_iters),
            runUntil: Int(stats.run_until),
            totalIterations: Int(stats.total_iters),
            finalPSNR: stats.final_eval_psnr.isNaN ? nil : Double(stats.final_eval_psnr),
            finalDepthRMSE: stats.final_depth_rmse.isNaN ? nil : Double(stats.final_depth_rmse),
            trainSeconds: stats.train_seconds,
            wallClockSeconds: snapshot.elapsed,
            splatCount: Int(stats.exported_splats),
            fileSizeBytes: fileSize,
            peakMemoryFootprintMB: snapshot.peakFootprintMB,
            minAvailableMemoryMB: snapshot.minAvailableMB,
            peakThermalState: snapshot.peakThermal)
    }
}

// MARK: - The progress relay (C callback thread → Swift closure, joined with memory/peak observation)

private final class ProgressRelay: Sendable {
    private struct Track {
        var peakFootprintMB = 0
        var minAvailableMB = 0 // 0 = not observed (macOS — same as the TrainingMetrics convention)
        var peakThermal = 0
        var lastIteration = 0
        var lastRunUntil = 0
        var lastTotal = 0
        var lastLoss = 0.0
        var lastSplats = 0
        var cancelled = false
        var stopForwarded = false
        var running = true
    }

    private let callback: @Sendable (SplatXTrainer.Progress) -> Void
    private let startedAt = Date()
    private let state = Mutex(Track())

    init(callback: @escaping @Sendable (SplatXTrainer.Progress) -> Void) {
        self.callback = callback
    }

    var stillRunning: Bool { state.withLock { $0.running } }
    func finish() { state.withLock { $0.running = false } }
    func noteCancelled() { state.withLock { $0.cancelled = true } }

    func snapshot() -> (peakFootprintMB: Int, minAvailableMB: Int, peakThermal: Int,
                        elapsed: TimeInterval) {
        let t = state.withLock { $0 }
        return (t.peakFootprintMB, t.minAvailableMB, t.peakThermal,
                Date().timeIntervalSince(startedAt))
    }

    /// The C callback (on the training-loop thread) — at every step boundary.
    func emit(_ p: SxTrainProgress) {
        let footprint = Self.footprintMB()
        let available = Self.availableMB()
        let forwardStop: Bool = state.withLock { t in
            t.peakFootprintMB = max(t.peakFootprintMB, footprint)
            if available > 0 {
                t.minAvailableMB = t.minAvailableMB == 0 ? available
                    : min(t.minAvailableMB, available)
            }
            t.peakThermal = max(t.peakThermal, Int(p.thermal_state))
            t.lastIteration = Int(p.iter)
            t.lastRunUntil = Int(p.run_until)
            t.lastTotal = Int(p.total_iters)
            t.lastLoss = Double(p.loss)
            t.lastSplats = Int(p.splat_count)
            if t.cancelled && !t.stopForwarded {
                t.stopForwarded = true
                return true // the cancellation arrived before control registration — we are now
                            // inside the loop, so delivery is certain
            }
            return false
        }
        if forwardStop { sx_train_request_stop() }
        callback(SplatXTrainer.Progress(
            iteration: Int(p.iter), runUntil: Int(p.run_until),
            totalIterations: Int(p.total_iters), loss: Double(p.loss),
            psnr: p.psnr.isNaN ? nil : Double(p.psnr), splatCount: Int(p.splat_count),
            stepMilliseconds: p.step_ms, nIsects: Int(p.n_isects),
            elapsedSeconds: Date().timeIntervalSince(startedAt),
            thermalState: Int(p.thermal_state), coolingDown: false,
            memoryFootprintMB: footprint,
            peakMemoryFootprintMB: state.withLock { $0.peakFootprintMB },
            availableMemoryMB: available,
            depthRMSE: p.depth_rmse.isNaN ? nil : Double(p.depth_rmse)))
    }

    /// The synthesized cooling tick — the input for the UI "Cooling down…" while parked (no step
    /// callbacks arrive).
    func emitCooling() {
        let footprint = Self.footprintMB()
        let available = Self.availableMB()
        let t = state.withLock { $0 }
        guard t.running else { return }
        callback(SplatXTrainer.Progress(
            iteration: t.lastIteration, runUntil: t.lastRunUntil,
            totalIterations: t.lastTotal, loss: t.lastLoss, psnr: nil,
            splatCount: t.lastSplats, stepMilliseconds: 0, nIsects: 0,
            elapsedSeconds: Date().timeIntervalSince(startedAt),
            thermalState: Int(sx_thermal_get_state()), coolingDown: true,
            memoryFootprintMB: footprint, peakMemoryFootprintMB: t.peakFootprintMB,
            availableMemoryMB: available, depthRMSE: nil))
    }

    /// task_vm_info.phys_footprint (MB) — the same metric as the Xcode memory gauge.
    private static func footprintMB() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return Int(info.phys_footprint / 1_048_576)
    }

    /// os_proc_available_memory (MB) — the headroom up to the jetsam limit. 0 on macOS (the
    /// not-reported convention — same as "0 = not reported" for the iOS capture app's
    /// TrainingProgress.availableMemoryMB).
    private static func availableMB() -> Int {
        #if os(iOS) || os(tvOS) || os(watchOS) || os(visionOS)
        return Int(os_proc_available_memory() / 1_048_576)
        #else
        return 0
        #endif
    }
}

// MARK: - The snapshot relay (C callback thread → synchronous handler call — app round 3 Task 1)

/// Responsible only for converting a C SxSplatSnapshot into a SplatSnapshot pointer view. The
/// handler runs synchronously on the training-loop thread, and the view's buffers become invalid
/// as soon as emit (= the C callback) returns (the authoritative contract is the pointer lifetimes
/// of splatx_train_c.h SxSplatSnapshot — copying is the handler's responsibility).
private final class SnapshotRelay: Sendable {
    private let handler: @Sendable (SplatXTrainer.SplatSnapshot) -> Void

    init(handler: @escaping @Sendable (SplatXTrainer.SplatSnapshot) -> Void) {
        self.handler = handler
    }

    /// The C callback (on the training-loop thread) — on every snapshotEvery tick.
    func emit(_ s: SxSplatSnapshot) {
        let n = Int(s.count)
        let k = (Int(s.sh_degree) + 1) * (Int(s.sh_degree) + 1) - 1 // higher-band coefficient count
        handler(SplatXTrainer.SplatSnapshot(
            iter: Int(s.iter), shDegree: Int(s.sh_degree), count: n,
            means: UnsafeBufferPointer(start: s.means, count: n * 3),
            quats: UnsafeBufferPointer(start: s.quats, count: n * 4),
            scales: UnsafeBufferPointer(start: s.scales, count: n * 3),
            opacities: UnsafeBufferPointer(start: s.opacities, count: n),
            sh0: UnsafeBufferPointer(start: s.sh0, count: n * 3),
            shN: s.shN.map { UnsafeBufferPointer(start: $0, count: n * k * 3) }))
    }
}

// MARK: - The C string pool (pins pointer lifetimes for the duration of the blocking run)

private final class CStringPool: @unchecked Sendable {
    private var ptrs: [UnsafeMutablePointer<CChar>] = []
    func add(_ s: String?) -> UnsafePointer<CChar>? {
        guard let s else { return nil }
        let p = strdup(s)! // cannot fail except on process memory exhaustion
        ptrs.append(p)
        return UnsafePointer(p)
    }
    deinit { for p in ptrs { free(p) } }
}

/// The bundle of C string pointers owned by the pool — consumed by exactly one training thread,
/// with the lifetimes guaranteed by the strong reference to the pool (annotated explicitly because
/// UnsafePointer is not Sendable).
private struct CPointerArgs: @unchecked Sendable {
    let dataDir: UnsafePointer<CChar>?
    let out: UnsafePointer<CChar>?
    let preview: UnsafePointer<CChar>?
    let metallib: UnsafePointer<CChar>?
    let kernelSrc: UnsafePointer<CChar>?
    let resume: UnsafePointer<CChar>?
    let saveCk: UnsafePointer<CChar>?
    let livePly: UnsafePointer<CChar>? // live PLY export (app round 2 Task 1)
}

/// err_buf (a NUL-terminated CChar array) → String (the same pattern as the CLI — a replacement
/// for the discouraged String(cString:)).
private func errBufString(_ buf: [CChar]) -> String {
    String(decoding: buf.prefix(while: { $0 != 0 }).map { UInt8(bitPattern: $0) },
           as: UTF8.self)
}
