// SplatXTrainer E2E (Phase 6 round 2 Task 5 — requires Metal): using a small synthetic COLMAP
// bundle, exercises the (a) async run-to-completion + progress stream, (b) extended resume
// (additionalIterations — fixed-denominator semantics), and (c) cooperative Task cancellation
// (safe-point stop + checkpoint) contracts end to end.
// thermalMode is .performance — pause never fires regardless of the machine's actual thermal
// state (auto-injected by the observer), so determinism is preserved (external stop/hold are
// mode-independent — ThermalGovernor.h).
import Foundation
import Synchronization
import Testing
@testable import SplatXMetal

extension GpuTests {
@Suite struct SplatXTrainerTests {

    /// Kernel source directory (the repo canonical tree via #filePath — tests use the
    /// source-compile path)
    private static var kernelSrcDir: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // SplatXMetalTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // repo root
            .appendingPathComponent("Sources/SplatXKernelSrc")
    }

    private static func makeWorkDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("splatx-trainer-test-\(UUID().uuidString)")
        try SyntheticColmap.makeBundle(at: dir)
        return dir
    }

    private static func makeOptions(in dir: URL, iterations: Int) -> SplatXTrainer.Options {
        var o = SplatXTrainer.Options(outputURL: dir.appendingPathComponent("out.ply"))
        o.iterations = iterations
        o.factor = 1
        o.shDegree = 1
        o.maxPoints = 200
        o.mcmc = false          // small-scale check — no MCMC arena needed (the contract is owned by the C++ tests)
        o.ssimWeight = 0        // bypasses the SSIM path — pure L1 (small and fast)
        o.testSplit = 0
        o.resident = true
        o.thermalMode = .performance
        o.logEvery = 1_000      // suppresses stdout noise (progress is verified via the callback)
        o.checkpointURL = dir.appendingPathComponent("ck.sxck")
        o.checkpointFP16 = true
        o.kernelSourceDirectory = kernelSrcDir
        return o
    }

    // (a) Run to completion: verifies a progress tick per step + the RunResult fields + the
    // artifacts (PLY/checkpoint)
    @Test func trainCompletesAndStreamsProgress() async throws {
        let dir = try Self.makeWorkDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let options = Self.makeOptions(in: dir, iterations: 40)

        let ticks = Mutex<[SplatXTrainer.Progress]>([])
        let result = try await SplatXTrainer().train(dataDirectory: dir, options: options) {
            p in ticks.withLock { $0.append(p) }
        }

        #expect(result.outcome == .completed)
        #expect(result.iterationsCompleted == 40)
        #expect(result.runUntil == 40)
        #expect(result.totalIterations == 40)
        #expect(result.splatCount > 0)          // the N that passed the export filter
        #expect(result.fileSizeBytes > 0)
        #expect(result.peakMemoryFootprintMB > 0)
        #expect(result.checkpointURL == options.checkpointURL)
        #expect(FileManager.default.fileExists(atPath: options.outputURL.path))

        let got = ticks.withLock { $0 }
        #expect(got.count == 40)                // one tick per step boundary
        #expect(got.last?.iteration == 40)
        #expect(got.allSatisfy { $0.loss.isFinite && $0.splatCount > 0 && !$0.coolingDown })
        // Monotonic progress + the schedule denominator as reported
        #expect(zip(got, got.dropFirst()).allSatisfy { $0.iteration < $1.iteration })
        #expect(got.allSatisfy { $0.totalIterations == 40 && $0.runUntil == 40 })

        // Checkpoint header query (the fp16 save contract + fingerprint round-trip)
        let info = try SplatXTrainer.checkpointInfo(at: options.checkpointURL!)
        #expect(info.iterationsDone == 40)
        #expect(info.totalIterations == 40)
        #expect(info.shDegree == 1)
        #expect(info.fp16)
        #expect(info.mcmc == false)
    }

    // (b) Extended resume (the TrainerBackend.resume(checkpoint, additionalIterations) semantics):
    // K more iters from the checkpoint iter — the ratio-schedule denominator (totalIterations)
    // stays fixed at the checkpoint value; only runUntil extends past it (pure refinement —
    // per splatx_train_c.h).
    @Test func resumeExtendsWithFixedScheduleDenominator() async throws {
        let dir = try Self.makeWorkDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let options = Self.makeOptions(in: dir, iterations: 25)
        let trainer = SplatXTrainer()

        let first = try await trainer.train(dataDirectory: dir, options: options) { _ in }
        #expect(first.outcome == .completed && first.iterationsCompleted == 25)

        let ticks = Mutex<[SplatXTrainer.Progress]>([])
        let second = try await trainer.resume(
            checkpoint: options.checkpointURL!, dataDirectory: dir,
            additionalIterations: 15, options: options
        ) { p in ticks.withLock { $0.append(p) } }

        #expect(second.outcome == .completed)
        #expect(second.iterationsCompleted == 40) // 25 + 15
        #expect(second.runUntil == 40)
        #expect(second.totalIterations == 25)     // fixed denominator — the core assertion of the extension semantics
        let got = ticks.withLock { $0 }
        #expect(got.count == 15)                  // steps only over the extension range
        #expect(got.first?.iteration == 26 && got.last?.iteration == 40)
        #expect(got.allSatisfy { $0.totalIterations == 25 && $0.runUntil == 40 })

        // The checkpoint after extension: iter is 40, the denominator fingerprint is still 25
        // (re-resumable state)
        let info = try SplatXTrainer.checkpointInfo(at: options.checkpointURL!)
        #expect(info.iterationsDone == 40)
        #expect(info.totalIterations == 25)
    }

    // (d) pauseAndDrain (§7.5 resign-active): hold-latch park → fp16 checkpoint from an external
    // thread → hold retained (isPaused) → resumeAfterPause resumes → cancellation terminates
    // regardless of park state.
    @Test func pauseAndDrainCheckpointsAtSafePoint() async throws {
        let dir = try Self.makeWorkDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        var options = Self.makeOptions(in: dir, iterations: 100_000) // a sufficiently long run
        options.checkpointURL = nil // decoupled from the end-of-run save — verifies the pause save only
        let trainer = SplatXTrainer()

        let seen = Mutex<Int>(0)
        let task = Task {
            try await trainer.train(dataDirectory: dir, options: options) { p in
                seen.withLock { $0 = max($0, p.iteration) }
            }
        }
        let deadline = Date().addingTimeInterval(60)
        while seen.withLock({ $0 }) < 5 {
            try #require(Date() < deadline, "training did not reach 5 progress ticks (60s)")
            try await Task.sleep(nanoseconds: 20_000_000)
        }

        // Stop submissions → drain → fp16 checkpoint (blocking — the test thread is the calling thread)
        let pauseCk = dir.appendingPathComponent("pause.sxck")
        try trainer.pauseAndDrain(checkpointURL: pauseCk, fp16: true, parkTimeout: 10.0)
        #expect(trainer.isPaused) // hold retained — parked state (no background GPU submission)
        let info = try SplatXTrainer.checkpointInfo(at: pauseCk)
        #expect(info.fp16)
        #expect(info.iterationsDone >= 5 && info.iterationsDone < 100_000)
        #expect(info.totalIterations == 100_000)

        // Confirms progress halts while parked (hold is an absolute latch — no step callbacks)
        let atPause = seen.withLock { $0 }
        try await Task.sleep(nanoseconds: 200_000_000)
        #expect(seen.withLock { $0 } == atPause)

        // Return → resume → terminate via cancellation (stop wakes regardless of park state —
        // ThermalGovernor.h)
        trainer.resumeAfterPause()
        task.cancel()
        let result = try await task.value
        #expect(result.outcome == .cancelled)
        #expect(result.iterationsCompleted >= info.iterationsDone)
    }

    // (c) Cooperative Task cancellation: stop at a safe point (step boundary) → save the
    // checkpoint + partial export → return outcome == .cancelled instead of throwing
    // CancellationError (the cancellation semantics in the SplatXTrainer.swift header — the
    // adapter restores the app convention (throw)).
    @Test func cancellationStopsAtSafePointAndCheckpoints() async throws {
        let dir = try Self.makeWorkDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let options = Self.makeOptions(in: dir, iterations: 5_000)

        let seen = Mutex<Int>(0)
        let task = Task {
            try await SplatXTrainer().train(dataDirectory: dir, options: options) { p in
                seen.withLock { $0 = max($0, p.iteration) }
            }
        }
        // Cancel only after training has actually progressed (waits for 10 ticks — the deadline
        // is a fail-safe)
        let deadline = Date().addingTimeInterval(60)
        while seen.withLock({ $0 }) < 10 {
            try #require(Date() < deadline, "training did not reach 10 progress ticks (60s)")
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        task.cancel()
        let result = try await task.value

        #expect(result.outcome == .cancelled)
        #expect(result.iterationsCompleted >= 10 && result.iterationsCompleted < 5_000)
        #expect(FileManager.default.fileExists(atPath: options.outputURL.path)) // partial export
        // A checkpoint is saved on cancellation — resume material (iter is consistent with the
        // stop point)
        let info = try SplatXTrainer.checkpointInfo(at: options.checkpointURL!)
        #expect(info.iterationsDone == result.iterationsCompleted)
        #expect(info.totalIterations == 5_000)
    }
}
}
