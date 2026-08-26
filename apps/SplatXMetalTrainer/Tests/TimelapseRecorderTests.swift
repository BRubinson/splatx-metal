// 2026-08-04 spec §3.2/§3.4: the pure surface of the timelapse recorder — the frame ledger
// (index assignment, the reset generation), the overlay label text, and the JPEG writer.
// The capture proper (a capture-only SplatRenderer, the offscreen render, and the readback) is
// GPU work and belongs to the manual checks (spec §4.3 M6) — this app does not test Metal
// rendering.
import XCTest
import Metal
import simd
@testable import SplatXMetalTrainer

final class TimelapseRecorderTests: XCTestCase {

    // MARK: - Frame ledger

    func testLedgerAssignsSequentialIndices() {
        var ledger = TimelapseFrameLedger()
        XCTAssertEqual(ledger.nextIndex, 0)
        let a = ledger.reserve()
        let b = ledger.reserve()
        XCTAssertEqual(a.index, 0)
        XCTAssertEqual(b.index, 1)
        XCTAssertEqual(a.generation, b.generation)
        XCTAssertEqual(ledger.nextIndex, 2)
        XCTAssertTrue(ledger.commit(generation: a.generation))
        XCTAssertTrue(ledger.commit(generation: b.generation))
        XCTAssertEqual(ledger.savedCount, 2)
    }

    /// Spec §3.3: pressing "shoot from here" deletes every frame collected so far and restarts
    /// from index 0 — a state where the viewpoint teleports mid-video cannot exist.
    func testLedgerResetRestartsAtZeroAndBumpsGeneration() {
        var ledger = TimelapseFrameLedger()
        let stale = ledger.reserve()
        XCTAssertTrue(ledger.commit(generation: stale.generation))
        ledger.reset()
        XCTAssertEqual(ledger.nextIndex, 0)
        XCTAssertEqual(ledger.savedCount, 0)
        XCTAssertNotEqual(ledger.generation, stale.generation)
        let fresh = ledger.reserve()
        XCTAssertEqual(fresh.index, 0)
        XCTAssertEqual(fresh.generation, ledger.generation)
    }

    /// The race guard: a capture that started before the reset must not write a file afterwards.
    func testLedgerRefusesStaleGenerationCommit() {
        var ledger = TimelapseFrameLedger()
        let inFlight = ledger.reserve()
        ledger.reset()
        XCTAssertFalse(ledger.commit(generation: inFlight.generation))
        XCTAssertEqual(ledger.savedCount, 0)
    }

    /// 메일박스는 미소비 스냅샷을 덮어쓰므로(LiveSplatStreamer.swift:88-98) 파이프라인이 실제로
    /// 보는 스왑 시퀀스가 everySteps의 배수를 통째로 건너뛸 수 있다. 임계값 레저는 그래도
    /// 주기당 1장을 남긴다 — 모듈로였다면 이 시퀀스에서 캡처가 0장이다.
    /// (spec §3.7 각주 — 50 격자 전제는 성립하지 않는다.)
    func testLedgerClaimSurvivesDroppedSnapshots() {
        var ledger = TimelapseFrameLedger()
        let dropped = [50, 150, 250, 350]        // 100의 배수가 전부 드롭된 경우
        XCTAssertEqual(dropped.filter { ledger.claim(iter: $0, everySteps: 100) },
                       [150, 250, 350])
        var clean = TimelapseFrameLedger()        // 드롭이 없으면 정확히 격자 위에 떨어진다
        XCTAssertEqual(stride(from: 50, through: 400, by: 50).filter {
            clean.claim(iter: $0, everySteps: 100)
        }, [100, 200, 300, 400])
        var reset = TimelapseFrameLedger()
        XCTAssertTrue(reset.claim(iter: 100, everySteps: 100))
        reset.reset()                             // "여기서 찍기" 후 다음 스왑에서 즉시 재개
        XCTAssertTrue(reset.claim(iter: 150, everySteps: 100))
    }

    /// FIX 2 (I-3): `permits` is the write permission and is **non-mutating** — the saved count
    /// must reflect files that actually reached disk, not captures that were allowed to try.
    /// Task 14 decides "fewer than 2 frames → skip encoding" from `savedCount`, so counting at
    /// permission time would run the encoder over an empty directory.
    func testLedgerCountsOnlyCommittedWrites() {
        var ledger = TimelapseFrameLedger()
        let a = ledger.reserve()
        let b = ledger.reserve()
        XCTAssertTrue(ledger.permits(generation: a.generation))
        XCTAssertTrue(ledger.permits(generation: b.generation))
        XCTAssertEqual(ledger.savedCount, 0)          // 허가만으로는 올라가지 않는다
        XCTAssertTrue(ledger.commit(generation: a.generation))
        XCTAssertEqual(ledger.savedCount, 1)          // 실제 쓰기 성공 1건
        ledger.reset()
        XCTAssertFalse(ledger.permits(generation: b.generation))
    }

    // MARK: - Overlay label (spec §3.4 — "iter 1500 / 30000", baked at capture time)

    func testOverlayText() {
        XCTAssertEqual(TimelapseRecorder.overlayText(iter: 1500, total: 30000),
                       "iter 1500 / 30000")
        XCTAssertEqual(TimelapseRecorder.overlayText(iter: 50, total: 0), "iter 50")
    }

    // MARK: - JPEG writer (CoreGraphics/ImageIO only — no GPU)

    func testWriteFrameProducesJPEGOfTheGivenSize() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("sx-timelapse-write-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let w = 64, h = 48
        // BGRA, opaque — the same memory layout the capture texture reads back.
        var bgra = [UInt8](repeating: 0, count: w * h * 4)
        for i in 0..<(w * h) {
            bgra[i * 4 + 0] = 200   // B
            bgra[i * 4 + 1] = 100   // G
            bgra[i * 4 + 2] = 40    // R
            bgra[i * 4 + 3] = 255   // A
        }
        let url = dir.appendingPathComponent(TimelapseSettings.frameFileName(index: 3))
        TimelapseRecorder.writeFrame(bgra: bgra, width: w, height: h, url: url,
                                     label: TimelapseRecorder.overlayText(iter: 150, total: 7000))
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        XCTAssertEqual(url.lastPathComponent, "frame-00003.jpg")

        let src = try XCTUnwrap(CGImageSourceCreateWithURL(url as CFURL, nil))
        let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(src, 0, nil))
        XCTAssertEqual(image.width, w)
        XCTAssertEqual(image.height, h)
    }

    /// FIX 2 (I-3), spec §3.4 "no silent truncation": a write into a directory that does not exist
    /// used to return silently, so a whole run's frames could evaporate while the recorder still
    /// reported them as saved. The writer now reports the outcome.
    func testWriteFrameReportsFailureWhenTheDirectoryIsMissing() {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("sx-timelapse-absent-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent(TimelapseSettings.frameFileName(index: 0))
        let bgra = [UInt8](repeating: 255, count: 8 * 8 * 4)
        XCTAssertFalse(TimelapseRecorder.writeFrame(bgra: bgra, width: 8, height: 8,
                                                    url: missing, label: nil))
        XCTAssertFalse(FileManager.default.fileExists(atPath: missing.path))
    }

    // MARK: - Pose / priming interlock (FIX 2 — I-1)

    /// `captureIfDue` suspends at `await addChunk` between reading the pose and finishing the
    /// priming render, and Task 12's "shoot from here" can land inside that window. If the flag
    /// were cleared unconditionally, the *new* pose's priming requirement would be consumed by a
    /// render done with the *old* pose — and because `depthFormat: .invalid` makes sort order the
    /// alpha compositing order, the first frame after the restart would show collapsed occlusion.
    /// So the flag is cleared only when the primed pose is still the current one.
    /// (The capture itself is GPU work and stays a manual check — this pins only the interlock.)
    func testPrimingFlagIsKeptWhenThePoseChangedDuringPriming() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let p1 = CameraPose(position: SIMD3<Float>(1, 2, 3),
                            rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), fov: 0.9)
        let p2 = CameraPose(position: SIMD3<Float>(9, 9, 9),
                            rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), fov: 0.9)
        let config = TimelapseRecorder.Config(
            settings: TimelapseSettings(), width: 320, height: 240, totalIterations: 7000,
            frameDirectory: FileManager.default.temporaryDirectory)
        let recorder = TimelapseRecorder(device: device, config: config, pose: p1)

        XCTAssertTrue(recorder.needsPriming)        // 새 렌더러의 정렬기는 원점·−Z 기준이다
        recorder.finishPriming(primedWith: p2)      // 다른 포즈로 프라이밍한 결과는 소진되지 않는다
        XCTAssertTrue(recorder.needsPriming)
        recorder.finishPriming(primedWith: p1)      // 현재 포즈로 프라이밍했을 때만 내려간다
        XCTAssertFalse(recorder.needsPriming)
    }

    // MARK: - The recorder holds the capture pose and its config without touching the GPU
    // (the capture-only SplatRenderer is created lazily on the first capture — spec §3.1)

    func testRecorderInitDoesNotAllocateTheCaptureRenderer() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("sx-timelapse-init-\(UUID().uuidString)", isDirectory: true)
        let config = TimelapseRecorder.Config(settings: TimelapseSettings(), width: 320,
                                              height: 240, totalIterations: 7000,
                                              frameDirectory: dir)
        let pose = CameraPose(position: SIMD3<Float>(1, 2, 3),
                              rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), fov: 0.9)
        let recorder = TimelapseRecorder(device: device, config: config, pose: pose)
        XCTAssertEqual(recorder.savedFrameCount, 0)
        XCTAssertEqual(recorder.failedFrameCount, 0)
        XCTAssertEqual(recorder.nextFrameIndex, 0)
        XCTAssertEqual(recorder.config.width, 320)
        XCTAssertEqual(recorder.config.frameDirectory, dir)
        XCTAssertFalse(recorder.hasCaptureRenderer)   // lazy — nothing is allocated until the first capture attempt
    }

    // MARK: - Config derivation (the wiring's test surface — spec §3.4 paths and §3.1 sizing)

    /// The frame folder is runs/<stamp>/timelapse/ (a sibling of output.ply's folder), and the
    /// capture size comes from frameSize with whatever ratio the caller resolved — since
    /// 2026-08-05 (§2.2) that is `TimelapseSettings.aspectPair`'s output, not the first camera's
    /// intrinsics by construction. makeConfig itself only knows a ratio.
    func testMakeConfigDerivesPathAndSize() {
        let runDirectory = URL(fileURLWithPath: "/p/runs/20260804-120000", isDirectory: true)
        var settings = TimelapseSettings()
        settings.enabled = true
        settings.shortSide = 720
        let config = TimelapseRecorder.makeConfig(settings: settings,
                                                  runDirectory: runDirectory,
                                                  totalIterations: 7000,
                                                  aspectWidth: 1920, aspectHeight: 1080)
        XCTAssertEqual(config.frameDirectory.path, "/p/runs/20260804-120000/timelapse")
        XCTAssertEqual(config.width, 1280)
        XCTAssertEqual(config.height, 720)
        XCTAssertEqual(config.totalIterations, 7000)
        XCTAssertEqual(config.settings, settings)
    }

    /// Neither aspect source had a size (no COLMAP context and no laid-out viewport) → the 16:9
    /// fallback, so a capture still has a sane shape instead of a zero-sized texture. The size
    /// follows the settings' short side, so this is stated in terms of the default rather than a
    /// fixed pair.
    func testMakeConfigFallsBackWhenIntrinsicsAreMissing() {
        let settings = TimelapseSettings()
        let config = TimelapseRecorder.makeConfig(
            settings: settings, runDirectory: URL(fileURLWithPath: "/p/runs/r"),
            totalIterations: 100, aspectWidth: 0, aspectHeight: 0)
        XCTAssertEqual(config.height, settings.shortSide)
        XCTAssertEqual(config.width, 854)          // 480 × 16/9, rounded to even
        XCTAssertEqual(config.height, 480)         // the default today — pinned so a change is deliberate
    }

    /// Spec §3.7(각주 포함): the streamer swaps on the fixed 50-step snapshot grid, so a capture
    /// can only land on that grid. 스냅샷이 드롭되지 않는 한 요청 간격과 실제 간격이 같다 —
    /// 이 테스트가 고정하는 것은 그 **표시상의 기대 간격**뿐이고, 드롭이 섞인 실제 런타임
    /// 판정은 `TimelapseFrameLedger.claim`이 담당한다
    /// (Task 10의 `testLedgerClaimSurvivesDroppedSnapshots`).
    func testCapturePointsLandOnTheSnapshotGrid() {
        var settings = TimelapseSettings()
        settings.enabled = true
        settings.everySteps = 100
        let swaps = stride(from: 50, through: 1000, by: 50).map { $0 }
        let captured = swaps.filter { settings.shouldCapture(iter: $0) }
        XCTAssertEqual(captured, [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        settings.everySteps = 50
        XCTAssertEqual(swaps.filter { settings.shouldCapture(iter: $0) }, swaps)
    }

    // MARK: - "Shoot from here" (spec §3.3) — a new anchor pose, every frame so far deleted,
    // and the index restarted at 0

    func testResetFramesDeletesFilesAndRestartsIndices() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("sx-timelapse-reset-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        // Frames collected so far, plus one unrelated file that must survive.
        for i in 0..<3 {
            try Data([0xFF]).write(
                to: dir.appendingPathComponent(TimelapseSettings.frameFileName(index: i)))
        }
        try Data([0x00]).write(to: dir.appendingPathComponent("keep.txt"))

        let config = TimelapseRecorder.Config(settings: TimelapseSettings(), width: 64,
                                              height: 48, totalIterations: 7000,
                                              frameDirectory: dir)
        let recorder = TimelapseRecorder(
            device: device, config: config,
            pose: CameraPose(position: .zero,
                             rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), fov: 0.9))
        recorder.resetFrames()
        recorder.waitForPendingWrites()

        let remaining = try FileManager.default.contentsOfDirectory(atPath: dir.path).sorted()
        XCTAssertEqual(remaining, ["keep.txt"])
        XCTAssertEqual(recorder.nextFrameIndex, 0)
        XCTAssertEqual(recorder.savedFrameCount, 0)
    }

    /// FIX 3 — spec §3.4 symmetry with the write-failure path (`noteWriteFailure`): a delete that
    /// `resetFrames()`'s sweep cannot perform must not vanish silently, because the file it leaves
    /// behind is a stale-generation frame the encoder will splice in next to the new generation's
    /// frames — the exact "viewpoint teleports mid-video" symptom this feature exists to prevent.
    /// Deleting a directory entry needs *write* permission on the *containing* directory, not on
    /// the file itself, so chmod-ing the temp directory to r-x reliably makes removeItem(at:) fail
    /// for a non-root test runner without touching the frame file's own permission bits.
    func testResetFramesReportsDeleteFailureWhenTheDirectoryIsReadOnly() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("sx-timelapse-reset-ro-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer {
            // Restore write permission first — otherwise removeItem on the directory itself
            // (inside the FileManager cleanup below) would fail too.
            try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: dir.path)
            try? FileManager.default.removeItem(at: dir)
        }
        try Data([0xFF]).write(
            to: dir.appendingPathComponent(TimelapseSettings.frameFileName(index: 0)))
        try FileManager.default.setAttributes([.posixPermissions: 0o555], ofItemAtPath: dir.path)

        let config = TimelapseRecorder.Config(settings: TimelapseSettings(), width: 64,
                                              height: 48, totalIterations: 100,
                                              frameDirectory: dir)
        let recorder = TimelapseRecorder(
            device: device, config: config,
            pose: CameraPose(position: .zero,
                             rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), fov: 0.9))
        recorder.resetFrames()
        recorder.waitForPendingWrites()

        XCTAssertEqual(recorder.failedFrameDeleteCount, 1)
        // Restore permission before inspecting the directory — the stale frame surviving is
        // exactly the observable signal this fix adds (previously the failure was silent).
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: dir.path)
        let remaining = try FileManager.default.contentsOfDirectory(atPath: dir.path)
        XCTAssertEqual(remaining, [TimelapseSettings.frameFileName(index: 0)])
    }

    func testSetPoseReplacesTheCaptureAnchor() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let config = TimelapseRecorder.Config(
            settings: TimelapseSettings(), width: 64, height: 48, totalIterations: 100,
            frameDirectory: FileManager.default.temporaryDirectory
                .appendingPathComponent("sx-timelapse-pose", isDirectory: true))
        let first = CameraPose(position: SIMD3<Float>(0, 0, 5),
                               rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), fov: 0.9)
        let recorder = TimelapseRecorder(device: device, config: config, pose: first)
        XCTAssertEqual(recorder.currentPose, first)
        let second = CameraPose(position: SIMD3<Float>(3, -1, 2),
                                rotation: simd_quatf(ix: 0, iy: 1, iz: 0, r: 0).normalized,
                                fov: 1.1)
        recorder.setPose(second)
        XCTAssertEqual(recorder.currentPose, second)
    }
}
