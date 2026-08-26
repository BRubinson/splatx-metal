// Training timelapse capture (2026-08-04 spec §3.1–§3.4).
//
// Why a **capture-only SplatRenderer** (spec §3.1): the sorter is per renderer and every
// render() triggers a resort against the camera pose, so alternating two cameras through one
// renderer would hand each render an index buffer sorted for the *other* camera. Gaussian
// splatting has no depth test (.always), so sort order **is** alpha compositing order — the
// occlusion collapses instantly. Worse, the fork's "skip the resort if the move is under 2.56°"
// threshold would never hold, and the sort loop would spin nonstop and contend with training.
// SplatChunk's fields are MetalBuffer (a class), so the splat and SH storage is **shared, not
// copied** — the extra cost is the sort structures (≈240MB at 5M splats), which is why the
// renderer is created lazily on the first capture instead of at start.
//
// Why the capture is **inline in the streamer's swap pipeline** (spec §3.2): LiveSplatStreamer
// alternates two ChunkBuffers sets under the invariant "set N is rewritten only after the
// previous swap's removeChunk has finished". A second consumer would break it. Running the
// capture inside the serial pipeline means the next encode cannot start, so the invariant holds
// **structurally** — no third buffer set and no second encode.
//
// Thread contract: captureIfDue is called only from that pipeline, which is serial, so it never
// reenters. The pose, the priming flag, the ledger, the capture target, and the failure counters
// are shared with the MainActor (the reset button, the observed properties) and are **all**
// guarded by `lock`; the file writes are ordered by the serial `writeQueue`. FIX 2 closed the two
// holes in that statement: the capture target used to be read without the lock, and the pose and
// the priming flag used to be read in two separate critical sections around an `await`.
import Foundation
import Metal
import CoreGraphics
import CoreText
import ImageIO
import UniformTypeIdentifiers
import simd
import os
import MetalSplatter

// MARK: - Frame ledger (pure — the test surface)

/// Index assignment and the reset generation for the captured frames.
/// A generation exists because "shoot from here" (spec §3.3) deletes every frame collected so
/// far and restarts at 0 **while a capture may be in flight** — a frame reserved before the
/// reset must never land on disk afterwards.
struct TimelapseFrameLedger: Equatable {
    private(set) var generation = 0
    private(set) var nextIndex = 0
    private(set) var savedCount = 0

    /// 다음 캡처가 due가 되는 iteration. 모듈로만으로는 부족하다 — SnapshotMailbox가 미소비
    /// 스냅샷을 덮어쓰므로(LiveSplatStreamer.swift:88-98 "one latest pending") everySteps의
    /// 정확한 배수가 파이프라인에 도달하지 못할 수 있고, 그러면 `iter % everySteps == 0`은
    /// 그 캡처 지점을 연기가 아니라 **소멸**시킨다(다음 기회는 한 주기 뒤).
    /// 임계값 방식은 "due 지점 이후 첫 스왑"에서 캡처하므로 주기당 1장이 보장된다.
    /// 0 = 아직 무장 전(첫 claim이 everySteps로 무장한다 — 레코더가 주기를 소유하고
    /// 레저는 값 타입이라 init 시그니처를 바꾸지 않기 위한 설계).
    private(set) var nextCaptureIter = 0

    /// 이 스왑이 캡처 지점인지 판정하고, 맞으면 임계값을 다음 주기로 전진시킨다.
    /// everySteps는 >= 1 이어야 한다(호출측이 max(_, 1)로 넘긴다).
    ///
    /// 전진값은 **격자 기준**(`((iter / everySteps) + 1) * everySteps`)이지 `iter + everySteps`가
    /// 아니다. 드롭 복구 직후에는 이 때문에 간격이 한 번 짧아진다(everySteps 100에서
    /// 250에 캡처하면 다음 임계값이 350이 아니라 300 — 실제 간격 50).
    /// **의도한 트레이드오프다.** `iter + everySteps`로 "고치면" 간격은 균일해지지만 임계값이
    /// 격자에서 영구히 이탈해(250→350→450…) 이후 모든 캡처가 사용자가 고른 주기의 배수
    /// 지점에서 벗어난다. 드물게 한 칸 짧은 간격 한 번 < 영구 위상 이탈이므로 격자 정렬을 택했다.
    mutating func claim(iter: Int, everySteps: Int) -> Bool {
        if nextCaptureIter == 0 { nextCaptureIter = everySteps }
        guard iter >= nextCaptureIter else { return false }
        nextCaptureIter = ((iter / everySteps) + 1) * everySteps
        return true
    }

    /// Reserves the index of a capture that is about to start, together with its generation.
    /// The index is consumed even if the capture later fails — the encoder lists the directory
    /// and sorts by name, so a gap is harmless.
    mutating func reserve() -> (index: Int, generation: Int) {
        defer { nextIndex += 1 }
        return (nextIndex, generation)
    }

    /// 쓰기 **허가** 판정(비변경). false면 reserve 이후 리셋이 있었다는 뜻이라 그 프레임은 버린다.
    /// FIX 2 (I-3): 허가와 집계를 분리했다 — 예전에는 `commit`이 허가 시점에 `savedCount`를 올려
    /// **저장된 수가 아니라 허가된 수**를 세었고, 디스크 쓰기가 전부 실패해도 카운트가 올라가
    /// Task 14의 "2장 미만이면 인코딩 스킵" 판정이 헛돌았다.
    func permits(generation: Int) -> Bool { generation == self.generation }

    /// 실제로 파일이 디스크에 쓰인 뒤에만 호출한다. 쓰기 도중 리셋이 들어왔으면(세대 불일치)
    /// 집계하지 않는다 — 그 파일은 리셋의 삭제가 같은 직렬 writeQueue에서 걷어간다.
    mutating func commit(generation: Int) -> Bool {
        guard generation == self.generation else { return false }
        savedCount += 1
        return true
    }

    /// The reset button: a new generation, indices restart at 0, and the saved count is zeroed.
    mutating func reset() {
        generation += 1
        nextIndex = 0
        savedCount = 0
        nextCaptureIter = 0   // 다음 claim이 현재 everySteps로 재무장한다
    }
}

// MARK: - Recorder

final class TimelapseRecorder: @unchecked Sendable {
    struct Config {
        var settings: TimelapseSettings
        var width: Int
        var height: Int
        /// The run's iteration total — the denominator of the "iter N / M" label.
        var totalIterations: Int
        /// runs/<stamp>/timelapse/
        var frameDirectory: URL
    }

    let config: Config

    private static let log = Logger(subsystem: "com.postmedia.splatx-metal-trainer",
                                    category: "timelapse")

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue?

    /// The capture-only renderer and its offscreen target, kept together so the pair is published
    /// in **one** atomic store — a half-built state (renderer set, texture nil) can never be
    /// observed, and `hasCaptureRenderer` cannot disagree with what a capture actually has.
    private struct CaptureTarget {
        let renderer: SplatRenderer
        let texture: MTLTexture
    }

    /// Created lazily on the **first capture attempt** — the first swap that arrives while the
    /// timelapse is enabled (spec §3.1: allocated only when the timelapse is on, never at start).
    /// Written only from captureIfDue (the serial swap pipeline never reenters), but **read from
    /// other threads too** (`hasCaptureRenderer` on the MainActor / in tests), so every access
    /// goes through `lock` — FIX 2 (I-2). An unsynchronized read here was the one field the
    /// `@unchecked Sendable` justification did not cover.
    private var target: CaptureTarget?

    private let lock = NSLock()
    private var ledger = TimelapseFrameLedger()
    private var pose: CameraPose

    /// One-shot log gates + the observable failure count (FIX 2 / I-3 — spec §3.4 forbids silent
    /// truncation). Repeating the same log on every swap would be noise, so each kind is logged
    /// once; `failedFrameCount` stays available for the completion summary.
    private var loggedSetupFailure = false
    private var loggedWriteFailure = false
    private var failedWrites = 0

    /// The reset-sweep counterpart of `loggedWriteFailure`/`failedWrites` (FIX 3 — spec §3.4).
    /// Kept as a **separate** counter rather than folded into `failedWrites`: a write failure just
    /// means one frame is missing, but a delete failure leaves a *stale-generation* frame on disk
    /// that the encoder will splice in next to the new generation's frames — the exact
    /// "viewpoint teleports mid-video" symptom "shoot from here" exists to prevent. Keeping the
    /// counters apart lets a future caller treat the two severities differently instead of
    /// collapsing them into one number that cannot tell "fewer frames" from "corrupted video".
    private var loggedDeleteFailure = false
    private var failedDeletes = 0

    /// MetalSplatter는 정렬기의 카메라 포즈를 `render()` 안에서만 갱신한다
    /// (`SplatRenderer.swift:831-832` `sorter.updateCameraPose`; `sorter`는 `:175`에서
    /// private). 새로 만든 캡처 렌더러의 정렬기 포즈는 원점·−Z이므로
    /// (`SplatSorter.swift:106`), 한 번이라도 render를 태우기 전까지는 캡처 카메라 기준
    /// 정렬이 존재하지 않는다. true면 다음 캡처가 프라이밍 render를 선행한다.
    private var poseDirty = true

    /// The serial queue that composites the label and writes the JPEG. Serial is load bearing:
    /// the reset's file deletion is enqueued here too, so it is ordered against the writes.
    private let writeQueue = DispatchQueue(label: "com.postmedia.splatx.timelapse-write",
                                           qos: .utility)

    init(device: MTLDevice, config: Config, pose: CameraPose) {
        self.device = device
        self.config = config
        self.pose = pose
        self.commandQueue = device.makeCommandQueue()
    }

    /// The config derived from a run's inputs (pure — the test surface of the wiring; building
    /// the recorder itself needs a Metal device).
    ///
    /// `aspectWidth/Height` carry **a ratio only** — `frameSize` consumes them as a proportion, so
    /// pt and px are equally valid here. Which source that ratio came from is decided one level up
    /// by `TimelapseSettings.aspectPair` (2026-08-05 spec §2.2/§2.3), which is why the arguments no
    /// longer say "source".
    ///
    /// 2026-08-04 이 자리의 주석은 "these are the **first** COLMAP camera's intrinsics … so the
    /// timelapse frame has the same aspect ratio as the training images"라고 단언했다. **더 이상
    /// 무조건 참이 아니다** — 기본값 `.viewport`에서는 비율이 Train 탭 뷰포트에서 오고, 첫
    /// 카메라에서 오는 것은 이제 포즈뿐이다. 종횡비가 어디서 오든 수직 화각은 동일하다(spec
    /// §1.1). 양쪽 소스 모두 크기가 없으면(0) frameSize 안의 16:9 폴백으로 떨어진다.
    static func makeConfig(settings: TimelapseSettings, runDirectory: URL,
                           totalIterations: Int,
                           aspectWidth: Int, aspectHeight: Int) -> Config {
        let size = TimelapseSettings.frameSize(shortSide: settings.shortSide,
                                               sourceWidth: aspectWidth,
                                               sourceHeight: aspectHeight)
        return Config(settings: settings, width: size.width, height: size.height,
                      totalIterations: totalIterations,
                      frameDirectory: runDirectory.appendingPathComponent("timelapse",
                                                                          isDirectory: true))
    }

    // MARK: Observed state

    /// Frames actually written to disk (FIX 2 / I-3 — no longer "frames permitted to write").
    var savedFrameCount: Int { lock.withLock { ledger.savedCount } }
    var nextFrameIndex: Int { lock.withLock { ledger.nextIndex } }
    var currentPose: CameraPose { lock.withLock { pose } }
    /// Captures that were rendered and read back but whose JPEG write failed (spec §3.4: the
    /// completion summary must not report a silent shortfall as success).
    var failedFrameCount: Int { lock.withLock { failedWrites } }
    /// Stale-generation frame files that `resetFrames()`'s sweep could not delete (FIX 3 — spec
    /// §3.4 symmetry with `failedFrameCount`). Nonzero means a leftover frame from before "shoot
    /// from here" may still be sitting in the frame directory for the encoder to pick up.
    var failedFrameDeleteCount: Int { lock.withLock { failedDeletes } }

    /// Whether creating the capture target ever failed (FIX 7). The other two counters report a
    /// shortfall of a few frames; this one means **every** capture returned early, so the run ends
    /// with zero frames. Without it the completion summary says only "0 frame(s) captured" and
    /// cannot tell "training was too short" from "the capture renderer could not be created" — the
    /// last silent failure left in this class, and the one spec §3.4 forbids most clearly.
    var captureSetupFailed: Bool { lock.withLock { loggedSetupFailure } }
    /// Whether the capture-only renderer has been allocated yet (the laziness contract).
    var hasCaptureRenderer: Bool { lock.withLock { target != nil } }

    /// Whether the next capture must run a priming render (see `poseDirty`). Exposed so the
    /// pose/priming interlock is testable without a GPU.
    var needsPriming: Bool { lock.withLock { poseDirty } }

    /// Clears the priming flag after a priming render — but **only if the pose is still the one
    /// that was primed** (FIX 2 / I-1). `captureIfDue` suspends at `await addChunk` between
    /// reading the pose and finishing the priming, and Task 12's "shoot from here" can land in
    /// that window. Clearing unconditionally would consume the *new* pose's priming with the
    /// *old* pose's render: the next capture would then see `poseDirty == false` with the sorter
    /// still ordered for the old camera, and since `depthFormat: .invalid` makes sort order the
    /// alpha compositing order, the very first frame of the restarted timelapse would show
    /// collapsed occlusion — the exact failure mode the priming exists to prevent.
    func finishPriming(primedWith pose: CameraPose) {
        lock.withLock { if self.pose == pose { poseDirty = false } }
    }

    /// A barrier on the write queue — the encoder calls it before listing the directory.
    func waitForPendingWrites() {
        writeQueue.sync {}
    }

    // MARK: "Shoot from here" (spec §3.3)

    /// Replaces the capture anchor with the pose currently on screen.
    func setPose(_ pose: CameraPose) {
        lock.withLock {
            self.pose = pose
            self.poseDirty = true   // 정렬기는 아직 직전 캡처의 포즈를 들고 있다
        }
    }

    /// Discards every frame collected so far and restarts from index 0 — so a state where the
    /// viewpoint teleports mid-video cannot exist (which is why there is no "segment" concept).
    /// The ordering against an in-flight capture is what makes this safe:
    ///   ① the generation is bumped **synchronously** here, so any capture that already reserved
    ///      an index fails its commit and never writes;
    ///   ② the deletion is enqueued on the same serial writeQueue as the writes, so a frame
    ///      written just before this call is ordered *before* the deletion and gets removed.
    /// Only frame-*.jpg files are touched — the folder holds nothing else, but being explicit
    /// keeps a hand-placed file from being destroyed.
    /// A per-file delete failure is recorded through `noteDeleteFailure` (FIX 3) instead of being
    /// swallowed by `try?` — a stale-generation frame left behind here is not just a missing
    /// frame, it is a frame the encoder will splice into the *new* generation's video.
    func resetFrames() {
        lock.withLock { ledger.reset() }
        let directory = config.frameDirectory
        writeQueue.async { [self] in
            let fm = FileManager.default
            let items = (try? fm.contentsOfDirectory(at: directory,
                                                     includingPropertiesForKeys: nil)) ?? []
            for url in items where url.lastPathComponent.hasPrefix("frame-")
                && url.pathExtension.lowercased() == "jpg" {
                do {
                    try fm.removeItem(at: url)
                } catch {
                    noteDeleteFailure(url: url, error: error)
                }
            }
        }
    }

    // MARK: Capture (spec §3.2, the six steps)

    /// One capture. Called **only** from LiveSplatStreamer's swap pipeline, right after the swap.
    func captureIfDue(chunk: SplatChunk, iter: Int) async {
        guard config.settings.enabled, iter > 0 else { return }
        // 렌더러·큐·텍스처 가드가 claim **앞**에 온다. 뒤에 두면 이 가드가 실패했을 때 claim이
        // 이미 nextCaptureIter를 전진시킨 뒤라 그 주기의 캡처가 연기가 아니라 **소멸**한다 —
        // claim이 내세운 "주기당 1장 보장"과 정면으로 모순이다. 앞에 두면 실패한 스왑은
        // 임계값을 건드리지 않고 지나가고, 다음 스왑이 같은 주기를 다시 시도한다.
        // 지연 생성 규약(spec §3.1 — "타임랩스가 켜졌고 첫 캡처가 발생할 때만 생성")은 위의
        // `config.settings.enabled` 가드가 이미 보장한다: 꺼져 있으면 여기까지 오지 않고,
        // 켜져 있으면 이 호출 자체가 첫 캡처 시도다.
        guard let commandQueue, let target = ensureTarget() else { return }
        let renderer = target.renderer, texture = target.texture
        // due 판정과 인덱스 예약은 한 번의 lock 안에서 원자적으로. 메일박스가 미소비 스냅샷을
        // 드롭하므로 due 조건은 "everySteps의 정확한 배수"가 아니라 "임계값 이후 첫 스왑"이다
        // (spec §3.7 본문은 끊김 없는 50 격자를 전제하지만 메일박스는 그것을 보장하지 않는다
        //  — 해당 절의 2026-08-05 각주가 정정한다).
        let every = max(config.settings.everySteps, 1)
        guard let reservation: (index: Int, generation: Int) = lock.withLock({
            ledger.claim(iter: iter, everySteps: every) ? ledger.reserve() : nil
        }) else { return }
        // 포즈와 프라이밍 플래그는 **한 임계 구역에서 함께** 읽는다(FIX 2 / I-1). 나눠 읽으면
        // 아래 `await addChunk`의 서스펜션에 setPose가 끼어들어 "P1로 프라이밍하고 P2의
        // 프라이밍 요구를 소진"하는 상태가 만들어진다 — finishPriming의 주석 참조.
        let (pose, needsPriming) = lock.withLock { (self.pose, self.poseDirty) }

        // ① Add the same chunk. sortByLocality: false is **mandatory** — the default (true)
        //    reorders the shared splat buffer in place, which invalidates the on-screen
        //    renderer's index buffer.
        _ = await renderer.addChunk(chunk, sortByLocality: false)

        // 뷰포트는 ①-b 프라이밍과 ③이 함께 쓰므로 여기서 한 번만 만든다.
        let viewport = SplatRenderer.ViewportDescriptor(
            viewport: MTLViewport(originX: 0, originY: 0,
                                  width: Double(config.width), height: Double(config.height),
                                  znear: 0, zfar: 1),
            projectionMatrix: CameraController.perspective(
                fovY: pose.fov,
                aspect: Float(config.width) / Float(max(config.height, 1)),
                near: 0.01, far: 1000),
            viewMatrix: pose.viewMatrix,
            screenSize: SIMD2<Int>(config.width, config.height))

        // ①-b 포즈 프라이밍. 정렬기 포즈를 갱신하는 유일한 지점이 render()이므로
        //     (SplatRenderer.swift:831-832), 프라이밍 없이 ②를 기다리면 그 정렬은
        //     원점·−Z 기준으로 돌고(SplatSorter.swift:106 초기값), ③의 updateCameraPose는
        //     needsSort만 세울 뿐 버퍼를 무효화하지 않아(:360-372) 바로 뒤의
        //     tryObtainSortedIndices(:422)가 그 원점 정렬본을 그대로 돌려준다.
        //     depthFormat .invalid라 depth 테스트가 없으니 정렬 순서 = 알파 합성 순서 —
        //     프레임 0에 가림 붕괴가 그대로 찍힌다(spec §3.1이 막으려던 실패 모드).
        //     결과를 버리는 render 1회면 충분하다: updateCameraPose가 인덱스 버퍼 guard보다
        //     앞이라 false 반환이어도 포즈는 갱신되고, addChunkToSort가
        //     writeSequentialIndicesToFreeBuffer(SplatSorter.swift:229)로 즉시 valid 버퍼를
        //     만들어 두므로 대기도 없다.
        if needsPriming {
            if let warmup = commandQueue.makeCommandBuffer() {
                _ = try? renderer.render(viewports: [viewport],
                                         colorTexture: texture,
                                         colorStoreAction: .dontCare,
                                         depthTexture: nil,
                                         rasterizationRateMap: nil,
                                         renderTargetArrayLength: 0,
                                         to: warmup)
                warmup.commit()
                warmup.waitUntilCompleted()   // maxSimultaneousRenders: 1 — 인플라이트를 비운다
            }
            finishPriming(primedWith: pose)   // 그 사이 포즈가 바뀌었으면 플래그를 유지한다
        }

        // ② Wait for the sort that includes it **before** rendering. render() otherwise blocks
        //    the calling thread spinning on the sort (accessTimeout 0.1 + sortTimeout 0.1 =
        //    up to 0.2s of Thread.sleep); waiting here avoids that entirely.
        //    ①-b가 포즈를 태웠으므로 poseRequiresResort(SplatSorter.swift:348)가 needsSort를
        //    다시 세우고, 여기서 기다리는 정렬은 캡처 카메라 기준이다. 정렬이 이미 끝나 있었
        //    더라도 addSortCompletionHandler(:474-490)가 idle일 때 needsSort를 재무장하므로
        //    핸들러는 반드시 발화한다.
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            renderer.afterNextSort { cont.resume() }
        }

        // ③ Offscreen render → commit → wait.
        var rendered = false
        if let cb = commandQueue.makeCommandBuffer() {
            // depthTexture nil is correct: depthFormat .invalid makes writeDepth false, so the
            // render pass has no depth attachment at all (SplatRenderer.renderEncoder).
            rendered = (try? renderer.render(viewports: [viewport],
                                             colorTexture: texture,
                                             colorStoreAction: .store,
                                             depthTexture: nil,
                                             rasterizationRateMap: nil,
                                             renderTargetArrayLength: 0,
                                             to: cb)) ?? false
            cb.commit()
            cb.waitUntilCompleted()
        }

        // ④ Read back. The texture is .storageModeShared, so getBytes needs no blit.
        var bytes: [UInt8]?
        if rendered {
            var buffer = [UInt8](repeating: 0, count: config.width * config.height * 4)
            buffer.withUnsafeMutableBytes { raw in
                guard let base = raw.baseAddress else { return }
                texture.getBytes(base, bytesPerRow: config.width * 4,
                                 from: MTLRegionMake2D(0, 0, config.width, config.height),
                                 mipmapLevel: 0)
            }
            bytes = buffer
        }

        // ⑤ Release the chunk immediately — the streamer may rewrite this buffer set on the next
        //    swap, and holding a reference past this point is exactly what would break its
        //    two-set invariant. It also lets the capture sorter's loop exit
        //    (SplatSorter: "shouldExit = !needsSort && chunks.isEmpty"), so it does not keep
        //    spinning between captures and contending with training.
        await renderer.removeAllChunks()

        // ⑥ Hand the byte copy to the write queue (label compositing + JPEG) — asynchronous, so
        //    the pipeline resumes without waiting for the disk.
        guard let bytes else { return }
        let label = config.settings.overlay
            ? Self.overlayText(iter: iter, total: config.totalIterations)
            : nil
        let url = config.frameDirectory
            .appendingPathComponent(TimelapseSettings.frameFileName(index: reservation.index))
        let width = config.width, height = config.height
        writeQueue.async { [self] in
            // 허가 → 쓰기 → **성공했을 때만** 집계. 예전에는 허가 시점에 집계해서 쓰기가 전부
            // 실패해도 savedFrameCount가 올라갔고, 그 값으로 인코딩 여부를 정하는 Task 14가
            // 빈 디렉터리에 인코더를 돌리게 됐다(FIX 2 / I-3).
            guard lock.withLock({ ledger.permits(generation: reservation.generation) }) else {
                return   // a reset happened after the reserve — drop it
            }
            guard Self.writeFrame(bgra: bytes, width: width, height: height,
                                  url: url, label: label) else {
                noteWriteFailure(url: url)
                return
            }
            lock.withLock { _ = ledger.commit(generation: reservation.generation) }
        }
    }

    /// Records a failed frame write and logs the **first** one (spec §3.4: no silent shortfall).
    /// Repeating it every swap would bury the console — the count carries the rest.
    private func noteWriteFailure(url: URL) {
        let first = lock.withLock { () -> Bool in
            failedWrites += 1
            defer { loggedWriteFailure = true }
            return !loggedWriteFailure
        }
        guard first else { return }
        // 이후 실패는 failedFrameCount로만 센다(스왑마다 같은 줄을 찍으면 콘솔이 묻힌다).
        Self.log.error("timelapse: frame write failed at \(url.path, privacy: .public) — the frame directory may be missing or unwritable")
    }

    /// Records a failed frame **deletion** during `resetFrames()`'s sweep and logs the **first**
    /// one (FIX 3 — spec §3.4: symmetric with `noteWriteFailure`). A file this leaves behind is a
    /// stale-generation frame sitting next to the new generation's frames, so it is a more severe
    /// failure than a missed write: the encoder cannot tell it apart from a fresh frame and will
    /// splice it into the video, reproducing the mid-video viewpoint jump "shoot from here" exists
    /// to prevent.
    private func noteDeleteFailure(url: URL, error: Error) {
        let first = lock.withLock { () -> Bool in
            failedDeletes += 1
            defer { loggedDeleteFailure = true }
            return !loggedDeleteFailure
        }
        guard first else { return }
        // 이후 실패는 failedFrameDeleteCount로만 센다.
        Self.log.error("timelapse: could not delete stale frame at \(url.path, privacy: .public) during reset (\(error.localizedDescription, privacy: .public)) — it may survive into the encoded video")
    }

    /// Creates the capture-only renderer and the offscreen target on first use.
    /// - `depthFormat: .invalid` forces the single-stage path (no depth texture needed).
    /// - `colorFormat: .bgra8Unorm_srgb` is the same colour space as the screen and as Cinema.
    /// - `maxSimultaneousRenders: 1` — the capture is serial (every render waits for completion).
    /// - An opaque black clearColor: the JPEG has no alpha channel, so a transparent background
    ///   would be recorded as arbitrary premultiplied values.
    private func ensureTarget() -> CaptureTarget? {
        if let existing = lock.withLock({ target }) { return existing }
        let created: SplatRenderer
        do {
            created = try SplatRenderer(
                device: device,
                colorFormat: .bgra8Unorm_srgb,
                depthFormat: .invalid,
                sampleCount: 1,
                maxViewCount: 1,
                maxSimultaneousRenders: 1,
                clearColor: MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1))
        } catch {
            noteSetupFailure("capture renderer: \(error)")
            return nil
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm_srgb, width: config.width, height: config.height,
            mipmapped: false)
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared   // CPU-readable without a blit
        guard let created2D = device.makeTexture(descriptor: descriptor) else {
            noteSetupFailure("offscreen texture \(config.width)×\(config.height)")
            return nil
        }
        let made = CaptureTarget(renderer: created, texture: created2D)
        lock.withLock { target = made }
        return made
    }

    /// Logs the first capture-target setup failure (spec §3.4). Without this the timelapse is on,
    /// every swap returns silently, and the run ends with zero frames and no explanation.
    private func noteSetupFailure(_ what: String) {
        let first = lock.withLock { () -> Bool in
            defer { loggedSetupFailure = true }
            return !loggedSetupFailure
        }
        guard first else { return }
        Self.log.error("timelapse: could not create the \(what, privacy: .public) — capture produces no frames for this run")
    }

    // MARK: Frame writing (pure — the test surface)

    /// The overlay label text (spec §3.4).
    static func overlayText(iter: Int, total: Int) -> String {
        total > 0 ? "iter \(iter) / \(total)" : "iter \(iter)"
    }

    /// BGRA8 (sRGB) bytes → an optional baked-in label → JPEG at quality 0.95.
    /// The label is drawn **here, at capture time** rather than at encode time (spec §3.4):
    /// VideoExporter's OverlayCompositor is a static-image struct (image/rect/opacity) and cannot
    /// carry per-frame text, and baking here leaves the label in the saved JPEG too.
    /// The bitmap layout matches VideoExporter.append: premultipliedFirst + byteOrder32Little
    /// = BGRA in memory, with memory row 0 as the topmost scanline (CG's origin is bottom-left,
    /// so the label drawn at y = pad lands in the bottom-left corner of the image).
    /// Returns whether the JPEG actually reached disk (FIX 2 / I-3). Every failure here used to be
    /// a bare `return`, so a missing frame directory made **every** frame evaporate with the run
    /// still reporting success. The caller counts and logs failures; the ledger only counts writes
    /// that returned true.
    @discardableResult
    static func writeFrame(bgra: [UInt8], width: Int, height: Int, url: URL,
                           label: String?) -> Bool {
        var pixels = bgra
        var image: CGImage?
        pixels.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress,
                  let ctx = CGContext(data: base, width: width, height: height,
                                      bitsPerComponent: 8, bytesPerRow: width * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)
                                          ?? CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                          | CGBitmapInfo.byteOrder32Little.rawValue)
            else { return }
            if let label { drawLabel(label, in: ctx, height: height) }
            image = ctx.makeImage()
        }
        // Written to a sibling temp name and moved into place (FIX 7). CGImageDestination writes
        // incrementally, so encoding straight to `url` leaves a window where the file exists at
        // its final name but is only partly written — and the encoder finds frames by listing the
        // directory. One half-written frame fails `decodeFailed`, which discards the **whole**
        // timelapse. The rename is atomic within a volume, so a listing sees the frame either
        // fully or not at all. The dot prefix also keeps the partial out of the `frame-` sweep
        // that `resetFrames` and the encoder both match on.
        guard let image else { return false }
        let temp = url.deletingLastPathComponent()
            .appending(path: ".\(url.lastPathComponent).partial")
        guard let destination = CGImageDestinationCreateWithURL(
                temp as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else { return false }
        CGImageDestinationAddImage(
            destination, image,
            [kCGImageDestinationLossyCompressionQuality: 0.95] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            try? FileManager.default.removeItem(at: temp)
            return false
        }
        do {
            try? FileManager.default.removeItem(at: url)   // a retry of the same index
            try FileManager.default.moveItem(at: temp, to: url)
            return true
        } catch {
            try? FileManager.default.removeItem(at: temp)
            return false
        }
    }

    /// Draws the label with CoreText (no AppKit — this runs on the write queue, off the main
    /// thread) over a translucent plate so it stays readable on a bright scene.
    private static func drawLabel(_ text: String, in ctx: CGContext, height: Int) {
        let size = max(CGFloat(height) / 36, 12)
        let font = CTFontCreateWithName("HelveticaNeue-Bold" as CFString, size, nil)
        let attributes: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String):
                CGColor(red: 1, green: 1, blue: 1, alpha: 0.95),
        ]
        let line = CTLineCreateWithAttributedString(
            NSAttributedString(string: text, attributes: attributes))
        let bounds = CTLineGetBoundsWithOptions(line, .useOpticalBounds)
        let pad = size * 0.5
        ctx.saveGState()
        ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 0.45))
        ctx.fill(CGRect(x: pad * 0.5, y: pad * 0.5,
                        width: bounds.width + pad, height: bounds.height + pad))
        ctx.textPosition = CGPoint(x: pad, y: pad * 0.5 + pad * 0.4)
        CTLineDraw(line, ctx)
        ctx.restoreGState()
    }
}
