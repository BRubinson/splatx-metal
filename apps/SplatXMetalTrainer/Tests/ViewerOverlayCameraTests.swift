import XCTest
import Metal
import simd
import SplatIO
@testable import SplatXMetalTrainer

/// The overlay camera selection rule (path transform spec footnote 3 — synchronizing with
/// the presented frame) and the **ownership decomposition** of the presentation notification
/// (path transform FIX 17 — spec footnote 16):
///   · ordering (the sequence) = the producer, `SplatRendererHost`
///   · origin (identity) = the receiver, `ViewerModel`
///   · wiring and teardown = the single didSet on `ViewerModel.host`
/// The GPU render loop (draw) itself and the SwiftUI consumers are the responsibility of the
/// manual checklist under this app's policy, but a host instance can be created as-is in the
/// test bundle (0.004 s), so the wiring discipline is verified against the real thing.
@MainActor
final class ViewerOverlayCameraTests: XCTestCase {
    private static func makeHost() throws -> SplatRendererHost {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        return try SplatRendererHost(device: device)
    }

    /// With no presentedCamera (the initial state) → overlayCamera falls back to the live
    /// camera.
    func testOverlayCameraFallsBackToLiveCamera() {
        let vm = ViewerModel()
        vm.camera.yaw = 1.25
        vm.camera.distance = 9
        XCTAssertNil(vm.presentedCamera)
        XCTAssertEqual(vm.overlayCamera, vm.camera)
    }

    /// When a presentedCamera exists it is returned and the live camera is untouched — the
    /// contract of the dual read: the overlay uses the presented frame, scene submission uses
    /// the live camera.
    func testOverlayCameraPrefersPresentedCamera() throws {
        let vm = ViewerModel()
        let host = try Self.makeHost()
        vm.host = host
        vm.camera.yaw = 1.25
        var shown = vm.camera
        shown.yaw = 0.5                     // the presented frame is the camera from a few ticks ago
        vm.notePresentedCamera(shown, from: host)
        XCTAssertEqual(vm.overlayCamera, shown)
        XCTAssertEqual(vm.camera.yaw, 1.25)   // the live state is not polluted
    }

    /// A no-scene notification (nil — the isReadyToRender == false path in
    /// SplatRendererHost.draw) clears presentedCamera, so the overlay returns to the live
    /// camera.
    func testNilPresentationRestoresLiveCamera() throws {
        let vm = ViewerModel()
        let host = try Self.makeHost()
        vm.host = host
        var shown = CameraController()
        shown.pitch = -0.4
        vm.notePresentedCamera(shown, from: host)
        XCTAssertNotNil(vm.presentedCamera)
        vm.notePresentedCamera(nil, from: host)
        XCTAssertNil(vm.presentedCamera)
        XCTAssertEqual(vm.overlayCamera, vm.camera)
    }

    /// clear() (destroying the host) also invalidates presentedCamera — the presented camera
    /// of a deleted scene does not linger into the next scene's overlay.
    func testClearResetsPresentedCamera() throws {
        let vm = ViewerModel()
        let host = try Self.makeHost()
        vm.host = host
        vm.notePresentedCamera(CameraController(), from: host)
        XCTAssertNotNil(vm.presentedCamera)
        vm.clear()
        XCTAssertNil(vm.presentedCamera)
        XCTAssertEqual(vm.overlayCamera, vm.camera)
    }

    // MARK: The ordering axis — owned by the producer (SplatRendererHost)

    /// The delivery decision (pure): only a sequence greater than the last delivered one
    /// passes. Equal values (duplicate delivery) and smaller values (out-of-order delivery) are
    /// dropped. The first notification (1 vs 0) is accepted.
    func testDeliversPresentedFrameIsStrictlyMonotone() {
        XCTAssertTrue(SplatRendererHost.deliversPresentedFrame(sequence: 1, lastDelivered: 0))
        XCTAssertTrue(SplatRendererHost.deliversPresentedFrame(sequence: 9, lastDelivered: 8))
        XCTAssertFalse(SplatRendererHost.deliversPresentedFrame(sequence: 8, lastDelivered: 8))
        XCTAssertFalse(SplatRendererHost.deliversPresentedFrame(sequence: 7, lastDelivered: 8))
    }

    /// With 2 frames in flight, the unstructured Task in the completion handler can deliver
    /// out of order — an **earlier** frame that arrives late never reaches the callback. If it
    /// were let through, the receiver's presentedCamera would rewind by one frame and the
    /// overlay would jump. If the decision is changed to always pass, this assertion breaks.
    func testHostDropsOutOfOrderAndDuplicateDeliveries() throws {
        let host = try Self.makeHost()
        var seen: [Float] = []
        host.onFramePresented = { _, cam in seen.append(cam?.yaw ?? -1) }
        var older = CameraController(); older.yaw = 0.2
        var newer = CameraController(); newer.yaw = 1.0
        var latest = CameraController(); latest.yaw = 1.5
        host.deliverPresented(newer, sequence: 12)
        host.deliverPresented(older, sequence: 11)    // out of order — dropped
        host.deliverPresented(nil, sequence: 12)      // duplicate sequence (no scene) — dropped
        host.deliverPresented(latest, sequence: 13)   // newest — delivered
        XCTAssertEqual(seen, [1.0, 1.5])
    }

    /// The notification's origin is the host itself — this is what lets the receiver decide
    /// by identity, and it is why the wiring closure does not need to capture the host (the
    /// host owns the callback).
    func testHostDeliversItselfAsSource() throws {
        let host = try Self.makeHost()
        var source: SplatRendererHost?
        host.onFramePresented = { src, _ in source = src }
        host.deliverPresented(CameraController(), sequence: 1)
        XCTAssertTrue(source === host)
    }

    // MARK: The origin axis — owned by the receiver (ViewerModel), decided by identity

    /// The acceptance decision (pure): **instance identity**, not value. A different
    /// producer, or no host at all, is rejected. If the decision is changed to always pass,
    /// the last two assertions break.
    func testAcceptsPresentedFrameRequiresIdenticalSource() {
        let a = NSObject(), b = NSObject()
        XCTAssertTrue(ViewerModel.acceptsPresentedFrame(source: a, currentSource: a))
        XCTAssertFalse(ViewerModel.acceptsPresentedFrame(source: a, currentSource: b))
        XCTAssertFalse(ViewerModel.acceptsPresentedFrame(source: a, currentSource: nil))
    }

    /// **The core case of this regression** (review point 3): even firing the still-live
    /// closure of a discarded host is ignored by the model. FIX 16 claimed to "preserve the
    /// wiring when it is the same instance" while bumping only the epoch, so the preserved
    /// closure held the old epoch and had every notification rejected from that moment on — a
    /// self-contradiction. Now the wiring is refreshed on every assignment and the decision is
    /// based on the live host itself, so there is no contradiction.
    func testStaleHostCallbackIsIgnoredByModel() throws {
        let vm = ViewerModel()
        let oldHost = try Self.makeHost()
        vm.host = oldHost
        let staleCallback = try XCTUnwrap(oldHost.onFramePresented)   // grabbed before the swap

        var watched = CameraController(); watched.yaw = 0.9
        vm.notePresentedCamera(watched, from: oldHost)
        XCTAssertEqual(vm.presentedCamera, watched)

        let newHost = try Self.makeHost()
        vm.host = newHost
        XCTAssertNil(oldHost.onFramePresented, "didSet severs the old host's callback")
        XCTAssertNil(vm.presentedCamera, "swapping the host invalidates the presented camera")

        // The in-flight completion Task of the discarded host drains late.
        var stale = CameraController(); stale.yaw = 0.91
        staleCallback(oldHost, stale)
        XCTAssertNil(vm.presentedCamera, "a discarded producer's notification is rejected")

        // The new host's notification is accepted normally — the old producer does not
        // pollute the baseline.
        var fresh = CameraController(); fresh.yaw = 0.1
        try XCTUnwrap(newHost.onFramePresented)(newHost, fresh)
        XCTAssertEqual(vm.presentedCamera, fresh)
    }

    // MARK: The wiring/teardown axis — owned by the single didSet on host

    /// Assigning host both wires the callback and invalidates the presentation state (a
    /// single point).
    func testHostAssignmentWiresCallbackAndResetsPresentedCamera() throws {
        let vm = ViewerModel()
        let host = try Self.makeHost()
        vm.host = host
        XCTAssertNotNil(host.onFramePresented)
        try XCTUnwrap(host.onFramePresented)(host, CameraController())
        XCTAssertNotNil(vm.presentedCamera)
        vm.host = nil
        XCTAssertNil(vm.presentedCamera)
    }

    /// **Regression from review point 1**: the path that plugs in a host without going
    /// through `open()` (TrainViewModel's acquireRenderer — live training streaming creates the
    /// host on the first snapshot) is wired too. Up to FIX 16 the wiring lived only inside
    /// open()'s `if host == nil` branch, so if the host was created through this path first, the
    /// later open() saw `host != nil` and skipped the wiring entirely — presentedCamera stayed
    /// nil forever → the overlay fell back to the live camera permanently (the original symptom
    /// in footnote 3).
    func testHostAssignedOutsideOpenIsWired() throws {
        let vm = ViewerModel()
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        vm.host = try? SplatRendererHost(device: device)   // the same shape as acquireRenderer
        let host = try XCTUnwrap(vm.host)
        var shown = CameraController(); shown.yaw = 0.42
        try XCTUnwrap(host.onFramePresented)(host, shown)
        XCTAssertEqual(vm.presentedCamera, shown)
    }

    /// Reassigning the same instance rewires without exception — only when there is no "it is
    /// still alive, so skip it" branch does the wiring discipline stay free of exceptions.
    func testSameHostReassignmentRewiresCallback() throws {
        let vm = ViewerModel()
        let host = try Self.makeHost()
        vm.host = host
        vm.host = host
        var shown = CameraController(); shown.yaw = 0.7
        try XCTUnwrap(host.onFramePresented)(host, shown)
        XCTAssertEqual(vm.presentedCamera, shown)
    }

    /// The wiring closure does not capture the host (the host owns the callback, so capturing
    /// it would be a retain cycle).
    /// Even if the model disappears without going through didSet (deinit), the host must not
    /// survive.
    func testPresentationWiringDoesNotRetainHost() throws {
        weak var weakHost: SplatRendererHost?
        func wireThenDropModel() throws {
            let vm = ViewerModel()
            vm.host = try Self.makeHost()
            weakHost = vm.host
            XCTAssertNotNil(weakHost)
        }
        try wireThenDropModel()
        XCTAssertNil(weakHost, "if the callback held the host strongly, the host would outlive the model")
    }

    // MARK: Resume-after-await safety in open() (review point 6) + abort propagation
    // (FIX 18)

    /// The idiom FIX 17 uses to observe the suspension point. open() is @MainActor, so
    /// whenever this test code runs, open() is necessarily suspended at an await — and if the
    /// host is visible and loading is on, that point can only be inside `load`. Returning false
    /// = observation failed (a race in timing) — callers filter that out with XCTSkipIf.
    private static func spinUntilSuspendedInLoad(_ vm: ViewerModel) async -> Bool {
        var spins = 0
        while !(vm.loading && vm.host != nil) && spins < 10_000 {
            await Task.yield(); spins += 1
        }
        return vm.loading && vm.host != nil
    }

    /// The scenario where the run is deleted mid-load (ProjectSession → clear() → host =
    /// nil). The previous code force-unwrapped `host!` again after `await host!.load(...)`
    /// resumed, and therefore **crashed**.
    func testOpenSurvivesHostRemovalMidLoad() async throws {
        let url = try await Self.writeSinglePointPLY()
        defer { try? FileManager.default.removeItem(at: url) }
        let vm = ViewerModel()
        let task = Task { await vm.open(SplatSource(file: url)) }
        let suspendedInLoad = await Self.spinUntilSuspendedInLoad(vm)
        if suspendedInLoad { vm.clear() }   // delete the run mid-load
        _ = await task.value                // resume — must not crash
        try XCTSkipIf(!suspendedInLoad, "could not observe the mid-load suspension point (race timing)")
        // The resumed open() does not resurrect the state that clear() emptied.
        XCTAssertNil(vm.host)
        XCTAssertNil(vm.currentSource)
        XCTAssertNil(vm.currentSplats)
        XCTAssertNil(vm.presentedCamera)
        XCTAssertNil(vm.errorMessage)
        XCTAssertFalse(vm.loading)
    }

    /// **FIX 18 (abort propagation)**: an abort is reported to the caller as a false return
    /// value — FIX 17 only returned silently, so the code after openRun/openFile saw only
    /// `errorMessage == nil` and ran anyway (the openRun test below pins that outcome). An
    /// abort is not an error (the user emptied it themselves), so errorMessage is not set
    /// either.
    func testOpenReportsAbortWhenClearedMidLoad() async throws {
        let url = try await Self.writeSinglePointPLY()
        defer { try? FileManager.default.removeItem(at: url) }
        let vm = ViewerModel()
        let task = Task { await vm.open(SplatSource(file: url)) }
        let suspendedInLoad = await Self.spinUntilSuspendedInLoad(vm)
        if suspendedInLoad { vm.clear() }
        let completed = await task.value
        try XCTSkipIf(!suspendedInLoad, "could not observe the mid-load suspension point (race timing)")
        XCTAssertFalse(completed, "an abort is reported to the caller as false")
        XCTAssertNil(vm.errorMessage, "an abort is not an error — showing one would be wrong")
        XCTAssertFalse(vm.loading)
    }

    /// **The core regression of FIX 18**: if the openRun that resumes after the run is
    /// deleted mid-load (clear()) updates lastRunID and currentRun anyway, the deleted run
    /// stays as "the run opened most recently", the next openRun is misjudged as a "switch",
    /// and the viewpoint the user had set up is blown away by a camera re-fit (a violation of
    /// spec §6). If openRun's new guard (the return value of open) is changed to always pass,
    /// this test breaks.
    func testOpenRunAbortedMidLoadSkipsStateWrites() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("sx-fix18-proj-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let record = RunRecord(id: "run-a", outputFile: "output.ply", previewFile: nil,
                               outcome: "completed", finalPSNR: nil, splatCount: 1,
                               optionsSummary: "", createdAt: Date())
        let runDir = dir.appendingPathComponent("runs/\(record.id)")
        try FileManager.default.createDirectory(at: runDir, withIntermediateDirectories: true)
        let ply = try await Self.writeSinglePointPLY()
        try FileManager.default.moveItem(
            at: ply, to: runDir.appendingPathComponent(record.outputFile))

        let vm = ViewerModel()
        let task = Task { await vm.openRun(record, projectDirectory: dir) }
        let suspendedInLoad = await Self.spinUntilSuspendedInLoad(vm)
        if suspendedInLoad { vm.clear() }   // the same intervention point as deleteRun
        await task.value
        try XCTSkipIf(!suspendedInLoad, "could not observe the mid-load suspension point (race timing)")
        XCTAssertNil(vm.lastRunID, "an aborted openRun does not record the deleted run as lastRunID")
        XCTAssertNil(vm.currentRun, "it does not resurrect the currentRun that clear() emptied")
        XCTAssertNil(vm.errorMessage)
        XCTAssertFalse(vm.loading)
    }

    /// **FIX 18 ⓑ**: the defer of an aborted old load must not turn off the spinner of a
    /// **new load** that has already started. This models the state the new load builds
    /// synchronously before it suspends (a new host mounted + loading on) and then resumes the
    /// old load — the discriminating signal is the same host identity used by the abort
    /// decision.
    func testAbortedOpenLeavesNewerLoadSpinnerOn() async throws {
        let url = try await Self.writeSinglePointPLY()
        defer { try? FileManager.default.removeItem(at: url) }
        let vm = ViewerModel()
        let task = Task { await vm.open(SplatSource(file: url)) }
        let suspendedInLoad = await Self.spinUntilSuspendedInLoad(vm)
        if suspendedInLoad {
            vm.clear()                        // after the run is deleted mid-load,
            vm.host = try Self.makeHost()     // the new load mounts a host and
            vm.loading = true                 // turns the spinner on
        }
        let completed = await task.value      // resume the old load — it must abort
        try XCTSkipIf(!suspendedInLoad, "could not observe the mid-load suspension point (race timing)")
        XCTAssertFalse(completed)
        XCTAssertTrue(vm.loading, "the old load's defer must not turn off the new load's spinner")
    }

    /// Builds the smallest actually loadable PLY (one point, SH degree 0) with the SplatIO
    /// writer — hand-writing the header would duplicate the format contract into this test.
    private static func writeSinglePointPLY() async throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("sx-fix17-\(UUID().uuidString).ply")
        let writer = try SplatPLYSceneWriter(toFileAtPath: url.path)
        try await writer.start(sphericalHarmonicDegree: 0, binary: true, pointCount: 1)
        try await writer.write([SplatPoint(position: SIMD3(0, 0, 0),
                                           color: .sphericalHarmonicFloat([SIMD3(1, 1, 1)]),
                                           opacity: .linearFloat(0.9),
                                           scale: .linearFloat(SIMD3(0.1, 0.1, 0.1)),
                                           rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1))])
        try await writer.close()
        return url
    }

    // (The three overlayEncoded signal tests from FIX 11 were deleted along with the signal
    //  itself, which FIX 13 removed once its consumer disappeared — the markers live in a
    //  SwiftUI view that is redrawn, so a ghost hit zone cannot form.)

    /// Equality comparison that includes overridePose (the Cinema skimmer pose) — the
    /// skip-the-write-when-equal in notePresentedCamera (which suppresses @Observable
    /// invalidation) relies on this synthesized Equatable.
    func testEquatableCoversOverridePose() {
        var a = CameraController()
        var b = a
        XCTAssertEqual(a, b)
        b.overridePose = CameraPose(
            position: SIMD3(1, 2, 3),
            rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), fov: 0.8)
        XCTAssertNotEqual(a, b)
        a.overridePose = b.overridePose
        XCTAssertEqual(a, b)
    }
}
