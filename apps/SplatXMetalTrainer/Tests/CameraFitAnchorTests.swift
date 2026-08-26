import XCTest
import simd
@testable import SplatXMetalTrainer

/// 2026-08-04 spec §2.3: the one-shot camera fit anchor.
/// Two contracts are pinned here. (a) `.middleOrbit` — the default — must stay **byte
/// identical** to the 2026-07-20 §6 behavior (opening a past run from the history, the
/// standalone PLY viewer). (b) `.firstCameraExact` must inject the first name-sorted camera as
/// an exact override pose on top of that fit, and must be consumed on **every** path so an
/// armed anchor can never leak into the next, unrelated open.
@MainActor
final class CameraFitAnchorTests: XCTestCase {

    /// Two identity-orientation views: sorted index 0 ("0001.jpg") sits at the origin and
    /// sorted index 1 ("0002.jpg") — the middle view of a 2-view model — sits at (0,0,−6).
    /// Both look along COLMAP +Z, so the scene up is (0,−1,0) and the centroid (0,0,4) puts the
    /// middle fit at target (0,0,4), distance 10.
    private func makeContext() -> ColmapContext {
        let first = ColmapView(id: 1, name: "0001.jpg", cameraId: 7,
                               qvec: SIMD4(1, 0, 0, 0), tvec: SIMD3(0, 0, 0))
        let middle = ColmapView(id: 2, name: "0002.jpg", cameraId: 7,
                                qvec: SIMD4(1, 0, 0, 0), tvec: SIMD3(0, 0, 6))
        let k = ColmapIntrinsics(fx: 1440, fy: 1440, cx: 960, cy: 720, width: 1920, height: 1440)
        return ColmapContext(model: ColmapModel(cameras: [7: k], views: [first, middle]),
                             middleView: middle, firstView: first,
                             pointsCentroid: SIMD3(0, 0, 4))
    }

    // MARK: - .middleOrbit — no regression

    /// The default anchor leaves overridePose nil and reproduces the existing fit numbers
    /// (target/distance/eye/fov) exactly — the contract that the history-open path is untouched.
    func testDefaultAnchorKeepsMiddleOrbitBehavior() {
        let vm = ViewerModel()
        XCTAssertEqual(vm.nextFitAnchor, .middleOrbit)
        XCTAssertTrue(vm.applyCameraFit(context: makeContext()))
        XCTAssertNil(vm.camera.overridePose)
        XCTAssertEqual(simd_length(vm.camera.target - SIMD3(0, 0, 4)), 0, accuracy: 1e-5)
        XCTAssertEqual(vm.camera.distance, 10, accuracy: 1e-5)
        XCTAssertEqual(simd_length(vm.camera.eye - SIMD3(0, 0, -6)), 0, accuracy: 1e-4)
        XCTAssertEqual(vm.camera.verticalFOV, Float(2 * atan(0.5)), accuracy: 1e-6)
        XCTAssertEqual(simd_length(vm.camera.upAxis - SIMD3<Float>(0, -1, 0)), 0, accuracy: 1e-6)
    }

    // MARK: - .firstCameraExact

    /// The first camera is injected as an exact override pose, and the viewport really renders
    /// that camera (the override's view matrix is that camera's COLMAP view matrix).
    /// The middle-camera orbit fit stays underneath so the first drag's adoptOverridePose has a
    /// sane target (spec §2.3).
    func testFirstCameraExactInjectsOverridePose() throws {
        let vm = ViewerModel()
        vm.nextFitAnchor = .firstCameraExact
        XCTAssertTrue(vm.applyCameraFit(context: makeContext()))
        let pose = try XCTUnwrap(vm.camera.overridePose)
        let expected = CameraConversion.pose(qvec: SIMD4(1, 0, 0, 0), tvec: SIMD3(0, 0, 0),
                                             verticalFOV: Float(2 * atan(0.5)))
        XCTAssertEqual(simd_length(pose.position - expected.position), 0, accuracy: 1e-5)
        XCTAssertEqual(simd_length(pose.rotation.vector - expected.rotation.vector), 0,
                       accuracy: 1e-5)
        XCTAssertEqual(pose.fov, expected.fov, accuracy: 1e-6)
        let colmap = CameraConversion.viewMatrix(qvec: SIMD4(1, 0, 0, 0), tvec: SIMD3(0, 0, 0))
        let v = pose.viewMatrix
        for c in 0..<4 {
            for r in 0..<4 {
                XCTAssertEqual(v[c][r], colmap[c][r], accuracy: 1e-5, "[\(c)][\(r)]")
            }
        }
        // The orbit state underneath is still the middle fit.
        XCTAssertEqual(simd_length(vm.camera.target - SIMD3(0, 0, 4)), 0, accuracy: 1e-5)
        XCTAssertEqual(vm.camera.distance, 10, accuracy: 1e-5)
    }

    /// One-shot: after a single fit the anchor is back to .middleOrbit, so the **next** fit is
    /// the previous behavior again.
    func testAnchorIsConsumedAfterOneFit() {
        let vm = ViewerModel()
        vm.nextFitAnchor = .firstCameraExact
        XCTAssertTrue(vm.applyCameraFit(context: makeContext()))
        XCTAssertEqual(vm.nextFitAnchor, .middleOrbit)
        XCTAssertTrue(vm.applyCameraFit(context: makeContext()))
        XCTAssertNil(vm.camera.overridePose)
    }

    /// The consumption happens **before** the context guard: a degenerate context (nil, or no
    /// points3D centroid) must not leave the anchor armed for the next, unrelated open.
    func testAnchorIsConsumedEvenWhenContextIsUnusable() {
        let vm = ViewerModel()
        vm.nextFitAnchor = .firstCameraExact
        XCTAssertFalse(vm.applyCameraFit(context: nil))
        XCTAssertEqual(vm.nextFitAnchor, .middleOrbit)
        XCTAssertNil(vm.camera.overridePose)

        let view = ColmapView(id: 1, name: "a.jpg", cameraId: 7,
                              qvec: SIMD4(1, 0, 0, 0), tvec: SIMD3(0, 0, 0))
        let noPoints = ColmapContext(model: ColmapModel(cameras: [:], views: [view]),
                                     middleView: view, firstView: view, pointsCentroid: nil)
        vm.nextFitAnchor = .firstCameraExact
        XCTAssertFalse(vm.applyCameraFit(context: noPoints))
        XCTAssertEqual(vm.nextFitAnchor, .middleOrbit)
        XCTAssertNil(vm.camera.overridePose)
    }

    /// The ordering contract (spec §2.3). open()'s frameToFit is a **system setting** and
    /// clears overridePose (CameraController.frameToFit), so the first-camera pose survives
    /// only when applyCameraFit runs **after** it — which is the order openRun already uses
    /// (ViewerView.swift: frameToFit inside open(), then applyCameraFit at the end of openRun).
    /// The reverse order is pinned as losing the pose so the two calls can never be swapped.
    func testFrameToFitBeforeFitKeepsPoseAndAfterWouldClearIt() {
        let box = BoundingBox(center: SIMD3(0, 0, 4), radius: 3)
        let real = ViewerModel()
        real.camera.frameToFit(box)                       // open()
        real.nextFitAnchor = .firstCameraExact
        XCTAssertTrue(real.applyCameraFit(context: makeContext()))   // openRun tail
        XCTAssertNotNil(real.camera.overridePose)

        let swapped = ViewerModel()
        swapped.nextFitAnchor = .firstCameraExact
        XCTAssertTrue(swapped.applyCameraFit(context: makeContext()))
        swapped.camera.frameToFit(box)
        XCTAssertNil(swapped.camera.overridePose)
    }

    /// An openRun that never reaches applyCameraFit (here: the output file does not exist, so
    /// open() sets errorMessage and openRun returns early) must still disarm the anchor — the
    /// defer at the top of openRun. Without it the user's next history open would silently
    /// inherit the first-camera anchor.
    func testFailedOpenRunDoesNotLeakArmedAnchor() async {
        let dir = FileManager.default.temporaryDirectory
            .appending(path: "sx-anchor-\(UUID().uuidString)", directoryHint: .isDirectory)
        let record = RunRecord(id: "20260804-000000", outputFile: "output.ply",
                               previewFile: nil, outcome: "completed", finalPSNR: nil,
                               splatCount: 0, optionsSummary: "-", createdAt: Date())
        let vm = ViewerModel()
        vm.nextFitAnchor = .firstCameraExact
        await vm.openRun(record, projectDirectory: dir, colmapContext: nil)
        XCTAssertNotNil(vm.errorMessage)                  // the premise: the load really failed
        XCTAssertEqual(vm.nextFitAnchor, .middleOrbit)
    }

    // MARK: - ColmapContext.firstView

    /// ColmapContext.load picks firstView from the **name-sorted** order (index 0), not the
    /// images.txt storage order — the same convention the middle camera already uses. The
    /// fixture stores the names out of order (c, a, b) on purpose, and middleView must stay at
    /// sorted index count/2 (unchanged).
    func testLoadPicksNameSortedFirstView() throws {
        let dir = FileManager.default.temporaryDirectory
            .appending(path: "sx-firstview-\(UUID().uuidString)/sparse/0")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let cameras = """
        # Camera list
        1 PINHOLE 1920 1440 1500 1510 960 720
        """
        let images = """
        # Image list
        1 1 0 0 0 0.1 -0.2 3.0 1 c.jpg
        512 340 1
        2 1 0 0 0 0.4 -0.5 6.0 1 a.jpg
        512 340 2
        3 1 0 0 0 0.7 -0.8 9.0 1 b.jpg
        512 340 3

        """
        try cameras.write(to: dir.appending(path: "cameras.txt"), atomically: true,
                          encoding: .utf8)
        try images.write(to: dir.appending(path: "images.txt"), atomically: true,
                         encoding: .utf8)
        let ctx = try XCTUnwrap(ColmapContext.load(sparse0: dir))
        XCTAssertEqual(ctx.firstView.name, "a.jpg")     // sorted index 0
        XCTAssertEqual(ctx.middleView.name, "b.jpg")    // sorted index 3/2 = 1 — unchanged
        XCTAssertNil(ctx.pointsCentroid)                // no points3D in this fixture
    }

    /// Back-compat of the three-argument construction (the four call sites written before this
    /// round): firstView defaults to middleView, so those tests keep compiling unchanged.
    func testInitDefaultsFirstViewToMiddleView() {
        let view = ColmapView(id: 1, name: "a.jpg", cameraId: 7,
                              qvec: SIMD4(1, 0, 0, 0), tvec: SIMD3(0, 0, 0))
        let ctx = ColmapContext(model: ColmapModel(cameras: [:], views: [view]),
                                middleView: view, pointsCentroid: nil)
        XCTAssertEqual(ctx.firstView.id, ctx.middleView.id)
    }
}
