import XCTest
import simd
@testable import SplatXMetalTrainer

/// ViewerModel.applyCameraFit — the middle camera's pose → OrbitFit → setOrbit, plus the FOV
/// wiring (Task 4, spec §6).
@MainActor
final class CameraFitTests: XCTestCase {
    /// An identity pose (at the origin, gazing along +Z) with a centroid of (0,0,4) →
    /// target=(0,0,4), distance=4, eye = the origin, and FOV = 2·atan(H/(2·fy)).
    func testApplyCameraFitFromMiddleView() {
        let vm = ViewerModel()
        let view = ColmapView(id: 1, name: "a.jpg", cameraId: 7,
                              qvec: SIMD4(1, 0, 0, 0), tvec: SIMD3(0, 0, 0))
        let k = ColmapIntrinsics(fx: 1440, fy: 1440, cx: 960, cy: 720, width: 1920, height: 1440)
        let ctx = ColmapContext(model: ColmapModel(cameras: [7: k], views: [view]),
                                middleView: view,
                                pointsCentroid: SIMD3(0, 0, 4))
        XCTAssertTrue(vm.applyCameraFit(context: ctx))
        XCTAssertEqual(simd_length(vm.camera.target - SIMD3(0, 0, 4)), 0, accuracy: 1e-5)
        XCTAssertEqual(vm.camera.distance, 4, accuracy: 1e-5)
        XCTAssertEqual(simd_length(vm.camera.eye), 0, accuracy: 1e-4)   // the camera position is reproduced
        XCTAssertEqual(vm.camera.verticalFOV, 2 * atan(0.5), accuracy: 1e-6)
    }

    /// A nil context, or a missing centroid → false with the camera unchanged (the caller's
    /// frameToFit fallback is kept).
    func testApplyCameraFitFallsBackWithoutCentroid() {
        let vm = ViewerModel()
        let before = vm.camera
        XCTAssertFalse(vm.applyCameraFit(context: nil))
        let view = ColmapView(id: 1, name: "a.jpg", cameraId: 7,
                              qvec: SIMD4(1, 0, 0, 0), tvec: SIMD3(0, 0, 0))
        let ctx = ColmapContext(model: ColmapModel(cameras: [:], views: [view]),
                                middleView: view,
                                pointsCentroid: nil)   // the points3D failure scenario
        XCTAssertFalse(vm.applyCameraFit(context: ctx))
        XCTAssertEqual(vm.camera.target, before.target)
        XCTAssertEqual(vm.camera.distance, before.distance)
        XCTAssertEqual(vm.camera.verticalFOV, before.verticalFOV)
    }
}
