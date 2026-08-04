// Introduced in path transform FIX 10 and kept by FIX 13: unit verification of the gizmo
// handles' **display guard and the single source of truth for their world placement**. The
// consumer is PathGizmoOverlay (one view for both drawing and input — FIX 13), and because the
// placement and the guard are pure functions outside the view body, they are pinned here. The
// appearance of the markers (the SwiftUI cross and circle) is the responsibility of the manual
// checklist — the policy of this app. (The five isCenterVisible cull-gate tests from FIX 11
// were removed along with the function when their consumer, the in-path gizmo marker, went
// back to SwiftUI in FIX 13.)
import XCTest
import simd
@testable import SplatXMetalTrainer

final class PathGizmoPlacementTests: XCTestCase {

    private let right = SIMD3<Float>(1, 0, 0)
    private let up = SIMD3<Float>(0, 1, 0)

    /// No pivot (no path, or zero keys) → there are no handles at all (the §4.3 upper guard).
    func testNoPivotYieldsNoHandles() {
        XCTAssertNil(PathGizmoPlacement.make(pivot: nil, cameraRight: right, upAxis: up))
    }

    /// meanRadius == 0 (a single-key static shot) → the center only; the radius handle is
    /// hidden (so it cannot overlap the pivot).
    func testZeroRadiusHidesRadiusHandle() {
        let pivot = PathTransform.Pivot(point: SIMD3(1, 2, 3), isAimLocked: false,
                                        meanRadius: 0)
        let placement = PathGizmoPlacement.make(pivot: pivot, cameraRight: right, upAxis: up)
        XCTAssertEqual(placement?.center, SIMD3(1, 2, 3))
        XCTAssertNil(placement?.radius)
    }

    /// The radius handle = pivot + û·R̄ — the direction comes from handleDirection (the
    /// orbit-plane projection) and the distance is exactly R̄.
    func testRadiusHandleIsPivotPlusHandleDirection() throws {
        let pivot = PathTransform.Pivot(point: SIMD3(0, 1, 0), isAimLocked: true,
                                        meanRadius: 4)
        let placement = try XCTUnwrap(
            PathGizmoPlacement.make(pivot: pivot, cameraRight: right, upAxis: up))
        let handle = try XCTUnwrap(placement.radius)
        // right=+X, up=+Y → the orbit-plane projection is +X unchanged.
        XCTAssertEqual(handle.x, 4, accuracy: 1e-5)
        XCTAssertEqual(handle.y, 1, accuracy: 1e-5)
        XCTAssertEqual(handle.z, 0, accuracy: 1e-5)
        XCTAssertEqual(simd_distance(handle, pivot.point), 4, accuracy: 1e-5)
    }

    /// Even in the near-top-view degeneracy (cameraRight ∥ upAxis — the FIX 2 history), a
    /// finite unit direction comes out and the distance stays exactly R̄ (the handle neither
    /// collapses onto the pivot nor becomes NaN).
    func testDegenerateCameraRightStillYieldsFiniteHandle() {
        let pivot = PathTransform.Pivot(point: .zero, isAimLocked: false, meanRadius: 2)
        let placement = PathGizmoPlacement.make(pivot: pivot, cameraRight: up, upAxis: up)
        let handle = placement?.radius ?? SIMD3<Float>(.nan, .nan, .nan)
        XCTAssertTrue(handle.x.isFinite && handle.y.isFinite && handle.z.isFinite)
        XCTAssertEqual(simd_distance(handle, .zero), 2, accuracy: 1e-5)
    }

    /// Extracting the camera axes: for the default camera looking at the origin from +Z,
    /// right is +X.
    /// (The fixed point of the contract that the placement and the center-handle drag basis use
    /// the same derivation.)
    func testCameraAxesRightIsScreenRight() {
        var cam = CameraController()
        cam.target = .zero; cam.distance = 5; cam.yaw = 0; cam.pitch = 0
        let axes = PathGizmoPlacement.cameraAxes(cam)
        XCTAssertEqual(axes.right.x, 1, accuracy: 1e-5)
        XCTAssertEqual(axes.right.y, 0, accuracy: 1e-5)
        XCTAssertEqual(axes.right.z, 0, accuracy: 1e-5)
        XCTAssertEqual(axes.forward.z, -1, accuracy: 1e-5)   // it looks along −Z
    }

    /// The same pivot and the same camera axes always yield **the same value** (purity — this
    /// is why the placement does not drift when the view is re-evaluated).
    func testDeterministicForSameInputs() {
        let pivot = PathTransform.Pivot(point: SIMD3(3, -1, 2), isAimLocked: true,
                                        meanRadius: 1.5)
        let a = PathGizmoPlacement.make(pivot: pivot, cameraRight: SIMD3(0.3, 0.2, -0.9),
                                        upAxis: up)
        let b = PathGizmoPlacement.make(pivot: pivot, cameraRight: SIMD3(0.3, 0.2, -0.9),
                                        upAxis: up)
        XCTAssertEqual(a, b)
    }
}
