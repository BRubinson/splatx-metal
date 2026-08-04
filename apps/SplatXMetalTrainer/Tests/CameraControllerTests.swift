import XCTest
import simd
@testable import SplatXMetalTrainer

final class CameraControllerTests: XCTestCase {
    func testEyeIsDistanceFromTarget() {
        var c = CameraController()
        c.target = .zero; c.distance = 5; c.yaw = 0; c.pitch = 0
        // The translation of viewMatrix.inverse = eye. The distance from eye to target =
        // distance.
        let eye = (c.viewMatrix.inverse.columns.3)
        let eyePos = SIMD3<Float>(eye.x, eye.y, eye.z)
        XCTAssertEqual(length(eyePos - c.target), 5, accuracy: 1e-3)
    }
    func testFrameToFitCentersTarget() {
        var c = CameraController()
        c.frameToFit(BoundingBox(center: SIMD3(1,2,3), radius: 4))
        XCTAssertEqual(c.target, SIMD3(1,2,3))
        XCTAssertGreaterThan(c.distance, 4)   // farther away than the radius
    }
    func testZoomChangesDistance() {
        var c = CameraController(); c.distance = 10
        c.zoom(0.5)
        XCTAssertEqual(c.distance, 5, accuracy: 1e-3)
    }
    // Task 4: setOrbit applies everything at once — feeding it an OrbitFit result reproduces
    // eye per the convention.
    func testSetOrbitAppliesParameters() {
        var c = CameraController()
        c.setOrbit(target: SIMD3(0, 0, -3), distance: 3, yaw: 0, pitch: 0)
        XCTAssertEqual(c.target, SIMD3(0, 0, -3))
        XCTAssertEqual(c.distance, 3)
        // dir=(cp·sy, sp, cp·cy)=(0,0,1) → eye = target + dir·3 = the origin.
        XCTAssertEqual(simd_length(c.eye), 0, accuracy: 1e-6)
        // A negative distance is guarded by the same positive floor as zoom().
        c.setOrbit(target: .zero, distance: -1, yaw: 0, pitch: 0)
        XCTAssertEqual(c.distance, 0.01)
    }
    // Task 4: the verticalFOV default (the current hard-coded 0.9 is kept) and how it reaches
    // the projection matrix (y = 1/tan(fov/2)).
    func testVerticalFOVDrivesProjection() {
        var c = CameraController()
        XCTAssertEqual(c.verticalFOV, 0.9, accuracy: 1e-6)   // the contract that the existing view behavior is unchanged
        c.verticalFOV = .pi / 2
        let p = c.projectionMatrix(aspect: 1)
        XCTAssertEqual(p.columns.1.y, 1 / tan(c.verticalFOV / 2), accuracy: 1e-6)   // = 1.0
    }

    // MARK: Adopting the override (Cinema footnote 2 ② — "grab it where you see it")

    /// The manipulation entry point (orbit) adopts the override pose into the orbit state and
    /// then applies the delta — the result matches, component by component, "a camera that
    /// already had that pose as its orbit receiving the same delta".
    /// Construction: reference and camera look at the same target (the centroid used for
    /// adoption = camera's old target), so the OrbitFit inversion recovers reference's orbit
    /// parameters exactly — the expected value is independent of the adoption implementation
    /// (non-circular). With the previous behavior (a plain release), camera would rotate from
    /// its pre-skim state (based on yaw −1.0) and diverge badly (the reproduction condition of
    /// the jump bug).
    func testOrbitAdoptsOverridePoseAndContinues() {
        let t0 = SIMD3<Float>(1, 2, 3)
        var reference = CameraController()
        reference.target = t0; reference.distance = 2.5
        reference.yaw = 0.8; reference.pitch = 0.3
        reference.verticalFOV = 1.2
        var camera = CameraController()
        camera.target = t0; camera.distance = 7
        camera.yaw = -1.0; camera.pitch = -0.5          // the orbit state before the skim
        camera.overridePose = CameraPose.effective(of: reference)   // on screen = reference's pose
        camera.orbit(dx: 12, dy: -6)
        reference.orbit(dx: 12, dy: -6)
        XCTAssertNil(camera.overridePose)               // released + adopted
        XCTAssertEqual(camera.yaw, reference.yaw, accuracy: 1e-4)
        XCTAssertEqual(camera.pitch, reference.pitch, accuracy: 1e-4)
        XCTAssertEqual(camera.distance, reference.distance, accuracy: 1e-4)
        XCTAssertLessThan(simd_distance(camera.target, reference.target), 1e-4)
        XCTAssertEqual(camera.verticalFOV, 1.2, accuracy: 1e-6)   // fov is adopted too (prevents a projection jump)
        let a = camera.viewMatrix, b = reference.viewMatrix
        for c in 0..<4 {
            for r in 0..<4 {
                XCTAssertEqual(a[c][r], b[c][r], accuracy: 1e-3, "col \(c) row \(r)")
            }
        }
    }

    /// Naturalizing the distance during adoption: the new distance = the depth of the old
    /// target along the view direction (a dot product), and the new target is that point on the
    /// pose's view ray. The zoom and pan entry points go through the same adoption.
    /// If the old target lies behind, OrbitFit's 0.5 floor is used as the fallback.
    func testZoomAdoptsOverrideUsingOldTargetDepth() {
        var camera = CameraController()
        camera.target = SIMD3(0, 0, -4)                 // depth 4 along the pose's view direction (−Z)
        camera.overridePose = CameraPose(
            position: .zero, rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), fov: 0.7)
        camera.zoom(0.5)
        XCTAssertNil(camera.overridePose)
        XCTAssertLessThan(simd_distance(camera.target, SIMD3(0, 0, -4)), 1e-4)
        XCTAssertEqual(camera.distance, 2, accuracy: 1e-4)          // depth 4 adopted → ×0.5
        XCTAssertEqual(camera.verticalFOV, 0.7, accuracy: 1e-6)
        XCTAssertLessThan(simd_distance(camera.eye, SIMD3(0, 0, -2)), 1e-3)   // on the view ray
        // The old target lies behind the pose — the minimum-depth 0.5 fallback (the
        // OrbitFit.fit floor)
        var behind = CameraController()
        behind.target = SIMD3(0, 0, 9)
        behind.overridePose = CameraPose(
            position: .zero, rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), fov: 0.9)
        behind.zoom(1.0)
        XCTAssertEqual(behind.distance, 0.5, accuracy: 1e-4)
    }
}
