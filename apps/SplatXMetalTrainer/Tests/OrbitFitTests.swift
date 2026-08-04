import XCTest
import simd
@testable import SplatXMetalTrainer

final class OrbitFitTests: XCTestCase {
    // Brief ①: a camera at the origin gazing along -Z with a centroid of (0,0,-3) →
    // target=(0,0,-3), distance=3; feeding yaw/pitch into the CameraController convention
    // reproduces eye = the origin and forward = -Z.
    func testFitMinusZGaze() {
        let r = OrbitFit.fit(cameraPosition: SIMD3(0, 0, 0),
                             forward: SIMD3(0, 0, -1),
                             up: SIMD3(0, 1, 0),
                             pointsCentroid: SIMD3(0, 0, -3))
        XCTAssertEqual(r.target.x, 0, accuracy: 1e-6)
        XCTAssertEqual(r.target.y, 0, accuracy: 1e-6)
        XCTAssertEqual(r.target.z, -3, accuracy: 1e-6)
        XCTAssertEqual(r.distance, 3, accuracy: 1e-6)
        // Round trip: the CameraController convention (eye = target + dir·distance,
        // dir = (cos p·sin y, sin p, cos p·cos y)) reproduces eye and forward.
        var cam = CameraController()
        cam.target = r.target; cam.distance = r.distance; cam.yaw = r.yaw; cam.pitch = r.pitch
        XCTAssertEqual(simd_length(cam.eye - SIMD3<Float>(0, 0, 0)), 0, accuracy: 1e-5)
        let f = simd_normalize(cam.target - cam.eye)
        XCTAssertEqual(simd_length(f - SIMD3<Float>(0, 0, -1)), 0, accuracy: 1e-5)
    }

    // Brief ②: when the centroid is behind the camera (its projection onto the view direction
    // is negative), distance is clamped to the 0.5 floor.
    func testFitClampsDistanceWhenCentroidBehindCamera() {
        let r = OrbitFit.fit(cameraPosition: SIMD3(0, 0, 0),
                             forward: SIMD3(0, 0, -1),
                             up: SIMD3(0, 1, 0),
                             pointsCentroid: SIMD3(0, 0, +4))   // projection = -4 < 0
        XCTAssertEqual(r.distance, 0.5, accuracy: 1e-6)
        XCTAssertEqual(r.target.z, -0.5, accuracy: 1e-6)        // C + f·0.5
    }

    // An arbitrary-pose round trip: feeding the inverted yaw/pitch/target/distance into
    // CameraController must reproduce the camera position as eye and the input gaze as forward.
    func testRoundTripArbitraryPose() {
        let C = SIMD3<Float>(1.5, -2.0, 3.0)
        let f = simd_normalize(SIMD3<Float>(0.3, -0.45, -0.85))
        let centroid = SIMD3<Float>(2.0, -3.5, -1.0)
        let r = OrbitFit.fit(cameraPosition: C, forward: f,
                             up: SIMD3(0, 1, 0), pointsCentroid: centroid)
        var cam = CameraController()
        cam.target = r.target; cam.distance = r.distance; cam.yaw = r.yaw; cam.pitch = r.pitch
        XCTAssertEqual(simd_length(cam.eye - C), 0, accuracy: 1e-4)
        let f2 = simd_normalize(cam.target - cam.eye)
        XCTAssertEqual(simd_length(f2 - f), 0, accuracy: 1e-5)
        // distance is the projection (centroid − C)·f.
        XCTAssertEqual(r.distance, simd_dot(centroid - C, f), accuracy: 1e-5)
    }

    // Brief ③: the vertical FOV = 2·atan(H / (2·fy)). fy=height → 2·atan(0.5) ≈ 53.13°.
    func testVerticalFOV() {
        let fov = OrbitFit.verticalFOV(fy: 1440, height: 1440)
        XCTAssertEqual(fov, 2 * atan(0.5), accuracy: 1e-6)                 // ≈ 0.9273 rad
        XCTAssertEqual(fov * 180 / .pi, 53.130, accuracy: 1e-2)            // ≈ 53.13°
        // fy = H/2 → 2·atan(1) = 90°.
        XCTAssertEqual(OrbitFit.verticalFOV(fy: 720, height: 1440), .pi / 2, accuracy: 1e-6)
    }
}
