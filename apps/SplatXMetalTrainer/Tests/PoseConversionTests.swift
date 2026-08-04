import XCTest
import simd
@testable import SplatXMetalTrainer

final class PoseConversionTests: XCTestCase {
    // An identity RealityKit c2w (the camera at the origin gazing along -Z) → the COLMAP w2c
    // is a 180° rotation about X (F).
    func testIdentityPose() {
        let p = PoseConversion.colmapFromRealityKit(
            rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), translation: .zero)
        XCTAssertEqual(p.qw, 0, accuracy: 1e-6)
        XCTAssertEqual(abs(p.qx), 1, accuracy: 1e-6)   // the qw≥0 normalization makes the sign +
        XCTAssertEqual(p.qy, 0, accuracy: 1e-6)
        XCTAssertEqual(p.qz, 0, accuracy: 1e-6)
        XCTAssertEqual(p.tx, 0, accuracy: 1e-6)
    }

    // The camera at (0,0,5) gazing along -Z: t_colmap = -R·C = (0,0,5).
    func testTranslationOnly() {
        let p = PoseConversion.colmapFromRealityKit(
            rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), translation: SIMD3(0, 0, 5))
        XCTAssertEqual(p.tx, 0, accuracy: 1e-5)
        XCTAssertEqual(p.ty, 0, accuracy: 1e-5)
        XCTAssertEqual(p.tz, 5, accuracy: 1e-5)
    }

    // A world point in front of the camera must have +Z in COLMAP camera coordinates (the
    // forwardness contract).
    func testPointInFrontProjectsToPositiveZ() {
        // An arbitrary rotation: a 60° yaw about Y plus the position (1,2,3)
        let rot = simd_quatf(angle: .pi / 3, axis: SIMD3(0, 1, 0))
        let cam = SIMD3<Float>(1, 2, 3)
        // A world point 2 m ahead along the camera's local -Z
        let forwardWorld = rot.act(SIMD3<Float>(0, 0, -2))
        let pWorld = cam + forwardWorld
        let p = PoseConversion.colmapFromRealityKit(rotation: rot, translation: cam)
        // p_cam = R·p_world + t (reconstructed in double)
        let q = simd_quatd(ix: p.qx, iy: p.qy, iz: p.qz, r: p.qw)
        let R = simd_double3x3(q)
        let pc = R * SIMD3<Double>(Double(pWorld.x), Double(pWorld.y), Double(pWorld.z))
               + SIMD3<Double>(p.tx, p.ty, p.tz)
        XCTAssertEqual(pc.z, 2, accuracy: 1e-4)
        XCTAssertEqual(pc.x, 0, accuracy: 1e-4)
        XCTAssertEqual(pc.y, 0, accuracy: 1e-4)
    }
}
