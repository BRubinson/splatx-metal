import XCTest
import simd
@testable import SplatXMetalTrainer

/// 2026-08-04 spec §2.2: CameraConversion.pose — the exact inverse of the COLMAP→Metal view
/// convention (the OrbitFit approximation drops roll and needs a centroid; this does not).
/// The golden is the round trip against ExportRenderMath.colmapViewmatRowMajor — the same
/// style as VideoExportTests.testColmapViewmatRoundtripWithCameraConversion, which pins the
/// opposite direction.
final class FirstCameraPoseTests: XCTestCase {

    /// The golden. (qvec,tvec) → pose → colmapViewmatRowMajor must recover the original [R | t]
    /// exactly (row-major world→cam, the COLMAP convention).
    func testPoseRoundTripsThroughColmapViewmat() {
        let qvec = SIMD4<Double>(0.82, 0.31, -0.41, 0.25)   // non-unit — rotation() normalizes it
        let tvec = SIMD3<Double>(0.4, -1.2, 2.5)
        let pose = CameraConversion.pose(qvec: qvec, tvec: tvec, verticalFOV: 0.75)
        let vm = ExportRenderMath.colmapViewmatRowMajor(position: pose.position,
                                                        rotation: pose.rotation)
        let R = CameraConversion.rotation(qvec)
        let t = SIMD3<Float>(Float(tvec.x), Float(tvec.y), Float(tvec.z))
        for r in 0..<3 {
            XCTAssertEqual(vm[r * 4 + 0], R.columns.0[r], accuracy: 1e-4, "R[\(r)][0]")
            XCTAssertEqual(vm[r * 4 + 1], R.columns.1[r], accuracy: 1e-4, "R[\(r)][1]")
            XCTAssertEqual(vm[r * 4 + 2], R.columns.2[r], accuracy: 1e-4, "R[\(r)][2]")
            XCTAssertEqual(vm[r * 4 + 3], t[r], accuracy: 1e-4, "t[\(r)]")
        }
        XCTAssertEqual(Array(vm[12..<16]), [0, 0, 0, 1])
    }

    /// The same contract stated directly on the viewport side: the pose's own view matrix is
    /// the Metal view matrix of that COLMAP camera (flip·[R|t]), column for column. This is
    /// what makes the viewport picture identical to the captured image.
    func testPoseViewMatrixEqualsCameraConversionViewMatrix() {
        let qvec = SIMD4<Double>(0.31, -0.55, 0.42, 0.64)
        let tvec = SIMD3<Double>(-1.7, 0.9, 3.3)
        let pose = CameraConversion.pose(qvec: qvec, tvec: tvec, verticalFOV: 0.9)
        let expected = CameraConversion.viewMatrix(qvec: qvec, tvec: tvec)
        let actual = pose.viewMatrix
        for c in 0..<4 {
            for r in 0..<4 {
                XCTAssertEqual(actual[c][r], expected[c][r], accuracy: 1e-4, "[\(c)][\(r)]")
            }
        }
    }

    /// position = the camera center C = −Rᵀ·t, and the rotation's basis columns agree with the
    /// existing world-direction helpers (camera→world: −Z forward, +Y up). fov passes through.
    func testPosePositionAndBasis() {
        let qvec = SIMD4<Double>(0.6, 0.2, -0.7, 0.33)
        let tvec = SIMD3<Double>(2.0, -0.5, 1.25)
        let pose = CameraConversion.pose(qvec: qvec, tvec: tvec, verticalFOV: 0.5)
        XCTAssertEqual(simd_length(pose.position - CameraConversion.cameraCenter(qvec: qvec,
                                                                                tvec: tvec)),
                       0, accuracy: 1e-5)
        let m = simd_float3x3(pose.rotation.normalized)
        XCTAssertEqual(simd_length(-m.columns.2 - CameraConversion.forwardWorld(qvec: qvec)),
                       0, accuracy: 1e-5)
        XCTAssertEqual(simd_length(m.columns.1 - CameraConversion.upWorld(qvec: qvec)),
                       0, accuracy: 1e-5)
        XCTAssertEqual(pose.fov, 0.5)
    }

    /// Drift guard: the rotation is by definition the same value as the Cinema capture-path
    /// preset's COLMAP→Metal rotation (PathPresets.metalCameraRotation). Both are quat(Rᵀ·flip)
    /// — if one is ever changed the other must change with it, and this pins that.
    func testPoseRotationMatchesPathPresets() {
        let qvec = SIMD4<Double>(0.82, 0.31, -0.41, 0.25)
        let pose = CameraConversion.pose(qvec: qvec, tvec: SIMD3(0.4, -1.2, 2.5),
                                         verticalFOV: 0.9)
        let expected = PathPresets.metalCameraRotation(qvec: qvec)
        XCTAssertEqual(simd_length(pose.rotation.vector - expected.vector), 0, accuracy: 1e-6)
    }

    /// fov = 2·atan(H / (2·fy)) — the existing OrbitFit.verticalFOV is reused unchanged
    /// (spec §2.2), table-driven.
    func testVerticalFOVTable() {
        let table: [(fy: Double, height: Int, expected: Double)] = [
            (1440, 1440, 2 * atan(0.5)),
            (800, 1600, 2 * atan(1.0)),
            (1200, 600, 2 * atan(0.25)),
        ]
        for row in table {
            XCTAssertEqual(Double(OrbitFit.verticalFOV(fy: row.fy, height: row.height)),
                           row.expected, accuracy: 1e-6, "fy \(row.fy) h \(row.height)")
        }
    }
}
