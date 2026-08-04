// Deleted in the first restructuring and revived for the camera fit (2026-07-20 spec §6)
import XCTest
import simd
@testable import SplatXMetalTrainer

final class CameraConversionTests: XCTestCase {
    // img1: q=(1,0,0,0), a unit rotation R=I, t=(0.1,-0.2,3.0). COLMAP: X_cam = R·X_world + t.
    // The camera center C = -Rᵀ t = -t = (-0.1, 0.2, -3.0).
    func testCameraCenterIdentityRotation() {
        let c = CameraConversion.cameraCenter(qvec: SIMD4(1,0,0,0), tvec: SIMD3(0.1,-0.2,3.0))
        XCTAssertEqual(c.x, -0.1, accuracy: 1e-5)
        XCTAssertEqual(c.y,  0.2, accuracy: 1e-5)
        XCTAssertEqual(c.z, -3.0, accuracy: 1e-5)
    }
    // Transforming the world origin by the view matrix gives the COLMAP camera coordinate t
    // moved into the Metal convention (Y and Z flipped).
    func testWorldOriginMapsToFlippedTranslation() {
        let v = CameraConversion.viewMatrix(qvec: SIMD4(1,0,0,0), tvec: SIMD3(0.1,-0.2,3.0))
        let o = v * SIMD4<Float>(0,0,0,1)
        // COLMAP cam coordinates (0.1,-0.2,3.0) → Metal (-Y,-Z): (0.1, 0.2, -3.0)
        XCTAssertEqual(o.x,  0.1, accuracy: 1e-5)
        XCTAssertEqual(o.y,  0.2, accuracy: 1e-5)
        XCTAssertEqual(o.z, -3.0, accuracy: 1e-5)
    }
    func testProjectionIsFinite() {
        let k = ColmapIntrinsics(fx: 1500, fy: 1510, cx: 960, cy: 720, width: 1920, height: 1440)
        let p = CameraConversion.projectionMatrix(k, near: 0.01, far: 100)
        XCTAssertTrue(p.columns.0.x.isFinite && p.columns.1.y.isFinite)
    }
    // Even with degenerate intrinsics (fx=fy=0, near==far) it must produce a finite matrix
    // with no crash, NaN, or Inf.
    func testProjectionFiniteOnDegenerateIntrinsics() {
        let k = ColmapIntrinsics(fx: 0, fy: 0, cx: 960, cy: 720, width: 1920, height: 1440)
        let p = CameraConversion.projectionMatrix(k, near: 1, far: 1)   // near==far is degenerate as well
        let cols = [p.columns.0, p.columns.1, p.columns.2, p.columns.3]
        for col in cols {
            for comp in [col.x, col.y, col.z, col.w] {
                XCTAssertTrue(comp.isFinite, "projection component is not finite: \(comp)")
            }
        }
    }
}
