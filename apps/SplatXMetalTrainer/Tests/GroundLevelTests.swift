import XCTest
import simd
@testable import SplatXMetalTrainer

/// 2026-08-04 spec §1.4 — the ground-leveling math baked into sparse/0 at align write time.
/// All of it is pure Double math, so it is fully unit-testable (the SwiftUI/Metal surface is
/// manual E2E — the explicit policy of this app).
///
/// The rotation oracle below is built independently of GroundLevel's own private helper on
/// purpose: if both sides shared one implementation, the invariance test would be checking the
/// implementation against itself.
final class GroundLevelTests: XCTestCase {

    // MARK: builders / independent oracle

    /// The COLMAP w2c rotation R of a pose quaternion (w,x,y,z).
    private func rotation(_ p: ColmapPose) -> simd_double3x3 {
        simd_double3x3(simd_quatd(ix: p.qx, iy: p.qy, iz: p.qz, r: p.qw).normalized)
    }

    private func pose(_ q: simd_quatd, _ t: SIMD3<Double>) -> ColmapPose {
        ColmapPose(qw: q.real, qx: q.imag.x, qy: q.imag.y, qz: q.imag.z,
                   tx: t.x, ty: t.y, tz: t.z)
    }

    private func frame(_ name: String, _ p: ColmapPose) -> AlignedFrame {
        AlignedFrame(name: name, width: 640, height: 480,
                     fx: 500, fy: 510, cx: 320, cy: 240, pose: p)
    }

    /// A three-view tilted set whose mean up is (-0.050811, -0.944508, 0.324536) — a measured
    /// tilt of 19.1771°, comfortably above minTiltDegrees so leveling really runs.
    private func tiltedFrames() -> [AlignedFrame] {
        let qs = [simd_quatd(angle: 0.3, axis: simd_normalize(SIMD3<Double>(1, 2, 3))),
                  simd_quatd(angle: -0.5, axis: simd_normalize(SIMD3<Double>(0, 1, 0.2))),
                  simd_quatd(angle: 1.1, axis: simd_normalize(SIMD3<Double>(2, -1, 0.5)))]
        let ts = [SIMD3<Double>(0.5, -1, 2), SIMD3<Double>(-0.3, 0.8, 1.5),
                  SIMD3<Double>(1.2, 0.4, -0.7)]
        return (0..<3).map { frame("img\($0).jpg", pose(qs[$0], ts[$0])) }
    }

    private func samplePoints() -> [AlignedPoint] {
        [AlignedPoint(x: 1, y: 2, z: 3, r: 10, g: 20, b: 30),
         AlignedPoint(x: -2, y: 0.5, z: 4, r: 40, g: 50, b: 60),
         AlignedPoint(x: 0.1, y: -0.2, z: 0.3, r: 70, g: 80, b: 90)]
    }

    // MARK: 1. estimateUp

    /// An empty list, and two poses whose ups cancel exactly, both fall back to the target
    /// itself — which makes the leveling rotation the identity, i.e. "do nothing". Note this
    /// differs from SceneUp, which falls back to +Y for the viewer's orbit axis.
    func testEmptyAndDegenerateUpFallBackToGravity() {
        XCTAssertEqual(GroundLevel.estimateUp(poses: []), SIMD3<Double>(0, -1, 0))
        // identity → up (0,−1,0); 180° about Z → up (0,+1,0). The sum is exactly zero.
        let a = ColmapPose(qw: 1, qx: 0, qy: 0, qz: 0, tx: 0, ty: 0, tz: 0)
        let b = ColmapPose(qw: 0, qx: 0, qy: 0, qz: 1, tx: 0, ty: 0, tz: 0)
        XCTAssertEqual(GroundLevel.estimateUp(poses: [a, b]), SIMD3<Double>(0, -1, 0))
    }

    /// The same formula as SceneUp.estimate / CameraConversion.upWorld, in Double over
    /// ColmapPose: up_i = Rᵀ·(0,−1,0) = −(row 1 of R), summed then normalized.
    func testEstimateUpIsNormalizedMean() {
        let h = (0.5 as Double).squareRoot()
        let a = ColmapPose(qw: 1, qx: 0, qy: 0, qz: 0, tx: 0, ty: 0, tz: 0)   // up (0,−1,0)
        let b = ColmapPose(qw: h, qx: h, qy: 0, qz: 0, tx: 0, ty: 0, tz: 0)   // up (0, 0, 1)
        XCTAssertEqual(simd_length(GroundLevel.estimateUp(poses: [a]) - SIMD3<Double>(0, -1, 0)),
                       0, accuracy: 1e-12)
        let mean = GroundLevel.estimateUp(poses: [a, b])
        XCTAssertEqual(simd_length(mean), 1, accuracy: 1e-12)
        XCTAssertEqual(simd_length(mean - simd_normalize(SIMD3<Double>(0, -1, 1))),
                       0, accuracy: 1e-12)
    }

    // MARK: 2. levelingRotation

    func testLevelingRotationSendsUpToGravity() {
        for up in [SIMD3<Double>(-0.05081117118318026, -0.9445075639685201, 0.3245361096846473),
                   simd_normalize(SIMD3<Double>(0.3, -0.9, 0.2)),
                   SIMD3<Double>(1, 0, 0),        // perpendicular
                   SIMD3<Double>(0, 0, 1),        // perpendicular
                   SIMD3<Double>(0, -4.25, 0)] {  // already level but not unit length
            let g = GroundLevel.levelingRotation(up: up)
            XCTAssertEqual(simd_length(g * simd_normalize(up) - SIMD3<Double>(0, -1, 0)),
                           0, accuracy: 1e-12)
            XCTAssertEqual(g.determinant, 1, accuracy: 1e-12)   // a proper rotation, not a reflection
        }
    }

    /// Already level → G is the identity, so a level scene is never perturbed.
    func testAlreadyLevelGivesIdentity() {
        let g = GroundLevel.levelingRotation(up: SIMD3(0, -1, 0))
        for c in 0..<3 {
            for r in 0..<3 { XCTAssertEqual(g[c][r], c == r ? 1 : 0, accuracy: 1e-12) }
        }
        XCTAssertEqual(GroundLevel.tiltDegrees(up: SIMD3(0, -1, 0)), 0, accuracy: 1e-12)
    }

    /// up ≈ +Y is exactly opposite the target, so the shortest-arc quaternion degenerates
    /// (w = 1 + a·b = 0) and the 180° branch must take over with an axis perpendicular to a.
    /// This is the branch that silently produces a zero quaternion if it is missing.
    func testAntiparallelUpRotatesOneEighty() {
        let g = GroundLevel.levelingRotation(up: SIMD3(0, 1, 0))
        XCTAssertEqual(simd_length(g * SIMD3<Double>(0, 1, 0) - SIMD3<Double>(0, -1, 0)),
                       0, accuracy: 1e-12)
        XCTAssertEqual(g.determinant, 1, accuracy: 1e-12)
        XCTAssertEqual(GroundLevel.tiltDegrees(up: SIMD3(0, 1, 0)), 180, accuracy: 1e-9)
        // Scaled but still antiparallel → the same branch.
        let g2 = GroundLevel.levelingRotation(up: SIMD3(0, 7.5, 0))
        XCTAssertEqual(simd_length(g2 * SIMD3<Double>(0, 1, 0) - SIMD3<Double>(0, -1, 0)),
                       0, accuracy: 1e-12)
    }

    // MARK: 3. the invariance contract (the heart of the feature)

    /// Rotating the world by G and compensating the poses with R' = R·Gᵀ, t' = t must leave
    /// every rendered image identical — i.e. X_cam = R·p + t is invariant for every
    /// (frame, point) pair. If this ever fails, leveling has corrupted the reconstruction.
    func testCameraPointGeometryIsInvariant() {
        let frames = tiltedFrames()
        let points = samplePoints()
        let out = GroundLevel.leveled(frames: frames, points: points)
        XCTAssertGreaterThan(out.tiltDegrees, GroundLevel.minTiltDegrees)   // leveling really ran
        XCTAssertEqual(out.frames.count, frames.count)
        XCTAssertEqual(out.points.count, points.count)
        for (i, f) in frames.enumerated() {
            let r = rotation(f.pose)
            let t = SIMD3<Double>(f.pose.tx, f.pose.ty, f.pose.tz)
            let rP = rotation(out.frames[i].pose)
            let tP = SIMD3<Double>(out.frames[i].pose.tx, out.frames[i].pose.ty,
                                   out.frames[i].pose.tz)
            XCTAssertEqual(simd_length(t - tP), 0, accuracy: 1e-15)   // t is untouched by design
            for (j, p) in points.enumerated() {
                let before = r * SIMD3<Double>(p.x, p.y, p.z) + t
                let q = out.points[j]
                let after = rP * SIMD3<Double>(q.x, q.y, q.z) + tP
                XCTAssertEqual(simd_length(before - after), 0, accuracy: 1e-12)
            }
        }
    }

    /// After leveling, the estimator converges on the target — so SceneUp downstream returns
    /// ≈(0,−1,0) and the existing camera.upAxis path keeps working without double-correcting.
    func testLeveledSceneEstimatesGravityUp() {
        let out = GroundLevel.leveled(frames: tiltedFrames(), points: samplePoints())
        let up = GroundLevel.estimateUp(poses: out.frames.map(\.pose))
        XCTAssertEqual(simd_length(up - SIMD3<Double>(0, -1, 0)), 0, accuracy: 1e-12)
        XCTAssertEqual(GroundLevel.tiltDegrees(up: up), 0, accuracy: 1e-6)
    }

    // MARK: 4. the threshold

    /// Table-driven: a rotation of θ about Z tilts the estimated up by exactly θ.
    func testTiltDegreesMeasuresAngle() {
        for degrees in [0.2, 0.5, 5.0, 30.0, 90.0] {
            let q = simd_quatd(angle: degrees * .pi / 180, axis: SIMD3(0, 0, 1))
            let p = pose(q, SIMD3(0, 0, 0))
            XCTAssertEqual(GroundLevel.tiltDegrees(up: GroundLevel.estimateUp(poses: [p])),
                           degrees, accuracy: 1e-9)
        }
        XCTAssertEqual(GroundLevel.tiltDegrees(up: SIMD3(0, 0, 0)), 0)   // degenerate → 0
    }

    /// 0.2° < minTiltDegrees (0.5°) — the arrays come back untouched and the applied correction
    /// is 0. The contract is that `tiltDegrees` reports what was **applied**, so the caller
    /// never has to repeat the threshold and the UI never shows "0.2°" for an untouched scene.
    func testBelowThresholdIsAPureNoOp() {
        let q = simd_quatd(angle: 0.2 * .pi / 180, axis: SIMD3(0, 0, 1))
        let frames = [frame("a.jpg", pose(q, SIMD3(1, 2, 3)))]
        let points = samplePoints()
        let out = GroundLevel.leveled(frames: frames, points: points)
        XCTAssertEqual(out.tiltDegrees, 0)
        XCTAssertEqual(out.frames[0].pose, frames[0].pose)   // ColmapPose is Equatable
        for (i, p) in points.enumerated() {
            XCTAssertEqual(out.points[i].x, p.x)
            XCTAssertEqual(out.points[i].y, p.y)
            XCTAssertEqual(out.points[i].z, p.z)
        }
    }

    // MARK: 5. what leveling must NOT touch

    /// ColmapTextWriter binds CAMERA_ID to the name sort, and intrinsics belong to the lens
    /// rather than to the world. Reordering or rescaling here would silently desynchronize
    /// cameras.txt from images.txt — a corruption with no error message.
    func testNamesOrderAndIntrinsicsSurvive() {
        let src = tiltedFrames()
        let frames = [frame("c.jpg", src[0].pose),
                      frame("a.jpg", src[1].pose),
                      frame("b.jpg", src[2].pose)]
        let points = samplePoints()
        let out = GroundLevel.leveled(frames: frames, points: points)
        XCTAssertGreaterThan(out.tiltDegrees, GroundLevel.minTiltDegrees)   // not a vacuous pass
        XCTAssertEqual(out.frames.map(\.name), ["c.jpg", "a.jpg", "b.jpg"])  // order preserved, not sorted
        for (i, f) in frames.enumerated() {
            XCTAssertEqual(out.frames[i].width, f.width)
            XCTAssertEqual(out.frames[i].height, f.height)
            XCTAssertEqual(out.frames[i].fx, f.fx)
            XCTAssertEqual(out.frames[i].fy, f.fy)
            XCTAssertEqual(out.frames[i].cx, f.cx)
            XCTAssertEqual(out.frames[i].cy, f.cy)
        }
        XCTAssertEqual(out.points.map(\.r), points.map(\.r))   // colors ride along untouched
        XCTAssertEqual(out.points.map(\.g), points.map(\.g))
        XCTAssertEqual(out.points.map(\.b), points.map(\.b))
    }

    /// PoseConversion normalizes to qw ≥ 0 and ColmapTextWriter copies the four components
    /// verbatim — leveled poses must keep the same convention and stay unit length.
    func testQuaternionOutputKeepsWriterConvention() {
        let out = GroundLevel.leveled(frames: tiltedFrames(), points: samplePoints())
        for f in out.frames {
            XCTAssertGreaterThanOrEqual(f.pose.qw, 0)
            let n = simd_length(SIMD4<Double>(f.pose.qw, f.pose.qx, f.pose.qy, f.pose.qz))
            XCTAssertEqual(n, 1, accuracy: 1e-12)
        }
    }
}
