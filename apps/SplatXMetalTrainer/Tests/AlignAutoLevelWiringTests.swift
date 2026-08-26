import XCTest
import simd
@testable import SplatXMetalTrainer

/// 2026-08-04 spec §1.2/§1.5 — the align write-time wiring of the auto-level correction.
///
/// The two decisions the wiring makes ("apply or not" and "report or not") are pure functions,
/// so they are pinned here. `AlignViewModel.start` itself needs a live PhotogrammetrySession and
/// is verified by the build plus the manual checks M1/M2 — the explicit policy of this app.
@MainActor
final class AlignAutoLevelWiringTests: XCTestCase {

    /// One frame tilted by exactly `tiltDegrees` about Z (a rotation about Z tilts the estimated
    /// up by the same angle — pinned in GroundLevelTests.testTiltDegreesMeasuresAngle).
    private func frames(tiltDegrees: Double) -> [AlignedFrame] {
        let q = simd_quatd(angle: tiltDegrees * .pi / 180, axis: SIMD3(0, 0, 1))
        return [AlignedFrame(name: "a.jpg", width: 640, height: 480,
                             fx: 500, fy: 510, cx: 320, cy: 240,
                             pose: ColmapPose(qw: q.real, qx: q.imag.x, qy: q.imag.y,
                                              qz: q.imag.z, tx: 0.5, ty: -1, tz: 2))]
    }

    private let points = [AlignedPoint(x: 1, y: 2, z: 3, r: 10, g: 20, b: 30)]

    /// Toggle off → an exact pass-through and nothing to report. This is the "unchanged from
    /// before" path (manual check M2).
    func testDisabledPassesThroughUntouched() {
        let f = frames(tiltDegrees: 20)
        let out = GroundLevel.applied(enabled: false, frames: f, points: points)
        XCTAssertNil(out.tiltDegrees)
        XCTAssertEqual(out.frames[0].pose, f[0].pose)
        XCTAssertEqual(out.points[0].x, points[0].x)
        XCTAssertEqual(out.points[0].y, points[0].y)
        XCTAssertEqual(out.points[0].z, points[0].z)
    }

    /// Toggle on with a real tilt → the scene is rotated and the applied angle is reported.
    func testEnabledReportsAppliedDegrees() {
        let f = frames(tiltDegrees: 20)
        let out = GroundLevel.applied(enabled: true, frames: f, points: points)
        XCTAssertEqual(out.tiltDegrees ?? 0, 20, accuracy: 1e-9)
        XCTAssertNotEqual(out.frames[0].pose, f[0].pose)
        let up = GroundLevel.estimateUp(poses: out.frames.map(\.pose))
        XCTAssertEqual(simd_length(up - SIMD3<Double>(0, -1, 0)), 0, accuracy: 1e-12)
    }

    /// Toggle on but under the threshold → nothing rotated, so nothing reported. The UI must
    /// never show "0.2°" for a scene that was left alone (spec §1.5).
    func testEnabledButBelowThresholdReportsNil() {
        let f = frames(tiltDegrees: 0.2)
        XCTAssertLessThan(0.2, GroundLevel.minTiltDegrees)
        let out = GroundLevel.applied(enabled: true, frames: f, points: points)
        XCTAssertNil(out.tiltDegrees)
        XCTAssertEqual(out.frames[0].pose, f[0].pose)
    }

    /// The console line and AlignSummary.levelTiltDegrees are driven by the same optional, so
    /// the log and the summary can never disagree about whether a correction happened.
    func testConsoleLineOnlyWhenApplied() {
        XCTAssertNil(AlignViewModel.levelConsoleLine(tiltDegrees: nil))
        XCTAssertEqual(AlignViewModel.levelConsoleLine(tiltDegrees: 19.177130068350046),
                       "· Auto-level: scene rotated 19.2° to level the ground")
        XCTAssertEqual(AlignViewModel.levelConsoleLine(tiltDegrees: 12.44),
                       "· Auto-level: scene rotated 12.4° to level the ground")
    }
}
