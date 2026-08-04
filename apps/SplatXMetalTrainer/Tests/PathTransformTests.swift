// Path transform Task 1 (2026-07-27 spec §3 and §4): the pure math of PathTransform — the
// least-squares convergence point (§3.1), the three decision conditions, pivot selection
// (§3.2), the invariants and guards of translation (§3.3) and scaling (§3.4), and the handle
// drag arithmetic (§4.1 translation, §4.2 scale ratio) plus handleDirection. The rotations of
// the synthetic orbits are built with the same convention as the presets
// (PathPresets.lookRotation) — consistent with the Task 3 assertions.
import XCTest
import simd
@testable import SplatXMetalTrainer

final class PathTransformTests: XCTestCase {

    // MARK: Synthetic orbit helpers

    /// A circular or arc orbit around center — the same basis formula as orbit360
    /// (orbitBasis + cp/sp), with every key gazing at center via lookRotation. closed=true uses
    /// i/n (the wrap segment closes the circle), false uses i/(n−1) (an open arc — the endpoint
    /// is included).
    private func makeOrbitKeys(center: SIMD3<Float>, radius: Float, pitch: Float,
                               upAxis: SIMD3<Float>, count: Int = 8,
                               sweep: Float = 2 * .pi, closed: Bool = true) -> [Keyframe] {
        let (bx, bu, bz) = CameraController.orbitBasis(upAxis: upAxis)
        let cp = cos(pitch), sp = sin(pitch)
        return (0..<count).map { i in
            let y = Float(i) / Float(closed ? count : count - 1) * sweep
            let dir = (cp * sin(y)) * bx + sp * bu + (cp * cos(y)) * bz
            let pos = center + dir * radius
            return Keyframe(time: Double(i), pos: pos,
                            rotation: PathPresets.lookRotation(eye: pos, target: center, up: bu),
                            fov: 0.9)
        }
    }

    private func makePath(_ keys: [Keyframe]) -> CameraPath {
        CameraPath(name: "t", keyframes: keys, duration: 10)
    }

    // MARK: The convergence point (§3.1)

    func testPivotConvergesOnOrbitWithAxisUp() {
        let c = SIMD3<Float>(1, 2, 3), r: Float = 4
        let keys = makeOrbitKeys(center: c, radius: r, pitch: 0.4,
                                 upAxis: SIMD3(0, 0, 1))
        let p = PathTransform.pivot(for: keys)
        XCTAssertNotNil(p)
        XCTAssertLessThanOrEqual(simd_distance(p!.point, c), 1e-3 * r)   // relative: 1e-3·R
        XCTAssertTrue(p!.isAimLocked)
        XCTAssertEqual(p!.meanRadius, r, accuracy: 1e-3 * r)
    }

    func testPivotConvergesOnOrbitWithObliqueUp() {
        let c = SIMD3<Float>(-2, 1, 5), r: Float = 3
        let keys = makeOrbitKeys(center: c, radius: r, pitch: -0.3,
                                 upAxis: simd_normalize(SIMD3(0.3, 0.5, 0.8)))
        let p = PathTransform.pivot(for: keys)
        XCTAssertNotNil(p)
        XCTAssertLessThanOrEqual(simd_distance(p!.point, c), 1e-3 * r)
        XCTAssertTrue(p!.isAimLocked)
        XCTAssertEqual(p!.meanRadius, r, accuracy: 1e-3 * r)
    }

    // MARK: The three ways the decision returns false (§3.1 conditions 1, 2, and 3)

    func testPivotNotAimLockedForParallelGaze() {
        // (a) A straight fly-through: every key gazes in the same direction (−Z) → A is
        // singular → det(A/n) fails.
        let keys = (0..<5).map { i -> Keyframe in
            let pos = SIMD3<Float>(Float(i), 0, 0)
            return Keyframe(time: Double(i), pos: pos,
                            rotation: PathPresets.lookRotation(
                                eye: pos, target: pos + SIMD3(0, 0, -1), up: SIMD3(0, 1, 0)),
                            fov: 0.9)
        }
        let p = PathTransform.pivot(for: keys)
        XCTAssertNotNil(p)
        XCTAssertFalse(p!.isAimLocked)
        // The fallback pivot = the centroid (§3.2), and meanRadius = the mean distance to
        // the centroid
        XCTAssertLessThanOrEqual(simd_distance(p!.point, SIMD3(2, 0, 0)), 1e-4)
        XCTAssertEqual(p!.meanRadius, 1.2, accuracy: 1e-4)   // (2+1+0+1+2)/5
    }

    func testPivotResidualRatioSeparatesDivergentGaze() {
        // (b) The residual-ratio decision: the same circular layout, with only the aim points
        // scattered. A scatter of 1.5 (37% of R=4) fails with a residual ratio > 0.10, while
        // 0.2 (5%) is still aim-locked — this pins what the 0.10 threshold separates, in both
        // directions. The det and forwardness conditions pass in both layouts.
        let c = SIMD3<Float>(1, 2, 3), r: Float = 4
        let (bx, bu, bz) = CameraController.orbitBasis(upAxis: SIMD3(0, 1, 0))
        let offsets = [bx, bz, -bx, -bz]
        func keys(scatter: Float) -> [Keyframe] {
            (0..<8).map { i -> Keyframe in
                let y = Float(i) / 8 * 2 * .pi
                let pos = c + (sin(y) * bx + cos(y) * bz) * r
                let aim = c + offsets[i % 4] * scatter
                return Keyframe(time: Double(i), pos: pos,
                                rotation: PathPresets.lookRotation(eye: pos, target: aim, up: bu),
                                fov: 0.9)
            }
        }
        XCTAssertFalse(PathTransform.pivot(for: keys(scatter: 1.5))!.isAimLocked)
        XCTAssertTrue(PathTransform.pivot(for: keys(scatter: 0.2))!.isAimLocked)
    }

    func testPivotNotAimLockedWhenConvergenceBehindCameras() {
        // (c) A forwardness violation: every key gazes away from the pivot — the infinite
        // lines still pass through c, so det and the residual pass, but the convergence point
        // lies behind the cameras → mean (p−o)·d < 0 fails.
        let c = SIMD3<Float>(1, 2, 3), r: Float = 4
        let (_, bu, _) = CameraController.orbitBasis(upAxis: SIMD3(0, 1, 0))
        let keys = makeOrbitKeys(center: c, radius: r, pitch: 0, upAxis: SIMD3(0, 1, 0))
            .map { k -> Keyframe in
                Keyframe(time: k.time, pos: k.pos,
                         rotation: PathPresets.lookRotation(
                             eye: k.pos, target: k.pos + (k.pos - c), up: bu),
                         fov: k.fov)
            }
        let p = PathTransform.pivot(for: keys)
        XCTAssertNotNil(p)
        XCTAssertFalse(p!.isAimLocked)
    }

    // MARK: Distinguishing the pivot (§3.2 — on a 220° arc the centroid ≠ the orbit center)

    func testPivotArc220UsesConvergenceNotCentroid() {
        let c = SIMD3<Float>(1, 2, 3), r: Float = 4
        let keys = makeOrbitKeys(center: c, radius: r, pitch: 0, upAxis: SIMD3(0, 1, 0),
                                 count: 6, sweep: 220 * .pi / 180, closed: false)
        let centroid = keys.reduce(SIMD3<Float>.zero) { $0 + $1.pos } / Float(keys.count)
        let p = PathTransform.pivot(for: keys)!
        XCTAssertTrue(p.isAimLocked)
        XCTAssertLessThanOrEqual(simd_distance(p.point, c), 1e-3 * r)    // matches the orbit center
        XCTAssertGreaterThan(simd_distance(centroid, c), 0.5)            // it is an arc, so the centroid is skewed
        XCTAssertGreaterThan(simd_distance(p.point, centroid), 0.5)      // significantly different from the centroid
    }

    // MARK: Translation (§3.3 — quat untouched, the camera→pivot vector unchanged)

    func testTranslatedKeepsQuatFovAndPivotOffsets() {
        let keys = makeOrbitKeys(center: SIMD3(1, 2, 3), radius: 4, pitch: 0.4,
                                 upAxis: SIMD3(0, 1, 0))
        let path = makePath(keys)
        let delta = SIMD3<Float>(5, -3, 2)
        let moved = PathTransform.translated(path, by: delta)

        XCTAssertEqual(moved.id, path.id)
        XCTAssertEqual(moved.keyframes.count, path.keyframes.count)
        let before = PathTransform.pivot(for: path.keyframes)!
        let after = PathTransform.pivot(for: moved.keyframes)!
        for (o, n) in zip(path.keyframes, moved.keyframes) {
            XCTAssertEqual(n.quat, o.quat)                 // byte-identical (re-aiming is a no-op)
            XCTAssertEqual(n.fov, o.fov)
            XCTAssertEqual(n.time, o.time)
            XCTAssertEqual(n.pos, o.pos + delta)
            // (new pivot − new position) == (old pivot − old position)
            XCTAssertLessThanOrEqual(
                simd_distance(after.point - n.pos, before.point - o.pos), 1e-3)
        }
    }

    // MARK: Scaling (§3.4 — concentricity kept, gaze preserved)

    func testScaledKeepsQuatFovGazeAndScalesRadiiUniformly() {
        let c = SIMD3<Float>(1, 2, 3)
        let path = makePath(makeOrbitKeys(center: c, radius: 4, pitch: 0.4,
                                          upAxis: SIMD3(0, 1, 0)))
        let factor: Float = 1.6
        let scaled = PathTransform.scaled(path, by: factor, about: c)
        XCTAssertNotNil(scaled)
        for (o, n) in zip(path.keyframes, scaled!.keyframes) {
            XCTAssertEqual(n.quat, o.quat)                 // byte-identical
            XCTAssertEqual(n.fov, o.fov)
            let ratio = simd_distance(n.pos, c) / simd_distance(o.pos, c)
            XCTAssertEqual(ratio, factor, accuracy: 1e-5)  // distance ratio = the factor
            // The pivot→camera direction is unchanged → the gaze is preserved
            XCTAssertLessThanOrEqual(
                simd_distance(simd_normalize(n.pos - c), simd_normalize(o.pos - c)), 1e-5)
        }
    }

    // MARK: Guards (§3.4)

    func testScaledClampsFactorToBounds() {
        let c = SIMD3<Float>(1, 2, 3)
        let path = makePath(makeOrbitKeys(center: c, radius: 4, pitch: 0,
                                          upAxis: SIMD3(0, 1, 0)))
        // 0.01 → the lower bound 0.05, 50 → the upper bound 20
        let lo = PathTransform.scaled(path, by: 0.01, about: c)!
        let hi = PathTransform.scaled(path, by: 50, about: c)!
        for (o, n) in zip(path.keyframes, lo.keyframes) {
            XCTAssertEqual(simd_distance(n.pos, c) / simd_distance(o.pos, c),
                           0.05, accuracy: 1e-5)
        }
        for (o, n) in zip(path.keyframes, hi.keyframes) {
            XCTAssertEqual(simd_distance(n.pos, c) / simd_distance(o.pos, c),
                           20, accuracy: 1e-4)
        }
    }

    func testScaledNilWhenResultRadiusDegenerates() {
        // Shrinking a radius-0.01 orbit by 0.05 → the resulting R̄ = 5e-4 < 1e-3 → the
        // transform is rejected (nil)
        let c = SIMD3<Float>(1, 2, 3)
        let tiny = makePath(makeOrbitKeys(center: c, radius: 0.01, pitch: 0,
                                          upAxis: SIMD3(0, 1, 0)))
        XCTAssertNil(PathTransform.scaled(tiny, by: 0.05, about: c))
        // Zero keys is nil too (nothing to transform — Task 2 hides the handle itself)
        XCTAssertNil(PathTransform.scaled(makePath([]), by: 2, about: .zero))
    }

    func testPivotNilForEmptyKeys() {
        XCTAssertNil(PathTransform.pivot(for: []))
    }

    // MARK: Drag arithmetic (§4.1)

    func testWorldPerPointFormula() {
        // k = 2·tan(fov/2)·|pivot−eye| / viewportHeight
        let k = PathTransform.worldPerPoint(verticalFOV: 1.0, pivotDistance: 4,
                                            viewportHeight: 800)
        XCTAssertEqual(k, 2 * tan(0.5) * 4 / 800, accuracy: 1e-7)
    }

    func testDragTranslationHorizontalSigns() {
        // Screen y is positive downward → moving the cursor up (Δy<0) gives −Δy·ê_f =
        // camera forward (§4.1).
        let d = PathTransform.dragTranslation(
            screenDelta: CGSize(width: 10, height: -20), vertical: false, worldPerPoint: 2,
            cameraRight: SIMD3(1, 0, 0), cameraForward: SIMD3(0, 0, -1),
            upAxis: SIMD3(0, 1, 0))
        XCTAssertLessThanOrEqual(simd_distance(d, SIMD3(20, 0, -40)), 1e-4)
    }

    func testDragTranslationProjectsBasesToOrbitPlane() {
        // A tilted forward is projected onto the orbit plane (⊥ upAxis) and normalized — no
        // vertical component leaks in.
        let d = PathTransform.dragTranslation(
            screenDelta: CGSize(width: 0, height: -10), vertical: false, worldPerPoint: 1,
            cameraRight: SIMD3(1, 0, 0),
            cameraForward: simd_normalize(SIMD3(0, -1, -1)),
            upAxis: SIMD3(0, 1, 0))
        XCTAssertLessThanOrEqual(simd_distance(d, SIMD3(0, 0, -10)), 1e-4)
    }

    func testDragTranslationTopViewVerticalDragMovesForward() {
        // The top-view regression (a real-usage bug on 2026-07-28): when forward ∥ upAxis the
        // orbit-plane projection of forward is 0, so a vertical drag did nothing. Deriving
        // ê_f = cross(up, ê_r) must make the cursor moving up (Δy<0) come out as the upward
        // direction on the camera's screen (−Z).
        let d = PathTransform.dragTranslation(
            screenDelta: CGSize(width: 0, height: -10), vertical: false, worldPerPoint: 2,
            cameraRight: SIMD3(1, 0, 0), cameraForward: SIMD3(0, -1, 0),   // straight down
            upAxis: SIMD3(0, 1, 0))
        XCTAssertGreaterThan(simd_length(d), 0)                            // not a no-op
        XCTAssertLessThanOrEqual(simd_distance(d, 2 * 10 * SIMD3(0, 0, -1)), 1e-4)
    }

    func testDragTranslationRollFallbackWhenRightParallelToUp() {
        // The 90° roll degeneracy (right ∥ upAxis): ê_r is nil → ê_f falls back to the
        // previous forward projection and the Δx contribution is 0 — the same behavior as
        // before the cross-product derivation was introduced.
        let d = PathTransform.dragTranslation(
            screenDelta: CGSize(width: 10, height: -20), vertical: false, worldPerPoint: 2,
            cameraRight: SIMD3(0, 2, 0), cameraForward: SIMD3(0, 0, -1),
            upAxis: SIMD3(0, 1, 0))
        XCTAssertLessThanOrEqual(simd_distance(d, SIMD3(0, 0, -40)), 1e-4)
    }

    func testDragTranslationVerticalIgnoresHorizontalAndNormalizesUp() {
        // ⇧: Δworld = k·(−Δy)·upAxis — the horizontal component (Δx) is ignored, and cursor
        // up = rise.
        let d = PathTransform.dragTranslation(
            screenDelta: CGSize(width: 10, height: -20), vertical: true, worldPerPoint: 2,
            cameraRight: SIMD3(1, 0, 0), cameraForward: SIMD3(0, 0, -1),
            upAxis: SIMD3(0, 2, 0))   // a non-unit upAxis is normalized too
        XCTAssertLessThanOrEqual(simd_distance(d, SIMD3(0, 40, 0)), 1e-4)
    }

    // MARK: Scale drag (§4.2 — a pure 2D ratio)

    func testDragScaleFactorRatioAndDenominatorGuard() {
        let pivot = CGPoint(x: 100, y: 100)
        // |cursor−pivot| / |handleStart−pivot| = 90 / 60 = 1.5
        XCTAssertEqual(PathTransform.dragScaleFactor(
            cursor: CGPoint(x: 100, y: 190), pivotScreen: pivot,
            handleStart: CGPoint(x: 160, y: 100))!, 1.5, accuracy: 1e-5)
        // A denominator under 8pt → nil (the pivot and the handle overlap on screen);
        // exactly 8pt is allowed
        XCTAssertNil(PathTransform.dragScaleFactor(
            cursor: CGPoint(x: 100, y: 190), pivotScreen: pivot,
            handleStart: CGPoint(x: 105, y: 100)))
        XCTAssertNotNil(PathTransform.dragScaleFactor(
            cursor: CGPoint(x: 100, y: 190), pivotScreen: pivot,
            handleStart: CGPoint(x: 108, y: 100)))
    }

    // MARK: handleDirection (§4.2 — û: the orbit-plane projection of right, with a
    // degenerate fallback)

    func testHandleDirectionIsUnitAndPerpendicularToUp() {
        let d = PathTransform.handleDirection(cameraRight: SIMD3(1, 0, 0),
                                              upAxis: SIMD3(0, 1, 0))
        XCTAssertLessThanOrEqual(simd_distance(d, SIMD3(1, 0, 0)), 1e-5)

        let up = simd_normalize(SIMD3<Float>(0.3, 0.5, 0.8))
        let d2 = PathTransform.handleDirection(cameraRight: SIMD3(1, 0, 0), upAxis: up)
        XCTAssertEqual(simd_length(d2), 1, accuracy: 1e-5)
        XCTAssertEqual(simd_dot(d2, up), 0, accuracy: 1e-5)
        XCTAssertGreaterThan(simd_dot(d2, SIMD3(1, 0, 0)), 0)   // stays on the right side
    }

    func testHandleDirectionFallbackWhenRightParallelToUp() {
        // The degeneracy (right ∥ upAxis): the fallback is still a unit vector perpendicular
        // to upAxis.
        let d = PathTransform.handleDirection(cameraRight: SIMD3(0, 2, 0),
                                              upAxis: SIMD3(0, 1, 0))
        XCTAssertEqual(simd_length(d), 1, accuracy: 1e-5)
        XCTAssertEqual(simd_dot(d, SIMD3(0, 1, 0)), 0, accuracy: 1e-5)
    }
}
