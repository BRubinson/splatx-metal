// Cinema Task 1: the numerical contract of PathSampler (TDD — expected values written first,
// hand-checked against an independent oracle).
// Every assertion is a hand-computed literal (never restate the implementation — that is
// tautology). Lineage:
// - Midpoint of the bottom edge of the square loop (0.5, -0.125): every chord length is 1 →
//   centripetal = uniform CR, and the Hermite expansion y(τ)=0.5τ(τ-1) → y(0.5)=-0.125
//   (matches the scratch oracle verify_campath.py).
// - Evenly spaced keys on a straight line: the reflected phantoms produce collinear, evenly
//   spaced control points, so CR degenerates to exactly linear (x(τ)=τ) → the constant-speed
//   assertions hold to floating-point precision.
// - quintic ease(0.25) = 6(0.25)⁵-15(0.25)⁴+10(0.25)³ = 0.103515625 (an exact binary
//   fraction).
// - slerp: identity→90°z, so at t=0.25 the rotation is 22.5° → (1,0,0) rotates to
//   (cos22.5°, sin22.5°, 0).
import XCTest
import simd
@testable import SplatXMetalTrainer

final class CameraPathSamplerTests: XCTestCase {

    // MARK: Helpers (they only assemble test inputs — no numerical computation)

    private func key(_ p: SIMD3<Float>,
                     quat: SIMD4<Float> = SIMD4(1, 0, 0, 0),
                     fov: Float = 0.9) -> Keyframe {
        Keyframe(time: 0, pos: p, quat: quat, fov: fov)
    }

    private func makePath(_ keys: [Keyframe], duration: Double,
                          ease: EaseProfile = .none, loop: Bool = false) -> CameraPath {
        CameraPath(id: UUID(), name: "test", keyframes: keys,
                   smoothness: 0.5, ease: ease, loop: loop, duration: duration)
    }

    /// w-first storage: a 90° rotation about z = (cos45°, 0, 0, sin45°).
    private let quat90z = SIMD4<Float>(0.70710678, 0, 0, 0.70710678)

    // MARK: Constant speed (arc-length table + binary search)

    // Three evenly spaced keys on a straight line — under constant speed the midpoint in time
    // is the midpoint in position. A total length of 2 is traversed at constant speed over a
    // duration of 2, so pos.x == t must be an identity.
    func testStraightLineConstantSpeed() {
        let p = makePath([key(SIMD3(0, 0, 0)), key(SIMD3(1, 0, 0)), key(SIMD3(2, 0, 0))],
                         duration: 2.0)
        let s = PathSampler(path: p)
        XCTAssertEqual(s.totalLength, 2.0, accuracy: 1e-6)   // straight, so the chord lengths sum to exactly 2
        XCTAssertEqual(s.pose(at: 0).pos.x, 0.0, accuracy: 1e-5)
        XCTAssertEqual(s.pose(at: 0.5).pos.x, 0.5, accuracy: 1e-4)   // midpoint in t (segment 0) = midpoint in position
        XCTAssertEqual(s.pose(at: 1.0).pos.x, 1.0, accuracy: 1e-4)   // passes exactly through the middle key
        XCTAssertEqual(s.pose(at: 1.7).pos.x, 1.7, accuracy: 1e-4)   // inside segment 1
        XCTAssertEqual(s.pose(at: 2.0).pos.x, 2.0, accuracy: 1e-5)
        for t in [0.0, 0.5, 1.0, 1.7, 2.0] {
            XCTAssertEqual(s.pose(at: t).pos.y, 0, accuracy: 1e-6)
            XCTAssertEqual(s.pose(at: t).pos.z, 0, accuracy: 1e-6)
        }
    }

    // MARK: Closed loop end condition + centripetal with no overshoot

    // A 4-key loop on the unit square: it closes (end = start), the outward bulge at each edge
    // midpoint is exactly 0.125 (equal-length chords → the analytic solution for a uniform CR
    // closed curve), and the whole curve stays inside [-0.125, 1.125]².
    func testSquareLoopClosedNoOvershoot() {
        let p = makePath([key(SIMD3(0, 0, 0)), key(SIMD3(1, 0, 0)),
                          key(SIMD3(1, 1, 0)), key(SIMD3(0, 1, 0))],
                         duration: 4.0, loop: true)
        let s = PathSampler(path: p)
        // Closure: t=0 and t=duration give the same pose
        let start = s.pose(at: 0).pos, end = s.pose(at: 4).pos
        XCTAssertEqual(start.x, 0, accuracy: 1e-5); XCTAssertEqual(start.y, 0, accuracy: 1e-5)
        XCTAssertEqual(end.x, 0, accuracy: 1e-5);   XCTAssertEqual(end.y, 0, accuracy: 1e-5)
        // Symmetry → four equal-length segments → at 1/4 it passes exactly through the
        // corner (1,0,0)
        let corner = s.pose(at: 1).pos
        XCTAssertEqual(corner.x, 1, accuracy: 1e-4); XCTAssertEqual(corner.y, 0, accuracy: 1e-4)
        // Bottom-edge midpoint (t=0.5 → the half-arc-length point = τ=0.5):
        // (0.5, -0.125, 0) — the analytic literal
        let mid = s.pose(at: 0.5).pos
        XCTAssertEqual(mid.x, 0.5, accuracy: 1e-4)
        XCTAssertEqual(mid.y, -0.125, accuracy: 1e-4)
        // Full scan: no overshoot beyond the 0.125 bulge limit, no cusp, and the curve stays
        // planar
        for t in stride(from: 0.0, through: 4.0, by: 0.05) {
            let q = s.pose(at: t).pos
            XCTAssertGreaterThan(q.x, -0.126); XCTAssertLessThan(q.x, 1.126)
            XCTAssertGreaterThan(q.y, -0.126); XCTAssertLessThan(q.y, 1.126)
            XCTAssertEqual(q.z, 0, accuracy: 1e-6)
        }
    }

    // MARK: slerp hemisphere normalization (sign flip when dot < 0)

    // q1 is given as -q(90°z) (an input with mixed signs). With hemisphere normalization the
    // path takes the shortest route (90°); without it, the long way around (270°). The t=0.25
    // assertion also discriminates constant angular velocity (a true slerp).
    func testSlerpHemisphereShortestPath() {
        let negQ90z = SIMD4<Float>(-0.70710678, 0, 0, -0.70710678)   // -(cos45°,0,0,sin45°)
        let p = makePath([key(SIMD3(0, 0, 0), quat: SIMD4(1, 0, 0, 0)),
                          key(SIMD3(1, 0, 0), quat: negQ90z)],
                         duration: 1.0)
        let s = PathSampler(path: p)
        // t=0.5 → 45°: (1,0,0) → (cos45°, sin45°, 0)
        let vHalf = s.pose(at: 0.5).quat.act(SIMD3<Float>(1, 0, 0))
        XCTAssertEqual(vHalf.x, 0.70710678, accuracy: 1e-3)
        XCTAssertEqual(vHalf.y, 0.70710678, accuracy: 1e-3)
        XCTAssertEqual(vHalf.z, 0, accuracy: 1e-4)
        // t=0.25 → 22.5°: (cos22.5°, sin22.5°, 0) = (0.9238795, 0.3826834, 0)
        let vQuarter = s.pose(at: 0.25).quat.act(SIMD3<Float>(1, 0, 0))
        XCTAssertEqual(vQuarter.x, 0.9238795, accuracy: 1e-3)
        XCTAssertEqual(vQuarter.y, 0.3826834, accuracy: 1e-3)
    }

    // MARK: FOV interpolation

    // Three keys with fov 0.5/0.75/1.25 (exact binary fractions) — linear within each
    // segment.
    func testFovInterpolationAcrossSegments() {
        let p = makePath([key(SIMD3(0, 0, 0), fov: 0.5),
                          key(SIMD3(1, 0, 0), fov: 0.75),
                          key(SIMD3(2, 0, 0), fov: 1.25)],
                         duration: 2.0)
        let s = PathSampler(path: p)
        XCTAssertEqual(s.pose(at: 0).fov, 0.5, accuracy: 1e-5)
        XCTAssertEqual(s.pose(at: 0.5).fov, 0.625, accuracy: 1e-4)   // segment 0 midpoint
        XCTAssertEqual(s.pose(at: 1.0).fov, 0.75, accuracy: 1e-4)    // the middle key
        XCTAssertEqual(s.pose(at: 1.5).fov, 1.0, accuracy: 1e-4)     // segment 1 midpoint
        XCTAssertEqual(s.pose(at: 2.0).fov, 1.25, accuracy: 1e-5)
    }

    // MARK: quintic ease composed onto the time axis

    func testQuinticEaseComposition() {
        // Direct assertions: the 6t⁵-15t⁴+10t³ literals at the boundaries, the midpoint,
        // and 1/4
        XCTAssertEqual(PathSampler.quintic(0), 0, accuracy: 1e-12)
        XCTAssertEqual(PathSampler.quintic(1), 1, accuracy: 1e-12)
        XCTAssertEqual(PathSampler.quintic(0.5), 0.5, accuracy: 1e-12)
        XCTAssertEqual(PathSampler.quintic(0.25), 0.103515625, accuracy: 1e-12)
        // Composition assertion: on an exactly linear 2-key straight line,
        // pos.x == ease(t/duration)
        let p = makePath([key(SIMD3(0, 0, 0)), key(SIMD3(1, 0, 0))],
                         duration: 1.0, ease: .easeInOut)
        let s = PathSampler(path: p)
        XCTAssertEqual(s.pose(at: 0.25).pos.x, 0.103515625, accuracy: 1e-4)
        XCTAssertEqual(s.pose(at: 0.5).pos.x, 0.5, accuracy: 1e-4)
        XCTAssertEqual(s.pose(at: 0).pos.x, 0, accuracy: 1e-6)      // exact match at the boundary
        XCTAssertEqual(s.pose(at: 1).pos.x, 1, accuracy: 1e-6)
    }

    // Even with ease active, t=0 and t=duration must match the key values (pos, quat, and
    // fov) exactly.
    func testEaseEndpointExactness() {
        let p = makePath([key(SIMD3(0, 0, 0), quat: SIMD4(1, 0, 0, 0), fov: 0.5),
                          key(SIMD3(1, 2, 3), quat: quat90z, fov: 1.25)],
                         duration: 2.0, ease: .easeInOut)
        let s = PathSampler(path: p)
        let a = s.pose(at: 0)
        XCTAssertEqual(a.pos.x, 0, accuracy: 1e-5)
        XCTAssertEqual(a.fov, 0.5, accuracy: 1e-5)
        let va = a.quat.act(SIMD3<Float>(1, 0, 0))              // identity → (1,0,0) unchanged
        XCTAssertEqual(va.x, 1, accuracy: 1e-4); XCTAssertEqual(va.y, 0, accuracy: 1e-4)
        let b = s.pose(at: 2)
        XCTAssertEqual(b.pos.x, 1, accuracy: 1e-5)
        XCTAssertEqual(b.pos.y, 2, accuracy: 1e-5)
        XCTAssertEqual(b.pos.z, 3, accuracy: 1e-5)
        XCTAssertEqual(b.fov, 1.25, accuracy: 1e-5)
        let vb = b.quat.act(SIMD3<Float>(1, 0, 0))              // 90°z → (0,1,0)
        XCTAssertEqual(vb.x, 0, accuracy: 1e-3); XCTAssertEqual(vb.y, 1, accuracy: 1e-3)
    }

    // MARK: The three degenerate-case contracts

    // ① One key = a static shot: the pose of that key at every t.
    func testSingleKeyStillShot() {
        let p = makePath([key(SIMD3(1, 2, 3), quat: quat90z, fov: 0.75)], duration: 5.0)
        let s = PathSampler(path: p)
        for t in [0.0, 2.5, 5.0, -1.0, 9.0] {
            let pose = s.pose(at: t)
            XCTAssertEqual(pose.pos.x, 1, accuracy: 1e-6)
            XCTAssertEqual(pose.pos.y, 2, accuracy: 1e-6)
            XCTAssertEqual(pose.pos.z, 3, accuracy: 1e-6)
            XCTAssertEqual(pose.fov, 0.75, accuracy: 1e-6)
            let v = pose.quat.act(SIMD3<Float>(1, 0, 0))        // 90°z → (0,1,0)
            XCTAssertEqual(v.y, 1, accuracy: 1e-3)
        }
    }

    // ② Two keys = a natural degeneration to lerp (the reflected endpoint phantoms make the
    // control points collinear, so it is a straight line with no special-case branch).
    func testTwoKeyLerpDegenerate() {
        let p = makePath([key(SIMD3(0, 0, 0)), key(SIMD3(2, 0, 0))], duration: 2.0)
        let s = PathSampler(path: p)
        XCTAssertEqual(s.pose(at: 0.5).pos.x, 0.5, accuracy: 1e-4)
        XCTAssertEqual(s.pose(at: 1.0).pos.x, 1.0, accuracy: 1e-4)
        XCTAssertEqual(s.pose(at: 1.5).pos.x, 1.5, accuracy: 1e-4)
    }

    // ③ Adjacent keys at the same position (Δt=0) — the ε guard reproduces the endpoints
    // without NaN or divergence.
    func testDuplicateAdjacentKeysNoNaN() {
        let p = makePath([key(SIMD3(0, 0, 0)), key(SIMD3(0, 0, 0)), key(SIMD3(1, 0, 0))],
                         duration: 1.0)
        let s = PathSampler(path: p)
        for t in stride(from: 0.0, through: 1.0, by: 0.05) {
            let q = s.pose(at: t)
            XCTAssertTrue(q.pos.x.isFinite && q.pos.y.isFinite && q.pos.z.isFinite,
                          "non-finite position at t=\(t): \(q.pos)")
            XCTAssertTrue(q.fov.isFinite)
            XCTAssertGreaterThan(q.pos.x, -0.01); XCTAssertLessThan(q.pos.x, 1.01)
        }
        XCTAssertEqual(s.pose(at: 0).pos.x, 0, accuracy: 1e-5)
        XCTAssertEqual(s.pose(at: 1).pos.x, 1, accuracy: 1e-3)
    }

    // All keys at the same position (total arc length 0) — the position is fixed while
    // rotation and FOV interpolate uniformly in time (a pan shot).
    func testAllKeysSamePositionRotationStillInterpolates() {
        let p = makePath([key(SIMD3(0, 0, 0), quat: SIMD4(1, 0, 0, 0), fov: 0.5),
                          key(SIMD3(0, 0, 0), quat: quat90z, fov: 1.0)],
                         duration: 2.0)
        let s = PathSampler(path: p)
        // t=1 (s=0.5) → 45°
        let v = s.pose(at: 1).quat.act(SIMD3<Float>(1, 0, 0))
        XCTAssertEqual(v.x, 0.70710678, accuracy: 1e-3)
        XCTAssertEqual(v.y, 0.70710678, accuracy: 1e-3)
        XCTAssertEqual(s.pose(at: 1).fov, 0.75, accuracy: 1e-4)
        // t=0.5 (s=0.25) → 22.5°
        let vQuarter = s.pose(at: 0.5).quat.act(SIMD3<Float>(1, 0, 0))
        XCTAssertEqual(vQuarter.x, 0.9238795, accuracy: 1e-3)
        XCTAssertEqual(vQuarter.y, 0.3826834, accuracy: 1e-3)
        XCTAssertEqual(s.pose(at: 1).pos.x, 0, accuracy: 1e-6)
    }

    // MARK: Boundary contract

    // t is clamped to 0…duration (a playback request outside the range never goes past the
    // end pose).
    func testClampOutsideRange() {
        let p = makePath([key(SIMD3(0, 0, 0)), key(SIMD3(1, 0, 0))], duration: 1.0)
        let s = PathSampler(path: p)
        XCTAssertEqual(s.pose(at: -5).pos.x, 0, accuracy: 1e-6)
        XCTAssertEqual(s.pose(at: 7).pos.x, 1, accuracy: 1e-6)
    }

    // An empty path = the feature is off — a neutral pose with no crash (origin, identity,
    // and the viewer's default FOV of 0.9 rad).
    func testEmptyPathNeutralPose() {
        let s = PathSampler(path: makePath([], duration: 1.0))
        let pose = s.pose(at: 0.5)
        XCTAssertEqual(pose.pos.x, 0, accuracy: 1e-6)
        XCTAssertEqual(pose.fov, 0.9, accuracy: 1e-6)
        let v = pose.quat.act(SIMD3<Float>(1, 0, 0))
        XCTAssertEqual(v.x, 1, accuracy: 1e-5)
    }

    // MARK: The <1ms rebuild contract — timing measurements are flaky and therefore
    // prohibited; assert it structurally instead
    // (a fixed 32 samples per segment → O(n·32) work for n keys. A few dozen keys is << 1ms.)

    func testRebuildStructureContract() {
        XCTAssertEqual(PathSampler.samplesPerSegment, 32)
        let keys = (0..<64).map { key(SIMD3(Float($0), 0, 0)) }
        let open = PathSampler(path: makePath(keys, duration: 10))
        XCTAssertEqual(open.segmentCount, 63)                    // n-1 segments
        let loop = PathSampler(path: makePath(keys, duration: 10, loop: true))
        XCTAssertEqual(loop.segmentCount, 64)                    // a loop adds one closing segment
    }
}
