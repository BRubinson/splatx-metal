// Cinema Task 3: hybrid pacing (spec footnote 1) — the sampler's manual mode, the derivation
// of keyframeTimes, the quintic inverse, and pacing Codable. Per the Task 1 convention, every
// assertion is a hand-computed literal.
// Lineage:
// - quinticInverse(0.103515625) = 0.25: the inverse of quintic(0.25) = 6/4⁵−15/4⁴+10/4³ =
//   0.103515625 (an exact binary fraction). By the symmetry quintic(1−s) = 1−quintic(s), the
//   complement of qinv(0.25)≈0.35944 — qinv(0.75)≈0.64056 — is pinned as well (the numbers
//   come from a bisection hand-check — f(0.359)=0.24931, f(0.36)=0.25090).
// - The straight-line hand-check for manual pacing: evenly spaced collinear keys make CR
//   exactly linear (the Task 1 lineage), so the piecewise-linear time → arc-length mapping can
//   be read directly off the position coordinates.
// - Midpoint of the closing segment of the square loop (−0.125, 0.5): the Task 1 analytic
//   solution (an outward bulge of 0.125 at each edge midpoint) applied to the left edge.
import XCTest
import simd
@testable import SplatXMetalTrainer

final class CameraPathPacingTests: XCTestCase {

    private func key(_ p: SIMD3<Float>,
                     quat: SIMD4<Float> = SIMD4(1, 0, 0, 0),
                     fov: Float = 0.9, time: Double = 0) -> Keyframe {
        Keyframe(time: time, pos: p, quat: quat, fov: fov)
    }

    private func makePath(_ keys: [Keyframe], duration: Double,
                          ease: EaseProfile = .none, loop: Bool = false,
                          pacing: PacingMode = .even) -> CameraPath {
        CameraPath(id: UUID(), name: "test", keyframes: keys,
                   smoothness: 0.5, ease: ease, loop: loop, pacing: pacing,
                   duration: duration)
    }

    // MARK: The quintic inverse (monotone bisection)

    func testQuinticInverseLiterals() {
        XCTAssertEqual(PathSampler.quinticInverse(0), 0, accuracy: 1e-12)
        XCTAssertEqual(PathSampler.quinticInverse(1), 1, accuracy: 1e-12)
        XCTAssertEqual(PathSampler.quinticInverse(0.5), 0.5, accuracy: 1e-9)
        XCTAssertEqual(PathSampler.quinticInverse(0.103515625), 0.25, accuracy: 1e-9)
        // Round-trip identity (verifying that the bisection converges) — a dense scan
        for s in stride(from: 0.0, through: 1.0, by: 0.05) {
            XCTAssertEqual(PathSampler.quinticInverse(PathSampler.quintic(s)), s,
                           accuracy: 1e-8, "s=\(s)")
        }
    }

    // MARK: keyframeTimes — derived under even pacing (the single source of truth for the
    // diamond positions)

    // Three evenly spaced keys on a straight line, no ease: the arc-length ratios pass
    // straight through → [0, 1, 2].
    func testKeyframeTimesEvenNoEase() {
        let p = makePath([key(SIMD3(0, 0, 0)), key(SIMD3(1, 0, 0)), key(SIMD3(2, 0, 0))],
                         duration: 2.0)
        let times = PathSampler(path: p).keyframeTimes
        XCTAssertEqual(times.count, 3)
        XCTAssertEqual(times[0], 0, accuracy: 1e-9)
        XCTAssertEqual(times[1], 1, accuracy: 1e-6)
        XCTAssertEqual(times[2], 2, accuracy: 1e-9)
    }

    // Five evenly spaced keys on a straight line + easeInOut: t_i = duration·qinv(i/4).
    // qinv(0.25)≈0.35944 and qinv(0.75)=1−qinv(0.25)≈0.64056 (the quintic symmetry).
    func testKeyframeTimesEvenWithEaseInverse() {
        let keys = (0..<5).map { key(SIMD3(Float($0), 0, 0)) }
        let p = makePath(keys, duration: 4.0, ease: .easeInOut)
        let times = PathSampler(path: p).keyframeTimes
        XCTAssertEqual(times[0], 0, accuracy: 1e-9)
        XCTAssertEqual(times[1], 4 * 0.35944, accuracy: 2e-3)
        XCTAssertEqual(times[2], 2, accuracy: 1e-4)                 // qinv(0.5) = 0.5
        XCTAssertEqual(times[3], 4 * 0.64056, accuracy: 2e-3)
        XCTAssertEqual(times[4], 4, accuracy: 1e-9)
        // Monotonically increasing whether or not ease is applied
        for i in 1..<times.count { XCTAssertGreaterThan(times[i], times[i - 1]) }
        // Consistency between the eased derivation and passing through the keys: sampling at
        // a derived time gives exactly that key's position
        let s = PathSampler(path: p)
        XCTAssertEqual(s.pose(at: times[1]).pos.x, 1, accuracy: 1e-3)
        XCTAssertEqual(s.pose(at: times[3]).pos.x, 3, accuracy: 1e-3)
    }

    // A 4-key square loop: equal-length segments → [0, 1, 2, 3] (the wrap time 4 is not a key
    // — the view draws it as a ghost).
    func testKeyframeTimesLoop() {
        let p = makePath([key(SIMD3(0, 0, 0)), key(SIMD3(1, 0, 0)),
                          key(SIMD3(1, 1, 0)), key(SIMD3(0, 1, 0))],
                         duration: 4.0, loop: true)
        let times = PathSampler(path: p).keyframeTimes
        XCTAssertEqual(times.count, 4)
        for (i, expected) in [0.0, 1.0, 2.0, 3.0].enumerated() {
            XCTAssertEqual(times[i], expected, accuracy: 1e-6)
        }
    }

    // A fully static path (arc length 0) — the derived times match the uniform fallback axis.
    func testKeyframeTimesZeroLengthUniformFallback() {
        let p = makePath([key(SIMD3(0, 0, 0)), key(SIMD3(0, 0, 0)), key(SIMD3(0, 0, 0))],
                         duration: 4.0)
        let times = PathSampler(path: p).keyframeTimes
        XCTAssertEqual(times[0], 0, accuracy: 1e-9)
        XCTAssertEqual(times[1], 2, accuracy: 1e-9)   // i/segments = 1/2
        XCTAssertEqual(times[2], 4, accuracy: 1e-9)
    }

    // One key = [0] under even, or the stored time clamped under manual. Zero keys = [].
    func testKeyframeTimesDegenerateCounts() {
        XCTAssertEqual(PathSampler(path: makePath([], duration: 5)).keyframeTimes, [])
        XCTAssertEqual(PathSampler(path: makePath([key(SIMD3(1, 2, 3))], duration: 5))
            .keyframeTimes, [0])
        let manual1 = makePath([key(SIMD3(1, 2, 3), time: 9)], duration: 5, pacing: .manual)
        XCTAssertEqual(PathSampler(path: manual1).keyframeTimes, [5])   // clamped to duration
    }

    // MARK: Manual pacing — respecting keyframe times (piecewise-linear time → arc length)

    // Three keys on a straight line with times [0, 8, 10] and duration 10: the first span
    // (length 1) takes 8 s, the second takes 2 s.
    func testManualPacingRespectsKeyTimes() {
        let p = makePath([key(SIMD3(0, 0, 0), time: 0), key(SIMD3(1, 0, 0), time: 8),
                          key(SIMD3(2, 0, 0), time: 10)],
                         duration: 10, pacing: .manual)
        let s = PathSampler(path: p)
        XCTAssertEqual(s.pose(at: 0).pos.x, 0, accuracy: 1e-5)
        XCTAssertEqual(s.pose(at: 4).pos.x, 0.5, accuracy: 1e-4)    // span 0 midpoint
        XCTAssertEqual(s.pose(at: 8).pos.x, 1.0, accuracy: 1e-4)    // passes exactly through key 1
        XCTAssertEqual(s.pose(at: 9).pos.x, 1.5, accuracy: 1e-4)    // span 1 midpoint
        XCTAssertEqual(s.pose(at: 10).pos.x, 2.0, accuracy: 1e-5)
        XCTAssertEqual(s.keyframeTimes, [0, 8, 10])                 // the stored times are authoritative
    }

    // Manual mode does not apply the global ease (it is incompatible with respecting
    // keyframe times — the PacingMode contract).
    func testManualPacingIgnoresEase() {
        let keys = [key(SIMD3(0, 0, 0), time: 0), key(SIMD3(1, 0, 0), time: 8),
                    key(SIMD3(2, 0, 0), time: 10)]
        let none = PathSampler(path: makePath(keys, duration: 10, ease: .none, pacing: .manual))
        let eased = PathSampler(path: makePath(keys, duration: 10, ease: .easeInOut,
                                               pacing: .manual))
        for t in stride(from: 0.0, through: 10.0, by: 0.5) {
            XCTAssertEqual(none.pose(at: t).pos.x, eased.pose(at: t).pos.x, accuracy: 1e-6,
                           "t=\(t)")
        }
    }

    // Before the first key time and after the last key time it holds (keeping the end pose).
    func testManualPacingHoldsOutsideKeyRange() {
        let p = makePath([key(SIMD3(0, 0, 0), time: 2), key(SIMD3(1, 0, 0), time: 8)],
                         duration: 10, pacing: .manual)
        let s = PathSampler(path: p)
        XCTAssertEqual(s.pose(at: 0).pos.x, 0, accuracy: 1e-5)
        XCTAssertEqual(s.pose(at: 2).pos.x, 0, accuracy: 1e-5)
        XCTAssertEqual(s.pose(at: 5).pos.x, 0.5, accuracy: 1e-4)
        XCTAssertEqual(s.pose(at: 8).pos.x, 1, accuracy: 1e-4)
        XCTAssertEqual(s.pose(at: 9.5).pos.x, 1, accuracy: 1e-4)    // hold
    }

    // loop + manual: the span from the last key time to duration is allocated to the
    // end → start wrap segment.
    func testManualPacingLoopWrap() {
        let p = makePath([key(SIMD3(0, 0, 0), time: 0), key(SIMD3(1, 0, 0), time: 0.5),
                          key(SIMD3(1, 1, 0), time: 2), key(SIMD3(0, 1, 0), time: 3)],
                         duration: 4, loop: true, pacing: .manual)
        let s = PathSampler(path: p)
        XCTAssertEqual(s.pose(at: 0.5).pos.x, 1, accuracy: 1e-4)    // key 1 (1,0)
        XCTAssertEqual(s.pose(at: 0.5).pos.y, 0, accuracy: 1e-4)
        XCTAssertEqual(s.pose(at: 2).pos.y, 1, accuracy: 1e-4)      // key 2 (1,1)
        // Wrap midpoint (t=3.5): the outward bulge at the left-edge midpoint (−0.125, 0.5) —
        // the Task 1 analytic solution applied to the left edge
        let wrapMid = s.pose(at: 3.5).pos
        XCTAssertEqual(wrapMid.x, -0.125, accuracy: 1e-4)
        XCTAssertEqual(wrapMid.y, 0.5, accuracy: 1e-4)
        // Closure: t=duration → back to key 0
        XCTAssertEqual(s.pose(at: 4).pos.x, 0, accuracy: 1e-4)
        XCTAssertEqual(s.pose(at: 4).pos.y, 0, accuracy: 1e-4)
    }

    // Adjacent keys at the same time (a zero-width span) — an instant jump with no NaN, then
    // a hold.
    func testManualPacingEqualTimesNoNaN() {
        let p = makePath([key(SIMD3(0, 0, 0), time: 0), key(SIMD3(1, 0, 0), time: 5),
                          key(SIMD3(2, 0, 0), time: 5)],
                         duration: 10, pacing: .manual)
        let s = PathSampler(path: p)
        for t in stride(from: 0.0, through: 10.0, by: 0.25) {
            let q = s.pose(at: t)
            XCTAssertTrue(q.pos.x.isFinite && q.pos.y.isFinite && q.pos.z.isFinite,
                          "non-finite position at t=\(t)")
        }
        XCTAssertEqual(s.pose(at: 2.5).pos.x, 0.5, accuracy: 1e-4)
        XCTAssertEqual(s.pose(at: 6).pos.x, 2, accuracy: 1e-4)      // hold after the jump
    }

    // Defense against unsorted or out-of-range times — clamping + a monotone correction
    // (running max).
    func testManualPacingUnsortedTimesDefense() {
        let p = makePath([key(SIMD3(0, 0, 0), time: 5), key(SIMD3(1, 0, 0), time: 3),
                          key(SIMD3(2, 0, 0), time: 20)],
                         duration: 10, pacing: .manual)
        let s = PathSampler(path: p)
        XCTAssertEqual(s.keyframeTimes, [5, 5, 10])   // 3→5 monotone correction, 20→10 clamp
        for t in stride(from: 0.0, through: 10.0, by: 0.5) {
            XCTAssertTrue(s.pose(at: t).pos.x.isFinite)
        }
    }

    // MARK: Mixed-sign quat chain (handed over from Task 1 review M1)

    // Even when the stored signs are mixed across a 3-key chain (q1 = −(90°z)), every span
    // takes the shortest route.
    // t=0.5 (span 0 midpoint) → 45°, t=1.5 (span 1 midpoint) → 135° — both are the halfway
    // angle of the shortest rotation about +z.
    func testMixedSignQuatChainShortestPath() {
        let negQ90z = SIMD4<Float>(-0.70710678, 0, 0, -0.70710678)
        let q180z = SIMD4<Float>(0, 0, 0, 1)
        let p = makePath([key(SIMD3(0, 0, 0), quat: SIMD4(1, 0, 0, 0)),
                          key(SIMD3(1, 0, 0), quat: negQ90z),
                          key(SIMD3(2, 0, 0), quat: q180z)],
                         duration: 2.0)
        let s = PathSampler(path: p)
        let v1 = s.pose(at: 0.5).quat.act(SIMD3<Float>(1, 0, 0))     // 45°
        XCTAssertEqual(v1.x, 0.70710678, accuracy: 1e-3)
        XCTAssertEqual(v1.y, 0.70710678, accuracy: 1e-3)
        let v2 = s.pose(at: 1.5).quat.act(SIMD3<Float>(1, 0, 0))     // 135°
        XCTAssertEqual(v2.x, -0.70710678, accuracy: 1e-3)
        XCTAssertEqual(v2.y, 0.70710678, accuracy: 1e-3)
        XCTAssertEqual(v2.z, 0, accuracy: 1e-4)
        // Passes exactly through the key: t=1 → 90°z
        let vKey = s.pose(at: 1).quat.act(SIMD3<Float>(1, 0, 0))
        XCTAssertEqual(vKey.x, 0, accuracy: 1e-3)
        XCTAssertEqual(vKey.y, 1, accuracy: 1e-3)
    }

    // MARK: Pacing Codable (the persistence format — project.json)

    func testPacingCodableRoundTripAndForwardCompat() throws {
        // manual round trip
        let manual = makePath([key(SIMD3(0, 0, 0), time: 1)], duration: 5, pacing: .manual)
        let data = try JSONEncoder().encode(manual)
        XCTAssertEqual(try JSONDecoder().decode(CameraPath.self, from: data), manual)

        // Old JSON with no pacing key → .even (forward compatibility)
        let legacy = """
        {"id": "6F1C6C2E-8C1A-4E1B-9C0D-123456789ABC", "name": "legacy", "duration": 8}
        """
        let l = try JSONDecoder().decode(CameraPath.self, from: Data(legacy.utf8))
        XCTAssertEqual(l.pacing, .even)

        // An unknown pacing string → .even (lenient decode)
        let unknown = """
        {"id": "6F1C6C2E-8C1A-4E1B-9C0D-123456789ABC", "name": "u", "duration": 8,
         "pacing": "hyperRamp"}
        """
        let u = try JSONDecoder().decode(CameraPath.self, from: Data(unknown.utf8))
        XCTAssertEqual(u.pacing, .even)

        // rawValue is the persistence format — the strings are fixed
        XCTAssertEqual(PacingMode.even.rawValue, "even")
        XCTAssertEqual(PacingMode.manual.rawValue, "manual")
        // The default is even pacing
        XCTAssertEqual(CameraPath(name: "d").pacing, .even)
    }
}
