// Cinema Task 3: the pure math of the timeline — the time↔x mapping, diamond hit testing,
// snapping, frame stepping, and the tick interval. The Canvas view consumes only these
// functions (which is what makes the no-tests-for-view-logic policy workable).
// Lineage of the literals: width 424 − insets 12×2 = a usable width of 400 → 40 px/s at
// duration 10.
import XCTest
@testable import SplatXMetalTrainer

final class TimelineMathTests: XCTestCase {

    // MARK: Time ↔ x (a linear mapping that includes the insets)

    func testTimeToXMapping() {
        XCTAssertEqual(TimelineMath.x(time: 0, duration: 10, width: 424), 12, accuracy: 1e-9)
        XCTAssertEqual(TimelineMath.x(time: 10, duration: 10, width: 424), 412, accuracy: 1e-9)
        XCTAssertEqual(TimelineMath.x(time: 5, duration: 10, width: 424), 212, accuracy: 1e-9)
        // Times outside the range are clamped
        XCTAssertEqual(TimelineMath.x(time: -3, duration: 10, width: 424), 12, accuracy: 1e-9)
        XCTAssertEqual(TimelineMath.x(time: 99, duration: 10, width: 424), 412, accuracy: 1e-9)
        // duration 0 guard — pinned to the left inset
        XCTAssertEqual(TimelineMath.x(time: 1, duration: 0, width: 424), 12, accuracy: 1e-9)
    }

    func testXToTimeInverseAndClamp() {
        // Round-trip identity
        for t in [0.0, 1.7, 5.0, 10.0] {
            let x = TimelineMath.x(time: t, duration: 10, width: 424)
            XCTAssertEqual(TimelineMath.time(x: x, duration: 10, width: 424), t,
                           accuracy: 1e-9, "t=\(t)")
        }
        // Coordinates outside the insets are clamped to [0, duration]
        XCTAssertEqual(TimelineMath.time(x: 0, duration: 10, width: 424), 0, accuracy: 1e-9)
        XCTAssertEqual(TimelineMath.time(x: 500, duration: 10, width: 424), 10, accuracy: 1e-9)
    }

    func testTimePerPixel() {
        // usable width 400 / duration 10 → 0.025 s/px
        XCTAssertEqual(TimelineMath.timePerPixel(duration: 10, width: 424), 0.025,
                       accuracy: 1e-9)
    }

    // MARK: Diamond hit testing (time→x math — the substitute for the contract that forbids
    // one View per diamond)

    func testHitIndexWithinRadius() {
        let times = [0.0, 5.0, 10.0]   // x = 12, 212, 412
        XCTAssertEqual(TimelineMath.hitIndex(x: 215, times: times, duration: 10, width: 424), 1)
        XCTAssertEqual(TimelineMath.hitIndex(x: 212, times: times, duration: 10, width: 424), 1)
        XCTAssertNil(TimelineMath.hitIndex(x: 220, times: times, duration: 10, width: 424))
        XCTAssertEqual(TimelineMath.hitIndex(x: 13, times: times, duration: 10, width: 424), 0)
        XCTAssertNil(TimelineMath.hitIndex(x: 100, times: [], duration: 10, width: 424))
    }

    func testHitIndexPicksNearestWhenOverlapping() {
        // 5.0 → x=212, 5.1 → x=216: x=213.5 is inside the radius of both — the nearest wins
        // (index 0)
        let times = [5.0, 5.1]
        XCTAssertEqual(TimelineMath.hitIndex(x: 213.5, times: times, duration: 10, width: 424), 0)
        XCTAssertEqual(TimelineMath.hitIndex(x: 215, times: times, duration: 10, width: 424), 1)
    }

    // MARK: Snapping

    func testSnappedToNearestCandidateWithinThreshold() {
        XCTAssertEqual(TimelineMath.snapped(4.9, candidates: [5], threshold: 0.15), 5)
        XCTAssertEqual(TimelineMath.snapped(4.9, candidates: [5], threshold: 0.05), 4.9)
        // Multiple candidates — the nearest wins
        XCTAssertEqual(TimelineMath.snapped(4.9, candidates: [5, 4.85], threshold: 0.15), 4.85)
        XCTAssertEqual(TimelineMath.snapped(3, candidates: [], threshold: 1), 3)
    }

    // MARK: Snap candidates for a key drag (FIX 1 — Task 3 review I2: preventing a
    // self-referential snap)

    // The candidates = the tick grid + every key other than the dragged one. During a drag the
    // playhead is a value that follows the dragged key, so including it as a candidate would
    // snap the key to its own previous position — do not include it.
    func testKeyDragSnapCandidatesExcludePlayheadAndSelf() {
        let candidates = TimelineMath.keyDragSnapCandidates(
            duration: 3, step: 1, keyframeTimes: [0.0, 1.37, 2.0], draggedIndex: 1)
        XCTAssertEqual(candidates, [0, 1, 2, 3, 0, 2])    // ticks [0,1,2,3] + the other keys [0,2]
        XCTAssertFalse(candidates.contains(1.37))          // itself (= the playhead during the drag) is absent
    }

    // MARK: Frame stepping (←→ — round to the frame grid, then ±1)

    func testFrameStepArithmetic() {
        // 0.5s @30fps = frame 15 → +1 = 16/30
        XCTAssertEqual(TimelineMath.frameStep(from: 0.5, delta: 1, fps: 30, duration: 10),
                       16.0 / 30.0, accuracy: 1e-12)
        XCTAssertEqual(TimelineMath.frameStep(from: 0.5, delta: -1, fps: 30, duration: 10),
                       14.0 / 30.0, accuracy: 1e-12)
        // A time off the grid is rounded to the nearest frame and then stepped (0.49 →
        // frame 15 → 16)
        XCTAssertEqual(TimelineMath.frameStep(from: 0.49, delta: 1, fps: 30, duration: 10),
                       16.0 / 30.0, accuracy: 1e-12)
        // Boundary clamps
        XCTAssertEqual(TimelineMath.frameStep(from: 0.01, delta: -1, fps: 30, duration: 10),
                       0, accuracy: 1e-12)
        XCTAssertEqual(TimelineMath.frameStep(from: 9.99, delta: 1, fps: 30, duration: 10),
                       10, accuracy: 1e-12)
        // fps 0 guard — no change
        XCTAssertEqual(TimelineMath.frameStep(from: 3, delta: 1, fps: 0, duration: 10), 3)
    }

    // MARK: Tick interval (the densest nice step that satisfies the minimum label spacing)

    func testTickStepPicksDensestNiceStep() {
        // duration 10, usable width 400: 1s = 40px < 48 → 2s = 80px ✓
        XCTAssertEqual(TimelineMath.tickStep(duration: 10, width: 424), 2)
        // duration 5: 1s = 80px ✓
        XCTAssertEqual(TimelineMath.tickStep(duration: 5, width: 424), 1)
        // duration 2: 0.25s = 50px ✓
        XCTAssertEqual(TimelineMath.tickStep(duration: 2, width: 424), 0.25)
        // duration 0 guard
        XCTAssertEqual(TimelineMath.tickStep(duration: 0, width: 424), 1)
    }
}
