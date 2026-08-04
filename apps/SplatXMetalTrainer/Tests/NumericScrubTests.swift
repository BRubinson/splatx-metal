// Panel redesign FIX 3 ② (2026-07-29 spec footnote 3): the pure arithmetic contract of drag
// scrubbing on the inspector's numeric fields. Not testing SwiftUI views (overlay hit testing,
// focus transitions, the cursor) is this repository's explicit policy, so this is the **only
// automated line of defense**. That clicking, typing, and text selection coexist is confirmed
// only by the manual pass checklist.
import CoreGraphics
import XCTest
@testable import SplatXMetalTrainer

final class NumericScrubTests: XCTestCase {

    // MARK: Sensitivity — proportional to the magnitude of the value

    /// The increment per pt must be proportional to the magnitude of the value. With a
    /// constant step, one of Radius 4.29 m and Duration 20 s ends up sluggish and the other
    /// jumpy (the rationale for adopting this, footnote 3 ②).
    func testStepScalesWithValueMagnitude() {
        XCTAssertEqual(NumericScrub.step(reference: 4.29), 0.02 * 4.29.squareRoot(),
                       accuracy: 1e-12)
        XCTAssertEqual(NumericScrub.step(reference: 20), 0.02 * 20.0.squareRoot(),
                       accuracy: 1e-12)
        // Measured expectations (the footnote 3 ② table): Radius 4.29 → ≈0.0414 m/pt,
        // Duration 20 → ≈0.0894 s/pt
        XCTAssertEqual(NumericScrub.step(reference: 4.29), 0.041424, accuracy: 1e-5)
        XCTAssertEqual(NumericScrub.step(reference: 20), 0.089442, accuracy: 1e-5)
        XCTAssertEqual(NumericScrub.step(reference: 2), 0.028284, accuracy: 1e-5)
        // The larger the value, the larger the step must be (monotone)
        XCTAssertGreaterThan(NumericScrub.step(reference: 100),
                             NumericScrub.step(reference: 10))
    }

    /// The floor is 0.01, the same as the display precision (2 decimals) — below that,
    /// dragging several pt leaves the display unchanged and reads as "it is broken". Signed and
    /// non-finite inputs converge to the floor as well.
    func testStepFloorsAtDisplayPrecision() {
        XCTAssertEqual(NumericScrub.step(reference: 0), NumericScrub.minimumStep)
        XCTAssertEqual(NumericScrub.step(reference: 0.01), NumericScrub.minimumStep)
        // A negative reference is taken by magnitude only (Height can be negative)
        XCTAssertEqual(NumericScrub.step(reference: -4.29),
                       NumericScrub.step(reference: 4.29), accuracy: 1e-12)
        XCTAssertEqual(NumericScrub.step(reference: .nan), NumericScrub.minimumStep)
        XCTAssertEqual(NumericScrub.step(reference: .infinity), NumericScrub.minimumStep)
    }

    /// ⇧ accelerates ×4, ⌥ fine-tunes ×0.25, and they multiply (Unity is the reference).
    func testAccelerationMultipliersAreMultiplicative() {
        XCTAssertEqual(NumericScrub.acceleration(shift: false, option: false), 1)
        XCTAssertEqual(NumericScrub.acceleration(shift: true, option: false), 4)
        XCTAssertEqual(NumericScrub.acceleration(shift: false, option: true), 0.25)
        XCTAssertEqual(NumericScrub.acceleration(shift: true, option: true), 1)
    }

    // MARK: Click protection — the condition under which typing is not lost

    /// A scrub engages at 3pt or more **horizontally**. Below the threshold the value is not
    /// touched — this is the premise of "a click (no movement) still enters editing as
    /// before".
    func testEngageRequiresHorizontalThreshold() {
        XCTAssertEqual(NumericScrub.engageThreshold, 3)
        XCTAssertFalse(NumericScrub.engaged(translationX: 0))
        XCTAssertFalse(NumericScrub.engaged(translationX: 2.9))
        XCTAssertFalse(NumericScrub.engaged(translationX: -2.9))
        XCTAssertTrue(NumericScrub.engaged(translationX: 3))
        XCTAssertTrue(NumericScrub.engaged(translationX: -3))
    }

    /// The click decision uses the **2D** distance — it is not the complement of `engaged`. A
    /// drag that moves 20pt vertically only must not change the value, but it must not count as
    /// a click either (if focus landed on release it would be an unintended entry into
    /// editing).
    func testIsClickUsesTwoDimensionalSlop() {
        XCTAssertTrue(NumericScrub.isClick(translation: .zero))
        XCTAssertTrue(NumericScrub.isClick(translation: CGSize(width: 2, height: 2)))
        XCTAssertFalse(NumericScrub.isClick(translation: CGSize(width: 3, height: 0)))
        // The band below the horizontal threshold that is still not a click (a vertical drag)
        XCTAssertFalse(NumericScrub.engaged(translationX: 0))
        XCTAssertFalse(NumericScrub.isClick(translation: CGSize(width: 0, height: 20)))
    }

    // MARK: Value conversion

    func testSnappedToDisplayPrecision() {
        XCTAssertEqual(NumericScrub.decimals, 2)
        XCTAssertEqual(NumericScrub.snapped(8.432463), 8.43, accuracy: 1e-12)
        XCTAssertEqual(NumericScrub.snapped(-0.004), 0, accuracy: 1e-12)
        XCTAssertEqual(NumericScrub.snapped(0.005), 0.01, accuracy: 1e-12)
    }

    /// Absolute-target arithmetic (relative to the latched starting value) — not accumulated
    /// increments.
    /// From Radius 4.29, 100pt to the right = +4.14 m (≈ ×1.96), and 100pt to the left is its
    /// mirror image.
    func testValueIsAbsoluteFromLatchedBase() {
        let base = 4.29
        let right = NumericScrub.value(base: base, translationX: 100, reference: base,
                                       shift: false, option: false)
        XCTAssertEqual(right, 8.43, accuracy: 1e-12)
        let left = NumericScrub.value(base: base, translationX: -100, reference: base,
                                      shift: false, option: false)
        XCTAssertEqual(left, 0.15, accuracy: 1e-12)
        // A round trip (returning to the origin within the same drag) → exactly the starting
        // value. This is the key property of the absolute-target approach, and it does not hold
        // if increments are multiplied and accumulated.
        XCTAssertEqual(NumericScrub.value(base: base, translationX: 0, reference: base,
                                          shift: false, option: false),
                       base, accuracy: 1e-12)
    }

    /// The measured expectations for Duration 20 s and 2 s (the footnote 3 ② table) — as the
    /// value shrinks, so does the sensitivity.
    func testValueForDurationScenarios() {
        XCTAssertEqual(NumericScrub.value(base: 20, translationX: 100, reference: 20,
                                          shift: false, option: false),
                       28.94, accuracy: 1e-12)
        XCTAssertEqual(NumericScrub.value(base: 2, translationX: 100, reference: 2,
                                          shift: false, option: false),
                       4.83, accuracy: 1e-12)
    }

    /// The modifiers multiply only the delta (never the starting value).
    /// The comparison tolerance is twice one display-snap quantum (0.01) — because the results
    /// are rounded to 2 decimals, the multiplicative relation does not hold exactly (each term
    /// wobbles by up to ±0.005).
    func testModifiersScaleOnlyTheDelta() {
        let base = 4.29
        let plain = NumericScrub.value(base: base, translationX: 100, reference: base,
                                       shift: false, option: false)
        let fine = NumericScrub.value(base: base, translationX: 100, reference: base,
                                      shift: false, option: true)
        let fast = NumericScrub.value(base: base, translationX: 100, reference: base,
                                      shift: true, option: false)
        // Measured expectations (the footnote 3 ② table): plain +4.14 / ⌥ +1.04 / ⇧ +16.57
        // (a 100pt drag)
        XCTAssertEqual(plain, 8.43, accuracy: 1e-12)
        XCTAssertEqual(fine, 5.33, accuracy: 1e-12)
        XCTAssertEqual(fast, 20.86, accuracy: 1e-12)
        XCTAssertEqual(fine - base, (plain - base) * 0.25, accuracy: 0.02)
        XCTAssertEqual(fast - base, (plain - base) * 4, accuracy: 0.02)
        XCTAssertLessThan(fine, plain)
        XCTAssertLessThan(plain, fast)
        // Even ⌥ fine-tuning must stay well above the display floor (0.01/pt) to be usable in
        // practice
        XCTAssertGreaterThanOrEqual(fine - base, 1.0)
    }

    /// Domain lower bounds: Duration 0.1 s (the same as the setDuration clamp) and Radius
    /// 0.01 m (at or below 0 the factor s = R/R̄ turns negative and the path collapses). Height
    /// has no lower bound — it is `dot(pivot, up)`, so 0 and negative values are normal.
    func testValueRespectsLowerBound() {
        XCTAssertEqual(NumericScrub.value(base: 2, translationX: -1000, reference: 2,
                                          shift: false, option: false, lowerBound: 0.1),
                       0.1, accuracy: 1e-12)
        XCTAssertEqual(NumericScrub.value(base: 4.29, translationX: -1000, reference: 4.29,
                                          shift: false, option: false, lowerBound: 0.01),
                       0.01, accuracy: 1e-12)
        // No lower bound → negative values pass through (Height)
        XCTAssertLessThan(NumericScrub.value(base: 0, translationX: -100, reference: 4,
                                             shift: false, option: false),
                          0)
    }

    func testNonFiniteBasePassesThrough() {
        XCTAssertTrue(NumericScrub.value(base: .nan, translationX: 100, reference: 1,
                                         shift: false, option: false).isNaN)
    }

    // MARK: The Height-specific reference — an app-specific decision

    /// The sensitivity reference for Height is **the scene scale (the pivot radius), not the
    /// value**. If the value were used, then even in a 10 m scene the step would drop to the
    /// 0.01 m/pt floor whenever the height is near 0, making it unusable.
    func testHeightReferenceUsesSceneScaleNotValue() {
        XCTAssertEqual(NumericScrub.heightReference(meanRadius: 10), 10, accuracy: 1e-12)
        // A 1 m floor so it does not die even on a static shot (R̄ == 0)
        XCTAssertEqual(NumericScrub.heightReference(meanRadius: 0), 1, accuracy: 1e-12)
        XCTAssertEqual(NumericScrub.heightReference(meanRadius: 0.2), 1, accuracy: 1e-12)
        XCTAssertEqual(NumericScrub.heightReference(meanRadius: .nan), 1, accuracy: 1e-12)
        // Dragging height 0 in a large scene still yields usable sensitivity (with the value
        // as the reference it would die at 0.01/pt)
        let step = NumericScrub.step(
            reference: NumericScrub.heightReference(meanRadius: 10))
        XCTAssertGreaterThan(step, NumericScrub.minimumStep * 5)
    }

    // MARK: The Radius reference — a scene-scale floor (FIX 4 ④)

    /// The Radius sensitivity reference is the value itself (R̄) with a **1 m scene-scale
    /// floor** underneath it. Previously (the reference was R̄ as-is), dragging down to the
    /// 0.01 m floor and releasing made the next drag latch a reference of 0.01 → a step of
    /// 0.01 m/pt → returning to the scene scale (4.29 m) required ~430pt, which exceeds the
    /// inspector width — a state you got into by dragging must be one you can get out of by
    /// dragging.
    func testRadiusReferenceFloorsAtSceneScale() {
        // Normal scale — proportionality to the value is kept (exactly the previous
        // sensitivity)
        XCTAssertEqual(NumericScrub.radiusReference(meanRadius: 4.29), 4.29,
                       accuracy: 1e-6)
        XCTAssertEqual(NumericScrub.radiusReference(meanRadius: 10), 10, accuracy: 1e-6)
        // The drag right after being dragged down to the floor — the reference bottoms out
        // at 1 m, guaranteeing a step of 0.02 m/pt
        XCTAssertEqual(NumericScrub.radiusReference(meanRadius: 0.01), 1, accuracy: 1e-12)
        let step = NumericScrub.step(
            reference: NumericScrub.radiusReference(meanRadius: 0.01))
        XCTAssertEqual(step, 0.02, accuracy: 1e-12)
        // Returning to the scene scale now fits within the inspector's minimum width (240pt):
        // ~214pt (previously ~428pt)
        XCTAssertLessThan((4.29 - 0.01) / step, 240)
        // Non-finite inputs converge to the floor too (the same defense as heightReference)
        XCTAssertEqual(NumericScrub.radiusReference(meanRadius: .nan), 1, accuracy: 1e-12)
    }
}
