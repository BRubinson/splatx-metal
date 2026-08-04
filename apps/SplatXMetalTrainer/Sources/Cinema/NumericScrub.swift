// Panel redesign FIX 3 ② (2026-07-29 spec footnote 3): the **pure arithmetic** of the inspector's
// numeric field drag scrubbing. Not testing SwiftUI views is this repository's explicit policy,
// so extracting every decision and computation here is the only automated line of defense
// (the source of truth for tests: Tests/NumericScrubTests.swift). The view wiring
// is ScrubbableNumberField.
//
// The interaction model follows **Unreal** (dragging the body of the field slides the value, and a
// click with no movement enters text editing), and the sensitivity curve follows **Unity**
// (proportional to the value magnitude, √|v|, with the multiplicative ⇧ acceleration ×4 / ⌥
// fine-tuning ×0.25). The user's request was "drag left/right while holding a click on the number
// input", so it is not Unity's label drag.
import CoreGraphics
import Foundation

enum NumericScrub {
    /// The scrub engagement threshold (pt). Click protection — it follows this app's precedent
    /// (3pt for timeline key retiming). The `minimumDistance: 1` of the gizmo handles is 1 because
    /// those are handle-only surfaces that need no click protection, not because it is an idiom —
    /// for a numeric field a click is exactly the entry into typing, so it is 3.
    static let engageThreshold: CGFloat = 3

    /// The commit precision — the same number of digits as the field display
    /// (`fractionLength(0...2)`).
    static let decimals = 2

    /// The floor on the increment per pt. The display has 2 digits, so anything smaller leaves the
    /// screen unchanged even after dragging several pt and reads as "it is broken".
    static let minimumStep = 0.01

    /// The value-magnitude proportionality coefficient — this app's adjusted version of Unity's
    /// `NumericFieldDraggerUtility.CalcDragSensitivity` (≈ 0.03·√|v|). 0.02 is the value that makes
    /// 100pt ≈ ×1.96 at Radius 4.29 m and 100pt ≈ +8.9 s at Duration 20 s (the measured table in
    /// footnote 3 ②).
    static let sensitivity = 0.02

    /// The increment per pt. **Proportionality to the value magnitude is required** — giving
    /// Radius 4.29 and Duration 20 the same constant makes one sluggish and the other jumpy. A
    /// non-finite input drops to the floor.
    static func step(reference: Double) -> Double {
        guard reference.isFinite else { return minimumStep }
        return max(minimumStep, sensitivity * (abs(reference)).squareRoot())
    }

    /// The modifier multiplier (multiplicative, exactly as Unity defines it): ⇧ accelerates ×4, ⌥
    /// fine-tunes ×0.25, and both together give ×1.
    /// ⚠ It is a multiplier, not a mode switch — the caller MUST read `NSEvent.modifierFlags` on
    /// **every tick** of the drag (the gizmo's "latch once on the first tick" is a different case,
    /// because there ⇧ = vertical is a mode switch. Latching here would break the convention of
    /// pressing ⌥ mid-drag to slow down).
    static func acceleration(shift: Bool, option: Bool) -> Double {
        (shift ? 4 : 1) * (option ? 0.25 : 1)
    }

    /// The scrub engagement decision — horizontal movement at or above the threshold. Only the
    /// x axis changes the value.
    static func engaged(translationX: CGFloat) -> Bool {
        abs(translationX) >= engageThreshold
    }

    /// The click (= entering text editing) decision — **2D** movement below the threshold. It is
    /// not the complement of `engaged`: a drag that moved 20pt purely vertically does not change
    /// the value, but it is not a click either (entering edit mode on release would be unintended
    /// focus).
    static func isClick(translation: CGSize) -> Bool {
        (translation.width * translation.width
         + translation.height * translation.height).squareRoot() < engageThreshold
    }

    /// Rounds to the display precision.
    static func snapped(_ value: Double) -> Double {
        let scale = pow(10.0, Double(decimals))
        return (value * scale).rounded() / scale
    }

    /// The **absolute target value** measured from the drag's starting value (not an accumulation
    /// of increments).
    ///
    /// Multiplying and accumulating increments accumulates the float32 error of
    /// `PathTransform.scaled` across every keyframe. Latching the starting value and the reference
    /// at drag start and recomputing the absolute value on every tick makes the error self-correct
    /// each tick, because the consumer setters are already absolute in meaning
    /// (`Float(r) / p.meanRadius`, `(h − dot(p.point, up))·up`) — it is the scalar version of the
    /// gizmo's `total − applied` idiom.
    ///
    /// `lowerBound` is the consumer's domain floor: 0.1 s for Duration (the same as setDuration's
    /// clamp) and 0.01 m for Radius (at or below 0 the factor s = R/R̄ turns negative and the path
    /// collapses).
    /// Height is `dot(pivot, up)`, where 0 and negative values are normal, so it has no floor.
    static func value(base: Double, translationX: CGFloat, reference: Double,
                      shift: Bool, option: Bool, lowerBound: Double? = nil) -> Double {
        guard base.isFinite else { return base }
        let delta = Double(translationX) * step(reference: reference)
            * acceleration(shift: shift, option: option)
        let next = snapped(base + delta)
        guard let lowerBound else { return next }
        return max(next, lowerBound)
    }

    /// The sensitivity reference used only by Height — it uses **the scene scale, not the value**.
    ///
    /// Height is `dot(pivot.point, upAxis)`, where 0 and negative values are normal. Using
    /// `√|value|` like Radius and Duration would drop step to the 0.01 m/pt floor whenever the
    /// height is near 0, even in a 10 m scene, making it unusable. So the reference is the pivot
    /// radius (= the scale of the scene itself), with a 1 m floor so it does not die on a still
    /// shot (R̄ == 0).
    static func heightReference(meanRadius: Float) -> Double {
        let r = Double(abs(meanRadius))
        return r.isFinite ? max(r, 1) : 1
    }

    /// The sensitivity reference used only by Radius (FIX 4 ④) — it uses the value itself (R̄) but
    /// lays a **scene-scale floor** under it.
    ///
    /// Previously R̄ was used as the reference directly: dragging down to the 0.01 m domain floor
    /// and releasing made the next drag latch reference 0.01 → step fell to the 0.01 m/pt floor,
    /// and returning to the scene scale (4.29 m, for example) required ~430pt in one go (beyond
    /// the 240–400pt inspector width) — the affordance defect of being able to drag into a state
    /// you cannot drag out of. With the floor in place, even in the worst case 0.02 m/pt (⇧ ×4 =
    /// 0.08 m/pt) is guaranteed and the return drops to ~214pt (~54pt with ⇧).
    /// At a normal scale (R̄ ≥ 1 m) the reference is still exactly R̄, so the sensitivity is
    /// completely identical to before.
    /// The floor arithmetic is the same max(R̄, 1 m) as heightReference — it is the same device as
    /// the still-shot floor, so it delegates (the reasons differ: there the value cannot serve as
    /// the reference; here it prevents the reference from collapsing).
    static func radiusReference(meanRadius: Float) -> Double {
        heightReference(meanRadius: meanRadius)
    }
}
