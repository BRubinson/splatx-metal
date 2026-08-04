// Path transform Task 1 (2026-07-27 spec §3 and §4): the pure math of the whole-path transform.
// Written in-house — it uses only standard simd, with no derivation from external code.
//
// - The pivot (§3.1–3.2): it solves for the convergence point of the view rays (o_i = pos,
//   d_i = quat·(0,0,−1)) with the least-squares normal equations A p = b
//   (A = Σ(I − d_i d_iᵀ), b = Σ(I − d_i d_iᵀ)o_i), and the aim is fixed only when all three
//   conditions hold (keys ≥ 2 ∧ det(A/n) > 1e-3, the residual ratio r/R̄ ≤ 0.10, and forwardness
//   mean (p−o)·d > 0) — then the pivot = the convergence point; otherwise it is the centroid.
// - Translation (§3.3): +delta on pos only; quat and fov are untouched. Re-aiming is a **no-op** —
//   a translation moves the pivot by the same delta, so the camera→pivot vector is unchanged and
//   the new pivot is aimed at exactly without updating the rotation (the basis for not creating a
//   re-aim branch or toggle).
// - Scaling (§3.4): pos ← pivot + (pos − pivot)·s; quat and fov are untouched (the pivot→camera
//   direction is unchanged → the view direction is preserved). s is clamped to [0.05, 20], and if
//   the resulting R̄ < 1e-3 the operation is refused (nil).
// - The drag arithmetic (§4.1–4.2): pure conversion with no unprojection — k =
//   2·tan(fov_v/2)·distance/height; a basic drag uses the orbit-plane (⊥ upAxis) basis (ê_r, ê_f)
//   as Δworld = k·(Δx·ê_r − Δy·ê_f) (screen y is positive downward → cursor up = forward), and ⇧
//   gives k·(−Δy)·upAxis. Scaling is a pure 2D ratio (a denominator under 8pt is ignored).
import Foundation
import simd

enum PathTransform {

    /// The pivot decision result. `meanRadius` is the arithmetic mean distance from the key
    /// positions to `point` (R̄).
    struct Pivot {
        var point: SIMD3<Float>
        var isAimLocked: Bool
        var meanRadius: Float
    }

    // The §3.1 decision thresholds — the spec constants verbatim.
    private static let detThreshold = 1e-3          // the det(A/n) floor (excluding parallel view rays)
    private static let residualRatioMax = 0.10      // the r / R̄ ceiling
    private static let scaleClamp: ClosedRange<Float> = 0.05...20      // §3.4
    private static let minMeanRadius: Float = 1e-3                     // the §3.4 resulting-R̄ floor
    private static let minScaleDenominator: CGFloat = 8                // the §4.2 denominator floor (pt)

    // MARK: The pivot (§3.1–3.2)

    /// 0 keys → nil. If all three aim-lock conditions hold, the pivot = the convergence point;
    /// otherwise the pivot = the centroid. The accumulation and the solve are in double (guarding
    /// the condition number of the normal equations).
    static func pivot(for keys: [Keyframe]) -> Pivot? {
        guard !keys.isEmpty else { return nil }
        let origins = keys.map { SIMD3<Double>($0.pos) }
        let dirs = keys.map {
            SIMD3<Double>(simd_normalize($0.quatValue.act(SIMD3<Float>(0, 0, -1))))
        }
        let n = Double(keys.count)
        let identity = simd_double3x3(1)

        var A = simd_double3x3()   // the zero matrix
        var b = SIMD3<Double>()
        for (o, d) in zip(origins, dirs) {
            // I − d dᵀ (column j = d·d_j — it is symmetric, so columns and rows are equivalent)
            let m = identity - simd_double3x3(columns: (d * d.x, d * d.y, d * d.z))
            A += m
            b += m * o
        }

        // Condition 1: keys ≥ 2 ∧ det(A/n) > 1e-3 — the eigenvalues of (I − d dᵀ) are (0,1,1), so
        // the eigenvalues of A/n lie in [0,1]. If all view rays are parallel, A is singular → it is
        // caught here.
        if keys.count >= 2, ((1 / n) * A).determinant > detThreshold {
            let p = A.inverse * b
            // Condition 2: the residual ratio ≤ 0.10 (measured against infinite lines — direction
            // is condition 3's job)
            var residual = 0.0, radius = 0.0, forwardness = 0.0
            for (o, d) in zip(origins, dirs) {
                let v = p - o
                residual += simd_length(v - simd_dot(v, d) * d)
                radius += simd_length(v)
                forwardness += simd_dot(v, d)   // the numerator of condition 3
            }
            residual /= n; radius /= n; forwardness /= n
            if radius > 0, residual / radius <= residualRatioMax, forwardness > 0 {
                return Pivot(point: SIMD3<Float>(p), isAimLocked: true,
                             meanRadius: Float(radius))
            }
        }

        // The fallback (§3.2): the centroid — the pivot for a path whose aim is not locked.
        let centroid = origins.reduce(SIMD3<Double>()) { $0 + $1 } / n
        let meanRadius = origins.reduce(0.0) { $0 + simd_distance($1, centroid) } / n
        return Pivot(point: SIMD3<Float>(centroid), isAimLocked: false,
                     meanRadius: Float(meanRadius))
    }

    // MARK: Transforms (§3.3–3.4)

    /// Translation — it moves pos only; quat and fov are untouched (§3.3: re-aiming is a no-op).
    static func translated(_ path: CameraPath, by delta: SIMD3<Float>) -> CameraPath {
        var out = path
        out.keyframes = path.keyframes.map { k in
            var k = k
            k.pos += delta
            return k
        }
        return out
    }

    /// Concentric scaling — pos ← pivot + (pos − pivot)·s; quat and fov are untouched (§3.4).
    /// factor is clamped to [0.05, 20]. If the resulting R̄ < 1e-3 (including the 0-key case) the
    /// operation is refused → nil.
    static func scaled(_ path: CameraPath, by factor: Float,
                       about pivot: SIMD3<Float>) -> CameraPath? {
        let s = min(max(factor, scaleClamp.lowerBound), scaleClamp.upperBound)
        let keys = path.keyframes
        guard !keys.isEmpty else { return nil }
        let meanRadius = keys.reduce(Float(0)) { $0 + simd_distance($1.pos, pivot) }
            / Float(keys.count)
        guard meanRadius * s >= minMeanRadius else { return nil }
        var out = path
        out.keyframes = keys.map { k in
            var k = k
            k.pos = pivot + (k.pos - pivot) * s
            return k
        }
        return out
    }

    // MARK: Handle drag arithmetic (§4.1–4.2)

    /// The world direction û of the radius handle — cameraRight projected onto the orbit plane
    /// (⊥ upAxis) and normalized.
    /// On a degeneracy (right ∥ upAxis) it falls back to an arbitrary unit vector perpendicular to
    /// upAxis (the same reference axis choice as orbitBasis — +X when upAxis≈±Z, otherwise +Z).
    static func handleDirection(cameraRight: SIMD3<Float>,
                                upAxis: SIMD3<Float>) -> SIMD3<Float> {
        let up = unitUp(upAxis)
        if let planar = planarUnit(cameraRight, up: up) { return planar }
        let ref: SIMD3<Float> = abs(up.z) < 0.99 ? SIMD3(0, 0, 1) : SIMD3(1, 0, 0)
        return simd_normalize(simd_cross(up, ref))
    }

    /// The screen-pixel → world conversion factor (§4.1):
    /// k = 2·tan(fov_v/2)·|pivot − eye| / viewportHeight.
    /// It is the factor for which "a 1pt drag = 1pt of movement on screen" holds at the pivot's
    /// depth.
    static func worldPerPoint(verticalFOV: Float, pivotDistance: Float,
                              viewportHeight: CGFloat) -> Float {
        guard viewportHeight > 0 else { return 0 }
        return 2 * tan(verticalFOV / 2) * pivotDistance / Float(viewportHeight)
    }

    /// The world translation of a center handle drag (§4.1). Screen y is positive downward.
    /// - Default (vertical=false): Δworld = k·(Δx·ê_r − Δy·ê_f) — ê_r is the camera right
    ///   projected onto the orbit plane (⊥ upAxis) and normalized, and ê_f = cross(up, ê_r).
    ///   Cursor up = camera forward.
    /// - ⇧ (vertical=true): Δworld = k·(−Δy)·upAxis — the horizontal component is ignored, and
    ///   cursor up = rising.
    static func dragTranslation(screenDelta: CGSize, vertical: Bool, worldPerPoint: Float,
                                cameraRight: SIMD3<Float>, cameraForward: SIMD3<Float>,
                                upAxis: SIMD3<Float>) -> SIMD3<Float> {
        let up = unitUp(upAxis)
        let dy = Float(screenDelta.height)
        if vertical {
            // A known limitation (FIX 19 ④ — no behavior change): when the camera looks exactly up
            // or down (the view direction ∥ upAxis) the movement becomes parallel to the view axis
            // and appears to do almost nothing on screen.
            // The horizontal branch removed the same degeneracy by deriving ê_f as cross(up, ê_r)
            // (see the comment below), but vertical movement is fixed to upAxis by definition and
            // has no corresponding device — if a ⇧ drag seems not to work in a top view, tilt the
            // camera slightly.
            return worldPerPoint * -dy * up
        }
        let dx = Float(screenDelta.width)
        let er = planarUnit(cameraRight, up: up)
        // ê_f is derived as cross(up, ê_r) rather than as the orbit-plane projection of forward —
        // ê_r is already a unit vector perpendicular to up, so cross(up, ê_r) is always unit and
        // perpendicular to both up and ê_r.
        // At an ordinary roll-free viewpoint it is the same vector as the horizontal projection of
        // forward (forward_h = cross(up, right)), so the existing behavior is unchanged, and only
        // the degeneracy of a top view (forward ∥ up → a zero projection → a vertical drag doing
        // nothing; a real-use bug on 2026-07-28) disappears. Sign check:
        // up=(0,1,0), ê_r=(1,0,0) → cross=(0,0,−1) = the forward direction of a camera looking
        // along −Z ✓.
        // Only when ê_r itself is nil (a 90° roll: right ∥ up) does it fall back to the previous
        // forward projection.
        let ef = er.map { simd_cross(up, $0) }
            ?? planarUnit(cameraForward, up: up) ?? .zero
        return worldPerPoint * (dx * (er ?? .zero) - dy * ef)
    }

    /// The factor of a radius handle drag (§4.2): s = |cursor − pivot| / |handleStart − pivot| — a
    /// pure 2D ratio (independent of the camera angle and the orbit tilt). A denominator under 8pt
    /// (an on-screen overlap) → nil.
    static func dragScaleFactor(cursor: CGPoint, pivotScreen: CGPoint,
                                handleStart: CGPoint) -> Float? {
        let denominator = hypot(handleStart.x - pivotScreen.x, handleStart.y - pivotScreen.y)
        guard denominator >= minScaleDenominator else { return nil }
        let numerator = hypot(cursor.x - pivotScreen.x, cursor.y - pivotScreen.y)
        return Float(numerator / denominator)
    }

    // MARK: Internal helpers

    /// upAxis normalization — a degenerate norm≈0 falls back to +Y (the same defense as
    /// CameraController.orbitBasis).
    private static func unitUp(_ upAxis: SIMD3<Float>) -> SIMD3<Float> {
        let n = simd_length(upAxis)
        return n > 1e-6 ? upAxis / n : SIMD3(0, 1, 0)
    }

    /// v projected onto the orbit plane (⊥ up) and normalized. nil on a degeneracy (v ∥ up).
    private static func planarUnit(_ v: SIMD3<Float>, up: SIMD3<Float>) -> SIMD3<Float>? {
        let planar = v - simd_dot(v, up) * up
        let len = simd_length(planar)
        guard len > 1e-5 else { return nil }
        return planar / len
    }
}
