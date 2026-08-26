import Foundation
import simd

/// Estimates the scene up axis as the mean of the world up vectors of the COLMAP registered views
/// (2026-07-26 spec decision 2).
///
/// The COLMAP world is not gravity-aligned (an arbitrary coordinate system — the MipNeRF360
/// convention is up ≈ −Y), so assuming the viewer's orbit up axis is +Y flips the initial view
/// upside down (garden, bonsai). The mean of `CameraConversion.upWorld` over all registered views
/// is the estimate of the scene up — applyCameraFit wires that value into
/// `CameraController.upAxis` and `OrbitFit.fit(upAxis:)`.
///
/// **Correction (2026-08-04).** This file used to state that the scene data is never reoriented
/// ("a non-goal — the export coordinates are preserved"). That is no longer true: `GroundLevel`
/// rotates the poses and points at align write time, so `sparse/0` — and everything downstream of
/// it — is written gravity-aligned (2026-08-04 spec §1; the 2026-07-26 spec carries the same
/// correction as a footnote). The reason the reversal is safe is that the target is the COLMAP
/// convention's up, `(0,−1,0)`, not three.js `+Y`.
///
/// The estimate below still runs, and on a levelled project it converges on `(0,−1,0)` by
/// construction — `up'ᵢ = G·upᵢ`, so the normalized mean is `G·up = gravityUp`. That is why the
/// correction and this orbit axis do not compound (`GroundLevelTests` pins it).
enum SceneUp {
    /// The normalized mean of upWorld over all views. On a degeneracy (an empty list, or a norm≈0
    /// from cancellation) it falls back to +Y.
    /// It sums in Double to avoid Float accumulation error over many views (the ColmapContext.load
    /// centroid idiom).
    static func estimate(views: [ColmapView]) -> SIMD3<Float> {
        var sum = SIMD3<Double>()
        for v in views { sum += SIMD3<Double>(CameraConversion.upWorld(qvec: v.qvec)) }
        let n = simd_length(sum)
        guard n > 1e-6 else { return SIMD3(0, 1, 0) }
        return SIMD3<Float>(sum / n)
    }
}
