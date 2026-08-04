import Foundation
import simd

/// Estimates the scene up axis as the mean of the world up vectors of the COLMAP registered views
/// (2026-07-26 spec decision 2).
///
/// The COLMAP world is not gravity-aligned (an arbitrary coordinate system — the MipNeRF360
/// convention is up ≈ −Y), so assuming the viewer's orbit up axis is +Y flips the initial view
/// upside down (garden, bonsai). The mean of `CameraConversion.upWorld` over all registered views
/// is the estimate of the scene up — applyCameraFit wires that value into
/// `CameraController.upAxis` and `OrbitFit.fit(upAxis:)`. The scene data itself is not reoriented
/// (a non-goal — the export coordinates are preserved).
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
