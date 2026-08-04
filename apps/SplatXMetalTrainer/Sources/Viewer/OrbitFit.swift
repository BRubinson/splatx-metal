import Foundation
import simd

/// Inverts a COLMAP camera pose into CameraController orbit parameters (2026-07-20 spec §6,
/// generalized over upAxis by the 2026-07-26 viewer up-axis spec).
///
/// The CameraController convention (measured — `eye` and `orbitBasis` in CameraController.swift):
///   eye = target + dir·distance, dir = cos p·sin y·bx + sin p·up + cos p·cos y·bz
///   ((bx,up,bz) = orbitBasis(upAxis), p=pitch, y=yaw)
/// dir is the target→eye direction, so the view forward f = normalize(target − eye) = −dir.
/// The inversion projects dir onto the basis: pitch = asin(dir·up), yaw = atan2(dir·bx, dir·bz).
/// With the default upAxis (0,1,0) that is the standard basis, so it is equivalent to the previous
/// +Y formulas (asin(−f.y), atan2(−f.x, −f.z)) — no regression for existing calls and tests.
/// (roll is ignored by the orbit model — spec §6.)
enum OrbitFit {
    struct Result: Equatable {
        var target: SIMD3<Float>
        var distance: Float
        var yaw: Float
        var pitch: Float
    }

    /// C = the camera position, f = the view direction. d = clamp((centroid − C)·f, 0.5, ∞),
    /// target = C + f·d, distance = d, and yaw/pitch are derived by projecting f onto
    /// orbitBasis(upAxis).
    /// `up` (the camera's own up) is unused under the roll-ignoring policy (spec §6) — upAxis is
    /// authoritative for the scene up (2026-07-26 spec decision 3: applyCameraFit passes the
    /// SceneUp estimate).
    static func fit(cameraPosition C: SIMD3<Float>,
                    forward: SIMD3<Float>,
                    up: SIMD3<Float>,
                    pointsCentroid: SIMD3<Float>,
                    upAxis: SIMD3<Float> = SIMD3(0, 1, 0)) -> Result {
        _ = up
        let f = simd_normalize(forward)
        let d = max(0.5, simd_dot(pointsCentroid - C, f))
        let dir = -f                                   // the target→eye direction (the CameraController convention)
        let (bx, bu, bz) = CameraController.orbitBasis(upAxis: upAxis)
        let pitch = asin(max(-1, min(1, simd_dot(dir, bu))))
        let yaw = atan2(simd_dot(dir, bx), simd_dot(dir, bz))   // atan2(x,z) in the standard basis
        return Result(target: C + f * d, distance: d, yaw: yaw, pitch: pitch)
    }

    /// The vertical FOV = 2·atan(H / (2·fy)) — derived from the COLMAP intrinsics' fy and the image
    /// height (spec §6).
    static func verticalFOV(fy: Double, height: Int) -> Float {
        Float(2 * atan(Double(height) / (2 * fy)))
    }
}
