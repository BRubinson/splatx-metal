import Foundation
import simd

/// Bakes a gravity-level world orientation into the COLMAP bundle at align write time
/// (2026-08-04 spec §1).
///
/// `SceneUp.estimate` already computes the same up vector, but it only feeds the viewer's orbit
/// `upAxis` — the coordinates written into `sparse/0` stay tilted, and training, the viewer,
/// Cinema and every export inherit that tilt, so the exported PLY/SPZ leans in third-party
/// viewers. This type rotates the world once, immediately before `ColmapTextWriter.write`, so
/// everything downstream inherits level coordinates instead (spec §1.2 — nothing is rewritten
/// after the fact).
///
/// The target up is **−Y, not +Y** (spec §1.3). COLMAP has +Y pointing down, so `(0,−1,0)` is the
/// gravity up of the file convention the 3DGS ecosystem assumes. Sending the scene to +Y would
/// flip it for every external viewer and would double-correct against `SceneUp`.
///
/// The correction is a pure world rotation with a compensating pose change, so **no rendered
/// image changes**: with `X' = G·X`, `R' = R·Gᵀ` and `t' = t`, the camera-space point
/// `X_cam = R·X + t` is invariant, and the camera centers ride along with the world
/// (`C' = −R'ᵀ·t' = G·C`). GroundLevelTests pins exactly that invariant.
///
/// Original work (spec §5): the up estimate is this repo's own `SceneUp` /
/// `CameraConversion.upWorld` formula evaluated in Double over `ColmapPose`, and the shortest-arc
/// rotation is the standard simd quaternion construction. The approach was suggested by the
/// auto-level of splatx-studio (same copyright holder, private) — no code was copied, and
/// three.js's `setFromUnitVectors` was not transcribed.
enum GroundLevel {
    /// Below this the correction is skipped entirely: a fraction of a degree is inside the
    /// estimator's own noise, and rotating anyway would perturb every pose and point for no
    /// visible gain (spec §1.4).
    static let minTiltDegrees: Double = 0.5

    /// The COLMAP-convention gravity up — the direction `levelingRotation` sends the scene up to.
    static let gravityUp = SIMD3<Double>(0, -1, 0)

    /// The mean world up over the poses — the same formula as `SceneUp.estimate`
    /// (`up_i = Rᵀ·(0,−1,0) = −(row 1 of R)`, summed then normalized), evaluated in Double over
    /// `ColmapPose` instead of in Float over `ColmapView` (alignment is precision-sensitive, and
    /// `ColmapPose`/`AlignedPoint` are Double throughout).
    ///
    /// On a degeneracy — an empty list, or a sum that cancels to ≈0 — it falls back to
    /// `gravityUp` rather than to the +Y of `SceneUp`: that makes the leveling rotation the
    /// identity, i.e. "do nothing", which is the only safe answer when the estimate is
    /// meaningless.
    static func estimateUp(poses: [ColmapPose]) -> SIMD3<Double> {
        var sum = SIMD3<Double>()
        for p in poses { sum -= rotation(p).transpose.columns.1 }
        let n = simd_length(sum)
        guard n > 1e-9 else { return gravityUp }
        return sum / n
    }

    /// The angle between `up` and `gravityUp` in degrees — the number the UI shows. A degenerate
    /// (≈0-length) input reports 0, i.e. "nothing to correct".
    static func tiltDegrees(up: SIMD3<Double>) -> Double {
        let n = simd_length(up)
        guard n > 1e-9 else { return 0 }
        return acos(min(max(simd_dot(up / n, gravityUp), -1), 1)) * 180 / .pi
    }

    /// The shortest-arc rotation taking `up` to `gravityUp`: `axis = up × gravityUp`,
    /// `w = 1 + up·gravityUp`, normalized.
    ///
    /// When the two are antiparallel (`up ≈ +Y`) that quaternion degenerates to zero, so a 180°
    /// branch takes over with an arbitrary axis perpendicular to `up` — every such axis is an
    /// equally correct answer, and the result is still a proper rotation (det = +1, never a
    /// reflection).
    static func levelingRotation(up: SIMD3<Double>) -> simd_double3x3 {
        let n = simd_length(up)
        guard n > 1e-9 else { return matrix_identity_double3x3 }
        let a = up / n
        let w = 1 + simd_dot(a, gravityUp)
        if w < 1e-9 {
            // `cross(a, ref)` is only degenerate when ref is parallel to a, so the reference axis
            // is chosen away from a.
            let ref: SIMD3<Double> = abs(a.x) < 0.9 ? SIMD3(1, 0, 0) : SIMD3(0, 0, 1)
            let axis = simd_normalize(simd_cross(a, ref))
            return simd_double3x3(simd_quatd(ix: axis.x, iy: axis.y, iz: axis.z, r: 0))
        }
        let axis = simd_cross(a, gravityUp)
        return simd_double3x3(simd_quatd(ix: axis.x, iy: axis.y, iz: axis.z, r: w).normalized)
    }

    /// Rotates the world level: `p' = G·p` for the points, `R' = R·Gᵀ` and `t' = t` for the poses.
    ///
    /// - Returns: `tiltDegrees` is the **applied** correction. When the measured tilt is below
    ///   `minTiltDegrees` nothing is rotated and it reports 0, so no caller has to repeat the
    ///   threshold. The frame order, names and intrinsics and the point colors pass through
    ///   untouched — `ColmapTextWriter` binds CAMERA_ID to the name sort, so reordering here
    ///   would desynchronize cameras.txt from images.txt with no error message.
    static func leveled(frames: [AlignedFrame], points: [AlignedPoint])
        -> (frames: [AlignedFrame], points: [AlignedPoint], tiltDegrees: Double) {
        let up = estimateUp(poses: frames.map(\.pose))
        let tilt = tiltDegrees(up: up)
        guard tilt >= minTiltDegrees else { return (frames, points, 0) }
        let g = levelingRotation(up: up)
        let gT = g.transpose
        let outFrames = frames.map { f -> AlignedFrame in
            var out = f
            // The qw ≥ 0 normalization is the same convention PoseConversion produces and
            // ColmapTextWriter copies verbatim.
            var q = simd_quatd(rotation(f.pose) * gT)
            if q.real < 0 {
                q = simd_quatd(ix: -q.imag.x, iy: -q.imag.y, iz: -q.imag.z, r: -q.real)
            }
            out.pose = ColmapPose(qw: q.real, qx: q.imag.x, qy: q.imag.y, qz: q.imag.z,
                                  tx: f.pose.tx, ty: f.pose.ty, tz: f.pose.tz)
            return out
        }
        let outPoints = points.map { p -> AlignedPoint in
            let v = g * SIMD3<Double>(p.x, p.y, p.z)
            var out = p
            out.x = v.x; out.y = v.y; out.z = v.z
            return out
        }
        return (outFrames, outPoints, tilt)
    }

    /// The world→camera rotation of a COLMAP pose quaternion (w,x,y,z).
    private static func rotation(_ p: ColmapPose) -> simd_double3x3 {
        simd_double3x3(simd_quatd(ix: p.qx, iy: p.qy, iz: p.qz, r: p.qw).normalized)
    }
}

extension GroundLevel {
    /// The align write-time decision (2026-08-04 spec §1.2): correct the scene when the project's
    /// Auto-level toggle is on, otherwise pass the arrays through verbatim.
    ///
    /// - Returns: `tiltDegrees` is nil whenever **nothing was rotated** — the toggle was off, or
    ///   the measured tilt was under `minTiltDegrees`. That single optional drives both the
    ///   console line and `AlignSummary.levelTiltDegrees`, so the log and the persisted summary
    ///   cannot disagree, and no caller repeats the threshold.
    static func applied(enabled: Bool, frames: [AlignedFrame], points: [AlignedPoint])
        -> (frames: [AlignedFrame], points: [AlignedPoint], tiltDegrees: Double?) {
        guard enabled else { return (frames, points, nil) }
        let out = leveled(frames: frames, points: points)
        return (out.frames, out.points, out.tiltDegrees > 0 ? out.tiltDegrees : nil)
    }
}
