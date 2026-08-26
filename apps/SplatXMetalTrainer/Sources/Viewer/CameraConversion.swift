// Deleted in the first reorganization and revived for the camera fit (2026-07-20 spec §6)
import Foundation
import simd

enum CameraConversion {
    /// COLMAP quaternion (w,x,y,z) → a 3x3 rotation (world→camera).
    static func rotation(_ q: SIMD4<Double>) -> simd_float3x3 {
        let w = Float(q.x), x = Float(q.y), y = Float(q.z), z = Float(q.w)
        // Note: ColmapView.qvec is stored in (w,x,y,z) order → q.x=w, q.y=x, q.z=y, q.w=z.
        let n = simd_quatf(ix: x, iy: y, iz: z, r: w).normalized
        return simd_float3x3(n)
    }

    /// COLMAP: X_cam = R·X_world + t. The camera center C = -Rᵀ·t.
    static func cameraCenter(qvec: SIMD4<Double>, tvec: SIMD3<Double>) -> SIMD3<Float> {
        let R = rotation(qvec)
        let t = SIMD3<Float>(Float(tvec.x), Float(tvec.y), Float(tvec.z))
        return -(R.transpose * t)
    }

    /// The COLMAP camera forward (+Z) as a world direction: f = Rᵀ·(0,0,1) (for the camera fit,
    /// spec §6).
    static func forwardWorld(qvec: SIMD4<Double>) -> SIMD3<Float> {
        rotation(qvec).transpose.columns.2
    }

    /// The COLMAP camera up as a world direction: COLMAP has +Y downward, so up = −Rᵀ·(0,1,0).
    static func upWorld(qvec: SIMD4<Double>) -> SIMD3<Float> {
        -rotation(qvec).transpose.columns.1
    }

    /// World→Metal view. COLMAP camera (+X right, +Y down, +Z forward) → Metal (+X right, +Y up,
    /// -Z forward) is a sign flip on Y and Z. view = flip · [R | t].
    static func viewMatrix(qvec: SIMD4<Double>, tvec: SIMD3<Double>) -> float4x4 {
        let R = rotation(qvec)
        let t = SIMD3<Float>(Float(tvec.x), Float(tvec.y), Float(tvec.z))
        let flip = simd_float3x3(diagonal: SIMD3<Float>(1, -1, -1))
        let Rm = flip * R
        let tm = flip * t
        return float4x4(columns: (
            SIMD4(Rm.columns.0, 0),
            SIMD4(Rm.columns.1, 0),
            SIMD4(Rm.columns.2, 0),
            SIMD4(tm, 1)))
    }

    /// COLMAP (qvec, tvec) → a free camera pose in the viewport convention (2026-08-04 spec
    /// §2.2). It is the **exact inverse** of `viewMatrix` above, not the OrbitFit
    /// approximation — position, orientation (roll included) and fov all reproduce the capture
    /// camera, so the viewport picture matches the captured image.
    ///
    /// Derivation (flip = diag(1,−1,−1), R = the world→camera 3×3 of qvec, t = tvec):
    ///   CameraPose.viewMatrix       = [Rvᵀ | −Rvᵀ·p]   (Rv = rotation, camera→world)
    ///   CameraConversion.viewMatrix = flip·[R | t]
    ///   ⟹ Rvᵀ = flip·R ⟹ Rv = (flip·R)ᵀ = Rᵀ·flipᵀ = Rᵀ·flip   (flip is diagonal → symmetric)
    ///   ⟹ −Rvᵀ·p = flip·t ⟹ −flip·R·p = flip·t ⟹ p = −Rᵀ·t = C  (the camera center)
    /// det(flip) = +1, so Rᵀ·flip stays a proper rotation.
    ///
    /// The rotation is the same value as `PathPresets.metalCameraRotation(qvec:)` (the Cinema
    /// capture-trajectory preset) — both are quat(Rᵀ·flip). Delegated to that function rather
    /// than recomputed here so the two can never drift apart;
    /// `FirstCameraPoseTests.testPoseRotationMatchesPathPresets` pins the equality.
    /// `verticalFOV` is supplied by the caller from `OrbitFit.verticalFOV(fy:height:)` — the
    /// intrinsics live in ColmapModel.cameras, which this function does not take.
    static func pose(qvec: SIMD4<Double>, tvec: SIMD3<Double>,
                     verticalFOV: Float) -> CameraPose {
        CameraPose(position: cameraCenter(qvec: qvec, tvec: tvec),
                  rotation: PathPresets.metalCameraRotation(qvec: qvec),
                  fov: verticalFOV)
    }

    /// intrinsics → an asymmetric frustum projection (off-center cx,cy). Right-handed / -Z, Metal
    /// NDC (z∈[0,1]).
    static func projectionMatrix(_ k: ColmapIntrinsics, near: Float, far: Float) -> float4x4 {
        let w = Float(k.width), h = Float(k.height)
        // Guard against degenerate intrinsics (fx/fy=0): clamp to a positive floor so that dividing
        // by 0 does not produce NaN/Inf.
        // Normal values (fx>0) are above the floor, so the result is unchanged.
        let fx = max(Float(k.fx), 1e-6), fy = max(Float(k.fy), 1e-6)
        let cx = Float(k.cx), cy = Float(k.cy)
        // The NDC left/right and top/bottom bounds relative to the principal point.
        let l = -cx / fx * near, r = (w - cx) / fx * near
        let b = -(h - cy) / fy * near, t = cy / fy * near   // accounting for the Y flip
        // Guard against the near==far degeneracy (or a zero-width frustum): a zero denominator is
        // replaced with a finite one.
        // With normal values the actual denominator is used and the result is unchanged.
        let x = 2 * near / safeDenom(r - l), y = 2 * near / safeDenom(t - b)
        let a = (r + l) / safeDenom(r - l), bb = (t + b) / safeDenom(t - b)
        let nf = safeDenom(near - far)
        let c = far / nf, d = far * near / nf
        return float4x4(columns: (
            SIMD4(x, 0, 0, 0), SIMD4(0, y, 0, 0), SIMD4(a, bb, c, -1), SIMD4(0, 0, d, 0)))
    }

    /// Replaces a zero denominator with a small sign-preserving value (a normal denominator passes
    /// through unchanged).
    private static func safeDenom(_ v: Float) -> Float {
        if abs(v) >= 1e-12 { return v }
        return v < 0 ? -1e-6 : 1e-6
    }
}
