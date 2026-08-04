import Foundation
import simd

/// The world→cam pose of the COLMAP images.txt convention (qvec w-first + tvec).
struct ColmapPose: Equatable {
    var qw: Double, qx: Double, qy: Double, qz: Double
    var tx: Double, ty: Double, tz: Double
}

enum PoseConversion {
    /// RealityKit camera c2w (a rotation quat + a position — camera-local +X right, +Y up, −Z
    /// viewing direction) → COLMAP w2c (+X right, +Y down, +Z viewing direction).
    ///   F = diag(1,-1,-1);  R_w2c = F · R_c2wᵀ;  t = -R_w2c · C
    /// ⚠ The optical axis convention of RealityKit poses is undocumented by Apple — this
    /// conversion is the form verified and settled in Task 9 (the garden measurements). If you
    /// modify it, you MUST re-verify it against measurements (spec §6).
    static func colmapFromRealityKit(rotation: simd_quatf, translation: SIMD3<Float>) -> ColmapPose {
        let F = simd_float3x3(diagonal: SIMD3<Float>(1, -1, -1))
        let rW2C = F * simd_float3x3(rotation).transpose
        let t = -(rW2C * translation)
        var q = simd_quatf(rW2C)
        if q.real < 0 { q = simd_quatf(ix: -q.imag.x, iy: -q.imag.y, iz: -q.imag.z, r: -q.real) }
        return ColmapPose(qw: Double(q.real), qx: Double(q.imag.x),
                          qy: Double(q.imag.y), qz: Double(q.imag.z),
                          tx: Double(t.x), ty: Double(t.y), tz: Double(t.z))
    }
}
