// Cinema Task 4 (2026-07-26 spec UX 4) → path transform FIX 14 (2026-07-28 spec footnote 13):
// the pure projection math of the path visualization (world→clip→NDC→viewport, clipping behind the
// camera, and deriving the frustum wire vertices). Its consumers are the two SwiftUI views —
// `PathOverlayLayer` (the ribbon, frustum, and diamond Canvas) and `PathGizmoOverlay` (the handles'
// on-screen placement). The views consume only these functions, and PathProjectionTests is the
// authoritative contract (the Canvas view itself is covered by the manual checklist — this app's
// policy).
//
// History: when FIX 6 moved the visualization to in-pass Metal it deleted `clippedSegment`,
// `polylineRuns` (FIX 4's polyline merging), and the internal clip helper as "consumers gone", but
// FIX 14 **restored them verbatim** when it returned to a Canvas (footnote 13). In particular
// `polylineRuns` is what footnote 5 judged "a pure win, so keep it", and it must exist only as a
// pair with the Canvas — deleting it reverts the ribbon to the pre-FIX 4 two-pass stroke over 100
// subpaths.
import SwiftUI
import simd

enum PathProjection {
    /// The w floor for deciding behind/in front of the camera plane. A clip-space w ≤ this value is
    /// treated as "behind the camera".
    /// (In the perspective matrix w = −z_view — positive in front of the camera.)
    static let minW: Float = 1e-4

    /// A world point → viewport coordinates (pt, origin at top-left). Behind the camera
    /// (w ≤ minW) gives nil.
    /// Points off screen are returned as-is without clamping — the Canvas clips them naturally.
    static func viewportPoint(_ world: SIMD3<Float>, viewProjection: float4x4,
                              viewport: CGSize) -> CGPoint? {
        let c = viewProjection * SIMD4(world, 1)
        guard c.w > minW else { return nil }
        return Self.point(clip: c, viewport: viewport)
    }

    /// A world segment → a viewport segment. If one end is behind the camera it is cut at the
    /// w = minW plane in clip space and only the front portion is returned (clip coordinates are
    /// linear in the world segment, so the 4-vector interpolation is exact).
    /// nil when both ends are behind.
    static func clippedSegment(_ a: SIMD3<Float>, _ b: SIMD3<Float>,
                               viewProjection: float4x4,
                               viewport: CGSize) -> (CGPoint, CGPoint)? {
        var ca = viewProjection * SIMD4(a, 1)
        var cb = viewProjection * SIMD4(b, 1)
        if ca.w <= minW && cb.w <= minW { return nil }
        if ca.w <= minW {
            ca = Self.clip(front: cb, back: ca)
        } else if cb.w <= minW {
            cb = Self.clip(front: ca, back: cb)
        }
        return (Self.point(clip: ca, viewport: viewport),
                Self.point(clip: cb, viewport: viewport))
    }

    /// A world polyline → screen polyline "runs". A run breaks only across segments behind the
    /// camera.
    /// Contiguous segments accumulate into a single run (minimizing stroke geometry — path
    /// transform FIX 4).
    /// Continuity is decided by exact coordinate equality — adjacent segments project the shared
    /// endpoint through the same clip computation, so it is bit-identical, and when it differs (a
    /// new endpoint created by the behind-camera clip) starting a new run is correct.
    /// Runs with fewer than 2 points are discarded.
    static func polylineRuns(_ worldPoints: [SIMD3<Float>], viewProjection: float4x4,
                             viewport: CGSize) -> [[CGPoint]] {
        guard worldPoints.count >= 2 else { return [] }
        var runs: [[CGPoint]] = []
        var current: [CGPoint] = []
        func closeRun() {
            if current.count >= 2 { runs.append(current) }
            current = []
        }
        for i in 1..<worldPoints.count {
            guard let seg = Self.clippedSegment(worldPoints[i - 1], worldPoints[i],
                                                viewProjection: viewProjection,
                                                viewport: viewport) else {
                closeRun()
                continue
            }
            if current.last == seg.0 {
                current.append(seg.1)
            } else {
                closeRun()
                current = [seg.0, seg.1]
            }
        }
        closeRun()
        return runs
    }

    /// The wire vertices of a keyframe camera frustum (a small rectangular pyramid): the apex (the
    /// camera position) plus the 4 corners of the base rectangle at distance depth ahead (tl, tr,
    /// br, bl — relative to the camera). rotation is camera→world (the Metal convention: forward
    /// is −Z) — exactly the Keyframe.quat convention.
    static func frustumCorners(position: SIMD3<Float>, rotation: simd_quatf,
                               fov: Float, aspect: Float, depth: Float)
        -> (apex: SIMD3<Float>, corners: [SIMD3<Float>]) {
        let forward = rotation.act(SIMD3<Float>(0, 0, -1))
        let up = rotation.act(SIMD3<Float>(0, 1, 0))
        let right = rotation.act(SIMD3<Float>(1, 0, 0))
        let center = position + forward * depth
        let hh = depth * tan(fov * 0.5)
        let hw = hh * aspect
        return (position, [
            center - right * hw + up * hh,   // tl
            center + right * hw + up * hh,   // tr
            center + right * hw - up * hh,   // br
            center - right * hw - up * hh,   // bl
        ])
    }

    // MARK: Internal

    /// Moves back onto the w = minW plane by linear interpolation in clip space (assuming
    /// front.w > minW).
    private static func clip(front: SIMD4<Float>, back: SIMD4<Float>) -> SIMD4<Float> {
        let t = (minW - back.w) / (front.w - back.w)
        return back + (front - back) * t
    }

    /// Clip (assuming w > 0) → NDC (x,y ∈ [−1,1], +y up) → viewport (pt, origin at top-left).
    private static func point(clip c: SIMD4<Float>, viewport: CGSize) -> CGPoint {
        let nx = CGFloat(c.x / c.w), ny = CGFloat(c.y / c.w)
        return CGPoint(x: (nx + 1) * 0.5 * viewport.width,
                       y: (1 - ny) * 0.5 * viewport.height)
    }
}
