// Path transform FIX 15 (2026-07-28 spec footnote 14): the **camera-independent** world geometry
// cache of the Cinema path visualization — a pure struct plus a factory.
//
// History: this cache was introduced by FIX 6 (the move to in-pass Metal) and deleted wholesale by
// FIX 14 (the return to Canvas), along with the in-pass wiring. But **the cache is not an in-pass-
// only device** — the ribbon samples, the frustum unit geometry, the arc length, and R̄ all depend
// only on the path, so the values are bit-identical while the camera orbits, and whoever draws
// them — the GPU or a SwiftUI Canvas — merely consumes them. The Canvas right after FIX 14 called
// `sampler.pose(at:)` 101 times per draw (2 binary searches + 6 Barry–Goldman lerps +
// slerpShortest per call), repeating **~300 transcendental calls per frame, ~18,000 per second at
// 60fps** just to obtain the same result. FIX 15 removes only that waste and leaves the projection
// in the Canvas (the hybrid noted at the end of footnote 13 as "the first candidate on
// re-measurement" is exactly this shape).
//
// There are **exactly two** refresh points: `CinemaViewModel.rebuildSampler()` (the single gate
// for path mutations — through commit) and `setOutputAspect` (the frustum aspect depends on
// outputAspect). The view never rebuilds it — if the view builds it, it turns back into a
// per-frame cost.
//
// The frustum **depth** is not cached (preserving the FIX 7 decomposition) — the depth includes a
// `cam.distance` term and is therefore camera-dependent, so only the **depth-1** corner offsets
// are baked and the depth is multiplied in at draw time. `PathProjection.frustumCorners` is linear
// in depth, so the reconstruction is exact.
import Foundation
import simd

/// The world-space geometry of the path overlay (camera-independent — rebuilt only when the path
/// or the aspect ratio changes).
struct PathOverlayGeometry: Equatable {
    /// The number of ribbon polyline sample segments (~100 evenly spaced in time) — the FIX 14
    /// Canvas value verbatim.
    static let sampleCount = 100

    /// The frustum geometry of one keyframe — the apex (= the keyframe position, which is also the
    /// diamond marker position) plus 4 corner offsets **relative to depth 1** (in tl, tr, br, bl
    /// order — the frustumCorners convention).
    /// Reconstruction at draw time: `corner_j = apex + cornerOffsets[j] × depth`.
    /// There is no separate array of keyframe positions because the apex is exactly that value
    /// (storing it twice = two sources of truth). PathOverlayGeometryTests pins the agreement
    /// itself.
    struct Frustum: Equatable {
        var apex: SIMD3<Float>
        var cornerOffsets: [SIMD3<Float>]
    }

    /// The spline ribbon's evenly-timed sample polyline (sampleCount+1 points; with loop the last
    /// point closes onto the first).
    /// An empty array when there are fewer than 2 keys or duration ≤ 0 (the FIX 14 Canvas guard
    /// verbatim).
    var ribbonPoints: [SIMD3<Float>]
    /// The per-keyframe frusta — the same order and count as `sampler.path.keyframes`.
    var frusta: [Frustum]
    /// The sampler's arc length (camera-independent) — the arc-length term of frustumDepth.
    var totalLength: Double
    /// The pivot's mean radius R̄ (camera-independent) — the upper-clamp term of frustumDepth. 0
    /// with a single key (not applied).
    var meanRadius: Float

    /// The frustum depth (the draw-time formula — the overdraw defense FIX 7 established. FIX 14
    /// inlined it into the Canvas closure and the regression test disappeared; FIX 15 revived it
    /// as a pure function).
    ///
    /// **Never use the scene bounding sphere radius.** FIX 6 replaced the `cam.distance` term with
    /// a scene bounding sphere radius term (×0.08) and FIX 7 reverted it — a min/max bounding
    /// sphere is polluted by far outlier splats (r_max 13,940 on garden 7k — measured 2026-07-28)
    /// and the frustum exploded to tens of meters, becoming a zigzag across the scene plus alpha
    /// overdraw of translucent lines covering the screen. The formula is the pre-FIX 6 original
    /// `max(arc length×0.02, camera distance×0.03, 1e-3)`, and the caller passes
    /// `overlayCamera.distance` as `cameraDistance` (relative to the presented frame —
    /// footnote 3).
    ///
    /// The upper bound `meanRadius × 0.25`: the line of defense that keeps the frustum from
    /// exceeding a quarter of the path radius no matter how the depth blows up. With meanRadius 0
    /// (a single key) the upper-bound test cannot hold at all, so it is not applied (pinning it to
    /// 0 would make the frustum vanish).
    static func frustumDepth(totalLength: Double, cameraDistance: Float,
                             meanRadius: Float) -> Float {
        let depth = max(Float(totalLength) * 0.02, cameraDistance * 0.03, 1e-3)
        return meanRadius > 0 ? min(depth, meanRadius * 0.25) : depth
    }

    /// Sampler → geometry. 0 keys (or no sampler) gives nil — no overlay.
    /// - `aspect`: the frustum aspect ratio (= the export aspect ratio, `cinema.outputAspect`).
    /// - `pivot`: the already-computed pivot snapshot (`CinemaViewModel.selectedPivot`) — only R̄
    ///   is used. It is not recomputed here so that the pivot solve (a 3×3 least squares) is not
    ///   run twice in the same rebuild.
    static func make(sampler: PathSampler?, aspect: Float,
                     pivot: PathTransform.Pivot?) -> PathOverlayGeometry? {
        guard let sampler else { return nil }
        let keys = sampler.path.keyframes
        guard !keys.isEmpty else { return nil }

        // ① The ribbon — evenly-timed samples (with loop, a closed circle including the wrap
        // segment). This loop is the entirety of the cost FIX 15 removed from the per-frame path
        // (101 pose() calls).
        var ribbon: [SIMD3<Float>] = []
        if keys.count >= 2, sampler.path.duration > 0 {
            ribbon.reserveCapacity(Self.sampleCount + 1)
            for i in 0...Self.sampleCount {
                let t = Double(i) / Double(Self.sampleCount) * sampler.path.duration
                ribbon.append(sampler.pose(at: t).pos)
            }
        }

        // ② The frusta — the vertex formula bakes PathProjection.frustumCorners (which stays
        // authoritative) at **depth 1** and subtracts the apex to turn them into offsets.
        // frustumCorners is linear in depth (both center and hh scale with depth), so
        // `apex + offset×d == frustumCorners(depth: d)` holds exactly.
        var frusta: [Frustum] = []
        frusta.reserveCapacity(keys.count)
        for key in keys {
            let (apex, corners) = PathProjection.frustumCorners(
                position: key.pos, rotation: key.quatValue.normalized,
                fov: key.fov, aspect: aspect, depth: 1)
            frusta.append(Frustum(apex: apex, cornerOffsets: corners.map { $0 - apex }))
        }

        return PathOverlayGeometry(ribbonPoints: ribbon, frusta: frusta,
                                   totalLength: sampler.totalLength,
                                   meanRadius: pivot?.meanRadius ?? 0)
    }
}
