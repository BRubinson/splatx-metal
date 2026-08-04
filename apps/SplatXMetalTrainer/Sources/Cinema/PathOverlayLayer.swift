// Cinema Task 4 (2026-07-26 spec UX 4) → path transform FIX 14 (2026-07-28 spec footnote 13):
// the in-scene path visualization — the spline ribbon (a polyline), the keyframe camera frusta,
// and the position diamonds.
// It is shown only in Cinema mode, and being `.allowsHitTesting(false)` it does not intercept
// camera drags.
//
// **This view is FIX 14 reinstating what FIX 6 deleted when it moved things to in-pass Metal
// (PathOverlayRenderer) (footnote 13 reverses footnote 6). Do not move it back to Metal** — what
// the history says:
// ① Even after FIX 6 (the in-pass move) the stutter the user reported remained, and the actual
//    resolution came in FIX 8 (the inspector reading the live camera in its body, so the whole
//    panel was re-evaluated on every orbit tick) and FIX 9 (the same type in the axis gizmo). The
//    A/B evidence that the full-viewport Canvas was the dominant cost is fully re-explained by the
//    FIX 8 cause, and a measurement of the Canvas cost never existed.
// ② In-pass created a new defect — when you drag a handle (SwiftUI), the ribbon and frustum pixels
//    trail behind by the whole present pipeline (real use, 2026-07-28). While a path is being
//    edited the camera is stationary, so the "consistency with the splats" that in-pass buys is
//    zero while the presentation latency is incurred in full.
//
// The core design — **the projection uses `viewer.overlayCamera` (the camera of the presented
// frame, footnote 3), not the live `viewer.camera`.** This single move satisfies both conditions at
// once:
// - During a handle drag: the camera is stationary so `overlayCamera == camera`, and the geometry
//   comes straight from the model, so the ribbon and frusta **stick to the cursor immediately**
//   (what in-pass could not do).
// - During a camera orbit: it projects against the presented frame, so it **does not drift from
//   the splats**.
// The same remedy was already verified on the handles of `PathGizmoOverlay` (FIX 13).
//
// There is no depth occlusion by the scene (it always draws on top) — an accepted v1 item from the
// preceding spec, and for an editing tool it is more practical for the path not to hide behind the
// scene (the convention of FCP-style path visualizations).
//
// The projection math (world→clip→NDC→viewport, back-plane clipping, and polyline merging) is
// factored out as the pure functions of PathProjection — PathProjectionTests is the source of
// truth. The Canvas view itself is covered by the manual checklist (this app's policy).
//
// **Path transform FIX 15 (2026-07-28 spec footnote 14): camera-independent values are not
// computed here.** The ribbon's world samples, the frustum unit geometry, the arc length, and R̄
// are authoritative in the `PathOverlayGeometry` cache
// (`CinemaViewModel.pathOverlayGeometry` — refreshed only in rebuildSampler and setOutputAspect),
// and this view **only consumes** them. Reverting that makes 101 `sampler.pose(at:)` calls
// (~18,000 transcendental calls per second) plus a 101-element array allocation recreate the same
// values every frame, even while only the camera is orbiting.
// What remains camera-dependent is only the projection (the vp matrix, polylineRuns, and
// clippedSegment) and the frustum depth scalar.
import SwiftUI
import simd

struct PathOverlayLayer: View {
    /// `overlayCamera` is the authoritative camera (the presented frame — see the comment at the
    /// top). **Do not switch it to the live `camera`**: during an orbit the overlay would move
    /// ahead on its own and appear detached from the splats (the original symptom of footnote 3).
    var viewer: ViewerModel
    var cinema: CinemaViewModel

    var body: some View {
        // Observed reads happen **inside the body** (the path transform FIX 9 discipline) — the
        // Canvas render closure runs at draw time, so there is no guarantee that an observation
        // dependency attaches to this view.
        // Everything read here determines this view's drawing, so the invalidation scope is
        // exactly this layer.
        let cam = viewer.overlayCamera
        let geometry = cinema.pathOverlayGeometry   // the camera-independent cache (FIX 15)
        let selectedIndex = cinema.selectedKeyframeIndex

        Canvas { g, size in
            guard let geometry, size.width > 1, size.height > 1 else { return }
            let vp = cam.projectionMatrix(aspect: Float(size.width / size.height))
                * cam.viewMatrix

            // ① The spline ribbon — it only projects the cached evenly-timed world samples (with
            // loop, a closed circle including the wrap segment). Contiguous segments are merged by
            // polylineRuns and stroked as one subpath per run (1 move + n−1 addLines) —
            // previously a move+addLine was repeated for each segment and 100 subpaths were
            // stroked in two passes (path transform FIX 4; footnote 5 judged it a "pure win" and
            // kept it. Removing the merge reverts to that regression).
            if geometry.ribbonPoints.count >= 2 {
                var ribbon = Path()
                for run in PathProjection.polylineRuns(geometry.ribbonPoints,
                                                       viewProjection: vp, viewport: size) {
                    ribbon.move(to: run[0])
                    for p in run.dropFirst() { ribbon.addLine(to: p) }
                }
                // The ribbon feel: a wide translucent underline plus a thin main line (two
                // screen-fixed-width passes instead of a world-width ribbon).
                g.stroke(ribbon, with: .color(SplatXTheme.accent.opacity(0.25)), lineWidth: 5)
                g.stroke(ribbon, with: .color(SplatXTheme.accent.opacity(0.9)), lineWidth: 1.5)
            }

            // ② The keyframe frusta plus the position diamonds — only the selected key is accent
            // (FCP's restrained use of color).
            // Only the depth is camera-dependent, so it is computed at draw time (the formula is
            // authoritative in PathOverlayGeometry.frustumDepth — the rationale for **never using
            // the scene bounding sphere radius** is in that comment) and multiplied into the
            // cached unit-depth offsets to reconstruct the corners (the FIX 7 decomposition).
            let depth = PathOverlayGeometry.frustumDepth(
                totalLength: geometry.totalLength, cameraDistance: cam.distance,
                meanRadius: geometry.meanRadius)
            for (i, frustum) in geometry.frusta.enumerated() {
                let selected = i == selectedIndex
                let color = selected ? SplatXTheme.accent : Color.white.opacity(0.55)
                let apex = frustum.apex
                let offsets = frustum.cornerOffsets
                var wire = Path()
                for j in offsets.indices {
                    let c = apex + offsets[j] * depth
                    if let seg = PathProjection.clippedSegment(apex, c, viewProjection: vp,
                                                              viewport: size) {
                        wire.move(to: seg.0); wire.addLine(to: seg.1)
                    }
                    let next = apex + offsets[(j + 1) % offsets.count] * depth
                    if let seg = PathProjection.clippedSegment(c, next, viewProjection: vp,
                                                              viewport: size) {
                        wire.move(to: seg.0); wire.addLine(to: seg.1)
                    }
                }
                g.stroke(wire, with: .color(color), lineWidth: selected ? 1.5 : 1)

                // The diamond = the keyframe position = the frustum apex (the cache is the single
                // source of truth).
                if let p = PathProjection.viewportPoint(apex, viewProjection: vp,
                                                        viewport: size) {
                    let d = diamond(at: p, r: selected ? 5 : 4)
                    g.fill(d, with: .color(selected ? SplatXTheme.accent : .white))
                    g.stroke(d, with: .color(SplatXTheme.ground.opacity(0.8)), lineWidth: 1)
                }
            }
        }
        // **Do not attach `.drawingGroup()` (path transform FIX 5 — measured by the user
        // 2026-07-28).** It was slower: the real cost is not the weight of the shapes but the area
        // of rebuilding a full-viewport layer every frame, so offscreen texture compositing simply
        // added one more layer. Do not retry it.
    }

    private func diamond(at c: CGPoint, r: CGFloat) -> Path {
        var p = Path()
        p.move(to: CGPoint(x: c.x, y: c.y - r))
        p.addLine(to: CGPoint(x: c.x + r, y: c.y))
        p.addLine(to: CGPoint(x: c.x, y: c.y + r))
        p.addLine(to: CGPoint(x: c.x - r, y: c.y))
        p.closeSubpath()
        return p
    }
}
