// Path transform Task 2 (2026-07-27 spec §4, §4.3, §5): the two viewport handle gizmos — center
// (a cross, translating the path) and radius (a circle, concentric scaling). They are always
// shown for the selected path (there is no mode button).
//
// - All the arithmetic lives in PathTransform (§4 — pure conversion with no unprojection), and
//   this view holds only the screen-coordinate round trip (PathProjection.viewportPoint) and the
//   drag state. Mutations go through
//   CinemaViewModel.translateSelectedPath/scaleSelectedPath — persistNow: false during the
//   drag, true in onEnded (one manifest save per drag).
// - The drag basis values (the conversion factor k, the basis vectors, and the starting screen
//   coordinates) are latched into @State on the first onChanged, and the increment is always
//   applied as the difference between the total measured from the start and what was applied
//   last (the transform version of the CinemaFramingOverlay idiom — no accumulation of
//   intermediate rounding. Pinning k for the duration of the drag serves the same purpose: it
//   prevents the feedback where the pivot moving makes k wobble).
// - The ⇧ decision uses NSEvent.modifierFlags (latched when the drag starts) — DragGesture does
//   not carry modifiers (a controller resolution item). Only a drag that starts on a handle is a
//   transform (its own DragGesture plus a CameraDragExclusion registration), and the rest of the
//   area still orbits the camera as before (the same structure as the selective hit testing
//   CinemaFramingOverlay uses for the watermark).
// - Both DragGestures MUST use the fixed named coordinate space (Self.space — attached to the
//   GeometryReader). Because the markers move with `.position`, a drag in the default (local)
//   space feeds its own movement back and oscillates: drag → the path moves → reprojecting the
//   pivot moves the marker (= the local origin) → the next tick reports a translation smaller
//   than the actual cursor movement → total − applied flips negative and it repeatedly snaps
//   back (a real-use bug on 2026-07-28 — the radius handle is the same, since handleScreen
//   moves with the radius). If you delete the coordinate space argument, the jitter comes back.
//   The `.gesture` → `.position` modifier order is kept so the hit area is confined to the
//   marker.
// - Display guard (§4.3): if the projection is nil (behind the camera) that marker is not drawn.
//   The radius handle is hidden when meanRadius == 0 (a single-key still shot) — so that it is
//   not drawn overlapping the pivot (the drag itself is already ignored by the 8pt denominator
//   guard in dragScaleFactor — brief Interfaces).
//
// **Path transform FIX 13 (2026-07-28 spec footnote 12 — reversing footnote 10): handle drawing
// is SwiftUI again. Do not move it to in-pass Metal.** FIX 10 moved the marker pixels in-pass and
// left only a transparent proxy here, but real use reversed it: **a direct-manipulation
// affordance (a drag handle) must follow the input, not the presented frame.** In-pass pixels lag
// the cursor by the whole present pipeline (2 in flight + GPU + presentation) while the hit area
// (this view) moves the instant the model changes, so the place you grabbed stuck to the cursor
// while the visible marker trailed behind (measured in real use 2026-07-28). During a drag the
// camera and the splats are stationary, so the "consistency with the splats" that in-pass buys
// gains nothing, and two handles are small `.position` layers with no area cost either (the
// bottleneck of this round was the camera dependency in the parent body, not the handles —
// FIX 8 and 9).
// Since drawing and input are back in the same view, the proxy-only wiring of FIX 10–12 (the
// overlayEncoded display gate and the shared isCenterVisible cull) is gone. Deriving the
// placement and the display guard still goes through PathGizmoPlacement (pure — unit-testable).
//
// **Path transform FIX 14 (2026-07-28 spec footnote 13)**: the principle established here was
// applied to the ribbon, frustum, and diamonds as well, and they went back to SwiftUI too
// (`PathOverlayLayer`) — since path editing is done by dragging, real use confirmed that the
// rationale "in-pass gives scene consistency" holds only during an orbit and merely incurs
// present latency while editing. Both views secure their consistency through the
// `viewer.overlayCamera` projection (reusing the device from footnote 3).
//
// **Path transform FIX 19 (2026-07-28 spec footnote 18 — affordance)**: hover highlights were
// added to both markers. The root problem was not "⇧ vertical movement does not work" but
// **failing to aim** — a ⇧+left drag that misses the 24pt hit area by a few pixels is assigned to
// a camera pan, the whole screen shifts, and the user reads it as "the path did not move". The
// cause was the complete absence of a signal telling the user whether the handle is grabbable, so
// on hover the stroke weight and brightness go up and the hit area (24pt) is revealed as a faint
// ring. The hit area, `contentShape`, `CameraDragExclusion`, and the drag arithmetic are
// unchanged.
// **Do not use NSCursor push/pop** — it collides with the local event monitor and MTKView cursor
// rects, and the accident of the cursor getting stuck is common (the highlight alone achieves the
// goal).
// (App finishing 2026-07-29 cross-reference: the inspector's numeric fields solved the same
// affordance problem with the SwiftUI `.pointerStyle` — the ScrubbableNumberField header,
// "Cursor feedback". The push/pop prohibition stands; the only difference is that a declarative
// modifier has no global stack. If you ever want a cursor in the viewport, evaluate that
// mechanism here too rather than push/pop.)
import SwiftUI
import AppKit
import simd

// MARK: - Handle markers (leaf Views — the hover state is confined here)

/// **Do not regress (FIX 19): the hover `@State` MUST live inside these leaf structs.**
/// Putting `@State private var isHovering` on `PathGizmoOverlay` re-evaluates **the entire
/// overlay body** (the GeometryReader, the projection, PathGizmoPlacement, and gesture
/// reconstruction) on every hover enter and exit — the very coupling FIX 15 had just removed by
/// stripping out the live camera dependency (the same type as footnotes 8, 9, and 14) would come
/// back along the cursor's path. Keeping the markers as **independent Views one level inside**
/// the point where `.gesture`/`.position` attach makes state invalidation stop at the 24pt leaf.
/// Do not turn the two markers back into computed properties.
private struct CenterHandleMarker: View {
    @State private var isHovering = false

    var body: some View {
        Path { p in
            p.move(to: CGPoint(x: 10, y: 0)); p.addLine(to: CGPoint(x: 10, y: 20))
            p.move(to: CGPoint(x: 0, y: 10)); p.addLine(to: CGPoint(x: 20, y: 10))
        }
        .stroke(SplatXTheme.accent, style: StrokeStyle(lineWidth: isHovering ? 3 : 2,
                                                      lineCap: .round))
        .shadow(color: SplatXTheme.ground.opacity(0.8), radius: 1)   // legibility on bright scenes (diamond convention)
        .shadow(color: SplatXTheme.accent.opacity(isHovering ? 0.9 : 0), radius: 3)
        .frame(width: 20, height: 20)
        .frame(width: 24, height: 24)
        .background(HandleHoverRing(isHovering: isHovering))
        .contentShape(Rectangle())
        .background(CameraDragExclusion())
        .onHover { isHovering = $0 }
        // ⇧ is latched **exactly once, on the first drag tick** (makeCenterDrag), and vertical
        // mode discards the horizontal component entirely (PathTransform.dragTranslation) — the
        // wording must reveal both.
        .help("Drag to move the whole path · Hold ⇧ before the drag starts to "
              + "raise/lower (vertical only — sideways is ignored)")
    }
}

/// The radius handle — the `CenterHandleMarker` comment is authoritative for the hover rule.
private struct RadiusHandleMarker: View {
    @State private var isHovering = false

    var body: some View {
        Circle()
            .fill(SplatXTheme.accent.opacity(isHovering ? 0.55 : 0.35))
            .overlay(Circle().stroke(SplatXTheme.accent, lineWidth: isHovering ? 2 : 1.5))
            .shadow(color: SplatXTheme.ground.opacity(0.8), radius: 1)
            .shadow(color: SplatXTheme.accent.opacity(isHovering ? 0.9 : 0), radius: 3)
            .frame(width: 12, height: 12)
            .frame(width: 24, height: 24)
            .background(HandleHoverRing(isHovering: isHovering))
            .contentShape(Circle())
            .background(CameraDragExclusion())
            .onHover { isHovering = $0 }
            .help("Drag toward or away from the ✛ center to scale the orbit radius — "
                  + "framing is preserved (⇧ does nothing here)")
    }
}

/// A 24pt ring visible only on hover — it draws **the hit area itself** (a decision area wider
/// than the marker pixels was the cause of the aiming failures, so the point is to show how far
/// the grab reaches).
/// The color uses only an opacity variation of the existing palette (`SplatXTheme.accent`) — no
/// new color.
private struct HandleHoverRing: View {
    var isHovering: Bool

    var body: some View {
        Circle()
            .fill(SplatXTheme.accent.opacity(0.14))
            .overlay(Circle().strokeBorder(SplatXTheme.accent.opacity(0.55), lineWidth: 1))
            .opacity(isHovering ? 1 : 0)
    }
}

// MARK: - Overlay

struct PathGizmoOverlay: View {
    /// The camera read is split in two (spec footnote 3): **marker placement (projection) uses
    /// the presented camera** (viewer.overlayCamera — consistent with the splats on screen.
    /// Since FIX 14 the ribbon and frustum Canvas (`PathOverlayLayer`) reads the same value —
    /// both surfaces reference the same frame, so they are consistent with each other too),
    /// while **the drag arithmetic (latching k and the basis, CameraPose.effective) uses the
    /// live viewer.camera**.
    /// During a handle drag the camera is stationary, so the two values are equal and the feel
    /// is identical; taking the drag basis from a presentation-delayed pose would in fact make
    /// it inconsistent.
    ///
    /// **Do not regress (path transform FIX 15 — spec footnote 14): do not read the live
    /// `viewer.camera` in the body.** The drag basis is not a drawn value but one used **once,
    /// on the first drag tick**, yet reading it in the body attaches an observation dependency
    /// to the entire @Observable camera struct and re-evaluates this body on every mouse event
    /// (and on every presentedCamera update) — per-frame re-evaluation doubles, and it is the
    /// same type as what footnote 8 (the inspector) and footnote 9 (the axis gizmo) caught. Read
    /// the live camera **inside the gesture action closure** (an action runs at event time
    /// rather than during body evaluation, so it creates no observation dependency — the same
    /// rationale as the Button action in `PathPresets` and the onChange closure of
    /// `playbackTick`).
    var viewer: ViewerModel
    var cinema: CinemaViewModel

    /// The center handle's drag basis (latched on the first onChanged). applied is the "total
    /// world delta from the start" applied so far — the reference point for difference
    /// application.
    private struct CenterDragState {
        var vertical: Bool
        var worldPerPoint: Float
        var right: SIMD3<Float>
        var forward: SIMD3<Float>
        var up: SIMD3<Float>
        var applied = SIMD3<Float>.zero
    }

    /// The radius handle's drag basis — the denominator (the pivot and the handle's starting
    /// screen coordinates) is pinned for the duration of the drag, and appliedFactor is the
    /// accumulated factor applied so far (the difference = total/applied).
    private struct RadiusDragState {
        var pivotScreen: CGPoint
        var handleStart: CGPoint
        var appliedFactor: Float = 1
    }

    @State private var centerDrag: CenterDragState?
    @State private var radiusDrag: RadiusDragState?

    /// The drag coordinate space name — the fixed space attached to the GeometryReader (see the
    /// comment at the top of this file).
    private static let space = "cinemaGizmo"

    var body: some View {
        GeometryReader { geo in
            let size = geo.size
            if let pivot = cinema.selectedPivot, size.width > 1, size.height > 1 {
                // The live camera is not read here (FIX 15) — makeCenterDrag reads its value at
                // that moment inside the gesture action for the drag basis (see the comment at
                // the top).
                let overlayCam = viewer.overlayCamera         // projection for marker placement (the presented frame)
                let vp = overlayCam.projectionMatrix(aspect: Float(size.width / size.height))
                    * overlayCam.viewMatrix
                // The placement and the display guard go through pure functions
                // (PathGizmoPlacement — introduced in FIX 10 and kept after FIX 13:
                // unit-testable derivation plus the same cameraAxes as the drag basis). The
                // handle direction is part of the placement too, so it is derived from the
                // **presented camera**'s axes (splat consistency).
                let placement = PathGizmoPlacement.make(
                    pivot: pivot, cameraRight: Self.cameraAxes(overlayCam).right,
                    upAxis: overlayCam.upAxis)
                // §4.3: if the pivot is behind the camera, neither handle exists — the radius
                // drag's 2D ratio also has the pivot's screen coordinates as its denominator, so
                // it cannot hold without the pivot.
                if let placement, let pivotScreen = PathProjection.viewportPoint(
                    placement.center, viewProjection: vp, viewport: size) {
                    centerMarker
                        .gesture(centerGesture(pivot: pivot, viewportHeight: size.height))
                        .position(pivotScreen)

                    if let handleWorld = placement.radius,
                       let handleScreen = PathProjection.viewportPoint(
                        handleWorld, viewProjection: vp, viewport: size) {
                        radiusMarker
                            .gesture(radiusGesture(pivotScreen: pivotScreen,
                                                   handleScreen: handleScreen))
                            .position(handleScreen)
                    }
                }
            }
        }
        .coordinateSpace(.named(Self.space))
    }

    // MARK: Markers (the color is SplatXTheme.accent — plan Step 4. Drawing + input in one view =
    // FIX 13)
    //
    // Pixels, the hit area, and hover are owned by the leaf structs (FIX 19 — the do-not-regress
    // comment at the top of this file). What remains here is only the orphaned-drag finalization
    // that **touches overlay state**: these closures touch centerDrag/radiusDrag (@State) and
    // cinema, so they cannot be pushed down into the leaves.

    /// The center handle: a cross marker. A 24pt square hit area (the drag start decision) plus
    /// the camera exclusion registration.
    private var centerMarker: some View {
        CenterHandleMarker()
            .onDisappear(perform: finishCenterDragIfOrphaned)
    }

    /// The radius handle: a 12pt-diameter circle plus a 24pt .contentShape(Circle()) hit area.
    private var radiusMarker: some View {
        RadiusHandleMarker()
            .onDisappear(perform: finishRadiusDragIfOrphaned)
    }

    // MARK: Center handle drag (§4.1)

    private func centerGesture(pivot: PathTransform.Pivot,
                               viewportHeight: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 1, coordinateSpace: .named(Self.space))
            .onChanged { v in
                // Basis latch (first tick): the ⇧ state, the conversion factor k, and the camera
                // basis — pinned for the duration of the drag.
                // The live camera is read **here** (FIX 15) — an action closure runs at event
                // time, so it creates no body observation dependency. The value itself is the
                // same "live camera on the first drag tick" as before, so the drag arithmetic
                // and the latch contract are unchanged.
                let drag = centerDrag
                    ?? Self.makeCenterDrag(pivot: pivot, cam: viewer.camera,
                                           viewportHeight: viewportHeight)
                let total = PathTransform.dragTranslation(
                    screenDelta: v.translation, vertical: drag.vertical,
                    worldPerPoint: drag.worldPerPoint, cameraRight: drag.right,
                    cameraForward: drag.forward, upAxis: drag.up)
                // source: .viewport — the drag arithmetic uses the camera as an input, so it is
                // excluded from the on-rail camera follow (the CinemaViewModel.EditSource
                // comment is the authoritative contract).
                cinema.translateSelectedPath(by: total - drag.applied, persistNow: false,
                                             source: .viewport)
                var next = drag
                next.applied = total
                centerDrag = next
            }
            .onEnded { v in
                guard let drag = centerDrag else { return }
                centerDrag = nil
                let total = PathTransform.dragTranslation(
                    screenDelta: v.translation, vertical: drag.vertical,
                    worldPerPoint: drag.worldPerPoint, cameraRight: drag.right,
                    cameraForward: drag.forward, upAxis: drag.up)
                cinema.translateSelectedPath(by: total - drag.applied, persistNow: true,
                                             source: .viewport)
            }
    }

    private static func makeCenterDrag(pivot: PathTransform.Pivot, cam: CameraController,
                                       viewportHeight: CGFloat) -> CenterDragState {
        // fov and eye come from the effective on-screen pose (WYSIWYG) — when an override is
        // active, that pose is authoritative for the projection (the same branch as
        // projectionMatrix/viewMatrix), so k must be computed from the same pose for "a 1pt drag
        // at the pivot depth = 1pt on screen" to hold.
        let pose = CameraPose.effective(of: cam)
        let axes = cameraAxes(cam)
        return CenterDragState(
            vertical: NSEvent.modifierFlags.contains(.shift),   // ⇧ — a controller resolution item
            worldPerPoint: PathTransform.worldPerPoint(
                verticalFOV: pose.fov,
                pivotDistance: simd_distance(pivot.point, pose.position),
                viewportHeight: viewportHeight),
            right: axes.right, forward: axes.forward, up: cam.upAxis)
    }

    // MARK: Radius handle drag (§4.2 — a pure 2D ratio)

    private func radiusGesture(pivotScreen: CGPoint, handleScreen: CGPoint) -> some Gesture {
        DragGesture(minimumDistance: 1, coordinateSpace: .named(Self.space))
            .onChanged { v in
                let drag = radiusDrag
                    ?? RadiusDragState(pivotScreen: pivotScreen, handleStart: handleScreen)
                radiusDrag = drag   // the basis is kept even when the denominator guard ignores it (a pinned denominator for the whole drag)
                guard let total = Self.totalScale(drag, translation: v.translation)
                else { return }     // denominator under 8pt (the pivot and handle overlap on screen) — the drag is ignored
                cinema.scaleSelectedPath(by: total / drag.appliedFactor, persistNow: false,
                                         source: .viewport)   // excluded from the camera follow (same as the center handle)
                var next = drag
                next.appliedFactor = total
                radiusDrag = next
            }
            .onEnded { v in
                guard let drag = radiusDrag else { return }
                radiusDrag = nil
                guard let total = Self.totalScale(drag, translation: v.translation)
                else { return }
                cinema.scaleSelectedPath(by: total / drag.appliedFactor, persistNow: true,
                                         source: .viewport)
            }
    }

    /// The accumulated factor from the start — cursor = the starting handle + translation
    /// (space-independent arithmetic).
    /// Lower bound 1e-3: if the cursor lands exactly on the pivot, total becomes 0 and the
    /// difference ratio (total/applied) diverges, so a finite floor guards it (the §3.4 clamp is
    /// applied separately inside the VM call).
    private static func totalScale(_ drag: RadiusDragState, translation: CGSize) -> Float? {
        PathTransform.dragScaleFactor(
            cursor: CGPoint(x: drag.handleStart.x + translation.width,
                            y: drag.handleStart.y + translation.height),
            pivotScreen: drag.pivotScreen, handleStart: drag.handleStart)
            .map { max($0, 1e-3) }
    }

    // MARK: Defense against a marker disappearing mid-drag

    /// If a drag pushes the pivot behind the camera, the marker drops out of the view tree and
    /// onEnded never arrives (SwiftUI cancels the gesture) — so the drag is finalized here and
    /// saved once, so that the mutations accumulated with persistNow: false are not left
    /// unsaved.
    private func finishCenterDragIfOrphaned() {
        guard centerDrag != nil else { return }
        centerDrag = nil
        cinema.persist()
    }

    private func finishRadiusDragIfOrphaned() {
        guard radiusDrag != nil else { return }
        radiusDrag = nil
        cinema.persist()
    }

    // MARK: Camera axes

    /// Extracts the camera right/forward from the view matrix rows. The arithmetic is owned by
    /// `PathGizmoPlacement.cameraAxes` (moved there in FIX 10 and kept after FIX 13) — so that
    /// marker placement and the drag basis use **the same derivation**; only the call name is
    /// kept here.
    private static func cameraAxes(_ cam: CameraController)
        -> (right: SIMD3<Float>, forward: SIMD3<Float>) {
        PathGizmoPlacement.cameraAxes(cam)
    }
}
