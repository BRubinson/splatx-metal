import Foundation
import simd
import AppKit
import SwiftUI

/// The scene bounding sphere (center + radius). Task 8 (SplatRendererHost) consumes it, but the
/// single definition of this type lives here in CameraController.swift.
/// (Do not redefine it when implementing Task 8.)
struct BoundingBox {
    var center: SIMD3<Float>
    var radius: Float
}

/// A free world camera pose — the Cinema skimmer and playback inject it into the viewport
/// (2026-07-26 spec UX 3). It represents poses outside the (roll-free) orbit parameterization
/// (the result of path interpolation), so it is applied directly through
/// `CameraController.overridePose` with no orbit inversion. rotation is **camera→world** (the
/// Metal convention: +X right, +Y up, −Z forward) — the same as the Keyframe.quat storage
/// convention.
struct CameraPose: Equatable {
    var position: SIMD3<Float>
    var rotation: simd_quatf
    /// Vertical FOV (radians) — the CameraController.verticalFOV convention.
    var fov: Float

    static func == (a: CameraPose, b: CameraPose) -> Bool {
        a.position == b.position && a.rotation.vector == b.rotation.vector && a.fov == b.fov
    }

    /// Camera→world pose → the view matrix (world→view): [Rᵀ | −Rᵀ·p].
    /// The round-trip golden in CinemaViewModelTests pins that the column layout matches
    /// lookAt.
    var viewMatrix: float4x4 {
        let rv = simd_float3x3(rotation).transpose
        let t = -(rv * position)
        return float4x4(columns: (
            SIMD4(rv.columns.0, 0), SIMD4(rv.columns.1, 0), SIMD4(rv.columns.2, 0),
            SIMD4(t, 1)))
    }

    /// The effective on-screen pose — the WYSIWYG source of truth for key capture. If an
    /// override is applied, that one (the camera visible on screen); otherwise it is derived
    /// from the orbit state (the view matrix rows = the camera→world columns).
    static func effective(of camera: CameraController) -> CameraPose {
        if let o = camera.overridePose { return o }
        let v = camera.viewMatrix   // override nil — orbit lookAt
        let c = simd_float3x3(columns: (
            SIMD3(v.columns.0.x, v.columns.1.x, v.columns.2.x),
            SIMD3(v.columns.0.y, v.columns.1.y, v.columns.2.y),
            SIMD3(v.columns.0.z, v.columns.1.z, v.columns.2.z)))
        return CameraPose(position: camera.eye,
                          rotation: simd_quatf(c).normalized,
                          fov: camera.verticalFOV)
    }
}

/// A pure value type that converts the orbit/pan/zoom state into view/projection matrices.
struct CameraController {
    var target = SIMD3<Float>(0, 0, 0)
    var distance: Float = 5
    var yaw: Float = 0      // radians
    var pitch: Float = 0.3
    /// Vertical FOV (radians). Default 0.9 (≈52°) — the camera fit (spec §6) overwrites it with
    /// the COLMAP intrinsics.
    var verticalFOV: Float = 0.9
    /// The orbit up axis (the scene world up — 2026-07-26 spec decision 1). Default +Y: direct
    /// capture (ARKit gravity-aligned), the standalone PLY viewer, and the frameToFit fallback
    /// keep the previous behavior, while for COLMAP scenes applyCameraFit sets it from the
    /// SceneUp estimate. orbitBasis normalizes it, so a non-unit vector is safe too, and a
    /// degenerate norm≈0 falls back to +Y.
    var upAxis = SIMD3<Float>(0, 1, 0)
    /// The pose override of the Cinema skimmer and playback (2026-07-26 spec UX 3). When
    /// non-nil, viewMatrix/projectionMatrix use this pose and **the orbit state is preserved
    /// exactly** — the implementation of the contract that the skimmer never destroys the user's
    /// orbit. Clearing it takes one of two branches (footnote 2 ②):
    /// - A user **manipulation** (orbit/pan/zoom) **adopts** the override pose into the orbit
    ///   state before clearing it — "you grab it where you see it". The previous behavior (a
    ///   plain clear) snapped back to the pre-skim orbit state and the camera teleported (the
    ///   core E2E bug).
    /// - A system **setting** (setOrbit/frameToFit) overwrites the entire orbit state, so it
    ///   only clears without adopting (unchanged from before).
    var overridePose: CameraPose?

    /// The orthonormal right-handed basis of upAxis (bx, up, bz): the axes of the spherical
    /// parameterization of eye, where
    /// dir = cosP·sinY·bx + sinP·up + cosP·cosY·bz. With upAxis=(0,1,0) it is exactly the
    /// standard basis, so it is bit-identical to the previous +Y formula
    /// dir=(cp·sy, sp, cp·cy) (proven by the existing tests passing unmodified — operations on
    /// length-1 and 0/1 components are exact in float). The reference axis is chosen as the
    /// world axis less aligned with upAxis (+Z, or +X when upAxis≈±Z) to avoid a degenerate
    /// cross product. The OrbitFit inversion shares this basis as well.
    static func orbitBasis(upAxis: SIMD3<Float>)
        -> (bx: SIMD3<Float>, up: SIMD3<Float>, bz: SIMD3<Float>) {
        let n = simd_length(upAxis)
        let up = n > 1e-6 ? upAxis / n : SIMD3(0, 1, 0)
        let ref: SIMD3<Float> = abs(up.z) < 0.99 ? SIMD3(0, 0, 1) : SIMD3(1, 0, 0)
        let c = simd_cross(up, ref)
        let bx = c / simd_length(c)
        return (bx, up, simd_cross(bx, up))
    }

    /// The camera position (world space). An orbit around target by yaw/pitch/distance — yaw
    /// rotates about upAxis, and pitch is the elevation relative to upAxis (roll-free).
    var eye: SIMD3<Float> {
        let cp = cos(pitch), sp = sin(pitch), cy = cos(yaw), sy = sin(yaw)
        let (bx, bu, bz) = Self.orbitBasis(upAxis: upAxis)
        let dir = (cp * sy) * bx + sp * bu + (cp * cy) * bz   // (cp·sy, sp, cp·cy) in the standard basis
        return target + dir * distance
    }

    /// lookAt up = upAxis. The view∥upAxis singularity is prevented by orbit()'s pitch clamp
    /// (±1.5 < π/2).
    /// If an overridePose is applied, that pose's view matrix (the Cinema skimmer and
    /// playback).
    var viewMatrix: float4x4 {
        if let pose = overridePose { return pose.viewMatrix }
        return Self.lookAt(eye: eye, center: target, up: Self.orbitBasis(upAxis: upAxis).up)
    }

    func projectionMatrix(aspect: Float) -> float4x4 {
        Self.perspective(fovY: overridePose?.fov ?? verticalFOV,
                         aspect: aspect, near: 0.01, far: 1000)
    }

    /// Applies the camera fit (OrbitFit) result in one shot (spec §6).
    /// Note: a pitch outside orbit()'s clamp range (±1.5) is applied as-is, but it snaps to ±1.5
    /// on the first drag (the narrow gap against the asin upper bound of ±π/2 — accepted, Task 3
    /// review Minor).
    mutating func setOrbit(target: SIMD3<Float>, distance: Float, yaw: Float, pitch: Float) {
        overridePose = nil   // a user or system orbit setting takes back the Cinema override
        self.target = target
        self.distance = max(0.01, distance)   // the same positive floor as zoom()
        self.yaw = yaw
        self.pitch = pitch
    }

    /// Footnote 2 ②: the override adoption at user manipulation entry points — it inverts the
    /// override pose (pos+quat+fov) into orbit parameters so the manipulation starts **exactly
    /// where it is seen on screen**.
    /// The inversion reuses OrbitFit.fit: passing the **existing target** as the centroid
    /// argument naturalizes the new distance as "the depth along the view direction of the old
    /// target" (behind-camera and too-near degeneracies are handled by fit's 0.5 floor).
    /// fov is adopted as well — if a keyframe's fov differs from the orbit verticalFOV, the
    /// projectionMatrix jumps the instant the override is cleared (overridePose?.fov ??
    /// verticalFOV), so it must be carried over too for the "no jump" contract to hold through
    /// the projection. roll is lost by the orbit model (roll-free — the same convention as
    /// spec §6). upAxis is kept.
    mutating func adoptOverridePose() {
        guard let pose = overridePose else { return }
        overridePose = nil
        let r = simd_float3x3(pose.rotation.normalized)
        let fit = OrbitFit.fit(cameraPosition: pose.position,
                               forward: -r.columns.2,   // camera→world −Z = the view direction
                               up: r.columns.1,
                               pointsCentroid: target,  // the existing target — naturalizes the distance
                               upAxis: upAxis)
        target = fit.target
        distance = max(0.01, fit.distance)
        yaw = fit.yaw
        pitch = fit.pitch
        verticalFOV = pose.fov
    }

    mutating func orbit(dx: Float, dy: Float) {
        adoptOverridePose()   // footnote 2 ② — the drag starts from the pose being seen
        yaw += dx * 0.01
        pitch = max(-1.5, min(1.5, pitch + dy * 0.01))
    }

    mutating func pan(dx: Float, dy: Float) {
        adoptOverridePose()   // adopt before reading viewMatrix — pan against the visible pose's axes
        // Extract the axes from viewMatrix — viewMatrix is generalized over upAxis, so pan
        // follows it automatically.
        let v = viewMatrix
        let right = SIMD3<Float>(v.columns.0.x, v.columns.1.x, v.columns.2.x)
        let up = SIMD3<Float>(v.columns.0.y, v.columns.1.y, v.columns.2.y)
        let s = distance * 0.002
        target += (-dx * s) * right + (dy * s) * up
    }

    mutating func zoom(_ factor: Float) {
        adoptOverridePose()   // footnote 2 ② — zoom against the pose being seen
        distance = max(0.01, distance * factor)
    }

    mutating func frameToFit(_ box: BoundingBox) {
        overridePose = nil
        target = box.center
        distance = box.radius / tan(verticalFOV * 0.5) * 1.3
    }

    static func lookAt(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> float4x4 {
        let f = normalize(center - eye)
        let s = normalize(cross(f, up))
        let u = cross(s, f)
        return float4x4(columns: (
            SIMD4(s.x, u.x, -f.x, 0),
            SIMD4(s.y, u.y, -f.y, 0),
            SIMD4(s.z, u.z, -f.z, 0),
            SIMD4(-dot(s, eye), -dot(u, eye), dot(f, eye), 1)))
    }

    static func perspective(fovY: Float, aspect: Float, near: Float, far: Float) -> float4x4 {
        let y = 1 / tan(fovY * 0.5), x = y / aspect
        let z = far / (near - far)
        return float4x4(columns: (
            SIMD4(x, 0, 0, 0), SIMD4(0, y, 0, 0), SIMD4(0, 0, z, -1), SIMD4(0, 0, z * near, 0)))
    }
}

/// Synthesized equality (every stored property is Equatable) — consumed by the equal-write skip
/// in ViewerModel.notePresentedCamera (overlay synchronization, spec footnote 3). @Observable
/// fires a notification even for a same-value write, so this is the device that keeps the 60fps
/// presentation notifications of a stationary camera from driving invalidations.
extension CameraController: Equatable {}

// MARK: - AppKit mouse input (a local event monitor)
// NSGestureRecognizer sometimes fails to receive mouse events on an MTKView hosted by SwiftUI
// (NSViewRepresentable). So, exactly like scrolling — which does work reliably — dragging is
// observed through an app-local event monitor and only what "starts in / occurs in this view's
// area" is applied to the camera (other views pass through unchanged).
//
// App round 5 Task 2: the monitor remains an app-global singleton, but registration was widened
// from a single weak reference to a weak-reference collection (ViewportRegistry) — with multiple
// viewports present at once (the main window viewer plus an independent viewer window), the last
// attach used to overwrite the previous registration and one side's orbit died. Events are routed
// to only the matching viewport by windowNumber match plus a hit inside the view bounds (the
// decision is the pure function ViewportRouting).

enum CameraGestures {
    @MainActor static func attach(to view: NSView, camera: Binding<CameraController>) {
        DragMonitor.shared.attach(to: view, camera: camera)
        ScrollMonitor.shared.attach(to: view, camera: camera)
    }

    /// Explicit deregistration at view teardown (FIX 3 item 6) — called by
    /// SplatMetalView.dismantleNSView.
    /// Cleaning dead entries out of the registry was originally deferred to the next input event,
    /// and until then a dead entry's camera Binding retained the ViewerModel (large scene
    /// memory) — this clears it at teardown so a closed window's memory is returned
    /// immediately.
    @MainActor static func detach(from view: NSView) {
        DragMonitor.shared.detach(view: view)
        ScrollMonitor.shared.detach(view: view)
    }
}

/// The pure decision for multi-viewport event routing (under test — ViewportRoutingTests).
enum ViewportRouting {
    struct Candidate {
        /// The window number the view belongs to (NSWindow.windowNumber). nil when not attached
        /// to a window → cannot match.
        var windowNumber: Int?
        /// The bounds in the view's coordinate space.
        var bounds: CGRect
        /// The event location (converted into that view's coordinate space).
        var location: CGPoint
    }

    /// The index of the candidate that is in the same window as the event
    /// (eventWindowNumber) and whose bounds contain the point. If several match, the last
    /// registration wins (the convention that a view attached later is on top — in the real app
    /// there is one viewport per window, so there are no ties). nil when there is none.
    static func targetIndex(eventWindowNumber: Int, candidates: [Candidate]) -> Int? {
        candidates.lastIndex {
            $0.windowNumber == eventWindowNumber && $0.bounds.contains($0.location)
        }
    }

    /// The exclusion-area decision (Cinema Task 4): true if the event hits one of the registered
    /// exclusion candidates — a drag starting at that point is not accepted as a camera session.
    /// The decision itself is identical to targetIndex (same window + a bounds hit).
    static func isExcluded(eventWindowNumber: Int, candidates: [Candidate]) -> Bool {
        targetIndex(eventWindowNumber: eventWindowNumber, candidates: candidates) != nil
    }
}

/// The registry of drag exclusion areas for direct-manipulation elements on the viewport (the
/// Cinema overlay image and the like).
/// The local event monitor does not know the SwiftUI z order, so a SwiftUI gesture element laid
/// over the viewport must explicitly register its own area in order not to contend with the
/// camera orbit (Cinema Task 4).
/// A registered view is a transparent marker that passes hitTest through as nil
/// (CameraDragExclusion) — only its frame is used for the decision.
@MainActor final class DragExclusionRegistry {
    private struct Entry { weak var view: NSView? }
    private var entries: [Entry] = []

    func add(_ view: NSView) {
        entries.removeAll { $0.view == nil }
        if !entries.contains(where: { $0.view === view }) {
            entries.append(Entry(view: view))
        }
    }

    func remove(_ view: NSView) {
        entries.removeAll { $0.view == nil || $0.view === view }
    }

    func contains(_ event: NSEvent) -> Bool {
        entries.removeAll { $0.view == nil }
        let candidates = entries.map { entry -> ViewportRouting.Candidate in
            guard let view = entry.view, let window = view.window else {
                return ViewportRouting.Candidate(windowNumber: nil, bounds: .zero,
                                                 location: .zero)
            }
            return ViewportRouting.Candidate(
                windowNumber: window.windowNumber,
                bounds: view.bounds,
                location: view.convert(event.locationInWindow, from: nil))
        }
        return ViewportRouting.isExcluded(eventWindowNumber: event.windowNumber,
                                          candidates: candidates)
    }
}

/// A transparent marker view that excludes left/right drags starting inside the area it covers
/// from the camera orbit/pan (the Cinema overlay drag and the like — attached with
/// `.background(CameraDragExclusion())`).
/// hitTest nil — AppKit events pass through so SwiftUI gestures are preserved, and only the frame
/// is used for the decision.
struct CameraDragExclusion: NSViewRepresentable {
    final class PassthroughView: NSView {
        override func hitTest(_ point: NSPoint) -> NSView? { nil }
    }

    func makeNSView(context: Context) -> PassthroughView {
        let view = PassthroughView()
        DragMonitor.shared.addExclusion(view: view)
        return view
    }

    func updateNSView(_ nsView: PassthroughView, context: Context) {}

    static func dismantleNSView(_ nsView: PassthroughView, coordinator: ()) {
        DragMonitor.shared.removeExclusion(view: nsView)
    }
}

/// The registry shared by the drag and scroll monitors: a collection of viewports (a weak view
/// reference plus a camera binding) and the event→viewport routing. A torn-down viewport is
/// removed immediately by the explicit detach in dismantleNSView (FIX 3 item 6), and the deferred
/// cleanup of dead entries during registration and routing is kept as a safety net.
@MainActor final class ViewportRegistry {
    struct Entry {
        weak var view: NSView?
        var camera: Binding<CameraController>
    }
    private var entries: [Entry] = []

    func attach(view: NSView, camera: Binding<CameraController>) {
        entries.removeAll { $0.view == nil }
        if let i = entries.firstIndex(where: { $0.view === view }) {
            entries[i].camera = camera      // reattaching the same view → refresh only the binding
        } else {
            entries.append(Entry(view: view, camera: camera))
        }
    }

    /// Explicit deregistration (FIX 3 item 6) — it immediately breaks the lingering retention of
    /// the camera Binding (→ the ViewerModel).
    /// The deferred cleanup (removing dead entries on input events) is kept as-is as a safety
    /// net.
    func detach(view: NSView) {
        entries.removeAll { $0.view == nil || $0.view === view }
    }

    /// The viewport the event hits (same window + inside bounds). nil when there is none.
    func hit(_ event: NSEvent) -> Entry? {
        entries.removeAll { $0.view == nil }
        let candidates = entries.map { entry -> ViewportRouting.Candidate in
            guard let view = entry.view, let window = view.window else {
                return ViewportRouting.Candidate(windowNumber: nil, bounds: .zero, location: .zero)
            }
            // locationInWindow is in the window coordinate space → converted into the view
            // coordinate space by view.convert(_, from: nil).
            return ViewportRouting.Candidate(
                windowNumber: window.windowNumber,
                bounds: view.bounds,
                location: view.convert(event.locationInWindow, from: nil))
        }
        return ViewportRouting.targetIndex(
            eventWindowNumber: event.windowNumber, candidates: candidates).map { entries[$0] }
    }

    /// The target view of the in-progress drag session → its camera binding.
    func camera(for view: NSView?) -> Binding<CameraController>? {
        guard let view else { return nil }
        return entries.first { $0.view === view }?.camera
    }
}

/// Left drag → orbit; right drag or ⇧+left drag → pan.
/// (Only drags starting inside the viewport area — pinned to the viewport that routing resolved
/// at mouseDown.)
@MainActor final class DragMonitor {
    static let shared = DragMonitor()
    private let registry = ViewportRegistry()
    /// The camera drag exclusion areas (the Cinema overlay and the like — registered by
    /// CameraDragExclusion).
    private let exclusions = DragExclusionRegistry()
    /// The target view of the in-progress drag session (resolved at mouseDown, cleared at
    /// mouseUp).
    private weak var activeView: NSView?
    private var monitor: Any?

    func addExclusion(view: NSView) { exclusions.add(view) }
    func removeExclusion(view: NSView) { exclusions.remove(view) }

    func attach(to view: NSView, camera: Binding<CameraController>) {
        registry.attach(view: view, camera: camera)
        guard monitor == nil else { return }
        monitor = NSEvent.addLocalMonitorForEvents(matching: [
            .leftMouseDown, .leftMouseUp, .leftMouseDragged,
            .rightMouseDown, .rightMouseUp, .rightMouseDragged,
        ]) { [weak self] event in
            guard let self else { return event }
            switch event.type {
            case .leftMouseDown, .rightMouseDown:
                // Only a drag starting inside the viewport area counts as a session (pinned to
                // that viewport). Starting inside an exclusion area (the Cinema overlay and the
                // like) is not a camera session — a SwiftUI gesture (the overlay placement drag)
                // consumes that event (Task 4).
                self.activeView = self.exclusions.contains(event)
                    ? nil : self.registry.hit(event)?.view
            case .leftMouseUp, .rightMouseUp:
                self.activeView = nil
            case .leftMouseDragged, .rightMouseDragged:
                guard let camera = self.registry.camera(for: self.activeView) else { return event }
                let dx = Float(event.deltaX), dy = Float(event.deltaY)
                let panning = event.type == .rightMouseDragged || event.modifierFlags.contains(.shift)
                if panning { camera.wrappedValue.pan(dx: dx, dy: dy) }
                else { camera.wrappedValue.orbit(dx: -dx, dy: dy) }   // invert horizontal only — the original vertical sign feels like direct manipulation (user-confirmed)
            default:
                break
            }
            return event
        }
    }

    func detach(view: NSView) { registry.detach(view: view) }
}

/// Scroll wheel → zoom. (Applied only to the camera of the viewport the scroll occurred in — a
/// scroll over another view, such as the sidebar, passes through without touching the camera.)
@MainActor final class ScrollMonitor {
    static let shared = ScrollMonitor()
    private let registry = ViewportRegistry()
    private var monitor: Any?

    func attach(to view: NSView, camera: Binding<CameraController>) {
        registry.attach(view: view, camera: camera)
        guard monitor == nil else { return }
        monitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { [weak self] event in
            guard let self, let entry = self.registry.hit(event) else { return event }
            entry.camera.wrappedValue.zoom(1 - Float(event.scrollingDeltaY) * 0.01)
            return event
        }
    }

    func detach(view: NSView) { registry.detach(view: view) }
}
