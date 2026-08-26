import SwiftUI
import Observation
import Metal
import MetalSplatter
import CryptoKit

/// Viewer state that loads one file source and renders it from a free viewpoint.
/// `SplatRendererHost` is created lazily on the first open.
@MainActor @Observable final class ViewerModel {
    /// The render host. **This didSet is the single point where the presentation notification is
    /// wired** (path transform FIX 17 — spec footnote 16). All three of unwiring, wiring, and
    /// invalidating the presentation state happen only here:
    ///   ① It unhooks the old host's callback (an abandoned host's in-flight completion Task
    ///      reads this property at the moment it fires, so this takes effect even for Tasks
    ///      already enqueued — the first line of defense).
    ///   ② It wires the new host. Because this happens **here rather than in `open()`**, the
    ///      wiring never leaks no matter which path creates and installs a host (open()'s lazy
    ///      creation, the live streaming creation in TrainViewModel.acquireRenderer, or an
    ///      external assignment). Up through FIX 16 the wiring existed only in open()'s
    ///      `if host == nil` branch, so on a run where acquireRenderer created the host first,
    ///      the notification never arrived at all (the overlay fell back to the live camera
    ///      permanently).
    ///   ③ It invalidates the presented camera — until the first frame from the new producer
    ///      arrives, the live fallback is correct (the old scene's presented camera MUST NOT
    ///      linger).
    /// Reassigning the same instance also walks ①→② again, with no exception. A "it is alive, so
    /// let's skip" optimization only adds rules (FIX 16 fell into that trap) and gains nothing —
    /// rewiring is a single closure assignment.
    var host: SplatRendererHost? {
        didSet {
            oldValue?.onFramePresented = nil
            // The notification's origin arrives as an argument (the callback is owned by the
            // host — capturing host here would create a cycle).
            host?.onFramePresented = { [weak self] source, cam in
                self?.notePresentedCamera(cam, from: source)
            }
            presentedCamera = nil
        }
    }
    var camera = CameraController()
    /// The camera anchor the **next** `applyCameraFit` uses (2026-08-04 spec §2.3). It is a
    /// one-shot: applyCameraFit reads it and immediately resets it to `.middleOrbit`, so any
    /// path that does not set it (opening a past run from the history, the standalone PLY
    /// viewer) keeps the previous behavior byte for byte.
    /// The two writers both live in the training flow — TrainViewModel.start (the live view's
    /// starting viewpoint) and ProjectSession.showFinishedRun (the result presentation).
    /// @ObservationIgnored: no view reads it — it is an internal one-shot ledger, the same
    /// convention as CinemaViewModel.lastAppliedPreviewPose.
    @ObservationIgnored var nextFitAnchor: CameraFitAnchor = .middleOrbit
    var errorMessage: String?
    var loading = false

    /// The camera of the frame that has finished being presented on screen (recorded by
    /// host.onFramePresented — spec footnote 3). It returns to nil on a no-scene notification
    /// (nil) and on a host assignment (the didSet above).
    private(set) var presentedCamera: CameraController?

    /// The camera selection rule for overlay projection (pure): prefer the presented camera,
    /// otherwise the live one. Splats are presented with the camera from several frames ago
    /// after passing through the in-flight pipeline, so scene-consistent SwiftUI surfaces read
    /// only this value.
    ///
    /// **Consumers since FIX 14**: **both** SwiftUI surfaces attached to the scene read this
    /// value — `PathOverlayLayer` (the path ribbon, frustum, and keyframe diamond Canvas, which
    /// FIX 14 reverted from in-pass Metal) and the handle markers of `PathGizmoOverlay` (drawing
    /// and input in one view, FIX 13).
    /// Projecting both with this value is what keeps them **on top of the splats on screen**
    /// even while the camera orbits.
    /// The gizmo drag arithmetic, however, keeps using the live camera (see the PathGizmoOverlay
    /// comment).
    ///
    /// **Warning: do not switch the overlay projection to the live `camera`.** Then, during an
    /// orbit, the overlay moves ahead of the presentation-delayed splats and appears detached
    /// from the scene (the original symptom of spec footnote 3). Conversely, with a stationary
    /// camera (while editing a path) the two values are equal and the delay is zero — that
    /// asymmetry is what let FIX 14 revert to a Canvas (footnote 13).
    var overlayCamera: CameraController { presentedCamera ?? camera }

    /// Scene up-axis change notification (path transform FIX 8) — called by applyCameraFit, the
    /// **only place that mutates** camera.upAxis (default +Y; orbit/pan/zoom and frameToFit
    /// leave it untouched).
    /// CinemaViewModel wires it to refresh its sceneUpAxis snapshot — the path that keeps view
    /// bodies from reading camera directly (the CinemaViewModel.sceneUpAxis comment is the
    /// authoritative contract).
    var onUpAxisChanged: (@MainActor () -> Void)?

    /// The entry point for presentation notifications. The only axis the model looks at is
    /// **producer identity** (path transform FIX 17 — spec footnote 16). Ordering (the sequence)
    /// is already filtered by the producer, the host — the model does not know about sequences,
    /// so the very pollution where an abandoned host's baseline swallows a new host's
    /// notification cannot arise (the problem FIX 15/16 tried to defend against with an epoch
    /// disappears structurally).
    /// ① **Origin defense** — if the producer carried in the notification is not the host
    ///    currently held, the notification is dropped. The path where an abandoned host's
    ///    in-flight completion Task drains late ends here.
    /// ② **Skipping equal writes** — @Observable fires a notification even for a same-value
    ///    write, so without the skip the overlay would be invalidated at 60fps even with a
    ///    stationary camera.
    func notePresentedCamera(_ cam: CameraController?, from source: AnyObject) {
        guard Self.acceptsPresentedFrame(source: source, currentSource: host) else { return }
        guard presentedCamera != cam else { return }
        presentedCamera = cam
    }

    /// The acceptance decision for a presentation notification (pure — FIX 17). It is
    /// **instance identity**, not a value: there is no value axis like a sequence whose
    /// cross-generation comparison becomes meaningless, so there is no baseline to reset
    /// either.
    /// If there is no host (nil) there is no producer to accept from, so it rejects
    /// unconditionally.
    static func acceptsPresentedFrame(source: AnyObject, currentSource: AnyObject?) -> Bool {
        guard let currentSource else { return false }
        return source === currentSource
    }

    /// Current source metadata for the viewport overlay and the Inspector.
    var currentSource: SplatSource?
    var currentTitle = ""
    /// The number of loaded splats (the overlay subtitle). nil before loading or on failure.
    var currentSplats: Int?

    /// The run open in the viewer (the Inspector properties and the Export target). nil when the
    /// file source does not originate from a run.
    var currentRun: RunRecord?

    /// The id of the run opened successfully last. The camera fit is applied only on a run
    /// **switch** (so that reloading the same run does not undo a camera the user moved —
    /// spec §6).
    /// (Why the read is internal: tests assert "not updated on abort" for the abort propagation
    /// (FIX 18) directly. Writes are still only the openRun success path and clear().)
    private(set) var lastRunID: String?

    /// The most recently framed scene bounding box (for the frame button's re-fit).
    private var lastBox: BoundingBox?

    /// Loads a source. The return value is **whether this call ran to completion and ended as
    /// the owner of the viewer state** (path transform FIX 18 — spec footnote 17): true =
    /// success or a real failure (errorMessage set), false = **abort** (a clear() or host
    /// replacement during the load). An abort is not an error — the user emptied it themselves,
    /// so errorMessage is not set. FIX 17 turned the crash into a silent abort, but the fact of
    /// the abort was not conveyed to the caller, so the follow-up code in openRun/openFile
    /// looked only at `errorMessage == nil` and went on to write lastRunID, currentRun, and the
    /// camera fit, resurrecting a deleted run as "the run opened last" (a ghost run). **A caller
    /// that receives false MUST NOT write a single line of viewer state** — the path that
    /// aborted it (clear() or a new load) already owns the state.
    @discardableResult
    func open(_ source: SplatSource) async -> Bool {
        errorMessage = nil; loading = true
        currentSource = source; currentTitle = source.label; currentSplats = nil
        // Hold the host that performed the load with a local strong reference (path transform
        // FIX 17). At the await resumption point `self.host` may have changed (a run deleted
        // during the load → ProjectSession → clear()), and the previous re-force-unwrap of
        // `host!` was a crash on that path. The presentation notification wiring is not here —
        // host's didSet is the only wiring point.
        var loader: SplatRendererHost?
        // The spinner is turned off only for its own load (FIX 18 ⓑ) — the discriminating signal
        // is host identity, the same as the abort decision. If host is still my loader (ran to
        // completion) or absent entirely (taken down by a failure, or no new load after
        // clear()), the spinner is mine; but if a **different** host is installed, the spinner
        // belongs to a new load that has already started, and turning it off here would remove
        // the indicator of the load in progress.
        defer { if host == nil || host === loader { loading = false } }
        do {
            if host == nil {
                guard let device = MTLCreateSystemDefaultDevice() else { errorMessage = "No Metal device"; return true }
                host = try SplatRendererHost(device: device)   // didSet wires and invalidates the presentation state
            }
            guard let live = host else { return false }   // right after the assignment above — unreachable in theory
            loader = live
            let box = try await live.load(url: source.url)
            // If host changed after resumption, abort silently — that path (clear()) already
            // emptied the state consistently, so writing more here would resurrect values from
            // the deleted scene.
            guard host === live else { return false }
            lastBox = box
            camera.frameToFit(box)
            currentSplats = live.renderer.splatCount   // the number of splats actually loaded
            return true
        } catch {
            // For the same reason, a load failure after host has changed also stays silent and
            // respects that side's state.
            if let loader, host !== loader { return false }
            let hint = "\(error)".lowercased().contains("unsupported") ? " If the PLY's f_rest is not 0 or 45, re-save it with `splatx convert`." : ""
            errorMessage = "Load failed: \(error)." + hint
            host = nil; currentSplats = nil   // didSet invalidates the presented camera (live fallback)
            return true
        }
    }

    /// Opens a run's output. If the environment cannot read SPZ through MetalSplatter, it falls
    /// back to a temporary PLY conversion.
    /// If `colmapContext` is present, the middle camera fit is applied only on a run **switch**
    /// (spec §6) — reloading the same run keeps the existing behavior (frameToFit).
    /// If open() reports an abort (false), it **returns immediately** (FIX 18) — if lastRunID
    /// were left pointing at a deleted run, the next openRun would be misjudged as a "switch"
    /// and the viewpoint the user had set up would be lost to a re-fit (violating spec §6), and
    /// the SPZ fallback's `currentRun = record` would resurrect the run clear() had emptied, so
    /// the Inspector, Export, and RenderJob would see a ghost run.
    func openRun(_ record: RunRecord, projectDirectory: URL,
                 colmapContext: ColmapContext? = nil) async {
        // The fit anchor is one-shot (2026-08-04 spec §2.3). applyCameraFit below consumes it on
        // the success path; this defer covers **every** other exit — an open() abort, a load
        // failure, and the `switched == false` branch — so an armed anchor can never leak into
        // the next, unrelated openRun. The .middleOrbit paths must stay byte-identical, and a
        // leaked anchor would silently move a history open onto the first camera.
        // (Idempotent on the success path: applyCameraFit already reset it.)
        defer { nextFitAnchor = .middleOrbit }
        let switched = record.id != lastRunID
        currentRun = record
        let url = projectDirectory.appendingPathComponent("runs/\(record.id)/\(record.outputFile)")
        guard await open(SplatSource(file: url)) else { return }   // abort — skip every follow-up state write
        if errorMessage != nil, url.pathExtension.lowercased() == "spz" {
            let tmp = FileManager.default.temporaryDirectory
                .appendingPathComponent("sx-view-\(record.id).ply")
            do {
                try ExportService.export(from: url, to: tmp)
                errorMessage = nil
                guard await open(SplatSource(file: tmp)) else { return }   // abort — currentRun untouched
                currentRun = record   // open swaps only currentSource — keep the run link
            } catch {
                errorMessage = "SPZ preview failed: \(error.localizedDescription)"
            }
        }
        guard errorMessage == nil else { return }   // a failed run is not recorded as "the run opened"
        lastRunID = record.id
        if switched { applyCameraFit(context: colmapContext) }   // on failure, open()'s frameToFit stands
    }

    /// Opens a standalone file (an independent viewer window — app round 5 item 5). It opens one
    /// URL with no run context.
    /// SPZ reuses the same temporary PLY conversion fallback as openRun
    /// (ExportService.export), but the overlay title keeps the original file name rather than
    /// the temporary one.
    /// The initial viewpoint is open()'s frameToFit as-is (a file source has no COLMAP context).
    func openFile(_ url: URL) async {
        guard await open(SplatSource(file: url)) else { return }   // abort (FIX 18) — state untouched
        if errorMessage != nil, url.pathExtension.lowercased() == "spz" {
            // The temporary file name is uniqued with a stable hash of the full path (the first
            // 8 bytes of SHA256) — this prevents collisions between same-named SPZ files in
            // different folders, while reopening the same file reuses the same name
            // (overwriting).
            let digest = SHA256.hash(data: Data(url.path.utf8))
                .prefix(8).map { String(format: "%02x", $0) }.joined()
            let tmp = FileManager.default.temporaryDirectory
                .appendingPathComponent("sx-view-file-\(digest).ply")
            do {
                try ExportService.export(from: url, to: tmp)
                errorMessage = nil
                guard await open(SplatSource(file: tmp)) else { return }   // abort — title untouched
            } catch {
                errorMessage = "SPZ preview failed: \(error.localizedDescription)"
            }
        }
        if errorMessage == nil { currentTitle = url.lastPathComponent }
    }

    /// Initializes the orbit and FOV from the pose and intrinsics of the middle sorted camera
    /// (middleView) (spec §6).
    /// If the context is nil or pointsCentroid is absent, it leaves the camera untouched and
    /// returns false — the caller's existing frameToFit result stands as the fallback (upAxis is
    /// untouched too — still +Y).
    /// Reused by Task 5 (live start).
    /// (It takes a ColmapContext rather than the session — avoiding a ViewerModel↔ProjectSession
    /// coupling.)
    @discardableResult
    func applyCameraFit(context: ColmapContext?) -> Bool {
        // The anchor is consumed **before** the guard (2026-08-04 spec §2.3, one-shot): if a
        // degenerate context let the flag survive, the next unrelated fit — opening a past run
        // from the history — would silently inherit the first-camera anchor.
        let anchor = nextFitAnchor
        nextFitAnchor = .middleOrbit
        guard let context, let centroid = context.pointsCentroid else { return false }
        let view = context.middleView
        // The scene up axis (the mean up over all registered views — the 2026-07-26 viewer
        // up-axis spec): it aligns the orbit and viewMatrix up with the COLMAP world, preventing
        // the vertically flipped initial view on up≠+Y scenes (garden, bonsai).
        let upAxis = SceneUp.estimate(views: context.model.views)
        camera.upAxis = upAxis
        onUpAxisChanged?()   // FIX 8: refresh the Cinema up-axis snapshot (including async loads while the tab is active)
        let fit = OrbitFit.fit(
            cameraPosition: CameraConversion.cameraCenter(qvec: view.qvec, tvec: view.tvec),
            forward: CameraConversion.forwardWorld(qvec: view.qvec),
            up: CameraConversion.upWorld(qvec: view.qvec),
            pointsCentroid: centroid,
            upAxis: upAxis)
        camera.setOrbit(target: fit.target, distance: fit.distance, yaw: fit.yaw, pitch: fit.pitch)
        if let k = context.model.cameras[view.cameraId] {
            camera.verticalFOV = OrbitFit.verticalFOV(fy: k.fy, height: k.height)
        }
        // .firstCameraExact (2026-08-04 spec §2.3): the middle-camera orbit fit above stays as
        // the **underlying orbit state** and the first camera's exact pose is layered on top as
        // an override. Keeping the orbit underneath is what makes the first drag sane —
        // adoptOverridePose inverts the override using `target` as the centroid argument, so
        // without a fitted target the distance would be derived from a stale orbit.
        // The assignment MUST come **after** setOrbit: setOrbit clears overridePose on its first
        // line (the system-setting branch of the CameraController contract), so the reverse
        // order silently loses the pose (pinned by CameraFitAnchorTests).
        // The fov comes from the **first** view's intrinsics; if that camera id is missing from
        // the model the value just set from the middle view stands (the same lenient shape as
        // the fov fallback in PathPresets.captureTrajectory).
        if anchor == .firstCameraExact {
            let first = context.firstView
            let fov = context.model.cameras[first.cameraId]
                .map { OrbitFit.verticalFOV(fy: $0.fy, height: $0.height) } ?? camera.verticalFOV
            camera.overridePose = CameraConversion.pose(qvec: first.qvec, tvec: first.tvec,
                                                        verticalFOV: fov)
        }
        return true
    }

    /// Re-fits the current scene to the frame (the overlay's frame button).
    func frameToFit() {
        if let box = lastBox { camera.frameToFit(box) }
    }

    /// Returns the viewer to an empty state (run deletion — app round 2 Task 7). It takes the
    /// host down as well — ViewerView gates rendering on `host != nil`, so keeping the host
    /// would keep drawing the deleted run's splats in what should be an empty state. The next
    /// open() lazily recreates the host (the same convention as open()'s error path). It also
    /// clears lastRunID so the next openRun is treated as a "switch" (re-applying the camera
    /// fit).
    func clear() {
        host = nil
        currentRun = nil
        currentSource = nil
        currentTitle = ""
        currentSplats = nil
        errorMessage = nil
        lastRunID = nil
        lastBox = nil
        // Invalidating the presented camera is owned by the didSet of `host = nil` above
        // (FIX 17 — a single point).
    }
}

struct ViewerView: View {
    @Bindable var vm: ViewerModel
    /// Cinema (2026-07-26 spec footnote 2 ①) — only the **Cinema tab** (RootDetail) injects it.
    /// The View tab and the independent viewer window (StandaloneViewerWindow) pass nil → no
    /// timeline and no path overlay.
    /// The active toggle was removed — cinemaVM.isActive is driven by the tab switch
    /// (RootDetail.onChange).
    var cinema: CinemaViewModel? = nil

    var body: some View {
        VStack(spacing: 0) {
            ZStack {
                if let host = vm.host {
                    SplatMetalView(host: host, camera: Binding(get: { vm.camera }, set: { vm.camera = $0 }))
                    // Non-invasive overlays (the top-left label, the top-right axis gizmo, and
                    // the frame button).
                    // The label and gizmo have hit testing off and only the frame button is
                    // interactive → no interference with drag or zoom events.
                    //
                    // **Do not read `vm.camera` in this body (path transform FIX 9).** Reading
                    // the @Observable model's camera in a parent body (it is a struct, so reading
                    // a single property depends on the whole camera) re-evaluates **this entire
                    // body** on every orbit tick (measured 2026-07-28 — the previous
                    // `camera: vm.camera` pass-down was a permanent cost in both the View and
                    // Cinema tabs). Camera consumption MUST happen inside the leaf view that
                    // actually draws it → only the model reference is passed here (passing a
                    // reference is not an observed read).
                    // The remaining reads
                    // (currentTitle/currentSplats/host/loading/errorMessage) change only at
                    // scene load time, so there is no per-frame invalidation source.
                    ViewportOverlay(
                        title: vm.currentTitle,
                        subtitle: vm.currentSplats.map { "\($0.formatted()) splats" } ?? "—",
                        viewer: vm,
                        onFrame: { vm.frameToFit() })
                    // Cinema visualization (Task 4): the path ribbon, frustum, and keyframe
                    // diamonds are layered first, then the aspect mask and the overlay WYSIWYG.
                    if let cinema, cinema.isActive {
                        // Path visualization (reverted to a SwiftUI Canvas in path transform
                        // FIX 14 — it was in-pass Metal during FIX 6–13. Footnote 13 is the
                        // authoritative record).
                        // It projects with `viewer.overlayCamera` (the presented frame), so it
                        // stays consistent with the splats during an orbit and sticks to the
                        // cursor immediately while editing a path.
                        // Hit testing off — no interference with camera orbit or handle drags.
                        PathOverlayLayer(viewer: vm, cinema: cinema)
                            .allowsHitTesting(false)
                        // The path transform handles (path transform Task 2; drawing returned in
                        // FIX 13): only the markers themselves are hit — a drag outside a handle
                        // still orbits the camera as before (allowsHitTesting(false) is
                        // forbidden here).
                        //
                        // The order places it **above** ViewportOverlay (back to its original
                        // position before FIX 10). FIX 11's rationale for moving it below was
                        // "the marker pixels are at the very bottom (in-pass) so they blur
                        // behind the frame button, while a transparent proxy on top eats the
                        // click", but that premise disappeared when FIX 13 made the markers a
                        // visible SwiftUI view again (drawing and input in one) — now "what you
                        // see is what you grab" holds in either order, and in the rare case
                        // where a handle overlaps the frame button, having the visible handle on
                        // top take the click is the WYSIWYG choice.
                        // `CinemaFramingOverlay` (the watermark drag) stays above the handles as
                        // before, and the framing mask is `allowsHitTesting(false)` so its order
                        // does not matter.
                        PathGizmoOverlay(viewer: vm, cinema: cinema)
                        CinemaFramingOverlay(cinema: cinema)
                    }
                }
                else {
                    ContentUnavailableView("Select a file", systemImage: "cube.transparent")
                }
                if vm.loading { ProgressView().controlSize(.large) }
                if let e = vm.errorMessage {
                    VStack {
                        Spacer()
                        Label(e, systemImage: "xmark.octagon")
                            .padding()
                            .background(.regularMaterial)
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            // The bottom timeline strip (the bottom third of the FCP three-way split) — only in
            // the Cinema tab (footnote 2 ① — the previous toggle at the top of the View tab was
            // removed; isActive is driven by the tab).
            if let cinema, cinema.isActive {
                Divider().overlay(SplatXTheme.hairline)
                TimelineStrip(vm: cinema)
            }
        }
    }
}
