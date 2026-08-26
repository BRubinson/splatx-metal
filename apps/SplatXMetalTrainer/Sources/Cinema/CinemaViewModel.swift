// Cinema Task 3 (2026-07-26 spec UX section + footnote 1): Cinema mode state — path CRUD,
// keyframe editing, the hybrid pacing state machine, playback, and the skimmer. Task 4
// (visualization and presets) and Task 5 (export) consume this VM.
//
// Viewport preview state machine (spec: "scrubbing/playback never hijacks the user's orbit"):
// - The camera override (CameraController.overridePose) is applied **only on explicit
//   navigation actions** — skimmer hover, scrub, frame step, key selection, and playback
//   ticks. Any other state change (a rename, for example) leaves the camera untouched.
//   There is one exception (app finishing round 6 Task 2 — spec footnote 9):
//   **a camera already on the rail follows inspector value edits** — commit(source: .inspector)
//   calls reapplyPreviewOnRail() after rebuildSampler to re-apply the pose of the refreshed
//   geometry. "On the rail" is decided by identity comparison against the ownership ledger
//   (lastAppliedPreviewPose), and the origin is decided by EditSource (the policy lives in
//   commit alone) — the comment at each definition is the authoritative contract.
// - A user orbit/pan/zoom clears the override immediately, but only after **adopting the
//   visible pose into the orbit state** (footnote 2 ② — owned by
//   CameraController.adoptOverridePose. The previous plain clear snapped back to the
//   pre-skim orbit and the camera jumped). The VM does not re-apply until the next explicit
//   action — a drag always takes the camera back. During playback, the next tick detects the
//   clear and **stops playback** (FIX 1 I1 — the "camera manipulation during playback = stop
//   playback" contract that prevents the tick from re-applying the override and flashing
//   alternately between the orbit pose and the path pose).
// - Effective preview time = skimTime (hover, highest priority) ?? previewTime (the residue
//   of scrub/playback/selection). On hover exit skimTime=nil falls back to previewTime; if
//   that is nil too, the camera returns to the orbit (the original camera).
//
// Hybrid pacing state machine (spec footnote 1):
// - A path defaults to .even — diamonds sit at positions derived from sampler.keyframeTimes.
// - **The moment a diamond drag actually moves**, the derived time is promoted to
//   Keyframe.time and the path switches to .manual (a click-select alone does not switch it).
//   The inspector's "Even pacing" reverts it.
// - Under .manual, a duration change rescales key times proportionally (preserving the
//   pacing shape — Task 3 decision).
import Foundation
import Observation
import simd

@MainActor @Observable final class CinemaViewModel {
    /// fps — the denominator for the derived frame display and for frame stepping, and also
    /// the **export frame rate** (time is always authoritative in seconds). Task 5 widened it
    /// to a choice (validated by setFPS — `VideoExportSettings.fpsChoices` is the list's single
    /// source): the preview step grid and the export share the same value, so "the frame you
    /// checked with ←→ = the output frame" holds.
    ///
    /// **That sharing has a second edge: `minKeySpacing` below is 0.5/fps, so widening the list
    /// downward widens the editing tolerances too.** The 2026-08-05 addition of 15 (spec §3.4)
    /// makes the reachable floor 0.0333s — twice 30fps's 0.0167s, and the same magnitude the
    /// pre-existing 24fps choice already had at 0.0208s. So this is not a new *kind* of
    /// behaviour, but the lower bound of the capture snap, the drag neighbour clamp, and the
    /// no-movement threshold now all sit where 15fps puts them whenever the user selects it.
    /// Anyone adding a lower rate here is also loosening those three, not only the export.
    private(set) var fps = 30

    /// Key time threshold (seconds) = half a frame (0.5/fps) — one constant with three uses
    /// (FIX 1): the capture-update snap threshold (pre-existing), the minimum neighbor spacing
    /// of the drag clamp (M2 — prevents an instant teleport during playback caused by keys at
    /// identical times), and the no-movement threshold of a drag (I2 — a change smaller than
    /// half a frame does not count as movement).
    private var minKeySpacing: Double { 0.5 / Double(fps) }

    /// Cinema active — **driven by the Cinema tab** (footnote 2 ① — RootDetail's tab onChange
    /// wires setActive. The previous toggle at the top of the View tab was removed).
    private(set) var isActive = false
    /// Path list — injected from project.cameraPaths on session load; changes are written back
    /// through onPersist.
    private(set) var paths: [CameraPath] = []
    /// Selected path id. **Never assign from outside — changes go through `selectPath(_:)`**
    /// (path transform FIX 16). When this value changes, `selectedPath` changes, and that makes
    /// all three derived caches — `sampler`, `selectedPivot`, and `pathOverlayGeometry` — stale.
    /// A direct assignment does not pass the refresh gate (`rebuildSampler()`), so the old
    /// path's sampler, pivot, and geometry survive — `private(set)` blocks that bypass at the
    /// type level (the enforcement device for the invariant FIX 15 established: "the refresh
    /// gates are exactly rebuildSampler and setOutputAspect").
    /// The two direct assignments inside the class (`load` and `setActive`) call
    /// rebuildSampler() immediately afterwards, so they pass the gate.
    private(set) var selectedPathID: UUID?
    /// Selected keyframe (from a timeline diamond click or a capture). The target of Delete.
    private(set) var selectedKeyframeIndex: Int?
    /// Playhead (seconds, 0…duration). The reference for key capture and for the playback
    /// start point.
    private(set) var playhead: Double = 0
    /// Skimmer (timeline hover) time — when non-nil it takes top priority for the viewport
    /// preview.
    private(set) var skimTime: Double?
    /// The preview time left behind by a scrub, playback, or key selection (retained even after
    /// hover exit — see the comment at the top of this file).
    private(set) var previewTime: Double?
    private(set) var isPlaying = false
    /// Inspector text editing in progress — gates the Space/Enter/Delete/←→ commands
    /// (AppCommands).
    var isEditingText = false

    /// Sampler cache for the selected path — rebuilt on every key edit (<1ms structural
    /// contract, Task 1).
    private(set) var sampler: PathSampler?

    /// **Camera-independent** world geometry snapshot for the path overlay (path transform
    /// FIX 15 — spec footnote 14). The `PathOverlayLayer` Canvas only consumes it.
    ///
    /// **Do not regress: never build the ribbon/frustum in the view by looping over
    /// `sampler.pose(at:)`.** Those values depend only on the path and the aspect ratio, so
    /// they are bit-identical while the camera orbits; if the view builds them it repeats 101
    /// pose() calls per frame (~18,000 transcendental calls per second) plus a 101-element
    /// array allocation to obtain the same result (the state right after FIX 14 — footnote 14).
    /// There are **exactly two** refresh points: rebuildSampler() and setOutputAspect.
    private(set) var pathOverlayGeometry: PathOverlayGeometry?
    /// The target the preview pose is applied to — wired by the session lifetime
    /// (ProjectSession). Wiring also hooks the up-axis change notification (FIX 8): the
    /// snapshot refreshes the moment the scene load (applyCameraFit) decides the up axis, so
    /// an automatic run load while the Cinema tab is active (the async completion of
    /// RootDetail.openLatestIfNeeded) is not missed either (a path that a setActive-time
    /// refresh alone would miss).
    /// (The path overlay provider wiring of path transform FIX 6 was removed in FIX 14 — the
    /// ribbon, frustum, and diamonds went back to a SwiftUI Canvas, and `PathOverlayLayer`
    /// reads the sampler directly.)
    weak var viewer: ViewerModel? {
        didSet {
            viewer?.onUpAxisChanged = { [weak self] in self?.refreshSceneUpAxis() }
            refreshSceneUpAxis()
        }
    }

    /// Scene up-axis snapshot (path transform FIX 8) — consumed in the body of the inspector's
    /// PATH TRANSFORM section (Center height).
    ///
    /// **Do not regress: never read `viewer.camera` in a view body.** The camera of the
    /// @Observable ViewerModel is a struct, so reading a single property (upAxis) attaches an
    /// observation dependency to the **entire** camera, and since the camera changes wholesale
    /// on every orbit tick, that view is invalidated wholesale (measured 2026-07-28: the Cinema
    /// inspector was re-evaluated 60 times a second and occupied the main thread; the MTKView
    /// draw on the same thread was pushed back and viewport dragging stuttered. It stuttered
    /// only when a path existed because the upAxis read happens only in the PATH TRANSFORM
    /// section, which is built only when a path and a pivot exist — spec footnote 8). When a
    /// value derived from the camera is needed, use a **snapshot that refreshes only when it
    /// changes**, like this one.
    ///
    /// Refresh points: the viewer wiring (didSet) and the onUpAxisChanged notification from
    /// applyCameraFit — since applyCameraFit is the only place that mutates camera.upAxis
    /// (default +Y, untouched by orbit), those two keep it always current. The defensive
    /// refresh in setActive (entering Cinema) is a catch-up for a future path where the
    /// notification wiring is broken.
    private(set) var sceneUpAxis: SIMD3<Float> = SIMD3(0, 1, 0)

    /// Persistence callback — AppModel.openProject wires it to write back into
    /// project.cameraPaths and save the manifest.
    var onPersist: (([CameraPath]) -> Void)?

    // MARK: Overlay (Task 4 — spec data model: one project-level static overlay)

    /// Static image overlay — **one per project**, not per path (shared by every export).
    /// Every unit is a percentage of the output's short side (the OverlaySpec comment and
    /// OverlayPlacement are the single source of truth).
    private(set) var overlay: OverlaySpec?
    /// Overlay persistence callback — AppModel.openProject wires it to write back into
    /// project.overlay and save.
    var onPersistOverlay: ((OverlaySpec?) -> Void)?
    /// Project directory — used to resolve the overlay image's relative path and as the copy
    /// destination (wired by ProjectSession; applyRename refreshes it on a rename).
    var projectDirectory: URL?
    /// WYSIWYG aspect ratio (the viewport letterbox mask and frustum aspect = the export
    /// aspect — one shared state). Since Task 5 it is **persisted at project level**
    /// (project.outputAspect — Task 4 review recommendation): changes go through
    /// setOutputAspect (except the load injection), and saving is wired through
    /// onPersistAspect.
    private(set) var outputAspect: OutputAspect = .wide
    /// Aspect persistence callback — AppModel.openProject wires it to write back into
    /// project.outputAspect and save.
    var onPersistAspect: ((OutputAspect) -> Void)?

    // MARK: Export settings (Task 5 — session state. Only the aspect ratio is persisted: it is
    // shared with the overlay WYSIWYG so it belongs to the project, while codec/format/resolution
    // /fps are choices made at export time. That non-persistence is also why the GIF work of the
    // 2026-08-05 round could introduce a brand-new enum here — see CinemaExportFormat.)

    /// Output codec — HEVC by default (spec).
    // Default H.264 (+MP4): the main use is sharing demos and portfolio pieces, so universal
    // playback comes first (user decision 2026-07-27 — spec footnote 3). Choose HEVC when the
    // file needs to be half the size.
    // It has no meaning under GIF (there is no codec axis) — the EXPORT section dims that row
    // rather than hiding it, and RenderJob.consoleFormatSummary drops it from the log line.
    var exportCodec: VideoCodec = .h264
    /// Output format — the EXPORT section's "File" row (MP4 by default, MOV or GIF; 2026-08-05
    /// spec §5.1: "Format" is the codec axis, so this axis is called File). It decides the save
    /// panel's extension and content type, and — through `videoContainer == nil` — which
    /// FrameSink RenderJob builds.
    /// **private(set): every change must go through setExportFormat**, which carries the
    /// resolution clamp. A direct assignment would skip it and leave the Size control showing
    /// nothing.
    private(set) var exportFormat: CinemaExportFormat = .mp4
    /// Resolution preset (short side) — the real pixel size combines it with outputAspect
    /// (VideoExportSettings.pixelSize). Which presets are *offered* depends on exportFormat
    /// (`ResolutionPreset.choices(for:)`), and the EXPORT section's Size row shows exactly that
    /// list, so the invariant "the selected preset is one of the current format's choices" is
    /// maintained by the pair (this stored value, setExportFormat's clamp).
    var exportResolution: ResolutionPreset = .p1080

    /// Export format change (the EXPORT "File" row) — a setter gate because of the **resolution
    /// clamp** (spec §5.3), following the same shape as setOutputAspect/setFPS.
    ///
    /// The offered resolutions differ per format (video 1080p–4K, GIF 240p–720p, disjoint sets),
    /// so switching GIF+480p → MP4 would leave `exportResolution` outside the new segmented
    /// control's options. That control renders such a value as **no segment highlighted at all**:
    /// the user sees an empty row while the export quietly still runs at 480p. Snapping to the
    /// new list's first entry keeps "what the control shows" and "what renders" the same thing.
    ///
    /// The equality guard is not just an optimization — re-selecting the current format must not
    /// disturb a resolution the user already chose.
    func setExportFormat(_ format: CinemaExportFormat) {
        guard format != exportFormat else { return }
        exportFormat = format
        let choices = ResolutionPreset.choices(for: format)
        // `first` rather than a hardcoded default: the choice lists live in ResolutionPreset and
        // are ordered smallest-first, so this follows them if they are ever re-cut. An empty list
        // would be a programming error, but it leaves the selection alone instead of crashing.
        if !choices.contains(exportResolution), let fallback = choices.first {
            exportResolution = fallback
        }
    }

    private var playAnchorDate: Date?
    private var playAnchorTime: Double = 0
    private var dragIndex: Int?
    private var dragDidMove = false
    /// Whether the previous playback tick (or the start of playback) applied an override to the
    /// viewport — if overridePose == nil on the next tick, a user manipulation has taken the
    /// camera back, so playback stops (FIX 1 I1). stopPlayback resets it.
    private var playbackAppliedOverride = false

    // MARK: Derived

    var selectedPath: CameraPath? {
        guard let id = selectedPathID else { return nil }
        return paths.first { $0.id == id }
    }

    var duration: Double { selectedPath?.duration ?? 0 }

    /// The single source of truth for timeline diamond positions (seconds) — derived under
    /// even, the stored times under manual.
    var keyframeTimes: [Double] { sampler?.keyframeTimes ?? [] }

    // MARK: Lifetime

    /// Injects the persisted paths, overlay, and aspect ratio when the session opens
    /// (ProjectSession.init). Injection does not fire the save callbacks (it must not re-save
    /// the values just read).
    func load(paths: [CameraPath], overlay: OverlaySpec? = nil,
              outputAspect: OutputAspect? = nil) {
        self.paths = paths
        self.overlay = overlay
        if let outputAspect { self.outputAspect = outputAspect }
        if selectedPathID == nil || !paths.contains(where: { $0.id == selectedPathID }) {
            selectedPathID = paths.first?.id
        }
        rebuildSampler()
    }

    /// Aspect ratio change (the FRAMING picker) — shared by WYSIWYG and export, and persisted
    /// per project (Task 5). Since the frustum aspect follows this value, it **rebuilds the
    /// geometry cache** (FIX 15) — one of the two cache-refresh gates, alongside
    /// rebuildSampler.
    func setOutputAspect(_ aspect: OutputAspect) {
        guard aspect != outputAspect else { return }
        outputAspect = aspect
        rebuildOverlayGeometry()
        onPersistAspect?(aspect)
    }

    /// fps change (the export Rate picker) — values outside `VideoExportSettings.fpsChoices` are
    /// ignored (the set is named rather than repeated here so the two cannot drift). The preview
    /// frame step and the key snap threshold (minKeySpacing) follow the same value (the
    /// one-shared-state contract — see the `fps` comment for what that costs at the low end).
    func setFPS(_ value: Int) {
        guard VideoExportSettings.fpsChoices.contains(value), value != fps else { return }
        fps = value
    }

    func setActive(_ on: Bool) {
        guard isActive != on else { return }
        isActive = on
        if on {
            refreshSceneUpAxis()   // FIX 8 defensive refresh — catch up on up-axis changes missed while inactive
            if selectedPathID == nil, let first = paths.first {
                selectedPathID = first.id
                rebuildSampler()
            }
            // The entry-time rebuild FIX 6 used to do here (to pick up sceneRadius) vanished in
            // FIX 7 — the geometry no longer depends on viewer state at all, so the
            // rebuildSampler and setOutputAspect paths alone keep it current.
        } else {
            stopPlayback()
            skimTime = nil
            previewTime = nil
            selectedKeyframeIndex = nil
            applyPreview()   // clears the override — back to the orbit
        }
    }

    // MARK: Path CRUD

    /// Creates and selects a new path. The name is auto-uniqued as "Path N".
    @discardableResult
    func addPath() -> UUID {
        var n = paths.count + 1
        while paths.contains(where: { $0.name == "Path \(n)" }) { n += 1 }
        let path = CameraPath(name: "Path \(n)")
        paths.append(path)
        selectPath(path.id)
        persist()
        return path.id
    }

    func duplicatePath(id: UUID) {
        guard let i = paths.firstIndex(where: { $0.id == id }) else { return }
        var copy = paths[i]
        copy.id = UUID()
        copy.name += " copy"
        paths.insert(copy, at: i + 1)
        selectPath(copy.id)
        persist()
    }

    func deletePath(id: UUID) {
        guard let i = paths.firstIndex(where: { $0.id == id }) else { return }
        paths.remove(at: i)
        if selectedPathID == id {
            selectPath(paths.indices.contains(i) ? paths[i].id : paths.last?.id)
        }
        persist()
    }

    /// Rename. persistNow=false permits intermediate states (an empty string, for example) —
    /// finalizePathNames() commits them. **Inspector inline editing no longer takes this path**
    /// (panel redesign Task 1: the draft lives in view @State and commitPathRename is the
    /// single commit gate).
    func renamePath(id: UUID, to name: String, persistNow: Bool = true) {
        guard let i = paths.firstIndex(where: { $0.id == id }) else { return }
        if persistNow {
            let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, paths[i].name != trimmed else { return }
            paths[i].name = trimmed
            persist()
        } else {
            paths[i].name = name
        }
    }

    /// Pure path-name normalization function (panel redesign Task 1 — spec §3). Every rule is
    /// an **extraction** of an existing convention, not a new one: trimming leading and
    /// trailing whitespace (renamePath), the "Path" fallback for an empty name
    /// (finalizePathNames), and uniquing a taken name with a " 2" suffix (the "Orbit 360° 2"
    /// convention of addPresetPath — counting up from 2 until a slot is free). It is the single
    /// source of truth shared by commitPathRename, finalizePathNames, and addPresetPath.
    static func normalizedPathName(_ raw: String, existing: [String]) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        let base = trimmed.isEmpty ? "Path" : trimmed
        guard existing.contains(base) else { return base }
        var n = 2
        while existing.contains("\(base) \(n)") { n += 1 }
        return "\(base) \(n)"
    }

    /// Commits an inline name edit (panel redesign Task 1. FIX 2 ⑤: Return is the only commit —
    /// Esc and focus-out are a cancel; CinemaInspector.renameExitCommits is the authoritative
    /// policy). The draft lives only in the inspector view's @State for the whole edit, and
    /// **this gate is the only entry into the VM** — a cancel is simply not calling the commit,
    /// so the original is preserved without any active restore (replacing the previous approach
    /// of mirroring every keystroke through renamePath(persistNow:false)).
    /// It uniques only the edited row against its sibling names — it does not do a bulk
    /// normalization of every row, which would cascade renames onto untouched siblings. A
    /// no-change commit (retyping the same name, or focus passing through) does not save
    /// either.
    func commitPathRename(id: UUID, draft: String) {
        guard let i = paths.firstIndex(where: { $0.id == id }) else { return }
        var siblings = paths.map(\.name)
        siblings.remove(at: i)
        let name = Self.normalizedPathName(draft, existing: siblings)
        guard paths[i].name != name else { return }
        paths[i].name = name
        persist()
    }

    /// Bulk-commits every row's name — cleans up whitespace and empty names, then saves once
    /// (the cleanup rules are shared with normalizedPathName). It does not unique — that is the
    /// job of commitPathRename, which alone knows the edited row, and doing it here would
    /// rename untouched siblings.
    func finalizePathNames() {
        for i in paths.indices {
            paths[i].name = Self.normalizedPathName(paths[i].name, existing: [])
        }
        persist()
    }

    /// Applying a preset = **adding a new path** (existing paths are unchanged — spec UX 6),
    /// then selecting and saving. The name is uniqued in the "Orbit 360° 2" style (the origin
    /// of the normalizedPathName convention — it now consumes that function). Generating the
    /// geometry is the job of the pure functions in PathPresets.
    func addPresetPath(_ path: CameraPath) {
        var p = path
        p.name = Self.normalizedPathName(path.name, existing: paths.map(\.name))
        paths.append(p)
        selectPath(p.id)
        persist()
    }

    func selectPath(_ id: UUID?) {
        stopPlayback()
        selectedPathID = id
        selectedKeyframeIndex = nil
        playhead = 0
        skimTime = nil
        previewTime = nil
        rebuildSampler()
        applyPreview()
    }

    // MARK: Inspector settings

    /// Changes duration (lower bound 0.1s). A manual path rescales key times proportionally
    /// (preserving the pacing shape).
    ///
    /// persistNow=false is the live update during a drag scrub — the commit is the true on
    /// release, once per drag (panel redesign FIX 3 ②. It is a default argument of the same
    /// shape as setSmoothness, setOverlay, and translateSelectedPath, so every existing call
    /// site is unchanged). Without the argument, every mouse event triggers `persist()` — a
    /// synchronous manifest write to disk.
    ///
    /// Note that the manual rescale is a **relative factor** (new/old): a scrub feeds the
    /// absolute target value on every tick, so the accumulated factor collapses to T_final/D_0,
    /// the key time shape is preserved, and a back-and-forth scrub returns to the original
    /// values (this does not hold with an incremental-accumulation scheme).
    func setDuration(_ seconds: Double, persistNow: Bool = true) {
        guard var path = selectedPath else { return }
        let newDuration = max(0.1, seconds)
        guard newDuration != path.duration else {
            // FIX 4 ①: a scrub release re-enters the **same absolute target value** as the
            // last tick (ScrubbableNumberField.onEnded — a consequence of absolute-target
            // arithmetic). Even when the value is unchanged, if a persistNow:false commit from
            // during the drag is still unsaved, the save must be discharged here — otherwise
            // the new duration (and, under manual, the rescaled key times) live only in memory
            // and silently vanish on restart. The early return itself stays — a no-mutation
            // commit (typing the same value and pressing Return) still saves nothing.
            if persistNow, hasUnpersistedChanges { persist() }
            return
        }
        if path.pacing == .manual, path.duration > 0 {
            let scale = newDuration / path.duration
            for i in path.keyframes.indices { path.keyframes[i].time *= scale }
        }
        path.duration = newDuration
        playhead = min(playhead, newDuration)
        commit(path, persistNow: persistNow, source: .inspector)
    }

    func setEase(_ ease: EaseProfile) {
        guard var path = selectedPath, path.ease != ease else { return }
        path.ease = ease
        commit(path, source: .inspector)
    }

    func setLoop(_ loop: Bool) {
        guard var path = selectedPath, path.loop != loop else { return }
        path.loop = loop
        commit(path, source: .inspector)
    }

    /// smoothness is stored only (Task 1 — the v1 sampler does not consume it). While the
    /// slider is being dragged the persist is deferred (onEditingChanged), and at the end it
    /// saves once via persistNow().
    func setSmoothness(_ value: Double, persistNow: Bool = true) {
        guard var path = selectedPath else { return }
        path.smoothness = min(max(value, 0), 1)
        commit(path, persistNow: persistNow, source: .inspector)
    }

    /// Reverts manual pacing to constant speed (the inspector's "Even pacing"). The stored key
    /// times are left in place but no longer consumed (they are re-promoted from derived times
    /// on the next drag switch).
    func revertToEvenPacing() {
        guard var path = selectedPath, path.pacing == .manual else { return }
        path.pacing = .even
        commit(path, source: .inspector)
    }

    // MARK: Whole-path transform (path transform Task 2 — spec §3 and §5. The math is owned by
    // PathTransform; this file only wires the selected path and routes through commit (the
    // single path for sampler rebuild and persistence).)

    /// **Snapshot** of the selected path's pivot (§3.1–3.2 — the convergence point when the
    /// aim is fixed, otherwise the centroid). No path / zero keys → nil (the display guard for
    /// the handles and the inspector section, §4.3).
    ///
    /// **Do not regress: never turn this back into a computed property (path transform FIX 15
    /// — the same pattern as sceneUpAxis).** The value depends only on the keyframes, yet it
    /// was recomputed at every read site — two origins/dirs array allocations per call, plus a
    /// 3×3 double outer-product accumulation per key, plus a determinant, plus a matrix
    /// inverse, plus an O(N) residual loop.
    /// Frequency (corrected in FIX 16 — the previous "3 times per frame, 180 times per second"
    /// was an overstatement): since FIX 8 removed the camera read from the inspector body,
    /// **during a pure orbit only the two overlay bodies are re-evaluated, so twice per frame
    /// (120 times per second at 60fps)**, and the count of three including the inspector holds
    /// only on drag ticks that actually mutate the path.
    /// The single refresh point is rebuildSampler(), and every path that changes keyframes
    /// goes through commit() → rebuildSampler() (including the direct calls in
    /// load/selectPath/setActive).
    private(set) var selectedPivot: PathTransform.Pivot?

    /// Translates the whole path (§3.3 — pos only; quat and fov are untouched. Re-aiming is a
    /// no-op).
    /// persistNow=false is the live update during a handle drag — onEnded / numeric entry pass
    /// true, so there is one save per drag (the same save convention as key drags and
    /// sliders).
    /// source defaults to .inspector (the inspector's Height field is the main consumer) — the
    /// viewport gizmo handles pass .viewport explicitly (PathGizmoOverlay. This is the origin
    /// report for the on-rail camera follow policy — the EditSource comment is the
    /// authoritative contract).
    func translateSelectedPath(by delta: SIMD3<Float>, persistNow: Bool = true,
                               source: EditSource = .inspector) {
        guard let path = selectedPath, !path.keyframes.isEmpty else { return }
        commit(PathTransform.translated(path, by: delta), persistNow: persistNow,
               source: source)
    }

    /// Scales the path concentrically (§3.4 — the pivot is selectedPivot; quat and fov are
    /// untouched).
    /// On a guard violation (post-clamp result R̄ < 1e-3) PathTransform.scaled returns nil — no
    /// change and no save.
    /// The source convention is the same as translateSelectedPath (default .inspector = the
    /// Radius field; the gizmo radius handle passes .viewport explicitly).
    func scaleSelectedPath(by factor: Float, persistNow: Bool = true,
                           source: EditSource = .inspector) {
        guard let path = selectedPath, let pivot = selectedPivot,
              let scaled = PathTransform.scaled(path, by: factor, about: pivot.point)
        else { return }
        commit(scaled, persistNow: persistNow, source: source)
    }

    // MARK: Keyframes — capture / delete / select

    /// Captures the current viewport camera (WYSIWYG — the override pose when an override is
    /// active) at the playhead. If the playhead is within an existing key's snap threshold
    /// (half a frame), that key is **updated** (spec UX 2 — recapture = update); otherwise a
    /// key is inserted at the time-ordered position. If there is no path, the first path is
    /// created automatically.
    func captureKeyframe(from camera: CameraController) {
        if paths.isEmpty {
            addPath()
        } else if selectedPath == nil {
            selectPath(paths.first?.id)
        }
        guard var path = selectedPath else { return }
        let pose = CameraPose.effective(of: camera)
        let times = keyframeTimes
        let tolerance = minKeySpacing   // half a frame — same constant as the drag floor (FIX 1)
        if let i = Self.captureTargetIndex(playhead: playhead, times: times, tolerance: tolerance) {
            var key = path.keyframes[i]
            key.pos = pose.position
            key.quat = SIMD4(pose.rotation.real, pose.rotation.imag.x,
                             pose.rotation.imag.y, pose.rotation.imag.z)
            key.fov = pose.fov
            path.keyframes[i] = key   // time is kept (preserving the manual source of truth)
            selectedKeyframeIndex = i
        } else {
            let insertAt = times.filter { $0 < playhead }.count
            path.keyframes.insert(
                Keyframe(time: playhead, pos: pose.position, rotation: pose.rotation,
                         fov: pose.fov),
                at: insertAt)
            selectedKeyframeIndex = insertAt
        }
        // .viewport: for a capture the camera itself is an arithmetic input
        // (CameraPose.effective), so following it would be feedback, and it also has the side
        // effect of jumping the camera to a stale preview time when previewTime ≠ playhead
        // (EditSource comment ①).
        commit(path, source: .viewport)
    }

    /// Wiring convenience (the timeline capture button and the Enter command) — captures from
    /// viewer.camera.
    func captureKeyframe() {
        guard let camera = viewer?.camera else { return }
        captureKeyframe(from: camera)
    }

    func deleteSelectedKeyframe() {
        guard let i = selectedKeyframeIndex else { return }
        deleteKeyframe(at: i)
    }

    /// Deletes by index (footnote 2 ③ — the timeline diamond context menu calls it directly
    /// with the hit index). Same semantics as deleting the selected key (clearing the
    /// previewTime residue — back to the orbit), and the selection index is adjusted for the
    /// deletion position (the deleted key = deselect, a later key = shifted by −1).
    func deleteKeyframe(at index: Int) {
        guard var path = selectedPath, path.keyframes.indices.contains(index) else { return }
        path.keyframes.remove(at: index)
        if let sel = selectedKeyframeIndex {
            if sel == index { selectedKeyframeIndex = nil }
            else if sel > index { selectedKeyframeIndex = sel - 1 }
        }
        previewTime = nil   // clears the deleted key's pose residue — back to the orbit
        commit(path, source: .timeline)   // the preview is handled directly by applyPreview below
        applyPreview()
    }

    /// Key selection — moves the playhead to that key's time and jumps the viewport
    /// (spec UX 2).
    func selectKeyframe(_ index: Int?) {
        selectedKeyframeIndex = index
        if let index, keyframeTimes.indices.contains(index) {
            playhead = keyframeTimes[index]
            previewTime = playhead
            applyPreview()
        }
    }

    // MARK: Timeline manipulation — scrubbing and key drags (pacing switch)

    /// Moves the playhead. With preview=true (scrub or click) the viewport moves to that time
    /// as well.
    func setPlayhead(_ t: Double, preview: Bool = false) {
        playhead = min(max(t, 0), duration)
        if preview {
            previewTime = playhead
            applyPreview()
        }
    }

    /// Starting a diamond drag = selection (and jump) only — the pacing switch happens on
    /// actual movement (click protection).
    func beginKeyframeDrag(_ index: Int) {
        guard let path = selectedPath, path.keyframes.indices.contains(index) else { return }
        dragIndex = index
        dragDidMove = false
        selectKeyframe(index)
    }

    /// Drag movement — on the first movement where the time **actually changes**, the derived
    /// times are promoted to stored times and the path switches to .manual (spec footnote 1).
    /// The time is clamped against the neighbor keys (securing a half-frame gap) and to
    /// [0, duration]; saving happens in endKeyframeDrag.
    /// FIX 1 (I2): a no-movement tick that snap/clamp quantized back to the current position (a
    /// change smaller than half a frame) neither promotes nor moves — this is the time-axis
    /// counterpart of the click protection (dragDidMove). A 3–6px micro-drag that does not
    /// change the time MUST NOT trigger a .manual promotion.
    func updateKeyframeDrag(to t: Double) {
        guard let i = dragIndex, var path = selectedPath,
              path.keyframes.indices.contains(i) else { return }
        // Under even, the derived times are authoritative for the current positions (before
        // promotion) — promotion happens only after real movement is confirmed.
        let times = path.pacing == .even ? keyframeTimes : path.keyframes.map(\.time)
        guard times.indices.contains(i) else { return }
        let clamped = Self.clampedDragTime(t, index: i, times: times, duration: path.duration,
                                           minSpacing: minKeySpacing)
        guard abs(clamped - times[i]) >= minKeySpacing else { return }   // no movement — no-op
        if path.pacing == .even {
            for (j, tj) in times.enumerated() where path.keyframes.indices.contains(j) {
                path.keyframes[j].time = tj
            }
            path.pacing = .manual
        }
        dragDidMove = true
        path.keyframes[i].time = clamped
        // Live rebuild; the save happens once at the end of the drag. .timeline: the preview is
        // handled directly by the previewTime update plus applyPreview below (a commit-level
        // follow would apply it twice with a stale time).
        commit(path, persistNow: false, source: .timeline)
        playhead = clamped
        previewTime = clamped
        applyPreview()
    }

    func endKeyframeDrag() {
        guard dragIndex != nil else { return }
        dragIndex = nil
        if dragDidMove { persist() }
        dragDidMove = false
    }

    // MARK: Playback and frame stepping

    func togglePlayback() {
        if isPlaying { stopPlayback() } else { startPlayback() }
    }

    private func startPlayback() {
        guard let path = selectedPath, !path.keyframes.isEmpty, duration > 0 else { return }
        if !path.loop, playhead >= duration - 1e-9 { playhead = 0 }   // play at the end = from the start
        playAnchorDate = Date()
        playAnchorTime = playhead
        isPlaying = true
        previewTime = playhead
        applyPreview()
        playbackAppliedOverride = viewer?.camera.overridePose != nil
    }

    /// Stop — previewTime stays at the stopped frame (returning to the orbit is the path where
    /// a user drag clears the override — see the state machine comment at the top).
    private func stopPlayback() {
        isPlaying = false
        playAnchorDate = nil
        playbackAppliedOverride = false
    }

    /// TimelineView(.animation) tick — updates the playback position (loop wrap, stop at the
    /// end) and follows it in the viewport.
    /// FIX 1 (I1): if the override the previous tick applied is now nil, a user manipulation
    /// (orbit/pan/zoom) has taken the camera back — re-applying it would flash alternately
    /// between the orbit pose and the path pose for the rest of playback. Under the "camera
    /// manipulation during playback = stop playback" contract it stops immediately without
    /// re-applying (the previewTime residue follows the usual stop convention — re-application
    /// happens only on the next explicit action).
    func playbackTick(now: Date) {
        guard isPlaying, let anchor = playAnchorDate else { return }
        if playbackAppliedOverride, viewer?.camera.overridePose == nil {
            stopPlayback()
            return
        }
        let (t, finished) = Self.playbackPosition(
            anchorTime: playAnchorTime, elapsed: now.timeIntervalSince(anchor),
            duration: duration, loop: selectedPath?.loop ?? false)
        playhead = t
        previewTime = t
        applyPreview()
        playbackAppliedOverride = viewer?.camera.overridePose != nil
        if finished { stopPlayback() }
    }

    /// ←→ frame step — rounds to the frame grid, then ±delta, clamped to [0, duration]. The
    /// viewport follows.
    func stepFrame(_ delta: Int) {
        playhead = TimelineMath.frameStep(from: playhead, delta: delta, fps: fps,
                                          duration: duration)
        previewTime = playhead
        applyPreview()
    }

    // MARK: Skimmer

    /// Timeline hover → live viewport preview. nil (exit) → falls back to previewTime, or back
    /// to the orbit.
    func skim(to t: Double?) {
        skimTime = t.map { min(max($0, 0), duration) }
        applyPreview()
    }

    // MARK: Internal — commit and preview

    /// The origin of a path mutation (app finishing round 6 Task 2 — spec footnote 9): the
    /// input to commit()'s "on-rail camera follow" policy. **The policy lives in commit() alone,
    /// and the caller only reports the fact of where it is** — letting the caller decide
    /// whether to follow would scatter the policy.
    enum EditSource {
        /// Inspector fields (Radius, Height, Duration — both typing and drag scrubbing) and
        /// setting controls (Ease, Loop, Even pacing, and so on). If the camera is on the rail
        /// it follows the edit result — live following on every scrub tick (including
        /// persistNow: false commits) is the core of the request.
        case inspector
        /// Direct viewport manipulation (a gizmo handle drag), or **an edit where the camera
        /// itself is an arithmetic input** (key capture — CameraPose.effective is the input).
        /// It does not follow (neither during the drag nor on release). Two reasons:
        /// ① Feedback — the handle drag arithmetic uses the camera as an input
        ///    (PathTransform's world-per-point k and the overlayCamera projection). If the drag
        ///    moves the camera, that camera in turn changes the drag mapping — the same class
        ///    of closed loop as the gizmo coordinate-frame oscillation (the comment at the top
        ///    of PathGizmoOverlay).
        /// ② Frame-of-reference stability — if the viewpoint moves while you are dragging
        ///    something directly on screen, the object slips out of your hand, and when the
        ///    camera is on the rail (= inside the path), moving the path while following it
        ///    visually cancels the motion out so that nothing appears to happen at all.
        case viewport
        /// The timeline family (key drag, key delete) — these are existing paths that update
        /// previewTime themselves and call applyPreview directly, so a commit-level follow is
        /// unnecessary (with one, it would apply once more with the pre-update, stale
        /// previewTime). It does not follow.
        case timeline
    }

    /// The pose applyPreview() applied last — the ownership ledger for the "on the rail"
    /// decision (Task 2).
    /// The decision is an **identity comparison**: if the current overridePose is exactly this
    /// value, the pose we applied is still in place, and a user orbit/pan/zoom naturally breaks
    /// the match because adoptOverridePose sets it to nil (the value itself is the signal, with
    /// no notification — the same principle as the playbackAppliedOverride polling). Never use
    /// an epsilon comparison — it would misjudge a user who orbited away and back as "still
    /// attached".
    /// nil ↔ nil is **not** on the rail — if nothing was ever applied there is nothing to
    /// follow, and conjuring an override that did not exist is a regression that snatches the
    /// camera away from a user who was in a free orbit.
    /// Keep it a **separate ledger** from `playbackAppliedOverride`: that one is "did the
    /// previous playback tick apply one" (a boolean — stopPlayback resets it) while this one is
    /// "the last applied pose itself" (nil only when the pose is actually cleared), so their
    /// reset points differ, and merging them would shake the contract of the playback-stop
    /// golden (testUserCameraGrabDuringPlaybackStopsPlayback).
    /// It is an internal ledger no view reads, so it is excluded from observation tracking.
    @ObservationIgnored private var lastAppliedPreviewPose: CameraPose?

    /// The ledger of **unsaved mutations** left by persistNow:false commits (panel redesign
    /// FIX 4 ①). It is the basis on which a setter that has a no-change early return
    /// (setDuration) discharges its save obligation regardless of the value comparison, when a
    /// scrub release re-enters the same value as the last tick. persist() clears it.
    /// It is an internal flag no view reads, so it is excluded from observation tracking.
    @ObservationIgnored private var hasUnpersistedChanges = false

    /// Applies the selected-path replacement and rebuilds the sampler (and persists). The
    /// single path for every path mutation — the on-rail camera follow policy (Task 2) also
    /// lives in this one place (the EditSource comment is the authoritative contract).
    private func commit(_ path: CameraPath, persistNow: Bool = true, source: EditSource) {
        guard let i = paths.firstIndex(where: { $0.id == path.id }) else { return }
        paths[i] = path
        rebuildSampler()
        if let sel = selectedKeyframeIndex, !path.keyframes.indices.contains(sel) {
            selectedKeyframeIndex = nil
        }
        playhead = min(playhead, path.duration)
        // The follow MUST come **after** rebuildSampler and the playhead clamp — it has to
        // apply the pose of the new geometry. Even if previewTime is left out of range by a
        // duration reduction, the [0, duration] clamp in sampler.pose(at:) (the
        // CameraPathSampler header contract) guarantees the correct pose.
        if source == .inspector { reapplyPreviewOnRail() }
        if persistNow { persist() } else { hasUnpersistedChanges = true }
    }

    /// The on-rail camera's edit follow (Task 2) — a hook used only by
    /// commit(source: .inspector).
    /// Only when the camera is "still on the rail" (overridePose == the ledger pose exactly)
    /// does it re-apply the pose of the refreshed geometry through applyPreview. Off the rail
    /// (a user manipulation set it to nil, or nothing was ever applied) it is untouched — it
    /// does not conjure an override that did not exist, and it does not touch the orbit
    /// parameters of viewer.camera either (rail attachment is an overridePose assignment and
    /// nothing else — the CameraController orbit-state preservation contract).
    private func reapplyPreviewOnRail() {
        guard let viewer, let applied = lastAppliedPreviewPose,
              viewer.camera.overridePose == applied else { return }
        applyPreview()
    }

    /// The only derived-value refresh gate for path mutations — it refreshes the sampler, the
    /// pivot snapshot, and the overlay geometry together (FIX 15). All three depend **only on
    /// the keyframes**, so this one place is enough, and if a derived value is omitted here and
    /// the view computes it instead, it turns back into a per-frame cost that instant.
    private func rebuildSampler() {
        sampler = selectedPath.map(PathSampler.init)
        selectedPivot = selectedPath.flatMap { PathTransform.pivot(for: $0.keyframes) }
        rebuildOverlayGeometry()
    }

    /// Rebuilds the overlay geometry — called from both the path side (rebuildSampler) and the
    /// aspect side (setOutputAspect). The pivot is passed in from the snapshot just refreshed,
    /// so the least-squares solve is not run twice.
    private func rebuildOverlayGeometry() {
        pathOverlayGeometry = PathOverlayGeometry.make(
            sampler: sampler, aspect: Float(outputAspect.ratio), pivot: selectedPivot)
    }

    /// Refreshes the scene up-axis snapshot (FIX 8) — the normalization and the degenerate
    /// fallback are exactly the previous inspector computed-property logic (normalizedUpAxis —
    /// +Y also when the viewer is unwired). It does not write when the value is equal:
    /// @Observable fires a notification even for a same-value write (the same defense as
    /// ViewerModel.notePresentedCamera). Every call site is an action or notification context —
    /// none of them is inside a view body evaluation, so the camera read here creates no
    /// observation dependency.
    private func refreshSceneUpAxis() {
        let next = viewer.map { Self.normalizedUpAxis($0.camera.upAxis) } ?? SIMD3(0, 1, 0)
        guard next != sceneUpAxis else { return }
        sceneUpAxis = next
    }

    /// Discharges a deferred save (slider onEditingChanged(false), name edit focus-out).
    func persist() {
        hasUnpersistedChanges = false
        onPersist?(paths)
    }

    // MARK: Overlay mutation (Task 4)

    /// Replaces or removes the overlay. persistNow=false is the live update during a drag or
    /// slider — persistOverlay() commits it (the same save convention as path drags).
    func setOverlay(_ spec: OverlaySpec?, persistNow: Bool = true) {
        overlay = spec
        if persistNow { persistOverlay() }
    }

    func persistOverlay() {
        onPersistOverlay?(overlay)
    }

    /// Absolute URL of the overlay image (resolving the project-relative path). nil when
    /// unwired or unset.
    var overlayImageURL: URL? {
        guard let overlay, let projectDirectory else { return nil }
        return OverlayStore.url(for: overlay.imagePath, projectDirectory: projectDirectory)
    }

    /// Applies the effective preview time (skim first) as the viewport camera override — nil
    /// clears it (back to the orbit).
    /// Every apply and every clear also updates the rail ownership ledger
    /// (lastAppliedPreviewPose) (Task 2) — a clear clears the ledger too (no rail: the pairing
    /// that keeps nil ↔ nil from counting as on the rail).
    private func applyPreview() {
        guard let viewer else { return }
        guard isActive, let t = skimTime ?? previewTime, let sampler,
              !sampler.path.keyframes.isEmpty else {
            viewer.camera.overridePose = nil
            lastAppliedPreviewPose = nil
            return
        }
        let p = sampler.pose(at: t)
        let pose = CameraPose(position: p.pos, rotation: p.quat, fov: p.fov)
        viewer.camera.overridePose = pose
        lastAppliedPreviewPose = pose
    }

    // MARK: Pure decisions (the source of truth for tests)

    /// Up-axis normalization (FIX 8) — a degenerate norm≈0 falls back to +Y (the same defense
    /// as CameraController.orbitBasis and PathTransform.unitUp). The single source of truth for
    /// the sceneUpAxis snapshot's normalization.
    static func normalizedUpAxis(_ u: SIMD3<Float>) -> SIMD3<Float> {
        let n = simd_length(u)
        return n > 1e-6 ? u / n : SIMD3(0, 1, 0)
    }

    /// The capture-update target: the index of the nearest key within tolerance of the playhead
    /// (nil when there is none = insert).
    static func captureTargetIndex(playhead: Double, times: [Double],
                                   tolerance: Double) -> Int? {
        var best: (index: Int, distance: Double)?
        for (i, t) in times.enumerated() {
            let d = abs(t - playhead)
            if d <= tolerance, best == nil || d < best!.distance { best = (i, d) }
        }
        return best?.index
    }

    /// Key drag time clamp: the [0, duration] range plus a minimum gap of minSpacing from the
    /// neighbor keys — a drag cannot reorder keys, and it cannot produce a time identical to a
    /// neighbor's (a zero-width segment — an instant teleport during playback) (FIX 1 M2 — the
    /// gap is half a frame, the same constant as the capture-update threshold).
    /// The range bounds [0, duration] are not neighbors, so no gap is reserved against them.
    static func clampedDragTime(_ t: Double, index: Int, times: [Double],
                                duration: Double, minSpacing: Double = 0) -> Double {
        var lo = 0.0
        var hi = max(duration, 0)
        if index > 0 { lo = max(lo, times[index - 1] + minSpacing) }
        if index < times.count - 1 { hi = min(hi, times[index + 1] - minSpacing) }
        if lo > hi, minSpacing > 0 {
            // Neighbor gap < 2·minSpacing (over-dense keys from data predating this clamp) —
            // securing the gap is impossible, so it degrades to the gapless clamp, which only
            // preserves ordering.
            return clampedDragTime(t, index: index, times: times, duration: duration)
        }
        guard lo <= hi else { return lo }   // guard (unsorted input) — no movement possible
        return min(max(t, lo), hi)
    }

    /// Playback position: anchor + elapsed. With loop it wraps by the remainder of duration (no
    /// finished); otherwise it returns (duration, finished) once duration is reached.
    static func playbackPosition(anchorTime: Double, elapsed: Double, duration: Double,
                                 loop: Bool) -> (time: Double, finished: Bool) {
        guard duration > 0 else { return (0, true) }
        let raw = anchorTime + max(elapsed, 0)
        if loop { return (raw.truncatingRemainder(dividingBy: duration), false) }
        return raw >= duration ? (duration, true) : (raw, false)
    }
}
