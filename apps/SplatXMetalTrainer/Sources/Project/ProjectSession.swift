import Foundation
import Observation
import simd

// Cinema is the fourth tab (2026-07-26 spec footnote 2 ①) — a discoverability decision that
// reversed the in-View-tab toggle after real use. It does not appear in the initial tab decision
// (project.state) (it is always entered explicitly).
enum ProjectTab: String, CaseIterable, Identifiable { case align, train, view, cinema; var id: String { rawValue } }

/// The product of parsing sparse/0 once — for the camera fit (2026-07-20 spec §6). Consumed by
/// Tasks 4 and 5.
struct ColmapContext {
    let model: ColmapModel
    /// The camera at the middle index (count/2) of the name-sorted views — the "middle camera"
    /// selection convention.
    let middleView: ColmapView
    /// The camera at index **0** of the same name-sorted order — the capture's first frame
    /// (2026-08-04 spec §2.1). It is the anchor of the training start and result viewpoints.
    let firstView: ColmapView
    /// The mean of the points3D positions. nil when points3D is absent or fails to parse (the
    /// consumer falls back to frameToFit).
    let pointsCentroid: SIMD3<Float>?

    /// `firstView` defaults to `middleView` so that the three-argument construction
    /// (model:middleView:pointsCentroid:) written before this round keeps compiling unchanged —
    /// those call sites build single-view fixtures where the two are the same value anyway.
    /// `load` below always passes both explicitly.
    init(model: ColmapModel, middleView: ColmapView, firstView: ColmapView? = nil,
         pointsCentroid: SIMD3<Float>?) {
        self.model = model
        self.middleView = middleView
        self.firstView = firstView ?? middleView
        self.pointsCentroid = pointsCentroid
    }

    /// Parses sparse/0 (nil on failure or empty views). A pure function — ProjectSession layers the
    /// cache on top.
    static func load(sparse0: URL) -> ColmapContext? {
        guard let model = try? ColmapCameraReader.read(sparseDir: sparse0),
              !model.views.isEmpty else { return nil }
        let sorted = model.views.sorted { $0.name < $1.name }
        let middle = sorted[sorted.count / 2]
        let points = (try? ColmapCameraReader.loadPoints(sparse0: sparse0)) ?? []
        var centroid: SIMD3<Float>?
        if !points.isEmpty {
            // Sums in Double to avoid Float accumulation error over many points.
            let sum = points.reduce(SIMD3<Double>()) { $0 + SIMD3<Double>($1) }
            centroid = SIMD3<Float>(sum / Double(points.count))
        }
        return ColmapContext(model: model, middleView: middle, firstView: sorted[0],
                             pointsCentroid: centroid)
    }
}

/// The scope of one open project. It is created when a project is opened and discarded wholesale
/// when returning Home — the VMs (Train/Viewer) are bound to the session lifetime, so no residue of
/// the previous project (chart or viewer state) is left behind.
@MainActor @Observable final class ProjectSession {
    var project: Project
    private(set) var directory: URL   // changes go only through applyRename (app round 4 spec item 6)
    var tab: ProjectTab

    let trainVM = TrainViewModel()
    let viewerVM = ViewerModel()
    let alignVM = AlignViewModel()
    /// Cinema mode (2026-07-26 spec) — camera path editing inside the View tab. Paths are persisted
    /// in project.cameraPaths, and the save wiring (onPersist) is connected by AppModel.openProject,
    /// which owns the store.
    let cinemaVM = CinemaViewModel()
    /// The video export job (Cinema Task 5) — the dedicated-Thread executor for sx_render_session.
    /// lockedTab is authoritative for exclusion against training and align (the isExporting term
    /// below).
    let renderJob = RenderJob()

    /// Preventing system idle sleep during a long-running task (audit cleanup Task 6). SleepGuard
    /// solely owns the token's lifetime, and this session watches lockedTab (the authoritative gate
    /// of the tasks) through observation and merely declares the state it wants — the start and
    /// every termination path (completion, cancellation, thermal abort, and error) already
    /// converge on lockedTab, so the activity lifetime code is not scattered across paths.
    @ObservationIgnored private let sleepGuard = SleepGuard()

    init(project: Project, directory: URL) {
        self.project = project
        self.directory = directory
        self.tab = switch project.state {
        case .imported: .align
        case .aligned: .train
        case .trained: .view
        }
        trainVM.datasetURL = directory   // the project directory = the COLMAP bundle (Task 12 also moved the output path here)
        trainVM.draft.timelapse = project.timelapse   // the persisted timelapse settings → the training draft (spec §3.6)
        cinemaVM.viewer = viewerVM       // the target the skimmer/playback pose override is applied to
        cinemaVM.projectDirectory = directory   // the target for resolving and copying overlay images (Task 4)
        cinemaVM.load(paths: project.cameraPaths, overlay: project.overlay,
                      outputAspect: project.outputAspect)
        armSleepGuard()
    }

    /// A lockedTab change → declaring the SleepGuard state (audit cleanup Task 6). The apply block
    /// of withObservationTracking reads lockedTab (→ trainVM.isTraining, alignVM.isRunning, and
    /// renderJob.isExporting) to register the dependency, and on a change it rearms and
    /// re-evaluates on the MainActor. update is idempotent, so duplicate firing from unrelated
    /// changes is harmless. When the session is discarded, SleepGuard.deinit is the line of
    /// defense.
    private func armSleepGuard() {
        withObservationTracking {
            sleepGuard.update(reason: SleepGuard.activityReason(lockedTab: lockedTab))
        } onChange: { [weak self] in
            Task { @MainActor [weak self] in self?.armSleepGuard() }
        }
    }

    /// The **authoritative gate** for stage locking during a long-running task (app round 5 spec
    /// item 8). When non-nil it pins the app to that stage — StageNav, the ⌥1/2/3 commands, and
    /// goHome (back navigation) all reference it.
    /// Even if a per-button gate (align's Start or train's canStart, for example) misses the other
    /// task, a locked tab makes it unreachable, so it is safe.
    var lockedTab: ProjectTab? {
        Self.lockedTab(isTraining: trainVM.isTraining, isAligning: alignVM.isRunning,
                       isExporting: renderJob.isExporting)
    }

    /// The pure decision. Concurrent execution is structurally impossible (the locks block each
    /// other's start paths), but the decision order is pinned with training first — it must be
    /// deterministic even in an unreachable state.
    /// isExporting (Cinema Task 5) locks .cinema (footnote 2 ① — the export is started from the
    /// Cinema tab, so that is the tab to lock) — it blocks reaching the Train tab or goHome at all
    /// during an export, enforcing sx_render_session's "must not run concurrently with training"
    /// contract through the UI structure.
    static func lockedTab(isTraining: Bool, isAligning: Bool,
                          isExporting: Bool = false) -> ProjectTab? {
        if isTraining { return .train }
        if isAligning { return .align }
        if isExporting { return .cinema }
        return nil
    }

    /// Syncing the session after a project rename (app round 4 spec item 6). It is called after
    /// ProjectStore.rename has moved the folder — if directory and **trainVM.datasetURL** are not
    /// updated to the new path, the next training looks at the old one. colmapContextCache is kept
    /// — it holds only parsed data (ColmapModel, middleView, and the centroid) rather than paths,
    /// and a rename is a folder move, so the contents of sparse/0 are unchanged. RunRecord paths
    /// are also recorded relative to the project, so the next openRun recovers naturally with the
    /// new directory (see the Project.swift comment).
    func applyRename(project: Project, directory: URL) {
        self.project = project
        self.directory = directory
        trainVM.datasetURL = directory
        cinemaVM.projectDirectory = directory   // so the overlay's relative path points at the new folder
    }

    /// Selecting a run from the history or finishing training → switch to the View tab and load it
    /// into the viewer (the camera fit on a run switch — spec §6).
    func showRun(_ record: RunRecord) {
        // openRun updates currentRun only after the Task starts, so racing with
        // RootDetail.openLatestIfNeeded (guarded by currentRun == nil), which the tab-switch
        // onChange calls, can load the latest run twice (the second load reverts the fit to
        // frameToFit — it is not idempotent).
        // A synchronous pre-assignment locks the guard deterministically (app round 2 Task 5).
        viewerVM.currentRun = record
        tab = .view
        Task {
            await viewerVM.openRun(record, projectDirectory: directory,
                                   colmapContext: loadColmapContext())
        }
    }

    /// Presents the result of a run that **just finished training** (2026-08-04 spec §2.1).
    /// Identical to showRun except that it arms the first-camera fit anchor first, so the
    /// result is shown from the same viewpoint the live view started at — "start and finish
    /// frame the same picture".
    /// It is a separate entry point rather than a branch inside showRun on purpose: showRun is
    /// also the history-open and post-delete auto-open path, and those must keep the middle
    /// camera behavior exactly (spec §2.3).
    /// Ordering: showRun → openRun → open() (whose frameToFit clears overridePose) →
    /// applyCameraFit is a fixed order, so the anchor armed here becomes the **final** state.
    /// CameraFitAnchorTests pins that ordering, and openRun's defer disarms the anchor if the
    /// load fails before the fit is reached.
    func showFinishedRun(_ record: RunRecord) {
        viewerVM.nextFitAnchor = .firstCameraExact
        showRun(record)
    }

    /// Deleting a run (app round 2 Task 7): remove the runs/<id>/ folder → remove the run from the
    /// manifest and recompute the state (the pure function Project.removingRun) → save. On a folder
    /// deletion failure it warns on the console and continues with the manifest cleanup — so that
    /// the most common failure (a folder already deleted by hand) does not leave a ghost run row.
    /// If it was the open run, the viewer is emptied and the latest remaining run is opened
    /// automatically (or left empty if there is none).
    /// The caller (the RUNS section) disables deletion during training — blocking contention over
    /// the live host (handed over from Task 5).
    func deleteRun(_ record: RunRecord, model: AppModel) {
        let fm = FileManager.default
        do {
            try fm.removeItem(at: directory.appending(path: "runs/\(record.id)"))
        } catch {
            model.console.append("⚠ run folder delete failed: \(error.localizedDescription)")
        }
        let hasSparse = fm.fileExists(atPath: directory.appending(path: "sparse/0").path)
        project = project.removingRun(id: record.id, hasSparse: hasSparse)
        do { try model.store.save(project) }
        catch { model.console.append("⚠ manifest save failed: \(error.localizedDescription)") }
        model.console.append("− Deleted run \(record.id)")
        if viewerVM.currentRun?.id == record.id {
            viewerVM.clear()
            if let latest = project.latestRun { showRun(latest) }
        }
    }

    /// Persisting the Inspector's Auto-level toggle (2026-08-04 spec §1.5, FIX 1/2 — audit review
    /// of Task 4). The write-back + save used to live directly in InspectorView's Toggle binding,
    /// making it the only View body in the app that mutates `project` and calls `store.save`
    /// itself — every other manifest write goes through a session/view-model method (deleteRun
    /// above is the precedent: the View calls the method, the method owns the try/catch).
    /// A save failure here is not a "settings" write to shrug off on the console (which starts
    /// collapsed, AppModel.consoleExpanded = false) — it changes the coordinate system the next
    /// Align run actually writes to sparse/0, so it is surfaced the same way AlignViewModel.start
    /// and TrainViewModel surface a manifest-save failure ("Audit cleanup Task 5"): through
    /// alignVM.errorMessage, which AlignView already renders as a red banner.
    func setAutoLevelOnAlign(_ on: Bool, model: AppModel) {
        project.autoLevelOnAlign = on
        do { try model.store.save(project) }
        catch {
            alignVM.errorMessage = "Auto-level setting could not be saved"
                + " — it may revert after the app restarts."
                + " (\(error.localizedDescription))"
            model.console.append("⚠ auto-level setting save failed: \(error.localizedDescription)")
        }
    }

    // The COLMAP parse cache for the camera fit. .none = not attempted, .some(nil) = a confirmed
    // failure (cached too — a failed parse of the same file fails again).
    // The cache is valid only until sparse/0 is rewritten, which a re-align does mid-session
    // (AlignViewModel.start) — invalidateColmapContext below is what keeps the two in step.
    private var colmapContextCache: ColmapContext??

    /// Lazily parses sparse/0 once and returns the camera fit context. Failure gives nil (the
    /// frameToFit fallback).
    func loadColmapContext() -> ColmapContext? {
        if let cached = colmapContextCache { return cached }
        let ctx = ColmapContext.load(sparse0: directory.appending(path: "sparse/0"))
        colmapContextCache = .some(ctx)
        return ctx
    }

    /// Drops the parse cache after sparse/0 has been rewritten (a re-align). **Must be called on
    /// every successful write**, not only when auto-level ran: the pose refinement alone already
    /// moves the fit, and with auto-level on (the default) the write rotates the whole world, so a
    /// stale context would feed the *previous* coordinate system to the `.firstCameraExact`
    /// viewport override, the timelapse capture anchor, and the Cinema trajectory — silently, for
    /// the rest of the session. Reachable with the mouse alone: train (fills the cache) →
    /// re-align → train.
    func invalidateColmapContext() {
        colmapContextCache = nil
    }
}
