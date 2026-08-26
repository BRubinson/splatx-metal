import Foundation
import Observation

/// The align tab's state. On a successful run it bakes sparse/0, updates the manifest (state and
/// alignSummary), and then moves automatically to the Train tab. A re-align overwrites sparse/0 —
/// existing runs are preserved (spec §6).
@MainActor @Observable final class AlignViewModel {
    var isRunning = false
    var fraction: Double = 0
    var errorMessage: String?
    private var task: Task<Void, Never>?

    func start(session: ProjectSession, model: AppModel) {
        guard !isRunning else { return }
        guard AlignEngine.isAvailable() else {
            errorMessage = AlignError.notSupported.errorDescription
            return
        }
        isRunning = true; errorMessage = nil; fraction = 0
        let dir = session.directory
        let sequential = Self.usesSequentialMatching(session.project.sourceKind)
        model.console.append("▶ Align started: \(session.project.name) (\(session.project.imageCount) images)")
        task = Task {   // inherits @MainActor — it leaves only at the engine await
            do {
                let result = try await AlignEngine.run(
                    imagesDir: dir.appendingPathComponent("images", isDirectory: true),
                    sequentialHint: sequential) { f in
                    Task { @MainActor [weak self] in self?.fraction = f }
                }
                // Auto-level (2026-08-04 spec §1.2): the correction happens **between** the
                // engine result and the write, so sparse/0 is written level from the start —
                // there is no rewrite path, and training, the viewer, Cinema and every export
                // inherit the corrected coordinates for free. It is a pure world rotation with a
                // compensating pose change, so the images the trainer sees are unchanged
                // (GroundLevelTests pins X_cam = R·p + t as invariant).
                let level = GroundLevel.applied(enabled: session.project.autoLevelOnAlign,
                                                frames: result.frames, points: result.points)
                try ColmapTextWriter.write(
                    frames: level.frames, points: level.points,
                    to: dir.appendingPathComponent("sparse/0", isDirectory: true))
                // sparse/0 now differs from whatever the session parsed earlier, so the camera-fit
                // cache has to go (FIX 6). The write throws on failure, which skips this line —
                // exactly right, because then the file is untouched and the cache still matches it.
                session.invalidateColmapContext()
                if let line = Self.levelConsoleLine(tiltDegrees: level.tiltDegrees) {
                    model.console.append(line)
                }
                session.project.alignSummary = AlignSummary(
                    poseCount: result.frames.count,
                    pointCount: result.points.count,
                    alignSeconds: result.seconds,
                    levelTiltDegrees: level.tiltDegrees)
                if session.project.state == .imported { session.project.state = .aligned }
                // Audit cleanup Task 5: a manifest save failure is the failure where "align
                // finished but the aligned state and the summary are lost after a restart" — as in
                // TrainViewModel, it is surfaced inline through errorMessage rather than as a
                // single console line.
                do { try model.store.save(session.project) }
                catch {
                    errorMessage = "Align finished, but saving the project manifest failed"
                        + " — align status may be lost after the app restarts."
                        + " (\(error.localizedDescription))"
                    model.console.append("⚠ manifest save failed: \(error.localizedDescription)")
                }
                model.console.append(
                    "■ Align finished: \(result.frames.count) poses · \(result.points.count) points"
                    + (result.skippedImages > 0 ? " · \(result.skippedImages) skipped" : ""))
                if result.points.count < 500 {
                    model.console.append("⚠ Sparse point cloud is very small (\(result.points.count)) — quality may suffer.")
                }
                isRunning = false
                session.tab = .train
                // The system notification (app round 5 item 6) — including the pose and point
                // counts.
                AppNotifier.shared.post(AppNotificationText.alignFinished(
                    project: session.project.name,
                    poseCount: result.frames.count, pointCount: result.points.count))
            } catch is CancellationError {
                isRunning = false
                model.console.append("· Align cancelled")
                // A cancellation is an action the user took themselves — no notification (app
                // round 5 item 6).
            } catch {
                isRunning = false
                let message = (error as? LocalizedError)?.errorDescription ?? "\(error)"
                errorMessage = message
                model.console.append("✗ Align failed: \(message)")
                AppNotifier.shared.post(AppNotificationText.alignFailed(
                    project: session.project.name, error: error))
            }
        }
    }

    /// Whether to hand `PhotogrammetrySession` the `.sequential` sample ordering.
    ///
    /// The hint is a promise about *name order*, not about the source: align feeds the session an
    /// images folder it enumerates by file name, so `.sequential` is correct exactly when adjacent
    /// names are adjacent viewpoints. It is a speed hint (the matcher searches neighbours first),
    /// and a wrong one costs accuracy on a shuffled set, so it is claimed only where the ordering
    /// is produced rather than assumed.
    ///
    /// - `.video` — the extractor writes `frame_%05d.jpg` precisely so that lexicographic order is
    ///   timeline order (`VideoFrameExtractor`'s header states this).
    /// - `.capture` — the phone writes `0001.jpg`, `0002.jpg`, … in capture order in **every**
    ///   mode; photo and video bundles differ in folder name and a manifest string, not in frame
    ///   naming (spec §1), so one answer covers both. A LiDAR bundle imports `.aligned` and does
    ///   not normally reach here at all — but a deliberate re-align gets the same ordering, since
    ///   the naming is the phone's regardless of what the poses were.
    /// - `.images` — a folder from anywhere. `IMG_0413.jpg` next to `DSC_9001.jpg` sorts by
    ///   whatever the camera named them, which says nothing about where the photographer stood.
    /// - `.colmap` — arrives solved and `.aligned`; a re-align of a third-party reconstruction has
    ///   no ordering guarantee either.
    ///
    /// `nonisolated` because it reads a `ProjectSourceKind` and nothing else — none of this view
    /// model's state is involved, so requiring the main actor to answer it would be a fiction that
    /// every caller and test then has to work around. Tests: CaptureImportFlowTests.
    nonisolated static func usesSequentialMatching(_ kind: ProjectSourceKind) -> Bool {
        switch kind {
        case .video, .capture: true
        case .images, .colmap: false
        }
    }

    /// The console line for an applied auto-level correction — nil when nothing was rotated (the
    /// toggle was off, or the tilt was under `GroundLevel.minTiltDegrees`). The same optional is
    /// what goes into `AlignSummary.levelTiltDegrees`, so the log and the summary always agree.
    /// Tests: AlignAutoLevelWiringTests.
    static func levelConsoleLine(tiltDegrees: Double?) -> String? {
        guard let d = tiltDegrees else { return nil }
        return String(format: "· Auto-level: scene rotated %.1f° to level the ground", d)
    }

    func cancel() { task?.cancel() }
}
