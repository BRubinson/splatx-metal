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
        let sequential = session.project.sourceKind == .video
        model.console.append("▶ Align started: \(session.project.name) (\(session.project.imageCount) images)")
        task = Task {   // inherits @MainActor — it leaves only at the engine await
            do {
                let result = try await AlignEngine.run(
                    imagesDir: dir.appendingPathComponent("images", isDirectory: true),
                    sequentialHint: sequential) { f in
                    Task { @MainActor [weak self] in self?.fraction = f }
                }
                try ColmapTextWriter.write(
                    frames: result.frames, points: result.points,
                    to: dir.appendingPathComponent("sparse/0", isDirectory: true))
                session.project.alignSummary = AlignSummary(
                    poseCount: result.frames.count,
                    pointCount: result.points.count,
                    alignSeconds: result.seconds)
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

    func cancel() { task?.cancel() }
}
