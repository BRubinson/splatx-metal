import Foundation
import Metal
import Observation
import SplatXMetal

@MainActor @Observable final class TrainViewModel {
    var draft = TrainOptionsDraft()
    // The stored preset property was removed (app round 3 spec §4) — the view derives the
    // highlight through TrainPreset.matching(draft), and the only selection action is apply
    // (filling in the values).
    /// The project directory (= the COLMAP bundle). ProjectSession injects it.
    var datasetURL: URL?
    var isTraining = false
    var errorMessage: String?
    let store = TrainProgressStore()

    private var task: Task<Void, Never>?
    private let trainer = SplatXTrainer()

    // MARK: In-memory live streaming (app round 3 Task 2 — spec §3. It replaces the round 2 file
    // polling)

    /// The snapshot callback interval (in iterations). spec §8 — a fixed value not exposed in the
    /// UI (round 2's live.ply 250 → round 3's 50).
    private static let snapshotEvery = 50
    /// The streamer alive for the duration of a training run. Created at start, and stopped and
    /// released at completion or on error (returning the staging and MetalBuffer memory).
    /// (@ObservationIgnored — observation is handled by streamer.state, and it permits stored
    /// access from the nonisolated deinit. The view reads it through liveIter.)
    @ObservationIgnored private var streamer: LiveSplatStreamer?

    /// The LIVE badge's iter — the iter of the snapshot that reached the screen (swap complete)
    /// (spec §3: switching the badge's source).
    var liveIter: Int { streamer?.state.latestIter ?? 0 }

    deinit {
        // A line of defense: the UI blocks returning Home while training, but this cleanly cuts the
        // pipeline when the session is discarded.
        streamer?.stop()
    }

    var canStart: Bool { datasetURL != nil && draft.isValid && !isTraining }

    /// The runs/ folder name stamp — fixed to UTC (excluding locale and time zone effects;
    /// lexicographic order = chronological order).
    private static let stampFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyyMMdd-HHmmss"
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC")
        return f
    }()

    func start(model: AppModel, session: ProjectSession) {
        guard let dataset = datasetURL, draft.isValid, !isTraining else { return }
        guard session.project.state != .imported else { errorMessage = "Run Align first."; return }
        let sparse = dataset.appendingPathComponent("sparse/0")
        let hasCameras = ["cameras.bin", "cameras.txt"].contains {
            FileManager.default.fileExists(atPath: sparse.appendingPathComponent($0).path)
        }
        guard hasCameras else { errorMessage = "No cameras.bin/txt in sparse/0."; return }

        let stamp = Self.stampFormatter.string(from: Date())
        let runDir = dataset.appendingPathComponent("runs/\(stamp)", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: runDir, withIntermediateDirectories: true)
        } catch {
            errorMessage = "Cannot create run directory: \(error.localizedDescription)"
            return
        }
        // Round 4: the training output is always PLY — SPZ is produced only through the View tab's
        // Export path.
        let outURL = runDir.appendingPathComponent("output.ply")
        let previewURL = runDir.appendingPathComponent("preview.png")
        let ckptURL = draft.saveCheckpoint ? runDir.appendingPathComponent("checkpoint.ckpt") : nil
        var options = draft.buildOptions(outputURL: outURL, previewURL: previewURL,
                                         checkpointURL: ckptURL, metallibURL: BundlePaths.metallibURL)

        store.reset(); errorMessage = nil; isTraining = true
        // The live view's viewpoint: the middle camera fit is applied immediately at start (the
        // camera is independent of the host, so it is valid even without one).
        // Round 3: the host is created by acquireRenderer on the first snapshot swap and does not
        // go through open(), so frameToFit does not overwrite this fit — the round 2 re-application
        // logic became unnecessary.
        session.viewerVM.applyCameraFit(context: session.loadColmapContext())
        // The in-memory live streaming wiring (app round 3 Task 2 — spec §3). If there is no Metal
        // device (in theory), training proceeds without the live view. The handler is called
        // synchronously on the training loop thread, and the streamer only copies and returns
        // immediately (the snapshot pointer lifetime contract).
        if let device = MTLCreateSystemDefaultDevice() {
            let streamer = LiveSplatStreamer(
                device: device,
                acquireRenderer: { [weak session] in
                    // Lazily creates the host on the MainActor at the first swap — sharing the same
                    // host as the viewer unifies TrainView's live view gate (host != nil) with the
                    // View tab's rendering.
                    // A creation failure returns nil (retried on the next snapshot — live is
                    // non-fatal).
                    guard let session else { return nil }
                    let viewer = session.viewerVM
                    if viewer.host == nil {
                        viewer.host = try? SplatRendererHost(device: device)
                    }
                    return viewer.host?.renderer
                },
                onSwap: { [weak session] count in
                    session?.viewerVM.currentSplats = count   // the count shown in the Inspector and status bar
                })
            self.streamer = streamer
            options.snapshotEvery = Self.snapshotEvery
            options.snapshotHandler = { snap in streamer.ingest(snap) }
        }
        model.job = .training(iter: 0, total: draft.iterations)
        model.console.append("▶ Training started: \(session.project.name) · \(draft.iterations)it → runs/\(stamp)")
        let storeCB = store.callback()
        let cb: @Sendable (SplatXTrainer.Progress) -> Void = { [weak model, storeCB] p in
            storeCB(p)
            Task { @MainActor in
                guard let model else { return }
                model.job = .training(iter: p.iteration, total: p.runUntil)
                model.console.append(
                    "iter \(p.iteration)/\(p.runUntil)  loss \(String(format: "%.5f", p.loss))"
                    + (p.psnr.map { String(format: "  PSNR %.2f dB", $0) } ?? ""))
            }
        }
        let draftSnapshot = draft
        task = Task { [weak self] in
            // Training speed Task 3: creating the images_<factor>/ prescale once — it structurally
            // removes the core's downscale-on-load fallback (decoding the full-resolution original
            // and shrinking it on every step) (the core consumes the folder in preference when it
            // exists — the ImagePrescaler header comment).
            // It is a synchronous job of seconds to tens of seconds, so it is run to completion
            // detached, off the MainActor, before entering the training thread. A failure is
            // non-fatal (nil) — training continues through the core fallback.
            if draftSnapshot.factor > 1 {
                let imagesDir = dataset.appendingPathComponent("images")
                let factor = draftSnapshot.factor
                let console: @Sendable (String) -> Void = { [weak model] line in
                    Task { @MainActor in model?.console.append(line) }
                }
                await Task.detached(priority: .userInitiated) {
                    ImagePrescaler.prescaleIfNeeded(
                        imagesDir: imagesDir, factor: factor,
                        progress: { done, total in
                            // Preventing console flooding — every 25 images plus at completion.
                            if done % 25 == 0 || done == total {
                                console("Preparing images_\(factor): \(done)/\(total)")
                            }
                        },
                        log: console)
                }.value
            }
            do {
                let result = try await self?.trainer.train(dataDirectory: dataset, options: options, progress: cb)
                await MainActor.run {
                    guard let self else { return }
                    let streamerRef = self.streamer   // held so we can wait for completion before showRun
                    self.teardownStreamer()
                    self.isTraining = false
                    model.job = .idle
                    guard let r = result else { return }
                    // A partial completion (thermal abort or stop) also has an output, so it is
                    // recorded together with its outcome.
                    let record = RunRecord(
                        id: stamp,
                        outputFile: outURL.lastPathComponent,
                        previewFile: FileManager.default.fileExists(atPath: previewURL.path)
                            ? previewURL.lastPathComponent : nil,
                        outcome: r.outcome.rawValue,
                        finalPSNR: r.finalPSNR,
                        splatCount: r.splatCount,
                        optionsSummary: Self.summarize(draftSnapshot),
                        createdAt: Date(),
                        // Cinema Task 5: the record that grounds sx_render_session AA consistency
                        // (the header contract — a standard PLY has no AA flag, so the caller
                        // passes the setting used at training time).
                        antialiased: draftSnapshot.antialiased)
                    session.project.runs.append(record)
                    session.project.state = .trained
                    // Audit cleanup Task 5: a manifest save failure is a result-loss failure where
                    // "training looked successful but the run disappears from the RUNS list after
                    // a restart". The console is collapsed by default (consoleExpanded=false), so
                    // a single console line would let it vanish silently — it is surfaced the same
                    // way as the other failures in this function (:61 the run directory and :54
                    // the missing cameras), inline through errorMessage.
                    var manifestSaveError: String?
                    do { try model.store.save(session.project) }
                    catch {
                        manifestSaveError = error.localizedDescription
                        model.console.append("⚠ manifest save failed: \(error.localizedDescription)")
                    }
                    model.splatCount = r.splatCount
                    model.console.append(
                        "■ Finished (\(r.outcome.rawValue)): N \(r.splatCount)"
                        + (r.finalPSNR.map { String(format: " · PSNR %.2f dB", $0) } ?? ""))
                    if r.outcome != .completed {
                        self.errorMessage = "Finished: \(r.outcome.rawValue) (partial result saved)"
                    }
                    // Result loss is more severe than the partial notice — assigned later so it
                    // takes display precedence.
                    if let manifestSaveError {
                        self.errorMessage = "Training finished, but saving the project manifest failed"
                            + " — this run will disappear from the RUNS list after the app restarts."
                            + " (\(manifestSaveError))"
                    }
                    // The system notification (app round 5 item 6) — only for a normal or partial
                    // completion (thermal abort).
                    // cancelled is an action the user took themselves, so the message builder
                    // returns nil.
                    AppNotifier.shared.post(AppNotificationText.trainingFinished(
                        project: session.project.name, outcome: r.outcome,
                        splatCount: r.splatCount, finalPSNR: r.finalPSNR))
                    // A normal or partial (thermal abort) completion = the result watched live is
                    // carried over into the View tab (spec §3). A cancellation (.cancelled) keeps
                    // the tab — the run is opened manually from the history.
                    // The streamer pipeline is run to completion first (the stop was requested
                    // above) to block the chunk replacement race between a live swap and
                    // openRun(output) (a stale live chunk overlapping or overwriting the final
                    // output) — the same purpose as the round 2 liveReloadTask await.
                    if r.outcome != .cancelled {
                        Task { @MainActor in
                            await streamerRef?.waitUntilIdle()
                            session.showRun(record)
                        }
                    }
                }
            } catch {
                await MainActor.run {
                    self?.teardownStreamer()
                    self?.isTraining = false; model.job = .idle
                    self?.errorMessage = "Training failed: \(error)"
                    model.console.append("✗ Training failed: \(error)")
                    // Summarizing the error (avoiding a raw enum dump) is owned by the message
                    // builder — FIX 3 item 4.
                    AppNotifier.shared.post(AppNotificationText.trainingFailed(
                        project: session.project.name, error: error))
                }
            }
        }
    }

    func cancel() { task?.cancel(); trainer.requestStop() }
    // Cancellation has no separate streamer cleanup path — a cooperative stop also has train()
    // "return" with outcome == .cancelled (it does not throw), so teardownStreamer in the
    // completion handler above is the common exit.

    /// Stops and releases the streamer (returning the memory of the two staging buffers and the
    /// two MetalBuffers). If waiting for an in-flight cycle to run to completion is needed, hold
    /// the reference before calling and await waitUntilIdle() separately.
    private func teardownStreamer() {
        streamer?.stop()
        streamer = nil
    }

    static func summarize(_ d: TrainOptionsDraft) -> String {
        "factor \(d.factor) · \(d.iterations)it · \(d.mcmc ? "MCMC" : "no-MCMC") · cap \(d.capMax/1000)k · SH\(d.shDegree)"
    }
}
