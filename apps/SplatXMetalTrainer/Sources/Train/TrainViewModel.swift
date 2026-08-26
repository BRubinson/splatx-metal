import CoreGraphics
import Foundation
import Metal
import MetalSplatter
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

    // MARK: Live viewport size (2026-08-05 spec §2.4)

    /// The Train tab's live viewport size in **points**, reported by TrainView's GeometryReader
    /// (`noteViewportSize`). `.zero` until the first layout.
    ///
    /// It exists because until this round **nothing in the app kept that size**: `drawableSize` is
    /// consumed inline in SplatRendererHost and never stored, and `ViewportRegistry` reads
    /// `view.bounds` at event-routing time without holding it (its instance is private to
    /// `DragMonitor.shared`). So there was no path from "what the user is looking at" to the
    /// timelapse frame, and the ratio came from the dataset while the pose came from the viewport.
    ///
    /// Points rather than pixels is deliberate, not a shortcut: the only consumer takes a **ratio**
    /// (`TimelapseSettings.frameSize`) and the backing scale cancels out of a ratio. Points are also
    /// available from layout alone, whereas a pixel size would mean asking the MTKView's drawable —
    /// which does not exist yet at the moment this value is read (`start`, before the first
    /// snapshot swap creates the live host).
    ///
    /// Not optional: `.zero` already means "no ratio here" to `TimelapseSettings.aspectPair`, which
    /// falls through to the dataset intrinsics (fallback ①). An `Optional` would encode the same
    /// state twice.
    private(set) var viewportSize: CGSize = .zero

    /// Records a size reported by the viewport's GeometryReader. Called from `.onAppear` and
    /// `.onChange(of:)` **only** — never during a body evaluation (spec §2.4).
    func noteViewportSize(_ size: CGSize) {
        guard let next = Self.viewportSizeUpdate(reported: size, current: viewportSize) else { return }
        viewportSize = next
    }

    /// The value to store for a newly reported viewport size, or nil to keep what is already there.
    /// Pure because the three rejection rules are the part that can be wrong, and the caller is a
    /// SwiftUI modifier — which this app does not test.
    ///
    /// ① An unchanged size is not written. This is `@Observable` state, so a redundant write is a
    ///    redundant invalidation of every view reading it. `.onChange` filters equal values by
    ///    itself, but `.onAppear` does not, and it fires again on every switch of the
    ///    live/placeholder conditional inside the reader.
    /// ② A non-finite size is dropped rather than stored. `aspectPair` would catch it (its
    ///    `isFinite` guard exists because `Int(CGFloat.nan)` traps), but it would catch it by
    ///    silently answering "dataset" for the rest of the session with nothing left to explain why.
    /// ③ A usable size is never replaced by an unusable one (< 1pt). The damage is asymmetric: the
    ///    value is read exactly once, at `start()`, and a zero read there produces a timelapse in a
    ///    ratio the user did not pick — the very failure this round exists to fix. The last size the
    ///    viewport actually had is a better answer than that even if the viewport has since
    ///    collapsed, so a transient zero from a layout pass cannot cost anything.
    static func viewportSizeUpdate(reported: CGSize, current: CGSize) -> CGSize? {
        // NaN != NaN is true, so an all-NaN report reaches the isFinite guard below rather than
        // being swallowed as "unchanged".
        guard reported != current else { return nil }
        guard reported.width.isFinite, reported.height.isFinite else { return nil }
        func usable(_ s: CGSize) -> Bool { s.width >= 1 && s.height >= 1 }
        guard usable(reported) || !usable(current) else { return nil }
        return reported
    }

    // MARK: Timelapse run state

    /// The timelapse recorder alive for the duration of a run (2026-08-04 spec §3). nil when the
    /// timelapse is off. Handed to the encoding thread at completion and released here.
    @ObservationIgnored private(set) var timelapseRecorder: TimelapseRecorder?
    /// Whether this run is recording a timelapse — the gate for the "shoot from here" button
    /// (observed, unlike the recorder itself).
    private(set) var timelapseActive = false

    /// The **pixel frame size** the running timelapse is actually capturing at, nil when no
    /// timelapse is running. Task 3's letterbox preview reads it to dim what is being framed out.
    ///
    /// Derived from the recorder's `Config`, not from `draft.timelapse.aspectSource`: the config is
    /// immutable and is what the capture thread reads (사용자 결정 3), so it is the only value that
    /// cannot disagree with the frames landing on disk. The chosen source is not even a reliable
    /// predictor of it — `aspectPair`'s two fallbacks can hand `.viewport` the dataset's ratio.
    /// That also makes this "the viewport **at start**" (spec §2.6): resizing the window mid-run
    /// changes neither the config nor this value, which is why a mask can appear afterwards.
    ///
    /// **The size, not the ratio** (timelapse-viewport-aspect FIX 1). This used to be a bare
    /// `CGFloat` ratio, and that was the reason `TrainView.timelapseMask` could not size its own
    /// near-match threshold: the drift the threshold has to absorb is a *rounding* artefact of
    /// `TimelapseSettings.frameSize`, so its size in points depends on how many pixels the long
    /// side has. 812×480 and 8120×4800 are the same ratio and admit ten-fold different drifts.
    /// A `CGSize` of whole pixel counts rather than a `(width: Int, height: Int)` pair only
    /// because every consumer is `CGRect` arithmetic; nothing here is fractional.
    private(set) var captureFrameSize: CGSize?

    /// The single writer of the three coupled timelapse-run fields. They move together because a
    /// `captureFrameSize` left behind would keep Task 3's letterbox dimming a viewport with no
    /// run behind it, and a run ends on three separate paths (normal completion, a thrown failure,
    /// and the recorder-creation guard in `start`) — one funnel is what keeps a fourth path from
    /// forgetting one of them.
    ///
    /// No guard against a degenerate `Config` size here: `frameSize` cannot return a non-positive
    /// dimension, and `timelapseMask` refuses one anyway rather than dividing a `CGRect` by zero.
    private func setTimelapseRun(_ recorder: TimelapseRecorder?) {
        timelapseRecorder = recorder
        timelapseActive = recorder != nil
        captureFrameSize = recorder.map {
            CGSize(width: $0.config.width, height: $0.config.height)
        }
    }

    /// Whether an encode is in flight (2026-08-05 spec §2.2/§2.3). It exists because
    /// `RunRecord.timelapseFile` is written only when the encode **finishes**: from the moment
    /// training ends until then — seconds for a short GIF, minutes for a long MP4 — the record
    /// says nil, and the View tab has already switched to that run. Without this flag the
    /// inspector would print "—" for a timelapse that is actively being produced, which reads as
    /// "it was never made". The encode does not join `lockedTab` (§3.5: it only touches files),
    /// so this is deliberately *not* a lock — it is a display state and nothing else.
    /// It is scoped to the session, not to a run id: the encode belongs to the run that just
    /// finished, and `timelapseState` resolves the ambiguity by letting an on-disk file win (see
    /// RunPropertiesView.timelapseState).
    private(set) var isEncodingTimelapse = false

    /// The cancellation flag of the timelapse encode (a plain class, so deinit may touch it).
    /// It is reset at the start of each encode.
    private let timelapseCancel = RenderCancelFlag()

    /// The LIVE badge's iter — the iter of the snapshot that reached the screen (swap complete)
    /// (spec §3: switching the badge's source).
    var liveIter: Int { streamer?.state.latestIter ?? 0 }

    deinit {
        // A line of defense: the UI blocks returning Home while training, but this cleanly cuts the
        // pipeline when the session is discarded.
        streamer?.stop()
        // A timelapse encode in flight is cancelled when the project is closed or the app quits
        // (spec §3.5) — the encoder deletes the partial file on the cancellation path.
        timelapseCancel.cancel()
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
        // The timelapse settings persist at project level (spec §3.6). Written **at start** so
        // that a run which is cancelled still keeps the user's choice.
        if session.project.timelapse != draft.timelapse {
            session.project.timelapse = draft.timelapse
            do { try model.store.save(session.project) }
            catch {
                model.console.append("⚠ timelapse settings save failed: \(error.localizedDescription)")
            }
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
        // 2026-08-04 spec §2.1: the anchor is the capture's **first** camera (name-sorted index
        // 0), injected exactly — position, orientation (roll included) and fov — instead of the
        // middle-camera orbit approximation, so the live view opens on the same picture as
        // images/ frame 0. The flag is one-shot and is consumed by the fit on the next line.
        session.viewerVM.nextFitAnchor = .firstCameraExact
        session.viewerVM.applyCameraFit(context: session.loadColmapContext())
        // The in-memory live streaming wiring (app round 3 Task 2 — spec §3). If there is no Metal
        // device (in theory), training proceeds without the live view. The handler is called
        // synchronously on the training loop thread, and the streamer only copies and returns
        // immediately (the snapshot pointer lifetime contract).
        if let device = MTLCreateSystemDefaultDevice() {
            // The timelapse recorder is created **before** the streamer so the swap hook can
            // capture it. The capture camera is snapshotted from the camera **after** the fit
            // above — with the first-camera anchor that is exactly the first COLMAP camera
            // (spec §2/§3.3). The recorder object is cheap; its capture-only SplatRenderer is
            // allocated lazily on the first capture (spec §3.1).
            var recorder: TimelapseRecorder?
            if draft.timelapse.enabled {
                let context = session.loadColmapContext()
                let intrinsics = context.flatMap { $0.model.cameras[$0.firstView.cameraId] }
                // 2026-08-05 spec §2.3: 종횡비의 출처만 사용자가 고른다. 포즈는 어느 쪽이든 위
                // fit이 넣은 첫 카메라 그대로이고 수직 화각도 이미 같으므로(§1.1), 이 한 쌍이
                // 바꾸는 것은 좌우로 얼마나 담기느냐뿐이다. `viewportSize`가 아직 0이면
                // aspectPair가 데이터셋으로 떨어뜨린다(폴백 ①) — 여기서 다시 판정하지 않는다.
                let aspect = TimelapseSettings.aspectPair(
                    source: draft.timelapse.aspectSource, viewport: viewportSize,
                    datasetWidth: intrinsics?.width ?? 0, datasetHeight: intrinsics?.height ?? 0)
                let config = TimelapseRecorder.makeConfig(
                    settings: draft.timelapse, runDirectory: runDir,
                    totalIterations: draft.iterations,
                    aspectWidth: aspect.width, aspectHeight: aspect.height)
                do {
                    try FileManager.default.createDirectory(at: config.frameDirectory,
                                                            withIntermediateDirectories: true)
                    recorder = TimelapseRecorder(
                        device: device, config: config,
                        pose: CameraPose.effective(of: session.viewerVM.camera))
                    model.console.append(
                        "◉ Timelapse: every \(config.settings.everySteps) steps · "
                        + "\(config.width)×\(config.height) · \(config.settings.format.displayName)")
                } catch {
                    model.console.append(
                        "⚠ timelapse folder create failed: \(error.localizedDescription)"
                        + " — training continues without a timelapse")
                }
            }
            self.setTimelapseRun(recorder)

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
                },
                onSwappedChunk: recorder.map { rec in
                    { @Sendable (chunk: SplatChunk, iter: Int) async in
                        await rec.captureIfDue(chunk: chunk, iter: iter)
                    }
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
                    let recorderRef = self.timelapseRecorder   // handed to the encode thread below
                    self.setTimelapseRun(nil)
                    self.teardownStreamer()
                    self.isTraining = false
                    model.job = .idle
                    guard let r = result else { return }
                    // The collapse verdict (2026-08-16): the core's `.completed` only means the
                    // loop reached run_until, so "did this produce anything usable" is judged
                    // here — see RunHealth for the criteria and the incident behind them. It
                    // decides four things: the recorded outcome, the console report, the inline
                    // banner, and whether the viewer is switched to.
                    //
                    // Three of those four re-derive the verdict from `r` instead of being handed
                    // this value (`recordOutcome`, `completionConsoleLines`, and the notification
                    // builder). That is deliberate: it is a pure function of exactly the three
                    // fields they each already receive, so deriving makes it impossible for a call
                    // site to pass a stale or defaulted verdict and quietly go back to reporting
                    // success — which is the whole failure being removed here. This local binding
                    // exists for the two decisions that are *not* wording (the banner and the tab
                    // switch).
                    let health = RunHealth.evaluate(outcome: r.outcome, splatCount: r.splatCount,
                                                    finalPSNR: r.finalPSNR)
                    // A partial completion (thermal abort or stop) also has an output, so it is
                    // recorded together with its outcome.
                    let record = RunRecord(
                        id: stamp,
                        outputFile: outURL.lastPathComponent,
                        previewFile: FileManager.default.fileExists(atPath: previewURL.path)
                            ? previewURL.lastPathComponent : nil,
                        outcome: RunHealth.recordOutcome(outcome: r.outcome,
                                                         splatCount: r.splatCount,
                                                         finalPSNR: r.finalPSNR),
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
                    for line in RunHealth.completionConsoleLines(
                        outcome: r.outcome, splatCount: r.splatCount, finalPSNR: r.finalPSNR) {
                        model.console.append(line)
                    }
                    if r.outcome != .completed {
                        self.errorMessage = "Finished: \(r.outcome.rawValue) (partial result saved)"
                    } else if health.isCollapsed {
                        // else-if, not a second if: RunHealth only ever judges `.completed`, so the
                        // two branches are mutually exclusive by construction and the shape of this
                        // code says so. A thermal abort or a user stop is a normal termination that
                        // keeps its own honest wording (README "Exit codes": 2 and 3 are not errors) and is never
                        // relabelled.
                        self.errorMessage = health.message
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
                    //
                    // The timelapse encode (spec §3.5) starts **inside** that same await, and on
                    // every outcome — a cancellation or thermal abort still encodes what was
                    // collected. It does **not** join lockedTab (it only reads files and uses
                    // neither the GPU nor the trainer).
                    // Ordering it after waitUntilIdle is load bearing (FIX 7): stop() is
                    // non-blocking, and the swap loop does not re-check isStopped between the
                    // MainActor update and the capture hook, so one more capture can still be
                    // enqueued after the encoder's waitForPendingWrites() barrier has passed.
                    // The encoder lists the directory; a frame written into that window would be
                    // read half-finished, and decodeFailed takes the **whole** timelapse with it
                    // (the video path deletes its partial file on cancel). Draining the pipeline
                    // first closes the window — the barrier alone is a point in time, not a seal.
                    Task { @MainActor in
                        await streamerRef?.waitUntilIdle()
                        if let recorderRef {
                            self.startTimelapseEncoding(recorder: recorderRef, runID: stamp,
                                                        runDirectory: runDir,
                                                        session: session, model: model)
                        }
                        if r.outcome != .cancelled && !health.isCollapsed {
                            // 2026-08-04 spec §2.1: the result is presented from the same
                            // first-camera viewpoint the run started at (showFinishedRun arms
                            // the one-shot anchor that openRun's applyCameraFit consumes).
                            //
                            // A collapsed run stays on the Train tab (2026-08-16). Two reasons,
                            // and the second is the load-bearing one: there is nothing to present
                            // (an `element vertex 0` PLY renders as an empty viewport that looks
                            // like a loading bug), and `errorMessage` — the sentence explaining
                            // what happened and what to try — is rendered **only** by TrainView
                            // (TrainView.swift:16). Switching tabs here would carry the user away
                            // from the one place the explanation is visible and leave them staring
                            // at an empty scene, which is the same silence in a new costume.
                            session.showFinishedRun(record)
                        }
                    }
                }
            } catch {
                await MainActor.run {
                    self?.teardownStreamer()
                    // A hard failure has no run record to attach a file to — the frames stay on
                    // disk under runs/<stamp>/timelapse/ and are removed with the run folder.
                    self?.setTimelapseRun(nil)
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

    /// The timelapse "shoot from here" action (2026-08-04 spec §3.3): the pose currently on
    /// screen becomes the new capture anchor, and **every frame collected so far is deleted** so
    /// the video restarts from index 0 with no viewpoint jump.
    func resetTimelapseAnchor(session: ProjectSession, model: AppModel) {
        guard let recorder = timelapseRecorder else { return }
        recorder.setPose(CameraPose.effective(of: session.viewerVM.camera))
        recorder.resetFrames()
        model.console.append("↺ Timelapse restarted from the current view")
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

    /// Encodes the collected frames into runs/<stamp>/timelapse.<ext> (spec §3.5).
    /// It runs on a **dedicated Thread**: VideoExporter's contract is that append/finish/cancel
    /// are called from exactly one thread and finish() blocks — the same discipline as RenderJob.
    /// Fewer than two frames is reported and skipped by the encoder itself.
    private func startTimelapseEncoding(recorder: TimelapseRecorder, runID: String,
                                        runDirectory: URL, session: ProjectSession,
                                        model: AppModel) {
        let settings = recorder.config.settings
        let width = recorder.config.width
        let height = recorder.config.height
        let frameDirectory = recorder.config.frameDirectory
        let output = TimelapseEncoder.outputURL(runDirectory: runDirectory,
                                                format: settings.format)
        let cancel = timelapseCancel
        cancel.reset()
        // Raised here rather than at the Thread.start() below so there is no window in which the
        // run is on screen with neither a file nor an "encoding…" state (this function is called
        // from the MainActor Task that also calls showFinishedRun).
        isEncodingTimelapse = true
        model.console.append("◍ Encoding timelapse → \(output.lastPathComponent)…")
        // FIX 5 / Important 2: the failure counters travel alongside the encode Result so the
        // completion summary can report them (Task 10's `failedFrameCount` doc comment: "stays
        // available for the completion summary" — this is that summary). Read regardless of
        // success/failure so a `notEnoughFrames`/`cancelled` outcome still explains *why* the
        // frame count looked short.
        let completion: @Sendable (Result<TimelapseEncoder.Result, any Error>, Int, Int, Bool) -> Void =
            { [weak self, weak model, weak session] result, failedWrites, failedDeletes, setupFailed in
            Task { @MainActor in
                self?.finishTimelapseEncoding(result, runID: runID, format: settings.format,
                                              failedWrites: failedWrites,
                                              failedDeletes: failedDeletes,
                                              setupFailed: setupFailed,
                                              model: model, session: session)
            }
        }
        // A `var` (not `let`) so it can be released below, before the encode — see the comment at
        // the nil-out site.
        var pendingRecorder: TimelapseRecorder? = recorder
        let thread = Thread {
            // The last frames may still be on the write queue — drain it before listing.
            // Known limitation: this only drains TimelapseRecorder's own serial writeQueue (the
            // label-composite-and-JPEG step) — it does **not** guarantee every training iteration
            // up to the last one was even *attempted* as a capture. LiveSplatStreamer's mailbox
            // drops an unconsumed pending snapshot when a newer one arrives (by design — "one
            // latest pending", see TimelapseRecorder.captureIfDue's claim() comment), so the very
            // last due iteration can be skipped upstream of this barrier entirely, not just
            // in-flight past it. Accepted per spec — a one-frame-short timelapse, not a bug.
            pendingRecorder?.waitForPendingWrites()
            // Read the failure counters and then drop the recorder — releasing its capture-only
            // SplatRenderer and offscreen texture (~240MB at typical splat counts) before the
            // encode below, which can run for minutes on a long MP4/MOV. Nothing past this point
            // needs the recorder; holding it alive for the encode's duration would just compete
            // with it for memory.
            let failedWrites = pendingRecorder?.failedFrameCount ?? 0
            let failedDeletes = pendingRecorder?.failedFrameDeleteCount ?? 0
            let setupFailed = pendingRecorder?.captureSetupFailed ?? false
            pendingRecorder = nil
            let frames = TimelapseEncoder.frameURLs(in: frameDirectory)
            do {
                completion(.success(try TimelapseEncoder.encode(
                    frames: frames, to: output, format: settings.format, fps: settings.fps,
                    width: width, height: height,
                    isCancelled: { cancel.isCancelled })), failedWrites, failedDeletes, setupFailed)
            } catch {
                completion(.failure(error), failedWrites, failedDeletes, setupFailed)
            }
        }
        thread.name = "splatx.timelapse-encode"
        thread.qualityOfService = .utility
        thread.start()
    }

    /// The MainActor side of the encode: the console line, and recording the produced file on the
    /// run (a second manifest save — the record was appended and saved long before the encode
    /// finished).
    /// `failedWrites`/`failedDeletes` are `TimelapseRecorder.failedFrameCount`/
    /// `failedFrameDeleteCount`, read at the start of the encode thread (FIX 5 / Important 2) —
    /// spec §3.4 forbids a silent shortfall, and Task 10 built those counters expressly for this
    /// summary.
    private func finishTimelapseEncoding(_ result: Result<TimelapseEncoder.Result, any Error>,
                                         runID: String, format: TimelapseFormat,
                                         failedWrites: Int, failedDeletes: Int,
                                         setupFailed: Bool,
                                         model: AppModel?, session: ProjectSession?) {
        // Lowered first, before any early `return` below — every outcome (success, cancelled,
        // notEnoughFrames, a hard failure) ends the encode, and a flag left raised would pin the
        // inspector at "encoding…" forever with no path back.
        isEncodingTimelapse = false
        let failureNote = Self.timelapseFailureNote(failedWrites: failedWrites,
                                                    failedDeletes: failedDeletes,
                                                    setupFailed: setupFailed)
        switch result {
        case .success(let produced):
            model?.console.append(
                "■ Timelapse: \(produced.frameCount) frames → "
                + "runs/\(runID)/\(produced.outputURL.lastPathComponent)" + failureNote)
            if let note = produced.note { model?.console.append("  \(note)") }
            guard let session else { return }
            session.project = session.project.settingTimelapseFile(
                runID: runID,
                path: TimelapseEncoder.relativePath(runID: runID, format: format))
            do { try model?.store.save(session.project) }
            catch {
                // FIX 5 / Important 1: a silent console-only failure here is exactly the manifest
                // desync this codebase's convention exists to prevent (AlignViewModel's save
                // failures, ProjectSession.setAutoLevelOnAlign, and the RunRecord save a few lines
                // above in this same completion handler all surface through errorMessage too,
                // console being collapsed by default). Concretely: the encode succeeded and the
                // file is on disk, but the manifest never learns its path — after an app restart
                // the video exists with no record pointing at it.
                errorMessage = "Timelapse encoded, but saving the project manifest failed"
                    + " — the video will not appear in the project after the app restarts."
                    + " (\(error.localizedDescription))"
                model?.console.append("⚠ manifest save failed: \(error.localizedDescription)")
            }
        case .failure(let error):
            switch error {
            case TimelapseEncoder.EncodeError.notEnoughFrames(let n):
                model?.console.append("· Timelapse skipped — only \(n) frame(s) captured" + failureNote)
            case TimelapseEncoder.EncodeError.cancelled:
                model?.console.append("· Timelapse encoding cancelled — partial file removed")
            default:
                model?.console.append("✗ Timelapse failed: \(error.localizedDescription)" + failureNote)
            }
        }
    }

    static func summarize(_ d: TrainOptionsDraft) -> String {
        "factor \(d.factor) · \(d.iterations)it · \(d.mcmc ? "MCMC" : "no-MCMC") · cap \(d.capMax/1000)k · SH\(d.shDegree)"
    }

    /// The failure clause appended to the timelapse completion summary (FIX 5 / Important 2 —
    /// spec §3.4 forbids a silent shortfall). Pure — TimelapseManifestTests exercises it directly
    /// without a GPU, unlike `startTimelapseEncoding`/`finishTimelapseEncoding` themselves, which
    /// need a real `TimelapseRecorder`/`AppModel`/`ProjectSession` and so are covered by build
    /// success + manual check rather than a unit test (the same split as `TrainStartViewpointTests`
    /// documents for `start`).
    /// Empty when both counts are zero, so the common all-succeeded case stays a single short line
    /// instead of "· 0 failed · 0 stale" noise. The two counters are kept as separate clauses
    /// rather than summed: a write failure just means one frame is missing, but a delete failure
    /// (TimelapseRecorder.failedFrameDeleteCount, FIX 3) leaves a *stale-generation* frame on disk
    /// that the encoder may splice into the video next to a "shoot from here" restart — a
    /// different, more alarming failure mode that deserves its own words.
    /// `setupFailed` (TimelapseRecorder.captureSetupFailed, FIX 7) is a different order of
    /// problem: the other two count a shortfall, this one means the capture renderer never came
    /// up so **every** frame is missing. It leads, because with it true the frame count is 0 and
    /// the other two are 0 as well — the summary would otherwise say "0 frame(s) captured" and
    /// leave the reason to the system log.
    static func timelapseFailureNote(failedWrites: Int, failedDeletes: Int,
                                     setupFailed: Bool = false) -> String {
        (setupFailed ? " · the capture renderer could not be created — no frames were captured" : "")
        + (failedWrites > 0 ? " · \(failedWrites) frame(s) failed to write" : "")
        + (failedDeletes > 0 ? " · \(failedDeletes) stale frame(s) from a prior"
            + " \"shoot from here\" could not be deleted (may appear in the video)" : "")
    }
}
