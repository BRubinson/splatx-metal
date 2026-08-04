// Cinema Task 5 (2026-07-26 spec, "render and export pipeline"): the final render job — it drives
// sx_render_session (the authoritative training renderer, Task 2) sequentially on a **dedicated
// Thread** and feeds VideoExporter (AVAssetWriter).
//
// Thread and exclusion contract (the C header sx_render_session_* comments are the authoritative
// contract):
// - The session is single-thread-owned and frame is a blocking GPU round trip — a Swift
//   concurrency executor is forbidden; the dedicated Thread discipline applies (replicating the
//   sx_train_run_args precedent in SplatXTrainer.run).
// - **Exclusive with training**: the start gate is the authoritative lock session.lockedTab (it
//   starts only when nil), and while running the lockedTab == .cinema lock (the isExporting term
//   of ProjectSession.lockedTab) blocks reaching the Train tab, goHome, and ⌥1/2/3/4 navigation
//   at all — exactly the convention "even if a per-button gate misses the other task, a locked tab
//   makes it unreachable".
// - metallib is **injected explicitly** from BundlePaths (an immediate error on failure — blocking
//   the silent substitution of a stale kernel, the Task 2 contract). AA passes
//   RunRecord.antialiased (the setting recorded at training time).
// - Cancellation is flag polling (at frame boundaries) → exporter.cancel() deletes the partial
//   file.
//   A cancellation is an action the user took themselves, so it produces no system notification
//   (the same convention as cancelled → nil in trainingFinished).
import Foundation
import AppKit
import SwiftUI
import Observation
import simd
import CryptoKit
import SplatXTrainC

// MARK: - Camera math (pure — VideoExportTests is the source of truth)

enum ExportRenderMath {
    /// CameraPose (Metal camera→world: +X right, +Y up, −Z forward) →
    /// sx_render_session_frame viewmat[16]: COLMAP (+Y down, +Z forward) world→cam,
    /// **row-major**. Derivation: left-multiply Metal world→cam = [R_cwᵀ | −R_cwᵀ·p] by
    /// flip = diag(1,−1,−1) — the exact inverse direction of CameraConversion.viewMatrix
    /// (COLMAP→Metal, view = flip·[R|t]).
    /// testColmapViewmatRoundtripWithCameraConversion pins the round trip.
    static func colmapViewmatRowMajor(position: SIMD3<Float>,
                                      rotation: simd_quatf) -> [Float] {
        let rwc = simd_float3x3(rotation.normalized).transpose   // the Metal world→cam rotation
        let flip = simd_float3x3(diagonal: SIMD3<Float>(1, -1, -1))
        let r = flip * rwc
        let t = -(r * position)
        return [r.columns.0.x, r.columns.1.x, r.columns.2.x, t.x,
                r.columns.0.y, r.columns.1.y, r.columns.2.y, t.y,
                r.columns.0.z, r.columns.1.z, r.columns.2.z, t.z,
                0, 0, 0, 1]
    }

    /// Vertical fov (rad) + output dimensions → PINHOLE intrinsics.
    /// fy = H/(2·tan(fov/2)), fx = fy (square pixels — the horizontal field of view is determined
    /// by W), and cx/cy = the center.
    static func intrinsics(verticalFOV: Float, width: Int, height: Int)
        -> (fx: Float, fy: Float, cx: Float, cy: Float) {
        let f = Float(height) / 2 / tan(max(verticalFOV, 1e-3) / 2)
        return (f, f, Float(width) / 2, Float(height) / 2)
    }
}

// MARK: - Thread boundary types

/// The cancellation flag — shared by the MainActor (the cancel button) and the render thread
/// (frame-boundary polling).
final class RenderCancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false

    var isCancelled: Bool { lock.withLock { flag } }
    func cancel() { lock.withLock { flag = true } }
    func reset() { lock.withLock { flag = false } }
}

/// The render thread's input snapshot — it pins the MainActor state as values at start time.
/// (CGImage is an immutable object — passing it across threads via OverlayCompositor is safe.)
struct RenderExportRequest {
    let plyPath: String
    let metallibPath: String
    let antialiased: Bool
    let path: CameraPath
    let width: Int
    let height: Int
    let fps: Int
    let codec: VideoCodec
    let container: VideoContainer
    let totalFrames: Int
    let outputURL: URL
    let compositor: OverlayCompositor?
}

enum RenderExportOutcome {
    case completed(frames: Int, seconds: Double)
    case cancelled
    case failed(String)
}

/// The render failure message wrapper — AppNotificationText.failureSummary extracts
/// errorDescription from it.
struct RenderExportError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

// MARK: - RenderJob

@MainActor @Observable final class RenderJob {
    /// Progress state (the inspector progress bar and the StatusBar). frame is the number of
    /// completed frames.
    private(set) var isExporting = false
    private(set) var frame = 0
    private(set) var total = 0

    private let cancelFlag = RenderCancelFlag()

    var progress: Double {
        total > 0 ? Double(frame) / Double(total) : 0
    }

    /// Starts the export. Gates: not already running + lockedTab == nil (exclusive with training
    /// and align) + an open run + ≥ 1 keyframe. Violations are reported through the viewer
    /// overlay (the same failure surface as ExportService).
    func start(model: AppModel, session: ProjectSession, outputURL: URL) {
        guard !isExporting else { return }
        let vm = session.cinemaVM
        guard session.lockedTab == nil else {
            session.viewerVM.errorMessage = "Rendering is unavailable while training or aligning."
            return
        }
        guard let record = session.viewerVM.currentRun else {
            session.viewerVM.errorMessage = "Open a run before rendering."
            return
        }
        guard let path = vm.selectedPath, !path.keyframes.isEmpty else {
            session.viewerVM.errorMessage = "Add at least one keyframe before rendering."
            return
        }

        let size = VideoExportSettings.pixelSize(preset: vm.exportResolution,
                                                 aspect: vm.outputAspect)
        let fps = vm.fps
        let totalFrames = FrameSchedule.frameCount(duration: path.duration, fps: fps,
                                                   loop: path.loop)
        // The overlay image is loaded once on the MainActor and passed as a snapshot (no file
        // access or state reads on the thread). A load failure is not fatal — it proceeds without
        // the overlay and warns on the console.
        var compositor: OverlayCompositor?
        if let spec = vm.overlay {
            if let url = vm.overlayImageURL,
               let image = NSImage(contentsOf: url)?
                   .cgImage(forProposedRect: nil, context: nil, hints: nil) {
                compositor = Self.overlayCompositor(spec: spec, image: image,
                                                    outputWidth: size.width,
                                                    outputHeight: size.height)
            } else {
                model.console.append("⚠ overlay image load failed — exporting without overlay")
            }
        }

        let request = RenderExportRequest(
            plyPath: session.directory
                .appendingPathComponent("runs/\(record.id)/\(record.outputFile)").path,
            metallibPath: BundlePaths.metallib,   // explicit injection — the no-fallback contract
            antialiased: record.antialiased,
            path: path,
            width: size.width, height: size.height, fps: fps,
            codec: vm.exportCodec, container: vm.exportContainer, totalFrames: totalFrames,
            outputURL: outputURL, compositor: compositor)

        isExporting = true
        frame = 0
        total = totalFrames
        cancelFlag.reset()
        model.job = .exporting(frame: 0, total: totalFrames)
        model.console.append(
            "▶ Render video: \(path.name) · \(size.width)×\(size.height) @ \(fps)fps · "
            + "\(totalFrames) frames · \(vm.exportCodec.rawValue) \(vm.exportContainer.rawValue)"
            + " → \(outputURL.lastPathComponent)")

        let projectName = session.project.name
        let progress: @Sendable (Int, Int) -> Void = { [weak self, weak model] done, count in
            Task { @MainActor in
                self?.applyProgress(frame: done, total: count, model: model)
            }
        }
        let completion: @Sendable (RenderExportOutcome) -> Void =
            { [weak self, weak model, weak session] outcome in
            Task { @MainActor in
                self?.finishExport(outcome, model: model, session: session,
                                   project: projectName, outputURL: outputURL)
            }
        }
        let cancelFlag = self.cancelFlag
        // A dedicated thread — frame is a blocking GPU round trip (0.1–0.24s at 4K × hundreds of
        // frames), so it MUST NOT occupy a Swift concurrency executor (the same discipline as
        // sx_train_run_args).
        let thread = Thread {
            RenderJob.run(request, cancel: cancelFlag, progress: progress,
                          completion: completion)
        }
        thread.name = "splatx.render-export"
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    /// The cancellation request — the render thread detects it at the next frame boundary and
    /// cleans up the partial file.
    func cancel() {
        guard isExporting else { return }
        cancelFlag.cancel()
    }

    /// The body of the progress callback — the isExporting guard is the authoritative contract
    /// (FIX 2, Task 5 review M2): it closes the reordering window in which a progress Task that
    /// runs late — after completion, once finishExport has returned job to .idle — puts the state
    /// back to .exporting (the MainActor arrival order of Tasks is not guaranteed to match the
    /// submission order). Contract test: testStaleProgressAfterCompletionIgnored.
    func applyProgress(frame done: Int, total count: Int, model: AppModel?) {
        guard isExporting else { return }
        frame = done
        model?.job = .exporting(frame: done, total: count)
    }

    private func finishExport(_ outcome: RenderExportOutcome, model: AppModel?,
                              session: ProjectSession?, project: String, outputURL: URL) {
        isExporting = false
        model?.job = .idle
        switch outcome {
        case .completed(let frames, let seconds):
            let ms = seconds / Double(max(frames, 1)) * 1000
            model?.console.append(String(
                format: "■ Render complete: %d frames · %.1fs (%.0f ms/frame) → %@",
                frames, seconds, ms, outputURL.lastPathComponent))
            AppNotifier.shared.post(AppNotificationText.renderFinished(
                project: project, fileName: outputURL.lastPathComponent,
                frames: frames, seconds: seconds))
        case .cancelled:
            model?.console.append("■ Render cancelled — partial file removed")
        case .failed(let message):
            model?.console.append("✗ Render failed: \(message)")
            session?.viewerVM.errorMessage = "Render failed: \(message)"
            AppNotifier.shared.post(AppNotificationText.renderFailed(
                project: project, error: RenderExportError(message: message)))
        }
    }

    /// The overlay compositor builder — rect is **OverlayPlacement.rect verbatim** (the Task 4
    /// contract: the same function as the viewport WYSIWYG, called with the output pixel
    /// dimensions — since the units are percentages, the two results are similar figures).
    nonisolated static func overlayCompositor(spec: OverlaySpec, image: CGImage,
                                              outputWidth: Int, outputHeight: Int)
        -> OverlayCompositor {
        let rect = OverlayPlacement.rect(
            spec: spec,
            imageSize: CGSize(width: image.width, height: image.height),
            outputWidth: CGFloat(outputWidth), outputHeight: CGFloat(outputHeight))
        return OverlayCompositor(image: image, rect: rect, opacity: spec.opacity)
    }

    /// The SPZ→PLY conversion scratch path (pure — VideoExportTests pins the contract). FIX 2
    /// (Task 5 review M1): the previous `sx-export-\(UUID()).ply` was an undeleted leak that piled
    /// up a new file per export (980k splats ≈ hundreds of MB each time). Exactly like the viewer
    /// fallback convention (sx-view-file-<digest> in ViewerModel.openFile), the name is pinned by
    /// a stable hash of the source's full path (the first 8 bytes of SHA256) — re-exporting the
    /// same source overwrites the same file (the PLY writer truncates with fopen "wb" — confirmed
    /// through sx_convert_run) and disk usage is bounded at one file per source. Invalidation
    /// review (requested in review): a different run or project has a different source path, so
    /// the digest differs, and the contents are re-converted on every export (there is no
    /// conversion skip — only the filename is cached, so there is no path by which stale contents
    /// are reused).
    nonisolated static func spzScratchURL(forSource path: String) -> URL {
        let digest = SHA256.hash(data: Data(path.utf8))
            .prefix(8).map { String(format: "%02x", $0) }.joined()
        return FileManager.default.temporaryDirectory
            .appendingPathComponent("sx-export-\(digest).ply")
    }

    // MARK: The render thread body (nonisolated — it never touches MainActor state; its input is
    // the Request snapshot)

    private nonisolated static func run(_ req: RenderExportRequest, cancel: RenderCancelFlag,
                                        progress: @escaping @Sendable (Int, Int) -> Void,
                                        completion: @escaping @Sendable (RenderExportOutcome) -> Void) {
        var err = [CChar](repeating: 0, count: 4096)

        // An SPZ output from an old manifest → a temporary PLY conversion (the same path as the
        // viewer preview fallback).
        // The scratch name is a stable hash of the source path, not a UUID — the spzScratchURL
        // comment is the authoritative contract.
        var plyPath = req.plyPath
        if (plyPath as NSString).pathExtension.lowercased() != "ply" {
            let tmp = spzScratchURL(forSource: plyPath)
            do {
                try ExportService.export(from: URL(fileURLWithPath: plyPath), to: tmp)
            } catch {
                completion(.failed("SPZ → PLY conversion failed: \(error.localizedDescription)"))
                return
            }
            plyPath = tmp.path
        }

        // The session is opened once — amortizing the PLY load and backend creation (the Task 2
        // contract). The string lifetime is needed only for the duration of the open call (it
        // copies internally) — nested withCString is sufficient.
        let session: OpaquePointer? = plyPath.withCString { ply in
            req.metallibPath.withCString { mtl in
                var args = SxRenderSessionArgs()   // zero-init, then only the needed fields (the convention)
                args.ply_path = ply
                args.metallib_path = mtl
                args.antialiased = req.antialiased ? 1 : 0
                args.sh_degree = 0   // derived from the PLY — the final render uses all bands
                return sx_render_session_open(&args, &err, Int32(err.count))
            }
        }
        guard let session else {
            completion(.failed("Render session open failed: \(String(cString: err))"))
            return
        }
        defer { sx_render_session_close(session) }

        let exporter: VideoExporter
        do {
            exporter = try VideoExporter(
                outputURL: req.outputURL,
                config: .init(codec: req.codec, container: req.container,
                              width: req.width, height: req.height, fps: req.fps))
        } catch {
            completion(.failed(error.localizedDescription))
            return
        }

        let sampler = PathSampler(path: req.path)
        var rgba = [UInt8](repeating: 0, count: req.width * req.height * 4)
        let t0 = CFAbsoluteTimeGetCurrent()

        for i in 0..<req.totalFrames {
            if cancel.isCancelled { break }
            let pose = sampler.pose(at: FrameSchedule.time(ofFrame: i, fps: req.fps))
            let viewmat = ExportRenderMath.colmapViewmatRowMajor(position: pose.pos,
                                                                 rotation: pose.quat)
            let k = ExportRenderMath.intrinsics(verticalFOV: pose.fov,
                                                width: req.width, height: req.height)
            let rc = viewmat.withUnsafeBufferPointer { vp in
                rgba.withUnsafeMutableBufferPointer { rp in
                    sx_render_session_frame(session, vp.baseAddress, k.fx, k.fy, k.cx, k.cy,
                                            Int32(req.width), Int32(req.height),
                                            rp.baseAddress, &err, Int32(err.count))
                }
            }
            guard rc == 0 else {
                exporter.cancel()
                completion(.failed("frame \(i) render rc=\(rc): \(String(cString: err))"))
                return
            }
            do {
                try rgba.withUnsafeBufferPointer { rp in
                    try exporter.append(rgba: rp.baseAddress!, frameIndex: i,
                                        compositor: req.compositor)
                }
            } catch {
                exporter.cancel()
                completion(.failed("frame \(i) encode: \(error.localizedDescription)"))
                return
            }
            progress(i + 1, req.totalFrames)
        }

        if cancel.isCancelled {
            exporter.cancel()   // deletes the partial file
            completion(.cancelled)
            return
        }
        do {
            try exporter.finish()
        } catch {
            // FIX 2 (Task 5 review I1): a finalize failure also cleans up the partial file —
            // without the cleanup, a "Render failed" notification and a broken .mov left behind
            // contradict each other. After finishWriting fails, writer.status != .writing, so
            // cancel() performs only removeItem without cancelWriting (the safety and idempotence
            // of cancel in a terminal state is pinned by
            // testExporterCancelIdempotentAndRemovesPartialFile).
            exporter.cancel()
            completion(.failed("finalize: \(error.localizedDescription)"))
            return
        }
        completion(.completed(frames: req.totalFrames,
                              seconds: CFAbsoluteTimeGetCurrent() - t0))
    }
}

// MARK: - The inspector's EXPORT section (inserted by CinemaInspector. Panel redesign Task 3 —
// spec §5: it absorbed the Aspect of the FRAMING section. FIX 2 ② — spec footnote 2: the
// > Advanced disclosure of Task 3 (codec, container, resolution) was withdrawn on the user's
// real-use feedback "just lay them all out in the open" — all five pickers are shown permanently
// at the same level, and the disclosure and its collapsed summary line were removed.

struct ExportInspectorSection: View {
    @Bindable var vm: CinemaViewModel
    @Bindable var model: AppModel
    @Bindable var session: ProjectSession

    var body: some View {
        PanelSection(title: "EXPORT") {
            if session.renderJob.isExporting {
                exportProgress
            } else {
                settings
            }
        }
    }

    // MARK: Settings + start

    @ViewBuilder private var settings: some View {
        // Aspect — absorbed from the former FRAMING section (Task 3, spec §5). The binding still
        // goes through setOutputAspect (the state shared by the viewport mask and the export, plus
        // project persistence — only the placement moved).
        // While rendering, the entire settings block is replaced by progress, so it is temporarily
        // inaccessible — consistent with freezing settings during a run (an aspect change in
        // progress would not be reflected in the running export, which is a misleading surface).
        SegmentedRow("Aspect",
                     options: OutputAspect.allCases.map { SegmentOption($0, $0.rawValue) },
                     selection: Binding(get: { vm.outputAspect },
                                        set: { vm.setOutputAspect($0) }))
            .help("Viewport shows this aspect as a letterbox mask.")
        SegmentedRow("Rate",
                     options: VideoExportSettings.fpsChoices.map { SegmentOption($0, "\($0)") },
                     selection: Binding(get: { vm.fps }, set: { vm.setFPS($0) }))

        // Format, File, and Size — shown permanently at the same level as Aspect and Rate
        // (FIX 2 ② — the > Advanced disclosure of the former Task 3 was withdrawn on user
        // feedback, spec footnote 2. The collapsed summary line was removed along with it — since
        // every choice is always visible, showing it alongside is redundant).
        SegmentedRow("Format",
                     options: VideoCodec.allCases.map { SegmentOption($0, $0.rawValue) },
                     selection: $vm.exportCodec)
        // The container axis (Cinema spec footnote 2 ⑤) — MP4 by default, or MOV. The extension
        // and the save panel type follow this choice.
        SegmentedRow("File",
                     options: VideoContainer.allCases.map { SegmentOption($0, $0.rawValue) },
                     selection: $vm.exportContainer)
        SegmentedRow("Size",
                     options: ResolutionPreset.allCases.map { SegmentOption($0, $0.rawValue) },
                     selection: $vm.exportResolution)

        // The CTA (footnote 2 ④): the main action style, on par with Start Training — reflecting
        // the E2E feedback that the previous small "Export Movie…" lost out on discoverability
        // and hierarchy.
        // The disabled reason is attached to the CTA with spacing 2 (panel redesign Task 2 —
        // spec §4 rules 5 and 6. The button dimming is already handled by
        // PrimaryActionButtonStyle's disabled presentation).
        VStack(alignment: .leading, spacing: 2) {
            Button {
                presentSavePanel()
            } label: {
                Text("Render Video…")
            }
            .buttonStyle(.primaryAction)
            .disabled(disabledReason != nil)
            .help("Render the selected path with the training renderer and encode to "
                  + "\(vm.exportContainer.rawValue)")

            if let reason = disabledReason {
                PanelCaption(reason)
            }
        }
    }

    /// The single source of truth for the Render CTA's disabled reason (pure — panel redesign
    /// Task 2, tests: CinemaInspectorAffordanceTests). Priority: no run → no keyframe → tab
    /// locked.
    static func renderDisabledReason(hasRun: Bool, hasKeyframes: Bool,
                                     isLocked: Bool) -> String? {
        if !hasRun { return "Open a run first — rendering uses its PLY." }
        if !hasKeyframes { return "Add at least one keyframe to the selected path." }
        if isLocked { return "Unavailable while training or aligning." }
        return nil
    }

    private var disabledReason: String? {
        Self.renderDisabledReason(
            hasRun: session.viewerVM.currentRun != nil,
            hasKeyframes: !(vm.selectedPath?.keyframes.isEmpty ?? true),
            isLocked: session.lockedTab != nil)
    }

    /// NSSavePanel — the extension and content type follow the container choice (footnote 2 ⑤).
    private func presentSavePanel() {
        let container = vm.exportContainer
        let panel = NSSavePanel()
        panel.allowedContentTypes = [container.contentType]
        panel.canSelectHiddenExtension = true
        panel.nameFieldStringValue = "\(session.project.name).\(container.fileExtension)"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        session.renderJob.start(model: model, session: session, outputURL: url)
    }

    // MARK: Progress + cancel (footnote 2 ④ — "Rendering… f/N" plus a destructive Cancel in the
    // same place, the same geometry as the Start↔Stop CTA switching convention)

    @ViewBuilder private var exportProgress: some View {
        let job = session.renderJob
        ProgressView(value: Double(job.frame), total: Double(max(job.total, 1)))
            .tint(SplatXTheme.accent)
        HStack {
            Text("Rendering… \(job.frame)/\(job.total)")
                .font(.caption).monospacedDigit()
                .foregroundStyle(SplatXTheme.textPrimary)
            Spacer()
            Text("\(Int((job.progress * 100).rounded()))%")
                .font(.caption).monospacedDigit()
                .foregroundStyle(SplatXTheme.textSecondary)
        }
        Button {
            job.cancel()
        } label: {
            Text("Cancel Render")
        }
        .buttonStyle(.destructiveAction)
        .help("Stop rendering — the partial file is deleted")
    }

    // (The former private labeledPicker was replaced by the shared SegmentedRow in panel redesign
    // FIX 3 ① — Theme.swift. macOS's `.pickerStyle(.segmented)` refusing the proposed width, so
    // that `.frame(maxWidth: .infinity)` could not stretch the control and only centered it, was
    // the cause of FIX 2 ③'s failure. The rationale and measurements are in spec footnote 3 ① and
    // the SegmentedControl comment.)
}
