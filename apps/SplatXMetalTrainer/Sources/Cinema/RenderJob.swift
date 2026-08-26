// Cinema Task 5 (2026-07-26 spec, "render and export pipeline"): the final render job — it drives
// sx_render_session (the authoritative training renderer, Task 2) sequentially on a **dedicated
// Thread** and feeds a FrameSink.
//
// Format branch (2026-08-05 spec §5.5): the sink is chosen **once, at creation** —
// `format.videoContainer` non-nil → VideoExporter (AVAssetWriter → MP4/MOV), nil → GifExporter
// (ImageIO → GIF). That is the only format-dependent line in this file: the frame loop below is
// written against FrameSink and is identical for both, which is the whole reason the protocol was
// extracted rather than duplicating the loop.
//
// Thread and exclusion contract (the C header sx_render_session_* comments are the authoritative
// contract):
// - The session is single-thread-owned and frame is a blocking GPU round trip — a Swift
//   concurrency executor is forbidden; the dedicated Thread discipline applies (replicating the
//   sx_train_run_args precedent in SplatXTrainer.run). **This applies to GIF unchanged**: the
//   reason is sx_render_session_frame, not the encoder, so "ImageIO does not need a thread" is
//   not an argument for moving the GIF path onto an executor.
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
    /// Ignored under GIF (no codec axis) — kept unconditional so the snapshot stays a plain value
    /// copy of the VM; the sink branch is what decides whether it is read.
    let codec: VideoCodec
    let format: CinemaExportFormat
    /// Also the GIF frame count — CGImageDestinationCreateWithURL demands it up front
    /// (GifExporter.Config.frameCount, spec §5.6 trap ④).
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
            codec: vm.exportCodec, format: vm.exportFormat, totalFrames: totalFrames,
            outputURL: outputURL, compositor: compositor)

        isExporting = true
        frame = 0
        total = totalFrames
        cancelFlag.reset()
        model.job = .exporting(frame: 0, total: totalFrames)
        model.console.append(
            "▶ Render video: \(path.name) · \(size.width)×\(size.height) @ \(fps)fps · "
            + "\(totalFrames) frames · "
            + Self.consoleFormatSummary(codec: vm.exportCodec, format: vm.exportFormat)
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

    /// The start line's format summary (pure — VideoExportTests pins it). The codec is dropped for
    /// GIF: it has no codec axis, so "H.264 GIF" would name a setting that had no effect on the
    /// file — the same reason the EXPORT section dims the Format row under GIF (spec §5.9).
    /// The branch is on `videoContainer == nil`, the one GIF discriminator this codebase uses
    /// (TimelapseEncoder.encode branches on the same shape), so a future format cannot land in the
    /// wrong arm by inheriting a `== .gif` comparison.
    nonisolated static func consoleFormatSummary(codec: VideoCodec,
                                                 format: CinemaExportFormat) -> String {
        guard format.videoContainer != nil else { return format.rawValue }
        return "\(codec.rawValue) \(format.rawValue)"
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

        // The only format-dependent line in the render body (spec §5.5). `exporter` keeps its name
        // and only its **static type** widens to FrameSink, so the loop below is untouched and
        // provably format-agnostic. Branching here rather than inside VideoExporter is deliberate:
        // the pool, BT.709 tagging, FrameQueue, and the pull loop are all AVAssetWriter-specific,
        // so an internal `if gif` would construct the whole machine and leave most of it dead (the
        // FrameSink comment in VideoExporter.swift is the authoritative rationale).
        let exporter: FrameSink
        do {
            if let container = req.format.videoContainer {
                exporter = try VideoExporter(
                    outputURL: req.outputURL,
                    config: .init(codec: req.codec, container: container,
                                  width: req.width, height: req.height, fps: req.fps))
            } else {
                exporter = try GifExporter(
                    outputURL: req.outputURL,
                    config: .init(width: req.width, height: req.height, fps: req.fps,
                                  frameCount: req.totalFrames))
            }
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
            // without the cleanup, a "Render failed" notification and a broken .mov/.gif left
            // behind contradict each other. "cancel() is safe twice and in a terminal state" is a
            // FrameSink contract, so this holds for either sink: VideoExporter's writer is no
            // longer .writing after a failed finishWriting and only removeItem runs
            // (testExporterCancelIdempotentAndRemovesPartialFile), and GifExporter has already
            // dropped its destination and removed the file in the throwing branch of finish().
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
//
// 2026-08-05 spec §5: the File row widened from MP4/MOV to MP4/MOV/GIF, and GIF makes three of
// the five rows format-dependent — Format dims (no codec axis, §5.9), Size offers a different
// and **four-cell** list (§5.4), and Rate grows a caption at 60fps (§5.7). All three decisions
// are static funcs below rather than inline conditions, because a view body is untestable in
// this repository and these are the strings and states a user actually reads.
// Everything format-dependent reads `vm.exportFormat`, never a view @State: while
// `isExporting` the entire settings block is swapped out for progress (§5.10), so view-local
// state would not survive a render.

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
        // Rate — the fps caption warns instead of restricting the choices (spec §5.7). Unlike
        // Size, 60fps GIF costs no extra memory; it just plays back wrong in some viewers, and
        // "no silent truncation" means telling the user rather than removing the option.
        VStack(alignment: .leading, spacing: 2) {
            SegmentedRow("Rate",
                         options: VideoExportSettings.fpsChoices.map { SegmentOption($0, "\($0)") },
                         selection: Binding(get: { vm.fps }, set: { vm.setFPS($0) }))
            if let note = Self.gifFpsNote(format: vm.exportFormat, fps: vm.fps) {
                PanelCaption(note)
            }
        }

        // Format, File, and Size — shown permanently at the same level as Aspect and Rate
        // (FIX 2 ② — the > Advanced disclosure of the former Task 3 was withdrawn on user
        // feedback, spec footnote 2. The collapsed summary line was removed along with it — since
        // every choice is always visible, showing it alongside is redundant).
        //
        // The codec row is **dimmed, not hidden**, under GIF (2026-08-05 spec §5.9): GIF has no
        // codec axis, but hiding the row would make the EXPORT section's height jump every time
        // the File choice changes, and the 2026-07-29 spec §4 rule 5 already settled that a
        // disabled control is expressed by dimming the whole row. SegmentedControl deliberately
        // does not dim itself (Theme.swift) — otherwise the label and the control would end up at
        // two different brightnesses — so the .disabled + .opacity pair belongs here, on the row.
        let codecDisabled = Self.codecRowDisabled(format: vm.exportFormat)
        SegmentedRow("Format",
                     options: VideoCodec.allCases.map { SegmentOption($0, $0.rawValue) },
                     selection: $vm.exportCodec)
            .disabled(codecDisabled)
            .opacity(codecDisabled ? 0.5 : 1)   // the app-wide dim factor (InspectorView:251 et al.)
            .help(codecDisabled ? "GIF has no codec — the palette is fixed by the format."
                                : "Video codec for the encoded file.")
        // The format axis (Cinema spec footnote 2 ⑤, widened to GIF by 2026-08-05 spec §5.1) —
        // MP4 by default, MOV, or GIF. The extension and the save panel type follow this choice.
        // The binding goes through setExportFormat, not $vm.exportFormat: the setter carries the
        // resolution clamp, and a direct binding would silently skip it (exportFormat is
        // private(set) precisely so that this cannot be written by accident).
        SegmentedRow("File",
                     options: CinemaExportFormat.allCases.map { SegmentOption($0, $0.rawValue) },
                     selection: Binding(get: { vm.exportFormat },
                                        set: { vm.setExportFormat($0) }))
        // Size — the offered presets depend on the format (ResolutionPreset.choices, spec §5.4:
        // GIF tops out at 720p because an ImageIO destination buffers frames). Under GIF this is
        // this app's **first four-cell SegmentedRow**, hence the shortened labels — see
        // sizeOptions.
        // The size caption is attached with spacing 2, not left to PanelSection's row spacing of
        // 8 (Theme.swift PanelCaption — a caption at the section's own spacing reads as belonging
        // to no control in particular, which is the problem rule 6 was written for).
        VStack(alignment: .leading, spacing: 2) {
            SegmentedRow("Size",
                         options: Self.sizeOptions(for: vm.exportFormat),
                         selection: $vm.exportResolution)
            if let note = Self.gifSizeNote(format: vm.exportFormat, frames: exportFrameCount,
                                           width: exportPixelSize.width,
                                           height: exportPixelSize.height) {
                PanelCaption(note)
            }
        }

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
                  + "\(vm.exportFormat.rawValue)")

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

    // MARK: - Format-dependent presentation (pure — tests: CinemaInspectorAffordanceTests)
    //
    // These four live here rather than inline in `settings` for the same reason
    // renderDisabledReason does: this repository does not test SwiftUI views, so a decision left
    // in a view body is a decision nothing pins. Wording that a user reads is a decision.

    /// GIF has no codec axis (spec §5.9). Returned as a decision rather than compared inline so
    /// that the "dim, do not hide" rule has one testable owner.
    static func codecRowDisabled(format: CinemaExportFormat) -> Bool {
        format.videoContainer == nil
    }

    /// The Size row's segments. The **labels shorten when the row has more than three cells**:
    /// the inspector is 240–400pt wide and the 52pt label column plus 4pt gaps leaves ~49pt per
    /// cell in a four-way split (against ~52pt in a three-way split, where "1440p" was already
    /// at the wrapping threshold — SegmentedControl's comment in Theme.swift is authoritative).
    /// Dropping the "p" buys back a glyph without relying on minimumScaleFactor to shrink the
    /// text below the surrounding rows' size.
    /// Keyed on the cell count, not on `format == .gif`, because the constraint is layout
    /// arithmetic: whichever list grows to four cells needs the short form.
    static func sizeOptions(for format: CinemaExportFormat) -> [SegmentOption<ResolutionPreset>] {
        let presets = ResolutionPreset.choices(for: format)
        return presets.map {
            SegmentOption($0, presets.count > 3 ? "\($0.shortSide)" : $0.rawValue)
        }
    }

    /// The GIF size caption (spec §5.8, user decision 4). Cinema keeps **every** frame — unlike
    /// the training timelapse, whose frames are automatic samples, these are the product of the
    /// user's keyframe editing, so dropping one in three would change the movement they authored.
    /// The caption therefore states facts and does not offer to reduce anything.
    ///
    /// The figure is `w × h × 4 × frames`, i.e. the decoded RGBA footprint of the whole sequence:
    /// an ImageIO GIF destination may buffer frames until finalize, so this is the realistic worst
    /// case rather than the file size (which palette + LZW makes far smaller and unpredictable).
    /// Calling it "in memory" instead of "file size" keeps that honest. It is also the **only**
    /// bound on the total: `ResolutionPreset.choices`' 720p cap limits one frame, not the frame
    /// count, so a long enough path reaches tens of GB at any allowed size (VideoExporter.swift
    /// states the arithmetic). Telling the user the real number before the render is what §5.8
    /// puts in place of a limit.
    ///
    /// FIX 1 — the gate is `videoContainer == nil`, not `== .gif`: the claim is about the
    /// **encoder arm** (ImageIO destinations buffer; AVAssetWriter streams), which is exactly what
    /// that property discriminates, and it is the form `consoleFormatSummary` and
    /// `codecRowDisabled` already use for the same reason ("a future format cannot land in the
    /// wrong arm by inheriting a `== .gif` comparison"). Behaviour is unchanged for today's three
    /// cases. `gifFpsNote` below deliberately keeps `== .gif`; the difference is explained there.
    static func gifSizeNote(format: CinemaExportFormat, frames: Int,
                            width: Int, height: Int) -> String? {
        guard format.videoContainer == nil, frames > 0, width > 0, height > 0 else { return nil }
        let bytes = Double(width) * Double(height) * 4 * Double(frames)
        let gb = bytes / 1_073_741_824
        let scale = gb >= 1
            ? String(format: "%.1f GB", gb)
            : String(format: "%.0f MB", bytes / 1_048_576)
        return "All \(frames) frames kept at \(width)×\(height) — about \(scale) in memory "
            + "while encoding."
    }

    /// The GIF frame-rate caption (spec §5.7). GIF delays are stored in 1/100s units and many
    /// viewers clamp anything under 0.02s to 0.1s, so 60fps (0.0167s) can play back at 10fps —
    /// six times too slow — while 24 (0.0417s) and 30 (0.0333s) are safe.
    ///
    /// A caption rather than a restricted choice list, because unlike the Size cap there is no
    /// resource that fails here; the cost is entirely the user's, so §3.4's "no silent
    /// truncation" points at telling them.
    ///
    /// The condition is the delay, not the literal 60: the training timelapse never hit this at
    /// all because its fps is capped at 30, and if Cinema ever offers 48 or 50 the arithmetic
    /// should classify them without anyone remembering this note exists.
    ///
    /// FIX 1 — this one **keeps `== .gif`** while `gifSizeNote` above moved to
    /// `videoContainer == nil`, and the split is deliberate rather than left over. The two gates
    /// answer different questions. Everywhere `videoContainer == nil` is used, the claim is about
    /// the encoder arm (no codec axis; ImageIO buffers frames) and holds for anything that is not
    /// AVAssetWriter. This claim is about the **GIF container's own file format** — a delay field
    /// quantised to 1/100s, plus the clamping convention viewers grew around it. APNG counts in
    /// numerator/denominator and WebP in milliseconds, so a future ImageIO format inheriting this
    /// note would print a sentence that names GIF and describes a limit it does not have. Here
    /// the general gate is the one that would be wrong.
    static func gifFpsNote(format: CinemaExportFormat, fps: Int) -> String? {
        guard format == .gif, fps > 0, 1.0 / Double(fps) < 0.02 else { return nil }
        // The alternatives are named from `VideoExportSettings.fpsChoices` minus the rates this
        // note can fire for, so 15 belongs here: `timelapse-viewport-aspect` Task 5 added it and
        // its 0.0667s delay is the roomiest margin over the 0.02s clamp any offered rate has —
        // leaving it out pointed the one user who cares about GIF size (the reason 15 was asked
        // for) at the two heaviest choices.
        return "\(fps) fps GIF: many viewers clamp delays under 0.02s to 0.1s and play it back "
            + "near 10 fps. 15, 24 or 30 avoids this."
    }

    /// The real pixel size of the current selection — the same call RenderJob.start makes, so the
    /// caption cannot state a size the render will not produce.
    private var exportPixelSize: (width: Int, height: Int) {
        VideoExportSettings.pixelSize(preset: vm.exportResolution, aspect: vm.outputAspect)
    }

    /// The frame count of the selected path, derived exactly as RenderJob.start derives it
    /// (FrameSchedule.frameCount, which is also GifExporter's declared count). 0 when no path is
    /// selected — the caption then disappears, which is correct: with no path there is nothing to
    /// warn about, and the CTA is already disabled for the same reason.
    private var exportFrameCount: Int {
        guard let path = vm.selectedPath else { return 0 }
        return FrameSchedule.frameCount(duration: path.duration, fps: vm.fps, loop: path.loop)
    }

    /// NSSavePanel — the extension and content type both come from the **format** (footnote 2 ⑤,
    /// widened by 2026-08-05 spec §5.2). Reading one enum for both is the point: they are two
    /// exhaustive switches over the same case, so an .mp4 name with a quickTimeMovie type — the
    /// exact mismatch a `case gif` on VideoContainer would have produced — cannot be assembled.
    private func presentSavePanel() {
        let format = vm.exportFormat
        let panel = NSSavePanel()
        panel.allowedContentTypes = [format.contentType]
        panel.canSelectHiddenExtension = true
        panel.nameFieldStringValue = "\(session.project.name).\(format.fileExtension)"
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
