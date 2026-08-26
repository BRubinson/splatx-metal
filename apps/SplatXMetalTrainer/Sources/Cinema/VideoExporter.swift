// Cinema Task 5 (2026-07-26 spec, "render and export pipeline"): the AVAssetWriter encoding
// wrapper plus pure configuration and conversion logic. Written in-house — it uses only the
// standard patterns of Apple's official APIs (WWDC19 608, pro app export), with no derivation from
// external code.
//
// Pipeline contract:
// - The container is **MP4 by default, MOV optional** (footnote 2 ⑤ — widened from the previous
//   .mov-only. The AVFileType mapping is VideoContainer, and AVAssetWriter records HEVC-in-MP4
//   with the hvc1 tag — demonstrated by the smoke test). HEVC by default, H.264 optional. The
//   resolution presets are the output's **short side** in pixels, and the long side is derived
//   from the aspect ratio (16:9/9:16/1:1) — the real pixel table is authoritative in
//   VideoExportTests.testResolutionTableAllCombos.
// - The user-facing export axis is **CinemaExportFormat** (MP4/MOV/GIF), which is one level above
//   VideoContainer: only the two video formats reach AVAssetWriter at all (2026-08-05 spec §5.2,
//   §5.5). VideoContainer stays a two-case AVFileType mapping — see the CinemaExportFormat
//   comment for why widening it instead would fail silently.
// - **All three BT.709 keys are stated explicitly** (AVVideoColorPropertiesKey —
//   Primaries/Transfer/YCbCrMatrix all ITU_R_709_2). This is the defense against the classic trap
//   where omitting them makes AVAssetWriter record 601 (Apple forum 5681). The render output is in
//   the sRGB encoding space (the sx_render_session_frame header contract — no gamma conversion),
//   and tagging sRGB content as 709 is the industry-standard compromise for screen-derived content
//   (accepting the small difference between the sRGB EOTF and the 709 OETF — research B §1.3). The
//   same three are attached to the pixel buffer as well, stating the input interpretation of the
//   BGRA→YUV conversion.
// - PixelBufferPool (BGRA + IOSurface backing + Metal compatible) — do not Create per frame; reuse
//   the pool.
// - `expectsMediaDataInRealTime = false` + requestMediaDataWhenReady, a **pull model**: a bounded
//   blocking queue (FrameQueue) sits between the render thread (producer) and the writer queue
//   (consumer). If the encoder falls behind, push blocks and the renderer idles (back pressure),
//   and if there are no frames, pop blocks and the writer queue idles. In practice the splat
//   render (tens to hundreds of ms/frame) is far slower than the hardware encoder, so back
//   pressure is a safety net.
// - Cancellation = cancelWriting + **deleting the partial file** (spec, "progress, cancellation,
//   and partial file cleanup").
// - The per-frame interface is the **FrameSink** protocol (2026-08-05 spec §5.5, declared below).
//   It was *extracted from* this class — every signature is unchanged — so that the GIF path
//   (GifExporter, ImageIO) can reuse RenderJob's frame loop instead of duplicating it. Nothing was
//   pushed into this class in return: see the FrameSink comment for why the format branch has to
//   sit one level above the sink.
//
// Thread contract: append/finish/cancel are called from exactly one place, the render job's
// dedicated Thread (RenderJob.run — AVAssetWriter is not Sendable, and the whole session is
// confined to one thread). The internal writer queue is a single serial queue owned by this class.
import Foundation
import AVFoundation
import CoreVideo
import CoreGraphics
import Accelerate
import UniformTypeIdentifiers

// MARK: - Codec, container, and resolution (pure — VideoExportTests is the source of truth)

/// The output codec. HEVC by default (~40% smaller than H.264 at the same quality); H.264 is for
/// legacy compatibility.
enum VideoCodec: String, CaseIterable, Identifiable {
    case hevc = "HEVC"
    case h264 = "H.264"

    var id: String { rawValue }
    var avCodec: AVVideoCodecType { self == .hevc ? .hevc : .h264 }
}

/// The output container (footnote 2 ⑤). MP4 by default (sharing and web compatibility come
/// first); MOV is the Apple-ecosystem option.
/// AVAssetWriter records HEVC-in-MP4 with the hvc1 codec tag (no hev1 incompatibility trap — the
/// smoke test's ffprobe codec_tag_string demonstrates it). The save panel's extension and content
/// type consume this single mapping as well (the file extension consistency contract).
enum VideoContainer: String, CaseIterable, Identifiable {
    case mp4 = "MP4"
    case mov = "MOV"

    var id: String { rawValue }
    var avFileType: AVFileType { self == .mp4 ? .mp4 : .mov }
    var fileExtension: String { self == .mp4 ? "mp4" : "mov" }
    /// For synchronizing NSSavePanel.allowedContentTypes.
    var contentType: UTType { self == .mp4 ? .mpeg4Movie : .quickTimeMovie }
}

/// The **export format** — the "File" row of the EXPORT section (2026-08-05 spec §5.1: "Format" is
/// the codec axis, so the format axis is named File in the UI).
///
/// Why this is a separate enum instead of `case gif` on VideoContainer (spec §5.2 — this is the
/// whole reason the type exists, so do not merge them back):
/// - VideoContainer's three derived properties were written as **ternaries**
///   (`self == .mp4 ? … : …`). A third case compiles with **zero diagnostics** and silently falls
///   into the MOV branch, so `VideoExporter.init` would hand `.mov` to AVAssetWriter and produce a
///   MOV file whose only GIF-ness is the `.gif` extension.
/// - Second blast radius: `TimelapseFormat.videoContainer == nil` **is** the GIF discriminator
///   (`TimelapseEncoder.swift` — `guard let container = … else { return try encodeGIF }`). Giving
///   gif a container collapses that guard and routes the *training* timelapse GIF through
///   AVAssetWriter too.
/// Every property below is therefore an **exhaustive switch**: a fourth format cannot be added
/// without the compiler naming each decision.
///
/// Not shared with TimelapseFormat because that one is a `Codable` persisted in project.json with
/// a lenient-decode contract, while Cinema's export settings are session state that is never
/// persisted (CinemaViewModel — only outputAspect is persisted).
enum CinemaExportFormat: String, CaseIterable, Identifiable {
    case mp4 = "MP4"
    case mov = "MOV"
    case gif = "GIF"

    var id: String { rawValue }

    /// The AVAssetWriter container this format encodes through — **nil means GIF**, which goes to
    /// ImageIO (GifExporter) instead. This mirrors TimelapseFormat.videoContainer so both features
    /// use one discriminator shape.
    var videoContainer: VideoContainer? {
        switch self {
        case .mp4: .mp4
        case .mov: .mov
        case .gif: nil
        }
    }

    var fileExtension: String {
        switch self {
        case .mp4: "mp4"
        case .mov: "mov"
        case .gif: "gif"
        }
    }

    /// For synchronizing NSSavePanel.allowedContentTypes with the extension above.
    var contentType: UTType {
        switch self {
        case .mp4: .mpeg4Movie
        case .mov: .quickTimeMovie
        case .gif: .gif
        }
    }
}

/// The resolution presets — the name refers to the output's **short side** in pixels.
/// The long side is derived from the aspect ratio (see VideoExportSettings.pixelSize); which
/// presets are offered depends on the export format (`choices(for:)`).
enum ResolutionPreset: String, CaseIterable, Identifiable {
    case p240 = "240p"
    case p360 = "360p"
    case p480 = "480p"
    case p720 = "720p"
    case p1080 = "1080p"
    case p1440 = "1440p"
    case uhd4K = "4K"

    var id: String { rawValue }
    var shortSide: Int {
        switch self {
        case .p240: 240
        case .p360: 360
        case .p480: 480
        case .p720: 720
        case .p1080: 1080
        case .p1440: 1440
        case .uhd4K: 2160
        }
    }

    /// The presets offered for a given format. **GIF is restricted to ≤720p, and that is a
    /// deliberate departure from this app's own precedent** (spec §5.4): the training timelapse
    /// leaves format and resolution independent in the UI and reports the GIF clamp afterwards
    /// (`gifReductionNote` — "no silent truncation"). That works there because the only clamp is
    /// `gifMaxLongSide = 960` on an already-written frame directory.
    /// Cinema cannot report afterwards: an ImageIO GIF destination buffers frames, so 4K × 301
    /// frames demands several GB **before** any report could be printed — the app dies first.
    ///
    /// **This ceiling bounds the cost of one frame, not the total** (FIX 1 — the previous wording,
    /// "720p × 301 is already ≈1.1GB, which is why the ceiling is there and not higher", invited
    /// the reading that the cap makes memory safe). The footprint is `w · h · 4 · frames` and
    /// **frames has no cap at all** — duration and fps are the user's to set, so 2 minutes at
    /// 60fps is 7201 frames ≈ 26GB even at 720p. What 720p buys is a 9× smaller frame than 4K
    /// (3.7MB against 33MB), which moves the ordinary export out of danger; it does not make the
    /// worst case finite.
    /// The defence for the total is a different one and lives elsewhere: `RenderJob.gifSizeNote`
    /// prints the real figure for the current selection **before** the render starts, so the
    /// user is told the number while it is still theirs to change (spec §5.8 — state facts, do
    /// not silently reduce). Neither mechanism substitutes for the other.
    ///
    /// Do not "unify with the timelapse" by deleting this — the resource argument is what differs.
    static let videoChoices: [ResolutionPreset] = [.p1080, .p1440, .uhd4K]
    static let gifChoices: [ResolutionPreset] = [.p240, .p360, .p480, .p720]

    /// Exhaustive switch rather than `format == .gif ? … : …` for the same reason
    /// CinemaExportFormat's properties are: a new format must not silently inherit a list.
    static func choices(for format: CinemaExportFormat) -> [ResolutionPreset] {
        switch format {
        case .mp4, .mov: videoChoices
        case .gif: gifChoices
        }
    }
}

/// The encoder settings builder (a set of pure functions). Only VideoExporter touches
/// AVAssetWriter.
enum VideoExportSettings {
    /// The fps choices. CinemaViewModel.setFPS uses them for validation.
    ///
    /// 15 was added on user request (2026-08-05 spec §3, "export rate에 15 옵션도 추가해야겠어.
    /// gif 때문에") — a GIF at 15fps is roughly half the frames of 30 for the same motion, which is
    /// the lever that actually moves a GIF's size, and `gifSizeNote`'s figure is linear in it.
    /// Every consumer was checked against the new low end rather than assumed safe (spec §3.1):
    /// `FrameSchedule.frameCount` rounds, so 15 stays integral; `CMTime` timescale 15 is exact in
    /// Int32 and keeps PTS rational; the GOP key (`AVVideoMaxKeyFrameIntervalKey: fps`) is still
    /// exactly one second; and `bitrate`'s 2Mbps floor is not reached — the smallest combination
    /// (1:1 1080p at 15fps, HEVC) is 2.62Mbps, so no selection silently lands on the clamp. Below
    /// about 12fps that floor would start to bind, which is where this list should stop growing
    /// downward without redoing that arithmetic.
    ///
    /// GIF-side quantisation is **not** a reason to exclude it: 1/15 = 6.67cs does not land on
    /// GIF's 1/100s grid, but neither does any other choice here (24 → 4.17, 30 → 3.33,
    /// 60 → 1.67 — only 10/20/25/50 are exact). ImageIO's measured rounding for each is pinned by
    /// GifExporterTests.testGifDelayQuantisationForEveryRateChoice, and `RenderJob.gifFpsNote`
    /// classifies 15 as safe by arithmetic (0.0667 ≫ 0.02) without needing a case for it.
    static let fpsChoices = [15, 24, 30, 60]

    /// Preset × aspect ratio → real pixels. The short side = the preset, and the long side is
    /// derived from the aspect ratio (all even — an encoder dimension requirement). The table is
    /// authoritative in the test (testResolutionTableAllCombos).
    ///
    /// The long side used to be `short / 9 * 16`, with the comment "1080→1920, 1440→2560,
    /// 2160→3840 (all integers)". **That premise broke** when the GIF presets were added (spec
    /// §5.3): integer division is only exact when the short side is a multiple of 9, and 240 and
    /// 480 are not — 240/9·16 = 416 against a true 426.67 (−2.5%), 480/9·16 = 848 against 853.33
    /// (−0.6%). Rounding the real ratio corrects those two to 428 and 854 and leaves every
    /// previously shipped value untouched (360→640, 720→1280, 1080→1920, 1440→2560, 2160→3840),
    /// which testResolutionTableAllCombos pins.
    ///
    /// evenUp is an encoder requirement that GIF does not have, but it is applied to **one shared
    /// formula** on purpose: if the pixel size depended on the format, "the same 480 gives
    /// different frames in MP4 and GIF" — and the WYSIWYG preview has only one size to show.
    static func pixelSize(preset: ResolutionPreset, aspect: OutputAspect)
        -> (width: Int, height: Int) {
        let short = preset.shortSide   // every preset is even, so only the long side needs evenUp
        let long = TimelapseSettings.evenUp(Int((Double(short) * 16.0 / 9.0).rounded()))
        switch aspect {
        case .wide: return (long, short)
        case .tall: return (short, long)
        case .square: return (short, short)
        }
    }

    /// The average bitrate = the pixel rate × bpp. HEVC 0.15 / H.264 0.20 — coefficients that land
    /// exactly inside research B's recommended band at 4K60 (HEVC 50–80Mbps, H.264 ≤100Mbps).
    static func bitrate(width: Int, height: Int, fps: Int, codec: VideoCodec) -> Int {
        let pixelsPerSecond = Double(width * height * fps)
        let bpp = codec == .hevc ? 0.15 : 0.20
        return max(Int(pixelsPerSecond * bpp), 2_000_000)
    }

    /// AVAssetWriterInput outputSettings — the three BT.709 keys + the average bitrate + a 1-second
    /// GOP.
    static func outputSettings(codec: VideoCodec, width: Int, height: Int, fps: Int)
        -> [String: Any] {
        [
            AVVideoCodecKey: codec.avCodec,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
            ],
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: bitrate(width: width, height: height,
                                                  fps: fps, codec: codec),
                AVVideoExpectedSourceFrameRateKey: fps,
                AVVideoMaxKeyFrameIntervalKey: fps,   // a 1-second GOP
            ],
        ]
    }

    /// The adaptor pool attributes — BGRA + IOSurface backing (leaving room for a zero-cost
    /// Metal↔CoreVideo conversion) + Metal compatibility.
    static func pixelBufferAttributes(width: Int, height: Int) -> [String: Any] {
        [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
        ]
    }
}

// MARK: - Frame schedule (t = i/fps — the brief's contract)

enum FrameSchedule {
    /// The total frame count. An open path **includes** t = duration as its last frame
    /// (N = round(d·fps)+1), while a loop path **excludes** it because t = duration is the same
    /// pose as t = 0 (N = round(d·fps)) — so concatenating produces no duplicate frame at the
    /// seam. Lower bounds: 1 frame for loop, 2 for open.
    static func frameCount(duration: Double, fps: Int, loop: Bool) -> Int {
        guard duration > 0, fps > 0 else { return 1 }
        let n = max(Int((duration * Double(fps)).rounded()), 1)
        return loop ? n : n + 1
    }

    /// The time (seconds) of frame i = i/fps. The sampler is responsible for the [0, duration]
    /// clamp.
    static func time(ofFrame i: Int, fps: Int) -> Double {
        Double(i) / Double(max(fps, 1))
    }
}

// MARK: - RGBA→BGRA (the session's RGBA8 output → CVPixelBuffer 32BGRA)

enum PixelPack {
    /// RGBA8 → BGRA8 (an R↔B swap) — vImagePermuteChannels, with support for dst rowBytes
    /// padding.
    /// map[i] = the src channel index that dst channel i reads:
    /// BGRA ← (src[2], src[1], src[0], src[3]).
    static func rgbaToBGRA(src: UnsafeRawPointer, width: Int, height: Int,
                           dst: UnsafeMutableRawPointer, dstBytesPerRow: Int) {
        var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src),
                                   height: vImagePixelCount(height),
                                   width: vImagePixelCount(width),
                                   rowBytes: width * 4)
        var dstBuf = vImage_Buffer(data: dst,
                                   height: vImagePixelCount(height),
                                   width: vImagePixelCount(width),
                                   rowBytes: dstBytesPerRow)
        let map: [UInt8] = [2, 1, 0, 3]
        vImagePermuteChannels_ARGB8888(&srcBuf, &dstBuf, map, vImage_Flags(kvImageNoFlags))
    }

    /// The array version (a convenience for tests and small inputs) — rowBytes = width·4, densely
    /// packed.
    static func rgbaToBGRA(_ rgba: [UInt8], width: Int, height: Int) -> [UInt8] {
        precondition(rgba.count >= width * height * 4)
        var out = [UInt8](repeating: 0, count: width * height * 4)
        rgba.withUnsafeBytes { src in
            out.withUnsafeMutableBytes { dst in
                rgbaToBGRA(src: src.baseAddress!, width: width, height: height,
                           dst: dst.baseAddress!, dstBytesPerRow: width * 4)
            }
        }
        return out
    }
}

// MARK: - Overlay compositor

/// Composites one static image over the frame. rect is **the value produced by
/// OverlayPlacement.rect** (output pixels, origin at top-left with +y downward — the Task 4
/// contract of reusing the same function as the WYSIWYG; the builder is
/// RenderJob.overlayCompositor). Drawing is performed flipped into CG coordinates (origin at
/// bottom-left).
struct OverlayCompositor {
    let image: CGImage
    let rect: CGRect
    let opacity: Double

    /// ctx = a bitmap context covering the whole output frame (memory row 0 = the topmost
    /// scanline).
    func draw(in ctx: CGContext, outputHeight: Int) {
        guard rect.width > 0, rect.height > 0, opacity > 0 else { return }
        let flipped = CGRect(x: rect.minX, y: CGFloat(outputHeight) - rect.maxY,
                             width: rect.width, height: rect.height)
        ctx.saveGState()
        ctx.setAlpha(CGFloat(min(max(opacity, 0), 1)))
        ctx.interpolationQuality = .high
        ctx.draw(image, in: flipped)
        ctx.restoreGState()
    }
}

// MARK: - Frame sink (what RenderJob's per-frame loop is written against)

/// The minimum an encoder has to offer the render loop: take one frame, then either finish or
/// cancel. Two conformers — `VideoExporter` (AVAssetWriter → MP4/MOV) and `GifExporter`
/// (ImageIO → GIF).
///
/// **Why the format branch is above the sink and not inside VideoExporter** (2026-08-05 spec §5.5,
/// and the reason this protocol exists at all): everything this file owns below the writer — the
/// pixel buffer pool, the BT.709 tagging, FrameQueue, the requestMediaDataWhenReady pull loop — is
/// AVAssetWriter-specific. An internal `if gif` would leave most of the class dead on the GIF path
/// while still constructing it. Timelapse already settled on the same shape one level up
/// (`TimelapseEncoder.encode` dispatches on `format.videoContainer == nil` and simply never builds
/// a VideoExporter for GIF).
///
/// Contracts every conformer inherits — both are load-bearing, and neither is expressible in the
/// signatures:
/// - **One thread.** append/finish/cancel come from exactly one thread (RenderJob's dedicated
///   Thread). The reason is *not* the encoder — it is `sx_render_session_frame`, a blocking GPU
///   round trip that must not occupy a Swift concurrency executor.
/// - **cancel() deletes the partial file.** A failure notification and a broken output file must
///   never coexist (RenderJob FIX 2 / I1). The caller invokes cancel() on the render-failure,
///   encode-failure, user-cancellation, and finalize-failure paths, so cancel() must be safe to
///   call twice and in a terminal state.
protocol FrameSink: AnyObject {
    /// One frame. `rgba` is width·height·4 bytes, densely packed, memory row 0 = the topmost
    /// scanline. **The caller repaints that same buffer for the next frame**, so a sink must copy
    /// what it needs and must not retain the pointer (the trap GifExporter documents at length).
    /// `compositor`, when present, is the watermark and must actually be drawn.
    func append(rgba: UnsafePointer<UInt8>, frameIndex: Int,
                compositor: OverlayCompositor?) throws
    func finish() throws
    func cancel()
}

/// Conformance only — the three members already existed with these exact signatures.
extension VideoExporter: FrameSink {}

// MARK: - Frame queue (the producing render thread ↔ the consuming writer queue — the
// implementation of back pressure)

/// A bounded blocking queue. push blocks when capacity is exceeded (encoder back pressure), and
/// pop blocks on an empty queue (waiting for the next frame). finish/cancel wake both sides.
private final class FrameQueue: @unchecked Sendable {
    enum Item {
        case frame(CVPixelBuffer, CMTime)
        case finished
        case cancelled
    }

    private let cond = NSCondition()
    private var frames: [(CVPixelBuffer, CMTime)] = []
    private var isFinished = false
    private var isCancelled = false
    private let capacity: Int

    init(capacity: Int) { self.capacity = capacity }

    /// true on success. false when cancelled (the buffer is discarded).
    func push(_ buffer: CVPixelBuffer, _ time: CMTime) -> Bool {
        cond.lock(); defer { cond.unlock() }
        while frames.count >= capacity && !isCancelled { cond.wait() }
        if isCancelled { return false }
        frames.append((buffer, time))
        cond.broadcast()
        return true
    }

    func pop() -> Item {
        cond.lock(); defer { cond.unlock() }
        while frames.isEmpty && !isFinished && !isCancelled { cond.wait() }
        if isCancelled { return .cancelled }
        if !frames.isEmpty {
            let f = frames.removeFirst()
            cond.broadcast()
            return .frame(f.0, f.1)
        }
        return .finished
    }

    func finish() {
        cond.lock(); isFinished = true; cond.broadcast(); cond.unlock()
    }

    func cancel() {
        cond.lock(); isCancelled = true; frames.removeAll(); cond.broadcast(); cond.unlock()
    }
}

// MARK: - VideoExporter

final class VideoExporter: @unchecked Sendable {
    struct Config {
        let codec: VideoCodec
        /// Footnote 2 ⑤ — the file type is decided once, at writer creation.
        let container: VideoContainer
        let width: Int
        let height: Int
        let fps: Int
    }

    enum ExporterError: LocalizedError {
        case setupFailed(String)
        case writeFailed(String)

        var errorDescription: String? {
            switch self {
            case .setupFailed(let why): "Encoder setup failed: \(why)"
            case .writeFailed(let why): "Encoder write failed: \(why)"
            }
        }
    }

    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    private let config: Config
    private let outputURL: URL
    /// A serial queue dedicated to the writer pull model — even when a blocking pop occupies this
    /// queue there is no other consumer.
    private let writerQueue = DispatchQueue(label: "com.postmedia.splatx.video-export")
    private let frames = FrameQueue(capacity: 4)
    /// The pull-loop termination signal — finishWriting completion, an append failure, and
    /// cancellation all signal it.
    private let finishSem = DispatchSemaphore(value: 0)
    private let stateLock = NSLock()
    private var writeError: String?

    /// Writer creation + startWriting + starting the pull loop, all in one. The output file is
    /// created at this point — every later failure path MUST clean up the partial file with
    /// cancel().
    init(outputURL: URL, config: Config) throws {
        // Reached after the NSSavePanel overwrite confirmation — the writer rejects an existing
        // file, so it is removed first.
        try? FileManager.default.removeItem(at: outputURL)
        self.outputURL = outputURL
        self.config = config
        writer = try AVAssetWriter(outputURL: outputURL, fileType: config.container.avFileType)
        writer.shouldOptimizeForNetworkUse = true   // moov placed at the front — sharing-friendly
        input = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: VideoExportSettings.outputSettings(
                codec: config.codec, width: config.width, height: config.height,
                fps: config.fps))
        input.expectsMediaDataInRealTime = false   // offline — the pull model
        adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: VideoExportSettings.pixelBufferAttributes(
                width: config.width, height: config.height))
        guard writer.canAdd(input) else {
            throw ExporterError.setupFailed("input rejected by writer")
        }
        writer.add(input)
        guard writer.startWriting() else {
            throw ExporterError.setupFailed(writer.error?.localizedDescription ?? "startWriting")
        }
        writer.startSession(atSourceTime: .zero)
        startPullLoop()
    }

    /// The requestMediaDataWhenReady pull loop: while ready, a blocking pop from the queue →
    /// append.
    /// .finished → markAsFinished + finishWriting; .cancelled or an append failure → exit
    /// immediately (every exit path signals finishSem — so the wait in finish() never hangs
    /// forever).
    private func startPullLoop() {
        input.requestMediaDataWhenReady(on: writerQueue) { [weak self] in
            guard let self else { return }
            while self.input.isReadyForMoreMediaData {
                switch self.frames.pop() {
                case .frame(let buffer, let time):
                    if !self.adaptor.append(buffer, withPresentationTime: time) {
                        self.setWriteError(self.writer.error?.localizedDescription
                                           ?? "append rejected")
                        self.frames.cancel()
                        self.finishSem.signal()
                        return
                    }
                case .finished:
                    self.input.markAsFinished()
                    self.writer.finishWriting { self.finishSem.signal() }
                    return
                case .cancelled:
                    self.finishSem.signal()
                    return
                }
            }
        }
    }

    /// Supplies one frame (from the render thread) — acquire a pool buffer → RGBA→BGRA →
    /// (optionally) composite the overlay through CG → attach BT.709 → push to the queue (blocking
    /// when capacity is exceeded = back pressure). rgba is width·height·4 bytes.
    func append(rgba: UnsafePointer<UInt8>, frameIndex: Int,
                compositor: OverlayCompositor?) throws {
        try throwIfWriteError()
        guard let pool = adaptor.pixelBufferPool else {
            throw ExporterError.writeFailed(writer.error?.localizedDescription
                                            ?? "pixel buffer pool unavailable")
        }
        var maybeBuffer: CVPixelBuffer?
        let rc = CVPixelBufferPoolCreatePixelBuffer(nil, pool, &maybeBuffer)
        guard rc == kCVReturnSuccess, let buffer = maybeBuffer else {
            throw ExporterError.writeFailed("pixel buffer create rc=\(rc)")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        if let base = CVPixelBufferGetBaseAddress(buffer) {
            let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
            PixelPack.rgbaToBGRA(src: rgba, width: config.width, height: config.height,
                                 dst: base, dstBytesPerRow: rowBytes)
            if let compositor,
               let ctx = CGContext(data: base, width: config.width, height: config.height,
                                   bitsPerComponent: 8, bytesPerRow: rowBytes,
                                   space: CGColorSpace(name: CGColorSpace.sRGB)
                                       ?? CGColorSpaceCreateDeviceRGB(),
                                   bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                       | CGBitmapInfo.byteOrder32Little.rawValue) {
                compositor.draw(in: ctx, outputHeight: config.height)
            }
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        Self.attachBT709(buffer)
        let time = CMTime(value: CMTimeValue(frameIndex), timescale: CMTimeScale(config.fps))
        if !frames.push(buffer, time) {
            try throwIfWriteError()
            throw ExporterError.writeFailed("cancelled")
        }
        try throwIfWriteError()
    }

    /// Drain the remaining queue → close the track → wait (blocking) for finishWriting to
    /// complete. Throws when the status is not completed.
    func finish() throws {
        try throwIfWriteError()
        frames.finish()
        finishSem.wait()
        try throwIfWriteError()
        guard writer.status == .completed else {
            throw ExporterError.writeFailed(writer.error?.localizedDescription
                                            ?? "status \(writer.status.rawValue)")
        }
    }

    /// Cancellation — it induces the pull loop to exit, then does cancelWriting plus **deleting
    /// the partial file** on the writer queue.
    /// frames.cancel() comes first: if a blocking pop is occupying the writer queue, sync
    /// deadlocks.
    func cancel() {
        frames.cancel()
        writerQueue.sync {
            if writer.status == .writing { writer.cancelWriting() }
            try? FileManager.default.removeItem(at: outputURL)
        }
    }

    // MARK: Internal

    private func setWriteError(_ message: String) {
        stateLock.lock(); defer { stateLock.unlock() }
        if writeError == nil { writeError = message }
    }

    private func throwIfWriteError() throws {
        stateLock.lock(); defer { stateLock.unlock() }
        if let writeError { throw ExporterError.writeFailed(writeError) }
    }

    /// Attaches the three BT.709 keys to the pixel buffer — pairing with the track tagging in
    /// outputSettings, it states the input interpretation of the BGRA→YUV conversion (the buffers
    /// come from a reused pool, so resetting them is idempotent).
    private static func attachBT709(_ buffer: CVPixelBuffer) {
        CVBufferSetAttachment(buffer, kCVImageBufferColorPrimariesKey,
                              kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(buffer, kCVImageBufferTransferFunctionKey,
                              kCVImageBufferTransferFunction_ITU_R_709_2, .shouldPropagate)
        CVBufferSetAttachment(buffer, kCVImageBufferYCbCrMatrixKey,
                              kCVImageBufferYCbCrMatrix_ITU_R_709_2, .shouldPropagate)
    }
}
