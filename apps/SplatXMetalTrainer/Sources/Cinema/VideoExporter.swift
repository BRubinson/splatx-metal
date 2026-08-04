// Cinema Task 5 (2026-07-26 spec, "render and export pipeline"): the AVAssetWriter encoding
// wrapper plus pure configuration and conversion logic. Written in-house — it uses only the
// standard patterns of Apple's official APIs (WWDC19 608, pro app export), with no derivation from
// external code.
//
// Pipeline contract:
// - The container is **MP4 by default, MOV optional** (footnote 2 ⑤ — widened from the previous
//   .mov-only. The AVFileType mapping is VideoContainer, and AVAssetWriter records HEVC-in-MP4
//   with the hvc1 tag — demonstrated by the smoke test). HEVC by default, H.264 optional. The
//   resolution presets (1080p/1440p/4K) are the output's **short side** in pixels, and the long
//   side is derived from the aspect ratio (16:9/9:16/1:1) — the real pixel table is authoritative
//   in VideoExportTests.testResolutionTableAllCombos.
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

/// The resolution presets — the name refers to the output's **short side** in pixels
/// (1080/1440/2160).
/// The long side is derived from the aspect ratio: for the 16:9 family, shortSide/9·16 is an
/// integer for all three values (1920/2560/3840).
enum ResolutionPreset: String, CaseIterable, Identifiable {
    case p1080 = "1080p"
    case p1440 = "1440p"
    case uhd4K = "4K"

    var id: String { rawValue }
    var shortSide: Int {
        switch self {
        case .p1080: 1080
        case .p1440: 1440
        case .uhd4K: 2160
        }
    }
}

/// The encoder settings builder (a set of pure functions). Only VideoExporter touches
/// AVAssetWriter.
enum VideoExportSettings {
    /// The fps choices (spec: 24/30/60). CinemaViewModel.setFPS uses them for validation.
    static let fpsChoices = [24, 30, 60]

    /// Preset × aspect ratio → real pixels. The short side = the preset, and the long side is
    /// derived from the aspect ratio (all even — an encoder dimension requirement). The table is
    /// authoritative in the test (testResolutionTableAllCombos).
    static func pixelSize(preset: ResolutionPreset, aspect: OutputAspect)
        -> (width: Int, height: Int) {
        let short = preset.shortSide
        let long = short / 9 * 16   // 1080→1920, 1440→2560, 2160→3840 (all integers)
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
