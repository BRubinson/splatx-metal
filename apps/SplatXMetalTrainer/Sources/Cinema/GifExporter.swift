// Cinema GIF export (2026-08-05 spec §5.5/§5.6): the FrameSink that writes an animated GIF through
// ImageIO. Written in-house from Apple's CGImageDestination API; the property idioms match the
// training timelapse's GIF path (TimelapseEncoder.encodeGIF), which is this app's settled form.
//
// Why a class of its own instead of a branch inside VideoExporter — the FrameSink comment in
// VideoExporter.swift is the authoritative answer (everything that class owns below the
// AVAssetWriter is writer-specific).
//
// **The four traps spec §5.6 names.** Every one of them compiles cleanly when omitted and produces
// a plausible-looking file, which is why each has its rationale written next to it below:
// 1. **Overlay compositing.** The watermark exists only because append() draws it. Consuming the
//    frame without calling `compositor.draw` yields a perfectly valid, silently watermark-less
//    GIF — no error, no warning, no diagnostic.
// 2. **Pixel copying.** RenderJob reuses **one** rgba array for every frame. Wrapping that buffer
//    in a CGDataProvider-backed CGImage hands ImageIO a lazy reference, and the pixels it reads
//    later are whatever the *next* frame overwrote them with — the classic outcome is N copies of
//    the last frame. Every frame is therefore copied into CoreGraphics-owned memory and
//    snapshotted with makeImage(). TimelapseEncoder's GIF path is **not** a precedent here: it
//    decodes a fresh CGImage from a file per frame and never touches a shared buffer.
// 3. **Partial file cleanup.** cancel() and every throwing exit delete the output, so "Render
//    failed" and a broken .gif on disk never coexist (the pairing VideoExporter.cancel already
//    makes — RenderJob FIX 2 / I1).
// 4. **The frame count is declared up front.** CGImageDestinationCreateWithURL takes it at
//    creation; Cinema already knows it from FrameSchedule.frameCount.
//
// Thread contract: identical to VideoExporter's — append/finish/cancel come from exactly one
// thread, RenderJob's dedicated Thread. The reason is not the encoder (ImageIO would not care) but
// `sx_render_session_frame`: a blocking GPU round trip that must not occupy a Swift concurrency
// executor. Because ownership is single-threaded this class needs no lock, unlike VideoExporter
// whose writer queue is a genuine second thread.
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

final class GifExporter: FrameSink {
    struct Config {
        let width: Int
        let height: Int
        /// Only used to derive the per-frame delay (1/fps). GIF has no global rate — the delay is
        /// stored per frame and quantized to 1/100 s by the format, which is why 60fps needs the
        /// caption warning in the EXPORT section (spec §5.7) rather than anything here.
        let fps: Int
        /// Trap ④ — CGImageDestinationCreateWithURL demands the count at creation.
        let frameCount: Int
    }

    enum ExporterError: LocalizedError {
        case setupFailed(String)
        case writeFailed(String)

        var errorDescription: String? {
            switch self {
            case .setupFailed(let why): "GIF setup failed: \(why)"
            case .writeFailed(let why): "GIF write failed: \(why)"
            }
        }
    }

    private let outputURL: URL
    private let config: Config
    private let frameProperties: CFDictionary
    /// nil once the file is no longer ours to write: after cancel(), after a throwing exit, and
    /// after a successful finish(). Doubling as the state flag keeps "released the destination"
    /// and "refuses further frames" from drifting apart.
    private var destination: CGImageDestination?
    private var added = 0

    /// Creates the destination and stamps the file-level properties. The output is removed first,
    /// so from here on the file at `outputURL` is this exporter's to delete (the invariant
    /// cancel() relies on) — the same opening move as VideoExporter.init, which is likewise
    /// reached only after the NSSavePanel overwrite confirmation.
    init(outputURL: URL, config: Config) throws {
        self.outputURL = outputURL
        self.config = config
        // Both delay keys, deliberately (trap ④'s companion): kCGImagePropertyGIFDelayTime is what
        // strict/legacy readers consume and is the one subject to the sub-0.02 s viewer clamp,
        // while UnclampedDelayTime is the key that asks for the stored number without that
        // convention applied. Writing only one leaves either the old readers or the modern ones
        // guessing. TimelapseEncoder.encodeGIF writes the same pair.
        //
        // "Unclamped" does **not** mean "unrounded", and an earlier wording here ("carries the
        // real number") invited that reading. A GIF Graphic Control Extension holds one 16-bit
        // centisecond field per frame, so whatever Double goes in comes back quantised in *both*
        // keys — 1/15 s is stored as 7cs and reads back as 0.07 either way. Measured and pinned by
        // GifExporterTests.testGifDelayQuantisationForEveryRateChoice, which tabulates every rate.
        let delay = 1.0 / Double(max(config.fps, 1))
        frameProperties = [
            kCGImagePropertyGIFDictionary: [
                kCGImagePropertyGIFDelayTime: delay,
                kCGImagePropertyGIFUnclampedDelayTime: delay,
            ],
        ] as CFDictionary

        guard config.width > 0, config.height > 0, config.frameCount > 0 else {
            throw ExporterError.setupFailed(
                "empty output (\(config.width)×\(config.height), \(config.frameCount) frames)")
        }
        try? FileManager.default.removeItem(at: outputURL)
        guard let destination = CGImageDestinationCreateWithURL(
            outputURL as CFURL, UTType.gif.identifier as CFString, config.frameCount, nil) else {
            throw ExporterError.setupFailed("cannot create \(outputURL.lastPathComponent)")
        }
        // Loop count 0 = forever. This is a file-level property (the Netscape application
        // extension), not a per-frame one — setting it in frameProperties silently does nothing.
        CGImageDestinationSetProperties(destination, [
            kCGImagePropertyGIFDictionary: [kCGImagePropertyGIFLoopCount: 0],
        ] as CFDictionary)
        self.destination = destination
    }

    /// Supplies one frame. rgba is width·height·4 bytes and is **borrowed only for the duration of
    /// this call** — makeFrameImage copies out of it before returning (trap ②).
    func append(rgba: UnsafePointer<UInt8>, frameIndex: Int,
                compositor: OverlayCompositor?) throws {
        guard let destination else {
            throw ExporterError.writeFailed("cancelled")
        }
        guard added < config.frameCount else {
            throw discarding("frame \(frameIndex) exceeds the declared count \(config.frameCount)")
        }
        // GIF has no random access — frames land in call order, so frameIndex is not a position
        // but a consistency check. VideoExporter turns the same argument into a presentation
        // timestamp, so a caller that skipped or reordered would get a correct video and a
        // silently mistimed GIF; refusing costs nothing next to that asymmetry.
        guard frameIndex == added else {
            throw discarding("frame \(frameIndex) out of order (expected \(added))")
        }
        guard let image = Self.makeFrameImage(rgba: rgba, width: config.width,
                                              height: config.height, compositor: compositor) else {
            throw discarding("frame \(frameIndex): bitmap context unavailable")
        }
        CGImageDestinationAddImage(destination, image, frameProperties)
        added += 1
    }

    /// Writes the file out. Fewer frames than declared is tolerated by ImageIO (it writes what it
    /// was given) and is not treated as an error here — RenderJob only reaches finish() after the
    /// full loop, and a cancelled run goes through cancel() instead.
    func finish() throws {
        guard let destination else {
            throw ExporterError.writeFailed("cancelled")
        }
        // Cleared before finalizing, not after: whichever way the next line goes the destination
        // is spent, and clearing it in only the success branch would leave a second finish() free
        // to finalize the same destination twice.
        self.destination = nil
        guard CGImageDestinationFinalize(destination) else {
            try? FileManager.default.removeItem(at: outputURL)   // trap ③
            throw ExporterError.writeFailed("finalize \(outputURL.lastPathComponent)")
        }
    }

    /// Cancellation — release the destination and **delete the partial file** (trap ③). Idempotent
    /// and safe in a terminal state, which is the contract RenderJob leans on: it calls cancel()
    /// on the render-failure, encode-failure, user-cancellation, and finalize-failure paths, and
    /// the last two can already have thrown here.
    ///
    /// Note that cancel() after a *successful* finish() deletes the finished file. That is exactly
    /// what VideoExporter.cancel does too (it is pinned by
    /// testExporterCancelIdempotentAndRemovesPartialFile), and RenderJob never does it — a
    /// successful finish goes straight to completion. Keeping the two sinks behaviourally
    /// identical is worth more than defending against a call that would be a caller bug in either
    /// implementation.
    func cancel() {
        destination = nil
        try? FileManager.default.removeItem(at: outputURL)
    }

    // MARK: Internal

    /// The one exit used by every throwing failure inside append: drop the destination and delete
    /// the output before handing the error up (trap ③). Doing the cleanup here rather than relying
    /// on the caller's `catch { sink.cancel() }` is the point — a sink that only cleans up when its
    /// caller remembers to is the precise shape of the bug being guarded against.
    private func discarding(_ message: String) -> ExporterError {
        cancel()
        return ExporterError.writeFailed(message)
    }

    /// One frame's RGBA8 → an **independent** CGImage (trap ②) with the overlay composited in
    /// (trap ①).
    ///
    /// `data: nil` is the load-bearing choice: CoreGraphics owns that memory, so the makeImage()
    /// snapshot can retain it (copy-on-write) and stays valid long after this function returns.
    /// Handing CGContext the caller's `rgba` pointer instead would make correctness depend on CG
    /// noticing writes it cannot see — the caller repaints that buffer directly for the next frame,
    /// bypassing the context entirely, and CG's copy-on-write has no way to fire.
    ///
    /// alphaNoneSkipLast, not premultipliedLast: `sx_render_session_frame` fixes alpha at 255 (its
    /// C header contract), and GIF transparency is 1-bit — a pixel that did end up transparent
    /// would show whatever the viewer paints behind the animation instead of image content
    /// (TimelapseEncoder.resize documents the same trap from the letterbox-bar side). Declaring the
    /// canvas opaque removes the possibility outright, and it matches the video path, where the
    /// BGRA→YUV conversion discards alpha anyway. A translucent overlay still blends correctly —
    /// it is composited *onto* this opaque canvas, not carried as alpha.
    private static func makeFrameImage(rgba: UnsafePointer<UInt8>, width: Int, height: Int,
                                       compositor: OverlayCompositor?) -> CGImage? {
        guard let ctx = CGContext(data: nil, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)
                                      ?? CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
              let base = ctx.data else { return nil }
        // Copied row by row through ctx.bytesPerRow: with bytesPerRow 0 CoreGraphics picks its own
        // (padded) stride, and one flat memcpy of width·height·4 would shear the image.
        let srcStride = width * 4
        let dstStride = ctx.bytesPerRow
        for y in 0..<height {
            memcpy(base.advanced(by: y * dstStride), rgba.advanced(by: y * srcStride), srcStride)
        }
        // Trap ① — without this line the export succeeds and the watermark is simply gone.
        // outputHeight is what flips OverlayPlacement's top-left rect into CG's bottom-left
        // coordinates, so it must be the frame height, not the overlay's.
        compositor?.draw(in: ctx, outputHeight: height)
        return ctx.makeImage()
    }
}
