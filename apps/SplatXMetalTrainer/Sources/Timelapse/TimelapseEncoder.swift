// Training timelapse encoding (2026-08-04 spec §3.4/§3.5): the collected JPEG frames become
// runs/<stamp>/timelapse.{mp4,mov,gif}.
//
// - MP4/MOV **reuse Cinema's VideoExporter unmodified** — it was written not knowing where its
//   frames come from (init(outputURL:config:) → append(rgba:frameIndex:compositor:) → finish()),
//   so a JPEG sequence is as valid a source as sx_render_session was. Its thread contract comes
//   along unchanged: append/finish/cancel are called from **exactly one thread**, which is why
//   TrainViewModel runs this on a dedicated Thread (the same discipline as RenderJob).
// - GIF is written with ImageIO (kCGImagePropertyGIFLoopCount = 0) and auto-reduced by
//   TimelapseSettings.gifPlan. The reduction is reported in Result.note — spec §3.4 forbids
//   silent truncation.
// - Every failure path deletes the partial file (the cancel() contract), so a "failed" message
//   and a broken file never coexist.
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import os

enum TimelapseEncoder {
    /// Fewer than this many frames skips encoding entirely (spec §3.5).
    static let minimumFrames = 2

    private static let log = Logger(subsystem: "com.postmedia.splatx-metal-trainer",
                                    category: "timelapse")

    struct Result {
        let outputURL: URL
        /// The number of frames actually written (for GIF, after sampling).
        let frameCount: Int
        /// The GIF reduction note — nil for video and when nothing was reduced.
        let note: String?
    }

    enum EncodeError: LocalizedError {
        case notEnoughFrames(Int)
        case cancelled
        case decodeFailed(String)
        case destinationFailed(String)
        case video(String)

        var errorDescription: String? {
            switch self {
            case .notEnoughFrames(let n):
                "Timelapse needs at least \(minimumFrames) frames (captured \(n))."
            case .cancelled: "Timelapse encoding was cancelled."
            case .decodeFailed(let name): "Cannot read timelapse frame \(name)."
            case .destinationFailed(let name): "Cannot create \(name)."
            case .video(let why): "Timelapse video encoding failed: \(why)"
            }
        }
    }

    // MARK: Paths

    /// The stem TimelapseSettings.frameFileName writes (`frame-%05d.jpg`).
    private static let framePrefix = "frame-"

    /// The captured frames in capture order. A gap (a capture whose render failed consumed its
    /// index — Task 10) is simply absent, and the survivors keep their order.
    ///
    /// Sorted by the **parsed index, not the file name**: the five-digit padding makes the two
    /// agree only up to 99999, past which a string sort puts `frame-100000` before
    /// `frame-99999`. That count is out of reach in practice, but the parse costs nothing and
    /// removes the ceiling instead of documenting it.
    ///
    /// A `frame-*.jpg` whose stem is not a plain number is **dropped and logged**, not carried
    /// along (FIX 4 / M3). It cannot be placed in capture order (there is no index to order it
    /// by), and keeping it would hand an unknown file to the decoder — one such name that is not
    /// a real JPEG fails `decodeFailed` and takes the **whole** timelapse with it. Dropping one
    /// stray file costs one frame; keeping it costs the run.
    static func frameURLs(in directory: URL) -> [URL] {
        let items = (try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles])) ?? []
        var indexed: [(index: Int, url: URL)] = []
        var skipped: [String] = []
        for url in items where url.lastPathComponent.hasPrefix(framePrefix)
            && url.pathExtension.lowercased() == "jpg" {
            if let index = frameIndex(of: url) {
                indexed.append((index, url))
            } else {
                skipped.append(url.lastPathComponent)
            }
        }
        if !skipped.isEmpty {
            log.error("""
                timelapse: ignoring \(skipped.count, privacy: .public) frame file(s) with an \
                unparseable index (\(skipped.sorted().joined(separator: ", "), privacy: .public)) \
                — they are excluded from the encode
                """)
        }
        return indexed.sorted { $0.index < $1.index }.map(\.url)
    }

    /// The integer in `frame-<n>.jpg`, or nil when the stem is not a plain number.
    private static func frameIndex(of url: URL) -> Int? {
        Int(url.deletingPathExtension().lastPathComponent.dropFirst(framePrefix.count))
    }

    /// runs/<stamp>/timelapse.<ext> — output.ply's sibling.
    static func outputURL(runDirectory: URL, format: TimelapseFormat) -> URL {
        runDirectory.appendingPathComponent("timelapse.\(format.fileExtension)")
    }

    /// The project-relative path recorded in RunRecord.timelapseFile (immune to the project
    /// folder being moved or renamed — the same convention as outputFile).
    static func relativePath(runID: String, format: TimelapseFormat) -> String {
        "runs/\(runID)/timelapse.\(format.fileExtension)"
    }

    // MARK: Encoding

    /// Blocking. **Call it from a dedicated Thread** — VideoExporter's contract is one thread for
    /// append/finish/cancel, and finish() blocks until the writer completes.
    static func encode(frames: [URL], to outputURL: URL, format: TimelapseFormat,
                       fps: Int, width: Int, height: Int, codec: VideoCodec = .hevc,
                       isCancelled: @Sendable () -> Bool = { false }) throws -> Result {
        guard frames.count >= minimumFrames else {
            throw EncodeError.notEnoughFrames(frames.count)
        }
        guard let container = format.videoContainer else {
            return try encodeGIF(frames: frames, to: outputURL, fps: fps,
                                 width: width, height: height, isCancelled: isCancelled)
        }
        return try encodeVideo(frames: frames, to: outputURL, container: container,
                               fps: fps, width: width, height: height, codec: codec,
                               isCancelled: isCancelled)
    }

    private static func encodeVideo(frames: [URL], to outputURL: URL,
                                    container: VideoContainer, fps: Int,
                                    width: Int, height: Int, codec: VideoCodec,
                                    isCancelled: @Sendable () -> Bool) throws -> Result {
        let exporter = try VideoExporter(
            outputURL: outputURL,
            config: .init(codec: codec, container: container,
                          width: width, height: height, fps: fps))
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        for (i, url) in frames.enumerated() {
            // Cancellation is polled at the frame boundary, and cancel() is what deletes the
            // partial file — throwing without it would leave a truncated video on disk.
            if isCancelled() { exporter.cancel(); throw EncodeError.cancelled }
            guard let image = loadImage(url),
                  drawRGBA(image, into: &rgba, width: width, height: height) else {
                exporter.cancel()
                throw EncodeError.decodeFailed(url.lastPathComponent)
            }
            do {
                try rgba.withUnsafeBufferPointer { p in
                    try exporter.append(rgba: p.baseAddress!, frameIndex: i, compositor: nil)
                }
            } catch {
                exporter.cancel()
                throw EncodeError.video(error.localizedDescription)
            }
        }
        do {
            try exporter.finish()
        } catch {
            exporter.cancel()   // a finalize failure cleans up the partial file too
            throw EncodeError.video(error.localizedDescription)
        }
        return Result(outputURL: outputURL, frameCount: frames.count, note: nil)
    }

    private static func encodeGIF(frames: [URL], to outputURL: URL, fps: Int,
                                  width: Int, height: Int,
                                  isCancelled: @Sendable () -> Bool) throws -> Result {
        let plan = TimelapseSettings.gifPlan(frameCount: frames.count,
                                             width: width, height: height)
        try? FileManager.default.removeItem(at: outputURL)
        guard let destination = CGImageDestinationCreateWithURL(
            outputURL as CFURL, UTType.gif.identifier as CFString,
            plan.indices.count, nil) else {
            throw EncodeError.destinationFailed(outputURL.lastPathComponent)
        }
        CGImageDestinationSetProperties(destination, [
            kCGImagePropertyGIFDictionary: [kCGImagePropertyGIFLoopCount: 0],
        ] as CFDictionary)
        let delay = 1.0 / Double(max(fps, 1))
        let frameProperties = [
            kCGImagePropertyGIFDictionary: [
                kCGImagePropertyGIFDelayTime: delay,
                kCGImagePropertyGIFUnclampedDelayTime: delay,
            ],
        ] as CFDictionary

        for index in plan.indices {
            if isCancelled() {
                try? FileManager.default.removeItem(at: outputURL)
                throw EncodeError.cancelled
            }
            guard let image = loadImage(frames[index]),
                  let scaled = resize(image, width: plan.width, height: plan.height) else {
                try? FileManager.default.removeItem(at: outputURL)
                throw EncodeError.decodeFailed(frames[index].lastPathComponent)
            }
            CGImageDestinationAddImage(destination, scaled, frameProperties)
        }
        guard CGImageDestinationFinalize(destination) else {
            try? FileManager.default.removeItem(at: outputURL)
            throw EncodeError.destinationFailed(outputURL.lastPathComponent)
        }
        return Result(outputURL: outputURL, frameCount: plan.indices.count,
                      note: TimelapseSettings.gifReductionNote(frameCount: frames.count,
                                                               width: width, height: height))
    }

    // MARK: Pixel helpers

    private static func loadImage(_ url: URL) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(
            source, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
    }

    /// Draws one frame into a densely packed RGBA8 (sRGB) buffer of exactly width×height — the
    /// input VideoExporter.append expects (it does the RGBA→BGRA permute itself), and its size is
    /// fixed for the whole session because the writer was configured with it.
    ///
    /// A frame of a different size should not exist (changing the resolution restarts the frame
    /// sequence), so this is a defensive path — but **not a memory-safety one**, and an earlier
    /// version of this comment was wrong to imply otherwise (FIX 4 / I-1). The buffer is exactly
    /// `width*4 × height` and CGContext clips every draw to its own bounds, so an image of any
    /// size drawn at any rect is already safe; stretching to the full rect would be equally
    /// safe. Fitting with the aspect ratio preserved buys exactly one thing, and it is cosmetic:
    /// a stray odd-sized frame is letterboxed instead of geometrically distorted. The opaque
    /// black prefill is what fills the resulting bars.
    private static func drawRGBA(_ image: CGImage, into buffer: inout [UInt8],
                                 width: Int, height: Int) -> Bool {
        var ok = false
        buffer.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress,
                  let ctx = CGContext(data: base, width: width, height: height,
                                      bitsPerComponent: 8, bytesPerRow: width * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)
                                          ?? CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return }
            ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
            ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
            ctx.interpolationQuality = .high
            ctx.draw(image, in: fitRect(imageWidth: image.width, imageHeight: image.height,
                                        width: width, height: height))
            ok = true
        }
        return ok
    }

    /// The centred aspect-preserving rect for an image inside width×height. The common case (an
    /// exact match) returns the full rect, so no scaling happens at all.
    ///
    /// Internal rather than private so the fit arithmetic is pinned directly as pure math —
    /// reaching it through an encode requires a size mismatch, which is exactly the state the
    /// rest of the feature works to make impossible (FIX 4 / I-1).
    static func fitRect(imageWidth: Int, imageHeight: Int,
                        width: Int, height: Int) -> CGRect {
        guard imageWidth > 0, imageHeight > 0,
              imageWidth != width || imageHeight != height else {
            return CGRect(x: 0, y: 0, width: width, height: height)
        }
        let scale = min(Double(width) / Double(imageWidth),
                        Double(height) / Double(imageHeight))
        let w = Double(imageWidth) * scale, h = Double(imageHeight) * scale
        return CGRect(x: (Double(width) - w) / 2, y: (Double(height) - h) / 2,
                      width: w, height: h)
    }

    /// Redraws one frame at the GIF plan size, using the **same fit rule as the video path**
    /// (FIX 4 / M1 — it used to stretch to the full rect, so the two formats disagreed about what
    /// a mismatched frame looks like and only the GIF distorted it). The canvas is filled opaque
    /// black first: a fresh CGContext is zeroed, which under premultipliedLast is *transparent*
    /// black, and GIF's 1-bit transparency would turn those letterbox bars into whatever the
    /// viewer paints behind them.
    private static func resize(_ image: CGImage, width: Int, height: Int) -> CGImage? {
        guard image.width != width || image.height != height else { return image }
        guard let ctx = CGContext(data: nil, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)
                                      ?? CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        ctx.interpolationQuality = .high
        ctx.draw(image, in: fitRect(imageWidth: image.width, imageHeight: image.height,
                                    width: width, height: height))
        return ctx.makeImage()
    }
}
