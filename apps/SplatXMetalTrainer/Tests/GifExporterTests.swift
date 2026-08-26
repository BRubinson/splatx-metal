// Cinema GIF export (2026-08-05 spec §5.6): real ImageIO encodes of synthetic RGBA frames, read
// back with CGImageSource. No GPU and no SwiftUI is involved — this is file I/O plus a system
// framework — so it is exercised for real rather than mocked, the same policy as
// TimelapseEncoderTests and VideoExportTests.testExporterCancelIdempotentAndRemovesPartialFile.
//
// Each test here exists because the corresponding mistake **compiles and produces a plausible
// file**, which is the whole hazard spec §5.6 enumerates:
// - a lazily referenced pixel buffer → every frame comes out the last colour
// - a dropped compositor → a silently watermark-less GIF
// - a missing cleanup → a broken file left beside a failure message
// - a missing loop/delay property → a GIF that plays once, or at the wrong speed
//
// Everything is written under FileManager.temporaryDirectory and removed in tearDown — the TCC
// rule (the real home, Desktop, and Downloads are off limits to tests).
import XCTest
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
@testable import SplatXMetalTrainer

final class GifExporterTests: XCTestCase {

    private var scratch: URL!

    override func setUpWithError() throws {
        scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("sx-gif-export-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let scratch { try? FileManager.default.removeItem(at: scratch) }
        scratch = nil
    }

    private func output(_ name: String = "out.gif") -> URL {
        scratch.appendingPathComponent(name)
    }

    // MARK: - Helpers

    /// Paints a solid colour over a densely packed RGBA8 buffer (alpha 255 — the
    /// sx_render_session_frame contract).
    private func paint(_ buffer: UnsafeMutablePointer<UInt8>, count: Int,
                       _ rgb: (UInt8, UInt8, UInt8)) {
        for p in stride(from: 0, to: count, by: 4) {
            buffer[p] = rgb.0
            buffer[p + 1] = rgb.1
            buffer[p + 2] = rgb.2
            buffer[p + 3] = 255
        }
    }

    /// Appends one solid-colour frame from a freshly allocated buffer (for the tests that do not
    /// care about buffer reuse).
    private func appendSolid(_ exporter: GifExporter, index: Int, width: Int, height: Int,
                             _ rgb: (UInt8, UInt8, UInt8),
                             compositor: OverlayCompositor? = nil) throws {
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        rgba.withUnsafeMutableBufferPointer { paint($0.baseAddress!, count: $0.count, rgb) }
        try rgba.withUnsafeBufferPointer {
            try exporter.append(rgba: $0.baseAddress!, frameIndex: index, compositor: compositor)
        }
    }

    private func decodeFrames(at url: URL) throws -> [CGImage] {
        let source = try XCTUnwrap(CGImageSourceCreateWithURL(url as CFURL, nil))
        return try (0..<CGImageSourceGetCount(source)).map {
            try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, $0, nil))
        }
    }

    /// The image as densely packed RGBA8, memory row 0 = the topmost scanline.
    private func rgbaBytes(of image: CGImage) throws -> [UInt8] {
        var buffer = [UInt8](repeating: 0, count: image.width * image.height * 4)
        let ok = buffer.withUnsafeMutableBytes { raw -> Bool in
            guard let base = raw.baseAddress,
                  let ctx = CGContext(data: base, width: image.width, height: image.height,
                                      bitsPerComponent: 8, bytesPerRow: image.width * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)
                                          ?? CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return false }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
            return true
        }
        guard ok else { throw XCTSkip("could not build an RGBA context") }
        return buffer
    }

    private func pixel(_ bytes: [UInt8], width: Int, x: Int, y: Int) -> (r: Int, g: Int, b: Int) {
        let i = (y * width + x) * 4
        return (Int(bytes[i]), Int(bytes[i + 1]), Int(bytes[i + 2]))
    }

    /// GIF is palettized, so an exact match is not guaranteed; a solid colour survives the palette
    /// build with room to spare, and the tolerance only has to be tight enough to tell the test's
    /// colours apart (they are ~200 apart per channel).
    private func assertColor(_ px: (r: Int, g: Int, b: Int), _ rgb: (UInt8, UInt8, UInt8),
                             tolerance: Int = 12, _ label: String,
                             file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertLessThanOrEqual(abs(px.r - Int(rgb.0)), tolerance, "\(label) R", file: file, line: line)
        XCTAssertLessThanOrEqual(abs(px.g - Int(rgb.1)), tolerance, "\(label) G", file: file, line: line)
        XCTAssertLessThanOrEqual(abs(px.b - Int(rgb.2)), tolerance, "\(label) B", file: file, line: line)
    }

    /// A solid-colour CGImage built by **writing bytes**, not by filling with a CGColor.
    /// `CGColor(red:green:blue:alpha:)` makes a *Generic RGB* colour, so filling an sRGB context
    /// with it silently colour-converts: measured here, (20, 20, 220) came back as G = 52 and made
    /// a correct compositor look like a compositing bug. Writing the bytes keeps the overlay in the
    /// same encoding as the frame, which is what makes an exact-colour assertion meaningful at all.
    private func solidImage(width: Int, height: Int,
                            _ rgb: (UInt8, UInt8, UInt8)) throws -> CGImage {
        let ctx = try XCTUnwrap(CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        let base = try XCTUnwrap(ctx.data).assumingMemoryBound(to: UInt8.self)
        for y in 0..<height {
            for x in 0..<width {
                let o = y * ctx.bytesPerRow + x * 4
                base[o] = rgb.0
                base[o + 1] = rgb.1
                base[o + 2] = rgb.2
                base[o + 3] = 255
            }
        }
        return try XCTUnwrap(ctx.makeImage())
    }

    // MARK: - Round trip

    /// The baseline: three frames in, three frames out, at the configured size.
    func testThreeFrameGifRoundTrip() throws {
        let url = output()
        let exporter = try GifExporter(
            outputURL: url, config: .init(width: 16, height: 12, fps: 24, frameCount: 3))
        try appendSolid(exporter, index: 0, width: 16, height: 12, (220, 20, 20))
        try appendSolid(exporter, index: 1, width: 16, height: 12, (20, 200, 20))
        try appendSolid(exporter, index: 2, width: 16, height: 12, (20, 20, 220))
        try exporter.finish()

        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        let frames = try decodeFrames(at: url)
        XCTAssertEqual(frames.count, 3)
        for frame in frames {
            XCTAssertEqual(frame.width, 16)
            XCTAssertEqual(frame.height, 12)
        }
    }

    /// Trap ④ + §5.7: loop 0 (forever) at the file level, and **both** delay keys per frame.
    /// fps 25 → 0.04 s, which is exact in GIF's 1/100 s unit, so a miss here is a missing key
    /// rather than a rounding argument.
    func testLoopsForeverAndWritesBothDelayKeys() throws {
        let url = output()
        let exporter = try GifExporter(
            outputURL: url, config: .init(width: 16, height: 12, fps: 25, frameCount: 2))
        try appendSolid(exporter, index: 0, width: 16, height: 12, (200, 200, 200))
        try appendSolid(exporter, index: 1, width: 16, height: 12, (40, 40, 40))
        try exporter.finish()

        let source = try XCTUnwrap(CGImageSourceCreateWithURL(url as CFURL, nil))
        let container = CGImageSourceCopyProperties(source, nil) as? [CFString: Any]
        let gif = container?[kCGImagePropertyGIFDictionary] as? [CFString: Any]
        XCTAssertEqual(gif?[kCGImagePropertyGIFLoopCount] as? Int, 0)   // 0 = forever

        let frame = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let frameGIF = frame?[kCGImagePropertyGIFDictionary] as? [CFString: Any]
        XCTAssertEqual(frameGIF?[kCGImagePropertyGIFUnclampedDelayTime] as? Double ?? 0,
                       0.04, accuracy: 0.005)
        // The clamped key is what strict/legacy readers use — it has to be present. Only presence
        // is asserted *here* because this test's subject is "both keys were written at all"; what
        // value either key comes back as for a rate that is **not** exact is a separate question,
        // and testGifDelayQuantisationForEveryRateChoice below answers it by measurement.
        XCTAssertNotNil(frameGIF?[kCGImagePropertyGIFDelayTime])
    }

    /// **Measured, not derived** (2026-08-05 spec §3.2). Adding 15 to `fpsChoices` raised the
    /// question of how `1/15 = 6.67cs` lands on GIF's 1/100 s grid — 6 or 7 — and nothing in this
    /// app's source answers it: the delay is handed to ImageIO as a `Double` and ImageIO owns the
    /// quantisation. It was therefore measured by writing a real GIF per rate and reading the field
    /// back, and the results are pinned here so the answer stops being folklore.
    ///
    /// Measured on macOS 26.5.2 (build 25F84), 2026-08-05 — ImageIO rounds to the **nearest**
    /// centisecond:
    ///
    /// | rate | exact  | written | effective |
    /// |------|--------|---------|-----------|
    /// | 15   | 6.67cs | 7cs     | 14.3 fps (−4.8%) |
    /// | 24   | 4.17cs | 4cs     | 25.0 fps (+4.2%) |
    /// | 30   | 3.33cs | 3cs     | 33.3 fps (+11.1%) |
    /// | 60   | 1.67cs | 2cs     | 50.0 fps (−16.7%) |
    ///
    /// So **15 introduces no new kind of inexactness and is the least inexact of the four** —
    /// only 10/20/25/50 divide the grid exactly (which is why the round-trip test above picks 25).
    ///
    /// The second measured fact is that `UnclampedDelayTime` reads back **equal to** DelayTime, not
    /// as the unrounded number that was passed in. That is the format, not a writer bug: a GIF
    /// Graphic Control Extension carries exactly one 16-bit centisecond field per frame, so the two
    /// property keys are two readings of one stored value and neither can preserve 6.67. Writing
    /// both still matters for what the keys *mean to readers* (see GifExporter.init), but no caller
    /// should expect sub-centisecond precision to survive the file.
    ///
    /// Driven off `fpsChoices` on purpose: a rate added without measuring it fails on the table
    /// coverage assertion rather than passing silently.
    func testGifDelayQuantisationForEveryRateChoice() throws {
        let measured: [Int: Double] = [15: 0.07, 24: 0.04, 30: 0.03, 60: 0.02]
        XCTAssertEqual(Set(measured.keys), Set(VideoExportSettings.fpsChoices),
                       "every export rate needs a measured delay — measure the new one, do not "
                       + "compute it")

        for fps in VideoExportSettings.fpsChoices {
            let expected = try XCTUnwrap(measured[fps])
            let url = output("rate-\(fps).gif")
            let exporter = try GifExporter(
                outputURL: url, config: .init(width: 16, height: 12, fps: fps, frameCount: 2))
            try appendSolid(exporter, index: 0, width: 16, height: 12, (200, 200, 200))
            try appendSolid(exporter, index: 1, width: 16, height: 12, (40, 40, 40))
            try exporter.finish()

            let source = try XCTUnwrap(CGImageSourceCreateWithURL(url as CFURL, nil))
            XCTAssertEqual(CGImageSourceGetCount(source), 2, "\(fps) fps")
            // Both frames, because the delay is per-frame in GIF: a writer that stamped only the
            // first would give a file whose first frame times correctly and whose rest do not.
            for index in 0..<2 {
                let props = CGImageSourceCopyPropertiesAtIndex(source, index, nil)
                    as? [CFString: Any]
                let gif = try XCTUnwrap(props?[kCGImagePropertyGIFDictionary] as? [CFString: Any],
                                        "\(fps) fps frame \(index)")
                // 0.0005 discriminates adjacent centisecond steps (0.06 from 0.07) while
                // tolerating the double round-trip through CFNumber.
                XCTAssertEqual(gif[kCGImagePropertyGIFDelayTime] as? Double ?? -1,
                               expected, accuracy: 0.0005,
                               "\(fps) fps frame \(index) clamped delay")
                XCTAssertEqual(gif[kCGImagePropertyGIFUnclampedDelayTime] as? Double ?? -1,
                               expected, accuracy: 0.0005,
                               "\(fps) fps frame \(index) unclamped delay — one stored field, so "
                               + "this key cannot carry more precision than the clamped one")
            }
        }
    }

    /// The consequence of the measurement above for the rate this round added, stated in the terms
    /// the user cares about: a 15fps GIF plays at 14.3fps, i.e. a 10-second animation runs ~0.5s
    /// long. That is the largest deviation 15 causes and it is smaller than what 30 and 60 already
    /// cause, so it is a fact to record rather than a reason for a caption — `gifFpsNote` stays
    /// silent (CinemaInspectorAffordanceTests pins that side).
    func testFifteenFpsGifPlaysSlightlySlowNotClamped() throws {
        let url = output("fifteen.gif")
        let exporter = try GifExporter(
            outputURL: url, config: .init(width: 16, height: 12, fps: 15, frameCount: 1))
        try appendSolid(exporter, index: 0, width: 16, height: 12, (120, 120, 120))
        try exporter.finish()

        let source = try XCTUnwrap(CGImageSourceCreateWithURL(url as CFURL, nil))
        let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let gif = try XCTUnwrap(props?[kCGImagePropertyGIFDictionary] as? [CFString: Any])
        let delay = try XCTUnwrap(gif[kCGImagePropertyGIFDelayTime] as? Double)

        // Comfortably above the 0.02s viewer clamp that gifFpsNote warns about — the whole point
        // of offering 15 for GIF is that it is on the safe side of that threshold.
        XCTAssertGreaterThan(delay, 0.02)
        XCTAssertEqual(1.0 / delay, 14.29, accuracy: 0.01, "effective playback rate")
    }

    // MARK: - Trap ② (pixel copying)

    /// The regression this whole class was written around. **One** allocation is repainted for
    /// every frame — exactly what RenderJob does with its single `rgba` array — so a sink that
    /// wrapped the caller's pointer in a CGDataProvider would emit three copies of the last
    /// colour instead of red/green/blue.
    func testFramePixelsAreCopiedNotReferenced() throws {
        let w = 16, h = 12, byteCount = w * h * 4
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: byteCount)
        defer { buffer.deallocate() }
        let colors: [(UInt8, UInt8, UInt8)] = [(220, 20, 20), (20, 200, 20), (20, 20, 220)]

        let url = output()
        let exporter = try GifExporter(
            outputURL: url, config: .init(width: w, height: h, fps: 24, frameCount: colors.count))
        for (i, rgb) in colors.enumerated() {
            paint(buffer, count: byteCount, rgb)
            try exporter.append(rgba: UnsafePointer(buffer), frameIndex: i, compositor: nil)
        }
        // The buffer is repainted once more **after** the last append and before finalize — a lazy
        // reference would be resolved here, at CGImageDestinationFinalize, and turn every frame
        // this colour.
        paint(buffer, count: byteCount, (0, 0, 0))
        try exporter.finish()

        let frames = try decodeFrames(at: url)
        XCTAssertEqual(frames.count, colors.count)
        for (i, rgb) in colors.enumerated() {
            let bytes = try rgbaBytes(of: frames[i])
            assertColor(pixel(bytes, width: w, x: w / 2, y: h / 2), rgb, "frame \(i)")
        }
    }

    // MARK: - Trap ① (overlay compositing)

    /// OverlayCompositor is a struct with a concrete `draw`, so there is nothing to subclass into a
    /// call-spy — and the observable effect is the stronger assertion anyway: it proves the draw
    /// happened *and* that it landed in the right place (the top-left rect, i.e. that the
    /// top-left→CG bottom-left flip was applied to the frame height).
    func testOverlayCompositorIsDrawnIntoTheFrame() throws {
        let w = 32, h = 24
        let background: (UInt8, UInt8, UInt8) = (220, 20, 20)
        let mark: (UInt8, UInt8, UInt8) = (20, 20, 220)
        // The overlay image is exactly the size of its rect, so no interpolation blurs the
        // sampled pixels. The rect is in OverlayPlacement's top-left origin convention.
        let compositor = OverlayCompositor(image: try solidImage(width: 16, height: 12, mark),
                                           rect: CGRect(x: 0, y: 0, width: 16, height: 12),
                                           opacity: 1)

        let url = output()
        let exporter = try GifExporter(
            outputURL: url, config: .init(width: w, height: h, fps: 24, frameCount: 1))
        try appendSolid(exporter, index: 0, width: w, height: h, background,
                        compositor: compositor)
        try exporter.finish()

        let frames = try decodeFrames(at: url)
        XCTAssertEqual(frames.count, 1)
        let bytes = try rgbaBytes(of: frames[0])
        assertColor(pixel(bytes, width: w, x: 4, y: 3), mark, "inside the overlay rect")
        assertColor(pixel(bytes, width: w, x: 28, y: 20), background, "outside the overlay rect")
    }

    /// The counterpart: no compositor → the frame is untouched. Without it, "the overlay pixels
    /// are there" could just mean the whole frame was painted blue by something else.
    func testNoCompositorLeavesTheFrameUntouched() throws {
        let w = 32, h = 24
        let background: (UInt8, UInt8, UInt8) = (220, 20, 20)
        let url = output()
        let exporter = try GifExporter(
            outputURL: url, config: .init(width: w, height: h, fps: 24, frameCount: 1))
        try appendSolid(exporter, index: 0, width: w, height: h, background)
        try exporter.finish()

        let bytes = try rgbaBytes(of: try XCTUnwrap(decodeFrames(at: url).first))
        assertColor(pixel(bytes, width: w, x: 4, y: 3), background, "top-left")
        assertColor(pixel(bytes, width: w, x: 28, y: 20), background, "bottom-right")
    }

    // MARK: - Trap ③ (no partial file survives)

    /// cancel() right after construction — nothing is left behind, and calling it twice is safe
    /// (RenderJob's failure paths can reach cancel() after this class has already cleaned up).
    func testCancelDeletesTheOutputAndIsIdempotent() throws {
        let url = output()
        let exporter = try GifExporter(
            outputURL: url, config: .init(width: 16, height: 12, fps: 24, frameCount: 3))
        exporter.cancel()
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
        exporter.cancel()
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
    }

    /// Cancelling partway through — the user's Cancel Render, the case that actually happens.
    /// No file, and finish() afterwards refuses instead of writing a truncated animation.
    func testCancelMidwayLeavesNoPartialFile() throws {
        let url = output()
        let exporter = try GifExporter(
            outputURL: url, config: .init(width: 16, height: 12, fps: 24, frameCount: 3))
        try appendSolid(exporter, index: 0, width: 16, height: 12, (220, 20, 20))
        try appendSolid(exporter, index: 1, width: 16, height: 12, (20, 200, 20))
        exporter.cancel()

        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
        XCTAssertThrowsError(try exporter.finish())
        XCTAssertThrowsError(try appendSolid(exporter, index: 2, width: 16, height: 12,
                                             (20, 20, 220)))
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
    }

    /// A rejected append cleans up on its own rather than trusting the caller's catch. Two ways in:
    /// more frames than the count declared to CGImageDestinationCreateWithURL, and an index that
    /// does not match the call order (GIF has no random access, so order *is* the timeline).
    func testRejectedAppendDiscardsTheOutput() throws {
        let overflowURL = output("overflow.gif")
        let overflow = try GifExporter(
            outputURL: overflowURL, config: .init(width: 16, height: 12, fps: 24, frameCount: 1))
        try appendSolid(overflow, index: 0, width: 16, height: 12, (220, 20, 20))
        XCTAssertThrowsError(try appendSolid(overflow, index: 1, width: 16, height: 12,
                                             (20, 200, 20)))
        XCTAssertFalse(FileManager.default.fileExists(atPath: overflowURL.path))

        let orderURL = output("order.gif")
        let ordered = try GifExporter(
            outputURL: orderURL, config: .init(width: 16, height: 12, fps: 24, frameCount: 3))
        XCTAssertThrowsError(try appendSolid(ordered, index: 1, width: 16, height: 12,
                                             (220, 20, 20)))
        XCTAssertFalse(FileManager.default.fileExists(atPath: orderURL.path))
    }

    /// A degenerate config fails at construction instead of producing an unusable file — the
    /// declared count reaches CGImageDestinationCreateWithURL directly, so 0 has to be refused
    /// before it gets there.
    func testEmptyConfigurationIsRejected() {
        XCTAssertThrowsError(try GifExporter(
            outputURL: output("zero-frames.gif"),
            config: .init(width: 16, height: 12, fps: 24, frameCount: 0)))
        XCTAssertThrowsError(try GifExporter(
            outputURL: output("zero-size.gif"),
            config: .init(width: 0, height: 12, fps: 24, frameCount: 3)))
    }

    // MARK: - FrameSink conformance

    /// The sink abstraction is the point of the protocol (spec §5.5): RenderJob keeps one frame
    /// loop and only the construction differs, so both conformances have to be real.
    func testBothExportersAreFrameSinks() throws {
        let url = output("sink.gif")
        let sink: FrameSink = try GifExporter(
            outputURL: url, config: .init(width: 16, height: 12, fps: 24, frameCount: 1))
        var rgba = [UInt8](repeating: 0, count: 16 * 12 * 4)
        rgba.withUnsafeMutableBufferPointer {
            paint($0.baseAddress!, count: $0.count, (10, 120, 200))
        }
        try rgba.withUnsafeBufferPointer {
            try sink.append(rgba: $0.baseAddress!, frameIndex: 0, compositor: nil)
        }
        try sink.finish()
        XCTAssertEqual(try decodeFrames(at: url).count, 1)

        let movURL = output("sink.mov")
        let video: FrameSink = try VideoExporter(
            outputURL: movURL,
            config: .init(codec: .h264, container: .mov, width: 64, height: 64, fps: 24))
        video.cancel()   // through the protocol — the cleanup contract, not a real encode
        XCTAssertFalse(FileManager.default.fileExists(atPath: movURL.path))
    }
}
