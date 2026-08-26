// 2026-08-04 spec §3.4/§3.5: the timelapse encoder — the frame listing convention, the output
// path convention, and real encodes through ImageIO (GIF) and the Cinema VideoExporter
// (MP4/MOV). Encoding is file I/O plus system frameworks, with no GPU and no SwiftUI, so it is
// exercised for real here rather than mocked (the same policy as
// VideoExportTests.testExporterCancelIdempotentAndRemovesPartialFile).
import XCTest
import AVFoundation
import ImageIO
import UniformTypeIdentifiers
@testable import SplatXMetalTrainer

final class TimelapseEncoderTests: XCTestCase {

    /// Writes one synthetic JPEG frame. The colour is a function of the index (R=160, G=80,
    /// B=index·30) so a decoded pixel identifies both "this is frame content" and which frame.
    private func writeFrame(into dir: URL, index: Int, width: Int, height: Int) {
        var bgra = [UInt8](repeating: 0, count: width * height * 4)
        for p in 0..<(width * height) {
            bgra[p * 4 + 0] = UInt8((index * 30) % 256)
            bgra[p * 4 + 1] = 80
            bgra[p * 4 + 2] = 160
            bgra[p * 4 + 3] = 255
        }
        TimelapseRecorder.writeFrame(
            bgra: bgra, width: width, height: height,
            url: dir.appendingPathComponent(TimelapseSettings.frameFileName(index: index)),
            label: nil)
    }

    /// A scratch folder under the temporary directory — never the real home (the TCC rule).
    private func makeScratch() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("sx-timelapse-enc-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// A scratch folder holding `count` real JPEG frames of the given size.
    private func makeFrames(count: Int, width: Int, height: Int) throws -> URL {
        let dir = try makeScratch()
        for i in 0..<count { writeFrame(into: dir, index: i, width: width, height: height) }
        return dir
    }

    // MARK: - Decoding helpers (for asserting on real encoder output)

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

    private func pixel(_ bytes: [UInt8], width: Int, x: Int, y: Int)
        -> (r: Int, g: Int, b: Int) {
        let i = (y * width + x) * 4
        return (Int(bytes[i]), Int(bytes[i + 1]), Int(bytes[i + 2]))
    }

    /// The exact number of encoded video frames — the video counterpart of
    /// CGImageSourceGetCount.
    ///
    /// Sums `CMSampleBufferGetNumSamples` rather than counting `copyNextSampleBuffer` calls.
    /// With `outputSettings: nil` the reader also hands back buffers that carry no frame at all,
    /// so counting calls **overcounts**: measured on an 8-frame encode, 12 buffers but 8 samples.
    /// Counting calls is the obvious-looking version and it is wrong.
    private func videoFrameCount(_ url: URL) async throws -> Int {
        let asset = AVURLAsset(url: url)
        // The await is hoisted out of XCTUnwrap — its argument is an autoclosure, which cannot
        // carry an async call.
        let tracks = try await asset.loadTracks(withMediaType: .video)
        let track = try XCTUnwrap(tracks.first)
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        reader.add(output)
        XCTAssertTrue(reader.startReading())
        var samples = 0
        while let buffer = output.copyNextSampleBuffer() {
            samples += CMSampleBufferGetNumSamples(buffer)
        }
        return samples
    }

    private func firstVideoImage(_ url: URL) async throws -> CGImage {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        return try await generator.image(at: .zero).image
    }

    // MARK: - Listing and paths

    func testFrameURLsAreSortedAndFiltered() throws {
        let dir = try makeFrames(count: 12, width: 32, height: 32)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data([0]).write(to: dir.appendingPathComponent("notes.txt"))
        try Data([0]).write(to: dir.appendingPathComponent("preview.png"))

        let urls = TimelapseEncoder.frameURLs(in: dir)
        XCTAssertEqual(urls.count, 12)
        // The full sequence comparison also pins the 9-before-10 ordering.
        XCTAssertEqual(urls.map(\.lastPathComponent),
                       (0..<12).map { TimelapseSettings.frameFileName(index: $0) })
    }

    /// A `frame-*.jpg` with a non-numeric stem is excluded rather than carried along. It has no
    /// place in capture order, and handing it to the decoder would fail the **whole** encode —
    /// dropping one stray file costs a frame, keeping it costs the run (FIX 4 / M3).
    func testFrameURLsDropsAnUnparseableName() throws {
        let dir = try makeFrames(count: 3, width: 32, height: 32)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data("not a jpeg".utf8).write(to: dir.appendingPathComponent("frame-abc.jpg"))
        try Data("not a jpeg".utf8).write(to: dir.appendingPathComponent("frame-.jpg"))

        XCTAssertEqual(TimelapseEncoder.frameURLs(in: dir).map(\.lastPathComponent),
                       (0..<3).map { TimelapseSettings.frameFileName(index: $0) })
    }

    /// …and because it is excluded, the encode still succeeds instead of dying on it.
    func testEncodeSurvivesAnUnparseableName() throws {
        let dir = try makeFrames(count: 3, width: 32, height: 32)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data("not a jpeg".utf8).write(to: dir.appendingPathComponent("frame-abc.jpg"))
        let out = TimelapseEncoder.outputURL(runDirectory: dir, format: .gif)
        let result = try TimelapseEncoder.encode(
            frames: TimelapseEncoder.frameURLs(in: dir), to: out, format: .gif,
            fps: 15, width: 32, height: 32)
        XCTAssertEqual(result.frameCount, 3)
        XCTAssertTrue(FileManager.default.fileExists(atPath: out.path))
    }

    func testFrameURLsOfAMissingDirectoryIsEmpty() {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("sx-timelapse-missing-\(UUID().uuidString)")
        XCTAssertTrue(TimelapseEncoder.frameURLs(in: missing).isEmpty)
    }

    /// A capture whose render failed consumes its index (Task 10), so the sequence has holes.
    /// They are simply absent — the survivors still come out in capture order.
    func testFrameURLsKeepCaptureOrderAcrossGaps() throws {
        let dir = try makeFrames(count: 12, width: 32, height: 32)
        defer { try? FileManager.default.removeItem(at: dir) }
        for missing in [1, 4, 5, 10] {
            try FileManager.default.removeItem(
                at: dir.appendingPathComponent(TimelapseSettings.frameFileName(index: missing)))
        }
        XCTAssertEqual(TimelapseEncoder.frameURLs(in: dir).map(\.lastPathComponent),
                       [0, 2, 3, 6, 7, 8, 9, 11].map(TimelapseSettings.frameFileName(index:)))
    }

    /// The five-digit padding runs out past 99999, where a plain string sort would put
    /// frame-100000 before frame-99999. The ordering is numeric, so it survives that.
    func testFrameURLsOrderIsNumericBeyondThePadding() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("sx-timelapse-wide-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        for index in [99999, 100000, 100001] {
            try Data([0]).write(
                to: dir.appendingPathComponent(TimelapseSettings.frameFileName(index: index)))
        }
        XCTAssertEqual(TimelapseEncoder.frameURLs(in: dir).map(\.lastPathComponent),
                       ["frame-99999.jpg", "frame-100000.jpg", "frame-100001.jpg"])
    }

    /// The final file is output.ply's sibling, and the run record stores a project-relative path
    /// (spec §3.4).
    func testOutputPathConventions() {
        let run = URL(fileURLWithPath: "/p/runs/20260804-120000", isDirectory: true)
        XCTAssertEqual(TimelapseEncoder.outputURL(runDirectory: run, format: .mp4).path,
                       "/p/runs/20260804-120000/timelapse.mp4")
        XCTAssertEqual(TimelapseEncoder.outputURL(runDirectory: run, format: .gif).path,
                       "/p/runs/20260804-120000/timelapse.gif")
        XCTAssertEqual(TimelapseEncoder.relativePath(runID: "20260804-120000", format: .mov),
                       "runs/20260804-120000/timelapse.mov")
    }

    // MARK: - Fewer than two frames is skipped (spec §3.5)

    func testEncodeRefusesFewerThanTwoFrames() throws {
        let dir = try makeFrames(count: 1, width: 32, height: 32)
        defer { try? FileManager.default.removeItem(at: dir) }
        let out = dir.appendingPathComponent("timelapse.gif")
        XCTAssertThrowsError(try TimelapseEncoder.encode(
            frames: TimelapseEncoder.frameURLs(in: dir), to: out, format: .gif,
            fps: 15, width: 32, height: 32)) { error in
            guard case TimelapseEncoder.EncodeError.notEnoughFrames(let n) = error else {
                return XCTFail("unexpected error \(error)")
            }
            XCTAssertEqual(n, 1)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: out.path))
    }

    // MARK: - GIF (ImageIO)

    func testEncodeGIFWritesEveryFrame() throws {
        let dir = try makeFrames(count: 4, width: 64, height: 48)
        defer { try? FileManager.default.removeItem(at: dir) }
        let out = TimelapseEncoder.outputURL(runDirectory: dir, format: .gif)
        let result = try TimelapseEncoder.encode(
            frames: TimelapseEncoder.frameURLs(in: dir), to: out, format: .gif,
            fps: 15, width: 64, height: 48)
        XCTAssertEqual(result.frameCount, 4)
        XCTAssertNil(result.note)                 // nothing was reduced
        XCTAssertTrue(FileManager.default.fileExists(atPath: out.path))
        let src = try XCTUnwrap(CGImageSourceCreateWithURL(out as CFURL, nil))
        XCTAssertEqual(CGImageSourceGetCount(src), 4)
        XCTAssertEqual(CGImageSourceGetType(src) as String?, UTType.gif.identifier)
    }

    func testEncodeGIFReportsTheReduction() throws {
        let dir = try makeFrames(count: 4, width: 64, height: 48)
        defer { try? FileManager.default.removeItem(at: dir) }
        let out = TimelapseEncoder.outputURL(runDirectory: dir, format: .gif)
        // Declared as 1920×1080 so the 960 clamp kicks in (the frames themselves are redrawn to
        // the plan size).
        let result = try TimelapseEncoder.encode(
            frames: TimelapseEncoder.frameURLs(in: dir), to: out, format: .gif,
            fps: 15, width: 1920, height: 1080)
        XCTAssertEqual(result.note, "GIF reduced: size 1920×1080→960×540")
        let src = try XCTUnwrap(CGImageSourceCreateWithURL(out as CFURL, nil))
        let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(src, 0, nil))
        XCTAssertEqual(image.width, 960)
        XCTAssertEqual(image.height, 540)

        // The canvas is 16:9 but the frames are 4:3, so the frame is fitted (11.25×) to 720×540
        // and centred at x = 120 — black bars either side, **not** a stretched frame. Asserting
        // only the canvas size would pass just as happily on a distorted frame (FIX 4 / M1), so
        // the bar and the content are sampled.
        let bytes = try rgbaBytes(of: image)
        let bar = pixel(bytes, width: 960, x: 20, y: 270)
        XCTAssertLessThan(max(bar.r, bar.g, bar.b), 24, "the letterbox bar should be black")
        let content = pixel(bytes, width: 960, x: 480, y: 270)
        XCTAssertEqual(Double(content.r), 160, accuracy: 24)
        XCTAssertEqual(Double(content.g), 80, accuracy: 24)
    }

    /// The fit arithmetic itself, pinned as pure math — reaching it through an encode needs a
    /// size mismatch, which the rest of the feature works to make impossible (FIX 4 / I-1).
    func testFitRectPreservesAspectAndCentres() {
        // The common case: an exact match is the full rect, so nothing is scaled at all.
        XCTAssertEqual(TimelapseEncoder.fitRect(imageWidth: 128, imageHeight: 128,
                                                width: 128, height: 128),
                       CGRect(x: 0, y: 0, width: 128, height: 128))
        // Wider-than-canvas source → bars top and bottom.
        XCTAssertEqual(TimelapseEncoder.fitRect(imageWidth: 64, imageHeight: 32,
                                                width: 128, height: 128),
                       CGRect(x: 0, y: 32, width: 128, height: 64))
        // Taller-than-canvas source → bars left and right (the GIF 4:3 → 16:9 case).
        XCTAssertEqual(TimelapseEncoder.fitRect(imageWidth: 64, imageHeight: 48,
                                                width: 960, height: 540),
                       CGRect(x: 120, y: 0, width: 720, height: 540))
        // A degenerate source cannot be scaled — fall back to the full rect rather than
        // dividing by zero.
        XCTAssertEqual(TimelapseEncoder.fitRect(imageWidth: 0, imageHeight: 0,
                                                width: 128, height: 128),
                       CGRect(x: 0, y: 0, width: 128, height: 128))
    }

    /// The GIF loops forever and each frame carries the requested delay — otherwise a viewer
    /// plays it once, at whatever rate it feels like.
    func testEncodeGIFSetsTheLoopAndDelay() throws {
        let dir = try makeFrames(count: 3, width: 32, height: 32)
        defer { try? FileManager.default.removeItem(at: dir) }
        let out = TimelapseEncoder.outputURL(runDirectory: dir, format: .gif)
        _ = try TimelapseEncoder.encode(
            frames: TimelapseEncoder.frameURLs(in: dir), to: out, format: .gif,
            fps: 10, width: 32, height: 32)
        let src = try XCTUnwrap(CGImageSourceCreateWithURL(out as CFURL, nil))
        let container = CGImageSourceCopyProperties(src, nil) as? [CFString: Any]
        let gif = container?[kCGImagePropertyGIFDictionary] as? [CFString: Any]
        XCTAssertEqual(gif?[kCGImagePropertyGIFLoopCount] as? Int, 0)   // 0 = forever
        let frame = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any]
        let frameGIF = frame?[kCGImagePropertyGIFDictionary] as? [CFString: Any]
        XCTAssertEqual(frameGIF?[kCGImagePropertyGIFUnclampedDelayTime] as? Double ?? 0,
                       0.1, accuracy: 0.001)
    }

    // MARK: - Video (the Cinema VideoExporter, reused unmodified)

    /// Every captured frame reaches the file. Asserting only "the file is non-empty" would pass
    /// on an encode that dropped half its frames — the exact sample count and the duration are
    /// what close that (FIX 4 / M5), matching what CGImageSourceGetCount pins on the GIF side.
    func testEncodeVideoProducesAPlayableFile() async throws {
        let dir = try makeFrames(count: 8, width: 128, height: 128)
        defer { try? FileManager.default.removeItem(at: dir) }
        let out = TimelapseEncoder.outputURL(runDirectory: dir, format: .mov)
        let result = try TimelapseEncoder.encode(
            frames: TimelapseEncoder.frameURLs(in: dir), to: out, format: .mov,
            fps: 15, width: 128, height: 128, codec: .h264)
        XCTAssertEqual(result.frameCount, 8)
        XCTAssertNil(result.note)                 // video never reduces
        XCTAssertTrue(FileManager.default.fileExists(atPath: out.path))
        let size = try FileManager.default.attributesOfItem(atPath: out.path)[.size] as? Int
        XCTAssertGreaterThan(size ?? 0, 0)

        let encoded = try await videoFrameCount(out)
        XCTAssertEqual(encoded, 8, "every captured frame must reach the file")
        let duration = try await AVURLAsset(url: out).load(.duration)
        XCTAssertEqual(duration.seconds, 8.0 / 15.0, accuracy: 0.02)
    }

    /// A frame whose size differs from the encode size is letterboxed, not stretched. This
    /// should be unreachable (changing the resolution restarts the sequence), but it was the one
    /// branch added beyond the brief and it had no coverage at all (FIX 4 / I-1).
    func testEncodeVideoLetterboxesAMismatchedFrame() async throws {
        let dir = try makeScratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        writeFrame(into: dir, index: 0, width: 64, height: 32)      // 2:1 — the odd one out
        writeFrame(into: dir, index: 1, width: 128, height: 128)
        writeFrame(into: dir, index: 2, width: 128, height: 128)

        let out = TimelapseEncoder.outputURL(runDirectory: dir, format: .mov)
        let result = try TimelapseEncoder.encode(
            frames: TimelapseEncoder.frameURLs(in: dir), to: out, format: .mov,
            fps: 15, width: 128, height: 128, codec: .h264)
        XCTAssertEqual(result.frameCount, 3)      // the mismatch does not drop the frame
        let encoded = try await videoFrameCount(out)
        XCTAssertEqual(encoded, 3)

        // 64×32 fitted into 128×128 is 128×64 centred at y = 32: rows 0–31 and 96–127 are bars.
        let bytes = try rgbaBytes(of: try await firstVideoImage(out))
        let bar = pixel(bytes, width: 128, x: 64, y: 8)
        XCTAssertLessThan(max(bar.r, bar.g, bar.b), 48, "the letterbox bar should be black")
        let content = pixel(bytes, width: 128, x: 64, y: 64)
        XCTAssertGreaterThan(content.r, 100, "the frame itself should be in the middle band")
    }

    /// Cancellation deletes the partial file (the VideoExporter.cancel contract).
    func testEncodeCancelledRemovesThePartialFile() throws {
        let dir = try makeFrames(count: 3, width: 128, height: 128)
        defer { try? FileManager.default.removeItem(at: dir) }
        let out = TimelapseEncoder.outputURL(runDirectory: dir, format: .mov)
        XCTAssertThrowsError(try TimelapseEncoder.encode(
            frames: TimelapseEncoder.frameURLs(in: dir), to: out, format: .mov,
            fps: 15, width: 128, height: 128, codec: .h264,
            isCancelled: { true })) { error in
            guard case TimelapseEncoder.EncodeError.cancelled = error else {
                return XCTFail("unexpected error \(error)")
            }
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: out.path))
    }

    /// The GIF path polls the same flag and leaves no half-written file behind either.
    func testEncodeGIFCancelledRemovesThePartialFile() throws {
        let dir = try makeFrames(count: 3, width: 32, height: 32)
        defer { try? FileManager.default.removeItem(at: dir) }
        let out = TimelapseEncoder.outputURL(runDirectory: dir, format: .gif)
        XCTAssertThrowsError(try TimelapseEncoder.encode(
            frames: TimelapseEncoder.frameURLs(in: dir), to: out, format: .gif,
            fps: 15, width: 32, height: 32, isCancelled: { true })) { error in
            guard case TimelapseEncoder.EncodeError.cancelled = error else {
                return XCTFail("unexpected error \(error)")
            }
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: out.path))
    }

    /// A frame that is not a decodable image fails loudly rather than silently shortening the
    /// video, and the partial file goes with it.
    func testEncodeFailsOnAnUnreadableFrame() throws {
        let dir = try makeFrames(count: 3, width: 64, height: 64)
        defer { try? FileManager.default.removeItem(at: dir) }
        let broken = dir.appendingPathComponent(TimelapseSettings.frameFileName(index: 1))
        try Data("not a jpeg".utf8).write(to: broken)
        let out = TimelapseEncoder.outputURL(runDirectory: dir, format: .mov)
        XCTAssertThrowsError(try TimelapseEncoder.encode(
            frames: TimelapseEncoder.frameURLs(in: dir), to: out, format: .mov,
            fps: 15, width: 64, height: 64, codec: .h264)) { error in
            guard case TimelapseEncoder.EncodeError.decodeFailed(let name) = error else {
                return XCTFail("unexpected error \(error)")
            }
            XCTAssertEqual(name, "frame-00001.jpg")
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: out.path))
    }
}
