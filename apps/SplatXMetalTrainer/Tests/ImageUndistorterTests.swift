// import-undistortion Task 2: tests for the resampler (`Sources/Import/ImageUndistorter.swift`).
//
// The strong test here is the synthetic round trip: a known pattern is warped through the
// *inverse* of a known distortion to synthesize a fake "as captured" photo, our rectification is
// applied to it, and the result is compared with the pattern it started from. Unlike the
// point-wise round trip in CameraDistortionTests — which inverts the same function it tests, and so
// passes for any invertible map — this one crosses the whole pipeline: the frame arithmetic, the
// pixel-center convention, the map, and the bilinear gather all have to agree with the geometry for
// the picture to come back. A wrong sign, a half-pixel shift or a swapped axis shows up as a
// blurred or doubled pattern, and the test carries its own control: the same comparison against the
// *unrectified* centre crop, which must be much worse.
//
// The pattern is a checkerboard with soft edges (a tanh-shaped square wave) rather than hard steps.
// Bilinear interpolation of a hard edge is off by up to half the step amplitude no matter how
// correct the geometry is, so a hard-edged board would measure the interpolator instead of the
// rectification. The soft edges keep interpolation error an order of magnitude below the error a
// geometry mistake produces, which is what makes the threshold meaningful.
//
// All files live under FileManager.default.temporaryDirectory and are removed in tearDown — the
// TCC rule (no ~/Desktop, no ~/Downloads, no real home).
import XCTest
import ImageIO
import UniformTypeIdentifiers
@testable import SplatXMetalTrainer

final class ImageUndistorterTests: XCTestCase {

    // MARK: - Fixtures

    /// The user's real camera (spec §1): OPENCV, 2160×3840, 110 images.
    private static let referenceParams: [Double] = [
        2630.110, 2633.084, 1080, 1920,
        0.043354, -0.079594, -0.005166, 0.000340,
    ]
    private static let referenceWidth = 2160
    private static let referenceHeight = 3840

    /// A small synthetic OPENCV camera with distortion strong enough to displace a corner by
    /// ~14 px — large compared with the 24 px pattern period, so a geometry error cannot hide.
    private static let syntheticWidth = 320
    private static let syntheticHeight = 240
    private static let syntheticParams: [Double] = [
        260, 260, 160, 120,
        -0.15, 0.05, 0.0012, -0.0009,
    ]

    private var workDir: URL!

    override func setUpWithError() throws {
        workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ImageUndistorterTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: workDir)
    }

    // MARK: - The map

    func testMapFrameMatchesUndistortedFrameAndLeavesNoOutsidePixels() throws {
        let (k, d) = try CameraUndistortion.parse(modelId: 4, params: Self.referenceParams)
        let frame = CameraUndistortion.undistortedFrame(intrinsics: k, distortion: d,
                                                        width: Self.referenceWidth,
                                                        height: Self.referenceHeight)
        let map = ImageUndistorter.makeMap(intrinsics: k, distortion: d,
                                           sourceWidth: Self.referenceWidth,
                                           sourceHeight: Self.referenceHeight)
        // The rectified frame is exactly what the crop arithmetic decided — the resampler never
        // second-guesses it, because cameras.txt will carry these numbers and the core requires the
        // GT image dimensions to match the camera exactly (spec §5.2).
        XCTAssertEqual(map.width, frame.width)
        XCTAssertEqual(map.height, frame.height)
        XCTAssertEqual(map.cx, frame.cx)
        XCTAssertEqual(map.cy, frame.cy)
        XCTAssertEqual(map.coordinates.count, 2 * map.width * map.height)
        // Zero is not a nicety: a single outside pixel means the crop arithmetic is wrong and black
        // borders would enter the training data.
        XCTAssertEqual(map.outsidePixels, 0)
    }

    func testFisheyeMapLeavesNoOutsidePixelsAndShowsTheFieldOfViewLoss() throws {
        // The fisheye branch reaches the frame solver through a different formula (θ = atan(r)),
        // and it fails in the opposite direction from the perspective models: the equidistant map
        // *compresses*, so a pinhole output of the same size samples well inside the source instead
        // of overrunning it. Nothing is cropped — the frame is simply never grown — and the outer
        // ring of the capture is therefore never read. That unread ring is the field of view loss
        // described in the CameraDistortion.swift header, not a resampling defect.
        let (k, d) = try CameraUndistortion.parse(modelId: 5,
                                                  params: [200, 200, 200, 150, -0.02, 0.004, 0, 0])
        let map = ImageUndistorter.makeMap(intrinsics: k, distortion: d,
                                           sourceWidth: 400, sourceHeight: 300)
        XCTAssertEqual(map.outsidePixels, 0)
        XCTAssertEqual(map.width, 400)      // never grown past the source
        XCTAssertEqual(map.height, 300)

        var minU = Float.infinity, maxU = -Float.infinity
        var minV = Float.infinity, maxV = -Float.infinity
        for i in stride(from: 0, to: map.coordinates.count, by: 2) {
            minU = min(minU, map.coordinates[i]);     maxU = max(maxU, map.coordinates[i])
            minV = min(minV, map.coordinates[i + 1]); maxV = max(maxV, map.coordinates[i + 1])
        }
        print("[fisheye] source region read: u \(minU)…\(maxU) of 400, v \(minV)…\(maxV) of 300")
        // Tens of pixels on every side go unused — the wide directions a pinhole cannot express.
        XCTAssertGreaterThan(minU, 30)
        XCTAssertLessThan(maxU, 400 - 30)
        XCTAssertGreaterThan(minV, 20)
        XCTAssertLessThan(maxV, 300 - 20)
    }

    func testMapCoordinatesFollowThePixelCenterConvention() throws {
        // The contract the crop arithmetic depends on (spec footnote 6): output pixel (u', v') is
        // evaluated at its centre, and the stored value is the source *texel* coordinate, i.e. the
        // source pixel centre minus 0.5. Checked against an independent evaluation.
        let (k, d) = try CameraUndistortion.parse(modelId: 4, params: Self.syntheticParams)
        let map = ImageUndistorter.makeMap(intrinsics: k, distortion: d,
                                           sourceWidth: Self.syntheticWidth,
                                           sourceHeight: Self.syntheticHeight)
        for (u, v) in [(0, 0), (7, 3), (map.width / 2, map.height / 2),
                       (map.width - 1, map.height - 1)] {
            let x = (Double(u) + 0.5 - map.cx) / k.fx
            let y = (Double(v) + 0.5 - map.cy) / k.fy
            let (xd, yd) = d.distort(x, y)
            let expectedU = k.fx * xd + k.cx - 0.5
            let expectedV = k.fy * yd + k.cy - 0.5
            let index = 2 * (v * map.width + u)
            XCTAssertEqual(Double(map.coordinates[index]), expectedU, accuracy: 1e-3)
            XCTAssertEqual(Double(map.coordinates[index + 1]), expectedV, accuracy: 1e-3)
        }
    }

    // MARK: - Identity

    func testZeroCoefficientPerspectiveIsPixelIdenticalAndUncropped() throws {
        // SIMPLE_RADIAL with k = 0 → the distortion is the exact identity, so the frame is carried
        // over untouched and every output pixel must be a verbatim copy. (The import skips this
        // case entirely rather than re-encoding — this test is what makes that skip safe.)
        let (k, d) = try CameraUndistortion.parse(modelId: 2, params: [50, 32, 24, 0])
        let map = ImageUndistorter.makeMap(intrinsics: k, distortion: d,
                                           sourceWidth: 64, sourceHeight: 48)
        XCTAssertEqual(map.width, 64)
        XCTAssertEqual(map.height, 48)
        XCTAssertEqual(map.outsidePixels, 0)

        let source = try makeImage(width: 64, height: 48) { u, v in
            // A deterministic, high-frequency pattern: any resampling at all would smear it.
            let n = (u &* 37 &+ v &* 101) & 0xFF
            return (Double(n), Double((n &* 7) & 0xFF), Double((n &* 13) & 0xFF))
        }
        let rectified = try ImageUndistorter.rectify(source, map: map)
        XCTAssertEqual(rectified.width, 64)
        XCTAssertEqual(rectified.height, 48)
        let before = try readPixels(source)
        let after = try readPixels(rectified)
        XCTAssertEqual(before, after)
    }

    func testZeroCoefficientCameraWithAnOffCentrePrincipalPointIsPixelIdenticalToo() throws {
        // The same claim as above on the camera shape that used to break it. The fixture above is
        // 64×48 with its principal point at (32, 24) — dead centre, the one place where "carry the
        // camera over" and "re-center on W/2" agree, which is why it passed while the bug was live.
        //
        // Here the principal point sits 7 px and 5 px off centre. Before 2026-08-15 FIX 2 the frame
        // came back 80×60 with `cx' = 40, cy' = 30`, so the map read the source 7 columns right and
        // 5 rows up: 80·60 − 73·55 = 785 pixels (16.4 %) fell outside the capture and were written
        // black, on the same camera the core had been handed as "already fine".
        let (k, d) = try CameraUndistortion.parse(modelId: 4,
                                                  params: [90, 90, 47, 25, 0, 0, 0, 0])
        let map = ImageUndistorter.makeMap(intrinsics: k, distortion: d,
                                           sourceWidth: 80, sourceHeight: 60)
        XCTAssertEqual(map.width, 80)
        XCTAssertEqual(map.height, 60)
        XCTAssertEqual(map.cx, 47, "the map must keep the camera's own principal point")
        XCTAssertEqual(map.cy, 25)
        XCTAssertEqual(map.outsidePixels, 0, "not one pixel may sample outside the capture")

        let source = try makeImage(width: 80, height: 60) { u, v in
            let n = (u &* 37 &+ v &* 101) & 0xFF
            return (Double(n), Double((n &* 7) & 0xFF), Double((n &* 13) & 0xFF))
        }
        let rectified = try ImageUndistorter.rectify(source, map: map)
        XCTAssertEqual(try readPixels(source), try readPixels(rectified))
    }

    // MARK: - Synthetic round trip (the strong test)

    func testSyntheticDistortionRoundTripRecoversThePattern() throws {
        let (k, d) = try CameraUndistortion.parse(modelId: 4, params: Self.syntheticParams)
        let map = ImageUndistorter.makeMap(intrinsics: k, distortion: d,
                                           sourceWidth: Self.syntheticWidth,
                                           sourceHeight: Self.syntheticHeight)
        XCTAssertEqual(map.outsidePixels, 0)

        // The truth: the pattern as an ideal pinhole camera with the rectified frame would see it.
        let ideal = try makeImage(width: map.width, height: map.height) { u, v in
            let g = Self.pattern(Double(u) + 0.5, Double(v) + 0.5)
            return (g, g, g)
        }

        // The fake capture: for each source pixel, undo the distortion numerically to recover the
        // ideal direction it saw, project that direction into the rectified pinhole frame, and read
        // the pattern there analytically (no resampling — the synthesis introduces no error of its
        // own, so what the comparison measures is our rectification).
        var newtonFailures = 0
        let captured = try makeImage(width: Self.syntheticWidth, height: Self.syntheticHeight) { u, v in
            let xd = (Double(u) + 0.5 - k.cx) / k.fx
            let yd = (Double(v) + 0.5 - k.cy) / k.fy
            guard let (x, y) = Self.newtonInverse(d, xd, yd) else {
                newtonFailures += 1
                return (0, 0, 0)
            }
            let g = Self.pattern(k.fx * x + map.cx, k.fy * y + map.cy)
            return (g, g, g)
        }
        XCTAssertEqual(newtonFailures, 0, "the synthesis itself must be well posed")

        let rectified = try ImageUndistorter.rectify(captured, map: map)
        XCTAssertEqual(rectified.width, map.width)
        XCTAssertEqual(rectified.height, map.height)

        // Compared one pixel in from the border: the outermost row and column read source samples
        // right at the edge of the bilinear domain, where there is no neighbour to average with.
        let corrected = try compare(rectified, ideal, inset: 1)
        // Control: the same measurement without rectification — the centre crop of the fake capture
        // against the same truth. If the rectification were a no-op this number would be the one
        // above, and the test would be vacuous.
        let uncorrected = try compare(centerCrop(captured, width: map.width, height: map.height),
                                      ideal, inset: 1)

        print("[round trip] rectified PSNR \(String(format: "%.2f", corrected.psnr)) dB, "
            + "max abs error \(corrected.maxAbs) levels, RMS \(String(format: "%.3f", corrected.rms))")
        print("[round trip] control (unrectified centre crop) PSNR "
            + "\(String(format: "%.2f", uncorrected.psnr)) dB, max abs error \(uncorrected.maxAbs)")

        XCTAssertGreaterThan(corrected.psnr, 40.0)
        XCTAssertLessThanOrEqual(corrected.maxAbs, 12)
        // The control must be far worse — 15 dB of headroom is well beyond any tuning noise.
        XCTAssertGreaterThan(corrected.psnr - uncorrected.psnr, 15.0)
    }

    // MARK: - Files

    func testDirectoryKeepsNamesAndExtensionsAndWritesJpegAtTheFrameSize() throws {
        let (k, d) = try CameraUndistortion.parse(modelId: 4, params: Self.syntheticParams)
        let map = ImageUndistorter.makeMap(intrinsics: k, distortion: d,
                                           sourceWidth: Self.syntheticWidth,
                                           sourceHeight: Self.syntheticHeight)
        let sourceDir = workDir.appendingPathComponent("images", isDirectory: true)
        try FileManager.default.createDirectory(at: sourceDir, withIntermediateDirectories: true)
        // Deliberately mixed: a PNG and an upper-case JPG. The core builds paths from the names in
        // images.bin, so both must come out under exactly these names.
        let names = ["frame_0001.png", "IMG_0002.JPG"]
        let sources = try names.map { try writePNG(named: $0, in: sourceDir) }

        let target = workDir.appendingPathComponent("images.rectified", isDirectory: true)
        var ticks: [(Int, Int)] = []
        let out = try ImageUndistorter.undistortDirectory(sources: sources, into: target, map: map,
                                                          progress: { ticks.append(($0, $1)) })
        XCTAssertEqual(out, target)
        XCTAssertEqual(ticks.map(\.0), [1, 2])
        XCTAssertEqual(ticks.map(\.1), [2, 2])

        let written = try FileManager.default.contentsOfDirectory(atPath: target.path).sorted()
        XCTAssertEqual(written, names.sorted())
        for name in names {
            let url = target.appendingPathComponent(name)
            // JPEG content regardless of the name (CGImageSource decides by content).
            let magic = try Data(contentsOf: url).prefix(2)
            XCTAssertEqual([UInt8](magic), [0xFF, 0xD8], "\(name) should be JPEG bytes")
            // Exactly the frame the map declares — cameras.txt will carry the same numbers.
            let dims = try XCTUnwrap(decodedDimensions(url))
            XCTAssertEqual(dims.0, map.width, "\(name) width")
            XCTAssertEqual(dims.1, map.height, "\(name) height")
        }
        // The staging folder is gone: it was renamed, not copied.
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: workDir.appendingPathComponent("images.rectified.building").path))
    }

    func testFailureLeavesNoStagingAndNoTarget() throws {
        let (k, d) = try CameraUndistortion.parse(modelId: 4, params: Self.syntheticParams)
        let map = ImageUndistorter.makeMap(intrinsics: k, distortion: d,
                                           sourceWidth: Self.syntheticWidth,
                                           sourceHeight: Self.syntheticHeight)
        let sourceDir = workDir.appendingPathComponent("images", isDirectory: true)
        try FileManager.default.createDirectory(at: sourceDir, withIntermediateDirectories: true)
        let good = try writePNG(named: "a.png", in: sourceDir)
        let bad = sourceDir.appendingPathComponent("b.png")
        try Data([0x00, 0x01, 0x02, 0x03]).write(to: bad)

        let target = workDir.appendingPathComponent("images.rectified", isDirectory: true)
        XCTAssertThrowsError(try ImageUndistorter.undistortDirectory(sources: [good, bad],
                                                                     into: target, map: map))
        // A half-rectified image set must not survive: its pixels and its cameras.txt disagree.
        XCTAssertFalse(FileManager.default.fileExists(atPath: target.path))
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: workDir.appendingPathComponent("images.rectified.building").path))
    }

    func testWrongSizedImageIsRejectedRatherThanScaled() throws {
        let (k, d) = try CameraUndistortion.parse(modelId: 4, params: Self.syntheticParams)
        let map = ImageUndistorter.makeMap(intrinsics: k, distortion: d,
                                           sourceWidth: Self.syntheticWidth,
                                           sourceHeight: Self.syntheticHeight)
        let odd = try makeImage(width: 64, height: 48) { _, _ in (128, 128, 128) }
        XCTAssertThrowsError(try ImageUndistorter.rectify(odd, map: map)) { error in
            XCTAssertTrue("\(error)".contains("64×48"), "\(error)")
        }
    }

    func testExistingTargetIsRefused() throws {
        let (k, d) = try CameraUndistortion.parse(modelId: 4, params: Self.syntheticParams)
        let map = ImageUndistorter.makeMap(intrinsics: k, distortion: d,
                                           sourceWidth: Self.syntheticWidth,
                                           sourceHeight: Self.syntheticHeight)
        let sourceDir = workDir.appendingPathComponent("images", isDirectory: true)
        try FileManager.default.createDirectory(at: sourceDir, withIntermediateDirectories: true)
        let source = try writePNG(named: "a.png", in: sourceDir)
        let target = workDir.appendingPathComponent("images.rectified", isDirectory: true)
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        // Never merge into an existing folder: the caller swaps directories, and a stale mix of
        // rectified and unrectified frames is undetectable downstream.
        XCTAssertThrowsError(try ImageUndistorter.undistortDirectory(sources: [source],
                                                                     into: target, map: map))
    }

    // MARK: - Pattern and numeric inverse (test-side only)

    /// A checkerboard with tanh-softened edges, in rectified-frame pixel coordinates.
    private static func pattern(_ x: Double, _ y: Double) -> Double {
        let period = 24.0
        let sx = tanh(2.5 * sin(2 * .pi * x / period))
        let sy = tanh(2.5 * sin(2 * .pi * y / period))
        return 127.5 + 100.0 * sx * sy
    }

    /// Newton's method on the forward distortion, used only to synthesize the fake capture.
    /// Production never needs it (see the CameraDistortion.swift header).
    private static func newtonInverse(_ d: CameraDistortion,
                                      _ targetX: Double, _ targetY: Double) -> (Double, Double)? {
        var x = targetX, y = targetY
        let h = 1e-7
        for _ in 0..<40 {
            let (fx0, fy0) = d.distort(x, y)
            let rx = fx0 - targetX, ry = fy0 - targetY
            if abs(rx) < 1e-13 && abs(ry) < 1e-13 { return (x, y) }
            let (axp, ayp) = d.distort(x + h, y)
            let (axm, aym) = d.distort(x - h, y)
            let (bxp, byp) = d.distort(x, y + h)
            let (bxm, bym) = d.distort(x, y - h)
            let j11 = (axp - axm) / (2 * h), j21 = (ayp - aym) / (2 * h)
            let j12 = (bxp - bxm) / (2 * h), j22 = (byp - bym) / (2 * h)
            let det = j11 * j22 - j12 * j21
            guard abs(det) > 1e-14 else { return nil }
            x -= (j22 * rx - j12 * ry) / det
            y -= (-j21 * rx + j11 * ry) / det
            guard x.isFinite, y.isFinite else { return nil }
        }
        let (cx, cy) = d.distort(x, y)
        guard abs(cx - targetX) < 1e-10, abs(cy - targetY) < 1e-10 else { return nil }
        return (x, y)
    }

    // MARK: - Bitmap helpers

    /// sRGB, 8 bit, R G B (unused) — the same layout the resampler addresses.
    private func makeImage(width: Int, height: Int,
                           _ sample: (Int, Int) -> (Double, Double, Double)) throws -> CGImage {
        let ctx = try XCTUnwrap(CGContext(data: nil, width: width, height: height,
                                          bitsPerComponent: 8, bytesPerRow: 0,
                                          space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        let base = try XCTUnwrap(ctx.data).assumingMemoryBound(to: UInt8.self)
        for v in 0..<height {
            let row = base + v * ctx.bytesPerRow
            for u in 0..<width {
                let (r, g, b) = sample(u, v)
                row[u * 4 + 0] = UInt8(min(255, max(0, r.rounded())))
                row[u * 4 + 1] = UInt8(min(255, max(0, g.rounded())))
                row[u * 4 + 2] = UInt8(min(255, max(0, b.rounded())))
                row[u * 4 + 3] = 255
            }
        }
        return try XCTUnwrap(ctx.makeImage())
    }

    /// Rows of tightly packed RGB triples (the row padding is dropped so two images of the same
    /// size compare byte for byte regardless of how CoreGraphics aligned them).
    private func readPixels(_ image: CGImage) throws -> [UInt8] {
        let w = image.width, h = image.height
        let ctx = try XCTUnwrap(CGContext(data: nil, width: w, height: h,
                                          bitsPerComponent: 8, bytesPerRow: 0,
                                          space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        let base = try XCTUnwrap(ctx.data).assumingMemoryBound(to: UInt8.self)
        var out = [UInt8](repeating: 0, count: w * h * 3)
        for v in 0..<h {
            let row = base + v * ctx.bytesPerRow
            for u in 0..<w {
                out[(v * w + u) * 3 + 0] = row[u * 4 + 0]
                out[(v * w + u) * 3 + 1] = row[u * 4 + 1]
                out[(v * w + u) * 3 + 2] = row[u * 4 + 2]
            }
        }
        return out
    }

    private func centerCrop(_ image: CGImage, width: Int, height: Int) throws -> CGImage {
        let x = (image.width - width) / 2
        let y = (image.height - height) / 2
        return try XCTUnwrap(image.cropping(to: CGRect(x: x, y: y, width: width, height: height)))
    }

    private func compare(_ a: CGImage, _ b: CGImage,
                         inset: Int) throws -> (psnr: Double, rms: Double, maxAbs: Int) {
        XCTAssertEqual(a.width, b.width)
        XCTAssertEqual(a.height, b.height)
        let w = a.width, h = a.height
        let pa = try readPixels(a), pb = try readPixels(b)
        var sum = 0.0, maxAbs = 0, n = 0
        for v in inset..<(h - inset) {
            for u in inset..<(w - inset) {
                for c in 0..<3 {
                    let i = (v * w + u) * 3 + c
                    let e = Int(pa[i]) - Int(pb[i])
                    sum += Double(e * e)
                    maxAbs = max(maxAbs, abs(e))
                    n += 1
                }
            }
        }
        XCTAssertGreaterThan(n, 0)
        let mse = sum / Double(n)
        let psnr = mse == 0 ? Double.infinity : 10 * log10(255.0 * 255.0 / mse)
        return (psnr, mse.squareRoot(), maxAbs)
    }

    // MARK: - File helpers

    /// A PNG of the synthetic camera's size carrying the soft checkerboard (a real image, so the
    /// decode path is exercised end to end).
    private func writePNG(named name: String, in dir: URL) throws -> URL {
        let image = try makeImage(width: Self.syntheticWidth, height: Self.syntheticHeight) { u, v in
            let g = Self.pattern(Double(u) + 0.5, Double(v) + 0.5)
            return (g, g, g)
        }
        let url = dir.appendingPathComponent(name)
        let dst = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL,
                                                                UTType.png.identifier as CFString,
                                                                1, nil))
        CGImageDestinationAddImage(dst, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dst))
        return url
    }

    private func decodedDimensions(_ url: URL) -> (Int, Int)? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? Int,
              let h = props[kCGImagePropertyPixelHeight] as? Int else { return nil }
        return (w, h)
    }
}
