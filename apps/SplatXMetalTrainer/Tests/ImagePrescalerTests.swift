// Training-speed Task 3: ImagePrescaler — the pure logic (dimensions, completeness, and the
// file-extension decision) plus a round trip with small real images. Real images are created
// and verified only in temporaryDirectory (the TCC rule — Desktop and Downloads are
// prohibited).
import XCTest
import ImageIO
import UniformTypeIdentifiers
@testable import SplatXMetalTrainer

final class ImagePrescalerTests: XCTestCase {

    // MARK: Pure logic

    func testExpectedDimensionMatchesCoreLroundConvention() {
        // The core's downscaled_dim = lround(v/factor) — measured on garden (the ImageIO.h
        // comment is the authoritative contract).
        XCTAssertEqual(ImagePrescaler.expectedDimension(5187, factor: 4), 1297)
        XCTAssertEqual(ImagePrescaler.expectedDimension(3361, factor: 4), 840)
        XCTAssertEqual(ImagePrescaler.expectedDimension(648, factor: 2), 324)
        // Halves round away from zero (C lround): 2.5 → 3, 23.5 → 24.
        XCTAssertEqual(ImagePrescaler.expectedDimension(5, factor: 2), 3)
        XCTAssertEqual(ImagePrescaler.expectedDimension(47, factor: 2), 24)
        XCTAssertEqual(ImagePrescaler.expectedDimension(65, factor: 2), 33)
        XCTAssertEqual(ImagePrescaler.expectedDimension(100, factor: 1), 100)
    }

    func testIsImageFileByExtensionCaseInsensitive() {
        XCTAssertTrue(ImagePrescaler.isImageFile("IMG_0001.JPG"))
        XCTAssertTrue(ImagePrescaler.isImageFile("frame.png"))
        XCTAssertTrue(ImagePrescaler.isImageFile("shot.jpeg"))
        XCTAssertTrue(ImagePrescaler.isImageFile("capture.HEIC"))
        XCTAssertFalse(ImagePrescaler.isImageFile("notes.txt"))
        XCTAssertFalse(ImagePrescaler.isImageFile(".DS_Store"))
        XCTAssertFalse(ImagePrescaler.isImageFile("images"))
    }

    func testMissingNamesCompletenessJudgment() {
        let sources = ["a.jpg", "b.jpg", "c.jpg"]
        XCTAssertEqual(ImagePrescaler.missingNames(sources: sources,
                                                   existing: ["a.jpg", "b.jpg", "c.jpg"]), [])
        // Extra files are fine: it is complete as long as every source name is present — the
        // comparison is by name.
        XCTAssertEqual(ImagePrescaler.missingNames(sources: sources,
                                                   existing: ["a.jpg", "b.jpg", "c.jpg", "x.jpg"]), [])
        XCTAssertEqual(ImagePrescaler.missingNames(sources: sources,
                                                   existing: ["a.jpg", "c.jpg"]), ["b.jpg"])
        XCTAssertEqual(ImagePrescaler.missingNames(sources: sources, existing: []), sources)
    }

    // MARK: Real-image round trip (temporaryDirectory)

    private var datasetDir: URL!
    private var imagesDir: URL!

    override func setUpWithError() throws {
        datasetDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ImagePrescalerTests-\(UUID().uuidString)", isDirectory: true)
        imagesDir = datasetDir.appendingPathComponent("images", isDirectory: true)
        try FileManager.default.createDirectory(at: imagesDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: datasetDir)
    }

    /// Creates a solid-color PNG (deliberately .png, to verify the contract that the original
    /// extension is preserved).
    private func writePNG(name: String, width: Int, height: Int) throws -> URL {
        let ctx = CGContext(data: nil, width: width, height: height,
                            bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(CGColor(srgbRed: 0.2, green: 0.6, blue: 0.4, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let image = ctx.makeImage()!
        let url = imagesDir.appendingPathComponent(name)
        let dst = CGImageDestinationCreateWithURL(url as CFURL,
                                                  UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dst, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dst))
        return url
    }

    private func decodedDims(_ url: URL) -> (Int, Int)? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? Int,
              let h = props[kCGImagePropertyPixelHeight] as? Int else { return nil }
        return (w, h)
    }

    func testFactorOneIsNoOp() throws {
        _ = try writePNG(name: "a.png", width: 64, height: 48)
        let out = ImagePrescaler.prescaleIfNeeded(imagesDir: imagesDir, factor: 1)
        XCTAssertNil(out)
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: datasetDir.appendingPathComponent("images_1").path))
    }

    func testGeneratesExactDimsJpegUnderOriginalNames() throws {
        _ = try writePNG(name: "a.png", width: 64, height: 48)
        _ = try writePNG(name: "b.png", width: 65, height: 47) // odd dimensions — the rounding path
        var ticks: [(Int, Int)] = []
        let out = ImagePrescaler.prescaleIfNeeded(imagesDir: imagesDir, factor: 2,
                                                  progress: { ticks.append(($0, $1)) })
        let target = try XCTUnwrap(out)
        XCTAssertEqual(target.lastPathComponent, "images_2")
        // The file names are the originals verbatim (extension included) — the core assembles
        // the paths from the names in images.bin.
        let names = try FileManager.default.contentsOfDirectory(atPath: target.path).sorted()
        XCTAssertEqual(names, ["a.png", "b.png"])
        // The dimensions match the core's GT dims guard exactly: lround(64/2)=32,
        // lround(48/2)=24, lround(65/2)=33 (2.5 rounds away from zero), lround(47/2)=24.
        let dimsA = try XCTUnwrap(decodedDims(target.appendingPathComponent("a.png")))
        XCTAssertEqual(dimsA.0, 32); XCTAssertEqual(dimsA.1, 24)
        let dimsB = try XCTUnwrap(decodedDims(target.appendingPathComponent("b.png")))
        XCTAssertEqual(dimsB.0, 33); XCTAssertEqual(dimsB.1, 24)
        // The contents are JPEG (the SOI magic FF D8) — CGImageSource decides by content, so
        // the name does not matter.
        let head = try Data(contentsOf: target.appendingPathComponent("a.png")).prefix(2)
        XCTAssertEqual([UInt8](head), [0xFF, 0xD8])
        // Progress callback: 1/2 → 2/2.
        XCTAssertEqual(ticks.map { $0.0 }, [1, 2])
        XCTAssertEqual(ticks.map { $0.1 }, [2, 2])
        // No staging residue (the rename happens after completion).
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: datasetDir.appendingPathComponent("images_2.building").path))
    }

    func testCompleteFolderIsNoOp() throws {
        _ = try writePNG(name: "a.png", width: 64, height: 48)
        let first = try XCTUnwrap(ImagePrescaler.prescaleIfNeeded(imagesDir: imagesDir, factor: 2))
        let file = first.appendingPathComponent("a.png")
        let mtime1 = try FileManager.default.attributesOfItem(atPath: file.path)[.modificationDate] as? Date
        var ticks = 0
        let second = ImagePrescaler.prescaleIfNeeded(imagesDir: imagesDir, factor: 2,
                                                     progress: { _, _ in ticks += 1 })
        XCTAssertEqual(second, first)
        XCTAssertEqual(ticks, 0) // no-op — nothing is regenerated
        let mtime2 = try FileManager.default.attributesOfItem(atPath: file.path)[.modificationDate] as? Date
        XCTAssertEqual(mtime1, mtime2)
    }

    func testTopUpFillsOnlyMissingFiles() throws {
        _ = try writePNG(name: "a.png", width: 64, height: 48)
        _ = try writePNG(name: "b.png", width: 64, height: 48)
        let target = try XCTUnwrap(ImagePrescaler.prescaleIfNeeded(imagesDir: imagesDir, factor: 2))
        try FileManager.default.removeItem(at: target.appendingPathComponent("b.png"))
        var ticks: [(Int, Int)] = []
        let out = ImagePrescaler.prescaleIfNeeded(imagesDir: imagesDir, factor: 2,
                                                  progress: { ticks.append(($0, $1)) })
        XCTAssertEqual(out, target)
        XCTAssertEqual(ticks.map { $0.0 }, [1]) // only the one missing file is generated
        XCTAssertEqual(ticks.map { $0.1 }, [1])
        let names = try FileManager.default.contentsOfDirectory(atPath: target.path).sorted()
        XCTAssertEqual(names, ["a.png", "b.png"])
    }

    func testFailureRemovesIncompleteFolderAndReturnsNil() throws {
        _ = try writePNG(name: "a.png", width: 64, height: 48)
        // Garbage bytes carrying an image extension — this induces a decode failure.
        try Data([0x00, 0x01, 0x02, 0x03]).write(to: imagesDir.appendingPathComponent("bad.jpg"))
        var logs: [String] = []
        let out = ImagePrescaler.prescaleIfNeeded(imagesDir: imagesDir, factor: 2,
                                                  log: { logs.append($0) })
        XCTAssertNil(out)
        XCTAssertFalse(logs.isEmpty)
        // No incomplete folder or staging directory is left behind — if one were, the core
        // would pick it up and training would die with a throw.
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: datasetDir.appendingPathComponent("images_2").path))
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: datasetDir.appendingPathComponent("images_2.building").path))
    }
}
