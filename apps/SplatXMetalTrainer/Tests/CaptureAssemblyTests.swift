import XCTest
@testable import SplatXMetalTrainer

/// `ImportService.assembleCaptureBundle` over synthetic bundle trees.
///
/// Unlike `CaptureBundleTests` (pure functions, no disk at all) this one has to touch the file
/// system — the thing under test *is* the copying. Everything lives under
/// `FileManager.temporaryDirectory`, so no test here can reach the real home, ~/Desktop or
/// ~/Downloads, which on this machine would mean a TCC prompt or a hang rather than a failure.
///
/// `testCopiesModelAndPointCloudByteForByte` is the load-bearing one: it is the regression guard for
/// the decision not to flip the LiDAR up axis (spec §2.3). Should anyone parse-and-rewrite
/// `images.txt`, `points3D.txt` or `pointcloud.ply` on the way in — for a coordinate flip or
/// anything else — it fails immediately instead of silently shipping edited user geometry.
final class CaptureAssemblyTests: XCTestCase {
    private var tmp: URL!

    override func setUp() {
        tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("sx-capture-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    }
    override func tearDown() { try? FileManager.default.removeItem(at: tmp) }

    private var src: URL { tmp.appendingPathComponent("bundle", isDirectory: true) }
    private var dst: URL { tmp.appendingPathComponent("assembled", isDirectory: true) }

    @discardableResult
    private func write(_ rel: String, _ body: String = "x") throws -> Data {
        let url = src.appendingPathComponent(rel)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        let data = Data(body.utf8)
        try data.write(to: url)
        return data
    }

    private func exists(_ rel: String) -> Bool {
        FileManager.default.fileExists(atPath: dst.appendingPathComponent(rel).path)
    }

    /// `rel == ""` means the assembly root itself — spelled out rather than relying on
    /// `appendingPathComponent("")`, whose trailing-slash behaviour is not worth depending on.
    private func entries(_ rel: String) -> [String] {
        let url = rel.isEmpty ? dst : dst.appendingPathComponent(rel)
        return ((try? FileManager.default.contentsOfDirectory(atPath: url.path)) ?? []).sorted()
    }

    /// A LiDAR bundle in the shape the phone writes, plus the noise a real one accumulates:
    /// the two log/readme files, an on-device `trained/` run, and Finder leftovers inside `images/`.
    private func makeLidarBundle() throws {
        try write("images/0001.jpg", "frame-one")
        try write("images/0002.jpg", "frame-two")
        try write("images/.DS_Store", "finder")
        try write("images/._0001.jpg", "appledouble")
        // A real one-line model rather than the word "PINHOLE": the import reads the cameras before
        // it decides whether the bundle needs rectifying, so this file is now parsed rather than
        // merely copied. Being pinhole is what sends this bundle down the copy path.
        try write("sparse/0/cameras.txt", ColmapFixture.pinholeCamerasText)
        try write("sparse/0/images.txt", "poses")
        try write("sparse/0/points3D.txt", "points")
        try write("pointcloud.ply", "ply")
        try write("splatx.json", #"{"mode":"lidar"}"#)
        try write("events.ndjson", "{}")
        try write("README.txt", "readme")
        try write("trained/splat.ply", "an on-device result, not an input")
    }

    // MARK: Whitelist

    func testCopiesOnlyWhitelistedEntries() throws {
        try makeLidarBundle()
        let n = try ImportService.assembleCaptureBundle(from: src, into: dst)
        XCTAssertEqual(n, 2)
        // Exact top level, so a future addition to the copy list has to be a deliberate edit here.
        XCTAssertEqual(entries(""), ["images", "pointcloud.ply", "sparse", "splatx.json"])
        XCTAssertEqual(entries("sparse/0"), ["cameras.txt", "images.txt", "points3D.txt"])
        XCTAssertFalse(exists("events.ndjson"))
        XCTAssertFalse(exists("README.txt"))
        XCTAssertFalse(exists("trained"))
    }

    /// The reported count is the count that was copied — the sidecars `images/` accumulates are not
    /// images and must not inflate `Project.imageCount`.
    func testDropsSidecarsInsideImages() throws {
        try makeLidarBundle()
        XCTAssertEqual(try ImportService.assembleCaptureBundle(from: src, into: dst), 2)
        XCTAssertEqual(entries("images"), ["0001.jpg", "0002.jpg"])
    }

    // MARK: Byte identity — the coordinate-preservation guard (spec §2.3)

    func testCopiesModelAndPointCloudByteForByte() throws {
        try makeLidarBundle()
        // Content that a parse/rewrite round trip could not reproduce verbatim: full float
        // precision, a trailing-space line, CRLF, and a stray blank line.
        let images = try write("sparse/0/images.txt",
                               "1 0.7071067811865476 -0.0000000001 0.5 0.5 1 1 0001.jpg \r\n\n")
        let points = try write("sparse/0/points3D.txt",
                               "1 -0.123456789012345 2.0 3.0 255 128 0 0.5\r\n")
        let cloud = try write("pointcloud.ply", "ply\nformat binary_little_endian 1.0\n\u{0}\u{1}\u{2}")
        let frame = try write("images/0001.jpg", "\u{FF}\u{D8}not-really-a-jpeg\u{FF}\u{D9}")

        try ImportService.assembleCaptureBundle(from: src, into: dst)

        let read = { (rel: String) in try Data(contentsOf: self.dst.appendingPathComponent(rel)) }
        XCTAssertEqual(try read("sparse/0/images.txt"), images)
        XCTAssertEqual(try read("sparse/0/points3D.txt"), points)
        XCTAssertEqual(try read("pointcloud.ply"), cloud)
        // Same assertion for a frame: no EXIF normalization pass, so no JPEG re-encode. (This file
        // is not decodable, which is itself the point — a re-encode path would have to fail or
        // corrupt it, while a copy is indifferent.)
        XCTAssertEqual(try read("images/0001.jpg"), frame)
    }

    // MARK: Model set selection

    /// photo/video bundles have no `sparse/` — absence is the normal case, not a failure.
    func testUnposedBundleAssemblesWithoutSparse() throws {
        try write("images/0001.jpg"); try write("images/0002.jpg")
        try write("splatx.json", #"{"mode":"photo"}"#)
        XCTAssertEqual(try ImportService.assembleCaptureBundle(from: src, into: dst), 2)
        XCTAssertEqual(entries(""), ["images", "splatx.json"])
    }

    /// An incomplete model is dropped rather than half-copied, and the verdict has to be the one
    /// `CaptureBundle.kind` reaches on the same tree — otherwise the project opens `.aligned` over
    /// nothing.
    func testPartialModelIsDroppedAndAgreesWithKind() throws {
        try write("images/0001.jpg"); try write("images/0002.jpg")
        try write("sparse/0/cameras.txt"); try write("sparse/0/images.txt")   // no points3D.txt
        try ImportService.assembleCaptureBundle(from: src, into: dst)
        XCTAssertFalse(exists("sparse"))
        XCTAssertEqual(CaptureBundle.kind(root: src, contents: CaptureBundle.fileSystemContents),
                       .unposed)
    }

    /// A bin/txt mixture completes neither set — same rule as `assembleColmap`, same outcome as
    /// `CaptureBundle.kind`.
    func testMixedBinAndTxtModelIsDropped() throws {
        try write("images/0001.jpg"); try write("images/0002.jpg")
        try write("sparse/0/cameras.bin"); try write("sparse/0/images.txt")
        try write("sparse/0/points3D.bin")
        try ImportService.assembleCaptureBundle(from: src, into: dst)
        XCTAssertFalse(exists("sparse"))
    }

    func testPrefersBinaryModelWhenBothArePresent() throws {
        try write("images/0001.jpg"); try write("images/0002.jpg")
        // `cameras.bin` has to be a real one, because preferring the binary set means the import
        // *reads* it: a placeholder would fail as a truncated model instead of being copied.
        try FileManager.default.createDirectory(
            at: src.appendingPathComponent("sparse/0"), withIntermediateDirectories: true)
        try ColmapFixture.pinholeCamerasBin.write(to: src.appendingPathComponent("sparse/0/cameras.bin"))
        for f in ["images.bin", "points3D.bin"] { try write("sparse/0/\(f)") }
        for f in ["cameras.txt", "images.txt", "points3D.txt"] { try write("sparse/0/\(f)") }
        try ImportService.assembleCaptureBundle(from: src, into: dst)
        XCTAssertEqual(entries("sparse/0"), ["cameras.bin", "images.bin", "points3D.bin"])
    }

    // MARK: Rejections

    func testRejectsFewerThanTwoImages() throws {
        try write("images/0001.jpg")
        try write("images/._0001.jpg", "appledouble")   // not an image, must not make the count 2
        XCTAssertThrowsError(try ImportService.assembleCaptureBundle(from: src, into: dst)) { e in
            XCTAssertEqual(e as? ImportError, .notEnoughImages(found: 1))
        }
    }

    func testRejectsMissingImagesDirectory() throws {
        try write("splatx.json", "{}")
        XCTAssertThrowsError(try ImportService.assembleCaptureBundle(from: src, into: dst)) { e in
            XCTAssertEqual(e as? CaptureBundleError, .missingImagesDirectory("bundle"))
        }
    }
}
