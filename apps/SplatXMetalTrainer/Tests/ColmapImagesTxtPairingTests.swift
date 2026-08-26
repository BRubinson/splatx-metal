import XCTest
import simd
@testable import SplatXMetalTrainer

/// Regression guards for FIX 1 (2026-08-15) — `images.txt` was losing half of its views.
///
/// `readImagesTxt` pairs the non-comment lines (header + POINTS2D) and steps `i += 2`. It used to
/// read them through the shared `lines()` helper, which drops empty lines — and the POINTS2D line
/// is empty for every image in the bundles `ColmapTextWriter` writes (and, from this round on, in
/// the rectified bundles the importer writes). The array collapsed to [pose, pose, pose, …] and
/// `i += 2` returned every other view: a real 110-image project parsed as 55.
///
/// These tests pin both shapes — empty POINTS2D and real observation lines — with ≥4 views so any
/// return to the every-other behaviour is unmistakable.
final class ColmapImagesTxtPairingTests: XCTestCase {
    private var dirs: [URL] = []

    override func tearDown() {
        for d in dirs { try? FileManager.default.removeItem(at: d) }
        dirs = []
        super.tearDown()
    }

    /// A temporary `sparse/0` holding `cameras.txt` (needed by `read`) and the given `images.txt`.
    private func makeSparse(images: String) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "cmpair-\(UUID().uuidString)", directoryHint: .isDirectory)
        dirs.append(root)
        let dir = root.appending(path: "sparse/0", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try "# Camera list\n1 PINHOLE 1920 1440 1500 1510 960 720\n"
            .write(to: dir.appending(path: "cameras.txt"), atomically: true, encoding: .utf8)
        try images.write(to: dir.appending(path: "images.txt"), atomically: true, encoding: .utf8)
        return dir
    }

    private func poseLine(_ i: Int, _ name: String) -> String {
        "\(i) 1 0 0 0 \(i).5 -0.2 3 1 \(name)"
    }

    /// THE fix. The `ColmapTextWriter` shape (an empty POINTS2D line per image): every view must
    /// come back, in file order, with its own pose — not every other one.
    func testEmptyPoints2DLinesYieldEveryView() throws {
        let names = ["f0.jpg", "f1.jpg", "f2.jpg", "f3.jpg", "f4.jpg"]
        var images = "# Image list: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
        for (i, n) in names.enumerated() { images += poseLine(i + 1, n) + "\n\n" }
        let model = try ColmapCameraReader.read(sparseDir: try makeSparse(images: images))
        XCTAssertEqual(model.views.count, names.count)
        XCTAssertEqual(model.views.map(\.name), names)          // order preserved
        XCTAssertEqual(model.views.map(\.id), [1, 2, 3, 4, 5])
        // Each view keeps its own tx, so a shifted pairing cannot pass either.
        for (i, v) in model.views.enumerated() {
            XCTAssertEqual(v.tvec.x, Double(i + 1) + 0.5, accuracy: 1e-9)
            XCTAssertEqual(v.cameraId, 1)
        }
    }

    /// The real COLMAP shape (a populated POINTS2D line per image) must keep pairing exactly — the
    /// observation lines have ≥10 tokens starting with an integer, so any "parse every line" repair
    /// would double the view count here.
    func testObservationLinesAreStillSkipped() throws {
        let names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
        var images = "# Image list: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
        for (i, n) in names.enumerated() {
            images += poseLine(i + 1, n) + "\n"
            images += "512 340 1 600 200 2 700 300 3 800 400 5\n"
        }
        let model = try ColmapCameraReader.read(sparseDir: try makeSparse(images: images))
        XCTAssertEqual(model.views.count, names.count)
        XCTAssertEqual(model.views.map(\.name), names)
        XCTAssertEqual(model.views.map { $0.tvec.x }, [1.5, 2.5, 3.5, 4.5])
    }

    /// A file whose POINTS2D lines are mixed — some images observe points, some do not. COLMAP
    /// itself emits this whenever a registered image contributes no track.
    func testMixedEmptyAndPopulatedObservationLines() throws {
        let images = """
        # Image list
        \(poseLine(1, "a.jpg"))
        512 340 1 600 200 2
        \(poseLine(2, "b.jpg"))

        \(poseLine(3, "c.jpg"))
        700 300 3
        \(poseLine(4, "d.jpg"))

        """
        let model = try ColmapCameraReader.read(sparseDir: try makeSparse(images: images))
        XCTAssertEqual(model.views.map(\.name), ["a.jpg", "b.jpg", "c.jpg", "d.jpg"])
    }

    /// The trailing-empty-line policy: `imageDataLines` trims empty lines only at the end, so the
    /// same content parses identically with or without a final newline — and the writer's
    /// `NAME\n\n` (two trailing empties) must not be parsed as a header and throw.
    func testTrailingNewlinePresentOrAbsent() throws {
        let names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
        var body = "# Image list\n"
        for (i, n) in names.enumerated() { body += poseLine(i + 1, n) + "\n\n" }
        XCTAssertTrue(body.hasSuffix("\n\n"))                       // the writer's shape
        for text in [body, String(body.dropLast()), String(body.dropLast(2))] {
            let model = try ColmapCameraReader.read(sparseDir: try makeSparse(images: text))
            XCTAssertEqual(model.views.map(\.name), names)
        }
        // The same for the observation-line shape, with and without the final newline.
        var obs = "# Image list\n"
        for (i, n) in names.enumerated() { obs += poseLine(i + 1, n) + "\n0 0 1\n" }
        for text in [obs, String(obs.dropLast())] {
            let model = try ColmapCameraReader.read(sparseDir: try makeSparse(images: text))
            XCTAssertEqual(model.views.map(\.name), names)
        }
    }

    /// Comment lines: COLMAP's real four-line banner up top, plus a comment between two complete
    /// image records. Comments are removed before pairing, so only whole records may be separated
    /// by them — which is what COLMAP emits.
    func testCommentLinesAreIgnored() throws {
        let images = """
        # Image list with two lines of data per image:
        #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        #   POINTS2D[] as (X, Y, POINT3D_ID)
        # Number of images: 4, mean observations per image: 1
        \(poseLine(1, "a.jpg"))
        512 340 1
        \(poseLine(2, "b.jpg"))

        # a stray comment between records
        \(poseLine(3, "c.jpg"))
        700 300 3
        \(poseLine(4, "d.jpg"))

        """
        let model = try ColmapCameraReader.read(sparseDir: try makeSparse(images: images))
        XCTAssertEqual(model.views.map(\.name), ["a.jpg", "b.jpg", "c.jpg", "d.jpg"])
    }

    /// The strongest fixation: `ColmapTextWriter` and `ColmapCameraReader` are a pair, so what the
    /// app writes at align time must read back whole. Frames are handed over unsorted because the
    /// writer sorts by name and binds CAMERA_ID to that order.
    func testRoundTripThroughColmapTextWriter() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "cmrt-\(UUID().uuidString)", directoryHint: .isDirectory)
        dirs.append(root)
        let sparse0 = root.appending(path: "sparse/0", directoryHint: .isDirectory)
        func frame(_ name: String, tx: Double) -> AlignedFrame {
            AlignedFrame(name: name, width: 640, height: 480, fx: 500, fy: 510, cx: 320, cy: 240,
                         pose: ColmapPose(qw: 1, qx: 0, qy: 0, qz: 0, tx: tx, ty: -1, tz: 2))
        }
        try ColmapTextWriter.write(
            frames: [frame("c.jpg", tx: 3), frame("a.jpg", tx: 1),
                     frame("d.jpg", tx: 4), frame("b.jpg", tx: 2)],
            points: [AlignedPoint(x: 0, y: 0, z: 1, r: 1, g: 2, b: 3)],
            to: sparse0)

        let model = try ColmapCameraReader.read(sparseDir: sparse0)
        XCTAssertEqual(model.views.count, 4)
        XCTAssertEqual(model.views.map(\.name), ["a.jpg", "b.jpg", "c.jpg", "d.jpg"])
        XCTAssertEqual(model.views.map { $0.tvec.x }, [1, 2, 3, 4])
        XCTAssertEqual(model.cameras.count, 4)
        // CAMERA_ID follows the name sort, so every view resolves to a real camera row.
        for v in model.views {
            let cam = try XCTUnwrap(model.cameras[v.cameraId])
            XCTAssertEqual(cam.fx, 500, accuracy: 1e-9)
            XCTAssertEqual(cam.width, 640)
        }
    }
}
