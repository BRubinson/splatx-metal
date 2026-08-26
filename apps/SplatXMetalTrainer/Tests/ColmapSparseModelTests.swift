// import-undistortion FIX 2: the two claims about `ColmapSparseModel` that the rectification tests
// do not reach — what a corrupt binary length field does, and what the points writer costs.
//
// Both are about files the app did not write. `ColmapRectificationTests` builds every fixture with
// `ColmapFixture`, i.e. a well-formed model of a handful of points; these two build the shapes a
// real reconstruction or a damaged download produces.
import XCTest
@testable import SplatXMetalTrainer

final class ColmapSparseModelTests: XCTestCase {

    private var tmp: URL!

    override func setUpWithError() throws {
        tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("sx-sparse-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tmp)
    }

    // MARK: - A length field that would overflow the cursor

    func testATrackLengthThatOverflowsTheCursorIsReportedAsTruncationRatherThanTrapping() throws {
        // `Cursor.skipRecords` checked its multiplication and not its addition. A track length of
        // `Int.max / 8` multiplies to `Int.max − 7` without overflowing — and then `offset +
        // bytesToSkip` traps, which in a shipped app is a crash on a corrupt file rather than the
        // error message this type exists to produce. The chosen value is exactly that shape.
        var data = Data()
        func append<T>(_ value: T) { withUnsafeBytes(of: value) { data.append(contentsOf: $0) } }
        append(UInt64(1))                       // one point
        append(UInt64(1))                       // POINT3D_ID
        append(0.5); append(-1.0); append(2.0)  // XYZ
        data.append(1); data.append(2); data.append(3)
        append(0.0)                             // reprojection error
        append(UInt64(Int.max / 8))             // track length — × 8 fits, + offset does not
        try data.write(to: tmp.appendingPathComponent("points3D.bin"))

        XCTAssertThrowsError(try ColmapSparse.readPoints(in: tmp, format: .binary)) { error in
            XCTAssertEqual(error as? ColmapSparseError, .truncated("points3D.bin"))
        }
    }

    func testAnObservationCountThatOverflowsTheMultiplicationIsAlsoTruncation() throws {
        // The multiplication guard, still doing its job: 24 bytes per observation overflows well
        // before the addition can.
        var data = Data()
        func append<T>(_ value: T) { withUnsafeBytes(of: value) { data.append(contentsOf: $0) } }
        append(UInt64(1))                       // one image
        append(Int32(1))                        // IMAGE_ID
        for v in [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] { append(v) }
        append(Int32(1))                        // CAMERA_ID
        data.append(contentsOf: Array("a.png".utf8)); data.append(0)
        append(UInt64(Int.max))                 // × 24 overflows
        try data.write(to: tmp.appendingPathComponent("images.bin"))

        XCTAssertThrowsError(try ColmapSparse.readImages(in: tmp, format: .binary)) { error in
            XCTAssertEqual(error as? ColmapSparseError, .truncated("images.bin"))
        }
    }

    // MARK: - CRLF

    func testACrlfTextModelReadsExactlyLikeAnLfOne() throws {
        // 2026-08-15 FIX 2. `physicalLines` split on the literal `"\n"` and stripped a trailing
        // `"\r"` afterwards, which looks like CRLF handling and is not: `"\r\n"` is a single
        // grapheme cluster in Swift and equals neither `"\n"` nor `"\r"`, so the split found no
        // separator at all. A model exported on Windows arrived as one `#`-prefixed line and was
        // refused with "sparse/0/cameras.txt holds no records".
        //
        // Two images, because the other half of the claim is the blank POINTS2D line: CRLF has to
        // count as one separator, or the pose/observation pairing shifts.
        let files = [
            "cameras.txt": "# Camera list\r\n"
                + "1 OPENCV 320 240 260 260 160 120 -0.15 0.05 0.0012 -0.0009\r\n",
            "images.txt": "# Image list\r\n7 1 0 0 0 0.5 -1 2 1 a.png\r\n\r\n"
                + "9 1 0 0 0 0 0 0 1 b.png\r\n\r\n",
            "points3D.txt": "# 3D point list\r\n1 0.5 -1.25 3 255 0 17 0.5\r\n",
        ]
        for (name, text) in files {
            try Data(text.utf8).write(to: tmp.appendingPathComponent(name))
        }

        let cameras = try ColmapSparse.readCameras(in: tmp, format: .text)
        XCTAssertEqual(cameras.count, 1)
        XCTAssertEqual(cameras[0].modelId, 4)
        // The last parameter is the token the `\r` used to be glued to.
        XCTAssertEqual(cameras[0].params.last, -0.0009)

        let images = try ColmapSparse.readImages(in: tmp, format: .text)
        XCTAssertEqual(images.map(\.name), ["a.png", "b.png"])
        XCTAssertEqual(images.map(\.id), [7, 9])
        XCTAssertEqual(try ColmapSparse.readPoints(in: tmp, format: .text),
                       [ColmapPointRecord(x: 0.5, y: -1.25, z: 3, r: 255, g: 0, b: 17)])
    }

    // MARK: - points3D.txt is written in chunks

    func testPointsAreWrittenIdenticallyAcrossTheFlushBoundary() throws {
        // The writer streams `points3D.txt` instead of assembling it, because that is the one file
        // that scales with the reconstruction (millions of points on a dense solve, ~0.87 GB of
        // resident string at 3 M). Chunking is where an off-by-one loses or repeats a line at the
        // seam, so the fixture is deliberately larger than one chunk and not a multiple of it.
        let count = 8192 * 2 + 7
        var points: [ColmapPointRecord] = []
        points.reserveCapacity(count)
        for i in 0..<count {
            points.append(ColmapPointRecord(x: Double(i) * 0.5, y: -Double(i), z: 1e-8 * Double(i),
                                            r: UInt8(i % 256), g: UInt8((i / 2) % 256),
                                            b: UInt8((i / 3) % 256)))
        }
        try ColmapSparse.writeText(cameras: [ColmapPinholeCamera(id: 1, width: 320, height: 240,
                                                                 fx: 260, fy: 260,
                                                                 cx: 160, cy: 120)],
                                   images: [ColmapImageRecord(id: 1, qw: 1, qx: 0, qy: 0, qz: 0,
                                                              tx: 0, ty: 0, tz: 0,
                                                              cameraId: 1, name: "a.png")],
                                   points: points, to: tmp)

        let text = try String(contentsOf: tmp.appendingPathComponent("points3D.txt"),
                              encoding: .utf8)
        let lines = text.split(separator: "\n").filter { !$0.hasPrefix("#") }
        XCTAssertEqual(lines.count, count)
        // Ids are 1-based and consecutive across every seam, and the values are the ones handed in.
        for i in [0, 8191, 8192, 8193, 16383, 16384, count - 1] {
            let tokens = lines[i].split(whereSeparator: \.isWhitespace).map(String.init)
            XCTAssertEqual(tokens.count, 8, "line \(i)")
            XCTAssertEqual(Int(tokens[0]), i + 1, "line \(i) POINT3D_ID")
            XCTAssertEqual(Double(tokens[1]), points[i].x, "line \(i) X")
            XCTAssertEqual(Double(tokens[2]), points[i].y, "line \(i) Y")
            XCTAssertEqual(Double(tokens[3]), points[i].z, "line \(i) Z")
            XCTAssertEqual(UInt8(tokens[4]), points[i].r, "line \(i) R")
            XCTAssertEqual(UInt8(tokens[5]), points[i].g, "line \(i) G")
            XCTAssertEqual(UInt8(tokens[6]), points[i].b, "line \(i) B")
        }
        // Nothing of the staging file survives.
        XCTAssertEqual(((try? FileManager.default.contentsOfDirectory(atPath: tmp.path)) ?? [])
            .sorted(), ["cameras.txt", "images.txt", "points3D.txt"])
    }

    func testAnEmptyPointSetStillWritesItsHeader() throws {
        // A reconstruction whose points were stripped is still a set of usable poses, and the core
        // accepts an empty `points3D.txt` — but the file has to exist, since the loader opens all
        // three unconditionally.
        try ColmapSparse.writeText(cameras: [ColmapPinholeCamera(id: 1, width: 320, height: 240,
                                                                 fx: 260, fy: 260,
                                                                 cx: 160, cy: 120)],
                                   images: [ColmapImageRecord(id: 1, qw: 1, qx: 0, qy: 0, qz: 0,
                                                              tx: 0, ty: 0, tz: 0,
                                                              cameraId: 1, name: "a.png")],
                                   points: [], to: tmp)
        let text = try String(contentsOf: tmp.appendingPathComponent("points3D.txt"),
                              encoding: .utf8)
        XCTAssertTrue(text.hasPrefix("# 3D point list:"), text)
        XCTAssertEqual(try ColmapSparse.readPoints(in: tmp, format: .text), [])
    }
}
