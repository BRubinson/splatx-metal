import XCTest
@testable import SplatXMetalTrainer

final class ColmapTextWriterTests: XCTestCase {
    private func frame(_ name: String) -> AlignedFrame {
        AlignedFrame(name: name, width: 640, height: 480, fx: 500, fy: 510, cx: 320, cy: 240,
                     pose: ColmapPose(qw: 1, qx: 0, qy: 0, qz: 0, tx: 0.5, ty: -1, tz: 2))
    }

    func testCamerasTextPinholePerFrame() {
        let text = ColmapTextWriter.camerasText(frames: [frame("a.jpg"), frame("b.jpg")])
        let rows = text.split(separator: "\n").filter { !$0.hasPrefix("#") }
        XCTAssertEqual(rows.count, 2)
        XCTAssertEqual(rows[0].split(separator: " ").count, 8)   // ID PINHOLE W H fx fy cx cy
        XCTAssertTrue(rows[0].contains(" PINHOLE 640 480 "))
    }

    func testImagesTextHasTenTokensAndBlankSecondLine() {
        let text = ColmapTextWriter.imagesText(frames: [frame("a.jpg"), frame("b.jpg")])
        // After the comments: [pose line, blank line, pose line, blank line] — the empty
        // POINTS2D line must physically exist.
        var lines = text.components(separatedBy: "\n")
        lines.removeAll { $0.hasPrefix("#") }
        if lines.last == "" { lines.removeLast() }   // the trailing newline at the end of the file
        XCTAssertEqual(lines.count, 4)
        XCTAssertEqual(lines[0].split(separator: " ").count, 10)
        XCTAssertEqual(lines[1], "")
        XCTAssertEqual(lines[2].split(separator: " ").count, 10)
        XCTAssertEqual(lines[3], "")
        XCTAssertTrue(lines[0].hasSuffix(" a.jpg"))
    }

    func testFramesSortedByNameAndCameraIDsMatch() {
        let text = ColmapTextWriter.imagesText(frames: [frame("b.jpg"), frame("a.jpg")])
        let rows = text.components(separatedBy: "\n").filter { !$0.hasPrefix("#") && !$0.isEmpty }
        XCTAssertTrue(rows[0].hasSuffix(" a.jpg"))   // sorting by image name is authoritative
        XCTAssertTrue(rows[1].hasSuffix(" b.jpg"))
        // IMAGE_ID (the first token) and CAMERA_ID (the ninth token) agree, 1-based
        let t0 = rows[0].split(separator: " ")
        XCTAssertEqual(t0.first, "1"); XCTAssertEqual(t0[8], "1")
    }

    func testPoints3DRowsHaveEightTokens() {
        let pts = [AlignedPoint(x: 1, y: 2, z: 3, r: 255, g: 128, b: 0)]
        let text = ColmapTextWriter.points3DText(points: pts)
        let rows = text.split(separator: "\n").filter { !$0.hasPrefix("#") }
        XCTAssertEqual(rows.count, 1)
        let tokens = rows[0].split(separator: " ")
        XCTAssertEqual(tokens.count, 8)              // ID X Y Z R G B ERROR (an empty TRACK is allowed)
        XCTAssertEqual(tokens[4], "255"); XCTAssertEqual(tokens[7], "0")
    }

    func testWriteRejectsSpaceInNameAndEmptyInputs() {
        XCTAssertThrowsError(try ColmapTextWriter.write(
            frames: [frame("has space.jpg")],
            points: [AlignedPoint(x: 0, y: 0, z: 0, r: 1, g: 1, b: 1)],
            to: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)))
        XCTAssertThrowsError(try ColmapTextWriter.write(
            frames: [], points: [AlignedPoint(x: 0, y: 0, z: 0, r: 1, g: 1, b: 1)],
            to: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)))
        XCTAssertThrowsError(try ColmapTextWriter.write(
            frames: [frame("a.jpg")], points: [],
            to: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)))
    }
}
