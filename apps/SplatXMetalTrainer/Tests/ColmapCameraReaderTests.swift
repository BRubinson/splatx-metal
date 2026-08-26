// Deleted in the first restructuring and revived for the camera fit (2026-07-20 spec §6)
import XCTest
import simd
@testable import SplatXMetalTrainer

final class ColmapCameraReaderTests: XCTestCase {
    func testReadBinFixture() throws {
        let model = try ColmapCameraReader.read(sparseDir: RepoFixtures.colmapMiniSparse)
        // Camera 1: PINHOLE 1920x1440 fx1500 fy1510 cx960 cy720
        let cam1 = try XCTUnwrap(model.cameras[1])
        XCTAssertEqual(cam1.fx, 1500, accuracy: 1e-6); XCTAssertEqual(cam1.fy, 1510, accuracy: 1e-6)
        XCTAssertEqual(cam1.cx, 960, accuracy: 1e-6);  XCTAssertEqual(cam1.cy, 720, accuracy: 1e-6)
        XCTAssertEqual(cam1.width, 1920); XCTAssertEqual(cam1.height, 1440)
        // Camera 2: SIMPLE_PINHOLE f500 → fx=fy=500
        let cam2 = try XCTUnwrap(model.cameras[2])
        XCTAssertEqual(cam2.fx, 500, accuracy: 1e-6); XCTAssertEqual(cam2.fy, 500, accuracy: 1e-6)
        // Image 1
        let v0 = try XCTUnwrap(model.views.first { $0.name == "frame_000.jpg" })
        XCTAssertEqual(v0.cameraId, 1)
        XCTAssertEqual(v0.qvec.x, 1, accuracy: 1e-9)   // (w,x,y,z)=(1,0,0,0)
        XCTAssertEqual(v0.tvec, SIMD3(0.1, -0.2, 3.0))
    }
    func testTxtMatchesBin() throws {
        // Writes an inline txt fixture into a temporary directory and parses it.
        let dir = FileManager.default.temporaryDirectory.appending(path: "cmtxt-\(UUID().uuidString)/sparse/0")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let cameras = """
        # Camera list
        1 PINHOLE 1920 1440 1500 1510 960 720
        """
        let images = """
        # Image list
        1 1 0 0 0 0.1 -0.2 3.0 1 frame_000.jpg

        """
        try cameras.write(to: dir.appending(path: "cameras.txt"), atomically: true, encoding: .utf8)
        try images.write(to: dir.appending(path: "images.txt"), atomically: true, encoding: .utf8)
        let model = try ColmapCameraReader.read(sparseDir: dir)
        let cam1 = try XCTUnwrap(model.cameras[1])
        XCTAssertEqual(cam1.fx, 1500, accuracy: 1e-6)
        let v0 = try XCTUnwrap(model.views.first)
        XCTAssertEqual(v0.name, "frame_000.jpg")
        XCTAssertEqual(v0.qvec.x, 1, accuracy: 1e-9)
    }
    func testCrlfLineEndingsParse() throws {
        // 2026-08-15 FIX 2. `trimmingCharacters(in: .whitespaces)` strips space and tab but **not**
        // `\r`, so a model exported on Windows broke this reader in two different ways:
        //   cameras.txt — the last parameter arrived as "720\r", which `Double(_:)` rejects, so a
        //                 PINHOLE row silently lost a parameter and threw "insufficient PINHOLE
        //                 parameters";
        //   images.txt  — the empty POINTS2D line arrived as "\r", which is not empty, so the
        //                 trailing-empty trim stopped early and "\r" was parsed as a pose header
        //                 (one token where ten are required).
        // The core's `raw_line` and `ColmapSparse.physicalLines` both handle CRLF; this reader was
        // the only one in the app that could not open such a model.
        let dir = FileManager.default.temporaryDirectory
            .appending(path: "cmcrlf-\(UUID().uuidString)/sparse/0")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let cameras = "# Camera list\r\n1 PINHOLE 1920 1440 1500 1510 960 720\r\n"
        let images = "# Image list\r\n1 1 0 0 0 0.1 -0.2 3.0 1 frame_000.jpg\r\n\r\n"
        let points = "# 3D point list\r\n1 0.5 -1.25 3 255 0 17 0.5\r\n"
        try Data(cameras.utf8).write(to: dir.appending(path: "cameras.txt"))
        try Data(images.utf8).write(to: dir.appending(path: "images.txt"))
        try Data(points.utf8).write(to: dir.appending(path: "points3D.txt"))

        let model = try ColmapCameraReader.read(sparseDir: dir)
        let cam1 = try XCTUnwrap(model.cameras[1])
        XCTAssertEqual(cam1.fx, 1500, accuracy: 1e-9)
        XCTAssertEqual(cam1.cy, 720, accuracy: 1e-9)   // the token that used to carry the \r
        XCTAssertEqual(model.views.count, 1)
        let v0 = try XCTUnwrap(model.views.first)
        XCTAssertEqual(v0.name, "frame_000.jpg")       // and not "frame_000.jpg\r"
        XCTAssertEqual(v0.tvec, SIMD3(0.1, -0.2, 3.0))
        XCTAssertEqual(try ColmapCameraReader.loadPoints(sparse0: dir),
                       [SIMD3<Float>(0.5, -1.25, 3)])
    }

    func testTxtPairwiseSkipsPointsLine() throws {
        // A real COLMAP images.txt has two lines per image (a header plus the 2D
        // observations). The observation line has ≥10 tokens with an integer first token, so a
        // token-count heuristic (count>=10 && Int(t[0])) mistakes it for an image. Only
        // pairwise parsing (2 lines per image) skips it correctly — a fixture with
        // discriminating power.
        let dir = FileManager.default.temporaryDirectory.appending(path: "cmpair-\(UUID().uuidString)/sparse/0")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let cameras = """
        # Camera list
        1 PINHOLE 1920 1440 1500 1510 960 720
        """
        let images = """
        # Image list
        1 1 0 0 0 0.1 -0.2 3.0 1 frame_000.jpg
        512 340 1 600 200 2 700 300 3 800 400 5
        """
        try cameras.write(to: dir.appending(path: "cameras.txt"), atomically: true, encoding: .utf8)
        try images.write(to: dir.appending(path: "images.txt"), atomically: true, encoding: .utf8)
        let model = try ColmapCameraReader.read(sparseDir: dir)
        // The observation line is not mistaken for an image, so exactly one is parsed.
        XCTAssertEqual(model.views.count, 1)
        let v0 = try XCTUnwrap(model.views.first)
        XCTAssertEqual(v0.name, "frame_000.jpg")
        XCTAssertEqual(v0.cameraId, 1)
    }
}
