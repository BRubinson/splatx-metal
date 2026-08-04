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
