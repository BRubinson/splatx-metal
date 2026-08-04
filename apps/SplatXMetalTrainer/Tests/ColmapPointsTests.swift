import XCTest
import simd
@testable import SplatXMetalTrainer

/// Tests for parsing points3D positions (spec §6 — for the camera fit centroid) and for
/// loading ColmapContext.
/// The revived file (ColmapCameraReaderTests) is left untouched as evidence of no change, and
/// the new surface is verified here.
final class ColmapPointsTests: XCTestCase {
    // colmap_mini/points3D.bin: (1,2,3), (-1,0.5,2), and (0,0,5) — tools/gen_colmap_fixture.py
    // is the authoritative producer.
    func testLoadPointsBinFixture() throws {
        let pts = try ColmapCameraReader.loadPoints(sparse0: RepoFixtures.colmapMiniSparse)
        XCTAssertEqual(pts.count, 3)
        XCTAssertEqual(simd_length(pts[0] - SIMD3<Float>(1, 2, 3)), 0, accuracy: 1e-6)
        XCTAssertEqual(simd_length(pts[1] - SIMD3<Float>(-1, 0.5, 2)), 0, accuracy: 1e-6)
        XCTAssertEqual(simd_length(pts[2] - SIMD3<Float>(0, 0, 5)), 0, accuracy: 1e-6)
    }

    // txt: only tokens 2–4 (X Y Z) — both the ColmapTextWriter output format (no track) and
    // the standard format (with a track).
    func testLoadPointsTxt() throws {
        let dir = FileManager.default.temporaryDirectory
            .appending(path: "cmpts-\(UUID().uuidString)/sparse/0")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let txt = """
        # 3D point list: POINT3D_ID X Y Z R G B ERROR TRACK[]
        1 1.0 2.0 3.0 255 0 0 0
        2 -1.0 0.5 2.0 0 255 0 0.5 1 0 2 3
        """
        try txt.write(to: dir.appending(path: "points3D.txt"), atomically: true, encoding: .utf8)
        let pts = try ColmapCameraReader.loadPoints(sparse0: dir)
        XCTAssertEqual(pts.count, 2)
        XCTAssertEqual(simd_length(pts[0] - SIMD3<Float>(1, 2, 3)), 0, accuracy: 1e-6)
        XCTAssertEqual(simd_length(pts[1] - SIMD3<Float>(-1, 0.5, 2)), 0, accuracy: 1e-6)
    }

    // ColmapContext: the model + the centroid + middleView (the middle after sorting by image
    // name = index 1 of the 2 views).
    func testColmapContextLoadFromFixture() throws {
        let ctx = try XCTUnwrap(ColmapContext.load(sparse0: RepoFixtures.colmapMiniSparse))
        XCTAssertEqual(ctx.model.views.count, 2)
        XCTAssertEqual(ctx.middleView.name, "frame_001.jpg")   // sorted[2/2]
        let centroid = try XCTUnwrap(ctx.pointsCentroid)
        XCTAssertEqual(centroid.x, 0, accuracy: 1e-5)
        XCTAssertEqual(centroid.y, 2.5 / 3, accuracy: 1e-5)
        XCTAssertEqual(centroid.z, 10.0 / 3, accuracy: 1e-5)
    }

    // No sparse/0 → the nil fallback (the contract that the existing frameToFit is kept).
    func testColmapContextLoadMissingDirReturnsNil() {
        let missing = FileManager.default.temporaryDirectory
            .appending(path: "cmctx-missing-\(UUID().uuidString)/sparse/0")
        XCTAssertNil(ColmapContext.load(sparse0: missing))
    }
}
