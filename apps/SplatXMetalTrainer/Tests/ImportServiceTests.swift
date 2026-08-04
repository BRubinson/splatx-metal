import XCTest
@testable import SplatXMetalTrainer

final class ImportServiceTests: XCTestCase {
    private var tmp: URL!

    override func setUp() {
        tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("sx-import-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    }
    override func tearDown() { try? FileManager.default.removeItem(at: tmp) }

    private func touch(_ rel: String) throws {
        let url = tmp.appendingPathComponent(rel)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try Data("x".utf8).write(to: url)
    }

    func testListImagesFiltersAndSortsByName() throws {
        try touch("src/b.PNG"); try touch("src/a.jpg"); try touch("src/c.txt"); try touch("src/d.heic")
        let names = ImportService.listImages(in: tmp.appendingPathComponent("src")).map(\.lastPathComponent)
        XCTAssertEqual(names, ["a.jpg", "b.PNG", "d.heic"])
    }

    func testAssembleImagesRejectsFewerThanTwo() throws {
        try touch("one/a.jpg")
        XCTAssertThrowsError(try ImportService.assembleImages(
            from: tmp.appendingPathComponent("one"), into: tmp.appendingPathComponent("dst"))) { e in
            XCTAssertEqual(e as? ImportError, .notEnoughImages(found: 1))
        }
    }

    func testAssembleImagesCopiesIntoImagesDir() throws {
        try touch("src/a.jpg"); try touch("src/b.jpg")
        let n = try ImportService.assembleImages(from: tmp.appendingPathComponent("src"),
                                                 into: tmp.appendingPathComponent("dst"))
        XCTAssertEqual(n, 2)
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: tmp.appendingPathComponent("dst/images/a.jpg").path))
    }

    func testAssembleColmapRequiresCompleteSet() throws {
        try touch("cm/sparse/0/cameras.txt"); try touch("cm/sparse/0/images.txt")
        // points3D.txt is missing → the error names the missing file
        try touch("cm/images/a.jpg"); try touch("cm/images/b.jpg")
        XCTAssertThrowsError(try ImportService.assembleColmap(
            from: tmp.appendingPathComponent("cm"), into: tmp.appendingPathComponent("dst"))) { e in
            guard case .incompleteColmapSet(let missing) = e as? ImportError else {
                return XCTFail("wrong error: \(e)")
            }
            XCTAssertEqual(missing, "points3D.txt")
        }
    }

    func testAssembleColmapCopiesTxtSetAndImages() throws {
        for f in ["cameras.txt", "images.txt", "points3D.txt"] { try touch("cm/sparse/0/\(f)") }
        try touch("cm/images/a.jpg"); try touch("cm/images/b.jpg")
        let n = try ImportService.assembleColmap(from: tmp.appendingPathComponent("cm"),
                                                 into: tmp.appendingPathComponent("dst"))
        XCTAssertEqual(n, 2)
        for f in ["cameras.txt", "images.txt", "points3D.txt"] {
            XCTAssertTrue(FileManager.default.fileExists(
                atPath: tmp.appendingPathComponent("dst/sparse/0/\(f)").path))
        }
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: tmp.appendingPathComponent("dst/images/b.jpg").path))
    }
}
