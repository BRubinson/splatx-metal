import XCTest
@testable import SplatXMetalTrainer

@MainActor
final class ProjectStoreTests: XCTestCase {
    private var root: URL!

    override func setUp() async throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("sx-store-\(UUID().uuidString)", isDirectory: true)
    }
    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: root)
    }

    func testCreateAssemblesAtomicallyAndScansBack() throws {
        let store = ProjectStore(rootURL: root)
        let p = try store.create(name: "garden", sourceKind: .images, state: .imported) { dir in
            let images = dir.appendingPathComponent("images", isDirectory: true)
            try FileManager.default.createDirectory(at: images, withIntermediateDirectories: true)
            try Data("x".utf8).write(to: images.appendingPathComponent("a.jpg"))
            return 1
        }
        XCTAssertEqual(p.name, "garden")
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: root.appendingPathComponent("garden/images/a.jpg").path))
        // No leftover temporary assembly directory
        let leftovers = (try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []
        XCTAssertEqual(leftovers.filter { $0.hasPrefix(".tmp-") }, [])
        // Rescanning with a new store restores it too
        let store2 = ProjectStore(rootURL: root)
        store2.scan()
        XCTAssertEqual(store2.projects.map(\.name), ["garden"])
    }

    func testCreateFailurePropagatesAndCleansUp() {
        let store = ProjectStore(rootURL: root)
        struct Boom: Error {}
        XCTAssertThrowsError(try store.create(name: "x", sourceKind: .images, state: .imported) { _ in
            throw Boom()
        })
        let leftovers = (try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []
        XCTAssertEqual(leftovers.filter { $0.hasPrefix(".tmp-") }, [])
        XCTAssertTrue(store.projects.isEmpty)
    }

    func testDuplicateNameGetsSuffix() throws {
        let store = ProjectStore(rootURL: root)
        _ = try store.create(name: "scene", sourceKind: .images, state: .imported) { _ in 2 }
        let p2 = try store.create(name: "scene", sourceKind: .images, state: .imported) { _ in 2 }
        XCTAssertEqual(p2.name, "scene-2")
    }

    func testScanSkipsCorruptManifestWithWarning() throws {
        let store = ProjectStore(rootURL: root)
        _ = try store.create(name: "good", sourceKind: .colmap, state: .aligned) { _ in 3 }
        let bad = root.appendingPathComponent("bad", isDirectory: true)
        try FileManager.default.createDirectory(at: bad, withIntermediateDirectories: true)
        try Data("not json".utf8).write(to: bad.appendingPathComponent(ProjectStore.manifestName))
        store.scan()
        XCTAssertEqual(store.projects.map(\.name), ["good"])
        XCTAssertEqual(store.scanWarnings.count, 1)
    }

    func testDeleteAndRename() throws {
        let store = ProjectStore(rootURL: root)
        let p = try store.create(name: "a", sourceKind: .images, state: .imported) { _ in 2 }
        let renamed = try store.rename(p, to: "b")
        XCTAssertEqual(renamed.name, "b")
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("b").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("a").path))
        try store.delete(renamed)
        XCTAssertTrue(store.projects.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("b").path))
    }
}
