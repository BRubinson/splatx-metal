import XCTest
@testable import SplatXMetalTrainer

/// A directory tree that exists only as a dictionary.
///
/// `CaptureBundle.locate` takes its filesystem access as parameters precisely so recognition can be
/// pinned without touching a disk: every shape below (a wrapper with no `images/`, two candidate
/// directories, a *file* named `images`) would otherwise need a temp tree, and none of them may be
/// built anywhere near the user's real folders — an XCTest host reading ~/Desktop or ~/Downloads
/// hangs on this machine's TCC prompt. The base path is synthetic and never created.
private final class FakeTree {
    let base: URL
    private var directories: Set<String>
    private var childNames: [String: [String]] = [:]
    private var fileData: [String: Data] = [:]

    init(base: URL = URL(fileURLWithPath: "/synthetic/selected", isDirectory: true)) {
        self.base = base
        directories = [base.path]
    }

    /// Registers a directory at `relative`, creating the intermediate directories.
    @discardableResult
    func dir(_ relative: String) -> FakeTree { add(relative, isDirectory: true, data: nil) }

    /// Registers a file. The default payload keeps image entries non-empty without meaning
    /// anything — nothing in `CaptureBundle` reads image bytes.
    @discardableResult
    func file(_ relative: String, _ data: Data = Data("x".utf8)) -> FakeTree {
        add(relative, isDirectory: false, data: data)
    }

    @discardableResult
    func file(_ relative: String, text: String) -> FakeTree {
        add(relative, isDirectory: false, data: Data(text.utf8))
    }

    /// A complete bundle: `images/` with `count` photos plus the always-present side files that the
    /// importer is expected to ignore.
    @discardableResult
    func capture(_ prefix: String, images count: Int = 2) -> FakeTree {
        let dot = prefix.isEmpty ? "" : prefix + "/"
        for i in 1...max(count, 1) where count > 0 {
            file(dot + "images/" + String(format: "%04d.jpg", i))
        }
        if count == 0 { dir(dot + "images") }
        file(dot + "events.ndjson")
        file(dot + "README.txt")
        return self
    }

    /// The LiDAR extras — a complete txt model plus the point cloud.
    @discardableResult
    func posedModel(_ prefix: String) -> FakeTree {
        let dot = prefix.isEmpty ? "" : prefix + "/"
        for f in ["cameras.txt", "images.txt", "points3D.txt"] { file(dot + "sparse/0/" + f) }
        return file(dot + "pointcloud.ply")
    }

    @discardableResult
    private func add(_ relative: String, isDirectory: Bool, data: Data?) -> FakeTree {
        var url = base
        let components = relative.split(separator: "/").map(String.init)
        for (i, name) in components.enumerated() {
            let parent = url.path
            url = url.appendingPathComponent(name)
            if !(childNames[parent]?.contains(name) ?? false) {
                childNames[parent, default: []].append(name)
            }
            if i < components.count - 1 || isDirectory { directories.insert(url.path) }
        }
        if let data { fileData[url.path] = data }
        return self
    }

    var isDirectory: (URL) -> Bool { { [self] in directories.contains($0.path) } }
    var contents: (URL) -> [String] { { [self] in childNames[$0.path] ?? [] } }
    var manifest: (URL) -> Data? { { [self] in fileData[$0.path] } }
}

final class CaptureBundleTests: XCTestCase {
    private func locate(_ tree: FakeTree, at url: URL? = nil) throws -> CaptureBundleLayout {
        try CaptureBundle.locate(at: url ?? tree.base,
                                 isDirectory: tree.isDirectory,
                                 contents: tree.contents,
                                 readManifest: tree.manifest)
    }

    private func manifest(mode: String) -> String {
        #"{"schema":"splatx.capture","mode":"\#(mode)"}"#
    }

    // MARK: Root resolution

    func testSelectionItselfIsTheRootWhenItHasImages() throws {
        let tree = FakeTree().capture("").file("splatx.json", text: manifest(mode: "photo"))
        let layout = try locate(tree)
        XCTAssertEqual(layout.root, tree.base)
        XCTAssertEqual(layout.kind, .unposed)
        XCTAssertEqual(layout.mode, "photo")
    }

    func testSelectionItselfIsTheRootWhenPosed() throws {
        let tree = FakeTree().capture("").posedModel("")
            .file("splatx.json", text: manifest(mode: "lidar"))
        let layout = try locate(tree)
        XCTAssertEqual(layout.root, tree.base)
        XCTAssertEqual(layout.kind, .posed)
        XCTAssertEqual(layout.mode, "lidar")
    }

    func testDescendsThroughTheZipWrapperDirectory() throws {
        let tree = FakeTree().capture("Photo_20260806_135343")
            .file("Photo_20260806_135343/splatx.json", text: manifest(mode: "video"))
        let layout = try locate(tree)
        XCTAssertEqual(layout.root,
                       tree.base.appendingPathComponent("Photo_20260806_135343", isDirectory: true))
        XCTAssertEqual(layout.kind, .unposed)
        XCTAssertEqual(layout.mode, "video")
    }

    func testDescendsThroughTheZipWrapperDirectoryWhenPosed() throws {
        let tree = FakeTree().capture("LiDAR_20260804_153039").posedModel("LiDAR_20260804_153039")
        let layout = try locate(tree)
        XCTAssertEqual(layout.root,
                       tree.base.appendingPathComponent("LiDAR_20260804_153039", isDirectory: true))
        XCTAssertEqual(layout.kind, .posed)
    }

    /// Finder's "Compress"/expand round trip leaves a `__MACOSX` sibling of the wrapper. It must
    /// not count as a second candidate, or every re-compressed bundle would report ambiguity.
    func testMacOSXShadowTreeIsNotACandidateDirectory() throws {
        let tree = FakeTree().capture("LiDAR_20260804_153039")
            .file("__MACOSX/._LiDAR_20260804_153039")
            .file("__MACOSX/LiDAR_20260804_153039/._images")
        let layout = try locate(tree)
        XCTAssertEqual(layout.root,
                       tree.base.appendingPathComponent("LiDAR_20260804_153039", isDirectory: true))
    }

    func testDotPrefixedDirectoriesAreNotCandidates() throws {
        let tree = FakeTree().capture("Photo_1").dir(".Trashes").dir(".fseventsd")
        XCTAssertEqual(try locate(tree).root,
                       tree.base.appendingPathComponent("Photo_1", isDirectory: true))
    }

    func testTwoCandidateDirectoriesAreAmbiguous() throws {
        let tree = FakeTree().capture("Photo_B").capture("Photo_A")
        XCTAssertThrowsError(try locate(tree)) { e in
            // Sorted, so the message does not depend on enumeration order.
            XCTAssertEqual(e as? CaptureBundleError, .ambiguousRoot(["Photo_A", "Photo_B"]))
        }
    }

    func testNoDirectoriesAtAllNamesTheSelection() throws {
        let tree = FakeTree().file("splatx.json", text: manifest(mode: "photo")).file("README.txt")
        XCTAssertThrowsError(try locate(tree)) { e in
            XCTAssertEqual(e as? CaptureBundleError, .missingImagesDirectory("selected"))
        }
    }

    func testSingleWrapperWithoutImagesIsRejected() throws {
        let tree = FakeTree().file("Something/notes.txt")
        XCTAssertThrowsError(try locate(tree)) { e in
            XCTAssertEqual(e as? CaptureBundleError, .missingImagesDirectory("Something"))
        }
    }

    /// A *file* named `images` must not be mistaken for the images folder — the descent has to
    /// continue. `isDirectory` is what separates the two, which is why it is injected separately
    /// from the listing.
    func testFileNamedImagesDoesNotStopTheDescent() throws {
        let tree = FakeTree().capture("Photo_1").file("images")
        XCTAssertEqual(try locate(tree).root,
                       tree.base.appendingPathComponent("Photo_1", isDirectory: true))
    }

    /// A real bundle root holds `sparse/` too, and after on-device training a `trained/` subtree as
    /// well (spec §1.2). Checking `images/` before the wrapper rule is what keeps those from
    /// reading as ambiguity.
    func testExtraDirectoriesInTheRootAreNotAmbiguity() throws {
        let tree = FakeTree().capture("").posedModel("").file("trained/step_7000.ply")
        let layout = try locate(tree)
        XCTAssertEqual(layout.root, tree.base)
        XCTAssertEqual(layout.kind, .posed)
    }

    // MARK: posed / unposed

    func testBinModelIsPosed() throws {
        let tree = FakeTree().capture("")
        for f in ["cameras.bin", "images.bin", "points3D.bin"] { tree.file("sparse/0/" + f) }
        XCTAssertEqual(try locate(tree).kind, .posed)
    }

    func testPartialModelFallsBackToUnposed() throws {
        let tree = FakeTree().capture("").file("sparse/0/cameras.txt")
        XCTAssertEqual(try locate(tree).kind, .unposed)
    }

    /// bin and txt halves of two different sets do not add up to a loadable model.
    func testMixedBinAndTxtModelIsUnposed() throws {
        let tree = FakeTree().capture("")
            .file("sparse/0/cameras.bin").file("sparse/0/images.txt").file("sparse/0/points3D.txt")
        XCTAssertEqual(try locate(tree).kind, .unposed)
    }

    func testMissingSparseDirectoryIsUnposed() throws {
        XCTAssertEqual(try locate(FakeTree().capture("")).kind, .unposed)
    }

    // MARK: splatx.json is a label, never a gate

    /// The manifest is SplatX-internal and explicitly not part of the Studio contract, so its
    /// absence is an ordinary success — including for a LiDAR bundle, where the interesting
    /// decision (skip alignment) still has to be made correctly without it.
    func testMissingManifestStillImports() throws {
        let layout = try locate(FakeTree().capture("").posedModel(""))
        XCTAssertEqual(layout.kind, .posed)
        XCTAssertNil(layout.mode)
    }

    func testBrokenManifestStillImports() throws {
        let tree = FakeTree().capture("").file("splatx.json", text: #"{"mode": "pho"#)
        let layout = try locate(tree)
        XCTAssertEqual(layout.kind, .unposed)
        XCTAssertNil(layout.mode)
    }

    /// The single test that pins the top constraint of this round in both directions: a manifest
    /// claiming `lidar` over a tree with no model is `.unposed`, and one claiming `photo` over a
    /// complete model is `.posed`. Structure decides; the string is a caption.
    func testModeStringNeverOverridesTheStructure() throws {
        let lying = FakeTree().capture("").file("splatx.json", text: manifest(mode: "lidar"))
        XCTAssertEqual(try locate(lying).kind, .unposed)

        let other = FakeTree().capture("").posedModel("")
            .file("splatx.json", text: manifest(mode: "photo"))
        let layout = try locate(other)
        XCTAssertEqual(layout.kind, .posed)
        XCTAssertEqual(layout.mode, "photo")
    }

    func testManifestModeVariants() {
        XCTAssertNil(CaptureBundle.manifestMode(nil))
        XCTAssertNil(CaptureBundle.manifestMode(Data("[1,2,3]".utf8)))          // not an object
        XCTAssertNil(CaptureBundle.manifestMode(Data(#"{"schema":"x"}"#.utf8))) // no mode
        XCTAssertNil(CaptureBundle.manifestMode(Data(#"{"mode":3}"#.utf8)))     // not a string
        XCTAssertNil(CaptureBundle.manifestMode(Data(#"{"mode":"  "}"#.utf8)))  // blank
        // Unknown values pass through unchanged: modes may be added by the app at any time, and a
        // caption has nothing to validate against.
        XCTAssertEqual(CaptureBundle.manifestMode(Data(#"{"mode":"gaussian"}"#.utf8)), "gaussian")
        XCTAssertEqual(CaptureBundle.manifestMode(Data(#"{"mode":" lidar\n"}"#.utf8)), "lidar")
    }

    // MARK: image count

    func testFewerThanTwoImagesFailsLikeEveryOtherImport() throws {
        let tree = FakeTree().capture("", images: 1).posedModel("")
        XCTAssertThrowsError(try locate(tree)) { e in
            XCTAssertEqual(e as? ImportError, .notEnoughImages(found: 1))
        }
    }

    func testEmptyImagesDirectoryReportsZero() throws {
        let tree = FakeTree().capture("", images: 0)
        XCTAssertThrowsError(try locate(tree)) { e in
            XCTAssertEqual(e as? ImportError, .notEnoughImages(found: 0))
        }
    }

    /// AppleDouble sidecars carry the image extension. Counting `._0001.jpg` as a photo would let
    /// a one-image bundle through the floor and fail later, inside the loader.
    func testAppleDoubleSidecarsAndNonImagesAreNotCounted() {
        XCTAssertEqual(
            CaptureBundle.imageNames(among: ["0002.jpg", "._0001.jpg", "0001.jpg", ".DS_Store",
                                             "notes.txt", "0003.HEIC"]),
            ["0001.jpg", "0002.jpg", "0003.HEIC"])
    }

    func testSidecarsAloneDoNotSatisfyTheImageFloor() throws {
        let tree = FakeTree()
        tree.file("images/0001.jpg").file("images/._0001.jpg").file("images/._0002.jpg")
        XCTAssertThrowsError(try locate(tree)) { e in
            XCTAssertEqual(e as? ImportError, .notEnoughImages(found: 1))
        }
    }
}
