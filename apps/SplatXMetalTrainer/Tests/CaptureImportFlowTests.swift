import XCTest
@testable import SplatXMetalTrainer

/// The capture import entry point: the selection → staged tree → project pipeline
/// (`ImportService.stageCapture` / `adoptStagedCapture`) plus every pure decision the sheet, the
/// Home card and align make from `ProjectSourceKind.capture`.
///
/// The judgement calls are static functions rather than expressions inside a view `body` precisely
/// so they can be asserted here — on this machine a SwiftUI view cannot be driven by a test at all
/// (TCC blocks GUI automation), so anything left inline in a body is untested by construction.
///
/// Everything that touches disk stays under `FileManager.temporaryDirectory`; nothing here can
/// reach the real home, ~/Desktop or ~/Downloads.
final class CaptureImportFlowTests: XCTestCase {
    private var tmp: URL!

    override func setUp() {
        tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("sx-capture-flow-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    }
    override func tearDown() { try? FileManager.default.removeItem(at: tmp) }

    // MARK: Fixtures

    private func write(_ url: URL, _ body: String) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try Data(body.utf8).write(to: url)
    }

    /// A LiDAR bundle folder in the shape the phone writes, with the files the importer must drop
    /// (`events.ndjson`, `README.txt`, an on-device `trained/` run) present so their absence
    /// downstream means something.
    private func makeLidarFolder(at root: URL, manifest: String? = #"{"mode":"lidar"}"#) throws {
        try write(root.appendingPathComponent("images/0001.jpg"), "frame-one")
        try write(root.appendingPathComponent("images/0002.jpg"), "frame-two")
        try write(root.appendingPathComponent("sparse/0/cameras.txt"),
                  ColmapFixture.pinholeCamerasText)
        try write(root.appendingPathComponent("sparse/0/images.txt"), "poses")
        try write(root.appendingPathComponent("sparse/0/points3D.txt"), "points")
        try write(root.appendingPathComponent("pointcloud.ply"), "ply-bytes")
        try write(root.appendingPathComponent("events.ndjson"), "{}")
        try write(root.appendingPathComponent("README.txt"), "readme")
        try write(root.appendingPathComponent("trained/splat.ply"), "on-device result")
        if let manifest { try write(root.appendingPathComponent("splatx.json"), manifest) }
    }

    private func entries(_ url: URL) -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: url.path)) ?? []).sorted()
    }

    // MARK: Staging — folder selection

    func testStagesABundleFolderIntoAProjectShapedTree() async throws {
        let bundle = tmp.appendingPathComponent("LiDAR_20260804_153039", isDirectory: true)
        try makeLidarFolder(at: bundle)
        let workspace = tmp.appendingPathComponent("ws", isDirectory: true)

        let staged = try await ImportService.stageCapture(source: bundle, workspace: workspace)

        XCTAssertEqual(staged.kind, .posed)
        XCTAssertEqual(staged.mode, "lidar")
        XCTAssertEqual(staged.imageCount, 2)
        XCTAssertEqual(entries(staged.directory),
                       ["images", "pointcloud.ply", "sparse", "splatx.json"])
        // A folder selection is never expanded, so no archive scratch directory appears.
        XCTAssertEqual(entries(workspace), ["assembled"])
    }

    /// The rule this round is built on: the manifest is a caption, not a gate (spec §1.1). A bundle
    /// the user assembled by hand — or that a future app version stopped writing — still imports.
    func testStagesSuccessfullyWithNoManifestAtAll() async throws {
        let bundle = tmp.appendingPathComponent("Handmade", isDirectory: true)
        try makeLidarFolder(at: bundle, manifest: nil)
        let staged = try await ImportService.stageCapture(
            source: bundle, workspace: tmp.appendingPathComponent("ws", isDirectory: true))
        XCTAssertNil(staged.mode)
        XCTAssertEqual(staged.kind, .posed)          // decided by sparse/0, not by the missing file
        XCTAssertEqual(staged.imageCount, 2)
    }

    /// A photo/video bundle: no `sparse/`, so it stages `.unposed` and will have to be aligned.
    func testAnUnposedBundleStagesWithoutAModel() async throws {
        let bundle = tmp.appendingPathComponent("Photo_20260806_135343", isDirectory: true)
        try write(bundle.appendingPathComponent("images/0001.jpg"), "a")
        try write(bundle.appendingPathComponent("images/0002.jpg"), "b")
        try write(bundle.appendingPathComponent("splatx.json"), #"{"mode":"photo"}"#)
        let staged = try await ImportService.stageCapture(
            source: bundle, workspace: tmp.appendingPathComponent("ws", isDirectory: true))
        XCTAssertEqual(staged.kind, .unposed)
        XCTAssertEqual(staged.mode, "photo")
        XCTAssertEqual(entries(staged.directory), ["images", "splatx.json"])
    }

    // MARK: Staging — zip selection

    /// The zip branch end to end, including the wrapper directory an iOS export always carries
    /// (`shouldKeepParent: true`): expand → descend → whitelist copy, with the payload arriving
    /// byte-identical after a deflate round trip.
    func testStagesAZipThroughItsWrapperDirectory() async throws {
        var builder = ZipBuilder()
        let prefix = "LiDAR_20260804_153039/"
        let poses = "1 0.7071067811865476 -0.0000000001 0.5 0.5 1 1 0001.jpg \r\n\n"
        builder.entries = [
            ZipBuilder.stored(prefix + "images/0001.jpg", "frame-one"),
            ZipBuilder.deflated(prefix + "images/0002.jpg", Array("frame-two".utf8)),
            ZipBuilder.stored(prefix + "sparse/0/cameras.txt", ColmapFixture.pinholeCamerasText),
            ZipBuilder.deflated(prefix + "sparse/0/images.txt", Array(poses.utf8)),
            ZipBuilder.stored(prefix + "sparse/0/points3D.txt", "points"),
            ZipBuilder.stored(prefix + "pointcloud.ply", "ply-bytes"),
            ZipBuilder.stored(prefix + "splatx.json", #"{"mode":"lidar"}"#),
            ZipBuilder.stored(prefix + "events.ndjson", "{}"),
        ]
        let zip = tmp.appendingPathComponent("LiDAR_20260804_153039.zip")
        try builder.build().write(to: zip)
        let workspace = tmp.appendingPathComponent("ws", isDirectory: true)

        let staged = try await ImportService.stageCapture(source: zip, workspace: workspace)

        XCTAssertEqual(staged.kind, .posed)
        XCTAssertEqual(staged.mode, "lidar")
        XCTAssertEqual(staged.imageCount, 2)
        XCTAssertEqual(entries(staged.directory),
                       ["images", "pointcloud.ply", "sparse", "splatx.json"])
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: staged.directory.appendingPathComponent("events.ndjson").path))
        // The coordinate-preservation guard restated on the zip path (spec §2.3): the model text
        // survives compression and copying unchanged, trailing space and CRLF included.
        XCTAssertEqual(
            try String(contentsOf: staged.directory.appendingPathComponent("sparse/0/images.txt"),
                       encoding: .utf8),
            poses)
    }

    /// The panel's `.zip` filter is a courtesy, not the gate — staging decides on "is it a
    /// directory", so a file that is not an archive fails with the reader's own diagnosis rather
    /// than being half-trusted.
    func testANonArchiveFileFailsAsANonArchive() async throws {
        let notAZip = tmp.appendingPathComponent("notes.zip")
        try write(notAZip, String(repeating: "this is not a zip. ", count: 20))
        do {
            _ = try await ImportService.stageCapture(
                source: notAZip, workspace: tmp.appendingPathComponent("ws", isDirectory: true))
            XCTFail("expected a rejection")
        } catch {
            XCTAssertEqual(error as? ZipArchiveError, .notAZipArchive)
        }
    }

    // MARK: Adoption

    /// Adoption moves rather than copies: the project directory ends up with the whole tree and the
    /// staging area is left empty, so the caller's cleanup has nothing to reclaim on success.
    func testAdoptionMovesTheStagedTreeAndEmptiesTheStagingArea() async throws {
        let bundle = tmp.appendingPathComponent("LiDAR_x", isDirectory: true)
        try makeLidarFolder(at: bundle)
        let staged = try await ImportService.stageCapture(
            source: bundle, workspace: tmp.appendingPathComponent("ws", isDirectory: true))
        let project = tmp.appendingPathComponent("project", isDirectory: true)

        XCTAssertEqual(try ImportService.adoptStagedCapture(staged, into: project), 2)

        XCTAssertEqual(entries(project), ["images", "pointcloud.ply", "sparse", "splatx.json"])
        XCTAssertEqual(entries(project.appendingPathComponent("sparse/0")),
                       ["cameras.txt", "images.txt", "points3D.txt"])
        XCTAssertEqual(entries(staged.directory), [])
        XCTAssertEqual(try Data(contentsOf: project.appendingPathComponent("pointcloud.ply")),
                       Data("ply-bytes".utf8))
    }

    // MARK: Sheet decisions

    func testPosedOpensAlignedAndUnposedOpensImported() {
        XCTAssertEqual(ImportSheet.captureState(for: .posed), .aligned)
        XCTAssertEqual(ImportSheet.captureState(for: .unposed), .imported)
    }

    func testProjectNameStripsOnlyAZipExtension() {
        let name = { (s: String) in ImportSheet.captureProjectName(for: URL(fileURLWithPath: s)) }
        XCTAssertEqual(name("/t/LiDAR_20260804_153039.zip"), "LiDAR_20260804_153039")
        XCTAssertEqual(name("/t/LiDAR_20260804_153039.ZIP"), "LiDAR_20260804_153039")
        // A selected folder keeps its name whole — including one that merely contains dots.
        XCTAssertEqual(name("/t/Photo_20260806_135343"), "Photo_20260806_135343")
        XCTAssertEqual(name("/t/capture v1.2"), "capture v1.2")
    }

    /// The mode is printed, never compared: an unknown string passes straight through, and a bundle
    /// without a manifest simply loses the word while the structural half of the sentence stands.
    func testConsoleLineLabelsTheModeAndReportsTheStructuralOutcome() {
        XCTAssertEqual(ImportSheet.captureConsoleLine(mode: "lidar", kind: .posed),
                       "· SplatX capture · lidar · on-device poses in use — alignment not needed")
        XCTAssertEqual(ImportSheet.captureConsoleLine(mode: "photo", kind: .unposed),
                       "· SplatX capture · photo · alignment required")
        XCTAssertEqual(ImportSheet.captureConsoleLine(mode: nil, kind: .unposed),
                       "· SplatX capture · alignment required")
        XCTAssertEqual(ImportSheet.captureConsoleLine(mode: "someFutureMode", kind: .posed),
                       "· SplatX capture · someFutureMode"
                       + " · on-device poses in use — alignment not needed")
    }

    /// The dismiss lock keys off `isBusy`, not off equality with one payload — the property exists
    /// so adding a busy kind cannot silently unlock the sheet mid-import.
    func testEveryBusyKindLocksTheSheet() {
        XCTAssertTrue(ImportPhase.busy(.video).isBusy)
        XCTAssertTrue(ImportPhase.busy(.capture).isBusy)
        XCTAssertFalse(ImportPhase.choose.isBusy)
        XCTAssertFalse(ImportPhase.videoOptions(URL(fileURLWithPath: "/t/a.mov")).isBusy)
    }

    // MARK: sourceKind consumers

    /// Sequential ordering is claimed where the frame names were produced in capture order — the
    /// phone names every mode's frames `0001.jpg`, `0002.jpg`, … so `.capture` qualifies alongside
    /// `.video`.
    func testSequentialMatchingIsClaimedOnlyForProducedOrderings() {
        XCTAssertTrue(AlignViewModel.usesSequentialMatching(.video))
        XCTAssertTrue(AlignViewModel.usesSequentialMatching(.capture))
        XCTAssertFalse(AlignViewModel.usesSequentialMatching(.images))
        XCTAssertFalse(AlignViewModel.usesSequentialMatching(.colmap))
    }

    func testHomeCardPlaceholderSymbolPerSourceKind() {
        XCTAssertEqual(ProjectCard.placeholderSymbol(for: .capture), "iphone")
        XCTAssertEqual(ProjectCard.placeholderSymbol(for: .video), "video")
        XCTAssertEqual(ProjectCard.placeholderSymbol(for: .images), "photo.on.rectangle")
        XCTAssertEqual(ProjectCard.placeholderSymbol(for: .colmap), "photo.on.rectangle")
    }

    /// The inspector prints the raw value verbatim, and the manifest stores it — so the spelling is
    /// both a file format and a piece of UI text, and neither may drift.
    func testCaptureSourceKindRoundTripsThroughTheManifest() throws {
        XCTAssertEqual(ProjectSourceKind.capture.rawValue, "capture")
        let project = Project(id: UUID(), name: "LiDAR_20260804_153039", sourceKind: .capture,
                              state: .aligned, imageCount: 47, alignSummary: nil, runs: [],
                              createdAt: Date())
        let data = try JSONEncoder.project.encode(project)
        let decoded = try JSONDecoder.project.decode(Project.self, from: data)
        XCTAssertEqual(decoded.sourceKind, .capture)
        XCTAssertEqual(decoded.state, .aligned)

        // And from a manifest written by hand — the shape a project folder actually holds.
        let json = """
        {"id":"F5B2D3A0-0000-0000-0000-000000000000","name":"c","sourceKind":"capture",
         "state":"aligned","imageCount":47,"runs":[],"createdAt":"2026-08-06T00:00:00Z"}
        """
        XCTAssertEqual(
            try JSONDecoder.project.decode(Project.self, from: Data(json.utf8)).sourceKind,
            .capture)
    }
}
