import XCTest
@testable import SplatXMetalTrainer

/// FIX 1/2 (audit review of Task 4) — `ProjectSession.setAutoLevelOnAlign(_:model:)` is the single
/// place that mutates `project.autoLevelOnAlign` and saves the manifest; `InspectorView`'s Toggle
/// only calls it. This covers the failure-surfacing contract (Important 1): a save failure must not
/// be console-only (the console starts collapsed, AppModel.consoleExpanded = false) — it has to
/// reach `alignVM.errorMessage`, which AlignView renders as a visible red banner.
///
/// The success path (mutate + save succeeds) is intentionally not exercised here: AppModel.store
/// has no rootURL injection point (unlike the standalone ProjectStore used by ProjectStoreTests), so
/// a real save would write under the developer's actual ~/SplatXMetalTrainer/Projects — every other
/// AppModel.store-dependent path in this app (AlignViewModel.start, TrainViewModel, deleteRun) is
/// left to build-success + manual check for the same reason. A project name that was never created
/// on disk deterministically fails `store.save` without touching any real file, which is exactly
/// the path under test.
@MainActor
final class ProjectSessionAutoLevelTests: XCTestCase {
    func testSaveFailureSurfacesThroughAlignErrorMessageAndConsole() {
        let model = AppModel()
        // A random, never-created project name — store.directory(for:) points at a folder that
        // does not exist, so writeManifest's Data.write(to:) fails before touching any real file.
        let project = Project(id: UUID(), name: "sx-autolevel-fix-\(UUID().uuidString)",
                              sourceKind: .images, state: .aligned, imageCount: 3,
                              alignSummary: nil, runs: [], createdAt: Date())
        let session = ProjectSession(project: project, directory: FileManager.default.temporaryDirectory)
        XCTAssertTrue(session.project.autoLevelOnAlign)   // the default (Project.swift)
        XCTAssertNil(session.alignVM.errorMessage)

        session.setAutoLevelOnAlign(false, model: model)

        // The in-memory field commits regardless of the save outcome — the same "mutate first,
        // surface the save failure separately" shape as AlignViewModel.start / TrainViewModel.
        XCTAssertFalse(session.project.autoLevelOnAlign)
        XCTAssertNotNil(session.alignVM.errorMessage)
        XCTAssertTrue(session.alignVM.errorMessage!.contains("restart"))
        XCTAssertTrue(model.console.lines.contains { $0.contains("auto-level setting save failed") })
    }

    /// FIX 6 (final branch review, C1) — the camera-fit cache must not outlive the sparse/0 it
    /// parsed. A re-align rewrites that file mid-session, and with auto-level on (the default) the
    /// rewrite rotates the whole world; a stale context would then feed the *previous* coordinate
    /// system to the `.firstCameraExact` override, the timelapse anchor, and the Cinema trajectory.
    ///
    /// This writes two real bundles into a temp directory and pins the observable consequence: the
    /// second load returns the second bundle only after the cache is dropped. Wiring the drop into
    /// `AlignViewModel.start` itself needs a live PhotogrammetrySession, so that half is manual
    /// check M1/M2 — the same split the rest of the align tests use.
    func testInvalidateColmapContextForcesAReparseAfterSparseIsRewritten() throws {
        let dir = FileManager.default.temporaryDirectory
            .appending(path: "sx-fix6-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let sparse0 = dir.appending(path: "sparse/0", directoryHint: .isDirectory)
        func writeBundle(tx: Double) throws {
            let pose = ColmapPose(qw: 1, qx: 0, qy: 0, qz: 0, tx: tx, ty: 0, tz: 0)
            try ColmapTextWriter.write(
                frames: [AlignedFrame(name: "a.jpg", width: 640, height: 480,
                                      fx: 500, fy: 510, cx: 320, cy: 240, pose: pose)],
                points: [AlignedPoint(x: 0, y: 0, z: 1, r: 1, g: 2, b: 3)],
                to: sparse0)
        }

        let project = Project(id: UUID(), name: "sx-fix6", sourceKind: .images, state: .aligned,
                              imageCount: 1, alignSummary: nil, runs: [], createdAt: Date())
        let session = ProjectSession(project: project, directory: dir)

        try writeBundle(tx: 1)
        let first = try XCTUnwrap(session.loadColmapContext())
        XCTAssertEqual(first.firstView.tvec.x, 1, accuracy: 1e-9)

        // A re-align: same path, different contents.
        try writeBundle(tx: 2)
        // Without the drop the session still answers with the parse of the *old* bundle.
        XCTAssertEqual(try XCTUnwrap(session.loadColmapContext()).firstView.tvec.x, 1, accuracy: 1e-9)

        session.invalidateColmapContext()
        XCTAssertEqual(try XCTUnwrap(session.loadColmapContext()).firstView.tvec.x, 2, accuracy: 1e-9)
    }
}
