import XCTest
@testable import SplatXMetalTrainer

/// App round 4 spec item 6: the pure logic for syncing the session after a rename. The core
/// contract is that `trainVM.datasetURL` moves to the new path along with it — if it does not,
/// training after the rename looks at the old path, a folder that no longer exists.
@MainActor
final class ProjectSessionRenameTests: XCTestCase {
    func testApplyRenameSyncsDirectoryAndDatasetURL() {
        let oldDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("sx-rename-old", isDirectory: true)
        let project = Project(id: UUID(), name: "old", sourceKind: .images, state: .aligned,
                              imageCount: 3, alignSummary: nil, runs: [], createdAt: Date())
        let session = ProjectSession(project: project, directory: oldDir)
        XCTAssertEqual(session.trainVM.datasetURL, oldDir)   // confirms the premise that init injects it

        var renamed = project
        renamed.name = "new"
        let newDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("sx-rename-new", isDirectory: true)
        session.applyRename(project: renamed, directory: newDir)

        XCTAssertEqual(session.project.name, "new")
        XCTAssertEqual(session.directory, newDir)
        XCTAssertEqual(session.trainVM.datasetURL, newDir)
        XCTAssertEqual(session.tab, .train)                  // the tab and the rest of the state are unchanged
    }
}
