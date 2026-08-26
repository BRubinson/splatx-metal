import XCTest
import simd
@testable import SplatXMetalTrainer

/// 2026-08-04 spec §2.1: the training viewpoint policy — **the result presentation path arms
/// the first-camera anchor, the history path does not.**
/// The other writer (TrainViewModel.start, which arms the anchor on the line before its own
/// applyCameraFit) needs a dataset, a Metal device and a minutes-long blocking run, so under
/// this app's policy it is covered by "build success + manual check M3" rather than a unit
/// test. What is unit-testable — and what a regression would actually break — is which of the
/// two presentation entry points arms the anchor, and that is pinned here.
@MainActor
final class TrainStartViewpointTests: XCTestCase {

    private func makeSession() -> ProjectSession {
        let dir = FileManager.default.temporaryDirectory
            .appending(path: "sx-startcam-\(UUID().uuidString)", directoryHint: .isDirectory)
        let project = Project(id: UUID(), name: "p", sourceKind: .images, state: .trained,
                              imageCount: 3, alignSummary: nil, runs: [], createdAt: Date())
        return ProjectSession(project: project, directory: dir)
    }

    private func makeRecord() -> RunRecord {
        RunRecord(id: "20260804-000000", outputFile: "output.ply", previewFile: nil,
                  outcome: "completed", finalPSNR: nil, splatCount: 0,
                  optionsSummary: "-", createdAt: Date())
    }

    /// The result of a run that just finished is presented from the capture's first camera.
    /// The assertion is synchronous: showRun's unstructured Task (the actual load) is only
    /// enqueued here, so the anchor is still armed and waiting for the applyCameraFit at the
    /// tail of openRun.
    func testShowFinishedRunArmsFirstCameraAnchor() {
        let session = makeSession()
        XCTAssertEqual(session.viewerVM.nextFitAnchor, .middleOrbit)
        session.showFinishedRun(makeRecord())
        XCTAssertEqual(session.viewerVM.nextFitAnchor, .firstCameraExact)
        XCTAssertEqual(session.tab, .view)                    // showRun's existing behavior
        XCTAssertEqual(session.viewerVM.currentRun?.id, "20260804-000000")
    }

    /// Opening a past run from the RUNS history goes through plain showRun and MUST keep the
    /// previous middle-camera behavior — the anchor is never armed on that path.
    func testShowRunKeepsMiddleOrbitAnchor() {
        let session = makeSession()
        session.showRun(makeRecord())
        XCTAssertEqual(session.viewerVM.nextFitAnchor, .middleOrbit)
        XCTAssertEqual(session.tab, .view)
    }
}
