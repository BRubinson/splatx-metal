import XCTest
@testable import SplatXMetalTrainer

/// App round 5 spec item 8: the stage-lock decision during long-running work (training and
/// align).
/// There is exactly one authoritative gate, `ProjectSession.lockedTab` — StageNav, ⌥1/2/3, and
/// goHome (back) all consult this decision. Only the pure decision function and the session
/// wiring are verified here (the SwiftUI application surface is manual E2E — the explicit
/// policy of this app).
@MainActor
final class StageLockTests: XCTestCase {
    func testNoWorkNoLock() {
        XCTAssertNil(ProjectSession.lockedTab(isTraining: false, isAligning: false))
    }

    func testTrainingLocksTrain() {
        XCTAssertEqual(ProjectSession.lockedTab(isTraining: true, isAligning: false), .train)
    }

    func testAligningLocksAlign() {
        XCTAssertEqual(ProjectSession.lockedTab(isTraining: false, isAligning: true), .align)
    }

    /// Running both at once is structurally impossible (the locks block each other's start
    /// path), but the decision order is pinned to prefer training — even an unreachable state
    /// must be deterministic.
    func testBothPrefersTraining() {
        XCTAssertEqual(ProjectSession.lockedTab(isTraining: true, isAligning: true), .train)
    }

    /// Session wiring: the VM flags → the `lockedTab` computed property.
    func testSessionWiringReflectsVMFlags() {
        let project = Project(id: UUID(), name: "p", sourceKind: .images, state: .aligned,
                              imageCount: 1, alignSummary: nil, runs: [], createdAt: Date())
        let session = ProjectSession(project: project,
                                     directory: FileManager.default.temporaryDirectory)
        XCTAssertNil(session.lockedTab)

        session.trainVM.isTraining = true
        XCTAssertEqual(session.lockedTab, .train)

        session.trainVM.isTraining = false
        session.alignVM.isRunning = true
        XCTAssertEqual(session.lockedTab, .align)
    }
}
