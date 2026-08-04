import XCTest
@testable import SplatXMetalTrainer

/// Audit cleanup Task 6 — pure-logic verification of sleep prevention during long-running
/// work.
/// beginActivity/endActivity are replaced with injected closure stubs so that only the token's
/// lifetime transitions are checked (the real ProcessInfo calls are not tested — wiring to the
/// system API is the responsibility of the default arguments).
@MainActor final class SleepGuardTests: XCTestCase {
    private final class Recorder {
        var begins: [String] = []
        var ends = 0
    }
    private final class Token: NSObject {}

    private func makeGuard(_ rec: Recorder) -> SleepGuard {
        SleepGuard(begin: { reason in rec.begins.append(reason); return Token() },
                   end: { _ in rec.ends += 1 })
    }

    // MARK: activityReason (lockedTab → reason), a pure function

    func testActivityReasonCoversAllLockedWork() {
        XCTAssertNotNil(SleepGuard.activityReason(lockedTab: .train))
        XCTAssertNotNil(SleepGuard.activityReason(lockedTab: .align))
        XCTAssertNotNil(SleepGuard.activityReason(lockedTab: .cinema))
    }

    func testActivityReasonNilWhenNoWork() {
        XCTAssertNil(SleepGuard.activityReason(lockedTab: nil))
        XCTAssertNil(SleepGuard.activityReason(lockedTab: .view))
    }

    func testActivityReasonsAreDistinctPerWork() {
        // Activity Monitor and the logs must show which work is preventing sleep.
        let reasons = [ProjectTab.train, .align, .cinema]
            .compactMap { SleepGuard.activityReason(lockedTab: $0) }
        XCTAssertEqual(Set(reasons).count, 3)
    }

    func testActivityReasonFollowsLockedTabPredicate() {
        // Composition with the authoritative lockedTab decision — all three kinds of work
        // hold an activity, and no work gives nil.
        XCTAssertNotNil(SleepGuard.activityReason(
            lockedTab: ProjectSession.lockedTab(isTraining: true, isAligning: false)))
        XCTAssertNotNil(SleepGuard.activityReason(
            lockedTab: ProjectSession.lockedTab(isTraining: false, isAligning: true)))
        XCTAssertNotNil(SleepGuard.activityReason(
            lockedTab: ProjectSession.lockedTab(isTraining: false, isAligning: false,
                                                isExporting: true)))
        XCTAssertNil(SleepGuard.activityReason(
            lockedTab: ProjectSession.lockedTab(isTraining: false, isAligning: false)))
    }

    // MARK: Token lifetime transitions (the single-owner invariant)

    func testBeginsOnceAndIsIdempotentForSameReason() {
        let rec = Recorder()
        let g = makeGuard(rec)
        g.update(reason: "training")
        g.update(reason: "training")
        XCTAssertEqual(rec.begins, ["training"])
        XCTAssertEqual(rec.ends, 0)
    }

    func testEndsWhenReasonCleared() {
        let rec = Recorder()
        let g = makeGuard(rec)
        g.update(reason: "training")
        g.update(reason: nil)
        XCTAssertEqual(rec.begins.count, 1)
        XCTAssertEqual(rec.ends, 1)
        XCTAssertNil(g.currentReason)
    }

    func testClearWithoutActivityIsNoop() {
        let rec = Recorder()
        let g = makeGuard(rec)
        g.update(reason: nil)
        XCTAssertEqual(rec.begins.count, 0)
        XCTAssertEqual(rec.ends, 0)
    }

    func testReasonChangeRestartsActivity() {
        let rec = Recorder()
        let g = makeGuard(rec)
        g.update(reason: "training")
        g.update(reason: "export")
        XCTAssertEqual(rec.begins, ["training", "export"])
        XCTAssertEqual(rec.ends, 1)   // there is never more than one token
        XCTAssertEqual(g.currentReason, "export")
    }

    func testDeinitEndsOutstandingToken() {
        let rec = Recorder()
        var g: SleepGuard? = makeGuard(rec)
        g?.update(reason: "training")
        g = nil
        XCTAssertEqual(rec.ends, 1)
    }

    func testDeinitWithoutTokenDoesNotEnd() {
        let rec = Recorder()
        var g: SleepGuard? = makeGuard(rec)
        _ = g
        g = nil
        XCTAssertEqual(rec.ends, 0)
    }

    func testFullWorkCycleTrainThenIdleThenExport() {
        // Replays the lockedTab transitions of training start → end → export start → end
        // verbatim.
        let rec = Recorder()
        let g = makeGuard(rec)
        let seq: [ProjectTab?] = [
            ProjectSession.lockedTab(isTraining: true, isAligning: false),
            ProjectSession.lockedTab(isTraining: false, isAligning: false),
            ProjectSession.lockedTab(isTraining: false, isAligning: false, isExporting: true),
            ProjectSession.lockedTab(isTraining: false, isAligning: false),
        ]
        for tab in seq { g.update(reason: SleepGuard.activityReason(lockedTab: tab)) }
        XCTAssertEqual(rec.begins.count, 2)
        XCTAssertEqual(rec.ends, 2)
        XCTAssertNil(g.currentReason)
    }
}
