import XCTest
import SplatXMetal
@testable import SplatXMetalTrainer

/// Verifies the pure functions that assemble the notification wording plus the gate that
/// makes the authorization request happen exactly once (SingleFlightGate) (app round 5 item 6,
/// FIX 3 items 4 and 5). The delivery wrapper (AppNotifier) is never called from the tests —
/// touching UNUserNotificationCenter is dangerous in the runner bundle context and has side
/// effects: the permission dialog and actually delivering notifications. Only the wording and
/// the gate's pure logic are pinned here.
final class AppNotificationTextTests: XCTestCase {

    // MARK: Training finished (distinguished by outcome)

    func testTrainingCompletedUsesCanonicalTitleAndStats() {
        let c = AppNotificationText.trainingFinished(
            project: "garden", outcome: .completed, splatCount: 123456, finalPSNR: 27.314)
        XCTAssertEqual(c?.title, "Training complete")   // spec contract: a normal finish uses exactly this title
        XCTAssertEqual(c?.body, "garden — 123456 splats · PSNR 27.31 dB")
    }

    func testTrainingCompletedWithoutPSNROmitsIt() {
        let c = AppNotificationText.trainingFinished(
            project: "garden", outcome: .completed, splatCount: 5000, finalPSNR: nil)
        XCTAssertEqual(c?.body, "garden — 5000 splats")
        XCTAssertFalse(c?.body.contains("PSNR") ?? true)
    }

    func testTrainingThermalAbortedUsesDistinctPartialWording() {
        let c = AppNotificationText.trainingFinished(
            project: "garden", outcome: .thermalAborted, splatCount: 5000, finalPSNR: nil)
        XCTAssertNotEqual(c?.title, "Training complete")   // a partial result must be distinguishable from a full run
        XCTAssertEqual(c?.title, "Training stopped early (thermal)")
        XCTAssertEqual(c?.body, "garden — partial result saved · 5000 splats")
    }

    /// 2026-08-16: the run that started this. `outcome == .completed` with zero splats used to
    /// produce the literal banner "Training complete — LiDAR_… — 0 splats · PSNR 3.02 dB". The
    /// contract now is that the word "complete" cannot appear for a run that produced nothing.
    func testCollapsedRunIsNotAnnouncedAsComplete() {
        let c = AppNotificationText.trainingFinished(
            project: "LiDAR_20260814_155238", outcome: .completed, splatCount: 0, finalPSNR: 3.02)
        XCTAssertNotNil(c)                                        // it must still be announced
        XCTAssertNotEqual(c?.title, "Training complete")
        XCTAssertFalse(c?.title.lowercased().contains("complete") ?? true)
        XCTAssertEqual(c?.title, "Training produced no usable result")
        XCTAssertTrue(c?.body.contains("LiDAR_20260814_155238") ?? false)
        XCTAssertTrue(c?.body.contains("no splats survived") ?? false)
        // A notification cannot hold the full advice paragraph, so it points at where it lives.
        XCTAssertTrue(c?.body.contains("Train tab") ?? false)
    }

    /// The verdict is derived inside the assembly function from the values it already receives
    /// (not passed in), so a caller cannot forget it — a render worse than a blank gray image is
    /// caught the same way as a zero count.
    func testDegenerateRenderIsAlsoNotAnnouncedAsComplete() {
        let c = AppNotificationText.trainingFinished(
            project: "garden", outcome: .completed, splatCount: 900_000, finalPSNR: 3.02)
        XCTAssertEqual(c?.title, "Training produced no usable result")
        XCTAssertTrue(c?.body.contains("3.02 dB") ?? false)
    }

    /// A thermal abort keeps its own wording even at zero splats — it is a normal termination
    /// (README "Exit codes": exit 2 is not an error), and the "stopped early" phrasing already tells the
    /// truth about why there is little to show.
    func testThermalAbortAtZeroSplatsKeepsItsOwnWording() {
        let c = AppNotificationText.trainingFinished(
            project: "garden", outcome: .thermalAborted, splatCount: 0, finalPSNR: nil)
        XCTAssertEqual(c?.title, "Training stopped early (thermal)")
    }

    /// A cancellation at zero splats is still the user's own action — still no notification.
    func testCancelledAtZeroSplatsStillProducesNoNotification() {
        XCTAssertNil(AppNotificationText.trainingFinished(
            project: "garden", outcome: .cancelled, splatCount: 0, finalPSNR: 1.0))
    }

    func testTrainingCancelledProducesNoNotification() {
        // Cancellation is an action the user took themselves — no notification (spec: the
        // same intent as avoiding over-notification).
        XCTAssertNil(AppNotificationText.trainingFinished(
            project: "garden", outcome: .cancelled, splatCount: 5000, finalPSNR: nil))
    }

    // MARK: Training failed (FIX 3 item 4 — making the failure wording templates symmetric)

    func testTrainingFailedUnwrapsTrainFailedPayload() {
        // The previous caller passed "\(error)" straight through, so the body exposed a raw
        // enum dump (trainFailed("…")), while align used the LocalizedError summary — an
        // asymmetry. Deriving the summary was moved inside the assembly function, and it is
        // now a contract that only the core's err_buf message (the payload) appears in the
        // sentence.
        let c = AppNotificationText.trainingFailed(
            project: "garden", error: SplatXTrainerError.trainFailed("metallib not found"))
        XCTAssertEqual(c.title, "Training failed")
        XCTAssertEqual(c.body, "garden — metallib not found")
        XCTAssertFalse(c.body.contains("trainFailed"))   // a raw enum dump MUST NOT be exposed
    }

    func testFailureBodiesUseLocalizedDescriptionWhenAvailable() {
        struct SampleError: LocalizedError { var errorDescription: String? { "disk is full" } }
        // The train and align failures use the same summary convention — the symmetry of the
        // wording templates is a contract.
        XCTAssertEqual(
            AppNotificationText.trainingFailed(project: "garden", error: SampleError()).body,
            "garden — disk is full")
        XCTAssertEqual(
            AppNotificationText.alignFailed(project: "garden", error: SampleError()).body,
            "garden — disk is full")
    }

    // MARK: align finished and failed

    func testAlignFinishedMentionsPoseAndPointCounts() {
        let c = AppNotificationText.alignFinished(project: "garden", poseCount: 185, pointCount: 12345)
        XCTAssertEqual(c.title, "Align complete")
        XCTAssertEqual(c.body, "garden — 185 poses · 12345 points")
    }

    func testAlignFailedFallsBackToPlainDescription() {
        struct PlainError: Error {}
        // When it is not a LocalizedError it falls back to String(describing:) — the same
        // string as the previous "\(error)".
        let c = AppNotificationText.alignFailed(project: "garden", error: PlainError())
        XCTAssertEqual(c.title, "Align failed")
        XCTAssertEqual(c.body, "garden — PlainError()")
    }

    // MARK: The gate that makes the authorization request happen once (FIX 3 item 5 — pure
    // logic that never touches UNUserNotificationCenter)

    /// A state box for observing the gate — a test-local var cannot be captured in a
    /// @Sendable closure, so it is wrapped in a @MainActor class.
    @MainActor private final class GateProbe {
        var starts = 0
        var firstFinished = false
        var secondOpRan = false
        var secondReturned = false
        var release: CheckedContinuation<Void, Never>?
    }

    @MainActor
    func testAuthGateRunsOperationExactlyOnceAcrossCalls() async {
        let gate = SingleFlightGate()
        let probe = GateProbe()
        await gate.run { probe.starts += 1 }
        await gate.run { probe.starts += 1 }   // calling again after completion does not re-run it (the once-per-process contract)
        XCTAssertEqual(probe.starts, 1)
    }

    @MainActor
    func testAuthGateMidFlightCallerAwaitsSameRunInsteadOfSkipping() async {
        // Reproducing the loss scenario: a later delivery arriving while the first delivery's
        // requestAuthorization was still in progress read the not-yet-determined settings
        // (looking only at the old didRequestAuthorization=true flag) and was silently lost —
        // the gate makes later callers wait until that same run **completes**.
        let gate = SingleFlightGate()
        let probe = GateProbe()
        let first = Task { @MainActor in
            await gate.run {
                probe.starts += 1
                await withCheckedContinuation { probe.release = $0 }   // pin the in-progress state
                probe.firstFinished = true
            }
        }
        while probe.release == nil { await Task.yield() }   // until the first run reaches its suspension
        let second = Task { @MainActor in
            await gate.run { probe.secondOpRan = true }
            probe.secondReturned = true
        }
        await Task.yield()
        XCTAssertFalse(probe.secondReturned)   // it MUST NOT return before the in-progress run finishes
        probe.release?.resume()
        _ = await first.value
        _ = await second.value
        XCTAssertEqual(probe.starts, 1)          // a later call does not re-run the request
        XCTAssertTrue(probe.firstFinished)
        XCTAssertFalse(probe.secondOpRan)        // the later operation is discarded (it only awaits the shared run)
        XCTAssertTrue(probe.secondReturned)
    }
}
