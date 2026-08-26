import XCTest
import SplatXMetal
@testable import SplatXMetalTrainer

/// The collapse verdict (2026-08-16). The incident this pins: a 15000-iteration MCMC run
/// (factor 2 · cap 1000k · SH3) finished with `splatCount 0` and `finalPSNR 3.02`, wrote an
/// `element vertex 0` PLY, and the app reported **"Training complete"** — a success notification, a
/// `completed` run record, and a tab switch to an empty viewer.
///
/// Two halves are under test here and both matter. The first is where the boundary sits: a
/// verdict that fires too eagerly would start calling small-but-real runs broken, which is a worse
/// failure than the one being fixed because it teaches the user to ignore the warning. The second
/// is the wording — that a collapsed run is never described with the success vocabulary, on any of
/// the three surfaces (the run record, the console, the notification).
final class RunHealthTests: XCTestCase {

    // MARK: The zero case — the certain signal

    func testCompletedWithZeroSplatsIsCollapsed() {
        // The incident, exactly: reached run_until, nothing survived the export filter.
        XCTAssertEqual(RunHealth.evaluate(outcome: .completed, splatCount: 0, finalPSNR: 3.02),
                       .noSplats)
    }

    func testZeroSplatsIsCollapsedEvenWithNoEvalPSNR() {
        // testSplit == 0 means no final eval and therefore no PSNR at all. The count alone has to
        // carry the judgement, because that configuration is perfectly ordinary.
        XCTAssertEqual(RunHealth.evaluate(outcome: .completed, splatCount: 0, finalPSNR: nil),
                       .noSplats)
    }

    func testNegativeSplatCountIsTreatedAsZero() {
        // Defensive: `<= 0` rather than `== 0`. A negative count could only come from a corrupt
        // manifest or an int overflow, and neither is a run worth calling complete.
        XCTAssertEqual(RunHealth.evaluate(outcome: .completed, splatCount: -1, finalPSNR: nil),
                       .noSplats)
    }

    func testNoSplatsWinsOverTheRenderTest() {
        // With nothing left to render the low PSNR is a consequence, not a second independent
        // symptom — the concrete sentence ("the export kept none") is the one worth showing.
        XCTAssertEqual(RunHealth.evaluate(outcome: .completed, splatCount: 0, finalPSNR: 0.5),
                       .noSplats)
    }

    // MARK: The PSNR floor — derived, not tuned

    func testRenderWorseThanFlatGrayIsCollapsed() {
        // 3.02 dB → mse ≈ 0.5, i.e. twice as wrong as painting every pixel mid-gray. No scene is
        // "hard" enough to explain a model that loses to a constant.
        XCTAssertEqual(RunHealth.evaluate(outcome: .completed, splatCount: 900_000,
                                          finalPSNR: 3.02),
                       .degenerateRender(psnrDB: 3.02))
    }

    func testFlatGrayFloorIsTheConstantPredictorBound() {
        // 10·log10(1/0.25) = 6.0206 dB is the *worst* a mid-gray constant can score against any
        // image in [0,1] — the same number the core's own hand check quotes
        // (TrainerTests.cpp:1271). The threshold sits a hair below it so rounding never trips the
        // check on the boundary itself.
        XCTAssertEqual(RunHealth.flatGrayFloorDB, 6.0, accuracy: 1e-12)
        XCTAssertLessThan(RunHealth.flatGrayFloorDB, 10.0 * log10(1.0 / 0.25))
    }

    func testJustAboveAndJustBelowTheFloor() {
        XCTAssertEqual(RunHealth.evaluate(outcome: .completed, splatCount: 1000, finalPSNR: 5.99),
                       .degenerateRender(psnrDB: 5.99))
        XCTAssertEqual(RunHealth.evaluate(outcome: .completed, splatCount: 1000, finalPSNR: 6.0),
                       .ok)   // exactly at the floor is NOT collapsed — the test is strict `<`
        XCTAssertEqual(RunHealth.evaluate(outcome: .completed, splatCount: 1000, finalPSNR: 6.01),
                       .ok)
    }

    func testNonFinitePSNRIsNoSignalRatherThanCollapse() {
        // The C boundary already maps the core's NaN to nil (SplatXTrainer.swift:574), so a
        // non-finite value cannot arrive. If one ever did, "no signal" is the conservative answer —
        // a false collapse is the failure mode this whole file is guarding against.
        XCTAssertEqual(RunHealth.evaluate(outcome: .completed, splatCount: 1000,
                                          finalPSNR: Double.nan),
                       .ok)
        XCTAssertEqual(RunHealth.evaluate(outcome: .completed, splatCount: 1000,
                                          finalPSNR: Double.infinity),
                       .ok)
    }

    // MARK: No false positives on small-but-real runs

    func testSmallRunsAreNotCollapsed() {
        // A tiny object, a short run, or a tight cap can legitimately finish with very few splats.
        // The app has no idea what "enough" means for a given scene, so anything above zero with a
        // sane render is left alone. If this test ever has to be relaxed, the threshold being added
        // is wrong.
        for count in [1, 42, 800, 5_000, 40_000] {
            XCTAssertEqual(RunHealth.evaluate(outcome: .completed, splatCount: count,
                                              finalPSNR: 18.4),
                           .ok, "\(count) splats at 18.4 dB is a real result")
        }
    }

    func testMediocreButRealPSNRIsNotCollapsed() {
        // A poor reconstruction is still a reconstruction. 8 dB is bad and 12 dB is disappointing;
        // neither is "worse than a blank image", so neither is this function's business.
        XCTAssertEqual(RunHealth.evaluate(outcome: .completed, splatCount: 200_000, finalPSNR: 8.0),
                       .ok)
        XCTAssertEqual(RunHealth.evaluate(outcome: .completed, splatCount: 200_000, finalPSNR: 12.5),
                       .ok)
    }

    func testHealthyRunIsOk() {
        XCTAssertEqual(RunHealth.evaluate(outcome: .completed, splatCount: 1_000_000,
                                          finalPSNR: 27.31),
                       .ok)
        XCTAssertFalse(RunHealth.evaluate(outcome: .completed, splatCount: 1_000_000,
                                          finalPSNR: 27.31).isCollapsed)
    }

    // MARK: The normal terminations are never relabelled (README "Exit codes": 2 / 3 are not errors)

    func testCancelledIsNeverCollapsedEvenWithZeroSplats() {
        // The user pressed Stop. Whatever the numbers look like, calling that a collapse would
        // blame the data for a decision the user made.
        XCTAssertEqual(RunHealth.evaluate(outcome: .cancelled, splatCount: 0, finalPSNR: nil), .ok)
        XCTAssertEqual(RunHealth.evaluate(outcome: .cancelled, splatCount: 0, finalPSNR: 1.0), .ok)
    }

    func testThermalAbortIsNeverCollapsedEvenWithZeroSplats() {
        // A graceful thermal abort (CLI exit 2) is a normal termination that already announces
        // itself as stopped-early with a partial result. Re-labelling it would erase the one
        // distinction that changes what the user should do next: cut short vs. ran and failed.
        XCTAssertEqual(RunHealth.evaluate(outcome: .thermalAborted, splatCount: 0, finalPSNR: nil),
                       .ok)
        XCTAssertEqual(RunHealth.evaluate(outcome: .thermalAborted, splatCount: 0, finalPSNR: 2.0),
                       .ok)
    }

    // MARK: What gets written into the manifest

    func testRecordOutcomeMintsCollapsedOnlyForCollapse() {
        XCTAssertEqual(RunHealth.recordOutcome(outcome: .completed, splatCount: 0, finalPSNR: 3.02),
                       "collapsed")
        XCTAssertEqual(RunHealth.recordOutcome(outcome: .completed, splatCount: 500_000,
                                               finalPSNR: 26.0),
                       "completed")
        XCTAssertEqual(RunHealth.recordOutcome(outcome: .cancelled, splatCount: 0, finalPSNR: nil),
                       "cancelled")
        XCTAssertEqual(RunHealth.recordOutcome(outcome: .thermalAborted, splatCount: 0,
                                               finalPSNR: nil),
                       "thermalAborted")
    }

    func testCollapsedOutcomeIsNotTheSuccessString() {
        // The RUNS badge (InspectorView.swift:337) shows every value that is not exactly
        // "completed" — so this inequality is the whole reason the badge appears at all.
        XCTAssertNotEqual(RunHealth.collapsedOutcome, "completed")
    }

    func testStoredOutcomeRecognitionRoundTripsWithWhatWasWritten() {
        // The reader (the RUNS badge tint, the inspector's note) and the writer must agree — this
        // is the pair, checked against each other rather than against a literal spelled twice.
        XCTAssertTrue(RunHealth.isCollapsed(
            recordOutcome: RunHealth.recordOutcome(outcome: .completed, splatCount: 0,
                                                   finalPSNR: 3.02)))
        for healthy in ["completed", "cancelled", "thermalAborted", ""] {
            XCTAssertFalse(RunHealth.isCollapsed(recordOutcome: healthy), healthy)
        }
    }

    func testStoredCollapseNoteExplainsAndAdvises() {
        // Reopened from the RUNS history, the banner is long gone and "outcome: collapsed" is a
        // label rather than an explanation — the note has to carry both halves on its own.
        XCTAssertTrue(RunHealth.storedCollapseNote.contains("no usable result"))
        XCTAssertTrue(RunHealth.storedCollapseNote.contains(RunHealth.advice))
    }

    func testCollapsedOutcomeSurvivesAManifestRoundTrip() throws {
        // The manifest keeps `outcome` as a plain String, so widening the vocabulary must cost no
        // compatibility. Encode → decode has to return the same value, and an older build reading
        // it would take the same path.
        let record = RunRecord(id: "20260816-065600", outputFile: "output.ply", previewFile: nil,
                               outcome: RunHealth.collapsedOutcome, finalPSNR: 3.02, splatCount: 0,
                               optionsSummary: "factor 2 · 15000it · MCMC · cap 1000k · SH3",
                               createdAt: Date(timeIntervalSince1970: 1_755_000_000))
        let project = Project(id: UUID(), name: "LiDAR_20260814_155238", sourceKind: .capture,
                              state: .trained, imageCount: 120, alignSummary: nil,
                              runs: [record], createdAt: Date(timeIntervalSince1970: 1_754_000_000))
        let data = try JSONEncoder.project.encode(project)
        let decoded = try JSONDecoder.project.decode(Project.self, from: data)
        XCTAssertEqual(decoded.runs.first?.outcome, "collapsed")
        XCTAssertEqual(decoded.runs.first?.splatCount, 0)
    }

    // MARK: The console report

    func testHealthyCompletionKeepsTheExistingConsoleWording() {
        // Unchanged from before this round — the success line is not what was broken.
        XCTAssertEqual(
            RunHealth.completionConsoleLines(outcome: .completed, splatCount: 123_456,
                                             finalPSNR: 27.314),
            ["■ Finished (completed): N 123456 · PSNR 27.31 dB"])
        XCTAssertEqual(
            RunHealth.completionConsoleLines(outcome: .thermalAborted, splatCount: 5_000,
                                             finalPSNR: nil),
            ["■ Finished (thermalAborted): N 5000"])
    }

    func testCollapsedCompletionNeverUsesTheSuccessWording() {
        let lines = RunHealth.completionConsoleLines(outcome: .completed, splatCount: 0,
                                                     finalPSNR: 3.02)
        let joined = lines.joined(separator: "\n")
        XCTAssertFalse(joined.contains("Finished"),
                       "a run that produced nothing must not be announced as finished")
        XCTAssertFalse(joined.lowercased().contains("complete"))
        XCTAssertTrue(joined.contains("collapsed"))
        XCTAssertTrue(joined.contains("N 0"))              // the evidence stays on the line
        XCTAssertTrue(joined.contains("3.02 dB"))
        XCTAssertEqual(lines.count, 2)                     // what happened, then what to try
        XCTAssertTrue(lines[1].contains("Align"))
    }

    // MARK: The inline banner text

    func testMessageSaysWhatHappenedAndWhatToTry() {
        let message = RunHealth.noSplats.message
        XCTAssertTrue(message.contains("collapsed"))
        XCTAssertTrue(message.contains("empty"))
        // The advice half is what makes this actionable rather than just alarming — the three
        // levers the user actually has, in the order they are worth pulling.
        XCTAssertTrue(message.contains("Align"))
        XCTAssertTrue(message.contains("iterations"))
        XCTAssertTrue(message.contains("splat budget"))
        XCTAssertTrue(message.contains("re-capture"))
    }

    func testDegenerateRenderMessageQuotesTheScore() {
        XCTAssertTrue(RunHealth.degenerateRender(psnrDB: 3.02).message.contains("3.02 dB"))
        XCTAssertTrue(RunHealth.degenerateRender(psnrDB: 3.02).message.contains("blank gray"))
    }

    func testOkHasNoHeadline() {
        // Callers branch on `isCollapsed` and never print this; an empty string keeps a mistaken
        // caller from showing a blank alarming banner with words in it.
        XCTAssertEqual(RunHealth.ok.headline, "")
        XCTAssertEqual(RunHealth.ok.shortReason, "")
    }
}
