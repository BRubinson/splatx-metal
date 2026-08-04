import XCTest
@testable import SplatXMetalTrainer

/// Audit cleanup Task 8 — verification of the pure single source of truth for the disabled
/// reasons of the Inspector CTAs (Start Align / Start Training). The `.disabled` decision and
/// the on-screen reason come from the same function, so verifying the decision here is exactly
/// verifying the disable conditions (the same idiom as CinemaInspectorAffordanceTests for the
/// Render CTA).
final class InspectorCTAReasonTests: XCTestCase {

    // MARK: Start Align

    func testAlignDisabledUnderTwoImages() {
        XCTAssertNotNil(InspectorView.startAlignDisabledReason(imageCount: 0))
        XCTAssertNotNil(InspectorView.startAlignDisabledReason(imageCount: 1))
    }

    func testAlignEnabledFromTwoImages() {
        XCTAssertNil(InspectorView.startAlignDisabledReason(imageCount: 2))
        XCTAssertNil(InspectorView.startAlignDisabledReason(imageCount: 200))
    }

    // MARK: Start Training — wording per cause

    func testTrainingBlockedBeforeAlign() {
        let reason = InspectorView.startTrainingDisabledReason(
            projectState: .imported, hasDataset: true, validationErrors: [])
        XCTAssertEqual(reason, "Run Align first — training needs camera poses.")
    }

    func testTrainingBlockedWithoutDataset() {
        let reason = InspectorView.startTrainingDisabledReason(
            projectState: .aligned, hasDataset: false, validationErrors: [])
        XCTAssertNotNil(reason)
        XCTAssertTrue(reason!.contains("reopen"))
    }

    func testTrainingBlockedByFirstValidationError() {
        let reason = InspectorView.startTrainingDisabledReason(
            projectState: .aligned, hasDataset: true,
            validationErrors: ["factor must be one of 1, 2, 4, 8.", "iterations must be ≥ 1."])
        XCTAssertEqual(reason, "Fix options: factor must be one of 1, 2, 4, 8.")
    }

    func testTrainingReasonsAreCauseSpecific() {
        // Different causes must produce different wording — the contract that "why it does
        // not work" is distinguishable on screen.
        let alignFirst = InspectorView.startTrainingDisabledReason(
            projectState: .imported, hasDataset: true, validationErrors: [])
        let noDataset = InspectorView.startTrainingDisabledReason(
            projectState: .trained, hasDataset: false, validationErrors: [])
        let badOptions = InspectorView.startTrainingDisabledReason(
            projectState: .trained, hasDataset: true, validationErrors: ["iterations must be ≥ 1."])
        XCTAssertEqual(Set([alignFirst!, noDataset!, badOptions!]).count, 3)
    }

    func testTrainingPriorityAlignBeforeOptions() {
        // If align is not done, that is reported before any option errors — the pipeline
        // prerequisite takes priority.
        let reason = InspectorView.startTrainingDisabledReason(
            projectState: .imported, hasDataset: false, validationErrors: ["iterations must be ≥ 1."])
        XCTAssertEqual(reason, "Run Align first — training needs camera poses.")
    }

    func testTrainingEnabledWhenAllClear() {
        for state in [ProjectState.aligned, .trained] {
            XCTAssertNil(InspectorView.startTrainingDisabledReason(
                projectState: state, hasDataset: true, validationErrors: []))
        }
    }

    func testTrainingReasonMatchesDefaultDraft() {
        // The default draft is valid — the CTA must not sit disabled for no reason.
        XCTAssertNil(InspectorView.startTrainingDisabledReason(
            projectState: .aligned, hasDataset: true,
            validationErrors: TrainOptionsDraft().validationErrors()))
    }
}
