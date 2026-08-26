import XCTest
@testable import SplatXMetalTrainer

/// 2026-08-04 spec §1.5 — how an applied auto-level correction is shown.
///
/// nil hides the row/line entirely (the toggle was off, or the tilt was under
/// GroundLevel.minTiltDegrees). That "do not show it" rule lives in exactly one pure function,
/// shared by the Inspector's ALIGNMENT section and AlignView's completion summary — the same
/// idiom as InspectorView.startAlignDisabledReason (InspectorCTAReasonTests).
final class AlignSummaryDisplayTests: XCTestCase {

    func testNilHidesTheRow() {
        XCTAssertNil(InspectorView.levelSummaryValue(nil))
    }

    func testOneDecimalDegreeSuffix() {
        XCTAssertEqual(InspectorView.levelSummaryValue(19.177130068350046), "19.2°")
        XCTAssertEqual(InspectorView.levelSummaryValue(12.44), "12.4°")
        XCTAssertEqual(InspectorView.levelSummaryValue(GroundLevel.minTiltDegrees), "0.5°")
    }

    /// The two ends of the pipe are pinned together: what align persisted is what the row shows.
    func testReadsPersistedSummaryField() {
        let applied = AlignSummary(poseCount: 28, pointCount: 5000, alignSeconds: 42.5,
                                   levelTiltDegrees: 7.24)
        XCTAssertEqual(InspectorView.levelSummaryValue(applied.levelTiltDegrees), "7.2°")
        let skipped = AlignSummary(poseCount: 28, pointCount: 5000, alignSeconds: 42.5)
        XCTAssertNil(InspectorView.levelSummaryValue(skipped.levelTiltDegrees))
    }
}
