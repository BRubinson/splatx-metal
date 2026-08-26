import XCTest
@testable import SplatXMetalTrainer

/// spec 2026-08-06-splatx-capture-bundle-import §5 / plan Task 5 — the Align tab's notice for a
/// project that already has an alignment we did not produce this session. `.capture` gets a
/// strengthened warning (re-aligning discards on-device LiDAR geometry that cannot be
/// recovered); `.colmap`/`.images`/`.video` keep the pre-existing neutral "if needed" wording;
/// `state == .imported` (nothing aligned yet) gets no notice at all — the view falls through to
/// its own "Not aligned yet" screen instead. Same pure-`static func` idiom as
/// InspectorView.startAlignDisabledReason (InspectorCTAReasonTests) and
/// InspectorView.levelSummaryValue (AlignSummaryDisplayTests).
final class AlignReAlignNoticeTests: XCTestCase {

    // MARK: capture + aligned — the strengthened warning is the whole point of this task.

    func testCaptureAlignedWarnsAboutDiscardedGeometry() {
        let notice = AlignView.existingAlignmentNotice(sourceKind: .capture, state: .aligned)
        XCTAssertEqual(notice, AlignView.ReAlignNotice(
            title: "Captured LiDAR alignment in use",
            systemImage: "cube.transparent",
            description: "This project's camera poses and point cloud were captured on-device"
                + " by LiDAR, not solved from images. Re-aligning replaces them with"
                + " images-only SfM output, and the captured geometry cannot be recovered."))
    }

    /// The generic COLMAP text never mentions geometry loss — the capture notice must not
    /// silently degrade into the same wording as a regression.
    func testCaptureNoticeDiffersFromColmapNotice() {
        let capture = AlignView.existingAlignmentNotice(sourceKind: .capture, state: .aligned)
        let colmap = AlignView.existingAlignmentNotice(sourceKind: .colmap, state: .aligned)
        XCTAssertNotEqual(capture, colmap)
    }

    /// `state != .imported` also covers `.trained` (aligned once, then trained past it) — the
    /// captured geometry was already at risk the moment align history was absent, independent of
    /// how far the project has since progressed.
    func testCaptureTrainedAlsoWarns() {
        XCTAssertEqual(AlignView.existingAlignmentNotice(sourceKind: .capture, state: .trained),
                       AlignView.existingAlignmentNotice(sourceKind: .capture, state: .aligned))
    }

    // MARK: colmap + aligned — unchanged from before this task.

    func testColmapAlignedKeepsExistingNeutralWording() {
        let notice = AlignView.existingAlignmentNotice(sourceKind: .colmap, state: .aligned)
        XCTAssertEqual(notice, AlignView.ReAlignNotice(
            title: "Imported COLMAP alignment in use",
            systemImage: "point.3.connected.trianglepath.dotted",
            description: "This project came with sparse/0. You can re-align from images if needed."))
    }

    /// `.images`/`.video` reaching `state != .imported` with no align history of our own should
    /// not happen in practice (their own Align run always sets `alignSummary` at the same time it
    /// advances state), but the switch must stay exhaustive — they fall into the same generic
    /// bucket as `.colmap` rather than silently going capture-flavored.
    func testImagesAndVideoFallBackToGenericWording() {
        let colmap = AlignView.existingAlignmentNotice(sourceKind: .colmap, state: .aligned)
        XCTAssertEqual(AlignView.existingAlignmentNotice(sourceKind: .images, state: .aligned), colmap)
        XCTAssertEqual(AlignView.existingAlignmentNotice(sourceKind: .video, state: .aligned), colmap)
    }

    // MARK: state == .imported — no notice from this function at any sourceKind.

    func testImportedStateHasNoNoticeRegardlessOfSourceKind() {
        for kind: ProjectSourceKind in [.images, .video, .colmap, .capture] {
            XCTAssertNil(AlignView.existingAlignmentNotice(sourceKind: kind, state: .imported),
                         "sourceKind \(kind) should not produce a notice while state is .imported")
        }
    }
}
