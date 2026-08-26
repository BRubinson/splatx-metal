// 2026-08-05 spec §2.2: the View tab Inspector's timelapse row and Export button both hang off
// one pure judgment, `RunPropertiesView.timelapseState`. These tests are the whole regression
// defense for it — the SwiftUI body that consumes it cannot be exercised on this machine (TCC
// privacy controls block GUI automation), which is exactly why the judgment was pulled
// out of the body in the first place.
//
// No file system is touched: `fileExists` is injected, so the "directory" below is a path that
// need not (and does not) exist. That is deliberate beyond convenience — a test that wrote real
// files would be reaching for a home directory, which this app's test host must never do.
import XCTest
@testable import SplatXMetalTrainer

final class TimelapseArtifactStateTests: XCTestCase {

    /// A path chosen so that a bug which ignored the injected predicate and asked the real file
    /// system would answer "does not exist" and fail the `.ready` cases loudly, rather than
    /// passing by luck.
    private let directory = URL(fileURLWithPath: "/nonexistent-splatx-test/project")

    private func record(timelapseFile: String?) -> RunRecord {
        RunRecord(id: "20260805-120000", outputFile: "output.ply", previewFile: nil,
                  outcome: "completed", finalPSNR: 28.5, splatCount: 1_000,
                  optionsSummary: "factor 2 · 7000it", createdAt: Date(),
                  antialiased: false, timelapseFile: timelapseFile)
    }

    private func alwaysExists(_: URL) -> Bool { true }
    private func neverExists(_: URL) -> Bool { false }

    // MARK: - ① No run open at all

    func testNilRecordIsNone() {
        XCTAssertEqual(
            RunPropertiesView.timelapseState(record: nil, directory: directory,
                                             isEncoding: false, fileExists: alwaysExists),
            .none)
    }

    // MARK: - ② A run with no timelapse recorded (off, or fewer than two frames captured)

    func testNilPathIsNone() {
        XCTAssertEqual(
            RunPropertiesView.timelapseState(record: record(timelapseFile: nil),
                                             directory: directory,
                                             isEncoding: false, fileExists: alwaysExists),
            .none)
    }

    // MARK: - ③ The manifest points at a file the user deleted by hand
    //
    // The case the existence check exists for. Trusting the record here would put an Export
    // button on screen whose save panel fails at copy time.

    func testPathWithMissingFileIsNone() {
        XCTAssertEqual(
            RunPropertiesView.timelapseState(record: record(timelapseFile: "runs/r/timelapse.gif"),
                                             directory: directory,
                                             isEncoding: false, fileExists: neverExists),
            .none)
    }

    // MARK: - ④ Recorded and present → ready, at the project-relative path resolved against the
    // project directory.
    //
    // The recorded string already contains `runs/<id>/`, so resolving it is a single append and
    // nothing here reconstructs the folder part. FIX 1: this used to be labelled "the outputFile
    // convention", which is the opposite of true — outputFile is a bare file name and
    // ExportService.presentPanel is the code that prepends `runs/<id>/` to it. The assertions
    // below pin the shape that actually holds.

    func testPathWithExistingFileIsReady() {
        var probed: [URL] = []
        let state = RunPropertiesView.timelapseState(
            record: record(timelapseFile: "runs/20260805-120000/timelapse.gif"),
            directory: directory, isEncoding: false,
            fileExists: { probed.append($0); return true })
        XCTAssertEqual(
            state,
            .ready(directory.appendingPathComponent("runs/20260805-120000/timelapse.gif")))
        // The predicate is consulted exactly once, on the resolved absolute URL — a regression
        // that checked the relative string, or skipped the check, shows up here.
        XCTAssertEqual(probed.count, 1)
        XCTAssertEqual(probed.first?.path,
                       directory.appendingPathComponent("runs/20260805-120000/timelapse.gif").path)
    }

    // MARK: - ⑤ The §2.3 gap: training finished, the encode is running, the record is still nil.

    func testEncodingWithNoFileIsEncoding() {
        XCTAssertEqual(
            RunPropertiesView.timelapseState(record: record(timelapseFile: nil),
                                             directory: directory,
                                             isEncoding: true, fileExists: neverExists),
            .encoding)
    }

    /// Encoding with no run open at all — the encode outlives the run being closed, so the flag
    /// alone must still resolve.
    func testEncodingWithNilRecordIsEncoding() {
        XCTAssertEqual(
            RunPropertiesView.timelapseState(record: nil, directory: directory,
                                             isEncoding: true, fileExists: neverExists),
            .encoding)
    }

    // MARK: - ⑥ Encoding **and** this record already has a file → ready wins.
    //
    // Reached by opening an older run while a new run's encode is in flight. isEncoding is a fact
    // about the session; a file at this record's path is a fact about this record. If the flag
    // won, the older run's working Export button would disappear behind a progress message about
    // a different run.

    func testExistingFileOutranksEncoding() {
        XCTAssertEqual(
            RunPropertiesView.timelapseState(record: record(timelapseFile: "runs/old/timelapse.mp4"),
                                             directory: directory,
                                             isEncoding: true, fileExists: alwaysExists),
            .ready(directory.appendingPathComponent("runs/old/timelapse.mp4")))
    }

    /// The other half of ⑥: with the record's file gone, the flag is what remains. Together these
    /// two pin the precedence from both sides.
    func testEncodingWinsWhenTheRecordedFileIsMissing() {
        XCTAssertEqual(
            RunPropertiesView.timelapseState(record: record(timelapseFile: "runs/old/timelapse.mp4"),
                                             directory: directory,
                                             isEncoding: true, fileExists: neverExists),
            .encoding)
    }

    // MARK: - The RESULT row's text

    func testRowValues() {
        XCTAssertEqual(RunPropertiesView.timelapseRowValue(.none), "—")
        XCTAssertEqual(RunPropertiesView.timelapseRowValue(.encoding), "encoding…")
        XCTAssertEqual(
            RunPropertiesView.timelapseRowValue(
                .ready(URL(fileURLWithPath: "/p/runs/r/timelapse.gif"))),
            "timelapse.gif")
    }

    /// The row never renders an empty string — an empty value cell in a KeyValueRow is
    /// indistinguishable from a layout bug.
    func testRowValueIsNeverEmpty() {
        for state: TimelapseArtifactState in [.none, .encoding,
                                              .ready(URL(fileURLWithPath: "/p/x.mov"))] {
            XCTAssertFalse(RunPropertiesView.timelapseRowValue(state).isEmpty)
        }
    }

    // MARK: - The Export button's label
    //
    // Derived from the file, not from the project's current TimelapseSettings.format — the two
    // disagree as soon as the user changes the setting after a run.

    func testButtonTitlePerContainer() {
        XCTAssertEqual(
            RunPropertiesView.timelapseButtonTitle(for: URL(fileURLWithPath: "/p/timelapse.gif")),
            "Timelapse (GIF)…")
        XCTAssertEqual(
            RunPropertiesView.timelapseButtonTitle(for: URL(fileURLWithPath: "/p/timelapse.mp4")),
            "Timelapse (MP4)…")
        XCTAssertEqual(
            RunPropertiesView.timelapseButtonTitle(for: URL(fileURLWithPath: "/p/timelapse.mov")),
            "Timelapse (MOV)…")
    }

    /// Every format the encoder can actually produce has a label — a new TimelapseFormat case
    /// fails here instead of silently shipping "Timelapse ()…".
    func testEveryTimelapseFormatHasAContainerLabel() {
        for format in TimelapseFormat.allCases {
            let url = URL(fileURLWithPath: "/p/timelapse.\(format.fileExtension)")
            XCTAssertEqual(RunPropertiesView.timelapseButtonTitle(for: url),
                           "Timelapse (\(format.displayName))…")
        }
    }

    func testButtonTitleWithoutExtension() {
        XCTAssertEqual(
            RunPropertiesView.timelapseButtonTitle(for: URL(fileURLWithPath: "/p/timelapse")),
            "Timelapse…")
    }
}
