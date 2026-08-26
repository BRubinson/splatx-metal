// 2026-08-04 spec §3.6: the timelapse settings persist at project level and the produced file
// persists on the run record. Every new field follows the existing forward-compatibility
// convention (decodeIfPresent ?? default) — the three ManifestTests cases plus the lenient
// decode of an unknown format string.
import XCTest
@testable import SplatXMetalTrainer

final class TimelapseManifestTests: XCTestCase {

    private func projectJSON(extra: String) -> Data {
        Data("""
        {"id":"\(UUID().uuidString)","name":"p","sourceKind":"images","state":"trained",
         "imageCount":1,"runs":[],"createdAt":"2026-08-04T00:00:00Z"\(extra)}
        """.utf8)
    }

    // MARK: - ① An old manifest with no timelapse key → the defaults

    func testOldProjectDecodesWithDefaultTimelapse() throws {
        let p = try JSONDecoder.project.decode(Project.self, from: projectJSON(extra: ""))
        XCTAssertEqual(p.timelapse, TimelapseSettings())
        XCTAssertFalse(p.timelapse.enabled)
    }

    // MARK: - ② Round trip

    func testTimelapseSettingsRoundTrip() throws {
        // createdAt uses a whole-second Date (not Date()) — JSONEncoder.project's .iso8601
        // strategy has no fractional-second option, so a sub-second Date() would round-trip
        // truncated and fail the back == p comparison below for a reason unrelated to
        // timelapse. ProjectManifestTests.sample() follows the same convention.
        var p = Project(id: UUID(), name: "p", sourceKind: .images, state: .trained,
                        imageCount: 1, alignSummary: nil, runs: [],
                        createdAt: Date(timeIntervalSince1970: 1_722_800_000))
        // aspectSource is set to the **non-default** case on purpose: a field that round-trips
        // only because both sides happen to hold the default proves nothing.
        p.timelapse = TimelapseSettings(enabled: true, everySteps: 250, format: .gif,
                                        fps: 24, shortSide: 720, aspectSource: .dataset,
                                        overlay: false)
        let back = try JSONDecoder.project.decode(Project.self,
                                                  from: JSONEncoder.project.encode(p))
        XCTAssertEqual(back.timelapse, p.timelapse)
        XCTAssertEqual(back, p)
    }

    // MARK: - ③ A partial object → the missing fields fall back to the defaults

    func testPartialTimelapseObjectFillsDefaults() throws {
        let p = try JSONDecoder.project.decode(
            Project.self, from: projectJSON(extra: ",\"timelapse\":{\"enabled\":true}"))
        XCTAssertTrue(p.timelapse.enabled)
        XCTAssertEqual(p.timelapse.everySteps, 100)
        XCTAssertEqual(p.timelapse.format, .gif)
        XCTAssertEqual(p.timelapse.fps, 15)
        XCTAssertEqual(p.timelapse.shortSide, 480)
        // 2026-08-05 §2.1: a manifest written before this round has no aspectSource key, so it
        // takes the new default — the documented behaviour change, pinned here.
        XCTAssertEqual(p.timelapse.aspectSource, .viewport)
        XCTAssertTrue(p.timelapse.overlay)
    }

    // MARK: - ④ An unknown format string decodes leniently to mp4 (the same convention as
    // Project.outputAspect — decoding the enum directly would throw on a future case)

    /// An unparseable format falls back to the **default**, so this tracks whatever
    /// `TimelapseSettings.format` defaults to rather than naming a fixed case.
    func testUnknownFormatStringDecodesToTheDefault() throws {
        let p = try JSONDecoder.project.decode(
            Project.self,
            from: projectJSON(extra: ",\"timelapse\":{\"enabled\":true,\"format\":\"webm\"}"))
        XCTAssertEqual(p.timelapse.format, TimelapseSettings().format)
        XCTAssertEqual(p.timelapse.format, .gif)   // the default today — pinned so a change is deliberate
        XCTAssertTrue(p.timelapse.enabled)   // the rest of the object still decodes
    }

    // MARK: - ⑤ aspectSource follows the same three-place convention (2026-08-05 §2.1)

    /// The stored key is honoured (not just written) — this is the half of the round trip that a
    /// missing `CodingKeys` entry would break while the encode side still looked fine.
    func testStoredAspectSourceDecodes() throws {
        let p = try JSONDecoder.project.decode(
            Project.self,
            from: projectJSON(extra: ",\"timelapse\":{\"aspectSource\":\"dataset\"}"))
        XCTAssertEqual(p.timelapse.aspectSource, .dataset)
    }

    /// An unknown aspect string decodes leniently to the default, like `format` — one unparseable
    /// value must not fail the whole project manifest.
    func testUnknownAspectSourceStringDecodesToTheDefault() throws {
        let p = try JSONDecoder.project.decode(
            Project.self,
            from: projectJSON(
                extra: ",\"timelapse\":{\"enabled\":true,\"aspectSource\":\"letterbox\"}"))
        XCTAssertEqual(p.timelapse.aspectSource, TimelapseSettings().aspectSource)
        XCTAssertEqual(p.timelapse.aspectSource, .viewport)   // the default today — pinned
        XCTAssertTrue(p.timelapse.enabled)   // the rest of the object still decodes
    }

    // MARK: - RunRecord.timelapseFile

    func testRunRecordTimelapseFileDefaultsNilAndRoundTrips() throws {
        let json = """
        {"id":"20260804-000000","outputFile":"output.ply","outcome":"completed",
         "splatCount":1000,"optionsSummary":"s","createdAt":"2026-08-04T00:00:00Z"}
        """
        let old = try JSONDecoder.project.decode(RunRecord.self, from: Data(json.utf8))
        XCTAssertNil(old.timelapseFile)

        let r = RunRecord(id: "20260804-000001", outputFile: "output.ply", previewFile: nil,
                          outcome: "completed", finalPSNR: nil, splatCount: 1,
                          optionsSummary: "s", createdAt: Date(), antialiased: false,
                          timelapseFile: "runs/20260804-000001/timelapse.mp4")
        let back = try JSONDecoder.project.decode(RunRecord.self,
                                                  from: JSONEncoder.project.encode(r))
        XCTAssertEqual(back.timelapseFile, "runs/20260804-000001/timelapse.mp4")
    }

    // MARK: - The session injects the persisted settings into the training draft

    @MainActor func testSessionInjectsTimelapseSettingsIntoDraft() {
        var project = Project(id: UUID(), name: "p", sourceKind: .images, state: .aligned,
                              imageCount: 3, alignSummary: nil, runs: [], createdAt: Date())
        project.timelapse = TimelapseSettings(enabled: true, everySteps: 200, format: .mov,
                                              fps: 30, shortSide: 1440, overlay: false)
        let session = ProjectSession(
            project: project,
            directory: FileManager.default.temporaryDirectory
                .appendingPathComponent("sx-timelapse-session", isDirectory: true))
        XCTAssertEqual(session.trainVM.draft.timelapse, project.timelapse)
    }

    // MARK: - The pure transition that records the produced file on the run (the same shape as
    // Project.removingRun — the encode finishes asynchronously, so the run may have been deleted
    // in the meantime)

    private func projectWithRuns() -> Project {
        Project(id: UUID(), name: "p", sourceKind: .images, state: .trained, imageCount: 1,
                alignSummary: nil,
                runs: [
                    RunRecord(id: "A", outputFile: "output.ply", previewFile: nil,
                              outcome: "completed", finalPSNR: nil, splatCount: 1,
                              optionsSummary: "s", createdAt: Date(timeIntervalSince1970: 1)),
                    RunRecord(id: "B", outputFile: "output.ply", previewFile: nil,
                              outcome: "cancelled", finalPSNR: nil, splatCount: 2,
                              optionsSummary: "s", createdAt: Date(timeIntervalSince1970: 2)),
                ],
                createdAt: Date(timeIntervalSince1970: 0))
    }

    func testSettingTimelapseFileTargetsOnlyTheMatchingRun() {
        let p = projectWithRuns().settingTimelapseFile(runID: "B",
                                                       path: "runs/B/timelapse.mp4")
        XCTAssertNil(p.runs[0].timelapseFile)
        XCTAssertEqual(p.runs[1].timelapseFile, "runs/B/timelapse.mp4")
    }

    func testSettingTimelapseFileOnAMissingRunChangesNothing() {
        let original = projectWithRuns()
        let p = original.settingTimelapseFile(runID: "Z", path: "runs/Z/timelapse.gif")
        XCTAssertEqual(p, original)
    }

    func testSettingTimelapseFileToNilClearsIt() {
        var original = projectWithRuns()
        original.runs[0].timelapseFile = "runs/A/timelapse.mov"
        let p = original.settingTimelapseFile(runID: "A", path: nil)
        XCTAssertNil(p.runs[0].timelapseFile)
    }

    // MARK: - The timelapse completion summary's failure clause (FIX 5 / Important 2 — spec §3.4
    // forbids a silent shortfall; TimelapseRecorder.failedFrameCount/failedFrameDeleteCount must
    // actually reach the console, not just exist).

    @MainActor func testTimelapseFailureNoteIsEmptyWhenNothingFailed() {
        XCTAssertEqual(TrainViewModel.timelapseFailureNote(failedWrites: 0, failedDeletes: 0), "")
    }

    @MainActor func testTimelapseFailureNoteReportsWriteFailures() {
        XCTAssertEqual(TrainViewModel.timelapseFailureNote(failedWrites: 3, failedDeletes: 0),
                       " · 3 frame(s) failed to write")
    }

    @MainActor func testTimelapseFailureNoteReportsDeleteFailuresSeparately() {
        let note = TrainViewModel.timelapseFailureNote(failedWrites: 0, failedDeletes: 2)
        XCTAssertTrue(note.contains("2 stale frame(s)"))
        XCTAssertTrue(note.contains("shoot from here"))
        XCTAssertFalse(note.contains("failed to write"))   // the two clauses stay distinct
    }

    @MainActor func testTimelapseFailureNoteConcatenatesBothClauses() {
        let note = TrainViewModel.timelapseFailureNote(failedWrites: 1, failedDeletes: 1)
        XCTAssertTrue(note.contains("1 frame(s) failed to write"))
        XCTAssertTrue(note.contains("1 stale frame(s)"))
    }
}
