import XCTest
@testable import SplatXMetalTrainer

final class ProjectManifestTests: XCTestCase {
    private func sample() -> Project {
        Project(id: UUID(), name: "garden", sourceKind: .images, state: .trained,
                imageCount: 30,
                alignSummary: AlignSummary(poseCount: 28, pointCount: 5000, alignSeconds: 42.5),
                runs: [
                    RunRecord(id: "20260719-100000", outputFile: "output.ply", previewFile: "preview.png",
                              outcome: "completed", finalPSNR: 24.5, splatCount: 120_000,
                              optionsSummary: "factor 4 · 7000it", createdAt: Date(timeIntervalSince1970: 100)),
                    RunRecord(id: "20260719-110000", outputFile: "output.spz", previewFile: nil,
                              outcome: "cancelled", finalPSNR: nil, splatCount: 80_000,
                              optionsSummary: "factor 4 · 3000it", createdAt: Date(timeIntervalSince1970: 200)),
                ],
                createdAt: Date(timeIntervalSince1970: 50))
    }

    func testManifestRoundTrip() throws {
        let p = sample()
        let data = try JSONEncoder.project.encode(p)
        let back = try JSONDecoder.project.decode(Project.self, from: data)
        XCTAssertEqual(back, p)
    }

    func testLatestRunIsNewestByCreatedAt() {
        XCTAssertEqual(sample().latestRun?.id, "20260719-110000")
    }

    func testDecodeToleratesMissingOptionals() throws {
        // A minimal manifest without alignSummary, runs, or cameraPaths must still open
        // (forward compatibility).
        let json = """
        {"id":"F5B2D3A0-0000-0000-0000-000000000000","name":"m","sourceKind":"colmap",
         "state":"aligned","imageCount":3,"createdAt":"2026-07-19T00:00:00Z"}
        """.data(using: .utf8)!
        let p = try JSONDecoder.project.decode(Project.self, from: json)
        XCTAssertNil(p.alignSummary)
        XCTAssertTrue(p.runs.isEmpty)
        XCTAssertNil(p.latestRun)
        XCTAssertTrue(p.cameraPaths.isEmpty)   // Cinema Task 3 — forward compatible with old manifests
        XCTAssertNil(p.overlay)                // Cinema Task 4 — the overlay is forward compatible too
        XCTAssertFalse(p.cinemaWatermarkSeeded)   // Cinema FIX 5 — the seed flag is forward compatible too
        XCTAssertTrue(p.autoLevelOnAlign)         // Auto-level Task 2 — absent key means the ON default
    }

    // Cinema FIX 5: the once-only watermark seed flag is persisted and restored (the
    // persistence basis for respecting removal).
    func testWatermarkSeededPersistRoundTrip() throws {
        var p = sample()
        p.cinemaWatermarkSeeded = true
        let data = try JSONEncoder.project.encode(p)
        let back = try JSONDecoder.project.decode(Project.self, from: data)
        XCTAssertTrue(back.cinemaWatermarkSeeded)
        XCTAssertEqual(back, p)
    }

    // Cinema Task 4: the single project-level overlay (shared by every export) is persisted
    // and restored.
    func testOverlayPersistRoundTrip() throws {
        var p = sample()
        p.overlay = OverlaySpec(imagePath: "overlay/logo.png", anchor: .topLeading,
                                offsetX: 3, offsetY: -2, sizePercent: 25,
                                margin: 2, opacity: 0.8)
        let data = try JSONEncoder.project.encode(p)
        let back = try JSONDecoder.project.decode(Project.self, from: data)
        XCTAssertEqual(back, p)
        XCTAssertEqual(back.overlay?.anchor, .topLeading)
        XCTAssertEqual(back.overlay?.opacity, 0.8)
    }

    // Cinema Task 3: camera paths are persisted to and restored from the manifest verbatim
    // (keyframes and pacing included).
    func testCameraPathsPersistRoundTrip() throws {
        var p = sample()
        p.cameraPaths = [
            CameraPath(name: "orbit", keyframes: [
                Keyframe(time: 0, pos: SIMD3(1, 2, 3),
                         quat: SIMD4(0.5, 0.25, 0.75, 1.0), fov: 0.8),
                Keyframe(time: 4, pos: SIMD3(-1, 0, 4),
                         quat: SIMD4(1, 0, 0, 0), fov: 1.1),
            ], ease: .none, loop: true, pacing: .manual, duration: 8),
        ]
        let data = try JSONEncoder.project.encode(p)
        let back = try JSONDecoder.project.decode(Project.self, from: data)
        XCTAssertEqual(back, p)
        XCTAssertEqual(back.cameraPaths[0].pacing, .manual)
        XCTAssertEqual(back.cameraPaths[0].keyframes.count, 2)
    }

    func testRemovingRunKeepsStateWhileRunsRemain() {
        let p = sample().removingRun(id: "20260719-100000", hasSparse: true)
        XCTAssertEqual(p.runs.map(\.id), ["20260719-110000"])
        XCTAssertEqual(p.state, .trained)
    }

    func testRemovingLastRunDemotesState() {
        // sparse present → demoted to aligned.
        var p = sample()
        p = p.removingRun(id: "20260719-100000", hasSparse: true)
        p = p.removingRun(id: "20260719-110000", hasSparse: true)
        XCTAssertTrue(p.runs.isEmpty)
        XCTAssertEqual(p.state, .aligned)

        // no sparse → demoted to imported.
        var q = sample()
        q.runs.removeFirst()   // leave only a single run
        q = q.removingRun(id: "20260719-110000", hasSparse: false)
        XCTAssertTrue(q.runs.isEmpty)
        XCTAssertEqual(q.state, .imported)
    }

    // MARK: the open run's manifest copy (2026-08-05 FIX 1)
    //
    // ViewerModel.currentRun is a value snapshot frozen when the run was opened, and RunRecord is a
    // struct, so a manifest write that lands afterwards never reaches it. timelapseFile is the only
    // field written after the open (the encode finishes minutes later), which is why the RESULT row
    // went "encoding…" → "—" at the exact moment the file appeared. `record(matching:)` is the
    // re-read: identity from the snapshot, contents from the manifest.

    /// Nothing open → nothing to show. The Inspector renders its "Train a run to view its result"
    /// placeholder off this nil, so it must survive the indirection.
    func testRecordMatchingNilIsNil() {
        XCTAssertNil(sample().record(matching: nil))
    }

    /// **The bug, in its exact shape.** The snapshot has `timelapseFile == nil` (it was taken when
    /// the run finished, before the encode) and the manifest has the path (the encode landed).
    /// Returning the snapshot here is what made a file sitting on disk read as "—".
    func testRecordMatchingPrefersTheManifestCopy() {
        var project = sample()
        project.runs[0].timelapseFile = "runs/20260719-100000/timelapse.gif"
        project.runs[0].finalPSNR = 31.0
        let snapshot = RunRecord(id: "20260719-100000", outputFile: "output.ply",
                                 previewFile: "preview.png", outcome: "completed",
                                 finalPSNR: 24.5, splatCount: 120_000,
                                 optionsSummary: "factor 4 · 7000it",
                                 createdAt: Date(timeIntervalSince1970: 100),
                                 antialiased: false, timelapseFile: nil)

        let resolved = project.record(matching: snapshot)

        XCTAssertEqual(resolved?.id, snapshot.id, "identity still comes from the snapshot")
        XCTAssertEqual(resolved?.timelapseFile, "runs/20260719-100000/timelapse.gif")
        // Not just the one field — the whole record is the manifest's, so any later write is seen.
        XCTAssertEqual(resolved?.finalPSNR, 31.0)
        XCTAssertEqual(resolved, project.runs[0])
    }

    /// The run was deleted from the manifest while still open in the viewer. Falling back to the
    /// snapshot keeps the panel rendering instead of blanking mid-frame — deleteRun clears the
    /// viewer itself, and this view must not race it into an empty state.
    func testRecordMatchingFallsBackToTheSnapshotWhenAbsent() {
        let snapshot = RunRecord(id: "not-in-manifest", outputFile: "output.ply", previewFile: nil,
                                 outcome: "completed", finalPSNR: 20.0, splatCount: 10,
                                 optionsSummary: "x", createdAt: Date(timeIntervalSince1970: 1),
                                 antialiased: true, timelapseFile: "runs/x/timelapse.mp4")
        XCTAssertEqual(sample().record(matching: snapshot), snapshot)
    }

    /// The empty-manifest edge of the same fallback (a project whose last run was just removed).
    func testRecordMatchingWithNoRunsReturnsTheSnapshot() {
        var project = sample()
        project.runs = []
        let snapshot = RunRecord(id: "20260719-100000", outputFile: "output.ply", previewFile: nil,
                                 outcome: "completed", finalPSNR: nil, splatCount: 1,
                                 optionsSummary: "x", createdAt: Date(timeIntervalSince1970: 1))
        XCTAssertEqual(project.record(matching: snapshot), snapshot)
    }

    // MARK: auto-level (2026-08-04 spec §1.5)

    /// The toggle is per project, defaults to ON, and survives a manifest round trip. The
    /// default matters: an existing project that never saw this feature must still be
    /// auto-leveled when the user re-aligns.
    func testAutoLevelDefaultsOnAndSurvivesRoundTrip() throws {
        XCTAssertTrue(sample().autoLevelOnAlign)
        var p = sample()
        p.autoLevelOnAlign = false
        let data = try JSONEncoder.project.encode(p)
        let back = try JSONDecoder.project.decode(Project.self, from: data)
        XCTAssertFalse(back.autoLevelOnAlign)
        XCTAssertEqual(back, p)
    }

    /// A manifest written before this feature has no key at all — it must decode to the ON
    /// default rather than throwing (the decodeIfPresent forward-compatibility convention).
    func testOldManifestWithoutAutoLevelKeyDefaultsToOn() throws {
        let json = """
        {"id":"F5B2D3A0-0000-0000-0000-000000000000","name":"m","sourceKind":"colmap",
         "state":"aligned","imageCount":3,"createdAt":"2026-07-19T00:00:00Z",
         "cinemaWatermarkSeeded":true}
        """.data(using: .utf8)!
        let p = try JSONDecoder.project.decode(Project.self, from: json)
        XCTAssertTrue(p.autoLevelOnAlign)
        XCTAssertTrue(p.cinemaWatermarkSeeded)   // the sibling flag still decodes as written
    }

    /// The applied leveling angle rides inside AlignSummary. It is nil in every manifest that
    /// predates the feature, and nil is also what "auto-level was off / below the threshold"
    /// records — the UI keys the row's visibility off exactly this nil.
    func testAlignSummaryLevelTiltPersistsAndToleratesAbsence() throws {
        var p = sample()
        p.alignSummary?.levelTiltDegrees = 19.2
        let data = try JSONEncoder.project.encode(p)
        let back = try JSONDecoder.project.decode(Project.self, from: data)
        XCTAssertEqual(back.alignSummary?.levelTiltDegrees, 19.2)
        XCTAssertEqual(back, p)

        let old = #"{"alignSeconds":42.5,"poseCount":28,"pointCount":5000}"#.data(using: .utf8)!
        let s = try JSONDecoder.project.decode(AlignSummary.self, from: old)
        XCTAssertNil(s.levelTiltDegrees)
        XCTAssertEqual(s.poseCount, 28)
        XCTAssertEqual(s.pointCount, 5000)
    }

    func testStateRawValuesAreStable() {
        // The single source of truth for manifest backward compatibility — changing a
        // rawValue makes existing projects fail to open.
        XCTAssertEqual(ProjectState.imported.rawValue, "imported")
        XCTAssertEqual(ProjectState.aligned.rawValue, "aligned")
        XCTAssertEqual(ProjectState.trained.rawValue, "trained")
        XCTAssertEqual(ProjectSourceKind.images.rawValue, "images")
        XCTAssertEqual(ProjectSourceKind.video.rawValue, "video")
        XCTAssertEqual(ProjectSourceKind.colmap.rawValue, "colmap")
    }
}
