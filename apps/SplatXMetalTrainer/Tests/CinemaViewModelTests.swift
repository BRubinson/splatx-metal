// Cinema Task 3: the CinemaViewModel state machine — path CRUD, keyframe capture/update/
// delete/snap, hybrid pacing transitions (drag → manual, revert → even), playback arithmetic,
// the skimmer/preview override, and persist callback timing. The CameraPose round-trip golden
// (orbit → world pose → view matrix) is pinned here as well.
// The SwiftUI application surface (Canvas, hover) is a manual checklist — the explicit policy
// of this app.
import XCTest
import simd
@testable import SplatXMetalTrainer

@MainActor
final class CinemaViewModelTests: XCTestCase {

    // MARK: Helpers

    /// Counter for persist calls (a reference type because the onPersist closure escapes).
    private final class SaveCounter {
        var count = 0
    }

    private func key(_ p: SIMD3<Float>, time: Double = 0) -> Keyframe {
        Keyframe(time: time, pos: p, quat: SIMD4(1, 0, 0, 0), fov: 0.9)
    }

    /// A VM loaded with a straight, evenly spaced 3-key path (derived times [0, 1, 2]).
    /// Counts persist calls.
    private func makeVM(withLinePath: Bool = true,
                        counter: SaveCounter? = nil) -> CinemaViewModel {
        let vm = CinemaViewModel()
        if withLinePath {
            let path = CameraPath(name: "line", keyframes: [
                key(SIMD3(0, 0, 0)), key(SIMD3(1, 0, 0)), key(SIMD3(2, 0, 0)),
            ], ease: .none, duration: 2)
            vm.load(paths: [path])
        }
        if let counter {
            vm.onPersist = { _ in counter.count += 1 }
        }
        return vm
    }

    // MARK: Path CRUD

    func testAddPathSelectsAndPersists() {
        let saves = SaveCounter()
        let vm = makeVM(withLinePath: false, counter: saves)
        let id = vm.addPath()
        XCTAssertEqual(vm.paths.count, 1)
        XCTAssertEqual(vm.selectedPathID, id)
        XCTAssertEqual(vm.paths[0].name, "Path 1")
        XCTAssertEqual(saves.count, 1)
        _ = vm.addPath()
        XCTAssertEqual(vm.paths.map(\.name), ["Path 1", "Path 2"])   // auto-uniquified
    }

    func testDuplicateAndDeletePath() {
        let saves = SaveCounter()
        let vm = makeVM(counter: saves)
        let original = vm.paths[0].id
        vm.duplicatePath(id: original)
        XCTAssertEqual(vm.paths.count, 2)
        XCTAssertEqual(vm.paths[1].name, "line copy")
        XCTAssertNotEqual(vm.paths[1].id, original)                  // new id
        XCTAssertEqual(vm.paths[1].keyframes.count, 3)               // copied keyframes
        XCTAssertEqual(vm.selectedPathID, vm.paths[1].id)            // the copy is selected
        vm.deletePath(id: vm.paths[1].id)
        XCTAssertEqual(vm.paths.count, 1)
        XCTAssertEqual(vm.selectedPathID, original)                  // neighbor reselected
        XCTAssertEqual(saves.count, 2)
    }

    func testFinalizePathNamesSanitizesEmpty() {
        let saves = SaveCounter()
        let vm = makeVM(counter: saves)
        let id = vm.paths[0].id
        vm.renamePath(id: id, to: "   ", persistNow: false)          // mid-edit state allowed
        XCTAssertEqual(vm.paths[0].name, "   ")
        XCTAssertEqual(saves.count, 0)                                     // no persist while editing
        vm.finalizePathNames()                                       // focus-out commit
        XCTAssertEqual(vm.paths[0].name, "Path")
        XCTAssertEqual(saves.count, 1)
    }

    /// Panel redesign Task 1: commitPathRename is the only VM entry point for inline name
    /// editing — it commits the draft through trimming, fallback, and sibling uniquification
    /// (normalizedPathName), and persists once.
    /// Only the row being edited changes name (no cascading rename that pushes sibling names
    /// aside).
    func testCommitPathRenameNormalizesUniquifiesAndPersistsOnce() {
        let saves = SaveCounter()
        let vm = makeVM(withLinePath: false, counter: saves)
        vm.load(paths: [CameraPath(name: "Alpha"), CameraPath(name: "Beta")])
        let betaID = vm.paths[1].id
        vm.commitPathRename(id: betaID, draft: "  Alpha ")
        XCTAssertEqual(vm.paths.map(\.name), ["Alpha", "Alpha 2"])   // trim + sibling collision " 2"
        XCTAssertEqual(saves.count, 1)                               // one persist on commit
        vm.commitPathRename(id: betaID, draft: "   ")
        XCTAssertEqual(vm.paths[1].name, "Path")                     // empty draft → fallback
        XCTAssertEqual(saves.count, 2)
        vm.commitPathRename(id: betaID, draft: "Path")               // unchanged draft
        XCTAssertEqual(saves.count, 2)                               // → no persist either
        vm.commitPathRename(id: UUID(), draft: "Ghost")              // dead id (deleted mid-edit)
        XCTAssertEqual(vm.paths.map(\.name), ["Alpha", "Path"])      // → harmless no-op
        XCTAssertEqual(saves.count, 2)
    }

    /// Panel redesign Task 1: the cancel contract — the draft lives only in the inspector
    /// view's @State, and commitPathRename is the sole VM entry. FIX 1 ④ (final review): this
    /// **actually exercises** the cancel path — cancelling clears the commit target (nil), and
    /// the commit decision for a cleared target (CinemaInspector.renameCommit — the pure single
    /// source of truth shared by the commit gates commitRename and commitPathNameField) is nil,
    /// so there is no VM call. Previously this path existed only as a comment, so it passed
    /// even if cancel were changed to call commit (a vacuous test). FIX 2 ⑤: the cancel entries
    /// grew from one (Esc) to two (Esc and blur, i.e. clicking elsewhere) — only Return commits
    /// (renameExitCommits is the single source of truth; the decision itself is pinned by
    /// testRenameExitCommitsOnlyOnReturn in AffordanceTests).
    func testRenameCancelContractKeepsOriginal() {
        let saves = SaveCounter()
        let vm = makeVM(withLinePath: false, counter: saves)
        vm.load(paths: [CameraPath(name: "Original")])
        let id = vm.paths[0].id
        // Start editing (target = id) → type "Changed" → Esc/blur (target cleared) → the
        // commit gate: the decision is nil → no commit and no persist; the name stays original
        // by construction, without any active restore.
        XCTAssertNil(CinemaInspector.renameCommit(target: nil, draft: "Changed"))
        XCTAssertEqual(vm.paths[0].name, "Original")
        XCTAssertEqual(saves.count, 0)
        // Contrast: if the target is alive (Return commits), the decision yields (id, draft)
        // and the commit goes through.
        let commit = CinemaInspector.renameCommit(target: id, draft: "Changed")
        XCTAssertEqual(commit?.id, id)
        XCTAssertEqual(commit?.draft, "Changed")
        vm.commitPathRename(id: commit!.id, draft: commit!.draft)
        XCTAssertEqual(vm.paths[0].name, "Changed")
        XCTAssertEqual(saves.count, 1)
    }

    // MARK: Keyframe capture — add/update/insert ordering (spec UX 2: capture is one action,
    // re-capture = update)

    func testCaptureAutoCreatesPathAndAddsKey() {
        let saves = SaveCounter()
        let vm = makeVM(withLinePath: false, counter: saves)
        var camera = CameraController()
        camera.verticalFOV = 1.1
        vm.captureKeyframe(from: camera)
        XCTAssertEqual(vm.paths.count, 1)                            // path auto-created
        XCTAssertEqual(vm.paths[0].keyframes.count, 1)
        XCTAssertEqual(vm.selectedKeyframeIndex, 0)
        let k = vm.paths[0].keyframes[0]
        XCTAssertEqual(k.fov, 1.1, accuracy: 1e-6)                   // captures the orbit FOV
        XCTAssertLessThan(simd_distance(k.pos, camera.eye), 1e-4)    // world pose = eye
        XCTAssertGreaterThan(saves.count, 0)
    }

    func testCaptureAtPlayheadOnKeyUpdatesInPlace() {
        let vm = makeVM()
        vm.setPlayhead(1.0)                                          // on key 1 of derived times [0,1,2]
        var camera = CameraController()
        camera.target = SIMD3(9, 9, 9)
        vm.captureKeyframe(from: camera)
        XCTAssertEqual(vm.paths[0].keyframes.count, 3)               // update — count unchanged
        XCTAssertEqual(vm.selectedKeyframeIndex, 1)
        XCTAssertLessThan(simd_distance(vm.paths[0].keyframes[1].pos, camera.eye), 1e-4)
    }

    func testCaptureBetweenKeysInsertsInTimeOrder() {
        let vm = makeVM()
        vm.setPlayhead(0.5)                                          // between key 0 (0) and key 1 (1)
        vm.captureKeyframe(from: CameraController())
        XCTAssertEqual(vm.paths[0].keyframes.count, 4)
        XCTAssertEqual(vm.selectedKeyframeIndex, 1)                  // inserted at its time-ordered position
    }

    func testCaptureTargetIndexTolerance() {
        // Half-frame threshold (@30fps = 1/60) — inside it is the update target, outside it
        // is nil (insert).
        XCTAssertEqual(CinemaViewModel.captureTargetIndex(
            playhead: 1.0, times: [0, 1.01, 2], tolerance: 1.0 / 60.0), 1)
        XCTAssertNil(CinemaViewModel.captureTargetIndex(
            playhead: 1.0, times: [0, 1.01, 2], tolerance: 0.005))
        // With multiple candidates, the nearest one wins.
        XCTAssertEqual(CinemaViewModel.captureTargetIndex(
            playhead: 1.0, times: [0.99, 1.005], tolerance: 0.02), 1)
        XCTAssertNil(CinemaViewModel.captureTargetIndex(
            playhead: 0, times: [], tolerance: 1))
    }

    func testDeleteSelectedKeyframe() {
        let saves = SaveCounter()
        let vm = makeVM(counter: saves)
        vm.selectKeyframe(1)
        vm.deleteSelectedKeyframe()
        XCTAssertEqual(vm.paths[0].keyframes.count, 2)
        XCTAssertNil(vm.selectedKeyframeIndex)
        XCTAssertEqual(saves.count, 1)
        vm.deleteSelectedKeyframe()                                  // no selection — no-op
        XCTAssertEqual(vm.paths[0].keyframes.count, 2)
    }

    // Footnote 2 ③: indexed deletion from the timeline diamond context menu — the selection
    // index is fixed up.
    func testDeleteKeyframeAtIndexAdjustsSelection() {
        let saves = SaveCounter()
        let vm = makeVM(counter: saves)
        vm.selectKeyframe(2)
        vm.deleteKeyframe(at: 0)                                     // delete a key ahead of the selection (2)
        XCTAssertEqual(vm.paths[0].keyframes.count, 2)
        XCTAssertEqual(vm.selectedKeyframeIndex, 1)                  // 2 → 1 shift
        XCTAssertEqual(saves.count, 1)
        vm.deleteKeyframe(at: 1)                                     // delete the selected key itself
        XCTAssertNil(vm.selectedKeyframeIndex)
        XCTAssertEqual(vm.paths[0].keyframes.count, 1)
        XCTAssertEqual(saves.count, 2)
        vm.deleteKeyframe(at: 5)                                     // out of range — no-op
        XCTAssertEqual(vm.paths[0].keyframes.count, 1)
        XCTAssertEqual(saves.count, 2)
    }

    // MARK: Hybrid pacing state machine (spec footnote 1)

    // A click (a drag with no movement) only selects — even pacing is kept. Actual movement
    // is what switches to manual.
    func testKeyDragClickOnlySelectsKeepsEven() {
        let saves = SaveCounter()
        let vm = makeVM(counter: saves)
        vm.beginKeyframeDrag(1)
        vm.endKeyframeDrag()
        XCTAssertEqual(vm.paths[0].pacing, .even)
        XCTAssertEqual(vm.selectedKeyframeIndex, 1)
        XCTAssertEqual(vm.playhead, 1.0, accuracy: 1e-6)             // selection = jump to the key time
        XCTAssertEqual(saves.count, 0)                                     // no mutation — no persist
    }

    // A drag that moves: the derived times [0,1,2] are promoted → manual, only the dragged
    // key moves, and the persist happens once at end.
    func testKeyDragMovePromotesToManualPacing() {
        let saves = SaveCounter()
        let vm = makeVM(counter: saves)
        vm.beginKeyframeDrag(1)
        vm.updateKeyframeDrag(to: 1.5)
        vm.updateKeyframeDrag(to: 1.6)
        XCTAssertEqual(vm.paths[0].pacing, .manual)
        XCTAssertEqual(vm.paths[0].keyframes.map(\.time), [0, 1.6, 2])
        XCTAssertEqual(vm.keyframeTimes[1], 1.6, accuracy: 1e-9)     // diamond = stored time
        XCTAssertEqual(saves.count, 0)                                     // no persist during the drag
        vm.endKeyframeDrag()
        XCTAssertEqual(saves.count, 1)                                     // once at the end
    }

    // The neighbor clamp reserves a half-frame (0.5/fps = 1/60 s) gap — this prevents keys at
    // the same time (an instant teleport during playback) (FIX 1 M2). The range bounds
    // [0, duration] are not neighbors, so they apply as-is with no gap.
    func testKeyDragClampsBetweenNeighborsAndRange() {
        let vm = makeVM()                                            // fps 30 → gap 1/60
        let gap = 0.5 / 30.0
        vm.beginKeyframeDrag(1)
        vm.updateKeyframeDrag(to: 5.0)                               // neighbor key 2 (2.0) − half a frame
        XCTAssertEqual(vm.paths[0].keyframes[1].time, 2.0 - gap, accuracy: 1e-9)
        vm.updateKeyframeDrag(to: -3.0)                              // neighbor key 0 (0) + half a frame
        XCTAssertEqual(vm.paths[0].keyframes[1].time, 0.0 + gap, accuracy: 1e-9)
        vm.endKeyframeDrag()
        // The pure decision directly: the first/last keys are bounded by [0, duration] (the
        // gap applies only to neighbor keys).
        XCTAssertEqual(CinemaViewModel.clampedDragTime(-1, index: 0, times: [0, 1, 2],
                                                       duration: 2, minSpacing: gap), 0)
        XCTAssertEqual(CinemaViewModel.clampedDragTime(9, index: 2, times: [0, 1, 2],
                                                       duration: 2, minSpacing: gap), 2)
        // Neighbor spacing < 2·gap (over-dense data from before the clamp) — degrades to an
        // order-preserving gapless clamp.
        XCTAssertEqual(CinemaViewModel.clampedDragTime(9, index: 1, times: [1, 1, 1],
                                                       duration: 2, minSpacing: gap), 1)
    }

    // FIX 1 (I2): a no-move drag tick that snapping quantizes back to its original position
    // (a change smaller than half a frame) causes neither promotion to manual nor a persist —
    // the time-axis counterpart of click protection.
    func testMicroDragWithoutTimeChangeDoesNotPromote() {
        let saves = SaveCounter()
        let vm = makeVM(counter: saves)
        vm.beginKeyframeDrag(1)
        vm.updateKeyframeDrag(to: 1.0)                               // snap result = the current derived time
        vm.updateKeyframeDrag(to: 1.0 + 0.25 / 30.0)                 // movement below half a frame (1/60)
        XCTAssertEqual(vm.paths[0].pacing, .even)                    // no promotion
        XCTAssertEqual(vm.keyframeTimes[1], 1.0, accuracy: 1e-6)     // derived time kept
        vm.endKeyframeDrag()
        XCTAssertEqual(saves.count, 0)                               // equivalent to a click — no persist
    }

    // FIX 1 (M3): on even+easeInOut, promotion by drag stores times that match the quintic
    // inverse derived values (keyframeTimes) — a regression guard that promotion preserves the
    // shape of the ease time axis.
    // Lineage: a straight, evenly spaced 5-key path with duration 4 → derived
    // [0, 4·qinv(0.25), 2, 4·qinv(0.75), 4] (the same axis as
    // CameraPathPacingTests.testKeyframeTimesEvenWithEaseInverse).
    func testPromotionUnderEasePreservesDerivedTimes() {
        let vm = CinemaViewModel()
        let keys = (0..<5).map { key(SIMD3(Float($0), 0, 0)) }
        vm.load(paths: [CameraPath(name: "eased", keyframes: keys, ease: .easeInOut,
                                   duration: 4)])
        let derived = vm.keyframeTimes
        XCTAssertEqual(derived[1], 4 * 0.35944, accuracy: 2e-3)      // qinv(0.25) hand-check lineage
        XCTAssertEqual(derived[3], 4 * 0.64056, accuracy: 2e-3)
        vm.beginKeyframeDrag(2)
        vm.updateKeyframeDrag(to: 2.3)                               // real movement — promotes
        vm.endKeyframeDrag()
        XCTAssertEqual(vm.paths[0].pacing, .manual)
        let stored = vm.paths[0].keyframes.map(\.time)
        XCTAssertEqual(stored[2], 2.3, accuracy: 1e-9)               // the dragged key
        for i in [0, 1, 3, 4] {                                      // the rest = derived values promoted as-is
            XCTAssertEqual(stored[i], derived[i], accuracy: 1e-9, "key \(i)")
        }
        XCTAssertEqual(vm.keyframeTimes, stored)                     // in manual, the stored times are authoritative
    }

    func testRevertToEvenPacing() {
        let vm = makeVM()
        vm.beginKeyframeDrag(1)
        vm.updateKeyframeDrag(to: 1.5)
        vm.endKeyframeDrag()
        XCTAssertEqual(vm.paths[0].pacing, .manual)
        vm.revertToEvenPacing()
        XCTAssertEqual(vm.paths[0].pacing, .even)
        XCTAssertEqual(vm.keyframeTimes[1], 1.0, accuracy: 1e-6)     // back to the derived times
    }

    // In manual, changing duration rescales the key times proportionally (this preserves the
    // pacing shape — Task 3 decision).
    func testManualDurationRescalesKeyTimes() {
        let vm = makeVM()
        vm.beginKeyframeDrag(1)
        vm.updateKeyframeDrag(to: 1.5)
        vm.endKeyframeDrag()
        vm.setDuration(4)
        XCTAssertEqual(vm.paths[0].keyframes.map(\.time), [0, 3.0, 4.0])
        XCTAssertEqual(vm.paths[0].duration, 4)
        // even paths are not rescaled (they are derived, so they scale automatically)
        vm.revertToEvenPacing()
        vm.setDuration(8)
        XCTAssertEqual(vm.keyframeTimes, [0, 4, 8])
    }

    /// FIX 4 ①: a scrub release re-submits **the same absolute target value** as the last
    /// tick (ScrubbableNumberField.onEnded — a consequence of absolute-target arithmetic). If
    /// setDuration's unchanged-value early return swallows that re-submission, the
    /// persistNow: false commits made during the drag stay unpersisted forever (they vanish on
    /// restart) — the persist obligation must be discharged even when the value is identical.
    /// The Duration counterpart of testTransformPersistTiming ("0 during the drag, 1 at the
    /// end").
    func testDurationScrubPersistTiming() {
        let saves = SaveCounter()
        let vm = makeVM(counter: saves)                       // duration 2
        vm.setDuration(3, persistNow: false)                  // drag tick (mutation)
        vm.setDuration(3, persistNow: false)                  // same-value tick (lower-clamp region)
        XCTAssertEqual(saves.count, 0)                        // no persist during the drag
        vm.setDuration(3, persistNow: true)                   // release — same value as the last tick
        XCTAssertEqual(saves.count, 1)                        // still discharges one persist
        XCTAssertEqual(vm.paths[0].duration, 3)
    }

    /// The other side of FIX 4 ①: an unchanged commit with **no** unpersisted mutation
    /// (typing the same value and pressing Return, or a duplicate release) still does not
    /// persist, as before — proof that the early return's justification is preserved.
    func testDurationNoOpConfirmDoesNotPersist() {
        let saves = SaveCounter()
        let vm = makeVM(counter: saves)                       // duration 2
        vm.setDuration(2, persistNow: true)                   // the current value as-is — no mutation
        XCTAssertEqual(saves.count, 0)
        vm.setDuration(3, persistNow: true)                   // mutating commit — one persist
        XCTAssertEqual(saves.count, 1)
        vm.setDuration(3, persistNow: true)                   // already persisted — no extra persist
        XCTAssertEqual(saves.count, 1)
    }

    /// Pinning the FIX 4 ① audit: a Radius or Height release always goes through commit →
    /// persist even when it re-submits a factor of ≈1 or a delta of 0 (neither setter has an
    /// unchanged-value early return) — so they have none of Duration's vanishing persists.
    /// This doubles as a warning: adding a value-comparison guard to these two setters makes
    /// the scrub-release persist vanish (see setDuration's flush history).
    func testTransformNoOpReleaseStillPersists() {
        let saves = SaveCounter()
        let vm = makeVM(counter: saves)
        vm.translateSelectedPath(by: .zero, persistNow: false)   // non-mutating drag tick
        vm.scaleSelectedPath(by: 1, persistNow: false)
        XCTAssertEqual(saves.count, 0)
        vm.translateSelectedPath(by: .zero, persistNow: true)    // release re-submission (delta 0)
        XCTAssertEqual(saves.count, 1)
        vm.scaleSelectedPath(by: 1, persistNow: true)            // release re-submission (factor 1)
        XCTAssertEqual(saves.count, 2)
    }

    // MARK: Playback and frame stepping

    func testPlaybackPositionArithmetic() {
        // Non-loop: on reaching duration, (duration, finished)
        var r = CinemaViewModel.playbackPosition(anchorTime: 0, elapsed: 2.5, duration: 2,
                                                 loop: false)
        XCTAssertEqual(r.time, 2); XCTAssertTrue(r.finished)
        r = CinemaViewModel.playbackPosition(anchorTime: 0.5, elapsed: 1, duration: 2,
                                             loop: false)
        XCTAssertEqual(r.time, 1.5, accuracy: 1e-9); XCTAssertFalse(r.finished)
        // Loop: wraps by the remainder, never finished
        r = CinemaViewModel.playbackPosition(anchorTime: 0, elapsed: 2.5, duration: 2,
                                             loop: true)
        XCTAssertEqual(r.time, 0.5, accuracy: 1e-9); XCTAssertFalse(r.finished)
        // duration 0 guard
        r = CinemaViewModel.playbackPosition(anchorTime: 0, elapsed: 1, duration: 0,
                                             loop: true)
        XCTAssertTrue(r.finished)
    }

    func testTogglePlaybackFromEndRestartsAndFinishes() {
        let vm = makeVM()
        vm.setPlayhead(2.0)                                          // play from the end → restarts from the beginning
        vm.togglePlayback()
        XCTAssertTrue(vm.isPlaying)
        XCTAssertEqual(vm.playhead, 0, accuracy: 1e-9)
        vm.playbackTick(now: Date(timeIntervalSinceNow: 60))         // tick past duration
        XCTAssertFalse(vm.isPlaying)                                 // reached the end — stops
        XCTAssertEqual(vm.playhead, 2, accuracy: 1e-9)
    }

    func testStepFrameFollowsFrameGrid() {
        let vm = makeVM()                                            // duration 2, fps 30
        vm.stepFrame(1)
        XCTAssertEqual(vm.playhead, 1.0 / 30.0, accuracy: 1e-9)
        vm.stepFrame(-1)
        XCTAssertEqual(vm.playhead, 0, accuracy: 1e-9)
        vm.stepFrame(-1)                                             // clamps below 0
        XCTAssertEqual(vm.playhead, 0, accuracy: 1e-9)
        vm.setPlayhead(2.0)
        vm.stepFrame(1)                                              // clamps above duration
        XCTAssertEqual(vm.playhead, 2.0, accuracy: 1e-9)
    }

    // MARK: Skimmer and preview override (spec UX 3 — the contract that orbit is never
    // hijacked)

    func testSkimAppliesAndReleasesOverride() {
        let vm = makeVM()
        let viewer = ViewerModel()
        vm.viewer = viewer
        vm.setActive(true)
        vm.skim(to: 0.5)
        let pose = try? XCTUnwrap(viewer.camera.overridePose)
        XCTAssertEqual(pose?.position.x ?? -1, 0.5, accuracy: 1e-4)  // straight, even path at t=0.5 → x=0.5
        vm.skim(to: nil)                                             // exit — no preview residue
        XCTAssertNil(viewer.camera.overridePose)                     // back to the original camera
    }

    func testSkimExitFallsBackToScrubPreview() {
        let vm = makeVM()
        let viewer = ViewerModel()
        vm.viewer = viewer
        vm.setActive(true)
        vm.setPlayhead(1.0, preview: true)                           // scrub — explicit navigation
        vm.skim(to: 0.25)
        XCTAssertEqual(viewer.camera.overridePose?.position.x ?? -1, 0.25, accuracy: 1e-4)
        vm.skim(to: nil)                                             // exit → keeps the scrubbed frame
        XCTAssertEqual(viewer.camera.overridePose?.position.x ?? -1, 1.0, accuracy: 1e-4)
    }

    // Footnote 2 ②: a user orbit during skimming releases the override but **adopts the
    // visible pose into orbit** before continuing — it does not snap back (teleport) to the
    // orbit state from before the skim.
    // Adoption arithmetic: pose (0.5,0,0) looking along −Z; the old target (0,0,0) has a
    // view-direction depth of 0 (it is off to the side) → the OrbitFit lower bound of 0.5 →
    // target (0.5,0,−0.5), distance 0.5, yaw/pitch 0.
    func testUserOrbitAdoptsDisplayedPoseFromOverride() {
        let vm = makeVM()
        let viewer = ViewerModel()
        vm.viewer = viewer
        vm.setActive(true)
        vm.skim(to: 0.5)
        let pose = viewer.camera.overridePose
        XCTAssertNotNil(pose)
        viewer.camera.orbit(dx: 10, dy: 0)                           // user drag
        XCTAssertNil(viewer.camera.overridePose)                     // the override is released immediately
        XCTAssertEqual(viewer.camera.distance, 0.5, accuracy: 1e-4)  // adopted depth (the lower bound)
        XCTAssertLessThan(simd_distance(viewer.camera.target, SIMD3(0.5, 0, -0.5)), 1e-3)
        XCTAssertEqual(viewer.camera.yaw, 0.1, accuracy: 1e-4)       // adopted yaw 0 + drag 0.1
        XCTAssertEqual(viewer.camera.pitch, 0, accuracy: 1e-4)       // with the previous behavior (snap back) 0.3 would remain
        // No jump — right after the drag the camera sits near the pose that was visible (with
        // the previous behavior it would fly off to the distance-5 orbit).
        XCTAssertLessThan(simd_distance(viewer.camera.eye, pose!.position), 0.1)
    }

    func testSelectKeyframeJumpsViewport() {
        let vm = makeVM()
        let viewer = ViewerModel()
        vm.viewer = viewer
        vm.setActive(true)
        vm.selectKeyframe(2)
        XCTAssertEqual(vm.playhead, 2.0, accuracy: 1e-6)             // playhead = key time
        XCTAssertEqual(viewer.camera.overridePose?.position.x ?? -1, 2.0, accuracy: 1e-3)
    }

    // FIX 1 (I1): a user camera manipulation during playback (orbit — which releases the
    // override) stops playback on the next tick — the tick must not put the override back on
    // and make the pose flicker between orbit and the path.
    func testUserCameraGrabDuringPlaybackStopsPlayback() {
        let vm = makeVM()
        let viewer = ViewerModel()
        vm.viewer = viewer
        vm.setActive(true)
        vm.togglePlayback()
        XCTAssertTrue(vm.isPlaying)
        vm.playbackTick(now: Date(timeIntervalSinceNow: 0.05))       // tick 1 — puts the override on
        XCTAssertNotNil(viewer.camera.overridePose)
        viewer.camera.orbit(dx: 5, dy: 0)                            // simulate a user manipulation
        XCTAssertNil(viewer.camera.overridePose)                     // the manipulation releases it immediately
        vm.playbackTick(now: Date(timeIntervalSinceNow: 0.1))        // next tick
        XCTAssertFalse(vm.isPlaying)                                 // the stop-playback contract
        XCTAssertNil(viewer.camera.overridePose)                     // not reapplied — the camera is kept
    }

    func testDeactivateClearsOverrideAndPlayback() {
        let vm = makeVM()
        let viewer = ViewerModel()
        vm.viewer = viewer
        vm.setActive(true)
        vm.togglePlayback()
        vm.skim(to: 0.5)
        vm.setActive(false)
        XCTAssertFalse(vm.isPlaying)
        XCTAssertNil(vm.skimTime)
        XCTAssertNil(viewer.camera.overridePose)                     // back to orbit
    }

    // Capture takes the effective pose on screen (WYSIWYG) — while an override is active, it
    // captures that pose.
    func testCaptureUsesEffectivePoseWhileOverridden() {
        let vm = makeVM()
        let viewer = ViewerModel()
        vm.viewer = viewer
        vm.setActive(true)
        vm.selectKeyframe(2)                                         // override with the key 2 pose
        vm.setPlayhead(0.5)                                          // outside the threshold — insert path
        vm.captureKeyframe()
        XCTAssertEqual(vm.paths[0].keyframes.count, 4)
        // The inserted key (index 1) is positioned at the override pose (key 2, x=2).
        XCTAssertEqual(vm.paths[0].keyframes[1].pos.x, 2.0, accuracy: 1e-3)
    }

    // MARK: The camera follows on the rail while values are edited (app finishing round 6
    // Task 2 — the commit(source:) policy. Being on the rail is decided by an identity
    // comparison against the ownership ledger (lastAppliedPreviewPose); the origin is
    // EditSource — the authoritative contract for the policy and its rationale is the
    // corresponding comment in CinemaViewModel plus campath spec footnote 9.)

    /// On the rail (an override placed by selecting a key) + an inspector Radius edit
    /// (scaleSelectedPath defaults to .inspector) → the override is updated to **the pose of
    /// the new geometry**. Both the scrub ticks (persistNow: false) and the release follow —
    /// real-time following is the core of the request.
    func testInspectorScaleFollowsCameraOnRail() {
        let vm = makeVM()
        let viewer = ViewerModel()
        vm.viewer = viewer
        vm.setActive(true)
        vm.selectKeyframe(2)                                         // on the rail — t=2, x=2
        XCTAssertEqual(viewer.camera.overridePose?.position.x ?? -1, 2.0, accuracy: 1e-3)
        vm.scaleSelectedPath(by: 2, persistNow: false)               // scrub tick — pivot (1,0,0) ×2
        XCTAssertEqual(viewer.camera.overridePose?.position.x ?? -1, 3.0, accuracy: 1e-3)
        vm.scaleSelectedPath(by: 1, persistNow: true)                // release re-submission (factor 1)
        XCTAssertEqual(viewer.camera.overridePose?.position.x ?? -1, 3.0, accuracy: 1e-3)
    }

    /// Height (translate) and Duration go through the same policy — one case each.
    func testInspectorTranslateAndDurationFollowOnRail() {
        let vm = makeVM()
        let viewer = ViewerModel()
        vm.viewer = viewer
        vm.setActive(true)
        vm.selectKeyframe(1)                                         // on the rail — t=1, x=1
        vm.translateSelectedPath(by: SIMD3(0, 5, 0), persistNow: true)
        XCTAssertEqual(viewer.camera.overridePose?.position.y ?? -1, 5.0, accuracy: 1e-3)
        // Duration 2→4: previewTime 1 stays put, and with even derivation t=1 maps to
        // s=0.25 → x=0.5.
        vm.setDuration(4)
        XCTAssertEqual(viewer.camera.overridePose?.position.x ?? -1, 0.5, accuracy: 1e-3)
    }

    /// After the user orbits off the rail (override nil), an inspector edit leaves the camera
    /// untouched — the whole orbit state (target/distance/yaw/pitch) is preserved.
    func testInspectorEditAfterUserOrbitDoesNotTouchCamera() {
        let vm = makeVM()
        let viewer = ViewerModel()
        vm.viewer = viewer
        vm.setActive(true)
        vm.selectKeyframe(2)
        viewer.camera.orbit(dx: 5, dy: 0)                            // leaves the rail — adopt, then release
        XCTAssertNil(viewer.camera.overridePose)
        let before = viewer.camera
        vm.scaleSelectedPath(by: 2, persistNow: true)
        XCTAssertNil(viewer.camera.overridePose)                     // not reapplied
        XCTAssertEqual(viewer.camera, before)                        // the whole camera is untouched
    }

    /// An inspector edit while no preview was ever placed (nil ↔ nil) touches nothing — it
    /// does not conjure an override that did not exist (never hijack a user who is orbiting
    /// freely).
    func testInspectorEditWithoutPriorPreviewDoesNotTouchCamera() {
        let vm = makeVM()
        let viewer = ViewerModel()
        vm.viewer = viewer
        vm.setActive(true)
        XCTAssertNil(viewer.camera.overridePose)
        vm.translateSelectedPath(by: SIMD3(0, 5, 0), persistNow: true)
        XCTAssertNil(viewer.camera.overridePose)
    }

    /// Even on the rail, an edit whose origin is **the viewport** (a gizmo handle) does not
    /// follow — neither on the drag ticks nor on the release. The path is mutated and only the
    /// camera stays put (a stable frame of reference — EditSource ②).
    func testViewportEditDoesNotFollowEvenOnRail() {
        let vm = makeVM()
        let viewer = ViewerModel()
        vm.viewer = viewer
        vm.setActive(true)
        vm.selectKeyframe(2)                                         // on the rail — pose x=2
        let before = viewer.camera.overridePose
        XCTAssertNotNil(before)
        vm.translateSelectedPath(by: SIMD3(0, 5, 0), persistNow: false, source: .viewport)
        vm.translateSelectedPath(by: .zero, persistNow: true, source: .viewport)
        XCTAssertEqual(viewer.camera.overridePose, before)           // camera unchanged
        XCTAssertEqual(vm.paths[0].keyframes[2].pos.y, 5, accuracy: 1e-6)  // the path is mutated
    }

    // MARK: Scene up-axis snapshot (path transform FIX 8 — CinemaViewModel holds the
    // snapshot so that the inspector does not read viewer.camera from its body. SwiftUI
    // invalidation itself is not tested — the explicit policy of this app. Only the
    // correctness of the snapshot value and the wiring of when it refreshes are covered here.)

    /// The single source of truth for normalization — non-unit input is normalized, and a
    /// degenerate norm ≈ 0 falls back to +Y (moved over from the previous inspector logic).
    func testNormalizedUpAxisNormalizesAndFallsBack() {
        XCTAssertEqual(CinemaViewModel.normalizedUpAxis(SIMD3(0, 0, 2)), SIMD3(0, 0, 1))
        XCTAssertEqual(CinemaViewModel.normalizedUpAxis(.zero), SIMD3(0, 1, 0))
        XCTAssertEqual(CinemaViewModel.normalizedUpAxis(SIMD3(0, -1, 0)), SIMD3(0, -1, 0))
    }

    /// With no viewer wired (test stubs and the like) the default is +Y — the same as the
    /// previous inspector fallback.
    func testSceneUpAxisDefaultsPlusYWithoutViewer() {
        XCTAssertEqual(CinemaViewModel().sceneUpAxis, SIMD3(0, 1, 0))
    }

    /// Wiring the viewer (didSet) takes the snapshot immediately — a non-unit up axis is
    /// normalized as it is taken.
    func testSceneUpAxisSnapshotsOnViewerWiring() {
        let vm = makeVM()
        let viewer = ViewerModel()
        viewer.camera.upAxis = SIMD3(0, 0, 2)
        vm.viewer = viewer
        XCTAssertEqual(vm.sceneUpAxis, SIMD3(0, 0, 1))
    }

    /// When a scene load (applyCameraFit) decides the up axis, the snapshot follows via the
    /// onUpAxisChanged notification — the automatic run-load path while the Cinema tab is
    /// active. For an identity-pose view, upWorld is (0,−1,0) (COLMAP has +Y pointing down —
    /// reuses the CameraFitTests fixture).
    func testSceneUpAxisFollowsCameraFit() {
        let vm = makeVM()
        let viewer = ViewerModel()
        vm.viewer = viewer
        XCTAssertEqual(vm.sceneUpAxis, SIMD3(0, 1, 0))
        let view = ColmapView(id: 1, name: "a.jpg", cameraId: 7,
                              qvec: SIMD4(1, 0, 0, 0), tvec: SIMD3(0, 0, 0))
        let k = ColmapIntrinsics(fx: 1440, fy: 1440, cx: 960, cy: 720,
                                 width: 1920, height: 1440)
        let ctx = ColmapContext(model: ColmapModel(cameras: [7: k], views: [view]),
                                middleView: view, pointsCentroid: SIMD3(0, 0, 4))
        XCTAssertTrue(viewer.applyCameraFit(context: ctx))
        XCTAssertEqual(vm.sceneUpAxis, SIMD3(0, -1, 0))
    }

    /// The defensive refresh in setActive (entering Cinema) — even on paths where the
    /// notification wiring is broken, entry catches up.
    func testSceneUpAxisRefreshesOnCinemaEntry() {
        let vm = makeVM()
        let viewer = ViewerModel()
        vm.viewer = viewer
        viewer.onUpAxisChanged = nil                 // assume the notification is lost
        viewer.camera.upAxis = SIMD3(1, 0, 0)
        vm.setActive(true)
        XCTAssertEqual(vm.sceneUpAxis, SIMD3(1, 0, 0))
    }

    // MARK: Whole-path transforms (path transform Task 2 — spec §3 and §5. The math is
    // pinned by PathTransformTests; only the VM wiring is here: restriction to the selected
    // path, persist timing via commit, and no change on a guard violation.)

    /// Translation (§3.3) changes only the selected path — unselected paths, quat, and fov
    /// are untouched, and it persists once.
    func testTranslateSelectedPathMovesOnlySelectedPath() {
        let saves = SaveCounter()
        let vm = CinemaViewModel()
        let a = CameraPath(name: "a", keyframes: [
            key(SIMD3(0, 0, 0)), key(SIMD3(1, 0, 0)), key(SIMD3(2, 0, 0)),
        ], ease: .none, duration: 2)
        let b = CameraPath(name: "b", keyframes: [key(SIMD3(5, 5, 5))], duration: 2)
        vm.load(paths: [a, b])                                       // the first path (a) is selected
        vm.onPersist = { _ in saves.count += 1 }
        vm.translateSelectedPath(by: SIMD3(1, 2, 3), persistNow: true)
        XCTAssertEqual(vm.paths[0].keyframes.map(\.pos),
                       [SIMD3(1, 2, 3), SIMD3(2, 2, 3), SIMD3(3, 2, 3)])
        XCTAssertEqual(vm.paths[1].keyframes[0].pos, SIMD3(5, 5, 5)) // unselected path unchanged
        XCTAssertEqual(vm.paths[0].keyframes.map(\.quat),
                       a.keyframes.map(\.quat))                      // quat is byte-identical
        XCTAssertEqual(vm.paths[0].keyframes.map(\.fov), a.keyframes.map(\.fov))
        XCTAssertEqual(saves.count, 1)
    }

    /// Scaling (§3.4) is concentric about the pivot (a line path with parallel view
    /// directions → centroid fallback) — the distance ratio equals the factor, quat is
    /// untouched, and meanRadius scales by the factor as well.
    func testScaleSelectedPathAboutPivot() {
        let saves = SaveCounter()
        let vm = makeVM(counter: saves)
        let pivot = vm.selectedPivot
        XCTAssertNotNil(pivot)
        XCTAssertEqual(pivot?.isAimLocked, false)                    // parallel view directions — not aim-locked
        XCTAssertLessThan(simd_distance(pivot?.point ?? .zero, SIMD3(1, 0, 0)), 1e-5)
        XCTAssertEqual(pivot?.meanRadius ?? -1, 2.0 / 3.0, accuracy: 1e-5)
        vm.scaleSelectedPath(by: 2, persistNow: true)
        let xs = vm.paths[0].keyframes.map(\.pos.x)                  // concentric about pivot (1,0,0), ×2
        XCTAssertEqual(xs[0], -1, accuracy: 1e-5)
        XCTAssertEqual(xs[1], 1, accuracy: 1e-5)
        XCTAssertEqual(xs[2], 3, accuracy: 1e-5)
        XCTAssertEqual(vm.selectedPivot?.meanRadius ?? -1, 4.0 / 3.0, accuracy: 1e-5)
        XCTAssertEqual(vm.paths[0].keyframes.map(\.quat),
                       [SIMD4<Float>(1, 0, 0, 0), SIMD4(1, 0, 0, 0), SIMD4(1, 0, 0, 0)])
        XCTAssertEqual(saves.count, 1)
    }

    /// A guard violation (§3.4: the resulting R̄ < 1e-3 after clamping) → no change and no
    /// persist. A factor of 1e-6 is clamped to 0.05, but on a tiny-radius path (R̄ ≈ 0.0027)
    /// the result still violates the lower bound.
    func testScaleGuardViolationLeavesPathUnchanged() {
        let saves = SaveCounter()
        let vm = CinemaViewModel()
        let tiny = CameraPath(name: "tiny", keyframes: [
            key(SIMD3(0, 0, 0)), key(SIMD3(0.004, 0, 0)), key(SIMD3(0.008, 0, 0)),
        ], duration: 2)
        vm.load(paths: [tiny])
        vm.onPersist = { _ in saves.count += 1 }
        let before = vm.paths[0].keyframes.map(\.pos)
        vm.scaleSelectedPath(by: 1e-6, persistNow: true)
        XCTAssertEqual(vm.paths[0].keyframes.map(\.pos), before)     // unchanged
        XCTAssertEqual(saves.count, 0)                               // not persisted
    }

    /// persistNow=false (mid-drag) does not call the persist callback, and true (onEnded)
    /// calls it exactly once — the convention of one manifest save per drag (the same as the
    /// existing drag-persist convention).
    func testTransformPersistTiming() {
        let saves = SaveCounter()
        let vm = makeVM(counter: saves)
        vm.translateSelectedPath(by: SIMD3(0.1, 0, 0), persistNow: false)
        vm.translateSelectedPath(by: SIMD3(0.1, 0, 0), persistNow: false)
        XCTAssertEqual(saves.count, 0)                               // no persist during the drag
        vm.translateSelectedPath(by: SIMD3(0.1, 0, 0), persistNow: true)
        XCTAssertEqual(saves.count, 1)                               // once at the end
        XCTAssertEqual(vm.paths[0].keyframes[0].pos.x, 0.3, accuracy: 1e-6)  // incremental sum
        vm.scaleSelectedPath(by: 1.5, persistNow: false)
        XCTAssertEqual(saves.count, 1)
        vm.scaleSelectedPath(by: 1.1, persistNow: true)
        XCTAssertEqual(saves.count, 2)
    }

    /// Harmless with no selected path and with a zero-key path — a no-op with no persist, and
    /// selectedPivot is nil too (the §4.3 display guard).
    func testTransformWithoutSelectionOrKeysIsHarmless() {
        let saves = SaveCounter()
        let vm = makeVM(withLinePath: false, counter: saves)
        XCTAssertNil(vm.selectedPivot)                               // no path
        vm.translateSelectedPath(by: SIMD3(1, 0, 0), persistNow: true)
        vm.scaleSelectedPath(by: 2, persistNow: true)
        XCTAssertEqual(saves.count, 0)
        vm.load(paths: [CameraPath(name: "empty")])                  // zero keyframes
        XCTAssertNil(vm.selectedPivot)                               // pivot(for: []) == nil
        vm.translateSelectedPath(by: SIMD3(1, 0, 0), persistNow: true)
        vm.scaleSelectedPath(by: 2, persistNow: true)
        XCTAssertEqual(saves.count, 0)
        XCTAssertTrue(vm.paths[0].keyframes.isEmpty)
    }

    /// A single-key path — the centroid fallback pivot has meanRadius == 0 (a static shot).
    /// This is the value that decides whether the radius handle is hidden (brief Interfaces),
    /// and scaling leaves it unchanged because of the resulting-R̄ lower-bound guard.
    func testSelectedPivotSingleKeyHasZeroMeanRadius() {
        let vm = CinemaViewModel()
        vm.load(paths: [CameraPath(name: "still", keyframes: [key(SIMD3(3, 1, 2))],
                                   duration: 2)])
        let pivot = vm.selectedPivot
        XCTAssertEqual(pivot?.meanRadius, 0)
        XCTAssertLessThan(simd_distance(pivot?.point ?? .zero, SIMD3(3, 1, 2)), 1e-6)
        vm.scaleSelectedPath(by: 2, persistNow: true)                // R̄·s = 0 < 1e-3 — rejected
        XCTAssertEqual(vm.paths[0].keyframes[0].pos, SIMD3(3, 1, 2))
    }

    // MARK: Derived snapshots (path transform FIX 15 — removes the per-frame recomputation)

    /// The pivot **snapshot** follows keyframe mutations — it yields the same values as the
    /// computed-property era, but the refresh happens in exactly one place, rebuildSampler()
    /// (the recomputation in the view body is gone).
    func testSelectedPivotSnapshotTracksKeyframeEdits() {
        let vm = makeVM()
        XCTAssertLessThan(simd_distance(vm.selectedPivot?.point ?? .zero,
                                        SIMD3(1, 0, 0)), 1e-5)
        vm.translateSelectedPath(by: SIMD3(0, 5, 0), persistNow: true)
        XCTAssertLessThan(simd_distance(vm.selectedPivot?.point ?? .zero,
                                        SIMD3(1, 5, 0)), 1e-5)      // snapshot refreshed
        vm.captureKeyframe(from: CameraController())                // the capture path also goes through commit
        XCTAssertEqual(vm.selectedPivot?.meanRadius ?? -1,
                       PathTransform.pivot(for: vm.paths[0].keyframes)!.meanRadius,
                       accuracy: 1e-5)
        vm.deletePath(id: vm.paths[0].id)                           // path gone → snapshot nil
        XCTAssertNil(vm.selectedPivot)
    }

    /// The overlay geometry cache follows path mutations (apex = key position, and the ribbon
    /// point count is fixed). With no path it is nil — this nil is the Canvas display guard.
    func testPathOverlayGeometryFollowsPathEdits() {
        let vm = makeVM()
        let g = vm.pathOverlayGeometry
        XCTAssertEqual(g?.frusta.map(\.apex),
                       [SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(2, 0, 0)])
        XCTAssertEqual(g?.ribbonPoints.count, PathOverlayGeometry.sampleCount + 1)
        XCTAssertEqual(g?.totalLength, vm.sampler?.totalLength)
        vm.translateSelectedPath(by: SIMD3(0, 0, 4), persistNow: false)
        XCTAssertEqual(vm.pathOverlayGeometry?.frusta.map(\.apex),
                       [SIMD3(0, 0, 4), SIMD3(1, 0, 4), SIMD3(2, 0, 4)])
        vm.deletePath(id: vm.paths[0].id)
        XCTAssertNil(vm.pathOverlayGeometry)
    }

    /// Changing the aspect ratio changes the frustum geometry, so **the cache must be
    /// rebuilt** (the second refresh gate). If you omit it, a FRAMING change is not reflected
    /// in the overlay (a visual regression).
    func testPathOverlayGeometryRebuildsOnAspectChange() {
        let vm = makeVM()
        let wide = vm.pathOverlayGeometry
        XCTAssertEqual(vm.outputAspect, .wide)
        vm.setOutputAspect(.square)
        let square = vm.pathOverlayGeometry
        XCTAssertNotNil(square)
        XCTAssertNotEqual(wide?.frusta, square?.frusta)              // aspect ratio applied
        XCTAssertEqual(wide?.ribbonPoints, square?.ribbonPoints)     // the ribbon is unaffected
    }

    // MARK: CameraPose round-trip golden (orbit → world pose → view matrix reassembly)

    // For a non-trivial orbit (up axis +Z with arbitrary yaw/pitch/fov), the viewMatrix of
    // the effective pose must match the lookAt viewMatrix component by component — the
    // consistency contract of the skimmer render path.
    func testCameraPoseRoundTripMatchesLookAt() {
        var camera = CameraController()
        camera.target = SIMD3(1, 2, 3)
        camera.distance = 3
        camera.yaw = 0.7
        camera.pitch = 0.4
        camera.upAxis = SIMD3(0, 0, 1)
        camera.verticalFOV = 1.1

        let pose = CameraPose.effective(of: camera)
        XCTAssertEqual(pose.fov, 1.1, accuracy: 1e-6)
        XCTAssertLessThan(simd_distance(pose.position, camera.eye), 1e-5)
        let a = pose.viewMatrix, b = camera.viewMatrix
        for c in 0..<4 {
            for r in 0..<4 {
                XCTAssertEqual(a[c][r], b[c][r], accuracy: 1e-4, "col \(c) row \(r)")
            }
        }
        // When an override is placed, effective returns that pose as-is (WYSIWYG capture).
        let injected = CameraPose(position: SIMD3(9, 8, 7),
                                  rotation: simd_quatf(angle: 0.3, axis: SIMD3(0, 1, 0)),
                                  fov: 0.6)
        camera.overridePose = injected
        XCTAssertEqual(CameraPose.effective(of: camera), injected)
        XCTAssertEqual(camera.projectionMatrix(aspect: 1).columns.1.y,
                       1 / tan(0.6 * 0.5), accuracy: 1e-4)           // the projection uses the override FOV too
    }
}
