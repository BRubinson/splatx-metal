// Panel redesign Task 1 (2026-07-29 spec §3): verifies only the **pure logic** of path name
// normalization. Not testing SwiftUI views themselves is the explicit policy of this
// repository, so only the parts extracted into static pure functions are
// pinned here.
import XCTest
import simd
@testable import SplatXMetalTrainer

final class CinemaInspectorAffordanceTests: XCTestCase {
    /// The single source of truth for the name-commit convention is the pure function
    /// `CinemaViewModel.normalizedPathName`.
    /// Every rule is an extraction of an existing convention, not a new one: trimming follows
    /// renamePath, the "Path" fallback for an empty name follows finalizePathNames, and the
    /// " 2" suffix uniquification follows the addPresetPath convention ("Orbit 360° 2")
    /// (spec §3).
    @MainActor
    func testNormalizedPathNameTrimsAndFallsBack() {
        XCTAssertEqual(CinemaViewModel.normalizedPathName("  Orbit  ", existing: []), "Orbit")
        XCTAssertEqual(CinemaViewModel.normalizedPathName("Solo", existing: ["Other"]), "Solo")
        // An empty string, or only whitespace and newlines → the "Path" fallback (the
        // current finalizePathNames convention)
        XCTAssertEqual(CinemaViewModel.normalizedPathName("", existing: []), "Path")
        XCTAssertEqual(CinemaViewModel.normalizedPathName("  \n ", existing: []), "Path")
        // If the fallback name itself is taken, the fallback gets a suffix too (the
        // uniqueness invariant holds)
        XCTAssertEqual(CinemaViewModel.normalizedPathName("   ", existing: ["Path"]), "Path 2")
    }

    /// Uniquification uses the same "name N" convention as addPresetPath — it starts at 2
    /// and increments until it finds a free slot.
    @MainActor
    func testNormalizedPathNameUniquifiesLikePresets() {
        XCTAssertEqual(CinemaViewModel.normalizedPathName("Orbit 360°",
                                                          existing: ["Orbit 360°"]),
                       "Orbit 360° 2")
        XCTAssertEqual(CinemaViewModel.normalizedPathName(
            "Orbit 360°", existing: ["Orbit 360°", "Orbit 360° 2"]), "Orbit 360° 3")
        // The collision decision is based on the trimmed name.
        XCTAssertEqual(CinemaViewModel.normalizedPathName("  A ", existing: ["A"]), "A 2")
    }

    /// FIX 1 ① (final review — spec footnote 1): the name draft seeding policy. There are two
    /// editing entries, so there are two drafts (the inline list row and the Name field of the
    /// PATH section); without cross-invalidation, when one entry's exit and the other's focus
    /// gain land on the **same event**, the entry that gains focus is left holding a stale
    /// name. After FIX 2 ⑤ (blur = cancel), the accident of a stale draft being committed on
    /// blur (FIX 1's reproduction: a rename reverts on blur and gets persisted) is closed off
    /// at the surface itself, but a stale seed still pollutes the display and Return would
    /// wrongly commit that value, so the policy is unchanged: focus gain = unconditionally
    /// reseed from the canonical value; observing a canonical change = follow it only while
    /// unfocused (input is preserved during editing).
    @MainActor
    func testNameDraftReseedsFromCanonicalOnFocusGain() {
        // Focus gain: takes the canonical value regardless of fieldFocused (the caller has
        // already explicitly discarded the other entry's uncommitted draft — FIX 2 ⑤, cancel
        // first — so canonical is exactly the valid name).
        XCTAssertEqual(CinemaInspector.nameDraft(after: .focusGained, draft: "Alpha",
                                                 canonical: "Bravo", fieldFocused: true),
                       "Bravo")
        XCTAssertEqual(CinemaInspector.nameDraft(after: .focusGained, draft: "Alpha",
                                                 canonical: "Bravo", fieldFocused: false),
                       "Bravo")
        // Observing a canonical change: while editing (focused) the user's input is
        // preserved; only while unfocused does the draft follow.
        XCTAssertEqual(CinemaInspector.nameDraft(after: .canonicalChanged, draft: "typing",
                                                 canonical: "Bravo", fieldFocused: true),
                       "typing")
        XCTAssertEqual(CinemaInspector.nameDraft(after: .canonicalChanged, draft: "stale",
                                                 canonical: "Bravo", fieldFocused: false),
                       "Bravo")
    }

    /// FIX 1 ④ (role reduced by FIX 2 ⑤): the cancel decision of the commit gate — cancelling
    /// (Esc, blur, or cross-invalidation) clears the target, and a cleared target (nil) means
    /// no commit. Under the blur = cancel policy the only caller of commit is the Return path,
    /// but this gate is still the defense that turns a late commit event arriving after the
    /// target was cleared into a no-op. Normalizing the draft belongs outside the gate
    /// (commitPathRename), so the draft passes through unchanged. (The full cancel scenario
    /// that reaches the VM is exercised by testRenameCancelContractKeepsOriginal in
    /// CinemaViewModelTests.)
    @MainActor
    func testRenameCommitGateDropsClearedTarget() {
        XCTAssertNil(CinemaInspector.renameCommit(target: nil, draft: "anything"))
        let id = UUID()
        let commit = CinemaInspector.renameCommit(target: id, draft: "  Alpha ")
        XCTAssertEqual(commit?.id, id)
        XCTAssertEqual(commit?.draft, "  Alpha ")
    }

    /// FIX 2 ⑤ (spec footnote 2): the exit policy for name editing — **only Return** commits;
    /// Esc and blur (clicking elsewhere) cancel. ⚠ blur = cancel departs from the macOS
    /// convention (Finder commits on blur) and is an **explicit user request** — to flip this
    /// test "to match the convention", get the user's agreement again first.
    /// Inline list editing (endRename) and the Name field of the PATH section
    /// (endPathNameField) consume the same single source of truth.
    @MainActor
    func testRenameExitCommitsOnlyOnReturn() {
        XCTAssertTrue(CinemaInspector.renameExitCommits(.returnKey))
        XCTAssertFalse(CinemaInspector.renameExitCommits(.escape))
        XCTAssertFalse(CinemaInspector.renameExitCommits(.blur),
                       "blur = cancel is an explicit user request (FIX 2 ⑤ — spec footnote 2)")
    }
}

/// Panel redesign Task 2 (spec §4): pins the two decisions of the tidy-up rules that can be
/// extracted purely — deriving the section visibility conditions (the "hide" side of the
/// hide/disable policy) and choosing the disabled-reason string (the "disable" side).
/// The body consumes the results of these functions as-is (the view layout itself is checked
/// manually, per the app policy).
final class CinemaPanelRuleTests: XCTestCase {
    // MARK: Section visibility conditions (spec §4 rule 6 + §5 — Task 3 restructured the
    // sections from 8 to 5)

    /// With no selected path, every conditional element (the PATH, ORBIT, and KEYFRAME rows)
    /// drops out.
    /// FRAMING was absorbed into EXPORT in Task 3 and is no longer an independent section
    /// (spec §5 table).
    /// PRESETS was absorbed into the New Path ▾ pulldown of the PATHS section in Task 4 and is
    /// no longer an independent section (spec §6 — the final section layout is the five
    /// sections in the §5 table).
    @MainActor
    func testVisibleSectionsWithoutSelectedPath() {
        XCTAssertEqual(
            CinemaInspector.visibleSections(hasSelectedPath: false, hasPivot: false,
                                            hasKeyframeSelection: false),
            [.paths, .overlay, .export])
        // A pivot or a key selection cannot exist without a path — the pure function
        // enforces the same guard
        XCTAssertEqual(
            CinemaInspector.visibleSections(hasSelectedPath: false, hasPivot: true,
                                            hasKeyframeSelection: true),
            [.paths, .overlay, .export])
    }

    /// A selected path → PATH. A pivot (keys ≥ 1 — the same condition as the viewport handle)
    /// → ORBIT (formerly PATH TRANSFORM — renamed in spec §5). A valid key selection → the
    /// lightweight KEYFRAME row. The order is exactly the on-screen layout order.
    @MainActor
    func testVisibleSectionsProgressiveDisclosure() {
        XCTAssertEqual(
            CinemaInspector.visibleSections(hasSelectedPath: true, hasPivot: false,
                                            hasKeyframeSelection: false),
            [.paths, .path, .overlay, .export])
        XCTAssertEqual(
            CinemaInspector.visibleSections(hasSelectedPath: true, hasPivot: true,
                                            hasKeyframeSelection: false),
            [.paths, .path, .orbit, .overlay, .export])
        XCTAssertEqual(
            CinemaInspector.visibleSections(hasSelectedPath: true, hasPivot: true,
                                            hasKeyframeSelection: true),
            [.paths, .path, .orbit, .keyframe, .overlay, .export])
    }

    /// The one-line summary of the lightweight KEYFRAME row (Task 3 — spec §5: a read-only
    /// three-row section → one line + delete).
    /// The information kept is the index (which identifies the selection) and the time (the
    /// authoritative value under manual pacing) — fov is excluded (the rationale is in
    /// task-3-report). If the time cannot be computed (no sampler), only the index is shown.
    @MainActor
    func testKeyframeSummaryLine() {
        XCTAssertEqual(CinemaInspector.keyframeSummary(index: 2, count: 8, time: 1.25),
                       "3/8 · 1.25 s")
        XCTAssertEqual(CinemaInspector.keyframeSummary(index: 0, count: 1, time: nil),
                       "1/1")
        XCTAssertFalse(CinemaInspector.keyframeSummary(index: 0, count: 3, time: 0)
            .contains("\n"), "the summary must be a single line (spec §5 lightweight row)")
    }

    // MARK: Choosing the disabled reason (spec §4 rule 5 — .disabled + row dimming + a
    // one-line reason)

    /// Radius is disabled only when R̄ == 0 (a static shot in which every key sits on the
    /// pivot) — the factor s = R/R̄ is undefined (in step with hiding the radius handle). When
    /// enabled there is no reason (nil).
    @MainActor
    func testRadiusDisabledReasonOnlyForZeroMeanRadius() {
        XCTAssertNil(CinemaInspector.radiusDisabledReason(meanRadius: 1.25))
        let reason = CinemaInspector.radiusDisabledReason(meanRadius: 0)
        XCTAssertNotNil(reason)
        XCTAssertFalse(reason!.contains("\n"), "the reason must be a single line (spec §4)")
    }

    /// Ease is disabled only under manual pacing (the contract of respecting keyframe times —
    /// the PacingMode comment is the authoritative contract).
    @MainActor
    func testEaseDisabledReasonOnlyInManualPacing() {
        XCTAssertNil(CinemaInspector.easeDisabledReason(pacing: .even))
        XCTAssertEqual(
            CinemaInspector.easeDisabledReason(pacing: .manual),
            "Ease is bypassed in manual pacing — keyframe times are authoritative.")
    }

    /// The Pacing status row is shown only in manual (FIX 2 ① — spec footnote 2: the default,
    /// even, is not worth reporting, so the row itself is hidden. Consistent with the spec §5
    /// table entry "(Pacing + Even when manual)").
    @MainActor
    func testPacingRowVisibleOnlyInManual() {
        XCTAssertFalse(CinemaInspector.pacingRowVisible(pacing: .even))
        XCTAssertTrue(CinemaInspector.pacingRowVisible(pacing: .manual))
    }

    /// The Capture Path reason — the string is the previous caption verbatim (wording already
    /// exposed to users is kept).
    @MainActor
    func testCapturePathDisabledReason() {
        XCTAssertNil(CinemaInspector.capturePathDisabledReason(hasColmapSparse: true))
        XCTAssertEqual(
            CinemaInspector.capturePathDisabledReason(hasColmapSparse: false),
            "Capture Path requires COLMAP-aligned views (sparse/0).")
    }

    /// OVERLAY Add Image — previously there was only .disabled and the reason stayed silent
    /// (the asymmetry in spec §1 ④).
    @MainActor
    func testOverlayAddImageDisabledReason() {
        XCTAssertNil(OverlayInspectorSection.addImageDisabledReason(hasProjectDirectory: true))
        XCTAssertNotNil(
            OverlayInspectorSection.addImageDisabledReason(hasProjectDirectory: false))
    }

    /// OVERLAY "Add SplatX watermark" (app finishing, 2026-07-29) — the auxiliary action that
    /// closes the defect where deleting the default watermark left no way at all to bring it
    /// back from the UI. The project gate reuses the same gate as Add Image, and only the
    /// missing bundle image is added on top — disabled up front instead of a failure you learn
    /// about only by clicking (§4 rule 5). This function is **only for the disable/dim
    /// decision** — the caption contract is
    /// testOverlayAddWatermarkCaptionOmitsDuplicatedProjectReason below (FIX 1 ⑤).
    @MainActor
    func testOverlayAddWatermarkDisabledReason() {
        XCTAssertNil(OverlayInspectorSection.addWatermarkDisabledReason(
            hasProjectDirectory: true, hasBundleImage: true))
        // No project directory disables it regardless of whether the bundle image exists.
        XCTAssertNotNil(OverlayInspectorSection.addWatermarkDisabledReason(
            hasProjectDirectory: false, hasBundleImage: true))
        XCTAssertNotNil(OverlayInspectorSection.addWatermarkDisabledReason(
            hasProjectDirectory: false, hasBundleImage: false))
        // A missing bundle image is a separate reason.
        let missing = OverlayInspectorSection.addWatermarkDisabledReason(
            hasProjectDirectory: true, hasBundleImage: false)
        XCTAssertNotNil(missing)
        XCTAssertNotEqual(missing,
                          OverlayInspectorSection.addImageDisabledReason(
                            hasProjectDirectory: false))
    }

    /// App finishing FIX 1 ⑤ — the single source of truth for the rendered caption is
    /// `addWatermarkCaption`: the Add Image row directly above already prints the
    /// no-project-directory reason as a caption, so the watermark row **omits it (dimming
    /// only)** — the same sentence is never rendered twice in a row (the original intent of
    /// "do not say the same reason in two sentences"). ⚠ The previous test asserted the
    /// **equality** of the two functions' strings and thereby froze that duplicate rendering
    /// into a contract — this test is its replacement. Do not revert it.
    @MainActor
    func testOverlayAddWatermarkCaptionOmitsDuplicatedProjectReason() {
        // No project directory: the row above is already showing the same reason — no
        // caption (regardless of the bundle image).
        XCTAssertNil(OverlayInspectorSection.addWatermarkCaption(
            hasProjectDirectory: false, hasBundleImage: true))
        XCTAssertNil(OverlayInspectorSection.addWatermarkCaption(
            hasProjectDirectory: false, hasBundleImage: false))
        // A missing bundle image is this row's own reason, so it is printed as-is (the same
        // sentence as the disabled reason).
        XCTAssertEqual(
            OverlayInspectorSection.addWatermarkCaption(
                hasProjectDirectory: true, hasBundleImage: false),
            OverlayInspectorSection.addWatermarkDisabledReason(
                hasProjectDirectory: true, hasBundleImage: false))
        // When enabled there is no caption either.
        XCTAssertNil(OverlayInspectorSection.addWatermarkCaption(
            hasProjectDirectory: true, hasBundleImage: true))
    }

    /// Priority of the Render CTA reasons: no run → no keyframes → tab locked. The strings
    /// are the previous disabledReason verbatim (session.lockedTab is authoritative for the
    /// lock decision — the FIX 3 convention).
    @MainActor
    func testRenderDisabledReasonPriority() {
        XCTAssertEqual(
            ExportInspectorSection.renderDisabledReason(hasRun: false, hasKeyframes: false,
                                                        isLocked: true),
            "Open a run first — rendering uses its PLY.")
        XCTAssertEqual(
            ExportInspectorSection.renderDisabledReason(hasRun: true, hasKeyframes: false,
                                                        isLocked: true),
            "Add at least one keyframe to the selected path.")
        XCTAssertEqual(
            ExportInspectorSection.renderDisabledReason(hasRun: true, hasKeyframes: true,
                                                        isLocked: true),
            "Unavailable while training or aligning.")
        XCTAssertNil(
            ExportInspectorSection.renderDisabledReason(hasRun: true, hasKeyframes: true,
                                                        isLocked: false))
    }

    // MARK: The EXPORT section's format-dependent presentation (2026-08-05 spec §5.4/§5.7–§5.9)

    /// §5.9 — the codec row is disabled under GIF and only under GIF. The decision is derived
    /// from `videoContainer == nil`, the same discriminator the sink branch uses, so a format
    /// cannot end up encoding through ImageIO while still offering HEVC/H.264.
    /// Note what is **not** asserted: that the row is hidden. It must stay laid out and dim —
    /// a row that disappears makes the section's height jump on every File change.
    @MainActor
    func testCodecRowDisabledOnlyForGif() {
        XCTAssertTrue(ExportInspectorSection.codecRowDisabled(format: .gif))
        XCTAssertFalse(ExportInspectorSection.codecRowDisabled(format: .mp4))
        XCTAssertFalse(ExportInspectorSection.codecRowDisabled(format: .mov))
    }

    /// §5.3/§5.4 — the Size row follows the format, and the labels shorten once the row splits
    /// four ways (~49pt per cell). The values must stay the real presets: only the displayed
    /// text is abbreviated.
    @MainActor
    func testSizeOptionsFollowFormatAndShortenAtFourCells() {
        let video = ExportInspectorSection.sizeOptions(for: .mp4)
        XCTAssertEqual(video.map(\.value), ResolutionPreset.videoChoices)
        // Three cells: the full preset names, unchanged from before GIF existed.
        XCTAssertEqual(video.map(\.label), ["1080p", "1440p", "4K"])
        XCTAssertEqual(ExportInspectorSection.sizeOptions(for: .mov).map(\.value),
                       ResolutionPreset.videoChoices)

        let gif = ExportInspectorSection.sizeOptions(for: .gif)
        XCTAssertEqual(gif.map(\.value), ResolutionPreset.gifChoices)
        // Four cells: suffix-less numbers (the first four-cell SegmentedRow in this app).
        XCTAssertEqual(gif.map(\.label), ["240", "360", "480", "720"])
        // The abbreviation is keyed on the cell count, not on the format — every label is
        // exactly the preset's short side, so the mapping is never a hand-kept table.
        for option in gif {
            XCTAssertEqual(option.label, "\(option.value.shortSide)")
        }
    }

    /// §5.8 — the size caption appears only under GIF, and it states the two facts the user
    /// needs before pressing Render: how many frames are kept (all of them — decision 4) and
    /// how large the sequence is. It is deliberately not a proposal to reduce anything.
    @MainActor
    func testGifSizeNoteReportsFramesAndPixels() {
        let note = ExportInspectorSection.gifSizeNote(format: .gif, frames: 301,
                                                      width: 854, height: 480)
        XCTAssertNotNil(note)
        XCTAssertTrue(note!.contains("301"), note ?? "")
        XCTAssertTrue(note!.contains("854×480"), note ?? "")

        // It tracks its inputs — a different path length and a different preset must produce a
        // different sentence (a constant string would pass the contains() checks above).
        let other = ExportInspectorSection.gifSizeNote(format: .gif, frames: 120,
                                                       width: 1280, height: 720)
        XCTAssertNotNil(other)
        XCTAssertNotEqual(note, other)
        XCTAssertTrue(other!.contains("120"), other ?? "")
        XCTAssertTrue(other!.contains("1280×720"), other ?? "")

        // Video formats never show it, however large the sequence.
        XCTAssertNil(ExportInspectorSection.gifSizeNote(format: .mp4, frames: 301,
                                                        width: 3840, height: 2160))
        XCTAssertNil(ExportInspectorSection.gifSizeNote(format: .mov, frames: 301,
                                                        width: 3840, height: 2160))
        // Degenerate inputs (no path selected → 0 frames) produce no caption rather than
        // "0 frames … 0 MB".
        XCTAssertNil(ExportInspectorSection.gifSizeNote(format: .gif, frames: 0,
                                                        width: 854, height: 480))
        XCTAssertNil(ExportInspectorSection.gifSizeNote(format: .gif, frames: 10,
                                                        width: 0, height: 0))
    }

    /// §5.7 — the fps caption fires only where the 1/100s quantization actually bites. 60fps
    /// is 0.0167s, which many viewers clamp to 0.1s (≈10fps); 24 (0.0417s) and 30 (0.0333s)
    /// are safe. The training timelapse never hit this because its fps is capped at 30.
    @MainActor
    func testGifFpsNoteWarnsOnlyAtSixtyUnderGif() {
        XCTAssertNotNil(ExportInspectorSection.gifFpsNote(format: .gif, fps: 60))
        XCTAssertNil(ExportInspectorSection.gifFpsNote(format: .gif, fps: 24))
        XCTAssertNil(ExportInspectorSection.gifFpsNote(format: .gif, fps: 30))
        // Video formats have no such quantization at any rate.
        for fps in VideoExportSettings.fpsChoices {
            XCTAssertNil(ExportInspectorSection.gifFpsNote(format: .mp4, fps: fps))
            XCTAssertNil(ExportInspectorSection.gifFpsNote(format: .mov, fps: fps))
        }
        // The threshold is the delay, not the literal 60: exactly 0.02s (50fps) is still safe,
        // anything faster is not. Pinning this keeps a future 48/50 choice classified by
        // arithmetic instead of by memory.
        XCTAssertNil(ExportInspectorSection.gifFpsNote(format: .gif, fps: 50))
        XCTAssertNotNil(ExportInspectorSection.gifFpsNote(format: .gif, fps: 51))
        // Guard against a division by zero in a degenerate call.
        XCTAssertNil(ExportInspectorSection.gifFpsNote(format: .gif, fps: 0))
        // The warning names the offending rate so the caption is actionable.
        XCTAssertTrue(ExportInspectorSection.gifFpsNote(format: .gif, fps: 60)!.contains("60"))
    }

    /// The 2026-08-05 addition of 15 to `fpsChoices` (spec §3.3) is the first exercise of the
    /// claim the comment above makes — that the *classifier* is arithmetic on the delay, so a new
    /// rate needs no edit to the gate. It did need one to the **prose**: `timelapse-viewport-aspect`
    /// FIX 1 found the caption still naming only "24 or 30", so a user who had just been given 15
    /// was steered away from it. The second half of this test is what turns "someone will
    /// remember" into a failure.
    ///
    /// Written as a sweep over `fpsChoices` rather than `gifFpsNote(15) == nil` alone, because the
    /// interesting property is the partition of the whole list: exactly one offered rate warns, and
    /// the caption recommends **every** rate on the other side of the partition. A future rate
    /// below 50 fails the second assertion until it is named; one above 50 fails the first and
    /// forces the author to decide what the advice should now read.
    @MainActor
    func testAddingFifteenNeedsNoChangeToTheFpsCaption() {
        XCTAssertNil(ExportInspectorSection.gifFpsNote(format: .gif, fps: 15),
                     "1/15 = 0.0667s is far above the 0.02s clamp — no caption is correct")
        let warned = VideoExportSettings.fpsChoices.filter {
            ExportInspectorSection.gifFpsNote(format: .gif, fps: $0) != nil
        }
        XCTAssertEqual(warned, [60], "60 is the only offered rate under the 0.02s clamp")
        let note = ExportInspectorSection.gifFpsNote(format: .gif, fps: 60)
        for safe in VideoExportSettings.fpsChoices where !warned.contains(safe) {
            XCTAssertTrue(note?.contains("\(safe)") == true,
                          "the caption tells the user what to switch to, so it must name \(safe)")
        }
    }

    /// FIX 1 — the two GIF-only captions gate on **different** discriminators on purpose, and this
    /// pins which is which so neither drifts into the other by tidying.
    ///
    /// `gifSizeNote` (and `codecRowDisabled`, and `consoleFormatSummary`) claim things about the
    /// **encoder arm** — no codec axis, ImageIO buffers frames — so they gate on
    /// `videoContainer == nil` and a future non-AVAssetWriter format inherits them correctly.
    /// `gifFpsNote` claims something about the **GIF container itself** (delays quantised to
    /// 1/100s); APNG and WebP time frames differently, so it gates on `== .gif` and a future
    /// ImageIO format must *not* inherit it. Both loops run over allCases, so adding a case forces
    /// the decision rather than letting it default.
    @MainActor
    func testGifCaptionDiscriminatorsAreDeliberate() {
        for format in CinemaExportFormat.allCases {
            let sizeNote = ExportInspectorSection.gifSizeNote(format: format, frames: 301,
                                                              width: 854, height: 480)
            XCTAssertEqual(sizeNote != nil, format.videoContainer == nil,
                           "\(format): the size caption follows the encoder arm")
            XCTAssertEqual(ExportInspectorSection.codecRowDisabled(format: format),
                           format.videoContainer == nil,
                           "\(format): the codec row follows the encoder arm")

            let fpsNote = ExportInspectorSection.gifFpsNote(format: format, fps: 60)
            XCTAssertEqual(fpsNote != nil, format == .gif,
                           "\(format): the fps caption is a GIF container fact, not an arm fact")
        }
    }

    // MARK: Prose moved into the ⓘ popover by the caption diet (preserving the FIX 19 ③
    // correction)

    /// The scope correction to the undo promise (FIX 19 ③) must not resurface during the move
    /// — a horizontal drag has no corresponding field, so only the radius and height axes are
    /// promised.
    @MainActor
    func testTransformInfoKeepsCorrectedUndoPromise() {
        XCTAssertTrue(CinemaInspector.transformInfo
            .contains("radius and height changes only"))
        XCTAssertTrue(CinemaInspector.transformInfo
            .contains("horizontal moves have no field"))
        // The handle-manipulation wording (footnote 18 is authoritative) moves with the
        // section — it must not be lost
        XCTAssertTrue(CinemaInspector.transformInfo.contains("⇧"))
    }
}
