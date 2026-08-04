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
