// Cinema Task 3 (2026-07-26 spec UX 5): the Cinema mode inspector — the path list, the
// selected path's settings (Duration in seconds with the derived frame count alongside,
// Smoothness, Ease, Loop, and the pacing state), and the selected keyframe. InspectorView
// switches to this view when Cinema is active in the View tab. Text editing focus is mirrored
// into vm.isEditingText so the Space/Enter/Delete/←→ commands (AppCommands) do not intercept
// typing.
//
// Path list (panel redesign Task 1 — spec §3, revised by FIX 2 ⑤ — spec footnote 2): the whole
// row is a selection Button (the RUNS row pattern) and the name is normally a Text. There are
// three entry points into editing — ① double-clicking the name (regardless of selection —
// replacing the previous "click the selected row again"), ② the context menu RenameButton, and
// ③ the hover ⋯ menu. Only the row being edited is a TextField (editingPathID); the commit is
// **Return only**, and Esc and blur (clicking elsewhere) are a cancel (renameExitCommits is the
// authoritative policy — an explicit user request. Note that this differs from the macOS
// convention, where blur = commit). The previous structure (a permanent TextField plus an
// ancestor onTapGesture for selection) let the AppKit field editor consume the mouse-down, so
// it became "left click = edit the name, far right = select", and the name of an unselected row
// went straight into editing too (spec §1).
//
// Tidiness rules (panel redesign Task 2 — spec §4; the Theme.swift comment is the authoritative
// decision record): labels use FieldLabel (a 52pt column), units use UnitLabel (fixed width —
// never reuse the placeholder), an attached caption is bound to its control with
// VStack(spacing: 2), and disabling is unified as ".disabled + row dimming + a one-line
// reason". Section visibility conditions live in visibleSections, and the reason strings are
// authoritative in the *DisabledReason pure functions (tests:
// CinemaInspectorAffordanceTests).
//
// Section restructuring (panel redesign Task 3 — spec §5): 8 sections → 5. FRAMING was absorbed
// into EXPORT (the Aspect picker is RenderJob.swift's ExportInspectorSection — the
// setOutputAspect binding is unchanged, only the placement moved), PATH TRANSFORM was renamed
// to ORBIT, KEYFRAME is no longer a section but a lightweight row of a one-line summary plus
// delete (keyframeRow), and the PATH section gained a parallel Name field entry point (spec §3).
//
// Preset pulldown (panel redesign Task 4 — spec §6): the PRESETS section was absorbed into the
// New Path ▾ pulldown of the PATHS section. Rationale: addPath() and addPresetPath() behave the
// same (both create a path and select it), and an action used once per path has no reason to
// permanently occupy the top of the inspector — the HIG also classifies this case as a pulldown
// button rather than a popup. The disabled reason for Capture Path accompanies it as a menu
// item subtitle (a Text second label) (capturePathDisabledReason remains authoritative).
//
// Cross-invalidation of name drafts (panel redesign FIX 1 ① — spec footnote 1. FIX 2 ⑤
// redefined blur as cancel — spec footnote 2): there are two editing entry points, so there are
// two drafts (the list inline editDraft and the PATH section pathNameDraft). Under the
// blur = cancel policy, one side gaining focus or edit rights is, from the other entry point's
// perspective, "a click elsewhere", so the side that gains it ⑴ **discards** the other entry
// point's uncommitted draft (clearing the gate target first makes it idempotent — no race with
// onChange ordering. The previous FIX 1 committed first because blur = commit) and ⑵ **reseeds
// its own draft from the source of truth**. Because blur no longer commits, the accident
// surface of FIX 1 — "a stale draft is committed on blur and undoes the rename you just made" —
// is itself closed, but a stale seed still pollutes the display and Return would wrongly commit
// that value, so the reseed invariant stands. The seeding policy is authoritative in
// nameDraft(after:), the commit/cancel policy in renameExitCommits, and the cancel decision of
// the commit gate in the pure function renameCommit(target:)
// (CinemaInspectorAffordanceTests and CinemaViewModelTests — this app does not test SwiftUI, so
// those are the only automated line of defense).
import SwiftUI
import simd

struct CinemaInspector: View {
    @Bindable var vm: CinemaViewModel
    /// The basis for starting the export section's job (Task 5) and for its state —
    /// RenderJob(session) and job/console(model). InspectorView is the only caller.
    @Bindable var model: AppModel
    @Bindable var session: ProjectSession
    /// COLMAP context for the capture-trajectory preset (lazily loaded — ProjectSession caches
    /// the parse). nil for a non-COLMAP project or a parse failure. hasColmapSparse decides
    /// whether the button is enabled (the context is parsed only on click — so that points3D
    /// is not read on every body evaluation).
    var loadColmap: () -> ColmapContext? = { nil }
    /// Whether sparse/0 exists (a cheap stat — InspectorView computes it and passes it in).
    var hasColmapSparse = false
    /// Inline name editing focus (the path id of the row being edited) — focus-out cancels
    /// (FIX 2 ⑤).
    @FocusState private var nameFocus: UUID?
    /// The id of the row being edited plus its draft (panel redesign Task 1). **These are view
    /// @State — do not put them in the VM**: this list is the only consumer, and keeping the
    /// draft out of the VM means an Esc cancel preserves the original by construction, with no
    /// active restore (the only VM entry is commitPathRename). Moving it into the VM would add
    /// one more derived-cache gate discipline, like selectedPathID.
    @State private var editingPathID: UUID?
    @State private var editDraft = ""
    /// The PATH section's Name field (panel redesign Task 3 — spec §3, "let it be fixed from
    /// the inspector too"). The draft is view @State — the only VM entry is the same
    /// commitPathRename gate as the list editing.
    /// The target id is pinned **at the moment focus is gained**: even if the selection changes
    /// in the list during editing and focus-out arrives afterwards, the commit does not leak
    /// onto the new path (reading selectedPathID at commit time would create a race between the
    /// selection change and the focus loss).
    @State private var pathNameDraft = ""
    @State private var pathNameTargetID: UUID?
    @FocusState private var pathNameFocused: Bool
    @FocusState private var durationFocused: Bool
    /// Focus of the PATH TRANSFORM section's numeric fields (path transform Task 2) — they join
    /// the isEditingText gate.
    @FocusState private var radiusFocused: Bool
    @FocusState private var heightFocused: Bool

    var body: some View {
        // The pure function visibleSections is authoritative for section visibility — the body
        // only consumes its result (the "hide" side of the hide/disable policy, spec §4
        // rule 6).
        let sections = Self.visibleSections(
            hasSelectedPath: vm.selectedPath != nil,
            hasPivot: vm.selectedPivot != nil,
            hasKeyframeSelection: vm.selectedKeyframeIndex.map {
                vm.selectedPath?.keyframes.indices.contains($0) ?? false
            } ?? false)
        ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                PanelSection(title: "PATHS") {
                    pathList
                    newPathMenu
                }
                if sections.contains(.path), let path = vm.selectedPath {
                    pathSection(path)
                    // Path transform Task 2: hide the section when there is no selected path or
                    // no pivot (zero keys).
                    if sections.contains(.orbit), let pivot = vm.selectedPivot {
                        orbitSection(pivot)
                    }
                    if sections.contains(.keyframe), let i = vm.selectedKeyframeIndex,
                       path.keyframes.indices.contains(i) {
                        keyframeRow(path, index: i)
                    }
                }
                OverlayInspectorSection(vm: vm)
                ExportInspectorSection(vm: vm, model: model, session: session)
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .onChange(of: nameFocus) { old, focus in
            // Handle the exit **only when the edited row had focus and then lost it** —
            // blur = cancel (FIX 2 ⑤; renameExitCommits is the authoritative decision).
            // Without the old condition, switching editing to another row (Rename on A while
            // editing B) would let the transient nil created by B's field disappearing
            // immediately close and kill A's new session. Esc (endRename(.escape)) and Return
            // (endRename(.returnKey) from onSubmit) set editingPathID to nil first and do not
            // pass through here — no double handling.
            if old == editingPathID, focus != editingPathID { endRename(after: .blur) }
            syncEditingFlag()
        }
        .onChange(of: pathNameFocused) { old, focused in
            // Gaining focus = pinning the commit target; losing it = cancel (FIX 2 ⑤ —
            // blur = cancel is an explicit user request. The only commit is Return via
            // onSubmit). Esc (onExitCommand) clears the target first and then blurs, so passing
            // through here does nothing.
            if focused {
                // Cross-invalidation (FIX 1 ①, redefined by FIX 2 ⑤ to cancel first): if list
                // inline editing is in progress, this click is "a click elsewhere" from the
                // inline edit's perspective — rather than relying on onChange ordering, the
                // draft is discarded here first (idempotent, since the cancel clears
                // editingPathID). Then the draft is **reseeded from the source of truth**:
                // without the reseed, the onChange(of: path.name) below is blocked by the
                // pathNameFocused == true guard, this field is left holding the stale name, and
                // Return would wrongly commit that value (the reseed invariant of FIX 1 ① —
                // only the blur mis-commit surface was closed).
                endRename(after: .blur)
                pathNameTargetID = vm.selectedPathID
                pathNameDraft = Self.nameDraft(
                    after: .focusGained, draft: pathNameDraft,
                    canonical: vm.selectedPath?.name ?? pathNameDraft, fieldFocused: true)
            } else if old {
                endPathNameField(after: .blur)
            }
            syncEditingFlag()
        }
        .onChange(of: durationFocused) { _, _ in syncEditingFlag() }
        .onChange(of: radiusFocused) { _, _ in syncEditingFlag() }
        .onChange(of: heightFocused) { _, _ in syncEditingFlag() }
    }

    private func syncEditingFlag() {
        vm.isEditingText = nameFocus != nil || pathNameFocused || durationFocused
            || radiusFocused || heightFocused
    }

    /// Commits the PATH section's Name field (the Return path only — FIX 2 ⑤: blur and Esc go
    /// to endPathNameField's cancel branch) — no-op if the target id has been cleared (a cancel
    /// ran first). The cancel decision belongs to the pure gate renameCommit (FIX 1 ④), while
    /// normalization, saving, and ignoring a no-change commit belong to commitPathRename (the
    /// single gate shared with list editing).
    private func commitPathNameField() {
        guard let commit = Self.renameCommit(target: pathNameTargetID,
                                             draft: pathNameDraft) else { return }
        pathNameTargetID = nil
        vm.commitPathRename(id: commit.id, draft: commit.draft)
    }

    /// The shared exit gate for the PATH section's Name field — the pure source of truth
    /// renameExitCommits decides commit versus cancel (FIX 2 ⑤). A cancel clears the commit
    /// gate's target and restores the draft from the source of truth (discarding the input —
    /// since the VM was never touched, the only thing to actively restore is the display).
    private func endPathNameField(after exit: RenameExit) {
        if Self.renameExitCommits(exit) {
            commitPathNameField()
            // Reflect the normalization result (trim, unique) in the draft — so the display
            // matches the source of truth
            pathNameDraft = vm.selectedPath?.name ?? pathNameDraft
        } else {
            pathNameTargetID = nil
            pathNameDraft = vm.selectedPath?.name ?? pathNameDraft
        }
    }

    // MARK: Pure decisions (panel redesign Task 2 — the source of truth for tests:
    // CinemaInspectorAffordanceTests)

    /// Name draft events — the input to the seeding policy nameDraft(after:) (FIX 1 ①).
    enum NameDraftEvent {
        /// The moment an editing entry point gains focus or edit rights (the Name field gaining
        /// focus, or entering beginRename)
        case focusGained
        /// Observing a change of the authoritative name (onChange(of: path.name) — including
        /// the initial seed)
        case canonicalChanged
    }

    /// The single source of truth for the name draft seeding policy (FIX 1 ① — tests:
    /// CinemaInspectorAffordanceTests). Without cross-invalidation, when one entry point's exit
    /// and the other's focus gain coincide in the same event, the side that gains focus is left
    /// holding the stale name. After blur = cancel (FIX 2 ⑤) the accident of a stale draft being
    /// committed on blur (the reproduction in final review ①) became impossible, but a stale
    /// seed still pollutes the display and Return would wrongly commit that value, so the reseed
    /// invariant is unchanged.
    /// - .focusGained: **always** reseed from the source of truth — the caller has already
    ///   explicitly discarded the other entry point's uncommitted draft (cancel first — the
    ///   previous FIX 1 committed first), so canonical is exactly the valid name. Independent of
    ///   fieldFocused.
    /// - .canonicalChanged: follows only when unfocused — while editing, the draft preserves the
    ///   user's input.
    static func nameDraft(after event: NameDraftEvent, draft: String, canonical: String,
                          fieldFocused: Bool) -> String {
        switch event {
        case .focusGained: return canonical
        case .canonicalChanged: return fieldFocused ? draft : canonical
        }
    }

    /// Name editing exit events — the input to the commit/cancel policy renameExitCommits
    /// (FIX 2 ⑤).
    enum RenameExit {
        case returnKey, escape, blur
    }

    /// The single source of truth for the name editing exit policy (FIX 2 ⑤ — spec footnote 2,
    /// tests: CinemaInspectorAffordanceTests): the commit is **Return only**, and Esc and blur
    /// (clicking elsewhere) are a cancel. ⚠ blur = cancel differs from the macOS convention
    /// (Finder commits on blur) and is an **explicit user request** ("clicking elsewhere must
    /// cancel edit mode") — do not revert it to the convention.
    /// The list inline editing and the PATH section's Name field consume the same source of
    /// truth (endRename and endPathNameField).
    static func renameExitCommits(_ exit: RenameExit) -> Bool {
        exit == .returnKey
    }

    /// The authoritative cancel decision of the name commit gate (FIX 1 ④ — narrowed in
    /// FIX 2 ⑤): if the target has been cleared (nil), there is no commit. Under the
    /// blur = cancel policy the only commit call site is the Return path, but this gate is
    /// still the line of defense that makes a following late commit event a no-op by having the
    /// cancel (Esc, blur, or cross-invalidation) **clear the target first** (tests:
    /// CinemaViewModelTests and AffordanceTests). Draft normalization belongs outside the gate
    /// (commitPathRename), so the draft passes through here unchanged.
    static func renameCommit(target: UUID?, draft: String) -> (id: UUID, draft: String)? {
        target.map { (id: $0, draft: draft) }
    }

    /// The single source of truth for the Pacing status row's visibility (FIX 2 ① — spec
    /// footnote 2, tests: CinemaInspectorAffordanceTests): the default (even) is not worth
    /// announcing, so the row itself is hidden — that is what "(Pacing + Even when manual)" in
    /// the spec §5 table meant all along. manual is a non-default state the user entered by
    /// dragging in the timeline, so it shows the status plus the Even pacing revert button.
    static func pacingRowVisible(pacing: PacingMode) -> Bool {
        pacing == .manual
    }

    /// Inspector section identifiers — the return order of visibleSections is the on-screen
    /// layout order.
    /// Task 3 (spec §5, 8 sections → 5): framing was absorbed into EXPORT and removed, and
    /// transform was renamed to orbit. keyframe is now a lightweight row (keyframeRow) rather
    /// than a whole section, but its visibility condition and layout order are still
    /// authoritative here. Task 4 (spec §6): presets was absorbed into the PATHS section's
    /// New Path ▾ pulldown and removed — the 5-section composition of the spec §5 table is the
    /// final form.
    enum Section: Equatable {
        case paths, path, orbit, keyframe, overlay, export
    }

    /// The single source of truth for section visibility (spec §4 rule 6 — hide at section
    /// granularity, disable individual controls inside a section). A pivot exists only with
    /// ≥ 1 key, so showing orbit has the same condition as showing the viewport handles
    /// (FIX 19 ③ — "if the handles are visible, the ⓘ manipulation notes are reachable" holds
    /// for free).
    static func visibleSections(hasSelectedPath: Bool, hasPivot: Bool,
                                hasKeyframeSelection: Bool) -> [Section] {
        var sections: [Section] = [.paths]
        if hasSelectedPath {
            sections.append(.path)
            if hasPivot { sections.append(.orbit) }
            if hasKeyframeSelection { sections.append(.keyframe) }
        }
        return sections + [.overlay, .export]
    }

    /// The one-line summary of the lightweight KEYFRAME row (Task 3 — spec §5: a read-only
    /// 3-row section → one line plus delete). What remains is the index (identifying which key
    /// is selected) and the time (under manual pacing the key's pass-through time is the
    /// authoritative value) — fov is excluded: it is a read-only value the panel cannot act on,
    /// and the viewport is authoritative for capture (the current field of view is exactly the
    /// next capture value). When the time cannot be computed (no sampler), only the index is
    /// shown.
    static func keyframeSummary(index: Int, count: Int, time: Double?) -> String {
        let base = "\(index + 1)/\(count)"
        guard let time else { return base }
        return base + String(format: " · %.2f s", time)
    }

    /// The Radius disabled reason — when R̄ == 0 (a still shot with every key sitting on the
    /// pivot) the factor s = R/R̄ is undefined (in sync with hiding the radius handle). When
    /// enabled it is nil (no reason = nothing displayed).
    static func radiusDisabledReason(meanRadius: Float) -> String? {
        meanRadius == 0 ? "Radius needs spread keys — a still shot has no scale." : nil
    }

    /// The Ease disabled reason — under manual pacing the key pass-through times are
    /// authoritative, so the global ease is not applied (the key-time respect contract in the
    /// PacingMode comment). The string is the previous caption verbatim.
    static func easeDisabledReason(pacing: PacingMode) -> String? {
        pacing == .manual
            ? "Ease is bypassed in manual pacing — keyframe times are authoritative."
            : nil
    }

    /// The Capture Path disabled reason — sparse/0 is absent. The string is the previous
    /// caption verbatim.
    static func capturePathDisabledReason(hasColmapSparse: Bool) -> String? {
        hasColmapSparse ? nil : "Capture Path requires COLMAP-aligned views (sparse/0)."
    }

    /// The ⓘ prose of the PATH TRANSFORM section (caption diet — spec §4 rule 4: relocating the
    /// two permanently on-screen captions). The sentences are verbatim from the correction
    /// history — the handle manipulation convention is footnote 18, and the scope of the undo
    /// promise (the radius and height axes only — a horizontal drag has no corresponding field)
    /// is FIX 19 ③.
    static let transformInfo =
        "Handles: ✛ drag to move · hold ⇧ before dragging for vertical only "
        + "· ○ drag to scale radius.\n\n"
        + "Absolute values — retyping undoes radius and height changes only; "
        + "horizontal moves have no field."

    // MARK: The New Path ▾ pulldown (panel redesign Task 4 — spec §6: absorbing the PRESETS
    // section. Applying = creating a new path, then ordinary editing — the previous spec UX 6
    // semantics are unchanged)

    /// Rationale for the menu order (a discretionary arrangement — task-4-brief, "by usage
    /// frequency and narrative order"):
    /// ① Empty Path — the default action (the successor of the previous New Path button), so it
    ///    goes at the very top.
    /// ② The closed-loop family (Orbit 360° → Oval Loop) — the turntable is the most-used
    ///    preset, so it leads the group, and Oval Loop is its variation, so it sits adjacent.
    /// ③ The open one-shot family in narrative progression order: reveal (Reveal Arc → Pull Back
    ///    Reveal) → ascent (Rising Orbit) → descent (Spiral Descent → Crane Down) — the
    ///    ascent/descent symmetry is adjacent so the two read as a pair.
    /// ④ Capture Path — it derives from data (COLMAP) rather than camera choreography, so it
    ///    sits apart at the end, with a reason subtitle when disabled (a Text second label —
    ///    menu items have no caption row).
    private var newPathMenu: some View {
        Menu {
            Button {
                vm.addPath()
            } label: {
                Label("Empty Path", systemImage: "plus")
            }
            Divider()
            Button {
                vm.applyOrbitPreset()
            } label: {
                Label("Orbit 360°", systemImage: "arrow.triangle.2.circlepath")
            }
            Button {
                vm.applyOvalLoopPreset()
            } label: {
                Label("Oval Loop", systemImage: "oval")
            }
            Divider()
            Button {
                vm.applyRevealArcPreset()
            } label: {
                Label("Reveal Arc", systemImage: "point.topleft.down.curvedto.point.bottomright.up")
            }
            Button {
                vm.applyPullBackRevealPreset()
            } label: {
                Label("Pull Back Reveal", systemImage: "arrow.up.backward")
            }
            Button {
                vm.applyRisingOrbitPreset()
            } label: {
                Label("Rising Orbit", systemImage: "hurricane")
            }
            Button {
                vm.applySpiralDescentPreset()
            } label: {
                Label("Spiral Descent", systemImage: "tornado")
            }
            Button {
                vm.applyCraneDownPreset()
            } label: {
                Label("Crane Down", systemImage: "arrow.down.forward")
            }
            Divider()
            Button {
                guard let context = loadColmap() else { return }
                vm.applyCapturePreset(context: context)
            } label: {
                Label("Capture Path", systemImage: "camera.on.rectangle")
                if let reason = Self.capturePathDisabledReason(hasColmapSparse: hasColmapSparse) {
                    Text(reason)   // disabled reason subtitle (the menu form of spec §4 rule 5)
                }
            }
            .disabled(!hasColmapSparse)
        } label: {
            // Pulldown visual form: the shared section-action chrome (sectionActionChrome — app
            // finishing round 6 Task 1 absorbed the inline copy that used to live here into the
            // Theme.swift source of truth. Geometry and colors are the same as before) plus a
            // trailing chevron (the HIG pulldown indicator). The Menu's own indicator overlaps
            // and misaligns with a custom label, so it is hidden and drawn on the label instead
            // (the same technique as pathList's rowMenu). The chevron's caption2 and
            // textSecondary are inner specifications, so they take precedence over the chrome's
            // callout and textPrimary.
            HStack(spacing: 4) {
                Label("New Path", systemImage: "plus")
                Image(systemName: "chevron.down")
                    .font(.caption2)
                    .foregroundStyle(SplatXTheme.textSecondary)
            }
            .sectionActionChrome()
        }
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .help("Add an empty path or start from a preset move")
    }

    // MARK: Path list (panel redesign Task 1 — spec §3, the RUNS row pattern)

    private var pathList: some View {
        ForEach(vm.paths) { path in
            CinemaPathRow(
                path: path,
                isSelected: path.id == vm.selectedPathID,
                isEditing: editingPathID == path.id,
                draft: $editDraft,
                nameFocus: $nameFocus,
                // Re-selection is a no-op — selectPath comes with stopping playback and
                // resetting the playhead and key selection (the gate contract), so a click on
                // an already selected row MUST NOT fire it.
                onSelect: { if vm.selectedPathID != path.id { vm.selectPath(path.id) } },
                onRename: { beginRename(path) },
                onCommit: { endRename(after: .returnKey) },
                onCancel: { endRename(after: .escape) },
                onDuplicate: { vm.duplicatePath(id: path.id) },
                onDelete: { vm.deletePath(id: path.id) })
        }
    }

    private func beginRename(_ path: CameraPath) {
        guard editingPathID != path.id else { return }   // already editing this row — leave it
        // Cross-invalidation (FIX 1 ①, redefined by FIX 2 ⑤ to cancel first): an uncommitted
        // draft on another row or at the other entry point is **discarded**, exactly as if it
        // were "a click elsewhere" from this entry's perspective (double-click or the Rename
        // menu) — the previous FIX 1 committed first because blur = commit. The gate targets
        // (editingPathID and pathNameTargetID) are cleared first so that a blur onChange
        // arriving afterwards cannot double-cancel or mis-commit (idempotent — no race with
        // onChange ordering).
        cancelRename()             // cancel inline editing on the other row (discard the draft)
        pathNameTargetID = nil     // disarm the Name field commit gate = cancel
        if pathNameFocused {
            pathNameDraft = vm.selectedPath?.name ?? pathNameDraft   // restore the displayed value
        }
        // The draft is seeded from the **source of truth**, not from the path.name captured
        // before the re-render — the "gaining focus = reseed from the source of truth"
        // invariant (FIX 1 ① — it holds under blur = cancel too; see the nameDraft comment).
        let canonical = vm.paths.first(where: { $0.id == path.id })?.name ?? path.name
        editDraft = Self.nameDraft(after: .focusGained, draft: editDraft,
                                   canonical: canonical, fieldFocused: false)
        editingPathID = path.id
    }

    /// The shared exit gate for list inline editing — the pure source of truth
    /// renameExitCommits decides commit versus cancel (FIX 2 ⑤: Return alone commits; Esc and
    /// blur cancel).
    private func endRename(after exit: RenameExit) {
        if Self.renameExitCommits(exit) { commitRename() } else { cancelRename() }
    }

    /// Return commits only (FIX 2 ⑤ — blur goes to endRename's cancel branch and never reaches
    /// here).
    /// The cancel decision belongs to the pure gate renameCommit (FIX 1 ④ — a cancel clears
    /// editingPathID first, making this a no-op). editingPathID is cleared first here too, which
    /// also blocks double entry — independently of the VM's no-change guard, the view layer
    /// calls it only once.
    private func commitRename() {
        guard let commit = Self.renameCommit(target: editingPathID,
                                             draft: editDraft) else { return }
        editingPathID = nil
        vm.commitPathRename(id: commit.id, draft: commit.draft)
    }

    /// Esc and blur (clicking elsewhere) — discards the draft (FIX 2 ⑤: blur = cancel is an
    /// explicit user request. The macOS convention is blur = commit, and this deliberately
    /// differs — see the renameExitCommits comment).
    /// The VM held the original throughout the edit, so there is nothing to restore.
    private func cancelRename() {
        editingPathID = nil
        nameFocus = nil
    }

    // MARK: Selected path settings

    @ViewBuilder private func pathSection(_ path: CameraPath) -> some View {
        PanelSection(title: "PATH") {
            // Name — a parallel entry point alongside list inline editing (Task 3, spec §3 "let
            // it be fixed from the inspector too — the Apple convention"). The commit is
            // **Return only**; Esc and focus-out (clicking elsewhere) cancel (FIX 2 ⑤ —
            // renameExitCommits is the authoritative policy). The commit convention is the same
            // commitPathRename as the list. The cancel wiring is the body's
            // onChange(of: pathNameFocused).
            HStack {
                FieldLabel("Name")
                TextField("Name", text: $pathNameDraft)
                    .textFieldStyle(.roundedBorder)
                    .focused($pathNameFocused)
                    .onSubmit {
                        endPathNameField(after: .returnKey)
                        // Focus remains after Return — rearm the gate so editing can continue
                        // (reflecting the normalization result is done by endPathNameField's
                        // commit branch).
                        pathNameTargetID = path.id
                    }
                    .onExitCommand {
                        endPathNameField(after: .escape)   // clear the gate first, then blur — cancel
                        pathNameFocused = false
                    }
            }
            .onChange(of: path.name, initial: true) { _, name in
                // Follows the source of truth (the initial seed plus list edits and selection
                // changes) — the seeding policy is authoritative in the pure function
                // nameDraft(after:) (FIX 1 ①): while editing, the draft must preserve the
                // user's input, so it is overwritten only when unfocused. A change to the
                // source of truth while focused is absorbed by the reseed in the focus-gain
                // branch (cross-invalidation — see the comment at the top of the body).
                pathNameDraft = Self.nameDraft(after: .canonicalChanged, draft: pathNameDraft,
                                               canonical: name, fieldFocused: pathNameFocused)
            }

            // Duration — seconds are authoritative for time, and the frame count is shown
            // alongside as a derived value (spec UX 5 — avoiding research mistake ⑤). The frame
            // count is an attached caption (spacing 2), and the placeholder is the field name,
            // not the unit (spec §4 rule 3 — removing the defect where the same word appears
            // twice, inside and outside, in an empty field).
            // FIX 3 ②: drag scrubbing was added (ScrubbableNumberField — the geometry is
            // unchanged). The 0.1 lower bound is the same value as setDuration's clamp, and the
            // typing path's binding did not change by a single character (the scrub goes
            // through the same VM entry point).
            ScrubbableNumberField(
                label: "Duration", unit: "s",
                value: Binding(get: { vm.selectedPath?.duration ?? 0 },
                               set: { vm.setDuration($0) }),
                reference: path.duration,
                lowerBound: 0.1,
                caption: "≈ \(Int((path.duration * Double(vm.fps)).rounded())) frames"
                    + " @ \(vm.fps) fps",
                focus: $durationFocused,
                onScrub: { seconds, persistNow in
                    vm.setDuration(seconds, persistNow: persistNow)
                },
                onScrubOrphaned: { vm.persist() })

            // (The Smoothness slider was removed in Task 3 — it is not in the PATH section
            // composition of the spec §5 table. It is a store-only value the v1 sampler does not
            // consume for interpolation (the setSmoothness comment — Task 1 decision), so there
            // is no basis for it to occupy screen space. The model field, the VM API, and the
            // persistence are retained — the control comes back if v2 consumes it for
            // interpolation.)

            // Ease — not applied under manual pacing (the key-time respect contract — the
            // PacingMode comment).
            // Unified disabling (spec §4 rule 5): dim the whole row plus a one-line reason
            // (previously only the Picker was dimmed, and the reason caption floated at the same
            // spacing as the next control).
            // FIX 3 ①: previously only this row lacked a `.frame`, so the segmented control sat
            // right after the label while the five EXPORT rows were center-aligned, and the
            // alignment rule for the same control differed from section to section. It now goes
            // through the **same single SegmentedRow** as EXPORT (Theme.swift rule B①).
            VStack(alignment: .leading, spacing: 2) {
                SegmentedRow("Ease",
                             options: [SegmentOption(EaseProfile.none, "None"),
                                       SegmentOption(EaseProfile.easeInOut, "Ease In-Out")],
                             selection: Binding(get: { vm.selectedPath?.ease ?? .easeInOut },
                                                set: { vm.setEase($0) }))
                    .disabled(path.pacing == .manual)
                    .opacity(path.pacing == .manual ? 0.5 : 1)
                if let reason = Self.easeDisabledReason(pacing: path.pacing) {
                    PanelCaption(reason)
                }
            }

            VStack(alignment: .leading, spacing: 2) {
                // FIX 3 ①: small fixed controls are unified as trailing-aligned (Theme.swift
                // rule B③). Previously only Loop was leading (with a Spacer after it) while the
                // OVERLAY Anchor grid was trailing, so the alignment was contradictory inside
                // the same panel — it now lines up with the right edge of Duration, Radius, and
                // Height.
                HStack {
                    FieldLabel("Loop")
                    Spacer()
                    Toggle("Loop", isOn: Binding(
                        get: { vm.selectedPath?.loop ?? false },
                        set: { vm.setLoop($0) }))
                        .toggleStyle(.switch)
                        .controlSize(.mini)
                        .labelsHidden()
                }
                if path.loop, path.ease == .easeInOut, path.pacing == .even {
                    // Handed over from Task 1: quintic has zero endpoint velocity — it pauses at
                    // every loop seam. It is a one-line, state-triggered warning, so it stays on
                    // screen (it is not static prose — moving it into the ⓘ would lose the
                    // active signal at the moment the combination is switched on). Task 3
                    // re-review: even after the section restructuring, both trigger conditions
                    // (Loop and Ease) are in this section — it stays where it is.
                    PanelCaption("Ease In-Out pauses at the loop seam — None keeps"
                                 + " constant motion.", color: .orange)
                }
            }

            // Pacing status — the row itself is shown only under manual (FIX 2 ① — spec
            // footnote 2: the default, even, is not worth announcing. pacingRowVisible is
            // authoritative for the visibility condition — consistent with the spec §5 table's
            // "(Pacing + Even when manual)").
            if Self.pacingRowVisible(pacing: path.pacing) {
                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        FieldLabel("Pacing")
                        Spacer()
                        Text("Manual")
                            .font(.callout)
                            .foregroundStyle(SplatXTheme.accent)
                    }
                    Button("Even pacing") { vm.revertToEvenPacing() }
                        .font(.caption)
                        .help("Revert to constant-speed pacing — keyframe times become"
                              + " derived again")
                }
            }
        }
    }

    // (The unitText unit column moved to the shared UnitLabel (Theme.swift) in panel redesign
    // FIX 3 — as the numeric fields converged on the single ScrubbableNumberField, its consumers
    // moved out of this file.)

    // MARK: ORBIT (the PATH TRANSFORM section created by path transform Task 2, renamed in panel
    // redesign Task 3 — spec §5. Because these are absolute-value fields, typing directly is the
    // means of undo (§6 — a deliberate substitute for the absent undo infrastructure). Mutations
    // go through the VM entry points, which guarantees the same result as a handle drag, and the
    // §3.3 and §3.4 guards apply identically inside them. The static names transformInfo and
    // radiusDisabledReason are signatures consumed by tests and are retained — Global
    // Constraints, "preserve the existing API".)
    // FIX 8: the up axis consumes only the vm.sceneUpAxis snapshot — reading vm.viewer?.camera in
    // the body would attach an observation dependency to the entire camera struct and
    // re-evaluate the whole inspector on every orbit tick (the CinemaViewModel.sceneUpAxis
    // comment is the authoritative contract, spec footnote 8).

    @ViewBuilder private func orbitSection(_ pivot: PathTransform.Pivot) -> some View {
        // The Elevation field is deferred — the code and the measurements are preserved on the
        // feat/gizmo-affordance-elevation branch, and the decision and the resumption condition
        // (persisting the pivot in CameraPath) are authoritative in footnote 19 of
        // 2026-07-27-cinema-path-transform-design.md. This section has two fields: Radius and
        // Height.
        //
        // Caption diet (spec §4 rule 4): the two previously permanent on-screen captions (the
        // handle manipulation convention — footnote 18, and the scope of the undo promise —
        // FIX 19 ③) moved into the section header ⓘ (transformInfo). Since the section's
        // visibility condition is the same as the viewport handles', the FIX 19 ③ property "if
        // the handles are visible, the ⓘ is reachable" is preserved.
        PanelSection(title: "ORBIT", info: Self.transformInfo) {
            // Radius = R̄ (the mean distance to the pivot, in m). Entering a value R applies the
            // factor s = R/R̄.
            // A single-key still shot (R̄ == 0) has an undefined factor, so it is disabled (in
            // sync with hiding the radius handle).
            transformField(
                "Radius", focus: $radiusFocused,
                // The sensitivity reference is the value itself (R̄) with a scene-scale floor
                // (FIX 4 ④ — radiusReference is authoritative: so that after dragging down to
                // the 0.01 m floor the next drag is not stuck at 0.01 m/pt).
                // At a normal scale (R̄ ≥ 1 m) it is identical to before — ≈0.041 m/pt at 4.29 m.
                reference: NumericScrub.radiusReference(meanRadius: pivot.meanRadius),
                // Going to 0 or below would make the factor s = R/R̄ negative and collapse the
                // path.
                lowerBound: 0.01,
                disabledReason: Self.radiusDisabledReason(meanRadius: pivot.meanRadius),
                value: Binding(
                    get: { Double(vm.selectedPivot?.meanRadius ?? 0) },
                    set: { r in setRadius(r, persistNow: true) }),
                scrub: { r, persistNow in setRadius(r, persistNow: persistNow) })
            // Height = dot(pivot, upAxis) (in m). Entering a value h translates by
            // (h − current value)·upAxis.
            // (The label was previously "Center height" — abbreviated to fit the 52pt column.
            // The meaning is still the pivot height.)
            transformField(
                "Height", focus: $heightFocused,
                // For Height, 0 and negative values are normal, so the reference is the **scene
                // scale** rather than the value (the NumericScrub.heightReference comment is the
                // authoritative contract).
                reference: NumericScrub.heightReference(meanRadius: pivot.meanRadius),
                value: Binding(
                    get: { Double(simd_dot(vm.selectedPivot?.point ?? .zero, vm.sceneUpAxis)) },
                    set: { h in setHeight(h, persistNow: true) }),
                scrub: { h, persistNow in setHeight(h, persistNow: persistNow) })
        }
    }

    /// Applies Radius — shared by typing and scrubbing. Entering a value R = applying the
    /// factor s = R/R̄ (absolute-value semantics, so feeding the scrub's absolute target value
    /// directly makes it self-correct on every tick).
    private func setRadius(_ radius: Double, persistNow: Bool) {
        guard let p = vm.selectedPivot, p.meanRadius > 0, Float(radius).isFinite else { return }
        vm.scaleSelectedPath(by: Float(radius) / p.meanRadius, persistNow: persistNow)
    }

    /// Applies Height — shared by typing and scrubbing. Entering a value h = a translation of
    /// (h − current value)·upAxis.
    private func setHeight(_ height: Double, persistNow: Bool) {
        guard let p = vm.selectedPivot, Float(height).isFinite else { return }
        let up = vm.sceneUpAxis
        vm.translateSelectedPath(by: (Float(height) - simd_dot(p.point, up)) * up,
                                 persistNow: persistNow)
    }

    /// An ORBIT numeric field row — the same geometry as the Duration field (a 52pt label
    /// column, a trailing-aligned field, and a fixed-width unit column). The placeholder is the
    /// field name, not the unit (rule 3). Disabling dims the whole row and adds a one-line
    /// reason (rule 5). In FIX 3 ② the body moved to ScrubbableNumberField — drag scrubbing was
    /// added and the geometry and disabling conventions are unchanged.
    private func transformField(_ label: String, focus: FocusState<Bool>.Binding,
                                reference: Double,
                                lowerBound: Double? = nil,
                                disabledReason: String? = nil,
                                value: Binding<Double>,
                                scrub: @escaping (Double, Bool) -> Void) -> some View {
        ScrubbableNumberField(
            label: label, unit: "m", value: value, reference: reference,
            lowerBound: lowerBound, disabledReason: disabledReason,
            focus: focus, onScrub: scrub,
            // The ORBIT section is hidden entirely when the selected path or the pivot
            // disappears (visibleSections) — if it disappears mid-drag, onEnded never arrives,
            // so it saves once here (the same defense as
            // PathGizmoOverlay.finishCenterDragIfOrphaned).
            onScrubOrphaned: { vm.persist() })
    }

    // (The previous upAxis computed property was replaced by the vm.sceneUpAxis snapshot in
    // FIX 8 — the normalization and the degenerate +Y fallback moved to
    // CinemaViewModel.normalizedUpAxis.)

    // MARK: Selected keyframe (panel redesign Task 3 — spec §5: a read-only 3-row section with a
    // button → a lightweight row. It is a one-line summary that appears only when a key is
    // selected (keyframeSummary is authoritative — index and time, fov excluded) plus a delete
    // button. With no section header, the vertical shift at the moment a key is set is reduced
    // by one section's worth (§1 ⑥).

    @ViewBuilder private func keyframeRow(_ path: CameraPath, index: Int) -> some View {
        HStack(spacing: 6) {
            FieldLabel("Key")
            Text(Self.keyframeSummary(
                index: index, count: path.keyframes.count,
                time: vm.keyframeTimes.indices.contains(index)
                    ? vm.keyframeTimes[index] : nil))
                .font(.callout).monospacedDigit()
                .foregroundStyle(SplatXTheme.textPrimary)
            Spacer()
            Button(role: .destructive) {
                vm.deleteSelectedKeyframe()
            } label: {
                Image(systemName: "trash")
            }
            .controlSize(.small)
            .help("Delete the selected keyframe (⌫)")
        }
    }
}

/// A path list row (panel redesign Task 1) — the whole row is a selection Button (the
/// InspectorView RUNS row pattern); the name is normally a Text and becomes a TextField only
/// while editing. It is a leaf that receives all state as values and closures.
///
/// The hover @State **MUST live inside this leaf** — placing it at the list or inspector level
/// re-evaluates the whole list body on every hover move (the trap of this app recorded in
/// spec §1 ① and footnotes 8 and 9).
///
/// The ⋯ menu is an **overlay**, not inside the Button label — it is ahead of the row button in
/// z order, so it is structurally guaranteed that the click goes to the menu (putting it inside
/// the label would mean relying on the event routing of nested controls). On hover it swaps
/// place with the keys count, so there is no layout shift.
private struct CinemaPathRow: View {
    let path: CameraPath
    let isSelected: Bool
    let isEditing: Bool
    @Binding var draft: String
    var nameFocus: FocusState<UUID?>.Binding
    let onSelect: () -> Void
    let onRename: () -> Void
    let onCommit: () -> Void
    let onCancel: () -> Void
    let onDuplicate: () -> Void
    let onDelete: () -> Void
    @State private var hovering = false

    var body: some View {
        Button {
            guard !isEditing else { return }   // clicking a row being edited does not re-fire selection
            onSelect()
        } label: {
            HStack(spacing: 6) {
                nameView
                Spacer(minLength: 4)
                Text("\(path.keyframes.count) keys")
                    .font(.caption2).monospacedDigit()
                    .foregroundStyle(SplatXTheme.textSecondary)
                    .opacity(hovering ? 0 : 1)   // swaps place with the ⋯ menu (overlay) on hover
            }
            .padding(.horizontal, 8).padding(.vertical, 5)
            .background(isSelected ? SplatXTheme.accentDim : SplatXTheme.elevated,
                        in: RoundedRectangle(cornerRadius: 6))
            .overlay(RoundedRectangle(cornerRadius: 6)
                .stroke(isSelected ? SplatXTheme.accent : .clear))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .overlay(alignment: .trailing) {
            rowMenu
                .padding(.trailing, 8)
                .opacity(hovering && !isEditing ? 1 : 0)
                // opacity 0 does not disable hit testing (final review FIX 1 ⑤) — the click must
                // be disabled along with the hiding, otherwise a click at the far right of a row
                // being edited opens an invisible menu.
                .allowsHitTesting(hovering && !isEditing)
        }
        .onHover { hovering = $0 }
        .contextMenu {
            RenameButton()                     // editing entry ② — the label and shortcut come from the system
            Button("Duplicate") { onDuplicate() }
            Divider()
            Button("Delete", role: .destructive) { onDelete() }
        }
        .renameAction { onRename() }           // defines the action of RenameButton (context and ⋯ menus)
    }

    /// The name cell — a TextField while editing, otherwise a Text plus **double-click to enter
    /// editing** (FIX 2 ⑤ — regardless of selection, replacing the previous "click the selected
    /// row again"). Because it is a simultaneousGesture the leaf does not swallow the click, so
    /// a single click flows through to the row Button (selection) — the constraint that the
    /// previous count-1 leaf onTapGesture had to be attached "only to the selected row" (the
    /// trap where the gesture swallowed the click and clicking an unselected row's name was
    /// neither selection nor editing) is gone, and the branching drops to one as well.
    /// The first click of a double-click flows into row selection (an unselected row is selected
    /// and then edited — the same feel as Finder), and the duplicate firing of the Button action
    /// is neutralized by onSelect's re-selection guard.
    @ViewBuilder private var nameView: some View {
        if isEditing {
            TextField("Name", text: $draft)
                .textFieldStyle(.plain)
                .font(.callout)
                .foregroundStyle(SplatXTheme.textPrimary)
                .focused(nameFocus, equals: path.id)
                .onSubmit { onCommit() }
                .onExitCommand { onCancel() }
                // The field is created only when editing begins, so onAppear is exactly "focus
                // on entering editing" — setting it in beginRename would be an assignment before
                // the field is inserted and could be lost.
                .onAppear { nameFocus.wrappedValue = path.id }
        } else {
            Text(path.name)
                .font(.callout)
                .foregroundStyle(SplatXTheme.textPrimary)
                .simultaneousGesture(TapGesture(count: 2).onEnded { onRename() })
        }
    }

    private var rowMenu: some View {
        Menu {
            RenameButton()
            Button("Duplicate") { onDuplicate() }
            Divider()
            Button("Delete", role: .destructive) { onDelete() }
        } label: {
            Image(systemName: "ellipsis.circle")
                .font(.callout)
                .foregroundStyle(SplatXTheme.textSecondary)
        }
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Rename, duplicate or delete this path")
    }
}
