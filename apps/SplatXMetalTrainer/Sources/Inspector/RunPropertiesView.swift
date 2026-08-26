import SwiftUI

/// What the inspector can say about one run's timelapse video (2026-08-05 spec §2.2).
///
/// Three states rather than an `URL?`, because "no file" splits into two situations the user must
/// be able to tell apart: the encode is still running (wait), and there is nothing coming (the
/// timelapse was off, or fewer than two frames were captured). Collapsing them is exactly the
/// misreading §2.3 was written to prevent.
enum TimelapseArtifactState: Equatable {
    case none
    case encoding
    case ready(URL)
}

/// The body of the View tab's Inspector: the properties of the run open in the viewer, plus the
/// EXPORT section (2026-08-05 spec §1.2).
///
/// **Why EXPORT lives here and not on the toolbar.** Until this round the only entry point for the
/// open run was the window toolbar's `ToolbarItem(.primaryAction)` Menu — "export ply, spz 버튼이
/// 우측위에 구석에 숨어 있는데" (the user's own words). Exporting is an action *on the run whose
/// properties are printed directly above*, so it belongs under RESULT rather than in a window-level
/// chrome slot that every tab shares. The toolbar item was removed **whole** rather than emptied by
/// a condition — a conditionally empty `.primaryAction` leaves a dead slot behind.
/// The section sits inside InspectorView's ScrollView, so it is not sticky: with a long RUNS list
/// you have to scroll to reach it. That is the accepted cost (spec §1.2 — the goal is proximity to
/// the run, not permanent visibility).
///
/// **The timelapse (spec §2).** RESULT carries a `timelapse` row and EXPORT a third button. Both
/// read `timelapseState`, and both are needed: the button alone would be absent for the minutes
/// the encode takes and absent when no timelapse was made, two situations that look identical
/// from the outside. This is the first and only place in the app that reads
/// `RunRecord.timelapseFile` — before it, the file was written, recorded in the manifest, and
/// never shown.
///
/// Two decisions here are load-bearing — do not "tidy" them away:
/// - **The chrome is `.sectionActionChrome()`, never `.primaryAction`.** Theme.swift:206-207 draws
///   the line explicitly: the 44pt semibold CTA is for "the CTA that closes a section" (Start
///   Align / Start Training / Render Video…). Export closes nothing; it is a section-level action,
///   width rule A (Theme.swift:153-166).
/// - **No `isLocked` guard.** RunsSection keeps Export deliberately enabled while training or align
///   holds the lock, because "a finished run's files are immutable, which is safe"
///   (InspectorView.swift:273-276). Adding a lock guard to these buttons would silently reverse
///   that documented decision — and the guard would be wrong for the same reason it is wrong there.
struct RunPropertiesView: View {
    let record: RunRecord?
    /// The number of splats the viewer actually loaded.
    let loadedSplats: Int?
    /// The two objects `ExportService.presentPanel` has always required — `session` supplies the
    /// run directory and the project name that seeds the save panel's file name, `model` receives
    /// the "↑ Exported …" console line. Only the call site moved off the toolbar; the service
    /// signature is untouched.
    @Bindable var session: ProjectSession
    @Bindable var model: AppModel

    /// The judgment is in `timelapseState` (a static, pure function); the view only reads it.
    /// `session.trainVM` supplies the in-flight flag — the encode is owned by TrainViewModel and
    /// survives the tab switch into View, so there is no separate observable to add here.
    private var timelapse: TimelapseArtifactState {
        Self.timelapseState(record: record, directory: session.directory,
                            isEncoding: session.trainVM.isEncodingTimelapse,
                            fileExists: { FileManager.default.fileExists(atPath: $0.path) })
    }

    var body: some View {
        if let record {
            VStack(alignment: .leading, spacing: 8) {
                PanelSection(title: "TRAINING") {
                    KeyValueRow(key: "run", value: record.id)
                    KeyValueRow(key: "options", value: record.optionsSummary)
                    KeyValueRow(key: "outcome", value: record.outcome)
                    // A collapsed run reopened from the RUNS history days later (2026-08-16). The
                    // banner that first explained it lives on the Train tab and is gone by then, and
                    // "outcome: collapsed" on its own is a label, not an explanation — the user is
                    // left knowing something failed without knowing what to do about it. The note
                    // only appears for that one value, so nothing changes for every other run.
                    if RunHealth.isCollapsed(recordOutcome: record.outcome) {
                        Text(RunHealth.storedCollapseNote)
                            .font(.caption)
                            .foregroundStyle(.red)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                PanelSection(title: "RESULT") {
                    KeyValueRow(key: "splats", value: "\(record.splatCount.formatted())")
                    KeyValueRow(key: "loaded", value: loadedSplats.map { $0.formatted() } ?? "—")
                    KeyValueRow(key: "PSNR", value: record.finalPSNR.map {
                        String(format: "%.2f dB", $0) } ?? "—")
                    // Always present, even as "—". The row is what makes the timelapse *exist*
                    // as far as the UI is concerned: before this round the produced file was
                    // announced on one console line and the console is collapsed by default, so
                    // the honest answer to "how do I check the timelapse?" was "you cannot"
                    // (spec §2.1). Hiding the row when there is no file would restore exactly
                    // that silence for the case that needs the explanation most.
                    KeyValueRow(key: "timelapse", value: Self.timelapseRowValue(timelapse))
                }
                PanelSection(title: "EXPORT") {
                    // The format is chosen up front, one button per format — not a single button
                    // that offers both UTIs in the panel (app round 4 FIX 3: the panel's default
                    // file name ends in .ply, which made picking SPZ effectively impossible).
                    // ExportFormat.allCases is the same enumeration the removed toolbar menu drove,
                    // so a new format shows up here automatically.
                    ForEach(ExportFormat.allCases, id: \.self) { fmt in
                        exportButton(record: record, format: fmt)
                    }
                    // Only in `.ready`. A button for a file that is not on disk yet would have to
                    // be either disabled (a dead control the user waits on) or a save panel that
                    // fails at copy time; the RESULT row's "encoding…" carries that interval
                    // instead, which is why the row and the button ship together (spec §2.3).
                    if case .ready(let url) = timelapse {
                        timelapseButton(url: url)
                    }
                }
            }
        } else {
            Text("Train a run to view its result")
                .font(.callout)
                .foregroundStyle(SplatXTheme.textSecondary)
        }
    }

    /// One full-width export row.
    ///
    /// The label is `ExportFormat.menuTitle` verbatim rather than a string invented here — that
    /// wording is authoritative in ExportService.swift:27-32 and is exactly what the removed
    /// toolbar menu showed, so relocating the control does not also re-teach the user new words.
    ///
    /// `lineLimit(1)` + `minimumScaleFactor`: the inspector column shrinks to 240pt
    /// (RootDetail.swift:80) and "PLY (uncompressed)…" is a borderline fit at that width. Without
    /// this the label wraps and the chip grows to two lines, breaking the uniform row height of
    /// rule A. This is the same defense FieldLabel and SegmentedControl already apply for the same
    /// reason — shrink rather than wrap.
    private func exportButton(record: RunRecord, format: ExportFormat) -> some View {
        Button {
            ExportService.presentPanel(for: record, session: session, model: model, format: format)
        } label: {
            Text(format.menuTitle)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
                .sectionActionChrome()   // the chrome is authoritative in Theme.swift
        }
        .buttonStyle(.plain)
    }

    /// The timelapse export row. Structurally the same chip as `exportButton` — the two are kept
    /// as separate functions rather than one generic over a protocol because the label sources
    /// differ in kind: ExportFormat.menuTitle is a fixed word chosen at design time, whereas this
    /// label is derived from the file that actually exists on disk.
    private func timelapseButton(url: URL) -> some View {
        Button {
            ExportService.presentTimelapsePanel(source: url, session: session, model: model)
        } label: {
            Text(Self.timelapseButtonTitle(for: url))
                .lineLimit(1)
                .minimumScaleFactor(0.8)
                .sectionActionChrome()
        }
        .buttonStyle(.plain)
    }

    // MARK: - Pure judgments (the reason these are static: the view body cannot be tested, these
    // can — macOS privacy controls (TCC) rule out GUI automation on the development machine, so
    // anything decidable has to be decidable outside SwiftUI.)

    /// Resolves what the inspector should say about `record`'s timelapse.
    ///
    /// **Why `fileExists` is a parameter and not `FileManager.default` inline.** Two things must
    /// happen at once: the check must really run (the manifest and the disk drift apart in both
    /// directions, see below), and it must be testable without writing files. Injecting the
    /// predicate is the only shape that gives both — the same trick as `Project.removingRun`'s
    /// `hasSparse` parameter.
    ///
    /// **Why the disk is consulted at all.** `RunRecord.timelapseFile` is not proof of a file:
    /// - the encode can succeed while the manifest save that records it fails
    ///   (`TrainViewModel.finishTimelapseEncoding` surfaces exactly that failure) — file on disk,
    ///   record nil. Nothing can be done for this one here; the record is the only pointer.
    /// - the user can delete the video by hand — record set, file gone. **This is the case the
    ///   check catches**: without it the Export button would open a save panel that fails at
    ///   copy time, reporting an error for something the user did on purpose.
    /// The precedent is `ProjectCard.thumbnailURL` (Home/HomeView.swift), which refuses to trust
    /// `previewFile` for the same reason.
    ///
    /// **Why an existing file outranks `isEncoding`.** `isEncoding` is a fact about the *session*
    /// (one encode at a time, owned by TrainViewModel), while a file at this record's path is a
    /// fact about *this record*. They can only disagree when the user opens an older run while a
    /// new run encodes — and in that situation the older run's video genuinely is ready, so
    /// letting the session-wide flag win would hide a working Export button behind a progress
    /// message about a different run. The encoding branch is what remains once this record is
    /// known to have no file, which is precisely the §2.3 gap it was written for (the just
    /// finished run's `timelapseFile` is still nil).
    static func timelapseState(record: RunRecord?, directory: URL, isEncoding: Bool,
                               fileExists: (URL) -> Bool) -> TimelapseArtifactState {
        if let relative = record?.timelapseFile {
            // timelapseFile is project-relative and **already carries the `runs/<id>/` prefix**
            // ("runs/<id>/timelapse.gif"), so resolving it is one appendingPathComponent against
            // the project directory. This is *not* the shape `outputFile` has (FIX 1 — the
            // previous comment called them "the same convention"): outputFile is a bare file
            // name ("output.ply") and every reader has to build the folder part itself, which is
            // why ExportService.presentPanel writes `runs/\(record.id)/\(record.outputFile)`.
            // Both are relative to the project directory — that much they share, and it is what
            // makes the folder movable — but the two strings are not interchangeable, and
            // prefixing this one the way presentPanel prefixes outputFile yields
            // "runs/<id>/runs/<id>/timelapse.gif".
            let url = directory.appendingPathComponent(relative)
            if fileExists(url) { return .ready(url) }
        }
        return isEncoding ? .encoding : .none
    }

    /// The RESULT row's value. The ready state prints the **file name**, not a "GIF · 240 frames"
    /// summary: the row's job is to let the user find the thing, and the frame count is already
    /// on the console line that produced it. The file name also states the container, which is
    /// what the Export button's label has to agree with.
    static func timelapseRowValue(_ state: TimelapseArtifactState) -> String {
        switch state {
        case .none: "—"
        // The ellipsis marks it as ongoing; lowercase matches the other machine-ish values in
        // this section rather than shouting a status.
        case .encoding: "encoding…"
        case .ready(let url): url.lastPathComponent
        }
    }

    /// The Export button label, derived from the extension rather than from
    /// `project.timelapse.format`. The settings hold what the *next* run will produce; a run's
    /// file was encoded with whatever the settings said back then, and those two disagree the
    /// moment the user changes the format. The file on disk is the only source that cannot lie.
    /// An extension-less file (nothing produces one today) degrades to the bare label instead of
    /// printing "Timelapse ()…".
    static func timelapseButtonTitle(for url: URL) -> String {
        let ext = url.pathExtension.uppercased()
        return ext.isEmpty ? "Timelapse…" : "Timelapse (\(ext))…"
    }
}
