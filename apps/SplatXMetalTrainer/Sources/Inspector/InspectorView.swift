import SwiftUI
import SplatXMetal

struct InspectorView: View {
    @Bindable var model: AppModel
    @Bindable var session: ProjectSession

    var body: some View {
        Group {
            switch session.tab {
            case .align:
                VStack(spacing: 0) {
                    // Start↔Stop switching (app round 4 spec item 3 — the same pattern as train,
                    // applied at discretion): while running, the CTA in the same place becomes a
                    // red Stop.
                    if session.alignVM.isRunning {
                        Button {
                            session.alignVM.cancel()
                        } label: {
                            Text("Stop Align")
                        }
                        .buttonStyle(.destructiveAction)
                        .padding(12)
                    } else {
                        let alignReason = Self.startAlignDisabledReason(
                            imageCount: session.project.imageCount)
                        VStack(spacing: 6) {
                            Button {
                                session.alignVM.start(session: session, model: model)
                            } label: {
                                Text(session.project.state == .imported ? "Start Align" : "Re-align")
                            }
                            .buttonStyle(.primaryAction)
                            .disabled(alignReason != nil)
                            if let alignReason { PanelCaption(alignReason) }
                        }
                        .padding(12)
                    }
                    Divider()
                    ScrollView {
                        VStack(alignment: .leading, spacing: 8) {
                            PanelSection(title: "PROJECT") {
                                KeyValueRow(key: "name", value: session.project.name)
                                KeyValueRow(key: "source", value: session.project.sourceKind.rawValue)
                                KeyValueRow(key: "images", value: "\(session.project.imageCount)")
                                KeyValueRow(key: "state", value: session.project.state.rawValue)
                            }
                            if let s = session.project.alignSummary {
                                PanelSection(title: "ALIGNMENT") {
                                    KeyValueRow(key: "poses", value: "\(s.poseCount)")
                                    KeyValueRow(key: "points", value: s.pointCount.formatted())
                                    KeyValueRow(key: "time", value: "\(Int(s.alignSeconds))s")
                                }
                            }
                        }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            case .train:
                // session.trainVM is a let (fixed for the session lifetime), so binding through
                // $session is impossible — a local @Bindable obtains the draft binding (the
                // standard Observation pattern).
                @Bindable var trainVM = session.trainVM
                VStack(spacing: 0) {
                    VStack(spacing: 12) {
                        // The preset highlight is derived from the draft (TrainPreset.matching) —
                        // the stored preset state was removed (app round 3 spec §4). A tap only
                        // fills in values (apply) and the form is always visible, so the
                        // "Custom is unreachable" problem (app round 2 FIX 1) disappeared entirely.
                        PresetRow(draft: $trainVM.draft)
                            .disabled(session.trainVM.isTraining)
                        // Start↔Stop switching (app round 4 spec item 3): while training, the CTA
                        // in the same place becomes a red Stop. The separate Stop button in the
                        // progress block was removed (a duplicate).
                        if session.trainVM.isTraining {
                            Button {
                                session.trainVM.cancel()
                            } label: {
                                Text("Stop Training")
                            }
                            .buttonStyle(.destructiveAction)
                            TrainProgressBlock(progress: session.trainVM.store.latest,
                                               fallbackTotal: session.trainVM.draft.iterations)
                        } else {
                            // Audit cleanup Task 8: the disable decision and the reason come from
                            // the same source of truth — the previous
                            // `.disabled(!canStart || imported)` stayed silent about which term
                            // applied (violating §4 rule 5 — the Render CTA, Add Image, and
                            // StageNav in this same app all attach a one-line reason).
                            let trainReason = Self.startTrainingDisabledReason(
                                projectState: session.project.state,
                                hasDataset: session.trainVM.datasetURL != nil,
                                validationErrors: session.trainVM.draft.validationErrors())
                            Button {
                                session.trainVM.start(model: model, session: session)
                            } label: {
                                Text("Start Training")
                            }
                            .buttonStyle(.primaryAction)
                            .disabled(trainReason != nil)
                            if let trainReason { PanelCaption(trainReason) }
                        }
                    }
                    .padding(12)
                    if !session.trainVM.store.log.loss.isEmpty {
                        TrainingChart(log: session.trainVM.store.log)
                            .frame(height: 140)
                            .padding(.horizontal, 12)
                            .padding(.bottom, 12)
                    }
                    Divider()
                    // The options form is always visible (the gating was removed — spec §4): a
                    // preset only fills in values, and the user edits the numbers right there and
                    // runs.
                    OptionsForm(draft: $trainVM.draft).disabled(session.trainVM.isTraining)
                }
            case .view:
                ScrollView {
                    VStack(alignment: .leading, spacing: 8) {
                        // The single home of the run history UI (app round 2 Task 7 — the list at
                        // the bottom of TrainView was removed).
                        if !session.project.runs.isEmpty {
                            RunsSection(model: model, session: session)
                        }
                        RunPropertiesView(
                            record: session.viewerVM.currentRun,
                            loadedSplats: session.viewerVM.currentSplats)
                    }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
                }
            case .cinema:
                // The Cinema tab (2026-07-26 spec footnote 2 ①): the path inspector — promoted from
                // the previous isActive branch inside the View tab to the tab's own screen.
                CinemaInspector(
                    vm: session.cinemaVM,
                    model: model,
                    session: session,
                    loadColmap: { session.loadColmapContext() },
                    hasColmapSparse: FileManager.default.fileExists(
                        atPath: session.directory.appending(path: "sparse/0").path))
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        // The branding footer shared by every tab (app round 2 Task 8, generalized into
        // Theme/BrandFooter.swift in app round 5) — attached in exactly one place, on the
        // container. It is a safeAreaInset, so each tab's ScrollView content is not hidden.
        .safeAreaInset(edge: .bottom, spacing: 0) { BrandFooter() }
        .background(SplatXTheme.panel)
    }

    // MARK: CTA disabled reasons (audit cleanup Task 8 — the pure source of truth; tests:
    // InspectorCTAReasonTests)
    // The rule is authoritative in the 2026-07-29 cinema-panel-redesign spec §4 ("disabling is
    // .disabled + row dimming + a one-line reason"), but that spec's scope was limited to
    // Cinema/Theme and the Inspector was left outside it — it is now unified with the same idiom as
    // the Render CTA (renderDisabledReason) and Add Image (addImageDisabledReason).

    /// The single source of truth for the Start Align CTA's disabled reason — the `.disabled`
    /// decision uses this return value too (never separate the decision from the reason: if they
    /// diverge, "a disable with no reason" comes back).
    static func startAlignDisabledReason(imageCount: Int) -> String? {
        imageCount < 2 ? "Needs at least 2 images — import more images first." : nil
    }

    /// The Start Training CTA's disabled reason. Priority: align not complete (a pipeline
    /// prerequisite) → no dataset (a session anomaly) → an options validation error (the wording
    /// per cause is authoritative in validationErrors() — only the first item is shown so the user
    /// fixes them one at a time).
    static func startTrainingDisabledReason(projectState: ProjectState, hasDataset: Bool,
                                            validationErrors: [String]) -> String? {
        if projectState == .imported { return "Run Align first — training needs camera poses." }
        if !hasDataset { return "Project folder is unavailable — reopen the project." }
        return validationErrors.first.map { "Fix options: \($0)" }
    }
}

/// The preset selection row (app round 3 spec §4) — four segments (Fast/Balanced/Quality/Ultra,
/// with no Custom).
/// It is a custom button row in the style of StageNav rather than a Picker: the highlight is
/// derived from TrainPreset.matching(draft) rather than from a stored selection, so editing a value
/// in the form (a .custom result) turns the emphasis off by itself. The tap action only applies
/// (fills in the values) — the moment the values match a preset again, it is emphasized again.
private struct PresetRow: View {
    @Binding var draft: TrainOptionsDraft
    @Environment(\.isEnabled) private var isEnabled

    var body: some View {
        let active = TrainPreset.matching(draft)
        HStack(spacing: 4) {
            ForEach([TrainPreset.fast, .balanced, .quality, .ultra]) { p in
                let isActive = p == active
                Button {
                    p.apply(to: &draft)
                } label: {
                    Text(p.label)
                        .font(.system(size: 12, weight: isActive ? .semibold : .regular))
                        .foregroundStyle(isActive ? SplatXTheme.accent : SplatXTheme.textSecondary)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 6)
                        .background(isActive ? SplatXTheme.accentDim : SplatXTheme.elevated,
                                    in: RoundedRectangle(cornerRadius: 6))
                        .overlay(RoundedRectangle(cornerRadius: 6)
                            .stroke(isActive ? SplatXTheme.accent : SplatXTheme.hairline))
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .opacity(isEnabled ? 1 : 0.5)
    }
}

/// The run history section of the View tab's Inspector (app round 2 Task 7). Newest first; a row
/// shows the stamp, PSNR, splat count, and an outcome badge, and the currently open run is
/// emphasized in accent. Clicking a row opens it (showRun), and the context menu offers Open /
/// Export… (a PLY/SPZ submenu — app round 4 FIX 3) / Delete… (a confirmation dialog).
/// Opening and deleting are blocked during a task (training or align) — the decision uses the
/// authoritative lock `session.lockedTab` (FIX 3 item 1: generalizing the T5 handover guard that
/// looked only at isTraining — blocking contention over the live reload and host replacement, and
/// interference with align's project state transition). Putting .disabled on the row Button would
/// also kill the contextMenu and make Export unusable (which is safe: a finished run's files are
/// immutable), so instead of a blanket .disabled it is implemented as an action guard plus opacity
/// dimming plus per-menu-item disabling.
private struct RunsSection: View {
    @Bindable var model: AppModel
    @Bindable var session: ProjectSession
    @State private var pendingDelete: RunRecord?

    private var runsNewestFirst: [RunRecord] {
        session.project.runs.sorted { $0.createdAt > $1.createdAt }
    }
    private var isLocked: Bool { session.lockedTab != nil }

    var body: some View {
        PanelSection(title: "RUNS") {
            ForEach(runsNewestFirst) { r in row(r) }
        }
        .confirmationDialog(
            "Delete run \(pendingDelete?.id ?? "")?",
            isPresented: Binding(get: { pendingDelete != nil },
                                 set: { if !$0 { pendingDelete = nil } }),
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) {
                if let r = pendingDelete { session.deleteRun(r, model: model) }
                pendingDelete = nil
            }
            Button("Cancel", role: .cancel) { pendingDelete = nil }
        } message: {
            Text("The run folder and its output file are deleted permanently.")
        }
    }

    private func row(_ r: RunRecord) -> some View {
        let isCurrent = session.viewerVM.currentRun?.id == r.id
        return Button {
            guard !isLocked else { return }
            session.showRun(r)
        } label: {
            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 1) {
                    Text(r.id).font(.caption).monospacedDigit()
                        .foregroundStyle(SplatXTheme.textPrimary)
                    HStack(spacing: 6) {
                        if let psnr = r.finalPSNR {
                            Text(String(format: "%.1f dB", psnr)).foregroundStyle(.green)
                        }
                        Text("\(r.splatCount / 1000)k")
                            .foregroundStyle(SplatXTheme.textSecondary)
                    }
                    .font(.caption2).monospacedDigit()
                }
                Spacer(minLength: 4)
                if r.outcome != "completed" {
                    Text(r.outcome)
                        .font(.caption2).foregroundStyle(.orange)
                        .padding(.horizontal, 5).padding(.vertical, 2)
                        .background(.orange.opacity(0.15), in: Capsule())
                }
            }
            .padding(.horizontal, 8).padding(.vertical, 5)
            .background(isCurrent ? SplatXTheme.accentDim : SplatXTheme.elevated,
                        in: RoundedRectangle(cornerRadius: 6))
            .overlay(RoundedRectangle(cornerRadius: 6)
                .stroke(isCurrent ? SplatXTheme.accent : .clear))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .opacity(isLocked ? 0.5 : 1)
        .contextMenu {
            Button("Open") { session.showRun(r) }.disabled(isLocked)
            Menu("Export…") {
                ForEach(ExportFormat.allCases, id: \.self) { fmt in
                    Button(fmt.menuTitle) {
                        ExportService.presentPanel(for: r, session: session, model: model,
                                                   format: fmt)
                    }
                }
            }
            Divider()
            Button("Delete…", role: .destructive) { pendingDelete = r }.disabled(isLocked)
        }
    }
}

/// The compact training progress block sized for the Inspector width.
/// The clock and ETA logic is ported from RunningCard (Shell/RunningCard.swift) — the same
/// convention.
/// There is no Stop button (app round 4 spec item 3) — stopping belongs to the CTA at the top (the
/// Start↔Stop switch).
private struct TrainProgressBlock: View {
    let progress: SplatXTrainer.Progress?
    /// The total fallback before the first tick arrives (draft.iterations).
    let fallbackTotal: Int

    private var iter: Int { progress?.iteration ?? 0 }
    private var total: Int { progress?.runUntil ?? max(fallbackTotal, 1) }
    private var elapsed: Double { progress?.elapsedSeconds ?? 0 }

    /// The approximation elapsed/iter × (total-iter). Undetermined when iter=0.
    private var eta: Double {
        guard iter > 0, total > iter else { return 0 }
        return elapsed / Double(iter) * Double(total - iter)
    }

    /// "iter N/M · loss X [· PSNR Y dB]"
    private var statusLine: String {
        var s = "iter \(iter)/\(total)"
        if let p = progress {
            s += String(format: " · loss %.5f", p.loss)
            if let psnr = p.psnr { s += String(format: " · PSNR %.2f dB", psnr) }
        }
        return s
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ProgressView(value: Double(iter), total: Double(max(total, 1)))
                .tint(SplatXTheme.accent)
            Text(statusLine)
                .font(.caption).monospacedDigit()
                .foregroundStyle(SplatXTheme.textPrimary)
            Text("\(Self.clock(elapsed)) elapsed · ETA \(Self.clock(eta))")
                .font(.caption).monospacedDigit()
                .foregroundStyle(SplatXTheme.textSecondary)
        }
        .padding(10)
        .background(SplatXTheme.elevated, in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(SplatXTheme.hairline))
    }

    /// Seconds → m:ss (negative or undetermined gives --:--)
    private static func clock(_ seconds: Double) -> String {
        guard seconds > 0, seconds.isFinite else { return "--:--" }
        let t = Int(seconds.rounded())
        return String(format: "%d:%02d", t / 60, t % 60)
    }
}
