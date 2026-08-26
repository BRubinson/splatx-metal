import SwiftUI
import Metal

/// The project detail = the central tab view + the console/status bar at the bottom + the
/// always-present Inspector on the right.
/// The AppModel.mode axis is gone — the tab axis is session.tab.
struct RootDetail: View {
    @Bindable var model: AppModel
    @Bindable var session: ProjectSession
    let registry: CommandRegistry

    var shownSplats: Int? {
        switch session.tab {
        case .align: nil
        case .train: model.splatCount
        case .view, .cinema: session.viewerVM.currentSplats
        }
    }

    var shownJob: JobState {
        if session.trainVM.isTraining { return model.job }
        // A video export (Cinema Task 5) takes priority over the tab-derived display — it carries
        // progress numbers.
        if case .exporting = model.job { return model.job }
        switch session.tab {
        case .view, .cinema: return session.viewerVM.host != nil ? .rendering : .idle
        case .align, .train: return .idle
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            Group {
                switch session.tab {
                case .align: AlignView(session: session, model: model)
                case .train: TrainView(session: session, model: model)
                // View and Cinema **share the same ViewerModel and host** (footnote 2 ① — no run
                // reload on a tab switch). Only the Cinema tab injects cinemaVM, which attaches
                // the timeline and the overlays.
                case .view: ViewerView(vm: session.viewerVM)
                case .cinema: ViewerView(vm: session.viewerVM, cinema: session.cinemaVM)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            Divider().overlay(SplatXTheme.hairline)
            ConsolePanel(log: model.console, expanded: $model.consoleExpanded)
            Divider().overlay(SplatXTheme.hairline)
            StatusBar(gpu: MTLCreateSystemDefaultDevice()?.name ?? "GPU",
                      splats: shownSplats,
                      thermal: StatusFormat.thermal(ProcessInfo.processInfo.thermalState),
                      job: shownJob)
        }
        .background(SplatXTheme.ground)
        .overlay {
            if model.showPalette {
                CommandPalette(registry: registry, isPresented: $model.showPalette)
            }
        }
        .onChange(of: session.tab) { _, tab in
            // The tab drives the Cinema VM's active state (footnote 2 ① — the previous toggle
            // inside the View tab was removed).
            // setActive(false) also cleans up by stopping playback and clearing the override (the
            // existing contract, unchanged).
            session.cinemaVM.setActive(tab == .cinema)
            openLatestIfNeeded(tab)
        }
        .task {   // for a trained project opened with .view as the initial tab, plus syncing the initial tab's active state
            session.cinemaVM.setActive(session.tab == .cinema)
            openLatestIfNeeded(session.tab)
        }
        // There is no `.toolbar` here on purpose (2026-08-05 spec §1.2). The window toolbar used to
        // carry the only Export… entry point for the open run; it now lives in the Inspector's
        // EXPORT section (RunPropertiesView), next to the run properties it acts on. The item was
        // deleted whole rather than made conditionally empty — an empty `.primaryAction` still
        // reserves its slot, which is what "training, result 아래쪽에 버튼을 배치하는게 더 깔끔할 거
        // 같아" was asking us to get rid of. Do not reintroduce it: two entry points for one action
        // was the explicitly rejected option (spec §0 decision 3).
        .inspector(isPresented: .constant(true)) {
            InspectorView(model: model, session: session)
                .inspectorColumnWidth(min: 240, ideal: 280, max: 400)
        }
    }

    /// On entering the View or Cinema tab, if no run is open yet it automatically opens the latest
    /// run (the same convention for .cinema — footnote 2 ①: so the Cinema tab does not start with
    /// an empty viewport).
    /// Only when currentRun == nil, so it does not overwrite a run the user picked from the
    /// history.
    /// It does not open during training (app round 3 Task 2) — openRun's removeAllChunks breaks the
    /// chunk tracking of the live swap chain and the old run and the live view are drawn
    /// **overlapping** (the round 2 file polling replaced everything and self-healed, but the
    /// round 3 incremental swap does not). On completion showRun opens it anyway.
    private func openLatestIfNeeded(_ tab: ProjectTab) {
        guard tab == .view || tab == .cinema, !session.trainVM.isTraining,
              session.viewerVM.currentRun == nil,
              let latest = session.project.latestRun else { return }
        Task {
            await session.viewerVM.openRun(latest, projectDirectory: session.directory,
                                           colmapContext: session.loadColmapContext())
        }
    }
}
