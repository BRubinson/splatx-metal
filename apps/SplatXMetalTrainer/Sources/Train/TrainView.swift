import SwiftUI

struct TrainView: View {
    @Bindable var session: ProjectSession
    @Bindable var model: AppModel

    private var vm: TrainViewModel { session.trainVM }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // The progress bar, the summary, and the chart are owned by the Inspector (app round 2
            // Task 2) — the main area is the live view (Task 5).
            // The run history was unified into the RUNS section of the View tab's Inspector (app
            // round 2 Task 7).
            mainArea
            if let e = vm.errorMessage {
                Label(e, systemImage: "xmark.octagon").foregroundStyle(.red)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .onExitCommand { if vm.isTraining { vm.cancel() } }
    }

    /// While training and once the live host is ready it shows the live splatting view; otherwise
    /// an informational placeholder (spec §3).
    /// The host is shared with the session's viewerVM — the same instance as the View tab. Since
    /// app round 5 Task 2 the input monitor is a ViewportRegistry (a weak-reference collection plus
    /// window/bounds routing), so registering multiple viewports is allowed, and when a tab switch
    /// tears down one SplatMetalView, the explicit detach in dismantleNSView removes the
    /// registration immediately (FIX 3 item 6 — the previous deferred automatic cleanup at the next
    /// input event survives as a safety net).
    @ViewBuilder private var mainArea: some View {
        if vm.isTraining, let host = session.viewerVM.host {
            ZStack(alignment: .topLeading) {
                SplatMetalView(host: host, camera: Binding(
                    get: { session.viewerVM.camera },
                    set: { session.viewerVM.camera = $0 }))
                liveBadge
                    .allowsHitTesting(false)   // no interference with orbit drags or zoom (the ViewportOverlay convention)
                    .padding(12)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ContentUnavailableView(
                "Live view",
                systemImage: "cube.transparent",
                description: Text(placeholderDescription))
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private var placeholderDescription: String {
        if session.project.state == .imported {
            "Run Align first. Training progress appears in the Inspector."
        } else if vm.isTraining {
            "Waiting for the first live update…"   // before the first snapshot (50 iter) swap
        } else {
            "Start training to see live reconstruction."
        }
    }

    /// The small top-left "LIVE · iter N" label — the same material and typography as
    /// ViewportOverlay.assetLabel.
    /// The iter source is not the progress callback but **the snapshot that reached the screen**
    /// (published at the streamer's swap completion, app round 3 spec §3) — so the badge matches
    /// the model actually being seen.
    private var liveBadge: some View {
        HStack(spacing: 6) {
            Circle().fill(.red).frame(width: 7, height: 7)
            Text("LIVE · iter \(vm.liveIter)")
                .font(.system(size: 11, weight: .semibold)).monospacedDigit()
                .foregroundStyle(SplatXTheme.textPrimary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 6))
    }
}
