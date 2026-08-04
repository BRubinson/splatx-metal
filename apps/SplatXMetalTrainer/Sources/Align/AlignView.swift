import SwiftUI

/// The central view of the align tab: three states — a not-yet-aligned notice, in progress, and a
/// completion summary.
struct AlignView: View {
    @Bindable var session: ProjectSession
    @Bindable var model: AppModel

    var body: some View {
        let vm = session.alignVM
        VStack(spacing: 16) {
            if vm.isRunning {
                ProgressView(value: vm.fraction) {
                    Text(String(format: "Aligning… %.0f%%", vm.fraction * 100))
                        .font(.callout).monospacedDigit()
                        .foregroundStyle(SplatXTheme.textPrimary)
                }
                .frame(maxWidth: 360)
                // The Cancel button was removed (applying discretion under app round 4 spec
                // item 3) — stopping is unified into the single Start↔Stop switching position of
                // the Inspector CTA (the same pattern as train, removing the duplicate).
            } else if let summary = session.project.alignSummary {
                VStack(spacing: 8) {
                    Image(systemName: "checkmark.circle")
                        .font(.system(size: 40, weight: .light))
                        .foregroundStyle(SplatXTheme.accent)
                    Text("Aligned").font(.title3).foregroundStyle(SplatXTheme.textPrimary)
                    Text("\(summary.poseCount) poses · \(summary.pointCount.formatted()) points · \(Int(summary.alignSeconds))s")
                        .font(.callout).monospacedDigit()
                        .foregroundStyle(SplatXTheme.textSecondary)
                    if !session.project.runs.isEmpty {
                        Text("Re-aligning invalidates existing runs' pose basis.")
                            .font(.caption).foregroundStyle(.orange)
                    }
                }
            } else if session.project.state != .imported {
                // A COLMAP import — the aligned state with no align history of our own
                ContentUnavailableView("Imported COLMAP alignment in use",
                                       systemImage: "point.3.connected.trianglepath.dotted",
                                       description: Text("This project came with sparse/0. You can re-align from images if needed."))
            } else {
                ContentUnavailableView("Not aligned yet",
                                       systemImage: "camera.metering.matrix",
                                       description: Text("Run Align to estimate camera poses and a seed point cloud."))
            }
            if let e = vm.errorMessage {
                Label(e, systemImage: "xmark.octagon").foregroundStyle(.red).font(.callout)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}
