import SwiftUI
import Foundation

enum StatusFormat {
    static func thermal(_ s: ProcessInfo.ThermalState) -> String {
        switch s { case .nominal: "nominal"; case .fair: "fair"; case .serious: "serious"; case .critical: "critical"; @unknown default: "?" }
    }
}

enum JobState: Equatable {
    // `.rendering` is already taken by the viewer-active display (derived from the View tab in
    // RootDetail.shownJob), so a video export (Cinema Task 5) is distinguished as a separate case
    // that carries progress numbers.
    case idle, training(iter: Int, total: Int), rendering, exporting(frame: Int, total: Int)
    var label: String {
        switch self {
        case .idle: "idle"
        case .training(let i, let t): "training \(i)/\(t)"
        case .rendering: "rendering"
        case .exporting(let f, let t): "render \(f)/\(t)"   // unified terminology (footnote 2 ④ — Render Video)
        }
    }
}

struct StatusBar: View {
    let gpu: String
    let splats: Int?
    let thermal: String
    let job: JobState
    var body: some View {
        HStack(spacing: 12) {
            Label(gpu, systemImage: "cpu").labelStyle(.titleAndIcon)
            if let n = splats { Text("\(n.formatted()) splats") }
            Text("thermal \(thermal)")
            Spacer()
            Text(job.label).foregroundStyle(job == .idle ? SplatXTheme.textSecondary : SplatXTheme.accent)
        }
        .font(.caption).monospacedDigit()
        .foregroundStyle(SplatXTheme.textSecondary)
        .padding(.horizontal, 10).padding(.vertical, 4)
        .frame(maxWidth: .infinity)
        .background(SplatXTheme.panel)
    }
}
