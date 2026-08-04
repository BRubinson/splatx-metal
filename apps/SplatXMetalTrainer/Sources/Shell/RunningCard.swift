import SwiftUI

/// The training progress card: an iter N/M progress bar, the elapsed time and ETA, and a stop
/// button.
struct RunningCard: View {
    let iter: Int
    let total: Int
    let elapsed: Double
    let onStop: () -> Void

    /// The approximation elapsed/iter × (total-iter). Undetermined when iter=0.
    private var eta: Double {
        guard iter > 0, total > iter else { return 0 }
        return elapsed / Double(iter) * Double(total - iter)
    }

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("iter \(iter)/\(total)")
                        .font(.callout.weight(.semibold)).monospacedDigit()
                        .foregroundStyle(SplatXTheme.textPrimary)
                    Spacer()
                    Text("\(Self.clock(elapsed)) elapsed · ETA \(Self.clock(eta))")
                        .font(.caption).monospacedDigit()
                        .foregroundStyle(SplatXTheme.textSecondary)
                }
                ProgressView(value: Double(iter), total: Double(max(total, 1)))
                    .tint(SplatXTheme.accent)
            }
            Button("Stop", role: .destructive, action: onStop)
                .buttonStyle(.bordered)
        }
        .padding(12)
        .background(SplatXTheme.elevated)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(SplatXTheme.hairline))
    }

    /// Seconds → m:ss (negative or undetermined gives --:--)
    private static func clock(_ seconds: Double) -> String {
        guard seconds > 0, seconds.isFinite else { return "--:--" }
        let t = Int(seconds.rounded())
        return String(format: "%d:%02d", t / 60, t % 60)
    }
}
