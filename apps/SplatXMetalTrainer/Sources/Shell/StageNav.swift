import SwiftUI

struct StageNav: View {
    @Binding var tab: ProjectTab
    /// App round 5 spec item 8: the lock during a long-running task — when non-nil, every button
    /// except that stage is disabled.
    /// `ProjectSession.lockedTab` is authoritative for the decision (the same gate as ⌥1/2/3 and
    /// goHome).
    var locked: ProjectTab? = nil
    private let stages: [(ProjectTab, String)] =
        [(.align, "Align"), (.train, "Train"), (.view, "View"), (.cinema, "Cinema")]
    var body: some View {
        HStack(spacing: 2) {
            ForEach(stages, id: \.0) { t, label in
                stage(t, label)
            }
        }
        .fixedSize()   // refuses toolbar principal compression → prevents label clipping
    }

    @ViewBuilder private func stage(_ t: ProjectTab, _ label: String) -> some View {
        let active = tab == t
        let lockedOut = locked != nil && locked != t
        let button = Button {
            tab = t
        } label: {
            VStack(spacing: 3) {
                Text(label)
                    .font(.system(size: 13, weight: active ? .semibold : .regular))
                    .foregroundStyle(active ? SplatXTheme.textPrimary : SplatXTheme.textSecondary)
                Rectangle().fill(active ? SplatXTheme.accent : .clear).frame(height: 2)
            }
            // App round 5 spec item 2: the tabs were widened horizontally (12 → 30) — "distinctly
            // wider".
            .padding(.horizontal, 30).padding(.top, 4)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(lockedOut)
        .opacity(lockedOut ? 0.4 : 1)
        // .help only while locked — it does not add a new tooltip to a tab in normal operation.
        if lockedOut {
            let reason: String = switch locked {
            case .train: "Locked while training is running"
            case .cinema: "Locked while a video render is running"
            default: "Locked while alignment is running"
            }
            button.help(reason)
        } else {
            button
        }
    }
}
