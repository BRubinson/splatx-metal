import SwiftUI

/// The bottom console, in the style of Xcode's debug area. A header strip plus (when expanded) a
/// scrolling log.
/// Collapsed by default — clicking the header or the chevron toggles `expanded` (the same state as
/// ⌘` and the View menu).
/// `log` is a value-type snapshot: when AppModel's console updates, the parent view passes a new
/// snapshot.
struct ConsolePanel: View {
    let log: ConsoleLog
    @Binding var expanded: Bool

    /// The stick-to-bottom state — stuck by default. The transition rules are in ConsoleAutoScroll:
    /// scrolling up releases it, and returning near the bottom re-sticks it.
    @State private var stickToBottom = true

    var body: some View {
        VStack(spacing: 0) {
            header
            if expanded {
                Divider().overlay(SplatXTheme.hairline)
                logBody
            }
        }
        .background(SplatXTheme.panel)
    }

    private var header: some View {
        Button { expanded.toggle() } label: {
            HStack(spacing: 6) {
                Image(systemName: expanded ? "chevron.down" : "chevron.right")
                    .font(.system(size: 9, weight: .bold))
                Text("Console")
                    .font(.system(size: 11, weight: .semibold)).tracking(0.6)
                Spacer()
                if !log.lines.isEmpty {
                    Text("\(log.lines.count)")
                        .font(.system(size: 10)).monospacedDigit()
                }
            }
            .foregroundStyle(SplatXTheme.textSecondary)
            .padding(.horizontal, 10).padding(.vertical, 4)
            .frame(maxWidth: .infinity)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private var logBody: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 1) {
                    ForEach(Array(log.lines.enumerated()), id: \.offset) { idx, line in
                        Text(line)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(SplatXTheme.textPrimary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(idx)
                    }
                }
                .padding(.horizontal, 10).padding(.vertical, 6)
            }
            .frame(height: 160)
            .background(SplatXTheme.elevated)
            // The stick-to-bottom state machine. Content growth (new lines) does not decrease the
            // offset, so it does not break the stick, and release happens only through the user
            // scrolling upward.
            .onScrollGeometryChange(for: ScrollGeometry.self) { $0 } action: { old, new in
                stickToBottom = ConsoleAutoScroll.nextStick(
                    current: stickToBottom,
                    previousOffsetY: old.contentOffset.y,
                    offsetY: new.contentOffset.y,
                    contentHeight: new.contentSize.height,
                    containerHeight: new.containerSize.height)
            }
            // Auto-scrolls to the bottom on a new line, but only while stuck.
            // It observes totalAppended rather than lines.count — once the ring buffer reaches its
            // cap, count stops while appends continue (and if it stops, the following breaks).
            .onChange(of: log.totalAppended) {
                guard stickToBottom, let last = log.lines.indices.last else { return }
                proxy.scrollTo(last, anchor: .bottom)
            }
            // Right after expanding it starts stuck — the latest lines are shown first.
            .onAppear {
                if let last = log.lines.indices.last { proxy.scrollTo(last, anchor: .bottom) }
            }
        }
    }
}
