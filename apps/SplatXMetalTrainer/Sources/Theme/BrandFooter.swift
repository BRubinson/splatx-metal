import SwiftUI
import AppKit

/// The shared branding footer — "SplatX Metal" + "POSTMEDIA" above a hairline (keeping the app
/// round 2 Task 8 visuals).
/// App round 5 spec items 4 and 7: promoted to a shared component attached with
/// safeAreaInset(bottom) in both the Inspector and Home, with the two wordmarks turned into link
/// buttons (NSWorkspace.open, with a link cursor on hover).
/// The fonts, colors, and spacing are identical to the previous static presentation — only the
/// clickability is revealed through the cursor (no decoration added).
///
/// trainer-help-menu Task 1: the "Open Source Licenses" entry point that used to live here moved to
/// the Help menu. The old justification — "the footer is this app's only permanent surface, since
/// there is no separate Settings scene" — was wrong about *permanent*: the footer is only present
/// on Home and in the Inspector (it vanishes wherever the Inspector is not shown), whereas the menu
/// bar is present in every state of the app and is where macOS users look for notices anyway. What
/// remains here is branding only.
struct BrandFooter: View {
    var body: some View {
        VStack(spacing: 1) {
            LinkText(text: "SplatX Metal",
                     font: .system(size: 12, weight: .semibold),
                     color: SplatXTheme.textPrimary,
                     url: "https://splatx.postmedia.ai")
            LinkText(text: "POSTMEDIA",
                     font: .caption2,
                     color: SplatXTheme.textSecondary,
                     tracking: 1.5,
                     url: "https://postmedia.ai")
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 10)
        .background(SplatXTheme.panel)
        .overlay(alignment: .top) { SplatXTheme.hairline.frame(height: 1) }
    }
}

/// A one-line wordmark link. Button + NSWorkspace.open instead of a SwiftUI Link — so the existing
/// visuals (color and font) can be used as-is (Link forces a tint).
///
/// The cursor is `.pointerStyle(.link)` (macOS 15+, unconditionally available on the 26.0
/// deployment target) — **`NSCursor.pointingHand.push()/pop()` is forbidden** (app finishing
/// FIX 1 ③; this was the app's last push/pop path). The push/pop stack is process-global, so if the
/// pairing ever breaks the cursor freezes permanently as a pointing hand — SwiftUI's
/// `onHover(false)` is not guaranteed to be delivered when the view is destroyed while hovered
/// (reproduction: open a second viewer window, hold the cursor over the inspector footer's
/// wordmark, and close the main window with ⌘W — the pop never arrives). A declarative modifier has
/// no global state to recover in the first place, so getting stuck is structurally impossible — the
/// authoritative rationale is the ScrubbableNumberField header's "Cursor feedback" (for the same
/// reason, **neither an `onHover` safety net nor `onDisappear` cleanup code is needed**).
private struct LinkText: View {
    let text: String
    let font: Font
    let color: Color
    var tracking: CGFloat = 0
    let url: String

    var body: some View {
        Button {
            if let u = URL(string: url) { NSWorkspace.shared.open(u) }
        } label: {
            Text(text)
                .font(font)
                .tracking(tracking)
                .foregroundStyle(color)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .pointerStyle(.link)
    }
}
