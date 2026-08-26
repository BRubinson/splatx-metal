import SwiftUI

extension Color {
    init(hex: UInt32) {
        self.init(.sRGB,
                  red:   Double((hex >> 16) & 0xff) / 255,
                  green: Double((hex >> 8) & 0xff) / 255,
                  blue:  Double(hex & 0xff) / 255)
    }
}

enum SplatXTheme {
    /// The single source of truth for the hex values — the `ground` and `accent` Colors below are
    /// derived from them. (Path transform FIX 6 split them out to share with the Metal overlay,
    /// and since FIX 14 returned the overlay to SwiftUI the only consumer is the Color
    /// definitions again — the values themselves are the source of truth, so they stay.)
    static let groundHex: UInt32 = 0x131517
    static let accentHex: UInt32 = 0x2dd4bf

    static let ground        = Color(hex: groundHex)
    static let panel         = Color(hex: 0x16191b)
    static let elevated      = Color(hex: 0x1c2023)
    static let hairline      = Color(hex: 0x2a2f33)
    static let textPrimary   = Color(hex: 0xe6e8ea)
    static let textSecondary = Color(hex: 0x9aa0a6)
    static let accent        = Color(hex: accentHex)
    static let accentDim     = Color(hex: 0x233a37)
    static let destructive   = Color(hex: 0xef4444)
}

/// The shared button style for the main CTA (app round 3 spec §5). Full width at ~44 height, an
/// accent background, semibold 15pt.
/// When pressed the background dims, and when disabled it sinks to an elevated background with
/// secondary text.
/// No excessive decoration (no gradients or shadows) — consistent with the existing dark theme.
/// The destructive variant (app round 4 spec item 3): the same geometry with a red background —
/// used when a CTA in the same place switches to a destructive action, as with Start↔Stop. The
/// text follows primary in being "a tone darker than the background" (dark theme consistency — a
/// more settled impression than white text).
struct PrimaryActionButtonStyle: ButtonStyle {
    /// The CTA color variant. `.primary` = accent (teal), `.destructive` = red (Stop, cancel, and
    /// the like).
    enum Variant { case primary, destructive }
    var variant: Variant = .primary
    @Environment(\.isEnabled) private var isEnabled

    private var fill: Color {
        variant == .destructive ? SplatXTheme.destructive : SplatXTheme.accent
    }
    private var labelColor: Color {
        variant == .destructive ? Color(hex: 0x2a0b0b) : Color(hex: 0x0c211e)
    }

    func makeBody(configuration: Configuration) -> some View {
        let shape = RoundedRectangle(cornerRadius: 10)
        configuration.label
            .font(.system(size: 15, weight: .semibold))
            .foregroundStyle(isEnabled ? labelColor : SplatXTheme.textSecondary)
            .frame(maxWidth: .infinity, minHeight: 44)
            .background(isEnabled
                        ? fill.opacity(configuration.isPressed ? 0.75 : 1)
                        : SplatXTheme.elevated,
                        in: shape)
            .overlay(shape.stroke(isEnabled ? Color.clear : SplatXTheme.hairline))
            .contentShape(shape)
            .animation(.easeOut(duration: 0.08), value: configuration.isPressed)
    }
}

extension ButtonStyle where Self == PrimaryActionButtonStyle {
    /// For the `.buttonStyle(.primaryAction)` spelling.
    static var primaryAction: PrimaryActionButtonStyle { .init() }
    /// For the `.buttonStyle(.destructiveAction)` spelling — the same-position Stop CTA.
    static var destructiveAction: PrimaryActionButtonStyle { .init(variant: .destructive) }
}

// MARK: - The three-level panel contrast (panel redesign Task 2 — 2026-07-29 spec §4)
//
// Section title textPrimary 11pt semibold > field label textSecondary .caption (a fixed 52pt
// column) > caption textSecondary .caption2. Previously all three were textSecondary and the
// label (.callout) was larger than the title (11pt), so there was no anchor for the eye to catch
// (spec §1 ①). Changes in this file propagate through the entire app (the Align, Train, and View
// inspectors) — that is the intent (unifying the grammar app-wide).

struct SectionHeader: View {
    let title: String
    /// The ⓘ popover prose (caption diet — spec §4 rule 4): explanations, cautions, and accuracy
    /// conditions move here instead of into permanently on-screen captions. nil means title only
    /// (existing call sites unchanged).
    var info: String? = nil
    @State private var showInfo = false

    var body: some View {
        HStack(spacing: 4) {
            Text(title.uppercased())
                .font(.system(size: 11, weight: .semibold))
                .tracking(0.8)
                .foregroundStyle(SplatXTheme.textPrimary)
            if let info {
                Button {
                    showInfo.toggle()
                } label: {
                    Image(systemName: "info.circle")
                        .font(.system(size: 10))
                        .foregroundStyle(SplatXTheme.textSecondary)
                }
                .buttonStyle(.plain)
                .help("More about this section")
                .popover(isPresented: $showInfo, arrowEdge: .bottom) {
                    Text(info)
                        .font(.caption)
                        .multilineTextAlignment(.leading)
                        .frame(width: 240, alignment: .leading)
                        .padding(12)
                }
            }
        }
        // Top 10 (previously 8 — panel redesign FIX 2 ④): the section boundary grows in
        // proportion to the row spacing going 6 → 8, preserving the "row < section" hierarchy
        // (the effective section boundary gap = the section stack spacing 8 + this 10).
        .padding(.top, 10).padding(.bottom, 2)
    }
}

/// The section container — row spacing 8 (previously 6 — panel redesign FIX 2 ④, spec footnote 2:
/// the user's real-use feedback, "everything is crammed together". The section boundary is
/// preserved in proportion by SectionHeader's top padding going 8 → 10).
/// A caption attached to a control is bound to that control at the call site with
/// VStack(spacing: 2):
/// "control↔attached caption" (2) < "caption↔next control" (8) — proximity states membership
/// (spec §4 rule 6. The previous uniform spacing of 4 made a caption's membership impossible to
/// determine — §1 ①. The 2 of the caption grouping stays as-is — increasing it along with the
/// rest would break rule 6).
struct PanelSection<Content: View>: View {
    let title: String
    var info: String? = nil
    @ViewBuilder var content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionHeader(title: title, info: info)
            content
        }
    }
}

// MARK: - Row width rules (panel redesign FIX 3 — 2026-07-29 spec footnote 3)
//
// Inspector rows come in **two tiers**, and within a tier there is exactly one width and
// alignment rule. If this rule wavers, "every control a different width with ragged margins"
// comes back (the point where FIX 2 ③ tried to fix it with `.frame(maxWidth: .infinity)` and
// failed — footnote 3 ①).
//
// **A. Section-level actions = full content width with no label column.**
//    New Path ▾ / Add Image… / Add SplatX watermark / Render Video… / the Start and Stop CTA /
//    PresetRow / the View-tab Inspector's EXPORT buttons (PLY / SPZ — RunPropertiesView, added
//    2026-08-05 spec §1.2 when the window toolbar's Export… menu was removed — plus the
//    conditional Timelapse (GIF/MP4/MOV) button from §2.2, which uses the same chrome so that a
//    row appearing and disappearing does not change the shape of the section).
//    `.frame(maxWidth: .infinity)` plus their own chrome. There is no label, so full width is
//    correct.
//    Note the EXPORT buttons are rule A and **not** the 44pt `.primaryAction` CTA: exporting does
//    not close its section (see the PrimaryActionButtonStyle note below), it is an action offered
//    alongside the run's read-only rows.
//    Among those, the elevated chip form (New Path ▾, Add Image…, Add SplatX watermark) is
//    authoritative in `sectionActionChrome` (app finishing round 6 Task 1 — absorbing the two
//    previous inline copies).
//
// **B. Label-value rows = FieldLabel (52pt) + the HStack default spacing of 8 → the rest belongs
//    to the value.**
//    The value side is one of three, depending on the nature of the control, and arbitrary mixing
//    is forbidden:
//    ① **Flexible-width controls (segmented, slider, name TextField) → fill the rest.**
//       Segmented controls MUST go through `SegmentedControl` — macOS's
//       `.pickerStyle(.segmented)` **refuses** a proposal larger than its own intrinsic width, so
//       `.frame(maxWidth: .infinity)` cannot stretch the control and only centers it (measured:
//       the same five rows came out anywhere from 102 to 180pt wide). Do not revert to the native
//       picker.
//    ② **Numeric input (a fixed 64pt field + a 14pt UnitLabel) → trailing-aligned with
//       `Spacer()`.**
//       Duration, Radius, and Height. Their right edges align with the right edge of the content.
//    ③ **Things that cannot fill the width (read-only values, small fixed controls) →
//       trailing-aligned with `Spacer()`.**
//       KeyValueRow, the Pacing status, the Loop switch, and the Anchor 3×3 grid. Before FIX 3
//       only Loop was leading (with a Spacer after it), so small-control alignment was
//       contradictory inside the same panel — it was unified to trailing so that the right edges
//       of ② and ③ line up.

/// The elevated chip chrome of section-level actions (row width rule A) — the single source of
/// truth (app finishing round 6 Task 1).
/// `Add Image…` and `Add SplatX watermark` (Button labels) and `New Path ▾` (a Menu label) all go
/// through this one: previously the same chrome was inline-duplicated in two places,
/// OverlayEditor and CinemaInspector.
///
/// The geometry is fixed (.callout, vertical padding 5, cornerRadius 6, full width, an elevated
/// background, and a hairline border) and `tint` varies **only the label color** — the
/// enforcement device for "the same style, only the color differs". tint is chosen only from the
/// nine existing tokens (SplatXTheme) — no new colors. There is no background or border variant —
/// an accentDim background with an accent border is the **selected state** combination of
/// SegmentedControl, so using it on a button would be misread as "selected".
///
/// Why a ViewModifier rather than a ButtonStyle: a custom label on a macOS `Menu` does not pass
/// through a custom ButtonStyle's makeBody, so one style cannot cover both Button and Menu — a
/// modifier attached on the label side applies identically to both. Partial fonts and colors
/// inside the content (the caption2 and textSecondary of the New Path chevron) are preserved by
/// SwiftUI's inner-specification-wins rule.
/// Do not confuse it with `PrimaryActionButtonStyle` (the 44pt semibold CTA) — that one is for
/// the CTA that closes a section.
struct SectionActionChrome: ViewModifier {
    var tint: Color = SplatXTheme.textPrimary

    func body(content: Content) -> some View {
        content
            .font(.callout)
            .foregroundStyle(tint)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 5)
            .background(SplatXTheme.elevated, in: RoundedRectangle(cornerRadius: 6))
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(SplatXTheme.hairline))
            .contentShape(Rectangle())
    }
}

extension View {
    /// For the `.sectionActionChrome()` spelling — per the row convention, the caller applies
    /// `.disabled + .opacity` for the disabled dimming (the same reason as SegmentedControl:
    /// dimming again here would split the brightness of the label and the chrome).
    func sectionActionChrome(tint: Color = SplatXTheme.textPrimary) -> some View {
        modifier(SectionActionChrome(tint: tint))
    }
}

/// One segmented option — a pairing of a value and a display label. The input to
/// `SegmentedControl`.
struct SegmentOption<Value: Hashable>: Identifiable {
    let value: Value
    let label: String
    var id: Value { value }
    init(_ value: Value, _ label: String) { self.value = value; self.label = label }
}

/// The single source of truth for the segmented control (panel redesign FIX 3 ① — spec
/// footnote 3).
///
/// **Why not `Picker(.segmented)`**: for the macOS segmented Picker, the SwiftUI adapter uses
/// `NSSegmentedControl`'s intrinsic width directly as the layout width — it does not accept a
/// larger proposal (asymmetrically, it partially honors shrinking). So
/// `.frame(maxWidth: .infinity)` could not stretch the control and acted only as a **centering
/// container** (the cause of the widths still being ragged in real use, when FIX 2 ③ was believed
/// to have made them uniform). Specifying an exact width, GeometryReader, the `.palette` style,
/// and expanding the segment labels were all measured and confirmed ineffective — do not retry
/// them.
///
/// Instead it generalizes an idiom that already existed in this app (`PresetRow` —
/// Inspector/InspectorView.swift): `HStack(spacing: 4)` with `.frame(maxWidth: .infinity)` on
/// each cell, so one cell fills, two split in half, three split in thirds, and four split in
/// quarters — **guaranteed by layout arithmetic**, for any count. The colors and chrome use the
/// dark theme tokens as-is (accent/accentDim/hairline/elevated).
///
/// Four cells first appear in EXPORT's Size row under GIF (2026-08-05 spec §5.3). The label
/// width note below is measured for thirds; at quarters it is ~49pt, so that row shortens its
/// own labels rather than leaning on minimumScaleFactor
/// (`ExportInspectorSection.sizeOptions` — the caller owns label text, this control does not).
///
/// **It does not dim on its own** — the disabled presentation is "dim the whole row" (spec §4
/// rule 5), so the caller applies `.disabled + .opacity` to the row. Dimming again here would
/// split the brightness of the label and the control.
///
/// Accessibility: it loses the native segmented control's arrow-key traversal and segmented role
/// (a loss `PresetRow` already carries — consistent within the app). In exchange, each cell
/// carries an `.isSelected` trait so VoiceOver reads the selection state.
struct SegmentedControl<Value: Hashable>: View {
    let options: [SegmentOption<Value>]
    @Binding var selection: Value

    var body: some View {
        HStack(spacing: 4) {
            ForEach(options) { option in
                let active = option.value == selection
                Button {
                    // Re-selection is a no-op — the setter is a gate (setOutputAspect, setEase,
                    // and so on), so reassigning the same value must not re-fire the derived
                    // refresh and the persistence.
                    if selection != option.value { selection = option.value }
                } label: {
                    Text(option.label)
                        .font(.system(size: 12, weight: active ? .semibold : .regular))
                        .foregroundStyle(active ? SplatXTheme.accent : SplatXTheme.textSecondary)
                        // The inspector width varies from 240 to 400 — split in thirds at 240pt
                        // that is ~52pt per cell, so labels like "1440p" can wrap (the same
                        // remedy as FieldLabel:145).
                        .lineLimit(1)
                        .minimumScaleFactor(0.7)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 5)
                        .background(active ? SplatXTheme.accentDim : SplatXTheme.elevated,
                                    in: RoundedRectangle(cornerRadius: 6))
                        .overlay(RoundedRectangle(cornerRadius: 6)
                            .stroke(active ? SplatXTheme.accent : SplatXTheme.hairline))
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(option.label)
                .accessibilityAddTraits(active ? [.isSelected] : [])
            }
        }
        .frame(maxWidth: .infinity)
    }
}

/// A label + segmented row (rule B①) — it splits **the entire remaining width** evenly after the
/// 52pt label column.
/// All five EXPORT rows (Aspect, Rate, Format, File, and Size) and PATH's Ease go through this
/// one, so identical widths are structurally guaranteed.
struct SegmentedRow<Value: Hashable>: View {
    let label: String
    let options: [SegmentOption<Value>]
    @Binding var selection: Value

    init(_ label: String, options: [SegmentOption<Value>], selection: Binding<Value>) {
        self.label = label
        self.options = options
        self._selection = selection
    }

    var body: some View {
        HStack {
            FieldLabel(label)
            SegmentedControl(options: options, selection: $selection)
        }
    }
}

/// The unit column — a fixed width puts m and s at the same x (spec §4 rule 3: previously the
/// width varied with the glyph, so the field's right edge shifted row by row). FIX 3 generalized
/// CinemaInspector's private `unitText` — it is now shared with the numeric field component
/// (ScrubbableNumberField).
struct UnitLabel: View {
    let text: String
    init(_ text: String) { self.text = text }
    var body: some View {
        Text(text)
            .font(.caption)
            .foregroundStyle(SplatXTheme.textSecondary)
            .frame(width: 14, alignment: .leading)
    }
}

/// The field label — a fixed 52pt column (spec §4 rule 2). It generalizes this app's existing
/// rule (originating in the OVERLAY sliders and the EXPORT pickers), so every label-value row
/// aligns to the same baseline.
struct FieldLabel: View {
    let text: String
    init(_ text: String) { self.text = text }
    var body: some View {
        Text(text)
            .font(.caption)
            .foregroundStyle(SplatXTheme.textSecondary)
            .lineLimit(1)
            .minimumScaleFactor(0.85)   // prevents clipping of 52pt-borderline labels like "Smoothness"
            .frame(width: 52, alignment: .leading)
    }
}

/// A one-line attached caption (.caption2) — it MUST be bound to the control it belongs to with
/// VStack(spacing: 2) (tighter than PanelSection's row spacing of 8 — spec §4 rule 6). The only
/// captions allowed to stay on screen are the one-liner explaining "why this is disabled right
/// now" and a derived value shown alongside (rules 4 and 5).
struct PanelCaption: View {
    let text: String
    var color: Color = SplatXTheme.textSecondary
    init(_ text: String, color: Color = SplatXTheme.textSecondary) {
        self.text = text; self.color = color
    }
    var body: some View {
        Text(text).font(.caption2).foregroundStyle(color)
    }
}

struct KeyValueRow: View {
    let key: String; let value: String
    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            FieldLabel(key)
            Spacer()
            Text(value)
                .font(.callout)
                .foregroundStyle(SplatXTheme.textPrimary)
                .monospacedDigit()
                .multilineTextAlignment(.trailing)
        }
    }
}
