// Panel redesign FIX 3 ② (2026-07-29 spec footnote 3): a numeric field row with drag scrubbing.
// The geometry is exactly the previous `transformField`/Duration row (a 52pt FieldLabel column +
// Spacer + a trailing-aligned 64pt TextField + a 14pt UnitLabel + row dimming and a one-line
// reason when disabled — §4 rules 3 and 5).
// The arithmetic is authoritative in NumericScrub (pure — this app's only automated line of
// defense); this file is wiring only.
//
// ── Why a transparent overlay ────────────────────────────────────────────
// Attaching `.gesture`/`.simultaneousGesture` directly to a TextField has already failed in this
// repo — the AppKit field editor consumes the mouse-down and the ancestor gesture never fires
// (the accident record at CinemaInspector.swift:12-14). So a transparent surface is laid over the
// field to take the hit first in z order. The cost is "clicks do not reach the TextField either",
// which is why **both** of the wirings below are required (miss either one and typing becomes
// impossible, or text selection does):
//   ⑴ A release with no movement = a tap → set @FocusState programmatically.
//   ⑵ While focused, turn the surface off with `.allowsHitTesting(false)` → caret movement, drag
//      selection, and double-click word selection all work as usual.
//   ⑶ Return and Esc → drop focus (.onSubmit/.onExitCommand — FIX 4 ②). This is the back door of
//      ⑵: blur-based restoration is free, but Return and Esc are wired manually, so omitting them
//      leaves focus on a field you typed into forever, the surface stays lifted, and scrubbing
//      dies (dragging only selects text).
// The resulting state machine is the same as Unreal's spin box: unfocused = the scrub surface,
// focused = the text surface, and Return/Esc/blur = back to the scrub surface. Losing the caret
// position of the first click and getting a select-all instead is the same in Unity and Unreal,
// so it is accepted.
// Do not set focus when a scrub ends — doing so makes the next scrub immediately impossible.
//
// ── Cursor feedback (app finishing 2026-07-29 — the conclusion reversed, the mechanism still
//    forbidden) ────────
// **`NSCursor.push()/pop()` is still forbidden.** The risk that ruled it out remains true: the
// push/pop stack is **process-global**, so if the pairing ever breaks (view teardown — the ORBIT
// section disappears entirely when the selected path changes (visibleSections) / a `.disabled`
// transition / a focus transition / window deactivation or ⌘Tab / `onEnded` never arriving), the
// cursor gets **permanently stuck** as a left-right arrow. In particular the premise that
// `onEnded` is a deterministic terminator does not hold — this very file acknowledges "paths
// where onEnded does not arrive" and defends with `onScrubOrphaned`/`onDisappear` (see the save
// convention below).
//
// What was reversed is **the availability of a mechanism**. The deployment target is macOS 26.0,
// so SwiftUI `.pointerStyle(_:)` (macOS 15+) is usable without conditional compilation. It is a
// **declarative** modifier with no stack and no pairing, so "getting stuck" is structurally
// impossible — there is no global state to recover in the first place, and every loss point above
// reduces to "the modifier drops out of the applied target", after which AppKit reapplies that
// region's default cursor. That is why having no cleanup code and no safety-net flag is
// **correct** (they are not missing — they are unnecessary; the moment state is introduced,
// getting stuck comes back). Do not "improve" this one line into an `@State isScrubbing` branch.
//
// It is applied **inside** `.allowsHitTesting(!focus)` — while focused (= typing mode) the whole
// subtree is off and the TextField's I-beam survives intact. It also applies on hover, and that is
// intentional: the original purpose was discoverability (previously a `.help()` tooltip stood in
// for it because of the cost), and a cursor that changes only during a drag tells you only after
// you have already manipulated it, so it is not an affordance. The same diagnosis is the
// conclusion of PathGizmoOverlay FIX 19 (aiming failure = absent affordance → a hover highlight).
// The `.help()` tooltip ("click to type") stays to compensate for the lost I-beam.
//
// ── Save convention ─────────────────────────────────────────────────────
// `persistNow: false` during the drag and `true` on release (the same as the gizmo handles — one
// save per drag). `persist()` → `store.save(project)` is a **synchronous manifest write to disk**,
// so saving on every tick hammers the disk at mouse-event frequency. The unsaved state of a case
// where the view disappears mid-drag and onEnded never arrives (section teardown) is prevented by
// `onScrubOrphaned` — the same defense as PathGizmoOverlay.finishCenterDragIfOrphaned.
// ⚠ Consumer setter trap (FIX 4 ①): onEnded re-enters **the same absolute target value** with the
// same translation as the last onChanged (a consequence of absolute-target arithmetic). If the
// setter has a no-change early return, the persistNow:true of the release never reaches the save
// and the drag result lives only in memory — such a setter must discharge its save obligation
// regardless of the value comparison (the hasUnpersistedChanges flush in setDuration is both the
// example and the source of truth. Test: testDurationScrubPersistTiming).
import AppKit
import SwiftUI

struct ScrubbableNumberField: View {
    let label: String
    /// The unit column string (m, s). It is not reused as the placeholder (§4 rule 3).
    var unit: String
    /// The binding of the typing path — it takes the existing setter **verbatim** (preserving
    /// absolute-value semantics).
    @Binding var value: Double
    /// The sensitivity reference. Duration uses the value itself, Radius uses the value itself
    /// with a scene-scale floor (NumericScrub.radiusReference — FIX 4 ④: so that after dragging
    /// down to the 0.01 m floor the next drag is not stuck), and Height uses the scene scale
    /// (heightReference — 0 and negative values are normal, so the value cannot serve as the
    /// reference).
    let reference: Double
    /// The consumer's domain lower bound (nil when there is none). The NumericScrub.value comment
    /// is the authoritative contract.
    var lowerBound: Double? = nil
    /// The attached derived caption (when enabled) — for example Duration's
    /// "≈ N frames @ M fps".
    var caption: String? = nil
    /// The disabled reason. When non-nil the whole row is disabled and dimmed, and this sentence
    /// takes the caption's place.
    var disabledReason: String? = nil
    var focus: FocusState<Bool>.Binding
    /// A scrub tick — (the absolute target value, persistNow). false during the drag, true on
    /// release.
    let onScrub: (Double, Bool) -> Void
    /// Saves once when the view disappears mid-drag (orphaned-drag finalization).
    let onScrubOrphaned: () -> Void

    /// The basis latched at drag start — the origin of the absolute-target arithmetic (the
    /// NumericScrub.value comment).
    private struct ScrubAnchor {
        let base: Double
        let reference: Double
    }
    @State private var anchor: ScrubAnchor?

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                FieldLabel(label)
                Spacer()          // rule B② — numeric input is trailing-aligned
                TextField(label, value: $value,
                          format: .number.precision(.fractionLength(0...2)))
                    .textFieldStyle(.roundedBorder)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 64)
                    .focused(focus)
                    // FIX 4 ②: Return and Esc = drop focus → back to the scrub surface (header
                    // state machine ⑶). It is in the same vein as the name field's manual Esc
                    // blur idiom (CinemaInspector's .onExitCommand), but deliberately differs
                    // from the Name field's "rearm focus after Return" — here it must return to
                    // scrubbing the instant the value is committed.
                    // Value commitment follows the standard convention of
                    // TextField(value:format:) (Return commits and then onSubmit fires; Esc ends
                    // editing via blur). Dropping focus flows into the consumer's
                    // .onChange(of: …Focused) → syncEditingFlag(), so isEditingText is cleared
                    // too (restoring shortcuts such as Space for playback).
                    .onSubmit { focus.wrappedValue = false }
                    .onExitCommand { focus.wrappedValue = false }
                    .overlay { scrubSurface }
                UnitLabel(unit)
            }
            // Disabling dims the whole row and adds a one-line reason (§4 rule 5). `.disabled`
            // propagates through the environment, so the overlay's gesture dies with it — no
            // separate scrub guard is needed.
            .disabled(disabledReason != nil)
            .opacity(disabledReason != nil ? 0.5 : 1)
            if let text = disabledReason ?? caption {
                PanelCaption(text)
            }
        }
        .help(disabledReason
              ?? "Drag left/right to scrub · ⇧ faster · ⌥ finer · click to type")
    }

    /// The transparent scrub surface — while focused it turns hit testing off and gives text
    /// selection back.
    private var scrubSurface: some View {
        Color.clear
            .contentShape(Rectangle())
            .gesture(scrubGesture)
            // The left-right arrow cursor (the header's "Cursor feedback"). nil on a disabled row
            // — a field you cannot drag MUST NOT say it can be dragged (`.disabled` leaves hit
            // testing in place, so this gate is necessary). It is view wiring rather than a pure
            // decision, so it lies outside the automated line of defense.
            .pointerStyle(disabledReason == nil ? .columnResize : nil)
            // It must be outermost for hit testing of the whole subtree (including the gesture
            // and the pointer style) to be turned off.
            .allowsHitTesting(!focus.wrappedValue)
            .onDisappear {
                guard anchor != nil else { return }
                anchor = nil
                onScrubOrphaned()
            }
    }

    /// It separates tap from drag **within a single gesture**. Attaching a separate
    /// `.onTapGesture` makes SwiftUI's tap slop fire even on a short scrub, so focus is set
    /// mid-drag and at that instant `allowsHitTesting(false)` lifts the surface and the scrub is
    /// cut off.
    private var scrubGesture: some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { v in
                // Nothing happens before the engagement threshold (click protection).
                guard let live = engagedAnchor(v.translation) else { return }
                onScrub(target(live, translation: v.translation), false)
            }
            .onEnded { v in
                if let live = anchor {
                    anchor = nil
                    onScrub(target(live, translation: v.translation), true)
                } else if NumericScrub.isClick(translation: v.translation) {
                    focus.wrappedValue = true      // a click with no movement = entering typing
                }
            }
    }

    /// The engagement decision plus the basis latch. The reference is latched too — for Radius
    /// the reference is the current value, so using the live value would change the sensitivity
    /// mid-drag and introduce hysteresis.
    private func engagedAnchor(_ translation: CGSize) -> ScrubAnchor? {
        if let anchor { return anchor }
        guard NumericScrub.engaged(translationX: translation.width) else { return nil }
        let latched = ScrubAnchor(base: value, reference: reference)
        anchor = latched
        return latched
    }

    private func target(_ anchor: ScrubAnchor, translation: CGSize) -> Double {
        // The modifiers are read on every tick — they are a multiplier, not a mode switch (the
        // NumericScrub comment).
        let flags = NSEvent.modifierFlags
        return NumericScrub.value(base: anchor.base, translationX: translation.width,
                                  reference: anchor.reference,
                                  shift: flags.contains(.shift),
                                  option: flags.contains(.option),
                                  lowerBound: lowerBound)
    }
}
