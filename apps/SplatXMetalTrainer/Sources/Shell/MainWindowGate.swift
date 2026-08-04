// App finishing FIX 2 (2026-07-29): the main window appearing the moment the splash ends — the
// window gate.
// The user's request: "I'd like the main window to appear at the moment the splash screen goes
// away."
//
// Why it is needed: the WindowGroup main window comes up immediately with app startup, and the
// splash (a .floating borderless 720×420) is **merely layered on top of it**, so the main window's
// chrome and content are already fully visible around the splash and there is no impression of
// "appearing" when the splash clears. So while the splash is up, eligible windows are taken off
// screen with orderOut and restored the moment the splash clears.
//
// ── The two arm paths — both are required ──────────────────────────────────────────
// **There is no guarantee whether SwiftUI creates and shows its scene windows before or after**
// applicationDidFinishLaunching (= showing the splash = arm time) (GUI measurement is impossible
// on this machine due to TCC).
// Doing only one of them makes it silently do nothing, depending on timing.
//   ⑴ At arm time, sweep NSApp.windows, take down the already visible eligible windows, and record
//      them in order.
//   ⑵ Subscribe to eligible windows that **become newly visible** while armed. AppKit has no
//      single public "a window became visible" notification (nothing like
//      NSWindowDidBecomeVisibleNotification exists in the SDK — verified by grep).
//      So a combination of three builds the same thing out of public APIs:
//        • didBecomeKey / didBecomeMain — delivered **synchronously inside**
//          makeKeyAndOrderFront, so the orderOut happens before the next screen composition and
//          there is no first-frame flicker. Both the SwiftUI scene windows and
//          ViewerWindowFronter take the makeKeyAndOrderFront path, so this is the primary signal.
//        • didChangeOcclusionState — an **asynchronous** catch-all from the window server. It
//          catches windows that are only orderFront-ed without becoming key or main (restored and
//          auxiliary windows). Only this path can be exposed for a frame or two.
//      Capture deduplicates the record, so overlapping signals record only once. An occlusion
//      notification triggered by our own orderOut is ignored by the isVisible == false guard (no
//      ping-pong).
//
// ── The rules of reveal ─────────────────────────────────────────────────────
// • **Idempotent**. It is safe even when all three call sites overlap:
//   ⑴ The **start** of the fade in SplashWindow.dismiss — the normal path. The main window is laid
//      **under** the splash (.floating > .normal — the order is guaranteed) while the splash
//      dissolves above it, producing a cross-fade. Calling it only in finish leaves an interval of
//      0.35s where only the desktop is visible.
//   ⑵ SplashWindow.finish — the backstop (the path where willClose and didResignActive skip the
//      fade and go straight there).
//   ⑶ The hard deadline timer below.
// • When active (NSApp.isActive), makeKeyAndOrderFront is applied to the last recorded window (the
//   window most recently trying to become visible = the most recent intent) and orderFront to the
//   rest. **When inactive, orderFront only** — `orderFrontRegardless` is forbidden (the user has
//   gone to another app with ⌘Tab, and it would push our window above theirs). An inactive app's
//   orderFront does not rise above another app's windows and only surfaces at the moment the app
//   becomes active — it looks like a delay, but it is **the intended result**.
// • A cold-launch Finder file open (application(_:open:) → FileOpenRouter → an independent viewer
//   window) also creates a window during the armed interval — with no special exception, it goes
//   down and comes back together with the others. That path calls NSApp.activate(), so the active
//   check is normally true at reveal time, and in that case the last recorded window (the viewer —
//   it becomes visible after the main window) becomes key. Consistent with FileOpenRouter FIX 2's
//   intent of "the new viewer window is frontmost and key".
// • If a hidden window closes while armed (willClose), it is removed from the record — preventing
//   reveal from resurrecting a closed window. Every reference is weak, so deallocated windows also
//   drop out naturally.
//
// ── The hard deadline — the structural defense of the "no windows" invariant ─────────────────────
// At arm time, a timer is set that unconditionally reveals after Timing.total + 1.0 seconds. It is
// **independent** of the splash lifetime logic — the only device that structurally prevents a bug
// or exception on the splash side from leaving the app with no windows. orderOut is not close and
// does not trigger applicationShouldTerminateAfterLastWindowClosed (true in this app), so the
// failure mode is not termination but "an app with no windows, indistinguishable from a hang to
// the user".
// On the normal path reveal has already torn it down, so it does nothing.
import AppKit

@MainActor
enum MainWindowGate {
    /// The eligibility decision (pure — the source of truth for tests): excluding the splash window
    /// itself, and only real titled windows.
    /// Borderless windows, panels, and system auxiliary windows (menus and tooltips are borderless
    /// too) are left untouched.
    /// The splash is borderless, so it already drops out on the styleMask check alone, but the
    /// identity exclusion is kept as a second line of defense so the gate does not swallow the
    /// splash even if the splash's style changes.
    static func isEligible(_ window: NSWindow, splash: NSWindow?) -> Bool {
        window !== splash && window.styleMask.contains(.titled)
    }

    /// The armed state (read-only — the source of truth for state machine tests). While armed, the
    /// splash absorbs the mouse (app finishing FIX 3 — the rationale is authoritative in the
    /// SplashWindow header, "ignoresMouseEvents has two stages"). The absorb→pass-through
    /// transition does not branch on this value at runtime; it is physically attached at the
    /// reveal call site — had it been a branch, it could not be verified without actually
    /// presenting a window.
    static var isArmed: Bool { armed }

    private struct WeakWindow { weak var window: NSWindow? }

    private static var armed = false
    /// The windows taken down (recording order is preserved — reveal restores them in the same
    /// order). Weak references, no duplicates.
    private static var hidden: [WeakWindow] = []
    private static var observers: [NSObjectProtocol] = []
    private static var deadline: DispatchWorkItem?
    private static weak var splashWindow: NSWindow?

    /// The splash registers its own window so it is excluded by identity from the eligibility
    /// check (see the header).
    /// arm is called **before** the splash window is created, so SplashWindow.present does the
    /// registration.
    static func registerSplash(_ window: NSWindow) { splashWindow = window }

    /// Arming — the header's "The two arm paths". It is called only on the path where the splash
    /// actually appears (immediately after every gate in SplashWindow.presentIfEnabled passes — a
    /// window MUST never go down on the test or environment-variable paths).
    static func arm() {
        guard !armed else { return }
        armed = true

        // Path ⑴: sweep the already visible eligible windows (NSApp.windows = creation order).
        for window in NSApp.windows where window.isVisible && isEligible(window, splash: splashWindow) {
            capture(window)
        }

        // Path ⑵: windows that become newly visible while armed — two synchronous signals plus one
        // asynchronous catch-all (see the header).
        let center = NotificationCenter.default
        for name in [NSWindow.didBecomeKeyNotification,
                     NSWindow.didBecomeMainNotification,
                     NSWindow.didChangeOcclusionStateNotification] {
            observers.append(center.addObserver(forName: name, object: nil, queue: .main) { note in
                MainActor.assumeIsolated {
                    guard armed, let window = note.object as? NSWindow,
                          window.isVisible, isEligible(window, splash: splashWindow) else { return }
                    capture(window)
                }
            })
        }
        // When a hidden window closes, remove it from the record — so reveal does not resurrect a
        // closed window (see the header).
        observers.append(center.addObserver(
            forName: NSWindow.willCloseNotification, object: nil, queue: .main
        ) { note in
            MainActor.assumeIsolated {
                guard let window = note.object as? NSWindow else { return }
                hidden.removeAll { $0.window === window }
            }
        })

        // The hard deadline (see the header) — independent of the splash lifetime logic. It does
        // nothing on the normal path.
        let work = DispatchWorkItem { reveal() }
        deadline = work
        DispatchQueue.main.asyncAfter(deadline: .now() + SplashWindow.Timing.total + 1.0,
                                      execute: work)
    }

    /// Reveal — **idempotent** (calling it before arming and calling it twice are both harmless).
    /// It tears down the observers, the timer, and the record, then restores the windows in the
    /// recorded order. Removing the observers comes before the restoration — reveal's
    /// makeKeyAndOrderFront MUST NOT be caught again by our own capture observation.
    static func reveal() {
        guard armed else { return }
        armed = false
        for observer in observers { NotificationCenter.default.removeObserver(observer) }
        observers.removeAll()
        deadline?.cancel()
        deadline = nil
        let windows = hidden.compactMap(\.window)
        hidden.removeAll()
        let active = NSApp.isActive
        for (index, window) in windows.enumerated() {
            if active && index == windows.count - 1 {
                window.makeKeyAndOrderFront(nil)
            } else {
                // When inactive, everything takes this path — orderFrontRegardless is forbidden
                // (the header's "The rules of reveal").
                window.orderFront(nil)
            }
        }
    }

    /// Takes the window down and records it (no duplicate records — the two notification signals
    /// can overlap on the same window).
    private static func capture(_ window: NSWindow) {
        if !hidden.contains(where: { $0.window === window }) {
            hidden.append(WeakWindow(window: window))
        }
        window.orderOut(nil)
    }
}
