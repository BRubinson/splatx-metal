// App finishing (2026-07-29 user requirement ③ — 2026-07-29 panel redesign spec footnote 5):
// the launch splash. It shows the program name (the brand lockup + TRAINER) and the company
// name (POSTMEDIA) for about 3 seconds, and a click or key press skips it immediately. It fades
// in and out.
//
// ── Why a separate borderless NSWindow (leaving the scene graph untouched is the real reason) ──
// ⑵ A main-window overlay (a ZStack inside the WindowGroup closure) cannot cover the title bar
//    or the StageNav toolbar, and since the WindowGroup body is evaluated **per window**, it
//    would reappear every time ⌘N or a viewer window is opened.
// ⑶ Adding a SwiftUI `Window` scene is technically possible on this macOS 26 target, but
//    (a) the **declaration order** of scenes can change which window launches by default and
//    what ⌘N targets, (b) scenes are subject to state restoration, so the splash could come back
//    on relaunch, and (c) the Window menu gets polluted. This app's window routing (AppDelegate
//    application(_:open:) → FileOpenRouter → openWindow(id:value:) → ViewerWindowFronter FIX 2)
//    was barely stabilized through measured multi-file opens, so the risk of adding a scene
//    outweighs the benefit.
// So applicationDidFinishLaunching in AppDelegate presents exactly one AppKit window — the scene
// declarations in `SplatXMetalTrainerApp` **do not change by a single character**.
//
// ── Window gate wiring (app finishing FIX 2 — the MainWindowGate.swift header is the
//    authoritative contract) ────────
// While the splash is up, the main window is taken off screen and then made to appear the moment
// the splash clears (a user request). There are only three wiring points in this file:
// • presentIfEnabled — `MainWindowGate.arm()` immediately after **all** the gates below pass and
//   before the splash window is created. If a gate blocks, there is no arming at all (a window
//   MUST never disappear on the test or environment-variable paths).
// • `reveal()` at the **start** of the fade in dismiss(fade:) — the normal path (a cross-fade).
// • `reveal()` in finish() — the backstop (the willClose and didResignActive direct paths). It
//   is idempotent, so it is safe.
// Two meanings change during the gated interval:
// ⓐ The first key press: the main window is off screen, so "it goes to the main window" does not
//   literally hold, but the local monitor's dismiss → reveal **runs synchronously before** the
//   event is returned, so makeKeyAndOrderFront brings the key window back and only then is the
//   event delivered — the contract that the first key reaches the main window is preserved.
// ⓑ Clicks (settled in app finishing FIX 3): during the gated interval the splash **absorbs**
//   the mouse (ignoresMouseEvents = false — see the reversal history in the FIX 1 ① item below).
//   A click inside the splash rectangle is delivered to our app, so the local monitor sees it and
//   the skip fires, and the app stays active, so the window appears **in front** via reveal's
//   makeKeyAndOrderFront path.
//   A click **outside** the rectangle (on the desktop) deactivating the app is **intended, normal
//   behavior** — a desktop click expresses the intent to leave the app, and the didResignActive
//   teardown (FIX 1 ②) already handles it correctly (the splash clears, and the window appears
//   behind, per the inactive-app convention). The alternative of absorbing the whole screen with
//   a full-screen transparent backdrop is **forbidden** — .floating plus full screen plus event
//   absorption becomes a hostile window that intercepts even other apps' clicks for 3 seconds.
//
// ── Rules that keep the main window undisturbed (all intentional — do not "fix" them) ──────
// • **Never call `NSApp.activate()`.** The splash is shown with `orderFrontRegardless()` alone.
//   Calling it would steal focus from whatever other app the user was using.
// • **borderless defaults to `canBecomeKey == false`, and that is not overridden.**
//   That is what sends the first key press to the main window (it does not intercept keyboard
//   focus for 3 seconds — since FIX 2, the path by which this holds is window gate ⓐ above).
// • **`ignoresMouseEvents` has two stages — false while armed (absorb), true once the window is
//   back (pass through)** (app finishing FIX 3 — the reversal history of FIX 1 ①):
//   ⑴ Why it was true (FIX 1 ①): at the time, **the main window was underneath the splash**.
//      Without pass-through, mouseDown was delivered to the splash contentView and then
//      discarded, so clicking Import or a recent project on the Home screen became "I clicked
//      and nothing happened" — pass-through delivered the click to the window below.
//   ⑵ What broke that premise (FIX 2): the window gate **took that window below off screen**.
//      The destination of the pass-through became the desktop (Finder) instead of our window —
//      so the click was delivered to another app, the local monitor never saw it (the skip did
//      not fire), the app was deactivated, and the didResignActive teardown fired instead, with
//      the window appearing behind. Zero benefit, only loss.
//   ⑶ Why it became false (FIX 3): absorbing while armed keeps a click inside the splash in our
//      app, restoring both the skip firing and the app staying active (= reveal's
//      makeKeyAndOrderFront path).
//      But it is **not unconditionally false** — it is restored to true on the line right after
//      `MainWindowGate.reveal()` in dismiss/finish. After the window returns it sits below the
//      splash again and the premise of ⑴ revives, so a second click during the 0.15–0.35s fade
//      MUST pass through (that interval is exactly where the original FIX 1 ① reproduction
//      happened). Invariant: **absorb ⟺ no window of ours below (while armed); pass through ⟺
//      after the window returns** — which is why the flag transition is placed at the same spot
//      as the reveal call.
//      arm and showing the splash happen as a pair on the single presentIfEnabled path (if a
//      gate blocks, neither happens) — if you make a change that breaks this pairing, re-derive
//      this invariant first.
// • The local event monitor **does not swallow** events (it always returns them unchanged) — the
//   skipping click or key also performs its main-window action. If it swallowed them, the first
//   interaction would silently vanish.
// • **Keep `level = .floating`, but tear it down immediately when the app deactivates** (app
//   finishing FIX 1 ②).
//   .floating is a global level that crosses process boundaries, so it floats above other apps'
//   windows even when the app is inactive, while the skip mechanism (the local monitor) is
//   app-local and cannot dismiss it while another app is in use — so it is taken down without a
//   fade in `didResignActiveNotification` (a ⌘Tab switch = no intent to watch the splash).
//   The alternative of lowering the level to `.normal` is forbidden — the SwiftUI main window
//   created later could cover it and the splash might not be visible at all.
// • `isExcludedFromWindowsMenu` + `[.transient, .ignoresCycle, .fullScreenNone]` — it drops out
//   of the Window menu, Mission Control, and the ⌘` window cycle.
// • **Defense against a reversed termination order**: because
//   `applicationShouldTerminateAfterLastWindowClosed` is true, if the user closes the main
//   window within those 3 seconds, a state can arise where the only "remaining window" is the
//   splash. A borderless window has `canBecomeMain == false`, so dropping out of AppKit's
//   last-window determination is the normal behavior, but this code does not rely on that
//   determination — on another window's `willCloseNotification` it clears the splash
//   **immediately (with no fade)** so that it does not exist at the moment of the determination.
//   And it is taken down with `orderOut(nil)` rather than `close()` (so the splash itself does
//   not trigger the last-window determination).
//   The observers are removed in **finish (after the window has actually gone down)**, not in
//   dismiss (the start of the fade) (app finishing FIX 1 ④ — the hole where, if the main window
//   closed during the 0.15–0.35s fade, the defense did not fire and a windowless zombie process
//   was left behind. willClose and didResignActive stay alive during the fade and call finish
//   directly to take it down immediately).
// • **No blocking**: making those 3 seconds with `Thread.sleep` or a nested run loop
//   (`runUntilDate`) blocks the main thread, serializing app startup delay and eating into test
//   timeouts. Use only a timer plus the animation completion callback.
//
// ── Position (app finishing round 6 Task 3) ────────────────────────────────
// It is the **exact center** of the screen — not `window.center()` (center() is biased with a
// 1:3 top-to-bottom margin ratio). The arithmetic is authoritative in the comment on the pure
// function `origin(forSize:in:)` (+ polish5 spec footnote 5), and the screen choice is
// `targetScreen()` (the pointer's screen → NSScreen.main → screens.first). None of the rules
// above are touched.
//
// ── Test avoidance (measured evidence) ─────────────────────────────────────
// `xcodebuild test` actually launches the app (measured: it creates on-screen windows). To keep
// the splash from flashing on the developer's screen for 3 seconds and disturbing focus, it is
// not created at all in a test environment.
// The decision is the pure function `shouldPresent(environment:)` — this app's automated line of
// defense, and `SplashWindowTests` asserts it against the **current process environment** as
// well, so the gate proves itself.
// ⚠ On this machine `XCTestConfigurationFilePath` was measured to have **the key present with an
//   empty-string value** (Xcode 26 / macOS 26.5). So the decision looks only at **key
//   existence** — "improving" it with an `isEmpty` check or by using the path after an optional
//   binding brings the splash back during the gate.
import AppKit
import SwiftUI

@MainActor
final class SplashWindow {
    /// Exposure timing (3.0 s total). Skipping uses a short fade to feel immediate.
    enum Timing {
        static let fadeIn: TimeInterval = 0.25
        static let hold: TimeInterval = 2.40
        static let fadeOut: TimeInterval = 0.35
        static let skip: TimeInterval = 0.15
        static var total: TimeInterval { fadeIn + hold + fadeOut }
    }

    /// The brand tagline — the text nodes of design/banner/github-banner.svg and
    /// design/social/github-social-preview.svg are the single source of truth (do not write a
    /// new sentence).
    static let tagline = "3D Gaussian Splatting training, native on Apple Silicon"

    /// Whether to present (pure — the source of truth for tests). The three environment
    /// variables are injected by the XCTest harness, and `SPLATX_NO_SPLASH` is the escape hatch
    /// a person uses (this app has no settings screen, and a Settings scene is not introduced
    /// for the sake of one toggle — three escape hatches are enough: ⑴ click/key instant skip,
    /// ⑵ this environment variable, and ⑶ the automatic test skip).
    nonisolated static func shouldPresent(environment: [String: String]) -> Bool {
        if environment["SPLATX_NO_SPLASH"] != nil { return false }
        // Look only at **key existence**, not the value (see the ⚠ in the header).
        for key in ["XCTestSessionIdentifier", "XCTestBundlePath",
                    "XCTestConfigurationFilePath"] where environment[key] != nil {
            return false
        }
        return true
    }

    /// The **exact center** origin of the screen (pure — the source of truth for tests.
    /// Precedent: OverlayPlacement.rect — separating placement arithmetic so coordinates can be
    /// asserted with an accuracy. App finishing round 6 Task 3, polish5 spec footnote 5).
    /// Why `window.center()` is not used: per the AppKit documentation, center() is "somewhat
    /// above center vertically" — measured on this machine the top-to-bottom margin ratio is 1:3
    /// (at 720×420, 183pt above the true center), which is not the "exact center" the user
    /// requested.
    /// - **Relative to visibleFrame**: it already excludes the Dock, the menu bar, and the notch
    ///   area (measured 7pt difference against frame). The documentation forbids caching it, so
    ///   it is read at call time and passed in.
    /// - **Clamping**: the "if the window is larger than the screen, fall back to top alignment"
    ///   defense of center() is preserved in this direct computation too — vertically the top
    ///   wins (min comes last) and horizontally the left wins (max comes last), preventing the
    ///   window from going off screen. A low-resolution external display is a real scenario.
    /// - **floor pixel alignment**: the exact center frequently lands on .5 (418.5, for
    ///   example), and a non-integer origin blurs the wordmark and the 1pt hairline
    ///   (SplashContent) on a 1x display — floor, matching center()'s measured behavior.
    nonisolated static func origin(forSize size: NSSize, in visibleFrame: NSRect) -> NSPoint {
        let x = max(min(visibleFrame.midX - size.width / 2,
                        visibleFrame.maxX - size.width),
                    visibleFrame.minX)
        let y = min(max(visibleFrame.midY - size.height / 2,
                        visibleFrame.minY),
                    visibleFrame.maxY - size.height)
        return NSPoint(x: x.rounded(.down), y: y.rounded(.down))
    }

    /// The screen to present the splash on — the fallback chain **the mouse pointer's screen →
    /// NSScreen.main → screens.first** (app finishing round 6 Task 3 decision). At the moment
    /// present() runs, MainWindowGate.arm() has already taken the main window down, so our app
    /// has no key window — per the documentation NSScreen.main is "the screen of the window with
    /// keyboard focus", so it can point at **another app's screen** at this point. The place
    /// where the launch action just happened (a Dock click or a Finder double-click) = the
    /// pointer's screen is the best proxy for where the user is looking, so it comes first; main
    /// is its fallback (for the instant of a configuration change where the pointer is on no
    /// screen), and screens.first is the last resort. Never force-unwrap `NSScreen.main!` — nil
    /// is genuinely possible when headless or during a display reconfiguration.
    private static func targetScreen() -> NSScreen? {
        let mouse = NSEvent.mouseLocation
        if let hit = NSScreen.screens.first(where: { NSMouseInRect(mouse, $0.frame, false) }) {
            return hit
        }
        return NSScreen.main ?? NSScreen.screens.first
    }

    /// The live splash (prevents duplicate presentation and holds the strong reference).
    private static var current: SplashWindow?

    /// Called once at app start. When blocked by a test or an environment variable it **does not
    /// even create** the window.
    static func presentIfEnabled(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) {
        guard current == nil else { return }
        guard shouldPresent(environment: environment) else { return }
        // Second line of defense: the XCTest bundle is loaded before main via
        // DYLD_INSERT_LIBRARIES, so this check survives even if the environment variable names
        // change.
        guard NSClassFromString("XCTestCase") == nil else { return }
        // App finishing FIX 2: arm only immediately after every gate passes and **before** the
        // splash window is created (header, "Window gate wiring") — a window MUST never go down
        // on a path where a gate blocked.
        MainWindowGate.arm()
        let splash = SplashWindow()
        current = splash
        splash.present()
    }

    private let window: NSWindow
    private var monitor: Any?
    private var closeObserver: NSObjectProtocol?
    private var resignObserver: NSObjectProtocol?
    private var autoDismiss: DispatchWorkItem?
    /// The fade has started (the input contract is released). The window may still be on screen.
    private var dismissed = false
    /// The window has actually gone down (every observer removed too) — the two-stage lifetime
    /// of FIX 1 ④.
    private var finished = false

    private init() {
        let size = NSSize(width: 720, height: 420)
        window = NSWindow(contentRect: NSRect(origin: .zero, size: size),
                          styleMask: .borderless, backing: .buffered, defer: false)
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = true
        window.level = .floating           // .normal is forbidden — header FIX 1 ② rule
        // FIX 3: absorb while armed (false) — with no window of ours below, a pass-through
        // leaking to the desktop means the skip does not fire and the app deactivates.
        // Pass-through (true) is restored right after reveal in dismiss/finish — the header
        // invariant "ignoresMouseEvents has two stages" is the authoritative contract.
        window.ignoresMouseEvents = false
        window.isExcludedFromWindowsMenu = true
        window.collectionBehavior = [.transient, .ignoresCycle, .fullScreenNone]
        window.isReleasedWhenClosed = false
        window.appearance = NSAppearance(named: .darkAqua)

        let host = NSHostingView(rootView: SplashContent(
            wordmark: Self.wordmark, version: Self.versionText))
        host.frame = NSRect(origin: .zero, size: size)
        host.wantsLayer = true
        host.layer?.cornerRadius = 18
        host.layer?.masksToBounds = true
        window.contentView = host
    }

    private func present() {
        // App finishing FIX 2: identify the splash itself so the gate's eligibility check
        // excludes it (a second line of defense — being borderless, it already drops out on the
        // styleMask check alone). This comes before orderFrontRegardless.
        MainWindowGate.registerSplash(window)
        window.alphaValue = 0
        // App finishing round 6 Task 3: the true exact center, not center() — the rationale is
        // authoritative in the origin(forSize:in:) comment. Only in the extreme case where there
        // is no screen information at all does it delegate to the AppKit default (center).
        if let screen = Self.targetScreen() {
            window.setFrameOrigin(Self.origin(forSize: window.frame.size,
                                              in: screen.visibleFrame))
        } else {
            window.center()
        }
        window.orderFrontRegardless()   // not activate() — header rule
        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = Timing.fadeIn
            ctx.timingFunction = CAMediaTimingFunction(name: .easeOut)
            window.animator().alphaValue = 1
        }

        let work = DispatchWorkItem { [weak self] in self?.dismiss(fade: Timing.fadeOut) }
        autoDismiss = work
        DispatchQueue.main.asyncAfter(deadline: .now() + Timing.fadeIn + Timing.hold,
                                      execute: work)

        // Skip — the event is returned unchanged (it is not swallowed; header rule).
        // FIX 3: while armed the splash absorbs, so a click inside the rectangle is delivered to
        // our app and the monitor sees it (the skip fires and the app stays active). The
        // dismiss → reveal inside the monitor callback runs synchronously before the event is
        // returned, so a key press is delivered to the restored key window (contract ⓐ) and a
        // click is harmlessly consumed by the splash contentView (during the interval when there
        // was nothing below to press).
        monitor = NSEvent.addLocalMonitorForEvents(
            matching: [.leftMouseDown, .rightMouseDown, .keyDown]
        ) { [weak self] event in
            self?.dismiss(fade: Timing.skip)
            return event
        }

        // Defense against a reversed termination order (header). When another window closes it
        // goes down immediately without waiting for the fade — going straight to finish() rather
        // than dismiss(fade: 0) is what also catches a window close **during** the fade (dismiss
        // is blocked by the dismissed guard — FIX 1 ④).
        closeObserver = NotificationCenter.default.addObserver(
            forName: NSWindow.willCloseNotification, object: nil, queue: .main
        ) { [weak self] note in
            MainActor.assumeIsolated {
                guard let self, (note.object as? NSWindow) !== self.window else { return }
                self.finish()
            }
        }

        // App deactivation (⌘Tab and the like) = immediate teardown — header FIX 1 ② rule.
        // .floating is a global level so it floats above other apps, while the only skip
        // mechanism is app-local. This also goes straight to finish().
        resignObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.didResignActiveNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.finish() }
        }
    }

    /// Whichever path it is entered from, it runs only once (auto-expiry, a click, and a key
    /// press can race).
    /// Here it tears down **only the input contract (the monitor and the auto-expiry)** — the
    /// willClose and didResignActive observers are kept until finish, when the window actually
    /// goes down (FIX 1 ④: the defense hole during the fade interval).
    private func dismiss(fade: TimeInterval) {
        guard !dismissed else { return }
        dismissed = true
        // App finishing FIX 2: the main window returns at the **start** of the fade — the main
        // window (.normal) is laid underneath the splash (.floating) while the splash dissolves
        // above it, producing a cross-fade. Calling it only in finish would show nothing but the
        // desktop for the 0.35s fade (header, "Window gate wiring").
        MainWindowGate.reveal()
        // FIX 3: pass-through is restored from the moment the window is back below — a second
        // click during the fade must reach the returned window (header invariant: absorb ⟺ no
        // window, pass through ⟺ window returned).
        window.ignoresMouseEvents = true
        autoDismiss?.cancel()
        autoDismiss = nil
        if let monitor { NSEvent.removeMonitor(monitor) }
        monitor = nil
        guard fade > 0 else { finish(); return }
        NSAnimationContext.runAnimationGroup({ ctx in
            ctx.duration = fade
            ctx.timingFunction = CAMediaTimingFunction(name: .easeIn)
            window.animator().alphaValue = 0
        }, completionHandler: {
            MainActor.assumeIsolated { self.finish() }
        })
    }

    /// The actual termination point — **idempotent**. `orderOut(nil)` rather than `close()`
    /// (header, "Defense against a reversed termination order"). If willClose or
    /// didResignActive calls it directly during the fade, it goes down immediately without
    /// waiting for the fade — the remaining animation completion callback is blocked by the
    /// finished guard and does nothing.
    private func finish() {
        guard !finished else { return }
        finished = true
        dismissed = true
        // App finishing FIX 2 backstop: it guarantees the window returns even on the paths where
        // willClose and didResignActive skip the fade and go straight here — reveal is
        // idempotent, so overlapping with the dismiss path is harmless.
        MainWindowGate.reveal()
        // FIX 3: the pass-through restore backstop — it has no practical effect since orderOut
        // follows immediately, but it keeps the header invariant (pass through ⟺ after the
        // window returns) whole on every path.
        window.ignoresMouseEvents = true
        autoDismiss?.cancel()
        autoDismiss = nil
        if let monitor { NSEvent.removeMonitor(monitor) }
        monitor = nil
        if let closeObserver { NotificationCenter.default.removeObserver(closeObserver) }
        closeObserver = nil
        if let resignObserver { NotificationCenter.default.removeObserver(resignObserver) }
        resignObserver = nil
        window.orderOut(nil)
        Self.current = nil
    }

    /// The brand lockup — it reuses the **same bundle resource** as the watermark seed (no new
    /// asset). The original is the repo's `design/wordmark/splatx-metal-lockup-dark@2x.png`, and
    /// this uses exactly what project.yml wired into the resources phase by relative path — the
    /// name is authoritative in `WatermarkSeeder.bundleResourceName`. Reading bundle resources
    /// works fine under the sandbox too (reading a home or external path fails silently, so do
    /// not stray outside it).
    private static var wordmark: NSImage? {
        WatermarkSeeder.bundleImageURL.flatMap { NSImage(contentsOf: $0) }
    }

    /// `1.0.0 (1)` — derived from MARKETING_VERSION/CURRENT_PROJECT_VERSION in project.yml.
    private static var versionText: String? {
        let info = Bundle.main.infoDictionary
        guard let short = info?["CFBundleShortVersionString"] as? String,
              !short.isEmpty else { return nil }
        if let build = info?["CFBundleVersion"] as? String, !build.isEmpty, build != short {
            return "\(short) (\(build))"
        }
        return short
    }
}

// MARK: - Visual composition
//
// The color language follows **the lockup's own** — the app UI accent is teal (#2dd4bf) while
// the lockup's METAL is a green gradient (#0B7A47→#7CE04A), so mixing teal into the splash would
// set the two greens fighting.
// The background radial gradient (#14181E → #0B0E12 → #060809, cx .5 cy .3 r 1.1) and the green
// glow (#1E9A55 @ .26) are transplanted verbatim from the values in
// design/banner/github-banner.svg (no new asset).
// The company name spelling follows `POSTMEDIA` (BrandFooter — .caption2 + tracking 1.5 +
// textSecondary). Public-pipeline Task 5 unified the legal notices (NOTICE and
// NSHumanReadableCopyright) to POSTMEDIA as well — the previous dual spelling `PoSTMEDIA` read
// as a typo from the outside and was removed.

private struct SplashContent: View {
    let wordmark: NSImage?
    let version: String?
    @State private var appeared = false

    var body: some View {
        ZStack {
            RadialGradient(
                gradient: Gradient(stops: [
                    .init(color: Color(hex: 0x14181E), location: 0),
                    .init(color: Color(hex: 0x0B0E12), location: 0.6),
                    .init(color: Color(hex: 0x060809), location: 1),
                ]),
                center: UnitPoint(x: 0.5, y: 0.3),
                startRadius: 0, endRadius: 790)

            RadialGradient(
                gradient: Gradient(colors: [Color(hex: 0x1E9A55).opacity(0.26), .clear]),
                center: UnitPoint(x: 0.5, y: 0.34),
                startRadius: 0, endRadius: 300)

            VStack(spacing: 0) {
                lockup
                Spacer().frame(height: 28)
                Text(SplashWindow.tagline)
                    .font(.system(size: 13))
                    .foregroundStyle(Color(hex: 0x7B8794))
                Spacer().frame(height: 30)
                Rectangle()
                    .fill(SplatXTheme.hairline)
                    .frame(width: 180, height: 1)
                Spacer().frame(height: 20)
                Text("POSTMEDIA")
                    .font(.caption2)
                    .tracking(1.5)
                    .foregroundStyle(SplatXTheme.textSecondary)
            }
            .scaleEffect(appeared ? 1 : 0.98)
            .opacity(appeared ? 1 : 0)

            if let version {
                VStack {
                    Spacer()
                    HStack {
                        Spacer()
                        Text(version)
                            .font(.caption2).monospacedDigit()
                            .foregroundStyle(SplatXTheme.textSecondary.opacity(0.6))
                    }
                }
                .padding(14)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .overlay(RoundedRectangle(cornerRadius: 18)
            .stroke(Color.white.opacity(0.06), lineWidth: 1))
        .onAppear { withAnimation(.easeOut(duration: 0.45)) { appeared = true } }
    }

    /// The lockup PNG (4320×816 → 520pt wide) + TRAINER. Together they form the app's display
    /// name "SplatX Metal Trainer" (project.yml PRODUCT_NAME). If the bundle resource is
    /// missing, text substitutes for it — the program name must be visible in every case.
    private var lockup: some View {
        VStack(spacing: 12) {
            if let wordmark {
                Image(nsImage: wordmark)
                    .resizable()
                    .interpolation(.high)
                    .scaledToFit()
                    .frame(width: 520)
            } else {
                Text("SplatX Metal")
                    .font(.system(size: 42, weight: .medium))
                    .tracking(4)
                    .foregroundStyle(SplatXTheme.textPrimary)
            }
            Text("TRAINER")
                .font(.system(size: 11, weight: .semibold))
                .tracking(6)
                .foregroundStyle(SplatXTheme.textSecondary)
        }
    }
}
