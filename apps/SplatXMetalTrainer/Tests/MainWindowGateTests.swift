// App finishing FIX 2: verifies the **pure decision and idempotence contract** of the window
// gate (MainWindowGate).
// NSWindow "behavior" (z-order, key assignment, the notification capture path) and SwiftUI are
// manual under this repository's policy (its app-work discipline — the manual checklist is an
// internal record).
//
// The arm→reveal round trip is automated, however — in the host app process, `isVisible` (being
// listed in the window list) after orderFront/orderOut is deterministic regardless of screen
// compositing or whether the app is active, and the existing gate already runs with the app
// actually shown (measured — see "test avoidance" in the SplashWindow header). The
// notification-based capture path (didChangeOcclusionState), by contrast, is asynchronous and
// originates in the window server, which would introduce run-loop pumping and timing
// dependence, so it is not automated (manual ⒅). If the round trip's premise (isVisible after
// orderFront) does not hold on a given host, the reason is left as a skip rather than a FAIL —
// the decision and idempotence cases never show a window, so they run on any host.
import XCTest
import AppKit
@testable import SplatXMetalTrainer

@MainActor
final class MainWindowGateTests: XCTestCase {
    private func makeWindow(styleMask: NSWindow.StyleMask) -> NSWindow {
        let w = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 220, height: 120),
                         styleMask: styleMask, backing: .buffered, defer: false)
        w.isReleasedWhenClosed = false
        return w
    }

    // A real titled window = eligible.
    func testTitledWindowIsEligible() {
        let w = makeWindow(styleMask: [.titled, .closable])
        XCTAssertTrue(MainWindowGate.isEligible(w, splash: nil))
    }

    // borderless (the shape of the splash, menus, tooltips, and system helper windows) = not
    // eligible.
    func testBorderlessWindowIsNotEligible() {
        let w = makeWindow(styleMask: .borderless)
        XCTAssertFalse(MainWindowGate.isEligible(w, splash: nil))
    }

    // The splash window itself is excluded from identification — even when titled (even if
    // its style changes) it is not swallowed.
    func testSplashWindowItselfIsNotEligible() {
        let splash = makeWindow(styleMask: .titled)
        XCTAssertFalse(MainWindowGate.isEligible(splash, splash: splash))
        // Other windows stay eligible regardless of the splash registration.
        let other = makeWindow(styleMask: .titled)
        XCTAssertTrue(MainWindowGate.isEligible(other, splash: splash))
    }

    // reveal while not armed is harmless (the idempotence guard) — no crash and no state
    // pollution.
    func testRevealWithoutArmIsHarmless() {
        MainWindowGate.reveal()
        MainWindowGate.reveal()
    }

    // The state machine (FIX 3): isArmed goes true on arm and false on reveal, and a second
    // reveal does nothing.
    // The splash's mouse absorption while armed (the two-stage ignoresMouseEvents) is paired
    // with this state — flipping the flag itself is window behavior and therefore manual (the
    // checklist), while the state transitions are pinned here.
    func testIsArmedTracksArmRevealLifecycle() {
        XCTAssertFalse(MainWindowGate.isArmed)
        MainWindowGate.arm()
        XCTAssertTrue(MainWindowGate.isArmed)
        MainWindowGate.arm()   // arming twice is harmless
        XCTAssertTrue(MainWindowGate.isArmed)
        MainWindowGate.reveal()
        XCTAssertFalse(MainWindowGate.isArmed)
        MainWindowGate.reveal()   // idempotent
        XCTAssertFalse(MainWindowGate.isArmed)
    }

    // The arm → reveal round trip: an eligible window that was visible goes down on arm and
    // comes back on reveal.
    // reveal is idempotent: the second call does nothing (the window state is unchanged).
    func testArmRevealRoundTripRestoresVisibility() throws {
        let w = makeWindow(styleMask: [.titled, .closable])
        addTeardownBlock { @MainActor in
            MainWindowGate.reveal()   // restore the gate and the host window even if an assertion aborts (the RAII intent)
            w.orderOut(nil)
        }
        w.orderFront(nil)
        // The round trip's premise — if orderFront does not list the window on this host, the
        // round trip cannot be established at all.
        try XCTSkipUnless(w.isVisible, "isVisible does not come up after orderFront on this host — round-trip verification impossible in this environment (covered by the manual checklist instead)")

        MainWindowGate.arm()
        XCTAssertFalse(w.isVisible, "arming must take down a visible eligible window")

        MainWindowGate.reveal()
        XCTAssertTrue(w.isVisible, "reveal must bring back the recorded window")

        MainWindowGate.reveal()   // idempotent — the second call does nothing
        XCTAssertTrue(w.isVisible)
    }

    // A window that closes while armed drops out of the record — reveal must not revive a
    // closed window.
    // A real close() is not used: arm has already taken down every other window of the host
    // app, so a genuine close could trip the "last window closed" termination decision (this
    // app's delegate returns true) — instead a synthetic notification is posted, aimed only at
    // the gate's willClose subscription (in the tests there is no other subscriber to this
    // notification — SplashWindow subscribes only inside present(), and present is blocked by
    // the XCTest gate).
    func testClosingWindowIsDroppedFromRevealRecord() throws {
        let w = makeWindow(styleMask: [.titled, .closable])
        addTeardownBlock { @MainActor in
            MainWindowGate.reveal()
            w.orderOut(nil)
        }
        w.orderFront(nil)
        try XCTSkipUnless(w.isVisible, "isVisible does not come up after orderFront on this host — round-trip verification impossible in this environment (covered by the manual checklist instead)")

        MainWindowGate.arm()
        XCTAssertFalse(w.isVisible)
        NotificationCenter.default.post(name: NSWindow.willCloseNotification, object: w)

        MainWindowGate.reveal()
        XCTAssertFalse(w.isVisible, "a closing window must drop out of the reveal record")
    }
}
