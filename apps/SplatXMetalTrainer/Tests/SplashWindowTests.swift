// App finishing (2026-07-29 requirement ③): verifies only the **pure decision and the
// constants** of the launch splash.
// Not testing NSWindow or SwiftUI views is the explicit policy of this repository, so the
// automated line of defense is exactly two things: `SplashWindow.shouldPresent` and the timing
// constants.
//
// The heart of this suite is `testCurrentProcessEnvironmentSuppressesSplash` — **the gate
// proves its own splash suppression**. If the decision logic drifts (for example "improving"
// it to judge the environment variable by value rather than by key), this case fails, which
// prevents the regression of a 3-second window flashing during xcodebuild test.
import XCTest
@testable import SplatXMetalTrainer

// `SplashWindow` is @MainActor, so even reading its constants is main-isolated (only the
// decision, `shouldPresent`, is nonisolated).
@MainActor
final class SplashWindowTests: XCTestCase {
    // A plain launch with no markers = present it (the reading of "on every launch" in
    // requirement ③).
    func testPlainEnvironmentPresentsSplash() {
        XCTAssertTrue(SplashWindow.shouldPresent(environment: [:]))
        XCTAssertTrue(SplashWindow.shouldPresent(environment: ["HOME": "/Users/x"]))
    }

    // The human escape hatch — the value does not matter (the key's presence = off).
    func testOptOutEnvironmentVariable() {
        XCTAssertFalse(SplashWindow.shouldPresent(environment: ["SPLATX_NO_SPLASH": "1"]))
        XCTAssertFalse(SplashWindow.shouldPresent(environment: ["SPLATX_NO_SPLASH": ""]))
        XCTAssertFalse(SplashWindow.shouldPresent(environment: ["SPLATX_NO_SPLASH": "0"]))
    }

    // Each of the three XCTest harness markers must suppress it on its own.
    func testXCTestMarkersSuppressSplash() {
        XCTAssertFalse(SplashWindow.shouldPresent(
            environment: ["XCTestSessionIdentifier": UUID().uuidString]))
        XCTAssertFalse(SplashWindow.shouldPresent(
            environment: ["XCTestBundlePath": "/path/to/Tests.xctest"]))
        XCTAssertFalse(SplashWindow.shouldPresent(
            environment: ["XCTestConfigurationFilePath": "/path/to.plist"]))
    }

    /// ⚠ A measured regression guard: on this machine `XCTestConfigurationFilePath` **has the
    /// key with an empty-string value** (Xcode 26 / macOS 26.5). If the decision is changed to
    /// use `isEmpty`, or to bind the optional and then use the path, it breaks here — look only
    /// at whether the key exists.
    func testEmptyValuedXCTestMarkerStillSuppresses() {
        XCTAssertFalse(SplashWindow.shouldPresent(
            environment: ["XCTestConfigurationFilePath": ""]))
        XCTAssertFalse(SplashWindow.shouldPresent(environment: ["XCTestBundlePath": ""]))
        XCTAssertFalse(SplashWindow.shouldPresent(
            environment: ["XCTestSessionIdentifier": ""]))
    }

    /// The gate proving itself — in this very process (= the xcodebuild test host app) it must
    /// suppress.
    func testCurrentProcessEnvironmentSuppressesSplash() {
        XCTAssertFalse(SplashWindow.shouldPresent(
            environment: ProcessInfo.processInfo.environment),
            "the splash must not appear during test runs — prevents gate-screen pollution and focus stealing")
        // The second layer of defense, which survives a rename of the environment variable
        // (the second guard in presentIfEnabled).
        XCTAssertNotNil(NSClassFromString("XCTestCase"))
    }

    // The exposure-time contract: the total is 3.0 s (the user's request, "about 3 seconds"),
    // and the skip is shorter than the normal fade.
    func testTimingSumsToThreeSeconds() {
        XCTAssertEqual(SplashWindow.Timing.total, 3.0, accuracy: 1e-9)
        XCTAssertGreaterThan(SplashWindow.Timing.fadeIn, 0)
        XCTAssertGreaterThan(SplashWindow.Timing.hold, 0)
        XCTAssertGreaterThan(SplashWindow.Timing.fadeOut, 0)
        XCTAssertLessThan(SplashWindow.Timing.skip, SplashWindow.Timing.fadeOut)
    }

    // The tagline is the brand's canonical sentence (the text nodes in design/banner and
    // design/social) — do not write a new one.
    func testTaglineIsBrandCanonical() {
        XCTAssertEqual(SplashWindow.tagline,
                       "3D Gaussian Splatting training, native on Apple Silicon")
    }

    // MARK: Position (app finishing round 6 Task 3 — only the pure function
    // origin(forSize:in:). Real NSWindow and NSScreen objects are not verified, per policy.
    // This pins by coordinates that it is truly centered, not center()'s 1:3 bias.)

    /// Dead center of visibleFrame — the top and bottom margins are equal (a difference of up
    /// to 1pt caused by floor is allowed).
    func testOriginCentersInVisibleFrame() {
        let frame = NSRect(x: 0, y: 76, width: 1512, height: 869)
        let size = NSSize(width: 720, height: 420)
        let o = SplashWindow.origin(forSize: size, in: frame)
        XCTAssertEqual(o.x, 396)                                  // (1512−720)/2
        XCTAssertEqual(o.y, 300)                                  // 76 + (869−420)/2 = 300.5 → floor
        let below = o.y - frame.minY
        let above = frame.maxY - (o.y + size.height)
        XCTAssertLessThanOrEqual(abs(below - above), 1)           // equal margins (with center() it is 1:3)
    }

    /// The Dock on the left (an x offset in visibleFrame) — the center is inside visibleFrame,
    /// not the full frame.
    func testOriginRespectsDockOffset() {
        let frame = NSRect(x: 70, y: 0, width: 1442, height: 945)
        let o = SplashWindow.origin(forSize: NSSize(width: 720, height: 420), in: frame)
        XCTAssertEqual(o.x, 431)                                  // 70 + (1442−720)/2
        XCTAssertEqual(o.y, 262)                                  // (945−420)/2 = 262.5 → floor
    }

    /// A window larger than the screen — center()'s defense is preserved: vertically it falls
    /// back to top alignment (the top edge sticks to maxY), and horizontally the left side wins
    /// (the left edge sticks to minX). The low-resolution external display scenario.
    func testOriginClampsOversizedWindow() {
        let frame = NSRect(x: 0, y: 76, width: 1512, height: 869)
        let tall = SplashWindow.origin(forSize: NSSize(width: 720, height: 1300), in: frame)
        XCTAssertEqual(tall.y + 1300, frame.maxY)                 // top edge = maxY
        XCTAssertEqual(tall.x, 396)                               // horizontally still centered
        let wide = SplashWindow.origin(forSize: NSSize(width: 2000, height: 420), in: frame)
        XCTAssertEqual(wide.x, frame.minX)                        // left edge = minX
    }

    /// A fractional center (.5) is floored for pixel alignment — this keeps the wordmark and
    /// the 1pt hairline from blurring on a 1x display (the same alignment as center()'s measured
    /// behavior).
    func testOriginFloorsFractionalCenter() {
        let frame = NSRect(x: 0, y: 0, width: 1001, height: 801)
        let o = SplashWindow.origin(forSize: NSSize(width: 720, height: 420), in: frame)
        XCTAssertEqual(o.x, 140)                                  // 140.5 → 140
        XCTAssertEqual(o.y, 190)                                  // 190.5 → 190
    }
}
