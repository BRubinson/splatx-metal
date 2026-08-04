import XCTest
import AppKit
import SwiftUI
@testable import SplatXMetalTrainer

/// The pure decision for routing events across multiple viewports (app round 5 Task 2) — only
/// candidates in the same window (matching windowNumber) whose bounds are hit are matched, and
/// on a tie the last registration wins.
final class ViewportRoutingTests: XCTestCase {
    private func cand(_ win: Int?, at p: CGPoint,
                      bounds: CGRect = CGRect(x: 0, y: 0, width: 100, height: 100))
        -> ViewportRouting.Candidate {
        ViewportRouting.Candidate(windowNumber: win, bounds: bounds, location: p)
    }

    func testRoutesToViewportInSameWindowContainingPoint() {
        let i = ViewportRouting.targetIndex(eventWindowNumber: 7, candidates: [
            cand(3, at: CGPoint(x: 10, y: 10)),   // a different window — excluded even on a hit
            cand(7, at: CGPoint(x: 10, y: 10)),   // same window + inside bounds → the target
        ])
        XCTAssertEqual(i, 1)
    }

    func testIgnoresOutsideBoundsOtherWindowsAndEmpty() {
        XCTAssertNil(ViewportRouting.targetIndex(eventWindowNumber: 7, candidates: [
            cand(7, at: CGPoint(x: 150, y: 10)),  // same window but outside bounds
            cand(3, at: CGPoint(x: 10, y: 10)),   // inside bounds but a different window
        ]))
        XCTAssertNil(ViewportRouting.targetIndex(eventWindowNumber: 7, candidates: []))
    }

    func testDetachedViewNeverMatchesAndLastMatchWins() {
        let i = ViewportRouting.targetIndex(eventWindowNumber: 7, candidates: [
            cand(nil, at: CGPoint(x: 10, y: 10)), // a view not attached to a window (windowNumber nil) — excluded
            cand(7, at: CGPoint(x: 10, y: 10)),
            cand(7, at: CGPoint(x: 20, y: 20)),   // a tie → the last registration (the topmost view) wins
        ])
        XCTAssertEqual(i, 2)
    }

    /// Cinema Task 4: the decision that excludes a camera drag — the same hit rule as
    /// targetIndex (same window + inside bounds). This is why a drag that starts on the overlay
    /// image does not become an orbit session.
    func testExclusionUsesSameHitJudgement() {
        XCTAssertTrue(ViewportRouting.isExcluded(eventWindowNumber: 7, candidates: [
            cand(7, at: CGPoint(x: 150, y: 10)),  // outside bounds — no hit
            cand(7, at: CGPoint(x: 10, y: 10)),   // a hit → excluded
        ]))
        XCTAssertFalse(ViewportRouting.isExcluded(eventWindowNumber: 2, candidates: [
            cand(7, at: CGPoint(x: 10, y: 10)),   // a different window — not excluded
        ]))
        XCTAssertFalse(ViewportRouting.isExcluded(eventWindowNumber: 7, candidates: []))
    }

    /// FIX 3 item 6: so that when a viewer window closes (the view is released), a dead entry
    /// does not keep holding the camera binding (→ ViewerModel, and a large scene's memory)
    /// until the next input event — an explicit detach removes the registration immediately and
    /// leaves the other viewports alone.
    @MainActor
    func testRegistryDetachDropsEntryImmediately() {
        let registry = ViewportRegistry()
        let a = NSView(frame: CGRect(x: 0, y: 0, width: 10, height: 10))
        let b = NSView(frame: CGRect(x: 0, y: 0, width: 10, height: 10))
        registry.attach(view: a, camera: .constant(CameraController()))
        registry.attach(view: b, camera: .constant(CameraController()))
        XCTAssertNotNil(registry.camera(for: a))
        registry.detach(view: a)
        XCTAssertNil(registry.camera(for: a))    // deregistered immediately — no waiting for the next event
        XCTAssertNotNil(registry.camera(for: b)) // the other viewport is untouched
    }
}
