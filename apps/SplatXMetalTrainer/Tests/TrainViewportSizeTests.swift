// 2026-08-05 spec §2.4: the Train viewport's size is now carried to the timelapse config, so the
// app keeps that size for the first time. The wiring itself (a GeometryReader plus
// TrainViewModel.start) needs SwiftUI layout, a Metal device and a real run, so per this app's
// policy it is covered by build success plus manual checks M20–M22 — the same split
// TrainStartViewpointTests documents. What *is* testable is the pure decision the wiring leans on:
// which reported sizes are worth storing.
//
// It used to also cover `TrainViewModel.aspectRatio`, the w/h the letterbox mask read.
// `timelapse-viewport-aspect` FIX 1 deleted that helper — a ratio is exactly what the mask must
// *not* be given, because the near-match tolerance has to know how many pixels produced it — so the
// mask's end of the chain is now asserted through `TrainView.timelapseMask` in
// TimelapseFramingMaskTests, which runs it at every `shortSideChoices` entry.
import CoreGraphics
import XCTest
@testable import SplatXMetalTrainer

/// `@MainActor` on the class rather than on each case (TrainStartViewpointTests' form): every
/// member of `TrainViewModel` inherits its isolation, including the static helper here.
@MainActor
final class TrainViewportSizeTests: XCTestCase {

    // MARK: - viewportSizeUpdate (which reports become state)

    func testAFirstSizeIsRecorded() {
        let next = TrainViewModel.viewportSizeUpdate(reported: CGSize(width: 1440, height: 810),
                                                     current: .zero)
        XCTAssertEqual(next, CGSize(width: 1440, height: 810))
    }

    /// Rule ① — an unchanged report writes nothing. `.onAppear` fires again on every switch of the
    /// live/placeholder conditional inside the reader, and this is @Observable state, so an
    /// unfiltered write would invalidate every reader of it for no change at all.
    func testAnUnchangedSizeIsNotWritten() {
        let size = CGSize(width: 1440, height: 810)
        XCTAssertNil(TrainViewModel.viewportSizeUpdate(reported: size, current: size))
    }

    func testAChangedSizeReplacesThePreviousOne() {
        let next = TrainViewModel.viewportSizeUpdate(reported: CGSize(width: 1200, height: 800),
                                                     current: CGSize(width: 1440, height: 810))
        XCTAssertEqual(next, CGSize(width: 1200, height: 800))
    }

    /// Rule ② — a non-finite report never becomes state. `aspectPair` guards `isFinite` (because
    /// `Int(CGFloat.nan)` traps), but storing NaN would make it answer "dataset" for the rest of
    /// the session with nothing left to say why.
    func testANonFiniteSizeIsDropped() {
        let current = CGSize(width: 1440, height: 810)
        // CGFloat is spelled out: CGSize.init has Int/Double/CGFloat overloads, so a bare `.nan`
        // is ambiguous (the same reason TimelapseSettingsTests writes it this way).
        let bad: [CGSize] = [
            CGSize(width: CGFloat.nan, height: CGFloat.nan),
            CGSize(width: CGFloat.nan, height: 810),
            CGSize(width: CGFloat.infinity, height: 810),
            CGSize(width: 1440, height: -CGFloat.infinity),
        ]
        for size in bad {
            XCTAssertNil(TrainViewModel.viewportSizeUpdate(reported: size, current: current),
                         "\(size)")
        }
        // …even when nothing usable has been recorded yet: a stored NaN is worse than a stored zero,
        // which at least reads as "no ratio here".
        XCTAssertNil(TrainViewModel.viewportSizeUpdate(
            reported: CGSize(width: CGFloat.nan, height: CGFloat.nan), current: .zero))
    }

    /// Rule ③ — a usable size is never replaced by an unusable one. The value is read exactly once,
    /// at Start Training; a transient zero from a layout pass landing just before that click would
    /// produce a timelapse in a ratio the user never chose.
    func testAUsableSizeSurvivesADegenerateReport() {
        let current = CGSize(width: 1440, height: 810)
        let degenerate: [CGSize] = [
            .zero,
            CGSize(width: 1440, height: 0),        // one axis is enough
            CGSize(width: 0.5, height: 0.4),       // sub-point: would round to 0 downstream
            CGSize(width: -1440, height: -810),
        ]
        for size in degenerate {
            XCTAssertNil(TrainViewModel.viewportSizeUpdate(reported: size, current: current),
                         "\(size)")
        }
    }

    /// The mirror of rule ③: with nothing usable recorded yet there is nothing to protect, so a
    /// degenerate report is stored as is and `aspectPair` resolves it (fallback ①). Otherwise the
    /// first genuine zero would be indistinguishable from the initial state forever.
    func testADegenerateReportIsStoredWhenNothingUsableWasRecorded() {
        let next = TrainViewModel.viewportSizeUpdate(reported: CGSize(width: 1440, height: 0),
                                                     current: .zero)
        XCTAssertEqual(next, CGSize(width: 1440, height: 0))
    }

    // MARK: - The wired chain (the point of the round)

    /// A recorded viewport size reaches the frame — and the ratio survives the trip. The pt→px
    /// scale is deliberately absent from this chain (§2.2: only a ratio is consumed), which is why
    /// a 1440×810 viewport and a 2880×1620 one must produce the same frame.
    func testARecordedViewportSizeReachesTheFrameAsItsRatio() {
        func frame(_ reported: CGSize) -> (width: Int, height: Int) {
            let stored = TrainViewModel.viewportSizeUpdate(reported: reported, current: .zero)
            let pair = TimelapseSettings.aspectPair(source: .viewport, viewport: stored!,
                                                    datasetWidth: 648, datasetHeight: 432)
            return TimelapseSettings.frameSize(shortSide: 480, sourceWidth: pair.width,
                                               sourceHeight: pair.height)
        }
        let onePoint = frame(CGSize(width: 1440, height: 810))       // 16:9
        XCTAssertEqual(onePoint.width, 854)                          // 480 × 16/9 → 853.33 → 854
        XCTAssertEqual(onePoint.height, 480)
        // The same viewport on a Retina display reports the same points; doubling stands in for the
        // pixel size nobody converts to, and must not change the answer.
        let doubled = frame(CGSize(width: 2880, height: 1620))
        XCTAssertEqual(doubled.width, onePoint.width)
        XCTAssertEqual(doubled.height, onePoint.height)
        // And that frame is what the mask is asked about — in `.viewport` mode it must read as no
        // mismatch at all, at either backing scale.
        let capture = CGSize(width: onePoint.width, height: onePoint.height)
        XCTAssertNil(TrainView.timelapseMask(active: true, capture: capture,
                                             in: CGSize(width: 1440, height: 810)))
        XCTAssertNil(TrainView.timelapseMask(active: true, capture: capture,
                                             in: CGSize(width: 2880, height: 1620)))
    }
}
