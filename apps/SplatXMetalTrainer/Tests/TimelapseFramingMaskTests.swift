// 2026-08-05 spec §2.5: the Train tab's live viewport dims what the running timelapse is framing
// out, reusing Cinema's letterbox *dimming*. The drawing itself needs SwiftUI layout, so what is
// tested here is the pure judgement underneath — the height-locked mask arithmetic and when to draw
// nothing at all. The visual result is manual check M22 (M23 is the step_ms check; the header used
// to name it, timelapse-viewport-aspect FIX 1 corrected it).
//
// `timelapse-viewport-aspect` FIX 1 is why the two halves of this file no longer meet. Task 3 built
// the Train mask on `OverlayPlacement.maskRect`, and that was wrong: the capture and the viewport
// share `fovY`, so they always cover the same **vertical** range and aspect-fit's "capture is wider
// → dim the top and bottom" branch dimmed strips that are genuinely recorded. `TrainView`'s
// arithmetic is now its own; the `OutputAspect` group below stays as Cinema's regression defence,
// because `maskRect` is deliberately untouched and this file is where its behaviour was pinned.
//
// The regression that let the defect through the gate was in the *coverage*, not the assertions:
// the round-trip residual case ran at shortSide 1080 only, the single most forgiving of the six
// choices (the residual is relative, so it is ~4.5× larger at 240). Both sweeps below iterate
// `TimelapseSettings.shortSideChoices` in full, so a seventh choice is covered the day it is added.
import CoreGraphics
import XCTest
@testable import SplatXMetalTrainer

/// `@MainActor` on the class because `TrainViewModel`'s statics inherit the view model's isolation
/// (TrainViewportSizeTests' form). `TrainView`'s and `OverlayPlacement`'s statics are not isolated,
/// and calling them from here is free.
@MainActor
final class TimelapseFramingMaskTests: XCTestCase {

    /// 4:3 — picked so "wider", "narrower" and "equal" are all expressible with ratios that are
    /// *not* OutputAspect members, keeping those cases independent of the enum.
    private let container = CGSize(width: 800, height: 600)

    /// The viewport sizes every sweep runs over: the app's minimum content size, a common windowed
    /// size, full-screen on this machine's display, and one portrait window (a viewport taller than
    /// it is wide is the orientation in which the tolerance arithmetic is tightest).
    private let containers = [CGSize(width: 1128, height: 668),
                              CGSize(width: 1400, height: 900),
                              CGSize(width: 1920, height: 1080),
                              CGSize(width: 700, height: 1100)]

    // MARK: - Cinema regression (maskRect's enum → ratio → rect chain is unchanged)

    /// Aspect-fit, both branches. These are Cinema's semantics and stay correct there — the Train
    /// tab simply stopped being a second caller.
    func testMaskRectStillAspectFitsOnBothAxes() {
        XCTAssertEqual(OverlayPlacement.maskRect(aspect: 2.0, in: container),
                       CGRect(x: 0, y: 100, width: 800, height: 400))
        XCTAssertEqual(OverlayPlacement.maskRect(aspect: 0.75, in: container),
                       CGRect(x: 175, y: 0, width: 450, height: 600))
    }

    func testOutputAspectRatiosStillProduceTheSameRects() {
        XCTAssertEqual(OverlayPlacement.maskRect(aspect: OutputAspect.wide.ratio, in: container),
                       CGRect(x: 0, y: 75, width: 800, height: 450))
        XCTAssertEqual(OverlayPlacement.maskRect(aspect: OutputAspect.tall.ratio, in: container),
                       CGRect(x: 231.25, y: 0, width: 337.5, height: 600))
        XCTAssertEqual(OverlayPlacement.maskRect(aspect: OutputAspect.square.ratio, in: container),
                       CGRect(x: 100, y: 0, width: 600, height: 600))
    }

    // MARK: - timelapseMask: the height-locked arithmetic

    /// A capture narrower than the viewport is the one case that draws: the mask keeps the **full
    /// height** and bars appear left and right. 1600×900 with a 3:2 capture → 1350 wide, 125pt of
    /// bar each side. This is the `.dataset` mismatch the preview exists to make visible.
    func testANarrowerCapturePillarboxesAtFullHeight() {
        let mask = TrainView.timelapseMask(active: true, capture: CGSize(width: 600, height: 400),
                                           in: CGSize(width: 1600, height: 900))
        XCTAssertEqual(mask, CGRect(x: 125, y: 0, width: 1350, height: 900))
    }

    /// A capture **wider** than the viewport draws nothing. Under the shared `fovY` the recorded
    /// frame includes more to the left and right and crops nothing top or bottom — the pre-FIX
    /// aspect-fit reused from Cinema dimmed two horizontal strips here, claiming the exact opposite
    /// of what happens. There is no mask that says "more than this is recorded", so the answer is
    /// to draw nothing.
    func testAWiderCaptureDrawsNothingRatherThanLetterboxing() {
        XCTAssertNil(TrainView.timelapseMask(active: true, capture: CGSize(width: 1600, height: 800),
                                             in: CGSize(width: 1600, height: 900)))
        // Same shape one notch further out, so this cannot pass on a near-match tolerance.
        XCTAssertNil(TrainView.timelapseMask(active: true, capture: CGSize(width: 3200, height: 800),
                                             in: CGSize(width: 1600, height: 900)))
    }

    /// Equal ratios draw nothing — the case that must not leave `LetterboxDim`'s border stroke
    /// framing the viewport forever.
    func testAMatchingRatioDrawsNothing() {
        XCTAssertNil(TrainView.timelapseMask(active: true, capture: CGSize(width: 1200, height: 675),
                                             in: CGSize(width: 1600, height: 900)))
    }

    /// A portrait viewport with a landscape capture. Height-locked means the mask can be **wider
    /// than the container** here, which is exactly the "nothing is cropped" case — a landscape
    /// capture in a tall window shows less vertically than the window, never less horizontally.
    func testAPortraitViewportWithALandscapeCaptureDrawsNothing() {
        XCTAssertNil(TrainView.timelapseMask(active: true, capture: CGSize(width: 1920, height: 1080),
                                             in: CGSize(width: 700, height: 1100)))
    }

    /// …and the mirror: a portrait capture in a portrait viewport still pillarboxes when it is the
    /// narrower of the two.
    func testAPortraitViewportWithANarrowerPortraitCapturePillarboxes() {
        let mask = TrainView.timelapseMask(active: true, capture: CGSize(width: 480, height: 960),
                                           in: CGSize(width: 700, height: 1100))
        XCTAssertEqual(mask, CGRect(x: 75, y: 0, width: 550, height: 1100))
    }

    // MARK: - timelapseMask: when to draw nothing

    func testNoMaskWhileNoTimelapseIsRunning() {
        XCTAssertNil(TrainView.timelapseMask(active: false, capture: CGSize(width: 600, height: 400),
                                             in: container))
    }

    /// `captureFrameSize` is nil outside a run and the two fields move together through
    /// `setTimelapseRun` — but the gate reads both, so a future path that forgets one of them
    /// still cannot dim a viewport with no recorder behind it.
    func testNoMaskWithoutACaptureSize() {
        XCTAssertNil(TrainView.timelapseMask(active: true, capture: nil, in: container))
    }

    /// A viewport not laid out yet, and sizes that could divide by zero downstream.
    func testDegenerateInputsProduceNoMask() {
        XCTAssertNil(TrainView.timelapseMask(active: true, capture: CGSize(width: 1600, height: 900),
                                             in: .zero))
        XCTAssertNil(TrainView.timelapseMask(active: true, capture: .zero, in: container))
        XCTAssertNil(TrainView.timelapseMask(active: true, capture: CGSize(width: 600, height: 0),
                                             in: container))
        XCTAssertNil(TrainView.timelapseMask(active: true, capture: CGSize(width: 0, height: 400),
                                             in: container))
        XCTAssertNil(TrainView.timelapseMask(active: true,
                                             capture: CGSize(width: -600, height: -400),
                                             in: container))
    }

    // MARK: - The round trip: `.viewport` mode must never draw

    /// The property this file exists for. In `.viewport` mode the capture ratio *is* this
    /// viewport's ratio, pushed through `aspectPair`'s pt→Int rounding and `frameSize`'s
    /// `rounded()` + `evenUp` — so a mask is always a false alarm, whatever the size preset and
    /// whatever the window.
    ///
    /// Every `shortSideChoices` entry × every container. The old version of this test used 1080
    /// alone and passed while 240/360/480 misfired: the residual is a *relative* rounding error, so
    /// converting it to points multiplies it by the container, and at shortSide 240 the long side
    /// is only ~350px. That is also why `timelapseMask`'s tolerance is derived from the capture's
    /// pixel size instead of being a fixed point count — and why it takes a `CGSize` rather than
    /// the ratio it used to take, since a ratio cannot tell you how coarse it is.
    func testViewportModeNeverMasksAtAnyShortSideOrWindowSize() {
        for shortSide in TimelapseSettings.shortSideChoices {
            for viewport in containers {
                let pair = TimelapseSettings.aspectPair(source: .viewport, viewport: viewport,
                                                        datasetWidth: 648, datasetHeight: 420)
                let frame = TimelapseSettings.frameSize(shortSide: shortSide,
                                                        sourceWidth: pair.width,
                                                        sourceHeight: pair.height)
                let capture = CGSize(width: frame.width, height: frame.height)
                XCTAssertNil(TrainView.timelapseMask(active: true, capture: capture, in: viewport),
                             "shortSide \(shortSide) in \(viewport) captured \(capture) — "
                             + "the viewport's own ratio must never read as a mismatch")
            }
        }
    }

    /// Non-integer viewport sizes, which is what a real `GeometryReader` reports. Same property,
    /// and it also exercises `aspectPair`'s `Int(_.rounded())` — the first of the two rounding
    /// stages the tolerance is derived from.
    func testViewportModeNeverMasksForFractionalViewportSizes() {
        let fractional = [CGSize(width: 1128, height: 634.5),
                          CGSize(width: 1399.5, height: 899.5),
                          CGSize(width: 1013.75, height: 1357.25)]
        for shortSide in TimelapseSettings.shortSideChoices {
            for viewport in fractional {
                let pair = TimelapseSettings.aspectPair(source: .viewport, viewport: viewport,
                                                        datasetWidth: 648, datasetHeight: 420)
                let frame = TimelapseSettings.frameSize(shortSide: shortSide,
                                                        sourceWidth: pair.width,
                                                        sourceHeight: pair.height)
                let capture = CGSize(width: frame.width, height: frame.height)
                XCTAssertNil(TrainView.timelapseMask(active: true, capture: capture, in: viewport),
                             "shortSide \(shortSide) in \(viewport) captured \(capture)")
            }
        }
    }

    // MARK: - The round trip: `.dataset` mode must draw

    /// The counterweight to the sweeps above: a tolerance wide enough to swallow a real mismatch
    /// would pass every one of them. The garden dataset (648×420, 1.543) in a 16:9-ish window is
    /// the case manual check M22 looks at, and its bars are two orders of magnitude past the
    /// tolerance at every preset.
    func testDatasetModeMasksAtEveryShortSide() throws {
        let viewport = CGSize(width: 1128, height: 634.5)
        for shortSide in TimelapseSettings.shortSideChoices {
            let pair = TimelapseSettings.aspectPair(source: .dataset, viewport: viewport,
                                                    datasetWidth: 648, datasetHeight: 420)
            let frame = TimelapseSettings.frameSize(shortSide: shortSide,
                                                    sourceWidth: pair.width,
                                                    sourceHeight: pair.height)
            let mask = try XCTUnwrap(
                TrainView.timelapseMask(active: true,
                                        capture: CGSize(width: frame.width, height: frame.height),
                                        in: viewport),
                "shortSide \(shortSide): a 648×420 dataset in a 1.778 viewport is a real mismatch")
            XCTAssertEqual(mask.height, viewport.height, accuracy: 1e-9,
                           "height-locked: the bars are on the sides, never top and bottom")
            XCTAssertEqual(mask.minY, 0, accuracy: 1e-9)
            // 978.19 (240) … 980.83 (480): the preset only moves the mask by the long side's
            // rounding, while the bar it leaves is ~147pt at every one of them.
            XCTAssertEqual(mask.width, 979.51, accuracy: 1.4)
            XCTAssertGreaterThan(viewport.width - mask.width, 100,
                                 "the mismatch this preview exists for is tens of points of bar")
            XCTAssertEqual(mask.midX, viewport.width / 2, accuracy: 1e-9, "centred")
        }
    }
}
