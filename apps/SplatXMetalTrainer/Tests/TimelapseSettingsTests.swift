// 2026-08-04 spec §3.4/§3.7: the pure math of the timelapse settings — the resolution
// derivation (short-side preset × source aspect → even dimensions), the GIF auto-reduction
// plan (960px long side, at most 200 uniformly sampled frames), the frame file name, and
// the capture-period rule. The renderer, the GPU, and SwiftUI are manual-check territory
// (the app's no-tests-for-views policy).
import XCTest
import CoreGraphics
@testable import SplatXMetalTrainer

final class TimelapseSettingsTests: XCTestCase {

    // MARK: - Defaults (spec §3.6 — timelapse is off by default)

    func testDefaults() {
        let s = TimelapseSettings()
        XCTAssertFalse(s.enabled)
        XCTAssertEqual(s.everySteps, 100)
        XCTAssertEqual(s.format, .gif)      // shareable inline; the video formats are opt-in
        XCTAssertEqual(s.fps, 15)
        XCTAssertEqual(s.shortSide, 480)    // stays under gifMaxLongSide at 16:9 — no clamp
        XCTAssertEqual(s.aspectSource, .viewport)   // 2026-08-05 §2.1: "보이는 그대로"가 기본
        XCTAssertTrue(s.overlay)
    }

    // MARK: - Format mapping

    func testFormatMapping() {
        XCTAssertEqual(TimelapseFormat.allCases, [.mp4, .mov, .gif])
        XCTAssertEqual(TimelapseFormat.mp4.rawValue, "mp4")
        XCTAssertEqual(TimelapseFormat.mov.rawValue, "mov")
        XCTAssertEqual(TimelapseFormat.gif.rawValue, "gif")
        XCTAssertEqual(TimelapseFormat.mp4.fileExtension, "mp4")
        XCTAssertEqual(TimelapseFormat.gif.fileExtension, "gif")
        XCTAssertEqual(TimelapseFormat.mp4.displayName, "MP4")
        XCTAssertEqual(TimelapseFormat.mov.displayName, "MOV")
        XCTAssertEqual(TimelapseFormat.gif.displayName, "GIF")
        // The video container axis reuses Cinema's VideoContainer verbatim (spec §3.4 — the
        // MP4/MOV path is VideoExporter with no modification). GIF has no container.
        XCTAssertEqual(TimelapseFormat.mp4.videoContainer, .mp4)
        XCTAssertEqual(TimelapseFormat.mov.videoContainer, .mov)
        XCTAssertNil(TimelapseFormat.gif.videoContainer)
    }

    // MARK: - Resolution derivation (short side = the preset, long side from the source aspect,
    // both rounded up to even — the encoder rejects odd dimensions)

    func testFrameSizeTable() {
        let table: [(shortSide: Int, srcW: Int, srcH: Int, w: Int, h: Int)] = [
            (1080, 1920, 1080, 1920, 1080),
            (720,  1920, 1080, 1280, 720),
            (1440, 1920, 1080, 2560, 1440),
            (1080,  648,  420, 1666, 1080),   // garden images_8 (648×420)
            (1080, 1080, 1920, 1080, 1920),   // portrait — the short side is the width
            (720,  1000, 1000,  720,  720),   // square
            // The GIF-friendly sizes. 480 → 853.33 rounds to 853 and the even-up gives 854;
            // 240 → 426.67 rounds to 427 and evens up to 428 (not the 426 the ratio suggests).
            (480,  1920, 1080,  854,  480),
            (360,  1920, 1080,  640,  360),
            (240,  1920, 1080,  428,  240),
        ]
        for row in table {
            let size = TimelapseSettings.frameSize(shortSide: row.shortSide,
                                                   sourceWidth: row.srcW,
                                                   sourceHeight: row.srcH)
            XCTAssertEqual(size.width, row.w, "\(row)")
            XCTAssertEqual(size.height, row.h, "\(row)")
            XCTAssertEqual(size.width % 2, 0, "\(row)")
            XCTAssertEqual(size.height % 2, 0, "\(row)")
        }
    }

    func testFrameSizeRoundsOddDimensionsUpToEven() {
        // 1080·1001/1000 = 1081.08 → 1081 (odd) → 1082
        let size = TimelapseSettings.frameSize(shortSide: 1080, sourceWidth: 1001,
                                               sourceHeight: 1000)
        XCTAssertEqual(size.width, 1082)
        XCTAssertEqual(size.height, 1080)
        // An odd preset is made even as well (the UI only offers even ones — a guard).
        let odd = TimelapseSettings.frameSize(shortSide: 721, sourceWidth: 1000,
                                              sourceHeight: 1000)
        XCTAssertEqual(odd.width, 722)
        XCTAssertEqual(odd.height, 722)
    }

    func testFrameSizeDegenerateSourceFallsBackTo16by9() {
        // No COLMAP context (the fit failed) → 16:9 landscape, so the capture still has a
        // sane shape.
        let size = TimelapseSettings.frameSize(shortSide: 1080, sourceWidth: 0, sourceHeight: 0)
        XCTAssertEqual(size.width, 1920)
        XCTAssertEqual(size.height, 1080)
    }

    // MARK: - Aspect source (2026-08-05 spec §2.1–§2.3)

    func testAspectSourceCasesAndRawValues() {
        XCTAssertEqual(TimelapseAspectSource.allCases, [.viewport, .dataset])
        // The raw values are what the manifest stores — pinned so a rename is a deliberate
        // migration rather than a silent reset to the default on the next open.
        XCTAssertEqual(TimelapseAspectSource.viewport.rawValue, "viewport")
        XCTAssertEqual(TimelapseAspectSource.dataset.rawValue, "dataset")
        XCTAssertEqual(TimelapseAspectSource.viewport.id, "viewport")
        XCTAssertEqual(TimelapseAspectSource.viewport.displayName, "Viewport")
        XCTAssertEqual(TimelapseAspectSource.dataset.displayName, "Dataset")
        XCTAssertEqual(TimelapseSettings.aspectSourceChoices, TimelapseAspectSource.allCases)
    }

    /// The viewport pair is taken as is (only its ratio is consumed downstream, so the pt size
    /// needs no backing-scale conversion).
    func testAspectPairViewportModeUsesTheViewport() {
        let pair = TimelapseSettings.aspectPair(source: .viewport,
                                                viewport: CGSize(width: 1600, height: 900),
                                                datasetWidth: 648, datasetHeight: 432)
        XCTAssertEqual(pair.width, 1600)
        XCTAssertEqual(pair.height, 900)
        // A fractional size (the usual GeometryReader output) rounds; the ratio is what matters.
        let fractional = TimelapseSettings.aspectPair(
            source: .viewport, viewport: CGSize(width: 1279.6, height: 720.4),
            datasetWidth: 0, datasetHeight: 0)
        XCTAssertEqual(fractional.width, 1280)
        XCTAssertEqual(fractional.height, 720)
    }

    /// Fallback ① (§2.3): the recorder is built in TrainViewModel.start, where the Train tab's
    /// MTKView may not have been laid out yet — an unusable viewport must land on the dataset,
    /// not on 16:9.
    func testAspectPairViewportModeFallsBackToTheDatasetWhenTheViewportIsUnusable() {
        let unusable: [CGSize] = [
            .zero,
            CGSize(width: 1600, height: 0),          // one axis missing is enough
            CGSize(width: -100, height: -50),        // a negative is not a ratio
            CGSize(width: 0.5, height: 0.4),         // sub-point: would round to 0
            CGSize(width: CGFloat.nan, height: CGFloat.nan),   // Int(nan) traps — must be caught
            CGSize(width: CGFloat.infinity, height: 900),
        ]
        for size in unusable {
            let pair = TimelapseSettings.aspectPair(source: .viewport, viewport: size,
                                                    datasetWidth: 648, datasetHeight: 432)
            XCTAssertEqual(pair.width, 648, "\(size)")
            XCTAssertEqual(pair.height, 432, "\(size)")
        }
    }

    func testAspectPairDatasetModeIgnoresTheViewport() {
        let pair = TimelapseSettings.aspectPair(source: .dataset,
                                                viewport: CGSize(width: 1600, height: 900),
                                                datasetWidth: 648, datasetHeight: 420)
        XCTAssertEqual(pair.width, 648)     // garden images_8
        XCTAssertEqual(pair.height, 420)
    }

    /// Fallback ② (§2.3): with neither source usable the degenerate pair is forwarded unchanged
    /// so `frameSize`'s own 16:9 guard answers — the single place that answer is written down.
    func testAspectPairFallsThroughToFrameSizeWhenNeitherSourceHasASize() {
        for source in TimelapseAspectSource.allCases {
            let pair = TimelapseSettings.aspectPair(source: source, viewport: .zero,
                                                    datasetWidth: 0, datasetHeight: 0)
            XCTAssertEqual(pair.width, 0, "\(source)")
            XCTAssertEqual(pair.height, 0, "\(source)")
            let size = TimelapseSettings.frameSize(shortSide: 480, sourceWidth: pair.width,
                                                   sourceHeight: pair.height)
            XCTAssertEqual(size.width, 854, "\(source)")    // 480 × 16/9, evened up
            XCTAssertEqual(size.height, 480, "\(source)")
        }
        // A negative dataset dimension is normalised to the same canonical degenerate pair.
        let negative = TimelapseSettings.aspectPair(source: .dataset, viewport: .zero,
                                                    datasetWidth: -1920, datasetHeight: -1080)
        XCTAssertEqual(negative.width, 0)
        XCTAssertEqual(negative.height, 0)
    }

    /// The point of the whole round: with a 16:9 viewport and a 3:2 dataset the two sources must
    /// produce **different** frames. If this ever passes with equal sizes the option is decorative.
    func testViewportAndDatasetSourcesProduceDifferentFrameSizes() {
        let viewport = CGSize(width: 1600, height: 900)     // 16:9
        let (dw, dh) = (648, 432)                           // 3:2
        func size(_ source: TimelapseAspectSource) -> (width: Int, height: Int) {
            let pair = TimelapseSettings.aspectPair(source: source, viewport: viewport,
                                                    datasetWidth: dw, datasetHeight: dh)
            return TimelapseSettings.frameSize(shortSide: 480, sourceWidth: pair.width,
                                               sourceHeight: pair.height)
        }
        XCTAssertEqual(size(.viewport).width, 854)          // 480 × 16/9 → 853.33 → 854
        XCTAssertEqual(size(.viewport).height, 480)
        XCTAssertEqual(size(.dataset).width, 720)           // 480 × 3/2
        XCTAssertEqual(size(.dataset).height, 480)
        XCTAssertNotEqual(size(.viewport).width, size(.dataset).width)
    }

    // MARK: - GIF plan (spec §3.4 — 960px long side, at most 200 frames, uniform sampling)

    func testGifPlanScalesDownButKeepsEveryFrame() {
        let plan = TimelapseSettings.gifPlan(frameCount: 150, width: 1920, height: 1080)
        XCTAssertEqual(plan.indices, Array(0..<150))
        XCTAssertEqual(plan.width, 960)
        XCTAssertEqual(plan.height, 540)
    }

    func testGifPlanPortraitClampsTheLongSide() {
        let plan = TimelapseSettings.gifPlan(frameCount: 10, width: 1080, height: 1920)
        XCTAssertEqual(plan.indices, Array(0..<10))
        XCTAssertEqual(plan.width, 540)
        XCTAssertEqual(plan.height, 960)
    }

    func testGifPlanKeepsSmallSizeUnchanged() {
        let plan = TimelapseSettings.gifPlan(frameCount: 500, width: 640, height: 480)
        XCTAssertEqual(plan.width, 640)
        XCTAssertEqual(plan.height, 480)
        XCTAssertEqual(plan.indices.count, 200)
        XCTAssertEqual(plan.indices.first, 0)
        XCTAssertEqual(plan.indices.last, 499)   // the last frame is always included
    }

    /// The reason the low end of `shortSideChoices` exists: at 16:9 those sizes stay under
    /// `gifMaxLongSide`, so the number the user picked survives to the file. The default (480) is
    /// the largest one that does — 720 and up are clamped, which is why the old 1080 default
    /// silently produced 540p once the format was GIF.
    func testLowShortSidesSurviveTheGifClamp() {
        for shortSide in [240, 360, 480] {
            let size = TimelapseSettings.frameSize(shortSide: shortSide,
                                                   sourceWidth: 1920, sourceHeight: 1080)
            let plan = TimelapseSettings.gifPlan(frameCount: 10,
                                                 width: size.width, height: size.height)
            XCTAssertEqual(plan.width, size.width, "shortSide \(shortSide)")
            XCTAssertEqual(plan.height, size.height, "shortSide \(shortSide)")
            XCTAssertNil(TimelapseSettings.gifReductionNote(frameCount: 10,
                                                            width: size.width,
                                                            height: size.height),
                         "shortSide \(shortSide) should need no reduction note")
        }
        // 720 is where the clamp starts biting — 1280×720 → 960×540.
        let clamped = TimelapseSettings.frameSize(shortSide: 720,
                                                  sourceWidth: 1920, sourceHeight: 1080)
        let clampedPlan = TimelapseSettings.gifPlan(frameCount: 10,
                                                    width: clamped.width, height: clamped.height)
        XCTAssertEqual(clampedPlan.width, TimelapseSettings.gifMaxLongSide)
        XCTAssertLessThan(clampedPlan.height, clamped.height)
    }

    func testGifPlanSamplingIsStrictlyIncreasing() {
        for count in [201, 250, 400, 1000, 3000] {
            let plan = TimelapseSettings.gifPlan(frameCount: count, width: 640, height: 480)
            XCTAssertEqual(plan.indices.count, 200, "count \(count)")
            XCTAssertEqual(plan.indices.first, 0, "count \(count)")
            XCTAssertEqual(plan.indices.last, count - 1, "count \(count)")
            for i in 1..<plan.indices.count {
                XCTAssertGreaterThan(plan.indices[i], plan.indices[i - 1],
                                     "count \(count) at \(i)")   // no duplicate frames
            }
        }
    }

    /// Spec §3.4: "no silent truncation" — the reduction is reported so the completion log can
    /// state it.
    func testGifReductionNote() {
        XCTAssertEqual(TimelapseSettings.gifReductionNote(frameCount: 150, width: 1920,
                                                          height: 1080),
                       "GIF reduced: size 1920×1080→960×540")
        XCTAssertEqual(TimelapseSettings.gifReductionNote(frameCount: 500, width: 640,
                                                          height: 480),
                       "GIF reduced: frames 500→200")
        XCTAssertEqual(TimelapseSettings.gifReductionNote(frameCount: 500, width: 1920,
                                                          height: 1080),
                       "GIF reduced: frames 500→200, size 1920×1080→960×540")
        XCTAssertNil(TimelapseSettings.gifReductionNote(frameCount: 10, width: 640, height: 480))
    }

    // MARK: - Capture period (spec §3.7 — multiples of 50 only)

    func testEveryStepsValidityAndNormalization() {
        XCTAssertTrue(TimelapseSettings.isValidEverySteps(50))
        XCTAssertTrue(TimelapseSettings.isValidEverySteps(100))
        XCTAssertTrue(TimelapseSettings.isValidEverySteps(2000))
        XCTAssertFalse(TimelapseSettings.isValidEverySteps(75))     // not a multiple of 50
        XCTAssertFalse(TimelapseSettings.isValidEverySteps(0))      // below the range
        XCTAssertFalse(TimelapseSettings.isValidEverySteps(2050))   // above the range

        XCTAssertEqual(TimelapseSettings.normalizedEverySteps(0), 50)
        XCTAssertEqual(TimelapseSettings.normalizedEverySteps(49), 50)
        XCTAssertEqual(TimelapseSettings.normalizedEverySteps(75), 100)
        XCTAssertEqual(TimelapseSettings.normalizedEverySteps(100), 100)
        XCTAssertEqual(TimelapseSettings.normalizedEverySteps(5000), 2000)
    }

    func testShouldCaptureFollowsThePeriodAndTheEnabledFlag() {
        var s = TimelapseSettings()
        s.enabled = true
        s.everySteps = 100
        XCTAssertFalse(s.shouldCapture(iter: 0))     // iteration 0 is not a capture point
        XCTAssertFalse(s.shouldCapture(iter: 50))    // a swap that is not a capture point
        XCTAssertTrue(s.shouldCapture(iter: 100))
        XCTAssertTrue(s.shouldCapture(iter: 7000))
        s.everySteps = 50                            // every swap (the live snapshot grid)
        XCTAssertTrue(s.shouldCapture(iter: 50))
        s.enabled = false                            // off → never
        XCTAssertFalse(s.shouldCapture(iter: 100))
    }

    // MARK: - Frame file name (zero padded so lexicographic order = capture order)

    func testFrameFileName() {
        XCTAssertEqual(TimelapseSettings.frameFileName(index: 0), "frame-00000.jpg")
        XCTAssertEqual(TimelapseSettings.frameFileName(index: 7), "frame-00007.jpg")
        XCTAssertEqual(TimelapseSettings.frameFileName(index: 12345), "frame-12345.jpg")
        XCTAssertLessThan(TimelapseSettings.frameFileName(index: 9),
                          TimelapseSettings.frameFileName(index: 10))
    }
}
