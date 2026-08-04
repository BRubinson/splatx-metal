import XCTest
@testable import SplatXMetalTrainer

final class TrainPresetTests: XCTestCase {
    /// Pins the value table: fast 3000/4/200k, balanced 7000/4/500k, quality 15000/2/1M, and
    /// ultra 30000/1/2M.
    /// (fast factor 8→4 comes from round 4 spec item 1 — an explicit reversal of round 3's
    /// "keep it unmodified".)
    func testValueTable() {
        let expected: [(TrainPreset, Int, Int, Int)] = [
            (.fast, 3_000, 4, 200_000),
            (.balanced, 7_000, 4, 500_000),
            (.quality, 15_000, 2, 1_000_000),
            (.ultra, 30_000, 1, 2_000_000),
        ]
        for (preset, iterations, factor, capMax) in expected {
            XCTAssertEqual(preset.values?.iterations, iterations, "\(preset) iterations")
            XCTAssertEqual(preset.values?.factor, factor, "\(preset) factor")
            XCTAssertEqual(preset.values?.capMax, capMax, "\(preset) capMax")
        }
        XCTAssertNil(TrainPreset.custom.values)                     // custom has no values
    }

    /// After apply, matching returns the same preset (a round trip).
    func testApplyThenMatchingRoundTrip() {
        for preset in [TrainPreset.fast, .balanced, .quality, .ultra] {
            var d = TrainOptionsDraft()
            preset.apply(to: &d)
            XCTAssertEqual(TrainPreset.matching(d), preset)
        }
    }

    /// apply touches only three fields (iterations/factor/capMax) and leaves the rest
    /// unchanged. custom.apply is a no-op.
    func testApplyTouchesOnlyThreeFields() {
        var d = TrainOptionsDraft()
        d.ssimWeight = 0.35; d.shDegree = 2; d.antialiased = true; d.testSplit = 16
        TrainPreset.fast.apply(to: &d)
        XCTAssertEqual(d.iterations, 3_000)
        XCTAssertEqual(d.ssimWeight, 0.35, accuracy: 1e-9)          // unchanged outside the three fields
        XCTAssertEqual(d.shDegree, 2)
        XCTAssertTrue(d.antialiased)
        XCTAssertEqual(d.testSplit, 16)                             // unchanged, confirmed with a value other than the default (8)

        var untouched = TrainOptionsDraft()
        TrainPreset.custom.apply(to: &untouched)                    // no-op
        XCTAssertEqual(untouched.iterations, TrainOptionsDraft().iterations)
        XCTAssertEqual(untouched.factor, TrainOptionsDraft().factor)
        XCTAssertEqual(untouched.capMax, TrainOptionsDraft().capMax)
    }

    /// Manually changing any one of the three fields makes matching == .custom (the default
    /// draft matches the balanced row).
    func testManualEditYieldsCustom() {
        XCTAssertEqual(TrainPreset.matching(TrainOptionsDraft()), .balanced)
        var d = TrainOptionsDraft(); d.iterations = 7_001
        XCTAssertEqual(TrainPreset.matching(d), .custom)
        d = TrainOptionsDraft(); d.factor = 2
        XCTAssertEqual(TrainPreset.matching(d), .custom)
        d = TrainOptionsDraft(); d.capMax = 400_000
        XCTAssertEqual(TrainPreset.matching(d), .custom)
    }
}
