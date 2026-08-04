import XCTest
@testable import SplatXMetalTrainer

final class SharpnessScorerTests: XCTestCase {
    func testUniformImageHasZeroVariance() {
        let gray = [UInt8](repeating: 128, count: 16 * 16)
        XCTAssertEqual(SharpnessScorer.laplacianVariance(gray: gray, width: 16, height: 16), 0, accuracy: 1e-9)
    }

    func testCheckerboardSharperThanGradient() {
        let w = 16, h = 16
        var checker = [UInt8](); var ramp = [UInt8]()
        for y in 0..<h { for x in 0..<w {
            checker.append(((x + y) % 2 == 0) ? 255 : 0)
            ramp.append(UInt8(x * 255 / (w - 1)))
        } }
        let sharp = SharpnessScorer.laplacianVariance(gray: checker, width: w, height: h)
        let soft = SharpnessScorer.laplacianVariance(gray: ramp, width: w, height: h)
        XCTAssertGreaterThan(sharp, soft * 10)
    }

    func testSelectBestPerWindowPicksPeaks() {
        // 10 frames split into 2 windows (0..4 and 5..9) — the peak of each span is at index 2
        // and index 7.
        let scores = [0.1, 0.2, 0.9, 0.3, 0.1, 0.5, 0.6, 0.95, 0.2, 0.1]
        XCTAssertEqual(SharpnessScorer.selectBestPerWindow(scores: scores, windowCount: 2), [2, 7])
    }

    func testSelectReturnsAllWhenFewerFramesThanWindows() {
        XCTAssertEqual(SharpnessScorer.selectBestPerWindow(scores: [0.5, 0.1], windowCount: 5), [0, 1])
        XCTAssertEqual(SharpnessScorer.selectBestPerWindow(scores: [], windowCount: 3), [])
    }
}
