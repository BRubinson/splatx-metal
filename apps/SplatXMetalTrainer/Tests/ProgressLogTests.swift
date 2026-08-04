import XCTest
@testable import SplatXMetalTrainer

final class ProgressLogTests: XCTestCase {
    func testPSNROnlyAppendedWhenPresent() {
        var log = ProgressLog()
        log.append(iteration: 1, loss: 0.5, psnr: nil)
        log.append(iteration: 2, loss: 0.4, psnr: 30.0)
        XCTAssertEqual(log.loss.count, 2)
        XCTAssertEqual(log.psnr.count, 1)
        XCTAssertEqual(log.psnr.first?.psnr, 30.0)
    }
    func testLossDownsampledToCap() {
        var log = ProgressLog(); log.maxLossPoints = 100
        for i in 1...1000 { log.append(iteration: i, loss: Double(i)) }
        XCTAssertLessThanOrEqual(log.loss.count, 100)
        XCTAssertEqual(log.loss.last?.iteration, 1000)   // the newest point is always kept
    }
}
