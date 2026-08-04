import XCTest
@testable import SplatXMetalTrainer

final class ConsoleLogTests: XCTestCase {
    func testCapEnforcedFIFO() {
        var log = ConsoleLog(); log.cap = 3
        for i in 1...5 { log.append("l\(i)") }
        XCTAssertEqual(log.lines, ["l3", "l4", "l5"])   // the oldest are dropped, the newest are kept
    }

    func testTotalAppendedMonotonicBeyondCap() {
        var log = ConsoleLog(); log.cap = 3
        for i in 1...5 { log.append("l\(i)") }
        XCTAssertEqual(log.lines.count, 3)      // the count stops at the cap, but
        XCTAssertEqual(log.totalAppended, 5)    // the append-detection value increases monotonically (it triggers the auto-scroll)
    }
}
