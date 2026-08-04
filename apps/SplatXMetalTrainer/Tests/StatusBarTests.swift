import XCTest
import Foundation
@testable import SplatXMetalTrainer

final class StatusBarTests: XCTestCase {
    func testThermalLabels() {
        XCTAssertEqual(StatusFormat.thermal(.nominal), "nominal")
        XCTAssertEqual(StatusFormat.thermal(.fair), "fair")
        XCTAssertEqual(StatusFormat.thermal(.serious), "serious")
        XCTAssertEqual(StatusFormat.thermal(.critical), "critical")
    }
    func testJobLabels() {
        XCTAssertEqual(JobState.idle.label, "idle")
        XCTAssertEqual(JobState.training(iter: 200, total: 7000).label, "training 200/7000")
        XCTAssertEqual(JobState.rendering.label, "rendering")
    }
}
