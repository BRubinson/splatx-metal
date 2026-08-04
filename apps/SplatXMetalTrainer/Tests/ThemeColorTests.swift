import XCTest
import SwiftUI
@testable import SplatXMetalTrainer

final class ThemeColorTests: XCTestCase {
    func testHexInitProducesExpectedComponents() {
        // 0x2dd4bf → (45,212,191)/255
        let c = Color(hex: 0x2dd4bf)
        let ns = NSColor(c).usingColorSpace(.sRGB)!
        XCTAssertEqual(Double(ns.redComponent),   45.0/255, accuracy: 0.01)
        XCTAssertEqual(Double(ns.greenComponent), 212.0/255, accuracy: 0.01)
        XCTAssertEqual(Double(ns.blueComponent),  191.0/255, accuracy: 0.01)
    }
    func testAccentIsTeal() {
        let ns = NSColor(SplatXTheme.accent).usingColorSpace(.sRGB)!
        XCTAssertEqual(Double(ns.greenComponent), 212.0/255, accuracy: 0.02)
    }
}
