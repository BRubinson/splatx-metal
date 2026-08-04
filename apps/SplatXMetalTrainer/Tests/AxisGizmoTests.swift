import XCTest
import simd
@testable import SplatXMetalTrainer

final class AxisGizmoTests: XCTestCase {
    /// The sign invariant of each axis's screen direction for an identity-ish camera (looking
    /// at the origin from +Z).
    /// It asserts only the direction signs, not exact coordinates: +X → screen right (dir.x>0),
    /// +Y → screen up (dir.y>0).
    func testAxisScreenSigns() {
        var c = CameraController(); c.target = .zero; c.distance = 5; c.yaw = 0; c.pitch = 0
        let axes = AxisGizmo.project(view: c.viewMatrix)
        let x = axes.first { $0.axis == "X" }!, y = axes.first { $0.axis == "Y" }!
        XCTAssertGreaterThan(x.dir.x, 0)   // +X → screen right
        XCTAssertGreaterThan(y.dir.y, 0)   // +Y → screen up
    }

    /// The draw-order contract (arithmetic moved out of the view body into a pure function by
    /// path transform FIX 9): descending z (far first), and a permutation of project's result
    /// (no axis is lost or duplicated).
    func testSortedForDrawIsFarToNearPermutation() {
        var c = CameraController(); c.target = .zero; c.distance = 5; c.yaw = 0.7; c.pitch = 0.4
        let projected = AxisGizmo.project(view: c.viewMatrix)
        let sorted = AxisGizmo.sortedForDraw(view: c.viewMatrix)
        XCTAssertEqual(sorted.count, projected.count)
        XCTAssertEqual(sorted.map(\.axis).sorted(), projected.map(\.axis).sorted())
        for (a, b) in zip(sorted, sorted.dropFirst()) {
            XCTAssertGreaterThanOrEqual(a.z, b.z)   // far axes first → nearer axes are drawn on top
        }
        // Sorting does not change each axis's projected values (only the order differs).
        for entry in sorted {
            let src = projected.first { $0.axis == entry.axis }!
            XCTAssertEqual(entry.z, src.z)
            XCTAssertEqual(entry.dir, src.dir)
        }
    }
}
