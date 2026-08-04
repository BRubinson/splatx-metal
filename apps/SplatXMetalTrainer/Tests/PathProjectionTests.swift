// Cinema Task 4: the pure projection math of the path visualization —
// world→clip→NDC→viewport, clipping behind the camera (segments are clipped against the w=minW
// plane), merging polyline runs, and deriving the frustum wireframe vertices. The Canvas views
// (PathOverlayLayer and PathGizmoOverlay) consume only these functions — with the same matrices
// as the scene render (reusing the projection×view that SplatRendererHost injects).
// The three clippedSegment cases and the four polylineRuns cases were deleted in FIX 6 (the
// move to Metal) and restored verbatim by FIX 14 along with the return to Canvas (spec
// footnote 13).
// Lineage of the literals: the camera at (0,0,5) looking at the origin, fovY 90° (y scale 1),
// aspect 1, and a 400×400 viewport.
import XCTest
import simd
@testable import SplatXMetalTrainer

final class PathProjectionTests: XCTestCase {

    private var viewProjection: float4x4 {
        let view = CameraController.lookAt(eye: SIMD3(0, 0, 5), center: .zero,
                                           up: SIMD3(0, 1, 0))
        let proj = CameraController.perspective(fovY: .pi / 2, aspect: 1,
                                                near: 0.01, far: 100)
        return proj * view
    }

    private let viewport = CGSize(width: 400, height: 400)

    func testCenterAndOffCenterProjection() {
        let center = PathProjection.viewportPoint(SIMD3(0, 0, 0),
                                                  viewProjection: viewProjection,
                                                  viewport: viewport)
        XCTAssertNotNil(center)
        XCTAssertEqual(center!.x, 200, accuracy: 1e-3)   // on the view axis → the center of the screen
        XCTAssertEqual(center!.y, 200, accuracy: 1e-3)
        // Distance 5, fovY 90° → a half-height of 5. y=+2.5 (up in world space) → NDC y=0.5
        // → viewport y=100 (toward the top).
        let up = PathProjection.viewportPoint(SIMD3(0, 2.5, 0),
                                              viewProjection: viewProjection,
                                              viewport: viewport)
        XCTAssertEqual(up!.x, 200, accuracy: 1e-3)
        XCTAssertEqual(up!.y, 100, accuracy: 1e-3)
    }

    func testBehindCameraIsNil() {
        XCTAssertNil(PathProjection.viewportPoint(SIMD3(0, 0, 10),
                                                  viewProjection: viewProjection,
                                                  viewport: viewport))
    }

    func testSegmentBothInFrontMatchesPointProjection() {
        let a = SIMD3<Float>(0, 0, 0), b = SIMD3<Float>(1, 0, 0)
        let seg = PathProjection.clippedSegment(a, b, viewProjection: viewProjection,
                                                viewport: viewport)
        XCTAssertNotNil(seg)
        let pa = PathProjection.viewportPoint(a, viewProjection: viewProjection,
                                              viewport: viewport)!
        let pb = PathProjection.viewportPoint(b, viewProjection: viewProjection,
                                              viewport: viewport)!
        XCTAssertEqual(seg!.0.x, pa.x, accuracy: 1e-6)
        XCTAssertEqual(seg!.0.y, pa.y, accuracy: 1e-6)
        XCTAssertEqual(seg!.1.x, pb.x, accuracy: 1e-6)
        XCTAssertEqual(seg!.1.y, pb.y, accuracy: 1e-6)
    }

    func testSegmentClipsBehindEndpoint() {
        // b is behind the camera — only the front half survives, and because the clipped
        // endpoint sits near the camera plane (world ≈ (0,1,5), offset upward) it projects far
        // above the screen. It must not be nil and must be finite.
        let seg = PathProjection.clippedSegment(SIMD3(0, 0, 0), SIMD3(0, 2, 10),
                                                viewProjection: viewProjection,
                                                viewport: viewport)
        XCTAssertNotNil(seg)
        XCTAssertEqual(seg!.0.x, 200, accuracy: 1e-3)    // the front endpoint is unchanged
        XCTAssertEqual(seg!.0.y, 200, accuracy: 1e-3)
        XCTAssertTrue(seg!.1.y.isFinite && seg!.1.x.isFinite)
        XCTAssertLessThan(seg!.1.y, 0)                   // extends upward (off screen)
    }

    func testSegmentBothBehindIsNil() {
        XCTAssertNil(PathProjection.clippedSegment(SIMD3(0, 1, 9), SIMD3(0, -1, 11),
                                                   viewProjection: viewProjection,
                                                   viewport: viewport))
    }

    // MARK: polylineRuns (path transform FIX 4 — merging the ribbon subpaths. Restored in
    // FIX 14)

    func testPolylineRunsAllInFrontIsSingleRun() {
        // Everything is in front of the camera → one run, with as many points as the input
        // (consecutive segments are merged).
        let pts: [SIMD3<Float>] = [SIMD3(-1, 0, 0), SIMD3(0, 0, 0),
                                   SIMD3(1, 0, 0), SIMD3(1, 1, 0)]
        let runs = PathProjection.polylineRuns(pts, viewProjection: viewProjection,
                                               viewport: viewport)
        XCTAssertEqual(runs.count, 1)
        XCTAssertEqual(runs[0].count, pts.count)
        // Merging does not change the coordinates — each point matches its standalone
        // projection.
        for (p, w) in zip(runs[0], pts) {
            let single = PathProjection.viewportPoint(w, viewProjection: viewProjection,
                                                      viewport: viewport)!
            XCTAssertEqual(p.x, single.x, accuracy: 1e-6)
            XCTAssertEqual(p.y, single.y, accuracy: 1e-6)
        }
    }

    func testPolylineRunsSplitAtBehindCameraPoint() {
        // The middle point is behind the camera (z=5) → the two front spans split into
        // separate runs.
        // Each run has 2 points: the front endpoint plus the w=minW clip point (the clip points
        // are discontinuous because their front anchors differ).
        let pts: [SIMD3<Float>] = [SIMD3(-1, 0, 0), SIMD3(0, 0, 10), SIMD3(1, 0, 0)]
        let runs = PathProjection.polylineRuns(pts, viewProjection: viewProjection,
                                               viewport: viewport)
        XCTAssertEqual(runs.count, 2)
        XCTAssertEqual(runs[0].count, 2)
        XCTAssertEqual(runs[1].count, 2)
    }

    func testPolylineRunsAllBehindIsEmpty() {
        let pts: [SIMD3<Float>] = [SIMD3(0, 0, 9), SIMD3(1, 0, 10), SIMD3(0, 1, 11)]
        XCTAssertTrue(PathProjection.polylineRuns(pts, viewProjection: viewProjection,
                                                  viewport: viewport).isEmpty)
    }

    func testPolylineRunsDegenerateInputs() {
        // 0 or 1 points → there are no segments, so there are no runs.
        XCTAssertTrue(PathProjection.polylineRuns([], viewProjection: viewProjection,
                                                  viewport: viewport).isEmpty)
        XCTAssertTrue(PathProjection.polylineRuns([SIMD3(0, 0, 0)],
                                                  viewProjection: viewProjection,
                                                  viewport: viewport).isEmpty)
    }

    func testFrustumCornersIdentityRotation() {
        // Identity rotation (forward −Z), fov 90° (half-height = depth), aspect 1, depth 2.
        let (apex, corners) = PathProjection.frustumCorners(
            position: SIMD3(1, 2, 3), rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1),
            fov: .pi / 2, aspect: 1, depth: 2)
        XCTAssertEqual(apex, SIMD3(1, 2, 3))
        XCTAssertEqual(corners.count, 4)
        let expected: [SIMD3<Float>] = [
            SIMD3(-1, 4, 1),   // tl: center(1,2,1) − right·2 + up·2
            SIMD3(3, 4, 1),    // tr
            SIMD3(3, 0, 1),    // br
            SIMD3(-1, 0, 1),   // bl
        ]
        for (c, e) in zip(corners, expected) {
            XCTAssertEqual(simd_distance(c, e), 0, accuracy: 1e-5)
        }
    }
}
