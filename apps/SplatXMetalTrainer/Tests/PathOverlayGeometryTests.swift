// Path transform FIX 15 (2026-07-28 spec footnote 14): pure verification of the
// camera-independent geometry cache — the ribbon point count (sampleCount+1) and its two ends,
// the **unit-depth** frustum offsets (four per key, apex == keyframes[i].pos, and
// apex + offset×d == frustumCorners(depth: d) recovered exactly), the degenerate cases of 0 and
// 1 keys, and the draw-time depth formula (dominated by arc length or camera distance, with a
// 1e-3 floor and an R̄ upper clamp).
//
// The two depth-formula cases are the regression gate for the overdraw defense FIX 7 put in
// place — they had disappeared when FIX 14 inlined the formula into the Canvas closure, and are
// restored here (footnote 14). Relaxing the upper bound, or reviving the scene bounding-sphere
// radius (which footnote 7 forbids), gets caught here.
import XCTest
import simd
@testable import SplatXMetalTrainer

final class PathOverlayGeometryTests: XCTestCase {

    private func key(_ pos: SIMD3<Float>, fov: Float = 0.9) -> Keyframe {
        Keyframe(time: 0, pos: pos, quat: SIMD4(1, 0, 0, 0), fov: fov)
    }

    private func sampler(keys: [Keyframe], loop: Bool = false,
                         duration: Double = 10) -> PathSampler {
        PathSampler(path: CameraPath(name: "t", keyframes: keys, loop: loop,
                                     duration: duration))
    }

    private func make(_ s: PathSampler?, aspect: Float = 16.0 / 9) -> PathOverlayGeometry? {
        PathOverlayGeometry.make(sampler: s, aspect: aspect,
                                 pivot: s.flatMap { PathTransform.pivot(for: $0.path.keyframes) })
    }

    func testNilSamplerAndEmptyPathProduceNil() {
        XCTAssertNil(make(nil))
        XCTAssertNil(make(sampler(keys: [])))
    }

    func testRibbonPointCountAndEndpoints() {
        let s = sampler(keys: [key(SIMD3(0, 0, 0)), key(SIMD3(4, 0, 0)),
                               key(SIMD3(4, 4, 0))])
        let g = make(s)!
        XCTAssertEqual(g.ribbonPoints.count, PathOverlayGeometry.sampleCount + 1)
        // Equal-time samples — the two ends match pose(at: 0) and pose(at: duration)
        // (exactly the Canvas formula).
        XCTAssertEqual(simd_distance(g.ribbonPoints.first!, s.pose(at: 0).pos), 0,
                       accuracy: 1e-6)
        XCTAssertEqual(simd_distance(g.ribbonPoints.last!, s.pose(at: 10).pos), 0,
                       accuracy: 1e-6)
    }

    func testSingleKeyHasNoRibbonButFrustumAndMarker() {
        let g = make(sampler(keys: [key(SIMD3(1, 2, 3))]), aspect: 1)!
        XCTAssertTrue(g.ribbonPoints.isEmpty)   // fewer than 2 keys — no ribbon (the Canvas guard)
        XCTAssertEqual(g.frusta.count, 1)
        XCTAssertEqual(g.frusta[0].cornerOffsets.count, 4)
        XCTAssertEqual(g.frusta[0].apex, SIMD3(1, 2, 3))   // apex = the diamond marker's position
        // A single key — PathTransform.pivot gives R̄ = 0 (the condition under which the
        // depth upper clamp does not apply).
        XCTAssertEqual(g.meanRadius, 0)
    }

    func testZeroDurationPathHasNoRibbon() {
        // duration ≤ 0 — equal-time sampling cannot be established (exactly the Canvas
        // guard).
        let g = make(sampler(keys: [key(SIMD3(0, 0, 0)), key(SIMD3(1, 0, 0))],
                             duration: 0))!
        XCTAssertTrue(g.ribbonPoints.isEmpty)
        XCTAssertEqual(g.frusta.count, 2)
    }

    func testFrustumOffsetsAreUnitDepthAndApexMatchesKeys() {
        let keys = [key(SIMD3(0, 0, 0)), key(SIMD3(6, 0, 0))]
        let aspect: Float = 16.0 / 9
        let g = make(sampler(keys: keys), aspect: aspect)!
        XCTAssertEqual(g.frusta.count, keys.count)
        for (k, kf) in keys.enumerated() {
            let f = g.frusta[k]
            // The apex is authoritative as keyframes[i].pos (the reference point for
            // draw-time reconstruction = the diamond's position).
            XCTAssertEqual(f.apex, kf.pos)
            XCTAssertEqual(f.cornerOffsets.count, 4)
            // Unit-depth check: apex + offset×1 == frustumCorners(depth: 1).
            let (_, corners1) = PathProjection.frustumCorners(
                position: kf.pos, rotation: kf.quatValue.normalized,
                fov: kf.fov, aspect: aspect, depth: 1)
            for j in 0..<4 {
                XCTAssertEqual(simd_distance(f.apex + f.cornerOffsets[j], corners1[j]),
                               0, accuracy: 1e-6)
            }
            // Scale reconstruction (the draw-time contract): apex + offset×d ==
            // frustumCorners(depth: d) — frustumCorners is linear in depth, so any depth is
            // recovered exactly.
            for d in [Float(0.001), 2.5, 137] {
                let (_, cornersD) = PathProjection.frustumCorners(
                    position: kf.pos, rotation: kf.quatValue.normalized,
                    fov: kf.fov, aspect: aspect, depth: d)
                for j in 0..<4 {
                    XCTAssertEqual(
                        simd_distance(f.apex + f.cornerOffsets[j] * d, cornersD[j]),
                        0, accuracy: 1e-4 * max(d, 1))
                }
            }
        }
    }

    func testAspectChangesFrustumGeometry() {
        // The frustum's aspect ratio depends on outputAspect — this is why the cache must be
        // rebuilt when the aspect ratio changes.
        let s = sampler(keys: [key(SIMD3(0, 0, 0)), key(SIMD3(2, 0, 0))])
        let wide = make(s, aspect: 16.0 / 9)!
        let square = make(s, aspect: 1)!
        XCTAssertNotEqual(wide.frusta, square.frusta)
        XCTAssertEqual(wide.ribbonPoints, square.ribbonPoints)   // the ribbon is independent of the aspect ratio
    }

    func testGeometryCarriesCameraIndependentDepthTerms() {
        let keys = [key(SIMD3(0, 0, 0)), key(SIMD3(4, 0, 0)), key(SIMD3(4, 4, 0))]
        let s = sampler(keys: keys)
        let g = make(s)!
        XCTAssertEqual(g.totalLength, s.totalLength)
        XCTAssertEqual(g.meanRadius, PathTransform.pivot(for: keys)!.meanRadius,
                       accuracy: 1e-6)
    }

    func testLoopRibbonCloses() {
        // A loop path — the last sample (pose(at: duration)) returns to the first key,
        // closing the circle.
        let s = sampler(keys: [key(SIMD3(2, 0, 0)), key(SIMD3(0, 2, 0)),
                               key(SIMD3(-2, 0, 0)), key(SIMD3(0, -2, 0))], loop: true)
        let g = make(s, aspect: 1)!
        XCTAssertEqual(simd_distance(g.ribbonPoints.first!, g.ribbonPoints.last!), 0,
                       accuracy: 1e-4)
    }

    // MARK: The depth formula (the regression gate for the FIX 7 overdraw defense — restored
    // in FIX 15)

    func testFrustumDepthFormula() {
        // Arc length dominates — the totalLength*0.02 term.
        XCTAssertEqual(PathOverlayGeometry.frustumDepth(totalLength: 100,
                                                        cameraDistance: 1,
                                                        meanRadius: 0),
                       2.0, accuracy: 1e-6)
        // A static path (arc length 0) — the cam.distance*0.03 fallback.
        XCTAssertEqual(PathOverlayGeometry.frustumDepth(totalLength: 0,
                                                        cameraDistance: 10,
                                                        meanRadius: 0),
                       0.3, accuracy: 1e-6)
        // Both are 0 — the minimum depth floor.
        XCTAssertEqual(PathOverlayGeometry.frustumDepth(totalLength: 0,
                                                        cameraDistance: 0,
                                                        meanRadius: 0),
                       1e-3, accuracy: 1e-9)
    }

    func testFrustumDepthMeanRadiusClamp() {
        // The overdraw-defense upper bound (FIX 7): however far depth runs away, it never
        // exceeds meanRadius×0.25.
        XCTAssertEqual(PathOverlayGeometry.frustumDepth(totalLength: 100,
                                                        cameraDistance: 0,
                                                        meanRadius: 4),
                       1.0, accuracy: 1e-6)   // min(2.0, 4×0.25)
        // Below the upper bound, the original formula's value passes through.
        XCTAssertEqual(PathOverlayGeometry.frustumDepth(totalLength: 100,
                                                        cameraDistance: 0,
                                                        meanRadius: 100),
                       2.0, accuracy: 1e-6)
        // meanRadius 0 (a single key) — the clamp itself does not apply (pinning it to 0
        // would make the frustum vanish).
        XCTAssertEqual(PathOverlayGeometry.frustumDepth(totalLength: 100,
                                                        cameraDistance: 0,
                                                        meanRadius: 0),
                       2.0, accuracy: 1e-6)
    }
}
