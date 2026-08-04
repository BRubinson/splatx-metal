import XCTest
import simd
@testable import SplatXMetalTrainer

/// Viewer up-axis Task 1 (2026-07-26 spec): estimating the mean COLMAP up vector (SceneUp) +
/// generalizing CameraController and OrbitFit to an upAxis + wiring it into applyCameraFit.
/// It is a contract that the existing +Y tests (CameraControllerTests, OrbitFitTests, and
/// CameraFitTests) pass unmodified; here the equivalence of the defaults and the reproduction
/// of up/down in an up=−Y scene are pinned in addition.
final class SceneUpTests: XCTestCase {
    // An empty view list (degenerate) → the +Y fallback.
    func testEmptyViewsFallsBackToPlusY() {
        XCTAssertEqual(SceneUp.estimate(views: []), SIMD3<Float>(0, 1, 0))
    }

    // An identity pose (the COLMAP camera's +Y points down) → world up = −Y. The minimal
    // reproduction of garden/bonsai-style up≈−Y scenes.
    func testIdentityPoseGivesMinusY() {
        let v = ColmapView(id: 1, name: "a.jpg", cameraId: 1,
                           qvec: SIMD4(1, 0, 0, 0), tvec: SIMD3(0, 0, 0))
        let up = SceneUp.estimate(views: [v])
        XCTAssertEqual(simd_length(up - SIMD3<Float>(0, -1, 0)), 0, accuracy: 1e-6)
    }

    // Two views with exactly opposite ups (the sum has norm ≈ 0 — degenerate) → the +Y
    // fallback.
    // Identity (up −Y) + a 180° rotation about Z, qvec=(w:0, x:0, y:0, z:1) (up +Y).
    func testDegenerateCancellationFallsBackToPlusY() {
        let a = ColmapView(id: 1, name: "a.jpg", cameraId: 1,
                           qvec: SIMD4(1, 0, 0, 0), tvec: SIMD3(0, 0, 0))
        let b = ColmapView(id: 2, name: "b.jpg", cameraId: 1,
                           qvec: SIMD4(0, 0, 0, 1), tvec: SIMD3(0, 0, 0))
        XCTAssertEqual(SceneUp.estimate(views: [a, b]), SIMD3<Float>(0, 1, 0))
    }

    // The mean of two different ups (−Y and +Z, a 90° rotation about X) is normalized to a
    // unit vector.
    func testEstimateIsNormalizedMean() {
        let h = 0.5.squareRoot()   // cos(π/4) = sin(π/4)
        let a = ColmapView(id: 1, name: "a.jpg", cameraId: 1,
                           qvec: SIMD4(1, 0, 0, 0), tvec: SIMD3(0, 0, 0))     // up (0,−1,0)
        let b = ColmapView(id: 2, name: "b.jpg", cameraId: 1,
                           qvec: SIMD4(h, h, 0, 0), tvec: SIMD3(0, 0, 0))     // up (0,0,1)
        let up = SceneUp.estimate(views: [a, b])
        XCTAssertEqual(simd_length(up), 1, accuracy: 1e-6)
        let expected = simd_normalize(SIMD3<Float>(0, -1, 1))
        XCTAssertEqual(simd_length(up - expected), 0, accuracy: 1e-6)
    }
}

final class CameraUpAxisTests: XCTestCase {
    /// With the default upAxis (0,1,0), orbitBasis is exactly the standard basis, so eye and
    /// viewMatrix are bit-identical to the previous hard-coded +Y formulas (compared exactly,
    /// with every component of the parameters non-zero).
    func testDefaultUpAxisMatchesLegacyFormula() {
        var c = CameraController()
        c.target = SIMD3(1.5, -2, 3); c.distance = 4.2; c.yaw = 0.7; c.pitch = 0.3
        let cp = cos(c.pitch), sp = sin(c.pitch), cy = cos(c.yaw), sy = sin(c.yaw)
        let legacyDir = SIMD3<Float>(cp * sy, sp, cp * cy)
        let legacyEye = c.target + legacyDir * c.distance
        XCTAssertEqual(c.eye, legacyEye)   // bit-identical
        let legacyView = CameraController.lookAt(eye: legacyEye, center: c.target,
                                                 up: SIMD3(0, 1, 0))
        XCTAssertEqual(c.viewMatrix, legacyView)
    }

    /// orbitBasis: the default +Y is exactly the standard basis, the up≈±Z singularity is
    /// avoided by swapping the reference axis, and it is always orthonormal.
    func testOrbitBasisOrthonormal() {
        let (bx0, bu0, bz0) = CameraController.orbitBasis(upAxis: SIMD3(0, 1, 0))
        XCTAssertEqual(bx0, SIMD3<Float>(1, 0, 0))
        XCTAssertEqual(bu0, SIMD3<Float>(0, 1, 0))
        XCTAssertEqual(bz0, SIMD3<Float>(0, 0, 1))
        for up in [SIMD3<Float>(0, -1, 0), SIMD3(0, 0, 1), SIMD3(0, 0, -1),
                   simd_normalize(SIMD3<Float>(0.2, -0.9, 0.4)), SIMD3(0, -2, 0)] {
            let (bx, bu, bz) = CameraController.orbitBasis(upAxis: up)
            XCTAssertEqual(simd_length(bx), 1, accuracy: 1e-6)
            XCTAssertEqual(simd_length(bu), 1, accuracy: 1e-6)
            XCTAssertEqual(simd_length(bz), 1, accuracy: 1e-6)
            XCTAssertEqual(simd_dot(bx, bu), 0, accuracy: 1e-6)
            XCTAssertEqual(simd_dot(bu, bz), 0, accuracy: 1e-6)
            XCTAssertEqual(simd_dot(bz, bx), 0, accuracy: 1e-6)
            XCTAssertEqual(simd_length(simd_cross(bx, bu) - bz), 0, accuracy: 1e-6)  // right-handed basis
            XCTAssertEqual(simd_dot(bu, simd_normalize(up)), 1, accuracy: 1e-6)
        }
        // A norm ≈ 0 degeneracy → the +Y fallback.
        XCTAssertEqual(CameraController.orbitBasis(upAxis: .zero).up, SIMD3<Float>(0, 1, 0))
    }

    /// upAxis=−Y: the render up (the u row) of viewMatrix lies in the −Y hemisphere — with the
    /// previous behavior (hard-coded +Y) it lay in the +Y hemisphere.
    func testMinusYUpAxisRendersUpInMinusYHemisphere() {
        var c = CameraController()
        c.upAxis = SIMD3(0, -1, 0)
        c.target = .zero; c.distance = 5; c.yaw = 0.4; c.pitch = 0.2
        XCTAssertEqual(simd_length(c.eye - c.target), 5, accuracy: 1e-4)
        let v = c.viewMatrix
        let u = SIMD3<Float>(v.columns.0.y, v.columns.1.y, v.columns.2.y)
        XCTAssertGreaterThan(simd_dot(u, SIMD3(0, -1, 0)), 0.5)
    }

    /// With OrbitFit's default upAxis argument (+Y) → bit-identical to the previous inversion
    /// formulas (asin(−f.y)/atan2(−f.x,−f.z)).
    func testOrbitFitDefaultMatchesLegacy() {
        let C = SIMD3<Float>(1.5, -2.0, 3.0)
        let f = simd_normalize(SIMD3<Float>(0.3, -0.45, -0.85))
        let r = OrbitFit.fit(cameraPosition: C, forward: f, up: SIMD3(0, 1, 0),
                             pointsCentroid: SIMD3(2.0, -3.5, -1.0))
        let dir = -simd_normalize(f)
        XCTAssertEqual(r.pitch, asin(max(-1, min(1, dir.y))))
        XCTAssertEqual(r.yaw, atan2(dir.x, dir.z))
    }

    /// An up=−Y scene (the spec's core scenario): after fit + setOrbit, viewMatrix reproduces
    /// the position and gaze of the middle camera, and the render up lies in the same
    /// hemisphere as the camera up (no flip — previously dot=−1).
    func testOrbitFitRoundTripMinusYScene() {
        let upAxis = SIMD3<Float>(0, -1, 0)
        let C = SIMD3<Float>(0, 0, 0)
        let f = SIMD3<Float>(0, 0, 1)        // the gaze of a COLMAP identity pose (+Z)
        let camUp = SIMD3<Float>(0, -1, 0)   // the world up of a COLMAP identity pose (−Y)
        let r = OrbitFit.fit(cameraPosition: C, forward: f, up: camUp,
                             pointsCentroid: SIMD3(0, 0, 4), upAxis: upAxis)
        XCTAssertEqual(simd_length(r.target - SIMD3<Float>(0, 0, 4)), 0, accuracy: 1e-6)
        XCTAssertEqual(r.distance, 4, accuracy: 1e-6)
        var cam = CameraController()
        cam.upAxis = upAxis
        cam.setOrbit(target: r.target, distance: r.distance, yaw: r.yaw, pitch: r.pitch)
        XCTAssertEqual(simd_length(cam.eye - C), 0, accuracy: 1e-4)          // position reproduced
        let f2 = simd_normalize(cam.target - cam.eye)
        XCTAssertEqual(simd_length(f2 - f), 0, accuracy: 1e-5)               // gaze reproduced
        let v = cam.viewMatrix
        let u = SIMD3<Float>(v.columns.0.y, v.columns.1.y, v.columns.2.y)
        XCTAssertGreaterThan(simd_dot(u, camUp), 0.9)                        // up/down reproduced
    }
}

/// applyCameraFit wiring: estimate SceneUp → set camera.upAxis → pass it to OrbitFit (spec
/// decision 3).
@MainActor
final class CameraFitUpAxisTests: XCTestCase {
    /// A context in which every view is an identity pose (world up −Y) → camera.upAxis is set
    /// to −Y, the existing fit numbers (target, distance, and eye) show no regression, and the
    /// render up lies in the same hemisphere as the camera's actual up.
    func testApplyCameraFitAlignsUpAxis() {
        let vm = ViewerModel()
        let views = (1...3).map {
            ColmapView(id: $0, name: "f_\($0).jpg", cameraId: 7,
                       qvec: SIMD4(1, 0, 0, 0), tvec: SIMD3(0, 0, 0))
        }
        let k = ColmapIntrinsics(fx: 1440, fy: 1440, cx: 960, cy: 720, width: 1920, height: 1440)
        let ctx = ColmapContext(model: ColmapModel(cameras: [7: k], views: views),
                                middleView: views[1],
                                pointsCentroid: SIMD3(0, 0, 4))
        XCTAssertTrue(vm.applyCameraFit(context: ctx))
        XCTAssertEqual(simd_length(vm.camera.upAxis - SIMD3<Float>(0, -1, 0)), 0, accuracy: 1e-6)
        // No regression in the existing fit contract (the same numbers as CameraFitTests).
        XCTAssertEqual(simd_length(vm.camera.target - SIMD3<Float>(0, 0, 4)), 0, accuracy: 1e-5)
        XCTAssertEqual(vm.camera.distance, 4, accuracy: 1e-5)
        XCTAssertEqual(simd_length(vm.camera.eye), 0, accuracy: 1e-4)
        // Up/down alignment: render up · camera up > 0 (with the previous hard-coded +Y it
        // was −1 — flipped).
        let v = vm.camera.viewMatrix
        let u = SIMD3<Float>(v.columns.0.y, v.columns.1.y, v.columns.2.y)
        XCTAssertGreaterThan(simd_dot(u, CameraConversion.upWorld(qvec: views[1].qvec)), 0.9)
    }

    /// The fallback path is untouched: with no context → upAxis stays at the default +Y (the
    /// frameToFit and standalone PLY viewer paths).
    func testFallbackKeepsPlusYUpAxis() {
        let vm = ViewerModel()
        XCTAssertFalse(vm.applyCameraFit(context: nil))
        XCTAssertEqual(vm.camera.upAxis, SIMD3<Float>(0, 1, 0))
    }
}
