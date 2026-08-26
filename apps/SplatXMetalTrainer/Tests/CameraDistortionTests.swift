import XCTest
@testable import SplatXMetalTrainer

/// Tests for the COLMAP camera models, their forward distortion and the undistorted frame
/// arithmetic (`Sources/Import/CameraDistortion.swift`).
///
/// On what the round-trip test does and does not prove. Distorting a point and then inverting that
/// same function numerically recovers the point for *any* locally invertible map, so on its own it
/// cannot catch a wrong sign or a missing term — it catches non-invertibility, singularities at
/// r → 0, NaN and discontinuity, which is worth having but is not a correctness proof. The formulas
/// are therefore also pinned against evidence that does not come from themselves:
///
///   • the reference OPENCV camera's measured boundary overrun (spec §4: 10.0 / 11.2 / 22.8 / 0 px),
///     which is COLMAP's own behaviour on a real dataset;
///   • closed forms written independently of the implementation — `atan(r)/r` for the equidistant
///     fisheye, `atan(2r·tan(ω/2))/(ω·r)` and its analytic inverse `tan(ω·rd)/(2tan(ω/2))` for FOV;
///   • structural identities the implementation does not assert for itself: rotation equivariance
///     of the radial-only models, odd parity, and `.thinPrismFisheye` collapsing onto `.fisheye`.
final class CameraDistortionTests: XCTestCase {

    // MARK: - Reference camera (spec §1 — the user's actual dataset, 110 images)

    private static let referenceParams: [Double] = [
        2630.110, 2633.084, 1080, 1920,           // fx, fy, cx, cy
        0.043354, -0.079594, -0.005166, 0.000340, // k1, k2, p1, p2
    ]
    private static let referenceWidth = 2160
    private static let referenceHeight = 3840

    private func referenceCamera() throws -> (CameraIntrinsics, CameraDistortion) {
        try CameraUndistortion.parse(modelId: 4, params: Self.referenceParams)
    }

    // MARK: - Parsing

    func testParameterOrderPerModelId() throws {
        // Distinct values per slot so a transposed pair cannot pass. Orders are the interface
        // convention from colmap/4.1.0_1/include/colmap/sensor/models.h.

        // 0 SIMPLE_PINHOLE: f, cx, cy  → fx = fy = f
        var (k, d) = try CameraUndistortion.parse(modelId: 0, params: [800, 640, 360])
        XCTAssertEqual(k, CameraIntrinsics(fx: 800, fy: 800, cx: 640, cy: 360))
        XCTAssertEqual(d, .none)

        // 1 PINHOLE: fx, fy, cx, cy
        (k, d) = try CameraUndistortion.parse(modelId: 1, params: [800, 810, 640, 360])
        XCTAssertEqual(k, CameraIntrinsics(fx: 800, fy: 810, cx: 640, cy: 360))
        XCTAssertEqual(d, .none)

        // 2 SIMPLE_RADIAL: f, cx, cy, k
        (k, d) = try CameraUndistortion.parse(modelId: 2, params: [800, 640, 360, 0.11])
        XCTAssertEqual(k, CameraIntrinsics(fx: 800, fy: 800, cx: 640, cy: 360))
        XCTAssertEqual(d, .brownConrady(k1: 0.11, k2: 0, p1: 0, p2: 0, k3: 0, k4: 0, k5: 0, k6: 0))

        // 3 RADIAL: f, cx, cy, k1, k2
        (k, d) = try CameraUndistortion.parse(modelId: 3, params: [800, 640, 360, 0.11, 0.22])
        XCTAssertEqual(k, CameraIntrinsics(fx: 800, fy: 800, cx: 640, cy: 360))
        XCTAssertEqual(d, .brownConrady(k1: 0.11, k2: 0.22, p1: 0, p2: 0, k3: 0, k4: 0, k5: 0, k6: 0))

        // 4 OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
        (k, d) = try CameraUndistortion.parse(
            modelId: 4, params: [800, 810, 640, 360, 0.11, 0.22, 0.33, 0.44])
        XCTAssertEqual(k, CameraIntrinsics(fx: 800, fy: 810, cx: 640, cy: 360))
        XCTAssertEqual(d, .brownConrady(k1: 0.11, k2: 0.22, p1: 0.33, p2: 0.44,
                                        k3: 0, k4: 0, k5: 0, k6: 0))

        // 5 OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4
        (k, d) = try CameraUndistortion.parse(
            modelId: 5, params: [800, 810, 640, 360, 0.11, 0.22, 0.33, 0.44])
        XCTAssertEqual(k, CameraIntrinsics(fx: 800, fy: 810, cx: 640, cy: 360))
        XCTAssertEqual(d, .fisheye(k1: 0.11, k2: 0.22, k3: 0.33, k4: 0.44))

        // 6 FULL_OPENCV: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
        // Note the p's sit *between* k2 and k3 — the one ordering most likely to be got wrong.
        (k, d) = try CameraUndistortion.parse(
            modelId: 6,
            params: [800, 810, 640, 360, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88])
        XCTAssertEqual(k, CameraIntrinsics(fx: 800, fy: 810, cx: 640, cy: 360))
        XCTAssertEqual(d, .brownConrady(k1: 0.11, k2: 0.22, p1: 0.33, p2: 0.44,
                                        k3: 0.55, k4: 0.66, k5: 0.77, k6: 0.88))

        // 7 FOV: fx, fy, cx, cy, omega
        (k, d) = try CameraUndistortion.parse(modelId: 7, params: [800, 810, 640, 360, 0.9])
        XCTAssertEqual(k, CameraIntrinsics(fx: 800, fy: 810, cx: 640, cy: 360))
        XCTAssertEqual(d, .fov(omega: 0.9))

        // 8 SIMPLE_RADIAL_FISHEYE: f, cx, cy, k
        (k, d) = try CameraUndistortion.parse(modelId: 8, params: [800, 640, 360, 0.11])
        XCTAssertEqual(k, CameraIntrinsics(fx: 800, fy: 800, cx: 640, cy: 360))
        XCTAssertEqual(d, .fisheye(k1: 0.11, k2: 0, k3: 0, k4: 0))

        // 9 RADIAL_FISHEYE: f, cx, cy, k1, k2
        (k, d) = try CameraUndistortion.parse(modelId: 9, params: [800, 640, 360, 0.11, 0.22])
        XCTAssertEqual(k, CameraIntrinsics(fx: 800, fy: 800, cx: 640, cy: 360))
        XCTAssertEqual(d, .fisheye(k1: 0.11, k2: 0.22, k3: 0, k4: 0))

        // 10 THIN_PRISM_FISHEYE: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
        (k, d) = try CameraUndistortion.parse(
            modelId: 10,
            params: [800, 810, 640, 360, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88])
        XCTAssertEqual(k, CameraIntrinsics(fx: 800, fy: 810, cx: 640, cy: 360))
        XCTAssertEqual(d, .thinPrismFisheye(k1: 0.11, k2: 0.22, p1: 0.33, p2: 0.44,
                                            k3: 0.55, k4: 0.66, sx1: 0.77, sy1: 0.88))
    }

    func testParameterCountsMatchLoaderTable() {
        // Same table as ColmapLoader.cpp:48-63 — the app and the core must agree on how many doubles
        // a camera record carries, or the app would mis-parse what the core happily reads.
        let expected: [Int: Int] = [0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8,
                                    6: 12, 7: 5, 8: 4, 9: 5, 10: 12]
        for model in ColmapCameraModel.allCases {
            XCTAssertEqual(model.parameterCount, expected[model.rawValue],
                           "\(model.name) parameter count")
        }
        XCTAssertEqual(ColmapCameraModel.allCases.count, 11)
    }

    func testModelNamesRoundTrip() {
        for model in ColmapCameraModel.allCases {
            XCTAssertEqual(ColmapCameraModel(name: model.name), model)
        }
        XCTAssertNil(ColmapCameraModel(name: "RAD_TAN_THIN_PRISM_FISHEYE"))
        XCTAssertNil(ColmapCameraModel(name: "SIMPLE_DIVISION"))
    }

    func testUnsupportedModelIdThrows() {
        // COLMAP 4.1 defines 11+ (RAD_TAN_THIN_PRISM_FISHEYE, SIMPLE_DIVISION, …). Spec §3.4: these
        // fail loudly rather than being approximated by a neighbouring model.
        for id in [11, 12, 13, 14, 15, 16, 99, -1] {
            XCTAssertThrowsError(try CameraUndistortion.parse(
                modelId: id, params: Array(repeating: 1.0, count: 16))) { error in
                XCTAssertEqual(error as? CameraModelError, .unsupportedModel(id: id))
            }
        }
    }

    func testParameterCountMismatchThrows() {
        XCTAssertThrowsError(try CameraUndistortion.parse(modelId: 4, params: [1, 2, 3, 4])) { error in
            XCTAssertEqual(error as? CameraModelError,
                           .parameterCountMismatch(model: "OPENCV", expected: 8, found: 4))
        }
        // One too many is rejected as firmly as one too few — a longer array would otherwise be
        // silently truncated and the extra coefficient dropped.
        XCTAssertThrowsError(try CameraUndistortion.parse(
            modelId: 2, params: [800, 640, 360, 0.1, 0.2])) { error in
            XCTAssertEqual(error as? CameraModelError,
                           .parameterCountMismatch(model: "SIMPLE_RADIAL", expected: 4, found: 5))
        }
        XCTAssertThrowsError(try CameraUndistortion.parse(modelId: 0, params: []))
    }

    func testNonPositiveFocalThrows() {
        XCTAssertThrowsError(try CameraUndistortion.parse(modelId: 0, params: [0, 640, 360]))
        XCTAssertThrowsError(try CameraUndistortion.parse(modelId: 1, params: [800, -1, 640, 360]))
    }

    // MARK: - Zero coefficients

    func testZeroCoefficientsAreIdentityForPerspectiveModels() throws {
        // Models 0, 1, 2, 3, 4, 6 and 7. For these the distortion polynomial is a correction added
        // to the identity, so zero coefficients leave the point where it was.
        let cases: [(Int, [Double])] = [
            (0, [800, 640, 360]),
            (1, [800, 810, 640, 360]),
            (2, [800, 640, 360, 0]),
            (3, [800, 640, 360, 0, 0]),
            (4, [800, 810, 640, 360, 0, 0, 0, 0]),
            (6, [800, 810, 640, 360, 0, 0, 0, 0, 0, 0, 0, 0]),
            (7, [800, 810, 640, 360, 0]),   // ω → 0 is the FOV model's identity limit
        ]
        for (id, params) in cases {
            let (_, d) = try CameraUndistortion.parse(modelId: id, params: params)
            XCTAssertTrue(d.isIdentity, "model \(id) with zero coefficients should be identity")
            for (x, y) in Self.normalizedGrid(extent: 0.8, steps: 8) {
                let (xd, yd) = d.distort(x, y)
                XCTAssertEqual(xd, x, accuracy: 1e-12, "model \(id) at (\(x), \(y))")
                XCTAssertEqual(yd, y, accuracy: 1e-12, "model \(id) at (\(x), \(y))")
            }
        }
    }

    func testZeroCoefficientFisheyeIsEquidistantNotIdentity() throws {
        // Correction to the plan's "zero coefficients ⇒ identity for every model": for ids 5, 8, 9
        // and 10 that is false and must be false. θ = atan(r) is the fisheye *projection*, not the
        // distortion polynomial, so a fisheye camera with every k at zero is a perfect equidistant
        // fisheye and still has to be re-projected. Asserting identity here would enshrine a bug
        // that leaves fisheye captures unrectified.
        //
        // The expectation is the independently written closed form x·atan(r)/r.
        let cases: [(Int, [Double])] = [
            (5, [800, 810, 640, 360, 0, 0, 0, 0]),
            (8, [800, 640, 360, 0]),
            (9, [800, 640, 360, 0, 0]),
            (10, [800, 810, 640, 360, 0, 0, 0, 0, 0, 0, 0, 0]),
        ]
        for (id, params) in cases {
            let (_, d) = try CameraUndistortion.parse(modelId: id, params: params)
            XCTAssertFalse(d.isIdentity, "fisheye model \(id) is never the identity")
            for (x, y) in Self.normalizedGrid(extent: 1.2, steps: 8) {
                let r = (x * x + y * y).squareRoot()
                let scale = r > 0 ? atan(r) / r : 1
                let (xd, yd) = d.distort(x, y)
                XCTAssertEqual(xd, x * scale, accuracy: 1e-13, "model \(id) at (\(x), \(y))")
                XCTAssertEqual(yd, y * scale, accuracy: 1e-13, "model \(id) at (\(x), \(y))")
            }
            // …and it really does move points: at 45° off axis the equidistant map pulls in by ~21%.
            let (xd, _) = d.distort(1, 0)
            XCTAssertEqual(xd, atan(1.0), accuracy: 1e-13)
            XCTAssertLessThan(xd, 0.8)
        }
    }

    func testPrincipalPointIsFixedByEveryModel() throws {
        // The r → 0 guards must not change the value they guard: (0,0) maps to (0,0) exactly.
        for (id, params) in Self.allModelFixtures {
            let (_, d) = try CameraUndistortion.parse(modelId: id, params: params)
            let (xd, yd) = d.distort(0, 0)
            XCTAssertEqual(xd, 0, "model \(id)")
            XCTAssertEqual(yd, 0, "model \(id)")
            XCTAssertFalse(xd.isNaN || yd.isNaN, "model \(id) produced NaN at the principal point")

            // Approaching the origin has to stay continuous — in particular the r → 0 guards must
            // not step to a different scale than the limit they stand in for. Note the limiting
            // scale is *not* 1 for every model: FOV tends to 2tan(ω/2)/ω. So what is asserted is
            // that the scale agrees across four decades either side of the guard's threshold,
            // whatever that shared value is.
            var scales: [Double] = []
            for r in [1e-3, 1e-6, 1e-9, 1e-11, 1e-13] {
                let (a, b) = d.distort(r, 0)
                XCTAssertFalse(a.isNaN || b.isNaN, "model \(id) NaN at r = \(r)")
                scales.append(a / r)
            }
            for s in scales {
                XCTAssertEqual(s, scales[0], accuracy: 1e-4,
                               "model \(id) scale is discontinuous near the principal point")
            }
        }
    }

    // MARK: - Round trip (forward → numerical inverse → origin)

    func testRoundTripAllTenModels() throws {
        for (id, params) in Self.allModelFixtures {
            let (_, d) = try CameraUndistortion.parse(modelId: id, params: params)
            var checked = 0
            for (x, y) in Self.normalizedGrid(extent: 0.8, steps: 8) {
                let (xd, yd) = d.distort(x, y)
                guard let (xr, yr) = Self.newtonInverse(d, xd, yd) else {
                    XCTFail("model \(id): Newton failed to invert at (\(x), \(y))")
                    continue
                }
                XCTAssertEqual(xr, x, accuracy: 1e-9, "model \(id) round trip x at (\(x), \(y))")
                XCTAssertEqual(yr, y, accuracy: 1e-9, "model \(id) round trip y at (\(x), \(y))")
                checked += 1
            }
            XCTAssertEqual(checked, 81, "model \(id): every grid point must be exercised")
        }
    }

    func testRoundTripFisheyeAtWideAngles() throws {
        // Fisheye is the family that is actually asked to work far off axis. r = 3 is 71.6°, well
        // past where a perspective model would be usable, and the point of the θ = atan(r) map.
        for (id, params) in Self.allModelFixtures where [5, 8, 9, 10].contains(id) {
            let (_, d) = try CameraUndistortion.parse(modelId: id, params: params)
            for (x, y) in Self.normalizedGrid(extent: 3.0, steps: 6) {
                let (xd, yd) = d.distort(x, y)
                XCTAssertTrue(xd.isFinite && yd.isFinite, "model \(id) at (\(x), \(y))")
                guard let (xr, yr) = Self.newtonInverse(d, xd, yd) else {
                    XCTFail("model \(id): Newton failed at wide angle (\(x), \(y))")
                    continue
                }
                XCTAssertEqual(xr, x, accuracy: 1e-9, "model \(id) wide x at (\(x), \(y))")
                XCTAssertEqual(yr, y, accuracy: 1e-9, "model \(id) wide y at (\(x), \(y))")
            }
            // Monotone and contracting along the axis all the way out — a fisheye that folded back
            // would still round-trip locally but would be unusable.
            var previous = 0.0
            for i in 1...200 {
                let r = Double(i) * 0.05
                let (xd, _) = d.distort(r, 0)
                XCTAssertGreaterThan(xd, previous, "model \(id) radial profile not monotone at \(r)")
                XCTAssertLessThan(xd, r, "model \(id) should contract at r = \(r)")
                previous = xd
            }
        }
    }

    // MARK: - Independent closed forms and structural identities

    func testFOVMatchesDevernayFaugerasClosedFormAndItsAnalyticInverse() {
        // rd = atan(2·r·tan(ω/2)) / ω, computed here straight from the paper's statement rather
        // than through the implementation's rearrangement, plus the exact inverse
        // r = tan(ω·rd) / (2·tan(ω/2)) — a genuine check, not a numerical round trip.
        for omega in [0.2, 0.6, 0.9, 1.3] {
            let d = CameraDistortion.fov(omega: omega)
            for r in stride(from: 0.05, through: 1.6, by: 0.05) {
                let expected = atan(2 * r * tan(omega / 2)) / omega
                let (xd, yd) = d.distort(r, 0)
                XCTAssertEqual(xd, expected, accuracy: 1e-12, "ω = \(omega), r = \(r)")
                XCTAssertEqual(yd, 0, accuracy: 1e-15)
                let back = tan(omega * expected) / (2 * tan(omega / 2))
                XCTAssertEqual(back, r, accuracy: 1e-10, "analytic inverse at ω = \(omega)")
            }
            // The r → 0 limit of the scale is 2tan(ω/2)/ω, not 1 — the field has no step at the
            // principal point. (A "scale → 1" guard would show up here as a discontinuity.)
            let tiny = 1e-9
            let (xs, _) = d.distort(tiny, 0)
            XCTAssertEqual(xs / tiny, 2 * tan(omega / 2) / omega, accuracy: 1e-6, "ω = \(omega)")
        }
    }

    func testFisheyeMatchesKannalaBrandtClosedForm() {
        let (k1, k2, k3, k4) = (-0.02, 0.004, -0.0006, 0.00008)
        let d = CameraDistortion.fisheye(k1: k1, k2: k2, k3: k3, k4: k4)
        for r in stride(from: 0.05, through: 3.0, by: 0.05) {
            let theta = atan(r)
            // Written out term by term rather than in Horner form, so a Horner slip would show.
            let thetaD = theta * (1
                + k1 * pow(theta, 2) + k2 * pow(theta, 4)
                + k3 * pow(theta, 6) + k4 * pow(theta, 8))
            let (xd, _) = d.distort(r, 0)
            XCTAssertEqual(xd, thetaD, accuracy: 1e-12, "r = \(r)")
        }
    }

    func testBrownConradyRadialOnlyIsRotationEquivariant() {
        // With the tangential terms off, distortion is a pure function of radius, so it must commute
        // with rotation about the principal point. A misplaced x/y in the radial term breaks this.
        let d = CameraDistortion.brownConrady(k1: 0.09, k2: -0.03, p1: 0, p2: 0,
                                              k3: 0.004, k4: 0.02, k5: -0.01, k6: 0.002)
        for r in stride(from: 0.1, through: 0.9, by: 0.1) {
            let (baseX, _) = d.distort(r, 0)
            for degrees in stride(from: 0.0, to: 360.0, by: 17.0) {
                let a = degrees * .pi / 180
                let (xd, yd) = d.distort(r * cos(a), r * sin(a))
                XCTAssertEqual((xd * xd + yd * yd).squareRoot(), abs(baseX), accuracy: 1e-12)
                // Direction is preserved too (no swirl).
                XCTAssertEqual(atan2(yd, xd), a > .pi ? a - 2 * .pi : a, accuracy: 1e-12)
            }
        }
    }

    func testRadialOnlyModelsAreOddParity() {
        // D(−p) = −D(p) for every purely radial model. Any stray even term (a misplaced tangential
        // contribution, an absolute value) violates it.
        let radialOnly: [CameraDistortion] = [
            .brownConrady(k1: 0.09, k2: -0.03, p1: 0, p2: 0, k3: 0, k4: 0, k5: 0, k6: 0),
            .fov(omega: 0.9),
            .fisheye(k1: -0.02, k2: 0.004, k3: -0.0006, k4: 0.00008),
        ]
        for d in radialOnly {
            for (x, y) in Self.normalizedGrid(extent: 0.9, steps: 6) {
                let (xa, ya) = d.distort(x, y)
                let (xb, yb) = d.distort(-x, -y)
                XCTAssertEqual(xb, -xa, accuracy: 1e-14)
                XCTAssertEqual(yb, -ya, accuracy: 1e-14)
            }
        }
    }

    func testBrownConradyTangentialTermsMatchHandComputation() {
        // A single hand-evaluated point isolating the decentering terms (all k at zero), checked
        // against arithmetic done outside the implementation:
        //   x = 0.3, y = -0.2, p1 = 0.01, p2 = -0.02
        //   r² = 0.09 + 0.04 = 0.13,  xy = -0.06
        //   xd = 0.3 + 2(0.01)(−0.06) + (−0.02)(0.13 + 2·0.09) = 0.3 − 0.0012 − 0.0062  = 0.2926
        //   yd = −0.2 + (0.01)(0.13 + 2·0.04) + 2(−0.02)(−0.06) = −0.2 + 0.0021 + 0.0024 = −0.1955
        let d = CameraDistortion.brownConrady(k1: 0, k2: 0, p1: 0.01, p2: -0.02,
                                              k3: 0, k4: 0, k5: 0, k6: 0)
        let (xd, yd) = d.distort(0.3, -0.2)
        XCTAssertEqual(xd, 0.2926, accuracy: 1e-12)
        XCTAssertEqual(yd, -0.1955, accuracy: 1e-12)
    }

    func testFullOpenCVUsesTheRationalDenominator() {
        // k4…k6 exist only in FULL_OPENCV and only in the denominator. Dropping them (the easy bug,
        // since the other four models parse them as zero) would make these two agree.
        let numeratorOnly = CameraDistortion.brownConrady(
            k1: 0.05, k2: -0.03, p1: 0, p2: 0, k3: 0.004, k4: 0, k5: 0, k6: 0)
        let rational = CameraDistortion.brownConrady(
            k1: 0.05, k2: -0.03, p1: 0, p2: 0, k3: 0.004, k4: 0.02, k5: -0.01, k6: 0.002)
        let (a, _) = numeratorOnly.distort(0.8, 0)
        let (b, _) = rational.distort(0.8, 0)
        XCTAssertNotEqual(a, b, accuracy: 1e-6)
        // Explicit expectation, computed outside the implementation:
        //   r² = 0.64
        //   num = 1 + 0.05·0.64 + (−0.03)·0.4096 + 0.004·0.262144
        //   den = 1 + 0.02·0.64 + (−0.01)·0.4096 + 0.002·0.262144
        let r2 = 0.64
        let num = 1 + 0.05 * r2 + (-0.03) * r2 * r2 + 0.004 * r2 * r2 * r2
        let den = 1 + 0.02 * r2 + (-0.01) * r2 * r2 + 0.002 * r2 * r2 * r2
        XCTAssertEqual(b, 0.8 * num / den, accuracy: 1e-14)
    }

    func testThinPrismCollapsesOntoFisheyeWhenTangentialAndPrismAreZero() {
        // Structural consistency of the THIN_PRISM_FISHEYE reading chosen in the implementation:
        // with p and s at zero it must be exactly the Kannala–Brandt fisheye with the same k's.
        let plain = CameraDistortion.fisheye(k1: -0.02, k2: 0.004, k3: -0.0006, k4: 0.00008)
        let prism = CameraDistortion.thinPrismFisheye(k1: -0.02, k2: 0.004, p1: 0, p2: 0,
                                                      k3: -0.0006, k4: 0.00008, sx1: 0, sy1: 0)
        for (x, y) in Self.normalizedGrid(extent: 2.0, steps: 8) {
            let (ax, ay) = plain.distort(x, y)
            let (bx, by) = prism.distort(x, y)
            XCTAssertEqual(bx, ax, accuracy: 1e-14, "at (\(x), \(y))")
            XCTAssertEqual(by, ay, accuracy: 1e-14, "at (\(x), \(y))")
        }
        // And the prism terms are a pure radial-squared offset: sx1·ρ², sy1·ρ² with ρ = θ.
        let onlyPrism = CameraDistortion.thinPrismFisheye(k1: 0, k2: 0, p1: 0, p2: 0,
                                                          k3: 0, k4: 0, sx1: 0.05, sy1: -0.03)
        let equidistant = CameraDistortion.fisheye(k1: 0, k2: 0, k3: 0, k4: 0)
        for (x, y) in Self.normalizedGrid(extent: 1.5, steps: 5) {
            let theta = atan((x * x + y * y).squareRoot())
            let (bx, by) = equidistant.distort(x, y)
            let (px, py) = onlyPrism.distort(x, y)
            XCTAssertEqual(px, bx + 0.05 * theta * theta, accuracy: 1e-14)
            XCTAssertEqual(py, by - 0.03 * theta * theta, accuracy: 1e-14)
        }
    }

    // MARK: - Reference camera against COLMAP's measured behaviour

    func testReferenceCameraBoundaryOverrunMatchesSpec() throws {
        // Spec §4's measurement of this exact dataset: the same-size undistorted frame overruns the
        // source by 10.0px left, 11.2px right, 22.8px top and 0px bottom. Those four numbers come
        // from the real camera and COLMAP's real output, so reproducing them tests the
        // Brown–Conrady implementation against something other than itself.
        let (k, d) = try referenceCamera()
        let w = Double(Self.referenceWidth), h = Double(Self.referenceHeight)
        var minU = Double.infinity, maxU = -Double.infinity
        var minV = Double.infinity, maxV = -Double.infinity
        let n = 2000
        for i in 0...n {
            let t = Double(i) / Double(n)
            for (outU, outV) in [(t * w, 0.0), (t * w, h), (0.0, t * h), (w, t * h)] {
                let x = (outU - w / 2) / k.fx
                let y = (outV - h / 2) / k.fy
                let (xd, yd) = d.distort(x, y)
                let u = k.fx * xd + k.cx
                let v = k.fy * yd + k.cy
                minU = min(minU, u); maxU = max(maxU, u)
                minV = min(minV, v); maxV = max(maxV, v)
            }
        }
        XCTAssertEqual(-minU, 10.0, accuracy: 0.1, "left overrun")
        XCTAssertEqual(maxU - w, 11.2, accuracy: 0.1, "right overrun")
        XCTAssertEqual(-minV, 22.8, accuracy: 0.1, "top overrun")
        XCTAssertLessThanOrEqual(maxV - h, 0.0, "bottom must not overrun")
    }

    func testReferenceCameraDisplacementIsTensOfPixels() throws {
        // Spec §1 puts the error from reading this camera as plain PINHOLE at "about 20px". Measured
        // per position it is not one number: the corners run 7.2 / 7.9 / 45.0 / 43.3 px (the two
        // bottom ones dominated by p1 = −0.005166 at |y| ≈ 0.73), and the midpoint of the top edge
        // is 22.77px — which is exactly spec §4's independently measured 22.8px top overrun. So §1's
        // figure is an order of magnitude, and the top-edge value is the one that is pinned.
        let (k, d) = try referenceCamera()
        let w = Double(Self.referenceWidth), h = Double(Self.referenceHeight)

        func displacement(_ u: Double, _ v: Double) -> Double {
            let (xd, yd) = d.distort((u - k.cx) / k.fx, (v - k.cy) / k.fy)
            return ((k.fx * xd + k.cx - u) * (k.fx * xd + k.cx - u)
                    + (k.fy * yd + k.cy - v) * (k.fy * yd + k.cy - v)).squareRoot()
        }

        XCTAssertEqual(displacement(w / 2, 0.5), 22.77, accuracy: 0.05, "top edge midpoint")
        let corners = [displacement(0.5, 0.5), displacement(w - 0.5, 0.5),
                       displacement(0.5, h - 0.5), displacement(w - 0.5, h - 0.5)]
        for c in corners {
            XCTAssertGreaterThan(c, 5.0)
            XCTAssertLessThan(c, 50.0)
        }
        XCTAssertEqual(corners[0], 7.17, accuracy: 0.05)
        XCTAssertEqual(corners[3], 43.32, accuracy: 0.05)
        // The centre is essentially untouched — the error is a rim effect, which is why the naive
        // "just read it as PINHOLE" failure mode is easy to miss on a thumbnail.
        XCTAssertLessThan(displacement(k.cx, k.cy), 0.01)
    }

    // MARK: - Undistorted frame

    func testUndistortedFrameReproducesColmapForReferenceCamera() throws {
        // COLMAP's image_undistorter emits 2137×3792 for this dataset with fx, fy unchanged and
        // cx, cy = 1068.5, 1896.0. Spec §4 allows ±2px.
        let (k, d) = try referenceCamera()
        let frame = CameraUndistortion.undistortedFrame(
            intrinsics: k, distortion: d,
            width: Self.referenceWidth, height: Self.referenceHeight)

        XCTAssertLessThanOrEqual(abs(frame.width - 2137), 2, "width was \(frame.width)")
        XCTAssertLessThanOrEqual(abs(frame.height - 3792), 2, "height was \(frame.height)")
        XCTAssertEqual(frame.cx, Double(frame.width) / 2)
        XCTAssertEqual(frame.cy, Double(frame.height) / 2)
        // Not rounded up to even — COLMAP's own width for this camera is the odd 2137.
        XCTAssertLessThan(frame.width, Self.referenceWidth)
        XCTAssertLessThan(frame.height, Self.referenceHeight)
    }

    func testUndistortedFrameLeavesNoOutOfBoundsSamples() throws {
        // The invariant the whole rule exists for: every pixel of the returned frame samples inside
        // the source, so the rectified image has no black rim to train on. Checked on the reference
        // camera and on one of each family.
        for (id, params, w, h) in Self.frameFixtures {
            let (k, d) = try CameraUndistortion.parse(modelId: id, params: params)
            let frame = CameraUndistortion.undistortedFrame(
                intrinsics: k, distortion: d, width: w, height: h)
            XCTAssertGreaterThan(frame.width, 0, "model \(id)")
            XCTAssertGreaterThan(frame.height, 0, "model \(id)")
            // Never grown: a pinhole frame wider than the source would be inventing resolution, and
            // for fisheye it is what bounds the otherwise divergent field of view.
            XCTAssertLessThanOrEqual(frame.width, w, "model \(id) must not grow width")
            XCTAssertLessThanOrEqual(frame.height, h, "model \(id) must not grow height")

            var worst = 0.0
            for j in 0..<frame.height {
                for i in [0, frame.width / 2, frame.width - 1] {
                    let x = (Double(i) + 0.5 - frame.cx) / k.fx
                    let y = (Double(j) + 0.5 - frame.cy) / k.fy
                    let (xd, yd) = d.distort(x, y)
                    let u = k.fx * xd + k.cx
                    let v = k.fy * yd + k.cy
                    worst = max(worst, max(-u, u - Double(w), -v, v - Double(h)))
                    XCTAssertTrue(u >= 0 && u <= Double(w) && v >= 0 && v <= Double(h),
                                  "model \(id): pixel (\(i), \(j)) samples (\(u), \(v)) outside "
                                  + "\(w)×\(h)")
                }
            }
            XCTAssertLessThanOrEqual(worst, 0.0, "model \(id)")
        }
    }

    func testIdentityDistortionKeepsTheFrameExactly() throws {
        // Zero distortion must not shave a pixel — ids 0 and 1 pass through untouched, and so does
        // any model whose coefficients happen to all be zero.
        for (id, params) in [(0, [800.0, 640, 360]),
                             (1, [800.0, 810, 640, 360]),
                             (4, [800.0, 810, 640, 360, 0, 0, 0, 0]),
                             (7, [800.0, 810, 640, 360, 0])] {
            let (k, d) = try CameraUndistortion.parse(modelId: id, params: params)
            let frame = CameraUndistortion.undistortedFrame(
                intrinsics: k, distortion: d, width: 1280, height: 720)
            XCTAssertEqual(frame.width, 1280, "model \(id)")
            XCTAssertEqual(frame.height, 720, "model \(id)")
            XCTAssertEqual(frame.cx, 640, "model \(id)")
            XCTAssertEqual(frame.cy, 360, "model \(id)")
        }
    }

    func testIdentityDistortionLeavesAnOffCentrePrincipalPointWhereItIs() throws {
        // The regression 2026-08-15 FIX 2 exists for, and the reason the test above did not catch
        // it: every fixture there — like `ImageUndistorterTests`' 64×48 at cx 32 — has its principal
        // point exactly at `W/2, H/2`, which is the one case where "carry the camera over" and
        // "re-center on W/2" give the same answer.
        //
        // 800×600 with the principal point at (470, 250) is 70 px and 50 px off centre. Returning
        // `W/2, H/2` shifted the sampling map by exactly that, and 800·600 − 730·550 = 78,500 output
        // pixels (16.4 %) came back black.
        for (id, params) in [(0, [900.0, 470, 250]),
                             (1, [900.0, 900, 470, 250]),
                             (2, [900.0, 470, 250, 0]),
                             (3, [900.0, 470, 250, 0, 0]),
                             (4, [900.0, 900, 470, 250, 0, 0, 0, 0]),
                             (6, [900.0, 900, 470, 250, 0, 0, 0, 0, 0, 0, 0, 0]),
                             (7, [900.0, 900, 470, 250, 0])] {
            let (k, d) = try CameraUndistortion.parse(modelId: id, params: params)
            XCTAssertTrue(d.isIdentity, "model \(id) fixture must be an identity distortion")
            let frame = CameraUndistortion.undistortedFrame(
                intrinsics: k, distortion: d, width: 800, height: 600)
            XCTAssertEqual(frame.width, 800, "model \(id)")
            XCTAssertEqual(frame.height, 600, "model \(id)")
            XCTAssertEqual(frame.cx, 470, "model \(id) must keep its own cx, not take W/2")
            XCTAssertEqual(frame.cy, 250, "model \(id) must keep its own cy, not take H/2")
        }
    }

    func testAnOffCentrePrincipalPointSamplesInsideTheSourceWithAndWithoutCoefficients() throws {
        // The invariant that matters, stated across the discontinuity. A zero-coefficient camera
        // keeps its full frame; the same camera with `k1 = 1e-9` is re-centered by the solver and
        // comes back smaller (measured 658×498 — a centered frame inside a source whose axis sits
        // at 470 of 800 cannot be wider than 2·(800−470)). The sizes differ by design; what neither
        // may do is read outside the capture, because that is what fills training images with black.
        for params in [[900.0, 900, 470, 250, 0, 0, 0, 0],
                       [900.0, 900, 470, 250, 1e-9, 0, 0, 0]] {
            let (k, d) = try CameraUndistortion.parse(modelId: 4, params: params)
            let frame = CameraUndistortion.undistortedFrame(
                intrinsics: k, distortion: d, width: 800, height: 600)
            for (i, j) in boundaryPixels(width: frame.width, height: frame.height) {
                let x = (Double(i) + 0.5 - frame.cx) / k.fx
                let y = (Double(j) + 0.5 - frame.cy) / k.fy
                let (xd, yd) = d.distort(x, y)
                let u = k.fx * xd + k.cx, v = k.fy * yd + k.cy
                XCTAssertTrue(u >= 0 && u <= 800 && v >= 0 && v <= 600,
                              "k1 = \(params[4]): output pixel (\(i), \(j)) samples (\(u), \(v))")
            }
        }
    }

    // MARK: - The frame floor

    func testFrameIsUsableRejectsACollapsedFrameAndAcceptsASmallOne() throws {
        // Spec §1.1's failure shape, caught at the arithmetic. A strong pincushion genuinely leaves
        // no distortion-free frame inside the capture — the solver is not wrong, it is answering
        // truthfully — and without a floor those answers were re-encoded as 1×1 JPEGs that only
        // failed minutes later, inside the core's SSIM validator.
        //
        // Measured on 320×240 SIMPLE_RADIAL at f = 160. `k = 0.7` is a coefficient COLMAP itself
        // accepts: its `max_extra_param` default is 1.0.
        let sweep: [(Double, Int, Int, Bool)] = [(0.4, 121, 90, true),
                                                 (0.5, 71, 53, true),
                                                 (0.6, 22, 16, false),
                                                 (0.7, 1, 1, false)]
        for (k, width, height, usable) in sweep {
            let (i, d) = try CameraUndistortion.parse(modelId: 2, params: [160, 160, 120, k])
            let frame = CameraUndistortion.undistortedFrame(
                intrinsics: i, distortion: d, width: 320, height: 240)
            XCTAssertEqual(frame.width, width, "k = \(k) width")
            XCTAssertEqual(frame.height, height, "k = \(k) height")
            XCTAssertEqual(CameraUndistortion.frameIsUsable(width: frame.width,
                                                            height: frame.height,
                                                            sourceWidth: 320, sourceHeight: 240),
                           usable, "k = \(k)")
        }
    }

    func testFrameIsUsableAppliesBothFloorsIndependently() {
        // The two thresholds catch different things, so each needs a case that only it rejects.
        // 22×16 out of 320×240 clears the 11 px side floor and fails the 0.2 scale floor (COLMAP's
        // own `min_scale`); 15×10 out of 30×22 is half the source — well past the scale floor — and
        // fails on the 10 px side, below the core's 11 px SSIM window.
        XCTAssertFalse(CameraUndistortion.frameIsUsable(width: 22, height: 16,
                                                        sourceWidth: 320, sourceHeight: 240))
        XCTAssertFalse(CameraUndistortion.frameIsUsable(width: 15, height: 10,
                                                        sourceWidth: 30, sourceHeight: 22))
        // And the floors are floors, not margins: exactly 11 px and exactly a fifth both pass.
        XCTAssertTrue(CameraUndistortion.frameIsUsable(width: 11, height: 11,
                                                       sourceWidth: 55, sourceHeight: 55))
        XCTAssertTrue(CameraUndistortion.frameIsUsable(width: 2136, height: 3792,
                                                       sourceWidth: 2160, sourceHeight: 3840))
    }

    func testFisheyeFrameKeepsSourceSizeAndLosesFieldOfView() throws {
        // Spec §3.3: fisheye → pinhole is lossy by construction. The equidistant map contracts, so
        // the mapped boundary lands well inside the source and no crop is needed — the frame comes
        // back at source size, and what is lost is the outer field of view, not pixels off the rim.
        let (k, d) = try CameraUndistortion.parse(
            modelId: 5, params: [800, 800, 960, 540, -0.02, 0.004, -0.0006, 0.00008])
        let frame = CameraUndistortion.undistortedFrame(
            intrinsics: k, distortion: d, width: 1920, height: 1080)
        XCTAssertEqual(frame.width, 1920)
        XCTAssertEqual(frame.height, 1080)

        // Half-angle the pinhole output covers along +x: its outermost column, straightened.
        let pinholeHalfAngle = atan((Double(frame.width) / 2) / k.fx)
        // Half-angle the fisheye source actually recorded along +x: take its outermost column in
        // distorted normalized coordinates and invert the radial profile to get the incident ray.
        let sourceEdgeDistorted = (1920.0 - 0.5 - k.cx) / k.fx
        guard let (xr, _) = Self.newtonInverse(d, sourceEdgeDistorted, 0) else {
            return XCTFail("could not invert the fisheye profile at the source edge")
        }
        let capturedHalfAngle = atan(xr)
        XCTAssertGreaterThan(capturedHalfAngle, pinholeHalfAngle,
                             "a fisheye source must have recorded wider than the pinhole output")
        // The loss is large, not marginal — about 70° recorded against 50° kept. This is spec §3.3's
        // intrinsic limit, and the import path reports it so it is not read as a defect.
        XCTAssertGreaterThan(capturedHalfAngle - pinholeHalfAngle, 0.3)
    }

    func testFrameSolverVariesSmoothlyWithSourceSize() throws {
        // The solved size must be a property of the camera, not of the sampler. Approached from the
        // other side: perturbing the source size by a pixel moves the answer by about a pixel, with
        // no jumps that would betray a sampling artefact.
        let (k, d) = try referenceCamera()
        var previous: Int? = nil
        for delta in 0...4 {
            let frame = CameraUndistortion.undistortedFrame(
                intrinsics: k, distortion: d,
                width: Self.referenceWidth + delta, height: Self.referenceHeight)
            if let p = previous { XCTAssertLessThanOrEqual(abs(frame.width - p), 2) }
            previous = frame.width
        }
    }

    // MARK: - Helpers

    /// One fixture per model id 0–10 with non-degenerate coefficients, used by the round-trip and
    /// origin tests. Values are chosen to stay invertible over the tested domain.
    private static let allModelFixtures: [(Int, [Double])] = [
        (0, [800, 640, 360]),
        (1, [800, 810, 640, 360]),
        (2, [800, 640, 360, 0.08]),
        (3, [800, 640, 360, 0.06, -0.02]),
        (4, referenceParams),
        (5, [800, 810, 640, 360, -0.02, 0.004, -0.0006, 0.00008]),
        (6, [800, 810, 640, 360, 0.05, -0.03, 0.001, -0.0008, 0.004, 0.02, -0.01, 0.002]),
        (7, [800, 810, 640, 360, 0.9]),
        (8, [800, 640, 360, -0.03]),
        (9, [800, 640, 360, -0.03, 0.005]),
        (10, [800, 810, 640, 360, -0.02, 0.004, 0.0012, -0.0009, -0.0006, 0.00008, 0.0007, -0.0005]),
    ]

    /// Fixtures for the frame solver: `(model id, params, source width, source height)`.
    private static let frameFixtures: [(Int, [Double], Int, Int)] = [
        (4, referenceParams, referenceWidth, referenceHeight),
        (2, [1200, 960, 540, 0.08], 1920, 1080),
        (3, [1200, 960, 540, 0.06, -0.02], 1920, 1080),
        (6, [1200, 1210, 960, 540, 0.05, -0.03, 0.001, -0.0008, 0.004, 0.02, -0.01, 0.002],
         1920, 1080),
        (7, [1200, 1210, 960, 540, 0.9], 1920, 1080),
        (5, [800, 810, 960, 540, -0.02, 0.004, -0.0006, 0.00008], 1920, 1080),
        (8, [800, 960, 540, -0.03], 1920, 1080),
        (9, [800, 960, 540, -0.03, 0.005], 1920, 1080),
        (10, [800, 810, 960, 540, -0.02, 0.004, 0.0012, -0.0009, -0.0006, 0.00008, 0.0007, -0.0005],
         1920, 1080),
        // An off-centre principal point — the case gsplat's width//2 recentring gets wrong.
        (4, [1200, 1210, 900, 500, 0.05, -0.02, 0.001, -0.0007], 1920, 1080),
    ]

    /// Every pixel on the rim of a frame. The mapped image of a rectangle is bounded by the image of
    /// its boundary for all of these models (`mappedBoundaryBox`'s premise), so this is the whole
    /// out-of-bounds question in `2(W + H)` samples rather than `W·H`.
    private func boundaryPixels(width: Int, height: Int) -> [(Int, Int)] {
        var out: [(Int, Int)] = []
        for i in 0..<width { out.append((i, 0)); out.append((i, height - 1)) }
        for j in 0..<height { out.append((0, j)); out.append((width - 1, j)) }
        return out
    }

    /// `(steps+1)²` points spanning `[-extent, extent]` in both normalized axes.
    private static func normalizedGrid(extent: Double, steps: Int) -> [(Double, Double)] {
        var out: [(Double, Double)] = []
        for i in 0...steps {
            for j in 0...steps {
                out.append((-extent + 2 * extent * Double(i) / Double(steps),
                            -extent + 2 * extent * Double(j) / Double(steps)))
            }
        }
        return out
    }

    /// Numerically inverts a forward distortion by damped Newton with a central-difference Jacobian.
    ///
    /// This lives in the tests on purpose: the resampler runs output → source and only ever needs
    /// the forward map, so shipping an iterative inverse would be dead weight (file header).
    private static func newtonInverse(
        _ d: CameraDistortion, _ targetX: Double, _ targetY: Double
    ) -> (Double, Double)? {
        var x = targetX, y = targetY
        let h = 1e-7
        for _ in 0..<400 {
            let (fx0, fy0) = d.distort(x, y)
            let rx = fx0 - targetX, ry = fy0 - targetY
            if abs(rx) < 1e-15 && abs(ry) < 1e-15 { return (x, y) }
            let (axp, ayp) = d.distort(x + h, y)
            let (axm, aym) = d.distort(x - h, y)
            let (bxp, byp) = d.distort(x, y + h)
            let (bxm, bym) = d.distort(x, y - h)
            let j11 = (axp - axm) / (2 * h), j21 = (ayp - aym) / (2 * h)
            let j12 = (bxp - bxm) / (2 * h), j22 = (byp - bym) / (2 * h)
            let det = j11 * j22 - j12 * j21
            guard abs(det) > 1e-14 else { return nil }
            let dx = (j22 * rx - j12 * ry) / det
            let dy = (-j21 * rx + j11 * ry) / det
            x -= dx
            y -= dy
            guard x.isFinite, y.isFinite else { return nil }
            if abs(dx) < 1e-16 && abs(dy) < 1e-16 { break }
        }
        let (cx, cy) = d.distort(x, y)
        guard abs(cx - targetX) < 1e-12, abs(cy - targetY) < 1e-12 else { return nil }
        return (x, y)
    }
}
