import Foundation

// COLMAP camera models and their forward distortion, plus the undistorted output frame arithmetic.
//
// ─────────────────────────────────────────────────────────────────────────────────────────────
// Why there is no iteration in this file
// ─────────────────────────────────────────────────────────────────────────────────────────────
// Undistorting an *image* runs output → source, and that direction needs the **forward** model:
//
//     output(u',v')  →  (x,y) = ((u'−cx')/fx', (v'−cy')/fy')      // normalize in the new pinhole
//                    →  (xd,yd) = D(x,y)                          // forward distortion, closed form
//                    →  source(u,v) = (fx·xd + cx, fy·yd + cy)    // project with the original K
//                    →  bilinear sample
//
// Every distortion model below states `xd` as a closed-form function of `(x,y)`, so the resampler
// only ever evaluates `D`. Iteration (Newton) is needed for the *opposite* question — "where does
// this one distorted point land once straightened" — which image resampling never asks. The
// undistortion tests do invert `D` numerically, but that inverse lives in the test file only.
//
// ─────────────────────────────────────────────────────────────────────────────────────────────
// Sources
// ─────────────────────────────────────────────────────────────────────────────────────────────
// The distortion mathematics is published, and each family is cited at its `distort` case:
//
//   • Brown, D.C. "Decentering distortion of lenses" (1966) / Conrady (1919) — the radial +
//     decentering (tangential) polynomial used by models 2, 3, 4, 6.
//   • Kannala, J. & Brandt, S. "A generic camera model and calibration method for conventional,
//     wide-angle, and fish-eye lenses", TPAMI 2006 — the θ = atan(r) equidistant family used by
//     models 5, 8, 9, 10.
//   • Devernay, F. & Faugeras, O. "Straight lines have to be straight", Machine Vision and
//     Applications 2001 — the FOV model, id 7.
//   • Weng, J. et al. "Camera Calibration with Distortion Models and Accuracy Evaluation",
//     TPAMI 1992 — the thin-prism terms of model 10.
//
// What COLMAP contributes is **not** the mathematics but the interface convention: which subset of
// coefficients each named model carries and in which order they sit in the `params` array. That was
// read off the local install's header, `colmap/4.1.0_1/include/colmap/sensor/models.h`, where each
// model's doc comment states "Parameter list is expected in the following order". This repository
// already treats that same header as an interface fact for the same purpose — `ColmapLoader.cpp:47`
// cites it as the source for the per-model parameter *count*. No COLMAP code is derived here; the
// implementations below are written from the published formulas.
//
// One further interface fact is read off the same install, at `undistortedFrame`: COLMAP's own
// undistorter refuses to shrink a frame below `min_scale = 0.2` of the source
// (`colmap/4.1.0_1/include/colmap/image/undistortion.h:44`). That default is quoted as the anchor
// for this file's frame-collapse floor — a number, not code.
//
// gsplat's undistortion (`examples/datasets/colmap.py`) was read but deliberately not ported: it
// re-centers fisheye output on `width//2, height//2`, which is only correct when the principal
// point sits dead center. This file keeps `cx', cy'` as the authority instead.
//
// ─────────────────────────────────────────────────────────────────────────────────────────────
// Fisheye is lossy by construction — not a defect of this code
// ─────────────────────────────────────────────────────────────────────────────────────────────
// A fisheye lens records directions a pinhole cannot express: as the incident angle approaches 90°
// the pinhole projection diverges. Re-projecting a fisheye capture onto a pinhole with the same
// `fx, fy` therefore *always* discards the outer field of view — `undistortedFrame` never grows the
// frame, so the wide directions simply fall outside it. That is the relationship between two
// projection models, not a bug, and the import path reports the loss so it is not mistaken for one.

// MARK: - Models

/// The COLMAP camera models this app can rectify, id 0–10.
///
/// Ids and parameter orders are the interface convention documented in
/// `colmap/4.1.0_1/include/colmap/sensor/models.h`. COLMAP 4.1 defines further ids (11+:
/// `RAD_TAN_THIN_PRISM_FISHEYE`, `SIMPLE_DIVISION`, …); those are rejected outright by
/// `CameraUndistortion.parse` rather than approximated by a neighbouring model.
enum ColmapCameraModel: Int, CaseIterable, Sendable {
    case simplePinhole = 0        // f, cx, cy
    case pinhole = 1              // fx, fy, cx, cy
    case simpleRadial = 2         // f, cx, cy, k
    case radial = 3               // f, cx, cy, k1, k2
    case openCV = 4               // fx, fy, cx, cy, k1, k2, p1, p2
    case openCVFisheye = 5        // fx, fy, cx, cy, k1, k2, k3, k4
    case fullOpenCV = 6           // fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
    case fov = 7                  // fx, fy, cx, cy, omega
    case simpleRadialFisheye = 8  // f, cx, cy, k
    case radialFisheye = 9        // f, cx, cy, k1, k2
    case thinPrismFisheye = 10    // fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1

    /// The `MODEL` token as it appears in `cameras.txt`.
    var name: String {
        switch self {
        case .simplePinhole: "SIMPLE_PINHOLE"
        case .pinhole: "PINHOLE"
        case .simpleRadial: "SIMPLE_RADIAL"
        case .radial: "RADIAL"
        case .openCV: "OPENCV"
        case .openCVFisheye: "OPENCV_FISHEYE"
        case .fullOpenCV: "FULL_OPENCV"
        case .fov: "FOV"
        case .simpleRadialFisheye: "SIMPLE_RADIAL_FISHEYE"
        case .radialFisheye: "RADIAL_FISHEYE"
        case .thinPrismFisheye: "THIN_PRISM_FISHEYE"
        }
    }

    /// Expected length of the `params` array. Same table as `ColmapLoader.cpp:48-63`.
    var parameterCount: Int {
        switch self {
        case .simplePinhole: 3
        case .pinhole, .simpleRadial, .simpleRadialFisheye: 4
        case .radial, .fov, .radialFisheye: 5
        case .openCV, .openCVFisheye: 8
        case .fullOpenCV, .thinPrismFisheye: 12
        }
    }

    init?(name: String) {
        guard let m = Self.allCases.first(where: { $0.name == name }) else { return nil }
        self = m
    }
}

/// The pinhole part of a COLMAP camera, in pixels.
struct CameraIntrinsics: Equatable, Sendable {
    var fx: Double
    var fy: Double
    var cx: Double
    var cy: Double
}

// MARK: - Distortion

/// The forward distortion of a COLMAP camera, normalized-plane in and normalized-plane out.
///
/// The cases are the *shapes* of distortion rather than one case per model id, because several ids
/// are the same formula with the trailing coefficients pinned to zero (e.g. `SIMPLE_RADIAL` is
/// `OPENCV` with `k2 = p1 = p2 = 0`). `CameraUndistortion.parse` performs that widening.
enum CameraDistortion: Equatable, Sendable {
    /// Ids 0, 1 — nothing to do.
    case none

    /// Ids 2, 3, 4, 6 — Brown–Conrady radial + decentering.
    /// Ids 2/3/4 leave `k3…k6` at zero; only `FULL_OPENCV` uses the rational denominator.
    case brownConrady(k1: Double, k2: Double, p1: Double, p2: Double,
                      k3: Double, k4: Double, k5: Double, k6: Double)

    /// Id 7 — Devernay–Faugeras FOV model.
    case fov(omega: Double)

    /// Ids 5, 8, 9 — Kannala–Brandt equidistant fisheye. Ids 8/9 leave the tail at zero.
    case fisheye(k1: Double, k2: Double, k3: Double, k4: Double)

    /// Id 10 — Kannala–Brandt radius with Weng-1992 decentering and thin-prism terms.
    case thinPrismFisheye(k1: Double, k2: Double, p1: Double, p2: Double,
                          k3: Double, k4: Double, sx1: Double, sy1: Double)

    /// Below this normalized radius the `θ/r` and `rd/r` scale factors are evaluated as 0/0, so the
    /// point is passed through. At `r == 0` the point *is* the principal point, `x = y = 0`, and
    /// every model maps it to itself exactly — the guard changes no value, it only avoids the NaN.
    private static let radiusEpsilon = 1e-12

    /// `|omega|` below this makes the FOV model the identity (its `ω → 0` limit) and the `1/ω`
    /// factor unrepresentable.
    private static let omegaEpsilon = 1e-9

    /// True when `distort` is the exact identity, i.e. rectification would only re-encode pixels.
    ///
    /// Note the fisheye family is **never** identity, not even with every `k` at zero: the
    /// `θ = atan(r)` remap is part of the projection, not part of the distortion polynomial. A
    /// fisheye camera with zero coefficients is a perfect equidistant fisheye and still has to be
    /// re-projected to become a pinhole.
    var isIdentity: Bool {
        switch self {
        case .none:
            true
        case .brownConrady(let k1, let k2, let p1, let p2, let k3, let k4, let k5, let k6):
            k1 == 0 && k2 == 0 && p1 == 0 && p2 == 0 && k3 == 0 && k4 == 0 && k5 == 0 && k6 == 0
        case .fov(let omega):
            abs(omega) <= Self.omegaEpsilon
        case .fisheye, .thinPrismFisheye:
            false
        }
    }

    /// Forward distortion: an ideal normalized point `(x, y)` → where the lens actually puts it.
    ///
    /// This is the only direction the resampler needs (see the file header).
    func distort(_ x: Double, _ y: Double) -> (Double, Double) {
        switch self {
        case .none:
            return (x, y)

        // Brown (1966) / Conrady (1919), in the OpenCV arrangement COLMAP's OPENCV/FULL_OPENCV
        // models follow:
        //
        //   radial = (1 + k1r² + k2r⁴ + k3r⁶) / (1 + k4r² + k5r⁴ + k6r⁶)
        //   xd = x·radial + 2p1·xy + p2(r² + 2x²)
        //   yd = y·radial + p1(r² + 2y²) + 2p2·xy
        //
        // The denominator is 1 for every model but FULL_OPENCV, whose k4…k6 are parsed as zero for
        // the others. Horner form keeps the r⁶ term from losing precision.
        case .brownConrady(let k1, let k2, let p1, let p2, let k3, let k4, let k5, let k6):
            let r2 = x * x + y * y
            let numerator = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
            let denominator = 1 + r2 * (k4 + r2 * (k5 + r2 * k6))
            let radial = numerator / denominator
            let xy = x * y
            return (x * radial + 2 * p1 * xy + p2 * (r2 + 2 * x * x),
                    y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * xy)

        // Devernay & Faugeras (2001):
        //
        //   rd = atan(2·r·tan(ω/2)) / ω          scale = rd / r
        //
        // Written as `scale = (2tan(ω/2)/ω) · (atan(z)/z)` with `z = 2r·tan(ω/2)`, because
        // `atan(z)/z` has the clean limit 1 at z → 0 while `rd/r` is 0/0 there. Note the r → 0 limit
        // of `scale` is `2tan(ω/2)/ω`, **not** 1 — using 1 near the principal point would introduce
        // a step in the distortion field. It is only the ω → 0 limit that is the identity.
        case .fov(let omega):
            guard abs(omega) > Self.omegaEpsilon else { return (x, y) }
            let r = (x * x + y * y).squareRoot()
            let tanHalfOmega = tan(omega / 2)
            let z = 2 * tanHalfOmega * r
            // atan(z)/z, series-expanded where the quotient would cancel catastrophically.
            let atanOverZ = abs(z) > 1e-8 ? atan(z) / z : 1 - z * z / 3
            let scale = 2 * tanHalfOmega * atanOverZ / omega
            return (x * scale, y * scale)

        // Kannala & Brandt (2006):
        //
        //   θ = atan(r)     θd = θ(1 + k1θ² + k2θ⁴ + k3θ⁶ + k4θ⁸)     scale = θd / r
        //
        // `θd/r → 1` as `r → 0`, which the epsilon guard's pass-through reproduces exactly.
        case .fisheye(let k1, let k2, let k3, let k4):
            let r = (x * x + y * y).squareRoot()
            guard r > Self.radiusEpsilon else { return (x, y) }
            let theta = atan(r)
            let t2 = theta * theta
            let thetaD = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
            let scale = thetaD / r
            return (x * scale, y * scale)

        // THIN_PRISM_FISHEYE — the one model whose reading had to be chosen.
        //
        // INTERPRETATION (mine, stated because the literature diverges): the equidistant map is
        // applied *first*, giving fisheye-plane coordinates
        //
        //     u = x·θ/r,   v = y·θ/r,   with ρ² = u² + v² = θ²
        //
        // and then the whole Weng-1992 correction — radial, decentering and thin-prism — is added
        // to that single base coordinate:
        //
        //     xd = u + u·(k1ρ² + k2ρ⁴ + k3ρ⁶ + k4ρ⁸) + 2p1·uv + p2(ρ² + 2u²) + sx1·ρ²
        //     yd = v + v·(k1ρ² + k2ρ⁴ + k3ρ⁶ + k4ρ⁸) + p1(ρ² + 2v²) + 2p2·uv + sy1·ρ²
        //
        // Grounds: Weng et al. 1992 — the paper COLMAP's own header names for this model — states
        // the model as an additive correction δ to one base coordinate, so radial and tangential
        // share that base rather than the tangential terms consuming the already-radially-corrected
        // point. It also makes the fisheye family internally consistent: `.fisheye` is exactly this
        // case with `p1 = p2 = sx1 = sy1 = 0`, since `u(1 + k1θ² + …) = x·θd/r`.
        //
        // The competing reading feeds the radially-corrected point `x·θd/r` into the tangential and
        // prism terms. The two agree to first order and differ only in the cross terms.
        //
        // CONFIDENCE: this case is supported by the round-trip and structural tests only. No real
        // THIN_PRISM_FISHEYE dataset has been rectified with it (see docs/manual-checks.md M36).
        case .thinPrismFisheye(let k1, let k2, let p1, let p2, let k3, let k4, let sx1, let sy1):
            let r = (x * x + y * y).squareRoot()
            guard r > Self.radiusEpsilon else { return (x, y) }
            let theta = atan(r)
            let equidistant = theta / r
            let u = x * equidistant
            let v = y * equidistant
            let rho2 = theta * theta
            let radial = rho2 * (k1 + rho2 * (k2 + rho2 * (k3 + rho2 * k4)))
            let uv = u * v
            return (u + u * radial + 2 * p1 * uv + p2 * (rho2 + 2 * u * u) + sx1 * rho2,
                    v + v * radial + p1 * (rho2 + 2 * v * v) + 2 * p2 * uv + sy1 * rho2)
        }
    }
}

// MARK: - Errors

enum CameraModelError: LocalizedError, Equatable {
    /// COLMAP knows this id but this app does not rectify it (11+). Never silently approximated.
    case unsupportedModel(id: Int)
    case parameterCountMismatch(model: String, expected: Int, found: Int)
    case degenerateIntrinsics(model: String)

    var errorDescription: String? {
        switch self {
        case .unsupportedModel(let id):
            "Unsupported COLMAP camera model id \(id) — this app rectifies ids 0–10 only. "
                + "Re-run COLMAP with a supported camera model, or undistort the dataset first."
        case .parameterCountMismatch(let model, let expected, let found):
            "COLMAP camera \(model) expects \(expected) parameters but the model has \(found)."
        case .degenerateIntrinsics(let model):
            "COLMAP camera \(model) has a non-positive focal length."
        }
    }
}

// MARK: - Parsing and output frame arithmetic

enum CameraUndistortion {
    /// `(model id, params)` straight out of `cameras.bin`/`cameras.txt` → intrinsics + distortion.
    ///
    /// Parameter order per model is the interface convention from `models.h` (see file header).
    /// Ids 0/2/3/8/9 carry a single focal length, which is widened to `fx = fy = f`.
    static func parse(modelId: Int, params: [Double]) throws -> (CameraIntrinsics, CameraDistortion) {
        guard let model = ColmapCameraModel(rawValue: modelId) else {
            throw CameraModelError.unsupportedModel(id: modelId)
        }
        guard params.count == model.parameterCount else {
            throw CameraModelError.parameterCountMismatch(
                model: model.name, expected: model.parameterCount, found: params.count)
        }
        let p = params

        let intrinsics: CameraIntrinsics
        let distortion: CameraDistortion
        switch model {
        case .simplePinhole:                                  // f, cx, cy
            intrinsics = CameraIntrinsics(fx: p[0], fy: p[0], cx: p[1], cy: p[2])
            distortion = .none
        case .pinhole:                                        // fx, fy, cx, cy
            intrinsics = CameraIntrinsics(fx: p[0], fy: p[1], cx: p[2], cy: p[3])
            distortion = .none
        case .simpleRadial:                                   // f, cx, cy, k
            intrinsics = CameraIntrinsics(fx: p[0], fy: p[0], cx: p[1], cy: p[2])
            distortion = .brownConrady(k1: p[3], k2: 0, p1: 0, p2: 0, k3: 0, k4: 0, k5: 0, k6: 0)
        case .radial:                                         // f, cx, cy, k1, k2
            intrinsics = CameraIntrinsics(fx: p[0], fy: p[0], cx: p[1], cy: p[2])
            distortion = .brownConrady(k1: p[3], k2: p[4], p1: 0, p2: 0, k3: 0, k4: 0, k5: 0, k6: 0)
        case .openCV:                                         // fx, fy, cx, cy, k1, k2, p1, p2
            intrinsics = CameraIntrinsics(fx: p[0], fy: p[1], cx: p[2], cy: p[3])
            distortion = .brownConrady(k1: p[4], k2: p[5], p1: p[6], p2: p[7],
                                       k3: 0, k4: 0, k5: 0, k6: 0)
        case .openCVFisheye:                                  // fx, fy, cx, cy, k1, k2, k3, k4
            intrinsics = CameraIntrinsics(fx: p[0], fy: p[1], cx: p[2], cy: p[3])
            distortion = .fisheye(k1: p[4], k2: p[5], k3: p[6], k4: p[7])
        case .fullOpenCV:                        // fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
            intrinsics = CameraIntrinsics(fx: p[0], fy: p[1], cx: p[2], cy: p[3])
            distortion = .brownConrady(k1: p[4], k2: p[5], p1: p[6], p2: p[7],
                                       k3: p[8], k4: p[9], k5: p[10], k6: p[11])
        case .fov:                                            // fx, fy, cx, cy, omega
            intrinsics = CameraIntrinsics(fx: p[0], fy: p[1], cx: p[2], cy: p[3])
            distortion = .fov(omega: p[4])
        case .simpleRadialFisheye:                            // f, cx, cy, k
            intrinsics = CameraIntrinsics(fx: p[0], fy: p[0], cx: p[1], cy: p[2])
            distortion = .fisheye(k1: p[3], k2: 0, k3: 0, k4: 0)
        case .radialFisheye:                                  // f, cx, cy, k1, k2
            intrinsics = CameraIntrinsics(fx: p[0], fy: p[0], cx: p[1], cy: p[2])
            distortion = .fisheye(k1: p[3], k2: p[4], k3: 0, k4: 0)
        case .thinPrismFisheye:                 // fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
            intrinsics = CameraIntrinsics(fx: p[0], fy: p[1], cx: p[2], cy: p[3])
            distortion = .thinPrismFisheye(k1: p[4], k2: p[5], p1: p[6], p2: p[7],
                                           k3: p[8], k4: p[9], sx1: p[10], sy1: p[11])
        }

        guard intrinsics.fx > 0, intrinsics.fy > 0 else {
            throw CameraModelError.degenerateIntrinsics(model: model.name)
        }
        return (intrinsics, distortion)
    }

    // MARK: Output frame

    /// Boundary samples per edge of the candidate output rectangle. The extremum of the mapped
    /// boundary is smooth in the sample parameter, so this is dense enough that the result is
    /// invariant to it — measured identical at 256, 1024 and 4096 on the reference camera.
    private static let boundarySamples = 1024

    /// How far inside the source a mapped sample must stay, in source pixels.
    ///
    /// 0.5 of this is a hard requirement rather than caution: COLMAP's convention puts the image on
    /// the continuous span `[0, W]` with pixel *centers* at `i + 0.5`, so a bilinear tap needs the
    /// mapped position in `[0.5, W − 0.5]` to have all four neighbours. The other 0.5 is the safety
    /// margin §4 of the spec asks for.
    private static let sourceMargin = 1.0

    /// The pinhole frame the rectified images will be written at.
    ///
    /// **Rule (spec §4):** `fx, fy` are carried over untouched; the output is the largest
    /// *center-aligned* rectangle every one of whose boundary pixels still samples inside the source
    /// image; the principal point moves to the middle of that rectangle, `cx' = W'/2, cy' = H'/2`.
    ///
    /// **The one exception is an identity distortion**, which returns the camera exactly as it came
    /// in — its own `cx, cy` included, *not* `W/2, H/2`. See the early return below: re-centering a
    /// camera nothing was done to is not a rectification, it is a translation.
    ///
    /// Where the rule comes from. Keeping the source size would leave black rims, and those black
    /// pixels become training data. Measuring the reference camera (OPENCV 2160×3840, fx 2630.110,
    /// fy 2633.084, c 1080/1920, k1 +0.043354, k2 −0.079594, p1 −0.005166, p2 +0.000340) shows the
    /// full-size output overrunning the source by 10.0px left, 11.2px right, 22.8px top and 0px
    /// bottom. A center-aligned rectangle has to give up the larger of each opposing pair, so the
    /// width becomes `2160 − 2×11.2 = 2137.6 → 2137` — and COLMAP's own `image_undistorter` emits
    /// exactly **2137×3792** for this dataset, with `fx, fy` unchanged and `cx, cy = 1068.5, 1896.0`,
    /// i.e. precisely `W'/2, H'/2`. The rule is a description of what COLMAP does, recovered from
    /// its output rather than from its source. This implementation returns 2136×3792 for that
    /// camera: it solves the constraint properly instead of subtracting the full-frame overrun once,
    /// and it spends the 1px safety margin (the same margin that makes COLMAP two rows shorter than
    /// a single-pass estimate).
    ///
    /// The two axes shrink independently — `fx, fy` are fixed, so there is no aspect ratio to
    /// preserve, and COLMAP's own 2137×3792 out of 2160×3840 is not aspect-preserving either.
    ///
    /// gsplat's alternative — the bounding box of the valid mask (`colmap.py:415-421`) — is not used:
    /// barrel distortion leaves its invalid region in the *corners*, which a bounding box keeps.
    ///
    /// The frame is never grown beyond the source. For fisheye that is the whole story of the field
    /// of view loss described in the file header: the directions a pinhole would need a far larger
    /// frame to hold are simply dropped.
    ///
    /// Dimensions are **not** rounded to even numbers. This is training input, not an encoder
    /// input — COLMAP itself emits the odd 2137.
    static func undistortedFrame(
        intrinsics: CameraIntrinsics,
        distortion: CameraDistortion,
        width: Int,
        height: Int
    ) -> (width: Int, height: Int, cx: Double, cy: Double) {
        let sourceW = Double(width)
        let sourceH = Double(height)

        // Nothing to rectify: the camera is carried over untouched, principal point included. Two
        // separate things are being kept exact here.
        //
        // The frame. An identity map puts the outermost pixel centers at 0.5 and W−0.5, which is
        // inside the sampling domain but not inside the safety margin, so the solver below would
        // otherwise shave a pixel off a frame that needs no work at all.
        //
        // The principal point. Returning `W/2, H/2` — as this did until 2026-08-15 FIX 2 — is only
        // the identity when `cx` already sits dead center, and it silently is not otherwise: with
        // `OPENCV 800 600 f f 470 250 0 0 0 0` (every coefficient zero, principal point offset) the
        // returned frame stayed 800×600 while `cx'` jumped to 400, so `ImageUndistorter`'s map read
        // the source 70 columns and 50 rows over, and 800·600 − 730·550 = 78,500 output pixels —
        // 16.4 % of the frame — sampled outside the capture and came back black. Black pixels are
        // training data, which is the one thing this whole arithmetic exists to prevent. The frame
        // and the principal point disagreed: the pixels were the camera's own, the row written for
        // them claimed the optical axis had moved 70 px.
        //
        // Note this branch and the solver below stay *discontinuous in frame size*, and that is not
        // a defect. Measured on that camera at `f = 900`: zero coefficients keep 800×600 here, while
        // `k1 = 1e-9` goes through the solver and comes out 658×498 — because a solved frame is
        // re-centered (`cx' = W'/2`), and a centered frame inside a source whose axis sits at 470 of
        // 800 cannot be wider than 2·(800−470). The two branches answer different questions. Nothing
        // is resampled here, so nothing has to be re-centered and the full capture is kept; once a
        // single pixel does have to move, spec §4's centered frame is the rule and its cost is real
        // frame. Both are internally consistent, which is all that keeps black out of the output.
        guard !distortion.isIdentity else {
            return (width, height, intrinsics.cx, intrinsics.cy)
        }

        // Shrink to the fixed point. Each pass measures how far the mapped boundary overruns the
        // permitted source region and removes that much from both sides of the offending axis;
        // shrinking a center-aligned rectangle only ever moves boundary samples inward, so the
        // sequence is monotone and settles in a handful of passes.
        var frameW = sourceW
        var frameH = sourceH
        for _ in 0..<64 {
            let box = mappedBoundaryBox(intrinsics: intrinsics, distortion: distortion,
                                        frameW: frameW, frameH: frameH)
            let overrunX = max(sourceMargin - box.minU, box.maxU - (sourceW - sourceMargin), 0)
            let overrunY = max(sourceMargin - box.minV, box.maxV - (sourceH - sourceMargin), 0)
            if overrunX < 1e-6 && overrunY < 1e-6 { break }
            frameW = max(1, frameW - 2 * overrunX)
            frameH = max(1, frameH - 2 * overrunY)
        }

        // Floor rather than round: the solved size is the largest feasible one, so rounding up would
        // step back outside it. Flooring shrinks a center-aligned rectangle, which stays feasible.
        var outW = max(1, Int(frameW.rounded(.down)))
        var outH = max(1, Int(frameH.rounded(.down)))

        // Feasibility is re-checked on the integer frame instead of assumed. Flooring also moves
        // `cx', cy'` by up to half a pixel, and while shrinking cannot make the constraint harder,
        // the cheap confirmation is worth more than the argument that it cannot.
        for _ in 0..<8 {
            let box = mappedBoundaryBox(intrinsics: intrinsics, distortion: distortion,
                                        frameW: Double(outW), frameH: Double(outH))
            let okX = box.minU >= sourceMargin && box.maxU <= sourceW - sourceMargin
            let okY = box.minV >= sourceMargin && box.maxV <= sourceH - sourceMargin
            if okX && okY { break }
            if !okX { outW = max(1, outW - 1) }
            if !okY { outH = max(1, outH - 1) }
        }

        return (outW, outH, Double(outW) / 2, Double(outH) / 2)
    }

    // MARK: Is the solved frame still a usable image?

    /// The smallest side, in pixels, a rectified frame may have.
    ///
    /// Anchored on the core rather than on taste: with `ssim_weight > 0` — the default — the
    /// training validator refuses any image narrower or shorter than the 11 px SSIM window, so a
    /// frame below this cannot train at all, whatever anyone thinks of it as a photograph.
    private static let minimumFrameSide = 11

    /// The smallest fraction of each source side a rectified frame may keep.
    ///
    /// `0.2` is COLMAP's own `UndistortCameraOptions::min_scale` default
    /// (`colmap/4.1.0_1/include/colmap/image/undistortion.h:44`), read off the same local install
    /// this file already treats as the interface authority. COLMAP applies it as a clamp — it
    /// accepts the blank pixels that clamping leaves — while this app has no such option: its
    /// contract is a frame with no black in it (spec §4), so the same threshold has to be a
    /// refusal instead. Once the largest distortion-free pinhole frame inside a capture is under a
    /// fifth of it on a side, the solver is no longer proposing a crop — it is reporting that the
    /// camera cannot be straightened.
    private static let minimumFrameScale = 0.2

    /// Whether a solved frame is still an image, as opposed to the arithmetically correct answer
    /// that the camera cannot be unwarped.
    ///
    /// The solver deliberately does **not** throw this itself. `undistortedFrame` is pure geometry
    /// and 1×1 is the true largest rectangle for a strong enough pincushion; what to do about that
    /// is import policy, so the caller (`ColmapRectification.rectify`) turns a `false` here into a
    /// refusal *before* it writes a single pixel. Measured on a 320×240 `SIMPLE_RADIAL` with
    /// `f = 160`: `k = 0.4` solves to 121×90, `k = 0.5` to 71×53, `k = 0.6` to 22×16 and `k = 0.7`
    /// to 1×1 — and COLMAP's own `max_extra_param` default of 1.0 (`incremental_mapper.h:120` of the
    /// same install) means COLMAP itself does not reject that last coefficient. Without this gate all 110 images were re-encoded at 1×1,
    /// `outsidePixels` stayed 0 so nothing warned, the project was created looking healthy, and the
    /// run died minutes later inside the core's SSIM validator — spec §1.1's failure shape exactly,
    /// by which point the originals only exist in the user's own zip.
    static func frameIsUsable(width: Int, height: Int, sourceWidth: Int, sourceHeight: Int) -> Bool {
        guard width >= minimumFrameSide, height >= minimumFrameSide else { return false }
        return Double(width) >= minimumFrameScale * Double(sourceWidth)
            && Double(height) >= minimumFrameScale * Double(sourceHeight)
    }

    /// The axis-aligned box in **source** pixels covered by the boundary of a candidate output
    /// frame, sampled at output pixel centers.
    ///
    /// Pixel centers, not the continuous frame edge: the resampler evaluates output pixel `i` at
    /// `i + 0.5`, so the extreme samples it will ever request sit at 0.5 and `W' − 0.5`. Sampling
    /// the edge at 0 and `W'` instead would model half a pixel that is never read and cost a row and
    /// a column of real image.
    ///
    /// Only the boundary is scanned. For every model here the mapped image of a center-aligned
    /// rectangle is bounded by the image of its boundary — the distortions are continuous and
    /// monotone in radius over the frame — so an interior sample cannot exceed it.
    private static func mappedBoundaryBox(
        intrinsics: CameraIntrinsics,
        distortion: CameraDistortion,
        frameW: Double,
        frameH: Double
    ) -> (minU: Double, maxU: Double, minV: Double, maxV: Double) {
        let cxOut = frameW / 2
        let cyOut = frameH / 2
        let loU = 0.5, hiU = max(0.5, frameW - 0.5)
        let loV = 0.5, hiV = max(0.5, frameH - 0.5)

        var minU = Double.infinity, maxU = -Double.infinity
        var minV = Double.infinity, maxV = -Double.infinity

        func visit(_ outU: Double, _ outV: Double) {
            let x = (outU - cxOut) / intrinsics.fx
            let y = (outV - cyOut) / intrinsics.fy
            let (xd, yd) = distortion.distort(x, y)
            let u = intrinsics.fx * xd + intrinsics.cx
            let v = intrinsics.fy * yd + intrinsics.cy
            minU = min(minU, u); maxU = max(maxU, u)
            minV = min(minV, v); maxV = max(maxV, v)
        }

        for i in 0...boundarySamples {
            let t = Double(i) / Double(boundarySamples)
            let u = loU + t * (hiU - loU)
            let v = loV + t * (hiV - loV)
            visit(u, loV)   // top edge
            visit(u, hiV)   // bottom edge
            visit(loU, v)   // left edge
            visit(hiU, v)   // right edge
        }
        return (minU, maxU, minV, maxV)
    }
}
