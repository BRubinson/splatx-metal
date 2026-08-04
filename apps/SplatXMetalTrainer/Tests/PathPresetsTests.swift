// Cinema Task 4: the pure geometry functions of the presets — Orbit 360° (circular key
// positions, quat gaze, equal angular spacing) + the capture trajectory (sorted by image name,
// even subsampling, COLMAP → Metal pose conversion, fov from intrinsics) + VM application
// (adds a new path, selects it, persists — existing paths unchanged). The SwiftUI half
// (the buttons) is the responsibility of the manual checklist.
// Path transform Task 3: Orbit 360° over 20 s (the 18°/s convention band) + Reveal Arc (a 220°
// arc with the current view at its center) + Rising Orbit (a 360° spiral — pitch rises by +25°
// and the radius converges to 0.85d) — spec §7 is the document of record.
// Panel redesign Task 4 (2026-07-29 spec §6): four new presets — Pull Back Reveal (a dolly-out
// at fixed azimuth) + Spiral Descent (a 540° descending spiral whose last key = the current
// view) + Crane Down (elevation descends only) + Oval Loop (a focal ellipse with sin²
// elevation modulation, a closed loop). The denominator rule (closed i/n, open i/(n−1)) and
// the aim-lock decision (PathTransform.pivot — the premise of how the handle and Radius
// behave) are pinned across every preset.
// Panel redesign FIX 1 (final review — spec footnote 1): the aim-lock contract is asserted
// across the whole pitch band {−1.5 … 1.5} (a single-point sample cannot catch a boundary
// regression) + Pull Back Reveal flips to a descent when the rise headroom is insufficient
// (fixed azimuth → the closed form det(A/n) = (1−R²)/4, boundary 1.4133 rad).
import XCTest
import simd
@testable import SplatXMetalTrainer

@MainActor
final class PathPresetsTests: XCTestCase {

    // MARK: Orbit 360°

    private let target = SIMD3<Float>(1, 2, 3)

    private func orbit(pitch: Float = 0.4) -> CameraPath {
        PathPresets.orbit360(target: target, distance: 4, yaw: 0.3, pitch: pitch,
                             upAxis: SIMD3(0, 1, 0), fov: 1.0)
    }

    /// The horizontal azimuth in the (0,1,0) up-axis standard basis (atan2(x, z)) — the same
    /// convention as y in the orbit placement formula.
    private func azimuth(_ p: SIMD3<Float>) -> Float {
        let d = p - target
        return atan2(d.x, d.z)
    }

    /// Angle-equality assertion that ignores 2π wrapping (atan2's range (−π, π] vs. a
    /// non-normalized angle such as yaw+315°).
    private func assertAngleEqual(_ a: Float, _ b: Float, accuracy: Float = 1e-3,
                                  _ message: String = "",
                                  file: StaticString = #filePath, line: UInt = #line) {
        var d = (a - b).truncatingRemainder(dividingBy: 2 * .pi)
        if d > .pi { d -= 2 * .pi }
        if d < -.pi { d += 2 * .pi }
        XCTAssertEqual(d, 0, accuracy: accuracy, message, file: file, line: line)
    }

    func testOrbit360Shape() {
        let path = orbit()
        XCTAssertEqual(path.keyframes.count, 8)
        XCTAssertTrue(path.loop)                       // a closed circle
        XCTAssertEqual(path.ease, EaseProfile.none)    // turntable, constant speed (avoids a stall at the seam)
        XCTAssertEqual(path.pacing, .even)
        XCTAssertEqual(path.duration, 20, accuracy: 1e-12)   // spec §7: the 18°/s convention band
        // The i/n denominator of a closed circle — the last key is yaw+315°. If you change
        // it to i/(n−1) the last key coincides with the first, producing a zero-length
        // segment, and arc-length even pacing allocates zero time to that span.
        assertAngleEqual(azimuth(path.keyframes.last!.pos), 0.3 + 7 * .pi / 4)
    }

    func testOrbit360KeysLieOnCircleAtConstantElevation() {
        let path = orbit()
        for k in path.keyframes {
            XCTAssertEqual(simd_distance(k.pos, target), 4, accuracy: 1e-4)      // radius
            let dir = simd_normalize(k.pos - target)
            XCTAssertEqual(simd_dot(dir, SIMD3(0, 1, 0)), sin(0.4), accuracy: 1e-4)  // elevation
        }
    }

    func testOrbit360StartsAtCurrentEyeAndGazesTarget() {
        var cam = CameraController()
        cam.target = target
        cam.distance = 4
        cam.yaw = 0.3
        cam.pitch = 0.4
        let path = orbit()
        // The first key = the current camera position (reproduces the CameraController.eye
        // formula).
        let first = path.keyframes[0].pos
        XCTAssertEqual(simd_distance(first, cam.eye), 0, accuracy: 1e-4)
        for k in path.keyframes {
            // Gaze: the −Z of the camera→world rotation points at the target.
            let forward = k.quatValue.act(SIMD3<Float>(0, 0, -1))
            let expected = simd_normalize(target - k.pos)
            XCTAssertEqual(simd_distance(forward, expected), 0, accuracy: 1e-3)
            // No roll: the camera up lies in the same hemisphere as the scene up.
            XCTAssertGreaterThan(simd_dot(k.quatValue.act(SIMD3(0, 1, 0)),
                                          SIMD3<Float>(0, 1, 0)), 0)
            XCTAssertEqual(k.fov, 1.0, accuracy: 1e-6)
        }
    }

    func testOrbit360EqualAngularSpacing() {
        let path = orbit()
        // upAxis (0,1,0) → the standard basis: horizontal angle = atan2(x, z). Adjacent key
        // spacing = 2π/8.
        let angles = path.keyframes.map { k -> Float in
            let d = k.pos - target
            return atan2(d.x, d.z)
        }
        for i in 1..<angles.count {
            var diff = angles[i] - angles[i - 1]
            while diff < 0 { diff += 2 * .pi }
            while diff >= 2 * .pi { diff -= 2 * .pi }
            XCTAssertEqual(diff, .pi / 4, accuracy: 1e-3, "seg \(i)")
        }
    }

    // MARK: Reveal Arc (spec §7 — a 220° open arc with the current view at its center)

    private func reveal(pitch: Float = 0.4) -> CameraPath {
        PathPresets.revealArc(target: target, distance: 4, yaw: 0.3, pitch: pitch,
                              upAxis: SIMD3(0, 1, 0), fov: 1.0)
    }

    func testRevealArcShape() {
        let path = reveal()
        XCTAssertEqual(path.keyframes.count, 6)
        XCTAssertFalse(path.loop)                          // an open arc
        XCTAssertEqual(path.ease, .easeInOut)              // slows in and out (the reveal-shot convention)
        XCTAssertEqual(path.pacing, .even)
        XCTAssertEqual(path.duration, 12, accuracy: 1e-12)
    }

    func testRevealArcSweepCenteredOnCurrentYaw() {
        let path = reveal()
        let n = path.keyframes.count
        let sweep = Float(220) * .pi / 180
        for (i, k) in path.keyframes.enumerated() {
            // The i/(n−1) placement formula of an open arc: yaw − 110° + i/(n−1)·220°
            assertAngleEqual(azimuth(k.pos),
                             0.3 - sweep / 2 + Float(i) / Float(n - 1) * sweep, "key \(i)")
            XCTAssertEqual(simd_distance(k.pos, target), 4, accuracy: 1e-4)   // constant radius
            let dir = simd_normalize(k.pos - target)
            XCTAssertEqual(simd_dot(dir, SIMD3(0, 1, 0)), sin(0.4),
                           accuracy: 1e-4)                                    // constant height
        }
        // The current view sits at the center of the arc: key i and key n−1−i are symmetric
        // about yaw (first −110° ↔ last +110°).
        for i in 0..<(n / 2) {
            let lo = azimuth(path.keyframes[i].pos)
            let hi = azimuth(path.keyframes[n - 1 - i].pos)
            assertAngleEqual(lo - 0.3, -(hi - 0.3), "pair \(i)")
        }
    }

    // MARK: Rising Orbit (spec §7 — a 360° open spiral that rises and converges)

    private func rising(pitch: Float = 0.4) -> CameraPath {
        PathPresets.risingOrbit(target: target, distance: 4, yaw: 0.3, pitch: pitch,
                                upAxis: SIMD3(0, 1, 0), fov: 1.0)
    }

    func testRisingOrbitShape() {
        let path = rising()
        XCTAssertEqual(path.keyframes.count, 10)
        XCTAssertFalse(path.loop)          // start and end heights differ, so it does not close (spec §7)
        XCTAssertEqual(path.ease, .easeInOut)
        XCTAssertEqual(path.pacing, .even)
        XCTAssertEqual(path.duration, 20, accuracy: 1e-12)
    }

    func testRisingOrbitFullTurnRisingAndApproaching() {
        let path = rising()
        let n = path.keyframes.count
        let riseAngle = Float(25) * .pi / 180
        for (i, k) in path.keyframes.enumerated() {
            let t = Float(i) / Float(n - 1)
            assertAngleEqual(azimuth(k.pos), 0.3 + t * 2 * .pi, "key \(i)")     // open i/(n−1)
            XCTAssertEqual(simd_distance(k.pos, target), 4 * (1 - 0.15 * t),
                           accuracy: 1e-4, "radius d → 0.85d linear, key \(i)")
            let elev = asin(simd_dot(simd_normalize(k.pos - target), SIMD3<Float>(0, 1, 0)))
            XCTAssertEqual(elev, 0.4 + t * riseAngle, accuracy: 1e-3,
                           "pitch → pitch+25° linear, key \(i)")
        }
        // The last key has the same azimuth as the start (a full turn) but is higher and
        // closer.
        assertAngleEqual(azimuth(path.keyframes.last!.pos), azimuth(path.keyframes[0].pos))
        XCTAssertEqual(simd_distance(path.keyframes.last!.pos, target), 4 * 0.85,
                       accuracy: 1e-4)
    }

    func testRisingOrbitClampsRisenPitch() {
        // Start 1.45 + a 25° rise (≈0.436) = 1.886 rad → the end value is clamped to
        // ±1.5 rad as well (the point the brief resolved).
        let path = rising(pitch: 1.45)
        let last = path.keyframes.last!
        let elev = asin(simd_dot(simd_normalize(last.pos - target), SIMD3<Float>(0, 1, 0)))
        XCTAssertEqual(elev, 1.5, accuracy: 1e-3)
    }

    // MARK: The four new presets (panel redesign Task 4 — the geometry of record is the
    // spec §6 table)

    private func pullBack(pitch: Float = 0.4) -> CameraPath {
        PathPresets.pullBackReveal(target: target, distance: 4, yaw: 0.3, pitch: pitch,
                                   upAxis: SIMD3(0, 1, 0), fov: 1.0)
    }

    /// The elevation in the (0,1,0) up-axis standard basis — asin(dir·up), the same
    /// convention as p in the orbit placement formula.
    private func elevation(_ p: SIMD3<Float>) -> Float {
        asin(simd_dot(simd_normalize(p - target), SIMD3<Float>(0, 1, 0)))
    }

    func testPullBackRevealShape() {
        let path = pullBack()
        XCTAssertEqual(path.keyframes.count, 6)
        XCTAssertFalse(path.loop)                          // an open dolly-out
        XCTAssertEqual(path.ease, .easeInOut)              // slows in and out (the reveal-shot convention)
        XCTAssertEqual(path.pacing, .even)
        XCTAssertEqual(path.duration, 10, accuracy: 1e-12)
    }

    func testPullBackRevealPullsBackAndRisesAtFixedAzimuth() {
        let path = pullBack()
        let n = path.keyframes.count
        let rise = Float(20) * .pi / 180
        // The first key = the current camera (the user's framing is the shot's starting
        // point — the same convention as orbit360).
        var cam = CameraController()
        cam.target = target; cam.distance = 4; cam.yaw = 0.3; cam.pitch = 0.4
        XCTAssertEqual(simd_distance(path.keyframes[0].pos, cam.eye), 0, accuracy: 1e-4)
        for (i, k) in path.keyframes.enumerated() {
            let t = Float(i) / Float(n - 1)
            assertAngleEqual(azimuth(k.pos), 0.3, "azimuth unchanged, key \(i)")
            XCTAssertEqual(simd_distance(k.pos, target), 4 * (1 + 1.5 * t),
                           accuracy: 1e-3, "distance d → 2.5d linear, key \(i)")
            XCTAssertEqual(elevation(k.pos), 0.4 + t * rise, accuracy: 1e-3,
                           "elevation rises +20° linearly, key \(i)")
        }
        // The open i/(n−1) denominator — the last key lands exactly on the range end value
        // (2.5d, +20°).
        XCTAssertEqual(simd_distance(path.keyframes.last!.pos, target), 10, accuracy: 1e-3)
        XCTAssertEqual(elevation(path.keyframes.last!.pos), 0.4 + rise, accuracy: 1e-3)
    }

    func testPullBackRevealFlipsToDescentWhenRiseHeadroomInsufficient() {
        // The rise-headroom boundary (final review FIX 1 ② — spec footnote 1): when the
        // starting pitch + 20° exceeds the 1.5 upper bound, the rise is cut off by the clamp,
        // the elevation spread collapses, and because the azimuth is fixed,
        // det(A/n) = (1 − R²)/4 ≤ 1e-3 used to make the aim lock fail (boundary 1.4133 rad).
        // This pins the prescription for that defect: with no headroom, **flip to a descent** —
        // previously 1.45 was already in the fallback region, yet the test asserted only the
        // clamp and never called pivot, so it was vacuous (review ③).
        let rise = Float(20) * .pi / 180
        let flipped = pullBack(pitch: 1.45)   // 1.45 + 0.349 > 1.5 → descent
        XCTAssertEqual(elevation(flipped.keyframes[0].pos), 1.45, accuracy: 1e-3,
                       "the first-key-equals-current-view property holds regardless of the branch")
        XCTAssertEqual(elevation(flipped.keyframes.last!.pos), 1.45 - rise, accuracy: 1e-3)
        XCTAssertEqual(PathTransform.pivot(for: flipped.keyframes)?.isAimLocked, true)
        // Headroom available (1.15 + 0.349 ≤ 1.5) → the previous rise is kept (inside the
        // boundary nothing changes).
        let risen = pullBack(pitch: 1.15)
        XCTAssertEqual(elevation(risen.keyframes.last!.pos), 1.15 + rise, accuracy: 1e-3)
        XCTAssertEqual(PathTransform.pivot(for: risen.keyframes)?.isAimLocked, true)
    }

    private func spiral(pitch: Float = 0.4) -> CameraPath {
        PathPresets.spiralDescent(target: target, distance: 4, yaw: 0.3, pitch: pitch,
                                  upAxis: SIMD3(0, 1, 0), fov: 1.0)
    }

    func testSpiralDescentShape() {
        let path = spiral()
        XCTAssertEqual(path.keyframes.count, 12)
        XCTAssertFalse(path.loop)          // start and end differ in height and distance, so it does not close
        XCTAssertEqual(path.ease, .easeInOut)
        XCTAssertEqual(path.pacing, .even)
        XCTAssertEqual(path.duration, 30, accuracy: 1e-12)   // 540° ÷ 18°/s, the convention band
    }

    func testSpiralDescentSweepsDescendsAndSettlesAtCurrentView() {
        let path = spiral()
        let n = path.keyframes.count
        let sweep = Float(540) * .pi / 180
        let drop = Float(35) * .pi / 180
        for (i, k) in path.keyframes.enumerated() {
            let t = Float(i) / Float(n - 1)
            // The open i/(n−1) placement formula: yaw + (t−1)·540° — the end (t=1) is
            // exactly yaw.
            assertAngleEqual(azimuth(k.pos), 0.3 + (t - 1) * sweep, "key \(i)")
            XCTAssertEqual(simd_distance(k.pos, target), 4 * (1 + 1.5 * (1 - t)),
                           accuracy: 1e-3, "distance 2.5d → d convergence, key \(i)")
            XCTAssertEqual(elevation(k.pos), 0.4 + (1 - t) * drop, accuracy: 1e-3,
                           "elevation descends +35° → current pitch, key \(i)")
        }
        // The last key = the current camera ("it sits down in front" — the framing the user
        // set up is the shot's landing point).
        var cam = CameraController()
        cam.target = target; cam.distance = 4; cam.yaw = 0.3; cam.pitch = 0.4
        XCTAssertEqual(simd_distance(path.keyframes.last!.pos, cam.eye), 0, accuracy: 1e-3)
        // Total sweep 540° — verified as the sum of the adjacent azimuth increments (all
        // positive), a check that is independent of 2π wrapping.
        var total: Float = 0
        for i in 1..<n {
            var d = azimuth(path.keyframes[i].pos) - azimuth(path.keyframes[i - 1].pos)
            while d < 0 { d += 2 * .pi }
            while d >= 2 * .pi { d -= 2 * .pi }
            total += d
        }
        XCTAssertEqual(total, sweep, accuracy: 1e-2)
    }

    private func crane(pitch: Float = 0.4) -> CameraPath {
        PathPresets.craneDown(target: target, distance: 4, yaw: 0.3, pitch: pitch,
                              upAxis: SIMD3(0, 1, 0), fov: 1.0)
    }

    func testCraneDownShape() {
        let path = crane()
        XCTAssertEqual(path.keyframes.count, 6)
        XCTAssertFalse(path.loop)                          // an open descent
        XCTAssertEqual(path.ease, .easeInOut)
        XCTAssertEqual(path.pacing, .even)
        XCTAssertEqual(path.duration, 12, accuracy: 1e-12)
    }

    func testCraneDownLowersElevationOnly() {
        // The current pitch (0.4) < the 1.0 rad high-angle lower bound → the start is 1.0
        // (the essence of a crane is to start looking down — this spread is also what
        // guarantees the aim-lock det condition). The end is eye level, 0.05 rad.
        let path = crane()
        let n = path.keyframes.count
        let hi: Float = 1.0, lo: Float = 0.05
        for (i, k) in path.keyframes.enumerated() {
            let t = Float(i) / Float(n - 1)
            assertAngleEqual(azimuth(k.pos), 0.3, "azimuth unchanged, key \(i)")
            XCTAssertEqual(simd_distance(k.pos, target), 4, accuracy: 1e-3,
                           "distance unchanged (elevation descends only — spec §6 table), key \(i)")
            XCTAssertEqual(elevation(k.pos), hi + t * (lo - hi), accuracy: 1e-3,
                           "elevation descends linearly, key \(i)")
        }
        // The open i/(n−1) denominator — the last key lands exactly at eye level (0.05 rad).
        XCTAssertEqual(elevation(path.keyframes.last!.pos), lo, accuracy: 1e-3)
    }

    func testCraneDownKeepsUserPitchWhenAlreadyHigh() {
        // If the current pitch is already above the high-angle lower bound (1.0), the path
        // starts from that value (respecting the user's framing).
        let path = crane(pitch: 1.2)
        XCTAssertEqual(elevation(path.keyframes[0].pos), 1.2, accuracy: 1e-3)
    }

    private func oval(pitch: Float = 0.4) -> CameraPath {
        PathPresets.ovalLoop(target: target, distance: 4, yaw: 0.3, pitch: pitch,
                             upAxis: SIMD3(0, 1, 0), fov: 1.0)
    }

    func testOvalLoopShape() {
        let path = oval()
        XCTAssertEqual(path.keyframes.count, 12)
        XCTAssertTrue(path.loop)                       // a closed ellipse
        XCTAssertEqual(path.ease, EaseProfile.none)    // easeInOut stalls at the loop seam
        XCTAssertEqual(path.pacing, .even)
        XCTAssertEqual(path.duration, 24, accuracy: 1e-12)
    }

    func testOvalLoopEllipseAndSineElevation() {
        let path = oval()
        let n = path.keyframes.count
        let e: Float = 0.3
        let rise = Float(20) * .pi / 180
        // The first key = the current camera (θ=0 periapsis: r=d, azimuth yaw, elevation
        // pitch).
        var cam = CameraController()
        cam.target = target; cam.distance = 4; cam.yaw = 0.3; cam.pitch = 0.4
        XCTAssertEqual(simd_distance(path.keyframes[0].pos, cam.eye), 0, accuracy: 1e-4)
        for (i, k) in path.keyframes.enumerated() {
            let th = Float(i) / Float(n) * 2 * .pi
            assertAngleEqual(azimuth(k.pos), 0.3 + th, "closed i/n azimuth, key \(i)")
            XCTAssertEqual(simd_distance(k.pos, target), 4 * (1 + e) / (1 + e * cos(th)),
                           accuracy: 1e-3, "focal ellipse radius r(θ), key \(i)")
            XCTAssertEqual(elevation(k.pos), 0.4 + rise * (1 - cos(th)) / 2,
                           accuracy: 1e-3, "elevation sin²(θ/2) modulation, key \(i)")
        }
        // The far side (θ=π, i=6) is the farthest (apoapsis d·1.3/0.7 ≈ 1.857d) and the
        // highest (+20°).
        XCTAssertEqual(simd_distance(path.keyframes[6].pos, target), 4 * 1.3 / 0.7,
                       accuracy: 1e-3)
        XCTAssertEqual(elevation(path.keyframes[6].pos), 0.4 + rise, accuracy: 1e-3)
    }

    func testOvalLoopClosedDenominatorRule() {
        // The i/n denominator of a closed orbit — the last key sits at yaw + 330° and does
        // not coincide with the first. If you change it to i/(n−1) the first and last keys
        // coincide, producing a zero-length segment; arc-length even pacing then allocates
        // zero time to that span and the rotation snaps at the seam (the same trap as
        // orbit360).
        let path = oval()
        let n = path.keyframes.count
        assertAngleEqual(azimuth(path.keyframes.last!.pos),
                         0.3 + Float(n - 1) / Float(n) * 2 * .pi)
        XCTAssertGreaterThan(
            simd_distance(path.keyframes.last!.pos, path.keyframes[0].pos), 1e-3)
    }

    func testOvalLoopClampsModulatedPitch() {
        // Start 1.45 + a peak modulation of 20° = 1.799 rad → the apoapsis key is clamped to
        // ±1.5 rad as well.
        let path = oval(pitch: 1.45)
        XCTAssertEqual(elevation(path.keyframes[6].pos), 1.5, accuracy: 1e-3)
    }

    // MARK: Common to all seven presets — aim lock (the premise of the pivot decision behind
    // the Task 1 handle)

    /// Pins the aim-lock contract across the **whole** pitch band (final review FIX 1 ③ —
    /// the previous single-point sample at 0.4 sits where Pull Back's det is 14× the 1e-3
    /// threshold, so it could not catch any boundary regression). 1.4 and 1.5 are the band in
    /// which Pull Back used to fail (boundary 1.4133 rad — revert the descent flip of FIX 1 ②
    /// and it fails here); ±1.5 are the clamp endpoints at which a viewer drag pins the pitch.
    /// Clamping the starting value is the VM's responsibility, so the band [−1.5, 1.5] is
    /// exactly the contract range.
    func testAllPresetsGazeTargetAndAreAimLocked() {
        for pitch: Float in [-1.5, -0.8, 0, 0.8, 1.4, 1.5] {
            for path in [orbit(pitch: pitch), reveal(pitch: pitch), rising(pitch: pitch),
                         pullBack(pitch: pitch), spiral(pitch: pitch), crane(pitch: pitch),
                         oval(pitch: pitch)] {
                let label = "\(path.name) pitch \(pitch)"
                for k in path.keyframes {
                    // The gaze ray (−Z) passes within 1e-3 of the target (perpendicular
                    // distance).
                    let forward = k.quatValue.act(SIMD3<Float>(0, 0, -1))
                    let toTarget = target - k.pos
                    let miss = simd_length(toTarget - simd_dot(toTarget, forward) * forward)
                    XCTAssertLessThan(miss, 1e-3, label)
                }
                let pivot = PathTransform.pivot(for: path.keyframes)
                XCTAssertNotNil(pivot, label)
                XCTAssertTrue(pivot!.isAimLocked, label)   // why the handle grabs the orbit center
                XCTAssertEqual(simd_distance(pivot!.point, target), 0, accuracy: 1e-2, label)
            }
        }
    }

    // MARK: Capture trajectory

    /// Views in reverse name order (to verify the sorting) — "v0"…"v7", identity rotation,
    /// tvec=(i,0,0) → center (−i,0,0).
    private func makeModel(count: Int = 8) -> ColmapModel {
        let views = (0..<count).reversed().map { i in
            ColmapView(id: i, name: "v\(i)", cameraId: 1,
                       qvec: SIMD4(1, 0, 0, 0), tvec: SIMD3(Double(i), 0, 0))
        }
        let cams = [1: ColmapIntrinsics(fx: 500, fy: 500, cx: 400, cy: 300,
                                        width: 800, height: 600)]
        return ColmapModel(cameras: cams, views: views)
    }

    func testCaptureTrajectorySortsByNameAndSubsamplesEvenly() {
        // M=8, N=4 → indices after sorting by name: [0, round(7/3)=2, round(14/3)=5, 7]
        let path = PathPresets.captureTrajectory(model: makeModel(), keyCount: 4)
        XCTAssertNotNil(path)
        XCTAssertEqual(path!.keyframes.count, 4)
        XCTAssertFalse(path!.loop)
        XCTAssertEqual(path!.pacing, .even)
        let xs = path!.keyframes.map { $0.pos.x }
        XCTAssertEqual(xs, [0, -2, -5, -7])   // identity R → center = −tvec (independent of the reversed array)
    }

    func testCaptureTrajectoryFovFromIntrinsics() {
        let path = PathPresets.captureTrajectory(model: makeModel(), keyCount: 4)!
        let expected = Float(2 * atan(600.0 / (2 * 500.0)))   // OrbitFit.verticalFOV
        for k in path.keyframes {
            XCTAssertEqual(k.fov, expected, accuracy: 1e-6)
        }
    }

    func testCaptureTrajectoryPoseConversionMatchesCameraConversion() {
        // A non-trivial rotation (world→camera, 90° about Y). The −Z/+Y of the Metal
        // camera→world quat must match CameraConversion.forwardWorld/upWorld (this keeps the
        // fly-through gaze consistent).
        let s: Double = 0.7071067811865476
        let qvec = SIMD4<Double>(s, 0, s, 0)   // (w,x,y,z)
        let rot = PathPresets.metalCameraRotation(qvec: qvec)
        let forward = rot.act(SIMD3<Float>(0, 0, -1))
        let up = rot.act(SIMD3<Float>(0, 1, 0))
        XCTAssertEqual(simd_distance(forward, CameraConversion.forwardWorld(qvec: qvec)),
                       0, accuracy: 1e-4)
        XCTAssertEqual(simd_distance(up, CameraConversion.upWorld(qvec: qvec)),
                       0, accuracy: 1e-4)
    }

    func testCaptureTrajectoryClampsAndRejectsEmpty() {
        XCTAssertNil(PathPresets.captureTrajectory(
            model: ColmapModel(cameras: [:], views: [])))          // zero views → nil
        let three = PathPresets.captureTrajectory(model: makeModel(count: 3), keyCount: 10)
        XCTAssertEqual(three?.keyframes.count, 3)                  // clamped to the view count
        let one = PathPresets.captureTrajectory(model: makeModel(count: 1), keyCount: 10)
        XCTAssertEqual(one?.keyframes.count, 1)                    // a single view = a static shot
    }

    // MARK: VM application (spec UX 6 — applying = creating a new path)

    func testApplyOrbitPresetAddsSelectsPersistsAndKeepsExisting() {
        let vm = CinemaViewModel()
        vm.load(paths: [CameraPath(name: "existing", keyframes: [], duration: 5)])
        let viewer = ViewerModel()
        viewer.camera.setOrbit(target: .zero, distance: 3, yaw: 0.1, pitch: 0.2)
        vm.viewer = viewer
        var saves = 0
        vm.onPersist = { _ in saves += 1 }

        vm.applyOrbitPreset()
        XCTAssertEqual(vm.paths.count, 2)
        XCTAssertEqual(vm.paths[0].name, "existing")               // existing path unchanged
        XCTAssertTrue(vm.paths[0].keyframes.isEmpty)
        XCTAssertEqual(vm.paths[1].name, "Orbit 360°")
        XCTAssertEqual(vm.paths[1].keyframes.count, 8)
        XCTAssertEqual(vm.selectedPathID, vm.paths[1].id)          // the new path is selected
        XCTAssertEqual(saves, 1)

        vm.applyOrbitPreset()                                      // name uniquification
        XCTAssertEqual(vm.paths[2].name, "Orbit 360° 2")
    }

    func testApplyCapturePresetUsesContextModel() {
        let vm = CinemaViewModel()
        var saves = 0
        vm.onPersist = { _ in saves += 1 }
        let model = makeModel()
        let context = ColmapContext(model: model, middleView: model.views[0],
                                    pointsCentroid: nil)
        vm.applyCapturePreset(context: context)
        XCTAssertEqual(vm.paths.count, 1)
        XCTAssertEqual(vm.paths[0].name, "Capture Path")
        XCTAssertEqual(vm.paths[0].keyframes.count, 8)             // min(10, 8)
        XCTAssertEqual(vm.selectedPathID, vm.paths[0].id)
        XCTAssertEqual(saves, 1)
    }

    func testApplyRevealArcAndRisingOrbitPresetsAddSelectPersist() {
        let vm = CinemaViewModel()
        let viewer = ViewerModel()
        viewer.camera.setOrbit(target: .zero, distance: 3, yaw: 0.1, pitch: 0.2)
        vm.viewer = viewer
        var saves = 0
        vm.onPersist = { _ in saves += 1 }

        vm.applyRevealArcPreset()
        XCTAssertEqual(vm.paths.count, 1)
        XCTAssertEqual(vm.paths[0].name, "Reveal Arc")
        XCTAssertEqual(vm.paths[0].keyframes.count, 6)
        XCTAssertEqual(vm.selectedPathID, vm.paths[0].id)

        vm.applyRisingOrbitPreset()
        XCTAssertEqual(vm.paths.count, 2)
        XCTAssertEqual(vm.paths[0].keyframes.count, 6)             // existing path unchanged
        XCTAssertEqual(vm.paths[1].name, "Rising Orbit")
        XCTAssertEqual(vm.paths[1].keyframes.count, 10)
        XCTAssertEqual(vm.selectedPathID, vm.paths[1].id)
        XCTAssertEqual(saves, 2)
    }

    /// The entry points of the four new presets (panel redesign Task 4) — the same structure
    /// as applyOrbitPreset (based on the orbit state, pitch clamped to ±1.5, and routed through
    /// addPresetPath = add, select, persist).
    func testApplyNewPresetEntryPointsAddSelectPersist() {
        let vm = CinemaViewModel()
        let viewer = ViewerModel()
        viewer.camera.setOrbit(target: .zero, distance: 3, yaw: 0.1, pitch: 0.2)
        vm.viewer = viewer
        var saves = 0
        vm.onPersist = { _ in saves += 1 }

        vm.applyPullBackRevealPreset()
        vm.applySpiralDescentPreset()
        vm.applyCraneDownPreset()
        vm.applyOvalLoopPreset()
        XCTAssertEqual(vm.paths.map(\.name),
                       ["Pull Back Reveal", "Spiral Descent", "Crane Down", "Oval Loop"])
        XCTAssertEqual(vm.paths.map(\.keyframes.count), [6, 12, 6, 12])
        XCTAssertEqual(vm.selectedPathID, vm.paths[3].id)          // the last one applied is selected
        XCTAssertEqual(saves, 4)
    }
}
