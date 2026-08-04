// Cinema Task 4 (2026-07-26 spec UX 6): two presets — Orbit 360° (8 keys on a circle around the
// current orbit camera, plus loop) and the capture trajectory (an even subsample of the COLMAP
// registered views in image-name-sorted order).
// Path transform Task 3 (2026-07-27 spec §7): Orbit 360° at 20 seconds (in the conventional
// 18°/s band) + Reveal Arc (a 220° open arc — the current view is the middle of the arc) +
// Rising Orbit (a 360° open helix — pitch rising by +25° and the radius converging to 0.85d).
// A closed circle uses the denominator i/n and an open arc uses i/(n−1) — using i/(n−1) on a
// closed circle makes the first and last keys coincide, producing a zero-length segment, and the
// arc-length constant speed snaps the rotation.
// Panel redesign Task 4 (2026-07-29 spec §6): four new ones — Pull Back Reveal (fixed azimuth, a
// dolly-out to distance 2.5d with elevation +20°) + Spiral Descent (a 540° descending helix — the
// last key is the current view, "it sits down in front") + Crane Down (azimuth and distance
// unchanged, only the elevation descending from max(pitch, 1.0) to 0.05) + Oval Loop (an ellipse
// with target at the focus, r(θ)=d(1+e)/(1+e·cosθ), with sin²(θ/2) elevation modulation, a closed
// loop). The denominator rules and the ±1.5 pitch clamp follow the conventions of the existing
// three exactly.
// The geometry is pure functions (PathPresetsTests is the source of truth), and applying goes
// through CinemaViewModel.addPresetPath = **creating a new path** (existing paths are unchanged —
// spec: after applying, the keyframes can be edited normally).
import Foundation
import simd

enum PathPresets {

    /// Orbit 360°: keyCount keys on a circle around target, plus loop — a closed circle. It takes
    /// the current orbit parameters (target, distance, yaw, pitch, and upAxis — the
    /// CameraController convention) as-is so that the first key is exactly the current camera
    /// position (reproducing the eye formula — the same basis as CameraController.eye).
    /// ease is .none — the goal is constant turntable speed, and easeInOut pauses at every loop
    /// seam (the same rationale as the orange hint in the inspector — handed over from Task 1).
    static func orbit360(target: SIMD3<Float>, distance: Float, yaw: Float, pitch: Float,
                         upAxis: SIMD3<Float>, fov: Float,
                         name: String = "Orbit 360°", keyCount: Int = 8,
                         duration: Double = 20) -> CameraPath {
        let n = max(keyCount, 3)   // minimum representation of a circle (under 3 keys the CR circle degenerates)
        let (bx, bu, bz) = CameraController.orbitBasis(upAxis: upAxis)
        let cp = cos(pitch), sp = sin(pitch)
        var keys: [Keyframe] = []
        keys.reserveCapacity(n)
        for i in 0..<n {
            let y = yaw + Float(i) / Float(n) * 2 * .pi
            let dir = (cp * sin(y)) * bx + sp * bu + (cp * cos(y)) * bz
            let pos = target + dir * distance
            // time is not consumed under even pacing (a residual stored value) — even spacing is
            // recorded for hygiene.
            keys.append(Keyframe(time: Double(i) / Double(n) * duration, pos: pos,
                                 rotation: lookRotation(eye: pos, target: target, up: bu),
                                 fov: fov))
        }
        return CameraPath(name: name, keyframes: keys, ease: .none, loop: true,
                          pacing: .even, duration: duration)
    }

    /// Reveal Arc: a 220° open arc in which the current view is the **middle** of the arc
    /// (yaw−110° → yaw+110°), with constant radius and height. The rationale is that it is
    /// natural for the composition the user set up to be the center of the shot (spec §7 —
    /// putting it at the end would spend most of the time at other angles). It is an open path,
    /// so easeInOut (ease in and out at the ends — the same convention as the capture
    /// trajectory). The denominator is i/(n−1) — the first and last keys land exactly on the two
    /// ends of the arc (different from the closed circle's i/n).
    static func revealArc(target: SIMD3<Float>, distance: Float, yaw: Float, pitch: Float,
                          upAxis: SIMD3<Float>, fov: Float,
                          name: String = "Reveal Arc", keyCount: Int = 6,
                          duration: Double = 12) -> CameraPath {
        let n = max(keyCount, 2)   // minimum representation of an open arc (one segment)
        let (bx, bu, bz) = CameraController.orbitBasis(upAxis: upAxis)
        let sweep = Float(220) * .pi / 180
        let cp = cos(pitch), sp = sin(pitch)
        var keys: [Keyframe] = []
        keys.reserveCapacity(n)
        for i in 0..<n {
            let t = Float(i) / Float(n - 1)
            let y = yaw - sweep / 2 + t * sweep
            let dir = (cp * sin(y)) * bx + sp * bu + (cp * cos(y)) * bz
            let pos = target + dir * distance
            // time is derived directly in Double rather than through t (Float) — so that Float
            // precision does not leak into the stored value (not consumed under even pacing; a
            // hygiene measure).
            keys.append(Keyframe(time: Double(i) / Double(n - 1) * duration, pos: pos,
                                 rotation: lookRotation(eye: pos, target: target, up: bu),
                                 fov: fov))
        }
        return CameraPath(name: name, keyframes: keys, ease: .easeInOut, loop: false,
                          pacing: .even, duration: duration)
    }

    /// Rising Orbit: an open helix that goes a full turn (360°) from yaw while pitch rises
    /// linearly by +25° (the end value is clamped to ±1.5 rad as well — clamping the start value
    /// is owned by the VM) and the radius converges linearly to 0.85d — the "rising and closing
    /// in" hero shot. The start and end heights differ so it does not close, which forces
    /// loop = false (spec §7). The last key is at the same azimuth as the start
    /// (i/(n−1)·360°) but higher and nearer.
    static func risingOrbit(target: SIMD3<Float>, distance: Float, yaw: Float, pitch: Float,
                            upAxis: SIMD3<Float>, fov: Float,
                            name: String = "Rising Orbit", keyCount: Int = 10,
                            duration: Double = 20) -> CameraPath {
        let n = max(keyCount, 2)   // minimum representation of an open helix (one segment)
        let (bx, bu, bz) = CameraController.orbitBasis(upAxis: upAxis)
        let rise = Float(25) * .pi / 180
        var keys: [Keyframe] = []
        keys.reserveCapacity(n)
        for i in 0..<n {
            let t = Float(i) / Float(n - 1)
            let y = yaw + t * 2 * .pi
            let p = min(max(pitch + t * rise, -1.5), 1.5)   // same range as the orbit() clamp
            let d = distance * (1 - 0.15 * t)               // d → 0.85d, linear
            let dir = (cos(p) * sin(y)) * bx + sin(p) * bu + (cos(p) * cos(y)) * bz
            let pos = target + dir * d
            // time is derived directly in Double rather than through t (Float) — so that Float
            // precision does not leak into the stored value (not consumed under even pacing; a
            // hygiene measure).
            keys.append(Keyframe(time: Double(i) / Double(n - 1) * duration, pos: pos,
                                 rotation: lookRotation(eye: pos, target: target, up: bu),
                                 fov: fov))
        }
        return CameraPath(name: name, keyframes: keys, ease: .easeInOut, loop: false,
                          pacing: .even, duration: duration)
    }

    /// Pull Back Reveal: an open dolly-out that keeps the azimuth fixed, pulls back (distance
    /// d → 2.5d), and swings the elevation by 20° to reveal the whole scene (spec §6) — the first
    /// key = the current camera (the user's composition is the shot's starting point, the same
    /// convention as orbit360). The default is a +20° rise, but if there is no headroom for the
    /// rise (starting pitch + 20° > 1.5) it **flips to a descent** — guaranteeing a fixed aim
    /// across the entire pitch band (see the formula comment in the body, spec footnote 1, and
    /// final review FIX 1 ②). Denominator i/(n−1) — it is an open trajectory, so the first and
    /// last keys land exactly on the two ends of the range. Clamping the start value is owned by
    /// the VM (the same division of labor as risingOrbit).
    static func pullBackReveal(target: SIMD3<Float>, distance: Float, yaw: Float, pitch: Float,
                               upAxis: SIMD3<Float>, fov: Float,
                               name: String = "Pull Back Reveal", keyCount: Int = 6,
                               duration: Double = 10) -> CameraPath {
        let n = max(keyCount, 2)   // minimum representation of an open trajectory (one segment)
        let (bx, bu, bz) = CameraController.orbitBasis(upAxis: upAxis)
        let rise = Float(20) * .pi / 180
        // Guaranteeing a fixed aim (final review FIX 1 ② — spec footnote 1): the azimuth is
        // fixed, so all n view directions lie in a single vertical plane, and in that case
        // PathTransform's decision expression collapses to the closed form
        //   det(A/n) = (1 − R²)/4,   R = |Σ_k exp(i·2p_k)| / n   (the length of the sum vector of
        //                                                        the doubled elevation angles)
        // (the out-of-plane eigenvalue 1 × the in-plane (1−R)/2·(1+R)/2). The narrower the
        // elevation spread, the closer R → 1, so if the 20° rise is cut off at the 1.5 upper
        // bound, det falls below the 1e-3 threshold — measured: starting pitch 1.41 → det
        // 1.078e-3, passes / 1.42 → 8.563e-4, fails (the boundary is 1.4133 rad). A viewer drag
        // pins pitch at ±1.5, so reaching it takes a single drag. The remedy: if there is no
        // headroom for the rise (pitch + rise > 1.5), flip to a descent — within the band
        // [−1.5, 1.5] the headroom on the opposite side is always ≥ 3 − rise ≈ 2.65 rad, so
        // either branch secures the full 20° spread, and with 6 evenly spaced keys
        // R = sin(3h)/(6·sin(h/2)) (h = 2·rise/5) → det ≈ 1.39e-2, 14× the threshold. The nature
        // of the shot (pulling back to reveal; the first key = the current view) is identical in
        // both branches.
        let signedRise: Float = pitch + rise <= 1.5 ? rise : -rise
        var keys: [Keyframe] = []
        keys.reserveCapacity(n)
        for i in 0..<n {
            let t = Float(i) / Float(n - 1)
            // The clamp guards against out-of-band input — within the band [−1.5, 1.5] the flip
            // above keeps it from firing.
            let p = min(max(pitch + t * signedRise, -1.5), 1.5)
            let d = distance * (1 + 1.5 * t)                // d → 2.5d, linear
            let dir = (cos(p) * sin(yaw)) * bx + sin(p) * bu + (cos(p) * cos(yaw)) * bz
            let pos = target + dir * d
            keys.append(Keyframe(time: Double(i) / Double(n - 1) * duration, pos: pos,
                                 rotation: lookRotation(eye: pos, target: target, up: bu),
                                 fov: fov))
        }
        return CameraPath(name: name, keyframes: keys, ease: .easeInOut, loop: false,
                          pacing: .even, duration: duration)
    }

    /// Spiral Descent: an open helix that starts high (pitch +35°) and far (2.5d), turns one and
    /// a half times (540°) while descending, and **sits down at the current camera position**
    /// (spec §6) — the opposite of risingOrbit, **the last key is the current view** ("it sits
    /// down in front" — the composition the user set up is the shot's landing point, so the final
    /// frame is framed directly in the viewport). The azimuth advances in the same + direction as
    /// the existing convention via yaw + (t−1)·540°, and the starting elevation is clamped to
    /// ±1.5 rad as well. duration 30s = 540° ÷ 18°/s (the orbit360 conventional band).
    static func spiralDescent(target: SIMD3<Float>, distance: Float, yaw: Float, pitch: Float,
                              upAxis: SIMD3<Float>, fov: Float,
                              name: String = "Spiral Descent", keyCount: Int = 12,
                              duration: Double = 30) -> CameraPath {
        let n = max(keyCount, 2)   // minimum representation of an open helix (one segment)
        let (bx, bu, bz) = CameraController.orbitBasis(upAxis: upAxis)
        let sweep = Float(540) * .pi / 180
        let drop = Float(35) * .pi / 180
        var keys: [Keyframe] = []
        keys.reserveCapacity(n)
        for i in 0..<n {
            let t = Float(i) / Float(n - 1)
            let y = yaw + (t - 1) * sweep                        // the end (t=1) is exactly yaw
            let p = min(max(pitch + (1 - t) * drop, -1.5), 1.5)  // +35° → descends to pitch
            let d = distance * (1 + 1.5 * (1 - t))               // 2.5d → converges to d
            let dir = (cos(p) * sin(y)) * bx + sin(p) * bu + (cos(p) * cos(y)) * bz
            let pos = target + dir * d
            keys.append(Keyframe(time: Double(i) / Double(n - 1) * duration, pos: pos,
                                 rotation: lookRotation(eye: pos, target: target, up: bu),
                                 fov: fov))
        }
        return CameraPath(name: name, keyframes: keys, ease: .easeInOut, loop: false,
                          pacing: .even, duration: duration)
    }

    /// Crane Down: a linear descent of **the elevation only**, from a looking-down start to eye
    /// level (0.05 rad) — the azimuth and distance are unchanged (spec §6 table, "elevation
    /// descends only"). The starting elevation is max(pitch, 1.0 rad ≈ 57°): if the current pitch
    /// is higher, that value is kept (respecting the user's composition); if lower, it is raised
    /// for the essence of a crane shot, which starts from a high angle — this floor guarantees a
    /// view-direction spread ≥ 0.95 rad, so the fixed-aim decision (the det condition) always
    /// holds. With min(…, 1.5) at the start and the constant 0.05 at the end, both ends are
    /// inside the ±1.5 clamp range.
    static func craneDown(target: SIMD3<Float>, distance: Float, yaw: Float, pitch: Float,
                          upAxis: SIMD3<Float>, fov: Float,
                          name: String = "Crane Down", keyCount: Int = 6,
                          duration: Double = 12) -> CameraPath {
        let n = max(keyCount, 2)   // minimum representation of an open descent (one segment)
        let (bx, bu, bz) = CameraController.orbitBasis(upAxis: upAxis)
        let hi = min(max(pitch, 1.0), 1.5)
        let lo: Float = 0.05
        var keys: [Keyframe] = []
        keys.reserveCapacity(n)
        for i in 0..<n {
            let t = Float(i) / Float(n - 1)
            let p = hi + t * (lo - hi)
            let dir = (cos(p) * sin(yaw)) * bx + sin(p) * bu + (cos(p) * cos(yaw)) * bz
            let pos = target + dir * distance
            keys.append(Keyframe(time: Double(i) / Double(n - 1) * duration, pos: pos,
                                 rotation: lookRotation(eye: pos, target: target, up: bu),
                                 fov: fov))
        }
        return CameraPath(name: name, keyframes: keys, ease: .easeInOut, loop: false,
                          pacing: .even, duration: duration)
    }

    /// Oval Loop: an elliptical orbit with target at the **focus** (polar form
    /// r(θ) = d(1+e)/(1+e·cosθ), e = 0.3 — the periapsis at θ=0 is exactly the current distance
    /// d, and the apoapsis on the far side is ≈ 1.86d) plus a sinusoidal elevation modulation
    /// pitch + 20°·(1−cosθ)/2 = sin²(θ/2) (spec §6, "elliptical orbit + sinusoidal elevation
    /// modulation") — it is highest on the far side, and the modulation's derivative is 0 at the
    /// seam (θ=0), so the loop is smooth. It is a closed trajectory, so the denominator is i/n
    /// (with i/(n−1) the first and last keys would coincide and create a zero-length segment —
    /// the same trap as orbit360), and ease is .none (easeInOut pauses at every loop seam — the
    /// same rationale as orbit360).
    /// The modulated elevation is clamped to ±1.5 rad per key. duration 24s — the orbit's mean
    /// radius is larger than the circle's (the 20s convention), so it runs slower.
    static func ovalLoop(target: SIMD3<Float>, distance: Float, yaw: Float, pitch: Float,
                         upAxis: SIMD3<Float>, fov: Float,
                         name: String = "Oval Loop", keyCount: Int = 12,
                         duration: Double = 24) -> CameraPath {
        let n = max(keyCount, 3)   // minimum representation of a closed trajectory (under 3 keys the CR circle degenerates)
        let (bx, bu, bz) = CameraController.orbitBasis(upAxis: upAxis)
        let e: Float = 0.3
        let rise = Float(20) * .pi / 180
        var keys: [Keyframe] = []
        keys.reserveCapacity(n)
        for i in 0..<n {
            let th = Float(i) / Float(n) * 2 * .pi
            let y = yaw + th
            let p = min(max(pitch + rise * (1 - cos(th)) / 2, -1.5), 1.5)
            let d = distance * (1 + e) / (1 + e * cos(th))
            let dir = (cos(p) * sin(y)) * bx + sin(p) * bu + (cos(p) * cos(y)) * bz
            let pos = target + dir * d
            keys.append(Keyframe(time: Double(i) / Double(n) * duration, pos: pos,
                                 rotation: lookRotation(eye: pos, target: target, up: bu),
                                 fov: fov))
        }
        return CameraPath(name: name, keyframes: keys, ease: .none, loop: true,
                          pacing: .even, duration: duration)
    }

    /// The capture trajectory: it **sorts the COLMAP registered views by name** (the same
    /// convention as ColmapContext.load — the storage order in images.bin is arbitrary), takes an
    /// even subsample of keyCount of them, and builds keys from each view's actual pose (pos plus
    /// the Metal camera→world quat) and the fov derived from the intrinsics.
    /// Zero views returns nil (the basis for disabling the button). If keyCount exceeds the view
    /// count, it is clamped to the view count.
    static func captureTrajectory(model: ColmapModel, keyCount: Int = 10,
                                  name: String = "Capture Path",
                                  duration: Double = 10) -> CameraPath? {
        let sorted = model.views.sorted { $0.name < $1.name }
        guard !sorted.isEmpty else { return nil }
        let m = sorted.count
        let n = min(max(keyCount, 2), m)
        // Evenly spaced indices that include the first and last views. Since n ≤ m the spacing is
        // ≥ 1, so the sequence stays strictly increasing even after rounding.
        let indices: [Int] = m == 1 ? [0]
            : (0..<n).map { Int((Double($0) * Double(m - 1) / Double(n - 1)).rounded()) }
        var keys: [Keyframe] = []
        keys.reserveCapacity(indices.count)
        for (k, i) in indices.enumerated() {
            let v = sorted[i]
            keys.append(Keyframe(
                time: Double(k) / Double(max(indices.count - 1, 1)) * duration,
                pos: CameraConversion.cameraCenter(qvec: v.qvec, tvec: v.tvec),
                rotation: metalCameraRotation(qvec: v.qvec),
                fov: model.cameras[v.cameraId]
                    .map { OrbitFit.verticalFOV(fy: $0.fy, height: $0.height) } ?? 0.9))
        }
        // An open path — easeInOut softens the start and end (the portfolio flythrough
        // convention).
        return CameraPath(name: name, keyframes: keys, ease: .easeInOut, loop: false,
                          pacing: .even, duration: duration)
    }

    // MARK: Rotation derivation (pure)

    /// The camera→world rotation looking at target from eye. The top-left 3×3 of the lookAt view
    /// matrix is (camera→world)ᵀ, so it is recovered by moving the rows into columns — the same
    /// procedure as CameraPose.effective.
    static func lookRotation(eye: SIMD3<Float>, target: SIMD3<Float>,
                             up: SIMD3<Float>) -> simd_quatf {
        let v = CameraController.lookAt(eye: eye, center: target, up: up)
        let c = simd_float3x3(columns: (
            SIMD3(v.columns.0.x, v.columns.1.x, v.columns.2.x),
            SIMD3(v.columns.0.y, v.columns.1.y, v.columns.2.y),
            SIMD3(v.columns.0.z, v.columns.1.z, v.columns.2.z)))
        return simd_quatf(c).normalized
    }

    /// COLMAP qvec (world→camera, +Y down and +Z forward) → the Metal camera→world rotation
    /// (forward −Z).
    /// The transpose of the top-left 3×3 of CameraConversion.viewMatrix (flip·R) = Rᵀ·flip — the
    /// flip diagonal (1,−1,−1) has det=+1, so it stays a proper rotation. PathPresetsTests pins
    /// that forward/up agree with CameraConversion.forwardWorld/upWorld.
    static func metalCameraRotation(qvec: SIMD4<Double>) -> simd_quatf {
        let camToWorld = CameraConversion.rotation(qvec).transpose
            * simd_float3x3(diagonal: SIMD3<Float>(1, -1, -1))
        return simd_quatf(camToWorld).normalized
    }
}

// MARK: - VM application (wiring — called by the New Path ▾ pulldown of the inspector's PATHS
// section, panel redesign Task 4)

extension CinemaViewModel {
    /// Adds and selects a new Orbit 360° path based on the current orbit camera.
    /// The reference is the **orbit state**, independent of overridePose (during skimming) —
    /// since an override preserves the orbit (the CameraController contract), it circles "the
    /// view the user last set up".
    func applyOrbitPreset() {
        guard let cam = viewer?.camera else { return }
        addPresetPath(PathPresets.orbit360(
            target: cam.target, distance: cam.distance, yaw: cam.yaw,
            pitch: min(max(cam.pitch, -1.5), 1.5),   // same guard range as the orbit() clamp
            upAxis: cam.upAxis, fov: cam.verticalFOV))
    }

    /// Adds and selects a new Reveal Arc path based on the current orbit camera (the same
    /// structure as applyOrbitPreset — the orbit state as reference and the same pitch clamp
    /// guard).
    func applyRevealArcPreset() {
        guard let cam = viewer?.camera else { return }
        addPresetPath(PathPresets.revealArc(
            target: cam.target, distance: cam.distance, yaw: cam.yaw,
            pitch: min(max(cam.pitch, -1.5), 1.5),
            upAxis: cam.upAxis, fov: cam.verticalFOV))
    }

    /// Adds and selects a new Rising Orbit path based on the current orbit camera (the same
    /// structure — clamping the end value after the rise is owned inside
    /// PathPresets.risingOrbit).
    func applyRisingOrbitPreset() {
        guard let cam = viewer?.camera else { return }
        addPresetPath(PathPresets.risingOrbit(
            target: cam.target, distance: cam.distance, yaw: cam.yaw,
            pitch: min(max(cam.pitch, -1.5), 1.5),
            upAxis: cam.upAxis, fov: cam.verticalFOV))
    }

    /// Adds and selects a new Pull Back Reveal path based on the current orbit camera (the same
    /// structure — clamping the end value after the rise is owned inside
    /// PathPresets.pullBackReveal).
    func applyPullBackRevealPreset() {
        guard let cam = viewer?.camera else { return }
        addPresetPath(PathPresets.pullBackReveal(
            target: cam.target, distance: cam.distance, yaw: cam.yaw,
            pitch: min(max(cam.pitch, -1.5), 1.5),
            upAxis: cam.upAxis, fov: cam.verticalFOV))
    }

    /// Adds and selects a new Spiral Descent path based on the current orbit camera (the same
    /// structure — the landing semantics where the last key becomes the current view are
    /// authoritative in the PathPresets.spiralDescent comment).
    func applySpiralDescentPreset() {
        guard let cam = viewer?.camera else { return }
        addPresetPath(PathPresets.spiralDescent(
            target: cam.target, distance: cam.distance, yaw: cam.yaw,
            pitch: min(max(cam.pitch, -1.5), 1.5),
            upAxis: cam.upAxis, fov: cam.verticalFOV))
    }

    /// Adds and selects a new Crane Down path based on the current orbit camera (the same
    /// structure — the high-angle floor max(pitch, 1.0) is owned inside PathPresets.craneDown).
    func applyCraneDownPreset() {
        guard let cam = viewer?.camera else { return }
        addPresetPath(PathPresets.craneDown(
            target: cam.target, distance: cam.distance, yaw: cam.yaw,
            pitch: min(max(cam.pitch, -1.5), 1.5),
            upAxis: cam.upAxis, fov: cam.verticalFOV))
    }

    /// Adds and selects a new Oval Loop path based on the current orbit camera (the same
    /// structure — the per-key clamp of the modulated elevation is owned inside
    /// PathPresets.ovalLoop).
    func applyOvalLoopPreset() {
        guard let cam = viewer?.camera else { return }
        addPresetPath(PathPresets.ovalLoop(
            target: cam.target, distance: cam.distance, yaw: cam.yaw,
            pitch: min(max(cam.pitch, -1.5), 1.5),
            upAxis: cam.upAxis, fov: cam.verticalFOV))
    }

    /// Adds and selects a new capture trajectory path. If there is no context (non-COLMAP), the
    /// caller disables the button — here there is only the empty-views guard.
    func applyCapturePreset(context: ColmapContext) {
        guard let path = PathPresets.captureTrajectory(model: context.model) else { return }
        addPresetPath(path)
    }
}
