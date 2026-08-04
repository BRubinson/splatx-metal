// Cinema Task 1 (2026-07-26 spec, "path math" section): the path math core — pure Swift, zero
// external dependencies.
// Written in-house (standard formulas) — centripetal Catmull-Rom positions (α=0.5, the
// Barry–Goldman pyramid), per-segment quaternion slerp (hemisphere normalization — the dot<0 sign
// flip is required), linear FOV per segment, arc-length constant speed (a 32-sample cumulative
// table per segment plus a binary-search inversion), quintic ease (6t⁵−15t⁴+10t³) composed on the
// time axis, and closed end conditions for loop (cyclic control points).
//
// Contract (plan Task 1 Interfaces):
// - pose(at t): t is in **seconds**, 0…duration. Out of range is clamped (wrapping for repeated
//   loop playback is the caller's job).
// - init(path:) rebuilds in <1ms — guaranteed structurally rather than by a timing assertion: an
//   O(n) build with a fixed 32 samples per segment (a few thousand CR evaluations for a few dozen
//   keys). After that it is fully immutable (a value type with no side effects).
// - Degeneracies: 0 keys = the neutral pose (feature inactive), 1 key = a still shot, 2 keys = the
//   reflected phantom (P₋₁=2P₀−P₁) creates collinear control points and degenerates naturally into
//   a lerp (no separate branch), and adjacent identical positions get an ε floor on the knot Δt.
// - A fully stationary path (total arc length ≈ 0) falls back to a uniform parameter, preserving
//   rotation and FOV interpolation (a pan shot).
// - Constant speed is measured by positional arc length: a zero-length segment (a duplicate key)
//   occupies zero time, so the rotation change over that segment snaps (per-segment dwell time is
//   a non-goal for v1 — spec).
//
// Hybrid pacing (spec footnote 1 — the Task 3 extension):
// - `.even` (the default): Keyframe.time is not consumed (array order = path order). The key
//   pass-through times are **derived** by `keyframeTimes` — r_i = the key's arc-length fraction,
//   inverted on the time axis by the quintic inverse (the monotone bisection `quinticInverse`)
//   when ease is active.
// - `.manual`: Keyframe.time (seconds) is authoritative for the pass-through times. The normalized
//   times u_i (clamped and monotonically corrected) are mapped monotonically **piecewise-linearly**
//   to the keys' cumulative arc lengths D_i, and then the existing distance inversion is reused.
//   Constant speed within a segment, C0 at key boundaries — the continuity of the spatial curve
//   itself (CR) is preserved regardless of pacing.
//   For loop, a wrap node (u=1, D=L) distributes the last→first segment. Before the first key and
//   after the last key it holds (keeping the end pose). The global ease is **not applied** in
//   manual mode (it is incompatible with respecting the key times — see the PacingMode comment).
import Foundation
import simd

struct PathSampler {
    /// The number of arc-length samples per segment — a fixed value (the basis of the rebuild-cost
    /// structural contract).
    static let samplesPerSegment = 32
    /// The knot spacing floor for adjacent keys at identical positions (Δt=0 → prevents division by
    /// zero, the spec degeneracy contract).
    private static let knotEpsilon = 1e-6
    /// If the total arc length is at or below this, it is treated as a "fully stationary path" and
    /// falls back to a uniform parameter.
    private static let zeroLengthEpsilon = 1e-9

    /// Everything the Barry–Goldman evaluation needs: 4 control points + the centripetal knots +
    /// the key indices + the arc-length table.
    private struct Segment {
        var p0, p1, p2, p3: SIMD3<Double>
        var t0, t1, t2, t3: Double
        var i1, i2: Int              // the key indices at both ends of the segment (for quat and fov interpolation)
        var cumulative: [Double]     // 33 entries — the chord length accumulated up to τ=j/32
        var length: Double { cumulative[PathSampler.samplesPerSegment] }
    }

    let path: CameraPath
    private var segments: [Segment] = []
    private var segmentStart: [Double] = []   // each segment's global start distance (for the binary search)
    private(set) var totalLength: Double = 0
    private var quats: [simd_quatf] = []      // the key rotations — a normalized cache
    /// The manual pacing nodes (the normalized time u — clamped and monotonically corrected — and
    /// the key's cumulative arc length d).
    /// Key i ↔ node i, plus a wrap node (1, totalLength) when loop — the node count − 1 ==
    /// segmentCount.
    private var manualNodes: [(u: Double, d: Double)] = []

    var segmentCount: Int { segments.count }

    init(path: CameraPath) {
        self.path = path
        let keys = path.keyframes
        quats = keys.map { k in
            let v = k.quatValue.vector
            let len = simd_length(v)
            // Guard against corrupted JSON (a zero vector) — substitute identity
            return len > 1e-12 ? simd_quatf(vector: v / len) : simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        }
        let n = keys.count
        guard n >= 2 else { return }   // 0 or 1 keys means no segments (the neutral pose / a still shot)
        let p = keys.map { k in
            SIMD3<Double>(Double(k.pos.x), Double(k.pos.y), Double(k.pos.z))
        }
        let segN = path.loop ? n : n - 1
        segments.reserveCapacity(segN)
        segmentStart.reserveCapacity(segN)
        var running = 0.0
        for i in 0..<segN {
            let i1 = i, i2 = (i + 1) % n
            let c0: SIMD3<Double>, c3: SIMD3<Double>
            if path.loop {
                c0 = p[(i - 1 + n) % n]          // closed end condition — cyclic control points
                c3 = p[(i + 2) % n]
            } else {
                c0 = i == 0 ? 2 * p[0] - p[1] : p[i - 1]              // endpoint reflected phantom
                c3 = i == n - 2 ? 2 * p[n - 1] - p[n - 2] : p[i + 2]
            }
            let t0 = 0.0
            let t1 = t0 + Self.knotDelta(c0, p[i1])
            let t2 = t1 + Self.knotDelta(p[i1], p[i2])
            let t3 = t2 + Self.knotDelta(p[i2], c3)
            var seg = Segment(p0: c0, p1: p[i1], p2: p[i2], p3: c3,
                              t0: t0, t1: t1, t2: t2, t3: t3,
                              i1: i1, i2: i2, cumulative: [])
            var cum = [Double](repeating: 0, count: Self.samplesPerSegment + 1)
            var prev = Self.point(seg, 0)
            for j in 1...Self.samplesPerSegment {
                let pt = Self.point(seg, Double(j) / Double(Self.samplesPerSegment))
                cum[j] = cum[j - 1] + simd_distance(prev, pt)
                prev = pt
            }
            seg.cumulative = cum
            segmentStart.append(running)
            running += cum[Self.samplesPerSegment]
            segments.append(seg)
        }
        totalLength = running

        // The manual pacing nodes: u_i = clamp01(time_i/duration) with a monotonic correction (a
        // running max), and d_i = the cumulative arc length up to key i. loop closes it with the
        // wrap node (1, L).
        let dur = path.duration
        var monotone = 0.0
        for i in 0..<n {
            var u = dur > 0 ? min(max(keys[i].time / dur, 0), 1) : 0
            u = max(u, monotone); monotone = u
            manualNodes.append((u: u, d: keyDistance(i)))
        }
        if path.loop { manualNodes.append((u: 1, d: totalLength)) }
    }

    /// The cumulative arc length up to key i (for an open path the last key = the total length).
    private func keyDistance(_ i: Int) -> Double {
        i < segmentStart.count ? segmentStart[i] : totalLength
    }

    /// The camera pose at t (seconds, 0…duration — out of range is clamped).
    func pose(at t: Double) -> (pos: SIMD3<Float>, quat: simd_quatf, fov: Float) {
        guard let first = path.keyframes.first else {
            // 0 keys = feature inactive — the neutral pose (the same as the viewer's default FOV
            // of 0.9 rad)
            return (SIMD3<Float>(repeating: 0), simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), 0.9)
        }
        guard !segments.isEmpty else {
            return (first.pos, quats[0], first.fov)   // 1 key = a still shot
        }
        if path.pacing == .manual {
            let (idx, tau) = manualParameter(at: t)
            return assemble(idx, tau)
        }
        var s = 0.0
        if path.duration > 0 { s = min(max(t / path.duration, 0), 1) }
        if path.ease == .easeInOut { s = Self.quintic(s) }
        let (idx, tau) = parameter(atNormalized: s)
        return assemble(idx, tau)
    }

    /// (segment, τ) → assembling the pose — shared by the constant-speed and manual paths.
    private func assemble(_ idx: Int, _ tau: Double) -> (pos: SIMD3<Float>, quat: simd_quatf, fov: Float) {
        let seg = segments[idx]
        let pd = Self.point(seg, tau)
        let pos = SIMD3<Float>(Float(pd.x), Float(pd.y), Float(pd.z))
        let quat = Self.slerpShortest(quats[seg.i1], quats[seg.i2], Float(tau))
        let f1 = path.keyframes[seg.i1].fov, f2 = path.keyframes[seg.i2].fov
        return (pos, quat, f1 + (f2 - f1) * Float(tau))
    }

    // MARK: Deriving the key pass-through times (authoritative for the timeline diamonds — spec
    // footnote 1)

    /// The actual pass-through time (seconds) of each key. Under `.even` it is derived from the
    /// arc-length fractions (via the quintic inverse when ease is active); under `.manual` it is
    /// the stored time clamped and monotonically corrected (= the same normalization as
    /// manualNodes).
    /// 0 keys = an empty array; 1 key = [0] (even) or the stored time (manual). The loop's
    /// last→first wrap time is not a key, so it is not included (the view draws it as a ghost at
    /// the duration position).
    var keyframeTimes: [Double] {
        let n = path.keyframes.count
        guard n > 0 else { return [] }
        if path.pacing == .manual {
            if manualNodes.isEmpty {   // 1 key — clamp the stored time
                let dur = max(path.duration, 0)
                return [min(max(path.keyframes[0].time, 0), dur)]
            }
            return manualNodes.prefix(n).map { $0.u * path.duration }
        }
        guard !segments.isEmpty else { return [0] }   // a 1-key still shot
        var times: [Double] = []
        times.reserveCapacity(n)
        for i in 0..<n {
            let r: Double
            if totalLength <= Self.zeroLengthEpsilon {
                // A fully stationary path — the same axis as the uniform parameter fallback
                r = i < segments.count ? Double(i) / Double(segments.count) : 1
            } else {
                r = i < segmentStart.count ? segmentStart[i] / totalLength : 1
            }
            let s = path.ease == .easeInOut ? Self.quinticInverse(r) : r
            times.append(s * path.duration)
        }
        return times
    }

    // MARK: The manual pacing inversion (piecewise-linear time→arc length — respecting the key
    // times)

    /// t (seconds) → (segment, τ). It linearly interpolates D within the node interval
    /// [u_i, u_{i+1}] and then inverts the distance.
    /// Before the first node and after the last node it holds. A fully stationary path
    /// (arc length ≈ 0) falls back to a uniform parameter through the node i ↔ segment i
    /// correspondence (the node count − 1 == segmentCount).
    private func manualParameter(at t: Double) -> (Int, Double) {
        let u = path.duration > 0 ? min(max(t / path.duration, 0), 1) : 0
        let nodes = manualNodes
        let degenerate = totalLength <= Self.zeroLengthEpsilon
        if u <= nodes[0].u {
            return degenerate ? (0, 0) : parameter(atDistance: nodes[0].d)
        }
        guard let lastU = nodes.last?.u, u < lastU else {
            return degenerate ? (segments.count - 1, 1)
                              : parameter(atDistance: nodes[nodes.count - 1].d)
        }
        // The last i with nodes[i].u ≤ u < nodes[i+1].u (n is on the order of tens — a linear scan
        // is sufficient)
        var i = 0
        for j in 1..<nodes.count where nodes[j].u <= u { i = j }
        let du = nodes[i + 1].u - nodes[i].u
        let frac = du > 0 ? (u - nodes[i].u) / du : 0
        if degenerate {
            let seg = min(i, segments.count - 1)
            return (seg, min(frac, 1))
        }
        let d = nodes[i].d + frac * (nodes[i + 1].d - nodes[i].d)
        return parameter(atDistance: d)
    }

    // MARK: The constant-speed inversion

    /// The normalized progress s∈[0,1] → (the segment index, τ∈[0,1] within the segment).
    /// Constant speed: it inverts the target distance d = s·L by a binary search over the segments,
    /// then a binary search over the cumulative table within the segment.
    private func parameter(atNormalized s: Double) -> (Int, Double) {
        if totalLength <= Self.zeroLengthEpsilon {
            // A fully stationary path — the uniform parameter fallback (rotation and FOV
            // interpolate uniformly in time)
            let u = s * Double(segments.count)
            let i = min(Int(u), segments.count - 1)
            return (i, min(u - Double(i), 1))
        }
        return parameter(atDistance: s * totalLength)
    }

    /// The target distance d (0…totalLength) → (segment, τ) — the distance inversion shared by the
    /// constant-speed and manual paths.
    private func parameter(atDistance d: Double) -> (Int, Double) {
        // The last i with segmentStart[i] ≤ d (a lower-bound variant)
        var lo = 0, hi = segments.count - 1
        while lo < hi {
            let mid = (lo + hi + 1) / 2
            if segmentStart[mid] <= d { lo = mid } else { hi = mid - 1 }
        }
        let seg = segments[lo]
        let local = min(max(d - segmentStart[lo], 0), seg.length)
        var a = 0, b = Self.samplesPerSegment
        while b - a > 1 {
            let m = (a + b) / 2
            if seg.cumulative[m] <= local { a = m } else { b = m }
        }
        let c0 = seg.cumulative[a], c1 = seg.cumulative[b]
        let frac = c1 > c0 ? (local - c0) / (c1 - c0) : 0   // deterministic handling of a zero-length span (duplicate keys)
        return (lo, (Double(a) + frac) / Double(Self.samplesPerSegment))
    }

    // MARK: Math primitives

    /// The centripetal knot spacing: |ΔP|^α (α=0.5) — with an ε floor for identical positions.
    private static func knotDelta(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> Double {
        max(simd_distance(a, b).squareRoot(), knotEpsilon)
    }

    /// The Barry–Goldman pyramid evaluation. τ∈[0,1] corresponds to the [t1,t2] interval.
    /// Every denominator is guaranteed positive by knotDelta's ε floor.
    private static func point(_ s: Segment, _ tau: Double) -> SIMD3<Double> {
        let t = s.t1 + tau * (s.t2 - s.t1)
        func lerp(_ a: SIMD3<Double>, _ b: SIMD3<Double>, _ w: Double) -> SIMD3<Double> {
            a + (b - a) * w
        }
        let a1 = lerp(s.p0, s.p1, (t - s.t0) / (s.t1 - s.t0))
        let a2 = lerp(s.p1, s.p2, (t - s.t1) / (s.t2 - s.t1))
        let a3 = lerp(s.p2, s.p3, (t - s.t2) / (s.t3 - s.t2))
        let b1 = lerp(a1, a2, (t - s.t0) / (s.t2 - s.t0))
        let b2 = lerp(a2, a3, (t - s.t1) / (s.t3 - s.t1))
        return lerp(b1, b2, (t - s.t1) / (s.t2 - s.t1))
    }

    /// Hemisphere-normalized slerp — a sign flip when dot<0 (the shortest path; required by the
    /// spec).
    /// A near-identical pair (dot>0.9995) degrades to nlerp (numerical stability as sinθ→0 — the
    /// difference in the result is at the floating-point level).
    static func slerpShortest(_ q0: simd_quatf, _ q1: simd_quatf, _ t: Float) -> simd_quatf {
        var b = q1.vector
        var d = simd_dot(q0.vector, q1.vector)
        if d < 0 { b = -b; d = -d }
        if d > 0.9995 {
            let v = q0.vector + (b - q0.vector) * t
            let len = simd_length(v)
            return len > 1e-12 ? simd_quatf(vector: v / len) : q0
        }
        let theta = acos(min(d, 1))
        let s = sin(theta)
        return simd_quatf(vector: (q0.vector * sin((1 - t) * theta) + b * sin(t * theta)) / s)
    }

    /// The quintic ease: 6t⁵−15t⁴+10t³ (the value and the first and second derivatives are
    /// continuous at the boundaries).
    static func quintic(_ s: Double) -> Double {
        s * s * s * (s * (6 * s - 15) + 10)
    }

    /// The inverse of quintic — it is strictly increasing on [0,1], so bisection (60 iterations →
    /// error < 2⁻⁶⁰).
    /// Used for the time-axis inversion in keyframeTimes (constant speed + ease) (spec
    /// footnote 1).
    static func quinticInverse(_ y: Double) -> Double {
        if y <= 0 { return 0 }
        if y >= 1 { return 1 }
        var lo = 0.0, hi = 1.0
        for _ in 0..<60 {
            let mid = (lo + hi) / 2
            if quintic(mid) < y { lo = mid } else { hi = mid }
        }
        return (lo + hi) / 2
    }
}
