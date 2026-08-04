import XCTest
import Metal
import simd
import MetalSplatter
import SplatXMetal
@testable import SplatXMetalTrainer

/// The pure logic of LiveSplatStreamer (app round 3 Task 2, spec §3) — the keep-only-the-
/// latest pending policy, the staging round trip, the shN interleave identity, and consistency
/// of the encoding on a real device. The streamer proper (the renderer swap) is the
/// responsibility of manual E2E (the no-tests-for-SwiftUI/Metal-views policy).
final class LiveSplatStreamerTests: XCTestCase {

    // MARK: Synthetic snapshot helpers

    /// Synthetic raw arrays matching count and degree. The values are filled from seed so
    /// they stay distinguishable.
    private struct Arrays {
        var means: [Float], quats: [Float], scales: [Float]
        var opacities: [Float], sh0: [Float], shN: [Float]?

        init(count: Int, shDegree: Int, seed: Float) {
            means = Self.ramp(count * 3) { (i: Int) -> Float in seed + Float(i) * 0.5 }
            quats = Self.ramp(count * 4) { (i: Int) -> Float in seed + 1 + Float(i) * 0.25 }
            scales = Self.ramp(count * 3) { (i: Int) -> Float in -2 + seed * 0.01 + Float(i) * 0.125 }
            opacities = Self.ramp(count) { (i: Int) -> Float in seed * 0.1 + Float(i) - 1 }
            sh0 = Self.ramp(count * 3) { (i: Int) -> Float in seed - Float(i) * 0.75 }
            let k = (shDegree + 1) * (shDegree + 1) - 1
            shN = shDegree > 0
                ? Self.ramp(count * k * 3) { (i: Int) -> Float in seed + Float(i) * 0.0625 }
                : nil
        }

        static func ramp(_ n: Int, _ f: (Int) -> Float) -> [Float] {
            var out: [Float] = []
            out.reserveCapacity(n)
            for i in 0..<n { out.append(f(i)) }
            return out
        }
    }

    /// Builds a pointer-view SplatSnapshot over the arrays and passes it in valid only for
    /// the duration of body — this reproduces the "invalid after the handler returns" contract
    /// (the memberwise public init exists for exactly this).
    private func withSnapshot(iter: Int, shDegree: Int, count: Int, _ a: Arrays,
                              _ body: (SplatXTrainer.SplatSnapshot) -> Void) {
        a.means.withUnsafeBufferPointer { (m: UnsafeBufferPointer<Float>) in
            a.quats.withUnsafeBufferPointer { (q: UnsafeBufferPointer<Float>) in
                a.scales.withUnsafeBufferPointer { (sc: UnsafeBufferPointer<Float>) in
                    a.opacities.withUnsafeBufferPointer { (op: UnsafeBufferPointer<Float>) in
                        a.sh0.withUnsafeBufferPointer { (c0: UnsafeBufferPointer<Float>) in
                            if let shN = a.shN {
                                shN.withUnsafeBufferPointer { (rest: UnsafeBufferPointer<Float>) in
                                    let snap = SplatXTrainer.SplatSnapshot(
                                        iter: iter, shDegree: shDegree, count: count,
                                        means: m, quats: q, scales: sc,
                                        opacities: op, sh0: c0, shN: rest)
                                    body(snap)
                                }
                            } else {
                                let snap = SplatXTrainer.SplatSnapshot(
                                    iter: iter, shDegree: shDegree, count: count,
                                    means: m, quats: q, scales: sc,
                                    opacities: op, sh0: c0, shN: nil)
                                body(snap)
                            }
                        }
                    }
                }
            }
        }
    }

    private func publish(_ mailbox: SnapshotMailbox, iter: Int, shDegree: Int = 0,
                         count: Int = 2, seed: Float = 1) {
        withSnapshot(iter: iter, shDegree: shDegree, count: count,
                     Arrays(count: count, shDegree: shDegree, seed: seed)) {
            mailbox.publish($0)
        }
    }

    // MARK: The keep-only-the-latest pending policy

    /// Two unconsumed publishes → take returns only the latest and the earlier one is
    /// discarded. A second take is nil.
    func testMailboxKeepsOnlyLatestPending() {
        let mb = SnapshotMailbox()
        publish(mb, iter: 50)
        publish(mb, iter: 100)
        let taken = mb.take()
        XCTAssertEqual(taken?.iter, 100)
        XCTAssertNil(mb.take())
        taken.map { mb.recycle($0) }
    }

    /// Even while the consumer holds one set (checked out), publishing, overwriting, and
    /// cycling continue — this verifies the two-set invariant (no starvation).
    func testMailboxCyclesWhileBufferCheckedOut() {
        let mb = SnapshotMailbox()
        publish(mb, iter: 50)
        let first = mb.take()
        XCTAssertEqual(first?.iter, 50)          // held
        publish(mb, iter: 100)
        publish(mb, iter: 150)                   // overwrites the unconsumed one
        let second = mb.take()
        XCTAssertEqual(second?.iter, 150)
        first.map { mb.recycle($0) }
        second.map { mb.recycle($0) }
        publish(mb, iter: 200)                   // reused after being returned
        let third = mb.take()
        XCTAssertEqual(third?.iter, 200)
        third.map { mb.recycle($0) }
    }

    // MARK: The staging round trip

    /// The publish→take round trip: the metadata (iter/shDegree/count) and all six arrays are
    /// element-wise identical to the originals.
    /// A nil shN (degree 0) is left as an empty array.
    func testStagingRoundtripCopiesAllFields() {
        let mb = SnapshotMailbox()
        let a = Arrays(count: 3, shDegree: 2, seed: 7)
        withSnapshot(iter: 250, shDegree: 2, count: 3, a) { mb.publish($0) }
        let st = mb.take()
        XCTAssertNotNil(st)
        guard let st else { return }
        XCTAssertEqual(st.iter, 250)
        XCTAssertEqual(st.shDegree, 2)
        XCTAssertEqual(st.count, 3)
        XCTAssertEqual(st.means, a.means)
        XCTAssertEqual(st.quats, a.quats)
        XCTAssertEqual(st.scales, a.scales)
        XCTAssertEqual(st.opacities, a.opacities)
        XCTAssertEqual(st.sh0, a.sh0)
        XCTAssertEqual(st.shN, a.shN)
        mb.recycle(st)

        publish(mb, iter: 300, shDegree: 0)      // degree 0 → shN is an empty array (clears the reused buffer's residue)
        let st0 = mb.take()
        XCTAssertEqual(st0?.shDegree, 0)
        XCTAssertEqual(st0?.shN, [])
        st0.map { mb.recycle($0) }
    }

    // MARK: The shN interleave identity

    /// The [N,K,3] coefficient-major raw shN lands at MetalSplatter's SH buffer offset
    /// `i·K·3 + k·3 + c` as-is (only converted to f16) — this verifies the index identity behind
    /// the claim that no transpose is needed (spec §3).
    func testShNInterleaveIdentityMapping() {
        let count = 2, degree = 2
        let k = (degree + 1) * (degree + 1) - 1   // 8
        var a = Arrays(count: count, shDegree: degree, seed: 0)
        // A value that identifies (i,k,c): i*1000 + k*10 + c
        var shN: [Float] = []
        for i in 0..<count {
            for kk in 0..<k {
                for c in 0..<3 { shN.append(Float(i * 1000 + kk * 10 + c)) }
            }
        }
        a.shN = shN
        let mb = SnapshotMailbox()
        withSnapshot(iter: 1, shDegree: degree, count: count, a) { mb.publish($0) }
        guard let st = mb.take() else { return XCTFail("no staging") }

        let total = LiveSplatEncoder.shElementCount(count: count, shDegree: degree)
        XCTAssertEqual(total, count * k * 3)
        let out = UnsafeMutablePointer<Float16>.allocate(capacity: total)
        defer { out.deallocate() }
        XCTAssertEqual(LiveSplatEncoder.encodeSH(st, into: out), total)
        for i in 0..<count {
            for kk in 0..<k {
                for c in 0..<3 {
                    XCTAssertEqual(out[i * k * 3 + kk * 3 + c],
                                   Float16(Float(i * 1000 + kk * 10 + c)),
                                   "mismatch at splat \(i) coeff \(kk) channel \(c)")
                }
            }
        }
        mb.recycle(st)
    }

    // MARK: Consistency of the encoding on a real device

    /// The direct-write path into MetalBuffer: a 32B stride and a consistent count, plus
    /// byte-identity with a reference EncodedSplatPoint built from the raw values after the
    /// activations (sigmoid/exp/normalize) are applied (memcmp is valid because the 32B layout
    /// has no padding). The SH buffer values match the f16 conversion as well.
    func testEncodeIntoMetalBuffersMatchesReferenceInit() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let count = 2, degree = 1
        let a = Arrays(count: count, shDegree: degree, seed: 3)
        let mb = SnapshotMailbox()
        withSnapshot(iter: 50, shDegree: degree, count: count, a) { mb.publish($0) }
        let st = try XCTUnwrap(mb.take())

        XCTAssertEqual(MemoryLayout<EncodedSplatPoint>.stride, 32)

        let splats = try MetalBuffer<EncodedSplatPoint>(device: device, capacity: count)
        LiveSplatEncoder.encodeSplats(st, into: splats.values)
        splats.count = count

        for i in 0..<count {
            let quat = simd_quatf(real: a.quats[4 * i],
                                  imag: SIMD3(a.quats[4 * i + 1], a.quats[4 * i + 2], a.quats[4 * i + 3]))
            var expected = EncodedSplatPoint(
                position: SIMD3(a.means[3 * i], a.means[3 * i + 1], a.means[3 * i + 2]),
                colorSH0: SIMD3(a.sh0[3 * i], a.sh0[3 * i + 1], a.sh0[3 * i + 2]),
                opacity: 1 / (1 + exp(-a.opacities[i])),
                scale: SIMD3(exp(a.scales[3 * i]), exp(a.scales[3 * i + 1]), exp(a.scales[3 * i + 2])),
                rotation: quat.normalized)
            XCTAssertEqual(memcmp(&expected, splats.values + i, MemoryLayout<EncodedSplatPoint>.stride), 0,
                           "splat \(i) bytes differ from reference init")
        }

        let total = LiveSplatEncoder.shElementCount(count: count, shDegree: degree)   // 2*3*3 = 18
        let sh = try MetalBuffer<Float16>(device: device, capacity: total)
        XCTAssertEqual(LiveSplatEncoder.encodeSH(st, into: sh.values), total)
        sh.count = total
        for j in 0..<total {
            XCTAssertEqual(sh.values[j], Float16(a.shN![j]))
        }
        mb.recycle(st)
    }
}
