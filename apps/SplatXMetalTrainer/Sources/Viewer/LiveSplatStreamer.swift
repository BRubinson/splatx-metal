import Foundation
import Metal
import simd
import MetalSplatter
import SplatIO
import SplatXMetal

// App round 3 Task 2 (spec §3): the training thread's snapshot callback → an in-memory,
// glitch-free live render update.
//
// The pipeline has three stages:
//   ingest (on the training thread, synchronous and brief) → SnapshotMailbox (one latest pending)
//   → runPipeline (a background serial task) encoding (writing directly into a MetalBuffer) →
//   the glitch-free swap (addChunk disabled → afterNextSort → enable new + remove old).
//
// The measured MetalSplatter 1.0.1 contract (research/MetalSplatter — confirmed no source diff
// against the 1.0.1 tag):
//   `EncodedSplatPoint.init(position:colorSH0:opacity:scale:rotation:)` takes the **linear
//   domain** — opacity is "Linear opacity (0-1)", scale is "Linear scale", and rotation is assumed
//   normalized (the init that goes through SplatPoint applies `.asLinearFloat`/`.normalized`
//   beforehand and passes them on).
//   So the snapshot's raw domains (logit opacity, log scale, and an unnormalized quat) have
//   sigmoid/exp/normalize applied **here** before being passed on. colorSH0 is the raw sh0
//   coefficient as-is ("NOT sRGB color").
//   The coefficient-first RGB interleave of shN [N,K,3] is 1:1 with the MetalSplatter SH buffer
//   layout (SplatChunk.swift:23-27, the higherOrderSHCoefficients flattening) — only the f16
//   conversion is performed.

// MARK: - Staging (device-independent — the pure logic test surface)

/// A reusable copy buffer sized for one snapshot. It memcpys the whole SplatSnapshot — whose
/// pointer lifetime is limited to the callback — on the training thread. Ownership is guaranteed
/// to be single-threaded by SnapshotMailbox's state transitions
/// (free → pending → checked-out → free).
final class SplatSnapshotStaging: @unchecked Sendable {
    private(set) var iter = 0
    private(set) var shDegree = 0
    private(set) var count = 0
    private(set) var means: [Float] = []      // [N,3]
    private(set) var quats: [Float] = []      // [N,4] w-first, raw and unnormalized
    private(set) var scales: [Float] = []     // [N,3] raw log-space
    private(set) var opacities: [Float] = []  // [N]   raw logit-space
    private(set) var sh0: [Float] = []        // [N,3] raw coefficients
    private(set) var shN: [Float] = []        // [N,K,3] coefficient-first (an empty array at degree 0)

    func copy(from s: SplatXTrainer.SplatSnapshot) {
        iter = s.iter
        shDegree = s.shDegree
        count = s.count
        Self.replace(&means, with: s.means)
        Self.replace(&quats, with: s.quats)
        Self.replace(&scales, with: s.scales)
        Self.replace(&opacities, with: s.opacities)
        Self.replace(&sh0, with: s.sh0)
        if let rest = s.shN {
            Self.replace(&shN, with: rest)
        } else {
            shN.removeAll(keepingCapacity: true)
        }
    }

    /// keepingCapacity reuse — it fills through the contiguous memcpy path without reallocating on
    /// every snapshot.
    private static func replace(_ dst: inout [Float], with src: UnsafeBufferPointer<Float>) {
        dst.removeAll(keepingCapacity: true)
        dst.append(contentsOf: src)
    }
}

/// The mailbox implementing the "one latest pending" policy. It assumes a single producer (the
/// training thread) and a single consumer (the encoding pipeline) — publish's "grab → copy outside
/// the lock → republish" is safe only on that premise.
/// Two staging buffers never run out: consumer checked-out ≤ 1 + producer grab ≤ 1 = 2.
final class SnapshotMailbox: @unchecked Sendable {
    private let lock = NSLock()
    private var free: [SplatSnapshotStaging]
    private var pending: SplatSnapshotStaging?

    init(bufferCount: Int = 2) {
        free = (0..<max(bufferCount, 2)).map { _ in SplatSnapshotStaging() }
    }

    /// Training thread: if there is an unconsumed pending, it overwrites that buffer (= discarding
    /// the previously unprocessed one).
    /// The bulk copy happens outside the lock — take() is not blocked for the duration of the copy
    /// (while it is running, the consumer sees pending == nil and sleeps, and the wake after
    /// publish rouses it again).
    func publish(_ snap: SplatXTrainer.SplatSnapshot) {
        lock.lock()
        let buf = pending ?? free.popLast()
        pending = nil
        lock.unlock()
        guard let buf else { return }   // unreachable by the invariant (2 buffers, single consumer) — a defensive drop
        buf.copy(from: snap)
        lock.lock()
        pending = buf
        lock.unlock()
    }

    /// Consumer: takes the pending buffer (checked-out). Return it with recycle when finished.
    func take() -> SplatSnapshotStaging? {
        lock.lock()
        defer { lock.unlock() }
        let p = pending
        pending = nil
        return p
    }

    func recycle(_ buf: SplatSnapshotStaging) {
        lock.lock()
        free.append(buf)
        lock.unlock()
    }

    /// Discards the pending buffer on stop (the buffer returns to the pool).
    func clear() {
        lock.lock()
        if let p = pending { free.append(p); pending = nil }
        lock.unlock()
    }
}

// MARK: - Encoder (pointer-targeted — testable without a MetalBuffer)

enum LiveSplatEncoder {
    static func sigmoid(_ x: Float) -> Float { 1 / (1 + exp(-x)) }

    /// Raw-domain staging → a contiguous write of EncodedSplatPoint. The activations
    /// (sigmoid/exp/normalize) are applied here and passed to the public init — never assemble the
    /// fields directly (spec §9: consistency with the 32B layout is secured by using only the
    /// public init).
    static func encodeSplats(_ st: SplatSnapshotStaging,
                             into out: UnsafeMutablePointer<EncodedSplatPoint>) {
        let n = st.count
        st.means.withUnsafeBufferPointer { m in
            st.quats.withUnsafeBufferPointer { q in
                st.scales.withUnsafeBufferPointer { sc in
                    st.opacities.withUnsafeBufferPointer { op in
                        st.sh0.withUnsafeBufferPointer { c0 in
                            for i in 0..<n {
                                let raw = simd_quatf(real: q[4 * i],
                                                     imag: SIMD3(q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]))
                                // Zero-norm guard: if normalize would produce NaN, substitute the
                                // identity rotation.
                                let rot = simd_length(raw.vector) > 1e-12
                                    ? raw.normalized
                                    : simd_quatf(real: 1, imag: .zero)
                                out[i] = EncodedSplatPoint(
                                    position: SIMD3(m[3 * i], m[3 * i + 1], m[3 * i + 2]),
                                    colorSH0: SIMD3(c0[3 * i], c0[3 * i + 1], c0[3 * i + 2]),
                                    opacity: sigmoid(op[i]),
                                    scale: SIMD3(exp(sc[3 * i]), exp(sc[3 * i + 1]), exp(sc[3 * i + 2])),
                                    rotation: rot)
                            }
                        }
                    }
                }
            }
        }
    }

    /// The SH buffer element count = N × K × 3 (K = (d+1)² − 1).
    static func shElementCount(count: Int, shDegree: Int) -> Int {
        let k = (shDegree + 1) * (shDegree + 1) - 1
        return count * k * 3
    }

    /// Coefficient-first shN [N,K,3] → the MetalSplatter SH interleave, 1:1 (identity indexing,
    /// only the f16 conversion).
    /// The return value is the number of elements written (0 when the staging size does not match
    /// — a guard; the core tests verify the convention).
    @discardableResult
    static func encodeSH(_ st: SplatSnapshotStaging,
                         into out: UnsafeMutablePointer<Float16>) -> Int {
        let total = shElementCount(count: st.count, shDegree: st.shDegree)
        guard total > 0, st.shN.count == total else { return 0 }
        st.shN.withUnsafeBufferPointer { src in
            for j in 0..<total { out[j] = Float16(src[j]) }
        }
        return total
    }
}

// MARK: - Streamer

/// The streamer that flows training snapshots into the MetalSplatter renderer without glitches.
/// - ingest: called synchronously on the training thread (the snapshot contract — the pointers are
///   invalid after the callback returns, so it copies immediately).
/// - The pipeline: background and serial — it picks up the next snapshot only after the encoding
///   and swap have run to completion. The two-way MetalBuffer alternation depends on this
///   seriality (only a set whose previous swap's removeChunk has finished is rewritten).
/// - stop(): discards the pending snapshot and ends the pipeline cooperatively. Afterwards you can
///   wait for completion with waitUntilIdle() (on completion it blocks the chunk contention
///   between showRun's final output load and a live swap — the same purpose as the round 2
///   liveReloadTask await).
final class LiveSplatStreamer: @unchecked Sendable {
    /// The observed state for the LIVE badge and statistics — updated **at swap completion (when
    /// it reaches the screen)**.
    /// (It publishes the iter being displayed rather than the one at ingest time — so the badge
    /// matches the screen.)
    @MainActor @Observable final class State {
        fileprivate(set) var latestIter = 0
        fileprivate(set) var splatCount = 0
    }

    let state: State

    private let device: MTLDevice
    /// Acquires the renderer on the MainActor when the first chunk is ready (lazy host creation is
    /// the owner's responsibility).
    /// Returning nil = skip this snapshot and retry on the next one.
    private let acquireRenderer: @MainActor @Sendable () -> SplatRenderer?
    /// A MainActor notification on every swap completion (the displayed splat count) — for wiring
    /// ViewerModel.currentSplats.
    private let onSwap: (@MainActor @Sendable (Int) -> Void)?

    private let mailbox = SnapshotMailbox()
    private let stopLock = NSLock()
    private var stopRequested = false
    private let wake: AsyncStream<Void>.Continuation
    private var pipelineTask: Task<Void, Never>?

    @MainActor
    init(device: MTLDevice,
         acquireRenderer: @escaping @MainActor @Sendable () -> SplatRenderer?,
         onSwap: (@MainActor @Sendable (Int) -> Void)? = nil) {
        self.device = device
        self.acquireRenderer = acquireRenderer
        self.onSwap = onSwap
        self.state = State()
        var continuation: AsyncStream<Void>.Continuation!
        let wakeups = AsyncStream<Void>(bufferingPolicy: .bufferingNewest(1)) { continuation = $0 }
        self.wake = continuation
        pipelineTask = Task.detached(priority: .utility) { [self] in
            await runPipeline(wakeups: wakeups)
        }
    }

    private var isStopped: Bool {
        stopLock.lock()
        defer { stopLock.unlock() }
        return stopRequested
    }

    /// Called on the training loop thread (Options.snapshotHandler). It only copies and returns
    /// immediately — if encoding falls behind, the "one latest pending" policy overwrites the
    /// previously unprocessed one (dropping is allowed, spec §9).
    func ingest(_ snap: SplatXTrainer.SplatSnapshot) {
        guard !isStopped, snap.count > 0 else { return }
        mailbox.publish(snap)
        wake.yield()
    }

    /// Discards the pending snapshot and ends the pipeline cooperatively (non-blocking). An
    /// in-flight cycle folds at its boundary checks.
    func stop() {
        stopLock.lock()
        stopRequested = true
        stopLock.unlock()
        wake.finish()
        mailbox.clear()
    }

    /// Waits for the pipeline to run to completion — call it after stop() and before loading the
    /// final output to block chunk contention.
    func waitUntilIdle() async {
        await pipelineTask?.value
    }

    // MARK: The pipeline (background, serial)

    /// A MetalBuffer set sized for one chunk. Two sets alternate — set N is rewritten only after
    /// the chunk that referenced it has been taken out of the renderer by removeChunk (guaranteed
    /// by the serial pipeline).
    private final class ChunkBuffers {
        let splats: MetalBuffer<EncodedSplatPoint>
        let sh: MetalBuffer<Float16>
        init(device: MTLDevice) throws {
            splats = try MetalBuffer(device: device)
            sh = try MetalBuffer(device: device)
        }
    }

    private func runPipeline(wakeups: AsyncStream<Void>) async {
        var renderer: SplatRenderer?
        var sets: [ChunkBuffers] = []
        var setIndex = 0
        var previousChunk: ChunkID?

        for await _ in wakeups {
            while let staging = mailbox.take() {
                if isStopped { mailbox.recycle(staging); return }

                // ① Acquire the renderer (once, the first time). It empties the leftover chunks of
                //    the previous run to establish the baseline state of the live chain
                //    (equivalent to the removeAllChunks in round 2's host.load).
                if renderer == nil {
                    renderer = await acquireRenderer()
                    if let r = renderer { await r.removeAllChunks() }
                }
                guard let renderer else { mailbox.recycle(staging); continue }

                // ② Encoding — the staging buffer is returned right after encoding (the next
                //    ingest reuses it).
                let iter = staging.iter
                let n = staging.count
                let chunk: SplatChunk?
                do {
                    if sets.isEmpty {
                        sets = [try ChunkBuffers(device: device), try ChunkBuffers(device: device)]
                    }
                    let set = sets[setIndex]
                    setIndex = (setIndex + 1) % sets.count
                    chunk = try Self.encode(staging, into: set)
                } catch {
                    chunk = nil   // buffer allocation failure and the like — skip this snapshot (live is non-fatal)
                }
                mailbox.recycle(staging)
                guard let chunk else { continue }

                // ③ The glitch-free swap (the pattern MetalSplatter documents): add it disabled →
                //    wait for one full sort that includes the new chunk → atomically enable new
                //    and remove old.
                let newID = await renderer.addChunk(chunk, sortByLocality: false, enabled: false)
                await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
                    renderer.afterNextSort { cont.resume() }   // a callback on the sort background thread
                }
                if isStopped {
                    // Termination race guard: enabling the live chunk after the final output load
                    // (removeAllChunks) would draw them overlapping — instead of enabling, remove
                    // the new chunk and fold.
                    await renderer.removeChunk(newID)
                    return
                }
                let old = previousChunk
                await renderer.withChunkAccess {
                    await renderer.setChunkEnabled(newID, enabled: true)
                    if let old { await renderer.removeChunk(old) }
                }
                previousChunk = newID

                let state = self.state
                let onSwap = self.onSwap
                await MainActor.run {
                    state.latestIter = iter
                    state.splatCount = n
                    onSwap?(n)
                }
            }
        }
    }

    private static func encode(_ staging: SplatSnapshotStaging, into set: ChunkBuffers) throws -> SplatChunk {
        let n = staging.count
        try set.splats.ensureCapacity(n)
        LiveSplatEncoder.encodeSplats(staging, into: set.splats.values)
        set.splats.count = n

        var sh: MetalBuffer<Float16>? = nil
        var degree = SHDegree.sh0
        if staging.shDegree > 0 {
            let total = LiveSplatEncoder.shElementCount(count: n, shDegree: staging.shDegree)
            try set.sh.ensureCapacity(total)
            if LiveSplatEncoder.encodeSH(staging, into: set.sh.values) == total {
                set.sh.count = total
                sh = set.sh
                degree = SHDegree(rawValue: UInt8(staging.shDegree)) ?? .sh0
            }
        }
        return SplatChunk(splats: set.splats, shCoefficients: sh, shDegree: degree)
    }
}
