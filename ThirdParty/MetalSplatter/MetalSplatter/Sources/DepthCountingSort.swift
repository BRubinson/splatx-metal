import Foundation

/// splatx-metal fork (viewer-sort Task 2, rewritten in viewer-sort FIX 1): exact
/// back-to-front depth ordering via a 2-pass LSD radix sort over a monotone float key,
/// replacing the upstream comparison sort (`Array.sort`, O(N log N), single-threaded)
/// in phase 2 of `SplatSorter.performSort`.
///
/// History: Task 2 shipped a 16-bit *quantized* counting sort (65,536 buckets linearly
/// spanning the finite [min, max] of the input). That collapses on fully-trained PLYs:
/// MCMC far outliers blow up the depth range (depth = camera distance *squared*; measured
/// garden 7k: d²max ≈ 1.9e8 vs a ~3.6e3 scene core), the bucket width grows to ~3e3 (d²),
/// and 99.8% of the scene lands in a single bucket — the "sort" degenerates to input
/// order and occlusion breaks. FIX 1 therefore sorts *exactly*, with no quantization.
///
/// Key mapping (IEEE 754 total order, monotone in float comparison):
///     bits = depth.bitPattern
///     key  = (bits & 0x8000_0000) != 0 ? ~bits : (bits | 0x8000_0000)
/// Negative floats are bit-inverted (reversing their order), positives are offset into
/// the upper half. Every input bit pattern gets a deterministic position: +inf sorts
/// beyond every finite depth (drawn first / farthest), -inf below every finite depth
/// (drawn last / nearest), and NaNs sort by their raw bit pattern — positive-sign NaNs
/// beyond +inf (farthest), negative-sign NaNs below -inf (nearest). -0.0 orders just
/// below +0.0 (distinct bit patterns).
///
/// Sort: LSD radix over the 32-bit key, 16-bit digits, 2 passes (low 16 then high 16),
/// each a stable counting scatter over 65,536 buckets emitted highest-bucket-first, so
/// both passes emit descending and the final order is fully descending (back-to-front,
/// matching the upstream `sort { $0.depth > $1.depth }` direction). Stability makes the
/// output bitwise deterministic: equal keys (identical bit patterns) keep input order,
/// and the result position-matches a descending comparison sort at every index.
///
/// O(3N + 2·65,536); all buffers (key array, ping-pong index arrays, two histogram
/// arrays) are reused across calls — no per-sort allocation after warm-up.
final class DepthCountingSorter {
    /// Buckets per radix digit (16-bit digits).
    static let bucketCount = 1 << 16

    /// Per-bucket counts, then running write offsets, for the low/high digit pass.
    /// Reused across calls.
    private var lowOffsets = [UInt32](repeating: 0, count: DepthCountingSorter.bucketCount)
    private var highOffsets = [UInt32](repeating: 0, count: DepthCountingSorter.bucketCount)

    /// Monotone sort keys, indexed by input element index. Reused across calls.
    private var keys: [UInt32] = []

    /// Intermediate permutation after the low-digit pass (ping buffer). Reused across calls.
    private var tempIndices: [UInt32] = []

    /// Output permutation: `sortedIndices[k]` is the index into the input `depths` of the
    /// k-th element in back-to-front order. Only `0..<count` of the most recent `sort`
    /// call is meaningful (storage is reused and may be larger than the last count).
    private(set) var sortedIndices: [UInt32] = []

    /// Fills `sortedIndices[0..<count]` with the back-to-front (descending-depth) order
    /// of `depths[0..<count]`. Exact: position-for-position identical to a descending
    /// comparison sort (ties = identical bit patterns, resolved stably by input order).
    func sort(depths: [Float], count: Int) {
        precondition(count >= 0 && count <= depths.count, "count out of range")
        precondition(count <= Int(UInt32.max), "count exceeds UInt32 index space")
        guard count > 0 else { return }

        if sortedIndices.count < count {
            sortedIndices = [UInt32](repeating: 0, count: count)
            tempIndices = [UInt32](repeating: 0, count: count)
            keys = [UInt32](repeating: 0, count: count)
        }

        depths.withUnsafeBufferPointer { depthsPtr in
            keys.withUnsafeMutableBufferPointer { keysPtr in
                lowOffsets.withUnsafeMutableBufferPointer { low in
                    highOffsets.withUnsafeMutableBufferPointer { high in
                        low.update(repeating: 0)
                        high.update(repeating: 0)

                        // Single pass: map to monotone keys and build both digit
                        // histograms.
                        for i in 0..<count {
                            let bits = depthsPtr[i].bitPattern
                            let key = (bits & 0x8000_0000) != 0 ? ~bits : (bits | 0x8000_0000)
                            keysPtr[i] = key
                            low[Int(key & 0xFFFF)] &+= 1
                            high[Int(key >> 16)] &+= 1
                        }

                        // Exclusive prefix sums in emission order: highest bucket first
                        // -> both passes emit descending keys (back-to-front).
                        var runningLow: UInt32 = 0
                        var runningHigh: UInt32 = 0
                        var bucket = Self.bucketCount - 1
                        while bucket >= 0 {
                            let lowSize = low[bucket]
                            low[bucket] = runningLow
                            runningLow &+= lowSize
                            let highSize = high[bucket]
                            high[bucket] = runningHigh
                            runningHigh &+= highSize
                            bucket -= 1
                        }

                        tempIndices.withUnsafeMutableBufferPointer { temp in
                            // Pass 1: stable scatter by low 16 bits (descending).
                            for i in 0..<count {
                                let bucket = Int(keysPtr[i] & 0xFFFF)
                                let position = low[bucket]
                                low[bucket] = position &+ 1
                                temp[Int(position)] = UInt32(i)
                            }

                            // Pass 2: stable scatter by high 16 bits (descending).
                            // Stability preserves the pass-1 low-digit order within
                            // each high-digit bucket -> fully descending output.
                            sortedIndices.withUnsafeMutableBufferPointer { output in
                                for k in 0..<count {
                                    let index = temp[k]
                                    let bucket = Int(keysPtr[Int(index)] >> 16)
                                    let position = high[bucket]
                                    high[bucket] = position &+ 1
                                    output[Int(position)] = index
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
