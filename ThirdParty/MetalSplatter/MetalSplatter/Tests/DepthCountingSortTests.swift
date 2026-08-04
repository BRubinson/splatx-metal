import XCTest
@testable import MetalSplatter

/// splatx-metal fork (viewer-sort Task 2, revised in viewer-sort FIX 1): tests for the
/// exact float-key 2-pass LSD radix sort that replaced the 16-bit quantized counting sort.
///
/// Contract under test: the output permutation visits depths in *exactly* descending
/// (back-to-front) order, position-for-position identical to the reference comparison
/// sort `depths.sorted(by: >)` — ties are identical bit patterns, so comparing the depth
/// sequence is sufficient. Non-finite depths follow the monotone bit-pattern mapping
/// deterministically: positive NaN > +inf > finite > -inf > negative NaN (in draw order:
/// farthest first).
final class DepthCountingSortTests: XCTestCase {

    /// Deterministic RNG (SplitMix64) so failures are reproducible.
    private struct SplitMix64: RandomNumberGenerator {
        var state: UInt64
        init(seed: UInt64) { state = seed }
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    /// Asserts that the first `count` entries of `order` are a permutation of 0..<count.
    private func assertValidPermutation(_ order: [UInt32], count: Int,
                                        file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertGreaterThanOrEqual(order.count, count, "output shorter than input", file: file, line: line)
        var seen = [Bool](repeating: false, count: count)
        for k in 0..<count {
            let i = Int(order[k])
            guard i < count, !seen[i] else {
                XCTFail("output is not a permutation (position \(k) -> index \(i))", file: file, line: line)
                return
            }
            seen[i] = true
        }
    }

    /// Asserts the depth sequence visited by `order` is position-for-position identical
    /// to the reference comparison sort `depths.sorted(by: >)` (exact-sort contract).
    private func assertMatchesComparisonSort(depths: [Float], order: [UInt32], count: Int,
                                             file: StaticString = #filePath, line: UInt = #line) {
        assertValidPermutation(order, count: count, file: file, line: line)
        let reference = depths.sorted(by: >)
        for k in 0..<count where depths[Int(order[k])] != reference[k] {
            XCTFail("depth mismatch at position \(k): \(depths[Int(order[k])]) != \(reference[k])",
                    file: file, line: line)
            return
        }
    }

    // MARK: - Regression: fully-trained-PLY outlier distribution (bucket collapse)

    /// Reproduces the real-world failure that motivated FIX 1: a fully-trained garden
    /// PLY (~0.98M splats) has MCMC far outliers, and depth = camera distance *squared*
    /// (measured: d²max ≈ 1.9e8 vs a scene core of ~3.6e3). Under the old 16-bit
    /// [min, max] quantization the bucket width became range/65535 ≈ 3e3 (d²), so 99.8%
    /// of the scene collapsed into bucket 0 and the "sort" kept input order — this test
    /// FAILS against that implementation (the 99% core spans ~1.2 buckets) and passes
    /// only with an exact sort.
    func testTrainedPlyOutlierDistributionSortsExactly() {
        let n = 100_000
        var rng = SplitMix64(seed: 0x6A2D_2026_0726)
        // 99%: scene core, d² uniform in [1.7, 3600] (garden-scale). 1%: far outliers
        // spreading exponentially (log-uniform tail) up to 2e8.
        let depths = (0..<n).map { _ -> Float in
            if Float.random(in: 0..<1, using: &rng) < 0.01 {
                let u = Float.random(in: 0..<1, using: &rng)
                return 3600 * pow(2e8 / 3600, u)
            }
            return Float.random(in: 1.7..<3600, using: &rng)
        }

        let sorter = DepthCountingSorter()
        sorter.sort(depths: depths, count: n)

        assertMatchesComparisonSort(depths: depths, order: sorter.sortedIndices, count: n)
    }

    // MARK: - Exact equivalence vs comparison sort

    func testRandomDepthsMatchComparisonSortExactly() {
        let n = 100_000
        var rng = SplitMix64(seed: 0x5EED_2026_0721)
        // Distance-squared-like magnitudes (the sortByDistance depth metric).
        let depths = (0..<n).map { _ in Float.random(in: 0..<400, using: &rng) }

        let sorter = DepthCountingSorter()
        sorter.sort(depths: depths, count: n)

        assertMatchesComparisonSort(depths: depths, order: sorter.sortedIndices, count: n)
    }

    /// Signed depth range (the !sortByDistance dot-product metric) — exercises the
    /// negative branch of the monotone key mapping (bit inversion).
    func testRandomSignedDepthsMatchComparisonSortExactly() {
        let n = 10_000
        var rng = SplitMix64(seed: 0xD07_2026)
        let depths = (0..<n).map { _ in Float.random(in: -50..<50, using: &rng) }

        let sorter = DepthCountingSorter()
        sorter.sort(depths: depths, count: n)

        assertMatchesComparisonSort(depths: depths, order: sorter.sortedIndices, count: n)
    }

    /// Small deterministic mixed-sign case: descending order of [-1.5, 2, -3, 0.5] is
    /// indices [1, 3, 0, 2].
    func testNegativeDepthsDeterministicOrder() {
        let depths: [Float] = [-1.5, 2.0, -3.0, 0.5]
        let sorter = DepthCountingSorter()
        sorter.sort(depths: depths, count: depths.count)
        XCTAssertEqual(Array(sorter.sortedIndices[0..<depths.count]), [1, 3, 0, 2])
    }

    // MARK: - Edge cases

    func testAllEqualDepthsAreStable() {
        let n = 1_000
        let depths = [Float](repeating: 7.25, count: n)

        let sorter = DepthCountingSorter()
        sorter.sort(depths: depths, count: n)

        // All keys identical: the stable radix scatter must keep input order exactly
        // (identity permutation).
        XCTAssertEqual(Array(sorter.sortedIndices[0..<n]), (0..<n).map { UInt32($0) })
    }

    func testSingleElement() {
        let sorter = DepthCountingSorter()
        sorter.sort(depths: [42.0], count: 1)
        XCTAssertEqual(sorter.sortedIndices[0], 0)
    }

    func testEmptyInputIsANoOp() {
        let sorter = DepthCountingSorter()
        sorter.sort(depths: [], count: 0)  // must not crash
        sorter.sort(depths: [1, 2, 3], count: 0)  // count 0 with non-empty storage
    }

    func testNonFiniteDepthsAreExactAndDeterministic() {
        // Monotone bit-pattern mapping semantics: positive NaN (sign bit 0) sorts
        // beyond +inf (farthest, drawn first); negative NaN (sign bit 1) sorts below
        // -inf (nearest, drawn last). Upstream `sort(by: >)` left NaN placement
        // arbitrary; the radix sort places every bit pattern deterministically.
        let negativeNaN = Float(bitPattern: 0xFFC0_0000)
        XCTAssertTrue(negativeNaN.isNaN)
        let depths: [Float] = [10, .infinity, 30, .nan, 20, -.infinity, negativeNaN, .infinity]
        let n = depths.count

        let sorter = DepthCountingSorter()
        sorter.sort(depths: depths, count: n)
        let order = Array(sorter.sortedIndices[0..<n])

        // Descending key order: +NaN(3), +inf(1), +inf(7) [stable input order],
        // 30(2), 20(4), 10(0), -inf(5), -NaN(6).
        XCTAssertEqual(order, [3, 1, 7, 2, 4, 0, 5, 6])

        // Deterministic: an identical rerun yields the identical permutation.
        let rerunSorter = DepthCountingSorter()
        rerunSorter.sort(depths: depths, count: n)
        XCTAssertEqual(Array(rerunSorter.sortedIndices[0..<n]), order,
                       "non-finite handling must be deterministic")
    }

    func testAllNonFiniteDepths() {
        let negativeNaN = Float(bitPattern: 0xFFC0_0000)
        let depths: [Float] = [.nan, .infinity, -.infinity, negativeNaN]
        let sorter = DepthCountingSorter()
        sorter.sort(depths: depths, count: depths.count)  // must not crash
        // +NaN > +inf > -inf > -NaN in key order.
        XCTAssertEqual(Array(sorter.sortedIndices[0..<depths.count]), [0, 1, 2, 3])
    }

    // MARK: - Storage reuse (mirrors SplatSorter's reuse across sort iterations)

    func testReuseWithSmallerCountAfterLarger() {
        var rng = SplitMix64(seed: 0xCAFE)
        let large = (0..<512).map { _ in Float.random(in: 0..<100, using: &rng) }
        let small: [Float] = [3, 1, 2]

        let sorter = DepthCountingSorter()
        sorter.sort(depths: large, count: large.count)
        sorter.sort(depths: small, count: small.count)

        let order = Array(sorter.sortedIndices[0..<small.count])
        XCTAssertEqual(order, [0, 2, 1], "descending order of [3,1,2] is indices [0,2,1]")
    }
}
