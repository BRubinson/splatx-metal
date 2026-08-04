import Foundation

/// The sharpness decision for video frames. It uses only "pick the maximum within a window", with
/// no global threshold — the absolute value of the Laplacian variance is scene-dependent, so a
/// threshold does not generalize (common practice in the industry).
enum SharpnessScorer {
    /// The variance of the 3×3 Laplacian (4-neighbor) response over a grayscale 8-bit buffer. The
    /// larger, the sharper.
    static func laplacianVariance(gray: [UInt8], width: Int, height: Int) -> Double {
        guard width > 2, height > 2, gray.count == width * height else { return 0 }
        var sum = 0.0, sumSq = 0.0
        let n = Double((width - 2) * (height - 2))
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let i = y * width + x
                let lap = 4.0 * Double(gray[i])
                    - Double(gray[i - 1]) - Double(gray[i + 1])
                    - Double(gray[i - width]) - Double(gray[i + width])
                sum += lap
                sumSq += lap * lap
            }
        }
        let mean = sum / n
        return sumSq / n - mean * mean
    }

    /// Splits the frame score array evenly into windowCount contiguous spans and returns one
    /// maximum-score index per span (in ascending order). When there are fewer frames than
    /// windows, it returns all of them.
    static func selectBestPerWindow(scores: [Double], windowCount: Int) -> [Int] {
        guard windowCount > 0, !scores.isEmpty else { return [] }
        guard scores.count > windowCount else { return Array(scores.indices) }
        var picked: [Int] = []
        for w in 0..<windowCount {
            let lo = scores.count * w / windowCount
            let hi = scores.count * (w + 1) / windowCount
            if let best = (lo..<hi).max(by: { scores[$0] < scores[$1] }) { picked.append(best) }
        }
        return picked
    }
}
