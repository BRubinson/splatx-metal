import SwiftUI
import Charts

/// The training chart (app round 4 spec item 4): a loss LineMark on the left axis plus a PSNR
/// LineMark and PointMark on the right axis.
/// Swift Charts has no independent per-series y scale, so PSNR is linearly normalized into the
/// loss domain for drawing and the right-axis labels (AxisMarks position: .trailing) show the
/// inverse transform — the contract is "each of the two series has an axis it can be read
/// against". When the psnr data is empty it shows loss alone (preserving the previous behavior).
/// PSNR eval ticks are sparse, so the points can be invisible with a LineMark alone; a PointMark
/// is overlaid.
struct TrainingChart: View {
    let log: ProgressLog

    private static let lossLabel = "loss"
    private static let psnrLabel = "PSNR (dB)"

    // The non-finite value guard (app round 4 FIX 1): if training diverges and NaN or Inf gets
    // mixed into loss/psnr, ① the ClosedRange precondition of `.chartYScale(domain: lo...hi)`
    // crashes on a NaN comparison and ② the mark coordinates are polluted. Both the domain
    // computation and the marks take only finite values.
    private var finiteLoss: [LossPoint] { log.loss.filter { $0.loss.isFinite } }
    private var finitePSNR: [PSNRPoint] { log.psnr.filter { $0.psnr.isFinite } }

    var body: some View {
        if log.psnr.isEmpty {
            lossOnlyChart
        } else {
            dualAxisChart
        }
    }

    // MARK: - loss alone (no psnr — testSplit 0, or before the first eval)

    private var lossOnlyChart: some View {
        Chart {
            lossSeries
        }
        .chartForegroundStyleScale([Self.lossLabel: SplatXTheme.accent])
        .chartYScale(domain: .automatic(includesZero: false))
        .frame(minHeight: 160)
    }

    // MARK: - The dual axis

    private var dualAxisChart: some View {
        let lossDomain = lossDomain()
        let psnrDomain = psnrDomain()
        return Chart {
            lossSeries
            ForEach(finitePSNR) { p in
                let y = Self.rescale(p.psnr, from: psnrDomain, to: lossDomain)
                LineMark(x: .value("iter", p.iteration),
                         y: .value("psnr", y),
                         series: .value("series", Self.psnrLabel))
                    .foregroundStyle(by: .value("series", Self.psnrLabel))
                PointMark(x: .value("iter", p.iteration),
                          y: .value("psnr", y))
                    .foregroundStyle(by: .value("series", Self.psnrLabel))
                    .symbolSize(18)
            }
        }
        .chartForegroundStyleScale([Self.lossLabel: SplatXTheme.accent,
                                    Self.psnrLabel: Color.green])
        .chartYScale(domain: lossDomain.lo...lossDomain.hi)
        .chartYAxis {
            AxisMarks(position: .leading)
            // The right axis: the tick positions share the loss domain coordinates (automatic
            // placement) and only the labels are transformed back into PSNR.
            AxisMarks(position: .trailing) { value in
                if let y = value.as(Double.self) {
                    AxisValueLabel {
                        Text(String(format: "%.1f",
                                    Self.rescale(y, from: lossDomain, to: psnrDomain)))
                            .foregroundStyle(Color.green)
                    }
                }
            }
        }
        .frame(minHeight: 160)
    }

    @ChartContentBuilder private var lossSeries: some ChartContent {
        ForEach(finiteLoss) {
            LineMark(x: .value("iter", $0.iteration),
                     y: .value("loss", $0.loss),
                     series: .value("series", Self.lossLabel))
                .foregroundStyle(by: .value("series", Self.lossLabel))
        }
    }

    // MARK: - Normalization

    /// The loss y domain: the finite values' min…max with 5% padding. A degeneracy (all values
    /// equal) is widened by ±0.5, and with no finite values it falls back to (0,1) (harmless, since
    /// there are no marks either).
    private func lossDomain() -> (lo: Double, hi: Double) {
        let values = finiteLoss.map(\.loss)
        guard let lo = values.min(), let hi = values.max() else { return (0, 1) }
        guard lo < hi else { return (lo - 0.5, hi + 0.5) }
        let pad = (hi - lo) * 0.05
        return (lo - pad, hi + pad)
    }

    /// The PSNR domain: the finite values' min…max. A degeneracy (a single point, for example) is
    /// widened by ±0.5 dB so the inverse transform is always defined, and with no finite values it
    /// falls back to (0,1).
    private func psnrDomain() -> (lo: Double, hi: Double) {
        let values = finitePSNR.map(\.psnr)
        guard let lo = values.min(), let hi = values.max() else { return (0, 1) }
        guard lo < hi else { return (lo - 0.5, hi + 0.5) }
        return (lo, hi)
    }

    /// The linear map [from.lo, from.hi] → [to.lo, to.hi]. The domain builders guarantee lo < hi.
    private static func rescale(_ x: Double,
                                from: (lo: Double, hi: Double),
                                to: (lo: Double, hi: Double)) -> Double {
        to.lo + (x - from.lo) / (from.hi - from.lo) * (to.hi - to.lo)
    }
}
