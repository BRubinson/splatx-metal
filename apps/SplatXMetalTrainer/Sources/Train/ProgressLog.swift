import Foundation

struct LossPoint: Identifiable { let id: Int; let iteration: Int; let loss: Double }
struct PSNRPoint: Identifiable { let id: Int; let iteration: Int; let psnr: Double }

struct ProgressLog {
    private(set) var loss: [LossPoint] = []
    private(set) var psnr: [PSNRPoint] = []
    var maxLossPoints = 400
    private var lossSeq = 0
    private var psnrSeq = 0

    mutating func append(iteration: Int, loss value: Double, psnr psnrValue: Double? = nil) {
        loss.append(LossPoint(id: lossSeq, iteration: iteration, loss: value)); lossSeq += 1
        if let p = psnrValue {
            psnr.append(PSNRPoint(id: psnrSeq, iteration: iteration, psnr: p)); psnrSeq += 1
        }
        if loss.count > maxLossPoints { downsampleLoss() }
    }

    private mutating func downsampleLoss() {
        // Thins to half at even intervals while preserving the most recent point.
        let last = loss.last!
        var kept: [LossPoint] = []
        var i = 0
        while i < loss.count { kept.append(loss[i]); i += 2 }
        if kept.last?.id != last.id { kept.append(last) }
        loss = kept
    }
}
