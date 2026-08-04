import Foundation
import SplatXMetal

struct TrainOptionsDraft {
    var factor = 4
    var iterations = 7000
    var mcmc = true
    var capMax = 500_000
    var shDegree = 3
    var ssimWeight = 0.2
    // Advanced
    var antialiased = false
    var resident = true
    // Round 5: .performance by default (no governor intervention — the fastest run to completion).
    // balanced and lowheat are selected in the advanced form.
    var thermalMode: SplatXTrainer.ThermalMode = .performance
    // Round 4: eval is enabled by default (testSplit 8, evalEvery 500) — periodic PSNR during
    // training plus finalPSNR.
    // A preset apply touches only three fields (iterations/factor/capMax), so these defaults
    // survive.
    var testSplit = 8
    var evalEvery = 500
    var autoCap = false
    var maxPoints = 120_000
    var logEvery = 50
    var saveCheckpoint = false
    var checkpointEvery = 0

    func validationErrors() -> [String] {
        var e: [String] = []
        if ![1,2,4,8].contains(factor) { e.append("factor must be one of 1, 2, 4, 8.") }
        if !(0...3).contains(shDegree) { e.append("SH degree must be 0–3.") }
        if !(ssimWeight.isFinite && (0.0...1.0).contains(ssimWeight)) { e.append("SSIM weight must be 0–1.") }
        if iterations < 1 { e.append("iterations must be ≥ 1.") }
        if testSplit < 0 || testSplit == 1 { e.append("test-split must be 0 or ≥ 2 (1 not allowed).") }
        if evalEvery > 0 && testSplit <= 0 { e.append("eval-every > 0 requires test-split > 0.") }
        if autoCap && !mcmc { e.append("auto-cap requires MCMC to be enabled.") }
        if maxPoints < 1 { e.append("max-points must be ≥ 1.") }
        if mcmc && capMax < 1 { e.append("cap-max must be ≥ 1.") }
        return e
    }
    var isValid: Bool { validationErrors().isEmpty }

    func buildOptions(outputURL: URL, previewURL: URL?, checkpointURL: URL?, metallibURL: URL) -> SplatXTrainer.Options {
        var o = SplatXTrainer.Options(outputURL: outputURL)
        o.factor = factor; o.iterations = iterations; o.mcmc = mcmc; o.capMax = capMax
        o.shDegree = shDegree; o.ssimWeight = Float(ssimWeight)
        o.antialiased = antialiased; o.resident = resident; o.thermalMode = thermalMode
        o.testSplit = testSplit; o.evalEvery = evalEvery; o.autoCap = autoCap
        o.maxPoints = maxPoints; o.logEvery = logEvery
        o.checkpointEvery = checkpointEvery
        o.previewURL = previewURL
        o.checkpointURL = checkpointURL
        o.metallibURL = metallibURL
        return o
    }
}
