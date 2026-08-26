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
    /// The training timelapse (2026-08-04 spec §3.6). An app-only field: it is **not** mapped into
    /// buildOptions (it is not a core option) — the same precedent as saveCheckpoint.
    var timelapse = TimelapseSettings()

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
        // The timelapse is an app-only option (spec §3.6) — it is validated here but not mapped
        // into buildOptions. The values are only read while it is enabled, so a stale value in a
        // disabled state never blocks Start.
        if timelapse.enabled {
            if !TimelapseSettings.isValidEverySteps(timelapse.everySteps) {
                e.append("timelapse capture interval must be a multiple of 50 between 50 and 2000.")
            }
            if !TimelapseSettings.fpsChoices.contains(timelapse.fps) {
                e.append("timelapse fps must be one of 10, 15, 24, 30.")
            }
            if !TimelapseSettings.shortSideChoices.contains(timelapse.shortSide) {
                // Derived from the choice array rather than hardcoded (Task 16 grew the array
                // from 3 to 6 entries and this message was left saying "720, 1080, 1440" — a
                // stale message that no longer matched the Picker it was describing). Deriving
                // it here means the two can never diverge again.
                let list = TimelapseSettings.shortSideChoices.map(String.init).joined(separator: ", ")
                e.append("timelapse resolution must be one of \(list).")
            }
            // 2026-08-05 spec §2.1. 이 조건은 **오늘은 발화할 수 없다** — 필드가 열거형이라
            // 목록 밖의 값이 표현 불가능하고, 매니페스트의 미지 문자열도 디코더가 기본값으로
            // 접는다. 그래도 두는 이유는 `aspectSourceChoices`가 전 케이스의 부분집합이 되는
            // 날(예: COLMAP intrinsics가 없는 프로젝트에서 dataset을 감추는 경우) 폼과 검증이
            // 같은 배열을 보게 하기 위해서다. 메시지도 배열에서 유도한다 — 위 resolution
            // 메시지가 바로 하드코딩 때문에 낡았던 자리다.
            if !TimelapseSettings.aspectSourceChoices.contains(timelapse.aspectSource) {
                let list = TimelapseSettings.aspectSourceChoices
                    .map(\.rawValue).joined(separator: ", ")
                e.append("timelapse aspect must be one of \(list).")
            }
        }
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
