import XCTest
import SplatXMetal
@testable import SplatXMetalTrainer

final class TrainOptionsDraftTests: XCTestCase {
    func testDefaultsAreValidAndMirrorOptions() {
        let d = TrainOptionsDraft()
        XCTAssertTrue(d.isValid, "defaults must be valid: \(d.validationErrors())")
        let o = d.buildOptions(outputURL: URL(fileURLWithPath: "/tmp/a.ply"),
                               previewURL: nil, checkpointURL: nil,
                               metallibURL: URL(fileURLWithPath: "/tmp/x.metallib"))
        XCTAssertEqual(o.factor, 4)
        XCTAssertEqual(o.iterations, 7000)
        XCTAssertTrue(o.mcmc)
        XCTAssertEqual(o.capMax, 500_000)
        XCTAssertEqual(o.shDegree, 3)
        XCTAssertEqual(o.ssimWeight, 0.2, accuracy: 1e-6)
        XCTAssertEqual(o.testSplit, 8)                              // round 4: eval is on by default
        XCTAssertEqual(o.evalEvery, 500)                            // periodic PSNR ticks during training
        XCTAssertEqual(o.thermalMode, .performance)                 // round 5: no governor intervention by default — fastest run
        XCTAssertEqual(o.metallibURL, URL(fileURLWithPath: "/tmp/x.metallib"))
    }

    func testBuildOptionsMapsURLs() {
        let d = TrainOptionsDraft()
        let previewURL = URL(fileURLWithPath: "/tmp/preview.png")
        let checkpointURL = URL(fileURLWithPath: "/tmp/ckpt.bin")
        let o = d.buildOptions(outputURL: URL(fileURLWithPath: "/tmp/a.ply"),
                               previewURL: previewURL, checkpointURL: checkpointURL,
                               metallibURL: URL(fileURLWithPath: "/tmp/x.metallib"))
        XCTAssertEqual(o.previewURL, previewURL)                    // previewURL passes through
        XCTAssertEqual(o.checkpointURL, checkpointURL)              // checkpointURL passes through
    }

    func testValidationRules() {
        var d = TrainOptionsDraft(); d.factor = 3
        XCTAssertFalse(d.isValid)                                   // factor ∈ {1,2,4,8}
        d = TrainOptionsDraft(); d.shDegree = 4
        XCTAssertFalse(d.isValid)                                   // sh 0..3
        d = TrainOptionsDraft(); d.ssimWeight = 1.5
        XCTAssertFalse(d.isValid)                                   // ssim [0,1]
        d = TrainOptionsDraft(); d.ssimWeight = .nan
        XCTAssertFalse(d.isValid)                                   // ssim isFinite
        d = TrainOptionsDraft(); d.iterations = 0
        XCTAssertFalse(d.isValid)                                   // iterations >= 1
        d = TrainOptionsDraft(); d.testSplit = 1
        XCTAssertFalse(d.isValid)                                   // testSplit != 1
        d = TrainOptionsDraft(); d.evalEvery = 100; d.testSplit = 0
        XCTAssertFalse(d.isValid)                                   // evalEvery>0 ⇒ testSplit>0
        d = TrainOptionsDraft(); d.evalEvery = 100; d.testSplit = 8
        XCTAssertTrue(d.isValid)
        d = TrainOptionsDraft(); d.autoCap = true; d.mcmc = false
        XCTAssertFalse(d.isValid)                                   // autoCap ⇒ mcmc
    }

    // MARK: - Timelapse options (2026-08-04 spec §3.6/§3.7)

    /// While the timelapse is off its fields are never read, so they never produce a validation
    /// error — the same shape as the checkpointEvery field under saveCheckpoint.
    func testTimelapseValidationIsSkippedWhileDisabled() {
        var d = TrainOptionsDraft()
        d.timelapse.enabled = false
        d.timelapse.everySteps = 75      // not a multiple of 50
        d.timelapse.fps = 7
        d.timelapse.shortSide = 999
        XCTAssertTrue(d.isValid)
        XCTAssertTrue(d.validationErrors().isEmpty)
    }

    func testTimelapseDefaultsAreValidWhenEnabled() {
        var d = TrainOptionsDraft()
        d.timelapse.enabled = true
        XCTAssertTrue(d.isValid, "\(d.validationErrors())")
    }

    func testTimelapseInvalidValuesAreReported() {
        var d = TrainOptionsDraft()
        d.timelapse.enabled = true
        d.timelapse.everySteps = 75
        d.timelapse.fps = 7
        d.timelapse.shortSide = 999
        let errors = d.validationErrors()
        XCTAssertFalse(d.isValid)
        XCTAssertEqual(errors.count, 3)
        XCTAssertTrue(errors.contains { $0.contains("multiple of 50") }, "\(errors)")
        XCTAssertTrue(errors.contains { $0.contains("timelapse fps") }, "\(errors)")
        XCTAssertTrue(errors.contains { $0.contains("timelapse resolution") }, "\(errors)")
    }

    /// Pins the message to the array rather than to hardcoded numbers, so a future resize of
    /// `shortSideChoices` (as happened once already — Task 16 grew it from 3 to 6 and the message
    /// was left behind) fails this test instead of silently going stale again.
    func testTimelapseResolutionMessageIsDerivedFromChoices() {
        var d = TrainOptionsDraft()
        d.timelapse.enabled = true
        d.timelapse.shortSide = 999_999   // not in shortSideChoices, any invalid value works
        let errors = d.validationErrors()
        guard let msg = errors.first(where: { $0.contains("timelapse resolution") }) else {
            XCTFail("expected a timelapse resolution error: \(errors)"); return
        }
        for choice in TimelapseSettings.shortSideChoices {
            XCTAssertTrue(msg.contains("\(choice)"), "message must mention \(choice): \(msg)")
        }
    }

    /// 2026-08-05 §2.1: every choice the Picker offers must pass validation. The enum makes an
    /// out-of-list value unrepresentable today, so this — not a "reject an invalid value" test —
    /// is what catches `aspectSourceChoices` and the accepted set drifting apart.
    func testEveryOfferedAspectSourceIsAccepted() {
        for source in TimelapseSettings.aspectSourceChoices {
            var d = TrainOptionsDraft()
            d.timelapse.enabled = true
            d.timelapse.aspectSource = source
            XCTAssertTrue(d.isValid, "\(source): \(d.validationErrors())")
        }
    }

    /// The timelapse is an app-only field and is **not** mapped into the core options
    /// (the saveCheckpoint precedent) — turning it on must not change a single core value.
    func testTimelapseDoesNotAffectBuildOptions() {
        let out = URL(fileURLWithPath: "/p/runs/r/output.ply")
        let lib = URL(fileURLWithPath: "/p/splatx.metallib")
        var off = TrainOptionsDraft()
        off.timelapse.enabled = false
        var on = TrainOptionsDraft()
        on.timelapse.enabled = true
        on.timelapse.everySteps = 200
        on.timelapse.format = .gif
        let a = off.buildOptions(outputURL: out, previewURL: nil, checkpointURL: nil,
                                 metallibURL: lib)
        let b = on.buildOptions(outputURL: out, previewURL: nil, checkpointURL: nil,
                                metallibURL: lib)
        XCTAssertEqual(a.iterations, b.iterations)
        XCTAssertEqual(a.factor, b.factor)
        XCTAssertEqual(a.capMax, b.capMax)
        XCTAssertEqual(a.shDegree, b.shDegree)
        XCTAssertEqual(a.ssimWeight, b.ssimWeight)
        XCTAssertEqual(a.evalEvery, b.evalEvery)
        XCTAssertEqual(a.checkpointEvery, b.checkpointEvery)
    }
}
