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
}
