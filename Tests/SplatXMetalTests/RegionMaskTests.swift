// Region masks and the shape regulariser (v0.3.0): the kernel checks run in the C++ suite
// (MaskTests.cpp, RegulariserTests.cpp); the end-to-end checks run here through SplatXTrainer.
import CoreGraphics
import Foundation
import ImageIO
import SplatXTests
import Testing
import UniformTypeIdentifiers
@testable import SplatXMetal

extension GpuTests {
@Suite struct RegionMaskTests {

    private static var testsDir: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent() }
    private static var kernelSrcDir: URL {
        testsDir.deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/SplatXKernelSrc")
    }

    private static func workDir(_ tag: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("splatx-mask-\(tag)-\(UUID().uuidString)")
    }

    private static func options(_ dir: URL, _ name: String, iterations: Int) -> SplatXTrainer.Options {
        var o = SplatXTrainer.Options(outputURL: dir.appendingPathComponent(name))
        o.iterations = iterations
        o.factor = 1
        o.shDegree = 0
        o.maxPoints = 400
        o.mcmc = false
        o.ssimWeight = 0.2
        o.testSplit = 0
        o.resident = true
        o.thermalMode = .performance
        o.logEvery = 1_000
        o.kernelSourceDirectory = kernelSrcDir
        return o
    }

    /// masks/<stem>.png at a quarter of the 64×48 view: columns ≥ inFromColumn are in-region.
    private static func writeMasks(in dir: URL, cameraCount: Int, inFromColumn: Int) throws {
        let masks = dir.appendingPathComponent("masks")
        try FileManager.default.createDirectory(at: masks, withIntermediateDirectories: true)
        let w = 16, h = 12
        var bytes = [UInt8](repeating: 0, count: w * h)
        for y in 0..<h { for x in inFromColumn..<w { bytes[y * w + x] = 255 } }
        for i in 0..<cameraCount {
            guard let provider = CGDataProvider(data: Data(bytes) as CFData),
                  let image = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 8,
                                      bytesPerRow: w, space: CGColorSpaceCreateDeviceGray(),
                                      bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                      provider: provider, decode: nil, shouldInterpolate: false,
                                      intent: .defaultIntent),
                  let dest = CGImageDestinationCreateWithURL(
                      masks.appendingPathComponent(String(format: "img%03d.png", i)) as CFURL,
                      UTType.png.identifier as CFString, 1, nil)
            else { throw NSError(domain: "RegionMaskTests", code: 1) }
            CGImageDestinationAddImage(dest, image, nil)
            guard CGImageDestinationFinalize(dest) else { throw NSError(domain: "RegionMaskTests", code: 2) }
        }
    }

    private static func sigmoid(_ x: Float) -> Double { 1 / (1 + exp(-Double(x))) }

    // The C++ mask_* and regulariser_* tests.
    @Test func kernelMaskSuite() {
        var total: Int32 = 0
        var failed: Int32 = 0
        let rc = sx_tests_run_prefix("mask_", nil, nil, &total, &failed)
        #expect(rc == 0)
        #expect(total == 10)
        #expect(failed == 0)
    }

    @Test func kernelRegulariserSuite() {
        var total: Int32 = 0
        var failed: Int32 = 0
        let rc = sx_tests_run_prefix("regulariser_", nil, nil, &total, &failed)
        #expect(rc == 0)
        #expect(total == 4)
        #expect(failed == 0)
    }

    // With every new weight 0 and no masks the run is v0.2.0's: the SyntheticColmap PLY matches the
    // v0.1.0 golden (the same options and gates as DepthSupervisionTests.depthWeightZeroMatchesV010).
    @Test func zeroWeightsNoMasksMatchGolden() async throws {
        let dir = Self.workDir("golden")
        defer { try? FileManager.default.removeItem(at: dir) }
        try SyntheticColmap.makeBundle(at: dir)
        var o = SplatXTrainer.Options(outputURL: dir.appendingPathComponent("out.ply"))
        o.iterations = 60
        o.factor = 1
        o.shDegree = 1
        o.maxPoints = 200
        o.mcmc = false
        o.ssimWeight = 0.2
        o.testSplit = 0
        o.resident = true
        o.thermalMode = .performance
        o.logEvery = 1_000
        o.kernelSourceDirectory = Self.kernelSrcDir
        o.depthWeight = 0
        o.flattenWeight = 0
        o.needleWeight = 0
        o.needleRatio = 5
        let r = try await SplatXTrainer().train(dataDirectory: dir, options: o) { _ in }
        #expect(r.maskViews == 0)
        #expect(r.finalMaskedPSNR == nil)

        let golden = try Ply.load(Self.testsDir.appendingPathComponent("Fixtures/v010-synthetic-60it.ply"))
        let got = try Ply.load(o.outputURL)
        #expect(got.names == golden.names)
        #expect(got.count == golden.count)
        guard got.names == golden.names, got.count == golden.count else { return }
        let gates: [(prefix: String, meanGate: Double)] = [
            ("x", 1e-5), ("y", 1e-5), ("z", 1e-5), ("f_dc_", 5e-3), ("f_rest_", 5e-4),
            ("opacity", 3e-2), ("scale_", 0.1), ("rot_", 6e-2),
        ]
        var worstElement = 0.0
        for (prefix, gate) in gates {
            let cols = got.names.indices.filter {
                prefix.count == 1 ? got.names[$0] == prefix : got.names[$0].hasPrefix(prefix)
            }
            var sum = 0.0
            var n = 0
            for c in cols {
                for row in 0..<got.count {
                    let d = abs(Double(got.value(row, c)) - Double(golden.value(row, c)))
                    sum += d
                    n += 1
                    worstElement = max(worstElement, d)
                }
            }
            let mean = n > 0 ? sum / Double(n) : 0
            print("  [v0.3.0 weights 0, golden] \(prefix) mean |d| \(mean) (gate \(gate))")
            #expect(mean <= gate, "\(prefix) drifted: mean |d| \(mean) > \(gate)")
        }
        print("  [v0.3.0 weights 0, golden] N \(got.count), worst element |d| \(worstElement) (gate 1.0)")
        #expect(worstElement <= 1.0)
    }

    // Masks covering the right half of every view confine MCMC: after 300 iterations at least 80%
    // of the opacity mass sits in the masked-in half (world x ≥ 0 projects to view u ≥ 32), and the
    // held-out masked PSNR is finite.
    @Test func maskedTrainingConfinesOpacityMass() async throws {
        let dir = Self.workDir("confine")
        defer { try? FileManager.default.removeItem(at: dir) }
        try SyntheticColmap.makeBundle(at: dir, cameraCount: 6, pointCount: 400)

        func run(_ name: String) async throws -> (SplatXTrainer.RunResult, Double) {
            var o = Self.options(dir, name, iterations: 300)
            o.mcmc = true
            o.capMax = 600
            o.testSplit = 3
            let r = try await SplatXTrainer().train(dataDirectory: dir, options: o) { _ in }
            let ply = try Ply.load(o.outputURL)
            let xi = try #require(ply.names.firstIndex(of: "x"))
            let oi = try #require(ply.names.firstIndex(of: "opacity"))
            var inMass = 0.0, total = 0.0
            for row in 0..<ply.count {
                let m = Self.sigmoid(ply.value(row, oi))
                total += m
                if ply.value(row, xi) >= 0 { inMass += m }
            }
            return (r, total > 0 ? inMass / total : 0)
        }
        let (plain, plainShare) = try await run("unmasked.ply")
        try Self.writeMasks(in: dir, cameraCount: 6, inFromColumn: 8)
        let (masked, share) = try await run("masked.ply")
        print("  [confine] in-region opacity mass share: masked \(share), unmasked \(plainShare); "
              + "masked PSNR \(masked.finalMaskedPSNR ?? .nan) dB, whole-image \(masked.finalPSNR ?? .nan) dB; "
              + "mask views \(masked.maskViews)")
        #expect(plain.maskViews == 0 && plain.finalMaskedPSNR == nil)
        #expect(masked.maskViews == 6)
        #expect(share >= 0.8)
        let mpsnr = try #require(masked.finalMaskedPSNR)
        #expect(mpsnr.isFinite)
    }

    // The needle hinge lowers the p99 of s1/s2 against the same run with it off.
    @Test func needleLowersP99Ratio() async throws {
        let dir = Self.workDir("needle")
        defer { try? FileManager.default.removeItem(at: dir) }
        try SyntheticColmap.makeBundle(at: dir, cameraCount: 6, pointCount: 400)

        func p99(_ weight: Float) async throws -> Double {
            var o = Self.options(dir, "needle-\(weight).ply", iterations: 300)
            o.needleWeight = weight
            o.needleRatio = 1.5
            _ = try await SplatXTrainer().train(dataDirectory: dir, options: o) { _ in }
            let ply = try Ply.load(o.outputURL)
            let si = (0..<3).compactMap { ply.names.firstIndex(of: "scale_\($0)") }
            #expect(si.count == 3)
            var ratios: [Double] = []
            for row in 0..<ply.count {
                let s = si.map { exp(Double(ply.value(row, $0))) }.sorted(by: >)
                ratios.append(s[0] / s[1])
            }
            ratios.sort()
            return ratios[min(ratios.count - 1, Int(Double(ratios.count) * 0.99))]
        }
        let off = try await p99(0)
        let on = try await p99(10)
        print("  [needle] p99 s1/s2: off \(off), on \(on)")
        #expect(on < off)
    }
}
}
