// Depth supervision (v0.2.0): the kernel checks run in the C++ suite (DepthTests.cpp — the
// expected-depth channel against RGB, the finite-difference gradient, the resident path and the
// sidecar loader); the end-to-end checks run here through SplatXTrainer.
import CoreGraphics
import Foundation
import ImageIO
import SplatXTests
import Testing
import UniformTypeIdentifiers
@testable import SplatXMetal

extension GpuTests {
@Suite struct DepthSupervisionTests {

    private static var testsDir: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent() }
    private static var kernelSrcDir: URL {
        testsDir.deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/SplatXKernelSrc")
    }

    private static func workDir(_ tag: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("splatx-depth-\(tag)-\(UUID().uuidString)")
    }

    // (1) and (2), plus the resident path and the sidecar loader: the C++ depth_* tests.
    @Test func kernelDepthSuite() {
        var total: Int32 = 0
        var failed: Int32 = 0
        let rc = sx_tests_run_prefix("depth_", nil, nil, &total, &failed)
        #expect(rc == 0)
        #expect(total == 4)
        #expect(failed == 0)
    }

    // (3) At depthWeight 0 training is v0.1.0's: the SyntheticColmap PLY matches the v0.1.0 golden
    // (Fixtures/v010-synthetic-60it.ply, trained by the v0.1.0 tag with these options) within ten
    // times the run-to-run spread v0.1.0 shows against itself (raster bwd accumulates atomically,
    // so no two runs are bit-identical; the forward is pinned bit-exactly in the C++ suite).
    @Test func depthWeightZeroMatchesV010() async throws {
        let dir = Self.workDir("v010")
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
        _ = try await SplatXTrainer().train(dataDirectory: dir, options: o) { _ in }

        let golden = try Ply.load(Self.testsDir.appendingPathComponent("Fixtures/v010-synthetic-60it.ply"))
        let got = try Ply.load(o.outputURL)
        #expect(got.names == golden.names)
        #expect(got.count == golden.count)
        guard got.names == golden.names, got.count == golden.count else { return }
        // Measured v0.1.0 self-spread (seven runs against the golden), worst group means:
        // xyz 6e-7, f_dc 5e-4, f_rest 3e-5, opacity 3.4e-3, scale 1.1e-2, rot 6e-3; worst element 0.15
        // (the element tail is Adam amplifying atomic-order noise, so it gets only a sanity bound).
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
                for r in 0..<got.count {
                    let d = abs(Double(got.value(r, c)) - Double(golden.value(r, c)))
                    sum += d
                    n += 1
                    worstElement = max(worstElement, d)
                }
            }
            let mean = n > 0 ? sum / Double(n) : 0
            print("  [v0.1.0 golden] \(prefix) mean |d| \(mean) (gate \(gate))")
            #expect(mean <= gate, "\(prefix) drifted from v0.1.0: mean |d| \(mean) > \(gate)")
        }
        print("  [v0.1.0 golden] N \(got.count), worst element |d| \(worstElement) (gate 1.0)")
        #expect(worstElement <= 1.0)
    }

    // (4) On a two-plane scene with exact LiDAR-style sidecars, depth supervision lowers the
    // held-out expected-depth RMSE against the unsupervised run from the same seed.
    @Test func supervisionLowersDepthRMSE() async throws {
        let dir = Self.workDir("planes")
        defer { try? FileManager.default.removeItem(at: dir) }
        try TwoPlaneColmap.makeBundle(at: dir)

        func run(_ weight: Float) async throws -> SplatXTrainer.RunResult {
            var o = SplatXTrainer.Options(outputURL: dir.appendingPathComponent("out-\(weight).ply"))
            o.iterations = 400
            o.factor = 1
            o.shDegree = 0
            o.maxPoints = 2_000
            o.mcmc = false
            o.ssimWeight = 0.2
            o.testSplit = 4
            o.resident = true
            o.thermalMode = .performance
            o.logEvery = 100
            o.kernelSourceDirectory = Self.kernelSrcDir
            o.depthWeight = weight
            return try await SplatXTrainer().train(dataDirectory: dir, options: o) { _ in }
        }
        let rgb = try await run(0)
        let depth = try await run(0.5)
        let rgbRMSE = try #require(rgb.finalDepthRMSE)
        let depthRMSE = try #require(depth.finalDepthRMSE)
        print("  [two planes] held-out depth RMSE: unsupervised \(rgbRMSE) m, supervised \(depthRMSE) m; "
              + "PSNR \(rgb.finalPSNR ?? .nan) vs \(depth.finalPSNR ?? .nan) dB; "
              + "\(rgb.trainSeconds * 1000 / 400) vs \(depth.trainSeconds * 1000 / 400) ms/iter")
        #expect(depthRMSE.isFinite && rgbRMSE.isFinite)
        #expect(depthRMSE < 0.8 * rgbRMSE)
    }
}
}

// MARK: - Binary little-endian PLY reader (the INRIA layout the trainer writes)

struct Ply {
    let names: [String]
    let count: Int
    let values: [Float]

    func value(_ row: Int, _ col: Int) -> Float { values[row * names.count + col] }

    static func load(_ url: URL) throws -> Ply {
        let data = try Data(contentsOf: url)
        guard let end = data.range(of: Data("end_header\n".utf8)) else {
            throw NSError(domain: "Ply", code: 1)
        }
        let header = String(decoding: data[..<end.lowerBound], as: UTF8.self)
        var names: [String] = []
        var count = 0
        for line in header.split(separator: "\n") {
            let t = line.split(separator: " ")
            if t.first == "element", t.count == 3, t[1] == "vertex" { count = Int(t[2]) ?? 0 }
            if t.first == "property", let n = t.last { names.append(String(n)) }
        }
        let body = data[end.upperBound...]
        var values = [Float](repeating: 0, count: count * names.count)
        values.withUnsafeMutableBytes { dst in
            _ = body.prefix(dst.count).copyBytes(to: dst)
        }
        return Ply(names: names, count: count, values: values)
    }
}

// MARK: - Two-plane COLMAP bundle with depth sidecars

/// A front panel at z = 2 m (x ∈ [-0.6, 0.2], y ∈ [-0.45, 0.45]) before a wall at z = 4 m, seen by
/// twelve forward-facing cameras on a 4×3 grid; images are ray-cast textures and the sidecars are
/// the exact camera-z depth at a quarter of the view resolution (16-bit mm, confidence 2). The
/// seed points scatter at uniformly random depths, so RGB alone does not pin the geometry.
private enum TwoPlaneColmap {
    static let width = 64, height = 48
    static let focal = 50.0
    static let depthWidth = 16, depthHeight = 12

    static func makeBundle(at root: URL) throws {
        let fm = FileManager.default
        let sparse = root.appendingPathComponent("sparse/0")
        let images = root.appendingPathComponent("images")
        let depths = root.appendingPathComponent("depths")
        for d in [sparse, images, depths] { try fm.createDirectory(at: d, withIntermediateDirectories: true) }

        var cams = Data()
        cams.le(UInt64(1)); cams.le(Int32(1)); cams.le(Int32(1))
        cams.le(UInt64(width)); cams.le(UInt64(height))
        for v in [focal, focal, Double(width) / 2, Double(height) / 2] { cams.le(v) }
        try cams.write(to: sparse.appendingPathComponent("cameras.bin"))

        var imgs = Data()
        let centres: [(Double, Double)] = (0..<12).map { (i: Int) -> (Double, Double) in
            let cx: Double = 0.12 * Double(i % 4) - 0.18
            let cy: Double = 0.1 * Double(i / 4) - 0.1
            return (cx, cy)
        }
        imgs.le(UInt64(centres.count))
        for (i, c) in centres.enumerated() {
            let name = String(format: "view%03d", i)
            imgs.le(Int32(i + 1))
            for q in [1.0, 0.0, 0.0, 0.0] { imgs.le(q) }
            for t in [-c.0, -c.1, 0.0] { imgs.le(t) } // world→cam translation = −centre
            imgs.le(Int32(1))
            imgs.append(contentsOf: Array("\(name).png".utf8)); imgs.append(0)
            imgs.le(UInt64(0))
            try writeView(centre: c, rgbURL: images.appendingPathComponent("\(name).png"),
                          depthURL: depths.appendingPathComponent("\(name).png"),
                          confURL: depths.appendingPathComponent("\(name).conf.png"))
        }
        try imgs.write(to: sparse.appendingPathComponent("images.bin"))

        var pts = Data()
        let count = 1_500
        pts.le(UInt64(count))
        var state: UInt64 = 0x2D_5EED
        func uniform(_ lo: Double, _ hi: Double) -> Double {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return lo + (hi - lo) * Double((state >> 33) % 1_000_000) / 1_000_000
        }
        for i in 0..<count {
            pts.le(UInt64(i + 1))
            let z: Double = uniform(1.5, 5.0)
            let x: Double = uniform(-0.65, 0.65) * z / 2
            let y: Double = uniform(-0.5, 0.5) * z / 2
            pts.le(x); pts.le(y); pts.le(z)
            pts.append(128); pts.append(128); pts.append(128)
            pts.le(Double(0)); pts.le(UInt64(0))
        }
        try pts.write(to: sparse.appendingPathComponent("points3D.bin"))
    }

    /// Ray-casts one pixel centre (u, v) in view pixels: (depth, rgb).
    static func trace(_ u: Double, _ v: Double, centre: (Double, Double)) -> (Double, (Double, Double, Double)) {
        let dx: Double = (u - Double(width) / 2) / focal
        let dy: Double = (v - Double(height) / 2) / focal
        let px: Double = centre.0 + 2 * dx
        let py: Double = centre.1 + 2 * dy
        if px >= -0.6 && px <= 0.2 && py >= -0.45 && py <= 0.45 {
            let r: Double = 0.5 + 0.4 * sin(25 * px)
            let g: Double = 0.3 + 0.25 * cos(19 * py)
            let b: Double = 0.6 + 0.3 * sin(13 * (px + py))
            return (2, (r, g, b))
        }
        let wx: Double = centre.0 + 4 * dx
        let wy: Double = centre.1 + 4 * dy
        let r: Double = 0.4 + 0.35 * cos(11 * wy)
        let g: Double = 0.55 + 0.35 * sin(9 * wx)
        let b: Double = 0.25 + 0.2 * cos(7 * (wx - wy))
        return (4, (r, g, b))
    }

    static func writeView(centre: (Double, Double), rgbURL: URL, depthURL: URL, confURL: URL) throws {
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let c = trace(Double(x) + 0.5, Double(y) + 0.5, centre: centre).1
                let o = (y * width + x) * 4
                rgba[o] = UInt8(max(0, min(255, c.0 * 255)))
                rgba[o + 1] = UInt8(max(0, min(255, c.1 * 255)))
                rgba[o + 2] = UInt8(max(0, min(255, c.2 * 255)))
            }
        }
        try writePNG(bytes: rgba, w: width, h: height, bitsPerComponent: 8, bitsPerPixel: 32,
                     space: CGColorSpace(name: CGColorSpace.sRGB)!,
                     info: CGImageAlphaInfo.noneSkipLast.rawValue, url: rgbURL)
        var mm = [UInt8](repeating: 0, count: depthWidth * depthHeight * 2)
        let scale = Double(width) / Double(depthWidth)
        for y in 0..<depthHeight {
            for x in 0..<depthWidth {
                let d = trace((Double(x) + 0.5) * scale, (Double(y) + 0.5) * scale, centre: centre).0
                let v = UInt16(d * 1000)
                mm[(y * depthWidth + x) * 2] = UInt8(v & 0xFF)
                mm[(y * depthWidth + x) * 2 + 1] = UInt8(v >> 8)
            }
        }
        try writePNG(bytes: mm, w: depthWidth, h: depthHeight, bitsPerComponent: 16, bitsPerPixel: 16,
                     space: CGColorSpaceCreateDeviceGray(),
                     info: CGImageAlphaInfo.none.rawValue | CGBitmapInfo.byteOrder16Little.rawValue,
                     url: depthURL)
        try writePNG(bytes: [UInt8](repeating: 2, count: depthWidth * depthHeight), w: depthWidth,
                     h: depthHeight, bitsPerComponent: 8, bitsPerPixel: 8,
                     space: CGColorSpaceCreateDeviceGray(), info: CGImageAlphaInfo.none.rawValue,
                     url: confURL)
    }

    static func writePNG(bytes: [UInt8], w: Int, h: Int, bitsPerComponent: Int, bitsPerPixel: Int,
                         space: CGColorSpace, info: UInt32, url: URL) throws {
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let image = CGImage(width: w, height: h, bitsPerComponent: bitsPerComponent,
                                  bitsPerPixel: bitsPerPixel, bytesPerRow: w * bitsPerPixel / 8,
                                  space: space, bitmapInfo: CGBitmapInfo(rawValue: info),
                                  provider: provider, decode: nil, shouldInterpolate: false,
                                  intent: .defaultIntent),
              let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw NSError(domain: "TwoPlaneColmap", code: 1) }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw NSError(domain: "TwoPlaneColmap", code: 2) }
    }
}

private extension Data {
    mutating func le(_ v: UInt64) { Swift.withUnsafeBytes(of: v.littleEndian) { append(contentsOf: $0) } }
    mutating func le(_ v: Int32) { Swift.withUnsafeBytes(of: v.littleEndian) { append(contentsOf: $0) } }
    mutating func le(_ v: Double) {
        Swift.withUnsafeBytes(of: v.bitPattern.littleEndian) { append(contentsOf: $0) }
    }
}
