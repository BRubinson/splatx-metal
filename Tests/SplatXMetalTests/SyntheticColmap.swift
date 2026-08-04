// Small synthetic COLMAP bundle generator (Phase 6 round 2 Task 5 — material for the
// SplatXTrainer E2E tests). Creates in a temporary directory the on-disk layout that sx_train_run
// requires (sparse/0/{cameras,images,points3D}.bin + images/). The format follows the COLMAP
// binary spec (https://colmap.github.io/format.html — exactly the fields that ColmapLoader.cpp
// parses, little-endian). The images are gradient PNGs whose phase differs per view — what is
// under test is the **completion/stop/resume contract**, not training convergence quality, so
// arbitrary content suffices.
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

enum SyntheticColmap {
    static let width = 64
    static let height = 48

    /// Creates the bundle at root — returns root (used directly as data_dir; factor 1 → images/).
    @discardableResult
    static func makeBundle(at root: URL, cameraCount: Int = 3, pointCount: Int = 80) throws
        -> URL {
        let fm = FileManager.default
        let sparse = root.appendingPathComponent("sparse/0")
        let imagesDir = root.appendingPathComponent("images")
        try fm.createDirectory(at: sparse, withIntermediateDirectories: true)
        try fm.createDirectory(at: imagesDir, withIntermediateDirectories: true)

        // cameras.bin — one shared PINHOLE camera (id 1): fx=fy=50, principal point at the center
        var cams = Data()
        cams.appendLE(UInt64(1))
        cams.appendLE(Int32(1))          // camera_id
        cams.appendLE(Int32(1))          // model 1 = PINHOLE (params 4)
        cams.appendLE(UInt64(width))
        cams.appendLE(UInt64(height))
        for v in [50.0, 50.0, Double(width) / 2, Double(height) / 2] { cams.appendLE(v) }
        try cams.write(to: sparse.appendingPathComponent("cameras.bin"))

        // images.bin — identity-rotation cameras translated along x (the points lie inside every
        // view's frustum)
        var imgs = Data()
        imgs.appendLE(UInt64(cameraCount))
        for i in 0..<cameraCount {
            let name = String(format: "img%03d.png", i)
            imgs.appendLE(Int32(i + 1))                      // image_id
            for q in [1.0, 0.0, 0.0, 0.0] { imgs.appendLE(q) } // qvec w,x,y,z (identity)
            let tx = 0.15 * (Double(i) - Double(cameraCount - 1) / 2)
            for t in [tx, 0.0, 0.0] { imgs.appendLE(t) }     // tvec
            imgs.appendLE(Int32(1))                          // camera_id
            imgs.append(contentsOf: Array(name.utf8))        // name (NUL-terminated)
            imgs.append(0)
            imgs.appendLE(UInt64(0))                         // no points2D
            try writeGradientPNG(to: imagesDir.appendingPathComponent(name), phase: i)
        }
        try imgs.write(to: sparse.appendingPathComponent("images.bin"))

        // points3D.bin — a deterministic scatter in front of the cameras (z 3..5) (LCG —
        // identical across runs)
        var pts = Data()
        pts.appendLE(UInt64(pointCount))
        var rng = LCG(seed: 0x5147_5321)
        for i in 0..<pointCount {
            pts.appendLE(UInt64(i + 1))                      // point3D_id
            pts.appendLE(rng.uniform(-1.2, 1.2))             // x
            pts.appendLE(rng.uniform(-0.9, 0.9))             // y
            pts.appendLE(rng.uniform(3.0, 5.0))              // z
            pts.append(UInt8(rng.uniform(30, 225)))          // r,g,b
            pts.append(UInt8(rng.uniform(30, 225)))
            pts.append(UInt8(rng.uniform(30, 225)))
            pts.appendLE(Double(0))                          // reprojection error
            pts.appendLE(UInt64(0))                          // no track
        }
        try pts.write(to: sparse.appendingPathComponent("points3D.bin"))
        return root
    }

    /// An RGB gradient PNG whose phase differs per view (dims match the camera — passes the
    /// TrainC dims validation)
    private static func writeGradientPNG(to url: URL, phase: Int) throws {
        guard let cs = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(data: nil, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: 0, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let buf = ctx.data else {
            throw NSError(domain: "SyntheticColmap", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "CGContext creation failed"])
        }
        let stride = ctx.bytesPerRow
        let p = buf.bindMemory(to: UInt8.self, capacity: stride * height)
        for y in 0..<height {
            for x in 0..<width {
                let o = y * stride + x * 4
                p[o + 0] = UInt8((x * 4 + phase * 40) % 256)
                p[o + 1] = UInt8((y * 5 + phase * 20) % 256)
                p[o + 2] = UInt8((x + y + phase * 60) % 256)
                p[o + 3] = 255 // opaque — RGB unchanged even when premultiplied
            }
        }
        guard let img = ctx.makeImage(),
              let dest = CGImageDestinationCreateWithURL(url as CFURL,
                                                         UTType.png.identifier as CFString,
                                                         1, nil) else {
            throw NSError(domain: "SyntheticColmap", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "PNG encoder creation failed"])
        }
        CGImageDestinationAddImage(dest, img, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw NSError(domain: "SyntheticColmap", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "PNG write failed"])
        }
    }

    /// Deterministic LCG (glibc coefficients) — test reproducibility
    /// (SystemRandomNumberGenerator cannot be seeded)
    private struct LCG {
        var state: UInt64
        init(seed: UInt64) { state = seed }
        mutating func next() -> UInt64 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return state >> 33
        }
        mutating func uniform(_ lo: Double, _ hi: Double) -> Double {
            lo + (hi - lo) * (Double(next() % 1_000_000) / 1_000_000.0)
        }
    }
}

private extension Data {
    mutating func appendLE(_ v: UInt64) {
        Swift.withUnsafeBytes(of: v.littleEndian) { append(contentsOf: $0) }
    }
    mutating func appendLE(_ v: Int32) {
        Swift.withUnsafeBytes(of: v.littleEndian) { append(contentsOf: $0) }
    }
    mutating func appendLE(_ v: Double) {
        Swift.withUnsafeBytes(of: v.bitPattern.littleEndian) { append(contentsOf: $0) }
    }
}
