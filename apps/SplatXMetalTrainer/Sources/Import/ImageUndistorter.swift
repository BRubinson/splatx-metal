// import-undistortion Task 2: rectifying the pixels once the geometry is decided.
//
// ─────────────────────────────────────────────────────────────────────────────────────────────
// One map per camera, evaluated output → source
// ─────────────────────────────────────────────────────────────────────────────────────────────
// `CameraDistortion.swift` states the geometry and why it needs no iteration; this file is the
// pixel work that follows from it. For every pixel of the *output* frame the map records where in
// the source that pixel comes from:
//
//     output pixel (u', v')  → center (u'+0.5, v'+0.5)
//                            → (x, y)   = ((u'+0.5 − cx')/fx', (v'+0.5 − cy')/fy')
//                            → (xd, yd) = distortion.distort(x, y)          // forward, closed form
//                            → source     (fx·xd + cx, fy·yd + cy)
//                            → bilinear sample
//
// The map depends only on the camera, never on the image, so it is built once and reused for every
// photo that camera took — 110 images of one camera cost one map. It is stored as two `Float32`
// per output pixel (spec §6), ~65 MB for the reference 2137×3792 frame. Float32 resolves ~2·10⁻⁴
// source pixels at a 4000 px coordinate, three orders of magnitude below what bilinear can express.
//
// ─────────────────────────────────────────────────────────────────────────────────────────────
// The pixel-center convention is shared with the crop arithmetic — it is not a free choice
// ─────────────────────────────────────────────────────────────────────────────────────────────
// `CameraUndistortion.undistortedFrame` sized the output frame by sampling its boundary at pixel
// *centers*, 0.5 and W'−0.5 (spec footnote 6), precisely because that is where this resampler
// reads. The stored coordinates are therefore texel coordinates — the source pixel center
// `(u, v)` shifted to `(u−0.5, v−0.5)` so that an integral value lands exactly on a sample — and a
// bilinear tap set is complete for texel coordinates in `[0, W−1] × [0, H−1]`, i.e. source
// positions in `[0.5, W−0.5] × [0.5, H−0.5]`. That is the same interval `undistortedFrame` treats
// as the hard part of its 1 px margin. Using any other convention here would make the crop it
// computed meaningless.
//
// ─────────────────────────────────────────────────────────────────────────────────────────────
// Outside pixels are black, and counted
// ─────────────────────────────────────────────────────────────────────────────────────────────
// A correct crop leaves no output pixel sampling outside the source, so `Map.outsidePixels` should
// be 0. It is measured rather than assumed: a non-zero count means the frame arithmetic is wrong,
// and black borders are exactly the training-data contamination §4 of the spec exists to prevent.
// The count is a property of the map (constant per camera), so callers can report it before a
// single image is touched.
//
// ─────────────────────────────────────────────────────────────────────────────────────────────
// Inherited from ImagePrescaler (`Sources/Train/ImagePrescaler.swift` is the canonical statement)
// ─────────────────────────────────────────────────────────────────────────────────────────────
// - **File names are preserved verbatim, extension included.** The core assembles
//   "<images_dir>/<name>" from the names in COLMAP's images.bin, and CGImageSource decides the
//   format from the content, so JPEG bytes under a .png name decode fine.
// - **EXIF orientation is not applied.** The core loader does not apply it, so neither may we —
//   `CGImageSourceCreateImageAtIndex` returns the native pixel grid untransformed.
// - **JPEG quality 0.95**, the same as the prescale.
// - **Staged in a temporary folder and renamed on completion.** A half-rectified images/ would be
//   a dataset whose pixels and whose cameras.txt disagree. This applies to `undistortDirectory`,
//   the folder entry point below; the import path stages one level up instead, in the workspace
//   `ImportSheet` owns (that function's header spells the division out).
//
// Two deliberate differences from the prescale:
//
// - Failure **throws** instead of returning nil. The prescale has a fallback (the core can
//   downscale on load); rectification has none — an unrectified dataset is one the core refuses.
//   The import surfaces the error instead of silently producing a project that cannot train.
// - Samples are read through an sRGB bitmap context. The prescale hands the image to ImageIO
//   untouched and never inspects a pixel, so it can preserve any source color space; a resampler
//   has to address individual samples, which requires committing to a layout.
//
// ─────────────────────────────────────────────────────────────────────────────────────────────
// Why CPU
// ─────────────────────────────────────────────────────────────────────────────────────────────
// A GPU path would need a .metal asset, and those live in the repository's kernel directory, which
// this app round may not touch — plus the "eight places for a new kernel" checklist. Rows are
// split across cores with `DispatchQueue.concurrentPerform` instead. Images are processed one at a
// time (parallel *within* an image rather than across images) so peak memory stays at two frame
// buffers rather than one per core; at 8 MP that is the difference between ~65 MB and ~700 MB.
//
// Measured (M4, 10 cores, optimized build, the reference camera's 2160×3840 → 2136×3792, source
// JPEG 7.5 MB so the codec is not flattered by a synthetic image):
//
//     map build (once per camera)   9 ms
//     per image  decode 20 ms + resample 13 ms + encode 63 ms = 96 ms
//     110 images                    ~10.5 s
//
// So the resample is 13% of the work and the JPEG codec is the rest — exactly what spec §6
// predicted, and the whole job is already 3.5× faster than the 37 s colmap CLI took on the same
// dataset. Moving the gather to the GPU could recover at most those 13 ms per image. The
// measurement is what closes the question: no GPU.
import CoreGraphics
import Dispatch
import Foundation
import ImageIO
import UniformTypeIdentifiers

enum ImageUndistorter {

    struct UndistortError: LocalizedError, Equatable {
        let message: String
        var errorDescription: String? { message }
    }

    // MARK: - The map

    /// The output → source sampling map of one camera, plus the pinhole frame it produces.
    struct Map: Sendable {
        /// The rectified frame, straight from `CameraUndistortion.undistortedFrame`. This is what
        /// `cameras.txt` must declare — the core requires GT images to match the camera exactly.
        let width: Int
        let height: Int
        /// The rectified principal point: the middle of the solved frame, `W'/2, H'/2` — or the
        /// camera's own `cx, cy` when the distortion was the identity and nothing was solved
        /// (`CameraUndistortion.undistortedFrame`'s early return is the statement of that).
        let cx: Double
        let cy: Double
        /// The frame this map reads from. Images of other sizes are rejected rather than scaled.
        let sourceWidth: Int
        let sourceHeight: Int
        /// Output pixels whose bilinear neighbourhood is not entirely inside the source; those are
        /// written black. **Expected to be 0** — see the file header.
        let outsidePixels: Int
        /// Two `Float32` per output pixel, row major: the source texel coordinate `(u−0.5, v−0.5)`.
        let coordinates: [Float]
    }

    /// Builds the map for one camera. Pure arithmetic — no image is touched.
    static func makeMap(intrinsics: CameraIntrinsics,
                        distortion: CameraDistortion,
                        sourceWidth: Int,
                        sourceHeight: Int) -> Map {
        let frame = CameraUndistortion.undistortedFrame(intrinsics: intrinsics,
                                                        distortion: distortion,
                                                        width: sourceWidth,
                                                        height: sourceHeight)
        let outW = frame.width, outH = frame.height
        // The bounds are tested on the Float32 values the resampler will actually read, not on the
        // Double they were rounded from, so the count here and the black fill there agree exactly.
        let maxFU = Float(sourceWidth - 1), maxFV = Float(sourceHeight - 1)
        let bands = bandCount(rows: outH)

        // An identity distortion has an exactly representable map, and it has to be computed as one
        // rather than through the general expression (2026-08-15 FIX 2).
        //
        // `fx·((u + 0.5 − cx')/fx) + cx − 0.5` is algebraically `u + (cx − cx')` when the distortion
        // is the identity, and `cx' == cx` there, so the answer is `u`. Numerically it is not: the
        // divide and the multiply each round. Measured on `OPENCV 80 60, f = 90, c = (47, 25)`, the
        // `u = 0` column came out at −7·10⁻¹⁵. Anywhere else in this file that is nothing; here it is
        // *outside the source*, and the identity frame is the one case with no margin left to absorb
        // it — `undistortedFrame`'s early return keeps the full frame and spends none of the 1 px
        // `sourceMargin` the solver spends. `resample`'s bounds guard then painted the whole first
        // column black: 60 px of an 80×60 image, on a camera nothing was supposed to happen to.
        let identity = distortion.isIdentity
        let dx = intrinsics.cx - frame.cx, dy = intrinsics.cy - frame.cy

        var coordinates = [Float](repeating: 0, count: 2 * outW * outH)
        var perBandOutside = [Int](repeating: 0, count: bands)
        coordinates.withUnsafeMutableBufferPointer { coords in
            perBandOutside.withUnsafeMutableBufferPointer { outside in
                DispatchQueue.concurrentPerform(iterations: bands) { band in
                    var count = 0
                    for v in rowRange(band: band, bands: bands, rows: outH) {
                        let y = (Double(v) + 0.5 - frame.cy) / intrinsics.fy
                        var index = 2 * v * outW
                        for u in 0..<outW {
                            let fu: Float, fv: Float
                            if identity {
                                fu = Float(Double(u) + dx)
                                fv = Float(Double(v) + dy)
                            } else {
                                let x = (Double(u) + 0.5 - frame.cx) / intrinsics.fx
                                let (xd, yd) = distortion.distort(x, y)
                                fu = Float(intrinsics.fx * xd + intrinsics.cx - 0.5)
                                fv = Float(intrinsics.fy * yd + intrinsics.cy - 0.5)
                            }
                            if !(fu >= 0 && fu <= maxFU && fv >= 0 && fv <= maxFV) { count += 1 }
                            coords[index] = fu
                            coords[index + 1] = fv
                            index += 2
                        }
                    }
                    outside[band] = count
                }
            }
        }

        return Map(width: outW, height: outH, cx: frame.cx, cy: frame.cy,
                   sourceWidth: sourceWidth, sourceHeight: sourceHeight,
                   outsidePixels: perBandOutside.reduce(0, +),
                   coordinates: coordinates)
    }

    // MARK: - One image

    /// Rectifies a decoded image. The seam the pixel tests use — no file, no JPEG, no loss.
    static func rectify(_ image: CGImage, map: Map) throws -> CGImage {
        guard image.width == map.sourceWidth, image.height == map.sourceHeight else {
            throw UndistortError(message: "image is \(image.width)×\(image.height) but its camera "
                + "declares \(map.sourceWidth)×\(map.sourceHeight)")
        }
        let source = try makeContext(width: map.sourceWidth, height: map.sourceHeight)
        source.draw(image, in: CGRect(x: 0, y: 0,
                                      width: map.sourceWidth, height: map.sourceHeight))
        let destination = try makeContext(width: map.width, height: map.height)
        guard let sourceData = source.data, let destinationData = destination.data else {
            throw UndistortError(message: "bitmap context has no backing store")
        }
        resample(source: sourceData, sourceBytesPerRow: source.bytesPerRow,
                 destination: destinationData, destinationBytesPerRow: destination.bytesPerRow,
                 map: map)
        guard let out = destination.makeImage() else {
            throw UndistortError(message: "cannot materialize the rectified image")
        }
        return out
    }

    /// Decodes, rectifies and re-encodes one file. JPEG at quality 0.95, name chosen by the caller
    /// (the directory entry point below keeps the original one).
    static func undistort(source: URL, destination: URL, map: Map) throws {
        guard let imageSource = CGImageSourceCreateWithURL(source as CFURL, nil) else {
            throw UndistortError(message: "cannot open \(source.lastPathComponent)")
        }
        // No kCGImageSourceCreateThumbnail*: this is a full-resolution decode of the native pixel
        // grid, and no orientation transform is requested (see the file header).
        let options: [CFString: Any] = [kCGImageSourceShouldCacheImmediately: true]
        guard let image = CGImageSourceCreateImageAtIndex(imageSource, 0, options as CFDictionary)
        else {
            throw UndistortError(message: "cannot decode \(source.lastPathComponent)")
        }
        let rectified: CGImage
        do {
            rectified = try rectify(image, map: map)
        } catch let error as UndistortError {
            throw UndistortError(message: "\(source.lastPathComponent): \(error.message)")
        }
        guard let out = CGImageDestinationCreateWithURL(destination as CFURL,
                                                        UTType.jpeg.identifier as CFString,
                                                        1, nil) else {
            throw UndistortError(message: "cannot create \(destination.lastPathComponent)")
        }
        CGImageDestinationAddImage(out, rectified,
                                   [kCGImageDestinationLossyCompressionQuality: 0.95] as CFDictionary)
        guard CGImageDestinationFinalize(out) else {
            throw UndistortError(message: "cannot write \(destination.lastPathComponent)")
        }
    }

    // MARK: - A folder of images

    /// Rectifies every source into `targetDir`, which must not exist yet.
    ///
    /// **No production caller today, and the comment here used to claim credit for a guarantee it
    /// does not provide** (corrected in 2026-08-15 FIX 2). The import writes through
    /// `ColmapRectification.rectify`, which resamples straight into the destination it is handed;
    /// what keeps a failed import from leaving a half-rectified project behind is that the
    /// destination is the temporary workspace `ImportSheet` creates and removes, adopted into the
    /// projects root only once the whole assembly succeeded. That is the workspace's guarantee, not
    /// this function's.
    ///
    /// What this function does guarantee is its own: it assembles into a sibling `.building` folder
    /// and renames once complete, so *its* caller — one writing into a live directory — cannot be
    /// left with a partial set. On failure the staging folder is removed and the error is rethrown;
    /// there is no fallback, because an unrectified dataset is one the core refuses to train. It is
    /// kept because it is where the file-level behaviour is pinned by test: names and extensions
    /// preserved verbatim, JPEG bytes regardless of the name, output at exactly the map's frame.
    ///
    /// `progress` is called after each image with `(done, total)`, on the calling thread. Blocking
    /// for tens of seconds at 110 × 8 MP — call it off the main thread.
    @discardableResult
    static func undistortDirectory(sources: [URL],
                                   into targetDir: URL,
                                   map: Map,
                                   progress: (Int, Int) -> Void = { _, _ in }) throws -> URL {
        guard !sources.isEmpty else {
            throw UndistortError(message: "no images to rectify")
        }
        let fm = FileManager.default
        guard !fm.fileExists(atPath: targetDir.path) else {
            throw UndistortError(message: "\(targetDir.lastPathComponent) already exists")
        }
        let staging = targetDir.deletingLastPathComponent()
            .appendingPathComponent(targetDir.lastPathComponent + ".building", isDirectory: true)
        do {
            try? fm.removeItem(at: staging) // leftovers from an earlier crash
            try fm.createDirectory(at: staging, withIntermediateDirectories: true)
            let total = sources.count
            for (i, source) in sources.enumerated() {
                try autoreleasepool {
                    try undistort(source: source,
                                  destination: staging.appendingPathComponent(source.lastPathComponent),
                                  map: map)
                }
                progress(i + 1, total)
            }
            try fm.moveItem(at: staging, to: targetDir)
            return targetDir
        } catch {
            try? fm.removeItem(at: staging)
            throw error
        }
    }

    // MARK: - Pixels

    /// 8 bit per component, 32 bit per pixel, sRGB, components laid out R G B (unused) — the layout
    /// `resample` addresses directly. `bytesPerRow: 0` lets CoreGraphics pad rows for alignment,
    /// which is why every access below goes through the context's own `bytesPerRow`.
    private static func makeContext(width: Int, height: Int) throws -> CGContext {
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(data: nil, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: 0, space: space,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            throw UndistortError(message: "cannot create a \(width)×\(height) bitmap context")
        }
        return ctx
    }

    /// Bilinear gather along the map, rows split across cores. Output rows are disjoint, so the
    /// bands never touch the same destination byte.
    private static func resample(source: UnsafeMutableRawPointer,
                                 sourceBytesPerRow: Int,
                                 destination: UnsafeMutableRawPointer,
                                 destinationBytesPerRow: Int,
                                 map: Map) {
        let outW = map.width, outH = map.height
        let maxFU = Float(map.sourceWidth - 1), maxFV = Float(map.sourceHeight - 1)
        let lastX = map.sourceWidth - 1, lastY = map.sourceHeight - 1
        let src = source.assumingMemoryBound(to: UInt8.self)
        let dst = destination.assumingMemoryBound(to: UInt8.self)

        map.coordinates.withUnsafeBufferPointer { coords in
            let bands = bandCount(rows: outH)
            DispatchQueue.concurrentPerform(iterations: bands) { band in
                for v in rowRange(band: band, bands: bands, rows: outH) {
                    let dstRow = dst + v * destinationBytesPerRow
                    var index = 2 * v * outW
                    for u in 0..<outW {
                        let fu = coords[index], fv = coords[index + 1]
                        index += 2
                        let o = u * 4
                        // Outside the source: black, opaque. Counted at map build time.
                        guard fu >= 0, fu <= maxFU, fv >= 0, fv <= maxFV else {
                            dstRow[o] = 0; dstRow[o + 1] = 0; dstRow[o + 2] = 0; dstRow[o + 3] = 255
                            continue
                        }
                        // fu, fv >= 0, so truncation is floor.
                        let x0 = Int(fu), y0 = Int(fv)
                        let tx = fu - Float(x0), ty = fv - Float(y0)
                        // At the very last texel the second tap folds onto the first; its weight is
                        // 0 there, so the clamp changes no value.
                        let x1 = x0 < lastX ? x0 + 1 : x0
                        let y1 = y0 < lastY ? y0 + 1 : y0
                        let row0 = src + y0 * sourceBytesPerRow
                        let row1 = src + y1 * sourceBytesPerRow
                        let p00 = row0 + x0 * 4, p10 = row0 + x1 * 4
                        let p01 = row1 + x0 * 4, p11 = row1 + x1 * 4
                        for channel in 0..<3 {
                            let a = Float(p00[channel]), b = Float(p10[channel])
                            let c = Float(p01[channel]), d = Float(p11[channel])
                            let top = a + (b - a) * tx
                            let bottom = c + (d - c) * tx
                            let value = top + (bottom - top) * ty
                            dstRow[o + channel] = UInt8(min(255, max(0, value.rounded())))
                        }
                        dstRow[o + 3] = 255
                    }
                }
            }
        }
    }

    // MARK: - Row banding

    /// Enough bands to keep every core fed even when they finish unevenly, never more than one per
    /// row. Four per core is the usual compromise between load balance and dispatch overhead.
    private static func bandCount(rows: Int) -> Int {
        max(1, min(rows, ProcessInfo.processInfo.activeProcessorCount * 4))
    }

    /// Contiguous, disjoint, covering — the arithmetic distributes the remainder instead of piling
    /// it onto the last band.
    private static func rowRange(band: Int, bands: Int, rows: Int) -> Range<Int> {
        (rows * band / bands)..<(rows * (band + 1) / bands)
    }
}
