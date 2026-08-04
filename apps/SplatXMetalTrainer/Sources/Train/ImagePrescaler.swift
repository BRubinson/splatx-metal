// Training speed Task 3: creating the images_<factor>/ prescale before training starts.
//
// Background (the 2026-07-21 training speed diagnosis): the app's project bundle has no
// images_<factor>/, so the core falls back to downscale-on-load (decoding the full-resolution
// original and shrinking it on every step). The core consumes an images_<factor>/ folder in
// preference when it exists (TrainingInput.h resolve_images_dir), so creating it once before
// training makes the fallback structurally disappear.
//
// Consistency with the core contract (all of it measured evidence — violating it kills training
// with a throw):
// - File names are kept exactly as the originals (including the extension): the core assembles
//   "<images_dir>/<name>" from the names in COLMAP's images.bin (TrainingInput.cpp:36), and
//   CGImageSource determines the format from the content (JPEG data under a .png name decodes fine
//   — ImageIO.h, "automatic format detection"), so preserving the names without unifying the
//   extensions is safe.
// - The output dimensions are exactly (lround(W/factor), lround(H/factor)): the core's GT dims
//   guard (TrainC.cpp decode_gt) requires an exact match with the camera dimensions. The rounding
//   of the short axis by an ImageIO thumbnail is not guaranteed, so the dimensions are verified
//   after the thumbnail and redrawn at exact dimensions with a CGContext when they differ.
// - EXIF orientation is not applied: the core loader does not apply orientation (ImageIO.h), so
//   kCGImageSourceCreateThumbnailWithTransform is not enabled (preserving the native
//   orientation).
// - resolve_images_dir selects on folder "existence" alone (there is no completeness check) — if a
//   partially generated folder is left behind, the core picks it and throws on a missing file. So
//   a fresh generation is completed in a temporary folder and then renamed, and on failure the
//   incomplete folder is deleted and nil is returned (training continues with the previous
//   downscale-on-load fallback — spec 2026-07-21 decision ①).
import Foundation
import ImageIO
import UniformTypeIdentifiers

enum ImagePrescaler {

    // MARK: Pure logic (the test surface)

    /// A mirror of the core's downscaled_dim (ImageIO.h): lround(v/factor).
    /// C's lround is half-away-from-zero — identical to Swift's default .rounded().
    static func expectedDimension(_ v: Int, factor: Int) -> Int {
        Int((Double(v) / Double(factor)).rounded())
    }

    /// Decides which image extensions the core loader (CGImageSource) handles. A conservative set:
    /// the COLMAP dataset convention (jpg/png) plus the formats CGImageSource supports —
    /// case-insensitive.
    static func isImageFile(_ name: String) -> Bool {
        let ext = (name as NSString).pathExtension.lowercased()
        return ["jpg", "jpeg", "png", "heic", "heif", "tiff", "tif", "bmp"].contains(ext)
    }

    /// The completeness check: whether each source name exists in the target folder (it returns
    /// the missing list — an empty array = complete).
    /// A name-by-name comparison, stronger than matching file counts — because the core assembles
    /// paths from names.
    static func missingNames(sources: [String], existing: Set<String>) -> [String] {
        sources.filter { !existing.contains($0) }
    }

    // MARK: Entry point

    /// Creates images_<factor>/ as a sibling of imagesDir ("<dataset>/images").
    /// - factor <= 1: a no-op (nil — the core uses images/ as-is).
    /// - Already complete (every source name present): a no-op (it returns the existing folder
    ///   URL).
    /// - Partially present: it fills in only the missing files (a top-up).
    /// - Failure (decoding, disk, and so on): it deletes the incomplete folder and returns nil plus
    ///   a log — it does not throw.
    ///   Training continues with the previous downscale-on-load fallback.
    /// Synchronous and blocking for seconds to tens of seconds — the caller invokes it from a
    /// background context (TrainViewModel).
    @discardableResult
    static func prescaleIfNeeded(imagesDir: URL, factor: Int,
                                 progress: (Int, Int) -> Void = { _, _ in },
                                 log: (String) -> Void = { _ in }) -> URL? {
        guard factor > 1 else { return nil }
        let fm = FileManager.default
        let parent = imagesDir.deletingLastPathComponent()
        let targetDir = parent.appendingPathComponent("images_\(factor)", isDirectory: true)

        guard let sources = try? fm.contentsOfDirectory(at: imagesDir,
                                                        includingPropertiesForKeys: nil,
                                                        options: [.skipsHiddenFiles]) else {
            log("images/ not readable — skipping prescale (core will handle)")
            return nil
        }
        let imageSources = sources.filter { isImageFile($0.lastPathComponent) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !imageSources.isEmpty else {
            log("images/ has no images — skipping prescale")
            return nil
        }

        var isDir: ObjCBool = false
        if fm.fileExists(atPath: targetDir.path, isDirectory: &isDir), isDir.boolValue {
            // An existing folder: a no-op when complete, otherwise it fills in only what is
            // missing. On failure the incomplete folder MUST be deleted for the core fallback to
            // hold (it is selected on existence alone — see the comment at the top).
            let existing = Set((try? fm.contentsOfDirectory(atPath: targetDir.path)) ?? [])
            let missing = missingNames(sources: imageSources.map(\.lastPathComponent),
                                       existing: existing)
            if missing.isEmpty { return targetDir }
            let missingSet = Set(missing)
            let todo = imageSources.filter { missingSet.contains($0.lastPathComponent) }
            do {
                try generate(sources: todo, into: targetDir, factor: factor, progress: progress)
                return targetDir
            } catch {
                log("prescale failed (\(error.localizedDescription)) — removing incomplete images_\(factor), training falls back to on-load downscale")
                try? fm.removeItem(at: targetDir)
                return nil
            }
        }

        // A fresh generation: completed in a temporary folder and then renamed — a crash partway
        // through leaves no incomplete images_<factor>/ (resolve_images_dir ignores the temporary
        // name).
        let stagingDir = parent.appendingPathComponent("images_\(factor).building", isDirectory: true)
        do {
            try? fm.removeItem(at: stagingDir) // remove leftovers from an earlier crash
            try fm.createDirectory(at: stagingDir, withIntermediateDirectories: true)
            try generate(sources: imageSources, into: stagingDir, factor: factor, progress: progress)
            try fm.moveItem(at: stagingDir, to: targetDir)
            return targetDir
        } catch {
            log("prescale failed (\(error.localizedDescription)) — training falls back to on-load downscale")
            try? fm.removeItem(at: stagingDir)
            return nil
        }
    }

    // MARK: Generation

    private struct PrescaleError: LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    private static func generate(sources: [URL], into dir: URL, factor: Int,
                                 progress: (Int, Int) -> Void) throws {
        let total = sources.count
        for (i, src) in sources.enumerated() {
            try autoreleasepool {
                try downscaleOne(source: src,
                                 dest: dir.appendingPathComponent(src.lastPathComponent),
                                 factor: factor)
            }
            progress(i + 1, total)
        }
    }

    private static func downscaleOne(source: URL, dest: URL, factor: Int) throws {
        guard let src = CGImageSourceCreateWithURL(source as CFURL, nil) else {
            throw PrescaleError(message: "cannot open \(source.lastPathComponent)")
        }
        // The native pixel dimensions (EXIF orientation is not applied — the same orientation
        // convention as the core loader).
        guard let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? Int,
              let h = props[kCGImagePropertyPixelHeight] as? Int, w > 0, h > 0 else {
            throw PrescaleError(message: "cannot read dims of \(source.lastPathComponent)")
        }
        let eW = expectedDimension(w, factor: factor)
        let eH = expectedDimension(h, factor: factor)
        guard eW >= 1, eH >= 1 else {
            throw PrescaleError(message: "factor \(factor) too large for \(w)x\(h)")
        }
        let opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true, // ignore the embedded EXIF thumbnail (high-quality resample)
            kCGImageSourceThumbnailMaxPixelSize: max(eW, eH),
            kCGImageSourceShouldCacheImmediately: true,
            // kCGImageSourceCreateThumbnailWithTransform is unspecified (false) — EXIF rotation is not applied (see the file header comment).
        ]
        guard var image = CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary) else {
            throw PrescaleError(message: "cannot decode \(source.lastPathComponent)")
        }
        if image.width != eW || image.height != eH {
            // The short-axis rounding disagreed with the core's lround convention (rare) — redraw
            // at exact dimensions.
            image = try exactResize(image, width: eW, height: eH,
                                    name: source.lastPathComponent)
        }
        guard let dst = CGImageDestinationCreateWithURL(dest as CFURL,
                                                        UTType.jpeg.identifier as CFString, 1, nil)
        else { throw PrescaleError(message: "cannot create \(dest.lastPathComponent)") }
        let jpegOpts: [CFString: Any] = [kCGImageDestinationLossyCompressionQuality: 0.95]
        CGImageDestinationAddImage(dst, image, jpegOpts as CFDictionary)
        guard CGImageDestinationFinalize(dst) else {
            throw PrescaleError(message: "cannot write \(dest.lastPathComponent)")
        }
    }

    private static func exactResize(_ image: CGImage, width: Int, height: Int,
                                    name: String) throws -> CGImage {
        guard let ctx = CGContext(data: nil, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw PrescaleError(message: "resize context failed for \(name)")
        }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let out = ctx.makeImage() else {
            throw PrescaleError(message: "resize failed for \(name)")
        }
        return out
    }
}
