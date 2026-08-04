import Foundation
import ImageIO
import UniformTypeIdentifiers

enum ImportError: LocalizedError, Equatable {
    case notEnoughImages(found: Int)
    case missingSparse
    case incompleteColmapSet(missing: String)
    case missingImagesDir
    case videoUnreadable(String)

    var errorDescription: String? {
        switch self {
        case .notEnoughImages(let n): "At least 2 images are required (found \(n))."
        case .missingSparse: "No sparse/0 directory — not a COLMAP project."
        case .incompleteColmapSet(let f): "Incomplete COLMAP set: \(f) is missing in sparse/0."
        case .missingImagesDir: "No images/ directory next to sparse/."
        case .videoUnreadable(let why): "Cannot read video: \(why)"
        }
    }
}

/// Import validation and copying. It fills the project assembly directory (the assemble argument
/// of ProjectStore.create) with content per the C loader contract (sparse/0 + images/).
enum ImportService {
    static let imageExtensions: Set<String> = ["jpg", "jpeg", "png", "heic"]

    /// Supported extensions only, sorted lexicographically by file name — sorting by name is
    /// authoritative for the training indexing.
    static func listImages(in dir: URL) -> [URL] {
        let entries = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)) ?? []
        return entries
            .filter { imageExtensions.contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// Copies an image folder → dst/images/. When the EXIF orientation is not 1, it bakes the
    /// rotation into the pixels and re-encodes (the C loader's GT dims guard plus consistency with
    /// the PhotogrammetrySession intrinsics).
    static func assembleImages(from src: URL, into dst: URL) throws -> Int {
        let images = listImages(in: src)
        guard images.count >= 2 else { throw ImportError.notEnoughImages(found: images.count) }
        let imagesDir = dst.appendingPathComponent("images", isDirectory: true)
        try FileManager.default.createDirectory(at: imagesDir, withIntermediateDirectories: true)
        for url in images {
            let out = imagesDir.appendingPathComponent(url.lastPathComponent)
            if exifOrientation(url) <= 1 {
                try FileManager.default.copyItem(at: url, to: out)
            } else {
                try normalizeOrientation(from: url, to: out)
            }
        }
        return images.count
    }

    /// A COLMAP bundle → a complete sparse/0 set (bin preferred, txt when absent) plus copying
    /// images/.
    static func assembleColmap(from src: URL, into dst: URL) throws -> Int {
        let fm = FileManager.default
        let sparse = src.appendingPathComponent("sparse/0", isDirectory: true)
        guard fm.fileExists(atPath: sparse.path) else { throw ImportError.missingSparse }
        let sets = [["cameras.bin", "images.bin", "points3D.bin"],
                    ["cameras.txt", "images.txt", "points3D.txt"]]
        let exists = { (f: String) in fm.fileExists(atPath: sparse.appendingPathComponent(f).path) }
        guard let set = sets.first(where: { $0.allSatisfy(exists) }) else {
            // Report the first missing file of whichever side has a partial set (cameras.bin when
            // neither exists)
            let started = sets.first { $0.contains(where: exists) } ?? sets[0]
            let missing = started.first { !exists($0) } ?? "cameras.bin"
            throw ImportError.incompleteColmapSet(missing: missing)
        }
        let srcImages = src.appendingPathComponent("images", isDirectory: true)
        guard fm.fileExists(atPath: srcImages.path) else { throw ImportError.missingImagesDir }
        let count = listImages(in: srcImages).count
        guard count >= 2 else { throw ImportError.notEnoughImages(found: count) }
        let dstSparse = dst.appendingPathComponent("sparse/0", isDirectory: true)
        try fm.createDirectory(at: dstSparse, withIntermediateDirectories: true)
        for f in set {
            try fm.copyItem(at: sparse.appendingPathComponent(f),
                            to: dstSparse.appendingPathComponent(f))
        }
        try fm.copyItem(at: srcImages, to: dst.appendingPathComponent("images", isDirectory: true))
        return count
    }

    // MARK: EXIF orientation

    private static func exifOrientation(_ url: URL) -> Int {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let o = props[kCGImagePropertyOrientation] as? Int else { return 1 }
        return o
    }

    /// Bakes the orientation into the pixels: it creates a "thumbnail" at the original maximum
    /// side so the CGImageSource thumbnail API decodes with the rotation applied, then re-saves it
    /// as JPEG.
    private static func normalizeOrientation(from url: URL, to out: URL) throws {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? Int,
              let h = props[kCGImagePropertyPixelHeight] as? Int,
              let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, [
                  kCGImageSourceCreateThumbnailFromImageAlways: true,
                  kCGImageSourceCreateThumbnailWithTransform: true,   // applies the rotation
                  kCGImageSourceThumbnailMaxPixelSize: max(w, h),
              ] as CFDictionary),
              let dest = CGImageDestinationCreateWithURL(
                  out as CFURL, UTType.jpeg.identifier as CFString, 1, nil)
        else { throw ImportError.videoUnreadable(url.lastPathComponent) }
        CGImageDestinationAddImage(dest, cg, [kCGImageDestinationLossyCompressionQuality: 0.95] as CFDictionary)
        guard CGImageDestinationFinalize(dest) else {
            throw ImportError.videoUnreadable(url.lastPathComponent)
        }
    }
}
