import Foundation
import AVFoundation
import CoreImage
import UniformTypeIdentifiers

/// Video → sharp frame extraction. Two passes:
/// ① score the Laplacian variance of every frame at low resolution (~320px wide) in grayscale
/// ② re-decode only the best-in-window selected indices at source resolution in BGRA and save as
///    JPEG.
/// The frame file names are frame_%05d.jpg — the zero padding guarantees that lexicographic order
/// equals chronological order (sorting by name is authoritative for the training indexing).
enum VideoFrameExtractor {
    static func extract(video url: URL, targetFrames: Int, into imagesDir: URL,
                        progress: @escaping @Sendable (Double) -> Void) async throws -> Int {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw ImportError.videoUnreadable("no video track")
        }
        // pass 1: low-resolution grayscale scoring
        let scores = try scorePass(asset: asset, track: track) { progress($0 * 0.5) }
        guard scores.count >= 2 else { throw ImportError.videoUnreadable("fewer than 2 frames") }
        let picked = Set(SharpnessScorer.selectBestPerWindow(scores: scores, windowCount: targetFrames))
        // pass 2: save only the selected indices at source resolution
        try FileManager.default.createDirectory(at: imagesDir, withIntermediateDirectories: true)
        let saved = try savePass(asset: asset, track: track, picked: picked, into: imagesDir) {
            progress(0.5 + $0 * 0.5)
        }
        return saved
    }

    private static func scorePass(asset: AVURLAsset, track: AVAssetTrack,
                                  progress: (Double) -> Void) throws -> [Double] {
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_OneComponent8,
            kCVPixelBufferWidthKey as String: 320,
            kCVPixelBufferHeightKey as String: 180,
        ])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        reader.startReading()
        var scores: [Double] = []
        while let sample = output.copyNextSampleBuffer() {
            guard let buf = CMSampleBufferGetImageBuffer(sample) else { continue }
            CVPixelBufferLockBaseAddress(buf, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
            let w = CVPixelBufferGetWidth(buf), h = CVPixelBufferGetHeight(buf)
            let stride = CVPixelBufferGetBytesPerRow(buf)
            guard let base = CVPixelBufferGetBaseAddress(buf) else { continue }
            var gray = [UInt8](repeating: 0, count: w * h)
            for y in 0..<h {
                memcpy(&gray[y * w], base.advanced(by: y * stride), w)
            }
            scores.append(SharpnessScorer.laplacianVariance(gray: gray, width: w, height: h))
            if scores.count % 60 == 0 { progress(min(0.95, Double(scores.count) / 3000.0)) }
        }
        guard reader.status == .completed else {
            throw ImportError.videoUnreadable("decode failed (\(reader.status.rawValue))")
        }
        return scores
    }

    private static func savePass(asset: AVURLAsset, track: AVAssetTrack, picked: Set<Int>,
                                 into imagesDir: URL, progress: (Double) -> Void) throws -> Int {
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        reader.startReading()
        let ci = CIContext()
        var index = 0, saved = 0
        while let sample = output.copyNextSampleBuffer() {
            defer { index += 1 }
            guard picked.contains(index), let buf = CMSampleBufferGetImageBuffer(sample) else { continue }
            let image = CIImage(cvPixelBuffer: buf)
            guard let cg = ci.createCGImage(image, from: image.extent) else { continue }
            let out = imagesDir.appendingPathComponent(String(format: "frame_%05d.jpg", index))
            guard let dest = CGImageDestinationCreateWithURL(
                out as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else { continue }
            CGImageDestinationAddImage(dest, cg,
                [kCGImageDestinationLossyCompressionQuality: 0.95] as CFDictionary)
            if CGImageDestinationFinalize(dest) { saved += 1 }
            progress(min(1.0, Double(saved) / Double(max(picked.count, 1))))
        }
        guard reader.status == .completed else {
            throw ImportError.videoUnreadable("decode failed (\(reader.status.rawValue))")
        }
        return saved
    }
}
