import Foundation
import RealityKit
import ImageIO

enum AlignError: LocalizedError {
    case notSupported
    case intrinsicsUnavailable(image: String)
    case tooFewPoses(got: Int)
    case emptyPointCloud
    case session(String)

    var errorDescription: String? {
        switch self {
        case .notSupported:
            "Photogrammetry is not supported on this Mac."
        case .intrinsicsUnavailable(let name):
            "No camera intrinsics for \(name) — macOS 26 Pose.intrinsics unavailable for this input."
        case .tooFewPoses(let n):
            "Alignment registered only \(n) image(s) — need at least 2. Add more overlap."
        case .emptyPointCloud:
            "Alignment produced an empty point cloud."
        case .session(let why):
            "Photogrammetry session failed: \(why)"
        }
    }
}

struct AlignResultData {
    var frames: [AlignedFrame]
    var points: [AlignedPoint]
    var skippedImages: Int   // the count excluded because the URL or dimensions could not be resolved, despite a pose being produced
    var seconds: Double
}

/// The PhotogrammetrySession wrapper. It requests only .poses and .pointCloud, with no mesh
/// (spec §6).
/// UI-independent — the Task 9 harness compiles it standalone together with PoseConversion and
/// ColmapTextWriter.
final class AlignEngine {
    static func isAvailable() -> Bool { PhotogrammetrySession.isSupported }

    static func run(imagesDir: URL, sequentialHint: Bool,
                    progress: @escaping @Sendable (Double) -> Void) async throws -> AlignResultData {
        guard PhotogrammetrySession.isSupported else { throw AlignError.notSupported }
        let t0 = Date()
        var config = PhotogrammetrySession.Configuration()
        config.isObjectMaskingEnabled = false        // the whole scene — preventing automatic object masking
        config.ignoreBoundingBox = true
        if sequentialHint { config.sampleOrdering = .sequential }   // speeds up video frames
        let session = try PhotogrammetrySession(input: imagesDir, configuration: config)
        try session.process(requests: [.poses, .pointCloud])

        var poses: PhotogrammetrySession.Poses?
        var cloud: PhotogrammetrySession.PointCloud?
        outputs: for try await output in session.outputs {
            try Task.checkCancellation()
            switch output {
            case .requestProgress(_, let fraction):
                progress(fraction)
            case .requestComplete(_, let result):
                switch result {
                case .poses(let p): poses = p
                case .pointCloud(let c): cloud = c
                default: break
                }
                // The outputs stream does not end on its own even after processingComplete
                // (measured in Task 9: an infinite 55-minute wait after processing finished —
                // Apple's sample also exits at completion).
                // As soon as both requested results are in, it breaks out of the loop.
                if poses != nil, cloud != nil { break outputs }
            case .requestError(_, let error):
                throw AlignError.session("\(error)")
            default:
                break
            }
        }
        guard let poses else { throw AlignError.session("no poses output") }
        guard let cloud, !cloud.points.isEmpty else { throw AlignError.emptyPointCloud }

        var frames: [AlignedFrame] = []
        var skipped = 0
        for (sampleID, pose) in poses.posesBySample {
            guard let url = poses.urlsBySample[sampleID] else { skipped += 1; continue }
            guard let k = pose.intrinsics else {
                throw AlignError.intrinsicsUnavailable(image: url.lastPathComponent)
            }
            guard let (w, h) = Self.imageDimensions(url) else { skipped += 1; continue }
            // The fx/fy/cx/cy mapping was verified by the Task 9 garden measurements (fx=fy≈962, cx≈W/2, cy≈H/2 → eval PSNR 18.4 dB).
            frames.append(AlignedFrame(
                name: url.lastPathComponent, width: w, height: h,
                fx: Double(k.columns.0.x), fy: Double(k.columns.1.y),
                cx: Double(k.columns.2.x), cy: Double(k.columns.2.y),
                pose: PoseConversion.colmapFromRealityKit(
                    rotation: pose.rotation, translation: pose.translation)))
        }
        guard frames.count >= 2 else { throw AlignError.tooFewPoses(got: frames.count) }
        let points = cloud.points.map {
            AlignedPoint(x: Double($0.position.x), y: Double($0.position.y), z: Double($0.position.z),
                         r: $0.color.x, g: $0.color.y, b: $0.color.z)
        }
        return AlignResultData(frames: frames, points: points, skippedImages: skipped,
                               seconds: Date().timeIntervalSince(t0))
    }

    private static func imageDimensions(_ url: URL) -> (Int, Int)? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? Int,
              let h = props[kCGImagePropertyPixelHeight] as? Int else { return nil }
        return (w, h)
    }
}
