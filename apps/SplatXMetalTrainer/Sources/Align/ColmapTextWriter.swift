import Foundation

struct AlignedFrame {
    var name: String       // the file name inside images/ (no spaces — the loader's 10-token contract)
    var width: Int
    var height: Int
    var fx: Double, fy: Double, cx: Double, cy: Double   // in pixels, relative to the source resolution
    var pose: ColmapPose
}

struct AlignedPoint {
    var x: Double, y: Double, z: Double
    var r: UInt8, g: UInt8, b: UInt8
}

enum ColmapWriteError: LocalizedError {
    case noFrames
    case noPoints
    case nameContainsSpace(String)
    var errorDescription: String? {
        switch self {
        case .noFrames: "Alignment produced no camera poses."
        case .noPoints: "Alignment produced an empty point cloud."
        case .nameContainsSpace(let n): "Image name contains a space: \(n)"
        }
    }
}

/// The COLMAP text bundle writer. It conforms to the C loader contract (ColmapLoader.cpp):
/// PINHOLE only, images.txt has [a 10-token pose line + an empty POINTS2D line] per image,
/// points3D is ID X Y Z R G B ERROR (no tracks), and frames are written sorted by name.
enum ColmapTextWriter {
    // Sorting by name is authoritative — camerasText and imagesText MUST share the same sort for
    // the CAMERA_ID (row number) mapping to agree across the two files.
    private static func sortedByName(_ frames: [AlignedFrame]) -> [AlignedFrame] {
        frames.sorted { $0.name < $1.name }
    }

    static func camerasText(frames: [AlignedFrame]) -> String {
        var out = "# Camera list: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n"
        for (i, f) in sortedByName(frames).enumerated() {
            out += "\(i + 1) PINHOLE \(f.width) \(f.height) "
                 + "\(fmt(f.fx)) \(fmt(f.fy)) \(fmt(f.cx)) \(fmt(f.cy))\n"
        }
        return out
    }

    static func imagesText(frames: [AlignedFrame]) -> String {
        var out = "# Image list: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
        for (i, f) in sortedByName(frames).enumerated() {
            let p = f.pose
            out += "\(i + 1) \(fmt(p.qw)) \(fmt(p.qx)) \(fmt(p.qy)) \(fmt(p.qz)) "
                 + "\(fmt(p.tx)) \(fmt(p.ty)) \(fmt(p.tz)) \(i + 1) \(f.name)\n\n"
        }
        return out
    }

    static func points3DText(points: [AlignedPoint]) -> String {
        var out = "# 3D point list: POINT3D_ID X Y Z R G B ERROR TRACK[]\n"
        for (i, p) in points.enumerated() {
            out += "\(i + 1) \(fmt(p.x)) \(fmt(p.y)) \(fmt(p.z)) \(p.r) \(p.g) \(p.b) 0\n"
        }
        return out
    }

    static func write(frames rawFrames: [AlignedFrame], points: [AlignedPoint], to sparse0: URL) throws {
        guard !rawFrames.isEmpty else { throw ColmapWriteError.noFrames }
        guard !points.isEmpty else { throw ColmapWriteError.noPoints }
        if let bad = rawFrames.first(where: { $0.name.contains(" ") }) {
            throw ColmapWriteError.nameContainsSpace(bad.name)
        }
        let frames = rawFrames.sorted { $0.name < $1.name }
        try FileManager.default.createDirectory(at: sparse0, withIntermediateDirectories: true)
        try camerasText(frames: frames).write(
            to: sparse0.appendingPathComponent("cameras.txt"), atomically: true, encoding: .utf8)
        try imagesText(frames: frames).write(
            to: sparse0.appendingPathComponent("images.txt"), atomically: true, encoding: .utf8)
        try points3DText(points: points).write(
            to: sparse0.appendingPathComponent("points3D.txt"), atomically: true, encoding: .utf8)
    }

    private static func fmt(_ v: Double) -> String { String(format: "%.9g", v) }
}
