// import-undistortion Task 3: synthetic COLMAP models for the import tests.
//
// The import now *reads* `sparse/0` before it decides what to do with it (`ColmapRectification`),
// so a fixture whose `cameras.txt` holds the word "PINHOLE" is no longer a stand-in for a model —
// it is a malformed file, and the import says so. This builder writes the real thing in both
// encodings, small enough to stay inline in a test.
//
// It writes bytes the *production* reader parses, so a round trip through it still has
// discriminating power: nothing here shares code with `ColmapSparseModel`, and the binary layouts
// are transcribed from the format the core reads (`ColmapLoader.cpp` `load_colmap_bundle_bin`),
// not from the Swift reader under test.
//
// Everything is little-endian, which is what COLMAP's readers assume and what arm64 writes natively.
import Foundation
@testable import SplatXMetalTrainer

enum ColmapFixture {

    /// A distortion-free camera — the early-exit case, and the fixture most tests want when the
    /// model is scenery rather than the subject.
    static let pinholeCamera = ColmapCameraRecord(id: 1, modelId: 1, width: 640, height: 480,
                                                  params: [500, 500, 320, 240])

    /// `cameras.txt` holding one pinhole camera. The one-liner the older fixtures wanted.
    static var pinholeCamerasText: String { camerasText([pinholeCamera]) }

    /// `cameras.bin` holding one pinhole camera.
    static var pinholeCamerasBin: Data { camerasBin([pinholeCamera]) }

    // MARK: Text

    static func camerasText(_ cameras: [ColmapCameraRecord]) -> String {
        var out = "# Camera list: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n"
        for c in cameras {
            let name = ColmapCameraModel(rawValue: c.modelId)?.name ?? "PINHOLE"
            out += "\(c.id) \(name) \(c.width) \(c.height) "
                + c.params.map(fmt).joined(separator: " ") + "\n"
        }
        return out
    }

    /// Pose line + an **empty** POINTS2D line, the shape COLMAP writes when there are no
    /// observations to record and the shape the app's own writer emits.
    static func imagesText(_ images: [ColmapImageRecord]) -> String {
        var out = "# Image list: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
        for i in images {
            out += "\(i.id) \(fmt(i.qw)) \(fmt(i.qx)) \(fmt(i.qy)) \(fmt(i.qz)) "
                + "\(fmt(i.tx)) \(fmt(i.ty)) \(fmt(i.tz)) \(i.cameraId) \(i.name)\n\n"
        }
        return out
    }

    static func pointsText(_ points: [ColmapPointRecord]) -> String {
        var out = "# 3D point list: POINT3D_ID X Y Z R G B ERROR TRACK[]\n"
        for (index, p) in points.enumerated() {
            out += "\(index + 1) \(fmt(p.x)) \(fmt(p.y)) \(fmt(p.z)) \(p.r) \(p.g) \(p.b) 0.5\n"
        }
        return out
    }

    // MARK: Binary

    static func camerasBin(_ cameras: [ColmapCameraRecord]) -> Data {
        var data = Data()
        append(UInt64(cameras.count), to: &data)
        for c in cameras {
            append(Int32(c.id), to: &data)
            append(Int32(c.modelId), to: &data)
            append(UInt64(c.width), to: &data)
            append(UInt64(c.height), to: &data)
            for p in c.params { append(p, to: &data) }
        }
        return data
    }

    /// `npoints2D` is written as 0, so no observation block follows — the reader has to step over a
    /// zero-length block correctly to find the next image.
    static func imagesBin(_ images: [ColmapImageRecord]) -> Data {
        var data = Data()
        append(UInt64(images.count), to: &data)
        for i in images {
            append(Int32(i.id), to: &data)
            for v in [i.qw, i.qx, i.qy, i.qz, i.tx, i.ty, i.tz] { append(v, to: &data) }
            append(Int32(i.cameraId), to: &data)
            data.append(contentsOf: Array(i.name.utf8))
            data.append(0)
            append(UInt64(0), to: &data)
        }
        return data
    }

    static func pointsBin(_ points: [ColmapPointRecord]) -> Data {
        var data = Data()
        append(UInt64(points.count), to: &data)
        for (index, p) in points.enumerated() {
            append(UInt64(index + 1), to: &data)
            append(p.x, to: &data); append(p.y, to: &data); append(p.z, to: &data)
            data.append(p.r); data.append(p.g); data.append(p.b)
            append(Double(0.5), to: &data)     // reprojection error
            append(UInt64(0), to: &data)       // empty track
        }
        return data
    }

    // MARK: Writing a whole model

    /// Writes the three files into `sparse0`, in either encoding.
    static func write(cameras: [ColmapCameraRecord],
                      images: [ColmapImageRecord],
                      points: [ColmapPointRecord],
                      to sparse0: URL,
                      binary: Bool) throws {
        try FileManager.default.createDirectory(at: sparse0, withIntermediateDirectories: true)
        let files: [(String, Data)] = binary
            ? [("cameras.bin", camerasBin(cameras)),
               ("images.bin", imagesBin(images)),
               ("points3D.bin", pointsBin(points))]
            : [("cameras.txt", Data(camerasText(cameras).utf8)),
               ("images.txt", Data(imagesText(images).utf8)),
               ("points3D.txt", Data(pointsText(points).utf8))]
        for (name, data) in files {
            try data.write(to: sparse0.appendingPathComponent(name))
        }
    }

    // MARK: Plumbing

    /// 17 significant digits, so a text fixture's poses survive the round trip exactly and an
    /// equality assertion means what it says.
    private static func fmt(_ v: Double) -> String { String(format: "%.17g", v) }

    private static func append<T>(_ value: T, to data: inout Data) {
        withUnsafeBytes(of: value) { data.append(contentsOf: $0) }
    }
}
