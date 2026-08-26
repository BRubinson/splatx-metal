// import-undistortion Task 3: reading a COLMAP model exactly as it is, and writing back the
// pinhole model rectification produces.
//
// ─────────────────────────────────────────────────────────────────────────────────────────────
// Why a second COLMAP reader exists
// ─────────────────────────────────────────────────────────────────────────────────────────────
// `Sources/Viewer/ColmapCameraReader.swift` already parses `sparse/0`, and it deliberately cannot
// be reused here. It resolves every camera to `fx, fy, cx, cy` and **throws on any model but
// PINHOLE / SIMPLE_PINHOLE** — so it refuses to open precisely the models this path exists to fix —
// and its `loadPoints` returns `[SIMD3<Float>]`, dropping the per-point colour. That colour is not
// decoration: the core seeds every Gaussian's SH DC term from it (`ColmapLoader.cpp:246` for txt and
// `:304` for bin both fill `p.rgb`, and `TrainC.cpp` hands those bytes to the initializer). Rewriting
// `points3D` through a positions-only reader would leave a project whose points are all one colour
// and whose training merely starts from a worse place — a quality loss with no error message.
//
// So this file reads what the viewer's reader discards: the raw model id and its parameter vector,
// the pose exactly as stored, and RGB.
//
// ─────────────────────────────────────────────────────────────────────────────────────────────
// bin or txt is the caller's decision, never `fileExists` here
// ─────────────────────────────────────────────────────────────────────────────────────────────
// Both import paths have already decided which complete triple they are working with —
// `ImportService.assembleColmap` by probing, `assembleCaptureBundle` through
// `CaptureBundle.completeColmapSet`, which matches names out of a directory *listing* on purpose
// (its header records why: `FileManager.fileExists` finds `Cameras.bin` for `cameras.bin` on a
// case-insensitive APFS volume, a listing comparison never does). Re-deciding here with a second
// probe would reintroduce exactly that disagreement, so the chosen set is passed in.
//
// ─────────────────────────────────────────────────────────────────────────────────────────────
// The output is always all three files as text, and the input .bin is never copied
// ─────────────────────────────────────────────────────────────────────────────────────────────
// `ColmapLoader.cpp:325` picks the format from **`cameras.bin`'s existence alone** and then reads
// all three files in that format. A rectified `cameras.txt` sitting next to the original
// `points3D.bin` is therefore not a mixture the loader tolerates — it is a loader that goes looking
// for `points3D.txt`. And keeping the original `cameras.bin` would be worse than a missing file: the
// stale distorted camera would win over the rectified text and train silently wrong. Hence all three
// out as text, and none of the input binaries copied.
//
// ─────────────────────────────────────────────────────────────────────────────────────────────
// %.17g, where `ColmapTextWriter` uses %.9g
// ─────────────────────────────────────────────────────────────────────────────────────────────
// `ColmapTextWriter` serializes a model this app *computed* — nine significant digits are more than
// its inputs carry. This writer serializes a model the **user** solved, and a bin → txt conversion
// is the one step where their numbers could quietly lose precision. Undistortion does not move a
// camera (spec §5.1), so the poses that come out have to be the poses that went in; 17 significant
// digits is the shortest field width that round-trips every `Double` exactly, and the import tests
// assert equality rather than approximate equality on the strength of it.
import Foundation

// MARK: - Records

/// A `cameras.{bin,txt}` row **unresolved**: the model id and its parameter vector as stored, with
/// no interpretation. `CameraUndistortion.parse` is what turns it into intrinsics + distortion.
struct ColmapCameraRecord: Equatable, Sendable {
    var id: Int
    var modelId: Int
    var width: Int
    var height: Int
    var params: [Double]

    /// True for the two models the core already accepts (`ColmapLoader.cpp` `derive_pinhole`) — the
    /// early-pass predicate. Ids 0 and 1 differ only in whether `fx` and `fy` are separate, and
    /// neither carries a distortion coefficient.
    var isPinhole: Bool { modelId == 0 || modelId == 1 }
}

/// One image's pose. Field for field what `images.{bin,txt}` stores, in COLMAP's own convention
/// (world → camera, quaternion `(w, x, y, z)` first). Nothing here is ever recomputed.
struct ColmapImageRecord: Equatable, Sendable {
    var id: Int
    var qw: Double, qx: Double, qy: Double, qz: Double
    var tx: Double, ty: Double, tz: Double
    var cameraId: Int
    var name: String
}

/// A sparse point with its colour. The track (which images observed it) is skipped on read and not
/// written back — the core skips it too, on both formats.
struct ColmapPointRecord: Equatable, Sendable {
    var x: Double, y: Double, z: Double
    var r: UInt8, g: UInt8, b: UInt8
}

/// The rectified camera, ready to be written. A dedicated type rather than a `ColmapCameraRecord`
/// with `modelId = 1`, so that "the writer only ever emits PINHOLE" is a fact of the type rather
/// than a validation that could be forgotten.
struct ColmapPinholeCamera: Equatable, Sendable {
    var id: Int
    var width: Int
    var height: Int
    var fx: Double, fy: Double, cx: Double, cy: Double
}

// MARK: - Errors

/// Everything that can go wrong reading or writing a model, phrased for the person who picked the
/// folder: what is wrong, and what they can do about it. These surface in the import sheet — the
/// import either converts the dataset or refuses it here, and never lets it reach training
/// (spec §1.1).
enum ColmapSparseError: LocalizedError, Equatable {
    case unreadable(String)
    case unwritable(String)
    case truncated(String)
    case malformed(file: String, line: Int, detail: String)
    case empty(String)
    case unsupportedModelName(String)
    case nameContainsSpace(String)

    var errorDescription: String? {
        switch self {
        case .unreadable(let file):
            "Cannot read sparse/0/\(file). Choose the folder that holds sparse/0 and images/, "
                + "or export the reconstruction from COLMAP again."
        case .unwritable(let file):
            "Cannot write \(file) into the new project. Check that the disk is not full and that "
                + "the projects folder is writable, then import again."
        case .truncated(let file):
            "sparse/0/\(file) ends in the middle of a record — the COLMAP model is incomplete. "
                + "Export the reconstruction from COLMAP again."
        case .malformed(let file, let line, let detail):
            "sparse/0/\(file) line \(line): \(detail). "
                + "Export the reconstruction from COLMAP again."
        case .empty(let file):
            "sparse/0/\(file) holds no records, so there is nothing to import. "
                + "Re-run the COLMAP reconstruction on this image set."
        case .unsupportedModelName(let name):
            "Unsupported COLMAP camera model \(name) — this app rectifies SIMPLE_PINHOLE, PINHOLE, "
                + "SIMPLE_RADIAL, RADIAL, OPENCV, OPENCV_FISHEYE, FULL_OPENCV, FOV, "
                + "SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE and THIN_PRISM_FISHEYE. Re-run COLMAP with "
                + "one of those camera models, or undistort the dataset with COLMAP first."
        case .nameContainsSpace(let name):
            "The image name \"\(name)\" contains a space, which COLMAP's text format cannot "
                + "represent (its pose line is exactly ten tokens). Rename the file — and its entry "
                + "in images.txt — without spaces, then import again."
        }
    }
}

// MARK: - Reading and writing

enum ColmapSparse {

    /// Which of the two on-disk encodings a complete file set is in.
    enum Format: Equatable, Sendable {
        case binary, text

        /// Derived from the set the caller already validated — see the file header on why this is
        /// not a fresh `fileExists` probe.
        static func of(set: [String]) -> Format {
            set.contains("cameras.bin") ? .binary : .text
        }

        var camerasFile: String { self == .binary ? "cameras.bin" : "cameras.txt" }
        var imagesFile: String { self == .binary ? "images.bin" : "images.txt" }
        var pointsFile: String { self == .binary ? "points3D.bin" : "points3D.txt" }
    }

    // MARK: Cameras

    /// Every camera, in file order.
    ///
    /// An unknown model id has to fail here rather than be skipped: in the binary format the
    /// parameter count *is* the record length, so a model this app does not know is a model whose
    /// record it cannot step over — everything after it would be misread as numbers. The text format
    /// could skip the line, and deliberately does not: quietly dropping a camera would leave images
    /// pointing at a camera id that no longer exists (spec §3.4 — no silent approximation).
    static func readCameras(in sparse0: URL, format: Format) throws -> [ColmapCameraRecord] {
        let file = format.camerasFile
        var out: [ColmapCameraRecord] = []
        switch format {
        case .binary:
            try withBinary(sparse0.appendingPathComponent(file), file) { cursor in
                let n = try cursor.count()
                out.reserveCapacity(n)
                for _ in 0..<n {
                    let id = Int(try cursor.i32())
                    let modelId = Int(try cursor.i32())
                    let width = try cursor.count()
                    let height = try cursor.count()
                    guard let model = ColmapCameraModel(rawValue: modelId) else {
                        // The numeric-id twin of `.unsupportedModelName`; `CameraModelError` already
                        // says the right thing for an id, including what to do about it.
                        throw CameraModelError.unsupportedModel(id: modelId)
                    }
                    var params = [Double]()
                    params.reserveCapacity(model.parameterCount)
                    for _ in 0..<model.parameterCount { params.append(try cursor.f64()) }
                    out.append(ColmapCameraRecord(id: id, modelId: modelId,
                                                  width: width, height: height, params: params))
                }
            }
        case .text:
            for (number, tokens) in try dataLines(sparse0.appendingPathComponent(file), file) {
                guard tokens.count >= 4 else {
                    throw ColmapSparseError.malformed(
                        file: file, line: number,
                        detail: "expected CAMERA_ID MODEL WIDTH HEIGHT PARAMS[], got "
                            + "\(tokens.count) token(s)")
                }
                guard let id = Int(tokens[0]), let width = Int(tokens[2]),
                      let height = Int(tokens[3]) else {
                    throw ColmapSparseError.malformed(
                        file: file, line: number, detail: "CAMERA_ID, WIDTH and HEIGHT must be whole numbers")
                }
                guard let model = ColmapCameraModel(name: tokens[1]) else {
                    throw ColmapSparseError.unsupportedModelName(tokens[1])
                }
                let raw = tokens[4...]
                guard raw.count == model.parameterCount else {
                    throw ColmapSparseError.malformed(
                        file: file, line: number,
                        detail: "\(model.name) takes \(model.parameterCount) parameters, got \(raw.count)")
                }
                var params = [Double]()
                params.reserveCapacity(raw.count)
                for token in raw {
                    guard let value = Double(token) else {
                        throw ColmapSparseError.malformed(
                            file: file, line: number, detail: "'\(token)' is not a number")
                    }
                    params.append(value)
                }
                out.append(ColmapCameraRecord(id: id, modelId: model.rawValue,
                                              width: width, height: height, params: params))
            }
        }
        guard !out.isEmpty else { throw ColmapSparseError.empty(file) }
        return out
    }

    // MARK: Images

    /// Every pose, in file order — the order is kept because the rewritten model keeps the original
    /// `IMAGE_ID`s, and a reader that reorders would make that claim harder to check.
    static func readImages(in sparse0: URL, format: Format) throws -> [ColmapImageRecord] {
        let file = format.imagesFile
        var out: [ColmapImageRecord] = []
        switch format {
        case .binary:
            try withBinary(sparse0.appendingPathComponent(file), file) { cursor in
                let n = try cursor.count()
                out.reserveCapacity(n)
                for _ in 0..<n {
                    let id = Int(try cursor.i32())
                    let qw = try cursor.f64(), qx = try cursor.f64()
                    let qy = try cursor.f64(), qz = try cursor.f64()
                    let tx = try cursor.f64(), ty = try cursor.f64(), tz = try cursor.f64()
                    let cameraId = Int(try cursor.i32())
                    let name = try cursor.cString()
                    // (x f64, y f64, point3D_id i64) per observation — skipped, exactly as the core
                    // does (`ColmapLoader.cpp:292`). The length is read into a local first: passing
                    // it inline would be two overlapping mutating accesses to the cursor.
                    let observations = try cursor.count()
                    try cursor.skipRecords(count: observations, stride: 24)
                    out.append(ColmapImageRecord(id: id, qw: qw, qx: qx, qy: qy, qz: qz,
                                                 tx: tx, ty: ty, tz: tz,
                                                 cameraId: cameraId, name: name))
                }
            }
        case .text:
            // The pose line is followed by a POINTS2D line that has to be consumed as a *physical*
            // line, not as the next data line: a blank POINTS2D line (what `ColmapTextWriter` and
            // this file both emit) would otherwise be skipped as whitespace and the following pose
            // line eaten as observations. This is the core's own rule — `ColmapLoader.cpp:219-222`
            // reads it with `raw_line` rather than `data_line`, unconditionally.
            let lines = try physicalLines(sparse0.appendingPathComponent(file), file)
            var i = 0
            while i < lines.count {
                let line = lines[i]
                i += 1
                guard isDataLine(line) else { continue }
                let tokens = line.split(whereSeparator: \.isWhitespace).map(String.init)
                guard tokens.count == 10 else {
                    throw ColmapSparseError.malformed(
                        file: file, line: i,
                        detail: "expected 10 tokens (IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME), "
                            + "got \(tokens.count) — an image name containing a space produces this")
                }
                guard let id = Int(tokens[0]), let cameraId = Int(tokens[8]) else {
                    throw ColmapSparseError.malformed(
                        file: file, line: i, detail: "IMAGE_ID and CAMERA_ID must be whole numbers")
                }
                var v = [Double]()
                for token in tokens[1...7] {
                    guard let value = Double(token) else {
                        throw ColmapSparseError.malformed(
                            file: file, line: i, detail: "'\(token)' is not a number")
                    }
                    v.append(value)
                }
                out.append(ColmapImageRecord(id: id, qw: v[0], qx: v[1], qy: v[2], qz: v[3],
                                             tx: v[4], ty: v[5], tz: v[6],
                                             cameraId: cameraId, name: tokens[9]))
                // One physical line, present or not: COLMAP writes it for every image including the
                // last, but a file that stops at EOF is still unambiguous, so it is tolerated rather
                // than made an error.
                if i < lines.count { i += 1 }
            }
        }
        guard !out.isEmpty else { throw ColmapSparseError.empty(file) }
        return out
    }

    // MARK: Points

    /// Every point **with its colour** — see the file header on why colour is load bearing.
    ///
    /// An empty `points3D` is allowed: the core allows it too (`ColmapLoader.cpp:227-228` says so in
    /// as many words), and a reconstruction whose points were stripped is still a set of usable
    /// poses.
    static func readPoints(in sparse0: URL, format: Format) throws -> [ColmapPointRecord] {
        let file = format.pointsFile
        var out: [ColmapPointRecord] = []
        switch format {
        case .binary:
            try withBinary(sparse0.appendingPathComponent(file), file) { cursor in
                let n = try cursor.count()
                out.reserveCapacity(n)
                for _ in 0..<n {
                    _ = try cursor.count()                                   // POINT3D_ID
                    let x = try cursor.f64(), y = try cursor.f64(), z = try cursor.f64()
                    let r = try cursor.u8(), g = try cursor.u8(), b = try cursor.u8()
                    _ = try cursor.f64()                                     // reprojection error
                    // (image_id i32, point2D_idx i32) per track entry.
                    let track = try cursor.count()
                    try cursor.skipRecords(count: track, stride: 8)
                    out.append(ColmapPointRecord(x: x, y: y, z: z, r: r, g: g, b: b))
                }
            }
        case .text:
            for (number, tokens) in try dataLines(sparse0.appendingPathComponent(file), file) {
                guard tokens.count >= 8 else {
                    throw ColmapSparseError.malformed(
                        file: file, line: number,
                        detail: "expected POINT3D_ID X Y Z R G B ERROR TRACK[], got "
                            + "\(tokens.count) token(s)")
                }
                guard let x = Double(tokens[1]), let y = Double(tokens[2]), let z = Double(tokens[3])
                else {
                    throw ColmapSparseError.malformed(
                        file: file, line: number, detail: "X, Y and Z must be numbers")
                }
                var rgb = [UInt8]()
                for token in tokens[4...6] {
                    guard let value = Int(token), (0...255).contains(value) else {
                        throw ColmapSparseError.malformed(
                            file: file, line: number, detail: "R, G and B must be 0…255, got '\(token)'")
                    }
                    rgb.append(UInt8(value))
                }
                out.append(ColmapPointRecord(x: x, y: y, z: z,
                                             r: rgb[0], g: rgb[1], b: rgb[2]))
            }
        }
        return out
    }

    // MARK: Writing

    /// Writes the rectified model as the three text files, in the shape the core parses.
    ///
    /// - `cameras.txt` — PINHOLE only, which is the whole point of the exercise.
    /// - `images.txt` — the pose line verbatim from the input, then an **empty POINTS2D line**. The
    ///   core consumes that second line without looking at it (`ColmapLoader.cpp:222`), and the
    ///   observations it would hold index into a `points2D` array this app never reconstructs. Note
    ///   the line has to be there: the loader fails with "missing POINTS2D line" if it is not.
    /// - `points3D.txt` — `ID X Y Z R G B ERROR`, colour preserved, no track, written in chunks by
    ///   `writePoints` because this is the file that grows with the reconstruction. ERROR is 0
    ///   because the reprojection error of the *original* solve no longer describes the rectified
    ///   images, and the core parses the column for validation only (`ColmapLoader.cpp:248`).
    ///
    /// Camera and image ids are the input's own. Nothing is renumbered and nothing is re-sorted:
    /// the sort that matters happens in the core (`TrainC.cpp` orders by image name), and a
    /// renumbering here would make "the poses came through unchanged" an assertion about a mapping
    /// rather than about values.
    static func writeText(cameras: [ColmapPinholeCamera],
                          images: [ColmapImageRecord],
                          points: [ColmapPointRecord],
                          to sparse0: URL) throws {
        // `ColmapRectification.rectify` refuses such a name in its own pre-checks, before it decodes
        // an image — this guard is the *writer's* invariant and stays, so that "what this writes is
        // what the core reads back" is a property of the writer rather than of one caller's ordering.
        if let bad = images.first(where: { $0.name.contains(" ") }) {
            throw ColmapSparseError.nameContainsSpace(bad.name)
        }
        try FileManager.default.createDirectory(at: sparse0, withIntermediateDirectories: true)

        var camerasText = "# Camera list: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n"
        for c in cameras {
            camerasText += "\(c.id) PINHOLE \(c.width) \(c.height) "
                + "\(fmt(c.fx)) \(fmt(c.fy)) \(fmt(c.cx)) \(fmt(c.cy))\n"
        }

        var imagesText = "# Image list: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
        for i in images {
            imagesText += "\(i.id) \(fmt(i.qw)) \(fmt(i.qx)) \(fmt(i.qy)) \(fmt(i.qz)) "
                + "\(fmt(i.tx)) \(fmt(i.ty)) \(fmt(i.tz)) \(i.cameraId) \(i.name)\n\n"
        }

        try camerasText.write(to: sparse0.appendingPathComponent("cameras.txt"),
                              atomically: true, encoding: .utf8)
        try imagesText.write(to: sparse0.appendingPathComponent("images.txt"),
                             atomically: true, encoding: .utf8)
        try writePoints(points, to: sparse0.appendingPathComponent("points3D.txt"))
    }

    /// Points written per `write` call. A few hundred KB in flight: enough that the syscall count is
    /// irrelevant, little enough that peak memory does not scale with the reconstruction.
    private static let pointsFlushInterval = 8192

    /// `points3D.txt`, streamed rather than assembled.
    ///
    /// The other two files are built in memory and that is right: a model has a handful of cameras
    /// and one line per photo. `points3D` is the one thing here that scales with the reconstruction
    /// — a dense COLMAP solve runs to millions of points — and building its text in a `String` first
    /// cost roughly 290 B resident per point (the ~85-byte line, the growth headroom `+=` leaves, and
    /// the `Data` copy `write(to:atomically:)` makes), about 0.87 GB at 3 M points. The reader
    /// already takes care to *map* `points3D.bin` instead of copying it (`withBinary`); the writer
    /// holding its whole output in RAM was the unaddressed half of that same concern.
    ///
    /// Written to a sibling `.partial` and renamed, so the atomicity `write(to:atomically:)` gives
    /// the other two files is kept: no reader ever meets a half-written `points3D.txt`.
    private static func writePoints(_ points: [ColmapPointRecord], to url: URL) throws {
        let fm = FileManager.default
        let partial = url.deletingLastPathComponent()
            .appendingPathComponent(url.lastPathComponent + ".partial")
        try? fm.removeItem(at: partial)
        guard fm.createFile(atPath: partial.path, contents: nil) else {
            throw ColmapSparseError.unwritable(url.lastPathComponent)
        }
        let handle = try FileHandle(forWritingTo: partial)
        do {
            var buffer = "# 3D point list: POINT3D_ID X Y Z R G B ERROR TRACK[]\n"
            buffer.reserveCapacity(pointsFlushInterval * 96)
            for (index, p) in points.enumerated() {
                buffer += "\(index + 1) \(fmt(p.x)) \(fmt(p.y)) \(fmt(p.z)) "
                    + "\(p.r) \(p.g) \(p.b) 0\n"
                // Flushed on a point count, not on a byte count: `buffer.utf8.count` walks the whole
                // buffer, which would make the size check quadratic in the chunk length.
                if (index + 1) % pointsFlushInterval == 0 {
                    try handle.write(contentsOf: Data(buffer.utf8))
                    buffer.removeAll(keepingCapacity: true)
                }
            }
            if !buffer.isEmpty { try handle.write(contentsOf: Data(buffer.utf8)) }
            try handle.close()
        } catch {
            try? handle.close()
            try? fm.removeItem(at: partial)
            throw error
        }
        try? fm.removeItem(at: url)
        try fm.moveItem(at: partial, to: url)
    }

    /// 17 significant digits — the shortest field that round-trips every `Double` (see the file
    /// header). `%g` drops the exponent and the trailing zeros whenever it can, so ordinary values
    /// still read as `1920` and `0.5`.
    private static func fmt(_ v: Double) -> String { String(format: "%.17g", v) }

    // MARK: Text plumbing

    /// Physical lines, CRLF tolerated, **empty lines kept** — `readImages` depends on the blank
    /// POINTS2D line still being there to consume.
    ///
    /// Split on `\.isNewline` (2026-08-15 FIX 2). This used to split on the literal `"\n"` and strip
    /// a trailing `"\r"` afterwards, which reads as CRLF handling and is not: Swift's `Character` is
    /// a grapheme cluster, `"\r\n"` is one cluster, and it equals neither `"\n"` nor `"\r"`. So the
    /// split found no separator anywhere in a CRLF file, `hasSuffix("\r")` was never true, and the
    /// whole model arrived as one line — beginning with `#`, hence no data lines, hence
    /// "sparse/0/cameras.txt holds no records" on a perfectly good model exported on Windows. The
    /// `isNewline` form treats CRLF as a single separator, so the blank POINTS2D line survives as an
    /// empty element exactly as it does in an LF file.
    private static func physicalLines(_ url: URL, _ file: String) throws -> [Substring] {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            throw ColmapSparseError.unreadable(file)
        }
        return text.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline)
    }

    /// A line the core would parse: not blank, not a comment. Same rule as `TextReader::data_line`.
    private static func isDataLine(_ line: Substring) -> Bool {
        guard let first = line.first(where: { !$0.isWhitespace }) else { return false }
        return first != "#"
    }

    /// `(1-based line number, whitespace-split tokens)` for every data line.
    private static func dataLines(_ url: URL, _ file: String) throws -> [(Int, [String])] {
        try physicalLines(url, file).enumerated()
            .compactMap { (index, line) -> (Int, [String])? in
                guard isDataLine(line) else { return nil }
                return (index + 1, line.split(whereSeparator: \.isWhitespace).map(String.init))
            }
    }

    // MARK: Binary plumbing

    /// Runs `body` over the whole file as raw bytes. The file is mapped rather than copied into an
    /// array: `points3D.bin` runs to tens of megabytes on a real reconstruction, and it is read once.
    private static func withBinary(_ url: URL, _ file: String,
                                   _ body: (inout Cursor) throws -> Void) throws {
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe) else {
            throw ColmapSparseError.unreadable(file)
        }
        try data.withUnsafeBytes { bytes in
            var cursor = Cursor(bytes: bytes, file: file)
            try body(&cursor)
        }
    }

    /// A bounds-checked forward reader. Every overrun is one error — a file that ends early is
    /// truncated, whichever field noticed.
    private struct Cursor {
        let bytes: UnsafeRawBufferPointer
        let file: String
        var offset = 0

        mutating func load<T>(_ type: T.Type) throws -> T {
            let size = MemoryLayout<T>.size
            guard offset + size <= bytes.count else { throw ColmapSparseError.truncated(file) }
            defer { offset += size }
            return bytes.loadUnaligned(fromByteOffset: offset, as: T.self)
        }

        mutating func u8() throws -> UInt8 { try load(UInt8.self) }
        mutating func i32() throws -> Int32 { try load(Int32.self) }
        mutating func f64() throws -> Double { try load(Double.self) }

        /// A `uint64_t` length field as an `Int`. A value past `Int.max` is a corrupt file, not a
        /// 9-exabyte model, so it is reported as truncation instead of trapping the conversion.
        mutating func count() throws -> Int {
            let raw: UInt64 = try load(UInt64.self)
            guard raw <= UInt64(Int.max) else { throw ColmapSparseError.truncated(file) }
            return Int(raw)
        }

        /// Skips `count` fixed-size records.
        ///
        /// Both arithmetic steps are checked, because `count` comes out of the file and a corrupt
        /// file must produce an error rather than a crash. The addition is the less obvious of the
        /// two: a `points3D.bin` claiming a track length of `Int.max / 8` multiplies to `Int.max − 7`
        /// without overflowing, and it is then `offset + bytesToSkip` that traps.
        mutating func skipRecords(count: Int, stride: Int) throws {
            let (bytesToSkip, mulOverflow) = count.multipliedReportingOverflow(by: stride)
            guard !mulOverflow, bytesToSkip >= 0 else {
                throw ColmapSparseError.truncated(file)
            }
            let (end, addOverflow) = offset.addingReportingOverflow(bytesToSkip)
            guard !addOverflow, end <= bytes.count else {
                throw ColmapSparseError.truncated(file)
            }
            offset = end
        }

        mutating func cString() throws -> String {
            var out: [UInt8] = []
            while true {
                let c = try u8()
                if c == 0 { return String(decoding: out, as: UTF8.self) }
                out.append(c)
            }
        }
    }
}
