// Deleted in the first reorganization and revived for the camera fit (2026-07-20 spec §6)
import Foundation
import simd

struct ColmapIntrinsics { let fx, fy, cx, cy: Double; let width, height: Int }
struct ColmapView: Identifiable { let id: Int; let name: String; let cameraId: Int; let qvec: SIMD4<Double>; let tvec: SIMD3<Double> }
struct ColmapModel { let cameras: [Int: ColmapIntrinsics]; let views: [ColmapView] }

enum ColmapReaderError: Error { case notFound, unsupportedModel(String), truncated, parseError(String) }

enum ColmapCameraReader {
    static func read(sparseDir: URL) throws -> ColmapModel {
        let fm = FileManager.default
        if fm.fileExists(atPath: sparseDir.appending(path: "cameras.bin").path) {
            let cams = try readCamerasBin(sparseDir.appending(path: "cameras.bin"))
            let views = try readImagesBin(sparseDir.appending(path: "images.bin"))
            return ColmapModel(cameras: cams, views: views)
        } else if fm.fileExists(atPath: sparseDir.appending(path: "cameras.txt").path) {
            let cams = try readCamerasTxt(sparseDir.appending(path: "cameras.txt"))
            let views = try readImagesTxt(sparseDir.appending(path: "images.txt"))
            return ColmapModel(cameras: cams, views: views)
        }
        throw ColmapReaderError.notFound
    }

    /// A minimal parse of **positions only** from points3D (bin and txt) — for computing the camera
    /// fit's centroid (2026-07-20 spec §6).
    static func loadPoints(sparse0: URL) throws -> [SIMD3<Float>] {
        let fm = FileManager.default
        let bin = sparse0.appending(path: "points3D.bin")
        if fm.fileExists(atPath: bin.path) { return try readPoints3DBin(bin) }
        let txt = sparse0.appending(path: "points3D.txt")
        if fm.fileExists(atPath: txt.path) { return try readPoints3DTxt(txt) }
        throw ColmapReaderError.notFound
    }

    // points3D.bin: num(u64) | per-pt: id(u64) xyz(3×f64) rgb(3B) error(f64)
    //               track_len(u64) track(image_id i32, point2D_idx i32)*  — everything but the position is skipped.
    private static func readPoints3DBin(_ url: URL) throws -> [SIMD3<Float>] {
        var r = ByteReader(try Data(contentsOf: url))
        let n = try r.count()
        var out: [SIMD3<Float>] = []
        out.reserveCapacity(n)
        for _ in 0..<n {
            try r.skip(8)                                     // point3D_id(u64)
            let x = try r.f64(), y = try r.f64(), z = try r.f64()
            try r.skip(3 + 8)                                 // rgb(3B) + error(f64)
            let trackLen = try r.count()
            let (bytes, overflow) = trackLen.multipliedReportingOverflow(by: 4 + 4)
            guard !overflow else { throw ColmapReaderError.truncated }
            try r.skip(bytes)
            out.append(SIMD3(Float(x), Float(y), Float(z)))
        }
        return out
    }

    // points3D.txt: POINT3D_ID X Y Z R G B ERROR TRACK[] — only token columns 2–4 (X Y Z) are used.
    private static func readPoints3DTxt(_ url: URL) throws -> [SIMD3<Float>] {
        var out: [SIMD3<Float>] = []
        for line in try lines(url) {
            let t = line.split(separator: " ")
            guard t.count >= 4 else { continue }
            guard let x = Double(t[1]), let y = Double(t[2]), let z = Double(t[3]) else {
                throw ColmapReaderError.parseError("points3D.txt: failed to parse position → \(line)")
            }
            out.append(SIMD3(Float(x), Float(y), Float(z)))
        }
        return out
    }

    // --- bin ---
    private static func readCamerasBin(_ url: URL) throws -> [Int: ColmapIntrinsics] {
        var r = ByteReader(try Data(contentsOf: url))
        let n = try r.count()
        var out: [Int: ColmapIntrinsics] = [:]
        for _ in 0..<n {
            let id = Int(try r.i32()); let model = Int(try r.u32())
            let w = try r.count(); let h = try r.count()
            switch model {
            case 0: let f = try r.f64(); let cx = try r.f64(); let cy = try r.f64()
                    out[id] = ColmapIntrinsics(fx: f, fy: f, cx: cx, cy: cy, width: w, height: h)
            case 1: let fx = try r.f64(); let fy = try r.f64(); let cx = try r.f64(); let cy = try r.f64()
                    out[id] = ColmapIntrinsics(fx: fx, fy: fy, cx: cx, cy: cy, width: w, height: h)
            default: throw ColmapReaderError.unsupportedModel("model_id \(model) — only PINHOLE supported (undistort required)")
            }
        }
        return out
    }
    private static func readImagesBin(_ url: URL) throws -> [ColmapView] {
        var r = ByteReader(try Data(contentsOf: url))
        let n = try r.count()
        var out: [ColmapView] = []
        for _ in 0..<n {
            let id = Int(try r.u32())
            let qw = try r.f64(), qx = try r.f64(), qy = try r.f64(), qz = try r.f64()
            let tx = try r.f64(), ty = try r.f64(), tz = try r.f64()
            let camId = Int(try r.u32())
            var nameBytes: [UInt8] = []; while let c = r.u8opt(), c != 0 { nameBytes.append(c) }
            let name = String(decoding: nameBytes, as: UTF8.self)
            let npts = try r.count()
            let (needed, overflow) = npts.multipliedReportingOverflow(by: 8 + 8 + 8)  // x(f64),y(f64),point3D_id(i64)
            guard !overflow else { throw ColmapReaderError.truncated }
            try r.skip(needed)
            out.append(ColmapView(id: id, name: name, cameraId: camId,
                                  qvec: SIMD4(qw,qx,qy,qz), tvec: SIMD3(tx,ty,tz)))
        }
        return out
    }

    // --- txt ---
    private static func readCamerasTxt(_ url: URL) throws -> [Int: ColmapIntrinsics] {
        var out: [Int: ColmapIntrinsics] = [:]
        for line in try lines(url) {
            let t = line.split(separator: " ").map(String.init)
            guard t.count >= 5 else { continue }
            guard let id = Int(t[0]), let w = Int(t[2]), let h = Int(t[3]) else {
                throw ColmapReaderError.parseError("cameras.txt: failed to parse ID/dimensions → \(line)")
            }
            let model = t[1]
            let p = t[4...].compactMap(Double.init)
            switch model {
            case "SIMPLE_PINHOLE":
                guard p.count >= 3 else { throw ColmapReaderError.parseError("cameras.txt: insufficient SIMPLE_PINHOLE parameters → \(line)") }
                out[id] = ColmapIntrinsics(fx: p[0], fy: p[0], cx: p[1], cy: p[2], width: w, height: h)
            case "PINHOLE":
                guard p.count >= 4 else { throw ColmapReaderError.parseError("cameras.txt: insufficient PINHOLE parameters → \(line)") }
                out[id] = ColmapIntrinsics(fx: p[0], fy: p[1], cx: p[2], cy: p[3], width: w, height: h)
            default: throw ColmapReaderError.unsupportedModel("\(model) — only PINHOLE supported")
            }
        }
        return out
    }
    private static func readImagesTxt(_ url: URL) throws -> [ColmapView] {
        // COLMAP images.txt consists of exactly two non-comment data lines per image:
        //   line 1 = the header (IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME)
        //   line 2 = the 2D observations (X Y POINT3D_ID ...)
        // If the first token of an observation line is integral, a heuristic would mistake it for
        // an image, so the non-comment lines are paired up and only the first of each pair is
        // parsed as a header while the second is skipped.
        //
        // That "exactly two lines" premise collided with the shared `lines()` helper, which drops
        // empty lines (2026-08-15 FIX 1). The observation line is legitimately **empty** whenever an
        // image carries no 2D points — which is every image in the bundles `ColmapTextWriter` emits
        // (it writes `NAME\n\n`) and, from this round on, in the rectified bundles the importer
        // writes as well. With those empty lines filtered away, the array degenerated into
        // [pose, pose, pose, …] and `i += 2` read **every other view**: a real 110-image project
        // parsed as 55 views, silently halving the up-axis estimate, the timelapse anchor, and the
        // Cinema trajectory. So images.txt uses its own reader (`imageDataLines`) that keeps empty
        // lines and therefore pairs correctly in both shapes:
        //   real COLMAP        → [pose, obs, pose, obs, …]
        //   ColmapTextWriter   → [pose, "",  pose, "",  …]
        // Do NOT "simplify" this back onto `lines()` — that is exactly the regression this fixes.
        var out: [ColmapView] = []
        let all = try imageDataLines(url)
        var i = 0
        while i < all.count {
            let header = all[i]
            let t = header.split(separator: " ").map(String.init)
            guard t.count >= 10 else {
                throw ColmapReaderError.parseError("images.txt: insufficient header tokens (≥10 required) → \(header)")
            }
            guard let id = Int(t[0]),
                  let qw = Double(t[1]), let qx = Double(t[2]), let qy = Double(t[3]), let qz = Double(t[4]),
                  let tx = Double(t[5]), let ty = Double(t[6]), let tz = Double(t[7]),
                  let camId = Int(t[8]) else {
                throw ColmapReaderError.parseError("images.txt: failed to parse header → \(header)")
            }
            let name = t[9...].joined(separator: " ")  // the NAME field (in case it contains spaces)
            out.append(ColmapView(id: id, name: name, cameraId: camId,
                                  qvec: SIMD4(qw,qx,qy,qz), tvec: SIMD3(tx,ty,tz)))
            i += 2  // skip the next line (the 2D observations)
        }
        return out
    }
    /// The line reader for `cameras.txt` / `points3D.txt`, where a record is exactly one line and a
    /// blank line carries no meaning — so dropping empties is correct here.
    /// `images.txt` must NOT use this: see `imageDataLines`.
    ///
    /// Split on `\.isNewline`, **not** on the literal `"\n"` (2026-08-15 FIX 2). Swift's `Character`
    /// is a grapheme cluster and `"\r\n"` is one such cluster — it is not equal to `"\n"` — so
    /// `split(separator: "\n")` does not split a CRLF file *anywhere*: the whole model came back as
    /// a single line, that line began with `#`, and the filter below dropped it. A model exported on
    /// Windows therefore read as zero cameras and zero views, with no error raised. `isNewline` is
    /// true for the `"\r\n"` cluster, for a lone `\r` and for `\n`, and treats CRLF as **one**
    /// separator rather than two, which is what keeps `imageDataLines`' pairing intact.
    ///
    /// The core has always handled this (`ColmapLoader.cpp:118-123` — `raw_line` pops a trailing
    /// `'\r'`); the two Swift readers had not.
    private static func lines(_ url: URL) throws -> [String] {
        try String(contentsOf: url, encoding: .utf8)
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }
    }

    /// The line reader for `images.txt` only: comments are dropped and each line is trimmed, but
    /// **empty lines are preserved** because an empty line is a valid (empty) POINTS2D record and
    /// removing it breaks the header/observation pairing (see `readImagesTxt`).
    ///
    /// Trailing empty lines are the one exception and are removed. Files normally end with a
    /// newline, so `split` yields at least one trailing empty element; `ColmapTextWriter`'s
    /// `NAME\n\n` yields two. Feeding those to the loop would parse an empty string as a header and
    /// throw on the `t.count >= 10` guard. Trimming from the **end** was chosen over a guard inside
    /// the loop because it cannot shift the index parity of any earlier line — pairing stays exact —
    /// whereas an in-loop `if header.isEmpty { break/continue }` would also swallow a genuinely
    /// malformed file in the middle, turning a parse error into a silently truncated view list. That
    /// is the same class of silent halving this FIX removes.
    ///
    /// Split on `\.isNewline` for the same reason as `lines` above (2026-08-15 FIX 2), and here it
    /// matters twice over: CRLF has to count as **one** separator, or every pose line would be
    /// followed by a phantom empty line and `i += 2` would pair a pose with the second half of its
    /// own line ending. `split(omittingEmptySubsequences: false, whereSeparator: \.isNewline)` does
    /// exactly that — measured, `"a\r\n\r\nb\r\n"` → `["a", "", "b", ""]`, identical to the LF form.
    private static func imageDataLines(_ url: URL) throws -> [String] {
        var out = try String(contentsOf: url, encoding: .utf8)
            .split(omittingEmptySubsequences: false, whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.hasPrefix("#") }
        while let last = out.last, last.isEmpty { out.removeLast() }
        return out
    }
}

private struct ByteReader {
    let d: [UInt8]; var o = 0
    init(_ data: Data) { d = [UInt8](data) }
    mutating func skip(_ n: Int) throws {
        guard n >= 0, o + n <= d.count else { throw ColmapReaderError.truncated }
        o += n
    }
    mutating func u8opt() -> UInt8? { guard o < d.count else { return nil }; defer { o += 1 }; return d[o] }
    mutating func read<T>(_ type: T.Type) throws -> T {
        let size = MemoryLayout<T>.size
        guard o + size <= d.count else { throw ColmapReaderError.truncated }
        let v = d[o..<o+size].withUnsafeBytes { $0.loadUnaligned(as: T.self) }
        o += size
        return v
    }
    mutating func u64() throws -> UInt64 { try read(UInt64.self) }
    mutating func u32() throws -> UInt32 { try read(UInt32.self) }
    mutating func i32() throws -> Int32 { try read(Int32.self) }
    mutating func f64() throws -> Double { try read(Double.self) }
    // Safely converts a u64 into an Int count (an explicit error instead of a trap when it exceeds
    // Int.max).
    mutating func count() throws -> Int {
        let v = try u64()
        guard v <= UInt64(Int.max) else { throw ColmapReaderError.truncated }
        return Int(v)
    }
}
