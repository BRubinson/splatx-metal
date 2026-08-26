import Compression
import XCTest
@testable import SplatXMetalTrainer

/// Assembles ZIP bytes in-process so no fixture archive has to be committed, and — more to the
/// point — so the malformed cases can exist at all: an archive with a symlink entry, a ZIP64
/// sentinel or a wrong CRC cannot be produced by `zip(1)` and would be unreviewable as a binary
/// blob in git. Every field the reader inspects is overridable here.
///
/// Internal rather than fileprivate so `CaptureImportFlowTests` can build the one archive it needs
/// to exercise the zip branch of the import end to end — the alternative was a second, near-identical
/// builder in that file, and two copies of a hand-rolled container format is exactly the duplication
/// that goes stale.
struct ZipBuilder {
    struct Entry {
        var name: String
        var nameBytes: [UInt8]?              // raw override — for CP437 / non-UTF-8 names
        var payload: [UInt8] = []
        var method: UInt16 = 0               // 0 stored, 8 deflate, anything else written verbatim
        var flags: UInt16 = 0
        var externalAttributes: UInt32 = UInt32(0o100644) << 16
        var crcOverride: UInt32?
        var uncompressedSizeOverride: UInt32?
        var compressedSizeOverride: UInt32?
        var localOffsetOverride: UInt32?
        var localExtra: [UInt8] = []
        var centralExtra: [UInt8] = []
        /// Writes the entry the way `ditto` (Finder's "Compress") does: GP flag bit 3 set, the
        /// local header's crc/csize/usize **zeroed**, and the real values in a data descriptor
        /// after the payload as well as in the central directory. See the reader's type header for
        /// the dump this reproduces.
        var dataDescriptor = false

        init(_ name: String) { self.name = name }
    }

    var entries: [Entry] = []
    var totalEntriesOverride: UInt16?
    var diskNumberOverride: UInt16?
    var centralOffsetOverride: UInt32?
    var centralSizeOverride: UInt32?
    var comment: [UInt8] = []

    static func directory(_ name: String) -> Entry {
        var e = Entry(name.hasSuffix("/") ? name : name + "/")
        e.externalAttributes = UInt32(0o040755) << 16
        return e
    }

    static func stored(_ name: String, _ text: String) -> Entry {
        var e = Entry(name)
        e.payload = Array(text.utf8)
        return e
    }

    static func deflated(_ name: String, _ bytes: [UInt8]) -> Entry {
        var e = Entry(name)
        e.payload = bytes
        e.method = 8
        return e
    }

    func build() -> Data {
        var local: [UInt8] = []
        var central: [UInt8] = []
        for e in entries {
            let nameBytes = e.nameBytes ?? Array(e.name.utf8)
            let body = e.method == 8 ? Self.deflateRaw(e.payload) : e.payload
            let crc = e.crcOverride ?? ZipCRC32.compute(e.payload)
            let usize = e.uncompressedSizeOverride ?? UInt32(e.payload.count)
            let csize = e.compressedSizeOverride ?? UInt32(body.count)
            let offset = e.localOffsetOverride ?? UInt32(local.count)

            let flags = e.flags | (e.dataDescriptor ? 0x0008 : 0)

            local += Self.le32(0x0403_4b50) + Self.le16(20) + Self.le16(flags)
            local += Self.le16(e.method) + Self.le16(0) + Self.le16(0)
            local += e.dataDescriptor
                ? Self.le32(0) + Self.le32(0) + Self.le32(0)
                : Self.le32(crc) + Self.le32(csize) + Self.le32(usize)
            local += Self.le16(UInt16(nameBytes.count)) + Self.le16(UInt16(e.localExtra.count))
            local += nameBytes + e.localExtra + body
            if e.dataDescriptor {
                local += Self.le32(0x0807_4b50) + Self.le32(crc) + Self.le32(csize)
                    + Self.le32(usize)
            }

            central += Self.le32(0x0201_4b50) + Self.le16((3 << 8) | 20) + Self.le16(20)
            central += Self.le16(flags) + Self.le16(e.method) + Self.le16(0) + Self.le16(0)
            central += Self.le32(crc) + Self.le32(csize) + Self.le32(usize)
            central += Self.le16(UInt16(nameBytes.count)) + Self.le16(UInt16(e.centralExtra.count))
            central += Self.le16(0) + Self.le16(0) + Self.le16(0)
            central += Self.le32(e.externalAttributes) + Self.le32(offset)
            central += nameBytes + e.centralExtra
        }
        let centralOffset = UInt32(local.count)
        var out = local + central
        let count = totalEntriesOverride ?? UInt16(entries.count)
        out += Self.le32(0x0605_4b50)
        out += Self.le16(diskNumberOverride ?? 0) + Self.le16(diskNumberOverride ?? 0)
        out += Self.le16(count) + Self.le16(count)
        out += Self.le32(centralSizeOverride ?? UInt32(central.count))
        out += Self.le32(centralOffsetOverride ?? centralOffset)
        out += Self.le16(UInt16(comment.count)) + comment
        return Data(out)
    }

    private static func le16(_ v: UInt16) -> [UInt8] { [UInt8(v & 0xFF), UInt8(v >> 8)] }

    private static func le32(_ v: UInt32) -> [UInt8] {
        [UInt8(v & 0xFF), UInt8((v >> 8) & 0xFF), UInt8((v >> 16) & 0xFF), UInt8(v >> 24)]
    }

    /// The encoder side of the same `COMPRESSION_ZLIB` = raw DEFLATE identity the reader relies on
    /// — if that identity were wrong, this round trip would not close.
    private static func deflateRaw(_ src: [UInt8]) -> [UInt8] {
        guard !src.isEmpty else { return [] }
        var dst = [UInt8](repeating: 0, count: max(src.count * 2 + 256, 1024))
        let n = dst.withUnsafeMutableBufferPointer { d in
            src.withUnsafeBufferPointer { s in
                compression_encode_buffer(d.baseAddress!, d.count,
                                          s.baseAddress!, s.count, nil, COMPRESSION_ZLIB)
            }
        }
        precondition(n > 0, "compression_encode_buffer failed")
        return Array(dst[0..<n])
    }
}

final class ZipArchiveReaderTests: XCTestCase {
    private var tmp: URL!

    override func setUp() {
        tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("sx-zip-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    }

    override func tearDown() { try? FileManager.default.removeItem(at: tmp) }

    private func write(_ builder: ZipBuilder, as name: String = "a.zip") throws -> URL {
        let url = tmp.appendingPathComponent(name)
        try builder.build().write(to: url)
        return url
    }

    private func extract(_ builder: ZipBuilder,
                         limits: ZipArchiveReader.Limits = .default) throws -> (URL, [String]) {
        let zip = try write(builder)
        let out = tmp.appendingPathComponent("out-\(UUID().uuidString)", isDirectory: true)
        return (out, try ZipArchiveReader.extract(zip: zip, to: out, limits: limits))
    }

    private func assertRejected(_ builder: ZipBuilder,
                                limits: ZipArchiveReader.Limits = .default,
                                _ expected: ZipArchiveError,
                                file: StaticString = #filePath, line: UInt = #line) throws {
        let zip = try write(builder, as: "\(UUID().uuidString).zip")
        let out = tmp.appendingPathComponent("out-\(UUID().uuidString)", isDirectory: true)
        XCTAssertThrowsError(try ZipArchiveReader.extract(zip: zip, to: out, limits: limits),
                             file: file, line: line) { error in
            XCTAssertEqual(error as? ZipArchiveError, expected, file: file, line: line)
        }
    }

    // MARK: CRC

    func testCRC32MatchesTheStandardCheckVector() {
        XCTAssertEqual(ZipCRC32.compute(Array("123456789".utf8)), 0xCBF4_3926)
        XCTAssertEqual(ZipCRC32.compute([]), 0)
    }

    // MARK: Happy path

    func testExtractsStoredDeflatedAndDirectoryEntries() throws {
        // ~120 KB of repeating text so deflate actually engages (and the payload is bigger than any
        // single internal buffer boundary).
        let big = Array(String(repeating: "splatx capture bundle 0123456789\n", count: 4000).utf8)
        var b = ZipBuilder()
        b.entries = [ZipBuilder.directory("bundle"),
                     ZipBuilder.stored("bundle/splatx.json", #"{"mode":"lidar"}"#),
                     ZipBuilder.deflated("bundle/images/0001.jpg", big)]
        let (out, written) = try extract(b)

        XCTAssertEqual(written, ["bundle/splatx.json", "bundle/images/0001.jpg"])
        var isDir: ObjCBool = false
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: out.appendingPathComponent("bundle").path, isDirectory: &isDir))
        XCTAssertTrue(isDir.boolValue)
        XCTAssertEqual(try Data(contentsOf: out.appendingPathComponent("bundle/splatx.json")),
                       Data(#"{"mode":"lidar"}"#.utf8))
        XCTAssertEqual(try Data(contentsOf: out.appendingPathComponent("bundle/images/0001.jpg")),
                       Data(big))
    }

    func testCreatesIntermediateDirectoriesWithoutDirectoryEntries() throws {
        var b = ZipBuilder()
        b.entries = [ZipBuilder.stored("bundle/sparse/0/cameras.txt", "1 PINHOLE\n")]
        let (out, written) = try extract(b)
        XCTAssertEqual(written, ["bundle/sparse/0/cameras.txt"])
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: out.appendingPathComponent("bundle/sparse/0/cameras.txt").path))
    }

    func testEmptyFileEntryRoundTrips() throws {
        var b = ZipBuilder()
        b.entries = [ZipBuilder.stored("bundle/empty.txt", ""),
                     ZipBuilder.deflated("bundle/empty2.bin", [])]
        let (out, written) = try extract(b)
        XCTAssertEqual(written.count, 2)
        XCTAssertEqual(try Data(contentsOf: out.appendingPathComponent("bundle/empty.txt")), Data())
    }

    func testTrailingArchiveCommentDoesNotHideTheEOCD() throws {
        var b = ZipBuilder()
        b.entries = [ZipBuilder.stored("bundle/a.txt", "a")]
        b.comment = Array("created by a tool that likes to talk".utf8)
        let (_, written) = try extract(b)
        XCTAssertEqual(written, ["bundle/a.txt"])
    }

    // MARK: Local header re-parse

    func testPayloadOffsetComesFromTheLocalHeaderNotTheCentralDirectory() throws {
        // A writer put an 11-byte extra field in the local header and none in the central one.
        // Trusting the central length would start the read 11 bytes early and hand back garbage.
        var e = ZipBuilder.stored("bundle/a.txt", "correct payload")
        e.localExtra = Array(repeating: 0x5A, count: 11)
        e.centralExtra = []
        var b = ZipBuilder()
        b.entries = [e]
        let (out, _) = try extract(b)
        XCTAssertEqual(try Data(contentsOf: out.appendingPathComponent("bundle/a.txt")),
                       Data("correct payload".utf8))
    }

    /// **The central directory is authoritative for crc/csize/usize.** This is not a style choice
    /// the reader is free to revisit: the instant a user re-compresses a bundle in Finder, `ditto`
    /// writes it with GP flag bit 3 and *zeroes* those three fields in every local header, keeping
    /// the real values only in the central directory (measured on a real recompressed capture —
    /// `LFH (0, 0, 0)` vs `CD (1210202599, 1488994, 1489313)` for `images/0012.jpg`).
    ///
    /// Without this test the suite passes with a reader that trusts the local header, and every
    /// Finder-recompressed bundle is then rejected as damaged: a zero `csize` reads no payload at
    /// all, so the failure lands on the CRC and reads like a corrupt archive rather than a bug.
    /// Both members here also carry 12 bytes of local extra against none in the central directory,
    /// the other half of what `ditto` actually emits.
    func testCentralDirectorySizesWinOverAZeroedLocalHeader() throws {
        let big = Array(String(repeating: "ditto writes data descriptors\n", count: 2000).utf8)
        var stored = ZipBuilder.stored("bundle/splatx.json", #"{"mode":"lidar"}"#)
        stored.dataDescriptor = true
        stored.localExtra = Array(repeating: 0x00, count: 12)
        var deflated = ZipBuilder.deflated("bundle/images/0012.jpg", big)
        deflated.dataDescriptor = true
        deflated.localExtra = Array(repeating: 0x00, count: 12)
        var b = ZipBuilder()
        b.entries = [stored, deflated]

        // The fixture has to *be* that shape or the assertions below prove nothing: bit 3 set and
        // crc/csize/usize (bytes 14…25 of the local file header) all zero. Checked here rather
        // than trusted, because a ZipBuilder regression would otherwise defang this test silently.
        let raw = [UInt8](b.build())
        XCTAssertEqual(raw[6] & 0x08, 0x08, "GP flag bit 3 is not set in the local header")
        XCTAssertEqual(Array(raw[14..<26]), [UInt8](repeating: 0, count: 12))

        let (out, written) = try extract(b)

        XCTAssertEqual(written, ["bundle/splatx.json", "bundle/images/0012.jpg"])
        XCTAssertEqual(try Data(contentsOf: out.appendingPathComponent("bundle/splatx.json")),
                       Data(#"{"mode":"lidar"}"#.utf8))
        XCTAssertEqual(try Data(contentsOf: out.appendingPathComponent("bundle/images/0012.jpg")),
                       Data(big))
    }

    // MARK: AppleDouble noise

    func testIgnoresMacOSXShadowTreeAndAppleDoubleFiles() throws {
        var b = ZipBuilder()
        b.entries = [ZipBuilder.directory("__MACOSX"),
                     ZipBuilder.stored("__MACOSX/bundle/._splatx.json", "resource fork"),
                     ZipBuilder.stored("bundle/._0001.jpg", "resource fork"),
                     ZipBuilder.stored("bundle/splatx.json", "{}"),
                     ZipBuilder.stored("bundle/images/0001.jpg", "jpeg")]
        let (out, written) = try extract(b)
        XCTAssertEqual(written, ["bundle/splatx.json", "bundle/images/0001.jpg"])
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: out.appendingPathComponent("__MACOSX").path))
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: out.appendingPathComponent("bundle/._0001.jpg").path))
    }

    // MARK: Name encoding

    func testDecodesCP437WhenTheEFSFlagIsAbsentAndUTF8WhenItIsSet() throws {
        var cp437 = ZipBuilder.Entry("ignored")
        cp437.nameBytes = Array("caf".utf8) + [0x82] + Array(".txt".utf8)   // 0x82 → é in CP437
        cp437.payload = Array("x".utf8)
        var utf8 = ZipBuilder.Entry("ignored")
        utf8.flags = 0x0800
        utf8.nameBytes = Array("café2.txt".utf8)
        utf8.payload = Array("y".utf8)
        var b = ZipBuilder()
        b.entries = [cp437, utf8]
        let (_, written) = try extract(b)
        XCTAssertEqual(written, ["café.txt", "café2.txt"])
    }

    // MARK: Rejections

    func testRejectsPathTraversal() throws {
        var up = ZipBuilder()
        up.entries = [ZipBuilder.stored("../evil.txt", "x")]
        try assertRejected(up, .unsafeEntryPath("../evil.txt"))

        var nested = ZipBuilder()
        nested.entries = [ZipBuilder.stored("bundle/../../evil.txt", "x")]
        try assertRejected(nested, .unsafeEntryPath("bundle/../../evil.txt"))

        var dot = ZipBuilder()
        dot.entries = [ZipBuilder.stored("bundle/./a.txt", "x")]
        try assertRejected(dot, .unsafeEntryPath("bundle/./a.txt"))
    }

    func testRejectsAbsoluteBackslashDriveAndNulNames() throws {
        for name in ["/etc/passwd", "bundle\\evil.txt", "C:/evil.txt", "bundle/a\u{0}b.txt"] {
            var b = ZipBuilder()
            b.entries = [ZipBuilder.stored(name, "x")]
            try assertRejected(b, .unsafeEntryPath(name))
        }
    }

    func testSanitizedComponentsIsTheLexicalGate() throws {
        XCTAssertEqual(try ZipArchiveReader.sanitizedComponents(entryName: "a//b/c.txt/"),
                       ["a", "b", "c.txt"])
        for bad in ["..", "a/../b", "/a", "a\\b", "a\u{0}b", "./a", "", "/"] {
            XCTAssertThrowsError(try ZipArchiveReader.sanitizedComponents(entryName: bad),
                                 "accepted \(bad)")
        }
        XCTAssertTrue(ZipArchiveReader.isIgnored(components: ["__MACOSX", "x"]))
        XCTAssertTrue(ZipArchiveReader.isIgnored(components: ["images", "._0001.jpg"]))
        XCTAssertFalse(ZipArchiveReader.isIgnored(components: ["images", "0001.jpg"]))
    }

    func testResolveKeepsEntriesInsideTheDestination() throws {
        // /private/var vs /var is exactly the asymmetry that makes standardizedFileURL unusable
        // here; the containment check must be indifferent to it.
        let base = URL(fileURLWithPath: "/private/var/folders/xx/out", isDirectory: true)
        let url = try ZipArchiveReader.resolve(destination: base, components: ["bundle", "a.txt"])
        XCTAssertEqual(url.path, "/private/var/folders/xx/out/bundle/a.txt")
    }

    func testRejectsSymlinkAndSpecialEntries() throws {
        var link = ZipBuilder()
        var e = ZipBuilder.stored("bundle/link", "/etc/passwd")
        e.externalAttributes = UInt32(0o120777) << 16          // S_IFLNK
        link.entries = [e]
        try assertRejected(link, .unsafeEntryType("bundle/link"))

        var fifo = ZipBuilder()
        var f = ZipBuilder.stored("bundle/pipe", "")
        f.externalAttributes = UInt32(0o010644) << 16          // S_IFIFO
        fifo.entries = [f]
        try assertRejected(fifo, .unsafeEntryType("bundle/pipe"))
    }

    func testRejectsZip64Sentinels() throws {
        var entrySentinel = ZipBuilder()
        var e = ZipBuilder.stored("bundle/a.txt", "x")
        e.uncompressedSizeOverride = 0xFFFF_FFFF
        entrySentinel.entries = [e]
        try assertRejected(entrySentinel, .unsupportedZip64)

        var eocdSentinel = ZipBuilder()
        eocdSentinel.entries = [ZipBuilder.stored("bundle/a.txt", "x")]
        eocdSentinel.totalEntriesOverride = 0xFFFF
        try assertRejected(eocdSentinel, .unsupportedZip64)

        var offsetSentinel = ZipBuilder()
        offsetSentinel.entries = [ZipBuilder.stored("bundle/a.txt", "x")]
        offsetSentinel.centralOffsetOverride = 0xFFFF_FFFF
        try assertRejected(offsetSentinel, .unsupportedZip64)
    }

    func testRejectsMultiDiskArchives() throws {
        var b = ZipBuilder()
        b.entries = [ZipBuilder.stored("bundle/a.txt", "x")]
        b.diskNumberOverride = 3
        try assertRejected(b, .unsupportedMultiDisk)
    }

    func testRejectsEncryptedEntries() throws {
        var b = ZipBuilder()
        var e = ZipBuilder.stored("bundle/a.txt", "x")
        e.flags = 0x0001
        b.entries = [e]
        try assertRejected(b, .encryptedEntry("bundle/a.txt"))
    }

    func testRejectsUnknownCompressionMethod() throws {
        var b = ZipBuilder()
        var e = ZipBuilder.stored("bundle/a.txt", "x")
        e.method = 12                                          // bzip2 — never produced by iOS
        b.entries = [e]
        try assertRejected(b, .unsupportedMethod(name: "bundle/a.txt", method: 12))
    }

    func testRejectsCRCMismatch() throws {
        var b = ZipBuilder()
        var e = ZipBuilder.stored("bundle/a.txt", "the real payload")
        e.crcOverride = 0xDEAD_BEEF
        b.entries = [e]
        try assertRejected(b, .corruptEntry("bundle/a.txt"))
    }

    func testRejectsStoredSizeMismatch() throws {
        var b = ZipBuilder()
        var e = ZipBuilder.stored("bundle/a.txt", "1234567890")
        e.uncompressedSizeOverride = 4                         // stored ⇒ csize must equal usize
        b.entries = [e]
        try assertRejected(b, .corruptEntry("bundle/a.txt"))
    }

    func testRejectsExpansionBudgetOverrun() throws {
        let big = Array(String(repeating: "A", count: 50_000).utf8)
        var b = ZipBuilder()
        b.entries = [ZipBuilder.deflated("bundle/bomb.bin", big)]
        // 50 KB from a ~120-byte deflate stream: the budget, not the ratio, is what refuses it.
        try assertRejected(b, limits: .init(maxEntries: 10,
                                            maxTotalUncompressedBytes: 1024,
                                            maxEntryUncompressedBytes: 1024),
                           .archiveTooLarge("bundle/bomb.bin expands to 50000 bytes"))

        var many = ZipBuilder()
        many.entries = (0..<5).map { ZipBuilder.stored("bundle/\($0).txt", "x") }
        try assertRejected(many, limits: .init(maxEntries: 2,
                                               maxTotalUncompressedBytes: 1 << 20,
                                               maxEntryUncompressedBytes: 1 << 20),
                           .archiveTooLarge("5 entries"))
    }

    /// A stream that decodes to *more* than the declared size, with a CRC that matches the
    /// truncated prefix — the shape a size check alone cannot see. `compression_decode_buffer`
    /// fills the destination and reports a full count rather than a short one (measured), so with
    /// a destination of exactly `usize` every downstream check passes and a silently truncated file
    /// is written. The reader asks for one byte more than it will accept precisely to catch this.
    func testRejectsAStreamThatDecodesPastTheDeclaredSize() throws {
        let payload = Array(String(repeating: "splatx 0123456789\n", count: 500).utf8)
        var e = ZipBuilder.deflated("bundle/truncated.txt", payload)
        e.uncompressedSizeOverride = 100
        e.crcOverride = ZipCRC32.compute(Array(payload.prefix(100)))   // matches what would be kept
        var b = ZipBuilder()
        b.entries = [e]
        try assertRejected(b, .corruptEntry("bundle/truncated.txt"))
    }

    func testRejectsDuplicateEntryNames() throws {
        var b = ZipBuilder()
        b.entries = [ZipBuilder.stored("bundle/a.txt", "first"),
                     ZipBuilder.stored("bundle/a.txt", "second")]
        try assertRejected(b, .malformed("duplicate entry bundle/a.txt"))
    }

    /// Two names that differ only in case are one file on an APFS volume in its default format, so
    /// the second entry would overwrite the first — the exact substitution the duplicate check
    /// exists to refuse. A byte-wise comparison lets it straight through.
    func testRejectsNamesThatCollideOnACaseInsensitiveVolume() throws {
        var b = ZipBuilder()
        b.entries = [ZipBuilder.stored("bundle/images/0001.JPG", "decoy"),
                     ZipBuilder.stored("bundle/images/0001.jpg", "payload")]
        try assertRejected(b, .malformed("duplicate entry bundle/images/0001.jpg"))

        // …and the same for Unicode composition, which such a volume also folds: "é" spelled
        // precomposed (U+00E9) and decomposed (e + U+0301) name one file.
        var nfc = ZipBuilder.Entry("ignored")
        nfc.flags = 0x0800
        nfc.nameBytes = Array("bundle/caf\u{00E9}.txt".utf8)
        nfc.payload = Array("x".utf8)
        var nfd = ZipBuilder.Entry("ignored")
        nfd.flags = 0x0800
        nfd.nameBytes = Array("bundle/cafe\u{0301}.txt".utf8)
        nfd.payload = Array("y".utf8)
        var u = ZipBuilder()
        u.entries = [nfc, nfd]
        try assertRejected(u, .malformed("duplicate entry bundle/cafe\u{0301}.txt"))
    }

    /// The compressed buffer is the one allocated *first*, and `csize` is free to exceed `usize`.
    /// Before this guard the only thing standing between a declared 2 GB stream and a 2 GB
    /// allocation was the file being big enough to hold it.
    func testRejectsAnOversizedCompressedStreamBeforeAllocatingIt() throws {
        var b = ZipBuilder()
        var e = ZipBuilder.stored("bundle/a.txt", String(repeating: "x", count: 4096))
        e.uncompressedSizeOverride = 8                            // small output, huge input
        b.entries = [e]
        try assertRejected(b, limits: .init(maxEntries: 10,
                                            maxTotalUncompressedBytes: 1 << 20,
                                            maxEntryUncompressedBytes: 1024),
                           .archiveTooLarge("bundle/a.txt reads 4096 compressed bytes"))
    }

    /// The ratio cap, which the cumulative budget does not imply: this entry is inside **every**
    /// other production limit — one entry, 2 MB, far under both budgets — and is still refused,
    /// because a ~2 KB stream asking for 2 MB is not a shape any capture bundle has (measured
    /// worst case over six real bundles: 6.56×).
    ///
    /// The exact byte count of the deflate stream is not asserted: it is whatever the system zlib
    /// produces, and pinning it would turn an OS update into a test failure.
    func testRejectsAnEntryPastTheCompressionRatioCap() throws {
        var b = ZipBuilder()
        b.entries = [ZipBuilder.deflated("bundle/bomb.bin", [UInt8](repeating: 0, count: 2_000_000))]
        let zip = try write(b, as: "\(UUID().uuidString).zip")
        XCTAssertLessThan(try Data(contentsOf: zip).count, 64 << 10)   // the archive itself is tiny
        XCTAssertThrowsError(try ZipArchiveReader.extract(
            zip: zip, to: tmp.appendingPathComponent("out-\(UUID().uuidString)"))) { error in
            guard case .archiveTooLarge(let why)? = error as? ZipArchiveError else {
                return XCTFail("expected archiveTooLarge, got \(error)")
            }
            XCTAssertTrue(why.hasPrefix("bundle/bomb.bin expands to 2000000 bytes from "), why)
            XCTAssertTrue(why.hasSuffix("refusing a compression bomb"), why)
        }
    }

    /// The other side of the same cap, and the reason it carries a flat allowance: repetitive text
    /// deflates hundredfold (132 KB → 438 bytes, measured), so a pure ratio would refuse ordinary
    /// small files. `testExtractsStoredDeflatedAndDirectoryEntries` is exactly that shape and must
    /// keep passing; this states the invariant where it can be seen.
    func testSmallHighlyCompressibleEntriesAreNotBombs() throws {
        var b = ZipBuilder()
        b.entries = [ZipBuilder.deflated(
            "bundle/notes.txt",
            Array(String(repeating: "one repeated line, deflated to nothing\n", count: 3000).utf8))]
        let (_, written) = try extract(b)
        XCTAssertEqual(written, ["bundle/notes.txt"])
    }

    func testRejectsOutOfBoundsLocalHeaderOffset() throws {
        var b = ZipBuilder()
        var e = ZipBuilder.stored("bundle/a.txt", "x")
        e.localOffsetOverride = 0xFFFF_FF00                    // large, but not the ZIP64 sentinel
        b.entries = [e]
        try assertRejected(b, .truncated("bundle/a.txt"))
    }

    func testRejectsCentralDirectoryPastEndOfFile() throws {
        var b = ZipBuilder()
        b.entries = [ZipBuilder.stored("bundle/a.txt", "x")]
        b.centralSizeOverride = 4096
        try assertRejected(b, .truncated("central directory"))
    }

    func testRejectsTruncatedArchiveAndMissingEOCD() throws {
        var b = ZipBuilder()
        b.entries = [ZipBuilder.stored("bundle/a.txt", "x")]
        var bytes = b.build()
        bytes.removeLast(5)                                    // clips the EOCD record itself
        let cut = tmp.appendingPathComponent("cut.zip")
        try bytes.write(to: cut)
        XCTAssertThrowsError(try ZipArchiveReader.extract(
            zip: cut, to: tmp.appendingPathComponent("out1"))) {
            XCTAssertEqual($0 as? ZipArchiveError, .notAZipArchive)
        }

        let junk = tmp.appendingPathComponent("junk.zip")
        try Data(repeating: 0x41, count: 4096).write(to: junk)
        XCTAssertThrowsError(try ZipArchiveReader.extract(
            zip: junk, to: tmp.appendingPathComponent("out2"))) {
            XCTAssertEqual($0 as? ZipArchiveError, .notAZipArchive)
        }

        let tiny = tmp.appendingPathComponent("tiny.zip")
        try Data([0x50, 0x4B]).write(to: tiny)
        XCTAssertThrowsError(try ZipArchiveReader.extract(
            zip: tiny, to: tmp.appendingPathComponent("out3"))) {
            XCTAssertEqual($0 as? ZipArchiveError, .notAZipArchive)
        }
    }

    func testNothingIsWrittenOutsideTheDestination() throws {
        // The escape target is inside the sandboxed temp tree, never a real path — the assertion is
        // that a sibling of the destination stays untouched.
        let sibling = tmp.appendingPathComponent("sibling", isDirectory: true)
        try FileManager.default.createDirectory(at: sibling, withIntermediateDirectories: true)
        var b = ZipBuilder()
        b.entries = [ZipBuilder.stored("../sibling/evil.txt", "x")]
        try assertRejected(b, .unsafeEntryPath("../sibling/evil.txt"))
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: sibling.appendingPathComponent("evil.txt").path))
    }
}
