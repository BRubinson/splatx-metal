import Compression
import Darwin
import Foundation

enum ZipArchiveError: LocalizedError, Equatable {
    case notAZipArchive
    case truncated(String)
    case unreadable(String)
    case unsupportedZip64
    case unsupportedMultiDisk
    case encryptedEntry(String)
    case unsupportedMethod(name: String, method: UInt16)
    case unsafeEntryPath(String)
    case unsafeEntryType(String)
    case corruptEntry(String)
    case archiveTooLarge(String)
    case malformed(String)

    var errorDescription: String? {
        switch self {
        case .notAZipArchive: "Not a .zip archive (no end-of-central-directory record)."
        case .truncated(let what): "The .zip archive is truncated or damaged (\(what))."
        case .unreadable(let what): "Cannot read the .zip archive (\(what))."
        case .unsupportedZip64: "ZIP64 archives are not supported."
        case .unsupportedMultiDisk: "Multi-part .zip archives are not supported."
        case .encryptedEntry(let n): "The .zip archive is encrypted (\(n))."
        case .unsupportedMethod(let n, let m): "Unsupported compression method \(m) in \(n)."
        case .unsafeEntryPath(let n): "The .zip archive contains an unsafe path: \(n)"
        case .unsafeEntryType(let n): "The .zip archive contains a link or device entry: \(n)"
        case .corruptEntry(let n): "Checksum or size mismatch in \(n) — the archive is damaged."
        case .archiveTooLarge(let why): "The .zip archive is too large to expand safely (\(why))."
        case .malformed(let why): "Malformed .zip archive (\(why))."
        }
    }
}

/// A read-only ZIP (PKWARE APPNOTE) extractor for SplatX capture bundles.
///
/// **Why this exists at all** (spec §3.1 — the alternatives were measured, not guessed):
/// AppleArchive handles AAR/AAB only and its 19 header entries name single-stream codecs, never a
/// container; Foundation's only zip-adjacent API (`NSFileCoordinator` `ForUploading`) *creates*
/// archives and cannot read them; libarchive ships a `.tbd` but **no header** in the SDK, so using
/// it means hand-redeclaring an ABI that breaks at runtime rather than at compile time on an OS
/// update. `/usr/bin/unzip` does work under the sandbox, but it **materializes symlink entries**
/// (it has no flag to refuse them) and has no expansion budget at all (measured: 204 KB → 209 MB,
/// silently) — and it would be this repository's first process spawn.
///
/// **`COMPRESSION_ZLIB` is the correct constant, despite the name.** It carries no RFC1950 wrapper:
/// `compression.h` states "The encoded format is the raw DEFLATE format as described in IETF RFC
/// 1951 … `deflateInit2(zstream,5,Z_DEFLATED,-15,8,…)`". Raw DEFLATE is exactly what ZIP method 8
/// stores, so the payload goes into `compression_decode_buffer` untouched. Reading the name alone
/// makes this look like the wrong choice — it is not; `COMPRESSION_ZLIB` is the *only* constant
/// here that works.
///
/// **This reader is deliberately narrow.** Real capture bundles are the mildest possible ZIPs
/// (measured over all 58 entries of a 90 MB bundle **as the phone writes it**: 54 deflated + 4
/// stored directories, zero bytes of extra field, no data descriptors, no ZIP64, ASCII names, GP
/// flags `0x0800` throughout). Everything outside that surface is **rejected explicitly rather than
/// guessed at** — a misread archive that half-extracts is worse than one that refuses. Rejected:
/// ZIP64 (the `0xFFFF`/`0xFFFFFFFF` sentinels), multi-disk, encryption (GP flag bit 0), any method
/// other than 0/8, symlink and special (FIFO/device/socket) entries, absolute/`..`/backslash/NUL
/// paths, duplicate names (folded the way the filesystem folds them), CRC or size mismatches, and
/// archives whose declared expansion exceeds `Limits`.
///
/// **Every size and CRC comes from the central directory, never from the local file header** — and
/// that is a hard requirement, not a preference. The moment a bundle is re-compressed in Finder it
/// is `ditto` that writes it, and `ditto` sets GP flag bit 3 (data descriptor) and leaves the local
/// header's crc/csize/usize **all zero**, with the real values only in the central directory
/// (measured on a Finder-recompressed capture: CD flags `{0x0, 0x8}`, `LFH crc/cs/us = (0, 0, 0)`
/// against `CD = (1210202599, 1488994, 1489313)` for `images/0012.jpg`, and 12 bytes of local extra
/// where the CD has none). Reading those fields from the local header would read zero bytes of
/// payload and fail every such bundle at the CRC. `ZipArchiveReaderTests` pins this with a bit-3
/// archive whose local headers are zeroed.
///
/// **Path containment is purely lexical, on sanitized components** — `URL.standardizedFileURL` and
/// `NSString.standardizingPath` are forbidden here (plan constraint 3). They strip the `/private`
/// prefix asymmetrically, so a naive `hasPrefix` on their output rejects legitimate files under
/// `/tmp` and, with the operands the other way round, lets an escape through. ZIPFoundation shipped
/// exactly that bug (issue #281). Instead an entry name is split into components, `.` and `..` are
/// hard errors, and the result is re-checked as an array prefix against the destination's own
/// components. The **one** thing that is folded away is the empty component — `a//b` and the
/// trailing `/` every directory entry carries — which is safe in a way `.`/`..` folding is not:
/// dropping an empty component removes a separator and can never introduce, rename or re-target a
/// component, so there is no normalization result for a second decoding layer to disagree with. A
/// name that is *nothing but* separators leaves no components at all and is rejected.
///
/// **Memory**: the archive is never loaded as one `Data` (plan constraint 6), and it is read with
/// `pread(2)` into one reusable buffer rather than through `FileHandle` — see `ArchiveFile`, where
/// the measurement that forced that choice is recorded. The peak the header used to claim was not
/// the one it had; these are measured `phys_footprint` peaks, sampled every 1.5 ms:
///
/// | bundle | zip | largest member | before | after |
/// |---|---|---|---|---|
/// | `LiDAR_20260804_153039` | 90.2 MiB | 10.62 MB | 131.5–136.5 MiB | **28.9–32.6 MiB** |
/// | `LiDAR_20260712_233227` | 288.2 MiB | 22.50 MB | 385.4 MiB | **55.2–58.8 MiB** |
///
/// So the peak now tracks the largest *member* (~2.5× it: the compressed buffer, the expanded
/// array, and the allocator's slack) and no longer the archive — the bound `Limits` states is the
/// one that holds. All three extractions used to check this — both bundles and a `ditto`-recompressed
/// copy of the first — were verified byte-identical to `/usr/bin/unzip` and to the source tree.
///
/// **Not restored on purpose**: unix permission bits and timestamps from the external attributes.
/// The attributes are read only to *reject* entry types; honoring the mode would let an archive
/// hand us setuid or non-readable files for no benefit to an image-and-text bundle.
///
/// Original work — written against the public PKWARE APPNOTE format description and Apple's public
/// `Compression` API. No third-party ZIP implementation was read or ported (`docs/provenance.md`).
enum ZipArchiveReader {
    /// The expansion budget. A ZIP declares its uncompressed sizes up front, so the whole budget is
    /// checked while parsing the central directory — **before a single byte is inflated** — which
    /// is what makes it a bomb defense rather than a post-mortem. Injectable so tests can pin the
    /// rejection with a few hundred bytes instead of gigabytes.
    struct Limits {
        var maxEntries: Int
        var maxTotalUncompressedBytes: Int
        var maxEntryUncompressedBytes: Int
        /// Per-entry ceiling on `usize / csize`. The cumulative budget alone leaves a real hole: a
        /// 4 MB archive can still ask for 4 GiB of expansion and lock the import for minutes, and
        /// this window has no cancel. A ratio cap turns "how big is the output" into "how big is
        /// the *input*", which is the number the user can see before they pick the file.
        ///
        /// **64 is ~10× the measured worst case.** Over six real bundles (LiDAR/photo/video, 24 to
        /// 197 entries) the highest per-entry ratio is **6.56×** (`events.ndjson`); the COLMAP text
        /// the whitelist actually keeps tops out at **2.72×** (`points3D.txt`, 6.7 MB → 2.5 MB) and
        /// JPEG frames sit at 1.00–1.02× — deflate on already-compressed pixels. Text is what sets
        /// the ceiling, so the headroom is spent there.
        ///
        /// Defaulted on the property rather than in `.default` so that the tests which inject a
        /// deliberately tiny budget keep getting the production ratio unless they say otherwise.
        var maxCompressionRatio: Int = 64

        /// Expansion allowed per entry *regardless* of ratio, so that a small highly repetitive
        /// file is not mistaken for a bomb. Repetition is not rare and the ratios are startling:
        /// 132 KB of one repeated line deflates to **438 bytes** (301×), 200 KB of zeros to 211
        /// (948×). Without this allowance the cap would refuse files nobody would call a bomb —
        /// including `ZipArchiveReaderTests`' own happy path, which is how the number was chosen:
        /// a megabyte of expansion is beneath notice for an importer that copies hundreds.
        ///
        /// So the guarantee is "no entry expands past 64× its input **plus** 1 MiB", not a pure
        /// ratio. The many-small-entries shape is unaffected by it and stays governed by
        /// `maxTotalUncompressedBytes` alone; what this closes is the single huge entry, where the
        /// ratio is the only thing that separates a 4 MB archive from a 4 GiB expansion.
        static let compressionRatioSlack = 1 << 20

        /// Sized for real capture bundles with headroom: the largest measured archive is 288 MB and
        /// its largest member is a 10.6 MB `pointcloud.ply`. Anything an order of magnitude past
        /// that is not a capture bundle.
        static let `default` = Limits(maxEntries: 50_000,
                                      maxTotalUncompressedBytes: 4 << 30,
                                      maxEntryUncompressedBytes: 512 << 20)
    }

    /// Expands `zip` into `destination` (created if needed) and returns the relative paths actually
    /// written, in central-directory order. Directory entries are created but not returned.
    ///
    /// Blocking and GPU-free, but a 90 MB bundle takes seconds — call it off the main thread.
    @discardableResult
    static func extract(zip: URL, to destination: URL,
                        limits: Limits = .default) throws -> [String] {
        let file = try ArchiveFile(zip: zip)
        defer { file.close() }
        guard file.size >= 22 else { throw ZipArchiveError.notAZipArchive }

        let eocd = try locateEndOfCentralDirectory(file)
        let cd = try file.read(at: eocd.offset, count: Int(eocd.size), what: "central directory")
        let entries = try parseCentralDirectory(cd, declaredCount: eocd.count, limits: limits)

        let fm = FileManager.default
        try fm.createDirectory(at: destination, withIntermediateDirectories: true)
        var written: [String] = []
        for entry in entries where !entry.ignored {
            let out = try resolve(destination: destination, components: entry.components)
            if entry.isDirectory {
                try fm.createDirectory(at: out, withIntermediateDirectories: true)
                continue
            }
            var bytes = try payload(file, entry: entry)
            try fm.createDirectory(at: out.deletingLastPathComponent(),
                                   withIntermediateDirectories: true)
            // `Data(bytes)` would copy the whole entry a second time purely to hand it to `write`;
            // on the 302 MB reference bundle that copy alone was 26.8 MiB of peak (87.3 → 60.5 MiB
            // measured). The `Data` here is a synchronous, read-only view of an array that outlives
            // it — `write(to:)` returns before the buffer does. The pointer is taken mutably only
            // because that is the signature `bytesNoCopy` has; nothing writes through it.
            // An empty array has no base address to hand over, so the zero-byte entry takes the
            // plain path. No measured bundle contains one, but it is a legal ZIP entry and
            // `testEmptyFileEntryRoundTrips` holds this branch open.
            if bytes.isEmpty {
                try Data().write(to: out)
            } else {
                try bytes.withUnsafeMutableBufferPointer { buf in
                    try Data(bytesNoCopy: buf.baseAddress!, count: buf.count, deallocator: .none)
                        .write(to: out)
                }
            }
            written.append(entry.components.joined(separator: "/"))
        }
        return written
    }

    // MARK: - Entry model

    struct Entry {
        var name: String            // as stored, for diagnostics
        var components: [String]    // sanitized — no "", ".", "..", "/", "\", NUL
        var ignored: Bool           // __MACOSX / AppleDouble — validated, then dropped
        var isDirectory: Bool
        var method: UInt16
        var crc: UInt32
        var compressedSize: UInt64
        var uncompressedSize: UInt64
        var localHeaderOffset: UInt64
    }

    // MARK: - Path sanitation

    /// Splits an entry name into path components, rejecting anything that could name a file outside
    /// the destination. There is intentionally **no normalization pass**: `.` and `..` are errors,
    /// not inputs to be folded away, because every historical zip-slip is a normalizer that a
    /// second decoding layer (percent-escapes, UTF-8 overlongs, `/private`) disagreed with.
    static func sanitizedComponents(entryName: String) throws -> [String] {
        func reject() -> ZipArchiveError { .unsafeEntryPath(entryName) }
        guard !entryName.isEmpty else { throw ZipArchiveError.malformed("empty entry name") }
        // Backslash: Windows separators would survive as a literal component here and become a
        // separator again on any consumer that splits on "\". NUL truncates C string APIs.
        guard !entryName.contains("\\"), !entryName.contains("\0") else { throw reject() }
        guard !entryName.hasPrefix("/") else { throw reject() }
        // "C:/x" and "C:x" — a drive-relative name has no meaning here and only invites a consumer
        // to re-interpret the colon.
        if let colon = entryName.firstIndex(of: ":"),
           entryName.distance(from: entryName.startIndex, to: colon) <= 2 { throw reject() }

        var components: [String] = []
        for part in entryName.split(separator: "/", omittingEmptySubsequences: false) {
            let s = String(part)
            if s.isEmpty { continue }          // "a//b" and the trailing "/" of directory entries
            guard s != ".", s != ".." else { throw reject() }
            components.append(s)
        }
        guard !components.isEmpty else { throw reject() }
        return components
    }

    /// Finder re-compression (`ditto -c -k --keepParent`, which is what "Compress" runs) injects an
    /// `__MACOSX/` shadow tree and `._name` AppleDouble side files. They are resource forks, never
    /// bundle content, and a `._0001.jpg` sitting in `images/` would be counted as an image. They
    /// are still fully validated above — only extraction is skipped.
    static func isIgnored(components: [String]) -> Bool {
        components.first == "__MACOSX" || (components.last?.hasPrefix("._") ?? false)
    }

    /// Joins sanitized components onto the destination and re-checks containment as an **array
    /// prefix** — no string normalization anywhere (see the type header on `standardizedFileURL`).
    /// Redundant by construction; kept because the cost of the invariant silently lapsing is a
    /// write outside the project folder.
    static func resolve(destination: URL, components: [String]) throws -> URL {
        let base = destination.pathComponents
        let url = components.reduce(destination) { $0.appendingPathComponent($1) }
        let full = url.pathComponents
        guard full.count == base.count + components.count,
              Array(full.prefix(base.count)) == base,
              Array(full.suffix(components.count)) == components else {
            throw ZipArchiveError.unsafeEntryPath(components.joined(separator: "/"))
        }
        return url
    }

    // MARK: - End of central directory

    private static func locateEndOfCentralDirectory(
        _ file: ArchiveFile
    ) throws -> (offset: UInt64, size: UInt64, count: Int) {
        let fileSize = file.size
        // The EOCD is last but a variable-length comment can follow it: 22 bytes of record plus at
        // most 0xFFFF of comment bounds the search window.
        let tailLength = Int(min(fileSize, UInt64(22 + 0xFFFF)))
        let tailStart = fileSize - UInt64(tailLength)
        let tail = try file.read(at: tailStart, count: tailLength,
                                 what: "end of central directory")

        var i = tailLength - 22
        while i >= 0 {
            defer { i -= 1 }
            guard u32(tail, i) == 0x0605_4b50 else { continue }
            // A four-byte signature can appear inside a comment or a stored payload. The record is
            // only real if its comment length accounts for exactly the rest of the file.
            let commentLength = Int(u16(tail, i + 20))
            guard tailLength - (i + 22) == commentLength else { continue }

            let disk = u16(tail, i + 4), cdDisk = u16(tail, i + 6)
            let entriesHere = u16(tail, i + 8), entriesTotal = u16(tail, i + 10)
            let cdSize = u32(tail, i + 12), cdOffset = u32(tail, i + 16)
            // ZIP64 puts sentinels in the classic record and the real values in an extended record
            // we do not parse. Reading the sentinels as literals would seek to entry 65535 at
            // offset 4 GiB — refuse instead of misinterpreting (spec §3.3).
            guard disk != 0xFFFF, cdDisk != 0xFFFF, entriesHere != 0xFFFF, entriesTotal != 0xFFFF,
                  cdSize != 0xFFFF_FFFF, cdOffset != 0xFFFF_FFFF else {
                throw ZipArchiveError.unsupportedZip64
            }
            guard disk == 0, cdDisk == 0, entriesHere == entriesTotal else {
                throw ZipArchiveError.unsupportedMultiDisk
            }
            guard UInt64(cdOffset) + UInt64(cdSize) <= fileSize else {
                throw ZipArchiveError.truncated("central directory")
            }
            return (UInt64(cdOffset), UInt64(cdSize), Int(entriesTotal))
        }
        throw ZipArchiveError.notAZipArchive
    }

    // MARK: - Central directory

    /// Only the mode bits of the external attributes are consulted, and only to reject. Declared as
    /// `UInt32` rather than reusing Darwin's `S_IF*` because those are `mode_t` (`UInt16`) while the
    /// field they must be compared against is the high half of a 32-bit word.
    private static let sIFMT: UInt32 = 0o170000
    private static let sIFREG: UInt32 = 0o100000
    private static let sIFDIR: UInt32 = 0o040000

    private static func parseCentralDirectory(
        _ cd: [UInt8], declaredCount: Int, limits: Limits
    ) throws -> [Entry] {
        guard declaredCount <= limits.maxEntries else {
            throw ZipArchiveError.archiveTooLarge("\(declaredCount) entries")
        }
        var entries: [Entry] = []
        var seen = Set<String>()
        var totalUncompressed = 0
        var i = 0
        for _ in 0..<declaredCount {
            guard i + 46 <= cd.count else { throw ZipArchiveError.truncated("central directory") }
            guard u32(cd, i) == 0x0201_4b50 else {
                throw ZipArchiveError.malformed("central directory header signature")
            }
            let flags = u16(cd, i + 8)
            let method = u16(cd, i + 10)
            let crc = u32(cd, i + 16)
            let csize = u32(cd, i + 20)
            let usize = u32(cd, i + 24)
            let nameLength = Int(u16(cd, i + 28))
            let extraLength = Int(u16(cd, i + 30))
            let commentLength = Int(u16(cd, i + 32))
            let diskStart = u16(cd, i + 34)
            let externalAttributes = u32(cd, i + 38)
            let localOffset = u32(cd, i + 42)
            guard i + 46 + nameLength + extraLength + commentLength <= cd.count else {
                throw ZipArchiveError.truncated("central directory")
            }
            let nameBytes = Array(cd[(i + 46)..<(i + 46 + nameLength)])
            i += 46 + nameLength + extraLength + commentLength

            // GP flag bit 11 (EFS) is the only in-band statement about the name encoding; without
            // it the APPNOTE mandates CP437, and mis-decoding a name as UTF-8 would either throw
            // away a byte or produce a different file name than the archive intends.
            let name = (flags & 0x0800) != 0
                ? (String(bytes: nameBytes, encoding: .utf8) ?? decodeCP437(nameBytes))
                : decodeCP437(nameBytes)

            guard (flags & 0x0001) == 0 else { throw ZipArchiveError.encryptedEntry(name) }
            guard csize != 0xFFFF_FFFF, usize != 0xFFFF_FFFF,
                  localOffset != 0xFFFF_FFFF, diskStart != 0xFFFF else {
                throw ZipArchiveError.unsupportedZip64
            }
            guard diskStart == 0 else { throw ZipArchiveError.unsupportedMultiDisk }

            // Everything is validated, including entries that will be ignored: an archive that
            // hides a symlink or a ZIP64 sentinel under __MACOSX/ is malformed regardless of
            // whether we would have written that entry out.
            //
            // The "creator system" byte of version-made-by is deliberately *not* consulted first.
            // It would say whether the high half is a unix mode, but a hostile writer picks that
            // byte freely; treating any populated S_IFMT nibble as authoritative rejects strictly
            // more. DOS-produced archives leave the high half zero, so they skip the check anyway.
            let mode = externalAttributes >> 16
            var isDirectory = name.hasSuffix("/")
            if (mode & sIFMT) != 0 {
                switch mode & sIFMT {
                case sIFREG: break
                case sIFDIR: isDirectory = true
                default: throw ZipArchiveError.unsafeEntryType(name)   // S_IFLNK, FIFO, device, …
                }
            }

            guard Int(usize) <= limits.maxEntryUncompressedBytes else {
                throw ZipArchiveError.archiveTooLarge("\(name) expands to \(usize) bytes")
            }
            // The compressed side needs its own ceiling, and it is not implied by the one above:
            // the buffer that is allocated *first* is the compressed one, and `csize` is free to
            // exceed `usize` (a hostile writer can declare a 2 GB stream for a 1 KB file, and the
            // only thing that would have refused it is running out of memory).
            guard Int(csize) <= limits.maxEntryUncompressedBytes else {
                throw ZipArchiveError.archiveTooLarge("\(name) reads \(csize) compressed bytes")
            }
            // Ratio cap — see `Limits.maxCompressionRatio`. Checked here, with the rest of the
            // budget, so a bomb is refused from the central directory alone, before any inflation.
            guard Int(usize) <= Int(csize) * limits.maxCompressionRatio
                + Limits.compressionRatioSlack else {
                throw ZipArchiveError.archiveTooLarge(
                    "\(name) expands to \(usize) bytes from \(csize) — refusing a compression bomb")
            }
            totalUncompressed += Int(usize)
            guard totalUncompressed <= limits.maxTotalUncompressedBytes else {
                throw ZipArchiveError.archiveTooLarge("expands past \(limits.maxTotalUncompressedBytes) bytes")
            }

            let components = try sanitizedComponents(entryName: name)
            let key = components.joined(separator: "/")
            // Two entries claiming one path means the second silently overwrites the first — the
            // classic way to smuggle content past a reviewer who read the listing.
            //
            // The key is folded the way the *destination filesystem* folds names, not the way Swift
            // compares strings: an APFS volume in its default (case-insensitive) format also folds
            // Unicode composition, so `a.txt`/`A.txt` and NFC/NFD spellings of one name are one
            // file there and the overwrite this guard exists to stop would happen behind it. Case
            // folding is not exactly the filesystem's own, and where the two differ this rejects an
            // archive the filesystem would have kept apart — the safe direction, and no capture
            // bundle has ever contained two names that differ only in case.
            guard seen.insert(key.precomposedStringWithCanonicalMapping.lowercased()).inserted
            else {
                throw ZipArchiveError.malformed("duplicate entry \(key)")
            }
            entries.append(Entry(name: name, components: components,
                                 ignored: isIgnored(components: components),
                                 isDirectory: isDirectory, method: method, crc: crc,
                                 compressedSize: UInt64(csize), uncompressedSize: UInt64(usize),
                                 localHeaderOffset: UInt64(localOffset)))
        }
        return entries
    }

    // MARK: - Payload

    private static func payload(_ file: ArchiveFile, entry: Entry) throws -> [UInt8] {
        // The local file header is re-parsed rather than trusting the central directory's name and
        // extra lengths: writers are free to make them differ (a ZIP64 or UT extra field is routine
        // in one and absent in the other — `ditto` writes 12 bytes of local extra against a central
        // 0), and the payload starts after the *local* copies. Using the central lengths reads the
        // wrong offset and fails at the CRC — or worse, does not. Only the two lengths are taken
        // from here; crc/csize/usize come from the central directory (see the type header on GP
        // flag bit 3, where the local copies are all zero).
        let lfh = try file.read(at: entry.localHeaderOffset, count: 30, what: entry.name)
        guard u32(lfh, 0) == 0x0403_4b50 else {
            throw ZipArchiveError.malformed("local header signature for \(entry.name)")
        }
        let nameLength = UInt64(u16(lfh, 26)), extraLength = UInt64(u16(lfh, 28))
        let start = entry.localHeaderOffset + 30 + nameLength + extraLength

        // The compressed bytes land in the archive's shared buffer and are consumed inside this
        // closure — nothing but the expanded result outlives it, which is what keeps the footprint
        // flat across a 200-entry archive (`ArchiveFile`).
        let expanded = try file.withPayload(
            at: start, count: Int(entry.compressedSize), what: entry.name
        ) { compressed -> [UInt8] in
            switch entry.method {
            case 0:
                guard entry.compressedSize == entry.uncompressedSize else {
                    throw ZipArchiveError.corruptEntry(entry.name)
                }
                return Array(compressed)
            case 8:
                return try inflate(compressed, expected: Int(entry.uncompressedSize),
                                   name: entry.name)
            default:
                throw ZipArchiveError.unsupportedMethod(name: entry.name, method: entry.method)
            }
        }
        guard expanded.count == Int(entry.uncompressedSize) else {
            throw ZipArchiveError.corruptEntry(entry.name)
        }
        // The size can match on a stream that decoded into garbage; the CRC is the only end-to-end
        // statement that what we wrote is what was archived.
        guard ZipCRC32.compute(expanded) == entry.crc else {
            throw ZipArchiveError.corruptEntry(entry.name)
        }
        return expanded
    }

    /// Raw DEFLATE (ZIP method 8) → bytes, into a buffer **one byte past** the declared size.
    ///
    /// That one byte is the whole guard, and it was measured rather than assumed: with a
    /// destination of exactly `expected`, `compression_decode_buffer` given a stream that produces
    /// more writes what fits and returns `dst_size` — a *full* count, not a short one (probed:
    /// 9000-byte payload, destinations of 8999/100/1 all returned exactly the destination size).
    /// So `produced == expected` would have accepted a silently truncated decode, and the CRC alone
    /// would have caught it. Asking for `expected + 1` makes over-production visible as
    /// `produced == expected + 1` and keeps the failure at this layer, where the entry name is.
    /// The CRC below remains the end-to-end statement; this is the local one.
    private static func inflate(_ source: UnsafeBufferPointer<UInt8>, expected: Int,
                                name: String) throws -> [UInt8] {
        if expected == 0 { return [] }
        guard let src = source.baseAddress, !source.isEmpty else {
            throw ZipArchiveError.corruptEntry(name)
        }
        var out = [UInt8](repeating: 0, count: expected + 1)
        let produced = out.withUnsafeMutableBufferPointer { dst in
            compression_decode_buffer(dst.baseAddress!, expected + 1,
                                      src, source.count, nil, COMPRESSION_ZLIB)
        }
        guard produced == expected else { throw ZipArchiveError.corruptEntry(name) }
        out.removeLast()
        return out
    }

    // MARK: - Bounded random access

    /// The open archive: one file descriptor plus one reusable payload buffer. Every read in this
    /// file goes through it, so the file-size bound is stated once instead of at each of the dozen
    /// offsets a ZIP asks us to trust.
    ///
    /// **`pread(2)` rather than `FileHandle.read(upToCount:)`, and that is a measurement, not a
    /// preference.** `FileHandle`'s reads grow the process footprint by the number of bytes read,
    /// not by the size of the buffers alive at any one moment: expanding the 90.2 MiB reference
    /// bundle — largest member 10.6 MB — peaked at **131.5 MiB**, and the 302.2 MB bundle at
    /// 385.4 MiB, i.e. the peak tracked the *archive*. Reading the same file twice in one process
    /// grew it again by the same amount, so it is not the page cache; `close()` and
    /// `malloc_zone_pressure_relief` did not return it either. The same loop over the same bytes
    /// with `pread` into one reused buffer stayed at 3.2 MiB across both passes. Control runs ruled
    /// out the rest of the pipeline: 193 `[UInt8]` allocate/free cycles cost 3.3 MB and 193
    /// `Data.write` calls 4.9 MB, neither growing.
    ///
    /// The descriptor's lifetime is the extraction: opened by `init`, closed by `close()` from the
    /// `defer` in `extract`. `FileHandle` is kept as the *opener* only — it is what turns a missing
    /// or unreadable file into a Cocoa error with a real message, which `open(2)` + `errno` would
    /// have had to reinvent — and nothing is ever read through it.
    private final class ArchiveFile {
        let size: UInt64
        private let handle: FileHandle
        /// Reused across entries. Grows to the largest payload seen and is never shrunk, so the
        /// steady state is one allocation of `max(csize)`.
        private var scratch: [UInt8] = []

        init(zip: URL) throws {
            handle = try FileHandle(forReadingFrom: zip)
            size = try handle.seekToEnd()
        }

        func close() { try? handle.close() }

        /// Reads exactly `count` bytes at `offset` into a fresh array — for the small structural
        /// reads (EOCD tail, central directory, local file headers) whose results are kept.
        func read(at offset: UInt64, count: Int, what: String) throws -> [UInt8] {
            try bound(offset: offset, count: count, what: what)
            if count == 0 { return [] }
            var out = [UInt8](repeating: 0, count: count)
            try out.withUnsafeMutableBufferPointer {
                try readExactly($0.baseAddress!, count: count, at: offset, what: what)
            }
            return out
        }

        /// Reads exactly `count` bytes at `offset` into the **shared** buffer and hands it to
        /// `body`. The buffer is overwritten by the next call, so `body` must not let it escape —
        /// this is the one place a payload-sized allocation would otherwise be made per entry.
        func withPayload<R>(at offset: UInt64, count: Int, what: String,
                            _ body: (UnsafeBufferPointer<UInt8>) throws -> R) throws -> R {
            try bound(offset: offset, count: count, what: what)
            if count == 0 { return try body(UnsafeBufferPointer(start: nil, count: 0)) }
            if scratch.count < count { scratch = [UInt8](repeating: 0, count: count) }
            return try scratch.withUnsafeMutableBufferPointer { buf in
                let base = buf.baseAddress!
                try readExactly(base, count: count, at: offset, what: what)
                return try body(UnsafeBufferPointer(start: base, count: count))
            }
        }

        /// The file-size bound, stated once. `count` is `Int` and every caller derives it from a
        /// 32-bit ZIP field, so the arithmetic cannot overflow.
        private func bound(offset: UInt64, count: Int, what: String) throws {
            guard count >= 0, offset <= size, UInt64(count) <= size - offset else {
                throw ZipArchiveError.truncated(what)
            }
        }

        /// `pread` may return a short count for reasons that are not EOF, and is interruptible, so
        /// the loop is mandatory — a single call that "usually" fills the buffer would corrupt an
        /// entry rarely and silently. A zero return is a real EOF earlier than `size` promised,
        /// which is truncation; any other error is an I/O failure and says so.
        private func readExactly(_ dst: UnsafeMutablePointer<UInt8>, count: Int,
                                 at offset: UInt64, what: String) throws {
            var done = 0
            while done < count {
                let n = pread(handle.fileDescriptor, dst + done, count - done,
                              off_t(offset) + off_t(done))
                if n > 0 { done += n; continue }
                if n == 0 { throw ZipArchiveError.truncated(what) }
                if errno == EINTR { continue }
                throw ZipArchiveError.unreadable(what)
            }
        }
    }

    private static func u16(_ b: [UInt8], _ i: Int) -> UInt16 {
        UInt16(b[i]) | (UInt16(b[i + 1]) << 8)
    }

    private static func u32(_ b: [UInt8], _ i: Int) -> UInt32 {
        UInt32(b[i]) | (UInt32(b[i + 1]) << 8) | (UInt32(b[i + 2]) << 16) | (UInt32(b[i + 3]) << 24)
    }

    // MARK: - CP437

    /// The APPNOTE default name encoding for entries without the EFS flag. Only the high half needs
    /// a table; 0x00–0x7F is ASCII.
    private static func decodeCP437(_ bytes: [UInt8]) -> String {
        var s = ""
        s.reserveCapacity(bytes.count)
        for b in bytes {
            if b < 0x80 { s.unicodeScalars.append(Unicode.Scalar(b)) }
            else { s.unicodeScalars.append(Unicode.Scalar(cp437High[Int(b) - 0x80])!) }
        }
        return s
    }

    private static let cp437High: [UInt32] = [
        0x00C7, 0x00FC, 0x00E9, 0x00E2, 0x00E4, 0x00E0, 0x00E5, 0x00E7,
        0x00EA, 0x00EB, 0x00E8, 0x00EF, 0x00EE, 0x00EC, 0x00C4, 0x00C5,
        0x00C9, 0x00E6, 0x00C6, 0x00F4, 0x00F6, 0x00F2, 0x00FB, 0x00F9,
        0x00FF, 0x00D6, 0x00DC, 0x00A2, 0x00A3, 0x00A5, 0x20A7, 0x0192,
        0x00E1, 0x00ED, 0x00F3, 0x00FA, 0x00F1, 0x00D1, 0x00AA, 0x00BA,
        0x00BF, 0x2310, 0x00AC, 0x00BD, 0x00BC, 0x00A1, 0x00AB, 0x00BB,
        0x2591, 0x2592, 0x2593, 0x2502, 0x2524, 0x2561, 0x2562, 0x2556,
        0x2555, 0x2563, 0x2551, 0x2557, 0x255D, 0x255C, 0x255B, 0x2510,
        0x2514, 0x2534, 0x252C, 0x251C, 0x2500, 0x253C, 0x255E, 0x255F,
        0x255A, 0x2554, 0x2569, 0x2566, 0x2560, 0x2550, 0x256C, 0x2567,
        0x2568, 0x2564, 0x2565, 0x2559, 0x2558, 0x2552, 0x2553, 0x256B,
        0x256A, 0x2518, 0x250C, 0x2588, 0x2584, 0x258C, 0x2590, 0x2580,
        0x03B1, 0x00DF, 0x0393, 0x03C0, 0x03A3, 0x03C3, 0x00B5, 0x03C4,
        0x03A6, 0x0398, 0x03A9, 0x03B4, 0x221E, 0x03C6, 0x03B5, 0x2229,
        0x2261, 0x00B1, 0x2265, 0x2264, 0x2320, 0x2321, 0x00F7, 0x2248,
        0x00B0, 0x2219, 0x00B7, 0x221A, 0x207F, 0x00B2, 0x25A0, 0x00A0,
    ]
}

/// CRC-32 (IEEE 802.3, reflected, polynomial 0xEDB88320) — the checksum a ZIP central directory
/// carries for every entry.
///
/// Hand-rolled rather than calling zlib's `crc32()`: libz is linked into the app (via SplatXSpz's
/// `.linkedLibrary("z")`), but there is no system Swift module for it, so reaching it would mean
/// adding a module map or a bridging declaration to the app target for forty lines of table lookup.
/// `ZipArchiveReaderTests` pins the standard "123456789" → 0xCBF43926 vector so the table cannot
/// silently rot.
enum ZipCRC32 {
    private static let table: [UInt32] = (0..<256).map { i -> UInt32 in
        var c = UInt32(i)
        for _ in 0..<8 { c = (c & 1) != 0 ? (0xEDB8_8320 ^ (c >> 1)) : (c >> 1) }
        return c
    }

    static func compute(_ bytes: [UInt8]) -> UInt32 {
        var c: UInt32 = 0xFFFF_FFFF
        for b in bytes { c = table[Int((c ^ UInt32(b)) & 0xFF)] ^ (c >> 8) }
        return c ^ 0xFFFF_FFFF
    }
}
