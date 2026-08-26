import Foundation

/// Whether a capture bundle brought its own camera poses.
///
/// Deliberately two cases, not three. The iOS app has three capture modes (`photo`/`video`/
/// `lidar`), but `video` writes a tree byte-identical in shape to `photo` — no `sparse/`, no
/// `pointcloud.ply`, `intrinsics`/`pointCount` forced to null — and differs only in the folder
/// name and a manifest string (spec §1). The only distinction that changes what the importer must
/// do is whether poses came along, so that is the only distinction this type makes.
enum CaptureBundleKind: Equatable {
    /// `sparse/0` holds a complete COLMAP model — LiDAR. The project opens `.aligned`; re-running
    /// SfM would throw the captured geometry away (the bundle README says so in as many words).
    case posed
    /// No usable model — photo/video. The project opens `.imported` and has to be aligned.
    case unposed
}

struct CaptureBundleLayout: Equatable {
    /// The directory that actually holds `images/` — the selection itself, or one level down
    /// through the ZIP's wrapper directory.
    let root: URL
    let kind: CaptureBundleKind
    /// `splatx.json`'s `mode`, when it could be read. **Display label only** — see the type
    /// header. `nil` is an ordinary, fully supported outcome, not a degraded one.
    let mode: String?
}

enum CaptureBundleError: LocalizedError, Equatable {
    /// Nothing here looks like a capture: no `images/`, and no single wrapper directory holding
    /// one. The payload is the folder name that was searched, so the message names what the user
    /// actually picked.
    case missingImagesDirectory(String)
    /// Several directory children and no `images/` — the wrapper rule cannot be applied without
    /// guessing which child is the capture, and guessing wrong imports someone else's folder.
    case ambiguousRoot([String])

    var errorDescription: String? {
        switch self {
        case .missingImagesDirectory(let name):
            "No images/ folder in \(name) — this is not a SplatX capture bundle."
        case .ambiguousRoot(let names):
            "Cannot tell which folder is the capture — several candidates: "
                + names.joined(separator: ", ")
        }
    }
}

/// Recognizes a SplatX iOS capture bundle: where its root is, whether it carries on-device poses,
/// and — as a label only — which mode produced it.
///
/// **The manifest is not a contract; the structure is.** The capture app's own design document is
/// explicit (splatx-ios `docs/superpowers/specs/2026-06-25-splatx-ios-capture-app-design.md:110`,
/// §1.9 / OQ-0):
///
/// > `splatx.json` is SplatX-INTERNAL metadata and is **NOT part of the Studio contract**. Studio
/// > reads only `images/` and `sparse/0/`; it never reads, requires, or is bound by any
/// > `splatx.json` field. … its fields may evolve freely across app versions.
///
/// So posed/unposed is decided by looking for `sparse/0`, never by reading `mode`, and **a bundle
/// with no `splatx.json` at all imports successfully**. Gating on the manifest would bind this
/// importer to a schema whose owner has reserved the right to change it — and it would fail on
/// bundles that exist today: zip creation on the phone is best-effort, the bundle folder itself is
/// exposed through Files/AirDrop, and on-device training adds a `trained/` subtree after the fact
/// (spec §1.2), so "a folder the user assembled by hand, minus a file" is a real shape.
///
/// **`.posed` does not mean "needs its up axis fixed".** Measured over all 47 poses of
/// `LiDAR_20260804_153039`: the bundle world is +Y up and its mean camera up sits **177.89°** from
/// the COLMAP gravity up `(0,−1,0)` that `GroundLevel` writes — i.e. essentially inverted relative
/// to this app's own convention. It is deliberately **not** flipped (spec §2.3): the viewer derives
/// its orbit up from the cameras through `SceneUp.estimate`, so the scene stands upright whatever
/// the world sign is, and the one argument for flipping ("external viewers would show it upside
/// down") was an inference nobody ever measured. `images.txt`/`points3D.txt`/`pointcloud.ply` are
/// copied byte-for-byte. Nothing here touches coordinates, and nothing downstream of it should
/// start to — a "fix" applied on top of `SceneUp` double-corrects and lands the scene back where
/// it started, only now with the file no longer matching what the phone captured.
///
/// **Directory access is injected**, following `GroundLevel`'s pure estimators and
/// `RunPropertiesView.timelapseState(fileExists:)`. The entire decision tree is then exercised over
/// a synthetic tree — no temp files, and (decisive on this machine) no code path that could ever
/// reach ~/Desktop or ~/Downloads under TCC.
enum CaptureBundle {
    static let imagesDirectoryName = "images"
    static let manifestName = "splatx.json"

    /// Directory children that never count as the bundle wrapper. `__MACOSX` is Finder's
    /// re-compression shadow tree (`ZipArchiveReader.isIgnored` drops it on the zip path, but a
    /// folder that was unzipped elsewhere and then dragged in still carries it); dot-prefixed names
    /// are OS bookkeeping (`.Trashes`, `.fseventsd`). Excluding them can only ever turn an
    /// "ambiguous" verdict into a successful descent — the real bundle root is recognized by
    /// `images/` before this rule is consulted at all, so no exclusion can mis-root a capture.
    static func isIgnoredDirectory(named name: String) -> Bool {
        name == "__MACOSX" || name.hasPrefix(".")
    }

    /// The COLMAP model triples, bin preferred over txt — the same rule and the same order as
    /// `ImportService.assembleColmap`, because whatever this classifies as `.posed` is what that
    /// path has to be able to copy. A *partial* set (a stray `cameras.txt`, or a bin/txt mixture)
    /// is `.unposed` on purpose: an incomplete model cannot be loaded, so calling it posed would
    /// skip alignment and hand the trainer a bundle it cannot open.
    static let colmapSets = [["cameras.bin", "images.bin", "points3D.bin"],
                             ["cameras.txt", "images.txt", "points3D.txt"]]

    /// The complete set present in a `sparse/0` listing, or `nil` — **the** posed/unposed predicate.
    ///
    /// One function rather than two matching implementations, because "written to match exactly"
    /// did not survive contact: `kind` compared names out of a directory listing while
    /// `ImportService.assembleCaptureBundle` asked `FileManager.fileExists`, and those two disagree
    /// on an APFS volume in its default case-insensitive format — `fileExists` would find
    /// `Cameras.bin` for `cameras.bin`, a listing comparison never does. A bundle spelled that way
    /// would then be assembled *with* a model but opened `.imported`, or the reverse. Neither
    /// verdict is dangerous on its own (both fail toward "align it again"), and no producer emits
    /// such a name — but a predicate that two call sites have to keep identical by hand is one
    /// edit away from being neither, so there is now nothing to keep in sync.
    ///
    /// Case-sensitive matching is the deliberate half of that: the C loader builds these paths as
    /// exact lowercase literals (`ColmapLoader.cpp:325`), so a differently-cased set only opens by
    /// accident of the volume's case folding and does not open at all on a case-sensitive one.
    /// Refusing to call it a model is the outcome that survives being moved to another disk.
    static func completeColmapSet(among names: [String]) -> [String]? {
        let present = Set(names)
        return colmapSets.first { $0.allSatisfy(present.contains) }
    }

    /// Resolves the bundle root, validates it, and reads the display label.
    ///
    /// Order matters: root first (everything else is relative to it), then the image count — the
    /// failure a user is most likely to hit and the one whose message must not be about `sparse/`.
    /// Classification and the manifest come last and cannot fail.
    ///
    /// `displayName` exists because the directory being searched is not always one the user has a
    /// name for: on the zip path it is a scratch directory inside a UUID workspace, and naming it
    /// in the failure produced "No images/ folder in **expanded**". The caller passes the name the
    /// user actually chose. It is a label only — nothing is resolved from it.
    static func locate(at selected: URL,
                       displayName: String? = nil,
                       isDirectory: (URL) -> Bool = CaptureBundle.fileSystemIsDirectory,
                       contents: (URL) -> [String] = CaptureBundle.fileSystemContents,
                       readManifest: (URL) -> Data? = CaptureBundle.fileSystemManifest)
        throws -> CaptureBundleLayout {
        let root = try resolveRoot(at: selected, displayName: displayName,
                                   isDirectory: isDirectory, contents: contents)
        let images = imageNames(
            among: contents(root.appendingPathComponent(imagesDirectoryName, isDirectory: true)))
        // Reuses ImportService's error rather than inventing a capture-specific one: it is the
        // same fact ("2 images minimum"), it already reads well, and the import sheet has one
        // message to present for every source kind.
        guard images.count >= 2 else { throw ImportError.notEnoughImages(found: images.count) }
        return CaptureBundleLayout(
            root: root,
            kind: kind(root: root, contents: contents),
            mode: manifestMode(readManifest(root.appendingPathComponent(manifestName))))
    }

    /// `images/` present → the selection *is* the root. Otherwise exactly one (non-ignored)
    /// directory child → descend into it.
    ///
    /// The single-child rule exists because iOS zips with `shouldKeepParent: true`, so an expanded
    /// archive is always `<folderName>/…` (spec §1.2 — an explicit argument, not an accident). It
    /// is one level and never recursive: descending further would let this happily "find" a capture
    /// several directories deep inside something the user did not mean to import.
    ///
    /// Checking `images/` **first** is what makes the rule safe against the extra directories a
    /// real bundle root can hold (`sparse/`, and `trained/` after on-device training) — those
    /// would otherwise read as ambiguity in the very case that is unambiguous.
    static func resolveRoot(at selected: URL,
                            displayName: String? = nil,
                            isDirectory: (URL) -> Bool,
                            contents: (URL) -> [String]) throws -> URL {
        // `images` has to be a *directory*: a stray file of that name (an AppleDouble sidecar, a
        // note) must not stop the wrapper descent — hence the shared directory-children helper
        // rather than a plain name lookup in `contents`.
        let top = directoryChildren(of: selected, isDirectory: isDirectory, contents: contents)
        if top.contains(imagesDirectoryName) { return selected }
        guard !top.isEmpty else {
            throw CaptureBundleError.missingImagesDirectory(
                displayName ?? selected.lastPathComponent)
        }
        guard top.count == 1 else { throw CaptureBundleError.ambiguousRoot(top) }
        let inner = selected.appendingPathComponent(top[0], isDirectory: true)
        guard directoryChildren(of: inner, isDirectory: isDirectory, contents: contents)
            .contains(imagesDirectoryName)
        else { throw CaptureBundleError.missingImagesDirectory(top[0]) }
        return inner
    }

    /// Directory children minus the ignored names, sorted — sorted so both the ambiguity message
    /// and the tests are independent of `contentsOfDirectory`'s arbitrary enumeration order.
    ///
    /// A free function taking the accessors explicitly rather than a closure capturing them:
    /// non-escaping parameters may be passed on as non-escaping arguments, but capturing them in a
    /// local closure is not guaranteed to be legal, and this keeps the helper independently
    /// testable.
    static func directoryChildren(of url: URL,
                                  isDirectory: (URL) -> Bool,
                                  contents: (URL) -> [String]) -> [String] {
        contents(url)
            .filter { !isIgnoredDirectory(named: $0) }
            .filter { isDirectory(url.appendingPathComponent($0, isDirectory: true)) }
            .sorted()
    }

    /// The structural verdict — the whole of the posed/unposed decision (see the type header).
    /// A missing `sparse/0` yields an empty listing and therefore `.unposed`, which is why no
    /// existence check is needed.
    static func kind(root: URL, contents: (URL) -> [String]) -> CaptureBundleKind {
        completeColmapSet(among: contents(root.appendingPathComponent("sparse/0",
                                                                     isDirectory: true))) != nil
            ? .posed : .unposed
    }

    /// Image file names, sorted by name — the order that is authoritative for training indexing
    /// (`ImportService.listImages` states the same rule).
    ///
    /// Dot-prefixed names are dropped: an AppleDouble `._0001.jpg` has an image extension and
    /// would otherwise be counted as a photo, letting a one-image bundle pass the two-image floor
    /// and then fail deep inside the loader.
    static func imageNames(among names: [String]) -> [String] {
        names
            .filter { !$0.hasPrefix(".") }
            .filter { ImportService.imageExtensions.contains(
                ($0 as NSString).pathExtension.lowercased()) }
            .sorted()
    }

    /// Reads `mode` out of the manifest, tolerating everything.
    ///
    /// Untyped `JSONSerialization` rather than `Codable` precisely because the manifest's owner has
    /// reserved the right to evolve its fields: a `Decodable` struct fails the whole document when
    /// a sibling key changes type, and this value is a caption. Missing file, invalid JSON, a
    /// non-object root, no `mode`, a non-string `mode`, an empty string — all `nil`, all fine.
    static func manifestMode(_ data: Data?) -> String? {
        guard let data,
              let root = try? JSONSerialization.jsonObject(with: data),
              let object = root as? [String: Any],
              let mode = object["mode"] as? String
        else { return nil }
        let trimmed = mode.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    // MARK: Real filesystem

    /// The production bindings for the injected accessors — every one of them answers "unknown" for
    /// a path that is missing or unreadable, so `locate` needs no separate existence checks.
    static let fileSystemIsDirectory: (URL) -> Bool = { url in
        var isDirectory: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory)
        return exists && isDirectory.boolValue
    }

    static let fileSystemContents: (URL) -> [String] = { url in
        (try? FileManager.default.contentsOfDirectory(atPath: url.path)) ?? []
    }

    static let fileSystemManifest: (URL) -> Data? = { url in try? Data(contentsOf: url) }
}
