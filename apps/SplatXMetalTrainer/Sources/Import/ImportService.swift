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
///
/// One entry point per source kind, and they differ in exactly how much they are allowed to touch:
///
/// - `assembleImages` — a folder of photos. **Rewrites pixels when the EXIF orientation is not 1**:
///   an unknown camera may have left the rotation in metadata, and the C loader compares decoded
///   dimensions against the intrinsics it is handed, so the orientation is baked in.
/// - `VideoFrameExtractor.extract` — a movie. Not in this file at all: the frames are *produced*,
///   so orientation is settled at extraction time and there is nothing to validate afterwards.
/// - `stageColmap` → `assembleColmap` — a solved project. Copy only **unless its cameras carry lens
///   distortion**, in which case the images are rectified and the model rewritten (below). The split
///   into two functions exists so that rectification — ~10 s for 110 × 8 MP — runs off the main
///   actor, the same division `stageCapture` already uses.
/// - `stageCapture` → `assembleCaptureBundle` — a SplatX iOS capture. Copy only, for the same
///   reason as COLMAP plus a stronger guarantee from the producer (see `assembleCaptureBundle`'s
///   header), and it runs the same distortion check for the same reason: the two paths are the two
///   ways a `sparse/0` enters this app, and a check on only one of them is a check with a hole in
///   it. A real capture bundle is always PINHOLE, so it always takes the early exit.
///
/// **The distortion step, on both `sparse/0` paths** (spec §5, `ColmapRectification`):
///
/// 1. Read `cameras.{bin,txt}`. Every camera already PINHOLE / SIMPLE_PINHOLE → nothing happens and
///    the copy below runs exactly as it always did, bytes and file format included.
/// 2. Otherwise the model's images are resampled to a pure pinhole frame and `sparse/0` is rewritten
///    as **all three files in text**, with no `.bin` copied — the loader picks its format from
///    `cameras.bin`'s existence alone (`ColmapLoader.cpp:325`), so a leftover binary would win over
///    the rectified text. Poses and points come through unchanged; point colour is preserved because
///    the core seeds Gaussians from it.
/// 3. An unsupported camera model (COLMAP id 11+) or an unreadable model file **fails the import**.
///    That is the point of doing this here: the same dataset used to import happily and fail minutes
///    later at training, after its images had already been copied into a project (spec §1.1).
enum ImportService {
    static let imageExtensions: Set<String> = ["jpg", "jpeg", "png", "heic"]

    /// The fused LiDAR point cloud a capture bundle ships next to its COLMAP model. Kept beside the
    /// import code rather than in `CaptureBundle` because recognition never looks at it — it is
    /// payload, not evidence.
    static let pointCloudName = "pointcloud.ply"

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

    /// A COLMAP bundle → a complete sparse/0 set (bin preferred, txt when absent) plus images/.
    ///
    /// Distortion-free models are **copied**, byte for byte and in their original format. A model
    /// with lens distortion is rectified instead: `ColmapRectification` writes the images and all
    /// three text files itself, and the copy below is skipped entirely (see this type's header for
    /// the whole rule, and why the input `.bin` must not survive alongside).
    ///
    /// `progress` reports `(done, total)` images while rectifying — never during a plain copy, which
    /// has no per-image step to report — and `note` collects the sentences the import console prints.
    /// Both are called on the calling thread, and rectifying costs ~0.1 s per 8 MP image
    /// (`ImageUndistorter`'s header has the breakdown), so a full dataset is tens of seconds of
    /// blocking work: call this off the main actor, as `stageColmap` does.
    static func assembleColmap(from src: URL, into dst: URL,
                               progress: (Int, Int) -> Void = { _, _ in },
                               note: (String) -> Void = { _ in }) throws -> Int {
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
        let dstImages = dst.appendingPathComponent("images", isDirectory: true)
        if let report = try ColmapRectification.rectify(sparse0: sparse, set: set,
                                                        imagesSrc: srcImages,
                                                        dstSparse0: dstSparse, dstImages: dstImages,
                                                        progress: progress) {
            report.lines.forEach(note)
            // The rectified count, not the folder listing: the model decides which files are part of
            // the dataset, and anything it does not reference was never going to be trained on.
            return report.images
        }
        try fm.createDirectory(at: dstSparse, withIntermediateDirectories: true)
        for f in set {
            try fm.copyItem(at: sparse.appendingPathComponent(f),
                            to: dstSparse.appendingPathComponent(f))
        }
        try fm.copyItem(at: srcImages, to: dstImages)
        return count
    }

    /// A SplatX iOS capture bundle → the assembly directory. `src` is the **resolved**
    /// `CaptureBundleLayout.root`, not the user's selection: the wrapper descent belongs to
    /// `CaptureBundle.locate` and is deliberately not repeated here.
    ///
    /// Copies a whitelist — `images/`, a complete `sparse/0` set, `splatx.json`, `pointcloud.ply` —
    /// and drops the rest. `events.ndjson` and `README.txt` are provenance for the phone, not
    /// inputs. `trained/` matters more: an on-device training run leaves a finished splat inside the
    /// bundle (spec §1.2), and carrying it into a project that is about to train its own would put
    /// two unrelated results under one name with nothing to tell them apart. A whitelist rather than
    /// a blocklist because the producer has reserved the right to add files (`CaptureBundle`'s
    /// header quotes it) — anything new should arrive as "ignored", never as "imported by accident".
    ///
    /// **Everything is `copyItem`. No coordinates are touched, and none should start to.** The
    /// measurement that will tempt the next reader: over all 47 poses of `LiDAR_20260804_153039`
    /// the bundle world is +Y up and the mean camera up sits **177.89°** from the COLMAP gravity up
    /// `(0,−1,0)` this app writes itself — essentially inverted relative to our own convention. That
    /// is a fact, and it is still not a bug (spec §2.3):
    ///
    /// - In-app it makes no difference. `SceneUp.estimate` derives the orbit up from the cameras
    ///   precisely because a COLMAP world is an arbitrary frame, so viewer, thumbnail and Cinema
    ///   stand the scene up whatever the world sign is.
    /// - The one argument for flipping — "external viewers would show it upside down" — has never
    ///   been measured in this repo. Auto-level's own manual check M1 ("the exported PLY is level in
    ///   an external viewer") is still unrecorded, so the premise is unverified on both sides.
    /// - Rotating a user's captured geometry on an unverified premise is the harder direction to
    ///   undo, and preserving the bytes keeps `images.txt`, `points3D.txt` and `pointcloud.ply`
    ///   mutually consistent for free — a parse/rewrite round trip is three chances to disagree.
    ///
    /// If an external viewer ever does prove a problem, the fix belongs in **export**, not here:
    /// one place that normalizes photo and LiDAR projects alike. A correction applied at import
    /// stacks on top of `SceneUp` instead, double-correcting back to where it started — with the
    /// file no longer matching what the phone captured.
    ///
    /// **No EXIF normalization either**, matching `assembleColmap` and on firmer ground than it
    /// stands on. The bundle README states the guarantee in the producer's own words: "Images are baked
    /// upright (EXIF orientation applied to pixels) and tagged sRGB", and "cameras.txt uses the
    /// PINHOLE model on undistorted images, matching the rectified frames in images/". Re-encoding
    /// would cost a lossy JPEG generation to fix nothing, and it would edit exactly the pixels the
    /// shipped intrinsics were solved against.
    ///
    /// **The one exception to "everything is `copyItem`"** is the distortion check every `sparse/0`
    /// goes through (this type's header). It cannot fire on a bundle the phone wrote — the README
    /// quoted below promises PINHOLE on already-rectified frames, and a PINHOLE model takes the early
    /// exit — so the copy semantics above are what a real capture still gets, byte for byte. The
    /// check is here for the bundle that is *not* what it claims: a hand-assembled folder, or a
    /// future producer. Skipping it on this path would leave one of the two doors into `sparse/0`
    /// unchecked, and the failure it prevents is a training run that dies minutes in.
    static func assembleCaptureBundle(from src: URL, into dst: URL,
                                      progress: (Int, Int) -> Void = { _, _ in },
                                      note: (String) -> Void = { _ in }) throws -> Int {
        let fm = FileManager.default
        let srcImages = src.appendingPathComponent(CaptureBundle.imagesDirectoryName,
                                                   isDirectory: true)
        // `CaptureBundleError` rather than `ImportError.missingImagesDir`, whose text ("next to
        // sparse/") describes the COLMAP layout. Both guards below restate what `locate` already
        // proved: cheap, and it keeps this callable — and testable — on its own, so the two cannot
        // silently drift apart.
        guard fm.fileExists(atPath: srcImages.path) else {
            throw CaptureBundleError.missingImagesDirectory(src.lastPathComponent)
        }
        // The same filter `locate` counted with, so the number reported as the project's image count
        // is the number of files actually copied. Copying `images/` wholesale would be shorter but
        // would also carry AppleDouble sidecars and `.DS_Store` into the project and make that
        // equality false.
        let names = CaptureBundle.imageNames(
            among: (try? fm.contentsOfDirectory(atPath: srcImages.path)) ?? [])
        guard names.count >= 2 else { throw ImportError.notEnoughImages(found: names.count) }
        let dstImages = dst.appendingPathComponent("images", isDirectory: true)
        let dstSparse = dst.appendingPathComponent("sparse/0", isDirectory: true)

        // Deliberately not an error when there is no complete set: photo/video bundles have no
        // `sparse/` at all, and a partial one is what `CaptureBundle.kind` calls `.unposed`. This
        // is not a predicate written to match that one — it is literally the same function, called
        // on the same kind of directory listing (`CaptureBundle.completeColmapSet`, whose header
        // records what went wrong when the two were only written to agree). If they could disagree,
        // a project would open `.aligned` with no model under it, skipping the alignment that was
        // its only way to get one.
        let sparse = src.appendingPathComponent("sparse/0", isDirectory: true)
        let set = CaptureBundle.completeColmapSet(
            among: (try? fm.contentsOfDirectory(atPath: sparse.path)) ?? [])

        // The model is inspected before a single frame is copied, so a bundle that has to be
        // rectified is never half-assembled as a copy first (`ColmapRectification` writes both
        // `images/` and `sparse/0/` itself when it fires).
        var imageCount = names.count
        var report: ColmapRectification.Report?
        if let set {
            report = try ColmapRectification.rectify(sparse0: sparse, set: set,
                                                     imagesSrc: srcImages,
                                                     dstSparse0: dstSparse, dstImages: dstImages,
                                                     progress: progress)
        }
        if let report {
            report.lines.forEach(note)
            imageCount = report.images
        } else {
            try fm.createDirectory(at: dstImages, withIntermediateDirectories: true)
            for name in names {
                try fm.copyItem(at: srcImages.appendingPathComponent(name),
                                to: dstImages.appendingPathComponent(name))
            }
            if let set {
                try fm.createDirectory(at: dstSparse, withIntermediateDirectories: true)
                for f in set {
                    try fm.copyItem(at: sparse.appendingPathComponent(f),
                                    to: dstSparse.appendingPathComponent(f))
                }
            }
        }

        // Both optional by construction: `splatx.json` is not part of the contract (CaptureBundle's
        // header), and `pointcloud.ply` only exists for LiDAR. Kept because they are the capture's
        // own record of itself — the manifest is the mode label the inspector shows, and the point
        // cloud is the captured geometry, which no later step can reconstruct if it is dropped here.
        for f in [CaptureBundle.manifestName, pointCloudName] {
            let url = src.appendingPathComponent(f)
            guard fm.fileExists(atPath: url.path) else { continue }
            try fm.copyItem(at: url, to: dst.appendingPathComponent(f))
        }
        return imageCount
    }

    // MARK: Capture staging

    /// A capture bundle expanded and assembled outside the projects root, waiting to be adopted.
    ///
    /// `Sendable` is load-bearing, not decoration: this value crosses back from the generic
    /// executor `stageCapture` runs on to the main actor that creates the project.
    struct StagedCapture: Sendable {
        /// Holds exactly what the project directory should end up containing — nothing more, so
        /// adopting it is "move everything", with no second whitelist to keep in sync.
        let directory: URL
        /// Structural, from `sparse/0` — the value the project's initial state is derived from.
        let kind: CaptureBundleKind
        /// `splatx.json`'s `mode`. **A caption only**, and `nil` is an ordinary outcome — see
        /// `CaptureBundle`'s header for why the manifest can never be a gate.
        let mode: String?
        let imageCount: Int
        /// Sentences the import console should print — today, what the distortion step did. Empty
        /// for every bundle the phone writes, which are all PINHOLE and take the early exit.
        var notes: [String] = []
    }

    /// Expands (when needed), recognizes and assembles a capture selection into `workspace`,
    /// touching nothing under the projects root.
    ///
    /// **`async` for exactly one reason**, and it is not concurrency: this is seconds of blocking
    /// file I/O — the measured bundles run to 288 MB, and 47 frames get copied afterwards — and a
    /// nonisolated `async` function called from the main actor runs on the generic executor
    /// (SE-0338). That is the same mechanism `VideoFrameExtractor.extract` leans on, and it is why
    /// there is no `Task.detached` here. Nothing inside suspends; the hop is the whole point.
    ///
    /// **Why stage at all instead of assembling straight into the project directory**:
    /// `ProjectStore.create` runs its `assemble` closure on the main actor, so copying done there
    /// is copying done on the main thread. Doing the expensive half here and leaving `create` a
    /// handful of renames (`adoptStagedCapture`) is the division the video import already uses, and
    /// it keeps `create`'s guarantee intact — a failed import still leaves no `.tmp-*` behind,
    /// because the failure now happens before `create` is ever entered.
    ///
    /// **A directory is taken as an already-expanded bundle; anything else goes to the zip reader**
    /// (which names the failure itself when the file is not an archive). Deciding on "is it a
    /// directory" rather than on the `.zip` extension means the open panel's file-type filter is a
    /// convenience rather than the thing that has to be right — a bundle folder is a first-class
    /// input here (the phone exposes it through Files and AirDrop, spec §1.2), not a fallback.
    static func stageCapture(source: URL, workspace: URL) async throws -> StagedCapture {
        var root = source
        if !CaptureBundle.fileSystemIsDirectory(source) {
            let expanded = workspace.appendingPathComponent("expanded", isDirectory: true)
            try ZipArchiveReader.extract(zip: source, to: expanded)
            root = expanded
        }
        // `locate` resolves the wrapper directory an iOS zip always carries and validates the image
        // count; `assembleCaptureBundle` is handed the resolved root, never the selection.
        //
        // `displayName` is the selection's own name in both cases. It matters on the zip path: the
        // directory being searched there is this function's scratch directory, so without it a zip
        // that is not a capture bundle failed with "No images/ folder in **expanded**" — an
        // implementation detail of a function the user never called, named at them.
        let layout = try CaptureBundle.locate(at: root, displayName: source.lastPathComponent)
        let assembled = workspace.appendingPathComponent("assembled", isDirectory: true)
        var notes: [String] = []
        let count = try assembleCaptureBundle(from: layout.root, into: assembled,
                                              note: { notes.append($0) })
        return StagedCapture(directory: assembled, kind: layout.kind,
                             mode: layout.mode, imageCount: count, notes: notes)
    }

    // MARK: COLMAP staging

    /// A COLMAP folder assembled outside the projects root, waiting to be adopted.
    struct StagedColmap: Sendable {
        let directory: URL
        let imageCount: Int
        /// What the distortion step did, for the console. Empty when the model was already pinhole.
        let notes: [String]
    }

    /// Assembles a COLMAP selection into `workspace`, off the main actor.
    ///
    /// This exists for the same single reason `stageCapture` does — a nonisolated `async` function
    /// called from the main actor runs on the generic executor (SE-0338), and `ProjectStore.create`
    /// runs its `assemble` closure *on* the main actor. Before rectification the COLMAP path could
    /// get away with assembling inside `create`, because a file copy of a folder someone had already
    /// solved is fast. Rectifying is not: 110 × 8 MP measured at ~10 s (`ImageUndistorter`'s header
    /// has the breakdown), which as a main-thread call is a frozen window with a spinning cursor.
    ///
    /// `progress` reports `(done, total)` images while rectifying. It is `@Sendable` because it
    /// crosses back to the main actor, exactly as `VideoFrameExtractor.extract`'s does.
    static func stageColmap(source: URL, workspace: URL,
                            progress: @escaping @Sendable (Int, Int) -> Void = { _, _ in })
        async throws -> StagedColmap {
        let assembled = workspace.appendingPathComponent("assembled", isDirectory: true)
        var notes: [String] = []
        let count = try assembleColmap(from: source, into: assembled,
                                       progress: { progress($0, $1) },
                                       note: { notes.append($0) })
        return StagedColmap(directory: assembled, imageCount: count, notes: notes)
    }

    /// Moves a staged bundle into the directory `ProjectStore.create` prepared, and returns the
    /// image count `create` records in the manifest.
    ///
    /// Top-level `moveItem`s rather than a recursive copy: the staging area and the projects root
    /// are both under the app's home (its sandbox container when signed), so each move is a rename
    /// and the main actor is blocked for microseconds instead of seconds. Should the two ever land
    /// on different volumes `moveItem` still does the right thing — it just stops being free.
    ///
    /// Moving also means the staged tree is consumed rather than duplicated, so the caller's
    /// cleanup of the workspace has nothing left to delete on the success path.
    @discardableResult
    static func adoptStagedCapture(_ staged: StagedCapture, into dst: URL) throws -> Int {
        try adoptStagedDirectory(staged.directory, into: dst)
        return staged.imageCount
    }

    /// The move itself, shared by both staged paths. One implementation rather than two identical
    /// ones: what the staging directory holds differs between a capture and a COLMAP folder, but
    /// "move everything, top level only" is the same operation and the same guarantee.
    static func adoptStagedDirectory(_ directory: URL, into dst: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: dst, withIntermediateDirectories: true)
        for name in (try fm.contentsOfDirectory(atPath: directory.path)).sorted() {
            try fm.moveItem(at: directory.appendingPathComponent(name),
                            to: dst.appendingPathComponent(name))
        }
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
