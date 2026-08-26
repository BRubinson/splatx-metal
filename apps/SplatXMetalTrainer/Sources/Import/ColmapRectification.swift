// import-undistortion Task 3: the import-time step that turns a distorted COLMAP dataset into the
// pure-pinhole one the core trains on.
//
// ─────────────────────────────────────────────────────────────────────────────────────────────
// Where this sits
// ─────────────────────────────────────────────────────────────────────────────────────────────
// `CameraDistortion` states the geometry, `ImageUndistorter` moves the pixels, `ColmapSparseModel`
// reads and writes the model. This file is the only place that decides *whether* any of that
// happens, and it is called from both import paths that can carry a `sparse/0`
// (`ImportService.assembleColmap` and `assembleCaptureBundle`).
//
// The core is not touched and must not be: `ColmapLoader.cpp` keeps rejecting every model but
// PINHOLE / SIMPLE_PINHOLE, and by the time an import finishes the project on disk is an ordinary
// pinhole dataset. Moving the check here is the whole design (spec §5) — the failure used to happen
// minutes later, at training, after 110 images had already been copied into a project.
//
// ─────────────────────────────────────────────────────────────────────────────────────────────
// Two early exits, at two different levels
// ─────────────────────────────────────────────────────────────────────────────────────────────
// **The model level, by camera model id.** A dataset whose every camera is `SIMPLE_PINHOLE` or
// `PINHOLE` is returned to the caller untouched — `nil`, "nothing to do" — so it keeps its original
// bytes: the same `.bin` files, the same JPEG data. Rectifying it would be a lossy re-encode that
// changes nothing geometric, and for a SplatX capture bundle (always PINHOLE, always already
// rectified on the phone) that would mean re-compressing every frame of every import forever. This
// is also what keeps the capture path's byte-identity guarantee — and its regression test — intact.
//
// **The camera level, by whether the distortion is the identity.** A camera can carry a distorted
// model *name* and no distortion at all: `OPENCV … 0 0 0 0` is what COLMAP writes when distortion
// estimation is off or converges to nothing. Its pixels are already pinhole pixels, so they are
// copied rather than resampled — but its row is still rewritten as PINHOLE, because the core reads
// the model id and refuses everything but 0 and 1. `isEffectivelyPinhole` is where that is decided,
// and its comment records what went wrong while this case fell through to the resampler.
//
// ─────────────────────────────────────────────────────────────────────────────────────────────
// One map at a time, not one per camera
// ─────────────────────────────────────────────────────────────────────────────────────────────
// Images are grouped by camera and the cameras processed one after another, so at most one
// `ImageUndistorter.Map` is alive at a time. The alternative — build every map up front — is fine
// for the usual single-camera dataset and catastrophic for the other real shape: COLMAP's
// `--ImageReader.single_camera 0` gives *every image its own camera*, and at ~65 MB per 8 MP map
// that is 7 GB for 110 images.
//
// ─────────────────────────────────────────────────────────────────────────────────────────────
// Reporting the field of view loss — measured as unread source, never as crop ratio
// ─────────────────────────────────────────────────────────────────────────────────────────────
// Spec §3.3 asks the import to say how much of the capture the user is losing. The obvious measure —
// how much the frame shrank — is wrong, and measurably so: for a fisheye the frame does **not**
// shrink at all (`ImageUndistorterTests.testFisheyeMapLeavesNoOutsidePixels…` pins that the output
// stays 400×300 out of a 400×300 source), because the equidistant map compresses, so a same-size
// pinhole output simply reads a small disc in the middle of the capture. A crop-ratio measure reports
// 0% loss for exactly the model whose loss is the point.
//
// So the loss is measured on the source side: `fieldOfViewLoss` walks the boundary of the output
// frame through the map — which holds the source coordinate of every output pixel — and takes the
// area it encloses. Everything outside that closed curve is capture that no output pixel reads.
import Foundation

enum ColmapRectification {

    enum RectificationError: LocalizedError, Equatable {
        case missingImage(String)
        case unknownCamera(image: String, cameraId: Int)
        case frameCollapsed(cameraId: Int, model: String,
                            sourceWidth: Int, sourceHeight: Int, width: Int, height: Int)

        var errorDescription: String? {
            switch self {
            case .missingImage(let name):
                "The COLMAP model lists \(name), but there is no such file in images/. "
                    + "Import the folder that holds both sparse/0 and the images the "
                    + "reconstruction was solved from."
            case .unknownCamera(let image, let cameraId):
                "The COLMAP model lists \(image) under camera \(cameraId), but sparse/0 has no "
                    + "camera with that id. Export the reconstruction from COLMAP again — a "
                    + "cameras file and an images file from two different solves produce this."
            case .frameCollapsed(let cameraId, let model, let sourceWidth, let sourceHeight,
                                 let width, let height):
                "Camera \(cameraId) (\(model) \(sourceWidth)×\(sourceHeight)) straightens to only "
                    + "\(width)×\(height) pixels — its distortion is too strong for a "
                    + "distortion-free frame to fit inside the capture. Check that the camera's "
                    + "coefficients are the ones COLMAP solved for these images, or undistort the "
                    + "dataset with COLMAP's image_undistorter first."
            }
        }
    }

    /// What one rectification did, for the import console. `lines` are whole sentences; the caller
    /// decides how to decorate them.
    struct Report: Equatable, Sendable {
        let cameras: Int
        let images: Int
        let lines: [String]
    }

    /// Rectifies a COLMAP model and its images, or reports that there is nothing to rectify.
    ///
    /// Returns `nil` when every camera is already pinhole **by model id** — the caller then performs
    /// its own copy, which is what preserves the input bytes. A non-`nil` return means
    /// **everything** has been written: `dstSparse0` holds the three text files and `dstImages` holds
    /// one image per pose the model lists (rectified, or copied for a camera that needed nothing).
    ///
    /// Order is deliberate, and it is the only guarantee this function makes about failure: every
    /// check that can refuse the dataset runs before the first pixel is decoded, so a refusal leaves
    /// the destination empty. That covers the unsupported model id (11+, spec §3.4), a malformed
    /// file, a missing image, an image referencing a camera that does not exist, a name COLMAP's
    /// text format cannot represent, and a camera whose distortion is too strong to straighten.
    ///
    /// `progress` is called after each image with `(done, total)`, on the calling thread — this
    /// blocks for ~10 s on 110 × 8 MP, so the caller must not be the main thread.
    static func rectify(sparse0: URL,
                        set: [String],
                        imagesSrc: URL,
                        dstSparse0: URL,
                        dstImages: URL,
                        progress: (Int, Int) -> Void = { _, _ in }) throws -> Report? {
        let format = ColmapSparse.Format.of(set: set)
        let cameras = try ColmapSparse.readCameras(in: sparse0, format: format)
        // The model-level exit is gated on the *model id*, and deliberately not on "no camera has
        // any distortion". `nil` here means the caller copies `sparse/0` verbatim — including the
        // `.bin` — so the on-disk model has to be one the core already accepts, and the core accepts
        // ids 0 and 1 only (`ColmapLoader.cpp` `derive_pinhole`). An `OPENCV` row whose coefficients
        // are all zero is a pinhole camera geometrically but still an `OPENCV` row on disk: copying
        // it out would hand the user a project that trains no better than it parses. Such a camera
        // takes the per-camera exit below instead — its pixels are copied, its row is rewritten.
        guard cameras.contains(where: { !$0.isPinhole }) else { return nil }

        // Parse first, decode later. `parse` is what rejects id 11+ and a wrong parameter count.
        var geometry: [Int: (CameraIntrinsics, CameraDistortion)] = [:]
        for camera in cameras {
            geometry[camera.id] = try CameraUndistortion.parse(modelId: camera.modelId,
                                                               params: camera.params)
        }

        let images = try ColmapSparse.readImages(in: sparse0, format: format)
        let points = try ColmapSparse.readPoints(in: sparse0, format: format)

        // ─────────────────────────────────────────────────────────────────────────────────────
        // Everything that can refuse this dataset runs here, before the first pixel is decoded
        // ─────────────────────────────────────────────────────────────────────────────────────
        // "Fail while the destination is still empty" is the rule this function is built on, and
        // each of these checks was found *outside* it once.
        let fm = FileManager.default
        let knownCameraIds = Set(cameras.map(\.id))
        for image in images {
            // A model whose images point at a camera id that no longer exists. Such an image is
            // never rectified (the loop below walks cameras and filters images by id) and never
            // copied, yet `writeText` records the whole image list — so the project would ship an
            // images.txt row with no file under it, and the core throws
            // "training_input: image references unknown camera" (`TrainingInput.cpp:19`) minutes
            // later. The file-existence check below cannot see this: the file may well be there.
            guard knownCameraIds.contains(image.cameraId) else {
                throw RectificationError.unknownCamera(image: image.name,
                                                       cameraId: image.cameraId)
            }
            // Every referenced file has to exist — a model that names a missing image is a
            // mis-picked folder, and finding out halfway through is worse than finding out first.
            guard fm.fileExists(atPath: imagesSrc.appendingPathComponent(image.name).path) else {
                throw RectificationError.missingImage(image.name)
            }
            // COLMAP's text pose line is exactly ten tokens, so a name with a space in it cannot be
            // written back. `ColmapSparse.writeText` refuses it as well — that guard is the writer's
            // own invariant and stays — but `writeText` is the *last* statement of this function,
            // so on a binary model the refusal used to arrive after all 110 images had been
            // rectified (~10 s of work thrown away, and the destination already full).
            guard !image.name.contains(" ") else {
                throw ColmapSparseError.nameContainsSpace(image.name)
            }
        }

        // The output frames are solved here rather than inside the loop, for the same reason: a
        // camera whose distortion is too strong to straighten has to be refused before it is
        // resampled, not after (`CameraUndistortion.frameIsUsable` carries the thresholds and the
        // measurements behind them). The solve is pure arithmetic and costs about a millisecond per
        // camera; `ImageUndistorter.makeMap` repeats it internally rather than being handed the
        // answer, so that a map's frame can never disagree with the row written for it.
        var frames: [Int: (width: Int, height: Int, cx: Double, cy: Double)] = [:]
        for camera in cameras {
            guard let (intrinsics, distortion) = geometry[camera.id],
                  !Self.isEffectivelyPinhole(camera, distortion) else { continue }
            let frame = CameraUndistortion.undistortedFrame(intrinsics: intrinsics,
                                                            distortion: distortion,
                                                            width: camera.width,
                                                            height: camera.height)
            guard CameraUndistortion.frameIsUsable(width: frame.width, height: frame.height,
                                                   sourceWidth: camera.width,
                                                   sourceHeight: camera.height) else {
                let name = ColmapCameraModel(rawValue: camera.modelId)?.name
                    ?? "model#\(camera.modelId)"
                throw RectificationError.frameCollapsed(
                    cameraId: camera.id, model: name,
                    sourceWidth: camera.width, sourceHeight: camera.height,
                    width: frame.width, height: frame.height)
            }
            frames[camera.id] = frame
        }

        try fm.createDirectory(at: dstImages, withIntermediateDirectories: true)
        var pinholes: [ColmapPinholeCamera] = []
        var lines: [String] = []
        var done = 0
        let total = images.count

        for camera in cameras {
            guard let (intrinsics, distortion) = geometry[camera.id] else { continue }
            let mine = images.filter { $0.cameraId == camera.id }

            // A mixed model — some cameras distorted, some not — is rare but legal, and the ones
            // with nothing to correct take the early exit *individually*: their images are copied,
            // and their row keeps the intrinsics it came with. Resampling them would be actively
            // wrong, not merely wasteful (`isEffectivelyPinhole` states what "nothing to correct"
            // means and why an identity distortion counts).
            if Self.isEffectivelyPinhole(camera, distortion) {
                pinholes.append(ColmapPinholeCamera(id: camera.id,
                                                    width: camera.width, height: camera.height,
                                                    fx: intrinsics.fx, fy: intrinsics.fy,
                                                    cx: intrinsics.cx, cy: intrinsics.cy))
                for image in mine {
                    let destination = dstImages.appendingPathComponent(image.name)
                    try fm.createDirectory(at: destination.deletingLastPathComponent(),
                                           withIntermediateDirectories: true)
                    try fm.copyItem(at: imagesSrc.appendingPathComponent(image.name),
                                    to: destination)
                    done += 1
                    progress(done, total)
                }
                // Worth a sentence only when the model *said* it was distorted: the row changed
                // model name, the pixels did not, and the console should say why no time was spent
                // on a camera the user knows carries an OPENCV label.
                if !camera.isPinhole {
                    let name = ColmapCameraModel(rawValue: camera.modelId)?.name
                        ?? "model#\(camera.modelId)"
                    lines.append("Camera \(camera.id) needed no rectification · \(name) with every "
                        + "distortion coefficient at zero is a pinhole camera · its \(mine.count) "
                        + "image(s) were copied unchanged and its row was rewritten as PINHOLE "
                        + "\(camera.width)×\(camera.height)")
                }
                continue
            }

            // A camera with no images still needs its row — dropping it would leave the file
            // inconsistent with a model that may reference it elsewhere — but it needs no map, and
            // the frame arithmetic alone gives the row its numbers. `makeMap` derives its frame from
            // the very same call, so the two cannot disagree. The pre-check already solved this
            // frame; the fallback keeps the expression total rather than relying on that.
            guard !mine.isEmpty else {
                let frame = frames[camera.id]
                    ?? CameraUndistortion.undistortedFrame(intrinsics: intrinsics,
                                                           distortion: distortion,
                                                           width: camera.width,
                                                           height: camera.height)
                pinholes.append(ColmapPinholeCamera(id: camera.id,
                                                    width: frame.width, height: frame.height,
                                                    fx: intrinsics.fx, fy: intrinsics.fy,
                                                    cx: frame.cx, cy: frame.cy))
                continue
            }

            let map = ImageUndistorter.makeMap(intrinsics: intrinsics, distortion: distortion,
                                               sourceWidth: camera.width,
                                               sourceHeight: camera.height)
            pinholes.append(ColmapPinholeCamera(id: camera.id,
                                                width: map.width, height: map.height,
                                                fx: intrinsics.fx, fy: intrinsics.fy,
                                                cx: map.cx, cy: map.cy))
            lines.append(summaryLine(camera: camera, map: map, loss: fieldOfViewLoss(map: map)))
            if map.outsidePixels > 0 {
                // Not reachable if the crop arithmetic is right, which is why it is reported rather
                // than assumed away: those pixels are black, and black pixels are training data.
                lines.append("Warning: \(map.outsidePixels) pixels of camera \(camera.id) fell "
                    + "outside the capture and were filled black.")
            }

            for image in mine {
                let source = imagesSrc.appendingPathComponent(image.name)
                let destination = dstImages.appendingPathComponent(image.name)
                // COLMAP names may carry a subdirectory ("frames/0001.jpg") — the core joins the
                // name onto images/ verbatim, so the structure is recreated rather than flattened.
                try fm.createDirectory(at: destination.deletingLastPathComponent(),
                                       withIntermediateDirectories: true)
                try autoreleasepool {
                    try ImageUndistorter.undistort(source: source, destination: destination, map: map)
                }
                done += 1
                progress(done, total)
            }
        }

        try ColmapSparse.writeText(cameras: pinholes, images: images, points: points,
                                   to: dstSparse0)
        return Report(cameras: cameras.count, images: done, lines: lines)
    }

    /// Whether this camera's pixels are already what a pinhole camera would have recorded.
    ///
    /// Two ways that happens, and only the first was honoured until 2026-08-15 FIX 2:
    ///
    /// 1. The **model** is `SIMPLE_PINHOLE` / `PINHOLE`, which carry no coefficient at all.
    /// 2. The model is a distorted one whose **coefficients are all zero** — `OPENCV … 0 0 0 0`,
    ///    which COLMAP emits whenever distortion estimation is switched off or converges to nothing.
    ///    `CameraDistortion.isIdentity` is the authority on this, and it says *no* for the whole
    ///    fisheye family: `θ = atan(r)` is part of that projection, not part of its polynomial, so a
    ///    zero-coefficient fisheye still has to be re-projected.
    ///
    /// Missing case 2 was not merely a wasted re-encode. The camera skipped the copy branch, went
    /// through the resampler, and `undistortedFrame` handed it back a frame re-centered on `W/2` —
    /// so an off-centre principal point turned into a black band across two edges of every image
    /// (see that function's identity early return for the measured 78,500 px). Both halves of that
    /// bug are fixed: this predicate keeps such a camera out of the resampler, and the arithmetic no
    /// longer moves a principal point it was not asked to move.
    static func isEffectivelyPinhole(_ camera: ColmapCameraRecord,
                                     _ distortion: CameraDistortion) -> Bool {
        camera.isPinhole || distortion.isIdentity
    }

    // MARK: - Field of view loss

    /// The fraction of the source image no output pixel reads, in `0…1`.
    ///
    /// The map stores, for every output pixel, the source position it samples. The boundary of the
    /// output frame therefore maps to a closed curve in source coordinates, and — because every
    /// distortion here is continuous and monotone in radius, the same property
    /// `CameraUndistortion.mappedBoundaryBox` relies on — that curve encloses exactly the region the
    /// resampler reads. The shoelace formula gives its area; what is left over is lost capture.
    ///
    /// Why not "how much did the frame shrink": for the fisheye family the frame does not shrink at
    /// all, and a crop-ratio measure would report no loss for the models that lose the most (see the
    /// file header).
    ///
    /// The curve runs through pixel *centers*, so it under-measures the read region by a half-pixel
    /// ring — 0.05 % on a 2160×3840 frame, an over-report of the loss by that much and no more.
    static func fieldOfViewLoss(map: ImageUndistorter.Map) -> Double {
        guard map.width >= 2, map.height >= 2, map.sourceWidth > 0, map.sourceHeight > 0 else {
            return 0
        }
        func at(_ u: Int, _ v: Int) -> (x: Double, y: Double) {
            let i = 2 * (v * map.width + u)
            return (Double(map.coordinates[i]), Double(map.coordinates[i + 1]))
        }
        var polygon: [(x: Double, y: Double)] = []
        polygon.reserveCapacity(2 * (map.width + map.height))
        for u in 0..<map.width { polygon.append(at(u, 0)) }
        for v in 1..<map.height { polygon.append(at(map.width - 1, v)) }
        for u in stride(from: map.width - 2, through: 0, by: -1) {
            polygon.append(at(u, map.height - 1))
        }
        for v in stride(from: map.height - 2, through: 1, by: -1) { polygon.append(at(0, v)) }

        var twiceArea = 0.0
        for i in polygon.indices {
            let a = polygon[i], b = polygon[(i + 1) % polygon.count]
            twiceArea += a.x * b.y - b.x * a.y
        }
        let read = abs(twiceArea) / 2
        let capture = Double(map.sourceWidth) * Double(map.sourceHeight)
        return min(1, max(0, 1 - read / capture))
    }

    /// The console sentence for one rectified camera.
    ///
    /// Split out as a pure function because it is the only part of the report a test can pin, and
    /// because the fisheye caveat has to be attached by model family rather than by how large the
    /// number happens to be: a fisheye's loss is structural (spec §3.3 — a pinhole cannot express
    /// those directions at all), while a 3 % loss on an OPENCV camera is just the black rim being
    /// cropped away.
    static func summaryLine(camera: ColmapCameraRecord,
                            map: ImageUndistorter.Map,
                            loss: Double) -> String {
        let name = ColmapCameraModel(rawValue: camera.modelId)?.name ?? "model#\(camera.modelId)"
        var line = "Camera \(camera.id) rectified · \(name) \(camera.width)×\(camera.height) → "
            + "PINHOLE \(map.width)×\(map.height) · "
            + String(format: "%.1f%%", loss * 100) + " of the captured field of view dropped"
        if isFisheye(modelId: camera.modelId) {
            line += " (a fisheye lens records directions a pinhole camera cannot express, so the "
                + "outer field of view cannot be kept)"
        }
        return line
    }

    /// Ids 5, 8, 9, 10 — the `θ = atan(r)` family (spec §3.2).
    static func isFisheye(modelId: Int) -> Bool {
        modelId == 5 || modelId == 8 || modelId == 9 || modelId == 10
    }
}
