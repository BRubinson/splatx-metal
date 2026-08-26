// import-undistortion Task 3: the import wiring — `ColmapRectification` plus the two
// `ImportService` paths that call it.
//
// The assertions here are deliberately made **without** the production reader wherever the claim is
// about content. `cameras.txt`, `images.txt` and `points3D.txt` are re-parsed by hand (split on
// whitespace, `Double(_:)`), because using `ColmapSparse` to check what `ColmapSparse` wrote would
// pass just as happily if the reader and the writer shared a mistake — swapped R and B, a dropped
// column. The fixtures on the way in are built by `ColmapFixture`, which transcribes the on-disk
// layouts from the core's C++ reader rather than from the Swift one.
//
// Three of these are regression guards for traps that were found before the code was written, and
// each is labelled at the test:
//
//   A — point colour survives. The core seeds every Gaussian's colour from `points3D` RGB, and the
//       nearest existing reader (`ColmapCameraReader.loadPoints`) returns positions only. Wiring
//       through it would have greyed out every rectified project with no error anywhere.
//   B — no `.bin` in the output. The loader picks its format from `cameras.bin`'s existence alone,
//       so a surviving binary silently outranks the rectified text.
//   C — the field of view loss is measured as unread *source*, not as how much the frame shrank.
//       For fisheye the frame does not shrink at all, so a crop-ratio measure reports 0 % for the
//       models that lose the most.
//
// Everything lives under `FileManager.default.temporaryDirectory` and is removed in tearDown — no
// real home, no ~/Desktop, no ~/Downloads (this machine's TCC rule).
import XCTest
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
@testable import SplatXMetalTrainer

final class ColmapRectificationTests: XCTestCase {

    // MARK: - Fixture cameras

    /// A small OPENCV camera with distortion strong enough that a geometry mistake cannot hide —
    /// the same one `ImageUndistorterTests` measures with.
    private static let distortedCamera = ColmapCameraRecord(
        id: 1, modelId: 4, width: 320, height: 240,
        params: [260, 260, 160, 120, -0.15, 0.05, 0.0012, -0.0009])

    /// Poses whose digits do not survive a low-precision round trip: the rectification must not move
    /// a camera (spec §5.1), and "must not move" is checked as exact `Double` equality.
    private static let poses = [
        ColmapImageRecord(id: 7, qw: 0.70710678118654757, qx: -1.2345678901234567e-09,
                          qy: 0.5, qz: -0.5,
                          tx: -0.12345678901234567, ty: 2.0000000000000004, tz: 3.25,
                          cameraId: 1, name: "a.png"),
        ColmapImageRecord(id: 9, qw: 1, qx: 0, qy: 0, qz: 0,
                          tx: 1e-17, ty: -0.99999999999999989, tz: 0,
                          cameraId: 1, name: "b.png"),
    ]

    /// Colours a positions-only reader would destroy: saturated, black, and off-grey.
    private static let points = [
        ColmapPointRecord(x: 0.5, y: -1.25, z: 3, r: 255, g: 0, b: 17),
        ColmapPointRecord(x: -2, y: 0.125, z: -0.5, r: 0, g: 0, b: 0),
        ColmapPointRecord(x: 1e-8, y: 4.5, z: 6.75, r: 3, g: 128, b: 254),
    ]

    private var tmp: URL!

    override func setUpWithError() throws {
        tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("sx-rectify-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tmp)
    }

    private var src: URL { tmp.appendingPathComponent("colmap", isDirectory: true) }
    private var dst: URL { tmp.appendingPathComponent("assembled", isDirectory: true) }

    // MARK: - Building a source tree

    /// A COLMAP folder: `sparse/0` in the requested encoding, plus one real PNG per pose at the
    /// camera's declared size.
    private func makeProject(camera: ColmapCameraRecord = ColmapRectificationTests.distortedCamera,
                             images: [ColmapImageRecord] = ColmapRectificationTests.poses,
                             binary: Bool) throws {
        try ColmapFixture.write(cameras: [camera], images: images, points: Self.points,
                                to: src.appendingPathComponent("sparse/0", isDirectory: true),
                                binary: binary)
        let dir = src.appendingPathComponent("images", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for image in images {
            try writePNG(named: image.name, width: camera.width, height: camera.height, in: dir)
        }
    }

    // MARK: - A — the whole conversion, text in

    func testDistortedTextModelBecomesAPinholeProjectWithItsPosesAndColoursIntact() throws {
        try makeProject(binary: false)
        var notes: [String] = []
        let count = try ImportService.assembleColmap(from: src, into: dst,
                                                     note: { notes.append($0) })
        XCTAssertEqual(count, 2)

        // Trap B, restated on the text path: all three files, text only, nothing else.
        XCTAssertEqual(entries("sparse/0"), ["cameras.txt", "images.txt", "points3D.txt"])
        XCTAssertEqual(entries("images"), ["a.png", "b.png"])

        // cameras.txt is PINHOLE, and its size is the size of the pixels that were actually
        // written — the core compares the two exactly and refuses the run if they differ
        // (spec §5.2, `TrainC.cpp:595`).
        let camera = try XCTUnwrap(dataLines("sparse/0/cameras.txt").first)
        XCTAssertEqual(camera[0], "1")                       // the input's camera id, not a new one
        XCTAssertEqual(camera[1], "PINHOLE")
        XCTAssertEqual(camera.count, 8)                      // ID MODEL W H fx fy cx cy
        let width = try XCTUnwrap(Int(camera[2])), height = try XCTUnwrap(Int(camera[3]))
        for name in ["a.png", "b.png"] {
            let size = try XCTUnwrap(decodedSize(dst.appendingPathComponent("images/\(name)")))
            XCTAssertEqual(size.width, width, "\(name) width vs cameras.txt")
            XCTAssertEqual(size.height, height, "\(name) height vs cameras.txt")
        }
        // Rectification keeps the focal length and re-centers the principal point (spec §4).
        XCTAssertEqual(try XCTUnwrap(Double(camera[4])), 260, accuracy: 0)
        XCTAssertEqual(try XCTUnwrap(Double(camera[5])), 260, accuracy: 0)
        XCTAssertEqual(try XCTUnwrap(Double(camera[6])), Double(width) / 2, accuracy: 0)
        XCTAssertEqual(try XCTUnwrap(Double(camera[7])), Double(height) / 2, accuracy: 0)
        // And the frame is the one the crop arithmetic decided, not a number this step invented.
        // (Measured: this camera's `k1 = −0.15` *compresses*, so the frame does not shrink at all —
        // 320×240 in, 320×240 out. Asserting "smaller than the source" here would have been the same
        // crop-ratio mistake trap C is about, on a perspective camera.)
        let frame = CameraUndistortion.undistortedFrame(
            intrinsics: CameraIntrinsics(fx: 260, fy: 260, cx: 160, cy: 120),
            distortion: .brownConrady(k1: -0.15, k2: 0.05, p1: 0.0012, p2: -0.0009,
                                      k3: 0, k4: 0, k5: 0, k6: 0),
            width: 320, height: 240)
        XCTAssertEqual(width, frame.width)
        XCTAssertEqual(height, frame.height)

        // The poses came through untouched — value for value, not "close enough".
        try assertPosesRoundTripped()

        // TRAP A. Every colour survives. A positions-only reader (`ColmapCameraReader.loadPoints`)
        // would have made every one of these 0 0 0, and nothing downstream would have complained.
        let written = try dataLines("sparse/0/points3D.txt")
        XCTAssertEqual(written.count, Self.points.count)
        for (line, expected) in zip(written, Self.points) {
            XCTAssertEqual(line.count, 8, "ID X Y Z R G B ERROR")
            XCTAssertEqual(Double(line[1]), expected.x)
            XCTAssertEqual(Double(line[2]), expected.y)
            XCTAssertEqual(Double(line[3]), expected.z)
            XCTAssertEqual(UInt8(line[4]), expected.r)
            XCTAssertEqual(UInt8(line[5]), expected.g)
            XCTAssertEqual(UInt8(line[6]), expected.b)
        }

        // The report tells the user what was done and what it cost (spec §3.3).
        XCTAssertEqual(notes.count, 1)
        let note = try XCTUnwrap(notes.first)
        XCTAssertTrue(note.contains("OPENCV 320×240"), note)
        XCTAssertTrue(note.contains("PINHOLE \(width)×\(height)"), note)
        XCTAssertTrue(note.contains("field of view"), note)
        XCTAssertFalse(note.contains("fisheye"), "a perspective camera must not carry the fisheye caveat")
    }

    /// The pose line must be followed by a POINTS2D line, and the core reads that second line
    /// unconditionally — a file without it fails with "missing POINTS2D line after pose line"
    /// (`ColmapLoader.cpp:216`).
    func testWrittenImagesTxtHasTwoPhysicalLinesPerImage() throws {
        try makeProject(binary: false)
        _ = try ImportService.assembleColmap(from: src, into: dst)
        let text = try String(contentsOf: dst.appendingPathComponent("sparse/0/images.txt"),
                              encoding: .utf8)
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
        // header + (pose, blank) × 2 + the trailing empty component after the final newline
        XCTAssertEqual(lines.count, 6)
        XCTAssertTrue(lines[0].hasPrefix("#"))
        XCTAssertEqual(lines[1].split(whereSeparator: \.isWhitespace).count, 10)
        XCTAssertEqual(lines[2], "")
        XCTAssertEqual(lines[3].split(whereSeparator: \.isWhitespace).count, 10)
        XCTAssertEqual(lines[4], "")
    }

    // MARK: - B — binary in, text out, and no .bin anywhere

    func testDistortedBinaryModelIsRewrittenAsTextAndTheBinIsNotCopied() throws {
        try makeProject(binary: true)
        XCTAssertEqual(try ImportService.assembleColmap(from: src, into: dst), 2)

        // TRAP B. `ColmapLoader.cpp:325` chooses the format on `cameras.bin`'s existence alone and
        // then reads all three files that way, so a copied binary would not merely be redundant —
        // the stale distorted camera would win over the rectified text and train silently wrong.
        XCTAssertEqual(entries("sparse/0"), ["cameras.txt", "images.txt", "points3D.txt"])

        let camera = try XCTUnwrap(dataLines("sparse/0/cameras.txt").first)
        XCTAssertEqual(camera[1], "PINHOLE")
        try assertPosesRoundTripped()
        // Trap A again on the binary path — RGB is three raw bytes there, easy to skip past.
        let written = try dataLines("sparse/0/points3D.txt")
        XCTAssertEqual(written.map { [$0[4], $0[5], $0[6]] },
                       Self.points.map { ["\($0.r)", "\($0.g)", "\($0.b)"] })
    }

    // MARK: - The early exit

    func testPinholeModelIsCopiedUntouchedInItsOriginalFormat() throws {
        // A pinhole dataset must not be re-encoded: there is nothing to correct, and a JPEG
        // generation would be lost for nothing (spec §7.1 "PINHOLE 통과").
        try makeProject(camera: ColmapFixture.pinholeCamera, binary: true)
        let before = try Data(
            contentsOf: src.appendingPathComponent("images/a.png"))

        var notes: [String] = []
        XCTAssertEqual(try ImportService.assembleColmap(from: src, into: dst,
                                                        note: { notes.append($0) }), 2)

        XCTAssertEqual(entries("sparse/0"), ["cameras.bin", "images.bin", "points3D.bin"])
        XCTAssertEqual(try Data(contentsOf: dst.appendingPathComponent("images/a.png")), before)
        XCTAssertEqual(notes, [], "an early exit has nothing to report")
    }

    // MARK: - The other early exit: a distorted model name with no distortion in it

    func testAZeroCoefficientCameraKeepsItsPrincipalPointAndItsImageBytes() throws {
        // 2026-08-15 FIX 2. `OPENCV … 0 0 0 0` is what COLMAP writes when distortion estimation is
        // off or converges to nothing, and such a camera is a pinhole camera — but only
        // geometrically. On disk it is still an OPENCV row, which the core refuses whatever its
        // coefficients say (`derive_pinhole`), so this dataset cannot take the model-level exit that
        // copies `sparse/0` verbatim: the row has to be rewritten, while the pixels must not be.
        //
        // The principal point is deliberately off centre (190, 100 of 320×240). That is what the bug
        // needed: the camera used to go through the resampler, `undistortedFrame` re-centered it on
        // `W/2, H/2`, and every image came out with a black band down two edges.
        let camera = ColmapCameraRecord(id: 1, modelId: 4, width: 320, height: 240,
                                        params: [260, 260, 190, 100, 0, 0, 0, 0])
        try makeProject(camera: camera, binary: true)
        let before = try Data(contentsOf: src.appendingPathComponent("images/a.png"))

        var notes: [String] = []
        XCTAssertEqual(try ImportService.assembleColmap(from: src, into: dst,
                                                        note: { notes.append($0) }), 2)

        XCTAssertEqual(entries("sparse/0"), ["cameras.txt", "images.txt", "points3D.txt"])
        let row = try XCTUnwrap(dataLines("sparse/0/cameras.txt").first)
        XCTAssertEqual(row[1], "PINHOLE")
        XCTAssertEqual(Int(row[2]), 320, "the frame must not shrink — nothing was resampled")
        XCTAssertEqual(Int(row[3]), 240)
        XCTAssertEqual(Double(row[4]), 260)
        XCTAssertEqual(Double(row[5]), 260)
        XCTAssertEqual(Double(row[6]), 190, "cx must stay where the lens put it, not move to W/2")
        XCTAssertEqual(Double(row[7]), 100)

        // Byte for byte: a copy, not a JPEG generation. This is also the 110-image re-encode the
        // fix saves on such a dataset.
        XCTAssertEqual(try Data(contentsOf: dst.appendingPathComponent("images/a.png")), before)
        XCTAssertEqual(entries("images"), ["a.png", "b.png"])
        try assertPosesRoundTripped()

        // And it is reported, because the console would otherwise be silent about a camera the user
        // knows is labelled OPENCV.
        XCTAssertEqual(notes.count, 1)
        let note = try XCTUnwrap(notes.first)
        XCTAssertTrue(note.contains("needed no rectification"), note)
        XCTAssertTrue(note.contains("PINHOLE 320×240"), note)
    }

    // MARK: - Refusals that used to arrive too late, or not at all

    func testAnImageReferencingACameraThatDoesNotExistIsRejectedBeforeAnythingIsWritten() throws {
        // The images list is written in full by `writeText`, but only images belonging to a camera
        // are copied or rectified — so an orphan produced an images.txt row with no file under it,
        // and the core threw "image references unknown camera" (`TrainingInput.cpp:19`) minutes into
        // a run. The file-existence check cannot catch it: b.png is right there on disk.
        let orphan = ColmapImageRecord(id: 9, qw: 1, qx: 0, qy: 0, qz: 0,
                                       tx: 0, ty: 0, tz: 0, cameraId: 4, name: "b.png")
        try makeProject(images: [Self.poses[0], orphan], binary: false)
        XCTAssertThrowsError(try ImportService.assembleColmap(from: src, into: dst)) { error in
            XCTAssertEqual(error as? ColmapRectification.RectificationError,
                           .unknownCamera(image: "b.png", cameraId: 4))
        }
        XCTAssertEqual(entries("images"), [])
        XCTAssertEqual(entries("sparse/0"), [])
    }

    func testAnImageNameWithASpaceIsRejectedBeforeAnyPixelIsRectified() throws {
        // `ColmapSparse.writeText` has always refused this — but `writeText` is the *last* statement
        // of `rectify`, so on a binary model the refusal arrived after every image had been
        // rectified. The binary path is the only one that can carry such a name at all: the text
        // pose line is exactly ten tokens, so a name with a space fails to parse first.
        let spaced = ColmapImageRecord(id: 9, qw: 1, qx: 0, qy: 0, qz: 0,
                                       tx: 0, ty: 0, tz: 0, cameraId: 1, name: "b 2.png")
        try makeProject(images: [Self.poses[0], spaced], binary: true)
        XCTAssertThrowsError(try ImportService.assembleColmap(from: src, into: dst)) { error in
            XCTAssertEqual(error as? ColmapSparseError, .nameContainsSpace("b 2.png"))
        }
        XCTAssertEqual(entries("images"), [], "not one image may have been rectified first")
    }

    func testACameraThatCannotBeStraightenedIsRefusedRatherThanShrunkToNothing() throws {
        // Spec §1.1's failure shape, and the one the solver cannot see: a strong pincushion leaves
        // no distortion-free frame inside the capture, so the honest answer is 1×1. Every image was
        // re-encoded at that size, `outsidePixels` stayed 0 so nothing warned, and the project died
        // minutes later in the core's SSIM validator (H, W ≥ 11) — by which time the originals only
        // exist in the user's own zip. `k = 0.7` is a coefficient COLMAP itself accepts; its
        // `max_extra_param` default is 1.0.
        let camera = ColmapCameraRecord(id: 1, modelId: 2, width: 320, height: 240,
                                        params: [160, 160, 120, 0.7])
        try makeProject(camera: camera, binary: false)
        XCTAssertThrowsError(try ImportService.assembleColmap(from: src, into: dst)) { error in
            XCTAssertEqual(error as? ColmapRectification.RectificationError,
                           .frameCollapsed(cameraId: 1, model: "SIMPLE_RADIAL",
                                           sourceWidth: 320, sourceHeight: 240,
                                           width: 1, height: 1))
            let message = (error as? LocalizedError)?.errorDescription ?? ""
            XCTAssertTrue(message.contains("SIMPLE_RADIAL 320×240"), message)
            XCTAssertTrue(message.contains("image_undistorter"),
                          "the message has to say what to do: \(message)")
        }
        XCTAssertEqual(entries("images"), [])
    }

    func testAStronglyDistortedButStillUsableCameraIsAccepted() throws {
        // The other side of the same gate: the floor must not refuse a dataset that can still train.
        // The same camera at `k = 0.5` solves to 71×53 — a fifth of each source side is the
        // threshold (COLMAP's own `min_scale`), and this clears it.
        let camera = ColmapCameraRecord(id: 1, modelId: 2, width: 320, height: 240,
                                        params: [160, 160, 120, 0.5])
        try makeProject(camera: camera, binary: false)
        XCTAssertEqual(try ImportService.assembleColmap(from: src, into: dst), 2)
        let row = try XCTUnwrap(dataLines("sparse/0/cameras.txt").first)
        XCTAssertEqual(row[1], "PINHOLE")
        XCTAssertEqual(Int(row[2]), 71)
        XCTAssertEqual(Int(row[3]), 53)
    }

    // MARK: - C — the field of view loss is a source-side measurement

    func testFisheyeLosesFieldOfViewWithoutLosingASinglePixelOfFrame() throws {
        // TRAP C. The equidistant map compresses, so a pinhole output of the same size reads a disc
        // well inside the capture rather than overrunning it. Nothing is cropped — and that is
        // exactly why a crop-ratio measure would report 0 % for the family that loses the most.
        let (k, d) = try CameraUndistortion.parse(modelId: 5,
                                                  params: [200, 200, 200, 150, -0.02, 0.004, 0, 0])
        let map = ImageUndistorter.makeMap(intrinsics: k, distortion: d,
                                           sourceWidth: 400, sourceHeight: 300)
        XCTAssertEqual(map.width, 400)      // the crop ratio is zero …
        XCTAssertEqual(map.height, 300)
        let loss = ColmapRectification.fieldOfViewLoss(map: map)
        XCTAssertGreaterThan(loss, 0.15)    // … and the loss is not
        XCTAssertLessThan(loss, 0.60)

        let line = ColmapRectification.summaryLine(
            camera: ColmapCameraRecord(id: 2, modelId: 5, width: 400, height: 300,
                                       params: [200, 200, 200, 150, -0.02, 0.004, 0, 0]),
            map: map, loss: loss)
        XCTAssertTrue(line.contains("OPENCV_FISHEYE 400×300 → PINHOLE 400×300"), line)
        XCTAssertTrue(line.contains("fisheye lens records directions a pinhole camera cannot"), line)
    }

    func testPerspectiveLossIsSmallButNotZero() throws {
        let (k, d) = try CameraUndistortion.parse(modelId: Self.distortedCamera.modelId,
                                                  params: Self.distortedCamera.params)
        let map = ImageUndistorter.makeMap(intrinsics: k, distortion: d,
                                           sourceWidth: Self.distortedCamera.width,
                                           sourceHeight: Self.distortedCamera.height)
        let loss = ColmapRectification.fieldOfViewLoss(map: map)
        // Cropping away the black rim costs real capture, and it is a fraction of the frame rather
        // than a fisheye's ring.
        XCTAssertGreaterThan(loss, 0)
        XCTAssertLessThan(loss, 0.5)
    }

    // MARK: - Refusals (spec §1.1 — the import converts or it refuses; nothing reaches training)

    func testUnsupportedModelIdIsRejectedOnTheBinaryPath() throws {
        // COLMAP 4.1 defines ids 11+ (`RAD_TAN_THIN_PRISM_FISHEYE` and five more). Rejecting them
        // is the point: an id whose parameter count is unknown is also a record whose length is
        // unknown, so continuing would misread every byte after it.
        try makeProject(camera: ColmapCameraRecord(id: 1, modelId: 11, width: 320, height: 240,
                                                   params: [1, 2, 3, 4]),
                        binary: true)
        XCTAssertThrowsError(try ImportService.assembleColmap(from: src, into: dst)) { error in
            XCTAssertEqual(error as? CameraModelError, .unsupportedModel(id: 11))
            let message = (error as? LocalizedError)?.errorDescription ?? ""
            XCTAssertTrue(message.contains("0–10"), message)
            XCTAssertTrue(message.contains("undistort"), "the message has to say what to do: \(message)")
        }
    }

    func testUnsupportedModelNameIsRejectedOnTheTextPath() throws {
        try makeProject(binary: false)
        try Data("1 RAD_TAN_THIN_PRISM_FISHEYE 320 240 1 2 3 4\n".utf8)
            .write(to: src.appendingPathComponent("sparse/0/cameras.txt"))
        XCTAssertThrowsError(try ImportService.assembleColmap(from: src, into: dst)) { error in
            XCTAssertEqual(error as? ColmapSparseError,
                           .unsupportedModelName("RAD_TAN_THIN_PRISM_FISHEYE"))
            let message = (error as? LocalizedError)?.errorDescription ?? ""
            XCTAssertTrue(message.contains("RAD_TAN_THIN_PRISM_FISHEYE"), message)
            XCTAssertTrue(message.contains("Re-run COLMAP"), message)
        }
    }

    func testMalformedCamerasFileIsRejected() throws {
        try makeProject(binary: false)
        try Data("PINHOLE\n".utf8)
            .write(to: src.appendingPathComponent("sparse/0/cameras.txt"))
        XCTAssertThrowsError(try ImportService.assembleColmap(from: src, into: dst)) { error in
            guard case .malformed(let file, _, _)? = error as? ColmapSparseError else {
                return XCTFail("wrong error: \(error)")
            }
            XCTAssertEqual(file, "cameras.txt")
        }
    }

    func testAModelReferencingAMissingImageIsRejectedBeforeAnythingIsWritten() throws {
        try makeProject(binary: false)
        try FileManager.default.removeItem(at: src.appendingPathComponent("images/b.png"))
        // Two images still on disk is what `assembleColmap`'s own floor counts, so the failure has
        // to come from the model, not from the count.
        try writePNG(named: "spare.png", width: 320, height: 240,
                     in: src.appendingPathComponent("images", isDirectory: true))
        XCTAssertThrowsError(try ImportService.assembleColmap(from: src, into: dst)) { error in
            XCTAssertEqual(error as? ColmapRectification.RectificationError,
                           .missingImage("b.png"))
        }
        // Nothing half-written: the check runs before the first pixel is decoded.
        XCTAssertEqual(entries("images"), [])
    }

    // MARK: - Progress

    func testProgressIsReportedOncePerImage() throws {
        try makeProject(binary: false)
        var seen: [String] = []
        _ = try ImportService.assembleColmap(from: src, into: dst,
                                             progress: { seen.append("\($0)/\($1)") })
        XCTAssertEqual(seen, ["1/2", "2/2"])
    }

    // MARK: - The capture path is wired too

    func testACaptureBundleWithADistortedModelIsRectifiedAsWell() throws {
        // A bundle the phone wrote is always PINHOLE and always takes the early exit — that case is
        // `CaptureAssemblyTests`. This is the other door into `sparse/0`, and it has to be checked
        // as well or the check has a hole in it.
        let bundle = tmp.appendingPathComponent("bundle", isDirectory: true)
        try ColmapFixture.write(cameras: [Self.distortedCamera], images: Self.poses,
                                points: Self.points,
                                to: bundle.appendingPathComponent("sparse/0", isDirectory: true),
                                binary: false)
        let dir = bundle.appendingPathComponent("images", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for name in ["a.png", "b.png"] { try writePNG(named: name, width: 320, height: 240, in: dir) }
        try Data(#"{"mode":"photo"}"#.utf8).write(to: bundle.appendingPathComponent("splatx.json"))

        var notes: [String] = []
        let count = try ImportService.assembleCaptureBundle(from: bundle, into: dst,
                                                            note: { notes.append($0) })
        XCTAssertEqual(count, 2)
        XCTAssertEqual(entries("sparse/0"), ["cameras.txt", "images.txt", "points3D.txt"])
        XCTAssertEqual(try XCTUnwrap(dataLines("sparse/0/cameras.txt").first)[1], "PINHOLE")
        XCTAssertEqual(notes.count, 1)
        // The bundle's own files still come along.
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: dst.appendingPathComponent("splatx.json").path))
    }

    // MARK: - Sheet plumbing

    func testColmapBusyLabelSaysWhatItIsDoing() {
        XCTAssertEqual(ImportSheet.colmapBusyLabel(done: 0, total: 0), "Reading the reconstruction…")
        XCTAssertEqual(ImportSheet.colmapBusyLabel(done: 7, total: 110),
                       "Removing lens distortion — 7 of 110 images")
    }

    func testStagingAssemblesOutsideTheProjectAndReportsWhatItDid() async throws {
        try makeProject(binary: false)
        let workspace = tmp.appendingPathComponent("ws", isDirectory: true)
        let staged = try await ImportService.stageColmap(source: src, workspace: workspace)
        XCTAssertEqual(staged.imageCount, 2)
        XCTAssertEqual(staged.notes.count, 1)
        XCTAssertEqual(((try? FileManager.default.contentsOfDirectory(
            atPath: staged.directory.path)) ?? []).sorted(), ["images", "sparse"])

        // Adoption moves the tree wholesale, which is all `ProjectStore.create` is left to do.
        let project = tmp.appendingPathComponent("project", isDirectory: true)
        try ImportService.adoptStagedDirectory(staged.directory, into: project)
        XCTAssertEqual(((try? FileManager.default.contentsOfDirectory(
            atPath: project.path)) ?? []).sorted(), ["images", "sparse"])
        XCTAssertEqual(((try? FileManager.default.contentsOfDirectory(
            atPath: staged.directory.path)) ?? []), [])
    }

    // MARK: - Helpers

    private func entries(_ rel: String) -> [String] {
        ((try? FileManager.default.contentsOfDirectory(
            atPath: dst.appendingPathComponent(rel).path)) ?? []).sorted()
    }

    /// Whitespace-split tokens of every non-comment, non-blank line — a hand parse, deliberately not
    /// `ColmapSparse` (see the file header).
    private func dataLines(_ rel: String) throws -> [[String]] {
        let text = try String(contentsOf: dst.appendingPathComponent(rel), encoding: .utf8)
        return text.split(separator: "\n")
            .filter { line in
                guard let first = line.first(where: { !$0.isWhitespace }) else { return false }
                return first != "#"
            }
            .map { $0.split(whereSeparator: \.isWhitespace).map(String.init) }
    }

    /// Every pose value identical to the input — undistortion does not move a camera (spec §5.1),
    /// and a bin → txt conversion is the one place the digits could quietly be shortened.
    private func assertPosesRoundTripped(file: StaticString = #filePath,
                                         line: UInt = #line) throws {
        let written = try dataLines("sparse/0/images.txt")
        XCTAssertEqual(written.count, Self.poses.count, file: file, line: line)
        for (tokens, pose) in zip(written, Self.poses) {
            XCTAssertEqual(Int(tokens[0]), pose.id, "IMAGE_ID", file: file, line: line)
            XCTAssertEqual(Double(tokens[1]), pose.qw, "QW", file: file, line: line)
            XCTAssertEqual(Double(tokens[2]), pose.qx, "QX", file: file, line: line)
            XCTAssertEqual(Double(tokens[3]), pose.qy, "QY", file: file, line: line)
            XCTAssertEqual(Double(tokens[4]), pose.qz, "QZ", file: file, line: line)
            XCTAssertEqual(Double(tokens[5]), pose.tx, "TX", file: file, line: line)
            XCTAssertEqual(Double(tokens[6]), pose.ty, "TY", file: file, line: line)
            XCTAssertEqual(Double(tokens[7]), pose.tz, "TZ", file: file, line: line)
            XCTAssertEqual(Int(tokens[8]), pose.cameraId, "CAMERA_ID", file: file, line: line)
            XCTAssertEqual(tokens[9], pose.name, "NAME", file: file, line: line)
        }
    }

    /// A real image at the camera's size — a gradient rather than a flat fill, so a resample that
    /// wrote nothing at all would still be visible as a size mismatch and not hide behind uniformity.
    @discardableResult
    private func writePNG(named name: String, width: Int, height: Int, in dir: URL) throws -> URL {
        let ctx = try XCTUnwrap(CGContext(data: nil, width: width, height: height,
                                          bitsPerComponent: 8, bytesPerRow: 0,
                                          space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        let base = try XCTUnwrap(ctx.data).assumingMemoryBound(to: UInt8.self)
        for v in 0..<height {
            let row = base + v * ctx.bytesPerRow
            for u in 0..<width {
                row[u * 4 + 0] = UInt8((u * 255) / max(1, width - 1))
                row[u * 4 + 1] = UInt8((v * 255) / max(1, height - 1))
                row[u * 4 + 2] = 128
                row[u * 4 + 3] = 255
            }
        }
        let image = try XCTUnwrap(ctx.makeImage())
        let url = dir.appendingPathComponent(name)
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return url
    }

    private func decodedSize(_ url: URL) -> (width: Int, height: Int)? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? Int,
              let h = props[kCGImagePropertyPixelHeight] as? Int else { return nil }
        return (w, h)
    }
}
