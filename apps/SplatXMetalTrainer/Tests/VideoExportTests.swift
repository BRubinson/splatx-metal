// Cinema Task 5: pure-logic tests for the export pipeline — the frame schedule (seconds →
// frame count, timestamps, loop boundary), the resolution preset table in real pixels, the
// bitrate formula, the encoder settings dict (all three BT.709 keys spelled out), RGBA→BGRA
// conversion (pattern + round trip), fov→intrinsics, pose→COLMAP viewmat (a CameraConversion
// round trip), the contract that the overlay compositing rect reuses OverlayPlacement, the
// notification wording, the JobState.exporting label, the lockedTab export lock, and
// persistence of RunRecord.antialiased and Project.outputAspect. Verifying AVAssetWriter for
// real is the job of the smoke test (an actual render), with no unit mock — the policy of this
// app.
// FIX 2 (Task 5 review): exporter.cancel idempotence and partial-file cleanup (standing in for
// I1), the SPZ scratch path contract (M1), and the stale progress guard (M2).
import XCTest
import AVFoundation
import UniformTypeIdentifiers
import simd
@testable import SplatXMetalTrainer

final class VideoExportTests: XCTestCase {

    // MARK: - Frame schedule (t = i/fps)

    /// Open paths: the last frame includes t = duration (N = round(d·fps) + 1).
    func testFrameScheduleOpenIncludesDuration() {
        XCTAssertEqual(FrameSchedule.frameCount(duration: 2, fps: 30, loop: false), 61)
        XCTAssertEqual(FrameSchedule.time(ofFrame: 0, fps: 30), 0)
        XCTAssertEqual(FrameSchedule.time(ofFrame: 1, fps: 30), 1.0 / 30, accuracy: 1e-12)
        XCTAssertEqual(FrameSchedule.time(ofFrame: 60, fps: 30), 2.0, accuracy: 1e-12)
    }

    /// Loop paths: t = duration is the same pose as t = 0, so it is excluded
    /// (N = round(d·fps)).
    func testFrameScheduleLoopExcludesDuration() {
        XCTAssertEqual(FrameSchedule.frameCount(duration: 2, fps: 30, loop: true), 60)
        XCTAssertLessThan(FrameSchedule.time(ofFrame: 59, fps: 30), 2.0)
    }

    /// Rounding and floor: a duration shorter than one frame interval still yields at least
    /// 1 frame (loop) or 2 frames (open).
    func testFrameScheduleRoundingAndFloor() {
        XCTAssertEqual(FrameSchedule.frameCount(duration: 0.01, fps: 24, loop: true), 1)
        XCTAssertEqual(FrameSchedule.frameCount(duration: 0.01, fps: 24, loop: false), 2)
        XCTAssertEqual(FrameSchedule.frameCount(duration: 1.5, fps: 24, loop: true), 36)
        XCTAssertEqual(FrameSchedule.frameCount(duration: 10, fps: 60, loop: false), 601)
    }

    // MARK: - Resolution preset table (the preset = the short-side pixel count)

    func testResolutionTableAllCombos() {
        let table: [(ResolutionPreset, OutputAspect, Int, Int)] = [
            (.p1080, .wide, 1920, 1080), (.p1080, .tall, 1080, 1920), (.p1080, .square, 1080, 1080),
            (.p1440, .wide, 2560, 1440), (.p1440, .tall, 1440, 2560), (.p1440, .square, 1440, 1440),
            (.uhd4K, .wide, 3840, 2160), (.uhd4K, .tall, 2160, 3840), (.uhd4K, .square, 2160, 2160),
        ]
        for (preset, aspect, w, h) in table {
            let size = VideoExportSettings.pixelSize(preset: preset, aspect: aspect)
            XCTAssertEqual(size.width, w, "\(preset.rawValue) \(aspect.rawValue)")
            XCTAssertEqual(size.height, h, "\(preset.rawValue) \(aspect.rawValue)")
            XCTAssertEqual(size.width % 2, 0)   // the encoder requires even dimensions
            XCTAssertEqual(size.height % 2, 0)
        }
    }

    // MARK: - Bitrate (the bpp formula — 4K60 HEVC lands in the 50–80Mbps band the research
    // recommends)

    func testBitrateFormula() {
        let hevc4k60 = VideoExportSettings.bitrate(width: 3840, height: 2160, fps: 60, codec: .hevc)
        XCTAssertEqual(hevc4k60, 74_649_600)   // 3840·2160·60·0.15
        XCTAssertTrue((50_000_000...80_000_000).contains(hevc4k60))
        let h264_4k60 = VideoExportSettings.bitrate(width: 3840, height: 2160, fps: 60, codec: .h264)
        XCTAssertEqual(h264_4k60, 99_532_800)  // 3840·2160·60·0.20
        XCTAssertGreaterThan(h264_4k60, hevc4k60)
        // The same pixel count (16:9 ↔ 9:16) gives the same bitrate
        XCTAssertEqual(VideoExportSettings.bitrate(width: 1920, height: 1080, fps: 30, codec: .hevc),
                       VideoExportSettings.bitrate(width: 1080, height: 1920, fps: 30, codec: .hevc))
    }

    // MARK: - Encoder settings dict (all three BT.709 keys spelled out — guards the trap
    // where omitting them records 601)

    func testOutputSettingsBT709AndCompression() {
        let s = VideoExportSettings.outputSettings(codec: .hevc, width: 1920, height: 1080, fps: 30)
        XCTAssertEqual(s[AVVideoCodecKey] as? AVVideoCodecType, .hevc)
        XCTAssertEqual(s[AVVideoWidthKey] as? Int, 1920)
        XCTAssertEqual(s[AVVideoHeightKey] as? Int, 1080)
        let color = s[AVVideoColorPropertiesKey] as? [String: Any]
        XCTAssertEqual(color?[AVVideoColorPrimariesKey] as? String, AVVideoColorPrimaries_ITU_R_709_2)
        XCTAssertEqual(color?[AVVideoTransferFunctionKey] as? String, AVVideoTransferFunction_ITU_R_709_2)
        XCTAssertEqual(color?[AVVideoYCbCrMatrixKey] as? String, AVVideoYCbCrMatrix_ITU_R_709_2)
        let comp = s[AVVideoCompressionPropertiesKey] as? [String: Any]
        XCTAssertEqual(comp?[AVVideoAverageBitRateKey] as? Int,
                       VideoExportSettings.bitrate(width: 1920, height: 1080, fps: 30, codec: .hevc))
        XCTAssertEqual(comp?[AVVideoMaxKeyFrameIntervalKey] as? Int, 30)   // a 1-second GOP
        let h = VideoExportSettings.outputSettings(codec: .h264, width: 1280, height: 720, fps: 24)
        XCTAssertEqual(h[AVVideoCodecKey] as? AVVideoCodecType, .h264)
    }

    func testPixelBufferAttributes() {
        let a = VideoExportSettings.pixelBufferAttributes(width: 1920, height: 1080)
        XCTAssertEqual(a[kCVPixelBufferPixelFormatTypeKey as String] as? OSType,
                       kCVPixelFormatType_32BGRA)
        XCTAssertEqual(a[kCVPixelBufferWidthKey as String] as? Int, 1920)
        XCTAssertEqual(a[kCVPixelBufferHeightKey as String] as? Int, 1080)
        XCTAssertEqual(a[kCVPixelBufferMetalCompatibilityKey as String] as? Bool, true)
        XCTAssertNotNil(a[kCVPixelBufferIOSurfacePropertiesKey as String])   // IOSurface backing
    }

    // MARK: - RGBA→BGRA (session output → the CVPixelBuffer channel order)

    func testRGBAtoBGRAPattern() {
        // 2×2 pixels, each with a distinguishable RGBA value
        let rgba: [UInt8] = [
            10, 20, 30, 255,   40, 50, 60, 255,
            70, 80, 90, 128,   1, 2, 3, 0,
        ]
        let bgra = PixelPack.rgbaToBGRA(rgba, width: 2, height: 2)
        XCTAssertEqual(Array(bgra[0..<4]), [30, 20, 10, 255])
        XCTAssertEqual(Array(bgra[4..<8]), [60, 50, 40, 255])
        XCTAssertEqual(Array(bgra[8..<12]), [90, 80, 70, 128])
        XCTAssertEqual(Array(bgra[12..<16]), [3, 2, 1, 0])
    }

    /// The R↔B swap is an involution — applying it twice returns the original (round-trip
    /// check).
    func testRGBAtoBGRARoundtrip() {
        var rgba: [UInt8] = []
        for i in 0..<(3 * 2 * 4) { rgba.append(UInt8((i * 37) % 256)) }
        let twice = PixelPack.rgbaToBGRA(PixelPack.rgbaToBGRA(rgba, width: 3, height: 2),
                                         width: 3, height: 2)
        XCTAssertEqual(twice, rgba)
    }

    /// dst rowBytes padding (the CVPixelBuffer alignment convention) — the pixels outside
    /// the padding are written exactly.
    func testRGBAtoBGRAWithRowPadding() {
        let rgba: [UInt8] = [255, 0, 0, 255,   0, 255, 0, 255]   // 2×1: red, green
        let rowBytes = 16   // 2px·4B = 8 + 8 bytes of padding
        var dst = [UInt8](repeating: 0xAA, count: rowBytes)
        rgba.withUnsafeBufferPointer { src in
            dst.withUnsafeMutableBufferPointer { d in
                PixelPack.rgbaToBGRA(src: src.baseAddress!, width: 2, height: 1,
                                     dst: d.baseAddress!, dstBytesPerRow: rowBytes)
            }
        }
        XCTAssertEqual(Array(dst[0..<4]), [0, 0, 255, 255])    // BGRA red
        XCTAssertEqual(Array(dst[4..<8]), [0, 255, 0, 255])    // BGRA green
    }

    // MARK: - fov → intrinsics (fy = H/(2·tan(fov/2)), fx = fy for square pixels)

    func testIntrinsicsFromFOV() {
        let k = ExportRenderMath.intrinsics(verticalFOV: .pi / 2, width: 1920, height: 1080)
        XCTAssertEqual(k.fy, 540, accuracy: 1e-3)   // 540/tan(45°)
        XCTAssertEqual(k.fx, k.fy)                  // square pixels — W carries the aspect ratio
        XCTAssertEqual(k.cx, 960)
        XCTAssertEqual(k.cy, 540)
        // fx == fy at other aspect ratios too (the fixed-vertical-fov contract)
        let tall = ExportRenderMath.intrinsics(verticalFOV: 0.9, width: 1080, height: 1920)
        XCTAssertEqual(tall.fx, tall.fy)
    }

    // MARK: - pose → COLMAP viewmat (row-major world→cam)

    /// An identity pose (origin, identity Metal rotation) → the COLMAP view matrix is the
    /// flip diag(1,−1,−1).
    func testColmapViewmatIdentityPose() {
        let vm = ExportRenderMath.colmapViewmatRowMajor(
            position: .zero, rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1))
        let expected: [Float] = [1, 0, 0, 0,
                                 0, -1, 0, 0,
                                 0, 0, -1, 0,
                                 0, 0, 0, 1]
        XCTAssertEqual(vm.count, 16)
        for (i, e) in expected.enumerated() {
            XCTAssertEqual(vm[i], e, accuracy: 1e-6, "index \(i)")
        }
    }

    /// Round trip: COLMAP (qvec,tvec) → the Metal pose (cameraCenter + metalCameraRotation —
    /// reusing the Task 4 verification functions) → colmapViewmat must recover the original
    /// [R|t].
    func testColmapViewmatRoundtripWithCameraConversion() {
        let qvec = SIMD4<Double>(0.82, 0.31, -0.41, 0.25)   // non-unit — rotation() normalizes it
        let tvec = SIMD3<Double>(0.4, -1.2, 2.5)
        let pos = CameraConversion.cameraCenter(qvec: qvec, tvec: tvec)
        let rot = PathPresets.metalCameraRotation(qvec: qvec)
        let vm = ExportRenderMath.colmapViewmatRowMajor(position: pos, rotation: rot)
        let R = CameraConversion.rotation(qvec)
        let t = SIMD3<Float>(Float(tvec.x), Float(tvec.y), Float(tvec.z))
        for r in 0..<3 {
            XCTAssertEqual(vm[r * 4 + 0], R.columns.0[r], accuracy: 1e-4, "R[\(r)][0]")
            XCTAssertEqual(vm[r * 4 + 1], R.columns.1[r], accuracy: 1e-4, "R[\(r)][1]")
            XCTAssertEqual(vm[r * 4 + 2], R.columns.2[r], accuracy: 1e-4, "R[\(r)][2]")
            XCTAssertEqual(vm[r * 4 + 3], t[r], accuracy: 1e-4, "t[\(r)]")
        }
        XCTAssertEqual(Array(vm[12..<16]), [0, 0, 0, 1])
    }

    // MARK: - Reuse of the overlay compositing rect (the contract that it is the same
    // function as the Task 4 WYSIWYG path)

    func testOverlayCompositorReusesOverlayPlacementRect() throws {
        let ctx = try XCTUnwrap(CGContext(
            data: nil, width: 4, height: 2, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        let image = try XCTUnwrap(ctx.makeImage())
        let spec = OverlaySpec(imagePath: "overlay/logo.png", anchor: .bottomTrailing,
                               offsetX: 3, offsetY: -2, sizePercent: 25, margin: 5,
                               opacity: 0.7)
        let comp = RenderJob.overlayCompositor(spec: spec, image: image,
                                               outputWidth: 1920, outputHeight: 1080)
        let expected = OverlayPlacement.rect(spec: spec, imageSize: CGSize(width: 4, height: 2),
                                             outputWidth: 1920, outputHeight: 1080)
        XCTAssertEqual(comp.rect, expected)
        XCTAssertEqual(comp.opacity, 0.7)
    }

    // MARK: - Notification wording (the convention of AppNotifier's existing four — render
    // finished/failed added)

    func testRenderNotificationTexts() {
        let done = AppNotificationText.renderFinished(project: "garden", fileName: "garden.mov",
                                                      frames: 61, seconds: 12.34)
        XCTAssertEqual(done.title, "Render complete")
        XCTAssertEqual(done.body, "garden — garden.mov · 61 frames · 12.3s")
        let failed = AppNotificationText.renderFailed(
            project: "garden", error: ExportError.convertFailed("boom"))
        XCTAssertEqual(failed.title, "Render failed")
        XCTAssertEqual(failed.body, "garden — Convert failed: boom")
    }

    // MARK: - JobState.exporting + lockedTab

    func testExportingJobLabel() {
        // Unified terminology (footnote 2 ④) — the same vocabulary as the "Render Video" CTA.
        XCTAssertEqual(JobState.exporting(frame: 3, total: 61).label, "render 3/61")
    }

    @MainActor func testStageLockIncludesExporting() {
        // Footnote 2 ①: the export starts from the Cinema tab, so the lock is .cinema too.
        XCTAssertEqual(ProjectSession.lockedTab(isTraining: false, isAligning: false,
                                                isExporting: true), .cinema)
        // Priority is training > align > export (reaching them at the same time is
        // structurally impossible — this only pins determinism)
        XCTAssertEqual(ProjectSession.lockedTab(isTraining: true, isAligning: false,
                                                isExporting: true), .train)
        XCTAssertNil(ProjectSession.lockedTab(isTraining: false, isAligning: false,
                                              isExporting: false))
    }

    // MARK: - Containers (footnote 2 ⑤ — MP4 by default / MOV, with AVFileType, the file
    // extension, and the panel type consistent)

    @MainActor func testVideoContainerMappingAndDefault() {
        XCTAssertEqual(VideoContainer.mp4.avFileType, .mp4)
        XCTAssertEqual(VideoContainer.mov.avFileType, .mov)
        XCTAssertEqual(VideoContainer.mp4.fileExtension, "mp4")
        XCTAssertEqual(VideoContainer.mov.fileExtension, "mov")
        XCTAssertEqual(VideoContainer.mp4.contentType, .mpeg4Movie)
        XCTAssertEqual(VideoContainer.mov.contentType, .quickTimeMovie)
        XCTAssertEqual(CinemaViewModel().exportContainer, .mp4)   // default = MP4
    }

    // MARK: - RunRecord.antialiased (forward compatibility with old manifests — false when
    // absent)

    func testRunRecordAntialiasedDecodeDefaultsFalse() throws {
        let json = """
        {"id":"20260726-000000","outputFile":"output.ply","outcome":"completed",
         "splatCount":1000,"optionsSummary":"s","createdAt":"2026-07-26T00:00:00Z"}
        """
        let r = try JSONDecoder.project.decode(RunRecord.self, from: Data(json.utf8))
        XCTAssertFalse(r.antialiased)
    }

    func testRunRecordAntialiasedRoundtrip() throws {
        let r = RunRecord(id: "20260726-000001", outputFile: "output.ply", previewFile: nil,
                          outcome: "completed", finalPSNR: nil, splatCount: 1,
                          optionsSummary: "s", createdAt: Date(), antialiased: true)
        let data = try JSONEncoder.project.encode(r)
        let back = try JSONDecoder.project.decode(RunRecord.self, from: data)
        XCTAssertTrue(back.antialiased)
    }

    // MARK: - Project.outputAspect persistence (recommended by the Task 4 review —
    // project-level decodeIfPresent)

    func testProjectOutputAspectPersistence() throws {
        var p = Project(id: UUID(), name: "p", sourceKind: .images, state: .trained,
                        imageCount: 1, alignSummary: nil, runs: [], createdAt: Date())
        XCTAssertNil(p.outputAspect)   // default — the same as an old manifest
        p.outputAspect = .tall
        let back = try JSONDecoder.project.decode(Project.self,
                                                  from: JSONEncoder.project.encode(p))
        XCTAssertEqual(back.outputAspect, .tall)
        // An old manifest (key absent) → nil, and an unknown string → nil (lenient decode)
        let old = """
        {"id":"\(UUID().uuidString)","name":"p","sourceKind":"images","state":"trained",
         "imageCount":1,"runs":[],"createdAt":"2026-07-26T00:00:00Z"}
        """
        XCTAssertNil(try JSONDecoder.project.decode(Project.self, from: Data(old.utf8)).outputAspect)
        let junk = """
        {"id":"\(UUID().uuidString)","name":"p","sourceKind":"images","state":"trained",
         "imageCount":1,"runs":[],"createdAt":"2026-07-26T00:00:00Z","outputAspect":"21:9"}
        """
        XCTAssertNil(try JSONDecoder.project.decode(Project.self, from: Data(junk.utf8)).outputAspect)
    }

    // MARK: - CinemaViewModel export settings

    @MainActor func testSetFPSValidatesChoices() {
        let vm = CinemaViewModel()
        XCTAssertEqual(vm.fps, 30)
        vm.setFPS(60)
        XCTAssertEqual(vm.fps, 60)
        vm.setFPS(25)   // outside the allowed set — ignored
        XCTAssertEqual(vm.fps, 60)
        vm.setFPS(24)
        XCTAssertEqual(vm.fps, 24)
    }

    @MainActor func testSetOutputAspectPersists() {
        let vm = CinemaViewModel()
        var persisted: [OutputAspect] = []
        vm.onPersistAspect = { persisted.append($0) }
        vm.setOutputAspect(.square)
        XCTAssertEqual(vm.outputAspect, .square)
        vm.setOutputAspect(.square)   // no change — no re-persist
        XCTAssertEqual(persisted, [.square])
        // Injection through load (restoring persisted state) — the value only, no callback
        vm.load(paths: [], overlay: nil, outputAspect: .tall)
        XCTAssertEqual(vm.outputAspect, .tall)
        XCTAssertEqual(persisted, [.square])
    }

    // MARK: - FIX 2 (Task 5 review I1, M1, and M2)

    /// A stand-in test for I1 — deterministically inducing a finish() failure would require
    /// injecting a failure inside the encoder, which is impossible in a unit test (this app's
    /// policy is that a real smoke test covers it, with no AVAssetWriter mock).
    /// Instead it pins the contract the I1 catch (`exporter.cancel()`) relies on directly: in
    /// .writing, cancel() does cancelWriting + deletes the partial file; in a terminal state
    /// (the same branch as status != .writing after a failed finishWriting) it only does
    /// removeItem, and it is safe to call again.
    func testExporterCancelIdempotentAndRemovesPartialFile() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("sx-test-cancel-\(UUID().uuidString).mov")
        defer { try? FileManager.default.removeItem(at: url) }
        let exporter = try VideoExporter(
            outputURL: url,
            config: .init(codec: .h264, container: .mov, width: 64, height: 64, fps: 24))
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))   // created by startWriting
        exporter.cancel()   // the .writing branch — cancelWriting + delete the partial file
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
        exporter.cancel()   // the terminal-state branch (same as the I1 catch) — removeItem only, idempotent
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
    }

    /// M1: the SPZ→PLY scratch path contract — the same source yields the same path (reused
    /// by overwriting, which removes the leak where every export piled up a new UUID file); a
    /// different source yields a different path (even for an identically named output.spz, the
    /// digest of the full path distinguishes them); and the file is pinned to sx-export-*.ply
    /// in the tmp directory.
    func testSpzScratchURLStablePerSourceAndUniqueAcrossSources() {
        let a = RenderJob.spzScratchURL(forSource: "/p/runs/20260726-000001/output.spz")
        let b = RenderJob.spzScratchURL(forSource: "/p/runs/20260726-000001/output.spz")
        let c = RenderJob.spzScratchURL(forSource: "/q/runs/20260726-000001/output.spz")
        XCTAssertEqual(a, b)          // the reuse contract — bounded at one file per source
        XCTAssertNotEqual(a, c)       // a different run/project → a different scratch file (no invalidation needed)
        XCTAssertEqual(a.pathExtension, "ply")
        XCTAssertTrue(a.lastPathComponent.hasPrefix("sx-export-"))
        XCTAssertEqual(a.deletingLastPathComponent().standardizedFileURL.path,
                       FileManager.default.temporaryDirectory.standardizedFileURL.path)
    }

    /// M2: a progress callback that arrives late, after completion (job = .idle), cannot roll
    /// the state back — the isExporting guard in applyProgress closes the reordering window (a
    /// delayed progress Task turning .idle back into .exporting).
    @MainActor func testStaleProgressAfterCompletionIgnored() {
        let job = RenderJob()
        let model = AppModel()
        XCTAssertFalse(job.isExporting)   // initial = the same idle state as after completion
        job.applyProgress(frame: 5, total: 61, model: model)
        XCTAssertEqual(job.frame, 0)      // progress state unchanged
        XCTAssertEqual(model.job, .idle)  // not reverted to .exporting
    }
}
