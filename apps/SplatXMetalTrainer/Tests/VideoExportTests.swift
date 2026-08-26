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

    /// 7 presets × 3 aspect ratios. The lower four rows arrived with Cinema GIF (2026-08-05 spec
    /// §5.3) and are the reason `short / 9 * 16` had to go: 240 and 480 are not multiples of 9, so
    /// the old integer division produced 416 (true 426.67) and 848 (true 853.33). The corrected
    /// values are 428 and 854.
    /// **The nine rows that existed before are unchanged** — that is the regression this table is
    /// really guarding (1920/2560/3840 came out identical under both formulas).
    func testResolutionTableAllCombos() {
        let table: [(ResolutionPreset, OutputAspect, Int, Int)] = [
            (.p240, .wide, 428, 240), (.p240, .tall, 240, 428), (.p240, .square, 240, 240),
            (.p360, .wide, 640, 360), (.p360, .tall, 360, 640), (.p360, .square, 360, 360),
            (.p480, .wide, 854, 480), (.p480, .tall, 480, 854), (.p480, .square, 480, 480),
            (.p720, .wide, 1280, 720), (.p720, .tall, 720, 1280), (.p720, .square, 720, 720),
            (.p1080, .wide, 1920, 1080), (.p1080, .tall, 1080, 1920), (.p1080, .square, 1080, 1080),
            (.p1440, .wide, 2560, 1440), (.p1440, .tall, 1440, 2560), (.p1440, .square, 1440, 1440),
            (.uhd4K, .wide, 3840, 2160), (.uhd4K, .tall, 2160, 3840), (.uhd4K, .square, 2160, 2160),
        ]
        XCTAssertEqual(table.count, ResolutionPreset.allCases.count * OutputAspect.allCases.count)
        for (preset, aspect, w, h) in table {
            let size = VideoExportSettings.pixelSize(preset: preset, aspect: aspect)
            XCTAssertEqual(size.width, w, "\(preset.rawValue) \(aspect.rawValue)")
            XCTAssertEqual(size.height, h, "\(preset.rawValue) \(aspect.rawValue)")
            XCTAssertEqual(size.width % 2, 0,                  // the encoder requires even dimensions
                           "odd width for \(preset.rawValue) \(aspect.rawValue)")
            XCTAssertEqual(size.height % 2, 0,
                           "odd height for \(preset.rawValue) \(aspect.rawValue)")
            XCTAssertEqual(min(size.width, size.height), preset.shortSide,
                           "the preset names the short side — \(preset.rawValue)")
        }
    }

    // MARK: - CinemaExportFormat (2026-08-05 spec §5.2 — the enum that exists so that adding GIF
    // to VideoContainer, where three ternaries would swallow it, never happens)

    func testCinemaExportFormatAllProperties() {
        XCTAssertEqual(CinemaExportFormat.allCases, [.mp4, .mov, .gif])
        XCTAssertEqual(CinemaExportFormat.mp4.videoContainer, .mp4)
        XCTAssertEqual(CinemaExportFormat.mov.videoContainer, .mov)
        XCTAssertNil(CinemaExportFormat.gif.videoContainer)   // nil == "not AVAssetWriter"
        XCTAssertEqual(CinemaExportFormat.mp4.fileExtension, "mp4")
        XCTAssertEqual(CinemaExportFormat.mov.fileExtension, "mov")
        XCTAssertEqual(CinemaExportFormat.gif.fileExtension, "gif")
        XCTAssertEqual(CinemaExportFormat.mp4.contentType, .mpeg4Movie)
        XCTAssertEqual(CinemaExportFormat.mov.contentType, .quickTimeMovie)
        XCTAssertEqual(CinemaExportFormat.gif.contentType, .gif)
        // The save panel's extension and content type must agree for every case — the mismatch
        // that "an extension-only .gif MOV" would have been.
        for f in CinemaExportFormat.allCases {
            XCTAssertEqual(f.contentType.preferredFilenameExtension, f.fileExtension, f.rawValue)
        }
        // Where a container exists it must agree with the container's own mapping (one source of
        // truth for the video path).
        for f in CinemaExportFormat.allCases {
            guard let c = f.videoContainer else { continue }
            XCTAssertEqual(c.fileExtension, f.fileExtension, f.rawValue)
            XCTAssertEqual(c.contentType, f.contentType, f.rawValue)
        }
    }

    /// Regression guard for the second blast radius in spec §5.2: the training timelapse decides
    /// "GIF" by `videoContainer == nil` (TimelapseEncoder's `guard let container = … else
    /// { encodeGIF }`). If anyone ever gives gif a container, the timelapse GIF leaks into
    /// AVAssetWriter — this test fails first.
    func testTimelapseGifDiscriminatorStillNil() {
        XCTAssertNil(TimelapseFormat.gif.videoContainer)
        XCTAssertEqual(TimelapseFormat.mp4.videoContainer, .mp4)
        XCTAssertEqual(TimelapseFormat.mov.videoContainer, .mov)
        XCTAssertEqual(VideoContainer.allCases.count, 2)   // still exactly MP4/MOV
    }

    // MARK: - Format → resolution choices (spec §5.4 — a deliberate departure: the UI restricts
    // GIF up front, unlike the timelapse's after-the-fact clamp report)

    func testResolutionChoicesPerFormat() {
        XCTAssertEqual(ResolutionPreset.choices(for: .mp4), [.p1080, .p1440, .uhd4K])
        XCTAssertEqual(ResolutionPreset.choices(for: .mov), [.p1080, .p1440, .uhd4K])
        XCTAssertEqual(ResolutionPreset.choices(for: .gif), [.p240, .p360, .p480, .p720])
        XCTAssertEqual(ResolutionPreset.choices(for: .mp4).count, 3)
        XCTAssertEqual(ResolutionPreset.choices(for: .gif).count, 4)
        // Disjoint — so a format switch always needs the clamp (there is no value that survives
        // both lists and could hide a missing snap).
        let video = Set(ResolutionPreset.choices(for: .mp4))
        let gif = Set(ResolutionPreset.choices(for: .gif))
        XCTAssertTrue(video.isDisjoint(with: gif))
        // Together they cover every case — no preset is unreachable from the UI.
        XCTAssertEqual(video.union(gif), Set(ResolutionPreset.allCases))
        // The GIF ceiling is the resource argument, not a style choice.
        XCTAssertEqual(gif.map(\.shortSide).max(), 720)
    }

    // MARK: - setExportFormat's resolution clamp (spec §5.3 — without it the Size segmented
    // control holds a value that is not one of its options, which AppKit draws as "nothing
    // selected" while the export quietly keeps using the invisible value)

    @MainActor func testSetExportFormatClampsResolutionIntoTheNewList() {
        let vm = CinemaViewModel()
        XCTAssertEqual(vm.exportFormat, .mp4)          // default = MP4
        XCTAssertEqual(vm.exportResolution, .p1080)

        // MP4 → GIF: 1080p is not offered for GIF, so it snaps to the GIF list's first entry.
        vm.setExportFormat(.gif)
        XCTAssertEqual(vm.exportFormat, .gif)
        XCTAssertEqual(vm.exportResolution, .p240)
        XCTAssertTrue(ResolutionPreset.choices(for: .gif).contains(vm.exportResolution))

        // GIF → MP4: the return trip must snap too — 240p is not a video option.
        vm.setExportFormat(.mp4)
        XCTAssertEqual(vm.exportFormat, .mp4)
        XCTAssertEqual(vm.exportResolution, .p1080)
        XCTAssertTrue(ResolutionPreset.choices(for: .mp4).contains(vm.exportResolution))
    }

    @MainActor func testSetExportFormatKeepsAResolutionTheNewListOffers() {
        // MP4 → MOV share one list, so a user's 4K choice must survive the switch. A clamp that
        // reset unconditionally would silently downgrade the output to 1080p.
        let vm = CinemaViewModel()
        vm.exportResolution = .uhd4K
        vm.setExportFormat(.mov)
        XCTAssertEqual(vm.exportFormat, .mov)
        XCTAssertEqual(vm.exportResolution, .uhd4K)
    }

    @MainActor func testSetExportFormatSameFormatIsNoOp() {
        // Re-selecting the current format must not touch the resolution. Pinned with a value the
        // current format does not offer, so a clamp that ran unguarded would be visible here —
        // the equality guard is load-bearing, not an optimization.
        let vm = CinemaViewModel()
        vm.exportResolution = .p240                    // out of the MP4 list on purpose
        vm.setExportFormat(.mp4)                       // same format as the default
        XCTAssertEqual(vm.exportFormat, .mp4)
        XCTAssertEqual(vm.exportResolution, .p240)
    }

    // MARK: - The console's start line (GIF has no codec axis)

    func testConsoleFormatSummaryDropsCodecForGif() {
        XCTAssertEqual(RenderJob.consoleFormatSummary(codec: .h264, format: .mp4), "H.264 MP4")
        XCTAssertEqual(RenderJob.consoleFormatSummary(codec: .hevc, format: .mov), "HEVC MOV")
        // "H.264 GIF" would name a setting that had no effect on the file.
        XCTAssertEqual(RenderJob.consoleFormatSummary(codec: .h264, format: .gif), "GIF")
        XCTAssertEqual(RenderJob.consoleFormatSummary(codec: .hevc, format: .gif), "GIF")
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
        // The VM's default is stated on the format axis now (CinemaExportFormat), which is what
        // the save panel and the sink branch read — VideoContainer stayed a two-case AVFileType
        // mapping (spec §5.2).
        XCTAssertEqual(CinemaViewModel().exportFormat, .mp4)   // default = MP4
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
        // 15 (2026-08-05 spec §3) — a member of the set, so it is accepted like any other. The
        // guard is `fpsChoices.contains`, so this would silently keep working after a bad edit to
        // the list; testFifteenIsAnOfferedRate below is what actually pins the list.
        vm.setFPS(15)
        XCTAssertEqual(vm.fps, 15)
    }

    /// The Rate picker's options are `fpsChoices` verbatim (RenderJob's SegmentedRow maps over it),
    /// so this is the test that says 15 is *offered*, not merely tolerated by setFPS.
    ///
    /// The order is asserted too: SegmentedControl lays the options out in array order, and a Rate
    /// row that reads 24 · 15 · 30 · 60 would be a real UI defect that no other test would catch.
    @MainActor func testFifteenIsAnOfferedRate() {
        XCTAssertEqual(VideoExportSettings.fpsChoices, [15, 24, 30, 60])
        // Every offered rate must survive its own validator — the picker cannot present a choice
        // that setFPS then refuses (the segment would read as inert).
        for rate in VideoExportSettings.fpsChoices {
            let vm = CinemaViewModel()
            vm.setFPS(rate)
            XCTAssertEqual(vm.fps, rate, "the Rate picker offers \(rate) but setFPS refused it")
        }
    }

    /// §3.1 — 15fps must not silently land on `bitrate`'s 2Mbps floor. The floor exists to keep a
    /// tiny frame from being encoded at an unusable rate, but a *selection* that reaches it means
    /// the user's resolution choice has stopped affecting quality, and at low fps that becomes
    /// reachable: the smallest combination Cinema can produce (1:1 1080p, HEVC's lower 0.15 bpp)
    /// is 2.62Mbps at 15fps, which clears it, while roughly 12fps and below would not. Pinned here
    /// so a future lower rate has to confront the arithmetic instead of inheriting this claim.
    func testSmallestFifteenFpsSelectionStaysAboveTheBitrateFloor() {
        let floor = 2_000_000
        for aspect in OutputAspect.allCases {
            for preset in ResolutionPreset.videoChoices {
                let size = VideoExportSettings.pixelSize(preset: preset, aspect: aspect)
                for codec in VideoCodec.allCases {
                    let rate = VideoExportSettings.bitrate(width: size.width, height: size.height,
                                                           fps: 15, codec: codec)
                    XCTAssertGreaterThan(rate, floor,
                                         "\(preset) \(aspect) \(codec) at 15fps sits on the floor")
                }
            }
        }
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
