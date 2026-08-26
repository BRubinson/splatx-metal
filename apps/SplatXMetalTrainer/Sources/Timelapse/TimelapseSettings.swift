// Training timelapse (2026-08-04 spec §3): the settings model plus the pure math the capture,
// encoding, and UI layers all read. Everything here is device-independent and is the test
// surface of this feature — the renderer, the GPU readback, and SwiftUI are manual-check
// territory (the app's no-tests-for-views policy).
//
// Why the capture period is restricted to multiples of 50 (spec §3.7): the live snapshot
// period is fixed at TrainViewModel.snapshotEvery = 50 and a capture is only possible right
// after a swap, so a period that is not a multiple of 50 would snap to the 50 grid and produce
// a **different real interval than the user asked for**. The Stepper never lets that state
// exist, and `isValidEverySteps` states the same rule as a value.
//
// 주의: 50 배수라고 해서 실제 간격이 요청 간격과 **항상** 같지는 않다 — SnapshotMailbox가
// 미소비 스냅샷을 덮어쓰므로(LiveSplatStreamer.swift:88-98) 배수 iter가 파이프라인에 아예
// 도달하지 않을 수 있다. 런타임 due 판정은 이 순수 함수가 아니라 TimelapseFrameLedger.claim
// (임계값 방식)이 담당한다. 여기 shouldCapture는 UI 표시·순수 판정용으로만 남는다.
import Foundation
import CoreGraphics

/// Where the timelapse frame's **aspect ratio** comes from (2026-08-05 spec §2.1).
///
/// 이 축이 생긴 이유: 한 그림의 출처가 둘로 갈려 있었다. 캡처 카메라의 **포즈**는 뷰포트
/// 카메라의 스냅샷인데(`TrainViewModel.start` → `CameraPose.effective(of: viewerVM.camera)`)
/// 종횡비만 학습 이미지 intrinsics에 묶여 있었고, 그래서 사용자가 "현재 보고 있는 뷰포트
/// 기준으로 나오는 줄 알았다"고 한 어긋남이 생겼다.
///
/// 화각 변환은 필요 없다(spec §1.1): 뷰포트와 캡처가 **같은** `CameraController.perspective`를
/// 같은 `fovY`로 부르고 다른 것은 `aspect` 인자 하나뿐이라, 수직 화각은 이미 일치한다. 좌우만
/// 더 넓거나 잘려 있었을 뿐이므로 프레임의 종횡비 하나만 바꾸면 "보이는 그대로"가 된다 —
/// 이 열거형이 투영 수식이 아니라 프레임 크기만 고르는 이유다.
enum TimelapseAspectSource: String, Codable, CaseIterable, Identifiable {
    /// 화면에서 본 그대로 — 기본값. "보이는 그대로"가 놀람이 없는 쪽이라는 판단이고(사용자
    /// 결정 1), 포즈가 이미 뷰포트를 따라가고 있었으므로 종횡비까지 맞춰야 출처가 하나가 된다.
    ///
    /// **"뷰포트"는 학습 시작 시점의 뷰포트를 뜻한다.** `TimelapseRecorder.Config`는 캡처
    /// 스레드가 읽는 불변 값이고(사용자 결정 3), 프레임 크기는 레코더 생성 시점에 고정된다 —
    /// 학습 중에 창 크기를 바꿔도 이미 찍힌 프레임과 크기가 갈라지지 않는다. GIF/비디오 한 편의
    /// 프레임 크기가 중간에 변할 수 없으므로 이것은 제약이 아니라 요구사항이다. "여기서 찍기"도
    /// 포즈만 갈아끼우고 종횡비는 그대로 둔다(같은 이유).
    case viewport
    /// 학습 이미지와 같은 비율(첫 COLMAP 카메라의 W:H) — 2026-08-04 라운드의 동작이다.
    case dataset

    var id: String { rawValue }
    var displayName: String {
        switch self {
        case .viewport: "Viewport"
        case .dataset: "Dataset"
        }
    }
}

/// The container a training timelapse is written to. The default lives on
/// `TimelapseSettings.format` below, where its reasoning is; this enum only enumerates the
/// choices and states how each one encodes (`videoContainer`).
///
/// FIX 1 (2026-08-05): this comment used to claim "MP4 by default … the same judgement as
/// Cinema's VideoContainer default", and both halves were false — the default became `.gif` in
/// the previous round (Task 16, bcf547a), directly contradicting the comment 30 lines below it,
/// and `CinemaViewModel.exportContainer` no longer exists (it is `exportFormat:
/// CinemaExportFormat` now). A default documented in two places is a default documented in
/// neither, so it is stated once, at the property.
enum TimelapseFormat: String, Codable, CaseIterable, Identifiable {
    case mp4
    case mov
    case gif

    var id: String { rawValue }
    var fileExtension: String { rawValue }
    var displayName: String {
        switch self {
        case .mp4: "MP4"
        case .mov: "MOV"
        case .gif: "GIF"
        }
    }

    /// The Cinema container this format encodes through — nil for GIF, which goes to ImageIO
    /// instead of AVAssetWriter (spec §3.4).
    var videoContainer: VideoContainer? {
        switch self {
        case .mp4: .mp4
        case .mov: .mov
        case .gif: nil
        }
    }
}

/// The per-project timelapse settings (persisted in project.json — spec §3.6).
struct TimelapseSettings: Codable, Equatable {
    var enabled: Bool = false
    var everySteps: Int = 100
    /// GIF by default: a training timelapse is something you glance at and pass along, and a GIF
    /// plays inline everywhere without a player. The video formats stay for keeping one.
    var format: TimelapseFormat = .gif
    var fps: Int = 15
    /// 480 by default — 854×480 at 16:9, which stays under `gifMaxLongSide`, so the number picked
    /// here **is** the resulting frame size. The old 1080 default silently became 540p as soon as
    /// the format was GIF.
    var shortSide: Int = 480
    /// Viewport by default (2026-08-05 spec §2.1, 사용자 결정 1). 기존 프로젝트의 결과물도
    /// 달라진다 — 옛 매니페스트에는 키가 없어 이 기본값을 받기 때문이다. 타임랩스가 기본 꺼짐인
    /// 도입 이틀째 기능이라 감수한 변경이고, 이전 동작이 필요하면 `.dataset`이 그 값이다.
    var aspectSource: TimelapseAspectSource = .viewport
    var overlay: Bool = true

    // The coding keys are declared explicitly so the forward-compatible decoder (Task 9, in an
    // extension) can name them. Declaring a nested type does **not** suppress the memberwise
    // and default initializers — declaring an `init` in the body would, which is exactly why
    // that decoder lives in an extension.
    // 새 필드는 **세 곳**(저장 프로퍼티 · 여기 · 아래 decodeIfPresent)에 동시에 등재해야 한다.
    // 하나라도 빠지면 컴파일은 통과하고 값만 조용히 증발한다.
    enum CodingKeys: String, CodingKey {
        case enabled, everySteps, format, fps, shortSide, aspectSource, overlay
    }

    // MARK: Choice sets (the single source of truth for the UI and for validation)

    static let fpsChoices = [10, 15, 24, 30]
    /// The low end (240/360/480) exists for the GIF default: those sizes stay under
    /// `gifMaxLongSide` at 16:9, so the choice made here survives to the file instead of being
    /// clamped. The high end stays for keeping a run as MP4/MOV.
    static let shortSideChoices = [240, 360, 480, 720, 1080, 1440]
    /// The aspect-ratio sources the Picker offers. Declared here next to the other choice sets
    /// rather than being read as `TimelapseAspectSource.allCases` at each call site so the form and
    /// `validationErrors` are pinned to **one** list: the day a case is offered conditionally (a
    /// plausible future — `.dataset` is meaningless for a project with no COLMAP intrinsics) the
    /// two cannot drift apart. Today it is every case, so validation cannot fire; the test that
    /// every offered choice validates is what keeps that true.
    static let aspectSourceChoices = TimelapseAspectSource.allCases
    static let everyStepsRange = 50...2000
    static let everyStepsStep = 50

    /// GIF auto-reduction (spec §3.4): a 256-colour palette with no inter-frame compression
    /// makes a full-resolution GIF hundreds of MB.
    static let gifMaxLongSide = 960
    static let gifMaxFrames = 200

    // MARK: Capture period

    static func isValidEverySteps(_ n: Int) -> Bool {
        everyStepsRange.contains(n) && n % everyStepsStep == 0
    }

    /// Snaps a value to the 50 grid and clamps it into range (the guard for a value restored
    /// from an old manifest or typed in by hand).
    static func normalizedEverySteps(_ n: Int) -> Int {
        let stepped = Int((Double(n) / Double(everyStepsStep)).rounded()) * everyStepsStep
        return min(max(stepped, everyStepsRange.lowerBound), everyStepsRange.upperBound)
    }

    /// Whether the swap at `iter` is a capture point. Iteration 0 is excluded (there is no swap
    /// before the first snapshot).
    ///
    /// **런타임 게이트가 아니다** — 메일박스 드롭 때문에 실제 due 판정은
    /// `TimelapseFrameLedger.claim(iter:everySteps:)`(임계값 방식)이 담당한다. 이 함수는
    /// UI 표시·순수 판정용이다.
    func shouldCapture(iter: Int) -> Bool {
        guard enabled, iter > 0 else { return false }
        return iter % max(everySteps, 1) == 0
    }

    // MARK: Frame naming

    /// Zero padded to five digits so lexicographic order equals capture order — the encoder
    /// lists the directory and sorts by name (TimelapseEncoder.frameURLs).
    static func frameFileName(index: Int) -> String {
        String(format: "frame-%05d.jpg", index)
    }

    // MARK: Resolution

    /// Rounds up to the nearest even number (floor 2) — video encoders reject odd dimensions.
    static func evenUp(_ v: Int) -> Int {
        let x = max(v, 2)
        return x % 2 == 0 ? x : x + 1
    }

    /// The capture resolution: the short side is the preset and the long side follows the ratio of
    /// the pair handed in.
    ///
    /// **The pair is consumed as a ratio and nothing else** — this used to name the first COLMAP
    /// camera's W:H as a fact, and since 2026-08-05 (spec §2.2) the caller chooses between the
    /// Train viewport and those intrinsics through `aspectPair`, so naming one here would be
    /// wrong half the time. Being a ratio is also why the viewport's **pt** size can be passed
    /// straight in: pt ratio == px ratio, so no backing-scale conversion is involved.
    /// A degenerate pair (a non-positive dimension = no ratio available from either source) falls
    /// back to 16:9 landscape. That fallback is the single place "no ratio anywhere" is answered —
    /// `aspectPair` deliberately forwards the degenerate pair here instead of repeating 16:9.
    static func frameSize(shortSide: Int, sourceWidth: Int, sourceHeight: Int)
        -> (width: Int, height: Int) {
        let short = evenUp(shortSide)
        guard sourceWidth > 0, sourceHeight > 0 else {
            return (evenUp(short * 16 / 9), short)
        }
        if sourceWidth >= sourceHeight {
            let long = Int((Double(short) * Double(sourceWidth) / Double(sourceHeight)).rounded())
            return (evenUp(long), short)
        }
        let long = Int((Double(short) * Double(sourceHeight) / Double(sourceWidth)).rounded())
        return (short, evenUp(long))
    }

    // MARK: Aspect source resolution (2026-08-05 spec §2.3)

    /// The (width, height) pair `frameSize` should take its ratio from, given the chosen source
    /// and what is actually known when the recorder is built.
    ///
    /// A pure function rather than an expression inside `TrainViewModel.start` because the two
    /// fallbacks below are the part that can silently be wrong, and that start path is not
    /// reachable from a test (it needs a Metal device, a project on disk and a real run).
    ///
    /// **Fallback ① — `.viewport` with no usable viewport size falls back to the dataset.** The
    /// recorder is created in `TrainViewModel.start`, and the Train tab's MTKView may not have
    /// been laid out at that moment (the live host is created lazily on the first snapshot swap),
    /// so a zero size is a reachable state rather than a defensive nicety. Falling straight
    /// through to 16:9 there would hand the user a ratio neither source ever offered.
    /// **Fallback ② — a degenerate dataset pair is forwarded unchanged** so it reaches
    /// `frameSize`'s existing 16:9 guard. That path is real: `ColmapCameraReader` throws on camera
    /// models other than PINHOLE/SIMPLE_PINHOLE, so an imported OPENCV/RADIAL project has no
    /// intrinsics at all.
    static func aspectPair(source: TimelapseAspectSource, viewport: CGSize,
                           datasetWidth: Int, datasetHeight: Int) -> (width: Int, height: Int) {
        // A negative is normalised to 0 so "no usable ratio" has one canonical value in the
        // return; frameSize guards on > 0 either way.
        let dataset = (width: max(datasetWidth, 0), height: max(datasetHeight, 0))
        guard source == .viewport else { return dataset }
        // isFinite is checked **before** the conversion, not after: Int(CGFloat.nan) traps.
        // The >= 1 bound (not > 0) is because a sub-point size would round to 0 and re-enter the
        // degenerate case one layer down, where the reason for it is no longer visible.
        guard viewport.width.isFinite, viewport.height.isFinite,
              viewport.width >= 1, viewport.height >= 1 else { return dataset }
        return (Int(viewport.width.rounded()), Int(viewport.height.rounded()))
    }

    // MARK: GIF plan

    /// The GIF reduction plan (spec §3.4): the frame indices to include and the output pixel
    /// size. The long side is clamped to 960 with the aspect ratio preserved, and more than 200
    /// frames are sampled uniformly — the first and the last frame are always included, and the
    /// sampling is strictly increasing (the step is > 1 whenever sampling kicks in), so no frame
    /// is duplicated.
    static func gifPlan(frameCount: Int, width: Int, height: Int)
        -> (indices: [Int], width: Int, height: Int) {
        let n = max(frameCount, 0)
        let indices: [Int]
        if n <= gifMaxFrames {
            indices = Array(0..<n)
        } else {
            indices = (0..<gifMaxFrames).map {
                Int((Double($0) * Double(n - 1) / Double(gifMaxFrames - 1)).rounded())
            }
        }
        let w = max(width, 1), h = max(height, 1)
        let long = max(w, h)
        guard long > gifMaxLongSide else { return (indices, evenUp(w), evenUp(h)) }
        let scale = Double(gifMaxLongSide) / Double(long)
        if w >= h {
            return (indices, gifMaxLongSide, evenUp(Int((Double(h) * scale).rounded())))
        }
        return (indices, evenUp(Int((Double(w) * scale).rounded())), gifMaxLongSide)
    }

    /// A human-readable statement of what gifPlan reduced — nil when nothing was reduced.
    /// Spec §3.4 forbids silent truncation, so the completion log prints this.
    static func gifReductionNote(frameCount: Int, width: Int, height: Int) -> String? {
        let plan = gifPlan(frameCount: frameCount, width: width, height: height)
        var parts: [String] = []
        if plan.indices.count < frameCount {
            parts.append("frames \(frameCount)→\(plan.indices.count)")
        }
        if plan.width != width || plan.height != height {
            parts.append("size \(width)×\(height)→\(plan.width)×\(plan.height)")
        }
        return parts.isEmpty ? nil : "GIF reduced: " + parts.joined(separator: ", ")
    }
}

// MARK: - Forward-compatible decoding (2026-08-04 spec §3.6)

/// Written as an **extension, not in the type body**: declaring an initializer in the body of a
/// struct suppresses the memberwise and default initializers, and `TimelapseSettings()` is used
/// everywhere (defaults, the draft, the Project fallback). Putting `init(from:)` here keeps both.
/// Every field is decodeIfPresent → a partially written object from an older or newer app version
/// still opens, and an unknown **enum** string (format, aspectSource) decodes leniently to that
/// field's default (the same convention as Project.outputAspect — decoding the enum directly would
/// throw on a future case). The lenient target is the property's default, not a named case: the
/// format default has already moved once (mp4 → gif, Task 16).
extension TimelapseSettings {
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init()
        enabled = try c.decodeIfPresent(Bool.self, forKey: .enabled) ?? enabled
        // Snapped on the way in (FIX 7). Without this the guard `normalizedEverySteps` documents
        // has no wiring, and an off-grid value from a hand-edited or foreign manifest reaches the
        // form: `validationErrors` then blocks Start Training while the Stepper's stride of 50
        // walks 75 → 125 → 175 and never lands on a multiple of 50 — a state the UI alone cannot
        // leave.
        everySteps = Self.normalizedEverySteps(
            try c.decodeIfPresent(Int.self, forKey: .everySteps) ?? everySteps)
        let rawFormat = try c.decodeIfPresent(String.self, forKey: .format)
        format = rawFormat.flatMap(TimelapseFormat.init(rawValue:)) ?? format
        fps = try c.decodeIfPresent(Int.self, forKey: .fps) ?? fps
        shortSide = try c.decodeIfPresent(Int.self, forKey: .shortSide) ?? shortSide
        // Decoded as a String and mapped, exactly like `format` above — decoding the enum
        // directly would throw on a value a future version writes, and one unknown string would
        // then fail the whole project manifest instead of one field.
        let rawAspect = try c.decodeIfPresent(String.self, forKey: .aspectSource)
        aspectSource = rawAspect.flatMap(TimelapseAspectSource.init(rawValue:)) ?? aspectSource
        overlay = try c.decodeIfPresent(Bool.self, forKey: .overlay) ?? overlay
    }
}
