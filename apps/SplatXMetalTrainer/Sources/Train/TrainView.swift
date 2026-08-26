import SwiftUI

struct TrainView: View {
    @Bindable var session: ProjectSession
    @Bindable var model: AppModel

    private var vm: TrainViewModel { session.trainVM }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // The progress bar, the summary, and the chart are owned by the Inspector (app round 2
            // Task 2) — the main area is the live view (Task 5).
            // The run history was unified into the RUNS section of the View tab's Inspector (app
            // round 2 Task 7).
            mainArea
            if let e = vm.errorMessage {
                Label(e, systemImage: "xmark.octagon").foregroundStyle(.red)
            }
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .onExitCommand { if vm.isTraining { vm.cancel() } }
    }

    /// The live viewport, wrapped in a `GeometryReader` that measures it (2026-08-05 spec §2.4).
    /// Nothing else in the app knew that size: `drawableSize` is consumed inline in
    /// SplatRendererHost and never kept, and `ViewportRegistry` holds no bounds — see
    /// `TrainViewModel.viewportSize` for why the timelapse needs it.
    ///
    /// The measurement has two consumers, and they are deliberately the **same** one: the view
    /// model (what the recorder's frame ratio is derived from at Start) and the letterbox mask
    /// (what that ratio is drawn against now). A second reader nested inside the live branch would
    /// measure a subtly different rect and the preview would stop being a preview.
    ///
    /// It wraps the **whole** conditional, placeholder branch included. The value is read exactly
    /// once, when Start Training is pressed — and at that moment there is no live host yet (it is
    /// created lazily on the first snapshot swap), so the placeholder is what is on screen. A reader
    /// inside the live branch alone would report `.zero` at the only moment that matters, and every
    /// timelapse would silently take the dataset's ratio through fallback ①.
    ///
    /// The size is written from `.onAppear`/`.onChange(of:)` and **never from the body**: a state
    /// write during body evaluation is a SwiftUI warning and the same class of trap that
    /// ViewerView.swift:356-360 documents ("reading vm.camera in the parent body re-evaluates
    /// everything on every orbit tick"). `noteViewportSize` drops a report that changes nothing, so
    /// the redundant `.onAppear` each live/placeholder switch fires costs one comparison.
    private var mainArea: some View {
        GeometryReader { geo in
            viewport(size: geo.size)
                .onAppear { vm.noteViewportSize(geo.size) }
                .onChange(of: geo.size) { _, size in vm.noteViewportSize(size) }
        }
    }

    /// While training and once the live host is ready it shows the live splatting view; otherwise
    /// an informational placeholder (spec §3).
    /// The host is shared with the session's viewerVM — the same instance as the View tab. Since
    /// app round 5 Task 2 the input monitor is a ViewportRegistry (a weak-reference collection plus
    /// window/bounds routing), so registering multiple viewports is allowed, and when a tab switch
    /// tears down one SplatMetalView, the explicit detach in dismantleNSView removes the
    /// registration immediately (FIX 3 item 6 — the previous deferred automatic cleanup at the next
    /// input event survives as a safety net).
    /// Both branches keep their `maxWidth/maxHeight: .infinity` now that a GeometryReader is above
    /// them: the reader hands down its own size as the proposal, and a child that did not claim it
    /// would sit at its ideal size in the reader's topLeading corner.
    ///
    /// `size` is the reader's measurement, passed in rather than re-measured — see `mainArea`.
    /// ZStack order is load-bearing: the letterbox mask sits directly on top of the splat view and
    /// **below** the LIVE badge and the "shoot from here" button. It is a full-container `Canvas`,
    /// so drawing it last would dim the badge and, worse, swallow the button's clicks — the mask
    /// describes the rendering, not the chrome.
    @ViewBuilder private func viewport(size: CGSize) -> some View {
        if vm.isTraining, let host = session.viewerVM.host {
            ZStack(alignment: .topLeading) {
                SplatMetalView(host: host, camera: Binding(
                    get: { session.viewerVM.camera },
                    set: { session.viewerVM.camera = $0 }))
                if let mask = Self.timelapseMask(active: vm.timelapseActive,
                                                 capture: vm.captureFrameSize,
                                                 in: size) {
                    LetterboxDim(mask: mask)
                        .allowsHitTesting(false)   // orbit drags and zoom must pass straight through
                }
                liveBadge
                    .allowsHitTesting(false)   // no interference with orbit drags or zoom (the ViewportOverlay convention)
                    .padding(12)
                if vm.timelapseActive {
                    timelapseResetButton
                        .padding(12)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ContentUnavailableView(
                "Live view",
                systemImage: "cube.transparent",
                description: Text(placeholderDescription))
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private var placeholderDescription: String {
        if session.project.state == .imported {
            "Run Align first. Training progress appears in the Inspector."
        } else if vm.isTraining {
            "Waiting for the first live update…"   // before the first snapshot (50 iter) swap
        } else {
            "Start training to see live reconstruction."
        }
    }

    /// The small top-left "LIVE · iter N" label — the same material and typography as
    /// ViewportOverlay.assetLabel.
    /// The iter source is not the progress callback but **the snapshot that reached the screen**
    /// (published at the streamer's swap completion, app round 3 spec §3) — so the badge matches
    /// the model actually being seen.
    private var liveBadge: some View {
        HStack(spacing: 6) {
            Circle().fill(.red).frame(width: 7, height: 7)
            Text("LIVE · iter \(vm.liveIter)")
                .font(.system(size: 11, weight: .semibold)).monospacedDigit()
                .foregroundStyle(SplatXTheme.textPrimary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 6))
    }

    /// The timelapse "shoot from here" button (2026-08-04 spec §3.3).
    /// It **must not** live inside liveBadge: that badge is `.allowsHitTesting(false)`, so a
    /// button placed in it would never receive a click. It follows ViewportOverlay.frameButton
    /// instead — a separate child, `.buttonStyle(.plain)`, 28×28, ultraThinMaterial, and a help
    /// string.
    private var timelapseResetButton: some View {
        TimelapseResetButton { vm.resetTimelapseAnchor(session: session, model: model) }
    }

    /// The timelapse framing mask for the live viewport, or nil when nothing should be drawn
    /// (2026-08-05 spec §2.5). A `static func` rather than an expression in the body, because the
    /// interesting part is the *judgement* of when to show nothing at all — the body only unwraps
    /// the result, and a SwiftUI view body is not reachable from a test. Tests:
    /// TimelapseFramingMaskTests.
    ///
    /// **A visible mask means "what is being recorded is narrower than what you are looking at."**
    /// The diagnosis behind this whole round was that the surprise came less from the ratio itself
    /// than from there being no preview of it. With `TimelapseAspectSource.viewport` the two
    /// ratios agree the moment recording starts, so nothing is drawn; a mask that appears later is
    /// the window having been made **wider** mid-run, which moves the viewport but never the
    /// recorder's immutable `Config` (spec §2.6). That is the intended signal, not a glitch.
    ///
    /// # Height-locked, not aspect-fit (timelapse-viewport-aspect FIX 1)
    ///
    /// This deliberately does **not** call `OverlayPlacement.maskRect`, and the reason is a
    /// difference in meaning between the two call sites rather than a difference in taste.
    ///
    /// The capture and the viewport share `fovY`. The viewport renders through
    /// `camera.projectionMatrix(aspect:)` → `perspective(fovY: overridePose?.fov ?? verticalFOV,
    /// …)`, and `TimelapseRecorder` renders through `perspective(fovY: pose.fov, …)` where `pose`
    /// is `CameraPose.effective(of:)` — literally that same value. And
    /// `CameraController.perspective` puts the aspect on the **x** term only (`y = 1/tan(fovY/2)`,
    /// `x = y/aspect`). So the two images always cover the **same vertical range** and differ only
    /// in how much is included left and right. Nothing is ever cropped top or bottom.
    ///
    /// `maskRect` is aspect-fit — the largest rect of the target ratio *inside* the container — so
    /// on a capture that is wider than the viewport it returns top and bottom bars. In Cinema that
    /// is correct: there the mask previews an independently rendered output that will be scaled
    /// down to fit the editor, so both axes really can be inset. Here it would dim two strips that
    /// **do** end up in the recorded frame — the exact opposite of the truth. That is the defect
    /// this FIX removes, and it is why `maskRect` is left untouched instead of being generalised:
    /// the arithmetic is not wrong, it is answering another question.
    ///
    /// So the mask is locked to the full container height and only its width is derived:
    ///
    ///     maskWidth = container.height × captureAspect
    ///
    /// - `maskWidth` < container width → real pillarbox bars; that much of what you see is being
    ///   framed out.
    /// - otherwise → **nil**. A capture that is equal to or wider than the viewport crops nothing;
    ///   it merely includes *more*, and a mask that works by dimming cannot draw "more". Drawing
    ///   nothing is the honest answer, drawing something is a lie.
    ///
    /// **This depends on the shared `fovY`.** The day the capture takes its own fov (a "record at
    /// 50mm" option, say) the vertical ranges stop matching and this arithmetic is wrong again —
    /// in that world the mask has two axes and neither of these two functions computes it.
    ///
    /// # Threshold
    ///
    /// The near-match threshold is derived from the quantisation, not a fixed point count: the
    /// residual bar in `.viewport` mode is a rounding artefact whose size in points scales with
    /// the container, so a fixed 1pt threshold misfires — at shortSide 240 the long side is only
    /// ~350px, and that same relative residual is three times the bar it is at 1080. Two rounding
    /// stages sit between the viewport and the capture ratio and the tolerance is their sum,
    /// expressed as a relative drift and then converted to points of bar:
    ///  ① `aspectPair` rounds the viewport's pt size to whole numbers (`Int(_.rounded())`) before
    ///     `frameSize` ever sees it — up to 0.5px on each side;
    ///  ② `frameSize` derives the long side from the (exactly even) short side with `rounded()`
    ///     and then `evenUp`s it, so the long side lands in [exact − 0.5, exact + 1.5] and 1.5 is
    ///     attainable exactly. That bound, over the long side, is the relative aspect error.
    /// Multiplying by the container width converts it to the bar it can produce in either
    /// orientation (for a landscape capture the worst case is 3× smaller, so the bound is loose
    /// there and tight for portrait). A genuinely different source ratio — a 3:2 dataset in a 16:9
    /// viewport — produces bars of tens of points, so there is nothing to trade off.
    ///
    /// # Two things it deliberately does not do
    ///
    /// - It does not read `draft.timelapse.aspectSource`. `captureFrameSize` comes from the
    ///   recorder's config, the only value that cannot disagree with the frames on disk — the
    ///   chosen source is not even a reliable predictor of it (`aspectPair` has two fallbacks).
    /// - It does not just hand a full-container rect to `LetterboxDim` on a match. That view also
    ///   strokes the mask border, so an exactly-matching mask would leave a 1pt white outline
    ///   framing the whole viewport forever — permanent decoration in the very mode whose promise
    ///   is that nothing shows.
    ///
    /// # Known limitation
    ///
    /// The mask is drawn against the **current viewport pose**, not the pose the capture uses
    /// (`TimelapseRecorder.pose`, fixed at Start or at "shoot from here"). Orbit the live view and
    /// the bars do not move. What the bars claim — *how much* is cut off each side — stays true
    /// regardless, because it follows only from the two ratios and the shared `fovY`; what they
    /// cannot show is *which* content the recorded frame contains once the two poses diverge.
    static func timelapseMask(active: Bool, capture: CGSize?, in container: CGSize) -> CGRect? {
        guard active, let capture, capture.width > 0, capture.height > 0,
              container.width > 0, container.height > 0 else { return nil }
        let maskWidth = container.height * (capture.width / capture.height)
        let drift = 1.5 / max(capture.width, capture.height)          // ② frameSize rounding
            + 0.5 / container.width + 0.5 / container.height          // ① aspectPair pt → Int
        let tolerance = max(1, container.width * drift)
        guard maskWidth < container.width - tolerance else { return nil }
        return CGRect(x: (container.width - maskWidth) / 2, y: 0,
                      width: maskWidth, height: container.height)
    }
}

/// The "shoot from here" control. A bare 28×28 viewfinder glyph does not say what it does, and the
/// system `.help` tooltip arrives after a delay and is easy to miss — so the label is spelled out
/// inline on hover and the icon-only form is just the resting state.
///
/// The inline label and `.help` string **used to carry the same sentence — that is no longer true**
/// (export-cinema-gif Task 8, user request: the inline label read too long). The discard warning was
/// dropped from the inline copy but kept in `.help`:
///  ⑴ the user asked for the inline text shortened,
///  ⑵ the label is `.fixedSize()` + `.lineLimit(1)`, so it pushes sideways with no wrap, and the
///     button is anchored `topTrailing` — it grows **leftward**, so a long label collides with the
///     `topLeading` LIVE badge on a narrow window; a shorter string shrinks that collision window
///     directly,
///  ⑶ the warning still reaches VoiceOver through `.help`, which is the accessibility read path for
///     this control.
///
/// A separate `View` rather than a computed property in TrainView, so `isHovering` dies with the
/// button. The button is removed the moment `timelapseActive` goes false (training ends), and
/// `onHover(false)` is not guaranteed to be delivered to a view that is destroyed while hovered
/// (the hazard BrandFooter documents) — owning the state here means a stale `true` cannot outlive
/// the run and greet the next one already expanded.
///
/// `.pointerStyle(.link)` (export-cinema-gif Task 8): `NSCursor.push()/pop()` stays forbidden for
/// the reason `ScrubbableNumberField.swift`'s "Cursor feedback" header spells out — this button is
/// the textbook trigger case, since it is destroyed the instant `timelapseActive` goes false
/// (training ends while hovered = pop never arrives = a permanently stuck cursor). `.pointerStyle`
/// is declarative, so there is no stack to leave unbalanced. `.link` was picked over leaving it
/// unset because this app's two existing `.pointerStyle` uses (`BrandFooter`'s web link and
/// `ScrubbableNumberField`'s `.columnResize`) both carry a specific meaning — there is no
/// established "ordinary action button" convention here to reuse. `.link` reads on macOS/web as
/// "this is clickable" rather than "this opens a URL"; nothing in this codebase has ever written
/// down `.link` as URL-exclusive, that was only ever true by coincidence of the two prior call
/// sites. **Unverified**: whether `.pointerStyle` actually takes effect over a SwiftUI overlay
/// stacked on an `MTKView` has no precedent in this codebase — the three existing uses are all
/// plain panels, and `PathGizmoOverlay.swift`'s "MTKView cursor rects / local event monitor" warning
/// lives at the same layer. Being declarative rules out the *stuck* failure mode, but not a *silent
/// no-op* — this needs a manual check.
private struct TimelapseResetButton: View {
    let action: () -> Void
    @State private var isHovering = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: "camera.viewfinder")
                    .font(.system(size: 13, weight: .medium))
                if isHovering {
                    // Shortened per user request (export-cinema-gif Task 8) — the discard warning
                    // moved to .help only; see the header comment above for the full rationale.
                    Text("Restart timelapse here")
                        .font(.system(size: 11))
                        .lineLimit(1)
                        .fixedSize()
                        .transition(.opacity)
                }
            }
            .foregroundStyle(SplatXTheme.textPrimary)
            .padding(.horizontal, isHovering ? 10 : 0)
            .frame(minWidth: 28, minHeight: 28)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 6))
        }
        .buttonStyle(.plain)
        .pointerStyle(.link)
        .onHover { isHovering = $0 }
        .animation(.easeOut(duration: 0.12), value: isHovering)
        .help("Restart the timelapse from this view — frames so far are discarded")
    }
}
