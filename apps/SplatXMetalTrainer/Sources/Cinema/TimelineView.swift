// Cinema Task 3 (2026-07-26 spec UX 2 and 3, research B §3.2): the FCP-style timeline strip.
// **A single Canvas** draws the ticks, diamonds, playhead, skimmer, and loop wrap directly — the
// keyframes are not made into individual Views (preventing the explosion of layout recomputation
// during a drag, research B).
// Hit testing, snapping, and the time↔x mapping are the pure functions of TimelineMath (the source
// of truth for tests — TimelineMathTests).
// Playback is driven by TimelineView(.animation) — it is paused while stopped, so it consumes no
// CPU.
import SwiftUI

// MARK: - Pure math (under test — the view consumes only these functions)

enum TimelineMath {
    /// The left and right inset (pt) — it maps the time axis inward so the end diamonds are not
    /// clipped.
    static let inset: CGFloat = 12

    /// Time (seconds) → x (pt). A duration of 0 pins to the left inset.
    static func x(time: Double, duration: Double, width: CGFloat) -> CGFloat {
        guard duration > 0 else { return inset }
        let usable = max(width - 2 * inset, 1)
        let f = min(max(time / duration, 0), 1)
        return inset + CGFloat(f) * usable
    }

    /// x (pt) → time (seconds, clamped to [0, duration]).
    static func time(x: CGFloat, duration: Double, width: CGFloat) -> Double {
        let usable = max(width - 2 * inset, 1)
        let f = Double(min(max((x - inset) / usable, 0), 1))
        return f * duration
    }

    /// The time (seconds) corresponding to 1px — it converts a px threshold into a time threshold
    /// (snapping and hit testing).
    static func timePerPixel(duration: Double, width: CGFloat) -> Double {
        let usable = max(width - 2 * inset, 1)
        return duration / Double(usable)
    }

    /// Diamond hit: the index of the nearest key within radius (pt) of x. nil when there is
    /// none.
    static func hitIndex(x: CGFloat, times: [Double], duration: Double, width: CGFloat,
                         radius: CGFloat = 6) -> Int? {
        var best: (index: Int, distance: CGFloat)?
        for (i, t) in times.enumerated() {
            let d = abs(Self.x(time: t, duration: duration, width: width) - x)
            if d <= radius, best == nil || d < best!.distance { best = (i, d) }
        }
        return best?.index
    }

    /// The snap candidates for a key drag: the tick grid plus the times of the other keys,
    /// excluding the dragged key (FIX 1 — Task 3 review I2). **The playhead is excluded** — during
    /// a key drag the playhead follows the dragged key
    /// (CinemaViewModel.updateKeyframeDrag: playhead = clamped), so including it as a candidate
    /// would become a self-referential quantization that snaps to the previous tick's own position.
    /// The alternative (pinning the drag's starting value) would be "snap back to the original
    /// position", but that value can already be approximately recovered from the tick and other-key
    /// candidates, so it gains nothing and exclusion was chosen.
    static func keyDragSnapCandidates(duration: Double, step: Double, keyframeTimes: [Double],
                                      draggedIndex: Int) -> [Double] {
        var candidates = stride(from: 0.0, through: duration, by: step).map { $0 }
        candidates += keyframeTimes.enumerated().filter { $0.offset != draggedIndex }
            .map(\.element)
        return candidates
    }

    /// Snapping: moves to the nearest candidate within threshold (seconds); the original value when
    /// there is none.
    static func snapped(_ t: Double, candidates: [Double], threshold: Double) -> Double {
        var best: (time: Double, distance: Double)?
        for c in candidates {
            let d = abs(c - t)
            if d <= threshold, best == nil || d < best!.distance { best = (c, d) }
        }
        return best?.time ?? t
    }

    /// Frame stepping: rounds the current time to the frame grid, then ±delta frames, clamped to
    /// [0, duration].
    static func frameStep(from t: Double, delta: Int, fps: Int, duration: Double) -> Double {
        guard fps > 0 else { return t }
        let frame = (t * Double(fps)).rounded() + Double(delta)
        return min(max(frame / Double(fps), 0), max(duration, 0))
    }

    /// The tick interval (seconds): the densest "nice" step whose pixel spacing between labels is at
    /// least minSpacing.
    static func tickStep(duration: Double, width: CGFloat, minSpacing: CGFloat = 48) -> Double {
        guard duration > 0 else { return 1 }
        let steps: [Double] = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120]
        let usable = max(width - 2 * inset, 1)
        for s in steps where CGFloat(s / duration) * usable >= minSpacing { return s }
        return steps.last!
    }
}

// MARK: - The timeline strip (Canvas + gestures)

struct TimelineStrip: View {
    @Bindable var vm: CinemaViewModel

    private enum DragMode: Equatable { case key(Int), scrub }
    @State private var dragMode: DragMode?
    /// The last hover x (footnote 2 ③) — at the moment the context menu opens it back-traces the
    /// diamond at the right-click point (the Canvas is a single view, so the click coordinates are
    /// not passed to the menu). A right-click necessarily happens over the Canvas, so the
    /// immediately preceding .active hover leaves those coordinates behind — which is why it is not
    /// cleared in .ended (opening the menu can end the hover session, and clearing it would race
    /// the menu into being empty).
    @State private var lastHoverX: CGFloat?

    var body: some View {
        VStack(spacing: 0) {
            controls
            Divider().overlay(SplatXTheme.hairline)
            timeline
                .frame(height: 56)
        }
        .background(SplatXTheme.panel)
    }

    // MARK: Transport controls (play, step, capture, readout)

    private var controls: some View {
        HStack(spacing: 12) {
            Button { vm.togglePlayback() } label: {
                Image(systemName: vm.isPlaying ? "pause.fill" : "play.fill")
                    .frame(width: 20)
            }
            .buttonStyle(.plain)
            .help(vm.isPlaying ? "Pause (Space)" : "Play (Space)")
            Button { vm.stepFrame(-1) } label: { Image(systemName: "backward.frame.fill") }
                .buttonStyle(.plain)
                .help("Step back one frame (←)")
            Button { vm.stepFrame(1) } label: { Image(systemName: "forward.frame.fill") }
                .buttonStyle(.plain)
                .help("Step forward one frame (→)")
            Button { vm.captureKeyframe() } label: {
                HStack(spacing: 4) {
                    Image(systemName: "plus.diamond.fill")
                    Text("Key")
                }
            }
            .buttonStyle(.plain)
            .foregroundStyle(SplatXTheme.accent)
            .help("Capture keyframe at playhead (Return) — updates the key under the playhead")
            if vm.selectedPath?.pacing == .manual {
                Text("MANUAL")
                    .font(.system(size: 9, weight: .semibold))
                    .tracking(0.6)
                    .foregroundStyle(SplatXTheme.accent)
                    .padding(.horizontal, 5).padding(.vertical, 2)
                    .background(SplatXTheme.accentDim, in: Capsule())
                    .help("Manual pacing — keyframe times are hand-placed (Inspector: Even pacing to revert)")
            }
            Spacer()
            Text(readout)
                .font(.system(size: 11))
                .monospacedDigit()
                .foregroundStyle(SplatXTheme.textSecondary)
        }
        .font(.system(size: 12))
        .foregroundStyle(SplatXTheme.textPrimary)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    /// "1.50 / 10.00 s · f 45/300 @30" — seconds as the source of truth with the derived frame
    /// count alongside (spec UX 5).
    private var readout: String {
        let d = vm.duration
        let f = Int((vm.playhead * Double(vm.fps)).rounded())
        let total = Int((d * Double(vm.fps)).rounded())
        return String(format: "%.2f / %.2f s · f %d/%d @%d", vm.playhead, d, f, total, vm.fps)
    }

    // MARK: The Canvas timeline

    private var timeline: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 60.0, paused: !vm.isPlaying)) { context in
            GeometryReader { geo in
                Canvas { g, size in
                    draw(&g, size: size)
                }
                .gesture(dragGesture(width: geo.size.width))
                .onContinuousHover { phase in
                    switch phase {
                    case .active(let p):
                        lastHoverX = p.x   // for the context menu hit decision (footnote 2 ③)
                        // Do not apply the skimmer during a drag (scrub or key) — preventing the
                        // preview from being driven twice.
                        if dragMode == nil {
                            vm.skim(to: TimelineMath.time(x: p.x, duration: vm.duration,
                                                          width: geo.size.width))
                        }
                    case .ended:
                        vm.skim(to: nil)   // lastHoverX is kept — see the declaration comment above
                    }
                }
                // Right-clicking a diamond → Delete Keyframe (footnote 2 ③ — the visible UI for the
                // E2E feedback that the Delete key alone was undiscoverable. Right-clicking an
                // empty span shows no menu).
                .contextMenu {
                    if let x = lastHoverX,
                       let i = TimelineMath.hitIndex(x: x, times: vm.keyframeTimes,
                                                     duration: vm.duration,
                                                     width: geo.size.width) {
                        Button("Delete Keyframe", role: .destructive) {
                            vm.deleteKeyframe(at: i)
                        }
                    }
                }
            }
            .onChange(of: context.date) { _, now in
                vm.playbackTick(now: now)
            }
        }
    }

    /// Click/drag: if the start point is a diamond it retimes the key (with snapping, switching to
    /// manual on real movement); otherwise it scrubs (the playhead plus the preview, snapping to
    /// key times).
    private func dragGesture(width: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { v in
                let duration = vm.duration
                if dragMode == nil {
                    if let i = TimelineMath.hitIndex(x: v.startLocation.x, times: vm.keyframeTimes,
                                                     duration: duration, width: width) {
                        dragMode = .key(i)
                        vm.beginKeyframeDrag(i)
                    } else {
                        dragMode = .scrub
                    }
                }
                let t = TimelineMath.time(x: v.location.x, duration: duration, width: width)
                let threshold = TimelineMath.timePerPixel(duration: duration, width: width) * 6
                switch dragMode {
                case .key(let i):
                    // Click protection: retiming starts only after 3pt or more of movement (a
                    // selection alone MUST NOT switch to manual).
                    guard abs(v.translation.width) >= 3 else { return }
                    let step = TimelineMath.tickStep(duration: duration, width: width)
                    let candidates = TimelineMath.keyDragSnapCandidates(
                        duration: duration, step: step, keyframeTimes: vm.keyframeTimes,
                        draggedIndex: i)
                    vm.updateKeyframeDrag(to: TimelineMath.snapped(t, candidates: candidates,
                                                                   threshold: threshold))
                case .scrub:
                    vm.setPlayhead(TimelineMath.snapped(t, candidates: vm.keyframeTimes,
                                                        threshold: threshold),
                                   preview: true)
                case nil:
                    break
                }
            }
            .onEnded { _ in
                if case .key = dragMode { vm.endKeyframeDrag() }
                dragMode = nil
            }
    }

    // MARK: Drawing (the FCP grammar: only three accents — teal for the selection, red for the
    // skimmer, and white for the playhead)

    private func draw(_ g: inout GraphicsContext, size: CGSize) {
        let w = size.width, h = size.height
        let duration = vm.duration
        let midY = h * 0.58
        let times = vm.keyframeTimes

        // The baseline plus the ticks and labels
        var base = Path()
        base.move(to: CGPoint(x: TimelineMath.inset, y: midY))
        base.addLine(to: CGPoint(x: w - TimelineMath.inset, y: midY))
        g.stroke(base, with: .color(SplatXTheme.hairline), lineWidth: 1)
        if duration > 0 {
            let step = TimelineMath.tickStep(duration: duration, width: w)
            var t = 0.0
            while t <= duration + 1e-9 {
                let x = TimelineMath.x(time: t, duration: duration, width: w)
                var tick = Path()
                tick.move(to: CGPoint(x: x, y: 8))
                tick.addLine(to: CGPoint(x: x, y: 14))
                g.stroke(tick, with: .color(SplatXTheme.hairline), lineWidth: 1)
                g.draw(Text(String(format: t == t.rounded() ? "%.0f" : "%.2g", t))
                    .font(.system(size: 9))
                    .foregroundStyle(SplatXTheme.textSecondary),
                    at: CGPoint(x: x, y: 3), anchor: .top)
                t += step
            }
        }

        // The loop wrap (showing the last→first segment's time — spec UX 5): a dashed line from the
        // last key's pass-through time to duration, plus a ghost diamond at the duration position
        // marking the return to the first key.
        if let path = vm.selectedPath, path.loop, times.count >= 2, duration > 0 {
            let xLast = TimelineMath.x(time: times[times.count - 1], duration: duration, width: w)
            let xEnd = TimelineMath.x(time: duration, duration: duration, width: w)
            var wrap = Path()
            wrap.move(to: CGPoint(x: xLast, y: midY))
            wrap.addLine(to: CGPoint(x: xEnd, y: midY))
            g.stroke(wrap, with: .color(SplatXTheme.accent.opacity(0.6)),
                     style: StrokeStyle(lineWidth: 1.5, dash: [3, 3]))
            g.stroke(diamond(at: CGPoint(x: xEnd, y: midY), r: 5),
                     with: .color(SplatXTheme.accent.opacity(0.6)), lineWidth: 1)
            g.draw(Text(Image(systemName: "arrow.counterclockwise"))
                .font(.system(size: 8))
                .foregroundStyle(SplatXTheme.accent.opacity(0.8)),
                at: CGPoint(x: xEnd, y: midY - 12), anchor: .center)
        }

        // The keyframe diamonds — only the selected key is accent (the FCP grammar: restrained use
        // of color)
        for (i, t) in times.enumerated() {
            let x = TimelineMath.x(time: t, duration: duration, width: w)
            let shape = diamond(at: CGPoint(x: x, y: midY), r: 5.5)
            if i == vm.selectedKeyframeIndex {
                g.fill(shape, with: .color(SplatXTheme.accent))
                g.stroke(shape, with: .color(SplatXTheme.accent), lineWidth: 1.5)
            } else {
                g.fill(shape, with: .color(SplatXTheme.textPrimary))
                g.stroke(shape, with: .color(SplatXTheme.ground), lineWidth: 1)
            }
        }

        // The playhead (a white vertical line plus a knob at the top)
        let px = TimelineMath.x(time: vm.playhead, duration: duration, width: w)
        var ph = Path()
        ph.move(to: CGPoint(x: px, y: 2))
        ph.addLine(to: CGPoint(x: px, y: h - 2))
        g.stroke(ph, with: .color(.white), lineWidth: 1.5)
        var knob = Path()
        knob.move(to: CGPoint(x: px - 4, y: 2))
        knob.addLine(to: CGPoint(x: px + 4, y: 2))
        knob.addLine(to: CGPoint(x: px, y: 8))
        knob.closeSubpath()
        g.fill(knob, with: .color(.white))

        // The skimmer (a red hover line — a preview with no commit, the heart of the FCP feel)
        if let s = vm.skimTime {
            let sx = TimelineMath.x(time: s, duration: duration, width: w)
            var line = Path()
            line.move(to: CGPoint(x: sx, y: 2))
            line.addLine(to: CGPoint(x: sx, y: h - 2))
            g.stroke(line, with: .color(SplatXTheme.destructive), lineWidth: 1)
        }
    }

    private func diamond(at c: CGPoint, r: CGFloat) -> Path {
        var p = Path()
        p.move(to: CGPoint(x: c.x, y: c.y - r))
        p.addLine(to: CGPoint(x: c.x + r, y: c.y))
        p.addLine(to: CGPoint(x: c.x, y: c.y + r))
        p.addLine(to: CGPoint(x: c.x - r, y: c.y))
        p.closeSubpath()
        return p
    }
}
