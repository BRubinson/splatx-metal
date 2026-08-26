// Cinema Task 4 (2026-07-26 spec UX 5 "overlay section" + the data model): static image overlay
// editing — image selection (NSOpenPanel) → a copy into the project directory (self-contained) →
// the OverlaySpec inspector (9 anchors + a free offset + size/margin in % + opacity) plus the
// viewport WYSIWYG (showing and drag-placing the overlay over the aspect mask).
//
// Unit contract (handed over from Task 1): every length in OverlaySpec (size, offset, margin) is
// **a percentage of the output's short side**. The placement arithmetic OverlayPlacement.rect is a
// pure function that takes the output W/H as arguments — the Task 5 export compositing calls
// **the same function** with the output pixel dimensions, which guarantees WYSIWYG (the viewport
// calls it with the mask's pt dimensions — since the units are percentages, the two results are
// similar figures).
import SwiftUI
import AppKit
import UniformTypeIdentifiers

// MARK: - Output aspect ratio (the WYSIWYG mask and frustum aspect = the export aspect — one
// shared state. Persisted per project since Task 5: rawValue is the storage format, so do not
// change the strings.)

enum OutputAspect: String, CaseIterable, Identifiable, Codable {
    case wide = "16:9"
    case tall = "9:16"
    case square = "1:1"

    var id: String { rawValue }
    var ratio: CGFloat {
        switch self {
        case .wide: 16.0 / 9.0
        case .tall: 9.0 / 16.0
        case .square: 1
        }
    }
}

// MARK: - Placement arithmetic (pure — OverlayPlacementTests is the source of truth; the
// signature is reused by Task 5)

enum OverlayPlacement {
    /// OverlaySpec → the output pixel rect (origin at top-left, +y downward).
    /// - Size: an aspect-preserving scale such that the image's **long side** becomes
    ///   `the output short side × sizePercent%` (both landscape and portrait images fit inside
    ///   the same square bound — predictable regardless of orientation).
    /// - Margin: applied only on the edges the anchor touches (a centered axis has no margin).
    /// - Offset: added after anchoring, as (+x right, +y down) — all in short-side percentage
    ///   units.
    /// Task 5 compositing calls it with the output pixel dimensions, and the viewport WYSIWYG
    /// calls it with the mask's pt dimensions.
    static func rect(spec: OverlaySpec, imageSize: CGSize,
                     outputWidth: CGFloat, outputHeight: CGFloat) -> CGRect {
        guard outputWidth > 0, outputHeight > 0,
              imageSize.width > 0, imageSize.height > 0 else { return .zero }
        let short = min(outputWidth, outputHeight)
        let maxSide = short * CGFloat(spec.sizePercent) / 100
        let scale = maxSide / max(imageSize.width, imageSize.height)
        let w = imageSize.width * scale
        let h = imageSize.height * scale
        let m = short * CGFloat(spec.margin) / 100

        let x: CGFloat = switch column(of: spec.anchor) {
        case .leading: m
        case .center: (outputWidth - w) / 2
        case .trailing: outputWidth - w - m
        }
        let y: CGFloat = switch row(of: spec.anchor) {
        case .top: m
        case .middle: (outputHeight - h) / 2
        case .bottom: outputHeight - h - m
        }
        return CGRect(x: x + short * CGFloat(spec.offsetX) / 100,
                      y: y + short * CGFloat(spec.offsetY) / 100,
                      width: w, height: h)
    }

    /// The largest centered rect that fits inside the container at the target aspect ratio (w/h)
    /// — the viewport letterbox mask.
    ///
    /// `aspect` is a plain ratio and always has been: Cinema's call site happens to pass
    /// `cinema.outputAspect.ratio`, but nothing in here is tied to the 16:9/9:16/1:1 enum. The
    /// `maskRect(ratio:in:)` overload that 2026-08-05 spec §2.5 and its plan call for was
    /// deliberately not added: Swift would accept it as an argument-label overload, but it would
    /// be the same body reachable two ways with nothing to tell a reader which is authoritative.
    ///
    /// **Cinema is the only call site.** Task 3 reused this for the Train tab's timelapse framing
    /// preview and `timelapse-viewport-aspect` FIX 1 took that back: aspect-fit is the right
    /// arithmetic *here*, where the mask previews an independently rendered output scaled down to
    /// fit the editor, and the wrong one there, where the capture and the viewport share `fovY` so
    /// only the horizontal extent can differ. See `TrainView.timelapseMask` for the full argument.
    /// Do not "unify" the two — the shapes coincide on one branch by accident, not by meaning.
    /// Tests: OverlayPlacementTests plus TimelapseFramingMaskTests, whose OutputAspect cases pin
    /// the enum → ratio → rect chain this call site depends on.
    static func maskRect(aspect: CGFloat, in container: CGSize) -> CGRect {
        guard aspect > 0, container.width > 0, container.height > 0 else { return .zero }
        if container.width / container.height > aspect {
            let w = container.height * aspect
            return CGRect(x: (container.width - w) / 2, y: 0,
                          width: w, height: container.height)
        }
        let h = container.width / aspect
        return CGRect(x: 0, y: (container.height - h) / 2,
                      width: container.width, height: h)
    }

    /// The drag translation (pt) → the offset increment (short-side %). rect is linear in the
    /// offset, so with this conversion the overlay follows the cursor exactly.
    static func offsetDelta(translation: CGSize, shortSide: CGFloat)
        -> (dx: Double, dy: Double) {
        guard shortSide > 0 else { return (0, 0) }
        return (Double(translation.width / shortSide * 100),
                Double(translation.height / shortSide * 100))
    }

    /// Changing the anchor — the offset is reset to 0 (keeping an offset measured from the
    /// previous anchor can fling the overlay off screen under the new anchor — a predictable jump
    /// is better).
    static func anchored(_ spec: OverlaySpec, to anchor: OverlayAnchor) -> OverlaySpec {
        var s = spec
        s.anchor = anchor
        s.offsetX = 0
        s.offsetY = 0
        return s
    }

    // MARK: Anchor decomposition

    enum Column { case leading, center, trailing }
    enum Row { case top, middle, bottom }

    static func column(of anchor: OverlayAnchor) -> Column {
        switch anchor {
        case .topLeading, .leading, .bottomLeading: .leading
        case .top, .center, .bottom: .center
        case .topTrailing, .trailing, .bottomTrailing: .trailing
        }
    }

    static func row(of anchor: OverlayAnchor) -> Row {
        switch anchor {
        case .topLeading, .top, .topTrailing: .top
        case .leading, .center, .trailing: .middle
        case .bottomLeading, .bottom, .bottomTrailing: .bottom
        }
    }
}

// MARK: - Image copy (project self-containment — the spec data model)

enum OverlayStore {
    static let directoryName = "overlay"

    /// Copies the source image to `<project>/overlay/<filename>` and returns the project-relative
    /// path. A file of the same name is replaced (reselecting = update semantics).
    static func importImage(from source: URL, projectDirectory: URL) throws -> String {
        let fm = FileManager.default
        let dir = projectDirectory.appendingPathComponent(directoryName, isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let dst = dir.appendingPathComponent(source.lastPathComponent)
        if fm.fileExists(atPath: dst.path) { try fm.removeItem(at: dst) }
        try fm.copyItem(at: source, to: dst)
        return "\(directoryName)/\(source.lastPathComponent)"
    }

    static func url(for relativePath: String, projectDirectory: URL) -> URL {
        projectDirectory.appendingPathComponent(relativePath)
    }

    /// Cleans up the previous copy when the overlay is removed or replaced — it deletes only
    /// files inside the project's overlay/ (paths outside it may be the user's own files, so they
    /// are untouched). A failure is harmless (an orphaned copy remains), so it is ignored.
    static func removeCopy(relativePath: String, projectDirectory: URL) {
        guard relativePath.hasPrefix("\(directoryName)/") else { return }
        try? FileManager.default.removeItem(
            at: url(for: relativePath, projectDirectory: projectDirectory))
    }
}

// MARK: - Inspector section (OVERLAY — inserted by CinemaInspector. The former FRAMING section
// was absorbed into EXPORT in panel redesign Task 3 — the Aspect picker lives in RenderJob.swift's
// ExportInspectorSection, and the binding (through setOutputAspect, persisted per project) and
// the consumers (the viewport mask CinemaFramingOverlay and the export pixelSize) are unchanged:
// only the placement moved.)

struct OverlayInspectorSection: View {
    @Bindable var vm: CinemaViewModel

    /// The ⓘ prose of the OVERLAY section (panel redesign Task 2 — caption diet, spec §4 rule 4):
    /// the relocation of the two previously permanent on-screen captions (the export compositing
    /// explanation and the viewport drag hint).
    static let overlayInfo =
        "A static image (logo/watermark) composited on every export.\n\n"
        + "Drag the image in the viewport to place it freely."

    /// The Add Image disabled reason (pure — the source of truth for tests). Previously it only
    /// had .disabled and stayed silent (one of the four disabled presentations in spec §1 ④ —
    /// unified by rule 5).
    static func addImageDisabledReason(hasProjectDirectory: Bool) -> String? {
        hasProjectDirectory ? nil : "Open a project first — the image is copied into it."
    }

    /// The "Add SplatX watermark" disabled reason (pure — the source of truth for tests). It
    /// **reuses the same gate** as Add Image for the project condition and adds only the missing
    /// bundle image. A missing bundle is treated as **disabling up front** rather than as a click
    /// failure (§4 rule 5: dim the row plus a one-line reason) — better than a failure you can
    /// only discover by pressing.
    /// ⚠ This return value is **only for the disable and dim decision** — the caption drawn on
    /// screen is authoritative in `addWatermarkCaption` (app finishing FIX 1 ⑤: drawing the
    /// missing-project reason straight from this return value would render the same sentence
    /// twice in a row, right below the Add Image caption).
    static func addWatermarkDisabledReason(hasProjectDirectory: Bool,
                                           hasBundleImage: Bool) -> String? {
        if let reason = addImageDisabledReason(hasProjectDirectory: hasProjectDirectory) {
            return reason
        }
        return hasBundleImage ? nil : "The bundled watermark image is missing from this build."
    }

    /// The caption the watermark row actually draws (pure — the source of truth for tests, app
    /// finishing FIX 1 ⑤). It implements the original intent of "do not say the same reason in two
    /// sentences": the Add Image row right above already prints the missing-project reason as a
    /// caption, so this row **only dims and omits the caption** (when the missing project applies
    /// it returns nil regardless of whether the bundle is missing — the row above's reason
    /// explains this row's dimming too). The missing-bundle reason is unique to this row, so it is
    /// printed as-is.
    static func addWatermarkCaption(hasProjectDirectory: Bool,
                                    hasBundleImage: Bool) -> String? {
        guard addImageDisabledReason(hasProjectDirectory: hasProjectDirectory) == nil else {
            return nil
        }
        return addWatermarkDisabledReason(hasProjectDirectory: hasProjectDirectory,
                                          hasBundleImage: hasBundleImage)
    }

    var body: some View {
        PanelSection(title: "OVERLAY", info: Self.overlayInfo) {
            if let spec = vm.overlay {
                editor(spec)
            } else {
                emptyState
            }
        }
    }

    /// The empty state — two actions, **both using the rule A full-width chrome**
    /// (`sectionActionChrome` — the single source of truth in Theme.swift. App finishing round 6
    /// Task 1 reversed the previous "small secondary text button" idiom: the user requested "the
    /// same style as the Add Image button, just a slightly different color". The history is
    /// authoritative in campath spec footnote 8 — per the retroactive-correction rule, the body of
    /// the previous decision (footnote 6 (a)) is left unedited).
    ///
    /// **Color variation**: only the watermark button uses `tint: SplatXTheme.accent` — the
    /// geometry, background, and border are identical to Add Image, and only the label (icon +
    /// text) color differs. accent (brand teal) is consistent with the meaning "restore the brand
    /// watermark", and even when disabled (row opacity 0.5) a brightness difference against the
    /// elevated background remains, so legibility holds. No new color — it is within the nine
    /// existing tokens.
    ///
    /// **Why not a pulldown like `New Path ▾`**: the rule of that precedent is "fold commands
    /// whose result kind is the same", and formally this qualifies too, but New Path had nine
    /// items so folding paid off, whereas here there are two. The moment they are folded, the
    /// default watermark hides behind the chevron — and **being hidden and unfindable is exactly
    /// the defect being fixed right now**, so that would reproduce the same defect at another
    /// level.
    /// Omitting the ellipsis is still intentional — there is no additional input dialog (HIG).
    /// The contrast with the ellipsis of `Add Image…` itself signals "this one does not open a
    /// panel".
    ///
    /// **It is not exposed in the editor branch (when an overlay already exists)** — the user's
    /// requirement is precisely the empty state, and four controls (icon, name, Replace, trash)
    /// already make that row crowded. As a result `installWatermark` is guaranteed
    /// `vm.overlay == nil`, so cleaning up a previous copy (the removeCopy in `pickImage`) is
    /// structurally unnecessary.
    private var emptyState: some View {
        // Rule 6: "control↔attached caption" (2) < "control↔control" (8).
        VStack(alignment: .leading, spacing: 8) {
            let reason = Self.addImageDisabledReason(
                hasProjectDirectory: vm.projectDirectory != nil)
            VStack(alignment: .leading, spacing: 2) {
                Button {
                    Self.pickImage(vm: vm)
                } label: {
                    Label("Add Image…", systemImage: "photo.badge.plus")
                        .sectionActionChrome()   // the chrome is authoritative in Theme.swift (absorbed in Task 1)
                }
                .buttonStyle(.plain)
                .disabled(reason != nil)
                .opacity(reason != nil ? 0.5 : 1)
                if let reason {
                    PanelCaption(reason)
                }
            }

            let wmReason = Self.addWatermarkDisabledReason(
                hasProjectDirectory: vm.projectDirectory != nil,
                hasBundleImage: WatermarkSeeder.bundleImageURL != nil)
            // The caption is authoritative separately from the disable decision (FIX 1 ⑤) — the
            // missing-project reason was already printed by the row above, so it is omitted here
            // (the dimming is still handled by wmReason).
            let wmCaption = Self.addWatermarkCaption(
                hasProjectDirectory: vm.projectDirectory != nil,
                hasBundleImage: WatermarkSeeder.bundleImageURL != nil)
            VStack(alignment: .leading, spacing: 2) {
                Button {
                    Self.installWatermark(vm: vm)
                } label: {
                    // The icon rule is the same as Add Image too (Label + systemImage) — seal is
                    // the "brand seal" metaphor. The absence of an ellipsis stands (the signal
                    // that no panel opens).
                    Label("Add SplatX watermark", systemImage: "seal")
                        .sectionActionChrome(tint: SplatXTheme.accent)
                }
                .buttonStyle(.plain)
                .disabled(wmReason != nil)
                .opacity(wmReason != nil ? 0.5 : 1)
                .help("Restore the default SplatX Metal brand watermark"
                      + " (bottom-right, 28%).")
                if let wmCaption {
                    PanelCaption(wmCaption)
                }
            }
        }
    }

    @ViewBuilder private func editor(_ spec: OverlaySpec) -> some View {
        HStack {
            Image(systemName: "photo")
                .foregroundStyle(SplatXTheme.textSecondary)
            Text((spec.imagePath as NSString).lastPathComponent)
                .font(.caption).lineLimit(1).truncationMode(.middle)
                .foregroundStyle(SplatXTheme.textPrimary)
            Spacer(minLength: 4)
            Button("Replace…") { Self.pickImage(vm: vm) }
                .font(.caption)
            Button {
                removeOverlay(spec)
            } label: {
                Image(systemName: "trash")
            }
            .help("Remove overlay")
        }

        anchorGrid(spec)

        slider("Size", value: overlayBinding(\.sizePercent, default: 20),
               in: 2...60, unit: "%")
        slider("Margin", value: overlayBinding(\.margin, default: 4),
               in: 0...20, unit: "%")
        slider("Opacity", value: overlayBinding(\.opacity, default: 1),
               in: 0...1, unit: nil)

        if spec.offsetX != 0 || spec.offsetY != 0 {
            HStack {
                Text(String(format: "offset %.1f, %.1f %%", spec.offsetX, spec.offsetY))
                    .font(.caption2).monospacedDigit()
                    .foregroundStyle(SplatXTheme.textSecondary)
                Spacer()
                Button("Reset") {
                    var s = spec
                    s.offsetX = 0; s.offsetY = 0
                    vm.setOverlay(s)
                }
                .font(.caption2)
            }
        }
        // (The previous permanent "Drag the image…" caption moved into the section header ⓘ
        // (overlayInfo) — rule 4)
    }

    /// The 3×3 anchor grid — only the selected anchor is accent. Changing it resets the offset
    /// (OverlayPlacement.anchored).
    private func anchorGrid(_ spec: OverlaySpec) -> some View {
        let rows: [[OverlayAnchor]] = [
            [.topLeading, .top, .topTrailing],
            [.leading, .center, .trailing],
            [.bottomLeading, .bottom, .bottomTrailing],
        ]
        return HStack {
            FieldLabel("Anchor")
            Spacer()
            VStack(spacing: 3) {
                ForEach(rows, id: \.self) { row in
                    HStack(spacing: 3) {
                        ForEach(row, id: \.self) { anchor in
                            let selected = anchor == spec.anchor
                            Button {
                                vm.setOverlay(OverlayPlacement.anchored(spec, to: anchor))
                            } label: {
                                RoundedRectangle(cornerRadius: 3)
                                    .fill(selected ? SplatXTheme.accent : SplatXTheme.elevated)
                                    .frame(width: 16, height: 12)
                                    .overlay(RoundedRectangle(cornerRadius: 3)
                                        .stroke(selected ? SplatXTheme.accent
                                                         : SplatXTheme.hairline))
                            }
                            .buttonStyle(.plain)
                            .help(anchor.rawValue)
                        }
                    }
                }
            }
        }
    }

    private func slider(_ label: String, value: Binding<Double>,
                        in range: ClosedRange<Double>, unit: String?) -> some View {
        HStack {
            FieldLabel(label)   // the origin of the 52pt label column — now consumed after being generalized (Theme.swift)
            Slider(value: value, in: range,
                   onEditingChanged: { editing in if !editing { vm.persistOverlay() } })
            Text(unit.map { String(format: "%.0f\($0)", value.wrappedValue) }
                 ?? String(format: "%.2f", value.wrappedValue))
                .font(.caption2).monospacedDigit()
                .foregroundStyle(SplatXTheme.textSecondary)
                .frame(width: 34, alignment: .trailing)
        }
    }

    /// The shared slider binding — the save is deferred during the drag (setOverlay
    /// persistNow: false) and persistOverlay() in onEditingChanged(false) saves once (the same
    /// pattern as the smoothness slider).
    private func overlayBinding(_ keyPath: WritableKeyPath<OverlaySpec, Double>,
                                default def: Double) -> Binding<Double> {
        Binding(
            get: { vm.overlay?[keyPath: keyPath] ?? def },
            set: { newValue in
                guard var s = vm.overlay else { return }
                s[keyPath: keyPath] = newValue
                vm.setOverlay(s, persistNow: false)
            })
    }

    private func removeOverlay(_ spec: OverlaySpec) {
        if let dir = vm.projectDirectory {
            OverlayStore.removeCopy(relativePath: spec.imagePath, projectDirectory: dir)
        }
        vm.setOverlay(nil)
    }

    /// NSOpenPanel (png/jpeg/heic) → copy into the project → update the spec (keeping the
    /// existing placement settings).
    /// On replacement the previous copy is cleaned up (if the names match, importImage has
    /// already overwritten it).
    @MainActor static func pickImage(vm: CinemaViewModel) {
        guard let dir = vm.projectDirectory else { return }
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, .heic]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let src = panel.url else { return }
        do {
            let rel = try OverlayStore.importImage(from: src, projectDirectory: dir)
            if let old = vm.overlay, old.imagePath != rel {
                OverlayStore.removeCopy(relativePath: old.imagePath, projectDirectory: dir)
            }
            var spec = vm.overlay ?? OverlaySpec(imagePath: rel)
            spec.imagePath = rel
            vm.setOverlay(spec)
        } catch {
            vm.viewer?.errorMessage = "Overlay import failed: \(error.localizedDescription)"
        }
    }

    /// Reinstalls the default brand watermark (app finishing 2026-07-29) — it goes through **the
    /// same copy mechanism and the same default placement** as the seed
    /// (WatermarkSeeder.install: same-name replacement makes it idempotent, so repeated clicks
    /// are safe).
    /// The seed flag is untouched — the rationale is authoritative in the
    /// `WatermarkSeeder.install` comment.
    ///
    /// **It does not stay silent on failure.** The rationale for the automatic seed's silence is
    /// "the user did not ask for it", and staying silent on an explicit click means "I clicked
    /// and nothing happened" — the same class of defect being fixed right now.
    /// The notification surface reuses `viewer.errorMessage`, which this section already uses
    /// (AppModel.console is out of reach from the inspector).
    @MainActor static func installWatermark(vm: CinemaViewModel) {
        guard let dir = vm.projectDirectory else { return }
        guard let spec = WatermarkSeeder.install(into: dir,
                                                 source: WatermarkSeeder.bundleImageURL) else {
            vm.viewer?.errorMessage =
                "Could not add the SplatX watermark — the bundled image could not be"
                + " copied into the project."
            return
        }
        vm.setOverlay(spec)
    }
}

// MARK: - Viewport WYSIWYG (the aspect mask plus showing and dragging the overlay — ViewerView
// layers it on)

struct CinemaFramingOverlay: View {
    @Bindable var cinema: CinemaViewModel
    @State private var image: NSImage?
    /// The offset (%) at the moment the drag started. The increment is always measured from the
    /// starting value — no accumulation of intermediate rounding.
    @State private var dragBase: (x: Double, y: Double)?

    var body: some View {
        GeometryReader { geo in
            let mask = OverlayPlacement.maskRect(aspect: cinema.outputAspect.ratio,
                                                in: geo.size)
            ZStack(alignment: .topLeading) {
                LetterboxDim(mask: mask)
                    .allowsHitTesting(false)
                if let spec = cinema.overlay, let image, mask.width > 0 {
                    let r = OverlayPlacement.rect(
                        spec: spec, imageSize: Self.pixelSize(of: image),
                        outputWidth: mask.width, outputHeight: mask.height)
                        .offsetBy(dx: mask.minX, dy: mask.minY)
                    Image(nsImage: image)
                        .resizable()
                        .interpolation(.high)
                        .opacity(spec.opacity)
                        .frame(width: max(r.width, 1), height: max(r.height, 1))
                        // A drag starting in this area is overlay placement, not a camera orbit —
                        // registered with the local event monitor as an exclusion area
                        // (CameraDragExclusion).
                        .background(CameraDragExclusion())
                        .position(x: r.midX, y: r.midY)
                        .gesture(dragGesture(shortSide: min(mask.width, mask.height)))
                }
            }
        }
        .task(id: cinema.overlay?.imagePath) { loadImage() }
    }

    private func dragGesture(shortSide: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 1)
            .onChanged { v in
                guard var spec = cinema.overlay else { return }
                let base = dragBase ?? (spec.offsetX, spec.offsetY)
                dragBase = base
                let d = OverlayPlacement.offsetDelta(translation: v.translation,
                                                     shortSide: shortSide)
                spec.offsetX = base.x + d.dx
                spec.offsetY = base.y + d.dy
                cinema.setOverlay(spec, persistNow: false)   // live; saved once at the end of the drag
            }
            .onEnded { _ in
                dragBase = nil
                cinema.persistOverlay()
            }
    }

    /// The pixel dimensions (preferring the representation — avoiding the pt/px mismatch of
    /// retina-tagged images).
    /// The rect arithmetic consumes only the aspect ratio, so the result is identical either way,
    /// but pixels are authoritative.
    private static func pixelSize(of image: NSImage) -> CGSize {
        if let rep = image.representations.first, rep.pixelsWide > 0, rep.pixelsHigh > 0 {
            return CGSize(width: rep.pixelsWide, height: rep.pixelsHigh)
        }
        return image.size
    }
}

/// The letterbox dimming outside the mask plus the mask border. It punches the hole with an
/// even-odd fill.
///
/// Internal rather than `private` since 2026-08-05: TrainView layers the same dimming over the
/// live viewport to show the timelapse capture framing (spec §2.5). It stays in this file instead
/// of moving to a shared one — Cinema is where the idiom was designed, the second call site is a
/// reuse of it rather than a new abstraction, and a "Shared/" home would hide that lineage.
/// Note for the Train call site: this always strokes the mask border, so handing it a mask equal
/// to the container is not a no-op — see `TrainView.timelapseMask`, which returns nil there.
/// It is the *dimming* that is shared, not the geometry: since `timelapse-viewport-aspect` FIX 1
/// the Train mask is computed height-locked in `TrainView` rather than by `OverlayPlacement`.
struct LetterboxDim: View {
    let mask: CGRect

    var body: some View {
        Canvas { g, size in
            guard mask.width > 0 else { return }
            var p = Path(CGRect(origin: .zero, size: size))
            p.addRect(mask)
            g.fill(p, with: .color(.black.opacity(0.55)), style: FillStyle(eoFill: true))
            g.stroke(Path(mask), with: .color(.white.opacity(0.25)), lineWidth: 1)
        }
    }
}

// MARK: - CinemaFramingOverlay image loading

extension CinemaFramingOverlay {
    /// Reloads when imagePath changes — .task(id:) calls it when the view attaches (including the
    /// initial attach).
    func loadImage() {
        guard let url = cinema.overlayImageURL else {
            image = nil
            return
        }
        image = NSImage(contentsOf: url)
    }
}
