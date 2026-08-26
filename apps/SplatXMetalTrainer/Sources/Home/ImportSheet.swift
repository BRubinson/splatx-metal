import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// The page state of the Import sheet. Video goes through a separate options page after the file
/// is chosen.
enum ImportPhase: Equatable {
    case choose             // the four-way list
    case videoOptions(URL)  // the extraction options for the chosen video (with a back path)
    case busy(ImportBusyKind)   // long-running work in progress (dismiss is locked)

    /// Pattern-matching `.busy` regardless of payload. It exists because the dismiss lock and the
    /// animation trigger care only *that* work is running — writing `phase == .busy(.video)` there
    /// would silently stop locking the sheet the day another kind is added.
    var isBusy: Bool { if case .busy = self { true } else { false } }
}

/// What the busy page is waiting on. The three jobs report progress differently and that is not
/// cosmetic: video extraction knows its frame count up front and drives a real fraction, while a
/// capture import is an unzip followed by a file copy whose only honest progress would be bytes —
/// a number that would sit at 0 through the whole central-directory parse and then sprint. It runs
/// indeterminate rather than fake.
///
/// `colmap` is the mixed case, and it says so instead of picking one: reading the model and copying
/// a folder have no per-item step worth counting, and then — only when the cameras carry lens
/// distortion — there is a genuine image count to run against. The page shows an indeterminate bar
/// until the first image is rectified and a determinate one after, so a plain copy never invents a
/// number and a ten-second rectification never pretends to be instant.
enum ImportBusyKind: Equatable { case video, capture, colmap }

/// The Import sheet: a four-way choice of Image Sequence / Video Clip / COLMAP Reconstruction /
/// SplatX Capture.
/// A validation failure = no project is created plus onError(reason). Success = the project is
/// created and the detail view is entered.
/// Video transitions through the pages choose → videoOptions (entering the target frames) → busy
/// (progress); a capture goes choose → busy directly, since a bundle carries its own settings.
struct ImportSheet: View {
    @Bindable var model: AppModel
    var onError: (String) -> Void
    @Environment(\.dismiss) private var dismiss

    @State private var phase: ImportPhase = .choose
    @State private var videoTargetFrames = 200.0   // a Double for the Slider binding (50...600, step 50)
    @State private var busyFraction = 0.0
    /// Rectified images so far / total, for the COLMAP page. `total == 0` means "no rectification is
    /// running", which is how the page knows to stay indeterminate.
    @State private var busyDone = 0
    @State private var busyTotal = 0

    var body: some View {
        Group {
            switch phase {
            case .choose: choosePage
            case .videoOptions(let url): videoOptionsPage(url)
            case .busy(let kind): busyPage(kind)
            }
        }
        .padding(24)
        .frame(width: 480)
        .background(SplatXTheme.panel)
        .interactiveDismissDisabled(phase.isBusy)
        .animation(.easeInOut(duration: 0.15), value: phase)
    }

    // MARK: Pages

    private var choosePage: some View {
        VStack(alignment: .leading, spacing: 16) {
            header(kicker: "Import", title: "Create a Project",
                   subtitle: "Every import becomes a self-contained project folder.")
            VStack(spacing: 0) {
                sourceRow(icon: "photo.stack", title: "Image Sequence",
                          detail: "A folder of 2+ JPEG, PNG, or HEIC photos.") { pickImages() }
                Divider().overlay(SplatXTheme.hairline)
                sourceRow(icon: "film", title: "Video Clip",
                          detail: "The sharpest frames are extracted in the next step.") { pickVideo() }
                Divider().overlay(SplatXTheme.hairline)
                sourceRow(icon: "cube.transparent", title: "COLMAP Reconstruction",
                          detail: "An aligned bundle with sparse/0 and images/.") { pickColmap() }
                Divider().overlay(SplatXTheme.hairline)
                sourceRow(icon: "iphone", title: "SplatX Capture",
                          detail: "A .zip or folder from the SplatX iOS app."
                                + " LiDAR captures skip alignment.") { pickCapture() }
            }
            .background(SplatXTheme.elevated, in: RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(SplatXTheme.hairline))

            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
            }
        }
    }

    private func videoOptionsPage(_ url: URL) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            header(kicker: "Import · Video", title: "Frame Extraction",
                   subtitle: "Choose how many frames to pull from the clip.")

            HStack(spacing: 10) {
                Image(systemName: "film")
                    .font(.system(size: 14, weight: .light))
                    .foregroundStyle(SplatXTheme.accent)
                Text(url.lastPathComponent)
                    .font(.callout)
                    .foregroundStyle(SplatXTheme.textPrimary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 12).padding(.vertical, 9)
            .background(SplatXTheme.elevated, in: RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(SplatXTheme.hairline))

            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Target frames")
                        .font(.callout)
                        .foregroundStyle(SplatXTheme.textSecondary)
                    Spacer()
                    Text("\(Int(videoTargetFrames))")
                        .font(.callout.weight(.semibold)).monospacedDigit()
                        .foregroundStyle(SplatXTheme.accent)
                }
                Slider(value: $videoTargetFrames, in: 50...600, step: 50)
                Text("The sharpest frames are picked evenly across the clip. More frames capture more detail but align and train slower.")
                    .font(.caption)
                    .foregroundStyle(SplatXTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack(spacing: 12) {
                Button("Back") { phase = .choose }
                    .keyboardShortcut(.cancelAction)
                Spacer()
                Button("Extract & Create") { startExtraction(url) }
                    .buttonStyle(.primaryAction)
                    .frame(maxWidth: 220)   // constrains the full-width style to the sheet's trailing CTA width
                    .keyboardShortcut(.defaultAction)
            }
        }
    }

    @ViewBuilder private func busyPage(_ kind: ImportBusyKind) -> some View {
        switch kind {
        case .video:
            VStack(alignment: .leading, spacing: 16) {
                header(kicker: "Import · Video", title: "Extracting Frames",
                       subtitle: "Scoring sharpness and saving the best frames.")
                VStack(alignment: .leading, spacing: 8) {
                    ProgressView(value: busyFraction)
                        .progressViewStyle(.linear)
                    Text(busyLabel)
                        .font(.caption).monospacedDigit()
                        .foregroundStyle(SplatXTheme.textSecondary)
                }
                .padding(.vertical, 8)
            }
        case .capture:
            VStack(alignment: .leading, spacing: 16) {
                header(kicker: "Import · Capture", title: "Opening Capture",
                       subtitle: "Expanding the bundle and copying its frames.")
                VStack(alignment: .leading, spacing: 8) {
                    // Value-less = indeterminate: see ImportBusyKind on why a made-up fraction
                    // would be worse than none.
                    ProgressView()
                        .progressViewStyle(.linear)
                    Text("A large capture takes a few seconds.")
                        .font(.caption)
                        .foregroundStyle(SplatXTheme.textSecondary)
                }
                .padding(.vertical, 8)
            }
        case .colmap:
            VStack(alignment: .leading, spacing: 16) {
                header(kicker: "Import · COLMAP", title: "Opening Reconstruction",
                       subtitle: "Checking the cameras and copying the model.")
                VStack(alignment: .leading, spacing: 8) {
                    if busyTotal > 0 {
                        ProgressView(value: Double(busyDone), total: Double(busyTotal))
                            .progressViewStyle(.linear)
                    } else {
                        ProgressView().progressViewStyle(.linear)
                    }
                    Text(Self.colmapBusyLabel(done: busyDone, total: busyTotal))
                        .font(.caption).monospacedDigit()
                        .foregroundStyle(SplatXTheme.textSecondary)
                }
                .padding(.vertical, 8)
            }
        }
    }

    /// The COLMAP busy line. Pure and static so it can be asserted — a SwiftUI body cannot be driven
    /// by a test on this machine (TCC blocks GUI automation), so a string built inline in one is a
    /// string nothing checks.
    ///
    /// Before the first image is rectified — and for the whole of a plain copy, which never rectifies
    /// anything — the text describes the work rather than a count, because there is no count.
    static func colmapBusyLabel(done: Int, total: Int) -> String {
        total > 0
            ? "Removing lens distortion — \(done) of \(total) images"
            : "Reading the reconstruction…"
    }

    /// An extraction fraction < 0.5 is the scoring pass and ≥ 0.5 is the saving pass (the
    /// VideoFrameExtractor two-pass contract).
    private var busyLabel: String {
        let pct = Int(busyFraction * 100)
        return busyFraction < 0.5
            ? "Pass 1 of 2 · scoring frame sharpness — \(pct)%"
            : "Pass 2 of 2 · saving selected frames — \(pct)%"
    }

    // MARK: Components

    private func header(kicker: String, title: String, subtitle: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(kicker.uppercased())
                .font(.system(size: 11, weight: .semibold)).tracking(1.4)
                .foregroundStyle(SplatXTheme.accent)
            Text(title)
                .font(.title2.weight(.semibold))
                .foregroundStyle(SplatXTheme.textPrimary)
            Text(subtitle)
                .font(.callout)
                .foregroundStyle(SplatXTheme.textSecondary)
        }
    }

    private func sourceRow(icon: String, title: String, detail: String,
                           action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 14) {
                Image(systemName: icon)
                    .font(.system(size: 17, weight: .light))
                    .foregroundStyle(SplatXTheme.accent)
                    .frame(width: 36, height: 36)
                    .overlay(RoundedRectangle(cornerRadius: 9).stroke(SplatXTheme.hairline))
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.callout.weight(.semibold))
                        .foregroundStyle(SplatXTheme.textPrimary)
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(SplatXTheme.textSecondary)
                }
                Spacer(minLength: 8)
                Image(systemName: "chevron.right")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(SplatXTheme.textSecondary)
            }
            .padding(.horizontal, 14).padding(.vertical, 12)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    // MARK: Selection → creation

    private func pickImages() {
        guard let url = chooseDirectory(prompt: "Choose Image Folder") else { return }
        do {
            let project = try model.store.create(
                name: url.lastPathComponent, sourceKind: .images, state: .imported) { dir in
                try ImportService.assembleImages(from: url, into: dir)
            }
            finish(project)
        } catch { fail(error) }
    }

    private func pickColmap() {
        guard let url = chooseDirectory(prompt: "Choose COLMAP Folder") else { return }
        startColmapImport(url)
    }

    /// The COLMAP import: assemble off the main actor, then create the project from what was staged.
    ///
    /// It used to assemble inside `ProjectStore.create`, which runs its closure on the main actor.
    /// That was tolerable while the work was a folder copy and is not now: a distorted model has
    /// every image decoded, resampled and re-encoded (~10 s for 110 × 8 MP), and on the main thread
    /// that is a frozen window. The shape is the capture path's, for the same reason and with the
    /// same workspace cleanup.
    private func startColmapImport(_ url: URL) {
        phase = .busy(.colmap)
        busyDone = 0
        busyTotal = 0
        Task {
            let workspace = FileManager.default.temporaryDirectory
                .appendingPathComponent("sx-colmap-\(UUID().uuidString)", isDirectory: true)
            // Removed on both paths. On success `adoptStagedDirectory` has already moved the tree
            // out, so this reclaims an empty shell; on failure it is what stops a half-rectified
            // image set from being left in the temp directory.
            defer { try? FileManager.default.removeItem(at: workspace) }
            do {
                // stageColmap is nonisolated async — it runs off the main actor (its header
                // explains why that is the entire reason it is async).
                let staged = try await ImportService.stageColmap(
                    source: url, workspace: workspace) { done, total in
                    Task { @MainActor in busyDone = done; busyTotal = total }
                }
                let project = try model.store.create(
                    name: url.lastPathComponent, sourceKind: .colmap, state: .aligned) { dir in
                    try ImportService.adoptStagedDirectory(staged.directory, into: dir)
                    return staged.imageCount
                }
                // After finish, which writes the headline — these are its detail lines, and they are
                // the only place the user is told the field of view they lost (spec §3.3).
                finish(project)
                for note in staged.notes { model.console.append("· \(note)") }
            } catch { fail(error) }
        }
    }

    private func pickVideo() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.movie, .mpeg4Movie, .quickTimeMovie]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        phase = .videoOptions(url)
    }

    /// One panel that accepts both a `.zip` and a bundle folder — which is why
    /// `chooseDirectory(prompt:)` cannot be reused: it turns file selection off. Both shapes are
    /// first-class (`ImportService.stageCapture`'s header), so a user who unzipped in Finder first,
    /// or who AirDropped the folder, must not have to re-zip it.
    ///
    /// `allowedContentTypes` filters *files* only — folders stay selectable because
    /// `canChooseDirectories` is what governs them — so the pair is not contradictory. The type
    /// filter is a courtesy anyway: the staging step decides by "is it a directory", so a
    /// mis-filtered file fails with the zip reader's own message rather than being trusted.
    private func pickCapture() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.zip]
        panel.prompt = "Import Capture"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        startCaptureImport(url)
    }

    /// The capture import: expand + assemble off the main actor, then create the project from what
    /// was staged. There is no options page in between — a bundle already carries every decision
    /// the video path asks the user to make.
    private func startCaptureImport(_ url: URL) {
        phase = .busy(.capture)
        Task {
            let workspace = FileManager.default.temporaryDirectory
                .appendingPathComponent("sx-capture-\(UUID().uuidString)", isDirectory: true)
            // Cleaned up on both paths, exactly as the video import does — and on the success path
            // it is doing real work, not tidying. `adoptStagedCapture` moves `workspace/assembled`
            // into the project, but a zip selection also left `workspace/expanded` here: the whole
            // unpacked archive, which the assembly *copied* from (measured: 95.2 MiB for the 90.2
            // MiB reference bundle, 294.5 MiB for the 288.2 MiB one). Without this line every
            // capture import would leak a second copy of itself into the temp directory until the
            // OS got around to reaping it. On failure it is also what stops a rejected archive from
            // staying expanded there.
            defer { try? FileManager.default.removeItem(at: workspace) }
            do {
                // stageCapture is nonisolated async — it runs off the main actor (its header
                // explains why that is the entire reason it is async).
                let staged = try await ImportService.stageCapture(source: url, workspace: workspace)
                let project = try model.store.create(
                    name: Self.captureProjectName(for: url),
                    sourceKind: .capture,
                    state: Self.captureState(for: staged.kind)) { dir in
                    try ImportService.adoptStagedCapture(staged, into: dir)
                }
                // After finish, not before: finish writes the headline ("Imported project …") and
                // this is its detail line. Dismissing the sheet does not close the console.
                finish(project)
                model.console.append(
                    Self.captureConsoleLine(mode: staged.mode, kind: staged.kind))
                // Empty for every bundle the phone writes (PINHOLE → early exit), so this line adds
                // nothing to the normal import and is the only report when it does fire.
                for note in staged.notes { model.console.append("· \(note)") }
            } catch {
                fail(error)
            }
        }
    }

    /// `.posed` opens `.aligned` — the poses came off the phone's own tracking, and re-solving them
    /// does not refine anything, it **replaces measured geometry with an SfM guess** (the bundle
    /// README is explicit). `.unposed` is a photo/video capture with no model at all, so it opens
    /// `.imported` like any other folder of frames and has to be aligned before it can train.
    ///
    /// The kind it switches on is structural (`sparse/0` present and complete), never the
    /// manifest's `mode` string — see `CaptureBundle`'s header.
    static func captureState(for kind: CaptureBundleKind) -> ProjectState {
        switch kind {
        case .posed: .aligned
        case .unposed: .imported
        }
    }

    /// The project name for a capture selection: the user's own file or folder name, minus a `.zip`
    /// suffix.
    ///
    /// Taken from the **selection** rather than from the resolved bundle root, which is the tempting
    /// alternative and is wrong twice over: a zip whose entries sit at the top level would resolve
    /// to the staging directory and name the project "expanded", and even when the wrapper is
    /// present the user picked a file with a name and expects to find a project by that name.
    /// `ProjectStore.create` de-duplicates and sanitizes from here, so re-importing the same bundle
    /// gives `…-2` rather than a collision.
    static func captureProjectName(for selection: URL) -> String {
        selection.pathExtension.lowercased() == "zip"
            ? selection.deletingPathExtension().lastPathComponent
            : selection.lastPathComponent
    }

    /// The console line that follows a capture import. This is the **only** place the manifest's
    /// `mode` is surfaced, and deliberately so: it is a caption whose vocabulary the producer has
    /// reserved the right to change (`CaptureBundle`'s header), so it is printed, never compared.
    /// A bundle with no readable manifest loses the word and nothing else.
    ///
    /// The alignment half of the sentence comes from the structural kind, so the log agrees with
    /// what the project actually did even if the two ever disagreed.
    static func captureConsoleLine(mode: String?, kind: CaptureBundleKind) -> String {
        let label = mode.map { "SplatX capture · \($0)" } ?? "SplatX capture"
        let outcome = kind == .posed
            ? "on-device poses in use — alignment not needed"
            : "alignment required"
        return "· \(label) · \(outcome)"
    }

    private func startExtraction(_ url: URL) {
        phase = .busy(.video)
        busyFraction = 0
        let target = Int(videoTargetFrames)
        Task {
            let tmpRoot = FileManager.default.temporaryDirectory
                .appendingPathComponent("sx-video-\(UUID().uuidString)", isDirectory: true)
            // The temporary extraction output is cleaned up on both success and failure (on success
            // images has already been moved, so only an empty shell remains).
            defer { try? FileManager.default.removeItem(at: tmpRoot) }
            do {
                let tmpImages = tmpRoot.appendingPathComponent("images", isDirectory: true)
                // extract is nonisolated async — it runs off the main actor.
                let count = try await VideoFrameExtractor.extract(
                    video: url, targetFrames: target, into: tmpImages) { fraction in
                    Task { @MainActor in busyFraction = fraction }
                }
                guard count >= 2 else { throw ImportError.notEnoughImages(found: count) }
                let project = try model.store.create(
                    name: url.deletingPathExtension().lastPathComponent,
                    sourceKind: .video, state: .imported) { dir in
                    try FileManager.default.moveItem(
                        at: tmpImages, to: dir.appendingPathComponent("images", isDirectory: true))
                    return count
                }
                finish(project)
            } catch {
                fail(error)
            }
        }
    }

    private func chooseDirectory(prompt: String) -> URL? {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = prompt
        return panel.runModal() == .OK ? panel.url : nil
    }

    private func finish(_ project: Project) {
        model.console.append("＋ Imported project \"\(project.name)\" (\(project.imageCount) images)")
        dismiss()
        model.openProject(project)
    }

    /// Raw Errors other than ImportError (AVError and the like) are also propagated through
    /// localizedDescription.
    private func fail(_ error: Error) {
        dismiss()
        onError((error as? LocalizedError)?.errorDescription ?? error.localizedDescription)
    }
}
