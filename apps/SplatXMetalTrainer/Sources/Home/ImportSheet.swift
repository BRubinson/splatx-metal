import SwiftUI
import AppKit
import UniformTypeIdentifiers

/// The page state of the Import sheet. Video goes through a separate options page after the file
/// is chosen.
enum ImportPhase: Equatable {
    case choose             // the three-way list
    case videoOptions(URL)  // the extraction options for the chosen video (with a back path)
    case busy               // frame extraction in progress (dismiss is locked)
}

/// The Import sheet: a three-way choice of Image Sequence / Video Clip / COLMAP Reconstruction.
/// A validation failure = no project is created plus onError(reason). Success = the project is
/// created and the detail view is entered.
/// Video transitions through the pages choose → videoOptions (entering the target frames) → busy
/// (progress).
struct ImportSheet: View {
    @Bindable var model: AppModel
    var onError: (String) -> Void
    @Environment(\.dismiss) private var dismiss

    @State private var phase: ImportPhase = .choose
    @State private var videoTargetFrames = 200.0   // a Double for the Slider binding (50...600, step 50)
    @State private var busyFraction = 0.0

    var body: some View {
        Group {
            switch phase {
            case .choose: choosePage
            case .videoOptions(let url): videoOptionsPage(url)
            case .busy: busyPage
            }
        }
        .padding(24)
        .frame(width: 480)
        .background(SplatXTheme.panel)
        .interactiveDismissDisabled(phase == .busy)
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

    private var busyPage: some View {
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
        do {
            let project = try model.store.create(
                name: url.lastPathComponent, sourceKind: .colmap, state: .aligned) { dir in
                try ImportService.assembleColmap(from: url, into: dir)
            }
            finish(project)
        } catch { fail(error) }
    }

    private func pickVideo() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.movie, .mpeg4Movie, .quickTimeMovie]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        phase = .videoOptions(url)
    }

    private func startExtraction(_ url: URL) {
        phase = .busy
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
