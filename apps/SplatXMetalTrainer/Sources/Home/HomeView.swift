import SwiftUI
import AppKit

/// Home: the project card grid. The first cell is always the Import card (app round 4 spec
/// item 7), so a single surface is preserved even with zero projects — the empty-state-only view
/// and the toolbar Import button were removed (the ⌘O command and menu item survive in
/// AppCommands).
struct HomeView: View {
    @Bindable var model: AppModel
    @State private var renaming: Project?
    @State private var renameText = ""
    @State private var deleting: Project?
    @State private var importError: String?

    var body: some View {
        grid
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        // Home branding (app round 5 spec item 7): the same brand footer (with links) even in the
        // Home window, which has no Inspector.
        // It is a safeAreaInset, so the grid's scrolling content is not hidden behind the footer.
        .safeAreaInset(edge: .bottom, spacing: 0) { BrandFooter() }
        .background(SplatXTheme.ground)
        .navigationTitle("Projects")
        .task {
            model.store.scan()
            for w in model.store.scanWarnings { model.console.append("⚠ \(w)") }
        }
        .sheet(isPresented: $model.showImportSheet) {
            ImportSheet(model: model) { importError = $0 }
        }
        .alert("Import failed", isPresented: Binding(
            get: { importError != nil }, set: { if !$0 { importError = nil } })) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(importError ?? "")
        }
        .alert("Rename Project", isPresented: Binding(
            get: { renaming != nil }, set: { if !$0 { renaming = nil } })) {
            TextField("Name", text: $renameText)
            Button("Rename") {
                // The same name is a no-op — ProjectStore.rename appends a "-2" suffix on a rename-to-self.
                if let p = renaming, renameText != p.name {
                    do { _ = try model.store.rename(p, to: renameText) }
                    catch { importError = "Rename failed: \(error.localizedDescription)" }
                }
                renaming = nil
            }
            Button("Cancel", role: .cancel) { renaming = nil }
        }
        .confirmationDialog("Delete \"\(deleting?.name ?? "")\"? This removes the project folder.",
                            isPresented: Binding(
                                get: { deleting != nil }, set: { if !$0 { deleting = nil } }),
                            titleVisibility: .visible) {
            Button("Delete Project", role: .destructive) {
                if let p = deleting {
                    do { try model.store.delete(p) }
                    catch { importError = "Delete failed: \(error.localizedDescription)" }
                }
                deleting = nil
            }
        }
    }

    private var grid: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 220, maximum: 300), spacing: 16)],
                      spacing: 16) {
                ImportCard { model.showImportSheet = true }
                ForEach(model.store.projects) { p in
                    ProjectCard(project: p, directory: model.store.directory(for: p))
                        .contentShape(Rectangle())
                        .onTapGesture { model.openProject(p) }
                        .contextMenu {
                            Button("Open") { model.openProject(p) }
                            Button("Rename…") { renameText = p.name; renaming = p }
                            Divider()
                            Button("Delete…", role: .destructive) { deleting = p }
                        }
                }
            }
            .padding(20)
        }
    }
}

/// The Import card pinned as the grid's first cell (app round 4 spec item 7, with the layout
/// restructured in app round 5 spec item 1).
/// Mimicking the thumbnail zone (130pt plus text below) was dropped on the feedback that it looked
/// awkward without an image — the plus icon, "Import Project", and the supporting caption are
/// centered vertically along the card's full height.
/// The card height MUST match ProjectCard for the grid rows to line up, so ProjectCard's skeleton
/// (the 130pt zone plus the text zone's fonts, spacing, and padding) is kept as a hidden sizing
/// view with the centered content layered over it — height parity does not break even when the
/// font metrics change.
/// The dashed border (marking it as a "placeholder") is kept, and there is no context menu.
private struct ImportCard: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack {
                sizingSkeleton.hidden()
                VStack(spacing: 8) {
                    Image(systemName: "plus")
                        .font(.system(size: 34, weight: .light))
                        .foregroundStyle(SplatXTheme.accent)
                    VStack(spacing: 3) {
                        Text("Import Project")
                            .font(.callout.weight(.semibold))
                            .foregroundStyle(SplatXTheme.textPrimary)
                            .lineLimit(1)
                        Text("Images · Video · COLMAP")
                            .font(.caption)
                            .foregroundStyle(SplatXTheme.textSecondary)
                            .lineLimit(1)
                    }
                }
                .padding(10)
            }
            .frame(maxWidth: .infinity)
            .background(SplatXTheme.panel)
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10)
                .strokeBorder(SplatXTheme.textSecondary.opacity(0.45),
                              style: StrokeStyle(lineWidth: 1, dash: [6, 4])))
            .contentShape(RoundedRectangle(cornerRadius: 10))
        }
        .buttonStyle(.plain)
    }

    /// The height reference view with the same skeleton as ProjectCard (layout only, hidden).
    /// If ProjectCard's cell composition changes, this MUST be synced too for height parity to
    /// hold.
    private var sizingSkeleton: some View {
        VStack(alignment: .leading, spacing: 0) {
            Color.clear.frame(height: 130)
            VStack(alignment: .leading, spacing: 3) {
                Text(verbatim: " ").font(.callout.weight(.semibold)).lineLimit(1)
                Text(verbatim: " ").font(.caption).monospacedDigit().lineLimit(1)
            }
            .padding(10)
        }
    }
}

/// If the cell skeleton (the 130pt thumbnail zone plus the text zone's fonts, spacing, and
/// padding) changes, ImportCard's sizingSkeleton MUST be synced too for grid row height parity to
/// hold (see the ImportCard comment).
struct ProjectCard: View {
    let project: Project
    let directory: URL

    private var thumbnailURL: URL? {
        if let run = project.latestRun, let preview = run.previewFile {
            let url = directory.appendingPathComponent("runs/\(run.id)/\(preview)")
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return ImportService.listImages(in: directory.appendingPathComponent("images")).first
    }

    private var badge: (String, Color) {
        switch project.state {
        case .imported: ("Imported", SplatXTheme.textSecondary)
        case .aligned: ("Aligned", SplatXTheme.accent)
        case .trained: ("Trained", Color.green)
        }
    }

    /// The card's fallback glyph — shown only until the project has a frame or a render preview to
    /// show, i.e. between the import finishing and the first image being readable.
    ///
    /// It answers "where did this come from", which is the only thing known at that moment, and
    /// `.capture` gets its own glyph rather than sharing the photo one because "this came off the
    /// phone" is the distinction a user scanning the grid is actually looking for. `.colmap` shares
    /// the photo glyph deliberately: it is a folder of stills like `.images`, differing in what
    /// came *with* them.
    ///
    /// Switched exhaustively so a future source kind is a compile error here rather than a silently
    /// wrong icon. Tests: CaptureImportFlowTests.
    static func placeholderSymbol(for kind: ProjectSourceKind) -> String {
        switch kind {
        case .video: "video"
        case .capture: "iphone"
        case .images, .colmap: "photo.on.rectangle"
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ZStack(alignment: .topTrailing) {
                thumbnail
                Text(badge.0)
                    .font(.caption2.weight(.semibold))
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(.black.opacity(0.55), in: Capsule())
                    .foregroundStyle(badge.1)
                    .padding(6)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(project.name)
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(SplatXTheme.textPrimary)
                    .lineLimit(1)
                HStack {
                    Text("\(project.imageCount) images")
                    Spacer()
                    if let psnr = project.latestRun?.finalPSNR {
                        Text(String(format: "%.1f dB", psnr)).foregroundStyle(Color.green)
                    }
                }
                .font(.caption).monospacedDigit()
                .foregroundStyle(SplatXTheme.textSecondary)
            }
            .padding(10)
        }
        .background(SplatXTheme.elevated)
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(SplatXTheme.hairline))
    }

    @ViewBuilder private var thumbnail: some View {
        if let url = thumbnailURL, let image = NSImage(contentsOf: url) {
            Image(nsImage: image)
                .resizable()
                .aspectRatio(contentMode: .fill)
                .frame(height: 130)
                .clipped()
        } else {
            ZStack {
                SplatXTheme.panel
                Image(systemName: Self.placeholderSymbol(for: project.sourceKind))
                    .font(.title)
                    .foregroundStyle(SplatXTheme.textSecondary)
            }
            .frame(height: 130)
        }
    }
}
