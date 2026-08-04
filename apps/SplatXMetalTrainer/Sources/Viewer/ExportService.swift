import Foundation
import AppKit
import UniformTypeIdentifiers
import SplatXTrainC

enum ExportError: LocalizedError {
    case convertFailed(String)
    var errorDescription: String? {
        switch self {
        case .convertFailed(let why): "Convert failed: \(why)"
        }
    }
}

/// The export target format (app round 4 FIX 3). It is chosen up front in the entry-point menu, and
/// the save panel exposes only the selected format — previously the panel offered both ply and spz
/// with a default file name ending in .ply, so choosing SPZ was impossible unless you retyped the
/// extension by hand.
enum ExportFormat: String, CaseIterable {
    case ply, spz

    /// The file extension = rawValue — the same extension convention as the C surface
    /// (sx_convert_run).
    var fileExtension: String { rawValue }
    /// The Export menu item label — it states whether the format is compressed, giving a basis for
    /// the choice.
    var menuTitle: String {
        switch self {
        case .ply: "PLY (uncompressed)…"
        case .spz: "SPZ (compressed)…"
        }
    }
    var contentType: UTType? { UTType(filenameExtension: rawValue) }
}

/// Exports a run's output. With the same extension it copies; otherwise it converts with
/// sx_convert_run (PLY↔SPZ).
/// The extension convention is the same as the C surface — .spz (case-insensitive) means SPZ,
/// anything else means INRIA PLY.
enum ExportService {
    static func export(from src: URL, to dst: URL) throws {
        let fm = FileManager.default
        if src.pathExtension.lowercased() == dst.pathExtension.lowercased() {
            if fm.fileExists(atPath: dst.path) { try fm.removeItem(at: dst) }
            try fm.copyItem(at: src, to: dst)
            return
        }
        var err = [CChar](repeating: 0, count: 1024)
        let rc = sx_convert_run(src.path, dst.path, &err, Int32(err.count))
        guard rc == 0 else { throw ExportError.convertFailed(String(cString: err)) }
    }

    /// Presents the save panel and exports a run's output in the selected format — the single path
    /// shared by the toolbar Export… menu (the current run) and the RUNS context Export… submenu
    /// (an arbitrary run) (generalized in app round 2 Task 7, with the format parameter added in
    /// app round 4 FIX 3). The panel's allowedContentTypes holds only the selected format.
    /// Failures are reported through the viewer overlay (viewerVM.errorMessage) — the same as the
    /// existing toolbar behavior.
    @MainActor
    static func presentPanel(for record: RunRecord, session: ProjectSession, model: AppModel,
                             format: ExportFormat) {
        let src = session.directory.appendingPathComponent("runs/\(record.id)/\(record.outputFile)")
        let panel = NSSavePanel()
        panel.allowedContentTypes = [format.contentType].compactMap { $0 }
        panel.canSelectHiddenExtension = true
        panel.nameFieldStringValue = "\(session.project.name).\(format.fileExtension)"
        guard panel.runModal() == .OK, let dst = panel.url else { return }
        do {
            try export(from: src, to: dst)
            model.console.append("↑ Exported \(dst.lastPathComponent)")
        } catch {
            session.viewerVM.errorMessage =
                (error as? LocalizedError)?.errorDescription ?? "\(error)"
        }
    }
}
