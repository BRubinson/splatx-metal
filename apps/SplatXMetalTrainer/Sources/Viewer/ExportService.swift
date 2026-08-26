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
    /// shared by both entry points (generalized in app round 2 Task 7, with the format parameter
    /// added in app round 4 FIX 3):
    /// - the View tab Inspector's EXPORT section buttons (the run open in the viewer —
    ///   `Inspector/RunPropertiesView.swift`). This replaced the window toolbar's Export… menu,
    ///   which was removed whole in 2026-08-05 spec §1.2; if you are looking for that menu, it is
    ///   gone deliberately and must not come back (two entry points was the rejected option).
    /// - the RUNS context Export… submenu (an arbitrary run — `Inspector/InspectorView.swift`).
    ///
    /// "Both entry points" counts the entry points *of the run's splat output*. The EXPORT
    /// section's third button (the timelapse video) does **not** come through here — see
    /// `presentTimelapsePanel` for why it must not.
    ///
    /// The panel's allowedContentTypes holds only the selected format.
    /// Failures are reported through the viewer overlay (viewerVM.errorMessage) — unchanged by the
    /// entry-point move, and correct for both callers because both live in the View tab where that
    /// overlay is on screen.
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

    /// Presents the save panel for a run's timelapse video (2026-08-05 spec §2.2) — called from
    /// the View tab Inspector's EXPORT section, and only when the file is known to exist on disk
    /// (`RunPropertiesView.timelapseState` returned `.ready`).
    ///
    /// **Why this is a separate function and not another `format:` case of `presentPanel`.**
    /// That function's whole body is the PLY↔SPZ format decision: the caller picks a target
    /// format up front and `export(from:to:)` either copies or calls `sx_convert_run`. A
    /// timelapse has no target format to pick — the container was decided at training time and is
    /// whatever the encoder produced, so the panel offers exactly one type and the transfer is a
    /// plain copy. Folding it into `presentPanel` would mean a format parameter that is ignored
    /// on one branch and a conversion path that must never be reached for video (`sx_convert_run`
    /// would happily be handed an .mp4 and fail deep inside the PLY reader).
    ///
    /// The source extension drives both the allowed type and the suggested name, so mp4/mov/gif
    /// need no branching here. The name is "<project>-timelapse.<ext>" rather than the source's
    /// own "timelapse.<ext>": every run in every project produces that same file name, and a
    /// Downloads folder full of `timelapse.gif`, `timelapse 2.gif` is the predictable result.
    ///
    /// Failures land on the viewer overlay for the same reason as `presentPanel` — this caller
    /// lives in the View tab where that overlay is on screen.
    @MainActor
    static func presentTimelapsePanel(source: URL, session: ProjectSession, model: AppModel) {
        let ext = source.pathExtension
        let panel = NSSavePanel()
        panel.allowedContentTypes = [UTType(filenameExtension: ext)].compactMap { $0 }
        panel.canSelectHiddenExtension = true
        panel.nameFieldStringValue = "\(session.project.name)-timelapse.\(ext)"
        guard panel.runModal() == .OK, let dst = panel.url else { return }
        do {
            let fm = FileManager.default
            // The panel already got the user's overwrite confirmation, but copyItem throws on an
            // existing destination — the same removeItem-then-copy shape as `export`.
            if fm.fileExists(atPath: dst.path) { try fm.removeItem(at: dst) }
            try fm.copyItem(at: source, to: dst)
            model.console.append("↑ Exported \(dst.lastPathComponent)")
        } catch {
            session.viewerVM.errorMessage =
                (error as? LocalizedError)?.errorDescription ?? "\(error)"
        }
    }
}
