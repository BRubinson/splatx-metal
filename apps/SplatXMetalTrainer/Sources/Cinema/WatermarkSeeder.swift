// Cinema FIX 5 (a 2026-07-27 user decision — spec footnote 4): automatic registration of the
// default brand watermark.
// Matching the main use of Cinema export (demos and portfolio pieces), it **seeds once** the
// SplatX Metal brand lockup as the default watermark — if the user deletes it, it does not come
// back (the Project.cinemaWatermarkSeeded flag is authoritative for respecting the removal).
//
// Wiring: AppModel.openProject calls seedIfNeeded **before** creating the ProjectSession and
// updates and saves the project — the existing injection wiring of cinemaVM.load(overlay:) then
// carries the seed result through unchanged.
// Additional wiring (app finishing FIX 1 ⑥): the onPersistOverlay handler in the same place
// promotes the flag at persist time through `seededFlag` — the seededFlag comment below is the
// authoritative rationale.
// Bundle access (Bundle.main) is factored out to the caller to keep the logic pure and
// TDD-friendly (WatermarkSeederTests — the contract is verified with only a temporary directory
// and a dummy source).
import Foundation

enum WatermarkSeeder {
    /// The file name of the project copy — a **neutral name** with the bundle original's @2x
    /// suffix removed (avoiding the trap of the resource scale suffix being interpreted, and a
    /// fixed name distinguishable from files the user imports).
    static let fileName = "splatx-metal-watermark.png"
    /// The project-relative path recorded in OverlaySpec.imagePath (the self-containment principle
    /// — the Task 4 idiom).
    static let relativePath = "\(OverlayStore.directoryName)/\(fileName)"
    /// The bundle resource name — project.yml references the repo's authoritative design/wordmark
    /// by relative path (no git duplication of an app-only copy). The name is kept exactly as the
    /// original (including @2x).
    static let bundleResourceName = "splatx-metal-lockup-dark@2x"

    /// The lockup PNG in the app bundle. The test host is the app, so Bundle.main is the app
    /// bundle in tests too.
    static var bundleImageURL: URL? {
        Bundle.main.url(forResource: bundleResourceName, withExtension: "png")
    }

    /// The seed's default placement (user decision 2026-07-27): bottomTrailing / 28% / margin 4 /
    /// opacity 0.9.
    /// sizePercent 28 = the width of the 5.3:1 lockup is 28% of the output's short side (about
    /// 302×57px at 1080p) — it is inside the OverlayEditor Size slider range (2...60), so it
    /// remains editable with no UI change.
    static var defaultSpec: OverlaySpec {
        OverlaySpec(imagePath: relativePath, anchor: .bottomTrailing,
                    sizePercent: 28, margin: 4, opacity: 0.9)
    }

    /// The seed-once semantics — it returns the updated project when an update is needed, and nil
    /// when nothing changed.
    /// - Already seeded (the flag is true): nil — it does not resurrect a watermark the user
    ///   deleted.
    /// - An overlay already exists: it only marks it seeded (the custom overlay is untouched and
    ///   no file is copied).
    /// - No overlay: on a successful source copy it registers the default spec and marks it. A
    ///   copy failure (a missing source, disk problems, and so on) returns nil silently — it does
    ///   **not** mark, so the next open retries (one attempt per open, so a retry loop that blocks
    ///   the UI is structurally impossible).
    static func seedIfNeeded(project: Project, projectDirectory: URL,
                             source: URL?) -> Project? {
        guard !project.cinemaWatermarkSeeded else { return nil }
        var updated = project
        if project.overlay == nil {
            guard let spec = install(into: projectDirectory, source: source) else {
                return nil
            }
            updated.overlay = spec
        }
        updated.cinemaWatermarkSeeded = true
        return updated
    }

    /// **Explicit user installation** (app finishing 2026-07-29 — the OVERLAY section's
    /// "Add SplatX watermark").
    /// The default spec on a successful copy; nil on failure (a missing source, disk problems).
    ///
    /// It **does not conflict** with footnote 4's "if the user deletes it, it does not come back" —
    /// the subject of that contract is "opening a project" and its object is **automatic
    /// registration**. It forbids automatic re-registration, not an installation the user
    /// requested with a button. Previously, once deleted there was no way in the UI to bring it
    /// back (there is not a single line of production code that sets `cinemaWatermarkSeeded` back
    /// to false, and the project's copy of the file is deleted along with it) — only that defect is
    /// closed.
    ///
    /// **It does not touch `cinemaWatermarkSeeded`.** ⑴ In the only situation where the button is
    /// needed (after the user deleted it) the flag is already true, so it would be meaningless;
    /// ⑵ the only case where the flag is false is "the marking was deferred by a copy failure so
    /// the next open retries", and overwriting it to true here would break that retry contract;
    /// and ⑶ CinemaViewModel cannot reach Project and has no means to touch it at all (it holds
    /// only overlay/projectDirectory/onPersistOverlay).
    /// Reopen safety comes from the previous "even if it were false, `overlay != nil` means it
    /// only marks without copying", plus the fact that the persist promotion (`seededFlag` — app
    /// finishing FIX 1 ⑥) now persists the flag as true right after installation, which also
    /// closes the remaining hole of "delete it before reopening and the automatic seed brings it
    /// back".
    static func install(into projectDirectory: URL, source: URL?) -> OverlaySpec? {
        guard let source, copyImage(from: source, into: projectDirectory) else { return nil }
        return defaultSpec
    }

    /// Promoting the seed flag at persist time (app finishing FIX 1 ⑥ — campath spec footnote 7).
    /// The `onPersistOverlay` handler in `AppModel.openProject` consumes it: on a persist where the
    /// overlay **came to exist** (including a manual installation or an image import) it raises the
    /// flag to true.
    ///
    /// It is not new semantics — it **brings forward, to persist time, the "an overlay already
    /// exists: only mark it" that the next open's seedIfNeeded used to do**. Previously, a deletion
    /// could slip in while that marking was deferred until the next reopen: the flag stayed false
    /// after a copy failure → the user installed it with the button → they deleted it **before**
    /// reopening → and on the next open the automatic seed resurrected the deleted watermark
    /// (violating footnote 4's "delete it and it does not come back"). Because the promotion is
    /// persisted immediately, this combination is closed.
    ///
    /// A removal (nil) persist does not promote (false stays false) — the "retry after a copy
    /// failure" contract for a project where the watermark never existed is unchanged. Nor does it
    /// demote a flag that is already true — exactly as the removal-respect source of truth
    /// (footnote 4) requires. And the constraint of install ⑶ is unchanged too — CinemaViewModel
    /// still does not know about Project, and the promotion is owned by the persist handler, the
    /// authority on the Project side.
    static func seededFlag(afterPersisting spec: OverlaySpec?, current: Bool) -> Bool {
        current || spec != nil
    }

    /// The bundle PNG → `<project>/overlay/splatx-metal-watermark.png`. It replaces a file of the
    /// same name — the same semantics as OverlayStore.importImage, but implemented separately
    /// because the destination name is fixed to a neutral one (importImage uses the source's
    /// lastPathComponent as-is).
    private static func copyImage(from source: URL, into projectDirectory: URL) -> Bool {
        let fm = FileManager.default
        let dir = projectDirectory.appendingPathComponent(OverlayStore.directoryName,
                                                          isDirectory: true)
        let dst = dir.appendingPathComponent(fileName)
        do {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
            if fm.fileExists(atPath: dst.path) { try fm.removeItem(at: dst) }
            try fm.copyItem(at: source, to: dst)
            return true
        } catch {
            return false
        }
    }
}
