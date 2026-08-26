import Foundation
import Observation

@MainActor @Observable final class AppModel {
    /// The single source of truth for the projects root. It is restored with scan() on entering
    /// Home.
    let store = ProjectStore()
    /// The open project (nil = Home). Switching it is the navigation.
    var session: ProjectSession?
    var showImportSheet = false

    var job: JobState = .idle
    var splatCount: Int?
    /// The bottom console's expanded state (the ⌘` toggle).
    var consoleExpanded = false
    /// The bottom console log (a ring buffer, cap 500).
    var console = ConsoleLog()
    /// The ⌘K command palette's visibility state.
    var showPalette = false
    /// The open source notices sheet's visibility state. It lives on the model (rather than as
    /// view-local @State) because the entry point is the Help menu — a menu command can only reach
    /// app-wide state, and the presenter has to sit above the Home/project branch so both show it.
    var showAcknowledgements = false

    func openProject(_ project: Project) {
        var project = project
        let directory = store.directory(for: project)
        // The default brand watermark seed (Cinema FIX 5 — spec footnote 4, seed-once semantics).
        // It must be applied **before** the session is created so the existing injection wiring of
        // cinemaVM.load(overlay:) carries the seed result.
        // On a save failure the flag is not persisted → the next open retries (the copy replaces a
        // file of the same name, so it is idempotent).
        if let seeded = WatermarkSeeder.seedIfNeeded(project: project,
                                                     projectDirectory: directory,
                                                     source: WatermarkSeeder.bundleImageURL) {
            project = seeded
            do { try store.save(project) }
            catch { console.append("⚠ watermark seed save failed: \(error.localizedDescription)") }
        }
        let session = ProjectSession(project: project, directory: directory)
        // Cinema path persistence (2026-07-26 spec): a VM change is written back into
        // project.cameraPaths and the manifest is saved (the existing convention — the same
        // store.save path as deleteRun). The store lives here, so the wiring does too.
        session.cinemaVM.onPersist = { [weak self, weak session] paths in
            guard let self, let session else { return }
            session.project.cameraPaths = paths
            do { try self.store.save(session.project) }
            catch { self.console.append("⚠ camera path save failed: \(error.localizedDescription)") }
        }
        // Overlay persistence (Task 4) — the same path as the camera paths (write back into
        // project and save the manifest).
        session.cinemaVM.onPersistOverlay = { [weak self, weak session] spec in
            guard let self, let session else { return }
            session.project.overlay = spec
            // App finishing FIX 1 ⑥: the seed flag is promoted immediately the moment the overlay
            // comes to exist — bringing forward, to persist time, the "only mark it" of the next
            // open's seedIfNeeded, which closes the combination "the flag stays false after a copy
            // failure → a manual installation → deletion before reopening → the automatic seed
            // resurrects it". The WatermarkSeeder.seededFlag comment is the authoritative rule.
            session.project.cinemaWatermarkSeeded = WatermarkSeeder.seededFlag(
                afterPersisting: spec, current: session.project.cinemaWatermarkSeeded)
            do { try self.store.save(session.project) }
            catch { self.console.append("⚠ overlay save failed: \(error.localizedDescription)") }
        }
        // Output aspect persistence (Task 5 — a Task 4 review recommendation) — the same path.
        session.cinemaVM.onPersistAspect = { [weak self, weak session] aspect in
            guard let self, let session else { return }
            session.project.outputAspect = aspect
            do { try self.store.save(session.project) }
            catch { self.console.append("⚠ aspect save failed: \(error.localizedDescription)") }
        }
        self.session = session
    }

    /// Returning Home during training is blocked by disabling it on the command and button side
    /// (the global contract of exactly one concurrent training run).
    func closeProject() { session = nil }
}
