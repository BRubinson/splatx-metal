import SwiftUI

/// Command identifiers (stable keys shared by the menus, the palette, and the tests).
enum AppCommandID {
    static let importProject = "file.importProject"
    static let goHome = "nav.home"
    static let tabAlign = "view.tabAlign"
    static let tabTrain = "view.tabTrain"
    static let tabView = "view.tabView"
    static let tabCinema = "view.tabCinema"
    static let toggleConsole = "view.toggleConsole"
    static let openPalette = "command.openPalette"
    // Cinema keyboard shortcuts (2026-07-26 spec UX 7) — enabled only in the Cinema **tab**
    // (footnote 2 ①).
    static let cinemaPlayPause = "cinema.playPause"
    static let cinemaCaptureKey = "cinema.captureKey"
    static let cinemaDeleteKey = "cinema.deleteKey"
    static let cinemaStepBack = "cinema.stepBack"
    static let cinemaStepForward = "cinema.stepForward"
}

extension CommandRegistry {
    /// The app's standard command set. Session-dependent commands read model.session at execution
    /// time — the registry has app lifetime (created once as @State) while the session is swapped
    /// out.
    static func standard(model: AppModel) -> CommandRegistry {
        let registry = CommandRegistry()

        registry.register(AppCommand(
            id: AppCommandID.importProject,
            title: "Import Project…",
            shortcut: KeyboardShortcut("o", modifiers: .command),
            isEnabled: { model.session == nil },   // only on the Home screen
            run: { model.showImportSheet = true }
        ))
        registry.register(AppCommand(
            id: AppCommandID.goHome,
            title: "Back to Projects",
            shortcut: KeyboardShortcut("h", modifiers: [.command, .shift]),
            // App round 5 spec item 8: resolving the asymmetry of blocking only training — the
            // authoritative lock (lockedTab) blocks align too.
            isEnabled: { model.session != nil && model.session?.lockedTab == nil },
            run: { model.closeProject() }
        ))

        let tabs: [(String, String, Character, ProjectTab)] = [
            (AppCommandID.tabAlign, "Align", "1", .align),
            (AppCommandID.tabTrain, "Train", "2", .train),
            (AppCommandID.tabView, "View", "3", .view),
            (AppCommandID.tabCinema, "Cinema", "4", .cinema),   // footnote 2 ① — the fourth tab
        ]
        for (id, title, key, tab) in tabs {
            registry.register(AppCommand(
                id: id, title: title,
                shortcut: KeyboardShortcut(KeyEquivalent(key), modifiers: .option),
                // App round 5 spec item 8: blocks switching to any tab other than the locked stage
                // (the locked tab itself is allowed).
                isEnabled: {
                    guard let session = model.session else { return false }
                    return session.lockedTab == nil || session.lockedTab == tab
                },
                run: {
                    guard let session = model.session else { return }
                    // A defensive re-check (the rename pattern) — independently of execute's enable
                    // gate, it guards against a state change between the menu/palette display
                    // snapshot and the moment of execution.
                    guard session.lockedTab == nil || session.lockedTab == tab else { return }
                    session.tab = tab
                }
            ))
        }

        registry.register(AppCommand(
            id: AppCommandID.toggleConsole,
            title: "Toggle Console",
            shortcut: KeyboardShortcut("`", modifiers: .command),
            run: { model.consoleExpanded.toggle() }
        ))

        // Cinema keyboard shortcuts (2026-07-26 spec UX 7: Space/Enter/Delete/←→ — keyboard
        // first).
        // They are unmodified key equivalents, so the gate is the contract: **the Cinema tab**
        // (footnote 2 ① — updated from the previous "View tab + Cinema active") + not editing text
        // (typing in the inline name or Duration fields MUST NOT be intercepted) + the palette
        // closed (protecting the palette's Enter and typing). A disabled menu item does not consume
        // the key, so ordinary typing is undisturbed.
        let cinema: () -> CinemaViewModel? = {
            guard let session = model.session, session.tab == .cinema,
                  !session.cinemaVM.isEditingText,
                  !model.showPalette else { return nil }
            return session.cinemaVM
        }
        registry.register(AppCommand(
            id: AppCommandID.cinemaPlayPause,
            title: "Play/Pause Path",
            subtitle: "Cinema",
            shortcut: KeyboardShortcut(.space, modifiers: []),
            isEnabled: { cinema() != nil },
            run: { cinema()?.togglePlayback() }
        ))
        registry.register(AppCommand(
            id: AppCommandID.cinemaCaptureKey,
            title: "Capture Keyframe",
            subtitle: "Cinema",
            shortcut: KeyboardShortcut(.return, modifiers: []),
            isEnabled: { cinema() != nil },
            run: { cinema()?.captureKeyframe() }
        ))
        registry.register(AppCommand(
            id: AppCommandID.cinemaDeleteKey,
            title: "Delete Keyframe",
            subtitle: "Cinema",
            shortcut: KeyboardShortcut(.delete, modifiers: []),
            isEnabled: { cinema()?.selectedKeyframeIndex != nil },
            run: { cinema()?.deleteSelectedKeyframe() }
        ))
        registry.register(AppCommand(
            id: AppCommandID.cinemaStepBack,
            title: "Step Back One Frame",
            subtitle: "Cinema",
            shortcut: KeyboardShortcut(.leftArrow, modifiers: []),
            isEnabled: { cinema() != nil },
            run: { cinema()?.stepFrame(-1) }
        ))
        registry.register(AppCommand(
            id: AppCommandID.cinemaStepForward,
            title: "Step Forward One Frame",
            subtitle: "Cinema",
            shortcut: KeyboardShortcut(.rightArrow, modifiers: []),
            isEnabled: { cinema() != nil },
            run: { cinema()?.stepFrame(1) }
        ))
        registry.register(AppCommand(
            id: AppCommandID.openPalette,
            title: "Command Palette…",
            shortcut: KeyboardShortcut("k", modifiers: .command),
            run: { model.showPalette = true }
        ))
        return registry
    }
}

/// The menu bar: File (Import) / View (Home, tabs, console, palette).
struct AppCommands: Commands {
    let registry: CommandRegistry

    var body: some Commands {
        CommandGroup(after: .newItem) {
            button(AppCommandID.importProject)
        }
        CommandGroup(after: .sidebar) {
            Divider()
            button(AppCommandID.goHome)
            Divider()
            button(AppCommandID.tabAlign)
            button(AppCommandID.tabTrain)
            button(AppCommandID.tabView)
            button(AppCommandID.tabCinema)
            Divider()
            button(AppCommandID.toggleConsole)
            Divider()
            button(AppCommandID.cinemaPlayPause)
            button(AppCommandID.cinemaCaptureKey)
            button(AppCommandID.cinemaDeleteKey)
            button(AppCommandID.cinemaStepBack)
            button(AppCommandID.cinemaStepForward)
            Divider()
            button(AppCommandID.openPalette)
        }
    }

    @ViewBuilder private func button(_ id: String) -> some View {
        if let command = registry.commands.first(where: { $0.id == id }) {
            Button(command.title) { registry.execute(id: command.id) }
                .disabled(!command.isEnabled())
                .keyboardShortcut(command.shortcut)
        }
    }
}
