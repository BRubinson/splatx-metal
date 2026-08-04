import SwiftUI

/// The single source of truth for app actions. One `AppCommand` simultaneously backs a menu bar
/// item, a keyboard shortcut, and (Task 10) the ⌘K palette.
///
/// - `isEnabled`: whether it can run in the current state (a tab switch requires an open project,
///   for example). The menus and palette show it disabled, and `execute` gates the run.
/// - `run`: the actual action. It runs on the main actor (mostly model and VM changes).
struct AppCommand: Identifiable {
    let id: String
    let title: String
    let subtitle: String?
    let shortcut: KeyboardShortcut?
    let isEnabled: () -> Bool
    let run: @MainActor () -> Void

    init(
        id: String,
        title: String,
        subtitle: String? = nil,
        shortcut: KeyboardShortcut? = nil,
        isEnabled: @escaping () -> Bool = { true },
        run: @escaping @MainActor () -> Void
    ) {
        self.id = id
        self.title = title
        self.subtitle = subtitle
        self.shortcut = shortcut
        self.isEnabled = isEnabled
        self.run = run
    }
}

/// The command bus: it looks up and runs registered commands by id. A run must pass the enable
/// gate.
@MainActor final class CommandRegistry {
    private(set) var commands: [AppCommand] = []

    /// Registers a command (registration order = the display order in the menus and palette).
    func register(_ command: AppCommand) {
        commands.append(command)
    }

    /// Runs the command with the given id, doing nothing when it is disabled.
    func execute(id: String) {
        guard let command = commands.first(where: { $0.id == id }), command.isEnabled() else { return }
        command.run()
    }

    /// Returns only the currently enabled commands.
    func enabled() -> [AppCommand] {
        commands.filter { $0.isEnabled() }
    }
}
