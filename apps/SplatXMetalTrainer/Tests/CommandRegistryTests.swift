import XCTest
@testable import SplatXMetalTrainer

/// The core contract of the command registry: execute runs only ENABLED entries, and
/// enabled() filters down to the active entries. These two invariants underpin the gating logic
/// of the menu bar and the ⌘K palette.
final class CommandRegistryTests: XCTestCase {
    @MainActor func testExecuteRunsOnlyEnabled() {
        let reg = CommandRegistry()
        var ran = 0
        reg.register(AppCommand(id: "a", title: "A", subtitle: nil, shortcut: nil, isEnabled: { true }, run: { ran += 1 }))
        reg.register(AppCommand(id: "b", title: "B", subtitle: nil, shortcut: nil, isEnabled: { false }, run: { ran += 100 }))
        reg.execute(id: "a"); reg.execute(id: "b")   // b is disabled → it does not run
        XCTAssertEqual(ran, 1)
        XCTAssertEqual(reg.enabled().map(\.id), ["a"])
    }

    /// trainer-help-menu Task 1 — the Help menu's "Open Source Licenses" entry. The SwiftUI menu
    /// itself is not tested (app policy); the registry is the automated verification surface, and
    /// the menu's button(_:) helper renders exactly what is registered here.
    @MainActor func testOpenSourceLicensesCommandIsRegistered() {
        let model = AppModel()
        let reg = CommandRegistry.standard(model: model)
        let command = reg.commands.first { $0.id == AppCommandID.openSourceLicenses }
        XCTAssertNotNil(command, "the Help menu item is fed by the registry")
        XCTAssertEqual(command?.title, "Open Source Licenses")
        // Help items carry no keyboard shortcut in this app.
        XCTAssertNil(command?.shortcut)
        // It must not depend on a session — Help works on Home and inside a project alike.
        XCTAssertNil(model.session)
        XCTAssertEqual(command?.isEnabled(), true)
    }

    @MainActor func testOpenSourceLicensesCommandPresentsTheSheet() {
        let model = AppModel()
        let reg = CommandRegistry.standard(model: model)
        XCTAssertFalse(model.showAcknowledgements)
        reg.execute(id: AppCommandID.openSourceLicenses)
        XCTAssertTrue(model.showAcknowledgements)
    }
}
