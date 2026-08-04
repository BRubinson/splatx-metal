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
}
