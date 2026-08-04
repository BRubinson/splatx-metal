// Cinema FIX 5 (a user decision on 2026-07-27 — spec footnote 4): the seeding contract for
// the default brand watermark.
// (a) Once only — reopening after seeding never registers it again
// (b) Removal is respected — a watermark the user deleted never comes back
// (c) An existing overlay is untouched — a custom overlay stays as-is and only the seeded mark
//     is set
// + the bundle resource must load (the @2x suffixed name is fixed) + the seed defaults
// (anchor/size/margin/opacity).
// Pure logic only — file I/O goes to a temporary directory and GUI/TCC are never touched (the
// app work policy).
import XCTest
@testable import SplatXMetalTrainer

final class WatermarkSeederTests: XCTestCase {
    private var tmp: URL!
    private var source: URL!

    override func setUpWithError() throws {
        tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("WatermarkSeederTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        // Arbitrary bytes suffice as the seed source — only the copy contract is verified
        // (interpreting the image is the renderer's responsibility).
        source = tmp.appendingPathComponent("source.png")
        try Data([0x89, 0x50, 0x4E, 0x47]).write(to: source)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tmp)
    }

    private func project(overlay: OverlaySpec? = nil, seeded: Bool = false) -> Project {
        Project(id: UUID(), name: "p", sourceKind: .images, state: .trained,
                imageCount: 1, alignSummary: nil, runs: [],
                createdAt: Date(timeIntervalSince1970: 0),
                overlay: overlay, cinemaWatermarkSeeded: seeded)
    }

    private var copiedURL: URL { tmp.appendingPathComponent(WatermarkSeeder.relativePath) }

    // Seeding: an unseeded project with no overlay → the default spec is registered, a copy
    // is placed in the project, and the mark is set.
    func testSeedRegistersDefaultWatermark() throws {
        let seeded = try XCTUnwrap(WatermarkSeeder.seedIfNeeded(
            project: project(), projectDirectory: tmp, source: source))
        XCTAssertTrue(seeded.cinemaWatermarkSeeded)
        XCTAssertEqual(seeded.overlay, WatermarkSeeder.defaultSpec)
        XCTAssertTrue(FileManager.default.fileExists(atPath: copiedURL.path))
    }

    // (a) Once only: calling it again with the seeded result changes nothing (nil).
    func testSeedIsOneShot() throws {
        let first = try XCTUnwrap(WatermarkSeeder.seedIfNeeded(
            project: project(), projectDirectory: tmp, source: source))
        XCTAssertNil(WatermarkSeeder.seedIfNeeded(
            project: first, projectDirectory: tmp, source: source))
    }

    // (b) Removal is respected: overlay nil + seeded true (the state after the user deleted
    // it) is left unchanged.
    func testRemovedWatermarkStaysRemoved() {
        XCTAssertNil(WatermarkSeeder.seedIfNeeded(
            project: project(overlay: nil, seeded: true),
            projectDirectory: tmp, source: source))
        XCTAssertFalse(FileManager.default.fileExists(atPath: copiedURL.path))
    }

    // (c) An existing overlay is untouched: the custom spec stays as-is and only the seeded
    // mark is set — no file is copied either.
    func testExistingOverlayUntouched() throws {
        let custom = OverlaySpec(imagePath: "overlay/custom.png", anchor: .topLeading,
                                 sizePercent: 10, margin: 1, opacity: 0.5)
        let seeded = try XCTUnwrap(WatermarkSeeder.seedIfNeeded(
            project: project(overlay: custom), projectDirectory: tmp, source: source))
        XCTAssertEqual(seeded.overlay, custom)
        XCTAssertTrue(seeded.cinemaWatermarkSeeded)
        XCTAssertFalse(FileManager.default.fileExists(atPath: copiedURL.path))
    }

    // A copy failure (a missing source, or nil) is skipped silently but is not marked — the
    // next open retries.
    func testCopyFailureDoesNotMarkSeeded() {
        XCTAssertNil(WatermarkSeeder.seedIfNeeded(
            project: project(), projectDirectory: tmp,
            source: tmp.appendingPathComponent("missing.png")))
        XCTAssertNil(WatermarkSeeder.seedIfNeeded(
            project: project(), projectDirectory: tmp, source: nil))
    }

    // On a re-seed (for example reopening after the mark failed to persist), the copy with
    // the same name is replaced — idempotent.
    func testSeedOverwritesExistingCopy() throws {
        let dir = tmp.appendingPathComponent(OverlayStore.directoryName, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data([0x00]).write(to: copiedURL)
        _ = try XCTUnwrap(WatermarkSeeder.seedIfNeeded(
            project: project(), projectDirectory: tmp, source: source))
        XCTAssertEqual(try Data(contentsOf: copiedURL), try Data(contentsOf: source))
    }

    // Bundle resource: it must load under the @2x suffixed name verbatim (this verifies the
    // resource wiring in project.yml — the test host is the app, so Bundle.main = the app
    // bundle).
    func testBundleResourceLoads() throws {
        let url = try XCTUnwrap(WatermarkSeeder.bundleImageURL)
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
    }

    // MARK: Explicit user installation (app finishing, 2026-07-29 — the OVERLAY section's
    // "Add SplatX watermark")
    //
    // This path is **orthogonal** to the once-only seeding contract ((a)(b)(c) above).
    // Footnote 4's "once deleted it never comes back" is a contract about automatic
    // registration; this one is an installation the user explicitly requested.

    // Installation produces the same copy path and the same default placement as seeding (the
    // copy mechanism is defined in one place).
    func testInstallCopiesAndReturnsDefaultSpec() throws {
        let spec = try XCTUnwrap(WatermarkSeeder.install(into: tmp, source: source))
        XCTAssertEqual(spec, WatermarkSeeder.defaultSpec)
        XCTAssertEqual(try Data(contentsOf: copiedURL), try Data(contentsOf: source))
    }

    // Idempotent — safe against button mashing and reinstallation (the same-named file is
    // replaced).
    func testInstallIsIdempotent() throws {
        _ = try XCTUnwrap(WatermarkSeeder.install(into: tmp, source: source))
        _ = try XCTUnwrap(WatermarkSeeder.install(into: tmp, source: source))
        XCTAssertEqual(try Data(contentsOf: copiedURL), try Data(contentsOf: source))
    }

    // A missing source, or nil, returns nil (the caller tells the user — never fail
    // silently).
    func testInstallFailsWithoutSource() {
        XCTAssertNil(WatermarkSeeder.install(
            into: tmp, source: tmp.appendingPathComponent("missing.png")))
        XCTAssertNil(WatermarkSeeder.install(into: tmp, source: nil))
        XCTAssertFalse(FileManager.default.fileExists(atPath: copiedURL.path))
    }

    // Regression guard for the seeding contract: even after an explicit installation
    // following a user deletion (overlay nil + seeded true), **automatic seeding still does
    // nothing** — install never touches the flag.
    func testInstallDoesNotReviveAutomaticSeeding() throws {
        _ = try XCTUnwrap(WatermarkSeeder.install(into: tmp, source: source))
        let removed = project(overlay: nil, seeded: true)
        XCTAssertNil(WatermarkSeeder.seedIfNeeded(
            project: removed, projectDirectory: tmp, source: source))
    }

    // Reopening a project whose overlay was set by an explicit installation does not
    // overwrite the user's installation (the scenario where the flag is still false —
    // overlay != nil, so only the mark is set and nothing is copied).
    func testReopenAfterManualInstallKeepsUserSpec() throws {
        let spec = try XCTUnwrap(WatermarkSeeder.install(into: tmp, source: source))
        let seeded = try XCTUnwrap(WatermarkSeeder.seedIfNeeded(
            project: project(overlay: spec, seeded: false), projectDirectory: tmp,
            source: source))
        XCTAssertEqual(seeded.overlay, spec)
        XCTAssertTrue(seeded.cinemaWatermarkSeeded)
    }

    // MARK: Promoting the flag at persist time (app finishing FIX 1 ⑥ — campath spec
    // footnote 7)
    //
    // While the mark was deferred to "the next open", a deletion could slip in and automatic
    // seeding would revive the watermark the user had deleted. Moving the promotion forward to
    // persist time closes that window — the consumer is the onPersistOverlay handler in
    // AppModel.openProject (only the pure rule is pinned here).

    // The promotion rule: it rises only on a persist that made an overlay actually exist, and
    // it never falls.
    func testSeededFlagPromotionOnPersist() {
        let spec = WatermarkSeeder.defaultSpec
        // A non-nil persist: false → true (moving the next open's "mark only" forward)
        XCTAssertTrue(WatermarkSeeder.seededFlag(afterPersisting: spec, current: false))
        XCTAssertTrue(WatermarkSeeder.seededFlag(afterPersisting: spec, current: true))
        // A removal (nil) persist does not promote — the "retry after a copy failure"
        // contract (testCopyFailureDoesNotMarkSeeded) for projects where the watermark never
        // existed stays intact.
        XCTAssertFalse(WatermarkSeeder.seededFlag(afterPersisting: nil, current: false))
        // Nor does a removal demote it — removal is still respected (footnote 4).
        XCTAssertTrue(WatermarkSeeder.seededFlag(afterPersisting: nil, current: true))
    }

    // End-to-end verification of the ⑥ reproduction scenario: a copy failure (the flag stays
    // false) → a manual installation (persist promotes it) → a deletion **before** reopening
    // (no demotion) → reopening — automatic seeding must do nothing.
    // Before the promotion existed, this reopen revived the deleted watermark (flag false +
    // overlay nil).
    func testInstallThenDeleteBeforeReopenDoesNotRevive() throws {
        // First open: the copy fails → not marked (the flag stays false — the retry contract)
        XCTAssertNil(WatermarkSeeder.seedIfNeeded(
            project: project(), projectDirectory: tmp, source: nil))
        // Manual installation → persist promotes the flag
        let spec = try XCTUnwrap(WatermarkSeeder.install(into: tmp, source: source))
        var flag = WatermarkSeeder.seededFlag(afterPersisting: spec, current: false)
        XCTAssertTrue(flag)
        // Deletion persist → no demotion
        flag = WatermarkSeeder.seededFlag(afterPersisting: nil, current: flag)
        XCTAssertTrue(flag)
        // Reopen: what the user deleted does not come back
        XCTAssertNil(WatermarkSeeder.seedIfNeeded(
            project: project(overlay: nil, seeded: flag),
            projectDirectory: tmp, source: source))
    }

    // The seed defaults (a user decision on 2026-07-27): bottomTrailing / 28% / margin 4 /
    // opacity 0.9.
    // sizePercent 28 is inside the OverlayEditor Size slider range (2...60) — no need to widen
    // the upper bound.
    func testDefaultSpecValues() {
        let spec = WatermarkSeeder.defaultSpec
        XCTAssertEqual(spec.imagePath, "overlay/splatx-metal-watermark.png")
        XCTAssertEqual(spec.anchor, .bottomTrailing)
        XCTAssertEqual(spec.sizePercent, 28)
        XCTAssertEqual(spec.margin, 4)
        XCTAssertEqual(spec.opacity, 0.9)
        XCTAssertEqual(spec.offsetX, 0)
        XCTAssertEqual(spec.offsetY, 0)
    }
}
