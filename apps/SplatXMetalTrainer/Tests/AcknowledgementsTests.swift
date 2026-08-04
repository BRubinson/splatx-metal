import XCTest
@testable import SplatXMetalTrainer

/// Audit cleanup Task 7 — the wiring contract for the in-bundle open-source notices.
/// Checks that the resource the postBuildScript "Assemble Acknowledgements.txt" assembles from
/// the repository originals (NOTICE, LICENSE, and ThirdParty/*/LICENSE) actually exists in the
/// app bundle (TEST_HOST), and that every required notice (the two MIT licenses + Apache-2.0 +
/// NOTICE) is present.
/// If the script breaks silently (a path typo, a moved file), it is caught here.
final class AcknowledgementsTests: XCTestCase {
    func testAcknowledgementsResourceExistsInBundle() {
        XCTAssertTrue(FileManager.default.fileExists(atPath: BundlePaths.acknowledgementsURL.path),
                      "the postBuildScript must assemble Acknowledgements.txt into the bundle")
    }

    func testAcknowledgementsContainsRequiredNotices() throws {
        let text = try XCTUnwrap(AcknowledgementsView.loadText())
        // The body of NOTICE (the derivation notices)
        XCTAssertTrue(text.contains("splatx-metal"))
        XCTAssertTrue(text.contains("gsplat"))
        XCTAssertTrue(text.contains("brush"))
        // The Apache-2.0 text
        XCTAssertTrue(text.contains("Apache License"))
        XCTAssertTrue(text.contains("Version 2.0"))
        // The two vendored MIT licenses — the copyright lines and the MIT text itself (the
        // substance of the notice-retention requirement)
        XCTAssertTrue(text.contains("MetalSplatter"))
        XCTAssertTrue(text.contains("Sean Cier"))
        XCTAssertTrue(text.contains("Niantic Labs"))
        XCTAssertTrue(text.contains("Permission is hereby granted, free of charge"))
    }

    func testAcknowledgementsIsAssembledFromRepoOriginals() throws {
        // The contract that forbids transcribing by hand — the full MIT text must appear
        // verbatim twice, once per component.
        let text = try XCTUnwrap(AcknowledgementsView.loadText())
        let mitPhrase = "THE SOFTWARE IS PROVIDED \"AS IS\""
        let count = text.components(separatedBy: mitPhrase).count - 1
        XCTAssertGreaterThanOrEqual(count, 2, "both MIT texts (MetalSplatter and spz) must be included in full")
    }
}
