// Cinema Task 4: the overlay placement arithmetic — every unit is a **percentage of the
// output's short side** (the contract handed over from Task 1).
// OverlayPlacement.rect is the single source of truth for the signature that the Task 5 export
// compositor reuses with output pixel dimensions (the viewport WYSIWYG path calls it with the
// mask's pt dimensions — the two are similar because the units are percentages).
// Lineage of the literals: output 1920×1080 (short side 1080), image 400×200, size 20% → long
// side 216 → 216×108, margin 4% → 43.2, offset 10% → 108.
import XCTest
@testable import SplatXMetalTrainer

final class OverlayPlacementTests: XCTestCase {

    private func spec(anchor: OverlayAnchor, offsetX: Double = 0, offsetY: Double = 0,
                      size: Double = 20, margin: Double = 4) -> OverlaySpec {
        OverlaySpec(imagePath: "overlay/logo.png", anchor: anchor,
                    offsetX: offsetX, offsetY: offsetY,
                    sizePercent: size, margin: margin, opacity: 1)
    }

    private let image = CGSize(width: 400, height: 200)

    func testCenterAnchorIgnoresMargin() {
        let r = OverlayPlacement.rect(spec: spec(anchor: .center), imageSize: image,
                                      outputWidth: 1920, outputHeight: 1080)
        XCTAssertEqual(r.width, 216, accuracy: 1e-9)     // long side = 1080×20%
        XCTAssertEqual(r.height, 108, accuracy: 1e-9)    // aspect ratio preserved
        XCTAssertEqual(r.minX, 852, accuracy: 1e-9)      // (1920−216)/2 — margin is not applied
        XCTAssertEqual(r.minY, 486, accuracy: 1e-9)      // (1080−108)/2
    }

    func testBottomTrailingWithMargin() {
        let r = OverlayPlacement.rect(spec: spec(anchor: .bottomTrailing), imageSize: image,
                                      outputWidth: 1920, outputHeight: 1080)
        XCTAssertEqual(r.minX, 1660.8, accuracy: 1e-9)   // 1920−216−43.2
        XCTAssertEqual(r.minY, 928.8, accuracy: 1e-9)    // 1080−108−43.2
    }

    func testTopLeadingWithFreeOffset() {
        let r = OverlayPlacement.rect(
            spec: spec(anchor: .topLeading, offsetX: 10, offsetY: 5, margin: 0),
            imageSize: image, outputWidth: 1920, outputHeight: 1080)
        XCTAssertEqual(r.minX, 108, accuracy: 1e-9)      // 0 + 1080×10%
        XCTAssertEqual(r.minY, 54, accuracy: 1e-9)       // 0 + 1080×5%
    }

    func testShortSideUnitIsInvariantAcrossAspects() {
        // Even for a 9:16 portrait output, as long as the short side is the same (1080) the
        // overlay's pixel size is identical — this is why the "percentage of the output's short
        // side" unit was chosen (WYSIWYG survives resolution and aspect-ratio changes).
        let r = OverlayPlacement.rect(spec: spec(anchor: .bottomTrailing), imageSize: image,
                                      outputWidth: 1080, outputHeight: 1920)
        XCTAssertEqual(r.width, 216, accuracy: 1e-9)
        XCTAssertEqual(r.height, 108, accuracy: 1e-9)
        XCTAssertEqual(r.minX, 820.8, accuracy: 1e-9)    // 1080−216−43.2
        XCTAssertEqual(r.minY, 1768.8, accuracy: 1e-9)   // 1920−108−43.2
    }

    func testTallImageBoundedByLongerSide() {
        // For a portrait image too, the long side is the limit — predictable regardless of
        // orientation (the contract in the function's comment).
        let r = OverlayPlacement.rect(spec: spec(anchor: .center),
                                      imageSize: CGSize(width: 200, height: 400),
                                      outputWidth: 1920, outputHeight: 1080)
        XCTAssertEqual(r.width, 108, accuracy: 1e-9)
        XCTAssertEqual(r.height, 216, accuracy: 1e-9)
    }

    func testDegenerateInputsReturnZero() {
        XCTAssertEqual(OverlayPlacement.rect(spec: spec(anchor: .center), imageSize: .zero,
                                             outputWidth: 1920, outputHeight: 1080), .zero)
        XCTAssertEqual(OverlayPlacement.rect(spec: spec(anchor: .center), imageSize: image,
                                             outputWidth: 0, outputHeight: 1080), .zero)
    }

    // MARK: The aspect-ratio mask (viewport letterboxing)

    func testMaskRectLetterboxAndPillarbox() {
        let container = CGSize(width: 800, height: 600)
        let wide = OverlayPlacement.maskRect(aspect: 16.0 / 9.0, in: container)
        XCTAssertEqual(wide, CGRect(x: 0, y: 75, width: 800, height: 450))
        let tall = OverlayPlacement.maskRect(aspect: 9.0 / 16.0, in: container)
        XCTAssertEqual(tall, CGRect(x: 231.25, y: 0, width: 337.5, height: 600))
        let square = OverlayPlacement.maskRect(aspect: 1, in: container)
        XCTAssertEqual(square, CGRect(x: 100, y: 0, width: 600, height: 600))
    }

    // MARK: Drag → offset conversion, and anchor switching

    func testOffsetDeltaConvertsPointsToShortSidePercent() {
        let d = OverlayPlacement.offsetDelta(translation: CGSize(width: 54, height: -27),
                                             shortSide: 540)
        XCTAssertEqual(d.dx, 10, accuracy: 1e-9)
        XCTAssertEqual(d.dy, -5, accuracy: 1e-9)
    }

    func testAnchoredResetsOffset() {
        let s = OverlayPlacement.anchored(
            spec(anchor: .bottomTrailing, offsetX: 7, offsetY: -3), to: .topLeading)
        XCTAssertEqual(s.anchor, .topLeading)
        XCTAssertEqual(s.offsetX, 0)
        XCTAssertEqual(s.offsetY, 0)
        XCTAssertEqual(s.sizePercent, 20)   // the remaining settings are kept
    }

    // MARK: The image copy (keeping the project self-contained)

    func testImportImageCopiesAndOverwrites() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory
            .appendingPathComponent("sx-overlay-test-\(UUID().uuidString)")
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }
        let src = dir.appendingPathComponent("logo.png")
        try Data([1, 2, 3]).write(to: src)

        let rel = try OverlayStore.importImage(from: src, projectDirectory: dir)
        XCTAssertEqual(rel, "overlay/logo.png")
        let copy = OverlayStore.url(for: rel, projectDirectory: dir)
        XCTAssertEqual(try Data(contentsOf: copy), Data([1, 2, 3]))

        try Data([9, 9]).write(to: src)                       // reselecting means updating
        _ = try OverlayStore.importImage(from: src, projectDirectory: dir)
        XCTAssertEqual(try Data(contentsOf: copy), Data([9, 9]))

        OverlayStore.removeCopy(relativePath: "elsewhere/logo.png", projectDirectory: dir)
        XCTAssertTrue(fm.fileExists(atPath: copy.path))       // paths outside overlay/ are untouched
        OverlayStore.removeCopy(relativePath: rel, projectDirectory: dir)
        XCTAssertFalse(fm.fileExists(atPath: copy.path))
    }
}

// MARK: - VM overlay state (the persistence timing wiring)

@MainActor
final class OverlayEditorVMTests: XCTestCase {

    func testSetOverlayPersistTiming() {
        let vm = CinemaViewModel()
        var saves = 0
        var last: OverlaySpec?
        vm.onPersistOverlay = { saves += 1; last = $0 }

        let spec = OverlaySpec(imagePath: "overlay/a.png")
        vm.setOverlay(spec, persistNow: false)     // during a drag or slider move — reflected live only
        XCTAssertEqual(vm.overlay, spec)
        XCTAssertEqual(saves, 0)
        vm.persistOverlay()                        // one persist at the end of the drag
        XCTAssertEqual(saves, 1)
        XCTAssertEqual(last, spec)

        vm.setOverlay(nil)                         // removal persists immediately
        XCTAssertEqual(saves, 2)
        XCTAssertNil(vm.overlay)
        XCTAssertNil(last)
    }

    func testLoadInjectsOverlayAndDirectoryResolvesURL() {
        let vm = CinemaViewModel()
        vm.projectDirectory = URL(fileURLWithPath: "/projects/p")
        vm.load(paths: [], overlay: OverlaySpec(imagePath: "overlay/logo.png"))
        XCTAssertEqual(vm.overlay?.imagePath, "overlay/logo.png")
        XCTAssertEqual(vm.overlayImageURL?.path, "/projects/p/overlay/logo.png")
        vm.load(paths: [])                         // default arguments — a project with no overlay
        XCTAssertNil(vm.overlay)
    }
}
