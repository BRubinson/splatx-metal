// Cinema Task 1: the data model contract — the Codable round trip, w-first quat
// serialization, and the conventions for reserved fields and defaults.
// project.json persistence (Task 3) uses this format verbatim, so the serialized shape is
// pinned here.
import XCTest
import simd
@testable import SplatXMetalTrainer

final class CameraPathModelTests: XCTestCase {

    // MARK: Codable round trip

    // A path with every field set to a non-default value must be identical after a JSON round
    // trip.
    func testCameraPathCodableRoundTrip() throws {
        let path = CameraPath(
            id: UUID(), name: "Orbit 360° A",  // non-ASCII kept for Codable round-trip coverage
            keyframes: [
                Keyframe(time: 0, pos: SIMD3(1, 2, 3),
                         quat: SIMD4(0.5, 0.25, 0.75, 1.0), fov: 0.8, roll: 0.1),
                Keyframe(time: 2.5, pos: SIMD3(-1, 0, 4),
                         quat: SIMD4(1, 0, 0, 0), fov: 1.1),
            ],
            smoothness: 0.7, ease: .none, loop: true, duration: 12.5)
        let data = try JSONEncoder().encode(path)
        let decoded = try JSONDecoder().decode(CameraPath.self, from: data)
        XCTAssertEqual(decoded, path)
    }

    func testOverlaySpecCodableRoundTrip() throws {
        let spec = OverlaySpec(imagePath: "overlays/logo.png", anchor: .topLeading,
                               offsetX: -2.5, offsetY: 3.0, sizePercent: 15,
                               margin: 6, opacity: 0.8)
        let data = try JSONEncoder().encode(spec)
        let decoded = try JSONDecoder().decode(OverlaySpec.self, from: data)
        XCTAssertEqual(decoded, spec)
    }

    // MARK: The w-first quat convention (the repository convention — the same as
    // CameraConversion.qvec)

    // Pins at the serialized-byte level that the JSON array comes out in [w, x, y, z] order.
    func testQuatSerializedWFirst() throws {
        let path = CameraPath(
            id: UUID(), name: "q",
            keyframes: [Keyframe(time: 0, pos: SIMD3(0, 0, 0),
                                 quat: SIMD4(0.5, 0.25, 0.75, 1.0), fov: 0.9)],
            duration: 1)
        let data = try JSONEncoder().encode(path)
        let root = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let keyframes = try XCTUnwrap(root["keyframes"] as? [[String: Any]])
        let quat = try XCTUnwrap(keyframes[0]["quat"] as? [Double])
        XCTAssertEqual(quat.count, 4)
        XCTAssertEqual(quat[0], 0.5, accuracy: 0)    // w comes first (an exact binary fraction — zero tolerance)
        XCTAssertEqual(quat[1], 0.25, accuracy: 0)   // x
        XCTAssertEqual(quat[2], 0.75, accuracy: 0)   // y
        XCTAssertEqual(quat[3], 1.0, accuracy: 0)    // z
    }

    // The stored SIMD4(w,x,y,z) ↔ simd_quatf conversion preserves w-first in both directions.
    func testQuatValueConvention() {
        let k = Keyframe(time: 0, pos: SIMD3(0, 0, 0),
                         quat: SIMD4(0.5, 0.25, 0.75, 1.0), fov: 0.9)
        XCTAssertEqual(k.quatValue.real, 0.5)
        XCTAssertEqual(k.quatValue.imag.x, 0.25)
        XCTAssertEqual(k.quatValue.imag.y, 0.75)
        XCTAssertEqual(k.quatValue.imag.z, 1.0)
        let fromRotation = Keyframe(time: 0, pos: SIMD3(0, 0, 0),
                                    rotation: simd_quatf(ix: 0.25, iy: 0.75, iz: 1.0, r: 0.5),
                                    fov: 0.9)
        XCTAssertEqual(fromRotation.quat, SIMD4(0.5, 0.25, 0.75, 1.0))
    }

    // MARK: Reserved fields and forward compatibility (the decodeIfPresent convention — the
    // Project.swift precedent)

    // Old JSON without roll (reserved) still decodes to 0.
    func testRollMissingDecodesToZero() throws {
        let json = """
        {"time": 1.5, "pos": [1, 2, 3], "quat": [1, 0, 0, 0], "fov": 0.9}
        """
        let k = try JSONDecoder().decode(Keyframe.self, from: Data(json.utf8))
        XCTAssertEqual(k.roll, 0)
        XCTAssertEqual(k.time, 1.5)
        XCTAssertEqual(k.pos, SIMD3(1, 2, 3))
    }

    // Minimal JSON without smoothness/ease/loop/keyframes → the defaults. An unknown ease
    // string is leniently decoded to the default easeInOut (so older builds stay backward
    // compatible when future profiles are added).
    func testCameraPathForwardCompatDefaults() throws {
        let json = """
        {"id": "6F1C6C2E-8C1A-4E1B-9C0D-123456789ABC", "name": "min", "duration": 8}
        """
        let p = try JSONDecoder().decode(CameraPath.self, from: Data(json.utf8))
        XCTAssertEqual(p.keyframes.count, 0)
        XCTAssertEqual(p.smoothness, 0.5, accuracy: 0)
        XCTAssertEqual(p.ease, .easeInOut)
        XCTAssertFalse(p.loop)
        XCTAssertEqual(p.duration, 8)

        let unknownEase = """
        {"id": "6F1C6C2E-8C1A-4E1B-9C0D-123456789ABC", "name": "u", "duration": 1,
         "ease": "hyperBounce"}
        """
        let u = try JSONDecoder().decode(CameraPath.self, from: Data(unknownEase.utf8))
        XCTAssertEqual(u.ease, .easeInOut)
    }

    // Minimal OverlaySpec JSON (imagePath only) → the watermark defaults.
    func testOverlaySpecDefaults() throws {
        let json = """
        {"imagePath": "overlays/mark.png"}
        """
        let s = try JSONDecoder().decode(OverlaySpec.self, from: Data(json.utf8))
        XCTAssertEqual(s.anchor, .bottomTrailing)
        XCTAssertEqual(s.offsetX, 0, accuracy: 0)
        XCTAssertEqual(s.offsetY, 0, accuracy: 0)
        XCTAssertEqual(s.sizePercent, 20, accuracy: 0)
        XCTAssertEqual(s.margin, 4, accuracy: 0)
        XCTAssertEqual(s.opacity, 1, accuracy: 0)
    }

    // MARK: The nine anchors — rawValue is the persistence format, so the strings are fixed
    func testOverlayAnchorNineCasesStableRawValues() {
        XCTAssertEqual(OverlayAnchor.allCases.count, 9)
        XCTAssertEqual(OverlayAnchor.topLeading.rawValue, "topLeading")
        XCTAssertEqual(OverlayAnchor.center.rawValue, "center")
        XCTAssertEqual(OverlayAnchor.bottomTrailing.rawValue, "bottomTrailing")
    }
}
