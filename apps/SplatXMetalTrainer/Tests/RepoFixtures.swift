// Deleted in the first restructuring and revived for the camera fit (2026-07-20 spec §6)
import Foundation

// A bundle anchor for pointing at the test bundle's resources (for Bundle(for:)).
private final class RepoFixturesAnchor {}

enum RepoFixtures {
    // The colmap_mini folder reference is copied wholesale under the test bundle's resource
    // directory.
    // Reading from the bundle itself instead of `~/Desktop` (#filePath) avoids the macOS TCC
    // consent prompt.
    static var colmapMiniSparse: URL {
        let bundle = Bundle(for: RepoFixturesAnchor.self)
        let resources = bundle.resourceURL ?? bundle.bundleURL
        // A folder reference (a blue folder) lands at the resource root as `colmap_mini/`.
        let expected = resources.appending(path: "colmap_mini/sparse/0")
        if FileManager.default.fileExists(atPath: expected.appending(path: "cameras.bin").path) {
            return expected
        }
        // Fallback: for cases such as only the leaf folder being referenced, so that
        // `sparse/0` sits directly under the resource root, actually search the bundle for
        // cameras.bin and return its parent directory.
        if let enumerator = FileManager.default.enumerator(at: resources, includingPropertiesForKeys: nil) {
            for case let url as URL in enumerator where url.lastPathComponent == "cameras.bin" {
                return url.deletingLastPathComponent()
            }
        }
        return expected
    }
}
