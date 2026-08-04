import Foundation

enum BundlePaths {
    static var resourceRoot: String { Bundle.main.resourcePath ?? Bundle.main.bundlePath }
    static var metallib: String { resourceRoot + "/splatx.metallib" }
    static var metallibURL: URL { URL(fileURLWithPath: metallib) }
    /// The full text of the open source notices (audit cleanup Task 7) — a bundle resource that
    /// postBuildScript assembles at build time from the repo's NOTICE, LICENSE, and the ThirdParty
    /// LICENSEs.
    static var acknowledgementsURL: URL {
        URL(fileURLWithPath: resourceRoot + "/Acknowledgements.txt")
    }
}
