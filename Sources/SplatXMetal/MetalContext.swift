import Foundation
import Metal

/// The Metal execution context. Inject it explicitly — global singletons are forbidden
/// (design document §3.3).
public final class MetalContext {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let library: MTLLibrary

    public enum Error: Swift.Error {
        case noDevice
        case noQueue
        case noLibrary
    }

    public init(device explicitDevice: MTLDevice?) throws {
        guard let device = explicitDevice ?? MTLCreateSystemDefaultDevice() else {
            throw Error.noDevice
        }
        guard let queue = device.makeCommandQueue() else { throw Error.noQueue }
        self.device = device
        self.queue = queue
        self.library = try Self.makeLibrary(device: device)
    }

    /// Layered metallib loading.
    ///
    /// 1. Precompiled fast path: holds only in configurations where the build system puts
    ///    `default.metallib` into the module bundle. Under the current `.copy("Kernels")`
    ///    resource configuration, even an Xcode build produces no metallib — measured to use
    ///    layer (b), runtime compilation.
    /// 2. Runtime compilation path: the SwiftPM native build system has no `.metal` compilation
    ///    rule, so no metallib is generated. Instead it reads the `.metal` sources copied
    ///    verbatim into the bundle by `resources: [.copy("Kernels")]` and compiles them at
    ///    runtime (`swift test` / `swift run`).
    private static func makeLibrary(device: MTLDevice) throws -> MTLLibrary {
        if let precompiled = try? device.makeDefaultLibrary(bundle: .module) {
            return precompiled
        }
        guard let urls = Bundle.module.urls(forResourcesWithExtension: "metal",
                                            subdirectory: "Kernels"),
              !urls.isEmpty else {
            throw Error.noLibrary
        }
        let source = try urls
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .map { try String(contentsOf: $0, encoding: .utf8) }
            .joined(separator: "\n")
        return try device.makeLibrary(source: source, options: nil)
    }
}
