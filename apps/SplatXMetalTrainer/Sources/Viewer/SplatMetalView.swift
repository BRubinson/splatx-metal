import SwiftUI
import MetalKit

/// Bridges the MTKView that drives `SplatRendererHost` into SwiftUI.
struct SplatMetalView: NSViewRepresentable {
    let host: SplatRendererHost
    @Binding var camera: CameraController

    func makeCoordinator() -> Coordinator { Coordinator(host: host, camera: $camera) }

    func makeNSView(context: Context) -> MTKView {
        let v = MTKView(frame: .zero, device: host.device)
        v.colorPixelFormat = .bgra8Unorm_srgb
        v.depthStencilPixelFormat = .depth32Float
        v.sampleCount = 1
        v.delegate = context.coordinator
        v.clearColor = MTLClearColorMake(0.05, 0.05, 0.06, 1)
        context.coordinator.attachGestures(to: v)
        return v
    }

    func updateNSView(_ nsView: MTKView, context: Context) {
        context.coordinator.camera = $camera
        // Reflect a host replacement (path transform FIX 17 — spec footnote 16). The coordinator
        // holds the host from makeCoordinator time, and this view lives inside
        // `if let host = vm.host`, so at present there is no replacement path that does not pass
        // through nil (open() and acquireRenderer create one only when it is nil, and clear()
        // assigns nil). If that premise breaks, the old host would keep drawing — and since the
        // notification wiring has already moved to the new host, the screen and the overlay would
        // look at different cameras — so as a minimal defense it is refreshed here every time (an
        // assignment of the same instance is harmless).
        context.coordinator.host = host
    }

    /// Explicit deregistration at view teardown (FIX 3 item 6) — cleaning dead entries out of the
    /// registry is deferred to the next input event, and until then an entry's camera Binding
    /// retained the ViewerModel (large scene memory) (observed after closing an independent viewer
    /// window). Clearing it at teardown returns it immediately.
    static func dismantleNSView(_ nsView: MTKView, coordinator: Coordinator) {
        CameraGestures.detach(from: nsView)
    }

    @MainActor final class Coordinator: NSObject, MTKViewDelegate {
        /// updateNSView refreshes it (FIX 17) — as a let, a host replacement would not be
        /// reflected.
        var host: SplatRendererHost
        var camera: Binding<CameraController>

        init(host: SplatRendererHost, camera: Binding<CameraController>) {
            self.host = host
            self.camera = camera
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

        func draw(in view: MTKView) {
            host.draw(in: view, camera: camera.wrappedValue)
        }

        func attachGestures(to view: MTKView) {
            CameraGestures.attach(to: view, camera: camera)
        }
    }
}
