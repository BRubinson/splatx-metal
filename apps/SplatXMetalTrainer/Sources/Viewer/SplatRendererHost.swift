import Foundation
import Metal
import MetalKit
import simd
import MetalSplatter
import SplatIO

// Note: BoundingBox has a single definition in CameraController.swift (Task 9), so it is not
// redefined here.

/// The host that wraps MetalSplatter's `SplatRenderer` and handles scene loading and rendering.
@MainActor final class SplatRendererHost {
    /// The in-flight frame limit of the MetalSplatter renderer. It MUST match the semaphore value.
    /// 3 → 2 (FIX 3, presentation latency against the overlay) → 1 (a 2026-07-28 experiment:
    /// shortening the relative latency by which the in-pass overlay appeared to trail the SwiftUI
    /// handles) → **2 (path transform FIX 14)**.
    /// The only motivation for lowering it to 1 was the presentation latency of the in-pass
    /// overlay, and that motivation vanished when FIX 14 returned the ribbon, frustum, and
    /// diamonds to a SwiftUI Canvas — the in-flight depth now affects **only splat presentation
    /// latency**, and overlay consistency is guaranteed by `overlayCamera` (projection with the
    /// presented frame). A value of 1 makes the CPU wait for the previous frame to complete and
    /// cuts throughput on heavy scenes, so it returns to FIX 3's value (2).
    static let maxSimultaneousRenders = 2

    /// The presentation notification (overlay synchronization — 2026-07-28): at the completion of
    /// the command buffer of a frame that actually rendered, it delivers the camera value written
    /// for that frame, or nil when there is no scene (isReadyToRender == false).
    /// The host knows only about rendering and nothing about the model — the wiring is owned by
    /// ViewerModel. Its consumers are every SwiftUI surface that reads `overlayCamera` (since
    /// FIX 14: the path ribbon, frustum, and diamonds plus the gizmo handles).
    ///
    /// **The first argument is the notification's origin (= this host)** (path transform FIX 17 —
    /// spec footnote 16). The receiver looks only at whether this reference is **the same
    /// instance** as the host it currently holds, and rejects leftover notifications from an
    /// abandoned host. Because self is carried as an argument, the wiring closure does not need to
    /// capture the host — the host owns the callback, so capturing it would be a retain cycle.
    ///
    /// **The ordering decision ends here, inside the producer** (FIX 17 consolidated the dispersion
    /// of FIX 15/16). The sequence is host-local and only this class issues it, so it discards
    /// stale and duplicate notifications itself, **before** calling the callback
    /// (`deliverPresented`). The receiver does not know about sequences — which is why an
    /// abandoned host's baseline disappears with that host and cross-contamination is structurally
    /// impossible.
    var onFramePresented: (@MainActor (SplatRendererHost, CameraController?) -> Void)?

    let device: MTLDevice
    let renderer: SplatRenderer
    /// One cached queue instead of recreating it every frame.
    private lazy var queue: MTLCommandQueue = device.makeCommandQueue()!
    /// The back pressure that limits the GPU to maxSimultaneousRenders in-flight frames.
    /// Identical to the reference MetalKitSceneRenderer — without it, continuous 60fps rendering
    /// submits command buffers without bound, the queue backs up, and the presented frame lags the
    /// camera state by whole seconds (= drag unresponsiveness and freezes).
    private let inFlight = DispatchSemaphore(value: SplatRendererHost.maxSimultaneousRenders)
    /// The monotonically increasing sequence of presentation notifications. It is issued **at
    /// submission time** (including the no-scene notification).
    /// It is @MainActor-isolated, so there is no race — both draw and the notification points are
    /// entirely on the main actor.
    private var frameSequence: UInt64 = 0
    /// The last sequence **delivered** (FIX 17 — ownership of the ordering decision moved to the
    /// producer).
    /// Why issuing (frameSequence) and delivering (this) diverge: the unstructured `Task` the
    /// completion handler creates can have two enqueued at once when in-flight is 2 or more, and
    /// Swift concurrency does not guarantee FIFO delivery of unstructured Tasks at different QoS
    /// levels. Letting out-of-order delivery through unchanged rewinds the receiver's
    /// `presentedCamera` by a frame and makes the overlay jump.
    private var lastDeliveredSequence: UInt64 = 0

    init(device: MTLDevice) throws {
        self.device = device
        self.renderer = try SplatRenderer(
            device: device,
            colorFormat: .bgra8Unorm_srgb, depthFormat: .depth32Float,
            sampleCount: 1, maxViewCount: 1, maxSimultaneousRenders: Self.maxSimultaneousRenders)
    }

    /// Reads the file, uploads it as a chunk, and returns the scene bounding sphere.
    @discardableResult
    func load(url: URL) async throws -> BoundingBox {
        let reader = try AutodetectSceneReader(url)
        let points = try await reader.readAll()
        let chunk = try SplatChunk(device: device, from: points)
        await renderer.removeAllChunks()
        await renderer.addChunk(chunk)
        return Self.bounds(of: points)
    }

    /// Delivers a presentation notification (the ordering decision is owned by the producer —
    /// FIX 17). It makes the **delivery order** monotonic rather than the issuing order:
    /// out-of-order and duplicate deliveries are discarded before the callback is called.
    /// draw's synchronous notification (no scene) and the completion handler's asynchronous
    /// notification pass through the same door — the two axes must share one sequence so that a
    /// late earlier frame does not overwrite "no scene".
    /// (Why internal rather than private: tests exercise the ordering axis directly, with no GPU.)
    func deliverPresented(_ camera: CameraController?, sequence: UInt64) {
        guard Self.deliversPresentedFrame(sequence: sequence, lastDelivered: lastDeliveredSequence)
        else { return }
        lastDeliveredSequence = sequence
        onFramePresented?(self, camera)
    }

    /// The delivery decision (pure — FIX 17). Only a sequence greater than the last delivered one
    /// is newer.
    /// An equal value (a duplicate delivery) is discarded too. The first notification is
    /// sequence 1 against the baseline 0, so it is always accepted.
    static func deliversPresentedFrame(sequence: UInt64, lastDelivered: UInt64) -> Bool {
        sequence > lastDelivered
    }

    static func bounds(of points: [SplatPoint]) -> BoundingBox {
        guard !points.isEmpty else { return BoundingBox(center: .zero, radius: 1) }
        var lo = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var hi = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        for p in points { let v = p.position; lo = min(lo, v); hi = max(hi, v) }
        let c = (lo + hi) * 0.5
        return BoundingBox(center: c, radius: max(length(hi - lo) * 0.5, 0.001))
    }
}

extension SplatRendererHost {
    /// The body of the MTKView render loop. The same semaphore pacing as the reference
    /// MetalKitSceneRenderer: wait on inFlight per frame → signal in the completion handler →
    /// always commit (present only on success).
    func draw(in view: MTKView, camera: CameraController) {
        // No scene — the overlay has nothing to align to, so it notifies the live camera fallback
        // (nil).
        // A missing drawable is a transient state and is not notified (the previously presented
        // camera is kept).
        guard renderer.isReadyToRender else {
            frameSequence &+= 1
            deliverPresented(nil, sequence: frameSequence)
            return
        }
        guard let drawable = view.currentDrawable else { return }

        // Back pressure up to the in-flight frame limit. This wait paces the 60fps over-submission
        // to the GPU's throughput.
        inFlight.wait()
        guard let cb = queue.makeCommandBuffer() else { inFlight.signal(); return }
        cb.addCompletedHandler { [inFlight] _ in inFlight.signal() }

        let w = view.drawableSize.width, h = view.drawableSize.height
        let size = SIMD2<Int>(Int(w), Int(h))
        let aspect = Float(w / max(h, 1))
        // The projection is injected here as an arbitrary matrix rather than computed inside
        // MetalSplatter — camera.verticalFOV is authoritative for the vertical FOV (the camera fit
        // sets it from the COLMAP intrinsics, spec §6).
        let vp = SplatRenderer.ViewportDescriptor(
            viewport: MTLViewport(originX: 0, originY: 0, width: w, height: h, znear: 0, zfar: 1),
            projectionMatrix: camera.projectionMatrix(aspect: aspect),
            viewMatrix: camera.viewMatrix,
            screenSize: size)

        let didRender: Bool
        do {
            didRender = try renderer.render(
                viewports: [vp],
                colorTexture: view.multisampleColorTexture ?? drawable.texture,
                colorStoreAction: view.multisampleColorTexture == nil ? .store : .multisampleResolve,
                depthTexture: view.depthStencilTexture,
                rasterizationRateMap: nil,
                renderTargetArrayLength: 0,
                to: cb)
        } catch { didRender = false }

        // The Cinema path overlay is not here — path transform FIX 14 returned the ribbon,
        // frustum, and diamonds to a SwiftUI Canvas (`PathOverlayLayer`) (footnote 13 reverses
        // footnote 6).
        // In-pass pixels lag the cursor by the whole present pipeline, and while a path is being
        // edited the camera is stationary, so the "consistency with the splats" that in-pass buys
        // is zero. Consistency is secured on the SwiftUI side by projecting with `overlayCamera`
        // (the notification below is its source of truth) — see the PathOverlayLayer comment.
        if didRender {
            cb.present(drawable)
            // The presented-camera notification: it returns the camera value actually written for
            // this frame (a struct copy) to the main actor at GPU completion. The overlay must
            // project with this value to be consistent with the on-screen splats, which passed
            // through the in-flight pipeline, on the same frame (2026-07-28).
            // The sequence is stamped **at submission time** (here) — because even with
            // out-of-order delivery it must be possible to decide which one is the newer frame.
            // The decision itself is made by deliverPresented at firing time (FIX 17 — the owner
            // of ordering is this class, which issues the sequence).
            frameSequence &+= 1
            let sequence = frameSequence
            cb.addCompletedHandler { [weak self] _ in
                Task { @MainActor in self?.deliverPresented(camera, sequence: sequence) }
            }
        }
        cb.commit()   // always commit — the completion handler must signal for the semaphore to stay balanced.
    }
}
