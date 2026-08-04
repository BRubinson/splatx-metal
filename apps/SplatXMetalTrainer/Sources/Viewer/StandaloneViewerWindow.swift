import SwiftUI
import os

/// The scene id of the independent viewer window (app round 5 spec item 5) — the
/// openWindow(id:value:) routing key.
enum ViewerWindow {
    static let id = "splat-viewer"
}

/// The content of the independent viewer window opened through the Finder file association
/// (.ply/.spz) (a document window similar to Preview.app).
/// With its own ViewerModel, unrelated to the project session, it reuses the existing viewer stack
/// as-is (ViewerView → SplatMetalView → SplatRendererHost + ViewportOverlay). The initial viewpoint
/// is open()'s frameToFit, and the title is the file name. An open failure is shown through
/// ViewerView's error overlay.
/// Reopening the same URL brings the existing window to the front through the value-identity reuse
/// of WindowGroup(for:).
struct StandaloneViewerWindow: View {
    /// The presented value of WindowGroup(for: URL.self). It can be nil on paths such as window
    /// restoration — in that case ViewerView shows the empty-state (host == nil) placeholder.
    let url: URL?
    @State private var vm = ViewerModel()

    var body: some View {
        ViewerView(vm: vm)
            .frame(minWidth: 480, minHeight: 320)
            .navigationTitle(url?.lastPathComponent ?? "Splat Viewer")
            .task(id: url) {
                guard let url else { return }
                await vm.openFile(url)
            }
            .background(ViewerWindowFronter(url: url))   // brought to front and made key once, right after creation (FIX 2)
            .splatFileOpenRouting()   // later Finder opens work even when only the viewer window remains
            .tint(SplatXTheme.accent)
            .preferredColorScheme(.dark)
    }
}

/// Bringing the viewer window to the front (app round 5 FIX 2) — the E2E report: a viewer window
/// created by a Finder open (openWindow through the delegate) is laid **behind** the existing key
/// window (Home). The minimal fix raises it to key/front **exactly once**, right after the window
/// is created (makeNSView runs once per window and updateNSView does nothing — no repeated
/// re-fronting). It also registers the attached window with ViewerWindowRegistry to support
/// fronting when the same file is reopened (dedupe) — the value dedupe of SwiftUI
/// openWindow(value:) reuses the window but does not front it (measured).
private struct ViewerWindowFronter: NSViewRepresentable {
    let url: URL?

    func makeNSView(context: Context) -> NSView {
        let v = NSView()
        let url = url
        // At makeNSView time the window is not attached yet — front the attached window on the next
        // run loop.
        DispatchQueue.main.async {
            guard let w = v.window else { return }
            w.makeKeyAndOrderFront(nil)
            if let url { ViewerWindowRegistry.shared.register(url, window: w) }
            Logger(subsystem: "com.postmedia.splatx-metal-trainer", category: "file-open")
                .info("fronted viewer window: key=\(w.isKeyWindow) front=\(NSApp.orderedWindows.first === w) title=\(w.title, privacy: .public)")
        }
        return v
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}

/// A URL → viewer NSWindow weak-reference map (FIX 2). On a reopen of the same file the
/// openWindow(value:) dedupe does not front the existing window, so right after routing this map is
/// used to find the existing window and raise it to front and key.
/// A new URL is unregistered (false) → the ViewerWindowFronter on the creation path handles the
/// fronting (no double front — registration happens on the run loop after the window attaches, so
/// at routing time it is always unregistered). Dead references are cleaned up on registration.
@MainActor final class ViewerWindowRegistry {
    static let shared = ViewerWindowRegistry()
    private struct WeakWindow { weak var window: NSWindow? }
    private var byURL: [URL: WeakWindow] = [:]

    func register(_ url: URL, window: NSWindow) {
        byURL = byURL.filter { $0.value.window != nil }
        byURL[url] = WeakWindow(window: window)
    }

    /// If the registered window is still alive it raises it to front and key and returns true (the
    /// reopen dedupe fronting path).
    @discardableResult
    func front(_ url: URL) -> Bool {
        guard let w = byURL[url]?.window else { return false }
        w.makeKeyAndOrderFront(nil)
        Logger(subsystem: "com.postmedia.splatx-metal-trainer", category: "file-open")
            .info("re-fronted existing viewer window: key=\(w.isKeyWindow) front=\(NSApp.orderedWindows.first === w) title=\(w.title, privacy: .public)")
        return true
    }
}

/// The Finder open bridge queue — AppDelegate.application(_:open:) enqueues into it (receiving the
/// full URL array, avoiding onOpenURL's limitation of delivering only the first URL for a multi-file
/// open — measured in FIX 1), and the scene side that can reach the openWindow environment
/// (SplatFileOpenRouting) drains it. With a single MainActor consumer, even when several windows
/// observe it the first drain takes everything and the rest are no-ops on an empty queue.
@MainActor @Observable final class FileOpenRouter {
    static let shared = FileOpenRouter()
    private(set) var pending: [URL] = []

    func enqueue(_ urls: [URL]) { pending.append(contentsOf: urls) }
    func drain() -> [URL] {
        let p = pending
        pending = []
        return p
    }
}

/// The attachment point that routes Finder open events (.ply/.spz) to an independent viewer window.
/// The observation is alive only in the view hierarchy of an open window, so it is attached to
/// **both** the main window and the viewer windows — it works when either kind of window is all
/// that remains, a duplicate drain is neutralized by the router's empty queue, and reopening the
/// same URL reuses the window by value identity in WindowGroup(for:) (fronting the existing
/// window), so there are no duplicate viewer windows either.
///
/// handlesExternalEvents(preferring:allowing:) is a line of defense (measured in app round 5
/// Task 2): if no existing window claims an external event (odoc), macOS SwiftUI delivers it by
/// **creating a new window in the primary WindowGroup** — without it, a Home window multiplied
/// every time a file was opened (measured: 3 opens → 3 Home windows). Currently AppDelegate
/// intercepts odoc so it does nothing in normal operation, but the "*" claim is kept.
private struct SplatFileOpenRouting: ViewModifier {
    @Environment(\.openWindow) private var openWindow
    private let router = FileOpenRouter.shared

    func body(content: Content) -> some View {
        content
            .handlesExternalEvents(preferring: ["*"], allowing: ["*"])
            .onOpenURL { open([$0]) }   // a fallback — the AppDelegate path is authoritative (FIX 1)
            .task { open(router.drain()) }   // cold launch: consume what arrived before the view appeared
            .onChange(of: router.pending.count) { _, count in
                guard count > 0 else { return }
                open(router.drain())
            }
    }

    private func open(_ urls: [URL]) {
        for url in urls {
            // The document type declaration means only ply/spz arrive, but this is a defensive
            // filter in case a URL scheme is added later.
            guard url.isFileURL,
                  ["ply", "spz"].contains(url.pathExtension.lowercased()) else { continue }
            Logger(subsystem: "com.postmedia.splatx-metal-trainer", category: "file-open")
                .info("route → viewer window: \(url.path, privacy: .public)")
            openWindow(id: ViewerWindow.id, value: url)
            // Reopening the same file (dedupe — when the call above did not create a new window):
            // front the existing window.
            // A new URL is not yet registered, so this is a no-op and the Fronter on the creation
            // path handles it (FIX 2).
            // It is deferred to the next run loop — raising it synchronously lets the app
            // activation that completes right afterwards (cooperative activation) undo the window
            // order (measured: front=true inside the app, but the final z-order did not reflect
            // it).
            DispatchQueue.main.async { ViewerWindowRegistry.shared.front(url) }
        }
    }
}

extension View {
    /// A .ply/.spz open event → an independent viewer window (app round 5 item 5).
    func splatFileOpenRouting() -> some View { modifier(SplatFileOpenRouting()) }
}
