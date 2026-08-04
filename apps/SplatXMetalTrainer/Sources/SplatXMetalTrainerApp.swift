import SwiftUI
import AppKit

/// Closing the last window terminates the app (a developer tool: a window is a session).
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    /// Wiring the system notification delegate (app round 5 item 6) — it is required for the
    /// foreground banner (willPresent → .banner), so it is installed once at app start. The
    /// authorization request is not made here (it happens immediately before the first post —
    /// HIG).
    /// App finishing 2026-07-29: the splash is also presented once here (leaving the scene graph
    /// untouched — the SplashWindow header is the authoritative rationale). In a test environment,
    /// or with SPLATX_NO_SPLASH, the window is not even created.
    /// App finishing FIX 2: MainWindowGate keeps the main window off screen until the splash clears
    /// (the arming and reveal wiring lives inside presentIfEnabled — if a gate blocks, there is no
    /// arming either).
    func applicationDidFinishLaunching(_ notification: Notification) {
        MainActor.assumeIsolated {
            AppNotifier.shared.install()
            SplashWindow.presentIfEnabled()
        }
    }

    /// Receiving Finder odoc (file open) events — measured in app round 5 FIX 1: SwiftUI's
    /// onOpenURL delivers **only the first URL** when several files are opened at once (opening 2
    /// files → the handler runs once and the rest are lost).
    /// application(_:open:) receives the whole array, so this is the authoritative reception point,
    /// and the scene side (which can reach openWindow) consumes it through the router queue.
    /// Implementing this method makes SwiftUI's own odoc routing (onOpenURL and external-event
    /// scene creation) stop intervening (measured).
    func application(_ application: NSApplication, open urls: [URL]) {
        MainActor.assumeIsolated {
            // FIX 2: if the app is inactive at the moment of the Finder open, the new viewer window
            // is laid behind the existing windows (the E2E report). Opening a file is an explicit
            // user intent, so the app is activated — it is right after Finder yielded activation
            // for the open, so a cooperative activate() is sufficient (verified by
            // measurement).
            NSApp.activate()
            FileOpenRouter.shared.enqueue(urls)
        }
    }
}

@main
struct SplatXMetalTrainerApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var model: AppModel
    /// The app command bus — created once as @State (resolving the problem where the previous
    /// computed property recreated it on every body evaluation).
    @State private var registry: CommandRegistry
    // Inline project name editing (app round 4 spec item 6) — reusing the Home rename alert
    // pattern.
    @State private var isRenaming = false
    @State private var renameText = ""
    @State private var renameError: String?

    init() {
        let model = AppModel()
        _model = State(initialValue: model)
        _registry = State(initialValue: .standard(model: model))
    }

    var body: some Scene {
        WindowGroup {
            Group {
                if let session = model.session {
                    RootDetail(model: model, session: session, registry: registry)
                        .toolbar {
                            ToolbarItem(placement: .navigation) {
                                Button {
                                    model.closeProject()
                                } label: {
                                    Label("Projects", systemImage: "chevron.left")
                                }
                                // App round 5 spec item 8: the asymmetry of blocking only training
                                // was made consistent with rename's both-pattern — the
                                // authoritative lock (lockedTab) blocks align too.
                                .disabled(session.lockedTab != nil)
                                .help("Back to Projects (⇧⌘H)")
                            }
                            // The pencil next to the title (app round 4 spec item 6) — a
                            // .navigation placement puts it right beside the title. During training
                            // or align, moving the folder contends with the live paths, so it is
                            // disabled — the decision uses the authoritative lock lockedTab (FIX 3
                            // item 2: unified with the same gate as back navigation and the tabs).
                            ToolbarItem(placement: .navigation) {
                                Button {
                                    renameText = session.project.name
                                    isRenaming = true
                                } label: {
                                    Label("Rename Project", systemImage: "pencil")
                                }
                                .disabled(session.lockedTab != nil)
                                .help("Rename project")
                            }
                            ToolbarItem(placement: .principal) {
                                StageNav(tab: Binding(get: { session.tab }, set: { session.tab = $0 }),
                                         locked: session.lockedTab)   // app round 5 spec item 8
                            }
                        }
                        .navigationTitle(session.project.name)
                } else {
                    HomeView(model: model)
                }
            }
            .alert("Rename Project", isPresented: $isRenaming) {
                TextField("Name", text: $renameText)
                Button("Rename") { performRename() }
                Button("Cancel", role: .cancel) {}
            }
            .alert("Rename failed", isPresented: Binding(
                get: { renameError != nil }, set: { if !$0 { renameError = nil } })) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(renameError ?? "")
            }
            .frame(minWidth: 1100, minHeight: 720)
            .tint(SplatXTheme.accent)
            .preferredColorScheme(.dark)
            .splatFileOpenRouting()   // Finder .ply/.spz opens → an independent viewer window (app round 5 item 5)
        }
        .commands { AppCommands(registry: registry) }

        // The independent viewer window (app round 5 item 5): one window per .ply/.spz file opened
        // from Finder (keyed by the URL value).
        // Measured 2026-07-20 (FIX 1): cold open of 2 files at once → 1 Home + 2 viewers, warm
        // reopen → 0 new windows (value dedupe), warm open of 2 new files → +2 viewer windows only
        // — through the AppDelegate application(_:open:) bridge (SwiftUI onOpenURL alone delivers
        // only the first URL for a multi-file open, so only one window appears).
        // On a cold-launch file open it coexists with the Home window (the simpler choice — stated
        // in the Task 2 report).
        WindowGroup(id: ViewerWindow.id, for: URL.self) { $url in
            StandaloneViewerWindow(url: url)
        }
        .defaultSize(width: 900, height: 600)
    }

    /// Syncs the session on a successful store.rename (applyRename). If rename throws, **the folder
    /// is guaranteed not to have moved** (the ProjectStore.rename invariant — it throws only on a
    /// moveItem failure), so leaving the session on the old path is the consistent choice. A
    /// manifest write failure is handled non-fatally by the store (scan self-heals).
    private func performRename() {
        guard let session = model.session else { return }
        // A defensive re-check — independently of the button being disabled, the state can change
        // while the alert is up.
        // The intent is to prevent the folder move from contending with the live paths — the same
        // authoritative lock as the button (lockedTab).
        guard session.lockedTab == nil else { return }
        let name = renameText.trimmingCharacters(in: .whitespacesAndNewlines)
        // An empty name or the same name is a no-op — ProjectStore.rename appends a "-2" suffix on
        // a rename-to-self.
        guard !name.isEmpty, name != session.project.name else { return }
        do {
            let renamed = try model.store.rename(session.project, to: name)
            session.applyRename(project: renamed, directory: model.store.directory(for: renamed))
            model.console.append("✎ Project renamed → \(renamed.name)")
        } catch {
            renameError = error.localizedDescription
        }
    }
}
