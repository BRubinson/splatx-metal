import Foundation
import UserNotifications
import SplatXMetal
import os

// MARK: - Message assembly (pure functions — under test)

/// The wording of the task completion system notifications (app round 5 item 6). Four
/// notifications — training complete (distinguishing normal from partial), training failed, align
/// complete (pose and point counts), and align failed. Including the project name in the wording is
/// the contract.
/// Even the decision of whether to post (cancelled → nil, for example) is put into the pure
/// functions so it is testable — only AppNotifier (the posting wrapper) touches
/// UNUserNotificationCenter.
/// Short tasks such as video extraction and export produce no notification (avoiding notification
/// overload — spec).
enum AppNotificationText {
    struct Content: Equatable {
        let title: String
        let body: String
    }

    /// The training-finished notification. cancelled is an action the user took themselves, so it
    /// is excluded (returns nil).
    /// The title for a normal completion is the spec's contract string "Training complete", and a
    /// partial completion (thermal abort) uses a distinguishing phrase plus "partial result saved"
    /// to announce that the partial output was preserved.
    ///
    /// A **collapsed** run (2026-08-16 — a 15000-iteration run that ended with 0 splats and 3.02 dB
    /// was announced as "Training complete") gets its own title, and the wording contract is that it
    /// never contains the word "complete". The verdict is **derived here** from the three values
    /// this function already receives rather than passed in as a fourth argument: it is a pure
    /// function of exactly `outcome`/`splatCount`/`finalPSNR`, so deriving it makes it structurally
    /// impossible for a call site to forget the parameter and go back to announcing success — which
    /// is the entire failure this round exists to remove.
    static func trainingFinished(project: String,
                                 outcome: SplatXTrainer.RunResult.Outcome,
                                 splatCount: Int,
                                 finalPSNR: Double?) -> Content? {
        let stats = "\(splatCount) splats"
            + (finalPSNR.map { String(format: " · PSNR %.2f dB", $0) } ?? "")
        let health = RunHealth.evaluate(outcome: outcome, splatCount: splatCount,
                                        finalPSNR: finalPSNR)
        if health.isCollapsed {
            return Content(title: "Training produced no usable result",
                           body: "\(project) — \(health.shortReason). Open the Train tab for what "
                               + "to try.")
        }
        switch outcome {
        case .completed:
            return Content(title: "Training complete", body: "\(project) — \(stats)")
        case .thermalAborted:
            return Content(title: "Training stopped early (thermal)",
                           body: "\(project) — partial result saved · \(stats)")
        case .cancelled:
            return nil
        }
    }

    /// The shared frame for failure wording (FIX 3 item 4): it puts an error **summary** into the
    /// sentence. Previously the train call site passed "\(error)" (a raw enum dump — for example
    /// trainFailed("…")) straight through, which was asymmetric with align (a LocalizedError
    /// summary) — moving the summary derivation inside the assembly function structurally enforces
    /// symmetry by making it impossible for a call site to expose the raw string.
    static func trainingFailed(project: String, error: Error) -> Content {
        Content(title: "Training failed", body: "\(project) — \(failureSummary(error))")
    }

    static func alignFinished(project: String, poseCount: Int, pointCount: Int) -> Content {
        Content(title: "Align complete", body: "\(project) — \(poseCount) poses · \(pointCount) points")
    }

    static func alignFailed(project: String, error: Error) -> Content {
        Content(title: "Align failed", body: "\(project) — \(failureSummary(error))")
    }

    /// Video render complete (Cinema Task 5) — a minutes-long task, so it warrants a notification.
    /// A cancellation is an action the user took themselves and produces no notification (the same
    /// convention as trainingFinished's cancelled → nil — the caller, RenderJob, omits the post
    /// entirely in the cancelled branch).
    static func renderFinished(project: String, fileName: String, frames: Int,
                               seconds: Double) -> Content {
        Content(title: "Render complete",
                body: "\(project) — \(fileName) · \(frames) frames · "
                    + String(format: "%.1fs", seconds))
    }

    static func renderFailed(project: String, error: Error) -> Content {
        Content(title: "Render failed", body: "\(project) — \(failureSummary(error))")
    }

    /// The error summary convention (shared by train and align): ① for
    /// SplatXTrainerError.trainFailed the core's err_buf message (the payload) is authoritative as
    /// the human-readable sentence, so it is extracted directly — SplatXTrainerError does not adopt
    /// LocalizedError, and the core (Sources/) is off-limits for app work, so the summary is done
    /// on the app side. ② otherwise LocalizedError.errorDescription → ③ the String(describing:)
    /// fallback (the same string as align's previous "\(error)", so the align wording is
    /// unchanged).
    static func failureSummary(_ error: Error) -> String {
        if case SplatXTrainerError.trainFailed(let message) = error { return message }
        return (error as? LocalizedError)?.errorDescription ?? String(describing: error)
    }
}

// MARK: - The single-flight gate for the authorization request

/// A shared once-per-process execution gate (FIX 3 item 5). Only the first run() executes the
/// operation, and every run() during or after that execution discards its own operation and awaits
/// the **completion** of that execution.
/// The previous didRequestAuthorization boolean recorded only "the request was started", so a
/// later post arriving while requestAuthorization was in progress read the undecided
/// (notDetermined) settings and was silently lost — this type's contract is to force every post to
/// wait so it reads the settings after the request completes.
/// It is pure logic that does not touch UNUserNotificationCenter, so it is under test
/// (AppNotificationTextTests).
@MainActor
final class SingleFlightGate {
    private var flight: Task<Void, Never>?

    /// Only the first call executes the operation — every call returns only after that execution
    /// finishes.
    func run(_ operation: @escaping @MainActor () async -> Void) async {
        if flight == nil { flight = Task { await operation() } }
        await flight?.value
    }
}

// MARK: - The posting wrapper plus the foreground banner delegate

/// The UNUserNotificationCenter posting wrapper (app round 5 item 6).
///
/// - Authorization is requested **immediately before the first notification is posted** (HIG — at a
///   moment with context. Requesting up front at app start is forbidden).
///   If denied or undecided it is silently omitted with only an os log (nothing in the app console
///   — "silently" is the contract).
/// - The delegate is wired once at app start with install() — willPresent → .banner so the banner
///   appears even in the foreground. (The delegate property is weak, so the singleton retains
///   ownership.)
/// - Touching UNUserNotificationCenter in the test runner can crash depending on the bundle
///   context and has the side effect of an authorization dialog — the tests verify only
///   AppNotificationText (the pure functions) and never call this class.
@MainActor
final class AppNotifier: NSObject, UNUserNotificationCenterDelegate {
    static let shared = AppNotifier()
    private static let log = Logger(subsystem: "com.postmedia.splatx-metal-trainer",
                                    category: "notifications")
    /// The single-flight gate for the authorization request (FIX 3 item 5) — the first post starts
    /// the request, and a later post arriving while the request is in progress also waits for
    /// completion before reading the settings (the previous boolean flag let arrivals during the
    /// request read the undecided settings and be lost).
    private let authorizationGate = SingleFlightGate()

    /// Wires the delegate — once at app start (AppDelegate). It does not request authorization.
    func install() {
        UNUserNotificationCenter.current().delegate = self
    }

    /// A no-op when content is nil (a cancellation, for example). Posting is asynchronous and every
    /// failure is non-fatal (logged only) — notifications are an observation convenience, not a
    /// workflow element.
    func post(_ content: AppNotificationText.Content?) {
        guard let content else { return }
        Task {   // inherits @MainActor — access to authorizationGate is serialized
            let center = UNUserNotificationCenter.current()
            // Authorization is requested once, immediately before the first notification is posted
            // (HIG). The gate forces the wait for completion, so no post ever reads the settings
            // while the request is undecided (notDetermined) — the contract of silently omitting
            // (logging only) on denial is still held by the guard below.
            await authorizationGate.run {
                do {
                    _ = try await center.requestAuthorization(options: [.alert, .sound])
                } catch {
                    Self.log.warning("authorization request failed: \(error.localizedDescription)")
                }
            }
            let settings = await center.notificationSettings()
            guard settings.authorizationStatus == .authorized
                || settings.authorizationStatus == .provisional else {
                Self.log.info("notification skipped (authorization status \(settings.authorizationStatus.rawValue))")
                return
            }
            let un = UNMutableNotificationContent()
            un.title = content.title
            un.body = content.body
            un.sound = .default
            do {
                try await center.add(UNNotificationRequest(
                    identifier: UUID().uuidString, content: un, trigger: nil))
            } catch {
                Self.log.warning("notification delivery failed: \(error.localizedDescription)")
            }
        }
    }

    // MARK: UNUserNotificationCenterDelegate

    /// Shows the banner even when the app is active (foreground) — spec item 6: willPresent →
    /// .banner (plus sound).
    nonisolated func userNotificationCenter(_ center: UNUserNotificationCenter,
                                            willPresent notification: UNNotification) async
        -> UNNotificationPresentationOptions {
        [.banner, .sound]
    }
}
