import Foundation

/// The **sole owner** of the activity token that prevents system idle sleep during long-running
/// tasks (training, align, and video export) (audit cleanup Task 6 — appstore-audit §4: the symptom
/// where, without beginActivity, system sleep during training halted the training).
///
/// The ownership rule: the token's begin/end happen only inside this class. Callers merely declare
/// the state they want (a reason string, or nil), and update converges idempotently on the
/// difference from the current state — avoiding "a lifetime invariant guarded piecemeal in several
/// places breaks with every piece added" (the lesson of the splash's two-stage lifetime), so that
/// on any termination path (normal completion, cancellation, thermal abort, or error), the release
/// converges on the single update(reason: nil) the moment lockedTab returns to nil.
///
/// Why these options: it uses only `.userInitiated` (= idleSystemSleepDisabled plus
/// suddenTermination/automaticTerminationDisabled). **Display sleep is deliberately not
/// prevented** — training, align, and export are GPU/CPU work that continues with the screen off
/// (preventing idle system sleep alone is sufficient), and keeping the display permanently lit is a
/// pure cost for training on laptop battery (a 30–90 minute Ultra/30K run). It runs to completion
/// with the screen off, and a notification (AppNotifier) delivers the result.
/// (Lid-close sleep is an OS policy this API cannot prevent — out of scope.)
@MainActor final class SleepGuard {
    /// begin/end injection — tests verify the transition counts with stubs (the pure-logic TDD
    /// idiom).
    private let begin: (String) -> NSObjectProtocol
    private let end: (NSObjectProtocol) -> Void

    private var token: NSObjectProtocol?
    private(set) var currentReason: String?

    init(begin: @escaping (String) -> NSObjectProtocol = {
             ProcessInfo.processInfo.beginActivity(options: .userInitiated, reason: $0)
         },
         end: @escaping (NSObjectProtocol) -> Void = {
             ProcessInfo.processInfo.endActivity($0)
         }) {
        self.begin = begin
        self.end = end
    }

    /// Declares the state you want. nil = no activity needed. Redeclaring the same reason is a
    /// no-op, and a reason change ends then begins (preserving the one-token invariant).
    func update(reason: String?) {
        guard reason != currentReason else { return }
        if let token { end(token); self.token = nil }
        currentReason = reason
        if let reason { token = begin(reason) }
    }

    /// A line of defense — on the normal path lockedTab returns to nil and update(nil) releases it,
    /// but this keeps the token from leaking even if a path arises where session teardown comes
    /// first.
    deinit {
        if let token { end(token) }
    }

    /// lockedTab (the authoritative gate of the long-running tasks) → the activity reason. A pure
    /// function — under test.
    /// lockedTab returns only train/align/cinema, so anything else (.view or nil) needs no
    /// activity.
    static func activityReason(lockedTab: ProjectTab?) -> String? {
        switch lockedTab {
        case .train: "SplatX Metal Trainer: training in progress"
        case .align: "SplatX Metal Trainer: alignment in progress"
        case .cinema: "SplatX Metal Trainer: video export in progress"
        case .view, nil: nil
        }
    }
}
