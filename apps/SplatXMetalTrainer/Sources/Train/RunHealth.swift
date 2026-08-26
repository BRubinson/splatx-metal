import Foundation
import SplatXMetal

/// Whether a finished training run actually produced something, judged from the three numbers the
/// core hands back (`RunResult.outcome` / `.splatCount` / `.finalPSNR`).
///
/// **Why this exists.** A real run ended like this on 2026-08-16:
///
///     factor 2 · 15000it · MCMC · cap 1000k · SH3
///     → splatCount 0, finalPSNR 3.02 dB, outcome "completed", output.ply = "element vertex 0"
///
/// Every gaussian had fallen below the export filter's `sigmoid(opacity) > 0.005` cut, so the PLY
/// was written with zero vertices — and the app said **"Training complete"**, posted a success
/// notification, filed a `completed` run record, and switched to the viewer. The user waited
/// minutes for an empty file and was told it worked. The core is not wrong here: `completed` means
/// "the loop reached run_until", which it did. Judging whether the *result* is usable is the app's
/// job, and this type is that judgement, pulled out of the view model so its boundaries are
/// testable (the same split as `TrainViewModel.viewportSizeUpdate` and
/// `TrainViewModel.timelapseFailureNote`).
///
/// **Scope: `.completed` only.** `.cancelled` (the user pressed Stop), `.thermalAborted` (CLI exit
/// 2) and the external stop (exit 3) are documented as *normal terminations, not errors*
/// (README "Exit codes"), and they already announce themselves honestly — "Training stopped early (thermal)
/// · partial result saved". Re-labelling one of those as a collapse would turn a truthful message
/// into a scarier one and would blur the one distinction that matters: whether the run was cut
/// short (retry with a smaller budget) or ran to the end and still produced nothing (something is
/// wrong with the data or the settings). So `evaluate` answers `.ok` for them unconditionally and
/// their existing wording is left exactly as it was.
///
/// **What is deliberately NOT a collapse signal: a low splat count.** A small object, a short run,
/// or a tight cap can legitimately finish with a few thousand splats — the app has no idea what
/// "enough" is for a given scene, and a threshold plucked out of the air would start calling
/// good runs broken. Only *zero* is judged, because zero is not a small result, it is the absence
/// of a file's worth of content.
enum RunHealth: Equatable {
    /// Nothing detectably wrong — the normal path, and the answer for every non-`.completed`
    /// outcome.
    case ok
    /// The export kept no gaussians at all (`splatCount == 0`). The output file has a
    /// `element vertex 0` header and there is nothing to view or export.
    case noSplats
    /// The final eval render is *worse than a blank gray image* — see `flatGrayFloorDB`.
    case degenerateRender(psnrDB: Double)

    var isCollapsed: Bool { self != .ok }

    /// The PSNR below which a render is provably degenerate, in dB.
    ///
    /// The core computes `psnr = 10·log10(1/mse)` over `[0,1]`-clamped renders
    /// (`Trainer.cpp:359`, and `splatx_train_c.h:53` — "the render is clamped to [0,1] before
    /// psnr"), so the data range is exactly 1. Now take the most useless renderer imaginable: one
    /// that ignores the scene and paints every pixel mid-gray. Its per-pixel error is at most 0.5
    /// whatever the target is, so `mse ≤ 0.25` and `psnr ≥ 10·log10(4) = 6.0206 dB` — a bound that
    /// holds for *any* image content, bright or dark, and is quoted verbatim by the core's own hand
    /// check (`TrainerTests.cpp:1271`).
    ///
    /// A model scoring below that is therefore worse than a constant, which no amount of "this is a
    /// hard scene" explains — it means the render does not correspond to the photographs at all.
    /// That is why this is a safe test and a splat-count threshold is not: the floor is derived, not
    /// tuned, and no legitimately trained model can fall under it.
    ///
    /// The reported figure is the arithmetic mean of the per-view dB values, not the pooled PSNR,
    /// and the bound survives that: `mean(-10·log10(mse_i)) ≥ -10·log10(mean(mse_i))` by Jensen, so
    /// a mean under the floor implies the pooled MSE is over 0.25 too.
    ///
    /// 6.0 rather than 6.0206 keeps float rounding on the safe side — the boundary itself is never
    /// what trips the check.
    static let flatGrayFloorDB = 6.0

    /// The verdict for a finished run.
    ///
    /// `finalPSNR` is `nil` whenever no final eval ran (`testSplit == 0`, or the run did not reach
    /// the end) — the C boundary already maps the core's NaN to nil (`SplatXTrainer.swift:574`), so
    /// nil here means "no signal", never "bad signal", and the count is then the only evidence. A
    /// non-finite value cannot arrive but is treated as no signal as well rather than formatted into
    /// a message.
    ///
    /// `.noSplats` is checked first: with nothing left to render the PSNR is a consequence rather
    /// than an independent symptom, and "the export kept none" is the more concrete sentence.
    static func evaluate(outcome: SplatXTrainer.RunResult.Outcome,
                         splatCount: Int,
                         finalPSNR: Double?) -> RunHealth {
        guard outcome == .completed else { return .ok }
        if splatCount <= 0 { return .noSplats }
        guard let psnr = finalPSNR, psnr.isFinite else { return .ok }
        if psnr < flatGrayFloorDB { return .degenerateRender(psnrDB: psnr) }
        return .ok
    }

    // MARK: - Wording

    /// One sentence naming what went wrong, in the user's terms. Empty for `.ok` (callers branch on
    /// `isCollapsed` and never print it).
    var headline: String {
        switch self {
        case .ok:
            return ""
        case .noSplats:
            return "Training collapsed — every gaussian faded out and the export kept none, "
                + "so the output file is empty."
        case .degenerateRender(let db):
            return String(format: "Training collapsed — the final render scores %.2f dB, "
                          + "worse than a blank gray image.", db)
        }
    }

    /// The short clause for a notification body, where a full sentence plus advice does not fit.
    var shortReason: String {
        switch self {
        case .ok: return ""
        case .noSplats: return "no splats survived — the output file is empty"
        case .degenerateRender(let db):
            return String(format: "the render never matched the photos (%.2f dB)", db)
        }
    }

    /// What to try next. One string for both collapse causes on purpose: they share a root cause
    /// list, and both mean "this dataset/settings pair produced nothing", so splitting the advice
    /// would only invite two half-maintained copies.
    ///
    /// Ordered by how often it is the answer. Alignment leads because a pose/image mismatch is the
    /// one failure that makes the photometric loss unreachable no matter how long the run goes —
    /// the optimiser's cheapest move is then to make every gaussian transparent, which is precisely
    /// the observed ending. The short probe run is second because it converts a "wait 20 minutes to
    /// find out" into a "wait 1 minute to find out". Re-capture is last: it is the expensive answer,
    /// and only worth reaching for once the cheap two are ruled out.
    static let advice =
        "What to try: re-run Align and check the poses match these images (a pose/image mismatch is "
        + "the usual cause); then probe the scene with a short run — a few thousand iterations and a "
        + "smaller splat budget — before committing to a long one. If the photos are blurry, too few, "
        + "or barely move around the subject, re-capture rather than re-train."

    /// The inline banner text on the Train tab: what happened, then what to do about it.
    var message: String { headline + " " + Self.advice }

    // MARK: - What gets written down

    /// The string stored in `RunRecord.outcome`.
    ///
    /// The vocabulary was the three `SplatXTrainer.RunResult.Outcome` raw values; this adds a fourth
    /// that the core does not have. It is minted **here, in the app**, and deliberately not added to
    /// `RunResult.Outcome` — that enum lives in `Sources/SplatXMetal` (the shipped core surface),
    /// where a new case would be a public API change made for one app's presentation.
    ///
    /// It stays camelCase like its neighbours because two readers print it raw: the RUNS row badge
    /// (`InspectorView.swift:337`, which shows any value other than "completed" in orange, so the
    /// badge appears with no change there) and the inspector's `outcome` row
    /// (`RunPropertiesView.swift:70`).
    ///
    /// Forward/backward compatibility: `RunRecord.outcome` is decoded as a plain `String`, so an
    /// older build opening a manifest containing "collapsed" shows the badge rather than failing to
    /// decode, and a newer build opening an older manifest sees the values it always saw.
    static func recordOutcome(outcome: SplatXTrainer.RunResult.Outcome,
                              splatCount: Int,
                              finalPSNR: Double?) -> String {
        let health = evaluate(outcome: outcome, splatCount: splatCount, finalPSNR: finalPSNR)
        return health.isCollapsed ? collapsedOutcome : outcome.rawValue
    }

    static let collapsedOutcome = "collapsed"

    /// The reader-side counterpart of `recordOutcome`, for the two views that render a stored
    /// record (`InspectorView`'s RUNS badge and `RunPropertiesView`). It exists so the literal
    /// "collapsed" is written down in exactly one place — a view comparing the raw string itself
    /// would silently stop matching the day the spelling changes.
    static func isCollapsed(recordOutcome raw: String) -> Bool { raw == collapsedOutcome }

    /// The explanation attached to a **stored** collapsed run, reopened from the RUNS history long
    /// after the banner that first reported it is gone.
    ///
    /// Cause-agnostic on purpose: the manifest records the verdict, not which of the two tests
    /// fired, and inventing a specific sentence from a value that does not carry that information
    /// would be a guess presented as a fact. The advice is what the user came for anyway, and it is
    /// the same paragraph either way.
    static let storedCollapseNote = "This run produced no usable result. " + advice

    /// The console lines for a finished run — the whole "what happened" report, both branches.
    ///
    /// Assembled here rather than inline at the call site so the contract "a collapsed run is never
    /// announced with the success wording" is a thing a test can hold, instead of a line of string
    /// interpolation inside a `MainActor.run` block that needs a dataset and a GPU to reach.
    ///
    /// The collapse branch is two lines: the console is collapsed by default
    /// (`AppModel.consoleExpanded == false`), so this is a record for whoever opens it later rather
    /// than the primary channel — the primary channel is the inline banner and the notification.
    static func completionConsoleLines(outcome: SplatXTrainer.RunResult.Outcome,
                                       splatCount: Int,
                                       finalPSNR: Double?) -> [String] {
        let stats = "N \(splatCount)"
            + (finalPSNR.map { String(format: " · PSNR %.2f dB", $0) } ?? "")
        let health = evaluate(outcome: outcome, splatCount: splatCount, finalPSNR: finalPSNR)
        guard health.isCollapsed else {
            return ["■ Finished (\(outcome.rawValue)): \(stats)"]
        }
        return ["✗ \(health.headline) (\(stats))", "  \(advice)"]
    }
}
