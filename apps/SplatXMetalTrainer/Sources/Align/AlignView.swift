import SwiftUI

/// The central view of the align tab: three states — a not-yet-aligned notice, in progress, and a
/// completion summary.
struct AlignView: View {
    @Bindable var session: ProjectSession
    @Bindable var model: AppModel

    var body: some View {
        let vm = session.alignVM
        VStack(spacing: 16) {
            if vm.isRunning {
                ProgressView(value: vm.fraction) {
                    Text(String(format: "Aligning… %.0f%%", vm.fraction * 100))
                        .font(.callout).monospacedDigit()
                        .foregroundStyle(SplatXTheme.textPrimary)
                }
                .frame(maxWidth: 360)
                // The Cancel button was removed (applying discretion under app round 4 spec
                // item 3) — stopping is unified into the single Start↔Stop switching position of
                // the Inspector CTA (the same pattern as train, removing the duplicate).
            } else if let summary = session.project.alignSummary {
                VStack(spacing: 8) {
                    Image(systemName: "checkmark.circle")
                        .font(.system(size: 40, weight: .light))
                        .foregroundStyle(SplatXTheme.accent)
                    Text("Aligned").font(.title3).foregroundStyle(SplatXTheme.textPrimary)
                    Text("\(summary.poseCount) poses · \(summary.pointCount.formatted()) points · \(Int(summary.alignSeconds))s")
                        .font(.callout).monospacedDigit()
                        .foregroundStyle(SplatXTheme.textSecondary)
                    // 2026-08-04 spec §1.5 — the applied ground-level correction, formatted by
                    // the same single source of truth as the Inspector's ALIGNMENT row. Absent
                    // whenever nothing was rotated.
                    if let level = InspectorView.levelSummaryValue(summary.levelTiltDegrees) {
                        Text("Ground leveled by \(level)")
                            .font(.callout).monospacedDigit()
                            .foregroundStyle(SplatXTheme.textSecondary)
                    }
                    if !session.project.runs.isEmpty {
                        Text("Re-aligning invalidates existing runs' pose basis.")
                            .font(.caption).foregroundStyle(.orange)
                    }
                }
            } else if let notice = Self.existingAlignmentNotice(
                sourceKind: session.project.sourceKind, state: session.project.state) {
                // A pre-existing alignment with no align history of our own this session —
                // either an imported COLMAP sparse/0, or a capture bundle's on-device poses.
                ContentUnavailableView(notice.title, systemImage: notice.systemImage,
                                       description: Text(notice.description))
            } else {
                ContentUnavailableView("Not aligned yet",
                                       systemImage: "camera.metering.matrix",
                                       description: Text("Run Align to estimate camera poses and a seed point cloud."))
            }
            if let e = vm.errorMessage {
                Label(e, systemImage: "xmark.octagon").foregroundStyle(.red).font(.callout)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    /// The text/icon for the "there is already an alignment, and we did not produce it this
    /// session" notice — `nil` when `state == .imported` (nothing to notice yet; the view falls
    /// through to "Not aligned yet"). Pulled out to a pure `static func` (the
    /// `startAlignDisabledReason`/`renderDisabledReason` idiom in this repo) so the wording is
    /// pinned by a test instead of only being checkable by eye. Tests: AlignReAlignNoticeTests.
    ///
    /// `.capture` gets a different message than everything else. The generic COLMAP wording
    /// ("You can re-align from images if needed") is accurate for an imported `sparse/0` — SfM
    /// is cheap to redo and nothing is lost. It undersells a capture bundle: its poses came from
    /// the device's own LiDAR at capture time, and the README that ships in the bundle says so
    /// explicitly — re-running SfM on the images "is unnecessary and would discard the captured
    /// geometry" (spec 2026-08-06 §5). That geometry cannot be re-measured after the fact (the
    /// LiDAR pass only happened once, while the phone was there), so the capture case says what
    /// would specifically be lost rather than reusing the neutral "if needed" framing — informative,
    /// not alarming, per the round's brief.
    static func existingAlignmentNotice(sourceKind: ProjectSourceKind, state: ProjectState) -> ReAlignNotice? {
        guard state != .imported else { return nil }
        switch sourceKind {
        case .capture:
            return ReAlignNotice(
                title: "Captured LiDAR alignment in use",
                systemImage: "cube.transparent",
                description: "This project's camera poses and point cloud were captured on-device"
                    + " by LiDAR, not solved from images. Re-aligning replaces them with"
                    + " images-only SfM output, and the captured geometry cannot be recovered.")
        case .images, .video, .colmap:
            return ReAlignNotice(
                title: "Imported COLMAP alignment in use",
                systemImage: "point.3.connected.trianglepath.dotted",
                description: "This project came with sparse/0. You can re-align from images if needed.")
        }
    }

    /// Payload for `existingAlignmentNotice` — a plain `Equatable` struct rather than a tuple so
    /// tests can `XCTAssertEqual` it directly (bare tuples do not conform to `Equatable`).
    struct ReAlignNotice: Equatable {
        let title: String
        let systemImage: String
        let description: String
    }
}
