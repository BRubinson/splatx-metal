import SwiftUI

/// The non-invasive overlay laid over the viewport: the top-left asset label, the top-right axis
/// gizmo, and the frame button.
///
/// The label and the gizmo are `.allowsHitTesting(false)`, so they do not intercept camera drag or
/// zoom events.
/// (Camera input is observed by the AppKit local event monitor, so this overlay is purely for
/// display.)
/// The only interactive element is the frame button.
///
/// The camera is received **as a model reference, not a value** (path transform FIX 9) — receiving
/// it as `camera: vm.camera` would put that read in the parent (`ViewerView.body`) and re-evaluate
/// the entire parent body on every orbit tick. The camera is read only inside `AxisGizmoView`,
/// which actually draws it (the AxisGizmoView comment is the authoritative rationale and
/// do-not-regress note). The label's title and subtitle change only at scene load, so they are
/// received as values, as before.
struct ViewportOverlay: View {
    let title: String
    let subtitle: String
    let viewer: ViewerModel
    let onFrame: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .top) {
                assetLabel
                    .allowsHitTesting(false)
                Spacer(minLength: 8)
                VStack(alignment: .trailing, spacing: 8) {
                    AxisGizmoView(viewer: viewer)
                        .allowsHitTesting(false)
                    frameButton
                }
            }
            Spacer(minLength: 0)
        }
        .padding(12)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    /// The top-left asset label: the source name plus the splat count. A blurred material
    /// background.
    private var assetLabel: some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(title.isEmpty ? "—" : title)
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(SplatXTheme.textPrimary)
                .lineLimit(1)
            Text(subtitle)
                .font(.system(size: 10))
                .foregroundStyle(SplatXTheme.textSecondary)
                .lineLimit(1)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 6))
    }

    /// The frame-to-fit button — the only interactive element.
    /// `.pointerStyle(.link)` (export-cinema-gif Task 8): applied for the same reason and with the
    /// same caveats as `TrainView.TimelapseResetButton` — see that header comment for the full
    /// rationale (why `.link`, why not `NSCursor.push/pop`, and the unverified MTKView-overlay
    /// hit-testing question). Both are the app's only interactive controls drawn over an `MTKView`,
    /// so they get the same treatment rather than letting one be an inconsistent outlier.
    private var frameButton: some View {
        Button(action: onFrame) {
            Image(systemName: "viewfinder")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(SplatXTheme.textPrimary)
                .frame(width: 28, height: 28)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 6))
        }
        .buttonStyle(.plain)
        .pointerStyle(.link)
        .help("Frame to fit")
    }
}
