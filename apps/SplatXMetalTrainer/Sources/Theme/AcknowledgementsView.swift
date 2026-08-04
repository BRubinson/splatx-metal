import SwiftUI

/// The open source notices sheet (audit cleanup Task 7 — the MIT distribution notice requirement).
///
/// Placement decision: this app has no Settings scene, and an About panel (Credits.rtf) would
/// create dual management in the form of an rtf conversion of the original text — a sheet opened
/// from BrandFooter, which is permanently present in both Home and the Inspector, satisfies both
/// discoverability and a single source of truth (the build-time assembled text).
///
/// The content is not transcribed by hand: it shows verbatim the bundle resource that the
/// postBuildScript "Assemble Acknowledgements.txt" assembles from the repo's authoritative NOTICE,
/// LICENSE, and ThirdParty/*/LICENSE — a structure that cannot diverge from the originals.
/// AcknowledgementsTests gates the resource's existence and its content markers.
struct AcknowledgementsView: View {
    @Environment(\.dismiss) private var dismiss

    /// Loads the bundle resource. Its absence is a build script regression — it degrades to an
    /// informational message for the user, while the tests (AcknowledgementsTests) catch the
    /// absence as a failure.
    static func loadText() -> String? {
        try? String(contentsOf: BundlePaths.acknowledgementsURL, encoding: .utf8)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Open Source Acknowledgements")
                    .font(.headline)
                Spacer()
                Button("Done") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
            .padding(12)
            SplatXTheme.hairline.frame(height: 1)
            ScrollView {
                Text(Self.loadText()
                     ?? "Acknowledgements resource missing from this build. See the NOTICE and ThirdParty license files in the source distribution.")
                    .font(.system(size: 11, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
            }
        }
        .frame(minWidth: 560, minHeight: 420)
        .background(SplatXTheme.panel)
    }
}
