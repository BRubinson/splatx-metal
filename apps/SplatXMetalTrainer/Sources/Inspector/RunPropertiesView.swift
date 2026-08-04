import SwiftUI

/// The body of the View tab's Inspector: the properties of the run open in the viewer.
struct RunPropertiesView: View {
    let record: RunRecord?
    /// The number of splats the viewer actually loaded.
    let loadedSplats: Int?

    var body: some View {
        if let record {
            VStack(alignment: .leading, spacing: 8) {
                PanelSection(title: "TRAINING") {
                    KeyValueRow(key: "run", value: record.id)
                    KeyValueRow(key: "options", value: record.optionsSummary)
                    KeyValueRow(key: "outcome", value: record.outcome)
                }
                PanelSection(title: "RESULT") {
                    KeyValueRow(key: "splats", value: "\(record.splatCount.formatted())")
                    KeyValueRow(key: "loaded", value: loadedSplats.map { $0.formatted() } ?? "—")
                    KeyValueRow(key: "PSNR", value: record.finalPSNR.map {
                        String(format: "%.2f dB", $0) } ?? "—")
                }
            }
        } else {
            Text("Train a run to view its result")
                .font(.callout)
                .foregroundStyle(SplatXTheme.textSecondary)
        }
    }
}
