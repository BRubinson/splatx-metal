import SwiftUI
import SplatXMetal

struct OptionsForm: View {
    @Binding var draft: TrainOptionsDraft
    @State private var showAdvanced = false

    var body: some View {
        Form {
            Section("Core") {
                Picker("factor", selection: $draft.factor) { ForEach([1,2,4,8], id: \.self) { Text("\($0)").tag($0) } }
                LabeledContent("iterations") {
                    HStack {
                        TextField("", value: $draft.iterations, format: .number)
                            .frame(width: 90)
                            .multilineTextAlignment(.trailing)
                            .textFieldStyle(.roundedBorder)
                        Stepper("", value: $draft.iterations, in: 1...200_000, step: 100).labelsHidden()
                    }
                }
                Toggle("MCMC", isOn: $draft.mcmc)
                Stepper("cap-max: \(draft.capMax)", value: $draft.capMax, in: 1...4_000_000, step: 50_000)
                Picker("SH degree", selection: $draft.shDegree) { ForEach(0...3, id: \.self) { Text("\($0)").tag($0) } }
                HStack { Text("SSIM \(draft.ssimWeight, specifier: "%.2f")"); Slider(value: $draft.ssimWeight, in: 0...1) }
            }
            // Round 4: in a DisclosureGroup only the chevron and the label respond to the toggle →
            // it was rebuilt as a custom header where the whole row is the tap surface (a Button +
            // a full-width HStack + contentShape) (the contract: pressing anywhere on the row
            // toggles it, and the visuals stay at the previous level). So that there is no double
            // firing with the native label toggle, this button owns the toggle entirely instead of
            // the DisclosureGroup.
            Section {
                Button {
                    withAnimation { showAdvanced.toggle() }
                } label: {
                    HStack {
                        Text("Advanced")
                        Spacer()
                        Image(systemName: "chevron.right")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .rotationEffect(.degrees(showAdvanced ? 90 : 0))
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                if showAdvanced {
                    Toggle("antialiased", isOn: $draft.antialiased)
                    Toggle("resident", isOn: $draft.resident)
                    Picker("thermal", selection: $draft.thermalMode) {
                        Text("performance").tag(SplatXTrainer.ThermalMode.performance)
                        Text("balanced").tag(SplatXTrainer.ThermalMode.balanced)
                        Text("lowheat").tag(SplatXTrainer.ThermalMode.lowheat)
                    }
                    Stepper("test-split: \(draft.testSplit)", value: $draft.testSplit, in: 0...64)
                    Stepper("eval-every: \(draft.evalEvery)", value: $draft.evalEvery, in: 0...50_000, step: 100)
                    Toggle("auto-cap", isOn: $draft.autoCap)
                    Stepper("max-points: \(draft.maxPoints)", value: $draft.maxPoints, in: 1...2_000_000, step: 10_000)
                    Stepper("log-every: \(draft.logEvery)", value: $draft.logEvery, in: 1...5000, step: 10)
                    Toggle("Save checkpoint", isOn: $draft.saveCheckpoint)
                    if draft.saveCheckpoint {
                        Stepper("checkpoint-every: \(draft.checkpointEvery)", value: $draft.checkpointEvery, in: 0...50_000, step: 500)
                    }
                }
            }
            if !draft.isValid {
                Section {
                    ForEach(Array(draft.validationErrors().enumerated()), id: \.offset) { _, msg in
                        Label(msg, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
                    }
                }
            }
        }
        .formStyle(.grouped)
    }
}
