import SwiftUI
import SplatXMetal

struct OptionsForm: View {
    @Binding var draft: TrainOptionsDraft
    @State private var showAdvanced = false
    // Task 7 (2026-08-05 spec §3.2): the timelapse controls used to live at the tail of Advanced;
    // they now get their own Section so a user who only cares about the timelapse doesn't have to
    // open the whole Advanced block to find it.
    @State private var showTimelapse: Bool

    // A custom init is required to seed showTimelapse from the draft (see below) — that forfeits
    // the implicit memberwise init SwiftUI would otherwise synthesize for the @Binding, so it is
    // wired by hand here.
    init(draft: Binding<TrainOptionsDraft>) {
        _draft = draft
        // Seed the disclosure open when the project already has timelapse.enabled == true (spec
        // §3.3). @State does not persist — this app has zero @AppStorage — so a plain `= false`
        // default would mean: a project with a saved timelapse.enabled==true (ProjectSession.swift
        // seeds TrainOptionsDraft from project.json) opens this form collapsed and shows no trace
        // that every run from here on is capturing frames and encoding a file. The user would only
        // find out by noticing the RESULT row after a run finishes, if they notice it at all.
        _showTimelapse = State(initialValue: draft.wrappedValue.timelapse.enabled)
    }

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
            // Timelapse (2026-08-04 spec §3.6, moved to its own Section by 2026-08-05 spec §3.2).
            // Same custom-header contract as Advanced above, deliberately duplicated rather than
            // factored into a shared "collapsible Section" helper that both drive — this is the
            // one and only place in this form where two independent expand states coexist, and
            // making that abstract now would just be speculative. Not a DisclosureGroup for the
            // same reason as Advanced: in a DisclosureGroup only the chevron/label are the tap
            // surface, and having one group whose contract differs from its neighbor is exactly
            // the shape that produced the round-4 double-fire bug (OptionsForm.swift:44-49).
            Section {
                Button {
                    withAnimation { showTimelapse.toggle() }
                } label: {
                    HStack {
                        Text("Timelapse")
                        Spacer()
                        Image(systemName: "chevron.right")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .rotationEffect(.degrees(showTimelapse ? 90 : 0))
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                if showTimelapse {
                    // The nested-options-under-a-toggle shape follows the saveCheckpoint
                    // precedent above. The period Stepper is fixed to steps of 50 because the
                    // live snapshot period is 50 and a capture is only possible right after a
                    // swap — a value that is not a multiple of 50 would snap to that grid and
                    // mean a different interval than the one shown (spec §3.7).
                    Toggle("Timelapse", isOn: $draft.timelapse.enabled)
                    if draft.timelapse.enabled {
                        Stepper("capture every: \(draft.timelapse.everySteps)",
                                value: $draft.timelapse.everySteps,
                                in: TimelapseSettings.everyStepsRange,
                                step: TimelapseSettings.everyStepsStep)
                        Picker("format", selection: $draft.timelapse.format) {
                            ForEach(TimelapseFormat.allCases) { Text($0.displayName).tag($0) }
                        }
                        Picker("fps", selection: $draft.timelapse.fps) {
                            ForEach(TimelapseSettings.fpsChoices, id: \.self) { Text("\($0)").tag($0) }
                        }
                        Picker("resolution", selection: $draft.timelapse.shortSide) {
                            ForEach(TimelapseSettings.shortSideChoices, id: \.self) {
                                Text("\($0)p").tag($0)
                            }
                        }
                        // 2026-08-05 spec §2.1/§2.5: which aspect ratio the frame takes, not the
                        // projection — CameraController.perspective and its fovY are untouched
                        // (plan Global Constraints #4). Display names come from
                        // TimelapseAspectSource.displayName, the single source of truth, so this
                        // view never invents its own strings for the choices.
                        Picker("aspect", selection: $draft.timelapse.aspectSource) {
                            ForEach(TimelapseSettings.aspectSourceChoices) { Text($0.displayName).tag($0) }
                        }
                        Toggle("overlay label", isOn: $draft.timelapse.overlay)
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
