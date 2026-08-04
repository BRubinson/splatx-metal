enum TrainPreset: String, CaseIterable, Identifiable {
    case fast, balanced, quality, ultra, custom
    var id: String { rawValue }
    var label: String {
        switch self {
        case .fast: "Fast"; case .balanced: "Balanced"; case .quality: "Quality"
        case .ultra: "Ultra"; case .custom: "Custom"
        }
    }
    /// (iterations, factor, capMax) — nil for custom.
    var values: (iterations: Int, factor: Int, capMax: Int)? {
        switch self {
        case .fast: (3_000, 4, 200_000)
        case .balanced: (7_000, 4, 500_000)
        case .quality: (15_000, 2, 1_000_000)
        case .ultra: (30_000, 1, 2_000_000)
        case .custom: nil
        }
    }
    func apply(to draft: inout TrainOptionsDraft) {
        guard let v = values else { return }
        draft.iterations = v.iterations; draft.factor = v.factor; draft.capMax = v.capMax
    }
    static func matching(_ draft: TrainOptionsDraft) -> TrainPreset {
        for preset in [TrainPreset.fast, .balanced, .quality, .ultra] {
            if let v = preset.values, v.iterations == draft.iterations,
               v.factor == draft.factor, v.capMax == draft.capMax { return preset }
        }
        return .custom
    }
}
