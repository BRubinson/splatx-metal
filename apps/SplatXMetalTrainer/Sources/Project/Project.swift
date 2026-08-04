import Foundation

enum ProjectSourceKind: String, Codable { case images, video, colmap }
enum ProjectState: String, Codable { case imported, aligned, trained }

struct AlignSummary: Codable, Equatable {
    var poseCount: Int
    var pointCount: Int
    var alignSeconds: Double
}

/// The output of one training run. id = the subfolder name under runs/ (a yyyyMMdd-HHmmss stamp).
/// File paths are recorded only relative to the project directory
/// ("runs/<id>/<outputFile>"), making them immune to the project folder being moved or renamed.
struct RunRecord: Codable, Equatable, Identifiable {
    var id: String
    var outputFile: String        // new runs are fixed to "output.ply" (round 4) — the "output.spz" of older manifests remains readable
    var previewFile: String?      // "preview.png"
    var outcome: String           // SplatXTrainer.RunResult.Outcome.rawValue
    var finalPSNR: Double?
    var splatCount: Int
    var optionsSummary: String
    var createdAt: Date
    /// The antialiased setting used at training time (Cinema Task 5). A standard PLY does not store
    /// an AA flag, so for sx_render_session to render consistently with training the caller must
    /// pass this record (the splatx_train_c.h contract). false when absent from an older manifest —
    /// the same as the past training default.
    var antialiased: Bool

    init(id: String, outputFile: String, previewFile: String?, outcome: String,
         finalPSNR: Double?, splatCount: Int, optionsSummary: String, createdAt: Date,
         antialiased: Bool = false) {
        self.id = id; self.outputFile = outputFile; self.previewFile = previewFile
        self.outcome = outcome; self.finalPSNR = finalPSNR; self.splatCount = splatCount
        self.optionsSummary = optionsSummary; self.createdAt = createdAt
        self.antialiased = antialiased
    }

    enum CodingKeys: String, CodingKey {
        case id, outputFile, previewFile, outcome, finalPSNR, splatCount, optionsSummary
        case createdAt, antialiased
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        outputFile = try c.decode(String.self, forKey: .outputFile)
        previewFile = try c.decodeIfPresent(String.self, forKey: .previewFile)
        outcome = try c.decode(String.self, forKey: .outcome)
        finalPSNR = try c.decodeIfPresent(Double.self, forKey: .finalPSNR)
        splatCount = try c.decode(Int.self, forKey: .splatCount)
        optionsSummary = try c.decode(String.self, forKey: .optionsSummary)
        createdAt = try c.decode(Date.self, forKey: .createdAt)
        antialiased = try c.decodeIfPresent(Bool.self, forKey: .antialiased) ?? false
    }
}

/// The root of the project.json manifest. It does not store the directory path — the file system
/// (ProjectStore) is authoritative for the folder's location and name.
struct Project: Codable, Equatable, Identifiable {
    var id: UUID
    var name: String
    var sourceKind: ProjectSourceKind
    var state: ProjectState
    var imageCount: Int
    var alignSummary: AlignSummary?
    var runs: [RunRecord]
    var createdAt: Date
    /// The Cinema camera paths (2026-07-26 spec — several at project level; they are in scene
    /// coordinates, so they are reused across runs).
    /// [] when absent from an older manifest (the decodeIfPresent forward-compatibility
    /// convention).
    var cameraPaths: [CameraPath]
    /// The Cinema static image overlay (Task 4) — **one per project**, not per path (shared by
    /// every export, spec data model). nil when absent from an older manifest.
    var overlay: OverlaySpec?
    /// The Cinema output aspect ratio (Task 5 — a Task 4 review recommendation): a project-level
    /// setting shared by the overlay editing WYSIWYG and the export. An older manifest or an
    /// unknown string gives nil (a lenient decode) — CinemaViewModel defaults to .wide.
    var outputAspect: OutputAspect?
    /// Whether the default brand watermark seed is complete (Cinema FIX 5 — spec footnote 4).
    /// overlay == nil alone cannot distinguish "not seeded yet" from "the user deleted it", so a
    /// separate flag is authoritative for respecting the removal.
    /// false when absent from an older manifest (the decodeIfPresent forward-compatibility
    /// convention) → it attempts to seed on the first open.
    var cinemaWatermarkSeeded: Bool

    var latestRun: RunRecord? { runs.max(by: { $0.createdAt < $1.createdAt }) }

    enum CodingKeys: String, CodingKey {
        case id, name, sourceKind, state, imageCount, alignSummary, runs, createdAt
        case cameraPaths, overlay, outputAspect, cinemaWatermarkSeeded
    }

    init(id: UUID, name: String, sourceKind: ProjectSourceKind, state: ProjectState,
         imageCount: Int, alignSummary: AlignSummary?, runs: [RunRecord], createdAt: Date,
         cameraPaths: [CameraPath] = [], overlay: OverlaySpec? = nil,
         outputAspect: OutputAspect? = nil, cinemaWatermarkSeeded: Bool = false) {
        self.id = id; self.name = name; self.sourceKind = sourceKind; self.state = state
        self.imageCount = imageCount; self.alignSummary = alignSummary
        self.runs = runs; self.createdAt = createdAt
        self.cameraPaths = cameraPaths
        self.overlay = overlay
        self.outputAspect = outputAspect
        self.cinemaWatermarkSeeded = cinemaWatermarkSeeded
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        name = try c.decode(String.self, forKey: .name)
        sourceKind = try c.decode(ProjectSourceKind.self, forKey: .sourceKind)
        state = try c.decode(ProjectState.self, forKey: .state)
        imageCount = try c.decode(Int.self, forKey: .imageCount)
        alignSummary = try c.decodeIfPresent(AlignSummary.self, forKey: .alignSummary)
        runs = try c.decodeIfPresent([RunRecord].self, forKey: .runs) ?? []
        createdAt = try c.decode(Date.self, forKey: .createdAt)
        cameraPaths = try c.decodeIfPresent([CameraPath].self, forKey: .cameraPaths) ?? []
        overlay = try c.decodeIfPresent(OverlaySpec.self, forKey: .overlay)
        // An unknown aspect string decodes leniently to nil (decoding the enum directly would throw
        // on a future case).
        let aspectRaw = try c.decodeIfPresent(String.self, forKey: .outputAspect)
        outputAspect = aspectRaw.flatMap(OutputAspect.init(rawValue:))
        cinemaWatermarkSeeded =
            try c.decodeIfPresent(Bool.self, forKey: .cinemaWatermarkSeeded) ?? false
    }
}

extension Project {
    /// The manifest snapshot after removing a run — the pure transition function used by app
    /// round 2 Task 7's Delete.
    /// If runs becomes empty it demotes the state: `.aligned` when sparse/0 exists, otherwise
    /// `.imported`.
    /// Whether sparse exists is received as a parameter rather than queried from the file system,
    /// preserving purity (and testability).
    func removingRun(id: String, hasSparse: Bool) -> Project {
        var p = self
        p.runs.removeAll { $0.id == id }
        if p.runs.isEmpty { p.state = hasSparse ? .aligned : .imported }
        return p
    }
}

extension JSONEncoder {
    /// The manifest-only encoder — ISO8601 dates plus sorted keys (diff stability).
    static var project: JSONEncoder {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        return e
    }
}

extension JSONDecoder {
    static var project: JSONDecoder {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }
}
