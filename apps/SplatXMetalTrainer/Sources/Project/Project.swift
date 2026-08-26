import Foundation

/// Where a project's images came from. **A label and a hint — never a gate.** Nothing validates
/// against it, nothing on the training path reads it, and no file layout follows from it (that is
/// `ProjectState`'s job), which is what makes adding a case cheap: the behaviours keyed off it are
/// the Home card's placeholder symbol (`ProjectCard.placeholderSymbol`), the align sample-ordering
/// hint (`AlignViewModel.usesSequentialMatching`), and the Align tab's re-align notice
/// (`AlignView.existingAlignmentNotice`) — all three switch exhaustively so a new case cannot be
/// forgotten silently.
///
/// `.capture` is a bundle from the SplatX iOS app. It is a separate case from `.colmap` even
/// though a LiDAR capture also arrives with a solved model, because the two are not
/// interchangeable downstream: a capture's poses were measured on-device and re-running SfM
/// discards them (the bundle README says so outright), which is a warning the Align tab has to be
/// able to raise — and it can only raise it if the manifest still remembers where the project came
/// from. (`AlignView.existingAlignmentNotice` is that warning.)
///
/// The raw value is what the inspector prints verbatim, so the case names are lower-case words
/// chosen to read as labels. An unknown raw value throws on decode, which `ProjectStore.scan`
/// turns into a skipped project plus a warning — i.e. a manifest from a newer build opened by an
/// older one is ignored, not misread.
enum ProjectSourceKind: String, Codable { case images, video, colmap, capture }
enum ProjectState: String, Codable { case imported, aligned, trained }

struct AlignSummary: Codable, Equatable {
    var poseCount: Int
    var pointCount: Int
    var alignSeconds: Double
    /// The ground-level correction actually applied at align write time, in degrees
    /// (2026-08-04 spec §1.5). nil = nothing was rotated — either the Auto-level toggle was off,
    /// or the measured tilt was under `GroundLevel.minTiltDegrees`. The UI hides the row on nil
    /// rather than printing "0.0°".
    /// Codable is synthesized: an Optional decodes a missing key as nil and encodes nothing when
    /// nil, so old manifests open unchanged (the forward-compatibility convention of this file).
    var levelTiltDegrees: Double?
}

/// The output of one training run. id = the subfolder name under runs/ (a yyyyMMdd-HHmmss stamp).
/// File paths are recorded only relative to the project directory
/// ("runs/<id>/<outputFile>"), making them immune to the project folder being moved or renamed.
struct RunRecord: Codable, Equatable, Identifiable {
    var id: String
    var outputFile: String        // new runs are fixed to "output.ply" (round 4) — the "output.spz" of older manifests remains readable
    var previewFile: String?      // "preview.png"
    /// How the run ended. Four values, and only three of them come from the core: the
    /// `SplatXTrainer.RunResult.Outcome` raw values ("completed", "thermalAborted", "cancelled")
    /// plus **"collapsed"**, which the app mints itself when a run that reached run_until produced
    /// nothing usable (`RunHealth.recordOutcome` — the 2026-08-16 incident, 0 splats reported as
    /// complete). It stays a plain `String` rather than an enum precisely so that widening the
    /// vocabulary costs no manifest compatibility: an older build decodes "collapsed" fine and
    /// shows it in the RUNS badge, which already displays anything that is not "completed".
    var outcome: String
    var finalPSNR: Double?
    var splatCount: Int
    var optionsSummary: String
    var createdAt: Date
    /// The antialiased setting used at training time (Cinema Task 5). A standard PLY does not store
    /// an AA flag, so for sx_render_session to render consistently with training the caller must
    /// pass this record (the splatx_train_c.h contract). false when absent from an older manifest —
    /// the same as the past training default.
    var antialiased: Bool
    /// The timelapse file produced by this run (2026-08-04 spec §3.4) — a **project-relative**
    /// path, "runs/<id>/timelapse.{mp4,mov,gif}". nil when the timelapse was off, when fewer than
    /// two frames were captured, or in an older manifest.
    ///
    /// **It is not shaped like `outputFile`** (FIX 1 — this line used to say "the same convention
    /// as outputFile"): the `runs/<id>/` prefix is part of the stored string here, whereas
    /// outputFile is a bare file name that every reader joins onto `runs/<id>/` itself. Both are
    /// interpreted relative to the project directory — the property that survives a folder move —
    /// but they resolve differently, so do not pass one where the other is expected.
    var timelapseFile: String?

    init(id: String, outputFile: String, previewFile: String?, outcome: String,
         finalPSNR: Double?, splatCount: Int, optionsSummary: String, createdAt: Date,
         antialiased: Bool = false, timelapseFile: String? = nil) {
        self.id = id; self.outputFile = outputFile; self.previewFile = previewFile
        self.outcome = outcome; self.finalPSNR = finalPSNR; self.splatCount = splatCount
        self.optionsSummary = optionsSummary; self.createdAt = createdAt
        self.antialiased = antialiased
        self.timelapseFile = timelapseFile
    }

    enum CodingKeys: String, CodingKey {
        case id, outputFile, previewFile, outcome, finalPSNR, splatCount, optionsSummary
        case createdAt, antialiased, timelapseFile
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
        timelapseFile = try c.decodeIfPresent(String.self, forKey: .timelapseFile)
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
    /// Whether align bakes a ground-level correction into sparse/0 (2026-08-04 spec §1.5).
    /// **Defaults to true when absent from an older manifest** — unlike the other new flags in
    /// this file, the default here is ON because a tilted export is a defect the user did not
    /// ask for, and the correction is provably image-preserving (GroundLevelTests).
    var autoLevelOnAlign: Bool
    /// The training timelapse settings (2026-08-04 spec §3.6) — a project-level setting shared by
    /// every run, exactly like the Cinema camera paths and overlay. The defaults when absent from
    /// an older manifest (the decodeIfPresent forward-compatibility convention) — and the default
    /// is **off**, so opening an old project changes no behaviour.
    var timelapse: TimelapseSettings

    var latestRun: RunRecord? { runs.max(by: { $0.createdAt < $1.createdAt }) }

    enum CodingKeys: String, CodingKey {
        case id, name, sourceKind, state, imageCount, alignSummary, runs, createdAt
        case cameraPaths, overlay, outputAspect, cinemaWatermarkSeeded
        case autoLevelOnAlign, timelapse
    }

    init(id: UUID, name: String, sourceKind: ProjectSourceKind, state: ProjectState,
         imageCount: Int, alignSummary: AlignSummary?, runs: [RunRecord], createdAt: Date,
         cameraPaths: [CameraPath] = [], overlay: OverlaySpec? = nil,
         outputAspect: OutputAspect? = nil, cinemaWatermarkSeeded: Bool = false,
         autoLevelOnAlign: Bool = true,
         timelapse: TimelapseSettings = TimelapseSettings()) {
        self.id = id; self.name = name; self.sourceKind = sourceKind; self.state = state
        self.imageCount = imageCount; self.alignSummary = alignSummary
        self.runs = runs; self.createdAt = createdAt
        self.cameraPaths = cameraPaths
        self.overlay = overlay
        self.outputAspect = outputAspect
        self.cinemaWatermarkSeeded = cinemaWatermarkSeeded
        self.autoLevelOnAlign = autoLevelOnAlign
        self.timelapse = timelapse
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
        autoLevelOnAlign = try c.decodeIfPresent(Bool.self, forKey: .autoLevelOnAlign) ?? true
        timelapse = try c.decodeIfPresent(TimelapseSettings.self, forKey: .timelapse)
            ?? TimelapseSettings()
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

    /// Records (or clears) the timelapse file of one run — the pure transition applied when the
    /// asynchronous encode finishes. The encode completes well after the run was appended, and
    /// the run may have been deleted meanwhile, so an unknown id is a no-op rather than an error.
    func settingTimelapseFile(runID: String, path: String?) -> Project {
        guard let index = runs.firstIndex(where: { $0.id == runID }) else { return self }
        var p = self
        p.runs[index].timelapseFile = path
        return p
    }

    /// The manifest's current copy of an open run.
    ///
    /// `ViewerModel.currentRun` is a **value snapshot** taken when the run was opened
    /// (`ProjectSession.showRun`), and `RunRecord` is a struct, so every manifest write that lands
    /// afterwards is invisible to it. `timelapseFile` is the first field ever written *after* the run
    /// opens — the encode finishes minutes later — and reading the snapshot made the RESULT row fall
    /// back from `encoding…` to `—` at exactly the moment the file appeared, which reads as "it failed"
    /// for a file that is sitting on disk.
    ///
    /// The split: **identity** comes from the snapshot (which run the viewer has loaded), **contents**
    /// come from the manifest. The fallback returns the snapshot itself so a run that is no longer in
    /// the manifest still renders instead of vanishing mid-frame.
    func record(matching open: RunRecord?) -> RunRecord? {
        guard let open else { return nil }
        return runs.first { $0.id == open.id } ?? open
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
