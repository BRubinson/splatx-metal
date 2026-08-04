// Cinema Task 1 (2026-07-26 spec, "data model and persistence" section): the camera path
// persistence model.
// Written in-house — it uses only standard data types, with no derivation from external code.
//
// Serialization contract (project.json stores this format verbatim — Task 3):
// - quat is stored w-first (w,x,y,z) in a SIMD4<Float>. That matches the repo convention
//   (CameraConversion.qvec, the COLMAP qvec), and SIMD4's Codable emits the JSON array in
//   component order as [w,x,y,z].
// - A keyframe is a world pose (pos+quat+fov) — derived from the view matrix rather than from the
//   orbit state (spec: free-flight expressiveness plus backend independence). roll is a field
//   reserved by the spec (unused in v1, always 0).
// - Fields that may be added later use decodeIfPresent plus a default (the Project.swift
//   precedent). ease decodes leniently, falling back to the default on an unknown rawValue, which
//   preserves backward compatibility with older versions when profiles are added.
import Foundation
import simd

/// The easing profile. PathSampler composes the quintic (6t⁵−15t⁴+10t³) onto the time axis (the
/// spec's path math).
enum EaseProfile: String, Codable {
    case none
    case easeInOut
}

/// Hybrid pacing (spec footnote 1 — Task 3). rawValue is the persistence format — do not change
/// the strings.
/// - `.even`: the constant-speed default — the key pass-through times are **derived** from the
///   arc-length fractions (PathSampler.keyframeTimes), and Keyframe.time is not consumed for
///   display.
/// - `.manual`: entered by dragging a timeline diamond — Keyframe.time (monotone) becomes
///   authoritative for the pass-through times, and the sampler respects the key times through a
///   piecewise-linear time→arc-length mapping.
///   The global ease is **not applied** in manual mode (applying it would shift the key
///   pass-through times — incompatible with the key-time respect contract. The inspector shows the
///   Ease control disabled).
enum PacingMode: String, Codable {
    case even
    case manual
}

/// One camera keyframe — a world pose.
struct Keyframe: Codable, Equatable {
    /// The timeline position (seconds). **Array order** is always authoritative for the path
    /// order, and whether time is consumed depends on the pacing (Task 3, spec footnote 1): under
    /// `.even` it is not consumed (the pass-through times are derived by
    /// PathSampler.keyframeTimes — this field is a residual stored value), and under `.manual` it
    /// is authoritative for the pass-through times (monotone — the drag clamp maintains it, and
    /// the sampler applies a defensive correction).
    var time: Double
    /// The world position.
    var pos: SIMD3<Float>
    /// The rotation — w-first (w,x,y,z). The JSON is [w,x,y,z] too.
    var quat: SIMD4<Float>
    /// The vertical FOV in radians — the same convention as CameraController.verticalFOV.
    var fov: Float
    /// Reserved (spec) — unused in v1, default 0. Decodes as 0 when absent from older JSON.
    var roll: Float

    /// The stored (w,x,y,z) → simd_quatf. The same component correspondence as
    /// CameraConversion.rotation.
    var quatValue: simd_quatf {
        simd_quatf(ix: quat.y, iy: quat.z, iz: quat.w, r: quat.x)
    }

    init(time: Double, pos: SIMD3<Float>, quat: SIMD4<Float>, fov: Float, roll: Float = 0) {
        self.time = time; self.pos = pos; self.quat = quat; self.fov = fov; self.roll = roll
    }

    /// simd_quatf → the w-first storage form. Used by the capture path (derived from the view
    /// matrix).
    init(time: Double, pos: SIMD3<Float>, rotation: simd_quatf, fov: Float, roll: Float = 0) {
        self.init(time: time, pos: pos,
                  quat: SIMD4(rotation.real, rotation.imag.x, rotation.imag.y, rotation.imag.z),
                  fov: fov, roll: roll)
    }

    enum CodingKeys: String, CodingKey { case time, pos, quat, fov, roll }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        time = try c.decode(Double.self, forKey: .time)
        pos = try c.decode(SIMD3<Float>.self, forKey: .pos)
        quat = try c.decode(SIMD4<Float>.self, forKey: .quat)
        fov = try c.decode(Float.self, forKey: .fov)
        roll = try c.decodeIfPresent(Float.self, forKey: .roll) ?? 0
    }
}

/// One camera path. Several are held at project level (they are in scene coordinates, so they are
/// reused across runs — spec).
struct CameraPath: Codable, Equatable, Identifiable {
    var id: UUID
    var name: String
    var keyframes: [Keyframe]
    /// The Smoothness knob (0…1, default 0.5). The v1 sampler is fixed at centripetal α=0.5, so it
    /// does not consume it yet — exposing it in the inspector and mapping it to the interpolation
    /// belongs to a later task (only the field is reserved).
    var smoothness: Double
    var ease: EaseProfile
    /// The closed end condition (adding the last→first segment). An explicit toggle (spec UX).
    var loop: Bool
    /// The pacing mode (spec footnote 1 — Task 3). Constant speed by default, switching to manual
    /// on a diamond drag.
    /// `.even` when absent from older JSON (forward compatibility — the decodeIfPresent
    /// convention).
    var pacing: PacingMode
    /// The total playback length (seconds) — seconds are always authoritative for time, and the
    /// frame count is a derived display (Global Constraints).
    var duration: Double

    init(id: UUID = UUID(), name: String, keyframes: [Keyframe] = [],
         smoothness: Double = 0.5, ease: EaseProfile = .easeInOut,
         loop: Bool = false, pacing: PacingMode = .even, duration: Double = 10) {
        self.id = id; self.name = name; self.keyframes = keyframes
        self.smoothness = smoothness; self.ease = ease; self.loop = loop
        self.pacing = pacing
        self.duration = duration
    }

    enum CodingKeys: String, CodingKey {
        case id, name, keyframes, smoothness, ease, loop, pacing, duration
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        name = try c.decode(String.self, forKey: .name)
        keyframes = try c.decodeIfPresent([Keyframe].self, forKey: .keyframes) ?? []
        smoothness = try c.decodeIfPresent(Double.self, forKey: .smoothness) ?? 0.5
        // An unknown profile string falls back to the default — decoding the enum directly would
        // throw on a future case.
        let easeRaw = try c.decodeIfPresent(String.self, forKey: .ease)
        ease = easeRaw.flatMap(EaseProfile.init(rawValue:)) ?? .easeInOut
        loop = try c.decodeIfPresent(Bool.self, forKey: .loop) ?? false
        // An unknown pacing string decodes leniently to the safe default (constant speed).
        let pacingRaw = try c.decodeIfPresent(String.self, forKey: .pacing)
        pacing = pacingRaw.flatMap(PacingMode.init(rawValue:)) ?? .even
        duration = try c.decode(Double.self, forKey: .duration)
    }
}

/// The nine overlay anchors. rawValue is the persistence format — do not change the strings.
enum OverlayAnchor: String, Codable, CaseIterable {
    case topLeading, top, topTrailing
    case leading, center, trailing
    case bottomLeading, bottom, bottomTrailing
}

/// The static image overlay (spec: detailed position and size settings). Every length unit is a
/// **percentage of the output's short side** — the choice that keeps the placement WYSIWYG even
/// when the resolution preset (1080p/1440p/4K) changes.
struct OverlaySpec: Codable, Equatable {
    /// The path relative to the project directory (the image is a copy inside the project — the
    /// self-containment principle).
    var imagePath: String
    var anchor: OverlayAnchor
    /// The free offset from the anchor (short-side %). +x is right and +y is down.
    var offsetX: Double
    var offsetY: Double
    /// The image display size — a percentage of the output's short side.
    var sizePercent: Double
    /// The anchor margin (short-side %).
    var margin: Double
    /// 0…1.
    var opacity: Double

    init(imagePath: String, anchor: OverlayAnchor = .bottomTrailing,
         offsetX: Double = 0, offsetY: Double = 0,
         sizePercent: Double = 20, margin: Double = 4, opacity: Double = 1) {
        self.imagePath = imagePath; self.anchor = anchor
        self.offsetX = offsetX; self.offsetY = offsetY
        self.sizePercent = sizePercent; self.margin = margin; self.opacity = opacity
    }

    enum CodingKeys: String, CodingKey {
        case imagePath, anchor, offsetX, offsetY, sizePercent, margin, opacity
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        imagePath = try c.decode(String.self, forKey: .imagePath)
        let anchorRaw = try c.decodeIfPresent(String.self, forKey: .anchor)
        anchor = anchorRaw.flatMap(OverlayAnchor.init(rawValue:)) ?? .bottomTrailing
        offsetX = try c.decodeIfPresent(Double.self, forKey: .offsetX) ?? 0
        offsetY = try c.decodeIfPresent(Double.self, forKey: .offsetY) ?? 0
        sizePercent = try c.decodeIfPresent(Double.self, forKey: .sizePercent) ?? 20
        margin = try c.decodeIfPresent(Double.self, forKey: .margin) ?? 4
        opacity = try c.decodeIfPresent(Double.self, forKey: .opacity) ?? 1
    }
}
