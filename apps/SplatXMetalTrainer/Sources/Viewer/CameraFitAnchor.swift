// 2026-08-04 spec §2.3: the one-shot selector for which camera the **next** applyCameraFit
// anchors on.
// It exists because two paths that must land on the capture's start viewpoint (training start,
// and the result presented at training completion) share the single applyCameraFit entry point
// with the paths that must keep the previous behavior (opening a past run from the history, the
// standalone PLY viewer window, the frameToFit fallback).
// Making it a **flag consumed by** applyCameraFit — rather than a new parameter — is what keeps
// every existing call site byte-identical.
import Foundation

enum CameraFitAnchor {
    /// The previous behavior (2026-07-20 spec §6): the middle name-sorted camera approximated
    /// onto the orbit parameterization through OrbitFit. Roll is dropped and the distance comes
    /// from the points3D centroid.
    case middleOrbit
    /// The **first** name-sorted camera (index 0) injected exactly as a free pose through
    /// `camera.overridePose` — position, orientation (roll included) and fov all reproduce the
    /// capture camera, so the viewport shows the same picture as images/ frame 0.
    /// The middle-camera orbit fit still runs underneath and stays as the orbit state: the
    /// user's first drag absorbs the override through the existing `adoptOverridePose`
    /// contract, which inverts it with `target` as the centroid argument (roll is lost there —
    /// the orbit model's known limit, spec §2.3).
    case firstCameraExact
}
