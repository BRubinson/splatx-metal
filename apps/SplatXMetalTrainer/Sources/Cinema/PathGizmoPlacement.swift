// Introduced in path transform FIX 10, kept by FIX 13 (2026-07-28 spec footnotes 10 and 12): the
// **single source of truth for the world placement and display guard** of the two gizmo handles
// (pure).
//
// FIX 10 introduced this file as "the shared source of conditions for the marker (the renderer) and
// the transparent proxy (the view)" when it moved the handle pixels to in-pass Metal. FIX 13
// returned the drawing to SwiftUI (a direct-manipulation affordance must follow the input — the
// comment at the top of PathGizmoOverlay is the authoritative contract), so the consumers dropped
// to just PathGizmoOverlay, but the file stays — the placement and the guard are pure functions
// outside a view body and are pinned by unit tests, and the contract that the placement and the
// drag basis use **the same derivation** through cameraAxes remains in one place. (FIX 11's
// isCenterVisible — sharing the cull decision between the in-pass marker and the proxy — was
// removed along with FIX 13, since its consumer was the renderer. The behind-camera cull is handled
// exactly as in §4.3 by the view's PathProjection.viewportPoint returning nil.)
//
// cameraRight is taken from **the camera that places the markers** (the presented camera,
// viewer.overlayCamera) — the handle direction is part of the placement too, so it follows the
// splats' on-screen consistency.
import Foundation
import simd

/// The world placement of the gizmo handles. A pure value depending only on the camera axes.
struct PathGizmoPlacement: Equatable {
    /// The world position of the center (cross) handle = the pivot.
    var center: SIMD3<Float>
    /// The world position of the radius (circle) handle. nil when `meanRadius == 0` (a single-key
    /// still shot) — the §4.3 guard against drawing it overlapping the pivot.
    var radius: SIMD3<Float>?

    /// The pivot + the camera axes → the placement. No pivot (no path, or 0 keys) gives nil = there
    /// are no handles at all.
    /// The radius handle position is §4.3's `pivot + û · R̄` (û = PathTransform.handleDirection —
    /// that function also absorbs the near-top-view degeneracy through its fallback, FIX 2).
    static func make(pivot: PathTransform.Pivot?, cameraRight: SIMD3<Float>,
                     upAxis: SIMD3<Float>) -> PathGizmoPlacement? {
        guard let pivot else { return nil }
        guard pivot.meanRadius > 0 else {
            return PathGizmoPlacement(center: pivot.point, radius: nil)
        }
        let dir = PathTransform.handleDirection(cameraRight: cameraRight, upAxis: upAxis)
        return PathGizmoPlacement(center: pivot.point,
                                  radius: pivot.point + dir * pivot.meanRadius)
    }

    /// Extracts the camera right/forward from the rows of the view matrix (world→view) — the same
    /// procedure as `CameraController.pan()` (the axes of the override pose when an override is
    /// active). It lives in this one place so that the placement (right) and the center handle's
    /// drag basis (right and forward) use the same derivation.
    static func cameraAxes(_ cam: CameraController)
        -> (right: SIMD3<Float>, forward: SIMD3<Float>) {
        let v = cam.viewMatrix
        return (right: SIMD3(v.columns.0.x, v.columns.1.x, v.columns.2.x),
                forward: -SIMD3(v.columns.0.z, v.columns.1.z, v.columns.2.z))
    }
}
