import SwiftUI
import simd

/// The pure logic that projects the world X/Y/Z unit axes into screen-space 2D directions plus a
/// depth, using the camera view matrix.
///
/// It transforms them as direction vectors (w=0), so only the rotation applies and the translation
/// is ignored.
/// The screen coordinate system is y-up (the mathematical convention): with an identity-ish camera
/// (looking at the origin from +Z), +X → screen right (dir.x>0) and +Y → screen up (dir.y>0).
/// The Canvas is y-down, so dir.y is flipped when drawing (see `AxisGizmoView`).
enum AxisGizmo {
    /// View-transforms each world axis (+X/+Y/+Z) as a direction vector → returns xy as the screen
    /// direction (a normalized 2D vector) and z as the depth.
    /// When an axis is perpendicular to the screen (facing toward or away) xy converges to 0 →
    /// dir is returned as (0,0) (avoiding NaN).
    static func project(view: float4x4) -> [(axis: String, dir: SIMD2<Float>, z: Float)] {
        let worldAxes: [(String, SIMD3<Float>)] = [
            ("X", SIMD3(1, 0, 0)),
            ("Y", SIMD3(0, 1, 0)),
            ("Z", SIMD3(0, 0, 1)),
        ]
        return worldAxes.map { name, axis in
            let v = view * SIMD4<Float>(axis, 0)   // w=0 → a direction (rotation only, translation ignored)
            let xy = SIMD2<Float>(v.x, v.y)
            let len = simd_length(xy)
            let dir = len > 1e-6 ? xy / len : SIMD2<Float>(0, 0)
            return (axis: name, dir: dir, z: v.z)
        }
    }

    /// The axis list sorted into drawing order (far → near) — `AxisGizmoView` consumes it in its
    /// body.
    /// The axis with the larger z (farther from the camera) MUST be drawn **first** so that the
    /// nearer axis lands on top.
    static func sortedForDraw(view: float4x4) -> [(axis: String, dir: SIMD2<Float>, z: Float)] {
        project(view: view).sorted { $0.z > $1.z }
    }
}

/// The top-right axis gizmo: a Canvas draws three colored lines, R=X / G=Y / B=Z, that turn with
/// the camera rotation.
///
/// **The camera MUST be read in this leaf view (path transform FIX 9).** The `camera` of the
/// @Observable ViewerModel is a struct, so reading a single property attaches an observation
/// dependency to the **entire** camera, and since the camera changes wholesale on every orbit tick,
/// **the body that read it is re-evaluated wholesale**. Previously `ViewerView.body` read
/// `vm.camera` and passed it down through `ViewportOverlay(camera:)`, and as a result the entire
/// ViewerView.body — including the viewport stack, the Cinema overlays, and the timeline — ran 60
/// times a second during a drag, all because of one 44pt gizmo (measured 2026-07-28. The inspector
/// case of the same type is spec footnote 8; this case is footnote 9).
///
/// The dependency itself is necessary because this widget **actually draws** the camera — it is not
/// something to remove but something to **narrow in scope**: the parent passes only the model
/// reference (passing a reference is not an observed read) and the read happens here, inside the
/// body. Then the invalidation stops at this view and the gizmo follows the camera with zero lag,
/// exactly as before. A view that needs only a value **derived from** the camera should instead use
/// a snapshot that refreshes only when it changes (`CinemaViewModel.sceneUpAxis`).
struct AxisGizmoView: View {
    let viewer: ViewerModel

    private static let colors: [String: Color] = ["X": .red, "Y": .green, "Z": .blue]

    var body: some View {
        // The observed read happens **inside the body** — the Canvas render closure runs at draw
        // time, so reading there gives no guarantee that a dependency attaches to this view. The
        // closure captures only the resulting array.
        let axes = AxisGizmo.sortedForDraw(view: viewer.camera.viewMatrix)
        Canvas { ctx, size in
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let radius = min(size.width, size.height) / 2 - 3
            for entry in axes {
                let color = Self.colors[entry.axis] ?? .white
                // Screen y-up → Canvas y-down: flip dir.y.
                let end = CGPoint(
                    x: center.x + CGFloat(entry.dir.x) * radius,
                    y: center.y - CGFloat(entry.dir.y) * radius)
                var line = Path()
                line.move(to: center)
                line.addLine(to: end)
                ctx.stroke(line, with: .color(color), style: StrokeStyle(lineWidth: 2, lineCap: .round))
                let dot = CGRect(x: end.x - 2.5, y: end.y - 2.5, width: 5, height: 5)
                ctx.fill(Path(ellipseIn: dot), with: .color(color))
            }
        }
        .frame(width: 44, height: 44)
    }
}
