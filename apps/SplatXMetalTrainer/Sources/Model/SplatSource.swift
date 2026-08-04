import Foundation

/// A single file for the viewer to render (a run's output or a temporary converted PLY).
struct SplatSource: Identifiable, Hashable {
    let id: UUID
    var label: String
    let url: URL

    init(file url: URL) {
        id = UUID(); label = url.lastPathComponent; self.url = url
    }
}
