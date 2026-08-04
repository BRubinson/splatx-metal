import Foundation

/// The ring buffer for the bottom console. A FIFO that removes from the front (the oldest) once
/// `cap` is exceeded — the newest lines are always preserved. It sits on AppModel (@MainActor) as a
/// value type, and Observation coalesces the UI updates at frame rate.
struct ConsoleLog {
    private(set) var lines: [String] = []
    var cap: Int = 500

    /// The cumulative append count — it increases monotonically even after `lines.count` stops at
    /// the cap.
    /// ConsolePanel's "a new line arrived" detection (the auto-scroll trigger) observes this
    /// value.
    private(set) var totalAppended: Int = 0

    mutating func append(_ line: String) {
        lines.append(line)
        totalAppended += 1
        if lines.count > cap {
            lines.removeFirst(lines.count - cap)
        }
    }
}
