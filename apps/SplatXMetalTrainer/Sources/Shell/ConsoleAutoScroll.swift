import CoreGraphics

/// The pure logic of the console's stick-to-bottom decision — `ConsolePanel` consumes it.
///
/// The contract: by default it follows the latest line (stuck). If the user scrolls up, the stick
/// is released; scrolling back down near the bottom re-sticks it. Content growth caused by new
/// lines arriving (the offset is unchanged, only the distance from the bottom grows) is not the
/// user's intent, so it does not release the stick — release happens only through "a decrease in
/// offset (scrolling upward)".
enum ConsoleAutoScroll {
    /// The near-bottom threshold (pt). Within this distance from the bottom it is treated as "at
    /// the bottom" and re-sticks.
    /// (A console line is ~13pt tall — a slip of less than two lines counts as the bottom.)
    static let bottomThreshold: CGFloat = 24

    /// The minimum offset decrease (pt) accepted as upward movement — ignoring subpixel jitter.
    static let upwardJitter: CGFloat = 0.5

    /// Whether the scroll geometry is near the bottom. If the content is smaller than the viewport
    /// (scrolling impossible) it is always true. Rubber-band overscroll (offsetY > maxOffset) also
    /// counts as the bottom.
    static func isNearBottom(contentHeight: CGFloat, containerHeight: CGFloat,
                             offsetY: CGFloat, threshold: CGFloat = bottomThreshold) -> Bool {
        let maxOffset = contentHeight - containerHeight
        guard maxOffset > 0 else { return true }
        return offsetY >= maxOffset - threshold
    }

    /// The stick-state transition for one geometry change.
    /// - Near the bottom → re-stick (true)
    /// - The offset decreased by more than the jitter (the user scrolled up) → release (false)
    /// - Otherwise (content growth from new lines, or a programmatic scroll in progress) → keep the
    ///   current state
    static func nextStick(current: Bool, previousOffsetY: CGFloat, offsetY: CGFloat,
                          contentHeight: CGFloat, containerHeight: CGFloat) -> Bool {
        if isNearBottom(contentHeight: contentHeight, containerHeight: containerHeight,
                        offsetY: offsetY) {
            return true
        }
        if offsetY < previousOffsetY - upwardJitter {
            return false
        }
        return current
    }
}
