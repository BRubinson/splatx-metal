import XCTest
@testable import SplatXMetalTrainer

/// Pure-logic tests for the console's stick-to-bottom behavior.
/// The contract: stuck by default (following the newest line) → released when the user scrolls
/// upward → re-stuck on returning near the bottom. Content growth does not break the stick.
final class ConsoleAutoScrollTests: XCTestCase {

    // MARK: - isNearBottom (the threshold for being at the bottom)

    func testContentSmallerThanContainerIsAlwaysBottom() {
        XCTAssertTrue(ConsoleAutoScroll.isNearBottom(contentHeight: 100, containerHeight: 160, offsetY: 0))
    }

    func testExactBottomIsNearBottom() {
        // maxOffset = 1000 - 160 = 840
        XCTAssertTrue(ConsoleAutoScroll.isNearBottom(contentHeight: 1000, containerHeight: 160, offsetY: 840))
    }

    func testWithinThresholdIsNearBottom() {
        XCTAssertTrue(ConsoleAutoScroll.isNearBottom(contentHeight: 1000, containerHeight: 160,
                                                     offsetY: 840 - ConsoleAutoScroll.bottomThreshold))
    }

    func testBeyondThresholdIsNotNearBottom() {
        XCTAssertFalse(ConsoleAutoScroll.isNearBottom(contentHeight: 1000, containerHeight: 160,
                                                      offsetY: 840 - ConsoleAutoScroll.bottomThreshold - 1))
    }

    func testRubberBandOverscrollPastBottomIsNearBottom() {
        XCTAssertTrue(ConsoleAutoScroll.isNearBottom(contentHeight: 1000, containerHeight: 160, offsetY: 900))
    }

    // MARK: - nextStick (state transitions)

    /// While stuck, new lines grow the content only (the offset is unchanged and the bottom
    /// moves away) → the stick is kept.
    func testContentGrowthDoesNotReleaseStick() {
        XCTAssertTrue(ConsoleAutoScroll.nextStick(current: true, previousOffsetY: 840, offsetY: 840,
                                                  contentHeight: 2000, containerHeight: 160))
    }

    /// While stuck, the user scrolls up (the offset decreases) → released.
    func testUserScrollUpReleasesStick() {
        XCTAssertFalse(ConsoleAutoScroll.nextStick(current: true, previousOffsetY: 840, offsetY: 700,
                                                   contentHeight: 1000, containerHeight: 160))
    }

    /// While released, moving up and down far from the bottom keeps it released.
    func testReleasedStaysReleasedMidContent() {
        XCTAssertFalse(ConsoleAutoScroll.nextStick(current: false, previousOffsetY: 300, offsetY: 400,
                                                   contentHeight: 1000, containerHeight: 160))
    }

    /// While released, returning near the bottom → re-stuck.
    func testReturnToBottomResticks() {
        XCTAssertTrue(ConsoleAutoScroll.nextStick(current: false, previousOffsetY: 700, offsetY: 830,
                                                  contentHeight: 1000, containerHeight: 160))
    }

    /// An offset decrease within the jitter band does not count as an upward scroll → the
    /// current state is kept.
    func testSubJitterDecreaseKeepsCurrentState() {
        XCTAssertTrue(ConsoleAutoScroll.nextStick(current: true, previousOffsetY: 500,
                                                  offsetY: 500 - ConsoleAutoScroll.upwardJitter / 2,
                                                  contentHeight: 1000, containerHeight: 160))
    }

    /// When the content is smaller than the viewport (scrolling is impossible), it is always
    /// stuck.
    func testShortContentAlwaysSticks() {
        XCTAssertTrue(ConsoleAutoScroll.nextStick(current: false, previousOffsetY: 0, offsetY: 0,
                                                  contentHeight: 50, containerHeight: 160))
    }
}
