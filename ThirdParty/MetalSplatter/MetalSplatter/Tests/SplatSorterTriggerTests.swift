import XCTest
import Metal
import simd
import Synchronization
@testable import MetalSplatter

/// splatx-metal fork (viewer-sort Task 2): tests for pose-thresholded resort triggering.
///
/// Contract under test:
/// - Pose-derived triggers fire only beyond thresholds (dot < 0.999 or move > 1e-3),
///   measured against the pose of the most recently started sort; an idle camera stops
///   re-sorting entirely.
/// - Content-derived triggers (chunk add/remove, noteContentChanged) fire unconditionally.
/// - One-shot sort completion handlers (afterNextSort) always fire promptly, even when
///   registered while the camera is stationary and no sort is pending.
final class SplatSorterTriggerTests: XCTestCase {
    var device: MTLDevice!

    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        XCTAssertNotNil(device, "Metal device required for tests")
    }

    // MARK: - Helpers (mirrors SplatSorterTests helpers)

    /// Thread-safe counter (Mutex itself is noncopyable, so wrap it in a class).
    private final class Counter: Sendable {
        private let mutex = Mutex(0)
        func increment() { mutex.withLock { $0 += 1 } }
        var value: Int { mutex.withLock { $0 } }
    }

    private func makeSplat(at position: SIMD3<Float>) -> EncodedSplatPoint {
        EncodedSplatPoint(position: position,
                          colorSH0: SIMD3<Float>(1, 1, 1),
                          opacity: 1.0,
                          scale: SIMD3<Float>(0.1, 0.1, 0.1),
                          rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1))
    }

    private func makeChunkReference(positions: [SIMD3<Float>], chunkIndex: UInt16) throws -> SplatSorter.ChunkReference {
        let buffer = try MetalBuffer<EncodedSplatPoint>(device: device)
        try buffer.ensureCapacity(positions.count)
        for position in positions {
            buffer.append(makeSplat(at: position))
        }
        return SplatSorter.ChunkReference(chunkIndex: chunkIndex, buffer: buffer)
    }

    private func waitForSortedBuffer(_ sorter: SplatSorter, seconds: TimeInterval = 2,
                                     file: StaticString = #filePath, line: UInt = #line) async {
        let buffer = await withTaskGroup(of: MetalBuffer<ChunkedSplatIndex>?.self) { group in
            group.addTask { await sorter.obtainSortedIndices() }
            group.addTask {
                try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
                return nil
            }
            let first = await group.next() ?? nil
            group.cancelAll()
            return first
        }
        XCTAssertNotNil(buffer, "sort should complete within \(seconds)s", file: file, line: line)
        if let buffer {
            sorter.releaseSortedIndices(buffer)
        }
    }

    private static let idlePose = (position: SIMD3<Float>(3, 1, 4), forward: normalize(SIMD3<Float>(0.2, -0.1, -1)))

    /// Sorter with one chunk, one completed sort at `idlePose`, and the sort loop idle.
    private func makeIdleSortedSorter(sortStarts: Counter? = nil) async throws -> SplatSorter {
        let sorter = try SplatSorter(device: device)
        if let sortStarts {
            sorter.onSortStart = { sortStarts.increment() }
        }
        let chunk = try makeChunkReference(positions: [
            SIMD3<Float>(0, 0, -5),
            SIMD3<Float>(0, 0, -2),
            SIMD3<Float>(0, 0, -8),
        ], chunkIndex: 0)
        sorter.setChunks([chunk])
        sorter.updateCameraPose(position: Self.idlePose.position, forward: Self.idlePose.forward)
        await waitForSortedBuffer(sorter)
        // Let the loop drain any residual pending sort (setChunks + first pose update can
        // race into two pickups) before tests assert on idle behavior.
        try? await Task.sleep(nanoseconds: 100_000_000)
        return sorter
    }

    // MARK: - Pure threshold function

    private typealias Pose = SplatSorter.CameraPose

    private let basePose = Pose(position: SIMD3<Float>(1, 2, 3), forward: SIMD3<Float>(0, 0, -1))

    private func forwardRotated(byDegrees degrees: Float) -> SIMD3<Float> {
        let radians = degrees * .pi / 180
        return SIMD3<Float>(sin(radians), 0, -cos(radians))
    }

    func testNilLastSortPoseAlwaysRequiresSort() {
        XCTAssertTrue(SplatSorter.poseRequiresResort(from: nil, to: basePose))
    }

    func testIdenticalPoseDoesNotRequireSort() {
        XCTAssertFalse(SplatSorter.poseRequiresResort(from: basePose, to: basePose))
    }

    func testRotationBelowThresholdDoesNotRequireSort() {
        // 1 degree: dot = cos(1°) ≈ 0.99985 >= 0.999
        let pose = Pose(position: basePose.position, forward: forwardRotated(byDegrees: 1))
        XCTAssertFalse(SplatSorter.poseRequiresResort(from: basePose, to: pose))
    }

    func testRotationAboveThresholdRequiresSort() {
        // 5 degrees: dot = cos(5°) ≈ 0.99619 < 0.999
        let pose = Pose(position: basePose.position, forward: forwardRotated(byDegrees: 5))
        XCTAssertTrue(SplatSorter.poseRequiresResort(from: basePose, to: pose))
    }

    func testTranslationBelowEpsilonDoesNotRequireSort() {
        // |delta| = 5e-4 < 1e-3
        let pose = Pose(position: basePose.position + SIMD3<Float>(5e-4, 0, 0), forward: basePose.forward)
        XCTAssertFalse(SplatSorter.poseRequiresResort(from: basePose, to: pose))
    }

    func testTranslationAboveEpsilonRequiresSort() {
        // |delta| = 0.1 > 1e-3
        let pose = Pose(position: basePose.position + SIMD3<Float>(0.1, 0, 0), forward: basePose.forward)
        XCTAssertTrue(SplatSorter.poseRequiresResort(from: basePose, to: pose))
    }

    // MARK: - Idle camera stops re-sorting

    func testIdenticalPoseUpdatesDoNotRetriggerSort() async throws {
        let sortStarts = Counter()
        let sorter = try await makeIdleSortedSorter(sortStarts: sortStarts)
        let baseline = sortStarts.value
        XCTAssertGreaterThanOrEqual(baseline, 1, "at least one sort must have run")

        // Simulate a stationary camera: repeated identical pose updates (one per frame).
        for _ in 0..<50 {
            sorter.updateCameraPose(position: Self.idlePose.position, forward: Self.idlePose.forward)
        }
        try? await Task.sleep(nanoseconds: 200_000_000)

        XCTAssertEqual(sortStarts.value, baseline,
                       "an idle camera must not schedule any further sorts")
    }

    // MARK: - Content changes always trigger (no-regression contract)

    func testSortCompletionHandlerFiresWithoutAnyCameraPoseUpdate() async throws {
        // The LiveSplatStreamer chain: addChunk(enabled:false) -> afterNextSort -> swap.
        // The handler must fire even if the camera never produces a pose update.
        let sorter = try SplatSorter(device: device)
        let chunk = try makeChunkReference(positions: [SIMD3<Float>(0, 0, -5)], chunkIndex: 0)
        try sorter.addChunkToSort(chunk)

        let handlerFired = expectation(description: "sort completion handler fires with no pose update")
        sorter.addSortCompletionHandler { handlerFired.fulfill() }
        await fulfillment(of: [handlerFired], timeout: 2)
    }

    func testSortCompletionHandlerFiresWhenRegisteredWhileIdle() async throws {
        // Registering a handler when no sort is pending or in progress (idle camera,
        // no content change) must force one sort so the handler still fires.
        let sorter = try await makeIdleSortedSorter()

        let handlerFired = expectation(description: "handler registered while idle still fires")
        sorter.addSortCompletionHandler { handlerFired.fulfill() }
        await fulfillment(of: [handlerFired], timeout: 2)
    }

    func testChunkAddTriggersResortDespiteIdlePose() async throws {
        let sortStarts = Counter()
        let sorter = try await makeIdleSortedSorter(sortStarts: sortStarts)
        let baseline = sortStarts.value

        let chunk = try makeChunkReference(positions: [SIMD3<Float>(1, 1, -3)], chunkIndex: 1)
        try sorter.addChunkToSort(chunk)

        // Content change must schedule a sort regardless of the (unchanged) pose.
        let deadline = Date().addingTimeInterval(2)
        while sortStarts.value <= baseline && Date() < deadline {
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertGreaterThan(sortStarts.value, baseline,
                             "adding a chunk must trigger a resort despite an idle camera")
    }

    func testNoteContentChangedTriggersResortDespiteIdlePose() async throws {
        let sortStarts = Counter()
        let sorter = try await makeIdleSortedSorter(sortStarts: sortStarts)
        let baseline = sortStarts.value

        sorter.noteContentChanged()

        let deadline = Date().addingTimeInterval(2)
        while sortStarts.value <= baseline && Date() < deadline {
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertGreaterThan(sortStarts.value, baseline,
                             "noteContentChanged must trigger a resort despite an idle camera")
    }

    func testNoteContentChangedWithoutChunksDoesNotSpin() throws {
        // Empty chunks: needsSort must stay false so the sort loop can exit.
        let sorter = try SplatSorter(device: device)
        sorter.noteContentChanged()  // must not schedule anything or crash
        XCTAssertNil(sorter.tryObtainSortedIndices())
    }
}
