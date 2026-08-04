import Foundation
import Metal

public struct ProbeReport: Codable {
    public var deviceName: String
    public var osVersion: String
    public var atomicFloatWorks: Bool
    public var simdWidth: Int
    public var maxThreadgroupMemoryBytes: Int
    public var recommendedMaxWorkingSetBytes: Int
    public var physicalMemoryBytes: Int
    public var availableMemoryBytes: Int
    /// Median GPU ms of the first 3 batches after warmup (meaning pinned for JSON schema consumers)
    public var sustainedFirstBatchMs: Double
    /// Median GPU ms of the last 3 batches after warmup (meaning pinned for JSON schema consumers)
    public var sustainedLastBatchMs: Double
    public var sustainedRatio: Double
}

public func runProbeSuite(_ ctx: MetalContext, sustainedSeconds: Double) throws -> ProbeReport {
    // 1) atomic float correctness
    // Pipeline creation may throw on a device without atomic float support — and that device is
    // exactly the most important data point of this probe, so the failure is recorded as
    // atomicFloatWorks = false and report generation continues.
    var atomicOk = false
    do {
        let threadCount = 4096
        let accBuf = ctx.device.makeBuffer(length: 4, options: .storageModeShared)!
        accBuf.contents().storeBytes(of: Float(0), as: Float.self)
        let atomicPipe = try ctx.device.makeComputePipelineState(
            function: ctx.library.makeFunction(name: "probe_atomic_float")!)
        _ = try timedCommit(ctx) { enc in
            enc.setComputePipelineState(atomicPipe)
            enc.setBuffer(accBuf, offset: 0, index: 0)
            enc.dispatchThreads(MTLSize(width: threadCount, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        atomicOk = accBuf.contents().load(as: Float.self) == Float(threadCount)
    } catch {
        atomicOk = false
    }

    // 2) simdgroup width
    let widthBuf = ctx.device.makeBuffer(length: 4, options: .storageModeShared)!
    let widthPipe = try ctx.device.makeComputePipelineState(
        function: ctx.library.makeFunction(name: "probe_simd_width")!)
    _ = try timedCommit(ctx) { enc in
        enc.setComputePipelineState(widthPipe)
        enc.setBuffer(widthBuf, offset: 0, index: 0)
        enc.dispatchThreads(MTLSize(width: 64, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }
    let simdWidth = Int(widthBuf.contents().load(as: UInt32.self))

    // 3) Sustained load: repeat a fixed workload batch for sustainedSeconds, recording the GPU ms trend
    let n: UInt32 = 1 << 20
    let dataBuf = ctx.device.makeBuffer(length: Int(n) * 4, options: .storageModeShared)!
    let loadPipe = try ctx.device.makeComputePipelineState(
        function: ctx.library.makeFunction(name: "probe_sustained_load")!)
    var nCopy = n
    let encodeBatch: (MTLComputeCommandEncoder) -> Void = { enc in
        enc.setComputePipelineState(loadPipe)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBytes(&nCopy, length: 4, index: 1)
        enc.dispatchThreads(MTLSize(width: Int(n) * 4, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    // Run one warmup batch and discard it: on the first dispatch a cold shader cache (pipeline
    // JIT), buffer page-in, and GPU power-ramp costs overlap, which can inflate the GPU time.
    // Including it in the measurement would contaminate sustainedRatio (the input to the
    // throttle factor), so it is not recorded.
    _ = try timedCommit(ctx, encode: encodeBatch)

    var batchMs: [Double] = []
    let deadline = Date().addingTimeInterval(sustainedSeconds)
    repeat {
        let t = try timedCommit(ctx, encode: encodeBatch)
        batchMs.append(t.gpuMilliseconds)
    } while Date() < deadline

    // Absorb single-batch noise: compare by the median of the first/last 3 batches
    // (if there are fewer than 3 batches, the median of whatever is present)
    let first = median(Array(batchMs.prefix(3)))
    let last = median(Array(batchMs.suffix(3)))

    var available = -1
    #if os(iOS)
    available = Int(os_proc_available_memory())
    #endif

    return ProbeReport(
        deviceName: ctx.device.name,
        osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
        atomicFloatWorks: atomicOk,
        simdWidth: simdWidth,
        maxThreadgroupMemoryBytes: ctx.device.maxThreadgroupMemoryLength,
        recommendedMaxWorkingSetBytes: Int(ctx.device.recommendedMaxWorkingSetSize),
        physicalMemoryBytes: Int(ProcessInfo.processInfo.physicalMemory),
        availableMemoryBytes: available,
        sustainedFirstBatchMs: first,
        sustainedLastBatchMs: last,
        sustainedRatio: first > 0 ? last / first : 0
    )
}

private func median(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let mid = sorted.count / 2
    return sorted.count % 2 == 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
}
