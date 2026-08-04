import Metal

public struct GpuTiming {
    public let gpuMilliseconds: Double
}

public enum GpuTimerError: Error {
    case commandBufferFailed(underlying: Error?)
}

/// Commits the encode closure as a single command buffer and returns the GPU time (ms).
/// The Phase 0 implementation of the no-wall-clock convention (design §10.2) — GPU time per
/// command buffer.
public func timedCommit(
    _ ctx: MetalContext,
    encode: (MTLComputeCommandEncoder) -> Void
) throws -> GpuTiming {
    guard let cb = ctx.queue.makeCommandBuffer(),
          let enc = cb.makeComputeCommandEncoder() else {
        throw GpuTimerError.commandBufferFailed(underlying: nil)
    }
    encode(enc)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    guard cb.status == .completed else {
        throw GpuTimerError.commandBufferFailed(underlying: cb.error)
    }
    return GpuTiming(gpuMilliseconds: (cb.gpuEndTime - cb.gpuStartTime) * 1000.0)
}
