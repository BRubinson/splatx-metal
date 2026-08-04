import Metal
import Testing
@testable import SplatXMetal

// Tests that share the GPU — serialized because, under parallel execution, other tests' GPU work
// inflates the sustained probe's batch GPU time and contaminates sustainedRatio. Nesting under
// GpuTests serializes **across** suites (the parent's .serialized applies recursively — the
// GpuTests.swift comment).
extension GpuTests {
@Suite struct ProbeTests {

    @Test func metalContextLoadsLibrary() throws {
        let ctx = try MetalContext(device: nil)
        #expect(ctx.library.functionNames.contains("probe_atomic_float"))
        #expect(ctx.library.functionNames.contains("probe_simd_width"))
    }

    @Test func atomicFloatAccumulatesExactly() throws {
        let ctx = try MetalContext(device: nil)
        let threadCount = 4096
        let buf = ctx.device.makeBuffer(length: 4, options: .storageModeShared)!
        buf.contents().storeBytes(of: Float(0), as: Float.self)

        let pipeline = try ctx.device.makeComputePipelineState(
            function: ctx.library.makeFunction(name: "probe_atomic_float")!)
        let timing = try timedCommit(ctx) { enc in
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(buf, offset: 0, index: 0)
            enc.dispatchThreads(MTLSize(width: threadCount, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        let sum = buf.contents().load(as: Float.self)
        #expect(sum == Float(threadCount))       // exactly the thread count if atomic float works
        #expect(timing.gpuMilliseconds > 0)
    }

    @Test func probeSuiteProducesReport() throws {
        let ctx = try MetalContext(device: nil)
        let report = try runProbeSuite(ctx, sustainedSeconds: 2.0)   // only 2 s in tests
        #expect(report.atomicFloatWorks)
        #expect(report.simdWidth == 32)                              // the Apple GPU convention
        #expect(report.maxThreadgroupMemoryBytes >= 32 * 1024)
        #expect(report.sustainedFirstBatchMs > 0)
        #expect(report.sustainedRatio > 0.5 && report.sustainedRatio < 10)

        let data = try JSONEncoder().encode(report)
        #expect(!data.isEmpty)
    }
}
}
