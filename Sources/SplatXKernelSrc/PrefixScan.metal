// PrefixScan.metal — scan_partial / scan_blocks_single_group / scan_add_offsets (D4 naming).
// Written in-house — implements the reduce-then-scan 3-dispatch design of
// docs/kernel-specs/03-tile-intersect.md §Kernel 2 (replaces gsplat at::cumsum
// (Intersect.cpp:258) / cub::DeviceScan::InclusiveSum (IntersectTile.cu:796–798, the multi-GPU
// path) — not derived code).
//
// Contract: in (int32 counts) → out (int64 inclusive cumsum). **The promotion point is the first
// accumulation** — elements are read as long, so everything from the thread-serial sum onward is
// int64 (n_isects can approach the int32 limit, spec 03).
// Integer addition is independent of the association order → exact agreement with the CPU
// cumsum_i64 is the contract (ScanTests, D9-1).
//
// Dispatch sequence (host MetalBackend::encode_prefix_scan, the same encoder plus a
// memoryBarrier(BarrierScopeBuffers) between dispatches):
//   A) scan_partial            — in-block inclusive scan of kScanBlockSize(=1024) elements per
//                                group + the block total block_sums[g].  A full threadgroup
//                                dispatch (dispatchThreadgroups(num_groups × 256); the tail is
//                                guarded by n).
//   B) scan_blocks_single_group— exclusive scan of block_sums. **A single threadgroup** handles
//                                all of it in a carry-keeping loop of 256 at a time — no upper
//                                bound on the block count (no multi-level recursion needed,
//                                spec 03 §Kernel 2 "single-group loop").
//   C) scan_add_offsets        — the broadcast addition out[i] += block_sums[i / kScanBlockSize].
//
// simd_prefix_inclusive_sum does not support 64 bits → the array of thread totals (long) is placed
// in threadgroup memory (long[256] = 2KB) and scanned with a Hillis–Steele inclusive scan
// (log2(256)=8 steps, 2 barriers per step — read and write separated). No atomics — bit-exact
// determinism.
#include <metal_stdlib>
#include "KernelConstants.h"
#include "KernelParams.h"
using namespace metal;

// Hillis–Steele inclusive scan over the long[256] in threadgroup memory.
// Before entry, each thread's value MUST already be stored in tg[lid] and a barrier MUST have completed.
static inline void tg_inclusive_scan_i64(threadgroup long* tg, const uint lid)
{
    for (uint offset = 1; offset < kScanThreadgroupSize; offset <<= 1) {
        const long v = (lid >= offset) ? tg[lid - offset] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup); // write only after all threads have read
        tg[lid] += v;
        threadgroup_barrier(mem_flags::mem_threadgroup); // next step only once the writes are visible
    }
}

// Dispatch A: in-block (1024 elements) inclusive scan + the block total.
// Thread t accumulates the contiguous range [base + t*4, base + t*4 + 4) serially (spec 03
// thread-serial); the thread totals are then scanned across the threadgroup and the exclusive
// offset is added to the local results.
kernel void scan_partial(
    device const int* in         [[buffer(0)]], // [n] int32 counts (tiles_per_gauss)
    device long*      out        [[buffer(1)]], // [n] in-block inclusive scan (int64)
    device long*      block_sums [[buffer(2)]], // [num_groups] block totals
    constant uint&    n          [[buffer(3)]],
    uint lid      [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]])
{
    threadgroup long tg[kScanThreadgroupSize]; // 256 × 8B = 2KB

    const uint base = group_id * kScanBlockSize + lid * kScanElemsPerThread;
    long local_scan[kScanElemsPerThread];
    long running = 0;
    for (uint k = 0; k < kScanElemsPerThread; k++) {
        const uint i = base + k;
        running += (i < n) ? (long)in[i] : 0; // int32 → int64 promotion point (out of range gives the identity 0)
        local_scan[k] = running;
    }
    tg[lid] = running;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tg_inclusive_scan_i64(tg, lid);

    const long thread_exclusive = tg[lid] - running; // sum up to (excluding) this thread in the block
    for (uint k = 0; k < kScanElemsPerThread; k++) {
        const uint i = base + k;
        if (i < n) out[i] = thread_exclusive + local_scan[k];
    }
    if (lid == kScanThreadgroupSize - 1) block_sums[group_id] = tg[lid]; // block total
}

// Dispatch B: exclusive scan of block_sums (in-place). A single threadgroup (256) walks 256 at a
// time while keeping the carry — no upper bound on num_groups. The loop bound (num_groups) is
// uniform, so the threadgroup barriers sit inside uniform control flow.
kernel void scan_blocks_single_group(
    device long*   block_sums [[buffer(0)]], // [num_groups] totals → exclusive offsets
    constant uint& num_groups [[buffer(1)]],
    uint lid [[thread_position_in_threadgroup]])
{
    threadgroup long tg[kScanThreadgroupSize];

    long carry = 0; // every thread computes the same value (the last element of tg — uniform)
    for (uint chunk = 0; chunk < num_groups; chunk += kScanThreadgroupSize) {
        const uint i = chunk + lid;
        const long v = (i < num_groups) ? block_sums[i] : 0;
        tg[lid] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tg_inclusive_scan_i64(tg, lid);

        if (i < num_groups) block_sums[i] = carry + tg[lid] - v; // inclusive → exclusive
        carry += tg[kScanThreadgroupSize - 1];                   // accumulate this chunk's total
        threadgroup_barrier(mem_flags::mem_threadgroup);         // guarantees reads finish before tg is reused
    }
}

// Dispatch C: broadcast addition of the block exclusive offsets. One thread = one element,
// dispatchThreads(n, tg 256). The offset of block 0 is 0 — the additive identity, so no branch is needed.
kernel void scan_add_offsets(
    device long*       out        [[buffer(0)]], // [n] in-block scan → global inclusive cumsum
    device const long* block_sums [[buffer(1)]], // [num_groups] exclusive block offsets
    constant uint&     n          [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] += block_sums[gid / kScanBlockSize];
}
