// RadixSort.metal — radix_sort_count / radix_sort_reduce / radix_sort_scan /
// radix_sort_scan_add / radix_sort_scatter (spec 04, D4 naming).
// Derived from ArthurBrussee/brush (Apache-2.0), commit 3b80985.
// Reimplemented for Metal/C++.
// Original: crates/brush-sort/src/kernels.rs (5 kernels: sort_count :28–65, sort_reduce :67–124,
//           sort_scan :126–212, sort_scan_add :214–300, sort_scatter :302–443 — a reduce-then-scan
//           of the FidelityFX Parallel Sort family, by way of compute-shader-101 PR#31); the host
//           pass loop is crates/brush-sort/src/lib.rs:16–125 (MetalBackend::sort_isect).
//
// A multi-pass LSD radix sort with a 4-bit digit — the number of passes = ceil(key_n_bits/4)
// (owned by the host).
// The keys are u64 (ulong — the spec 03 D3 unsigned contract; the brush original uses u32 keys in
// 2 stages, but the reference path of this pipeline is the gsplat-style single u64 key — only the
// digit extraction is widened to a ulong shift; the structure is identical).
// The values are uint (preserving the host i32 bits).
//
// Stability guarantee (spec 04 §The brush kernel formulation): count writes bin-major
// (counts[digit*num_wgs+wg]) → a linear exclusive scan yields global offsets in (ascending digit,
// ascending workgroup) order → the EPT iteration order of scatter × the local 2-bit split ranking
// (in thread index order) = stable overall.
// OneSweep and decoupled-lookback are forbidden: the global scan is 3 stages — reduce → a single
// threadgroup scan → scan_add — with no assumption of forward progress between threadgroups
// (spec 04 §Rationale for the prohibition).
//
// Determinism: no global atomics — only threadgroup atomic histograms (order-independent integer
// addition) and simd scans → bit-identical output for identical input (spec 04 §Thread and memory
// configuration).
//
// Simplifications decided relative to brush (spec 04 §Constants and parameters):
//  - the subgroup width is fixed at 32 (D8-3: the host pipeline() enforces
//    threadExecutionWidth==32) → kRadixNumSimdgroups = 8 <= 32, so only the
//    `num_subgroups <= PLANE_DIM` branch of brush is ported (the serial thread-0 fallback path is
//    unreachable — omitted), and the MAX_SUBGROUPS padding is unnecessary.
//  - num_keys comes from params (an exact size — spec 04 Correction 6 + the D9 readback path), not
//    from a GPU buffer [1]. The group_id >= num_wgs gate is retained for parity with the original
//    (it never fires under exact-size dispatch).
//
// The barrier placement follows Verification Notes Corrections 1–3 of spec 04 plus the second
// reinforcement, and is spelled out in the comments in 1:1 correspondence with the original line
// numbers (kernels.rs) — 5 barriers per scan chunk, 3 per scatter split, a barrier separating the
// register pre-read of bin_offset_cache at the end of each scatter iteration, the zero-init of
// local_histogram at the start of every EPT iteration (its visibility relies on the barrier inside
// the split loop — reinforcement 1), and one barrier immediately after the initial load of
// bin_offset_cache (:332–334).
#include <metal_stdlib>
#include "KernelConstants.h"
#include "KernelParams.h"
using namespace metal;

// Extracts the 4-bit digit of the current pass from a u64 key (kernels.rs:55/:353 — the ulong
// widening of the u32 `>> shift & 0xf`; shift ∈ {0,4,...,60}).
static inline uint radix_digit(const ulong key, const uint shift)
{
    return uint((key >> shift) & 0xFul);
}

// Kernel 1: histogram — BLOCK_SIZE(1024) keys per workgroup, bin-major output.
// (sort_count_kernel, kernels.rs:28–65)
kernel void radix_sort_count(
    device const ulong*       src_keys [[buffer(0)]], // [num_keys] input keys of the current pass
    device uint*              counts   [[buffer(1)]], // [BIN_COUNT * num_wgs] bin-major
    constant RadixSortParams& p        [[buffer(2)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    if (gid >= p.num_wgs) return; // :40 gate (never fires under exact-size dispatch — parity)

    threadgroup atomic_uint histogram[kRadixBinCount];
    if (lid < kRadixBinCount)
        atomic_store_explicit(&histogram[lid], 0u, memory_order_relaxed); // :45–47
    threadgroup_barrier(mem_flags::mem_threadgroup);                      // :48

    uint data_index = kRadixBlockSize * gid + lid; // :50–51
    for (uint e = 0; e < kRadixElemsPerThread; e++) {
        if (data_index < p.num_keys)
            atomic_fetch_add_explicit(&histogram[radix_digit(src_keys[data_index], p.shift)],
                                      1u, memory_order_relaxed); // :54–56
        data_index += kRadixWG;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // :60
    if (lid < kRadixBinCount)                        // bin-major — the premise of the stable LSD (:63)
        counts[lid * p.num_wgs + gid] =
            atomic_load_explicit(&histogram[lid], memory_order_relaxed);
}

// Kernel 2: reduce — the per-bin BLOCK_SIZE chunk sums of counts → reduced[num_reduce_wgs].
// (sort_reduce_kernel, kernels.rs:67–124)
kernel void radix_sort_reduce(
    device const uint*        counts  [[buffer(0)]],
    device uint*              reduced [[buffer(1)]], // [align(num_reduce_wgs, BLOCK_SIZE)]
    constant RadixSortParams& p       [[buffer(2)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint lid       [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    if (gid >= p.num_reduce_wgs) return; // :78 gate

    const uint num_reduce_wg_per_bin = p.num_reduce_wgs / kRadixBinCount; // :82
    const uint bin_offset = (gid / num_reduce_wg_per_bin) * p.num_wgs;    // :83–85
    const uint base_index = (gid % num_reduce_wg_per_bin) * kRadixBlockSize; // :86

    uint sum = 0;
    for (uint i = 0; i < kRadixElemsPerThread; i++) { // :89–94
        const uint data_index = base_index + i * kRadixWG + lid;
        if (data_index < p.num_wgs) sum += counts[bin_offset + data_index];
    }
    const uint subgroup_sum = simd_sum(sum); // :96 plane_sum

    threadgroup uint partials[kRadixNumSimdgroups];
    if (simd_lane == 0) partials[simd_id] = subgroup_sum; // :102–104
    threadgroup_barrier(mem_flags::mem_threadgroup);      // :105

    // num_subgroups(8) <= the simd width(32) — only the brush :107–116 branch (the fallback is unreachable)
    if (simd_id == 0) {
        const uint v = (simd_lane < kRadixNumSimdgroups) ? partials[simd_lane] : 0u;
        const uint total = simd_sum(v);
        if (simd_lane == 0) reduced[gid] = total; // :114–115
    }
}

// Kernel 3: global exclusive scan — **MUST be dispatchThreadgroups(1)** (a single threadgroup walks
// all of reduced in a BLOCK_SIZE chunk loop, with the carry in a thread register — Correction 4).
// 5 barriers per chunk (Correction 1 — kernels.rs:152, 165, 190, 197, 207).
// (sort_scan_kernel, kernels.rs:126–212)
kernel void radix_sort_scan(
    device uint*              reduced [[buffer(0)]], // in-place: chunk sums → global exclusive
    constant RadixSortParams& p       [[buffer(1)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint partials[kRadixNumSimdgroups];
    threadgroup uint lds[kRadixBlockSize];
    threadgroup uint chunk_total;

    uint carry = 0; // thread register (Correction 4 — :139; every thread converges to the same value)
    // The multi-chunk path (num_reduce_wgs > BLOCK_SIZE ⇔ n > ~67M keys) is not verified locally —
    // porting the 67M+ regression test of brush lib.rs:291–339 is under consideration
    // (spec 04 open question 6).
    for (uint chunk_start = 0; chunk_start < p.num_reduce_wgs;
         chunk_start += kRadixBlockSize) {
        // Transposed load: laid out as lds[row*WG+col] so each thread serially scans EPT contiguous elements
        for (uint i = 0; i < kRadixElemsPerThread; i++) { // :142–151
            const uint data_index = chunk_start + i * kRadixWG + lid;
            const uint col = (i * kRadixWG + lid) / kRadixElemsPerThread;
            const uint row = (i * kRadixWG + lid) % kRadixElemsPerThread;
            lds[row * kRadixWG + col] =
                (data_index < p.num_reduce_wgs) ? reduced[data_index] : 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // barrier 1/5 (:152) — after the load

        uint thread_sum = 0; // serial exclusive scan of 4 contiguous elements per thread (:154–159)
        for (uint i = 0; i < kRadixElemsPerThread; i++) {
            const uint tmp = lds[i * kRadixWG + lid];
            lds[i * kRadixWG + lid] = thread_sum;
            thread_sum += tmp;
        }

        const uint sg_inclusive = simd_prefix_inclusive_sum(thread_sum); // :161
        if (simd_lane == kRadixSimdWidth - 1) partials[simd_id] = sg_inclusive; // :162–164
        threadgroup_barrier(mem_flags::mem_threadgroup); // barrier 2/5 (:165) — after storing partials

        if (simd_id == 0) { // :166–180 (the num_subgroups <= PLANE_DIM path)
            const uint v = (simd_lane < kRadixNumSimdgroups) ? partials[simd_lane] : 0u;
            const uint scanned = simd_prefix_exclusive_sum(v);
            if (simd_lane < kRadixNumSimdgroups) partials[simd_lane] = scanned;
            if (simd_lane == kRadixNumSimdgroups - 1) chunk_total = scanned + v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // barrier 3/5 (:190) — after the scan and total

        const uint workgroup_exclusive = partials[simd_id] + sg_inclusive - thread_sum; // :192
        const uint base = carry + workgroup_exclusive;                                  // :193
        for (uint i = 0; i < kRadixElemsPerThread; i++) lds[i * kRadixWG + lid] += base;
        threadgroup_barrier(mem_flags::mem_threadgroup); // barrier 4/5 (:197) — after the rewrite

        for (uint i = 0; i < kRadixElemsPerThread; i++) { // :199–206 re-read the transposed positions → store
            const uint data_index = chunk_start + i * kRadixWG + lid;
            const uint col = (i * kRadixWG + lid) / kRadixElemsPerThread;
            const uint row = (i * kRadixWG + lid) % kRadixElemsPerThread;
            if (data_index < p.num_reduce_wgs) reduced[data_index] = lds[row * kRadixWG + col];
        }
        // barrier 5/5 (:207) — end of the chunk: guarantees this happens before the next chunk
        // overwrites lds/partials/chunk_total, and after every thread has read chunk_total (:209)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        carry += chunk_total; // :209
    }
}

// Kernel 4: scan_add — locally exclusive-scans each BLOCK_SIZE chunk of counts and adds
// reduced[gid] (the global base of the chunk) → counts becomes the global exclusive offsets.
// (sort_scan_add_kernel, kernels.rs:214–300)
kernel void radix_sort_scan_add(
    device const uint*        reduced [[buffer(0)]],
    device uint*              counts  [[buffer(1)]], // in-place: bin-major counts → offsets
    constant RadixSortParams& p       [[buffer(2)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint lid       [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    if (gid >= p.num_reduce_wgs) return; // :225 gate

    const uint num_reduce_wg_per_bin = p.num_reduce_wgs / kRadixBinCount; // :229
    const uint bin_offset = (gid / num_reduce_wg_per_bin) * p.num_wgs;    // :230–231
    const uint base_index =
        (gid % num_reduce_wg_per_bin) * kRadixElemsPerThread * kRadixWG;  // :232

    threadgroup uint partials[kRadixNumSimdgroups];
    threadgroup uint lds[kRadixBlockSize];

    for (uint i = 0; i < kRadixElemsPerThread; i++) { // :237–248 transposed load (explicit OOB gate)
        const uint data_index = base_index + i * kRadixWG + lid;
        const uint col = (i * kRadixWG + lid) / kRadixElemsPerThread;
        const uint row = (i * kRadixWG + lid) % kRadixElemsPerThread;
        lds[row * kRadixWG + col] =
            (data_index < p.num_wgs) ? counts[bin_offset + data_index] : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // :249

    uint thread_sum = 0; // :251–256 per-thread serial exclusive scan
    for (uint i = 0; i < kRadixElemsPerThread; i++) {
        const uint tmp = lds[i * kRadixWG + lid];
        lds[i * kRadixWG + lid] = thread_sum;
        thread_sum += tmp;
    }

    const uint sg_inclusive = simd_prefix_inclusive_sum(thread_sum);           // :261
    if (simd_lane == kRadixSimdWidth - 1) partials[simd_id] = sg_inclusive;    // :262–264
    threadgroup_barrier(mem_flags::mem_threadgroup);                           // :265

    if (simd_id == 0) { // :266–275
        const uint v = (simd_lane < kRadixNumSimdgroups) ? partials[simd_lane] : 0u;
        const uint scanned = simd_prefix_exclusive_sum(v);
        if (simd_lane < kRadixNumSimdgroups) partials[simd_lane] = scanned;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup); // :284

    const uint workgroup_exclusive = partials[simd_id] + sg_inclusive - thread_sum; // :286
    const uint total_base = reduced[gid] + workgroup_exclusive;                     // :287
    for (uint i = 0; i < kRadixElemsPerThread; i++) lds[i * kRadixWG + lid] += total_base;
    threadgroup_barrier(mem_flags::mem_threadgroup); // :291

    for (uint i = 0; i < kRadixElemsPerThread; i++) { // :292–299 re-read transposed → store
        const uint data_index = base_index + i * kRadixWG + lid;
        const uint col = (i * kRadixWG + lid) / kRadixElemsPerThread;
        const uint row = (i * kRadixWG + lid) % kRadixElemsPerThread;
        if (data_index < p.num_wgs) counts[bin_offset + data_index] = lds[row * kRadixWG + col];
    }
}

// Kernel 5: scatter — processes BLOCK_SIZE keys per workgroup in EPT=4 iterations (WG=256 keys per
// iteration): a local stable sort (the 4-bit digit is split twice, 2 bits at a time — the packed
// 8-bit counter trick, which assumes WG=256), then a scatter to the bin-major global offsets.
// (sort_scatter_kernel, kernels.rs:302–443)
// Out-of-range lanes pad local_key with an all-ones ulong (its digit is always 0xF — in the local
// sort it is pushed behind the real keys, and the global write drops out naturally via the
// total_offset < num_keys gate :343/:432).
kernel void radix_sort_scatter(
    device const ulong*       src_keys [[buffer(0)]],
    device const uint*        src_vals [[buffer(1)]],
    device const uint*        counts   [[buffer(2)]], // global exclusive offsets (bin-major)
    device ulong*             dst_keys [[buffer(3)]],
    device uint*              dst_vals [[buffer(4)]],
    constant RadixSortParams& p        [[buffer(5)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint lid       [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]])
{
    if (gid >= p.num_wgs) return; // :316 gate

    threadgroup ulong lds_keys[kRadixWG];               // 2KB (u64 keys)
    threadgroup uint  lds_values[kRadixWG];             // 1KB
    threadgroup uint  lds_scratch[kRadixBinCount];      // histogram inclusive scan
    threadgroup uint  bin_offset_cache[kRadixBinCount]; // per-digit global cursor (advances each iteration)
    threadgroup atomic_uint local_histogram[kRadixBinCount];
    threadgroup uint  partials[kRadixNumSimdgroups];
    threadgroup uint  chunk_total;
    // brush over-allocates lds_scratch/bin_offset_cache at WG(256) — here only the 16 actually used
    // are allocated (spec 04 §Thread and memory configuration, "actual use is 16").
    // Total ≈ 3.3KB ≪ 32KB.

    if (lid < kRadixBinCount)
        bin_offset_cache[lid] = counts[lid * p.num_wgs + gid]; // :331–333
    threadgroup_barrier(mem_flags::mem_threadgroup); // :334 — right after the initial load (second reinforcement 1)

    uint data_index = kRadixBlockSize * gid + lid; // :335–337
    for (uint e = 0; e < kRadixElemsPerThread; e++) {
        // Zero-init of local_histogram — at the **start of every** EPT iteration (:338–341). There
        // is no dedicated barrier: visibility relies on the barrier inside the split loop
        // (corresponding to :361) (second reinforcement 1 — if you change the split structure, you
        // MUST re-confirm that a barrier exists between the zero-init and the accumulation at :404).
        if (lid < kRadixBinCount)
            atomic_store_explicit(&local_histogram[lid], 0u, memory_order_relaxed);

        ulong local_key = ~0ul; // padding — digit = 0xF in every pass (:343)
        uint local_value = 0u;
        if (data_index < p.num_keys) { // :346–349
            local_key = src_keys[data_index];
            local_value = src_vals[data_index];
        }

        // Local stable sort: 2-bit split × 2 (:351–401). 3 barriers per split (Correction 2).
        for (uint bit_shift = 0; bit_shift < kRadixBitsPerPass; bit_shift += 2) {
            const uint bit_key = (radix_digit(local_key, p.shift) >> bit_shift) & 3u; // :353–354
            // Packed 8-bit counters: ranks 4 bins simultaneously inside a u32 — a single simd scan
            // (:355–357). Assumes WG=256: the ranks and partial sums are ≤ 256, so the carry/borrow
            // cancels (spec 04 reinforcement).
            const uint packed_input = 1u << (bit_key * 8u);
            const uint sg_inclusive = simd_prefix_inclusive_sum(packed_input);
            if (simd_lane == kRadixSimdWidth - 1) partials[simd_id] = sg_inclusive; // :358–360
            threadgroup_barrier(mem_flags::mem_threadgroup); // split 1/3 (:361)

            if (simd_id == 0) { // :362–376
                const uint v = (simd_lane < kRadixNumSimdgroups) ? partials[simd_lane] : 0u;
                const uint scanned = simd_prefix_exclusive_sum(v);
                if (simd_lane < kRadixNumSimdgroups) partials[simd_lane] = scanned;
                if (simd_lane == kRadixNumSimdgroups - 1) chunk_total = scanned + v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup); // split 2/3 (:386)

            const uint total = chunk_total; // :388
            // the start of byte lane b = Σ_{a<b} total_a (:389)
            const uint bin_offsets = (total << 8u) + (total << 16u) + (total << 24u);
            const uint exclusive_at_thread =
                partials[simd_id] + sg_inclusive - packed_input;                  // :390
            const uint key_offset =
                ((bin_offsets + exclusive_at_thread) >> (bit_key * 8u)) & 0xFFu;  // :391–392

            lds_keys[key_offset] = local_key;   // :394 — within one bit_key the rank follows
            lds_values[key_offset] = local_value; // thread index order → stable
            threadgroup_barrier(mem_flags::mem_threadgroup); // split 3/3 (:396)
            local_key = lds_keys[lid];     // :397–398
            local_value = lds_values[lid];
        }

        // Accumulate the local histogram → inclusive scan → global scatter (:403–435)
        const uint key_index = radix_digit(local_key, p.shift); // :403
        atomic_fetch_add_explicit(&local_histogram[key_index], 1u, memory_order_relaxed); // :404
        threadgroup_barrier(mem_flags::mem_threadgroup); // :405

        // PLANE_DIM(32) >= BIN_COUNT(16) — only the brush :407–416 path (the serial fallback is unreachable)
        if (simd_id == 0) {
            const uint v = (simd_lane < kRadixBinCount)
                               ? atomic_load_explicit(&local_histogram[simd_lane],
                                                      memory_order_relaxed)
                               : 0u;
            const uint inclusive = simd_prefix_inclusive_sum(v);
            if (simd_lane < kRadixBinCount) lds_scratch[simd_lane] = inclusive;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup); // :424

        const uint global_offset = bin_offset_cache[key_index]; // :425 — pre-read into a register
        threadgroup_barrier(mem_flags::mem_threadgroup); // :426 — separates the pre-read from the update (Correction 3)

        uint local_offset = lid; // :427–430 — the local rank within the digit (lid >= scratch[d-1] guaranteed)
        if (key_index > 0u) local_offset -= lds_scratch[key_index - 1u];
        const uint total_offset = global_offset + local_offset; // :431
        if (total_offset < p.num_keys) { // :432 — the gate that naturally drops the padding lanes
            dst_keys[total_offset] = local_key;
            dst_vals[total_offset] = local_value;
        }
        if (lid < kRadixBinCount) // :436–439 — advance the global cursor for the next iteration
            bin_offset_cache[lid] +=
                atomic_load_explicit(&local_histogram[lid], memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup); // :440 — end of the iteration
        data_index += kRadixWG; // :441
    }
}
