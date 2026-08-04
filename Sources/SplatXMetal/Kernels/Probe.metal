#include <metal_stdlib>
using namespace metal;

kernel void probe_atomic_float(device atomic_float* acc [[buffer(0)]],
                               uint tid [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(acc, 1.0f, memory_order_relaxed);
}

kernel void probe_simd_width(device uint* out [[buffer(0)]],
                             uint tid [[thread_position_in_grid]],
                             uint simd_width [[threads_per_simdgroup]]) {
    if (tid == 0) { out[0] = simd_width; }
}

// Sustained-load probe: an ALU-heavy FMA loop (used in Task 10)
kernel void probe_sustained_load(device float* data [[buffer(0)]],
                                 constant uint& n [[buffer(1)]],
                                 uint tid [[thread_position_in_grid]]) {
    float x = data[tid % n];
    for (uint i = 0; i < 4096; i++) { x = fma(x, 1.0000001f, 1e-7f); }
    data[tid % n] = x;
}
