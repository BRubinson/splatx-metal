// Verifies the GPU radix sort (Task 7 — spec 04) — checks that the brush port in
// MetalBackend::sort_isect (a 5-kernel reduce-then-scan LSD radix with a 4-bit digit) is exactly
// equivalent to the CPU reference (sort_isect_cpu, std::stable_sort). The sort contract
// (KernelBackend.h): stably sort (u64 key, i32 value) pairs by unsigned ascending key, sorting only
// the key_n_bits significant bits (the number of passes = ceil(key_n_bits/4)) — that the high bits
// are 0 is the caller's contract (the GPU does not check it, the CPU throws).
//
// The verification axes (the brief plus the spec 04 determinism checklist):
//  (b) Direct stability verification based on isect_ties — a mandatory handover item from the
//      Task 6 review. After the delegation to the CPU was removed, this is the only test covering
//      the preservation of tie order by the GPU sort (both against the golden and independently of
//      the golden).
//  (c) Determinism repetition — 20 runs on identical input produce the same hash (the spec 04
//      checklist says 100 — reduced to 20 locally, leaving room to raise the repetition count in
//      CI).
//  (d) A large synthetic array (millions of keys, random mixed with ties) matches the CPU
//      stable_sort exactly.
//  (e) Partially occupied threadgroup boundaries — many cases where n is not a multiple of
//      WG (256) or BLOCK (1024) (Correction 6: covers partial occupancy of the final workgroup on
//      the readback plus exact-size path).
// (a) Golden chain regressions for isect_std/isect_ties are covered automatically by the existing
//     chain tests in IsectTests through backend polymorphism (the Task 6 report — no test changes
//     needed).
#include "TestHarness.h"
#include "TestUtils.h" // shared_metal_backend / SX_EXACT_I32 / SX_EXACT_U64
#include "splatx/CpuKernels.h"
#include "splatx/Json.h"
#include "splatx/KernelBackend.h"
#include "splatx/SXTensor.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

using namespace splatx;

namespace {

// splitmix64 — a deterministic RNG with a fixed seed (for generating synthetic keys; the standard
// constants, the same family as the brush tests)
uint64_t splitmix64(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// FNV-1a 64 — an output hash for the repeated determinism comparison (it addresses the spec 04
// checklist item "the same hash across 100 runs")
uint64_t fnv1a(uint64_t h, const void* data, size_t bytes) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < bytes; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

uint64_t hash_outputs(const SortIsectOutputs& out) {
    uint64_t h = 1469598103934665603ull;
    h = fnv1a(h, out.keys_sorted.data(), out.keys_sorted.size() * sizeof(uint64_t));
    h = fnv1a(h, out.values_sorted.data(), out.values_sorted.size() * sizeof(int32_t));
    return h;
}

// Generates synthetic (key, value) pairs — random mixed with ties. When tie_pool > 0, roughly half
// are drawn from the pool to form tie groups. The value is the index i (every value is unique, so
// exact equality of values with the CPU stable_sort is itself the proof of stability — if the sort
// were not stable on tied keys, the value order would necessarily diverge).
struct Synth {
    std::vector<uint64_t> keys;
    std::vector<int32_t> values;
};

Synth make_synth(size_t n, uint32_t key_n_bits, uint64_t seed, size_t tie_pool) {
    const uint64_t mask =
        key_n_bits >= 64 ? ~0ull : ((1ull << key_n_bits) - 1ull);
    Synth s;
    s.keys.resize(n);
    s.values.resize(n);
    uint64_t rng = seed;
    std::vector<uint64_t> pool(tie_pool);
    for (auto& k : pool) k = splitmix64(rng) & mask;
    for (size_t i = 0; i < n; i++) {
        const uint64_t r = splitmix64(rng);
        s.keys[i] = (tie_pool > 0 && (r & 1ull))
                        ? pool[(r >> 1) % tie_pool] // tie groups (about n/(2·pool) members each)
                        : (splitmix64(rng) & mask);
        s.values[i] = static_cast<int32_t>(i);
    }
    return s;
}

// Metal vs CPU (the std::stable_sort reference) must match exactly — both keys and values
// (D9-1 exact integer equality)
void check_metal_matches_cpu(const char* tag, const Synth& s, uint32_t key_n_bits) {
    SortIsectInputs in;
    in.keys = s.keys.data();
    in.values = s.values.data();
    in.n = static_cast<int64_t>(s.keys.size());
    in.key_n_bits = key_n_bits;
    const SortIsectOutputs gpu = sxtest::shared_metal_backend().sort_isect(in);
    const SortIsectOutputs cpu = sort_isect_cpu(in);
    SX_EXACT_U64((std::string(tag) + " keys").c_str(), gpu.keys_sorted,
                 cpu.keys_sorted.data(), cpu.keys_sorted.size());
    SX_EXACT_I32((std::string(tag) + " values").c_str(), gpu.values_sorted,
                 cpu.values_sorted.data(), cpu.values_sorted.size());
}

std::string fx(const std::string& dir, const char* rel) {
    return sxtest::fixture_dir() + "/" + dir + "/" + rel;
}

// SXT does not support int64 → reassemble the lo/hi int32 split storage into u64 (the task-2
// convention, the same as IsectTests)
std::vector<uint64_t> load_u64_pair(const std::string& dir, const char* lo_rel,
                                    const char* hi_rel) {
    const SXTensor lo = SXTensor::load(fx(dir, lo_rel));
    const SXTensor hi = SXTensor::load(fx(dir, hi_rel));
    std::vector<uint64_t> keys(lo.count());
    for (size_t i = 0; i < keys.size(); i++)
        keys[i] = (static_cast<uint64_t>(static_cast<uint32_t>(hi.i32()[i])) << 32) |
                  static_cast<uint64_t>(static_cast<uint32_t>(lo.i32()[i]));
    return keys;
}

} // namespace

// (1) [mandatory handover from the Task 6 review] Metal stability verification based on isect_ties
//     — after the delegation to the CPU was removed, this is the only coverage that directly
//     verifies the GPU sort preserves tie order. Both against the golden and independently of the
//     golden (the Metal counterpart of sort_isect_cpu_stable_matches_golden — the high-bit throw
//     check is excluded: the GPU contract is that it does not check, so the CPU reference test
//     covers that). Includes the n=0 edge case.
SX_TEST(sort_isect_metal_stable_ties) {
    const auto meta = json_parse_flat(fx("isect_ties", "meta.json"));
    const std::vector<uint64_t> emit_keys =
        load_u64_pair("isect_ties", "isect_ids_lo.sxt", "isect_ids_hi.sxt");
    const SXTensor emit_vals = SXTensor::load(fx("isect_ties", "flatten_ids.sxt"));
    const std::vector<uint64_t> golden_keys =
        load_u64_pair("isect_ties", "isect_ids_sorted_lo.sxt", "isect_ids_sorted_hi.sxt");
    const SXTensor golden_vals = SXTensor::load(fx("isect_ties", "flatten_ids_sorted.sxt"));

    SortIsectInputs in;
    in.keys = emit_keys.data();
    in.values = emit_vals.i32();
    in.n = static_cast<int64_t>(emit_keys.size());
    in.key_n_bits = 32 + static_cast<uint32_t>(meta.at("tile_n_bits")) +
                    static_cast<uint32_t>(meta.at("image_n_bits")); // 32+4+0 = 36 → 9 passes
    const SortIsectOutputs out = sxtest::shared_metal_backend().sort_isect(in);
    SX_EXACT_U64("metal sort keys(ties)", out.keys_sorted, golden_keys.data(),
                 golden_keys.size());
    SX_EXACT_I32("metal sort flatten(ties)", out.values_sorted, golden_vals.i32(),
                 golden_vals.count());

    // (i) Golden-independent verification of stability: for key k, the sequence of values in the
    //     sorted output == the sequence of values in the input (emit order) — for every tie group.
    std::map<uint64_t, std::vector<int32_t>> emit_order;
    for (size_t i = 0; i < emit_keys.size(); i++)
        emit_order[emit_keys[i]].push_back(emit_vals.i32()[i]);
    size_t max_group = 0;
    for (const auto& [key, vals] : emit_order) max_group = std::max(max_group, vals.size());
    // The fixture contract: ties must actually exist for the stability verification to be
    // meaningful — only a lower bound is required (>= 3). The exact value (14 in the current
    // fixture) can change when the fixture is regenerated, so it is not hardcoded.
    SX_CHECK(max_group >= 3);
    std::map<uint64_t, size_t> cursor;
    size_t stable_violations = 0;
    for (size_t i = 0; i < out.keys_sorted.size(); i++) {
        const std::vector<int32_t>& seq = emit_order.at(out.keys_sorted[i]);
        if (out.values_sorted[i] != seq[cursor[out.keys_sorted[i]]++]) stable_violations++;
    }
    SX_CHECK(stable_violations == 0);

    // (ii) The postcondition that keys are in unsigned ascending order
    size_t order_violations = 0;
    for (size_t i = 1; i < out.keys_sorted.size(); i++)
        if (out.keys_sorted[i - 1] > out.keys_sorted[i]) order_violations++;
    SX_CHECK(order_violations == 0);

    // n=0 — empty input and output, with no dispatch (the upstream n_isects==0 contract)
    const SortIsectOutputs empty = sxtest::shared_metal_backend().sort_isect(SortIsectInputs{});
    SX_CHECK(empty.keys_sorted.empty() && empty.values_sorted.empty());
}

// (2) Determinism repetition — 20 runs on identical input produce an exactly identical output hash
//     (a locally reduced version of the spec 04 determinism checklist item "100 runs on identical
//     input → the same hash" — the repetition count can be raised to 100 in CI. Parallel sort bugs
//     can manifest intermittently — on failure, whether it reproduces is itself diagnostic
//     information).
//     The radix sort itself has no global atomics (only threadgroup atomic summation — integer
//     addition, independent of order), so bit-exact determinism must be structurally guaranteed
//     (spec 04 §Thread and memory layout).
SX_TEST(sort_isect_metal_deterministic_repeat) {
    const Synth s = make_synth(300000, 40, 0xD15EA5Eull, 512); // random mixed with ties, 10 passes
    SortIsectInputs in;
    in.keys = s.keys.data();
    in.values = s.values.data();
    in.n = static_cast<int64_t>(s.keys.size());
    in.key_n_bits = 40;
    const uint64_t h0 = hash_outputs(sxtest::shared_metal_backend().sort_isect(in));
    size_t mismatches = 0;
    for (int rep = 1; rep < 20; rep++)
        if (hash_outputs(sxtest::shared_metal_backend().sort_isect(in)) != h0) mismatches++;
    SX_CHECK(mismatches == 0);
}

// (3) A large synthetic array (millions of keys) — random mixed with ties, matching the CPU
//     stable_sort exactly in both keys and values at the real-world isect width (40 bits =
//     10 passes). Because the value is the index, matching values is itself the proof of
//     stability.
//     A rough performance figure is recorded (not a gate — 1b is the correctness stage, D11): the
//     Metal time is wall time including upload, encoding, and readback.
SX_TEST(sort_isect_metal_large_matches_cpu) {
    // Pipeline warm-up (separates the PSO creation cost of the first call from the measurement)
    { const Synth w = make_synth(1024, 40, 1, 0); check_metal_matches_cpu("warmup", w, 40); }

    const size_t n = 3000000;
    const Synth s = make_synth(n, 40, 0xC0FFEEull, 4096); // tie groups averaging ~366 members each
    SortIsectInputs in;
    in.keys = s.keys.data();
    in.values = s.values.data();
    in.n = static_cast<int64_t>(n);
    in.key_n_bits = 40;

    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    const SortIsectOutputs gpu = sxtest::shared_metal_backend().sort_isect(in);
    const auto t1 = clk::now();
    const SortIsectOutputs cpu = sort_isect_cpu(in);
    const auto t2 = clk::now();
    const double gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::printf("  [perf] n=%zu key_n_bits=40 (10 pass): metal %.1f ms (including upload+readback), "
                "cpu stable_sort %.1f ms\n",
                n, gpu_ms, cpu_ms);

    SX_EXACT_U64("large keys", gpu.keys_sorted, cpu.keys_sorted.data(),
                 cpu.keys_sorted.size());
    SX_EXACT_I32("large values", gpu.values_sorted, cpu.values_sorted.data(),
                 cpu.values_sorted.size());
}

// (4) Full-width 64-bit keys (key_n_bits=64 → all 16 passes, including the shift=60 top digit) —
//     covers digit extraction from the upper 32 bits and the u64 padding key (~0) path as well.
//     Ties are mixed in to confirm stability is preserved.
SX_TEST(sort_isect_metal_full64_matches_cpu) {
    const Synth s = make_synth(1000000, 64, 0xFEED5EEDull, 1024);
    check_metal_matches_cpu("full64", s, 64);
}

// (5) Partially occupied threadgroup boundaries — many cases where n is not a multiple of WG (256)
//     or BLOCK (1024) (partial occupancy of the final workgroup plus full padding across the EPT
//     iterations). This is the boundary coverage for the "actual count size" path (adopted in D9)
//     that Correction 6 flagged. key_n_bits also cycles through 4, 13, 36, and 64, so bit widths
//     that are not multiples of 4 (13 → 4 passes) and extremely small bit widths (4 → 1 pass, with
//     massive numbers of ties) are covered together.
SX_TEST(sort_isect_metal_partial_occupancy) {
    const size_t sizes[] = {1,    2,    3,    255,  256,  257,  511,   512,  513,
                            1023, 1024, 1025, 2047, 2048, 2049, 4095,  4096, 4097,
                            16383, 16384, 16385, 17 * 1024 - 1, 17 * 1024 + 1};
    const uint32_t bit_widths[] = {4, 13, 36, 64};
    int case_idx = 0;
    for (const size_t n : sizes) {
        const uint32_t bits = bit_widths[case_idx % 4];
        const Synth s = make_synth(n, bits, 0xB0B0ull + case_idx, n > 64 ? 32 : 0);
        char tag[64];
        std::snprintf(tag, sizeof(tag), "occupancy n=%zu bits=%u", n, bits);
        check_metal_matches_cpu(tag, s, bits);
        case_idx++;
    }
}
