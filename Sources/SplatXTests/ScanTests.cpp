// Verifies the GPU prefix scan (spec 03 §Kernel 2 reduce-then-scan, 3 dispatches: scan_partial →
// scan_blocks_single_group → scan_add_offsets) and isect_offset_encode (§Kernel 4).
// The scan accumulates with promotion from int32 to int64 — integer addition is independent of
// association order, so **exact equality** with the CPU cumsum_i64 is the contract (D9-1, no
// tolerance). Rationale for the chosen sizes: the block (kScanBlockSize=1024) boundary ±1, exact
// occupancy of the single scan_blocks threadgroup (256) (262144 = 256 blocks), and a large 300k+
// case (a partially occupied tail plus a second pass through the scan_blocks carry loop —
// 293 blocks > 256).
// offsets uses the sorted golden keys of isect_std/isect_ties (reassembled from *_sorted lo/hi),
// requiring exact equality with offsets.sxt (both CPU and Metal), and includes a reproduction of
// the example from the original comment at IntersectTile.cu:936–939.
#include "TestHarness.h"
#include "TestUtils.h" // shared_metal_backend / SX_EXACT_I32 / SX_EXACT_U64
#include "splatx/CpuKernels.h"
#include "splatx/Json.h"
#include "splatx/KernelBackend.h"
#include "splatx/SXTensor.h"

#include <cstdint>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace splatx;

namespace {

std::string fx(const std::string& dir, const char* rel) {
    return sxtest::fixture_dir() + "/" + dir + "/" + rel;
}

struct OffsetFixture {
    std::map<std::string, double> meta;
    std::vector<uint64_t> sorted_ids; // reassembled as (uint32)hi << 32 | (uint32)lo (D3 split storage)
    SXTensor offsets;                 // [I*tile_h*tile_w] int32 golden
};

OffsetFixture load_offset_fixture(const std::string& dir) {
    OffsetFixture f;
    f.meta = json_parse_flat(fx(dir, "meta.json"));
    const SXTensor lo = SXTensor::load(fx(dir, "isect_ids_sorted_lo.sxt"));
    const SXTensor hi = SXTensor::load(fx(dir, "isect_ids_sorted_hi.sxt"));
    f.offsets = SXTensor::load(fx(dir, "offsets.sxt"));
    f.sorted_ids.resize(lo.count());
    for (size_t i = 0; i < f.sorted_ids.size(); i++)
        f.sorted_ids[i] =
            (static_cast<uint64_t>(static_cast<uint32_t>(hi.i32()[i])) << 32) |
            static_cast<uint64_t>(static_cast<uint32_t>(lo.i32()[i]));
    return f;
}

IsectOffsetInputs make_offset_inputs(const OffsetFixture& f) {
    IsectOffsetInputs in;
    in.isect_ids = f.sorted_ids.data();
    in.n_isects = static_cast<int64_t>(f.sorted_ids.size());
    in.I = static_cast<int64_t>(f.meta.at("I"));
    in.tile_width = static_cast<uint32_t>(f.meta.at("tile_width"));
    in.tile_height = static_cast<uint32_t>(f.meta.at("tile_height"));
    return in;
}

// A synthetic count array — it mimics the tiles_per_gauss distribution: roughly 30% zeros (culled
// or off screen), many small values, and occasional large ones (×100). The mt19937 seed is fixed
// (identical input for CPU and GPU within the same process is sufficient).
std::vector<int32_t> synth_counts(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> small(0, 9);
    std::vector<int32_t> v(n);
    for (size_t i = 0; i < n; i++) {
        const uint32_t r = rng() % 100u;
        v[i] = (r < 30) ? 0 : (r < 95 ? small(rng) : small(rng) * 100);
    }
    return v;
}

} // namespace

// (1) The Metal GPU 3-dispatch scan vs the CPU cumsum_i64 — exact equality (compared through a u64
//     reinterpretation — the accumulated values are non-negative, so int64 and uint64 are
//     equivalent). The i64 promotion case: the sum 6.0e9 > UINT32_MAX (the promotion point in
//     spec 03).
SX_TEST(prefix_scan_metal_matches_cpu) {
    auto& mtl = sxtest::shared_metal_backend();
    const size_t sizes[] = {1, 1023, 1024, 1025, 262144, 300007};
    for (const size_t n : sizes) {
        const std::vector<int32_t> v = synth_counts(n, static_cast<uint32_t>(n));
        const std::vector<int64_t> cpu = cumsum_i64(v.data(), n);
        const std::vector<int64_t> gpu = mtl.prefix_scan_i64(v.data(), static_cast<int64_t>(n));
        SX_CHECK(gpu.size() == n);
        if (gpu.size() != n) continue;
        SX_EXACT_U64(("scan n=" + std::to_string(n)).c_str(),
                     reinterpret_cast<const uint64_t*>(gpu.data()), gpu.size(),
                     reinterpret_cast<const uint64_t*>(cpu.data()), n);
    }
    { // int32 → int64 promotion: the accumulated sum exceeds even UINT32_MAX (300007 * 20000 = 6.0e9)
        const size_t n = 300007;
        const std::vector<int32_t> v(n, 20000);
        const std::vector<int64_t> cpu = cumsum_i64(v.data(), n);
        SX_CHECK(cpu.back() == 6000140000LL);
        const std::vector<int64_t> gpu = mtl.prefix_scan_i64(v.data(), static_cast<int64_t>(n));
        SX_CHECK(gpu.size() == n);
        if (gpu.size() == n)
            SX_EXACT_U64("scan i64 promotion", reinterpret_cast<const uint64_t*>(gpu.data()),
                         gpu.size(), reinterpret_cast<const uint64_t*>(cpu.data()), n);
    }
    SX_CHECK(mtl.prefix_scan_i64(nullptr, 0).empty()); // n==0 — the dispatch is skipped (returns empty)
}

// (1b) i64 promotion inside a single block (kScanBlockSize=1024) — every element is INT32_MAX. The
//      accumulator exceeds int32 at the 2nd element (4.29e9 > INT32_MAX) and uint32 at the 3rd, so
//      if the block-local accumulator in scan_partial were not i64, this case would collapse. The
//      promotion case in (1) above (6.0e9) exercises the block-sum carry (scan_blocks) path — this
//      test observes promotion **inside** a block, with no carry involved (1b final review
//      carry-over — 1c Task 1 item 4).
SX_TEST(prefix_scan_single_block_int32max) {
    auto& mtl = sxtest::shared_metal_backend();
    const size_t n = 1024; // == kScanBlockSize — a single threadgroup, zero scan_add_offsets carry
    const std::vector<int32_t> v(n, INT32_MAX);
    const std::vector<int64_t> cpu = cumsum_i64(v.data(), n);
    SX_CHECK(cpu.back() == 1024LL * INT32_MAX); // 2199023254528 — confirms it exceeds 32 bits
    const std::vector<int64_t> gpu = mtl.prefix_scan_i64(v.data(), static_cast<int64_t>(n));
    SX_CHECK(gpu.size() == n);
    if (gpu.size() == n)
        SX_EXACT_U64("scan single-block INT32_MAX", reinterpret_cast<const uint64_t*>(gpu.data()),
                     gpu.size(), reinterpret_cast<const uint64_t*>(cpu.data()), n);
}

// (2) Bit-exact determinism of the Metal scan — memcmp across two runs on identical input (an
//     integer pipeline — bit-identical).
SX_TEST(prefix_scan_metal_bitwise_deterministic) {
    auto& mtl = sxtest::shared_metal_backend();
    const size_t n = 300007;
    const std::vector<int32_t> v = synth_counts(n, 7);
    const std::vector<int64_t> a = mtl.prefix_scan_i64(v.data(), static_cast<int64_t>(n));
    const std::vector<int64_t> b = mtl.prefix_scan_i64(v.data(), static_cast<int64_t>(n));
    SX_CHECK(a.size() == n && b.size() == n);
    if (a.size() == n && b.size() == n)
        SX_CHECK(std::memcmp(a.data(), b.data(), n * sizeof(int64_t)) == 0);
}

// (3) CPU isect_offset_encode vs the golden — the example from the IntersectTile.cu:936–939 comment
//     (ids=[1,1,1,3,3], n_tiles=6 → offsets [0,0,3,3,5,5]), plus exact equality with the
//     isect_std/isect_ties offsets.sxt, plus n_isects==0 → all zeros
//     (IntersectTile.cu:1004–1008).
SX_TEST(isect_offset_encode_cpu_matches_golden) {
    const std::unique_ptr<KernelBackend> cpu = make_cpu_backend();
    { // the example from the upstream original comment — tiles 1,1,1,3,3 (I=1, tile 6×1, tile_n_bits=3, depth bits 0)
        const uint64_t keys[5] = {1ull << 32, 1ull << 32, 1ull << 32, 3ull << 32, 3ull << 32};
        IsectOffsetInputs in;
        in.isect_ids = keys;
        in.n_isects = 5;
        in.I = 1;
        in.tile_width = 6;
        in.tile_height = 1;
        const std::vector<int32_t> off = cpu->isect_offset_encode(in);
        const int32_t expect[6] = {0, 0, 3, 3, 5, 5};
        SX_CHECK(off.size() == 6);
        if (off.size() == 6) SX_EXACT_I32("doc-example offsets", off, expect, 6);
    }
    for (const char* dir : {"isect_std", "isect_ties"}) {
        const OffsetFixture f = load_offset_fixture(dir);
        const std::vector<int32_t> off = cpu->isect_offset_encode(make_offset_inputs(f));
        SX_CHECK(off.size() == f.offsets.count());
        if (off.size() == f.offsets.count())
            SX_EXACT_I32((std::string(dir) + " offsets(cpu)").c_str(), off,
                         f.offsets.i32(), f.offsets.count());
    }
    { // n_isects == 0 → filled with zeros, with no dispatch and no kernel
        IsectOffsetInputs in;
        in.n_isects = 0;
        in.I = 2;
        in.tile_width = 3;
        in.tile_height = 2;
        const std::vector<int32_t> off = cpu->isect_offset_encode(in);
        size_t nonzero = 0;
        for (const int32_t o : off)
            if (o != 0) nonzero++;
        SX_CHECK(off.size() == 12 && nonzero == 0);
    }
}

// (4) Metal isect_offset_encode vs the golden — the same contract as the CPU, plus bit-exact
//     determinism, plus the n_isects==0 path.
SX_TEST(isect_offset_encode_metal_matches_golden) {
    auto& mtl = sxtest::shared_metal_backend();
    for (const char* dir : {"isect_std", "isect_ties"}) {
        const OffsetFixture f = load_offset_fixture(dir);
        const IsectOffsetInputs in = make_offset_inputs(f);
        const std::vector<int32_t> a = mtl.isect_offset_encode(in);
        SX_CHECK(a.size() == f.offsets.count());
        if (a.size() == f.offsets.count())
            SX_EXACT_I32((std::string(dir) + " offsets(metal)").c_str(), a,
                         f.offsets.i32(), f.offsets.count());
        const std::vector<int32_t> b = mtl.isect_offset_encode(in); // bit-exact determinism
        SX_CHECK(a.size() == b.size() &&
                 std::memcmp(a.data(), b.data(), a.size() * sizeof(int32_t)) == 0);
    }
    { // n_isects == 0 → filled with zeros, with no dispatch (a host fallback — the upstream original's offsets.fill_(0))
        IsectOffsetInputs in;
        in.n_isects = 0;
        in.I = 1;
        in.tile_width = 4;
        in.tile_height = 4;
        const std::vector<int32_t> off = mtl.isect_offset_encode(in);
        size_t nonzero = 0;
        for (const int32_t o : off)
            if (o != 0) nonzero++;
        SX_CHECK(off.size() == 16 && nonzero == 0);
    }
}
