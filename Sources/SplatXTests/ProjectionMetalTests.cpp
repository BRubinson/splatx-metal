// Verifies projection_fused_fwd/bwd — spec 01 F0–F8 / B0–B8 (read together with the corrections in
// the verification notes), PINHOLE only (disposition #3).
// The authoritative fixtures (task-3): proj_fused_qs/{c1,c2} (the quat/scale path, seed 11,
//                                      grad 1011),
//                                      proj_fully_fused (the covars path, seed 7,
//                                      covars_triu [N,6], fwd only).
// The comparison convention (07 Correction 3 / D3): radii is exact, and the remaining fwd outputs
// are compared only under the valid=(radii>0).all(-1) mask (the culled slots are unwritten garbage —
// the kernel contract). meta.json is the single source of truth for tolerances.
// The proj_fused_qs golden covers only the opacities=None path (a task-3 concern — the
// opacity-aware bbox is not in the golden).
// bwd is verified by comparing directly against the autograd golden (00-index §6 — a backward CPU
// reference is not required).
#include "TestHarness.h"
#include "TestUtils.h" // shared_metal_backend / rel_l2 / count_bitwise_mismatch / SX_EXACT_I32
#include "splatx/CpuKernels.h"
#include "splatx/Json.h"
#include "splatx/KernelBackend.h"
#include "splatx/SXTensor.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace splatx;

namespace {

std::string fx(const std::string& dir, const char* rel) {
    return sxtest::fixture_dir() + "/" + dir + "/" + rel;
}

struct FwdGolden {
    SXTensor radii, means2d, depths, conics, comps;
};

FwdGolden load_golden(const std::string& dir) {
    return {SXTensor::load(fx(dir, "radii.sxt")), SXTensor::load(fx(dir, "means2d.sxt")),
            SXTensor::load(fx(dir, "depths.sxt")), SXTensor::load(fx(dir, "conics.sxt")),
            SXTensor::load(fx(dir, "compensations.sxt"))};
}

// meta.json scalars → ProjectionFusedInputs (the caller fills in the tensor pointers).
ProjectionFusedInputs make_inputs(const std::map<std::string, double>& meta) {
    ProjectionFusedInputs in;
    in.N = static_cast<int64_t>(meta.at("N"));
    in.C = static_cast<int64_t>(meta.at("C"));
    in.width = static_cast<int>(meta.at("width"));
    in.height = static_cast<int>(meta.at("height"));
    in.eps2d = static_cast<float>(meta.at("eps2d"));
    in.near_plane = static_cast<float>(meta.at("near_plane"));
    in.far_plane = static_cast<float>(meta.at("far_plane"));
    in.radius_clip = static_cast<float>(meta.at("radius_clip"));
    in.camera_model = static_cast<uint32_t>(meta.at("camera_model"));
    in.calc_compensations = meta.at("calc_compensations") != 0.0;
    return in;
}

// valid = (radii > 0).all(-1) — a [C,N] uchar mask (07 Correction 3).
std::vector<unsigned char> valid_mask(const int32_t* radii, size_t CN) {
    std::vector<unsigned char> v(CN);
    for (size_t i = 0; i < CN; i++)
        v[i] = (radii[i * 2] > 0 && radii[i * 2 + 1] > 0) ? 1 : 0;
    return v;
}

// The fwd outputs vs the golden: radii is exact, the rest is allclose under the valid mask (using
// the meta.json tolerances).
// compensations is compared only for fixtures with calc_compensations=1 — when it is 0,
// out.compensations is an empty vector and even calling .data() is UB (a guard flagged in the
// Phase 1a final review).
void compare_fwd_golden(const char* tag, const ProjectionFusedOutputs& out, const FwdGolden& g,
                        const std::map<std::string, double>& meta) {
    const size_t CN = g.depths.count();
    SX_EXACT_I32((std::string(tag) + " radii").c_str(), out.radii, g.radii.i32(),
                 CN * 2);

    const std::vector<unsigned char> valid = valid_mask(g.radii.i32(), CN);
    size_t nvalid = 0;
    for (unsigned char m : valid) nvalid += m;
    SX_CHECK(nvalid > 0 && nvalid < CN); // the fixture contract: culling is mixed in (task-3)

    SX_ALL_CLOSE_MASKED(out.means2d.data(), g.means2d.f32(), CN * 2, meta, "means2d",
                        valid.data(), 2);
    SX_ALL_CLOSE_MASKED(out.depths.data(), g.depths.f32(), CN, meta, "depths", valid.data(), 1);
    SX_ALL_CLOSE_MASKED(out.conics.data(), g.conics.f32(), CN * 3, meta, "conics",
                        valid.data(), 3);
    if (meta.at("calc_compensations") != 0.0)
        SX_ALL_CLOSE_MASKED(out.compensations.data(), g.comps.f32(), CN, meta, "compensations",
                            valid.data(), 1);
}

} // namespace

// (1) The CPU quat/scale path vs the proj_fused_qs fwd golden (c1, c2) — brief Step 1-(1).
SX_TEST(projection_fused_fwd_cpu_qs_matches_golden) {
    for (const char* variant : {"proj_fused_qs/c1", "proj_fused_qs/c2"}) {
        auto meta = json_parse_flat(fx(variant, "meta.json"));
        SXTensor means = SXTensor::load(fx(variant, "means.sxt"));
        SXTensor quats = SXTensor::load(fx(variant, "quats.sxt"));
        SXTensor scales = SXTensor::load(fx(variant, "scales.sxt"));
        SXTensor viewmats = SXTensor::load(fx(variant, "viewmats.sxt"));
        SXTensor Ks = SXTensor::load(fx(variant, "Ks.sxt"));
        FwdGolden g = load_golden(variant);

        ProjectionFusedInputs in = make_inputs(meta);
        in.means = means.f32();
        in.quats = quats.f32(); // covars == nullptr → the quat/scale path
        in.scales = scales.f32();
        in.viewmats = viewmats.f32();
        in.Ks = Ks.f32();

        ProjectionFusedOutputs out = projection_fused_fwd_cpu(in);
        compare_fwd_golden(variant, out, g, meta);
    }
}

// (2) The covars (6-triu) path vs the proj_fully_fused golden — on both the CPU and Metal backends
// (brief Step 1-(2)).
SX_TEST(projection_fused_fwd_covars_matches_golden) {
    const std::string dir = "proj_fully_fused";
    auto meta = json_parse_flat(fx(dir, "meta.json"));
    SXTensor means = SXTensor::load(fx(dir, "means.sxt"));
    SXTensor covars6 = SXTensor::load(fx(dir, "covars_triu.sxt")); // [N,6] (07 Correction 5)
    SXTensor viewmats = SXTensor::load(fx(dir, "viewmats.sxt"));
    SXTensor Ks = SXTensor::load(fx(dir, "Ks.sxt"));
    FwdGolden g = load_golden(dir);

    ProjectionFusedInputs in = make_inputs(meta);
    in.means = means.f32();
    in.covars = covars6.f32(); // covars != nullptr → the kProjUseCovars path
    in.viewmats = viewmats.f32();
    in.Ks = Ks.f32();

    ProjectionFusedOutputs cpu = projection_fused_fwd_cpu(in);
    compare_fwd_golden("covars/cpu", cpu, g, meta);

    ProjectionFusedOutputs mtl = sxtest::shared_metal_backend().projection_fused_fwd(in);
    compare_fwd_golden("covars/metal", mtl, g, meta);
}

// (3) The Metal quat/scale path vs the proj_fused_qs golden (c1: C=1, c2: C=2) — brief Step 1-(3).
SX_TEST(projection_fused_fwd_metal_qs_matches_golden) {
    for (const char* variant : {"proj_fused_qs/c1", "proj_fused_qs/c2"}) {
        auto meta = json_parse_flat(fx(variant, "meta.json"));
        SXTensor means = SXTensor::load(fx(variant, "means.sxt"));
        SXTensor quats = SXTensor::load(fx(variant, "quats.sxt"));
        SXTensor scales = SXTensor::load(fx(variant, "scales.sxt"));
        SXTensor viewmats = SXTensor::load(fx(variant, "viewmats.sxt"));
        SXTensor Ks = SXTensor::load(fx(variant, "Ks.sxt"));
        FwdGolden g = load_golden(variant);

        ProjectionFusedInputs in = make_inputs(meta);
        in.means = means.f32();
        in.quats = quats.f32();
        in.scales = scales.f32();
        in.viewmats = viewmats.f32();
        in.Ks = Ks.f32();

        ProjectionFusedOutputs out = sxtest::shared_metal_backend().projection_fused_fwd(in);
        compare_fwd_golden(variant, out, g, meta);
    }
}

// (4) Bit-exact determinism of the C=1 path — two runs on identical input match under memcmp, over
//     all radii slots and, for the remaining outputs, over the valid slots only (the culled slots
//     are unwritten garbage, so they are excluded from the comparison — D3).
SX_TEST(projection_fused_fwd_metal_c1_bit_deterministic) {
    const std::string dir = "proj_fused_qs/c1";
    auto meta = json_parse_flat(fx(dir, "meta.json"));
    SXTensor means = SXTensor::load(fx(dir, "means.sxt"));
    SXTensor quats = SXTensor::load(fx(dir, "quats.sxt"));
    SXTensor scales = SXTensor::load(fx(dir, "scales.sxt"));
    SXTensor viewmats = SXTensor::load(fx(dir, "viewmats.sxt"));
    SXTensor Ks = SXTensor::load(fx(dir, "Ks.sxt"));

    ProjectionFusedInputs in = make_inputs(meta);
    in.means = means.f32();
    in.quats = quats.f32();
    in.scales = scales.f32();
    in.viewmats = viewmats.f32();
    in.Ks = Ks.f32();

    ProjectionFusedOutputs a = sxtest::shared_metal_backend().projection_fused_fwd(in);
    ProjectionFusedOutputs b = sxtest::shared_metal_backend().projection_fused_fwd(in);
    const size_t CN = static_cast<size_t>(in.C * in.N);

    SX_CHECK(a.radii.size() == CN * 2 && a.radii == b.radii); // every radii slot is written → exact equality
    const std::vector<unsigned char> valid = valid_mask(a.radii.data(), CN);
    size_t nvalid = 0;
    for (unsigned char m : valid) nvalid += m;
    SX_CHECK(nvalid > 0);

    SX_CHECK(sxtest::count_bitwise_mismatch(a.means2d.data(), b.means2d.data(), CN * 2,
                                            valid.data(), 2) == 0);
    SX_CHECK(sxtest::count_bitwise_mismatch(a.depths.data(), b.depths.data(), CN, valid.data(),
                                            1) == 0);
    SX_CHECK(sxtest::count_bitwise_mismatch(a.conics.data(), b.conics.data(), CN * 3,
                                            valid.data(), 3) == 0);
    if (in.calc_compensations) // an empty vector for a calc_comp=0 fixture — guards against .data() UB
        SX_CHECK(sxtest::count_bitwise_mismatch(a.compensations.data(), b.compensations.data(),
                                                CN, valid.data(), 1) == 0);
}

// (5) Metal vs CPU cross-check — under the mask, using the meta.json tolerances (brief Step 1-(5)).
//     ③ The opacities variant verifies the opacity-aware bbox path (kProjHasOpacities=true), for
//     which there is no golden, by comparing CPU and Metal against each other (this also measures
//     the precise::log ↔ std::log parity).
SX_TEST(projection_fused_fwd_metal_matches_cpu) {
    struct Case {
        const char* name;
        const char* dir;
        bool use_covars;
        bool with_opacities;
    };
    const Case cases[] = {
        {"qs/c1", "proj_fused_qs/c1", false, false},
        {"covars/ff", "proj_fully_fused", true, false},
        {"qs+opacity/c1", "proj_fused_qs/c1", false, true},
    };
    for (const Case& tc : cases) {
        auto meta = json_parse_flat(fx(tc.dir, "meta.json"));
        SXTensor means = SXTensor::load(fx(tc.dir, "means.sxt"));
        SXTensor viewmats = SXTensor::load(fx(tc.dir, "viewmats.sxt"));
        SXTensor Ks = SXTensor::load(fx(tc.dir, "Ks.sxt"));
        SXTensor covars6, quats, scales, opacities;

        ProjectionFusedInputs in = make_inputs(meta);
        in.means = means.f32();
        in.viewmats = viewmats.f32();
        in.Ks = Ks.f32();
        if (tc.use_covars) {
            covars6 = SXTensor::load(fx(tc.dir, "covars_triu.sxt"));
            in.covars = covars6.f32();
        } else {
            quats = SXTensor::load(fx(tc.dir, "quats.sxt"));
            scales = SXTensor::load(fx(tc.dir, "scales.sxt"));
            in.quats = quats.f32();
            in.scales = scales.f32();
        }
        if (tc.with_opacities) {
            opacities = SXTensor::load(fx(tc.dir, "opacities.sxt"));
            in.opacities = opacities.f32();
        }

        ProjectionFusedOutputs cpu = projection_fused_fwd_cpu(in);
        ProjectionFusedOutputs mtl = sxtest::shared_metal_backend().projection_fused_fwd(in);
        const size_t CN = static_cast<size_t>(in.C * in.N);

        // The culling decision (radii) must match exactly as integers between CPU and Metal for the
        // mask to be consistent.
        SX_EXACT_I32((std::string(tc.name) + " radii").c_str(), mtl.radii,
                     cpu.radii.data(), CN * 2);
        const std::vector<unsigned char> valid = valid_mask(cpu.radii.data(), CN);
        size_t nvalid = 0;
        for (unsigned char m : valid) nvalid += m;
        SX_CHECK(nvalid > 0);

        SX_ALL_CLOSE_MASKED(mtl.means2d.data(), cpu.means2d.data(), CN * 2, meta, "means2d",
                            valid.data(), 2);
        SX_ALL_CLOSE_MASKED(mtl.depths.data(), cpu.depths.data(), CN, meta, "depths",
                            valid.data(), 1);
        SX_ALL_CLOSE_MASKED(mtl.conics.data(), cpu.conics.data(), CN * 3, meta, "conics",
                            valid.data(), 3);
        if (in.calc_compensations) // an empty vector for a calc_comp=0 fixture — guards against .data() UB
            SX_ALL_CLOSE_MASKED(mtl.compensations.data(), cpu.compensations.data(), CN, meta,
                                "compensations", valid.data(), 1);

        // Measured record (not a gate): the bitwise mismatch count over the valid slots — used to
        // track the per-device fast-math ulp deviation.
        const size_t d_m2 = sxtest::count_bitwise_mismatch(mtl.means2d.data(),
                                                           cpu.means2d.data(), CN * 2,
                                                           valid.data(), 2);
        const size_t d_cn = sxtest::count_bitwise_mismatch(mtl.conics.data(), cpu.conics.data(),
                                                           CN * 3, valid.data(), 3);
        const size_t d_cp = in.calc_compensations
                                ? sxtest::count_bitwise_mismatch(mtl.compensations.data(),
                                                                 cpu.compensations.data(), CN,
                                                                 valid.data(), 1)
                                : 0;
        std::printf("  INFO %s: valid %zu/%zu, metal vs cpu bitwise mismatch means2d %zu/%zu "
                    "conics %zu/%zu comps %zu/%zu\n",
                    tc.name, nvalid, CN, d_m2, nvalid * 2, d_cn, nvalid * 3, d_cp, nvalid);
    }
}

// ===== backward (Task 6) — spec 01 B0–B8 plus Corrections 1–6 =====
// The bwd inputs are the fixture's "golden fwd outputs" (radii, conics, and compensations),
// injected directly — chaining the Metal fwd output is forbidden (D7 blocks error propagation,
// task-5 §6). The upstream v_* are injected exactly as stored, after multiplication by the valid
// mask (task-3). The expected gradients v_means, v_quats, and v_scales are already summed over the
// camera axis, and the golden grad of a Gaussian culled in every camera is exactly 0 (the VJP is
// linear in upstream, and the upstream mask is 0) — the kernel likewise contributes nothing, thanks
// to the radii gate (the zero-initialized buffer is preserved) → everything is compared with no
// mask.
// kProjViewmatsGrad is enabled from pose Task 1 — v_viewmats [C,16] is compared against the golden
// too (B8 reduction strategy 2: a cid-uniform simdgroup uses simd_sum plus a lane0 atomic; summed
// over the whole Gaussian axis). Because it accumulates through a per-camera atomicAdd it is
// non-deterministic → the c1/c2 golden comparison uses tolerances, and c2 additionally applies the
// self-consistency rel_l2 gate.

namespace {

struct BwdFixture {
    std::map<std::string, double> meta;
    SXTensor means, quats, scales, viewmats, Ks;     // fwd inputs (for rematerialization)
    SXTensor radii, conics, comps;                   // golden fwd outputs → bwd inputs
    SXTensor v_means2d, v_depths, v_conics, v_comps; // upstream gradients
    SXTensor gv_means, gv_quats, gv_scales;          // expected gradients (the autograd golden)
    SXTensor gv_viewmats;                            // the [C,16] v_viewmats golden (kProjViewmatsGrad)
    bool inject_v_comp = false;                      // meta.bwd_inject_v_compensations
};

BwdFixture load_bwd_fixture(const std::string& dir) {
    BwdFixture f;
    f.meta = json_parse_flat(fx(dir, "meta.json"));
    f.means = SXTensor::load(fx(dir, "means.sxt"));
    f.quats = SXTensor::load(fx(dir, "quats.sxt"));
    f.scales = SXTensor::load(fx(dir, "scales.sxt"));
    f.viewmats = SXTensor::load(fx(dir, "viewmats.sxt"));
    f.Ks = SXTensor::load(fx(dir, "Ks.sxt"));
    f.radii = SXTensor::load(fx(dir, "radii.sxt"));
    f.conics = SXTensor::load(fx(dir, "conics.sxt"));
    f.comps = SXTensor::load(fx(dir, "compensations.sxt"));
    f.v_means2d = SXTensor::load(fx(dir, "v_means2d.sxt"));
    f.v_depths = SXTensor::load(fx(dir, "v_depths.sxt"));
    f.v_conics = SXTensor::load(fx(dir, "v_conics.sxt"));
    f.gv_means = SXTensor::load(fx(dir, "v_means.sxt"));
    f.gv_quats = SXTensor::load(fx(dir, "v_quats.sxt"));
    f.gv_scales = SXTensor::load(fx(dir, "v_scales.sxt"));
    f.gv_viewmats = SXTensor::load(fx(dir, "v_viewmats.sxt"));
    f.inject_v_comp = f.meta.at("bwd_inject_v_compensations") != 0.0;
    if (f.inject_v_comp) f.v_comps = SXTensor::load(fx(dir, "v_compensations.sxt"));
    return f; // the SXTensor payload lives on the vector heap — the f32() pointers stay valid after the move
}

ProjectionFusedInputs bwd_inputs(const BwdFixture& f) {
    ProjectionFusedInputs in = make_inputs(f.meta);
    in.means = f.means.f32();
    in.quats = f.quats.f32(); // covars == nullptr → the quat/scale path
    in.scales = f.scales.f32();
    in.viewmats = f.viewmats.f32();
    in.Ks = f.Ks.f32();
    return in;
}

ProjectionFusedGradsIn bwd_grads_in(const BwdFixture& f) {
    ProjectionFusedGradsIn g;
    g.radii = f.radii.i32();
    g.conics = f.conics.f32();
    g.v_means2d = f.v_means2d.f32();
    g.v_depths = f.v_depths.f32();
    g.v_conics = f.v_conics.f32();
    g.viewmats_grad = true; // kProjViewmatsGrad enabled — verifies the v_viewmats golden (pose Task 1)
    if (f.inject_v_comp) {
        // The CUDA semantics: v_compensations != nullptr is the gate for add_blur_vjp (B2)
        // (kProjCalcComp).
        g.compensations = f.comps.f32();
        g.v_compensations = f.v_comps.f32();
    }
    return g;
}

// Compares all four expected gradients and confirms the shape and path contracts.
void compare_bwd_golden(const BwdFixture& f, const ProjectionFusedGrads& g) {
    const size_t N = static_cast<size_t>(f.meta.at("N"));
    const size_t C = static_cast<size_t>(f.meta.at("C"));
    SX_CHECK(g.v_means.size() == N * 3 && g.v_quats.size() == N * 4 &&
             g.v_scales.size() == N * 3);
    // The quat/scale path (no v_covars) with viewmats_grad=true (v_viewmats [C,16] is present).
    SX_CHECK(g.v_covars.empty() && g.v_viewmats.size() == C * 16);
    SX_ALL_CLOSE(g.v_means.data(), f.gv_means.f32(), N * 3, f.meta, "v_means");
    SX_ALL_CLOSE(g.v_quats.data(), f.gv_quats.f32(), N * 4, f.meta, "v_quats");
    SX_ALL_CLOSE(g.v_scales.data(), f.gv_scales.f32(), N * 3, f.meta, "v_scales");
    // v_viewmats: simd_sum (over every Gaussian) plus a per-camera atomicAdd (B8 reduction
    // strategy 2). The bottom row, flat 12–15, is 0 in both the golden and the kernel (only
    // viewmats [:3,:3] and [:3,3] lie on the differentiation path) — it is included in the
    // allclose.
    SX_ALL_CLOSE(g.v_viewmats.data(), f.gv_viewmats.f32(), C * 16, f.meta, "v_viewmats");
    // Measured record (not a gate): the Metal↔golden rel_l2 — the evidence behind the meta
    // tolerances (task-1 §Tolerances).
    {
        const float* mv = g.v_viewmats.data();
        const float* gv = f.gv_viewmats.f32();
        double maxabs = 0, maxrel = 0, maxmag = 0;
        for (size_t i = 0; i < C * 16; i++) {
            const double d = std::fabs((double)mv[i] - (double)gv[i]);
            if (d > maxabs) maxabs = d;
            const double m = std::fabs((double)gv[i]);
            if (m > maxmag) maxmag = m;
            if (m > 1.0 && d / m > maxrel) maxrel = d / m;
        }
        std::printf("  INFO v_viewmats: metal vs golden rel_l2 %.3g maxabs %.3g maxrel(|g|>1) "
                    "%.3g maxmag %.3g\n",
                    sxtest::rel_l2(mv, gv, C * 16, gv), maxabs, maxrel, maxmag);
    }
}

bool bits_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

} // namespace

// (6) Metal bwd C=1 vs the proj_fused_qs/c1 autograd golden — the non-atomic kProjSingleCamera
//     path. c1 has bwd_inject_v_compensations=0 → kProjCalcComp=false (B2 is skipped; equivalent to
//     v_comp≡0).
SX_TEST(projection_fused_bwd_metal_c1_matches_golden) {
    BwdFixture f = load_bwd_fixture("proj_fused_qs/c1");
    ProjectionFusedInputs in = bwd_inputs(f);
    ProjectionFusedGradsIn gin = bwd_grads_in(f);
    ProjectionFusedGrads g = sxtest::shared_metal_backend().projection_fused_bwd(in, gin);
    compare_bwd_golden(f, g);
}

// (7) C=1 bit-exact determinism plus verification of the zero-initialization helper (D8-1) — three
//     consecutive calls on identical input produce bit-identical output everywhere. If
//     encode_zero_fill were missing, leftover values in the reused buffers would accumulate and the
//     results would diverge from call to call.
SX_TEST(projection_fused_bwd_metal_c1_bit_deterministic) {
    BwdFixture f = load_bwd_fixture("proj_fused_qs/c1");
    ProjectionFusedInputs in = bwd_inputs(f);
    ProjectionFusedGradsIn gin = bwd_grads_in(f);
    ProjectionFusedGrads a = sxtest::shared_metal_backend().projection_fused_bwd(in, gin);
    ProjectionFusedGrads b = sxtest::shared_metal_backend().projection_fused_bwd(in, gin);
    ProjectionFusedGrads c = sxtest::shared_metal_backend().projection_fused_bwd(in, gin);
    SX_CHECK(bits_equal(a.v_means, b.v_means) && bits_equal(b.v_means, c.v_means));
    SX_CHECK(bits_equal(a.v_quats, b.v_quats) && bits_equal(b.v_quats, c.v_quats));
    SX_CHECK(bits_equal(a.v_scales, b.v_scales) && bits_equal(b.v_scales, c.v_scales));
    // Confirms live gradients exist (this prevents an all-zeros regression — comparison against the
    // golden is test (6)'s job).
    bool any_nonzero = false;
    for (float v : a.v_means)
        if (v != 0.f) any_nonzero = true;
    SX_CHECK(any_nonzero);
}

// (8) The C=2 atomic path vs the proj_fused_qs/c2 golden — with v_compensations injected
//     (kProjCalcComp=true, B2 add_blur_vjp enabled). CUDA's promotion to fp64 (Utils.cuh:486,
//     Correction 2) is computed in MSL float — this confirms the difference is absorbed by the
//     meta.json tolerances (the entries specific to v_*). The atomic accumulation order is
//     non-deterministic → the golden comparison uses tolerances, and two-run self-consistency is
//     gated at ||r1-r2||₂/||golden||₂ ≤ 0.1·rel_l2.
SX_TEST(projection_fused_bwd_metal_c2_matches_golden) {
    BwdFixture f = load_bwd_fixture("proj_fused_qs/c2");
    ProjectionFusedInputs in = bwd_inputs(f);
    ProjectionFusedGradsIn gin = bwd_grads_in(f);
    ProjectionFusedGrads r1 = sxtest::shared_metal_backend().projection_fused_bwd(in, gin);
    ProjectionFusedGrads r2 = sxtest::shared_metal_backend().projection_fused_bwd(in, gin);
    compare_bwd_golden(f, r1);

    struct Pair {
        const char* tensor;
        const std::vector<float>* a;
        const std::vector<float>* b;
        const float* golden;
    };
    const Pair pairs[] = {{"v_means", &r1.v_means, &r2.v_means, f.gv_means.f32()},
                          {"v_quats", &r1.v_quats, &r2.v_quats, f.gv_quats.f32()},
                          {"v_scales", &r1.v_scales, &r2.v_scales, f.gv_scales.f32()},
                          // v_viewmats: the per-camera atomicAdd accumulation order is
                          // non-deterministic — self-consistency
                          // (||run1-run2||₂/||golden||₂ ≤ 0.1·rel_l2) gates the stability of the
                          // atomic path.
                          {"v_viewmats", &r1.v_viewmats, &r2.v_viewmats, f.gv_viewmats.f32()}};
    for (const Pair& p : pairs) {
        const double gate = 0.1 * f.meta.at(std::string("tolerances.") + p.tensor + ".rel_l2");
        // Two-run self-consistency: ||run1 - run2||₂ / ||golden||₂ (the 07 / task-3 gate).
        const double rel = sxtest::rel_l2(p.a->data(), p.b->data(), p.a->size(), p.golden);
        if (!(rel <= gate)) {
            std::printf("  FAIL %s self-consistency rel_l2 %g > %g\n", p.tensor, rel, gate);
            ::sxtest::Registry::shared().failures++;
        }
        std::printf("  INFO %s: run1 vs run2 rel_l2 %.3g (gate %.3g)\n", p.tensor, rel, gate);
    }
}
