// The negative case table for Validate.h (a 1b final review carry-over — 1c Task 1 item 3).
// Each case = (a single mutation of a valid baseline input, a substring of the expected throw
// message). The validators are pure functions (they never dereference pointers), so no backend and
// no real buffers are needed — dummy addresses merely satisfy the non-null checks.
// Messages are matched as substrings (the full wording is not pinned down — only the combination
// with the what prefix ("vt: ") is protected against regression).
// Every throw type is std::invalid_argument, with the single exception of the projection
// camera_model, which throws std::runtime_error (the existing contract — the Validate.h comment) —
// the type distinction is asserted as well.
#include "TestHarness.h"
#include "splatx/KernelBackend.h"
#include "splatx/Validate.h"

#include <cstdint>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>

using namespace splatx;

namespace {

// Dummy non-null pointers (the validators never dereference them — only the address matters)
float f_dummy = 0.f;
int32_t i_dummy = 0;
uint64_t u_dummy = 0;
const float* F = &f_dummy;
const int32_t* I32 = &i_dummy;
const uint64_t* U64 = &u_dummy;

// ===== Valid baseline input factories — each case mutates exactly one field of these =====

AdamInputs adam_ok() {
    AdamInputs in;
    in.param = &f_dummy;
    in.grad = F;
    in.exp_avg = &f_dummy;
    in.exp_avg_sq = &f_dummy;
    in.n_elems = 6;
    in.n_gauss = 2;
    return in;
}

ProjectionFusedInputs proj_ok() {
    ProjectionFusedInputs in;
    in.means = F;
    in.quats = F;
    in.scales = F;
    in.viewmats = F;
    in.Ks = F;
    in.N = 4;
    in.C = 1;
    return in;
}

ProjectionFusedGradsIn grads_ok() {
    ProjectionFusedGradsIn g;
    g.radii = I32;
    g.conics = F;
    g.v_means2d = F;
    g.v_depths = F;
    g.v_conics = F;
    return g;
}

ShInputs sh_ok() {
    ShInputs in;
    in.dirs = F;
    in.coeffs = F;
    in.degrees_to_use = 3;
    in.B = 1;
    in.N = 4;
    in.K = 16;
    in.D = 3;
    return in;
}

IsectTilesInputs isect_ok() {
    IsectTilesInputs in;
    in.means2d = F;
    in.radii = I32;
    in.depths = F;
    in.I = 1;
    in.N = 4;
    in.tile_size = 16;
    in.tile_width = 10;
    in.tile_height = 8;
    return in;
}

SortIsectInputs sort_ok() {
    SortIsectInputs in;
    in.keys = U64;
    in.values = I32;
    in.n = 1;
    in.key_n_bits = 36;
    return in;
}

IsectOffsetInputs offset_ok() {
    IsectOffsetInputs in;
    in.isect_ids = U64;
    in.n_isects = 1;
    in.I = 1;
    in.tile_width = 10;
    in.tile_height = 8;
    return in;
}

RasterizeFwdInputs rast_ok() {
    RasterizeFwdInputs in;
    in.means2d = F;
    in.conics = F;
    in.colors = F;
    in.opacities = F;
    in.isect_offsets = I32;
    in.flatten_ids = I32;
    in.I = 1;
    in.N = 4;
    in.n_isects = 1;
    in.image_width = 160;
    in.image_height = 120;
    in.tile_size = 16;
    in.tile_width = 10;
    in.tile_height = 8;
    return in;
}

RasterizeBwdGradsIn rast_bwd_ok() {
    RasterizeBwdGradsIn g;
    g.render_alphas = F;
    g.last_ids = I32;
    g.v_render_colors = F;
    g.v_render_alphas = F;
    return g;
}

RngNormalInputs rng_ok() {
    RngNormalInputs in;
    in.seed = 1;
    in.offset = 2;
    in.count = 8;
    return in;
}

RelocationInputs reloc_ok() {
    RelocationInputs in;
    in.opacities = F;
    in.scales = F;
    in.ratios = I32;
    in.binoms = F;
    in.N = 4;
    return in;
}

PerturbInputs perturb_ok() {
    PerturbInputs in;
    in.positions = &f_dummy;
    in.quats = F;
    in.scales_log = F;
    in.opacities_logit = F;
    in.noise = F;
    in.N = 4;
    return in;
}

} // namespace

// The exhaustive table of throw paths in Validate.h — the valid baseline inputs pass without
// throwing (checked first), and each negative mutation throws with the expected message included.
// The runtime_error case has expect_invalid=false.
SX_TEST(validate_negative_table) {
    // First: every baseline input passes (this pins down that the mutation alone causes the throw)
    validate_adam_inputs(adam_ok(), "vt");
    validate_projection_fused_inputs(proj_ok(), "vt");
    validate_projection_fused_grads_in(grads_ok(), "vt");
    validate_sh_inputs(sh_ok(), "vt");
    validate_isect_tiles_inputs(isect_ok(), "vt");
    validate_sort_isect_inputs(sort_ok(), "vt");
    validate_isect_offset_inputs(offset_ok(), "vt");
    validate_rasterize_fwd_inputs(rast_ok(), "vt");
    validate_rasterize_bwd_inputs(rast_ok(), rast_bwd_ok(), "vt");
    validate_rng_inputs(rng_ok(), "vt");
    validate_relocation_inputs(reloc_ok(), "vt");
    validate_perturb_inputs(perturb_ok(), "vt");

    struct Case {
        const char* name;
        std::function<void()> fn;
        const char* expect;         // a substring of e.what()
        bool expect_invalid = true; // true: invalid_argument, false: runtime_error
    };

    const Case cases[] = {
        // --- General-purpose primitives ---
        {"nonnull", [] { validate_nonnull(nullptr, "vt", "ptr"); }, "null ptr"},
        {"range_u32", [] { validate_range_u32(65, 1, 64, "vt", "key_n_bits"); },
         "key_n_bits must be 1..64"},

        // --- validate_adam_inputs ---
        {"adam null tensor", [] { auto in = adam_ok(); in.grad = nullptr;
                                  validate_adam_inputs(in, "vt"); }, "null tensor"},
        {"adam n_elems % n_gauss", [] { auto in = adam_ok(); in.n_elems = 5;
                                        validate_adam_inputs(in, "vt"); },
         "n_elems must be a positive multiple of n_gauss"},
        {"adam n_gauss <= 0", [] { auto in = adam_ok(); in.n_gauss = 0;
                                   validate_adam_inputs(in, "vt"); },
         "positive multiple of n_gauss"},

        // --- validate_projection_fused_inputs ---
        {"proj camera_model", [] { auto in = proj_ok(); in.camera_model = 2;
                                   validate_projection_fused_inputs(in, "vt"); },
         "only PINHOLE (camera_model=0) supported", /*expect_invalid=*/false},
        {"proj null means", [] { auto in = proj_ok(); in.means = nullptr;
                                 validate_projection_fused_inputs(in, "vt"); },
         "null means/viewmats/Ks"},
        {"proj no covar path", [] { auto in = proj_ok(); in.quats = nullptr;
                                    validate_projection_fused_inputs(in, "vt"); },
         "need covars (6-triu) or quats+scales"},
        {"proj negative N", [] { auto in = proj_ok(); in.N = -1;
                                 validate_projection_fused_inputs(in, "vt"); },
         "negative N or C"},

        // --- validate_projection_fused_grads_in ---
        {"grads null radii", [] { auto g = grads_ok(); g.radii = nullptr;
                                  validate_projection_fused_grads_in(g, "vt"); },
         "null radii/conics/v_means2d/v_depths/v_conics"},
        {"grads v_comp without comp", [] { auto g = grads_ok(); g.v_compensations = F;
                                           validate_projection_fused_grads_in(g, "vt"); },
         "v_compensations requires compensations"},

        // --- validate_sh_inputs ---
        {"sh null dirs", [] { auto in = sh_ok(); in.dirs = nullptr;
                              validate_sh_inputs(in, "vt"); }, "null dirs/coeffs"},
        {"sh negative B", [] { auto in = sh_ok(); in.B = -1;
                               validate_sh_inputs(in, "vt"); }, "negative B or N"},
        {"sh K < 1", [] { auto in = sh_ok(); in.K = 0;
                          validate_sh_inputs(in, "vt"); }, "K and D must be positive"},
        {"sh degrees > 4", [] { auto in = sh_ok(); in.degrees_to_use = 5;
                                validate_sh_inputs(in, "vt"); },
         "degrees_to_use must be 0..4"},
        {"sh bands^2 > K", [] { auto in = sh_ok(); in.degrees_to_use = 4; // 25 > K=16
                                validate_sh_inputs(in, "vt"); },
         "(degrees_to_use+1)^2 must be <= K"},

        // --- validate_isect_tiles_inputs ---
        {"isect null means2d", [] { auto in = isect_ok(); in.means2d = nullptr;
                                    validate_isect_tiles_inputs(in, "vt"); },
         "null means2d/radii/depths"},
        {"isect conics without opacities", [] { auto in = isect_ok(); in.conics = F;
                                                validate_isect_tiles_inputs(in, "vt"); },
         "conics and opacities must be given together"},
        {"isect negative I", [] { auto in = isect_ok(); in.I = -1;
                                  validate_isect_tiles_inputs(in, "vt"); }, "negative I or N"},
        {"isect tile_size 0", [] { auto in = isect_ok(); in.tile_size = 0;
                                   validate_isect_tiles_inputs(in, "vt"); },
         "tile_size/tile_width/tile_height must be positive"},
        {"isect bit budget", [] { auto in = isect_ok(); // 20 + bits(4096*4096)=24 → 44 > 32
                                  in.I = 1 << 20; in.tile_width = 4096; in.tile_height = 4096;
                                  validate_isect_tiles_inputs(in, "vt"); },
         "image_n_bits + tile_n_bits must be <= 32"},
        {"isect I*N overflow", [] { auto in = isect_ok(); // 2^30 * 4 > INT32_MAX, bits 30+0
                                    in.I = int64_t{1} << 30; in.N = 4;
                                    in.tile_width = 1; in.tile_height = 1;
                                    validate_isect_tiles_inputs(in, "vt"); },
         "I*N must fit in int32"},

        // --- validate_sort_isect_inputs ---
        {"sort negative n", [] { auto in = sort_ok(); in.n = -1;
                                 validate_sort_isect_inputs(in, "vt"); }, "negative n"},
        {"sort null keys", [] { auto in = sort_ok(); in.keys = nullptr;
                                validate_sort_isect_inputs(in, "vt"); }, "null keys/values"},
        {"sort n > int32", [] { auto in = sort_ok(); in.n = int64_t{INT32_MAX} + 1;
                                validate_sort_isect_inputs(in, "vt"); },
         "n must fit in int32"},
        {"sort key_n_bits 0", [] { auto in = sort_ok(); in.key_n_bits = 0;
                                   validate_sort_isect_inputs(in, "vt"); },
         "key_n_bits must be 1..64"},

        // --- validate_isect_offset_inputs ---
        {"offset negative n_isects", [] { auto in = offset_ok(); in.n_isects = -1;
                                          validate_isect_offset_inputs(in, "vt"); },
         "negative n_isects"},
        {"offset null isect_ids", [] { auto in = offset_ok(); in.isect_ids = nullptr;
                                       validate_isect_offset_inputs(in, "vt"); },
         "null isect_ids"},
        {"offset negative I", [] { auto in = offset_ok(); in.I = -1;
                                   validate_isect_offset_inputs(in, "vt"); }, "negative I"},
        {"offset I == 0", [] { auto in = offset_ok(); in.I = 0;
                               validate_isect_offset_inputs(in, "vt"); },
         "n_isects > 0 requires I > 0"},
        {"offset tile_width 0", [] { auto in = offset_ok(); in.tile_width = 0;
                                     validate_isect_offset_inputs(in, "vt"); },
         "tile_width/tile_height must be positive"},
        {"offset bit budget", [] { auto in = offset_ok(); in.I = 1 << 20;
                                   in.tile_width = 4096; in.tile_height = 4096;
                                   validate_isect_offset_inputs(in, "vt"); },
         "image_n_bits + tile_n_bits must be <= 32"},
        {"offset n_isects > int32", [] { auto in = offset_ok();
                                         in.n_isects = int64_t{INT32_MAX} + 1;
                                         in.tile_width = 1; in.tile_height = 1;
                                         validate_isect_offset_inputs(in, "vt"); },
         "n_isects must fit in int32"},
        {"offset I*n_tiles overflow", [] { auto in = offset_ok(); // 2^20 * 4096 = 2^32, bits 20+12
                                           in.I = 1 << 20; in.tile_width = 64;
                                           in.tile_height = 64;
                                           validate_isect_offset_inputs(in, "vt"); },
         "I*n_tiles must fit in int32"},

        // --- validate_rasterize_fwd_inputs ---
        {"rast null conics", [] { auto in = rast_ok(); in.conics = nullptr;
                                  validate_rasterize_fwd_inputs(in, "vt"); },
         "null means2d/conics/colors/opacities/isect_offsets"},
        {"rast null flatten_ids", [] { auto in = rast_ok(); in.flatten_ids = nullptr;
                                       validate_rasterize_fwd_inputs(in, "vt"); },
         "null flatten_ids"},
        {"rast negative n_isects", [] { auto in = rast_ok(); in.n_isects = -1;
                                        validate_rasterize_fwd_inputs(in, "vt"); },
         "negative I/N/n_isects"},
        {"rast n_isects without N", [] { auto in = rast_ok(); in.N = 0; // n_isects=1 is kept
                                         validate_rasterize_fwd_inputs(in, "vt"); },
         "n_isects > 0 requires N > 0"},
        {"rast tile_size 4", [] { auto in = rast_ok(); in.tile_size = 4; // supported upstream, but not ported
                                  validate_rasterize_fwd_inputs(in, "vt"); },
         "tile_size must be 16"},
        {"rast image 0", [] { auto in = rast_ok(); in.image_width = 0;
                              validate_rasterize_fwd_inputs(in, "vt"); },
         "image_width/image_height must be positive"},
        {"rast grid uncovered", [] { auto in = rast_ok(); in.tile_width = 9; // 144 < 160
                                     validate_rasterize_fwd_inputs(in, "vt"); },
         "tile grid must cover the image"},
        {"rast n_isects > int32", [] { auto in = rast_ok();
                                       in.n_isects = int64_t{INT32_MAX} + 1;
                                       validate_rasterize_fwd_inputs(in, "vt"); },
         "n_isects must fit in int32"},

        // --- validate_rasterize_bwd_inputs (reuses the fwd rules, plus 4 bwd-specific ones) ---
        {"rast bwd fwd rule pass-through", [] { auto in = rast_ok(); in.tile_size = 4;
                                        validate_rasterize_bwd_inputs(in, rast_bwd_ok(), "vt"); },
         "tile_size must be 16"},
        {"rast bwd null last_ids", [] { auto g = rast_bwd_ok(); g.last_ids = nullptr;
                                        validate_rasterize_bwd_inputs(rast_ok(), g, "vt"); },
         "null render_alphas/last_ids (fwd outputs)"},
        {"rast bwd null v_render", [] { auto g = rast_bwd_ok(); g.v_render_alphas = nullptr;
                                        validate_rasterize_bwd_inputs(rast_ok(), g, "vt"); },
         "null v_render_colors/v_render_alphas"},

        // --- validate_rng_inputs (spec 06 — Task 5) ---
        {"rng negative count", [] { auto in = rng_ok(); in.count = -1;
                                    validate_rng_inputs(in, "vt"); }, "negative count"},
        {"rng count > uint32", [] { auto in = rng_ok();
                                    in.count = int64_t{UINT32_MAX} + 1;
                                    validate_rng_inputs(in, "vt"); },
         "count must fit in uint32"},

        // --- validate_relocation_inputs ---
        {"reloc null binoms", [] { auto in = reloc_ok(); in.binoms = nullptr;
                                   validate_relocation_inputs(in, "vt"); },
         "null opacities/scales/ratios/binoms"},
        {"reloc negative N", [] { auto in = reloc_ok(); in.N = -1;
                                  validate_relocation_inputs(in, "vt"); }, "negative N"},
        {"reloc n_max < 1", [] { auto in = reloc_ok(); in.n_max = 0;
                                 validate_relocation_inputs(in, "vt"); },
         "n_max must be >= 1"},

        // --- validate_perturb_inputs ---
        {"perturb null noise", [] { auto in = perturb_ok(); in.noise = nullptr;
                                    validate_perturb_inputs(in, "vt"); },
         "null positions/quats/scales_log/opacities_logit/noise"},
        {"perturb negative N", [] { auto in = perturb_ok(); in.N = -1;
                                    validate_perturb_inputs(in, "vt"); }, "negative N"},
        {"perturb N > uint32", [] { auto in = perturb_ok();
                                    in.N = int64_t{UINT32_MAX} + 1;
                                    validate_perturb_inputs(in, "vt"); },
         "N exceeds uint32"},
    };

    auto& reg = sxtest::Registry::shared();
    for (const Case& c : cases) {
        bool threw = false, invalid = false;
        std::string msg;
        try {
            c.fn();
        } catch (const std::invalid_argument& e) {
            threw = invalid = true;
            msg = e.what();
        } catch (const std::exception& e) { // runtime_error (proj camera_model)
            threw = true;
            msg = e.what();
        }
        if (!threw) {
            std::printf("  FAIL %s:%d: [%s] did not throw\n", __FILE__, __LINE__, c.name);
            reg.failures++;
            continue;
        }
        if (msg.find(c.expect) == std::string::npos) {
            std::printf("  FAIL %s:%d: [%s] message '%s' missing '%s'\n", __FILE__, __LINE__,
                        c.name, msg.c_str(), c.expect);
            reg.failures++;
        }
        if (invalid != c.expect_invalid) {
            std::printf("  FAIL %s:%d: [%s] exception type %s (expected %s)\n", __FILE__,
                        __LINE__, c.name, invalid ? "invalid_argument" : "runtime_error",
                        c.expect_invalid ? "invalid_argument" : "runtime_error");
            reg.failures++;
        }
        if (msg.find("vt: ") != 0) { // the what prefix contract (shared by throw_invalid and runtime_error)
            std::printf("  FAIL %s:%d: [%s] message '%s' missing 'vt: ' prefix\n", __FILE__,
                        __LINE__, c.name, msg.c_str());
            reg.failures++;
        }
    }
}

// The n_isects readback guard of the D11 device-resident executor (Phase 3.5 Task 1 — spelling out
// the uint32 cast contract).
// The passing boundaries (0 and INT32_MAX) return the value unchanged; negative values and values
// beyond int32 (including the range that uint32 would truncate) give invalid_argument plus the
// message substring plus the "vt: " prefix — the same convention as the table above.
SX_TEST(validate_n_isects_readback_guard) {
    SX_CHECK(validate_n_isects_readback(0, "vt") == 0u);
    SX_CHECK(validate_n_isects_readback(12345, "vt") == 12345u);
    SX_CHECK(validate_n_isects_readback(INT32_MAX, "vt") ==
             static_cast<std::uint32_t>(INT32_MAX));

    struct Case {
        const char* name;
        std::int64_t v;
        const char* expect;
    };
    const Case cases[] = {
        {"negative", -1, "negative n_isects"},
        {"over int32", std::int64_t{INT32_MAX} + 1, "n_isects must fit in int32"},
        // Representable in uint32 but beyond int32 — even though the PR/PO cast is lossless, this is
        // the range in which the int32 writes to offsets and last_ids would be truncated (rejected
        // at the same boundary as the host-path validator)
        {"uint32-representable", std::int64_t{UINT32_MAX}, "n_isects must fit in int32"},
        {"over uint32", std::int64_t{UINT32_MAX} + 1, "n_isects must fit in int32"},
    };
    auto& reg = sxtest::Registry::shared();
    for (const Case& c : cases) {
        bool threw = false;
        std::string msg;
        try {
            (void)validate_n_isects_readback(c.v, "vt");
        } catch (const std::invalid_argument& e) {
            threw = true;
            msg = e.what();
        }
        if (!threw) {
            std::printf("  FAIL %s:%d: [%s] did not throw invalid_argument\n", __FILE__,
                        __LINE__, c.name);
            reg.failures++;
            continue;
        }
        if (msg.find(c.expect) == std::string::npos || msg.find("vt: ") != 0) {
            std::printf("  FAIL %s:%d: [%s] message '%s' missing '%s' or 'vt: ' prefix\n",
                        __FILE__, __LINE__, c.name, msg.c_str(), c.expect);
            reg.failures++;
        }
    }
}
