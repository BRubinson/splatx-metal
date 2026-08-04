// Offline compilation of .metallib + layered loading (Phase 2 Task 2 — a v0.1 redistribution
// requirement).
//
// (1) Runs tools/build_metallib.sh **once, from inside the test** (the output goes to a temporary
//     directory — build artifacts are never committed as test fixtures. The script has no
//     incremental cache and fully recompiles on every run, so even on CI it always reproduces
//     against the current sources; xcrun metal takes a few seconds).
//     A backend that loads the produced metallib **through the explicit-path layer (a) alone**
//     - passes the adam fixture golden (drives function constant specialization —
//       kAdamHasValidMask)
//     - passes the raster_std fwd golden (render allclose + last_ids exact)
//     - produces the same results as the source-compiled backend: the meta.json tolerance gate +
//       an INFO record of the bitwise agreement count (if the offline and runtime compilers emit
//       the same AIR the results are bit-identical — but the gate is the tolerance).
// (2) An explicit path (layer a) that fails = an immediate throw (it MUST NOT engage even when the
//     fallback layers are alive — review Important 1) / auto-discovery (layer b) absent → the
//     source fallback, present → the metallib is chosen / all layers failing → a throw with the
//     accumulated diagnostics.
//
// Running the whole suite against a metallib is a separate switch: SPLATX_TEST_METALLIB=<path>
// (TestUtils.h — shared_metal_backend is then created with the explicit path, so every existing
// test runs through the metallib path).
//
// ===== Host-only TU (Phase 6 Task 3 — excluded on iOS) =====
// This compiles on macOS only, because it needs std::system (to run the build script — unavailable
// in the iOS SDK) and has to construct the exe-adjacent scenario. On an iOS build the #else branch
// below registers the test names in host_only_excluded(), so the runner report states them
// explicitly as "EXCLUDED N host-only test(s)"
// (the registration and the excluded name sit adjacent in the same file — when renaming, fix
// both).
#include <TargetConditionals.h>
#if !TARGET_OS_OSX

#include "TestHarness.h"

static ::sxtest::RegisterHostOnlyExcluded sx_excl_metallib_load(
    "metallib_explicit_load_matches_source");
static ::sxtest::RegisterHostOnlyExcluded sx_excl_metallib_layers(
    "metallib_explicit_throws_and_exe_adjacent_fallback");

#else // TARGET_OS_OSX — the original content follows unchanged

#include "TestHarness.h"
#include "TestUtils.h" // shared_metal_backend / count_bitwise_mismatch / SX_EXACT_I32
#include "splatx/Json.h"
#include "splatx/KernelBackend.h"
#include "splatx/MetalBackend.h"
#include "splatx/SXTensor.h"

#include <mach-o/dyld.h> // _NSGetExecutablePath — sets up the layer (b) scenario
#include <unistd.h>      // getpid — keeps the temporary output paths from colliding

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace splatx;

namespace {

namespace fs = std::filesystem;

std::string fx(const char* rel) { return sxtest::fixture_dir() + "/" + rel; }

// Runs build_metallib.sh once → the path of the metallib it produced (an empty string on failure —
// the caller FAILs). A function-local static, so it happens once per process. The output path sits
// under temp and includes the PID — this keeps concurrently running test processes from colliding
// and rebuilds fresh on every run (no stale cache; the OS cleans temp up).
const std::string& built_metallib() {
    static const std::string path = [] {
        const fs::path repo = fs::path(sxtest::kernel_src_dir()).parent_path().parent_path();
        const fs::path script = repo / "tools" / "build_metallib.sh";
        const fs::path out = fs::temp_directory_path() /
                             ("splatx-core-tests-metallib." + std::to_string(getpid())) /
                             "splatx.metallib";
        const std::string cmd =
            "/bin/sh \"" + script.string() + "\" -o \"" + out.string() + "\"";
        std::printf("  INFO running: %s\n", cmd.c_str());
        std::fflush(stdout); // guarantees ordering against the child output of std::system
        if (std::system(cmd.c_str()) != 0) return std::string();
        std::error_code ec;
        return fs::is_regular_file(out, ec) ? out.string() : std::string();
    }();
    return path;
}

// The directory of the test executable — used to construct the layer (b) exe-adjacent scenario
// (resolved exactly as executable_dir() in MetalBackend.cpp does: weakly_canonical cleans up
// symlinks).
fs::path test_exe_dir() {
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    if (size == 0) return {};
    std::vector<char> buf(static_cast<std::size_t>(size) + 1, '\0');
    if (_NSGetExecutablePath(buf.data(), &size) != 0) return {};
    std::error_code ec;
    fs::path p = fs::weakly_canonical(fs::path(buf.data()), ec);
    if (ec) p = fs::path(buf.data());
    return p.parent_path();
}

// Removes the file when the scope ends (RAII) — so that even on an SX_CHECK failure or an
// exception exit, the splatx.metallib copied next to the exe cannot survive and contaminate the
// auto-discovery layer (b) of the CLI or of a later run.
struct RemoveFileOnExit {
    fs::path path;
    ~RemoveFileOnExit() {
        std::error_code ec;
        fs::remove(path, ec); // no-op if absent, failures ignored (best-effort cleanup)
    }
};

// ===== adam fixture helpers (same convention as AdamTests.cpp — in-place, so copy inputs) =====
struct AdamState {
    std::vector<float> param, exp_avg, exp_avg_sq;
};

AdamState load_adam_state() {
    SXTensor p = SXTensor::load(fx("adam_basic/param.sxt"));
    SXTensor m = SXTensor::load(fx("adam_basic/exp_avg.sxt"));
    SXTensor v = SXTensor::load(fx("adam_basic/exp_avg_sq.sxt"));
    return {std::vector<float>(p.f32(), p.f32() + p.count()),
            std::vector<float>(m.f32(), m.f32() + m.count()),
            std::vector<float>(v.f32(), v.f32() + v.count())};
}

// Runs one adam_basic step on backend (no valid mask — the kAdamHasValidMask=false specialization).
AdamState run_adam(KernelBackend& backend, const SXTensor& grad, int64_t n_gauss,
                   const std::map<std::string, double>& meta) {
    AdamState s = load_adam_state();
    AdamInputs in;
    in.param = s.param.data();
    in.grad = grad.f32();
    in.exp_avg = s.exp_avg.data();
    in.exp_avg_sq = s.exp_avg_sq.data();
    in.valid = nullptr;
    in.n_elems = static_cast<int64_t>(s.param.size());
    in.n_gauss = n_gauss;
    in.lr = static_cast<float>(meta.at("lr"));
    in.b1 = static_cast<float>(meta.at("b1"));
    in.b2 = static_cast<float>(meta.at("b2"));
    in.eps = static_cast<float>(meta.at("eps"));
    backend.fused_adam(in);
    return s;
}

// ===== raster_std fwd helpers (minimal fwd subset of RasterTests.cpp's loading convention) =====
struct RasterFwdFixture {
    std::map<std::string, double> meta;
    SXTensor means2d, conics, colors, opacities, flatten_ids, isect_offsets, backgrounds;
    SXTensor render_colors, render_alphas, last_ids; // goldens
    bool has_bg = false;
};

RasterFwdFixture load_raster_std() {
    RasterFwdFixture f;
    f.meta = json_parse_flat(fx("raster_std/meta.json"));
    f.means2d = SXTensor::load(fx("raster_std/means2d.sxt"));
    f.conics = SXTensor::load(fx("raster_std/conics.sxt"));
    f.colors = SXTensor::load(fx("raster_std/colors.sxt"));
    f.opacities = SXTensor::load(fx("raster_std/opacities.sxt"));
    f.flatten_ids = SXTensor::load(fx("raster_std/flatten_ids.sxt"));
    f.isect_offsets = SXTensor::load(fx("raster_std/isect_offsets.sxt"));
    f.has_bg = f.meta.at("has_background") != 0.0;
    if (f.has_bg) f.backgrounds = SXTensor::load(fx("raster_std/backgrounds.sxt"));
    f.render_colors = SXTensor::load(fx("raster_std/render_colors.sxt"));
    f.render_alphas = SXTensor::load(fx("raster_std/render_alphas.sxt"));
    f.last_ids = SXTensor::load(fx("raster_std/last_ids.sxt"));
    return f;
}

RasterizeFwdInputs make_raster_inputs(const RasterFwdFixture& f) {
    RasterizeFwdInputs in;
    in.means2d = f.means2d.f32();
    in.conics = f.conics.f32();
    in.colors = f.colors.f32();
    in.opacities = f.opacities.f32();
    if (f.has_bg) in.backgrounds = f.backgrounds.f32();
    in.isect_offsets = f.isect_offsets.i32();
    in.flatten_ids = f.flatten_ids.i32();
    in.I = static_cast<int64_t>(f.meta.at("I"));
    in.N = static_cast<int64_t>(f.meta.at("N"));
    in.n_isects = static_cast<int64_t>(f.meta.at("n_isects"));
    in.image_width = static_cast<uint32_t>(f.meta.at("width"));
    in.image_height = static_cast<uint32_t>(f.meta.at("height"));
    in.tile_size = static_cast<uint32_t>(f.meta.at("tile_size"));
    in.tile_width = static_cast<uint32_t>(f.meta.at("tile_width"));
    in.tile_height = static_cast<uint32_t>(f.meta.at("tile_height"));
    return in;
}

} // namespace

// (1) Offline build → load through the explicit path → two representative kernels match the
// goldens and the source backend.
SX_TEST(metallib_explicit_load_matches_source) {
    const std::string& lib = built_metallib();
    SX_CHECK(!lib.empty()); // stops here if build_metallib.sh failed
    if (lib.empty()) return;

    // Layer (a) only — the fallbacks are sealed off (kernel_src_dir left empty): if the load fails,
    // construction itself throws.
    MetalBackendDesc desc;
    desc.metallib_path = lib;
    desc.search_exe_dir = false;
    auto backend = make_metal_backend(nullptr, desc);
    SX_CHECK(std::string(backend->name()) == "metal(metallib)");

    // -- adam fixture: the metallib backend vs the golden + vs the source-compiled backend --
    {
        auto meta = json_parse_flat(fx("adam_basic/meta.json"));
        SXTensor grad = SXTensor::load(fx("adam_basic/grad.sxt"));
        SXTensor g_param = SXTensor::load(fx("adam_basic/param_out.sxt"));
        SXTensor g_m = SXTensor::load(fx("adam_basic/exp_avg_out.sxt"));
        SXTensor g_v = SXTensor::load(fx("adam_basic/exp_avg_sq_out.sxt"));
        const int64_t N = g_param.shape[0];

        const AdamState ml = run_adam(*backend, grad, N, meta);
        SX_ALL_CLOSE(ml.param.data(), g_param.f32(), ml.param.size(), meta, "param_out");
        SX_ALL_CLOSE(ml.exp_avg.data(), g_m.f32(), ml.exp_avg.size(), meta, "exp_avg_out");
        SX_ALL_CLOSE(ml.exp_avg_sq.data(), g_v.f32(), ml.exp_avg_sq.size(), meta,
                     "exp_avg_sq_out");

        // Same results as the source-compiled path — the gate is the meta tolerance, while the
        // bitwise agreement is only recorded as INFO (the AIR comes from the same compiler
        // frontend, so bit-identical results are expected — but that is not promoted into a gate).
        const AdamState src = run_adam(sxtest::shared_metal_backend(), grad, N, meta);
        SX_ALL_CLOSE(ml.param.data(), src.param.data(), ml.param.size(), meta, "param_out");
        SX_ALL_CLOSE(ml.exp_avg.data(), src.exp_avg.data(), ml.exp_avg.size(), meta,
                     "exp_avg_out");
        SX_ALL_CLOSE(ml.exp_avg_sq.data(), src.exp_avg_sq.data(), ml.exp_avg_sq.size(), meta,
                     "exp_avg_sq_out");
        std::printf("  INFO adam metallib vs source bitwise mismatches: param %zu/%zu, "
                    "exp_avg %zu/%zu, exp_avg_sq %zu/%zu\n",
                    sxtest::count_bitwise_mismatch(ml.param.data(), src.param.data(),
                                                   ml.param.size()),
                    ml.param.size(),
                    sxtest::count_bitwise_mismatch(ml.exp_avg.data(), src.exp_avg.data(),
                                                   ml.exp_avg.size()),
                    ml.exp_avg.size(),
                    sxtest::count_bitwise_mismatch(ml.exp_avg_sq.data(), src.exp_avg_sq.data(),
                                                   ml.exp_avg_sq.size()),
                    ml.exp_avg_sq.size());
    }

    // -- raster_std fwd: the metallib backend vs the golden + vs the source-compiled backend --
    {
        const RasterFwdFixture f = load_raster_std();
        const RasterizeFwdInputs in = make_raster_inputs(f);
        const RasterizeFwdOutputs ml = backend->rasterize_fwd(in);
        SX_CHECK(ml.render_colors.size() == f.render_colors.count());
        SX_ALL_CLOSE(ml.render_colors.data(), f.render_colors.f32(), f.render_colors.count(),
                     f.meta, "render_colors");
        SX_ALL_CLOSE(ml.render_alphas.data(), f.render_alphas.f32(), f.render_alphas.count(),
                     f.meta, "render_alphas");
        SX_EXACT_I32("last_ids", ml.last_ids, f.last_ids.i32(), f.last_ids.count());

        const RasterizeFwdOutputs src = sxtest::shared_metal_backend().rasterize_fwd(in);
        SX_ALL_CLOSE(ml.render_colors.data(), src.render_colors.data(), src.render_colors.size(),
                     f.meta, "render_colors");
        SX_ALL_CLOSE(ml.render_alphas.data(), src.render_alphas.data(), src.render_alphas.size(),
                     f.meta, "render_alphas");
        SX_EXACT_I32("last_ids(vs source)", ml.last_ids, src.last_ids.data(),
                     src.last_ids.size());
        std::printf("  INFO raster_std fwd metallib vs source bitwise mismatches: "
                    "colors %zu/%zu, alphas %zu/%zu\n",
                    sxtest::count_bitwise_mismatch(ml.render_colors.data(),
                                                   src.render_colors.data(),
                                                   src.render_colors.size()),
                    src.render_colors.size(),
                    sxtest::count_bitwise_mismatch(ml.render_alphas.data(),
                                                   src.render_alphas.data(),
                                                   src.render_alphas.size()),
                    src.render_alphas.size());
    }
}

// (2) Negative and fallback checks of the layer convention — a failing explicit path throws +
// auto-discovery (b) absent/present + all layers failing.
SX_TEST(metallib_explicit_throws_and_exe_adjacent_fallback) {
    // (2a) An explicit path (layer a) that fails = an immediate throw — the fallback layers (b/c)
    //      MUST NOT engage even when they are valid (this blocks the trap in which a typo in the
    //      path is silently replaced by a stale kernel or a source compile — review Important 1).
    {
        MetalBackendDesc desc;
        desc.metallib_path = "/nonexistent/splatx-does-not-exist.metallib";
        desc.search_exe_dir = true;          // even with (b) valid
        desc.kernel_src_dir = sxtest::kernel_src_dir(); // and (c) valid — it must still throw
        bool threw = false;
        try {
            make_metal_backend(nullptr, desc);
        } catch (const std::exception& e) {
            threw = true;
            std::printf("  INFO explicit-path-fail message: %s\n", e.what());
        }
        SX_CHECK(threw);
    }

    // (2b) Auto-discovery (layer b) absent → the source fallback. Leftovers from a previous run are
    //      removed to keep this deterministic.
    const fs::path exe_dir = test_exe_dir();
    SX_CHECK(!exe_dir.empty());
    const fs::path adjacent = exe_dir / "splatx.metallib";
    std::error_code ec;
    fs::remove(adjacent, ec); // no-op if absent
    {
        MetalBackendDesc desc;
        desc.search_exe_dir = true;
        desc.kernel_src_dir = sxtest::kernel_src_dir();
        auto backend = make_metal_backend(nullptr, desc);
        SX_CHECK(std::string(backend->name()) == "metal(source)");

        // Drive the fallback backend for real — the philox integer stage is under the
        // bit-identical contract (McmcTests is the reference).
        RngNormalInputs rin;
        rin.seed = 42;
        rin.offset = 7;
        rin.count = 64;
        const std::vector<uint32_t> a = backend->rng_philox_u32(rin);
        const std::vector<uint32_t> b = sxtest::shared_metal_backend().rng_philox_u32(rin);
        SX_CHECK(a == b);
    }

    // (2c) Auto-discovery (layer b) present → the metallib is chosen (it wins over the source
    //      fallback).
    const std::string& lib = built_metallib();
    SX_CHECK(!lib.empty());
    if (!lib.empty()) {
        fs::copy_file(lib, adjacent, fs::copy_options::overwrite_existing, ec);
        SX_CHECK(!ec);
        const RemoveFileOnExit cleanup{adjacent}; // cleans up on exception exit too — no pollution
        MetalBackendDesc desc;
        desc.search_exe_dir = true;
        desc.kernel_src_dir = sxtest::kernel_src_dir(); // even when present, (b) wins
        auto backend = make_metal_backend(nullptr, desc);
        SX_CHECK(std::string(backend->name()) == "metal(metallib)");
    }

    // (2d) All layers fail (no explicit path + auto-discovery off + no source directory)
    //      → throws with the accumulated diagnostics.
    bool threw = false;
    try {
        MetalBackendDesc none;
        none.search_exe_dir = false;
        none.kernel_src_dir = "/nonexistent/splatx-kernel-src";
        make_metal_backend(nullptr, none);
    } catch (const std::exception& e) {
        threw = true;
        std::printf("  INFO all-layers-fail message: %s\n", e.what());
    }
    SX_CHECK(threw);
}

#endif // TARGET_OS_OSX
