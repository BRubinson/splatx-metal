// The splatx train C API implementation — COLMAP load → init_splats → Trainer loop (D7) →
// PLY/preview.
// Written from scratch (the recipe numbers are in Trainer.h TrainConfig — see the gsplat
// simple_trainer derivation notice there).
#include "splatx_train_c.h"

#include "splatx/Checkpoint.h"
#include "splatx/ColmapLoader.h"
#include "splatx/ImageIO.h"
#include "splatx/MemoryGovernor.h"
#include "splatx/MetalBackend.h"
#include "splatx/PlyIO.h"
#include "splatx/SpzIO.h"
#include "splatx/ThermalGovernor.h"
#include "splatx/Trainer.h"
#include "splatx/TrainingInput.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <TargetConditionals.h> // TARGET_OS_OSX — the iOS branch of the GT cache limit (compile time)
#include <sys/sysctl.h>         // hw.memsize — queries the value injected into the GT cache limit

#ifndef KERNEL_SRC_DIR
#error "KERNEL_SRC_DIR must be defined (Package.swift cxxSettings)"
#endif

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void fill_err(char* buf, int32_t len, const std::string& msg) {
    if (!buf || len <= 0) return;
    std::snprintf(buf, static_cast<size_t>(len), "%s", msg.c_str());
}

// Auto-detects a .spz extension on --out (case-insensitive — the default macOS filesystem is
// case-insensitive). Anything other than .spz stays INRIA PLY as before (no extension is
// enforced — the existing contract is preserved).
bool wants_spz(const std::string& path) {
    if (path.size() < 4) return false;
    const std::string ext = path.substr(path.size() - 4);
    return ext.size() == 4 && (ext[0] == '.') && (ext[1] == 's' || ext[1] == 'S') &&
           (ext[2] == 'p' || ext[2] == 'P') && (ext[3] == 'z' || ext[3] == 'Z');
}

// Saves a PLY through the export filter — the assembly used by the PLY branch of the final export
// (design §5.4 — the PlyIO.h convention: remove NaN/Inf rows and sigmoid(op) <= 0.005). The single
// source of truth for the filter is the one function filter_cloud_for_export — the live PLY and
// snapshot hooks (app rounds 2 and 3, Task 1) call the filter directly inside the loop so that they
// can share one copy per tick (see the comment on that block).
// On the resident path trainer.cloud() downloads all parameters through sync_host_from_device
// (a read — Home is kept, this is not a mutation. See the location state machine comment in
// Trainer.cpp).
splatx::ExportFilterStats export_filtered_ply(splatx::Trainer& trainer,
                                              const std::string& path) {
    splatx::ExportFilterStats fst;
    const splatx::SplatCloud filtered =
        splatx::filter_cloud_for_export(trainer.cloud(), 0.005f, &fst);
    splatx::write_inria_ply(filtered, path);
    return fst;
}

// ===== Thermal state injection globals (sx_thermal_set_state — splatx_train_c.h contract) =====
// The governor lifetime is owned by the running sx_train_run stack — registration and
// deregistration are mutex-protected, and an injection made before a run starts is carried over as
// the last value (g_thermal_last_state) (the premise of exactly one concurrent training run).
std::mutex g_thermal_mu;
splatx::ThermalGovernor* g_thermal_gov = nullptr;
int32_t g_thermal_last_state = 0; // nominal

// RAII registration: on construction it wires up the global and carries over the last injected
// state; on destruction it deregisters. Declare it after the governor so that it is destroyed
// before the governor (this prevents a dangling pointer after deregistration). Deregister only when
// the registration is its own (the own_ guard — Task 4 carry-over ①: unconditionally clearing the
// global to null would, if the premise of exactly one concurrent training run were broken and
// registrations overlapped, also cut the live registration of another run. The guard pins that
// failure mode to the safe side, "whichever is destroyed later survives").
struct ThermalGovernorRegistration {
    splatx::ThermalGovernor* own_;
    explicit ThermalGovernorRegistration(splatx::ThermalGovernor& gov) : own_(&gov) {
        std::lock_guard<std::mutex> lk(g_thermal_mu);
        if (g_thermal_last_state != 0)
            gov.set_thermal_state(static_cast<splatx::ThermalState>(g_thermal_last_state));
        g_thermal_gov = &gov;
    }
    ~ThermalGovernorRegistration() {
        std::lock_guard<std::mutex> lk(g_thermal_mu);
        if (g_thermal_gov == own_) g_thermal_gov = nullptr;
    }
    ThermalGovernorRegistration(const ThermalGovernorRegistration&) = delete;
    ThermalGovernorRegistration& operator=(const ThermalGovernorRegistration&) = delete;
};

// ===== Training control globals (Task 5 — sx_train_request_stop/pause_checkpoint surface) =====
// The control block is owned by the running run stack. g_ctl_mu serializes registration,
// deregistration, and external access — sx_train_pause_checkpoint takes the Trainer snapshot while
// holding the mutex, so deregistration (teardown) blocks for that duration and the trainer and
// governor lifetimes are guaranteed (a parked loop cannot finish — the hold latch prevents the
// wake). A stop request never blocks on the mutex — see the g_stop_pending flag plus try-lock
// delivery below (a request arriving during a save is delivered once the save completes).
struct TrainControl {
    splatx::ThermalGovernor* gov;
    splatx::Trainer* trainer;
    int factor;
    int test_split;
    std::atomic<int32_t> completed{0};   // updated at step boundaries — the snapshot iter while parked
    std::atomic<bool> loop_active{false}; // loop entry to exit (the stretch where park is reachable)
    TrainControl(splatx::ThermalGovernor& g, splatx::Trainer& t, int f, int ts)
        : gov(&g), trainer(&t), factor(f), test_split(ts) {}
};
std::mutex g_ctl_mu;
TrainControl* g_ctl = nullptr;

// The pending-stop flag (Task 5 review 2 — making sx_train_request_stop non-blocking):
// so that the stop-calling thread (the Task cancellation handler) is not held while
// pause_checkpoint saves (several seconds) with g_ctl_mu held, the request is recorded to the flag
// first and delivery is attempted with a try-lock. When the try-lock fails, delivery is performed
// by **the mutex holder** as it leaves the critical section (deliver_pending_stop_locked — called
// at the exit of every g_ctl_mu critical section).
// If there is no active run the flag is discarded (it is not carried over to the next run — the
// header contract).
std::atomic<bool> g_stop_pending{false};

// Call with g_ctl_mu held: deliver the pending stop to the current run, or discard it.
void deliver_pending_stop_locked() {
    if (!g_stop_pending.exchange(false, std::memory_order_acq_rel)) return;
    if (g_ctl) g_ctl->gov->request_external_stop();
    // no g_ctl = the target run has already finished — discard it (it MUST NOT be carried over)
}

struct TrainControlRegistration {
    TrainControl* own_;
    explicit TrainControlRegistration(TrainControl& ctl) : own_(&ctl) {
        std::lock_guard<std::mutex> lk(g_ctl_mu);
        // A stop flag left over from a previous run (or from no run at all) is not carried over
        // to the new run (header contract)
        g_stop_pending.store(false, std::memory_order_release);
        g_ctl = &ctl;
    }
    ~TrainControlRegistration() {
        std::lock_guard<std::mutex> lk(g_ctl_mu); // wait for in-flight external access to finish
        if (g_ctl == own_) g_ctl = nullptr;       // the own_ guard — same rule as carry-over ①
        // An undelivered stop aimed at this run dies here (the run has finished — there is no
        // delivery target)
        g_stop_pending.store(false, std::memory_order_release);
    }
    TrainControlRegistration(const TrainControlRegistration&) = delete;
    TrainControlRegistration& operator=(const TrainControlRegistration&) = delete;
};

// The SPLATX_THERMAL_INJECT parser: "OFFSET_MS:STATE[,...]" — STATE is nominal|fair|serious|
// critical or 0..3. A malformed spec throws (even on a hidden test path, silent misinterpretation
// is forbidden).
std::vector<std::pair<int, int>> parse_thermal_inject(const std::string& spec) {
    std::vector<std::pair<int, int>> out;
    size_t pos = 0;
    while (pos <= spec.size()) {
        size_t comma = spec.find(',', pos);
        if (comma == std::string::npos) comma = spec.size();
        const std::string item = spec.substr(pos, comma - pos);
        pos = comma + 1;
        if (item.empty()) continue;
        const size_t colon = item.find(':');
        if (colon == std::string::npos)
            throw std::invalid_argument("SPLATX_THERMAL_INJECT: 'ms:state' format required — " + item);
        const int off = std::stoi(item.substr(0, colon));
        if (off < 0) throw std::invalid_argument("SPLATX_THERMAL_INJECT: negative offset — " + item);
        const std::string st = item.substr(colon + 1);
        int state = -1;
        if (st == "nominal" || st == "0") state = 0;
        else if (st == "fair" || st == "1") state = 1;
        else if (st == "serious" || st == "2") state = 2;
        else if (st == "critical" || st == "3") state = 3;
        else throw std::invalid_argument("SPLATX_THERMAL_INJECT: unknown state — " + st);
        out.emplace_back(off, state);
    }
    std::stable_sort(out.begin(), out.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    return out;
}

// The hidden test injection thread (CLI/E2E — this path exists because macOS has no thermal
// injection source): it calls sx_thermal_set_state at ms offsets measured from the start of
// training — going through the same surface as a real injection (from an arbitrary thread). The
// destructor stops and then joins (declare and destroy it before the governor deregistration).
struct ThermalInjector {
    std::thread th;
    std::atomic<bool> stop{false};
    void start(std::vector<std::pair<int, int>> sched) {
        if (sched.empty()) return;
        th = std::thread([this, sched = std::move(sched)] {
            const auto t0 = Clock::now();
            for (const auto& [off_ms, state] : sched) {
                while (!stop.load(std::memory_order_relaxed) &&
                       ms_since(t0) < static_cast<double>(off_ms))
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                if (stop.load(std::memory_order_relaxed)) return;
                std::printf("thermal-inject: +%d ms -> state %d\n", off_ms, state);
                std::fflush(stdout);
                sx_thermal_set_state(state);
            }
        });
    }
    ~ThermalInjector() {
        stop.store(true);
        if (th.joinable()) th.join();
    }
};

} // namespace

extern "C" void sx_thermal_set_state(int32_t state) {
    if (state < 0 || state > 3) return; // robustness of the observation path — ignore (header contract)
    std::lock_guard<std::mutex> lk(g_thermal_mu);
    g_thermal_last_state = state;
    if (g_thermal_gov)
        g_thermal_gov->set_thermal_state(static_cast<splatx::ThermalState>(state));
}

extern "C" int32_t sx_train_run_args(const SxTrainArgs* args, char* err_buf,
                                     int32_t err_buf_len) {
    try {
        if (!args) throw std::invalid_argument("args must be given");
        // Unpack (names consistent with the existing 25-argument body — the auto-cap and
        // default-value paths reset some of these)
        const char* data_dir = args->data_dir;
        const int32_t factor = args->factor;
        int32_t iters = args->iters;              // reset to the checkpoint value on extended resume
        int64_t max_points = args->max_points;    // auto-cap may lower it to cap
        const int32_t sh_degree = args->sh_degree;
        const char* out_ply = args->out_path;
        const char* preview_png = args->preview_png;
        const int32_t eval_image = args->eval_image;
        const char* kernel_src_dir = args->kernel_src_dir;
        const char* metallib_path = args->metallib_path;
        int32_t log_every = args->log_every;
        const int32_t mcmc = args->mcmc;
        int64_t cap_max = args->cap_max;          // auto-cap resets it to the estimated value
        const int32_t test_split = args->test_split;
        const int32_t eval_every = args->eval_every;
        const float ssim_weight = args->ssim_weight;
        const int32_t antialiased = args->antialiased;
        const int32_t resident = args->resident;
        const char* resume_checkpoint = args->resume_checkpoint;
        const char* save_checkpoint_path = args->save_checkpoint_path;
        const int32_t checkpoint_every = args->checkpoint_every;
        const int32_t auto_cap = args->auto_cap;
        const int64_t memory_budget_bytes = args->memory_budget_bytes;
        const int32_t thermal_mode = args->thermal_mode;
        const int32_t extend_iters = args->resume_extend_iters;
        // live PLY export (app round 2 Task 1 — the contract is the comment at the end of
        // SxTrainArgs in splatx_train_c.h): enabled only when both path and every are valid.
        // No validation throw — an off combination is valid input.
        const int32_t live_ply_every = args->live_ply_every;
        const bool live_on =
            args->live_ply_path && *args->live_ply_path && live_ply_every > 0;
        const std::string live_ply_path = live_on ? std::string(args->live_ply_path)
                                                  : std::string();
        // The in-memory snapshot callback (app round 3 Task 1 — the authoritative contract is the
        // SxSplatSnapshot comment in splatx_train_c.h): enabled only when both cb and every are
        // valid (the same rule as live PLY — an off combination is valid input, and there is no
        // validation throw).
        const int32_t snapshot_every = args->snapshot_every;
        const bool snapshot_on = args->snapshot_cb && snapshot_every > 0;
        const float depth_weight = args->depth_weight;

        if (!data_dir || !out_ply) throw std::invalid_argument("data_dir/out_ply must be given");
        if (thermal_mode < 0 || thermal_mode > 2)
            throw std::invalid_argument(
                "thermal_mode must be 0(performance)|1(balanced)|2(lowheat)");
        if (factor != 1 && factor != 2 && factor != 4 && factor != 8)
            throw std::invalid_argument("factor must be 1|2|4|8");
        if (extend_iters < 0) throw std::invalid_argument("resume_extend_iters must be >= 0");
        if (extend_iters > 0 && !resume_checkpoint)
            throw std::invalid_argument("resume_extend_iters requires resume_checkpoint");
        // Extended resume (extend_iters > 0) ignores iters and uses the checkpoint's total_iters
        // (the semantics comment on SxTrainArgs.resume_extend_iters in the header) — every other
        // path keeps the previous validation.
        if (extend_iters == 0 && iters < 1) throw std::invalid_argument("iters must be >= 1");
        if (auto_cap && !mcmc)
            throw std::invalid_argument("auto_cap requires mcmc (cap_max is an MCMC arena size)");
        if (mcmc && !auto_cap && cap_max < 1)
            throw std::invalid_argument("cap_max must be >= 1");
        if (checkpoint_every < 0) throw std::invalid_argument("checkpoint_every must be >= 0");
        if (checkpoint_every > 0 && !save_checkpoint_path)
            throw std::invalid_argument("checkpoint_every > 0 requires save_checkpoint_path");
        if (test_split < 0) throw std::invalid_argument("test_split must be >= 0");
        if (test_split == 1)
            throw std::invalid_argument("test_split 1 leaves no train views (i % 1 == 0 all test)");
        if (eval_every < 0) throw std::invalid_argument("eval_every must be >= 0");
        if (eval_every > 0 && test_split <= 0)
            throw std::invalid_argument("--eval-every requires --test-split > 0 (no test views)");
        if (!(ssim_weight >= 0.0f && ssim_weight <= 1.0f)) // rejects NaN (negated form — same as the Trainer guard)
            throw std::invalid_argument("ssim_weight must be in [0,1]");
        if (!(depth_weight >= 0.0f && std::isfinite(depth_weight)))
            throw std::invalid_argument("depth_weight must be finite and >= 0");
        if (log_every < 1) log_every = 10;

        // Extended resume: for the ratio-based schedule denominator (total_iters) the checkpoint is
        // authoritative — only the header is read up front to inherit iters (a few KB; the full
        // load and the fingerprint verification stay on the previous resume path below. After this,
        // cur.total_iters = iters trivially matches the checkpoint value = the "same run"
        // semantics).
        if (extend_iters > 0)
            iters = splatx::load_checkpoint_header(resume_checkpoint).config.total_iters;

        const std::string data(data_dir);
        // Selects the images_{factor} folder, falling back to downscale-on-load from the original
        // when it is absent (integration Task 2 — an iOS capture app bundle has no pre-resized
        // folder. The convention is in TrainingInput.h).
        const splatx::ImagesDirChoice img_choice = splatx::resolve_images_dir(data, factor);
        if (img_choice.downscale_on_load) {
            std::printf("images_%d not found — downscaling images/ by factor %d on load\n",
                        factor, factor);
            std::fflush(stdout);
        }

        // COLMAP load → TrainingInput (the intrinsics are relative to the original resolution — scaled below)
        const splatx::ColmapBundle bundle = splatx::load_colmap_bundle(data + "/sparse/0");
        splatx::TrainingInput ti = splatx::training_input_from_colmap(bundle, img_choice.dir);
        if (ti.cameras.empty()) throw std::runtime_error("no cameras in COLMAP bundle");

        // Sorted by image name (the gsplat datasets/colmap.py:194–199 convention — the storage
        // order of images.bin is arbitrary, so the sort is required for the split index
        // (i % test_split) to point at the same cameras as gsplat. The sort is kept even when the
        // split is off — reproducibility of the traversal order). It is one folder, so sorting by
        // path == sorting by name.
        std::sort(ti.cameras.begin(), ti.cameras.end(),
                  [](const splatx::TrainingCamera& a, const splatx::TrainingCamera& b) {
                      return a.image_path < b.image_path;
                  });

        // --factor: scales K, width, and height by 1/factor (task-6 handover 4 — the COLMAP
        // intrinsics are relative to the original resolution, and training_input_from_colmap does
        // not scale them). The actual size of a downsample folder follows the round(orig/factor)
        // convention — fx/cx and fy/cy are scaled exactly by the per-axis ratio. The dimension
        // convention has its single source of truth in downscaled_dim (ImageIO.h) — shared with
        // the fallback downscaling loader so that the fallback path also passes through the GT
        // dims guard (the formula is the same as the previous lround(orig/factor)).
        if (factor > 1) {
            for (auto& cam : ti.cameras) {
                const int w0 = cam.width, h0 = cam.height;
                const int w1 = splatx::downscaled_dim(w0, factor);
                const int h1 = splatx::downscaled_dim(h0, factor);
                if (w1 < 1 || h1 < 1) throw std::runtime_error("factor too large for image dims");
                const double sx = static_cast<double>(w1) / w0, sy = static_cast<double>(h1) / h0;
                cam.K[0] = static_cast<float>(cam.K[0] * sx); // fx
                cam.K[2] = static_cast<float>(cam.K[2] * sx); // cx
                cam.K[4] = static_cast<float>(cam.K[4] * sy); // fy
                cam.K[5] = static_cast<float>(cam.K[5] * sy); // cy
                cam.width = w1;
                cam.height = h1;
            }
        }

        // The GT cache on/off decision (Phase 6 round 2 Task 3 — moved up from just before
        // training: the auto-cap fixed component and the gt_cache vector must share **the same
        // decision** for the estimate and the actual residency to be consistent).
        // The cache covers all train + test views (eval goes through the cache too — load_gt).
        std::size_t gt_bytes_estimate = 0;
        int train_w = 0, train_h = 0; // training resolution (after factor, the camera max — conservative)
        for (const auto& cam : ti.cameras) {
            gt_bytes_estimate +=
                static_cast<std::size_t>(cam.width) * cam.height * 3 * sizeof(float);
            train_w = std::max(train_w, cam.width);
            train_h = std::max(train_h, cam.height);
        }
        // The limit is adaptive (training-speed Task 2 — the computation contract is the
        // sx_gt_cache_limit_bytes header comment): physical RAM is queried here with sysctl
        // hw.memsize and injected into the pure function (on failure 0 → the function guards with
        // the 4 GiB floor), and the iOS branch is the compile-time TARGET_OS_OSX.
        uint64_t hw_memsize = 0;
        {
            size_t memsize_len = sizeof(hw_memsize);
            if (sysctlbyname("hw.memsize", &hw_memsize, &memsize_len, nullptr, 0) != 0)
                hw_memsize = 0;
        }
#if TARGET_OS_OSX
        constexpr int32_t kGtCacheIos = 0;
#else
        constexpr int32_t kGtCacheIos = 1;
#endif
        const std::size_t gt_cache_limit =
            static_cast<std::size_t>(sx_gt_cache_limit_bytes(hw_memsize, kGtCacheIos));
        const bool cache_gt = gt_bytes_estimate <= gt_cache_limit;

        // --auto-cap: automatic cap_max estimation (design §7.2 — the formula and the rationale for
        // the constants are in MemoryGovernor.h). Caller injection of the budget is first-class
        // (iOS: os_proc_available_memory); without it the macOS fallback applies (50% of sysctl
        // hw.memsize — MemoryGovernor.cpp). The basis of the estimate is left in a one-line log.
        if (mcmc && auto_cap) {
            const int64_t budget = memory_budget_bytes > 0
                                       ? memory_budget_bytes
                                       : splatx::default_memory_budget_bytes();
            const splatx::CapMaxSuggestion s = splatx::suggest_cap_max(
                budget, train_w, train_h, static_cast<int>(ti.cameras.size()), sh_degree,
                cache_gt);
            cap_max = s.cap_max;
            const auto mib = [](int64_t b) { return static_cast<double>(b) / 1048576.0; };
            std::printf("auto-cap: cap_max %lld <- (budget %.1f MiB (%s) x %lld/%lld = "
                        "usable %.1f MiB - fixed %.1f MiB (gt cache %.1f + image %.1f + "
                        "isect %.1f + base %.1f)) / %lld B/splat (sh_degree %d)%s%s\n",
                        static_cast<long long>(s.cap_max), mib(budget),
                        memory_budget_bytes > 0 ? "injected" : "sysctl hw.memsize/2 fallback",
                        static_cast<long long>(splatx::kCapSafetyNumer),
                        static_cast<long long>(splatx::kCapSafetyDenom), mib(s.usable_bytes),
                        mib(s.fixed_bytes()), mib(s.gt_cache_bytes), mib(s.image_bytes),
                        mib(s.isect_bytes), mib(s.base_bytes),
                        static_cast<long long>(s.bytes_per_splat), sh_degree,
                        s.clamped_min ? " [clamped to min 100000 — budget too small]" : "",
                        s.clamped_max ? " [clamped to max 4000000]" : "");
            // If the initial N > cap the Trainer throws (the cap_max >= initial N contract) —
            // lower the init subsample cap to cap as a pre-emptive guard (irrelevant on the resume
            // path, which does not apply max_points).
            if (max_points > cap_max) {
                std::printf("auto-cap: max_points %lld -> %lld (init subsample cap <= cap_max)\n",
                            static_cast<long long>(max_points),
                            static_cast<long long>(cap_max));
                max_points = cap_max;
            }
            std::fflush(stdout);
        }

        // scene_scale (based on the spread of the camera centers — the gsplat convention) + initialization
        splatx::TrainConfig cfg;
        cfg.total_iters = iters;
        cfg.scene_scale = splatx::scene_scale_from_cameras(ti.cameras);
        cfg.ssim_weight = ssim_weight; // λ — loss = (1-λ)·L1 + λ·(1-SSIM) (Phase 3 Task 3)
        // AA (Phase 3 Task 4 — reproduces gsplat rasterize_mode="antialiased"): isect and raster
        // consume opacity_aa = sigmoid(op)·comp. Default off = consistent with the gsplat default
        // "classic".
        cfg.antialiased = antialiased != 0;
        // device-resident train_step (Phase 3 Task 5 — default on, with the --no-resident escape hatch)
        cfg.resident = resident != 0;
        cfg.depth_weight = depth_weight;

        splatx::InitOptions io;
        io.max_points = max_points;
        io.sh_degree = sh_degree;

        // --mcmc: automatically applies the gsplat "mcmc" preset (simple_trainer.py:1579–1587 —
        // init_opa 0.5 / init_scale 0.1 / opacity_reg and scale_reg 0.01 / strategy on) + cap_max.
        if (mcmc) {
            splatx::apply_mcmc_presets(io, cfg);
            cfg.mcmc.cap_max = cap_max;
        }

        const std::string mcmc_desc =
            mcmc ? "on (cap_max " + std::to_string(cap_max) + ")" : std::string("off");
        std::printf("splatx train: %zu cameras, %zu seed points, factor %d (%dx%d), "
                    "scene_scale %.4f, mcmc %s, ssim_weight %.2f, antialiased %s, "
                    "resident %s\n",
                    ti.cameras.size(), ti.seed_points.size() / 3, factor, ti.cameras[0].width,
                    ti.cameras[0].height, static_cast<double>(cfg.scene_scale),
                    mcmc_desc.c_str(), static_cast<double>(ssim_weight),
                    cfg.antialiased ? "on" : "off", cfg.resident ? "on" : "off");
        std::fflush(stdout);

        // --resume: loads the checkpoint and verifies the config fingerprint up front **against the
        // values the user requested** (design §7.5/§8.1 — the internal fingerprint of
        // Trainer::restore_checkpoint is based on the cloud built from ck, so a mismatch in the
        // requested sh_degree is caught only here). On success init_splats (KNN) is skipped — the
        // checkpoint is authoritative for the parameters, and the training loop re-enters from
        // ck.iter (the ratio-based schedule stays continuous).
        std::optional<splatx::TrainerCheckpoint> resume;
        int start_iter = 0;
        if (resume_checkpoint) {
            resume = splatx::load_checkpoint_file(resume_checkpoint);
            splatx::CheckpointConfig cur;
            cur.sh_degree = sh_degree;
            cur.factor = factor;
            cur.test_split = test_split; // the split convention — prevents view-stream/eval
                                         // contamination (review fix 1)
            cur.total_iters = iters;
            cur.mcmc_enabled = mcmc ? 1 : 0;
            cur.antialiased = cfg.antialiased ? 1 : 0;
            cur.cap_max = mcmc ? cap_max : 0;
            cur.mcmc_seed = mcmc ? cfg.mcmc.seed : 0;
            cur.scene_scale = cfg.scene_scale;
            cur.ssim_weight = cfg.ssim_weight;
            splatx::verify_checkpoint_config(resume->config, cur);
            start_iter = resume->iter;
            // Extended resume means "K more iters", so there is always something left to train —
            // only the previous path performs the refusal check.
            if (extend_iters == 0 && start_iter >= iters)
                throw std::invalid_argument(
                    "resume checkpoint already contains " + std::to_string(start_iter) +
                    " iters (>= --iters " + std::to_string(iters) +
                    ") — nothing to train (to extend, use resume_extend_iters/--extend-iters)");
            std::printf("resume: %s (iter %d/%d, N %lld — init_splats/KNN skipped, "
                        "max_points not applied%s)\n",
                        resume_checkpoint, start_iter, iters,
                        static_cast<long long>(resume->cloud.count),
                        extend_iters > 0 ? ", extended resume" : "");
            std::fflush(stdout);
        }
        // The iteration this run ends at. Extended resume proceeds past total_iters (the schedule
        // denominator) — in the p = iter/total > 1 region lr keeps decaying monotonically, SH is
        // saturated, and refine does not run = pure refinement (the rationale for the semantics is
        // the SxTrainArgs.resume_extend_iters comment in splatx_train_c.h).
        const int run_until = extend_iters > 0 ? start_iter + extend_iters : iters;
        if (extend_iters > 0) {
            std::printf("extend: %d + %d -> run_until %d (schedule denominator total_iters fixed "
                        "at %d — pure refinement outside the refine window)\n",
                        start_iter, extend_iters, run_until, iters);
            std::fflush(stdout);
        }

        splatx::SplatCloud cloud;
        if (resume) {
            cloud = resume->cloud; // the checkpoint is authoritative for the parameters
                                   // (the moments and the RNG come from restore)
        } else {
            auto t0 = Clock::now();
            cloud = splatx::init_splats(ti, io);
            std::printf("init_splats: N=%lld (max %lld), sh_degree %d, KNN %.0f ms\n",
                        static_cast<long long>(cloud.count), static_cast<long long>(max_points),
                        sh_degree, ms_since(t0));
            std::fflush(stdout);
        }

        // Layered loading (the MetalBackend.h convention): with --metallib given, only that path is
        // used — a load failure throws immediately (falling back to another layer is forbidden).
        // Without it, splatx.metallib next to the executable (the relocatable distribution of
        // record) is searched automatically → only if it is absent does the source compilation
        // fallback apply. The chosen layer is reported in name().
        splatx::MetalBackendDesc bdesc;
        if (metallib_path) bdesc.metallib_path = metallib_path;
        bdesc.search_exe_dir = true;
        bdesc.kernel_src_dir = kernel_src_dir ? std::string(kernel_src_dir) : KERNEL_SRC_DIR;
        const auto t_lib = Clock::now();
        const std::unique_ptr<splatx::KernelBackend> backend =
            splatx::make_metal_backend(nullptr, bdesc);
        std::printf("metal backend ready (%s, library load %.0f ms)\n", backend->name(),
                    ms_since(t_lib));
        std::fflush(stdout);

        splatx::Trainer trainer(*backend, std::move(cloud), cfg);
        // resume: restores the Adam moments and the MCMC RNG (+ re-verifies the internal Trainer
        // fingerprint — defense in depth). Released immediately after restoration — this prevents
        // ~700MB of params plus moments from staying resident at 1M (review fix 3).
        if (resume) {
            trainer.restore_checkpoint(*resume, factor, test_split);
            resume.reset();
        }

        // train/test split (Trainer.h make_train_test_split — the MipNeRF360 convention: the
        // name-sorted indices with i % test_split == 0 are test, gsplat colmap.py:456–461).
        // With 0 everything is train.
        const splatx::TrainTestSplit split = splatx::make_train_test_split(
            static_cast<int64_t>(ti.cameras.size()), test_split);
        if (split.train.empty())
            throw std::runtime_error("train split is empty — need more cameras or test_split 0");
        if (test_split > 0) {
            std::printf("split: %zu train / %zu test (index %% %d == 0 -> test)\n",
                        split.train.size(), split.test.size(), test_split);
            std::fflush(stdout);
        }

        // GT decode + the dims-vs-camera check (task-6 handover Minor — a guard against EXIF
        // orientation rotation). The fallback path downscales the original by factor at decode
        // time — since the target dimensions are derived from the native dimensions with
        // downscaled_dim (ImageIO.h), an original-vs-camera mismatch such as an EXIF rotation is
        // caught by the guard below exactly as before. Measured difference between the resampling
        // kernels (garden factor 4, 300 iter, test-split 8): images_4 (pre-resized) loss
        // 0.1707/PSNR 20.21 dB vs. the fallback (CGImageSource downscale) loss 0.1569/PSNR
        // 20.75 dB — the fallback GT is smoother, so the numbers measured against its own GT come
        // out slightly favorable (run-to-run noise ±0.001 dB). This is not a difference in
        // absolute quality but a difference in the GT reference — details are in internal
        // measurement records.
        const bool downscale_on_load = img_choice.downscale_on_load;
        const auto decode_gt = [factor, downscale_on_load](const splatx::TrainingCamera& cam) {
            splatx::ImageF32 gt =
                downscale_on_load ? splatx::load_image_rgb_f32_downscaled(cam.image_path, factor)
                                  : splatx::load_image_rgb_f32(cam.image_path);
            if (gt.width != cam.width || gt.height != cam.height)
                throw std::runtime_error(
                    "GT image dims " + std::to_string(gt.width) + "x" + std::to_string(gt.height) +
                    " != camera " + std::to_string(cam.width) + "x" + std::to_string(cam.height) +
                    " (" + cam.image_path + ") — suspect a factor mismatch or an EXIF orientation rotation");
            return gt;
        };

        // The GT cache (Phase 3 Task 5 — one decode per view): decoding JPEG/PNG every iter is the
        // dominant cost relative to a resident step, so the decode results are kept in RAM (decode
        // is deterministic — the training result is unchanged whether the cache is on or off). If
        // the total estimate exceeds the limit, it decodes every iter as before (a guard for large
        // scenes). The limit is adaptive (training-speed Task 2): macOS uses max(4 GiB, 50% of
        // physical RAM) and iOS a fixed 4 GiB (conservative for jetsam) — with the previous fixed
        // 4 GiB, real use at factor 2 (an estimated 9.7 GB for garden) exceeded the limit and
        // degenerated into decoding every step (the 2026-07-21 training-speed diagnosis). The
        // computation contract and its rationale are in sx_gt_cache_limit_bytes
        // (splatx_train_c.h). The decision (cache_gt) is shared with the auto-cap fixed
        // component — made once above. The limit is printed alongside in the log so that the
        // reason for on/off can be diagnosed from the user's log alone.
        std::vector<splatx::ImageF32> gt_cache(cache_gt ? ti.cameras.size() : 0);
        std::printf("gt cache: %s (estimated %.2f GiB, limit %.2f GiB)\n",
                    cache_gt ? "on" : "off",
                    static_cast<double>(gt_bytes_estimate) / (1024.0 * 1024.0 * 1024.0),
                    static_cast<double>(gt_cache_limit) / (1024.0 * 1024.0 * 1024.0));
        splatx::ImageF32 gt_scratch; // the reuse slot for the cache-off path (single-threaded loop)
        const auto load_gt = [&](std::size_t cam_idx) -> const splatx::ImageF32& {
            if (!cache_gt) {
                gt_scratch = decode_gt(ti.cameras[cam_idx]);
                return gt_scratch;
            }
            splatx::ImageF32& slot = gt_cache[cam_idx];
            if (slot.rgb.empty()) slot = decode_gt(ti.cameras[cam_idx]);
            return slot;
        };

        // Depth sidecars (DepthSupervision.h): decoded once per view on first use and kept (the
        // ARKit maps are 256×192 floats). They supervise train views when depth_weight > 0 and give
        // the held-out depth RMSE whenever test views have them.
        std::size_t n_depth_views = 0;
        for (const auto& cam : ti.cameras) n_depth_views += cam.depth_path.empty() ? 0 : 1;
        if (n_depth_views > 0 || depth_weight > 0.0f) {
            std::printf("depth sidecars: %zu/%zu views, depth_weight %.3f\n", n_depth_views,
                        ti.cameras.size(), static_cast<double>(depth_weight));
            std::fflush(stdout);
        }
        std::vector<splatx::DepthTarget> depth_cache(ti.cameras.size());
        std::vector<char> depth_loaded(ti.cameras.size(), 0);
        const auto load_depth = [&](std::size_t cam_idx) -> const splatx::DepthTarget* {
            const splatx::TrainingCamera& cam = ti.cameras[cam_idx];
            if (cam.depth_path.empty()) return nullptr;
            if (!depth_loaded[cam_idx]) {
                splatx::DepthTarget t =
                    splatx::load_depth_sidecar(cam.depth_path, cam.confidence_path);
                t.view_valid = splatx::depth_valid_count(t, cam.width, cam.height);
                depth_cache[cam_idx] = std::move(t);
                depth_loaded[cam_idx] = 1;
            }
            return &depth_cache[cam_idx];
        };

        // eval: renders every test view at full sh_degree → clamps to [0,1] → takes the arithmetic
        // mean of the per-view PSNRs (gsplat simple_trainer.py:1240 clamp + :1254/:1271 per-view
        // psnr then mean — not a pooled MSE).
        // cc reported alongside (color-correction Task 2 — spec 2026-07-15-color-correct-eval §4):
        // the clamped render is aligned to the GT with color_correct_affine (Trainer.h — per-channel
        // affine appearance alignment) and the resulting cc PSNR is computed side by side under the
        // same averaging convention (corresponding to gsplat simple_trainer.py:1258–1266 — an
        // **additional** metric that does not replace raw, with no flag). The return value stays
        // the raw mean as before (→ last_eval_psnr/final_psnr/the progress tick p.psnr) — cc
        // travels only through the out-param.
        // SSIM and ccSSIM reported alongside (perceptual-metrics Task 2 — spec
        // 2026-07-15-perceptual-eval §2–§3): eval_ssim (Trainer.h — the same convention as the
        // training loss) is reported on the same clamped render and cc (corresponding to gsplat
        // simple_trainer.py:1255–1281). Passing backend routes it through the GPU ssim_fwd
        // (training-speed Task 1 — the previous synchronous ssim_fwd_cpu execution was the
        // dominant term of the eval tick: ~0.7G FMA per view, twice. The inputs, the clamp, the cc
        // ordering, and the meaning of the two calls are unchanged — only the engine is swapped,
        // and the metric may deviate from the CPU one at the floating-point level). The argument
        // order is fixed as (img=render|cc, ref=GT) — swapping them changes the lowest digits
        // because of the f32 operation order. Views with h<11||w<11 are NaN (not measurable) and
        // are excluded from the SSIM mean (the guard precedes the backend validator throw); with
        // zero valid views the mean is NaN as well.
        const auto run_eval = [&](int iter_done, double& cc_mean_out, double& ssim_mean_out,
                                  double& cc_ssim_mean_out, double& depth_rmse_out) -> double {
            double sum = 0.0, cc_sum = 0.0, ssim_sum = 0.0, cc_ssim_sum = 0.0;
            splatx::DepthErrorSum depth_err;
            size_t ssim_views = 0; // count excluding views with h/w < 11 (the NaN convention)
            std::vector<float> cc; // buffer reused across views (out must be a separate buffer — not in-place)
            for (int64_t t : split.test) {
                const splatx::TrainingCamera& cam = ti.cameras[static_cast<size_t>(t)];
                const splatx::ImageF32& gt = load_gt(static_cast<size_t>(t));
                const splatx::DepthTarget* dt = load_depth(static_cast<size_t>(t));
                std::vector<float> render;
                if (dt) {
                    splatx::DepthRender dr = trainer.render_with_depth(cam);
                    splatx::accumulate_depth_error(dr.depth.data(), *dt, cam.width, cam.height,
                                                   depth_err);
                    render = std::move(dr.rgb);
                } else {
                    render = trainer.render(cam);
                }
                for (float& v : render) v = std::clamp(v, 0.0f, 1.0f);
                const int64_t n = static_cast<int64_t>(render.size());
                sum += splatx::psnr(render.data(), gt.rgb.data(), n);
                cc.resize(render.size());
                splatx::color_correct_affine(render.data(), gt.rgb.data(), n, cc.data());
                cc_sum += splatx::psnr(cc.data(), gt.rgb.data(), n);
                const double s = splatx::eval_ssim(render.data(), gt.rgb.data(),
                                                   static_cast<int32_t>(cam.height),
                                                   static_cast<int32_t>(cam.width),
                                                   backend.get());
                if (std::isfinite(s)) {
                    ssim_sum += s;
                    cc_ssim_sum += splatx::eval_ssim(cc.data(), gt.rgb.data(),
                                                     static_cast<int32_t>(cam.height),
                                                     static_cast<int32_t>(cam.width),
                                                     backend.get());
                    ssim_views++;
                }
            }
            const double mean = sum / static_cast<double>(split.test.size());
            const double cc_mean = cc_sum / static_cast<double>(split.test.size());
            const double nan = std::numeric_limits<double>::quiet_NaN();
            const double ssim_mean = ssim_views ? ssim_sum / static_cast<double>(ssim_views) : nan;
            const double cc_ssim_mean =
                ssim_views ? cc_ssim_sum / static_cast<double>(ssim_views) : nan;
            const double depth_rmse =
                depth_err.n > 0 ? std::sqrt(depth_err.sq / static_cast<double>(depth_err.n)) : nan;
            std::printf("eval  iter %5d/%d  PSNR %.3f dB | cc %.3f dB | SSIM %.4f | ccSSIM %.4f "
                        "(%zu test views)\n",
                        iter_done, run_until, mean, cc_mean, ssim_mean, cc_ssim_mean,
                        split.test.size());
            if (depth_err.n > 0)
                std::printf("eval  iter %5d/%d  depth RMSE %.4f m (%lld sidecar samples)\n",
                            iter_done, run_until, depth_rmse,
                            static_cast<long long>(depth_err.n));
            std::fflush(stdout);
            depth_rmse_out = depth_rmse;
            cc_mean_out = cc_mean;
            ssim_mean_out = ssim_mean;
            cc_ssim_mean_out = cc_ssim_mean;
            return mean;
        };

        // Checkpoint save — at a step-boundary safe point (the Trainer.h make_checkpoint contract).
        // fp32 by default (lossless — the reference for desktop resume); with
        // args->checkpoint_fp16 nonzero it is fp16 half precision (the iOS surface — the lossy
        // contract in Checkpoint.h). completed = the completed step count (= the iter that resume
        // re-enters at). The time is logged — for observing the §7.5 3-second contract.
        const bool ck_fp16 = args->checkpoint_fp16 != 0;
        const auto save_ck = [&](int completed) {
            const auto t_ck = Clock::now();
            const splatx::TrainerCheckpoint ck =
                trainer.make_checkpoint(completed, factor, test_split);
            splatx::save_checkpoint_file(ck, save_checkpoint_path, ck_fp16);
            std::printf("checkpoint saved: %s (iter %d, N %lld, %s, %.0f ms)\n",
                        save_checkpoint_path, completed,
                        static_cast<long long>(ck.cloud.count), ck_fp16 ? "fp16" : "fp32",
                        ms_since(t_ck));
            std::fflush(stdout);
        };

        // The thermal governor (design §7.4 — the policy contract of record is ThermalGovernor.h):
        // polls at step-boundary safe points and waits idle on a cv while paused. Injection through
        // sx_thermal_set_state is the first-class source of state (the app observer) — the macOS
        // CLI has no injection source, so it stays fixed at nominal (no intervention). The hidden
        // test path (for driving CLI E2E without inducing real device heat (serious/critical) —
        // referenced from the thermal_mode comment in splatx_train_c.h):
        //   SPLATX_THERMAL_INJECT="500:critical,3000:serious" — an injection schedule at ms
        //     offsets from the start of training
        //   SPLATX_THERMAL_RESUME_MS / SPLATX_THERMAL_ABORT_MS / SPLATX_THERMAL_IDLE_MS
        //     — shorten the policy times (defaults 5000/10000/100)
        splatx::ThermalOptions topt;
        topt.mode = static_cast<splatx::ThermalMode>(thermal_mode);
        const auto env_ms = [](const char* name, int32_t fallback) -> int32_t {
            const char* v = std::getenv(name);
            return (v && *v) ? static_cast<int32_t>(std::atoi(v)) : fallback;
        };
        topt.resume_hold_ms = env_ms("SPLATX_THERMAL_RESUME_MS", topt.resume_hold_ms);
        topt.critical_abort_ms = env_ms("SPLATX_THERMAL_ABORT_MS", topt.critical_abort_ms);
        topt.lowheat_idle_ms = env_ms("SPLATX_THERMAL_IDLE_MS", topt.lowheat_idle_ms);
        splatx::ThermalGovernor governor(topt);
        // Declaration order = the reverse-destruction contract: ctl_reg (deregisters the control
        // global — waits for in-flight external access) → injector (joins the injection thread) →
        // thermal_reg (deregisters the global) → governor → trainer. The registration is kept
        // alive for as long as the injection thread lives, and the trainer and governor used by
        // external control (stop/pause) stay valid until ctl_reg deregisters.
        ThermalGovernorRegistration thermal_reg(governor);
        ThermalInjector thermal_injector;
        if (const char* inject = std::getenv("SPLATX_THERMAL_INJECT"); inject && *inject)
            thermal_injector.start(parse_thermal_inject(inject));
        TrainControl ctl(governor, trainer, factor, test_split);
        TrainControlRegistration ctl_reg(ctl);
        std::printf("thermal: mode %s (resume hold %d ms, critical abort %d ms, idle %d ms)\n",
                    thermal_mode == 0 ? "performance" : thermal_mode == 1 ? "balanced"
                                                                          : "lowheat",
                    topt.resume_hold_ms, topt.critical_abort_ms, topt.lowheat_idle_ms);
        std::fflush(stdout);

        // The training loop — walks the train indices (iter % n_train, deterministic — the test
        // views are untouched). GT goes through the cache (one decode per view — Task 5; only
        // scenes that exceed the limit decode every iter).
        // On resume it re-enters from start_iter — the view selection (it % n_train), the schedules
        // (ratios), and the noise (Philox offset=it) are all iter-based, so it rides the same
        // stream as a continuous run.
        const auto t_train = Clock::now();
        float first_loss = 0.0f, last_loss = 0.0f;
        int last_eval_iter = -1;
        double last_eval_psnr = 0.0;
        double last_eval_cc = 0.0; // travels alongside raw — exposed only in the final stats and
                                   // log (not in the progress tick)
        // The SSIM pair (perceptual-metrics Task 2) — the same carrying convention as cc. run_eval
        // delivers NaN when there are zero valid views, so the initial value is NaN as well (0.0
        // would be indistinguishable from "a measured 0").
        double last_eval_ssim = std::numeric_limits<double>::quiet_NaN();
        double last_eval_cc_ssim = std::numeric_limits<double>::quiet_NaN();
        double last_eval_depth_rmse = std::numeric_limits<double>::quiet_NaN();
        bool aborted_thermal = false;
        bool stopped_external = false;
        ctl.loop_active.store(true, std::memory_order_release);
        int it = start_iter;
        for (; it < run_until; it++) {
            // The §7.4 step-boundary poll: on a pause (thermal or an external hold) it waits idle
            // inside this call until resumption (the Trainer is untouched — this is the safe point
            // for an external thread's checkpoint snapshot and drain, consumed by
            // sx_train_pause_checkpoint. The completed update happens before the poll, so it stays
            // consistent with the snapshot iter taken while parked). AbortCritical = critical
            // sustained — a partial export (graceful). ExternalStop = sx_train_request_stop — the
            // same wrap-up, then return 3.
            ctl.completed.store(it, std::memory_order_release);
            const splatx::ThermalDecision d = governor.before_step();
            if (d == splatx::ThermalDecision::AbortCritical) {
                aborted_thermal = true;
                break;
            }
            if (d == splatx::ThermalDecision::ExternalStop) {
                stopped_external = true;
                break;
            }
            const int64_t cam_idx = split.train[static_cast<size_t>(it) % split.train.size()];
            const splatx::TrainingCamera& cam = ti.cameras[static_cast<size_t>(cam_idx)];
            const splatx::ImageF32& gt = load_gt(static_cast<size_t>(cam_idx));
            const splatx::DepthTarget* depth =
                depth_weight > 0.0f ? load_depth(static_cast<size_t>(cam_idx)) : nullptr;

            const auto t_it = Clock::now();
            const splatx::TrainStepResult r = trainer.step(cam, gt.rgb.data(), it, depth);
            const double step_ms = ms_since(t_it);

            if (it == start_iter) first_loss = r.loss;
            last_loss = r.loss;
            // A refine step is the record of the growth trajectory (N) and of relocation
            // observations — logged regardless of log_every
            if (r.mcmc_refined) {
                std::printf("mcmc refine iter %5d: relocated %lld, added %lld -> N %lld\n", it,
                            static_cast<long long>(r.n_relocated),
                            static_cast<long long>(r.n_added),
                            static_cast<long long>(trainer.splat_count()));
                std::fflush(stdout);
            }
            if (it % log_every == 0 || it == run_until - 1) {
                // splat_count(): a lightweight observation with no download — cloud() triggers a
                // full parameter synchronization on the resident path, so it is not used in logs
                // (Task 5).
                std::printf("iter %5d/%d  loss %.6f  step %.1f ms  n_isects %lld  N %lld\n", it,
                            run_until, static_cast<double>(r.loss), step_ms,
                            static_cast<long long>(r.n_isects),
                            static_cast<long long>(trainer.splat_count()));
                std::fflush(stdout);
            }
            bool eval_this_tick = false;
            if (eval_every > 0 && (it + 1) % eval_every == 0) {
                last_eval_psnr = run_eval(it + 1, last_eval_cc, last_eval_ssim,
                                          last_eval_cc_ssim, last_eval_depth_rmse);
                last_eval_iter = it + 1;
                eval_this_tick = true;
            }
            // Periodic save (at a step boundary) — the last iter is covered by the final save
            // outside the loop (this avoids a duplicate)
            if (checkpoint_every > 0 && (it + 1) % checkpoint_every == 0 && it + 1 < run_until)
                save_ck(it + 1);
            // live PLY export (app round 2 Task 1) + the in-memory snapshot callback (app round 3
            // Task 1) — at a step boundary, in the same phase as checkpoint_every. When both fire
            // on the same tick they share a single filter_cloud_for_export copy (sharing the
            // host_fresh_ download of cloud() is automatic — here even the filter copy cost is
            // reduced to one). A failure is non-fatal on both sides — it warns on stderr and
            // training continues (the authoritative contract is the comment in splatx_train_c.h).
            // On the resident path this is where all parameters are downloaded (cloud() — a read,
            // Home is kept).
            const bool live_tick = live_on && (it + 1) % live_ply_every == 0;
            const bool snap_tick = snapshot_on && (it + 1) % snapshot_every == 0;
            if (live_tick || snap_tick) {
                try {
                    const splatx::SplatCloud filtered =
                        splatx::filter_cloud_for_export(trainer.cloud(), 0.005f);
                    // live PLY: written to "<path>.tmp" and then swapped in atomically with
                    // rename — the reader (the app viewer) never sees a partial file. A separate
                    // try so that a PLY failure does not swallow the snapshot as well.
                    if (live_tick) {
                        const std::string tmp = live_ply_path + ".tmp";
                        try {
                            splatx::write_inria_ply(filtered, tmp);
                            if (std::rename(tmp.c_str(), live_ply_path.c_str()) != 0)
                                throw std::runtime_error("rename to " + live_ply_path +
                                                         " failed");
                        } catch (const std::exception& e) {
                            std::remove(tmp.c_str()); // clean up the .tmp leftover (no partial file is exposed)
                            std::fprintf(stderr,
                                         "warning: live PLY export failed (iter %d): %s — "
                                         "training continues\n",
                                         it + 1, e.what());
                        }
                    }
                    // Snapshot: passes the raw-domain pointers of the filtered copy as they are —
                    // their lifetime lasts only for this callback invocation (filtered is released
                    // at the end of this block. The contract for the synchronous call on the
                    // training-loop thread and the prohibition on re-entry is splatx_train_c.h
                    // SxSnapshotFn).
                    if (snap_tick) {
                        SxSplatSnapshot s{};
                        s.iter = it + 1;
                        s.sh_degree = static_cast<int32_t>(filtered.sh_degree);
                        s.count = filtered.count;
                        s.means = filtered.means.data();
                        s.quats = filtered.quats.data();
                        s.scales = filtered.scales.data();
                        s.opacities = filtered.opacities.data();
                        s.sh0 = filtered.sh0.data();
                        s.shN = filtered.sh_degree > 0 ? filtered.shN.data() : nullptr;
                        args->snapshot_cb(&s, args->snapshot_user);
                    }
                } catch (const std::exception& e) {
                    std::fprintf(stderr,
                                 "warning: live snapshot/export failed (iter %d): %s — "
                                 "training continues\n",
                                 it + 1, e.what());
                }
            }
            // The progress callback (Task 5 — a first-class stream instead of parsing stdout):
            // called at the step boundary after eval and the periodic save have finished — so it
            // can carry this tick's PSNR. Called synchronously on the training-loop thread (the
            // splatx_train_c.h SxTrainProgressFn contract).
            if (args->progress) {
                SxTrainProgress p{};
                p.iter = it + 1;
                p.run_until = run_until;
                p.total_iters = iters;
                p.loss = r.loss;
                p.psnr = eval_this_tick ? static_cast<float>(last_eval_psnr) : NAN;
                p.step_ms = step_ms;
                p.n_isects = r.n_isects;
                p.splat_count = trainer.splat_count();
                p.thermal_state = static_cast<int32_t>(governor.thermal_state());
                p.depth_rmse = eval_this_tick ? static_cast<float>(last_eval_depth_rmse) : NAN;
                args->progress(&p, args->progress_user);
            }
        }
        ctl.loop_active.store(false, std::memory_order_release);
        // completed = the completed step count: run_until on normal termination, or it at the point
        // of the break on a thermal abort/stop (the exit happens before the step runs, so 0..it-1
        // are complete — consistent with the checkpoint and resume iter).
        const int completed = it;
        const int ran_iters = completed - start_iter;
        const double train_s = ms_since(t_train) / 1000.0;
        if (aborted_thermal) {
            std::printf("thermal abort: critical sustained for %d ms — training aborted "
                        "(iter %d/%d complete, %.1f s). Partial export%s then graceful exit "
                        "(return 2; eval/preview skipped — avoids sustained load)\n",
                        topt.critical_abort_ms, completed, run_until, train_s,
                        save_checkpoint_path ? "+checkpoint" : "");
        } else if (stopped_external) {
            std::printf("stop: external stop requested — training aborted (iter %d/%d complete, "
                        "%.1f s). Partial export%s then graceful exit (return 3; eval/preview "
                        "skipped)\n",
                        completed, run_until, train_s,
                        save_checkpoint_path ? "+checkpoint" : "");
        } else {
            std::printf("train done: %d iters in %.1f s (%.0f ms/iter), loss %.6f -> %.6f\n",
                        ran_iters, train_s,
                        ran_iters > 0 ? train_s * 1000.0 / ran_iters : 0.0,
                        static_cast<double>(first_loss), static_cast<double>(last_loss));
        }
        std::fflush(stdout);

        // The final save — regardless of checkpoint_every, if a path is given the final state is
        // written once (the file always holds the last step-boundary state — temp+rename means even
        // a previous save is swapped atomically. A thermal abort or an external stop is saved
        // too — this is the path for resuming with --resume after cooling down or returning).
        if (save_checkpoint_path) save_ck(completed);

        // The final eval (always once when split is on — the re-evaluation is skipped if the last
        // iter already evaluated. Skipped on a thermal abort or an external stop — rendering every
        // test view is sustained GPU load and delay.)
        // raw and cc are a pair from the same run_eval call — the reuse branch carries them over as
        // a pair too (color-correction Task 2).
        // SSIM and ccSSIM are carried over as a quartet from the same call (perceptual-metrics
        // Task 2 — all NaN if it was not performed).
        double final_psnr = std::numeric_limits<double>::quiet_NaN();
        double final_cc = std::numeric_limits<double>::quiet_NaN();
        double final_ssim = std::numeric_limits<double>::quiet_NaN();
        double final_cc_ssim = std::numeric_limits<double>::quiet_NaN();
        double final_depth_rmse = std::numeric_limits<double>::quiet_NaN();
        if (test_split > 0 && !aborted_thermal && !stopped_external) {
            if (last_eval_iter == completed) {
                final_psnr = last_eval_psnr;
                final_cc = last_eval_cc;
                final_ssim = last_eval_ssim;
                final_cc_ssim = last_eval_cc_ssim;
                final_depth_rmse = last_eval_depth_rmse;
            } else {
                final_psnr = run_eval(completed, final_cc, final_ssim, final_cc_ssim,
                                      final_depth_rmse);
            }
            std::printf(
                "final eval PSNR: %.3f dB | cc %.3f dB | SSIM %.4f | ccSSIM %.4f (%zu test views)\n",
                final_psnr, final_cc, final_ssim, final_cc_ssim, split.test.size());
            std::fflush(stdout);
        }

        // === train-view reconstruction PSNR (a reconstruction quality metric) ===
        // Separately from the test PSNR (the holdout), the reconstruction PSNR of the train views
        // used for training is reported alongside — the basis for judging the reconstruction effect
        // of the input poses and the data quality (read-only — the training bits are unchanged).
        if (!aborted_thermal && !stopped_external && !split.train.empty()) {
            // train-view reconstruction PSNR (subsample ≤48 — GT comes from the training cache).
            const size_t cap = 48;
            const size_t n = split.train.size();
            const size_t stride = std::max<size_t>(1, (n + cap - 1) / cap);
            double sum = 0.0;
            size_t cnt = 0;
            for (size_t i = 0; i < n; i += stride) {
                const int64_t idx = split.train[i];
                const splatx::TrainingCamera& rcam = ti.cameras[static_cast<size_t>(idx)];
                const splatx::ImageF32& gt = load_gt(static_cast<size_t>(idx));
                std::vector<float> render = trainer.render(rcam);
                for (float& v : render) v = std::clamp(v, 0.0f, 1.0f);
                sum += splatx::psnr(render.data(), gt.rgb.data(),
                                    static_cast<int64_t>(render.size()));
                cnt++;
            }
            std::printf("final train-view PSNR: %.3f dB (%zu/%zu train views)\n",
                        sum / static_cast<double>(cnt), cnt, n);
            std::fflush(stdout);
        }

        // Saved through the export filter (design §5.4 — the PlyIO.h convention: remove NaN/Inf
        // rows and sigmoid(op) <= 0.005; there is no scale clamp, following the gsplat standard ply
        // convention. The arena tail is already excluded by count.)
        // A .spz extension on --out means SPZ v3 (Niantic spz), anything else INRIA PLY (SpzIO.h —
        // on the SPZ path save_spz applies the same filter internally: quantized packing makes
        // NaN/Inf UB — the SpzIO.h contract).
        // For PLY, the default zero-padding of write_inria_ply gives 45 f_rest entries even for
        // sh_degree<3 — compatible with viewers (SplatIO and others require exactly 45) (design
        // §8.1, PlyIO.h). SPZ states shDegree explicitly in its header, so no padding is needed —
        // the actual degree is written as is.
        const bool spz_out = wants_spz(out_ply);
        splatx::ExportFilterStats fst;
        if (spz_out) {
            splatx::SpzSaveOptions so;
            so.antialiased = cfg.antialiased; // the SPZ header antialiased flag (information PLY does not carry)
            so.filter_stats = &fst;
            splatx::save_spz(trainer.cloud(), out_ply, so);
        } else {
            fst = export_filtered_ply(trainer, out_ply); // shared with the live export (app round 2 Task 1)
        }
        if (fst.kept == 0)
            std::printf("warning: export filter removed all splats — wrote empty %s\n",
                        spz_out ? "SPZ" : "PLY");
        std::printf("export filter: %lld -> %lld (nonfinite %lld, low-opacity %lld)\n",
                    static_cast<long long>(fst.input), static_cast<long long>(fst.kept),
                    static_cast<long long>(fst.removed_nonfinite),
                    static_cast<long long>(fst.removed_low_opacity));
        std::printf("%s saved: %s (N=%lld)\n", spz_out ? "SPZ" : "PLY", out_ply,
                    static_cast<long long>(fst.kept));

        // Skipped on abort/stop — rendering is GPU load and delay too
        if (preview_png && !aborted_thermal && !stopped_external) {
            const size_t idx = static_cast<size_t>(
                std::clamp<int64_t>(eval_image, 0,
                                    static_cast<int64_t>(ti.cameras.size()) - 1));
            const splatx::TrainingCamera& cam = ti.cameras[idx];
            splatx::ImageF32 img;
            img.width = cam.width;
            img.height = cam.height;
            img.rgb = trainer.render(cam);
            splatx::save_png_rgb_f32(img, preview_png);
            std::printf("preview saved: %s (camera %zu, %dx%d)\n", preview_png, idx,
                        cam.width, cam.height);
        }
        std::fflush(stdout);

        // The final summary (Task 5 — valid for return values 0/2/3, the splatx_train_c.h SxTrainStats contract)
        if (args->stats) {
            SxTrainStats st{};
            st.completed_iters = completed;
            st.run_until = run_until;
            st.total_iters = iters;
            st.exported_splats = fst.kept;
            st.final_eval_psnr = static_cast<float>(final_psnr); // NaN if not performed
            st.final_cc_psnr = static_cast<float>(final_cc);     // NaN if not performed (same convention)
            st.final_ssim = static_cast<float>(final_ssim);      // NaN if not performed / zero valid views
            st.final_cc_ssim = static_cast<float>(final_cc_ssim); // same convention (perceptual-metrics Task 2)
            st.final_depth_rmse = static_cast<float>(final_depth_rmse); // NaN without sidecars/eval
            st.train_seconds = train_s;
            st.checkpoint_saved = save_checkpoint_path ? 1 : 0;
            *args->stats = st;
        }
        // 2 = thermal graceful abort, 3 = external stop, graceful halt (header contract — neither is an error)
        return aborted_thermal ? 2 : (stopped_external ? 3 : 0);
    } catch (const std::exception& e) {
        fill_err(err_buf, err_buf_len, e.what());
        return 1;
    } catch (...) {
        fill_err(err_buf, err_buf_len, "unknown error");
        return 1;
    }
}

// The 25-argument compatibility wrapper (Task 5 — extension fields exist only in the struct form:
// existing call sites and ABI consumers are unchanged)
extern "C" int32_t sx_train_run(const char* data_dir, int32_t factor, int32_t iters,
                                int64_t max_points, int32_t sh_degree, const char* out_ply,
                                const char* preview_png, int32_t eval_image,
                                const char* kernel_src_dir, const char* metallib_path,
                                int32_t log_every, int32_t mcmc, int64_t cap_max,
                                int32_t test_split, int32_t eval_every, float ssim_weight,
                                int32_t antialiased, int32_t resident,
                                const char* resume_checkpoint,
                                const char* save_checkpoint_path, int32_t checkpoint_every,
                                int32_t auto_cap, int64_t memory_budget_bytes,
                                int32_t thermal_mode, char* err_buf, int32_t err_buf_len) {
    SxTrainArgs a{};
    a.data_dir = data_dir;
    a.factor = factor;
    a.iters = iters;
    a.max_points = max_points;
    a.sh_degree = sh_degree;
    a.out_path = out_ply;
    a.preview_png = preview_png;
    a.eval_image = eval_image;
    a.kernel_src_dir = kernel_src_dir;
    a.metallib_path = metallib_path;
    a.log_every = log_every;
    a.mcmc = mcmc;
    a.cap_max = cap_max;
    a.test_split = test_split;
    a.eval_every = eval_every;
    a.ssim_weight = ssim_weight;
    a.antialiased = antialiased;
    a.resident = resident;
    a.resume_checkpoint = resume_checkpoint;
    a.save_checkpoint_path = save_checkpoint_path;
    a.checkpoint_every = checkpoint_every;
    a.auto_cap = auto_cap;
    a.memory_budget_bytes = memory_budget_bytes;
    a.thermal_mode = thermal_mode;
    return sx_train_run_args(&a, err_buf, err_buf_len);
}

// ===== The training control surface (Task 5 — the contract is splatx_train_c.h) =====

extern "C" void sx_train_request_stop(void) {
    // Non-blocking (header contract — Task 5 review 2): even when a pause save holds g_ctl_mu for
    // several seconds, the cancellation handler is not held. Record the flag first → deliver
    // immediately if the try-lock succeeds; on failure the mutex holder delivers at the exit of the
    // critical section (deliver_pending_stop_locked).
    g_stop_pending.store(true, std::memory_order_release);
    if (g_ctl_mu.try_lock()) {
        deliver_pending_stop_locked(); // discarded when there is no active run (no carry-over)
        g_ctl_mu.unlock();
    }
}

extern "C" int32_t sx_train_pause_checkpoint(const char* path, int32_t fp16,
                                             int32_t timeout_ms, char* err_buf,
                                             int32_t err_buf_len) {
    // Proceeds while holding g_ctl_mu — deregistration (run teardown) blocks for this duration, so
    // the trainer and governor lifetimes are guaranteed (the concurrency argument in the
    // TrainControl comment). A stop request arriving in the meantime is held in the flag, and every
    // return path delivers it just before lk is released (StopDeliverer — the stop is delivered
    // after the save completes, which preserves the checkpoint).
    std::lock_guard<std::mutex> lk(g_ctl_mu);
    struct StopDeliverer { // declared after lk = destroyed first — delivery happens while the mutex is held
        ~StopDeliverer() { deliver_pending_stop_locked(); }
    } stop_deliverer;
    if (!g_ctl || !g_ctl->loop_active.load(std::memory_order_acquire)) {
        fill_err(err_buf, err_buf_len,
                 "pause: no active training loop (in the load/init phase or no run — there is "
                 "no training state to save yet/anymore)");
        return -1;
    }
    if (!path) {
        fill_err(err_buf, err_buf_len, "pause: checkpoint path must be given");
        return 1;
    }
    splatx::ThermalGovernor* gov = g_ctl->gov;
    gov->set_external_hold(true); // ① stop submitting — an absolute latch a thermal transition cannot release
    // ② Wait to park at a step boundary (sliced polling — if the loop finishes its last step and
    //    exits, no park ever arrives, so it is detected through loop_active. wait_paused is
    //    cv-based — not a busy-wait).
    const auto t0 = Clock::now();
    bool parked = false;
    bool loop_ended = false;
    while (ms_since(t0) < static_cast<double>(timeout_ms)) {
        if (!g_ctl->loop_active.load(std::memory_order_acquire)) { // the loop has ended — park is impossible
            loop_ended = true;
            break;
        }
        if (gov->wait_paused(20)) {
            parked = true;
            break;
        }
    }
    if (!parked) {
        gov->set_external_hold(false);
        // Separate rc values (Task 5 review 1): -1 (no loop — safe to ignore) vs. -2 (timeout —
        // the loop is still running while no checkpoint was saved: the caller must retry or
        // report, header contract).
        if (loop_ended) {
            fill_err(err_buf, err_buf_len,
                     "pause: training loop ended while waiting (the run is finishing — no "
                     "training state to save; the exit path's final save takes over)");
            return -1;
        }
        fill_err(err_buf, err_buf_len,
                 "pause: training loop did not park within timeout (a long step/eval in "
                 "progress — hold released, checkpoint not saved: retry or report needed)");
        return -2;
    }
    // ③ The safe point — the loop thread is parked on the hold (the Trainer is untouched, and the
    //    GPU drain is guaranteed by the waitUntilCompleted sync point inside the step — the
    //    Trainer.h make_checkpoint contract). The snapshot and the save happen on the external
    //    (calling) thread. Because the hold is an absolute latch there is no resume race during
    //    the save (even the thermal critical abort decision is deferred while the hold is
    //    active — ThermalGovernor.h).
    try {
        const auto t_ck = Clock::now();
        const splatx::TrainerCheckpoint ck = g_ctl->trainer->make_checkpoint(
            g_ctl->completed.load(std::memory_order_acquire), g_ctl->factor,
            g_ctl->test_split);
        splatx::save_checkpoint_file(ck, path, fp16 != 0);
        std::printf("pause checkpoint saved: %s (iter %d, N %lld, %s, %.0f ms — hold kept, "
                    "resume via sx_train_resume_after_pause)\n",
                    path, ck.iter, static_cast<long long>(ck.cloud.count),
                    fp16 ? "fp16" : "fp32", ms_since(t_ck));
        std::fflush(stdout);
        return 0; // the hold is kept — release it on return (§7.5 contract: no background GPU submission)
    } catch (const std::exception& e) {
        gov->set_external_hold(false);
        fill_err(err_buf, err_buf_len, e.what());
        return 1;
    } catch (...) {
        gov->set_external_hold(false);
        fill_err(err_buf, err_buf_len, "unknown error");
        return 1;
    }
}

extern "C" void sx_train_resume_after_pause(void) {
    std::lock_guard<std::mutex> lk(g_ctl_mu);
    deliver_pending_stop_locked(); // deliver the pending stop (the critical-section rule — see
                                   // the globals comment above)
    if (g_ctl) g_ctl->gov->set_external_hold(false);
}

extern "C" int32_t sx_train_paused(void) {
    std::lock_guard<std::mutex> lk(g_ctl_mu);
    deliver_pending_stop_locked(); // the critical-section rule: every g_ctl_mu holder delivers or
                                   // discards the pending stop at the exit — this closes the
                                   // try-lock failure window
    if (!g_ctl) return -1;
    return g_ctl->gov->paused() ? 1 : 0;
}

extern "C" int32_t sx_thermal_get_state(void) {
    std::lock_guard<std::mutex> lk(g_thermal_mu);
    return g_thermal_last_state;
}

extern "C" int32_t sx_checkpoint_info(const char* path, SxCheckpointInfo* out, char* err_buf,
                                      int32_t err_buf_len) {
    try {
        if (!path || !out) throw std::invalid_argument("path/out must be given");
        const splatx::CheckpointHeader h = splatx::load_checkpoint_header(path);
        SxCheckpointInfo info{};
        info.iter = h.iter;
        info.total_iters = h.config.total_iters;
        info.sh_degree = h.config.sh_degree;
        info.factor = h.config.factor;
        info.test_split = h.config.test_split;
        info.mcmc_enabled = h.config.mcmc_enabled ? 1 : 0;
        info.antialiased = h.config.antialiased ? 1 : 0;
        info.cap_max = h.config.cap_max;
        info.mcmc_seed = h.config.mcmc_seed;
        info.scene_scale = h.config.scene_scale;
        info.ssim_weight = h.config.ssim_weight;
        info.alive_count = h.alive_count;
        info.fp16 = h.fp16 ? 1 : 0;
        *out = info;
        return 0;
    } catch (const std::exception& e) {
        fill_err(err_buf, err_buf_len, e.what());
        return 1;
    } catch (...) {
        fill_err(err_buf, err_buf_len, "unknown error");
        return 1;
    }
}

// The C surface for automatic cap_max estimation and the factor policy (design §7.2/§7.3 —
// consumed by the iOS capture app's TrainerBackend). These are pure functions, so there is no
// err_buf and an argument violation returns -1 (header contract). The formula is delegated to the
// single source of truth in SplatXCore MemoryGovernor — the same constants as the sx_train_run
// auto_cap path.
extern "C" int64_t sx_suggest_cap_max(int64_t budget_bytes, int32_t image_w, int32_t image_h,
                                      int32_t n_gt_images, int32_t sh_degree,
                                      int32_t gt_cache_on,
                                      int32_t* out_budget_insufficient) {
    try {
        const int64_t budget =
            budget_bytes > 0 ? budget_bytes : splatx::default_memory_budget_bytes();
        const splatx::CapMaxSuggestion s = splatx::suggest_cap_max(
            budget, image_w, image_h, n_gt_images, sh_degree, gt_cache_on != 0);
        // Carry-over ④: a lower-bound clamp signals an insufficient budget (raw < 100k) — the
        // input for the caller's UI warning or refusal decision
        if (out_budget_insufficient) *out_budget_insufficient = s.clamped_min ? 1 : 0;
        return s.cap_max;
    } catch (...) {
        if (out_budget_insufficient) *out_budget_insufficient = 0;
        return -1;
    }
}

extern "C" int32_t sx_suggest_factor(int32_t image_w, int32_t image_h, int32_t ios) {
    try {
        return splatx::suggest_factor(image_w, image_h,
                                      ios ? splatx::ResolutionPlatform::iOS
                                          : splatx::ResolutionPlatform::macOS);
    } catch (...) {
        return -1;
    }
}

// The GT cache limit (training-speed Task 2). A pure function — the header comment is the
// authoritative contract for the policy and its rationale.
// The division precedes the comparison, so there is no overflow or wrap even at the maximum value
// of physical_ram_bytes, and 0 (an anomalous value) is absorbed by the max and resolves to the
// 4 GiB floor.
extern "C" uint64_t sx_gt_cache_limit_bytes(uint64_t physical_ram_bytes, int32_t ios) {
    const uint64_t floor_bytes = uint64_t{4} << 30; // the previous fixed 4 GiB limit, kept as the floor
    if (ios) return floor_bytes;                    // conservative for jetsam — the previous behavior as is
    const uint64_t half_ram = physical_ram_bytes / 2;
    return half_ram > floor_bytes ? half_ram : floor_bytes;
}

extern "C" int32_t sx_convert_run(const char* in_path, const char* out_path, char* err_buf,
                                  int32_t err_buf_len) {
    try {
        if (!in_path || !out_path) throw std::invalid_argument("in/out path must be given");
        const std::string in(in_path), out(out_path);
        const bool in_spz = wants_spz(in), out_spz = wants_spz(out);

        bool aa = false; // the antialiased header flag of an SPZ input (a PLY input carries no
                         // such information — false)
        const splatx::SplatCloud cloud =
            in_spz ? splatx::load_spz(in, &aa) : splatx::read_inria_ply(in);
        std::printf("convert: loaded %s (%s, N=%lld, sh_degree %d%s)\n", in.c_str(),
                    in_spz ? "SPZ" : "PLY", static_cast<long long>(cloud.count), cloud.sh_degree,
                    in_spz && aa ? ", antialiased" : "");
        std::fflush(stdout);

        splatx::ExportFilterStats fst;
        if (out_spz) {
            // The filter built into the save_spz contract (SpzIO.h — NaN/Inf rows are UB for
            // quantized packing). SPZ→SPZ preserves the input antialiased flag; a PLY input gives
            // false (the filter/flag policy in the header comment).
            splatx::SpzSaveOptions so;
            so.antialiased = aa;
            so.filter_stats = &fst;
            splatx::save_spz(cloud, out, so);
            if (fst.kept == 0)
                std::printf("warning: export filter removed all splats — wrote empty SPZ\n");
            std::printf("export filter: %lld -> %lld (nonfinite %lld, low-opacity %lld)\n",
                        static_cast<long long>(fst.input), static_cast<long long>(fst.kept),
                        static_cast<long long>(fst.removed_nonfinite),
                        static_cast<long long>(fst.removed_low_opacity));
            std::printf("SPZ saved: %s (N=%lld%s)\n", out.c_str(),
                        static_cast<long long>(fst.kept), aa ? ", antialiased" : "");
        } else {
            // No re-filtering — this is a format conversion (header comment: for an SPZ input
            // load_spz guarantees finiteness, and a PLY input preserves the original rows; this
            // prevents rows near the threshold from being lost to a double filter).
            // Default zero-padding — even an input with sh_degree<3 (SPZ deg1/2 and the like) is
            // written with 45 f_rest entries for viewer-compatible output (design §8.1, PlyIO.h).
            splatx::write_inria_ply(cloud, out);
            if (in_spz && aa)
                std::printf("warning: antialiased SPZ -> PLY — standard PLY has no AA flag, "
                            "so it is lost (viewers render without compensation)\n");
            std::printf("PLY saved: %s (N=%lld)\n", out.c_str(),
                        static_cast<long long>(cloud.count));
        }
        std::fflush(stdout);
        return 0;
    } catch (const std::exception& e) {
        fill_err(err_buf, err_buf_len, e.what());
        return 1;
    } catch (...) {
        fill_err(err_buf, err_buf_len, "unknown error");
        return 1;
    }
}

// Forward-renders a trained PLY from a COLMAP camera → PNG. No training — it injects the SplatCloud
// from read_inria_ply straight into the Trainer(cloud) constructor and calls Trainer::render (the
// same path as --save-preview and eval) once. With mcmc off there is no cap_max arena reshape, so
// the cloud is rendered as is.
// The factor scaling and the sort by image name follow the same conventions as the training path
// (sx_train_run_args) — the point is to render different PLYs (an iOS output vs. a macOS output,
// for example) with the same cameras and resolution for visual and PSNR comparison. Returns 0 on
// success.
extern "C" int32_t sx_render_ply(const char* ply_path, const char* data_dir, int32_t factor,
                                 int32_t cam_index, const char* metallib_path,
                                 const char* out_png, char* err_buf, int32_t err_buf_len) {
    try {
        if (!ply_path || !data_dir || !out_png)
            throw std::invalid_argument("ply_path/data_dir/out_png must be given");
        if (factor != 1 && factor != 2 && factor != 4 && factor != 8)
            throw std::invalid_argument("factor must be 1|2|4|8");

        splatx::SplatCloud cloud = splatx::read_inria_ply(ply_path);
        std::printf("render: loaded PLY %s (N=%lld, sh_degree %d)\n", ply_path,
                    static_cast<long long>(cloud.count), cloud.sh_degree);
        std::fflush(stdout);

        // Loads the COLMAP cameras, sorts by image name, and applies the factor scaling (the same
        // convention as sx_train_run_args)
        const std::string data(data_dir);
        const splatx::ImagesDirChoice img_choice = splatx::resolve_images_dir(data, factor);
        const splatx::ColmapBundle bundle = splatx::load_colmap_bundle(data + "/sparse/0");
        splatx::TrainingInput ti = splatx::training_input_from_colmap(bundle, img_choice.dir);
        if (ti.cameras.empty()) throw std::runtime_error("no cameras in COLMAP bundle");
        std::sort(ti.cameras.begin(), ti.cameras.end(),
                  [](const splatx::TrainingCamera& a, const splatx::TrainingCamera& b) {
                      return a.image_path < b.image_path;
                  });
        if (factor > 1) {
            for (auto& cam : ti.cameras) {
                const int w0 = cam.width, h0 = cam.height;
                const int w1 = splatx::downscaled_dim(w0, factor);
                const int h1 = splatx::downscaled_dim(h0, factor);
                if (w1 < 1 || h1 < 1) throw std::runtime_error("factor too large for image dims");
                const double sxr = static_cast<double>(w1) / w0, syr = static_cast<double>(h1) / h0;
                cam.K[0] = static_cast<float>(cam.K[0] * sxr); // fx
                cam.K[2] = static_cast<float>(cam.K[2] * sxr); // cx
                cam.K[4] = static_cast<float>(cam.K[4] * syr); // fy
                cam.K[5] = static_cast<float>(cam.K[5] * syr); // cy
                cam.width = w1;
                cam.height = h1;
            }
        }
        const size_t idx = static_cast<size_t>(std::clamp<int64_t>(
            cam_index, 0, static_cast<int64_t>(ti.cameras.size()) - 1));

        // A render-only config — mcmc off (no arena reshape). The SH degree is held by the
        // SplatCloud (Trainer::render renders at cloud.sh_degree — TrainConfig has no max degree
        // field).
        const int sh_deg = cloud.sh_degree; // captured before std::move (for the log)
        splatx::TrainConfig cfg;

        splatx::MetalBackendDesc bdesc;
        if (metallib_path) bdesc.metallib_path = metallib_path;
        const std::unique_ptr<splatx::KernelBackend> backend =
            splatx::make_metal_backend(nullptr, bdesc);
        splatx::Trainer trainer(*backend, std::move(cloud), cfg);

        splatx::TrainingCamera cam = ti.cameras[idx];
        splatx::ImageF32 img;
        img.width = cam.width;
        img.height = cam.height;
        img.rgb = trainer.render(cam);
        splatx::save_png_rgb_f32(img, out_png);
        std::printf("render saved: %s (camera %zu, %dx%d, sh_degree %d)\n", out_png, idx,
                    cam.width, cam.height, sh_deg);
        std::fflush(stdout);
        return 0;
    } catch (const std::exception& e) {
        fill_err(err_buf, err_buf_len, e.what());
        return 1;
    } catch (...) {
        fill_err(err_buf, err_buf_len, "unknown error");
        return 1;
    }
}

// ===== Cinema Task 2: the session-style render C API =====
// The authoritative contract is the splatx_train_c.h header comment.
// This decomposes the "load PLY → make_metal_backend → Trainer::render" assembly of sx_render_ply
// into a session form: open does the load and creates the backend and the Trainer (once), frame
// renders an arbitrary camera (repeatedly), and close releases everything. **With AA off, the
// TrainConfig assembly in open MUST be identical to the default assembly of sx_render_ply** (the
// sx_render_ply PNG pixel comparison in RenderSessionTests gates that identity — change only one
// side and the test breaks). sx_render_ply itself is unchanged (the existing contract is preserved).

// The opaque handle definition — in the same global scope as the forward declaration in the header
// (an anonymous namespace is forbidden: type identity). backend is declared before trainer — with
// reverse-order destruction the Trainer releases its KernelBackend reference first (the Trainer
// holds backend_ by reference).
struct SxRenderSession {
    std::unique_ptr<splatx::KernelBackend> backend;
    std::unique_ptr<splatx::Trainer> trainer;
};

extern "C" SxRenderSession* sx_render_session_open(const SxRenderSessionArgs* args,
                                                   char* err_buf, int32_t err_buf_len) {
    try {
        if (!args) throw std::invalid_argument("args must be given (the 0/NULL initialization convention)");
        if (!args->ply_path) throw std::invalid_argument("ply_path must be given");
        if (args->sh_degree < 0 || args->sh_degree > 3)
            throw std::invalid_argument("sh_degree must be 0..3 (0 = derived from the PLY)");

        splatx::SplatCloud cloud = splatx::read_inria_ply(args->ply_path);
        // No re-filtering — the same as sx_render_ply (the export filter was already applied to the
        // training output). An empty cloud is rejected with an explicit error by the Trainer
        // constructor below (count >= 1).

        // The sh_degree upper bound (header contract): exceeding the PLY degree is an error (the
        // higher-band coefficients are absent); below it, the higher bands of shN are truncated
        // session-locally (the file is unchanged). Trainer::render evaluates the full
        // cloud.sh_degree, so the truncation is exactly the render degree.
        if (args->sh_degree > 0 && args->sh_degree > cloud.sh_degree)
            throw std::invalid_argument(
                "sh_degree " + std::to_string(args->sh_degree) + " exceeds PLY sh_degree " +
                std::to_string(cloud.sh_degree) + " (higher-band coefficients absent)");
        if (args->sh_degree > 0 && args->sh_degree < cloud.sh_degree) {
            const int Kn = (args->sh_degree + 1) * (args->sh_degree + 1) - 1;
            const int Ko = cloud.sh_rest_coeffs();
            std::vector<float> shN(static_cast<size_t>(cloud.count) * Kn * 3);
            for (int64_t i = 0; i < cloud.count; i++)
                std::copy_n(cloud.shN.begin() + i * Ko * 3, static_cast<size_t>(Kn) * 3,
                            shN.begin() + i * Kn * 3);
            cloud.shN = std::move(shN);
            cloud.sh_degree = args->sh_degree;
        }
        const int64_t n = cloud.count;      // captured before std::move (for the log)
        const int sh_deg = cloud.sh_degree; // "

        // A render-only config — the same defaults as sx_render_ply, with only antialiased applied
        // (mcmc off: no cap_max arena reshape, so the cloud renders as is. resident defaults to
        // on — forward goes through the resident executor path, and the parameter upload happens
        // once on the first frame and is then kept).
        splatx::TrainConfig cfg;
        cfg.antialiased = args->antialiased != 0;

        // Layered loading — the same three layers as sx_train_run (an explicit metallib failure
        // throws immediately, with no fallback, the MetalBackend.h (a) contract. A NULL
        // kernel_src_dir falls back to the build-time macro).
        splatx::MetalBackendDesc bdesc;
        if (args->metallib_path) bdesc.metallib_path = args->metallib_path;
        bdesc.search_exe_dir = true;
        bdesc.kernel_src_dir =
            args->kernel_src_dir ? std::string(args->kernel_src_dir) : KERNEL_SRC_DIR;

        auto session = std::make_unique<SxRenderSession>();
        session->backend = splatx::make_metal_backend(nullptr, bdesc);
        session->trainer =
            std::make_unique<splatx::Trainer>(*session->backend, std::move(cloud), cfg);
        std::printf("render session open: %s (N=%lld, sh_degree %d, %s, backend %s)\n",
                    args->ply_path, static_cast<long long>(n), sh_deg,
                    cfg.antialiased ? "antialiased" : "classic", session->backend->name());
        std::fflush(stdout);
        return session.release();
    } catch (const std::exception& e) {
        fill_err(err_buf, err_buf_len, e.what());
        return nullptr;
    } catch (...) {
        fill_err(err_buf, err_buf_len, "unknown error");
        return nullptr;
    }
}

extern "C" int32_t sx_render_session_frame(SxRenderSession* session, const float* viewmat,
                                           float fx, float fy, float cx, float cy,
                                           int32_t width, int32_t height, uint8_t* out_rgba,
                                           char* err_buf, int32_t err_buf_len) {
    // An argument violation returns -1 (header contract). The contents of viewmat are not
    // validated — a NaN render is guarded by the NaN→0 handling in the conversion below.
    if (!session || !viewmat || !out_rgba) {
        fill_err(err_buf, err_buf_len, "session/viewmat/out_rgba must be given");
        return -1;
    }
    if (width < 1 || height < 1) {
        fill_err(err_buf, err_buf_len, "width/height must be positive");
        return -1;
    }
    if (!(fx > 0.0f) || !(fy > 0.0f) || !std::isfinite(cx) || !std::isfinite(cy)) {
        fill_err(err_buf, err_buf_len, "fx/fy must be positive and cx/cy finite");
        return -1;
    }
    try {
        splatx::TrainingCamera cam;
        std::memcpy(cam.viewmat, viewmat, 16 * sizeof(float));
        const float K[9] = {fx, 0.0f, cx, 0.0f, fy, cy, 0.0f, 0.0f, 1.0f};
        std::memcpy(cam.K, K, sizeof(K));
        cam.width = width;
        cam.height = height;
        const std::vector<float> rgb = session->trainer->render(cam);

        const size_t n_px = static_cast<size_t>(width) * static_cast<size_t>(height);
        if (rgb.size() != n_px * 3) {
            fill_err(err_buf, err_buf_len, "internal: render size mismatch");
            return -2;
        }
        // f32 [H,W,3] → RGBA8 — **the same quantization** as save_png_rgb_f32 (NaN→0, clamp to
        // [0,1], round(v·255)) + alpha 255. This is the basis for pixel consistency with the
        // sx_render_ply PNG (header contract — if the formula on the ImageIO.cpp side changes, this
        // one MUST be changed with it. A test gates this).
        for (size_t p = 0; p < n_px; p++) {
            for (int c = 0; c < 3; c++) {
                float v = rgb[p * 3 + c];
                if (!(v > 0.0f)) v = 0.0f; // NaN included
                if (v > 1.0f) v = 1.0f;
                out_rgba[p * 4 + static_cast<size_t>(c)] =
                    static_cast<uint8_t>(lrintf(v * 255.0f));
            }
            out_rgba[p * 4 + 3] = 255;
        }
        return 0;
    } catch (const std::exception& e) {
        fill_err(err_buf, err_buf_len, e.what());
        return -2;
    } catch (...) {
        fill_err(err_buf, err_buf_len, "unknown error");
        return -2;
    }
}

extern "C" void sx_render_session_close(SxRenderSession* session) {
    delete session; // NULL is ignored (delete nullptr is a no-op — header contract)
}
