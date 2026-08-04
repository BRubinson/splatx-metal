// Thermal governance tests (Phase 6 2nd round Task 4 — design §7.4 pause-and-cool; the
// authoritative policy contract is ThermalGovernor.h). Pure CPU and threads — no backend and no
// fixture required. The policy is verified through **injected state sequences** (without inducing
// serious on a real device — the "injectable by design" item in the plan's Global Constraints).
//
// Timing discipline: short ms parameters make the tests deterministic, but every assertion is a
// **lower bound or an invariant property** only —
//   · zero step progress while paused (the counter is unchanged over the observation window)
//   · resume and abort happen **only after** the hysteresis or sustained duration (a lower bound on
//     the elapsed time since injection)
//   · LowHeat idling makes N steps take >= N·idle
// Upper-bound assertions ("resumes within X ms") are not used, because they are fragile under CI
// scheduling noise — reaching a state is observed by polling with a generous timeout (wait_for).
// The only exception is the balanced-mode no-idle comparison (which has >100× of slack).
#include "TestHarness.h"
#include "TestPaths.h"
#include "splatx/PlyIO.h"
#include "splatx/ThermalGovernor.h"
#include "splatx/Trainer.h" // logit_f32 — for specifying post-sigmoid opacity (the internal representation is a logit)

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace splatx;

namespace {

using Clock = std::chrono::steady_clock;

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Polls until a condition is reached (with a generous timeout — this is not an upper-bound
// assertion; only failing to reach the condition fails the test)
template <typename Pred>
bool wait_for(Pred&& p, int timeout_ms) {
    const auto end = Clock::now() + std::chrono::milliseconds(timeout_ms);
    while (Clock::now() < end) {
        if (p()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return p();
}

// A worker that reproduces the structure of the TrainC training loop: at each step boundary it
// polls before_step → on Proceed it runs a step (incrementing the counter plus a 1 ms load), and on
// AbortCritical it exits. stop is for a normal shutdown (the tests stop the worker only while it is
// not paused — stopping during a pause would wait until the resume, so each test returns the state
// to something resumable before finishing).
struct StepWorker {
    ThermalGovernor& gov;
    std::atomic<int64_t> steps{0};
    std::atomic<bool> stop{false};
    std::atomic<bool> aborted{false};
    std::thread th;

    explicit StepWorker(ThermalGovernor& g) : gov(g) {}
    void run() {
        th = std::thread([this] {
            while (!stop.load()) {
                if (gov.before_step() == ThermalDecision::AbortCritical) {
                    aborted.store(true);
                    return;
                }
                steps.fetch_add(1);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }
    ~StepWorker() {
        stop.store(true);
        if (th.joinable()) th.join();
    }
};

// The input for the partial export — a small cloud (sh_degree 1). Opacity is specified as the
// post-sigmoid value: this also drives the case where one low-opacity row (0.002 <= 0.005) is caught
// by the export filter (the existing export convention).
SplatCloud make_export_cloud() {
    const std::vector<float> op_act = {0.9f, 0.8f, 0.002f};
    const int64_t n = static_cast<int64_t>(op_act.size());
    SplatCloud c;
    c.count = n;
    c.sh_degree = 1; // sh_rest_coeffs 3 → shN [N,3,3]
    c.means.resize(n * 3);
    c.scales.resize(n * 3);
    c.quats.assign(n * 4, 0.0f);
    c.opacities.resize(n);
    c.sh0.resize(n * 3);
    c.shN.resize(n * 9);
    for (int64_t i = 0; i < n; i++) {
        for (int k = 0; k < 3; k++) {
            c.means[i * 3 + k] = static_cast<float>(i) + 0.25f * static_cast<float>(k);
            c.scales[i * 3 + k] = std::log(0.05f + 0.01f * static_cast<float>(i));
            c.sh0[i * 3 + k] = 0.1f * static_cast<float>(k);
        }
        c.quats[i * 4 + 0] = 1.0f;
        c.opacities[i] = logit_f32(op_act[static_cast<size_t>(i)]);
        for (int k = 0; k < 9; k++) c.shN[i * 9 + k] = 0.01f * static_cast<float>(k);
    }
    return c;
}

template <typename Fn>
bool throws_invalid(Fn&& fn) {
    try {
        fn();
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    }
}

} // namespace

// (a) Balanced mode: entering critical → pause (observing that progress stops) → hysteresis at
// serious or below (continuously sustained — any re-escalation resets the clock) → resume. serious
// alone triggers **no intervention** — it is the normal operating state of an iPhone under a
// sustained GPU load (the §7.4 wording is "stop when serious is **exceeded**": the previous serious
// pause threshold of splatx was one level lower than the design — this guards against a regression
// that would lower the threshold again. In lowheat, serious does pause — see (c)).
SX_TEST(thermal_balanced_critical_pause_serious_resume_hysteresis) {
    // The guard against invalid options (the ctor throw contract)
    SX_CHECK(throws_invalid([] {
        ThermalOptions bad;
        bad.resume_hold_ms = -1;
        ThermalGovernor g(bad);
    }));

    ThermalOptions o;
    o.mode = ThermalMode::Balanced;
    o.resume_hold_ms = 120;      // the hysteresis — shortened for the test (the default is 5000)
    o.critical_abort_ms = 60000; // isolated so it never fires — abort is covered in (b)
    ThermalGovernor gov(o);

    StepWorker w(gov);
    w.run();
    // nominal — normal progress
    SX_CHECK(wait_for([&] { return w.steps.load() >= 3; }, 2000));
    // fair — below the balanced-mode pause threshold (critical): no intervention, progress continues
    gov.set_thermal_state(ThermalState::Fair);
    const int64_t at_fair = w.steps.load();
    SX_CHECK(wait_for([&] { return w.steps.load() >= at_fair + 3; }, 2000));
    SX_CHECK(!gov.paused());
    // serious — likewise no intervention (just below the pause threshold boundary — the crux of
    // being consistent with the "exceeded" wording in §7.4)
    gov.set_thermal_state(ThermalState::Serious);
    const int64_t at_serious = w.steps.load();
    SX_CHECK(wait_for([&] { return w.steps.load() >= at_serious + 3; }, 2000));
    SX_CHECK(!gov.paused());
    SX_CHECK(gov.pause_count() == 0);

    // Entering critical → pause: observe that step progress stops (submission halts — §7.4)
    gov.set_thermal_state(ThermalState::Critical);
    SX_CHECK(wait_for([&] { return gov.paused(); }, 2000));
    const int64_t frozen = w.steps.load(); // read after observing paused — the worker is inside before_step
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    SX_CHECK(w.steps.load() == frozen); // zero progress while paused
    SX_CHECK(gov.paused());
    SX_CHECK(gov.pause_count() == 1);

    // Hysteresis reset: after returning to serious, re-escalating to critical before the hold
    // (120 ms) elapses → the sustained duration is not met, so the old serious clock MUST NOT
    // trigger a resume (the continuously-sustained contract)
    gov.set_thermal_state(ThermalState::Serious);
    std::this_thread::sleep_for(std::chrono::milliseconds(25)); // < hold
    gov.set_thermal_state(ThermalState::Critical);
    std::this_thread::sleep_for(std::chrono::milliseconds(250)); // long enough to have resumed if the old clock still counted
    SX_CHECK(gov.paused());
    SX_CHECK(w.steps.load() == frozen);
    SX_CHECK(gov.pause_count() == 1); // it never resumed — the pause count stays at 1

    // The hold is sustained **continuously** at serious or below → resume (the resume threshold =
    // the pause threshold − 1 level: a critical pause resumes once it has cooled only as far as
    // serious — this prevents deadlock). The lower bound: from injection to the first progress
    // >= hold
    const auto t0 = Clock::now();
    gov.set_thermal_state(ThermalState::Serious);
    SX_CHECK(wait_for([&] { return w.steps.load() > frozen; }, 5000));
    SX_CHECK(ms_between(t0, Clock::now()) >= 120.0);
    SX_CHECK(!gov.paused());

    gov.set_thermal_state(ThermalState::Nominal); // return to a state where the worker can stop normally (dtor)
}

// (b) Sustained critical → AbortCritical → a partial export file is written and the loop exits with
// aborted_thermal (the same loop structure and export path as return code 2 of TrainC
// sx_train_run — filter_cloud_for_export → write_inria_ply). A transient critical (shorter than the
// sustained duration) resumes via hysteresis with no abort, and re-entering starts a new clock (no
// accumulation from the previous episode). Because the balanced-mode pause threshold is critical,
// the abort fires **while paused** — this doubles as verification of the contract that the abort
// clock keeps running during a pause.
SX_TEST(thermal_critical_abort_partial_export) {
    ThermalOptions o;
    o.mode = ThermalMode::Balanced;
    o.resume_hold_ms = 60;
    o.critical_abort_ms = 150; // the sustained-duration decision — shortened for the test (the default is 10000)
    ThermalGovernor gov(o);

    const SplatCloud cloud = make_export_cloud();
    const std::string out = sxtest::tmp_path("splatx_thermal_partial_export.ply");
    std::remove(out.c_str());

    std::atomic<int64_t> steps{0};
    std::atomic<bool> stop{false};
    std::atomic<bool> aborted_thermal{false};
    ExportFilterStats fst;
    // Reproduces the TrainC training loop: on Abort → partial export (the currently alive state),
    // then the graceful-exit flag
    std::thread loop([&] {
        while (!stop.load()) {
            if (gov.before_step() == ThermalDecision::AbortCritical) {
                const SplatCloud filtered = filter_cloud_for_export(cloud, 0.005f, &fst);
                write_inria_ply(filtered, out);
                aborted_thermal.store(true);
                return;
            }
            steps.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    SX_CHECK(wait_for([&] { return steps.load() >= 3; }, 2000));

    // A transient critical (30 ms < 150 ms) — no abort: it returns to fair (at or below the serious
    // resume threshold) → resumes after the hysteresis (60 ms)
    gov.set_thermal_state(ThermalState::Critical);
    SX_CHECK(wait_for([&] { return gov.paused(); }, 2000)); // critical = the balanced pause threshold
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    gov.set_thermal_state(ThermalState::Fair);
    const int64_t before_resume = steps.load();
    SX_CHECK(wait_for([&] { return steps.load() > before_resume; }, 5000));
    SX_CHECK(!aborted_thermal.load()); // a transient critical MUST NOT abort (the sustained-duration contract)

    // Sustained critical — a **new clock** starts at the moment of re-entry (if the previous episode
    // had accumulated, it would abort immediately → the lower-bound check below catches that). The
    // loop exits on its own at the abort.
    const auto t0 = Clock::now();
    gov.set_thermal_state(ThermalState::Critical);
    SX_CHECK(wait_for([&] { return gov.paused() || aborted_thermal.load(); }, 2000));
    const int64_t frozen = steps.load();
    SX_CHECK(wait_for([&] { return aborted_thermal.load(); }, 10000));
    const double abort_ms = ms_between(t0, Clock::now());
    if (!aborted_thermal.load()) { // escape hatch to avoid a hang if the contract fails (the check above already failed)
        stop.store(true);
        gov.set_thermal_state(ThermalState::Nominal);
    }
    loop.join();
    SX_CHECK(aborted_thermal.load());
    SX_CHECK(abort_ms >= 150.0);        // the sustained-duration lower bound (measured from re-entry — this also verifies the clock reset)
    SX_CHECK(steps.load() == frozen);   // the abort happens with no steps run since the pause (submission stays halted)

    // Verifying the partial export: the alive state is saved, and the existing export filter
    // convention applies (one low-opacity row is removed)
    SX_CHECK(fst.input == 3 && fst.kept == 2 && fst.removed_low_opacity == 1);
    const SplatCloud reread = read_inria_ply(out);
    SX_CHECK(reread.count == 2);
    // The default write zero-pads f_rest to 45 entries (design §8.1 — PlyIO.h): deg1 → deg3 is inferred
    SX_CHECK(reread.sh_degree == 3);
    if (reread.count == 2) SX_CHECK(reread.means[0] == cloud.means[0]); // a smoke check that rows are preserved
    std::remove(out.c_str());

    // The abort is terminal — every later poll also returns AbortCritical immediately (the caller's
    // break contract; there is no recovery, not even by leaving critical)
    gov.set_thermal_state(ThermalState::Nominal);
    SX_CHECK(gov.before_step() == ThermalDecision::AbortCritical);
}

// (c) LowHeat mode: the pause threshold is serious — the previous behavior of balanced mode (the
// resume threshold is fair — staying at serious never resumes) — plus idling inserted between steps
// (the lower bound that N steps take >= N·idle; balanced mode is the no-idle comparison). The fair
// no-intervention boundary is covered in (f).
SX_TEST(thermal_lowheat_threshold_and_idle) {
    ThermalOptions o;
    o.mode = ThermalMode::LowHeat;
    o.resume_hold_ms = 60;
    o.critical_abort_ms = 60000;
    o.lowheat_idle_ms = 0; // (1) the threshold verification is isolated, with no idling
    {
        ThermalGovernor gov(o);
        StepWorker w(gov);
        w.run();
        SX_CHECK(wait_for([&] { return w.steps.load() >= 3; }, 2000));

        // Entering serious alone causes a pause (in balanced mode serious triggers no intervention,
        // per (a) — the comparison for the lower threshold)
        gov.set_thermal_state(ThermalState::Serious);
        SX_CHECK(wait_for([&] { return gov.paused(); }, 2000));
        const int64_t frozen = w.steps.load();
        std::this_thread::sleep_for(std::chrono::milliseconds(150)); // > hold — serious MUST NOT resume
        SX_CHECK(gov.paused());
        SX_CHECK(w.steps.load() == frozen); // the resume threshold is fair — staying at serious stays halted

        // fair sustained continuously → resume (the resume threshold = the pause threshold − 1
        // level; the hysteresis lower bound is 60 ms)
        const auto t0 = Clock::now();
        gov.set_thermal_state(ThermalState::Fair);
        SX_CHECK(wait_for([&] { return w.steps.load() > frozen; }, 5000));
        SX_CHECK(ms_between(t0, Clock::now()) >= 60.0);
        SX_CHECK(!gov.paused());
        gov.set_thermal_state(ThermalState::Nominal);
    } // the worker stops (nominal — no waiting)

    // (2) Idling between steps: idle 40 ms — 5 polls take at least 5×40 ms (this eases the duty
    // cycle of a sustained load). Polled directly on this thread, per the single-consumer
    // convention.
    o.lowheat_idle_ms = 40;
    ThermalGovernor idle_gov(o);
    const auto t_idle = Clock::now();
    for (int i = 0; i < 5; i++)
        SX_CHECK(idle_gov.before_step() == ThermalDecision::Proceed);
    SX_CHECK(ms_between(t_idle, Clock::now()) >= 5 * 40.0);

    // The balanced-mode comparison — no idling is inserted (5 polls take less than half the total
    // idle time; the slack is >100×)
    ThermalOptions ob;
    ob.mode = ThermalMode::Balanced;
    ThermalGovernor fast_gov(ob);
    const auto t_fast = Clock::now();
    for (int i = 0; i < 5; i++)
        SX_CHECK(fast_gov.before_step() == ThermalDecision::Proceed);
    SX_CHECK(ms_between(t_fast, Clock::now()) < 100.0);
}

// (f) The LowHeat mode boundary: fair sits just below the pause threshold (serious) — **no
// intervention**, progress continues. The previous lowheat behavior of splatx paused at fair (too
// aggressive a threshold — fair is an everyday state, so it paused essentially all the time) — this
// guards against a regression that would lower the threshold back from serious. Entering serious
// pauses immediately, giving the comparison just above the boundary as well.
SX_TEST(thermal_lowheat_fair_no_pause_boundary) {
    ThermalOptions o;
    o.mode = ThermalMode::LowHeat;
    o.resume_hold_ms = 60;
    o.critical_abort_ms = 60000; // isolated so it never fires
    o.lowheat_idle_ms = 0;       // isolates the threshold boundary — idling is covered in (c)
    ThermalGovernor gov(o);

    StepWorker w(gov);
    w.run();
    SX_CHECK(wait_for([&] { return w.steps.load() >= 3; }, 2000));

    // fair — no intervention: progress continues and no pause occurs (just below the boundary)
    gov.set_thermal_state(ThermalState::Fair);
    const int64_t at_fair = w.steps.load();
    SX_CHECK(wait_for([&] { return w.steps.load() >= at_fair + 3; }, 2000));
    SX_CHECK(!gov.paused());
    SX_CHECK(gov.pause_count() == 0);

    // serious — pauses immediately (the comparison just above the boundary)
    gov.set_thermal_state(ThermalState::Serious);
    SX_CHECK(wait_for([&] { return gov.paused(); }, 2000));
    SX_CHECK(gov.pause_count() == 1);

    gov.set_thermal_state(ThermalState::Nominal); // resumes after the hysteresis → the worker stops normally (dtor)
}

// (d) The external hold latch (Phase 6 2nd round Task 5 — the "pause pinning surface" of
// pauseAndDrain, Task 4 carry-over ②): a hold parks the loop regardless of the thermal state or the
// mode, and **no thermal transition can release it, including one that satisfies the resume
// hysteresis** — this verifies the resume race window that existed in the state-injection pause. It
// works even in Performance mode (no intervention) — external control is orthogonal to the mode
// axis.
SX_TEST(thermal_external_hold_latch) {
    ThermalOptions o;
    o.mode = ThermalMode::Performance; // a hold MUST park the loop even in the no-intervention mode
    o.resume_hold_ms = 30;
    ThermalGovernor gov(o);

    StepWorker w(gov);
    w.run();
    SX_CHECK(wait_for([&] { return w.steps.load() >= 3; }, 2000));

    // Setting the hold → park (wait_paused is the observation surface — based on a condition variable)
    gov.set_external_hold(true);
    SX_CHECK(gov.external_hold());
    SX_CHECK(gov.wait_paused(2000));
    const int64_t frozen = w.steps.load();
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    SX_CHECK(w.steps.load() == frozen); // zero progress while held

    // The crux of the resume race verification: not even a storm of thermal transitions (including
    // a fully-cooled signal) releases the hold
    gov.set_thermal_state(ThermalState::Serious);
    gov.set_thermal_state(ThermalState::Nominal);
    std::this_thread::sleep_for(std::chrono::milliseconds(120)); // > resume_hold_ms
    SX_CHECK(gov.paused());
    SX_CHECK(w.steps.load() == frozen);

    // Releasing it → resume (thermal is nominal — immediately)
    gov.set_external_hold(false);
    SX_CHECK(wait_for([&] { return w.steps.load() > frozen; }, 2000));
    SX_CHECK(!gov.paused());

    // In Balanced mode, releasing the hold still leaves the loop parked while the thermal pause
    // condition (critical) holds (releasing the hold does not guarantee an immediate resume — the
    // header contract. The critical period of 100 ms is far below the 10 s abort default)
    ThermalOptions ob;
    ob.mode = ThermalMode::Balanced;
    ob.resume_hold_ms = 30;
    ThermalGovernor gov2(ob);
    StepWorker w2(gov2);
    w2.run();
    SX_CHECK(wait_for([&] { return w2.steps.load() >= 1; }, 2000));
    gov2.set_external_hold(true);
    SX_CHECK(gov2.wait_paused(2000));
    gov2.set_thermal_state(ThermalState::Critical); // thermal escalates while held (the pause threshold)
    gov2.set_external_hold(false);                  // the hold is released — critical takes over
    const int64_t frozen2 = w2.steps.load();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    SX_CHECK(gov2.paused());
    SX_CHECK(w2.steps.load() == frozen2);
    gov2.set_thermal_state(ThermalState::Nominal); // resumes after the hysteresis → the worker stops normally
    SX_CHECK(wait_for([&] { return w2.steps.load() > frozen2; }, 5000));
}

// (e) External stop (a Task 4 carry-over — "the external surface by which a cancellation wakes the
// training thread during a pause"): request_external_stop wakes the loop immediately even while
// parked by a hold or by thermal, returns ExternalStop, and is terminal (every later poll also
// returns ExternalStop — the same result-preservation contract as abort).
SX_TEST(thermal_external_stop_wakes_pause) {
    ThermalOptions o;
    o.mode = ThermalMode::Balanced;
    o.resume_hold_ms = 60000; // isolated so a resume is impossible — to show that stop is the only wake source
    ThermalGovernor gov(o);

    std::atomic<int64_t> steps{0};
    std::atomic<bool> stopped{false};
    std::thread th([&] {
        for (;;) {
            const ThermalDecision d = gov.before_step();
            if (d == ThermalDecision::ExternalStop) {
                stopped.store(true);
                return;
            }
            if (d == ThermalDecision::AbortCritical) return; // never fires (the abort default is 60 s here)
            steps.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    SX_CHECK(wait_for([&] { return steps.load() >= 3; }, 2000));
    // Park it with a hold and then stop — verifies waking during a park (the same wait point as a
    // thermal pause)
    gov.set_external_hold(true);
    SX_CHECK(gov.wait_paused(2000));
    const int64_t frozen = steps.load();
    gov.request_external_stop();
    SX_CHECK(wait_for([&] { return stopped.load(); }, 2000));
    SX_CHECK(steps.load() == frozen); // after waking it exits immediately, with no step run
    th.join();

    // Terminal: every later poll also returns ExternalStop immediately (the single-consumer
    // convention — polled directly on this thread)
    SX_CHECK(gov.before_step() == ThermalDecision::ExternalStop);
}
