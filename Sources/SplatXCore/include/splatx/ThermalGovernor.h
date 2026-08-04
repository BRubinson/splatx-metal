#pragma once
// Thermal governance (Phase 6 round 2 Task 4 — design §7.4 pause-and-cool).
//
// Principle: **state injection is a first-class design**. This library does not subscribe to the
// Foundation thermal API directly — observing ProcessInfo.thermalState is owned by the app (Swift)
// layer, and the observer injects it through set_thermal_state() (the C surface is
// sx_thermal_set_state — splatx_train_c.h).
// That makes it possible to unit-verify the entire policy from an injection sequence alone,
// without driving a real device into serious (ThermalGovernorTests — the plan's Global Constraints
// entry "thermal-injectable design").
// The macOS CLI has no injection source, so the state stays fixed at nominal = observation with no
// action (the hidden test injection path is described in the SPLATX_THERMAL_* comments in
// TrainC.cpp).
//
// Policy (§7.4 — three modes; resume threshold = pause threshold − 1 step, generalized — cooling to
// just below the pause threshold must be enough to resume, otherwise there is deadlock):
//   Performance: no intervention — only an observation log of state transitions
//     (set_thermal_state logs the transition).
//   Balanced (default): entering critical → before_step() pauses (GPU submission stops — it waits
//     idle on a condition variable at a step-boundary safe point; busy-wait is forbidden). After
//     staying at serious or below **continuously** for resume_hold_ms (hysteresis — any
//     re-escalation resets the clock), it resumes (not an abort).
//     serious itself triggers no intervention — under sustained GPU load on iPhone, serious is a
//     normal operating state (measured in the A17 bake-off: msplat/brush ran to completion in
//     16–19 minutes at serious, whereas the previous serious pause took 39 minutes (2x) because of
//     repeated pausing — this restores consistency with the §7.4 wording "stop GPU submission when
//     thermalState **exceeds** serious").
//     critical sustained **continuously** for critical_abort_ms → returns AbortCritical —
//     **the abort clock stays live during a pause as well** (a critical pause sleeps only until the
//     abort deadline, and decides when the deadline is reached). The caller performs a partial
//     export (the currently alive state as PLY/SPZ — reusing the existing export path) and then a
//     graceful abort (a distinct termination, not an error — sx_train_run returns 2). The abort is
//     terminal — every subsequent poll immediately returns AbortCritical as well (leaving critical
//     does not recover it either: the contract is to preserve the result and stop).
//   LowHeat: Balanced + the pause threshold lowered to serious (resume threshold fair — the
//     previous Balanced threshold) + an idle gap of lowheat_idle_ms inserted between steps on
//     every Proceed (easing the duty cycle under sustained load — if the state escalates to the
//     pause threshold or above during the idle gap, it wakes early and re-evaluates the pause).
//     The previous fair pause threshold was excessive — fair is an everyday state, so it amounted
//     to pausing all the time.
//
// Thread contract: set_thermal_state/thermal_state/paused/pause_count/set_external_hold/
// request_external_stop/wait_paused may be called from any thread. before_step() is for the
// training-loop thread only (a single consumer). **While paused, the training thread is stopped at
// a step boundary and does not touch Trainer** — this is the point at which a checkpoint snapshot
// (Trainer::make_checkpoint) or a drain from an external thread is safe (consumed by Task 5's
// pauseAndDrain — observed through paused()/wait_paused()).
// The wake sources for the wait during a pause are set_thermal_state, set_external_hold, and
// request_external_stop.
//
// External hold latch (Phase 6 round 2 Task 5 — the "pause pinning surface" of pauseAndDrain,
// Task 4 carry-over ②):
// With set_external_hold(true), before_step parks at a step boundary **regardless of** the thermal
// state and the mode, and while the hold is set no thermal transition (including one that
// satisfies the cooling resume hysteresis) can resume training — this structurally removes the
// "observer injection → resume race" window that existed when a pause was imitated by injecting a
// state (serious) (an external thread MUST NOT be woken while it is snapshotting Trainer).
// While the hold is set, the sustained-critical abort **decision is deferred as well** (not waking
// the loop during a snapshot takes precedence — the critical clock itself keeps running, so the
// decision follows immediately after the hold is released). hold/stop work in Performance mode too
// (the no-intervention contract covers thermal decisions only — external control is orthogonal to
// the mode axis).
//
// External stop (Task 4 carry-over — "the external surface by which a cancel wakes the training
// thread during a pause"):
// request_external_stop() is terminal — it wakes the thread immediately even while parked (hold or
// thermal pause), before_step returns ExternalStop, and every subsequent poll returns ExternalStop
// as well (the same preserve-the-result-then-stop contract as abort — the caller performs a partial
// export and/or checkpoint and then terminates gracefully).
//
// Time: steady_clock + millisecond parameters. Tests use short values (tens of ms) and assert only
// lower-bound properties (zero progress while paused, resume only after the hysteresis, idle of at
// least N·idle — no upper-bound assertions).
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace splatx {

// 1:1 with Foundation ProcessInfo.ThermalState (nominal=0 .. critical=3).
enum class ThermalState : int32_t { Nominal = 0, Fair = 1, Serious = 2, Critical = 3 };

enum class ThermalMode : int32_t {
    Performance = 0, // no intervention — only an observation log of state transitions
    Balanced = 1,    // default — pause at critical / hysteresis resume at serious or below /
                     //   abort on sustained critical
    LowHeat = 2,     // Balanced + idle between steps + pause threshold lowered to serious
                     //   (resume threshold fair)
};

struct ThermalOptions {
    ThermalMode mode = ThermalMode::Balanced;
    // Hysteresis: how long the state must stay **continuously** at or below the resume threshold.
    // Default 5s — thermalState transitions are a slow signal on the order of tens of seconds, so
    // a shorter value oscillates between pause and resume and produces no cooling.
    int32_t resume_hold_ms = 5000;
    // Sustained **continuous** critical → decide on a partial export followed by a graceful abort.
    // Default 10s — critical is the stage of forced OS intervention (heavy throttling), so if a
    // pause cannot escape it within this time, preserving the result takes priority
    // (§7.4 "on exceeding the limit, partial export then graceful abort").
    int32_t critical_abort_ms = 10000;
    // The LowHeat idle gap between steps. Default 100ms — sized to lower the GPU duty cycle by
    // ~25% against the ~291ms/iter measured in the A17 smoke run (Phase 6 round 1, docs/bench).
    int32_t lowheat_idle_ms = 100;
};

enum class ThermalDecision : int32_t {
    Proceed = 0,       // proceed with submitting the step (after resuming, if it paused)
    AbortCritical = 1, // sustained critical exceeded — caller: partial export then graceful abort
    ExternalStop = 2,  // request_external_stop — caller: checkpoint/partial export, graceful stop
};

class ThermalGovernor {
public:
    // A negative ms field in opts, or a mode/initial outside its range, throws. initial = the state
    // at construction time (for inheriting the most recent value the observer injected before
    // training started — TrainC ThermalGovernorRegistration).
    explicit ThermalGovernor(const ThermalOptions& opts = {},
                             ThermalState initial = ThermalState::Nominal);

    // State injection — any thread. Logs the transition (observation common to all modes) and
    // wakes the pause/idle wait.
    void set_thermal_state(ThermalState s);
    ThermalState thermal_state() const;

    // Step-boundary poll (training-loop thread only — the thread contract at the top of this
    // header). If the pause condition holds, it blocks inside this call until the resume condition
    // is met (a cv wait — no busy-wait). Performance returns Proceed immediately. LowHeat inserts
    // the idle gap just before Proceed.
    ThermalDecision before_step();

    bool paused() const;         // in a pause wait inside before_step (Task 5 observation surface)
    int64_t pause_count() const; // cumulative pause entries (log/test observation)

    // ===== External control (Phase 6 round 2 Task 5 — the "external hold latch / external stop"
    // comments at the top of this header) =====
    // Sets/clears the hold latch — any thread. When true it parks regardless of mode and thermal
    // state, and this call is the only way to clear it (a thermal transition cannot clear it —
    // this blocks the resume race). After it is cleared, if the thermal pause condition holds, the
    // thermal policy takes over (clearing the hold is not a guarantee of immediate resumption).
    void set_external_hold(bool hold);
    bool external_hold() const;
    // Terminal stop request — any thread. It wakes the thread immediately even while parked, and
    // from then on before_step always returns ExternalStop. There is no surface to cancel the
    // request (the contract is to preserve the result and stop — the same as abort).
    void request_external_stop();
    // Waits until paused() == true (external thread — cv based, no busy-wait). The return value is
    // whether it was reached. Guarding against an infinite wait when the training loop has already
    // finished and no park will ever come is the caller's responsibility (TrainC polls in slices
    // together with a loop-active flag).
    bool wait_paused(int32_t timeout_ms);

private:
    using Clock = std::chrono::steady_clock;

    mutable std::mutex mu_;
    std::condition_variable cv_;
    const ThermalOptions opts_;
    const ThermalState pause_at_;  // pause at or above this (Balanced: critical, LowHeat: serious)
    const ThermalState resume_at_; // resume when continuously at or below — pause_at_ − 1 step
                                   // (Balanced: serious, LowHeat: fair)
    ThermalState state_;
    Clock::time_point critical_since_{}; // critical entry time (leaving then re-entering resets it)
    Clock::time_point below_since_{};    // time it dropped to or below the resume threshold
                                         // (going back above and re-entering resets it)
    Clock::time_point pause_began_{};    // for logging
    bool paused_ = false;
    bool aborted_ = false;
    bool hold_ = false;    // external hold latch (set_external_hold — orthogonal to thermal)
    bool stopped_ = false; // external stop (request_external_stop — terminal)
    int64_t pause_count_ = 0;

    // Shared handling for entering/leaving a pause while holding mu_ (log + wake wait_paused)
    void begin_pause_locked(Clock::time_point now, const char* reason);
    void end_pause_locked(Clock::time_point now, const char* reason);
};

} // namespace splatx
