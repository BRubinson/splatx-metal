// Thermal governance implementation (Phase 6 round 2 Task 4 — the policy and thread contract are in
// the ThermalGovernor.h header comment).
#include "splatx/ThermalGovernor.h"

#include <cstdio>
#include <stdexcept>

namespace splatx {

namespace {

const char* state_name(ThermalState s) {
    switch (s) {
        case ThermalState::Nominal: return "nominal";
        case ThermalState::Fair: return "fair";
        case ThermalState::Serious: return "serious";
        case ThermalState::Critical: return "critical";
    }
    return "?";
}

const char* mode_name(ThermalMode m) {
    switch (m) {
        case ThermalMode::Performance: return "performance";
        case ThermalMode::Balanced: return "balanced";
        case ThermalMode::LowHeat: return "lowheat";
    }
    return "?";
}

std::chrono::milliseconds ms(int32_t v) { return std::chrono::milliseconds(v); }

} // namespace

ThermalGovernor::ThermalGovernor(const ThermalOptions& opts, ThermalState initial)
    : opts_(opts),
      // pause threshold (§7.4 "stop submitting **above** serious" — for Balanced that is critical):
      // serious is the normal operating state of a sustained GPU load on iPhone, so it triggers no
      // intervention (measured in the A17 bake-off — the header policy comment). Only LowHeat pauses
      // at serious.
      pause_at_(opts.mode == ThermalMode::LowHeat ? ThermalState::Serious
                                                  : ThermalState::Critical),
      // resume threshold = pause threshold − 1 step (generalized): while paused at critical, the
      // device must resume even if it only cools back down to serious, or it deadlocks (the header
      // policy comment).
      resume_at_(static_cast<ThermalState>(static_cast<int32_t>(pause_at_) - 1)),
      state_(initial) {
    if (opts.mode != ThermalMode::Performance && opts.mode != ThermalMode::Balanced &&
        opts.mode != ThermalMode::LowHeat)
        throw std::invalid_argument("ThermalGovernor: mode must be performance|balanced|lowheat");
    if (opts.resume_hold_ms < 0 || opts.critical_abort_ms < 0 || opts.lowheat_idle_ms < 0)
        throw std::invalid_argument("ThermalGovernor: ms options must be >= 0");
    if (initial < ThermalState::Nominal || initial > ThermalState::Critical)
        throw std::invalid_argument("ThermalGovernor: invalid initial thermal state");
    const auto now = Clock::now();
    if (state_ == ThermalState::Critical) critical_since_ = now;
    if (state_ <= resume_at_) below_since_ = now;
}

void ThermalGovernor::set_thermal_state(ThermalState s) {
    if (s < ThermalState::Nominal || s > ThermalState::Critical)
        throw std::invalid_argument("ThermalGovernor: invalid thermal state");
    std::lock_guard<std::mutex> lk(mu_);
    if (s == state_) return;
    const auto now = Clock::now();
    // Transition observation log — common to every mode, including the Performance mode contract of
    // "observation logs only".
    std::printf("thermal: %s -> %s (mode %s%s)\n", state_name(state_), state_name(s),
                mode_name(opts_.mode),
                opts_.mode == ThermalMode::Performance ? " — no intervention" : "");
    std::fflush(stdout);
    // critical duration clock: only **continuous** duration counts — leaving and re-entering resets it (header contract).
    if (s == ThermalState::Critical && state_ != ThermalState::Critical) critical_since_ = now;
    // resume hysteresis clock: runs from **the moment it drops** to or below the threshold — moving
    // around inside the below-threshold band (fair→nominal, etc.) does not reset it; going back above
    // and re-entering does.
    if (s <= resume_at_ && state_ > resume_at_) below_since_ = now;
    state_ = s;
    cv_.notify_all();
}

ThermalState ThermalGovernor::thermal_state() const {
    std::lock_guard<std::mutex> lk(mu_);
    return state_;
}

bool ThermalGovernor::paused() const {
    std::lock_guard<std::mutex> lk(mu_);
    return paused_;
}

int64_t ThermalGovernor::pause_count() const {
    std::lock_guard<std::mutex> lk(mu_);
    return pause_count_;
}

void ThermalGovernor::set_external_hold(bool hold) {
    std::lock_guard<std::mutex> lk(mu_);
    if (hold_ == hold) return;
    hold_ = hold;
    std::printf("thermal: external hold %s\n", hold ? "set — park requested at the step boundary" : "released");
    std::fflush(stdout);
    cv_.notify_all();
}

bool ThermalGovernor::external_hold() const {
    std::lock_guard<std::mutex> lk(mu_);
    return hold_;
}

void ThermalGovernor::request_external_stop() {
    std::lock_guard<std::mutex> lk(mu_);
    if (stopped_) return;
    stopped_ = true;
    std::printf("thermal: external stop requested — exiting at the next step boundary (or park wake-up)\n");
    std::fflush(stdout);
    cv_.notify_all();
}

bool ThermalGovernor::wait_paused(int32_t timeout_ms) {
    std::unique_lock<std::mutex> lk(mu_);
    return cv_.wait_for(lk, ms(timeout_ms), [&] { return paused_; });
}

void ThermalGovernor::begin_pause_locked(Clock::time_point now, const char* reason) {
    paused_ = true;
    pause_count_++;
    pause_began_ = now;
    std::printf("thermal: pause (%s) — step submission suspended\n", reason);
    std::fflush(stdout);
    cv_.notify_all(); // wakes the wait_paused observers (Task 5 pauseAndDrain)
}

void ThermalGovernor::end_pause_locked(Clock::time_point now, const char* reason) {
    paused_ = false;
    std::printf("thermal: resume (%s) — pause %.1f s\n", reason,
                std::chrono::duration<double>(now - pause_began_).count());
    std::fflush(stdout);
}

ThermalDecision ThermalGovernor::before_step() {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
        // ── Decision loop: waits on cv until Proceed is settled (break) — no busy-wait ──
        for (;;) {
            const auto now = Clock::now();
            // External stop — terminal, and takes precedence over hold/pause/abort (user intent. Header contract).
            if (stopped_) {
                if (paused_) end_pause_locked(now, "external stop");
                return ThermalDecision::ExternalStop;
            }
            // External hold latch — parks unconditionally, regardless of mode or thermal state.
            // While held, even the critical abort decision is deferred (an external thread MUST NOT
            // be woken mid Trainer snapshot — header comment).
            if (hold_) {
                if (!paused_) begin_pause_locked(now, "external hold");
                cv_.wait(lk);
                continue;
            }
            if (opts_.mode == ThermalMode::Performance) { // no intervention — no thermal decision
                if (paused_) end_pause_locked(now, "external hold released");
                return ThermalDecision::Proceed;
            }
            if (aborted_) return ThermalDecision::AbortCritical; // terminal (header contract)
            if (state_ == ThermalState::Critical &&
                now - critical_since_ >= ms(opts_.critical_abort_ms)) {
                aborted_ = true;
                paused_ = false;
                std::printf("thermal: critical sustained for %d ms continuously — graceful "
                            "abort decided (caller: partial export then exit)\n",
                            opts_.critical_abort_ms);
                std::fflush(stdout);
                return ThermalDecision::AbortCritical;
            }
            if (!paused_) {
                if (state_ < pause_at_) break; // Proceed (for LowHeat, the idle insertion below applies)
                std::printf("thermal: state %s >= %s (resume: held at or below %s for %d ms continuously)\n",
                            state_name(state_), state_name(pause_at_), state_name(resume_at_),
                            opts_.resume_hold_ms);
                begin_pause_locked(now, "thermal");
                continue; // on to the wait branch
            }
            // paused — waits until the resume (hysteresis) or critical abort condition is met
            if (state_ <= resume_at_) {
                const auto deadline = below_since_ + ms(opts_.resume_hold_ms);
                if (now >= deadline) {
                    end_pause_locked(now, "thermal hysteresis");
                    break; // Proceed
                }
                cv_.wait_until(lk, deadline);
            } else if (state_ == ThermalState::Critical) {
                cv_.wait_until(lk, critical_since_ + ms(opts_.critical_abort_ms));
            } else {
                cv_.wait(lk); // stays paused (above the resume threshold) — idles until the state changes
            }
        }
        // ── Proceed — LowHeat inserts idle between steps (eases the sustained load) ──
        if (opts_.mode != ThermalMode::LowHeat || opts_.lowheat_idle_ms <= 0)
            return ThermalDecision::Proceed;
        const auto idle_end = Clock::now() + ms(opts_.lowheat_idle_ms);
        bool escalated = false;
        while (Clock::now() < idle_end) {
            if (state_ >= pause_at_ || hold_ || stopped_) { // escalation or external control during idle — early wake
                escalated = true;
                break;
            }
            cv_.wait_until(lk, idle_end);
        }
        if (!escalated) return ThermalDecision::Proceed;
        // re-enters the decision loop (handles pause/abort/hold/stop)
    }
}

} // namespace splatx
