// Adam moment state (exp_avg/exp_avg_sq) — the minimal type Trainer consumes (split into its own
// header: this resolves the layer inversion where one moment type pulled in all of Trainer.h).
// The fields and their meaning are exactly the previous Trainer::AdamState (do not change them —
// the using alias in Trainer.h keeps every existing `Trainer::AdamState` spelling valid).
#pragma once
#include <vector>

namespace splatx {

struct AdamState { std::vector<float> exp_avg, exp_avg_sq; };

} // namespace splatx
