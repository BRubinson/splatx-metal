import Foundation
import Observation
import SplatXMetal

@MainActor @Observable final class TrainProgressStore {
    var log = ProgressLog()
    var latest: SplatXTrainer.Progress?

    func reset() { log = ProgressLog(); latest = nil }

    /// The callback invoked from the training thread (@Sendable). It copies only the values and
    /// hops to the MainActor.
    /// Observation coalesces the UI at frame rate, so appending on every step costs little.
    nonisolated func callback() -> @Sendable (SplatXTrainer.Progress) -> Void {
        { [weak self] p in
            Task { @MainActor in
                guard let self else { return }
                self.latest = p
                self.log.append(iteration: p.iteration, loss: p.loss, psnr: p.psnr)
            }
        }
    }
}
