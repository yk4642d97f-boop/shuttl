import Foundation

/// A stage's own watchdog saw no progress for `seconds` (e.g. Vision perform() blocked or
/// starved in Low Power Mode). Retryable: the stage's partial checkpoint keeps what was done.
public struct StageStalledError: Error, CustomStringConvertible, Sendable, Equatable {
    public var stage: String
    public var seconds: Double
    /// Last frame that completed (-1 = none in this pass).
    public var lastFrame: Int

    public init(stage: String, seconds: Double, lastFrame: Int) {
        self.stage = stage; self.seconds = seconds; self.lastFrame = lastFrame
    }

    public var description: String {
        "\(stage) stalled: no progress for \(Int(seconds.rounded())) s after frame \(lastFrame)"
    }
}
