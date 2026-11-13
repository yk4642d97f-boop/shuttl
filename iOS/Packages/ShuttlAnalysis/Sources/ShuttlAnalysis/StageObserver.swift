import Foundation
import ShuttlCore

/// Optional per-stage hooks for the app (resource sampling, logging). Called for every
/// stage that actually runs (not for stages restored from a checkpoint), on the task
/// running the pipeline; `stageDidFinish` is called on success and on failure alike.
/// Keep them cheap: "shots" runs once per rally.
public protocol StageObserver: Sendable {
    func stageDidStart(_ name: String)
    func stageDidFinish(_ name: String, seconds: Double)
    /// Extra keys merged into diagnostics["stages"][name] after `stageDidFinish`
    /// (e.g. "peak_phys_footprint_mb", "thermal_state"). Existing keys always win.
    func stageDiagnostics(_ name: String) -> [String: JSONValue]
}

extension StageObserver {
    public func stageDiagnostics(_ name: String) -> [String: JSONValue] { [:] }
}

/// The default observer: does nothing.
public struct NoStageObserver: StageObserver {
    public init() {}
    public func stageDidStart(_ name: String) {}
    public func stageDidFinish(_ name: String, seconds: Double) {}
}
