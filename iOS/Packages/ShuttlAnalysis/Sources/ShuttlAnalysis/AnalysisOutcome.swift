import Foundation
import ShuttlCore

/// What a finished `analyze` call means for the app. The pipeline degrades instead of
/// throwing (Python parity), so "returned" ≠ "worked": classify before marking a match done.
public enum AnalysisOutcome: Sendable, Equatable {
    /// Rallies found. Enrichment stages may still have failed (see `MatchAnalysis.degradedStages`).
    case complete
    /// The shuttle was (almost) never seen: no track, or visible frames under the threshold.
    case noShuttle
    /// The shuttle was seen but no rally came out.
    case noRallies
    /// A stage failed so the result is unusable (essential stage) or worth redoing (retryable).
    /// Checkpoints of the stages that succeeded are kept, so a retry resumes.
    case stageFailed(stage: String, message: String, retryable: Bool)
    /// The phone was hand-held or kept moving: every stage after detection assumes a fixed
    /// camera, so rallies, speeds and calls would be invented. Not retryable.
    case cameraUnsteady

    public var isRetryable: Bool {
        if case .stageFailed(_, _, let r) = self { return r }
        return false
    }

    /// Chinese text for the user.
    public var userMessage: String { analysisOutcomeMessage(self) }

    /// Classifies an error thrown before / outside the stages (e.g. the video would not open).
    public static func failure(_ error: Error, stage: String = "open") -> AnalysisOutcome {
        let f = StageFailure(stage: stage, error: error)
        return .stageFailed(stage: stage, message: f.message, retryable: f.retryable)
    }
}

/// Thresholds for `MatchAnalysis.outcome(policy:)`.
public struct AnalysisOutcomePolicy: Sendable, Equatable {
    /// Fewer visible (really detected) shuttle frames than this ⇒ `.noShuttle` …
    public var minVisibleFrames: Int
    /// … or fewer than this fraction of all frames (whichever is larger).
    public var minVisibleFraction: Double
    /// Without these there is nothing to show; their failure is `.stageFailed` even when
    /// not retryable. Other stages only enrich (speeds, positions, shot types, score).
    public var essentialStages: Set<String>

    /// When a recording counts as hand-held (see `CameraSteadiness`).
    public var steadiness: SteadinessPolicy

    public init(minVisibleFrames: Int = 30, minVisibleFraction: Double = 0.01,
                essentialStages: Set<String> = ["open", "detect", "link", "hits", "rallies"],
                steadiness: SteadinessPolicy = .default) {
        self.minVisibleFrames = minVisibleFrames
        self.minVisibleFraction = minVisibleFraction
        self.essentialStages = essentialStages
        self.steadiness = steadiness
    }

    public static let `default` = AnalysisOutcomePolicy()
}

extension MatchAnalysis {
    /// See `AnalysisOutcome`; uses the default thresholds.
    public var outcome: AnalysisOutcome { outcome(policy: .default) }

    /// Order: a retryable failure anywhere (retrying can fix it, checkpoints make it cheap),
    /// then a failed essential stage, then content (rallies / shuttle visibility).
    public func outcome(policy: AnalysisOutcomePolicy) -> AnalysisOutcome {
        let failures = stageFailures
        if let f = failures.first(where: \.retryable) {
            return .stageFailed(stage: f.stage, message: f.message, retryable: true)
        }
        if let f = failures.first(where: { policy.essentialStages.contains($0.stage) }) {
            return .stageFailed(stage: f.stage, message: f.message, retryable: false)
        }
        if cameraSteadiness?.verdict(policy: policy.steadiness) == .handHeld { return .cameraUnsteady }
        if !rallies.isEmpty { return .complete }
        let needed = max(policy.minVisibleFrames, Int((Double(meta.nFrames) * policy.minVisibleFraction).rounded(.up)))
        if track == nil || visibleShuttleFrames < needed { return .noShuttle }
        return .noRallies
    }

    /// Every failed stage. Reads diagnostics["stage_failures"]; for analyses saved before
    /// that key existed, falls back to the ok:false entries of diagnostics["stages"].
    public var stageFailures: [StageFailure] {
        if let list = diagnostics[StageFailure.diagnosticsKey]?.array {
            return list.compactMap(StageFailure.init(json:))
        }
        guard case .object(let stages)? = diagnostics["stages"] else { return [] }
        return stages.keys.sorted().compactMap { name -> StageFailure? in
            guard let e = stages[name], e["ok"]?.bool == false else { return nil }
            let message = e["error"]?.string ?? ""
            let domain = e["error_domain"]?.string ?? ""
            return StageFailure(stage: name, message: message, domain: domain, code: Int(e["error_code"]?.double ?? 0),
                                retryable: e["retryable"]?.bool
                                    ?? AnalysisErrorClassifier.isRetryable(message: message, domain: domain))
        }
    }

    /// Stages that failed without making the result unusable (no speeds, no shot types …).
    public func degradedStages(policy: AnalysisOutcomePolicy = .default) -> [String] {
        var seen = Set<String>()
        return stageFailures.map(\.stage).filter { !policy.essentialStages.contains($0) && seen.insert($0).inserted }
    }

    /// How still the phone was, from diagnostics["camera_motion"] (nil for analyses without it).
    public var cameraSteadiness: CameraSteadiness? {
        guard let m = diagnostics[CameraSteadiness.diagnosticsKey],
              let shifts = m["shifts_px"]?.array?.compactMap(\.double),
              let corr = m["correlations"]?.array?.compactMap(\.double), shifts.count == corr.count,
              let width = m["image_width"]?.double else { return nil }
        return CameraSteadiness(shiftsPx: shifts, correlations: corr, imageWidth: Int(width),
                                searchPx: m["search_px"]?.double ?? 0)
    }

    /// A line for the result screen when the phone moved during part of the recording, so
    /// speeds and landings in that part may be off (nil when steady or unknown).
    public var steadinessWarning: String? {
        cameraSteadiness?.verdict() == .moved ? "录像中途手机被碰过，有一部分球速和落点可能不准。" : nil
    }

    /// Frames with a real shuttle detection (not gap-filled).
    public var visibleShuttleFrames: Int { track?.visible.reduce(0) { $1 ? $0 + 1 : $0 } ?? 0 }

    /// There are rallies and no essential stage failed: worth showing even when `outcome`
    /// is a retryable enrichment failure whose retries ran out (e.g. no speeds, still rallies).
    public func hasUsableResult(policy: AnalysisOutcomePolicy = .default) -> Bool {
        !rallies.isEmpty && !stageFailures.contains { policy.essentialStages.contains($0.stage) }
    }
}

/// Chinese text for an outcome (what the match card / alert shows).
public func analysisOutcomeMessage(_ outcome: AnalysisOutcome) -> String {
    switch outcome {
    case .complete:
        return "分析完成。"
    case .noShuttle:
        return "没有看到羽毛球。请确认整片球场都在画面里，光线充足。"
    case .noRallies:
        return "看到了羽毛球，但没有找到完整的回合。请拍下完整的来回，并让两边球员都在画面里。"
    case .stageFailed(_, _, true):
        return "分析被打断了，回到 App 后会自动继续。"
    case .stageFailed(let stage, _, false):
        return "\(analysisStageDisplayName(stage))这一步出错了。可以点“重试”再分析一次。"
    case .cameraUnsteady:
        return "拍的时候手机在晃，算出来的球速和落点都不可靠。请把手机固定在三脚架上再录。"
    }
}

/// Chinese name of a pipeline stage (same wording as the app's progress UI).
public func analysisStageDisplayName(_ stage: String) -> String {
    switch stage {
    case "open": return "打开视频"
    case "detect": return "识别羽毛球"
    case "link": return "连接飞行轨迹"
    case "calibrate": return "标定球场"
    case "players": return "追踪球员"
    case "hits": return "判断击球"
    case "reconstruct3d": return "计算落点和球速"
    case "rallies": return "划分回合"
    case "shots": return "识别球路"
    case "score": return "计分"
    case "done": return "整理结果"
    default: return "分析"
    }
}
