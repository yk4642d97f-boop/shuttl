import SwiftUI

// MARK: - Display helpers shared by list, detail and statistics

extension Match {
    var allShots: [ShotPoint] { rallies.flatMap(\.shots) }

    /// Shots with a measured launch speed. Speeds exist only when the court was calibrated
    /// (the analysis maps "no 3-D flight" to 0), so this is the one rule every speed on
    /// screen goes through: list card, detail hero, statistics.
    var measuredShots: [ShotPoint] { allShots.filter(\.hasMeasuredSpeed) }

    /// Finished with shots, but none of them has a speed: the court wasn't found.
    var lacksSpeeds: Bool { status == .done && !allShots.isEmpty && measuredShots.isEmpty }

    var maxSpeed: Float { measuredShots.map(\.speedKmh).max() ?? 0 }

    var maxSpeedText: String { maxSpeed > 0 ? String(format: "%.0f", maxSpeed) : "--" }

    var durationText: String {
        let m = Int(durationSeconds) / 60
        let s = Int(durationSeconds) % 60
        return String(format: "%d:%02d", m, s)
    }

    var courtLabel: String { courtType == .singles ? "单打" : "双打" }

    var canReplay: Bool { status == .done && localVideoPath != nil }

    /// A failed match whose failure would come back the same way (no shuttle in view,
    /// no complete rally, hand-held phone): retrying can't help, re-recording can.
    /// Unknown (older rows, network failures) counts as worth a retry.
    var retryIsPointless: Bool {
        guard status == .failed else { return false }
        return failureRetryable == false
    }

    /// What "重试" will do, for the failed state.
    var retryHint: String {
        switch pipelineStep {
        case .create?, .upload?, nil: return "会重新上传这段视频"
        case .analyze?:               return "视频已在服务器上，会重新开始分析"
        case .poll?:                  return "会继续查询分析结果"
        case .local?:                 return "会在这台手机上重新分析这段视频"
        }
    }

    /// Analysed (or being analysed) on this phone rather than on the server.
    var isOnDevice: Bool { pipelineStep == .local || analysisPath != nil }

    /// One-line live status for list rows and the detail hero.
    func progressText(_ activity: PipelineActivity?) -> String {
        switch status {
        case .uploading:
            switch activity?.phase {
            case .creating?: return "正在连接分析服务器…"
            case .uploading?:
                if let f = activity?.uploadFraction { return "上传中 \(Int((f * 100).rounded()))%" }
                return "上传中…"
            case nil: return "等待上传…"
            default: return "上传完成，准备分析"
            }
        case .analyzing:
            if activity?.phase == .analyzingOnDevice {
                // Waiting (queued / cooling) is not analysing yet: say only why it waits.
                if MatchWaiting(localStage: activity?.localStage) != nil {
                    return AnalysisStageName.display(activity?.localStage ?? "")
                }
                let stage = AnalysisStageName.display(activity?.localStage ?? "")
                guard let f = activity?.localFraction else { return "手机分析中 · \(stage)" }
                return "手机分析中 \(Int((f * 100).rounded(.down)))% · \(stage)"
            }
            if pipelineStep == .local { return "等待在手机上分析…" }
            if activity?.phase == .fetching { return "分析完成，正在取回结果…" }
            if let p = activity?.analysisPercent { return "分析中 \(p)%" }
            if activity?.isQueued == true { return "已排队，等待分析服务开始…" }
            return activity == nil ? "分析中（等待连接服务器）" : "分析中…"
        default:
            return status.displayName
        }
    }

    /// 0…1 for the determinate bar, nil when only an indeterminate one makes sense.
    func progressFraction(_ activity: PipelineActivity?) -> Double? {
        switch (status, activity?.phase) {
        case (.uploading, .uploading?): return activity?.uploadFraction
        case (.analyzing, .analyzing?): return activity?.analysisPercent.map { Double($0) / 100 }
        case (.analyzing, .analyzingOnDevice?): return activity?.localFraction
        default: return nil
        }
    }
}

extension ShotPoint {
    var hasMeasuredSpeed: Bool { speedKmh.isFinite && speedKmh > 0 }
}

// MARK: - Replay route (distinct from Match so both can live on one stack)

struct ReplayRoute: Hashable {
    let match: Match
    /// Frame to open at (a rally's first hit); nil starts from the beginning.
    var startFrame: Int? = nil
}

// MARK: - ShotType

extension ShotType: CaseIterable {
    public static var allCases: [ShotType] {
        [.smash, .clear, .drop, .net, .drive, .serve, .unknown]
    }

    var displayName: String {
        switch self {
        case .smash:   return "扣杀"
        case .clear:   return "高远"
        case .drop:    return "吊球"
        case .net:     return "网前"
        case .drive:   return "平抽"
        case .serve:   return "发球"
        case .unknown: return "其他"
        }
    }
}

// MARK: - ShotMixBar
// A match's "DNA": one capsule split into spot-ink segments by shot type.

struct ShotMixBar: View {
    let shots: [ShotPoint]
    var height: CGFloat = 10
    /// Thin list-row version: hairline dividers and outline.
    var hairline = false

    private var counts: [(type: ShotType, count: Int)] {
        ShotType.allCases.compactMap { t in
            let n = shots.filter { $0.shotType == t }.count
            return n > 0 ? (t, n) : nil
        }
    }

    var body: some View {
        let total = max(1, shots.count)
        GeometryReader { geo in
            HStack(spacing: 0) {
                ForEach(counts, id: \.type) { entry in
                    Rectangle()
                        .fill(entry.type.ink)
                        .frame(width: geo.size.width * CGFloat(entry.count) / CGFloat(total))
                        .overlay(alignment: .trailing) {
                            Rectangle().fill(Color.ink).frame(width: hairline ? AppStroke.hair : 1.25)
                        }
                }
            }
        }
        .frame(height: height)
        .background(Color.paperDeep)
        .clipShape(Capsule())
        .overlay(Capsule().strokeBorder(Color.ink, lineWidth: hairline ? AppStroke.hair : 1.25))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("击球构成：" + counts.map { "\($0.type.displayName) \($0.count)" }.joined(separator: "，"))
    }
}
