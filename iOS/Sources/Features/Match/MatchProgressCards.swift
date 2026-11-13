import SwiftUI

// What fills the detail page while a match isn't finished: a printed checklist of the
// analysis stages (where it is, what's left) and, when it failed, what to change next time.

// MARK: - Stage ladder

struct AnalysisStageLadder: View {
    let match: Match
    let activity: PipelineActivity?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private struct Step {
        let title: String
        let note: String
    }

    /// Pipeline stages folded into five things a player understands.
    private static let steps: [Step] = [
        Step(title: "找羽毛球", note: "每一帧里找到球，连成飞行轨迹"),
        Step(title: "认球场", note: "找四条边线，算出相机在哪儿"),
        Step(title: "看人", note: "追踪两边球员的站位和跑动"),
        Step(title: "判每一拍", note: "谁击的球、什么球路、多快、落在哪"),
        Step(title: "记分", note: "分回合，算比分"),
    ]

    private static func index(forStage stage: String) -> Int {
        switch stage {
        case "detect", "link": return 0
        case "calibrate": return 1
        case "players": return 2
        case "hits", "reconstruct3d", "shots": return 3
        case "rallies", "score", "done": return 4
        default: return 0
        }
    }

    /// The step in progress, or nil before analysis starts (uploading / queued).
    private var current: Int? {
        switch match.status {
        case .uploading, .recording: return nil
        case .analyzing:
            // Waiting its turn (queue, heat, camera, a take): no step is in progress yet.
            if MatchWaiting(localStage: activity?.localStage) != nil { return nil }
            if let stage = activity?.localStage {
                return Self.index(forStage: stage)
            }
            if let fraction = activity?.localFraction { return min(4, Int(fraction * 5)) }
            if let p = activity?.analysisPercent { return min(4, p / 20) }
            return 0
        default: return nil
        }
    }

    private var waiting: String? { Self.waitingLine(status: match.status, activity: activity) }

    /// The ladder's note above the steps. While a match waits, the hero already says why in full,
    /// so the card shows only the short reason (等相机 / 排队中 / 降温中 / 等录完).
    static func waitingLine(status: MatchStatus, activity: PipelineActivity?) -> String? {
        if status == .uploading { return "上传完成后从第一步开始" }
        return MatchWaiting(localStage: activity?.localStage)?.label
    }

    var body: some View {
        AppCard(style: .print(.ink), padding: AppSpacing.md + 2) {
            VStack(alignment: .leading, spacing: AppSpacing.md) {
                SectionHeader(title: "分析进度", folio: "01")
                if let waiting {
                    Label(waiting, systemImage: MatchWaiting(localStage: activity?.localStage)?.symbol ?? "hourglass")
                        .font(AppFont.callout.weight(.semibold))
                        .foregroundStyle(Color.inkSoft)
                }
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(Self.steps.indices, id: \.self) { i in
                        row(i)
                    }
                }
            }
        }
        .accessibilityElement(children: .combine)
    }

    private func row(_ i: Int) -> some View {
        let state: RowState = current.map { i < $0 ? .done : (i == $0 ? .now : .later) } ?? .later
        return HStack(alignment: .top, spacing: AppSpacing.md) {
            VStack(spacing: 0) {
                marker(state)
                if i < Self.steps.count - 1 {
                    Rectangle()
                        .fill(state == .done ? Color.ink : Color.ink.opacity(0.2))
                        .frame(width: 2, height: 26)
                }
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(Self.steps[i].title)
                    .font(AppFont.numeral(20, relativeTo: .headline))
                    .foregroundStyle(state == .later ? Color.inkFaint : Color.ink)
                if state == .now {
                    Text(Self.steps[i].note)
                        .font(AppFont.caption)
                        .foregroundStyle(Color.inkSoft)
                        .transition(.opacity)
                }
            }
            .padding(.top, 1)
            Spacer(minLength: 0)
            if state == .done {
                Text("好了").instrumentLabel(AppFont.labelSmall, color: .pistachioText)
            }
        }
        .accessibilityLabel("\(Self.steps[i].title)，\(state == .done ? "已完成" : state == .now ? "进行中" : "未开始")")
    }

    private enum RowState { case done, now, later }

    @ViewBuilder
    private func marker(_ state: RowState) -> some View {
        switch state {
        case .done:
            Image(systemName: "checkmark")
                .font(.system(size: 11, weight: .black))
                .foregroundStyle(Color.ink)
                .frame(width: 24, height: 24)
                .background(Color.pistachio, in: Circle())
                .overlay(Circle().strokeBorder(Color.ink, lineWidth: AppStroke.fine))
        case .now:
            Circle()
                .fill(Color.tomato)
                .frame(width: 10, height: 10)
                .pulse(maxScale: 1.4, duration: 0.8)
                .frame(width: 24, height: 24)
                .background(Color.card, in: Circle())
                .overlay(Circle().strokeBorder(Color.ink, lineWidth: 2))
        case .later:
            Circle()
                .strokeBorder(Color.ink.opacity(0.35), style: StrokeStyle(lineWidth: 1.5, dash: [3, 3]))
                .frame(width: 24, height: 24)
        }
    }
}

// MARK: - While you wait

struct WhileYouWaitCard: View {
    let onDevice: Bool

    var body: some View {
        AppCard(padding: AppSpacing.md + 2) {
            HStack(alignment: .top, spacing: AppSpacing.md) {
                Image(systemName: "lightbulb.max")
                    .font(.system(size: 18, weight: .bold))
                    .foregroundStyle(Color.ink)
                    .frame(width: 36, height: 36)
                    .background(Color.card, in: Circle())
                    .overlay(Circle().strokeBorder(Color.ink, lineWidth: AppStroke.fine))
                    .rotationEffect(.degrees(-6))
                VStack(alignment: .leading, spacing: AppSpacing.xs) {
                    Text("等的时候").font(AppFont.title3).foregroundStyle(Color.ink)
                    Text(onDevice
                         ? "可以接着录下一场，分析会一场接一场排队。别锁屏、别切走 Shuttl，回来后会从断点接着算。"
                         : "可以接着录下一场。上传时让 Shuttl 留在前台，分析在服务器上进行。")
                        .font(AppFont.callout)
                        .foregroundStyle(Color.inkSoft)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }
}

// MARK: - Next time (failed)

struct ShootingTipsCard: View {
    private let tips: [(String, String)] = [
        ("camera.metering.center.weighted", "三脚架约 1.6 米高，底线后约 5 米，正对球场中线"),
        ("viewfinder", "开拍前看一眼：四个角都在画面里，球网横平"),
        ("sun.max.fill", "场馆灯都打开；别对着窗户或灯拍"),
        ("timer", "每段至少录完一整个回合，30 秒以上"),
    ]

    var body: some View {
        AppCard(style: .print(.ink), padding: AppSpacing.md + 2) {
            VStack(alignment: .leading, spacing: AppSpacing.md) {
                SectionHeader(title: "下次这样拍", folio: "01")
                ForEach(tips.indices, id: \.self) { i in
                    HStack(alignment: .top, spacing: AppSpacing.sm) {
                        Image(systemName: tips[i].0)
                            .font(.system(size: 13, weight: .bold))
                            .foregroundStyle(Color.ink)
                            .frame(width: 28, height: 28)
                            .background(Color.card, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                            .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .strokeBorder(Color.ink, lineWidth: AppStroke.fine))
                        Text(tips[i].1)
                            .font(AppFont.callout)
                            .foregroundStyle(Color.ink)
                            .fixedSize(horizontal: false, vertical: true)
                            .padding(.top, 4)
                    }
                    .accessibilityElement(children: .combine)
                }
            }
        }
    }
}
