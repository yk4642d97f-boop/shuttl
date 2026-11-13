import SwiftUI
import ShuttlCore
import ShuttlDesign

public enum MatchResultTab: String, CaseIterable, Sendable {
    case overview, court, rallies, movement

    var title: String {
        switch self {
        case .overview: return "概览"
        case .court: return "落点"
        case .rallies: return "回合"
        case .movement: return "移动"
        }
    }
}

// MARK: - MatchResultView
// Stand-alone results page: masthead + every section.

public struct MatchResultView: View {
    public let viewModel: MatchResultViewModel
    public let onReplay: (Int) -> Void

    public init(viewModel: MatchResultViewModel, onReplay: @escaping (Int) -> Void = { _ in }) {
        self.viewModel = viewModel
        self.onReplay = onReplay
    }

    public var body: some View {
        ResultsScrollView {
            VStack(alignment: .leading, spacing: AppSpacing.lg) {
                VStack(alignment: .leading, spacing: AppSpacing.xs) {
                    Text("SHUTTL · 比赛档案").instrumentLabel(AppFont.labelSmall)
                    OverprintText("比赛结果", font: AppFont.numeral(52, relativeTo: .largeTitle),
                                  plate: .tomato, offset: CGSize(width: 3.5, height: 3), blend: .multiply)
                        .accessibilityAddTraits(.isHeader)
                    Text("从第一拍到最后一分，全部在这里。")
                        .font(AppFont.callout)
                        .foregroundStyle(Color.inkSoft)
                }
                .padding(.top, AppSpacing.md)
                MatchResultSections(viewModel: viewModel, onReplay: onReplay)
            }
            .padding(.horizontal, AppSpacing.gutter)
            .padding(.bottom, AppSpacing.xl)
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }
}

// MARK: - MatchResultSections
// The tab strip + section content, embeddable in a host page's scroll view.

public struct MatchResultSections: View {
    public let viewModel: MatchResultViewModel
    public let onReplay: (Int) -> Void
    public let showsHero: Bool
    /// The user's call for a rally, by its start frame. nil hides score correction.
    public let onCorrect: ((Int, RallyCall) -> Void)?
    private let initialCorrection: Int?
    private let initialDetail: Int?
    @State private var tab: MatchResultTab = .overview

    public init(viewModel: MatchResultViewModel, showsHero: Bool = true, initialTab: MatchResultTab = .overview,
                initialCorrection: Int? = nil, initialDetail: Int? = nil,
                onReplay: @escaping (Int) -> Void = { _ in },
                onCorrect: ((Int, RallyCall) -> Void)? = nil) {
        self.viewModel = viewModel
        self.showsHero = showsHero
        self.onReplay = onReplay
        self.onCorrect = onCorrect
        self.initialCorrection = initialCorrection
        self.initialDetail = initialDetail
        _tab = State(initialValue: initialTab)
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: AppSpacing.lg) {
            InkSegmented(MatchResultTab.allCases, selection: $tab) { $0.title }
                .accessibilityLabel("结果视图")
            Group {
                switch tab {
                case .overview: ResultsOverview(viewModel: viewModel, showsHero: showsHero)
                case .court: CourtDiagramView(viewModel: viewModel, onReplay: onReplay)
                case .rallies: RallyListView(viewModel: viewModel, initialCorrection: initialCorrection,
                                             initialDetail: initialDetail,
                                             onReplay: onReplay, onCorrect: onCorrect)
                case .movement: PlayerMovementView(viewModel: viewModel)
                }
            }
            .transition(.opacity)
        }
        .animation(AppAnimation.standard, value: tab)
    }
}

// MARK: - Overview

struct ResultsOverview: View {
    let viewModel: MatchResultViewModel
    let showsHero: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: AppSpacing.lg) {
            if showsHero { ResultsHero(summary: viewModel.summary) }
            ForEach(viewModel.notices) { ResultNoticeView(notice: $0) }
            numbersCard
            scoreCard
            shotMixCard
            if viewModel.rallies.isEmpty {
                EmptyResultsView(title: "还没有回合", message: "分析没有找到完整回合。确认视频里有完整球场后再试一次。")
            }
        }
    }

    private var numbersCard: some View {
        AppCard(padding: AppSpacing.md + 2) {
            VStack(alignment: .leading, spacing: AppSpacing.md) {
                SectionHeader(title: "本场数字", folio: "01")
                // The fastest smash is already the hero; this row is about how the match was played.
                StatRow(stats: [
                    .init("\(viewModel.summary.totalRallies)", "回合"),
                    .init("\(viewModel.summary.totalShots)", "击球"),
                    .init(String(format: "%.1f", viewModel.summary.averageRallyShots), "平均拍数"),
                    .init("\(longestRally)", "最长", unit: "拍", hero: true, ink: .butter),
                ], numeralFont: AppFont.numeral(28, relativeTo: .title2))
            }
        }
    }

    private var scoreCard: some View {
        AppCard(padding: AppSpacing.md + 2) {
            VStack(alignment: .leading, spacing: AppSpacing.md) {
                SectionHeader(title: "比分走势", folio: "02")
                if viewModel.scoreProgression.isEmpty {
                    Text("还没有比分记录").font(AppFont.callout).foregroundStyle(Color.inkSoft)
                } else {
                    scoreboard
                    ScoreProgressionCanvas(steps: viewModel.scoreProgression)
                        .frame(height: 132)
                }
            }
        }
    }

    private var scoreboard: some View {
        let near = viewModel.summary.pointsWon[.near, default: 0]
        let far = viewModel.summary.pointsWon[.far, default: 0]
        return HStack(alignment: .center, spacing: AppSpacing.sm) {
            Sticker("近端", fill: .cobalt, foreground: .onCobalt, tilt: -4)
            OverprintText("\(near)", font: AppFont.numeral(54, relativeTo: .largeTitle), plate: .cobalt,
                          offset: CGSize(width: 3, height: 3), blend: .multiply)
            Text(":").font(AppFont.numeral(40, relativeTo: .title)).foregroundStyle(Color.inkFaint)
            OverprintText("\(far)", font: AppFont.numeral(54, relativeTo: .largeTitle), plate: .flamingo,
                          offset: CGSize(width: 3, height: 3), blend: .multiply)
            Sticker("远端", fill: .flamingo, tilt: 4)
            Spacer(minLength: 0)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("得分，近端 \(near)，远端 \(far)")
    }

    private var shotMixCard: some View {
        AppCard(padding: AppSpacing.md + 2) {
            VStack(alignment: .leading, spacing: AppSpacing.md) {
                SectionHeader(title: "击球构成", folio: "03")
                if viewModel.summary.totalShots == 0 {
                    Text("暂无击球类型").font(AppFont.callout).foregroundStyle(Color.inkSoft)
                } else {
                    let total = max(1, viewModel.summary.totalShots)
                    let entries = ShotType.allCases
                        .map { ($0, viewModel.summary.shotTypeCounts[$0, default: 0]) }
                        .filter { $0.1 > 0 }
                        .sorted { $0.1 > $1.1 }
                    ForEach(entries, id: \.0) { type, count in
                        HStack(spacing: AppSpacing.sm) {
                            ShotChip(type.glyphKind, size: 28, style: .paper)
                            Text(type.chineseName)
                                .font(AppFont.numeral(18, relativeTo: .headline))
                                .foregroundStyle(Color.ink)
                                .frame(width: 44, alignment: .leading)
                            InkProgressBar(value: Double(count) / Double(total), fill: type.ink, height: 12)
                            Text("\(count)")
                                .font(AppFont.numeral(18, relativeTo: .headline))
                                .monospacedDigit()
                                .foregroundStyle(Color.ink)
                                .frame(width: 30, alignment: .trailing)
                        }
                        .accessibilityElement(children: .ignore)
                        .accessibilityLabel("\(type.chineseName) \(count) 次")
                    }
                }
            }
        }
    }

    private var longestRally: Int { viewModel.rallies.map(\.shots.count).max() ?? 0 }
}

// MARK: - Hero

struct ResultsHero: View {
    let summary: MatchSummary

    var body: some View {
        let smash = summary.maxSmashKmh
        let speed = smash ?? summary.maxSpeedKmh
        let title = smash != nil ? "本场最快扣杀" : (speed == nil ? "分析完成 · 本场击球" : "本场最快球速")
        let value = speed.map { String(Int($0.rounded())) } ?? "\(summary.totalShots)"
        let unit = speed == nil ? "拍" : "km/h"
        let note: String = speed == nil ? "没有可靠的球场标定，球速和落点暂不显示。"
            : smash == nil ? "本场没有识别到扣杀，显示所有击球里最快的一拍。"
            : "来自手机端的三维轨迹拟合。"
        let shape = RoundedRectangle(cornerRadius: AppRadius.lg, style: .continuous)
        return VStack(alignment: .leading, spacing: 0) {
            Text(title).instrumentLabel(AppFont.label, color: .ink)
            HStack(alignment: .firstTextBaseline, spacing: AppSpacing.sm) {
                OverprintText(value, font: AppFont.hero, plate: .butter, offset: CGSize(width: 5, height: 4))
                Text(unit).font(AppFont.unit).foregroundStyle(Color.ink)
            }
            Text(note)
                .font(AppFont.callout.weight(.semibold))
                .foregroundStyle(Color.ink)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(AppSpacing.lg)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            ZStack {
                Color.tomato
                CourtWatermark(color: .ink.opacity(0.14))
                Halftone(color: .ink.opacity(0.18), spacing: 9, maxDot: 3.4, focus: .bottomTrailing)
            }
            .clipShape(shape)
        }
        .overlay(shape.strokeBorder(Color.ink, lineWidth: AppStroke.ink))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(title) \(value) \(unit == "km/h" ? "公里每小时" : unit)")
    }
}

// MARK: - Court (landings)

public struct CourtDiagramView: View {
    public let viewModel: MatchResultViewModel
    public let onReplay: (Int) -> Void
    @State private var side: ResultsSideFilter = .all
    @State private var shotType: ShotType?
    @State private var selectedPoint: ShotPoint?

    public init(viewModel: MatchResultViewModel, onReplay: @escaping (Int) -> Void = { _ in }) {
        self.viewModel = viewModel
        self.onReplay = onReplay
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: AppSpacing.md) {
            SectionHeader(title: "落点地图", folio: "01")
            Text("每个印章是一个回合的最后落点。点一下，看这一分是怎么打的。")
                .font(AppFont.callout)
                .foregroundStyle(Color.inkSoft)
                .fixedSize(horizontal: false, vertical: true)
            if let caution = viewModel.cautionNotice { ResultNoticeView(notice: caution) }
            InkSegmented(ResultsSideFilter.allCases, selection: $side, ink: .butter, onInk: .ink) { $0.chineseName }
                .accessibilityLabel("落点一侧")
            shotTypeFilter
            if filteredPoints.isEmpty {
                EmptyResultsView(title: "没有可显示的落点",
                                 message: viewModel.isCalibrated ? "当前筛选没有对应的击球。" : "没有球场标定，无法显示米制落点。")
            } else {
                ResultsCourtCanvas(points: filteredPoints, selectedID: selectedPoint?.id, showLandingStatus: true) {
                    selectedPoint = $0
                }
                .frame(maxWidth: 290)
                .frame(maxWidth: .infinity)
                legend
            }
        }
        .sheet(item: $selectedPoint) { point in
            if let rally = viewModel.rally(at: point.rallyIndex) {
                RallyDetailView(rally: rally, focusShotID: point.id) { frame in
                    selectedPoint = nil
                    onReplay(frame)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }

    private var shotTypeFilter: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: AppSpacing.xs) {
                ResultsFilterChip(title: "全部类型", selected: shotType == nil) { shotType = nil }
                ForEach(ShotType.allCases.filter { $0 != .unknown }, id: \.self) { type in
                    ResultsFilterChip(title: type.chineseName, kind: type.glyphKind, selected: shotType == type) {
                        shotType = shotType == type ? nil : type
                    }
                }
            }
            .padding(.vertical, 2)
        }
        .accessibilityLabel("击球类型")
    }

    private var filteredPoints: [ShotPoint] {
        viewModel.filteredLandingPoints(side: side, shotType: shotType)
    }

    private var legend: some View {
        HStack(spacing: AppSpacing.lg) {
            legendItem(inCourt: true)
            legendItem(inCourt: false)
            Spacer(minLength: 0)
            Text("\(filteredPoints.count) 个落点").instrumentLabel(AppFont.labelSmall)
        }
        .accessibilityElement(children: .combine)
    }

    private func legendItem(inCourt: Bool) -> some View {
        HStack(spacing: 6) {
            ZStack {
                Circle().fill(inCourt ? Color.cobalt : Color.tomato)
                Circle().strokeBorder(Color.ink, lineWidth: 1.25)
                if !inCourt { Image(systemName: "xmark").font(.system(size: 7, weight: .black)).foregroundStyle(Color.ink) }
            }
            .frame(width: 14, height: 14)
            Text(inCourt ? "界内" : "出界").font(AppFont.callout.weight(.semibold)).foregroundStyle(Color.ink)
        }
    }
}

// MARK: - Rallies

public struct RallyListView: View {
    public let viewModel: MatchResultViewModel
    public let onReplay: (Int) -> Void
    public let onCorrect: ((Int, RallyCall) -> Void)?
    @State private var selectedRally: RallySummary?
    @State private var correcting: RallySummary?
    private let initialCorrection: Int?
    private let initialDetail: Int?

    public init(viewModel: MatchResultViewModel, initialCorrection: Int? = nil, initialDetail: Int? = nil,
                onReplay: @escaping (Int) -> Void = { _ in },
                onCorrect: ((Int, RallyCall) -> Void)? = nil) {
        self.viewModel = viewModel
        self.initialCorrection = initialCorrection
        self.initialDetail = initialDetail
        self.onReplay = onReplay
        self.onCorrect = onCorrect
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: AppSpacing.md) {
            SectionHeader(title: "回合记分卡", folio: "01")
            if onCorrect != nil && !viewModel.rallies.isEmpty {
                HStack(spacing: 6) {
                    Image(systemName: "hand.tap.fill").font(.system(size: 12, weight: .bold))
                    Text(uncertainCount > 0 ? "有 \(uncertainCount) 分 AI 不太确定 · 长按任一行改判" : "判错了？长按那一行改判")
                }
                .font(AppFont.callout)
                .foregroundStyle(Color.inkSoft)
            }
            if viewModel.rallies.isEmpty {
                EmptyResultsView(title: "没有识别到回合", message: "当前视频没有足够信号形成可查看的回合。")
            } else {
                AppCard(padding: AppSpacing.md) {
                    VStack(spacing: 0) {
                        ForEach(viewModel.rallies) { rally in
                            Button { selectedRally = rally } label: {
                                RallyListRow(rally: rally, correctable: onCorrect != nil) { correcting = rally }
                            }
                            .buttonStyle(ScaleButtonStyle())
                            .simultaneousGesture(LongPressGesture(minimumDuration: 0.45).onEnded { _ in
                                guard onCorrect != nil else { return }
                                correcting = rally
                            })
                            .modifier(RallyCallActions(rally: rally, onCorrect: onCorrect) { correcting = rally })
                            if rally.id != viewModel.rallies.last?.id { AppDivider() }
                        }
                    }
                }
            }
        }
        .sheet(item: $selectedRally) { rally in
            RallyDetailView(rally: rally, onReplay: { frame in
                selectedRally = nil
                onReplay(frame)
            }, onCorrect: onCorrect == nil ? nil : {
                selectedRally = nil
                Task { @MainActor in
                    try? await Task.sleep(for: .milliseconds(420))
                    correcting = rally
                }
            })
        }
        .sheet(item: $correcting) { rally in
            RallyCorrectionSheet(rally: rally) { call in
                correcting = nil
                onCorrect?(rally.startFrame, call)
            }
        }
        .sensoryFeedback(.impact(weight: .medium), trigger: correcting?.id)
        .onAppear {
            if let i = initialCorrection, correcting == nil, let r = viewModel.rallies.first(where: { $0.index == i }) {
                correcting = r
            } else if let i = initialDetail, selectedRally == nil, let r = viewModel.rallies.first(where: { $0.index == i }) {
                selectedRally = r
            }
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }

    private var uncertainCount: Int { viewModel.rallies.filter(\.isUncertain).count }
}

private struct RallyListRow: View {
    let rally: RallySummary
    var correctable = false
    var onStatusTap: () -> Void = {}

    var body: some View {
        HStack(spacing: AppSpacing.sm + 4) {
            Text(String(format: "%02d", rally.index))
                .font(AppFont.numeral(28, relativeTo: .title2))
                .monospacedDigit()
                .foregroundStyle(Color.ink)
                .frame(width: 40, alignment: .leading)
            VStack(alignment: .leading, spacing: 6) {
                // As many beads as the row has room for; the row never grows past the card.
                ViewThatFits(in: .horizontal) {
                    beads(8)
                    beads(6)
                    beads(4)
                }
                Text("\(rally.shots.count) 拍 · \(rally.endReason.chineseName)")
                    .font(AppFont.caption)
                    .foregroundStyle(Color.inkSoft)
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
            }
            .layoutPriority(-1)
            Spacer(minLength: 0)
            // Status next to the call it qualifies, so every row keeps one height.
            if correctable { statusChip }
            Sticker(rally.winner == .unknown ? "未判定" : "\(rally.winner.chineseName)得分",
                    fill: rally.winner.ink, foreground: rally.winner == .unknown ? .inkSoft : rally.winner.onInk, tilt: 2)
                .fixedSize()
            Image(systemName: "chevron.right").font(.system(size: 12, weight: .bold)).foregroundStyle(Color.inkFaint)
        }
        .padding(.vertical, AppSpacing.sm + 2)
        .contentShape(Rectangle())
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("回合 \(rally.index)，\(rally.shots.count) 拍，\(rally.endReason.chineseName)，\(rally.winner.chineseName)得分"
                            + (rally.isUserCorrected ? "，已改" : rally.isUncertain ? "，AI 不确定" : ""))
        .accessibilityAddTraits(.isButton)
    }

    private func beads(_ limit: Int) -> some View {
        HStack(spacing: 3) {
            ForEach(rally.shots.prefix(limit)) { shot in
                Circle().fill(shot.shotType.ink).frame(width: 11, height: 11)
                    .overlay(Circle().strokeBorder(Color.ink, lineWidth: 1))
            }
            if rally.shots.count > limit {
                Text("+\(rally.shots.count - limit)").font(AppFont.labelSmall).foregroundStyle(Color.inkSoft).fixedSize()
            }
        }
        .fixedSize()
    }

    @ViewBuilder
    private var statusChip: some View {
        if rally.isUserCorrected {
            chip("已改", fill: .ink, text: .paper)
        } else if rally.isUncertain {
            chip("不确定", fill: .butter, text: .ink).onTapGesture(perform: onStatusTap)
        }
    }

    private func chip(_ text: String, fill: Color, text textColor: Color) -> some View {
        Text(text)
            .font(.system(size: 10, weight: .heavy))
            .foregroundStyle(textColor)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(fill, in: Capsule())
            .overlay(Capsule().strokeBorder(Color.ink, lineWidth: 1))
            .rotationEffect(.degrees(-3))
            .fixedSize()
    }
}

// MARK: - Rally detail (sheet)

public struct RallyDetailView: View {
    public let rally: RallySummary
    public let focusShotID: Int?
    public let onReplay: (Int) -> Void
    /// Opens the score-correction slip for this rally; nil hides the button.
    public let onCorrect: (() -> Void)?

    public init(rally: RallySummary, focusShotID: Int? = nil, onReplay: @escaping (Int) -> Void = { _ in },
                onCorrect: (() -> Void)? = nil) {
        self.rally = rally
        self.focusShotID = focusShotID
        self.onReplay = onReplay
        self.onCorrect = onCorrect
    }

    public var body: some View {
        #if os(iOS)
        ScrollViewReader { reader in
            ScrollView { detailContent }
                .paperPage()
                .presentationDragIndicator(.visible)
                .onAppear {
                    if let focusShotID { reader.scrollTo(focusShotID, anchor: .center) }
                }
        }
        #else
        detailContent
        #endif
    }

    private var detailContent: some View {
        VStack(alignment: .leading, spacing: AppSpacing.lg) {
            VStack(alignment: .leading, spacing: AppSpacing.xs) {
                Text("回合 \(String(format: "%02d", rally.index)) · \(rally.server.chineseName)发球")
                    .instrumentLabel(AppFont.labelSmall)
                HStack(alignment: .firstTextBaseline, spacing: AppSpacing.md) {
                    OverprintText("\(rally.shots.count) 拍", font: AppFont.numeral(56, relativeTo: .largeTitle),
                                  plate: rally.winner == .unknown ? .lilac : rally.winner.ink,
                                  offset: CGSize(width: 3.5, height: 3), blend: .multiply)
                    Sticker("\(rally.endReason.chineseName) · \(rally.winner.chineseName)得分",
                            fill: rally.winner.ink, foreground: rally.winner == .unknown ? .inkSoft : rally.winner.onInk, tilt: -3)
                }
            }
            .accessibilityElement(children: .combine)
            .accessibilityAddTraits(.isHeader)

            timeline

            AppCard(padding: AppSpacing.md + 2) {
                VStack(alignment: .leading, spacing: AppSpacing.sm) {
                    SectionHeader(title: "这一分", folio: "02")
                    infoRow("结束方式", rally.endReason.chineseName)
                    infoRow("得分方", rally.winner.chineseName)
                    infoRow("发球方", rally.server.chineseName)
                    if rally.serveMissed { infoRow("提示", "第一拍之前已有飞行轨迹") }
                }
            }
            PillButton(title: "回放这一分", icon: "play.fill") { onReplay(rally.startFrame) }
                .accessibilityHint("从第 \(rally.startFrame) 帧开始")
            if let onCorrect {
                PillButton(title: rally.isUserCorrected ? "再改一次" : "判错了？改判这一分", icon: "pencil",
                           style: .secondary, action: onCorrect)
            }
        }
        .padding(AppSpacing.gutter)
        .padding(.top, AppSpacing.md)
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }

    private var timeline: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionHeader(title: "击球时间线", folio: "01")
                .padding(.bottom, AppSpacing.md)
            if rally.shots.isEmpty {
                Text("没有可显示的击球").font(AppFont.callout).foregroundStyle(Color.inkSoft)
            } else {
                ForEach(rally.shots) { shot in
                    HStack(alignment: .top, spacing: AppSpacing.md) {
                        VStack(spacing: 0) {
                            ShotChip(shot.shotType.glyphKind, size: 34)
                            if shot.id != rally.shots.last?.id {
                                Rectangle().fill(Color.ink).frame(width: 1.5).frame(minHeight: 30, maxHeight: .infinity)
                            }
                        }
                        .frame(width: 34)
                        VStack(alignment: .leading, spacing: 4) {
                            HStack(alignment: .firstTextBaseline, spacing: AppSpacing.sm) {
                                Text(shot.shotType.chineseName)
                                    .font(AppFont.numeral(22, relativeTo: .title3))
                                    .foregroundStyle(Color.ink)
                                Text("第 \(shot.index) 拍 · \(shot.side.chineseName)")
                                    .instrumentLabel(AppFont.labelSmall)
                                Spacer(minLength: 0)
                            }
                            HStack(spacing: AppSpacing.lg) {
                                metric(shot.speedKmh.map { String(Int($0.rounded())) }, unit: "km/h", label: "球速")
                                metric(shot.heightM.map { String(format: "%.1f", $0) }, unit: "m", label: "击球高度")
                            }
                        }
                        .padding(.vertical, 6)
                        .padding(.horizontal, shot.id == focusShotID ? AppSpacing.sm : 0)
                        .background {
                            if shot.id == focusShotID {
                                RoundedRectangle(cornerRadius: AppRadius.sm, style: .continuous).fill(Color.butter)
                                    .overlay(RoundedRectangle(cornerRadius: AppRadius.sm, style: .continuous)
                                        .strokeBorder(Color.ink, lineWidth: AppStroke.fine))
                            }
                        }
                        .padding(.bottom, AppSpacing.sm)
                    }
                    .id(shot.id)
                    .accessibilityElement(children: .combine)
                }
            }
        }
    }

    private func metric(_ value: String?, unit: String, label: String) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .firstTextBaseline, spacing: 2) {
                Text(value ?? "--").font(AppFont.numeral(20, relativeTo: .headline)).monospacedDigit()
                if value != nil { Text(unit).font(AppFont.labelSmall) }
            }
            .foregroundStyle(Color.ink)
            Text(value == nil ? "\(label)不可用" : label).instrumentLabel(AppFont.labelSmall)
        }
    }

    private func infoRow(_ title: String, _ value: String) -> some View {
        HStack {
            Text(title).font(AppFont.callout).foregroundStyle(Color.inkSoft)
            Spacer()
            Text(value).font(AppFont.headline).foregroundStyle(Color.ink)
        }
    }
}

// MARK: - Movement

public struct PlayerMovementView: View {
    public let viewModel: MatchResultViewModel
    @State private var side: CourtSide = .near

    public init(viewModel: MatchResultViewModel) {
        self.viewModel = viewModel
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: AppSpacing.md) {
            SectionHeader(title: "场上移动", folio: "01")
            if !viewModel.isCalibrated {
                EmptyResultsView(title: "移动数据不可用", message: "没有球场标定，不能可靠地计算距离、速度或热力图。")
            } else {
                InkSegmented([CourtSide.near, CourtSide.far], selection: $side,
                             ink: side.ink, onInk: side.onInk) { $0.chineseName }
                    .accessibilityLabel("查看一侧")
                let stats = viewModel.movement[side] ?? MovementStats()
                Text("点越大，停留越久。覆盖了半场的 \(Int((stats.courtCoverage * 100).rounded()))%。")
                    .font(AppFont.callout)
                    .foregroundStyle(Color.inkSoft)
                ResultsHeatmapCanvas(stats: stats, side: side)
                    .frame(maxWidth: 320)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, AppSpacing.sm)
                movementStats(stats)
            }
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }

    private func movementStats(_ stats: MovementStats) -> some View {
        AppCard(padding: AppSpacing.md + 2) {
            StatRow(stats: [
                .init(String(format: "%.1f", stats.distanceM), "移动距离", unit: "m"),
                .init(String(format: "%.1f", stats.averageSpeedMS), "平均速度", unit: "m/s"),
                .init(String(format: "%.1f", stats.maxSpeedMS), "最高速度", unit: "m/s"),
            ], numeralFont: AppFont.numeral(28, relativeTo: .title2))
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(side.chineseName)移动距离 \(String(format: "%.1f", stats.distanceM))米，平均速度 \(String(format: "%.1f", stats.averageSpeedMS))米每秒，最高速度 \(String(format: "%.1f", stats.maxSpeedMS))米每秒")
    }
}
