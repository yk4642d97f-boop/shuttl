import SwiftUI
import SwiftData

// MARK: - Match ticket
// Each match is a ticket: a spot-ink stub with the date, torn along a
// perforation from the body that carries the numbers.

private struct MatchCard: View {
    let match: Match
    var activity: PipelineActivity?

    private let stubWidth: CGFloat = 78

    var body: some View {
        let shape = Ticket(stubX: stubWidth, radius: AppRadius.md)
        HStack(spacing: 0) {
            dateStub
                .frame(width: stubWidth)
            VStack(alignment: .leading, spacing: AppSpacing.sm) {
                HStack(alignment: .top) {
                    Text("\(match.courtLabel) · \(match.durationText)")
                        .font(AppFont.title3)
                        .foregroundStyle(Color.ink)
                        .lineLimit(1)
                        .minimumScaleFactor(0.8)
                    Spacer(minLength: AppSpacing.sm)
                    if archived {
                        Label("已完成", systemImage: "checkmark")
                            .font(.system(size: 11, weight: .bold))
                            .foregroundStyle(Color.pistachioText)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .overlay(Capsule().strokeBorder(Color.pistachioText, lineWidth: 1))
                            .accessibilityLabel("已完成")
                    } else {
                        ShuttlBadge(status: match.status, tilt: 3,
                                    waiting: MatchWaiting(localStage: activity?.localStage))
                    }
                }
                detailLine
            }
            .padding(.leading, AppSpacing.md)
            .padding(.trailing, AppSpacing.md - 2)
            .padding(.vertical, AppSpacing.md - 2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .background {
            ZStack(alignment: .leading) {
                Color.card
                stubFill.frame(width: stubWidth)
            }
            .clipShape(shape)
        }
        .overlay {
            Path { p in
                p.move(to: CGPoint(x: stubWidth, y: 12))
                p.addLine(to: CGPoint(x: stubWidth, y: 400))
            }
            .stroke(Color.ink, style: StrokeStyle(lineWidth: 1.25, dash: [3, 4]))
            .mask(shape.inset(by: 10))
        }
        .overlay(shape.strokeBorder(Color.ink, lineWidth: 1.5))
        .printShadow(shape, color: .ink, y: 4)
        .accessibilityElement(children: .combine)
    }

    /// Colour means state: only matches that are still in motion (or need a retry) get an
    /// ink-filled stub; finished ones stay paper, so a long archive reads calm.
    private var archived: Bool { match.status == .done }
    private var stubFill: Color { archived ? Color.paperDeep : match.status.ink }

    private var dateStub: some View {
        VStack(spacing: 0) {
            Text(match.createdAt.formatted(.dateTime.month(.abbreviated)))
                .instrumentLabel(AppFont.labelSmall, color: .ink)
            Text("\(Calendar.current.component(.day, from: match.createdAt))")
                .font(AppFont.numeral(40, relativeTo: .title))
                .foregroundStyle(archived ? Color.cobalt : Color.ink)
                .monospacedDigit()
            Text(match.createdAt.formatted(.dateTime.hour().minute()))
                .font(AppFont.labelSmall)
                .foregroundStyle(Color.ink.opacity(0.8))
        }
        .padding(.vertical, AppSpacing.sm)
    }

    @ViewBuilder
    private var detailLine: some View {
        switch match.status {
        case .done:
            VStack(alignment: .leading, spacing: AppSpacing.sm) {
                HStack(spacing: AppSpacing.md) {
                    if match.lacksSpeeds {
                        // Same rule as the detail hero: no court, no speed; the shot count leads.
                        figure("\(match.allShots.count)", "击球")
                        figure("\(match.rallies.count)", "回合")
                        figure("—", "标定后才有球速", dim: true)
                    } else {
                        figure(match.maxSpeedText, "km/h 最高")
                        figure("\(match.rallies.count)", "回合")
                        figure("\(match.allShots.count)", "击球")
                    }
                }
                if !match.allShots.isEmpty {
                    ShotMixBar(shots: match.allShots, height: 4, hairline: true)
                }
            }
        case .analyzing, .uploading:
            VStack(alignment: .leading, spacing: AppSpacing.sm) {
                note(match.progressText(activity),
                     symbol: MatchWaiting(localStage: activity?.localStage)?.symbol ?? match.status.symbol)
                    .contentTransition(.numericText())
                if let fraction = match.progressFraction(activity) {
                    InkProgressBar(value: fraction, fill: match.status.ink, height: 8)
                }
            }
        case .recording:
            note("正在录制", symbol: "record.circle")
        case .failed:
            VStack(alignment: .leading, spacing: 2) {
                note(match.failureReason ?? "分析没有成功", symbol: "exclamationmark.triangle")
                    .lineLimit(2)
                Text("点开可重试")
                    .instrumentLabel(AppFont.labelSmall, color: .tomatoText)
            }
        }
    }

    private func figure(_ value: String, _ label: String, dim: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(value)
                .font(AppFont.numeral(24, relativeTo: .title3))
                .foregroundStyle(dim ? Color.inkFaint : Color.ink)
                .monospacedDigit()
            Text(label)
                .instrumentLabel(AppFont.labelSmall, color: .inkSoft)
        }
    }

    private func note(_ text: String, symbol: String) -> some View {
        Label(text, systemImage: symbol)
            .font(AppFont.callout.weight(.medium))
            .foregroundStyle(Color.inkSoft)
            .labelStyle(.titleAndIcon)
    }
}

// MARK: - Empty state

private struct EmptyMatchesView: View {
    var body: some View {
        VStack(spacing: AppSpacing.md) {
            ZStack {
                Circle().fill(Color.butter)
                    .overlay(Circle().strokeBorder(Color.ink, lineWidth: 1.5))
                    .frame(width: 150, height: 150)
                Circle().fill(Color.lilac)
                    .overlay(Circle().strokeBorder(Color.ink, lineWidth: 1.5))
                    .frame(width: 54, height: 54)
                    .offset(x: -66, y: 52)
                ShuttlecockMark(skirt: .card, cork: .tomato, ink: .ink)
                    .frame(width: 96, height: 96)
                    .rotationEffect(.degrees(-24))
            }
            .accessibilityHidden(true)
            .padding(.bottom, AppSpacing.sm)

            OverprintText("还没有比赛", font: AppFont.numeral(34, relativeTo: .title), plate: .cobalt,
                          offset: CGSize(width: 2.5, height: 2.5), blend: .multiply)
            Text("去「录制」拍下第一场，\nAI 会把每一拍的轨迹画出来。")
                .font(AppFont.body)
                .foregroundStyle(Color.inkSoft)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, AppSpacing.xxl)
    }
}

// MARK: - MatchListView

struct MatchListView: View {
    @Query(sort: \Match.createdAt, order: .reverse) private var matches: [Match]
    @Environment(\.modelContext) private var modelContext
    @Environment(MatchPipeline.self) private var pipeline: MatchPipeline?
    @State private var path = NavigationPath()
    @State private var deletionError: String?
    @State private var pendingDeletion: Match?

    private var active: [Match] { matches.filter { $0.status != .done && $0.status != .failed } }
    private var archived: [Match] { matches.filter { $0.status == .done || $0.status == .failed } }

    var body: some View {
        NavigationStack(path: $path) {
            ScrollView {
                VStack(alignment: .leading, spacing: AppSpacing.lg) {
                    PageHeader(kicker: "SHUTTL · 比赛档案", title: "比赛", ink: .cobalt) {
                        if !matches.isEmpty { countSeal }
                    }
                    if let heat = demoSystemItem(pipeline?.thermalChip, "有点热，这场分析完歇会儿", .mild) {
                        SystemChip(text: heat.text, symbol: "thermometer.medium", severity: heat.severity, spoken: heat.spoken)
                            .padding(.horizontal, AppSpacing.gutter)
                            .padding(.top, -AppSpacing.sm)
                    }

                    if let message = pipeline?.remoteSyncMessage {
                        VStack(alignment: .leading) {
                            Text(message).font(AppFont.caption)
                            Button("重试同步") { Task { await pipeline?.syncRemoteCommands() } }
                        }.padding(.horizontal, AppSpacing.gutter)
                    }
                    if matches.isEmpty {
                        EmptyMatchesView()
                    } else {
                        if !active.isEmpty { section("处理中", folio: "01", active) }
                        if !archived.isEmpty { section("已归档", folio: active.isEmpty ? "01" : "02", archived) }
                    }
                }
                .padding(.bottom, AppSpacing.xl)
            }
            .tabBarClearance()
            .paperPage()
            .toolbar(.hidden, for: .navigationBar)
            .navigationTitle("比赛")
            .navigationDestination(for: Match.self) { match in
                MatchDetailView(match: match)
            }
            .navigationDestination(for: ReplayRoute.self) { route in
                ReplayView(match: route.match, startFrame: route.startFrame)
            }
        }
        .onAppear(perform: openDemoMatchIfNeeded)
        .alert("没能删除比赛", isPresented: Binding(get: { deletionError != nil }, set: { if !$0 { deletionError = nil } })) {
            Button("知道了", role: .cancel) {}
        } message: { Text(deletionError ?? "") }
        .confirmationDialog("删除这场比赛？", isPresented: Binding(get: { pendingDeletion != nil }, set: { if !$0 { pendingDeletion = nil } })) {
            if let match = pendingDeletion { Button("删除这场", role: .destructive) { delete(match) } }
            Button("取消", role: .cancel) {}
        } message: {
            Text("删除录像和分析记录；已上传的比赛也会请求云端删除，离线时将在联网并登录原账号后同步。此操作不能恢复。")
        }
    }

    private func section(_ title: String, folio: String, _ items: [Match]) -> some View {
        VStack(alignment: .leading, spacing: AppSpacing.md) {
            SectionHeader(title: title, folio: folio)
            ForEach(items) { match in
                NavigationLink(value: match) {
                    MatchCard(match: match, activity: pipeline?.activity(for: match))
                }
                .buttonStyle(ScaleButtonStyle())
                .contextMenu {
                    if match.status == .failed, let pipeline {
                        Button {
                            pipeline.retry(match)
                        } label: {
                            Label("重试", systemImage: "arrow.clockwise")
                        }
                    }
                    Button(role: .destructive) {
                        pendingDeletion = match
                    } label: {
                        Label("删除", systemImage: "trash")
                    }
                }
            }
        }
        .padding(.horizontal, AppSpacing.gutter)
    }

    /// Rosette sticker with the match count.
    private var countSeal: some View {
        Seal(value: "\(matches.count)", caption: "场", fill: .butter, size: 78, tilt: 10)
            .padding(.bottom, AppSpacing.sm)
            .accessibilityLabel("共 \(matches.count) 场比赛")
    }

    private func delete(_ match: Match) {
        guard let pipeline else { return }
        Task {
            do {
                try await pipeline.discard(match)
                withAnimation(AppAnimation.spring) { modelContext.delete(match) }
                try modelContext.save()
            } catch { deletionError = MatchPipeline.failureReason(for: error) }
        }
    }

    private func openDemoMatchIfNeeded() {
        #if DEBUG
        let wanted = DemoMode.openStatus ?? .done
        guard path.isEmpty, DemoMode.opensMatch,
              let match = matches.first(where: {
                  $0.status == wanted && (!DemoMode.opensServerMatch || $0.analysisPath == nil)
              }) else { return }
        path.append(match)
        if DemoMode.opensReplay { path.append(ReplayRoute(match: match)) }
        #endif
    }
}
