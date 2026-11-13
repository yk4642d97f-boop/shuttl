import SwiftUI
import Charts
import SwiftData

// MARK: - Speed zones (shared by histogram and court map)

private enum SpeedZone: CaseIterable {
    case soft, mid, fast

    init(_ kmh: Float) {
        switch kmh {
        case ..<100: self = .soft
        case ..<200: self = .mid
        default:     self = .fast
        }
    }

    var ink: Color {
        switch self {
        case .soft: return .lilac
        case .mid:  return .ink   // cobalt means "near side" everywhere, so not here
        case .fast: return .tomato
        }
    }

    var label: String {
        switch self {
        case .soft: return "< 100"
        case .mid:  return "100–200"
        case .fast: return "> 200"
        }
    }
}

// MARK: - Hero: career top speed

private struct CareerHero: View {
    let shots: [ShotPoint]

    /// Uncalibrated matches have no speeds; they don't count toward the record.
    private var maxSpeed: Float { shots.filter(\.hasMeasuredSpeed).map(\.speedKmh).max() ?? 0 }
    private var smashes: Int { shots.filter { $0.shotType == .smash }.count }

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: AppRadius.lg, style: .continuous)
        ZStack(alignment: .topLeading) {
            CourtWatermark(color: .ink.opacity(0.14), angle: .degrees(16))
                .clipShape(shape)
            Halftone(color: .ink.opacity(0.16), spacing: 10, maxDot: 3.8, focus: .topTrailing)
                .clipShape(shape)
            VStack(alignment: .leading, spacing: 0) {
                Text("生涯最高球速")
                    .instrumentLabel(AppFont.label, color: .ink)
                HStack(alignment: .firstTextBaseline, spacing: AppSpacing.sm) {
                    OverprintText(maxSpeed > 0 ? String(format: "%.0f", maxSpeed) : "--", font: AppFont.hero,
                                  ink: .ink, plate: .cobalt, offset: CGSize(width: 5, height: 4))
                    Text("km/h")
                        .font(AppFont.unit)
                        .foregroundStyle(Color.ink)
                }
                Text("比一辆高铁出站还快一点")
                    .font(AppFont.callout.weight(.semibold))
                    .foregroundStyle(Color.ink)
            }
            .padding(AppSpacing.lg)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.pistachio, in: shape)
        .overlay(shape.strokeBorder(Color.ink, lineWidth: 1.5))
        .overlay(alignment: .bottomTrailing) { smashSeal.offset(x: -12, y: 28) }
        .padding(.bottom, AppSpacing.md)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("生涯最高球速 \(Int(maxSpeed)) 公里每小时，累计扣杀 \(smashes) 次")
    }

    private var smashSeal: some View {
        Seal(value: "\(smashes)", caption: "扣杀", fill: .butter, size: 84, tilt: -10, captionOnTop: true)
    }
}

// MARK: - Speed histogram

private struct SpeedHistogramCard: View {
    let allShots: [ShotPoint]
    var uncalibratedMatches = 0

    /// Only shots with a measured speed (from calibrated matches) go into the chart.
    private var shots: [ShotPoint] { allShots.filter(\.hasMeasuredSpeed) }

    private struct Bucket: Identifiable {
        let id: Int
        let lo: Int
        let count: Int
        var zone: SpeedZone { SpeedZone(Float(lo)) }
    }

    private var buckets: [Bucket] {
        stride(from: 0, through: 350, by: 25).compactMap { lo in
            let n = shots.filter { $0.speedKmh >= Float(lo) && $0.speedKmh < Float(lo + 25) }.count
            return n > 0 ? Bucket(id: lo, lo: lo, count: n) : nil
        }
    }

    private var average: Double {
        guard !shots.isEmpty else { return 0 }
        return Double(shots.map(\.speedKmh).reduce(0, +)) / Double(shots.count)
    }

    var body: some View {
        AppCard(padding: AppSpacing.md + 2) {
            VStack(alignment: .leading, spacing: AppSpacing.md) {
                SectionHeader(title: "球速分布", folio: "01")

                Chart {
                    ForEach(buckets) { b in
                        RectangleMark(
                            xStart: .value("球速", b.lo + 2),
                            xEnd: .value("球速", b.lo + 23),
                            yStart: .value("次数", 0),
                            yEnd: .value("次数", b.count)
                        )
                        .foregroundStyle(b.zone.ink)
                        .clipShape(UnevenRoundedRectangle(topLeadingRadius: 5, topTrailingRadius: 5))
                    }
                    RuleMark(x: .value("平均", average))
                        .foregroundStyle(Color.ink)
                        .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [4, 4]))
                        .annotation(position: .top, alignment: .leading, spacing: 4) {
                            Text("平均 \(Int(average))")
                                .font(.system(size: 11, weight: .bold))
                                .foregroundStyle(Color.ink)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.butter, in: Capsule())
                                .overlay(Capsule().strokeBorder(Color.ink, lineWidth: 1))
                        }
                }
                .chartXScale(domain: 0...360)
                .chartYScale(domain: 0...Double(max(4, buckets.map(\.count).max() ?? 0)) * 1.2)
                .chartXAxis {
                    AxisMarks(values: [0, 100, 200, 300]) { v in
                        AxisTick(stroke: StrokeStyle(lineWidth: 1)).foregroundStyle(Color.ink)
                        AxisValueLabel {
                            if let n = v.as(Int.self) { Text("\(n)") }
                        }
                        .font(AppFont.labelSmall)
                        .foregroundStyle(Color.inkSoft)
                    }
                }
                .chartYAxis {
                    AxisMarks(position: .trailing, values: .automatic(desiredCount: 3)) {
                        AxisGridLine(stroke: StrokeStyle(lineWidth: 1, dash: [2, 3]))
                            .foregroundStyle(Color.rule)
                        AxisValueLabel()
                            .font(AppFont.labelSmall)
                            .foregroundStyle(Color.inkSoft)
                    }
                }
                .frame(height: 170)
                .accessibilityLabel("球速分布柱状图，平均 \(Int(average)) 公里每小时")

                HStack(spacing: AppSpacing.md) {
                    ForEach(SpeedZone.allCases, id: \.self) { z in
                        HStack(spacing: 5) {
                            Circle().fill(z.ink)
                                .overlay(Circle().strokeBorder(Color.ink, lineWidth: 1))
                                .frame(width: 10, height: 10)
                            Text(z.label).font(AppFont.labelSmall).foregroundStyle(Color.inkSoft)
                        }
                    }
                    Spacer(minLength: 0)
                    Text("km/h").instrumentLabel(AppFont.labelSmall, color: .inkSoft)
                }
                .accessibilityHidden(true)

                if uncalibratedMatches > 0 {
                    Label("另有 \(uncalibratedMatches) 场没认出球场，不算球速", systemImage: "viewfinder")
                        .font(AppFont.caption)
                        .foregroundStyle(Color.inkSoft)
                }
            }
        }
    }
}

// MARK: - Shot type donut

private struct ShotTypeCard: View {
    let shots: [ShotPoint]

    private struct Entry: Identifiable {
        let id: ShotType
        let count: Int
    }

    private var entries: [Entry] {
        ShotType.allCases.compactMap { t in
            let n = shots.filter { $0.shotType == t }.count
            return n > 0 ? Entry(id: t, count: n) : nil
        }
    }

    private var favourite: Entry? {
        entries.filter { $0.id != .unknown }.max { $0.count < $1.count }
    }

    var body: some View {
        AppCard(padding: AppSpacing.md + 2) {
            VStack(alignment: .leading, spacing: AppSpacing.md) {
                SectionHeader(title: "击球类型", folio: "02")

                HStack(spacing: AppSpacing.lg) {
                    Chart(entries) { e in
                        SectorMark(
                            angle: .value("次数", e.count),
                            innerRadius: .ratio(0.56),
                            angularInset: 2.5
                        )
                        .foregroundStyle(e.id.ink)
                        .cornerRadius(4)
                    }
                    .chartLegend(.hidden)
                    .chartBackground { _ in
                        VStack(spacing: -2) {
                            Text("\(shots.count)")
                                .font(AppFont.numeral(32, relativeTo: .title))
                                .monospacedDigit()
                            Text("拍").font(.system(size: 11, weight: .bold))
                        }
                        .foregroundStyle(Color.ink)
                    }
                    .frame(width: 148, height: 148)
                    .accessibilityHidden(true)

                    VStack(alignment: .leading, spacing: 7) {
                        ForEach(entries) { e in
                            HStack(spacing: 7) {
                                ShotChip(e.id.glyphKind, size: 20)
                                Text(e.id.displayName)
                                    .font(AppFont.callout.weight(.semibold))
                                    .foregroundStyle(Color.ink)
                                Spacer(minLength: 4)
                                Text("\(Int((Double(e.count) / Double(max(1, shots.count)) * 100).rounded()))%")
                                    .font(AppFont.labelSmall)
                                    .foregroundStyle(Color.inkSoft)
                                    .monospacedDigit()
                            }
                            .accessibilityElement(children: .combine)
                        }
                    }
                }

                if let fav = favourite {
                    (Text("招牌动作：").foregroundColor(.inkSoft)
                     + Text(fav.id.displayName).foregroundColor(.ink).bold()
                     + Text("，占全部击球的 \(Int((Double(fav.count) / Double(max(1, shots.count)) * 100).rounded()))%").foregroundColor(.inkSoft))
                        .font(AppFont.callout)
                }
            }
        }
    }
}

// MARK: - Landing map on a to-scale court

private struct LandingZoneCard: View {
    let shots: [ShotPoint]

    /// Hitter positions in world metres (centre origin under the net, ShuttlCore.CourtModel frame).
    private var placed: [ShotPoint] { shots.filter { $0.xMeters != 0 || $0.yMeters != 0 } }

    var body: some View {
        AppCard(padding: AppSpacing.md + 2) {
            VStack(alignment: .leading, spacing: AppSpacing.md) {
                SectionHeader(title: "击球站位", folio: "03")
                Text("每个点是一次击球时你站的位置，越大越快。")
                    .font(AppFont.callout)
                    .foregroundStyle(Color.inkSoft)

                GeometryReader { geo in
                    let court = CGRect(origin: .zero, size: geo.size).insetBy(dx: 8, dy: 8)
                    ZStack {
                        Color.pistachio
                        CourtPlan(.landscape).stroke(Color.card, lineWidth: 1.8)
                            .frame(width: court.width, height: court.height)
                        CourtNet(.landscape).stroke(Color.ink, style: StrokeStyle(lineWidth: 3, dash: [2, 2]))
                            .frame(width: court.width, height: court.height)
                        Canvas { ctx, _ in drawShots(ctx: ctx, court: court) }
                    }
                }
                .aspectRatio(CourtGeometry2D.aspect(.landscape) * 0.97, contentMode: .fit)
                .clipShape(RoundedRectangle(cornerRadius: AppRadius.sm, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: AppRadius.sm, style: .continuous).strokeBorder(Color.ink, lineWidth: AppStroke.ink))
                .accessibilityLabel("击球站位图，共 \(placed.count) 次击球")

                HStack {
                    Text("← 近端").instrumentLabel(AppFont.labelSmall, color: .inkSoft)
                    Spacer()
                    Text("球网").instrumentLabel(AppFont.labelSmall, color: .inkSoft)
                    Spacer()
                    Text("远端 →").instrumentLabel(AppFont.labelSmall, color: .inkSoft)
                }
                .accessibilityHidden(true)
            }
        }
    }

    private func drawShots(ctx: GraphicsContext, court: CGRect) {
        for shot in placed.sorted(by: { $0.speedKmh < $1.speedKmh }) {
            let p = CourtGeometry2D.point(x: Double(shot.xMeters), y: Double(shot.yMeters), in: court, .landscape)
            let zone = SpeedZone(shot.speedKmh)
            let r: CGFloat = zone == .fast ? 6 : zone == .mid ? 4.5 : 3.5
            let dot = Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: r * 2, height: r * 2))
            ctx.fill(dot, with: .color(zone.ink))
            ctx.stroke(dot, with: .color(.ink), lineWidth: 1)
        }
    }
}

// MARK: - Empty state

private struct EmptyStatsView: View {
    var body: some View {
        VStack(spacing: AppSpacing.md) {
            HStack(alignment: .bottom, spacing: 8) {
                bar(.lilac, 54); bar(.cobalt, 92); bar(.tomato, 132); bar(.pistachio, 76)
            }
            .overlay(alignment: .topTrailing) {
                Scallop(lobes: 12, depth: 0.12)
                    .fill(Color.butter)
                    .overlay(Scallop(lobes: 12, depth: 0.12).strokeBorder(Color.ink, lineWidth: 1.5))
                    .frame(width: 46, height: 46)
                    .offset(x: 22, y: -26)
            }
            .accessibilityHidden(true)
            .padding(.bottom, AppSpacing.sm)

            OverprintText("数据还在路上", font: AppFont.numeral(34, relativeTo: .title), plate: .pistachio,
                          offset: CGSize(width: 2.5, height: 2.5), blend: .multiply)
            Text("完成一场比赛的分析后，\n球速、击球类型和落点都会出现在这里。")
                .font(AppFont.body)
                .foregroundStyle(Color.inkSoft)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, AppSpacing.xxl)
    }

    private func bar(_ c: Color, _ h: CGFloat) -> some View {
        UnevenRoundedRectangle(topLeadingRadius: 10, topTrailingRadius: 10, style: .continuous)
            .fill(c)
            .overlay(UnevenRoundedRectangle(topLeadingRadius: 10, topTrailingRadius: 10, style: .continuous)
                .strokeBorder(Color.ink, lineWidth: 1.5))
            .frame(width: 34, height: h)
    }
}

// MARK: - StatisticsView

struct StatisticsView: View {
    @Query private var allMatches: [Match]

    private var completed: [Match] { allMatches.filter { $0.status == .done } }
    private var shots: [ShotPoint] { completed.flatMap(\.allShots) }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: AppSpacing.lg) {
                    PageHeader(kicker: "SHUTTL · 数据年鉴", title: "统计", ink: .pistachio)

                    if completed.isEmpty {
                        EmptyStatsView()
                    } else {
                        Group {
                            CareerHero(shots: shots)
                            AppCard(padding: AppSpacing.md + 2) {
                                StatRow(stats: [
                                    .init("\(completed.count)", "场比赛"),
                                    .init("\(completed.reduce(0) { $0 + $1.rallies.count })", "回合"),
                                    .init("\(shots.count)", "击球"),
                                ])
                            }
                            SpeedHistogramCard(allShots: shots, uncalibratedMatches: completed.filter(\.lacksSpeeds).count)
                            ShotTypeCard(shots: shots)
                            LandingZoneCard(shots: shots)
                        }
                        .padding(.horizontal, AppSpacing.gutter)
                    }

                    // Colophon: a quiet line at the foot of the almanac, in every state.
                    NavigationLink(value: AboutRoute()) {
                        HStack(spacing: AppSpacing.sm) {
                            Image(systemName: "info.circle").font(.system(size: 13, weight: .bold))
                            Text("关于与致谢 · 开源许可")
                                .font(AppFont.callout.weight(.semibold))
                            Spacer(minLength: 0)
                            Image(systemName: "chevron.right").font(.system(size: 11, weight: .bold))
                        }
                        .foregroundStyle(Color.inkSoft)
                        .padding(.vertical, AppSpacing.sm)
                        .overlay(alignment: .top) { AppDivider() }
                    }
                    .buttonStyle(ScaleButtonStyle())
                    .padding(.horizontal, AppSpacing.gutter)
                }
                .padding(.bottom, AppSpacing.xl)
            }
            .tabBarClearance()
            .paperPage()
            #if DEBUG
            .demoScrollAnchor()
            #endif
            .toolbar(.hidden, for: .navigationBar)
            .navigationDestination(for: AboutRoute.self) { _ in AcknowledgementsView() }
            #if DEBUG
            .navigationDestination(isPresented: .constant(DemoMode.opensAbout)) { AcknowledgementsView() }
            #endif
        }
    }
}

struct AboutRoute: Hashable {}
