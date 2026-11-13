import SwiftUI
import ShuttlCore
import ShuttlDesign

// MARK: - Landing map
// A to-scale court printed on pistachio stock. Landings are ink stamps: a cobalt
// dot for "in", a tomato dot crossed out for "out" (so in/out never relies on hue
// alone). Near end at the bottom, like holding the phone at the baseline.

struct ResultsCourtCanvas: View {
    let points: [ShotPoint]
    let selectedID: Int?
    let showLandingStatus: Bool
    let onTap: (ShotPoint) -> Void

    private let shape = RoundedRectangle(cornerRadius: AppRadius.sm, style: .continuous)

    var body: some View {
        VStack(spacing: AppSpacing.xs) {
            Text("远端").instrumentLabel(AppFont.labelSmall, color: .inkSoft)
            GeometryReader { proxy in
                let court = courtRect(in: proxy.size)
                ZStack {
                    Color.pistachio
                    Halftone(color: .ink.opacity(0.10), spacing: 8, maxDot: 2.6, focus: .center)
                    CourtPlan(.portrait).stroke(Color.card, lineWidth: 2)
                        .frame(width: court.width, height: court.height)
                    CourtNet(.portrait).stroke(Color.ink, style: StrokeStyle(lineWidth: 3.5, dash: [3, 2.5]))
                        .frame(width: court.width, height: court.height)
                    Canvas { context, _ in
                        for point in points where point.id != selectedID { stamp(point, in: court, context: &context) }
                        if let selected = points.first(where: { $0.id == selectedID }) {
                            stamp(selected, in: court, context: &context)
                        }
                    }
                }
                .contentShape(Rectangle())
                .gesture(SpatialTapGesture().onEnded { value in
                    guard let nearest = nearestPoint(value.location, in: court) else { return }
                    onTap(nearest)
                })
            }
            .aspectRatio(CourtGeometry2D.aspect(.portrait) * 1.08, contentMode: .fit)
            .clipShape(shape)
            .overlay(shape.strokeBorder(Color.ink, lineWidth: AppStroke.ink))
            .printShadow(shape, color: .ink, x: 4, y: 4)
            Text("近端 · 你在这边").instrumentLabel(AppFont.labelSmall, color: .inkSoft)
                .padding(.top, 4)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("球场落点图，共 \(points.count) 个落点，界内 \(points.filter(\.isIn).count) 个。近端在下方，远端在上方。")
    }

    /// The court drawn inside the card with a small run-off margin.
    private func courtRect(in size: CGSize) -> CGRect {
        size == .zero ? .zero : CGRect(origin: .zero, size: size).insetBy(dx: size.width * 0.04, dy: size.width * 0.04)
    }

    private func location(_ p: SIMD2<Double>, in court: CGRect) -> CGPoint {
        CourtGeometry2D.point(x: p.x, y: p.y, in: court, .portrait)
    }

    private func stamp(_ point: ShotPoint, in court: CGRect, context: inout GraphicsContext) {
        let c = location(point.point, in: court)
        let selected = point.id == selectedID
        let r: CGFloat = selected ? 9 : 6.5
        let dot = Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: r * 2, height: r * 2))
        if selected {
            let halo = Path(ellipseIn: CGRect(x: c.x - r - 5, y: c.y - r - 5, width: (r + 5) * 2, height: (r + 5) * 2))
            context.fill(halo, with: .color(Color.butter))
            context.stroke(halo, with: .color(Color.ink), lineWidth: 1.5)
        }
        let inCourt = point.isIn || !showLandingStatus
        context.fill(dot, with: .color(inCourt ? Color.cobalt : Color.tomato))
        context.stroke(dot, with: .color(Color.ink), lineWidth: 1.5)
        if !inCourt {
            var cross = Path()
            let k = r * 0.5
            cross.move(to: CGPoint(x: c.x - k, y: c.y - k)); cross.addLine(to: CGPoint(x: c.x + k, y: c.y + k))
            cross.move(to: CGPoint(x: c.x + k, y: c.y - k)); cross.addLine(to: CGPoint(x: c.x - k, y: c.y + k))
            context.stroke(cross, with: .color(Color.ink), style: StrokeStyle(lineWidth: 1.8, lineCap: .round))
        }
    }

    private func nearestPoint(_ tap: CGPoint, in court: CGRect) -> ShotPoint? {
        points.min { hypot(location($0.point, in: court).x - tap.x, location($0.point, in: court).y - tap.y)
            < hypot(location($1.point, in: court).x - tap.x, location($1.point, in: court).y - tap.y) }
    }
}

// MARK: - Movement heatmap
// One half-court (the player's own), printed as a riso halftone: every cell is a
// cluster of dots whose size is the time spent there. The net edge is marked.

struct ResultsHeatmapCanvas: View {
    let stats: MovementStats
    let side: CourtSide

    private let shape = RoundedRectangle(cornerRadius: AppRadius.sm, style: .continuous)

    var body: some View {
        GeometryReader { proxy in
            let size = proxy.size
            // A full portrait court twice the visible height; the visible half is the player's.
            let full = CGRect(x: size.width * 0.05, y: side == .far ? size.height * 0.05 : -size.height * 0.9,
                              width: size.width * 0.9, height: size.height * 1.9)
            ZStack {
                Color.card
                Canvas { context, _ in drawDots(context: &context, court: full) }
                CourtPlan(.portrait).stroke(Color.ink.opacity(0.75), lineWidth: 1.4)
                    .frame(width: full.width, height: full.height)
                    .position(x: full.midX, y: full.midY)
                CourtNet(.portrait).stroke(Color.ink, style: StrokeStyle(lineWidth: 4, dash: [3, 2.5]))
                    .frame(width: full.width, height: full.height)
                    .position(x: full.midX, y: full.midY)
            }
        }
        .aspectRatio(CourtGeometry2D.aspect(.portrait) * 2 * 0.95, contentMode: .fit)
        .clipShape(shape)
        .overlay(shape.strokeBorder(Color.ink, lineWidth: AppStroke.ink))
        .overlay(alignment: side == .far ? .bottom : .top) {
            Text("球网").instrumentLabel(AppFont.labelSmall, color: .ink)
                .padding(.horizontal, 6).padding(.vertical, 2)
                .background(Color.butter, in: Capsule())
                .overlay(Capsule().strokeBorder(Color.ink, lineWidth: 1))
                .offset(y: side == .far ? 9 : -9)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(heatmapLabel)
    }

    private var heatmapLabel: String {
        let occupied = stats.heatmap.flatMap { $0 }.filter { $0 > 0 }.count
        return "\(side.chineseName)移动热力图，已覆盖 \(occupied) 个区域，场地覆盖率 \(Int((stats.courtCoverage * 100).rounded()))%。"
    }

    /// Heatmap rows run from the net (row 0) to the baseline; columns across the doubles width.
    private func drawDots(context: inout GraphicsContext, court: CGRect) {
        let cells = stats.heatmap
        let maxValue = cells.flatMap { $0 }.max() ?? 0
        guard maxValue > 0 else { return }
        let rows = cells.count, cols = cells.first?.count ?? 0
        guard rows > 0, cols > 0 else { return }
        let L = CourtGeometry2D.halfLength, W = CourtGeometry2D.halfWidth
        let sign: Double = side == .far ? 1 : -1
        let sub = 3
        let ink = side.ink
        let cellH = court.height / 2 / CGFloat(rows)
        let pitch = min(court.width / CGFloat(cols), cellH) / CGFloat(sub)
        for row in 0..<rows {
            for col in 0..<min(cols, cells[row].count) {
                let v = cells[row][col] / maxValue
                guard v > 0.02 else { continue }
                let r = pitch * 0.52 * CGFloat(0.35 + 0.65 * v.squareRoot())
                for i in 0..<sub {
                    for j in 0..<sub {
                        let dist = (Double(row) + (Double(i) + 0.5) / Double(sub)) / Double(rows) * L
                        let across = W - (Double(col) + (Double(j) + 0.5) / Double(sub)) / Double(cols) * 2 * W
                        let c = CourtGeometry2D.point(x: sign * dist, y: across, in: court, .portrait)
                        context.fill(Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: r * 2, height: r * 2)),
                                     with: .color(ink))
                    }
                }
            }
        }
    }
}

// MARK: - Score progression
// Two stepped lines, like a hand-kept scoresheet: cobalt for near, pink for far.
// Each rally is a column; the final score sits on stickers at the right edge.

struct ScoreProgressionCanvas: View {
    let steps: [ScoreStep]

    var body: some View {
        Canvas { context, size in
            let maxScore = max(3, steps.map { max($0.near, $0.far) }.max() ?? 1)
            let inset: CGFloat = 8
            let plot = CGRect(x: inset, y: inset, width: size.width - inset * 2 - 46, height: size.height - inset * 2)
            let n = max(1, steps.count)
            func x(_ i: Int) -> CGFloat { plot.minX + CGFloat(i) / CGFloat(n) * plot.width }
            func y(_ s: Int) -> CGFloat { plot.maxY - CGFloat(s) / CGFloat(maxScore) * plot.height }

            var grid = Path()
            for s in stride(from: 0, through: maxScore, by: max(1, maxScore / 4)) {
                grid.move(to: CGPoint(x: plot.minX, y: y(s))); grid.addLine(to: CGPoint(x: plot.maxX, y: y(s)))
            }
            context.stroke(grid, with: .color(Color.rule), style: StrokeStyle(lineWidth: 1, dash: [2, 3]))

            func stepPath(_ value: (ScoreStep) -> Int) -> Path {
                var path = Path()
                path.move(to: CGPoint(x: x(0), y: y(0)))
                var last = 0
                for (i, step) in steps.enumerated() {
                    let v = value(step)
                    path.addLine(to: CGPoint(x: x(i), y: y(last)))
                    path.addLine(to: CGPoint(x: x(i), y: y(v)))
                    last = v
                }
                path.addLine(to: CGPoint(x: plot.maxX, y: y(last)))
                return path
            }
            let series: [(Path, Color)] = [(stepPath { $0.far }, Color.flamingo), (stepPath { $0.near }, Color.cobalt)]
            for (path, ink) in series {
                context.stroke(path, with: .color(Color.ink), style: StrokeStyle(lineWidth: 5.5, lineCap: .round, lineJoin: .round))
                context.stroke(path, with: .color(ink), style: StrokeStyle(lineWidth: 3.2, lineCap: .round, lineJoin: .round))
            }
            for (i, step) in steps.enumerated() where step.isUncertain {
                let p = CGPoint(x: x(i), y: plot.maxY + 2)
                context.fill(Path(ellipseIn: CGRect(x: p.x - 2, y: p.y - 2, width: 4, height: 4)), with: .color(Color.inkFaint))
            }
        }
        .overlay(alignment: .trailing) {
            if let last = steps.last {
                VStack(spacing: 6) {
                    finalTag("远", last.far, ink: .flamingo, text: .ink)
                    finalTag("近", last.near, ink: .cobalt, text: .onCobalt)
                }
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("比分走势，共 \(steps.count) 分，最终近端 \(steps.last?.near ?? 0) 比远端 \(steps.last?.far ?? 0)")
    }

    /// The final score for one side, labelled so the side never depends on colour alone.
    private func finalTag(_ side: String, _ value: Int, ink: Color, text: Color) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 2) {
            Text(side).font(.system(size: 10, weight: .heavy))
            Text("\(value)").font(DisplayFace.fixed(18))
        }
        .foregroundStyle(text)
        .padding(.horizontal, 7)
        .frame(minWidth: 40, minHeight: 26)
        .background(ink, in: Capsule())
        .overlay(Capsule().strokeBorder(Color.ink, lineWidth: 1.25))
        .fixedSize()
    }
}
