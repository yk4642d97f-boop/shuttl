import SwiftUI

// MARK: - CourtPlan
// A to-scale badminton court (BWF) drawn as ink lines — the app's structural motif.
// Used big (landing maps, heatmaps), and faint as a watermark behind hero blocks,
// so every screen quietly sits on the geometry of a court.
//
// World frame matches ShuttlCore.CourtModel: origin at the centre under the net,
// +x toward the far baseline, +y to the left as seen from the near baseline.

public enum CourtGeometry2D {
    public static let halfLength = 6.70      // net → baseline
    public static let halfWidth = 3.05       // doubles
    public static let singlesHalfWidth = 2.59
    public static let shortServiceX = 1.98
    public static let longServiceDoublesX = 5.94

    public enum Orientation: Sendable {
        /// Near baseline at the bottom (phone-held portrait maps).
        case portrait
        /// Near baseline on the left (wide cards, video-like).
        case landscape
    }

    public static func aspect(_ o: Orientation) -> CGFloat {
        o == .portrait ? CGFloat(halfWidth / halfLength) : CGFloat(halfLength / halfWidth)
    }

    /// World metres (x along the court, y across) → a point in `rect`.
    public static func point(x: Double, y: Double, in rect: CGRect, _ o: Orientation) -> CGPoint {
        let u = (x + halfLength) / (2 * halfLength)      // 0 near baseline … 1 far baseline
        let v = (halfWidth - y) / (2 * halfWidth)        // 0 left sideline … 1 right
        switch o {
        case .portrait:  return CGPoint(x: rect.minX + v * rect.width, y: rect.maxY - u * rect.height)
        case .landscape: return CGPoint(x: rect.minX + u * rect.width, y: rect.minY + v * rect.height)
        }
    }
}

public struct CourtPlan: Shape {
    public var orientation: CourtGeometry2D.Orientation
    public var includeNet: Bool

    public init(_ orientation: CourtGeometry2D.Orientation = .portrait, includeNet: Bool = false) {
        self.orientation = orientation
        self.includeNet = includeNet
    }

    public func path(in rect: CGRect) -> Path {
        typealias G = CourtGeometry2D
        func p(_ x: Double, _ y: Double) -> CGPoint { G.point(x: x, y: y, in: rect, orientation) }
        var path = Path()
        func line(_ a: (Double, Double), _ b: (Double, Double)) {
            path.move(to: p(a.0, a.1)); path.addLine(to: p(b.0, b.1))
        }
        let L = G.halfLength, W = G.halfWidth, S = G.singlesHalfWidth
        // Outer boundary (doubles sidelines + baselines)
        path.addLines([p(-L, W), p(L, W), p(L, -W), p(-L, -W)])
        path.closeSubpath()
        // Singles sidelines
        line((-L, S), (L, S)); line((-L, -S), (L, -S))
        // Short service lines and doubles long service lines
        for sx in [G.shortServiceX, -G.shortServiceX, G.longServiceDoublesX, -G.longServiceDoublesX] {
            line((sx, W), (sx, -W))
        }
        // Centre lines, service line → baseline on each half
        line((G.shortServiceX, 0), (L, 0)); line((-G.shortServiceX, 0), (-L, 0))
        if includeNet { line((0, W + 0.3), (0, -W - 0.3)) }
        return path
    }
}

public struct CourtNet: Shape {
    public var orientation: CourtGeometry2D.Orientation
    public init(_ orientation: CourtGeometry2D.Orientation = .portrait) { self.orientation = orientation }

    public func path(in rect: CGRect) -> Path {
        let W = CourtGeometry2D.halfWidth + 0.35
        var path = Path()
        path.move(to: CourtGeometry2D.point(x: 0, y: W, in: rect, orientation))
        path.addLine(to: CourtGeometry2D.point(x: 0, y: -W, in: rect, orientation))
        return path
    }
}

// MARK: - CourtWatermark
// The court printed faint and oversized behind a hero block, rotated like a
// poster crop. Purely decorative.

public struct CourtWatermark: View {
    public var color: Color
    public var lineWidth: CGFloat
    public var angle: Angle

    public init(color: Color = .ink.opacity(0.16), lineWidth: CGFloat = 1.5, angle: Angle = .degrees(-14)) {
        self.color = color; self.lineWidth = lineWidth; self.angle = angle
    }

    public var body: some View {
        GeometryReader { geo in
            let h = geo.size.height * 1.9
            let w = h * CourtGeometry2D.aspect(.portrait)
            ZStack {
                CourtPlan(.portrait).stroke(color, lineWidth: lineWidth)
                CourtNet(.portrait).stroke(color, style: StrokeStyle(lineWidth: lineWidth * 2.2, dash: [2, 3]))
            }
            .frame(width: w, height: h)
            .rotationEffect(angle)
            .position(x: geo.size.width * 0.78, y: geo.size.height * 0.52)
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

#Preview("CourtPlan") {
    HStack(spacing: 20) {
        ZStack {
            Color.pistachio
            CourtPlan(.portrait).stroke(Color.card, lineWidth: 2)
            CourtNet(.portrait).stroke(Color.ink, style: StrokeStyle(lineWidth: 3, dash: [2, 2]))
        }
        .aspectRatio(CourtGeometry2D.aspect(.portrait), contentMode: .fit)
        .frame(height: 300)
        ZStack { Color.tomato; CourtWatermark() }
            .frame(width: 160, height: 120)
            .clipShape(RoundedRectangle(cornerRadius: 20))
    }
    .padding(24)
    .paperPage()
}
