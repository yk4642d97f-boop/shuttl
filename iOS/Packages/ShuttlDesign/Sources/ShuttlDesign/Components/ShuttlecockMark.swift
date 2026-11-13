import SwiftUI

// MARK: - ShuttlecockMark
// Geometric shuttlecock: round cork, flared feather skirt with scalloped tips,
// ribs and a thread band. Drawn in a unit square so it scales to any size.

public struct ShuttlecockMark: View {
    public var skirt: Color
    public var cork: Color
    public var ink: Color
    public var band: Color?

    public init(skirt: Color, cork: Color, ink: Color, band: Color? = nil) {
        self.skirt = skirt; self.cork = cork; self.ink = ink; self.band = band
    }

    public var body: some View {
        Canvas { ctx, size in
            let s = min(size.width, size.height)
            let ox = (size.width - s) / 2, oy = (size.height - s) / 2
            func p(_ x: CGFloat, _ y: CGFloat) -> CGPoint { CGPoint(x: ox + x * s, y: oy + y * s) }
            let stroke = StrokeStyle(lineWidth: max(1, s * 0.035), lineCap: .round, lineJoin: .round)

            let tips = 6
            let topL: CGFloat = 0.12, topR: CGFloat = 0.88, topY: CGFloat = 0.12
            let baseL: CGFloat = 0.37, baseR: CGFloat = 0.63, baseY: CGFloat = 0.66

            var skirtPath = Path()
            skirtPath.move(to: p(baseL, baseY))
            skirtPath.addLine(to: p(topL, topY))
            let step = (topR - topL) / CGFloat(tips)
            for i in 0..<tips {
                let x0 = topL + CGFloat(i) * step
                skirtPath.addQuadCurve(to: p(x0 + step, topY), control: p(x0 + step / 2, topY - 0.09))
            }
            skirtPath.addLine(to: p(baseR, baseY))
            skirtPath.closeSubpath()
            ctx.fill(skirtPath, with: .color(skirt))

            var ribs = Path()
            for i in 1..<tips {
                let t = CGFloat(i) / CGFloat(tips)
                ribs.move(to: p(baseL + (baseR - baseL) * t, baseY))
                ribs.addLine(to: p(topL + (topR - topL) * t, topY))
            }
            ctx.stroke(ribs, with: .color(ink.opacity(0.55)), style: StrokeStyle(lineWidth: max(0.8, s * 0.018), lineCap: .round))

            let bandT: CGFloat = 0.42
            let bandY = baseY + (topY - baseY) * bandT
            var bandPath = Path()
            bandPath.move(to: p(baseL + (topL - baseL) * bandT, bandY))
            bandPath.addLine(to: p(baseR + (topR - baseR) * bandT, bandY))
            ctx.stroke(bandPath, with: .color(band ?? ink), style: StrokeStyle(lineWidth: max(1, s * 0.05), lineCap: .round))

            ctx.stroke(skirtPath, with: .color(ink), style: stroke)

            var corkPath = Path()
            corkPath.move(to: p(0.33, baseY))
            corkPath.addLine(to: p(0.67, baseY))
            corkPath.addArc(center: p(0.5, baseY), radius: 0.17 * s,
                            startAngle: .degrees(0), endAngle: .degrees(180), clockwise: false)
            corkPath.closeSubpath()
            ctx.fill(corkPath, with: .color(cork))
            ctx.stroke(corkPath, with: .color(ink), style: stroke)
        }
        .accessibilityHidden(true)
    }
}
