import SwiftUI

// MARK: - ShotGlyph
// Pictograms for shot types, in the spirit of Otl Aicher's Munich '72 system:
// every glyph is the same grammar — floor, net post, one flight path — so a smash
// (steep and straight) and a clear (a tall arc) are told apart by shape alone,
// not only by colour. That keeps shot charts readable in greyscale and for
// colour-blind players.

public enum ShotGlyphKind: String, CaseIterable, Sendable {
    case smash, clear, drop, net, drive, serve, lift, push, kill, block, unknown

    /// Accepts both the app's and ShuttlCore's raw values.
    public init(raw: String) { self = ShotGlyphKind(rawValue: raw) ?? .unknown }

    /// The ink a shot type is charted in (fine types share their family's ink).
    public var ink: Color {
        switch self {
        case .smash, .kill:  return .tomato
        case .clear, .lift:  return .cobalt
        case .drop:          return .lilac
        case .net, .block:   return .pistachio
        case .drive, .push:  return .butter
        case .serve:         return .flamingo
        case .unknown:       return .inkFaint
        }
    }

    public var chineseName: String {
        switch self {
        case .smash: return "扣杀"
        case .clear: return "高远"
        case .drop:  return "吊球"
        case .net:   return "网前"
        case .drive: return "平抽"
        case .serve: return "发球"
        case .lift:  return "挑球"
        case .push:  return "推球"
        case .kill:  return "扑球"
        case .block: return "挡网"
        case .unknown: return "其他"
        }
    }

    /// Start, control and end of the flight in a unit square (y down, net at x = 0.5).
    var flight: (CGPoint, CGPoint, CGPoint)? {
        func p(_ x: CGFloat, _ y: CGFloat) -> CGPoint { CGPoint(x: x, y: y) }
        switch self {
        case .smash:  return (p(0.12, 0.14), p(0.45, 0.36), p(0.88, 0.86))
        case .kill:   return (p(0.34, 0.20), p(0.50, 0.40), p(0.70, 0.86))
        case .clear:  return (p(0.12, 0.44), p(0.50, -0.34), p(0.90, 0.60))
        case .lift:   return (p(0.30, 0.82), p(0.60, -0.30), p(0.92, 0.58))
        case .drop:   return (p(0.10, 0.20), p(0.46, 0.14), p(0.66, 0.86))
        case .net:    return (p(0.34, 0.66), p(0.50, 0.14), p(0.66, 0.66))
        case .block:  return (p(0.30, 0.52), p(0.46, 0.30), p(0.62, 0.84))
        case .drive:  return (p(0.10, 0.44), p(0.50, 0.36), p(0.90, 0.48))
        case .push:   return (p(0.30, 0.48), p(0.52, 0.34), p(0.84, 0.72))
        case .serve:  return (p(0.12, 0.76), p(0.46, 0.00), p(0.74, 0.70))
        case .unknown: return nil
        }
    }
}

public struct ShotGlyph: View {
    public var kind: ShotGlyphKind
    public var color: Color
    public var showsCourt: Bool

    public init(_ kind: ShotGlyphKind, color: Color = .ink, showsCourt: Bool = true) {
        self.kind = kind; self.color = color; self.showsCourt = showsCourt
    }

    public var body: some View {
        Canvas { ctx, size in
            let s = min(size.width, size.height)
            let o = CGPoint(x: (size.width - s) / 2, y: (size.height - s) / 2)
            func q(_ p: CGPoint) -> CGPoint { CGPoint(x: o.x + p.x * s, y: o.y + p.y * s) }
            let weight = max(1.2, s * 0.085)

            if showsCourt {
                var floor = Path()
                floor.move(to: q(CGPoint(x: 0.04, y: 0.92))); floor.addLine(to: q(CGPoint(x: 0.96, y: 0.92)))
                ctx.stroke(floor, with: .color(color.opacity(0.45)), style: StrokeStyle(lineWidth: max(1, s * 0.05), lineCap: .round))
                var post = Path()
                post.move(to: q(CGPoint(x: 0.5, y: 0.92))); post.addLine(to: q(CGPoint(x: 0.5, y: 0.58)))
                ctx.stroke(post, with: .color(color), style: StrokeStyle(lineWidth: max(1, s * 0.07), lineCap: .round))
            }

            guard let (a, c, b) = kind.flight else {
                let d = s * 0.12
                for i in 0..<3 {
                    let x = o.x + s * (0.28 + CGFloat(i) * 0.22)
                    ctx.fill(Path(ellipseIn: CGRect(x: x - d / 2, y: o.y + s * 0.42, width: d, height: d)), with: .color(color))
                }
                return
            }
            var flight = Path()
            flight.move(to: q(a)); flight.addQuadCurve(to: q(b), control: q(c))
            ctx.stroke(flight, with: .color(color), style: StrokeStyle(lineWidth: weight, lineCap: .round))
            let head = s * 0.2
            let e = q(b)
            ctx.fill(Path(ellipseIn: CGRect(x: e.x - head / 2, y: e.y - head / 2, width: head, height: head)), with: .color(color))
        }
        .accessibilityHidden(true)
    }
}

// MARK: - ShotChip
// A printed tile: the shot's ink, an outline, and its pictogram — used in legends,
// timelines and filters.

public struct ShotChip: View {
    public enum Style: Sendable {
        /// Filled with the shot's ink — the chip *is* the colour key (legends, timelines).
        case ink
        /// Card stock with an ink glyph — when something next to it already carries the ink
        /// (a bar), so the colour isn't printed twice.
        case paper
    }

    public var kind: ShotGlyphKind
    public var size: CGFloat
    public var style: Style

    public init(_ kind: ShotGlyphKind, size: CGFloat = 22, style: Style = .ink) {
        self.kind = kind; self.size = size; self.style = style
    }

    public var body: some View {
        let shape = RoundedRectangle(cornerRadius: size * 0.28, style: .continuous)
        let onInk: Color = kind == .clear || kind == .lift ? .card : .ink
        ShotGlyph(kind, color: style == .ink ? onInk : .ink)
            .padding(size * 0.14)
            .frame(width: size, height: size)
            .background(style == .ink ? kind.ink : Color.card, in: shape)
            .overlay(shape.strokeBorder(Color.ink, lineWidth: max(1, size * 0.055)))
            .accessibilityHidden(true)
    }
}

#Preview("ShotGlyph") {
    VStack(alignment: .leading, spacing: 10) {
        ForEach(ShotGlyphKind.allCases, id: \.self) { k in
            HStack(spacing: 10) {
                ShotChip(k, size: 34)
                ShotGlyph(k).frame(width: 44, height: 44)
                ShotChip(k, size: 16)
                Text(k.chineseName).font(AppFont.headline)
            }
        }
    }
    .padding(24)
    .paperPage()
}
