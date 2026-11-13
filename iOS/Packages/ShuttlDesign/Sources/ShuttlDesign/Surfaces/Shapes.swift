import SwiftUI

// MARK: - Scallop
// A rosette / rubber-stamp outline: a circle with `lobes` soft bumps.
// Used for "sticker" badges and the hero speed seal.

public struct Scallop: InsettableShape {
    public var lobes: Int = 12
    public var depth: CGFloat = 0.08
    public var inset: CGFloat = 0

    public init(lobes: Int = 12, depth: CGFloat = 0.08) { self.lobes = lobes; self.depth = depth }

    public func path(in rect: CGRect) -> Path {
        let r = rect.insetBy(dx: inset, dy: inset)
        let c = CGPoint(x: r.midX, y: r.midY)
        let outer = min(r.width, r.height) / 2
        let inner = outer * (1 - depth)
        let steps = lobes * 16
        var p = Path()
        for i in 0...steps {
            let a = Double(i) / Double(steps) * 2 * .pi - .pi / 2
            let wave = (cos(a * Double(lobes)) + 1) / 2
            let rad = inner + (outer - inner) * wave
            let pt = CGPoint(x: c.x + CGFloat(cos(a)) * rad, y: c.y + CGFloat(sin(a)) * rad)
            i == 0 ? p.move(to: pt) : p.addLine(to: pt)
        }
        p.closeSubpath()
        return p
    }

    public func inset(by amount: CGFloat) -> Scallop {
        var s = self; s.inset += amount; return s
    }
}

// MARK: - Squiggle
// A hand-drawn wave rule. Drawn as a stroke; use as an underline or divider.

public struct Squiggle: Shape {
    public var waves: CGFloat = 6

    public init(waves: CGFloat = 6) { self.waves = waves }

    public func path(in rect: CGRect) -> Path {
        var p = Path()
        let mid = rect.midY
        let amp = rect.height / 2
        let steps = 120
        for i in 0...steps {
            let t = CGFloat(i) / CGFloat(steps)
            let x = rect.minX + rect.width * t
            let y = mid + sin(t * waves * 2 * .pi) * amp
            i == 0 ? p.move(to: CGPoint(x: x, y: y)) : p.addLine(to: CGPoint(x: x, y: y))
        }
        return p
    }
}

// MARK: - Halftone field
// Dot grid whose dot size fades across the rect — the riso/newsprint screen.

public struct Halftone: View {
    public var color: Color
    public var spacing: CGFloat = 9
    public var maxDot: CGFloat = 4
    /// Where dots are largest, in unit coords.
    public var focus: UnitPoint = .topTrailing

    public init(color: Color, spacing: CGFloat = 9, maxDot: CGFloat = 4, focus: UnitPoint = .topTrailing) {
        self.color = color; self.spacing = spacing; self.maxDot = maxDot; self.focus = focus
    }

    public var body: some View {
        Canvas { ctx, size in
            let fx = focus.x * size.width, fy = focus.y * size.height
            let reach = hypot(size.width, size.height)
            var y: CGFloat = spacing / 2
            var row = 0
            while y < size.height {
                var x: CGFloat = (row % 2 == 0 ? spacing / 2 : spacing)
                while x < size.width {
                    let d = hypot(x - fx, y - fy) / reach
                    let dot = maxDot * max(0, 1 - d * 1.6)
                    if dot > 0.4 {
                        ctx.fill(Path(ellipseIn: CGRect(x: x - dot / 2, y: y - dot / 2, width: dot, height: dot)),
                                 with: .color(color))
                    }
                    x += spacing
                }
                y += spacing * 0.866
                row += 1
            }
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

// MARK: - Ticket
// Rounded card with two half-circle bites at `stubX` — a torn-off ticket stub.

public struct Ticket: InsettableShape {
    public var stubX: CGFloat
    public var radius: CGFloat = AppRadius.md
    public var notch: CGFloat = 9
    public var inset: CGFloat = 0

    public init(stubX: CGFloat, radius: CGFloat = AppRadius.md, notch: CGFloat = 9) {
        self.stubX = stubX; self.radius = radius; self.notch = notch
    }

    public func path(in rect: CGRect) -> Path {
        let r = rect.insetBy(dx: inset, dy: inset)
        let body = Path(roundedRect: r, cornerRadius: max(0, radius - inset), style: .continuous)
        let x = rect.minX + stubX
        let n = notch + inset
        let bites = Path { p in
            p.addEllipse(in: CGRect(x: x - n, y: r.minY - n, width: n * 2, height: n * 2))
            p.addEllipse(in: CGRect(x: x - n, y: r.maxY - n, width: n * 2, height: n * 2))
        }
        return body.subtracting(bites)
    }

    public func inset(by amount: CGFloat) -> Ticket {
        var t = self; t.inset += amount; return t
    }
}
