import UIKit
import SwiftUI
import CoreGraphics
import QuartzCore

// MARK: - TrajectoryPoint

struct TrajectoryPoint: Sendable {
    var position: CGPoint   // normalised [0, 1] in both axes
    var frameIndex: Int
    var speedKmh: Float
}

// MARK: - TrajectoryRenderer

@MainActor
final class TrajectoryRenderer {

    static let shared = TrajectoryRenderer()

    private init() {}

    // MARK: - CGContext rendering

    // Trail body in tomato, head in butter with an ink ring — matches the SwiftUI overlays.
    private static let trailColor = UIColor(Color.tomato)
    private static let headColor  = UIColor(Color.butter)
    private static let ringColor  = UIColor(Color.ink)

    /// Draws a fading tomato trail into `context` and a butter dot at the
    /// newest position.  `trajectory` is ordered oldest → newest; up to the
    /// last 30 points are drawn.  Positions are normalised [0, 1] and are
    /// scaled to `bounds`.
    func render(trajectory: [TrajectoryPoint], in context: CGContext, bounds: CGRect) {
        let points = Array(trajectory.suffix(30))
        guard points.count >= 2 else {
            // Draw a lone dot if there is exactly one point.
            if let sole = points.first {
                let pt = toScreen(sole.position, bounds: bounds)
                drawDot(at: pt, alpha: 1.0, in: context)
            }
            return
        }

        let count = points.count

        // Draw each segment with linearly decreasing alpha (oldest → newest).
        context.saveGState()
        context.setLineWidth(3.0)
        context.setLineCap(.round)
        context.setLineJoin(.round)

        for i in 1..<count {
            let alpha = CGFloat(i) / CGFloat(count - 1)
            context.setStrokeColor(Self.trailColor.withAlphaComponent(alpha).cgColor)
            let from = toScreen(points[i - 1].position, bounds: bounds)
            let to   = toScreen(points[i].position,     bounds: bounds)
            context.move(to: from)
            context.addLine(to: to)
            context.strokePath()
        }

        context.restoreGState()

        // Filled dot at the current (newest) position, full alpha.
        let current = toScreen(points[count - 1].position, bounds: bounds)
        drawDot(at: current, alpha: 1.0, in: context)
    }

    // MARK: - CAShapeLayer rendering

    /// Returns a `CAShapeLayer` whose path traces the fading trail.  The
    /// layer is configured with the same tomato trail and a butter dot at the
    /// end; the caller is responsible for adding it to a view hierarchy.
    func renderLayer(trajectory: [TrajectoryPoint], bounds: CGRect) -> CAShapeLayer {
        let container = CAShapeLayer()
        container.frame = bounds
        container.fillColor   = UIColor.clear.cgColor
        container.strokeColor = UIColor.clear.cgColor

        let points = Array(trajectory.suffix(30))
        guard points.count >= 2 else {
            if let sole = points.first {
                let pt    = toScreen(sole.position, bounds: bounds)
                let dot   = dotLayer(at: pt, radius: 6, alpha: 1.0)
                container.addSublayer(dot)
            }
            return container
        }

        let count = points.count

        // One sub-layer per segment so each can carry its own opacity.
        for i in 1..<count {
            let alpha  = CGFloat(i) / CGFloat(count - 1)
            let from   = toScreen(points[i - 1].position, bounds: bounds)
            let to     = toScreen(points[i].position,     bounds: bounds)

            let segment        = CAShapeLayer()
            segment.frame      = bounds
            let path           = UIBezierPath()
            path.move(to: from)
            path.addLine(to: to)
            segment.path        = path.cgPath
            segment.strokeColor = Self.trailColor.cgColor
            segment.lineWidth   = 3.0
            segment.lineCap     = .round
            segment.fillColor   = UIColor.clear.cgColor
            segment.opacity     = Float(alpha)
            container.addSublayer(segment)
        }

        // Dot at newest point.
        let current = toScreen(points[count - 1].position, bounds: bounds)
        container.addSublayer(dotLayer(at: current, radius: 6, alpha: 1.0))

        return container
    }

    // MARK: - Helpers

    private func toScreen(_ normalised: CGPoint, bounds: CGRect) -> CGPoint {
        CGPoint(
            x: normalised.x * bounds.size.width  + bounds.origin.x,
            y: normalised.y * bounds.size.height + bounds.origin.y
        )
    }

    private func drawDot(at center: CGPoint, alpha: CGFloat, in context: CGContext) {
        let radius: CGFloat = 6
        let rect = CGRect(
            x: center.x - radius,
            y: center.y - radius,
            width:  radius * 2,
            height: radius * 2
        )
        context.saveGState()
        context.setFillColor(Self.ringColor.withAlphaComponent(alpha).cgColor)
        context.fillEllipse(in: rect.insetBy(dx: -2, dy: -2))
        context.setFillColor(Self.headColor.withAlphaComponent(alpha).cgColor)
        context.fillEllipse(in: rect)
        context.restoreGState()
    }

    private func dotLayer(at center: CGPoint, radius: CGFloat, alpha: CGFloat) -> CAShapeLayer {
        let layer       = CAShapeLayer()
        let rect        = CGRect(
            x: center.x - radius,
            y: center.y - radius,
            width:  radius * 2,
            height: radius * 2
        )
        layer.path      = UIBezierPath(ovalIn: rect).cgPath
        layer.fillColor   = Self.headColor.withAlphaComponent(alpha).cgColor
        layer.strokeColor = Self.ringColor.cgColor
        layer.lineWidth   = 2
        layer.opacity   = Float(alpha)
        return layer
    }
}
