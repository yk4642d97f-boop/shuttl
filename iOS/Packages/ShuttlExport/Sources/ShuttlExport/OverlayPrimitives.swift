import CoreGraphics
import CoreText
import Foundation

/// One line of Core Text built from styled runs, measured once, drawn at a baseline.
struct TextRun {
    let line: CTLine
    let width: CGFloat
    let ascent: CGFloat
    let descent: CGFloat

    init(_ parts: [(String, CTFont, CGColor)]) {
        let text = NSMutableAttributedString()
        for (string, font, color) in parts {
            text.append(NSAttributedString(string: string, attributes: [
                NSAttributedString.Key(kCTFontAttributeName as String): font,
                NSAttributedString.Key(kCTForegroundColorAttributeName as String): color,
            ]))
        }
        line = CTLineCreateWithAttributedString(text)
        var a: CGFloat = 0, d: CGFloat = 0, l: CGFloat = 0
        width = CGFloat(CTLineGetTypographicBounds(line, &a, &d, &l))
        ascent = a; descent = d
    }

    init(_ string: String, font: CTFont, color: CGColor) { self.init([(string, font, color)]) }

    var height: CGFloat { ascent + descent }

    /// Draws with the baseline starting at `origin`, in the renderer's y-down space.
    func draw(baseline origin: CGPoint, in context: CGContext) {
        context.saveGState()
        context.translateBy(x: origin.x, y: origin.y)
        context.scaleBy(x: 1, y: -1)
        context.textMatrix = .identity
        context.textPosition = .zero
        CTLineDraw(line, context)
        context.restoreGState()
    }

    /// Baseline that centres the line's ascent+descent box on `y`.
    func baseline(centeredAt y: CGFloat) -> CGFloat { y + (ascent - descent) / 2 }
}

/// A printed sticker: a spot-ink capsule with an ink outline and a solid offset ink
/// "print shadow", tilted like the SwiftUI `Sticker`. Geometry is in the renderer's
/// y-down space, so a negative `tilt` leans the sticker the same way SwiftUI's does.
struct StickerMark {
    var text: TextRun
    var fill: CGColor
    var tiltDegrees: CGFloat
    /// Horizontal / vertical padding around the text.
    var padX: CGFloat
    var padY: CGFloat
    var outline: CGFloat
    var shadow: CGSize

    var size: CGSize { CGSize(width: text.width + padX * 2, height: text.height + padY * 2) }

    func draw(centeredAt center: CGPoint, alpha: CGFloat = 1, scale: CGFloat = 1, in context: CGContext) {
        let s = size
        let body = CGRect(x: -s.width / 2, y: -s.height / 2, width: s.width, height: s.height)
        let capsule = CGPath(roundedRect: body, cornerWidth: s.height / 2, cornerHeight: s.height / 2, transform: nil)
        context.saveGState()
        context.translateBy(x: center.x, y: center.y)
        context.rotate(by: tiltDegrees * .pi / 180)
        context.scaleBy(x: scale, y: scale)
        let transparent = alpha < 0.999
        if transparent {
            context.setAlpha(alpha)
            context.beginTransparencyLayer(in: body.insetBy(dx: -abs(shadow.width) - outline, dy: -abs(shadow.height) - outline),
                                           auxiliaryInfo: nil)
        }
        // Print shadow, then the ink fill, then the outline laid inside the edge.
        context.saveGState()
        context.translateBy(x: shadow.width, y: shadow.height)
        context.addPath(capsule)
        context.setFillColor(ExportTheme.ink)
        context.fillPath()
        context.restoreGState()
        context.addPath(capsule)
        context.setFillColor(fill)
        context.fillPath()
        let inset = body.insetBy(dx: outline / 2, dy: outline / 2)
        let r = inset.height / 2
        context.addPath(CGPath(roundedRect: inset, cornerWidth: r, cornerHeight: r, transform: nil))
        context.setStrokeColor(ExportTheme.ink)
        context.setLineWidth(outline)
        context.strokePath()
        text.draw(baseline: CGPoint(x: -text.width / 2, y: text.baseline(centeredAt: 0)), in: context)
        if transparent { context.endTransparencyLayer() }
        context.restoreGState()
    }
}
