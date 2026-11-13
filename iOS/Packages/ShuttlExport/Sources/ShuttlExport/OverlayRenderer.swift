import CoreGraphics
import CoreText
import CoreVideo
import Foundation
import ShuttlCore
import ShuttlDesign

/// Burns the "Riso Court · Overprint" overlay into each exported frame: a tapered
/// lilac → tomato → butter trail with an ink edge, butter hit discs, tilted sticker
/// labels (shot + km/h), landing stamps, a paper score card and the Shuttl. mark.
/// Drawing happens in source-pixel space (y down); `u` is one design point at 1080p.
struct OverlayRenderer {
    let analysis: MatchAnalysis
    let options: ExportOptions
    let width: Int
    let height: Int
    let scaleX: CGFloat
    let scaleY: CGFloat
    /// One design point in source pixels (1 at 1080p, 2 at 2160p).
    let u: CGFloat
    /// Ink outline weight: AppStroke.ink (1.5 pt) scaled like the rest of the type.
    let outline: CGFloat
    private let shots: [ShotEventV2]
    private let sourceWidth: CGFloat
    private let sourceHeight: CGFloat
    private let labelFont: CTFont
    private let numeralFont: CTFont
    private let stampFont: CTFont
    private let scoreFont: CTFont
    private let tagFont: CTFont
    private let markFont: CTFont

    static let trailSeconds = 0.42

    init(analysis: MatchAnalysis, options: ExportOptions, width: Int, height: Int) {
        self.analysis = analysis
        self.options = options
        self.width = width
        self.height = height
        sourceWidth = CGFloat(max(analysis.meta.width, 1))
        sourceHeight = CGFloat(max(analysis.meta.height, 1))
        scaleX = CGFloat(width) / sourceWidth
        scaleY = CGFloat(height) / sourceHeight
        u = CGFloat(min(analysis.meta.width, analysis.meta.height)) / 1080
        outline = max(1, 1.5 * 1.5 * u)
        shots = Self.collectShots(analysis)
        DisplayFace.register()
        labelFont = ExportTheme.displayFont(28 * u)
        numeralFont = ExportTheme.displayFont(37 * u)
        stampFont = ExportTheme.displayFont(30 * u)
        scoreFont = ExportTheme.displayFont(46 * u)
        tagFont = ExportTheme.monoFont(16 * u)
        markFont = ExportTheme.displayFont(42 * u)
    }

    func draw(frame: Int, pixelBuffer: CVPixelBuffer) {
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA else { return }
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer),
              let context = CGContext(
                data: base,
                width: CVPixelBufferGetWidth(pixelBuffer),
                height: CVPixelBufferGetHeight(pixelBuffer),
                bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
                space: ExportTheme.sRGB,
                bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
              ) else { return }
        draw(frame: frame, in: context)
    }

    /// Draws one frame's overlay into a context whose origin is the bottom-left of the
    /// output image (as for a CVPixelBuffer-backed bitmap).
    func draw(frame: Int, in context: CGContext) {
        context.saveGState()
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)
        context.scaleBy(x: scaleX, y: scaleY)
        context.setLineCap(.round)
        context.setLineJoin(.round)
        context.setShouldAntialias(true)
        context.setShouldSmoothFonts(false)

        let time = analysis.meta.time(ofFrame: frame)
        drawTrail(time: time, frame: frame, in: context)
        drawRallyEnd(frame: frame, time: time, in: context)
        drawHits(frame: frame, time: time, in: context)
        drawScore(frame: frame, in: context)
        if options.watermark { drawWatermark(in: context) }
        context.restoreGState()
    }

    // MARK: Trail
    // A ribbon that is widest and butter at the shuttle, runs through tomato and
    // tapers to a lilac hairline at the tail (`Gradient.trajectory`). Every piece is
    // opaque, so round caps merge into one clean ribbon instead of beading; the ink
    // edge is one pass under the colour pass. Gap-filled (interpolated) stretches are
    // printed as a faint dotted lilac line, so guessed flight never looks measured.

    private func drawTrail(time: Double, frame: Int, in context: CGContext) {
        guard let track = analysis.track, !track.t.isEmpty else { return }
        let lowerTime = time - Self.trailSeconds
        var end = min(frame, track.n - 1, track.t.count - 1)
        while end >= 0 && track.t[end] > time { end -= 1 }
        guard end >= 0 else { return }
        var start = end
        while start > 0 && track.t[start - 1] >= lowerTime { start -= 1 }
        guard start < end else { return }

        struct Piece { let a: CGPoint; let b: CGPoint; let fresh: Double; let guessed: Bool }
        var pieces: [Piece] = []
        for index in (start + 1)...end {
            let previous = index - 1
            guard index < track.xy.count, track.xy[previous].isFinite, track.xy[index].isFinite,
                  track.t[index] - track.t[previous] <= 0.12 else { continue }
            let age = max(0, min(1, (time - track.t[index]) / Self.trailSeconds))
            let guessed = (index < track.interpolated.count && track.interpolated[index]) ||
                (previous < track.interpolated.count && track.interpolated[previous])
            pieces.append(Piece(a: CGPoint(x: track.xy[previous].x, y: track.xy[previous].y),
                                b: CGPoint(x: track.xy[index].x, y: track.xy[index].y),
                                fresh: 1 - age, guessed: guessed))
        }
        guard !pieces.isEmpty else { return }

        // Guessed stretches: dotted lilac, drawn as whole runs so the dash never restarts.
        var run: [CGPoint] = []
        func flushRun() {
            if run.count > 1 {
                context.addLines(between: run)
                context.setStrokeColor(ExportTheme.lilac.with(alpha: 0.7))
                context.setLineWidth(max(1.5, 4 * u))
                context.setLineDash(phase: 0, lengths: [0.01, 10 * u])
                context.strokePath()
                context.setLineDash(phase: 0, lengths: [])
            }
            run = []
        }
        for piece in pieces {
            if piece.guessed {
                if run.last != piece.a { flushRun(); run = [piece.a] }
                run.append(piece.b)
            } else { flushRun() }
        }
        flushRun()

        let solid = pieces.filter { !$0.guessed }
        func core(_ fresh: Double) -> CGFloat { max(2.5, CGFloat(2 + 10 * fresh) * u) }
        for piece in solid {                                    // ink edge
            context.move(to: piece.a); context.addLine(to: piece.b)
            context.setStrokeColor(ExportTheme.ink)
            context.setLineWidth(core(piece.fresh) + max(1.5, 3.4 * u))
            context.strokePath()
        }
        for piece in solid {                                    // spot-ink body
            context.move(to: piece.a); context.addLine(to: piece.b)
            context.setStrokeColor(Self.trailColor(piece.fresh))
            context.setLineWidth(core(piece.fresh))
            context.strokePath()
        }
    }

    /// lilac (tail) → tomato → butter (head), as in `Gradient.trajectory`.
    static func trailColor(_ fresh: Double) -> CGColor {
        let stops: [(Double, CGColor)] = [(0, ExportTheme.lilac), (0.55, ExportTheme.tomato), (0.9, ExportTheme.butter)]
        if fresh <= stops[0].0 { return stops[0].1 }
        for i in 1..<stops.count where fresh <= stops[i].0 {
            let t = (fresh - stops[i - 1].0) / (stops[i].0 - stops[i - 1].0)
            return mix(stops[i - 1].1, stops[i].1, CGFloat(t))
        }
        return stops[stops.count - 1].1
    }

    private static func mix(_ a: CGColor, _ b: CGColor, _ t: CGFloat) -> CGColor {
        guard let ca = a.components, let cb = b.components, ca.count >= 3, cb.count >= 3 else { return b }
        let c = (0..<3).map { ca[$0] + (cb[$0] - ca[$0]) * t }
        return CGColor(colorSpace: ExportTheme.sRGB, components: [c[0], c[1], c[2], 1]) ?? b
    }

    // MARK: Hits
    // A butter disc with an ink ring (and a quick ping ring), a hairline leader and
    // a tilted sticker in the shot's ink carrying the name and the speed numeral.

    private func drawHits(frame: Int, time: Double, in context: CGContext) {
        for shot in shots {
            let hit = shot.hit
            let hitTime = hit.t.isFinite ? hit.t : analysis.meta.time(ofFrame: hit.frame)
            let age = time - hitTime
            guard hit.frame <= frame, age <= 1.0, hit.xyPx.isFinite else { continue }
            let point = CGPoint(x: hit.xyPx.x, y: hit.xyPx.y)
            let fade = CGFloat(max(0, min(1, (1.0 - age) / 0.2)))
            let radius = 12 * u
            let sticker = label(for: shot)
            let center = sticker.map { stickerCenter(for: $0, near: point, hitFrame: hit.frame) }

            var bounds = CGRect(x: point.x - radius * 3, y: point.y - radius * 3, width: radius * 6, height: radius * 6)
            if let sticker, let center {
                let s = sticker.size
                bounds = bounds.union(CGRect(x: center.x - s.width, y: center.y - s.height, width: s.width * 2, height: s.height * 2))
            }
            let layered = fade < 0.999
            if layered {
                context.saveGState()
                context.setAlpha(fade)
                context.beginTransparencyLayer(in: bounds, auxiliaryInfo: nil)
            }
            if let center {                                     // leader, hidden under disc + sticker
                context.move(to: point); context.addLine(to: center)
                context.setStrokeColor(ExportTheme.ink)
                context.setLineWidth(outline)
                context.strokePath()
            }
            if age < 0.5 {                                      // ping
                let t = CGFloat(age / 0.5)
                let r = radius * (1.4 + 1.6 * t)
                context.setStrokeColor(ExportTheme.butter.with(alpha: 0.9 * (1 - t)))
                context.setLineWidth(max(1, 3 * u * (1 - t)))
                context.strokeEllipse(in: CGRect(x: point.x - r, y: point.y - r, width: r * 2, height: r * 2))
            }
            let disc = CGRect(x: point.x - radius, y: point.y - radius, width: radius * 2, height: radius * 2)
            context.setFillColor(ExportTheme.butter)
            context.fillEllipse(in: disc)
            context.setStrokeColor(ExportTheme.ink)
            context.setLineWidth(max(1, 3 * u))
            context.strokeEllipse(in: disc.insetBy(dx: 1.5 * u, dy: 1.5 * u))
            if let sticker, let center { sticker.draw(centeredAt: center, in: context) }
            if layered {
                context.endTransparencyLayer()
                context.restoreGState()
            }
        }
    }

    private func label(for shot: ShotEventV2) -> StickerMark? {
        let ink = ExportTheme.onShotColor(shot.shotType.rawValue)
        var parts: [(String, CTFont, CGColor)] = []
        if options.showShotType, let name = ExportTheme.shotName(shot.shotType) {
            parts.append((name, labelFont, ink))
        }
        if options.showSpeed, analysis.camera != nil, let segment = shot.segment,
           segment.speed0Kmh.isFinite, segment.speed0Kmh > 0 {
            if !parts.isEmpty { parts.append(("  ", labelFont, ink)) }
            parts.append(("\(Int(segment.speed0Kmh.rounded()))", numeralFont, ink))
            parts.append((" km/h", labelFont, ink))
        }
        guard !parts.isEmpty else { return nil }
        return StickerMark(text: TextRun(parts), fill: ExportTheme.shotColor(shot.shotType.rawValue),
                           tiltDegrees: shot.hit.frame % 2 == 0 ? -4 : 3,
                           padX: 17 * u, padY: 8 * u, outline: outline,
                           shadow: CGSize(width: 2 * u, height: 3 * u))
    }

    /// Picks the diagonal beside the hit that the flight does not pass through (the
    /// trail before and after the hit is sampled), preferring up-right, and keeps the
    /// sticker inside the frame. Depends only on the hit, so it never jumps between frames.
    private func stickerCenter(for sticker: StickerMark, near point: CGPoint, hitFrame: Int) -> CGPoint {
        let s = sticker.size
        let margin = 18 * u
        let dx = 26 * u + s.width / 2, dy = 30 * u + s.height / 2
        var flight: [CGPoint] = []
        if let track = analysis.track {
            let lo = max(0, hitFrame - 26), hi = min(track.xy.count - 1, hitFrame + 60)
            if lo <= hi {
                for f in lo...hi where track.xy[f].isFinite {
                    flight.append(CGPoint(x: track.xy[f].x, y: track.xy[f].y))
                }
            }
        }
        let samples = flight + zip(flight, flight.dropFirst()).map {
            CGPoint(x: ($0.x + $1.x) / 2, y: ($0.y + $1.y) / 2)
        }
        func clamp(_ c: CGPoint) -> CGPoint {
            CGPoint(x: min(max(c.x, s.width / 2 + margin), sourceWidth - s.width / 2 - margin),
                    y: min(max(c.y, s.height / 2 + margin), sourceHeight - s.height / 2 - margin))
        }
        let candidates = [(1.0, -1.0), (-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].map { sx, sy in
            CGPoint(x: point.x + CGFloat(sx) * dx, y: point.y + CGFloat(sy) * dy)
        }
        var best = clamp(candidates[0]), bestScore = CGFloat.infinity
        for (rank, raw) in candidates.enumerated() {
            let c = clamp(raw)
            let box = CGRect(x: c.x - s.width / 2, y: c.y - s.height / 2, width: s.width, height: s.height)
                .insetBy(dx: -8 * u, dy: -8 * u)
            let hits = samples.filter { box.contains($0) }.count
            let shift = hypot(c.x - raw.x, c.y - raw.y) / max(s.width, 1)   // pushed off-frame
            let score = CGFloat(hits) * 10 + shift * 4 + CGFloat(rank) * 0.5
            if score < bestScore { bestScore = score; best = c }
        }
        return best
    }

    // MARK: Rally end
    // The landing spot as a ring laid flat on the floor (projected through the camera
    // when there is one) and a stamp sticker: 界内 pistachio, 出界 tomato, 触网 butter.

    private func drawRallyEnd(frame: Int, time: Double, in context: CGContext) {
        for rally in analysis.rallies {
            let endTime = analysis.meta.time(ofFrame: rally.endFrame)
            guard rally.endFrame <= frame, time - endTime <= 1.0 else { continue }
            let fill: CGColor
            switch rally.endReason {
            case .inCourt: fill = ExportTheme.pistachio
            case .out:     fill = ExportTheme.tomato
            case .net:     fill = ExportTheme.butter
            case .unknown: fill = ExportTheme.card
            }
            var ring: [CGPoint] = []
            if let landing = rally.landingXY, let camera = analysis.camera {
                ring = (0..<32).compactMap { i -> CGPoint? in
                    let a = Double(i) / 32 * 2 * .pi
                    let p = camera.project(SIMD3(landing.x + 0.5 * cos(a), landing.y + 0.5 * sin(a), 0))
                    return p.isFinite ? CGPoint(x: p.x, y: p.y) : nil
                }
            }
            let center: CGPoint
            if ring.count == 32 {
                center = CGPoint(x: ring.map(\.x).reduce(0, +) / 32, y: ring.map(\.y).reduce(0, +) / 32)
            } else if let track = analysis.track, rally.endFrame >= 0, rally.endFrame < track.xy.count,
                      track.xy[rally.endFrame].isFinite {
                center = CGPoint(x: track.xy[rally.endFrame].x, y: track.xy[rally.endFrame].y)
                let r = 34 * u
                ring = (0..<32).map { i in
                    let a = CGFloat(i) / 32 * 2 * .pi
                    return CGPoint(x: center.x + r * cos(a), y: center.y + r * 0.45 * sin(a))
                }
            } else { continue }

            let path = CGMutablePath()
            path.addLines(between: ring)
            path.closeSubpath()
            context.addPath(path)                               // a flat spot of ink on the floor
            context.setFillColor(fill.with(alpha: 0.35))
            context.fillPath()
            context.addPath(path)
            context.setStrokeColor(ExportTheme.ink)
            context.setLineWidth(max(2, 9 * u))
            context.strokePath()
            context.addPath(path)
            context.setStrokeColor(fill)
            context.setLineWidth(max(1, 5 * u))
            context.strokePath()
            let dot = max(2, 5 * u)
            context.setFillColor(ExportTheme.ink)
            context.fillEllipse(in: CGRect(x: center.x - dot, y: center.y - dot, width: dot * 2, height: dot * 2))

            guard let word = Self.endLabel[rally.endReason] else { continue }
            let stamp = StickerMark(text: TextRun(word, font: stampFont, color: ExportTheme.ink), fill: fill,
                                    tiltDegrees: 6, padX: 16 * u, padY: 7 * u, outline: outline,
                                    shadow: CGSize(width: 2 * u, height: 3 * u))
            let s = stamp.size
            let reach = (ring.map { abs($0.x - center.x) }.max() ?? 24 * u) + 14 * u + s.width / 2
            var x = center.x + reach
            if x + s.width / 2 > sourceWidth - 18 * u { x = center.x - reach }
            let y = min(max(center.y - 10 * u, s.height / 2 + 18 * u), sourceHeight - s.height / 2 - 18 * u)
            stamp.draw(centeredAt: CGPoint(x: x, y: y), in: context)
        }
    }

    // MARK: Score card
    // A paper card with an ink outline and an offset ink print shadow: a mono game tag
    // and two display-face scores on spot-ink chips, cobalt for near and pink for far.

    private func drawScore(frame: Int, in context: CGContext) {
        guard options.showScore, !analysis.score.isEmpty else { return }
        let event = analysis.score
            .compactMap { $0.objectValue }
            .filter { ($0["end_frame"]?.intValue ?? Int.max) <= frame }
            .max { ($0["end_frame"]?.intValue ?? 0) < ($1["end_frame"]?.intValue ?? 0) }
        guard let event, let score = event["score"]?.objectValue else { return }
        let near = score["near"]?.intValue ?? 0
        let far = score["far"]?.intValue ?? 0
        let game = event["game"]?.intValue ?? 1

        let tag = TextRun("第 \(game) 局", font: tagFont, color: ExportTheme.inkSoft)
        let nearRun = TextRun("\(near)", font: scoreFont, color: ExportTheme.card)
        let farRun = TextRun("\(far)", font: scoreFont, color: ExportTheme.ink)
        let colon = TextRun(":", font: scoreFont, color: ExportTheme.ink)
        let nearTag = TextRun("近端", font: tagFont, color: ExportTheme.inkSoft)
        let farTag = TextRun("远端", font: tagFont, color: ExportTheme.inkSoft)
        let pad = 14 * u, chipH = 58 * u, gap = 9 * u, lineGap = 7 * u
        let chipW = max(nearRun.width, farRun.width, 26 * u) + 30 * u
        let rowW = chipW * 2 + gap * 2 + colon.width
        let cardW = max(rowW, tag.width) + pad * 2
        let cardH = pad + tag.height + lineGap + chipH + lineGap + nearTag.height + pad - 2 * u
        let card = CGRect(x: 28 * u, y: 28 * u, width: cardW, height: cardH)
        let radius = 16 * u
        let cardPath = CGPath(roundedRect: card, cornerWidth: radius, cornerHeight: radius, transform: nil)

        context.saveGState()                                    // offset ink print shadow
        context.translateBy(x: 4 * u, y: 5 * u)
        context.addPath(cardPath); context.setFillColor(ExportTheme.ink); context.fillPath()
        context.restoreGState()
        context.addPath(cardPath); context.setFillColor(ExportTheme.paper); context.fillPath()
        strokeInside(card, radius: radius, in: context)
        tag.draw(baseline: CGPoint(x: card.minX + pad, y: card.minY + pad + tag.ascent), in: context)

        let rowY = card.minY + pad + tag.height + lineGap
        let rowX = card.minX + (cardW - rowW) / 2
        let nearChip = CGRect(x: rowX, y: rowY, width: chipW, height: chipH)
        let farChip = CGRect(x: nearChip.maxX + gap * 2 + colon.width, y: rowY, width: chipW, height: chipH)
        let captionY = nearChip.maxY + lineGap + nearTag.ascent
        for (chip, fill, run, caption) in [(nearChip, ExportTheme.cobalt, nearRun, nearTag),
                                           (farChip, ExportTheme.pink, farRun, farTag)] {
            context.addPath(CGPath(roundedRect: chip, cornerWidth: 12 * u, cornerHeight: 12 * u, transform: nil))
            context.setFillColor(fill); context.fillPath()
            strokeInside(chip, radius: 12 * u, in: context)
            run.draw(baseline: CGPoint(x: chip.midX - run.width / 2, y: run.baseline(centeredAt: chip.midY)), in: context)
            caption.draw(baseline: CGPoint(x: chip.midX - caption.width / 2, y: captionY), in: context)
        }
        colon.draw(baseline: CGPoint(x: nearChip.maxX + gap, y: colon.baseline(centeredAt: nearChip.midY)), in: context)
    }

    private func strokeInside(_ rect: CGRect, radius: CGFloat, in context: CGContext) {
        let inset = rect.insetBy(dx: outline / 2, dy: outline / 2)
        let r = max(0, radius - outline / 2)
        context.addPath(CGPath(roundedRect: inset, cornerWidth: r, cornerHeight: r, transform: nil))
        context.setStrokeColor(ExportTheme.ink)
        context.setLineWidth(outline)
        context.strokePath()
    }

    // MARK: Watermark
    // "Shuttl." in the display face, card white with a tomato full stop over an
    // offset ink plate, bottom-right at ~70 % so it signs the clip without shouting.

    private func drawWatermark(in context: CGContext) {
        let mark = TextRun([("Shuttl", markFont, ExportTheme.card), (".", markFont, ExportTheme.tomato)])
        let plate = TextRun("Shuttl.", font: markFont, color: ExportTheme.ink)
        let margin = 30 * u
        let origin = CGPoint(x: sourceWidth - margin - mark.width, y: sourceHeight - margin - mark.descent)
        let bounds = CGRect(x: origin.x - 4 * u, y: origin.y - mark.ascent - 4 * u,
                            width: mark.width + 12 * u, height: mark.height + 12 * u)
        context.saveGState()
        context.setAlpha(0.7)
        context.beginTransparencyLayer(in: bounds, auxiliaryInfo: nil)
        plate.draw(baseline: CGPoint(x: origin.x + 2.5 * u, y: origin.y + 3 * u), in: context)
        mark.draw(baseline: origin, in: context)
        context.endTransparencyLayer()
        context.restoreGState()
    }

    // MARK: Data

    private static func collectShots(_ analysis: MatchAnalysis) -> [ShotEventV2] {
        let rallyShots = analysis.rallies.flatMap(\.shots)
        let coveredFrames = Set(rallyShots.map { $0.hit.frame })
        let extraShots = analysis.hits
            .filter { !coveredFrames.contains($0.frame) }
            .map { hit in
                let segment = analysis.segments.first { abs($0.startFrame - hit.frame) <= 3 }
                return ShotEventV2(hit: hit, segment: segment)
            }
        return rallyShots + extraShots
    }

    private static let endLabel: [RallyEndReason: String] = [.inCourt: "界内", .out: "出界", .net: "触网"]
}

private extension JSONValue {
    var objectValue: [String: JSONValue]? { if case .object(let value) = self { return value }; return nil }
    var intValue: Int? { double.map(Int.init) }
}
