import CoreGraphics
import CoreText
import Foundation
import ImageIO
import ShuttlCore
import ShuttlDesign
import SwiftUI
import XCTest
@testable import ShuttlExport

/// Design-system checks for the burnt-in overlay and the export control: token colours
/// are exact, text uses the display face, and full-HD frames / panel states render to
/// /tmp/restyle_export for visual review.
final class OverlayDesignTests: XCTestCase {
    private let outputDirectory = URL(fileURLWithPath: "/tmp/restyle_export", isDirectory: true)

    func testThemeColoursAreTheExactDesignTokens() throws {
        let expected: [(CGColor, UInt32)] = [
            (ExportTheme.ink, 0x221B2E), (ExportTheme.card, 0xFFFBF4), (ExportTheme.paper, 0xF4EEE3),
            (ExportTheme.tomato, 0xF2542D), (ExportTheme.cobalt, 0x2F4BDE), (ExportTheme.pistachio, 0xB9D66A),
            (ExportTheme.lilac, 0xC7B6F2), (ExportTheme.butter, 0xF7D95C), (ExportTheme.pink, 0xFF7AB6),
        ]
        for (color, hex) in expected {
            XCTAssertEqual(try renderedRGB(color), hex, String(format: "expected %06X", hex))
        }
        XCTAssertEqual(try renderedRGB(ExportTheme.shotColor("smash")), 0xF2542D)
        XCTAssertEqual(try renderedRGB(ExportTheme.shotColor("lift")), 0x2F4BDE)
        XCTAssertEqual(ExportTheme.shotName(.kill), ShotGlyphKind.kill.chineseName)
        XCTAssertNil(ExportTheme.shotName(.unknown))
    }

    func testOverlayTextUsesTheDisplayFace() {
        let font = ExportTheme.displayFont(24)
        XCTAssertEqual(CTFontGetSize(font), 24)
        let run = TextRun([("扣杀", font, ExportTheme.ink), ("312", ExportTheme.displayFont(31), ExportTheme.ink)])
        XCTAssertGreaterThan(run.width, 40)
        XCTAssertGreaterThan(run.ascent, 0)
    }

    func testTrailRunsLilacToTomatoToButter() throws {
        XCTAssertEqual(try renderedRGB(OverlayRenderer.trailColor(0)), 0xC7B6F2)
        XCTAssertEqual(try renderedRGB(OverlayRenderer.trailColor(0.55)), 0xF2542D)
        XCTAssertEqual(try renderedRGB(OverlayRenderer.trailColor(1)), 0xF7D95C)
    }

    func testFullHDFramesRenderForReview() throws {
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        let scene = OverlayScene()
        let renderer = OverlayRenderer(analysis: scene.analysis, options: ExportOptions(), width: 1920, height: 1080)
        for (name, frame) in [("frame.png", 96), ("frame-landing.png", 110), ("frame-clear.png", 62)] {
            let context = try XCTUnwrap(scene.backgroundContext())
            renderer.draw(frame: frame, in: context)
            let image = try XCTUnwrap(context.makeImage())
            try writePNG(image, to: outputDirectory.appendingPathComponent(name))
        }
        // The smash frame carries butter at the shuttle and the tomato smash sticker near the hit.
        let context = try XCTUnwrap(scene.backgroundContext())
        renderer.draw(frame: 96, in: context)
        XCTAssertTrue(try colours(in: context, around: scene.pixel(ofFrame: 96), radius: 5).contains(0xF7D95C),
                      "butter trail head")
        XCTAssertTrue(try colours(in: context, around: scene.pixel(ofFrame: 84), radius: 260).contains(0xF2542D),
                      "tomato smash sticker")
        XCTAssertTrue(try colours(in: context, around: CGPoint(x: 150, y: 90), radius: 60).contains(0x2F4BDE),
                      "cobalt near-score chip")
    }

    @MainActor
    func testExportPanelStatesRenderForReview() throws {
        DisplayFace.register()
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        let states: [(ExportPhase, ExportSaveStatus)] = [
            (.idle, .none), (.exporting(0.42), .none), (.done, .none), (.done, .saved),
            (.done, .failed("没有照片图库添加权限，请在系统设置中允许 Shuttl 保存视频。")),
            (.failed("读取视频失败：文件已被移动"), .none),
        ]
        let sheet = VStack(alignment: .leading, spacing: 28) {
            ForEach(states.indices, id: \.self) { i in
                ExportPanel(phase: states[i].0, saveStatus: states[i].1, start: {}, save: {}) {
                    Button {} label: { ExportShareLabel() }
                        .buttonStyle(PrintButtonStyle(shape: Capsule(), plate: .ink, depth: 4))
                }
            }
        }
        .padding(20)
        .frame(width: 390)
        .background(Color.paper)
        let renderer = ImageRenderer(content: sheet)
        renderer.scale = 2
        let image = try XCTUnwrap(renderer.cgImage)
        try writePNG(image, to: outputDirectory.appendingPathComponent("panel.png"))
    }

    // MARK: Helpers

    /// Paints `color` into a 1×1 sRGB bitmap and reads back 0xRRGGBB.
    private func renderedRGB(_ color: CGColor) throws -> UInt32 {
        let context = try XCTUnwrap(CGContext(data: nil, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
            space: ExportTheme.sRGB, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        context.setFillColor(color)
        context.fill(CGRect(x: 0, y: 0, width: 1, height: 1))
        let p = try XCTUnwrap(context.data).assumingMemoryBound(to: UInt8.self)
        return UInt32(p[0]) << 16 | UInt32(p[1]) << 8 | UInt32(p[2])
    }

    /// Opaque colours (0xRRGGBB) in a square around a y-down image point of a
    /// 32BGRA little-endian context.
    private func colours(in context: CGContext, around point: CGPoint, radius: Int) throws -> Set<UInt32> {
        let data = try XCTUnwrap(context.data).assumingMemoryBound(to: UInt8.self)
        var found = Set<UInt32>()
        let cx = Int(point.x), cy = Int(point.y)
        for y in max(0, cy - radius)..<min(context.height, cy + radius) {
            for x in max(0, cx - radius)..<min(context.width, cx + radius) {
                let p = data + y * context.bytesPerRow + x * 4
                found.insert(UInt32(p[2]) << 16 | UInt32(p[1]) << 8 | UInt32(p[0]))
            }
        }
        return found
    }

    private func writePNG(_ image: CGImage, to url: URL) throws {
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
    }
}

/// A 1080p60 rally from behind the near baseline: a clear from the near player
/// (frame 40), a smash back from the far player (frame 84) landing in at frame 104.
/// Frames 88–90 are gap-filled. Score events: 11:9 after rally 0, 11:10 after rally 1.
private struct OverlayScene {
    let camera = CameraModel.lookAt(eye: [-10.5, 0, 3.4], target: [0.5, 0, 0], focalPx: 1250,
                                    imageSize: ImageSize(width: 1920, height: 1080))
    let fps = 60.0
    let frames = 140
    let clear = (start: 40, end: 84, p0: SIMD3<Double>(-4.5, 1.2, 2.3), p1: SIMD3<Double>(4.2, -1.0, 2.6))
    let smash = (start: 84, end: 104, p1: SIMD3<Double>(-3.4, 1.9, 0))

    var analysis: MatchAnalysis {
        let t = (0..<frames).map { Double($0) / fps }
        let xy = (0..<frames).map(pixelOrNaN)
        let track = Track2D(t: t, xy: xy, conf: xy.map { $0.isFinite ? 0.9 : 0 }, visible: xy.map(\.isFinite),
                            interpolated: (0..<frames).map { (88...90).contains($0) })
        let clearHit = HitEvent(frame: clear.start, t: t[clear.start], xyPx: xy[clear.start], hitter: .near)
        let smashHit = HitEvent(frame: smash.start, t: t[smash.start], xyPx: xy[smash.start], hitter: .far)
        let clearSeg = segment(clear.start, clear.end, p0: clear.p0, v0: velocity(clear.p0, clear.p1, clear.start, clear.end))
        let smashV = SIMD3<Double>(-55, 15, -60)                  // |v| ≈ 82.8 m/s → 298 km/h
        let smashSeg = segment(smash.start, smash.end, p0: clear.p1, v0: smashV)
        let rally = RallyV2(startFrame: clear.start, endFrame: smash.end,
                            shots: [ShotEventV2(hit: clearHit, segment: clearSeg, shotType: .clear),
                                    ShotEventV2(hit: smashHit, segment: smashSeg, shotType: .smash)],
                            endReason: .inCourt, landingXY: SIMD2(smash.p1.x, smash.p1.y), winner: .far, confidence: 0.9)
        func score(_ end: Int, _ near: Int, _ far: Int) -> JSONValue {
            .object(["end_frame": .number(Double(end)), "game": .number(1),
                     "score": .object(["near": .number(Double(near)), "far": .number(Double(far))])])
        }
        return MatchAnalysis(meta: VideoMeta(fps: fps, width: 1920, height: 1080, nFrames: frames, timestamps: t),
                             track: track, camera: camera, hits: [clearHit, smashHit],
                             segments: [clearSeg, smashSeg], rallies: [rally],
                             score: [score(20, 11, 9), score(smash.end, 11, 10)])
    }

    /// Ballistic position for a frame (no drag — the overlay only needs a plausible arc).
    func world(ofFrame f: Int) -> SIMD3<Double>? {
        if f >= clear.start && f <= clear.end {
            return ballistic(clear.p0, velocity(clear.p0, clear.p1, clear.start, clear.end), Double(f - clear.start) / fps)
        }
        if f > smash.start && f <= smash.end {
            let v = velocity(clear.p1, smash.p1, smash.start, smash.end)
            return ballistic(clear.p1, v, Double(f - smash.start) / fps)
        }
        return nil
    }

    func pixel(ofFrame f: Int) -> CGPoint {
        let p = pixelOrNaN(f)
        return CGPoint(x: p.x, y: p.y)
    }

    private func pixelOrNaN(_ f: Int) -> SIMD2<Double> {
        guard let w = world(ofFrame: f) else { return SIMD2(.nan, .nan) }
        return camera.project(w)
    }

    private func velocity(_ a: SIMD3<Double>, _ b: SIMD3<Double>, _ f0: Int, _ f1: Int) -> SIMD3<Double> {
        let dt = Double(f1 - f0) / fps
        let d = b - a
        return SIMD3(d.x / dt, d.y / dt, (d.z + 0.5 * 9.81 * dt * dt) / dt)
    }

    private func ballistic(_ p: SIMD3<Double>, _ v: SIMD3<Double>, _ t: Double) -> SIMD3<Double> {
        p + v * t + SIMD3(0, 0, -0.5 * 9.81 * t * t)
    }

    private func segment(_ f0: Int, _ f1: Int, p0: SIMD3<Double>, v0: SIMD3<Double>) -> ShotSegment3D {
        let fs = Array(f0...f1)
        return ShotSegment3D(startFrame: f0, endFrame: f1, t0: Double(f0) / fps, p0: p0, v0: v0, rmsePx: 1.5,
                             frames: fs, positions: fs.map { world(ofFrame: $0) ?? p0 }, confidence: 0.9)
    }

    /// A 32BGRA sRGB bitmap (as the exporter's pixel buffers) painted with a stand-in
    /// gym: hall, floor, court, lines, net and two player silhouettes.
    func backgroundContext() -> CGContext? {
        guard let c = CGContext(data: nil, width: 1920, height: 1080, bitsPerComponent: 8, bytesPerRow: 1920 * 4,
                                space: ExportTheme.sRGB,
                                bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue)
        else { return nil }
        c.saveGState()
        c.translateBy(x: 0, y: 1080)
        c.scaleBy(x: 1, y: -1)
        func p(_ x: Double, _ y: Double, _ z: Double = 0) -> CGPoint {
            let q = camera.project(SIMD3(x, y, z)); return CGPoint(x: q.x, y: q.y)
        }
        func quad(_ pts: [CGPoint], _ r: CGFloat, _ g: CGFloat, _ b: CGFloat, _ a: CGFloat = 1) {
            c.addLines(between: pts); c.closePath()
            c.setFillColor(CGColor(colorSpace: ExportTheme.sRGB, components: [r, g, b, a])!)
            c.fillPath()
        }
        quad([CGPoint(x: 0, y: 0), CGPoint(x: 1920, y: 0), CGPoint(x: 1920, y: 1080), CGPoint(x: 0, y: 1080)], 0.15, 0.16, 0.20)
        quad([p(-9.5, -12), p(16, -12), p(16, 12), p(-9.5, 12)], 0.27, 0.33, 0.31)
        let L = CourtModel.halfLength, W = CourtModel.doublesHalfWidth
        quad([p(-L - 1.2, -W - 1), p(L + 1.2, -W - 1), p(L + 1.2, W + 1), p(-L - 1.2, W + 1)], 0.18, 0.42, 0.34)
        c.setStrokeColor(CGColor(colorSpace: ExportTheme.sRGB, components: [0.94, 0.93, 0.87, 1])!)
        c.setLineWidth(4)
        for line in CourtModel.lines {
            c.move(to: p(line.a.x, line.a.y)); c.addLine(to: p(line.b.x, line.b.y))
        }
        c.strokePath()
        quad([p(0, -W - 0.3), p(0, W + 0.3), p(0, W + 0.3, 1.55), p(0, -W - 0.3, 1.55)], 0.9, 0.9, 0.9, 0.16)
        c.setLineWidth(5)
        c.move(to: p(0, -W - 0.3, 1.55)); c.addLine(to: p(0, W + 0.3, 1.55))
        for y in [-W - 0.3, W + 0.3] { c.move(to: p(0, y)); c.addLine(to: p(0, y, 1.55)) }
        c.strokePath()
        for (x, y) in [(-4.8, 1.3), (4.5, -1.0)] {                        // players
            let feet = p(x, y), head = p(x, y, 1.78)
            let w = abs(p(x, y + 0.28).x - p(x, y - 0.28).x)
            let body = CGRect(x: feet.x - w / 2, y: head.y, width: w, height: feet.y - head.y)
            c.addPath(CGPath(roundedRect: body, cornerWidth: w / 2, cornerHeight: w / 2, transform: nil))
            c.setFillColor(CGColor(colorSpace: ExportTheme.sRGB, components: [0.10, 0.10, 0.14, 1])!)
            c.fillPath()
        }
        c.restoreGState()
        return c
    }
}
