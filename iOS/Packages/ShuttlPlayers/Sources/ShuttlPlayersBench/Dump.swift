import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import ShuttlCore
import ShuttlPlayers
import UniformTypeIdentifiers

/// PNG of a 32BGRA frame with crop rects (colour per layout) and the projected court
/// outline of `camera` — a visual check that the emulated frame and its crops line up.
func dumpFrame(_ pb: CVPixelBuffer, rects: [(PixelRect, (CGFloat, CGFloat, CGFloat))], camera: CameraModel, to url: URL) {
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
    guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: w, height: h, bitsPerComponent: 8,
                              bytesPerRow: CVPixelBufferGetBytesPerRow(pb), space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue),
          let base = ctx.makeImage(),
          let out = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                              space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { return }
    out.draw(base, in: CGRect(x: 0, y: 0, width: w, height: h))
    out.translateBy(x: 0, y: CGFloat(h)); out.scaleBy(x: 1, y: -1)            // top-left origin like the pixels
    out.setLineWidth(3)
    for (r, c) in rects {
        out.setStrokeColor(red: c.0, green: c.1, blue: c.2, alpha: 1)
        out.stroke(CGRect(x: r.x1, y: r.y1, width: r.width, height: r.height))
    }
    out.setStrokeColor(red: 1, green: 1, blue: 0, alpha: 1)
    out.setLineWidth(2)
    for line in CourtModel.lines {
        let a = camera.project(SIMD3(line.a.x, line.a.y, 0)), b = camera.project(SIMD3(line.b.x, line.b.y, 0))
        guard a.x.isFinite, b.x.isFinite else { continue }
        out.move(to: CGPoint(x: a.x, y: a.y)); out.addLine(to: CGPoint(x: b.x, y: b.y))
    }
    out.strokePath()
    guard let img = out.makeImage(),
          let dst = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return }
    CGImageDestinationAddImage(dst, img, nil)
    CGImageDestinationFinalize(dst)
}

/// Colour of each layout's crops in the dumps.
let layoutColour: [FarCropLayout: (CGFloat, CGFloat, CGFloat)] = [
    .single: (1, 0, 0), .lateral: (0, 0.8, 1), .depth: (0, 1, 0), .none: (1, 1, 1),
]
