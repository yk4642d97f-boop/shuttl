import CoreVideo
import Foundation
import ShuttlCore
import simd

/// Geometry and SimCC decode of the CoreML RTMPose-m port.
///
/// Contract: `research/coreml/REPORT.md` section 5 (authoritative). The model input is
/// **192 wide x 256 high** (W x H), takes raw 0..255 RGB in NCHW, and its SimCC heads use a
/// split ratio of 2.0, so one bin is half a pixel. Everything here is a faithful port of
/// rtmlib's `bbox_xyxy2cs` / `top_down_affine` / `get_simcc_maximum`.
public enum RTMPoseGeometry {
    /// Model input frame: 192 wide, 256 high — NOT 256x192.
    public static let modelWidth = 192
    public static let modelHeight = 256
    /// SimCC split ratio: `pixel = argmax_bin / 2.0`.
    public static let splitRatio = 2.0
    /// The person box is grown by this fraction of its size on each side first (REPORT 5.2.1).
    public static let boxPadFraction = 0.10
    /// `bbox_xyxy2cs` padding (REPORT 5.2.2).
    public static let boxPadding = 1.25

    // MARK: Box -> centre / scale

    /// Expanded box -> centre/scale (rtmlib `bbox_xyxy2cs(padding=1.25)`), aspect-fixed to
    /// the model's 192:256 (REPORT 5.2 steps 1-3). Coordinates in frame px.
    public static func centerScale(box: BBox, imageSize: ImageSize) -> (center: SIMD2<Double>, scale: SIMD2<Double>) {
        let w = Double(imageSize.width), h = Double(imageSize.height)
        let mw = boxPadFraction * box.width, mh = boxPadFraction * box.height
        let x1 = min(max(box.x1 - mw, 0), w - 1)
        let y1 = min(max(box.y1 - mh, 0), h - 1)
        let x2 = min(max(box.x2 + mw, 0), w - 1)
        let y2 = min(max(box.y2 + mh, 0), h - 1)
        let center = SIMD2((x1 + x2) / 2, (y1 + y2) / 2)
        var sw = (x2 - x1) * boxPadding
        var sh = (y2 - y1) * boxPadding
        let ar = Double(modelWidth) / Double(modelHeight)
        if sw > sh * ar { sh = sw / ar } else { sw = sh * ar }
        return (center, SIMD2(sw, sh))
    }

    // MARK: Affine warp matrix (rtmlib get_warp_matrix, rot = 0)

    /// Returns the 2x3 matrix `(m00, m01, m02, m10, m11, m12)` mapping frame px to the model
    /// frame, from the same three-point correspondence as `get_warp_matrix`.
    public static func warpMatrix(center: SIMD2<Double>, scale: SIMD2<Double>) -> [Double] {
        let srcW = scale.x
        let dstW = Double(modelWidth), dstH = Double(modelHeight)
        let s0 = SIMD2<Double>(center.x, center.y)
        let s1 = SIMD2<Double>(center.x, center.y - srcW * 0.5)
        let s2 = thirdPoint(s0, s1)
        let d0 = SIMD2<Double>(dstW * 0.5, dstH * 0.5)
        let d1 = SIMD2<Double>(dstW * 0.5, dstH * 0.5 - dstW * 0.5)
        let d2 = thirdPoint(d0, d1)
        return affineTransform(src: [s0, s1, s2], dst: [d0, d1, d2])
    }

    /// rtmlib `_get_3rd_point`: `b + perpendicular(a - b)`.
    @inline(__always)
    static func thirdPoint(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> SIMD2<Double> {
        let direct = a - b
        return b + SIMD2(-direct.y, direct.x)
    }

    /// cv2.getAffineTransform: solve the 2x3 matrix taking the three `src` points to `dst`.
    static func affineTransform(src: [SIMD2<Double>], dst: [SIMD2<Double>]) -> [Double] {
        let (x0, y0) = (src[0].x, src[0].y)
        let (x1, y1) = (src[1].x, src[1].y)
        let (x2, y2) = (src[2].x, src[2].y)
        let det = x0 * (y1 - y2) - y0 * (x1 - x2) + (x1 * y2 - x2 * y1)
        precondition(abs(det) > 1e-12, "degenerate affine source points")
        func solve(_ u0: Double, _ u1: Double, _ u2: Double) -> (Double, Double, Double) {
            let a = (u0 * (y1 - y2) - y0 * (u1 - u2) + (u1 * y2 - u2 * y1)) / det
            let b = (x0 * (u1 - u2) - u0 * (x1 - x2) + (x1 * u2 - x2 * u1)) / det
            let c = (x0 * (y1 * u2 - y2 * u1) - y0 * (x1 * u2 - x2 * u1) + u0 * (x1 * y2 - x2 * y1)) / det
            return (a, b, c)
        }
        let (a, b, c) = solve(dst[0].x, dst[1].x, dst[2].x)
        let (d, e, f) = solve(dst[0].y, dst[1].y, dst[2].y)
        return [a, b, c, d, e, f]
    }

    // MARK: Frame -> model input tensor

    /// Warp `src` to the 192x256 model frame and pack it as NCHW float32 **RGB, raw 0..255**
    /// (no /255, no mean subtraction — the graph normalises). The warp is cv2
    /// `warpAffine(..., INTER_LINEAR)` reproduced in OpenCV's fixed point (AB_BITS 10,
    /// INTER_BITS 5), the same algorithm as ShuttlDetector's `CVExact.warpAffineLinear`, so the
    /// pixels match what the reference fixtures were generated from.
    public static func modelInput(src: PosePixelView, matrix: [Double]) -> [Float] {
        let W = modelWidth, H = modelHeight
        var rgb = [UInt8](repeating: 0, count: W * H * 3)
        var d = matrix[0] * matrix[4] - matrix[1] * matrix[3]
        d = d != 0 ? 1.0 / d : 0
        let i0 = matrix[4] * d, i4 = matrix[0] * d, i1 = -matrix[1] * d, i3 = -matrix[3] * d
        let b1 = -i0 * matrix[2] - i1 * matrix[5], b2 = -i3 * matrix[2] - i4 * matrix[5]
        let abScale = 1024.0, roundDelta = 16
        var adelta = [Int](repeating: 0, count: W), bdelta = [Int](repeating: 0, count: W)
        for x in 0..<W {
            adelta[x] = Int((i0 * Double(x) * abScale).rounded(.toNearestOrEven))
            bdelta[x] = Int((i3 * Double(x) * abScale).rounded(.toNearestOrEven))
        }
        let w = src.width, h = src.height, rb = src.rowBytes, ps = src.pixelStride
        let (cr, cg, cb) = (src.r, src.g, src.b)
        let base = src.base
        let stride = W * 3
        adelta.withUnsafeBufferPointer { ad in
            bdelta.withUnsafeBufferPointer { bd in
                rgb.withUnsafeMutableBufferPointer { out in
                    for y in 0..<H {
                        let x0 = Int(((i1 * Double(y) + b1) * abScale).rounded(.toNearestOrEven)) + roundDelta
                        let y0 = Int(((i4 * Double(y) + b2) * abScale).rounded(.toNearestOrEven)) + roundDelta
                        var o = y * stride
                        for x in 0..<W {
                            let X = (x0 + ad[x]) >> 5, Y = (y0 + bd[x]) >> 5
                            let sx = X >> 5, sy = Y >> 5, fx = X & 31, fy = Y & 31
                            let w00 = (32 - fx) * (32 - fy) * 32, w01 = fx * (32 - fy) * 32
                            let w10 = (32 - fx) * fy * 32, w11 = fx * fy * 32
                            if sx >= 0 && sy >= 0 && sx + 1 < w && sy + 1 < h {
                                let p0 = base + sy * rb + sx * ps, p1 = p0 + rb
                                out[o] = UInt8(truncatingIfNeeded: (Int(p0[cr]) * w00 + Int(p0[cr + ps]) * w01 + Int(p1[cr]) * w10 + Int(p1[cr + ps]) * w11 + 16384) >> 15)
                                out[o + 1] = UInt8(truncatingIfNeeded: (Int(p0[cg]) * w00 + Int(p0[cg + ps]) * w01 + Int(p1[cg]) * w10 + Int(p1[cg + ps]) * w11 + 16384) >> 15)
                                out[o + 2] = UInt8(truncatingIfNeeded: (Int(p0[cb]) * w00 + Int(p0[cb + ps]) * w01 + Int(p1[cb]) * w10 + Int(p1[cb + ps]) * w11 + 16384) >> 15)
                            } else if sx >= w || sx + 1 < 0 || sy >= h || sy + 1 < 0 {
                                out[o] = 0; out[o + 1] = 0; out[o + 2] = 0
                            } else {
                                // Border: taps outside the image read the constant 0 (cv2 BORDER_CONSTANT).
                                let in0x = sx >= 0 && sx < w, in1x = sx + 1 >= 0 && sx + 1 < w
                                let in0y = sy >= 0 && sy < h, in1y = sy + 1 >= 0 && sy + 1 < h
                                for (k, c) in [cr, cg, cb].enumerated() {
                                    var acc = 0
                                    if in0y && in0x { acc += Int(base[sy * rb + sx * ps + c]) * w00 }
                                    if in0y && in1x { acc += Int(base[sy * rb + (sx + 1) * ps + c]) * w01 }
                                    if in1y && in0x { acc += Int(base[(sy + 1) * rb + sx * ps + c]) * w10 }
                                    if in1y && in1x { acc += Int(base[(sy + 1) * rb + (sx + 1) * ps + c]) * w11 }
                                    out[o + k] = UInt8(truncatingIfNeeded: (acc + 16384) >> 15)
                                }
                            }
                            o += 3
                        }
                    }
                }
            }
        }
        return packedNCHW(rgb)
    }

    /// Interleaved RGB uint8 (row stride `modelWidth*3`) -> NCHW float32 raw 0..255.
    public static func packedNCHW(_ rgb: [UInt8]) -> [Float] {
        let W = modelWidth, H = modelHeight, plane = W * H
        var out = [Float](repeating: 0, count: 3 * plane)
        rgb.withUnsafeBufferPointer { p in
            out.withUnsafeMutableBufferPointer { o in
                var s = 0
                for i in 0..<plane {
                    o[i] = Float(p[s]); o[plane + i] = Float(p[s + 1]); o[2 * plane + i] = Float(p[s + 2])
                    s += 3
                }
            }
        }
        return out
    }

    // MARK: SimCC decode (REPORT 5.3)

    /// `simcc_x` (17x384) / `simcc_y` (17x512) logits -> 17 COCO keypoints in frame px.
    /// Confidence is `(max(simcc_x) + max(simcc_y)) / 2`; a keypoint whose confidence is <= 0
    /// is invalid and gets location (-1, -1), exactly as rtmlib gates it.
    public static func decode(simccX: UnsafePointer<Float>, simccY: UnsafePointer<Float>,
                              center: SIMD2<Double>, scale: SIMD2<Double>) -> [Keypoint] {
        let binsX = modelWidth * 2   // 384
        let binsY = modelHeight * 2  // 512
        var out: [Keypoint] = []
        out.reserveCapacity(17)
        for k in 0..<17 {
            let rx = simccX + k * binsX
            let ry = simccY + k * binsY
            var lx = 0, ly = 0
            var mx = -Float.infinity, my = -Float.infinity
            for i in 0..<binsX where rx[i] > mx { mx = rx[i]; lx = i }
            for i in 0..<binsY where ry[i] > my { my = ry[i]; ly = i }
            let conf = Double(mx + my) / 2.0
            if conf <= 0 {
                out.append(Keypoint(x: -1, y: -1, conf: conf))
                continue
            }
            let px = Double(lx) / splitRatio
            let py = Double(ly) / splitRatio
            let x = px / Double(modelWidth) * scale.x + center.x - scale.x / 2
            let y = py / Double(modelHeight) * scale.y + center.y - scale.y / 2
            out.append(Keypoint(x: x, y: y, conf: conf))
        }
        return out
    }

    /// Array convenience (tests, reference checks).
    public static func decode(simccX: [Float], simccY: [Float],
                              center: SIMD2<Double>, scale: SIMD2<Double>) -> [Keypoint] {
        simccX.withUnsafeBufferPointer { px in
            simccY.withUnsafeBufferPointer { py in
                decode(simccX: px.baseAddress!, simccY: py.baseAddress!, center: center, scale: scale)
            }
        }
    }
}

/// Read-only view of an interleaved 8-bit image the RTMPose warp reads. `r`, `g`, `b` are the
/// byte offsets of the channels within one pixel (32BGRA buffers: 2, 1, 0; RGB images: 0, 1, 2).
public struct PosePixelView: @unchecked Sendable {
    public let base: UnsafePointer<UInt8>
    public let width: Int, height: Int, rowBytes: Int, pixelStride: Int
    public let r: Int, g: Int, b: Int

    public init(base: UnsafePointer<UInt8>, width: Int, height: Int, rowBytes: Int,
                pixelStride: Int, r: Int, g: Int, b: Int) {
        self.base = base; self.width = width; self.height = height
        self.rowBytes = rowBytes; self.pixelStride = pixelStride
        self.r = r; self.g = g; self.b = b
    }

    /// Calls `body` with a view of a locked 32BGRA pixel buffer.
    public static func withBGRA<T>(_ pb: CVPixelBuffer, _ body: (PosePixelView) throws -> T) throws -> T {
        let fmt = CVPixelBufferGetPixelFormatType(pb)
        guard fmt == kCVPixelFormatType_32BGRA else { throw ShuttlCoreError.unsupportedPixelFormat(fmt) }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else {
            throw ShuttlCoreError.readerFailed("pixel buffer has no base address")
        }
        let v = PosePixelView(base: base.assumingMemoryBound(to: UInt8.self),
                              width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb),
                              rowBytes: CVPixelBufferGetBytesPerRow(pb), pixelStride: 4, r: 2, g: 1, b: 0)
        return try body(v)
    }
}
