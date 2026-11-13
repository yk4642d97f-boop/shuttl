import Accelerate
import CoreMedia
import CoreVideo
import Foundation
import ShuttlCore

/// One sampled camera frame at the calibration working resolution.
public struct WorkingFrame: Sendable {
    /// RGB, `workingSize`.
    public var image: ImageU8
    /// Upright size of the camera frame it came from.
    public var fullSize: ImageSize
    public var time: Double
    public var workingSize: ImageSize { image.size }
    /// Luma statistics (0…255) and a small grey thumbnail for motion checks.
    public var luma: LumaStats
    public var thumb: GrayThumb
}

public struct LumaStats: Sendable, Equatable {
    public var mean: Double
    public var p10: Double
    public var p90: Double
}

/// Small grey thumbnail (≈ 64 px wide) used to detect camera motion between samples.
public struct GrayThumb: Sendable {
    public var width: Int
    public var height: Int
    public var data: [Float]
}

/// Upright camera frames → RGB working frames (box / area downscale, the resampling the
/// calibrator's cv2 INTER_AREA path expects). Not thread-safe: one instance per queue.
public final class FrameDownsampler {
    /// Default long side of the working frame (0 = keep full resolution).
    public let workingLongSide: Int
    private var yuvInfo: [String: vImage_YpCbCrToARGB] = [:]

    public init(workingLongSide: Int = 0) { self.workingLongSide = workingLongSide }

    /// Working size for a full-resolution upright size: prefers an exact integer divisor
    /// (1920x1080 → 640x360 for 640), else rounds the aspect-preserving size.
    public static func workingSize(for full: ImageSize, longSide: Int) -> ImageSize {
        let L = max(full.width, full.height)
        guard longSide > 0, L > longSide else { return full }
        let k = Int((Double(L) / Double(longSide)).rounded())
        if k >= 2, full.width % k == 0, full.height % k == 0, abs(L / k - longSide) <= longSide / 8 {
            return ImageSize(width: full.width / k, height: full.height / k)
        }
        let s = Double(longSide) / Double(L)
        return ImageSize(width: max(1, Int((Double(full.width) * s).rounded())),
                         height: max(1, Int((Double(full.height) * s).rounded())))
    }

    // MARK: Entry points

    public func frame(from image: ImageU8, time: Double) -> WorkingFrame {
        frame(from: image, workingSize: Self.workingSize(for: image.size, longSide: workingLongSide), time: time)
    }

    public func frame(from image: ImageU8, workingSize ws: ImageSize, time: Double) -> WorkingFrame {
        precondition(image.channels == 3, "RGB ImageU8 expected")
        let small: ImageU8
        if ws == image.size {
            small = image
        } else if let k = Self.integerFactor(image.size, ws) {
            small = image.data.withUnsafeBytes { raw in
                Self.boxDown(raw.baseAddress!, rowBytes: image.width * 3, pixelStride: 3, r: 0, g: 1, b: 2,
                             outW: ws.width, outH: ws.height, k: k)
            }
        } else {
            small = Self.areaResize(image, to: ws)
        }
        return Self.finish(small, full: image.size, time: time)
    }

    /// 32BGRA or 420YpCbCr8BiPlanar (video / full range) upright pixel buffer.
    public func frame(from pb: CVPixelBuffer, time: Double) throws -> WorkingFrame {
        let full = ImageSize(width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb))
        return try frame(from: pb, workingSize: Self.workingSize(for: full, longSide: workingLongSide), time: time)
    }

    public func frame(from pb: CVPixelBuffer, workingSize ws: ImageSize, time: Double) throws -> WorkingFrame {
        let fmt = CVPixelBufferGetPixelFormatType(pb)
        let full = ImageSize(width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb))
        switch fmt {
        case kCVPixelFormatType_32BGRA:
            CVPixelBufferLockBaseAddress(pb, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
            guard let base = CVPixelBufferGetBaseAddress(pb) else { throw ShuttlCoreError.readerFailed("no base address") }
            let rb = CVPixelBufferGetBytesPerRow(pb)
            if let k = Self.integerFactor(full, ws) ?? (ws == full ? 1 : nil) {
                let img = Self.boxDown(UnsafeRawPointer(base), rowBytes: rb, pixelStride: 4, r: 2, g: 1, b: 0,
                                       outW: ws.width, outH: ws.height, k: k)
                return Self.finish(img, full: full, time: time)
            }
            let img = Self.boxDown(UnsafeRawPointer(base), rowBytes: rb, pixelStride: 4, r: 2, g: 1, b: 0,
                                   outW: full.width, outH: full.height, k: 1)
            return Self.finish(Self.areaResize(img, to: ws), full: full, time: time)
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
            return try yuvFrame(pb, fullRange: fmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                                full: full, ws: ws, time: time)
        default:
            throw ShuttlCoreError.unsupportedPixelFormat(fmt)
        }
    }

    // MARK: YpCbCr

    private func conversionInfo(fullRange: Bool, bt601: Bool) throws -> vImage_YpCbCrToARGB {
        let key = "\(fullRange)-\(bt601)"
        if let info = yuvInfo[key] { return info }
        var range = fullRange
            ? vImage_YpCbCrPixelRange(Yp_bias: 0, CbCr_bias: 128, YpRangeMax: 255, CbCrRangeMax: 255,
                                      YpMax: 255, YpMin: 0, CbCrMax: 255, CbCrMin: 0)
            : vImage_YpCbCrPixelRange(Yp_bias: 16, CbCr_bias: 128, YpRangeMax: 235, CbCrRangeMax: 240,
                                      YpMax: 235, YpMin: 16, CbCrMax: 240, CbCrMin: 16)
        var info = vImage_YpCbCrToARGB()
        // Local copies of kvImage_YpCbCrToARGBMatrix_ITU_R_601_4 / _709_2 (those globals are
        // mutable C vars, not concurrency-safe to reference).
        var matrix = bt601
            ? vImage_YpCbCrToARGBMatrix(Yp: 1, Cr_R: 1.402, Cr_G: -0.714136286, Cb_G: -0.344136286, Cb_B: 1.772)
            : vImage_YpCbCrToARGBMatrix(Yp: 1, Cr_R: 1.5748, Cr_G: -0.468124273, Cb_G: -0.187324273, Cb_B: 1.8556)
        let err = vImageConvert_YpCbCrToARGB_GenerateConversion(&matrix, &range, &info, kvImage420Yp8_CbCr8,
                                                                kvImageARGB8888, vImage_Flags(kvImageNoFlags))
        guard err == kvImageNoError else { throw ShuttlCoreError.vImage(err) }
        yuvInfo[key] = info
        return info
    }

    private func yuvFrame(_ pb: CVPixelBuffer, fullRange: Bool, full: ImageSize, ws: ImageSize,
                          time: Double) throws -> WorkingFrame {
        let bt601: Bool = {
            guard let m = CVBufferCopyAttachment(pb, kCVImageBufferYCbCrMatrixKey, nil) as? String else { return false }
            return m == (kCVImageBufferYCbCrMatrix_ITU_R_601_4 as String)
        }()
        var info = try conversionInfo(fullRange: fullRange, bt601: bt601)
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let yBase = CVPixelBufferGetBaseAddressOfPlane(pb, 0),
              let cBase = CVPixelBufferGetBaseAddressOfPlane(pb, 1) else { throw ShuttlCoreError.readerFailed("no planes") }
        var yBuf = vImage_Buffer(data: yBase, height: vImagePixelCount(full.height), width: vImagePixelCount(full.width),
                                 rowBytes: CVPixelBufferGetBytesPerRowOfPlane(pb, 0))
        var cBuf = vImage_Buffer(data: cBase, height: vImagePixelCount(full.height / 2),
                                 width: vImagePixelCount(full.width / 2), rowBytes: CVPixelBufferGetBytesPerRowOfPlane(pb, 1))
        let argbRB = full.width * 4
        var argb = [UInt8](repeating: 0, count: argbRB * full.height)
        let img: ImageU8 = try argb.withUnsafeMutableBytes { raw in
            var dst = vImage_Buffer(data: raw.baseAddress, height: vImagePixelCount(full.height),
                                    width: vImagePixelCount(full.width), rowBytes: argbRB)
            let permute: [UInt8] = [0, 1, 2, 3]            // memory order A R G B
            let err = vImageConvert_420Yp8_CbCr8ToARGB8888(&yBuf, &cBuf, &dst, &info, permute, 255,
                                                          vImage_Flags(kvImageNoFlags))
            guard err == kvImageNoError else { throw ShuttlCoreError.vImage(err) }
            let k = Self.integerFactor(full, ws) ?? 1
            let outW = k == 1 ? full.width : ws.width, outH = k == 1 ? full.height : ws.height
            return Self.boxDown(UnsafeRawPointer(raw.baseAddress!), rowBytes: argbRB, pixelStride: 4, r: 1, g: 2, b: 3,
                                outW: outW, outH: outH, k: k)
        }
        let small = img.size == ws ? img : Self.areaResize(img, to: ws)
        return Self.finish(small, full: full, time: time)
    }

    // MARK: Kernels

    /// Bit-identical to ShuttlCore `ImageU8.resized(width:height:method: .area)` when shrinking
    /// (same box weights, same Float accumulation order, same rounding) but on raw pointers:
    /// ≈ 20× faster, which makes a non-integer 1920x1080 → 1440x810 affordable per sample.
    public static func areaResize(_ img: ImageU8, to ws: ImageSize) -> ImageU8 {
        let W = img.width, H = img.height, C = img.channels, w = ws.width, h = ws.height
        let sx = Double(W) / Double(w), sy = Double(H) / Double(h)
        guard sx >= 1, sy >= 1, ws != img.size else { return img.resized(width: w, height: h, method: .area) }
        func weights(_ n: Int, _ srcN: Int, _ scale: Double) -> (idx: [Int32], wt: [Float], start: [Int]) {
            var idx: [Int32] = [], wt: [Float] = [], start: [Int] = [0]
            for d in 0..<n {
                let a = Double(d) * scale, b = Swift.min(Double(d + 1) * scale, Double(srcN))
                var i = Int(a.rounded(.down))
                while Double(i) < b - 1e-12 {
                    let lo = Swift.max(a, Double(i)), hi = Swift.min(b, Double(i + 1))
                    if hi > lo { idx.append(Int32(Swift.min(i, srcN - 1))); wt.append(Float((hi - lo) / (b - a))) }
                    i += 1
                }
                start.append(idx.count)
            }
            return (idx, wt, start)
        }
        let xw = weights(w, W, sx), yw = weights(h, H, sy)
        var tmp = [Float](repeating: 0, count: w * H * C)
        var out = [UInt8](repeating: 0, count: w * h * C)
        img.data.withUnsafeBufferPointer { src in
            tmp.withUnsafeMutableBufferPointer { t in
                for y in 0..<H {
                    let srow = y * W * C, trow = y * w * C
                    for dx in 0..<w {
                        for c in 0..<C {
                            var s: Float = 0
                            for k in xw.start[dx]..<xw.start[dx + 1] {
                                s += Float(src[srow + Int(xw.idx[k]) * C + c]) * xw.wt[k]
                            }
                            t[trow + dx * C + c] = s
                        }
                    }
                }
            }
        }
        tmp.withUnsafeBufferPointer { t in
            out.withUnsafeMutableBufferPointer { o in
                for dy in 0..<h {
                    let a = yw.start[dy], b = yw.start[dy + 1]
                    for x in 0..<(w * C) {
                        var s: Float = 0
                        for k in a..<b { s += t[Int(yw.idx[k]) * w * C + x] * yw.wt[k] }
                        o[dy * w * C + x] = UInt8(Swift.min(Swift.max(s.rounded(), 0), 255))
                    }
                }
            }
        }
        return ImageU8(width: w, height: h, channels: C, data: out)
    }

    static func integerFactor(_ full: ImageSize, _ ws: ImageSize) -> Int? {
        guard ws.width > 0, ws.height > 0, full.width % ws.width == 0, full.height % ws.height == 0 else { return nil }
        let kx = full.width / ws.width, ky = full.height / ws.height
        return kx == ky && kx >= 1 ? kx : nil
    }

    /// k×k box average (rounded) of an interleaved 8-bit source → RGB ImageU8 outW×outH.
    static func boxDown(_ src: UnsafeRawPointer, rowBytes: Int, pixelStride: Int, r: Int, g: Int, b: Int,
                        outW: Int, outH: Int, k: Int) -> ImageU8 {
        var out = [UInt8](repeating: 0, count: outW * outH * 3)
        let s = src.assumingMemoryBound(to: UInt8.self)
        let n = UInt32(k * k), half = n / 2
        out.withUnsafeMutableBufferPointer { o in
            for oy in 0..<outH {
                for ox in 0..<outW {
                    var sr: UInt32 = 0, sg: UInt32 = 0, sb: UInt32 = 0
                    for dy in 0..<k {
                        let row = s + (oy * k + dy) * rowBytes + ox * k * pixelStride
                        for dx in 0..<k {
                            let p = row + dx * pixelStride
                            sr += UInt32(p[r]); sg += UInt32(p[g]); sb += UInt32(p[b])
                        }
                    }
                    let i = (oy * outW + ox) * 3
                    o[i] = UInt8((sr + half) / n); o[i + 1] = UInt8((sg + half) / n); o[i + 2] = UInt8((sb + half) / n)
                }
            }
        }
        return ImageU8(width: outW, height: outH, channels: 3, data: out)
    }

    static func finish(_ img: ImageU8, full: ImageSize, time: Double) -> WorkingFrame {
        WorkingFrame(image: img, fullSize: full, time: time, luma: lumaStats(img), thumb: thumbnail(img))
    }

    /// BT.601 luma histogram → mean / 10th / 90th percentile.
    static func lumaStats(_ img: ImageU8) -> LumaStats {
        var hist = [Int](repeating: 0, count: 256)
        var total = 0.0
        let n = img.width * img.height
        img.data.withUnsafeBufferPointer { d in
            var i = 0
            while i < n * 3 {
                let y = (77 * Int(d[i]) + 150 * Int(d[i + 1]) + 29 * Int(d[i + 2]) + 128) >> 8
                hist[y] += 1; total += Double(y)
                i += 3 * 2                                   // every other pixel is plenty
            }
        }
        let cnt = hist.reduce(0, +)
        func pct(_ q: Double) -> Double {
            let target = Int(Double(cnt) * q)
            var acc = 0
            for (v, h) in hist.enumerated() { acc += h; if acc > target { return Double(v) } }
            return 255
        }
        return LumaStats(mean: cnt > 0 ? total / Double(cnt) : 0, p10: pct(0.1), p90: pct(0.9))
    }

    /// ≈64-px-wide grey thumbnail by block averaging.
    static func thumbnail(_ img: ImageU8, width tw: Int = 64) -> GrayThumb {
        let w = max(1, min(tw, img.width)), h = max(1, Int((Double(img.height) * Double(w) / Double(img.width)).rounded()))
        var out = [Float](repeating: 0, count: w * h)
        var cnt = [Float](repeating: 0, count: w * h)
        img.data.withUnsafeBufferPointer { d in
            for y in 0..<img.height {
                let ty = min(h - 1, y * h / img.height)
                for x in 0..<img.width {
                    let tx = min(w - 1, x * w / img.width)
                    let i = (y * img.width + x) * 3
                    let lum = 0.299 * Float(d[i]) + 0.587 * Float(d[i + 1]) + 0.114 * Float(d[i + 2])
                    out[ty * w + tx] += lum; cnt[ty * w + tx] += 1
                }
            }
        }
        for i in out.indices where cnt[i] > 0 { out[i] /= cnt[i] }
        return GrayThumb(width: w, height: h, data: out)
    }
}
