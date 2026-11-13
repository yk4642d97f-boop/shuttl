import CoreVideo
import Foundation
import ShuttlCore

/// Read access to an upright colour frame for the appearance descriptors.
/// Python works on OpenCV BGR frames; implementations hand out (r, g, b) so the
/// HSV conversion below can apply OpenCV's BGR2HSV exactly.
public protocol RGBPixelSource {
    var width: Int { get }
    var height: Int { get }
    /// Visit every pixel of rows `y0..<y1`, columns `x0..<x1` (already clipped).
    func forEachPixel(x0: Int, x1: Int, y0: Int, y1: Int, _ body: (_ r: Int, _ g: Int, _ b: Int) -> Void)
}

extension ImageBuffer: RGBPixelSource where Pixel == UInt8 {
    /// Interleaved RGB (3 channels) — ShuttlCore's `ImageU8.rgb(from:)` layout.
    public func forEachPixel(x0: Int, x1: Int, y0: Int, y1: Int, _ body: (Int, Int, Int) -> Void) {
        precondition(channels >= 3, "appearance needs an RGB image")
        data.withUnsafeBufferPointer { p in
            for y in y0..<y1 {
                var o = (y * width + x0) * channels
                for _ in x0..<x1 {
                    body(Int(p[o]), Int(p[o + 1]), Int(p[o + 2]))
                    o += channels
                }
            }
        }
    }
}

/// A locked 32BGRA pixel buffer (no copy). Use `BGRAPixels.with(_:_:)`.
public struct BGRAPixels: RGBPixelSource {
    public let width: Int
    public let height: Int
    let base: UnsafeRawPointer
    let rowBytes: Int

    public static func with<T>(_ pb: CVPixelBuffer, _ body: (BGRAPixels) throws -> T) throws -> T {
        let fmt = CVPixelBufferGetPixelFormatType(pb)
        guard fmt == kCVPixelFormatType_32BGRA else { throw ShuttlCoreError.unsupportedPixelFormat(fmt) }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { throw ShuttlCoreError.readerFailed("no base address") }
        return try body(BGRAPixels(width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb),
                                   base: UnsafeRawPointer(base), rowBytes: CVPixelBufferGetBytesPerRow(pb)))
    }

    public func forEachPixel(x0: Int, x1: Int, y0: Int, y1: Int, _ body: (Int, Int, Int) -> Void) {
        for y in y0..<y1 {
            let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: UInt8.self)
            var o = x0 * 4
            for _ in x0..<x1 {
                body(Int(row[o + 2]), Int(row[o + 1]), Int(row[o]))
                o += 4
            }
        }
    }
}

/// OpenCV `cvtColor(BGR2HSV)` for 8-bit images (H in 0..<180), bit-exact with
/// opencv-python 4.11 (verified over the full RGB cube): hue through OpenCV's
/// 12-bit fixed-point tables, saturation as float32 `floor(diff·255 · (1/v) + 0.5)`.
public enum OpenCVHSV {
    static let hdiv180: [Int32] = (0..<256).map { i in
        i == 0 ? 0 : Int32(((Double(180 << 12)) / (6.0 * Double(i))).rounded(.toNearestOrEven))
    }
    /// S for (v, diff) pairs, index v * 256 + diff.
    static let sTable: [UInt8] = {
        var t = [UInt8](repeating: 0, count: 256 * 256)
        for v in 1..<256 {
            let inv: Float = 1 / Float(v)
            for d in 0...v {
                let p: Float = Float(d) * 255                 // exact
                let q: Float = p * inv                        // rounded product (no FMA)
                let s: Float = (q + 0.5).rounded(.down)
                t[v * 256 + d] = UInt8(min(max(s, 0), 255))
            }
        }
        return t
    }()

    @inline(__always)
    public static func hsv(r: Int, g: Int, b: Int) -> (h: Int, s: Int, v: Int) {
        let v = max(b, g, r), vmin = min(b, g, r)
        let diff = v - vmin
        let vr = v == r ? -1 : 0, vg = v == g ? -1 : 0
        let s = Int(sTable[v * 256 + diff])
        var h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + (~vg & (r - g + 4 * diff))))
        h = (h * Int(hdiv180[diff]) + (1 << 11)) >> 12
        h += h < 0 ? 180 : 0
        return (min(max(h, 0), 255), s, v)
    }
}

public enum AppearanceDescriptor {
    /// Normalised HSV histogram of a region (players._region_hist), float32.
    public static func regionHist<S: RGBPixelSource>(_ img: S, _ x1: Double, _ y1: Double,
                                                     _ x2: Double, _ y2: Double) -> [Float] {
        let W = img.width, H = img.height
        guard x1.isFinite, y1.isFinite, x2.isFinite, y2.isFinite else { return [Float](repeating: 0, count: Appearance.region) }
        let a = Int(pyMax(0, pyMin(x1, x2))), b = Int(pyMin(Double(W), pyMax(x1, x2) + 1))
        let c = Int(pyMax(0, pyMin(y1, y2))), d = Int(pyMin(Double(H), pyMax(y1, y2) + 1))
        if b - a < 2 || d - c < 2 { return [Float](repeating: 0, count: Appearance.region) }
        var counts = [Int](repeating: 0, count: Appearance.region)
        counts.withUnsafeMutableBufferPointer { cnt in
            img.forEachPixel(x0: a, x1: b, y0: c, y1: d) { r, g, bl in
                let (h, s, v) = OpenCVHSV.hsv(r: r, g: g, b: bl)
                let idx: Int
                if s >= 50 && v >= 40 {
                    idx = (h * Appearance.hBins / 180) * Appearance.sBins + min(s / 150, Appearance.sBins - 1)
                } else {
                    idx = Appearance.hBins * Appearance.sBins + (v < 80 ? 0 : (v < 170 ? 1 : 2))
                }
                cnt[idx] += 1
            }
        }
        let hist = counts.map { Float($0) }
        let total = hist.withUnsafeBufferPointer { 0 + NP.sum($0) }
        let denom: Float = total > 1.0 ? total : 1.0          // max(hist.sum(), 1.0)
        return hist.map { $0 / denom }
    }

    /// (N, 38) descriptors: shirt (shoulders→hips) and shorts (hips→knees) histograms;
    /// bbox fractions when the torso keypoints are unreliable (players.appearance_descriptors).
    public static func descriptors<S: RGBPixelSource>(_ img: S, boxes: [BBox], kps: KeypointArray,
                                                      kpConf: Double = PlayersConfig.default.kpConf) -> [Float] {
        var out = [Float](repeating: 0, count: boxes.count * Appearance.dim)
        for (i, bx) in boxes.enumerated() {
            let w = bx.x2 - bx.x1, h = bx.y2 - bx.y1
            let c = { (j: Int) in kps.c(i, j) >= kpConf }
            let top: (Double, Double, Double, Double), bot: (Double, Double, Double, Double)
            if c(KP.lSh), c(KP.rSh), c(KP.lHip), c(KP.rHip) {
                let xs = [KP.lSh, KP.rSh, KP.lHip, KP.rHip].map { kps.x(i, $0) }
                let m = NP.mean(xs)
                let hw = pyMax(0.35 * (xs.max()! - xs.min()!), 0.08 * w)
                let ys = (kps.y(i, KP.lSh) + kps.y(i, KP.rSh)) / 2, yh = (kps.y(i, KP.lHip) + kps.y(i, KP.rHip)) / 2
                top = (m - hw, ys + 0.15 * (yh - ys), m + hw, yh)
                let yk = (c(KP.lKnee) && c(KP.rKnee)) ? (kps.y(i, KP.lKnee) + kps.y(i, KP.rKnee)) / 2 : yh + 0.25 * h
                bot = (m - hw, yh, m + hw, yh + 0.6 * (yk - yh))
            } else {
                top = (bx.x1 + 0.3 * w, bx.y1 + 0.22 * h, bx.x2 - 0.3 * w, bx.y1 + 0.48 * h)
                bot = (bx.x1 + 0.3 * w, bx.y1 + 0.5 * h, bx.x2 - 0.3 * w, bx.y1 + 0.64 * h)
            }
            let ht = regionHist(img, top.0, top.1, top.2, top.3)
            let hb = regionHist(img, bot.0, bot.1, bot.2, bot.3)
            let o = i * Appearance.dim
            for j in 0..<Appearance.region {
                out[o + j] = ht[j]; out[o + Appearance.region + j] = hb[j]
            }
        }
        return out
    }

    /// Mean Hellinger distance of the two region histograms (0 same … 1 disjoint);
    /// 0 when either descriptor is missing/empty (players.appearance_distance).
    public static func distance(_ a: ArraySlice<Float>?, _ b: ArraySlice<Float>?) -> Double {
        guard let a, let b, a.count == Appearance.dim, b.count == Appearance.dim else { return 0.0 }
        var d = 0.0
        let a0 = a.startIndex, b0 = b.startIndex
        for off in [0, Appearance.region] {
            let pa = a[(a0 + off)..<(a0 + off + Appearance.region)]
            let pb = b[(b0 + off)..<(b0 + off + Appearance.region)]
            if NP.sum(pa) < 0.5 || NP.sum(pb) < 0.5 { return 0.0 }
            var prod = [Float](repeating: 0, count: Appearance.region)
            for j in 0..<Appearance.region { prod[j] = (pa[pa.startIndex + j] * pb[pb.startIndex + j]).squareRoot() }
            let bc = Double(prod.withUnsafeBufferPointer { 0 + NP.sum($0) })
            d += max(0.0, 1.0 - bc).squareRoot()
        }
        return d / 2
    }

    /// Mean descriptor over samples, float32 like `np.mean(list_of_float32, axis=0)`.
    static func mean(_ rows: [ArraySlice<Float>]) -> [Float]? {
        guard !rows.isEmpty else { return nil }
        var acc = [Float](repeating: 0, count: Appearance.dim)
        for r in rows { for j in 0..<Appearance.dim { acc[j] += r[r.startIndex + j] } }
        let n = Float(rows.count)
        return acc.map { $0 / n }
    }
}
