import Foundation

/// Camera-motion check between two consecutive samples (≈ 0.5 s apart) on 64-px grey thumbnails.
///
/// Camera motion shifts the WHOLE picture, so some global shift of the new thumbnail aligns it
/// with the old one much better than no shift; moving players change a few pixels and no shift
/// helps. Thumbnails are compared zero-mean (auto-exposure steps are not motion).
public struct MotionConfiguration: Sendable, Equatable {
    /// Search ±this many thumbnail pixels (≈ 20–30 full-res px each).
    public var maxShift = 4
    /// Mean |difference| (grey levels) at zero shift below this = static.
    public var minChange: Float = 2.5
    /// The best shift must reduce the zero-shift difference to below this fraction of it.
    public var alignedRatio: Float = 0.7
    /// Zero-shift mean |difference| above this = the view changed completely.
    public var sceneChange: Float = 28
    public init() {}
}

public struct MotionResult: Sendable, Equatable {
    public var moved: Bool
    /// Estimated image shift in FULL-resolution pixels (0 when not moved; .infinity for a scene change).
    public var shiftPx: Double
    public var zeroShiftDiff: Float
    public var bestDiff: Float
}

public enum MotionDetector {
    public static func compare(_ a: GrayThumb, _ b: GrayThumb, fullWidth: Int,
                               config: MotionConfiguration = MotionConfiguration()) -> MotionResult {
        guard a.width == b.width, a.height == b.height, a.width > 2 * config.maxShift + 2,
              a.height > 2 * config.maxShift + 2 else {
            return MotionResult(moved: false, shiftPx: 0, zeroShiftDiff: 0, bestDiff: 0)
        }
        let w = a.width, h = a.height, S = config.maxShift
        let ma = a.data.reduce(0, +) / Float(a.data.count), mb = b.data.reduce(0, +) / Float(b.data.count)
        func mad(_ dx: Int, _ dy: Int) -> Float {
            var s: Float = 0
            var n = 0
            for y in S..<(h - S) {
                let ra = y * w, rb = (y + dy) * w
                for x in S..<(w - S) {
                    s += abs((a.data[ra + x] - ma) - (b.data[rb + x + dx] - mb))
                    n += 1
                }
            }
            return n > 0 ? s / Float(n) : 0
        }
        let e0 = mad(0, 0)
        if e0 < config.minChange { return MotionResult(moved: false, shiftPx: 0, zeroShiftDiff: e0, bestDiff: e0) }
        var best = e0, bdx = 0, bdy = 0
        for dy in -S...S {
            for dx in -S...S where dx != 0 || dy != 0 {
                let e = mad(dx, dy)
                if e < best { best = e; bdx = dx; bdy = dy }
            }
        }
        let scale = Double(fullWidth) / Double(w)
        if best < config.alignedRatio * e0 && (bdx != 0 || bdy != 0) {
            return MotionResult(moved: true, shiftPx: (Double(bdx * bdx + bdy * bdy)).squareRoot() * scale,
                                zeroShiftDiff: e0, bestDiff: best)
        }
        if e0 > config.sceneChange {
            return MotionResult(moved: true, shiftPx: .infinity, zeroShiftDiff: e0, bestDiff: best)
        }
        return MotionResult(moved: false, shiftPx: 0, zeroShiftDiff: e0, bestDiff: best)
    }
}
