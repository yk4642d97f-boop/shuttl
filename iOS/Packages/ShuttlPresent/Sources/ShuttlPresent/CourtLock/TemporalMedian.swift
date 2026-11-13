import Foundation
import ShuttlCore

/// Per-pixel temporal median of up to 7 evenly spaced frames — exactly the frame selection and
/// k-th element of ShuttlCourt's `_median_image(max_frames=7)` (≤ 2 frames → the first frame),
/// but vectorised (16 pixels per step, compare-exchange network) instead of a sort per pixel.
public enum TemporalMedian {
    public static func median(_ frames: [ImageU8], maxFrames: Int = 7) -> ImageU8 {
        precondition(!frames.isEmpty)
        let first = frames[0]
        if frames.count <= 2 { return first }
        let n = min(maxFrames, frames.count)
        let span = Double(frames.count - 1)
        let idx = (0..<n).map { i -> Int in
            let v = n == 1 ? 0 : (i == n - 1 ? span : Double(i) * (span / Double(n - 1)))
            return Int(v.rounded(.toNearestOrEven))
        }
        let k = idx.count / 2
        let count = first.data.count
        precondition(frames.allSatisfy { $0.data.count == count && $0.width == first.width }, "frames differ in size")
        var out = [UInt8](repeating: 0, count: count)
        let srcs = idx.map { frames[$0].data }
        let m = srcs.count
        withPointers(srcs) { ptrs in
            out.withUnsafeMutableBufferPointer { o in
                var p = 0
                var v = [SIMD16<UInt8>](repeating: .zero, count: m)
                while p + 16 <= count {
                    for j in 0..<m { v[j] = UnsafeRawPointer(ptrs[j] + p).loadUnaligned(as: SIMD16<UInt8>.self) }
                    // odd-even transposition sort network (m ≤ 7 → ≤ 21 compare-exchanges)
                    for pass in 0..<m {
                        var j = pass & 1
                        while j + 1 < m {
                            let a = v[j], b = v[j + 1]
                            v[j] = pointwiseMin(a, b); v[j + 1] = pointwiseMax(a, b)
                            j += 2
                        }
                    }
                    UnsafeMutableRawPointer(o.baseAddress! + p).storeBytes(of: v[k], as: SIMD16<UInt8>.self)
                    p += 16
                }
                var buf = [UInt8](repeating: 0, count: m)
                while p < count {
                    for j in 0..<m { buf[j] = ptrs[j][p] }
                    buf.sort()
                    o[p] = buf[k]
                    p += 1
                }
            }
        }
        return ImageU8(width: first.width, height: first.height, channels: first.channels, data: out)
    }

    private static func withPointers(_ arrays: [[UInt8]], _ body: ([UnsafePointer<UInt8>]) -> Void) {
        func go(_ i: Int, _ acc: [UnsafePointer<UInt8>]) {
            if i == arrays.count { body(acc); return }
            arrays[i].withUnsafeBufferPointer { go(i + 1, acc + [$0.baseAddress!]) }
        }
        go(0, [])
    }
}
