/// cv2.HoughLinesP — exact port of OpenCV 4.x HoughLinesProbabilistic (imgproc/src/hough.cpp),
/// including its cv::RNG((uint64)-1) point order, float trig table, cvRound (ties to even)
/// and the fused multiply-add clang emits for `j*ttab[2n] + i*ttab[2n+1]` on arm64.
import Foundation

enum Hough {
    /// cv::RNG multiply-with-carry.
    struct CvRNG {
        var state: UInt64
        init(_ seed: UInt64) { state = seed != 0 ? seed : 0xffff_ffff }
        @inline(__always) mutating func next() -> UInt32 {
            state = UInt64(UInt32(truncatingIfNeeded: state)) &* 4_164_903_690 &+ (state >> 32)
            return UInt32(truncatingIfNeeded: state)
        }
        @inline(__always) mutating func uniform(_ a: Int, _ b: Int) -> Int {
            a == b ? a : Int(next() % UInt32(b - a)) + a
        }
    }

    /// cv2.HoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap) → [x1, y1, x2, y2].
    /// `nonzero` is the 8-bit input (any non-zero = point).
    static func linesP(_ image: PlaneU8, rho: Double, theta: Double, threshold: Int,
                       minLineLength: Double, maxLineGap: Double, linesMax: Int = Int(Int32.max)) -> [SIMD4<Int32>] {
        let rhoF = Float(rho), thetaF = Float(theta)
        let lineLength = Int(minLineLength.rounded(.toNearestOrEven))
        let lineGap = Int(maxLineGap.rounded(.toNearestOrEven))
        let irho: Float = 1 / rhoF
        var rng = CvRNG(UInt64.max)
        let width = image.w, height = image.h
        // computeNumangle(0, CV_PI, theta)
        let thetaD = Double(thetaF)
        var numangle = Int(floor(Double.pi / thetaD)) + 1
        if numangle > 1 && abs(Double.pi - Double(numangle - 1) * thetaD) < thetaD / 2 { numangle -= 1 }
        let numrho = Int((Double((width + height) * 2 + 1) / Double(rhoF)).rounded(.toNearestOrEven))
        var accum = [Int32](repeating: 0, count: numangle * numrho)
        var mask = [UInt8](repeating: 0, count: width * height)
        var ttab = [Float](repeating: 0, count: numangle * 2)
        for n in 0..<numangle {
            ttab[2 * n] = Float(cos(Double(n) * thetaD) * Double(irho))
            ttab[2 * n + 1] = Float(sin(Double(n) * thetaD) * Double(irho))
        }
        var nzloc: [(x: Int, y: Int)] = []
        for y in 0..<height {
            for x in 0..<width where image.data[y * width + x] != 0 {
                mask[y * width + x] = 1
                nzloc.append((x, y))
            }
        }
        let rhoOff = (numrho - 1) / 2
        var lines: [SIMD4<Int32>] = []
        let shift = 16
        var count = nzloc.count
        accum.withUnsafeMutableBufferPointer { acc in
        mask.withUnsafeMutableBufferPointer { mdata in
        ttab.withUnsafeBufferPointer { tt in
            @inline(__always) func rIndex(_ j: Int, _ i: Int, _ n: Int) -> Int {
                let v = (Float(i) * tt[2 * n + 1]).addingProduct(Float(j), tt[2 * n])
                return Int(v.rounded(.toNearestOrEven)) + rhoOff
            }
            while count > 0 {
                defer { count -= 1 }
                let idx = rng.uniform(0, count)
                var maxVal = threshold - 1, maxN = 0
                let point = nzloc[idx]
                let i = point.y, j = point.x
                nzloc[idx] = nzloc[count - 1]
                if mdata[i * width + j] == 0 { continue }
                for n in 0..<numangle {
                    let r = rIndex(j, i, n)
                    let p = n * numrho + r
                    acc[p] += 1
                    let val = Int(acc[p])
                    if maxVal < val { maxVal = val; maxN = n }
                }
                if maxVal < threshold { continue }
                let a = -tt[maxN * 2 + 1], b = tt[maxN * 2]
                var x0 = j, y0 = i
                let dx0: Int, dy0: Int, xflag: Bool
                if abs(a) > abs(b) {
                    xflag = true
                    dx0 = a > 0 ? 1 : -1
                    dy0 = Int((b * Float(1 << shift) / abs(a)).rounded(.toNearestOrEven))
                    y0 = (y0 << shift) + (1 << (shift - 1))
                } else {
                    xflag = false
                    dy0 = b > 0 ? 1 : -1
                    dx0 = Int((a * Float(1 << shift) / abs(b)).rounded(.toNearestOrEven))
                    x0 = (x0 << shift) + (1 << (shift - 1))
                }
                var lineEnd = [(x: 0, y: 0), (x: 0, y: 0)]
                for k in 0..<2 {
                    var gap = 0, x = x0, y = y0, dx = dx0, dy = dy0
                    if k > 0 { dx = -dx; dy = -dy }
                    while true {
                        let i1: Int, j1: Int
                        if xflag { j1 = x; i1 = y >> shift } else { j1 = x >> shift; i1 = y }
                        if j1 < 0 || j1 >= width || i1 < 0 || i1 >= height { break }
                        if mdata[i1 * width + j1] != 0 {
                            gap = 0
                            lineEnd[k] = (j1, i1)
                        } else {
                            gap += 1
                            if gap > lineGap { break }
                        }
                        x += dx; y += dy
                    }
                }
                let goodLine = abs(lineEnd[1].x - lineEnd[0].x) >= lineLength ||
                    abs(lineEnd[1].y - lineEnd[0].y) >= lineLength
                for k in 0..<2 {
                    var x = x0, y = y0, dx = dx0, dy = dy0
                    if k > 0 { dx = -dx; dy = -dy }
                    while true {
                        let i1: Int, j1: Int
                        if xflag { j1 = x; i1 = y >> shift } else { j1 = x >> shift; i1 = y }
                        let p = i1 * width + j1
                        if mdata[p] != 0 {
                            if goodLine {
                                for n in 0..<numangle { acc[n * numrho + rIndex(j1, i1, n)] -= 1 }
                            }
                            mdata[p] = 0
                        }
                        if i1 == lineEnd[k].y && j1 == lineEnd[k].x { break }
                        x += dx; y += dy
                    }
                }
                if goodLine {
                    lines.append(SIMD4(Int32(lineEnd[0].x), Int32(lineEnd[0].y), Int32(lineEnd[1].x), Int32(lineEnd[1].y)))
                    if lines.count >= linesMax { return }
                }
            }
        }
        }
        }
        return lines
    }
}
