import Foundation

/// Bit-level numpy / CPython semantics the players port depends on. Every helper was
/// checked against numpy 1.26 (OpenBLAS arm64) in the Python venv:
///  • `np.sum` / `np.mean` of a contiguous 1-D array = pairwise summation
///    (8 accumulators up to 128 elements), float32 arrays accumulate in float32;
///  • BLAS 2-term dot products (`A @ B`, `x.dot(x)`, 1-D `np.linalg.norm`) are
///    `fma(a1, b1, a0 * b0)`; `np.linalg.norm(axis=…)` is a plain `sqrt(a*a + b*b)`;
///  • `np.interp`, float `np.arange`, `np.percentile` ("linear" + `_lerp`),
///    `np.median` (mean of the middle pair), Python `round(x, n)` (correctly rounded)
///    and `np.round(x, n)` (`rint(x * 10**n) / 10**n`).
enum NP {
    // MARK: sums

    /// numpy pairwise sum (float64).
    static func sum(_ a: UnsafeBufferPointer<Double>) -> Double {
        let n = a.count
        if n < 8 {
            var res = -0.0
            for x in a { res += x }
            return res
        }
        if n <= 128 {
            var r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3], r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7]
            var i = 8
            let end = n - n % 8
            while i < end {
                r0 += a[i]; r1 += a[i + 1]; r2 += a[i + 2]; r3 += a[i + 3]
                r4 += a[i + 4]; r5 += a[i + 5]; r6 += a[i + 6]; r7 += a[i + 7]
                i += 8
            }
            var res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
            while i < n { res += a[i]; i += 1 }
            return res
        }
        var n2 = n / 2
        n2 -= n2 % 8
        return sum(UnsafeBufferPointer(rebasing: a[0..<n2])) + sum(UnsafeBufferPointer(rebasing: a[n2..<n]))
    }

    static func sum(_ a: [Double]) -> Double {
        a.withUnsafeBufferPointer { 0.0 + sum($0) }
    }

    /// numpy pairwise sum (float32 accumulators).
    static func sum(_ a: UnsafeBufferPointer<Float>) -> Float {
        let n = a.count
        if n < 8 {
            var res: Float = -0.0
            for x in a { res += x }
            return res
        }
        if n <= 128 {
            var r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3], r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7]
            var i = 8
            let end = n - n % 8
            while i < end {
                r0 += a[i]; r1 += a[i + 1]; r2 += a[i + 2]; r3 += a[i + 3]
                r4 += a[i + 4]; r5 += a[i + 5]; r6 += a[i + 6]; r7 += a[i + 7]
                i += 8
            }
            var res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
            while i < n { res += a[i]; i += 1 }
            return res
        }
        var n2 = n / 2
        n2 -= n2 % 8
        return sum(UnsafeBufferPointer(rebasing: a[0..<n2])) + sum(UnsafeBufferPointer(rebasing: a[n2..<n]))
    }

    static func sum(_ a: ArraySlice<Float>) -> Float {
        a.withUnsafeBufferPointer { 0 + sum($0) }
    }

    /// `np.mean` of a 1-D float64 array (NaN for empty, like numpy's warning path).
    static func mean(_ a: [Double]) -> Double {
        a.isEmpty ? .nan : sum(a) / Double(a.count)
    }

    /// `np.median`: middle element, or the mean of the middle pair.
    static func median(_ a: [Double]) -> Double {
        guard !a.isEmpty else { return .nan }
        if a.contains(where: \.isNaN) { return .nan }
        let s = a.sorted()
        let n = s.count
        if n % 2 == 1 { return s[n / 2] }
        return (s[n / 2 - 1] + s[n / 2]) / 2
    }

    // MARK: BLAS-like products

    /// Two-term BLAS dot product as OpenBLAS computes it: fma(a1, b1, a0*b0).
    @inline(__always)
    static func fma2(_ a0: Double, _ b0: Double, _ a1: Double, _ b1: Double) -> Double {
        let p = a0 * b0
        return p.addingProduct(a1, b1)            // fma(a1, b1, a0*b0)
    }

    /// 1-D `np.linalg.norm` of a 2-vector: `sqrt(x.dot(x))` through BLAS.
    @inline(__always)
    static func norm1D(_ v: SIMD2<Double>) -> Double {
        fma2(v.x, v.x, v.y, v.y).squareRoot()
    }

    /// `np.linalg.norm(..., axis=k)` of a 2-vector: plain sum of squares.
    @inline(__always)
    static func normAxis(_ v: SIMD2<Double>) -> Double {
        (v.x * v.x + v.y * v.y).squareRoot()
    }

    // MARK: interpolation / ranges

    /// `np.interp` (compiled path): exact-hit and end shortcuts, `slope*(x-xp[j]) + fp[j]`.
    static func interp(_ x: Double, _ xp: [Double], _ fp: [Double]) -> Double {
        let n = xp.count
        precondition(n == fp.count && n > 0)
        if n == 1 { return fp[0] }
        if x > xp[n - 1] { return fp[n - 1] }
        if x < xp[0] { return fp[0] }
        // largest j with xp[j] <= x
        var lo = 0, hi = n
        while lo < hi {
            let mid = (lo + hi) / 2
            if xp[mid] <= x { lo = mid + 1 } else { hi = mid }
        }
        let j = lo - 1
        if j == n - 1 || xp[j] == x { return fp[j] }
        let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
        var r = slope * (x - xp[j]) + fp[j]
        if r.isNaN {
            r = slope * (x - xp[j + 1]) + fp[j + 1]
            if r.isNaN, fp[j] == fp[j + 1] { r = fp[j] }
        }
        return r
    }

    /// `np.arange(start, stop, step)` for Python floats.
    static func arange(_ start: Double, _ stop: Double, _ step: Double) -> [Double] {
        let len = Int(((stop - start) / step).rounded(.up))
        guard len > 0 else { return [] }
        var out = [Double](repeating: 0, count: len)
        out[0] = start
        if len > 1 { out[1] = start + step }
        let delta = out.count > 1 ? out[1] - start : 0
        if len > 2 { for i in 2..<len { out[i] = start + Double(i) * delta } }
        return out
    }

    /// `np.percentile(a, q)` with the default "linear" method.
    static func percentile(_ a: [Double], _ q: Double) -> Double {
        guard !a.isEmpty else { return .nan }
        if a.contains(where: \.isNaN) { return .nan }
        let s = a.sorted()
        let n = s.count
        let quant = q / 100
        let virtual = Double(n - 1) * quant
        var prev = virtual.rounded(.down)
        var next = prev + 1
        if virtual >= Double(n - 1) { prev = -1; next = -1 }
        if virtual < 0 { prev = 0; next = 0 }
        let ia = Int(prev), ib = Int(next)
        let lo = s[ia < 0 ? n + ia : ia], hi = s[ib < 0 ? n + ib : ib]
        let gamma = virtual - prev
        let diff = hi - lo
        if gamma >= 0.5 { return hi - diff * (1 - gamma) }
        return lo + diff * gamma
    }

    // MARK: rounding

    /// Python `round(x, digits)` for floats (correctly rounded, ties-to-even on the
    /// exact binary value) — via C's correctly rounded printf.
    static func pyRound(_ x: Double, _ digits: Int) -> Double {
        guard x.isFinite else { return x }
        let s = String(format: "%.\(digits)f", x)
        let r = Double(s) ?? x
        return r == 0 ? (x.sign == .minus ? -0.0 : 0.0) : r
    }

    /// `np.round(x, decimals)`: `rint(x * 10**d) / 10**d`.
    static func npRound(_ x: Double, _ decimals: Int) -> Double {
        let p = pow(10.0, Double(decimals))
        return (x * p).rounded(.toNearestOrEven) / p
    }

    /// numpy `array ** 2` (np.square: x * x).
    @inline(__always)
    static func sq(_ x: Double) -> Double { x * x }

    /// Python / numpy-scalar `x ** e` on floats (C `pow`).
    @inline(__always)
    static func pyPow(_ x: Double, _ e: Double) -> Double { Foundation.pow(x, e) }
}
