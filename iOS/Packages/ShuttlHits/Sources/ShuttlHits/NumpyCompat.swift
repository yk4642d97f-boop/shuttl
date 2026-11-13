import Foundation

/// Bit-exact stand-ins for the numpy / CPython primitives hits.py relies on
/// (verified empirically against numpy 1.26.4 / CPython 3.12 on arm64):
///  • `np.sum` / `np.mean` = pairwise summation (8-way unrolled blocks, split at 128);
///  • `np.dot` / `np.linalg.norm` of a 2-vector = OpenBLAS ddot = fma(a1, b1, a0*b0);
///  • `np.linalg.norm(A, axis=1)` of an (N,2) array = sqrt(x*x + y*y);
///  • `math.hypot` = CPython's correctly rounded `vector_norm`;
///  • `np.median` = mean of the two middles; `np.percentile` = numpy's lerp.
enum NP {
    // MARK: reductions

    /// numpy `pairwise_sum` over `p[0..<n]`.
    static func pairwiseSum(_ p: UnsafePointer<Double>, _ n: Int) -> Double {
        if n < 8 {
            var res = -0.0
            for i in 0..<n { res += p[i] }
            return res
        } else if n <= 128 {
            var r0 = p[0], r1 = p[1], r2 = p[2], r3 = p[3], r4 = p[4], r5 = p[5], r6 = p[6], r7 = p[7]
            var i = 8
            let top = n - (n % 8)
            while i < top {
                r0 += p[i]; r1 += p[i + 1]; r2 += p[i + 2]; r3 += p[i + 3]
                r4 += p[i + 4]; r5 += p[i + 5]; r6 += p[i + 6]; r7 += p[i + 7]
                i += 8
            }
            var res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
            while i < n { res += p[i]; i += 1 }
            return res
        } else {
            var n2 = n / 2
            n2 -= n2 % 8
            return pairwiseSum(p, n2) + pairwiseSum(p + n2, n - n2)
        }
    }

    static func sum(_ xs: [Double]) -> Double {
        xs.withUnsafeBufferPointer { b in b.baseAddress.map { pairwiseSum($0, b.count) } ?? 0 }
    }

    static func mean(_ xs: [Double]) -> Double { xs.isEmpty ? .nan : sum(xs) / Double(xs.count) }

    /// `np.median` of a finite array (mean of the two middles for even n).
    static func median(_ xs: [Double]) -> Double {
        let n = xs.count
        guard n > 0 else { return .nan }
        if xs.contains(where: \.isNaN) { return .nan }
        let s = xs.sorted()
        if n % 2 == 1 { return s[n / 2] }
        return (s[n / 2 - 1] + s[n / 2]) / 2
    }

    /// `np.percentile(xs, q)` ("linear"), numpy's lerp incl. the gamma >= 0.5 branch.
    static func percentile(_ xs: [Double], _ q: Double) -> Double {
        let n = xs.count
        guard n > 0 else { return .nan }
        let s = xs.sorted()
        let v = q / 100 * Double(n - 1)
        let lo = Int(v.rounded(.down))
        let g = v - Double(lo)
        let a = s[min(max(lo, 0), n - 1)], b = s[min(max(lo + 1, 0), n - 1)]
        let d = b - a
        return g >= 0.5 ? b - d * (1 - g) : a + d * g
    }

    /// `np.linspace(a, b, n)` (endpoint included).
    static func linspace(_ a: Double, _ b: Double, _ n: Int) -> [Double] {
        guard n > 1 else { return n == 1 ? [a] : [] }
        let step = (b - a) / Double(n - 1)
        var out = [Double](repeating: 0, count: n)
        for i in 0..<n { out[i] = a + Double(i) * step }
        out[n - 1] = b
        return out
    }

    // MARK: vector norms

    /// `np.dot(a, b)` for 2-vectors (OpenBLAS ddot on arm64).
    @inline(__always) static func dot2(_ a0: Double, _ a1: Double, _ b0: Double, _ b1: Double) -> Double {
        (a0 * b0).addingProduct(a1, b1)          // fma(a1, b1, a0*b0)
    }

    /// `np.linalg.norm(v)` for a 1-D 2-vector.
    @inline(__always) static func norm2(_ x: Double, _ y: Double) -> Double { dot2(x, y, x, y).squareRoot() }

    /// `np.linalg.norm(A, axis=1)` row of an (N,2) array.
    @inline(__always) static func rowNorm(_ x: Double, _ y: Double) -> Double { (x * x + y * y).squareRoot() }

    /// CPython 3.12 `math.hypot(x, y)` (vector_norm: scaled, compensated, one Newton correction).
    static func pyHypot(_ x: Double, _ y: Double) -> Double {
        let ax = abs(x), ay = abs(y)
        if ax.isInfinite || ay.isInfinite { return .infinity }
        if ax.isNaN || ay.isNaN { return .nan }
        let mx = max(ax, ay)
        if mx == 0 { return mx }
        let e = frexp(mx).1
        let scale = Double(sign: .plus, exponent: -e, significand: 1)
        var csum = 1.0, f1 = 0.0, f2 = 0.0
        for v0 in [ax, ay] {
            let v = v0 * scale
            let hi = v * v
            let lo = (-hi).addingProduct(v, v)
            let s = csum + hi
            let sl = (csum - s) + hi
            csum = s; f1 += lo; f2 += sl
        }
        var h = (csum - 1.0 + (f1 + f2)).squareRoot()
        let hi = -h * h
        let lo = (-hi).addingProduct(-h, h)
        let s = csum + hi
        let sl = (csum - s) + hi
        csum = s; f1 += lo; f2 += sl
        let xr = csum - 1.0 + (f1 + f2)
        h += xr / (2.0 * h)
        return h / scale
    }

    // MARK: searching / sorting

    /// `np.searchsorted(a, v, side)` over `p[0..<n]` (ascending).
    @inline(__always) static func searchsorted(_ p: UnsafePointer<Double>, _ n: Int, _ v: Double, right: Bool = false) -> Int {
        var lo = 0, hi = n
        if right {
            while lo < hi { let mid = (lo + hi) >> 1; if p[mid] <= v { lo = mid + 1 } else { hi = mid } }
        } else {
            while lo < hi { let mid = (lo + hi) >> 1; if p[mid] < v { lo = mid + 1 } else { hi = mid } }
        }
        return lo
    }

    static func searchsorted(_ a: [Double], _ v: Double, right: Bool = false) -> Int {
        a.withUnsafeBufferPointer { b in b.baseAddress.map { searchsorted($0, b.count, v, right: right) } ?? 0 }
    }

    /// Stable argsort ascending (numpy kind="stable"; NaN last).
    static func argsortStable(_ xs: [Double]) -> [Int] {
        xs.indices.sorted { a, b in
            let x = xs[a], y = xs[b]
            if x.isNaN != y.isNaN { return y.isNaN }
            if x.isNaN { return a < b }
            return x < y || (x == y && a < b)
        }
    }

    /// `np.clip(x, lo, hi)` (NaN propagates).
    @inline(__always) static func clip(_ x: Double, _ lo: Double, _ hi: Double) -> Double {
        x.isNaN ? x : Swift.min(Swift.max(x, lo), hi)
    }

    /// `np.sign`.
    @inline(__always) static func sign(_ x: Double) -> Double { x.isNaN ? .nan : (x > 0 ? 1 : (x < 0 ? -1 : 0)) }

    /// Python builtin `max(a, b)` (keeps `a` unless `b > a`).
    @inline(__always) static func pyMax(_ a: Double, _ b: Double) -> Double { b > a ? b : a }
    @inline(__always) static func pyMin(_ a: Double, _ b: Double) -> Double { b < a ? b : a }

    /// Python `math.log1p(math.exp(-x))`-based log-sigmoid from hits._log_sigmoid.
    @inline(__always) static func logSigmoid(_ x: Double) -> Double { x > -30 ? -Foundation.log1p(Foundation.exp(-x)) : x }

    /// `np.logaddexp(a, b)` (npy_logaddexp).
    @inline(__always) static func logaddexp(_ x: Double, _ y: Double) -> Double {
        if x == y { return x + Double.ln2 }
        let tmp = x - y
        if tmp > 0 { return x + Foundation.log1p(Foundation.exp(-tmp)) }
        if tmp <= 0 { return y + Foundation.log1p(Foundation.exp(tmp)) }
        return tmp                                // NaN
    }
}

extension Double {
    static let ln2 = 0.693147180559945309417232121458176568
}
