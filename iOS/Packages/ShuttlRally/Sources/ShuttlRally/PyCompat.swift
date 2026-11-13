import Foundation
import ShuttlCore

/// Bit-exact equivalents of the few numpy / Python builtins rally.py, shots.py
/// and serialize.py rely on (ShuttlCore.Num uses naive summation and
/// multiply-round-divide rounding, which differ in the last bits / at decimal
/// ties from `np.mean` and Python's `round`).
enum Py {
    /// numpy's pairwise summation (`pairwise_sum_DOUBLE`), as used by
    /// `np.sum` / `np.mean` on contiguous float64 arrays: 0.0 + pairwise(a).
    static func npSum(_ a: [Double]) -> Double {
        a.withUnsafeBufferPointer { 0.0 + pairwise($0.baseAddress!, a.count) }
    }

    private static func pairwise(_ a: UnsafePointer<Double>, _ n: Int) -> Double {
        if n < 8 {
            var res = 0.0
            for i in 0..<n { res += a[i] }
            return res
        } else if n <= 128 {
            var r = (a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])
            var i = 8
            while i < n - (n % 8) {
                r.0 += a[i]; r.1 += a[i + 1]; r.2 += a[i + 2]; r.3 += a[i + 3]
                r.4 += a[i + 4]; r.5 += a[i + 5]; r.6 += a[i + 6]; r.7 += a[i + 7]
                i += 8
            }
            var res = ((r.0 + r.1) + (r.2 + r.3)) + ((r.4 + r.5) + (r.6 + r.7))
            while i < n { res += a[i]; i += 1 }
            return res
        } else {
            var n2 = n / 2
            n2 -= n2 % 8
            return pairwise(a, n2) + pairwise(a + n2, n - n2)
        }
    }

    /// CPython 3.12 builtin `sum(floats)` (start 0): Neumaier compensated summation.
    static func sum(_ a: [Double]) -> Double {
        guard var f = a.first else { return 0 }
        var c = 0.0
        for x in a.dropFirst() {
            let t = f + x
            if Swift.abs(f) >= Swift.abs(x) { c += (f - t) + x } else { c += (x - t) + f }
            f = t
        }
        if c != 0 && c.isFinite { f += c }
        return f
    }

    /// `float(np.mean(a))` (NaN for empty input, like numpy's warning path).
    static func npMean(_ a: [Double]) -> Double {
        a.isEmpty ? .nan : npSum(a) / Double(a.count)
    }

    /// `np.mean(mask)` of a boolean array.
    static func npMeanBool(_ a: [Bool]) -> Double {
        a.isEmpty ? .nan : Double(a.reduce(0) { $0 + ($1 ? 1 : 0) }) / Double(a.count)
    }

    /// Python's `round(x, nd)` for a finite float: correctly rounded decimal
    /// (ties to even on the exact binary value), read back as the nearest double.
    /// `%.*f` in Apple's libc is correctly rounded like CPython's dtoa mode 3.
    static func round(_ x: Double, _ nd: Int) -> Double {
        guard x.isFinite else { return x }
        return Double(String(format: "%.\(nd)f", x)) ?? x
    }

    /// serialize._f: None for None / non-finite, else round(x, nd).
    static func f(_ x: Double?, _ nd: Int = 3) -> JSONValue {
        guard let x, x.isFinite else { return .null }
        return .number(round(x, nd))
    }

    /// Python's builtin `max(iterable, default=d)` on floats: first element,
    /// replaced only by a strictly greater one (NaN handling follows from `>`).
    static func max(_ xs: [Double], default d: Double) -> Double {
        guard var m = xs.first else { return d }
        for x in xs.dropFirst() where x > m { m = x }
        return m
    }

    /// `np.percentile(v, q)` ("linear", numpy 1.26 `_lerp`) on finite values.
    static func percentile(_ v: [Double], _ q: Double) -> Double {
        guard !v.isEmpty else { return .nan }
        let s = v.sorted()
        let n = s.count
        let virt = (q / 100) * Double(n - 1)
        let prev = virt.rounded(.down)
        let lo = Swift.min(Swift.max(Int(prev), 0), n - 1)
        let hi = Swift.min(lo + 1, n - 1)
        let g = virt - prev
        let a = s[lo], b = s[hi]
        let d = b - a
        return g >= 0.5 ? b - d * (1 - g) : a + d * g
    }

    /// `np.median` on finite values (mean of the two middle ones for even n).
    static func median(_ v: [Double]) -> Double {
        guard !v.isEmpty else { return .nan }
        let s = v.sorted()
        let n = s.count
        return n % 2 == 1 ? s[n / 2] : (0.0 + s[n / 2 - 1] + s[n / 2]) / 2
    }

    /// Python's `math.erf`-based standard normal CDF used by rally.py (`_phi`).
    static func phi(_ z: Double) -> Double { 0.5 * (1.0 + erf(z / 2.0.squareRoot())) }
}

extension JSONValue {
    static func int(_ n: Int) -> JSONValue { .number(Double(n)) }
    static func str(_ s: String) -> JSONValue { .string(s) }
}
