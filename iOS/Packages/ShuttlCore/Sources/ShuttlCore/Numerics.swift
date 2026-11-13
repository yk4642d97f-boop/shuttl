import Foundation

/// numpy-semantics helpers so every ported stage computes statistics identically
/// to the Python reference (same interpolation, same NaN handling).
public enum Num {
    /// `np.median` (NaN if any element is NaN, NaN for empty input).
    public static func median(_ xs: [Double]) -> Double {
        percentile(xs, 50)
    }

    /// `np.nanmedian`.
    public static func nanmedian(_ xs: [Double]) -> Double { median(xs.filter { !$0.isNaN }) }

    /// `np.percentile(xs, q)` with the default "linear" method. q in 0...100.
    public static func percentile(_ xs: [Double], _ q: Double) -> Double {
        guard !xs.isEmpty, !xs.contains(where: \.isNaN) else { return .nan }
        let s = xs.sorted()
        let pos = (q / 100) * Double(s.count - 1)
        let lo = Int(pos.rounded(.down)), hi = min(lo + 1, s.count - 1)
        let frac = pos - Double(lo)
        return s[lo] + (s[hi] - s[lo]) * frac
    }

    public static func nanpercentile(_ xs: [Double], _ q: Double) -> Double { percentile(xs.filter { !$0.isNaN }, q) }

    /// `np.quantile(xs, q)`, q in 0...1.
    public static func quantile(_ xs: [Double], _ q: Double) -> Double { percentile(xs, q * 100) }

    public static func mean(_ xs: [Double]) -> Double { xs.isEmpty ? .nan : xs.reduce(0, +) / Double(xs.count) }
    public static func nanmean(_ xs: [Double]) -> Double { mean(xs.filter { !$0.isNaN }) }

    /// `np.std` (population, ddof = 0 by default).
    public static func std(_ xs: [Double], ddof: Int = 0) -> Double {
        guard xs.count > ddof else { return .nan }
        let m = mean(xs)
        return (xs.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(xs.count - ddof)).squareRoot()
    }

    /// `np.interp(x, xp, fp)` — xp increasing; clamps to the end values outside the range.
    public static func interp(_ x: Double, _ xp: [Double], _ fp: [Double]) -> Double {
        precondition(xp.count == fp.count && !xp.isEmpty)
        if x <= xp[0] { return fp[0] }
        if x >= xp[xp.count - 1] { return fp[fp.count - 1] }
        var lo = 0, hi = xp.count - 1
        while hi - lo > 1 {
            let mid = (lo + hi) / 2
            if xp[mid] <= x { lo = mid } else { hi = mid }
        }
        let span = xp[hi] - xp[lo]
        return span == 0 ? fp[lo] : fp[lo] + (fp[hi] - fp[lo]) * (x - xp[lo]) / span
    }

    public static func interp(_ xs: [Double], _ xp: [Double], _ fp: [Double]) -> [Double] { xs.map { interp($0, xp, fp) } }

    public static func clip<T: Comparable>(_ x: T, _ lo: T, _ hi: T) -> T { min(max(x, lo), hi) }

    /// `np.searchsorted(a, v, side)` on an ascending array.
    public static func searchsorted(_ a: [Double], _ v: Double, right: Bool = false) -> Int {
        var lo = 0, hi = a.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if right ? (a[mid] <= v) : (a[mid] < v) { lo = mid + 1 } else { hi = mid }
        }
        return lo
    }

    /// Stable argsort (numpy `kind="stable"`); NaN sorts last like numpy.
    public static func argsort(_ xs: [Double]) -> [Int] {
        xs.indices.sorted { a, b in
            let x = xs[a], y = xs[b]
            switch (x.isNaN, y.isNaN) {
            case (false, true): return true
            case (true, false): return false
            case (true, true): return a < b
            case (false, false): return x < y || (x == y && a < b)
            }
        }
    }

    /// `np.convolve(a, v, mode="same")`.
    public static func convolveSame(_ a: [Double], _ v: [Double]) -> [Double] {
        guard !a.isEmpty, !v.isEmpty else { return [] }
        let n = a.count, m = v.count
        let full = n + m - 1
        var out = [Double](repeating: 0, count: full)
        for i in 0..<n { for j in 0..<m { out[i + j] += a[i] * v[j] } }
        let start = (full - max(n, m)) / 2
        return Array(out[start..<(start + max(n, m))])
    }

    /// Round half to even like Python's `round` / `np.round`.
    public static func roundHalfEven(_ x: Double, digits: Int = 0) -> Double {
        let p = pow(10.0, Double(digits))
        return (x * p).rounded(.toNearestOrEven) / p
    }

    /// Python-style `int(round(x))`.
    public static func iround(_ x: Double) -> Int { Int(x.rounded(.toNearestOrEven)) }
}
