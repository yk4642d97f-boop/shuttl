import Foundation

/// Bit-level numpy 1.26 semantics for the reductions metrics.py / sim.py use, so Swift
/// scores equal Python's exactly rather than "to 1e-12":
///  • `sum`/`mean`: pairwise summation (8 accumulators, blocks ≤ 128) over 8192-element
///    buffer chunks accumulated left to right (measured against numpy on this machine);
///  • `percentile` "linear": virtual index (n-1)·q with numpy's two-sided `_lerp`;
///  • `median`: middle element / mean of the two middle elements;
///  • 1-D `np.linalg.norm` = sqrt(OpenBLAS ddot) which accumulates with FMA
///    (`norm(v)` here), while `norm(axis=1)` of an (N,2) array is plain sqrt(x·x + y·y).
public enum NP {
    // MARK: summation

    static func pairwise(_ a: UnsafeBufferPointer<Double>, _ lo: Int, _ n: Int) -> Double {
        if n < 8 {
            var r = -0.0
            for k in 0..<n { r += a[lo + k] }
            return r
        }
        if n <= 128 {
            var r0 = a[lo], r1 = a[lo + 1], r2 = a[lo + 2], r3 = a[lo + 3]
            var r4 = a[lo + 4], r5 = a[lo + 5], r6 = a[lo + 6], r7 = a[lo + 7]
            var i = 8
            let stop = n - (n % 8)
            while i < stop {
                r0 += a[lo + i]; r1 += a[lo + i + 1]; r2 += a[lo + i + 2]; r3 += a[lo + i + 3]
                r4 += a[lo + i + 4]; r5 += a[lo + i + 5]; r6 += a[lo + i + 6]; r7 += a[lo + i + 7]
                i += 8
            }
            var res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
            while i < n { res += a[lo + i]; i += 1 }
            return res
        }
        var n2 = n / 2
        n2 -= n2 % 8
        return pairwise(a, lo, n2) + pairwise(a, lo + n2, n - n2)
    }

    /// `np.sum` of a contiguous float64 vector.
    public static func sum(_ xs: [Double]) -> Double {
        xs.withUnsafeBufferPointer { buf in
            var acc = 0.0
            var start = 0
            while start < buf.count {
                let len = min(8192, buf.count - start)
                acc += pairwise(buf, start, len)
                start += len
            }
            return acc
        }
    }

    /// `np.mean` (NaN for empty input, like numpy's warning + nan).
    public static func mean(_ xs: [Double]) -> Double { xs.isEmpty ? .nan : sum(xs) / Double(xs.count) }

    /// `np.mean` of a bool array (exact: a count).
    public static func mean(_ xs: [Bool]) -> Double {
        xs.isEmpty ? .nan : Double(xs.lazy.filter { $0 }.count) / Double(xs.count)
    }

    /// `np.nanmean`: NaN → 0, summed with the full length, divided by the finite count.
    public static func nanmean(_ xs: [Double]) -> Double {
        let cnt = xs.lazy.filter { !$0.isNaN }.count
        guard cnt > 0 else { return .nan }
        return sum(xs.map { $0.isNaN ? 0 : $0 }) / Double(cnt)
    }

    // MARK: order statistics

    /// `np.median` (NaN if any NaN or empty).
    public static func median(_ xs: [Double]) -> Double {
        guard !xs.isEmpty else { return .nan }
        if xs.contains(where: \.isNaN) { return .nan }
        let s = xs.sorted()
        let n = s.count, h = n / 2
        if n % 2 == 1 { return s[h] }
        return mean([s[h - 1], s[h]])
    }

    /// `np.nanmedian`.
    public static func nanmedian(_ xs: [Double]) -> Double { median(xs.filter { !$0.isNaN }) }

    /// `np.percentile(xs, q)` default method "linear" (q in 0…100).
    public static func percentile(_ xs: [Double], _ q: Double) -> Double {
        guard !xs.isEmpty else { return .nan }
        if xs.contains(where: \.isNaN) { return .nan }
        let s = xs.sorted()
        let n = s.count
        let virtualIndex = Double(n - 1) * (q / 100)
        var prev: Int, next: Int, gamma: Double
        if virtualIndex >= Double(n - 1) {
            prev = n - 1; next = n - 1; gamma = virtualIndex - Double(n - 1)
        } else if virtualIndex < 0 {
            prev = 0; next = 0; gamma = virtualIndex
        } else {
            let f = virtualIndex.rounded(.down)
            prev = Int(f); next = prev + 1
            gamma = virtualIndex - f
        }
        return lerp(s[prev], s[next], gamma)
    }

    /// numpy `_lerp`: a + (b-a)·t, or b - (b-a)·(1-t) where t ≥ 0.5.
    @inline(__always)
    public static func lerp(_ a: Double, _ b: Double, _ t: Double) -> Double {
        let d = b - a
        return t >= 0.5 ? b - d * (1 - t) : a + d * t
    }

    // MARK: norms

    /// 1-D `np.linalg.norm` of a length-2 vector (OpenBLAS ddot → FMA chain).
    @inline(__always)
    public static func norm(_ x: Double, _ y: Double) -> Double { fma(y, y, x * x).squareRoot() }

    /// 1-D `np.linalg.norm` of a length-3 vector.
    @inline(__always)
    public static func norm(_ x: Double, _ y: Double, _ z: Double) -> Double {
        fma(z, z, fma(y, y, x * x)).squareRoot()
    }

    @inline(__always)
    public static func norm(_ v: SIMD3<Double>) -> Double { norm(v.x, v.y, v.z) }

    /// Row norm of an (N,2) array along axis 1 (plain multiply-add).
    @inline(__always)
    public static func rowNorm(_ x: Double, _ y: Double) -> Double { (x * x + y * y).squareRoot() }

    /// Row norm of an (N,3) array along axis 1.
    @inline(__always)
    public static func rowNorm(_ x: Double, _ y: Double, _ z: Double) -> Double { ((x * x + y * y) + z * z).squareRoot() }

    // MARK: grids / interpolation (sim.py)

    /// `np.linspace(start, stop, num)` (endpoint=True).
    public static func linspace(_ start: Double, _ stop: Double, _ num: Int) -> [Double] {
        guard num > 0 else { return [] }
        let div = Double(num - 1)
        let delta = stop - start
        var y = (0..<num).map(Double.init)
        if num > 1 {
            let step = delta / div
            if step == 0 {
                y = y.map { $0 / div * delta }
            } else {
                y = y.map { $0 * step }
            }
        } else {
            y = y.map { $0 * delta }
        }
        y = y.map { $0 + start }
        if num > 1 { y[num - 1] = stop }
        return y
    }

    /// `np.arange(start, stop, step)` for floats: start + i·step, length ceil((stop-start)/step).
    public static func arange(_ start: Double, _ stop: Double, _ step: Double) -> [Double] {
        let n = max(0, Int(((stop - start) / step).rounded(.up)))
        return (0..<n).map { start + Double($0) * step }
    }

    /// `np.power(x, y)` for float64 = libm `pow` (numpy's loop calls `npy_pow` → `pow`).
    /// Kept opaque on purpose: in -O builds LLVM's LibCallSimplifier rewrites
    /// `pow(10.0, y)` into Darwin's `__exp10(y)`, which is not bit-identical to `pow`
    /// (1 ulp apart on some inputs) — optnone keeps the real libm call in every configuration.
    @inline(never) @_optimize(none)
    public static func pow(_ x: Double, _ y: Double) -> Double { Foundation.pow(x, y) }

    /// `np.geomspace(start, stop, num)` for positive start/stop.
    public static func geomspace(_ start: Double, _ stop: Double, _ num: Int) -> [Double] {
        let ls = log10(start), le = log10(stop)
        var r = linspace(ls, le, num).map { NP.pow(10.0, $0) }        // logspace: power(10, y)
        if num > 0 { r[0] = start }
        if num > 1 { r[num - 1] = stop }
        return r
    }

    /// `np.interp(x, xp, fp)` — numpy's binary search and slope form
    /// `slope*(x - xp[j]) + fp[j]` (not fused; measured bit-exact against numpy 1.26 arm64).
    public static func interp(_ x: Double, _ xp: [Double], _ fp: [Double]) -> Double {
        let n = xp.count
        precondition(n == fp.count && n > 0)
        if x.isNaN { return x }
        if n == 1 { return fp[0] }
        if x > xp[n - 1] { return fp[n - 1] }
        if x < xp[0] { return fp[0] }
        // j with xp[j] <= x < xp[j+1] (numpy binary_search_with_guess); j == n-1 → fp[-1];
        // an exact hit returns fp[j] ("avoid potential non-finite interpolation").
        var lo = 0, hi = n
        while lo < hi {
            let mid = (lo + hi) / 2
            if xp[mid] <= x { lo = mid + 1 } else { hi = mid }
        }
        let j = lo - 1
        if j >= n - 1 { return fp[n - 1] }
        if xp[j] == x { return fp[j] }
        let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
        var r = slope * (x - xp[j]) + fp[j]
        if r.isNaN {
            r = slope * (x - xp[j + 1]) + fp[j + 1]
            if r.isNaN, fp[j] == fp[j + 1] { r = fp[j] }
        }
        return r
    }

    /// `np.degrees` / `np.radians` (numpy multiplies by the rounded constants).
    @inline(__always) public static func radians(_ d: Double) -> Double { d * (Double.pi / 180) }
    @inline(__always) public static func degrees(_ r: Double) -> Double { r * (180 / Double.pi) }

    /// `np.clip(x, lo, hi)` = minimum(maximum(x, lo), hi) (NaN propagates).
    @inline(__always)
    public static func clip(_ x: Double, _ lo: Double, _ hi: Double) -> Double {
        if x.isNaN { return x }
        return Swift.min(Swift.max(x, lo), hi)
    }

    /// Python `sorted(strings)` order (code-point order, not locale).
    public static func pySorted(_ xs: some Sequence<String>) -> [String] { xs.sorted(by: pyLess) }

    /// Python `str < str` (code-point order).
    public static func pyLess(_ a: String, _ b: String) -> Bool {
        a.unicodeScalars.map(\.value).lexicographicallyPrecedes(b.unicodeScalars.map(\.value))
    }
}
