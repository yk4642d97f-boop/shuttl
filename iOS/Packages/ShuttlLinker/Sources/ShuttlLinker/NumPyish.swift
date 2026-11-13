import Foundation

// numpy-semantics helpers local to the linker (bit-for-bit where numpy is
// elementwise; ShuttlCore.Num is not modified).

typealias V2 = SIMD2<Double>

let linkerINF = 1e30                       // _INF

@inline(__always) func isFin(_ v: V2) -> Bool { v.x.isFinite }
@inline(__always) func nanV2() -> V2 { V2(.nan, .nan) }
@inline(__always) func hyp(_ v: V2) -> Double { hypot(v.x, v.y) }

/// `np.median`: NaN if empty or any NaN; mean of the two middle values = (a + b) / 2.
func npMedian(_ a: [Double]) -> Double {
    if a.isEmpty { return .nan }
    for v in a where v.isNaN { return .nan }
    let s = a.sorted()
    let n = s.count
    return n % 2 == 1 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2
}

/// `np.median(np.diff(t))` if len(t) > 1 else `fallback`.
func medianDiff(_ t: [Double], fallback: Double = 1.0 / 30.0) -> Double {
    guard t.count > 1 else { return fallback }
    var d = [Double](repeating: 0, count: t.count - 1)
    for i in 0..<(t.count - 1) { d[i] = t[i + 1] - t[i] }
    return npMedian(d)
}

/// `np.mean(bool array)`.
@inline(__always) func boolMean(_ n: Int, _ count: Int) -> Double { Double(count) / Double(n) }

/// np.clip that propagates NaN (np.minimum/np.maximum semantics).
@inline(__always) func npClip(_ x: Double, _ lo: Double, _ hi: Double) -> Double {
    if x.isNaN { return x }
    return Swift.min(Swift.max(x, lo), hi)
}

/// np.maximum / np.minimum (NaN-propagating).
@inline(__always) func npMax(_ a: Double, _ b: Double) -> Double { a.isNaN || b.isNaN ? .nan : (a >= b ? a : b) }
@inline(__always) func npMin(_ a: Double, _ b: Double) -> Double { a.isNaN || b.isNaN ? .nan : (a <= b ? a : b) }

/// Python builtin max/min for floats: `max(a, b)` returns a unless b > a.
@inline(__always) func pyMax(_ a: Double, _ b: Double) -> Double { b > a ? b : a }
@inline(__always) func pyMin(_ a: Double, _ b: Double) -> Double { b < a ? b : a }

/// Python float floor division `a // b`.
func pyFloorDiv(_ vx: Double, _ wx: Double) -> Double {
    var mod = fmod(vx, wx)
    var div = (vx - mod) / wx
    if mod != 0 {
        if (wx < 0) != (mod < 0) { mod += wx; div -= 1.0 }
    }
    if div != 0 {
        var fd = div.rounded(.down)
        if div - fd > 0.5 { fd += 1.0 }
        return fd
    }
    return copysign(0.0, vx / wx)
}

/// Integer floor division / modulo (numpy int64 `//`, `%`).
@inline(__always) func floorDiv(_ a: Int, _ b: Int) -> Int {
    let q = a / b
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q
}
@inline(__always) func floorMod(_ a: Int, _ b: Int) -> Int { a - floorDiv(a, b) * b }

/// `np.searchsorted` on an ascending array (generic).
@inline(__always) func searchSorted<T: Comparable>(_ a: [T], _ v: T, right: Bool = false) -> Int {
    var lo = 0, hi = a.count
    while lo < hi {
        let mid = (lo + hi) >> 1
        if right ? (a[mid] <= v) : (a[mid] < v) { lo = mid + 1 } else { hi = mid }
    }
    return lo
}

/// Same on a sub-range [lo0, hi0) of `a` (returns an absolute index).
@inline(__always) func searchSorted(_ a: UnsafeBufferPointer<Double>, _ lo0: Int, _ hi0: Int, _ v: Double,
                                    right: Bool = false) -> Int {
    var lo = lo0, hi = hi0
    while lo < hi {
        let mid = (lo + hi) >> 1
        if right ? (a[mid] <= v) : (a[mid] < v) { lo = mid + 1 } else { hi = mid }
    }
    return lo
}

/// `np.interp(x, xp, fp)` exactly as numpy computes it (slope · (x − xp[j]) + fp[j]).
func npInterp(_ x: Double, _ xp: UnsafeBufferPointer<Double>, _ fp: UnsafeBufferPointer<Double>) -> Double {
    let n = xp.count
    if n == 0 { return .nan }                       // numpy raises; callers pass ≥ 3 points
    if n == 1 { return fp[0] }
    if x > xp[n - 1] { return fp[n - 1] }
    if x < xp[0] { return fp[0] }
    if x == xp[n - 1] { return fp[n - 1] }
    // binary search: j with xp[j] <= x < xp[j+1]
    var lo = 0, hi = n - 1
    while hi - lo > 1 {
        let mid = (lo + hi) >> 1
        if xp[mid] <= x { lo = mid } else { hi = mid }
    }
    let j = lo
    if xp[j] == x { return fp[j] }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
    var r = slope * (x - xp[j]) + fp[j]
    if r.isNaN {
        r = slope * (x - xp[j + 1]) + fp[j + 1]
        if r.isNaN && fp[j] == fp[j + 1] { r = fp[j] }
    }
    return r
}

/// Stable argsort by key (ties keep index order; NaN last like numpy).
func stableArgsort(_ keys: [Double]) -> [Int] {
    var idx = Array(0..<keys.count)
    idx.sort { a, b in
        let x = keys[a], y = keys[b]
        if x.isNaN || y.isNaN {
            if x.isNaN && y.isNaN { return a < b }
            return !x.isNaN
        }
        return x < y || (x == y && a < b)
    }
    return idx
}

/// Sorted unique of an Int array (`np.unique`).
func uniqueSorted(_ a: [Int]) -> [Int] {
    if a.isEmpty { return [] }
    let s = a.sorted()
    var out: [Int] = [s[0]]
    out.reserveCapacity(s.count)
    for v in s.dropFirst() where v != out[out.count - 1] { out.append(v) }
    return out
}

/// `np.flatnonzero`.
@inline(__always) func flatnonzero(_ m: [Bool]) -> [Int] {
    var out: [Int] = []
    for i in 0..<m.count where m[i] { out.append(i) }
    return out
}
