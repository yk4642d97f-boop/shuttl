/// numpy reduction / sorting primitives whose exact behaviour affects tie-breaking:
/// pairwise summation (np.add.reduce along a contiguous axis) and np.argsort's default
/// introsort ('quicksort', numpy 1.26 generic C++ path — not stable).
import Foundation

enum NumpyCompat {
    /// numpy pairwise_sum (loops_utils.h.src) for any floating type; `a` read with `stride`.
    static func pairwiseSum<T: BinaryFloatingPoint>(_ a: UnsafeBufferPointer<T>, _ start: Int, _ n: Int) -> T {
        if n < 8 {
            var res: T = 0
            for i in 0..<n { res += a[start + i] }
            return res
        } else if n <= 128 {
            var r0 = a[start], r1 = a[start + 1], r2 = a[start + 2], r3 = a[start + 3]
            var r4 = a[start + 4], r5 = a[start + 5], r6 = a[start + 6], r7 = a[start + 7]
            var i = 8
            let lim = n - (n % 8)
            while i < lim {
                r0 += a[start + i]; r1 += a[start + i + 1]; r2 += a[start + i + 2]; r3 += a[start + i + 3]
                r4 += a[start + i + 4]; r5 += a[start + i + 5]; r6 += a[start + i + 6]; r7 += a[start + i + 7]
                i += 8
            }
            var res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
            while i < n { res += a[start + i]; i += 1 }
            return res
        } else {
            var n2 = n / 2
            n2 -= n2 % 8
            return pairwiseSum(a, start, n2) + pairwiseSum(a, start + n2, n - n2)
        }
    }

    /// `np.sum(x)` for a contiguous 1-D array (identity 0 + pairwise sum).
    static func sum<T: BinaryFloatingPoint>(_ x: [T]) -> T {
        x.withUnsafeBufferPointer { 0 + pairwiseSum($0, 0, x.count) }
    }

    @inline(__always) static func less(_ a: Double, _ b: Double) -> Bool { a < b || (b != b && a == a) }

    /// `np.argsort(v)` (kind='quicksort'): numpy's aquicksort_ introsort.
    static func argsortQuick(_ v: [Double]) -> [Int] {
        let num = v.count
        var tosort = Array(0..<num)
        guard num > 1 else { return tosort }
        tosort.withUnsafeMutableBufferPointer { t in
            v.withUnsafeBufferPointer { v in
                var stack: [Int] = []
                var depthStack: [Int] = []
                var pl = 0, pr = num - 1
                var cdepth = (Int.bitWidth - 1 - num.leadingZeroBitCount) * 2   // npy_get_msb(num) * 2
                outer: while true {
                    if cdepth < 0 {
                        aheapsort(v, t, pl, pr - pl + 1)
                    } else {
                        while pr - pl > 16 {
                            let pm = pl + ((pr - pl) >> 1)
                            if less(v[t[pm]], v[t[pl]]) { t.swapAt(pm, pl) }
                            if less(v[t[pr]], v[t[pm]]) { t.swapAt(pr, pm) }
                            if less(v[t[pm]], v[t[pl]]) { t.swapAt(pm, pl) }
                            let vp = v[t[pm]]
                            var pi = pl, pj = pr - 1
                            t.swapAt(pm, pj)
                            while true {
                                repeat { pi += 1 } while less(v[t[pi]], vp)
                                repeat { pj -= 1 } while less(vp, v[t[pj]])
                                if pi >= pj { break }
                                t.swapAt(pi, pj)
                            }
                            let pk = pr - 1
                            t.swapAt(pi, pk)
                            if pi - pl < pr - pi {
                                stack.append(pi + 1); stack.append(pr)
                                pr = pi - 1
                            } else {
                                stack.append(pl); stack.append(pi - 1)
                                pl = pi + 1
                            }
                            cdepth -= 1
                            depthStack.append(cdepth)
                        }
                        // insertion sort
                        if pl < pr {
                            for pi in (pl + 1)...pr {
                                let vi = t[pi]
                                let vp = v[vi]
                                var pj = pi, pk = pi - 1
                                while pj > pl && less(vp, v[t[pk]]) {
                                    t[pj] = t[pk]; pj -= 1; pk -= 1
                                }
                                t[pj] = vi
                            }
                        }
                    }
                    // stack_pop
                    if stack.isEmpty { break outer }
                    pr = stack.removeLast()
                    pl = stack.removeLast()
                    cdepth = depthStack.removeLast()
                }
            }
        }
        return tosort
    }

    /// numpy aheapsort_ on t[start ..< start+n].
    static func aheapsort(_ v: UnsafeBufferPointer<Double>, _ t: UnsafeMutableBufferPointer<Int>, _ start: Int, _ n0: Int) {
        var n = n0
        @inline(__always) func a(_ k: Int) -> Int { t[start + k - 1] }
        @inline(__always) func set(_ k: Int, _ x: Int) { t[start + k - 1] = x }
        var l = n >> 1
        while l > 0 {
            let tmp = a(l)
            var i = l, j = l << 1
            while j <= n {
                if j < n && less(v[a(j)], v[a(j + 1)]) { j += 1 }
                if less(v[tmp], v[a(j)]) { set(i, a(j)); i = j; j += j } else { break }
            }
            set(i, tmp)
            l -= 1
        }
        while n > 1 {
            let tmp = a(n)
            set(n, a(1))
            n -= 1
            var i = 1, j = 2
            while j <= n {
                if j < n && less(v[a(j)], v[a(j + 1)]) { j += 1 }
                if less(v[tmp], v[a(j)]) { set(i, a(j)); i = j; j += j } else { break }
            }
            set(i, tmp)
        }
    }

    /// `np.argsort(v, kind="stable")`
    static func argsortStable<T: Comparable>(_ v: [T]) -> [Int] {
        Array(0..<v.count).sorted { v[$0] < v[$1] || (v[$0] == v[$1] && $0 < $1) }
    }
}

/// np.cross for 3-vectors (separate multiply / subtract, no fused ops).
@inline(__always)
func cross3(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> SIMD3<Double> {
    let c0 = a.y * b.z - a.z * b.y
    let c1 = a.z * b.x - a.x * b.z
    let c2 = a.x * b.y - a.y * b.x
    return SIMD3(c0, c1, c2)
}

@inline(__always)
func dot3(_ a: SIMD3<Double>, _ b: SIMD3<Double>) -> Double { a.x * b.x + a.y * b.y + a.z * b.z }
@inline(__always)
func dot2(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double { a.x * b.x + a.y * b.y }
@inline(__always)
func norm2(_ a: SIMD2<Double>) -> Double { (a.x * a.x + a.y * a.y).squareRoot() }
@inline(__always)
func norm3(_ a: SIMD3<Double>) -> Double { (a.x * a.x + a.y * a.y + a.z * a.z).squareRoot() }
