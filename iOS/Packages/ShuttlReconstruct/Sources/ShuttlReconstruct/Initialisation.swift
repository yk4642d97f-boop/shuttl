import Foundation
import ShuttlCore
import simd

// Port of the initialisation part of reconstruct3d.py: _ray_box, _v0_linear,
// _outlier_mask, _hypotheses, _shoot, _hypotheses_contact (+ numpy helpers with the
// exact numpy semantics the solver path depends on).

// MARK: - numpy helpers

/// `np.linspace(start, stop, num)` (endpoint=True): y[i] = i*step + start, y[-1] = stop.
func linspace(_ start: Double, _ stop: Double, _ num: Int) -> [Double] {
    if num <= 0 { return [] }
    if num == 1 { return [start] }
    let div = Double(num - 1)
    let delta = stop - start
    let step = delta / div
    var y = [Double](repeating: 0, count: num)
    for i in 0..<num {
        y[i] = step == 0 ? (Double(i) / div) * delta + start : Double(i) * step + start
    }
    y[num - 1] = stop
    return y
}

/// `np.arange(start, stop, step)` for floats: len = ceil((stop - start) / step),
/// y[0] = start, y[1] = start + step, y[i] = start + i*(y[1] - y[0]) (numpy's DOUBLE_fill).
func arange(_ start: Double, _ stop: Double, _ step: Double) -> [Double] {
    let lenD = ((stop - start) / step).rounded(.up)
    guard lenD.isFinite, lenD > 0 else { return [] }
    let n = Int(lenD)
    var y = [Double](repeating: 0, count: n)
    y[0] = start
    if n == 1 { return y }
    y[1] = start + step
    let delta = y[1] - y[0]
    if n > 2 { for i in 2..<n { y[i] = start + Double(i) * delta } }
    return y
}

/// `np.argmin` (first minimum; a NaN wins like numpy).
func argmin(_ xs: [Double]) -> Int {
    var best = 0
    for (i, x) in xs.enumerated() {
        if x.isNaN { return i }
        if x < xs[best] { best = i }
    }
    return best
}

/// Stable sort of indices by key using only `<` (Python's list.sort(key=...)).
func stableOrder(_ keys: [Double]) -> [Int] {
    var idx = Array(keys.indices)
    if idx.count < 2 { return idx }
    var tmp = idx
    var width = 1
    let n = idx.count
    while width < n {
        var lo = 0
        while lo < n {
            let mid = min(lo + width, n), hi = min(lo + 2 * width, n)
            var i = lo, j = mid, k = lo
            while i < mid && j < hi {
                if keys[idx[j]] < keys[idx[i]] { tmp[k] = idx[j]; j += 1 } else { tmp[k] = idx[i]; i += 1 }
                k += 1
            }
            while i < mid { tmp[k] = idx[i]; i += 1; k += 1 }
            while j < hi { tmp[k] = idx[j]; j += 1; k += 1 }
            lo += 2 * width
        }
        swap(&idx, &tmp)
        width *= 2
    }
    return idx
}

/// LU with partial pivoting (first largest |pivot|, multipliers a·(1/p)); fails only on an
/// exactly zero pivot — the `numpy.linalg.solve` / `inv` (LAPACK gesv) contract, which
/// ShuttlCore.LinAlg.solve tightens to a relative 1e-13 pivot. Solves A X = B (n × n, n × m).
func gesv(_ a: [Double], _ n: Int, _ b: [Double], _ m: Int) -> [Double]? {
    var lu = a, x = b
    for col in 0..<n {
        var piv = col
        var best = abs(lu[col * n + col])
        var r = col + 1
        while r < n {
            let v = abs(lu[r * n + col])
            if v > best { best = v; piv = r }
            r += 1
        }
        if lu[piv * n + col] == 0 { return nil }
        if piv != col {
            for c in 0..<n { lu.swapAt(col * n + c, piv * n + c) }
            for c in 0..<m { x.swapAt(col * m + c, piv * m + c) }
        }
        let inv = 1.0 / lu[col * n + col]
        r = col + 1
        while r < n {
            let f = lu[r * n + col] * inv
            lu[r * n + col] = f
            if f != 0 {
                for c in (col + 1)..<n { lu[r * n + c] -= f * lu[col * n + c] }
                for c in 0..<m { x[r * m + c] -= f * x[col * m + c] }
            }
            r += 1
        }
    }
    var rr = n - 1
    while rr >= 0 {
        for c in 0..<m {
            var s = x[rr * m + c]
            var k = rr + 1
            while k < n { s -= lu[rr * n + k] * x[k * m + c]; k += 1 }
            x[rr * m + c] = s / lu[rr * n + rr]
        }
        rr -= 1
    }
    return x
}

/// `np.linalg.inv` (LAPACK semantics: nil only for an exactly singular matrix).
func invert(_ a: [Double], _ n: Int) -> [Double]? {
    var eye = [Double](repeating: 0, count: n * n)
    for i in 0..<n { eye[i * n + i] = 1 }
    return gesv(a, n, eye, n)
}

@inline(__always) func norm3(_ v: SIMD3<Double>) -> Double { (v.x * v.x + v.y * v.y + v.z * v.z).squareRoot() }

// MARK: - initialisation

/// `_ray_box`: parameter interval [s0, s1] (s > 0) of the ray o + s d inside a box.
func rayBox(_ o: SIMD3<Double>, _ d: SIMD3<Double>, _ lo: SIMD3<Double>, _ hi: SIMD3<Double>) -> (Double, Double) {
    let a = (lo - o) / d, b = (hi - o) / d
    var s0 = -Double.infinity, s1 = Double.infinity
    var any0 = false, any1 = false
    for k in 0..<3 {
        // np.minimum / np.maximum propagate NaN; nanmax / nanmin then skip it
        let mn = (a[k].isNaN || b[k].isNaN) ? Double.nan : Swift.min(a[k], b[k])
        let mx = (a[k].isNaN || b[k].isNaN) ? Double.nan : Swift.max(a[k], b[k])
        if !mn.isNaN { s0 = any0 ? Swift.max(s0, mn) : mn; any0 = true }
        if !mx.isNaN { s1 = any1 ? Swift.min(s1, mx) : mx; any1 = true }
    }
    if !any0 { s0 = .nan }
    if !any1 { s1 = .nan }
    return (max(s0, 0.3), s1)
}

/// `_v0_linear`: v0 so that p0 + v0 τ + ½ g t² lies on the (unit) rays d_i: linear LS in
/// v0 (normal matrix τ² (I − d dᵀ)). Drag enters through the time warp
/// τ = L ln(1 + |v| t / L) / |v|, iterated from |v| = 0.
func v0Linear(_ p0: SIMD3<Double>, _ o: SIMD3<Double>, _ d: [SIMD3<Double>], _ tr: [Double], _ L: Double) -> SIMD3<Double> {
    var v = SIMD3<Double>(0, 0, 0)
    var A = [Double](repeating: 0, count: 9)
    var b = [Double](repeating: 0, count: 3)
    var out = [Double](repeating: 0, count: 3)
    var Lc = [Double](repeating: 0, count: 9)
    var yv = [Double](repeating: 0, count: 3)
    for _ in 0..<4 {
        let sp = max((v.x * v.x + v.y * v.y + v.z * v.z).squareRoot(), 1e-3)
        for k in 0..<9 { A[k] = 0 }
        for k in 0..<3 { b[k] = 0 }
        for i in 0..<tr.count {
            let tau = L * log1p(sp * tr[i] / L) / sp
            let di = d[i]
            let q0 = p0.x - o.x, q1 = p0.y - o.y, q2 = p0.z - o.z - 0.5 * R3.g * (tr[i] * tr[i])
            let dq = di.x * q0 + di.y * q1 + di.z * q2
            let q = [q0 - dq * di.x, q1 - dq * di.y, q2 - dq * di.z]
            let dd = [di.x, di.y, di.z]
            for a in 0..<3 {
                b[a] -= tau * q[a]
                for c in 0..<3 {
                    A[a * 3 + c] += tau * tau * ((a == c ? 1.0 : 0.0) - dd[a] * dd[c])
                }
            }
        }
        for a in 0..<3 { A[a * 3 + a] += 1e-9 }
        let ok = A.withUnsafeBufferPointer { ab in b.withUnsafeBufferPointer { bb in
            out.withUnsafeMutableBufferPointer { ob in Lc.withUnsafeMutableBufferPointer { lb in
                yv.withUnsafeMutableBufferPointer { yb in
                    cholSolve(ab.baseAddress!, 3, bb.baseAddress!, ob.baseAddress!, lb.baseAddress!, yb.baseAddress!)
                } } } } }
        if !ok { break }
        v = SIMD3(out[0], out[1], out[2])
    }
    let sp = (v.x * v.x + v.y * v.y + v.z * v.z).squareRoot()
    if sp > R3.maxSpeed { v *= R3.maxSpeed / sp }
    return v
}

/// `_outlier_mask`: true = likely outlier, > thr px from a quadratic fit of its
/// (unflagged) neighbours; iterated so an outlier does not mask itself via its neighbours.
func outlierMask(_ t: [Double], _ uv: [SIMD2<Double>], half: Int = 4, thr: Double = 10.0,
                 passes: Int = 3) -> [Bool] {
    let n = t.count
    var bad = [Bool](repeating: false, count: n)
    if n < 6 { return bad }
    var Arows: [Double] = [], Brows: [Double] = []
    Arows.reserveCapacity(3 * 2 * half); Brows.reserveCapacity(2 * 2 * half)
    for _ in 0..<passes {
        var new = [Bool](repeating: false, count: n)
        for i in 0..<n {
            Arows.removeAll(keepingCapacity: true)
            Brows.removeAll(keepingCapacity: true)
            var m = 0
            for j in max(0, i - 2 * half)..<min(n, i + 2 * half + 1) {
                if j == i || bad[j] || m >= 2 * half { continue }
                if abs(j - i) > half && m >= 4 { continue }
                let tt = t[j] - t[i]
                Arows += [1.0, tt, tt * tt]
                Brows += [uv[j].x, uv[j].y]
                m += 1
            }
            if m < 5 { continue }
            guard let c = LinAlg.lstsq(Matrix(rows: m, cols: 3, data: Arows), Matrix(rows: m, cols: 2, data: Brows)) else {
                continue
            }
            var res = [Double](repeating: 0, count: m)
            for k in 0..<m {
                let a0 = Arows[3 * k], a1 = Arows[3 * k + 1], a2 = Arows[3 * k + 2]
                let du = a0 * c[0, 0] + a1 * c[1, 0] + a2 * c[2, 0] - Brows[2 * k]
                let dv = a0 * c[0, 1] + a1 * c[1, 1] + a2 * c[2, 1] - Brows[2 * k + 1]
                res[k] = (du * du + dv * dv).squareRoot()
            }
            let ex = c[0, 0] - uv[i].x, ey = c[0, 1] - uv[i].y
            let e = (ex * ex + ey * ey).squareRoot()
            new[i] = e > max(thr, 6.0 * Num.median(res))
        }
        if new == bad { break }
        bad = new
    }
    return bad
}

/// `_shoot`: launch velocity taking the drag flight from pC to q in dt seconds (Newton on
/// v with the ODE sensitivities), or nil if unreachable.
func shoot(_ pC: SIMD3<Double>, _ q: SIMD3<Double>, _ dt: Double, _ L: Double, _ ws: ODEWorkspace) -> SIMD3<Double>? {
    var v = (q - pC) / dt
    v.z += 0.5 * R3.g * dt
    var P = SIMD3<Double>.zero, V = SIMD3<Double>.zero
    var S = [Double](repeating: 0, count: 21)
    var tt = dt
    var err = Double.infinity
    for _ in 0..<20 {
        S.withUnsafeMutableBufferPointer { sb in
            ws.propagate(pC, v, L, &tt, 1, R3.ncol - 1, &P, &V, sb.baseAddress!)
        }
        let r = P - q
        err = norm3(r)
        if err < 2e-3 { break }
        // S[0, :, 3:6]: row r, column j = dp_r / dv0_j
        var M = [Double](repeating: 0, count: 9)
        for rr in 0..<3 { for j in 0..<3 { M[rr * 3 + j] = S[(3 + j) * 3 + rr] } }
        guard let st = gesv(M, 3, [r.x, r.y, r.z], 1) else { return nil }
        var step = SIMD3(st[0], st[1], st[2])
        let nrm = norm3(step)
        if nrm > 30.0 { step *= 30.0 / nrm }
        v = v - step
        let sp = norm3(v)
        if sp > R3.maxSpeed { v *= R3.maxSpeed / sp }
    }
    return err < 0.05 ? v : nil
}

@inline(__always) func theta(_ p: SIMD3<Double>, _ v: SIMD3<Double>, _ L: Double, _ fitDrag: Bool) -> [Double] {
    fitDrag ? [p.x, p.y, p.z, v.x, v.y, v.z, log(L)] : [p.x, p.y, p.z, v.x, v.y, v.z]
}

/// `_hypotheses`: initial θ candidates — a sweep of start depths along the first ray
/// (uniform in inverse depth over the court volume) plus depths at the hitter's feet
/// prior; v0 for each from the early rays by linear least squares.
func hypotheses(_ prob: Problem, _ t: [Double], _ uv: [SIMD2<Double>], feetOnly: Bool = false) -> [[Double]] {
    let cam = prob.cam, pr = prob.prior
    if feetOnly && pr.startXYWorld == nil { return [] }
    var good = outlierMask(t, uv).map { !$0 }
    if good.filter({ $0 }).count < 5 { good = [Bool](repeating: true, count: good.count) }
    good[0] = true                                   // the first ray anchors the depth sweep
    var tg: [Double] = [], uvg: [SIMD2<Double>] = []
    for i in 0..<t.count where good[i] { tg.append(t[i]); uvg.append(uv[i]) }
    let o = cam.center
    let d = uvg.map { cam.ray($0).direction }
    let tr = tg.map { $0 - tg[0] }
    // short-time window only; >= 2 rays; at most 10 spread over the window
    var early = tr.indices.filter { tr[$0] <= 0.25 }
    if early.count < 2 { early = Array(0..<min(2, tr.count)) }
    if early.count > 10 {
        let pick = linspace(0, Double(early.count - 1), 10).map { Int($0.rounded(.toNearestOrEven)) }
        early = pick.map { early[$0] }
    }
    let (s0, s1) = rayBox(o, d[0], R3.volumeLo, R3.volumeHi)
    var depths: [Double] = []
    if s1 > s0 && !feetOnly {
        // uniform in inverse depth → denser sampling far away where rays graze
        depths += linspace(1 / s0, 1 / s1, R3.nDepths).map { 1 / $0 }
    }
    if let sxy = pr.startXYWorld {
        // point on the first ray horizontally closest to the hitter's feet
        let dh = SIMD2(d[0].x, d[0].y)
        let s = ((sxy.x - o.x) * dh.x + (sxy.y - o.y) * dh.y) / max(dh.x * dh.x + dh.y * dh.y, 1e-9)
        if s > 0.3 {
            depths.append(s)
            depths += [s * 0.93, s * 1.07]
        }
    }
    let L = prob.LFixed
    var hyps: [[Double]] = []
    let dE = early.map { d[$0] }, trE = early.map { tr[$0] }
    for s in depths {
        let p0 = o + s * d[0]
        let v0 = v0Linear(p0, o, dE, trE, L)
        hyps.append(theta(p0, v0, L, prob.fitDrag))
    }
    // a detection gap right after the start leaves the linear v0 ill-posed: also shoot
    // each start depth to the first ray after the gap, at the paired depth of that
    // ray's sweep and its two neighbours
    let span = tr[early[early.count - 1]]
    if span < 0.15 && depths.count >= R3.nDepths, let j = tr.indices.first(where: { tr[$0] > span + 0.05 }) {
        let (q0, q1) = rayBox(o, d[j], R3.volumeLo, R3.volumeHi)
        if q1 > q0 {
            let qd = linspace(1 / q0, 1 / q1, R3.nDepths).map { 1 / $0 }
            for i in 0..<R3.nDepths {
                let p0 = o + depths[i] * d[0]
                for k in [i - 1, i, i + 1] where k >= 0 && k < R3.nDepths {
                    if let v0 = shoot(p0, o + qd[k] * d[j], tr[j], L, prob.ws) {
                        hyps.append(theta(p0, v0, L, prob.fitDrag))
                    }
                }
            }
        }
    }
    return hyps
}

/// `_hypotheses_contact`: hypotheses at the contact epoch (prob.tRef < first observation):
/// contact points (hitter's feet, else a line down the hitter's half) × contact heights,
/// each shot through a depth sweep on the first observed ray.
func hypothesesContact(_ prob: Problem, _ t: [Double], _ uv: [SIMD2<Double>]) -> [[Double]] {
    let pr = prob.prior
    let good = outlierMask(t, uv).map { !$0 }
    let i0 = good.firstIndex(of: true) ?? 0
    let o = prob.cam.center
    let d0 = prob.cam.ray(uv[i0]).direction
    let dt = t[i0] - prob.tRef
    var xys: [SIMD2<Double>]
    if let s = pr.startXYWorld {
        xys = [s]
    } else {
        let sgn = pr.expectedSide == .near ? -1.0 : 1.0
        xys = [2.0, 4.0, 6.0].map { SIMD2(sgn * $0, 0.0) }
    }
    let (lo, hi) = pr.startHeight ?? (0.05, 3.5)
    let heights = Array(Set([1.0, 2.0, 2.7].map { min(max($0, lo), hi) })).sorted()
    let (s0, s1) = rayBox(o, d0, R3.volumeLo, R3.volumeHi)
    if !(s1 > s0) { return [] }
    let depths = linspace(1 / s0, 1 / s1, R3.nDepths).map { 1 / $0 }
    let L = prob.LFixed
    var hyps: [[Double]] = []
    for xy in xys {
        for h in heights {
            let pC = SIMD3(xy.x, xy.y, h)
            for sd in depths {
                if let v = shoot(pC, o + sd * d0, dt, L, prob.ws) {
                    hyps.append(theta(pC, v, L, prob.fitDrag))
                }
            }
        }
    }
    return hyps
}
