import Foundation

// Stage 4 — kinks (hits): _segments, _pair_windows, _hinge_fit, _sigma_from_rss,
// _kink_scan, _locate_stage, _kink_locate, _nms_1d, _estimate_sigma.

/// `_segments`: segment label per observation (new segment at chain starts / long gaps).
func segments(_ tObs: [Double], _ startObs: [Bool], _ maxGapS: Double) -> [Int] {
    let n = tObs.count
    var out = [Int](repeating: 0, count: n)
    var c = -1
    for i in 0..<n {
        let brk = i == 0 || (tObs[i] - tObs[i - 1] > maxGapS) || startObs[i]
        if brk { c += 1 }
        out[i] = c
    }
    return out
}

/// Two-sided windows of every consecutive same-segment observation pair (m, m+1).
/// Row r: columns 0..<W are the left side (m−W+1 … m), W..<2W the right (m+1 … m+W).
struct PairWindows {
    var m: [Int]
    var W: Int
    var idx: [Int]       // M × 2W
    var val: [Bool]      // M × 2W
    var K: Int { 2 * W }
    var count: Int { m.count }
}

/// `_pair_windows(t_obs, seg, cfg, max_pts)`.
func pairWindows(_ t: [Double], _ seg: [Int], _ cfg: LinkerConfig, maxPts: Int? = nil) -> PairWindows {
    let n = t.count
    let dtMed = n > 1 ? medianDiff(t) : 1.0 / 30.0
    let hiPts = (maxPts ?? 0) != 0 ? maxPts! : cfg.kinkMaxPts        // `max_pts or kink_max_pts`
    // np.clip(round(w / dt) + 1, lo, hi) in Double first: Int() of a huge ratio would trap
    let wr = (cfg.kinkWindowS / pyMax(dtMed, 1e-4)).rounded(.toNearestOrEven) + 1
    let W = wr.isNaN ? cfg.kinkMinPts
        : Int(Swift.min(Swift.max(wr, Double(cfg.kinkMinPts)), Double(hiPts)))
    let K = 2 * W
    let tw = pyMax(cfg.kinkWindowS, 1e-3)
    let nm1 = Swift.max(n - 1, 0)
    var ms: [Int] = [], idx: [Int] = [], val: [Bool] = []
    ms.reserveCapacity(nm1); idx.reserveCapacity(nm1 * K); val.reserveCapacity(nm1 * K)
    var li = [Int](repeating: 0, count: W), lvv = [Bool](repeating: false, count: W)
    var ri = [Int](repeating: 0, count: W), rvv = [Bool](repeating: false, count: W)
    for m in 0..<nm1 where seg[m] == seg[m + 1] {
        var nl = 0, nr = 0
        for k in 0..<W {
            let l = m - k, lc = Swift.min(Swift.max(l, 0), nm1)
            let lv = l >= 0 && seg[lc] == seg[m] && (t[m] - t[lc] <= tw || k < cfg.kinkMinPts)
            li[k] = lc; lvv[k] = lv; if lv { nl += 1 }
            let r = m + 1 + k, rc = Swift.min(Swift.max(r, 0), nm1)
            let rv = r < n && seg[rc] == seg[m] && (t[rc] - t[m + 1] <= tw || k < cfg.kinkMinPts)
            ri[k] = rc; rvv[k] = rv; if rv { nr += 1 }
        }
        guard nl >= cfg.kinkMinPts && nr >= cfg.kinkMinPts else { continue }
        ms.append(m)
        for k in stride(from: W - 1, through: 0, by: -1) { idx.append(li[k]); val.append(lvv[k]) }
        for k in 0..<W { idx.append(ri[k]); val.append(rvv[k]) }
    }
    return PairWindows(m: ms, W: W, idx: idx, val: val)
}

struct HingeOut {
    var b: [V2]          // velocity step, px/s
    var cbb: [Double]    // Var(b)/σ²
    var rss: [Double]
    var dof: [Double]
}

/// Dense P×P solve with R right-hand sides (LU, partial pivoting — LAPACK gesv).
/// `a` row-major P×P (destroyed), `b` row-major P×R (overwritten by the solution).
@inline(__always)
func luSolve(_ a: UnsafeMutablePointer<Double>, _ b: UnsafeMutablePointer<Double>, _ P: Int, _ R: Int) {
    for k in 0..<P {
        var piv = k
        var best = abs(a[k * P + k])
        for i in (k + 1)..<Swift.max(P, k + 1) where abs(a[i * P + k]) > best { best = abs(a[i * P + k]); piv = i }
        if piv != k {
            for j in 0..<P { let tmp = a[k * P + j]; a[k * P + j] = a[piv * P + j]; a[piv * P + j] = tmp }
            for j in 0..<R { let tmp = b[k * R + j]; b[k * R + j] = b[piv * R + j]; b[piv * R + j] = tmp }
        }
        let d = a[k * P + k]
        if d == 0 { continue }
        let rcp = 1.0 / d
        for i in (k + 1)..<Swift.max(P, k + 1) { a[i * P + k] *= rcp }
        for i in (k + 1)..<Swift.max(P, k + 1) {
            let l = a[i * P + k]
            if l == 0 { continue }
            for j in (k + 1)..<P { a[i * P + j] -= l * a[k * P + j] }
        }
    }
    for j in 0..<R {                                     // L y = b (unit lower)
        for k in 0..<P {
            let bk = b[k * R + j]
            if bk != 0 { for i in (k + 1)..<Swift.max(P, k + 1) { b[i * R + j] -= bk * a[i * P + k] } }
        }
        for k in stride(from: P - 1, through: 0, by: -1) {   // U x = y
            if b[k * R + j] != 0 {
                b[k * R + j] /= a[k * P + k]
                let bk = b[k * R + j]
                for i in 0..<k { b[i * R + j] -= bk * a[i * P + k] }
            }
        }
    }
}

/// `_hinge_fit`: batched LS over each window, s = (t − τ)/tw:
///   p(t) = Σ_d c_d s^d [+ c₊ s₊²] + b s₊. Row r uses window `rows?[r] ?? r`.
func hingeFit(_ t: [Double], _ p: [V2], _ pw: PairWindows, rows: [Int]?, tau: [Double], tw: Double,
              deg: Int, freeCurv: Bool = true) -> HingeOut {
    let M = tau.count, K = pw.K
    let P = deg + 2 + (freeCurv ? 1 : 0)
    var bOut = [V2](repeating: .zero, count: M)
    var cbb = [Double](repeating: .infinity, count: M)
    var rss = [Double](repeating: 0, count: M)
    var dof = [Double](repeating: 0, count: M)
    let X = UnsafeMutablePointer<Double>.allocate(capacity: K * P)
    let Y = UnsafeMutablePointer<Double>.allocate(capacity: K * 2)
    let A = UnsafeMutablePointer<Double>.allocate(capacity: P * P)
    let rhs = UnsafeMutablePointer<Double>.allocate(capacity: P * 3)
    let sol = UnsafeMutablePointer<Double>.allocate(capacity: P * 3)
    defer { X.deallocate(); Y.deallocate(); A.deallocate(); rhs.deallocate(); sol.deallocate() }
    let tw2 = pow(tw, 2.0)
    pw.idx.withUnsafeBufferPointer { idxB in
    pw.val.withUnsafeBufferPointer { valB in
    t.withUnsafeBufferPointer { tB in
    p.withUnsafeBufferPointer { pB in
        for r in 0..<M {
            let w = rows?[r] ?? r
            let base = w * K
            let ta = tau[r]
            var nv = 0
            for k in 0..<K {
                let v: Double = valB[base + k] ? 1.0 : 0.0
                if valB[base + k] { nv += 1 }
                let o = idxB[base + k]
                let sv = (tB[o] - ta) / tw
                let sp = sv > 0.0 ? sv : 0.0
                var c = 1.0
                X[k * P] = c * v
                if deg > 0 { for d in 1...deg { c = c * sv; X[k * P + d] = c * v } }
                var col = deg + 1
                if freeCurv { X[k * P + col] = (sp * sp) * v; col += 1 }
                X[k * P + col] = sp * v
                let q = pB[o]
                Y[2 * k] = q.x * v; Y[2 * k + 1] = q.y * v
            }
            dof[r] = 2.0 * Double(nv - P)
            for i in 0..<P {
                for j in 0..<P {
                    var s = 0.0
                    for k in 0..<K { s += X[k * P + i] * X[k * P + j] }
                    A[i * P + j] = s
                }
                A[i * P + i] += 1e-9
                var s0 = 0.0, s1 = 0.0
                for k in 0..<K { s0 += X[k * P + i] * Y[2 * k]; s1 += X[k * P + i] * Y[2 * k + 1] }
                rhs[i * 3] = s0; rhs[i * 3 + 1] = s1; rhs[i * 3 + 2] = i == P - 1 ? 1.0 : 0.0
            }
            for i in 0..<(P * 3) { sol[i] = rhs[i] }
            luSolve(A, sol, P, 3)
            bOut[r] = V2(sol[(P - 1) * 3] / tw, sol[(P - 1) * 3 + 1] / tw)
            cbb[r] = npMax(sol[(P - 1) * 3 + 2], 1e-12) / tw2
            var yy = 0.0
            for k in 0..<(2 * K) { yy += Y[k] * Y[k] }
            var rs = 0.0
            for i in 0..<P { rs += rhs[i * 3] * sol[i * 3]; rs += rhs[i * 3 + 1] * sol[i * 3 + 1] }
            rss[r] = npMax(yy - rs, 0.0)
        }
    }}}}
    return HingeOut(b: bOut, cbb: cbb, rss: rss, dof: dof)
}

/// `_sigma_from_rss`: robust per-axis σ from window residuals (NaN if < 20 windows).
func sigmaFromRss(_ rss: [Double], _ dof: [Double]) -> Double {
    var r: [Double] = []
    for i in 0..<rss.count where dof[i] >= 2 {
        let k = dof[i]
        r.append(rss[i] / k / pow(1.0 - 2.0 / (9.0 * k), 3.0))
    }
    if r.count < 20 { return .nan }
    return npMedian(r).squareRoot()
}

struct KinkScan {
    var m: [Int]
    var z: [Double]
    var dv: [Double]
    var sigma: Double
}

/// `_kink_scan(t_obs, p, seg, sigma, cfg)`: GLRT z of a velocity step between
/// every consecutive observation pair. sigma nil → estimated from the residuals.
func kinkScan(_ t: [Double], _ p: [V2], _ seg: [Int], sigma: Double?, _ cfg: LinkerConfig) -> KinkScan {
    let pw = pairWindows(t, seg, cfg)
    if pw.count == 0 {
        let s: Double = (sigma == nil || sigma! == 0) ? .nan : sigma!    // float(sigma or nan)
        return KinkScan(m: [], z: [], dv: [], sigma: s)
    }
    let tm = pw.m.map { 0.5 * (t[$0] + t[$0 + 1]) }
    let h = hingeFit(t, p, pw, rows: nil, tau: tm, tw: pyMax(cfg.kinkWindowS, 1e-3), deg: cfg.kinkDeg)
    let sig = sigma ?? sigmaFromRss(h.rss, h.dof)
    let sg = sig.isFinite ? sig : 1.0
    let sgc = pyMax(sg, 1e-6)
    var z = [Double](repeating: 0, count: pw.count), dv = z
    for i in 0..<pw.count {
        dv[i] = hypot(h.b[i].x, h.b[i].y)
        z[i] = dv[i] / (sgc * h.cbb[i].squareRoot())
    }
    return KinkScan(m: pw.m, z: z, dv: dv, sigma: sig)
}

/// `_locate_stage`: min-RSS hinge position inside every pair of the kink's window
/// within `reach` of τ (n_grid positions per pair).
func locateStage(_ t: [Double], _ p: [V2], _ seg: [Int], _ m: [Int], _ tau: [Double], _ cfg: LinkerConfig,
                 maxPts: Int?, deg: Int, reach: Double, nGrid: Int) -> ([Int], [Double]) {
    let pw = pairWindows(t, seg, cfg, maxPts: maxPts)
    if pw.count == 0 || m.isEmpty { return (m, tau) }
    let mm = pw.m
    let tm = mm.map { 0.5 * (t[$0] + t[$0 + 1]) }
    var g = [Double](repeating: 0, count: nGrid)                  // np.linspace(0, 1, n_grid)
    if nGrid > 1 {
        let step = 1.0 / Double(nGrid - 1)
        for i in 0..<nGrid { g[i] = Double(i) * step }
        g[nGrid - 1] = 1.0
    }
    var rowsT: [Double] = [], rowsM: [Int] = [], rowsI: [Int] = [], owner: [Int] = []
    let K = pw.K
    tm.withUnsafeBufferPointer { tmB in
        for j in 0..<m.count {
            let pos = Swift.min(searchSorted(mm, m[j]), mm.count - 1)
            guard mm[pos] == m[j] else { continue }
            let k = pos
            var vmin = Double.infinity, vmax = -Double.infinity
            for c in 0..<K where pw.val[k * K + c] {
                let tv = t[pw.idx[k * K + c]]
                if tv < vmin { vmin = tv }
                if tv > vmax { vmax = tv }
            }
            let lo = searchSorted(tmB, 0, tmB.count, pyMax(tau[j] - reach, vmin) - 1e-9)
            let hi = searchSorted(tmB, 0, tmB.count, pyMin(tau[j] + reach, vmax) + 1e-9, right: true)
            let a0 = Swift.min(lo, k), a1 = Swift.max(hi, k + 1)
            for ks in a0..<Swift.max(a1, a0) {
                guard seg[mm[ks]] == seg[mm[k]] && t[mm[ks]] >= vmin && t[mm[ks] + 1] <= vmax else { continue }
                let ta = t[mm[ks]], tb = t[mm[ks] + 1]
                for gi in g {
                    rowsT.append(ta + gi * (tb - ta))
                    rowsM.append(mm[ks]); rowsI.append(k); owner.append(j)
                }
            }
        }
    }
    if owner.isEmpty { return (m, tau) }
    let rss = hingeFit(t, p, pw, rows: rowsI, tau: rowsT, tw: pyMax(cfg.kinkWindowS, 1e-3), deg: deg).rss
    var mOut = m, tauOut = tau
    var r = 0
    while r < owner.count {                                    // lexsort((rss, own)): first min per owner
        let o = owner[r]
        var best = r
        var e = r + 1
        while e < owner.count && owner[e] == o {
            let a = rss[e], b = rss[best]
            if (!a.isNaN && b.isNaN) || a < b { best = e }
            e += 1
        }
        mOut[o] = rowsM[best]; tauOut[o] = rowsT[best]
        r = e
    }
    return (mOut, tauOut)
}

/// `_kink_locate`: hit time τ and pair for detected kinks (coarse + fine min-RSS).
func kinkLocate(_ t: [Double], _ p: [V2], _ seg: [Int], _ m: [Int], _ cfg: LinkerConfig) -> ([Int], [Double]) {
    if m.isEmpty { return (m, []) }
    let nt = t.count
    let tau = m.map { 0.5 * (t[$0] + t[Swift.min($0 + 1, nt - 1)]) }
    let dt = nt > 1 ? medianDiff(t) : 1.0 / 30.0
    let ws = (cfg.kinkWindowS / pyMax(dt, 1e-4)).rounded(.toNearestOrEven) + 1     // only compared below
    let wScan = ws.isNaN ? cfg.kinkMaxPts : Int(Swift.max(Swift.min(ws, Double(cfg.kinkMaxPts)), -1e9))
    var mm = m, tt = tau
    if cfg.kinkLocCoarse && wScan >= 2 * cfg.kinkLocPts {
        (mm, tt) = locateStage(t, p, seg, mm, tt, cfg, maxPts: nil, deg: cfg.kinkDeg, reach: .infinity, nGrid: 3)
    }
    return locateStage(t, p, seg, mm, tt, cfg, maxPts: Swift.min(cfg.kinkLocPts, cfg.kinkMaxPts),
                       deg: cfg.kinkLocDeg, reach: pyMax(cfg.kinkLocS, 2.5 * dt), nGrid: 9)
}

/// `_nms_1d`: greedy NMS in time (highest z first). Sorted indices.
func nms1d(_ tau: [Double], _ z: [Double], _ sep: Double) -> [Int] {
    var keptT: [Double] = []
    var keep: [Int] = []
    for i in stableArgsort(z.map { -$0 }) {
        let ti = tau[i]
        let pos = searchSorted(keptT, ti)                    // bisect_left
        if pos > 0 && ti - keptT[pos - 1] < sep { continue }
        if pos < keptT.count && keptT[pos] - ti < sep { continue }
        keptT.insert(ti, at: pos)
        keep.append(i)
    }
    return keep.sorted()
}

/// `_estimate_sigma`: per-axis detector noise from the kink-scan residuals,
/// clipped to [0.4, 2.5]·prior; prior on short tracks.
func estimateSigma(_ t: [Double], _ p: [V2], _ seg: [Int], _ prior: Double, _ cfg: LinkerConfig) -> Double {
    if t.count < 4 * cfg.kinkMinPts { return prior }
    let s = kinkScan(t, p, seg, sigma: nil, cfg).sigma
    if !s.isFinite { return prior }
    return npClip(s, 0.4 * prior, 2.5 * prior)
}
