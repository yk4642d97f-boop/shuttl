import Foundation

// Stage 5 — piecewise robust smoothing: _pieces, _solve_pieces (banded MAP with an
// integrated-white-jerk prior; LAPACK dpbsv reproduced by a band-Cholesky below).

struct Pieces {
    var a: [Int], b: [Int], lo: [Int], hi: [Int]
}

/// `_pieces(fr, t, seg, kink_m, kink_tau, cfg)`: observation ranges [a, b) and frame
/// ranges [lo, hi] of flight pieces; at a kink both sides extrapolate toward τ.
func pieces(_ fr: [Int], _ t: [Double], _ seg: [Int], _ kinkM: [Int], _ kinkTau: [Double],
            _ cfg: LinkerConfig) -> Pieces {
    let nObs = fr.count
    var cut = [Bool](repeating: false, count: nObs + 1)
    cut[0] = true; cut[nObs] = true
    if nObs > 1 { for i in 1..<nObs where seg[i] != seg[i - 1] { cut[i] = true } }
    for k in kinkM { cut[k + 1] = true }
    var edges: [Int] = []
    for i in 0...nObs where cut[i] { edges.append(i) }
    let np_ = edges.count - 1
    var a = [Int](repeating: 0, count: np_), b = a, lo = a, hi = a
    for p in 0..<np_ {
        a[p] = edges[p]; b[p] = edges[p + 1]
        lo[p] = fr[a[p]]; hi[p] = fr[b[p] - 1]
    }
    var tauAt = [Double](repeating: .nan, count: nObs)
    for (k, m) in kinkM.enumerated() { tauAt[m] = kinkTau[k] }
    let ex = cfg.maxExtrapFrames
    for p in 0..<np_ {
        if ex <= 0 { break }
        var m = b[p] - 1                                   // extend forward to τ
        if m < nObs - 1 && tauAt[m].isFinite {
            let f = fr[m]
            while hi[p] - f < ex && hi[p] + 1 < fr[m + 1] && t[hi[p] + 1] <= tauAt[m] { hi[p] += 1 }
        }
        m = a[p] - 1                                       // extend backward to τ
        if m >= 0 && tauAt[m].isFinite {
            let f = fr[a[p]]
            while f - lo[p] < ex && lo[p] - 1 > fr[m] && t[lo[p] - 1] > tauAt[m] { lo[p] -= 1 }
        }
    }
    return Pieces(a: a, b: b, lo: lo, hi: hi)
}

/// Symmetric positive-definite banded solve, upper storage `ab[(3 - (j - i)) * V + j] = A[i, j]`
/// (kd = 3), two right-hand sides — LAPACK dpbsv (dpbtf2 + dpbtrs, reference loop order).
/// Returns false if the matrix is not positive definite.
func bandedSolve(_ ab: UnsafeMutablePointer<Double>, _ V: Int, _ rx: UnsafeMutablePointer<Double>,
                 _ ry: UnsafeMutablePointer<Double>) -> Bool {
    let kd = 3
    @inline(__always) func at(_ i: Int, _ j: Int) -> Int { (kd + i - j) * V + j }   // A[i, j], i ≤ j
    for j in 0..<V {
        var ajj = ab[at(j, j)]
        if ajj <= 0 { return false }             // dpbtf2: info > 0 only for ajj ≤ 0 (NaN proceeds)
        ajj = ajj.squareRoot()
        ab[at(j, j)] = ajj
        let kn = Swift.min(kd, V - 1 - j)
        if kn > 0 {
            let rcp = 1.0 / ajj
            for k in 1...kn { ab[at(j, j + k)] *= rcp }
            for c in 1...kn {                                // dsyr upper: A(r, c) -= x_r x_c, r ≤ c
                let xc = ab[at(j, j + c)]
                if xc != 0 {
                    let temp = -1.0 * xc
                    for r in 1...c { ab[at(j + r, j + c)] = ab[at(j + r, j + c)] + ab[at(j, j + r)] * temp }
                }
            }
        }
    }
    for rhs in [rx, ry] {
        for j in 0..<V {                                     // Uᵀ y = b
            var temp = rhs[j]
            for i in Swift.max(0, j - kd)..<j { temp = temp - ab[at(i, j)] * rhs[i] }
            temp = temp / ab[at(j, j)]
            rhs[j] = temp
        }
        for j in stride(from: V - 1, through: 0, by: -1) where rhs[j] != 0 {   // U x = y
            rhs[j] = rhs[j] / ab[at(j, j)]
            let temp = rhs[j]
            var i = j - 1
            while i >= Swift.max(0, j - kd) { rhs[i] = rhs[i] - temp * ab[at(i, j)]; i -= 1 }
        }
    }
    return true
}

struct Smoothed {
    var vf: [Int]        // frames of the variables
    var x: [V2]          // fitted positions
    var resid: [Double]  // residual/σ per observation, NaN for unsmoothed
}

/// `_solve_pieces`: robust (IRLS Huber + hard gate) smoothing of all pieces at once.
/// Banded SPD solver signature (`bandedSolve`); injectable so tests can force the
/// per-piece fallback deterministically (Python: monkeypatched solveh_banded).
typealias BandedSolver = (UnsafeMutablePointer<Double>, Int, UnsafeMutablePointer<Double>,
                          UnsafeMutablePointer<Double>) -> Bool

func solvePieces(_ t: [Double], _ fr: [Int], _ p: [V2], _ pc: Pieces, _ sigma: Double, _ diag: Double,
                 _ cfg: LinkerConfig, solve: BandedSolver = bandedSolve) -> Smoothed {
    let nObs = fr.count
    var resid = [Double](repeating: .nan, count: nObs)
    var a: [Int] = [], b: [Int] = [], lo: [Int] = [], hi: [Int] = []
    for k in 0..<pc.a.count where pc.b[k] - pc.a[k] >= 3 {        // need ≥ 3 obs per piece
        a.append(pc.a[k]); b.append(pc.b[k]); lo.append(pc.lo[k]); hi.append(pc.hi[k])
    }
    if a.isEmpty { return Smoothed(vf: [], x: [], resid: resid) }
    let np_ = a.count
    var poff = [Int](repeating: 0, count: np_ + 1)
    for k in 0..<np_ { poff[k + 1] = poff[k] + (hi[k] - lo[k] + 1) }
    let V = poff[np_]
    var vp = [Int](repeating: 0, count: V), vf = [Int](repeating: 0, count: V)
    for k in 0..<np_ { for v in poff[k]..<poff[k + 1] { vp[v] = k; vf[v] = lo[k] + (v - poff[k]) } }
    let h0 = t.count > 1 ? medianDiff(t) : 1.0 / 30.0
    var tn = [Double](repeating: 0, count: V)
    let tv0 = t[vf[0]]
    for v in 0..<V { tn[v] = (t[vf[v]] - tv0) / h0 }
    var oi: [Int] = [], ov: [Int] = []
    for k in 0..<np_ { for o in a[k]..<b[k] { oi.append(o); ov.append(poff[k] + (fr[o] - lo[k])) } }
    let nOi = oi.count
    let po = oi.map { p[$0] }
    // reference for the tiny ridge: piecewise-linear interpolation of the data
    var xref = [V2](repeating: .zero, count: V)
    for k in 0..<np_ {
        let xp = (a[k]..<b[k]).map { Double(fr[$0]) }
        let fx = (a[k]..<b[k]).map { p[$0].x }, fy = (a[k]..<b[k]).map { p[$0].y }
        xp.withUnsafeBufferPointer { xpB in
        fx.withUnsafeBufferPointer { fxB in
        fy.withUnsafeBufferPointer { fyB in
            for v in poff[k]..<poff[k + 1] {
                let xv = Double(vf[v])
                xref[v] = V2(npInterp(xv, xpB, fxB), npInterp(xv, xpB, fyB))
            }
        }}}
    }
    // third-divided-difference rows over 4 consecutive vars of one piece
    var r0: [Int] = []
    if V >= 4 { for v in 0..<(V - 3) where vp[v] == vp[v + 3] { r0.append(v) } }
    let R = r0.count
    var C = [Double](repeating: 0, count: R * 4), hbar = [Double](repeating: 0, count: R)
    for r in 0..<R {
        let T0 = tn[r0[r]], T1 = tn[r0[r] + 1], T2 = tn[r0[r] + 2], T3 = tn[r0[r] + 3]
        let T = SIMD4<Double>(T0, T1, T2, T3)
        for k in 0..<4 {
            var prod = 1.0
            for j in 0..<4 where j != k { prod *= T[k] - T[j] }
            C[r * 4 + k] = 6.0 / prod
        }
        hbar[r] = (T3 - T0) / 3.0
    }
    let Lj = cfg.jerkDragLen * diag
    let jfloor = cfg.jerkFloor * diag
    let sgc = pyMax(sigma, 1e-6)
    let invS2 = 1.0 / pow(sgc, 2.0)
    let ridge = 1e-8 * invS2
    let h05 = pow(h0, 5.0)

    /// IRLS over vars v0..<v1 (whole pieces), jerk rows ra..<rb, observations oa..<ob;
    /// nil where the banded system is not positive definite (Python: LinAlgError).
    func irls(_ v0: Int, _ v1: Int, _ ra: Int, _ rb: Int, _ oa: Int, _ ob: Int) -> ([V2], [Double])? {
        let nv = v1 - v0, no = ob - oa
        var wObs = [Double](repeating: 1, count: no), rr = [Double](repeating: 0, count: no)
        var x = Array(xref[v0..<v1])
        let ab = UnsafeMutablePointer<Double>.allocate(capacity: 4 * nv)
        let rx = UnsafeMutablePointer<Double>.allocate(capacity: nv)
        let ry = UnsafeMutablePointer<Double>.allocate(capacity: nv)
        defer { ab.deallocate(); rx.deallocate(); ry.deallocate() }
        var segV = [Double](repeating: 0, count: Swift.max(nv - 1, 0))
        var wr = [Double](repeating: 0, count: rb - ra)
        for it in 0..<Swift.max(1, cfg.irlsIters) {
            for v in 0..<Swift.max(nv - 1, 0) {                   // speed-adaptive jerk prior
                let d = x[v + 1] - x[v]
                segV[v] = hypot(d.x, d.y) / npMax(t[vf[v0 + v + 1]] - t[vf[v0 + v]], 1e-6)
            }
            for r in ra..<rb {
                let i0 = r0[r] - v0
                let vr = npMax(npMax(segV[i0], segV[i0 + 1]), segV[i0 + 2])
                let J = jfloor + pow(vr, 3.0) / (Lj * Lj)
                let qn = J * J * cfg.jerkCorrS * h05
                wr[r - ra] = hbar[r] / (0.55 * qn)
            }
            ab.initialize(repeating: 0, count: 4 * nv)
            for v in 0..<nv { ab[3 * nv + v] = 0.0 + ridge }
            for k in oa..<ob { ab[3 * nv + ov[k] - v0] += wObs[k - oa] * invS2 }
            for i in 0..<4 {
                for j in i..<4 {
                    let row = (3 - (j - i)) * nv
                    for r in ra..<rb { ab[row + r0[r] - v0 + j] += wr[r - ra] * C[r * 4 + i] * C[r * 4 + j] }
                }
            }
            for v in 0..<nv { rx[v] = ridge * xref[v0 + v].x; ry[v] = ridge * xref[v0 + v].y }
            for k in oa..<ob {
                let w = wObs[k - oa] * invS2
                rx[ov[k] - v0] += w * po[k].x; ry[ov[k] - v0] += w * po[k].y
            }
            if !solve(ab, nv, rx, ry) { return nil }
            for v in 0..<nv { x[v] = V2(rx[v], ry[v]) }
            for k in oa..<ob {
                let d = x[ov[k] - v0] - po[k]
                rr[k - oa] = hypot(d.x, d.y) / sgc
                wObs[k - oa] = rr[k - oa] <= cfg.huberK ? 1.0 : cfg.huberK / npMax(rr[k - oa], 1e-9)
                if it >= 1 && rr[k - oa] > cfg.outlierK { wObs[k - oa] = 0.0 }
            }
        }
        return (x, rr)
    }

    var x: [V2], rr: [Double]
    if let whole = irls(0, V, 0, R, 0, nOi) {
        (x, rr) = whole
    } else {
        // a piece whose jerk prior is not positive definite (near-duplicate timestamps): the
        // system is block diagonal, so solve piece by piece; failing pieces stay NaN (unsmoothed)
        x = [V2](repeating: nanV2(), count: V); rr = [Double](repeating: .nan, count: nOi)
        var ra = 0, oa = 0
        for k in 0..<np_ {
            var rb = ra
            while rb < R && vp[r0[rb]] == k { rb += 1 }
            let ob = oa + (b[k] - a[k])
            if let sub = irls(poff[k], poff[k + 1], ra, rb, oa, ob) {
                for v in 0..<sub.0.count { x[poff[k] + v] = sub.0[v] }
                for o in 0..<sub.1.count { rr[oa + o] = sub.1[o] }
            }
            ra = rb; oa = ob
        }
    }
    for k in 0..<nOi { resid[oi[k]] = rr[k] }
    return Smoothed(vf: vf, x: x, resid: resid)
}
