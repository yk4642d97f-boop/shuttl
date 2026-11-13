import Foundation

// Change-point scan — hits.py `_windows`, `_fit_batch`, `_lambda`, `_scan`,
// `_local_maxima`, `_nms`, `_refine`. One tau at a time with a shared 6-column
// Gram matrix (columns 1, s, s², s³, sp, sp²) instead of numpy's padded batch;
// every design's normal equations are sub-blocks of it (same products, same sums).

struct FitOut {
    var rss0 = 0.0, rssv = 0.0, rss1 = 0.0, rss0c = 0.0, rssvc = 0.0
    var p = SIMD2<Double>(0, 0), vIn = SIMD2<Double>(0, 0), vOut = SIMD2<Double>(0, 0)
    var nLeft = 0, nRight = 0
}

struct ScanResult {
    var lam: [Double]
    var lamV: [Double]
    var ok: [Bool]
    var lo: [Int]
    var hi: [Int]
    var m: [Int]
}

struct RefineResult {
    var tau: Double
    var m: Int
    var lam: Double
    var lamV: Double
    var p: SIMD2<Double>
    var vIn: SIMD2<Double>
    var vOut: SIMD2<Double>
}

/// Least-squares fits continuous at tau over observations [lo, hi) split at m.
final class Fitter {
    private let t: UnsafeMutablePointer<Double>
    private let x: UnsafeMutablePointer<Double>
    private let y: UnsafeMutablePointer<Double>
    let n: Int
    private let work: UnsafeMutablePointer<Double>   // A (25) | B (10) | betas (5 x 10) | G (36) | R (12)

    init(t: [Double], xy: [SIMD2<Double>]) {
        n = t.count
        self.t = .allocate(capacity: max(n, 1))
        x = .allocate(capacity: max(n, 1))
        y = .allocate(capacity: max(n, 1))
        for i in 0..<n { self.t[i] = t[i]; x[i] = xy[i].x; y[i] = xy[i].y }
        work = .allocate(capacity: 25 + 10 + 50 + 36 + 12)
    }

    deinit { t.deallocate(); x.deallocate(); y.deallocate(); work.deallocate() }

    var times: UnsafePointer<Double> { UnsafePointer(t) }

    // design → basis columns (0:1, 1:s, 2:s2, 3:s3, 4:sp, 5:sp2), numpy order "0","v","1","0c","vc"
    private static let designs: [[Int]] = [[0, 1, 2], [0, 1, 2, 4], [0, 1, 2, 4, 5], [0, 1, 2, 3], [0, 1, 2, 3, 4]]

    func fit(tau: Double, lo: Int, hi: Int, m: Int, scale: Double) -> FitOut {
        var out = FitOut()
        var g = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        var rx = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), ry = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        var nL = 0, nR = 0
        var j = lo
        while j < hi {
            let s = (t[j] - tau) / scale
            let right = j >= m
            if right { nR += 1 } else { nL += 1 }
            let s2 = s * s, s3 = s * s * s
            let sp = right ? s : 0.0
            let sp2 = sp * sp
            let yx = x[j], yy = y[j]
            g.0 += 1.0; g.1 += s; g.2 += s2; g.3 += s3; g.4 += sp; g.5 += sp2
            g.6 += s * s; g.7 += s * s2; g.8 += s * s3; g.9 += s * sp; g.10 += s * sp2
            g.11 += s2 * s2; g.12 += s2 * s3; g.13 += s2 * sp; g.14 += s2 * sp2
            g.15 += s3 * s3; g.16 += s3 * sp; g.17 += s3 * sp2
            g.18 += sp * sp; g.19 += sp * sp2; g.20 += sp2 * sp2
            rx.0 += yx; rx.1 += s * yx; rx.2 += s2 * yx; rx.3 += s3 * yx; rx.4 += sp * yx; rx.5 += sp2 * yx
            ry.0 += yy; ry.1 += s * yy; ry.2 += s2 * yy; ry.3 += s3 * yy; ry.4 += sp * yy; ry.5 += sp2 * yy
            j += 1
        }
        out.nLeft = nL; out.nRight = nR
        let G = work + 85, R = work + 121          // G 6x6 row-major, R 6x2
        G[0] = g.0; G[1] = g.1; G[2] = g.2; G[3] = g.3; G[4] = g.4; G[5] = g.5
        G[6] = g.1; G[7] = g.6; G[8] = g.7; G[9] = g.8; G[10] = g.9; G[11] = g.10
        G[12] = g.2; G[13] = g.7; G[14] = g.11; G[15] = g.12; G[16] = g.13; G[17] = g.14
        G[18] = g.3; G[19] = g.8; G[20] = g.12; G[21] = g.15; G[22] = g.16; G[23] = g.17
        G[24] = g.4; G[25] = g.9; G[26] = g.13; G[27] = g.16; G[28] = g.18; G[29] = g.19
        G[30] = g.5; G[31] = g.10; G[32] = g.14; G[33] = g.17; G[34] = g.19; G[35] = g.20
        R[0] = rx.0; R[1] = ry.0; R[2] = rx.1; R[3] = ry.1; R[4] = rx.2; R[5] = ry.2
        R[6] = rx.3; R[7] = ry.3; R[8] = rx.4; R[9] = ry.4; R[10] = rx.5; R[11] = ry.5

        let A = work, B = work + 25, betas = work + 35
        for (di, cols) in Fitter.designs.enumerated() {
            let p = cols.count
            for a in 0..<p {
                for b in 0..<p { A[a * p + b] = G[cols[a] * 6 + cols[b]] }
                A[a * p + a] += 1e-9
                B[a * 2] = R[cols[a] * 2]; B[a * 2 + 1] = R[cols[a] * 2 + 1]
            }
            Fitter.luSolve(A, B, p)
            let beta = betas + di * 10
            for k in 0..<(2 * p) { beta[k] = B[k] }
        }
        // residuals of all five designs in one pass
        var rss = (0.0, 0.0, 0.0, 0.0, 0.0)
        let b0 = betas, bv = betas + 10, b1 = betas + 20, b0c = betas + 30, bvc = betas + 40
        j = lo
        while j < hi {
            let s = (t[j] - tau) / scale
            let right = j >= m
            let s2 = s * s, s3 = s * s * s
            let sp = right ? s : 0.0
            let sp2 = sp * sp
            let yx = x[j], yy = y[j]
            // unrolled per design (columns in design order)
            do {
                var fx = 1.0 * b0[0], fy = 1.0 * b0[1]
                fx += s * b0[2]; fy += s * b0[3]; fx += s2 * b0[4]; fy += s2 * b0[5]
                let ex = yx - fx, ey = yy - fy
                rss.0 += ex * ex; rss.0 += ey * ey
            }
            do {
                var fx = 1.0 * bv[0], fy = 1.0 * bv[1]
                fx += s * bv[2]; fy += s * bv[3]; fx += s2 * bv[4]; fy += s2 * bv[5]; fx += sp * bv[6]; fy += sp * bv[7]
                let ex = yx - fx, ey = yy - fy
                rss.1 += ex * ex; rss.1 += ey * ey
            }
            do {
                var fx = 1.0 * b1[0], fy = 1.0 * b1[1]
                fx += s * b1[2]; fy += s * b1[3]; fx += s2 * b1[4]; fy += s2 * b1[5]
                fx += sp * b1[6]; fy += sp * b1[7]; fx += sp2 * b1[8]; fy += sp2 * b1[9]
                let ex = yx - fx, ey = yy - fy
                rss.2 += ex * ex; rss.2 += ey * ey
            }
            do {
                var fx = 1.0 * b0c[0], fy = 1.0 * b0c[1]
                fx += s * b0c[2]; fy += s * b0c[3]; fx += s2 * b0c[4]; fy += s2 * b0c[5]; fx += s3 * b0c[6]; fy += s3 * b0c[7]
                let ex = yx - fx, ey = yy - fy
                rss.3 += ex * ex; rss.3 += ey * ey
            }
            do {
                var fx = 1.0 * bvc[0], fy = 1.0 * bvc[1]
                fx += s * bvc[2]; fy += s * bvc[3]; fx += s2 * bvc[4]; fy += s2 * bvc[5]
                fx += s3 * bvc[6]; fy += s3 * bvc[7]; fx += sp * bvc[8]; fy += sp * bvc[9]
                let ex = yx - fx, ey = yy - fy
                rss.4 += ex * ex; rss.4 += ey * ey
            }
            j += 1
        }
        out.rss0 = rss.0; out.rssv = rss.1; out.rss1 = rss.2; out.rss0c = rss.3; out.rssvc = rss.4
        out.p = SIMD2(bvc[0], bvc[1])
        out.vIn = SIMD2(bvc[2] / scale, bvc[3] / scale)
        out.vOut = SIMD2((bvc[2] + bvc[8]) / scale, (bvc[3] + bvc[9]) / scale)
        return out
    }

    /// LU with partial pivoting (LAPACK gesv algorithm), p x p system, 2 right-hand sides in place.
    static func luSolve(_ a: UnsafeMutablePointer<Double>, _ b: UnsafeMutablePointer<Double>, _ p: Int) {
        for k in 0..<p {
            var piv = k
            var best = abs(a[k * p + k])
            var r = k + 1
            while r < p { let v = abs(a[r * p + k]); if v > best { best = v; piv = r }; r += 1 }
            if piv != k {
                for c in 0..<p { let tmp = a[k * p + c]; a[k * p + c] = a[piv * p + c]; a[piv * p + c] = tmp }
                for c in 0..<2 { let tmp = b[k * 2 + c]; b[k * 2 + c] = b[piv * 2 + c]; b[piv * 2 + c] = tmp }
            }
            let inv = 1.0 / a[k * p + k]
            r = k + 1
            while r < p {
                let f = a[r * p + k] * inv
                a[r * p + k] = f
                var c = k + 1
                while c < p { a[r * p + c] -= f * a[k * p + c]; c += 1 }
                r += 1
            }
        }
        for c in 0..<2 {
            for r in 0..<p {                    // L (unit lower)
                var s = b[r * 2 + c]
                for q in 0..<r { s -= a[r * p + q] * b[q * 2 + c] }
                b[r * 2 + c] = s
            }
            var r = p - 1
            while r >= 0 {                      // U
                var s = b[r * 2 + c]
                var q = r + 1
                while q < p { s -= a[r * p + q] * b[q * 2 + c]; q += 1 }
                b[r * 2 + c] = s / a[r * p + r]
                r -= 1
            }
        }
    }
}

extension Hits {
    /// Observation window [lo, hi) and split m for one tau.
    static func window(_ t: UnsafePointer<Double>, _ n: Int, _ tau: Double, _ cfg: HitConfig) -> (lo: Int, hi: Int, m: Int) {
        let m = NP.searchsorted(t, n, tau)
        let tL = m > 0 ? t[min(max(m - 1, 0), n - 1)] : tau
        let tR = m < n ? t[min(max(m, 0), n - 1)] : tau
        let loW = NP.searchsorted(t, n, Swift.min(tL, tau) - cfg.windowS)
        let hiW = NP.searchsorted(t, n, Swift.max(tR, tau) + cfg.windowS, right: true)
        let loX = NP.searchsorted(t, n, tau - cfg.maxWindowS)
        let hiX = NP.searchsorted(t, n, tau + cfg.maxWindowS, right: true)
        let want = Swift.max(cfg.wantSide, cfg.minSide)
        var lo = Swift.max(Swift.min(loW, Swift.max(m - want, loX)), loX)
        var hi = Swift.min(Swift.max(hiW, Swift.min(m + want, hiX)), hiX)
        lo = Swift.max(lo, m - cfg.maxSide)
        hi = Swift.min(hi, m + cfg.maxSide)
        return (Swift.min(Swift.max(lo, 0), n), Swift.min(Swift.max(hi, 0), n), m)
    }

    /// (lam, lam_v, s2) — likelihood-ratio statistics with a misfit-aware noise variance.
    @inline(__always) static func lambda(_ f: FitOut, _ sigma: Double) -> (lam: Double, lamV: Double) {
        let nObs = 2 * (f.nLeft + f.nRight)
        let dof1 = Double(Swift.max(nObs - 10, 1))
        let mn = f.rss1.isNaN || f.rssvc.isNaN ? Double.nan : Swift.min(f.rss1, f.rssvc)
        let s2 = mn.isNaN ? Double.nan : Swift.max(Foundation.pow(sigma, 2), mn / dof1)
        return ((f.rss0 - f.rss1) / s2, (f.rss0c - f.rssvc) / s2)
    }

    static func scan(_ fitter: Fitter, taus: [Double], cfg: HitConfig, sigma: Double) -> ScanResult {
        let T = taus.count, n = fitter.n
        var r = ScanResult(lam: .init(repeating: 0, count: T), lamV: .init(repeating: 0, count: T),
                           ok: .init(repeating: false, count: T), lo: .init(repeating: 0, count: T),
                           hi: .init(repeating: 0, count: T), m: .init(repeating: 0, count: T))
        let t = fitter.times
        for g in 0..<T {
            let w = window(t, n, taus[g], cfg)
            r.lo[g] = w.lo; r.hi[g] = w.hi; r.m[g] = w.m
            guard w.m - w.lo >= cfg.minSide && w.hi - w.m >= cfg.minSide else { continue }
            let f = fitter.fit(tau: taus[g], lo: w.lo, hi: w.hi, m: w.m, scale: cfg.windowS)
            let l = lambda(f, sigma)
            r.lam[g] = l.lam; r.lamV[g] = l.lamV; r.ok[g] = true
        }
        return r
    }

    static func localMaxima(_ lam: [Double], _ ok: [Bool], _ thr: Double) -> [Int] {
        let n = lam.count
        var out: [Int] = []
        for i in 0..<n {
            let x = ok[i] ? lam[i] : -Double.infinity
            let left = i > 0 ? (ok[i - 1] ? lam[i - 1] : -Double.infinity) : -Double.infinity
            let right = i + 1 < n ? (ok[i + 1] ? lam[i + 1] : -Double.infinity) : -Double.infinity
            if x >= left && x > right && x >= thr { out.append(i) }
        }
        return out
    }

    /// Greedy NMS by score (stable), result in time order (stable).
    static func nms(times: [Double], scores: [Double], radius: Double) -> [Int] {
        var keep: [Int] = []
        for i in NP.argsortStable(scores.map { -$0 }) where keep.allSatisfy({ abs(times[i] - times[$0]) >= radius }) {
            keep.append(i)
        }
        return keep.enumerated().sorted { a, b in
            times[a.element] < times[b.element] || (times[a.element] == times[b.element] && a.offset < b.offset)
        }.map(\.element)
    }

    /// ML change time on a sub-frame grid with the window fixed.
    static func refine(_ fitter: Fitter, tau: Double, lo: Int, hi: Int, dt: Double, cfg: HitConfig,
                       sigma: Double) -> RefineResult? {
        let n = fitter.n
        let t = fitter.times
        let m0 = NP.searchsorted(t, n, tau)
        let a = NP.pyMin(tau - dt, m0 >= 2 ? t[m0 - 2] : tau - dt)
        let b = NP.pyMax(tau + dt, m0 + 1 < n ? t[m0 + 1] : tau + dt)
        let steps = Swift.max(cfg.refineSteps, Int(((b - a) / dt * 4).rounded(.up)) + 1)
        var best: (key: Double, tau: Double, m: Int, f: FitOut)? = nil
        for tauK in NP.linspace(a, b, steps) {
            let m = NP.searchsorted(t, n, tauK)
            guard m - lo >= cfg.minSide && hi - m >= cfg.minSide else { continue }
            let f = fitter.fit(tau: tauK, lo: lo, hi: hi, m: m, scale: cfg.windowS)
            let key: Double
            switch cfg.refineModel {
            case "1": key = f.rss1
            case "v": key = f.rssv
            default: key = f.rssvc
            }
            // np.argmin: first minimum; a NaN is the minimum (first NaN wins)
            if let cur = best {
                if cur.key.isNaN { continue }
                if key.isNaN || key < cur.key { best = (key, tauK, m, f) }
            } else {
                best = (key, tauK, m, f)
            }
        }
        guard let bst = best else { return nil }
        let l = lambda(bst.f, sigma)
        return RefineResult(tau: bst.tau, m: bst.m, lam: l.lam, lamV: l.lamV, p: bst.f.p, vIn: bst.f.vIn, vOut: bst.f.vOut)
    }
}
