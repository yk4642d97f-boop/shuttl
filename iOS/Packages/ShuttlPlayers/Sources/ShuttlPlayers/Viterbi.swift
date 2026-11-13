import Foundation
import ShuttlCore

/// Global Viterbi per court half + RTS smoother (players.py identity / smoothing).
public enum PlayerViterbi {
    /// (len(ia), len(ib)) motion cost, row-major; inf where the implied speed is impossible.
    static func transCost(_ a: PlayerCandidates, _ ia: [Int], _ b: PlayerCandidates, _ ib: [Int], dt: Double,
                          config: PlayersConfig) -> [Double] {
        var c = [Double](repeating: 0, count: ia.count * ib.count)
        let vdt = config.vMax * dt, sdt = 3.0 * dt
        for (r, i) in ia.enumerated() {
            let pa = a.pos[i], ua = a.unit[i], na = a.noise[i]
            for (q, j) in ib.enumerated() {
                let d = NP.normAxis(pa - b.pos[j]) * (ua + b.unit[j]) / 2
                let noise = na + b.noise[j]
                let allow = vdt + 2.0 * noise
                let sig = sdt + noise
                let x = d / sig
                c[r * ib.count + q] = d > allow ? .infinity : 0.5 * (x * x)
            }
        }
        return c
    }

    /// Globally optimal single-target track through per-frame candidates
    /// (players.viterbi_path). Returns [(k, j)] in time order; empty = never present.
    public static func path(_ frames: [PlayerCandidates], allowed: [[Int]], times: [Double],
                            extra: [[Double]]?, config: PlayersConfig = .default) -> [(Int, Int)] {
        let K = frames.count
        if K == 0 { return [] }
        let cMiss = config.cMiss
        var best: [[Double]] = []
        var back: [[(Int, Int)?]] = []
        best.reserveCapacity(K); back.reserveCapacity(K)
        var ends = [Double](repeating: .infinity, count: K)
        var prefMin = Double.infinity
        var prefArg: Int? = nil
        var p = 0
        for k in 0..<K {
            let idx = allowed[k]
            let n = idx.count
            var cst = [Double](repeating: .infinity, count: n)
            var bk = [(Int, Int)?](repeating: nil, count: n)
            if n > 0 {
                var un = [Double](repeating: 0, count: n)
                for (q, j) in idx.enumerated() {
                    un[q] = frames[k].unary[j] + (extra != nil ? extra![k][j] : 0.0)
                }
                let base = cMiss * Double(k)
                for q in 0..<n { cst[q] = base + un[q] }                     // start here
                while p < k && times[k] - times[p] > config.lookbackS {
                    let v = ends[p] - cMiss * Double(p)
                    if v < prefMin { prefMin = v; prefArg = p }
                    p += 1
                }
                if let pa = prefArg {                                          // re-acquire after a long gap
                    let rc0 = prefMin + cMiss * Double(k - 1) + config.cRestart
                    let jb = argmin(best[pa])
                    for q in 0..<n {
                        let rc = rc0 + un[q]
                        if rc < cst[q] { cst[q] = rc; bk[q] = (pa, jb) }
                    }
                }
                let unMin = un.min()!
                var kp = k - 1
                while kp >= p {                                                // motion-gated bridges
                    defer { kp -= 1 }
                    let ia = allowed[kp]
                    if ia.isEmpty || !ends[kp].isFinite { continue }
                    let gapCost = cMiss * Double(k - kp - 1)
                    if ends[kp] + gapCost + unMin >= npMax(cst) { continue }
                    let tc = transCost(frames[kp], ia, frames[k], idx, dt: times[k] - times[kp], config: config)
                    let bkp = best[kp]
                    for q in 0..<n {
                        var bestVal = Double.infinity, arg = 0                // np.argmin(tot, axis=0)
                        for r in 0..<ia.count {
                            let tot = bkp[r] + tc[r * n + q] + gapCost
                            if r == 0 || tot < bestVal { bestVal = tot; arg = r }
                        }
                        let val = bestVal + un[q]
                        if val < cst[q] { cst[q] = val; bk[q] = (kp, arg) }
                    }
                }
                ends[k] = cst.min()!
            }
            best.append(cst)
            back.append(bk)
        }
        var kBest = 0
        var fBest = Double.infinity
        for k in 0..<K {
            let f = ends[k] + cMiss * Double(K - 1 - k)
            if k == 0 || f < fBest { fBest = f; kBest = k }
        }
        if !fBest.isFinite || fBest >= cMiss * Double(K) { return [] }
        var out: [(Int, Int)] = []
        var node: (Int, Int)? = (kBest, argmin(best[kBest]))
        while let (kk, jj) = node {
            out.append((kk, allowed[kk][jj]))
            node = back[kk][jj]
        }
        return out.reversed()
    }

    /// Rauch–Tung–Striebel smoother, constant-velocity model per axis (players.rts_smooth).
    /// `z` rows with NaN x = no measurement; `r` measurement std (m).
    public static func rtsSmooth(t: [Double], z: [SIMD2<Double>], r: [Double],
                                 accSigma: Double = PlayersConfig.default.accSigma) -> [SIMD2<Double>] {
        let n = z.count
        guard n > 0 else { return [] }
        let q = NP.pyPow(accSigma, 2)
        // state per axis: rows [pos, vel] × columns [x, y]
        typealias M = (a: Double, b: Double, c: Double, d: Double)          // [[a, b], [c, d]]
        typealias X = (p: SIMD2<Double>, v: SIMD2<Double>)
        var xs = [X](repeating: (.zero, .zero), count: n)
        var Ps = [M](repeating: (0, 0, 0, 0), count: n)
        var xp = [X](repeating: (.zero, .zero), count: n)
        var Pp = [M](repeating: (0, 0, 0, 0), count: n)
        let first = z.firstIndex { $0.x.isFinite } ?? 0
        var x: X = (z[first], .zero)
        var P: M = (NP.pyPow(r[first], 2) + 1e-4, 0, 0, 4.0)
        for i in 0..<n {
            let dt = i > 0 ? t[i] - t[i - 1] : 0.0
            // x = F @ x  (F = [[1, dt], [0, 1]]), BLAS 2-term dots
            x = (SIMD2(NP.fma2(1, x.p.x, dt, x.v.x), NP.fma2(1, x.p.y, dt, x.v.y)),
                 SIMD2(NP.fma2(0, x.p.x, 1, x.v.x), NP.fma2(0, x.p.y, 1, x.v.y)))
            // P = F @ P @ F.T + Q
            let FP = matmul((1, dt, 0, 1), P)
            let FPFt = matmul(FP, (1, 0, dt, 1))
            let dt3 = NP.pyPow(dt, 3) / 3, dt2 = NP.pyPow(dt, 2) / 2
            let Q: M = (q * dt3, q * dt2, q * dt2, q * dt)
            P = (FPFt.a + Q.a, FPFt.b + Q.b, FPFt.c + Q.c, FPFt.d + Q.d)
            xp[i] = x; Pp[i] = P
            if z[i].x.isFinite {
                let S = P.a + NP.pyPow(r[i], 2)
                let k0 = P.a / S, k1 = P.c / S                       // Kg = P[:, 0] / S
                let inn = z[i] - x.p
                x = (x.p + k0 * inn, x.v + k1 * inn)
                P = (P.a - k0 * P.a, P.b - k0 * P.b, P.c - k1 * P.a, P.d - k1 * P.b)
            }
            xs[i] = x; Ps[i] = P
        }
        var out = xs
        if n >= 2 {
            for i in stride(from: n - 2, through: 0, by: -1) {
                let dt = t[i + 1] - t[i]
                let pp = Pp[i + 1]
                let det = pp.a * pp.d - pp.b * pp.c
                let inv: M = (pp.d / det, -pp.b / det, -pp.c / det, pp.a / det)
                let C = matmul(matmul(Ps[i], (1, 0, dt, 1)), inv)
                let dp = out[i + 1].p - xp[i + 1].p, dv = out[i + 1].v - xp[i + 1].v
                let cp = SIMD2(NP.fma2(C.a, dp.x, C.b, dv.x), NP.fma2(C.a, dp.y, C.b, dv.y))
                let cv = SIMD2(NP.fma2(C.c, dp.x, C.d, dv.x), NP.fma2(C.c, dp.y, C.d, dv.y))
                out[i] = (xs[i].p + cp, xs[i].v + cv)
            }
        }
        return out.map(\.p)
    }

    /// 2×2 BLAS matmul (row-major tuples).
    @inline(__always)
    static func matmul(_ A: (a: Double, b: Double, c: Double, d: Double),
                       _ B: (a: Double, b: Double, c: Double, d: Double)) -> (a: Double, b: Double, c: Double, d: Double) {
        (NP.fma2(A.a, B.a, A.b, B.c), NP.fma2(A.a, B.b, A.b, B.d),
         NP.fma2(A.c, B.a, A.d, B.c), NP.fma2(A.c, B.b, A.d, B.d))
    }
}

/// `np.argmin` (first minimum; first NaN wins like numpy).
func argmin(_ a: [Double]) -> Int {
    var best = 0
    for i in 1..<max(a.count, 1) {
        if a[best].isNaN { return best }
        if a[i].isNaN || a[i] < a[best] { best = i }
    }
    return best
}

/// `ndarray.max()` (NaN propagates).
func npMax(_ a: [Double]) -> Double {
    var m = -Double.infinity
    for x in a {
        if x.isNaN { return .nan }
        if x > m { m = x }
    }
    return m
}
