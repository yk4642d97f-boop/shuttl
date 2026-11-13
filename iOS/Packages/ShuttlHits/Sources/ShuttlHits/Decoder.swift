import Foundation

/// Viterbi over candidates x {near, far} with alternation / timing priors, then
/// forward-backward on the accepted chain for hitter posteriors (hits.py `_Decoder`).
struct HitDecoder {
    let c: [HitCandidate]
    let cfg: HitConfig
    let diag: Double
    let pairW: Double
    let hit: [Double]
    let side: [(near: Double, far: Double)]

    init(_ cands: [HitCandidate], _ cfg: HitConfig, _ diag: Double, _ pairW: Double) {
        c = cands; self.cfg = cfg; self.diag = diag; self.pairW = pairW
        hit = cands.map { $0.kin + $0.reach }
        let l2 = Foundation.log(2.0)
        side = cands.map { (NP.logSigmoid($0.sideLLR) + l2, NP.logSigmoid(-$0.sideLLR) + l2) }
    }

    @inline(__always) func sideLL(_ k: Int, _ s: Int) -> Double { s == 0 ? side[k].near : side[k].far }

    /// Log prior of one flight lasting d seconds (0 on the rally-tempo plateau).
    static func flightLogprior(_ d: Double, _ cfg: HitConfig) -> Double {
        if d > cfg.longFlightS { return -Foundation.pow((d - cfg.longFlightS) / cfg.longFlightSd, 2) }
        if d < cfg.shortFlightS { return -Foundation.pow((cfg.shortFlightS - d) / 0.08, 2) }
        return 0.0
    }

    /// Image-y ordering cue: the near one of two consecutive opposite-side contacts appears lower.
    func pair(_ j: Int, _ k: Int, _ sj: Int, _ sk: Int) -> Double {
        if sj == sk || pairW <= 0 { return 0.0 }
        let dy = (c[k].xy.y - c[j].xy.y) / (0.05 * diag)
        let sign = sk == 0 ? 1.0 : -1.0                // k near => expect dy > 0
        return -pairW * NP.clip(-sign * dy, 0.0, 3.0)
    }

    /// Transition log-prob between consecutive accepted contacts of one rally segment.
    func trans(_ j: Int, _ k: Int, _ sj: Int, _ sk: Int) -> Double {
        let gap = c[k].tau - c[j].tau
        if gap < cfg.minSepS { return -.infinity }
        let same = sj == sk
        if same && gap < cfg.missedHitS { return cfg.logEpsFastSame }
        var best = -Double.infinity
        var m = same ? 1 : 0
        while m < 8 {
            let v = Double(m) * cfg.logEpsSame + Double(m + 1) * HitDecoder.flightLogprior(gap / Double(m + 1), cfg)
            best = NP.pyMax(best, v)
            if gap / Double(m + 1) < cfg.longFlightS { break }
            m += 2
        }
        let pr = gap < 2 * cfg.longFlightS ? pair(j, k, sj, sk) : 0.0
        return best + pr
    }

    /// Best chain [(candidate, side 0=near/1=far)].
    func viterbi() -> [(k: Int, s: Int)] {
        let n = c.count
        if n == 0 { return [] }
        var V = [Double](repeating: -.infinity, count: 2 * n)
        var bp = [(Int, Int)?](repeating: nil, count: 2 * n)
        var prevBest = 0.0
        var prevArg: (Int, Int)? = nil
        var s0 = 0
        for k in 0..<n {
            if k > 0 && c[k].seg != c[k - 1].seg {
                for j in s0..<k {
                    for s in 0..<2 where V[2 * j + s] > prevBest { prevBest = V[2 * j + s]; prevArg = (j, s) }
                }
                s0 = k
            }
            for sk in 0..<2 {
                var best = prevBest
                var arg = prevArg
                for j in s0..<k {
                    for sj in 0..<2 {
                        let vj = V[2 * j + sj]
                        if vj == -.infinity { continue }
                        let v = vj + trans(j, k, sj, sk)
                        if v > best { best = v; arg = (j, sj) }
                    }
                }
                V[2 * k + sk] = hit[k] + sideLL(k, sk) + best
                bp[2 * k + sk] = arg
            }
        }
        // np.argmax over the flattened (n, 2) array: first maximum; a NaN wins (first NaN)
        var bi = 0
        for i in 1..<(2 * n) {
            if V[bi].isNaN { break }
            if V[i].isNaN || V[i] > V[bi] { bi = i }
        }
        if V[bi] <= 0 { return [] }
        var chain: [(k: Int, s: Int)] = []
        var cur: (Int, Int)? = (bi / 2, bi % 2)
        while let (k, s) = cur {
            chain.append((k, s))
            cur = bp[2 * k + s]
        }
        return chain.reversed()
    }

    /// P(near) for each accepted candidate (no coupling across rally breaks).
    func posteriors(_ idx: [Int]) -> [Double] {
        let m = idx.count
        if m == 0 { return [] }
        var T = [Double](repeating: 0, count: 4 * max(m - 1, 0))      // T[i][a][b]
        for i in 0..<max(m - 1, 0) where c[idx[i]].seg == c[idx[i + 1]].seg {
            for a in 0..<2 {
                for b in 0..<2 {
                    let v = trans(idx[i], idx[i + 1], a, b)
                    T[4 * i + 2 * a + b] = v.isFinite ? v : -50.0
                }
            }
        }
        var fw = [Double](repeating: 0, count: 2 * m)
        fw[0] = sideLL(idx[0], 0); fw[1] = sideLL(idx[0], 1)
        for i in 1..<max(m, 1) where i < m {
            for b in 0..<2 {
                let r = NP.logaddexp(fw[2 * (i - 1)] + T[4 * (i - 1) + b], fw[2 * (i - 1) + 1] + T[4 * (i - 1) + 2 + b])
                fw[2 * i + b] = sideLL(idx[i], b) + r
            }
        }
        var bw = [Double](repeating: 0, count: 2 * m)
        var i = m - 2
        while i >= 0 {
            for a in 0..<2 {
                let e0 = sideLL(idx[i + 1], 0) + bw[2 * (i + 1)]
                let e1 = sideLL(idx[i + 1], 1) + bw[2 * (i + 1) + 1]
                bw[2 * i + a] = NP.logaddexp(T[4 * i + 2 * a] + e0, T[4 * i + 2 * a + 1] + e1)
            }
            i -= 1
        }
        var out = [Double](repeating: 0, count: m)
        for i in 0..<m {
            let p0 = fw[2 * i] + bw[2 * i], p1 = fw[2 * i + 1] + bw[2 * i + 1]
            out[i] = Foundation.exp(p0 - NP.logaddexp(p0, p1))
        }
        return out
    }
}
