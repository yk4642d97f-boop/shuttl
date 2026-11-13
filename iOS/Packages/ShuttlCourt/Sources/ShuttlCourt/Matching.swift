/// Model-line ↔ image-ridge matching, fit statistics, evidence gate and dense refinement.
/// Port of court_calibration._match_lines, _fit_stats, _behind_net, _refine, _valid,
/// _confidence.
import Foundation
import simd
import ShuttlCore

/// `_fit_stats` result.
public struct FitStats: Sendable, Equatable {
    public var coverage: Double = 0
    public var rmse: Double = .infinity
    public var nT: Int = 0
    public var nL: Int = 0
    public var nVis: Int = 0
    public var nInl: Int = 0
    public var nAbsent: Int = 0
    public var absentFrac: Double = 0
    /// nil when no model sample was visible (Python omits the keys).
    public var explained: Double?
    public var explainedFrac: Double?
    /// `t_straddle`: matched transverse lines on both sides of the net.
    public var tStraddle: Bool = false
}

enum K1Mode: Sendable { case off, auto, on }

struct MatchResult {
    var idx: [Int] = []
    var found: [Bool] = []
    var target: [SIMD2<Double>] = []
    var nrm: [SIMD2<Double>] = []
    var wexp: [Double] = []
}

struct Matcher {
    let feat: Features
    let config: CourtCalibrationConfig

    /// `_match_lines`
    func matchLines(_ p: [Double], cx: Double, cy: Double, _ samples: LineSamples, search: Double) -> MatchResult {
        let pr = ParamProjector(p, cx: cx, cy: cy)
        let N = samples.P.count
        let halfW = 0.5 * CourtModel.lineWidth
        var uvAll = [SIMD2<Double>](repeating: .zero, count: N)
        var nrmAll = [SIMD2<Double>](repeating: .zero, count: N)
        var wexpAll = [Double](repeating: 0, count: N)
        var vis: [Int] = []
        let m = 2.0
        let wlim = Double(feat.w - 1) - m, hlim = Double(feat.h - 1) - m
        for i in 0..<N {
            let P = samples.P[i], Dir = samples.dir[i]
            let (uv, z) = pr.project(P)
            let uv2 = pr.project(P + 0.05 * Dir).uv
            let Nw = SIMD3(-Dir.y, Dir.x, 0)
            let uw1 = pr.project(P + halfW * Nw).uv
            let uw0 = pr.project(P - halfW * Nw).uv
            var tan = uv2 - uv
            tan /= (tan.x * tan.x + tan.y * tan.y).squareRoot() + 1e-12
            let nrm = SIMD2(-tan.y, tan.x)
            let dw = uw1 - uw0
            uvAll[i] = uv; nrmAll[i] = nrm
            wexpAll[i] = abs(dw.x * nrm.x + dw.y * nrm.y)
            if z > 0.1 && uv.x.isFinite && uv.y.isFinite && uv.x >= m && uv.y >= m && uv.x <= wlim && uv.y <= hlim {
                vis.append(i)
            }
        }
        var res = MatchResult()
        res.idx = vis
        let n = vis.count
        guard n > 0 else { return res }
        var maxW = -Double.infinity
        for i in vis { maxW = max(maxW, wexpAll[i]) }
        let S = Int((search + min(maxW, 0.02 * feat.D) + 4).rounded(.up))
        let L = 2 * S + 1
        let offs = (0..<L).map { Float($0 - S) }
        let thr = Float(0.5) * config.minContrast
        let minC = config.minContrast
        res.found = [Bool](repeating: false, count: n)
        res.target = [SIMD2<Double>](repeating: .zero, count: n)
        res.nrm = vis.map { nrmAll[$0] }
        res.wexp = vis.map { wexpAll[$0] }
        var prof = [Float](repeating: 0, count: L)
        // per-run accumulators (a row has at most L/2+1 runs)
        var rStrong = [Double](), rPk = [Double](), rFw = [Double](), rWs = [Double](), rOs = [Double](), rTouch = [Double]()
        var runId = [Int](repeating: 0, count: L)
        feat.strength.data.withUnsafeBufferPointer { s in
            for (row, i) in vis.enumerated() {
                let uv = uvAll[i], nr = nrmAll[i]
                for k in 0..<L {
                    let o = Double(offs[k])
                    let X = Float(uv.x + o * nr.x), Y = Float(uv.y + o * nr.y)
                    prof[k] = ImageOps.remapSample(s, w: feat.w, h: feat.h, X, Y)
                }
                // runs of prof > thr (1-based ids)
                var nRuns = 0
                var prevAbove = false
                for k in 0..<L {
                    let above = prof[k] > thr
                    if above && !prevAbove { nRuns += 1 }
                    runId[k] = above ? nRuns : 0
                    prevAbove = above
                }
                if nRuns == 0 { continue }
                rStrong = [Double](repeating: 0, count: nRuns + 1); rPk = rStrong; rFw = rStrong
                rWs = rStrong; rOs = rStrong; rTouch = rStrong
                for k in 0..<L {
                    let r = runId[k]
                    if prof[k] >= minC { rStrong[r] += 1 }
                    if r > 0 { rPk[r] = max(rPk[r], Double(prof[k])) }
                }
                if runId[0] > 0 { rTouch[runId[0]] += 1 }
                if runId[L - 1] > 0 { rTouch[runId[L - 1]] += 1 }
                for k in 0..<L {
                    let r = runId[k]
                    guard r > 0 else { continue }
                    let half = 0.5 * rPk[r]
                    let pv = Double(prof[k])
                    if pv >= half {
                        rFw[r] += 1
                        let wt = pv - half + 1e-3
                        rWs[r] += wt
                        rOs[r] += wt * Double(offs[k])
                    }
                }
                var best = Double.infinity, bestCen = 0.0
                let wlimit = 2 * wexpAll[i] + 4
                for r in 1...nRuns {
                    let cen = rOs[r] / rWs[r]
                    let ok = rFw[r] > 0 && rStrong[r] > 0 && rTouch[r] == 0 && rFw[r] <= wlimit && abs(cen) <= search
                    if ok && abs(cen) < best { best = abs(cen); bestCen = cen }
                }
                if best.isFinite {
                    res.found[row] = true
                    res.target[row] = uv + bestCen * nr
                } else {
                    res.target[row] = uv
                }
            }
        }
        return res
    }

    /// `_behind_net`
    func behindNet(_ pr: ParamProjector, _ Pw: [SIMD3<Double>], _ uv: [SIMD2<Double>]) -> [Bool] {
        let C = -(pr.R.transpose * pr.t)
        let W = CourtModel.doublesHalfWidth + 0.15
        let q = [SIMD3(0, -W, 0.65), SIMD3(0, W, 0.65), SIMD3(0, W, 1.7), SIMD3(0, -W, 1.7)].map { pr.project($0).uv }
        @inline(__always) func sgn(_ v: Double) -> Double { v > 0 ? 1 : v < 0 ? -1 : 0 }
        return (0..<Pw.count).map { k in
            guard sgn(Pw[k].x) * sgn(C.x) < 0 else { return false }
            var allPos = true, allNeg = true
            for i in 0..<4 {
                let a = q[i], b = q[(i + 1) % 4]
                let cr = (b.x - a.x) * (uv[k].y - a.y) - (b.y - a.y) * (uv[k].x - a.x)
                if !(cr > 0) { allPos = false }
                if !(cr < 0) { allNeg = false }
            }
            return allPos || allNeg
        }
    }

    /// `_fit_stats`
    func fitStats(_ p: [Double], cx: Double, cy: Double, _ samples: LineSamples, search: Double) -> FitStats {
        let mr = matchLines(p, cx: cx, cy: cy, samples, search: search)
        let n = mr.idx.count
        guard n > 0 else { return FitStats() }
        let pr = ParamProjector(p, cx: cx, cy: cy)
        let Pv = mr.idx.map { samples.P[$0] }
        let uv = Pv.map { pr.project($0).uv }
        var r = [Double](repeating: 0, count: n)
        var inl = [Bool](repeating: false, count: n)
        let inTol = max(1.5, 0.0015 * feat.D)
        for i in 0..<n {
            let d = uv[i] - mr.target[i]
            r[i] = d.x * mr.nrm[i].x + d.y * mr.nrm[i].y
            inl[i] = mr.found[i] && abs(r[i]) < inTol
        }
        let occ = behindNet(pr, Pv, uv)
        var xs = Set<Int>(), ys = Set<Int>()
        var nAbsent = 0
        var absentLen = 0.0, visLen = 0.0, explained = 0.0
        let segCap = 0.05 * feat.D
        let lid = mr.idx.map { samples.ids[$0] }
        for k in 0..<CourtModel.lines.count {
            let sel = (0..<n).filter { lid[$0] == k }
            let nv = sel.count
            if nv < 2 { continue }
            let frac = Double(sel.filter { inl[$0] }.count) / Double(nv)
            if nv >= 5 && frac >= 0.35 {
                let a = CourtModel.lines[k].a
                let key = Int(((isTransverse[k] ? a.x : a.y) * 100).rounded())
                if isTransverse[k] { xs.insert(key) } else { ys.insert(key) }
            }
            var seg = [Double](repeating: 0, count: nv - 1)
            var Lk = 0.0
            for j in 0..<(nv - 1) {
                seg[j] = min(simd_distance(uv[sel[j + 1]], uv[sel[j]]), segCap)
                Lk += seg[j]
            }
            visLen += Lk
            var e = 0.0
            for j in 0..<nv {
                var stp = 0.0
                if j < nv - 1 { stp += 0.5 * seg[j] }
                if j > 0 { stp += 0.5 * seg[j - 1] }
                let rk = mr.found[sel[j]] ? r[sel[j]] : Double.infinity
                let z = rk / config.explainSigma
                e += stp * exp(-0.5 * (z * z))
            }
            explained += e
            let clear = sel.filter { !occ[$0] }
            if clear.count < 2 { continue }
            var Lc = 0.0
            for j in 0..<(clear.count - 1) { Lc += min(simd_distance(uv[clear[j + 1]], uv[clear[j]]), segCap) }
            let medW = Num.median(clear.map { mr.wexp[$0] })
            let foundFrac = Double(clear.filter { mr.found[$0] }.count) / Double(clear.count)
            if Lc >= config.absentMinLength * feat.D && medW >= config.absentMinWidth && foundFrac < config.absentFound {
                nAbsent += 1
                absentLen += Lc
            }
        }
        var st = FitStats()
        let nInl = inl.filter { $0 }.count
        st.coverage = Double(nInl) / Double(n)
        if nInl > 0 {
            var s2 = 0.0
            for i in 0..<n where inl[i] { s2 += r[i] * r[i] }
            st.rmse = (s2 / Double(nInl)).squareRoot()
        }
        st.nT = xs.count; st.nL = ys.count; st.nVis = n; st.nInl = nInl
        st.tStraddle = xs.contains { $0 < 0 } && xs.contains { $0 > 0 }
        st.nAbsent = nAbsent
        st.absentFrac = absentLen / max(visLen, 1e-9)
        st.explained = explained
        st.explainedFrac = explained / max(visLen, 1e-9)
        return st
    }

    /// `_valid`: hard evidence gate for an automatic camera. >= 3 transverse and >= 3
    /// longitudinal lines, or — low / close tripods — exactly 2 transverse lines on
    /// opposite sides of the net with all 5 longitudinal lines and coverage >=
    /// `nT2MinCoverage`; never an absent line; explained >= `minExplained`.
    func valid(_ s: FitStats) -> Bool {
        guard s.nAbsent == 0, (s.explainedFrac ?? 1.0) >= config.minExplained, s.rmse.isFinite else { return false }
        if s.nT >= 3 && s.nL >= 3 { return true }
        return s.nT == 2 && s.nL == 5 && s.tStraddle && s.coverage >= config.nT2MinCoverage
    }

    /// `_confidence`
    func confidence(_ s: FitStats, margin: Double) -> Double {
        guard valid(s) else { return 0 }
        var c = min(1.0, (s.explainedFrac ?? 1.0) / 0.75) * exp(-s.rmse / (0.004 * feat.D))
        c *= pow(min(1.0, margin / 0.3), 0.5)
        c *= pow(min(1.0, Double(s.nT + s.nL) / 6.0), 0.5)
        return min(max(c, 0), 1)
    }

    /// `_refine`: ICP-style refinement — normal search → robust LM (soft-L1) → shrink window.
    func refine(_ cam: CameraModel, search0: Double, fitK1: K1Mode = .auto, nIter: Int = 4,
                samples: LineSamples = .fine) -> (CameraModel, FitStats)? {
        let cx = mget(cam.K, 0, 2), cy = mget(cam.K, 1, 2)
        let logf0 = log(defaultFocal(cam.imageSize))

        func solve(_ p0: [Double], _ search0: Double, _ iters: Int) -> ([Double], Double) {
            var p = p0, search = search0
            for _ in 0..<iters {
                let mr = matchLines(p, cx: cx, cy: cy, samples, search: search)
                var Ps: [SIMD3<Double>] = [], tg: [SIMD2<Double>] = [], nn: [SIMD2<Double>] = []
                for (row, i) in mr.idx.enumerated() where mr.found[row] {
                    Ps.append(samples.P[i]); tg.append(mr.target[row]); nn.append(mr.nrm[row])
                }
                if Ps.count < 20 { return (p, .infinity) }
                let res = { (q: [Double]) -> [Double] in
                    let pr = ParamProjector(q, cx: cx, cy: cy)
                    var out = [Double](repeating: 0, count: Ps.count + 1)
                    for j in 0..<Ps.count {
                        let d = pr.project(Ps[j]).uv - tg[j]
                        out[j] = d.x * nn[j].x + d.y * nn[j].y
                    }
                    out[Ps.count] = 0.05 * (q[0] - logf0)
                    return out
                }
                // least_squares(res, p, loss="soft_l1", f_scale=1, max_nfev=50, x_scale="jac")
                p = TRF.solve(p, loss: .softL1, maxNfev: 50, residuals: res)
                let d = res(p).dropLast()
                let lim = max(2.0, 0.5 * search)
                let sq = d.filter { abs($0) < lim }.map { $0 * $0 }      // np.mean: pairwise sum
                let rmse = sq.isEmpty ? Double.infinity : (NumpyCompat.sum(sq) / Double(sq.count)).squareRoot()
                search = min(max(4 * rmse + 2.0, 2.5), search)
            }
            return (p, search)
        }

        var (p, search) = solve(camParams(cam, withK1: false), search0, nIter)
        if !search.isFinite { return nil }
        var stats = fitStats(p, cx: cx, cy: cy, samples, search: search)
        if fitK1 != .off {
            let (pk, sk) = solve(p + [0.0], search, 2)
            if sk.isFinite && abs(pk[7]) < 0.6 {
                let skStats = fitStats(pk, cx: cx, cy: cy, samples, search: sk)
                let better = skStats.rmse < 0.95 * stats.rmse && skStats.coverage >= stats.coverage - 0.01
                    && (skStats.explained ?? 0) >= (stats.explained ?? 0)
                if fitK1 == .on || better { p = pk; search = sk; stats = skStats }
            }
        }
        var out = makeCamera(f: exp(p[0]), rvec: SIMD3(p[1], p[2], p[3]), t: SIMD3(p[4], p[5], p[6]),
                             size: cam.imageSize, k1: p.count > 7 ? p[7] : 0, cx: cx, cy: cy)
        if out.center.z <= 0.2 { return nil }
        out.reprojRmsePx = stats.rmse
        return (out, stats)
    }
}
