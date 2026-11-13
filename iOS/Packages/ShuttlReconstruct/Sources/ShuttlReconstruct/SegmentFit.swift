import Foundation
import ShuttlCore
import simd

// Port of the segment-fit driver of reconstruct3d.py: _plausible, _pick, _solve,
// fit_segment/_fit_segment, _fit_core, _pred_err, _consistent_prefix, _init_theta,
// _trailing_outliers, _th7.

/// `(ok, -cost) > (best_ok, -best_cost)` with Python tuple semantics.
@inline(__always) func keyGreater(_ ok: Bool, _ cost: Double, _ bestOk: Bool, _ bestCost: Double) -> Bool {
    if ok != bestOk { return ok && !bestOk }
    return -cost > -bestCost
}

/// numpy `.min()` (NaN propagates).
@inline(__always) func npMin(_ xs: [Double]) -> Double {
    var m = Double.infinity
    for x in xs {
        if x.isNaN { return .nan }
        if x < m { m = x }
    }
    return m
}

/// numpy mean of a boolean mask (NaN for an empty slice).
@inline(__always) func fracTrue(_ xs: ArraySlice<Bool>) -> Double {
    xs.isEmpty ? .nan : Double(xs.filter { $0 }.count) / Double(xs.count)
}

/// `_th7`.
@inline(__always) func th7(_ th: [Double], _ L: Double) -> [Double] { th.count == 7 ? th : th + [log(L)] }

/// `_plausible`.
func plausible(_ prob: Problem, _ th: [Double]) -> Bool {
    let (Pa, Va) = prob.simulate(th)
    let n = prob.nObs
    let lo = R3.volumeLo, hi = R3.volumeHi
    var zMin = Double.infinity, cMin = Double.infinity
    var zNaN = false, cNaN = false
    for i in 0..<n {
        let p = Pa[i]
        if p.z.isNaN { zNaN = true } else if p.z < zMin { zMin = p.z }
        let cz = p.x * prob.r20 + p.y * prob.r21 + p.z * prob.r22 + prob.t2
        if cz.isNaN { cNaN = true } else if cz < cMin { cMin = cz }
    }
    if zNaN { zMin = .nan }
    if cNaN { cMin = .nan }
    var ok = th[0] >= lo.x - 0 && th[1] >= lo.y - 0 && th[2] >= lo.z - 0.1
        && th[0] <= hi.x && th[1] <= hi.y && th[2] <= hi.z
        && norm3(SIMD3(th[3], th[4], th[5])) < R3.maxSpeed
        && zMin > -0.15 && cMin > 0.5
    if ok && prob.iStart >= 0 {                        // also at the (extrapolated) contact
        ok = norm3(Va[prob.iStart]) < R3.maxSpeed && Pa[prob.iStart].z > -0.15
    }
    return ok
}

/// `_pick`.
func pick(_ prob: Problem, _ fits: [Fit]) -> Fit? {
    var best: Fit? = nil
    var bestOk = false
    for f in fits {
        let ok = plausible(prob, f.th)
        if best == nil || keyGreater(ok, f.cost, bestOk, best!.cost) { best = f; bestOk = ok }
    }
    return best
}

/// `_solve`: multi-hypothesis coarse LM → refine the 3 best distinct starts → pick.
func solve(_ prob: Problem, _ t: [Double], _ uv: [SIMD2<Double>], initTheta: [Double]?) -> Fit? {
    let contactEpoch = prob.tRef < t[0] - 1e-6
    if let ini = initTheta, !contactEpoch {                  // warm start (+ feet depths)
        let hyps = [ini] + hypotheses(prob, t, uv, feetOnly: true)
        return pick(prob, hyps.map { lm(prob, $0, maxIter: 40) })
    }
    var hyps = contactEpoch ? hypothesesContact(prob, t, uv) : hypotheses(prob, t, uv)
    if let ini = initTheta { hyps = [ini] + hyps }
    if hyps.isEmpty { return nil }
    let coarse0 = hyps.map { lm(prob, $0, maxIter: 6, tol: 1e-4) }
    let coarse = stableOrder(coarse0.map(\.cost)).map { coarse0[$0] }
    var best: Fit? = nil
    var bestOk = false
    var kept: [[Double]] = []
    for f in coarse {
        if kept.count >= 3 { break }
        if kept.contains(where: { k in
            norm3(SIMD3(f.th[0] - k[0], f.th[1] - k[1], f.th[2] - k[2])) < 0.15
        }) { continue }
        kept.append(f.th)
        let ff = lm(prob, f.th, maxIter: 40)
        let ok = plausible(prob, ff.th)
        if best == nil || keyGreater(ok, ff.cost, bestOk, best!.cost) { best = ff; bestOk = ok }
    }
    return best
}

/// Everything the segment fit needs besides the data (config + camera).
struct Engine {
    let cfg: ReconstructConfig
    let camera: CameraModel

    /// `_init_theta`: warm start (θ7, epoch) moved to prob's epoch (nil if it blows up).
    func initTheta(_ ini: (theta: [Double], epoch: Double)?, _ prob: Problem) -> [Double]? {
        guard let ini else { return nil }
        var th7i = ini.theta
        let ti = ini.epoch
        if abs(ti - prob.tRef) > 1e-9 {
            if abs(ti - prob.tRef) > 1.0 { return nil }
            let (pi, vi) = stateAt(th7i, prob.tRef - ti)
            if !(vi.isFinite && norm3(vi) < 1.5 * R3.maxSpeed) { return nil }
            th7i = [pi.x, pi.y, pi.z, vi.x, vi.y, vi.z, th7i[6]]
        }
        return prob.fitDrag ? th7i : Array(th7i[0..<6])
    }

    /// `_fit_core`: fit sorted valid observations → (problem, fit, contact_observed) or nil.
    func fitCore(_ to: ArraySlice<Double>, _ uvo: ArraySlice<SIMD2<Double>>, _ wo: ArraySlice<Double>,
                 _ prior: SegmentPrior, _ fitDrag: Bool, _ ini: (theta: [Double], epoch: Double)?)
        -> (Problem, Fit, Bool)? {
        let to = Array(to), uvo = Array(uvo), wo = Array(wo)
        let contactSeen = prior.startT == nil || to[0] - prior.startT! <= cfg.maxBackExtrap
        // drag is weakly observable on short/slow flights → keep it at the prior
        let fitDragC = fitDrag && to.count >= 10
        let fd = fitDragC && (to[to.count - 1] - to[0] > 0.3)
        if !contactSeen && (prior.startXYWorld != nil || prior.expectedSide != nil) {
            // contact long before the first sighting: fit at the contact epoch
            let prob = Problem(t: to, uv: uvo, w: wo, camera: camera, prior: prior, fitDrag: fitDragC,
                               tRef: prior.startT)
            if let fit = solve(prob, to, uvo, initTheta: initTheta(ini, prob)), plausible(prob, fit.th) {
                return (prob, fit, true)
            }
        }
        var pr = prior
        if !contactSeen { pr.startT = nil; pr.startXYWorld = nil; pr.expectedSide = nil }
        let prob = Problem(t: to, uv: uvo, w: wo, camera: camera, prior: pr, fitDrag: fd)
        guard let fit = solve(prob, to, uvo, initTheta: initTheta(ini, prob)), plausible(prob, fit.th) else {
            return nil
        }
        return (prob, fit, contactSeen)
    }

    /// `_pred_err`: pixel distance between observations and the fitted flight at `to`.
    func predErr(_ prob: Problem, _ th: [Double], _ to: ArraySlice<Double>, _ uvo: ArraySlice<SIMD2<Double>>) -> [Double] {
        let (P, _) = stateAt(th7(th, prob.LFixed), to.map { $0 - prob.tRef })
        var e = [Double](repeating: 0, count: P.count)
        for (k, (p, q)) in zip(P, uvo).enumerated() {
            let d = camera.project(p) - q
            e[k] = (d.x * d.x + d.y * d.y).squareRoot()
        }
        return e
    }

    /// `_consistent_prefix`: longest prefix one free flight explains when the whole-arc fit
    /// looks contaminated by post-flight points; else nil.
    func consistentPrefix(_ to: [Double], _ uvo: [SIMD2<Double>], _ wo: [Double], _ prior: SegmentPrior,
                          _ fitDrag: Bool, _ prob: Problem?, _ fit: Fit?) -> Int? {
        let n = to.count
        if n < 12 { return nil }
        let pr = prior.withoutEnd
        let k0 = max(6, Int(0.3 * Double(n)))
        guard let r0 = fitCore(to[..<k0], uvo[..<k0], wo[..<k0], pr, fitDrag, nil) else { return nil }
        let e0 = predErr(r0.0, r0.1.th, to[..<k0], uvo[..<k0])
        let noise = max(Num.median(e0), 0.5)
        let thr = max(cfg.prefixMinThr, cfg.prefixThrMult * noise / 1.177)   // Rayleigh median → sigma
        var eAll: [Double]? = nil
        if let prob, let fit {
            let ea = predErr(prob, fit.th, to[...], uvo[...])
            eAll = ea
            if !(fracTrue(ea[Int(0.6 * Double(n))...].map { $0 > thr }[...]) > cfg.tailBadFrac) {
                return nil                                   // the whole arc is consistent
            }
        }
        var best: Int? = nil
        for frac in [0.3, 0.5, 0.7] {
            var k = max(6, Int(frac * Double(n)))
            var good: Int? = nil
            for it in 0..<8 {                                // grow → refit until stable
                let res = (frac == 0.3 && it == 0) ? r0 : fitCore(to[..<k], uvo[..<k], wo[..<k], pr, fitDrag, nil)
                guard let res else { break }
                let ek = predErr(res.0, res.1.th, to[...], uvo[...])
                if fracTrue(ek[Int(0.6 * Double(k))..<k].map { $0 > thr }[...]) > cfg.tailBadFrac {
                    break                                    // its own tail is inconsistent
                }
                good = k
                var j = k
                while j < n && !(ek[j] > thr && (j + 1 >= n || ek[j + 1] > thr)) { j += 1 }
                if j >= n && eAll != nil { good = nil }      // explains everything: no trim
                if j <= k || j >= n { break }
                k = j
            }
            if let g = good, 5 <= g && g < n - 1 && (best == nil || g > best!) { best = g }
        }
        if let b = best, let eAll {
            guard let rb = fitCore(to[..<b], uvo[..<b], wo[..<b], pr, fitDrag, nil) else { return nil }
            let eb = predErr(rb.0, rb.1.th, to[..<b], uvo[..<b])
            // accept if the whole arc really misses the dropped points and the trimmed flight
            // follows the kept ones better — or if the whole arc is a compromise that misses
            // the kept points by far (> PREFIX_GAIN x)
            let eKept = Num.median(Array(eAll[..<b])), eTrim = Num.median(eb)
            if !((Num.median(Array(eAll[b...])) > thr && eTrim < 0.8 * eKept)
                 || (b >= 10 && eKept > cfg.prefixGain * eTrim)) {
                return nil
            }
        }
        return best
    }

    /// `_trailing_outliers`: length of the run of gross outliers (> max(8 px, 4× median))
    /// at the end of the observations, if >= 2 and >= 5 points remain; else 0.
    func trailingOutliers(_ prob: Problem, _ th: [Double]) -> Int {
        let (r, _) = prob.residuals(th, jac: false)
        let n = prob.nObs
        var e = [Double](repeating: 0, count: n)
        for i in 0..<n {
            e[i] = (r[2 * i] * r[2 * i] + r[2 * i + 1] * r[2 * i + 1]).squareRoot() * R3.sigmaPx / max(prob.sw[i], 1e-9)
        }
        let thr = max(8.0, 4.0 * Num.median(e))
        var k = 0
        while k < n && e[n - 1 - k] > thr { k += 1 }
        return k >= 2 && n - k >= 5 ? k : 0
    }

    /// `fit_segment`: fit one ODE flight to pixel observations (NaN / zero-weight rows skipped).
    func fitSegment(t: [Double], uv: [SIMD2<Double>], w: [Double]?, prior: SegmentPrior?, fitDrag: Bool,
                    frames: [Int]?, ini: (theta: [Double], epoch: Double)?) -> ReconstructedSegment? {
        let w = w ?? [Double](repeating: 1, count: t.count)
        let frames = frames ?? Array(0..<t.count)
        var okIdx: [Int] = []
        for i in 0..<t.count where uv[i].x.isFinite && uv[i].y.isFinite && t[i].isFinite && w[i] > 0 {
            okIdx.append(i)
        }
        if okIdx.count < 5 { return nil }
        okIdx = Num.argsort(okIdx.map { t[$0] }).map { okIdx[$0] }
        let to = okIdx.map { t[$0] }, uvo = okIdx.map { uv[$0] }, wo = okIdx.map { w[$0] }
        let prior = prior ?? SegmentPrior()
        let gap = prior.startT.map { to[0] - $0 } ?? 0.0
        guard let res = fitCore(to[...], uvo[...], wo[...], prior, fitDrag, ini) else {
            // the whole arc is no plausible single flight (post-flight points) → longest prefix
            guard let k = consistentPrefix(to, uvo, wo, prior, fitDrag, nil, nil),
                  let res = fitCore(to[..<k], uvo[..<k], wo[..<k], prior.withoutEnd, fitDrag, nil) else {
                return nil
            }
            var seg = finalize(res.0, res.1, t, okIdx[..<k].map { frames[$0] }, frames)
            seg.contactObserved = res.2; seg.contactGapS = gap
            seg.nTrimmed = to.count - k
            return seg
        }
        var (prob, fit, contactSeen) = res
        var nKeep = to.count
        // a trailing run of outliers = the flight ended (net, floor skid, mislink)
        let cut = trailingOutliers(prob, fit.th)
        // a tail the whole-arc fit cannot follow → longest consistent prefix
        let k = cut != 0 ? nil : consistentPrefix(to, uvo, wo, prior, fitDrag, prob, fit)
        if cut != 0 || (k != nil && k! != 0) {
            let keep = cut != 0 ? to.count - cut : k!
            let pr = cut != 0 ? prior : prior.withoutEnd
            if let res2 = fitCore(to[..<keep], uvo[..<keep], wo[..<keep], pr, fitDrag,
                                  (th7(fit.th, prob.LFixed), prob.tRef)) {
                (prob, fit, contactSeen) = res2
                nKeep = keep
            }
        }
        var seg = finalize(prob, fit, t, okIdx[..<nKeep].map { frames[$0] }, frames)
        seg.contactObserved = contactSeen; seg.contactGapS = gap
        seg.nTrimmed = to.count - nKeep
        return seg
    }
}
