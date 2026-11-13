import Foundation
import ShuttlCore

// Observation selection and cleaning — hits.py `_observed`, `_prefer_raw`,
// `_content_clock`, `_estimate_sigma`, `_reject_outliers`.

extension Hits {
    /// Indices of frames with a real (non gap-filled) finite detection.
    static func observed(_ track: Track2D, xy: [SIMD2<Double>]) -> [Int] {
        let n = xy.count
        var ok = xy.map { $0.x.isFinite && $0.y.isFinite }
        let vis = track.visible
        var anyVis = false
        if vis.count == n { for i in 0..<n where vis[i] && ok[i] { anyVis = true; break } }
        if anyVis {
            for i in 0..<n { ok[i] = ok[i] && vis[i] }
        } else if track.interpolated.count == n {
            for i in 0..<n { ok[i] = ok[i] && !track.interpolated[i] }
        }
        var out: [Int] = []
        out.reserveCapacity(n)
        for i in 0..<n where ok[i] { out.append(i) }
        return out
    }

    /// Track xy with the unsmoothed raw detections substituted where present.
    /// Only when raw_xy AND xy have one entry per timestamp: a track whose
    /// arrays differ in length (a corrupt checkpoint decoded before ShuttlCore
    /// checked it) used to trap here with "Index out of range" (E, T10 repro
    /// track-len-mismatch); it now keeps xy unchanged (detectHits then refuses
    /// the inconsistent track, see Track2D.inconsistency).
    static func preferRaw(_ track: Track2D) -> [SIMD2<Double>] {
        guard let raw = track.rawXY, raw.count == track.n, track.xy.count == track.n else { return track.xy }
        var xy = track.xy
        for i in 0..<raw.count where raw[i].x.isFinite && raw[i].y.isFinite { xy[i] = raw[i] }
        return xy
    }

    /// Repeated-frame handling: (keep mask, content times) for the observations.
    static func contentClock(frames: [Int], xy: [SIMD2<Double>], trackT: [Double]) -> (keep: [Bool], t: [Double]) {
        let n = frames.count
        var keep = [Bool](repeating: true, count: n)
        let tc = frames.map { trackT[$0] }
        if n < 4 { return (keep, tc) }
        var step = [Double](repeating: 0, count: n - 1)
        for i in 0..<(n - 1) { step[i] = NP.rowNorm(xy[i + 1].x - xy[i].x, xy[i + 1].y - xy[i].y) }
        var consec = [Bool](repeating: false, count: n - 1)
        for i in 0..<(n - 1) { consec[i] = frames[i + 1] - frames[i] == 1 }
        var rep = [Bool](repeating: false, count: n)
        var anyRep = false
        if n >= 4 {
            for i in 2..<(n - 1) {           // obs i repeats obs i-1
                let around = Swift.min(step[i - 2], step[i])
                let sCur = step[i - 1]
                let r = consec[i - 1] && consec[i - 2] && consec[i] && around > 3.0
                    && sCur < Swift.max(0.25 * around, 0.35)
                rep[i] = r
                if r { anyRep = true }
            }
        }
        if !anyRep { return (keep, tc) }
        for i in 0..<n where rep[i] { keep[i] = false }
        let fr = (0..<n).filter { rep[$0] }.map { frames[$0] }
        if fr.count >= 4 {
            let d = (0..<(fr.count - 1)).map { fr[$0 + 1] - fr[$0] }
            let P = modeSmallest(d)
            let okMult: Double = P >= 3 ? Double(d.filter { pyMod($0, P) == 0 }.count) / Double(d.count) : 0
            if P >= 3 && okMult >= 0.8 {
                let phase = modeSmallest(fr.map { pyMod($0, P) })
                let nt = trackT.count
                var isDup = [Bool](repeating: false, count: nt)
                for f in 0..<nt { isDup[f] = pyMod(f - phase, P) == 0 }
                let dt: Double
                if nt > 1 {
                    dt = NP.median((0..<(nt - 1)).map { trackT[$0 + 1] - trackT[$0] })
                } else { dt = 1.0 / 30 }
                var tContent = [Double](repeating: 0, count: nt)
                var c = -1
                for f in 0..<nt {
                    if !isDup[f] { c += 1 }
                    tContent[f] = trackT[0] + Double(c) * dt * Double(P) / Double(P - 1)
                }
                for i in 0..<n where isDup[frames[i]] { keep[i] = false }
                return (keep, frames.map { tContent[$0] })
            }
        }
        return (keep, tc)
    }

    /// `np.unique(values, return_counts=True)` → value with the largest count (smallest on ties).
    static func modeSmallest(_ v: [Int]) -> Int {
        var counts: [Int: Int] = [:]
        for x in v { counts[x, default: 0] += 1 }
        var best = Int.max, bestC = -1
        for (k, c) in counts where c > bestC || (c == bestC && k < best) { best = k; bestC = c }
        return best
    }

    @inline(__always) static func pyMod(_ a: Int, _ b: Int) -> Int { let r = a % b; return r != 0 && (r < 0) != (b < 0) ? r + b : r }

    /// Robust per-axis noise from 3rd differences of consecutive, moving samples.
    static func estimateSigma(t: [Double], xy: [SIMD2<Double>], default def: Double = 1.5) -> Double {
        let n = t.count
        if n < 8 { return def }
        var dt = [Double](repeating: 0, count: n - 1)
        for i in 0..<(n - 1) { dt[i] = t[i + 1] - t[i] }
        let med = NP.median(dt)
        let uni = dt.map { abs($0 - med) < 0.25 * med }
        var v: [Double] = []
        v.reserveCapacity(2 * n)
        for k in 0..<(n - 3) {
            let ok = uni[k] && uni[k + 1] && uni[k + 2]
            let s = NP.rowNorm(xy[k + 1].x - xy[k].x, xy[k + 1].y - xy[k].y)
            guard ok && s > 0.5 else { continue }
            let a = xy[k + 3], b = xy[k + 2], c = xy[k + 1], d = xy[k]
            v.append(a.x - 3 * b.x + 3 * c.x - d.x)
            v.append(a.y - 3 * b.y + 3 * c.y - d.y)
        }
        if v.count / 2 < 6 { return def }
        let m = NP.median(v)
        let s = 1.4826 * NP.median(v.map { abs($0 - m) }) / (20.0).squareRoot()
        return NP.clip(s, 0.5, 8.0)
    }

    /// Keep-mask: a point is an outlier if constant-velocity extrapolations from
    /// BOTH sides miss it by far more than noise plus a motion allowance.
    static func rejectOutliers(t: [Double], xy: [SIMD2<Double>], sigma: Double, diag: Double) -> [Bool] {
        let n = t.count
        var keep = [Bool](repeating: true, count: n)
        if n < 5 { return keep }
        for _ in 0..<2 {
            let idx = (0..<n).filter { keep[$0] }
            let m = idx.count
            if m < 5 { break }
            var anyBad = false
            var bad = [Bool](repeating: false, count: m)
            for q in 0..<m {
                var errL = Double.infinity, stepL = Double.infinity
                var errR = Double.infinity, stepR = Double.infinity
                let pq = xy[idx[q]], tq = t[idx[q]]
                if q >= 2 {
                    let a = xy[idx[q - 2]], b = xy[idx[q - 1]]
                    let ta = t[idx[q - 2]], tb = t[idx[q - 1]]
                    let h = tb - ta
                    let vx = (b.x - a.x) / h, vy = (b.y - a.y) / h
                    let dtq = tq - tb
                    let px = b.x + vx * dtq, py = b.y + vy * dtq
                    errL = NP.rowNorm(pq.x - px, pq.y - py)
                    stepL = NP.rowNorm(vx, vy) * dtq
                }
                if q <= m - 3 {
                    let c = xy[idx[q + 2]], d = xy[idx[q + 1]]
                    let tc = t[idx[q + 2]], td = t[idx[q + 1]]
                    let h = td - tc
                    let vx = (d.x - c.x) / h, vy = (d.y - c.y) / h
                    let dtq = tq - td
                    let px = d.x + vx * dtq, py = d.y + vy * dtq
                    errR = NP.rowNorm(pq.x - px, pq.y - py)
                    stepR = NP.rowNorm(vx, vy) * (td - tq)
                }
                let tolL = 6 * sigma + 0.02 * diag + 1.0 * (stepL.isFinite ? stepL : 0)
                let tolR = 6 * sigma + 0.02 * diag + 1.0 * (stepR.isFinite ? stepR : 0)
                if errL > tolL && errR > tolR { bad[q] = true; anyBad = true }
            }
            if !anyBad { break }
            for q in 0..<m where bad[q] { keep[idx[q]] = false }
        }
        return keep
    }
}
