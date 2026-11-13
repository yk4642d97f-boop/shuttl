import Foundation
import ShuttlCore

// Stage 6 — repeated video frames and scene cuts: _find_repeats, _probe_candidates,
// _expand, _kinks_to_frames, _cut_list, _shot0, _cut_starts, _detect_cuts.

/// `_find_repeats(t, path, probe, diag, cfg)`: (is_dup, content clock) or nil.
func findRepeats(_ t: [Double], _ path: [V2], _ probe: [V2], _ diag: Double,
                 _ cfg: LinkerConfig) -> (isDup: [Bool], clock: [Double])? {
    let n = t.count
    if !cfg.repeatDetect || n < 8 { return nil }
    var ok = [Bool](repeating: false, count: n)
    var tf: [Double] = [], pf: [V2] = []
    for f in 0..<n where path[f].x.isFinite { ok[f] = true; tf.append(t[f]); pf.append(path[f]) }
    if tf.count < 4 * cfg.repeatMinCount { return nil }
    let sig = estimateSigma(tf, pf, [Int](repeating: 0, count: tf.count), cfg.sigmaPx * diag, cfg)
    var dupF: [Int] = []
    var anyF = false
    let fastMin = pyMax(6.0 * sig, cfg.repeatMinMove * diag)
    if n - 1 > 2 {
        for f in 2..<(n - 1) where ok[f - 1] && ok[f - 2] && ok[f + 1] && probe[f].x.isFinite {
            anyF = true
            let a = path[f - 1] - path[f - 2], b = path[f + 1] - path[f - 1]
            let la = hypot(a.x, a.y), lb = hypot(b.x, b.y)
            let cos = (a.x * b.x + a.y * b.y) / npMax(la * lb, 1e-9)
            let fast = npMin(la, lb) > fastMin
            let dp = probe[f] - path[f - 1]
            let d = hypot(dp.x, dp.y)
            if fast && cos > 0.8 && lb < 1.6 * la && lb > 0.6 * la && d <= npMax(2.0 * sig, 0.15 * la) {
                dupF.append(f)
            }
        }
    }
    if !anyF { return nil }
    if dupF.count < 2 { return nil }
    var isDup = [Bool](repeating: false, count: n)
    for f in dupF { isDup[f] = true }
    // contradiction: the path clearly moves between slot f−1 and f
    var contra = [Bool](repeating: false, count: n)
    if n > 2 {
        for g in 2..<n where ok[g] && ok[g - 1] && ok[g - 2] {
            let s1 = path[g] - path[g - 1], s0 = path[g - 1] - path[g - 2]
            contra[g] = hypot(s1.x, s1.y) > npMax(3.0 * sig, 0.5 * hypot(s0.x, s0.y))
        }
    }
    let prOk = probe.map { $0.x.isFinite }

    func gridSlots(_ q: Int) -> [Int] {
        var out: [Int] = []
        for k in 0..<(dupF.count - 1) {
            let a_ = dupF[k], b_ = dupF[k + 1]
            if (b_ - a_) % q == 0 && b_ - a_ <= 100 * q {
                var s = a_ + q
                while s < b_ { out.append(s); s += q }
            }
        }
        return out.filter { 1 < $0 && $0 < n }
    }

    func localExt(_ q: Int) -> [Int] {
        var e: [Int] = []
        for s in dupF { for o in -3...3 where o != 0 { let v = s + q * o; if v > 1 && v < n { e.append(v) } } }
        e = uniqueSorted(e)
        var out: [Int] = []
        for v in e {
            var laE = 0.0
            if ok[v - 1] && ok[v - 2] { let d = path[v - 1] - path[v - 2]; laE = hypot(d.x, d.y) }
            var de = Double.infinity
            if prOk[v] && ok[v - 1] { let d = probe[v] - path[v - 1]; de = hypot(d.x, d.y) }
            let ev = de <= npMax(2.0 * sig, 0.35 * laE)
            if ev || !prOk[v] { out.append(v) }
        }
        return out
    }

    var P = 0
    var fill: [Int] = []
    if dupF.count >= cfg.repeatMinCount {
        var dd: [Int] = []
        for k in 0..<(dupF.count - 1) { let v = dupF[k + 1] - dupF[k]; if v <= 300 { dd.append(v) } }
        for q in 3...30 {                    // smallest period whose grid looks like repeats
            if dd.isEmpty { continue }
            let hitCount = dd.reduce(0) { $0 + ($1 % q == 0 ? 1 : 0) }
            if boolMean(dd.count, hitCount) < 0.6 { continue }
            let gq = gridSlots(q)
            var ev: [Int] = [], laQ: [Double] = []
            for g in gq where ok[g - 1] && ok[g - 2] && prOk[g] {
                let d = path[g - 1] - path[g - 2]
                let l = hypot(d.x, d.y)
                if l > 4.0 * sig { ev.append(g); laQ.append(l) }
            }
            if ev.count >= 3 {
                var good = 0
                for (k, g) in ev.enumerated() {
                    let d = probe[g] - path[g - 1]
                    if hypot(d.x, d.y) <= npMax(2.0 * sig, 0.35 * laQ[k]) { good += 1 }
                }
                if boolMean(ev.count, good) < 0.7 { continue }
            }
            P = q
            fill = uniqueSorted(gq + localExt(q))
            break
        }
    }
    if P != 0 && !fill.isEmpty { for f in fill where !contra[f] { isDup[f] = true } }
    var tc = t
    if P != 0 {
        var tu: [Double] = []
        for f in 0..<n where !isDup[f] { tu.append(t[f]) }
        let w: [Double] = (P - 1) % 2 != 0 ? [Double](repeating: 1, count: P - 1)
            : [0.5] + [Double](repeating: 1, count: P - 2) + [0.5]
        let h = w.count / 2
        if h > 0 && tu.count > 2 * h + 1 {
            var pad: [Double] = []
            pad.reserveCapacity(tu.count + 2 * h)
            for k in stride(from: h, through: 1, by: -1) { pad.append(2 * tu[0] - tu[k]) }
            pad.append(contentsOf: tu)
            let L = tu.count
            for k in 2...(h + 1) { pad.append(2 * tu[L - 1] - tu[L - k]) }
            let wsum = w.reduce(0, +)
            let wn = w.map { $0 / wsum }
            let nw = wn.count
            let nOut = pad.count - nw + 1
            var sm = [Double](repeating: 0, count: Swift.max(nOut, 0))
            for i in 0..<Swift.max(nOut, 0) {
                var s = 0.0
                for k in 0..<nw { s += pad[i + k] * wn[nw - 1 - k] }
                sm[i] = s
            }
            var incr = true
            for i in 0..<Swift.max(sm.count - 1, 0) where !(sm[i + 1] - sm[i] > 0) { incr = false; break }
            if sm.count == tu.count && incr {
                var k = 0
                for f in 0..<n where !isDup[f] { tc[f] = sm[k]; k += 1 }
            }
        }
    }
    // repeated slots share the previous slot's (content) time
    var clock = [Double](repeating: 0, count: n)
    var last = 0
    for f in 0..<n {
        if !isDup[f] { last = Swift.max(last, f) }
        clock[f] = tc[last]
    }
    return (isDup, clock)
}

/// `_probe_candidates`: per frame the candidate nearest to the previous path point
/// (NaN when a stronger candidate sits where the shuttle should be now).
func probeCandidates(_ candidates: [[ShuttleCandidate]], _ path: [V2], _ cfg: LinkerConfig,
                     _ width: Double, _ height: Double) -> [V2] {
    let n = candidates.count
    var q = [V2](repeating: nanV2(), count: n)
    let fl = flatten(candidates, cfg, width, height)
    if fl.count == 0 || n < 3 { return q }
    var prev = [V2](repeating: nanV2(), count: n), fwd = prev
    for f in 1..<n { prev[f] = path[f - 1] }
    for f in 2..<n { fwd[f] = 2 * path[f - 1] - path[f - 2] }
    var bestNear = [Double](repeating: 0, count: n)
    for f in 0..<n {
        var best = -1
        var bd = Double.infinity
        for i in fl.off[f]..<fl.off[f + 1] {
            var d = hypot(fl.xs[i] - prev[f].x, fl.ys[i] - prev[f].y)
            if !d.isFinite { d = .infinity }
            if best < 0 || d < bd { best = i; bd = d }
        }
        if best >= 0 && bd.isFinite {
            q[f] = V2(fl.xs[best], fl.ys[best])
            bestNear[f] = fl.score[best]
        }
    }
    for f in 0..<n {
        let sd = fwd[f] - prev[f]
        let step = hypot(sd.x, sd.y)
        for i in fl.off[f]..<fl.off[f + 1] {
            let df = hypot(fl.xs[i] - fwd[f].x, fl.ys[i] - fwd[f].y)
            if df.isFinite && df <= 0.5 * step && fl.score[i] > bestNear[f] { q[f] = nanV2() }
        }
    }
    return q
}

/// `_expand`: unique-slot track → container frames (a repeated slot gets the previous slot's state).
func expandTrack(_ tr: Track2D, _ keep: [Bool], _ t: [Double]) -> Track2D {
    var src = [Int](repeating: 0, count: keep.count)
    var c = -1
    for i in 0..<keep.count { if keep[i] { c += 1 }; src[i] = c }
    // numpy: a leading repeat (src −1) wraps to the last slot; unreachable (slots 0, 1 are
    // never repeats), mirrored so an empty or short unique track can never be over-read
    if tr.n == 0 { return tr }
    for i in 0..<src.count where src[i] < 0 { src[i] += tr.n }
    let raw = tr.rawXY.map { r in src.map { r[$0] } }
    return Track2D(t: t, xy: src.map { tr.xy[$0] }, conf: src.map { tr.conf[$0] },
                   visible: src.map { tr.visible[$0] }, interpolated: src.map { tr.interpolated[$0] }, rawXY: raw)
}

/// `_kinks_to_frames`.
func kinksToFrames(_ kf: [Int], _ keep: [Bool]) -> [Int] {
    if kf.isEmpty { return kf }
    let k = flatnonzero(keep)
    return kf.map { k[$0] }
}

/// `_cut_list`: sorted unique first-frames of new shots (0 < f < n).
func cutList(_ n: Int, _ explicit: [Int]?, _ detected: [Int]?) -> [Int] {
    uniqueSorted((explicit ?? []) + (detected ?? [])).filter { $0 > 0 && $0 < n }
}

/// `_shot0`: first frame of each frame's shot.
func shot0(_ n: Int, _ cuts: [Int]) -> [Int] {
    if cuts.isEmpty { return [Int](repeating: 0, count: n) }
    let b = [0] + cuts
    return (0..<n).map { b[searchSorted(b, $0, right: true) - 1] }
}

/// `_cut_starts`: start flags + the first observed frame at/after every cut.
func cutStarts(_ xy: [V2], _ start: [Bool], _ cuts: [Int]) -> [Bool] {
    var st = start
    let fr = (0..<xy.count).filter { xy[$0].x.isFinite }
    if !cuts.isEmpty && !fr.isEmpty {
        for c in cuts { let j = searchSorted(fr, c); if j < fr.count { st[fr[j]] = true } }
    }
    return st
}

/// `_detect_cuts`: scene cuts from the candidate stream alone (≥ cut_min_cells
/// persistent static spots ending/starting within ±0.1 s).
func detectCuts(_ candidates: [[ShuttleCandidate]], _ t: [Double], _ width: Double, _ height: Double,
                _ cfg: LinkerConfig) throws -> [Int] {
    let n = t.count
    if !cfg.cutDetect || n < 30 { return [] }
    let fl = flatten(candidates, cfg, width, height)
    let m = fl.count
    if m < 30 { return [] }
    let diag = hypot(width, height)
    let cell = pyMax(cfg.staticCell * diag, 2.0)
    var cx = [Int](repeating: 0, count: m), cy = cx
    var cxMax = Int.min
    for i in 0..<m {
        cx[i] = npInt64((fl.xs[i] / cell).rounded(.down)) &+ 2      // .astype(int64) + 2
        cy[i] = npInt64((fl.ys[i] / cell).rounded(.down)) &+ 2
        cxMax = Swift.max(cxMax, cx[i])
    }
    let ncx = cxMax &+ 3
    var pair = [Int](repeating: 0, count: m)
    // the frame is decoded back from the key (pf indexes t), so an overflow must not wrap here
    for i in 0..<m { pair[i] = try mulAdd(try mulAdd(cy[i], ncx, cx[i]), n + 1, fl.fidx[i]) }
    pair = uniqueSorted(pair)
    let pc = pair.map { floorDiv($0, n + 1) }, pf = pair.map { floorMod($0, n + 1) }
    var ukey: [Int] = [], occ: [Int] = []
    for v in pc { if ukey.last == v { occ[occ.count - 1] += 1 } else { ukey.append(v); occ.append(1) } }
    var nb = [Int](repeating: 0, count: ukey.count)
    for ox in -1...1 {
        for oy in -1...1 where ox != 0 || oy != 0 {
            for k in 0..<ukey.count {
                let q = ukey[k] &+ oy &* ncx &+ ox
                let pos = Swift.min(searchSorted(ukey, q), ukey.count - 1)
                if ukey[pos] == q { nb[k] = Swift.max(nb[k], occ[pos]) }
            }
        }
    }
    var peak = Set<Int>()
    for k in 0..<ukey.count where occ[k] > nb[k] { peak.insert(ukey[k]) }
    let dt = medianDiff(t)
    var r0: [Int] = []
    for i in 0..<pc.count where i == 0 || pc[i] != pc[i - 1] || t[pf[i]] - t[pf[i - 1]] > 1.0 { r0.append(i) }
    var evT: [Double] = [], evC: [Int] = []
    for k in 0..<r0.count {
        let a = r0[k], b = (k + 1 < r0.count ? r0[k + 1] : pc.count) - 1
        let span = t[pf[b]] - t[pf[a]]
        let dens = Double(b - a + 1) * dt / npMax(span, dt)
        guard span >= 3.0 && dens >= 0.3 && peak.contains(pc[a]) else { continue }
        if pf[a] > 0 && t[pf[a]] - t[0] > 0.5 { evT.append(t[pf[a]]); evC.append(pc[a]) }        // appears
        if pf[b] < n - 1 && t[n - 1] - t[pf[b]] > 0.5 {                                            // vanishes
            evT.append(t[Swift.min(pf[b] + 1, n - 1)]); evC.append(pc[a])
        }
    }
    if evT.count < cfg.cutMinCells { return [] }
    let o = stableArgsort(evT)
    let et = o.map { evT[$0] }, ec = o.map { evC[$0] }
    var cuts: [Int] = []
    var i = 0
    while i < et.count {
        let j = searchSorted(et, et[i] + 0.2, right: true)
        // et is sorted and finite, so j > i; the guard keeps a NaN clock from forming a
        // reversed range (Python: an empty slice → no cut, i += 1)
        if j > i && Set(ec[i..<j]).count >= cfg.cutMinCells {
            cuts.append(searchSorted(t, npMedian(Array(et[i..<j])) - 1e-9))
            i = j
        } else {
            i += 1
        }
    }
    return cuts
}
