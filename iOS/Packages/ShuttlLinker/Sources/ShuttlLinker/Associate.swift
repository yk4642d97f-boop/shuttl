import Foundation
import ShuttlCore

// Stage 3 — _weak_unsupported and _associate (global association, 2 static
// passes + ≤ weak_ban_rounds re-associations).

/// `_weak_unsupported`: adopted weak (< strong_score) observations without shuttle support.
func weakUnsupported(_ t: [Double], _ path: [V2], _ sc: [Double], _ start: [Bool], _ kink: [Bool],
                     _ diag: Double, _ cfg: LinkerConfig, strongElsewhere: [Bool]?) throws -> [Bool] {
    let n = t.count
    var out = [Bool](repeating: false, count: n)
    var fr: [Int] = []
    for f in 0..<n where path[f].x.isFinite { fr.append(f) }
    let no = fr.count
    if no == 0 { return out }
    var pid = [Int](repeating: 0, count: no)
    var c = -1
    for i in 0..<no {
        if start[fr[i]] || kink[fr[i]] { c += 1 }
        pid[i] = c
    }
    let pmin = pid[0]
    for i in 0..<no { pid[i] -= pmin }
    let np_ = pid[no - 1] + 1
    // piece ranges (pieces are contiguous)
    var pLo = [Int](repeating: 0, count: np_), pHi = [Int](repeating: 0, count: np_)
    for i in 0..<no { if i == 0 || pid[i] != pid[i - 1] { pLo[pid[i]] = i }; pHi[pid[i]] = i + 1 }
    var strong = [Bool](repeating: false, count: no)
    var nStrong = [Double](repeating: 0, count: np_)
    var to = [Double](repeating: 0, count: no)
    for i in 0..<no {
        strong[i] = sc[fr[i]] >= cfg.strongScore
        if strong[i] { nStrong[pid[i]] += 1 }
        to[i] = t[fr[i]]
    }
    // time to the nearest strong obs of the same piece
    let big = 1e9
    var key = [Double](repeating: 0, count: no)
    var sk: [Double] = []
    for i in 0..<no {
        key[i] = Double(pid[i]) * big + to[i]
        if strong[i] { sk.append(key[i]) }
    }
    var d = [Double](repeating: .infinity, count: no)
    if !sk.isEmpty {
        for i in 0..<no {
            let j = searchSorted(sk, key[i])
            for jj in (j - 1)...j {
                guard jj >= 0 && jj < sk.count else { continue }
                if (sk[jj] / big).rounded(.down) == Double(pid[i]) {
                    d[i] = npMin(d[i], abs(sk[jj] - key[i]))
                }
            }
        }
    }
    var bad = [Bool](repeating: false, count: no)
    for i in 0..<no where !strong[i] {
        bad[i] = nStrong[pid[i]] < Double(cfg.minStrong) || d[i] > cfg.weakReachS
    }
    let flyV = cfg.weakFlySpeed * diag
    // leading weak runs (before the first strong obs of their piece) must fly
    var firstStrong = [Double](repeating: .infinity, count: np_)
    var lastStrong = [Double](repeating: -.infinity, count: np_)
    for i in 0..<no where strong[i] {
        firstStrong[pid[i]] = npMin(firstStrong[pid[i]], to[i])
        lastStrong[pid[i]] = npMax(lastStrong[pid[i]], to[i])
    }
    var lead = [Bool](repeating: false, count: no)
    for i in 0..<no { lead[i] = !strong[i] && to[i] < firstStrong[pid[i]] }
    var leadPieces: [Int] = []
    for i in 0..<no where lead[i] && !bad[i] { if leadPieces.last != pid[i] { leadPieces.append(pid[i]) } }
    for p in leadPieces {
        let sel = (pLo[p]..<pHi[p]).filter { lead[$0] }
        var v: [Double]
        if sel.count > 1 {
            v = []
            for k in 0..<(sel.count - 1) {
                let dd = path[fr[sel[k + 1]]] - path[fr[sel[k]]]
                v.append(hypot(dd.x, dd.y) / npMax(to[sel[k + 1]] - to[sel[k]], 1e-6))
            }
        } else { v = [0] }
        if npMedian(v) < flyV { for s in sel { bad[s] = true } }
    }
    // trailing weak runs (after the last strong obs) must keep the flight direction
    var trail = [Bool](repeating: false, count: no)
    for i in 0..<no {
        let ls = lastStrong[pid[i]]
        trail[i] = !strong[i] && !bad[i] && to[i] > ls && ls.isFinite
    }
    let tol = 4.0 * cfg.sigmaPx * diag
    var trailPieces: [Int] = []
    for i in 0..<no where trail[i] { if trailPieces.last != pid[i] { trailPieces.append(pid[i]) } }
    for p in trailPieces {
        // k = ps[np.searchsorted(to[ps], last_strong[p])] — numpy's binary search (also on a
        // non-monotone clock); k past the piece is Python's IndexError, so throw, never read it
        var lo = pLo[p], hi = pHi[p]
        while lo < hi { let mid = lo + (hi - lo) >> 1; if to[mid] < lastStrong[p] { lo = mid + 1 } else { hi = mid } }
        let k = lo
        if k >= pHi[p] { throw LinkerError.invalidTimestamps("non-monotone clock in piece \(p)") }
        if k == pLo[p] { continue }
        let v = (path[fr[k]] - path[fr[k - 1]]) / pyMax(to[k] - to[k - 1], 1e-6)
        let sp = hypot(v.x, v.y)
        if sp < flyV { continue }
        let u = v / sp
        var off = false
        for s in pLo[p]..<pHi[p] where trail[s] {
            let dd = path[fr[s]] - path[fr[k]]
            let along = dd.x * u.x + dd.y * u.y
            let perp = abs(dd.x * u.y - dd.y * u.x)
            if along < -tol || perp > tol + 0.5 * npMax(along, 0.0) { off = true }
            if off { bad[s] = true }                     // from the first departure on
        }
    }
    if let se = strongElsewhere {
        for i in 0..<no where !strong[i] && se[fr[i]] { bad[i] = true }
    }
    // flying weak-only pieces survive
    var badPieces: [Int] = []
    for i in 0..<no where bad[i] { if badPieces.last != pid[i] { badPieces.append(pid[i]) } }
    for p in badPieces {
        let a = pLo[p], b = pHi[p]
        if b - a < 3 || to[b - 1] - to[a] < cfg.weakFlyMinS { continue }
        var v = [Double](repeating: 0, count: b - a - 1)
        for k in a..<(b - 1) {
            let dd = path[fr[k + 1]] - path[fr[k]]
            v[k - a] = hypot(dd.x, dd.y) / npMax(to[k + 1] - to[k], 1e-6)
        }
        if npMedian(v) >= flyV { for s in a..<b { bad[s] = false } }
    }
    for i in 0..<no where bad[i] { out[fr[i]] = true }
    return out
}

struct Association {
    var xy: [V2]
    var sc: [Double]
    var start: [Bool]
}

/// `_associate(candidates, t, width, height, cfg, shot0)`.
func associate(_ candidates: [[ShuttleCandidate]], _ t: [Double], _ width: Double, _ height: Double,
               _ cfg: LinkerConfig, shot0: [Int]?) throws -> Association {
    let n = t.count
    let diag = hypot(width, height)
    var xy = [V2](repeating: nanV2(), count: n)
    var sc = [Double](repeating: 0, count: n)
    let fl = flatten(candidates, cfg, width, height)
    let m = fl.count
    if m == 0 { return Association(xy: xy, sc: sc, start: [Bool](repeating: false, count: n)) }
    var u0 = [Double](repeating: 0, count: m)
    for i in 0..<m {
        let s = npClip(fl.score[i], 1e-3, 1 - 1e-3)
        u0[i] = -(cfg.detReward + cfg.scoreWeight * log(s / (1 - s)))
    }
    var (prm, gap) = dpParams(cfg, diag, t)
    let sh = shot0 ?? [Int](repeating: 0, count: n)
    var ban = [Double](repeating: 0, count: m)
    var path: [V2]? = nil
    var stat = [Double](repeating: 0, count: m)
    var vr = ViterbiResult(chosen: [], start: [], kink: [])
    var bad = [Bool](repeating: false, count: n)
    var ok = [Bool](repeating: false, count: n)
    var u = [Double](repeating: 0, count: m)
    let wbr = Swift.max(cfg.weakBanRounds, 0)
    for rnd in 0..<(2 + wbr) {
        if rnd < 2 { stat = try staticCosts(fl, t, diag, cfg, path: path) }
        for i in 0..<m { u[i] = u0[i] + stat[i] + ban[i] }
        vr = viterbi(fl, u: u, t: t, prm: prm, gap: gap, shot0: sh)
        var p = [V2](repeating: nanV2(), count: n)
        var nOk = 0
        for f in 0..<n {
            ok[f] = vr.chosen[f] >= 0
            if ok[f] { let c = vr.chosen[f]; p[f] = V2(fl.xs[c], fl.ys[c]); nOk += 1 }
        }
        path = p
        if rnd == 0 && nOk >= 8 * cfg.kinkMinPts {          // detector noise differs per detector / resolution
            var tf: [Double] = [], pf: [V2] = [], sf: [Bool] = []
            for f in 0..<n where ok[f] { tf.append(t[f]); pf.append(p[f]); sf.append(vr.start[f]) }
            let seg0 = segments(tf, sf, cfg.maxGapS)
            let sEst = estimateSigma(tf, pf, seg0, cfg.sigmaPx * diag, cfg)
            prm.sig2 = pyMax(prm.sig2, pow(sEst, 2.0))
        }
        var se = [Bool](repeating: false, count: n)          // a real strong candidate in the frame
        for i in 0..<m where fl.score[i] >= cfg.strongScore && stat[i] < 1.0 { se[fl.fidx[i]] = true }
        var scf = [Double](repeating: 0, count: n)
        var stOk = [Bool](repeating: false, count: n), kOk = stOk
        for f in 0..<n where ok[f] {
            scf[f] = fl.score[vr.chosen[f]]; stOk[f] = vr.start[f]; kOk[f] = vr.kink[f]
        }
        bad = try weakUnsupported(t, p, scf, stOk, kOk, diag, cfg, strongElsewhere: se)
        let anyBad = bad.contains(true)
        if rnd >= 1 && (!anyBad || rnd == 1 + wbr) { break }
        if rnd >= 1 { for f in 0..<n where bad[f] { ban[vr.chosen[f]] = linkerINF / 1e6 } }   // never adopt again
    }
    xy = path!
    var adopted: [Int] = []
    for f in 0..<n where ok[f] { sc[f] = fl.score[vr.chosen[f]]; adopted.append(f) }
    for f in 0..<n where bad[f] { xy[f] = nanV2(); sc[f] = 0 }
    // restart the chain after a dropped run
    var cid0 = [Int](repeating: 0, count: n)
    var c = -1
    for f in 0..<n { if vr.start[f] { c += 1 }; cid0[f] = c }
    var prevBad = [Bool](repeating: false, count: n)
    if adopted.count > 1 { for k in 1..<adopted.count { prevBad[adopted[k]] = bad[adopted[k - 1]] } }
    var obs: [Int] = []
    for f in 0..<n where xy[f].x.isFinite { obs.append(f) }
    var start = [Bool](repeating: false, count: n)
    if !obs.isEmpty {
        start[obs[0]] = true
        for k in 1..<Swift.max(obs.count, 1) where cid0[obs[k]] != cid0[obs[k - 1]] || prevBad[obs[k]] {
            start[obs[k]] = true
        }
        var cid = [Int](repeating: 0, count: n)
        var cc = -1
        for f in 0..<n { if start[f] { cc += 1 }; cid[f] = cc }
        var lens = [Int](repeating: 0, count: cid[obs[obs.count - 1]] + 1)
        for f in obs { lens[cid[f]] += 1 }
        for f in obs where lens[cid[f]] < cfg.minChainLen { xy[f] = nanV2(); sc[f] = 0; start[f] = false }
    }
    return Association(xy: xy, sc: sc, start: start)
}
