import Foundation
import ShuttlCore

// Stage 1 — candidate preparation and the static-distractor map
// (_timestamps, _flatten, _neighbour_counts, _static_costs, _dp_params).

/// `_timestamps(meta, n)`: meta.timestamps (length-fixed, strictly increasing) or nominal fps.
func linkerTimestamps(_ meta: VideoMeta, _ n: Int) -> [Double] {
    let fps = meta.fps.isFinite && meta.fps > 0 ? meta.fps : 30.0   // `fps and isfinite(fps) and fps > 0`
    var ts = meta.timestamps
    if ts.count < n {
        let start = ts.isEmpty ? 0.0 : ts[ts.count - 1] + 1.0 / fps
        let k = n - ts.count
        for i in 0..<k { ts.append(start + Double(i) / fps) }
    }
    if ts.count > n { ts = Array(ts[0..<n]) }
    if n > 1 && !ts.allSatisfy({ $0.isFinite }) {
        ts = (0..<n).map { Double($0) / fps }
    }
    if n > 1 {
        var anyBad = false
        for i in 0..<(n - 1) where ts[i + 1] - ts[i] <= 0 { anyBad = true; break }
        if anyBad { ts = retimeDuplicates(ts, 1.0 / fps) }   // repair duplicates / reversals
    }
    return ts
}

/// `_retime_duplicates(ts, step0)`: duplicate PTS = repeated frames. A reversed or stalled
/// clock repeats the previous time (running max); each run of equal times keeps its first
/// frame's time and spreads its repeats evenly up to the next distinct time. A trailing run
/// (or a whole-clip run) advances by the clip's median positive step (else `step0`).
/// Strictly increasing afterwards (a step below the float resolution becomes one ulp).
func retimeDuplicates(_ ts: [Double], _ step0: Double) -> [Double] {
    let n = ts.count
    guard n > 1 else { return ts }
    var t = ts
    for i in 1..<n where t[i] < t[i - 1] { t[i] = t[i - 1] }          // np.maximum.accumulate (finite)
    var pos: [Double] = []
    for i in 1..<n where t[i] - t[i - 1] > 0 { pos.append(t[i] - t[i - 1]) }
    let stepEnd = pos.isEmpty ? step0 : npMedian(pos)
    var out = t
    var a = 0
    while a < n {
        var b = a + 1
        while b < n && !(t[b] - t[b - 1] > 0) { b += 1 }             // run a..<b of equal times
        if b - a >= 2 {
            let step = b < n ? (t[b] - t[a]) / Double(b - a) : stepEnd
            for k in 1..<(b - a) { out[a + k] = t[a] + Double(k) * step }
        }
        a = b
    }
    for i in 1..<n where !(out[i] > out[i - 1]) { out[i] = out[i - 1].nextUp }   // np.nextafter(·, inf)
    return out
}

/// CSR layout of the candidates: frame f owns rows off[f]..<off[f+1] (top-k by score).
struct Flat {
    var off: [Int]
    var fidx: [Int]
    var xs: [Double]
    var ys: [Double]
    var score: [Double]
    var count: Int { xs.count }
}

/// `_flatten(candidates, cfg, width, height)`.
func flatten(_ candidates: [[ShuttleCandidate]], _ cfg: LinkerConfig, _ width: Double, _ height: Double) -> Flat {
    let n = candidates.count
    var off = [Int](repeating: 0, count: n + 1)
    var fidx: [Int] = [], xs: [Double] = [], ys: [Double] = [], sc: [Double] = []
    let est = candidates.reduce(0) { $0 + Swift.min($1.count, cfg.maxCandidates) }
    fidx.reserveCapacity(est); xs.reserveCapacity(est); ys.reserveCapacity(est); sc.reserveCapacity(est)
    let x0 = -0.05 * width, x1 = 1.05 * width, y0 = -0.05 * height, y1 = 1.05 * height
    var cs: [ShuttleCandidate] = []
    for f in 0..<n {
        cs.removeAll(keepingCapacity: true)
        for c in candidates[f] where c.score >= cfg.minScore && c.x.isFinite && c.y.isFinite { cs.append(c) }
        if cs.count > cfg.maxCandidates {                 // sorted(key=-score) is stable
            let order = cs.indices.sorted { a, b in
                let sa = -cs[a].score, sb = -cs[b].score
                return sa < sb || (sa == sb && a < b)
            }
            cs = order.prefix(cfg.maxCandidates).map { cs[$0] }
        }
        for c in cs {
            fidx.append(f)
            xs.append(npClip(c.x, x0, x1))
            ys.append(npClip(c.y, y0, y1))
            sc.append(npClip(c.score, 0.0, 1.0))
        }
        off[f + 1] = off[f] + cs.count
    }
    return Flat(off: off, fidx: fidx, xs: xs, ys: ys, score: sc)
}

/// `_neighbour_counts`: for every candidate, distinct frames in its block with a
/// masked candidate in the 3×3 cell neighbourhood.
func neighbourCounts(key: [Int], fidx: [Int], mask: [Bool], cx: [Int], cy: [Int], blk: [Int],
                     ncx: Int, ncy: Int, nFrames: Int) -> [Double] {
    let m = fidx.count
    var out = [Double](repeating: 0, count: m)
    var fk: [Int] = []
    fk.reserveCapacity(m)
    for i in 0..<m where mask[i] { fk.append(key[i] &* (nFrames + 1) &+ fidx[i]) }
    if fk.isEmpty { return out }
    fk = uniqueSorted(fk)
    var ukey: [Int] = [], cnt: [Int] = []
    for v in fk {
        let k = floorDiv(v, nFrames + 1)
        if let last = ukey.last, last == k { cnt[cnt.count - 1] += 1 } else { ukey.append(k); cnt.append(1) }
    }
    let nu = ukey.count
    for ox in -1...1 {
        for oy in -1...1 {
            for i in 0..<m {
                let q = (blk[i] &* ncy &+ (cy[i] &+ oy)) &* ncx &+ (cx[i] &+ ox)
                let pos = Swift.min(searchSorted(ukey, q), nu - 1)
                if ukey[pos] == q { out[i] += Double(cnt[pos]) }
            }
        }
    }
    return out
}

/// `_static_costs(fidx, xs, ys, t, diag, cfg, path_xy)`: additive cost per candidate
/// for persistent static distractors. Throws where numpy would raise (np.bincount of a
/// negative / gigantic block index) or an index key would overflow Int64.
func staticCosts(_ fl: Flat, _ t: [Double], _ diag: Double, _ cfg: LinkerConfig, path: [V2]?) throws -> [Double] {
    let m = fl.count, n = t.count
    if m == 0 { return [] }
    let fps = 1.0 / pyMax(n > 1 ? medianDiff(t) : 1.0 / 30.0, 1e-4)
    let cell = pyMax(cfg.staticCell * diag, 2.0)
    var cx = [Int](repeating: 0, count: m), cy = [Int](repeating: 0, count: m)
    var cxMin = Int.max, cyMin = Int.max
    for i in 0..<m {                                    // floor(...).astype(int64), shifts wrap like numpy
        cx[i] = npInt64((fl.xs[i] / cell).rounded(.down)); cy[i] = npInt64((fl.ys[i] / cell).rounded(.down))
        cxMin = Swift.min(cxMin, cx[i]); cyMin = Swift.min(cyMin, cy[i])
    }
    var cxMax = Int.min, cyMax = Int.min
    for i in 0..<m {
        cx[i] = cx[i] &- cxMin &+ 2; cy[i] = cy[i] &- cyMin &+ 2
        cxMax = Swift.max(cxMax, cx[i]); cyMax = Swift.max(cyMax, cy[i])
    }
    let ncx = cxMax &+ 3, ncy = cyMax &+ 3
    let dur = n > 1 ? t[n - 1] - t[0] : 0.0
    let q = pyFloorDiv(dur, cfg.staticBlockS)           // Python: int(dur // block_s)
    if q.isNaN { throw LinkerError.invalidTimestamps("clip duration \(dur) s") }        // int(nan)
    if q >= 9.2e18 { throw LinkerError.inputTooLarge("clip duration \(dur) s") }        // int(inf) / bincount
    let nblk = q < 1 ? 1 : Int(q)
    // int64 keys below wrap silently like numpy's; they are only sorted and compared, never indices
    var fblk = [Int](repeating: 0, count: n)
    let durD = pyMax(dur, 1e-9)
    for i in 0..<n {
        fblk[i] = npInt64(npMin(((t[i] - t[0]) / durD * Double(nblk)).rounded(.down), Double(nblk - 1)))
        if fblk[i] < 0 { throw LinkerError.invalidTimestamps("frame \(i) before frame 0 (np.bincount: negative)") }
    }
    // np.bincount(fblk, minlength=nblk) only over the blocks that occur (a garbage clock can
    // ask for 10⁸ blocks; the counts of the occupied ones are identical)
    let ub = uniqueSorted(fblk)
    let fbc = fblk.map { searchSorted(ub, $0) }
    var blk = [Int](repeating: 0, count: m), key = [Int](repeating: 0, count: m)
    for i in 0..<m {
        blk[i] = fblk[fl.fidx[i]]
        key[i] = (blk[i] &* ncy &+ cy[i]) &* ncx &+ cx[i]
    }
    let floorFrames = cfg.staticMinS * fps

    @inline(__always) func ramp(_ frac: Double, _ lo: Double, _ hi: Double) -> Double {
        cfg.staticCost * npClip((frac - lo) / pyMax(hi - lo, 1e-6), 0.0, 1.0)
    }

    var framesB = [Double](repeating: 0, count: ub.count)
    for b in fbc { framesB[b] += 1 }
    let occ = neighbourCounts(key: key, fidx: fl.fidx, mask: [Bool](repeating: true, count: m), cx: cx, cy: cy,
                              blk: blk, ncx: ncx, ncy: ncy, nFrames: n)
    var cost = [Double](repeating: 0, count: m)
    for i in 0..<m {
        cost[i] = ramp(occ[i] / npMax(framesB[fbc[fl.fidx[i]]], floorFrames), cfg.staticOccLo, cfg.staticOccHi)
    }
    if let path {
        var far = [Bool](repeating: false, count: m)
        let farD = cfg.staticFar * diag
        for i in 0..<m {
            let p = path[fl.fidx[i]]
            far[i] = p.x.isFinite && hypot(fl.xs[i] - p.x, fl.ys[i] - p.y) > farD
        }
        var presB = [Double](repeating: 0, count: ub.count)
        for f in 0..<n where path[f].x.isFinite { presB[fbc[f]] += 1 }
        let minPres = cfg.staticMinPresentS * fps
        let els = neighbourCounts(key: key, fidx: fl.fidx, mask: far, cx: cx, cy: cy, blk: blk,
                                  ncx: ncx, ncy: ncy, nFrames: n)
        for i in 0..<m {
            let denom = npMax(presB[fbc[fl.fidx[i]]], minPres)
            cost[i] = npMax(cost[i], ramp(els[i] / denom, cfg.staticLo, cfg.staticHi))
        }
    }
    return cost
}

/// DP parameters (`prm` of linker.py) and the link gap in frames.
struct DPParams {
    var vmax: Double, sig2: Double, acc2: Double, lmin: Double, up: Double
    var kcap: Double, cfirst: Double, cbirth: Double, cmiss: Double, lyoung: Double
    var kgap: Int
}

/// `_dp_params(cfg, diag, t)`.
func dpParams(_ cfg: LinkerConfig, _ diag: Double, _ t: [Double]) -> (DPParams, Int) {
    let dtMed = t.count > 1 ? medianDiff(t) : 1.0 / 30.0
    // max(1, int(ceil(x))); a gap ≥ n frames links exactly like n (g0 = max(f − gap, shot0) and
    // f − g ≤ kgap for every f < n), so huge values are capped instead of overflowing Int.
    func frames(_ s: Double) -> Int {
        let x = (s / pyMax(dtMed, 1e-4) - 1e-6).rounded(.up)
        let cap = Double(Swift.max(t.count, 1))
        return x >= cap ? Swift.max(t.count, 1) : (x >= 1 ? Int(x) : 1)
    }
    let gap = frames(cfg.maxLinkGapS)
    let kgap = frames(cfg.kinkMaxGapS)
    let prm = DPParams(vmax: cfg.maxSpeed * diag, sig2: pow(cfg.sigmaPx * diag, 2.0),
                       acc2: pow(cfg.accelSigma * diag, 2.0), lmin: cfg.dragLengthMin * diag,
                       up: cfg.speedUpMax, kcap: cfg.kinkCost, cfirst: cfg.firstCost, cbirth: cfg.birthCost,
                       cmiss: cfg.missCost, lyoung: cfg.dragLengthYoung * diag, kgap: kgap)
    return (prm, gap)
}
