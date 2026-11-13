import Foundation
import ShuttlCore

/// Shuttle linker: noisy per-frame detector candidates → one clean 2-D track.
/// Faithful Swift port of `backend/app/services/tracking/linker.py`
/// (static-distractor map → Viterbi association with weak-candidate bans →
/// repeated-frame content clock → GLRT kinks → piecewise robust banded smoothing
/// → gap filling → re-adoption of off-clock detections).
public enum ShuttlLinkerVersion {
    public static let version = "1.0.0"
}

// Stage 7 — _finish, _obs_pairs_for_frames, _readopt, link_candidates,
// smooth_track, find_kinks (+ _track_obs, _prepare, _geometry).

struct FinishOut {
    var track: Track2D
    var kinkFrames: [Int]
    var kinkTau: [Double]
    var kinkZ: [Double]
}

/// `_obs_pairs_for_frames`: hit frames → observation pairs (strongest GLRT among
/// the pairs touching [t_f − dt/2, t_f + dt/2]).
func obsPairsForFrames(_ t: [Double], _ fr: [Int], _ kf: [Int], _ mScan: [Int], _ zScan: [Double]) -> [Int] {
    if fr.count < 2 || kf.isEmpty { return [] }
    let dt = t.count > 1 ? medianDiff(t) : 1.0 / 30.0
    let to = fr.map { t[$0] }
    var zmap = [Double](repeating: 0, count: fr.count - 1)
    for (k, m) in mScan.enumerated() { zmap[m] = zScan[k] }
    var out: [Int] = []
    for f in kf {
        var lo = searchSorted(to, t[f] - 0.5 * dt, right: true) - 1
        var hi = searchSorted(to, t[f] + 0.5 * dt)
        lo = Swift.max(lo, 0); hi = Swift.min(hi, fr.count - 1)
        if hi <= lo { continue }
        var best = lo
        for c in lo..<hi where zmap[c] > zmap[best] || (zmap[c].isNaN && !zmap[best].isNaN) { best = c; if zmap[c].isNaN { break } }
        out.append(best)
    }
    return uniqueSorted(out)
}

/// `_finish(t, xy, sc, start, diag, cfg, kink_frames)`: kinks → pieces → robust
/// smoothing → gap filling. Returns the track plus the kink frames, times and scores.
func finish(_ t: [Double], _ xy: [V2], _ sc: [Double], _ start: [Bool], _ diag: Double, _ cfg: LinkerConfig,
            kinkFrames: [Int]? = nil) -> FinishOut {
    let n = t.count
    var outXY = [V2](repeating: nanV2(), count: n)
    var fr: [Int] = []
    for f in 0..<n where xy[f].x.isFinite { fr.append(f) }
    if fr.isEmpty {
        return FinishOut(track: Track2D(t: t, xy: outXY, conf: [Double](repeating: 0, count: n),
                                        visible: [Bool](repeating: false, count: n),
                                        interpolated: [Bool](repeating: false, count: n),
                                        rawXY: [V2](repeating: nanV2(), count: n)),
                         kinkFrames: [], kinkTau: [], kinkZ: [])
    }
    let p = fr.map { xy[$0] }
    var st = fr.map { start[$0] }
    st[0] = true
    let to = fr.map { t[$0] }
    let seg = segments(to, st, cfg.maxGapS)
    let sigma = estimateSigma(to, p, seg, cfg.sigmaPx * diag, cfg)
    let scan = kinkScan(to, p, seg, sigma: sigma, cfg)
    let ms = scan.m, zs = scan.z
    let bigDv = scan.dv.map { $0 > cfg.kinkMinDv * diag }
    var km: [Int] = [], kz: [Double] = [], ktau: [Double] = []
    var forced = [Bool](repeating: false, count: ms.count)
    let rounds: Int
    if let kfr = kinkFrames {
        km = obsPairsForFrames(t, fr, kfr, ms, zs)
        kz = [Double](repeating: .infinity, count: km.count)
        rounds = 0
    } else {
        rounds = cfg.refineKinks ? cfg.refineRounds : 0
    }
    var sm = Smoothed(vf: [], x: [], resid: [])
    for rnd in 0...Swift.max(rounds, 0) {
        if kinkFrames == nil {
            km = []; kz = []
            for i in 0..<ms.count where (zs[i] > cfg.kinkZ || forced[i]) && bigDv[i] { km.append(ms[i]); kz.append(zs[i]) }
            if !km.isEmpty {
                let keep = nms1d(km.map { 0.5 * (to[$0] + to[$0 + 1]) }, kz, cfg.kinkNmsS)
                km = keep.map { km[$0] }; kz = keep.map { kz[$0] }
            }
        }
        (km, ktau) = kinkLocate(to, p, seg, km, cfg)
        if !km.isEmpty {
            let keep = nms1d(ktau, kz, cfg.kinkNmsS)
            km = keep.map { km[$0] }; ktau = keep.map { ktau[$0] }; kz = keep.map { kz[$0] }
        }
        let pc = pieces(fr, t, seg, km, ktau, cfg)
        sm = solvePieces(t, fr, p, pc, sigma, diag, cfg)
        if rnd == rounds || ms.isEmpty { break }
        // a run of ≥ 2 large residuals among obs m−1 … m+2 around a pair
        var cb = [Int](repeating: 0, count: fr.count + 1)
        for i in 0..<fr.count {
            let r = sm.resid[i].isNaN ? 0.0 : sm.resid[i]          // nan_to_num (resid is never ±inf)
            cb[i + 1] = cb[i] + (r > cfg.kinkResidK ? 1 : 0)
        }
        var anyNew = false
        var newF = [Bool](repeating: false, count: ms.count)
        for i in 0..<ms.count {
            let loI = Swift.max(ms[i] - 1, 0), hiI = Swift.min(ms[i] + 3, fr.count)
            let run = cb[hiI] - cb[loI] >= 2
            if run && zs[i] > cfg.kinkZRefine && bigDv[i] && !forced[i] && !(zs[i] > cfg.kinkZ) {
                newF[i] = true; anyNew = true
            }
        }
        if !anyNew { break }
        for i in 0..<ms.count where newF[i] { forced[i] = true }
    }
    for (k, f) in sm.vf.enumerated() { outXY[f] = sm.x[k] }
    var obsOk = [Bool](repeating: false, count: n)
    for (i, f) in fr.enumerated() {
        if !sm.resid[i].isFinite { outXY[f] = p[i] }              // tiny pieces: pass through
        obsOk[f] = !(sm.resid[i] > cfg.outlierK)
    }
    var vis = [Bool](repeating: false, count: n), interp = vis
    var conf = [Double](repeating: 0, count: n)
    var raw = [V2](repeating: nanV2(), count: n)
    for f in 0..<n {
        let fin = outXY[f].x.isFinite
        vis[f] = obsOk[f] && fin
        interp[f] = fin && !vis[f]
        if vis[f] { conf[f] = npClip(sc[f], 0.0, 1.0); raw[f] = xy[f] }
    }
    var kf: [Int] = []
    if !ktau.isEmpty && n >= 2 {                                  // frame nearest τ (kinks need ≥ 6 obs)
        kf = ktau.map { tau in
            let k = Swift.min(Swift.max(searchSorted(t, tau), 1), n - 1)
            return abs(t[k - 1] - tau) <= abs(t[k] - tau) ? k - 1 : k
        }
    }
    return FinishOut(track: Track2D(t: t, xy: outXY, conf: conf, visible: vis, interpolated: interp, rawXY: raw),
                     kinkFrames: kf, kinkTau: ktau, kinkZ: kz)
}

/// `_readopt`: interpolated frames that hold a strong, non-static, on-trajectory
/// candidate within readopt_gate of the fill take that detection.
func readopt(_ tr: Track2D, _ candidates: [[ShuttleCandidate]], _ width: Double, _ height: Double,
             _ cfg: LinkerConfig) throws -> Track2D {
    let fl = flatten(candidates, cfg, width, height)
    let m = fl.count
    if m == 0 { return tr }
    let t = tr.t
    let diag = hypot(width, height)
    let stat = try staticCosts(fl, t, diag, cfg, path: nil)
    var d = [Double](repeating: 0, count: m)
    var ok = [Bool](repeating: false, count: m)
    let gate = cfg.readoptGate * diag
    for i in 0..<m {
        let fill = tr.xy[fl.fidx[i]]
        d[i] = hypot(fl.xs[i] - fill.x, fl.ys[i] - fill.y)
        ok[i] = tr.interpolated[fl.fidx[i]] && fl.score[i] >= cfg.strongScore && stat[i] < 1.0 && d[i] <= gate
    }
    if !ok.contains(true) { return tr }
    // … and lies ON the local trajectory; the fill must be clearly wrong for that image
    let n = t.count
    let tol4 = 4.0 * cfg.sigmaPx * diag, tol6 = 6.0 * cfg.sigmaPx * diag
    for i in 0..<m where ok[i] {
        let pt = V2(fl.xs[i], fl.ys[i])
        var dmin = Double.infinity
        for g in -2...1 {
            let ai = Swift.min(Swift.max(fl.fidx[i] + g, 0), n - 1)
            let bi = Swift.min(Swift.max(fl.fidx[i] + g + 1, 0), n - 1)
            let a = tr.xy[ai], b = tr.xy[bi]
            let ab = b - a
            let pa = pt - a
            let lam = npClip((pa.x * ab.x + pa.y * ab.y) / npMax(ab.x * ab.x + ab.y * ab.y, 1e-9), 0.0, 1.0)
            let r = pa - lam * ab
            let dd = hypot(r.x, r.y)
            if dd.isFinite { dmin = npMin(dmin, dd) }
        }
        let ai = Swift.min(Swift.max(fl.fidx[i] - 1, 0), n - 1), bi = Swift.min(Swift.max(fl.fidx[i] + 1, 0), n - 1)
        let sd = tr.xy[bi] - tr.xy[ai]
        var locStep = 0.5 * hypot(sd.x, sd.y)
        if locStep.isNaN { locStep = 0 } else if locStep == .infinity { locStep = .greatestFiniteMagnitude }
        let far = d[i] >= npMax(tol6, 0.35 * locStep)
        if !((dmin <= tol4) && far) { ok[i] = false }
    }
    if !ok.contains(true) { return tr }
    var xy = tr.xy, vis = tr.visible, itp = tr.interpolated, conf = tr.conf
    var raw = tr.rawXY ?? [V2](repeating: nanV2(), count: xy.count)
    for f in 0..<n {                                              // strongest per frame
        var best = -1
        for i in fl.off[f]..<fl.off[f + 1] where ok[i] {
            if best < 0 || -fl.score[i] < -fl.score[best] { best = i }
        }
        guard best >= 0 else { continue }
        xy[f] = V2(fl.xs[best], fl.ys[best])
        raw[f] = xy[f]
        vis[f] = true; itp[f] = false; conf[f] = fl.score[best]
    }
    return Track2D(t: tr.t, xy: xy, conf: conf, visible: vis, interpolated: itp, rawXY: raw)
}

/// Per-frame detector candidates → one clean, gap-filled 2-D shuttle track
/// (`link_candidates`). `sceneCuts` = first frames of new shots; cuts detected from
/// the candidate stream are added. Nothing is linked, filled or extrapolated across a
/// cut. `rawXY` = the adopted detection (unsmoothed), NaN elsewhere.
public func linkCandidates(_ candidates: [[ShuttleCandidate]], meta: VideoMeta,
                           config: LinkerConfig = LinkerConfig(), sceneCuts: [Int]? = nil) throws -> Track2D {
    let cfg = config
    try cfg.validate()                                // Python would raise on these configs
    let n = candidates.count
    let t = linkerTimestamps(meta, n)
    if n == 0 {
        return Track2D(t: t, xy: [], conf: [], visible: [], interpolated: [], rawXY: [])
    }
    let width = Double(meta.width != 0 ? meta.width : 1920)
    let height = Double(meta.height != 0 ? meta.height : 1080)
    let diag = hypot(width, height)
    // a hole in the presentation clock (PTSGaps) is a cut: nothing is linked, filled,
    // smoothed or kink-fitted across time the camera never delivered
    let cuts = cutList(n, sceneCuts, try detectCuts(candidates, t, width, height, cfg) + PTSGaps.frames(t))
    let asc = try associate(candidates, t, width, height, cfg, shot0: shot0(n, cuts))
    var rep = cfg.repeatDetect
        ? findRepeats(t, asc.xy, probeCandidates(candidates, asc.xy, cfg, width, height), diag, cfg) : nil
    if rep != nil && !cuts.isEmpty { for c in cuts { rep!.isDup[c] = false } }   // a new shot is never a repeat
    guard let rep else {
        let tr = finish(t, asc.xy, asc.sc, cutStarts(asc.xy, asc.start, cuts), diag, cfg).track
        return try readopt(tr, candidates, width, height, cfg)
    }
    let keep = rep.isDup.map { !$0 }                  // re-link the unique frames on the content clock
    let ku = flatnonzero(keep)
    let cu = uniqueSorted(cuts.map { searchSorted(ku, $0) }).filter { $0 > 0 && $0 < ku.count }
    let candU = ku.map { candidates[$0] }
    let tU = ku.map { rep.clock[$0] }
    let a2 = try associate(candU, tU, width, height, cfg, shot0: shot0(ku.count, cu))
    let trU = finish(tU, a2.xy, a2.sc, cutStarts(a2.xy, a2.start, cu), diag, cfg).track
    return expandTrack(try readopt(trU, candU, width, height, cfg), keep, t)
}

// MARK: - smooth_track / find_kinks

/// A caller-built track the stages can index safely: all arrays of length t.count, t finite.
/// (Python raises on mismatched shapes and on round(NaN) of a NaN clock; these non-throwing
/// APIs return the track unchanged / no kinks instead of crashing.)
func trackIsUsable(_ tr: Track2D) -> Bool {
    let n = tr.t.count
    guard tr.xy.count == n, tr.conf.count == n, tr.visible.count == n, tr.interpolated.count == n,
          tr.rawXY == nil || tr.rawXY!.count == n else { return false }
    return tr.t.allSatisfy(\.isFinite)
}

/// `_track_obs`: visible observations (raw_xy where present and preferRaw).
func trackObs(_ track: Track2D, preferRaw: Bool) -> ([V2], [Double], [Bool]) {
    var xy = track.xy
    if preferRaw, let raw = track.rawXY, raw.count == xy.count {
        for i in 0..<xy.count {
            xy[i] = V2(raw[i].x.isFinite ? raw[i].x : xy[i].x, raw[i].y.isFinite ? raw[i].y : xy[i].y)
        }
    }
    var sc = [Double](repeating: 0, count: xy.count)
    for i in 0..<xy.count {
        let vis = track.visible[i] && xy[i].x.isFinite
        if !vis { xy[i] = nanV2() } else { sc[i] = track.conf[i] }
    }
    return (xy, sc, [Bool](repeating: false, count: xy.count))
}

/// `_geometry(width, height, xy)`: image diagonal (guessed ≥ 720p from the track span if unknown).
func geometry(_ imageSize: (width: Int, height: Int)?, _ xy: [V2]) -> Double {
    if let s = imageSize, s.width != 0, s.height != 0 { return hypot(Double(s.width), Double(s.height)) }
    let fin = xy.filter { $0.x.isFinite }
    if fin.isEmpty { return hypot(1920.0, 1080.0) }
    var lo = fin[0], hi = fin[0]
    for v in fin { lo = V2(npMin(lo.x, v.x), npMin(lo.y, v.y)); hi = V2(npMax(hi.x, v.x), npMax(hi.y, v.y)) }
    let span = hi - lo
    return pyMax(hypot(span.x, span.y) * 1.2, hypot(1280.0, 720.0))
}

/// Re-smooth an existing track from its visible frames (raw_xy when present)
/// (`smooth_track`). `kinks` = frame indices of hits; nil → detect them.
public func smoothTrack(_ track: Track2D, kinks: [Int]? = nil, config: LinkerConfig = LinkerConfig(),
                        imageSize: (width: Int, height: Int)? = nil, sceneCuts: [Int]? = nil) -> Track2D {
    let cfg = config
    guard trackIsUsable(track), (try? cfg.validate()) != nil else { return track }
    let t = track.t
    let (xy, sc, start0) = trackObs(track, preferRaw: true)
    let diag = geometry(imageSize, xy)
    let cuts = cutList(t.count, sceneCuts, PTSGaps.frames(t))    // clock holes are cuts too
    let start = cutStarts(xy, start0, cuts)
    var kf: [Int]? = nil
    if let k = kinks { kf = k.filter { $0 >= 0 && $0 < t.count } }
    var rep = findRepeats(t, xy, xy, diag, cfg)
    if rep != nil && !cuts.isEmpty { for c in cuts { rep!.isDup[c] = false } }
    guard let rep else { return finish(t, xy, sc, start, diag, cfg, kinkFrames: kf).track }
    let keep = rep.isDup.map { !$0 }
    let ku = flatnonzero(keep)
    if let k = kf {                                   // frame → unique-slot index
        var cs = [Int](repeating: 0, count: keep.count)
        var c = 0
        for i in 0..<keep.count { if keep[i] { c += 1 }; cs[i] = c }
        kf = uniqueSorted(k.map { cs[$0] - 1 })
    }
    let trU = finish(ku.map { rep.clock[$0] }, ku.map { xy[$0] }, ku.map { sc[$0] }, ku.map { start[$0] },
                     diag, cfg, kinkFrames: kf).track
    return expandTrack(trU, keep, t)
}

/// Frame indices (sorted) of sharp velocity changes in a track's visible frames and
/// their GLRT scores (`find_kinks(..., return_scores=True)`).
public func findKinks(_ track: Track2D, config: LinkerConfig = LinkerConfig(),
                      imageSize: (width: Int, height: Int)? = nil,
                      sceneCuts: [Int]? = nil) -> (frames: [Int], scores: [Double]) {
    let cfg = config
    guard trackIsUsable(track), (try? cfg.validate()) != nil else { return ([], []) }
    let t = track.t
    let (xy, sc, start0) = trackObs(track, preferRaw: false)
    let diag = geometry(imageSize, xy)
    let cuts = cutList(t.count, sceneCuts, PTSGaps.frames(t))    // clock holes are cuts too
    let start = cutStarts(xy, start0, cuts)
    var rep = findRepeats(t, xy, xy, diag, cfg)
    if rep != nil && !cuts.isEmpty { for c in cuts { rep!.isDup[c] = false } }
    var kf: [Int], z: [Double]
    if let rep {
        let keep = rep.isDup.map { !$0 }
        let ku = flatnonzero(keep)
        let o = finish(ku.map { rep.clock[$0] }, ku.map { xy[$0] }, ku.map { sc[$0] }, ku.map { start[$0] }, diag, cfg)
        kf = kinksToFrames(o.kinkFrames, keep); z = o.kinkZ
    } else {
        let o = finish(t, xy, sc, start, diag, cfg)
        kf = o.kinkFrames; z = o.kinkZ
    }
    let order = stableArgsort(kf.map(Double.init))
    return (order.map { kf[$0] }, order.map { z[$0] })
}
