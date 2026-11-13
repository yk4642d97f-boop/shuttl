import Foundation
import ShuttlCore
import simd

/// Racket-contact (hit) detection from a 2-D shuttle track, optionally helped by a
/// calibrated camera and player observations. Port of backend/app/services/tracking/hits.py
/// (change-point scan → candidates → hitter evidence → Viterbi over {near, far}).
public enum Hits {
    public static let version = "hits-port-1"
}

/// Racket contacts in time order (hits.py `detect_hits`). `t` is the sub-frame ML contact
/// time, `frame` the nearest frame, `xyPx` the fitted contact pixel, `hitter` near/far
/// (unknown if the posterior is in (0.4, 0.6)), `confidence` = P(contact),
/// `playerTrackID` = the hitter's track if a player was in reach.
/// `rescorer` defaults to the model installed in `HitModelRegistry` (none by default).
public func detectHits(track: Track2D, meta: VideoMeta, camera: CameraModel? = nil,
                       players: [PlayerObservation]? = nil, config: HitConfig = HitConfig(),
                       rescorer: (any HitRescorer)? = HitModelRegistry.installed) -> [HitEvent] {
    let cfg = config
    // per-frame arrays of different lengths are indexed by one frame number below:
    // no hits instead of a trap (the stage's caller sees an empty result, not a crash)
    if track.inconsistency != nil { return [] }
    let xy = cfg.useRawXY ? Hits.preferRaw(track) : track.xy
    let cands = Hits.candidates(track: track, xy: xy, meta: meta, camera: camera, players: players,
                                cfg: cfg, rescorer: rescorer)
    if cands.isEmpty { return [] }
    let diag = NP.pyHypot(Double(meta.width != 0 ? meta.width : 1280), Double(meta.height != 0 ? meta.height : 720))
    let hasPlayers = !(players?.isEmpty ?? true)
    let pairW = cfg.pairYWeight * ((camera != nil || hasPlayers) ? 0.5 : 1.0)
    let dec = HitDecoder(cands, cfg, diag, pairW)
    let chain = dec.viterbi().filter { dec.hit[$0.k] >= cfg.minHitLogodds }
    let idx = chain.map(\.k)
    let pNear = dec.posteriors(idx)
    var out: [HitEvent] = []
    out.reserveCapacity(idx.count)
    for (k, pn) in zip(idx, pNear) {
        let c = cands[k]
        let side: CourtSide = pn >= 0.6 ? .near : (pn <= 0.4 ? .far : .unknown)
        let tid = side == .near ? c.nearTID : (side == .far ? c.farTID : nil)
        out.append(HitEvent(frame: c.frame, t: c.tau, xyPx: c.xy, hitter: side,
                            confidence: 1.0 / (1.0 + Foundation.exp(-dec.hit[k])), playerTrackID: tid,
                            source: c.gapContact ? "gap_contact" : HitEvent.defaultSource))
    }
    return out
}

/// All scored hit candidates before sequence decoding (hits.py `detect_candidates`).
/// Note: like Python, this does NOT apply `use_raw_xy` (detectHits does).
public func detectCandidates(track: Track2D, meta: VideoMeta, camera: CameraModel? = nil,
                             players: [PlayerObservation]? = nil, config: HitConfig = HitConfig(),
                             rescorer: (any HitRescorer)? = HitModelRegistry.installed) -> [HitCandidate] {
    if track.inconsistency != nil { return [] }
    return Hits.candidates(track: track, xy: track.xy, meta: meta, camera: camera, players: players,
                           cfg: config, rescorer: rescorer)
}

extension Hits {
    static func candidates(track: Track2D, xy trackXY: [SIMD2<Double>], meta: VideoMeta, camera: CameraModel?,
                           players: [PlayerObservation]?, cfg: HitConfig,
                           rescorer: (any HitRescorer)?) -> [HitCandidate] {
        let W = Double(meta.width != 0 ? meta.width : (camera?.imageSize.width ?? 1280))
        let H = Double(meta.height != 0 ? meta.height : (camera?.imageSize.height ?? 720))
        let diag = NP.pyHypot(W, H)
        var oi = observed(track, xy: trackXY)
        if oi.count < 2 * cfg.minSide { return [] }
        let trackT = track.t
        var xy = oi.map { trackXY[$0] }
        let cc = contentClock(frames: oi, xy: xy, trackT: trackT)
        var t: [Double] = []
        do {
            var o2: [Int] = [], x2: [SIMD2<Double>] = []
            for i in oi.indices where cc.keep[i] { o2.append(oi[i]); t.append(cc.t[i]); x2.append(xy[i]) }
            oi = o2; xy = x2
        }
        let sigma: Double
        if let s = cfg.sigmaPx, s != 0 { sigma = s } else { sigma = estimateSigma(t: t, xy: xy) }
        do {
            let keep = rejectOutliers(t: t, xy: xy, sigma: sigma, diag: diag)
            var o2: [Int] = [], t2: [Double] = [], x2: [SIMD2<Double>] = []
            for i in keep.indices where keep[i] { o2.append(oi[i]); t2.append(t[i]); x2.append(xy[i]) }
            oi = o2; t = t2; xy = x2
        }
        if t.count < 2 * cfg.minSide { return [] }
        let dt: Double = trackT.count > 1
            ? NP.median((0..<(trackT.count - 1)).map { trackT[$0 + 1] - trackT[$0] })
            : 1.0 / NP.pyMax(meta.fps, 1.0)
        let tFirstObs = t[0], tLastObs = t[t.count - 1]
        let grid = trackT.filter { $0 >= tFirstObs && $0 <= tLastObs }
        let fitter = Fitter(t: t, xy: xy)
        let obs = ObsSeries(t: t, xy: xy)
        let sc = scan(fitter, taus: grid, cfg: cfg, sigma: sigma)
        let score = sc.lamV                              // velocity-kink statistic peaks at the contact
        let hasPlayers = !(players?.isEmpty ?? true)
        let lamThr = hasPlayers ? Swift.min(cfg.lamMin, cfg.lamMinWeak) : cfg.lamMin
        var peaks = localMaxima(score, sc.ok, lamThr)
        if let kinks = cfg.linkerKinkTimes, !grid.isEmpty {
            for tk in kinks where tk.isFinite {
                var g = 0
                for i in 1..<grid.count where abs(grid[i] - tk) < abs(grid[g] - tk) { g = i }
                if sc.ok[g] && peaks.allSatisfy({ abs(grid[g] - grid[$0]) > 2 * dt }) { peaks.append(g) }
            }
        }
        if !peaks.isEmpty {
            let keep = nms(times: peaks.map { grid[$0] }, scores: peaks.map { score[$0] }, radius: cfg.nmsS)
            peaks = keep.map { peaks[$0] }
        }
        var cands: [HitCandidate] = []
        let staticTol = cfg.staticPx * diag / NP.pyHypot(1280, 720)
        for g in peaks {
            guard let r = refine(fitter, tau: grid[g], lo: sc.lo[g], hi: sc.hi[g], dt: dt, cfg: cfg, sigma: sigma)
            else { continue }
            let m = r.m
            let gap = (0 < m && m < t.count) ? t[m] - t[m - 1] : 0.0
            cands.append(HitCandidate(tau: r.tau, frame: 0, xy: r.p, lam: r.lam, lamV: r.lamV, vIn: r.vIn,
                                      vOut: r.vOut, gapS: gap, weak: score[g] < cfg.lamMin))
        }
        let dead = deadGaps(obs, cfg, W, H)
        for fs in flightStarts(obs, cfg, diag, dt, dead, trackT[0]) where cands.allSatisfy({ abs($0.tau - fs.tau) > cfg.nmsS }) {
            cands.append(HitCandidate(tau: fs.tau, frame: 0, xy: fs.xy, lam: 0, lamV: 0, vIn: SIMD2(0, 0), vOut: fs.v,
                                      gapS: fs.i > 0 ? t[fs.i] - t[fs.i - 1] : .infinity, isStart: true))
        }
        if cfg.hiddenContacts {
            for hc in hiddenContacts(obs, cfg, W, H) where cands.allSatisfy({ abs($0.tau - hc.tau) > cfg.nmsS }) {
                cands.append(HitCandidate(tau: hc.tau, frame: 0, xy: hc.xy, lam: 0, lamV: 0, vIn: SIMD2(0, 0),
                                          vOut: SIMD2(0, 0), gapS: t[hc.i] - t[hc.i - 1], hidden: true))
            }
        }
        if cfg.gapContacts {
            for gc in gapContacts(fitter, oi: oi, t: t, trackT: trackT, cfg, sigma: sigma)
            where cands.allSatisfy({ abs($0.tau - gc.tau) > cfg.nmsS }) {
                cands.append(HitCandidate(tau: gc.tau, frame: 0, xy: xy[gc.i], lam: gc.lam, lamV: 0, vIn: gc.vIn,
                                          vOut: gc.vOut, gapS: t[gc.i] - t[gc.i - 1], gapContact: true))
            }
        }
        cands = cands.enumerated().sorted { a, b in
            a.element.tau < b.element.tau || (a.element.tau == b.element.tau && a.offset < b.offset)
        }.map(\.element)
        let breaks = rallyBreaks(obs, cfg, staticTol, dead)
        for i in cands.indices {
            // a start candidate sits just before its break-ending first detection
            cands[i].seg = NP.searchsorted(breaks, cands[i].tau + (cands[i].isStart ? dt : 0.0))
        }
        let pidx = hasPlayers ? PlayerIndex(players: players!, camera: camera) : nil
        let radius = Swift.max(1, Int((0.05 * (meta.fps != 0 ? meta.fps : 30.0)).rounded(.toNearestOrEven)))
        var camW = 0.0
        if let camera {
            camW = cfg.cameraWeight * (camera.confidence > 0 ? NP.clip(camera.confidence, 0.3, 1.0) : 1.0)
        }
        let b = cfg.borderPx * diag / NP.pyHypot(1280, 720)
        let nT = trackT.count
        for i in cands.indices {
            var c = cands[i]
            c.frame = Swift.min(Swift.max(NP.searchsorted(trackT, c.tau), 0), nT - 1)
            if c.frame > 0 && abs(trackT[c.frame - 1] - c.tau) < abs(trackT[c.frame] - c.tau) { c.frame -= 1 }
            if !c.hidden {                   // a hidden contact has no pixel to test these on
                c.outside = c.xy.x < -b || c.xy.x > W + b || c.xy.y < -b || c.xy.y > H + b
                c.staticAfter = isStatic(obs, c.tau, c.tau + cfg.staticS, staticTol)
                c.staticBefore = isStatic(obs, c.tau - cfg.staticS, c.tau, staticTol)
                c.settles = settles(obs, c.tau, cfg, staticTol, cfg.settleFrac * diag)
            }
            c.kin = kinematicLogodds(c, diag, cfg)
            if let pidx, !c.hidden { playerEvidence(&c, pidx, radius, cfg) }
            if camW > 0, let camera, c.xy.x.isFinite, c.xy.y.isFinite {
                c.sideLLR += camW * raySideLLR(camera, c.xy)
            }
            c.feats = features(c, diag)
            cands[i] = c
        }
        cands = cands.filter { !$0.weak || $0.reach >= cfg.weakReachMin * cfg.playerWeight }
        if let rescorer, !cands.isEmpty {
            if let z = try? rescorer.logits(cands.map(\.feats)) {
                for (i, zi) in zip(cands.indices, z) {        // zip truncates like Python
                    cands[i].kin = (1 - cfg.learnedBlend) * cands[i].kin + cfg.learnedBlend * zi
                }
            }
        }
        return cands
    }
}
