import Foundation
import ShuttlCore
import simd

// Port of the rally-level part of reconstruct3d.py: _track_weights, _rest_start,
// _near_border, _Players, _closest_approach, _reepoch, refine_hit_time, _Piece,
// _fit_piece, reconstruct_segments.

/// numpy pairwise summation (`np.sum` / `.sum()` on a contiguous float64 array).
func npSum(_ a: ArraySlice<Double>) -> Double {
    let n = a.count, s = a.startIndex
    if n < 8 {
        var res = 0.0
        for i in 0..<n { res += a[s + i] }
        return res
    }
    if n <= 128 {
        var r = (0..<8).map { a[s + $0] }
        var i = 8
        while i < n - (n % 8) {
            for j in 0..<8 { r[j] += a[s + i + j] }
            i += 8
        }
        var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        while i < n { res += a[s + i]; i += 1 }
        return res
    }
    var n2 = n / 2
    n2 -= n2 % 8
    return npSum(a[s..<(s + n2)]) + npSum(a[(s + n2)..<(s + n)])
}

/// Work queue of `parallelMap`: a lock-guarded index counter and one result slot per index
/// (each slot is written by exactly one worker, read after all workers joined).
private final class ParallelMapState<T>: @unchecked Sendable {
    let body: (Int) -> T
    let n: Int
    private let lock = NSLock()
    private var next = 0
    private var slots: [T?]

    init(_ n: Int, _ body: @escaping (Int) -> T) {
        self.n = n; self.body = body
        slots = [T?](repeating: nil, count: n)
    }

    func claim() -> Int? {
        lock.lock(); defer { lock.unlock() }
        guard next < n else { return nil }
        next += 1
        return next - 1
    }

    func store(_ i: Int, _ v: T) {
        lock.lock(); defer { lock.unlock() }
        slots[i] = v
    }

    var results: [T] { slots.map { $0! } }
}

/// `(0..<n).map(body)` on up to `maxConcurrency` threads (dynamic scheduling, results in
/// index order). `body` must only read immutable captured state (each index is an
/// independent, deterministic fit), so the result does not depend on the thread count.
func parallelMap<T>(_ n: Int, maxConcurrency: Int, _ body: @escaping (Int) -> T) -> [T] {
    if n <= 1 || maxConcurrency <= 1 { return (0..<n).map(body) }
    let state = ParallelMapState(n, body)
    DispatchQueue.concurrentPerform(iterations: min(n, maxConcurrency)) { _ in
        while let i = state.claim() { state.store(i, state.body(i)) }
    }
    return state.results
}

/// Per-frame fit weight: real detections by confidence, filled frames 0.
func trackWeights(_ track: Track2D) -> [Double] {
    let n = track.t.count
    var w = [Double](repeating: 0, count: n)
    var any = false
    for i in 0..<n {
        let ok = track.xy[i].x.isFinite && track.xy[i].y.isFinite
        let vis = i < track.visible.count ? track.visible[i] : ok
        let interp = i < track.interpolated.count ? track.interpolated[i] : false
        var c = i < track.conf.count ? track.conf[i] : 1.0
        if c.isNaN { c = 1.0 }
        if vis && !interp && ok {
            w[i] = min(max(c, 0.3), 1.0)
            if w[i] != 0 { any = true }
        }
    }
    if !any {                                         // no flags: trust every finite point
        for i in 0..<n { w[i] = (track.xy[i].x.isFinite && track.xy[i].y.isFinite) ? 1.0 : 0.0 }
    }
    return w
}

/// Everything reconstructSegments needs about players: feet by frame.
struct PlayersIndex {
    var byFrame: [Int: [PlayerObservation]] = [:]
    var frames: [Int] = []

    init(_ players: [PlayerObservation]?) {
        for o in players ?? [] {
            if let f = o.footWorld, f.x.isFinite, f.y.isFinite { byFrame[o.frame, default: []].append(o) }
        }
        frames = byFrame.keys.sorted()
    }

    /// `_Players.feet`: feet (x, y) of the hitter at `frame` (nearest observation within maxDF).
    func feet(_ frame: Int, _ side: CourtSide, _ trackID: Int?, _ uv: SIMD2<Double>, maxDF: Int = 6) -> SIMD2<Double>? {
        if frames.isEmpty { return nil }
        var lo = 0, hi = frames.count
        while lo < hi { let mid = (lo + hi) / 2; if frames[mid] < frame { lo = mid + 1 } else { hi = mid } }
        let k = min(max(lo, 0), frames.count - 1)
        let cands = [frames[max(k - 1, 0)], frames[k]].filter { abs($0 - frame) <= maxDF }
        guard var bestF = cands.first else { return nil }
        for f in cands.dropFirst() where abs(f - frame) < abs(bestF - frame) { bestF = f }
        let obs = byFrame[bestF]!
        if let trackID {
            for o in obs where o.trackID == trackID { return o.footWorld! }
        }
        var pool = (side == .near || side == .far) ? obs.filter { $0.side == side } : []
        if pool.isEmpty { return nil }
        if pool.count > 1 {                           // doubles: player whose box is nearest the shuttle
            func dpx(_ o: PlayerObservation) -> Double {
                let b = o.bbox
                return hypot(uv.x - min(max(uv.x, b.x1), b.x2), uv.y - min(max(uv.y, b.y1), b.y2))
            }
            let keys = pool.map(dpx)
            pool = stableOrder(keys).map { pool[$0] }
        }
        return pool[0].footWorld!
    }
}

/// `_closest_approach`: time in [tLo, tHi] where two fitted flights are closest in 3-D, and
/// that distance (m).
func closestApproach(_ a: ReconstructedSegment, _ b: ReconstructedSegment, _ tLo: Double, _ tHi: Double) -> (t: Double, d: Double)? {
    if tHi <= tLo { return nil }
    let ts = linspace(tLo, tHi, max(Int((tHi - tLo) * 2000), 8))
    let Pa = stateAt(a.thetaRef, ts.map { $0 - a.tRef }).P
    let Pb = stateAt(b.thetaRef, ts.map { $0 - b.tRef }).P
    let d = zip(Pa, Pb).map { norm3($0 - $1) }
    let k = argmin(d)
    return (ts[k], d[k])
}

extension Engine {
    /// `_reepoch`: move the reported contact state (t0, p0, v0) of a fitted segment to tC.
    func reepoch(_ s: inout ReconstructedSegment, _ tC: Double) {
        let tau = tC - s.tRef
        if !(-cfg.maxBackExtrap <= tau && tau <= 0.05) { return }
        let (p, v) = stateAt(s.thetaRef, tau)
        s.segment.t0 = tC; s.segment.p0 = p; s.segment.v0 = v
        if s.covTheta.allSatisfy(\.isFinite) {
            s.speed0SigmaKmh = speedSigma(s.thetaRef, s.covTheta, tau) * 3.6
        }
    }

    /// `_rest_start`: first index in [i0, i1) from which every detection for REST_S stays
    /// within restPx (the shuttle lies on the floor), or nil.
    func restStart(_ t: [Double], _ xy: [SIMD2<Double>], _ w: [Double], _ i0: Int, _ i1: Int, _ restPx: Double) -> Int? {
        let idx = (i0..<max(i0, i1)).filter { w[$0] > 0 }
        for (a, i) in idx.enumerated() {
            var j = a + 1
            var broke = false
            while j < idx.count && t[idx[j]] - t[i] <= cfg.restS {
                let d = xy[idx[j]] - xy[i]
                if hypot(d.x, d.y) > restPx { broke = true; break }
                j += 1
            }
            if !broke && j - a >= 4 && t[idx[j - 1]] - t[i] >= 0.6 * cfg.restS { return i }
        }
        return nil
    }

    /// `_fit_piece`: fit one hit → end piece; drop post-contact (skid / bounce) detections the
    /// robust loss had to absorb and refit.
    func fitPiece(_ i0: Int, _ i1: Int, _ t: [Double], _ xy: [SIMD2<Double>], _ w: [Double], _ frames: [Int],
                  _ prior: SegmentPrior, _ ini: (theta: [Double], epoch: Double)?) -> ReconstructedSegment? {
        let ts = Array(t[i0..<i1]), xs = Array(xy[i0..<i1]), fs = Array(frames[i0..<i1])
        var ws = Array(w[i0..<i1])
        guard let seg = fitSegment(t: ts, uv: xs, w: ws, prior: prior, fitDrag: true, frames: fs, ini: ini) else { return nil }
        guard prior.end == .floor, let lt = seg.segment.landingT else { return seg }
        let obsT = zip(ts, ws).filter { $0.1 > 0 }.map(\.0)
        var dt = 1.0 / 30.0
        if obsT.count > 1 { dt = Num.median(zip(obsT.dropFirst(), obsT).map { $0 - $1 }) }
        let late = ts.indices.map { ws[$0] > 0 && ts[$0] > lt + dt }
        let nLate = late.filter { $0 }.count, nObs = ws.filter { $0 > 0 }.count
        if nLate > 0 && nObs - nLate >= cfg.minObs {
            for i in ws.indices where late[i] { ws[i] = 0 }
            if let seg2 = fitSegment(t: ts, uv: xs, w: ws, prior: prior, fitDrag: true, frames: fs,
                                     ini: (seg.thetaRef, seg.tRef)) {
                return seg2
            }
        }
        return seg
    }
}

extension Reconstruct3D {
    /// `_near_border`: pixel within frac of the image size from the top / left / right edge.
    static func nearBorder(_ uv: SIMD2<Double>, _ size: ImageSize, frac: Double = 0.1) -> Bool {
        let w = Double(size.width), h = Double(size.height)
        return uv.y < frac * h || uv.x < frac * w || uv.x > (1 - frac) * w
    }

    /// `refine_hit_time`: choose the split within ±window frames where separate smooth fits
    /// before/after explain the 2-D track best, then fit the drag ODE to each side and take
    /// the time where the two 3-D flights meet (closest approach). Falls back to the input.
    public static func refineHitTime(track: Track2D, approxFrame: Int, camera: CameraModel, window: Int = 6,
                                     config: ReconstructConfig = .init()) -> (frame: Int, t: Double) {
        let eng = Engine(cfg: config, camera: camera)
        let t = track.t, xy = track.xy
        let w = trackWeights(track)
        let n = t.count
        let f0 = min(max(approxFrame, 0), n - 1)
        let ok = w.map { $0 > 0 }
        let span = 0.25
        var best: (Double, Int)? = nil
        // presentation-clock gaps split the track: neither side of a split may reach across
        // one (span id per frame; all zeros on a gap-free clip)
        var sid = [Int](repeating: 0, count: n)
        let gaps = PTSGaps.frames(t)
        if !gaps.isEmpty {
            for g in gaps { sid[g] = 1 }
            for i in 1..<n { sid[i] += sid[i - 1] }
        }
        let fLo = max(1, f0 - window), fHi = min(n - 1, f0 + window)
        if fLo <= fHi {
            for f in fLo...fHi {
                if sid[f] != sid[f - 1] { continue }        // a split AT a gap pairs across it
                let a = (0..<n).filter { ok[$0] && t[$0] < t[f] && t[$0] >= t[f] - span && sid[$0] == sid[f] }
                let b = (0..<n).filter { ok[$0] && t[$0] >= t[f] && t[$0] <= t[f] + span && sid[$0] == sid[f] }
                if a.count < 4 || b.count < 4 { continue }
                var err = 0.0
                for idx in [a, b] {
                    let tt = idx.map { t[$0] - t[f] }
                    var A = Matrix(rows: idx.count, cols: 3)
                    var B = Matrix(rows: idx.count, cols: 2)
                    for (r, i) in idx.enumerated() {
                        A[r, 0] = 1; A[r, 1] = tt[r]; A[r, 2] = tt[r] * tt[r]
                        B[r, 0] = xy[i].x; B[r, 1] = xy[i].y
                    }
                    guard let c = LinAlg.lstsq(A, B) else { err = .nan; continue }
                    var e = [Double](repeating: 0, count: idx.count)
                    for r in 0..<idx.count {
                        let du = A[r, 0] * c[0, 0] + A[r, 1] * c[1, 0] + A[r, 2] * c[2, 0] - B[r, 0]
                        let dv = A[r, 0] * c[0, 1] + A[r, 1] * c[1, 1] + A[r, 2] * c[2, 1] - B[r, 1]
                        let ee = (du * du + dv * dv).squareRoot()
                        e[r] = ee.isNaN ? .nan : min(ee, 15.0)
                    }
                    err += npSum(e[...])
                }
                if best == nil || err < best!.0 { best = (err, f) }
            }
        }
        guard let bf = best?.1 else { return (f0, t[f0]) }
        let a = (0..<n).filter { ok[$0] && t[$0] < t[bf] && t[$0] >= t[bf] - 0.6 && sid[$0] == sid[bf] }
        let b = (0..<n).filter { ok[$0] && t[$0] >= t[bf] && t[$0] <= t[bf] + 0.6 && sid[$0] == sid[bf] }
        var tC = (!a.isEmpty && !b.isEmpty) ? 0.5 * (t[a.last!] + t[b[0]]) : t[bf]
        if a.count >= config.minObs && b.count >= config.minObs {
            let sIn = eng.fitSegment(t: a.map { t[$0] }, uv: a.map { xy[$0] }, w: a.map { w[$0] },
                                     prior: SegmentPrior(), fitDrag: true, frames: a, ini: nil)
            let sOut = eng.fitSegment(t: b.map { t[$0] }, uv: b.map { xy[$0] }, w: b.map { w[$0] },
                                      prior: SegmentPrior(), fitDrag: true, frames: b, ini: nil)
            if let sIn, let sOut, let ca = closestApproach(sIn, sOut, t[a.last!] - 0.01, t[b[0]] + 0.01), ca.d < 0.5 {
                tC = ca.t
            }
        }
        let frame = argmin(t.map { abs($0 - tC) })
        return (frame, tC)
    }

    /// `reconstruct_segments`: fit a 3-D flight to every hit → (next hit | floor | out of
    /// view) piece. Pass 1 fits each piece with side/height priors; contact times are then
    /// refined where consecutive flights meet in 3-D; pass 2 refits with the hitter's and
    /// receiver's feet (if players are given) at the refined contact times. Pieces ending
    /// without a next hit look for the shuttle at rest on the floor (landing bracket) and
    /// extrapolate the ODE to z = 0.
    public static func reconstructSegments(track: Track2D, hits: [HitEvent], camera: CameraModel?,
                                           meta: VideoMeta?, players: [PlayerObservation]? = nil,
                                           config: ReconstructConfig = .init()) -> [ReconstructedSegment] {
        let t = track.t, xy = track.xy
        let n = t.count
        guard n >= config.minObs, !hits.isEmpty, let camera else { return [] }
        let eng = Engine(cfg: config, camera: camera)
        let cfg = config
        let w = trackWeights(track)
        let tsMeta = meta?.timestamps ?? []
        var frames: [Int]
        let close = tsMeta.count == n && zip(tsMeta, t).allSatisfy { a, b in
            (a.isNaN && b.isNaN) || abs(a - b) <= 1e-4 + 1e-5 * abs(b)
        }
        if close {
            frames = Array(0..<n)
        } else if !tsMeta.isEmpty {
            // NaN sorts last in numpy, so searchsorted(NaN) = len → clipped to the last frame
            frames = t.map { $0.isNaN ? tsMeta.count - 1 : min(max(Num.searchsorted(tsMeta, $0 - 1e-6), 0), tsMeta.count - 1) }
        } else {
            frames = Array(0..<n)
        }
        let width = Double((meta?.width ?? 0) != 0 ? meta!.width : camera.imageSize.width)
        let restPx = max(4.0, 6.0 * width / 1920.0)
        let finite = hits.filter { $0.t.isFinite }
        let hs = stableOrder(finite.map(\.t)).map { finite[$0] }
        let feet = PlayersIndex(players)

        struct Piece: Sendable {
            var hit: Int
            var i0: Int, i1: Int
            var end: String                         // 'hit' | 'floor' | 'open'
            var endT: Double?
            var nextHit: Int?
            var restXY: SIMD2<Double>?
            var seg: ReconstructedSegment? = nil
            var tContact: Double = .nan
            var atGap = false                       // cut at a presentation-clock gap (end unseen)
            var gapStart = false                    // hit before a gap, first frame after it (contact unseen)
        }
        // presentation-clock gaps (PTSGaps): a flight is never fitted across one. A piece keeps
        // only the frames of its hit's gap-free span; a hit inside a gap starts its piece
        // after the gap (contact unobserved, as when first seen late).
        let gaps = PTSGaps.frames(t)
        var pieces: [Piece] = []
        for (k, h) in hs.enumerated() {
            let nxt: HitEvent? = k + 1 < hs.count ? hs[k + 1] : nil
            let i0 = Num.searchsorted(t, h.t + 1e-6)
            let g = gaps.firstIndex(where: { $0 > i0 }) ?? gaps.count      // searchsorted(side="right")
            let iGap = g < gaps.count ? gaps[g] : n                          // first frame after the next gap
            let gapStart = g > 0 && gaps[g - 1] == i0
            let tLim = h.t + cfg.maxFlightS
            var chained = nxt != nil && nxt!.t <= tLim
            if chained && iGap < n && nxt!.t >= t[iGap - 1] + 1e-6 { chained = false }   // next hit beyond the gap
            var i1 = chained ? Num.searchsorted(t, nxt!.t - 1e-6) : Num.searchsorted(t, tLim)
            var atGap = !chained && iGap < n && i1 >= iGap          // flight window reaches the gap
            i1 = min(i1, iGap)
            let valid = (i0..<max(i0, i1)).filter { w[$0] > 0 }
            if !chained && valid.count > 1 {                // a long gap ends the flight ...
                for g in 0..<(valid.count - 1) where t[valid[g + 1]] - t[valid[g]] > cfg.maxGapS {
                    let a = valid[g], b = valid[g + 1]      // ... unless it left and re-entered the frame
                    if !(nearBorder(xy[a], camera.imageSize) && nearBorder(xy[b], camera.imageSize)) {
                        i1 = a + 1
                        atGap = false                   // a detection gap ended it first
                        break
                    }
                }
            }
            var end = chained ? "hit" : "open"
            var endT: Double? = chained ? nxt!.t : nil
            var restXY: SIMD2<Double>? = nil
            if !chained {
                let rest = eng.restStart(t, xy, w, i0, i1, restPx)
                if rest != nil { atGap = false }                // it came to rest before the gap
                if let r = rest, (i0..<max(i0, r)).filter({ w[$0] > 0 }).count >= cfg.minObs {
                    end = "floor"; endT = t[r]
                    let pRest = camera.pixelToFloor(xy[r])
                    restXY = pRest.isFinite ? SIMD2(pRest.x, pRest.y) : nil
                    i1 = r
                }
            }
            if (i0..<max(i0, i1)).filter({ w[$0] > 0 }).count < cfg.minObs { continue }
            pieces.append(Piece(hit: k, i0: i0, i1: i1, end: end, endT: endT, nextHit: chained ? k + 1 : nil,
                                restXY: restXY, atGap: atGap && end == "open", gapStart: gapStart))
        }

        func priorFor(_ pc: Piece, _ tC: Double, _ withFeet: Bool) -> SegmentPrior {
            let hit = hs[pc.hit]
            let side: CourtSide? = (hit.hitter == .near || hit.hitter == .far) ? hit.hitter : nil
            // contact inside unseen time: no contact epoch (p0/v0 at the first sighting,
            // never back-extrapolated across the gap)
            var pr = SegmentPrior(startT: pc.gapStart ? nil : tC, expectedSide: side)
            if pc.end == "hit" {
                pr.end = .hit; pr.endT = pc.endT
            } else if pc.end == "floor" {
                pr.end = .floor; pr.endT = pc.endT
                pr.restXYWorld = pc.restXY
            }
            pr.endAtGap = pc.atGap
            if withFeet {
                if !pc.gapStart, let f = feet.feet(hit.frame, hit.hitter, hit.playerTrackID, hit.xyPx) {
                    pr.startXYWorld = f; pr.startXYSigma = cfg.feetSigmaM
                }
                if let nh = pc.nextHit.map({ hs[$0] }),
                   let g = feet.feet(nh.frame, nh.hitter, nh.playerTrackID, nh.xyPx) {
                    pr.endXYWorld = g; pr.endXYSigma = cfg.feetSigmaRecvM
                }
            }
            return pr
        }

        // pass 1: no feet priors; contact time = detected hit time (pieces are independent)
        for p in pieces.indices { pieces[p].tContact = hs[pieces[p].hit].t }
        let snap1 = pieces
        let pass1 = parallelMap(snap1.count, maxConcurrency: cfg.maxConcurrency) { p in
            eng.fitPiece(snap1[p].i0, snap1[p].i1, t, xy, w, frames, priorFor(snap1[p], snap1[p].tContact, false), nil)
        }
        for p in pieces.indices { pieces[p].seg = pass1[p] }

        // contact-time refinement where an incoming and an outgoing flight meet
        if pieces.count > 1 {
            for p in 0..<(pieces.count - 1) {
                let a = pieces[p], b = pieces[p + 1]
                guard a.nextHit == b.hit, !b.gapStart, let sa = a.seg, let sb = b.seg else { continue }
                guard let ia = (a.i0..<a.i1).last(where: { w[$0] > 0 }),
                      let ib = (b.i0..<b.i1).first(where: { w[$0] > 0 }) else { continue }
                if let ca = closestApproach(sa, sb, t[ia] - 0.005, t[ib] + 0.005), ca.d < 0.6 {
                    pieces[p + 1].tContact = ca.t
                    pieces[p].endT = ca.t
                }
            }
        }

        // pass 2: refined contact times (+ feet priors) warm-started from pass 1
        let useFeet = !feet.frames.isEmpty
        let frozen = pieces
        let pass2 = parallelMap(frozen.count, maxConcurrency: cfg.maxConcurrency) { p -> ReconstructedSegment? in
            let pc = frozen[p]
            let hit = hs[pc.hit]
            let moved = abs(pc.tContact - hit.t) > 1e-4 || (pc.end == "hit" && pc.endT != hs[pc.nextHit!].t)
            if var seg = pc.seg, useFeet || moved {
                let pr = priorFor(pc, pc.tContact, useFeet)
                if let seg2 = eng.fitPiece(pc.i0, pc.i1, t, xy, w, frames, pr, (seg.thetaRef, seg.tRef)) {
                    return seg2
                }
                if !pc.gapStart { eng.reepoch(&seg, pc.tContact) }   // never re-epoch across a gap
                return seg
            } else if pc.seg == nil && useFeet {
                return eng.fitPiece(pc.i0, pc.i1, t, xy, w, frames, priorFor(pc, pc.tContact, true), nil)
            }
            return pc.seg
        }
        var out: [ReconstructedSegment] = []
        for p in pieces.indices {
            let pc = pieces[p]
            let hit = hs[pc.hit]
            guard var seg = pass2[p] else { continue }
            if pc.atGap {                                // end unseen: only a well-posed speed survives
                guard let sg = seg.speed0SigmaKmh, sg.isFinite, !(sg > cfg.gapSpeedRelSigma * seg.segment.speed0Kmh)
                else { continue }
            }
            seg.hitFrame = hit.frame
            seg.hitTRefined = pc.tContact
            seg.segment.hitterFeetXY = feet.feet(hit.frame, hit.hitter, hit.playerTrackID, hit.xyPx)
            seg.restXY = pc.restXY
            seg.endsAtPTSGap = pc.atGap                 // end unseen: a clock hole follows
            seg.startsAfterPTSGap = pc.gapStart         // contact unseen: p0/v0 at first sighting
            if pc.end == "hit" { seg.segment.endReason = .hit }
            out.append(seg)
        }
        return out
    }

    /// Contract-only view of `reconstructSegments` (`[ShotSegment3D]`).
    public static func reconstructShotSegments(track: Track2D, hits: [HitEvent], camera: CameraModel?,
                                               meta: VideoMeta?, players: [PlayerObservation]? = nil,
                                               config: ReconstructConfig = .init()) -> [ShotSegment3D] {
        reconstructSegments(track: track, hits: hits, camera: camera, meta: meta, players: players,
                            config: config).map(\.segment)
    }
}
