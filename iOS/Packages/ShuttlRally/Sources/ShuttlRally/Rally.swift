import Foundation
import ShuttlCore

// Port of backend/app/services/tracking/rally.py (segment_rallies and helpers).
// See the Python module docstring for the method; this file follows it
// statement by statement (same constants, thresholds, tie-breaking and
// evaluation order of floating-point expressions).

private let SIDES: [CourtSide] = [.near, .far]

// MARK: - court rules

/// rally.service_court_margin: signed distance (m, + = inside) of a serve
/// landing from the service court diagonal to the server.
public func serviceCourtMargin(x: Double, y: Double, server: CourtSide, serverY: Double?,
                               doubles: Bool) -> Double {
    let s = server == .near ? 1.0 : -1.0
    let lw = CourtModel.lineWidth / 2
    let xr = s * x
    let back = (doubles ? CourtModel.doublesLongServiceX : CourtModel.halfLength) + lw
    let halfW = (doubles ? CourtModel.doublesHalfWidth : CourtModel.singlesHalfWidth) + lw
    var m = xr - (CourtModel.shortServiceX - lw)          // Python min(a, b, c)
    if back - xr < m { m = back - xr }
    if halfW - Swift.abs(y) < m { m = halfW - Swift.abs(y) }
    if let serverY, Swift.abs(serverY) > 0.15 {
        let ty = -(serverY.sign == .minus ? -1.0 : 1.0)    // -copysign(1, server_y)
        let c = ty * y + lw
        if c < m { m = c }
    }
    return m
}

/// rally.detect_doubles: >= 2 players per side together in >= min_frac of the
/// frames that have players; nil without player observations.
public func detectDoubles(_ players: [PlayerObservation]?, minFrac: Double = 0.3) -> Bool? {
    guard let players, !players.isEmpty else { return nil }
    var per: [Int: (near: Int, far: Int)] = [:]
    for p in players {
        if p.score < 0.3 || p.side == .unknown { continue }
        var d = per[p.frame] ?? (0, 0)
        if p.side == .near { d.near += 1 } else { d.far += 1 }
        per[p.frame] = d
    }
    if per.isEmpty { return nil }
    let four = per.values.filter { $0.near >= 2 && $0.far >= 2 }.count
    return Double(four) >= minFrac * Double(per.count)
}

// MARK: - drag ODE (pure scalars, like the Python float code)

struct S6 {
    var x, y, z, vx, vy, vz: Double
    init(_ x: Double, _ y: Double, _ z: Double, _ vx: Double, _ vy: Double, _ vz: Double) {
        self.x = x; self.y = y; self.z = z; self.vx = vx; self.vy = vy; self.vz = vz
    }
    init(p: SIMD3<Double>, v: SIMD3<Double>) { self.init(p.x, p.y, p.z, v.x, v.y, v.z) }
    var allFinite: Bool { x.isFinite && y.isFinite && z.isFinite && vx.isFinite && vy.isFinite && vz.isFinite }
}

/// rally._rk4: one RK4 step of the physics.py drag ODE on [x, y, z, vx, vy, vz].
@inline(__always)
func rallyRK4(_ s: S6, _ h: Double, _ L: Double) -> S6 {
    @inline(__always) func f(_ q: S6) -> S6 {
        let k = (q.vx * q.vx + q.vy * q.vy + q.vz * q.vz).squareRoot() / L
        return S6(q.vx, q.vy, q.vz, -k * q.vx, -k * q.vy, -Physics.g - k * q.vz)
    }
    @inline(__always) func add(_ a: S6, _ c: Double, _ b: S6) -> S6 {
        S6(a.x + c * b.x, a.y + c * b.y, a.z + c * b.z, a.vx + c * b.vx, a.vy + c * b.vy, a.vz + c * b.vz)
    }
    let k1 = f(s)
    let k2 = f(add(s, 0.5 * h, k1))
    let k3 = f(add(s, 0.5 * h, k2))
    let k4 = f(add(s, h, k3))
    let w = h / 6
    @inline(__always) func comb(_ a: Double, _ b: Double, _ c: Double, _ d: Double, _ e: Double) -> Double {
        a + w * (b + 2 * c + 2 * d + e)
    }
    return S6(comb(s.x, k1.x, k2.x, k3.x, k4.x), comb(s.y, k1.y, k2.y, k3.y, k4.y),
              comb(s.z, k1.z, k2.z, k3.z, k4.z), comb(s.vx, k1.vx, k2.vx, k3.vx, k4.vx),
              comb(s.vy, k1.vy, k2.vy, k3.vy, k4.vy), comb(s.vz, k1.vz, k2.vz, k3.vz, k4.vz))
}

/// `float(seg.drag_length or DRAG_LENGTH)`.
@inline(__always)
private func dragL(_ seg: ShotSegment3D) -> Double { seg.dragLength != 0 ? seg.dragLength : Physics.dragLength }

/// rally._floor_time: absolute time the fitted flight reaches the floor, if before t_until.
func floorTime(_ seg: ShotSegment3D, _ tUntil: Double, h: Double = 1.0 / 120) -> Double? {
    if let lt = seg.landingT { return lt }
    var s = S6(p: seg.p0, v: seg.v0)
    if !s.allFinite || s.z <= 0.0 { return nil }
    let L = dragL(seg)
    var t = 0.0
    let stop = tUntil - seg.t0
    while t < stop {
        let n = rallyRK4(s, h, L)
        if n.z <= 0.0 { return seg.t0 + t + h * s.z / (s.z - n.z) }
        s = n
        t += h
    }
    return nil
}

/// rally._flight_end: (landing_xy, landing_t, extrapolated s, net height, net y).
func flightEndODE(_ seg: ShotSegment3D, _ tLast: Double, h: Double = 1.0 / 240, tMax: Double = 6.0)
    -> (land: SIMD2<Double>?, landT: Double?, extra: Double, netH: Double?, netY: Double?) {
    let L = dragL(seg)
    var landXY = seg.landingXY, landT = seg.landingT, netH = seg.netCrossingHeight
    var netY = seg.netCrossingY
    if landXY != nil && (netH == nil || netY != nil) { return (landXY, landT, 0.0, netH, netY) }
    var s = S6(p: seg.p0, v: seg.v0)
    var t = 0.0
    let tStop = landT != nil ? landT! - seg.t0 : tMax
    var extra = 0.0
    while t < tStop {
        let n = rallyRK4(s, h, L)
        if s.x * n.x < 0 && netY == nil {
            let a = s.x / (s.x - n.x)
            netY = s.y + a * (n.y - s.y)
            if netH == nil { netH = s.z + a * (n.z - s.z) }
            if landXY != nil { break }
        }
        if n.z <= 0.0 && s.z > 0.0 {
            if landXY == nil {
                let a = s.z / (s.z - n.z)
                landXY = SIMD2(s.x + a * (n.x - s.x), s.y + a * (n.y - s.y))
                let lt = seg.t0 + t + a * h
                landT = lt
                extra = Swift.max(0.0, lt - tLast)
            }
            break
        }
        s = n
        t += h
    }
    return (landXY, landT, extra, netH, netY)
}

/// rally._state_at: state of the fitted flight at absolute time t (drag ODE
/// from p0/v0 in steps of h, the last one shortened to end exactly at t).
func stateAt(_ seg: ShotSegment3D, _ t: Double, h: Double = 1.0 / 60) -> S6 {
    var s = S6(p: seg.p0, v: seg.v0)
    let L = dragL(seg)
    var tt = 0.0
    let stop = t - seg.t0
    while stop - tt > 1e-9 {
        let step = Swift.min(h, stop - tt)
        s = rallyRK4(s, step, L)
        tt += step
    }
    return s
}

/// rally._merge_same_flight: drop contacts where the 3-D velocity barely
/// changes (|dv| < min_contact_dv) while the earlier fit still covers them.
/// Returns kept hits, their segment indices and per hit the continuation
/// segment index (nil = its own segment describes the end of its flight).
func mergeSameFlight(_ hits: [HitEvent], _ segs: [Int?], _ segments: [ShotSegment3D], _ t: [Double],
                     _ cfg: RallyConfig) -> (hits: [HitEvent], segs: [Int?], conts: [Int?]) {
    var oh: [HitEvent] = [], os: [Int?] = [], oc: [Int?] = []
    let n = t.count
    for (h, si) in zip(hits, segs) {
        if !oh.isEmpty, let si {
            let s = segments[si]
            if let pi = oc.last! ?? os.last! {
                let prev = segments[pi]
                let tCov = t[Swift.min(Swift.max(prev.endFrame, 0), n - 1)] + 0.25
                let t0 = s.t0
                let vxn = s.v0.x
                // exact shortcut (drag never flips vx): see rally._merge_same_flight
                let reversedX = prev.v0.x * vxn < 0 && Swift.abs(vxn) >= cfg.minContactDv
                // exact bound: under drag |v| grows at most by g*dt (see Python)
                let dt0 = t0 - prev.t0
                let spN = (s.v0.x * s.v0.x + s.v0.y * s.v0.y + s.v0.z * s.v0.z).squareRoot()
                let spP = (prev.v0.x * prev.v0.x + prev.v0.y * prev.v0.y + prev.v0.z * prev.v0.z).squareRoot()
                let faster = spN >= spP + Physics.g * dt0 + cfg.minContactDv + 0.5
                if !reversedX && !faster && prev.t0 < t0 && t0 <= tCov {
                    let st = stateAt(prev, t0)
                    let dx = st.vx - s.v0.x, dy = st.vy - s.v0.y, dz = st.vz - s.v0.z
                    if (dx * dx + dy * dy + dz * dz).squareRoot() < cfg.minContactDv {
                        oc[oc.count - 1] = si
                        continue
                    }
                }
            }
        }
        oh.append(h); os.append(si); oc.append(nil)
    }
    return (oh, os, oc)
}

// MARK: - outcome

struct Outcome {
    var endReason: RallyEndReason = .unknown
    var winner: CourtSide = .unknown
    var landingXY: SIMD2<Double>? = nil
    var margin: Double? = nil
    var confidence = 0.0
    var endT: Double? = nil
    var source = "none"
}

/// rally._landing_sigma. `sigma` is the segment's landing_sigma_m extra
/// (reconstruct3d's landing covariance), which ShotSegment3D does not carry:
/// it arrives through segmentRallies(landingSigmaM:).
private func landingSigma(_ sigma: Double?, _ extraS: Double, _ cfg: RallyConfig) -> Double {
    let s = (sigma == nil || !sigma!.isFinite) ? cfg.defaultLandingSigmaM : sigma!
    return Swift.max(s, 0.03) + 0.6 * extraS
}

/// rally._landing_plausible: floor point within landingOutsideMaxM of the doubles court.
func landingPlausible(_ x: Double, _ y: Double, _ cfg: RallyConfig) -> Bool {
    x.isFinite && y.isFinite && Swift.abs(x) <= CourtModel.halfLength + cfg.landingOutsideMaxM
        && Swift.abs(y) <= CourtModel.doublesHalfWidth + cfg.landingOutsideMaxM
}

func outcome3D(hit: HitEvent, seg: ShotSegment3D, sigma: Double?, tLast: Double, isServe: Bool,
               doubles: Bool?, serverY: Double?, cfg: RallyConfig, launch: ShotSegment3D? = nil,
               sideP: Double? = nil) -> Outcome {
    let ls = launch ?? seg
    let hitter = hit.hitter != .unknown ? hit.hitter : CourtModel.sideOf(x: ls.p0.x)
    var (landXY, landT, extra, netH, netY) = flightEndODE(seg, tLast)
    if let l = landXY, !landingPlausible(l.x, l.y, cfg) {
        // the fitted flight comes down outside the hall (F3): no call from its geometry; the rally
        // keeps its span, exactly what ShuttlAnalysis LandingGuard does to such a call
        return Outcome(endT: landT, source: "implausible_landing")
    }
    var segNet = seg.endReason == .net
    if let launch {
        segNet = false
        if let lh = launch.netCrossingHeight {
            netH = lh
            netY = launch.netCrossingY
        }
    }
    if let ny = netY, Swift.abs(ny) > CourtModel.doublesHalfWidth + 0.05 { netH = nil }
    var fitQ = 0.6 + 0.4 * Swift.min(Swift.max(seg.confidence, 0.0), 1.0)
    let vx0 = ls.v0.x
    if let sideP {
        fitQ *= sideP
    } else if hit.hitter != .unknown && Swift.abs(vx0) > 2.0 && (vx0 > 0) != (hitter == .near) {
        fitQ *= cfg.sideConflictQ
    }
    var out = Outcome(endT: landT, source: "3d")
    let netLine = CourtModel.netHeightCentre - cfg.netMarginM
    let fitLand = seg.landingXY
    let wentOver = fitLand != nil && fitLand!.x != 0 && CourtModel.sideOf(x: fitLand!.x) != hitter
    var netCall = segNet
    if !netCall, let nh = netH, nh < netLine, !(wentOver && nh > netLine - cfg.netCordTolM) { netCall = true }
    if netCall {
        out.endReason = .net; out.winner = hitter.opposite
        out.landingXY = landXY
        out.confidence = fitQ * cfg.netCallQ * (netH != nil ? Py.phi((netLine - netH!) / 0.08) : 0.8)
        return out
    }
    guard let land = landXY else { return out }
    out.landingXY = land
    if land.x != 0 && CourtModel.sideOf(x: land.x) == hitter {
        out.endReason = .net; out.winner = hitter.opposite
        out.confidence = fitQ * cfg.netCallQ * Py.phi(Swift.abs(land.x) / 0.3)
        return out
    }
    let sig = landingSigma(sigma, extra, cfg)
    let x = land.x, y = land.y
    let margin: Double, ambiguous: Bool
    if isServe {
        margin = serviceCourtMargin(x: x, y: y, server: hitter, serverY: serverY, doubles: doubles ?? false)
        ambiguous = doubles == nil && CourtModel.doublesLongServiceX < Swift.abs(x) && Swift.abs(x) < CourtModel.halfLength
    } else {
        margin = CourtModel.inOutMargin(x: x, y: y, doubles: doubles ?? false)
        ambiguous = doubles == nil && CourtModel.singlesHalfWidth < Swift.abs(y) && Swift.abs(y) < CourtModel.doublesHalfWidth
    }
    out.margin = margin
    out.endReason = margin >= 0 ? .inCourt : .out
    out.winner = margin >= 0 ? hitter : hitter.opposite
    out.confidence = fitQ * Py.phi(Swift.abs(margin) / sig) * (ambiguous ? 0.5 : 1.0)
    return out
}

func outcome2D(hit: HitEvent, fRest: Int?, mo: Motion, camera: CameraModel?, doubles: Bool?,
               cfg: RallyConfig) -> Outcome {
    var out = Outcome(source: "2d")
    if let fRest { out.endT = mo.t[fRest] }
    guard let camera, let fRest, hit.hitter != .unknown else { return out }
    let p = camera.pixelToFloor(mo.xy[fRest])
    guard p.x.isFinite && p.y.isFinite else { return out }
    let x = p.x, y = p.y
    if !landingPlausible(x, y, cfg) { return out }       // not on the floor near the court
    out.landingXY = SIMD2(x, y)
    if CourtModel.sideOf(x: x) == hit.hitter {
        out.endReason = .net; out.winner = hit.hitter.opposite; out.confidence = 0.3
        return out
    }
    let margin = CourtModel.inOutMargin(x: x, y: y, doubles: doubles ?? false)
    out.margin = margin
    out.endReason = margin >= 0 ? .inCourt : .out
    out.winner = margin >= 0 ? hit.hitter : hit.hitter.opposite
    out.confidence = 0.4 * Py.phi(Swift.abs(margin) / 0.3)
    return out
}

// MARK: - candidates

enum Cue: String { case gap, landing, `static` }

private func cueScore(_ c: Cue?) -> Double {
    switch c {
    case .landing?, .static?: return 1.0
    case .gap?: return 0.8
    case nil: return 0.7
    }
}

struct Cand {
    var idx: [Int]
    var hits: [HitEvent]
    var segs: [Int?]                 // indices into `segments`
    var startFrame: Int
    var endFrame: Int
    var endT: Double
    var outcome: Outcome
    var segConf: Double
    var cueBefore: Cue?
    var serveMissed: Bool
    var firstSideP: Double = 1.0     // P(first hitter side) from the joint decode
    var heldBefore: Bool = true      // shuttle seen resting right before the first hit

    var server: CourtSide {
        guard let h = hits.first, h.hitter != .unknown else { return .unknown }
        return serveMissed ? h.hitter.opposite : h.hitter
    }
}

/// Everything segment_rallies' helpers share.
private struct Ctx {
    let mo: Motion
    let cfg: RallyConfig
    let segments: [ShotSegment3D]
    let sigmas: [Double?]

    func seg(_ i: Int?) -> ShotSegment3D? { i.map { segments[$0] } }

    /// rally._median_speed.
    func medianSpeed(_ t0: Double, _ t1: Double) -> Double? {
        let a = mo.searchsorted(t0, right: true), b = mo.searchsorted(t1, right: true)
        let v = mo.finiteSpeeds(a, b)
        return v.count >= 3 ? Py.median(v) : nil
    }

    /// rally._is_flight.
    func isFlight(_ h: HitEvent, _ segI: Int?, tNext: Double? = nil) -> Bool {
        if let s = seg(segI), Py.norm3(s.v0) > 4.0 { return true }
        var tEnd = h.t + cfg.launchWindowS
        if let tNext { tEnd = Swift.min(tEnd, tNext - 1e-3) }
        guard let after = medianSpeed(h.t, tEnd) else {
            let a = mo.frameAt(h.t - 0.5)
            var seenBefore = false
            if h.frame > a { for f in a..<h.frame where mo.fin[f] { seenBefore = true; break } }
            let seenAfter = medianSpeed(h.t, tNext == nil ? h.t + 0.8 : Swift.min(h.t + 0.8, tNext! - 1e-3)) != nil
            return seenAfter || !seenBefore
        }
        if after >= cfg.minFlightSpeedHps { return true }
        let before = medianSpeed(h.t - 0.4, h.t - 0.05)
        return after >= cfg.minLaunchSpeedHps && (before == nil || after >= 2.5 * before!)
    }

    /// rally._continuous_flight.
    func continuousFlight(_ f0: Int, _ f1: Int) -> Bool {
        if f1 <= f0 + 1 { return true }
        if mo.anyRest(f0, f1 + 1) { return false }
        let hi = Swift.min(f1 + 1, mo.n)
        var tt = [mo.t[f0]]
        for f in f0..<hi where mo.fin[f] { tt.append(mo.t[f]) }
        tt.append(mo.t[f1])
        var mx = -Double.infinity
        for k in 1..<tt.count { let d = tt[k] - tt[k - 1]; if d > mx || d.isNaN { mx = d } }
        if mx > cfg.maxUnseenS { return false }
        let v = mo.finiteSpeeds(f0, f1 + 1)
        if v.count < 3 { return false }
        return Py.npMeanBool(v.map { $0 < cfg.minFlightSpeedHps }) < 0.5
    }

    /// rally._unseen_flight: a long hit gap that is flight out of view
    /// (`fStart` replaces a.frame as the first frame looked at, see pastHold).
    func unseenFlight(_ a: HitEvent, _ b: HitEvent, fStart: Int? = nil) -> Bool {
        let f0 = fStart ?? a.frame, f1 = b.frame
        if f1 <= f0 + 1 || mo.anyRest(f0, f1 + 1) { return false }
        let hi = Swift.min(f1 + 1, mo.n)
        var tt = [mo.t[f0]]
        for f in f0..<hi where mo.fin[f] { tt.append(mo.t[f]) }
        tt.append(mo.t[f1])
        var mx = -Double.infinity
        for k in 1..<tt.count { let d = tt[k] - tt[k - 1]; if d > mx || d.isNaN { mx = d } }
        if mx > cfg.maxUnseenFlightS || mx <= cfg.maxUnseenS { return false }
        let v = mo.finiteSpeeds(f0, f1 + 1)
        if v.count < 3 || Py.npMeanBool(v.map { $0 < cfg.minFlightSpeedHps }) >= 0.5 { return false }
        guard let arr = medianSpeed(b.t - cfg.launchWindowS, b.t - 1e-3) else {
            // arrival unseen: judged by the last sighting before b (see rally.py)
            guard cfg.unseenArrivalOK else { return false }
            let idx = (f0..<Swift.min(f1, mo.n)).filter { mo.fin[$0] }
            if idx.count < 3 { return false }
            // ... and it left the view high up, not low where a dead shuttle falls
            if mo.xy[idx.last!].y > cfg.unseenExitMaxY * mo.h { return false }
            let sp = idx.suffix(3).map { mo.speed[$0] }.filter { $0.isFinite }
            return sp.count >= 2 && Py.median(sp) >= cfg.minFlightSpeedHps
        }
        return arr >= cfg.minFlightSpeedHps
    }

    /// rally._past_hold: first frame after the rest run touching contact frame f0
    /// (a serve's pre-contact hold), only if that run ends within maxS of it.
    func pastHold(_ f0: Int, _ f1: Int, maxS: Double = 0.1) -> Int {
        var f = f0
        while f < f1 && mo.rest[f] && mo.t[f] - mo.t[f0] <= maxS { f += 1 }
        return f
    }

    /// rally._launch_frame: last frame of a rest run touching contact f0 (a
    /// serve's own hold, see pastHold), else f0 (skipServeHold = false: f0).
    func launchFrame(_ f0: Int, _ f1: Int) -> Int {
        cfg.skipServeHold ? Swift.max(f0, pastHold(f0, f1) - 1) : f0
    }

    /// rally._moving_after: observed moving time after contact h (past its own
    /// hold) until the next rest / horizon.
    func movingAfter(_ h: HitEvent, horizonS: Double = 4.0) -> Double {
        let fHi = Swift.min(mo.n, mo.frameAt(h.t + horizonS) + 1)
        let f = launchFrame(h.frame, fHi)
        let end = mo.firstRestAfter(f, fHi) ?? fHi
        if end <= f + 1 { return 0.0 }
        var k = 0
        for i in (f + 1)..<end where mo.fin[i] && !mo.rest[i] { k += 1 }
        return Double(k) * mo.dt
    }

    /// rally._hold_before: +1 rest right before the hit, -1 visible and moving, 0 unobserved.
    func holdBefore(_ f: Int, _ t: Double) -> Double {
        let f0 = mo.frameAt(t - cfg.holdWindowS)
        if f <= f0 { return 0.0 }
        if mo.anyRest(f0, f) { return 1.0 }
        let hi = Swift.min(f, mo.n)
        let finFrac = Py.npMeanBool(Array(mo.fin[f0..<hi]))
        return finFrac >= 0.5 ? -1.0 : 0.0
    }

    /// rally._missed_serve_start.
    func missedServeStart(_ f: Int, _ t: Double, fMin: Int = 0) -> Int? {
        let fLo = Swift.max(fMin, mo.frameAt(t - 3.0))
        var a = fLo
        if f > fLo { for k in stride(from: f - 1, through: fLo, by: -1) where mo.rest[k] { a = k + 1; break } }
        var ok: [Int] = []
        if f > a { for k in a..<f where mo.fin[k] { ok.append(k) } }
        if ok.count < 3 { return nil }
        var lastGap: Int? = nil
        for k in 0..<(ok.count - 1) where mo.t[ok[k + 1]] - mo.t[ok[k]] > 0.4 { lastGap = k }
        if let g = lastGap { ok = Array(ok[(g + 1)...]) }
        if ok.count < 3 || mo.t[f] - mo.t[ok[0]] < 0.3 { return nil }
        let v = mo.finiteSpeeds(ok[0] + 1, f)
        if v.count < 3 || Py.npMeanBool(v.map { $0 < cfg.minFlightSpeedHps }) >= 0.3 { return nil }
        return ok[0]
    }

    /// rally._extend_flight.
    func extendFlight(_ f0: Int, _ fEnd: Int, _ fLimit: Int) -> Int {
        let f1 = Swift.min(fLimit, mo.frameAt(mo.t[f0] + 4.0))
        let lo = f0 + 1, hi = Swift.min(f1 + 1, mo.n)
        var obs: [Int] = []
        var firstRest: Int? = nil
        if hi > lo {
            for f in lo..<hi {
                if mo.fin[f] && !mo.rest[f] { obs.append(f) }
                if mo.rest[f] && firstRest == nil { firstRest = f }
            }
        }
        if obs.isEmpty || obs.last! <= fEnd { return fEnd }
        if let r = firstRest { obs = obs.filter { $0 < r } }
        var tt = [mo.t[f0]]
        tt.append(contentsOf: obs.map { mo.t[$0] })
        for k in 0..<(tt.count - 1) where tt[k + 1] - tt[k] > cfg.maxUnseenS { obs = Array(obs[..<k]); break }
        if obs.isEmpty || obs.last! <= fEnd { return fEnd }
        return continuousFlight(f0, obs.last!) ? obs.last! : fEnd
    }

    /// rally._leading_nonhit.
    func leadingNonhit(_ g: [Int], _ hs: [HitEvent], _ segs: [Int?], _ have3D: Bool) -> Bool {
        let h0 = hs[g[0]]
        let h1: HitEvent? = g.count > 1 ? hs[g[1]] : nil
        if !isFlight(h0, segs[g[0]], tNext: h1?.t) { return true }
        guard let h1, !(h1.t - h0.t > 0.6) else { return false }
        let c0 = h0.confidence, c1 = h1.confidence
        if h0.hitter != .unknown && h0.hitter == h1.hitter && c0 <= c1 { return true }
        return have3D && segs[g[0]] == nil && c0 < 0.4 && 0.4 <= c1
    }

    /// rally._group_hits.
    func groupHits(_ hits: [HitEvent], _ segs: [Int?], _ conts: [Int?]? = nil) -> (groups: [[Int]], cues: [Cue]) {
        guard !hits.isEmpty else { return ([], []) }
        var groups: [[Int]] = [], cues: [Cue] = [], cur = [0]
        for i in 1..<hits.count {
            let a = hits[i - 1], b = hits[i]
            var cue: Cue? = nil
            let gap = b.t - a.t
            // a serve's pre-contact hold touching a's contact frame is not dead time after a
            let fa = cfg.skipServeHold ? pastHold(a.frame, b.frame) : a.frame
            if gap > cfg.maxHitGapS && !(gap < cfg.maxHitGapMissedS && continuousFlight(fa, b.frame))
                && !(gap < cfg.maxHitGapUnseenS && unseenFlight(a, b, fStart: fa)) {
                cue = .gap
            } else {
                var ft: Double? = nil
                let si: Int? = (conts != nil && conts![i - 1] != nil) ? conts![i - 1] : segs[i - 1]
                if let s = seg(si), s.confidence >= cfg.minLandingConf {
                    let tCov = mo.t[Swift.min(Swift.max(s.endFrame, 0), mo.n - 1)] + 0.25
                    ft = floorTime(s, Swift.min(b.t - cfg.landingSlackS, tCov))
                }
                if let ft, ft < b.t - cfg.landingSlackS, !continuousFlight(mo.frameAt(ft), b.frame) {
                    cue = .landing
                } else if mo.staticBetween(a.frame, b.frame) {
                    cue = .static
                }
            }
            if let cue {
                groups.append(cur); cues.append(cue); cur = [i]
            } else {
                cur.append(i)
            }
        }
        groups.append(cur)
        return (groups, cues)
    }

    /// rally._nonrally_log_odds: max(feed-vs-rally, spurious-vs-rally) of a one-hit group.
    func nonrallyLogOdds(_ c: Cand, _ prev: Cand?, _ nxt: Cand?, _ doubles: Bool?, _ have3D: Bool) -> Double {
        let h = c.hits[0], sg = seg(c.segs[0])
        let hold = holdBefore(h.frame, h.t)
        let conf = h.confidence
        let prevW = prev?.outcome.winner ?? .unknown
        let rule: Double = (h.hitter == .unknown || prevW == .unknown) ? 0 : (h.hitter != prevW ? 1 : -1)
        var wRule = 0.0
        if let prev {
            let r = (prev.outcome.confidence - 0.45) / 0.35
            wRule = 3.0 * Swift.min(Swift.max(r, 0.0), 1.0)
        }
        var feed = -1.0 + wRule * rule
        if let nxt, h.hitter != .unknown, nxt.hits[0].hitter != .unknown, nxt.hits[0].t - h.t < 15.0 {
            feed += nxt.hits[0].hitter != h.hitter ? 0.8 : -1.5
        }
        if let prev {
            let dtp = h.t - prev.endT
            feed += dtp < 4.0 ? 0.8 : (dtp > 8.0 ? -0.8 : 0.0)
        }
        feed += hold == 1.0 ? -1.5 : (hold == -1.0 ? 1.0 : 0.0)
        if let sg {
            let sp = Py.norm3(sg.v0) * 3.6
            feed += sp < 45 ? 0.3 : (sp > 90 ? -1.0 : 0.0)
            let caught = sg.landingXY == nil && c.outcome.endReason != .net
            feed += caught ? 1.0 : -0.3
            let xr = Swift.abs(sg.p0.x), y = sg.p0.y
            let back = doubles == true ? CourtModel.doublesLongServiceX : CourtModel.halfLength
            if xr < CourtModel.shortServiceX - 0.4 || xr > back + 0.6 || Swift.abs(y) > CourtModel.doublesHalfWidth + 0.6 {
                feed += 0.7
            }
        }
        var spur = -1.5 + 0.6 * rule
        spur += conf < 0.4 ? 1.2 : (conf > 0.7 ? -1.0 : 0.0)
        spur += hold == 1.0 ? -1.5 : (hold == -1.0 ? 1.0 : 0.0)
        if have3D { spur += sg == nil ? 1.2 : -1.5 }
        if let pk = mo.peakSpeed(h.frame, cfg.flightWindowS) {
            if pk < 2 * cfg.minFlightSpeedHps { spur += 0.5 }
        } else {
            spur += 1.0
        }
        return spur > feed ? spur : feed                 // Python max(feed, spur)
    }
}

/// rally._serve_launch_ok: the first contact can be the serve (no 3-D fit, or
/// launched on the server's half inside [short service line - 0.3, baseline + 0.8]).
private func serveLaunchOK(_ c: Cand, _ segments: [ShotSegment3D]) -> Bool {
    guard !c.serveMissed, let f = c.segs.first, let si = f else { return true }
    let x = segments[si].p0.x
    return CourtModel.sideOf(x: x) == c.server
        && CourtModel.shortServiceX - 0.3 < Swift.abs(x) && Swift.abs(x) < CourtModel.halfLength + 0.8
}

/// rally._infer_from_next_serve: resolve unknown / low-confidence winners from who serves next.
private func inferFromNextServe(_ cands: inout [Cand], _ cfg: RallyConfig, _ segments: [ShotSegment3D],
                                firstServer: CourtSide = .near) {
    var st = BWFScore(firstServer: firstServer)
    for i in cands.indices {
        let w = cands[i].outcome.winner
        var ns: CourtSide = .unknown
        var seen = false
        if i + 1 < cands.count {
            let nxt = cands[i + 1]
            if let h0 = nxt.hits.first, nxt.server != .unknown, h0.confidence >= 0.3, nxt.segConf >= 0.5,
               serveLaunchOK(nxt, segments),
               !cfg.nextServeNeedsHold || nxt.serveMissed || nxt.heldBefore {
                ns = nxt.server
                seen = nxt.heldBefore && !nxt.serveMissed          // the serve itself was observed
            }
        }
        let implied = impliedWinner(st, ns)
        let q0 = cands[i].outcome.confidence
        if implied != .unknown && (w == .unknown || (w != implied && cands[i].outcome.confidence < cfg.overrideConf)) {
            if w != .unknown && w != implied { cands[i].outcome.endReason = .unknown }
            cands[i].outcome.winner = implied
            let q = cfg.nextServeSideQ ? cands[i + 1].firstSideP : 1.0
            cands[i].outcome.confidence = 0.8 * cands[i].segConf * q
            cands[i].outcome.source = "next_serve"
            // a verdict from a serve nobody saw (inferred from a missed serve) is not certain
            if !seen { cands[i].outcome.confidence = Swift.min(cands[i].outcome.confidence, cfg.nsConflictConf) }
        } else if implied != .unknown && w != .unknown && w != implied && q0 < cfg.nsConflictMaxQ {
            // the call is kept but the next serve says the other side won it: not certain either
            cands[i].outcome.confidence = Swift.min(cands[i].outcome.confidence, cfg.nsConflictConf)
        }
        if cands[i].outcome.winner != .unknown { _ = st.point(cands[i].outcome.winner) }
    }
}

private func makeShot(_ h: HitEvent, _ seg: ShotSegment3D?, first: Bool) -> ShotEventV2 {
    ShotEventV2(hit: h, segment: seg, shotType: first ? .serve : .unknown,
                shotTypeConfidence: first ? 0.3 : 0.0,
                hitterPosWorld: seg.map { SIMD2($0.p0.x, $0.p0.y) }, hitHeightM: seg?.p0.z)
}

private func toRally(_ c: Cand, _ segments: [ShotSegment3D]) -> RallyV2 {
    let o = c.outcome
    let known = o.winner != .unknown
    let conf = c.segConf * (known ? o.confidence : 1.0)
    let shots = c.hits.indices.map { k in
        makeShot(c.hits[k], c.segs[k].map { segments[$0] }, first: k == 0 && !c.serveMissed)
    }
    return RallyV2(startFrame: c.startFrame, endFrame: c.endFrame, shots: shots, endReason: o.endReason,
                   landingXY: o.landingXY, inOutMarginM: o.margin, winner: o.winner,
                   confidence: Swift.min(Swift.max(conf, 0.0), 1.0), server: c.server,
                   serveMissed: c.serveMissed, segmentationConfidence: c.segConf,
                   outcomeConfidence: o.confidence, outcomeSource: o.source)
}

/// rally._activity_rallies: no hits at all → contiguous moving stretches of the track.
private func activityRallies(_ mo: Motion, _ cfg: RallyConfig) -> [RallyV2] {
    let idx = (0..<mo.n).filter { f in
        let sp = mo.speed[f].isNaN ? 0.0 : mo.speed[f]          // np.nan_to_num
        return mo.fin[f] && !mo.rest[f] && sp >= cfg.minFlightSpeedHps
    }
    guard var s = idx.first else { return [] }
    var runs: [(Int, Int)] = []
    var p = s
    for f in idx.dropFirst() {
        if mo.t[f] - mo.t[p] > cfg.activityGapS { runs.append((s, p)); s = f }
        p = f
    }
    runs.append((s, p))
    return runs.filter { mo.t[$0.1] - mo.t[$0.0] >= cfg.activityMinS }
        .map { RallyV2(startFrame: $0.0, endFrame: $0.1, confidence: 0.3) }
}

/// rally._clean_hits: sorted, one hit per frame (highest confidence), frames inside the track.
private func cleanHits(_ hits: [HitEvent], _ n: Int) -> [HitEvent] {
    var best: [Int: HitEvent] = [:]
    for h in hits where 0 <= h.frame && h.frame < n {
        if let b = best[h.frame], !(h.confidence > b.confidence) { continue }
        best[h.frame] = h
    }
    return best.keys.sorted().map { best[$0]! }
}

/// `np.searchsorted` on an ascending Int array.
private func searchsortedInt(_ a: [Int], _ v: Int, right: Bool) -> Int {
    var lo = 0, hi = a.count
    while lo < hi {
        let mid = (lo + hi) / 2
        if right ? (a[mid] <= v) : (a[mid] < v) { lo = mid + 1 } else { hi = mid }
    }
    return lo
}

/// rally._match_segments: index of the flight starting at each hit.
private func matchSegments(_ hits: [HitEvent], _ segments: [ShotSegment3D], _ t: [Double],
                           _ cfg: RallyConfig) -> [Int?] {
    var out = [Int?](repeating: nil, count: hits.count)
    if hits.isEmpty || segments.isEmpty { return out }
    let frames = hits.map(\.frame)
    var bestD = [Int](repeating: Int.max, count: hits.count)
    let n = t.count
    for (si, seg) in segments.enumerated() {
        let s = seg.startFrame
        let k = searchsortedInt(frames, s + cfg.segMatchBefore, right: true) - 1
        if k < 0 { continue }
        let f = frames[k]
        let nxt = k + 1 < frames.count ? frames[k + 1] : n
        let tS = t[Swift.min(Swift.max(s, 0), n - 1)]
        if s >= nxt || tS - t[f] > cfg.segMatchAfterS { continue }
        let d = Swift.abs(s - f)
        if d < bestD[k] { out[k] = si; bestD[k] = d }
    }
    return out
}

/// rally._fill_hitters.
private func fillHitters(_ hits: [HitEvent], _ segs: [Int?], _ segments: [ShotSegment3D],
                         _ groups: [[Int]], sureX: Double = 1.5) -> [HitEvent] {
    var out = hits
    for i in out.indices {
        guard let si = segs[i] else { continue }
        let x = segments[si].p0.x
        let side = CourtModel.sideOf(x: x)
        if out[i].hitter == .unknown || (Swift.abs(x) > sureX && side != out[i].hitter) { out[i].hitter = side }
    }
    for g in groups {
        let known = g.enumerated().filter { out[$0.element].hitter != .unknown }
        guard let first = known.first else { continue }
        for (pos, i) in g.enumerated() where out[i].hitter == .unknown {
            var best = first, bd = Swift.abs(first.offset - pos)
            for kk in known.dropFirst() {
                let d = Swift.abs(kk.offset - pos)
                if d < bd { best = kk; bd = d }
            }
            let hs = out[best.element].hitter
            out[i].hitter = bd % 2 == 0 ? hs : hs.opposite
        }
    }
    return out
}

// MARK: - joint hitter-side decode (rally._decode_sides)

/// rally._logaddexp: log(e^a + e^b) on plain doubles (same formula as Python).
@inline(__always)
func logAddExp(_ a0: Double, _ b0: Double) -> Double {
    var a = a0, b = b0
    if a < b { swap(&a, &b) }
    if b == -Double.infinity { return a }
    return a + Foundation.log1p(Foundation.exp(b - a))
}

enum SideReading: Equatable { case near, far, mixed }

@inline(__always)
private func readingSide(_ r: SideReading?) -> CourtSide? {
    switch r {
    case .near?: return .near
    case .far?: return .far
    default: return nil
    }
}

/// rally._own_side_reading: hitter side from the hit's own launch fit.
func ownSideReading(_ seg: ShotSegment3D?, _ cfg: RallyConfig) -> SideReading? {
    guard let seg else { return nil }
    let vx = seg.v0.x, x = seg.p0.x
    let rv: SideReading? = vx > cfg.sideVxDead ? .near : (vx < -cfg.sideVxDead ? .far : nil)
    let rx: SideReading? = x < -cfg.sideXDead ? .near : (x > cfg.sideXDead ? .far : nil)
    if let rv, let rx, rv != rx { return .mixed }
    return rv ?? rx
}

/// rally._incoming_side_reading: half the incoming fit is on at this contact.
func incomingSideReading(_ prev: ShotSegment3D?, _ h: HitEvent, _ t: [Double], _ cfg: RallyConfig) -> SideReading? {
    guard let prev, prev.t0 < h.t else { return nil }
    let n = t.count
    let tCov = t[Swift.min(Swift.max(prev.endFrame, 0), n - 1)] + 0.25
    if h.t > tCov { return nil }
    let (lo, hi) = xBracket(prev, h.t)
    let m = 0.05                                // m: RK4 (h = 1/60 s) exceeds the exact bracket by <= 5.6 mm
    if lo - m > cfg.sideXDead { return .far }
    if hi + m < -cfg.sideXDead { return .near }
    if lo - m >= -cfg.sideXDead && hi + m <= cfg.sideXDead { return nil }
    let x = stateAt(prev, h.t).x
    return x < -cfg.sideXDead ? .near : (x > cfg.sideXDead ? .far : nil)
}

/// rally._x_bracket: exact bounds of the fitted flight's x at time t without
/// integrating (vx keeps its sign, |vx| <= |v| <= the 1-D Riccati speed).
func xBracket(_ seg: ShotSegment3D, _ t: Double) -> (Double, Double) {
    let x0 = seg.p0.x, vx0 = seg.v0.x
    let tau = t - seg.t0
    let L = seg.dragLength != 0 ? seg.dragLength : Physics.dragLength
    let ax = Swift.abs(vx0)
    if ax == 0.0 || tau <= 0.0 { return (x0, x0) }
    let dHi = L * Foundation.log1p(ax * tau / L)
    let sp = (seg.v0.x * seg.v0.x + seg.v0.y * seg.v0.y + seg.v0.z * seg.v0.z).squareRoot()
    let vt = (Physics.g * L).squareRoot()
    let k = vt / L
    let dLo: Double
    if sp > vt {
        let c = Foundation.atanh(vt / sp)
        dLo = ax * Foundation.sinh(c) / k * Foundation.log(Foundation.tanh((k * tau + c) / 2) / Foundation.tanh(c / 2))
    } else {
        dLo = ax * (1.0 - Foundation.exp(-k * tau)) / k
    }
    let sgn = vx0 > 0 ? 1.0 : -1.0
    let a = x0 + sgn * dLo, b = x0 + sgn * dHi
    return a <= b ? (a, b) : (b, a)
}

/// rally._fit_weight: ramp of the fit's confidence between sideFitQLo and sideFitQHi.
func fitWeight(_ seg: ShotSegment3D?, _ cfg: RallyConfig) -> Double {
    guard let seg else { return 0.0 }
    let q = seg.confidence
    return Swift.min(Swift.max((q - cfg.sideFitQLo) / (cfg.sideFitQHi - cfg.sideFitQLo), 0.0), 1.0)
}

/// rally._departure: image distance (x image height) between the observed track
/// and the incoming fit's projection at the moving sample closest to h.t + lag
/// (before the next rest, within 0.06 s of it); nil when not observed.
func departure(_ mo: Motion, _ h: HitEvent, _ prev: ShotSegment3D, _ camera: CameraModel,
               _ lag: Double) -> Double? {
    let fHi = Swift.min(mo.n, mo.frameAt(h.t + lag + 0.1) + 1)
    let end = mo.firstRestAfter(h.frame, fHi) ?? fHi
    var best = -1, bd = 0.0
    if end > h.frame + 1 {
        for k in (h.frame + 1)..<end where mo.fin[k] {
            let d = Swift.abs(mo.t[k] - (h.t + lag))
            if best < 0 || d < bd { best = k; bd = d }
        }
    }
    if best < 0 || bd > 0.06 { return nil }
    let st = stateAt(prev, mo.t[best])
    let uv = camera.project(SIMD3(st.x, st.y, st.z))
    guard uv.x.isFinite && uv.y.isFinite else { return nil }
    return Py.hypot(uv.x - mo.xy[best].x, uv.y - mo.xy[best].y) / mo.h
}

/// rally._launches: contact h has 3-D evidence of a racket launch (own fit
/// determined, incoming shuttle airborne, x-velocity reversed with |dv| >=
/// minContactDv) and, with a camera, the 2-D track left the incoming fit.
func launches(_ h: HitEvent, _ seg: ShotSegment3D?, _ prev: ShotSegment3D?, _ cfg: RallyConfig,
              _ mo: Motion? = nil, _ camera: CameraModel? = nil) -> Bool {
    guard let seg, let prev, fitWeight(seg, cfg) >= cfg.trailLaunchW else { return false }
    let st = stateAt(prev, h.t)
    if !(st.z > 0.0) { return false }             // incoming flight already on the floor: no reading
    let vx = seg.v0.x
    if st.vx * vx >= 0.0 { return false }
    let dx = st.vx - vx, dy = st.vy - seg.v0.y, dz = st.vz - seg.v0.z
    if (dx * dx + dy * dy + dz * dz).squareRoot() < cfg.minContactDv { return false }
    if let mo, let camera, let dep = departure(mo, h, prev, camera, cfg.trailDepartLagS),
       dep < cfg.trailDepartMin {
        return false                              // one flight: the track stays on the incoming fit
    }
    return true
}

/// rally._side_llr: log P(evidence | near) - log P(evidence | far).
func sideLLR(_ label: CourtSide, _ own: SideReading?, _ inc: SideReading?, _ cfg: RallyConfig,
             wOwn: Double = 1.0, wInc: Double = 1.0) -> Double {
    if label != .unknown {
        var e = cfg.sideLabelLLR
        if let o = readingSide(own) {
            e += wOwn * (o == label ? cfg.sideOwnAgree : cfg.sideOwnConflict)
        } else if own == .mixed {
            e += wOwn * cfg.sideOwnMixed
        }
        if let i = readingSide(inc) {
            e += wInc * (i == label ? cfg.sideInAgree : cfg.sideInConflict)
        }
        return label == .near ? e : -e
    }
    var e = 0.0
    for (r, w) in [(own, wOwn), (inc, wInc)] {
        if r == .near { e += w * cfg.sideUnlabelled } else if r == .far { e -= w * cfg.sideUnlabelled }
    }
    return e
}

/// rally._decode_sides: 2-state forward-backward per rally group. Returns the
/// relabelled hits and per hit P(that side) (0.5 = no evidence, label kept).
func decodeSides(_ hits: [HitEvent], _ segs: [Int?], _ conts: [Int?], _ segments: [ShotSegment3D],
                 _ groups: [[Int]], _ t: [Double], _ cfg: RallyConfig) -> (hits: [HitEvent], prob: [Double]) {
    var out = hits
    var prob = [Double](repeating: 0.5, count: hits.count)
    for g in groups {
        let m = g.count
        var e = [Double]()
        e.reserveCapacity(m)
        for i in g {
            let h = hits[i]
            var prev: ShotSegment3D? = nil
            if i > 0, let pi = conts[i - 1] ?? segs[i - 1] { prev = segments[pi] }
            let ownSeg = segs[i].map { segments[$0] }
            let own = ownSideReading(ownSeg, cfg)
            e.append(sideLLR(h.hitter, own, incomingSideReading(prev, h, t, cfg), cfg,
                             wOwn: fitWeight(ownSeg, cfg), wInc: fitWeight(prev, cfg)))
        }
        var ls = [Double](repeating: 0.0, count: m), ld = [Double](repeating: 0.0, count: m)
        for k in 1..<Swift.max(m, 1) {
            var z = (hits[g[k]].t - hits[g[k - 1]].t - cfg.sideSameMidS) / cfg.sideSameWidthS
            z = Swift.max(z, -30.0)
            let ps = Swift.min(Swift.max(1.0 / (1.0 + Foundation.exp(-z)), cfg.sideSameEps), 1.0 - cfg.sideSameEps)
            ls[k] = Foundation.log(ps); ld[k] = Foundation.log(1.0 - ps)
        }
        var fn = [Double](repeating: 0.0, count: m), ff = [Double](repeating: 0.0, count: m)
        fn[0] = 0.5 * e[0]; ff[0] = -0.5 * e[0]
        for k in 1..<Swift.max(m, 1) {
            fn[k] = 0.5 * e[k] + logAddExp(fn[k - 1] + ls[k], ff[k - 1] + ld[k])
            ff[k] = -0.5 * e[k] + logAddExp(fn[k - 1] + ld[k], ff[k - 1] + ls[k])
        }
        var bn = [Double](repeating: 0.0, count: m), bf = [Double](repeating: 0.0, count: m)
        var k = m - 2
        while k >= 0 {
            let en = 0.5 * e[k + 1] + bn[k + 1], ef = -0.5 * e[k + 1] + bf[k + 1]
            bn[k] = logAddExp(ls[k + 1] + en, ld[k + 1] + ef)
            bf[k] = logAddExp(ld[k + 1] + en, ls[k + 1] + ef)
            k -= 1
        }
        for (k, i) in g.enumerated() {
            let d = (fn[k] + bn[k]) - (ff[k] + bf[k])
            if d != 0.0 {
                out[i].hitter = d > 0 ? .near : .far
                prob[i] = 1.0 / (1.0 + Foundation.exp(-Swift.abs(d)))
            }
        }
    }
    return (out, prob)
}

/// rally.refine_hitters: hitter sides from the joint decode on the INPUT hit list
/// (same order), plus per hit whether its side changed. For an orchestrator that
/// re-runs reconstruct with the corrected sides, then calls segmentRallies again.
public func refineHitters(track: Track2D?, meta: VideoMeta?, hits: [HitEvent], segments: [ShotSegment3D],
                          camera: CameraModel? = nil, config: RallyConfig = RallyConfig())
    -> (hits: [HitEvent], changed: [Bool]) {
    let cfg = config
    guard let track, track.n > 0, !hits.isEmpty else { return (hits, [Bool](repeating: false, count: hits.count)) }
    let height: Double = (meta != nil && meta!.height != 0) ? Double(meta!.height)
        : (camera != nil ? Double(camera!.imageSize.height) : 720.0)
    let mo = Motion(track: track, height: height, cfg: cfg)
    let hs0 = cleanHits(hits, mo.n).map { h -> HitEvent in
        var h = h
        if !h.t.isFinite { h.t = mo.t[h.frame] }
        return h
    }
    let ctx = Ctx(mo: mo, cfg: cfg, segments: segments, sigmas: [Double?](repeating: nil, count: segments.count))
    let segs0 = matchSegments(hs0, segments, mo.t, cfg)
    let merged = mergeSameFlight(hs0, segs0, segments, mo.t, cfg)
    let (groups, _) = ctx.groupHits(merged.hits, merged.segs, merged.conts)
    let dec = decodeSides(merged.hits, merged.segs, merged.conts, segments, groups, mo.t, cfg)
    var side: [Int: CourtSide] = [:]
    for h in dec.hits { side[h.frame] = h.hitter }
    var out = hits, changed = [Bool](repeating: false, count: hits.count)
    for i in hits.indices {
        let s = side[hits[i].frame] ?? hits[i].hitter
        changed[i] = s != hits[i].hitter
        out[i].hitter = s
    }
    return (out, changed)
}

/// rally._alternation: fraction of consecutive known-hitter pairs that alternate.
private func alternation(_ hits: [HitEvent]) -> Double {
    let s = hits.map(\.hitter).filter { $0 != .unknown }
    if s.count < 2 { return 1.0 }
    return Py.npMeanBool((1..<s.count).map { s[$0 - 1] != s[$0] })
}

// MARK: - public API

/// rally.segment_rallies: split a match into rallies with shots, end reason,
/// landing and winner. `segments` may be empty (no camera).
/// `landingSigmaM` (parallel to `segments`) carries reconstruct3d's
/// landing_sigma_m extra, which ShotSegment3D has no field for; nil entries
/// (or a nil array) use `config.defaultLandingSigmaM` like Python does when the
/// segment has no landing covariance.
public func segmentRallies(track: Track2D?, meta: VideoMeta?, hits: [HitEvent], segments: [ShotSegment3D],
                           camera: CameraModel? = nil, players: [PlayerObservation]? = nil,
                           config: RallyConfig = RallyConfig(), landingSigmaM: [Double?]? = nil) -> [RallyV2] {
    let cfg = config
    guard let track, track.n > 0 else { return [] }
    let height: Double = (meta != nil && meta!.height != 0) ? Double(meta!.height)
        : (camera != nil ? Double(camera!.imageSize.height) : 720.0)
    let mo = Motion(track: track, height: height, cfg: cfg)
    let doubles = cfg.doubles ?? detectDoubles(players)
    var hs = cleanHits(hits, mo.n).map { h -> HitEvent in
        var h = h
        if !h.t.isFinite { h.t = mo.t[h.frame] }
        return h
    }
    if hs.isEmpty { return activityRallies(mo, cfg) }
    let sigmas: [Double?] = (0..<segments.count).map { i in
        landingSigmaM != nil && i < landingSigmaM!.count ? landingSigmaM![i] : nil
    }
    let ctx = Ctx(mo: mo, cfg: cfg, segments: segments, sigmas: sigmas)
    let segs0 = matchSegments(hs, segments, mo.t, cfg)
    let merged = mergeSameFlight(hs, segs0, segments, mo.t, cfg)
    hs = merged.hits
    let segs = merged.segs, conts = merged.conts
    let (groups, cues) = ctx.groupHits(hs, segs, conts)
    let labels0 = hs.map(\.hitter)                  // the side prior the 3-D fits were made with
    var sideP: [Double]? = nil
    if cfg.jointSides {
        let dec = decodeSides(hs, segs, conts, segments, groups, mo.t, cfg)
        hs = dec.hits
        sideP = dec.prob
    } else {
        hs = fillHitters(hs, segs, segments, groups)
    }
    let have3D = camera != nil && !segments.isEmpty

    var cands: [Cand] = []
    for (gi, g0) in groups.enumerated() {
        var g = g0
        while !g.isEmpty && ctx.leadingNonhit(g, hs, segs, have3D) { g.removeFirst() }
        while g.count > 1 && segs[g.last!] == nil && !ctx.isFlight(hs[g.last!], nil) { g.removeLast() }
        var contLast: Int? = nil          // fit of a dropped spurious trailing kink: the real end
        while g.count > 1 && hs[g.last!].confidence < cfg.trailMaxConf
                && ctx.movingAfter(hs[g.last!]) < cfg.trailMinFlightS
                && !(cfg.trailLaunchGuard && launches(hs[g.last!], ctx.seg(segs[g.last!]),
                                                      ctx.seg(conts[g[g.count - 2]] ?? segs[g[g.count - 2]]),
                                                      cfg, mo, camera)) {
            let dropped = conts[g.last!] ?? segs[g.last!]
            if cfg.trailKeepFit && contLast == nil && dropped != nil { contLast = dropped }
            g.removeLast()
        }
        if g.count == 1 && have3D && segs[g[0]] == nil && ctx.movingAfter(hs[g[0]]) < cfg.onehitMinFlightS {
            continue
        }
        if g.isEmpty { continue }
        let gHits = g.map { hs[$0] }, gSegs = g.map { segs[$0] }
        let last = gHits.last!
        let limit = gi + 1 < groups.count ? hs[groups[gi + 1][0]].frame - 1 : mo.n - 1
        // the flight is looked at from past a hold touching the last contact (a
        // one-shot rally's serve): that hold is not where the rally ended
        let fLa = ctx.launchFrame(last.frame, limit + 1)
        var fEnd2D = mo.flightEnd(fLa, limit + 1)
        var fRest = mo.firstRestAfter(fLa, Swift.min(limit + 1, mo.frameAt(last.t + 4.0) + 1))
        if let fr = fRest, mo.t[fr] - mo.t[fEnd2D] > cfg.maxUnseenLandS, !ctx.continuousFlight(fLa, fr - 1) {
            fRest = nil
        }
        if fRest == nil { fEnd2D = ctx.extendFlight(fLa, fEnd2D, limit) }
        var contI = conts[g.last!] ?? contLast
        if contI != nil && gSegs.last! == nil { contI = nil }   // no launch fit to pair it with
        let segLastI: Int? = contI ?? gSegs.last!
        let segLast = ctx.seg(segLastI)
        let launch: ShotSegment3D? = contI != nil ? ctx.seg(gSegs.last!) : nil
        var serverY: Double? = ctx.seg(gSegs[0]).map { $0.p0.y }
        let fPrev = gi > 0 ? hs[groups[gi - 1].last!].frame + 1 : 0
        let ms = g.count >= 2 ? ctx.missedServeStart(gHits[0].frame, gHits[0].t, fMin: fPrev) : nil
        if ms != nil { serverY = nil }
        var missedFinal = false
        if let sl = segLast, sl.landingXY == nil, last.hitter != .unknown {
            let nh = sl.netCrossingHeight
            var netBound = sl.endReason == .net || (nh != nil && nh! < CourtModel.netHeightCentre)
            if let launch {
                let lh = launch.netCrossingHeight
                netBound = lh != nil && lh! < CourtModel.netHeightCentre
            }
            let fSe = Swift.min(Swift.max(sl.endFrame, 0), mo.n - 1)
            let fFly = fRest == nil ? fEnd2D : Swift.min(fEnd2D, fRest! - 1)
            missedFinal = !netBound && fFly > fSe && mo.t[fFly] - mo.t[fSe] > cfg.missedFinalS
                && ctx.continuousFlight(fSe, fFly)
        }
        // the last flight's landing is judged for the side its fit is consistent
        // with (own launch reading if clear, else the label it was fitted with);
        // confidence carries P(decoded side == that side) (see rally.py)
        var lastP: Double? = sideP.map { $0[g.last!] }
        var callHit = last
        if let lp = lastP, segLast != nil {
            var own = readingSide(ownSideReading(ctx.seg(gSegs.last!), cfg))
            if fitWeight(ctx.seg(gSegs.last!), cfg) < 0.5 { own = nil }   // an undetermined fit reads no side
            let fitSide = (cfg.sideCallMode == "label" || own == nil) ? labels0[g.last!] : own!
            if fitSide != .unknown && fitSide != last.hitter {
                callHit.hitter = fitSide
                lastP = 1.0 - lp
            }
        }
        var o: Outcome
        if missedFinal {
            o = Outcome(source: "missed_final")
        } else if let sl = segLast {
            var fe = sl.endFrame
            if fEnd2D > last.frame { fe = Swift.min(fe, fEnd2D) }
            fe = Swift.min(Swift.max(fe, 0), mo.n - 1)
            o = outcome3D(hit: callHit, seg: sl, sigma: ctx.sigmas[segLastI!], tLast: mo.t[fe],
                          isServe: g.count == 1 && ms == nil, doubles: doubles, serverY: serverY, cfg: cfg,
                          launch: launch, sideP: lastP)
        } else if camera != nil {
            o = outcome2D(hit: last, fRest: fRest, mo: mo, camera: camera, doubles: doubles, cfg: cfg)
            if let lp = lastP { o.confidence *= lp }
        } else {
            o = Outcome()
        }
        let fEnd: Int
        if let et = o.endT, o.source == "3d" || o.source == "implausible_landing" {
            fEnd = Swift.min(Swift.max(mo.frameAt(et), last.frame), Swift.max(limit, last.frame))
        } else if let fr = fRest {
            fEnd = fr
        } else {
            fEnd = Swift.max(fEnd2D, last.frame)
        }
        let cb: Cue? = gi > 0 ? cues[gi - 1] : nil
        let ca: Cue? = gi < cues.count ? cues[gi] : nil
        let boundary = 0.5 * (cueScore(cb) + cueScore(ca))
        let hconf = Py.npMean(gHits.map { Swift.min(Swift.max($0.confidence, 0.0), 1.0) })
        let segConf = 0.3 + 0.35 * boundary + 0.2 * alternation(gHits) + 0.15 * hconf
        cands.append(Cand(idx: g, hits: gHits, segs: gSegs, startFrame: ms ?? gHits[0].frame,
                          endFrame: fEnd, endT: mo.t[fEnd], outcome: o, segConf: Swift.min(segConf, 1.0),
                          cueBefore: cb, serveMissed: ms != nil,
                          firstSideP: sideP.map { $0[g[0]] } ?? 1.0,
                          heldBefore: ctx.holdBefore(gHits[0].frame, gHits[0].t) == 1.0))
    }

    // one-hit groups: feeds (shuttle returned to the server) / spurious hits
    var kept: [Cand] = []
    for (i, c) in cands.enumerated() {
        if c.hits.count == 1 {
            let prev = kept.last
            let nxt = i + 1 < cands.count ? cands[i + 1] : nil
            if ctx.nonrallyLogOdds(c, prev, nxt, doubles, have3D) > cfg.feedThreshold { continue }
        }
        kept.append(c)
    }
    inferFromNextServe(&kept, cfg, segments)
    return kept.map { toRally($0, segments) }
}
