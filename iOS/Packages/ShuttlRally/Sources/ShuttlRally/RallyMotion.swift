import Foundation
import ShuttlCore

/// rally.RallyConfig — Python defaults.
public struct RallyConfig: Sendable, Equatable {
    public var maxHitGapS = 2.6            // longer without a hit -> dead ball ...
    public var maxHitGapMissedS = 4.5      // ... unless the track keeps flying (missed hit)
    public var maxUnseenS = 2.2            // ... with no longer unobserved stretch
    public var maxHitGapUnseenS = 6.0      // <= 3 flights of ~2 s whose contacts all went unseen
    public var maxUnseenFlightS = 4.5      // ... out of view at most two of the longest flights
    public var minContactDv = 5.0          // m/s: a racket contact changes the velocity by more
    public var maxUnseenLandS = 1.5        // last seen in flight -> seen at rest within this
    public var missedFinalS = 0.35         // track flies on this long past the last fit's end
    public var landingSlackS = 0.35        // fitted floor contact this long before the next hit
    public var minLandingConf = 0.25       // trust landing_t for splitting above this fit confidence
    public var staticS = 0.6               // 2-D stillness this long = shuttle at rest
    public var staticRadiusFrac = 0.01     // of image height
    public var holdWindowS = 0.7           // pre-serve hold must end this close to the serve
    public var minFlightSpeedHps = 0.3     // image heights / s right after a real hit
    public var minLaunchSpeedHps = 0.08    // ... or this and 2.5x the speed before it
    public var launchWindowS = 0.3
    public var flightWindowS = 0.5
    public var feedThreshold = 0.0         // feed log-odds above this -> dropped
    public var netMarginM = 0.0            // crossing below net height - margin -> 'net' ...
    public var netCordTolM = 0.12          // ... unless within this and seen landing beyond the net
    public var defaultLandingSigmaM = 0.12
    public var landingOutsideMaxM = 3.0    // floor point this far beyond the doubles court = no landing
                                           // (F3 timestamp jump: -11.1 m); = ShuttlAnalysis LandingGuard.marginM
    public var overrideConf = 0.55         // next-serve evidence overrides outcomes below this
    public var nsConflictConf = 0.5        // cap when the next serve contradicts a call < nsConflictMaxQ,
    public var nsConflictMaxQ = 0.8        // or the verdict rests on an unseen (inferred) serve (rally.py)
    public var netCallQ = 0.75             // net faults: hitter side least certain near the net
    public var sideConflictQ = 0.6         // launch direction contradicts the label (jointSides == false only)
    // joint hitter-side decode (rally._decode_sides); LLRs measured on sim dev seeds 0-39
    public var jointSides = true
    public var sideLabelLLR = 2.0          // hits.py label alone: ~88 % right from behind
    public var sideOwnAgree = 1.6          // own launch fit agrees with the label it was fitted with
    public var sideOwnConflict = -4.5      // ... contradicts it although pulled towards it
    public var sideOwnMixed = -2.2         // launch x and launch vx point at different sides
    public var sideInAgree = 1.5           // incoming fit at the contact is on the label's half
    public var sideInConflict = -3.8       // ... on the other half
    public var sideUnlabelled = 2.5        // |LLR| of one fit reading for a hit without a label
    public var sideFitQLo = 0.1            // fit readings count x ramp(fit confidence, lo -> hi)
    public var sideFitQHi = 0.3
    public var sideVxDead = 0.5            // m/s: |launch vx| below this reads no direction
    public var sideXDead = 0.1             // m: a fit position this close to the net reads no side
    public var sideSameMidS = 1.9          // P(same side) logistic in the gap between contacts
    public var sideSameWidthS = 0.15
    public var sideSameEps = 0.02          // floor / ceiling (spurious contacts)
    public var sideCallMode = "fit"        // 'fit' | 'label': side the last landing is judged for
    public var nextServeSideQ = true       // next-serve verdict confidence x P(next server's side)
    public var trailMaxConf = 0.8          // a rally's last contact below this confidence ...
    public var trailMinFlightS = 0.6       // ... followed by less observed motion is dropped
    public var trailLaunchGuard = true     // ... unless its own fit shows a racket launch (rally._launches)
    public var trailLaunchW = 0.5          // fit weight needed to read a launch
    public var trailDepartLagS = 0.1       // ... and the 2-D track this long after the contact left the
    public var trailDepartMin = 0.01       // incoming fit's projection by >= this x image height
    public var trailKeepFit = true         // the dropped kink's fit describes the end of the last flight
    public var onehitMinFlightS = 0.3      // lone contact, no 3-D fit (others have one), less motion: no rally
    public var nextServeNeedsHold = true   // only a first contact struck from a seen hold names the server
    public var skipServeHold = true        // long-gap test starts after a hold touching the first contact
    public var unseenArrivalOK = true      // unseen arrival: judged by the last sighting (fast = no hold)
    public var unseenExitMaxY = 0.35       // ... which must be in the top 35 % of the frame
    public var doubles: Bool? = nil        // nil = auto from players (4 per frame)
    public var activityMinS = 1.0          // hit-less fallback: min moving stretch
    public var activityGapS = 2.2
    public var segMatchBefore = 3          // segment may start this many frames before its hit
    public var segMatchAfterS = 0.4        // ... or this long after it

    public init() {}
}

/// rally._Motion: per-frame image kinematics of a Track2D.
struct Motion {
    let t: [Double]
    let n: Int
    let xy: [SIMD2<Double>]
    let fin: [Bool]
    let real: [Bool]
    let h: Double
    let dt: Double                             // np.median(np.diff(t)) (1/30 for < 2 frames)
    private(set) var speed: [Double] = []
    private(set) var rest: [Bool] = []

    init(track: Track2D, height: Double, cfg: RallyConfig) {
        let tt = track.t, n0 = tt.count
        let xy0 = track.xy.count == n0 ? track.xy : (0..<n0).map { $0 < track.xy.count ? track.xy[$0] : SIMD2(.nan, .nan) }
        t = tt
        n = n0
        xy = xy0
        let fin = (0..<n0).map { xy0[$0].x.isFinite && xy0[$0].y.isFinite && tt[$0].isFinite }
        let vis = track.visible.count == n0 ? track.visible : fin
        let interp = track.interpolated.count == n0 ? track.interpolated : [Bool](repeating: false, count: n0)
        var real = (0..<n0).map { fin[$0] && vis[$0] && !interp[$0] }
        if !real.contains(true) { real = fin }
        self.fin = fin
        self.real = real
        h = Swift.max(height, 1.0)
        if n0 > 1 {
            let d = (1..<n0).map { tt[$0] - tt[$0 - 1] }
            dt = d.contains(where: { $0.isNaN }) ? .nan : Py.median(d)
        } else {
            dt = 1.0 / 30.0
        }
        speed = computeSpeed()
        rest = computeRest(staticS: cfg.staticS, radius: cfg.staticRadiusFrac * h)
    }

    /// Image speed over a >= `base` s baseline.
    private func computeSpeed(base: Double = 0.05) -> [Double] {
        var sp = [Double](repeating: .nan, count: n)
        let idx = (0..<n).filter { fin[$0] }
        if idx.count < 2 { return sp }
        let tt = idx.map { t[$0] }
        for (k, i) in idx.enumerated() {
            let j = Num.searchsorted(tt, tt[k] - base + 1e-9, right: true) - 1
            guard j >= 0 else { continue }
            let dt = tt[k] - tt[j]
            guard dt > 0 && dt <= 4 * base else { continue }
            let d = Py.rowNorm2(xy[i].x - xy[idx[j]].x, xy[i].y - xy[idx[j]].y)
            sp[i] = d / dt / h
        }
        return sp
    }

    /// Greedy scan of real samples for >= static_s clusters within `radius` px.
    private func computeRest(staticS: Double, radius: Double, maxBad: Int = 2) -> [Bool] {
        var rest = [Bool](repeating: false, count: n)
        let idx = (0..<n).filter { real[$0] }
        if idx.count < 3 { return rest }
        let tt = idx.map { t[$0] }, x = idx.map { xy[$0].x }, y = idx.map { xy[$0].y }
        let r2 = radius * radius
        let maxSkip = Swift.max(0.25, staticS / 2)
        var i = 0
        let m = idx.count
        while i < m {
            var last = i, bad = 0, k = i + 1
            let xi = x[i], yi = y[i]
            while k < m && tt[k] - tt[last] <= maxSkip {
                let dx = x[k] - xi, dy = y[k] - yi
                if dx * dx + dy * dy <= r2 {
                    last = k; bad = 0
                } else {
                    bad += 1
                    if bad > maxBad { break }
                }
                k += 1
            }
            if last > i && tt[last] - tt[i] >= staticS {
                for f in idx[i]...idx[last] { rest[f] = true }
                i = last + 1
            } else {
                i += 1
            }
        }
        return rest
    }

    func searchsorted(_ v: Double, right: Bool = false) -> Int { Num.searchsorted(t, v, right: right) }

    func frameAt(_ tv: Double) -> Int {
        let k = searchsorted(tv)
        if k <= 0 { return 0 }
        if k >= n { return n - 1 }
        return Swift.abs(t[k] - tv) < Swift.abs(t[k - 1] - tv) ? k : k - 1
    }

    /// Finite speeds in [a, b) (numpy slice semantics, clamped).
    func finiteSpeeds(_ a: Int, _ b: Int) -> [Double] {
        let lo = Swift.max(a, 0), hi = Swift.min(b, n)
        guard hi > lo else { return [] }
        return speed[lo..<hi].filter { $0.isFinite }
    }

    func anyRest(_ a: Int, _ b: Int) -> Bool {
        let lo = Swift.max(a, 0), hi = Swift.min(b, n)
        guard hi > lo else { return false }
        return rest[lo..<hi].contains(true)
    }

    func peakSpeed(_ f0: Int, _ windowS: Double, _ q: Double = 60.0) -> Double? {
        if f0 < 0 || f0 >= n || windowS <= 0 { return nil }
        let f1 = searchsorted(t[f0] + windowS, right: true)
        let v = finiteSpeeds(f0 + 1, f1)
        if v.count < 3 { return nil }
        return Py.percentile(v, q)
    }

    /// First rest frame in (f0, f1).
    func firstRestAfter(_ f0: Int, _ f1: Int) -> Int? {
        let a = Swift.max(f0 + 1, 0), b = Swift.min(f1, n)
        if b <= a { return nil }
        for f in a..<b where rest[f] { return f }
        return nil
    }

    /// Last finite frame of the flight starting at f0.
    func flightEnd(_ f0: Int, _ f1In: Int, maxSkipS: Double = 0.5) -> Int {
        let f1 = Swift.min(f1In, n)
        var last = f0
        var lastT = (0 <= f0 && f0 < n) ? t[f0] : 0.0
        var f = f0 + 1
        while f < f1 {
            if rest[f] { return f }
            if fin[f] {
                if t[f] - lastT > maxSkipS { break }
                last = f; lastT = t[f]
            } else if t[f] - lastT > maxSkipS {
                break
            }
            f += 1
        }
        return last
    }

    /// Rest strictly inside two hits.
    func staticBetween(_ f0: Int, _ f1: Int, marginS: Double = 0.15) -> Bool {
        let a = searchsorted(t[f0] + marginS, right: true)
        let b = searchsorted(t[f1] - marginS, right: false)
        return b > a && anyRest(a, b)
    }
}
