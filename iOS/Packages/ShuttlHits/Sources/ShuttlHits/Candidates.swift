import Foundation
import ShuttlCore

/// One scored hit candidate before sequence decoding (hits.py `_Cand`).
/// Also the training row for a learned re-scorer (`feats` in `HIT_FEATURES` order).
public struct HitCandidate: Sendable, Equatable {
    public var tau: Double                  // sub-frame change time (s)
    public var frame: Int                   // nearest frame
    public var xy: SIMD2<Double>            // fitted contact pixel
    public var lam: Double
    public var lamV: Double
    public var vIn: SIMD2<Double>           // px/s
    public var vOut: SIMD2<Double>
    public var gapS: Double                 // observation gap bracketing tau (inf: none before)
    public var isStart = false
    public var staticBefore = false
    public var staticAfter = false
    public var outside = false
    public var settles = false              // small bounce then rest: landing, not a contact
    public var kin = 0.0                    // kinematic log-odds of being a contact
    public var reach = 0.0                  // player-proximity log-odds term
    public var sideLLR = 0.0                // log p(near)/p(far) from unary cues
    public var nearTID: Int? = nil
    public var farTID: Int? = nil
    public var seg = 0                      // rally segment (between dead times / rest)
    public var weak = false                 // below lam_min: kept only if a player is in reach
    public var hidden = false               // contact above the frame inside a top-exit/top-entry gap
    public var gapContact = false           // inferred at the first observation after a PTS gap (F4)
    public var feats: [Double] = []

    public init(tau: Double, frame: Int, xy: SIMD2<Double>, lam: Double, lamV: Double,
                vIn: SIMD2<Double>, vOut: SIMD2<Double>, gapS: Double, isStart: Bool = false, weak: Bool = false,
                hidden: Bool = false, gapContact: Bool = false) {
        self.tau = tau; self.frame = frame; self.xy = xy; self.lam = lam; self.lamV = lamV
        self.vIn = vIn; self.vOut = vOut; self.gapS = gapS; self.isStart = isStart; self.weak = weak
        self.hidden = hidden; self.gapContact = gapContact
    }
}

/// Cleaned observation series (times ascending) with range helpers.
struct ObsSeries {
    let t: [Double]
    let xy: [SIMD2<Double>]
    let sorted: Bool

    init(t: [Double], xy: [SIMD2<Double>]) {
        self.t = t; self.xy = xy
        var s = true
        for i in 1..<max(t.count, 1) where t[i] < t[i - 1] { s = false; break }
        sorted = s
    }

    var count: Int { t.count }

    /// Indices with t0 < t < t1 (the numpy boolean mask), ascending.
    func open(_ t0: Double, _ t1: Double) -> Range<Int>? {
        guard sorted else { return nil }
        let a = NP.searchsorted(t, t0, right: true)
        let b = NP.searchsorted(t, t1)
        return a < b ? a..<b : a..<a
    }

    func indices(_ t0: Double, _ t1: Double) -> [Int] {
        if let r = open(t0, t1) { return Array(r) }
        return t.indices.filter { t[$0] > t0 && t[$0] < t1 }
    }
}

extension Hits {
    /// max |p - median(p)| over the points (np.median along axis 0).
    static func spread(_ pts: [SIMD2<Double>]) -> Double {
        let mx = NP.median(pts.map(\.x)), my = NP.median(pts.map(\.y))
        var best = -Double.infinity
        for p in pts {
            let d = NP.rowNorm(p.x - mx, p.y - my)
            if d.isNaN { return .nan }
            if d > best { best = d }
        }
        return best
    }

    static func isStatic(_ o: ObsSeries, _ t0: Double, _ t1: Double, _ tol: Double) -> Bool {
        let sel = o.indices(t0, t1)
        if sel.count < 3 { return false }
        return spread(sel.map { o.xy[$0] }) < tol
    }

    /// Indices i where the gap t[i] -> t[i+1] is dead time (play interrupted).
    static func deadGaps(_ o: ObsSeries, _ cfg: HitConfig, _ W: Double, _ H: Double) -> [Int] {
        let bx = cfg.borderBand * W, by = cfg.borderBand * H
        func atBorder(_ p: SIMD2<Double>) -> Bool { p.x < bx || p.x > W - bx || p.y < by || p.y > H - by }
        var out: [Int] = []
        for i in 0..<max(o.count - 1, 0) where o.t[i + 1] - o.t[i] > cfg.deadTimeS {
            if o.t[i + 1] - o.t[i] <= cfg.maxOutOfViewS && atBorder(o.xy[i]) && atBorder(o.xy[i + 1]) { continue }
            out.append(i)
        }
        return out
    }

    /// After tau the shuttle stays within `radius` and ends at rest before settle_s.
    static func settles(_ o: ObsSeries, _ tau: Double, _ cfg: HitConfig, _ tol: Double, _ radius: Double) -> Bool {
        let sel = o.indices(tau, tau + cfg.settleS)
        if sel.count < 4 { return false }
        let p0 = o.xy[sel[0]]
        var mx = -Double.infinity
        for k in sel {
            let d = NP.rowNorm(o.xy[k].x - p0.x, o.xy[k].y - p0.y)
            if d.isNaN { mx = .nan; break }
            if d > mx { mx = d }
        }
        if mx >= radius { return false }
        let tl = o.t[sel[sel.count - 1]]
        return isStatic(o, tl - cfg.staticS, tl + 1e-9, 2.0 * tol)
    }

    /// Times where play is certainly interrupted (dead gaps, start of every rest stretch).
    static func rallyBreaks(_ o: ObsSeries, _ cfg: HitConfig, _ tol: Double, _ dead: [Int]) -> [Double] {
        var br = dead.map { 0.5 * (o.t[$0] + o.t[$0 + 1]) }
        var j = 0
        var wasStatic = false
        for i in 0..<o.count {
            while o.t[j] < o.t[i] - cfg.staticS { j += 1 }
            var st = false
            if i - j >= 2 && o.t[i] - o.t[j] >= 0.8 * cfg.staticS {
                st = spread(Array(o.xy[j...i])) < tol
            }
            if st && !wasStatic { br.append(o.t[j]) }
            wasStatic = st
        }
        return br.sorted()
    }

    /// (tau, obs index, xy, v_out) for flights that begin after dead time.
    static func flightStarts(_ o: ObsSeries, _ cfg: HitConfig, _ diag: Double, _ dt: Double, _ dead: [Int],
                             _ tFirst: Double) -> [(tau: Double, i: Int, xy: SIMD2<Double>, v: SIMD2<Double>)] {
        var out: [(tau: Double, i: Int, xy: SIMD2<Double>, v: SIMD2<Double>)] = []
        let n = o.count
        for i in [0] + dead.map({ $0 + 1 }) {
            if i == 0 && o.t[0] - tFirst < 2.0 * dt { continue }
            var j = NP.searchsorted(o.t, o.t[i] + cfg.windowS, right: true)
            j = Swift.max(j, i + cfg.minSide)
            if j > n || o.t[j - 1] - o.t[i] > cfg.maxWindowS { continue }
            let k = j - i
            var A = Matrix(rows: k, cols: 2), B = Matrix(rows: k, cols: 2)
            for r in 0..<k {
                A[r, 0] = 1; A[r, 1] = o.t[i + r] - o.t[i]
                B[r, 0] = o.xy[i + r].x; B[r, 1] = o.xy[i + r].y
            }
            guard let coef = LinAlg.lstsq(A, B) else { continue }
            let v = SIMD2(coef[1, 0], coef[1, 1])
            if NP.norm2(v.x, v.y) / diag >= cfg.minSpeedOut {
                out.append((o.t[i] - 0.5 * dt, i, o.xy[i], v))
            }
        }
        return out
    }

    /// hits._hidden_contacts: (tau, obs index after the gap, contact pixel above the frame) for
    /// gaps of hiddenGapS ... maxOutOfViewS left upwards and re-entered downwards through the top band.
    static func hiddenContacts(_ o: ObsSeries, _ cfg: HitConfig, _ W: Double, _ H: Double)
        -> [(tau: Double, i: Int, xy: SIMD2<Double>)] {
        var out: [(tau: Double, i: Int, xy: SIMD2<Double>)] = []
        let by = cfg.borderBand * H
        let n = o.count
        if n < 2 { return out }
        for i in 0..<(n - 1) where o.t[i + 1] - o.t[i] >= cfg.hiddenGapS {
            if i < 2 || i + 3 > n || o.t[i + 1] - o.t[i] > cfg.maxOutOfViewS { continue }
            if !(o.xy[i].y < by && o.xy[i + 1].y < by) { continue }
            if !(o.xy[i].y < o.xy[i - 2].y && o.xy[i + 2].y > o.xy[i + 1].y) { continue }
            let uv = SIMD2(0.5 * (o.xy[i].x + o.xy[i + 1].x), -0.1 * H)
            out.append((0.5 * (o.t[i] + o.t[i + 1]), i + 1, uv))
        }
        return out
    }

    /// hits._gap_contacts: (tau = first observation after the gap, its index, Lambda of one
    /// quadratic across the gap vs one each side, vIn, vOut) per presentation-clock gap
    /// (ShuttlCore PTSGaps of the track clock) whose first observation is within gapFirstObsS
    /// and that has >= minSide observations on each side (up to gapSide).
    static func gapContacts(_ fitter: Fitter, oi: [Int], t: [Double], trackT: [Double], _ cfg: HitConfig,
                            sigma: Double) -> [(tau: Double, i: Int, lam: Double, vIn: SIMD2<Double>, vOut: SIMD2<Double>)] {
        var out: [(tau: Double, i: Int, lam: Double, vIn: SIMD2<Double>, vOut: SIMD2<Double>)] = []
        for g in PTSGaps.frames(trackT) {
            var m = 0
            while m < oi.count && oi[m] < g { m += 1 }            // np.searchsorted(oi, g) (left)
            if m < 1 || m >= t.count || t[m] - trackT[g] > cfg.gapFirstObsS { continue }
            let nl = Swift.min(cfg.gapSide, m), nr = Swift.min(cfg.gapSide, t.count - m)
            if nl < cfg.minSide || nr < cfg.minSide { continue }
            let f = fitter.fit(tau: 0.5 * (t[m - 1] + t[m]), lo: m - nl, hi: m + nr, m: m, scale: cfg.windowS)
            out.append((t[m], m, lambda(f, sigma).lam, f.vIn, f.vOut))
        }
        return out
    }

    @inline(__always) static func cosTurn(_ c: HitCandidate) -> Double {
        NP.dot2(c.vIn.x, c.vIn.y, c.vOut.x, c.vOut.y)
            / (NP.norm2(c.vIn.x, c.vIn.y) * NP.norm2(c.vOut.x, c.vOut.y) + 1e-12)
    }

    static func kinematicLogodds(_ c: HitCandidate, _ diag: Double, _ cfg: HitConfig) -> Double {
        if c.hidden { return cfg.hiddenLogodds }
        let sIn = NP.norm2(c.vIn.x, c.vIn.y) / diag
        let sOut = NP.norm2(c.vOut.x, c.vOut.y) / diag
        if c.gapContact {                                                        // hits.py (F4 gap contact)
            var lo = NP.pyMin(1.5 * Foundation.log(NP.pyMax(c.lam, 1e-6) / cfg.gapLamRef), cfg.gapLogoddsMax)
            lo -= 4.0 * NP.clip((cfg.minSpeedOut - sOut) / cfg.minSpeedOut, 0, 1)
            if c.staticAfter || (c.settles && sOut < cfg.settleSpeed) { lo -= 6.0 }
            if c.outside { lo -= 5.0 }
            return NP.clip(lo, -8.0, 8.0)
        }
        var lo: Double
        if c.isStart {
            lo = 1.0
        } else {
            lo = 1.5 * Foundation.log(NP.pyMax(c.lamV, 1e-6) / cfg.lamRef)
            if cosTurn(c) > 0.95 && sOut < 1.2 * sIn { lo -= 2.0 }       // decelerating continuation
        }
        lo -= 4.0 * NP.clip((cfg.minSpeedOut - sOut) / cfg.minSpeedOut, 0, 1)
        if c.staticAfter || (c.settles && sOut < cfg.settleSpeed) { lo -= 6.0 }   // came to rest: landing
        if c.outside { lo -= 5.0 }                                               // apex / exit out of view
        return NP.clip(lo, -8.0, 8.0)
    }

    /// The per-candidate feature vector, in `HIT_FEATURES` order.
    static func features(_ c: HitCandidate, _ diag: Double) -> [Double] {
        let sIn = NP.norm2(c.vIn.x, c.vIn.y) / diag
        let sOut = NP.norm2(c.vOut.x, c.vOut.y) / diag
        let cos = cosTurn(c)
        return [
            Foundation.log(NP.pyMax(c.lam, 1e-6)),                 // log_lam
            Foundation.log(NP.pyMax(c.lamV, 1e-6)),                // log_lam_v
            sIn,                                                   // speed_in
            sOut,                                                  // speed_out
            Foundation.log((sOut + 1e-3) / (sIn + 1e-3)),          // log_speed_ratio
            cos,                                                   // cos_turn
            NP.sign(c.vIn.y) != NP.sign(c.vOut.y) ? 1 : 0,        // vy_flip
            c.staticBefore ? 1 : 0,                                // static_before
            c.staticAfter ? 1 : 0,                                 // static_after
            c.outside ? 1 : 0,                                     // outside
            c.gapS,                                                // gap_s
            c.reach,                                               // player_reach
            c.isStart ? 1 : 0,                                     // is_flight_start
        ]
    }
}
