import Foundation
import ShuttlCore

// Port of backend/app/services/tracking/shots.py: shot-type classification
// (transparent fuzzy decision tree over the 3-D flight at contact + pose),
// hitter position / contact height, and stroke side from COCO keypoints.

public enum ShotClassifier {
    public static let maxBackpropS = 0.25          // never extrapolate a fit further back than this
    static let maxSpeed = 500.0 / 3.6              // m/s, plausibility bound for the backward solution
    static let kpConf = 0.3
    static let minWidthRatio = 0.4                 // across-body width / torso length: side-on below
    static let deadZone = 0.25                     // |wrist lateral offset| / shoulder width: ambiguous

    // COCO indices
    static let NOSE = 0, LSH = 5, RSH = 6, LWR = 9, RWR = 10, LHIP = 11, RHIP = 12

    // MARK: soft conditions

    @inline(__always) static func sig(_ x: Double) -> Double {
        if x >= 0 { return 1.0 / (1.0 + exp(-x)) }
        let e = exp(x)
        return e / (1.0 + e)
    }

    @inline(__always) static func gt(_ a: Double, _ b: Double, _ s: Double) -> Double { sig((a - b) / s) }
    @inline(__always) static func lt(_ a: Double, _ b: Double, _ s: Double) -> Double { sig((b - a) / s) }
    /// NaN input → 0.5 (uninformative).
    @inline(__always) static func g(_ a: Double, _ b: Double, _ s: Double) -> Double { a.isNaN ? 0.5 : gt(a, b, s) }
    @inline(__always) static func l(_ a: Double, _ b: Double, _ s: Double) -> Double { a.isNaN ? 0.5 : lt(a, b, s) }
    @inline(__always) static func between(_ a: Double, _ lo: Double, _ hi: Double, _ s: Double) -> Double {
        g(a, lo, s) * l(a, hi, s)
    }

    // MARK: scalar shuttle integrator

    typealias V3 = (Double, Double, Double)

    @inline(__always) static func acc(_ vx: Double, _ vy: Double, _ vz: Double, _ k: Double) -> V3 {
        let sp = (vx * vx + vy * vy + vz * vz).squareRoot() * k
        return (-sp * vx, -sp * vy, -Physics.g - sp * vz)
    }

    @inline(__always) static func rk4(_ p: V3, _ v: V3, _ h: Double, _ k: Double) -> (V3, V3) {
        let (x, y, z) = p, (vx, vy, vz) = v
        let a1 = acc(vx, vy, vz, k)
        let b = (vx + 0.5 * h * a1.0, vy + 0.5 * h * a1.1, vz + 0.5 * h * a1.2)
        let a2 = acc(b.0, b.1, b.2, k)
        let c = (vx + 0.5 * h * a2.0, vy + 0.5 * h * a2.1, vz + 0.5 * h * a2.2)
        let a3 = acc(c.0, c.1, c.2, k)
        let d = (vx + h * a3.0, vy + h * a3.1, vz + h * a3.2)
        let a4 = acc(d.0, d.1, d.2, k)
        let w = h / 6
        let p2 = (x + w * (vx + 2 * b.0 + 2 * c.0 + d.0),
                  y + w * (vy + 2 * b.1 + 2 * c.1 + d.1),
                  z + w * (vz + 2 * b.2 + 2 * c.2 + d.2))
        let v2 = (vx + w * (a1.0 + 2 * a2.0 + 2 * a3.0 + a4.0),
                  vy + w * (a1.1 + 2 * a2.1 + 2 * a3.1 + a4.1),
                  vz + w * (a1.2 + 2 * a2.2 + 2 * a3.2 + a4.2))
        return (p2, v2)
    }

    static func dragK(_ dl: Double) -> Double {
        let L = (dl != 0 && dl.isFinite && dl > 0.5) ? dl : Physics.dragLength
        return 1.0 / L
    }

    /// shots.contact_state: (p, v) at the contact time, back-propagated through
    /// the drag ODE (<= 0.25 s, stopped before the backward solution blows up).
    public static func contactState(_ seg: ShotSegment3D, tHit: Double?) -> (p: SIMD3<Double>, v: SIMD3<Double>) {
        let p = seg.p0, v = seg.v0
        let t0 = seg.t0
        guard let tHit, tHit.isFinite && t0.isFinite else { return (p, v) }
        let dt = t0 - tHit
        guard 1e-4 < dt && dt <= maxBackpropS else { return (p, v) }
        let k = dragK(seg.dragLength)
        let n = Swift.max(1, Int((dt * 480).rounded(.up)))
        let h = -dt / Double(n)
        var pp: V3 = (p.x, p.y, p.z), vv: V3 = (v.x, v.y, v.z)
        for _ in 0..<n {
            let (p2, v2) = rk4(pp, vv, h, k)
            let sp = (v2.0 * v2.0 + v2.1 * v2.1 + v2.2 * v2.2).squareRoot()
            if !(sp <= maxSpeed && p2.2 >= 0.0 && p2.0.isFinite && p2.1.isFinite && p2.2.isFinite) { break }
            pp = p2; vv = v2
        }
        return (SIMD3(pp.0, pp.1, pp.2), SIMD3(vv.0, vv.1, vv.2))
    }

    /// shots.FlightFeatures: physical descriptors of one stroke (m, km/h, degrees).
    public struct FlightFeatures: Sendable, Equatable {
        public var contactHeight: Double
        public var contactDepth: Double
        public var speedKmh: Double
        public var elevationDeg: Double
        public var apex: Double
        public var netClearance: Double          // NaN if not crossed
        public var landingDepth: Double
        public var descentDeg: Double
        public var flightTime: Double
    }

    /// shots.flight_features: integrate the free flight to the floor and summarise it.
    public static func flightFeatures(p0: SIMD3<Double>, v0: SIMD3<Double>, dragLength: Double,
                                      dt: Double = 1.0 / 240, tMax: Double = 5.0) -> FlightFeatures {
        let k = dragK(dragLength)
        var p: V3 = (p0.x, p0.y, p0.z), v: V3 = (v0.x, v0.y, v0.z)
        let toward: Double
        if Swift.abs(p.0) > 0.3 || v.0 == 0.0 {
            toward = p.0 > 0 ? -1.0 : 1.0
        } else {
            toward = v.0 > 0 ? 1.0 : -1.0
        }
        var apex = p.2, netZ = Double.nan, t = 0.0
        while t < tMax {
            let (p2, v2) = rk4(p, v, dt, k)
            if netZ.isNaN && (p.0 - 0.0) * (p2.0 - 0.0) <= 0 && p.0 != p2.0 {
                let f = p.0 / (p.0 - p2.0)
                netZ = p.2 + f * (p2.2 - p.2)
            }
            if p2.2 <= 0.0 {
                let d = p.2 - p2.2
                let f = p.2 / (1e-12 > d ? 1e-12 : d)
                p = (p.0 + f * (p2.0 - p.0), p.1 + f * (p2.1 - p.1), p.2 + f * (p2.2 - p.2))
                v = (v.0 + f * (v2.0 - v.0), v.1 + f * (v2.1 - v.1), v.2 + f * (v2.2 - v.2))
                t += f * dt
                break
            }
            p = p2; v = v2; t = t + dt
            if p.2 > apex { apex = p.2 }
        }
        let hv0 = Py.hypot(v0.x, v0.y)
        let hv = Py.hypot(v.0, v.1)
        return FlightFeatures(
            contactHeight: p0.z, contactDepth: Swift.abs(p0.x), speedKmh: Py.norm3(v0) * 3.6,
            elevationDeg: Py.degrees(atan2(v0.z, hv0)), apex: apex,
            netClearance: netZ.isNaN ? .nan : netZ - CourtModel.netHeightCentre,
            landingDepth: p.0 * toward, descentDeg: Py.degrees(atan2(-v.2, hv)), flightTime: t)
    }

    /// Scores in the Python dict's insertion order (matters for the sum).
    typealias Scores = [(String, Double)]

    static func scores3D(_ f: FlightFeatures, _ overhead: Double) -> Scores {
        let h = f.contactHeight, d = f.contactDepth, s = f.speedKmh, e = f.elevationDeg
        let rise = f.apex - f.contactHeight, L = f.landingDepth, nc = f.netClearance
        let low = 1.0 - overhead
        let under = l(h, 1.35, 0.12)
        let down = l(e, -4.0, 2.0)
        let flatFast = l(e, 2.0, 2.0) * g(s, 220.0, 15.0)
        let lofted = g(f.apex, 3.2, 0.25)
        let front = l(d, 2.5, 0.25), rear = g(d, 2.5, 0.25)
        let smash = overhead * (flatFast > down ? flatFast : down) * g(s, 140.0, 12.0) * rear
        let kill = g(h, 1.7, 0.12) * down * g(s, 60.0, 10.0) * front
        let clear = overhead * g(e, 5.0, 3.0) * g(L, 4.4, 0.35) * g(rise, 0.6, 0.15)
        let drop = (1.0 - under) * g(d, 2.5, 0.3) * l(L, 3.0, 0.3) * l(s, 130.0, 12.0) * l(e, 40.0, 5.0)
        // net 70 km/h, push 6.2 m (DELIVERY_R3; shots.py _scores_3d, same evidence)
        let net = low * front * l(f.apex, 3.0, 0.2) * l(L, 2.3, 0.3) * l(s, 70.0, 8.0) * l(nc, 0.9, 0.15)
        let block = under * rear * l(L, 2.7, 0.25) * l(s, 75.0, 10.0)
        let lift = low * lofted * g(e, 8.0, 3.0) * g(L, 4.2, 0.35)
        let drive = low * g(h, 0.5, 0.15) * (1.0 - lofted) * between(e, -12.0, 25.0, 2.5)
            * g(s, 85.0, 10.0) * g(L, 2.5, 0.3)
        let push = low * l(h, 1.95, 0.12) * l(f.apex, 2.7, 0.15)
            * between(s, 22.0, 90.0, 8.0) * between(L, 2.3, 6.2, 0.25) * l(d, 4.0, 0.3)
        return [("smash", smash), ("kill", kill), ("clear", clear), ("drop", drop), ("net", net),
                ("block", block), ("lift", lift), ("drive", drive), ("push", push)]
    }

    static func serveScore(_ f: FlightFeatures) -> Double {
        l(f.contactHeight, 1.35, 0.1) * g(f.contactDepth, 1.7, 0.2) * g(f.elevationDeg, -2.0, 4.0)
    }

    static func scores2D(depth: Double?, nextDepth: Double?, dt: Double?, overhead: Double, distM: Double?) -> Scores {
        let d = depth ?? .nan, L = nextDepth ?? .nan, T = dt ?? .nan
        let v = (distM != nil && dt != nil && dt! != 0) ? distM! / dt! : .nan
        let low = 1 - overhead
        return [
            ("smash", overhead * g(d, 2.5, 0.3) * l(T, 0.75, 0.1)),
            ("kill", overhead * l(d, 2.5, 0.3) * l(T, 0.6, 0.1)),
            ("clear", overhead * g(L, 4.0, 0.4) * g(T, 1.1, 0.15)),
            ("drop", overhead * g(d, 3.0, 0.3) * l(L, 3.0, 0.4) * g(T, 0.8, 0.15)),
            ("net", low * l(d, 2.3, 0.3) * l(L, 2.5, 0.4)),
            ("lift", low * g(L, 4.0, 0.4) * g(T, 1.1, 0.15)),
            ("drive", low * l(T, 0.8, 0.1) * g(v, 8.0, 1.5)),
            ("push", low * between(L, 2.0, 4.5, 0.4) * l(v, 8.0, 1.5)),
            ("block", low * g(d, 2.5, 0.3) * l(L, 2.5, 0.4)),
        ]
    }

    /// shots._normalise: argmax by (score, name); confidence = share × support.
    static func normalise(_ scores: Scores, floor: Double = 0.02) -> (ShotType, Double) {
        var best = scores[0]
        for kv in scores.dropFirst() where kv.1 > best.1 || (kv.1 == best.1 && kv.0 > best.0) { best = kv }
        let total = Py.sum(scores.map(\.1))
        if best.1 < floor { return (.unknown, 0.0) }
        let conf = (best.1 / (total + 0.05)) * Swift.min(1.0, 0.5 + best.1)
        return (ShotType(rawValue: best.0) ?? .unknown, Swift.min(Swift.max(conf, 0.0), 1.0))
    }

    // MARK: pose

    @inline(__always) static func kp(_ k: [Keypoint], _ i: Int) -> SIMD2<Double>? {
        k[i].conf >= kpConf ? SIMD2(k[i].x, k[i].y) : nil
    }

    /// shots.wrist_above_head: (P(overhead contact), raised wrist index or nil).
    public static func wristAboveHead(_ kps: [Keypoint]?) -> (Double, Int?) {
        guard let k = kps, k.count >= 17 else { return (0.5, nil) }
        let sh = [kp(k, LSH), kp(k, RSH)].compactMap { $0 }
        let hp = [kp(k, LHIP), kp(k, RHIP)].compactMap { $0 }
        if sh.isEmpty { return (0.5, nil) }
        let shY = Py.npMean(sh.map(\.y))
        var torso = hp.isEmpty ? 0.0 : Swift.abs(Py.npMean(hp.map(\.y)) - shY)
        let ys = k.filter { $0.conf >= kpConf }.map(\.y)
        let span = 1e-3 + 0.25 * (npMax(ys) - npMin(ys))
        if span > torso { torso = span }                   // Python max(torso, span)
        let nose = kp(k, NOSE)
        let headY = nose != nil ? nose!.y : shY - 0.35 * torso
        var best: Double? = nil, idx: Int? = nil
        for w in [LWR, RWR] {
            if let p = kp(k, w), best == nil || p.y < best! { best = p.y; idx = w }
        }
        guard let best else { return (0.5, nil) }
        return (gt((headY - best) / torso, 0.0, 0.15), idx)
    }

    /// np.max / np.min (NaN-propagating) of a non-empty array.
    private static func npMax(_ a: [Double]) -> Double {
        var m = a[0]
        for x in a.dropFirst() { if x.isNaN || m.isNaN { m = .nan } else if x > m { m = x } }
        return m
    }

    private static func npMin(_ a: [Double]) -> Double {
        var m = a[0]
        for x in a.dropFirst() { if x.isNaN || m.isNaN { m = .nan } else if x < m { m = x } }
        return m
    }

    /// `a @ b` for 2-vectors (OpenBLAS ddot = FMA chain).
    @inline(__always) static func dot2(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double {
        (a.x * b.x).addingProduct(a.y, b.y)
    }

    /// shots._body_right: (unit vector towards the anatomical right, width px) or nil (side-on).
    static func bodyRight(_ k: [Keypoint]) -> (SIMD2<Double>, Double)? {
        let sh = (kp(k, LSH), kp(k, RSH)), hp = (kp(k, LHIP), kp(k, RHIP))
        let ms: SIMD2<Double>? = sh.0 != nil && sh.1 != nil ? (sh.0! + sh.1!) / 2 : nil
        let mh: SIMD2<Double>? = hp.0 != nil && hp.1 != nil ? (hp.0! + hp.1!) / 2 : nil
        let torso = ms != nil && mh != nil ? Py.norm2(ms! - mh!) : 0.0
        var perp: SIMD2<Double>? = nil
        if torso > 1e-6 {
            let dvec = ms! - mh!
            perp = SIMD2(-dvec.y / torso, dvec.x / torso)
        }
        for (lft, rgt) in [sh, hp] {
            guard let lp = lft, let rp = rgt else { continue }
            let v = rp - lp
            let n = Py.norm2(v)
            if n < 1e-6 { continue }
            if let perp, Swift.abs(dot2(v, perp)) < minWidthRatio * torso { continue }
            return (SIMD2(v.x / n, v.y / n), n)
        }
        return nil
    }

    /// shots.stroke_side: overhead | forehand | backhand | unknown.
    public static func strokeSide(_ kps: [Keypoint]?, overhead: Double, rightHanded: Bool?) -> StrokeSide {
        if overhead >= 0.6 { return .overhead }
        guard let k = kps, let right = rightHanded else { return .unknown }
        if k.count < 17 { return .unknown }
        let br = bodyRight(k)
        let wr = kp(k, right ? RWR : LWR)
        let pts = [LSH, RSH, LHIP, RHIP].compactMap { kp(k, $0) }
        guard let (rv, width) = br, let wr, !pts.isEmpty else { return .unknown }
        let mean = SIMD2(Py.npMean(pts.map(\.x)), Py.npMean(pts.map(\.y)))
        var lateral = dot2(wr - mean, rv)
        if !right { lateral = -lateral }
        if Swift.abs(lateral) < deadZone * width { return .unknown }
        return lateral > 0 ? .forehand : .backhand
    }

    static func logit(_ p0: Double) -> Double {
        let p = Swift.min(Swift.max(p0, 1e-3), 1 - 1e-3)
        return log(p / (1 - p))
    }

    /// shots.overhead_prob: fuse contact height (overhead ≈ above 2.1 m) with pose, in log-odds.
    public static func overheadProb(contactHeight: Double?, poseOverhead: Double?) -> Double {
        var z = 0.0
        if let ch = contactHeight, ch.isFinite { z += logit(gt(ch, 2.1, 0.12)) }
        if let po = poseOverhead, po != 0.5 { z += 0.7 * logit(po) }
        return sig(z)
    }

    // MARK: hitter lookup

    /// shots._PlayerIndex: per-track sorted frames (tracks in first-seen order).
    struct PlayerIndex {
        var tracks: [(frames: [Int], obs: [PlayerObservation])] = []

        init(_ players: [PlayerObservation]?) {
            var order: [Int: Int] = [:]
            var groups: [[PlayerObservation]] = []
            for o in players ?? [] {
                if let gi = order[o.trackID] { groups[gi].append(o) } else {
                    order[o.trackID] = groups.count
                    groups.append([o])
                }
            }
            for gobs in groups {
                // list.sort(key=frame) is stable
                let s = gobs.enumerated().sorted { a, b in
                    a.element.frame < b.element.frame || (a.element.frame == b.element.frame && a.offset < b.offset)
                }.map(\.element)
                tracks.append((s.map(\.frame), s))
            }
        }

        func at(_ frame: Int, maxGap: Int = 8) -> [PlayerObservation] {
            var out: [PlayerObservation] = []
            for (frames, obs) in tracks {
                var lo = 0, hi = frames.count
                while lo < hi { let mid = (lo + hi) / 2; if frames[mid] < frame { lo = mid + 1 } else { hi = mid } }
                let i = lo
                let cand = [i - 1, i].filter { $0 >= 0 && $0 < obs.count }
                var j = cand[0]
                for c in cand.dropFirst() where Swift.abs(frames[c] - frame) < Swift.abs(frames[j] - frame) { j = c }
                if Swift.abs(frames[j] - frame) <= maxGap { out.append(obs[j]) }
            }
            return out
        }
    }

    /// Python min(iterable, key=...): first element with the smallest key (NaN keys never win later).
    private static func argminFirst<T>(_ xs: [T], _ key: (T) -> Double) -> T {
        var best = xs[0], bk = key(xs[0])
        for x in xs.dropFirst() { let k = key(x); if k < bk { best = x; bk = k } }
        return best
    }

    /// shots._pick_hitter.
    static func pickHitter(_ hit: HitEvent, _ contact: SIMD3<Double>?, _ cands: [PlayerObservation]) -> PlayerObservation? {
        if cands.isEmpty { return nil }
        if let tid = hit.playerTrackID {
            for o in cands where o.trackID == tid { return o }
        }
        var side = hit.hitter
        if side == .unknown, let c = contact { side = CourtModel.sideOf(x: c.x) }
        let sided = cands.filter { $0.side == side }
        let pool = sided.isEmpty ? cands : sided
        if let c = contact, pool.allSatisfy({ $0.footWorld != nil }) {
            let p = SIMD2(c.x, c.y)
            return argminFirst(pool) { Py.norm2($0.footWorld! - p) }
        }
        let u = hit.xyPx.x, v = hit.xyPx.y
        return argminFirst(pool) { o in
            let cx = Swift.min(Swift.max(u, o.bbox.x1), o.bbox.x2)
            let cy = Swift.min(Swift.max(v, o.bbox.y1), o.bbox.y2)
            return Py.hypot(u - cx, v - cy)
        }
    }

    // MARK: public API

    static func valid(_ seg: ShotSegment3D?) -> Bool {
        guard let seg else { return false }
        return seg.p0.x.isFinite && seg.p0.y.isFinite && seg.p0.z.isFinite
            && seg.v0.x.isFinite && seg.v0.y.isFinite && seg.v0.z.isFinite
    }

    /// shots.classify_segment: (type, confidence, features) of one 3-D flight.
    public static func classifySegment(_ seg: ShotSegment3D?, poseOverhead: Double? = nil, isFirst: Bool = false,
                                       tHit: Double? = nil) -> (ShotType, Double, FlightFeatures?) {
        guard valid(seg), let seg else { return (.unknown, 0.0, nil) }
        let (p0, v0) = contactState(seg, tHit: tHit)
        let f = flightFeatures(p0: p0, v0: v0, dragLength: seg.dragLength)
        let oh = overheadProb(contactHeight: f.contactHeight, poseOverhead: poseOverhead)
        if isFirst {
            let sv = serveScore(f)
            if sv >= 0.1 { return (.serve, Swift.min(1.0, 0.55 + 0.45 * sv), f) }
        }
        let (label, conf) = normalise(scores3D(f, oh))
        let fitQ = (seg.confidence != 0 && seg.confidence > 0) ? seg.confidence : 1.0
        return (label, conf * (0.5 + 0.5 * Swift.min(fitQ, 1.0)), f)
    }

    /// shots._classify_2d: fallback without a 3-D segment (confidence <= 0.45).
    static func classify2D(_ i: Int, _ shots: [ShotEventV2], _ pos: SIMD2<Double>?,
                           _ hitters: [PlayerObservation?], _ poseOverhead: Double) -> (ShotType, Double) {
        if i == 0 { return (.serve, 0.4) }
        let nxt: ShotEventV2? = i + 1 < shots.count ? shots[i + 1] : nil
        let nh: PlayerObservation? = i + 1 < hitters.count ? hitters[i + 1] : nil
        var npos: SIMD2<Double>? = nil
        if let nh, let fw = nh.footWorld { npos = fw }
        else if let nxt, valid(nxt.segment) { npos = SIMD2(nxt.segment!.p0.x, nxt.segment!.p0.y) }
        let depth = pos.map { Swift.abs($0.x) }
        let ndepth = npos.map { Swift.abs($0.x) }
        let dt = nxt.map { $0.hit.t - shots[i].hit.t }
        let dist = (pos != nil && npos != nil) ? Py.dist(pos!, npos!) : nil
        let scores = scores2D(depth: depth, nextDepth: ndepth, dt: dt, overhead: poseOverhead, distM: dist)
        if depth == nil && ndepth == nil && dt == nil { return (.unknown, 0.0) }
        let (label, conf) = normalise(scores, floor: 0.05)
        return (label, Swift.min(conf * 0.6, 0.45))
    }

    /// shots.classify_shots: fill shot_type, shot_type_confidence, hitter_pos_world,
    /// hit_height_m and stroke_side for every shot (returns a new rally).
    /// `handedness` (track_id → right-handed) overrides per-rally inference.
    ///
    /// With `learned` (the default) the rule-based verdict is handed to `StrokeClassifier`, which
    /// replaces the shot *type* wherever the bundled BST model can rule: pass `meta` and `track`
    /// (the same stages the hit came from) for that; without them this is the pure port.
    /// `learned: false` skips it (the A/B switch, and what the port's parity tests use).
    public static func classifyShots(_ rally: RallyV2, camera: CameraModel?, players: [PlayerObservation]?,
                                     handedness: [Int: Bool]? = nil, meta: VideoMeta? = nil, track: Track2D? = nil,
                                     learned: Bool = true) -> RallyV2 {
        let shots = rally.shots.enumerated().sorted { a, b in
            a.element.hit.frame < b.element.hit.frame || (a.element.hit.frame == b.element.hit.frame && a.offset < b.offset)
        }.map(\.element)
        let contacts: [SIMD3<Double>?] = shots.map { s in
            valid(s.segment) ? contactState(s.segment!, tHit: s.hit.t).p : nil
        }
        let idx = PlayerIndex(players)
        let hitters: [PlayerObservation?] = shots.indices.map { i in
            pickHitter(shots[i].hit, contacts[i], idx.at(shots[i].hit.frame))
        }
        let poses = hitters.map { wristAboveHead($0?.keypoints) }

        var raisedOrder: [Int] = []
        var raised: [Int: [Int]] = [:]
        for i in shots.indices {
            let hz = contacts[i]?.z
            if let h = hitters[i], let w = poses[i].1,
               overheadProb(contactHeight: hz, poseOverhead: poses[i].0) > 0.7 {
                if raised[h.trackID] == nil { raisedOrder.append(h.trackID) }
                raised[h.trackID, default: []].append(w)
            }
        }
        var hand: [Int: Bool] = [:]
        for tid in raisedOrder {
            let ws = raised[tid]!
            hand[tid] = ws.filter { $0 == RWR }.count >= ws.filter { $0 == LWR }.count
        }
        for (k, v) in handedness ?? [:] { hand[k] = v }

        var out: [ShotEventV2] = []
        for i in shots.indices {
            var s = shots[i]
            let c = contacts[i], h = hitters[i], po = poses[i].0
            let seg = c != nil ? s.segment : nil
            var pos: SIMD2<Double>? = nil
            if let h, let fw = h.footWorld { pos = fw } else if let c { pos = SIMD2(c.x, c.y) }
            let height = c?.z
            let label: ShotType, conf: Double
            if let seg {
                (label, conf, _) = classifySegment(seg, poseOverhead: po, isFirst: i == 0, tHit: s.hit.t)
            } else {
                (label, conf) = classify2D(i, shots, pos, hitters, po)
            }
            let oh = overheadProb(contactHeight: height, poseOverhead: po)
            let right: Bool? = h != nil ? (hand[h!.trackID] ?? true) : nil
            let side = strokeSide(h?.keypoints, overhead: oh, rightHanded: right)
            s.shotType = label
            s.shotTypeConfidence = Py.round(conf, 4)
            s.hitterPosWorld = pos
            s.hitHeightM = height
            s.strokeSide = side
            out.append(s)
        }
        var r = rally
        r.shots = out
        return learned ? StrokeClassifier.refine(r, meta: meta, track: track, players: players) : r
    }
}

/// shots.classify_shots — free-function spelling of the public API.
public func classifyShots(_ rally: RallyV2, camera: CameraModel?, players: [PlayerObservation]?,
                          handedness: [Int: Bool]? = nil, meta: VideoMeta? = nil, track: Track2D? = nil,
                          learned: Bool = true) -> RallyV2 {
    ShotClassifier.classifyShots(rally, camera: camera, players: players, handedness: handedness,
                                 meta: meta, track: track, learned: learned)
}
