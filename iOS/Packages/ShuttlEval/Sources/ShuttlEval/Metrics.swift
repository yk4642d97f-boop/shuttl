import Foundation
import ShuttlCore

// Port of backend/app/services/tracking/metrics.py — same matching rules, tolerances,
// aggregates, output keys (and key order), int/float types and NaN behaviour.
//
// Per-frame detection classification follows TrackNetV3 (test.py `evaluate`):
//   pred ∧ truth: TP if ‖pred − truth‖ ≤ tol else FP1;  pred ∧ ¬truth: FP2;
//   ¬pred ∧ truth: FN;  neither: TN.
//   precision = TP/(TP+FP1+FP2), recall = TP/(TP+FN), accuracy = (TP+TN)/N.
// TrackNetV3 measures 4 px on the 512×288 grid with integer-truncated coordinates;
// `scaledTolerance(4, (1280, 720))` = 10 px is the full-resolution equivalent.
//
// Python duck-types its inputs; here the plain records below (EvalHit / EvalSegment /
// EvalRally) play that role, with converters from the ShuttlCore pipeline types.

// MARK: - Records

/// A frame event for hit_metrics: `frame` + `hitter` (+ the truth `shot_type`).
public struct EvalHit: Sendable, Equatable, Codable {
    public var frame: Int
    public var hitter: String
    public var shotType: String?

    public init(frame: Int, hitter: String = "unknown", shotType: String? = nil) {
        self.frame = frame; self.hitter = hitter; self.shotType = shotType
    }
    public init(_ h: HitEvent) { self.init(frame: h.frame, hitter: h.hitter.rawValue) }

    enum CodingKeys: String, CodingKey { case frame, hitter, shotType = "shot_type" }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        frame = try c.decode(Int.self, forKey: .frame)
        hitter = try c.decodeIfPresent(String.self, forKey: .hitter) ?? "unknown"
        shotType = try c.decodeIfPresent(String.self, forKey: .shotType)
    }
}

/// A flight for segment_metrics (benchmark `_seg_d` / `_true_shot_d`).
/// `nil` = Python None; a NaN-valued landing point is kept as NaN (the in/out fallback
/// in benchmark `_call` distinguishes the two).
public struct EvalSegment: Sendable, Equatable, Codable {
    public var startFrame: Int
    public var p0: SIMD3<Double>
    public var v0: SIMD3<Double>
    public var netCrossingHeight: Double?
    public var landingXY: SIMD2<Double>?

    public init(startFrame: Int, p0: SIMD3<Double>, v0: SIMD3<Double>, netCrossingHeight: Double? = nil,
                landingXY: SIMD2<Double>? = nil) {
        self.startFrame = startFrame; self.p0 = p0; self.v0 = v0
        self.netCrossingHeight = netCrossingHeight; self.landingXY = landingXY
    }

    public init(_ s: ShotSegment3D) {
        self.init(startFrame: s.startFrame, p0: s.p0, v0: s.v0, netCrossingHeight: s.netCrossingHeight,
                  landingXY: s.landingXY)
    }

    enum CodingKeys: String, CodingKey {
        case startFrame = "start_frame", frame, p0, v0, netCrossingHeight = "net_crossing_height", landingXY = "landing_xy"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        startFrame = try c.decodeIfPresent(Int.self, forKey: .startFrame) ?? c.decodeIfPresent(Int.self, forKey: .frame) ?? -1
        p0 = SIMD3(row: try c.decodeNaNableArray(forKey: .p0) ?? [.nan, .nan, .nan])
        v0 = SIMD3(row: try c.decodeNaNableArray(forKey: .v0) ?? [.nan, .nan, .nan])
        netCrossingHeight = try c.decodeIfPresent(Double.self, forKey: .netCrossingHeight)
        landingXY = try c.decodeNaNableArray(forKey: .landingXY).map(SIMD2.init(row:))
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(startFrame, forKey: .startFrame)
        try c.encodeNaNableArray(p0.row, forKey: .p0)
        try c.encodeNaNableArray(v0.row, forKey: .v0)
        try c.encodeIfPresent(netCrossingHeight, forKey: .netCrossingHeight)
        if let landingXY { try c.encodeNaNableArray(landingXY.row, forKey: .landingXY) }
    }
}

/// A rally for rally_metrics (benchmark `_rally_d`). `margin` falls back to
/// `in_out_margin_m` when the key is absent, like `_get(r, "margin", _get(r, "in_out_margin_m"))`.
public struct EvalRally: Sendable, Equatable, Codable {
    public var startFrame: Int
    public var endFrame: Int
    public var endReason: String
    public var winner: String
    public var landingXY: SIMD2<Double>?
    public var margin: Double?

    public init(startFrame: Int, endFrame: Int, endReason: String = "unknown", winner: String = "unknown",
                landingXY: SIMD2<Double>? = nil, margin: Double? = nil) {
        self.startFrame = startFrame; self.endFrame = endFrame; self.endReason = endReason
        self.winner = winner; self.landingXY = landingXY; self.margin = margin
    }

    public init(_ r: RallyV2) {
        self.init(startFrame: r.startFrame, endFrame: r.endFrame, endReason: r.endReason.rawValue,
                  winner: r.winner.rawValue, landingXY: r.landingXY, margin: r.inOutMarginM)
    }

    enum CodingKeys: String, CodingKey {
        case startFrame = "start_frame", endFrame = "end_frame", endReason = "end_reason", winner
        case landingXY = "landing_xy", margin, inOutMarginM = "in_out_margin_m"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        startFrame = try c.decode(Int.self, forKey: .startFrame)
        endFrame = try c.decode(Int.self, forKey: .endFrame)
        endReason = try c.decodeIfPresent(String.self, forKey: .endReason) ?? "unknown"
        winner = try c.decodeIfPresent(String.self, forKey: .winner) ?? "unknown"
        landingXY = try c.decodeNaNableArray(forKey: .landingXY).map(SIMD2.init(row:))
        margin = c.contains(.margin) ? try c.decodeIfPresent(Double.self, forKey: .margin)
                                     : try c.decodeIfPresent(Double.self, forKey: .inOutMarginM)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(startFrame, forKey: .startFrame)
        try c.encode(endFrame, forKey: .endFrame)
        try c.encode(endReason, forKey: .endReason)
        try c.encode(winner, forKey: .winner)
        if let landingXY { try c.encodeNaNableArray(landingXY.row, forKey: .landingXY) } else { try c.encodeNil(forKey: .landingXY) }
        if let margin { try c.encode(margin, forKey: .margin) } else { try c.encodeNil(forKey: .margin) }
    }
}

public enum MetricsError: Error, CustomStringConvertible, Equatable {
    case lengthMismatch(String)
    case needsImageSize
    public var description: String {
        switch self {
        case .lengthMismatch(let m): return "ValueError: length mismatch: \(m)"
        case .needsImageSize: return "ValueError: tracknet_int needs image_size"
        }
    }
}

// MARK: - Metrics

public enum Metrics {
    public static let refSize = ImageSize(width: 512, height: 288)

    // MARK: helpers

    /// `_ratio(a, b)`: a / b, NaN when b == 0.
    public static func ratio(_ a: Double, _ b: Double) -> Double { b != 0 ? a / b : .nan }
    public static func ratio(_ a: Int, _ b: Int) -> Double { ratio(Double(a), Double(b)) }

    /// `_f1(p, r)`: 2pr/(p+r) if p+r > 0 and finite else 0.
    public static func f1(_ p: Double, _ r: Double) -> Double {
        let s = p + r
        return s > 0 && s.isFinite ? 2 * p * r / s : 0
    }

    /// `_stats(err, prefix)`: mean / median / p95 / n of the finite values.
    public static func stats(_ err: [Double], prefix: String = "err") -> PyDict {
        let e = err.filter(\.isFinite)
        guard !e.isEmpty else {
            return ["\(prefix)_mean": .float(.nan), "\(prefix)_median": .float(.nan),
                    "\(prefix)_p95": .float(.nan), "\(prefix)_n": .int(0)]
        }
        return ["\(prefix)_mean": .float(NP.mean(e)), "\(prefix)_median": .float(NP.median(e)),
                "\(prefix)_p95": .float(NP.percentile(e, 95)), "\(prefix)_n": .int(e.count)]
    }

    /// A tolerance defined on the ref grid (TrackNet: 4 px @ 512×288) in image px.
    public static func scaledTolerance(_ tolRef: Double = 4.0, imageSize: ImageSize = ImageSize(width: 1280, height: 720),
                                       refSize: ImageSize = refSize) -> Double {
        tolRef * Double(imageSize.width) / Double(refSize.width)
    }

    /// Top-scoring candidate per frame → (xy with NaN below threshold, score).
    public static func candidatesToXY(_ cands: [[ShuttleCandidate]], scoreThresh: Double = 0.5)
        -> (xy: [SIMD2<Double>], score: [Double]) {
        var xy = [SIMD2<Double>](repeating: SIMD2(.nan, .nan), count: cands.count)
        var sc = [Double](repeating: 0, count: cands.count)
        for (i, cs) in cands.enumerated() {
            guard var best = cs.first else { continue }
            for c in cs.dropFirst() where c.score > best.score { best = c }   // Python max: first maximum
            sc[i] = best.score
            if best.score >= scoreThresh { xy[i] = SIMD2(best.x, best.y) }
        }
        return (xy, sc)
    }

    static func truthVis(_ t: [SIMD2<Double>], _ vis: [Bool]?) throws -> [Bool] {
        let ok = t.map(\.isFinite)
        guard let vis else { return ok }
        guard vis.count == ok.count else { throw MetricsError.lengthMismatch("true_vis \(vis.count) vs truth \(ok.count)") }
        return zip(ok, vis).map { $0 && $1 }
    }

    static func evalMask(_ mask: [Bool]?, _ n: Int) throws -> [Bool] {
        guard let mask else { return [Bool](repeating: true, count: n) }
        guard mask.count == n else { throw MetricsError.lengthMismatch("mask \(mask.count) vs truth \(n)") }
        return mask
    }

    // MARK: per-frame detection / track

    /// TrackNetV3-style TP/TN/FP1/FP2/FN counts and P/R/F1 + pixel error stats.
    /// pred: px with NaN where nothing is predicted. `tracknetInt`: compare int-truncated
    /// coordinates on the 512×288 grid (`tolPx` is then on that grid, i.e. use 4).
    public static func detectionMetrics(_ pred: [SIMD2<Double>], _ truth: [SIMD2<Double>], vis: [Bool]? = nil,
                                        tolPx: Double = 10, mask: [Bool]? = nil, tracknetInt: Bool = false,
                                        imageSize: ImageSize? = nil) throws -> PyDict {
        let n = truth.count
        guard pred.count == n else { throw MetricsError.lengthMismatch("pred \(pred.count) vs truth \(n)") }
        let tv = try truthVis(truth, vis)
        let pv = pred.map(\.isFinite)
        let m = try evalMask(mask, n)
        var d = [Double](repeating: .nan, count: n)
        if tracknetInt {
            guard let size = imageSize else { throw MetricsError.needsImageSize }
            let s0 = Double(refSize.width) / Double(size.width), s1 = Double(refSize.height) / Double(size.height)
            for i in 0..<n {
                if pv[i] && tv[i] {
                    let dx = (pred[i].x * s0).rounded(.towardZero) - (truth[i].x * s0).rounded(.towardZero)
                    let dy = (pred[i].y * s1).rounded(.towardZero) - (truth[i].y * s1).rounded(.towardZero)
                    d[i] = NP.rowNorm(dx, dy)
                } else {
                    d[i] = .infinity
                }
            }
        } else {
            for i in 0..<n { d[i] = NP.rowNorm(pred[i].x - truth[i].x, pred[i].y - truth[i].y) }
        }
        var tp = 0, fp1 = 0, fp2 = 0, fn = 0, tn = 0, nm = 0
        var err: [Double] = []
        for i in 0..<n where m[i] {
            nm += 1
            switch (pv[i], tv[i]) {
            case (true, true):
                if d[i] <= tolPx { tp += 1 } else { fp1 += 1 }
                err.append(NP.rowNorm(pred[i].x - truth[i].x, pred[i].y - truth[i].y))
            case (true, false): fp2 += 1
            case (false, true): fn += 1
            case (false, false): tn += 1
            }
        }
        let prec = ratio(tp, tp + fp1 + fp2), rec = ratio(tp, tp + fn)
        var out: PyDict = ["tp": .int(tp), "tn": .int(tn), "fp1": .int(fp1), "fp2": .int(fp2), "fn": .int(fn),
                           "n_frames": .int(nm), "precision": .float(prec), "recall": .float(rec),
                           "f1": .float(f1(prec, rec)), "accuracy": .float(ratio(tp + tn, nm)),
                           "miss_rate": .float(ratio(fn, tp + fn)), "tol_px": .float(tolPx)]
        out.update(stats(err))
        return out
    }

    /// detection_metrics on per-frame candidate lists (top-1 at score ≥ 0.5, like `_as_xy`).
    public static func detectionMetrics(_ cands: [[ShuttleCandidate]], _ truth: [SIMD2<Double>], vis: [Bool]? = nil,
                                        tolPx: Double = 10, mask: [Bool]? = nil, tracknetInt: Bool = false,
                                        imageSize: ImageSize? = nil) throws -> PyDict {
        try detectionMetrics(candidatesToXY(cands).xy, truth, vis: vis, tolPx: tolPx, mask: mask,
                             tracknetInt: tracknetInt, imageSize: imageSize)
    }

    /// detection_metrics on a Track2D (filled frames count as predictions).
    public static func detectionMetrics(_ track: Track2D, _ truth: [SIMD2<Double>], vis: [Bool]? = nil,
                                        tolPx: Double = 10, mask: [Bool]? = nil, tracknetInt: Bool = false,
                                        imageSize: ImageSize? = nil) throws -> PyDict {
        try detectionMetrics(track.xy, truth, vis: vis, tolPx: tolPx, mask: mask, tracknetInt: tracknetInt, imageSize: imageSize)
    }

    /// detection_metrics plus jitter_px / accel_rms_px / jumps / wrong_frames and, when
    /// `interpolated` marks gap-filled frames, interp_frames / interp_within_tol / interp_err_*.
    /// Pass `interpolated: nil` for a plain array (Python only reads it from a Track2D).
    public static func trackMetrics(_ pred: [SIMD2<Double>], interpolated: [Bool]?, _ truth: [SIMD2<Double>],
                                    vis: [Bool]? = nil, tolPx: Double = 10, jumpFactor: Double = 3,
                                    mask: [Bool]? = nil) throws -> PyDict {
        var out = try detectionMetrics(pred, truth, vis: vis, tolPx: tolPx, mask: mask)
        let n = truth.count
        let tv = try truthVis(truth, vis)
        let pv = pred.map(\.isFinite)
        let m = try evalMask(mask, n)
        let ex = (0..<n).map { pred[$0].x - truth[$0].x }, ey = (0..<n).map { pred[$0].y - truth[$0].y }
        let en = (0..<n).map { NP.rowNorm(ex[$0], ey[$0]) }
        let thr = jumpFactor * tolPx
        let ok = (0..<n).map { pv[$0] && tv[$0] && m[$0] && en[$0] <= thr }
        var jit: [Double] = [], acc: [Double] = []
        if n >= 3 {
            for k in 0..<(n - 2) {
                if ok[k] && ok[k + 1] && ok[k + 2] {
                    let dx = ex[k + 2] - 2 * ex[k + 1] + ex[k], dy = ey[k + 2] - 2 * ey[k + 1] + ey[k]
                    jit.append(dx * dx + dy * dy)
                }
                if pv[k] && pv[k + 1] && pv[k + 2] && m[k + 1] {
                    let ax = pred[k + 2].x - 2 * pred[k + 1].x + pred[k].x
                    let ay = pred[k + 2].y - 2 * pred[k + 1].y + pred[k].y
                    acc.append(ax * ax + ay * ay)
                }
            }
        }
        out["jitter_px"] = .float(jit.isEmpty ? .nan : NP.mean(jit).squareRoot())
        out["accel_rms_px"] = .float(acc.isEmpty ? .nan : NP.mean(acc).squareRoot())
        let wrong = (0..<n).map { pv[$0] && m[$0] && (!tv[$0] || en[$0] > thr) }
        var jumps = 0
        for i in 0..<n where wrong[i] && (i == 0 || !wrong[i - 1]) { jumps += 1 }
        out["jumps"] = .int(jumps)
        out["wrong_frames"] = .int(wrong.lazy.filter { $0 }.count)
        let interp = interpolated ?? [Bool](repeating: false, count: n)
        if interp.contains(true) {
            guard interp.count == n else { throw MetricsError.lengthMismatch("interpolated \(interp.count) vs truth \(n)") }
            var within = 0, cnt = 0
            var dsub: [Double] = []
            for i in 0..<n where interp[i] && pv[i] && tv[i] && m[i] {
                cnt += 1
                if en[i] <= tolPx { within += 1 }
                dsub.append(en[i])
            }
            out["interp_frames"] = .int(interp.lazy.filter { $0 }.count)
            out["interp_within_tol"] = .float(ratio(within, cnt))
            out.update(stats(dsub, prefix: "interp_err"))
        }
        return out
    }

    /// track_metrics on a Track2D (uses its `interpolated` flags).
    public static func trackMetrics(_ track: Track2D, _ truth: [SIMD2<Double>], vis: [Bool]? = nil, tolPx: Double = 10,
                                    jumpFactor: Double = 3, mask: [Bool]? = nil) throws -> PyDict {
        try trackMetrics(track.xy, interpolated: track.interpolated, truth, vis: vis, tolPx: tolPx,
                         jumpFactor: jumpFactor, mask: mask)
    }

    // MARK: hits

    /// `_match_events`: one-to-one matching minimising |Δframe| (candidate pairs in
    /// row-major order, stable-sorted by distance, taken greedily).
    public static func matchEvents(_ pf: [Int], _ tf: [Int], tol: Double) -> [(Int, Int)] {
        guard !pf.isEmpty, !tf.isEmpty else { return [] }
        var cand: [(i: Int, j: Int, d: Double)] = []
        for i in pf.indices { for j in tf.indices {
            let d = Double(abs(pf[i] - tf[j]))
            if d <= tol { cand.append((i, j, d)) }
        } }
        let order = cand.indices.sorted { a, b in cand[a].d < cand[b].d || (cand[a].d == cand[b].d && a < b) }
        var usedP = Set<Int>(), usedT = Set<Int>(), out: [(Int, Int)] = []
        for k in order {
            let (i, j, _) = cand[k]
            if usedP.contains(i) || usedT.contains(j) { continue }
            usedP.insert(i); usedT.insert(j)
            out.append((i, j))
        }
        return out
    }

    /// Event F1 at ±tolFrames, mean |Δframe| of matches, hitter side accuracy
    /// ('unknown' counts as wrong — hitter_acc_known excludes it).
    public static func hitMetrics(_ pred: [EvalHit], _ truth: [EvalHit], tolFrames: Int = 3) -> PyDict {
        let pf = pred.map(\.frame), tf = truth.map(\.frame)
        let pairs = matchEvents(pf, tf, tol: Double(tolFrames))
        let tp = pairs.count
        let prec = ratio(tp, pf.count), rec = ratio(tp, tf.count)
        var out: PyDict = ["tp": .int(tp), "fp": .int(pf.count - tp), "fn": .int(tf.count - tp),
                           "n_pred": .int(pf.count), "n_true": .int(tf.count), "precision": .float(prec),
                           "recall": .float(rec), "f1": .float(f1(prec, rec)), "tol_frames": .int(tolFrames)]
        let off = pairs.map { Double(pf[$0.0] - tf[$0.1]) }
        out["offset_abs_mean"] = .float(tp > 0 ? NP.mean(off.map(abs)) : .nan)
        out["offset_mean"] = .float(tp > 0 ? NP.mean(off) : .nan)
        let ok = pairs.map { pred[$0.0].hitter == truth[$0.1].hitter }
        let known = pairs.filter { pred[$0.0].hitter != "unknown" }.map { pred[$0.0].hitter == truth[$0.1].hitter }
        out["hitter_acc"] = .float(ok.isEmpty ? .nan : NP.mean(ok))
        out["hitter_acc_known"] = .float(known.isEmpty ? .nan : NP.mean(known))
        return out
    }

    public static func hitMetrics(_ pred: [HitEvent], _ truth: [EvalHit], tolFrames: Int = 3) -> PyDict {
        hitMetrics(pred.map(EvalHit.init), truth, tolFrames: tolFrames)
    }

    // MARK: 3-D

    /// Euclidean floor error (m) for paired landing points (nil / NaN skipped).
    public static func landingMetrics(_ pred: [SIMD2<Double>?], _ truth: [SIMD2<Double>?]) -> PyDict {
        var err: [Double] = []
        for (p, t) in zip(pred, truth) {
            guard let p, let t else { continue }
            if p.isFinite && t.isFinite { err.append(NP.norm(p.x - t.x, p.y - t.y)) }
        }
        return stats(err, prefix: "landing_err_m")
    }

    /// {'labels': [...], 'matrix': rows = truth, cols = prediction}.
    public static func confusionMatrix(_ pred: [String], _ truth: [String], labels: [String]? = nil) -> PyDict {
        let labs = labels ?? NP.pySorted(Set(truth).union(pred))
        var idx: [String: Int] = [:]
        for (i, l) in labs.enumerated() { idx[l] = i }
        var mat = [[Int]](repeating: [Int](repeating: 0, count: labs.count), count: labs.count)
        for (p, t) in zip(pred, truth) {
            if let pi = idx[p], let ti = idx[t] { mat[ti][pi] += 1 }
        }
        return ["labels": .strings(labs), "matrix": .list(mat.map { .ints($0) })]
    }

    /// Call accuracy ('in'|'out'|'net'; anything else = 'unknown' = wrong), confusion
    /// matrix, and accuracy on close calls (|true margin| < closeM).
    public static func inOutMetrics(_ predCalls: [String], _ trueCalls: [String], margins: [Double?]? = nil,
                                    closeM: Double = 0.2) throws -> PyDict {
        let pred = predCalls.map { ["in", "out", "net"].contains($0) ? $0 : "unknown" }
        let ok = zip(pred, trueCalls).map { $0 == $1 }
        var out: PyDict = ["n": .int(trueCalls.count), "accuracy": .float(ok.isEmpty ? .nan : NP.mean(ok)),
                           "confusion": .dict(confusionMatrix(pred, trueCalls, labels: ["in", "out", "net", "unknown"]))]
        let decided = pred.map { $0 != "unknown" }
        if decided.contains(true) {
            guard decided.count == ok.count else { throw MetricsError.lengthMismatch("boolean index \(decided.count) vs \(ok.count)") }
            out["accuracy_decided"] = .float(NP.mean(zip(ok, decided).filter(\.1).map(\.0)))
        } else {
            out["accuracy_decided"] = .float(.nan)
        }
        out["coverage"] = .float(decided.isEmpty ? .nan : NP.mean(decided))
        if let margins {
            let close = margins.map { m -> Bool in guard let m, m.isFinite else { return false }; return abs(m) < closeM }
            out["n_close"] = .int(close.lazy.filter { $0 }.count)
            if close.contains(true) {
                guard close.count == ok.count else { throw MetricsError.lengthMismatch("boolean index \(close.count) vs \(ok.count)") }
                out["accuracy_close"] = .float(NP.mean(zip(ok, close).filter(\.1).map(\.0)))
            } else {
                out["accuracy_close"] = .float(.nan)
            }
        }
        return out
    }

    /// Python `max(a, b)` (returns a unless b > a; NaN-first stays NaN).
    @inline(__always) static func pyMax(_ a: Double, _ b: Double) -> Double { b > a ? b : a }

    /// Match predicted flights to true shots by start frame; errors on initial speed
    /// (km/h), contact height (m), contact position (m), net-crossing height (m) and
    /// landing point (when both present).
    public static func segmentMetrics(_ pred: [EvalSegment], _ truth: [EvalSegment], tolFrames: Int = 4) -> PyDict {
        let pf = pred.map(\.startFrame), tf = truth.map(\.startFrame)
        let pairs = matchEvents(pf, tf, tol: Double(tolFrames))
        var out: PyDict = ["n_pred": .int(pf.count), "n_true": .int(tf.count), "matched": .int(pairs.count),
                           "recall": .float(ratio(pairs.count, tf.count))]
        var sp: [Double] = [], spRel: [Double] = [], hz: [Double] = [], pos: [Double] = []
        var znet: [Double] = [], land: [Double] = []
        for (i, j) in pairs {
            let ps = pred[i], ts = truth[j]
            let tsKmh = NP.norm(ts.v0) * 3.6
            sp.append(abs(NP.norm(ps.v0) * 3.6 - tsKmh))
            spRel.append(sp[sp.count - 1] / pyMax(tsKmh, 1e-6))
            hz.append(abs(ps.p0.z - ts.p0.z))
            pos.append(NP.norm(ps.p0 - ts.p0))
            if let a = ps.netCrossingHeight, let b = ts.netCrossingHeight, a.isFinite, b.isFinite { znet.append(abs(a - b)) }
            if let a = ps.landingXY, let b = ts.landingXY, a.isFinite, b.isFinite { land.append(NP.norm(a.x - b.x, a.y - b.y)) }
        }
        out.update(stats(sp, prefix: "speed_err_kmh"))
        out.update(stats(spRel, prefix: "speed_err_rel"))
        out.update(stats(hz, prefix: "height_err_m"))
        out.update(stats(pos, prefix: "contact_err_m"))
        out.update(stats(znet, prefix: "net_height_err_m"))
        out.update(stats(land, prefix: "landing_err_m"))
        return out
    }

    public static func segmentMetrics(_ pred: [ShotSegment3D], _ truth: [EvalSegment], tolFrames: Int = 4) -> PyDict {
        segmentMetrics(pred.map(EvalSegment.init), truth, tolFrames: tolFrames)
    }

    // MARK: shot type / rally

    public static func shotTypeMetrics(_ pred: [String], _ truth: [String], labels: [String]? = nil) -> PyDict {
        let ok = zip(pred, truth).map { $0 == $1 }
        let cmx = confusionMatrix(pred, truth, labels: labels)
        let labs = cmx["labels"]!.list!.map { $0.string! }
        let mat = cmx["matrix"]!.list!.map { $0.list!.map { Double($0.int!) } }
        var per = PyDict()
        for (k, l) in labs.enumerated() {
            let n = NP.sum(mat[k])
            if n != 0 { per[l] = .float(mat[k][k] / n) }
        }
        return ["n": .int(ok.count), "accuracy": .float(ok.isEmpty ? .nan : NP.mean(ok)),
                "per_class_recall": .dict(per), "confusion": .dict(cmx)]
    }

    public static func intervalIoU(_ a: (Int, Int), _ b: (Int, Int)) -> Double {
        let inter = min(a.1, b.1) - max(a.0, b.0) + 1
        let union = max(a.1, b.1) - min(a.0, b.0) + 1
        return union > 0 ? Double(max(inter, 0)) / Double(union) : 0
    }

    /// Rally segmentation F1 at interval IoU ≥ iouThresh (greedy by IoU), mean IoU,
    /// start/end offsets, end-reason / winner accuracy over matches, end-reason confusion,
    /// landing error of matched rallies and end-reason accuracy on close calls (|margin| < 0.2 m).
    public static func rallyMetrics(_ pred: [EvalRally], _ truth: [EvalRally], iouThresh: Double = 0.5) -> PyDict {
        let P = pred.map { ($0.startFrame, $0.endFrame) }, T = truth.map { ($0.startFrame, $0.endFrame) }
        var cand: [(iou: Double, i: Int, j: Int)] = []
        for (i, p) in P.enumerated() { for (j, t) in T.enumerated() { cand.append((intervalIoU(p, t), i, j)) } }
        cand.sort { a, b in                              // sorted(..., reverse=True) on (iou, i, j)
            if a.iou != b.iou { return a.iou > b.iou }
            if a.i != b.i { return a.i > b.i }
            return a.j > b.j
        }
        var up = Set<Int>(), ut = Set<Int>(), pairs: [(i: Int, j: Int, iou: Double)] = []
        for c in cand {
            if c.iou < iouThresh { break }
            if up.contains(c.i) || ut.contains(c.j) { continue }
            up.insert(c.i); ut.insert(c.j)
            pairs.append((c.i, c.j, c.iou))
        }
        let tp = pairs.count
        let prec = ratio(tp, P.count), rec = ratio(tp, T.count)
        var out: PyDict = ["n_pred": .int(P.count), "n_true": .int(T.count), "tp": .int(tp), "precision": .float(prec),
                           "recall": .float(rec), "f1": .float(f1(prec, rec))]
        let has = !pairs.isEmpty
        out["iou_mean"] = .float(has ? NP.mean(pairs.map(\.iou)) : .nan)
        out["start_offset_abs_mean"] = .float(has ? NP.mean(pairs.map { Double(abs(P[$0.i].0 - T[$0.j].0)) }) : .nan)
        out["end_offset_abs_mean"] = .float(has ? NP.mean(pairs.map { Double(abs(P[$0.i].1 - T[$0.j].1)) }) : .nan)
        let pe = pairs.map { pred[$0.i].endReason }, te = pairs.map { truth[$0.j].endReason }
        let pw = pairs.map { pred[$0.i].winner }, tw = pairs.map { truth[$0.j].winner }
        let okE = zip(pe, te).map { $0 == $1 }
        out["end_reason_acc"] = .float(has ? NP.mean(okE) : .nan)
        out["winner_acc"] = .float(has ? NP.mean(zip(pw, tw).map { $0 == $1 }) : .nan)
        out["end_reason_confusion"] = .dict(confusionMatrix(pe, te, labels: ["in", "out", "net", "unknown"]))
        out.update(landingMetrics(pairs.map { pred[$0.i].landingXY }, pairs.map { truth[$0.j].landingXY }))
        let close = pairs.map { p -> Bool in
            guard let m = truth[p.j].margin, m.isFinite else { return false }
            return abs(m) < 0.2
        }
        out["n_close"] = .int(close.lazy.filter { $0 }.count)
        out["accuracy_close"] = .float(close.contains(true) ? NP.mean(zip(okE, close).filter(\.1).map(\.0)) : .nan)
        return out
    }

    public static func rallyMetrics(_ pred: [RallyV2], _ truth: [EvalRally], iouThresh: Double = 0.5) -> PyDict {
        rallyMetrics(pred.map(EvalRally.init), truth, iouThresh: iouThresh)
    }

    // MARK: reporting

    public static let tableKeys = ["f1", "precision", "recall", "accuracy", "err_median", "err_p95", "jitter_px",
                                   "jumps", "offset_abs_mean", "hitter_acc", "speed_err_kmh_median",
                                   "height_err_m_median", "landing_err_m_median", "accuracy_close",
                                   "end_reason_acc", "winner_acc", "iou_mean", "n_frames", "n_true", "n"]

    /// Compact text table: one row per stage, selected metrics (ints as `k=3`, finite
    /// floats as `k=0.975`, everything else omitted).
    public static func formatTable(_ results: PyDict, keys: [String] = tableKeys) -> String {
        var rows: [String] = []
        for (stage, res) in results {
            guard let r = res.dict else {
                rows.append("\(PyFormat.ljust(stage, 22)) \(PyFormat.str(res))")
                continue
            }
            if let why = r["skipped"] {
                rows.append("\(PyFormat.ljust(stage, 22)) SKIPPED (\(PyFormat.str(why)))")
                continue
            }
            var cells: [String] = []
            for k in keys {
                switch r[k] {
                case .int(let v): cells.append("\(k)=\(v)")
                case .float(let v) where v.isFinite: cells.append("\(k)=\(PyFormat.fixed(v, 3))")
                default: break
                }
            }
            rows.append("\(PyFormat.ljust(stage, 22)) " + cells.joined(separator: "  "))
        }
        return rows.joined(separator: "\n")
    }
}
