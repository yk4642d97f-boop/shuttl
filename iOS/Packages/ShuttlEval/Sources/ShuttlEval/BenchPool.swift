import Foundation
import ShuttlCore

/// Insertion-ordered String-keyed map (Python dict / defaultdict order semantics).
public struct OrderedMap<Value> {
    public private(set) var keys: [String] = []
    private var dict: [String: Value] = [:]

    public init() {}

    public subscript(key: String) -> Value? {
        get { dict[key] }
        set {
            guard let v = newValue else {
                if dict.removeValue(forKey: key) != nil { keys.removeAll { $0 == key } }
                return
            }
            if dict.updateValue(v, forKey: key) == nil { keys.append(key) }
        }
    }

    /// defaultdict access: creates `make()` at the end if absent, then mutates in place.
    public mutating func modify(_ key: String, default make: @autoclosure () -> Value, _ body: (inout Value) -> Void) {
        if dict[key] == nil { keys.append(key); dict[key] = make() }
        body(&dict[key]!)
    }

    public var isEmpty: Bool { keys.isEmpty }
    public var items: [(key: String, value: Value)] { keys.map { ($0, dict[$0]!) } }
}

extension OrderedMap: Sendable where Value: Sendable {}

/// Port of benchmark.py `Pool`: per-sample payloads (frame indices shifted by a
/// per-sample offset) pooled into one metrics dict per stage. Stage order in the
/// results = first appearance in kind, skipped, errors, lists, scalars (like the Python
/// `dict.fromkeys` union), so tables and JSON line up row for row.
public final class BenchPool {
    struct FrameChunk {
        var pred: [SIMD2<Double>]
        var truth: [SIMD2<Double>]
        var vis: [Bool]
        var interp: [Bool]
        var mask: [Bool]
    }

    enum Events {
        case hits([EvalHit], [EvalHit])
        case segments([EvalSegment], [EvalSegment])
        case rallies([EvalRally], [EvalRally])
    }

    /// TrackNetV3-protocol chunk (`lists[st]["_tn4"]`).
    public struct TN4Chunk {
        public var pred: [SIMD2<Double>]
        public var truth: [SIMD2<Double>]
        public var vis: [Bool]
        public var mask: [Bool]
        public var size: ImageSize
    }

    /// `lists[stage]` (only the keys benchmark.py uses). A key is "present" once add_list
    /// touched it, even with an empty list — Python's defaultdict creates it.
    public struct StageLists {
        public var typePred: [String] = []
        public var typeTrue: [String] = []
        public var typeCov: [Double] = []
        public var callPred: [String] = []
        public var callTrue: [String] = []
        public var callMargin: [Double?] = []
        public var landPred: [SIMD2<Double>?] = []
        public var landTrue: [SIMD2<Double>?] = []
        public var tn4: [TN4Chunk]? = nil
    }

    public private(set) var kind = OrderedMap<String>()
    var frames: [String: [FrameChunk]] = [:]
    var events: [String: [Events]] = [:]
    public var lists = OrderedMap<StageLists>()
    public var scalars = OrderedMap<OrderedMap<[Double]>>()
    public var errors = OrderedMap<[String]>()
    public var skipped = OrderedMap<String>()
    public var seconds: [String: Double] = [:]

    public init() {}

    public func addFrames(_ stage: String, pred: [SIMD2<Double>], truth: [SIMD2<Double>], vis: [Bool],
                          interp: [Bool]? = nil, mask: [Bool]? = nil) {
        let n = truth.count
        kind[stage] = "frames"
        frames[stage, default: []].append(FrameChunk(pred: pred, truth: truth, vis: vis,
                                                     interp: interp ?? [Bool](repeating: false, count: n),
                                                     mask: mask ?? [Bool](repeating: true, count: n)))
    }

    public func addHits(_ stage: String, pred: [EvalHit], truth: [EvalHit], offset: Int) {
        kind[stage] = "hits"
        let sh = { (h: EvalHit) -> EvalHit in var h = h; h.frame += offset; return h }
        events[stage, default: []].append(.hits(pred.map(sh), truth.map(sh)))
    }

    public func addSegments(_ stage: String, pred: [EvalSegment], truth: [EvalSegment], offset: Int) {
        kind[stage] = "segments"
        let sh = { (s: EvalSegment) -> EvalSegment in var s = s; s.startFrame += offset; return s }
        events[stage, default: []].append(.segments(pred.map(sh), truth.map(sh)))
    }

    public func addRallies(_ stage: String, pred: [EvalRally], truth: [EvalRally], offset: Int) {
        kind[stage] = "rallies"
        let sh = { (r: EvalRally) -> EvalRally in var r = r; r.startFrame += offset; r.endFrame += offset; return r }
        events[stage, default: []].append(.rallies(pred.map(sh), truth.map(sh)))
    }

    public func addLists(_ stage: String, _ body: (inout StageLists) -> Void) {
        lists.modify(stage, default: StageLists(), body)
    }

    public func addScalar(_ stage: String, _ key: String, _ value: Double) {
        scalars.modify(stage, default: OrderedMap()) { m in m.modify(key, default: []) { $0.append(value) } }
    }

    /// Record an already formatted Pool.error message (`f"{sample}: {tb}"[:400]`).
    public func appendError(_ stage: String, _ message: String) {
        errors.modify(stage, default: []) { $0.append(String(message.unicodeScalars.prefix(400))) }
    }

    /// `Pool.error(stage, sample, exc)` for a Swift error.
    public func error(_ stage: String, sample: String, _ err: Error) {
        let text: String
        if let s = err as? StageError { text = s.message } else { text = "\(type(of: err)): \(err)" }
        appendError(stage, "\(sample): \(text)")
    }

    public func skip(_ stage: String, _ why: String) {           // Runner.skip: setdefault
        if skipped[stage] == nil { skipped[stage] = why }
    }

    /// `Pool.results(tol_px, tol_frames)`.
    public func results(tolPx: Double, tolFrames: Int) throws -> PyDict {
        var out = PyDict()
        var stages: [String] = []
        var seen = Set<String>()
        for k in kind.keys + skipped.keys + errors.keys + lists.keys + scalars.keys where !seen.contains(k) {
            seen.insert(k); stages.append(k)
        }
        for st in stages {
            var res = PyDict()
            switch kind[st] {
            case "frames":
                if let ch = frames[st], !ch.isEmpty { res.update(try Self.pooledFrames(ch, tolPx: tolPx, track: st.contains("linker"))) }
            case "hits":
                var P: [EvalHit] = [], T: [EvalHit] = []
                for case .hits(let p, let t) in events[st] ?? [] { P += p; T += t }
                res.update(Metrics.hitMetrics(P, T, tolFrames: tolFrames))
            case "segments":
                var P: [EvalSegment] = [], T: [EvalSegment] = []
                for case .segments(let p, let t) in events[st] ?? [] { P += p; T += t }
                res.update(Metrics.segmentMetrics(P, T, tolFrames: tolFrames + 1))
            case "rallies":
                var P: [EvalRally] = [], T: [EvalRally] = []
                for case .rallies(let p, let t) in events[st] ?? [] { P += p; T += t }
                res.update(Metrics.rallyMetrics(P, T))
            default: break
            }
            if let L = lists[st] {
                if !L.typePred.isEmpty {
                    let sm = Metrics.shotTypeMetrics(L.typePred, L.typeTrue)
                    for (k, v) in sm { res[k == "n" ? "n_types" : k] = v }
                    if !L.typeCov.isEmpty { res["type_coverage"] = .float(NP.mean(L.typeCov)) }
                }
                if !L.callPred.isEmpty {
                    let io = try Metrics.inOutMetrics(L.callPred, L.callTrue, margins: L.callMargin)
                    for (k, v) in io { res["inout_\(k)"] = v }
                }
                if !L.landPred.isEmpty { res.update(Metrics.landingMetrics(L.landPred, L.landTrue)) }
            }
            if let sc = scalars[st] {
                for (k, v) in sc.items {
                    let any = v.contains(where: \.isFinite)
                    res["\(k)_mean"] = .float(any ? NP.nanmean(v) : .nan)
                    res["\(k)_median"] = .float(any ? NP.nanmedian(v) : .nan)
                }
            }
            if let why = skipped[st], res.isEmpty { res["skipped"] = .str(why) }
            if let errs = errors[st], !errs.isEmpty {
                res["n_errors"] = .int(errs.count)
                res["errors"] = .strings(Array(errs.prefix(5)))
            }
            if let s = seconds[st], s != 0 { res["seconds"] = .float(PyFormat.round(s, 3)) }
            out[st] = .dict(res)
        }
        return out
    }

    /// `_pooled_frames`: chunks concatenated with a NaN / not-visible / masked-out separator
    /// row after each (so runs never join across samples), then track_metrics (linker
    /// stages; visible = finite ∧ ¬interpolated) or detection_metrics.
    static func pooledFrames(_ chunks: [FrameChunk], tolPx: Double, track: Bool) throws -> PyDict {
        var pred: [SIMD2<Double>] = [], truth: [SIMD2<Double>] = [], vis: [Bool] = [], interp: [Bool] = [], mask: [Bool] = []
        let nan = SIMD2<Double>(.nan, .nan)
        for c in chunks {
            pred += c.pred; truth += c.truth; vis += c.vis; interp += c.interp; mask += c.mask
            pred.append(nan); truth.append(nan); vis.append(false); interp.append(false); mask.append(false)
        }
        if track {
            return try Metrics.trackMetrics(pred, interpolated: interp, truth, vis: vis, tolPx: tolPx, mask: mask)
        }
        return try Metrics.detectionMetrics(pred, truth, vis: vis, tolPx: tolPx, mask: mask)
    }

    /// `add_tn4`: TrackNetV3-protocol P/R/F1 (4 px @ 512×288, int truncation) pooled over clips.
    public func addTN4(to results: inout PyDict) throws {
        for (st, L) in lists.items {
            guard let chunks = L.tn4, !chunks.isEmpty else { continue }
            var tp = 0, fp1 = 0, fp2 = 0, fn = 0, tn = 0
            for c in chunks {
                let r = try Metrics.detectionMetrics(c.pred, c.truth, vis: c.vis, tolPx: 4, mask: c.mask,
                                                     tracknetInt: true, imageSize: c.size)
                tp += r["tp"]!.int!; fp1 += r["fp1"]!.int!; fp2 += r["fp2"]!.int!; fn += r["fn"]!.int!; tn += r["tn"]!.int!
            }
            let p = Metrics.ratio(tp, tp + fp1 + fp2), rc = Metrics.ratio(tp, tp + fn)
            var r = results[st]?.dict ?? PyDict()
            r.update(["tn4_precision": .float(p), "tn4_recall": .float(rc), "tn4_f1": .float(Metrics.f1(p, rc)),
                      "tn4_accuracy": .float(Metrics.ratio(tp + tn, tp + fp1 + fp2 + fn + tn))])
            results[st] = .dict(r)
        }
        for k in lists.keys { lists.modify(k, default: StageLists()) { $0.tn4 = nil } }   // L.pop("_tn4")
    }
}

/// A stage failure carrying the exact text Python's Pool recorded
/// (`"{sample}: {ExcType}: {msg}"` → stored as `message` without the sample prefix).
public struct StageError: Error, CustomStringConvertible, Sendable {
    public var message: String
    public init(_ message: String) { self.message = message }
    public var description: String { message }
}
