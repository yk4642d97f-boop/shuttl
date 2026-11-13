import Foundation
import XCTest
import ShuttlCore
@testable import ShuttlRally

// MARK: - fixture loading

enum Fixture {
    /// Fixtures are raw-DEFLATE compressed JSON (`<name>.json.zz`) to keep the bundle small.
    static func data(_ name: String) throws -> Data {
        guard let u = Bundle.module.url(forResource: name, withExtension: "json.zz", subdirectory: "Fixtures") else {
            throw XCTSkip("fixture \(name).json.zz missing — run backend/scripts/swift_fixtures/rally.py")
        }
        let z = try Data(contentsOf: u)
        return try (z as NSData).decompressed(using: .zlib) as Data
    }

    static func load<T: Decodable>(_ type: T.Type, _ name: String) throws -> T {
        try ShuttlJSON.decoder().decode(T.self, from: data(name))
    }
}

/// VideoMeta without the timestamp table (== track.t, dropped from fixtures).
struct MetaLite: Decodable {
    var fps: Double
    var width: Int
    var height: Int
    var nFrames: Int
    enum CodingKeys: String, CodingKey { case fps, width, height, nFrames = "n_frames" }
}

/// One exported pipeline case (see backend/scripts/swift_fixtures/rally.py).
struct RallyCase: Decodable {
    var name: String
    var metaLite: MetaLite
    var track: Track2D
    var hits: [HitEvent]
    var segments: [ShotSegment3D]
    var landingSigmaM: [Double?]
    var camera: CameraModel?
    var playersRally: [PlayerObservation]?
    var playersShots: [PlayerObservation]?
    var rallies: [RallyV2]
    var shots: [RallyV2]
    var score: [JSONValue]
    var summary: JSONValue
    var rallyDicts: [JSONValue]?
    var truth: [[Int]]?
    var pySeconds: [String: Double]?
    var refinedSides: [CourtSide]?

    var meta: VideoMeta {
        VideoMeta(fps: metaLite.fps, width: metaLite.width, height: metaLite.height,
                  nFrames: metaLite.nFrames, timestamps: track.t)
    }

    enum CodingKeys: String, CodingKey {
        case name, metaLite = "meta", track, hits, segments, landingSigmaM = "landing_sigma_m", camera
        case playersRally = "players_rally", playersShots = "players_shots", rallies, shots, score, summary
        case rallyDicts = "rally_dicts", truth, pySeconds = "py_seconds", refinedSides = "refined_sides"
    }
}

// MARK: - comparison helpers

struct Mismatches {
    var items: [String] = []
    mutating func add(_ s: String) { if items.count < 40 { items.append(s) } else if items.count == 40 { items.append("…") } }
    var isEmpty: Bool { items.isEmpty }
}

func close(_ a: Double?, _ b: Double?, _ tol: Double) -> Bool {
    switch (a, b) {
    case (nil, nil): return true
    case let (x?, y?):
        if x.isNaN && y.isNaN { return true }
        return Swift.abs(x - y) <= tol
    default: return false
    }
}

func close(_ a: SIMD2<Double>?, _ b: SIMD2<Double>?, _ tol: Double) -> Bool {
    switch (a, b) {
    case (nil, nil): return true
    case let (x?, y?): return close(x.x, y.x, tol) && close(x.y, y.y, tol)
    default: return false
    }
}

/// Field-by-field RallyV2 comparison (exact labels / frames, `tol` on reals).
func compareRallies(_ got: [RallyV2], _ want: [RallyV2], tol: Double, _ m: inout Mismatches, _ ctx: String) {
    guard got.count == want.count else {
        m.add("\(ctx): \(got.count) rallies vs \(want.count): got \(got.map { [$0.startFrame, $0.endFrame] }) want \(want.map { [$0.startFrame, $0.endFrame] })")
        return
    }
    for (k, (g, w)) in zip(got, want).enumerated() {
        let c = "\(ctx)[\(k)]"
        if g.startFrame != w.startFrame || g.endFrame != w.endFrame { m.add("\(c) frames \(g.startFrame)-\(g.endFrame) vs \(w.startFrame)-\(w.endFrame)") }
        if g.endReason != w.endReason { m.add("\(c) end_reason \(g.endReason) vs \(w.endReason)") }
        if g.winner != w.winner { m.add("\(c) winner \(g.winner) vs \(w.winner)") }
        if g.server != w.server { m.add("\(c) server \(g.server) vs \(w.server)") }
        if g.serveMissed != w.serveMissed { m.add("\(c) serve_missed") }
        if g.outcomeSource != w.outcomeSource { m.add("\(c) outcome_source \(g.outcomeSource) vs \(w.outcomeSource)") }
        if !close(g.confidence, w.confidence, tol) { m.add("\(c) confidence \(g.confidence) vs \(w.confidence)") }
        if !close(g.segmentationConfidence, w.segmentationConfidence, tol) { m.add("\(c) seg_conf \(g.segmentationConfidence) vs \(w.segmentationConfidence)") }
        if !close(g.outcomeConfidence, w.outcomeConfidence, tol) { m.add("\(c) outcome_conf \(g.outcomeConfidence) vs \(w.outcomeConfidence)") }
        if !close(g.landingXY, w.landingXY, tol) { m.add("\(c) landing \(String(describing: g.landingXY)) vs \(String(describing: w.landingXY))") }
        if !close(g.inOutMarginM, w.inOutMarginM, tol) { m.add("\(c) margin \(String(describing: g.inOutMarginM)) vs \(String(describing: w.inOutMarginM))") }
        guard g.shots.count == w.shots.count else { m.add("\(c) \(g.shots.count) shots vs \(w.shots.count)"); continue }
        for (j, (gs, ws)) in zip(g.shots, w.shots).enumerated() {
            let cs = "\(c).shot[\(j)]"
            if gs.hit.frame != ws.hit.frame || gs.hit.hitter != ws.hit.hitter { m.add("\(cs) hit \(gs.hit.frame)/\(gs.hit.hitter) vs \(ws.hit.frame)/\(ws.hit.hitter)") }
            if !close(gs.hit.t, ws.hit.t, 1e-12) { m.add("\(cs) hit.t") }
            if gs.shotType != ws.shotType { m.add("\(cs) type \(gs.shotType) vs \(ws.shotType)") }
            if !close(gs.shotTypeConfidence, ws.shotTypeConfidence, tol) { m.add("\(cs) type_conf \(gs.shotTypeConfidence) vs \(ws.shotTypeConfidence)") }
            if gs.strokeSide != ws.strokeSide { m.add("\(cs) stroke \(gs.strokeSide) vs \(ws.strokeSide)") }
            if !close(gs.hitterPosWorld, ws.hitterPosWorld, tol) { m.add("\(cs) hitter_pos \(String(describing: gs.hitterPosWorld)) vs \(String(describing: ws.hitterPosWorld))") }
            if !close(gs.hitHeightM, ws.hitHeightM, tol) { m.add("\(cs) height") }
            if (gs.segment == nil) != (ws.segment == nil) || gs.segment?.startFrame != ws.segment?.startFrame { m.add("\(cs) segment") }
        }
    }
}

/// Structural JSON diff; numbers compared with `tol(path)` (0 = exact).
func jsonDiff(_ a: JSONValue, _ b: JSONValue, path: String = "$", tol: (String) -> Double = { _ in 0 },
              _ m: inout Mismatches) {
    switch (a, b) {
    case let (.number(x), .number(y)):
        let t = tol(path)
        if !(x == y || Swift.abs(x - y) <= t) { m.add("\(path): \(x) vs \(y)") }
    case let (.array(x), .array(y)):
        guard x.count == y.count else { m.add("\(path): len \(x.count) vs \(y.count)"); return }
        for i in x.indices { jsonDiff(x[i], y[i], path: "\(path)[\(i)]", tol: tol, &m) }
    case let (.object(x), .object(y)):
        if Set(x.keys) != Set(y.keys) { m.add("\(path): keys \(x.keys.sorted()) vs \(y.keys.sorted())"); return }
        for k in x.keys.sorted() { jsonDiff(x[k]!, y[k]!, path: "\(path).\(k)", tol: tol, &m) }
    default:
        if a != b { m.add("\(path): \(a) vs \(b)") }
    }
}

/// benchmark rally F1: greedy one-to-one matching by inclusive interval IoU >= 0.5
/// (metrics.rally_metrics), pooled over cases.
struct RallyF1 {
    var tp = 0, nPred = 0, nTrue = 0

    static func iou(_ a: (Int, Int), _ b: (Int, Int)) -> Double {
        let inter = Swift.min(a.1, b.1) - Swift.max(a.0, b.0) + 1
        let union = Swift.max(a.1, b.1) - Swift.min(a.0, b.0) + 1
        return union > 0 ? Double(Swift.max(inter, 0)) / Double(union) : 0
    }

    mutating func add(pred: [(Int, Int)], truth: [(Int, Int)], thr: Double = 0.5) {
        var cand: [(Double, Int, Int)] = []
        for (i, p) in pred.enumerated() { for (j, t) in truth.enumerated() { cand.append((RallyF1.iou(p, t), i, j)) } }
        cand.sort { ($0.0, $0.1, $0.2) > ($1.0, $1.1, $1.2) }
        var up = Set<Int>(), ut = Set<Int>()
        for (v, i, j) in cand {
            if v < thr { break }
            if up.contains(i) || ut.contains(j) { continue }
            up.insert(i); ut.insert(j)
        }
        tp += up.count; nPred += pred.count; nTrue += truth.count
    }

    var f1: Double {
        let p = nPred > 0 ? Double(tp) / Double(nPred) : .nan
        let r = nTrue > 0 ? Double(tp) / Double(nTrue) : .nan
        return (p + r) > 0 && (p + r).isFinite ? 2 * p * r / (p + r) : 0
    }
}
