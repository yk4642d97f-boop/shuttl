import Foundation
import ShuttlCore
import XCTest
@testable import ShuttlHits

/// Collects parity mismatches and the largest deviations seen.
final class Parity {
    var failures: [String] = []
    var maxDev: [String: Double] = [:]
    var nHits = 0, nCands = 0, nCases = 0

    func dev(_ key: String, _ a: Double, _ b: Double, tol: Double, rel: Bool = false, _ ctx: @autoclosure () -> String) {
        if a.isNaN && b.isNaN { return }
        if a.isInfinite || b.isInfinite {
            if a != b { failures.append("\(ctx()) \(key): \(a) vs \(b)") }
            return
        }
        let d = abs(a - b) / (rel ? Swift.max(1, abs(b)) : 1)
        maxDev[key] = Swift.max(maxDev[key] ?? 0, d)
        if !(d <= tol) { failures.append("\(ctx()) \(key): swift \(a) python \(b) (dev \(d))") }
    }

    func same<T: Equatable>(_ key: String, _ a: T, _ b: T, _ ctx: @autoclosure () -> String) {
        if a != b { failures.append("\(ctx()) \(key): swift \(a) python \(b)") }
    }

    func hits(_ sw: [HitEvent], _ py: [FHit], _ name: String) {
        nCases += 1
        guard sw.count == py.count else {
            failures.append("\(name): \(sw.count) hits vs python \(py.count); swift frames \(sw.map(\.frame)) python \(py.map(\.frame))")
            return
        }
        for (i, (s, p)) in zip(sw, py).enumerated() {
            nHits += 1
            let ctx = "\(name) hit \(i)"
            same("frame", s.frame, p.frame, ctx)
            same("hitter", s.hitter.rawValue, p.hitter, ctx)
            same("player_track_id", s.playerTrackID, p.player_track_id, ctx)
            same("source", s.source, p.source ?? HitEvent.defaultSource, ctx)
            dev("confidence", s.confidence, p.confidence, tol: 1e-6, ctx)
            dev("t", s.t, p.t, tol: 1e-6, ctx)
            dev("xy_px", s.xyPx.x, p.xy_px[0], tol: 1e-6, rel: true, ctx)
            dev("xy_px", s.xyPx.y, p.xy_px[1], tol: 1e-6, rel: true, ctx)
        }
    }

    func cands(_ sw: [HitCandidate], _ py: [FCand], _ name: String) {
        guard sw.count == py.count else {
            failures.append("\(name): \(sw.count) candidates vs python \(py.count); swift tau \(sw.map(\.tau)) python \(py.map(\.tau))")
            return
        }
        for (i, (s, p)) in zip(sw, py).enumerated() {
            nCands += 1
            let ctx = "\(name) cand \(i)"
            same("frame", s.frame, p.frame, ctx)
            same("seg", s.seg, p.seg, ctx)
            same("flags", [s.isStart, s.staticBefore, s.staticAfter, s.outside, s.settles, s.weak, s.hidden, s.gapContact],
                 [p.is_start, p.static_before, p.static_after, p.outside, p.settles, p.weak, p.hidden ?? false,
                  p.gap_contact ?? false], ctx)
            same("near_tid", s.nearTID, p.near_tid, ctx)
            same("far_tid", s.farTID, p.far_tid, ctx)
            dev("tau", s.tau, p.tau, tol: 1e-9, ctx)
            dev("lam", s.lam, p.lam, tol: 1e-6, rel: true, ctx)
            dev("lam_v", s.lamV, p.lam_v, tol: 1e-6, rel: true, ctx)
            dev("gap_s", s.gapS, p.gap_s ?? .infinity, tol: 1e-12, ctx)
            if let pk = p.kin { dev("kin", s.kin, pk, tol: 1e-6, ctx) }
            else if s.kin.isFinite { failures.append("\(ctx) kin: swift \(s.kin) python non-finite") }
            dev("reach", s.reach, p.reach, tol: 1e-6, ctx)
            dev("side_llr", s.sideLLR, p.side_llr, tol: 1e-6, ctx)
            dev("xy", s.xy.x, p.xy[0], tol: 1e-6, rel: true, ctx)
            dev("xy", s.xy.y, p.xy[1], tol: 1e-6, rel: true, ctx)
            for k in 0..<2 {
                dev("v", s.vIn[k], p.v_in[k], tol: 1e-6, rel: true, ctx)
                dev("v", s.vOut[k], p.v_out[k], tol: 1e-6, rel: true, ctx)
            }
            same("n_feats", s.feats.count, p.feats.count, ctx)
            for (k, (a, b)) in zip(s.feats, p.feats).enumerated() {
                dev("feat_\(HIT_FEATURES[k])", a, b ?? (k == 10 ? .infinity : .nan), tol: 1e-6, rel: true, ctx)
            }
        }
    }

    func assertClean(_ label: String, file: StaticString = #filePath, line: UInt = #line) {
        let devs = maxDev.sorted { $0.key < $1.key }.map { "\($0.key)=\(String(format: "%.2e", $0.value))" }.joined(separator: " ")
        print("[\(label)] cases \(nCases) hits \(nHits) cands \(nCands) failures \(failures.count) | max dev: \(devs)")
        XCTAssertTrue(failures.isEmpty, "\(label): \(failures.count) mismatches, first: \(failures.prefix(8).joined(separator: "\n"))",
                      file: file, line: line)
    }
}

/// metrics.hit_metrics: greedy one-to-one matching by |Δframe| <= tol (stable on ties).
struct HitMetrics {
    var tp = 0, nPred = 0, nTrue = 0, sideOK = 0
    var precision: Double { nPred > 0 ? Double(tp) / Double(nPred) : 0 }
    var recall: Double { nTrue > 0 ? Double(tp) / Double(nTrue) : 0 }
    var f1: Double { precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0 }
    var hitterAcc: Double { tp > 0 ? Double(sideOK) / Double(tp) : .nan }

    mutating func add(pred: [HitEvent], trueFrames: [Int], trueSides: [String], tol: Int = 3) {
        var pairs: [(d: Int, k: Int, i: Int, j: Int)] = []
        for (i, p) in pred.enumerated() {
            for (j, f) in trueFrames.enumerated() where abs(p.frame - f) <= tol {
                pairs.append((abs(p.frame - f), pairs.count, i, j))
            }
        }
        pairs.sort { $0.d < $1.d || ($0.d == $1.d && $0.k < $1.k) }
        var usedP = Set<Int>(), usedT = Set<Int>()
        for p in pairs where !usedP.contains(p.i) && !usedT.contains(p.j) {
            usedP.insert(p.i); usedT.insert(p.j)
            tp += 1
            if pred[p.i].hitter.rawValue == trueSides[p.j] { sideOK += 1 }
        }
        nPred += pred.count; nTrue += trueFrames.count
    }
}
