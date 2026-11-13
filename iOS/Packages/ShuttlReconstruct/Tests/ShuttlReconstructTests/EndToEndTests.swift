import XCTest
import ShuttlCore
import simd
@testable import ShuttlReconstruct

/// `metrics._match_events`: one-to-one matching minimising |Δframe| (greedy on sorted
/// distances, stable), pairs (pred, true).
func matchEvents(_ pf: [Int], _ tf: [Int], tol: Int) -> [(Int, Int)] {
    var pairs: [(Int, Int, Int)] = []
    for i in pf.indices { for j in tf.indices where abs(pf[i] - tf[j]) <= tol { pairs.append((i, j, abs(pf[i] - tf[j]))) } }
    let order = pairs.indices.sorted { pairs[$0].2 < pairs[$1].2 || (pairs[$0].2 == pairs[$1].2 && $0 < $1) }
    var usedP = Set<Int>(), usedT = Set<Int>(), out: [(Int, Int)] = []
    for k in order {
        let (i, j, _) = pairs[k]
        if usedP.contains(i) || usedT.contains(j) { continue }
        usedP.insert(i); usedT.insert(j); out.append((i, j))
    }
    return out
}

/// benchmark.py --stage e2e --n 12 (sim → corrupt_detections → linker → hits → 3-D), the
/// run behind the 6.4 cm landing figure: the Swift port gets the Python stage inputs and
/// must reproduce the segments and the metric.
final class EndToEndTests: XCTestCase {
    func testSimE2ELandingError() throws {
        let fx = try fixture("e2e")
        let offset = 100_000
        var pred: [(frame: Int, seg: ShotSegment3D)] = [], truth: [(frame: Int, j: JSONValue)] = []
        var worst = 0.0, worstLand = 0.0, nSeg = 0, ms = 0.0, pyms = 0.0, frames = 0
        for (k, c) in fx["cases"]!.array!.enumerated() {
            let r = checkRally(c, "e2e seed \(c["seed"]!.int)")
            worst = max(worst, r.worst); worstLand = max(worstLand, r.worstLand); nSeg += r.segs.count
            ms += r.ms; pyms += c["py_ms"]!.d; frames += c["track"]!["t"]!.doubles.count
            pred += r.segs.map { ($0.segment.startFrame + k * offset, $0.segment) }
            truth += c["truth"]!.array!.map { ($0["start_frame"]!.int + k * offset, $0) }
        }
        let pairs = matchEvents(pred.map(\.frame), truth.map(\.frame), tol: fx["tol_frames"]!.int)
        var land: [Double] = [], spRel: [Double] = []
        for (i, j) in pairs {
            let s = pred[i].seg, t = truth[j].j
            let tv = simd_length(SIMD3(row: t["v0"]!.doubles)) * 3.6
            spRel.append(abs(s.speed0Kmh - tv) / max(tv, 1e-6))
            if let a = s.landingXY, let b = t["landing_xy"]?.optV2, a.isFinite, b.isFinite { land.append(simd_length(a - b)) }
        }
        let py = fx["py_metrics"]!
        XCTAssertEqual(pred.count, py["n_pred"]!.int)
        XCTAssertEqual(pairs.count, py["matched"]!.int)
        XCTAssertEqual(land.count, py["landing_err_m_n"]!.int)
        assertClose(Num.median(land), py["landing_err_m_median"]!.d, tol: 1e-6, "landing median")
        assertClose(Num.mean(land), py["landing_err_m_mean"]!.d, tol: 1e-6, "landing mean")
        assertClose(Num.median(spRel), py["speed_err_rel_median"]!.d, tol: 1e-6, "speed rel median")
        XCTAssertLessThan(Num.median(land), 0.065)
        print(String(format: "sim e2e (12 seeds, %d frames): %d segs, matched %d, landing err median %.2f cm (n %d), " +
                     "speed rel median %.2f %% | parity max p0/v0 %.1e, landing diff %.4f mm | " +
                     "Swift %.0f ms (%.1f ms/seg, %.0f ms/1000 frames) vs Python CPU %.0f ms",
                     frames, nSeg, pairs.count, 100 * Num.median(land), land.count, 100 * Num.median(spRel), worst,
                     worstLand * 1000, ms, ms / Double(nSeg), 1000 * ms / Double(frames), pyms))
    }
}
