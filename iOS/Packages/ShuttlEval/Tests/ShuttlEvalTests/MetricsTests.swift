import Foundation
import XCTest
import ShuttlCore
@testable import ShuttlEval

/// Port of backend/tests/test_metrics.py (perfect input → perfect scores, known counts).
final class MetricsTests: XCTestCase {

    func truth(_ n: Int = 50) -> ([SIMD2<Double>], [Bool]) {
        var xy = (0..<n).map { i -> SIMD2<Double> in let t = Double(i); return SIMD2(100 + 5 * t, 300 - 0.2 * (t - 25) * (t - 25)) }
        var vis = [Bool](repeating: true, count: n)
        for i in [3, 4, 30] { vis[i] = false; xy[i] = SIMD2(.nan, .nan) }
        return (xy, vis)
    }

    func f(_ d: PyDict, _ k: String) -> Double { d[k]!.double! }
    func i(_ d: PyDict, _ k: String) -> Int { d[k]!.int! }

    func testScaledTolerance() {
        XCTAssertEqual(Metrics.scaledTolerance(4, imageSize: ImageSize(width: 1280, height: 720)), 10, accuracy: 1e-12)
        XCTAssertEqual(Metrics.scaledTolerance(4, imageSize: ImageSize(width: 1920, height: 1080)), 15, accuracy: 1e-12)
    }

    func testDetectionPerfectAndCounts() throws {
        let (xy, vis) = truth()
        let r = try Metrics.detectionMetrics(xy, xy, vis: vis)
        for k in ["precision", "recall", "f1", "accuracy"] { XCTAssertEqual(f(r, k), 1.0) }
        XCTAssertEqual(i(r, "tp"), 47); XCTAssertEqual(i(r, "tn"), 3); XCTAssertEqual(f(r, "err_median"), 0)
        var p = xy
        p[10] += SIMD2(20, 0)                      // FP1 (too far)
        p[11] = SIMD2(.nan, .nan)                  // FN
        p[3] = SIMD2(5, 5)                         // FP2 (nothing there)
        p[12] += SIMD2(3, 4)                       // 5 px: TP at tol 10
        let r2 = try Metrics.detectionMetrics(p, xy, vis: vis, tolPx: 10)
        XCTAssertEqual([i(r2, "tp"), i(r2, "fp1"), i(r2, "fp2"), i(r2, "fn"), i(r2, "tn")], [45, 1, 1, 1, 2])
        XCTAssertEqual(f(r2, "precision"), 45.0 / 47, accuracy: 1e-12)
        XCTAssertEqual(f(r2, "recall"), 45.0 / 46, accuracy: 1e-12)
        XCTAssertEqual(i(try Metrics.detectionMetrics(p, xy, vis: vis, tolPx: 4), "tp"), 44)
        var m = [Bool](repeating: false, count: xy.count)
        for k in 20..<40 { m[k] = true }
        XCTAssertEqual(i(try Metrics.detectionMetrics(p, xy, vis: vis, mask: m), "n_frames"), 20)
        XCTAssertThrowsError(try Metrics.detectionMetrics(Array(p.prefix(3)), xy))
    }

    func testDetectionTracknetIntRule() throws {
        let t = [SIMD2<Double>(640, 360)], size = ImageSize(width: 1280, height: 720)
        let ok = try Metrics.detectionMetrics([SIMD2(649.9, 360)], t, tolPx: 4, tracknetInt: true, imageSize: size)
        let bad = try Metrics.detectionMetrics([SIMD2(652.6, 360)], t, tolPx: 4, tracknetInt: true, imageSize: size)
        XCTAssertEqual(i(ok, "tp"), 1); XCTAssertEqual(i(bad, "fp1"), 1)
        XCTAssertThrowsError(try Metrics.detectionMetrics([SIMD2(1, 1)], t, tracknetInt: true))
    }

    func testCandidatesToXYThreshold() throws {
        let C = ShuttleCandidate.init(x:y:score:)
        let c = [[C(1, 2, 0.9), C(5, 5, 0.95)], [], [C(3, 3, 0.3)]]
        let (xy, sc) = Metrics.candidatesToXY(c, scoreThresh: 0.5)
        XCTAssertEqual(xy[0], SIMD2(5, 5)); XCTAssertFalse(xy[1].isFinite); XCTAssertFalse(xy[2].isFinite)
        XCTAssertEqual(sc[2], 0.3, accuracy: 1e-12)
        let r = try Metrics.detectionMetrics(c, [SIMD2(5, 5), SIMD2(.nan, .nan), SIMD2(3, 3)])
        XCTAssertEqual([i(r, "tp"), i(r, "tn"), i(r, "fn")], [1, 1, 1])
    }

    func testTrackMetrics() throws {
        let (xy, vis) = truth()
        let tr = Track2D(t: (0..<50).map(Double.init), xy: xy, conf: vis.map { $0 ? 1 : 0 }, visible: vis,
                         interpolated: [Bool](repeating: false, count: 50))
        let r = try Metrics.trackMetrics(tr, xy, vis: vis)
        XCTAssertEqual(f(r, "f1"), 1); XCTAssertEqual(f(r, "jitter_px"), 0); XCTAssertEqual(i(r, "jumps"), 0)
        var p = xy
        for k in 20..<23 { p[k] += SIMD2(200, 0) }  // one excursion onto something else
        p[40] += SIMD2(0, 1)                         // small noise → jitter
        let r2 = try Metrics.trackMetrics(p, interpolated: nil, xy, vis: vis, tolPx: 10)
        XCTAssertEqual(i(r2, "jumps"), 1); XCTAssertEqual(i(r2, "wrong_frames"), 3)
        XCTAssert(f(r2, "jitter_px") > 0 && f(r2, "jitter_px") < 2)
        var filled = xy
        filled[3] = SIMD2(115, 200); filled[4] = SIMD2(120, 200)
        let tr2 = Track2D(t: (0..<50).map(Double.init), xy: filled, conf: vis.map { $0 ? 1 : 0 }, visible: vis,
                          interpolated: (0..<50).map { $0 == 3 || $0 == 4 })
        let r3 = try Metrics.trackMetrics(tr2, xy, vis: vis)
        XCTAssertEqual(i(r3, "fp2"), 2); XCTAssertEqual(i(r3, "interp_frames"), 2)
    }

    func testHitMetrics() {
        let truth = [EvalHit(frame: 10, hitter: "near"), EvalHit(frame: 50, hitter: "far"), EvalHit(frame: 90, hitter: "near")]
        let perfect = truth.map { HitEvent(frame: $0.frame, t: Double($0.frame) / 30, xyPx: SIMD2(0, 0),
                                           hitter: CourtSide(rawValue: $0.hitter)!) }
        let r = Metrics.hitMetrics(perfect, truth)
        XCTAssertEqual(f(r, "f1"), 1); XCTAssertEqual(f(r, "offset_abs_mean"), 0); XCTAssertEqual(f(r, "hitter_acc"), 1)
        let pred = [EvalHit(frame: 12, hitter: "near"), EvalHit(frame: 11, hitter: "far"),
                    EvalHit(frame: 49, hitter: "unknown"), EvalHit(frame: 70, hitter: "far")]
        let r2 = Metrics.hitMetrics(pred, truth, tolFrames: 3)
        XCTAssertEqual([i(r2, "tp"), i(r2, "fp"), i(r2, "fn")], [2, 2, 1])      // one-to-one: 11 ↔ 10 (closest)
        XCTAssertEqual(f(r2, "offset_abs_mean"), 1, accuracy: 1e-12)
        XCTAssertEqual(f(r2, "hitter_acc"), 0); XCTAssertEqual(f(r2, "hitter_acc_known"), 0)   // 'unknown' excluded
    }

    func testSegmentAndLandingMetrics() {
        let shots = [EvalSegment(startFrame: 10, p0: [-5, 0, 2.8], v0: [80, 0, -10], netCrossingHeight: 2.0, landingXY: [3, 1]),
                     EvalSegment(startFrame: 60, p0: [4, 1, 1], v0: [-20, 0, 5], netCrossingHeight: 1.8, landingXY: nil)]
        let r = Metrics.segmentMetrics(shots, shots)
        XCTAssertEqual(f(r, "recall"), 1); XCTAssertEqual(f(r, "speed_err_kmh_median"), 0); XCTAssertEqual(i(r, "landing_err_m_n"), 1)
        var p = shots[0]
        p.v0 *= 1.1; p.landingXY = [3.3, 1.4]; p.startFrame = 12
        let r2 = Metrics.segmentMetrics([p], shots)
        XCTAssertEqual(i(r2, "matched"), 1); XCTAssertEqual(f(r2, "recall"), 0.5)
        XCTAssertEqual(f(r2, "speed_err_kmh_median"), 0.1 * (80.0 * 80 + 100).squareRoot() * 3.6, accuracy: 1e-9)
        XCTAssertEqual(f(r2, "landing_err_m_median"), 0.5, accuracy: 1e-9)
        XCTAssertEqual(i(Metrics.landingMetrics([SIMD2(0, 0), nil], [SIMD2(0, 1), SIMD2(2, 2)]), "landing_err_m_n"), 1)
    }

    func testInOutMetrics() throws {
        let r = try Metrics.inOutMetrics(["in", "out", "net", "unsure"], ["in", "in", "net", "out"], margins: [0.05, 0.1, nil, -0.5])
        XCTAssertEqual(f(r, "accuracy"), 0.5, accuracy: 1e-12)
        XCTAssertEqual(f(r, "accuracy_decided"), 2.0 / 3, accuracy: 1e-12)
        XCTAssertEqual(i(r, "n_close"), 2); XCTAssertEqual(f(r, "accuracy_close"), 0.5, accuracy: 1e-12)
        let m = r["confusion"]!["matrix"]!.list!.map { $0.list!.map { $0.int! } }
        XCTAssertEqual(m[0][0], 1); XCTAssertEqual(m[0][1], 1); XCTAssertEqual(m[1][3], 1)
        XCTAssertEqual(f(try Metrics.inOutMetrics(["in", "out"], ["in", "out"]), "accuracy"), 1)
    }

    func testShotTypeMetrics() {
        let r = Metrics.shotTypeMetrics(["smash", "clear", "drop"], ["smash", "clear", "net"])
        XCTAssertEqual(f(r, "accuracy"), 2.0 / 3, accuracy: 1e-12)
        XCTAssertEqual(r["per_class_recall"]!["net"]!.double!, 0)
        XCTAssertEqual(f(Metrics.shotTypeMetrics(["a", "b"], ["a", "b"]), "accuracy"), 1)
    }

    func testRallyMetrics() {
        let truth = [EvalRally(startFrame: 0, endFrame: 100, endReason: "in", winner: "near", landingXY: [1, 2], margin: 0.1),
                     EvalRally(startFrame: 300, endFrame: 400, endReason: "net", winner: "far", landingXY: nil, margin: nil)]
        let r = Metrics.rallyMetrics(truth, truth)
        for k in ["f1", "iou_mean", "end_reason_acc", "winner_acc", "accuracy_close"] { XCTAssertEqual(f(r, k), 1, k) }
        XCTAssertEqual(f(r, "landing_err_m_median"), 0)
        var p0 = truth[0]
        p0.startFrame = 10; p0.endReason = "out"; p0.winner = "far"
        let pred = [p0, EvalRally(startFrame: 150, endFrame: 200, endReason: "in", winner: "near")]
        let r2 = Metrics.rallyMetrics(pred, truth)
        XCTAssertEqual([i(r2, "tp"), i(r2, "n_pred"), i(r2, "n_true")], [1, 2, 2])
        XCTAssertEqual(f(r2, "iou_mean"), 91.0 / 101, accuracy: 1e-12)
        XCTAssertEqual(f(r2, "end_reason_acc"), 0); XCTAssertEqual(f(r2, "winner_acc"), 0); XCTAssertEqual(f(r2, "accuracy_close"), 0)
    }

    func testFormatTable() {
        let s = Metrics.formatTable(["linker": .dict(["f1": 0.9, "jumps": 2]), "hits": .dict(["skipped": "no module"])])
        XCTAssert(s.contains("f1=0.900") && s.contains("jumps=2") && s.contains("SKIPPED"))
    }
}
