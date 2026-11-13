import Foundation
import ShuttlCore
import XCTest
@testable import ShuttlHits

/// Native ports of backend/tests/test_hits.py (same generated inputs, via the unit fixture).
final class BehaviourTests: XCTestCase {
    private static let unit: [String: FCase] = {
        let f = try! Fixtures.load("hits_unit.json")
        return Dictionary(uniqueKeysWithValues: f.cases.map { ($0.name, $0) })
    }()

    override func setUp() { HitModelRegistry.install(nil) }

    private func hits(_ name: String, config: HitConfig? = nil, rescorer: (any HitRescorer)? = nil) -> [HitEvent] {
        let c = BehaviourTests.unit[name]!
        let i = c.inputs
        return detectHits(track: i.track, meta: i.meta, camera: i.camera, players: i.players,
                          config: config ?? c.config, rescorer: rescorer)
    }

    func testHitEventContractAndEmptyInputs() {
        let hs = hits("contract_arc")
        XCTAssertFalse(hs.isEmpty)
        for h in hs {
            XCTAssertTrue([CourtSide.near, .far, .unknown].contains(h.hitter))
            XCTAssertTrue((0...1).contains(h.confidence))
            XCTAssertTrue(h.t.isFinite && h.xyPx.x.isFinite && h.xyPx.y.isFinite)
        }
        XCTAssertEqual(hs.map(\.t), hs.map(\.t).sorted())
        XCTAssertEqual(hits("empty"), [])
        // too few observations / no observations at all
        let t = (0..<5).map { Double($0) / 30 }
        let tr = Track2D(t: t, xy: .init(repeating: SIMD2(.nan, .nan), count: 5), conf: .init(repeating: 0, count: 5),
                         visible: .init(repeating: false, count: 5), interpolated: .init(repeating: false, count: 5))
        XCTAssertEqual(detectHits(track: tr, meta: VideoMeta(fps: 30, width: 1280, height: 720, nFrames: 5, timestamps: t)), [])
    }

    func testSingleKinkTiming() {
        for fps in [30.0, 60.0, 120.0] {
            let hs = hits("single_kink_\(Int(fps))").filter { 0.2 < $0.t && $0.t < 1.9 }
            XCTAssertEqual(hs.count, 1, "fps \(fps)")
            if let h = hs.first { XCTAssertLessThan(abs(h.t - 1.0), 1.0 / fps + 0.004, "fps \(fps)") }
        }
    }

    func testHitInsideOcclusionGap() {
        let hs = hits("occlusion_gap").filter { 0.2 < $0.t && $0.t < 1.9 }
        XCTAssertEqual(hs.count, 1)
        XCTAssertLessThan(abs(hs[0].t - 1.0), 0.05)
    }

    func testLandingAndRestIsNotAHit() {
        XCTAssertTrue(hits("landing_rest").allSatisfy { abs($0.t - 1.0) > 0.2 })
    }

    func testServeAfterDeadTimeAndOutliers() {
        let hs = hits("serve_dead_time_outliers")
        XCTAssertTrue(hs.contains { abs($0.t - 1.0) < 1.0 / 60 + 0.004 })
        XCTAssertEqual(hs.filter { $0.t > 0.55 }.count, 1)          // outliers create no hits
    }

    func testRepeatedFramesRetimed() {
        let hs = hits("repeated_frames_25p_in_30p").filter { 0.2 < $0.t && $0.t < 1.9 }
        XCTAssertEqual(hs.count, 1)
        XCTAssertLessThan(abs(hs[0].t - 1.0), 0.05)
    }

    func testVariableFrameRateTimestamps() {
        let c = BehaviourTests.unit["variable_frame_rate"]!
        var m = HitMetrics()
        m.add(pred: hits("variable_frame_rate"), trueFrames: c.true_frame!, trueSides: c.true_side!)
        XCTAssertGreaterThanOrEqual(Double(m.tp), 0.85 * Double(m.nTrue))
        XCTAssertLessThanOrEqual(m.nPred, m.tp + 2)
    }

    func testLearnedModelHook() {
        let base = hits("contract_arc")
        let zero = LinearHitModel(w: [Double](repeating: 0, count: HIT_FEATURES.count), b: 0)
        XCTAssertEqual(hits("contract_arc", rescorer: zero).count, base.count)
        // a failing model falls back to the model-based scores (Python logs and continues)
        struct Broken: HitRescorer { func logits(_ f: [[Double]]) throws -> [Double] { throw HitModelError.badOutput } }
        XCTAssertEqual(hits("contract_arc", rescorer: Broken()), base)
        XCTAssertThrowsError(try zero.logits([[1, 2, 3]]))
        let json = #"{"w": [1,2,3,4,5,6,7,8,9,10,11,12,13], "b": [0.5], "mu": 0.25}"#.data(using: .utf8)!
        let m = try! JSONDecoder().decode(LinearHitModel.self, from: json)
        XCTAssertEqual(m.b, 0.5); XCTAssertEqual(m.mu, .init(repeating: 0.25, count: 13)); XCTAssertEqual(m.sd, .init(repeating: 1, count: 13))
    }

    /// E's T10 repro (tools/robustness/repro/track-len-mismatch): a Track2D whose
    /// xy is one shorter than t used to trap in preferRaw (Observations.swift:30,
    /// "Index out of range"). Every length mismatch now returns without trapping.
    func testInconsistentTrackLengthsDoNotTrap() {
        let c = BehaviourTests.unit["prefers_raw_xy"]!.inputs
        let tr = c.track
        XCTAssertNil(tr.inconsistency)
        XCTAssertFalse(hits("prefers_raw_xy").isEmpty)
        var shortXY = tr; shortXY.xy.removeLast()
        var shortRaw = tr; shortRaw.rawXY!.removeLast()
        var shortConf = tr; shortConf.conf.removeLast()
        var shortVis = tr; shortVis.visible.removeLast()
        var shortInterp = tr; shortInterp.interpolated.removeLast()
        var longXY = tr; longXY.xy.append(SIMD2(1, 1))
        for (name, bad) in [("xy-1", shortXY), ("raw-1", shortRaw), ("conf-1", shortConf), ("visible-1", shortVis),
                            ("interpolated-1", shortInterp), ("xy+1", longXY)] {
            XCTAssertNotNil(bad.inconsistency, name)
            let xy = Hits.preferRaw(bad)                       // the old trap site
            XCTAssertEqual(xy.count, bad.xy.count, name)
            XCTAssertEqual(detectHits(track: bad, meta: c.meta, camera: c.camera, players: c.players), [], name)
            XCTAssertTrue(detectCandidates(track: bad, meta: c.meta, camera: c.camera, players: c.players).isEmpty, name)
        }
        // raw_xy one short, xy intact: preferRaw keeps xy (no substitution from a misaligned raw)
        XCTAssertEqual(Hits.preferRaw(shortRaw), tr.xy)
    }

    func testPrefersUnsmoothedRawXY() {
        let c = BehaviourTests.unit["prefers_raw_xy"]!.inputs
        let raw = c.track.rawXY!
        let xy = Hits.preferRaw(c.track)
        for i in raw.indices {
            if raw[i].x.isFinite && raw[i].y.isFinite { XCTAssertEqual(xy[i], raw[i]) } else { XCTAssertEqual(xy[i], c.track.xy[i]) }
        }
        // detectHits scores raw_xy by default; use_raw_xy=false the smoothed xy (fixture parity covers both)
        var off = HitConfig(); off.useRawXY = false
        XCTAssertEqual(hits("raw_xy_disabled"), hits("prefers_raw_xy", config: off))
    }

    func testLinkerKinkTimesAddCandidates() {
        let c = BehaviourTests.unit["contract_arc"]!.inputs
        let base = detectCandidates(track: c.track, meta: c.meta)
        var cfg = HitConfig()
        cfg.linkerKinkTimes = [base[0].tau, 0.5]              // one duplicate of a peak, one new time
        let more = detectCandidates(track: c.track, meta: c.meta, config: cfg)
        XCTAssertGreaterThanOrEqual(more.count, base.count)
        XCTAssertEqual(more.filter { abs($0.tau - base[0].tau) < 0.05 }.count, 1)
    }

    func testFeatureOrderMatchesPython() {
        XCTAssertEqual(HIT_FEATURES, ["log_lam", "log_lam_v", "speed_in", "speed_out", "log_speed_ratio",
                                      "cos_turn", "vy_flip", "static_before", "static_after", "outside",
                                      "gap_s", "player_reach", "is_flight_start"])
        let c = BehaviourTests.unit["contract_arc"]!.inputs
        XCTAssertTrue(detectCandidates(track: c.track, meta: c.meta).allSatisfy { $0.feats.count == HIT_FEATURES.count })
    }

    func testRealTracksSmoke() throws {
        // test_real_labels_smoke bound (contacts >= ~0.3 s apart), on the linker tracks
        for c in try Fixtures.load("hits_real.json").cases {
            let i = c.inputs
            let hs = detectHits(track: i.track, meta: i.meta)
            let vis = i.track.visible.indices.filter { i.track.visible[$0] }
            let span = Double(vis.last! - vis.first!) / 30.0
            XCTAssertTrue(1 <= hs.count && Double(hs.count) <= span / 0.3 + 2, "\(c.name): \(hs.count) hits, span \(span)")
        }
    }

    func testConfigDefaultsMatchPython() {
        let c = HitConfig()
        XCTAssertEqual([c.windowS, c.maxWindowS, c.nmsS, c.lamMin, c.lamMinWeak, c.weakReachMin, c.lamRef],
                       [0.15, 0.40, 0.20, 12.0, 4.0, 0.9, 15.0])
        XCTAssertEqual([c.minSide, c.wantSide, c.maxSide, c.refineSteps], [3, 5, 24, 9])
        XCTAssertEqual([c.minSpeedOut, c.staticPx, c.staticS, c.settleS, c.settleFrac, c.settleSpeed, c.deadTimeS,
                        c.maxOutOfViewS, c.borderBand, c.borderPx], [0.10, 6.0, 0.35, 0.8, 0.05, 0.25, 1.0, 4.0, 0.06, 8.0])
        XCTAssertEqual([c.minSepS, c.shortFlightS, c.longFlightS, c.longFlightSd, c.missedHitS, c.logEpsSame,
                        c.logEpsFastSame, c.pairYWeight, c.playerWeight, c.cameraWeight, c.minHitLogodds, c.learnedBlend],
                       [0.22, 0.28, 1.7, 0.45, 0.6, -3.0, -9.0, 1.0, 1.0, 1.0, -4.0, 0.5])
        XCTAssertEqual(c.refineModel, "1"); XCTAssertNil(c.sigmaPx); XCTAssertTrue(c.useRawXY); XCTAssertNil(c.linkerKinkTimes)
    }
}
