import Foundation
import XCTest
import ShuttlCore
import simd
@testable import ShuttlRally

// MARK: - fixture shapes

private struct ScoreCase: Decodable {
    struct Kw: Decodable {
        var firstServer: CourtSide?
        var gamesToWin: Int?
        enum CodingKeys: String, CodingKey { case firstServer = "first_server", gamesToWin = "games_to_win" }
    }
    var rallies: [RallyV2]
    var kwargs: Kw
    var expected: [JSONValue]
}

private struct FlightCase: Decodable {
    var kind: String
    var segment: ShotSegment3D
    var isFirst: Bool
    var label: ShotType
    var conf: Double
    var features: [String: Double?]?
    enum CodingKeys: String, CodingKey { case kind, segment, isFirst = "is_first", label, conf, features }
}

private struct ShotRallyCase: Decodable {
    var name: String
    var rally: RallyV2
    var players: [PlayerObservation]?
    var handedness: [[JSONValue]]?
    var expected: RallyV2
}

private struct StrokeCase: Decodable {
    var kps: [Keypoint]
    var right: Bool
    var side: StrokeSide
    var above: [Double?]
}

private struct ContactCase: Decodable {
    var segment: ShotSegment3D
    var tHit: Double
    var p: [Double]
    var v: [Double]
    enum CodingKeys: String, CodingKey { case segment, tHit = "t_hit", p, v }
}

private struct OverheadCase: Decodable { var h: Double?; var po: Double?; var p: Double }

private struct ShotsFixture: Decodable {
    var flights: [FlightCase]
    var noisy: [FlightCase]
    var rallies: [ShotRallyCase]
    var strokes: [StrokeCase]
    var contact: [ContactCase]
    var overhead: [OverheadCase]
}

// MARK: - score_match

final class ScoreMatchTests: XCTestCase {
    func testScoreMatchParity() throws {
        let cases = try Fixture.load([ScoreCase].self, "score")
        XCTAssertGreaterThanOrEqual(cases.count, 18)
        var m = Mismatches()
        for (k, c) in cases.enumerated() {
            let got = scoreMatch(c.rallies, firstServer: c.kwargs.firstServer ?? .near,
                                 gamesToWin: c.kwargs.gamesToWin ?? 2)
            jsonDiff(.array(got), .array(c.expected), path: "score[\(k)]", &m)
        }
        XCTAssert(m.isEmpty, m.items.joined(separator: "\n"))
    }

    private func R(_ w: CourtSide, _ i: Int, _ first: CourtSide? = nil) -> RallyV2 {
        let shots = first.map { [ShotEventV2(hit: HitEvent(frame: i * 100, t: Double(i * 100) / 30, xyPx: [0, 0],
                                                           hitter: $0, confidence: 0.9), segment: nil)] } ?? []
        return RallyV2(startFrame: i * 100, endFrame: i * 100 + 50, shots: shots, winner: w, confidence: 0.9)
    }

    private func play(_ seq: [CourtSide]) -> [JSONValue] { scoreMatch(seq.enumerated().map { R($0.element, $0.offset) }) }

    // test_rally.test_score_straight_game_then_ends_change
    func testStraightGameThenEndsChange() {
        let sc = play(Array(repeating: .near, count: 21) + [.far])
        XCTAssertEqual(sc[..<21].map { $0["game_over"]?.bool }, Array(repeating: false, count: 20) + [true])
        XCTAssertEqual(sc[20]["score"], .object(["near": .number(21), "far": .number(0)]))
        XCTAssertEqual(sc[20]["ends_changed"]?.bool, true)
        XCTAssertEqual(sc[20]["next_server"]?.string, "far")
        XCTAssertEqual(sc[21]["game"]?.double, 2)
        XCTAssertEqual(sc[21]["server"]?.string, "far")
        XCTAssertEqual(sc[21]["service_court"]?.string, "right")
        XCTAssertEqual(sc[21]["score_by_player"], .object(["A": .number(1), "B": .number(0)]))
    }

    // test_rally.test_unknown_winner_inferred_from_next_serve_or_left_uncertain (20-0 ambiguity)
    func testImpliedWinnerAmbiguousAtGamePoint() {
        var st = BWFScore()
        for _ in 0..<20 { _ = st.point(.near) }
        XCTAssertEqual(impliedWinner(st, .far), .unknown)
        XCTAssertEqual(impliedWinner(st, .near), .unknown)
        let sc = scoreMatch([R(.near, 0, .near), R(.unknown, 1, .near), R(.near, 2, .far)])
        XCTAssertEqual(sc[1]["winner"]?.string, "far")
        XCTAssertEqual(sc[1]["uncertain"]?.bool, true)
    }

    func testEmpty() { XCTAssertEqual(scoreMatch([]).count, 0) }
}

// MARK: - rally.py unit cases

final class RallyUnitTests: XCTestCase {
    // test_rally.test_service_court_margin_and_doubles_detection
    func testServiceCourtMarginAndDoubles() {
        let mIn = serviceCourtMargin(x: 3.0, y: 1.0, server: .near, serverY: -0.8, doubles: false)
        let mWrong = serviceCourtMargin(x: 3.0, y: -1.0, server: .near, serverY: -0.8, doubles: false)
        let mShort = serviceCourtMargin(x: 1.5, y: 1.0, server: .near, serverY: -0.8, doubles: false)
        XCTAssert(mIn > 0 && 0 > mWrong && mShort < 0)
        XCTAssertLessThan(serviceCourtMargin(x: 6.3, y: 1.0, server: .near, serverY: -0.8, doubles: true), 0)
        XCTAssertGreaterThan(serviceCourtMargin(x: 6.3, y: 1.0, server: .near, serverY: -0.8, doubles: false), 0)
        XCTAssertGreaterThan(serviceCourtMargin(x: -3.0, y: -1.0, server: .far, serverY: 0.8, doubles: false), 0)
        let sides: [CourtSide] = [.near, .near, .far, .far]
        let obs = (0..<10).flatMap { f in sides.enumerated().map { k, s in
            PlayerObservation(frame: f, trackID: k, bbox: BBox(x1: 0, y1: 0, x2: 0, y2: 0), score: 0.9, side: s)
        } }
        XCTAssertEqual(detectDoubles(obs), true)
        XCTAssertEqual(detectDoubles(obs.enumerated().filter { $0.offset % 2 == 0 }.map(\.element)), false)
        XCTAssertNil(detectDoubles(nil))
    }

    // test_rally.test_landing_outside_the_hall_makes_no_call (robustness F3, same numbers)
    func testLandingOutsideTheHallMakesNoCall() {
        let cfg = RallyConfig()
        let hit = HitEvent(frame: 60, t: 1.0, xyPx: [0, 0], hitter: .far, confidence: 0.9)
        func seg(_ land: SIMD2<Double>) -> ShotSegment3D {
            ShotSegment3D(startFrame: 60, endFrame: 110, t0: 1.0, p0: [5.0, 0.0, 1.0], v0: [-20.0, 0.0, 3.0],
                          rmsePx: 1.0, frames: Array(60...110), positions: Array(repeating: .zero, count: 51),
                          landingXY: land, landingT: 2.0, netCrossingHeight: 2.0, confidence: 0.6,
                          netCrossingY: 0.0, endReason: .floor)
        }
        let o = outcome3D(hit: hit, seg: seg([-11.1, -0.3]), sigma: nil, tLast: 2.0, isServe: false, doubles: false,
                          serverY: nil, cfg: cfg, sideP: 1.0)
        XCTAssertEqual(o.winner, .unknown); XCTAssertEqual(o.endReason, .unknown); XCTAssertNil(o.landingXY)
        XCTAssertEqual(o.source, "implausible_landing"); XCTAssertEqual(o.confidence, 0.0); XCTAssertEqual(o.endT, 2.0)
        let ok = outcome3D(hit: hit, seg: seg([-(CourtModel.halfLength + 2.9), -0.3]), sigma: nil, tLast: 2.0,
                           isServe: false, doubles: false, serverY: nil, cfg: cfg, sideP: 1.0)
        XCTAssertEqual(ok.endReason, .out); XCTAssertEqual(ok.winner, .near); XCTAssertEqual(ok.source, "3d")
        let wide = outcome3D(hit: hit, seg: seg([-4.0, CourtModel.doublesHalfWidth + 3.2]), sigma: nil, tLast: 2.0,
                             isServe: false, doubles: false, serverY: nil, cfg: cfg, sideP: 1.0)
        XCTAssertEqual(wide.source, "implausible_landing")
        var loose = cfg; loose.landingOutsideMaxM = 5.0
        XCTAssertEqual(outcome3D(hit: hit, seg: seg([-11.1, -0.3]), sigma: nil, tLast: 2.0, isServe: false,
                                 doubles: false, serverY: nil, cfg: loose, sideP: 1.0).endReason, .out)
        XCTAssertTrue(landingPlausible(CourtModel.halfLength + 3.0, 0.0, cfg))
        XCTAssertFalse(landingPlausible(.nan, 0.0, cfg))
    }

    // test_rally.test_degenerate_inputs
    func testDegenerateInputs() {
        let t = (0..<300).map { Double($0) / 30 }
        let meta = VideoMeta(fps: 30, width: 1280, height: 720, nFrames: 300, timestamps: t)
        let empty = Track2D(t: [], xy: [], conf: [], visible: [], interpolated: [])
        XCTAssertTrue(segmentRallies(track: empty, meta: meta, hits: [], segments: []).isEmpty)
        let nan = Track2D(t: t, xy: Array(repeating: SIMD2(.nan, .nan), count: 300), conf: Array(repeating: 0, count: 300),
                          visible: Array(repeating: false, count: 300), interpolated: Array(repeating: false, count: 300))
        XCTAssertTrue(segmentRallies(track: nan, meta: meta, hits: [], segments: []).isEmpty)
        XCTAssertTrue(segmentRallies(track: nan, meta: meta, hits: [HitEvent(frame: 900, t: 30, xyPx: [0, 0])],
                                     segments: []).isEmpty)
        XCTAssertTrue(segmentRallies(track: nil, meta: meta, hits: [], segments: []).isEmpty)
    }

    // test_rally.test_hitless_activity_fallback (on the exported 2-D-only world clip)
    func testHitlessActivityFallback() throws {
        let cases = try Fixture.load([RallyCase].self, "world")
        guard let c = cases.first(where: { $0.name.hasSuffix("_2d") }) else { throw XCTSkip("no 2-D world case") }
        let pred = segmentRallies(track: c.track, meta: c.meta, hits: [], segments: [])
        XCTAssertFalse(pred.isEmpty)
        XCTAssert(pred.allSatisfy { $0.shots.isEmpty && $0.winner == .unknown && $0.confidence == 0.3 })
        XCTAssert(pred.allSatisfy { $0.startFrame <= $0.endFrame })
    }

    // invariants of test_rally.test_segmentation_quality on every exported case
    func testInvariants() throws {
        for f in ["sim_e2e", "sim_match", "world"] {
            for c in try Fixture.load([RallyCase].self, f) {
                let pred = segmentRallies(track: c.track, meta: c.meta, hits: c.hits, segments: c.segments,
                                          camera: c.camera, players: c.playersRally, landingSigmaM: c.landingSigmaM)
                for p in pred {
                    XCTAssertLessThanOrEqual(p.startFrame, p.endFrame)
                    XCTAssert((0...1).contains(p.confidence))
                    XCTAssert(zip(p.shots, p.shots.dropFirst()).allSatisfy { $0.hit.frame < $1.hit.frame })
                    // test_shots_carry_their_segments: first shot is 'serve' unless the serve was missed
                    if let s0 = p.shots.first { XCTAssertNotEqual(s0.shotType == .serve, p.serveMissed) }
                }
            }
        }
    }
}

// MARK: - shots.py

final class ShotsTests: XCTestCase {
    static let tol = 1e-6

    private func fixture() throws -> ShotsFixture { try Fixture.load(ShotsFixture.self, "shots_cases") }

    private static let featureKeys: [(String, KeyPath<ShotClassifier.FlightFeatures, Double>)] = [
        ("contact_height", \.contactHeight), ("contact_depth", \.contactDepth), ("speed_kmh", \.speedKmh),
        ("elevation_deg", \.elevationDeg), ("apex", \.apex), ("net_clearance", \.netClearance),
        ("landing_depth", \.landingDepth), ("descent_deg", \.descentDeg), ("flight_time", \.flightTime),
    ]

    // test_shots.test_accuracy_clean_3d / test_accuracy_noisy_3d: parity + the Python gates
    func testClassifySegmentParityAndAccuracy() throws {
        let fx = try fixture()
        var m = Mismatches()
        for (name, set, gate) in [("clean", fx.flights, 0.90), ("noisy", fx.noisy, 0.85)] {
            var ok = 0
            for (k, c) in set.enumerated() {
                let (lab, conf, f) = ShotClassifier.classifySegment(c.segment, poseOverhead: nil, isFirst: c.isFirst)
                if lab != c.label { m.add("\(name)[\(k)] \(c.kind): \(lab) vs \(c.label)") }
                if !close(conf, c.conf, Self.tol) { m.add("\(name)[\(k)] conf \(conf) vs \(c.conf)") }
                if let feats = c.features, let f {
                    for (key, kp) in Self.featureKeys where !close(f[keyPath: kp], feats[key] ?? nil ?? .nan, 1e-9) {
                        m.add("\(name)[\(k)] feature \(key) \(f[keyPath: kp]) vs \(String(describing: feats[key]))")
                    }
                }
                if lab.rawValue == String(c.kind.split(separator: "_")[0]) { ok += 1 }
                XCTAssert((0...1).contains(conf))
            }
            let acc = Double(ok) / Double(set.count)
            print("[shots] \(name) accuracy \(acc) on \(set.count) flights")
            XCTAssertGreaterThanOrEqual(acc, gate)
        }
        XCTAssert(m.isEmpty, m.items.joined(separator: "\n"))
    }

    // test_classify_rally_end_to_end / handedness / fallback / late back-prop / serve_only
    func testClassifyShotsRallies() throws {
        let fx = try fixture()
        var m = Mismatches()
        for c in fx.rallies {
            var hand: [Int: Bool]? = nil
            if let h = c.handedness {
                hand = Dictionary(uniqueKeysWithValues: h.map { (Int($0[0].double!), $0[1].bool!) })
            }
            let got = classifyShots(c.rally, camera: nil, players: c.players, handedness: hand)
            compareRallies([got], [c.expected], tol: Self.tol, &m, c.name)
        }
        XCTAssert(m.isEmpty, m.items.joined(separator: "\n"))
        let pose = fx.rallies.first { $0.name == "pose" }!
        let out = classifyShots(pose.rally, camera: nil, players: pose.players)
        XCTAssertEqual(out.shots.map(\.shotType), [.serve, .clear, .smash, .block, .net, .lift, .drop])
        XCTAssertEqual(out.shots.map(\.strokeSide), [.forehand, .overhead, .overhead, .backhand, .forehand, .backhand, .overhead])
        XCTAssert(pose.rally.shots.allSatisfy { $0.shotType == .unknown })        // input not mutated
    }

    // test_stroke_side_projected_skeletons + wrist_above_head
    func testStrokeSideParity() throws {
        let fx = try fixture()
        for (k, c) in fx.strokes.enumerated() {
            XCTAssertEqual(ShotClassifier.strokeSide(c.kps, overhead: 0.0, rightHanded: c.right), c.side, "stroke[\(k)]")
            let (p, idx) = ShotClassifier.wristAboveHead(c.kps)
            XCTAssertEqual(p, c.above[0] ?? .nan, accuracy: Self.tol, "above[\(k)]")
            XCTAssertEqual(idx.map(Double.init), c.above[1] ?? nil)
        }
    }

    func testContactStateAndOverheadProb() throws {
        let fx = try fixture()
        for c in fx.contact {
            let (p, v) = ShotClassifier.contactState(c.segment, tHit: c.tHit)
            for i in 0..<3 {
                XCTAssertEqual(p[i], c.p[i], accuracy: 1e-12)
                XCTAssertEqual(v[i], c.v[i], accuracy: 1e-12)
            }
        }
        for c in fx.overhead {
            XCTAssertEqual(ShotClassifier.overheadProb(contactHeight: c.h, poseOverhead: c.po), c.p, accuracy: 1e-15)
        }
    }

    // deterministic cases from test_shots.py rebuilt in Swift
    private func seg(_ p0: SIMD3<Double>, _ v0: SIMD3<Double>, conf: Double = 0.9) -> ShotSegment3D {
        ShotSegment3D(startFrame: 0, endFrame: 10, t0: 0, p0: p0, v0: v0, rmsePx: 1, frames: [0, 1, 2],
                      positions: Array(repeating: .zero, count: 3), confidence: conf)
    }

    private func v(_ kmh: Double, _ el: Double, _ toward: Double = 1) -> SIMD3<Double> {
        let s = kmh / 3.6, e = el * .pi / 180
        return [toward * s * cos(e), 0, s * sin(e)]
    }

    func testDecisionBoundaries() {
        let cls = { (p: SIMD3<Double>, v: SIMD3<Double>) in ShotClassifier.classifySegment(self.seg(p, v)) }
        let hi = cls([-5.0, 0, 2.8], v(300, -12)), lo = cls([-5.0, 0, 2.15], v(145, -5))
        XCTAssert(hi.0 == .smash && hi.1 > 0.6 && lo.1 < hi.1)
        XCTAssertEqual(cls([-5.5, 0, 2.7], v(260, -10)).0, .smash)
        XCTAssertEqual(cls([-5.5, 0, 2.7], v(75, -8)).0, .drop)
        XCTAssertEqual(cls([-1.5, 0, 2.3], v(160, -20)).0, .kill)
        let mid: [(SIMD3<Double>, SIMD3<Double>, ShotType)] = [
            ([-2.0, 0, 1.8], v(80, 38), .lift), ([-2.0, 0, 2.5], v(85, 38), .clear),
            ([-4.0, 0, 1.6], v(45, 15), .drop), ([-4.0, 0, 0.8], v(50, 30), .block),
            ([-4.0, 0, 0.9], v(130, 14), .drive), ([-1.2, 0, 1.6], v(22, 15), .net),
            ([-1.5, 0, 0.5], v(42, 45), .push)]
        for (p, vv, want) in mid {
            let (lab, conf, _) = cls(p, vv)
            XCTAssertEqual(lab, want, "\(p)")
            XCTAssertGreaterThan(conf, 0.5, "\(p)")
        }
    }

    func testBackpropStopsBeforeBlowUp() {
        var s = seg([-5.0, 0.0, 2.0], v(200, -5))
        s.t0 = 0.25
        let (p, vv) = ShotClassifier.contactState(s, tHit: 0.0)
        XCTAssertLessThanOrEqual(simd_length(vv), 500 / 3.6 + 1e-6)
        XCTAssert(p.isFinite)
    }

    func testEmptyAndMissing() {
        XCTAssertTrue(classifyShots(RallyV2(startFrame: 0, endFrame: 0), camera: nil, players: nil).shots.isEmpty)
        let r = RallyV2(startFrame: 0, endFrame: 5, shots: [ShotEventV2(hit: HitEvent(frame: 0, t: 0, xyPx: [1, 1]), segment: nil)])
        XCTAssertEqual(classifyShots(r, camera: nil, players: nil).shots[0].shotType, .serve)
    }
}
