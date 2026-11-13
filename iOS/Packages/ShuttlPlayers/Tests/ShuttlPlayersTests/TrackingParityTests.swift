import Foundation
import XCTest
import ShuttlCore
@testable import ShuttlPlayers

/// Acceptance (a): identical raw detections in → identical tracks out.
///  • gated candidates: same survivors, floor position / costs within 1e-9;
///  • observations: same count, frame, track id and side for every one;
///    foot_world within 1e-6 m, bbox within 1e-9 px, keypoints within the fixture rounding;
///  • movement_stats and diagnostics equal (rounded numbers compared exactly).
final class TrackingParityTests: XCTestCase {
    static let footTol = 1e-6

    func testSyntheticCases() throws {
        let cases = Fx.list(try Fx.load("synthetic"))
        XCTAssertGreaterThanOrEqual(cases.count, 8)
        for c in cases { try checkCase(PlayerCase(c)) }
    }

    func testRealTestClips() throws {
        let cases = Fx.list(try Fx.load("real"))
        XCTAssertGreaterThanOrEqual(cases.count, 2)
        for c in cases {
            let pc = try PlayerCase(c)
            XCTAssertNotNil(pc.camera, "\(pc.name): calibrated camera expected")
            try checkCase(pc)
        }
    }

    func checkCase(_ pc: PlayerCase, file: StaticString = #filePath, line: UInt = #line) throws {
        let cands = pc.candidates()
        checkCandidates(pc, cands)
        let tracks = PlayerTracker.assembleTracks(cands, meta: pc.meta, camera: pc.camera, doubles: pc.doubles)
        XCTAssertEqual(tracks.doubles, pc.raw["doubles_auto"] as? Bool, "\(pc.name) doubles")
        let stats = checkObservations(pc, tracks)
        checkStats(pc, tracks)
        checkDiagnostics(pc, tracks)
        print(String(format: "[parity] %@: %d obs, max |Δfoot| %.2e m, max |Δbbox| %.2e px",
                     pc.name, tracks.observations.count, stats.foot, stats.bbox))
    }

    func checkCandidates(_ pc: PlayerCase, _ cands: [PlayerCandidates]) {
        let exp = Fx.list(pc.raw["candidates"]).map(Fx.dict)
        XCTAssertEqual(cands.count, exp.count, "\(pc.name) sampled frames")
        for (c, e) in zip(cands, exp) {
            let tag = "\(pc.name) frame \(c.frame)"
            XCTAssertEqual(c.n, Int(Fx.num(e["n"])), "\(tag) survivors")
            guard c.n == Int(Fx.num(e["n"])) else { continue }
            let foot = Fx.v2(e["foot_px"]), pos = Fx.v2(e["pos"])
            let unary = Fx.nums(e["unary"]), unit = Fx.nums(e["unit"]), noise = Fx.nums(e["noise"])
            for i in 0..<c.n {
                assertClose(c.footPx[i].x, foot[i].x, tol: 1e-9, tag); assertClose(c.footPx[i].y, foot[i].y, tol: 1e-9, tag)
                assertClose(c.pos[i].x, pos[i].x, tol: 1e-9, tag); assertClose(c.pos[i].y, pos[i].y, tol: 1e-9, tag)
                assertClose(c.unary[i], unary[i], tol: 1e-9, "\(tag) unary")
                assertClose(c.unit[i], unit[i], tol: 1e-12, "\(tag) unit")
                assertClose(c.noise[i], noise[i], tol: 1e-9, "\(tag) noise")
            }
        }
    }

    func checkObservations(_ pc: PlayerCase, _ tracks: PlayerTracks) -> (foot: Double, bbox: Double) {
        let exp = Fx.observations(pc.raw["expect"])
        let got = tracks.observations
        XCTAssertEqual(got.count, exp.count, "\(pc.name) observation count")
        var maxFoot = 0.0, maxBox = 0.0, idMismatch = 0
        for (o, e) in zip(got, exp) {
            let tag = "\(pc.name) frame \(o.frame) id \(o.trackID)"
            if o.frame != Int(Fx.num(e["frame"])) || o.trackID != Int(Fx.num(e["track_id"])) || o.side != Fx.side(e["side"]) {
                idMismatch += 1
                if idMismatch <= 5 {
                    XCTFail("\(tag): expected frame \(Fx.num(e["frame"])) id \(Fx.num(e["track_id"])) side \(e["side"] ?? "?"), got side \(o.side)")
                }
                continue
            }
            let eb = Fx.nums(e["bbox"])
            for (a, b) in zip([o.bbox.x1, o.bbox.y1, o.bbox.x2, o.bbox.y2], eb) { maxBox = max(maxBox, abs(a - b)) }
            XCTAssertEqual(o.score, Fx.num(e["score"]), accuracy: 1e-12, tag)
            if let ef = e["foot_world"] as? [Any] {
                guard let fw = o.footWorld else { XCTFail("\(tag): missing foot_world"); continue }
                let ex = Fx.nums(ef)
                maxFoot = max(maxFoot, abs(fw.x - ex[0]), abs(fw.y - ex[1]))
            } else {
                XCTAssertNil(o.footWorld, "\(tag): foot_world should be None")
            }
            if let ek = e["keypoints"] as? [Any], let kp = o.keypoints {
                for (k, r) in zip(kp, Fx.rows(ek)) {
                    XCTAssertEqual(k.x, r[0], accuracy: 1e-3, tag)
                    XCTAssertEqual(k.y, r[1], accuracy: 1e-3, tag)
                    XCTAssertEqual(k.conf, r[2], accuracy: 1e-3, tag)
                }
            }
        }
        XCTAssertEqual(idMismatch, 0, "\(pc.name): frame/track/side mismatches")
        XCTAssertLessThanOrEqual(maxFoot, Self.footTol, "\(pc.name) foot_world")
        XCTAssertLessThanOrEqual(maxBox, 1e-9, "\(pc.name) bbox")
        return (maxFoot, maxBox)
    }

    func checkStats(_ pc: PlayerCase, _ tracks: PlayerTracks) {
        for side in [CourtSide.near, .far] {
            let e = Fx.dict(Fx.dict(pc.raw["expect_stats"])[side.rawValue])
            let s = Movement.stats(tracks, side: side)
            compareStats(s, e, "\(pc.name) \(side)")
        }
    }

    func checkDiagnostics(_ pc: PlayerCase, _ tracks: PlayerTracks) {
        let e = Fx.dict(pc.raw["expect_diagnostics"])
        guard let d = tracks.diagnostics else { return XCTFail("\(pc.name): no diagnostics") }
        XCTAssertEqual(d.sampledFrames, Int(Fx.num(e["sampled_frames"])), pc.name)
        XCTAssertEqual(d.doubles, e["doubles"] as? Bool, pc.name)
        let cov = Fx.dict(e["coverage"])
        XCTAssertEqual(Set(d.coverage.keys.map(String.init)), Set(cov.keys), "\(pc.name) coverage ids")
        for (k, v) in d.coverage { XCTAssertEqual(v, Fx.num(cov[String(k)]), "\(pc.name) coverage \(k)") }
    }
}

/// Exact comparison of Python-rounded movement stats.
func compareStats(_ s: MovementStats, _ e: [String: Any], _ tag: String,
                  file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(s.distanceM, Fx.num(e["distance_m"]), "\(tag) distance_m", file: file, line: line)
    XCTAssertEqual(s.avgSpeedMs, Fx.num(e["avg_speed_ms"]), "\(tag) avg_speed_ms", file: file, line: line)
    XCTAssertEqual(s.maxSpeedMs, Fx.num(e["max_speed_ms"]), "\(tag) max_speed_ms", file: file, line: line)
    XCTAssertEqual(s.courtCoverage, Fx.num(e["court_coverage"]), "\(tag) court_coverage", file: file, line: line)
    XCTAssertEqual(s.trackedS, Fx.num(e["tracked_s"]), "\(tag) tracked_s", file: file, line: line)
    XCTAssertEqual(s.heatmap, Fx.rows(e["heatmap"]), "\(tag) heatmap", file: file, line: line)
    if let players = e["players"] as? [String: Any] {
        XCTAssertEqual(Set(s.players.keys.map(String.init)), Set(players.keys), "\(tag) players", file: file, line: line)
        for (k, v) in s.players { compareStats(v, Fx.dict(players[String(k)]), "\(tag) player \(k)", file: file, line: line) }
    }
}
