import Foundation
import XCTest
@testable import ShuttlLinker
import ShuttlCore

/// Cases ported from backend/tests/test_linker.py (the synthetic generators live in
/// Python; their outputs are in linker_sim.json and checked here for the same
/// properties the Python tests assert).
final class BehaviourTests: XCTestCase {
    let W = 1920, H = 1080

    private func meta(_ n: Int, _ ts: [Double]? = nil) -> VideoMeta {
        VideoMeta(fps: 30, width: W, height: H, nFrames: n, timestamps: ts ?? (0..<n).map { Double($0) / 30 })
    }

    // test_degenerate_inputs
    func testDegenerateInputs() throws {
        let tr0 = try linkCandidates([], meta: meta(0, []))
        XCTAssertEqual(tr0.xy.count, 0)
        let n = 50
        let empty = try linkCandidates([[ShuttleCandidate]](repeating: [], count: n), meta: meta(n))
        XCTAssertTrue(empty.xy.allSatisfy { $0.x.isNaN }); XCTAssertFalse(empty.visible.contains(true))
        var one = [[ShuttleCandidate]](repeating: [], count: n)
        one[10] = [ShuttleCandidate(x: 100, y: 100, score: 0.9)]
        XCTAssertFalse(try linkCandidates(one, meta: meta(n)).visible.contains(true))   // a lone blip is not a track
        // missing / duplicate timestamps are repaired, not fatal
        let badT = (0..<(n - 5)).map { Double($0) / 30 } + [Double](repeating: Double(n - 6) / 30, count: 5)
        let line = (0..<n).map { [ShuttleCandidate(x: 100 + 5 * Double($0), y: 200 + 2 * Double($0), score: 0.9)] }
        let tr = try linkCandidates(line, meta: meta(n, badT))
        XCTAssertGreaterThanOrEqual(tr.visible.filter { $0 }.count, n - 6)
        XCTAssertTrue(zip(tr.t.dropFirst(), tr.t).allSatisfy { $0 > $1 })
        _ = findKinks(tr)
    }

    // invariants of test_link_accuracy_per_case / test_raw_xy_is_the_adopted_detection
    func testTrackInvariants() throws {
        for c in try loadCases("linker_sim") + loadCases("linker_real") {
            let cands = c.cands.candidates
            let tr = try linkCandidates(cands, meta: c.meta.meta, sceneCuts: c.scene_cuts)
            XCTAssertEqual(tr.n, cands.count, c.name)
            XCTAssertEqual(tr.xy.count, tr.n); XCTAssertEqual(tr.rawXY?.count, tr.n)
            for f in 0..<tr.n {
                XCTAssertFalse(tr.visible[f] && tr.interpolated[f], "\(c.name) \(f)")
                if !tr.visible[f] { XCTAssertEqual(tr.conf[f], 0, "\(c.name) \(f)") }
                if tr.visible[f] || tr.interpolated[f] { XCTAssertTrue(tr.xy[f].x.isFinite && tr.xy[f].y.isFinite) }
                let raw = tr.rawXY![f]
                XCTAssertEqual(raw.x.isFinite, tr.visible[f], "\(c.name) \(f)")
                // raw_xy is the adopted detection itself — except in a repeated-frame slot, which
                // (like Python's _expand) carries the previous slot's state
                let expandedRepeat = f > 0 && tr.rawXY![f - 1] == raw && tr.xy[f - 1] == tr.xy[f]
                if raw.x.isFinite && !expandedRepeat {
                    let d = cands[f].map { hypot($0.x - raw.x, $0.y - raw.y) }.min() ?? .infinity
                    XCTAssertLessThan(d, 1e-6, "\(c.name) \(f)")
                }
            }
        }
    }

    // test_never_links_across_scene_cut (explicit and inferred)
    func testNeverLinksAcrossSceneCut() throws {
        let cases = try loadCases("linker_sim").filter { $0.name.hasPrefix("sim_cut") }
        XCTAssertEqual(cases.count, 2)
        for c in cases {
            let r = try XCTUnwrap(c.cut_range)
            let tr = try linkCandidates(c.cands.candidates, meta: c.meta.meta, sceneCuts: c.scene_cuts)
            for f in r[0]..<r[1] { XCTAssertTrue(tr.xy[f].x.isNaN, "\(c.name) frame \(f)") }
            let cands = c.cands.candidates
            let t = linkerTimestamps(c.meta.meta, cands.count)
            XCTAssertEqual(try detectCuts(cands, t, 1920, 1080, LinkerConfig()), [r[0]])
        }
        // normal clips never show a cut
        for c in try loadCases("linker_sim") where !c.name.hasPrefix("sim_cut") && !c.name.hasPrefix("deg") {
            let cands = c.cands.candidates
            let t = linkerTimestamps(c.meta.meta, cands.count)
            XCTAssertEqual(try detectCuts(cands, t, Double(c.meta.width), Double(c.meta.height), LinkerConfig()), [], c.name)
        }
    }

    // test_repeated_frames_25p_in_30p: the periodic repeats are found (and match Python)
    func testRepeatedFrames() throws {
        let c = try XCTUnwrap(try loadCases("linker_sim").first { $0.name == "sim_repeat_25p_in_30p" })
        let cands = c.cands.candidates, cfg = LinkerConfig()
        let t = linkerTimestamps(c.meta.meta, cands.count)
        let a = try associate(cands, t, 1920, 1080, cfg, shot0: nil)
        let rep = try XCTUnwrap(findRepeats(t, a.xy, probeCandidates(cands, a.xy, cfg, 1920, 1080), hypot(1920, 1080), cfg))
        XCTAssertEqual(flatnonzero(rep.isDup), c.inter?.rep_dup)
        XCTAssertGreaterThan(rep.isDup.filter { $0 }.count, 60)
        let ref = (c.inter?.rep_clock ?? []).map { $0 ?? .nan }
        for (x, y) in zip(rep.clock, ref) { XCTAssertEqual(x, y, accuracy: 1e-12) }
    }

    // test_weak_walking_distractors_not_adopted: a held shuttle is kept iff seen strongly
    func testHeldShuttleStrongVsWeak() throws {
        let cs = try loadCases("linker_sim")
        let strong = try XCTUnwrap(cs.first { $0.name == "sim_weak_walk_strong_held" })
        let weak = try XCTUnwrap(cs.first { $0.name == "sim_weak_walk_weak_held" })
        let ts = try linkCandidates(strong.cands.candidates, meta: strong.meta.meta)
        let tw = try linkCandidates(weak.cands.candidates, meta: weak.meta.meta)
        // kept = fraction of held frames with the track < 10 px from the held shuttle
        func kept(_ tr: Track2D, _ c: FxCase) throws -> Double {
            let r = try XCTUnwrap(c.held), hp = pairs(try XCTUnwrap(c.held_xy))
            var k = 0
            for (j, f) in (r[0]..<r[1]).enumerated() where tr.xy[f].x.isFinite && hypot(tr.xy[f].x - hp[j].x, tr.xy[f].y - hp[j].y) < 10 { k += 1 }
            return Double(k) / Double(r[1] - r[0])
        }
        XCTAssertGreaterThanOrEqual(try kept(ts, strong), 0.8)
        XCTAssertLessThanOrEqual(try kept(tw, weak), 0.05)
    }

    // test_resolution_invariance: config is diagonal-relative
    func testScaledConfigParams() {
        let (p, gap) = dpParams(LinkerConfig(), hypot(1280, 720), (0..<100).map { Double($0) / 60 })
        XCTAssertEqual(gap, 18); XCTAssertEqual(p.kgap, 6)
        XCTAssertEqual(p.vmax, 8 * hypot(1280, 720), accuracy: 1e-9)
    }
}
