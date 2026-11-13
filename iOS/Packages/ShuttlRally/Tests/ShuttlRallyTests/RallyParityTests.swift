import Foundation
import XCTest
import ShuttlCore
@testable import ShuttlRally

/// Parity with the Python reference on exported pipeline inputs
/// (backend/scripts/swift_fixtures/rally.py). Acceptance: identical rally
/// boundaries, end_reason, winner, server, shot types and stroke sides;
/// confidences within 1e-6; score_match / summarize / rally dicts identical as JSON.
final class RallyParityTests: XCTestCase {
    static let tol = 1e-6

    struct Result {
        var cases: [RallyCase]
        var rallies: [[RallyV2]]
        var typed: [[RallyV2]]
    }

    private func run(_ file: String) throws -> Result {
        let cases = try Fixture.load([RallyCase].self, file)
        var rallies: [[RallyV2]] = [], typed: [[RallyV2]] = []
        for c in cases {
            let got = segmentRallies(track: c.track, meta: c.meta, hits: c.hits, segments: c.segments,
                                     camera: c.camera, players: c.playersRally, landingSigmaM: c.landingSigmaM)
            rallies.append(got)
            typed.append(got.map { classifyShots($0, camera: c.camera, players: c.playersShots) })
        }
        return Result(cases: cases, rallies: rallies, typed: typed)
    }

    private func check(_ file: String, minCases: Int = 1) throws {
        let res = try run(file)
        XCTAssertGreaterThanOrEqual(res.cases.count, minCases)
        var m = Mismatches()
        var nR = 0, nS = 0, maxConfErr = 0.0
        for (k, c) in res.cases.enumerated() {
            let got = res.rallies[k], typed = res.typed[k]
            compareRallies(got, c.rallies, tol: Self.tol, &m, "\(c.name) segment_rallies")
            // classify_shots on Python's own rallies (stage in isolation) …
            let typedPy = c.rallies.map { classifyShots($0, camera: c.camera, players: c.playersShots) }
            compareRallies(typedPy, c.shots, tol: Self.tol, &m, "\(c.name) classify_shots")
            // … and end to end on the Swift rallies
            compareRallies(typed, c.shots, tol: Self.tol, &m, "\(c.name) e2e")
            for (g, w) in zip(typed, c.shots) {
                maxConfErr = Swift.max(maxConfErr, Swift.abs(g.confidence - w.confidence),
                                       Swift.abs(g.outcomeConfidence - w.outcomeConfidence),
                                       Swift.abs(g.segmentationConfidence - w.segmentationConfidence))
                for (a, b) in zip(g.shots, w.shots) {
                    maxConfErr = Swift.max(maxConfErr, Swift.abs(a.shotTypeConfidence - b.shotTypeConfidence))
                }
            }
            let score = scoreMatch(typed)
            jsonDiff(.array(score), .array(c.score), path: "\(c.name).score", &m)
            jsonDiff(summarize(typed), c.summary, path: "\(c.name).summary", &m)
            if let dicts = c.rallyDicts {
                let w = c.metaLite.width != 0 ? c.metaLite.width : 1, h = c.metaLite.height != 0 ? c.metaLite.height : 1
                let mine = typed.map { RallySerialization.rallyDict($0, width: w, height: h, camera: c.camera) }
                // path_norm is a re-projection (simd vs numpy matmul: last-bit
                // differences may flip a 4th-decimal rounding); everything else exact
                jsonDiff(.array(mine), .array(dicts), path: "\(c.name).rally_dicts",
                         tol: { $0.contains("path_norm") ? 1.00001e-4 : 0 }, &m)
            }
            if let want = c.refinedSides {
                let r = refineHitters(track: c.track, meta: c.meta, hits: c.hits, segments: c.segments, camera: c.camera)
                if r.hits.map(\.hitter) != want { m.add("\(c.name) refine_hitters sides differ") }
            }
            nR += got.count
            nS += typed.reduce(0) { $0 + $1.shots.count }
        }
        print("[parity] \(file): \(res.cases.count) cases, \(nR) rallies, \(nS) shots, max |Δconf| = \(maxConfErr)")
        XCTAssert(m.isEmpty, "\(file):\n" + m.items.joined(separator: "\n"))
    }

    func testSimE2EParity() throws { try check("sim_e2e", minCases: 12) }
    func testSimMatchParity() throws { try check("sim_match") }
    func testSimCorruptParity() throws { try check("sim_corrupt") }
    func testWorldParity() throws { try check("world") }
    func testRealClipsParity() throws { try check("real", minCases: 5) }

    /// Addendum 3: rally 77495 of the 45-min H.264 / HEVC pair. H.264 keeps its 3-D call (far) against a
    /// contradicting next serve, HEVC takes near from a serve that was only inferred: both capped at
    /// nsConflictConf (0.5 < the app's 0.6), identical to Python.
    func testNextServeConflictParity() throws {
        try check("codec", minCases: 2)
        let res = try run("codec")
        let h = res.rallies[0].first!, v = res.rallies[1].first!
        XCTAssertEqual(h.winner, .far); XCTAssertEqual(h.outcomeSource, "3d"); XCTAssertEqual(h.outcomeConfidence, 0.5)
        XCTAssertEqual(v.winner, .near); XCTAssertEqual(v.outcomeSource, "next_serve"); XCTAssertEqual(v.outcomeConfidence, 0.5)
    }

    /// benchmark.py --stage e2e --n 12: rally F1 of the Swift port on the same
    /// inputs must equal the Python value (0.667 when the benchmark was run).
    func testSimE2ERallyF1() throws {
        let res = try run("sim_e2e")
        var swift = RallyF1(), python = RallyF1()
        for (k, c) in res.cases.enumerated() {
            let truth = (c.truth ?? []).map { ($0[0], $0[1]) }
            swift.add(pred: res.rallies[k].map { ($0.startFrame, $0.endFrame) }, truth: truth)
            python.add(pred: c.rallies.map { ($0.startFrame, $0.endFrame) }, truth: truth)
        }
        print(String(format: "[F1] sim e2e rally F1: swift %.4f (tp %d pred %d true %d), python %.4f",
                     swift.f1, swift.tp, swift.nPred, swift.nTrue, python.f1))
        XCTAssertEqual(swift.tp, python.tp)
        XCTAssertEqual(swift.nPred, python.nPred)
        XCTAssertEqual(swift.f1, python.f1, accuracy: 1e-12)
    }

    /// Release-mode timing (run `swift test -c release`).
    func testTiming() throws {
        var files: [Result] = []
        for f in ["sim_e2e", "sim_match", "sim_corrupt", "world", "real"] {
            if let r = try? run(f) { files.append(r) }
        }
        let cases = files.flatMap(\.cases)
        guard !cases.isEmpty else { throw XCTSkip("no fixtures") }
        var frames = 0, shots = 0
        var tSeg = 0.0, tShots = 0.0, tScore = 0.0, pySeg = 0.0, pyShots = 0.0
        let reps = 3
        for c in cases {
            frames += c.track.n
            pySeg += c.pySeconds?["segment_rallies"] ?? 0
            pyShots += c.pySeconds?["classify_shots"] ?? 0
            var best = (Double.infinity, Double.infinity, Double.infinity)
            for _ in 0..<reps {
                let t0 = DispatchTime.now().uptimeNanoseconds
                let rs = segmentRallies(track: c.track, meta: c.meta, hits: c.hits, segments: c.segments,
                                        camera: c.camera, players: c.playersRally, landingSigmaM: c.landingSigmaM)
                let t1 = DispatchTime.now().uptimeNanoseconds
                let typed = rs.map { classifyShots($0, camera: c.camera, players: c.playersShots) }
                let t2 = DispatchTime.now().uptimeNanoseconds
                _ = scoreMatch(typed); _ = summarize(typed)
                let t3 = DispatchTime.now().uptimeNanoseconds
                best = (Swift.min(best.0, Double(t1 - t0) / 1e9), Swift.min(best.1, Double(t2 - t1) / 1e9),
                        Swift.min(best.2, Double(t3 - t2) / 1e9))
            }
            tSeg += best.0; tShots += best.1; tScore += best.2
            shots += c.shots.reduce(0) { $0 + $1.shots.count }
        }
        print(String(format: "[timing] %d cases, %d frames (%.1f min @30-60 fps), %d shots: segmentRallies %.1f ms (python %.1f ms), classifyShots %.2f ms = %.3f ms/shot (python %.1f ms), score+summary %.2f ms",
                     cases.count, frames, Double(frames) / 60 / 60, shots, tSeg * 1e3, pySeg * 1e3,
                     tShots * 1e3, tShots * 1e3 / Double(Swift.max(shots, 1)), pyShots * 1e3, tScore * 1e3))
    }
}
