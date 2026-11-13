import Foundation
import XCTest
import ShuttlCore
@testable import ShuttlEval

/// Port of backend/tests/test_sim.py on the native Swift RNG (statistical / physical
/// properties; exact parity with numpy draws is SimReplayTests).
final class SimNativeTests: XCTestCase {
    let src = NativeRandomSource()

    lazy var rallies: [SimScenario] = {
        var out = (0..<8).map { try! Sim.simulateRally(seed: $0, fps: 30, doubles: $0 % 4 == 3, source: src) }
        out += (0..<6).map { try! Sim.simulateRally(seed: 100 + $0, fps: 30, nShots: 1, source: src) }   // serve-only finals
        return out
    }()

    func testDeterministic() throws {
        let a = try Sim.simulateRally(seed: 5, fps: 30, source: src), b = try Sim.simulateRally(seed: 5, fps: 30, source: src)
        XCTAssertEqual(a.shuttleWorld.flatMap(\.row).map(\.bitPattern), b.shuttleWorld.flatMap(\.row).map(\.bitPattern))
        XCTAssertEqual(a.hits.map(\.frame), b.hits.map(\.frame))
    }

    func testSimulatePositionsMatchesPhysics() {
        let p0 = SIMD3(-5.0, 1.0, 2.7), v0 = SIMD3(60.0, -5.0, -8.0)
        let ts = NP.linspace(0, 1.2, 37)
        let a = SimPhysics.simulatePositions(p0: p0, v0: v0, times: ts, L: 4.5)
        let b = Physics.simulate(p0: p0, v0: v0, times: ts, dragLength: 4.5)
        for (x, y) in zip(a, b) { XCTAssertLessThan(simd_length_d(x - y), 1e-9) }
    }

    func testProjectionConsistentWithCamera() {
        for s in rallies {
            let w = Double(s.imageSize.width), h = Double(s.imageSize.height)
            for i in 0..<s.n {
                if s.inPlay[i] { XCTAssertLessThan(simd_length_d(s.shuttlePx[i] - s.camera.project(s.shuttleWorld[i])), 1e-6) }
                else { XCTAssertFalse(s.shuttlePx[i].x.isFinite) }
                if s.visible[i] { XCTAssert(s.shuttlePx[i].x >= 0 && s.shuttlePx[i].x < w && s.shuttlePx[i].y >= 0 && s.shuttlePx[i].y < h) }
                XCTAssertFalse(s.visible[i] && s.occluded[i])
            }
            for p in s.players {
                for i in 0..<s.n {
                    let f = s.camera.project(SIMD3(p.foot[i].x, p.foot[i].y, 0))
                    guard f.isFinite else { continue }
                    let b = p.bbox[i]
                    XCTAssert(f.x >= b.x1 - 1e-6 && f.x <= b.x2 + 1e-6 && b.y2 - b.y1 > 0)
                }
            }
        }
    }

    func testShotsLandWhereIntended() {
        var n = 0
        for s in rallies { for sh in s.shots where sh.inOut != "net" {
            let err = hypot(sh.landingXY.x - sh.targetXY.x, sh.landingXY.y - sh.targetXY.y)
            XCTAssertLessThan(err, 0.05, "seed \(s.seed) shot \(sh.index) \(sh.shotType)")
            n += 1
        } }
        XCTAssertGreaterThan(n, 30)
    }

    func testTrajectoryIsTheDragODE() {
        let s = rallies[0]
        for sh in s.shots {
            let fr = (sh.frame..<max(sh.frame, sh.endFrame)).filter { s.timestamps[$0] >= sh.t0 }
            let exp = Physics.simulate(p0: sh.p0, v0: sh.v0, times: fr.map { s.timestamps[$0] - sh.t0 }, dragLength: sh.dragLength)
            for (f, e) in zip(fr, exp) { XCTAssertLessThan(simd_length_d(s.shuttleWorld[f] - e), 1e-6) }
        }
        XCTAssert(rallies.allSatisfy { r in r.shots.allSatisfy { abs($0.dragLength / Physics.dragLength - 1) <= 0.0801 } })
    }

    func testTerminalVelocity() throws {
        let L = Physics.dragLength
        let P = SimPhysics.simulatePositions(p0: SIMD3(0, 0, 60), v0: .zero, times: [4.0, 4.01], L: L)
        XCTAssertLessThan(abs(simd_length_d(P[1] - P[0]) / 0.01 - Physics.terminalVelocity), 0.05)
        for k in 0..<12 {
            for sh in try Sim.simulateRally(seed: k, fps: 30, source: src).shots where ["clear", "lift"].contains(sh.shotType) && sh.landingT.isFinite {
                let tl = sh.landingT - sh.t0
                let Q = SimPhysics.simulatePositions(p0: sh.p0, v0: sh.v0, times: [tl - 0.02, tl - 0.01], L: sh.dragLength)
                let v = simd_length_d(Q[1] - Q[0]) / 0.01, vt = (Physics.g * sh.dragLength).squareRoot()
                XCTAssert(0.5 * vt < v && v < 1.15 * vt)
            }
        }
    }

    func testShotParametersRealistic() throws {
        var smash: [(Double, Double)] = [], clears: [Double] = [], serves: [Double] = []
        for k in 0..<30 {
            for sh in try Sim.simulateRally(seed: k, fps: 30, source: src).shots {
                if sh.shotType == "smash" { smash.append((sh.speed0Kmh, sh.p0.z)) }
                if sh.shotType == "clear" { clears.append(sh.speed0Kmh) }
                if sh.shotType == "serve" { serves.append(sh.p0.z) }
            }
        }
        XCTAssertGreaterThanOrEqual(smash.count, 5)
        XCTAssert(smash.allSatisfy { $0.0 > 140 && $0.0 <= 421 && $0.1 > 1.9 })
        XCTAssertGreaterThan(NP.median(smash.map(\.0)), 200)
        XCTAssert(clears.allSatisfy { $0 > 50 && $0 < 300 })
        XCTAssert(serves.allSatisfy { $0 <= 1.15 })
    }

    func testHitsAlternateAndRallyTruth() {
        for s in rallies {
            let sides = s.hits.map(\.hitter)
            XCTAssert(zip(sides, sides.dropFirst()).allSatisfy { $0 != $1 })
            XCTAssert(zip(s.hits, s.hits.dropFirst()).allSatisfy { $1.t > $0.t })
            XCTAssertEqual(s.shots[0].shotType, "serve")
            var held: [Int: SIMD3<Double>] = [:]
            for r in s.resting { for f in r.from...r.to { held[f] = r.p } }
            for h in s.hits {
                let p = s.shuttleWorld[h.frame].isFinite ? s.shuttleWorld[h.frame] : held[h.frame]!
                XCTAssertLessThan(simd_length_d(p - h.pos), 60 / s.fps + 0.05)
            }
            let r = s.rallies[0], last = s.shots[s.shots.count - 1]
            XCTAssertEqual(r.endReason, last.outcome); XCTAssertEqual(last.outcome, last.inOut)
            if r.endReason == "in" || r.endReason == "out" {
                XCTAssertEqual(r.margin! >= 0, r.endReason == "in")
                XCTAssertEqual(r.winner, r.endReason == "in" ? last.hitter : (last.hitter == "near" ? "far" : "near"))
            } else {
                XCTAssertNil(r.margin); XCTAssertLessThan(last.netCrossingHeight, CourtModel.netHeightCentre)
            }
            for sh in s.shots.dropLast() { XCTAssert(sh.outcome == "returned" && sh.netCrossingHeight > CourtModel.netHeightCentre) }
            let after = (r.endFrame + 2..<s.n).filter { s.inPlay[$0] }
            if r.endReason == "net" { XCTAssert(after.isEmpty || Double(after.last! - r.endFrame - 2) < 1.5 * s.fps) }
            else { XCTAssert(after.isEmpty) }
        }
    }

    func testServeMargin() {
        let m = { (x: Double, y: Double, d: Bool) in SimEngine.serveMargin(x, y, recvSign: 1, court: 1, doubles: d) }
        XCTAssertGreaterThan(m(3.0, -1.0, false), 0); XCTAssertLessThan(m(1.5, -1.0, false), 0)
        XCTAssertLessThan(m(3.0, 1.0, false), 0); XCTAssertGreaterThan(m(6.3, -1.0, false), 0)
        XCTAssertLessThan(m(6.3, -1.0, true), 0)
    }

    func testNShotsAndDoubles() throws {
        let s = try Sim.simulateRally(seed: 3, fps: 30, nShots: 6, doubles: true, source: src)
        XCTAssert(s.shots.count <= 6 && s.shots.count >= 4)
        XCTAssertEqual(Set(s.players.map(\.key)), ["near", "far", "near2", "far2"])
    }

    func testRandomCameraStyles() throws {
        let size = ImageSize(width: 1280, height: 720)
        for style in ["behind", "elevated_side", "broadcast"] {
            for k in 0..<5 {
                let cam = try SimCameras.randomCamera(src.stream([k, 3]), imageSize: size, style: style)
                let c = cam.center
                for uv in cam.project([SIMD3(CourtModel.halfLength, 0, 0), SIMD3(0, 0, CourtModel.netHeightCentre)]) {
                    XCTAssert(uv.x >= 0 && uv.x < 1280 && uv.y >= 0 && uv.y < 720, "\(style) \(uv)")
                }
                XCTAssert(c.x < 0 && c.z > 0.9)
                if style == "behind" {
                    XCTAssert(c.z >= 1.0 && c.z <= 2.0 && c.x >= -CourtModel.halfLength - 6.01 && c.x <= -CourtModel.halfLength - 0.99)
                    XCTAssert(cam.K[0, 0] >= 0.8 * 1280 && cam.K[0, 0] <= 1280)
                }
                if style == "broadcast" {
                    let cr = cam.project(SimCameras.corners)
                    XCTAssert(c.z >= 6 && c.z <= 10 && cam.K[0, 0] > 1.5 * 1280)
                    XCTAssert(cr.allSatisfy { $0.x >= 0 && $0.x < 1280 && $0.y >= 0 && $0.y < 720 })
                    XCTAssertGreaterThan(abs(cr[1].x - cr[0].x), 0.5 * 1280)
                }
            }
        }
        XCTAssertThrowsError(try SimCameras.randomCamera(src.stream(1), style: "fisheye"))
    }

    func testCorruptDetectionsNoiseModel() throws {
        let cam = try SimCameras.randomCamera(src.stream(1), imageSize: ImageSize(width: 1280, height: 720), style: "broadcast")
        let s = try Sim.simulateMatch(seed: 2, nRallies: 3, camera: cam, fps: 30, source: src)
        let vis = (0..<s.n).filter { s.visible[$0] }
        var rates: [Double] = [], errs: [Double] = []
        for seed in 0..<3 {
            let c = SimNoise.corruptDetections(s, rng: src.stream([seed, 50]))
            XCTAssertEqual(c.count, s.n)
            XCTAssert(c.allSatisfy { $0.count <= 3 && zip($0, $0.dropFirst()).allSatisfy { $0.score >= $1.score } })
            let d = vis.map { i in c[i].map { hypot($0.x - s.shuttlePx[i].x, $0.y - s.shuttlePx[i].y) }.min() ?? .infinity }
            rates.append(Double(d.filter { $0 < 8 }.count) / Double(d.count))
            errs += d.filter { $0 < 8 }
        }
        XCTAssertGreaterThan(vis.count, 150)
        XCTAssert(NP.mean(rates) > 0.82 && NP.mean(rates) < 0.96, "\(rates)")
        XCTAssert(NP.median(errs) > 1.0 && NP.median(errs) < 3.5, "\(NP.median(errs))")
        let clean = SimNoise.corruptDetections(s, rng: src.stream(0), sigmaPx: 0, missRate: 0, falsePositiveRate: 0, deadShuttleRate: 0)
        XCTAssert((0..<s.n).allSatisfy { (clean[$0].count == 1) == s.visible[$0] })
    }

    func testTruthConverters() throws {
        let s = try Sim.simulateRally(seed: 4, fps: 30, source: src)
        let tr = s.truthTrack()
        XCTAssertEqual(tr.n, s.n); XCTAssertEqual(tr.visible, s.visible)
        XCTAssertEqual(s.truthHits().map(\.hitter.rawValue), s.hits.map(\.hitter))
        let segs = s.truthSegments()
        XCTAssertEqual(segs.count, s.shots.count)
        XCTAssert(zip(segs, s.shots).allSatisfy { $0.v0 == $1.v0 })
        XCTAssertEqual(Set(s.truthPlayers(every: 5).map(\.side)), [.near, .far])
        let r = s.truthRallies()[0]
        XCTAssertEqual(r.endReason.rawValue, s.rallies[0].endReason); XCTAssertEqual(r.shots.count, s.shots.count)
        let (_, m) = s.imageTruth(occluded: true)
        XCTAssert((0..<s.n).allSatisfy { m[$0] || !(s.visible[$0] || s.occluded[$0]) })
        XCTAssertGreaterThan(m.filter { $0 }.count, s.visible.filter { $0 }.count)
    }

    func testSimulateMatch() throws {
        let m = try Sim.simulateMatch(seed: 7, nRallies: 3, fps: 30, source: src)
        XCTAssertEqual(m.rallies.count, 3)
        XCTAssertEqual(m.rallies.map(\.nShots).reduce(0, +), m.shots.count)
        for (a, b) in zip(m.rallies, m.rallies.dropFirst()) {
            XCTAssertGreaterThanOrEqual(m.timestamps[b.startFrame] - m.timestamps[a.endFrame], 3.0)
            // dead time is out of play — except a net fault's fall off the net (≤ 1.5 s, as in
            // test_hits_alternate_and_rally_truth; Python's seed 7 happens to have none)
            let grace = a.endReason == "net" ? Int(1.5 * m.fps) : 0
            XCTAssertFalse((a.endFrame + 3 + grace..<(b.startFrame - 1)).contains { m.inPlay[$0] }, "\(a.endReason)")
            XCTAssertEqual(m.shots[b.firstShot].hitter, a.winner)          // winner serves
        }
        XCTAssertEqual(m.meta.nFrames, m.n)
    }

    /// test_metrics.test_perfect_sim_scores on a native Swift scenario.
    func testPerfectSimScores() throws {
        let s = try Sim.simulateMatch(seed: 3, nRallies: 2, fps: 30, source: src)
        XCTAssertEqual(try Metrics.trackMetrics(s.truthTrack(), s.shuttlePx, vis: s.visible)["f1"]!.double, 1)
        XCTAssertEqual(Metrics.hitMetrics(s.truthHits(), s.hits.map { EvalHit(frame: $0.frame, hitter: $0.hitter) })["f1"]!.double, 1)
        let tshots = s.benchTruth(stage: "e2e").shots
        XCTAssertEqual(Metrics.segmentMetrics(s.truthSegments(), tshots)["speed_err_kmh_median"]!.double, 0)
        let r = Metrics.rallyMetrics(s.truthRallies(), s.benchTruth(stage: "e2e").rallies)
        for k in ["f1", "end_reason_acc", "winner_acc"] { XCTAssertEqual(r[k]!.double, 1, k) }
        XCTAssertEqual(Metrics.shotTypeMetrics(s.hits.map(\.shotType), s.shots.map(\.shotType))["accuracy"]!.double, 1)
    }

    /// Native benchmark samples score like the Python ones (detections row; other stages skipped).
    func testNativeBenchSamples() async throws {
        var o = BenchOptions.defaults(backend: backendRoot)
        o.args["stage"] = "e2e"; o.args["n"] = 3; o.args["fps"] = 30.0
        let samples = try SimBench.samples(options: o)
        XCTAssertEqual(samples.count, 3)
        final class NoStages: BenchStages {
            var available: Set<String> { [] }
            var missing: PyDict { ["linker": "not linked"] }
            func detect(_ c: StageContext) async throws -> [[ShuttleCandidate]]? { nil }
            func linker(_ c: StageContext, candidates: [[ShuttleCandidate]]) async throws -> Track2D? { nil }
            func hits(_ c: StageContext, track: Track2D?) async throws -> [BenchHit]? { nil }
            func reconstruct(_ c: StageContext, track: Track2D?) async throws -> [BenchSegment]? { nil }
            func rallies(_ c: StageContext, track: Track2D?) async throws -> [EvalRally]? { nil }
            func shots(_ c: StageContext, rally: Int) async throws -> BenchTypedRally? { nil }
            func callInOut(_ c: StageContext, segment: Int, isServe: Bool) -> InOutCall { .unavailable }
        }
        let rep = try await BenchRunner(options: o, stages: NoStages()).run(.synthetic(samples))
        let det = try XCTUnwrap(rep.results["detections"]?.dict)
        XCTAssert((det["recall"]?.double ?? 0) > 0.6 && (det["precision"]?.double ?? 0) > 0.9, "\(det)")
        XCTAssertEqual(rep.results["linker"]?["skipped"]?.string, "not linked")
        XCTAssertEqual(rep.results["hits"]?["skipped"]?.string, "no track (linker missing/failed)")
    }
}

func simd_length_d(_ v: SIMD3<Double>) -> Double { (v * v).sum().squareRoot() }
func simd_length_d(_ v: SIMD2<Double>) -> Double { (v * v).sum().squareRoot() }
