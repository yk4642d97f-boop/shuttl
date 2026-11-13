import Foundation
import XCTest
import ShuttlCore
@testable import ShuttlEval

/// sim.py parity: Python's numpy draws (recorded per stream) replayed into the Swift
/// simulator must give the Python scenario — positions within 1e-9, identical labels.
final class SimReplayTests: XCTestCase {

    // MARK: fixture model

    struct Fixture: Decodable {
        var scenarios: [Scenario]
        var corrupt: [Corrupt]
        var cameras: [Cam]
        var libm: [String: [Double?]]
    }
    struct Args: Decodable {
        var seed: Int; var fps: Double; var nShots: Int?; var doubles: Bool?; var nRallies: Int?
        enum CodingKeys: String, CodingKey { case seed, fps, nShots = "n_shots", doubles, nRallies = "n_rallies" }
    }
    struct Scenario: Decodable {
        var name: String; var kind: String; var args: Args; var camera: CameraModel?
        var streams: [ReplayRandomSource.Stream]; var scenario: Truth
    }
    struct Corrupt: Decodable {
        var name: String; var of: String; var fn: String; var args: [String: JSONValue]
        var streams: [ReplayRandomSource.Stream]; var candidates: CandidateFrames
    }
    struct Cam: Decodable { var style: String; var size: [Int]; var streams: [ReplayRandomSource.Stream]; var camera: CameraModel }
    struct Truth: Decodable {
        var n: Int; var fps: Double; var seed: Int; var doubles: Bool; var dragLength: Double; var camera: CameraModel
        var shuttleWorld: [[Double?]]; var shuttlePx: [[Double?]]; var visible: [Bool]; var inPlay: [Bool]; var occluded: [Bool]
        var hits: [Hit]; var shots: [Shot]; var rallies: [Rally]; var players: [Player]; var resting: [Rest]
        enum CodingKeys: String, CodingKey {
            case n, fps, seed, doubles, dragLength = "drag_length", camera, shuttleWorld = "shuttle_world"
            case shuttlePx = "shuttle_px", visible, inPlay = "in_play", occluded, hits, shots, rallies, players, resting
        }
    }
    struct Hit: Decodable {
        var frame: Int; var t: Double; var hitter: String; var player: String; var pos: [Double]; var shotType: String; var shot: Int
        enum CodingKeys: String, CodingKey { case frame, t, hitter, player, pos, shotType = "shot_type", shot }
    }
    struct Shot: Decodable {
        var index: Int; var hitter: String; var player: String; var shotType: String; var frame: Int; var t0: Double
        var p0: [Double]; var v0: [Double]; var dragLength: Double; var targetXY: [Double]; var landingXY: [Double?]
        var landingT: Double?; var netCrossingHeight: Double?; var inOut: String; var margin: Double?; var endT: Double
        var endFrame: Int; var outcome: String
        enum CodingKeys: String, CodingKey {
            case index, hitter, player, shotType = "shot_type", frame, t0, p0, v0, dragLength = "drag_length"
            case targetXY = "target_xy", landingXY = "landing_xy", landingT = "landing_t"
            case netCrossingHeight = "net_crossing_height", inOut = "in_out", margin, endT = "end_t", endFrame = "end_frame", outcome
        }
    }
    struct Rally: Decodable {
        var startFrame: Int; var endFrame: Int; var endReason: String; var winner: String; var server: String
        var landingXY: [Double]?; var margin: Double?; var firstShot: Int; var nShots: Int
        enum CodingKeys: String, CodingKey {
            case startFrame = "start_frame", endFrame = "end_frame", endReason = "end_reason", winner, server
            case landingXY = "landing_xy", margin, firstShot = "first_shot", nShots = "n_shots"
        }
    }
    struct Player: Decodable {
        var key: String; var side: String; var height: Double; var racketSide: Double
        var foot: [[Double]]; var bbox: [[Double?]]; var racket: [[Double]]; var depth: [Double]
        enum CodingKeys: String, CodingKey { case key, side, height, racketSide = "racket_side", foot, bbox, racket, depth }
    }
    struct Rest: Decodable { var from: Int; var to: Int; var p: [Double] }

    static let fixture: Result<Fixture, Error> = Result {
        try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fixtureURL("sim_replay")))
    }

    func load() throws -> Fixture { try Self.fixture.get() }

    // MARK: comparison helpers

    final class Dev { var max: [String: Double] = [:] }

    func near(_ a: Double, _ b: Double?, _ tol: Double, _ what: String, _ dev: Dev, file: StaticString = #filePath, line: UInt = #line) {
        let b = b ?? .nan
        if a.isNaN || b.isNaN || a.isInfinite || b.isInfinite {
            XCTAssert(a.isNaN == b.isNaN && (a.isNaN || a == b), "\(what): \(a) vs \(b)", file: file, line: line); return
        }
        let d = abs(a - b)
        dev.max[what] = Swift.max(dev.max[what] ?? 0, d)
        XCTAssert(d <= tol, "\(what): \(a) vs \(b) (Δ \(d) > \(tol))", file: file, line: line)
    }

    func near(_ a: [Double], _ b: [Double?], _ tol: Double, _ what: String, _ dev: Dev, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.count, b.count, "\(what) length", file: file, line: line)
        var firstBad = true
        for (x, y) in zip(a, b) {
            let y = y ?? .nan
            let ok = (x.isNaN && y.isNaN) || abs(x - y) <= tol
            if !x.isNaN && !y.isNaN { dev.max[what] = Swift.max(dev.max[what] ?? 0, abs(x - y)) }
            if !ok && firstBad { XCTFail("\(what): \(x) vs \(y)", file: file, line: line); firstBad = false }
        }
    }

    /// Replay one scenario.
    func replay(_ s: Scenario) throws -> (SimScenario, [String]) {
        let src = ReplayRandomSource(s.streams)
        let a = s.args
        let sim = s.kind == "rally"
            ? try Sim.simulateRally(seed: a.seed, camera: s.camera, fps: a.fps, nShots: a.nShots, doubles: a.doubles ?? false, source: src)
            : try Sim.simulateMatch(seed: a.seed, nRallies: a.nRallies ?? 1, camera: s.camera, fps: a.fps, doubles: a.doubles ?? false,
                                    nShots: a.nShots, source: src)
        return (sim, src.problems)
    }

    // MARK: tests

    func testScenariosReplayExactly() throws {
        let fx = try load()
        let dev = Dev()
        for s in fx.scenarios {
            let t0 = Date()
            let (sim, problems) = try replay(s)
            XCTAssert(problems.isEmpty, "\(s.name) draw log:\n" + problems.prefix(8).joined(separator: "\n"))
            let T = s.scenario
            XCTAssertEqual(sim.n, T.n, s.name); XCTAssertEqual(sim.doubles, T.doubles); XCTAssertEqual(sim.seed, T.seed)
            guard sim.n == T.n else { continue }
            near(sim.dragLength, T.dragLength, 1e-12, "drag_length", dev)
            XCTAssertEqual(sim.visible, T.visible, "\(s.name) visible"); XCTAssertEqual(sim.inPlay, T.inPlay, "\(s.name) in_play")
            XCTAssertEqual(sim.occluded, T.occluded, "\(s.name) occluded")
            near(sim.shuttleWorld.flatMap(\.row), T.shuttleWorld.flatMap { $0 }, 1e-9, "shuttle_world m", dev)
            near(sim.shuttlePx.flatMap(\.row), T.shuttlePx.flatMap { $0 }, 1e-6, "shuttle_px", dev)
            XCTAssertEqual(sim.hits.map(\.frame), T.hits.map(\.frame), "\(s.name) hit frames")
            XCTAssertEqual(sim.hits.map { "\($0.hitter)/\($0.player)/\($0.shotType)/\($0.shot)" },
                           T.hits.map { "\($0.hitter)/\($0.player)/\($0.shotType)/\($0.shot)" }, "\(s.name) hit labels")
            for (h, e) in zip(sim.hits, T.hits) { near(h.t, e.t, 1e-9, "hit t", dev); near(h.pos.row, e.pos, 1e-9, "hit pos", dev) }
            XCTAssertEqual(sim.shots.count, T.shots.count, "\(s.name) shots")
            for (a, e) in zip(sim.shots, T.shots) {
                XCTAssertEqual("\(a.index) \(a.hitter) \(a.player) \(a.shotType) \(a.frame) \(a.endFrame) \(a.inOut) \(a.outcome)",
                               "\(e.index) \(e.hitter) \(e.player) \(e.shotType) \(e.frame) \(e.endFrame) \(e.inOut) \(e.outcome)", s.name)
                near(a.p0.row, e.p0, 1e-9, "shot p0", dev); near(a.v0.row, e.v0, 1e-7, "shot v0 m/s", dev)
                near(a.t0, e.t0, 1e-9, "shot t0", dev); near(a.endT, e.endT, 1e-9, "shot end_t", dev)
                near(a.targetXY.row, e.targetXY, 1e-9, "shot target", dev); near(a.landingXY.row, e.landingXY, 1e-9, "shot landing", dev)
                near(a.landingT, e.landingT, 1e-9, "shot landing_t", dev); near(a.netCrossingHeight, e.netCrossingHeight, 1e-9, "shot z_net", dev)
                near(a.margin, e.margin, 1e-9, "shot margin", dev); near(a.dragLength, e.dragLength, 1e-12, "shot L", dev)
            }
            XCTAssertEqual(sim.rallies.count, T.rallies.count)
            for (a, e) in zip(sim.rallies, T.rallies) {
                XCTAssertEqual("\(a.startFrame) \(a.endFrame) \(a.endReason) \(a.winner) \(a.server) \(a.firstShot) \(a.nShots)",
                               "\(e.startFrame) \(e.endFrame) \(e.endReason) \(e.winner) \(e.server) \(e.firstShot) \(e.nShots)", s.name)
                XCTAssertEqual(a.landingXY == nil, e.landingXY == nil)
                if let l = a.landingXY, let el = e.landingXY { near(l.row, el, 1e-9, "rally landing", dev) }
                near(a.margin ?? .nan, e.margin, 1e-9, "rally margin", dev)
            }
            XCTAssertEqual(sim.players.map(\.key), T.players.map(\.key))
            for (p, e) in zip(sim.players, T.players) {
                XCTAssertEqual(p.side, e.side); near(p.height, e.height, 1e-12, "height", dev); XCTAssertEqual(p.racketSide, e.racketSide)
                near(p.foot.flatMap(\.row), e.foot.flatMap { $0 }, 1e-9, "foot m", dev)
                near(p.bbox.flatMap { [$0.x1, $0.y1, $0.x2, $0.y2] }, e.bbox.flatMap { $0 }, 1e-6, "bbox px", dev)
                near(p.racket.flatMap(\.row), e.racket.flatMap { $0 }, 1e-9, "racket m", dev)
                near(p.depth, e.depth, 1e-9, "depth m", dev)
            }
            XCTAssertEqual(sim.resting.map { "\($0.from)-\($0.to)" }, T.resting.map { "\($0.from)-\($0.to)" }, s.name)
            for (a, e) in zip(sim.resting, T.resting) { near(a.p.row, e.p, 1e-9, "resting", dev) }
            print("sim replay \(s.name): \(sim.n) frames, \(sim.shots.count) shots, \(String(format: "%.2f", Date().timeIntervalSince(t0))) s")
        }
        print("max |Δ| per field: " + dev.max.sorted { $0.key < $1.key }.map { "\($0.key)=\(String(format: "%.1e", $0.value))" }.joined(separator: " "))
    }

    func testCorruptionReplaysExactly() throws {
        let fx = try load()
        let dev = Dev()
        for c in fx.corrupt {
            let base = try XCTUnwrap(fx.scenarios.first { $0.name == c.of })
            let sim = try replay(base).0
            let src = ReplayRandomSource(c.streams)
            let a = c.args.mapValues { v -> Double in v.double ?? (v.bool.map { $0 ? 1 : 0 } ?? .nan) }
            let got: [[ShuttleCandidate]]
            if c.fn == "detections" {
                got = SimNoise.corruptDetections(sim, rng: src.stream([sim.seed, 11]), sigmaPx: a["sigma_px"] ?? 1.5,
                                                 missRate: a["miss_rate"] ?? 0.1, falsePositiveRate: a["false_positive_rate"] ?? 0.03,
                                                 deadShuttleRate: a["dead_shuttle_rate"] ?? 0.5)
            } else {
                got = SimNoise.corruptPx(sim.shuttlePx, visible: sim.visible, imageSize: sim.imageSize, rng: src.stream([sim.seed, 12]),
                                         missRate: a["miss_rate"] ?? 0.1, burstMiss: (a["burst_miss"] ?? 1) != 0)
            }
            XCTAssert(src.problems.isEmpty, "\(c.name):\n" + src.problems.prefix(8).joined(separator: "\n"))
            let want = c.candidates.frames
            XCTAssertEqual(got.map(\.count), want.map(\.count), "\(c.name) candidates per frame")
            for (g, w) in zip(got, want) {
                near(g.flatMap { [$0.x, $0.y] }, w.flatMap { [$0.x, $0.y] }, 1e-6, "cand px", dev)
                near(g.map(\.score), w.map(\.score), 1e-12, "cand score", dev)
            }
            if c.name == "clean_e" { XCTAssert(zip(got, sim.visible).allSatisfy { ($0.count == 1) == $1 }, "clean: one candidate per visible frame") }
        }
        print("corrupt max |Δ|: " + dev.max.map { "\($0.key)=\(String(format: "%.1e", $0.value))" }.joined(separator: " "))
    }

    func testRandomCamerasReplayExactly() throws {
        let fx = try load()
        for c in fx.cameras {
            let src = ReplayRandomSource(c.streams)
            let key = c.streams.first?.key ?? "?"
            let cam = try SimCameras.randomCamera(src.stream(key), imageSize: ImageSize(width: c.size[0], height: c.size[1]), style: c.style)
            XCTAssert(src.problems.isEmpty, "\(c.style) \(key): \(src.problems.prefix(4))")
            for (a, b) in zip(cam.K.rowMajor.flatMap { $0 } + cam.R.rowMajor.flatMap { $0 } + cam.t.row,
                              c.camera.K.rowMajor.flatMap { $0 } + c.camera.R.rowMajor.flatMap { $0 } + c.camera.t.row) {
                XCTAssertEqual(a, b, accuracy: 1e-9 * max(1, abs(b)), "\(c.style) \(key)")
            }
        }
    }

    /// Transcendentals numpy uses in sim.py vs Darwin libm (bit-exact → no drift source).
    func testLibmMatchesNumpy() throws {
        let L = try load().libm
        let x = L["x"]!.map { $0! }, xp = L["xp"]!.map { $0! }
        assertBits(x.map(cos), L["cos"]!.map { $0! }, "cos")
        assertBits(x.map(sin), L["sin"]!.map { $0! }, "sin")
        assertBits(x.map { exp(-abs($0)) }, L["exp"]!.map { $0! }, "exp")
        assertBits(xp.map(log10), L["log10"]!.map { $0! }, "log10")
        // np.power(10.0, x) = libm pow. NP.pow, not a bare pow(10.0, …): -O builds rewrite the
        // literal-base call into Darwin __exp10, 1 ulp off pow on some inputs (see NP.pow).
        assertBits(x.prefix(200).map { NP.pow(10.0, $0 / 20) }, L["pow10"]!.map { $0! }, "power")
        assertBits(xp.map { pow($0, 3) }, L["cube"]!.map { $0! }, "x**3")
        assertBits(xp.map { pow($0, 0.5) }, L["sqrt_pow"]!.map { $0! }, "x**0.5")
        assertBits(xp.map { $0.squareRoot() }, L["sqrt"]!.map { $0! }, "sqrt")
    }
}
