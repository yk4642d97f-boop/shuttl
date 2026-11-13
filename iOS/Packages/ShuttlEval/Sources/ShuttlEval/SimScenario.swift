import Foundation
import ShuttlCore
import simd

// sim.py — ground-truth containers and `_simulate` (continuous truth → per-frame arrays).
// Frame i ↔ timestamps[i]; shuttleWorld is NaN when the shuttle is not flying;
// visible = in play ∧ inside the image ∧ not behind a player's torso.

/// One stroke: true free-flight parameters and where it would land (sim.SimShot).
public struct SimShot: Sendable {
    public var index: Int
    public var hitter: String              // 'near' | 'far'
    public var player: String              // 'near', 'far', 'near2', 'far2'
    public var shotType: String
    public var frame: Int                  // contact frame (nearest timestamp)
    public var t0: Double
    public var p0: SIMD3<Double>
    public var v0: SIMD3<Double>
    public var dragLength: Double
    public var targetXY: SIMD2<Double>
    public var landingXY: SIMD2<Double>    // free-flight floor contact (NaN if none)
    public var landingT: Double
    public var netCrossingHeight: Double   // z at x = 0 (NaN if not crossed)
    public var inOut: String               // 'in' | 'out' | 'net' — for the free flight
    public var margin: Double              // in_out_margin (serve: serve_margin), NaN for 'net'
    public var endT: Double
    public var endFrame: Int
    public var outcome: String             // 'returned' | 'in' | 'out' | 'net'
    public var speed0Kmh: Double { NP.norm(v0) * 3.6 }
}

/// Rally-level ground truth (sim.RallyTruth).
public struct SimRallyTruth: Sendable {
    public var startFrame: Int
    public var endFrame: Int
    public var endReason: String
    public var winner: String
    public var server: String
    public var landingXY: SIMD2<Double>?
    public var margin: Double?
    public var firstShot: Int
    public var nShots: Int
}

/// sim hit dict {frame, t, hitter, player, pos, shot_type, shot}.
public struct SimHit: Sendable {
    public var frame: Int
    public var t: Double
    public var hitter: String
    public var player: String
    public var pos: SIMD3<Double>
    public var shotType: String
    public var shot: Int
}

/// Per-player truth (sim players[key]).
public struct SimPlayerTrack: Sendable {
    public var key: String
    public var side: String
    public var height: Double
    public var racketSide: Double
    public var foot: [SIMD2<Double>]       // (N,) floor xy
    public var bbox: [BBox]                // (N,) px [x1,y1,x2,y2] (NaN when off-camera)
    public var racket: [SIMD3<Double>]     // (N,) racket head world xyz
    public var depth: [Double]             // camera depth of the body centre
}

/// A simulated rally / match (sim.SimRally).
public struct SimScenario: Sendable {
    public var camera: CameraModel
    public var fps: Double
    public var timestamps: [Double]
    public var shuttleWorld: [SIMD3<Double>]
    public var shuttlePx: [SIMD2<Double>]
    public var visible: [Bool]
    public var inPlay: [Bool]
    public var occluded: [Bool]
    public var hits: [SimHit]
    public var shots: [SimShot]
    public var rallies: [SimRallyTruth]
    public var players: [SimPlayerTrack]   // Python dict order
    public var doubles: Bool
    public var dragLength: Double
    public var seed: Int
    public var resting: [(from: Int, to: Int, p: SIMD3<Double>)]

    public var n: Int { timestamps.count }
    public var imageSize: ImageSize { camera.imageSize }
    public var meta: VideoMeta {
        VideoMeta(fps: fps, width: camera.imageSize.width, height: camera.imageSize.height, nFrames: n, timestamps: timestamps)
    }
    public func player(_ key: String) -> SimPlayerTrack? { players.first { $0.key == key } }

    /// `image_truth`: where the shuttle is in the image like real TrackNet labels — visible
    /// in-play frames (+ occluded ones if `occluded`) plus the dead shuttle when `resting`.
    public func imageTruth(occluded occ: Bool = false, resting: Bool = true) -> (px: [SIMD2<Double>], mask: [Bool]) {
        var px = shuttlePx
        var mask = (0..<n).map { visible[$0] || (occ && occluded[$0]) }
        if resting {
            let w = Double(camera.imageSize.width), h = Double(camera.imageSize.height)
            for r in self.resting {
                let uv = camera.project(r.p)
                if uv.isFinite && 0 <= uv.x && uv.x < w && 0 <= uv.y && uv.y < h {
                    for i in r.from...r.to where i < n { px[i] = uv; mask[i] = true }
                }
            }
        }
        for i in 0..<n where !mask[i] { px[i] = SIMD2(.nan, .nan) }
        return (px, mask)
    }
}

public enum Sim {
    /// One rally (serve → floor/net) with a short lead-in and tail (sim.simulate_rally).
    /// camera nil → random_camera(style: random) from stream [seed, 99].
    public static func simulateRally(seed: Int, camera: CameraModel? = nil, fps: Double = 60, nShots: Int? = nil,
                                     doubles: Bool = false, source: any SimRandomSource = NativeRandomSource()) throws -> SimScenario {
        try simulate(seed: seed, camera: camera, fps: fps, nRallies: 1, nShots: nShots, doubles: doubles,
                     deadTime: (3.0, 8.0), source: source)
    }

    /// nRallies rallies separated by dead time; winner serves next (sim.simulate_match).
    public static func simulateMatch(seed: Int, nRallies: Int, camera: CameraModel? = nil, fps: Double = 60,
                                     doubles: Bool = false, nShots: Int? = nil, deadTime: (Double, Double) = (3.0, 8.0),
                                     source: any SimRandomSource = NativeRandomSource()) throws -> SimScenario {
        try simulate(seed: seed, camera: camera, fps: fps, nRallies: nRallies, nShots: nShots, doubles: doubles,
                     deadTime: deadTime, source: source)
    }

    static func simulate(seed: Int, camera cameraIn: CameraModel?, fps: Double, nRallies: Int, nShots: Int?, doubles: Bool,
                         deadTime: (Double, Double), source: any SimRandomSource) throws -> SimScenario {
        let rng = source.stream([seed, 17])
        let camera = try cameraIn ?? SimCameras.randomCamera(source.stream([seed, 99]), style: "random")
        let players = SimEngine.newPlayers(rng, doubles)
        let nearB = SimEngine.bases(-1.0, doubles, rng), farB = SimEngine.bases(1.0, doubles, rng)
        var bases: [String: SIMD2<Double>] = [:]
        for (k, b) in zip(players.sideKeys("near"), nearB) { bases[k] = b }
        for (k, b) in zip(players.sideKeys("far"), farB) { bases[k] = b }
        var movers: [String: SimMover] = [:]
        for k in players.keys { movers[k] = SimMover(bases[k]!) }
        var t = rng.uniform(0.5, 1.2)
        var server = rng.pick(["near", "far"])
        var raw: [SimEngine.RawRally] = []
        var L = Physics.dragLength
        for r in 0..<nRallies {
            L = Physics.dragLength * rng.uniform(0.92, 1.08)
            let ral = try SimEngine.playRally(rng, players, movers, bases, tServe: t, serverSide: server, nShots: nShots,
                                              doubles: doubles, L: L, first: r == 0)
            raw.append(ral)
            server = ral.end.winner
            if r < nRallies - 1 { t = ral.end.tFloor + rng.uniform(deadTime.0, deadTime.1) }
        }
        let tTotal = raw[raw.count - 1].end.tFloor + rng.uniform(0.6, 1.2)
        let n = Int((tTotal * fps).rounded(.up))
        let ts = (0..<n).map { Double($0) / fps }
        func frameOf(_ tt: Double) -> Int { Int(NP.clip((tt * fps).rounded(.toNearestOrEven), 0, Double(n - 1))) }
        var world = [SIMD3<Double>](repeating: SIMD3(repeating: .nan), count: n)
        var shots: [SimShot] = [], hits: [SimHit] = [], rallies: [SimRallyTruth] = []
        var resting: [(from: Int, to: Int, p: SIMD3<Double>)] = []
        var prevRestEnd = 0
        for (r, ral) in raw.enumerated() {
            for s in ral.segments {
                let idx = ts.indices.filter { ts[$0] >= s.a && ts[$0] < s.b }
                if !idx.isEmpty {
                    let P = SimPhysics.simulatePositions(p0: s.p0, v0: s.v0, times: idx.map { ts[$0] - s.a }, L: s.L)
                    for (k, i) in idx.enumerated() { world[i] = P[k] }
                }
            }
            let base = shots.count
            for var sh in ral.shots {
                sh.index = shots.count
                sh.frame = frameOf(sh.t0); sh.endFrame = frameOf(sh.endT)
                shots.append(sh)
            }
            for h in ral.hits {
                hits.append(SimHit(frame: frameOf(h.t), t: h.t, hitter: h.hitter, player: h.player, pos: h.pos,
                                   shotType: h.shotType, shot: h.shot + base))
            }
            let e = ral.end
            let fServe = frameOf(ral.tServe), fEnd = frameOf(e.tEnd), fFloor = frameOf(e.tFloor)
            rallies.append(SimRallyTruth(startFrame: fServe, endFrame: fEnd, endReason: e.reason, winner: e.winner,
                                         server: ral.server, landingXY: e.landing, margin: e.margin, firstShot: base,
                                         nShots: ral.shots.count))
            // dead shuttle: held by the server before the serve, lying on the floor afterwards
            let holdFrom = r > 0 ? max(prevRestEnd, fServe - Int((fps * rng.uniform(1.0, 2.0)).rounded(.towardZero))) : 0
            let holdTo = Num.searchsorted(ts, ral.tServe) - 1
            if holdTo >= holdFrom { resting.append((holdFrom, holdTo, ral.pServe)) }
            let rest = SIMD3(e.rest.x, e.rest.y, 0.03)
            let lieTo = r == nRallies - 1 ? n - 1 : min(n - 1, fFloor + Int((fps * rng.uniform(1.0, 2.5)).rounded(.towardZero)))
            if lieTo > fFloor { resting.append((fFloor + 1, lieTo, rest)) }
            prevRestEnd = lieTo + 1
        }
        let inPlay = world.map { $0.x.isFinite }
        // camera.project(nan_to_num(world)); px[~in_play] = NaN
        let px = (0..<n).map { inPlay[$0] ? camera.project(world[$0]) : SIMD2<Double>(.nan, .nan) }
        var tracks: [SimPlayerTrack] = []
        for k in players.keys {
            let p = players.spec[k]!
            let foot = movers[k]!.positions(ts)
            let (bbox, depth) = playerBoxes(camera, foot, p.height)
            let racket = racketTrack(ts, foot, p, hits.filter { $0.player == k })
            tracks.append(SimPlayerTrack(key: k, side: p.side, height: p.height, racketSide: p.racketSide, foot: foot,
                                         bbox: bbox, racket: racket, depth: depth))
        }
        let w = Double(camera.imageSize.width), h = Double(camera.imageSize.height)
        let inside = (0..<n).map { inPlay[$0] && px[$0].x >= 0 && px[$0].x < w && px[$0].y >= 0 && px[$0].y < h }
        let occ = occlusion(camera, world, px, inside, tracks)
        return SimScenario(camera: camera, fps: fps, timestamps: ts, shuttleWorld: world, shuttlePx: px,
                           visible: (0..<n).map { inside[$0] && !occ[$0] }, inPlay: inPlay, occluded: occ, hits: hits,
                           shots: shots, rallies: rallies, players: tracks, doubles: doubles, dragLength: L, seed: seed,
                           resting: resting)
    }

    /// `_player_boxes`: pixel bbox of a (2·BODY_HALF_WIDTH)² × height box on the feet, and the
    /// camera depth of the body centre (for occlusion).
    static func playerBoxes(_ camera: CameraModel, _ foot: [SIMD2<Double>], _ height: Double) -> ([BBox], [Double]) {
        var offs: [SIMD3<Double>] = []
        for dx in [-1.0, 1.0] { for dy in [-1.0, 1.0] { for dz in [0.0, 1.0] {
            offs.append(SIMD3(dx * SimConst.bodyHalfWidth, dy * SimConst.bodyHalfWidth, dz * height))
        } } }
        var boxes: [BBox] = [], depth: [Double] = []
        for f in foot {
            var x1 = Double.nan, y1 = Double.nan, x2 = Double.nan, y2 = Double.nan
            for o in offs {
                let uv = camera.project(SIMD3(f.x, f.y, 0) + o)
                if !uv.x.isNaN { x1 = x1.isNaN ? uv.x : Swift.min(x1, uv.x); x2 = x2.isNaN ? uv.x : Swift.max(x2, uv.x) }
                if !uv.y.isNaN { y1 = y1.isNaN ? uv.y : Swift.min(y1, uv.y); y2 = y2.isNaN ? uv.y : Swift.max(y2, uv.y) }
            }
            boxes.append(BBox(x1: x1, y1: y1, x2: x2, y2: y2))
            depth.append((camera.R * SIMD3(f.x, f.y, height * 0.55) + camera.t).z)
        }
        return (boxes, depth)
    }

    /// `_racket_track`: racket head carried beside the body, swung onto each contact.
    static func racketTrack(_ ts: [Double], _ foot: [SIMD2<Double>], _ pl: SimPlayerSpec, _ own: [SimHit]) -> [SIMD3<Double>] {
        var out = ts.indices.map { SIMD3(foot[$0].x - pl.sign * 0.25, foot[$0].y + pl.racketSide * 0.45, 0.62 * pl.height) }
        for h in own {
            for i in ts.indices {
                let q = (ts[i] - h.t) / 0.07
                let w = exp(-0.5 * (q * q))
                out[i] = out[i] * (1 - w) + h.pos * w
            }
        }
        return out
    }

    /// `_occlusion`: shuttle behind a player's torso (central 55 % width, head→knee band).
    static func occlusion(_ camera: CameraModel, _ world: [SIMD3<Double>], _ px: [SIMD2<Double>], _ inside: [Bool],
                          _ players: [SimPlayerTrack]) -> [Bool] {
        var occ = [Bool](repeating: false, count: world.count)
        for i in world.indices where inside[i] {
            let zc = (camera.R * world[i] + camera.t).z
            for p in players {
                let b = p.bbox[i]
                let cx = (b.x1 + b.x2) / 2, bw = b.x2 - b.x1, bh = b.y2 - b.y1
                let u = px[i].x, v = px[i].y
                if abs(u - cx) < 0.275 * bw && v > b.y1 + 0.08 * bh && v < b.y1 + 0.75 * bh && zc > p.depth[i] + 0.25 { occ[i] = true }
            }
        }
        return occ
    }
}
