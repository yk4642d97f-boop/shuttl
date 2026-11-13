import Foundation
import ShuttlCore

// sim.py — rally engine: players, min-jerk footwork, free flights, receiver contact
// search, serve formation / targets and the rally loop (`_play_rally`).

public struct SimPlayerSpec: Sendable, Equatable {
    public var side: String            // 'near' | 'far'
    public var sign: Double            // -1 near, +1 far
    public var height: Double
    public var racketSide: Double      // world-y direction of the racket hand
}

/// Player keys in Python dict order ('near', 'far' [, 'near2', 'far2']).
struct SimPlayers {
    var keys: [String] = []
    var spec: [String: SimPlayerSpec] = [:]
    func sideKeys(_ side: String) -> [String] { keys.filter { spec[$0]!.side == side } }
}

@inline(__always) func minjerk(_ s: Double) -> Double { pow(s, 3) * (10 - 15 * s + 6 * s * s) }

/// `_Mover`: piecewise min-jerk footwork; a new move starts from wherever the player is
/// and interrupts later moves.
final class SimMover {
    var p0: SIMD2<Double>
    var moves: [(t0: Double, t1: Double, a: SIMD2<Double>, b: SIMD2<Double>)] = []
    init(_ xy: SIMD2<Double>) { p0 = xy }

    func at(_ t: Double) -> SIMD2<Double> {
        for m in moves.reversed() where t >= m.t0 {
            let s = t >= m.t1 ? 1.0 : minjerk((t - m.t0) / (m.t1 - m.t0))
            return m.a + (m.b - m.a) * s
        }
        return p0
    }

    func move(_ t0: Double, _ t1: Double, _ goal: SIMD2<Double>) {
        moves = moves.filter { $0.t0 <= t0 }
        let a = at(t0)
        moves.append((t0, pyMax(t1, t0 + 0.05), a, goal))
    }

    func recover(_ t0: Double, _ goal: SIMD2<Double>, speed: Double = 2.2) {
        let g = goal - at(t0)
        let d = NP.norm(g.x, g.y)
        move(t0, t0 + 0.3 + d / speed, goal)
    }

    func positions(_ ts: [Double]) -> [SIMD2<Double>] {
        var out = [SIMD2<Double>](repeating: p0, count: ts.count)
        for (k, m) in moves.enumerated() {
            let nxt = k + 1 < moves.count ? moves[k + 1].t0 : .infinity
            for i in ts.indices where ts[i] >= m.t0 && ts[i] < nxt {
                let s = minjerk(NP.clip((ts[i] - m.t0) / (m.t1 - m.t0), 0.0, 1.0))
                out[i] = m.a + (m.b - m.a) * s
            }
        }
        return out
    }
}

/// `_Flight`: free flight from (p0, v0) at absolute t0; 1/120 s samples before landing.
struct SimFlightPath {
    var t0: Double
    var p0: SIMD3<Double>
    var v0: SIMD3<Double>
    var L: Double
    var ts: [Double]
    var P: [SIMD3<Double>]
    var tLand: Double                  // relative; inf if it never lands within t_max
    var pLand: SIMD3<Double>
    var tNet: Double? = nil
    var pNet: SIMD3<Double>? = nil
}

enum SimEngine {
    @inline(__always) static func npSign(_ x: Double) -> Double { x > 0 ? 1 : (x < 0 ? -1 : (x.isNaN ? .nan : 0)) }

    /// `_refine_crossing`: linear refinement of the first f ≤ 0 on 33 samples in [ta, tb].
    static func refineCrossing(_ p0: SIMD3<Double>, _ v0: SIMD3<Double>, _ L: Double, _ ta: Double, _ tb: Double,
                               _ f: (SIMD3<Double>) -> Double) -> Double {
        let tf = NP.linspace(ta, tb, 33)
        let val = SimPhysics.simulatePositions(p0: p0, v0: v0, times: tf, L: L).map(f)
        let j = val.firstIndex { $0 <= 0 } ?? (tf.count - 1)
        if j == 0 { return tf[0] }
        return tf[j - 1] + val[j - 1] / (val[j - 1] - val[j]) * (tf[j] - tf[j - 1])
    }

    /// `_free_flight`.
    static func freeFlight(_ t0: Double, _ p0: SIMD3<Double>, _ v0: SIMD3<Double>, _ L: Double, tMax: Double = 6.0) -> SimFlightPath {
        var ts = NP.arange(0.0, tMax, 1.0 / 120)
        var P = SimPhysics.simulatePositions(p0: p0, v0: v0, times: ts, L: L)
        var tLand = Double.infinity, pLand = P[P.count - 1]
        if let i = P.firstIndex(where: { $0.z <= 0.0 }), i > 0 {
            tLand = refineCrossing(p0, v0, L, ts[i - 1], ts[i]) { $0.z }
            pLand = SimPhysics.simulatePositions(p0: p0, v0: v0, times: [tLand], L: L)[0]
            let keep = ts.indices.filter { ts[$0] < tLand }
            ts = keep.map { ts[$0] }; P = keep.map { P[$0] }
        }
        var fl = SimFlightPath(t0: t0, p0: p0, v0: v0, L: L, ts: ts, P: P, tLand: tLand, pLand: pLand)
        let side0 = npSign(p0.x - SimConst.netX)
        if side0 != 0, let i = P.firstIndex(where: { npSign($0.x - SimConst.netX) != side0 }) {
            let tn = refineCrossing(p0, v0, L, ts[(i - 1 + ts.count) % ts.count], ts[i]) { side0 * ($0.x - SimConst.netX) }
            fl.tNet = tn
            fl.pNet = SimPhysics.simulatePositions(p0: p0, v0: v0, times: [tn], L: L)[0]
        }
        return fl
    }

    /// `_reach`: horizontal foot→contact distance covered with arm + racket (+ lunge).
    @inline(__always) static func reach(_ z: Double) -> Double { z >= 1.9 ? 0.35 : (z >= 0.9 ? 0.8 : 1.15) }

    /// `_find_contact`: the receiver's contact sample (player key, sample index, foot xy).
    static func findContact(_ rng: any SimRandom, _ fl: SimFlightPath, keys: [String], movers: [String: SimMover],
                            players: SimPlayers, recvSign: Double, incoming: String, hw: Double) -> (String, Int, SIMD2<Double>)? {
        let ts = fl.ts, P = fl.P, n = ts.count
        let baseOK = (0..<n).map { i -> Bool in
            let depth = recvSign * (P[i].x - SimConst.netX)
            return depth > 0.3 && depth < CourtModel.halfLength + 1.3 && abs(P[i].y) < hw + 1.0
                && P[i].z > 0.12 && ts[i] > SimConst.reactionS + 0.12
        }
        var (mu, sd) = SimShots.prefZ[incoming] ?? (1.5, 0.6)
        if ["net", "push", "block", "drop", "serve_short"].contains(incoming) && rng.random() < 0.45 { (mu, sd) = (1.9, 0.3) }
        var best: [(String, [Double], SIMD2<Double>)] = []
        for key in keys {
            let pl = players.spec[key]!, mv = movers[key]!
            let zmax = 1.33 * pl.height + 0.55
            var ok = (0..<n).map { baseOK[$0] && P[$0].z < zmax }
            if !ok.contains(true) { continue }
            let start = mv.at(fl.t0 + SimConst.reactionS)
            let dist = (0..<n).map { i -> Double in
                let d = NP.rowNorm(P[i].x - start.x, P[i].y - start.y) - reach(P[i].z)
                return d.isNaN ? d : Swift.max(d, 0.0)
            }
            for i in 0..<n { ok[i] = ok[i] && dist[i] <= SimConst.maxAvgSpeed * (ts[i] - SimConst.reactionS) + 0.05 }
            if !ok.contains(true) { continue }
            let w = (0..<n).map { i -> Double in
                guard ok[i] else { return 0 }
                let q = (P[i].z - mu) / sd
                return (exp(-0.5 * (q * q)) + 0.03) * exp(-dist[i] / 2.0)
            }
            best.append((key, w, start))
        }
        if best.isEmpty { return nil }
        let W = best.flatMap(\.1)
        let tot = NP.sum(W)
        let pick = rng.choice(W.count, p: W.map { $0 / tot })
        let (key, _, start) = best[pick / n]
        let j = pick % n
        let c = P[j]
        var foot: SIMD2<Double>
        if c.z >= 1.9 {                                          // overhead: feet behind and beside the contact
            let dx = recvSign * rng.uniform(0.1, 0.35)
            foot = SIMD2(c.x + dx, c.y + (-players.spec[key]!.racketSide * 0.2))
        } else {
            let d = c.xy - start
            let nn = NP.norm(d.x, d.y)
            let u = nn > 1e-6 ? d / nn : SIMD2(-recvSign, 0.0)
            foot = c.xy - u * pyMin(reach(c.z) * rng.uniform(0.7, 1.0), pyMax(nn, 0.3))
        }
        foot.x = recvSign * pyMax(recvSign * foot.x, 0.35)
        return (key, j, foot)
    }

    static func bases(_ sideSign: Double, _ doubles: Bool, _ rng: any SimRandom) -> [SIMD2<Double>] {
        if !doubles {
            let x = sideSign * rng.uniform(3.4, 4.1)
            return [SIMD2(x, rng.uniform(-0.3, 0.3))]
        }
        let x = sideSign * rng.uniform(3.4, 4.2)
        return [SIMD2(x, -1.3), SIMD2(x, 1.3)]
    }

    static func newPlayers(_ rng: any SimRandom, _ doubles: Bool) -> SimPlayers {
        var keys: [(String, Double)] = [("near", -1.0), ("far", 1.0)]
        if doubles { keys += [("near2", -1.0), ("far2", 1.0)] }
        var out = SimPlayers()
        for (k, s) in keys {
            let rightHanded = rng.random() < 0.87
            let height = rng.uniform(SimConst.playerHeight.0, SimConst.playerHeight.1)
            out.keys.append(k)
            out.spec[k] = SimPlayerSpec(side: s < 0 ? "near" : "far", sign: s, height: height,
                                        racketSide: s * (rightHanded ? 1.0 : -1.0))
        }
        return out
    }

    /// `_serve_positions`: (server key, foot xy per player in dict order, y-sign of the server's court).
    static func servePositions(_ rng: any SimRandom, _ players: SimPlayers, _ serverSide: String, _ doubles: Bool)
        -> (String, [(String, SIMD2<Double>)], Double) {
        let s: Double = serverSide == "near" ? -1.0 : 1.0
        let court = rng.pick([-1.0, 1.0])
        let srv = players.sideKeys(serverSide), rcv = players.sideKeys(serverSide == "near" ? "far" : "near")
        let server = rng.pick(srv)
        let receiver = rng.pick(rcv)
        var pos: [(String, SIMD2<Double>)] = []
        if doubles {
            pos.append((server, SIMD2(s * rng.uniform(2.1, 2.6), court * rng.uniform(0.2, 0.6))))
            pos.append((receiver, SIMD2(-s * rng.uniform(2.3, 2.8), -court * rng.uniform(0.3, 0.8))))
        } else {
            pos.append((server, SIMD2(s * rng.uniform(2.4, 3.4), court * rng.uniform(0.25, 0.9))))
            pos.append((receiver, SIMD2(-s * rng.uniform(2.9, 3.8), -court * rng.uniform(0.3, 1.0))))
        }
        for k in srv where k != server { pos.append((k, SIMD2(s * rng.uniform(3.8, 4.4), -court * rng.uniform(0.2, 0.8)))) }
        for k in rcv where k != receiver { pos.append((k, SIMD2(-s * rng.uniform(3.8, 4.4), court * rng.uniform(0.3, 0.9)))) }
        return (server, pos, court)
    }

    /// `serve_margin`: signed distance of a serve landing from the receiver's service-court edge.
    public static func serveMargin(_ x: Double, _ y: Double, recvSign: Double, court: Double, doubles: Bool) -> Double {
        let lw = CourtModel.lineWidth / 2
        let longX = doubles ? CourtModel.doublesLongServiceX : CourtModel.halfLength
        let d = recvSign * x, yy = -court * y
        return pyMin(d - (CourtModel.shortServiceX - lw), longX + lw - d, yy + lw, SimShots.halfWidth(doubles) + lw - yy)
    }

    /// `_serve_target`: target in the diagonally opposite service court.
    static func serveTarget(_ rng: any SimRandom, _ kind: String, recvSign: Double, court: Double, doubles: Bool,
                            hw: Double, outcome: String) -> SIMD2<Double> {
        let lw = CourtModel.lineWidth / 2
        let longX = doubles ? CourtModel.doublesLongServiceX : CourtModel.halfLength
        let xLo = CourtModel.shortServiceX - lw, xHi = longX + lw, yLo = -lw, yHi = hw + lw
        let short = kind == "serve_short"
        if outcome == "in" || outcome == "out" {
            let m = rng.random() < 0.5 ? rng.uniform(0.005, 0.2) : rng.uniform(0.2, 0.7)
            let sg: Double = outcome == "in" ? 1.0 : -1.0
            let edges = short ? ["short", "side", "centre"] : ["long", "side", "centre"]
            let edge = rng.pick(edges, p: [0.6, 0.25, 0.15])
            var depth = short ? rng.uniform(2.1, 2.8) : rng.uniform(longX - 0.8, longX - 0.15)
            var yy = rng.uniform(0.35, hw - 0.35)
            switch edge {
            case "short": depth = xLo + sg * m
            case "long": depth = xHi - sg * m
            case "side": yy = yHi - sg * m
            default: yy = yLo + sg * m
            }
            return SIMD2(recvSign * depth, -court * yy)
        }
        var depth: Double, y: Double
        if short { depth = rng.uniform(2.05, 2.8); y = rng.uniform(0.1, 1.4) }
        else { depth = doubles ? rng.uniform(5.2, 5.9) : rng.uniform(5.6, 6.6); y = rng.uniform(0.2, hw - 0.15) }
        return SIMD2(recvSign * depth, -court * y)
    }

    struct RawSegment { var a: Double; var b: Double; var p0: SIMD3<Double>; var v0: SIMD3<Double>; var L: Double }
    struct RawHit { var t: Double; var hitter: String; var player: String; var pos: SIMD3<Double>; var shotType: String; var shot: Int }
    struct RawEnd {
        var reason: String; var tEnd: Double; var tFloor: Double; var rest: SIMD3<Double>
        var landing: SIMD2<Double>?; var margin: Double?; var winner: String
    }
    struct RawRally {
        var shots: [SimShot]; var segments: [RawSegment]; var hits: [RawHit]; var end: RawEnd
        var server: String; var tServe: Double; var pServe: SIMD3<Double>
    }

    /// `_play_rally`: one rally from the serve at tServe.
    static func playRally(_ rng: any SimRandom, _ players: SimPlayers, _ movers: [String: SimMover], _ bases: [String: SIMD2<Double>],
                          tServe: Double, serverSide: String, nShots: Int?, doubles: Bool, L: Double, first: Bool) throws -> RawRally {
        let hw = SimShots.halfWidth(doubles)
        let (server, pos, court) = servePositions(rng, players, serverSide, doubles)
        for (k, xy) in pos {                                   // walk into the serve formation
            if first { movers[k]!.p0 = xy } else { movers[k]!.move(tServe - 2.2, tServe - 0.6, xy) }
        }
        let targetN = nShots ?? Int(NP.clip(Double(rng.geometric(1 / 8.5)), 1, 40))
        let pl = players.spec[server]!
        var p0 = SIMD3(pos[0].1.x - pl.sign * 0.35, pos[0].1.y + pl.racketSide * 0.3, rng.uniform(0.88, 1.12))
        var kind = rng.random() < (doubles ? 0.88 : 0.45) ? "serve_short" : "serve_long"
        var hitter = server, t0 = tServe
        var shots: [SimShot] = [], segments: [RawSegment] = [], hits: [RawHit] = []
        var end: RawEnd? = nil
        for i in 0..<10_000 {
            let hs = players.spec[hitter]!.sign
            let recvSign = -hs, recvSide = hs < 0 ? "far" : "near"
            let final = i >= targetN - 1
            var outcome = "returned"
            if final {
                let baseKind = kind.hasPrefix("serve") ? "serve" : kind
                let w = baseKind == "serve" ? [0.45, 0.25, 0.30] : (SimShots.endW[baseKind] ?? [0.5, 0.25, 0.25])
                let ws = w.reduce(0, +)
                outcome = rng.pick(["in", "out", "net"], p: w.map { $0 / ws })
            }
            var kinds = [kind] + fallbackKindsExcluding(kind, z: p0.z, depth: abs(p0.x))
            if kind.hasPrefix("serve") { kinds = [kind, kind == "serve_short" ? "serve_long" : "serve_short"] }
            let tries = nShots != nil ? 10 : 3
            var chosen: (String, SIMD3<Double>, SIMD2<Double>, SimFlightPath, (String, Int, SIMD2<Double>)?, String)? = nil
            for kk in kinds {
                for attempt in 0..<tries {
                    let tgt: SIMD2<Double>? = kk.hasPrefix("serve")
                        ? serveTarget(rng, kk, recvSign: recvSign, court: court, doubles: doubles, hw: hw, outcome: outcome) : nil
                    guard let planned = SimShots.planShot(rng, kind: kk, p0: p0, recvSign: recvSign, L: L, hw: hw,
                                                          outcome: outcome, target: tgt, tries: 3) else { continue }
                    let fl = freeFlight(t0, p0, planned.v0, L)
                    if fl.ts.count <= 3 { continue }                    // _flight_px_ok
                    if outcome != "returned" { chosen = (kk, planned.v0, planned.target, fl, nil, outcome); break }
                    if let c = findContact(rng, fl, keys: players.sideKeys(recvSide), movers: movers, players: players,
                                           recvSign: recvSign, incoming: kk, hw: hw) {
                        chosen = (kk, planned.v0, planned.target, fl, c, outcome); break
                    }
                    if attempt == tries - 1 { chosen = (kk, planned.v0, planned.target, fl, nil, "winner") }
                }
                if chosen != nil { break }
            }
            guard let ch = chosen else { throw SimError.planningFailed }
            let v0 = ch.1, tgt = ch.2, fl = ch.3, contact = ch.4
            kind = ch.0
            // free-flight truth of this stroke
            let landXY = fl.tLand.isFinite ? SIMD2(fl.pLand.x, fl.pLand.y) : SIMD2<Double>(.nan, .nan)
            let zNet = fl.pNet?.z ?? .nan
            let crosses = fl.tNet != nil && fl.tNet! < fl.tLand
            var ffIO: String, ffMargin: Double
            if crosses && zNet < CourtModel.netHeightCentre {
                ffIO = "net"; ffMargin = .nan
            } else {
                ffMargin = i == 0 ? serveMargin(landXY.x, landXY.y, recvSign: recvSign, court: court, doubles: doubles)
                                  : CourtModel.inOutMargin(x: landXY.x, y: landXY.y, doubles: doubles)
                ffIO = ffMargin >= 0 ? "in" : "out"
            }
            let hitterSide = players.spec[hitter]!.side
            let shotType = kind.hasPrefix("serve") ? "serve" : kind
            hits.append(RawHit(t: t0, hitter: hitterSide, player: hitter, pos: p0, shotType: shotType, shot: shots.count))
            var home = bases[hitter]!
            if ["net", "push", "block", "drop", "kill", "smash"].contains(kind) && rng.random() < 0.5 {
                home = SIMD2(hs * rng.uniform(2.0, 2.7), home.y * 0.5)          // follow in to the net
            }
            movers[hitter]!.recover(t0 + 0.12, home)
            var segEnd: Double, shotOutcome: String
            if let (key, j, foot) = contact {                                  // returned by the receiver
                let tc = t0 + fl.ts[j]
                movers[key]!.move(t0 + SimConst.reactionS, tc - 0.03, foot)
                for k in players.sideKeys(recvSide) where k != key { movers[k]!.recover(t0 + SimConst.reactionS + 0.1, bases[k]!) }
                segEnd = tc; shotOutcome = "returned"
                segments.append(RawSegment(a: t0, b: tc, p0: p0, v0: v0, L: L))
            } else if ffIO == "net" {
                let tn = t0 + fl.tNet!
                segments.append(RawSegment(a: t0, b: tn, p0: p0, v0: v0, L: L))
                let pd = fl.pNet! + SIMD3(hs * 0.05, 0.0, 0.0)           // drops off the net on the hitter's side
                let vdx = hs * rng.uniform(0.2, 1.0), vdy = rng.normal(0, 0.2), vdz = rng.uniform(-0.5, 0.3)
                let vd = SIMD3(vdx, vdy, vdz)
                let drop = freeFlight(tn, pd, vd, L, tMax: 3.0)
                let tFloor = tn + drop.tLand
                segments.append(RawSegment(a: tn, b: tFloor, p0: pd, v0: vd, L: L))
                segEnd = tn; shotOutcome = "net"
                end = RawEnd(reason: "net", tEnd: tn, tFloor: tFloor, rest: drop.pLand, landing: nil, margin: nil,
                             winner: hitterSide == "near" ? "far" : "near")
            } else {                                                            // lands: in / out
                let tl = t0 + fl.tLand
                segments.append(RawSegment(a: t0, b: tl, p0: p0, v0: v0, L: L))
                segEnd = tl; shotOutcome = ffIO
                let other = hitterSide == "near" ? "far" : "near"
                end = RawEnd(reason: ffIO, tEnd: tl, tFloor: tl, rest: fl.pLand, landing: landXY, margin: ffMargin,
                             winner: ffIO == "in" ? hitterSide : other)
                for k in players.sideKeys(recvSide) { movers[k]!.recover(t0 + SimConst.reactionS + 0.15, bases[k]!, speed: 1.2) }
            }
            shots.append(SimShot(index: shots.count, hitter: hitterSide, player: hitter, shotType: shotType, frame: -1, t0: t0,
                                 p0: p0, v0: v0, dragLength: L, targetXY: tgt, landingXY: landXY, landingT: t0 + fl.tLand,
                                 netCrossingHeight: zNet, inOut: ffIO, margin: ffMargin, endT: segEnd, endFrame: -1,
                                 outcome: shotOutcome))
            guard let (key, j, _) = contact else { break }
            hitter = key; t0 = t0 + fl.ts[j]; p0 = fl.P[j]
            kind = SimShots.chooseKind(rng, z: p0.z, depth: abs(p0.x))
        }
        guard let e = end else { throw SimError.planningFailed }
        for k in players.keys { movers[k]!.recover(e.tFloor + 0.4, bases[k]!, speed: 1.0) }
        return RawRally(shots: shots, segments: segments, hits: hits, end: e, server: serverSide, tServe: tServe,
                        pServe: hits[0].pos)
    }

    static func fallbackKindsExcluding(_ kind: String, z: Double, depth: Double) -> [String] {
        SimShots.fallbackKinds(z: z, depth: depth).filter { $0 != kind }
    }
}

public enum SimError: Error, CustomStringConvertible {
    case planningFailed
    case unknownCameraStyle(String)
    public var description: String {
        switch self {
        case .planningFailed: return "RuntimeError: shot planning failed"
        case .unknownCameraStyle(let s): return "ValueError: unknown camera style '\(s)'"
        }
    }
}
