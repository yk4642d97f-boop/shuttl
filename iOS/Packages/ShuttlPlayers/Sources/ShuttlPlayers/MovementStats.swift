import Foundation
import ShuttlCore

/// Footwork statistics (players.movement_stats), numbers rounded exactly like Python.
public struct MovementStats: Sendable, Equatable {
    public var distanceM: Double
    public var avgSpeedMs: Double
    public var maxSpeedMs: Double
    /// HEAT_ROWS × HEAT_COLS fraction of tracked time; row 0 = at the net, column 0 = world y −3.05 m.
    public var heatmap: [[Double]]
    public var courtCoverage: Double
    public var trackedS: Double
    /// Same numbers per track id (doubles); empty inside a per-player entry.
    public var players: [Int: MovementStats]

    public static let empty = MovementStats(distanceM: 0, avgSpeedMs: 0, maxSpeedMs: 0,
                                            heatmap: Array(repeating: Array(repeating: 0, count: Heat.cols), count: Heat.rows),
                                            courtCoverage: 0, trackedS: 0, players: [:])

    /// JSON-ready dictionary with the Python keys (serialize.py `movement`).
    public var jsonValue: JSONValue {
        var o: [String: JSONValue] = [
            "distance_m": .number(distanceM), "avg_speed_ms": .number(avgSpeedMs),
            "max_speed_ms": .number(maxSpeedMs), "court_coverage": .number(courtCoverage),
            "tracked_s": .number(trackedS),
            "heatmap": .array(heatmap.map { .array($0.map { .number($0) }) }),
        ]
        if !isPerPlayer {
            o["players"] = .object(Dictionary(uniqueKeysWithValues: players.map { (String($0.key), $0.value.jsonValue) }))
        }
        return .object(o)
    }

    var isPerPlayer = false

    public init(distanceM: Double, avgSpeedMs: Double, maxSpeedMs: Double, heatmap: [[Double]],
                courtCoverage: Double, trackedS: Double, players: [Int: MovementStats]) {
        self.distanceM = distanceM; self.avgSpeedMs = avgSpeedMs; self.maxSpeedMs = maxSpeedMs
        self.heatmap = heatmap; self.courtCoverage = courtCoverage; self.trackedS = trackedS; self.players = players
    }
}

public enum Movement {
    /// Stats for the player(s) on `side`. `timestamps`/`fps` default to the track list's.
    public static func stats(_ tracks: PlayerTracks, side: CourtSide, config: PlayersConfig = .default) -> MovementStats {
        stats(tracks.observations, side: side, timestamps: tracks.timestamps, fps: tracks.fps, config: config)
    }

    public static func stats(_ obs: [PlayerObservation], side: CourtSide, timestamps: [Double]?, fps: Double?,
                             config: PlayersConfig = .default) -> MovementStats {
        let fps = (fps == nil || fps == 0) ? 30.0 : fps!
        func tOf(_ f: Int) -> Double {
            if let ts = timestamps, f < ts.count { return ts[f] }
            return Double(f) / fps
        }
        var groups: [Int: [PlayerObservation]] = [:]
        for o in obs where o.side == side {
            if let fw = o.footWorld, fw.x.isFinite, fw.y.isFinite { groups[o.trackID, default: []].append(o) }
        }
        var per: [(Int, MovementStats, [Double])] = []
        for tid in groups.keys.sorted() {
            let g = groups[tid]!.enumerated().sorted { a, b in
                a.element.frame < b.element.frame || (a.element.frame == b.element.frame && a.offset < b.offset)
            }.map(\.element)
            let (s, dwell) = statsOne(t: g.map { tOf($0.frame) }, xy: g.map { $0.footWorld! }, config: config)
            per.append((tid, s, dwell))
        }
        if per.isEmpty { return .empty }
        var totT = 0.0
        for p in per { totT = totT + p.1.trackedS }
        var heat = [Double](repeating: 0, count: Heat.rows * Heat.cols)
        for p in per {
            for r in 0..<Heat.rows { for c in 0..<Heat.cols { heat[r * Heat.cols + c] += p.1.heatmap[r][c] * p.1.trackedS } }
        }
        let denom = pyMax(totT, 1e-9)
        heat = heat.map { $0 / denom }
        var dwell = [Double](repeating: 0, count: Heat.rows * Heat.cols)
        for p in per { for i in 0..<dwell.count { dwell[i] += p.2[i] } }
        var dist = 0.0
        for p in per { dist = dist + p.1.distanceM }
        var maxSpeed = per[0].1.maxSpeedMs
        for p in per.dropFirst() where p.1.maxSpeedMs > maxSpeed { maxSpeed = p.1.maxSpeedMs }
        let cov = Double(dwell.filter { $0 >= 0.2 }.count) / Double(dwell.count)
        var players: [Int: MovementStats] = [:]
        for p in per { players[p.0] = p.1 }
        return MovementStats(distanceM: NP.pyRound(dist, 2),
                             avgSpeedMs: totT > 0 ? NP.pyRound(dist / totT, 3) : 0.0,
                             maxSpeedMs: maxSpeed,
                             heatmap: grid(heat.map { NP.npRound($0, 5) }),
                             courtCoverage: NP.pyRound(cov, 3),
                             trackedS: NP.pyRound(totT, 2), players: players)
    }

    static func grid(_ flat: [Double]) -> [[Double]] {
        (0..<Heat.rows).map { r in Array(flat[(r * Heat.cols)..<((r + 1) * Heat.cols)]) }
    }

    /// players._stats_one: per-track stats + raw dwell (flat, row-major).
    static func statsOne(t: [Double], xy: [SIMD2<Double>], config: PlayersConfig) -> (MovementStats, [Double]) {
        var cuts: [Int] = []
        for i in 0..<max(t.count - 1, 0) where t[i + 1] - t[i] > config.maxInterpS { cuts.append(i + 1) }
        var dist = 0.0, tracked = 0.0
        var speeds: [Double] = []
        var dwell = [Double](repeating: 0, count: Heat.rows * Heat.cols)
        let starts = [0] + cuts, stops = cuts + [t.count]
        for (a, b) in zip(starts, stops) {
            let tt = Array(t[a..<b]), pp = Array(xy[a..<b])
            if tt.count < 2 { continue }
            tracked += tt[tt.count - 1] - tt[0]
            // path length at ~10 Hz
            let gridT = NP.arange(tt[0], tt[tt.count - 1] + 1e-9, 0.1)
            let px = pp.map(\.x), py = pp.map(\.y)
            let rs = gridT.map { SIMD2(NP.interp($0, tt, px), NP.interp($0, tt, py)) }
            var seglen = [Double]()
            seglen.reserveCapacity(max(rs.count - 1, 0))
            for i in 0..<max(rs.count - 1, 0) { seglen.append(NP.normAxis(rs[i + 1] - rs[i])) }
            dist += NP.sum(seglen)
            // windowed speed
            let h = config.speedWinS / 2
            let idx = (0..<tt.count).map(Double.init)
            for i in 0..<tt.count {
                let lo = NP.interp(tt[i] - h, tt, idx), hi = NP.interp(tt[i] + h, tt, idx)
                let ia = Int(lo.rounded(.down)), ib = Int(hi.rounded(.up))
                if tt[ib] - tt[ia] >= 0.5 * config.speedWinS {
                    let v = NP.norm1D(pp[ib] - pp[ia]) / (tt[ib] - tt[ia])
                    if v <= config.speedCap { speeds.append(v) }
                }
            }
            // dwell time per half-court cell
            var w = [Double](repeating: 0, count: tt.count)
            for i in 0..<(tt.count - 1) { w[i] = tt[i + 1] - tt[i] }
            w[tt.count - 1] = tt[tt.count - 1] - tt[tt.count - 2]
            for i in 0..<tt.count {
                let ri = min(max(truncInt(abs(pp[i].x) / CourtModel.halfLength * Double(Heat.rows)), 0), Heat.rows - 1)
                let ci = min(max(truncInt((pp[i].y + CourtModel.doublesHalfWidth) / CourtModel.doublesWidth * Double(Heat.cols)), 0),
                             Heat.cols - 1)
                dwell[ri * Heat.cols + ci] += w[i]
            }
        }
        let total = NP.sum(dwell)
        let heat = total > 0 ? dwell.map { $0 / total } : dwell
        let cov = Double(dwell.filter { $0 >= 0.2 }.count) / Double(dwell.count)
        var s = MovementStats(distanceM: NP.pyRound(dist, 2),
                              avgSpeedMs: tracked > 0 ? NP.pyRound(dist / tracked, 3) : 0.0,
                              maxSpeedMs: speeds.isEmpty ? 0.0 : NP.pyRound(NP.percentile(speeds, 95), 3),
                              heatmap: grid(heat.map { NP.npRound($0, 5) }),
                              courtCoverage: NP.pyRound(cov, 3), trackedS: NP.pyRound(tracked, 2), players: [:])
        s.isPerPlayer = true
        return (s, dwell)
    }
}

/// numpy `astype(int)` (truncation) without trapping on out-of-range values.
@inline(__always) func truncInt(_ x: Double) -> Int {
    guard x.isFinite else { return Int.min }
    return Int(max(min(x, 1e15), -1e15))
}
