import Foundation
import CoreGraphics
import simd

// MARK: - Per-frame association

extension TrajectoryTracker {

    func predict(_ tr: inout Track, to t: Double) {
        let dt = t - tr.time
        guard dt > 0 else { return }
        let q = jerkDensity(speedPx: hypot(tr.fx.velocity, tr.fy.velocity))
        tr.fx.predict(dt: dt, q: q)
        tr.fy.predict(dt: dt, q: q)
        tr.time = t
    }

    /// Gated cost of assigning `c` to a predicted track, nil when outside the gates.
    func associationCost(_ pred: Track, _ c: TrackerCandidate, staticCost: Double) -> Double? {
        let ex = Double(c.pixel.x) - pred.fx.position, ey = Double(c.pixel.y) - pred.fy.position
        let d2 = ex * ex / (pred.fx.P[0, 0] + measurementVar) + ey * ey / (pred.fy.P[0, 0] + measurementVar)
        guard d2 <= config.gateChi2, let last = pred.points.last else { return nil }
        let reach = maxSpeedPx * (pred.time - pred.lastHit) + 4 * sigmaPx
        guard hypot(Double(c.pixel.x - last.pixel.x), Double(c.pixel.y - last.pixel.y)) <= reach else { return nil }
        let p = Double(min(max(c.score, 0.05), 0.95))
        let cost = 0.5 * d2 + staticCost - log(p / (1 - p))
        return cost <= config.acceptCost ? cost : nil
    }

    func apply(_ c: TrackerCandidate, staticCost: Double, to tr: inout Track) {
        tr.fx.update(Double(c.pixel.x), r: measurementVar)
        tr.fy.update(Double(c.pixel.y), r: measurementVar)
        tr.lastHit = tr.time
        tr.points.append(TrackPoint(time: tr.time, pixel: c.pixel, score: c.score))
        tr.staticSum += staticCost
        tr.maxTravel = max(tr.maxTravel, hypot(Double(c.pixel.x - tr.firstPixel.x), Double(c.pixel.y - tr.firstPixel.y)))
    }

    func step(time t: Double, candidates: [TrackerCandidate]) {
        guard var map = staticMap else { return }
        var pred: Track?
        if var a = active { predict(&a, to: t); pred = a }
        let shuttle = pred.map { CGPoint(x: $0.fx.position, y: $0.fy.position) }
        map.observe(candidates.map(\.pixel), shuttle: shuttle, farPx: config.staticFar * diag,
                    time: t, tau: config.staticMemory)
        staticMap = map
        let statics = candidates.map { map.cost(at: $0.pixel, time: t, frameInterval: frameInterval, config: config) }
        var used = [Bool](repeating: false, count: candidates.count)

        // 1. Active track: best gated candidate, else coast.
        var activeMeasured = false
        if var a = pred {
            var best: (i: Int, cost: Double)?
            for (i, c) in candidates.enumerated() {
                if let cost = associationCost(a, c, staticCost: statics[i]), cost < (best?.cost ?? .infinity) {
                    best = (i, cost)
                }
            }
            if let b = best {
                used[b.i] = true
                apply(candidates[b.i], staticCost: statics[b.i], to: &a)
                activeMeasured = true
            }
            active = a
        }
        let dupRadius = config.staticFar * diag
        let activeNow = active?.points.last?.pixel

        // 2. Tentatives: predict, greedy global assignment by cost.
        for i in tentatives.indices { predict(&tentatives[i], to: t) }
        if activeMeasured, let p = activeNow {    // a twin riding along the shuttle blob would fake hits
            tentatives.removeAll { tr in
                tr.points.last.map { hypot(Double($0.pixel.x - p.x), Double($0.pixel.y - p.y)) < dupRadius } ?? false
            }
        }
        var pairs: [(ti: Int, ci: Int, cost: Double)] = []
        for (ti, tr) in tentatives.enumerated() {
            for (ci, c) in candidates.enumerated() where !used[ci] {
                if let cost = associationCost(tr, c, staticCost: statics[ci]) { pairs.append((ti, ci, cost)) }
            }
        }
        var tUsed = [Bool](repeating: false, count: tentatives.count)
        for p in pairs.sorted(by: { $0.cost < $1.cost }) where !tUsed[p.ti] && !used[p.ci] {
            tUsed[p.ti] = true
            used[p.ci] = true
            apply(candidates[p.ci], staticCost: statics[p.ci], to: &tentatives[p.ti])
        }
        tentatives.removeAll { t - $0.lastHit > config.tentativeMaxGap }

        // 3. Spawn from leftovers (not known distractors, not twins of the live shuttle).
        for (ci, c) in candidates.enumerated() where !used[ci] && statics[ci] < 0.9 * config.staticCost {
            if activeMeasured, let p = activeNow, hypot(Double(c.pixel.x - p.x), Double(c.pixel.y - p.y)) < dupRadius {
                continue
            }
            tentatives.append(newTrack(at: c, time: t, staticCost: statics[ci]))
        }
        if tentatives.count > config.maxTentatives {
            tentatives.sort { ($0.points.count, $0.lastHit) > ($1.points.count, $1.lastHit) }
            tentatives.removeLast(tentatives.count - config.maxTentatives)
        }

        // 4. Termination, then promotion / re-acquisition.
        if let a = active, t - a.lastHit > config.maxCoast { endActive(.lost) }
        promoteOrMerge(time: t)

        // 5. Hits.
        detectKinks()
        trimHistory(time: t)
    }

    func isConfirmed(_ tr: Track) -> Bool {
        tr.points.count >= config.confirmHits
            && tr.maxTravel >= max(config.confirmMinTravel * diag, 4 * sigmaPx)
            && tr.staticSum / Double(tr.points.count) < 0.5 * config.staticCost
    }

    func promoteOrMerge(time t: Double) {
        let confirmed = tentatives.indices.filter { isConfirmed(tentatives[$0]) }
        guard let idx = confirmed.max(by: { tentatives[$0].points.count < tentatives[$1].points.count }) else { return }
        guard let a = active else {
            active = tentatives.remove(at: idx)
            segmentStart = 0
            kinkCursor = 0
            pendingKink = nil
            return
        }
        guard t - a.lastHit > 1e-9, let last = a.points.last else { return }   // active is healthy
        // Re-acquisition after a hit: fresh points that start within reach of the lost track.
        let tr = tentatives[idx]
        let fresh = tr.points.filter { $0.time > a.lastHit }
        guard fresh.count >= config.confirmHits, let first = fresh.first else { return }
        let gap = first.time - a.lastHit
        let jump = hypot(Double(first.pixel.x - last.pixel.x), Double(first.pixel.y - last.pixel.y))
        guard gap <= config.maxCoast, jump <= maxSpeedPx * gap + 4 * sigmaPx else { return }
        tentatives.remove(at: idx)
        merge(tr, fresh: fresh)
    }
}
