import Foundation
import CoreGraphics
import simd

// MARK: - Hits, flight segments, output

extension TrajectoryTracker {

    /// Joins a re-acquired track onto the coasting active one; the junction is a hit.
    func merge(_ tr: Track, fresh: [TrackPoint]) {
        commitPendingKink()
        guard var a = active, let last = a.points.last, let first = fresh.first else { return }
        let vin = fitVelocity(a.points.suffix(4))?.v
        let vout = fitVelocity(fresh[...])?.v
        registerHit(HitEvent(time: 0.5 * (last.time + first.time), pixel: last.pixel, source: .reacquired,
                             turnDegrees: vin.flatMap { i in vout.map { angle(i, $0) } }),
                    closingAt: a.points.count - 1)
        segmentStart = a.points.count
        kinkCursor = segmentStart
        a.points += fresh
        a.fx = tr.fx
        a.fy = tr.fy
        a.time = tr.time
        a.lastHit = tr.lastHit
        a.staticSum += tr.staticSum
        active = a
    }

    /// Records a hit (NMS in time) and closes the flight that ends at `closingAt`.
    func registerHit(_ hit: HitEvent, closingAt end: Int) {
        closeSegment(through: end, reason: .hit)
        if let prev = hits.last, hit.time - prev.time < config.kinkNMS { return }
        hits.append(hit)
        if hits.count > 32 { hits.removeFirst(hits.count - 32) }
    }

    func closeSegment(through end: Int, reason: FlightSegment.End) {
        guard let a = active, segmentStart < a.points.count else { return }
        let hi = min(end, a.points.count - 1)
        guard hi - segmentStart + 1 >= 3 else { return }
        let pts = Array(a.points[segmentStart...hi])
        let speed = calibration.flatMap {
            FlightSpeedEstimator.estimateKmh(points: pts, calibration: $0, imageSize: imageSize)
        }
        completedFlights.append(FlightSegment(points: pts, end: reason, approxSpeedKmh: speed))
        if completedFlights.count > 16 { completedFlights.removeFirst(completedFlights.count - 16) }
        if let s = speed, let t = pts.last?.time { lastSpeed = (s, t) }
    }

    func endActive(_ reason: FlightSegment.End) {
        commitPendingKink()
        guard let a = active else { return }
        closeSegment(through: a.points.count - 1, reason: reason)
        recentTrail = a.points.suffix(64)
        active = nil
        segmentStart = 0
        kinkCursor = 0
        pendingKink = nil
    }

    // MARK: Kinks

    func detectKinks() {
        guard let a = active, let tLast = a.points.last?.time else { return }
        kinkCursor = max(kinkCursor, segmentStart + 1)
        while kinkCursor < a.points.count, a.points[kinkCursor].time <= tLast - config.kinkWindow + 1e-9 {
            let i = kinkCursor
            kinkCursor += 1
            let ti = a.points[i].time
            if let k = kinkTest(at: i, in: a.points), k.ratio > 1 {
                if let p = pendingKink, ti - a.points[p.index].time < config.kinkNMS {
                    if k.ratio > p.ratio { pendingKink = (i, k.ratio, k.turn) }
                } else {
                    commitPendingKink()
                    pendingKink = (i, k.ratio, k.turn)
                }
            } else if let p = pendingKink, ti - a.points[p.index].time >= config.kinkNMS {
                commitPendingKink()
            }
            if active == nil { return }
        }
    }

    /// Unexplained Δv at point `i` relative to the drag/noise/gravity threshold (> 1 = hit).
    func kinkTest(at i: Int, in pts: [TrackPoint]) -> (ratio: Double, turn: Double)? {
        let ti = pts[i].time
        var lo = i, hi = i
        while lo > segmentStart, ti - pts[lo - 1].time <= config.kinkWindow + 1e-9, i - lo < config.kinkMaxPoints { lo -= 1 }
        while hi < pts.count - 1, pts[hi + 1].time - ti <= config.kinkWindow + 1e-9, hi - i < config.kinkMaxPoints { hi += 1 }
        guard i - lo + 1 >= config.kinkMinPoints, hi - i + 1 >= config.kinkMinPoints,
              let fin = fitVelocity(pts[lo...i]), let fout = fitVelocity(pts[i...hi]) else { return nil }
        let vin = fin.v, vout = fout.v
        let s2 = simd_length_squared(vin)
        guard s2 > 1e-9 else { return nil }
        let rho = min(max(simd_dot(vout, vin) / s2, 0), config.kinkRhoHi)
        let err = simd_length(vout - rho * vin)
        let noise = config.kinkZ * sigmaPx * (fin.seScale * fin.seScale + fout.seScale * fout.seScale).squareRoot()
        let span = pts[hi].time - pts[lo].time
        let thresh = noise + config.kinkAccel * diag * span + config.kinkMinDV * diag
        return (err / thresh, angle(vin, vout))
    }

    func angle(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double {
        let la = simd_length(a), lb = simd_length(b)
        guard la > 1e-9, lb > 1e-9 else { return 0 }
        return acos(min(max(simd_dot(a, b) / (la * lb), -1), 1)) * 180 / .pi
    }

    func commitPendingKink() {
        guard let p = pendingKink, let a = active, p.index < a.points.count else { pendingKink = nil; return }
        pendingKink = nil
        let pt = a.points[p.index]
        registerHit(HitEvent(time: pt.time, pixel: pt.pixel, source: .kink, turnDegrees: p.turn), closingAt: p.index)
        segmentStart = p.index                   // the hit point belongs to both flights
        kinkCursor = max(kinkCursor, p.index + 1)
        refilter(from: p.index)
    }

    /// Restarts the filter at the hit so the new flight's velocity is not smeared across it.
    func refilter(from index: Int) {
        guard var a = active, index < a.points.count else { return }
        let first = a.points[index]
        var tr = newTrack(at: TrackerCandidate(pixel: first.pixel, score: first.score), time: first.time, staticCost: 0)
        for p in a.points[(index + 1)...] {
            predict(&tr, to: p.time)
            apply(TrackerCandidate(pixel: p.pixel, score: p.score), staticCost: 0, to: &tr)
        }
        predict(&tr, to: a.time)
        a.fx = tr.fx
        a.fy = tr.fy
        active = a
    }

    // MARK: Housekeeping / output

    func trimHistory(time t: Double) {
        recentTrail.removeAll { t - $0.time > config.trailDuration }
        guard var a = active else { return }
        let keepFrom = max(0, min(segmentStart - 1, kinkCursor - config.kinkMaxPoints - 1, a.points.count - 600))
        let hardCap = a.points.count - 2400
        let drop = max(keepFrom, hardCap)
        guard drop > 64 else { return }
        a.points.removeFirst(drop)
        segmentStart = max(0, segmentStart - drop)
        kinkCursor = max(0, kinkCursor - drop)
        if let p = pendingKink { pendingKink = p.index >= drop ? (p.index - drop, p.ratio, p.turn) : nil }
        active = a
    }

    func snapshot() -> TrackerSnapshot {
        guard let t = currentTime, imageSize.width > 0 else { return .empty }
        let firstActive = active?.points.first?.time ?? .infinity
        let pts = recentTrail.filter { $0.time < firstActive } + (active?.points.suffix(128) ?? [])
        let w = Double(imageSize.width), h = Double(imageSize.height)
        let trail = pts.filter { t - $0.time <= config.trailDuration }
            .map { CGPoint(x: Double($0.pixel.x) / w, y: Double($0.pixel.y) / h) }
        let status: TrackerSnapshot.Status = active.map { t - $0.lastHit < 1e-9 ? .tracking : .coasting } ?? .searching
        let speed = lastSpeed.flatMap { t - $0.time <= config.speedHold ? $0.kmh : nil }
        return TrackerSnapshot(status: status, trail: trail, imageSize: imageSize, time: t,
                               lastHit: hits.last, approxSpeedKmh: speed)
    }
}
