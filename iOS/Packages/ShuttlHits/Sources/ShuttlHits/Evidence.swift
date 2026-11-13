import Foundation
import ShuttlCore
import simd

/// Player observations by frame with a resolved court side per observation (hits.py `_PlayerIndex`).
public struct PlayerIndex: Sendable {
    public struct Entry: Sendable { public var obs: PlayerObservation; public var side: CourtSide }
    public private(set) var byFrame: [Int: [Entry]] = [:]

    public init(players: [PlayerObservation], camera: CameraModel?) {
        var unknownYs: [Double] = []
        var unknownRefs: [(frame: Int, pos: Int)] = []
        for o in players {
            let s = PlayerIndex.side(o, camera)
            let f = o.frame
            var list = byFrame[f] ?? []
            if s == .unknown {
                unknownYs.append(o.bbox.y2)
                unknownRefs.append((f, list.count))
            }
            list.append(Entry(obs: o, side: s))
            byFrame[f] = list
        }
        if !unknownYs.isEmpty { resolveByImageOrder(unknownYs, unknownRefs) }
    }

    static func side(_ o: PlayerObservation, _ camera: CameraModel?) -> CourtSide {
        if o.side == .near || o.side == .far { return o.side }
        if let fw = o.footWorld, fw.x.isFinite, fw.y.isFinite { return CourtModel.sideOf(x: fw.x) }
        if let camera {
            let b = o.bbox
            let p = camera.pixelToFloor(SIMD2((b.x1 + b.x2) / 2, b.y2))
            if p.x.isFinite && p.y.isFinite && p.z.isFinite { return CourtModel.sideOf(x: p.x) }
        }
        return .unknown
    }

    /// No geometry: lower bbox bottom (larger y) = near; threshold from a 1-D 2-means.
    private mutating func resolveByImageOrder(_ ys: [Double], _ refs: [(frame: Int, pos: Int)]) {
        var c = [NP.percentile(ys, 20), NP.percentile(ys, 80)]
        var a = [Int](repeating: 0, count: ys.count)
        for _ in 0..<10 {
            for i in ys.indices { a[i] = abs(ys[i] - c[1]) < abs(ys[i] - c[0]) ? 1 : 0 }   // argmin: first on ties
            var nc = c
            for k in 0..<2 {
                let members = ys.indices.filter { a[$0] == k }.map { ys[$0] }
                if !members.isEmpty { nc[k] = NP.mean(members) }
            }
            c = nc
        }
        let thr = abs(c[1] - c[0]) > 1 ? NP.mean(c) : Double.infinity
        for r in refs {
            let y2 = byFrame[r.frame]![r.pos].obs.bbox.y2
            byFrame[r.frame]![r.pos].side = y2 > thr ? .near : .far
        }
    }

    /// Nearest observation per track within ±radius frames (insertion order of first sighting).
    public func around(_ frame: Int, _ radius: Int) -> [Entry] {
        var order: [Int] = []
        var best: [Int: (d: Int, e: Entry)] = [:]
        for d in -radius...radius {
            guard let lst = byFrame[frame + d] else { continue }
            for e in lst {
                let tid = e.obs.trackID
                if let cur = best[tid] {
                    if abs(d) < cur.d { best[tid] = (abs(d), e) }
                } else {
                    best[tid] = (abs(d), e); order.append(tid)
                }
            }
        }
        return order.map { best[$0]!.e }
    }
}

extension Hits {
    /// Distance (in bbox heights) from pixel uv to the region this player's racket can reach.
    static func reachDistance(_ o: PlayerObservation, _ uv: SIMD2<Double>) -> Double {
        let x1 = o.bbox.x1, y1 = o.bbox.y1, x2 = o.bbox.x2, y2 = o.bbox.y2
        let h = NP.pyMax(y2 - y1, 1.0)
        if let kp = o.keypoints {
            let r = (HitConstants.racketReachM / HitConstants.playerHeightM) * h * 1.25
            var dw: [Double] = []
            for k in [HitConstants.leftWrist, HitConstants.rightWrist]
            where kp.count > k && kp[k].conf > 0.3 && kp[k].x.isFinite && kp[k].y.isFinite {
                dw.append(NP.pyMax(0.0, Foundation.hypot(uv.x - kp[k].x, uv.y - kp[k].y) - r) / h)
            }
            if !dw.isEmpty { return dw.dropFirst().reduce(dw[0]) { NP.pyMin($0, $1) } }
        }
        let dx = NP.pyMax(NP.pyMax(x1 - 0.45 * h - uv.x, 0.0), uv.x - (x2 + 0.45 * h))
        let dy = NP.pyMax(NP.pyMax(y1 - 0.65 * h - uv.y, 0.0), uv.y - (y2 + 0.05 * h))
        return NP.pyHypot(dx, dy) / h
    }

    /// Fill reach (hit log-odds term), side_llr (player part) and track ids.
    static func playerEvidence(_ c: inout HitCandidate, _ idx: PlayerIndex, _ radius: Int, _ cfg: HitConfig) {
        let obs = idx.around(c.frame, radius)
        if obs.isEmpty { return }
        var bestN: (lp: Double, tid: Int?) = (-.infinity, nil)
        var bestF: (lp: Double, tid: Int?) = (-.infinity, nil)
        for e in obs {
            guard e.side == .near || e.side == .far else { continue }
            let q = reachDistance(e.obs, c.xy) / 0.15
            let lp = -0.5 * Foundation.pow(q, 2)
            if e.side == .near { if lp > bestN.lp { bestN = (lp, e.obs.trackID) } }
            else if lp > bestF.lp { bestF = (lp, e.obs.trackID) }
        }
        var ln = bestN.lp, lf = bestF.lp
        let top = NP.pyMax(ln, lf)
        if top.isFinite { c.reach = cfg.playerWeight * NP.clip(1.0 + top, -3.0, 1.0) }
        c.nearTID = ln > -4.5 ? bestN.tid : nil
        c.farTID = lf > -4.5 ? bestF.tid : nil
        if !ln.isFinite { ln = -3.0 }            // side not seen: weak default
        if !lf.isFinite { lf = -3.0 }
        c.sideLLR += cfg.playerWeight * NP.clip(ln - lf, -8.0, 8.0)
    }

    @inline(__always) static func heightPrior(_ z: Double) -> Double {
        NP.clip((z - 0.05) / 0.25, 0, 1) * NP.clip((3.3 - z) / 0.5, 0, 1)
    }

    /// log P(near)/P(far) that a contact seen at pixel uv lies in the near / far hitting volume.
    static func raySideLLR(_ camera: CameraModel, _ uv: SIMD2<Double>, doublesMargin: Double = 1.2) -> Double {
        let (o, d) = camera.ray(uv)
        let lam = NP.linspace(0.5, 45.0, 400)
        let yLim = CourtModel.doublesHalfWidth + doublesMargin
        let xLim = CourtModel.halfLength + 1.8
        var near: [Double] = [], far: [Double] = []
        near.reserveCapacity(400); far.reserveCapacity(400)
        for l in lam {
            let x = o.x + l * d.x, y = o.y + l * d.y, z = o.z + l * d.z
            var w = l * l * heightPrior(z)
            w *= abs(y) <= yLim ? 1.0 : 0.0
            w *= abs(x) <= xLim ? 1.0 : 0.0
            if x < -0.05 { near.append(w) }
            if x > 0.05 { far.append(w) }
        }
        let sn = NP.sum(near), sf = NP.sum(far)
        if sn + sf <= 0 { return 0.0 }
        let eps = 1e-3 * (sn + sf)
        return NP.clip(Foundation.log((sn + eps) / (sf + eps)), -6.0, 6.0)
    }
}
