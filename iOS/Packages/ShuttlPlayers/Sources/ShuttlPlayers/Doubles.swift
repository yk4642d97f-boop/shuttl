import Foundation
import ShuttlCore

/// Doubles partner identity by motion + appearance continuity (players.py):
///  1. tracklets: frame-to-frame association on constant-velocity Kalman predictions,
///     cut wherever the swapped association is almost as likely;
///  2. stitching: exact DP labelling tracklets with 2 identities (junction Gaussian
///     NLL, forward + reversed-time Kalman, + shirt/shorts colour distance).
public enum PartnerIdentity {
    /// Tracklet = [(k, j)] sampled-frame index + candidate index.
    public typealias Tracklet = [(Int, Int)]

    /// Constant-velocity Kalman state at one end of a tracklet (metres; isotropic).
    struct KF {
        var x: SIMD2<Double>
        var v: SIMD2<Double>
        var p00, p01, p11: Double
        var t: Double
        var unit: Double
        let qLink: Double

        @inline(__always) func mean(_ g: Double) -> SIMD2<Double> { x + v * g }
        @inline(__always) func variance(_ g: Double) -> Double {
            p00 + 2 * g * p01 + g * g * p11 + qLink * NP.pyPow(g, 3) / 3
        }
    }

    /// Filter the last (atEnd) or, in reversed time, the first KF_TAIL samples.
    static func kfEnd(_ frames: [PlayerCandidates], _ times: [Double], _ pts: ArraySlice<(Int, Int)>, atEnd: Bool,
                      config: PlayersConfig) -> KF {
        let seq: [(Int, Int)] = atEnd ? Array(pts.suffix(config.kfTail)) : Array(pts.prefix(config.kfTail).reversed())
        let unit = NP.mean(seq.map { frames[$0.0].unit[$0.1] })
        let (k0, j0) = seq[0]
        var x = frames[k0].pos[j0] * unit
        var v = SIMD2<Double>(0, 0)
        var p00 = NP.pyPow(frames[k0].noise[j0], 2), p01 = 0.0, p11 = 9.0
        var tp = times[k0]
        let Q = config.qLink
        for (k, j) in seq.dropFirst() {
            let g = abs(times[k] - tp)
            tp = times[k]
            x = x + v * g
            (p00, p01, p11) = (p00 + 2 * g * p01 + g * g * p11 + Q * NP.pyPow(g, 3) / 3,
                               p01 + g * p11 + Q * g * g / 2, p11 + Q * g)
            let s = p00 + NP.pyPow(frames[k].noise[j], 2)
            let k0_ = p00 / s, k1_ = p01 / s
            let inn = frames[k].pos[j] * unit - x
            (x, v) = (x + k0_ * inn, v + k1_ * inn)
            (p00, p01, p11) = (p00 - k0_ * p00, p01 - k0_ * p01, p11 - k1_ * p01)
        }
        return KF(x: x, v: v, p00: p00, p01: p01, p11: p11, t: tp, unit: unit, qLink: Q)
    }

    /// Isotropic 2-D Gaussian negative log-likelihood (constants dropped).
    @inline(__always)
    static func nll(_ e2: Double, _ variance: Double, _ v0: Double) -> Double {
        0.5 * e2 / variance + log(variance / v0)
    }

    /// Cost of detection j (frame time tk) continuing a tracklet with end state st.
    static func assocCost(_ st: KF, _ c: PlayerCandidates, _ j: Int, _ tk: Double, config: PlayersConfig) -> Double {
        let g = tk - st.t
        let z = c.pos[j] * c.unit[j]
        let r = c.noise[j]
        if NP.norm1D(z - st.x) > config.vMax * g + 2.0 * (st.p00.squareRoot() + r) { return .infinity }
        let e = z - st.mean(g)
        return nll(e.x * e.x + e.y * e.y, st.variance(g) + r * r, config.v0Link)
    }

    /// All partial injective matchings {(track, det)} in players._matchings order.
    static func matchings(_ nT: Int, _ nD: Int) -> [[(Int, Int)]] {
        var out: [[(Int, Int)]] = []
        func rec(_ d: Int, _ used: Set<Int>, _ acc: [(Int, Int)]) {
            if d == nD { out.append(acc); return }
            rec(d + 1, used, acc)
            for t in 0..<nT where !used.contains(t) {
                var u = used; u.insert(t)
                rec(d + 1, u, acc + [(t, d)])
            }
        }
        rec(0, [], [])
        return out
    }

    /// Cut the pooled detections of one half into unambiguous tracklets.
    public static func buildTracklets(_ frames: [PlayerCandidates], times: [Double], pooled: [[Int]],
                                      config: PlayersConfig = .default) -> [Tracklet] {
        var done: [Tracklet] = []
        var active: [Tracklet] = []
        for (k, js) in pooled.enumerated() {
            var still: [Tracklet] = []
            for tr in active {
                if times[k] - times[tr.last!.0] > config.trackletGapS { done.append(tr) } else { still.append(tr) }
            }
            active = still
            if js.isEmpty { continue }
            let states = active.map { kfEnd(frames, times, $0[...], atEnd: true, config: config) }
            let nA = active.count, nD = js.count
            var cost = [Double](repeating: 0, count: nA * nD)
            for (t, st) in states.enumerated() {
                for (d, j) in js.enumerated() { cost[t * nD + d] = assocCost(st, frames[k], j, times[k], config: config) }
            }
            var scored: [(Double, [(Int, Int)], Int)] = []
            for (mi, m) in matchings(nA, nD).enumerated() {
                var c = 0.0
                for (t, d) in m { c = c + cost[t * nD + d] }
                c += config.cNew * Double(nD - m.count) + config.cCoast * Double(nA - m.count)
                if c.isFinite { scored.append((c, m, mi)) }
            }
            scored.sort { $0.0 < $1.0 || ($0.0 == $1.0 && $0.2 < $1.2) }       // stable
            let (bestC, best, _) = scored[0]
            var pairs: [(Int, Int)] = best                                        // ordered dict track → det
            func pairDet(_ t: Int) -> Int? { pairs.first { $0.0 == t }?.1 }
            var unsure = Set<Int>()
            for (c, m, _) in scored.dropFirst() {
                if c - bestC >= config.ambig { break }
                for (t, d) in m {
                    let pd = pairDet(t) ?? d
                    if pd != d || best.contains(where: { $0.0 != t && $0.1 == d }) {
                        unsure.insert(t)
                        for (t2, d2) in best where d2 == d { unsure.insert(t2) }
                    }
                }
            }
            var newD = (0..<nD).filter { d in !pairs.contains { $0.1 == d } }
            for t in unsure.sorted() {
                if let i = pairs.firstIndex(where: { $0.0 == t }) { newD.append(pairs.remove(at: i).1) }
            }
            for (t, d) in pairs { active[t].append((k, js[d])) }
            if !newD.isEmpty {
                let paired = Set(pairs.map(\.0))
                var keep: [Tracklet] = []
                for (t, tr) in active.enumerated() {
                    if paired.contains(t) { keep.append(tr) } else { done.append(tr) }
                }
                active = keep + newD.sorted().map { [(k, js[$0])] }
            }
        }
        done.append(contentsOf: active)
        // sorted(done, key=(start k, end k)), stable
        return done.enumerated().sorted { a, b in
            let ka = (a.element[0].0, a.element.last!.0), kb = (b.element[0].0, b.element.last!.0)
            return ka < kb || (ka == kb && a.offset < b.offset)
        }.map(\.element)
    }

    static func meanApp(_ frames: [PlayerCandidates], _ pts: ArraySlice<(Int, Int)>) -> [Float]? {
        let rows = pts.compactMap { frames[$0.0].appRow($0.1) }
        return AppearanceDescriptor.mean(rows)
    }

    /// Cost of tracklet b continuing tracklet a (players._link_cost).
    static func linkCost(_ frames: [PlayerCandidates], _ times: [Double], _ a: Tracklet, _ b: Tracklet,
                         config: PlayersConfig) -> Double {
        let sa = kfEnd(frames, times, a[...], atEnd: true, config: config)
        let sb = kfEnd(frames, times, b[...], atEnd: false, config: config)
        let g = pyMax(sb.t - sa.t, 1e-3)
        let e1 = sa.mean(g) - sb.x, e2 = sb.mean(g) - sa.x
        var c = 0.5 * (nll(e1.x * e1.x + e1.y * e1.y, sa.variance(g) + sb.p00, config.v0Link)
                       + nll(e2.x * e2.x + e2.y * e2.y, sb.variance(g) + sa.p00, config.v0Link))
        if NP.norm1D(sb.x - sa.x) > config.vMax * g + 2.0 * (sa.p00.squareRoot() + sb.p00.squareRoot()) {
            c += 50.0
        }
        let ma = meanApp(frames, a.suffix(10)), mb = meanApp(frames, b.prefix(10))
        return c + config.wApp * AppearanceDescriptor.distance(ma?[...], mb?[...])
    }

    struct PairKey: Hashable { var a: Int; var b: Int }

    /// Exact DP labelling tracklets (sorted by start) with ≤ 2 identities
    /// (players.stitch_tracklets). Returns tracklet indices per identity.
    public static func stitch(_ frames: [PlayerCandidates], times: [Double], tracklets trk: [Tracklet],
                              config: PlayersConfig = .default) -> [[Int]] {
        let n = trk.count
        let start = trk.map { $0[0].0 }, end = trk.map { $0.last!.0 }
        var cache: [PairKey: Double] = [:]
        func link(_ a: Int, _ b: Int) -> Double {
            if a < 0 { return 0.0 }
            if let v = cache[PairKey(a: a, b: b)] { return v }
            let v = linkCost(frames, times, trk[a], trk[b], config: config)
            cache[PairKey(a: a, b: b)] = v
            return v
        }
        // insertion-ordered dictionaries
        var layerKeys: [PairKey] = [PairKey(a: -1, b: -1)]
        var layerVal: [PairKey: Double] = [PairKey(a: -1, b: -1): 0.0]
        var back: [[PairKey: (PairKey, Int?)]] = []
        for c in 0..<n {
            var nxtKeys: [PairKey] = []
            var nxt: [PairKey: Double] = [:]
            var bk: [PairKey: (PairKey, Int?)] = [:]
            func push(_ key0: PairKey, _ val: Double, _ prev: PairKey, _ act: Int?) {
                let key = PairKey(a: min(key0.a, key0.b), b: max(key0.a, key0.b))
                if let old = nxt[key] {
                    if val < old { nxt[key] = val; bk[key] = (prev, act) }
                } else if val < .infinity {
                    nxt[key] = val; bk[key] = (prev, act); nxtKeys.append(key)
                }
            }
            for lk in layerKeys {
                let v = layerVal[lk]!
                let (l1, l2) = (lk.a, lk.b)
                for (last, other) in [(l1, l2), (l2, l1)] {
                    if last < 0 || end[last] < start[c] {
                        push(PairKey(a: c, b: other), v + link(last, c), lk, last)
                    }
                    if l1 == l2 { break }
                }
                push(lk, v + config.cDrop * Double(trk[c].count), lk, nil)
            }
            if nxtKeys.count > config.stitchBeam {
                let order = nxtKeys.enumerated().sorted { a, b in
                    let va = nxt[a.element]!, vb = nxt[b.element]!
                    return va < vb || (va == vb && a.offset < b.offset)
                }.prefix(config.stitchBeam).map(\.element)
                nxtKeys = Array(order)
                let keepSet = Set(nxtKeys)
                nxt = nxt.filter { keepSet.contains($0.key) }
                bk = bk.filter { keepSet.contains($0.key) }
            }
            layerKeys = nxtKeys
            layerVal = nxt
            back.append(bk)
        }
        var key = layerKeys[0]
        for k in layerKeys.dropFirst() where layerVal[k]! < layerVal[key]! { key = k }
        var acts = [Int](repeating: -2, count: n)
        for c in stride(from: n - 1, through: 0, by: -1) {
            let (prev, act) = back[c][key]!
            acts[c] = act ?? -2                   // −2 = dropped, −1 = new identity
            key = prev
        }
        var ids: [[Int]] = []
        var owner: [Int: Int] = [:]
        for (c, act) in acts.enumerated() {
            if act == -2 { continue }
            if act == -1 {
                ids.append([c]); owner[c] = ids.count - 1
            } else {
                owner[c] = owner[act]!
                ids[owner[c]!].append(c)
            }
        }
        return ids
    }

    /// Re-assign the pooled detections of two same-half paths to two identities.
    public static func splitPartners(_ frames: [PlayerCandidates], times: [Double], paths: [[Int: Int]],
                                     config: PlayersConfig = .default) -> [[Int: Int]] {
        var pooled = [[Int]](repeating: [], count: frames.count)
        for p in paths {
            for (k, j) in p where !pooled[k].contains(j) { pooled[k].append(j) }
        }
        let trk = buildTracklets(frames, times: times, pooled: pooled, config: config)
        if trk.isEmpty { return [] }
        var out: [[Int: Int]] = []
        for ident in stitch(frames, times: times, tracklets: trk, config: config) {
            var d: [Int: Int] = [:]
            for i in ident { for (k, j) in trk[i] { d[k] = j } }
            out.append(d)
        }
        // sorted(out, key=len, reverse=True): stable descending
        return out.enumerated().sorted { a, b in
            a.element.count > b.element.count || (a.element.count == b.element.count && a.offset < b.offset)
        }.map(\.element)
    }
}
