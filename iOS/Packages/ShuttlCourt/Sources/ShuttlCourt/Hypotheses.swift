/// Line-quad × model-rectangle homography hypotheses. Port of court_calibration._hypotheses.
import Foundation
import simd

/// Hypotheses of one (transverse family, longitudinal family) pair, in Python order.
struct HypothesisSet {
    var H: [M9] = []
    var score: [Float] = []
    var ref: [SIMD2<Double>] = []

    mutating func append(_ o: HypothesisSet) {
        H += o.H; score += o.score; ref += o.ref
    }
}

enum Hypotheses {
    /// `_hypotheses(feat, fam_t, fam_l, max_lines=8)`. `lines` are the merged image lines
    /// the families index into.
    static func enumerate(_ feat: Features, lines: [ImageLine], famT: VPFamily, famL: VPFamily,
                          maxLines: Int = 8) -> HypothesisSet {
        let c = SIMD2(Double(feat.w) / 2, Double(feat.h) / 2)
        // sorted(key=-weight) is stable; then [:max_lines]
        func top(_ f: VPFamily) -> [ImageLine] {
            let ls = f.members.map { lines[$0] }
            return NumpyCompat.argsortStable(ls.map { -$0.weight }).prefix(maxLines).map { ls[$0] }
        }
        let t0 = top(famT), l0 = top(famL)
        let T = ModelCombos.orderFamily(famT.v, t0, c).map { t0[$0] }
        let L = ModelCombos.orderFamily(famL.v, l0, c).map { l0[$0] }
        guard T.count >= 2, L.count >= 2 else { return HypothesisSet() }
        var pts = [[SIMD2<Double>]](repeating: [], count: T.count)
        var far = [[Bool]](repeating: [], count: T.count)
        for i in 0..<T.count {
            for k in 0..<L.count {
                let X = cross3(T[i].l, L[k].l)
                let p = SIMD2(X.x / X.z, X.y / X.z)
                let d = p - c
                pts[i].append(p)
                far[i].append(!(p.x.isFinite && p.y.isFinite) || (d.x * d.x + d.y * d.y).squareRoot() > 5 * feat.D)
            }
        }
        var Hq: [M9] = []
        for i in 0..<T.count {
            for j in (i + 1)..<T.count {
                for k in 0..<L.count {
                    for m in (k + 1)..<L.count {
                        if far[i][k] || far[j][k] || far[j][m] || far[i][m] { continue }
                        let h = ModelCombos.squareToQuad(pts[i][k], pts[j][k], pts[j][m], pts[i][m])
                        if ModelCombos.allFinite(h) { Hq.append(h) }
                    }
                }
            }
        }
        var out = HypothesisSet()
        let combos = ModelCombos.all
        for h in Hq {                                   // chunking in Python does not change order
            for a in combos {
                let Hb = ModelCombos.compose(h, a)
                if HypothesisScore.plausible(Hb, ref: a.ref, cx: c.x, cy: c.y, D: feat.D) {
                    out.H.append(Hb); out.ref.append(a.ref)
                }
            }
        }
        out.score = HypothesisScore.lineScore(feat, out.H, .vcoarse)
        return out
    }

    /// All ordered family pairs (a != b) of the families with >= 2 members, concatenated.
    static func all(_ feat: Features, lines: [ImageLine], families: [VPFamily]) -> (HypothesisSet, [Int]) {
        var out = HypothesisSet()
        var perPair: [Int] = []
        for a in families.indices {
            for b in families.indices where a != b {
                let s = enumerate(feat, lines: lines, famT: families[a], famL: families[b])
                perPair.append(s.H.count)
                out.append(s)
            }
        }
        return (out, perPair)
    }
}
