/// Model-rectangle assignments and quad homographies.
/// Port of court_calibration._XT/_YL, _model_combos, _order_family, _square_to_quad.
/// Arithmetic order follows numpy: elementwise ufuncs are unfused, 1-D `@` / norm are
/// OpenBLAS FMA chains, einsum is a plain multiply-add.
import Foundation
import simd
import ShuttlCore

/// Row-major 3x3 as 9 doubles (numpy layout) — used for the vectorised hypothesis maths.
typealias M9 = (Double, Double, Double, Double, Double, Double, Double, Double, Double)

enum ModelCombos {
    /// `_XT`: transverse line x positions; `_YL`: longitudinal line y positions.
    static let XT: [Double] = [-CourtModel.halfLength, -CourtModel.doublesLongServiceX, -CourtModel.shortServiceX,
                               CourtModel.shortServiceX, CourtModel.doublesLongServiceX, CourtModel.halfLength]
    static let YL: [Double] = [-CourtModel.doublesHalfWidth, -CourtModel.singlesHalfWidth, 0.0,
                               CourtModel.singlesHalfWidth, CourtModel.doublesHalfWidth]

    /// One `_model_combos` entry: A = [[sx, 0, ox], [0, sy, oy], [0, 0, 1]] and its ref point.
    struct Combo: Sendable { let sx: Double, ox: Double, sy: Double, oy: Double, ref: SIMD2<Double> }

    /// `_COMBO_A`, `_COMBO_REF` (300 entries, Python loop order).
    static let all: [Combo] = {
        var out: [Combo] = []
        for a in 0..<6 {
            for b in 0..<6 where a != b {
                for c in 0..<5 {
                    for d in (c + 1)..<5 {
                        let xa = XT[a], xb = XT[b], yc = YL[c], yd = YL[d]
                        out.append(Combo(sx: 1 / (xb - xa), ox: -xa / (xb - xa),
                                         sy: 1 / (yd - yc), oy: -yc / (yd - yc),
                                         ref: SIMD2((xa + xb) / 2, (yc + yd) / 2)))
                    }
                }
            }
        }
        return out
    }()

    /// einsum("qij,mjk->qmik", Hq, A) for one (q, m): Hq · A (plain multiply-add).
    @inline(__always) static func compose(_ h: M9, _ a: Combo) -> M9 {
        (h.0 * a.sx, h.1 * a.sy, (h.0 * a.ox + h.1 * a.oy) + h.2 * 1.0,
         h.3 * a.sx, h.4 * a.sy, (h.3 * a.ox + h.4 * a.oy) + h.5 * 1.0,
         h.6 * a.sx, h.7 * a.sy, (h.6 * a.ox + h.7 * a.oy) + h.8 * 1.0)
    }

    /// `_order_family`: indices of `lines` sorted by where they cross the transversal
    /// through c (np.argsort default kind).
    static func orderFamily(_ v: SIMD3<Double>, _ lines: [ImageLine], _ c: SIMD2<Double>) -> [Int] {
        var d = SIMD2(v.x - v.z * c.x, v.y - v.z * c.y)
        d = d / (fma(d.y, d.y, d.x * d.x).squareRoot() + 1e-18)
        let lt = SIMD3(d.x, d.y, -fma(d.y, c.y, d.x * c.x))
        let n = SIMD2(-d.y, d.x)
        let s: [Double] = lines.map { ln in
            let X = cross3(ln.l, lt)
            guard abs(X.z) > 1e-12 else { return 0.0 }
            let q = SIMD2(X.x / X.z - c.x, X.y / X.z - c.y)
            return fma(n.y, q.y, n.x * q.x)
        }
        return NumpyCompat.argsortQuick(s)
    }

    /// `_square_to_quad` (Heckbert) for one quad of corners q0..q3 → H (row-major).
    static func squareToQuad(_ q0: SIMD2<Double>, _ q1: SIMD2<Double>, _ q2: SIMD2<Double>,
                             _ q3: SIMD2<Double>) -> M9 {
        let x0 = q0.x, x1 = q1.x, x2 = q2.x, x3 = q3.x
        let y0 = q0.y, y1 = q1.y, y2 = q2.y, y3 = q3.y
        let dx1 = x1 - x2, dx2 = x3 - x2, dx3 = ((x0 - x1) + x2) - x3
        let dy1 = y1 - y2, dy2 = y3 - y2, dy3 = ((y0 - y1) + y2) - y3
        let den = dx1 * dy2 - dx2 * dy1
        let g = (dx3 * dy2 - dx2 * dy3) / den
        let h = (dx1 * dy3 - dx3 * dy1) / den
        return (((x1 - x0) + g * x1), ((x3 - x0) + h * x3), x0,
                ((y1 - y0) + g * y1), ((y3 - y0) + h * y3), y0,
                g, h, 1.0)
    }

    @inline(__always) static func allFinite(_ m: M9) -> Bool {
        m.0.isFinite && m.1.isFinite && m.2.isFinite && m.3.isFinite && m.4.isFinite
            && m.5.isFinite && m.6.isFinite && m.7.isFinite && m.8.isFinite
    }
}
