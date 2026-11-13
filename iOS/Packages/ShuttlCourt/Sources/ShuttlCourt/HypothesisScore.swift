/// Hypothesis plausibility and model/mask agreement scoring.
/// Port of court_calibration._focal_from_homography (batched form), _plausible, _line_score.
/// Arithmetic mirrors numpy: BLAS matmuls are FMA chains in k order (verified), ufuncs
/// are unfused, float32 where numpy is float32, reductions use numpy pairwise summation.
import Foundation
import simd

enum HypothesisScore {
    /// numpy `A @ B` for 3x3 row-major (OpenBLAS dgemm: fma chain over k).
    @inline(__always) static func matmul(_ a: M9, _ b: M9) -> M9 {
        @inline(__always) func e(_ a0: Double, _ a1: Double, _ a2: Double,
                                 _ b0: Double, _ b1: Double, _ b2: Double) -> Double {
            fma(a2, b2, fma(a1, b1, a0 * b0))
        }
        return (e(a.0, a.1, a.2, b.0, b.3, b.6), e(a.0, a.1, a.2, b.1, b.4, b.7), e(a.0, a.1, a.2, b.2, b.5, b.8),
                e(a.3, a.4, a.5, b.0, b.3, b.6), e(a.3, a.4, a.5, b.1, b.4, b.7), e(a.3, a.4, a.5, b.2, b.5, b.8),
                e(a.6, a.7, a.8, b.0, b.3, b.6), e(a.6, a.7, a.8, b.1, b.4, b.7), e(a.6, a.7, a.8, b.2, b.5, b.8))
    }

    /// `_focal_from_homography` focal (NaN when w <= 0), numpy operation order.
    static func focal(_ H: M9, cx: Double, cy: Double, scale: Double) -> Double {
        let T: M9 = (1 / scale, 0, -cx / scale, 0, 1 / scale, -cy / scale, 0, 0, 1)
        let c = matmul(T, H)
        // np.linalg.norm over 9 → sqrt(pairwise sum of squares)
        let s8 = (((c.0 * c.0 + c.1 * c.1) + (c.2 * c.2 + c.3 * c.3)) + ((c.4 * c.4 + c.5 * c.5) + (c.6 * c.6 + c.7 * c.7)))
        let nrm = (s8 + c.8 * c.8).squareRoot()
        let h1 = SIMD3(c.0 / nrm, c.3 / nrm, c.6 / nrm), h2 = SIMD3(c.1 / nrm, c.4 / nrm, c.7 / nrm)
        return Homography.focalCore(h1: h1, h2: h2, scale: scale).f
    }

    /// `_plausible` for one hypothesis: a real camera above the floor?
    static func plausible(_ H: M9, ref: SIMD2<Double>, cx: Double, cy: Double, D: Double) -> Bool {
        let f = focal(H, cx: cx, cy: cy, scale: D)
        guard f.isFinite && f > 0.2 * D && f < 6 * D else { return false }
        let m1 = SIMD3((H.0 - cx * H.6) / f, (H.3 - cy * H.6) / f, H.6)
        let m2 = SIMD3((H.1 - cx * H.7) / f, (H.4 - cy * H.7) / f, H.7)
        let m3 = SIMD3((H.2 - cx * H.8) / f, (H.5 - cy * H.8) / f, H.8)
        let n1 = norm3(m1), n2 = norm3(m2)
        let ortho = abs(dot3(m1, m2)) / (n1 * n2 + 1e-18)
        let ratio = abs(n1 / (n2 + 1e-18) - 1)
        var lam = 2 / (n1 + n2 + 1e-18)
        let depth = m1.z * ref.x + m2.z * ref.y + m3.z
        lam = lam * (depth > 0 ? 1 : depth < 0 ? -1 : (depth == 0 ? 0 : Double.nan))    // np.sign
        let r1 = m1 * lam, r2 = m2 * lam, t = m3 * lam
        var r3 = cross3(r1, r2)
        r3 /= norm3(r3) + 1e-18
        let height = -dot3(r3, t)
        let a = dot3(r1, t), b = dot3(r2, t)
        let dist = (a * a + b * b + height * height).squareRoot()
        return ortho < 0.15 && ratio < 0.15 && height > 0.3 && height < 80 && dist < 150
    }

    /// `_line_score`: projected-length-weighted model/mask agreement (float32 like numpy).
    /// `tol`: good-sample DT tolerance (px); nil = max(2, 0.25 % D) as in `_line_score`.
    static func lineScore(_ feat: Features, _ Hs: [M9], _ samples: LineSamples, tol tolPx: Double? = nil) -> [Float] {
        let M = samples.xf.count
        let tol = tolPx ?? max(2.0, 0.0025 * feat.D)
        let goodT = Float(tol), badT = Float(3 * tol), cap = Float(0.05 * feat.D)
        let wl = Float(feat.w - 1), hl = Float(feat.h - 1), W = feat.w
        let same = (0..<max(M - 1, 0)).map { samples.ids[$0 + 1] == samples.ids[$0] }
        var u = [Float](repeating: 0, count: M), v = u, w = u, half = u, gs = u, bs = u
        var out = [Float](repeating: 0, count: Hs.count)
        feat.dt.data.withUnsafeBufferPointer { dt in
            for (k, H) in Hs.enumerated() {
                let e = [H.0, H.1, H.2, H.3, H.4, H.5, H.6, H.7, H.8]
                var mx = 0.0
                for x in e { mx = max(mx, abs(x)) }
                let h = e.map { Float($0 / mx) }
                for m in 0..<M {
                    let X = samples.xf[m], Y = samples.yf[m]
                    let p0 = (h[0] * X).addingProduct(h[1], Y) + h[2]
                    let p1 = (h[3] * X).addingProduct(h[4], Y) + h[5]
                    let p2 = (h[6] * X).addingProduct(h[7], Y) + h[8]
                    u[m] = p0 / p2; v[m] = p1 / p2; w[m] = p2
                }
                for j in 0..<(M - 1) {
                    var d = hypot(u[j + 1] - u[j], v[j + 1] - v[j])
                    if !(same[j] && d.isFinite) { d = 0 }
                    half[j] = 0.5 * d
                }
                for m in 0..<M {
                    var st: Float = m < M - 1 ? half[m] : 0
                    if m > 0 { st += half[m - 1] }
                    st = min(max(st, 0), cap)
                    let inside = w[m] > 0 && u[m] >= 0 && v[m] >= 0 && u[m] <= wl && v[m] <= hl
                    let d = inside ? dt[Int(v[m]) * W + Int(u[m])] : 0
                    gs[m] = inside && d <= goodT ? st : 0
                    bs[m] = inside && d > badT ? st : 0
                }
                let g = gs.withUnsafeBufferPointer { 0 + NumpyCompat.pairwiseSum($0, 0, M) }
                let b = bs.withUnsafeBufferPointer { 0 + NumpyCompat.pairwiseSum($0, 0, M) }
                out[k] = g - 0.5 * b
            }
        }
        return out
    }
}
