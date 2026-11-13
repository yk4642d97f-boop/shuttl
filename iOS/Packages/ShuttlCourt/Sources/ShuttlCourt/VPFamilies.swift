/// Vanishing-point line families. Port of court_calibration._vp_inliers / _vp_families.
///
/// Deviation: numpy takes the family VP from LAPACK dgesdd, whose sign is arbitrary
/// (no simple rule reproduces it, and Accelerate's dgesdd differs from OpenBLAS's in
/// ~10 % of cases). Memberships never depend on the sign (|sin| test); only the ordering
/// direction in `_order_family` does, which yields the same hypotheses (transverse) or
/// their 180°-twin set (longitudinal). Here the sign is fixed deterministically:
/// largest-magnitude component positive.
import Foundation
import simd
import ShuttlCore

/// One family: homogeneous VP and member indices into the merged line list (ascending).
struct VPFamily: Sendable {
    var v: SIMD3<Double>
    var members: [Int]
}

enum VanishingPoints {
    /// np.linalg.norm of a 1-D 2-vector (OpenBLAS ddot: FMA chain).
    @inline(__always) static func blasNorm(_ d: SIMD2<Double>) -> Double {
        fma(d.y, d.y, d.x * d.x).squareRoot()
    }

    /// `_vp_inliers` for one VP over all lines.
    static func inliers(_ v: SIMD3<Double>, _ mids: [SIMD2<Double>], _ dirs: [SIMD2<Double>],
                        _ tol: [Double]) -> [Bool] {
        (0..<mids.count).map { k in
            let ux = v.x - v.z * mids[k].x
            let uy = v.y - v.z * mids[k].y
            let nu = hypot(ux, uy) + 1e-12
            let s = abs(dirs[k].x * uy - dirs[k].y * ux) / nu
            return s < tol[k]
        }
    }

    /// Smallest right singular vector of A (rows 3-vectors), sign-normalised.
    static func nullVector(_ rows: [SIMD3<Double>]) -> SIMD3<Double> {
        var m = Matrix(rows: rows.count, cols: 3)
        for (i, r) in rows.enumerated() { m[i, 0] = r.x; m[i, 1] = r.y; m[i, 2] = r.z }
        let vt = LinAlg.svd(m).vt
        var v = SIMD3(vt[2, 0], vt[2, 1], vt[2, 2])
        let a = abs(v)
        let big = a.x >= a.y && a.x >= a.z ? v.x : (a.y >= a.z ? v.y : v.z)
        if big < 0 { v = -v }
        return v
    }

    /// `_vp_families`: greedy exhaustive pair hypotheses, up to `maxFam` families.
    static func families(_ lines: [ImageLine], maxFam: Int = 3) -> [VPFamily] {
        var fams: [VPFamily] = []
        var remaining = Array(lines.indices)
        while remaining.count >= 2 && fams.count < maxFam {
            let R = remaining.map { lines[$0] }
            let Lh = R.map(\.l)
            let mids = R.map(\.mid)
            let dirs = R.map { ln -> SIMD2<Double> in let d = ln.p1 - ln.p0; return d / (blasNorm(d) + 1e-12) }
            let tol = R.map { min(max(2.5 / blasNorm($0.p1 - $0.p0), 0.003), 0.03) }
            let wts = R.map(\.weight)
            let n = R.count
            var bestScore = -Double.infinity
            var bestInl: [Bool] = []
            for i in 0..<n {                                    // np.triu_indices order
                for j in (i + 1)..<n {
                    var V = cross3(Lh[i], Lh[j])
                    V /= (V.x * V.x + V.y * V.y + V.z * V.z).squareRoot() + 1e-18
                    let inl = inliers(V, mids, dirs, tol)
                    var score = 0.0
                    for k in 0..<n where inl[k] { score += wts[k] }
                    if score > bestScore { bestScore = score; bestInl = inl }   // argmax: first max
                }
            }
            let members = (0..<n).filter { bestInl[$0] }
            if members.count < 2 { break }
            let A = members.map { Lh[$0] * wts[$0].squareRoot() }
            let v = nullVector(A)
            let inl2 = inliers(v, mids, dirs, tol)
            let mem2 = (0..<n).filter { inl2[$0] || bestInl[$0] }
            fams.append(VPFamily(v: v, members: mem2.map { remaining[$0] }))
            let taken = Set(mem2)
            remaining = (0..<n).filter { !taken.contains($0) }.map { remaining[$0] }
        }
        return fams
    }
}
