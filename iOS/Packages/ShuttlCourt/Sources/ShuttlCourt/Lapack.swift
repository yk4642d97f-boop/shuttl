/// numpy.linalg.eigh for 2x2 symmetric matrices, reproducing LAPACK's (OpenBLAS reference
/// dsyevd → dsteqr → dlaev2) eigenvector signs, which fix the orientation of fitted lines.
import Foundation

enum Lapack {
    /// dlaev2(a, b, c): eigen-decomposition of [[a, b], [b, c]] → (rt1, rt2, cs1, sn1) with
    /// |rt1| >= |rt2| and (cs1, sn1) the unit eigenvector of rt1.
    static func dlaev2(_ a: Double, _ b: Double, _ c: Double) -> (rt1: Double, rt2: Double, cs1: Double, sn1: Double) {
        let sm = a + c, df = a - c, adf = abs(df), tb = b + b, ab = abs(tb)
        let acmx: Double, acmn: Double
        if abs(a) > abs(c) { acmx = a; acmn = c } else { acmx = c; acmn = a }
        let rt: Double
        if adf > ab { rt = adf * (1 + (ab / adf) * (ab / adf)).squareRoot() }
        else if adf < ab { rt = ab * (1 + (adf / ab) * (adf / ab)).squareRoot() }
        else { rt = ab * 2.0.squareRoot() }
        let rt1: Double, rt2: Double, sgn1: Int
        if sm < 0 {
            rt1 = 0.5 * (sm - rt); sgn1 = -1
            rt2 = (acmx / rt1) * acmn - (b / rt1) * b
        } else if sm > 0 {
            rt1 = 0.5 * (sm + rt); sgn1 = 1
            rt2 = (acmx / rt1) * acmn - (b / rt1) * b
        } else {
            rt1 = 0.5 * rt; rt2 = -0.5 * rt; sgn1 = 1
        }
        let cs: Double, sgn2: Int
        if df >= 0 { cs = df + rt; sgn2 = 1 } else { cs = df - rt; sgn2 = -1 }
        var cs1: Double, sn1: Double
        if abs(cs) > ab {
            let ct = -tb / cs
            sn1 = 1 / (1 + ct * ct).squareRoot()
            cs1 = ct * sn1
        } else if ab == 0 {
            cs1 = 1; sn1 = 0
        } else {
            let tn = -cs / tb
            cs1 = 1 / (1 + tn * tn).squareRoot()
            sn1 = tn * cs1
        }
        if sgn1 == sgn2 {
            let tn = cs1
            cs1 = -sn1; sn1 = tn
        }
        return (rt1, rt2, cs1, sn1)
    }

    /// np.linalg.eigh([[a, b], [b, c]]) → (eigenvalues ascending, eigenvector columns v0, v1).
    static func eigh2(_ a: Double, _ b: Double, _ c: Double) -> (values: SIMD2<Double>, v0: SIMD2<Double>, v1: SIMD2<Double>) {
        let eps = Double.ulpOfOne * 0.5      // dlamch('E')
        var d = SIMD2(a, c)
        var z0 = SIMD2<Double>(1, 0), z1 = SIMD2<Double>(0, 1)   // columns of Z
        let e = b
        let split = e == 0 || abs(e) <= (abs(d.x).squareRoot() * abs(d.y).squareRoot()) * eps
        if !split {
            let r = dlaev2(d.x, e, d.y)
            // DLASR on Z = I with one rotation (c, s) → [[c, -s], [s, c]]
            z0 = SIMD2(r.cs1, r.sn1); z1 = SIMD2(-r.sn1, r.cs1)
            d = SIMD2(r.rt1, r.rt2)
        }
        if d.y < d.x {                          // selection sort (ascending)
            d = SIMD2(d.y, d.x)
            swap(&z0, &z1)
        }
        return (d, z0, z1)
    }
}
