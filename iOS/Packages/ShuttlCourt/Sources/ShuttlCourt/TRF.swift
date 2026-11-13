/// scipy.optimize.least_squares(method="trf", x_scale="jac", jac="2-point", tr_solver="exact",
/// ftol = xtol = gtol = 1e-8, no bounds): port of scipy 1.14 _lsq/trf.py `trf_no_bounds`
/// and the common.py / _numdiff.py helpers it uses, so iterates AND the early stop follow
/// scipy (running LM to full convergence differed by up to 6e-4 px on real images).
/// Singular-vector signs cancel in every product used, so any SVD gives the same step.
import Foundation
import ShuttlCore

enum TRF {
    static let EPS = Double.ulpOfOne

    /// 0.5 Σ ρ(f²) — `loss_function(f, cost_only=True)` / `0.5 * dot(f, f)`.
    static func cost(_ f: [Double], _ loss: LeastSquares.Loss) -> Double {
        switch loss {
        case .linear:
            var s = 0.0
            for v in f { s = fma(v, v, s) }
            return 0.5 * s
        case .softL1:
            let r0 = f.map { 2 * ((1 + $0 * $0).squareRoot() - 1) }
            return 0.5 * NumpyCompat.sum(r0)
        }
    }

    /// `approx_derivative(fun, x, method="2-point", f0=f)` → row-major m×n.
    static func jacobian(_ x: [Double], _ f0: [Double], _ fun: ([Double]) -> [Double]) -> [Double] {
        let n = x.count, m = f0.count
        let rstep = EPS.squareRoot()
        var J = [Double](repeating: 0, count: m * n)
        var x1 = x
        for i in 0..<n {
            let sgn: Double = x[i] >= 0 ? 1 : -1
            let h = rstep * sgn * max(1.0, abs(x[i]))        // (x0+h)-x0 == 0 fallback: same value
            x1[i] += h
            let dx = x1[i] - x[i]
            let f1 = fun(x1)
            for k in 0..<m { J[k * n + i] = (f1[k] - f0[k]) / dx }
            x1[i] = x[i]
        }
        return J
    }

    /// `scale_for_robust_loss_function` for soft_l1 (f_scale = 1), in place.
    static func robustScale(_ J: inout [Double], _ f: inout [Double], n: Int) {
        for i in 0..<f.count {
            let z = f[i] * f[i], t = 1 + z
            let r1 = pow(t, -0.5), r2 = -0.5 * pow(t, -1.5)
            var js = r1 + 2 * r2 * z
            if js < EPS { js = EPS }
            js = js.squareRoot()
            f[i] *= r1 / js
            for j in 0..<n { J[i * n + j] *= js }
        }
    }

    /// `compute_jac_scale` → scale_inv (column norms; zeros → 1, or max with the old ones).
    static func jacScaleInv(_ J: [Double], m: Int, n: Int, old: [Double]?) -> [Double] {
        var s = [Double](repeating: 0, count: n)
        for i in 0..<m { for j in 0..<n { s[j] += J[i * n + j] * J[i * n + j] } }   // axis-0 sum
        s = s.map { $0.squareRoot() }
        if let old { return zip(s, old).map { max($0, $1) } }
        return s.map { $0 == 0 ? 1 : $0 }
    }

    @inline(__always) static func norm(_ v: [Double]) -> Double {
        var s = 0.0
        for x in v { s = fma(x, x, s) }
        return s.squareRoot()
    }

    /// `solve_lsq_trust_region` (More's LM root-finding over the SVD of J_h).
    static func trustRegion(n: Int, m: Int, uf: [Double], s: [Double], vt: Matrix, Delta: Double,
                            initialAlpha: Double) -> (p: [Double], alpha: Double) {
        func vdot(_ w: [Double]) -> [Double] {                 // V.dot(w), V = vt.T
            (0..<n).map { r in var a = 0.0; for c in 0..<n { a = fma(vt[c, r], w[c], a) }; return a }
        }
        let suf = (0..<n).map { s[$0] * uf[$0] }
        func phi(_ alpha: Double) -> (Double, Double) {
            let denom = s.map { $0 * $0 + alpha }
            let pn = norm((0..<n).map { suf[$0] / denom[$0] })
            let dsum = NumpyCompat.sum((0..<n).map { suf[$0] * suf[$0] / pow(denom[$0], 3) })
            return (pn - Delta, -dsum / pn)
        }
        let fullRank = m >= n ? s[n - 1] > EPS * Double(m) * s[0] : false
        if fullRank {
            let p = vdot((0..<n).map { uf[$0] / s[$0] }).map { -$0 }
            if norm(p) <= Delta { return (p, 0) }
        }
        var alphaUpper = norm(suf) / Delta
        var alphaLower = 0.0
        if fullRank { let (ph, pp) = phi(0); alphaLower = -ph / pp }
        var alpha = (!fullRank && initialAlpha == 0)
            ? max(0.001 * alphaUpper, (alphaLower * alphaUpper).squareRoot()) : initialAlpha
        for _ in 0..<10 {
            if alpha < alphaLower || alpha > alphaUpper {
                alpha = max(0.001 * alphaUpper, (alphaLower * alphaUpper).squareRoot())
            }
            let (ph, pp) = phi(alpha)
            if ph < 0 { alphaUpper = alpha }
            let ratio = ph / pp
            alphaLower = max(alphaLower, alpha - ratio)
            alpha -= (ph + Delta) * ratio / Delta
            if abs(ph) < 0.01 * Delta { break }
        }
        var p = vdot((0..<n).map { suf[$0] / (s[$0] * s[$0] + alpha) }).map { -$0 }
        let k = Delta / norm(p)
        p = p.map { $0 * k }
        return (p, alpha)
    }

    /// `least_squares(fun, x0, loss, max_nfev, x_scale="jac")` → x.
    static func solve(_ x0: [Double], loss: LeastSquares.Loss, maxNfev: Int,
                      ftol: Double = 1e-8, xtol: Double = 1e-8, gtol: Double = 1e-8,
                      residuals fun: ([Double]) -> [Double]) -> [Double] {
        let n = x0.count
        var x = x0
        var f = fun(x)
        let m = f.count
        guard m > 0, f.allSatisfy(\.isFinite) else { return x0 }    // scipy raises here
        var nfev = 1
        var cost = TRF.cost(f, loss)
        var J = jacobian(x, f, fun)
        if loss == .softL1 { robustScale(&J, &f, n: n) }
        func grad() -> [Double] {
            (0..<n).map { j in var a = 0.0; for i in 0..<m { a = fma(J[i * n + j], f[i], a) }; return a }
        }
        var g = grad()
        var scaleInv = jacScaleInv(J, m: m, n: n, old: nil)
        var scale = scaleInv.map { 1 / $0 }
        var Delta = norm((0..<n).map { x0[$0] * scaleInv[$0] })
        if Delta == 0 { Delta = 1 }
        var alpha = 0.0
        var status: Int? = nil
        while true {
            if (g.map(abs).max() ?? 0) < gtol { status = 1 }
            if status != nil || nfev == maxNfev { break }
            let d = scale
            let gh = (0..<n).map { d[$0] * g[$0] }
            var Jh = Matrix(rows: m, cols: n)
            for i in 0..<m { for j in 0..<n { Jh[i, j] = J[i * n + j] * d[j] } }
            let (U, s, vt) = LinAlg.svd(Jh)
            let uf = (0..<n).map { j in var a = 0.0; for i in 0..<m { a = fma(U[i, j], f[i], a) }; return a }
            var actual = -1.0
            var xNew = x, fNew = f
            while actual <= 0 && nfev < maxNfev {
                let (sh, a) = trustRegion(n: n, m: m, uf: uf, s: s, vt: vt, Delta: Delta, initialAlpha: alpha)
                alpha = a
                var q = 0.0
                for i in 0..<m {
                    var js = 0.0
                    for j in 0..<n { js = fma(Jh[i, j], sh[j], js) }
                    q = fma(js, js, q)
                }
                var l = 0.0
                for j in 0..<n { l = fma(sh[j], gh[j], l) }
                let predicted = -(0.5 * q + l)
                let step = (0..<n).map { d[$0] * sh[$0] }
                xNew = (0..<n).map { x[$0] + step[$0] }
                fNew = fun(xNew)
                nfev += 1
                let shNorm = norm(sh)
                if !fNew.allSatisfy(\.isFinite) { Delta = 0.25 * shNorm; continue }
                let costNew = TRF.cost(fNew, loss)
                actual = cost - costNew
                // update_tr_radius
                let ratio = predicted > 0 ? actual / predicted : (predicted == 0 && actual == 0 ? 1 : 0)
                var DeltaNew = Delta
                if ratio < 0.25 { DeltaNew = 0.25 * shNorm } else if ratio > 0.75 && shNorm > 0.95 * Delta { DeltaNew *= 2 }
                // check_termination
                let ftolOK = actual < ftol * cost && ratio > 0.25
                let xtolOK = norm(step) < xtol * (xtol + norm(x))
                if ftolOK || xtolOK { status = ftolOK && xtolOK ? 4 : (ftolOK ? 2 : 3); break }
                alpha *= Delta / DeltaNew
                Delta = DeltaNew
            }
            if actual > 0 {
                x = xNew; f = fNew; cost = TRF.cost(f, loss)
                J = jacobian(x, f, fun)
                if loss == .softL1 { robustScale(&J, &f, n: n) }
                g = grad()
                scaleInv = jacScaleInv(J, m: m, n: n, old: scaleInv)
                scale = scaleInv.map { 1 / $0 }
            }
        }
        return x
    }
}
