/// Small dense non-linear least squares (Levenberg–Marquardt) standing in for
/// scipy.optimize.least_squares in court_calibration.py:
///   • `_refine_on_points`: method="lm" (MINPACK), linear loss;
///   • `_refine`: method="trf", loss="soft_l1", f_scale=1, x_scale="jac".
/// Both are run to convergence of the SAME objective, so the optimum matches scipy's
/// to within scipy's own termination tolerances (ftol = xtol = gtol = 1e-8); the
/// iteration path (trust region vs. damping) is not replicated.
import Foundation
import ShuttlCore

enum LeastSquares {
    enum Loss: Sendable { case linear, softL1 }

    /// 0.5 * Σ ρ(r²) — scipy's `cost`.
    static func cost(_ r: [Double], _ loss: Loss) -> Double {
        var c = 0.0
        switch loss {
        case .linear: for v in r { c += v * v }
        case .softL1: for v in r { c += 2 * ((1 + v * v).squareRoot() - 1) }
        }
        return 0.5 * c
    }

    /// Minimise 0.5 Σ ρ(f_i(p)²). Jacobian by central differences.
    static func minimize(_ p0: [Double], loss: Loss, maxIterations: Int = 100,
                         residuals: ([Double]) -> [Double]) -> [Double] {
        let n = p0.count
        var p = p0
        var r = residuals(p)
        let m = r.count
        guard m > 0, r.allSatisfy(\.isFinite) else { return p }
        var c = cost(r, loss)
        var mu = -1.0
        var nu = 2.0
        var J = [Double](repeating: 0, count: m * n)
        for _ in 0..<maxIterations {
            // numerical Jacobian of the raw residuals
            for j in 0..<n {
                let h = 1e-6 * max(1, abs(p[j]))
                var pp = p, pm = p
                pp[j] += h; pm[j] -= h
                let rp = residuals(pp), rm = residuals(pm)
                let inv = 1 / (pp[j] - pm[j])
                for i in 0..<m { J[i * n + j] = (rp[i] - rm[i]) * inv }
            }
            // robust scaling (scipy scale_for_robust_loss_function)
            var fs = r
            var Js = J
            if loss == .softL1 {
                for i in 0..<m {
                    let z = r[i] * r[i]
                    let rho1 = 1 / (1 + z).squareRoot()
                    var js = rho1 / (1 + z)                 // rho' + 2 rho'' z = (1+z)^-1.5
                    if js < Double.ulpOfOne { js = Double.ulpOfOne }
                    js = js.squareRoot()
                    fs[i] = r[i] * rho1 / js
                    for j in 0..<n { Js[i * n + j] *= js }
                }
            }
            // normal equations
            var A = [Double](repeating: 0, count: n * n)
            var g = [Double](repeating: 0, count: n)
            for i in 0..<m {
                let base = i * n
                let fi = fs[i]
                for a in 0..<n {
                    let ja = Js[base + a]
                    if ja == 0 { continue }
                    g[a] += ja * fi
                    for b in a..<n { A[a * n + b] += ja * Js[base + b] }
                }
            }
            for a in 0..<n { for b in 0..<a { A[a * n + b] = A[b * n + a] } }
            let gmax = g.map(abs).max() ?? 0
            if gmax < 1e-14 * max(1, c) { break }
            if mu < 0 { mu = 1e-3 * ((0..<n).map { A[$0 * n + $0] }.max() ?? 1) }
            var improved = false
            var converged = false
            for _ in 0..<40 {
                var Ad = A
                for a in 0..<n { Ad[a * n + a] += mu * max(A[a * n + a], 1e-12) }
                guard let step = LinAlg.solve(Matrix(rows: n, cols: n, data: Ad), g.map { -$0 }) else {
                    mu *= nu; nu *= 2; continue
                }
                var pn = p
                for j in 0..<n { pn[j] += step[j] }
                let rn = residuals(pn)
                let cn = rn.allSatisfy(\.isFinite) ? cost(rn, loss) : .infinity
                // gain ratio
                var pred = 0.0
                for j in 0..<n { pred += step[j] * (mu * max(A[j * n + j], 1e-12) * step[j] - g[j]) }
                pred *= 0.5
                if cn < c {
                    let rho = pred > 0 ? (c - cn) / pred : 1
                    mu *= max(1.0 / 3.0, 1 - pow(2 * rho - 1, 3)); nu = 2
                    let stepNorm = step.reduce(0) { $0 + $1 * $1 }.squareRoot()
                    let pNorm = p.reduce(0) { $0 + $1 * $1 }.squareRoot()
                    converged = (c - cn) <= 1e-15 * c || stepNorm <= 1e-13 * (pNorm + 1e-13)
                    p = pn; r = rn; c = cn
                    improved = true
                    break
                }
                mu *= nu; nu *= 2
                if !mu.isFinite || mu > 1e30 { break }
            }
            if !improved || converged { break }
        }
        return p
    }
}
