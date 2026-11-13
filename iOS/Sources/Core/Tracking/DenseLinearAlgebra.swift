import Foundation
import simd

// MARK: - DenseLinearAlgebra
// Tiny dense solvers for the handful of small systems the tracking stack needs
// (8×8 DLT, 7×7 Levenberg–Marquardt normal equations). Row-major storage.

enum DenseLinearAlgebra {

    /// Solves `A x = b` for square `A` (n×n, row-major) by Gaussian elimination
    /// with partial pivoting. Returns nil when `A` is numerically singular.
    static func solve(_ a: [Double], _ b: [Double], n: Int) -> [Double]? {
        precondition(a.count == n * n && b.count == n)
        var m = a
        var x = b
        for col in 0..<n {
            var pivot = col
            var best = abs(m[col * n + col])
            for r in (col + 1)..<max(col + 1, n) where abs(m[r * n + col]) > best {
                best = abs(m[r * n + col])
                pivot = r
            }
            guard best > 1e-14 else { return nil }
            if pivot != col {
                for c in 0..<n { m.swapAt(col * n + c, pivot * n + c) }
                x.swapAt(col, pivot)
            }
            let inv = 1.0 / m[col * n + col]
            for r in (col + 1)..<max(col + 1, n) {
                let f = m[r * n + col] * inv
                guard f != 0 else { continue }
                for c in col..<n { m[r * n + c] -= f * m[col * n + c] }
                x[r] -= f * x[col]
            }
        }
        for r in stride(from: n - 1, through: 0, by: -1) {
            var s = x[r]
            for c in (r + 1)..<max(r + 1, n) { s -= m[r * n + c] * x[c] }
            x[r] = s / m[r * n + r]
        }
        return x.allSatisfy(\.isFinite) ? x : nil
    }

    /// Least-squares solution of the over-determined system `A x ≈ b`
    /// (rows×n, row-major) via the normal equations.
    static func leastSquares(_ a: [Double], _ b: [Double], rows: Int, n: Int) -> [Double]? {
        var ata = [Double](repeating: 0, count: n * n)
        var atb = [Double](repeating: 0, count: n)
        for r in 0..<rows {
            for i in 0..<n {
                let ai = a[r * n + i]
                guard ai != 0 else { continue }
                atb[i] += ai * b[r]
                for j in 0..<n { ata[i * n + j] += ai * a[r * n + j] }
            }
        }
        return solve(ata, atb, n: n)
    }

    // MARK: Nonlinear least squares

    /// Minimises Σ residuals(p)² with Levenberg–Marquardt and a central-difference
    /// Jacobian. Meant for a handful of parameters (≤ ~10).
    static func levenbergMarquardt(_ start: [Double], residuals: ([Double]) -> [Double],
                                   maxIterations: Int = 60, step: Double = 1e-6) -> [Double] {
        let n = start.count
        var p = start
        var r = residuals(p)
        var cost = r.reduce(0) { $0 + $1 * $1 }
        var lambda = 1e-3
        for _ in 0..<maxIterations {
            guard cost > 1e-24 else { break }
            let m = r.count
            var jac = [Double](repeating: 0, count: m * n)
            for j in 0..<n {
                let hstep = step * max(1, abs(p[j]))
                var pp = p, pm = p
                pp[j] += hstep; pm[j] -= hstep
                let rp = residuals(pp), rm = residuals(pm)
                for i in 0..<m { jac[i * n + j] = (rp[i] - rm[i]) / (2 * hstep) }
            }
            var jtj = [Double](repeating: 0, count: n * n)
            var jtr = [Double](repeating: 0, count: n)
            for i in 0..<m {
                for a in 0..<n {
                    let ja = jac[i * n + a]
                    guard ja != 0 else { continue }
                    jtr[a] += ja * r[i]
                    for b in 0..<n { jtj[a * n + b] += ja * jac[i * n + b] }
                }
            }
            var improved = false
            for _ in 0..<12 {
                var damped = jtj
                for d in 0..<n { damped[d * n + d] += lambda * max(jtj[d * n + d], 1e-12) }
                guard let delta = solve(damped, jtr.map { -$0 }, n: n) else { lambda *= 10; continue }
                let cand = zip(p, delta).map { $0 + $1 }
                let rc = residuals(cand)
                let cc = rc.reduce(0) { $0 + $1 * $1 }
                if cc < cost {
                    let gain = cost - cc
                    p = cand; r = rc; cost = cc
                    lambda = max(lambda / 10, 1e-12)
                    improved = true
                    if gain < 1e-15 * max(cost, 1e-30) { return p }
                    break
                }
                lambda *= 10
            }
            if !improved { break }
        }
        return p
    }

    // MARK: Rotations

    /// Rodrigues: axis-angle vector → rotation matrix.
    static func rotation(from w: SIMD3<Double>) -> simd_double3x3 {
        let theta = simd_length(w)
        let wx = skew(w)
        if theta < 1e-12 { return matrix_identity_double3x3 + wx }
        let k = skew(w / theta)
        return matrix_identity_double3x3 + sin(theta) * k + (1 - cos(theta)) * (k * k)
    }

    static func skew(_ v: SIMD3<Double>) -> simd_double3x3 {
        simd_double3x3(rows: [
            SIMD3(0, -v.z, v.y),
            SIMD3(v.z, 0, -v.x),
            SIMD3(-v.y, v.x, 0),
        ])
    }

    /// Nearest proper rotation to `m` (polar decomposition by Newton iteration).
    static func orthonormalized(_ m: simd_double3x3) -> simd_double3x3 {
        var r = m
        for _ in 0..<20 {
            let next = 0.5 * (r + r.inverse.transpose)
            let delta = next - r
            r = next
            let err = simd_length(delta.columns.0) + simd_length(delta.columns.1) + simd_length(delta.columns.2)
            if err < 1e-13 { break }
        }
        return r
    }
}

extension simd_double3x3 {
    /// Row-major nested arrays, matching numpy's `ndarray.tolist()`.
    var rowArrays: [[Double]] {
        (0..<3).map { r in (0..<3).map { c in self[c, r] } }
    }

    init?(rowArrays rows: [[Double]]) {
        guard rows.count == 3, rows.allSatisfy({ $0.count == 3 }) else { return nil }
        self.init(rows: rows.map { SIMD3($0[0], $0[1], $0[2]) })
    }
}
