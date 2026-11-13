import Foundation
import simd

/// Dense row-major Double matrix — the numpy.ndarray stand-in for the small
/// systems in the tracking stack (DLT, LM normal equations, polynomial fits).
public struct Matrix: Sendable, Equatable, CustomStringConvertible {
    public let rows: Int
    public let cols: Int
    public var data: [Double]

    public init(rows: Int, cols: Int, repeating value: Double = 0) {
        self.rows = rows; self.cols = cols
        data = Array(repeating: value, count: rows * cols)
    }

    public init(rows: Int, cols: Int, data: [Double]) {
        precondition(data.count == rows * cols, "Matrix data size mismatch")
        self.rows = rows; self.cols = cols; self.data = data
    }

    /// From nested rows, e.g. `Matrix([[1, 2], [3, 4]])`.
    public init(_ nested: [[Double]]) {
        let c = nested.first?.count ?? 0
        precondition(nested.allSatisfy { $0.count == c }, "ragged rows")
        rows = nested.count
        cols = c
        data = nested.flatMap { $0 }
    }

    public static func identity(_ n: Int) -> Matrix {
        var m = Matrix(rows: n, cols: n)
        for i in 0..<n { m[i, i] = 1 }
        return m
    }

    public subscript(r: Int, c: Int) -> Double {
        get { data[r * cols + c] }
        set { data[r * cols + c] = newValue }
    }

    public func row(_ r: Int) -> [Double] { Array(data[(r * cols)..<((r + 1) * cols)]) }
    public func column(_ c: Int) -> [Double] { (0..<rows).map { data[$0 * cols + c] } }
    public var nested: [[Double]] { (0..<rows).map(row) }

    public var transposed: Matrix {
        var t = Matrix(rows: cols, cols: rows)
        for r in 0..<rows { for c in 0..<cols { t.data[c * rows + r] = data[r * cols + c] } }
        return t
    }

    public var description: String { nested.map { $0.map { String(format: "%.6g", $0) }.joined(separator: " ") }.joined(separator: "\n") }

    public static func * (a: Matrix, b: Matrix) -> Matrix {
        precondition(a.cols == b.rows, "Matrix * shape mismatch")
        var out = Matrix(rows: a.rows, cols: b.cols)
        for i in 0..<a.rows {
            for k in 0..<a.cols {
                let aik = a.data[i * a.cols + k]
                if aik == 0 { continue }
                for j in 0..<b.cols { out.data[i * b.cols + j] += aik * b.data[k * b.cols + j] }
            }
        }
        return out
    }

    public static func * (a: Matrix, x: [Double]) -> [Double] {
        precondition(a.cols == x.count, "Matrix * vector shape mismatch")
        return (0..<a.rows).map { r in
            var s = 0.0
            for c in 0..<a.cols { s += a.data[r * a.cols + c] * x[c] }
            return s
        }
    }

    public static func * (s: Double, a: Matrix) -> Matrix { Matrix(rows: a.rows, cols: a.cols, data: a.data.map { $0 * s }) }
    public static func + (a: Matrix, b: Matrix) -> Matrix { Matrix(rows: a.rows, cols: a.cols, data: zip(a.data, b.data).map(+)) }
    public static func - (a: Matrix, b: Matrix) -> Matrix { Matrix(rows: a.rows, cols: a.cols, data: zip(a.data, b.data).map(-)) }

    /// AᵀA without forming Aᵀ.
    public var gram: Matrix {
        var g = Matrix(rows: cols, cols: cols)
        for r in 0..<rows {
            let base = r * cols
            for i in 0..<cols {
                let ai = data[base + i]
                if ai == 0 { continue }
                for j in i..<cols { g.data[i * cols + j] += ai * data[base + j] }
            }
        }
        for i in 0..<cols { for j in 0..<i { g.data[i * cols + j] = g.data[j * cols + i] } }
        return g
    }
}

// MARK: - simd bridges

public extension Matrix {
    init(_ m: simd_double3x3) {
        self.init([[m[0, 0], m[1, 0], m[2, 0]], [m[0, 1], m[1, 1], m[2, 1]], [m[0, 2], m[1, 2], m[2, 2]]])
    }

    /// Requires a 3×3 matrix.
    var simd3x3: simd_double3x3 {
        precondition(rows == 3 && cols == 3)
        return simd_double3x3(rows: [SIMD3(data[0], data[1], data[2]), SIMD3(data[3], data[4], data[5]), SIMD3(data[6], data[7], data[8])])
    }
}

public extension simd_double3x3 {
    /// Row-major nested arrays (numpy `.tolist()` layout).
    var rowMajor: [[Double]] { (0..<3).map { r in (0..<3).map { c in self[c, r] } } }

    init(rowMajor rows: [[Double]]) {
        precondition(rows.count == 3 && rows.allSatisfy { $0.count == 3 }, "expected 3x3")
        self.init(rows: rows.map { SIMD3($0[0], $0[1], $0[2]) })
    }
}

// MARK: - Solvers

/// numpy.linalg equivalents. All return nil on (numerical) singularity instead of throwing.
public enum LinAlg {

    /// Solves A x = b (A square) by LU with partial pivoting.
    public static func solve(_ a: Matrix, _ b: [Double]) -> [Double]? {
        solve(a, Matrix(rows: b.count, cols: 1, data: b))?.data
    }

    /// Solves A X = B (A square) by LU with partial pivoting.
    public static func solve(_ a: Matrix, _ b: Matrix) -> Matrix? {
        precondition(a.rows == a.cols && b.rows == a.rows, "solve shape mismatch")
        let n = a.rows, m = b.cols
        var lu = a.data, x = b.data
        let scale = max(a.data.map(abs).max() ?? 0, 1e-300)
        for col in 0..<n {
            var pivot = col
            var best = abs(lu[col * n + col])
            for r in (col + 1)..<max(col + 1, n) where abs(lu[r * n + col]) > best {
                best = abs(lu[r * n + col]); pivot = r
            }
            guard best > 1e-13 * scale else { return nil }
            if pivot != col {
                for c in 0..<n { lu.swapAt(col * n + c, pivot * n + c) }
                for c in 0..<m { x.swapAt(col * m + c, pivot * m + c) }
            }
            let inv = 1 / lu[col * n + col]
            for r in (col + 1)..<max(col + 1, n) {
                let f = lu[r * n + col] * inv
                if f == 0 { continue }
                for c in col..<n { lu[r * n + c] -= f * lu[col * n + c] }
                for c in 0..<m { x[r * m + c] -= f * x[col * m + c] }
            }
        }
        for r in stride(from: n - 1, through: 0, by: -1) {
            for c in 0..<m {
                var s = x[r * m + c]
                for k in (r + 1)..<max(r + 1, n) { s -= lu[r * n + k] * x[k * m + c] }
                x[r * m + c] = s / lu[r * n + r]
            }
        }
        return Matrix(rows: n, cols: m, data: x)
    }

    public static func inverse(_ a: Matrix) -> Matrix? { solve(a, .identity(a.rows)) }

    /// Cholesky factor L (lower) with A = L Lᵀ; nil unless A is symmetric positive definite.
    public static func cholesky(_ a: Matrix) -> Matrix? {
        precondition(a.rows == a.cols)
        let n = a.rows
        var l = Matrix(rows: n, cols: n)
        for j in 0..<n {
            var d = a[j, j]
            for k in 0..<j { d -= l[j, k] * l[j, k] }
            guard d > 0, d.isFinite else { return nil }
            let ljj = d.squareRoot()
            l[j, j] = ljj
            for i in (j + 1)..<max(j + 1, n) {
                var s = a[i, j]
                for k in 0..<j { s -= l[i, k] * l[j, k] }
                l[i, j] = s / ljj
            }
        }
        return l
    }

    /// Least squares like `numpy.linalg.lstsq(A, b, rcond=None)`: minimum-norm solution,
    /// singular values below eps·max(m, n)·σmax treated as zero.
    public static func lstsq(_ a: Matrix, _ b: [Double]) -> [Double]? {
        lstsq(a, Matrix(rows: b.count, cols: 1, data: b))?.data
    }

    public static func lstsq(_ a: Matrix, _ b: Matrix) -> Matrix? {
        precondition(a.rows == b.rows, "lstsq shape mismatch")
        guard a.rows > 0, a.cols > 0 else { return nil }
        let (u, s, vt) = svd(a)
        let cutoff = Double.ulpOfOne * Double(max(a.rows, a.cols)) * (s.first ?? 0)
        let k = s.count
        // x = V Σ⁺ Uᵀ b
        var tmp = Matrix(rows: k, cols: b.cols)
        for i in 0..<k where s[i] > cutoff {
            for c in 0..<b.cols {
                var dot = 0.0
                for r in 0..<a.rows { dot += u[r, i] * b[r, c] }
                tmp[i, c] = dot / s[i]
            }
        }
        var x = Matrix(rows: a.cols, cols: b.cols)
        for j in 0..<a.cols {
            for c in 0..<b.cols {
                var acc = 0.0
                for i in 0..<k { acc += vt[i, j] * tmp[i, c] }
                x[j, c] = acc
            }
        }
        return x.data.allSatisfy(\.isFinite) ? x : nil
    }

    /// Symmetric eigen-decomposition like `numpy.linalg.eigh` (cyclic Jacobi):
    /// eigenvalues ascending, eigenvectors as the matching COLUMNS of `vectors`.
    public static func symmetricEigen(_ a: Matrix) -> (values: [Double], vectors: Matrix) {
        precondition(a.rows == a.cols)
        let n = a.rows
        var m = a
        var v = Matrix.identity(n)
        for _ in 0..<100 {
            var off = 0.0
            for i in 0..<n { for j in (i + 1)..<max(i + 1, n) { off += m[i, j] * m[i, j] } }
            if off < 1e-30 { break }
            for p in 0..<n {
                for q in (p + 1)..<max(p + 1, n) where abs(m[p, q]) > 1e-300 {
                    let theta = (m[q, q] - m[p, p]) / (2 * m[p, q])
                    let t = (theta >= 0 ? 1.0 : -1.0) / (abs(theta) + (theta * theta + 1).squareRoot())
                    let c = 1 / (t * t + 1).squareRoot(), s = t * c
                    for k in 0..<n {
                        let mkp = m[k, p], mkq = m[k, q]
                        m[k, p] = c * mkp - s * mkq
                        m[k, q] = s * mkp + c * mkq
                    }
                    for k in 0..<n {
                        let mpk = m[p, k], mqk = m[q, k]
                        m[p, k] = c * mpk - s * mqk
                        m[q, k] = s * mpk + c * mqk
                    }
                    for k in 0..<n {
                        let vkp = v[k, p], vkq = v[k, q]
                        v[k, p] = c * vkp - s * vkq
                        v[k, q] = s * vkp + c * vkq
                    }
                }
            }
        }
        let order = (0..<n).sorted { m[$0, $0] < m[$1, $1] }
        var vecs = Matrix(rows: n, cols: n)
        for (dst, src) in order.enumerated() { for k in 0..<n { vecs[k, dst] = v[k, src] } }
        return (order.map { m[$0, $0] }, vecs)
    }

    /// Eigenvalues only (ascending), like `numpy.linalg.eigvalsh`.
    public static func eigvalsh(_ a: Matrix) -> [Double] { symmetricEigen(a).values }

    /// SVD by one-sided Jacobi. For an m×n matrix returns U (m×k), singular values
    /// (k, descending) and Vᵀ (n×n), k = n. When m < n the matrix is zero-padded so Vᵀ
    /// is still the full basis — its last row is the null vector of a DLT system.
    public static func svd(_ a: Matrix) -> (u: Matrix, s: [Double], vt: Matrix) {
        let m = a.rows, n = a.cols
        let rows = max(m, n)
        var w = Matrix(rows: rows, cols: n)               // columns get orthogonalised
        for r in 0..<m { for c in 0..<n { w[r, c] = a[r, c] } }
        var v = Matrix.identity(n)
        for _ in 0..<80 {
            var rotated = false
            for i in 0..<n {
                for j in (i + 1)..<max(i + 1, n) {
                    var alpha = 0.0, beta = 0.0, gamma = 0.0
                    for r in 0..<rows {
                        let wi = w[r, i], wj = w[r, j]
                        alpha += wi * wi; beta += wj * wj; gamma += wi * wj
                    }
                    guard abs(gamma) > 1e-15 * (alpha * beta).squareRoot(), gamma != 0 else { continue }
                    rotated = true
                    let zeta = (beta - alpha) / (2 * gamma)
                    let t = (zeta >= 0 ? 1.0 : -1.0) / (abs(zeta) + (1 + zeta * zeta).squareRoot())
                    let c = 1 / (1 + t * t).squareRoot(), s = c * t
                    for r in 0..<rows {
                        let wi = w[r, i], wj = w[r, j]
                        w[r, i] = c * wi - s * wj
                        w[r, j] = s * wi + c * wj
                    }
                    for r in 0..<n {
                        let vi = v[r, i], vj = v[r, j]
                        v[r, i] = c * vi - s * vj
                        v[r, j] = s * vi + c * vj
                    }
                }
            }
            if !rotated { break }
        }
        let norms = (0..<n).map { c in (0..<rows).reduce(0.0) { $0 + w[$1, c] * w[$1, c] }.squareRoot() }
        let order = (0..<n).sorted { norms[$0] > norms[$1] }
        var u = Matrix(rows: m, cols: n)
        var vt = Matrix(rows: n, cols: n)
        for (dst, src) in order.enumerated() {
            let sigma = norms[src]
            for r in 0..<m { u[r, dst] = sigma > 1e-300 ? w[r, src] / sigma : 0 }
            for c in 0..<n { vt[dst, c] = v[c, src] }
        }
        return (u, order.map { norms[$0] }, vt)
    }

    /// Unit vector x minimising |A x| (last right-singular vector).
    public static func nullVector(_ a: Matrix) -> [Double] {
        let vt = svd(a).vt
        return vt.row(vt.rows - 1)
    }
}
