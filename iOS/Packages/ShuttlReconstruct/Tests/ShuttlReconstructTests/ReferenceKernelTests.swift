import XCTest
@testable import ShuttlReconstruct

/// Verbatim scalar port of the numba `_rhs` / `_rk4` kernels (column-major (3, 8) state,
/// element (r, c) at c*3 + r) — the reference the production SIMD8 kernel must match
/// BIT FOR BIT (NaN in both counts as equal).
enum ScalarReference {
    static func rhs(_ X: [Double], _ L: Double, _ ncol: Int) -> [Double] {
        var out = [Double](repeating: 0, count: 24)
        let v0 = X[0], v1 = X[1], v2 = X[2]
        let sp = (v0 * v0 + v1 * v1 + v2 * v2).squareRoot()
        let k = sp / L
        out[0] = -k * v0; out[1] = -k * v1; out[2] = -R3.g - k * v2
        let inv = sp > 1e-9 ? 1.0 / sp : 0.0
        for c in 1..<max(ncol, 1) {
            let vs = (v0 * X[3 * c] + v1 * X[3 * c + 1] + v2 * X[3 * c + 2]) * inv
            for r in 0..<3 { out[3 * c + r] = -(sp * X[3 * c + r] + X[r] * vs) / L }
        }
        if ncol == 8 { for r in 0..<3 { out[21 + r] += k * X[r] } }
        return out
    }

    static func rk4(_ Y: inout [Double], _ X: inout [Double], _ h: Double, _ L: Double, _ ncol: Int) {
        let n = 3 * ncol
        var SY = [Double](repeating: 0, count: 24), Xt = SY
        let K1 = rhs(X, L, ncol)
        for i in 0..<n { SY[i] = X[i]; Xt[i] = X[i] + 0.5 * h * K1[i]; SY[i] += 2.0 * Xt[i] }
        let K2 = rhs(Xt, L, ncol)
        for i in 0..<n { Xt[i] = X[i] + 0.5 * h * K2[i]; SY[i] += 2.0 * Xt[i] }
        let K3 = rhs(Xt, L, ncol)
        for i in 0..<n { Xt[i] = X[i] + h * K3[i]; SY[i] += Xt[i] }
        let K4 = rhs(Xt, L, ncol)
        for i in 0..<n {
            Y[i] += h / 6.0 * SY[i]
            X[i] += h / 6.0 * (K1[i] + 2.0 * K2[i] + 2.0 * K3[i] + K4[i])
        }
    }
}

final class ReferenceKernelTests: XCTestCase {
    func testSIMD8KernelIsBitwiseTheScalarKernel() {
        var rng = SystemRandomNumberGenerator()
        var nCmp = 0, nNaN = 0
        for trial in 0..<2000 {
            let ncol = trial % 2 == 0 ? 8 : 7
            let p0 = SIMD3(Double.random(in: -8...8, using: &rng), Double.random(in: -4...4, using: &rng), Double.random(in: 0...4, using: &rng))
            let v0 = SIMD3(Double.random(in: -120...120, using: &rng), Double.random(in: -20...20, using: &rng), Double.random(in: -40...40, using: &rng))
            let L = Double.random(in: 2...9, using: &rng)
            var Y = [Double](repeating: 0, count: 24), X = Y
            for r in 0..<3 { Y[r] = p0[r]; X[r] = v0[r]; Y[3 * (1 + r) + r] = 1; X[3 * (4 + r) + r] = 1 }
            var st = AugState(p0, v0)
            let sgn = trial % 4 < 2 ? 1.0 : -1.0
            for _ in 0..<20 {
                let h = sgn * Double.random(in: 0...0.01, using: &rng)
                ScalarReference.rk4(&Y, &X, h, L, ncol)
                st.rk4(h, L, ncol == 8)
                let rows = [(st.y0, st.x0), (st.y1, st.x1), (st.y2, st.x2)]
                for c in 0..<ncol {
                    for r in 0..<3 {
                        for (a, b) in [(Y[3 * c + r], rows[r].0[c]), (X[3 * c + r], rows[r].1[c])] {
                            if a.isNaN || b.isNaN { XCTAssert(a.isNaN && b.isNaN); nNaN += 1 } else {
                                XCTAssertEqual(a.bitPattern, b.bitPattern, "r\(r) c\(c)")
                            }
                            nCmp += 1
                        }
                    }
                }
            }
        }
        print("SIMD8 vs scalar RK4: \(nCmp) values, \(nNaN) NaN in both, all others bit-identical")
    }
}
