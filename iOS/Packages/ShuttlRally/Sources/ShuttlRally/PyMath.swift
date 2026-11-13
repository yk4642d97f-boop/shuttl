import Foundation
import ShuttlCore

// Exact spellings of the NumPy / CPython float kernels used by the reference
// (checked against numpy 1.26 + OpenBLAS and CPython 3.12 on macOS).
extension Py {
    /// `np.linalg.norm` of a 1-D length-3 vector (OpenBLAS ddot = FMA chain).
    @inline(__always)
    static func norm3(_ v: SIMD3<Double>) -> Double {
        (v.x * v.x).addingProduct(v.y, v.y).addingProduct(v.z, v.z).squareRoot()
    }

    /// `np.linalg.norm` of a 1-D length-2 vector.
    @inline(__always)
    static func norm2(_ v: SIMD2<Double>) -> Double {
        (v.x * v.x).addingProduct(v.y, v.y).squareRoot()
    }

    /// `np.linalg.norm(a, axis=1)` row of an (N, 2) array (no BLAS: x² + y²).
    @inline(__always)
    static func rowNorm2(_ dx: Double, _ dy: Double) -> Double { (dx * dx + dy * dy).squareRoot() }

    /// CPython 3.12 `math.hypot(*xs)` / `math.dist` (vector_norm: lossless
    /// scaling + double-length accumulation + one Newton correction).
    static func hypot(_ xs: Double...) -> Double { vectorNorm(xs.map { Swift.abs($0) }) }

    static func dist(_ p: SIMD2<Double>, _ q: SIMD2<Double>) -> Double {
        vectorNorm([Swift.abs(p.x - q.x), Swift.abs(p.y - q.y)])
    }

    private static func vectorNorm(_ v: [Double]) -> Double {
        var mx = 0.0
        var foundNaN = false
        for x in v { foundNaN = foundNaN || x.isNaN; if x > mx { mx = x } }
        return vectorNorm(v, mx, foundNaN)
    }

    private static func vectorNorm(_ vec0: [Double], _ mx: Double, _ foundNaN: Bool) -> Double {
        var vec = vec0
        if mx.isInfinite { return mx }
        if foundNaN { return .nan }
        if mx == 0.0 || vec.count <= 1 { return mx }
        let maxE = Foundation.frexp(mx).1
        if maxE < -1023 {
            for i in vec.indices { vec[i] /= Double.leastNormalMagnitude }
            return Double.leastNormalMagnitude * vectorNorm(vec, mx / Double.leastNormalMagnitude, foundNaN)
        }
        let scale = Foundation.scalbn(1.0, -maxE)
        var csum = 1.0, frac1 = 0.0, frac2 = 0.0
        for x0 in vec {
            let x = x0 * scale
            let prHi = x * x
            let prLo = fmaErr(x, x, prHi)
            let smHi = csum + prHi
            let smLo = prHi - (smHi - csum)
            csum = smHi
            frac1 += prLo
            frac2 += smLo
        }
        var h = (csum - 1.0 + (frac1 + frac2)).squareRoot()
        let prHi = -h * h
        let prLo = fmaErr(-h, h, prHi)
        let smHi = csum + prHi
        let smLo = prHi - (smHi - csum)
        csum = smHi
        frac1 += prLo
        frac2 += smLo
        let x = csum - 1.0 + (frac1 + frac2)
        h += x / (2.0 * h)
        return h / scale
    }

    /// Exact error of a product: fma(a, b, -p) with p = a * b.
    @inline(__always)
    private static func fmaErr(_ a: Double, _ b: Double, _ p: Double) -> Double { (-p).addingProduct(a, b) }

    /// `math.degrees`.
    @inline(__always)
    static func degrees(_ x: Double) -> Double { x * (180.0 / Double.pi) }
}
