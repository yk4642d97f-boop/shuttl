/// Homography helpers: cv2.findHomography(method=0) and the floor-homography → camera
/// decomposition of court_calibration.py (_proper_orientation, _focal_from_homography,
/// _pose_from_homography, _camera_from_homography).
import Foundation
import simd
import ShuttlCore

enum Homography {
    /// cv2.findHomography(src, dst, 0): normalised DLT on float32-converted points
    /// (HomographyEstimatorCallback::runKernel), then — for more than 4 points — a
    /// reprojection-error refinement (OpenCV runs 10 LMSolver iterations; here LM runs to
    /// convergence of the same objective). H[2,2] = 1. nil when degenerate.
    static func find(_ src: [SIMD2<Double>], _ dst: [SIMD2<Double>]) -> simd_double3x3? {
        let count = src.count
        guard count >= 4, dst.count == count else { return nil }
        let M = src.map { SIMD2(Double(Float($0.x)), Double(Float($0.y))) }
        let m = dst.map { SIMD2(Double(Float($0.x)), Double(Float($0.y))) }
        var cM = SIMD2<Double>(0, 0), cm = SIMD2<Double>(0, 0)
        for i in 0..<count { cm += m[i]; cM += M[i] }
        cm /= Double(count); cM /= Double(count)
        var sM = SIMD2<Double>(0, 0), sm = SIMD2<Double>(0, 0)
        for i in 0..<count {
            sm += SIMD2(abs(m[i].x - cm.x), abs(m[i].y - cm.y))
            sM += SIMD2(abs(M[i].x - cM.x), abs(M[i].y - cM.y))
        }
        let eps = Double.ulpOfOne
        if abs(sm.x) < eps || abs(sm.y) < eps || abs(sM.x) < eps || abs(sM.y) < eps { return nil }
        sm = SIMD2(Double(count) / sm.x, Double(count) / sm.y)
        sM = SIMD2(Double(count) / sM.x, Double(count) / sM.y)
        let invHnorm = rowMajor3(SIMD3(1 / sm.x, 0, cm.x), SIMD3(0, 1 / sm.y, cm.y), SIMD3(0, 0, 1))
        let Hnorm2 = rowMajor3(SIMD3(sM.x, 0, -cM.x * sM.x), SIMD3(0, sM.y, -cM.y * sM.y), SIMD3(0, 0, 1))
        var LtL = Matrix(rows: 9, cols: 9)
        for i in 0..<count {
            let x = (m[i].x - cm.x) * sm.x, y = (m[i].y - cm.y) * sm.y
            let X = (M[i].x - cM.x) * sM.x, Y = (M[i].y - cM.y) * sM.y
            let Lx: [Double] = [X, Y, 1, 0, 0, 0, -x * X, -x * Y, -x]
            let Ly: [Double] = [0, 0, 0, X, Y, 1, -y * X, -y * Y, -y]
            for j in 0..<9 { for k in j..<9 { LtL[j, k] += Lx[j] * Lx[k] + Ly[j] * Ly[k] } }
        }
        for j in 0..<9 { for k in 0..<j { LtL[j, k] = LtL[k, j] } }
        let (_, vecs) = LinAlg.symmetricEigen(LtL)   // ascending → column 0 = smallest
        let h = (0..<9).map { vecs[$0, 0] }
        let H0 = rowMajor3(SIMD3(h[0], h[1], h[2]), SIMD3(h[3], h[4], h[5]), SIMD3(h[6], h[7], h[8]))
        var H = invHnorm * H0 * Hnorm2
        let h22 = mget(H, 2, 2)
        guard h22 != 0, h22.isFinite else { return nil }
        H = H * (1 / h22)
        if count > 4 {
            let p0 = (0..<8).map { mget(H, $0 / 3, $0 % 3) }
            let p = LeastSquares.minimize(p0, loss: .linear, maxIterations: 50) { q in
                var r = [Double](repeating: 0, count: 2 * count)
                for i in 0..<count {
                    let X = M[i].x, Y = M[i].y
                    let ww = q[6] * X + q[7] * Y + 1
                    let w = abs(ww) > Double.ulpOfOne ? 1 / ww : 0
                    r[2 * i] = (q[0] * X + q[1] * Y + q[2]) * w - m[i].x
                    r[2 * i + 1] = (q[3] * X + q[4] * Y + q[5]) * w - m[i].y
                }
                return r
            }
            H = rowMajor3(SIMD3(p[0], p[1], p[2]), SIMD3(p[3], p[4], p[5]), SIMD3(p[6], p[7], 1))
        }
        return H
    }

    /// `_proper_orientation`: H maps the floor as seen from ABOVE.
    static func properOrientation(_ H: simd_double3x3, _ xy: SIMD2<Double>) -> Bool {
        let w = simd_dot(mrow(H, 2), SIMD3(xy.x, xy.y, 1))
        return H.determinant * w * w * w < 0
    }

    /// `_focal_from_homography` for one H → (focal or NaN, conditioning weight).
    static func focal(_ H: simd_double3x3, cx: Double, cy: Double, scale: Double) -> (f: Double, weight: Double) {
        let T = rowMajor3(SIMD3(1 / scale, 0, -cx / scale), SIMD3(0, 1 / scale, -cy / scale), SIMD3(0, 0, 1))
        var Hc = T * H
        var nrm = 0.0
        for c in 0..<3 { for r in 0..<3 { nrm += Hc[c, r] * Hc[c, r] } }
        Hc = Hc * (1 / nrm.squareRoot())
        return focalCore(h1: Hc[0], h2: Hc[1], scale: scale)
    }

    @inline(__always)
    static func focalCore(h1: SIMD3<Double>, h2: SIMD3<Double>, scale: Double) -> (f: Double, weight: Double) {
        let a1 = h1.x * h2.x + h1.y * h2.y
        let b1 = h1.z * h2.z
        let a2 = h1.x * h1.x + h1.y * h1.y - h2.x * h2.x - h2.y * h2.y
        let b2 = h1.z * h1.z - h2.z * h2.z
        let den = a1 * a1 + a2 * a2
        let w = -(a1 * b1 + a2 * b2) / den
        let f = w > 0 ? scale / abs(w).squareRoot() : Double.nan
        return (f, den.squareRoot())
    }

    /// `_pose_from_homography`
    static func pose(_ H: simd_double3x3, K: simd_double3x3, ref: SIMD2<Double>) -> (R: simd_double3x3, t: SIMD3<Double>) {
        let Mx = K.inverse * H
        let M0 = Mx[0], M1 = Mx[1], M2 = Mx[2]
        var lam = 2.0 / (simd_length(M0) + simd_length(M1))
        if (Mx * SIMD3(ref.x, ref.y, 1)).z * lam < 0 { lam = -lam }
        let r1 = M0 * lam, r2 = M1 * lam, t = M2 * lam
        let R0 = simd_double3x3(columns: (r1, r2, simd_cross(r1, r2)))
        let (u, _, vt) = LinAlg.svd(Matrix(R0))
        let U = u.simd3x3, Vt = vt.simd3x3
        var R = U * Vt
        if R.determinant < 0 {
            R = U * simd_double3x3(diagonal: SIMD3(1, 1, -1)) * Vt
        }
        return (R, t)
    }

    /// `_camera_from_homography`
    static func camera(_ H: simd_double3x3, size: ImageSize, refXY: [SIMD2<Double>], f fGiven: Double? = nil) -> CameraModel? {
        let w = Double(size.width), h = Double(size.height)
        let cx = w / 2, cy = h / 2
        var f: Double
        if let fGiven { f = fGiven } else {
            let fe = focal(H, cx: cx, cy: cy, scale: max(w, h)).f
            f = fe.isFinite && 0.15 * w < fe && fe < 8 * w ? fe : defaultFocal(size)
        }
        let K = rowMajor3(SIMD3(f, 0, cx), SIMD3(0, f, cy), SIMD3(0, 0, 1))
        var ref = SIMD2<Double>(0, 0)
        for p in refXY { ref += p }
        ref /= Double(refXY.count)
        let (R, t) = pose(H, K: K, ref: ref)
        let cam = CameraModel(K: K, R: R, t: t, imageSize: size)
        if cam.center.z <= 0 { return nil }
        return cam
    }
}
