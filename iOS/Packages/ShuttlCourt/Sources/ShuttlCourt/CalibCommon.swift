/// Shared helpers of the court-calibration port.
/// Port of the module-level helpers in backend/app/services/tracking/court_calibration.py
/// (_ROT_Z_180, _make_camera, _project_params, _cam_params, _flip_180, _default_focal)
/// plus cv2.Rodrigues.
import Foundation
import simd
import ShuttlCore

/// `_ROT_Z_180 = diag([-1, -1, 1])`.
let rotZ180 = simd_double3x3(diagonal: SIMD3(-1.0, -1.0, 1.0))

/// `_flip_180`: re-express the camera in the court frame rotated 180° about z (near ↔ far).
func flip180(_ cam: CameraModel) -> CameraModel {
    var out = cam
    out.R = cam.R * rotZ180
    return out
}

/// `_default_focal`: 0.9 * max(image_size).
@inline(__always)
func defaultFocal(_ size: ImageSize) -> Double { 0.9 * Double(max(size.width, size.height)) }

@inline(__always)
func rowMajor3(_ r0: SIMD3<Double>, _ r1: SIMD3<Double>, _ r2: SIMD3<Double>) -> simd_double3x3 {
    simd_double3x3(rows: [r0, r1, r2])
}

@inline(__always) func mrow(_ m: simd_double3x3, _ r: Int) -> SIMD3<Double> { SIMD3(m[0, r], m[1, r], m[2, r]) }
@inline(__always) func mcol(_ m: simd_double3x3, _ c: Int) -> SIMD3<Double> { m[c] }
/// numpy-style element access m[r][c].
@inline(__always) func mget(_ m: simd_double3x3, _ r: Int, _ c: Int) -> Double { m[c, r] }

// MARK: - cv2.Rodrigues

enum Rodrigues {
    /// rvec → R exactly like cv2.Rodrigues (calib3d/src/calibration.cpp).
    static func matrix(_ r: SIMD3<Double>) -> simd_double3x3 {
        let theta = (r.x * r.x + r.y * r.y + r.z * r.z).squareRoot()
        if theta < Double.ulpOfOne { return matrix_identity_double3x3 }
        let c = cos(theta), s = sin(theta), c1 = 1 - c
        let itheta = theta != 0 ? 1 / theta : 0
        let u = r * itheta
        let rrt = rowMajor3(SIMD3(u.x * u.x, u.x * u.y, u.x * u.z),
                            SIMD3(u.x * u.y, u.y * u.y, u.y * u.z),
                            SIMD3(u.x * u.z, u.y * u.z, u.z * u.z))
        let rx = rowMajor3(SIMD3(0, -u.z, u.y), SIMD3(u.z, 0, -u.x), SIMD3(-u.y, u.x, 0))
        // R = cos(theta)*I + (1 - cos(theta))*r*rT + sin(theta)*[r_x]
        return c * matrix_identity_double3x3 + c1 * rrt + s * rx
    }

    /// R → rvec like cv2.Rodrigues (the SVD re-orthonormalisation is skipped: every R
    /// reaching it here is orthonormal to ~1e-15 already).
    static func vector(_ R: simd_double3x3) -> SIMD3<Double> {
        let g = { (r: Int, c: Int) -> Double in mget(R, r, c) }
        var rv = SIMD3(g(2, 1) - g(1, 2), g(0, 2) - g(2, 0), g(1, 0) - g(0, 1))
        let s = ((rv.x * rv.x + rv.y * rv.y + rv.z * rv.z) * 0.25).squareRoot()
        var c = (g(0, 0) + g(1, 1) + g(2, 2) - 1) * 0.5
        c = c > 1 ? 1 : c < -1 ? -1 : c
        var theta = acos(c)
        if s < 1e-5 {
            if c > 0 { return .zero }
            var t = (g(0, 0) + 1) * 0.5
            rv.x = max(t, 0).squareRoot()
            t = (g(1, 1) + 1) * 0.5
            rv.y = max(t, 0).squareRoot() * (g(0, 1) < 0 ? -1 : 1)
            t = (g(2, 2) + 1) * 0.5
            rv.z = max(t, 0).squareRoot() * (g(0, 2) < 0 ? -1 : 1)
            if abs(rv.x) < abs(rv.y) && abs(rv.x) < abs(rv.z) && (g(1, 2) > 0) != (rv.y * rv.z > 0) {
                rv.z = -rv.z
            }
            theta /= simd_length(rv)
            return rv * theta
        }
        let vth = 1 / (2 * s) * theta
        return rv * vth
    }
}

// MARK: - camera parameter vectors  p = [log f, rvec(3), t(3), (k1)]

/// `_make_camera`
func makeCamera(f: Double, rvec: SIMD3<Double>, t: SIMD3<Double>, size: ImageSize,
                k1: Double = 0, cx: Double? = nil, cy: Double? = nil) -> CameraModel {
    let cx = cx ?? Double(size.width) / 2
    let cy = cy ?? Double(size.height) / 2
    let K = rowMajor3(SIMD3(f, 0, cx), SIMD3(0, f, cy), SIMD3(0, 0, 1))
    return CameraModel(K: K, R: Rodrigues.matrix(rvec), t: t, imageSize: size, dist: [k1, 0, 0, 0, 0])
}

/// `_cam_params`
func camParams(_ cam: CameraModel, withK1: Bool) -> [Double] {
    let f = 0.5 * (mget(cam.K, 0, 0) + mget(cam.K, 1, 1))
    let r = Rodrigues.vector(cam.R)
    var p = [log(f), r.x, r.y, r.z, cam.t.x, cam.t.y, cam.t.z]
    if withK1 { p.append(cam.dist.first ?? 0) }
    return p
}

/// Precomputed pieces of `_project_params` for one parameter vector.
struct ParamProjector {
    let f: Double, R: simd_double3x3, t: SIMD3<Double>, k1: Double?, cx: Double, cy: Double

    init(_ p: [Double], cx: Double, cy: Double) {
        f = exp(p[0])
        R = Rodrigues.matrix(SIMD3(p[1], p[2], p[3]))
        t = SIMD3(p[4], p[5], p[6])
        k1 = p.count > 7 ? p[7] : nil
        self.cx = cx; self.cy = cy
    }

    /// `_project_params` for one point → (uv, camera z before clamping).
    @inline(__always)
    func project(_ P: SIMD3<Double>) -> (uv: SIMD2<Double>, z: Double) {
        // c = P @ R.T + t
        let c = R * P + t
        let z = max(c.z, 1e-6)
        var x = c.x / z, y = c.y / z
        if let k1 {
            let rad = 1 + k1 * (x * x + y * y)
            x *= rad; y *= rad
        }
        return (SIMD2(f * x + cx, f * y + cy), c.z)
    }

    func project(_ P: [SIMD3<Double>]) -> [SIMD2<Double>] { P.map { project($0).uv } }
}

extension CameraModel {
    /// numpy's `cam.K[0, 0]`.
    var focal: Double { mget(K, 0, 0) }
}
