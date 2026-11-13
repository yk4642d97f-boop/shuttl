/// Manual path: camera from user-tapped named court keypoints.
/// Port of court_calibration.calibrate_from_points / _refine_on_points.
import Foundation
import simd
import ShuttlCore

public enum CourtCalibrationError: Error, CustomStringConvertible, Equatable {
    case insufficientKeypoints(Int)
    case nonFiniteKeypoints
    case degenerateConfiguration
    case notPhysicalCamera

    public var description: String {
        switch self {
        case .insufficientKeypoints(let n): return "need >= 4 known court keypoints, got \(n)"
        case .nonFiniteKeypoints: return "non-finite keypoint coordinates"
        case .degenerateConfiguration: return "degenerate keypoint configuration"
        case .notPhysicalCamera: return "keypoints do not correspond to a physical camera"
        }
    }
}

enum ManualCalibration {
    /// `calibrate_from_points`. Point order: CourtModel.keypointNames order (Python uses the
    /// caller's dict insertion order; only rounding-level differences result).
    static func calibrate(_ imagePts: [String: SIMD2<Double>], size: ImageSize) throws -> CameraModel {
        let names = CourtModel.keypointNames.filter { imagePts[$0] != nil }
        guard names.count >= 4 else { throw CourtCalibrationError.insufficientKeypoints(names.count) }
        let uv = names.map { imagePts[$0]! }
        var xy = names.map { CourtModel.keypoints[$0]! }
        guard uv.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else { throw CourtCalibrationError.nonFiniteKeypoints }
        guard var H = Homography.find(xy, uv),
              (0..<3).allSatisfy({ c in (0..<3).allSatisfy { H[c, $0].isFinite } }),
              abs(H.determinant) >= 1e-12
        else { throw CourtCalibrationError.degenerateConfiguration }
        var mean = SIMD2<Double>(0, 0)
        for p in xy { mean += p }
        mean /= Double(xy.count)
        if !Homography.properOrientation(H, mean) {
            // keypoints are left/right mirrored w.r.t. court_model; swap labels
            xy = xy.map { SIMD2($0.x, -$0.y) }
            guard let H2 = Homography.find(xy, uv) else { throw CourtCalibrationError.degenerateConfiguration }
            H = H2
        }
        guard let cam0 = Homography.camera(H, size: size, refXY: xy) else { throw CourtCalibrationError.notPhysicalCamera }
        var cam = refineOnPoints(cam0, xy.map { SIMD3($0.x, $0.y, 0) }, uv)
        if cam.center.x > 0.5 { cam = flip180(cam) }
        let n = Double(names.count)
        cam.confidence = min(max(exp(-cam.reprojRmsePx / 4.0) * min(1.0, 0.75 + 0.03 * n), 0), 1)
        return cam
    }

    /// `_refine_on_points`: LM over (log f, rvec, t) on point reprojection, weak focal prior.
    static func refineOnPoints(_ cam: CameraModel, _ P: [SIMD3<Double>], _ uv: [SIMD2<Double>],
                               sigmaPx: Double = 1.0, logfSigma: Double = 1.0) -> CameraModel {
        let cx = mget(cam.K, 0, 2), cy = mget(cam.K, 1, 2)
        let p0 = camParams(cam, withK1: false)
        let logfPrior = log(defaultFocal(cam.imageSize))
        let sol = LeastSquares.minimize(p0, loss: .linear, maxIterations: 200) { p in
            let pr = ParamProjector(p, cx: cx, cy: cy)
            var r = [Double](); r.reserveCapacity(2 * P.count + 1)
            for i in 0..<P.count {
                let q = pr.project(P[i]).uv
                r.append((q.x - uv[i].x) / sigmaPx)
                r.append((q.y - uv[i].y) / sigmaPx)
            }
            r.append((p[0] - logfPrior) / logfSigma)
            return r
        }
        var out = makeCamera(f: exp(sol[0]), rvec: SIMD3(sol[1], sol[2], sol[3]), t: SIMD3(sol[4], sol[5], sol[6]),
                             size: cam.imageSize, cx: cx, cy: cy)
        if out.center.z <= 0 { out = cam }
        var s = 0.0
        for i in 0..<P.count {
            let e = out.project(P[i]) - uv[i]
            s += simd_length_squared(e)
        }
        out.reprojRmsePx = (s / Double(P.count)).squareRoot()
        return out
    }
}
