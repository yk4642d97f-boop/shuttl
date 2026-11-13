/// Single-frame fallback of the automatic path (the median image failed).
/// Port of court_calibration._fallback_indices, _camera_agreement_px, _frame_fallback.
import Foundation
import simd
import ShuttlCore

/// `FRAME_AGREE_PX`, `FALLBACK_FRAMES`
let frameAgreePx = 3.0
let fallbackFrames = 5

enum AutoFallback {
    /// `_fallback_indices`: 1/2, 1/4, 3/4, 1/8, 7/8 of the list (distinct, in that order).
    static func indices(_ n: Int) -> [Int] {
        var out: [Int] = []
        for (num, den) in [(1, 2), (1, 4), (3, 4), (1, 8), (7, 8)] {
            let i = (n * num) / den
            if !out.contains(i) { out.append(i) }
        }
        return Array(out.prefix(fallbackFrames))
    }

    /// Court keypoints + net-post tops (ShuttlPresent CameraAgreement.points).
    static let agreementPoints: [SIMD3<Double>] = CourtModel.keypointsArray().points + CourtModel.netPostTops

    /// `_camera_agreement_px` = ShuttlPresent CameraAgreement.maxDisagreementPx (same size):
    /// symmetric max projection distance over points inside either image.
    static func agreementPx(_ a: CameraModel, _ b: CameraModel) -> Double {
        func one(_ x: CameraModel, _ y: CameraModel) -> Double {
            let w = Double(x.imageSize.width), h = Double(x.imageSize.height)
            var worst = -Double.infinity, any = false
            for p in agreementPoints {
                let u = x.project(p)
                guard u.x.isFinite, u.y.isFinite, u.x >= 0, u.y >= 0, u.x <= w, u.y <= h else { continue }
                any = true
                let v = y.project(p)
                worst = max(worst, v.x.isFinite && v.y.isFinite ? simd_distance(u, v) : .infinity)
            }
            return any ? worst : .infinity
        }
        return max(one(a, b), one(b, a))
    }

    struct Outcome {
        var camera: CameraModel?
        var diagnostics: CourtCalibrationDiagnostics?      // the agreeing earlier frame's
        var nFrameAccepts = 0
        var framesTried: [Int] = []
    }

    /// `_frame_fallback`: try frames in `indices(n)` order until two DIFFERENT frames agree
    /// (the earlier frame's camera is returned); stop once a pair is out of reach.
    /// `attempt(i)` calibrates frame i (vetoDominated, focal hint) → (camera?, diagnostics).
    static func run(n: Int, attempt: (Int) -> (CameraModel?, CourtCalibrationDiagnostics)) -> Outcome {
        let idx = indices(n)
        var accepted: [(Int, CameraModel, CourtCalibrationDiagnostics)] = []
        var out = Outcome()
        for (k, i) in idx.enumerated() {
            if accepted.count + (idx.count - k) < 2 { break }
            out.framesTried.append(i)
            let (cam, d) = attempt(i)
            guard let cam else { continue }
            for (j, cj, dj) in accepted {
                let px = agreementPx(cj, cam)
                if px <= frameAgreePx {
                    var dd = dj
                    dd.framesAgree = [j, i]; dd.frameAgreePx = px
                    dd.nFrameAccepts = accepted.count + 1; dd.framesTried = out.framesTried
                    out.camera = cj; out.diagnostics = dd; out.nFrameAccepts = accepted.count + 1
                    return out
                }
            }
            accepted.append((i, cam, d))
        }
        out.nFrameAccepts = accepted.count
        return out
    }

    /// The fallback's tail in `calibrate_from_frames`: the agreeing frame's camera, else the
    /// median's diagnostics with the fallback counts ("inconsistent_frames" when a frame
    /// gave a camera that no other frame confirmed).
    static func finish(median: CourtCalibrationDiagnostics, n: Int,
                       attempt: (Int) -> (CameraModel?, CourtCalibrationDiagnostics)) -> AutoCalibrationResult {
        let o = run(n: n, attempt: attempt)
        if let cam = o.camera, let d = o.diagnostics { return AutoCalibrationResult(camera: cam, diagnostics: d) }
        var d = median
        d.nFrameAccepts = o.nFrameAccepts
        d.framesTried = o.framesTried
        if o.nFrameAccepts > 0 { d.reason = .inconsistentFrames }
        return AutoCalibrationResult(camera: nil, diagnostics: d)
    }
}

extension AutoCalibration {
    /// `far_m_per_px`: floor metres per image pixel at the far-baseline centre (the baseline
    /// on the side of the net away from the camera), worst direction = 1 / smallest singular
    /// value of d(u, v)/d(X, Y); central differences of `project` (includes k1) and the same
    /// closed-form 2x2 singular value as Python.
    static func farMPerPx(_ cam: CameraModel, h: Double = 1e-3) -> Double {
        let xf = cam.center.x < 0 ? CourtModel.halfLength : -CourtModel.halfLength
        let p = [SIMD3(xf + h, 0, 0), SIMD3(xf - h, 0, 0), SIMD3(xf, h, 0), SIMD3(xf, -h, 0)].map { cam.project($0) }
        guard p.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else { return .infinity }
        let dX = (p[0] - p[1]) / (2 * h), dY = (p[2] - p[3]) / (2 * h)
        let a = dX.x, c = dX.y, b = dY.x, d = dY.y
        let s = a * a + b * b + c * c + d * d
        let det = abs(a * d - b * c)
        let smax = (0.5 * s + max(0.25 * s * s - det * det, 0).squareRoot()).squareRoot()
        return det > 0 ? smax / det : .infinity
    }
}
