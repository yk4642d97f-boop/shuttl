import Foundation
import simd
import ShuttlCore

public extension CameraModel {
    /// The same physical camera expressed at another resolution of the SAME field of view
    /// (a uniform resize of the upright frame, e.g. 640x360 working frames → 1920x1080 video).
    ///
    /// Pixel-centre convention (what cv2 INTER_AREA / box downscaling implies): full-res pixel
    /// index u maps to u' = (u + 0.5)·s − 0.5, so fx, fy scale by s and the principal point by
    /// c' = (c + 0.5)·s − 0.5. Lens distortion (k1…) acts on normalised coordinates and is
    /// unchanged. Returns nil when the aspect ratios differ by more than 1 % (that is a crop,
    /// not a resize, and the field of view would differ).
    func rescaled(to size: ImageSize) -> CameraModel? {
        CameraModel.rescale(self, to: size, pixelCentre: true)
    }

    /// Frame size helper for the pipeline: `rescaled(to:)` of the upright video size.
    func rescaled(width: Int, height: Int) -> CameraModel? { rescaled(to: ImageSize(width: width, height: height)) }

    /// `rescaled(to:)` with an explicit convention (`pixelCentre: false` = plain c' = c·s).
    static func rescale(_ c: CameraModel, to size: ImageSize, pixelCentre: Bool) -> CameraModel? {
        let w0 = Double(c.imageSize.width), h0 = Double(c.imageSize.height)
        let w1 = Double(size.width), h1 = Double(size.height)
        guard w0 > 0, h0 > 0, w1 > 0, h1 > 0 else { return nil }
        if size == c.imageSize { return c }
        let sx = w1 / w0, sy = h1 / h0
        guard abs(sx / sy - 1) <= 0.01 else { return nil }
        var K = c.K
        let off = pixelCentre ? 0.5 : 0.0
        K[0, 0] *= sx                 // fx
        K[1, 0] *= sx                 // skew
        K[1, 1] *= sy                 // fy
        K[2, 0] = (K[2, 0] + off) * sx - off
        K[2, 1] = (K[2, 1] + off) * sy - off
        var out = CameraModel(K: K, R: c.R, t: c.t, imageSize: size, dist: c.dist,
                              reprojRmsePx: c.reprojRmsePx * (sx + sy) / 2, confidence: c.confidence)
        if !out.reprojRmsePx.isFinite { out.reprojRmsePx = c.reprojRmsePx }
        return out
    }

    /// Roll about the optical axis in degrees (0 = phone level, the floor's horizon is horizontal),
    /// same sign as `CameraModel.lookAt(rollDeg:)`. Positive = the scene appears rotated clockwise
    /// in the picture (turn the phone clockwise, as seen from behind it, to level it).
    var rollDegrees: Double {
        // Rows 0 / 1 of R = image x / y axes in world coordinates; their world-z components are
        // sin(roll)·cos(pitch) and −cos(roll)·cos(pitch).
        let xz = R[2, 0], yz = R[2, 1]
        guard xz.isFinite, yz.isFinite, xz != 0 || yz != 0 else { return 0 }
        return atan2(xz, -yz) * 180 / .pi
    }
}

/// Max / median pixel distance between two cameras' projections of the 30 court keypoints
/// (28 floor keypoints + 2 net-post tops) that land inside `a`'s frame (±margin px).
/// Used for lock agreement: two solutions "agree" when the court they draw coincides.
public enum CameraAgreement {
    public static let points: [SIMD3<Double>] =
        CourtModel.keypointsArray().points + CourtModel.netPostTops

    public static func disagreementPx(_ a: CameraModel, _ b: CameraModel, margin: Double = 0) -> (max: Double, median: Double, count: Int) {
        let w = Double(a.imageSize.width), h = Double(a.imageSize.height)
        var d: [Double] = []
        d.reserveCapacity(points.count)
        for p in points {
            let u = a.project(p)
            guard u.x.isFinite, u.y.isFinite, u.x >= -margin, u.y >= -margin, u.x <= w + margin, u.y <= h + margin
            else { continue }
            let v = b.project(p)
            d.append(v.x.isFinite && v.y.isFinite ? simd_distance(u, v) : .infinity)
        }
        guard !d.isEmpty else { return (.infinity, .infinity, 0) }
        d.sort()
        let m = d.count % 2 == 1 ? d[d.count / 2] : (d[d.count / 2 - 1] + d[d.count / 2]) / 2
        return (d.last!, m, d.count)
    }

    /// Symmetric max disagreement (keypoints visible in either frame), in `a`'s pixels.
    public static func maxDisagreementPx(_ a: CameraModel, _ b: CameraModel) -> Double {
        let bb = b.imageSize == a.imageSize ? b : (b.rescaled(to: a.imageSize) ?? b)
        return max(disagreementPx(a, bb).max, disagreementPx(bb, a).max)
    }
}
