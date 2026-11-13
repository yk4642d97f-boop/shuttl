import Foundation
import CoreGraphics
import simd

// MARK: - CourtCalibration
// Pinhole camera in the backend court frame (see CourtGeometry):
//   camera = R · world + t,   pixel = K · (camera / z)   (+ OpenCV k1 k2 p1 p2 k3 distortion)
// Pixels are full-resolution pixels of the upright frame, origin top-left, y down.
// Mirrors backend/app/services/tracking/camera.py `CameraModel`; the Codable form matches
// `CameraModel.to_dict()` / `from_dict()` key for key.

struct CourtCalibration: Sendable {
    struct PixelSize: Hashable, Sendable {
        var width: Int
        var height: Int
        var cgSize: CGSize { CGSize(width: width, height: height) }
    }

    /// A known world point (metres) seen at a pixel.
    struct Correspondence: Sendable {
        var world: SIMD3<Double>
        var pixel: CGPoint
    }

    enum CalibrationError: Error, Equatable {
        /// Fewer than four floor (z = 0) points.
        case notEnoughFloorPoints
        /// Collinear / coincident points or no valid pose.
        case degenerate
        /// The solved camera sits under the floor — the corners are almost always
        /// mislabelled (mirrored left/right or near/far).
        case cameraBelowFloor
    }

    var K: simd_double3x3
    var R: simd_double3x3
    var t: SIMD3<Double>
    /// OpenCV k1 k2 p1 p2 k3, always 5 entries.
    var dist: [Double]
    var imageSize: PixelSize
    /// Calibration fit quality; nil when unknown (backend sends NaN → null).
    var reprojRMSEPx: Double?
    /// 0…1, calibration self-assessed.
    var confidence: Double

    init(K: simd_double3x3, R: simd_double3x3, t: SIMD3<Double>, imageSize: PixelSize,
         dist: [Double] = [], reprojRMSEPx: Double? = nil, confidence: Double = 0) {
        self.K = K
        self.R = R
        self.t = t
        self.imageSize = imageSize
        self.dist = Self.padded(dist)
        self.reprojRMSEPx = reprojRMSEPx
        self.confidence = confidence
    }

    private static func padded(_ d: [Double]) -> [Double] {
        Array((d + [0, 0, 0, 0, 0]).prefix(5))
    }
}

// MARK: - Projection

extension CourtCalibration {
    /// Camera centre in world coordinates.
    var center: SIMD3<Double> { -(R.transpose * t) }
    var focalLengthPx: Double { K[0, 0] }
    var hasDistortion: Bool { dist.contains { $0 != 0 } }

    /// World (metres) → pixel. Nil for points behind the camera.
    func project(_ world: SIMD3<Double>) -> CGPoint? {
        let c = R * world + t
        guard c.z > 1e-9 else { return nil }
        var xy = SIMD2(c.x / c.z, c.y / c.z)
        if hasDistortion { xy = Self.distort(xy, dist) }
        return CGPoint(x: K[0, 0] * xy.x + K[1, 0] * xy.y + K[2, 0],
                       y: K[0, 1] * xy.x + K[1, 1] * xy.y + K[2, 1])
    }

    /// Pixel → world ray (origin = camera centre, unit direction).
    func ray(through pixel: CGPoint) -> (origin: SIMD3<Double>, direction: SIMD3<Double>) {
        let a = K[0, 0], b = K[1, 0], c = K[0, 1], d = K[1, 1]
        let du = Double(pixel.x) - K[2, 0], dv = Double(pixel.y) - K[2, 1]
        let det = a * d - b * c
        var xy = SIMD2((d * du - b * dv) / det, (a * dv - c * du) / det)
        if hasDistortion { xy = Self.undistort(xy, dist) }
        return (center, simd_normalize(R.transpose * SIMD3(xy.x, xy.y, 1)))
    }

    /// Intersects the pixel's ray with the horizontal plane Z = `height`.
    /// Nil when the ray is parallel to the plane or the plane is behind the camera.
    func pixelToFloor(_ pixel: CGPoint, height: Double = 0) -> SIMD3<Double>? {
        let (o, d) = ray(through: pixel)
        guard abs(d.z) > 1e-12 else { return nil }
        let s = (height - o.z) / d.z
        guard s > 0, s.isFinite else { return nil }
        return o + d * s
    }

    /// 3×3 homography mapping floor (X, Y, 1) → pixels (ignores lens distortion).
    var floorHomography: simd_double3x3 {
        let m = K * simd_double3x3(columns: (R.columns.0, R.columns.1, t))
        return (1 / m[2, 2]) * m
    }

    /// Same camera for a resized image (e.g. taps made on a preview-sized frame).
    func scaled(to size: PixelSize) -> CourtCalibration {
        let sx = Double(size.width) / Double(imageSize.width)
        let sy = Double(size.height) / Double(imageSize.height)
        var out = self
        out.K = simd_double3x3(diagonal: SIMD3(sx, sy, 1)) * K
        out.imageSize = size
        out.reprojRMSEPx = reprojRMSEPx.map { $0 * max(sx, sy) }
        return out
    }

    /// Camera at `eye` looking at `target` (world up = +z) — same as `CameraModel.look_at`.
    static func lookAt(eye: SIMD3<Double>, target: SIMD3<Double>, focalPx: Double,
                       imageSize: PixelSize, rollDegrees: Double = 0) -> CourtCalibration {
        let fwd = simd_normalize(target - eye)
        let right = simd_normalize(simd_cross(fwd, SIMD3(0, 0, 1)))
        let down = simd_cross(fwd, right)
        var r = simd_double3x3(rows: [right, down, fwd])
        if rollDegrees != 0 {
            let a = rollDegrees * .pi / 180
            let rz = simd_double3x3(rows: [SIMD3(cos(a), -sin(a), 0), SIMD3(sin(a), cos(a), 0), SIMD3(0, 0, 1)])
            r = rz * r
        }
        let k = simd_double3x3(rows: [
            SIMD3(focalPx, 0, Double(imageSize.width) / 2),
            SIMD3(0, focalPx, Double(imageSize.height) / 2),
            SIMD3(0, 0, 1),
        ])
        return CourtCalibration(K: k, R: r, t: -(r * eye), imageSize: imageSize)
    }

    // MARK: Lens distortion (identical to camera.py _distort / _undistort)

    static func distort(_ p: SIMD2<Double>, _ d: [Double]) -> SIMD2<Double> {
        let k1 = d[0], k2 = d[1], p1 = d[2], p2 = d[3], k3 = d[4]
        let x = p.x, y = p.y
        let r2 = x * x + y * y
        let radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        return SIMD2(x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                     y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)
    }

    static func undistort(_ pd: SIMD2<Double>, _ d: [Double], iterations: Int = 8) -> SIMD2<Double> {
        var p = pd
        for _ in 0..<iterations { p += pd - distort(p, d) }
        return p
    }
}

// MARK: - Solving from correspondences

extension CourtCalibration {
    /// Typical phone main camera in 16:9 video: f ≈ 0.75 × long side (≈ 67° HFOV).
    static let defaultNormalisedFocal = 0.75

    /// Calibrates from the four tapped outer doubles corners.
    static func calibrate(corners: [CourtCorner: CGPoint], imageSize: PixelSize) throws -> CourtCalibration {
        let pairs = CourtCorner.allCases.compactMap { c in
            corners[c].map { Correspondence(world: c.world3, pixel: $0) }
        }
        return try calibrate(correspondences: pairs, imageSize: imageSize)
    }

    /// DLT floor homography → focal length (principal point at the image centre, square
    /// pixels, no skew) → pose decomposition → Levenberg–Marquardt refinement of
    /// (f, R, t) on all correspondences (non-floor points such as net-post tops help).
    static func calibrate(correspondences: [Correspondence], imageSize: PixelSize) throws -> CourtCalibration {
        let w = Double(imageSize.width), h = Double(imageSize.height)
        let s = max(w, h), cx = w / 2, cy = h / 2
        func norm(_ p: CGPoint) -> SIMD2<Double> { SIMD2((Double(p.x) - cx) / s, (Double(p.y) - cy) / s) }

        let floor = correspondences.filter { abs($0.world.z) < 1e-9 }
        guard floor.count >= 4 else { throw CalibrationError.notEnoughFloorPoints }

        // 1. Homography floor (metres) → normalised pixels, h33 = 1, Hartley-style scaling.
        let ws = max(floor.map { max(abs($0.world.x), abs($0.world.y)) }.max() ?? 1, 1e-6)
        var a: [Double] = [], b: [Double] = []
        for c in floor {
            let X = c.world.x / ws, Y = c.world.y / ws
            let p = norm(c.pixel)
            a += [X, Y, 1, 0, 0, 0, -p.x * X, -p.x * Y]; b.append(p.x)
            a += [0, 0, 0, X, Y, 1, -p.y * X, -p.y * Y]; b.append(p.y)
        }
        let rows = 2 * floor.count
        guard let hv = rows == 8
            ? DenseLinearAlgebra.solve(a, b, n: 8)
            : DenseLinearAlgebra.leastSquares(a, b, rows: rows, n: 8)
        else { throw CalibrationError.degenerate }
        let g = simd_double3x3(rows: [
            SIMD3(hv[0] / ws, hv[1] / ws, hv[2]),
            SIMD3(hv[3] / ws, hv[4] / ws, hv[5]),
            SIMD3(hv[6] / ws, hv[7] / ws, 1),
        ])

        // 2. Focal length: r1 ⟂ r2 and |r1| = |r2| with r_i = K'^-1 g_i, linear in u = 1/f².
        let g1 = g.columns.0, g2 = g.columns.1
        let a1 = g1.x * g2.x + g1.y * g2.y, b1 = g1.z * g2.z
        let a2 = g1.x * g1.x + g1.y * g1.y - g2.x * g2.x - g2.y * g2.y, b2 = g1.z * g1.z - g2.z * g2.z
        var fn = defaultNormalisedFocal
        let denom = a1 * a1 + a2 * a2
        if denom > 1e-18 {
            let u = -(a1 * b1 + a2 * b2) / denom
            if u > 0, (0.25...4).contains(1 / u.squareRoot()) { fn = 1 / u.squareRoot() }
        }

        // 3. Pose from G ∝ K' [r1 r2 t].
        guard let pose0 = decompose(g, normalisedFocal: fn) else { throw CalibrationError.degenerate }

        // 4. Refine f, R, t on reprojection error (normalised pixel units).
        let all = correspondences.map { (world: $0.world, pixel: norm($0.pixel)) }
        let r0 = pose0.R
        func unpack(_ p: [Double]) -> (f: Double, R: simd_double3x3, t: SIMD3<Double>) {
            (exp(p[0]), DenseLinearAlgebra.rotation(from: SIMD3(p[1], p[2], p[3])) * r0, SIMD3(p[4], p[5], p[6]))
        }
        func residuals(_ p: [Double]) -> [Double] {
            let (f, rot, tt) = unpack(p)
            var out: [Double] = []
            out.reserveCapacity(all.count * 2)
            for c in all {
                let q = rot * c.world + tt
                guard q.z > 1e-9 else { out += [10, 10]; continue }
                out.append(f * q.x / q.z - c.pixel.x)
                out.append(f * q.y / q.z - c.pixel.y)
            }
            return out
        }
        let start = [log(fn), 0, 0, 0, pose0.t.x, pose0.t.y, pose0.t.z]
        var best = DenseLinearAlgebra.levenbergMarquardt(start, residuals: residuals)
        if !(0.25...4).contains(exp(best[0])) { best = start }   // diverged: keep closed form

        let (f, rot, tt) = unpack(best)
        let res = residuals(best)
        let rmse = (res.reduce(0) { $0 + $1 * $1 } / Double(all.count)).squareRoot() * s
        let k = simd_double3x3(rows: [SIMD3(f * s, 0, cx), SIMD3(0, f * s, cy), SIMD3(0, 0, 1)])
        var cal = CourtCalibration(K: k, R: rot, t: tt, imageSize: imageSize, reprojRMSEPx: rmse)
        guard cal.center.z > 0 else { throw CalibrationError.cameraBelowFloor }
        cal.confidence = confidence(rmsePx: rmse, normalisedFocal: f, cameraHeight: cal.center.z,
                                    pointCount: all.count)
        return cal
    }

    /// G ∝ diag(f, f, 1) · [r1 r2 t] → (R, t) with the floor in front of the camera.
    private static func decompose(_ g: simd_double3x3, normalisedFocal fn: Double)
        -> (R: simd_double3x3, t: SIMD3<Double>)? {
        let m = simd_double3x3(diagonal: SIMD3(1 / fn, 1 / fn, 1)) * g
        let n1 = simd_length(m.columns.0), n2 = simd_length(m.columns.1)
        guard n1 > 1e-12, n2 > 1e-12 else { return nil }
        var lambda = 2 / (n1 + n2)
        if m.columns.2.z * lambda < 0 { lambda = -lambda }
        let r1 = lambda * m.columns.0, r2 = lambda * m.columns.1
        let r3 = simd_normalize(simd_cross(r1, r2))
        let rot = DenseLinearAlgebra.orthonormalized(simd_double3x3(columns: (r1, r2, r3)))
        let t = lambda * m.columns.2
        guard rot.determinant > 0, t.z > 0 else { return nil }
        return (rot, t)
    }

    /// Heuristic 0…1: reprojection fit × plausibility of the solved camera.
    /// With exactly four points the fit has a single residual degree of freedom, so the
    /// RMSE only weakly measures tap accuracy — capped accordingly.
    static func confidence(rmsePx: Double, normalisedFocal: Double, cameraHeight: Double,
                           pointCount: Int) -> Double {
        guard (0.3...3).contains(normalisedFocal), (0.2...40).contains(cameraHeight) else { return 0.1 }
        let fit = exp(-rmsePx / 3)
        let redundancy = pointCount > 4 ? 1.0 : 0.85
        return max(0, min(1, fit * redundancy))
    }
}

// MARK: - Equatable

extension CourtCalibration: Equatable {
    static func == (l: CourtCalibration, r: CourtCalibration) -> Bool {
        l.K.rowArrays == r.K.rowArrays && l.R.rowArrays == r.R.rowArrays && l.t == r.t
            && l.dist == r.dist && l.imageSize == r.imageSize
            && l.reprojRMSEPx == r.reprojRMSEPx && l.confidence == r.confidence
    }
}

// MARK: - Codable (backend CameraModel JSON)
// {"K": [[..]], "R": [[..]], "t": [..], "dist": [..], "image_size": [w, h],
//  "reproj_rmse_px": .. | null, "confidence": ..}

extension CourtCalibration: Codable {
    private enum CodingKeys: String, CodingKey {
        case K, R, t, dist
        case imageSize = "image_size"
        case reprojRMSEPx = "reproj_rmse_px"
        case confidence
    }

    /// Accepts `[a, b, c]` or a column/row vector `[[a], [b], [c]]` (numpy shapes vary).
    private struct FlexibleVector: Decodable {
        var values: [Double]
        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let flat = try? c.decode([Double].self) {
                values = flat
            } else {
                values = try c.decode([[Double]].self).flatMap { $0 }
            }
        }
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        guard let k = simd_double3x3(rowArrays: try c.decode([[Double]].self, forKey: .K)) else {
            throw DecodingError.dataCorruptedError(forKey: .K, in: c, debugDescription: "K must be 3×3")
        }
        guard let r = simd_double3x3(rowArrays: try c.decode([[Double]].self, forKey: .R)) else {
            throw DecodingError.dataCorruptedError(forKey: .R, in: c, debugDescription: "R must be 3×3")
        }
        let tv = try c.decode(FlexibleVector.self, forKey: .t).values
        guard tv.count == 3 else {
            throw DecodingError.dataCorruptedError(forKey: .t, in: c, debugDescription: "t must have 3 values")
        }
        let size = try c.decode([Double].self, forKey: .imageSize)
        guard size.count == 2, size[0] > 0, size[1] > 0 else {
            throw DecodingError.dataCorruptedError(forKey: .imageSize, in: c, debugDescription: "image_size must be [w, h]")
        }
        let rmse = (try? c.decodeIfPresent(Double.self, forKey: .reprojRMSEPx)) ?? nil
        self.init(K: k, R: r, t: SIMD3(tv[0], tv[1], tv[2]),
                  imageSize: PixelSize(width: Int(size[0].rounded()), height: Int(size[1].rounded())),
                  dist: (try c.decodeIfPresent(FlexibleVector.self, forKey: .dist))?.values ?? [],
                  reprojRMSEPx: rmse.flatMap { $0.isFinite ? $0 : nil },
                  confidence: (try? c.decodeIfPresent(Double.self, forKey: .confidence)) ?? 0)
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(K.rowArrays, forKey: .K)
        try c.encode(R.rowArrays, forKey: .R)
        try c.encode([t.x, t.y, t.z], forKey: .t)
        try c.encode(dist, forKey: .dist)
        try c.encode([imageSize.width, imageSize.height], forKey: .imageSize)
        // Omitted when unknown so the backend's from_dict falls back to NaN.
        if let rmse = reprojRMSEPx, rmse.isFinite { try c.encode(rmse, forKey: .reprojRMSEPx) }
        try c.encode(confidence.isFinite ? confidence : 0, forKey: .confidence)
    }
}
