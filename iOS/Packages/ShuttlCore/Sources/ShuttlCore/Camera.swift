import Foundation
import simd

/// Frame size in pixels. Encodes as `[width, height]` like the Python tuple.
public struct ImageSize: Sendable, Hashable, Codable {
    public var width: Int
    public var height: Int
    public init(width: Int, height: Int) { self.width = width; self.height = height }

    public init(from decoder: Decoder) throws {
        var c = try decoder.unkeyedContainer()
        width = try c.decode(Int.self); height = try c.decode(Int.self)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.unkeyedContainer()
        try c.encode(width); try c.encode(height)
    }
}

/// Pinhole camera shared by calibration, 3-D reconstruction and simulation.
/// Port of backend/app/services/tracking/camera.py.
///
/// Pixel convention: full-resolution pixels of the UPRIGHT (rotation-applied) frame,
/// origin top-left, x right, y down. World frame: see `CourtModel`.
/// Functions return NaN (not nil) for invalid geometry, mirroring numpy, so ported
/// code can keep its `isFinite` checks.
public struct CameraModel: Sendable, Equatable {
    public var K: simd_double3x3              // intrinsics
    public var R: simd_double3x3              // world → camera rotation
    public var t: SIMD3<Double>               // world → camera translation
    public var imageSize: ImageSize
    public var dist: [Double]                 // OpenCV k1 k2 p1 p2 k3
    public var reprojRmsePx: Double
    public var confidence: Double

    public init(K: simd_double3x3, R: simd_double3x3, t: SIMD3<Double>, imageSize: ImageSize,
                dist: [Double] = [0, 0, 0, 0, 0], reprojRmsePx: Double = .nan, confidence: Double = 0) {
        self.K = K; self.R = R; self.t = t; self.imageSize = imageSize
        self.dist = dist; self.reprojRmsePx = reprojRmsePx; self.confidence = confidence
    }

    public var isDistorted: Bool { dist.contains { $0 != 0 } }

    /// Camera centre in world coordinates.
    public var center: SIMD3<Double> { -(R.transpose * t) }

    /// World point → pixel; NaN when the point is behind the camera.
    public func project(_ p: SIMD3<Double>) -> SIMD2<Double> {
        let cam = R * p + t
        guard cam.z > 1e-9 else { return SIMD2(.nan, .nan) }
        var xy = SIMD2(cam.x / cam.z, cam.y / cam.z)
        if isDistorted { xy = CameraModel.distort(xy, dist) }
        return SIMD2(K[0, 0] * xy.x + K[1, 0] * xy.y + K[2, 0],
                     K[0, 1] * xy.x + K[1, 1] * xy.y + K[2, 1])
    }

    public func project(_ pts: [SIMD3<Double>]) -> [SIMD2<Double>] { pts.map(project) }

    /// Pixel → world ray (origin = camera centre, unit direction).
    public func ray(_ uv: SIMD2<Double>) -> (origin: SIMD3<Double>, direction: SIMD3<Double>) {
        let kInv = K.inverse
        var xy = SIMD2(kInv[0, 0] * uv.x + kInv[1, 0] * uv.y + kInv[2, 0],
                       kInv[0, 1] * uv.x + kInv[1, 1] * uv.y + kInv[2, 1])
        if isDistorted { xy = CameraModel.undistort(xy, dist) }
        let d = R.transpose * SIMD3(xy.x, xy.y, 1)
        return (center, simd_normalize(d))
    }

    /// Intersect the pixel's ray with the plane Z = z. NaN if parallel or behind.
    public func pixelToFloor(_ uv: SIMD2<Double>, z: Double = 0) -> SIMD3<Double> {
        let (o, d) = ray(uv)
        let s = (z - o.z) / d.z
        guard s > 0, s.isFinite else { return SIMD3(repeating: .nan) }
        return o + d * s
    }

    public func pixelToFloor(_ uv: [SIMD2<Double>], z: Double = 0) -> [SIMD3<Double>] { uv.map { pixelToFloor($0, z: z) } }

    /// H mapping floor (X, Y, 1) → pixels (ignores lens distortion), normalised H[2,2] = 1.
    public func floorHomography() -> simd_double3x3 {
        let h = K * simd_double3x3(columns: (R.columns.0, R.columns.1, t))
        return h * (1 / h[2, 2])
    }

    /// Camera at `eye` looking at `target` (world up = +z).
    public static func lookAt(eye: SIMD3<Double>, target: SIMD3<Double>, focalPx: Double,
                              imageSize: ImageSize, rollDeg: Double = 0) -> CameraModel {
        let fwd = simd_normalize(target - eye)
        let right = simd_normalize(simd_cross(fwd, SIMD3(0, 0, 1)))
        let down = simd_cross(fwd, right)
        var R = simd_double3x3(rows: [right, down, fwd])
        if rollDeg != 0 {
            let r = rollDeg * .pi / 180, c = cos(r), s = sin(r)
            R = simd_double3x3(rows: [SIMD3(c, -s, 0), SIMD3(s, c, 0), SIMD3(0, 0, 1)]) * R
        }
        let K = simd_double3x3(rows: [SIMD3(focalPx, 0, Double(imageSize.width) / 2),
                                      SIMD3(0, focalPx, Double(imageSize.height) / 2),
                                      SIMD3(0, 0, 1)])
        return CameraModel(K: K, R: R, t: -(R * eye), imageSize: imageSize)
    }

    public static func distort(_ xy: SIMD2<Double>, _ dist: [Double]) -> SIMD2<Double> {
        let d = dist + Array(repeating: 0, count: max(0, 5 - dist.count))
        let (k1, k2, p1, p2, k3) = (d[0], d[1], d[2], d[3], d[4])
        let x = xy.x, y = xy.y, r2 = x * x + y * y
        let radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        return SIMD2(x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                     y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)
    }

    /// Fixed-point inversion of `distort`, adequate for phone lenses.
    public static func undistort(_ xyD: SIMD2<Double>, _ dist: [Double], iterations: Int = 8) -> SIMD2<Double> {
        var xy = xyD
        for _ in 0..<iterations { xy += xyD - distort(xy, dist) }
        return xy
    }
}

extension CameraModel: Codable {
    enum CodingKeys: String, CodingKey {
        case K, R, t, dist, imageSize = "image_size", reprojRmsePx = "reproj_rmse_px", confidence
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func matrix(_ key: CodingKeys) throws -> simd_double3x3 {
            let rows = try c.decode([[Double]].self, forKey: key)
            guard rows.count == 3, rows.allSatisfy({ $0.count == 3 }) else {
                throw DecodingError.dataCorruptedError(forKey: key, in: c, debugDescription: "expected a 3x3 matrix")
            }
            return simd_double3x3(rowMajor: rows)
        }
        K = try matrix(.K)
        R = try matrix(.R)
        let tv = try c.decode([Double].self, forKey: .t)
        guard tv.count == 3 else {
            throw DecodingError.dataCorruptedError(forKey: .t, in: c, debugDescription: "expected 3 values, got \(tv.count)")
        }
        t = SIMD3(tv[0], tv[1], tv[2])
        imageSize = try c.decode(ImageSize.self, forKey: .imageSize)
        dist = try c.decodeIfPresent([Double].self, forKey: .dist) ?? [0, 0, 0, 0, 0]
        reprojRmsePx = try c.decodeNaNable(forKey: .reprojRmsePx) ?? .nan
        confidence = try c.decodeIfPresent(Double.self, forKey: .confidence) ?? 0
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(K.rowMajor, forKey: .K)
        try c.encode(R.rowMajor, forKey: .R)
        try c.encode([t.x, t.y, t.z], forKey: .t)
        try c.encode(dist, forKey: .dist)
        try c.encode(imageSize, forKey: .imageSize)
        try c.encodeNaNable(reprojRmsePx, forKey: .reprojRmsePx)
        try c.encode(confidence, forKey: .confidence)
    }
}
