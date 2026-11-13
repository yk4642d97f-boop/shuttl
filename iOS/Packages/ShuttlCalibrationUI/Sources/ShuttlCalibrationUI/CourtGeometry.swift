import CoreGraphics
import Foundation
import ShuttlCore
import simd

public enum CourtQuadIssue: String, Sendable, Equatable {
    case missingCorner
    case pointOutOfBounds
    case nonConvex
    case degenerate
    case farBaselineNotAbove
    case farBaselineNotShorter
}

public enum CourtQuadValidator {
    /// Checks the active corner set (`taps.cornerSet`): its near pair (baseline or service
    /// line) must sit below and be wider than the far baseline, and the quad must be convex.
    public static func issue(for taps: CourtCornerTaps, minimumAreaFraction: Double = 0.0001) -> CourtQuadIssue? {
        let names = taps.cornerSet.names
        guard let nbLD = taps[names[0]], let nbRD = taps[names[1]],
              let fbLD = taps[names[2]], let fbRD = taps[names[3]] else { return .missingCorner }
        let points = [nbLD, nbRD, fbRD, fbLD]
        guard points.allSatisfy({ $0.x.isFinite && $0.y.isFinite && (0...1).contains($0.x) && (0...1).contains($0.y) }) else {
            return .pointOutOfBounds
        }
        guard fbLD.y < nbLD.y, fbRD.y < nbRD.y else { return .farBaselineNotAbove }

        let nearWidth = distance(nbLD, nbRD, imageSize: taps.imageSize)
        let farWidth = distance(fbLD, fbRD, imageSize: taps.imageSize)
        let nearMeanY = (Double(nbLD.y) + Double(nbRD.y)) / 2
        let farMeanY = (Double(fbLD.y) + Double(fbRD.y)) / 2
        let frameArea = Double(taps.imageSize.width) * Double(taps.imageSize.height)
        let area = abs(polygonArea(points.map { CGPoint(x: $0.x * CGFloat(taps.imageSize.width), y: $0.y * CGFloat(taps.imageSize.height)) }))
        guard frameArea > 0,
              nearMeanY - farMeanY > 0.005 else { return .degenerate }

        let cross = points.indices.map { index in
            crossProduct(points[index], points[(index + 1) % points.count], points[(index + 2) % points.count])
        }
        guard cross.allSatisfy({ $0 > 1e-8 }) || cross.allSatisfy({ $0 < -1e-8 }) else { return .nonConvex }
        guard nearWidth > 0, farWidth > nearWidth * 0.02,
              area / frameArea >= minimumAreaFraction else { return .degenerate }
        guard farWidth < nearWidth * 0.995 else { return .farBaselineNotShorter }
        return nil
    }

    public static func isValid(_ taps: CourtCornerTaps) -> Bool { issue(for: taps) == nil }

    private static func distance(_ a: CGPoint, _ b: CGPoint, imageSize: ImageSize) -> Double {
        hypot(Double(a.x - b.x) * Double(imageSize.width),
              Double(a.y - b.y) * Double(imageSize.height))
    }

    private static func crossProduct(_ a: CGPoint, _ b: CGPoint, _ c: CGPoint) -> Double {
        let abx = Double(b.x - a.x), aby = Double(b.y - a.y)
        let bcx = Double(c.x - b.x), bcy = Double(c.y - b.y)
        return abx * bcy - aby * bcx
    }

    private static func polygonArea(_ points: [CGPoint]) -> Double {
        guard points.count > 2 else { return 0 }
        var twiceArea = 0.0
        for index in points.indices {
            let current = points[index]
            let next = points[(index + 1) % points.count]
            twiceArea += Double(current.x) * Double(next.y) - Double(next.x) * Double(current.y)
        }
        return twiceArea / 2
    }
}

/// Projective mapping from the court floor's world (x, y) metres to image pixels.
public struct CourtFloorHomography: Sendable, Equatable {
    public let matrix: Matrix

    public init?(taps: CourtCornerTaps) {
        guard let imageQuad = taps.outerPixelQuad,
              CourtQuadValidator.isValid(taps) else { return nil }
        let set = taps.cornerSet
        let world = set.names.compactMap { set.world($0) }
        guard world.count == 4 else { return nil }
        let image = imageQuad.map { SIMD2(Double($0.x), Double($0.y)) }
        guard let fitted = Self.fit(world: world, image: image) else { return nil }
        matrix = fitted
    }

    public func project(_ world: SIMD2<Double>) -> SIMD2<Double>? {
        let v = matrix * [world.x, world.y, 1]
        guard v.count == 3, v.allSatisfy(\.isFinite), abs(v[2]) > 1e-12 else { return nil }
        return SIMD2(v[0] / v[2], v[1] / v[2])
    }

    public func project(_ world: SIMD3<Double>) -> SIMD2<Double>? { project(world.xy) }

    public func project(line: CourtModel.Line, samples: Int = 20) -> [CGPoint] {
        project(from: line.a, to: line.b, samples: samples)
    }

    public func project(from start: SIMD2<Double>, to end: SIMD2<Double>, samples: Int = 20) -> [CGPoint] {
        let count = max(2, samples)
        return (0..<count).compactMap { index in
            let t = Double(index) / Double(count - 1)
            let xy = start + t * (end - start)
            guard let pixel = project(xy) else { return nil }
            return CGPoint(x: pixel.x, y: pixel.y)
        }
    }

    /// A homography fitted to four non-collinear point pairs using normalized DLT.
    public static func fit(world: [SIMD2<Double>], image: [SIMD2<Double>]) -> Matrix? {
        guard world.count == 4, image.count == 4,
              world.allSatisfy(\.isFinite), image.allSatisfy(\.isFinite) else { return nil }
        guard let (src, srcTransform) = normalize(world),
              let (dst, dstTransform) = normalize(image) else { return nil }
        var a = Matrix(rows: 8, cols: 9)
        for i in 0..<4 {
            let x = src[i].x, y = src[i].y
            let u = dst[i].x, v = dst[i].y
            a[2 * i, 0] = -x; a[2 * i, 1] = -y; a[2 * i, 2] = -1
            a[2 * i, 6] = u * x; a[2 * i, 7] = u * y; a[2 * i, 8] = u
            a[2 * i + 1, 3] = -x; a[2 * i + 1, 4] = -y; a[2 * i + 1, 5] = -1
            a[2 * i + 1, 6] = v * x; a[2 * i + 1, 7] = v * y; a[2 * i + 1, 8] = v
        }
        let h = LinAlg.nullVector(a)
        guard h.count == 9 else { return nil }
        let normalized = Matrix([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]]])
        guard let dstInverse = LinAlg.inverse(dstTransform) else { return nil }
        let denormalized = dstInverse * normalized * srcTransform
        let scale = denormalized[2, 2]
        guard abs(scale) > 1e-12 else { return nil }
        let result = (1 / scale) * denormalized
        return result.data.allSatisfy(\.isFinite) ? result : nil
    }

    private static func normalize(_ points: [SIMD2<Double>]) -> (points: [SIMD2<Double>], transform: Matrix)? {
        guard !points.isEmpty else { return nil }
        let center = points.reduce(SIMD2<Double>(repeating: 0), +) / Double(points.count)
        let meanDistance = points.map { simd_length($0 - center) }.reduce(0, +) / Double(points.count)
        guard meanDistance.isFinite, meanDistance > 1e-12 else { return nil }
        let scale = sqrt(2) / meanDistance
        let t = Matrix([[scale, 0, -scale * center.x], [0, scale, -scale * center.y], [0, 0, 1]])
        return (points.map { scale * ($0 - center) }, t)
    }
}

public enum CourtOverlayGeometry {
    public static func projectedLines(for taps: CourtCornerTaps, samplesPerLine: Int = 20) -> [String: [CGPoint]] {
        guard let homography = CourtFloorHomography(taps: taps) else { return [:] }
        return Dictionary(uniqueKeysWithValues: CourtModel.lines.map {
            ($0.name, homography.project(line: $0, samples: samplesPerLine))
        })
    }

    /// Net centre span on the floor. The painted net fabric is above this line,
    /// but the floor projection remains a useful alignment reference.
    public static func projectedNetLine(for taps: CourtCornerTaps, samples: Int = 20) -> [CGPoint] {
        guard let homography = CourtFloorHomography(taps: taps) else { return [] }
        return homography.project(from: [0, -CourtModel.doublesHalfWidth],
                                  to: [0, CourtModel.doublesHalfWidth], samples: samples)
    }
}
