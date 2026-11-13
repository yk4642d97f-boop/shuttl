import CoreGraphics
import Foundation
import ShuttlCore

/// The four manually marked court corners, in normalized coordinates of the
/// upright full-resolution frame. Optional service-line points are accepted by
/// the downstream calibrator when present.
public struct CourtCornerTaps: Codable, Sendable, Equatable {
    public var points: [String: CGPoint]
    public var imageSize: ImageSize

    public init(points: [String: CGPoint], imageSize: ImageSize) {
        self.points = points
        self.imageSize = imageSize
    }

    public init(imageSize: ImageSize, points: [String: CGPoint] = [:]) {
        self.imageSize = imageSize
        self.points = points
    }

    public static let outerNames = ["nb_ld", "nb_rd", "fb_ld", "fb_rd"]
    public static let optionalServiceNames = ["ns_ld", "ns_rd"]
    public static let requiredNames = outerNames

    public subscript(_ name: String) -> CGPoint? {
        get { points[name] }
        set { points[name] = newValue }
    }

    /// Complete for whichever corner set is active (baseline corners, or near service line + far baseline).
    public var isComplete: Bool {
        cornerSet.names.allSatisfy { Self.isValidNormalizedPoint(points[$0]) }
    }

    /// Which four keypoints this calibration uses: near service-line corners stand in for
    /// the near baseline ones when those are out of frame (tripod close behind the baseline).
    public var cornerSet: CalibrationCornerSet {
        let service = CalibrationCornerSet.nearServiceLine.names.allSatisfy { Self.isValidNormalizedPoint(points[$0]) }
        let baseline = ["nb_ld", "nb_rd"].allSatisfy { Self.isValidNormalizedPoint(points[$0]) }
        return service && !baseline ? .nearServiceLine : .baseline
    }

    /// Only the active set's points: what goes to the solver and the store.
    public var activePoints: [String: CGPoint] {
        points.filter { cornerSet.names.contains($0.key) }
    }

    public var pixelPoints: [String: CGPoint] {
        Dictionary(uniqueKeysWithValues: points.compactMap { name, point in
            guard Self.isValidNormalizedPoint(point) else { return nil }
            return (name, CGPoint(x: point.x * CGFloat(imageSize.width), y: point.y * CGFloat(imageSize.height)))
        })
    }

    public func normalized(to size: ImageSize) -> CourtCornerTaps {
        CourtCornerTaps(points: points, imageSize: size)
    }

    static func isValidNormalizedPoint(_ point: CGPoint?) -> Bool {
        guard let point else { return false }
        return point.x.isFinite && point.y.isFinite && (0...1).contains(point.x) && (0...1).contains(point.y)
    }
}

/// The four court keypoints the user drags.
public enum CalibrationCornerSet: String, Codable, Sendable, CaseIterable {
    /// The doubles court's four outer corners.
    case baseline
    /// Near short-service-line corners + far baseline corners, for when the near baseline is
    /// at or past the bottom of the frame.
    case nearServiceLine

    /// Near-right, near-left, far-right, far-left (the `ld` side is the camera's right).
    public var names: [String] {
        switch self {
        case .baseline: return ["nb_ld", "nb_rd", "fb_ld", "fb_rd"]
        case .nearServiceLine: return ["ns_ld", "ns_rd", "fb_ld", "fb_rd"]
        }
    }

    /// World floor position (metres) of each name, from ShuttlCore.CourtModel.
    public func world(_ name: String) -> SIMD2<Double>? { CourtModel.keypoints[name] }
}

public extension CourtCornerTaps {
    /// Pixel coordinates in the image's top-left origin convention, in `cornerSet.names` order.
    var outerPixelQuad: [CGPoint]? {
        guard isComplete else { return nil }
        return cornerSet.names.compactMap { points[$0] }.map {
            CGPoint(x: $0.x * CGFloat(imageSize.width), y: $0.y * CGFloat(imageSize.height))
        }
    }

    /// The same court with the other corner set: the new points are projected through the
    /// current quad when it is valid, otherwise slid along the side lines, and clamped to the frame.
    func switched(to set: CalibrationCornerSet) -> CourtCornerTaps {
        guard set != cornerSet else { return self }
        var out = CourtCornerTaps(imageSize: imageSize, points: activePoints)
        let homography = CourtFloorHomography(taps: self)
        let nearFrom = cornerSet == .baseline ? ["nb_ld", "nb_rd"] : ["ns_ld", "ns_rd"]
        let nearTo = set == .baseline ? ["nb_ld", "nb_rd"] : ["ns_ld", "ns_rd"]
        for (from, to) in zip(nearFrom, nearTo) {
            out.points[from] = nil
            var target: CGPoint?
            if let homography, let world = set.world(to), let px = homography.project(world) {
                target = CGPoint(x: px.x / Double(imageSize.width), y: px.y / Double(imageSize.height))
            } else if let near = points[from], let far = points[from.replacingOccurrences(of: from.prefix(2), with: "fb")] {
                // baseline → service line is 4.72 m of the 13.4 m side line; the reverse extrapolates.
                let t: CGFloat = set == .nearServiceLine ? 4.72 / 13.4 : -4.72 / 8.68
                target = CGPoint(x: near.x + (far.x - near.x) * t, y: near.y + (far.y - near.y) * t)
            }
            if let t = target, t.x.isFinite, t.y.isFinite {
                out.points[to] = CGPoint(x: min(max(t.x, 0), 1), y: min(max(t.y, 0), 1))
            }
        }
        return out
    }
}

/// Labels are kept in this package so the view does not import the app target.
public enum CourtCornerLabel {
    public static func chinese(for name: String) -> String {
        switch name {
        case "nb_ld": return "近端底线右角"
        case "nb_rd": return "近端底线左角"
        case "fb_ld": return "远端底线右角"
        case "fb_rd": return "远端底线左角"
        case "ns_ld": return "近端发球线右角"
        case "ns_rd": return "近端发球线左角"
        default: return name
        }
    }
}
