import Foundation
import ShuttlCore

/// Layout of the far-half crop passes of the Vision detector (the full-frame pass always runs).
///
/// Vision resizes every region to its fixed network input, so whether a far player is found
/// depends on the region's size relative to the player, k = √(crop area) / player px — not on
/// the player's absolute size (an ultra-wide phone's 78 px far player behaves like the main
/// camera's 165 px one). ShuttlPlayersBench --layouts, far detection recall vs YOLO tracks:
///
///                         none         single       lateral      depth
///   broadcast (k)         0.38 (6.5)   0.94 (4.3)   0.96 (3.2)   0.93 (4.0)
///   phone tripod (k)      0.07–0.46    0.55–0.59    0.96–0.97    0.78–0.81
///                         (8.1–9.7)    (5.1–5.7)    (3.8–4.3)    (4.9–5.7)
///   Vision passes         1            2            3            3
///
/// A phone behind the near baseline sees the far half nearly as wide as the frame, so one
/// crop barely magnifies; splitting it left/right does. Default: `.lateral`
/// (`PlayerTracker.defaultFarLayout`).
public enum FarCropLayout: String, Sendable, CaseIterable {
    /// Full frame only.
    case none
    /// One crop over the whole far half (players.py `_far_roi`).
    case single
    /// Left and right halves of the far half, overlapping across the centre line.
    case lateral
    /// Front (net → mid court) and back (mid court → baseline) of the far half.
    case depth
}

public enum FarCrops {
    /// World box of a crop: x (along the court), y (across), z from 0 to 2.6 m.
    public struct WorldBox: Sendable, Equatable {
        public var x: ClosedRange<Double>
        public var y: ClosedRange<Double>
        public var zTop = 2.6
    }

    /// Lateral overlap (m) either side of the centre line: a player plus racket reach
    /// straddling y = 0 is whole in at least one crop.
    public static let lateralOverlap = 0.6
    /// Depth split line (m past the net) and overlap either side of it.
    public static let depthSplitX = 3.3
    public static let depthOverlap = 0.6

    /// World boxes of a layout, in merge order (the later crop wins an overlap).
    public static func worldBoxes(_ layout: FarCropLayout, courtMargin: Double = PlayersConfig.default.courtMargin) -> [WorldBox] {
        let x0 = -0.3, x1 = CourtModel.halfLength + courtMargin
        let y0 = -CourtModel.doublesHalfWidth - courtMargin, y1 = CourtModel.doublesHalfWidth + courtMargin
        switch layout {
        case .none: return []
        case .single: return [WorldBox(x: x0...x1, y: y0...y1)]
        case .lateral:
            return [WorldBox(x: x0...x1, y: y0...lateralOverlap), WorldBox(x: x0...x1, y: -lateralOverlap...y1)]
        case .depth:
            return [WorldBox(x: x0...(depthSplitX + depthOverlap), y: y0...y1),
                    WorldBox(x: (depthSplitX - depthOverlap)...x1, y: y0...y1)]
        }
    }

    /// Bounding rect of a world box's 8 corners in image px (players._far_roi rule:
    /// clipped to the image, integer truncation, nil when non-finite or < 32 px).
    /// `clip: false` returns the raw bounds (Double) for callers that map rects.
    public static func imageBounds(_ b: WorldBox, camera: CameraModel)
        -> (x1: Double, y1: Double, x2: Double, y2: Double)? {
        var pts: [SIMD3<Double>] = []
        for x in [b.x.lowerBound, b.x.upperBound] {
            for y in [b.y.lowerBound, b.y.upperBound] { for z in [0.0, b.zTop] { pts.append([x, y, z]) } }
        }
        let uv = camera.project(pts)
        guard uv.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else { return nil }
        return (uv.map(\.x).min()!, uv.map(\.y).min()!, uv.map(\.x).max()!, uv.map(\.y).max()!)
    }

    public static func rect(_ b: WorldBox, camera: CameraModel, size: ImageSize) -> PixelRect? {
        guard let r = imageBounds(b, camera: camera) else { return nil }
        let W = Double(size.width), H = Double(size.height)
        let x1 = Int(min(max(r.x1, 0), W)), y1 = Int(min(max(r.y1, 0), H))
        let x2 = Int(min(max(r.x2, 0), W)), y2 = Int(min(max(r.y2, 0), H))
        if x2 - x1 < 32 || y2 - y1 < 32 { return nil }
        return PixelRect(x1: x1, y1: y1, x2: x2, y2: y2)
    }

    /// Crop rects of a layout for a camera (unusable crops dropped). `.single` equals
    /// `FarROI.compute(...).rect`.
    public static func rects(_ layout: FarCropLayout, camera: CameraModel, size: ImageSize,
                             courtMargin: Double = PlayersConfig.default.courtMargin) -> [PixelRect] {
        worldBoxes(layout, courtMargin: courtMargin).compactMap { rect($0, camera: camera, size: size) }
    }
}
