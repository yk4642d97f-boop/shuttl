import Foundation

/// Network canvas size of both detector nets (detector.py NET_W, NET_H).
public let detectorNetWidth = 512
public let detectorNetHeight = 288

/// Aspect-preserving letterbox of a (width, height) frame into the net canvas
/// (port of detector.py ``FrameGeometry``).
///
/// Mapping (continuous coords, pixel-centre convention unless `corner`):
///     net = full * scale + offset          full = (net - offset) / scale
/// With pixel centres, full px centre i maps to net (i + .5) * s - .5 + pad, which is what
/// cv2.resize / INTER_AREA sampling implements, so decoding is unbiased.
/// `corner = true` reproduces WASB's centre-affine convention (net = s*full + pad).
public struct FrameGeometry: Sendable, Equatable, Hashable {
    public let width: Int
    public let height: Int
    public let netWidth: Int
    public let netHeight: Int
    public let corner: Bool

    public init(width: Int, height: Int, netWidth: Int = detectorNetWidth, netHeight: Int = detectorNetHeight,
                corner: Bool = false) {
        precondition(width > 0 && height > 0, "frame size must be positive")
        self.width = width; self.height = height
        self.netWidth = netWidth; self.netHeight = netHeight; self.corner = corner
    }

    public var scale: Double {
        Swift.min(Double(netWidth) / Double(width), Double(netHeight) / Double(height))
    }

    /// Size of the resized image inside the canvas (Python `round` = half to even).
    public var contentSize: (width: Int, height: Int) {
        let s = scale
        return (Swift.min(netWidth, Int((Double(width) * s).rounded(.toNearestOrEven))),
                Swift.min(netHeight, Int((Double(height) * s).rounded(.toNearestOrEven))))
    }

    public var pad: (x: Double, y: Double) {
        if corner {
            let s = scale
            return ((Double(netWidth) - Double(width) * s) / 2.0, (Double(netHeight) - Double(height) * s) / 2.0)
        }
        let (cw, ch) = contentSize
        return (Double((netWidth - cw) / 2), Double((netHeight - ch) / 2))
    }

    /// (sx, sy, ox, oy) with net = full*s + o.
    public var axis: (sx: Double, sy: Double, ox: Double, oy: Double) {
        let (px, py) = pad
        if corner {
            let s = scale
            return (s, s, px, py)
        }
        let (cw, ch) = contentSize
        let sx = Double(cw) / Double(width), sy = Double(ch) / Double(height)
        return (sx, sy, px + 0.5 * sx - 0.5, py + 0.5 * sy - 0.5)
    }

    public func toNet(x: Double, y: Double) -> (x: Double, y: Double) {
        let a = axis
        return (x * a.sx + a.ox, y * a.sy + a.oy)
    }

    public func toFull(x: Double, y: Double) -> (x: Double, y: Double) {
        let a = axis
        return ((x - a.ox) / a.sx, (y - a.oy) / a.sy)
    }

    /// Decoded heatmap peak (net px) → full-res px, removing the net's learned offset
    /// `peakOffset` = (target peak − image content position) in net px (see `NetSpec`).
    public func peakToFull(x: Double, y: Double, peakOffset: Double = 0) -> (x: Double, y: Double) {
        toFull(x: x - peakOffset, y: y - peakOffset)
    }

    /// Affine full → net used by `warp` in corner mode (WASB get_affine_transform + warpAffine).
    public var cornerAffine: (Double, Double, Double, Double, Double, Double) {
        let s = scale
        let (px, py) = pad
        return (s, 0, px, 0, s, py)
    }

    public static func == (a: FrameGeometry, b: FrameGeometry) -> Bool {
        a.width == b.width && a.height == b.height && a.netWidth == b.netWidth && a.netHeight == b.netHeight
            && a.corner == b.corner
    }

    public func hash(into h: inout Hasher) {
        h.combine(width); h.combine(height); h.combine(netWidth); h.combine(netHeight); h.combine(corner)
    }
}

/// Resize filter of the non-corner letterbox (cv2 interpolation flag).
public enum ResizeInterpolation: Sendable, Equatable {
    /// cv2.INTER_AREA (TrackNetV3 default).
    case area
    /// cv2.INTER_LINEAR.
    case linear
}
