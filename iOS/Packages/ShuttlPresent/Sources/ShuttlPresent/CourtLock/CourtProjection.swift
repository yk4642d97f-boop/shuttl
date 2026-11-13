import CoreGraphics
import Foundation
import simd
import ShuttlCore

/// The court as it appears in the upright camera frame, in NORMALISED coordinates
/// (x: 0 = left edge … 1 = right edge, y: 0 = top … 1 = bottom), ready to draw over the preview.
/// Lines are sampled every 25 cm so lens distortion bends them like the real paint; a line is
/// split where it leaves the (slightly enlarged) frame or passes behind the camera.
public struct CourtProjection: Sendable, Equatable {
    public enum LineKind: String, Sendable, Equatable, CaseIterable {
        /// Baselines + doubles side lines (the outer boundary).
        case boundary
        /// Singles side lines.
        case singlesSide
        /// Short service + doubles long service lines.
        case service
        /// Centre lines.
        case centre
        /// Top of the net tape (posts 1.55 m, centre 1.524 m).
        case netTop
        /// Net posts (floor → 1.55 m).
        case netPost
    }

    public struct Line: Sendable, Equatable {
        /// CourtModel line name ("near_baseline", …), or "net_top" / "net_post_left" / "net_post_right".
        public var name: String
        public var kind: LineKind
        /// One or more polylines (split at frame exits), normalised.
        public var polylines: [[CGPoint]]
    }

    public var lines: [Line]
    /// Outer corners by CourtModel keypoint name: "nb_ld", "nb_rd", "fb_ld", "fb_rd" (normalised;
    /// may lie outside 0…1 when a corner is out of frame; absent if behind the camera).
    /// Note "ld" = world y < 0 = the RIGHT side as seen from behind the near baseline.
    public var corners: [String: CGPoint]
    /// The four outer corners in drawing order near-left, near-right, far-right, far-left
    /// (as seen from the camera), nil when one is behind the camera.
    public var outline: [CGPoint]?
    /// Upright frame size the projection was made for (px).
    public var frameSize: ImageSize
    /// Framing quality of this camera (what could be better, even though the court is found).
    public var framing: FramingCheck

    public init(camera: CameraModel, sampleStepM: Double = 0.25) {
        let w = Double(camera.imageSize.width), h = Double(camera.imageSize.height)
        let proj = SafeProjector(camera: camera)
        func norm(_ uv: SIMD2<Double>) -> CGPoint { CGPoint(x: uv.x / w, y: uv.y / h) }
        var out: [Line] = []
        for l in CourtModel.lines {
            let kind: LineKind
            switch l.name {
            case "near_baseline", "far_baseline", "left_doubles_side", "right_doubles_side": kind = .boundary
            case "left_singles_side", "right_singles_side": kind = .singlesSide
            case "near_centre", "far_centre": kind = .centre
            default: kind = .service
            }
            let a = SIMD3(l.a.x, l.a.y, 0), b = SIMD3(l.b.x, l.b.y, 0)
            out.append(Line(name: l.name, kind: kind, polylines: proj.polylines(a, b, step: sampleStepM).map { $0.map(norm) }))
        }
        // Net tape: parabola from post height at the side lines to centre height at y = 0.
        let W = CourtModel.doublesHalfWidth, hp = CourtModel.netHeightPost, hc = CourtModel.netHeightCentre
        let netPts = stride(from: -W, through: W + 1e-9, by: sampleStepM).map { y -> SIMD3<Double> in
            SIMD3(0, y, hc + (hp - hc) * (y * y) / (W * W))
        }
        out.append(Line(name: "net_top", kind: .netTop, polylines: proj.polyline(netPts).map { $0.map(norm) }))
        for (name, y) in [("net_post_right", -W), ("net_post_left", W)] {
            let pts = stride(from: 0.0, through: hp + 1e-9, by: sampleStepM / 2).map { SIMD3(0, y, $0) } + [SIMD3(0, y, hp)]
            out.append(Line(name: name, kind: .netPost, polylines: proj.polyline(pts).map { $0.map(norm) }))
        }
        lines = out
        var c: [String: CGPoint] = [:]
        for name in ["nb_ld", "nb_rd", "fb_ld", "fb_rd"] {
            let p = CourtModel.keypoints[name]!
            let uv = proj.point(SIMD3(p.x, p.y, 0))
            if uv.x.isFinite && uv.y.isFinite { c[name] = norm(uv) }
        }
        corners = c
        // Seen from behind the near baseline, +y is LEFT: near-left = nb_rd (y = +W).
        let order = ["nb_rd", "nb_ld", "fb_ld", "fb_rd"]
        let o = order.compactMap { c[$0] }
        outline = o.count == 4 ? o : nil
        frameSize = camera.imageSize
        framing = FramingCheck(camera: camera)
    }

    public func line(named name: String) -> Line? { lines.first { $0.name == name } }
}

/// Projection that refuses points behind the camera or where the lens model folds back.
struct SafeProjector {
    let camera: CameraModel
    let w: Double, h: Double
    /// Max normalised radius² where the k1…k3 radial model is still monotonic.
    let r2Max: Double

    init(camera: CameraModel) {
        self.camera = camera
        w = Double(camera.imageSize.width); h = Double(camera.imageSize.height)
        let d = camera.dist + Array(repeating: 0, count: max(0, 5 - camera.dist.count))
        // d/dr [r (1 + k1 r² + k2 r⁴ + k3 r⁶)] = 1 + 3k1 r² + 5k2 r⁴ + 7k3 r⁶ must stay > 0.2.
        var lim = 4.0                                    // r = 2 (≈ 127° FOV) is beyond any phone lens
        var r2 = 0.0
        while r2 < 4.0 {
            let dr = 1 + 3 * d[0] * r2 + 5 * d[1] * r2 * r2 + 7 * d[4] * r2 * r2 * r2
            if dr < 0.2 { lim = r2; break }
            r2 += 0.01
        }
        r2Max = lim
    }

    /// Pixel of a world point, NaN when behind the camera / outside the valid lens model.
    func point(_ p: SIMD3<Double>) -> SIMD2<Double> {
        let cam = camera.R * p + camera.t
        guard cam.z > 1e-6 else { return SIMD2(.nan, .nan) }
        let x = cam.x / cam.z, y = cam.y / cam.z
        guard x * x + y * y <= r2Max else { return SIMD2(.nan, .nan) }
        return camera.project(p)
    }

    /// Inside the frame enlarged by 25 % on each side (lines are clipped by the view anyway).
    func usable(_ uv: SIMD2<Double>) -> Bool {
        uv.x.isFinite && uv.y.isFinite && uv.x >= -0.25 * w && uv.x <= 1.25 * w && uv.y >= -0.25 * h && uv.y <= 1.25 * h
    }

    func polylines(_ a: SIMD3<Double>, _ b: SIMD3<Double>, step: Double) -> [[SIMD2<Double>]] {
        let L = simd_distance(a, b)
        let n = max(2, Int((L / step).rounded(.up)) + 1)
        let pts = (0..<n).map { i in a + (b - a) * (Double(i) / Double(n - 1)) }
        return polyline(pts)
    }

    func polyline(_ pts: [SIMD3<Double>]) -> [[SIMD2<Double>]] {
        var out: [[SIMD2<Double>]] = [], cur: [SIMD2<Double>] = []
        for p in pts {
            let uv = point(p)
            if usable(uv) { cur.append(uv) } else {
                if cur.count >= 2 { out.append(cur) }
                cur = []
            }
        }
        if cur.count >= 2 { out.append(cur) }
        return out
    }
}

/// How well a solved camera frames the court (all geometry, no guessing).
public struct FramingCheck: Sendable, Equatable {
    /// Roll in degrees (`CameraModel.rollDegrees`).
    public var rollDegrees: Double
    /// Area of the outer court quad / frame area (0 when unknown).
    public var courtAreaFraction: Double
    public var farBaselineInFrame: Bool
    public var nearBaselineInFrame: Bool
    public var leftSideInFrame: Bool
    public var rightSideInFrame: Bool
    /// Every applicable framing hint, most important first.
    public var hints: [CourtLockHint]
    /// The one to show (nil = framing is fine).
    public var advice: CourtLockHint? { hints.first }

    public static let maxRollDegrees = 4.0
    /// Court quad smaller than this share of the frame → "come closer".
    public static let minCourtAreaFraction = 0.10
    /// A line must be at least this far inside the frame edge (fraction of the side).
    public static let edgeMargin = 0.01

    public init(camera: CameraModel) {
        let proj = SafeProjector(camera: camera)
        let w = Double(camera.imageSize.width), h = Double(camera.imageSize.height)
        func inFrame(_ uv: SIMD2<Double>) -> Bool {
            uv.x.isFinite && uv.y.isFinite && uv.x >= Self.edgeMargin * w && uv.x <= (1 - Self.edgeMargin) * w
                && uv.y >= Self.edgeMargin * h && uv.y <= (1 - Self.edgeMargin) * h
        }
        func lineInFrame(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Bool {
            (0...10).allSatisfy { i in
                let p = a + (b - a) * (Double(i) / 10)
                return inFrame(proj.point(SIMD3(p.x, p.y, 0)))
            }
        }
        let L = CourtModel.halfLength, W = CourtModel.doublesHalfWidth
        rollDegrees = camera.rollDegrees
        farBaselineInFrame = lineInFrame([L, -W], [L, W])
        nearBaselineInFrame = lineInFrame([-L, -W], [-L, W])
        // Seen from behind the near baseline, the left side line is y = +W.
        leftSideInFrame = lineInFrame([-L, W], [L, W])
        rightSideInFrame = lineInFrame([-L, -W], [L, -W])
        let quad = [SIMD2(-L, W), SIMD2(-L, -W), SIMD2(L, -W), SIMD2(L, W)].map { proj.point(SIMD3($0.x, $0.y, 0)) }
        if quad.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) {
            var a = 0.0
            for i in 0..<4 { let p = quad[i], q = quad[(i + 1) % 4]; a += p.x * q.y - q.x * p.y }
            courtAreaFraction = min(1, abs(a) / 2 / (w * h))
        } else {
            courtAreaFraction = 0
        }
        var hs: [CourtLockHint] = []
        if !leftSideInFrame && !rightSideInFrame { hs.append(.bothSidesOutOfFrame) }
        if !farBaselineInFrame { hs.append(.farBaselineOutOfFrame) }
        if leftSideInFrame != rightSideInFrame { hs.append(leftSideInFrame ? .rightSideOutOfFrame : .leftSideOutOfFrame) }
        if !nearBaselineInFrame { hs.append(.nearBaselineOutOfFrame) }
        if courtAreaFraction > 0 && courtAreaFraction < Self.minCourtAreaFraction { hs.append(.courtTooSmall) }
        if abs(rollDegrees) > Self.maxRollDegrees { hs.append(.tilted) }
        hints = hs
    }
}

/// Where normalised upright-frame points land in a view that shows the video with
/// aspect-fit (`.resizeAspect`) or aspect-fill (`.resizeAspectFill`) gravity.
public struct VideoGeometry: Sendable, Equatable {
    public enum Gravity: Sendable, Equatable { case fit, fill }
    public var videoRect: CGRect

    public init(videoSize: CGSize, in container: CGSize, gravity: Gravity) {
        guard videoSize.width > 0, videoSize.height > 0 else { videoRect = CGRect(origin: .zero, size: container); return }
        let sx = container.width / videoSize.width, sy = container.height / videoSize.height
        let s = gravity == .fit ? min(sx, sy) : max(sx, sy)
        let size = CGSize(width: videoSize.width * s, height: videoSize.height * s)
        videoRect = CGRect(x: (container.width - size.width) / 2, y: (container.height - size.height) / 2,
                           width: size.width, height: size.height)
    }

    /// Normalised (0…1 of the upright frame) → view point.
    public func point(_ n: CGPoint) -> CGPoint {
        CGPoint(x: videoRect.minX + n.x * videoRect.width, y: videoRect.minY + n.y * videoRect.height)
    }

    /// View point → normalised (for taps).
    public func normalized(_ p: CGPoint) -> CGPoint {
        guard videoRect.width > 0, videoRect.height > 0 else { return .zero }
        return CGPoint(x: (p.x - videoRect.minX) / videoRect.width, y: (p.y - videoRect.minY) / videoRect.height)
    }
}
