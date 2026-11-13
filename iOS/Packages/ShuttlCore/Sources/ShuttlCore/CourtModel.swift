import simd

/// BWF badminton court geometry — the single source of truth for every stage.
/// Port of backend/app/services/tracking/court_model.py; keep the two identical.
///
/// World frame (metres, right-handed):
///   origin = court centre on the floor, directly under the net
///   +x     = along the court length, towards the FAR baseline (away from a camera
///            behind the near baseline). Near half: x < 0, far half: x > 0.
///   +y     = across the court, pointing LEFT as seen from behind the near baseline
///   +z     = up
/// Lines are 40 mm wide and belong to the area they bound (touching the line is IN).
/// Coordinates are line centre-lines; use `lineWidth` when judging in/out.
public enum CourtModel {
    public static let halfLength = 6.70
    public static let doublesHalfWidth = 3.05
    public static let singlesHalfWidth = 2.59
    public static let shortServiceX = 1.98
    public static let doublesLongServiceX = 5.94
    public static let lineWidth = 0.04
    public static let netHeightCentre = 1.524
    public static let netHeightPost = 1.55

    public static let courtLength = 2 * halfLength
    public static let doublesWidth = 2 * doublesHalfWidth
    public static let singlesWidth = 2 * singlesHalfWidth

    public struct Line: Sendable, Hashable {
        public let name: String
        public let a: SIMD2<Double>
        public let b: SIMD2<Double>
    }

    /// Every painted floor segment (z = 0), in the same order as the Python dict.
    public static let lines: [Line] = {
        let l = halfLength, w = doublesHalfWidth, s = singlesHalfWidth
        return [
            Line(name: "near_baseline", a: [-l, -w], b: [-l, w]),
            Line(name: "far_baseline", a: [l, -w], b: [l, w]),
            Line(name: "near_long_service", a: [-doublesLongServiceX, -w], b: [-doublesLongServiceX, w]),
            Line(name: "far_long_service", a: [doublesLongServiceX, -w], b: [doublesLongServiceX, w]),
            Line(name: "near_short_service", a: [-shortServiceX, -w], b: [-shortServiceX, w]),
            Line(name: "far_short_service", a: [shortServiceX, -w], b: [shortServiceX, w]),
            Line(name: "left_doubles_side", a: [-l, -w], b: [l, -w]),
            Line(name: "right_doubles_side", a: [-l, w], b: [l, w]),
            Line(name: "left_singles_side", a: [-l, -s], b: [l, -s]),
            Line(name: "right_singles_side", a: [-l, s], b: [l, s]),
            Line(name: "near_centre", a: [-l, 0], b: [-shortServiceX, 0]),
            Line(name: "far_centre", a: [shortServiceX, 0], b: [l, 0]),
        ]
    }()

    public static func line(named name: String) -> Line? { lines.first { $0.name == name } }

    /// Named floor keypoints = every line intersection, in Python's insertion order:
    /// "{nb,nl,ns,fs,fl,fb}_{ld,ls,rs,rd}" then "{nb,ns,fs,fb}_c".
    public static let keypointNames: [String] = {
        var names: [String] = []
        for (xn, _) in xStations { for (yn, _) in yStations { names.append("\(xn)_\(yn)") } }
        for xn in ["nb", "ns", "fs", "fb"] { names.append("\(xn)_c") }
        return names
    }()

    public static let keypoints: [String: SIMD2<Double>] = {
        var out: [String: SIMD2<Double>] = [:]
        let xs = Dictionary(uniqueKeysWithValues: xStations)
        for (xn, xv) in xStations { for (yn, yv) in yStations { out["\(xn)_\(yn)"] = [xv, yv] } }
        for xn in ["nb", "ns", "fs", "fb"] { out["\(xn)_c"] = [xs[xn]!, 0] }
        return out
    }()

    private static let xStations: [(String, Double)] = [
        ("nb", -halfLength), ("nl", -doublesLongServiceX), ("ns", -shortServiceX),
        ("fs", shortServiceX), ("fl", doublesLongServiceX), ("fb", halfLength),
    ]
    private static let yStations: [(String, Double)] = [
        ("ld", -doublesHalfWidth), ("ls", -singlesHalfWidth), ("rs", singlesHalfWidth), ("rd", doublesHalfWidth),
    ]

    /// Net posts stand on the doubles side lines; post tops are 3-D keypoints.
    public static let netPostTops: [SIMD3<Double>] = [
        [0, -doublesHalfWidth, netHeightPost], [0, doublesHalfWidth, netHeightPost],
    ]

    /// (names, world points with z = 0). `nil` names = all keypoints in canonical order.
    public static func keypointsArray(_ names: [String]? = nil) -> (names: [String], points: [SIMD3<Double>]) {
        let n = names ?? keypointNames
        return (n, n.map { let p = keypoints[$0]!; return SIMD3(p.x, p.y, 0) })
    }

    /// Which half a floor x coordinate lies in.
    public static func sideOf(x: Double) -> CourtSide { x < 0 ? .near : .far }

    /// Signed distance (m) from the outer edge of the boundary lines. Positive = inside
    /// (the line counts as in). Singles or doubles side lines; baselines shared.
    public static func inOutMargin(x: Double, y: Double, doubles: Bool = false) -> Double {
        let halfW = doubles ? doublesHalfWidth : singlesHalfWidth
        let edgeX = halfLength + lineWidth / 2
        let edgeY = halfW + lineWidth / 2
        return min(edgeX - abs(x), edgeY - abs(y))
    }
}
