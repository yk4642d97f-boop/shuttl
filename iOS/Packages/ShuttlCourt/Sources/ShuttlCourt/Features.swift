/// Per-image line evidence and court-model line samples.
/// Port of court_calibration._Features, _line_samples, _median_image, _to_image.
import Foundation
import simd
import ShuttlCore

/// Tunables (Python module constants and defaults).
public struct CourtCalibrationConfig: Sendable, Equatable {
    /// `_MIN_CONTRAST`: whiteness units a line must exceed its neighbours by.
    public var minContrast: Float = 18
    /// `_MAX_SEGMENTS`: longest Hough segments kept.
    public var maxSegments: Int = 600
    /// `_ABSENT_FOUND`, `_ABSENT_MIN_W`, `_ABSENT_MIN_LEN`, `_EXPLAIN_SIGMA`.
    public var absentFound: Double = 0.15
    public var absentMinWidth: Double = 1.0
    public var absentMinLength: Double = 0.03
    public var explainSigma: Double = 1.0
    /// `MIN_CONFIDENCE`, `MIN_MARGIN`, `MIN_EXPLAINED`.
    public var minConfidence: Double = 0.35
    public var minMargin: Double = 0.05
    public var minExplained: Double = 0.55
    /// `_N_T2_MIN_COVERAGE`: coverage an n_t == 2 camera needs (see `Matcher.valid`).
    public var nT2MinCoverage: Double = 0.70
    /// `FOCAL_HINT_TOL`: max relative focal error of the WINNING camera vs `focalHintPx`
    /// (when given) — a veto only, it never changes which camera wins.
    public var focalHintTolerance: Double = 0.15
    /// `_median_image(max_frames=7)`.
    public var medianMaxFrames: Int = 7
    public init() {}
    public static let `default` = CourtCalibrationConfig()
}

/// Line paint colour model (`LINE_COLOURS`, tried in `allCases` order): the channels the
/// paint is bright in. Swift images are RGB (Python BGR: same channels by name).
public enum LineColour: String, Sendable, CaseIterable {
    case white, yellow, cyan, red
    var rgbChannels: [Int] {
        switch self {
        case .white: return [0, 1, 2]
        case .yellow: return [0, 1]      // R, G   (Python BGR (1, 2): G, R — min/max order-free)
        case .cyan: return [1, 2]        // G, B   (Python (0, 1): B, G)
        case .red: return [0]            // R      (Python (2,))
        }
    }
}

/// `_Features`
struct Features: Sendable {
    let w: Int, h: Int
    let D: Double
    let strength: PlaneF
    let mask: PlaneU8
    let dt: PlaneF          // distance to the nearest line pixel (chamfer)
    let thin: PlaneU8

    init(_ img: ImageU8, config: CourtCalibrationConfig, colour: LineColour = .white) {
        w = img.width; h = img.height
        D = Double(max(w, h))
        let (sh, sv) = ImageOps.ridgeStrength(img, colour: colour)
        var st = [Float](repeating: 0, count: w * h)
        for i in 0..<(w * h) { st[i] = max(sh.data[i], sv.data[i]) }     // np.maximum
        strength = PlaneF(w: w, h: h, data: st)
        let m = ImageOps.maskFromStrength(strength, minLen: 0.012 * D, minContrast: config.minContrast)
        mask = m
        dt = ImageOps.distanceTransform3x3({ m.data[$0] == 0 }, w: w, h: h)
        let dti = ImageOps.distanceTransform3x3({ m.data[$0] != 0 }, w: w, h: h)
        var th = [UInt8](repeating: 0, count: w * h)
        let W = w, H = h
        dti.data.withUnsafeBufferPointer { d in
            for y in 0..<H {
                for x in 0..<W {
                    let i = y * W + x
                    guard m.data[i] > 0 else { continue }
                    let v = d[i]
                    let ok: Bool
                    if sh.data[i] >= sv.data[i] {
                        let l: Float = x > 0 ? d[i - 1] : 0, r: Float = x < W - 1 ? d[i + 1] : 0
                        ok = v >= l && v >= r
                    } else {
                        let u: Float = y > 0 ? d[i - W] : 0, b: Float = y < H - 1 ? d[i + W] : 0
                        ok = v >= u && v >= b
                    }
                    if ok { th[i] = 255 }
                }
            }
        }
        thin = PlaneU8(w: w, h: h, data: th)
    }
}

/// `_line_samples(step, inset)`: points, unit directions and line ids along COURT_LINES.
struct LineSamples: Sendable {
    let P: [SIMD3<Double>]
    let dir: [SIMD3<Double>]
    let ids: [Int]
    /// float32 (X, Y) for `_line_score`.
    let xf: [Float], yf: [Float]

    init(step: Double, inset: Double = 0.08) {
        var P: [SIMD3<Double>] = [], dir: [SIMD3<Double>] = [], ids: [Int] = []
        for (k, line) in CourtModel.lines.enumerated() {
            let a = line.a, b = line.b
            let d = b - a
            let L = (d.x * d.x + d.y * d.y).squareRoot()
            let u = d / L
            let n = max(2, Int(((L - 2 * inset) / step).rounded(.toNearestOrEven)) + 1)
            let start = inset, stop = L - inset
            let stp = (stop - start) / Double(n - 1)
            for i in 0..<n {
                let s = i == n - 1 ? stop : Double(i) * stp + start
                P.append(SIMD3(a.x + s * u.x, a.y + s * u.y, 0))
                dir.append(SIMD3(u.x, u.y, 0))
                ids.append(k)
            }
        }
        self.P = P; self.dir = dir; self.ids = ids
        xf = P.map { Float($0.x) }; yf = P.map { Float($0.y) }
    }

    static let vcoarse = LineSamples(step: 1.0)
    static let coarse = LineSamples(step: 0.3)
    static let medium = LineSamples(step: 0.2)
    static let fine = LineSamples(step: 0.1)
}

/// `_IS_TRANSVERSE`
let isTransverse: [Bool] = CourtModel.lines.map { abs($0.a.x - $0.b.x) < 1e-9 }
/// `_KP`
let courtKeypointsXY: [SIMD2<Double>] = CourtModel.keypointNames.map { CourtModel.keypoints[$0]! }

enum FrameOps {
    /// `_median_image`: temporal median of up to `maxFrames` evenly spaced frames.
    static func medianImage(_ frames: [ImageU8], maxFrames: Int = 7) -> ImageU8? {
        guard let first = frames.first else { return nil }
        if frames.count <= 2 { return first }
        let n = min(maxFrames, frames.count)
        let span = Double(frames.count - 1)
        let idx = (0..<n).map { i -> Int in
            let v = n == 1 ? 0 : (i == n - 1 ? span : Double(i) * (span / Double(n - 1)))
            return Int(v.rounded(.toNearestOrEven))
        }
        let k = idx.count / 2
        let count = first.data.count
        precondition(frames.allSatisfy { $0.data.count == count && $0.width == first.width }, "frames differ in size")
        var out = [UInt8](repeating: 0, count: count)
        var buf = [UInt8](repeating: 0, count: idx.count)
        let srcs = idx.map { frames[$0].data }
        for p in 0..<count {
            for (j, s) in srcs.enumerated() { buf[j] = s[p] }
            // np.partition(stack, k)[k] = k-th smallest
            buf.sort()
            out[p] = buf[k]
        }
        return ImageU8(width: first.width, height: first.height, channels: first.channels, data: out)
    }

    /// `_to_image`: median, then INTER_AREA resize to `size` if needed.
    static func toImage(_ frames: [ImageU8], size: ImageSize, maxFrames: Int = 7) -> ImageU8? {
        guard let img = medianImage(frames, maxFrames: maxFrames) else { return nil }
        if img.width != size.width || img.height != size.height {
            return img.resized(width: size.width, height: size.height, method: .area)
        }
        return img
    }
}
