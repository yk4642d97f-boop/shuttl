import Foundation
import ShuttlCore

/// Was the phone still? Every stage after detection assumes a fixed camera (the detector's
/// background model, one court calibration for the whole video), so a hand-held or bumped
/// recording gives confident nonsense instead of an error. Measured on the calibration frames:
/// each frame's shift against their median, on gradient images at low resolution.
///
/// Measured (prototype, 2026-09-28): 13 tripod / broadcast clips → 0 px shift, correlation ≥ 0.95;
/// the user's hand-held recording of a screen → median 21.5 px, p90 46.6 px, correlation 0.44–0.45.
public struct CameraSteadiness: Sendable, Equatable, Codable {
    /// Where the pipeline stores it: diagnostics["camera_motion"], and a checkpoint of the same name.
    public static let diagnosticsKey = "camera_motion"

    /// Per sampled frame, shift in full-resolution px (search-limited, see `searchPx`).
    public var shiftsPx: [Double]
    /// Per sampled frame, normalised correlation at the best shift (1 = identical structure).
    public var correlations: [Double]
    public var imageWidth: Int
    /// Largest shift the search can see, in full-resolution px (bigger moves read as a low correlation).
    public var searchPx: Double

    public init(shiftsPx: [Double], correlations: [Double], imageWidth: Int, searchPx: Double) {
        self.shiftsPx = shiftsPx; self.correlations = correlations; self.imageWidth = imageWidth; self.searchPx = searchPx
    }

    /// Frames where the view is not where it is most of the time.
    public func unsteadyFraction(policy: SteadinessPolicy = .default) -> Double {
        guard !shiftsPx.isEmpty else { return 0 }
        let limit = policy.steadyShiftFraction * Double(imageWidth)
        let bad = zip(shiftsPx, correlations).filter { $0 > limit || $1 < policy.minCorrelation }.count
        return Double(bad) / Double(shiftsPx.count)
    }

    public func verdict(policy: SteadinessPolicy = .default) -> SteadinessVerdict {
        guard shiftsPx.count >= 3 else { return .unknown }
        let f = unsteadyFraction(policy: policy)
        if f >= policy.unusableFraction { return .handHeld }
        if f >= policy.warnFraction { return .moved }
        return .steady
    }

    public var json: JSONValue {
        .object(["shifts_px": .array(shiftsPx.map { .number(($0 * 10).rounded() / 10) }),
                 "correlations": .array(correlations.map { .number(($0 * 1000).rounded() / 1000) }),
                 "image_width": .number(Double(imageWidth)), "search_px": .number(searchPx),
                 "unsteady_fraction": .number(unsteadyFraction()), "verdict": .string(verdict().rawValue)])
    }
}

public enum SteadinessVerdict: String, Sendable, Equatable, Codable {
    /// On a tripod: every result can be trusted as far as the camera goes.
    case steady
    /// The phone moved part of the time (bumped, tripod slipped): results kept, with a warning.
    case moved
    /// Hand-held or constantly moving: speeds, landings and calls would be invented.
    case handHeld = "hand_held"
    /// Too few frames to say.
    case unknown
}

public struct SteadinessPolicy: Sendable, Equatable {
    /// A frame shifted by more than this share of the image width is "moved" (1080p: ≈ 9.6 px).
    public var steadyShiftFraction: Double
    /// Below this correlation the view has changed beyond the search (a big move or a new scene).
    public var minCorrelation: Double
    /// At least this share of moved frames → unusable.
    public var unusableFraction: Double
    /// At least this share → keep the result but warn.
    public var warnFraction: Double

    public init(steadyShiftFraction: Double = 0.005, minCorrelation: Double = 0.6,
                unusableFraction: Double = 0.3, warnFraction: Double = 0.1) {
        self.steadyShiftFraction = steadyShiftFraction; self.minCorrelation = minCorrelation
        self.unusableFraction = unusableFraction; self.warnFraction = warnFraction
    }

    public static let `default` = SteadinessPolicy()
}

public enum CameraSteadinessEstimator {
    /// Working width of the gradient images; the search is ±`search` px at this width.
    static let workWidth = 320
    static let search = 8

    /// nil with fewer than 3 frames or frames of different sizes.
    public static func estimate(_ frames: [ImageU8]) -> CameraSteadiness? {
        guard frames.count >= 3, let first = frames.first,
              frames.allSatisfy({ $0.width == first.width && $0.height == first.height }) else { return nil }
        let w = workWidth, h = max(2 * search + 8, Int((Double(first.height) * Double(w) / Double(first.width)).rounded()))
        let scale = Double(first.width) / Double(w)
        let small = frames.map { downscaleLuma($0, to: w, h) }
        let reference = gradient(median(small, w, h), w, h)
        var shifts: [Double] = [], correlations: [Double] = []
        for img in small {
            let (ncc, dx, dy) = bestShift(gradient(img, w, h), reference, w, h)
            shifts.append((Double(dx * dx + dy * dy)).squareRoot() * scale)
            correlations.append(ncc)
        }
        return CameraSteadiness(shiftsPx: shifts, correlations: correlations, imageWidth: first.width,
                                searchPx: Double(search) * scale)
    }

    /// Area-averaged luma (BT.601) of an RGB or grey image.
    static func downscaleLuma(_ img: ImageU8, to w: Int, _ h: Int) -> [Float] {
        var out = [Float](repeating: 0, count: w * h)
        let c = img.channels
        img.data.withUnsafeBufferPointer { src in
            for y in 0..<h {
                let y0 = y * img.height / h, y1 = max(y0 + 1, (y + 1) * img.height / h)
                for x in 0..<w {
                    let x0 = x * img.width / w, x1 = max(x0 + 1, (x + 1) * img.width / w)
                    var sum: Float = 0
                    for yy in y0..<y1 {
                        var i = (yy * img.width + x0) * c
                        for _ in x0..<x1 {
                            sum += c >= 3 ? 0.299 * Float(src[i]) + 0.587 * Float(src[i + 1]) + 0.114 * Float(src[i + 2])
                                          : Float(src[i])
                            i += c
                        }
                    }
                    out[y * w + x] = sum / Float((y1 - y0) * (x1 - x0))
                }
            }
        }
        return out
    }

    static func median(_ imgs: [[Float]], _ w: Int, _ h: Int) -> [Float] {
        var out = [Float](repeating: 0, count: w * h)
        var column = [Float](repeating: 0, count: imgs.count)
        for i in 0..<(w * h) {
            for k in imgs.indices { column[k] = imgs[k][i] }
            column.sort()
            out[i] = column[column.count / 2]
        }
        return out
    }

    /// |∂x| + |∂y|: court lines and edges, insensitive to exposure changes.
    static func gradient(_ g: [Float], _ w: Int, _ h: Int) -> [Float] {
        var out = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            for x in 0..<w {
                let i = y * w + x
                let gx = x > 0 ? abs(g[i] - g[i - 1]) : 0
                let gy = y > 0 ? abs(g[i] - g[i - w]) : 0
                out[i] = gx + gy
            }
        }
        return out
    }

    /// Integer shift (dx, dy) of `img` that best matches `ref` by normalised cross-correlation
    /// over the central window.
    static func bestShift(_ img: [Float], _ ref: [Float], _ w: Int, _ h: Int) -> (Double, Int, Int) {
        let s = search
        let ww = w - 2 * s, hh = h - 2 * s
        let n = Float(ww * hh)
        var a = [Float](repeating: 0, count: ww * hh)
        for y in 0..<hh { for x in 0..<ww { a[y * ww + x] = ref[(y + s) * w + x + s] } }
        let aMean = a.reduce(0, +) / n
        var aNorm: Float = 0
        for i in a.indices { a[i] -= aMean; aNorm += a[i] * a[i] }
        aNorm = aNorm.squareRoot()
        var best: (Double, Int, Int) = (-2, 0, 0)
        for dy in -s...s {
            for dx in -s...s {
                var sb: Float = 0, sbb: Float = 0, sab: Float = 0
                for y in 0..<hh {
                    let row = (y + s + dy) * w + s + dx
                    let arow = y * ww
                    for x in 0..<ww {
                        let b = img[row + x]
                        sb += b; sbb += b * b; sab += a[arow + x] * b
                    }
                }
                let bVar = sbb - sb * sb / n
                guard bVar > 1e-6, aNorm > 1e-6 else { continue }
                let ncc = Double(sab / (aNorm * bVar.squareRoot()))
                // Ties go to the smallest move, so a featureless frame reads as "not moved".
                if ncc > best.0 + 1e-9 || (abs(ncc - best.0) <= 1e-9 && dx * dx + dy * dy < best.1 * best.1 + best.2 * best.2) {
                    best = (ncc, dx, dy)
                }
            }
        }
        return best
    }
}
