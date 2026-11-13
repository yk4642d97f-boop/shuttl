import CoreGraphics
import Foundation
import ShuttlCore

/// Manual corner taps ⇄ pipeline inputs. Keys are CourtModel keypoint names, as used by
/// ShuttlCalibrationUI's `CourtCornerTaps`: "nb_ld", "nb_rd", "fb_ld", "fb_rd" (+ optional
/// "ns_ld", "ns_rd", …). Remember "ld" is world y < 0 = the RIGHT side seen from the near baseline.
public enum CourtCorners {
    public static let outerNames = ["nb_ld", "nb_rd", "fb_ld", "fb_rd"]

    /// Normalised taps (0…1 of the upright frame) → `PipelineOptions.manualCorners` with
    /// `cornersNormalized = true`. Unknown names and non-finite / out-of-frame points are dropped.
    public static func manualCorners(_ taps: [String: CGPoint]) -> [String: SIMD2<Double>] {
        var out: [String: SIMD2<Double>] = [:]
        for (name, p) in taps where CourtModel.keypoints[name] != nil {
            let x = Double(p.x), y = Double(p.y)
            guard x.isFinite, y.isFinite, (0...1).contains(x), (0...1).contains(y) else { continue }
            out[name] = SIMD2(x, y)
        }
        return out
    }

    /// Same, in pixels of `size` (for `CourtCalibration.calibrateFromPoints` / `cornersNormalized = false`).
    public static func manualCornersPixels(_ taps: [String: CGPoint], size: ImageSize) -> [String: SIMD2<Double>] {
        manualCorners(taps).mapValues { SIMD2($0.x * Double(size.width), $0.y * Double(size.height)) }
    }

    /// True when all four outer corners are present (what `calibrateFromPoints` needs at minimum).
    public static func isComplete(_ taps: [String: CGPoint]) -> Bool {
        let m = manualCorners(taps)
        return outerNames.allSatisfy { m[$0] != nil }
    }

    /// Pre-fill the manual tap screen from a found court (corners inside the frame only).
    public static func taps(from projection: CourtProjection) -> [String: CGPoint] {
        projection.corners.filter { (0...1).contains($0.value.x) && (0...1).contains($0.value.y) }
    }
}
