import Foundation
import CoreGraphics
import simd

// MARK: - CourtMapper (legacy)
// Thin pixel → floor-metres mapper kept for old call sites. New code should use
// `CourtCalibration` (full camera, backend court frame, origin at court centre).

struct CourtMapper: Sendable {
    /// 3×3 row-major homography: image pixels → floor metres (X, Y).
    let homographyMatrix: [[Double]]

    init(homographyMatrix: [[Double]]) {
        self.homographyMatrix = homographyMatrix
    }

    /// Pixels → floor of `calibration` (inverse of its floor homography, no distortion).
    init(calibration: CourtCalibration) {
        homographyMatrix = calibration.floorHomography.inverse.rowArrays
    }

    /// Normalised image point (0–1) → floor metres.
    func imageToMeters(_ point: CGPoint, imageSize: CGSize) -> CGPoint? {
        applyHomography(x: Double(point.x) * Double(imageSize.width),
                        y: Double(point.y) * Double(imageSize.height))
    }

    func imagePixelToMeters(_ point: CGPoint) -> CGPoint? {
        applyHomography(x: Double(point.x), y: Double(point.y))
    }

    private func applyHomography(x: Double, y: Double) -> CGPoint? {
        guard homographyMatrix.count == 3,
              homographyMatrix.allSatisfy({ $0.count == 3 }) else { return nil }
        let h = homographyMatrix
        let wx = h[0][0] * x + h[0][1] * y + h[0][2]
        let wy = h[1][0] * x + h[1][1] * y + h[1][2]
        let w  = h[2][0] * x + h[2][1] * y + h[2][2]
        guard abs(w) > 1e-10 else { return nil }
        return CGPoint(x: wx / w, y: wy / w)
    }

    /// Identity mapper for when the court is unknown — returns pixel coords unchanged.
    static var identity: CourtMapper {
        CourtMapper(homographyMatrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    }
}
