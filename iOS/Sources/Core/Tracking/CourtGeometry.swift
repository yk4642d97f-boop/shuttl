import Foundation
import simd

// MARK: - CourtGeometry
// Mirrors backend/app/services/tracking/court_model.py — keep the two in sync.
//
// World frame (metres, right-handed):
//   origin = court centre on the floor, under the net
//   +x     = towards the FAR baseline (near half x < 0)
//   +y     = LEFT as seen from behind the near baseline
//   +z     = up

enum CourtGeometry {
    static let halfLength: Double = 6.70
    static let doublesHalfWidth: Double = 3.05
    static let singlesHalfWidth: Double = 2.59
    static let shortServiceX: Double = 1.98
    static let doublesLongServiceX: Double = 5.94
    static let lineWidth: Double = 0.04
    static let netHeightCentre: Double = 1.524
    static let netHeightPost: Double = 1.55

    static let courtLength: Double = 2 * halfLength
    static let doublesWidth: Double = 2 * doublesHalfWidth

    /// Top of the two net posts (3-D keypoints, standing on the doubles side lines).
    static let netPostTops: [SIMD3<Double>] = [
        SIMD3(0, -doublesHalfWidth, netHeightPost),
        SIMD3(0, doublesHalfWidth, netHeightPost),
    ]
}

// MARK: - CourtCorner
// Outer doubles corners, named exactly as the backend keypoints (court_model.COURT_KEYPOINTS).
//
// CAREFUL: the backend suffix "ld" means y = -3.05 and "rd" means y = +3.05. Because +y is
// the near-baseline viewer's left, a camera standing behind the near baseline sees the
// "ld" corners on the RIGHT of its image. Use `CourtCorner.assign(tapped:)` to label taps by
// their image position instead of by name.

enum CourtCorner: String, CaseIterable, Codable, Sendable {
    case nbLd = "nb_ld"   // (-6.70, -3.05)
    case nbRd = "nb_rd"   // (-6.70, +3.05)
    case fbLd = "fb_ld"   // (+6.70, -3.05)
    case fbRd = "fb_rd"   // (+6.70, +3.05)

    /// Floor position in world metres (z = 0).
    var world: SIMD2<Double> {
        let l = CourtGeometry.halfLength, w = CourtGeometry.doublesHalfWidth
        switch self {
        case .nbLd: return SIMD2(-l, -w)
        case .nbRd: return SIMD2(-l, w)
        case .fbLd: return SIMD2(l, -w)
        case .fbRd: return SIMD2(l, w)
        }
    }

    var world3: SIMD3<Double> { SIMD3(world.x, world.y, 0) }

    /// Labels four tapped outer corners from a camera behind the NEAR baseline (upright
    /// frame, y down): the two lowest taps are the near baseline; within each baseline the
    /// image-left tap is the viewer's left (+y → "rd"), the image-right tap is "ld".
    /// Returns nil unless exactly four points are given.
    static func assign(tapped points: [CGPoint]) -> [CourtCorner: CGPoint]? {
        guard points.count == 4 else { return nil }
        let byY = points.sorted { $0.y < $1.y }
        let far = byY[0..<2].sorted { $0.x < $1.x }
        let near = byY[2..<4].sorted { $0.x < $1.x }
        return [.fbRd: far[0], .fbLd: far[1], .nbRd: near[0], .nbLd: near[1]]
    }
}
