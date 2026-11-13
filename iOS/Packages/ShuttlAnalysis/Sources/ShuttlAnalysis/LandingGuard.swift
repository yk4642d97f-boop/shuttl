import Foundation
import ShuttlCore

/// A landing far outside the hall is not a landing: it comes from a flight fitted across a
/// timestamp jump or a mis-linked track (robustness F3: -11.1 m, 4.4 m behind the baseline,
/// still scored). Such a landing is dropped, and a rally decided by it becomes undecided,
/// before the score is computed. Normal play never comes near the bound.
public enum LandingGuard {
    /// Metres beyond the doubles court; the same bound ShuttlRally's 2-D outcome uses.
    public static let marginM = 3.0
    public static let diagnosticsKey = "implausible_landings"
    public static let outcomeSource = "implausible_landing"

    public static func isPlausible(_ p: SIMD2<Double>) -> Bool {
        p.x.isFinite && p.y.isFinite
            && abs(p.x) <= CourtModel.halfLength + marginM && abs(p.y) <= CourtModel.doublesHalfWidth + marginM
    }

    /// Returns how many landings were dropped (rally outcomes + shot segments).
    @discardableResult
    public static func apply(rallies: inout [RallyV2], segments: inout [ShotSegment3D]) -> Int {
        var dropped = 0
        func clean(_ s: inout ShotSegment3D) {
            guard let l = s.landingXY, !isPlausible(l) else { return }
            s.landingXY = nil; s.landingT = nil
            dropped += 1
        }
        for i in segments.indices { clean(&segments[i]) }
        for r in rallies.indices {
            for s in rallies[r].shots.indices where rallies[r].shots[s].segment != nil {
                clean(&rallies[r].shots[s].segment!)
            }
            guard let l = rallies[r].landingXY, !isPlausible(l) else { continue }
            rallies[r].landingXY = nil
            rallies[r].inOutMarginM = nil
            rallies[r].endReason = .unknown
            rallies[r].winner = .unknown
            rallies[r].confidence = 0
            rallies[r].outcomeConfidence = 0
            rallies[r].outcomeSource = outcomeSource
            dropped += 1
        }
        return dropped
    }
}

extension MatchAnalysis {
    /// Rallies whose call was withdrawn because their landing was impossible.
    public var undecidedByImplausibleLanding: Int {
        rallies.filter { $0.outcomeSource == LandingGuard.outcomeSource }.count
    }

    /// A line for the result screen (nil when every landing was plausible).
    public var landingWarning: String? {
        let n = undecidedByImplausibleLanding
        return n == 0 ? nil : "有 \(n) 个回合的落点算不出来（视频的时间可能被剪过），这些回合的输赢没有计入比分。"
    }
}
