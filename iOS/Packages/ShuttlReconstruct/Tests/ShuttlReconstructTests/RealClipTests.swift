import XCTest
import ShuttlCore
@testable import ShuttlReconstruct

/// Real TrackNetV2 Test clips (broadcast 1280×720@30): cached TrackNetV3 candidates →
/// Python linker, Python court_calibration camera, Python hits → reconstructSegments.
///
/// Acceptance: the same segments come out — count, hit frames, start/end frames, frames
/// covered, end_reason, landing / net-crossing presence — exactly, on every segment; p0/v0
/// (and every other continuous output) to 1e-6 on the segments the Python reference itself
/// reproduces under 1-ulp input perturbations (`self_spread` <= 1e-6, see the exporter).
/// On the others (ill-conditioned far-court flights where the LM amplifies round-off) the
/// reference is not round-off-reproducible — Python differs from itself as much as Swift
/// differs from it — so their continuous values are reported, not asserted.
///
/// One documented flat-valley case: match1/1_05_02 seg 13 (distorted camera, k1 = 0.21).
/// Its pass-1 `_solve` refines three starts that end 1e-3 apart in p0 with costs equal to
/// 2e-8 relative (45.3714275 / 45.3714265 / 45.3714272 in Python itself); which one wins
/// flips on the ~1e-11 cost differences left by the pinhole-isation ulps (ShuttlCore simd
/// ray vs numpy BLAS), and pass 2 inherits it. It is held to that valley's own width.
final class RealClipTests: XCTestCase {
    static let flatValley: [String: Double] = ["match1/1_05_02#13": 2e-3]

    func testRealClipsParity() throws {
        let clips = try fixture("real").array!
        XCTAssertGreaterThanOrEqual(clips.count, 3)
        var worst = 0.0, worstLand = 0.0, nSeg = 0, nStable = 0, ms = 0.0, pyms = 0.0, frames = 0
        var reasons: [String: Int] = [:]
        var unstable: [String] = []
        for c in clips {
            XCTAssertEqual(c["self_structure_stable"]?.bool, true)
            let spread = c["self_spread"]!.doubles
            let stable = { (i: Int) in i < spread.count && spread[i] <= 1e-6 }
            let clip = c["clip"]!.str!
            let valley = { (i: Int) in Self.flatValley["\(clip)#\(i)"] }
            let r = checkRally(c, clip, valueCheck: { stable($0) && valley($0) == nil })
            for i in r.segs.indices { if let b = valley(i) { XCTAssertLessThanOrEqual(r.errs[i], b, "\(clip)#\(i) flat valley") } }
            worst = max(worst, r.worst); worstLand = max(worstLand, r.worstLand); nSeg += r.segs.count
            ms += r.ms; pyms += c["py_ms"]!.d; frames += c["track"]!["t"]!.doubles.count
            for s in r.segs { reasons[s.endReason.rawValue, default: 0] += 1 }
            for i in r.segs.indices {
                if stable(i) && valley(i) == nil { nStable += 1 } else {
                    unstable.append(String(format: "%@#%d swift-vs-py %.1e (py self-spread %.1e)%@", clip, i,
                                           r.errs[i], spread[i], valley(i) != nil ? " [flat valley]" : ""))
                }
            }
            print(String(format: "  %@: %d frames, %d segs, Swift %.0f ms vs Python CPU %.0f ms", clip,
                         c["track"]!["t"]!.doubles.count, r.segs.count, r.ms, c["py_ms"]!.d))
        }
        print(String(format: "real clips: %d clips, %d frames, %d segs %@ — structure identical on all; " +
                     "%d self-stable segs: max p0/v0 rel err %.1e, landing diff %.4f mm | " +
                     "Swift %.0f ms (%.1f ms/seg, %.0f ms/1000 frames) vs Python CPU %.0f ms",
                     clips.count, frames, nSeg, reasons.description, nStable, worst, worstLand * 1000,
                     ms, ms / Double(nSeg), 1000 * ms / Double(frames), pyms))
        print("  not round-off-reproducible in Python (values reported only):\n    " + unstable.joined(separator: "\n    "))
    }
}
