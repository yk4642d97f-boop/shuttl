import XCTest
import ShuttlCore
import simd
@testable import ShuttlReconstruct

/// Presentation-clock gaps (PTSGaps = video_io.pts_gaps): no flight is fitted across one.
/// Parity with Python on gaps.json (synthetic 0.43 s drops / 5 s jumps with and without a
/// hit inside the gap, the real F3 / F4 repros, refine_hit_time next to a gap) plus the
/// structural invariants.
final class PTSGapTests: XCTestCase {
    static let fx: JSONValue = try! fixture("gaps")

    func testGapRallyParity() {
        var n = 0, cut = 0, late = 0
        for c in Self.fx["rallies"]!.array! {
            let tag = c["tag"]!.str!
            let r = checkRally(c, tag)
            let t = c["track"]!["t"]!.doubles
            let gaps = PTSGaps.frames(t)
            XCTAssertFalse(gaps.isEmpty, tag)
            for (s, p) in zip(r.segs, c["segs"]!.array!) {
                n += 1
                XCTAssertEqual(s.endsAtPTSGap, p["ends_at_pts_gap"]?.bool ?? false, "\(tag) gap-end flag")
                XCTAssertEqual(s.startsAfterPTSGap, p["starts_after_pts_gap"]?.bool ?? false, "\(tag) gap-start flag")
                let fr = s.segment.frames
                for g in gaps {                                          // one side of every gap
                    XCTAssert(fr.allSatisfy { $0 < g } || fr.allSatisfy { $0 >= g }, "\(tag): segment spans gap \(g)")
                }
                if s.endsAtPTSGap {
                    cut += 1
                    XCTAssertEqual(s.segment.endReason, .unknown, tag)
                    XCTAssertNil(s.segment.landingXY, tag)
                    XCTAssertLessThanOrEqual(s.speed0SigmaKmh ?? .infinity, 0.15 * s.segment.speed0Kmh, tag)
                }
                if s.startsAfterPTSGap { late += 1 }
                if let l = s.segment.landingXY {
                    XCTAssertLessThan(abs(l.x), 6.70 + 1, "\(tag) landing x"); XCTAssertLessThan(abs(l.y), 3.05 + 1, tag)
                }
            }
        }
        XCTAssertGreaterThanOrEqual(cut, 1); XCTAssertGreaterThanOrEqual(late, 1)
        print("gap parity: \(n) segments, \(cut) cut at a gap, \(late) first seen after one")
    }

    func testRefineHitTimeNextToGap() {
        for c in Self.fx["refine"]!.array! {
            let (f, tc) = Reconstruct3D.refineHitTime(track: c["track"]!.track, approxFrame: c["approx_frame"]!.int,
                                                      camera: c["camera"]!.camera)
            XCTAssertEqual(f, c["frame"]!.int)
            assertClose(tc, c["t_c"]!.d, tol: 1e-6, "refine t_c")
        }
    }
}
