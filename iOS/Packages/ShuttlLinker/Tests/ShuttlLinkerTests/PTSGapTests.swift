import Foundation
import XCTest
@testable import ShuttlLinker
import ShuttlCore

/// Presentation-clock gaps (PTSGaps = video_io.pts_gaps) act as scene cuts: parity with the
/// Python reference on synthetic drops / jumps, the real F3 / F4 repro candidates and
/// gap-rule edge clocks (linker_gaps.json), plus the cut-equivalence invariant.
final class PTSGapTests: XCTestCase {

    func testGapParity() throws {
        let cases = try loadCases("linker_gaps")
        XCTAssertGreaterThanOrEqual(cases.count, 14)
        var withGap = 0
        for c in cases {
            let meta = c.meta.meta
            XCTAssertEqual(PTSGaps.frames(meta.timestamps), c.pts_gaps, "\(c.name): gap rule")
            if !(c.pts_gaps ?? []).isEmpty { withGap += 1 }
            let tr = try linkCandidates(c.cands.candidates, meta: meta, sceneCuts: c.scene_cuts)
            let d = compare(tr, c.out)
            XCTAssert(d.ok, "\(c.name): \(d.problems.joined(separator: "; "))")
            // smooth_track / find_kinks on the Python-linked track
            let inTrack = try XCTUnwrap(c.in_track)
            let pyTrack = inTrack.track(t: (inTrack.t ?? []).map { $0 ?? .nan })
            let size = (width: meta.width, height: meta.height)
            let d2 = compare(smoothTrack(pyTrack, imageSize: size), try XCTUnwrap(c.smooth))
            XCTAssert(d2.ok, "\(c.name) smooth: \(d2.problems.joined(separator: "; "))")
            let fk = findKinks(pyTrack, imageSize: size)
            XCTAssertEqual(fk.frames, try XCTUnwrap(c.find_kinks).frames, "\(c.name) kinks")
        }
        XCTAssertGreaterThanOrEqual(withGap, 10)
    }

    /// A gap is exactly a scene cut at its first frame.
    func testGapEqualsExplicitCut() throws {
        for c in try loadCases("linker_gaps") {
            guard let g = c.pts_gaps, !g.isEmpty else { continue }
            let meta = c.meta.meta
            let a = try linkCandidates(c.cands.candidates, meta: meta)
            let b = try linkCandidates(c.cands.candidates, meta: meta, sceneCuts: g)
            XCTAssertEqual(a.visible, b.visible, c.name); XCTAssertEqual(a.interpolated, b.interpolated, c.name)
            XCTAssert(zip(a.xy, b.xy).allSatisfy { ($0.x == $1.x || ($0.x.isNaN && $1.x.isNaN))
                && ($0.y == $1.y || ($0.y.isNaN && $1.y.isNaN)) }, c.name)
            // nothing filled between the last observation before and the first after a gap
            for f in g {
                guard let l = (0..<f).last(where: { a.visible[$0] }),
                      let r = (f..<a.n).first(where: { a.visible[$0] }) else { continue }
                XCTAssertFalse((l...r).contains { a.interpolated[$0] }, "\(c.name): fill across gap \(f)")
            }
        }
    }

    /// Normal clocks never have a gap: every stored sim / real / edge fixture without a jump.
    func testNoGapOnNormalFixtures() throws {
        for name in ["linker_sim", "linker_real"] {
            for c in try loadCases(name) {
                XCTAssertEqual(PTSGaps.frames(c.meta.meta.timestamps), [], "\(name)/\(c.name)")
            }
        }
    }
}
