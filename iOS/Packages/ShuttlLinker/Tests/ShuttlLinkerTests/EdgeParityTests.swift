import Foundation
import XCTest
@testable import ShuttlLinker
import ShuttlCore

/// Parity on degenerate inputs (backend/scripts/swift_fixtures/linker.py --edge): the venue
/// 192×144 weak-only clips, 0/1/2/3-frame videos, empty frames, single candidates and broken
/// clocks (missing / short / duplicated / zero / NaN / reversed timestamps, fps 0/inf/nan).
/// Python links all of them after the venue fixes; Swift must return the same track, exactly
/// (duplicate-PTS runs are re-timed onto the frame grid, so no case is ill-conditioned any more).
final class EdgeParityTests: XCTestCase {

    func testEdgeParity() throws {
        let cases = try loadCases("linker_edge")
        XCTAssertGreaterThan(cases.count, 1000)
        var bad = 0, fills = 0, kinks = 0
        let trace = ProcessInfo.processInfo.environment["LINKER_TRACE"] != nil
        for c in cases {
            if trace { FileHandle.standardError.write("\(c.name)\n".data(using: .utf8)!) }
            let tr: Track2D
            do { tr = try linkCandidates(c.cands.candidates, meta: c.videoMeta, sceneCuts: c.scene_cuts) } catch {
                XCTFail("\(c.name): Swift threw \(error), Python linked"); bad += 1; continue
            }
            var d = compare(tr, c.out)
            let pyT = (c.out.t ?? []).map { $0 ?? .nan }
            if !zip(tr.t, pyT).allSatisfy({ $0 == $1 || ($0.isNaN && $1.isNaN) }) || tr.t.count != pyT.count {
                d.problems.append("t differs")
            }
            if let ref = c.smooth {
                let size: (width: Int, height: Int)? = c.meta.width > 0 && c.meta.height > 0 ? (c.meta.width, c.meta.height) : nil
                let sd = compare(smoothTrack(tr, imageSize: size), ref)
                d.problems += sd.problems.map { "smooth: " + $0 }
                fills += sd.fillsOver1e6
                if let fk = c.find_kinks {
                    let k = findKinks(tr, imageSize: size)
                    kinks += fk.frames.count
                    if k.frames != fk.frames { d.problems.append("find_kinks \(k.frames) vs \(fk.frames)") }
                }
            }
            fills += d.fillsOver1e6
            if tr.t.count > 1 && !zip(tr.t.dropFirst(), tr.t).allSatisfy({ $0 > $1 }) {
                bad += 1; XCTFail("\(c.name): t not strictly increasing"); continue
            }
            if !d.ok { bad += 1; XCTFail("\(c.name): \(d.problems.joined(separator: "; "))") }
        }
        print("[edge] \(cases.count) cases, \(cases.count - bad) identical to Python "
              + "(\(fills) fills > 1e-6 px, \(kinks) kinks)")
    }

    /// The real venue clip that made the Python link stage fail (IndexError): links, and
    /// adopts nothing (weak, non-flying candidates; the footage is not badminton).
    func testVenueClip() throws {
        guard let c = try loadCases("linker_edge").first(where: { $0.name == "edge_venue_match_1790485029" }) else {
            throw XCTSkip("venue candidates were not available when the fixture was exported")
        }
        XCTAssertEqual(c.meta.width, 192); XCTAssertEqual(c.meta.height, 144)
        let cands = c.cands.candidates
        XCTAssertLessThanOrEqual(cands.flatMap { $0 }.filter { $0.score >= 0.5 }.count, 1)
        let tr = try linkCandidates(cands, meta: c.videoMeta)
        Degenerate.check(tr, n: cands.count, c.name)
        XCTAssertEqual(tr.visible.filter { $0 }.count, c.out.visible.count)
    }
}
