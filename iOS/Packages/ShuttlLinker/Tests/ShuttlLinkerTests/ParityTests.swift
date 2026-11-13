import Foundation
import XCTest
@testable import ShuttlLinker
import ShuttlCore

/// Parity with the Python reference (backend/app/services/tracking/linker.py):
/// identical visible / interpolated masks, xy within 1e-6 px, raw_xy and conf exact.
final class ParityTests: XCTestCase {

    private func runParity(_ fixture: String, file: StaticString = #filePath, line: UInt = #line) throws {
        let cases = try loadCases(fixture)
        XCTAssertFalse(cases.isEmpty)
        var worst = 0.0, worstFill = 0.0, fills = 0, nFrames = 0
        for c in cases {
            let tr = try linkCandidates(c.cands.candidates, meta: c.meta.meta, sceneCuts: c.scene_cuts)
            let d = compare(tr, c.out)
            worst = max(worst, d.maxXY); worstFill = max(worstFill, d.maxFill)
            fills += d.fillsOver1e6; nFrames += tr.n
            if d.fillsOver1e6 > 0 { print("[parity] \(c.name): \(d.fillsOver1e6) fills > 1e-6 px (max \(d.maxFill))") }
            XCTAssert(d.ok, "\(c.name): \(d.problems.joined(separator: "; "))", file: file, line: line)
            if !d.ok, let inter = c.inter { try diagnose(c, inter) }
        }
        print("[parity] \(fixture): \(cases.count) cases, \(nFrames) frames, max |Δxy| visible \(worst) px, "
              + "fills \(worstFill) px, \(fills) fills > 1e-6 px")
    }

    /// Localise a mismatch: compare the first-pass internals.
    private func diagnose(_ c: FxCase, _ inter: FxInter) throws {
        let cfg = LinkerConfig()
        let cands = c.cands.candidates
        let meta = c.meta.meta
        let n = cands.count
        let t = linkerTimestamps(meta, n)
        let w = Double(meta.width != 0 ? meta.width : 1920), h = Double(meta.height != 0 ? meta.height : 1080)
        let cuts = cutList(n, c.scene_cuts, try detectCuts(cands, t, w, h, cfg))
        if cuts != inter.cuts { print("  [\(c.name)] cuts \(cuts) vs \(inter.cuts)"); return }
        let a = try associate(cands, t, w, h, cfg, shot0: shot0(n, cuts))
        let obs = flatnonzero(a.xy.map { $0.x.isFinite })
        if obs != inter.assoc_obs {
            let diff = Array(Set(obs).symmetricDifference(inter.assoc_obs)).sorted()
            print("  [\(c.name)] association differs at \(diff.count) frames: \(diff.prefix(12))"); return
        }
        if flatnonzero(a.start) != inter.assoc_start { print("  [\(c.name)] association starts differ"); return }
        let rep = findRepeats(t, a.xy, probeCandidates(cands, a.xy, cfg, w, h), hypot(w, h), cfg)
        let dup = rep.map { flatnonzero($0.isDup) }
        if dup != inter.rep_dup { print("  [\(c.name)] repeats \(dup ?? []) vs \(inter.rep_dup ?? [])"); return }
        print("  [\(c.name)] association + repeats agree → mismatch is in _finish / _readopt")
    }

    func testSimParity() throws { try runParity("linker_sim") }

    func testRealParity() throws { try runParity("linker_real") }

    /// smooth_track (auto kinks and forced kinks) and find_kinks on a linked track.
    func testSmoothTrackAndFindKinksParity() throws {
        let cases = try loadCases("linker_sim")
        let c = try XCTUnwrap(cases.first { $0.find_kinks != nil })
        let inTrack = try XCTUnwrap(c.in_track)
        let tr = inTrack.track(t: (inTrack.t ?? []).map { $0 ?? .nan })
        let size = (width: c.meta.width, height: c.meta.height)
        let fk = findKinks(tr, imageSize: size)
        let ref = try XCTUnwrap(c.find_kinks)
        XCTAssertEqual(fk.frames, ref.frames)
        for (a, b) in zip(fk.scores, ref.z) {
            let bb = b ?? .infinity
            XCTAssert(a == bb || abs(a - bb) <= 1e-6 * max(1, abs(bb)), "z \(a) vs \(bb)")
        }
        let sm = smoothTrack(tr, imageSize: size)
        let d1 = compare(sm, try XCTUnwrap(c.smooth))
        XCTAssert(d1.ok, "smooth_track: \(d1.problems)")
        let sk = try XCTUnwrap(c.smooth_kinks)
        let d2 = compare(smoothTrack(tr, kinks: sk.kinks, imageSize: size), sk.out)
        XCTAssert(d2.ok, "smooth_track(kinks:): \(d2.problems)")
    }

    /// Full 29-clip × 2-detector parity (from /tmp, not bundled). Also writes the
    /// Swift outputs for `python -m scripts.swift_fixtures.linker --score`.
    func testFullRealParity() throws {
        guard let cases = try loadCases(path: "/tmp/shuttl_linker_parity/real_full.json") else {
            throw XCTSkip("run the exporter to create /tmp/shuttl_linker_parity/real_full.json")
        }
        let outDir = URL(fileURLWithPath: "/tmp/shuttl_linker_parity/swift")
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        var nOK = 0, worst = 0.0, worstFill = 0.0, fills = 0, frames = 0
        var seconds = 0.0
        for c in cases {
            let cands = c.cands.candidates, meta = c.meta.meta
            let t0 = Date()
            let tr = try linkCandidates(cands, meta: meta)
            seconds += Date().timeIntervalSince(t0)
            frames += tr.n
            let d = compare(tr, c.out)
            worst = max(worst, d.maxXY); worstFill = max(worstFill, d.maxFill); fills += d.fillsOver1e6
            if d.ok { nOK += 1 } else { XCTFail("\(c.name): \(d.problems.joined(separator: "; "))") }
            let data = try JSONSerialization.data(withJSONObject: encodeTrack(tr))
            try data.write(to: outDir.appendingPathComponent("\(c.name).json"))
        }
        print(String(format: "[parity] full real: %d/%d clips pass, max |Δxy| visible %.3g px, fills %.3g px (%d fills > 1e-6), %d frames, %.1f ms / 1000 frames",
                     nOK, cases.count, worst, worstFill, fills, frames, seconds * 1e6 / Double(max(frames, 1))))
    }
}
