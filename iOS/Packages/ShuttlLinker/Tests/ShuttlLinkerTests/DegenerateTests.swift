import Foundation
import XCTest
@testable import ShuttlLinker
import ShuttlCore

/// Degenerate inputs must never crash the phone: every case either returns a track that
/// satisfies the Track2D invariants or throws a LinkerError (where Python raises).
final class DegenerateTests: XCTestCase {

    func testDegenerateSweepNeverCrashes() {
        #if DEBUG
        let stride = 11                                  // debug is ~10× slower: sample the grid
        #else
        let stride = 1
        #endif
        var rng = SplitMix64(s: 7)
        var idx = 0, ran = 0, threw = 0
        let trace = ProcessInfo.processInfo.environment["LINKER_TRACE"] != nil
        for n in [0, 1, 2, 3, 5, 13, 40, 150] {
            for (w, h) in Degenerate.sizes { for kind in Degenerate.kinds { for fps in Degenerate.fpsList {
                for clock in Degenerate.clocks {
                    idx += 1
                    if idx % stride != 0 { continue }
                    let name = "n\(n)_\(w)x\(h)_\(kind)_fps\(fps)_\(clock)"
                    if trace { FileHandle.standardError.write("\(name)\n".data(using: .utf8)!) }
                    let cands = Degenerate.candidates(kind, n, w, h, &rng)
                    let meta = VideoMeta(fps: fps, width: w, height: h, nFrames: n,
                                         timestamps: Degenerate.timestamps(clock, n, fps))
                    do {
                        let tr = try linkCandidates(cands, meta: meta)
                        Degenerate.check(tr, n: n, name)
                        if n > 1 { XCTAssertTrue(zip(tr.t.dropFirst(), tr.t).allSatisfy { $0 > $1 }, "t \(name)") }
                        let size: (width: Int, height: Int)? = w > 0 ? (w, h) : nil
                        Degenerate.check(smoothTrack(tr, imageSize: size), n: n, "smooth " + name)
                        let k = findKinks(tr, imageSize: size)
                        XCTAssertEqual(k.frames.count, k.scores.count, name)
                        XCTAssertTrue(k.frames.allSatisfy { $0 >= 0 && $0 < max(n, 1) }, name)
                        ran += 1
                    } catch is LinkerError {
                        threw += 1
                    } catch { XCTFail("\(name): unexpected \(error)") }
                } } } }
        }
        print("[degenerate] \(ran) linked, \(threw) threw LinkerError (grid stride \(stride))")
        XCTAssertGreaterThan(ran, 0)
    }

    /// Venue regression: a path with no strong observation (Python IndexError before the fix).
    func testWeakOnlyCandidates() throws {
        for (w, h) in [(192, 144), (64, 48), (1920, 1080)] {
            let diag = hypot(Double(w), Double(h)), n = 12, v = 0.5 * diag / 30
            let meta = VideoMeta(fps: 30, width: w, height: h, nFrames: n, timestamps: (0..<n).map { Double($0) / 30 })
            let fly = (0..<n).map { [ShuttleCandidate(x: 0.1 * Double(w) + v * Double($0) * 0.9,
                                                      y: 0.2 * Double(h) + v * Double($0) * 0.4, score: 0.3)] }
            XCTAssertEqual(try linkCandidates(fly, meta: meta).visible.filter { $0 }.count, n, "\(w)x\(h) flying")
            var still = (0..<n).map { [ShuttleCandidate(x: 0.3 * Double(w) + 0.01 * Double($0), y: 0.6 * Double(h), score: 0.3)] }
            XCTAssertFalse(try linkCandidates(still, meta: meta).visible.contains(true), "\(w)x\(h) still")
            still[3].append(ShuttleCandidate(x: 0.8 * Double(w), y: 0.2 * Double(h), score: 0.51))
            XCTAssertFalse(try linkCandidates(still, meta: meta).visible.contains(true), "\(w)x\(h) + strong blip")
        }
    }

    func testInvalidConfigsThrowExtremeConfigsRun() throws {
        let n = 60
        let meta = VideoMeta(fps: 30, width: 192, height: 144, nFrames: n, timestamps: (0..<n).map { Double($0) / 30 })
        let cands = (0..<n).map { [ShuttleCandidate(x: 20 + 2 * Double($0), y: 30 + Double($0), score: 0.9)] }
        let bad: [(inout LinkerConfig) -> Void] = [
            { $0.staticBlockS = 0 }, { $0.maxCandidates = -1 }, { $0.kinkMaxPts = 0 }, { $0.kinkDeg = -1 },
            { $0.refineRounds = -1 }, { $0.maxLinkGapS = .nan }, { $0.sigmaPx = .infinity }, { $0.cutMinCells = 0 }]
        for (i, f) in bad.enumerated() {
            var c = LinkerConfig(); f(&c)
            XCTAssertThrowsError(try linkCandidates(cands, meta: meta, config: c), "bad config \(i)")
        }
        let extreme: [(inout LinkerConfig) -> Void] = [
            { $0.maxLinkGapS = 1e12; $0.kinkMaxGapS = 1e12 }, { $0.kinkWindowS = 1e9 }, { $0.staticBlockS = 1e-9 },
            { $0.staticBlockS = -5 }, { $0.maxCandidates = 0 }, { $0.kinkMinPts = 1; $0.kinkMaxPts = 1; $0.kinkLocPts = 1 },
            { $0.irlsIters = 0; $0.refineRounds = 0; $0.repeatMinCount = 0 }, { $0.weakBanRounds = -3; $0.minChainLen = -1 },
            { $0.maxExtrapFrames = -1 }, { $0.staticCell = -1 }, { $0.staticCell = 1e300 }]
        for (i, f) in extreme.enumerated() {
            var c = LinkerConfig(); f(&c)
            Degenerate.check(try linkCandidates(cands, meta: meta, config: c), n: n, "extreme config \(i)")
        }
    }

    func testMalformedTracksForSmoothAndKinks() throws {
        let n = 40
        let t = (0..<n).map { Double($0) / 30 }
        let good = try linkCandidates((0..<n).map { [ShuttleCandidate(x: 20 + 3 * Double($0), y: 40, score: 0.9)] },
                                      meta: VideoMeta(fps: 30, width: 192, height: 144, nFrames: n, timestamps: t))
        var short = good; short.xy.removeLast()
        var nanT = good; nanT.t[5] = .nan
        var badRaw = good; badRaw.rawXY = [SIMD2(1, 1)]
        let empty = Track2D(t: [], xy: [], conf: [], visible: [], interpolated: [])
        for (name, tr) in [("short xy", short), ("nan t", nanT), ("raw len", badRaw), ("empty", empty)] {
            XCTAssertEqual(smoothTrack(tr).xy.count, tr.xy.count, name)
            XCTAssertTrue(findKinks(tr).frames.isEmpty, name)
        }
        Degenerate.check(smoothTrack(good, kinks: [-5, 0, 3, n - 1, n, 10_000]), n: n, "odd kinks")
    }
}
