import Foundation
import XCTest
@testable import ShuttlLinker
import ShuttlCore

/// test_one_hour_60fps_runtime: tiled synthetic 60 fps rallies (with distractors).
/// Python asserts < 20 s for 216 000 frames. Run in release: `swift test -c release`.
final class PerformanceTests: XCTestCase {
    func testOneHour60fpsRuntime() throws {
        #if DEBUG
        throw XCTSkip("timing only meaningful in release (-c release)")
        #else
        let src = try loadCases("linker_sim").filter { $0.meta.fps == 60 && $0.name.hasPrefix("sim_") && $0.scene_cuts == nil }
        XCTAssertFalse(src.isEmpty)
        let target = 216_000
        var cands: [[ShuttleCandidate]] = []
        var ts: [Double] = []
        cands.reserveCapacity(target); ts.reserveCapacity(target)
        var t0 = 0.0
        while cands.count < target {
            for c in src {
                let cc = c.cands.candidates
                let tt = linkerTimestamps(c.meta.meta, cc.count)
                cands += cc
                ts += tt.map { $0 + t0 }
                t0 = ts[ts.count - 1] + 1.0 / 60.0
            }
        }
        cands = Array(cands.prefix(target)); ts = Array(ts.prefix(target))
        let meta = VideoMeta(fps: 60, width: 1920, height: 1080, nFrames: target, timestamps: ts)
        // thread CPU time: on a shared, overloaded Mac wall time measures the other jobs
        func cpu() -> Double { Double(clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID)) / 1e9 }
        let start = Date(), c0 = cpu()
        let tr = try linkCandidates(cands, meta: meta)
        let el = Date().timeIntervalSince(start), cp = cpu() - c0
        XCTAssertEqual(tr.n, target)
        print(String(format: "[perf] %d frames: cpu %.2f s = %.1f ms / 1000 frames (wall %.2f s)",
                     target, cp, cp * 1e6 / Double(target), el))
        XCTAssertLessThan(cp, 20.0)
        #endif
    }
}
