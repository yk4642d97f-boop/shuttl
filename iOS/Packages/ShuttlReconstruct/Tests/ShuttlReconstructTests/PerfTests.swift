import XCTest
import ShuttlCore
@testable import ShuttlReconstruct

/// Release-mode timing of reconstructSegments on the real clips and the sim e2e rallies.
/// `SHUTTL_PERF=<n>` repeats the workload n times (for profiling with `sample`); without it
/// one pass is timed and reported.
final class PerfTests: XCTestCase {
    /// The thread count must not change a single bit of the output.
    func testParallelEqualsSerial() throws {
        var serial = ReconstructConfig(); serial.maxConcurrency = 1
        var par = ReconstructConfig(); par.maxConcurrency = 8
        var n = 0
        for c in try fixture("real").array! + (try fixture("rallies")["rallies"]!.array!) {
            let p = c["players"]!.players
            let args = (c["track"]!.track, c["hits"]!.hits, c["camera"]!.camera, c["meta"]!.meta, p.isEmpty ? nil : p)
            let a = Reconstruct3D.reconstructSegments(track: args.0, hits: args.1, camera: args.2, meta: args.3, players: args.4, config: serial)
            let b = Reconstruct3D.reconstructSegments(track: args.0, hits: args.1, camera: args.2, meta: args.3, players: args.4, config: par)
            XCTAssertEqual(a.count, b.count)
            for (x, y) in zip(a, b) {
                XCTAssertEqual((x.segment.p0.row + x.segment.v0.row + x.thetaRef + x.covTheta).map(\.bitPattern),
                               (y.segment.p0.row + y.segment.v0.row + y.thetaRef + y.covTheta).map(\.bitPattern))
                XCTAssertEqual(x.segment.frames, y.segment.frames)
                XCTAssertEqual(x.endReason, y.endReason)
                n += 1
            }
        }
        print("parallel == serial bitwise on \(n) segments")
    }

    func testTiming() throws {
        #if DEBUG
        throw XCTSkip("timing is only meaningful in release: swift test -c release --filter PerfTests")
        #endif
        let reps = Int(ProcessInfo.processInfo.environment["SHUTTL_PERF"] ?? "") ?? 1
        let sets: [(String, [JSONValue])] = [("real", try fixture("real").array!),
                                             ("sim e2e", try fixture("e2e")["cases"]!.array!)]
        for (name, cases) in sets {
            let inputs = cases.map { c -> (Track2D, [HitEvent], CameraModel, VideoMeta, [PlayerObservation]?) in
                let p = c["players"]!.players
                return (c["track"]!.track, c["hits"]!.hits, c["camera"]!.camera, c["meta"]!.meta, p.isEmpty ? nil : p)
            }
            let frames = inputs.reduce(0) { $0 + $1.0.t.count }
            for threads in [1, 2, 6, ProcessInfo.processInfo.activeProcessorCount] {
                var cfg = ReconstructConfig(); cfg.maxConcurrency = threads
                var best = Double.infinity, nSeg = 0
                for _ in 0..<max(reps, 2) {
                    let t0 = DispatchTime.now().uptimeNanoseconds
                    nSeg = 0
                    for (tr, h, cam, meta, pl) in inputs {
                        nSeg += Reconstruct3D.reconstructSegments(track: tr, hits: h, camera: cam, meta: meta, players: pl,
                                                                  config: cfg).count
                    }
                    best = min(best, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
                }
                print(String(format: "PERF %@ threads=%d: %d frames, %d segs: %.0f ms (%.1f ms/seg, %.0f ms/1000 frames)",
                             name, threads, frames, nSeg, best, best / Double(nSeg), 1000 * best / Double(frames)))
            }
        }
    }
}
