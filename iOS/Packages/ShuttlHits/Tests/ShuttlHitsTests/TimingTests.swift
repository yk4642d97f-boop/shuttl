import Foundation
import XCTest
@testable import ShuttlHits

/// Throughput of detectHits (run with `swift test -c release -Xswiftc -enable-testing --filter Timing`).
final class TimingTests: XCTestCase {
    func testThroughputPer1000Frames() throws {
        for file in ["hits_sim_e2e.json", "hits_real.json"] {
            let f = try Fixtures.load(file)
            let inputs = f.cases.map { ($0.inputs, $0.config) }
            let frames = inputs.reduce(0) { $0 + $1.0.track.n }
            var best = Double.infinity
            for _ in 0..<3 {
                let t0 = DispatchTime.now().uptimeNanoseconds
                for (i, cfg) in inputs {
                    _ = detectHits(track: i.track, meta: i.meta, camera: i.camera, players: i.players, config: cfg, rescorer: nil)
                }
                best = min(best, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9)
            }
            let py = f.python_timing?.ms_per_1000_frames ?? .nan
            print(String(format: "[timing] %@: %d frames, %.1f ms total, %.1f ms / 1000 frames (python %.0f ms / 1000 frames)",
                         file, frames, best * 1000, best * 1e6 / Double(frames), py))
        }
    }
}
