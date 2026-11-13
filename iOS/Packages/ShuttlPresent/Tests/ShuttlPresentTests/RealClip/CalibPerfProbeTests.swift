import XCTest
import simd
import ShuttlCore
import ShuttlCourt
import ShuttlPresent

/// Where does a calibration attempt spend its time? Gated: SHUTTL_PRESENT_PERF=1.
final class CalibPerfProbeTests: XCTestCase {
    func testAttemptTiming() async throws {
        guard ProcessInfo.processInfo.environment["SHUTTL_PRESENT_PERF"] == "1" else { throw XCTSkip("SHUTTL_PRESENT_PERF=1") }
        let url = testSplit.appendingPathComponent("match3/video/1_05_02.mp4")
        let (meta, dec) = try await decodeFrames(url, step: 15, count: 7)
        let full = ImageSize(width: meta.width, height: meta.height)
        let imgs = dec.map(\.1)
        let loops = Int(ProcessInfo.processInfo.environment["SHUTTL_PRESENT_LOOPS"] ?? "3") ?? 3
        let tMed = ms { _ = TemporalMedian.median(imgs) }
        print(String(format: "PERF own median 7 frames full-res: %.1f ms", tMed))
        for L in [1280, 960, 640] {
            let ds = FrameDownsampler(workingLongSide: L)
            let wf = imgs.map { ds.frame(from: $0, time: 0).image }
            let m = TemporalMedian.median(wf)
            var r: AutoCalibrationResult!
            var tSeven = 0.0, tOne = 0.0
            for _ in 0..<loops { tSeven += ms { r = CourtCalibration.calibrateFromFramesDetailed(wf, size: m.size) } }
            for _ in 0..<loops { tOne += ms { r = CourtCalibration.calibrateFromFramesDetailed([m], size: m.size) } }
            var cfg = CourtCalibrationConfig(); cfg.maxSegments = 250
            var tSeg = 0.0
            var r2: AutoCalibrationResult!
            for _ in 0..<loops { tSeg += ms { r2 = CourtCalibration.calibrateFromFramesDetailed([m], size: m.size, config: cfg) } }
            var tRef = 0.0
            if let cam = r.camera {
                for _ in 0..<loops { tRef += ms { _ = CourtCalibration.refineCamera(cam, frames: [m]) } }
            }
            let d = r.diagnostics
            print(String(format: "PERF L=%d %dx%d: 7-frame %.0f ms, 1-median %.0f ms (nLines %d nHyp %d), maxSeg250 %.0f ms (%@, nHyp %d), refine %.0f ms",
                         L, m.width, m.height, tSeven / Double(loops), tOne / Double(loops), d.nLines, d.nHyp,
                         tSeg / Double(loops), r2.camera == nil ? "reject" : "accept", r2.diagnostics.nHyp, tRef / Double(loops)))
        }
        _ = full
    }
}
