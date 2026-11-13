import CoreVideo
import XCTest
@testable import ShuttlCore

/// Rotation (F1) and presentation-clock parity with backend video_io.py on the robustness
/// corpus (fixture video_rotation.json, exported by scripts/swift_fixtures/video_io.py):
/// same upright size, same orientation of the pixels, same clock and the same PTS gaps.
final class VideoRotationGapTests: XCTestCase {

    func testRotatedCorpusMatchesPython() async throws {
        let cases = try XCTUnwrap(try fixture("video_rotation").array)
        XCTAssertGreaterThanOrEqual(cases.count, 11)
        var checked = 0
        for c in cases {
            let rel = try XCTUnwrap(c["relative_path"]?.string)
            let url = repoRoot.appendingPathComponent(rel)
            guard FileManager.default.fileExists(atPath: url.path) else { continue }
            let src = try await VideoSource.open(url)
            let m = src.meta
            XCTAssertEqual(m.width, Int(c["width"]!.double!), rel)
            XCTAssertEqual(m.height, Int(c["height"]!.double!), rel)
            XCTAssertEqual(m.rotationDeg, Int(c["rotation_deg"]!.double!), rel)
            XCTAssertEqual(m.nFrames, Int(c["n_frames"]!.double!), rel)
            // REAL clock on both sides (cv2 POS_MSEC is ms-rounded)
            assertClose(m.timestamps, c["timestamps"]!.doubles, tol: 1e-3, "\(rel) clock")
            XCTAssertEqual(m.ptsGaps, c["pts_gaps"]!.doubles.map { Int($0) }, rel)
            XCTAssertEqual(PTSGaps.info(m.timestamps).array?.count, c["pts_gap_info"]?.array?.count, rel)
            // the upright pixels agree (VideoToolbox vs FFmpeg decode noise only)
            let thumbs = try XCTUnwrap(c["thumbs"])
            let want: [Int: JSONValue] = Dictionary(uniqueKeysWithValues: [0, m.nFrames - 1].compactMap { i in
                thumbs[String(i)].map { (i, $0) } })
            XCTAssertEqual(want.count, 2, rel)
            for (i, pb) in try src.frames() where want[i] != nil {
                let t = want[i]!
                let tw = Int(t["width"]!.double!), th = Int(t["height"]!.double!)
                XCTAssertEqual(CVPixelBufferGetWidth(pb), m.width, rel)
                XCTAssertEqual(CVPixelBufferGetHeight(pb), m.height, rel)
                let g = try ImageU8.gray(from: pb).resized(width: tw, height: th, method: .area)
                let ref = t["gray"]!.doubles
                let d = zip(g.data, ref).map { abs(Double($0) - $1) }
                // a wrong orientation gives mean |d| ≈ 17-80 on these clips (checked)
                XCTAssertLessThan(Num.mean(d), 4.0, "\(rel) frame \(i) mean |d|")
                XCTAssertLessThan(Num.percentile(d, 99), 20, "\(rel) frame \(i) p99")
            }
            checked += 1
        }
        XCTAssertGreaterThanOrEqual(checked, 11)
    }

    func testPTSGapRule() {
        var base = [Double](repeating: 0, count: 100)
        for i in 0..<100 { base[i] = Double(i) / 30 }
        XCTAssertEqual(PTSGaps.frames(base), [Int]())
        var j = base
        for i in 40..<100 { j[i] += 5 }
        XCTAssertEqual(PTSGaps.frames(j), [40])
        let probes: [(Double, Bool)] = [(0.29, false), (0.2999, false), (0.3001, true), (0.31, true)]
        for (dt, gap) in probes {
            var u = [Double](repeating: 0, count: 100)
            for i in 0..<100 { u[i] = i < 50 ? Double(i) / 60 : 49.0 / 60 + dt + Double(i - 50) / 60 }
            let got: [Int] = PTSGaps.frames(u)
            XCTAssertEqual(got == [50], gap, "\(dt)")
        }
        var slow = [Double](repeating: 0, count: 40)
        for i in 0..<40 { slow[i] = Double(i) * 0.1 }
        for i in 20..<40 { slow[i] += 0.25 }
        XCTAssertEqual(PTSGaps.frames(slow), [Int]())                  // 0.35 s < 4 x 0.1
        for i in 20..<40 { slow[i] += 0.1 }
        XCTAssertEqual(PTSGaps.frames(slow), [20])                     // 0.45 s
        let bads: [[Double]] = [[], [0], [0, 1], [0, 0, 0], [0, .nan, 5, 5.1], [0, 1.0 / 30, 2.0 / 30, .infinity]]
        for bad in bads {
            XCTAssertEqual(PTSGaps.frames(bad), [Int](), "\(bad)")
        }
        var t = [Double](repeating: 0, count: 20)
        for i in 0..<20 { t[i] = (i < 10 ? 0.0 : 5.0) + Double(i % 10) / 30 }
        let entry: [String: JSONValue] = ["frame": .number(10), "t_before": .number(t[9]),
                                          "t_after": .number(t[10]), "dt_s": .number(t[10] - t[9])]
        let want = JSONValue.array([JSONValue.object(entry)])
        XCTAssertEqual(PTSGaps.info(t), want)
    }
}
