import Foundation
import simd
import XCTest
import ShuttlAnalysis
import ShuttlCore
@testable import ShuttlLive

/// Whole on-device pipeline on real Test clips, against pipeline_v2 run with the
/// same corner taps (fixture from backend/scripts/swift_fixtures/live_e2e.py).
///
/// Exact equality isn't expected: the detector is fp16 CoreML (not torch) on
/// AVFoundation-decoded frames, and players come from Apple Vision (not YOLO), so
/// hits and rallies can differ at the margins. The test bounds those differences
/// and prints a comparison table.
///
/// Slow (CoreML + Vision over real video); run in release:
///   SHUTTL_LIVE_E2E=1 swift test -c release --filter LiveEndToEndTests
final class LiveEndToEndTests: XCTestCase {

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }

    private var modelsDirectory: URL { repoRoot.appendingPathComponent("iOS/Packages/ShuttlDetector/Models") }

    func testRealClipsMatchPython() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["SHUTTL_LIVE_E2E"] == "1",
                          "set SHUTTL_LIVE_E2E=1 (slow: CoreML + Vision over real video)")
        let fixture = repoRoot.appendingPathComponent("iOS/Packages/ShuttlLive/Tests/ShuttlLiveTests/Fixtures/live_e2e.json")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: fixture.path),
                          "run `.venv/bin/python -m scripts.swift_fixtures.live_e2e` in backend/ first")
        let cases = try XCTUnwrap(ShuttlJSON.load(JSONValue.self, from: fixture)["cases"]?.array)
        XCTAssertFalse(cases.isEmpty)

        let cache = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-live-models")
        let stages = try await LiveStages.make(modelsDirectory: modelsDirectory, cacheDirectory: cache)
        let pipeline = AnalysisPipeline(stages: stages)
        var report: [String] = ["clip | py hits | sw hits | matched ±2f | side agree | py rallies | sw rallies | py s | sw s"]

        for c in cases {
            let rel = try XCTUnwrap(c["relative_path"]?.string)
            let url = repoRoot.appendingPathComponent(rel)
            try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path), "Test split missing")
            var taps: [String: SIMD2<Double>] = [:]
            if case .object(let o)? = c["taps"] {
                for (k, v) in o { let a = v.array ?? []; taps[k] = SIMD2(a[0].double ?? .nan, a[1].double ?? .nan) }
            }

            let t0 = Date()
            let res = try await pipeline.analyze(url, options: PipelineOptions(manualCorners: taps))
            let secs = Date().timeIntervalSince(t0)

            // Every stage ran (auto-calibration isn't used: taps are given).
            for s in ["detect", "link", "calibrate", "players", "hits", "reconstruct3d", "rallies", "score"] {
                XCTAssertEqual(res.diagnostics["stages"]?[s]?["ok"]?.bool, true,
                               "\(rel) stage \(s): \(res.diagnostics["stages"]?[s]?["error"]?.string ?? "missing")")
            }

            // Same taps → same calibration path → cameras agree on the court.
            let pyCam = try XCTUnwrap(c["camera"]).decodeCamera()
            let cam = try XCTUnwrap(res.camera)
            let (_, kp) = CourtModel.keypointsArray()
            let dpx = zip(cam.project(kp), pyCam.project(kp)).map { simd_length($0 - $1) }.filter(\.isFinite)
            XCTAssertLessThan(Num.median(dpx), 2.0, "\(rel) camera keypoints median px")

            // Track: similar visible count (detector F1 gap is ~0.0003 on these clips).
            let pyVisible = c["n_visible"]?.double ?? 0
            let swVisible = Double(res.track?.visible.filter { $0 }.count ?? 0)
            XCTAssertLessThan(abs(swVisible - pyVisible), max(8, 0.03 * pyVisible), "\(rel) visible frames")

            // Hits: most Python hits found within ±2 frames, sides mostly agree.
            let pyHits = (c["hits"]?.array ?? []).map { (Int($0["frame"]?.double ?? -99), $0["hitter"]?.string ?? "") }
            let swHits = res.hits.map { ($0.frame, $0.hitter.rawValue) }
            var used = Set<Int>(), matched = 0, sideAgree = 0
            for (f, side) in pyHits {
                if let j = swHits.indices.first(where: { !used.contains($0) && abs(swHits[$0].0 - f) <= 2 }) {
                    used.insert(j); matched += 1
                    if swHits[j].1 == side { sideAgree += 1 }
                }
            }
            if !pyHits.isEmpty {
                XCTAssertGreaterThanOrEqual(Double(matched) / Double(pyHits.count), 0.8, "\(rel) hit recall vs Python")
                XCTAssertLessThanOrEqual(abs(swHits.count - pyHits.count), max(2, pyHits.count / 5), "\(rel) hit count")
            }
            let pyRallies = c["rallies"]?.array?.count ?? 0
            XCTAssertLessThanOrEqual(abs(res.rallies.count - pyRallies), 1, "\(rel) rally count")

            // The saved analysis must round-trip (the app stores it as JSON).
            let back = try ShuttlJSON.decoder().decode(MatchAnalysis.self, from: ShuttlJSON.encoder().encode(res))
            XCTAssertEqual(back.rallies.count, res.rallies.count)

            report.append("\(url.lastPathComponent) | \(pyHits.count) | \(swHits.count) | \(matched) | \(sideAgree) | "
                          + "\(pyRallies) | \(res.rallies.count) | "
                          + String(format: "%.1f | %.1f", c["python_seconds"]?.double ?? .nan, secs))
        }
        print("[live e2e]\n" + report.joined(separator: "\n"))
    }

    /// Whole pipeline on any file (venue footage, measurement clips):
    ///   SHUTTL_ANALYZE_FILE=/path/clip.mp4 [SHUTTL_STAGE_METRICS=/tmp/m.jsonl] swift test -c release --filter testAnalyzeFileFromEnv
    func testAnalyzeFileFromEnv() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["SHUTTL_ANALYZE_FILE"] != nil, "set SHUTTL_ANALYZE_FILE")
        let path = ProcessInfo.processInfo.environment["SHUTTL_ANALYZE_FILE"]!
        let url = URL(fileURLWithPath: path)
        let stages = try await LiveStages.make(modelsDirectory: modelsDirectory,
                                               cacheDirectory: FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-live-models"))
        let t0 = Date()
        let res = try await AnalysisPipeline(stages: stages).analyze(url, options: PipelineOptions(doubles: path.contains("doubles") ? true : nil))
        let speeds = res.rallies.flatMap(\.shots).compactMap { $0.segment?.speed0Kmh }.filter(\.isFinite)
        let visible = res.track?.visible.filter { $0 }.count ?? 0
        print("[analyze] \(url.lastPathComponent): \(res.meta.nFrames) frames @\(Int(res.meta.fps.rounded())) fps, "
              + "camera \(res.camera != nil ? "yes" : "no"), visible \(visible), hits \(res.hits.count), "
              + "rallies \(res.rallies.count), shots \(res.rallies.flatMap(\.shots).count), "
              + String(format: "max %.0f km/h, %.1f s", speeds.max() ?? 0, Date().timeIntervalSince(t0)))
    }

    /// The real app path: no taps, the court is found automatically. Python's auto
    /// camera for each clip is in the fixture (`auto_camera`).
    func testAutomaticCalibrationRealClips() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["SHUTTL_LIVE_E2E"] == "1", "set SHUTTL_LIVE_E2E=1")
        let fixture = repoRoot.appendingPathComponent("iOS/Packages/ShuttlLive/Tests/ShuttlLiveTests/Fixtures/live_e2e.json")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: fixture.path), "fixture missing")
        let cases = try XCTUnwrap(ShuttlJSON.load(JSONValue.self, from: fixture)["cases"]?.array)
        let stages = try await LiveStages.make(modelsDirectory: modelsDirectory,
                                               cacheDirectory: FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-live-models"))
        var report = ["clip | auto cam | kp px vs py auto | hits | rallies | max km/h | s"]
        for c in cases {
            let url = repoRoot.appendingPathComponent(try XCTUnwrap(c["relative_path"]?.string))
            try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path), "Test split missing")
            let t0 = Date()
            let res = try await AnalysisPipeline(stages: stages).analyze(url)
            let secs = Date().timeIntervalSince(t0)
            XCTAssertEqual(res.diagnostics["stages"]?["calibrate"]?["ok"]?.bool, true, "\(url.lastPathComponent) calibrate")
            let cam = try XCTUnwrap(res.camera, "\(url.lastPathComponent): Python calibrates this clip automatically")
            let pyCam = try XCTUnwrap(c["auto_camera"]).decodeCamera()
            let (_, kp) = CourtModel.keypointsArray()
            let d = zip(cam.project(kp), pyCam.project(kp)).map { simd_length($0 - $1) }.filter(\.isFinite)
            XCTAssertLessThan(d.max() ?? .infinity, 1.5, "\(url.lastPathComponent) auto camera vs Python")
            XCTAssertFalse(res.segments.isEmpty, "\(url.lastPathComponent): a camera must give 3-D flights")
            let speeds = res.rallies.flatMap(\.shots).compactMap { $0.segment?.speed0Kmh }.filter(\.isFinite)
            report.append("\(url.lastPathComponent) | yes | " + String(format: "%.3f", d.max() ?? .nan)
                          + " | \(res.hits.count) | \(res.rallies.count) | "
                          + String(format: "%.0f | %.1f", speeds.max() ?? 0, secs))
        }
        print("[live auto]\n" + report.joined(separator: "\n"))
    }
}

private extension JSONValue {
    func decodeCamera() throws -> CameraModel {
        try ShuttlJSON.decoder().decode(CameraModel.self, from: ShuttlJSON.encoder().encode(self))
    }
}
