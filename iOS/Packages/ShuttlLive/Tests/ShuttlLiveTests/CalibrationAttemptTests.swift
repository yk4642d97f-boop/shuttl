import Foundation
import XCTest
import ShuttlAnalysis
import ShuttlCore
@testable import ShuttlCourt
@testable import ShuttlLive

/// LiveStages' calibration evidence has the same keys (and, on the same input, the same
/// values) as Python's calibrate_from_frames diagnostics, so pipeline_v2's and the
/// phone's `calibration_attempt` read alike. Fixture: calibration_attempt.json.
final class CalibrationAttemptTests: XCTestCase {
    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }

    func testSameKeysAndValuesAsPython() async throws {
        let fixture = repoRoot.appendingPathComponent("iOS/Packages/ShuttlLive/Tests/ShuttlLiveTests/Fixtures/calibration_attempt.json")
        let cases = try XCTUnwrap(ShuttlJSON.load(JSONValue.self, from: fixture)["cases"]?.array)
        for c in cases {
            let py = try XCTUnwrap(c["diagnostics"]).objectValue
            let frames: [ImageU8]
            let size: ImageSize
            if c["blank"]?.bool == true {
                frames = Array(repeating: ImageU8(width: 1280, height: 720, channels: 3, repeating: 90), count: 3)
                size = ImageSize(width: 1280, height: 720)
            } else {
                let url = repoRoot.appendingPathComponent(try XCTUnwrap(c["relative_path"]?.string))
                try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path), "Test split missing")
                let src = try await VideoSource.open(url)
                frames = try src.sampleFrames(24) { try ImageU8.rgb(from: $0) }.map(\.value)
                size = src.meta.size
            }
            let (cam, sw) = CourtStage.automaticDetailed(frames, size, nil)
            let name = c["relative_path"]?.string ?? "blank"
            XCTAssertEqual(cam != nil, c["accepted"]?.bool, name)
            XCTAssertEqual(Set(sw.keys), Set(py.keys), "\(name): key sets differ")
            XCTAssertEqual(sw["source"], py["source"], name)
            // Values on decoded real video aren't compared here: AVFoundation vs FFmpeg
            // pixel noise shifts which lines are found, and one more line multiplies n_hyp.
            // Value parity on identical input is ShuttlCourt's AutoTraceTests; the field
            // mapping is testFieldMapping below.
            if c["blank"]?.bool == true { XCTAssertEqual(sw, py, "blank frame: only 'source'") }
        }
    }

    /// Through the protocol and the real pipeline: LiveStages' evidence (not the empty
    /// default extension) ends up in diagnostics["calibration_auto"].
    func testPipelineStoresCalibrationEvidence() async throws {
        let clip = repoRoot.appendingPathComponent("backend/ml/data/raw/shuttlecock_trajectory/TrackNetV2/Test/match1/video/1_07_04.mp4")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: clip.path), "Test split missing")
        let stages = try await LiveStages.make(modelsDirectory: repoRoot.appendingPathComponent("iOS/Packages/ShuttlDetector/Models"),
                                               cacheDirectory: FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-live-models"),
                                               metricsURL: nil)
        let any: any AnalysisStages = stages
        let blank = Array(repeating: ImageU8(width: 1280, height: 720, channels: 3, repeating: 90), count: 3)
        let viaProtocol = try any.calibrateAutomaticDetailed(frames: blank, size: ImageSize(width: 1280, height: 720), doublesHint: nil)
        XCTAssertEqual(viaProtocol.diagnostics["source"]?.string, "median", "default extension (empty) was dispatched")

        let res = try await AnalysisPipeline(stages: stages).analyze(clip)
        let auto = try XCTUnwrap(res.diagnostics["calibration_auto"], "pipeline must store the evidence")
        XCTAssertNotNil(res.camera)
        XCTAssertNotNil(auto["n_t"]?.double)
        XCTAssertNotNil(auto["confidence"]?.double)
        XCTAssertEqual(auto["reason"]?.string, "ok", "an accepted camera reports reason \"ok\"")
    }

    /// Each Swift field lands on its Python key (distinct values catch swaps).
    func testFieldMapping() {
        var d = CourtCalibrationDiagnostics()
        d.source = "frame12"; d.reason = .focalMismatch; d.nHyp = 11; d.nLines = 12; d.nValid = 13
        var s = FitStats()
        s.coverage = 0.21; s.rmse = 0.22; s.nT = 23; s.nL = 24; s.nVis = 25; s.nInl = 26
        s.nAbsent = 27; s.absentFrac = 0.28; s.explained = 29; s.explainedFrac = 0.30; s.tStraddle = true
        d.stats = s
        d.margin = 0.31; d.confidence = 0.32; d.score = 33; d.score2 = 34
        let j = CourtStage.diagnosticsJSON(d, frames: [])
        let want: [String: JSONValue] = [
            "source": .string("frame12"), "reason": .string("focal_mismatch"), "t_straddle": .bool(true), "n_hyp": .number(11), "n_lines": .number(12), "n_valid": .number(13),
            "coverage": .number(0.21), "rmse": .number(0.22), "n_t": .number(23), "n_l": .number(24),
            "n_vis": .number(25), "n_inl": .number(26), "n_absent": .number(27), "absent_frac": .number(0.28),
            "explained": .number(29), "explained_frac": .number(0.30), "margin": .number(0.31),
            "confidence": .number(0.32), "score": .number(33), "score2": .number(34),
        ]
        XCTAssertEqual(j, want)
        // No visible model sample: Python omits explained / explained_frac.
        s.explained = nil; s.explainedFrac = nil; d.stats = s
        let j2 = CourtStage.diagnosticsJSON(d, frames: [])
        XCTAssertNil(j2["explained"]); XCTAssertNil(j2["explained_frac"])
    }
}

private extension JSONValue {
    var objectValue: [String: JSONValue] {
        if case .object(let o) = self { return o }
        return [:]
    }
}
