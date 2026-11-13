import Foundation
import XCTest
import ShuttlAnalysis
import ShuttlCore
@testable import ShuttlLive

/// The 4-argument calibrateAutomaticDetailed is the one dispatched (through the
/// protocol), the hint reaches the calibrator, and a wrong hint vetoes the camera.
final class FocalHintTests: XCTestCase {
    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }

    func testFocalHintReachesTheCalibratorThroughTheProtocol() async throws {
        let clip = repoRoot.appendingPathComponent("backend/ml/data/raw/shuttlecock_trajectory/TrackNetV2/Test/match1/video/1_07_04.mp4")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: clip.path), "Test split missing")
        let src = try await VideoSource.open(clip)
        let frames = try src.sampleFrames(24) { try ImageU8.rgb(from: $0) }.map(\.value)
        let stages = try await LiveStages.make(modelsDirectory: repoRoot.appendingPathComponent("iOS/Packages/ShuttlDetector/Models"),
                                               cacheDirectory: FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-live-models"),
                                               metricsURL: nil)
        let any: any AnalysisStages = stages
        let free = try any.calibrateAutomaticDetailed(frames: frames, size: src.meta.size, doublesHint: nil, focalHintPx: nil)
        let cam = try XCTUnwrap(free.camera, "this clip calibrates without a hint")
        let f = cam.K[0, 0]

        let right = try any.calibrateAutomaticDetailed(frames: frames, size: src.meta.size, doublesHint: nil, focalHintPx: f * 1.03)
        XCTAssertNotNil(right.camera, "a hint within tolerance keeps the camera")
        XCTAssertEqual(right.diagnostics["reason"]?.string, "ok")
        XCTAssertEqual(right.camera?.K[0, 0] ?? 0, f, accuracy: 1e-9, "the hint vetoes, it never changes the answer")

        let wrong = try any.calibrateAutomaticDetailed(frames: frames, size: src.meta.size, doublesHint: nil, focalHintPx: f * 0.6)
        XCTAssertNil(wrong.camera, "a hint 40 % off must veto the camera")
        XCTAssertEqual(wrong.diagnostics["reason"]?.string, "focal_mismatch",
                       "if this is \"ok\", the default extension (hint dropped) was dispatched")
    }
}
