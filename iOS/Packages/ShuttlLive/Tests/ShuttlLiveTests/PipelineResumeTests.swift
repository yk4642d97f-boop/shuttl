import AVFoundation
import Foundation
import XCTest
import ShuttlAnalysis
import ShuttlCore
@testable import ShuttlLive

/// The app's real path end to end: AnalysisPipeline + FileCheckpointStore + LiveStages.
/// Kill the run once detect has written detect_partial.bin, run it again with the same
/// store, and the analysis equals an uninterrupted run. Gated (real CoreML + Vision):
///   SHUTTL_LIVE_E2E=1 swift test -c release -Xswiftc -enable-testing --filter PipelineResumeTests
/// SHUTTL_RESUME_CLIP=<file> picks another clip; SHUTTL_RESUME_EVERY=<s> the checkpoint interval.
final class PipelineResumeTests: XCTestCase {
    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }

    func testKilledPipelineResumesDetectAndMatches() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["SHUTTL_LIVE_E2E"] == "1", "set SHUTTL_LIVE_E2E=1")
        let clip = env["SHUTTL_RESUME_CLIP"].map { URL(fileURLWithPath: $0) }
            ?? repoRoot.appendingPathComponent("backend/ml/data/raw/shuttlecock_trajectory/TrackNetV2/Test/match1/video/1_07_03.mp4")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: clip.path), "clip missing")
        var stages = try await LiveStages.make(modelsDirectory: repoRoot.appendingPathComponent("iOS/Packages/ShuttlDetector/Models"),
                                               kind: .tracknetv3,
                                               cacheDirectory: FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-live-models"),
                                               metricsURL: nil)
        stages.checkpointSeconds = Double(env["SHUTTL_RESUME_EVERY"] ?? "") ?? 4
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-pipeline-resume-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = FileCheckpointStore(directory: dir, fingerprint: "resume-test")

        let tFull = Date()
        let full = try await AnalysisPipeline(stages: stages).analyze(clip)
        let fullSeconds = Date().timeIntervalSince(tFull)

        // Run 1: cancel once detect_partial.bin has at least two checkpoints (so the
        // append-only delta path is exercised).
        let partialURL = store.detectPartialURL
        let run = Task { [stages] in try await AnalysisPipeline(stages: stages, checkpoints: store).analyze(clip) }
        var sizes: [Int] = []
        let t0 = Date()
        while Date().timeIntervalSince(t0) < max(60, fullSeconds * 3) {
            try await Task.sleep(nanoseconds: 50_000_000)
            let size = (try? FileManager.default.attributesOfItem(atPath: partialURL.path)[.size] as? Int) ?? 0
            if size > 0, sizes.last != size { sizes.append(size) }
            if sizes.count >= 2 { break }
        }
        run.cancel()
        _ = try? await run.value
        let saved = try XCTUnwrap(store.loadDetectPartial(), "detect_partial.bin must survive the kill (sizes \(sizes))")
        XCTAssertGreaterThan(saved.nextFrame, 0)
        print("[pipeline resume] killed with detect_partial.bin at frame \(saved.nextFrame)/\(full.meta.nFrames), "
              + "file sizes seen \(sizes) bytes")

        // Run 2: same store → detect resumes from the partial and the result matches.
        let res = try await AnalysisPipeline(stages: stages, checkpoints: store).analyze(clip)
        XCTAssertEqual(res.diagnostics["stages"]?["detect"]?["resume_offered_frame"]?.double, Double(saved.nextFrame),
                       "the pipeline must offer the saved partial to detect")
        XCTAssertNil(stages.detector.lastResumeProblem, stages.detector.lastResumeProblem ?? "")
        XCTAssertEqual(res.track?.visible, full.track?.visible)
        XCTAssertEqual(res.hits.map(\.frame), full.hits.map(\.frame))
        XCTAssertEqual(res.rallies.map(\.startFrame), full.rallies.map(\.startFrame))
        XCTAssertEqual(res.rallies.flatMap(\.shots).map(\.shotType), full.rallies.flatMap(\.shots).map(\.shotType))
        XCTAssertFalse(FileManager.default.fileExists(atPath: partialURL.path), "cleared once candidates are saved")
    }

    /// Phone pass 09-30 F-P3: detect FAILED mid-way (-11847, decoder taken away) and the app's
    /// in-process retry restarted detect from frame 0. Same process, same store: a detect that
    /// errors after detect_partial.bin was written must resume from it on the next attempt.
    func testDetectErrorThenInProcessRetryResumes() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["SHUTTL_LIVE_E2E"] == "1", "set SHUTTL_LIVE_E2E=1")
        let clip = env["SHUTTL_RESUME_CLIP"].map { URL(fileURLWithPath: $0) }
            ?? repoRoot.appendingPathComponent("backend/ml/data/raw/shuttlecock_trajectory/TrackNetV2/Test/match1/video/1_07_03.mp4")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: clip.path), "clip missing")
        var stages = try await LiveStages.make(modelsDirectory: repoRoot.appendingPathComponent("iOS/Packages/ShuttlDetector/Models"),
                                               kind: .tracknetv3,
                                               cacheDirectory: FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-live-models"),
                                               metricsURL: nil)
        stages.checkpointSeconds = Double(env["SHUTTL_RESUME_EVERY"] ?? "") ?? 4
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-pipeline-err-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir); FrameSequence.copyNextHook = nil; FrameSequence.stallTimeout = 20 }
        let store = FileCheckpointStore(directory: dir, fingerprint: "resume-error-test")
        let full = try await AnalysisPipeline(stages: stages).analyze(clip)

        // Run 1: once detect_partial.bin exists, the decoder stops delivering → decodeStalled.
        let armed = InjectFlag()
        FrameSequence.stallTimeout = 1
        FrameSequence.copyNextHook = { output in
            if armed.isSet { Thread.sleep(forTimeInterval: 2); return nil }
            return output.copyNextSampleBuffer()
        }
        let partialURL = store.detectPartialURL
        let watcher = Task.detached {
            while !Task.isCancelled {
                if FileManager.default.fileExists(atPath: partialURL.path) { armed.set(); return }
                try? await Task.sleep(nanoseconds: 20_000_000)
            }
        }
        let first = try await AnalysisPipeline(stages: stages, checkpoints: store).analyze(clip)
        watcher.cancel()
        FrameSequence.copyNextHook = nil
        FrameSequence.stallTimeout = 20
        XCTAssertEqual(first.diagnostics["stages"]?["detect"]?["ok"]?.bool, false, "detect must have failed in run 1")
        XCTAssertEqual(first.diagnostics["stopped_after"]?.string, "detect", "a retryable detect failure ends the run")
        for later in ["link", "calibrate", "players", "hits"] {
            XCTAssertNil(first.diagnostics["stages"]?[later], "\(later) must not run on the interrupted decoder")
        }
        XCTAssertTrue(first.outcome.isRetryable)
        let saved = try XCTUnwrap(store.loadDetectPartial(), "detect_partial.bin must survive a detect ERROR")
        XCTAssertGreaterThan(saved.nextFrame, 0)

        // Run 2 in the same process (the app's retry loop): resumes from the partial.
        let res = try await AnalysisPipeline(stages: stages, checkpoints: store).analyze(clip)
        XCTAssertEqual(res.diagnostics["stages"]?["detect"]?["resume_offered_frame"]?.double, Double(saved.nextFrame),
                       "the retry must offer the saved partial to detect")
        XCTAssertNil(stages.detector.lastResumeProblem, stages.detector.lastResumeProblem ?? "")
        XCTAssertEqual(res.diagnostics["stages"]?["detect"]?["partial_load"]?.string, "ok")
        XCTAssertEqual(res.track?.visible, full.track?.visible)
        XCTAssertEqual(res.hits.map(\.frame), full.hits.map(\.frame))
        XCTAssertEqual(res.rallies.map(\.startFrame), full.rallies.map(\.startFrame))
    }
}

private final class InjectFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return value }
    func set() { lock.lock(); value = true; lock.unlock() }
}
