import Foundation
import simd
import XCTest
import ShuttlAnalysis
import ShuttlCore
import ShuttlPlayers
@testable import ShuttlLive

/// P2-2 on the app's types: LiveStages' resumable players (Apple Vision) through the protocol
/// and FileCheckpointStore's players_partial.bin, interrupted and resumed: same observations.
final class PlayersResumeLiveTests: XCTestCase {
    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }

    private func stages() async throws -> LiveStages {
        var s = try await LiveStages.make(modelsDirectory: repoRoot.appendingPathComponent("iOS/Packages/ShuttlDetector/Models"),
                                          kind: .tracknetv3,
                                          cacheDirectory: FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-live-models"),
                                          metricsURL: nil)
        s.checkpointSeconds = 1                              // 30 frames → a save every 15 sampled frames
        s.checkpointWallSeconds = nil
        return s
    }

    /// A camera with awkward bits (1/3, a tiny dist term) and its JSON checkpoint copy: the
    /// players resume context must be the same, or a resumed run (camera from camera.json)
    /// would never accept the partial its first run wrote.
    private func camera(_ meta: VideoMeta) -> CameraModel {
        let f = 1.1 * Double(meta.width)
        let K = simd_double3x3(rows: [SIMD3(f, 0, Double(meta.width) / 2), SIMD3(0, f, Double(meta.height) / 2), SIMD3(0, 0, 1)])
        let a = 1.0 / 3
        let R = simd_double3x3(rows: [SIMD3(1, 0, 0), SIMD3(0, -sin(a), -cos(a)), SIMD3(0, cos(a), -sin(a))])
        return CameraModel(K: K, R: R, t: SIMD3(0.1 / 3, 1.7, 9.2), imageSize: ImageSize(width: meta.width, height: meta.height),
                           dist: [1e-7 / 3, 0, 0, 0, 0], reprojRmsePx: 1.0 / 7, confidence: 0.83)
    }

    func testCameraFromItsCheckpointKeepsThePlayersResumeContext() async throws {
        let clip = repoRoot.appendingPathComponent("backend/ml/data/raw/shuttlecock_trajectory/TrackNetV2/Test/match1/video/1_09_06.mp4")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: clip.path), "Test split missing")
        let meta = try await VideoSource.open(clip).meta
        let cam = camera(meta)
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-p2-cam-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = FileCheckpointStore(directory: dir, fingerprint: "fp")
        store.save(Optional(cam), key: "camera")
        let back = try XCTUnwrap(store.load(CameraModel?.self, key: "camera") ?? nil)
        let rois = PlayerTracker.defaultFarROIs(camera: cam, meta: meta)
        XCTAssertEqual(PlayerTracker.defaultFarROIs(camera: back, meta: meta), rois)
        XCTAssertEqual(PlayerTracker.resumeContext(camera: back, meta: meta, step: 2, rois: rois, detector: "v"),
                       PlayerTracker.resumeContext(camera: cam, meta: meta, step: 2, rois: rois, detector: "v"))
    }

    /// Through `any AnalysisStages` (the pipeline's view): saves come from LiveStages, not the
    /// default extension; cancelled at three points, resumed from players_partial.bin each time.
    func testResumingPlayersIsDispatchedAndExactWithVision() async throws {
        let clip = repoRoot.appendingPathComponent("backend/ml/data/raw/shuttlecock_trajectory/TrackNetV2/Test/match1/video/1_09_06.mp4")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: clip.path), "Test split missing")
        let any: any AnalysisStages = try await stages()
        let source = try await VideoSource.open(clip)
        let cam = camera(source.meta)
        let full = try await any.trackPlayers(source, camera: cam, step: 2, doubles: nil, progress: { _ in })
        XCTAssertFalse(full.isEmpty)

        // 133 frames, a save every 30 → saves at 30, 60, 90, 120. Cancel after 1, 2, 3 of them
        // (where the pass really stops depends on when the cancel lands; any save is a valid cut).
        for cutAfter in [1, 2, 3] {
            let dir = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-p2-live-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: dir) }
            let store = FileCheckpointStore(directory: dir, fingerprint: "fp")
            let saves = SaveLog()
            let run = Task {
                try await any.trackPlayers(source, camera: cam, step: 2, doubles: nil, resume: nil,
                                           checkpoint: { c in saves.add(store.savePlayersPartialReporting(c).verdict) },
                                           progress: { _ in })
            }
            let t0 = Date()
            while saves.verdicts.count < cutAfter, Date().timeIntervalSince(t0) < 120 {
                try await Task.sleep(nanoseconds: 2_000_000)
            }
            run.cancel()
            _ = try? await run.value
            let saved = try XCTUnwrap(store.loadPlayersPartial(), "@\(cutAfter): a partial must exist (saves \(saves.verdicts))")
            XCTAssertEqual(saves.verdicts.first, "created")
            XCTAssertTrue(saves.verdicts.dropFirst().allSatisfy { $0 == "appended" }, "\(saves.verdicts)")

            let resumed = try await any.trackPlayers(source, camera: cam, step: 2, doubles: nil, resume: saved,
                                                     checkpoint: { _ in }, progress: { _ in })
            XCTAssertEqual(resumed, full, "@\(cutAfter): resumed from frame \(saved.nextFrame) must equal the uninterrupted run")
        }
    }
}

extension PlayersResumeLiveTests {
    /// The phone's -11847 during players, on the app's whole path (AnalysisPipeline + FileCheckpointStore
    /// + LiveStages): once players_partial.bin exists the decoder stalls → players fails retryable →
    /// stopped_after "players"; the in-process retry (same store) resumes players and matches. Gated:
    ///   SHUTTL_LIVE_E2E=1 swift test -c release --filter PlayersResumeLiveTests
    func testPlayersDecoderErrorStopsAfterPlayersAndTheRetryResumes() async throws {
        let env = ProcessInfo.processInfo.environment
        try XCTSkipUnless(env["SHUTTL_LIVE_E2E"] == "1", "set SHUTTL_LIVE_E2E=1")
        let clip = env["SHUTTL_RESUME_CLIP"].map { URL(fileURLWithPath: $0) }
            ?? repoRoot.appendingPathComponent("backend/ml/data/raw/shuttlecock_trajectory/TrackNetV2/Test/match1/video/1_07_03.mp4")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: clip.path), "clip missing")
        let stages = try await stages()
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-p2-e2e-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir); FrameSequence.copyNextHook = nil; FrameSequence.stallTimeout = 20 }
        let store = FileCheckpointStore(directory: dir, fingerprint: "players-error-test")
        let full = try await AnalysisPipeline(stages: stages).analyze(clip)
        XCTAssertFalse(full.players.isEmpty)

        let armed = Flag()
        FrameSequence.stallTimeout = 1
        FrameSequence.copyNextHook = { output in
            if armed.isSet { Thread.sleep(forTimeInterval: 2); return nil }
            return output.copyNextSampleBuffer()
        }
        let partialURL = store.playersPartialURL
        let watcher = Task.detached {
            while !Task.isCancelled {
                if FileManager.default.fileExists(atPath: partialURL.path) { armed.set(); return }
                try? await Task.sleep(nanoseconds: 5_000_000)
            }
        }
        let first = try await AnalysisPipeline(stages: stages, checkpoints: store).analyze(clip)
        watcher.cancel()
        FrameSequence.copyNextHook = nil
        FrameSequence.stallTimeout = 20
        XCTAssertEqual(first.diagnostics["stages"]?["detect"]?["ok"]?.bool, true)
        XCTAssertEqual(first.diagnostics["stages"]?["players"]?["ok"]?.bool, false, "players must have failed in run 1")
        XCTAssertEqual(first.diagnostics["stopped_after"]?.string, "players", "a retryable players failure ends the run")
        for later in ["hits", "rallies", "score"] { XCTAssertNil(first.diagnostics["stages"]?[later], later) }
        XCTAssertTrue(first.outcome.isRetryable)
        let saved = try XCTUnwrap(store.loadPlayersPartial(), "players_partial.bin must survive a players ERROR")
        let bytes = (try? FileManager.default.attributesOfItem(atPath: partialURL.path)[.size] as? Int) ?? 0
        print("[players e2e] players_partial.bin \(bytes) bytes for \(saved.frames.count) sampled frames")
        XCTAssertGreaterThan(saved.nextFrame, 0)

        let res = try await AnalysisPipeline(stages: stages, checkpoints: store).analyze(clip)
        XCTAssertEqual(res.diagnostics["stages"]?["players"]?["resume_offered_frame"]?.double, Double(saved.nextFrame))
        XCTAssertEqual(res.diagnostics["stages"]?["players"]?["partial_load"]?.string, "ok")
        XCTAssertEqual(res.diagnostics["stages"]?["detect"]?["resumed"]?.bool, true, "detect came from its checkpoint")
        XCTAssertEqual(res.players, full.players, "players resumed from frame \(saved.nextFrame) must equal the uninterrupted run")
        XCTAssertEqual(res.hits.map(\.frame), full.hits.map(\.frame))
        XCTAssertEqual(res.rallies.map(\.startFrame), full.rallies.map(\.startFrame))
        XCTAssertFalse(res.outcome.isRetryable)
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir.path), "cleared after a clean run")
        print("[players e2e] stopped after players with players_partial.bin at frame \(saved.nextFrame)/\(full.meta.nFrames)")
    }
}

private final class Flag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    var isSet: Bool { lock.withLock { value } }
    func set() { lock.withLock { value = true } }
}

private final class SaveLog: @unchecked Sendable {
    private let lock = NSLock()
    private var v: [String] = []
    var verdicts: [String] { lock.withLock { v } }
    func add(_ s: String) { lock.withLock { v.append(s) } }
}
