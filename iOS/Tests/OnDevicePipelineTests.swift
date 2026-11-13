import XCTest
import simd
import SwiftData
import ShuttlAnalysis
import ShuttlCore
@testable import Shuttl

/// Stands in for the CoreML/Vision pipeline: returns a fixed analysis, or fails, or cancels.
final class FakeAnalyzer: OnDeviceAnalyzer, @unchecked Sendable {
    enum Mode { case succeed, fail, cancel }
    private let lock = NSLock()
    var mode: Mode = .succeed
    private var _calls: [PipelineOptions] = []
    var calls: [PipelineOptions] { lock.withLock { _calls } }

    func analyze(video: URL, options: PipelineOptions, checkpoints: CheckpointStore,
                 progress: @escaping @Sendable (AnalysisProgress) -> Void,
                 stopOnRetryableFailure: Bool) async throws -> MatchAnalysis {
        lock.withLock { _calls.append(options) }
        progress(AnalysisProgress(fraction: 0.3, stage: "detect"))
        progress(AnalysisProgress(fraction: 0.9, stage: "rallies"))
        switch mode {
        case .fail: throw NSError(domain: "fake", code: 1, userInfo: [NSLocalizedDescriptionKey: "no court"])
        case .cancel: throw CancellationError()
        case .succeed: return Self.sample
        }
    }

    static var sample: MatchAnalysis {
        let meta = VideoMeta(fps: 60, width: 1920, height: 1080, nFrames: 600,
                             timestamps: (0..<600).map { Double($0) / 60 })
        let seg = ShotSegment3D(startFrame: 30, endFrame: 80, t0: 0.5, p0: [-5, 0, 2.5], v0: [80, 0, -5], rmsePx: 1.2,
                                frames: [30, 80], positions: [[-5, 0, 2.5], [4, 0, 0]])
        let smash = ShotEventV2(hit: HitEvent(frame: 30, t: 0.5, xyPx: [960, 540], hitter: .near, confidence: 0.9),
                                segment: seg, shotType: .kill, hitterPosWorld: [-5.2, 0.3])
        let clear = ShotEventV2(hit: HitEvent(frame: 300, t: 5, xyPx: [480, 270], hitter: .far, confidence: 0.8),
                                segment: nil, shotType: .lift)
        return MatchAnalysis(meta: meta, rallies: [
            RallyV2(startFrame: 0, endFrame: 200, shots: [smash], endReason: .inCourt, winner: .near),
            RallyV2(startFrame: 250, endFrame: 500, shots: [clear], endReason: .out, winner: .unknown),
        ])
    }
}

@MainActor
final class OnDevicePipelineTests: XCTestCase {
    private var container: ModelContainer!
    private var backend: MockBackend!
    private var analyzer: FakeAnalyzer!
    private var pipeline: MatchPipeline!
    private var videoURL: URL!

    override func setUp() async throws {
        let schema = Schema([Match.self, Rally.self, ShotPoint.self])
        container = try ModelContainer(for: schema, configurations: [ModelConfiguration(isStoredInMemoryOnly: true)])
        backend = MockBackend()
        analyzer = FakeAnalyzer()
        pipeline = MatchPipeline(context: container.mainContext, backend: backend, analyzer: analyzer,
                                 resumesAutomatically: true)
        videoURL = VideoStore.newRecordingURL(now: Date(timeIntervalSince1970: Double.random(in: 1...1_000_000)))
        try Data(repeating: 7, count: 2048).write(to: videoURL)
    }

    override func tearDown() async throws {
        if let videoURL { try? FileManager.default.removeItem(at: videoURL) }
        pipeline = nil
        container = nil
    }

    func testRecordingIsAnalysedOnDeviceWithoutTouchingTheServer() async throws {
        XCTAssertTrue(pipeline.analyzesOnDevice)
        let match = try pipeline.ingestRecording(fileURL: videoURL, durationSeconds: 10, courtType: .doubles)
        XCTAssertEqual(match.status, .analyzing)
        XCTAssertEqual(match.pipelineStep, .local)
        await pipeline.waitUntilIdle()

        XCTAssertEqual(match.status, .done)
        XCTAssertNil(match.pipelineStep)
        XCTAssertNil(match.failureReason)
        let calls = await backend.calls
        XCTAssertEqual(calls, [], "on-device analysis must never call the server")
        XCTAssertEqual(analyzer.calls.first?.doubles, true, "court type is passed as the doubles hint")

        // SwiftData rows mapped like the server's /analysis document.
        XCTAssertEqual(match.rallies.count, 2)
        let rallies = match.rallies.sorted { $0.startFrame < $1.startFrame }
        XCTAssertEqual(rallies[0].winner, .near)
        let smash = try XCTUnwrap(rallies[0].shots.first)
        XCTAssertEqual(smash.shotType, .smash, "kill collapses to the legacy smash")
        XCTAssertEqual(smash.normalizedX, 0.5, accuracy: 1e-6)
        XCTAssertEqual(smash.speedKmh, Float(simd_length(SIMD3<Double>(80, 0, -5)) * 3.6), accuracy: 0.01)
        XCTAssertEqual(smash.xMeters, -5.2, accuracy: 1e-5)
        let lift = try XCTUnwrap(rallies[1].shots.first)
        XCTAssertEqual(lift.shotType, .clear, "lift collapses to the legacy clear")
        XCTAssertEqual(lift.speedKmh, 0, "no 3-D flight → no speed, never a made-up one")

        // The full analysis is stored and reloads.
        let saved = try XCTUnwrap(AnalysisStore.load(match.analysisPath))
        XCTAssertEqual(saved.rallies.count, 2)
        XCTAssertEqual(saved.rallies[0].shots[0].shotType, .kill, "fine-grained type kept in the JSON")
        let video = try XCTUnwrap(match.localVideoURL)
        XCTAssertTrue(FileManager.default.fileExists(atPath: video.path))
        try await pipeline.discard(match)
        XCTAssertNil(AnalysisStore.load(match.analysisPath), "discard removes the analysis file")
        XCTAssertFalse(FileManager.default.fileExists(atPath: video.path), "and the recording itself")
    }

    func testFailureIsRetriedOnThePhone() async throws {
        analyzer.mode = .fail
        let match = try pipeline.ingestRecording(fileURL: videoURL, durationSeconds: 10, courtType: .singles)
        await pipeline.waitUntilIdle()
        XCTAssertEqual(match.status, .failed)
        XCTAssertEqual(match.pipelineStep, .local)
        XCTAssertEqual(match.retryHint, "会在这台手机上重新分析这段视频")
        XCTAssertTrue(match.failureReason?.contains("no court") == true, match.failureReason ?? "nil")

        analyzer.mode = .succeed
        pipeline.retry(match)
        XCTAssertEqual(match.status, .analyzing)
        await pipeline.waitUntilIdle()
        XCTAssertEqual(match.status, .done)
        let calls = await backend.calls
        XCTAssertEqual(calls, [])
    }

    func testCancelledAnalysisIsNotMarkedFailedAndResumesLater() async throws {
        analyzer.mode = .cancel
        let match = try pipeline.ingestRecording(fileURL: videoURL, durationSeconds: 10, courtType: .singles)
        await pipeline.waitUntilIdle()
        XCTAssertEqual(match.status, .analyzing, "an interrupted analysis stays pending, not failed")
        XCTAssertEqual(match.pipelineStep, .local)

        analyzer.mode = .succeed
        pipeline.resumePending()
        await pipeline.waitUntilIdle()
        XCTAssertEqual(match.status, .done)
    }

    func testMissingVideoFails() async throws {
        let match = try pipeline.ingestRecording(fileURL: videoURL, durationSeconds: 10, courtType: .singles)
        try FileManager.default.removeItem(at: videoURL)
        pipeline.cancel(match)
        match.status = .failed
        pipeline.retry(match)
        await pipeline.waitUntilIdle()
        XCTAssertEqual(match.status, .failed)
        XCTAssertEqual(match.failureReason, "找不到这场比赛的本地视频文件")
    }

    func testLegacyServerMatchesStayOnTheServerPath() async throws {
        let legacy = Match(status: .analyzing, localVideoPath: VideoStore.storedPath(for: videoURL),
                           cloudJobID: "job-9")
        container.mainContext.insert(legacy)
        XCTAssertEqual(pipeline.inferredStep(for: legacy), .poll, "a server job in flight keeps polling the server")
        let fresh = Match(status: .failed, localVideoPath: VideoStore.storedPath(for: videoURL))
        XCTAssertEqual(pipeline.inferredStep(for: fresh), .local)
    }

    func testLocalProgressIsReportedWithChineseStage() {
        let m = Match(status: .analyzing, pipelineStep: .local)
        let a = PipelineActivity(phase: .analyzingOnDevice, localFraction: 0.423, localStage: "players")
        XCTAssertEqual(m.progressText(a), "手机分析中 42% · 追踪球员")
        XCTAssertEqual(m.progressFraction(a), 0.423)
        XCTAssertEqual(m.progressText(nil), "等待在手机上分析…")
    }
}
