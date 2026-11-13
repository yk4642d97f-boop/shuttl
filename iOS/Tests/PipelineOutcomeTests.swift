import SwiftData
import XCTest
import ShuttlAnalysis
import ShuttlCore
@testable import Shuttl

/// Returns a queue of scripted analyses (one per call) so each outcome path can be driven.
/// When the script entry has stopped_after="players", it returns withoutRallies variant when
/// stopOnRetryableFailure: true, or withRallies variant when stopOnRetryableFailure: false.
final class ScriptedAnalyzer: OnDeviceAnalyzer, @unchecked Sendable {
    private let lock = NSLock()
    private var script: [MatchAnalysis]
    private var _calls = 0
    var calls: Int { lock.withLock { _calls } }
    var lastStopOnRetryableFailure: Bool?

    init(_ script: [MatchAnalysis]) { self.script = script }

    func analyze(video: URL, options: PipelineOptions, checkpoints: CheckpointStore,
                 progress: @escaping @Sendable (AnalysisProgress) -> Void,
                 stopOnRetryableFailure: Bool) async throws -> MatchAnalysis {
        lock.withLock {
            _calls += 1
            lastStopOnRetryableFailure = stopOnRetryableFailure
            // Get the template (never mutate it directly to avoid shared-reference bugs)
            let template = script.count > 1 ? script.removeFirst() : script[0]

            // Create a fresh copy so Array(repeating:count:) in tests doesn't share one instance
            var result = template

            // Match real AnalysisPipeline contract: if result has stopped_after="players",
            // return it as-is when stopOnRetryableFailure: true (no rallies),
            // or add rallies when stopOnRetryableFailure: false (like the real pipeline continuing)
            if !stopOnRetryableFailure,
               case .string(let stage) = result.diagnostics["stopped_after"],
               stage == "players",
               result.rallies.isEmpty {
                // Real AnalysisPipeline would continue to rallies stage; simulate that
                result.rallies = [RallyV2(startFrame: 100, endFrame: 200, winner: .near)]
                // Don't clear stopped_after - the failure still happened, rallies just ran anyway
            }
            return result
        }
    }
}

@MainActor
final class PipelineOutcomeTests: XCTestCase {
    private var container: ModelContainer!
    private var videoURL: URL!

    override func setUp() async throws {
        let schema = Schema([Match.self, Rally.self, ShotPoint.self])
        container = try ModelContainer(for: schema, configurations: [ModelConfiguration(isStoredInMemoryOnly: true)])
        videoURL = VideoStore.newRecordingURL(now: Date(timeIntervalSince1970: Double.random(in: 1...1_000_000)))
        try Data(repeating: 7, count: 2048).write(to: videoURL)
    }

    override func tearDown() async throws {
        if let videoURL { try? FileManager.default.removeItem(at: videoURL) }
        container = nil
    }

    private func makePipeline(_ analyzer: ScriptedAnalyzer, active: Bool = true) -> MatchPipeline {
        let p = MatchPipeline(context: container.mainContext, backend: MockBackend(), analyzer: analyzer,
                              resumesAutomatically: true)
        p.isAppActive = { active }
        p.retryDelay = { _ in }
        return p
    }

    private static let meta = VideoMeta(fps: 60, width: 1920, height: 1080, nFrames: 600,
                                        timestamps: (0..<600).map { Double($0) / 60 })

    /// No track at all: the shuttle was never seen.
    private static var noShuttle: MatchAnalysis { MatchAnalysis(meta: meta) }

    /// A detect stage the OS refused (backgrounded CoreML), recorded the way StageRunner does.
    private static var interrupted: MatchAnalysis {
        var a = MatchAnalysis(meta: meta)
        let failure = StageFailure(stage: "detect", message: "MLModelError: error computing NN outputs",
                                   domain: "com.apple.CoreML", code: 0, retryable: true)
        a.diagnostics[StageFailure.diagnosticsKey] = .array([failure.json])
        return a
    }

    /// Players failed retryable, stopped before rallies (P2 stopOnRetryableFailure: true behavior).
    private static var playersRetryableWithoutRallies: MatchAnalysis {
        var a = MatchAnalysis(meta: meta, track: Track2D(t: meta.timestamps, xy: Array(repeating: [100, 100], count: 600),
                                                          conf: Array(repeating: 0.9, count: 600),
                                                          visible: Array(repeating: true, count: 600),
                                                          interpolated: Array(repeating: false, count: 600)))
        let failure = StageFailure(stage: "players", message: "VNError: interrupted", domain: "com.apple.Vision",
                                   code: 0, retryable: true)
        a.diagnostics[StageFailure.diagnosticsKey] = .array([failure.json])
        a.diagnostics["stopped_after"] = .string("players")  // P2 sets this when stopOnRetryableFailure: true
        return a
    }

    /// Players failed retryable, but rallies ran anyway (P2 stopOnRetryableFailure: false behavior).
    private static var playersRetryableWithRallies: MatchAnalysis {
        var a = MatchAnalysis(meta: meta, track: Track2D(t: meta.timestamps, xy: Array(repeating: [100, 100], count: 600),
                                                          conf: Array(repeating: 0.9, count: 600),
                                                          visible: Array(repeating: true, count: 600),
                                                          interpolated: Array(repeating: false, count: 600)))
        let failure = StageFailure(stage: "players", message: "VNError: interrupted", domain: "com.apple.Vision",
                                   code: 0, retryable: true)
        a.diagnostics[StageFailure.diagnosticsKey] = .array([failure.json])
        // Add a rally so hasUsableResult() returns true
        a.rallies = [RallyV2(startFrame: 100, endFrame: 200, winner: .near)]
        return a
    }

    /// Calibrate's sampleFrames threw -11847 (retryable), then players also failed and set stopped_after="players".
    /// This is the on-device case the monitor found: outcome="calibrate" but stopped_after="players".
    private static var calibrateAndPlayersRetryable: MatchAnalysis {
        var a = MatchAnalysis(meta: meta, track: Track2D(t: meta.timestamps, xy: Array(repeating: [100, 100], count: 600),
                                                          conf: Array(repeating: 0.9, count: 600),
                                                          visible: Array(repeating: true, count: 600),
                                                          interpolated: Array(repeating: false, count: 600)))
        // Calibrate failed first (outcome), then players failed and stopped the run
        let calibrateFailure = StageFailure(stage: "calibrate", message: "AVError -11847", domain: "AVFoundation",
                                            code: -11847, retryable: true)
        let playersFailure = StageFailure(stage: "players", message: "VNError: interrupted", domain: "com.apple.Vision",
                                          code: 0, retryable: true)
        a.diagnostics[StageFailure.diagnosticsKey] = .array([calibrateFailure.json, playersFailure.json])
        a.diagnostics["stopped_after"] = .string("players")
        return a
    }

    func testNoShuttleFailsWithAPlainReasonInsteadOfAnEmptyDone() async throws {
        let analyzer = ScriptedAnalyzer([Self.noShuttle])
        let pipeline = makePipeline(analyzer)
        let match = try pipeline.ingestRecording(fileURL: videoURL, durationSeconds: 10, courtType: .singles)
        await pipeline.waitUntilIdle()
        XCTAssertEqual(match.status, .failed)
        XCTAssertEqual(match.failureReason, analysisOutcomeMessage(.noShuttle))
        XCTAssertEqual(match.pipelineStep, .local, "重试 stays on the phone")
        XCTAssertNil(match.analysisPath, "nothing is saved for an unusable run")
        XCTAssertEqual(match.failureRetryable, false, "the same video can only fail the same way")
        match.pipelineStep = .local
        pipeline.retry(match)
        XCTAssertNil(match.failureRetryable, "cleared while it runs again")
        await pipeline.waitUntilIdle()
        XCTAssertEqual(match.failureRetryable, false)
    }

    func testInterruptedRunIsRetriedOnceInTheForeground() async throws {
        let analyzer = ScriptedAnalyzer([Self.interrupted, FakeAnalyzer.sample])
        let pipeline = makePipeline(analyzer, active: true)
        let match = try pipeline.ingestRecording(fileURL: videoURL, durationSeconds: 10, courtType: .singles)
        await pipeline.waitUntilIdle()
        XCTAssertEqual(analyzer.calls, 2)
        XCTAssertEqual(match.status, .done)
        XCTAssertEqual(match.rallies.count, 2)
        try await pipeline.discard(match)
    }

    func testInterruptedRunInTheBackgroundStaysPendingForLater() async throws {
        let analyzer = ScriptedAnalyzer([Self.interrupted, FakeAnalyzer.sample])
        let pipeline = makePipeline(analyzer, active: false)
        let match = try pipeline.ingestRecording(fileURL: videoURL, durationSeconds: 10, courtType: .singles)
        await pipeline.waitUntilIdle()
        XCTAssertEqual(analyzer.calls, 1)
        XCTAssertEqual(match.status, .analyzing, "not failed: it resumes when the app is back in front")
        XCTAssertEqual(match.pipelineStep, .local)

        pipeline.isAppActive = { true }
        pipeline.resumePending()
        await pipeline.waitUntilIdle()
        XCTAssertEqual(match.status, .done)
        try await pipeline.discard(match)
    }

    func testRepeatedInterruptionFailsAsRetryable() async throws {
        let analyzer = ScriptedAnalyzer([Self.interrupted])
        let pipeline = makePipeline(analyzer, active: true)
        let match = try pipeline.ingestRecording(fileURL: videoURL, durationSeconds: 10, courtType: .singles)
        await pipeline.waitUntilIdle()
        XCTAssertEqual(analyzer.calls, MatchPipeline.maxLocalAttempts, "retries with backoff, then it stops")
        XCTAssertEqual(match.status, .failed)
        XCTAssertEqual(match.failureReason, analysisOutcomeMessage(.stageFailed(stage: "detect", message: "", retryable: true)))
        XCTAssertEqual(match.failureRetryable, true, "an interruption may go through next time")
    }

    func testAllAttemptsFailingWithPlayersRetryableProducesDegradedResult() async throws {
        // Attempts 0-5: all have players failure with stopped_after="players"
        // Attempts 0-4: stopOnRetryableFailure: true → ScriptedAnalyzer returns as-is (no rallies)
        // Attempt 5 (last): stopOnRetryableFailure: false → ScriptedAnalyzer adds rallies (simulating continuation)
        let script = Array(repeating: Self.playersRetryableWithoutRallies, count: 6)
        let analyzer = ScriptedAnalyzer(script)
        let pipeline = makePipeline(analyzer, active: true)
        let match = try pipeline.ingestRecording(fileURL: videoURL, durationSeconds: 10, courtType: .singles)
        await pipeline.waitUntilIdle()
        XCTAssertEqual(analyzer.calls, MatchPipeline.maxLocalAttempts, "retries, then last attempt runs with stopOnRetryableFailure: false")
        XCTAssertEqual(match.status, .done, "last attempt continues to rallies and produces a degraded result")
        XCTAssertNotNil(match.analysisPath, "the degraded result is saved")
        XCTAssertEqual(analyzer.lastStopOnRetryableFailure, false, "last attempt had stopOnRetryableFailure: false")
    }

    func testPlayersRetryableOnceRecoveriesOnSecondAttempt() async throws {
        // Attempt 0: players fails retryable → retry
        // Attempt 1: succeeds with full result
        let analyzer = ScriptedAnalyzer([Self.playersRetryableWithoutRallies,
                                         FakeAnalyzer.sample])
        let pipeline = makePipeline(analyzer, active: true)
        let match = try pipeline.ingestRecording(fileURL: videoURL, durationSeconds: 10, courtType: .singles)
        await pipeline.waitUntilIdle()
        XCTAssertEqual(analyzer.calls, 2, "retries once after players failure")
        XCTAssertEqual(match.status, .done, "second attempt succeeds")
        XCTAssertNotNil(match.analysisPath)
        XCTAssertEqual(match.rallies.count, 2, "full result with players tracking")
    }

    func testPlayersRetryableOnEveryAttemptProducesDegradedResult() async throws {
        // Attempts 0-4: players fails retryable, stopped before rallies (stopOnRetryableFailure: true)
        // Attempt 5 (last): players fails retryable, but rallies ran (stopOnRetryableFailure: false)
        let script = Array(repeating: Self.playersRetryableWithoutRallies, count: 5) + [Self.playersRetryableWithRallies]
        let analyzer = ScriptedAnalyzer(script)
        let pipeline = makePipeline(analyzer, active: true)
        let match = try pipeline.ingestRecording(fileURL: videoURL, durationSeconds: 10, courtType: .singles)
        await pipeline.waitUntilIdle()
        XCTAssertEqual(analyzer.calls, MatchPipeline.maxLocalAttempts, "retries up to max attempts")
        XCTAssertEqual(match.status, .done, "last attempt produces degraded result (rallies without players)")
        XCTAssertNotNil(match.analysisPath)
        XCTAssertEqual(match.rallies.count, 1, "rallies present despite players failure")
    }

    /// Detect and link ran fine and found nothing; only an enrichment stage was interrupted.
    private static var noShuttleWithInterruptedPlayers: MatchAnalysis {
        var a = MatchAnalysis(meta: meta, track: Track2D(t: meta.timestamps, xy: Array(repeating: [.nan, .nan], count: 600),
                                                           conf: Array(repeating: 0, count: 600),
                                                           visible: Array(repeating: false, count: 600),
                                                           interpolated: Array(repeating: false, count: 600)))
        let f = StageFailure(stage: "players", message: "VNError: interrupted", domain: "com.apple.Vision",
                             code: 0, retryable: true)
        a.diagnostics[StageFailure.diagnosticsKey] = .array([f.json])
        return a
    }

    func testANoShuttleVerdictIsNotReanalysedJustBecauseAnotherStageWasInterrupted() async throws {
        let analyzer = ScriptedAnalyzer([Self.noShuttleWithInterruptedPlayers])
        let pipeline = makePipeline(analyzer)
        let match = try pipeline.ingestRecording(fileURL: videoURL, durationSeconds: 10, courtType: .singles)
        await pipeline.waitUntilIdle()
        XCTAssertEqual(analyzer.calls, 1, "the same video can't grow a shuttle on a second run")
        XCTAssertEqual(match.status, .failed)
    }

    func testCalibrateAndPlayersRetryableRetriesDespiteMismatchedOutcome() async throws {
        // Calibrate failed first (outcome stage), players failed with stopped_after="players".
        // Monitor's R4 requirement: any stopped_after → retry, even when outcome != stopped_after.
        let analyzer = ScriptedAnalyzer([Self.calibrateAndPlayersRetryable, FakeAnalyzer.sample])
        let pipeline = makePipeline(analyzer, active: true)
        let match = try pipeline.ingestRecording(fileURL: videoURL, durationSeconds: 10, courtType: .singles)
        await pipeline.waitUntilIdle()
        XCTAssertEqual(analyzer.calls, 2, "stopped_after players triggers retry despite outcome=calibrate")
        XCTAssertEqual(match.status, .done, "second attempt succeeds")
        XCTAssertNotNil(match.analysisPath)
    }

    func testRecordingPausesAnalysisAndResumesAfterwards() async throws {
        let gate = GatedAnalyzer()
        let p = MatchPipeline(context: container.mainContext, backend: MockBackend(), analyzer: gate,
                              resumesAutomatically: true)
        p.isAppActive = { true }
        p.recordingDidStart()
        let match = try p.ingestRecording(fileURL: videoURL, durationSeconds: 10, courtType: .singles)
        try await Task.sleep(for: .milliseconds(700))
        XCTAssertEqual(gate.started, 0, "nothing starts while a take is being recorded")
        // A waiting row shows only why it waits (monitor 14:58 fix 3).
        XCTAssertEqual(match.progressText(p.activity(for: match)), "正在录像，录完接着分析")
        p.recordingDidStop()
        try await Task.sleep(for: .milliseconds(700))
        XCTAssertEqual(gate.started, 1)
        p.recordingDidStart()
        XCTAssertEqual(match.status, .analyzing, "a take interrupting the analysis leaves it pending, never failed")
        gate.release()
        p.recordingDidStop()
        await p.waitUntilIdle()
        XCTAssertEqual(match.status, .done)
        try await p.discard(match)
    }

    func testSavedLibraryCopyDropsPoseKeypoints() throws {
        var a = FakeAnalyzer.sample
        a.players = [PlayerObservation(frame: 0, trackID: 1, bbox: BBox(x1: 0, y1: 0, x2: 10, y2: 20), score: 0.9,
                                       side: .near, footWorld: [-4, 0.5],
                                       keypoints: Array(repeating: Keypoint(x: 1, y: 2, conf: 0.5), count: 17))]
        let docs = FileManager.default.temporaryDirectory.appendingPathComponent("slim-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: docs) }
        let path = try AnalysisStore.save(a, matchID: UUID(), documents: docs)
        let back = try XCTUnwrap(AnalysisStore.load(path, documents: docs))
        XCTAssertEqual(back.players.count, 1)
        XCTAssertNil(back.players[0].keypoints)
        XCTAssertEqual(back.players[0].footWorld, [-4, 0.5], "movement stats still have feet")
        XCTAssertEqual(back.rallies.count, a.rallies.count)
    }


    func testPlayersRetryableWithRalliesRetries() async throws {
        // Players failed retryable AND rallies exist.
        // Q's behavior: WITH rallies → retry (the rallies.isEmpty check is false, falls through to return true).
        // This ensures we maintain Q's original logic.
        let analyzer = ScriptedAnalyzer([Self.playersRetryableWithRallies, FakeAnalyzer.sample])
        let pipeline = makePipeline(analyzer, active: true)
        let match = try pipeline.ingestRecording(fileURL: videoURL, durationSeconds: 10, courtType: .singles)
        await pipeline.waitUntilIdle()
        XCTAssertEqual(analyzer.calls, 2, "retries when rallies exist (Q's original behavior)")
        XCTAssertEqual(match.status, .done, "second attempt succeeds")
        XCTAssertNotNil(match.analysisPath)
    }

    func testPlayersRetryableWithoutRalliesButValidTrackDoesNotRetry() async throws {
        // Players failed retryable, no rallies, but track has valid positions.
        // Q's behavior: WITHOUT rallies + valid track → no retry (Q's line 605-609 check applies).
        var analysis = Self.playersRetryableWithoutRallies
        analysis.diagnostics.removeValue(forKey: "stopped_after")  // Remove stopped_after to test the enrichment logic
        let analyzer = ScriptedAnalyzer([analysis])
        let pipeline = makePipeline(analyzer, active: true)
        let match = try pipeline.ingestRecording(fileURL: videoURL, durationSeconds: 10, courtType: .singles)
        await pipeline.waitUntilIdle()
        XCTAssertEqual(analyzer.calls, 1, "does not retry when track is valid but no rallies")
        XCTAssertEqual(match.status, .failed, "fails without usable result")
        XCTAssertEqual(match.failureRetryable, false, "not worth retrying")
    }
}
