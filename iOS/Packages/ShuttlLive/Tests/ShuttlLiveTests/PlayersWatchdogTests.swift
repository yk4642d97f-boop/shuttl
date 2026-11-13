import CoreVideo
import Foundation
import XCTest
import ShuttlAnalysis
import ShuttlCore
import ShuttlPlayers
@testable import ShuttlLive

/// P2-4 (phone T6, match 24: > 1 h in players with nothing logged): a players_progress heartbeat
/// that is written while Vision is stuck, and a stall watchdog that fails the stage RETRYABLE so
/// the retry resumes from players_partial.bin. The fake detectors really block their thread.
final class PlayersWatchdogTests: XCTestCase {
    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }

    private var clip: URL {
        repoRoot.appendingPathComponent("backend/ml/data/raw/shuttlecock_trajectory/TrackNetV2/Test/match1/video/1_09_06.mp4")
    }

    private let gate = DispatchSemaphore(value: 0)
    override func tearDown() {
        for _ in 0..<64 { gate.signal() }                  // let abandoned threads go
        super.tearDown()
    }

    private func temp(_ name: String) -> URL {
        let u = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-p2-4-\(name)-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: u) }
        return u
    }

    private func events(_ url: URL) -> [[String: JSONValue]] {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        return text.split(separator: "\n").compactMap {
            if case .object(let o)? = try? ShuttlJSON.decoder().decode(JSONValue.self, from: Data($0.utf8)) { return o }
            return nil
        }
    }

    private func stages(metrics: URL?, stall: Double?, heartbeat: Double?, detector: (@Sendable () -> PoseDetector)?) async throws -> LiveStages {
        var s = try await LiveStages.make(modelsDirectory: repoRoot.appendingPathComponent("iOS/Packages/ShuttlDetector/Models"),
                                          kind: .tracknetv3,
                                          cacheDirectory: FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-live-models"),
                                          metricsURL: metrics)
        s.checkpointSeconds = 1                            // a players save every 30 frames
        s.checkpointWallSeconds = nil
        s.playersStallSeconds = stall
        s.playersHeartbeatSeconds = heartbeat
        s.watchPollSeconds = 0.05
        s.poseDetectorOverride = detector
        return s
    }

    /// Vision, but `perform` blocks forever on sampled frame `blockAt` while `armed`.
    final class BlockingDetector: PoseDetector, @unchecked Sendable {
        let inner = VisionPoseDetector()
        let blockAt: Int, gate: DispatchSemaphore, armed: Flag, delay: Double
        init(blockAt: Int, gate: DispatchSemaphore, armed: Flag, delay: Double = 0) {
            self.blockAt = blockAt; self.gate = gate; self.armed = armed; self.delay = delay
        }
        func detect(frames: [CVPixelBuffer], frameIndices: [Int]) throws -> [FrameDetections] {
            if armed.isSet, frameIndices.contains(blockAt) { gate.wait() }
            if delay > 0 { Thread.sleep(forTimeInterval: delay) }
            return try inner.detect(frames: frames, frameIndices: frameIndices)
        }
    }

    final class Flag: @unchecked Sendable {
        private let lock = NSLock()
        private var v: Bool
        init(_ v: Bool) { self.v = v }
        var isSet: Bool { lock.withLock { v } }
        func set(_ x: Bool) { lock.withLock { v = x } }
    }

    /// The whole app path: Vision blocks at frame 80 → heartbeats show frame 78 frozen → the
    /// watchdog fails players retryable → stopped_after "players", players_partial.bin at 60 →
    /// the retry resumes from 60 and the result equals an uninterrupted run.
    func testBlockedVisionStallsRetryablyAndTheRetryResumes() async throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: clip.path), "Test split missing")
        let metricsURL = temp("metrics.jsonl")
        let armed = Flag(true), gate = self.gate
        let s = try await stages(metrics: metricsURL, stall: 3, heartbeat: 0.3,
                                 detector: { BlockingDetector(blockAt: 80, gate: gate, armed: armed) })
        let store = FileCheckpointStore(directory: temp("ckpt"), fingerprint: "p2-4")

        let t0 = Date()
        let first = try await AnalysisPipeline(stages: s, checkpoints: store).analyze(clip)
        XCTAssertLessThan(Date().timeIntervalSince(t0), 60, "the stall must end the run, not wait for Vision")
        XCTAssertEqual(first.diagnostics["stopped_after"]?.string, "players")
        let failure = try XCTUnwrap(first.stageFailures.first { $0.stage == "players" })
        XCTAssertTrue(failure.retryable)
        XCTAssertTrue(failure.message.contains("players stalled"), failure.message)
        XCTAssertTrue(first.outcome.isRetryable)
        XCTAssertEqual(store.loadPlayersPartial()?.nextFrame, 60)

        let ev = events(metricsURL)
        let beats = ev.filter { $0["event"]?.string == "players_progress" }
        XCTAssertGreaterThanOrEqual(beats.count, 3, "heartbeats keep coming while Vision is stuck")
        XCTAssertEqual(beats.last?["last_frame"]?.double, 78, "the last completed sampled frame")
        XCTAssertGreaterThan(beats.last?["since_progress_s"]?.double ?? 0, 0.5)
        let stall = try XCTUnwrap(ev.first { $0["event"]?.string == "players_stall" })
        XCTAssertEqual(stall["last_frame"]?.double, 78)
        XCTAssertNotNil(ev.first { $0["event"]?.string == "players_start" })
        XCTAssertEqual(ev.last { $0["event"]?.string == "stage" && $0["stage"]?.string == "players" }?["outcome"]?.string, "error")

        // Vision comes back late: the abandoned pass must not touch the partial.
        let before = try Data(contentsOf: store.playersPartialURL)
        gate.signal()
        try await Task.sleep(nanoseconds: 300_000_000)
        XCTAssertEqual(try Data(contentsOf: store.playersPartialURL), before, "no save from an abandoned pass")

        armed.set(false)
        let res = try await AnalysisPipeline(stages: s, checkpoints: store).analyze(clip)
        XCTAssertEqual(res.diagnostics["stages"]?["players"]?["resume_offered_frame"]?.double, 60)
        XCTAssertNil(res.diagnostics["stopped_after"])
        XCTAssertFalse(res.outcome.isRetryable)
        let full = try await AnalysisPipeline(stages: s).analyze(clip)
        XCTAssertEqual(res.players, full.players, "resumed after the stall = uninterrupted")
    }

    /// Slow but moving (a hot phone): 40 ms per sampled frame, a 3-s bound, never a stall.
    func testSlowProgressIsNotAStall() async throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: clip.path), "Test split missing")
        let metricsURL = temp("metrics.jsonl")
        let armed = Flag(false), gate = self.gate
        let s = try await stages(metrics: metricsURL, stall: 3, heartbeat: 0.5,
                                 detector: { BlockingDetector(blockAt: -1, gate: gate, armed: armed, delay: 0.04) })
        let source = try await VideoSource.open(clip)
        let out = try await s.trackPlayers(source, camera: nil, step: 2, doubles: nil, progress: { _ in })
        XCTAssertFalse(out.isEmpty)
        let ev = events(metricsURL)
        XCTAssertNil(ev.first { $0["event"]?.string == "players_stall" })
        let beats = ev.filter { $0["event"]?.string == "players_progress" }
        XCTAssertGreaterThanOrEqual(beats.count, 2)
        XCTAssertEqual(beats.map { $0["frames_done"]?.double ?? 0 }, beats.map { $0["frames_done"]?.double ?? 0 }.sorted())
        XCTAssertTrue(beats.allSatisfy { ($0["since_progress_s"]?.double ?? 99) < 3 })
    }

    /// iOS froze the process (app suspended): one clock jump is excused, not a stall; a stall
    /// made of ordinary poll gaps still fires.
    func testSuspendedTimeIsNotCountedAsAStall() async throws {
        final class ManualClock: @unchecked Sendable {
            private let lock = NSLock(); private var t = 1000.0
            var now: Double { lock.withLock { t } }
            func advance(_ d: Double) { lock.withLock { t += d } }
        }
        final class Events: @unchecked Sendable {
            private let lock = NSLock(); private var e: [(String, [String: JSONValue])] = []
            func add(_ n: String, _ f: [String: JSONValue]) { lock.withLock { e.append((n, f)) } }
            var all: [(String, [String: JSONValue])] { lock.withLock { e } }
        }
        let clock = ManualClock(), log = Events(), gate = self.gate
        let config = StageWatchConfig(stallSeconds: 10, heartbeatSeconds: 1, pollSeconds: 0.02, suspendGapSeconds: 5,
                                      clock: { clock.now })
        let run = Task {
            try await watchedDetached(stage: "players", config: config, onEvent: { log.add($0, $1) }) { watch -> Int in
                watch.frameDone(0)
                gate.wait()                                  // stuck until the test lets it go
                if watch.flag.isSet { throw CancellationError() }
                watch.frameDone(2)
                return 7
            }
        }
        try await Task.sleep(nanoseconds: 100_000_000)
        clock.advance(3600)                                  // an hour suspended in one step
        try await Task.sleep(nanoseconds: 200_000_000)
        XCTAssertFalse(log.all.contains { $0.0 == "players_stall" }, "suspension is not a stall")
        XCTAssertGreaterThanOrEqual(log.all.last { $0.0 == "players_progress" }?.1["suspended_s"]?.double ?? 0, 3600)
        gate.signal()
        let value = try await run.value
        XCTAssertEqual(value, 7)

        // The same 3600 s as ordinary 2-s poll gaps: a stall after ~10 s.
        let log2 = Events()
        let run2 = Task {
            try await watchedDetached(stage: "players", config: config, onEvent: { log2.add($0, $1) }) { watch -> Int in
                gate.wait(); return 1
            }
        }
        for _ in 0..<12 { clock.advance(2); try await Task.sleep(nanoseconds: 60_000_000) }
        do { _ = try await run2.value; XCTFail("must stall") } catch let e as StageStalledError {
            XCTAssertEqual(e.stage, "players")
            XCTAssertEqual(e.lastFrame, -1)
            XCTAssertGreaterThanOrEqual(e.seconds, 10)
        }
        XCTAssertNotNil(log2.all.first { $0.0 == "players_stall" })
    }

    /// Cancelling (scene to background) returns at once even while the thread is stuck in Vision.
    func testCancelReturnsWhileTheStageIsBlocked() async throws {
        let gate = self.gate
        let config = StageWatchConfig(stallSeconds: nil, heartbeatSeconds: nil, pollSeconds: 0.05)
        let run = Task {
            try await watchedDetached(stage: "players", config: config, onEvent: { _, _ in }) { _ -> Int in gate.wait(); return 1 }
        }
        try await Task.sleep(nanoseconds: 100_000_000)
        let t0 = Date()
        run.cancel()
        do { _ = try await run.value; XCTFail("must be cancelled") } catch is CancellationError {}
        XCTAssertLessThan(Date().timeIntervalSince(t0), 1)
    }
}
