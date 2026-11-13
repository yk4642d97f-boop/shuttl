import AVFoundation
import CoreML
import Vision
import XCTest
import ShuttlCore
@testable import ShuttlAnalysis

final class OutcomeAndCheckpointTests: XCTestCase {
    nonisolated(unsafe) static var videoURL: URL!

    override class func setUp() {
        super.setUp()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-outcome-\(UUID().uuidString).mov")
        let done = DispatchSemaphore(value: 0)
        Task.detached { try? await writeTestMovie(to: url, frames: 12, fps: 30); done.signal() }
        done.wait()
        videoURL = url
    }

    override class func tearDown() {
        try? FileManager.default.removeItem(at: videoURL)
        super.tearDown()
    }

    private func tempDir() -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-hard-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: d) }
        return d
    }

    private func run(_ stages: ScriptedStages, checkpoints: CheckpointStore = NoCheckpoints(),
                     observer: any StageObserver = NoStageObserver()) async throws -> MatchAnalysis {
        try await AnalysisPipeline(stages: stages, checkpoints: checkpoints, observer: observer).analyze(Self.videoURL)
    }

    static let avInterrupted = NSError(domain: AVFoundationErrorDomain, code: -11847,
                                       userInfo: [NSLocalizedDescriptionKey: "Operation Interrupted"])

    // MARK: Landing guard (robustness F3)

    private static func segment(landing: SIMD2<Double>?) -> ShotSegment3D {
        ShotSegment3D(startFrame: 2, endFrame: 7, t0: 0.1, p0: [0, 0, 2], v0: [10, 0, 0], rmsePx: 1,
                      frames: [2, 7], positions: [[0, 0, 2], [1, 0, 1]], landingXY: landing, landingT: landing == nil ? nil : 0.4)
    }

    private static func rally(_ start: Int, landing: SIMD2<Double>, winner: CourtSide) -> RallyV2 {
        let hit = HitEvent(frame: start + 1, t: Double(start) / 30, xyPx: [2, 10], hitter: .near)
        return RallyV2(startFrame: start, endFrame: start + 4, shots: [ShotEventV2(hit: hit, segment: segment(landing: landing))],
                       endReason: .out, landingXY: landing, inOutMarginM: -4.4, winner: winner, confidence: 0.9,
                       outcomeConfidence: 0.9, outcomeSource: "3d")
    }

    func testLandingGuardBounds() {
        XCTAssertTrue(LandingGuard.isPlausible([6.7 + 3.0, 3.05 + 3.0]))
        XCTAssertTrue(LandingGuard.isPlausible([-7.5, 0.2]), "a long out stays a real call")
        XCTAssertFalse(LandingGuard.isPlausible([-11.1, -0.3]), "the F3 repro landing")
        XCTAssertFalse(LandingGuard.isPlausible([0, 6.2]))
        XCTAssertFalse(LandingGuard.isPlausible([.nan, 0]))
    }

    func testImplausibleLandingIsWithdrawnBeforeScoring() async throws {
        let stages = ScriptedStages()
        stages.rallies = [Self.rally(0, landing: [-11.1, -0.3], winner: .far), Self.rally(6, landing: [-7.2, 0.5], winner: .far)]
        let res = try await run(stages)
        let bad = res.rallies[0], good = res.rallies[1]
        XCTAssertNil(bad.landingXY); XCTAssertNil(bad.inOutMarginM)
        XCTAssertEqual(bad.winner, .unknown); XCTAssertEqual(bad.endReason, .unknown)
        XCTAssertEqual(bad.outcomeSource, LandingGuard.outcomeSource)
        XCTAssertNil(bad.shots[0].segment?.landingXY, "no dot drawn 4 m behind the baseline")
        XCTAssertNotNil(bad.shots[0].segment, "the flight itself (speed) is kept")
        XCTAssertEqual(good.winner, .far); XCTAssertEqual(good.landingXY, [-7.2, 0.5]); XCTAssertEqual(good.outcomeSource, "3d")
        let scored = try XCTUnwrap(stages.ralliesSeenByScore)
        XCTAssertEqual(scored.map(\.winner), [.unknown, .far], "the score never counts the impossible call")
        XCTAssertEqual(res.diagnostics[LandingGuard.diagnosticsKey]?.double, 2, "rally outcome + its shot segment")
        XCTAssertEqual(res.undecidedByImplausibleLanding, 1)
        XCTAssertTrue(try XCTUnwrap(res.landingWarning).contains("1 个回合"))
        XCTAssertEqual(res.outcome, .complete, "the other rallies are still shown")
    }

    func testPlausibleLandingsLeaveNoTrace() async throws {
        let stages = ScriptedStages()
        stages.rallies = [Self.rally(0, landing: [5.9, 1.0], winner: .near)]
        let res = try await run(stages)
        XCTAssertNil(res.diagnostics[LandingGuard.diagnosticsKey])
        XCTAssertNil(res.landingWarning)
        XCTAssertEqual(res.rallies[0].landingXY, [5.9, 1.0])
    }

    // MARK: P1a — checkpoints survive a failed stage

    func testRetryableDetectFailureStopsTheRunAndRetryCompletes() async throws {
        let dir = tempDir()
        let store = FileCheckpointStore(directory: dir, fingerprint: "fp")
        let failing = ScriptedStages(); failing.detectError = Self.avInterrupted
        let res = try await run(failing, checkpoints: store)
        XCTAssertNil(res.track)
        XCTAssertEqual(res.outcome, .stageFailed(stage: "detect", message: res.stageFailures[0].message, retryable: true))
        XCTAssertEqual(res.outcome.userMessage, "分析被打断了，回到 App 后会自动继续。")
        // F-P3: a retryable detect failure ends the run; nothing after detect runs interrupted.
        XCTAssertEqual(failing.calls, ["detect"], "no stage after a retryable detect failure")
        XCTAssertEqual(res.diagnostics["stopped_after"]?.string, "detect")
        let fm = FileManager.default
        for key in ["candidates.json", "camera.json", "players.bin"] {
            XCTAssertFalse(fm.fileExists(atPath: dir.appendingPathComponent(key).path), key)
        }

        let retry = ScriptedStages()
        let ok = try await run(retry, checkpoints: store)
        XCTAssertEqual(retry.calls.filter { ["detect", "auto", "players"].contains($0) }, ["detect", "auto", "players"])
        XCTAssertEqual(ok.outcome, .complete)
        XCTAssertNil(ok.diagnostics["stopped_after"])
        XCTAssertFalse(fm.fileExists(atPath: dir.path), "cleared after a run without failures")
    }

    func testFailedHitsKeepsTrackCheckpointAndPlayersRoundTripIntoHits() async throws {
        let dir = tempDir()
        let store = FileCheckpointStore(directory: dir, fingerprint: "fp")
        let failing = ScriptedStages(); failing.hitsError = DetectorLikeError.logic
        let res = try await run(failing, checkpoints: store)
        XCTAssertEqual(res.outcome, .stageFailed(stage: "hits", message: "DetectorLikeError: logic", retryable: false))
        XCTAssertTrue(res.outcome.userMessage.hasPrefix("判断击球"))
        for key in ["candidates.json", "track.json", "camera.json", "players.bin"] {
            XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent(key).path), key)
        }
        let retry = ScriptedStages(); retry.players = []        // would produce nothing if it ran
        let ok = try await run(retry, checkpoints: store)
        XCTAssertFalse(retry.calls.contains("players"))
        assertBitIdentical(try XCTUnwrap(retry.playersSeenByHits), failing.players)
        XCTAssertEqual(ok.outcome, .complete)
    }

    func testCancelAfterPlayersResumesWithIdenticalObservations() async throws {
        let dir = tempDir()
        let store = FileCheckpointStore(directory: dir, fingerprint: "fp")
        let killed = ScriptedStages(); killed.cancelInHits = true
        do { _ = try await run(killed, checkpoints: store); XCTFail("expected cancel") } catch is CancellationError {}
        let resumed = ScriptedStages(); resumed.players = []
        let res = try await run(resumed, checkpoints: store)
        XCTAssertEqual(resumed.calls.first, "hits", "everything before hits came from checkpoints")
        assertBitIdentical(res.players, killed.players)
    }

    func testShotFailureOnEarlyRallyIsRecordedEvenWhenLastRallySucceeds() async throws {
        let dir = tempDir()
        let store = FileCheckpointStore(directory: dir, fingerprint: "fp")
        let stages = ScriptedStages(); stages.shotsError = [0: DetectorLikeError.logic]
        let res = try await run(stages, checkpoints: store)
        XCTAssertEqual(res.diagnostics["stages"]?["shots"]?["ok"]?.bool, true, "Python keeps the last record")
        XCTAssertEqual(res.stageFailures.map(\.stage), ["shots"])
        XCTAssertEqual(res.outcome, .complete, "shot types only enrich")
        XCTAssertEqual(res.degradedStages(), ["shots"])
        XCTAssertTrue(res.hasUsableResult())
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir.path), "a failed stage keeps checkpoints")
    }

    func testRepeatedFailuresAreCounted() async throws {
        let stages = ScriptedStages(); stages.shotsError = [0: DetectorLikeError.logic, 6: DetectorLikeError.logic]
        let res = try await run(stages)
        XCTAssertEqual(res.stageFailures.count, 1)
        XCTAssertEqual(res.stageFailures.first?.occurrences, 2)
        XCTAssertEqual(res.diagnostics["stages"]?["shots"]?["ok"]?.bool, false)
    }

    // MARK: P1b/c — classification

    func testFailureDiagnosticsCarryDomainCodeAndRetryable() async throws {
        let stages = ScriptedStages(); stages.detectError = Self.avInterrupted
        let res = try await run(stages)
        let e = try XCTUnwrap(res.diagnostics["stages"]?["detect"])
        XCTAssertEqual(e["ok"]?.bool, false)
        XCTAssertEqual(e["error_domain"]?.string, AVFoundationErrorDomain)
        XCTAssertEqual(e["error_code"]?.double, Double(-11847))
        XCTAssertEqual(e["retryable"]?.bool, true)
        XCTAssertEqual(e["error"]?.string?.hasPrefix("NSError"), true)
        XCTAssertEqual(res.diagnostics[StageFailure.diagnosticsKey]?.array?.count, 1)
    }

    func testRetryableClassifier() {
        typealias C = AnalysisErrorClassifier
        XCTAssertTrue(C.isRetryable(Self.avInterrupted))
        XCTAssertTrue(C.isRetryable(NSError(domain: MLModelErrorDomain, code: MLModelError.Code.io.rawValue)))
        XCTAssertTrue(C.isRetryable(NSError(domain: VNErrorDomain, code: 9)))
        XCTAssertTrue(C.isRetryable(CancellationError()))
        XCTAssertTrue(C.isRetryable(DetectorLikeError.prediction(
            "Error computing NN outputs: Insufficient Permission (to submit GPU work from background)")))
        XCTAssertTrue(C.isRetryable(ShuttlCoreError.readerFailed("Operation Interrupted")))
        XCTAssertTrue(C.isRetryable(NSError(domain: NSCocoaErrorDomain, code: 1,
                                            userInfo: [NSUnderlyingErrorKey: NSError(domain: AVFoundationErrorDomain, code: -11819)])))
        XCTAssertFalse(C.isRetryable(DetectorLikeError.logic))
        XCTAssertFalse(C.isRetryable(ShuttlCoreError.noVideoTrack(URL(fileURLWithPath: "/x.mov"))))
        XCTAssertFalse(C.isRetryable(DetectorLikeError.prediction("detector model unavailable: TrackNetV3 does not exist")))
        XCTAssertFalse(C.isRetryable(NSError(domain: MLModelErrorDomain, code: 0,
                                             userInfo: [NSLocalizedDescriptionKey: "model unavailable"])),
                       "permanent markers win over the domain")
        XCTAssertFalse(C.isRetryable(message: "background subtraction produced no frames", domain: ""))
        XCTAssertTrue(C.isRetryable(message: "MLModelError: io", domain: ""))
        // VideoIO's decode watchdog is retryable by its case, whatever its text says.
        XCTAssertTrue(C.isRetryable(ShuttlCoreError.decodeStalled(seconds: 20, frame: 1234)))
        let stalled = StageFailure(stage: "players", error: ShuttlCoreError.decodeStalled(seconds: 20, frame: 1234))
        XCTAssertTrue(stalled.retryable)
        XCTAssertTrue(StageFailure(json: stalled.json)?.retryable ?? false, "survives the saved diagnostics")
        // ShuttlDetector DetectorError.backgroundChanged (same wording): the checkpoint has the good bytes.
        XCTAssertTrue(C.isRetryable(DetectorLikeError.prediction(
            "background plan segments [15] changed during detection (memory damage)")))
        XCTAssertTrue(C.isRetryable(message: "DetectorError: background plan segments [3, 15] changed during detection (memory damage)",
                                    domain: "ShuttlDetector.DetectorError"))
    }

    func testOutcomeFromContent() {
        let meta = VideoMeta(fps: 30, width: 64, height: 48, nFrames: 1000, timestamps: [])
        func track(visible: Int) -> Track2D {
            Track2D(t: Array(repeating: 0, count: 1000), xy: Array(repeating: SIMD2(0, 0), count: 1000),
                    conf: Array(repeating: 0, count: 1000),
                    visible: (0..<1000).map { $0 < visible }, interpolated: Array(repeating: false, count: 1000))
        }
        XCTAssertEqual(MatchAnalysis(meta: meta).outcome, .noShuttle, "no track")
        XCTAssertEqual(MatchAnalysis(meta: meta, track: track(visible: 29)).outcome, .noShuttle)
        XCTAssertEqual(MatchAnalysis(meta: meta, track: track(visible: 30)).outcome, .noRallies)
        let big = VideoMeta(fps: 60, width: 64, height: 48, nFrames: 100_000, timestamps: [])
        var a = MatchAnalysis(meta: big, track: track(visible: 500))
        XCTAssertEqual(a.outcome, .noShuttle, "1 % of 100k frames = 1000 needed")
        a.rallies = [RallyV2(startFrame: 0, endFrame: 10)]
        XCTAssertEqual(a.outcome, .complete, "rallies win over a thin track")
        XCTAssertEqual(MatchAnalysis(meta: meta, track: track(visible: 5)).outcome(policy: .init(minVisibleFrames: 1, minVisibleFraction: 0)),
                       .noRallies)
        XCTAssertEqual(AnalysisOutcome.noShuttle.userMessage, "没有看到羽毛球。请确认整片球场都在画面里，光线充足。")
        XCTAssertFalse(AnalysisOutcome.noRallies.userMessage.isEmpty)
        XCTAssertEqual(AnalysisOutcome.complete.userMessage, "分析完成。")
    }

    func testRetryableEnrichmentFailureStillAsksForRetry() {
        var a = MatchAnalysis(meta: VideoMeta(fps: 30, width: 1, height: 1, nFrames: 10, timestamps: []))
        a.rallies = [RallyV2(startFrame: 0, endFrame: 5)]
        a.diagnostics[StageFailure.diagnosticsKey] = .array([
            StageFailure(stage: "players", message: "NSError: interrupted", domain: VNErrorDomain, code: 9, retryable: true).json,
        ])
        XCTAssertEqual(a.outcome, .stageFailed(stage: "players", message: "NSError: interrupted", retryable: true))
        XCTAssertTrue(a.hasUsableResult(), "if retries run out, the rallies are still worth showing")
    }

    func testLegacyDiagnosticsWithoutFailureListStillClassify() {
        var a = MatchAnalysis(meta: VideoMeta(fps: 30, width: 1, height: 1, nFrames: 10, timestamps: []))
        a.diagnostics["stages"] = .object([
            "detect": .object(["ok": .bool(false), "error": .string("DetectorError: detector prediction failed: Error computing NN outputs")]),
            "calibrate": .object(["ok": .bool(true)]),
        ])
        XCTAssertEqual(a.stageFailures.map(\.stage), ["detect"])
        XCTAssertEqual(a.outcome.isRetryable, true)
    }

    func testThrownErrorOutsideStages() {
        let o = AnalysisOutcome.failure(ShuttlCoreError.noVideoTrack(URL(fileURLWithPath: "/a.mov")))
        guard case .stageFailed(let stage, _, let retryable) = o else { return XCTFail() }
        XCTAssertEqual(stage, "open"); XCTAssertFalse(retryable)
        XCTAssertTrue(o.userMessage.hasPrefix("打开视频"))
        XCTAssertTrue(AnalysisOutcome.failure(Self.avInterrupted).isRetryable)
    }

    // MARK: P5 — observer

    func testObserverSeesEveryRunStageAndItsKeysAreMerged() async throws {
        let dir = tempDir()
        let store = FileCheckpointStore(directory: dir, fingerprint: "fp")
        let obs = RecordingObserver()
        let stages = ScriptedStages(); stages.hitsError = DetectorLikeError.logic
        let res = try await run(stages, checkpoints: store, observer: obs)
        XCTAssertEqual(obs.events.filter { $0.hasPrefix("start:") }.map { String($0.dropFirst(6)) },
                       ["detect", "link", "calibrate", "players", "hits", "reconstruct3d", "rallies", "score"])
        XCTAssertEqual(obs.events.filter { $0.hasPrefix("start:") }.count, obs.events.filter { $0.hasPrefix("finish:") }.count)
        let hits = try XCTUnwrap(res.diagnostics["stages"]?["hits"])
        XCTAssertEqual(hits["peak_phys_footprint_mb"]?.double, 123.4)
        XCTAssertEqual(hits["thermal_state"]?.string, "nominal")
        XCTAssertEqual(hits["ok"]?.bool, false, "observer keys never override the pipeline's")
        XCTAssertEqual(res.diagnostics["stages"]?["detect"]?["ok"]?.bool, true)

        let obs2 = RecordingObserver()
        _ = try await run(ScriptedStages(), checkpoints: store, observer: obs2)
        XCTAssertFalse(obs2.events.contains("start:detect"), "resumed stages don't run, so aren't observed")
        XCTAssertTrue(obs2.events.contains("start:hits"))
    }

    // MARK: P6 — fingerprint

    func testFingerprintDependsOnOptions() {
        let a: [String: SIMD2<Double>] = ["nb_ld": [0.75, 0.9], "nb_rd": [0.25, 0.9], "fb_ld": [0.6, 0.3], "fb_rd": [0.4, 0.3]]
        var b = a; b["fb_rd"] = [0.41, 0.3]
        let date = Date(timeIntervalSince1970: 1_790_000_000.123)
        func fp(_ o: PipelineOptions, size: Int = 1000, build: String = "1.0-7", modified: Date? = date) -> String {
            checkpointFingerprint(videoSize: size, modified: modified, build: build, options: o)
        }
        let base = fp(PipelineOptions(manualCorners: a))
        XCTAssertEqual(base, fp(PipelineOptions(manualCorners: a)))
        XCTAssertEqual(base, fp(PipelineOptions(manualCorners: Dictionary(uniqueKeysWithValues: a.map { ($0.key, $0.value) }.reversed()))),
                       "insertion order doesn't matter")
        XCTAssertNotEqual(base, fp(PipelineOptions(manualCorners: b)), "different corners")
        XCTAssertNotEqual(base, fp(PipelineOptions()))
        XCTAssertNotEqual(base, fp(PipelineOptions(doubles: true, manualCorners: a)))
        XCTAssertNotEqual(base, fp(PipelineOptions(manualCorners: a, cornersNormalized: false)))
        XCTAssertNotEqual(base, fp(PipelineOptions(manualCorners: a, cameraHint: FakeStages.camera)))
        XCTAssertNotEqual(base, fp(PipelineOptions(manualCorners: a), size: 1001))
        XCTAssertNotEqual(base, fp(PipelineOptions(manualCorners: a), build: "1.0-8"))
        XCTAssertNotEqual(base, fp(PipelineOptions(manualCorners: a), modified: date.addingTimeInterval(1)))
        XCTAssertNotEqual(base, fp(PipelineOptions(manualCorners: a), modified: nil))
        var hint2 = FakeStages.camera; hint2.confidence = 0.777
        XCTAssertNotEqual(fp(PipelineOptions(cameraHint: FakeStages.camera)), fp(PipelineOptions(cameraHint: hint2)))
        XCTAssertEqual(fp(PipelineOptions(cameraHint: FakeStages.camera)), fp(PipelineOptions(cameraHint: FakeStages.camera)))
        XCTAssertTrue(base.hasPrefix("v\(checkpointFormatVersion)-"))
        XCTAssertEqual(base.count, 3 + 64)
        // NaN corners must not crash and must stay stable.
        let nan = PipelineOptions(manualCorners: ["nb_ld": [.nan, 0.5]])
        XCTAssertEqual(fp(nan), fp(nan))
    }
}

enum DetectorLikeError: Error, CustomStringConvertible {
    case logic
    case prediction(String)
    var description: String {
        switch self {
        case .logic: return "logic"
        case .prediction(let s): return "detector prediction failed: \(s)"
        }
    }
}

final class RecordingObserver: StageObserver, @unchecked Sendable {
    private let lock = NSLock()
    private var _events: [String] = []
    var events: [String] { lock.withLock { _events } }
    func stageDidStart(_ name: String) { lock.withLock { _events.append("start:\(name)") } }
    func stageDidFinish(_ name: String, seconds: Double) { lock.withLock { _events.append("finish:\(name)") } }
    func stageDiagnostics(_ name: String) -> [String: JSONValue] {
        ["peak_phys_footprint_mb": .number(123.4), "thermal_state": .string("nominal"), "ok": .bool(true)]
    }
}
