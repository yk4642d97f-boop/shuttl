import SwiftData
import XCTest
import ShuttlAnalysis
import ShuttlCore
import ShuttlRuntime
@testable import Shuttl

@MainActor
final class PipelineCorrectionsTests: XCTestCase {
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

    private func pipeline(_ analyzer: any OnDeviceAnalyzer) -> MatchPipeline {
        let p = MatchPipeline(context: container.mainContext, backend: MockBackend(), analyzer: analyzer,
                              resumesAutomatically: true)
        p.isAppActive = { true }
        return p
    }

    func testCorrectionFlipsTheWinnerDropsANonRallyAndSurvivesReload() async throws {
        let p = pipeline(FakeAnalyzer())
        let match = try p.ingestRecording(fileURL: videoURL, durationSeconds: 10, courtType: .singles)
        await p.waitUntilIdle()
        XCTAssertEqual(match.status, .done)
        let second = FakeAnalyzer.sample.rallies[1].startFrame
        let first = FakeAnalyzer.sample.rallies[0].startFrame

        try await p.correctRally(match, startFrame: second, fix: .farWon)
        var rows = match.rallies.sorted { $0.startFrame < $1.startFrame }
        XCTAssertEqual(rows.map(\.winner), [.near, .far])
        let reloaded = try XCTUnwrap(AnalysisStore.load(match.analysisPath))
        XCTAssertEqual(reloaded.rallies[1].winner, .far)
        XCTAssertEqual(reloaded.rallies[1].outcomeSource, "user")
        XCTAssertFalse(reloaded.score.isEmpty, "the score is recomputed from the corrected rallies")
        XCTAssertEqual(AnalysisStore.loadOriginal(match.analysisPath)?.rallies[1].winner, .unknown,
                       "the AI's result is kept untouched")

        try await p.correctRally(match, startFrame: first, fix: .notARally)
        rows = match.rallies.sorted { $0.startFrame < $1.startFrame }
        XCTAssertEqual(rows.count, 1)

        await p.clearCorrections(match)
        XCTAssertEqual(match.rallies.count, 2)
        XCTAssertEqual(AnalysisStore.load(match.analysisPath)?.rallies[1].winner, .unknown)
        try await p.discard(match)
    }

    func testCameraHintAndCornersReachThePipelineOptions() async throws {
        let analyzer = FakeAnalyzer()
        let p = pipeline(analyzer)
        let hint = CameraModel.lookAt(eye: [-9, 0, 1.6], target: [0, 0, 0], focalPx: 1400,
                                      imageSize: ImageSize(width: 1920, height: 1080))
        let match = try p.ingestRecording(fileURL: videoURL, durationSeconds: 10, courtType: .doubles, cameraHint: hint)
        await p.waitUntilIdle()
        assertSameCamera(analyzer.calls.last?.cameraHint, hint)
        XCTAssertNil(analyzer.calls.last?.manualCorners)

        let taps: [String: CGPoint] = ["nb_ld": CGPoint(x: 0.9, y: 0.9), "nb_rd": CGPoint(x: 0.1, y: 0.9),
                                       "fb_ld": CGPoint(x: 0.7, y: 0.3), "fb_rd": CGPoint(x: 0.3, y: 0.3)]
        p.reanalyze(match, corners: taps)
        XCTAssertEqual(match.status, .analyzing)
        await p.waitUntilIdle()
        XCTAssertEqual(match.status, .done)
        let options = try XCTUnwrap(analyzer.calls.last)
        XCTAssertEqual(options.manualCorners?["nb_ld"], SIMD2(0.9, 0.9))
        assertSameCamera(options.cameraHint, hint, "the lock hint is kept alongside the taps")
        XCTAssertTrue(options.cornersNormalized)
        try await p.discard(match)
        XCTAssertEqual(MatchInputs.load(matchID: match.id), MatchInputs(), "discard removes the inputs")
    }

    /// Same camera after a JSON round trip: it projects the court corners to the same pixels.
    private func assertSameCamera(_ got: CameraModel?, _ want: CameraModel, _ message: String = "",
                                  file: StaticString = #filePath, line: UInt = #line) {
        guard let got else { return XCTFail("no camera hint \(message)", file: file, line: line) }
        for corner: SIMD3<Double> in [[-6.7, -3.05, 0], [-6.7, 3.05, 0], [6.7, -3.05, 0], [6.7, 3.05, 0]] {
            let a = got.project(corner), b = want.project(corner)
            XCTAssertEqual(a.x, b.x, accuracy: 1e-6, message, file: file, line: line)
            XCTAssertEqual(a.y, b.y, accuracy: 1e-6, message, file: file, line: line)
        }
    }

    func testLensFocalReachesThePipelineAndSurvivesReanalysis() async throws {
        let analyzer = FakeAnalyzer()
        let p = pipeline(analyzer)
        let match = try p.ingestRecording(fileURL: videoURL, durationSeconds: 10, courtType: .singles, focalHintPx: 1412.5)
        await p.waitUntilIdle()
        XCTAssertEqual(analyzer.calls.last?.focalHintPx, 1412.5)
        p.reanalyze(match, corners: ["nb_ld": CGPoint(x: 0.9, y: 0.9), "nb_rd": CGPoint(x: 0.1, y: 0.9),
                                     "fb_ld": CGPoint(x: 0.7, y: 0.3), "fb_rd": CGPoint(x: 0.3, y: 0.3)])
        await p.waitUntilIdle()
        XCTAssertEqual(analyzer.calls.last?.focalHintPx, 1412.5, "the lens focal is kept with the taps")
        try await p.discard(match)
    }

    func testDiscardRemovesTheVideoToo() async throws {
        let p = pipeline(FakeAnalyzer())
        let match = try p.ingestRecording(fileURL: videoURL, durationSeconds: 10, courtType: .singles)
        await p.waitUntilIdle()
        let path = try XCTUnwrap(match.analysisPath)
        try await p.discard(match)
        XCTAssertFalse(FileManager.default.fileExists(atPath: videoURL.path), "no orphaned recordings")
        XCTAssertNil(AnalysisStore.load(path))
    }

    private func admissionLines(_ url: URL) throws -> [[String: Any]] {
        guard FileManager.default.fileExists(atPath: url.path) else { return [] }
        return try String(contentsOf: url, encoding: .utf8).split(separator: "\n").compactMap {
            try JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any]
        }
    }

    private func logURL() -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("admission-\(UUID().uuidString).jsonl")
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    func testThermalWaitIsLoggedWithItsReasonAndDuration() async throws {
        let gate = GatedAnalyzer()
        let p = pipeline(gate)
        let log = logURL()
        p.admissionLog = AdmissionLog(url: log)
        var decision: ThermalDecision = .hold
        p.thermalGate = { decision }
        p.thermalLevel = { decision == .run ? .fair : .serious }

        let a = try p.ingestRecording(fileURL: videoURL, durationSeconds: 10, courtType: .singles)
        try await Task.sleep(for: .milliseconds(500))
        XCTAssertEqual(gate.started, 0)
        XCTAssertEqual(p.admissionBlock(), .thermal)
        decision = .run
        try await Task.sleep(for: .milliseconds(700))
        XCTAssertEqual(gate.started, 1)
        gate.release()
        await p.waitUntilIdle()
        XCTAssertEqual(a.status, .done)

        let lines = try admissionLines(log)
        XCTAssertEqual(lines.map { $0["phase"] as? String }, ["begin", "end"])
        XCTAssertEqual(lines.map { $0["reason"] as? String }, ["thermal", "thermal"])
        XCTAssertEqual(lines[0]["thermal"] as? String, "serious")
        XCTAssertEqual(lines[0]["camera_live"] as? Bool, false)
        XCTAssertEqual(lines[1]["outcome"] as? String, "admitted")
        XCTAssertEqual(lines[1]["thermal"] as? String, "fair")
        XCTAssertGreaterThan(lines[1]["seconds"] as? Double ?? 0, 0.3)
        XCTAssertEqual(lines[0]["event"] as? String, "admission_wait")
    }

    func testCameraLivePausesAndBlocksAnalysisUntilItStops() async throws {
        let gate = GatedAnalyzer()
        let p = pipeline(gate)
        let log = logURL()
        p.admissionLog = AdmissionLog(url: log)

        p.cameraDidStart()
        XCTAssertFalse(p.hasLocalWork)
        let a = try p.ingestRecording(fileURL: videoURL, durationSeconds: 10, courtType: .singles)
        try await Task.sleep(for: .milliseconds(500))
        XCTAssertEqual(gate.started, 0, "no decode while the camera runs")
        XCTAssertTrue(p.hasLocalWork, "waiting counts as local work")
        XCTAssertEqual(p.activity(for: a)?.localStage, "camera")
        XCTAssertEqual(a.progressText(p.activity(for: a)), "相机开着 · 离开录像页或相机休息后就开始")

        p.cameraDidStop()
        try await Task.sleep(for: .milliseconds(700))
        XCTAssertEqual(gate.started, 1, "starts once the camera stops")

        // The camera coming back while it runs pauses it (checkpoints keep the work) …
        p.cameraDidStart()
        XCTAssertEqual(a.status, .analyzing)
        XCTAssertEqual(p.activity(for: a)?.localStage, "camera")
        // … and it runs again when the camera stops.
        p.cameraDidStop()
        try await Task.sleep(for: .milliseconds(700))
        XCTAssertEqual(gate.started, 2)
        XCTAssertTrue(p.hasLocalWork)
        gate.release()
        await p.waitUntilIdle()
        XCTAssertEqual(a.status, .done)
        XCTAssertFalse(p.hasLocalWork)
        XCTAssertEqual(try admissionLines(log).first?["reason"] as? String, "camera")
    }

    private func pauseLines(_ url: URL) throws -> [[String: Any]] {
        try admissionLines(url).filter { ["analysis_paused", "analysis_resumed"].contains($0["event"] as? String) }
    }

    func testCameraPauseMidRunIsLoggedAndTheCardSaysWhy() async throws {
        let gate = GatedAnalyzer()
        gate.progressSteps = [AnalysisProgress(fraction: 0.10 + 0.40 * 0.25, stage: "detect")]
        let p = pipeline(gate)
        p.frameCount = { _ in 3600 }
        let log = logURL()
        p.admissionLog = AdmissionLog(url: log)

        let a = try p.ingestRecording(fileURL: videoURL, durationSeconds: 60, courtType: .singles)
        try await Task.sleep(for: .milliseconds(400))
        XCTAssertEqual(gate.started, 1)
        XCTAssertEqual(p.activity(for: a)?.localStage, "detect")

        p.cameraDidStart()
        let card = p.activity(for: a)
        XCTAssertEqual(card?.localStage, "camera")
        XCTAssertEqual(MatchWaiting(localStage: card?.localStage), .camera)
        XCTAssertEqual(MatchWaiting.camera.label, "等相机")
        XCTAssertEqual(AnalysisStageLadder.waitingLine(status: a.status, activity: card), "等相机",
                       "the progress card shows only the short reason; the hero has the sentence")
        XCTAssertEqual(a.progressText(card), "相机开着 · 离开录像页或相机休息后就开始")
        try await Task.sleep(for: .milliseconds(400))
        XCTAssertEqual(gate.started, 1, "nothing restarts while the camera runs")
        XCTAssertEqual(a.status, .analyzing)
        XCTAssertEqual(a.progressText(p.activity(for: a)), "相机开着 · 离开录像页或相机休息后就开始",
                       "never blank or 等待在手机上分析 while paused")

        p.cameraDidStop()
        try await Task.sleep(for: .milliseconds(400))
        XCTAssertEqual(gate.started, 2, "runs again once the camera stops")
        gate.release()
        await p.waitUntilIdle()
        XCTAssertEqual(a.status, .done)

        let lines = try admissionLines(log)
        XCTAssertEqual(lines.map { $0["event"] as? String }, ["analysis_paused", "analysis_resumed"],
                       "a pause mid-run is not an admission wait")
        XCTAssertEqual(lines[0]["reason"] as? String, "camera")
        XCTAssertEqual(lines[0]["stage"] as? String, "detect")
        XCTAssertEqual(lines[0]["frame"] as? Double ?? -1, 900, accuracy: 1)
        XCTAssertEqual(lines[0]["frames"] as? Double, 3600)
        XCTAssertEqual(lines[0]["camera_live"] as? Bool, true)
        XCTAssertEqual(lines[1]["reason"] as? String, "camera")
        XCTAssertGreaterThan(lines[1]["seconds"] as? Double ?? 0, 0.35)
        for line in lines {
            XCTAssertEqual(line["match"] as? String, a.id.uuidString, "judge ties a pause to its match row")
            XCTAssertEqual(line["video"] as? String, videoURL.lastPathComponent)
        }
    }

    func testTakeAndHeatPausesAreLoggedButAJobOnlyWaitingIsNot() async throws {
        let gate = GatedAnalyzer()
        gate.progressSteps = [AnalysisProgress(fraction: 0.60 + 0.20 * 0.25, stage: "players")]
        let p = pipeline(gate)
        p.frameCount = { _ in 3600 }
        let log = logURL()
        p.admissionLog = AdmissionLog(url: log)
        var decision: ThermalDecision = .run
        p.thermalGate = { decision }

        let a = try p.ingestRecording(fileURL: videoURL, durationSeconds: 60, courtType: .singles)
        try await Task.sleep(for: .milliseconds(400))
        XCTAssertEqual(gate.started, 1)

        p.recordingDidStart()
        XCTAssertEqual(a.progressText(p.activity(for: a)), "正在录像，录完接着分析")
        p.recordingDidStop()
        try await Task.sleep(for: .milliseconds(400))
        XCTAssertEqual(gate.started, 2)

        decision = .pause
        p.thermalPauseDidBegin()
        XCTAssertEqual(p.activity(for: a)?.localStage, "cooling")
        decision = .hold
        p.resumePending()
        try await Task.sleep(for: .milliseconds(400))
        XCTAssertEqual(gate.started, 2, "still too hot")
        // The camera stopping a job that is only waiting its turn is not a second pause.
        p.cameraDidStart()
        XCTAssertEqual(p.activity(for: a)?.localStage, "camera")
        try await Task.sleep(for: .milliseconds(400))
        p.cameraDidStop()
        try await Task.sleep(for: .milliseconds(400))
        decision = .run
        try await Task.sleep(for: .milliseconds(700))
        XCTAssertEqual(gate.started, 3)
        gate.release()
        await p.waitUntilIdle()
        XCTAssertEqual(a.status, .done)

        let lines = try pauseLines(log)
        XCTAssertEqual(lines.map { $0["event"] as? String },
                       ["analysis_paused", "analysis_resumed", "analysis_paused", "analysis_resumed"])
        XCTAssertEqual(lines.map { $0["reason"] as? String }, ["recording", "recording", "thermal", "thermal"])
        XCTAssertEqual(lines[2]["stage"] as? String, "players")
        XCTAssertEqual(lines[2]["frame"] as? Double ?? -1, 900, accuracy: 1)
        XCTAssertGreaterThan(lines[3]["seconds"] as? Double ?? 0, 1.1, "the whole pause, across the waits")
    }

    func testProgressCardShowsOnlyTheShortWaitingReason() {
        let stages = ["queued": "排队中", "cooling": "降温中", "cooling_camera": "降温中", "camera": "等相机", "recording": "等录完"]
        for (stage, label) in stages {
            let activity = PipelineActivity(phase: .analyzingOnDevice, localStage: stage)
            XCTAssertEqual(AnalysisStageLadder.waitingLine(status: .analyzing, activity: activity), label, stage)
        }
        XCTAssertNil(AnalysisStageLadder.waitingLine(status: .analyzing,
                                                     activity: PipelineActivity(phase: .analyzingOnDevice, localStage: "detect")))
        XCTAssertEqual(AnalysisStageLadder.waitingLine(status: .uploading, activity: nil), "上传完成后从第一步开始")
    }

    func testPauseFrameMapsTheStageFraction() {
        XCTAssertEqual(MatchPipeline.frame(of: AnalysisProgress(fraction: 0.30, stage: "detect"), frames: 1000), 500)
        XCTAssertEqual(MatchPipeline.frame(of: AnalysisProgress(fraction: 0.80, stage: "players"), frames: 1000), 1000)
        XCTAssertNil(MatchPipeline.frame(of: AnalysisProgress(fraction: 0.55, stage: "calibrate"), frames: 1000))
    }

    func testVideoFrameCountReadsTheMovie() async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("frames-\(UUID().uuidString).mov")
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        try await PipelineDebugInboxTests.writeVideo(to: url, frames: 60, fps: 30)
        let n = await MatchPipeline.videoFrameCount(url)
        XCTAssertEqual(Double(n ?? 0), 60, accuracy: 1)
        let junk = await MatchPipeline.videoFrameCount(videoURL)
        XCTAssertNil(junk, "not a movie: no frame count, the pause logs frame null")
    }

    func testAdmissionReasonOrder() {
        let p = pipeline(FakeAnalyzer())
        XCTAssertNil(p.admissionBlock())
        p.thermalGate = { .hold }
        XCTAssertEqual(p.admissionBlock(), .thermal)
        XCTAssertEqual(AdmissionBlock.thermal.localStage(cameraLive: false), "cooling")
        XCTAssertEqual(AnalysisStageName.display(AdmissionBlock.thermal.localStage(cameraLive: true)),
                       "降温中 · 手机凉下来就开始 · 录像页相机也在发热")
        p.cameraDidStart()
        XCTAssertEqual(p.admissionBlock(), .camera)
        p.recordingDidStart()
        XCTAssertEqual(p.admissionBlock(), .recording, "a take outranks everything")
        p.recordingDidStop(); p.cameraDidStop()
        XCTAssertEqual(p.admissionBlock(), .thermal)
    }

    func testOnlyOneAnalysisRunsAtATimeAndHeatHoldsTheQueue() async throws {
        let gate = GatedAnalyzer()
        let p = pipeline(gate)
        var decision: ThermalDecision = .hold
        p.thermalGate = { decision }

        let a = try p.ingestRecording(fileURL: videoURL, durationSeconds: 10, courtType: .singles)
        let otherURL = VideoStore.newRecordingURL(now: Date(timeIntervalSince1970: Double.random(in: 1...1_000_000)))
        try Data(repeating: 7, count: 2048).write(to: otherURL)
        defer { try? FileManager.default.removeItem(at: otherURL) }
        let b = try p.ingestRecording(fileURL: otherURL, durationSeconds: 10, courtType: .singles)

        try await Task.sleep(for: .milliseconds(700))
        XCTAssertEqual(gate.started, 0, "nothing starts while the phone is hot")
        // Waiting states read only their own line (monitor review 16:58, fix 1): no "手机分析中 ·" prefix.
        XCTAssertEqual(a.progressText(p.activity(for: a)), "降温中 · 手机凉下来就开始")

        decision = .run
        try await Task.sleep(for: .milliseconds(900))
        XCTAssertEqual(gate.started, 1, "one at a time")
        XCTAssertTrue([a, b].contains { $0.progressText(p.activity(for: $0)).contains("排队中") })

        gate.release()
        await p.waitUntilIdle()
        XCTAssertEqual(gate.started, 2)
        XCTAssertEqual(a.status, .done)
        XCTAssertEqual(b.status, .done)
        try await p.discard(a); try await p.discard(b)
    }
}

/// Blocks each analysis until `release()`, so concurrency can be observed.
final class GatedAnalyzer: OnDeviceAnalyzer, @unchecked Sendable {
    private let lock = NSLock()
    private var _started = 0
    private var released = false
    /// Reported at the start of every run (set before the first run).
    var progressSteps: [AnalysisProgress] = []
    var started: Int { lock.withLock { _started } }
    func release() { lock.withLock { released = true } }

    func analyze(video: URL, options: PipelineOptions, checkpoints: CheckpointStore,
                 progress: @escaping @Sendable (AnalysisProgress) -> Void,
                 stopOnRetryableFailure: Bool) async throws -> MatchAnalysis {
        lock.withLock { _started += 1 }
        progressSteps.forEach(progress)
        while !lock.withLock({ released }) { try await Task.sleep(for: .milliseconds(20)) }
        return FakeAnalyzer.sample
    }
}
