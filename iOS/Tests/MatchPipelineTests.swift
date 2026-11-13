import XCTest
import SwiftData
import ShuttlAnalysis
import ShuttlCore
@testable import Shuttl

// MARK: - Mock backend

/// Records every call; each endpoint's behaviour is a replaceable closure.
actor MockBackend: MatchBackend {
    private(set) var calls: [String] = []
    private(set) var analyzeDoublesHints: [Bool?] = []
    private(set) var forgottenUploads: [UUID] = []

    var nativeArchive: MatchAnalysis?
    private(set) var lastInputs: MatchInputs?
    func set(archive: MatchAnalysis) { nativeArchive = archive }
    func getAnalysisArchive(matchID: UUID) async throws -> MatchAnalysis? { nativeArchive }
    func triggerAnalysis(matchID: UUID, doubles: Bool?, inputs: MatchInputs) async throws -> JobResponse {
        lastInputs = inputs
        return try await triggerAnalysis(matchID: matchID, doubles: doubles)
    }

    var serverID = UUID()
    var createHook: (@Sendable (UUID) async throws -> MatchResponse)?
    var commandHook: (@Sendable () async throws -> Void)?
    private(set) var remoteDeletes: [UUID] = []
    private(set) var remoteCancels: [String] = []
    func set(createHook: @escaping @Sendable (UUID) async throws -> MatchResponse) { self.createHook = createHook }
    func set(commandHook: @escaping @Sendable () async throws -> Void) { self.commandHook = commandHook }
    func deleteRemote(matchID: UUID, account: RemoteAccount) async throws {
        guard self.account == account else { throw APIError.unauthorized }
        remoteDeletes.append(matchID)
        try await commandHook?()
    }
    func cancelRemote(matchID: UUID, jobID: String, account: RemoteAccount) async throws {
        guard self.account == account else { throw APIError.unauthorized }
        remoteCancels.append(jobID)
        try await commandHook?()
    }
    var createHandler: (@Sendable () throws -> MatchResponse)?
    var getMatchHandler: @Sendable (UUID) throws -> MatchResponse
    var uploadURLHandler: @Sendable (UUID) throws -> UploadURLResponse
    var uploadHandler: @Sendable () throws -> Void = {}
    var triggerHandler: @Sendable (UUID) throws -> JobResponse = { _ in JobResponse(jobId: "job-1", status: "queued") }
    var jobStates: [String] = ["SUCCESS"]
    var pollErrors: [Error] = []
    var analysisHandler: @Sendable (UUID) throws -> AnalysisDocumentDTO = { _ in try MockBackend.sampleDocument() }
    var trajectoryHandler: @Sendable (UUID) throws -> TrajectoryResponse = { id in
        TrajectoryResponse(matchId: id, fps: 60, points: [])
    }

    init() {
        getMatchHandler = { MockBackend.remote($0, status: "done") }
        uploadURLHandler = { _ in
            UploadURLResponse(uploadUrl: URL(string: "http://127.0.0.1:8000/matches/x/video?token=t")!, ossKey: "videos/u/x.mov")
        }
    }

    func set(create: @escaping @Sendable () throws -> MatchResponse) { createHandler = create }
    func set(getMatch: @escaping @Sendable (UUID) throws -> MatchResponse) { getMatchHandler = getMatch }
    func set(uploadURL: @escaping @Sendable (UUID) throws -> UploadURLResponse) { uploadURLHandler = uploadURL }
    func set(upload: @escaping @Sendable () throws -> Void) { uploadHandler = upload }
    func set(trigger: @escaping @Sendable (UUID) throws -> JobResponse) { triggerHandler = trigger }
    func set(jobStates: [String]) { self.jobStates = jobStates }
    func set(pollErrors: [Error]) { self.pollErrors = pollErrors }
    func set(analysis: @escaping @Sendable (UUID) throws -> AnalysisDocumentDTO) { analysisHandler = analysis }
    func set(trajectory: @escaping @Sendable (UUID) throws -> TrajectoryResponse) { trajectoryHandler = trajectory }
    func set(serverID: UUID) { self.serverID = serverID }

    static func remote(_ id: UUID, status: String, jobID: String? = nil) -> MatchResponse {
        MatchResponse(id: id, status: status, durationSeconds: 12, videoOssKey: nil, celeryJobId: jobID, createdAt: .now)
    }

    var account: RemoteAccount? = RemoteAccount(baseURL: "http://unit.test", userID: "unit-user")
    func remoteAccount() async -> RemoteAccount? { account }
    func set(account: RemoteAccount?) { self.account = account }

    // MARK: MatchBackend

    func createMatch(id: UUID, durationSeconds: Double) async throws -> MatchResponse {
        calls.append("create"); serverID = id
        if let createHook { return try await createHook(id) }
        return try createHandler?() ?? Self.remote(id, status: "uploading")
    }
    func getMatch(id: UUID) async throws -> MatchResponse {
        calls.append("getMatch"); return try getMatchHandler(id)
    }
    func getUploadURL(matchID: UUID) async throws -> UploadURLResponse {
        calls.append("uploadURL"); return try uploadURLHandler(matchID)
    }
    func uploadVideo(matchID: UUID, fileURL: URL, progress: @escaping @Sendable (Double) -> Void) async throws {
        _ = try await getUploadURL(matchID: matchID)
        calls.append("upload")
        progress(0.5)
        try uploadHandler()
        progress(1)
    }
    func forgetUpload(matchID: UUID) async throws { forgottenUploads.append(matchID) }
    func triggerAnalysis(matchID: UUID, doubles: Bool?) async throws -> JobResponse {
        calls.append("analyze"); analyzeDoublesHints.append(doubles); return try triggerHandler(matchID)
    }
    func pollJob(matchID: UUID, jobID: String) async throws -> JobStatusResponse {
        calls.append("poll")
        if !pollErrors.isEmpty { throw pollErrors.removeFirst() }
        let state = jobStates.count > 1 ? jobStates.removeFirst() : (jobStates.first ?? "SUCCESS")
        let info = state == "PROGRESS" ? "{'match_id': 'x', 'percent': 42, 'stage': 'detect'}" : "None"
        return JobStatusResponse(jobId: jobID, state: state, info: state == "FAILURE" ? "ValueError('court not found')" : info)
    }
    func getTrajectory(matchID: UUID) async throws -> TrajectoryResponse {
        calls.append("trajectory"); return try trajectoryHandler(matchID)
    }
    func getAnalysis(matchID: UUID) async throws -> AnalysisDocumentDTO {
        calls.append("analysis"); return try analysisHandler(matchID)
    }

    /// Shaped like backend serialize.to_dict (snake_case, nulls included).
    static func sampleDocument() throws -> AnalysisDocumentDTO {
        let json = """
        {"schema_version": 2, "video": {"fps": 60.0, "width": 1920, "height": 1080, "frames": 900},
         "rallies": [
           {"start_frame": 10, "end_frame": 300, "winner": "near", "shots": [
             {"frame": 40, "xy_norm": [0.25, 0.5], "shot_type": "serve", "legacy_shot_type": "serve",
              "hitter_pos": [1.0, -5.0], "flight": {"speed0_kmh": 80.5}},
             {"frame": 90, "xy_norm": [0.75, 0.3], "shot_type": "kill", "legacy_shot_type": "smash",
              "hitter_pos": null, "flight": null}
           ]},
           {"start_frame": 500, "end_frame": 700, "winner": null, "shots": [
             {"frame": 520, "xy_norm": [null, 0.4], "shot_type": "lift", "legacy_shot_type": "clear",
              "hitter_pos": [0.5, 2.0], "flight": {"speed0_kmh": null}}
           ]}
         ]}
        """
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(AnalysisDocumentDTO.self, from: Data(json.utf8))
    }
}

// MARK: - Pipeline tests

@MainActor
final class MatchPipelineTests: XCTestCase {
    private var container: ModelContainer!
    private var context: ModelContext { container.mainContext }
    private var backend: MockBackend!
    private var pipeline: MatchPipeline!
    private var videoURL: URL!

    private static let fastPolling = JobPollingService.Options(
        interval: .milliseconds(1), maxBackoff: .milliseconds(2),
        pendingTimeout: .milliseconds(20), overallTimeout: .seconds(5), maxConsecutiveFailures: 3
    )

    override func setUp() async throws {
        let schema = Schema([Match.self, Rally.self, ShotPoint.self])
        container = try ModelContainer(for: schema, configurations: [ModelConfiguration(isStoredInMemoryOnly: true)])
        backend = MockBackend()
        pipeline = MatchPipeline(context: container.mainContext, backend: backend,
                                 pollOptions: Self.fastPolling, resumesAutomatically: true)
        videoURL = VideoStore.newRecordingURL(now: Date(timeIntervalSince1970: Double.random(in: 1...1_000_000)))
        try Data(repeating: 7, count: 2048).write(to: videoURL)
    }

    override func tearDown() async throws {
        if let videoURL { try? FileManager.default.removeItem(at: videoURL) }
        pipeline = nil
        container = nil
    }

    private func ingest(court: CourtType = .singles) throws -> Match {
        try pipeline.ingestRecording(fileURL: videoURL, durationSeconds: 12.5, courtType: court)
    }

    func testCloudArchiveFeedsExistingRichReportsAndForwardsCalibration() async throws {
        let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/cloud_archive.json")
        let archive = try ShuttlJSON.decoder().decode(MatchAnalysis.self, from: Data(contentsOf: fixture))
        await backend.set(archive: archive)
        let match = try pipeline.ingestRecording(fileURL: videoURL, durationSeconds: 12.5, courtType: .singles,
                                                  cameraHint: archive.camera)
        defer { AnalysisStore.delete(match.analysisPath); MatchInputs.delete(matchID: match.id) }
        await pipeline.waitUntilIdle()
        XCTAssertEqual(match.status, .done)
        let restored = try XCTUnwrap(AnalysisStore.load(match.analysisPath))
        XCTAssertEqual(restored.meta.nFrames, archive.meta.nFrames)
        XCTAssertEqual(restored.players.first?.trackID, 7)
        XCTAssertEqual(restored.rallies[0].shots[0].shotType, .kill)
        XCTAssertEqual(restored.track?.xy.count, 100)
        XCTAssertTrue(restored.track!.xy[0].x.isNaN)
        XCTAssertEqual(match.rallies[0].shots[0].shotType, .smash)
        let inputs = await backend.lastInputs
        XCTAssertEqual(inputs?.cameraHint?.imageSize.width, 1920)
        let calls = await backend.calls
        XCTAssertFalse(calls.contains("analysis"), "native archive should not download the same report twice")
    }

    // MARK: Happy path

    func testHappyPathPreservesClientIDAndPersistsAnalysis() async throws {
        await backend.set(jobStates: ["PENDING", "PROGRESS", "SUCCESS"])

        let match = try ingest(court: .doubles)
        let serverID = match.id
        XCTAssertEqual(match.status, .uploading)
        XCTAssertFalse(match.localVideoPath!.hasPrefix("/"), "video path must be stored relative to Documents")
        await pipeline.waitUntilIdle()

        XCTAssertEqual(match.status, .done)
        XCTAssertEqual(match.id, serverID)
        XCTAssertNil(match.pipelineStep)
        XCTAssertNil(match.failureReason)
        XCTAssertEqual(match.cloudJobID, "job-1")
        XCTAssertEqual(match.rallies.count, 2)
        let calls = await backend.calls
        XCTAssertEqual(calls, ["create", "uploadURL", "upload", "analyze", "getMatch", "analysis"])
        let hints = await backend.analyzeDoublesHints
        XCTAssertEqual(hints, [true])

        let rally = try XCTUnwrap(match.rallies.first { $0.startFrame == 10 })
        XCTAssertEqual(rally.winner, .near)
        let shots = rally.shots.sorted { $0.frameIndex < $1.frameIndex }
        XCTAssertEqual(shots.map(\.shotType), [.serve, .smash])
        XCTAssertEqual(shots[0].normalizedX, 0.25, accuracy: 1e-6)
        XCTAssertEqual(shots[0].normalizedY, 0.5, accuracy: 1e-6)
        XCTAssertEqual(shots[0].speedKmh, 80.5, accuracy: 1e-4)
        XCTAssertEqual(shots[0].yMeters, -5, accuracy: 1e-6)
        let other = try XCTUnwrap(match.rallies.first { $0.startFrame == 500 })
        XCTAssertEqual(other.winner, .unknown)
        XCTAssertEqual(other.shots.first?.shotType, .clear)
        XCTAssertEqual(other.shots.first?.normalizedX ?? -1, 0.5, accuracy: 1e-6, "null coordinate falls back to centre")
    }

    func testPollingRunsUntilJobSucceedsWhenServerStillAnalyzing() async throws {
        await backend.set(getMatch: { MockBackend.remote($0, status: "analyzing") })
        await backend.set(jobStates: ["PENDING", "STARTED", "PROGRESS", "SUCCESS"])

        let match = try ingest()
        await pipeline.waitUntilIdle()

        XCTAssertEqual(match.status, .done)
        let calls = await backend.calls
        XCTAssertEqual(calls.filter { $0 == "poll" }.count, 4)
        XCTAssertEqual(calls.last, "analysis")
    }

    // MARK: Failures + retry

    func testUploadNetworkFailureThenRetryResumesAtUpload() async throws {
        await backend.set(upload: { throw APIError.networkError(URLError(.cannotConnectToHost)) })
        let match = try ingest()
        await pipeline.waitUntilIdle()

        XCTAssertEqual(match.status, .failed)
        XCTAssertEqual(match.pipelineStep, .upload)
        let serverID = await backend.serverID
        XCTAssertEqual(match.id, serverID, "server id adopted before the upload failed")
        XCTAssertTrue(match.failureReason?.contains("Wi-Fi") == true, match.failureReason ?? "nil")
        XCTAssertEqual(match.retryHint, "会重新上传这段视频")

        await backend.set(upload: {})
        pipeline.retry(match)
        XCTAssertNil(match.failureReason)
        XCTAssertEqual(match.status, .uploading)
        await pipeline.waitUntilIdle()

        XCTAssertEqual(match.status, .done)
        let calls = await backend.calls
        XCTAssertEqual(calls.filter { $0 == "create" }.count, 1, "retry must not create a second server match")
        XCTAssertEqual(calls.filter { $0 == "upload" }.count, 2)
    }

    func testUnauthorizedFailsWithLoginReasonAndPostsSessionExpired() async throws {
        await backend.set(create: { throw APIError.unauthorized })
        let expired = expectation(forNotification: .shuttlSessionExpired, object: nil)

        let match = try ingest()
        await pipeline.waitUntilIdle()
        await fulfillment(of: [expired], timeout: 1)

        XCTAssertEqual(match.status, .failed)
        XCTAssertEqual(match.pipelineStep, .create)
        XCTAssertEqual(match.failureReason, "登录已过期，请重新登录后点重试")
    }

    func testAnalysisFailureRetriesByRetriggeringAnalysisNotUpload() async throws {
        await backend.set(getMatch: { MockBackend.remote($0, status: "analyzing") })
        await backend.set(jobStates: ["STARTED", "FAILURE"])
        let match = try ingest()
        await pipeline.waitUntilIdle()

        XCTAssertEqual(match.status, .failed)
        XCTAssertEqual(match.pipelineStep, .analyze)
        XCTAssertTrue(match.failureReason?.contains("court not found") == true, match.failureReason ?? "nil")
        XCTAssertEqual(match.retryHint, "视频已在服务器上，会重新开始分析")

        await backend.set(jobStates: ["SUCCESS"])
        pipeline.retry(match)
        XCTAssertEqual(match.status, .analyzing)
        await pipeline.waitUntilIdle()

        XCTAssertEqual(match.status, .done)
        let calls = await backend.calls
        XCTAssertEqual(calls.filter { $0 == "upload" }.count, 1)
        XCTAssertEqual(calls.filter { $0 == "analyze" }.count, 2)
    }

    func testFailedCloudReanalysisGetsNewIdempotencyKeyOnlyAfterTerminalFailure() async throws {
        await backend.set(getMatch: { MockBackend.remote($0, status: "analyzing") })
        await backend.set(jobStates: ["STARTED", "FAILURE"])
        let match = try ingest()
        let oldKey = UUID()
        try MatchInputs(cloudReanalysisID: oldKey).save(matchID: match.id)
        defer { MatchInputs.delete(matchID: match.id) }
        await pipeline.waitUntilIdle()
        XCTAssertNotEqual(MatchInputs.load(matchID: match.id).cloudReanalysisID, oldKey)

        XCTAssertEqual(match.status, .failed)
        XCTAssertEqual(match.pipelineStep, .analyze)
        XCTAssertTrue(match.failureReason?.contains("court not found") == true, match.failureReason ?? "nil")
        XCTAssertEqual(match.retryHint, "视频已在服务器上，会重新开始分析")

        await backend.set(jobStates: ["SUCCESS"])
        pipeline.retry(match)
        XCTAssertEqual(match.status, .analyzing)
        await pipeline.waitUntilIdle()

        XCTAssertEqual(match.status, .done)
        let calls = await backend.calls
        XCTAssertEqual(calls.filter { $0 == "upload" }.count, 1)
        XCTAssertEqual(calls.filter { $0 == "analyze" }.count, 2)
    }

    func testServerReportsFailedStatusOnResume() async throws {
        await backend.set(getMatch: { MockBackend.remote($0, status: "failed", jobID: "job-1") })
        await backend.set(jobStates: ["FAILURE"])
        let match = try ingest()
        await pipeline.waitUntilIdle()

        XCTAssertEqual(match.status, .failed)
        XCTAssertEqual(match.pipelineStep, .analyze)
        XCTAssertTrue(match.failureReason?.hasPrefix("AI 分析失败") == true)
    }

    func testJobThatNeverLeavesPendingFailsAsStalled() async throws {
        await backend.set(getMatch: { MockBackend.remote($0, status: "analyzing") })
        await backend.set(jobStates: ["PENDING"])
        let match = try ingest()
        await pipeline.waitUntilIdle()

        XCTAssertEqual(match.status, .failed)
        XCTAssertEqual(match.pipelineStep, .analyze)
        XCTAssertTrue(match.failureReason?.contains("Celery") == true, match.failureReason ?? "nil")
    }

    func testMissingLocalVideoFailsWithoutNetwork() async throws {
        let match = try ingest()
        pipeline.cancel(match)
        try FileManager.default.removeItem(at: videoURL)
        match.status = .failed
        match.pipelineStep = .create
        pipeline.retry(match)
        await pipeline.waitUntilIdle()

        XCTAssertEqual(match.status, .failed)
        XCTAssertEqual(match.failureReason, "找不到这场比赛的本地视频文件")
        let calls = await backend.calls
        XCTAssertTrue(calls.isEmpty, "\(calls)")
    }

    func testEmptyRecordingIsRejected() throws {
        try Data().write(to: videoURL)
        XCTAssertThrowsError(try ingest()) { error in
            XCTAssertEqual(error as? PipelineError, .invalidRecording)
        }
        XCTAssertTrue(try context.fetch(FetchDescriptor<Match>()).isEmpty)
    }

    // MARK: Server-directed recovery

    func testServerLostMatchIsRecreatedOnce() async throws {
        let uploads = Counter()
        await backend.set(uploadURL: { _ in
            if uploads.next() == 1 { throw APIError.serverError(404, "比赛记录不存在") }
            return UploadURLResponse(uploadUrl: URL(string: "http://h/v")!, ossKey: "k")
        })
        let match = try ingest()
        await pipeline.waitUntilIdle()

        let calls = await backend.calls
        XCTAssertEqual(match.status, .done, match.failureReason ?? "")
        XCTAssertEqual(calls.filter { $0 == "create" }.count, 2)
        let createdID = await backend.serverID
        XCTAssertEqual(match.id, createdID)
    }

    func testRepeated404DoesNotLoop() async throws {
        await backend.set(uploadURL: { _ in throw APIError.serverError(404, "比赛记录不存在") })
        let match = try ingest()
        await pipeline.waitUntilIdle()

        XCTAssertEqual(match.status, .failed)
        let calls = await backend.calls
        XCTAssertEqual(calls.filter { $0 == "create" }.count, 2, "one recreate, then give up")
        XCTAssertTrue(match.failureReason?.contains("404") == true)
    }

    func testAnalyze400ReuploadsOnce() async throws {
        let triggers = Counter()
        await backend.set(trigger: { _ in
            if triggers.next() == 1 { throw APIError.serverError(400, "视频尚未上传") }
            return JobResponse(jobId: "job-2", status: "queued")
        })
        let match = try ingest()
        await pipeline.waitUntilIdle()

        XCTAssertEqual(match.status, .done, match.failureReason ?? "")
        XCTAssertEqual(match.cloudJobID, "job-2")
        let calls = await backend.calls
        XCTAssertEqual(calls.filter { $0 == "upload" }.count, 2)
        XCTAssertEqual(calls.filter { $0 == "create" }.count, 1)
        let forgotten = await backend.forgottenUploads
        XCTAssertEqual(forgotten, [match.id], "server lost the file: discard the local completion receipt before retrying")
    }

    // MARK: Resume on launch

    func testResumeContinuesAnalyzingMatchFromPoll() async throws {
        let id = UUID()
        let match = Match(id: id, status: .analyzing, localVideoPath: VideoStore.storedPath(for: videoURL),
                          cloudJobID: "job-9", pipelineStep: .poll)
        context.insert(match)
        try context.save()
        await backend.set(getMatch: { MockBackend.remote($0, status: "analyzing", jobID: "job-9") })

        pipeline.resumePending()
        await pipeline.waitUntilIdle()

        XCTAssertEqual(match.status, .done)
        let calls = await backend.calls
        XCTAssertEqual(calls, ["getMatch", "getMatch", "poll", "analysis"])
    }

    func testResumeLegacyAnalyzingMatchWithoutStepInfersPoll() async throws {
        let match = Match(status: .analyzing, localVideoPath: VideoStore.storedPath(for: videoURL), cloudJobID: "job-3")
        context.insert(match)
        try context.save()

        pipeline.resumePending()
        await pipeline.waitUntilIdle()

        XCTAssertEqual(match.status, .done)
        let calls = await backend.calls
        XCTAssertEqual(calls.first, "getMatch")
    }

    func testResumeUploadingMatchStartsFromItsStepAndSkipsDoneAndFailed() async throws {
        let uploading = Match(status: .uploading, localVideoPath: VideoStore.storedPath(for: videoURL), pipelineStep: .upload)
        let failed = Match(status: .failed, localVideoPath: VideoStore.storedPath(for: videoURL), pipelineStep: .upload)
        let done = Match(status: .done)
        for match in [uploading, failed, done] { context.insert(match) }
        try context.save()

        pipeline.resumePending()
        await pipeline.waitUntilIdle()

        XCTAssertEqual(uploading.status, .done)
        XCTAssertEqual(failed.status, .failed, "failed matches wait for an explicit 重试")
        let calls = await backend.calls
        XCTAssertEqual(calls.prefix(2), ["getMatch", "uploadURL"], "verify legacy ownership before uploading")
        XCTAssertFalse(calls.contains("create"))
    }

    func testServerUploadingStatusDuringPollRetriggersAnalysis() async throws {
        let match = Match(id: UUID(), status: .analyzing, localVideoPath: VideoStore.storedPath(for: videoURL),
                          cloudJobID: nil, pipelineStep: .poll)
        context.insert(match)
        try context.save()
        match.cloudAccount = await backend.remoteAccount()
        let statuses = Counter()
        await backend.set(getMatch: { id in
            MockBackend.remote(id, status: statuses.next() == 1 ? "uploading" : "done")
        })

        pipeline.resumePending()
        await pipeline.waitUntilIdle()

        XCTAssertEqual(match.status, .done)
        let calls = await backend.calls
        XCTAssertEqual(calls, ["getMatch", "analyze", "getMatch", "analysis"])
    }

    func testResumeDisabledDoesNothing() async throws {
        let offline = MatchPipeline(context: context, backend: backend, resumesAutomatically: false)
        let match = Match(status: .uploading, localVideoPath: VideoStore.storedPath(for: videoURL), pipelineStep: .create)
        context.insert(match)
        try context.save()

        offline.resumePending()
        await offline.waitUntilIdle()

        XCTAssertEqual(match.status, .uploading)
        let calls = await backend.calls
        XCTAssertTrue(calls.isEmpty)
    }

    // MARK: Legacy trajectory fallback

    func testFallsBackToTrajectoryWhenAnalysisDocumentMissing() async throws {
        await backend.set(analysis: { _ in throw APIError.serverError(404, "该比赛没有 v2 分析数据") })
        await backend.set(trajectory: { id in
            TrajectoryResponse(matchId: id, fps: 60, points: [
                TrajectoryPointDTO(frame: 10, xPx: 0.2, yPx: 0.4, xMeters: 0, yMeters: 0, speedKmh: 100, shotType: "clear"),
                TrajectoryPointDTO(frame: 70, xPx: 0.6, yPx: 0.5, xMeters: 0, yMeters: 0, speedKmh: 200, shotType: "smash"),
                TrajectoryPointDTO(frame: 2000, xPx: 0.5, yPx: 0.5, xMeters: 0, yMeters: 0, speedKmh: 50, shotType: "serve"),
            ])
        })
        let match = try ingest()
        await pipeline.waitUntilIdle()

        XCTAssertEqual(match.status, .done)
        XCTAssertEqual(match.rallies.count, 2, "hits a second apart stay in one rally")
        XCTAssertEqual(match.allShots.count, 3)
        let calls = await backend.calls
        XCTAssertEqual(Array(calls.suffix(2)), ["analysis", "trajectory"])
    }

    func testServerAnalysisWithoutRalliesFailsInsteadOfEmptyDone() async throws {
        await backend.set(analysis: { _ in
            let json = #"{"schema_version": 2, "video": {"fps": 60.0, "width": 1920, "height": 1080, "frames": 900}, "rallies": []}"#
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            return try decoder.decode(AnalysisDocumentDTO.self, from: Data(json.utf8))
        })
        let match = try ingest()
        await pipeline.waitUntilIdle()

        XCTAssertEqual(match.status, .failed)
        XCTAssertTrue(match.rallies.isEmpty)
        XCTAssertEqual(match.failureReason, analysisOutcomeMessage(.noRallies))
        XCTAssertEqual(match.pipelineStep, .analyze, "重试 re-triggers the server analysis")
        XCTAssertEqual(match.failureRetryable, false)
    }

    func testEmptyLegacyTrajectoryFailsInsteadOfEmptyDone() async throws {
        await backend.set(analysis: { _ in throw APIError.serverError(404, "该比赛没有 v2 分析数据") })
        let match = try ingest()
        await pipeline.waitUntilIdle()

        XCTAssertEqual(match.status, .failed)
        XCTAssertEqual(match.failureReason, analysisOutcomeMessage(.noRallies))
    }

    // MARK: Policy helpers

    func testNormalisationAcceptsFractionsAndPixels() {
        XCTAssertEqual(JobPollingService.normalized(0.3, extent: 1920), 0.3, accuracy: 1e-6)
        XCTAssertEqual(JobPollingService.normalized(960, extent: 1920), 0.5, accuracy: 1e-6)
        XCTAssertEqual(JobPollingService.normalized(-0.2, extent: nil), 0)
        XCTAssertEqual(JobPollingService.normalized(1.2, extent: nil), 1)
    }

    func testProgressPercentParsing() {
        XCTAssertEqual(JobPollingService.progressPercent(from: "{'match_id': 'a', 'percent': 42, 'stage': 'detect'}"), 42)
        XCTAssertEqual(JobPollingService.progressPercent(from: #"{"percent": 140}"#), 100)
        XCTAssertNil(JobPollingService.progressPercent(from: "None"))
        XCTAssertNil(JobPollingService.progressPercent(from: nil))
    }

    func testResumeStepPolicy() {
        XCTAssertEqual(MatchPipeline.resumeStep(after: PipelineError.analysisFailed(nil), at: .poll), .analyze)
        XCTAssertEqual(MatchPipeline.resumeStep(after: PipelineError.analysisStalled, at: .poll), .analyze)
        XCTAssertEqual(MatchPipeline.resumeStep(after: APIError.timeout, at: .poll), .poll)
        XCTAssertEqual(MatchPipeline.resumeStep(after: APIError.unauthorized, at: .upload), .upload)
    }

    func testFailureReasonsAreReadable() {
        XCTAssertEqual(MatchPipeline.failureReason(for: APIError.serverError(401, "x")), "登录已过期，请重新登录后点重试")
        XCTAssertTrue(MatchPipeline.failureReason(for: APIError.networkError(URLError(.cannotConnectToHost))).contains("同一 Wi-Fi"))
        XCTAssertTrue(MatchPipeline.failureReason(for: APIError.uploadFailed("视频文件过大，服务器拒收")).contains("过大"))
        XCTAssertTrue(MatchPipeline.failureReason(for: PipelineError.analysisFailed(nil)).contains("光线"))
        // Celery reports the worker's AnalysisUnusable as its repr.
        let noRallies = analysisOutcomeMessage(.noRallies)
        XCTAssertEqual(MatchPipeline.failureReason(for: PipelineError.analysisFailed("AnalysisUnusable('\(noRallies)')")), noRallies)
    }

    func testOnlyContentVerdictsAreNotWorthRetrying() {
        for outcome in [AnalysisOutcome.noShuttle, .noRallies, .cameraUnsteady] {
            XCTAssertFalse(MatchPipeline.isWorthRetrying(PipelineError.analysisUnusable(analysisOutcomeMessage(outcome))), "\(outcome)")
        }
        let serverNoShuttle = "AnalysisUnusable('\(analysisOutcomeMessage(.noShuttle))')"
        XCTAssertFalse(MatchPipeline.isWorthRetrying(PipelineError.analysisFailed(serverNoShuttle)))
        XCTAssertFalse(MatchPipeline.isWorthRetrying(PipelineError.invalidRecording))
        XCTAssertFalse(MatchPipeline.isWorthRetrying(PipelineError.videoMissing))
        // A failed stage asks for 重试 in its own sentence: keep offering it.
        let stage = analysisOutcomeMessage(.stageFailed(stage: "players", message: "", retryable: false))
        XCTAssertTrue(MatchPipeline.isWorthRetrying(PipelineError.analysisUnusable(stage)))
        XCTAssertTrue(MatchPipeline.isWorthRetrying(PipelineError.analysisFailed("ValueError('court not found')")))
        XCTAssertTrue(MatchPipeline.isWorthRetrying(PipelineError.analysisFailed(nil)))
        XCTAssertTrue(MatchPipeline.isWorthRetrying(PipelineError.analysisStalled))
        XCTAssertTrue(MatchPipeline.isWorthRetrying(APIError.networkError(URLError(.networkConnectionLost))))
        XCTAssertTrue(MatchPipeline.isWorthRetrying(APIError.unauthorized))
    }

    func testPollerRetriesTransientErrorsThenGivesUp() async throws {
        await backend.set(pollErrors: [APIError.timeout, APIError.serverError(502, "bad gateway")])
        await backend.set(jobStates: ["SUCCESS"])
        let poller = JobPollingService(backend: backend, options: Self.fastPolling, sleep: { _ in })
        let outcome = try await poller.waitForCompletion(matchID: UUID(), jobID: "j")
        XCTAssertEqual(outcome, .succeeded)

        await backend.set(pollErrors: Array(repeating: APIError.timeout, count: 10))
        do {
            _ = try await poller.waitForCompletion(matchID: UUID(), jobID: "j")
            XCTFail("expected to give up")
        } catch {
            XCTAssertEqual(MatchPipeline.failureReason(for: error), "连接分析服务器超时，请检查手机和电脑是否在同一 Wi-Fi")
        }

        await backend.set(pollErrors: [APIError.unauthorized])
        do {
            _ = try await poller.waitForCompletion(matchID: UUID(), jobID: "j")
            XCTFail("401 is terminal")
        } catch {
            XCTAssertTrue(MatchPipeline.isAuthFailure(error))
        }
    }

    // MARK: - R4: worthRetrying with stopped_after
    // These test the static function MatchPipeline.worthRetrying(_:) which is only exercised
    // in the on-device analysis path. The function receives a ShuttlAnalysis.MatchAnalysis.

    func testWorthRetryingStoppedAfterMatch() {
        // stopped_after matches the failed stage → always retry (P2's incremental resume)
        let timestamps = Array(stride(from: 0.0, to: 10.0, by: 1.0/60.0))
        let meta = VideoMeta(fps: 60, width: 1920, height: 1080, nFrames: 600, timestamps: timestamps)
        let track = Track2D(
            t: timestamps,
            xy: Array(repeating: .init(x: 100, y: 100), count: 600),
            conf: Array(repeating: 0.9, count: 600),
            visible: Array(repeating: true, count: 600),
            interpolated: Array(repeating: false, count: 600)
        )
        let analysis = MatchAnalysis(
            meta: meta,
            track: track,
            diagnostics: [
                "stopped_after": .string("players"),
                "stage_failures": .array([
                    .object(["stage": .string("players"), "message": .string("test"), "retryable": .bool(true)])
                ])
            ]
        )
        XCTAssertTrue(MatchPipeline.worthRetrying(analysis), "stopped_after=players + stageFailed(players) should always retry")
    }

    func testWorthRetryingStoppedAfterMismatch() {
        // stopped_after doesn't match the failed stage → fall through to rallies.isEmpty guard
        // Failed at "calibrate" (not a shuttle stage), rallies empty, track exists → not worth retrying
        let timestamps = Array(stride(from: 0.0, to: 10.0, by: 1.0/60.0))
        let meta = VideoMeta(fps: 60, width: 1920, height: 1080, nFrames: 600, timestamps: timestamps)
        let track = Track2D(
            t: timestamps,
            xy: Array(repeating: .init(x: 100, y: 100), count: 600),
            conf: Array(repeating: 0.9, count: 600),
            visible: Array(repeating: true, count: 600),
            interpolated: Array(repeating: false, count: 600)
        )
        let analysis = MatchAnalysis(
            meta: meta,
            track: track,
            rallies: [],
            diagnostics: [
                "stopped_after": .string("players"),
                "stage_failures": .array([
                    .object(["stage": .string("calibrate"), "message": .string("test"), "retryable": .bool(true)])
                ])
            ]
        )
        XCTAssertFalse(MatchPipeline.worthRetrying(analysis), "stopped_after=players but failed at calibrate, rallies empty + track exists → shuttle ran fine, not worth retrying")
    }

    func testWorthRetryingNoStoppedAfterEmptyRallies() {
        // No stopped_after, empty rallies, track exists, shuttle stages ran fine → not worth retrying
        let timestamps = Array(stride(from: 0.0, to: 10.0, by: 1.0/60.0))
        let meta = VideoMeta(fps: 60, width: 1920, height: 1080, nFrames: 600, timestamps: timestamps)
        let track = Track2D(
            t: timestamps,
            xy: Array(repeating: .init(x: 100, y: 100), count: 600),
            conf: Array(repeating: 0.9, count: 600),
            visible: Array(repeating: true, count: 600),
            interpolated: Array(repeating: false, count: 600)
        )
        let analysis = MatchAnalysis(
            meta: meta,
            track: track,
            rallies: [],
            diagnostics: [
                "stage_failures": .array([
                    .object(["stage": .string("players"), "message": .string("test"), "retryable": .bool(true)])
                ])
            ]
        )
        XCTAssertFalse(MatchPipeline.worthRetrying(analysis), "empty rallies + track exists → shuttle ran fine, not worth retrying")
    }

    func testWorthRetryingShuttleStageFailure() {
        // Shuttle stage (detect/link/open) failed → always worth retrying
        let timestamps = Array(stride(from: 0.0, to: 10.0, by: 1.0/60.0))
        let meta = VideoMeta(fps: 60, width: 1920, height: 1080, nFrames: 600, timestamps: timestamps)
        let analysis = MatchAnalysis(
            meta: meta,
            track: nil,
            rallies: [],
            diagnostics: [
                "stage_failures": .array([
                    .object(["stage": .string("detect"), "message": .string("test"), "retryable": .bool(true)])
                ])
            ]
        )
        XCTAssertTrue(MatchPipeline.worthRetrying(analysis), "detect stage failed → worth retrying")
    }
}

/// Thread-safe call counter for handlers invoked inside the mock actor.
final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0
    func next() -> Int { lock.lock(); defer { lock.unlock() }; value += 1; return value }
}
