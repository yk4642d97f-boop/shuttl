import AVFoundation
import Foundation
import Network
import ShuttlAnalysis
import ShuttlCore
import ShuttlPresent
import ShuttlRuntime
import SwiftData
import UIKit

extension Notification.Name {
    /// Posted when a pipeline call was rejected as unauthenticated (refresh failed too).
    static let shuttlSessionExpired = Notification.Name("ShuttlSessionExpired")
}

enum PipelineError: Error, Equatable, Sendable {
    case videoMissing
    case invalidRecording
    case analysisFailed(String?)
    /// The phone's analysis ran but found nothing usable; the payload is already the user sentence.
    case analysisUnusable(String)
    case analysisStalled
    case matchDeleted
    case originalAccountRequired
    case pendingRemoteCommand
}

/// Live (non-persisted) progress of a running pipeline task.
struct PipelineActivity: Equatable, Sendable {
    enum Phase: Equatable, Sendable { case creating, uploading, analyzing, fetching, analyzingOnDevice }

    var phase: Phase
    /// 0–1 while uploading, nil until the first progress callback.
    var uploadFraction: Double?
    /// Celery-reported percent while analysing.
    var analysisPercent: Int?
    /// The job is queued (PENDING) and no worker has picked it up yet.
    var isQueued = false
    /// 0–1 overall while analysing on this phone.
    var localFraction: Double?
    /// `AnalysisPipeline` stage name while analysing on this phone.
    var localStage: String?
}

/// App-lifetime coordinator: finished recording → analysis → results in SwiftData.
/// With an on-device analyzer the video is analysed on this phone (`.local` step);
/// otherwise it goes to the server (create → upload → analyze → poll).
///
/// Everything touching SwiftData runs on the main actor against the container's
/// main context (the same one the views' @Query reads). Network calls go through
/// a Sendable `MatchBackend`. Progress is persisted in `Match.status` +
/// `Match.pipelineStep`, so a killed app resumes from the right step and a failed
/// match retries from where it stopped.
@MainActor
@Observable
final class MatchPipeline {
    /// Durable operations awaiting a server acknowledgement.
    private(set) var pendingRemoteCount = 0
    private(set) var remoteSyncMessage: String?
    @ObservationIgnored private let remoteCommands: RemoteCommandQueue
    @ObservationIgnored private var networkMonitor: NWPathMonitor?
    @ObservationIgnored private var discarding: Set<PersistentIdentifier> = []
    /// Progress for matches with a running task, keyed by model id.
    private(set) var activities: [PersistentIdentifier: PipelineActivity] = [:]

    @ObservationIgnored let context: ModelContext
    @ObservationIgnored private let backend: any MatchBackend
    @ObservationIgnored private let analyzer: (any OnDeviceAnalyzer)?
    @ObservationIgnored private let poller: JobPollingService
    @ObservationIgnored private let resumesAutomatically: Bool
    @ObservationIgnored private var running: [PersistentIdentifier: Running] = [:]
    /// Whether the app is in front (CoreML / Vision work is only reliable then). Tests replace it.
    @ObservationIgnored var isAppActive: @MainActor () -> Bool = {
        UIApplication.shared.applicationState == .active
    }
    /// Waits before retry number `attempt + 1` of an interrupted analysis. Tests replace it.
    @ObservationIgnored var retryDelay: @MainActor (Int) async throws -> Void = { attempt in
        try await Task.sleep(for: .seconds(min(30, 1 << attempt)))   // 1, 2, 4, 8, 16 s
    }
    /// Interrupted analyses are retried up to this many runs in a row while the app is in front.
    static let maxLocalAttempts = 6
    /// A take is being recorded: analysis steps aside so encoding 1080p60 never drops frames.
    private(set) var isRecordingPaused = false
    /// What the phone's temperature allows (run / hold new starts / pause). Tests replace it.
    @ObservationIgnored var thermalGate: @MainActor () -> ThermalDecision = { .run }
    /// The current decision, for the UI's heat chip (tomato when .pause, butter when .hold).
    private(set) var thermalDecision: ThermalDecision = .run
    /// A line for the UI when heat or Low Power Mode affects analysis (nil = nothing to say).
    private(set) var thermalMessage: String?
    @ObservationIgnored private var localAnalysisActive = false
    @ObservationIgnored private var localAnalysisID: PersistentIdentifier?
    @ObservationIgnored private var thermalObserver: NSObjectProtocol?
    @ObservationIgnored private var thermalStateChanged: (@MainActor () -> Void)?
    /// The record screen's camera session is running (preview or take). Analysis decode waits
    /// for it: both use the hardware codec, and the phone stalled / failed with -11847 that way.
    private(set) var isCameraLive = false
    /// Where admission waits are logged (stage-metrics.jsonl on the phone). Tests replace it.
    @ObservationIgnored var admissionLog = AdmissionLog.default
    /// Thermal level for the log (the decision alone can't tell serious from critical). Tests replace it.
    @ObservationIgnored var thermalLevel: @MainActor () -> ThermalLevel = { ThermalLevel(ProcessInfo.processInfo.thermalState) }
    /// Analyses stopped mid-run (camera, take, heat), until they are admitted again.
    @ObservationIgnored private var paused: [PersistentIdentifier: Pause] = [:]
    /// The newest unthrottled progress of the running analysis, for the pause event's stage and frame.
    @ObservationIgnored private var lastProgress: [PersistentIdentifier: AnalysisProgress] = [:]
    @ObservationIgnored private var frameCounts: [PersistentIdentifier: Int] = [:]
    /// Frames in a match video (nil when it has no readable video track). Tests replace it.
    @ObservationIgnored var frameCount: @Sendable (URL) async -> Int? = { await MatchPipeline.videoFrameCount($0) }

    private struct Pause {
        let reason: AdmissionBlock
        let at: ContinuousClock.Instant
    }

    private struct Running {
        let token: UUID
        let task: Task<Void, Never>
    }

    init(
        context: ModelContext,
        backend: any MatchBackend,
        analyzer: (any OnDeviceAnalyzer)? = nil,
        pollOptions: JobPollingService.Options = .standard,
        resumesAutomatically: Bool = true,
        remoteCommands: RemoteCommandQueue = .shared
    ) {
        self.remoteCommands = remoteCommands
        self.context = context
        self.backend = backend
        self.analyzer = analyzer
        self.poller = JobPollingService(backend: backend, options: pollOptions)
        self.resumesAutomatically = resumesAutomatically
    }

    /// The app's pipeline. Demo mode (and unit-test hosting) never touches the network.
    static func live(container: ModelContainer) -> MatchPipeline {
        #if DEBUG
        if DemoMode.isOn {
            // -UIDemoOnDevice: behave like Release (analysis on this phone, so no sign-in screen). The analyzer
            // is never called: demo pipelines don't resume, and demo matches are seeded, not ingested.
            let onDevice = ProcessInfo.processInfo.arguments.contains("-UIDemoOnDevice")
            return MatchPipeline(context: container.mainContext, backend: OfflineMatchBackend(),
                                 analyzer: onDevice ? DemoAnalyzer() : nil, resumesAutomatically: false)
        }
        #endif
        let underTests = ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
        let pipeline = MatchPipeline(context: container.mainContext, backend: LiveMatchBackend(),
                                     analyzer: OnDeviceAnalyzers.makeDefault(), resumesAutomatically: !underTests)
        if !underTests {
            pipeline.startRemoteSyncMonitoring()
            pipeline.startThermalMonitoring()
            #if os(iOS)
            metricKit.start()
            #endif
        }
        return pipeline
    }

    #if os(iOS)
    /// Crash / hang / memory reports from iOS, kept under Documents/Diagnostics/MetricKit.
    private static let metricKit = MetricKitSink()
    #endif

    /// True when matches are analysed on this phone rather than uploaded.
    var analyzesOnDevice: Bool { analyzer != nil }

    // MARK: - Public API

    func activity(for match: Match) -> PipelineActivity? { activities[match.persistentModelID] }

    func isRunning(_ match: Match) -> Bool { running[match.persistentModelID] != nil }

    /// Called once the movie file is completely written. Inserts the match and starts processing.
    /// - Parameter cameraHint: the court the record screen locked onto; the analysis falls back
    ///   to it when it can't calibrate from the file itself.
    @discardableResult
    func ingestRecording(fileURL: URL, durationSeconds: Double, courtType: CourtType,
                         cameraHint: CameraModel? = nil, focalHintPx: Double? = nil) throws -> Match {
        let size = (try? FileManager.default.attributesOfItem(atPath: fileURL.path)[.size] as? Int) ?? 0
        guard size > 0 else { throw PipelineError.invalidRecording }
        let match = Match(
            durationSeconds: durationSeconds.isFinite ? max(0, durationSeconds) : 0,
            status: analyzesOnDevice ? .analyzing : .uploading,
            localVideoPath: VideoStore.storedPath(for: fileURL),
            courtType: courtType,
            pipelineStep: analyzesOnDevice ? .local : .create
        )
        if cameraHint != nil || focalHintPx != nil {
            try? MatchInputs(cameraHint: cameraHint, focalHintPx: focalHintPx).save(matchID: match.id)
        }
        context.insert(match)
        persist()
        start(match)
        return match
    }

    /// Runs (or resumes) the pipeline for a match. No-op if it is already running or done.
    func start(_ match: Match) {
        guard match.status != .done, !discarding.contains(match.persistentModelID) else { return }
        if match.pipelineStep == nil { match.pipelineStep = inferredStep(for: match) }
        persist()  // a permanent persistentModelID before we key anything on it
        let id = match.persistentModelID
        guard running[id] == nil else { return }
        let token = UUID()
        let task = Task { [weak self] () -> Void in
            guard let self else { return }
            await self.run(id, token: token)
        }
        running[id] = Running(token: token, task: task)
    }

    /// "重试": resumes a failed match from the step it stopped at.
    func retry(_ match: Match) {
        guard match.status == .failed, running[match.persistentModelID] == nil else { return }
        let step = match.pipelineStep ?? inferredStep(for: match)
        match.pipelineStep = step
        match.failureReason = nil
        match.failureRetryable = nil
        match.status = step >= .analyze ? .analyzing : .uploading
        start(match)
    }

    /// Stops work for a match (e.g. before it is deleted).
    func cancel(_ match: Match) {
        let id = match.persistentModelID
        running[id]?.task.cancel()
        running[id] = nil
        activities[id] = nil
        forgetLocalProgress(id)
    }

    /// Signing out detaches the cloud pipeline; pending steps can resume after sign-in.
    func suspendCloudWork() {
        guard !analyzesOnDevice else { return }
        for run in running.values { run.task.cancel() }
        running = [:]
        activities = [:]
    }

    /// Save the remote intent before deleting any local files or rows.
    func discard(_ match: Match) async throws {
        let id = match.persistentModelID
        guard !discarding.contains(id) else { throw PipelineError.pendingRemoteCommand }
        discarding.insert(id)
        defer { discarding.remove(id) }
        cancel(match)
        if hasCloudCopy(match) {
            let account: RemoteAccount
            if let stored = match.cloudAccount { account = stored }
            else {
                // Legacy rows need a successful authenticated ownership lookup first.
                _ = try await scopedBackend(for: match, allowNew: false)
                guard let verified = match.cloudAccount else { throw PipelineError.originalAccountRequired }
                account = verified
            }
            try await remoteCommands.enqueue(account: account, matchID: match.id, kind: .delete)
            // Upload cancellation is idempotent. A failed receipt cleanup must keep the row visible.
            try await backend.scoped(to: account).forgetUpload(matchID: match.id)
        }
        AnalysisStore.delete(match.analysisPath)
        AnalysisStore.archive().deleteAll(matchID: match.id, video: match.localVideoURL)
        MatchInputs.delete(matchID: match.id)
        Task { await syncRemoteCommands() }
    }

    func hasCloudCopy(_ match: Match) -> Bool {
        guard backend.usesRemoteStorage else { return false }
        if match.cloudAccount != nil || match.cloudJobID != nil { return true }
        if match.pipelineStep == .create && match.cloudCreateRequested == false { return false }
        return !match.isOnDevice && !analyzesOnDevice
    }

    func canCancelRemotely(_ match: Match) -> Bool {
        match.status == .analyzing && match.pipelineStep == .poll && match.cloudJobID != nil && hasCloudCopy(match)
    }

    func cancelRemotely(_ match: Match) async throws {
        guard canCancelRemotely(match), let jobID = match.cloudJobID else { return }
        _ = try await scopedBackend(for: match, allowNew: false)
        guard let account = match.cloudAccount else { throw PipelineError.originalAccountRequired }
        try await remoteCommands.enqueue(account: account, matchID: match.id, kind: .cancel, jobID: jobID)
        cancel(match)
        match.status = .failed
        match.failureReason = "已请求取消云端分析；联网后同步，可稍后重试"
        match.failureRetryable = true
        match.pipelineStep = .analyze
        try context.save()
        await syncRemoteCommands()
    }

    func syncRemoteCommands() async {
        await remoteCommands.flush(using: backend)
        pendingRemoteCount = await remoteCommands.count()
        remoteSyncMessage = await remoteCommands.storageErrorMessage()
        if pendingRemoteCount > 0 && remoteSyncMessage == nil {
            remoteSyncMessage = "还有 \(pendingRemoteCount) 项云端删除或取消待同步，请联网并登录原账号"
        }
    }

    private func startRemoteSyncMonitoring() {
        guard networkMonitor == nil else { return }
        let monitor = NWPathMonitor()
        monitor.pathUpdateHandler = { [weak self] path in
            guard path.status == .satisfied else { return }
            Task { @MainActor [weak self] in await self?.syncRemoteCommands() }
        }
        monitor.start(queue: DispatchQueue(label: "com.shuttl.remote-actions"))
        networkMonitor = monitor
    }

    /// Bind new uploads before their first RPC; verify legacy rows instead of silently moving them.
    private func scopedBackend(for match: Match, allowNew: Bool) async throws -> any MatchBackend {
        guard let account = await backend.remoteAccount() else { throw PipelineError.originalAccountRequired }
        try Task.checkCancellation()
        if let original = match.cloudAccount {
            guard original == account else { throw PipelineError.originalAccountRequired }
        } else {
            if !allowNew {
                do { _ = try await backend.scoped(to: account).getMatch(id: match.id) }
                catch APIError.serverError(404, _) { throw PipelineError.originalAccountRequired }
                try Task.checkCancellation()
                guard await backend.remoteAccount() == account else { throw PipelineError.originalAccountRequired }
            }
            match.cloudAccount = account
        }
        if allowNew { match.cloudCreateRequested = true }
        // A previous save may have failed while the values remain in memory: retry it before any RPC.
        try context.save()
        return backend.scoped(to: account)
    }

    private func requireNoPendingCommand(_ match: Match) async throws {
        guard let account = match.cloudAccount else { return }
        await remoteCommands.flush(using: backend)
        let pending = try await remoteCommands.pending(account: account, matchID: match.id)
        guard pending.isEmpty else { throw PipelineError.pendingRemoteCommand }
        try Task.checkCancellation()
    }

    // MARK: - User fixes

    /// The user's call for one rally (from the rally list): who won it, or "not a rally";
    /// nil restores the AI's call. The rows, the score and the saved result follow.
    func correctRally(_ match: Match, startFrame: Int, fix: ScoreCorrections.Fix?) async throws {
        guard let stored = match.analysisPath, let url = VideoStore.resolve(stored) else { return }
        let corrected = try await Task.detached(priority: .userInitiated) { () -> MatchAnalysis? in
            guard let original = try? AnalysisArchive.load(analysisAt: url) else { return nil }
            var corrections = ScoreCorrections.load(forAnalysisAt: url)
            corrections.fixes[startFrame] = fix
            try corrections.save(forAnalysisAt: url)
            return corrections.applied(to: original)
        }.value
        guard let corrected else { return }
        MatchAnalysisMapper.apply(corrected, to: match)
        persist()
    }

    /// Drops every fix the user made to this match's rallies.
    func clearCorrections(_ match: Match) async {
        guard let stored = match.analysisPath, let url = VideoStore.resolve(stored) else { return }
        let original = await Task.detached(priority: .userInitiated) { () -> MatchAnalysis? in
            try? FileManager.default.removeItem(at: ScoreCorrections.url(forAnalysisAt: url))
            return try? AnalysisArchive.load(analysisAt: url)
        }.value
        guard let original else { return }
        MatchAnalysisMapper.apply(original, to: match)
        persist()
    }

    /// Analyses the match again with court corners the user placed
    /// (normalised 0…1 of the upright frame, CourtModel names such as "nb_ld").
    /// The previous result stays on screen until the new one is saved; old fixes are dropped
    /// because the rallies may come out differently.
    func reanalyze(_ match: Match, corners: [String: CGPoint]) {
        cancel(match)
        var inputs = MatchInputs.load(matchID: match.id)
        inputs.manualCorners = CourtCorners.manualCorners(corners)
        if !analyzesOnDevice { inputs.cloudReanalysisID = UUID() }
        try? inputs.save(matchID: match.id)
        AnalysisStore.archive().removeCheckpoints(matchID: match.id)
        if let stored = match.analysisPath, let url = VideoStore.resolve(stored) {
            try? FileManager.default.removeItem(at: ScoreCorrections.url(forAnalysisAt: url))
        }
        match.status = .analyzing
        match.pipelineStep = analyzesOnDevice ? .local : (match.cloudAccount == nil ? .create : .analyze)
        match.failureReason = nil
        persist()
        start(match)
    }

    /// An upright still from the match video for the corner-tap screen (about a quarter in,
    /// when players are usually in position), plus its size in pixels.
    nonisolated static func calibrationFrame(for videoURL: URL) async -> (CGImage, CGSize)? {
        let asset = AVURLAsset(url: videoURL)
        guard let duration = try? await asset.load(.duration), duration.seconds.isFinite else { return nil }
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = CMTime(seconds: 0.5, preferredTimescale: 600)
        let at = CMTime(seconds: max(0, duration.seconds * 0.25), preferredTimescale: 600)
        guard let (image, _) = try? await generator.image(at: at) else { return nil }
        return (image, CGSize(width: image.width, height: image.height))
    }

    /// Picks up matches left uploading/analysing (app killed, suspended mid-upload, re-login).
    func resumePending() {
        guard resumesAutomatically else { return }
        Task { await syncRemoteCommands() }
        let matches = (try? context.fetch(FetchDescriptor<Match>())) ?? []
        for match in matches where match.status == .uploading || match.status == .analyzing {
            start(match)
        }
        if !DebugInbox.pendingVideos().isEmpty {
            Task { await DebugInbox.drain(into: self) }
            #if DEBUG
            DebugInbox.startPolling(pipeline: self)
            #endif
        }
    }

    /// Test helper: waits until no pipeline task is running.
    func waitUntilIdle() async {
        while let next = running.values.first {
            await next.task.value
        }
    }

    // MARK: - Run loop

    private enum StepResult {
        case next(PipelineStep)
        case finished
    }

    private enum Heal: Hashable {
        case recreate   // server no longer knows the match
        case reupload   // server has the match but not the video
    }

    private func run(_ id: PersistentIdentifier, token: UUID) async {
        defer {
            if running[id]?.token == token {
                running[id] = nil
                activities[id] = nil
                forgetLocalProgress(id)
            }
        }
        var healed: Set<Heal> = []
        // Four steps plus the odd server-directed detour; bounded so nothing can spin.
        for _ in 0..<12 {
            guard !Task.isCancelled, let match = fetch(id) else { return }
            let step = match.pipelineStep ?? inferredStep(for: match)
            do {
                switch try await perform(step, id) {
                case .finished:
                    return
                case .next(let nextStep):
                    try current(id).pipelineStep = nextStep
                    persist()
                }
            } catch is CancellationError {
                return
            } catch PipelineError.matchDeleted {
                return
            } catch {
                if Task.isCancelled { return }
                if let heal = Self.heal(for: error, at: step), !healed.contains(heal), let match = fetch(id) {
                    healed.insert(heal)
                    if heal == .reupload {
                        do { try await backend.forgetUpload(matchID: match.id) }
                        catch { fail(id, at: step, error: error); return }
                    }
                    match.pipelineStep = heal == .recreate ? .create : .upload
                    persist()
                    continue
                }
                fail(id, at: step, error: error)
                return
            }
        }
    }

    private func perform(_ step: PipelineStep, _ id: PersistentIdentifier) async throws -> StepResult {
        if step != .local {
            let match = try current(id)
            _ = try await scopedBackend(for: match, allowNew: step == .create)
            try await requireNoPendingCommand(match)
        }
        switch step {
        case .create:  try await createRemote(id); return .next(.upload)
        case .upload:  try await upload(id); return .next(.analyze)
        case .analyze: try await analyze(id); return .next(.poll)
        case .poll:    return try await poll(id)
        case .local:
            // A build without the on-device analyzer falls back to the server.
            guard let analyzer else { return .next(.create) }
            try await analyzeLocally(id, analyzer: analyzer)
            return .finished
        }
    }

    // MARK: Steps

    /// The whole pipeline on this phone: video → MatchAnalysis → SwiftData rows + JSON file.
    private func analyzeLocally(_ id: PersistentIdentifier, analyzer: any OnDeviceAnalyzer) async throws {
        let match = try current(id)
        guard let video = match.localVideoURL, FileManager.default.fileExists(atPath: video.path) else {
            throw PipelineError.videoMissing
        }
        let matchID = match.id
        let options = MatchInputs.load(matchID: matchID).options(doubles: match.courtType == .doubles)
        match.status = .analyzing
        persist()

        // One analysis at a time (one detector, one GPU), and none while the phone is hot.
        try await admitLocalAnalysis(id)
        defer { localAnalysisActive = false }
        activities[id] = PipelineActivity(phase: .analyzingOnDevice, localFraction: 0, localStage: "detect")
        if frameCounts[id] == nil {
            let count = frameCount
            Task { [weak self] in
                guard let n = await count(video) else { return }
                self?.frameCounts[id] = n
            }
        }

        let checkpoints = AnalysisStore.checkpoints(matchID: matchID, video: video, options: options)
        let progress: @Sendable (AnalysisProgress) -> Void = { [weak self] p in
            Task { @MainActor in self?.reportLocal(p, id) }
        }
        IdleTimerGuard.acquire("analysis")
        defer { IdleTimerGuard.release("analysis") }
        var analysis: MatchAnalysis?
        // The pipeline degrades instead of throwing, so a returned analysis can still be an
        // interrupted one: CoreML / Vision / the hardware decoder refused work, which on a phone
        // happens even in the foreground (a camera session starting, a media-services reset).
        // Retry with backoff while the app is in front; checkpoints (and the resumable detect)
        // make each retry re-run only what failed.
        for attempt in 0..<Self.maxLocalAttempts {
            let result: MatchAnalysis
            do {
                // On the last attempt, pass stopOnRetryableFailure: false so AnalysisPipeline
                // continues to rallies even when detect/players fail retryable. That way ALL
                // attempts failing with stopped_after=players still produces a degraded result
                // (rallies without player tracking) instead of throwing analysisUnusable.
                let isLastAttempt = (attempt + 1 == Self.maxLocalAttempts)
                result = try await Self.withBackgroundTime(named: "shuttl.analysis") {
                    try await analyzer.analyze(video: video, options: options, checkpoints: checkpoints,
                                               progress: progress, stopOnRetryableFailure: !isLastAttempt)
                }
            } catch is CancellationError {
                throw CancellationError()
            } catch let error as PipelineError {
                throw error
            } catch {
                guard AnalysisErrorClassifier.isRetryable(error) else {
                    throw PipelineError.analysisFailed(String(describing: error))
                }
                // Interrupted while away: stay pending; resumePending picks it up on return.
                if !isAppActive() { throw CancellationError() }
                if attempt + 1 == Self.maxLocalAttempts { throw PipelineError.analysisFailed(String(describing: error)) }
                try await retryDelay(attempt)
                continue
            }
            try Task.checkCancellation()
            analysis = result
            guard Self.worthRetrying(result) else { break }
            if !isAppActive() { throw CancellationError() }
            if attempt + 1 < Self.maxLocalAttempts { try await retryDelay(attempt) }
        }
        guard let analysis else { throw PipelineError.analysisFailed(nil) }

        switch analysis.outcome {
        case .complete:
            break
        case .stageFailed(_, _, true) where analysis.hasUsableResult():
            break   // rallies are there; only an enrichment stage (speeds, shot types) kept failing
        case let other:
            // Set failureRetryable based on the MatchAnalysis-based worthRetrying decision
            // before throwing, so fail() doesn't overwrite it with the Error-based heuristic.
            try current(id).failureRetryable = Self.worthRetrying(analysis)
            persist()
            throw PipelineError.analysisUnusable(other.userMessage)
        }

        let path = try await Task.detached(priority: .utility) {
            let path = try AnalysisStore.save(analysis, matchID: matchID)
            // Accepted as final (possibly with an enrichment stage missing): nothing to resume.
            AnalysisStore.archive().removeCheckpoints(matchID: matchID)
            return path
        }.value
        let done = try current(id)
        done.analysisPath = path
        MatchAnalysisMapper.apply(analysis, to: done)
        done.status = .done
        done.pipelineStep = nil
        done.failureReason = nil
        done.failureRetryable = nil
        persist()
    }

    private func createRemote(_ id: PersistentIdentifier) async throws {
        let match = try current(id)
        guard let url = match.localVideoURL, FileManager.default.fileExists(atPath: url.path) else {
            throw PipelineError.videoMissing
        }
        match.status = .uploading
        persist()
        activities[id] = PipelineActivity(phase: .creating)
        let service = try await scopedBackend(for: match, allowNew: true)
        let remote = try await service.createMatch(id: match.id, durationSeconds: match.durationSeconds)

        let updated = try current(id)
        guard remote.id == updated.id else { throw APIError.serverError(409, "服务器未保留比赛编号，请更新服务器") }
        updated.cloudJobID = nil
        persist()
    }

    private func upload(_ id: PersistentIdentifier) async throws {
        let match = try current(id)
        guard let fileURL = match.localVideoURL, FileManager.default.fileExists(atPath: fileURL.path) else {
            throw PipelineError.videoMissing
        }
        let matchID = match.id
        match.status = .uploading
        persist()
        activities[id] = PipelineActivity(phase: .uploading)

        let progress: @Sendable (Double) -> Void = { [weak self] fraction in
            Task { @MainActor in self?.reportUpload(fraction, id) }
        }
        let service = try await scopedBackend(for: match, allowNew: false)
        try await service.uploadVideo(matchID: matchID, fileURL: fileURL, progress: progress)
    }

    private func analyze(_ id: PersistentIdentifier) async throws {
        let match = try current(id)
        let matchID = match.id
        let doubles = match.courtType == .doubles
        match.status = .analyzing
        persist()
        activities[id] = PipelineActivity(phase: .analyzing, isQueued: true)

        let service = try await scopedBackend(for: match, allowNew: false)
        let job = try await service.triggerAnalysis(matchID: matchID, doubles: doubles, inputs: MatchInputs.load(matchID: matchID))
        let updated = try current(id)
        updated.cloudJobID = job.jobId
        persist()
    }

    private func poll(_ id: PersistentIdentifier) async throws -> StepResult {
        let match = try current(id)
        let matchID = match.id
        let localJobID = match.cloudJobID
        match.status = .analyzing
        persist()
        if activities[id]?.phase != .analyzing { activities[id] = PipelineActivity(phase: .analyzing) }

        // The server's match status is the truth (the job may have finished while we were away).
        let service = try await scopedBackend(for: match, allowNew: false)
        let remote = try await service.getMatch(id: matchID)
        switch remote.status {
        case "done":
            try await fetchResults(id)
            return .finished
        case "failed":
            var info: String?
            if let jobID = localJobID ?? remote.celeryJobId {
                info = try? await service.pollJob(matchID: matchID, jobID: jobID).info
            }
            throw PipelineError.analysisFailed(info == "None" ? nil : info)
        case "uploading":
            // Never analysed (or the trigger was lost): trigger again; a 400 there means re-upload.
            return .next(.analyze)
        default:
            guard let jobID = localJobID ?? remote.celeryJobId else { return .next(.analyze) }
            try current(id).cloudJobID = jobID
            try context.save()
            let scopedPoller = JobPollingService(backend: service, options: poller.options)
            let outcome = try await scopedPoller.waitForCompletion(matchID: matchID, jobID: jobID) { [weak self] status in
                await self?.reportJob(status, id)
            }
            if case .failed(let info) = outcome { throw PipelineError.analysisFailed(info) }
            try await fetchResults(id)
            return .finished
        }
    }

    private func fetchResults(_ id: PersistentIdentifier) async throws {
        let initial = try current(id)
        let service = try await scopedBackend(for: initial, allowNew: false)
        let matchID = initial.id
        activities[id] = PipelineActivity(phase: .fetching)
        if let archive = try await service.getAnalysisArchive(matchID: matchID) {
            guard !archive.rallies.isEmpty else {
                throw PipelineError.analysisUnusable(analysisOutcomeMessage(.noRallies))
            }
            // JSON encoding and full-video archive I/O stay off the UI thread.
            let path = try await Task.detached(priority: .utility) {
                try AnalysisStore.save(archive, matchID: matchID)
            }.value
            let match = try current(id)
            match.analysisPath = path
            MatchAnalysisMapper.apply(archive, to: match)
        } else {
            do {
                let doc = try await service.getAnalysis(matchID: matchID)
                let match = try current(id)
                JobPollingService.apply(doc, to: match)
            } catch let error as APIError where Self.isMissingAnalysis(error) {
                // Older server without the v2 document: fall back to the legacy trajectory.
                let trajectory = try await service.getTrajectory(matchID: matchID)
                let videoURL = try current(id).localVideoURL
                let frameSize = await Self.frameSize(of: videoURL)
                let match = try current(id)
                JobPollingService.apply(trajectory, to: match, frameSize: frameSize)
            }
        }
        let match = try current(id)
        guard !match.rallies.isEmpty else {
            throw PipelineError.analysisUnusable(analysisOutcomeMessage(.noRallies))
        }
        match.status = .done
        match.pipelineStep = nil
        match.failureReason = nil
        match.failureRetryable = nil
        persist()
    }

    /// Retrying helps only when a stage the verdict depends on was interrupted. A clean
    /// "no shuttle" / "no rallies" is deterministic, so running the same video again can't change it.
    static func worthRetrying(_ analysis: MatchAnalysis) -> Bool {
        guard analysis.outcome.isRetryable else { return false }

        guard case .stageFailed(let stage, _, _) = analysis.outcome else { return false }

        // P2's stopped_after means the run ended early on purpose to retry from partial.
        // R4: any stopped_after whose stage failed retryably → retry, even if outcome != stopped_after.
        if case .string(let stoppedAfter) = analysis.diagnostics["stopped_after"],
           analysis.stageFailures.contains(where: { $0.stage == stoppedAfter && $0.retryable }) {
            return true
        }

        // Q's original logic: for enrichment stages, if rallies are empty but we have a valid track,
        // and detect/link succeeded, then retrying the enrichment stage won't help.
        if analysis.rallies.isEmpty, analysis.track != nil,
           !["detect", "link", "open"].contains(stage),
           !analysis.stageFailures.contains(where: { ["detect", "link"].contains($0.stage) }) {
            return false   // the shuttle stages ran fine and found nothing to build on
        }

        return true
    }

    // MARK: - Recording has priority

    /// The record screen started a take: stop the running analysis (its finished stages are
    /// checkpointed) and start nothing new until the take ends. Writing 1080p60 and decoding
    /// another video at the same time can interrupt the hardware codec (AVError -11847).
    func recordingDidStart() {
        isRecordingPaused = true
        pauseLocalAnalysis(.recording, localStage: "recording")
    }

    func recordingDidStop() {
        guard isRecordingPaused else { return }
        isRecordingPaused = false
        resumePending()
    }

    /// An on-device analysis is running or waiting its turn (the record screen then rests its camera).
    var hasLocalWork: Bool { activities.values.contains { $0.phase == .analyzingOnDevice } }

    /// The record screen's camera started (preview or take): stop the running analysis (its
    /// finished stages are checkpointed) and start nothing new until the camera stops.
    func cameraDidStart() {
        isCameraLive = true
        pauseLocalAnalysis(.camera, localStage: AdmissionBlock.camera.localStage(cameraLive: true))
    }

    /// The phone reached "critical": stop the running analysis until it is cool again.
    func thermalPauseDidBegin() {
        pauseLocalAnalysis(.thermal, localStage: "cooling")
    }

    /// Stops the analysis holding the slot (its finished stages are checkpointed) and shows why it
    /// waits. Only a job that was really analysing logs `analysis_paused`, one still waiting its turn
    /// is already explained by its admission_wait; `analysis_resumed` follows once it is admitted again.
    private func pauseLocalAnalysis(_ reason: AdmissionBlock, localStage: String) {
        guard let id = localAnalysisID, let running = running[id] else { return }
        let wasAnalysing = localAnalysisActive && activities[id].map { MatchWaiting(localStage: $0.localStage) == nil } == true
        running.task.cancel()
        self.running[id] = nil
        activities[id] = PipelineActivity(phase: .analyzingOnDevice, localFraction: nil, localStage: localStage)
        guard wasAnalysing, paused[id] == nil else { return }
        let last = lastProgress[id]
        var fields = matchFields(id)
        fields.merge([
            "reason": .string(reason.rawValue),
            "stage": .string(last?.stage ?? "detect"),
            "thermal": .string(thermalLevel().name),
            "camera_live": .bool(isCameraLive),
            "frame": .null,
        ]) { $1 }
        if let last { fields["fraction"] = .number(last.fraction) }
        if let n = frameCounts[id] {
            fields["frames"] = .number(Double(n))
            if let last, let frame = Self.frame(of: last, frames: n) { fields["frame"] = .number(Double(frame)) }
        }
        admissionLog.record(event: "analysis_paused", fields)
        paused[id] = Pause(reason: reason, at: .now)
    }

    /// The video frame a detect / players progress stands for: both report frame ÷ frames
    /// (detect 10 → 50 %, players 60 → 80 % of the whole run, as AnalysisPipeline maps them).
    static func frame(of progress: AnalysisProgress, frames: Int) -> Int? {
        let range: (Double, Double)
        switch progress.stage {
        case "detect": range = (0.10, 0.40)
        case "players": range = (0.60, 0.20)
        default: return nil
        }
        let f = min(max((progress.fraction - range.0) / range.1, 0), 1)
        return Int((f * Double(frames)).rounded())
    }

    /// Ties a pause line to its match: `Match.id` and the video's file name (as in `localVideoPath`).
    private func matchFields(_ id: PersistentIdentifier) -> [String: JSONValue] {
        guard let match = fetch(id) else { return [:] }
        var fields: [String: JSONValue] = ["match": .string(match.id.uuidString)]
        if let path = match.localVideoPath { fields["video"] = .string((path as NSString).lastPathComponent) }
        return fields
    }

    private func forgetLocalProgress(_ id: PersistentIdentifier) {
        paused[id] = nil
        lastProgress[id] = nil
        frameCounts[id] = nil
    }

    nonisolated static func videoFrameCount(_ url: URL) async -> Int? {
        let asset = AVURLAsset(url: url)
        guard let track = try? await asset.loadTracks(withMediaType: .video).first,
              case let (fps, range)? = try? await track.load(.nominalFrameRate, .timeRange),
              fps > 0, range.duration.seconds.isFinite else { return nil }
        return Int((range.duration.seconds * Double(fps)).rounded())
    }

    func cameraDidStop() {
        guard isCameraLive else { return }
        isCameraLive = false
        resumePending()
    }

    // MARK: - On-device admission (one at a time, not while hot)

    /// What keeps an analysis from starting now (nil = it may start). Order = what the user can
    /// act on first: a take, the camera, the queue, then heat.
    func admissionBlock() -> AdmissionBlock? {
        if isRecordingPaused { return .recording }
        if isCameraLive { return .camera }
        if localAnalysisActive { return .busy }
        if thermalGate() != .run { return .thermal }
        return nil
    }

    /// Waits (cancellably) until the phone may start this analysis. Every wait is logged
    /// (begin + end with its reason) so a late start on a phone explains itself.
    private func admitLocalAnalysis(_ id: PersistentIdentifier) async throws {
        var waitingFor: AdmissionBlock?
        var began = ContinuousClock.now
        func end(_ reason: AdmissionBlock, outcome: String) {
            let s = began.duration(to: .now)
            admissionLog.record(["phase": .string("end"), "reason": .string(reason.rawValue), "outcome": .string(outcome),
                                 "seconds": .number(Double(s.components.seconds) + Double(s.components.attoseconds) * 1e-18),
                                 "thermal": .string(thermalLevel().name)])
        }
        while true {
            do { try Task.checkCancellation() } catch {
                if let w = waitingFor { end(w, outcome: "cancelled") }
                throw error
            }
            let block = admissionBlock()
            if block != waitingFor {
                if let w = waitingFor { end(w, outcome: block == nil ? "admitted" : "changed") }
                if let block {
                    began = .now
                    admissionLog.record(["phase": .string("begin"), "reason": .string(block.rawValue),
                                         "thermal": .string(thermalLevel().name), "camera_live": .bool(isCameraLive)])
                }
                waitingFor = block
            }
            guard let block else {
                localAnalysisActive = true
                localAnalysisID = id
                lastProgress[id] = nil
                if let pause = paused.removeValue(forKey: id) {
                    admissionLog.record(event: "analysis_resumed", matchFields(id).merging([
                        "reason": .string(pause.reason.rawValue),
                        "seconds": .number(AdmissionLog.seconds(pause.at.duration(to: .now))),
                        "thermal": .string(thermalLevel().name),
                    ]) { $1 })
                }
                return
            }
            let stage = block.localStage(cameraLive: isCameraLive)
            if activities[id]?.localStage != stage {
                activities[id] = PipelineActivity(phase: .analyzingOnDevice, localFraction: nil, localStage: stage)
            }
            do { try await Task.sleep(for: .milliseconds(300)) } catch {
                end(block, outcome: "cancelled")
                throw error
            }
        }
    }

    /// Follows the phone's temperature: at "critical" the running analysis is stopped (its
    /// finished stages are checkpointed, so it resumes from there) and restarted once cool.
    func startThermalMonitoring() {
        guard thermalObserver == nil else { return }
        var last: ThermalDecision = .run
        let evaluate: @MainActor () -> Void = { [weak self] in
            guard let self else { return }
            let info = ProcessInfo.processInfo
            let decision = ThermalPolicy.decide(ThermalLevel(info.thermalState), previous: last)
            last = decision
            self.thermalDecision = decision
            self.thermalMessage = ThermalPolicy.message(decision, lowPower: info.isLowPowerModeEnabled)
            if decision == .pause { self.thermalPauseDidBegin() }
            if decision == .run { self.resumePending() }
        }
        thermalGate = { last }
        thermalObserver = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: .main
        ) { [weak self] _ in MainActor.assumeIsolated { self?.thermalStateChanged?() } }
        thermalStateChanged = evaluate
        evaluate()
    }

    // MARK: - Failure

    private func fail(_ id: PersistentIdentifier, at step: PipelineStep, error: Error) {
        guard let match = fetch(id) else { return }
        match.status = .failed
        match.failureReason = Self.failureReason(for: error)
        if match.failureRetryable == nil {
            match.failureRetryable = Self.isWorthRetrying(error)
        }
        match.pipelineStep = Self.resumeStep(after: error, at: step)
        if case .analysisFailed? = error as? PipelineError {
            var inputs = MatchInputs.load(matchID: match.id)
            if inputs.cloudReanalysisID != nil {
                // A terminal job cannot be revived by repeating its idempotency key.
                // Keep network-only retries on their original key; a failed job gets a new intent.
                inputs.cloudReanalysisID = UUID()
                try? inputs.save(matchID: match.id)
            }
        }
        persist()
        if Self.isAuthFailure(error) {
            NotificationCenter.default.post(name: .shuttlSessionExpired, object: nil)
        }
    }

    // MARK: - Progress reporting

    private func reportUpload(_ fraction: Double, _ id: PersistentIdentifier) {
        guard var activity = activities[id], activity.phase == .uploading else { return }
        let previous = activity.uploadFraction ?? 0
        guard fraction >= 1 || fraction - previous >= 0.01 || activity.uploadFraction == nil else { return }
        activity.uploadFraction = max(previous, min(fraction, 1))
        activities[id] = activity
    }

    private func reportJob(_ status: JobStatusResponse, _ id: PersistentIdentifier) {
        guard running[id] != nil else { return }
        var activity = activities[id] ?? PipelineActivity(phase: .analyzing)
        guard activity.phase == .analyzing else { return }
        activity.isQueued = status.state == "PENDING"
        if let percent = JobPollingService.progressPercent(from: status.info) { activity.analysisPercent = percent }
        if activities[id] != activity { activities[id] = activity }
    }

    /// Throttled to whole percents / stage changes so SwiftUI isn't re-rendered per frame.
    private func reportLocal(_ p: AnalysisProgress, _ id: PersistentIdentifier) {
        guard running[id] != nil, var activity = activities[id], activity.phase == .analyzingOnDevice else { return }
        lastProgress[id] = p
        let previous = activity.localFraction ?? 0
        let fraction = max(previous, min(max(p.fraction, 0), 1))
        guard p.stage != activity.localStage || fraction - previous >= 0.01 || fraction >= 1 else { return }
        activity.localFraction = fraction
        activity.localStage = p.stage
        activities[id] = activity
    }

    // MARK: - SwiftData

    /// The live model, or `matchDeleted` if it is gone. Also a cancellation point.
    private func current(_ id: PersistentIdentifier) throws -> Match {
        try Task.checkCancellation()
        guard let match = fetch(id) else { throw PipelineError.matchDeleted }
        return match
    }

    private func fetch(_ id: PersistentIdentifier) -> Match? {
        if let match: Match = context.registeredModel(for: id) {
            return match.isDeleted ? nil : match
        }
        // Not loaded into this context yet (rare): a user has at most a few hundred matches.
        return (try? context.fetch(FetchDescriptor<Match>()))?.first { $0.persistentModelID == id }
    }

    private func persist() {
        do { try context.save() } catch { print("[MatchPipeline] save failed: \(error)") }
    }

    // MARK: - Policy (static, unit-tested)

    /// Where a match without a stored step resumes: on this phone when possible,
    /// unless the server already has a job for it.
    func inferredStep(for match: Match) -> PipelineStep {
        if analyzesOnDevice && match.cloudJobID == nil { return .local }
        return Self.inferredStep(for: match)
    }

    /// Server-path resume point (legacy rows, or rows from older builds).
    static func inferredStep(for match: Match) -> PipelineStep {
        match.cloudJobID != nil ? .poll : .create
    }

    /// Server-directed recoveries, each tried at most once per run.
    private static func heal(for error: Error, at step: PipelineStep) -> Heal? {
        guard case .serverError(let code, _)? = error as? APIError else { return nil }
        if code == 404 && step != .create { return .recreate }      // match gone (server DB reset)
        if code == 400 && step == .analyze { return .reupload }     // "视频尚未上传"
        return nil
    }

    /// The step "重试" should run: the one that failed, except that a failed or
    /// stalled server analysis is re-triggered rather than re-polled. On-device
    /// analysis always retries on the phone.
    static func resumeStep(after error: Error, at step: PipelineStep) -> PipelineStep {
        if step == .local { return .local }
        switch error as? PipelineError {
        case .analysisFailed?, .analysisUnusable?, .analysisStalled?: return .analyze
        default: return step
        }
    }

    private static func isMissingAnalysis(_ error: APIError) -> Bool {
        switch error {
        case .serverError(let code, _): return code == 404 || code == 400
        case .decodingError: return true
        default: return false
        }
    }

    /// Verdicts about the video's content: re-analysing the same file gives the same answer.
    static let contentVerdicts = [AnalysisOutcome.noShuttle, .noRallies, .cameraUnsteady].map(analysisOutcomeMessage)

    /// For `Match.failureRetryable`. Only a failure known to be deterministic is false;
    /// network, interruptions, stage errors and unknowns stay worth a retry.
    static func isWorthRetrying(_ error: Error) -> Bool {
        switch error as? PipelineError {
        case .invalidRecording?, .videoMissing?:
            return false
        case .analysisUnusable(let sentence)?:
            return !contentVerdicts.contains(sentence)
        case .analysisFailed(let info?)?:
            return !contentVerdicts.contains(where: info.contains)
        default:
            return true
        }
    }

    static func isAuthFailure(_ error: Error) -> Bool {
        switch error as? APIError {
        case .unauthorized?: return true
        case .serverError(401, _)?: return true
        default: return false
        }
    }

    /// One readable Chinese sentence for the failed-match UI.
    static func failureReason(for error: Error) -> String {
        if let error = error as? PipelineError {
            switch error {
            case .videoMissing:
                return "找不到这场比赛的本地视频文件"
            case .invalidRecording:
                return "录像文件为空，无法分析"
            case .analysisFailed(let info):
                guard let info, !info.isEmpty else { return "AI 分析没有成功：画面可能没拍全场地、光线太暗，或视频太短" }
                // The worker's "found nothing" verdicts are already the user sentence (tasks.py AnalysisUnusable).
                if let sentence = [analysisOutcomeMessage(.noShuttle), analysisOutcomeMessage(.noRallies)]
                    .first(where: info.contains) { return sentence }
                return "AI 分析失败：\(info.count > 120 ? String(info.prefix(120)) + "…" : info)"
            case .analysisUnusable(let sentence):
                return sentence
            case .analysisStalled:
                return "分析任务迟迟没有开始，请确认电脑上的分析服务（Celery worker）正在运行"
            case .matchDeleted:
                return "比赛已删除"
            case .originalAccountRequired:
                return "请登录这场比赛的原账号，并连接原分析服务器后重试"
            case .pendingRemoteCommand:
                return "这场比赛的云端删除或取消还在同步，请联网后重试"
            }
        }
        if isAuthFailure(error) { return "登录已过期，请重新登录后点重试" }
        if let error = error as? APIError {
            switch error {
            case .networkError(let underlying):
                return networkReason(underlying)
            case .timeout:
                return "连接分析服务器超时，请检查手机和电脑是否在同一 Wi-Fi"
            case .uploadFailed(let reason):
                return "视频上传失败：\(reason)"
            case .serverError(let code, let message):
                return "服务器错误（\(code)）：\(message)"
            case .decodingError:
                return "服务器返回的数据无法解析，App 和服务器版本可能不一致"
            case .unauthorized:
                return "登录已过期，请重新登录后点重试"
            }
        }
        return error.localizedDescription
    }

    private static func networkReason(_ error: Error) -> String {
        let code = (error as? URLError)?.code
        switch code {
        case .cannotConnectToHost?, .cannotFindHost?, .notConnectedToInternet?, .dnsLookupFailed?:
            return "连不上分析服务器：请确认手机和电脑在同一 Wi-Fi、服务已启动，并允许 Shuttl 访问本地网络"
        case .networkConnectionLost?:
            return "网络连接中断（上传时请保持 Shuttl 在前台）"
        default:
            return "网络错误：\(error.localizedDescription)"
        }
    }

    // MARK: - Platform helpers

    private static func frameSize(of url: URL?) async -> CGSize? {
        guard let url else { return nil }
        return await JobPollingService.uprightVideoSize(of: url)
    }

    /// Asks iOS for extra time so a short trip to the home screen doesn't kill the work.
    private static func withBackgroundTime<T: Sendable>(named name: String, _ work: () async throws -> T) async throws -> T {
        let holder = BackgroundTaskHolder()
        holder.id = UIApplication.shared.beginBackgroundTask(withName: name) {
            MainActor.assumeIsolated { holder.end() }
        }
        defer { holder.end() }
        return try await work()
    }

    // MARK: - Demo

    #if DEBUG
    /// Frozen progress for seeded demo matches so screenshots show the live states.
    func seedDemoActivity() {
        let matches = (try? context.fetch(FetchDescriptor<Match>())) ?? []
        // -UIDemoWaitReason <queued|recording|camera|cooling|cooling_camera>: every analysing local match
        // waits for that reason, as on the phone (a take, the camera or heat stops every start).
        let args = ProcessInfo.processInfo.arguments
        if let i = args.firstIndex(of: "-UIDemoWaitReason"), i + 1 < args.count {
            let reason = args[i + 1]
            for match in matches where match.status == .analyzing && match.pipelineStep == .local {
                activities[match.persistentModelID] = PipelineActivity(phase: .analyzingOnDevice, localStage: reason)
            }
            return
        }
        // Like the phone: the newest analysing match runs, the others wait their turn.
        let newestLocal = matches.filter { $0.status == .analyzing && $0.pipelineStep == .local }
            .max { $0.createdAt < $1.createdAt }?.persistentModelID
        for match in matches {
            let id = match.persistentModelID
            switch match.status {
            case .uploading:
                activities[id] = PipelineActivity(phase: .uploading, uploadFraction: 0.42)
            case .analyzing where match.pipelineStep == .local:
                activities[id] = id == newestLocal
                    ? PipelineActivity(phase: .analyzingOnDevice, localFraction: 0.63, localStage: "players")
                    : PipelineActivity(phase: .analyzingOnDevice, localStage: "queued")
            case .analyzing:
                activities[id] = PipelineActivity(phase: .analyzing, analysisPercent: 63)
            default: break
            }
        }
    }
    #endif
}

#if DEBUG
/// Stands in for the on-device analyzer in demo mode; demo matches are seeded, so it is never asked.
private struct DemoAnalyzer: OnDeviceAnalyzer {
    func analyze(video: URL, options: PipelineOptions, checkpoints: CheckpointStore,
                 progress: @escaping @Sendable (AnalysisProgress) -> Void,
                 stopOnRetryableFailure: Bool) async throws -> MatchAnalysis {
        throw CancellationError()
    }
}
#endif

@MainActor
private final class BackgroundTaskHolder {
    var id: UIBackgroundTaskIdentifier = .invalid

    func end() {
        guard id != .invalid else { return }
        UIApplication.shared.endBackgroundTask(id)
        id = .invalid
    }
}
