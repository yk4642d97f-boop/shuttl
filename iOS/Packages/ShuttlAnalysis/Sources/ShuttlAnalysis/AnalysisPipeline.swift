import Foundation
import ShuttlCore

/// Where the run is, for the progress UI. `fraction` is 0…1 overall.
public struct AnalysisProgress: Sendable, Equatable {
    public var fraction: Double
    public var stage: String
    public init(fraction: Double, stage: String) { self.fraction = fraction; self.stage = stage }
}

/// Port of pipeline_v2.analyze_video.
///
///   probe → detect shuttle candidates → link into a 2-D track
///         → calibrate camera (user-tapped corners, else automatic, else hint)
///         → track players + pose → detect hits → fit 3-D flights
///         → segment rallies → classify shots → score
///
/// Every stage is isolated: a failure is recorded in diagnostics["stages"] and
/// downstream stages degrade (no camera → no metric speeds / landings, never
/// fabricated ones). Task cancellation is NOT swallowed — it propagates.
/// Stage percentages match Python so progress reads the same on both.
///
/// Failures: every failed stage is also listed in diagnostics["stage_failures"]
/// (with the NSError domain / code and a `retryable` verdict); `MatchAnalysis.outcome`
/// turns that into what the app should do. Checkpoints are cleared only after a run
/// in which no stage failed, so a retry resumes from the stages that succeeded.
/// Exception to "downstream stages degrade": a RETRYABLE detect failure ends the run right
/// after detect (diagnostics["stopped_after"] = "detect"), so the caller retries at once and
/// detect resumes from detect_partial.bin instead of later stages running interrupted first.
/// players is resumable the same way (players_partial.bin, stopped_after = "players").
/// `stopOnRetryableFailure: false` turns both stops off (a caller's last attempt: the later
/// stages then run on what there is, as before; the partials are kept either way).
public struct AnalysisPipeline: Sendable {
    public let stages: AnalysisStages
    public let checkpoints: CheckpointStore
    /// Per-stage hooks (resource sampling, logging); no-op by default.
    public let observer: any StageObserver
    /// End the run right after a retryable detect / players failure (diagnostics["stopped_after"]).
    public let stopOnRetryableFailure: Bool

    public init(stages: AnalysisStages, checkpoints: CheckpointStore = NoCheckpoints(),
                observer: any StageObserver = NoStageObserver(), stopOnRetryableFailure: Bool = true) {
        self.stages = stages
        self.checkpoints = checkpoints
        self.observer = observer
        self.stopOnRetryableFailure = stopOnRetryableFailure
    }

    public func analyze(_ url: URL, options: PipelineOptions = PipelineOptions(),
                        progress: @escaping @Sendable (AnalysisProgress) -> Void = { _ in }) async throws -> MatchAnalysis {
        let start = ContinuousClock.now
        let source = try await VideoSource.open(url)
        return try await analyze(source, options: options, startedAt: start, progress: progress)
    }

    public func analyze(_ source: VideoSource, options: PipelineOptions = PipelineOptions(),
                        progress: @escaping @Sendable (AnalysisProgress) -> Void = { _ in }) async throws -> MatchAnalysis {
        try await analyze(source, options: options, startedAt: .now, progress: progress)
    }

    private func analyze(_ source: VideoSource, options opts: PipelineOptions, startedAt: ContinuousClock.Instant,
                         progress: @escaping @Sendable (AnalysisProgress) -> Void) async throws -> MatchAnalysis {
        let meta = source.meta
        var res = MatchAnalysis(meta: meta)
        let runner = StageRunner(progress: progress, observer: observer)
        res.diagnostics["video"] = .object([
            "fps": .number(meta.fps),
            "size": .array([.number(Double(meta.width)), .number(Double(meta.height))]),
            "frames": .number(Double(meta.nFrames)),
        ])
        let size = meta.size

        // detect: 10 → 50 %. Resumable: progress goes to the "detect_partial" file every
        // couple of minutes of video, and a killed run continues from the last one.
        res.diagnostics["detector"] = .string(stages.detectorName)
        let partial = checkpoints as? any DetectPartialStore
        let candidates: [[ShuttleCandidate]]? = try await runner.run("detect", 0.10, checkpoint: ("candidates", checkpoints)) {
            var resume: DetectCheckpoint?
            if let partial {
                // Every load and save is logged: a nil load has several causes and saves are best
                // effort, so without this a phone run can't show why a retry started from frame 0.
                let (loaded, report) = partial.loadDetectPartialReporting()
                resume = loaded
                stages.recordEvent("detect_partial_load", report.json)
                runner.annotate(["partial_load": .string(report.verdict)])
            }
            if let resume { runner.annotate(["resume_offered_frame": .number(Double(resume.nextFrame))]) }
            return try await stages.detect(source, resume: resume, checkpoint: { c in
                guard let partial else { return }
                let report = partial.saveDetectPartialReporting(c)
                stages.recordEvent("detect_partial_save", report.json)
                runner.annotate(["partial_last_save": .string(report.verdict),
                                 "partial_last_frame": .number(Double(c.nextFrame))])
            }) { f in
                progress(AnalysisProgress(fraction: 0.10 + 0.40 * clamp01(f), stage: "detect"))
            }
        }
        // The partial progress is obsolete only once "candidates" is really on disk: saves are best effort, and on a
        // full disk that save is a no-op, so dropping the partial would restart detect from frame 0 next time.
        if let partial, candidates != nil, checkpoints.load([[ShuttleCandidate]].self, key: "candidates") != nil {
            partial.clearDetectPartial()
            stages.recordEvent("detect_partial_clear", ["reason": .string("candidates_saved")])
        }
        // A retryable detect failure (decoder taken away, app sent to the background) ends the run
        // here. Everything after detect would run in the same interrupted state and checkpoint what it
        // got, and whatever cancels it next (the camera, a lock) ends the caller's in-process retry,
        // the one path that reuses this store. The caller retries now and detect resumes from its partial.
        if stopOnRetryableFailure, candidates == nil, let f = runner.failures.first(where: { $0.stage == "detect" }), f.retryable {
            res.diagnostics["stopped_after"] = .string("detect")
            return finish(res, runner, startedAt: startedAt, progress: progress)
        }
        if let candidates {
            res.track = try await runner.run("link", 0.50, checkpoint: ("track", checkpoints)) {
                try stages.link(candidates, meta: meta)
            }
        }

        let autoDiagnostics = DiagnosticsBox()
        let measured = Box<CameraSteadiness>()
        res.camera = try await runner.run("calibrate", 0.55, checkpoint: ("camera", checkpoints)) {
            try calibrate(source, size: size, opts, steadiness: { measured.value = $0 }) { autoDiagnostics.value = $0 }
        } ?? nil
        // Measured on the calibration frames; kept as its own checkpoint so a resumed run
        // (camera restored, frames never decoded) still knows whether the phone was still.
        let motionKey = CameraSteadiness.diagnosticsKey
        if let m = measured.value { checkpoints.save(m, key: motionKey) }
        if let m = measured.value ?? checkpoints.load(CameraSteadiness.self, key: motionKey) {
            res.diagnostics[motionKey] = m.json
        }
        if let cam = res.camera {
            res.diagnostics["calibration"] = .object([
                "confidence": .number(cam.confidence),
                "reproj_rmse_px": .number(cam.reprojRmsePx),
            ])
        }
        // Why the automatic path accepted or rejected the court (absent when manual corners
        // were used, or when the camera came back from a checkpoint).
        if let auto = autoDiagnostics.value { res.diagnostics["calibration_auto"] = auto }

        // players: 60 → 80 % (Python reports 60 then jumps to 80)
        let step = playerStep(fps: meta.fps, override: opts.playerStep)
        let camera = res.camera
        // Resumable like detect: progress goes to "players_partial" every couple of minutes of
        // video or ~30 s of wall time, and a killed / failed run continues from the last save.
        let playersPartial = checkpoints as? any PlayersPartialStore
        let playersOut: [PlayerObservation]? = try await runner.run("players", 0.60, checkpoint: ("players", checkpoints)) {
            var resume: PlayersCheckpoint?
            if let playersPartial {
                let (loaded, report) = playersPartial.loadPlayersPartialReporting()
                resume = loaded
                stages.recordEvent("players_partial_load", report.json)
                runner.annotate(["partial_load": .string(report.verdict)])
            }
            if let resume { runner.annotate(["resume_offered_frame": .number(Double(resume.nextFrame))]) }
            return try await stages.trackPlayers(source, camera: camera, step: step, doubles: opts.doubles,
                                                 resume: resume, checkpoint: { c in
                guard let playersPartial else { return }
                let report = playersPartial.savePlayersPartialReporting(c)
                stages.recordEvent("players_partial_save", report.json)
                runner.annotate(["partial_last_save": .string(report.verdict),
                                 "partial_last_frame": .number(Double(c.nextFrame))])
            }) { f in
                progress(AnalysisProgress(fraction: 0.60 + 0.20 * clamp01(f), stage: "players"))
            }
        }
        res.players = playersOut ?? []
        // As for detect: obsolete only once "players" is really on disk (saves are best effort).
        if let playersPartial, playersOut != nil, playersPartial.hasPlayersPartial,
           checkpoints.load([PlayerObservation].self, key: "players") != nil {
            playersPartial.clearPlayersPartial()
            stages.recordEvent("players_partial_clear", ["reason": .string("players_saved")])
        }
        // A retryable players failure ends the run here, like detect's: the later stages would
        // run without players and checkpoint nothing useful; the caller retries and players resumes.
        if stopOnRetryableFailure, playersOut == nil, let f = runner.failures.first(where: { $0.stage == "players" }), f.retryable {
            res.diagnostics["stopped_after"] = .string("players")
            return finish(res, runner, startedAt: startedAt, progress: progress)
        }
        let players: [PlayerObservation]? = res.players.isEmpty ? nil : res.players

        guard let track = res.track else {
            return finish(res, runner, startedAt: startedAt, progress: progress)
        }

        res.hits = try await runner.run("hits", 0.80) {
            try stages.detectHits(track: track, meta: meta, camera: camera, players: players)
        } ?? []
        let hits = res.hits
        var reconstruction = ReconstructionResult.empty
        if let camera {
            reconstruction = try await runner.run("reconstruct3d", 0.85) {
                try stages.reconstruct(track: track, hits: hits, camera: camera, meta: meta, players: players)
            } ?? .empty
        }
        res.segments = reconstruction.segments
        let recon = reconstruction

        res.rallies = try await runner.run("rallies", 0.92) {
            try stages.segmentRallies(track: track, hits: hits, reconstruction: recon, meta: meta,
                                      camera: camera, players: players)
        } ?? []

        var classified: [RallyV2] = []
        classified.reserveCapacity(res.rallies.count)
        for rally in res.rallies {
            // Python records one "shots" entry per rally; the last one wins.
            let r = try await runner.run("shots", 0.95) {
                try stages.classifyShots(rally, camera: camera, players: players, meta: meta, track: track)
            } ?? rally
            classified.append(r)
        }
        res.rallies = classified
        let dropped = LandingGuard.apply(rallies: &res.rallies, segments: &res.segments)
        if dropped > 0 { res.diagnostics[LandingGuard.diagnosticsKey] = .number(Double(dropped)) }
        let rallies = res.rallies

        res.score = try await runner.run("score", 0.98) { try stages.scoreMatch(rallies, players: players) } ?? []

        return finish(res, runner, startedAt: startedAt, progress: progress)
    }

    /// Stamps diagnostics; clears checkpoints only when every stage that ran succeeded,
    /// so a failed run keeps what it computed and a retry resumes from there.
    private func finish(_ result: MatchAnalysis, _ runner: StageRunner, startedAt: ContinuousClock.Instant,
                        progress: @Sendable (AnalysisProgress) -> Void) -> MatchAnalysis {
        var res = result
        res.diagnostics["stages"] = runner.stagesJSON
        let failures = runner.failures
        if !failures.isEmpty {
            res.diagnostics[StageFailure.diagnosticsKey] = .array(failures.map(\.json))
        }
        res.processingTimeSeconds = seconds(since: startedAt)
        if failures.isEmpty {
            checkpoints.clear()
            stages.recordEvent("checkpoints_clear", ["reason": .string("run_without_failures")])
        }
        progress(AnalysisProgress(fraction: 1, stage: "done"))
        return res
    }

    /// Prefer user-tapped corners (then refine on lines), else fully automatic, else the hint.
    private func calibrate(_ source: VideoSource, size: ImageSize, _ opts: PipelineOptions,
                           steadiness: (CameraSteadiness) -> Void = { _ in },
                           report: (JSONValue) -> Void = { _ in }) throws -> CameraModel? {
        let frames = try stages.calibrationFrames(source, count: opts.calibrationFrames)
        // Manual corners can't rescue a hand-held clip, so this runs on both paths.
        if let s = CameraSteadinessEstimator.estimate(frames) { steadiness(s) }
        if let corners = opts.manualCorners {
            let sx = opts.cornersNormalized ? Double(size.width) : 1
            let sy = opts.cornersNormalized ? Double(size.height) : 1
            let pts = corners.mapValues { SIMD2($0.x * sx, $0.y * sy) }
            var cam = try stages.calibrateFromPoints(pts, size: size)
            if !frames.isEmpty {
                // Refinement failure keeps the DLT result (pipeline_v2 logs a warning).
                if let refined = try? stages.refineCamera(cam, frames: frames) { cam = refined }
            }
            return cam
        }
        let attempt = try stages.calibrateAutomaticDetailed(frames: frames, size: size, doublesHint: opts.doubles,
                                                            focalHintPx: opts.focalHintPx)
        report(.object(attempt.diagnostics))
        var cam = attempt.camera
        if cam == nil, let hint = opts.cameraHint { cam = hint }
        return cam
    }

    private func seconds(since t: ContinuousClock.Instant) -> Double {
        let d = ContinuousClock.now - t
        return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
    }
}

@inline(__always) private func clamp01(_ x: Double) -> Double { x.isFinite ? min(max(x, 0), 1) : 0 }

/// Carries a typed value out of a stage closure.
private final class Box<T: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: T?
    var value: T? {
        get { lock.withLock { stored } }
        set { lock.withLock { stored = newValue } }
    }
}

/// Carries a value out of a stage closure (the runner owns the closure's lifetime).
private final class DiagnosticsBox: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: JSONValue?
    var value: JSONValue? {
        get { lock.withLock { stored } }
        set { lock.withLock { stored = newValue } }
    }
}

/// pipeline_v2._Stages: timing + failure isolation per stage.
private final class StageRunner: @unchecked Sendable {
    private let progress: @Sendable (AnalysisProgress) -> Void
    private let observer: any StageObserver
    private var stages: [String: JSONValue] = [:]
    /// Every failure, in order (diagnostics["stages"] keeps only the last per name,
    /// like Python; per-rally "shots" failures would otherwise hide behind a later success).
    private var failureList: [StageFailure] = []
    private let lock = NSLock()

    init(progress: @escaping @Sendable (AnalysisProgress) -> Void, observer: any StageObserver) {
        self.progress = progress
        self.observer = observer
    }

    var stagesJSON: JSONValue { lock.withLock { .object(stages) } }
    var failures: [StageFailure] { lock.withLock { failureList } }

    /// Extra keys for the stage that is running now (added to its record; existing keys win).
    private var pendingExtras: [String: JSONValue] = [:]
    func annotate(_ extra: [String: JSONValue]) { lock.withLock { pendingExtras.merge(extra) { _, new in new } } }

    private func record(_ name: String, _ entry: [String: JSONValue]) {
        lock.withLock {
            var e = entry
            e.merge(pendingExtras) { old, _ in old }
            pendingExtras = [:]
            stages[name] = .object(e)
        }
    }

    private func addFailure(_ f: StageFailure) {
        lock.withLock {
            // One entry per stage + message; repeats (e.g. every rally's shots) are counted.
            if let i = failureList.firstIndex(where: { $0.stage == f.stage && $0.message == f.message }) {
                failureList[i].occurrences += 1
            } else {
                failureList.append(f)
            }
        }
    }

    /// Runs `body`; on error records it and returns nil. Cancellation propagates.
    /// With a checkpoint, a stored result short-circuits the stage and a fresh one is saved.
    func run<T: Codable & Sendable>(_ name: String, _ pct: Double,
                                    checkpoint: (key: String, store: CheckpointStore)? = nil,
                                    _ body: () async throws -> T) async throws -> T? {
        progress(AnalysisProgress(fraction: pct, stage: name))
        if let (key, store) = checkpoint, let cached = store.load(T.self, key: key) {
            record(name, ["ok": .bool(true), "seconds": .number(0), "resumed": .bool(true)])
            return cached
        }
        try Task.checkCancellation()
        let t0 = ContinuousClock.now
        observer.stageDidStart(name)
        let result: Result<T, Error>
        do { result = .success(try await body()) } catch { result = .failure(error) }
        let secs = rounded(t0)
        observer.stageDidFinish(name, seconds: secs)
        let extra = observer.stageDiagnostics(name)

        switch result {
        case .success(let out):
            var entry: [String: JSONValue] = ["ok": .bool(true), "seconds": .number(secs)]
            entry.merge(extra) { old, _ in old }
            record(name, entry)
            // Save before re-checking cancellation: a stage that completed is worth keeping
            // even when a pause arrived at the last moment.
            if let (key, store) = checkpoint { store.save(out, key: key) }
            try Task.checkCancellation()
            return out
        case .failure(let error):
            // A pause / cancel can surface as a stage-specific error (reader stopped, GPU
            // work refused); it is still a cancellation, not a failure of the stage.
            if error is CancellationError || Task.isCancelled { throw CancellationError() }
            let failure = StageFailure(stage: name, error: error)
            var entry: [String: JSONValue] = [
                "ok": .bool(false),
                "error": .string("\(type(of: error)): \(error)"),
                "seconds": .number(secs),
                "error_domain": .string(failure.domain),
                "error_code": .number(Double(failure.code)),
                "retryable": .bool(failure.retryable),
            ]
            entry.merge(extra) { old, _ in old }
            record(name, entry)
            addFailure(failure)
            return nil
        }
    }

    private func rounded(_ t0: ContinuousClock.Instant) -> Double {
        let d = ContinuousClock.now - t0
        let s = Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
        return (s * 1000).rounded() / 1000
    }
}
