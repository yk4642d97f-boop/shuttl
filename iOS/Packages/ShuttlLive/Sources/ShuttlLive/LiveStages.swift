import CoreML
import Foundation
import ShuttlAnalysis
import ShuttlCore
import ShuttlCourt
import ShuttlDetector
import ShuttlHitModel
import ShuttlPlayers
import ShuttlRally

/// `AnalysisStages` over the ported packages — the on-device equivalent of
/// pipeline_v2's module imports.
public struct LiveStages: AnalysisStages {
    /// Every pipeline_v2 stage is ported and parity-checked, so the app analyses on
    /// the phone by default.
    public static let isComplete = true

    /// Where stage metrics go by default: Documents/Diagnostics/stage-metrics.jsonl on
    /// iOS; on macOS only if SHUTTL_STAGE_METRICS names a file.
    public static var defaultMetricsURL: URL? { StageMetrics.defaultURL }

    let detector: HeatmapDetector
    /// Seconds of video between detect checkpoints (the pipeline persists the latest).
    public var checkpointSeconds: Double = 120
    /// Wall-clock seconds since the last detect / players save after which the next boundary
    /// saves anyway (a hot phone needs up to ~130 s of wall time for 120 s of video). nil = off.
    public var checkpointWallSeconds: Double? = 30
    /// players fails with a retryable `StageStalledError` when no sampled frame completes for
    /// this long (nil = off). A hot phone does a sampled frame in ~45 ms (players 385-411 s for
    /// ~9000 frames of a 10-min clip), Vision's first request a few s; 60 s is > 1000× the mean,
    /// longer than the 20-s decode watchdog (a decoder stall keeps its own error), and much
    /// shorter than the > 1 h T6 hang. A false alarm costs only a resume from players_partial.bin.
    public var playersStallSeconds: Double? = 60
    /// A players_progress event this often (nil = off), also while no frame completes.
    public var playersHeartbeatSeconds: Double? = 10
    /// Tests: the pose detector to use instead of Vision (built per pass).
    var poseDetectorOverride: (@Sendable () -> PoseDetector)?
    var watchClock: @Sendable () -> TimeInterval = { ProcessInfo.processInfo.systemUptime }
    var watchPollSeconds: Double = 1
    /// HeatmapDetector is not reentrant; every copy of these stages shares one gate.
    private let detectorGate: AsyncGate
    private let metrics: StageMetrics
    public let detectorName: String

    /// Loads the detector the way build_detector does: `kind` from `modelsDirectory`
    /// (auto = TrackNetV3, then WASB; fine-tuned exports first). Compiling a model
    /// happens once and is cached in `cacheDirectory`. `metricsURL` receives one JSON
    /// line per stage (default: Documents/Diagnostics/stage-metrics.jsonl on iOS).
    public static func make(modelsDirectory: URL, kind: DetectorKind = .auto,
                            computeUnits: MLComputeUnits = .all, cacheDirectory: URL? = nil,
                            metricsURL: URL? = LiveStages.defaultMetricsURL) async throws -> LiveStages {
        let metrics = StageMetrics(url: metricsURL)
        let det = try await metrics.measure("load_detector", ["models": .string(modelsDirectory.lastPathComponent)]) {
            try await HeatmapDetector.build(kind: kind, modelsDirectory: modelsDirectory,
                                            computeUnits: computeUnits, cacheDirectory: cacheDirectory)
        }
        let name = det.network.spec.name
        if metricsURL != nil {
            let order = kind == .auto ? HeatmapDetector.autoOrder : [kind]
            if let url = order.lazy.compactMap({ HeatmapDetector.findModel($0, in: modelsDirectory) }).first {
                Task.detached(priority: .background) {
                    var plan = await ComputePlanReport.summarize(compiledModel: url)
                    plan["model"] = .string(name)
                    metrics.record("compute_plan", plan)
                }
            }
        }
        return LiveStages(detector: det, detectorName: name, metrics: metrics)
    }

    init(detector: HeatmapDetector, detectorName: String, metrics: StageMetrics = StageMetrics(url: nil)) {
        self.detector = detector
        self.detectorName = detectorName
        self.detectorGate = AsyncGate()
        self.metrics = metrics
    }

    /// Resumable detect. The signature must match the AnalysisStages requirement exactly,
    /// or Swift silently binds the default (non-resuming) extension instead.
    public func detect(_ source: VideoSource, resume: DetectCheckpoint?,
                       checkpoint: @escaping @Sendable (DetectCheckpoint) -> Void,
                       progress: @escaping StageProgress) async throws -> [[ShuttleCandidate]] {
        let m = source.meta
        let r = resume.map { DetectResume(nextFrame: $0.nextFrame, candidates: $0.candidates,
                                          background: $0.background, detectorName: $0.detectorName) }
        metrics.record("run_start", ["frames": .number(Double(m.nFrames)), "fps": .number(m.fps),
                                     "width": .number(Double(m.width)), "height": .number(Double(m.height)),
                                     "detector": .string(detectorName),
                                     "resume_from": .number(Double(r?.nextFrame ?? 0))])
        let every = checkpointSeconds, wall = checkpointWallSeconds
        let out = try await metrics.measure("detect") {
            try await withGate(detectorGate) {
                try await detector.detectVideo(source, resume: r, checkpointSeconds: every, checkpointWallSeconds: wall,
                                               checkpoint: { c in
                    checkpoint(DetectCheckpoint(nextFrame: c.nextFrame, candidates: c.candidates,
                                                background: c.background, detectorName: c.detectorName))
                }, progress: { progress($0) })
            }
        }
        if r != nil {
            metrics.record("detect_resume", ["used": .bool(detector.lastResumeProblem == nil),
                                             "problem": detector.lastResumeProblem.map(JSONValue.string) ?? .null])
        }
        return out
    }

    public func detect(_ source: VideoSource, progress: @escaping StageProgress) async throws -> [[ShuttleCandidate]] {
        let m = source.meta
        metrics.record("run_start", ["frames": .number(Double(m.nFrames)), "fps": .number(m.fps),
                                     "width": .number(Double(m.width)), "height": .number(Double(m.height)),
                                     "detector": .string(detectorName)])
        return try await metrics.measure("detect") {
            try await withGate(detectorGate) {
                try await detector.detectVideo(source, progress: { progress($0) })
            }
        }
    }

    public func link(_ candidates: [[ShuttleCandidate]], meta: VideoMeta) throws -> Track2D {
        try timed("link") { try PortedStages.link(candidates, meta: meta) }
    }

    /// RGB frames at the upright size, the indices video_io.sample_frames picks.
    public func calibrationFrames(_ source: VideoSource, count: Int) throws -> [ImageU8] {
        try timed("calibration_frames") { try source.sampleFrames(count) { try ImageU8.rgb(from: $0) }.map(\.value) }
    }

    public func calibrateAutomatic(frames: [ImageU8], size: ImageSize, doublesHint: Bool?) throws -> CameraModel? {
        try timed("calibrate_auto") { try CourtStage.automatic(frames, size, doublesHint) }
    }

    /// Automatic calibration with its evidence. The pipeline stores it as
    /// diagnostics["calibration_auto"], the same key and fields pipeline_v2 writes.
    /// The signature must match the AnalysisStages requirement exactly, or the default
    /// (no evidence) is used silently.
    public func calibrateAutomaticDetailed(frames: [ImageU8], size: ImageSize, doublesHint: Bool?) throws -> CalibrationAttempt {
        try calibrateAutomaticDetailed(frames: frames, size: size, doublesHint: doublesHint, focalHintPx: nil)
    }

    /// With the lens's measured focal (px at `size`) as a veto on implausible cameras.
    /// The pipeline calls this one; same exact-signature caveat.
    public func calibrateAutomaticDetailed(frames: [ImageU8], size: ImageSize, doublesHint: Bool?,
                                           focalHintPx: Double?) throws -> CalibrationAttempt {
        let r = timed("calibrate_auto") { CourtStage.automaticDetailed(frames, size, doublesHint, focalHintPx: focalHintPx) }
        // The device check the pipeline owner asked for: measured vs solved focal.
        var focal: [String: JSONValue] = ["focal_hint_px": focalHintPx.map(JSONValue.number) ?? .null]
        if let cam = r.camera { focal["focal_solved_px"] = .number(cam.K[0, 0]) }
        if let hint = focalHintPx, let cam = r.camera, hint > 0 { focal["focal_rel_diff"] = .number(cam.K[0, 0] / hint - 1) }
        focal["reason"] = r.diagnostics["reason"] ?? .null
        metrics.record("calibration_focal", focal)
        return CalibrationAttempt(camera: r.camera, diagnostics: r.diagnostics)
    }

    public func calibrateFromPoints(_ points: [String: SIMD2<Double>], size: ImageSize) throws -> CameraModel {
        try timed("calibrate_points") { try CourtCalibration.calibrateFromPoints(points, size: size) }
    }

    public func refineCamera(_ camera: CameraModel, frames: [ImageU8]) throws -> CameraModel {
        timed("refine_camera") { CourtCalibration.refineCamera(camera, frames: frames) }
    }

    public func trackPlayers(_ source: VideoSource, camera: CameraModel?, step: Int, doubles: Bool?,
                             progress: @escaping StageProgress) async throws -> [PlayerObservation] {
        try await trackPlayers(source, camera: camera, step: step, doubles: doubles, resume: nil,
                               checkpoint: { _ in }, progress: progress)
    }

    /// Resumable players: continues from `resume` when its context (camera, step, crop rects)
    /// matches this run, saves at the first sampled frame after `checkpointSeconds` of video or
    /// `checkpointWallSeconds` of wall time since the last save. Exact signature (see detect).
    public func trackPlayers(_ source: VideoSource, camera: CameraModel?, step: Int, doubles: Bool?,
                             resume: PlayersCheckpoint?, checkpoint: @escaping @Sendable (PlayersCheckpoint) -> Void,
                             progress: @escaping StageProgress) async throws -> [PlayerObservation] {
        let r = resume.map { PlayersResume(nextFrame: $0.nextFrame, context: $0.context, frames: $0.frames) }
        let policy = PlayersCheckpointPolicy(everyFrames: Int((checkpointSeconds * max(source.meta.fps, 1)).rounded()),
                                             wallSeconds: checkpointWallSeconds)
        let problem = ProblemBox()
        let watchConfig = StageWatchConfig(stallSeconds: playersStallSeconds, heartbeatSeconds: playersHeartbeatSeconds,
                                           pollSeconds: watchPollSeconds, clock: watchClock)
        let makeDetector = poseDetectorOverride, metrics = metrics
        let fields: [String: JSONValue] = ["frames": .number(Double(source.meta.nFrames)), "step": .number(Double(step)),
                                           "resume_from": .number(Double(r?.nextFrame ?? 0))]
        let out = try await metrics.measure("players", ["step": .number(Double(step)),
                                                        "resume_from": .number(Double(r?.nextFrame ?? 0))]) {
            try await watchedDetached(stage: "players", config: watchConfig, fields: fields,
                                      onEvent: { metrics.record($0, $1) }) { watch in
                try VideoStages.trackPlayers(source, camera: camera, step: step, doubles: doubles, resume: r,
                                             checkpointPolicy: policy,
                                             checkpoint: { c in
                                                 watch.ifLive { checkpoint(PlayersCheckpoint(nextFrame: c.nextFrame,
                                                                                             context: c.context, frames: c.frames)) }
                                             },
                                             resumeReport: { problem.value = $0 },
                                             detector: makeDetector?(), onFrame: { watch.frameDone($0) },
                                             progress: progress, isCancelled: { watch.flag.isSet })
            }
        }
        if r != nil {
            metrics.record("players_resume", ["used": .bool(problem.value == nil),
                                              "problem": problem.value.map(JSONValue.string) ?? .null])
        }
        return out
    }

    /// The learned detector is in front of the ported rule detector here. Which one runs is
    /// `HitStage.Registry.detector` (default `.learned`); `.rules` restores the kinematic
    /// detector unchanged, and a missing model falls back to it rather than failing the stage.
    public func detectHits(track: Track2D, meta: VideoMeta, camera: CameraModel?, players: [PlayerObservation]?) throws -> [HitEvent] {
        try Self.requireConsistent(track)
        guard HitStage.Registry.detector == .learned, let model = HitStage.Registry.model else {
            return timed("hits") { PortedStages.hits(track: track, meta: meta, camera: camera, players: players) }
        }
        do {
            let hits = try timed("hits") {
                try HitStage.detect(track: track, meta: meta, players: players, model: model)
            }
            metrics.record("hits_detector", ["detector": .string("hitnet"), "hits": .number(Double(hits.count)),
                                             "frames": .number(Double(track.n))])
            return hits
        } catch {
            // A broken model must not lose the match: the rules still produce hits, and the
            // failure is on the record in stage-metrics.jsonl.
            metrics.record("hits_detector", ["detector": .string("hitnet"), "outcome": .string("error"),
                                             "error": .string(String(describing: error))])
            return timed("hits") { PortedStages.hits(track: track, meta: meta, camera: camera, players: players) }
        }
    }

    /// Installs the bundled learned hit detector (idempotent). `LiveAnalyzer` calls it when it
    /// builds the stages; without it the hit stage falls back to the rules.
    @discardableResult
    public static func loadHitModel(in bundle: Bundle = .main, cacheDirectory: URL? = nil) async -> Bool {
        await HitStage.Registry.loadBundled(in: bundle, cacheDirectory: cacheDirectory)
    }

    public func reconstruct(track: Track2D, hits: [HitEvent], camera: CameraModel, meta: VideoMeta,
                            players: [PlayerObservation]?) throws -> ReconstructionResult {
        try Self.requireConsistent(track)
        return timed("reconstruct3d") { PortedStages.reconstruct(track: track, hits: hits, camera: camera, meta: meta, players: players) }
    }

    /// The stage packages index every per-frame array with one frame number and trap on a
    /// mismatch; a track built in memory (not decoded) gets the same check the decoder does.
    static func requireConsistent(_ track: Track2D) throws {
        if let why = track.inconsistency { throw LiveStageError.inconsistentTrack(why) }
    }

    public func segmentRallies(track: Track2D, hits: [HitEvent], reconstruction: ReconstructionResult, meta: VideoMeta,
                               camera: CameraModel?, players: [PlayerObservation]?) throws -> [RallyV2] {
        try Self.requireConsistent(track)
        return timed("rallies") {
            PortedStages.rallies(track: track, hits: hits, reconstruction: reconstruction, meta: meta,
                                 camera: camera, players: players)
        }
    }

    public func classifyShots(_ rally: RallyV2, camera: CameraModel?, players: [PlayerObservation]?) throws -> RallyV2 {
        try classifyShots(rally, camera: camera, players: players, meta: nil, track: nil)
    }

    /// The learned stroke classifier is in front of the rule-based port here: it needs the meta
    /// and the linked track to build its clips, and the pipeline passes both.
    public func classifyShots(_ rally: RallyV2, camera: CameraModel?, players: [PlayerObservation]?,
                              meta: VideoMeta?, track: Track2D?) throws -> RallyV2 {
        let r = timed("shots") { PortedStages.shots(rally, camera: camera, players: players, meta: meta, track: track) }
        if let d = StrokeClassifier.lastDiagnostics {
            metrics.record("shots_classifier", d.fields)
        }
        return r
    }

    public func scoreMatch(_ rallies: [RallyV2]) throws -> [JSONValue] {
        timed("score") { PortedStages.score(rallies) }
    }

    /// Doubles service order needs to know it's doubles (pipeline_v2: detect_doubles(players)).
    public func scoreMatch(_ rallies: [RallyV2], players: [PlayerObservation]?) throws -> [JSONValue] {
        timed("score") { PortedStages.score(rallies, players: players) }
    }

    /// Pipeline events (detect_partial_load / detect_partial_save / …) go to stage-metrics.jsonl.
    public func recordEvent(_ name: String, _ fields: [String: JSONValue]) {
        metrics.record(name, fields)
    }

    /// Synchronous stages: same record as `metrics.measure`, without an async hop.
    private func timed<T>(_ name: String, _ body: () throws -> T) rethrows -> T {
        guard metrics.url != nil else { return try body() }
        let start = Snapshot.now()
        do {
            let v = try body()
            metrics.record("stage", ["stage": .string(name), "outcome": .string("ok"),
                                     "seconds": .number(ProcessInfo.processInfo.systemUptime - start.uptime),
                                     "footprint_peak_mb": .number(Snapshot.memory().1)])
            return v
        } catch {
            metrics.record("stage", ["stage": .string(name), "outcome": .string("error"),
                                     "error": .string(String(describing: error))])
            throw error
        }
    }
}

/// court_calibration.calibrate_from_frames (nil = no court evidence good enough).
enum CourtStage {
    static func automatic(_ frames: [ImageU8], _ size: ImageSize, _ doublesHint: Bool?) throws -> CameraModel? {
        automaticDetailed(frames, size, doublesHint).camera
    }

    /// Camera plus the evidence, as the same flat JSON dict Python's
    /// calibrate_from_frames `diagnostics` gives (keys, and which keys are absent).
    static func automaticDetailed(_ frames: [ImageU8], _ size: ImageSize, _ doublesHint: Bool?, focalHintPx: Double? = nil)
        -> (camera: CameraModel?, diagnostics: [String: JSONValue]) {
        let r = CourtCalibration.calibrateFromFramesDetailed(frames, size: size, doublesHint: doublesHint, focalHintPx: focalHintPx)
        return (r.camera, diagnosticsJSON(r.diagnostics, frames: frames))
    }

    /// `reason` is always present: "ok" when accepted, else one of court_calibration.REASONS
    /// (no_frames, no_line_families, no_hypotheses, no_candidates, evidence_gate,
    /// focal_mismatch, low_confidence, ambiguous, low_coverage, implausible_focal, high_rmse).
    static func diagnosticsJSON(_ d: CourtCalibrationDiagnostics, frames: [ImageU8]) -> [String: JSONValue] {
        var out: [String: JSONValue] = ["source": .string(d.source), "reason": .string(d.reason.rawValue)]
        // single-frame fallback (Python: n_frame_accepts + frames_tried whenever it ran;
        // frames_agree + frame_agree_px when two frames agreed)
        if let n = d.nFrameAccepts {
            out["n_frame_accepts"] = .number(Double(n))
            out["frames_tried"] = .array(d.framesTried.map { .number(Double($0)) })
        }
        if let px = d.frameAgreePx {
            out["frame_agree_px"] = .number(px)
            out["frames_agree"] = .array(d.framesAgree.map { .number(Double($0)) })
        }
        // Python only fills these once lines were found (a blank frame gives just "source").
        guard d.nLines > 0 || d.nHyp > 0 || d.stats != nil else { return out }
        out["n_hyp"] = .number(Double(d.nHyp))
        out["n_lines"] = .number(Double(d.nLines))
        out["n_valid"] = .number(Double(d.nValid))
        if let s = d.stats {
            out["coverage"] = .number(s.coverage)
            out["rmse"] = .number(s.rmse)
            out["n_t"] = .number(Double(s.nT))
            out["n_l"] = .number(Double(s.nL))
            out["n_vis"] = .number(Double(s.nVis))
            out["n_inl"] = .number(Double(s.nInl))
            out["n_absent"] = .number(Double(s.nAbsent))
            out["absent_frac"] = .number(s.absentFrac)
            if let e = s.explained { out["explained"] = .number(e) }
            if let e = s.explainedFrac { out["explained_frac"] = .number(e) }
            out["t_straddle"] = .bool(s.tStraddle)
        }
        for (k, v) in [("margin", d.margin), ("confidence", d.confidence), ("score", d.score), ("score2", d.score2),
                       ("far_m_per_px", d.farMPerPx), ("dominated_by", d.dominatedBy)] {
            if let v { out[k] = .number(v) }
        }
        if let c = d.lineColour { out["line_colour"] = .string(c.rawValue) }     // non-white paint model
        if !d.paintFit.isEmpty {                                                  // Python "paint_fit"
            out["paint_fit"] = .object(Dictionary(uniqueKeysWithValues: d.paintFit.map { ($0.key.rawValue, JSONValue.number($0.value)) }))
        }
        return out
    }
}

public enum LiveStageError: Error, CustomStringConvertible {
    /// Per-frame track arrays of different lengths (a damaged or mis-built track).
    case inconsistentTrack(String)

    public var description: String {
        switch self {
        case .inconsistentTrack(let why): return why
        }
    }
}

/// Why the last players resume was not used (nil = it was).
private final class ProblemBox: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: String?
    var value: String? {
        get { lock.withLock { stored } }
        set { lock.withLock { stored = newValue } }
    }
}
