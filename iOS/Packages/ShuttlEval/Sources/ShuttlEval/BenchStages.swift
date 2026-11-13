import Foundation
import ShuttlCore

// Where stage outputs come from. benchmark.py calls each stage function on each
// sample; here a `BenchStages` provider answers those calls — from a recorded Python
// run (`RecordedStages`), from MatchAnalysis files, or from the on-device Swift
// pipeline through `AnalysisRunner` (wired in by the integration step).

/// What a stage call is about (benchmark.py passes the sample name + stage inputs).
public struct StageContext: Sendable {
    public enum Input: Sendable {
        case detector          // real: detector candidates of the clip
        case oracle            // real_oracle: corrupt_px(GT) candidates
        case synthetic         // sim sample: corrupt_detections candidates / truth inputs
    }
    public var sample: String          // benchmark sample name ("seed3", clip "1_05_02")
    public var index: Int              // sample index in the run
    public var input: Input
    public var meta: VideoMeta
    public var camera: CameraModel?
    public var doubles: Bool
    public var video: URL?

    public init(sample: String, index: Int, input: Input, meta: VideoMeta, camera: CameraModel? = nil,
                doubles: Bool = false, video: URL? = nil) {
        self.sample = sample; self.index = index; self.input = input; self.meta = meta
        self.camera = camera; self.doubles = doubles; self.video = video
    }
}

/// Result of `reconstruct3d.call_in_out(seg, doubles, is_serve)["call"]`.
public enum InOutCall: Sendable, Equatable {
    case unavailable          // the function does not exist (benchmark falls back to landing_xy)
    case raised               // it raised (same fallback)
    case call(String)         // its verdict ('in' | 'out' | 'unsure' …)
}

/// One provider per benchmark run. Return nil where Python's stage returned None;
/// throw to record a stage exception (the runner stores it like Pool.error).
/// Stages not in `available` are never called (reported SKIPPED with `missing[stage]`).
public protocol BenchStages: AnyObject {
    /// benchmark.py `resolve_stages()` names: detector, linker, hits, reconstruct3d, rally, shots, calibration.
    var available: Set<String> { get }
    /// stage → reason (benchmark.py `missing`), in Python's order.
    var missing: PyDict { get }
    /// Python stage timings to report instead of Swift's (recorded runs), nil = measure.
    var fixedSeconds: [String: Double]? { get }

    /// real: detector candidates of a clip (benchmark `detect_cached`).
    func detect(_ ctx: StageContext) async throws -> [[ShuttleCandidate]]?
    func linker(_ ctx: StageContext, candidates: [[ShuttleCandidate]]) async throws -> Track2D?
    /// `track` = the linker output (e2e) or the noisy truth track (oracle; nil when the
    /// provider has no sim to build it — recorded providers ignore inputs).
    func hits(_ ctx: StageContext, track: Track2D?) async throws -> [BenchHit]?
    func reconstruct(_ ctx: StageContext, track: Track2D?) async throws -> [BenchSegment]?
    func rallies(_ ctx: StageContext, track: Track2D?) async throws -> [EvalRally]?
    /// Shot classification of rally `rally` (e2e: the predicted rally; oracle: truth
    /// rally with types blanked).
    func shots(_ ctx: StageContext, rally: Int) async throws -> BenchTypedRally?
    /// `call_in_out` on a segment returned by `reconstruct` (by index).
    func callInOut(_ ctx: StageContext, segment: Int, isServe: Bool) -> InOutCall
    /// Pool scalars benchmark.py recorded while detecting this sample (e.g. detector fps).
    func scalars(_ ctx: StageContext) -> [(stage: String, key: String, value: Double)]
}

public extension BenchStages {
    var fixedSeconds: [String: Double]? { nil }
    func scalars(_ ctx: StageContext) -> [(stage: String, key: String, value: Double)] { [] }
}

// MARK: - Recorded Python run

/// Replays a bench bundle: each stage call pops the next recorded call of that stage
/// for the sample (Python's per-sample call order), rethrowing recorded exceptions.
public final class RecordedBenchStages: BenchStages {
    public let bundle: BenchBundle
    public let available: Set<String>
    public let missing: PyDict
    public let fixedSeconds: [String: Double]?
    private var cursor: [String: Int] = [:]
    /// Optional replacement for the detector + detector-linker outputs (e.g. MatchAnalysis
    /// files); the recorded entries are still consumed so the queues stay aligned.
    public var detectorOverride: (any BenchStages)?

    public init(_ bundle: BenchBundle) {
        self.bundle = bundle
        available = Set(bundle.available)
        missing = bundle.missing
        fixedSeconds = bundle.poolSeconds
    }

    func next<T>(_ ctx: StageContext, _ stage: String, _ list: [RecordedCall<T>], rethrow: Bool = true) throws -> T? {
        let key = "\(ctx.index)/\(stage)"
        let calls = list.filter(\.present)
        let k = cursor[key, default: 0]
        cursor[key] = k + 1
        guard k < calls.count else {
            throw StageError("RecordedBenchStages: no recorded \(stage) call #\(k + 1) for sample \(ctx.sample)")
        }
        if rethrow, let e = calls[k].error { throw RecordedStageError(message: e) }
        return calls[k].out
    }

    func sample(_ ctx: StageContext) -> BenchBundle.Sample { bundle.samples[ctx.index] }

    public func detect(_ ctx: StageContext) async throws -> [[ShuttleCandidate]]? {
        let s = sample(ctx)
        if let o = detectorOverride { return try await o.detect(ctx) }
        if let e = s.detectorError { throw RecordedStageError(message: e) }
        return s.candidates?.frames
    }

    public func linker(_ ctx: StageContext, candidates: [[ShuttleCandidate]]) async throws -> Track2D? {
        let overridden = ctx.input == .detector && detectorOverride != nil
        let rec = try next(ctx, "linker", sample(ctx).stages.linker, rethrow: !overridden)
        if overridden, let o = detectorOverride { return try await o.linker(ctx, candidates: candidates) }
        return rec
    }

    public func hits(_ ctx: StageContext, track: Track2D?) async throws -> [BenchHit]? {
        try next(ctx, "hits", sample(ctx).stages.hits)
    }

    public func reconstruct(_ ctx: StageContext, track: Track2D?) async throws -> [BenchSegment]? {
        try next(ctx, "reconstruct3d", sample(ctx).stages.reconstruct3d)
    }

    public func rallies(_ ctx: StageContext, track: Track2D?) async throws -> [EvalRally]? {
        try next(ctx, "rally", sample(ctx).stages.rally)
    }

    public func shots(_ ctx: StageContext, rally: Int) async throws -> BenchTypedRally? {
        try next(ctx, "shots", sample(ctx).stages.shots)
    }

    public func callInOut(_ ctx: StageContext, segment: Int, isServe: Bool) -> InOutCall {
        guard bundle.hasCallInOut else { return .unavailable }
        let calls = sample(ctx).stages.reconstruct3d.filter(\.present)
        // the oracle chain calls reconstruct3d once per sample: its (only) output
        guard let segs = calls.first?.out, segment < segs.count, segs[segment].hasCalls else { return .raised }
        let c = isServe ? segs[segment].callServe : segs[segment].callRally
        return c.map { .call($0) } ?? .raised
    }

    public func scalars(_ ctx: StageContext) -> [(stage: String, key: String, value: Double)] { sample(ctx).scalars }
}

/// A recorded Python exception: `message` is the full Pool.error text ("sample: Exc: msg").
public struct RecordedStageError: Error, Sendable, CustomStringConvertible {
    public var message: String
    public var description: String { message }
}

// MARK: - On-device pipeline hook

/// Options for one analysis run.
public struct AnalysisOptions: Sendable {
    /// Synthetic / oracle input: use these per-frame detector candidates instead of
    /// running the detector on a video.
    public var candidates: [[ShuttleCandidate]]?
    /// Video metadata when there is no video (synthetic samples).
    public var meta: VideoMeta?
    /// Known camera (synthetic: the true camera) — skip court calibration.
    public var camera: CameraModel?
    public var players: [PlayerObservation]?
    public var doubles: Bool?
    public var maxFrames: Int?
    public var detector: String?

    public init(candidates: [[ShuttleCandidate]]? = nil, meta: VideoMeta? = nil, camera: CameraModel? = nil,
                players: [PlayerObservation]? = nil, doubles: Bool? = nil, maxFrames: Int? = nil, detector: String? = nil) {
        self.candidates = candidates; self.meta = meta; self.camera = camera; self.players = players
        self.doubles = doubles; self.maxFrames = maxFrames; self.detector = detector
    }
}

/// The Swift analysis pipeline as the benchmark sees it: video (or injected candidates)
/// → MatchAnalysis. Put detector candidates in `diagnostics["candidates"]` (per frame
/// `[[x, y, score], …]`) to have the detector itself scored in `real` mode.
public protocol AnalysisRunner: Sendable {
    var name: String { get }
    func analyze(video: URL?, options: AnalysisOptions) async throws -> MatchAnalysis
}

/// Candidates stored in `MatchAnalysis.diagnostics["candidates"]`.
public func diagnosticCandidates(_ a: MatchAnalysis) -> [[ShuttleCandidate]]? {
    guard let rows = a.diagnostics["candidates"]?.array else { return nil }
    return rows.map { frame in
        (frame.array ?? []).compactMap { c -> ShuttleCandidate? in
            if let v = c.array, v.count >= 3, let x = v[0].double, let y = v[1].double, let s = v[2].double, s.isFinite {
                return ShuttleCandidate(x: x, y: y, score: s)
            }
            if let x = c["x"]?.double, let y = c["y"]?.double, let s = c["score"]?.double, s.isFinite {
                return ShuttleCandidate(x: x, y: y, score: s)
            }
            return nil
        }
    }
}

/// Answers benchmark stage calls from whole-pipeline runs (one `analyze` per sample
/// input, cached). Oracle-mode stage isolation (truth track → hits, truth hits → 3-D, …)
/// needs per-stage entry points, so only the linker is available there.
public final class AnalysisRunnerStages: BenchStages {
    public let runner: any AnalysisRunner
    public let mode: String                           // benchmark --stage
    public let options: AnalysisOptions
    private var cache: [String: MatchAnalysis] = [:]

    public init(runner: any AnalysisRunner, stage: String, options: AnalysisOptions = AnalysisOptions()) {
        self.runner = runner; self.mode = stage; self.options = options
    }

    public var available: Set<String> {
        mode == "oracle" || mode == "real_oracle" ? ["linker"] : ["detector", "linker", "hits", "reconstruct3d", "rally", "shots"]
    }

    public var missing: PyDict {
        var m = PyDict()
        for s in ["detector", "hits", "reconstruct3d", "rally", "shots", "calibration"] where !available.contains(s) {
            m[s] = .str(s == "calibration" ? "court calibration runs inside \(runner.name)"
                                           : "\(runner.name): per-stage entry point not exposed (whole-pipeline runner)")
        }
        return m
    }

    func run(_ ctx: StageContext, candidates: [[ShuttleCandidate]]?) async throws -> MatchAnalysis {
        let key = "\(ctx.index)/\(ctx.input)"
        if let a = cache[key] { return a }
        var o = options
        o.candidates = candidates
        o.camera = ctx.camera ?? o.camera
        o.doubles = ctx.doubles
        if ctx.video == nil || candidates != nil { o.meta = ctx.meta }
        let a = try await runner.analyze(video: candidates == nil ? ctx.video : nil, options: o)
        cache[key] = a
        return a
    }

    func cached(_ ctx: StageContext) -> MatchAnalysis? { cache["\(ctx.index)/\(ctx.input)"] }

    public func detect(_ ctx: StageContext) async throws -> [[ShuttleCandidate]]? {
        diagnosticCandidates(try await run(ctx, candidates: nil))
    }

    public func linker(_ ctx: StageContext, candidates: [[ShuttleCandidate]]) async throws -> Track2D? {
        if ctx.input == .detector, let a = cached(ctx) { return a.track }
        return try await run(ctx, candidates: candidates).track
    }

    public func hits(_ ctx: StageContext, track: Track2D?) async throws -> [BenchHit]? {
        cached(ctx).map { $0.hits.map { BenchHit(frame: $0.frame, hitter: $0.hitter.rawValue) } }
    }

    public func reconstruct(_ ctx: StageContext, track: Track2D?) async throws -> [BenchSegment]? {
        cached(ctx).map { $0.segments.map { BenchSegment(segment: EvalSegment($0)) } }
    }

    public func rallies(_ ctx: StageContext, track: Track2D?) async throws -> [EvalRally]? {
        cached(ctx).map { $0.rallies.map(EvalRally.init) }
    }

    public func shots(_ ctx: StageContext, rally: Int) async throws -> BenchTypedRally? {
        guard let a = cached(ctx), rally < a.rallies.count else { return nil }
        return BenchTypedRally(a.rallies[rally])
    }

    public func callInOut(_ ctx: StageContext, segment: Int, isServe: Bool) -> InOutCall { .unavailable }
}

/// `--analysis-dir`: MatchAnalysis JSON files `<match>_<clip>.json` (+ optional
/// `<match>_<clip>.candidates.json` = `{"candidates": [[[x, y, score], …], …]}`) as an
/// AnalysisRunner — the path a Swift-produced analysis takes into the benchmark.
public struct MatchAnalysisFileRunner: AnalysisRunner {
    public let directory: URL
    public var name: String { "analysis-dir \(directory.lastPathComponent)" }
    public init(directory: URL) { self.directory = directory }

    public func analyze(video: URL?, options: AnalysisOptions) async throws -> MatchAnalysis {
        guard let video else { throw StageError("StageError: MatchAnalysisFileRunner needs a real clip") }
        let stem = "\(video.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent)_\(video.deletingPathExtension().lastPathComponent)"
        let url = directory.appendingPathComponent("\(stem).json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw StageError("FileNotFoundError: \(url.path)")
        }
        var a = try ShuttlJSON.load(MatchAnalysis.self, from: url)
        let cu = directory.appendingPathComponent("\(stem).candidates.json")
        if a.diagnostics["candidates"] == nil, let data = try? Data(contentsOf: cu),
           let v = try? ShuttlJSON.decoder().decode([String: JSONValue].self, from: data), let c = v["candidates"] {
            a.diagnostics["candidates"] = c
        }
        return a
    }
}
