import CryptoKit
import Foundation
import ShuttlCore

public typealias StageProgress = @Sendable (Double) -> Void

/// 3-D stage output. Python's reconstruct3d attaches `landing_sigma_m` to each
/// segment object and rally.py reads it for close in/out calls; ShotSegment3D
/// doesn't carry it, so it travels alongside (same order, nil = unknown).
public struct ReconstructionResult: Sendable, Codable {
    public var segments: [ShotSegment3D]
    public var landingSigmaM: [Double?]

    public init(segments: [ShotSegment3D], landingSigmaM: [Double?]? = nil) {
        self.segments = segments
        self.landingSigmaM = landingSigmaM ?? Array(repeating: nil, count: segments.count)
        precondition(self.landingSigmaM.count == segments.count, "one sigma per segment")
    }

    public static let empty = ReconstructionResult(segments: [])

    enum CodingKeys: String, CodingKey { case segments, landingSigmaM = "landing_sigma_m" }
}

/// The analysis stages, one method per Python module entry point. The live
/// implementation forwards to the ShuttlDetector / ShuttlLinker / ShuttlCourt /
/// ShuttlPlayers / ShuttlHits / ShuttlReconstruct / ShuttlRally packages; tests
/// substitute fakes. Every method may throw — the pipeline isolates failures per
/// stage exactly like pipeline_v2._Stages (except cancellation, which propagates).
public protocol AnalysisStages: Sendable {
    /// Short name recorded as diagnostics["detector"].
    var detectorName: String { get }

    /// Per-frame shuttle candidates in full-res upright px. `progress` goes 0 → 1.
    func detect(_ source: VideoSource, progress: @escaping StageProgress) async throws -> [[ShuttleCandidate]]
    /// Resumable detect: continue from `resume` (the detector validates it and ignores a
    /// bad one) and hand a `DetectCheckpoint` to `checkpoint` now and then. The default
    /// implementation ignores both and calls `detect(_:progress:)`.
    func detect(_ source: VideoSource, resume: DetectCheckpoint?,
                checkpoint: @escaping @Sendable (DetectCheckpoint) -> Void,
                progress: @escaping StageProgress) async throws -> [[ShuttleCandidate]]
    func link(_ candidates: [[ShuttleCandidate]], meta: VideoMeta) throws -> Track2D

    /// Frames for calibration, in the image type the calibrator expects.
    func calibrationFrames(_ source: VideoSource, count: Int) throws -> [ImageU8]
    func calibrateAutomatic(frames: [ImageU8], size: ImageSize, doublesHint: Bool?) throws -> CameraModel?
    /// `calibrateAutomatic` with its diagnostics. The default wraps `calibrateAutomatic`
    /// with none; the signature must match exactly for a conformer's version to be used.
    func calibrateAutomaticDetailed(frames: [ImageU8], size: ImageSize, doublesHint: Bool?) throws -> CalibrationAttempt
    /// Same, with the lens's real focal length (px at `size`) as a veto on implausible
    /// cameras. The default ignores the hint and calls the 3-argument variant.
    func calibrateAutomaticDetailed(frames: [ImageU8], size: ImageSize, doublesHint: Bool?,
                                    focalHintPx: Double?) throws -> CalibrationAttempt
    func calibrateFromPoints(_ points: [String: SIMD2<Double>], size: ImageSize) throws -> CameraModel
    func refineCamera(_ camera: CameraModel, frames: [ImageU8]) throws -> CameraModel

    func trackPlayers(_ source: VideoSource, camera: CameraModel?, step: Int, doubles: Bool?,
                      progress: @escaping StageProgress) async throws -> [PlayerObservation]
    /// Resumable players: continue from `resume` (the stage validates it and ignores a bad one)
    /// and hand a `PlayersCheckpoint` to `checkpoint` now and then. The default ignores both and
    /// calls the 5-argument variant; the signature must match exactly for a conformer's version
    /// to be used.
    func trackPlayers(_ source: VideoSource, camera: CameraModel?, step: Int, doubles: Bool?,
                      resume: PlayersCheckpoint?, checkpoint: @escaping @Sendable (PlayersCheckpoint) -> Void,
                      progress: @escaping StageProgress) async throws -> [PlayerObservation]
    func detectHits(track: Track2D, meta: VideoMeta, camera: CameraModel?, players: [PlayerObservation]?) throws -> [HitEvent]
    func reconstruct(track: Track2D, hits: [HitEvent], camera: CameraModel, meta: VideoMeta,
                     players: [PlayerObservation]?) throws -> ReconstructionResult
    func segmentRallies(track: Track2D, hits: [HitEvent], reconstruction: ReconstructionResult, meta: VideoMeta,
                        camera: CameraModel?, players: [PlayerObservation]?) throws -> [RallyV2]
    func classifyShots(_ rally: RallyV2, camera: CameraModel?, players: [PlayerObservation]?) throws -> RallyV2
    /// Same, with the stages the hits came from. The learned stroke classifier needs the video's
    /// meta (frame rate, size) and the linked shuttle track to build its clips, and cannot work
    /// without them; `meta`/`track` nil means the rule-based port only. The default drops both and
    /// calls the 3-argument variant; the signature must match exactly for a conformer's version
    /// to be used.
    func classifyShots(_ rally: RallyV2, camera: CameraModel?, players: [PlayerObservation]?,
                       meta: VideoMeta?, track: Track2D?) throws -> RallyV2
    func scoreMatch(_ rallies: [RallyV2]) throws -> [JSONValue]
    /// Score with the players, so doubles service order (server / receiver / positions) can be
    /// followed. The default ignores them and calls `scoreMatch(_:)` (singles behaviour).
    func scoreMatch(_ rallies: [RallyV2], players: [PlayerObservation]?) throws -> [JSONValue]
    /// A pipeline event for the device log (detect_partial load / save, checkpoint clears).
    /// The default drops it; the signature must match exactly for a conformer's version to be used.
    func recordEvent(_ name: String, _ fields: [String: JSONValue])
}

extension AnalysisStages {
    public func recordEvent(_ name: String, _ fields: [String: JSONValue]) {}

    public func detect(_ source: VideoSource, resume: DetectCheckpoint?,
                       checkpoint: @escaping @Sendable (DetectCheckpoint) -> Void,
                       progress: @escaping StageProgress) async throws -> [[ShuttleCandidate]] {
        try await detect(source, progress: progress)
    }

    public func scoreMatch(_ rallies: [RallyV2], players: [PlayerObservation]?) throws -> [JSONValue] {
        try scoreMatch(rallies)
    }

    public func classifyShots(_ rally: RallyV2, camera: CameraModel?, players: [PlayerObservation]?,
                              meta: VideoMeta?, track: Track2D?) throws -> RallyV2 {
        try classifyShots(rally, camera: camera, players: players)
    }

    public func trackPlayers(_ source: VideoSource, camera: CameraModel?, step: Int, doubles: Bool?,
                             resume: PlayersCheckpoint?, checkpoint: @escaping @Sendable (PlayersCheckpoint) -> Void,
                             progress: @escaping StageProgress) async throws -> [PlayerObservation] {
        try await trackPlayers(source, camera: camera, step: step, doubles: doubles, progress: progress)
    }

    public func calibrateAutomaticDetailed(frames: [ImageU8], size: ImageSize, doublesHint: Bool?) throws -> CalibrationAttempt {
        CalibrationAttempt(camera: try calibrateAutomatic(frames: frames, size: size, doublesHint: doublesHint))
    }

    public func calibrateAutomaticDetailed(frames: [ImageU8], size: ImageSize, doublesHint: Bool?,
                                           focalHintPx: Double?) throws -> CalibrationAttempt {
        try calibrateAutomaticDetailed(frames: frames, size: size, doublesHint: doublesHint)
    }
}

/// Automatic calibration plus the evidence behind its verdict (lines matched, coverage,
/// fit error, margin …), kept in the analysis so the app can say why a court was not found.
public struct CalibrationAttempt: Sendable {
    public var camera: CameraModel?
    /// Calibrator-defined keys (ShuttlCourt: n_t, n_l, n_absent, coverage, rmse, margin, …).
    public var diagnostics: [String: JSONValue]

    public init(camera: CameraModel?, diagnostics: [String: JSONValue] = [:]) {
        self.camera = camera
        self.diagnostics = diagnostics
    }
}

/// Progress of a detect pass that can continue after the app was killed. Same fields as
/// ShuttlDetector's `DetectResume` (map 1:1; ShuttlAnalysis doesn't depend on it).
public struct DetectCheckpoint: Codable, Sendable, Equatable {
    /// Frames `< nextFrame` are final.
    public var nextFrame: Int
    /// Final candidates of frames `0..<nextFrame`.
    public var candidates: [[ShuttleCandidate]]
    /// Detector-private state (e.g. the serialized background plan), nil if none.
    public var background: Data?
    /// Resume only with the same model.
    public var detectorName: String

    public init(nextFrame: Int, candidates: [[ShuttleCandidate]], background: Data?, detectorName: String) {
        self.nextFrame = nextFrame; self.candidates = candidates
        self.background = background; self.detectorName = detectorName
    }
}

/// Persists expensive intermediate results so an analysis the OS interrupted
/// (app backgrounded / killed) resumes instead of starting over. Keys are stage names.
public protocol CheckpointStore: Sendable {
    func load<T: Decodable>(_ type: T.Type, key: String) -> T?
    func save<T: Encodable>(_ value: T, key: String)
    func clear()
}

/// Directory-backed checkpoints: one file per key, invalidated when the video changes.
/// `[PlayerObservation]` goes to `<key>.bin` via `PlayersCodec` (streamed, compact —
/// JSON-encoding a long match's poses would need gigabytes); everything else is `<key>.json`.
public struct FileCheckpointStore: CheckpointStore {
    public let directory: URL
    let fingerprint: String

    /// `fingerprint` identifies the input AND the analysis code (e.g. file size + frame
    /// count + app build); checkpoints written for a different fingerprint are ignored.
    /// See `checkpointFingerprint(videoSize:modified:build:options:)`.
    public init(directory: URL, fingerprint: String) {
        self.directory = directory
        self.fingerprint = fingerprint
    }

    private func url(_ key: String) -> URL { directory.appendingPathComponent("\(key).json") }
    private func binaryURL(_ key: String) -> URL { directory.appendingPathComponent("\(key).bin") }

    /// A checkpoint that decodes but whose bytes changed (disk damage, a partial write, an
    /// edit) must be a cache miss: every stage after it would silently compute on wrong input.
    public func load<T: Decodable>(_ type: T.Type, key: String) -> T? {
        if T.self == [PlayerObservation].self, FileManager.default.fileExists(atPath: binaryURL(key).path) {
            guard let obs = try? PlayersCodec.read(from: binaryURL(key), expectedFingerprint: fingerprint,
                                                   requireChecksum: true) else { return nil }
            return obs as? T
        }
        guard let data = try? Data(contentsOf: url(key), options: .mappedIfSafe),
              let value = CheckpointFile.verifiedValue(in: data, fingerprint: fingerprint) else { return nil }
        return try? ShuttlJSON.decoder().decode(T.self, from: value)
    }

    public func save<T: Encodable>(_ value: T, key: String) {
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            if T.self == [PlayerObservation].self, let obs = value as? [PlayerObservation] {
                try PlayersCodec.write(obs, to: binaryURL(key), fingerprint: fingerprint)
                try? FileManager.default.removeItem(at: url(key))       // a stale JSON copy
                return
            }
            let data = try CheckpointFile.encode(value, fingerprint: fingerprint)
            try data.write(to: url(key), options: .atomic)
        } catch {
            // Checkpoints are an optimisation; never fail the analysis over one.
        }
    }

    public func clear() { try? FileManager.default.removeItem(at: directory) }
}

/// JSON checkpoint layout: `{"fingerprint":<fp>,"sha256":"<64 hex>","value":<value>}`, written
/// byte-for-byte in that order so the digest covers the exact stored value bytes (re-encoding a
/// decoded value is not guaranteed to reproduce them). Files without the digest are misses.
enum CheckpointFile {
    static func encode<T: Encodable>(_ value: T, fingerprint: String) throws -> Data {
        let body = try ShuttlJSON.encoder().encode(value)
        let head = try prefix(fingerprint) + Data(hex(SHA256.hash(data: body)).utf8) + Data(#"","value":"#.utf8)
        var out = Data(capacity: head.count + body.count + 1)
        out.append(head); out.append(body); out.append(UInt8(ascii: "}"))
        return out
    }

    /// The value bytes, if `data` is a checkpoint for `fingerprint` whose digest matches.
    static func verifiedValue(in data: Data, fingerprint: String) -> Data? {
        guard let head = try? prefix(fingerprint) else { return nil }
        let marker = Data(#"","value":"#.utf8)
        let hexStart = data.startIndex + head.count, valueStart = hexStart + 64 + marker.count
        guard data.count > head.count + 64 + marker.count, data.last == UInt8(ascii: "}"),
              data.prefix(head.count) == head,
              data[(hexStart + 64)..<valueStart] == marker else { return nil }
        let value = data[valueStart..<(data.endIndex - 1)]
        guard String(decoding: data[hexStart..<(hexStart + 64)], as: UTF8.self) == hex(SHA256.hash(data: value)) else {
            return nil
        }
        return value
    }

    private static func prefix(_ fingerprint: String) throws -> Data {
        Data(#"{"fingerprint":"#.utf8) + (try ShuttlJSON.encoder().encode(fingerprint)) + Data(#","sha256":""#.utf8)
    }

    private static func hex(_ digest: SHA256.Digest) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }
}

/// No checkpoints (tests, one-shot runs).
public struct NoCheckpoints: CheckpointStore {
    public init() {}
    public func load<T: Decodable>(_ type: T.Type, key: String) -> T? { nil }
    public func save<T: Encodable>(_ value: T, key: String) {}
    public func clear() {}
}
