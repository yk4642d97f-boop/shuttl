import XCTest
import ShuttlCore
@testable import ShuttlAnalysis

/// A detector that behaves like HeatmapDetector's resumable pass: validates `resume`
/// itself (ignores a bad one), checkpoints every `every` frames, can be "killed".
final class ResumableStages: AnalysisStages, @unchecked Sendable {
    private let lock = NSLock()
    private var _detected: [Int] = []
    private var _offered: [DetectCheckpoint?] = []
    private var _linkInput: [[ShuttleCandidate]]?
    var detectedFrames: [Int] { lock.withLock { _detected } }
    var offered: [DetectCheckpoint?] { lock.withLock { _offered } }
    var linkInput: [[ShuttleCandidate]]? { lock.withLock { _linkInput } }

    private var _laterCalls: [String] = []
    private var _timeline: [String] = []
    private var _events: [(String, [String: JSONValue])] = []
    /// Stages after detect that actually ran, in order.
    var laterCalls: [String] { lock.withLock { _laterCalls } }
    /// Every stage call including detect, in order, across all runs of this instance.
    var timeline: [String] { lock.withLock { _timeline } }
    /// `recordEvent` calls (what LiveStages writes to stage-metrics.jsonl).
    var events: [(name: String, fields: [String: JSONValue])] { lock.withLock { _events.map { ($0.0, $0.1) } } }
    private func later(_ s: String) { lock.withLock { _laterCalls.append(s); _timeline.append(s) } }

    let name: String
    let every: Int
    var cancelAtFrame: Int?
    /// Throws `error` when detect reaches `frame` (after the checkpoints before it were handed out).
    var failAt: (frame: Int, error: Error)?
    var failHits = false

    func recordEvent(_ name: String, _ fields: [String: JSONValue]) { lock.withLock { _events.append((name, fields)) } }
    static let background = Data((0..<600).map { UInt8($0 % 251) })

    init(name: String = "fakenet", every: Int = 8) { self.name = name; self.every = every }

    var detectorName: String { name }

    static func candidates(_ f: Int) -> [ShuttleCandidate] {
        f % 5 == 3 ? [] : (0..<(f % 3 + 1)).map { ShuttleCandidate(x: Double(f) * 1.5 + Double($0), y: 7 / Double(f + 1), score: 0.1 * Double($0 + 1)) }
    }

    func detect(_ source: VideoSource, progress: @escaping StageProgress) async throws -> [[ShuttleCandidate]] {
        try await detect(source, resume: nil, checkpoint: { _ in }, progress: progress)
    }

    func detect(_ source: VideoSource, resume: DetectCheckpoint?, checkpoint: @escaping @Sendable (DetectCheckpoint) -> Void,
                progress: @escaping StageProgress) async throws -> [[ShuttleCandidate]] {
        lock.withLock { _offered.append(resume); _timeline.append("detect") }
        let n = source.meta.nFrames
        var out: [[ShuttleCandidate]] = [], start = 0
        if let r = resume, r.detectorName == name, r.nextFrame > 0, r.nextFrame % every == 0,
           r.candidates.count == r.nextFrame, r.nextFrame < n, r.background == Self.background {
            out = r.candidates; start = r.nextFrame
        }
        for f in start..<n {
            if f == cancelAtFrame { throw CancellationError() }
            if let failAt, f == failAt.frame { throw failAt.error }
            lock.withLock { _detected.append(f) }
            out.append(Self.candidates(f))
            if (f + 1) % every == 0, f + 1 < n {
                checkpoint(DetectCheckpoint(nextFrame: f + 1, candidates: out, background: Self.background, detectorName: name))
            }
        }
        return out
    }

    func link(_ candidates: [[ShuttleCandidate]], meta: VideoMeta) throws -> Track2D {
        later("link")
        lock.withLock { _linkInput = candidates }
        let n = candidates.count
        return Track2D(t: meta.timestamps, xy: candidates.map { SIMD2($0.first?.x ?? .nan, $0.first?.y ?? .nan) },
                       conf: Array(repeating: 0.5, count: n), visible: candidates.map { !$0.isEmpty },
                       interpolated: Array(repeating: false, count: n))
    }

    func calibrationFrames(_ source: VideoSource, count: Int) throws -> [ImageU8] { later("calibration_frames"); return [] }
    func calibrateAutomatic(frames: [ImageU8], size: ImageSize, doublesHint: Bool?) throws -> CameraModel? {
        later("calibrate_auto"); return nil
    }
    func calibrateFromPoints(_ points: [String: SIMD2<Double>], size: ImageSize) throws -> CameraModel { FakeStages.camera }
    func refineCamera(_ camera: CameraModel, frames: [ImageU8]) throws -> CameraModel { camera }
    func trackPlayers(_ source: VideoSource, camera: CameraModel?, step: Int, doubles: Bool?,
                      progress: @escaping StageProgress) async throws -> [PlayerObservation] { later("players"); return [] }

    // Resumable players (P2-2): one observation per sampled frame, a checkpoint every
    // `playersEvery` sampled frames; the resume is validated like PlayerTracker does.
    var playersEvery = 4
    var playersFailAt: (frame: Int, error: Error)?
    var playersCancelAtFrame: Int?
    private var _playersOffered: [PlayersCheckpoint?] = []
    private var _playersSampled: [Int] = []
    var playersOffered: [PlayersCheckpoint?] { lock.withLock { _playersOffered } }
    var playersSampled: [Int] { lock.withLock { _playersSampled } }

    static func playersContext(_ step: Int) -> String { "fake-players-step\(step)" }
    static func playerRecord(_ f: Int) -> Data {
        var d = Data()
        withUnsafeBytes(of: Int64(f).littleEndian) { d.append(contentsOf: $0) }
        d.append(contentsOf: (0..<(f % 7)).map { UInt8(($0 * 37 + f) % 256) })
        return d
    }
    static func observation(_ f: Int) -> PlayerObservation {
        PlayerObservation(frame: f, trackID: 1 + f % 2, bbox: BBox(x1: Double(f), y1: 1 / Double(f + 3), x2: Double(f) + 10, y2: 40),
                          score: 0.5 + 0.01 * Double(f % 9), side: f % 2 == 0 ? .near : .far)
    }

    func trackPlayers(_ source: VideoSource, camera: CameraModel?, step: Int, doubles: Bool?,
                      resume: PlayersCheckpoint?, checkpoint: @escaping @Sendable (PlayersCheckpoint) -> Void,
                      progress: @escaping StageProgress) async throws -> [PlayerObservation] {
        later("players")
        lock.withLock { _playersOffered.append(resume) }
        let n = source.meta.nFrames, ctx = Self.playersContext(step)
        var records: [Data] = [], start = 0
        if let r = resume, r.context == ctx, r.nextFrame > 0, r.nextFrame % step == 0, r.nextFrame < n,
           r.frames.count == r.nextFrame / step,
           r.frames.enumerated().allSatisfy({ $0.element == Self.playerRecord($0.offset * step) }) {
            records = r.frames; start = r.nextFrame
        }
        for f in stride(from: start, to: n, by: step) {
            if f == playersCancelAtFrame { throw CancellationError() }
            if let playersFailAt, f == playersFailAt.frame { throw playersFailAt.error }
            lock.withLock { _playersSampled.append(f) }
            records.append(Self.playerRecord(f))
            if records.count % playersEvery == 0, f + step < n {
                checkpoint(PlayersCheckpoint(nextFrame: f + step, context: ctx, frames: records))
            }
        }
        return records.indices.map { Self.observation($0 * step) }
    }
    func detectHits(track: Track2D, meta: VideoMeta, camera: CameraModel?, players: [PlayerObservation]?) throws -> [HitEvent] {
        later("hits")
        if failHits { throw DetectorLikeError.logic }
        return []
    }
    func reconstruct(track: Track2D, hits: [HitEvent], camera: CameraModel, meta: VideoMeta,
                     players: [PlayerObservation]?) throws -> ReconstructionResult { .empty }
    func segmentRallies(track: Track2D, hits: [HitEvent], reconstruction: ReconstructionResult, meta: VideoMeta,
                        camera: CameraModel?, players: [PlayerObservation]?) throws -> [RallyV2] { [] }
    func classifyShots(_ rally: RallyV2, camera: CameraModel?, players: [PlayerObservation]?) throws -> RallyV2 { rally }
    func scoreMatch(_ rallies: [RallyV2]) throws -> [JSONValue] { [] }
}
