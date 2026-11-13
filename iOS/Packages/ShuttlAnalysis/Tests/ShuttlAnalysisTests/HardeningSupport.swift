import Darwin
import Foundation
import XCTest
import ShuttlCore
@testable import ShuttlAnalysis

/// Configurable stages for the hardening tests (failures, cancellation, rich players).
final class ScriptedStages: AnalysisStages, @unchecked Sendable {
    private let lock = NSLock()
    private var _calls: [String] = []
    var calls: [String] { lock.withLock { _calls } }
    private func log(_ s: String) { lock.withLock { _calls.append(s) } }

    var detectError: Error?
    var hitsError: Error?
    var cancelInHits = false
    var shotsError: [Int: Error] = [:]
    var makeRallies = true
    /// Replaces the default rallies when set.
    var rallies: [RallyV2]?
    private(set) var ralliesSeenByScore: [RallyV2]?
    var visible = true
    var camera: CameraModel? = FakeStages.camera
    var players: [PlayerObservation] = ScriptedStages.samplePlayers(count: 40)

    var detectorName: String { "scripted" }

    func detect(_ source: VideoSource, progress: @escaping StageProgress) async throws -> [[ShuttleCandidate]] {
        log("detect")
        if let detectError { throw detectError }
        return (0..<source.meta.nFrames).map { [ShuttleCandidate(x: Double($0), y: 10, score: 0.9)] }
    }

    func link(_ candidates: [[ShuttleCandidate]], meta: VideoMeta) throws -> Track2D {
        log("link")
        let n = candidates.count
        return Track2D(t: meta.timestamps, xy: candidates.map { SIMD2($0[0].x, $0[0].y) },
                       conf: Array(repeating: 0.9, count: n), visible: Array(repeating: visible, count: n),
                       interpolated: Array(repeating: false, count: n))
    }

    func calibrationFrames(_ source: VideoSource, count: Int) throws -> [ImageU8] { log("frames"); return [] }
    func calibrateAutomatic(frames: [ImageU8], size: ImageSize, doublesHint: Bool?) throws -> CameraModel? {
        log("auto"); return camera
    }
    func calibrateFromPoints(_ points: [String: SIMD2<Double>], size: ImageSize) throws -> CameraModel {
        log("points"); return FakeStages.camera
    }
    func refineCamera(_ camera: CameraModel, frames: [ImageU8]) throws -> CameraModel { log("refine"); return camera }

    func trackPlayers(_ source: VideoSource, camera: CameraModel?, step: Int, doubles: Bool?,
                      progress: @escaping StageProgress) async throws -> [PlayerObservation] {
        log("players")
        return players
    }

    private(set) var playersSeenByHits: [PlayerObservation]?

    func detectHits(track: Track2D, meta: VideoMeta, camera: CameraModel?, players: [PlayerObservation]?) throws -> [HitEvent] {
        log("hits")
        lock.withLock { playersSeenByHits = players }
        if cancelInHits { throw CancellationError() }
        if let hitsError { throw hitsError }
        return [HitEvent(frame: 2, t: 0.1, xyPx: [2, 10], hitter: .near), HitEvent(frame: 8, t: 0.3, xyPx: [8, 10], hitter: .far)]
    }

    func reconstruct(track: Track2D, hits: [HitEvent], camera: CameraModel, meta: VideoMeta,
                     players: [PlayerObservation]?) throws -> ReconstructionResult {
        log("reconstruct"); return .empty
    }

    func segmentRallies(track: Track2D, hits: [HitEvent], reconstruction: ReconstructionResult, meta: VideoMeta,
                        camera: CameraModel?, players: [PlayerObservation]?) throws -> [RallyV2] {
        log("rallies")
        if let rallies { return rallies }
        guard makeRallies, hits.count >= 2 else { return [] }
        return [RallyV2(startFrame: 0, endFrame: 5, shots: [ShotEventV2(hit: hits[0], segment: nil)]),
                RallyV2(startFrame: 6, endFrame: 11, shots: [ShotEventV2(hit: hits[1], segment: nil)])]
    }

    func classifyShots(_ rally: RallyV2, camera: CameraModel?, players: [PlayerObservation]?) throws -> RallyV2 {
        log("shots:\(rally.startFrame)")
        if let e = shotsError[rally.startFrame] { throw e }
        return rally
    }

    func scoreMatch(_ rallies: [RallyV2]) throws -> [JSONValue] {
        log("score"); lock.withLock { ralliesSeenByScore = rallies }; return []
    }

    /// Observations with every field populated, including NaN edge cases.
    static func samplePlayers(count: Int, keypoints: Bool = true) -> [PlayerObservation] {
        (0..<count).map { i in samplePlayer(i, keypoints: keypoints) }
    }

    static func samplePlayer(_ i: Int, keypoints: Bool) -> PlayerObservation {
        let d = Double(i)
        var kps: [Keypoint]?
        if keypoints && i % 6 != 5 {
            var k: [Keypoint] = []
            k.reserveCapacity(17)
            for j in 0..<17 {
                let dj = Double(j)
                let x: Double = (j == 3 && i % 5 == 0) ? .nan : 100 + d * 0.37 + dj * 1.1
                let y: Double = 200 + d * 0.11 + dj * 2.3
                k.append(Keypoint(x: x, y: y, conf: dj / 17))
            }
            kps = k
        }
        var foot: SIMD2<Double>?
        if i % 4 != 0 {
            let fy: Double = i % 8 == 1 ? .nan : 1.0 / 3.0
            foot = SIMD2(-3.25 + d * 1e-3, fy)
        }
        let sides: [CourtSide] = [.near, .far, .unknown]
        let bbox = BBox(x1: 10 + d * 0.1, y1: 20.25, x2: 60 + d / 3, y2: 190.125)
        let score: Double = 0.3 + Double(i % 7) / 10
        return PlayerObservation(frame: i * 2, trackID: i % 3 + 1, bbox: bbox, score: score,
                                 side: sides[i % 3], footWorld: foot, keypoints: kps)
    }
}

/// Bit-exact comparison (NaN == NaN when the bit patterns match).
func assertBitIdentical(_ a: [PlayerObservation], _ b: [PlayerObservation], file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(a.count, b.count, "count", file: file, line: line)
    for (i, (x, y)) in zip(a, b).enumerated() {
        if !bitsEqual(x, y) { XCTFail("observation \(i) differs: \(x) vs \(y)", file: file, line: line); return }
    }
}

func bitsEqual(_ x: PlayerObservation, _ y: PlayerObservation) -> Bool {
    func b(_ v: Double) -> UInt64 { v.bitPattern }
    guard x.frame == y.frame, x.trackID == y.trackID, x.side == y.side else { return false }
    let bx: [UInt64] = [b(x.bbox.x1), b(x.bbox.y1), b(x.bbox.x2), b(x.bbox.y2), b(x.score)]
    let by: [UInt64] = [b(y.bbox.x1), b(y.bbox.y1), b(y.bbox.x2), b(y.bbox.y2), b(y.score)]
    guard bx == by else { return false }
    let fx: [UInt64]? = x.footWorld.map { [b($0.x), b($0.y)] }
    let fy: [UInt64]? = y.footWorld.map { [b($0.x), b($0.y)] }
    guard fx == fy else { return false }
    let kx: [UInt64]? = x.keypoints.map { $0.flatMap { [b($0.x), b($0.y), b($0.conf)] } }
    let ky: [UInt64]? = y.keypoints.map { $0.flatMap { [b($0.x), b($0.y), b($0.conf)] } }
    return kx == ky
}

/// Peak phys_footprint (what jetsam judges) while `body` runs, sampled every ~1 ms on a
/// background thread and cross-checked against the kernel's lifetime peak.
final class FootprintProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var running = true
    private var peak: UInt64 = 0

    static func current() -> UInt64 {
        var info = rusage_info_v4()
        let rc = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0) }
        }
        return rc == 0 ? info.ri_phys_footprint : 0
    }

    static func lifetimePeak() -> UInt64 {
        var info = rusage_info_v4()
        let rc = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0) }
        }
        return rc == 0 ? info.ri_lifetime_max_phys_footprint : 0
    }

    /// (baseline MB, peak growth MB) for `body`.
    static func measure(_ body: () throws -> Void) rethrows -> (baseline: Double, growth: Double) {
        let probe = FootprintProbe()
        let base = current()
        let lifeBefore = lifetimePeak()
        probe.peak = base
        let done = DispatchSemaphore(value: 0)
        Thread.detachNewThread {
            while probe.lock.withLock({ probe.running }) {
                let now = current()
                probe.lock.withLock { probe.peak = max(probe.peak, now) }
                usleep(1000)
            }
            done.signal()
        }
        try body()
        probe.lock.withLock { probe.running = false }
        done.wait()
        var peak = probe.lock.withLock { probe.peak }
        let lifeAfter = lifetimePeak()
        if lifeAfter > lifeBefore { peak = max(peak, lifeAfter) }   // exact kernel peak when it rose
        let mb = 1024.0 * 1024.0
        return (Double(base) / mb, Double(peak > base ? peak - base : 0) / mb)
    }
}

var repoTestsDirectory: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent() }
