import AVFoundation
import CoreVideo
import XCTest
import ShuttlCore
@testable import ShuttlAnalysis

/// Records calls and injects failures; stands in for the real stage packages.
final class FakeStages: AnalysisStages, @unchecked Sendable {
    enum Failure: Error { case boom }

    private let lock = NSLock()
    private var _calls: [String] = []
    var calls: [String] { lock.withLock { _calls } }
    private func log(_ s: String) { lock.withLock { _calls.append(s) } }

    var failDetect = false
    var cancelInPlayers = false
    var autoCamera: CameraModel? = FakeStages.camera
    var failRefine = false
    var failShotsForRally: Int?
    var lastPoints: [String: SIMD2<Double>]?

    static let camera = CameraModel.lookAt(eye: [-11, 0, 3], target: [0, 0, 0], focalPx: 1000,
                                           imageSize: ImageSize(width: 64, height: 48))

    var detectorName: String { "fake" }

    func detect(_ source: VideoSource, progress: @escaping StageProgress) async throws -> [[ShuttleCandidate]] {
        log("detect")
        if failDetect { throw Failure.boom }
        for i in 0...4 { progress(Double(i) / 4) }
        return (0..<source.meta.nFrames).map { [ShuttleCandidate(x: Double($0), y: 10, score: 0.9)] }
    }

    func link(_ candidates: [[ShuttleCandidate]], meta: VideoMeta) throws -> Track2D {
        log("link")
        let n = candidates.count
        return Track2D(t: meta.timestamps, xy: candidates.map { SIMD2($0[0].x, $0[0].y) },
                       conf: Array(repeating: 0.9, count: n), visible: Array(repeating: true, count: n),
                       interpolated: Array(repeating: false, count: n))
    }

    func calibrationFrames(_ source: VideoSource, count: Int) throws -> [ImageU8] {
        log("frames")
        return [ImageU8(width: 4, height: 4)]
    }

    func calibrateAutomatic(frames: [ImageU8], size: ImageSize, doublesHint: Bool?) throws -> CameraModel? {
        log("auto"); return autoCamera
    }

    func calibrateFromPoints(_ points: [String: SIMD2<Double>], size: ImageSize) throws -> CameraModel {
        log("points"); lock.withLock { lastPoints = points }
        var c = FakeStages.camera; c.confidence = 0.5; return c
    }

    func refineCamera(_ camera: CameraModel, frames: [ImageU8]) throws -> CameraModel {
        log("refine")
        if failRefine { throw Failure.boom }
        var c = camera; c.confidence = 0.9; return c
    }

    func trackPlayers(_ source: VideoSource, camera: CameraModel?, step: Int, doubles: Bool?,
                      progress: @escaping StageProgress) async throws -> [PlayerObservation] {
        log("players:\(step)")
        if cancelInPlayers { throw CancellationError() }
        progress(0.5); progress(1)
        return [PlayerObservation(frame: 0, trackID: 1, bbox: BBox(x1: 0, y1: 0, x2: 1, y2: 2), score: 1, side: .near)]
    }

    func detectHits(track: Track2D, meta: VideoMeta, camera: CameraModel?, players: [PlayerObservation]?) throws -> [HitEvent] {
        log("hits")
        return [HitEvent(frame: 2, t: 0.1, xyPx: [2, 10], hitter: .near), HitEvent(frame: 8, t: 0.3, xyPx: [8, 10], hitter: .far)]
    }

    func reconstruct(track: Track2D, hits: [HitEvent], camera: CameraModel, meta: VideoMeta,
                     players: [PlayerObservation]?) throws -> ReconstructionResult {
        log("reconstruct")
        return ReconstructionResult(
            segments: [ShotSegment3D(startFrame: 2, endFrame: 8, t0: 0.1, p0: [-5, 0, 2], v0: [20, 0, 3], rmsePx: 1,
                                     frames: [2, 8], positions: [[-5, 0, 2], [3, 0, 1]])],
            landingSigmaM: [0.07])
    }

    private(set) var sigmasSeenByRallies: [Double?]?

    func segmentRallies(track: Track2D, hits: [HitEvent], reconstruction: ReconstructionResult, meta: VideoMeta,
                        camera: CameraModel?, players: [PlayerObservation]?) throws -> [RallyV2] {
        log("rallies")
        lock.withLock { sigmasSeenByRallies = reconstruction.landingSigmaM }
        return [RallyV2(startFrame: 0, endFrame: 5, shots: [ShotEventV2(hit: hits[0], segment: reconstruction.segments.first)]),
                RallyV2(startFrame: 6, endFrame: 11, shots: [ShotEventV2(hit: hits[1], segment: nil)])]
    }

    func classifyShots(_ rally: RallyV2, camera: CameraModel?, players: [PlayerObservation]?) throws -> RallyV2 {
        log("shots:\(rally.startFrame)")
        if rally.startFrame == failShotsForRally { throw Failure.boom }
        var r = rally
        r.shots = r.shots.map { var s = $0; s.shotType = .clear; return s }
        return r
    }

    func scoreMatch(_ rallies: [RallyV2]) throws -> [JSONValue] {
        log("score")
        return rallies.map { _ in .object(["score": .array([.number(1), .number(0)])]) }
    }
}

final class AnalysisPipelineTests: XCTestCase {
    nonisolated(unsafe) static var videoURL: URL!

    override class func setUp() {
        super.setUp()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-analysis-\(UUID().uuidString).mov")
        let done = DispatchSemaphore(value: 0)
        Task.detached { try? await writeTestMovie(to: url, frames: 12, fps: 30); done.signal() }
        done.wait()
        videoURL = url
    }

    override class func tearDown() {
        try? FileManager.default.removeItem(at: videoURL)
        super.tearDown()
    }

    private func run(_ stages: FakeStages, options: PipelineOptions = PipelineOptions(),
                     checkpoints: CheckpointStore = NoCheckpoints()) async throws -> (MatchAnalysis, [AnalysisProgress]) {
        let box = ProgressBox()
        let res = try await AnalysisPipeline(stages: stages, checkpoints: checkpoints)
            .analyze(Self.videoURL, options: options) { box.append($0) }
        return (res, box.items)
    }

    func stageOK(_ res: MatchAnalysis, _ name: String) -> Bool? {
        res.diagnostics["stages"]?[name]?["ok"]?.bool
    }

    func testHappyPathRunsEveryStageInPythonOrder() async throws {
        let stages = FakeStages()
        let (res, prog) = try await run(stages)
        XCTAssertEqual(stages.calls, ["detect", "link", "frames", "auto", "players:2", "hits", "reconstruct",
                                      "rallies", "shots:0", "shots:6", "score"])
        XCTAssertEqual(res.track?.n, 12)
        XCTAssertNotNil(res.camera)
        XCTAssertEqual(res.segments.count, 1)
        XCTAssertEqual(stages.sigmasSeenByRallies, [0.07], "landing sigma must reach the rally stage")
        XCTAssertEqual(res.rallies.map { $0.shots.map(\.shotType) }, [[.clear], [.clear]])
        XCTAssertEqual(res.score.count, 2)
        for s in ["detect", "link", "calibrate", "players", "hits", "reconstruct3d", "rallies", "shots", "score"] {
            XCTAssertEqual(stageOK(res, s), true, s)
        }
        XCTAssertEqual(res.diagnostics["detector"]?.string, "fake")
        XCTAssertEqual(res.diagnostics["video"]?["frames"]?.double, 12)
        XCTAssertNotNil(res.diagnostics["calibration"]?["confidence"])
        let f = prog.map(\.fraction)
        XCTAssertEqual(f, f.sorted(), "progress must never go backwards")
        XCTAssertEqual(f.last, 1)
        XCTAssertGreaterThan(res.processingTimeSeconds, 0)
    }

    func testDetectFailureDegradesWithoutTrack() async throws {
        let stages = FakeStages(); stages.failDetect = true
        let (res, _) = try await run(stages)
        XCTAssertNil(res.track)
        XCTAssertEqual(stageOK(res, "detect"), false)
        XCTAssertEqual(res.diagnostics["stages"]?["detect"]?["error"]?.string?.contains("boom"), true)
        // Python still calibrates and tracks players, then stops.
        XCTAssertEqual(stages.calls, ["detect", "frames", "auto", "players:2"])
        XCTAssertTrue(res.hits.isEmpty && res.rallies.isEmpty)
    }

    func testNoCameraSkipsReconstructionAndNeverFabricatesSpeeds() async throws {
        let stages = FakeStages(); stages.autoCamera = nil
        let (res, _) = try await run(stages)
        XCTAssertNil(res.camera)
        XCTAssertFalse(stages.calls.contains("reconstruct"))
        XCTAssertNil(res.diagnostics["stages"]?["reconstruct3d"])
        XCTAssertTrue(res.segments.isEmpty)
        XCTAssertTrue(res.rallies.flatMap(\.shots).allSatisfy { $0.segment == nil })
    }

    func testCameraHintUsedOnlyWhenAutomaticFails() async throws {
        var hint = FakeStages.camera; hint.confidence = 0.33
        let failing = FakeStages(); failing.autoCamera = nil
        let (res, _) = try await run(failing, options: PipelineOptions(cameraHint: hint))
        XCTAssertEqual(res.camera?.confidence, 0.33)
        let ok = FakeStages()
        let (res2, _) = try await run(ok, options: PipelineOptions(cameraHint: hint))
        XCTAssertEqual(res2.camera?.confidence, FakeStages.camera.confidence)
    }

    func testManualCornersAreScaledThenRefined() async throws {
        let stages = FakeStages()
        let corners: [String: SIMD2<Double>] = ["nb_ld": [0.75, 0.9], "nb_rd": [0.25, 0.9], "fb_ld": [0.6, 0.3], "fb_rd": [0.4, 0.3]]
        let (res, _) = try await run(stages, options: PipelineOptions(manualCorners: corners))
        XCTAssertEqual(stages.lastPoints?["nb_ld"], SIMD2(48, 43.2))   // 64 × 48 frame
        XCTAssertTrue(stages.calls.contains("refine"))
        XCTAssertFalse(stages.calls.contains("auto"))
        XCTAssertEqual(res.camera?.confidence, 0.9)

        let noRefine = FakeStages(); noRefine.failRefine = true
        let (res2, _) = try await run(noRefine, options: PipelineOptions(manualCorners: corners, cornersNormalized: false))
        XCTAssertEqual(noRefine.lastPoints?["nb_ld"], SIMD2(0.75, 0.9))
        XCTAssertEqual(res2.camera?.confidence, 0.5, "refine failure keeps the DLT camera")
        XCTAssertEqual(stageOK(res2, "calibrate"), true)
    }

    func testShotClassificationFailureKeepsRally() async throws {
        let stages = FakeStages(); stages.failShotsForRally = 6
        let (res, _) = try await run(stages)
        XCTAssertEqual(res.rallies.count, 2)
        XCTAssertEqual(res.rallies[0].shots[0].shotType, .clear)
        XCTAssertEqual(res.rallies[1].shots[0].shotType, .unknown)
        XCTAssertEqual(stageOK(res, "shots"), false, "Python keeps the last per-rally record")
    }

    func testCancellationPropagates() async throws {
        let stages = FakeStages(); stages.cancelInPlayers = true
        do {
            _ = try await run(stages)
            XCTFail("expected CancellationError")
        } catch is CancellationError {}
        XCTAssertFalse(stages.calls.contains("hits"))
    }

    func testCheckpointsResumeInterruptedRunAndAreClearedAfter() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-ckpt-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = FileCheckpointStore(directory: dir, fingerprint: "v1")

        let killed = FakeStages(); killed.cancelInPlayers = true
        do { _ = try await run(killed, checkpoints: store); XCTFail("expected cancel") } catch is CancellationError {}
        for key in ["candidates", "track", "camera"] {
            XCTAssertTrue(FileManager.default.fileExists(atPath: dir.appendingPathComponent("\(key).json").path), key)
        }

        let resumed = FakeStages()
        let (res, _) = try await run(resumed, checkpoints: store)
        XCTAssertFalse(resumed.calls.contains("detect"), "detect must not rerun")
        XCTAssertFalse(resumed.calls.contains("link"))
        XCTAssertFalse(resumed.calls.contains("auto"))
        XCTAssertTrue(resumed.calls.contains("players:2"))
        XCTAssertEqual(res.track?.n, 12)
        XCTAssertNotNil(res.camera)
        XCTAssertEqual(res.diagnostics["stages"]?["detect"]?["resumed"]?.bool, true)
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir.path), "cleared after success")

        // A different fingerprint (new video or new app build) ignores old checkpoints.
        let old = FileCheckpointStore(directory: dir, fingerprint: "v1")
        old.save([[ShuttleCandidate(x: 1, y: 1, score: 1)]], key: "candidates")
        let fresh = FakeStages()
        _ = try await run(fresh, checkpoints: FileCheckpointStore(directory: dir, fingerprint: "v2"))
        XCTAssertTrue(fresh.calls.contains("detect"))
    }

    func testCameraCheckpointKeepsAConfirmedMissingCamera() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-ckpt-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = FileCheckpointStore(directory: dir, fingerprint: "v1")
        let killed = FakeStages(); killed.autoCamera = nil; killed.cancelInPlayers = true
        do { _ = try await run(killed, checkpoints: store); XCTFail() } catch is CancellationError {}
        let resumed = FakeStages()     // would find a camera if asked
        let (res, _) = try await run(resumed, checkpoints: store)
        XCTAssertFalse(resumed.calls.contains("auto"))
        XCTAssertNil(res.camera, "a checkpointed 'no camera' result is reused, not recomputed")
    }

    func testPlayerStepMatchesPython() {
        XCTAssertEqual(playerStep(fps: 60), 4)
        XCTAssertEqual(playerStep(fps: 30), 2)
        XCTAssertEqual(playerStep(fps: 25), 2)
        XCTAssertEqual(playerStep(fps: 120), 8)
        XCTAssertEqual(playerStep(fps: nil), 2)
        XCTAssertEqual(playerStep(fps: 7), 1)
        XCTAssertEqual(playerStep(fps: 60, override: 0), 1)
        XCTAssertEqual(playerStep(fps: 60, override: 3), 3)
    }
}

final class ProgressBox: @unchecked Sendable {
    private let lock = NSLock()
    private var _items: [AnalysisProgress] = []
    var items: [AnalysisProgress] { lock.withLock { _items } }
    func append(_ p: AnalysisProgress) { lock.withLock { _items.append(p) } }
}

func writeTestMovie(to url: URL, frames: Int, fps: Int32, width: Int = 64, height: Int = 48) async throws {
    let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
        AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
    ])
    input.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
    ])
    writer.add(input)
    guard writer.startWriting() else { throw writer.error ?? CocoaError(.fileWriteUnknown) }
    writer.startSession(atSourceTime: .zero)
    for f in 0..<frames {
        while !input.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 2_000_000) }
        var pb: CVPixelBuffer?
        guard let pool = adaptor.pixelBufferPool else { throw CocoaError(.fileWriteUnknown) }
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb)
        guard let buf = pb else { throw CocoaError(.fileWriteUnknown) }
        CVPixelBufferLockBaseAddress(buf, [])
        memset(CVPixelBufferGetBaseAddress(buf), Int32(40 + f * 10), CVPixelBufferGetDataSize(buf))
        CVPixelBufferUnlockBaseAddress(buf, [])
        adaptor.append(buf, withPresentationTime: CMTime(value: CMTimeValue(f), timescale: fps))
    }
    input.markAsFinished()
    await writer.finishWriting()
    if writer.status != .completed { throw writer.error ?? CocoaError(.fileWriteUnknown) }
}
