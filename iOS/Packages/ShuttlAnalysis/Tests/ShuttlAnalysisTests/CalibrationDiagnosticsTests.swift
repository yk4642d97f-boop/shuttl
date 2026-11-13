import XCTest
import ShuttlCore
@testable import ShuttlAnalysis

/// A calibrator that explains its verdict, the way LiveStages forwards ShuttlCourt's diagnostics.
final class ExplainingStages: AnalysisStages, @unchecked Sendable {
    let base = FakeStages()
    var accept = false

    var detectorName: String { base.detectorName }
    func detect(_ source: VideoSource, progress: @escaping StageProgress) async throws -> [[ShuttleCandidate]] {
        try await base.detect(source, progress: progress)
    }
    func link(_ candidates: [[ShuttleCandidate]], meta: VideoMeta) throws -> Track2D { try base.link(candidates, meta: meta) }
    func calibrationFrames(_ source: VideoSource, count: Int) throws -> [ImageU8] { try base.calibrationFrames(source, count: count) }
    func calibrateAutomatic(frames: [ImageU8], size: ImageSize, doublesHint: Bool?) throws -> CameraModel? {
        XCTFail("the pipeline must use the detailed variant"); return nil
    }
    func calibrateAutomaticDetailed(frames: [ImageU8], size: ImageSize, doublesHint: Bool?) throws -> CalibrationAttempt {
        CalibrationAttempt(camera: accept ? FakeStages.camera : nil,
                           diagnostics: ["n_t": .number(2), "n_l": .number(5), "valid": .bool(accept)])
    }
    func calibrateFromPoints(_ points: [String: SIMD2<Double>], size: ImageSize) throws -> CameraModel {
        try base.calibrateFromPoints(points, size: size)
    }
    func refineCamera(_ camera: CameraModel, frames: [ImageU8]) throws -> CameraModel { try base.refineCamera(camera, frames: frames) }
    func trackPlayers(_ source: VideoSource, camera: CameraModel?, step: Int, doubles: Bool?,
                      progress: @escaping StageProgress) async throws -> [PlayerObservation] {
        try await base.trackPlayers(source, camera: camera, step: step, doubles: doubles, progress: progress)
    }
    func detectHits(track: Track2D, meta: VideoMeta, camera: CameraModel?, players: [PlayerObservation]?) throws -> [HitEvent] {
        try base.detectHits(track: track, meta: meta, camera: camera, players: players)
    }
    func reconstruct(track: Track2D, hits: [HitEvent], camera: CameraModel, meta: VideoMeta,
                     players: [PlayerObservation]?) throws -> ReconstructionResult {
        try base.reconstruct(track: track, hits: hits, camera: camera, meta: meta, players: players)
    }
    func segmentRallies(track: Track2D, hits: [HitEvent], reconstruction: ReconstructionResult, meta: VideoMeta,
                        camera: CameraModel?, players: [PlayerObservation]?) throws -> [RallyV2] {
        try base.segmentRallies(track: track, hits: hits, reconstruction: reconstruction, meta: meta,
                                camera: camera, players: players)
    }
    func classifyShots(_ rally: RallyV2, camera: CameraModel?, players: [PlayerObservation]?) throws -> RallyV2 {
        try base.classifyShots(rally, camera: camera, players: players)
    }
    func scoreMatch(_ rallies: [RallyV2]) throws -> [JSONValue] { try base.scoreMatch(rallies) }
}

final class CalibrationDiagnosticsTests: XCTestCase {
    private func analyze(_ stages: some AnalysisStages, options: PipelineOptions = PipelineOptions()) async throws -> MatchAnalysis {
        try await AnalysisPipeline(stages: stages).analyze(AnalysisPipelineTests.videoURL, options: options)
    }

    override class func setUp() {
        super.setUp()
        AnalysisPipelineTests.setUp()
    }

    func testRejectedAutomaticCalibrationKeepsItsReason() async throws {
        let stages = ExplainingStages()
        let res = try await analyze(stages)
        XCTAssertNil(res.camera)
        XCTAssertEqual(res.diagnostics["calibration_auto"]?["n_t"]?.double, 2)
        XCTAssertEqual(res.diagnostics["calibration_auto"]?["valid"]?.bool, false)
    }

    func testAcceptedCalibrationAlsoRecordsItsEvidence() async throws {
        let stages = ExplainingStages(); stages.accept = true
        let res = try await analyze(stages)
        XCTAssertNotNil(res.camera)
        XCTAssertEqual(res.diagnostics["calibration_auto"]?["valid"]?.bool, true)
    }

    func testManualCornersSkipTheAutomaticPath() async throws {
        let stages = ExplainingStages()
        let corners: [String: SIMD2<Double>] = ["nb_ld": [0.75, 0.9], "nb_rd": [0.25, 0.9], "fb_ld": [0.6, 0.3], "fb_rd": [0.4, 0.3]]
        let res = try await analyze(stages, options: PipelineOptions(manualCorners: corners))
        XCTAssertNotNil(res.camera)
        XCTAssertNil(res.diagnostics["calibration_auto"])
    }

    func testFocalHintReachesTheCalibratorAndKeysTheCheckpoints() async throws {
        final class HintStages: AnalysisStages, @unchecked Sendable {
            let inner = ExplainingStages()
            var seen: Double??
            var detectorName: String { inner.detectorName }
            func detect(_ source: VideoSource, progress: @escaping StageProgress) async throws -> [[ShuttleCandidate]] {
                try await inner.detect(source, progress: progress)
            }
            func link(_ c: [[ShuttleCandidate]], meta: VideoMeta) throws -> Track2D { try inner.link(c, meta: meta) }
            func calibrationFrames(_ s: VideoSource, count: Int) throws -> [ImageU8] { try inner.calibrationFrames(s, count: count) }
            func calibrateAutomatic(frames: [ImageU8], size: ImageSize, doublesHint: Bool?) throws -> CameraModel? { nil }
            func calibrateAutomaticDetailed(frames: [ImageU8], size: ImageSize, doublesHint: Bool?,
                                            focalHintPx: Double?) throws -> CalibrationAttempt {
                seen = .some(focalHintPx)
                return CalibrationAttempt(camera: nil, diagnostics: ["reason": .string("focal_mismatch")])
            }
            func calibrateFromPoints(_ p: [String: SIMD2<Double>], size: ImageSize) throws -> CameraModel {
                try inner.calibrateFromPoints(p, size: size)
            }
            func refineCamera(_ c: CameraModel, frames: [ImageU8]) throws -> CameraModel { try inner.refineCamera(c, frames: frames) }
            func trackPlayers(_ s: VideoSource, camera: CameraModel?, step: Int, doubles: Bool?,
                              progress: @escaping StageProgress) async throws -> [PlayerObservation] {
                try await inner.trackPlayers(s, camera: camera, step: step, doubles: doubles, progress: progress)
            }
            func detectHits(track: Track2D, meta: VideoMeta, camera: CameraModel?, players: [PlayerObservation]?) throws -> [HitEvent] {
                try inner.detectHits(track: track, meta: meta, camera: camera, players: players)
            }
            func reconstruct(track: Track2D, hits: [HitEvent], camera: CameraModel, meta: VideoMeta,
                             players: [PlayerObservation]?) throws -> ReconstructionResult {
                try inner.reconstruct(track: track, hits: hits, camera: camera, meta: meta, players: players)
            }
            func segmentRallies(track: Track2D, hits: [HitEvent], reconstruction: ReconstructionResult, meta: VideoMeta,
                                camera: CameraModel?, players: [PlayerObservation]?) throws -> [RallyV2] {
                try inner.segmentRallies(track: track, hits: hits, reconstruction: reconstruction, meta: meta,
                                         camera: camera, players: players)
            }
            func classifyShots(_ r: RallyV2, camera: CameraModel?, players: [PlayerObservation]?) throws -> RallyV2 {
                try inner.classifyShots(r, camera: camera, players: players)
            }
            func scoreMatch(_ r: [RallyV2]) throws -> [JSONValue] { try inner.scoreMatch(r) }
        }
        let stages = HintStages()
        let res = try await analyze(stages, options: PipelineOptions(focalHintPx: 1370))
        XCTAssertEqual(stages.seen, .some(1370))
        XCTAssertEqual(res.diagnostics["calibration_auto"]?["reason"]?.string, "focal_mismatch")

        let base = checkpointFingerprint(videoSize: 1, modified: nil, build: "b", options: PipelineOptions())
        XCTAssertNotEqual(base, checkpointFingerprint(videoSize: 1, modified: nil, build: "b",
                                                      options: PipelineOptions(focalHintPx: 1370)))
        XCTAssertFalse(canonicalOptions(PipelineOptions()).contains { $0.hasPrefix("focalHintPx") },
                       "runs without a hint keep their old fingerprint")
    }

    func testTheDefaultImplementationWrapsTheOldCall() throws {
        let stages = FakeStages()
        let attempt = try stages.calibrateAutomaticDetailed(frames: [], size: ImageSize(width: 64, height: 48), doublesHint: nil)
        XCTAssertNotNil(attempt.camera)
        XCTAssertTrue(attempt.diagnostics.isEmpty)
    }
}
