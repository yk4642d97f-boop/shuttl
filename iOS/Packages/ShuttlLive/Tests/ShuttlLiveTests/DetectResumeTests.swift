import Foundation
import XCTest
import ShuttlAnalysis
import ShuttlCore
@testable import ShuttlLive

/// Real TrackNetV3 (CoreML) on a real clip: cancel right after the first checkpoint,
/// resume from it, and the candidates equal an uninterrupted run.
final class DetectResumeTests: XCTestCase {
    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }

    func testCancelledDetectResumesToTheSameCandidates() async throws {
        let clip = repoRoot.appendingPathComponent("backend/ml/data/raw/shuttlecock_trajectory/TrackNetV2/Test/match1/video/1_07_03.mp4")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: clip.path), "Test split missing")
        let models = repoRoot.appendingPathComponent("iOS/Packages/ShuttlDetector/Models")
        var stages = try await LiveStages.make(modelsDirectory: models, kind: .tracknetv3,
                                               cacheDirectory: FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-live-models"),
                                               metricsURL: nil)
        stages.checkpointSeconds = 4
        XCTAssertEqual(stages.detectorName, "tracknetv3")
        let source = try await VideoSource.open(clip)

        let full = try await stages.detect(source, progress: { _ in })

        // Interrupted run: cancel as soon as the first checkpoint arrives.
        let box = CheckpointBox()
        let run = Task { [stages] in
            try await stages.detect(source, resume: nil, checkpoint: { box.store($0) }, progress: { _ in })
        }
        box.onFirst = { run.cancel() }
        do {
            _ = try await run.value
            XCTFail("expected the interrupted run to be cancelled")
        } catch is CancellationError {}
        let cp = try XCTUnwrap(box.first, "a checkpoint must arrive before the end of a \(source.meta.nFrames)-frame clip")
        XCTAssertGreaterThan(cp.nextFrame, 0)
        XCTAssertLessThan(cp.nextFrame, source.meta.nFrames)
        XCTAssertNotNil(cp.background, "TrackNetV3 checkpoints carry the background plan")
        XCTAssertEqual(Array(full.prefix(cp.nextFrame)), cp.candidates, "the checkpoint prefix is already final")

        // Through JSON, as the pipeline persists it, then resume.
        let restored = try JSONDecoder().decode(DetectCheckpoint.self, from: JSONEncoder().encode(cp))
        let resumed = try await stages.detect(source, resume: restored, checkpoint: { _ in }, progress: { _ in })
        XCTAssertNil(stages.detector.lastResumeProblem, stages.detector.lastResumeProblem ?? "")
        XCTAssertEqual(resumed.count, full.count)
        XCTAssertEqual(resumed, full, "resumed from frame \(cp.nextFrame) must equal the uninterrupted run")

        // A checkpoint from another model is rejected, and the result is still right.
        var foreign = restored; foreign.detectorName = "wasb"
        let fresh = try await stages.detect(source, resume: foreign, checkpoint: { _ in }, progress: { _ in })
        XCTAssertEqual(fresh, full)
        XCTAssertNotNil(stages.detector.lastResumeProblem)
    }

    /// The pipeline goes through the protocol, not the concrete type: make sure the
    /// resuming implementation is the one that's dispatched.
    func testProtocolDispatchesTheResumingDetect() async throws {
        let clip = repoRoot.appendingPathComponent("backend/ml/data/raw/shuttlecock_trajectory/TrackNetV2/Test/match1/video/1_07_04.mp4")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: clip.path), "Test split missing")
        var stages = try await LiveStages.make(modelsDirectory: repoRoot.appendingPathComponent("iOS/Packages/ShuttlDetector/Models"),
                                               kind: .tracknetv3,
                                               cacheDirectory: FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-live-models"),
                                               metricsURL: nil)
        stages.checkpointSeconds = 2
        let any: any AnalysisStages = stages
        let box = CheckpointBox()
        _ = try await any.detect(try await VideoSource.open(clip), resume: nil, checkpoint: { box.store($0) }, progress: { _ in })
        XCTAssertGreaterThan(box.count, 0, "the default extension never checkpoints; LiveStages must")
    }
}

final class CheckpointBox: @unchecked Sendable {
    private let lock = NSLock()
    private var _first: DetectCheckpoint?
    private var _count = 0
    var onFirst: (@Sendable () -> Void)? {
        get { lock.withLock { _onFirst } }
        set { lock.withLock { _onFirst = newValue } }
    }
    private var _onFirst: (@Sendable () -> Void)?
    var first: DetectCheckpoint? { lock.withLock { _first } }
    var count: Int { lock.withLock { _count } }

    func store(_ c: DetectCheckpoint) {
        let fire: (@Sendable () -> Void)? = lock.withLock {
            _count += 1
            guard _first == nil else { return nil }
            _first = c
            return _onFirst
        }
        fire?()
    }
}
