import AVFoundation
import XCTest
import ShuttlCore
@testable import ShuttlAnalysis

/// Phone pass 09-30 (F-P3): a 10-min detect failed with -11847 after 185 s; the pipeline then ran
/// calibrate + players, players was cancelled, and the next run started detect from frame 0.
/// A retryable detect failure must end the run at once, keep detect_partial.bin, and the retry
/// (same store, or a fresh store on the same directory) must resume from the saved frame.
final class RetryableDetectErrorTests: XCTestCase {
    nonisolated(unsafe) static var videoURL: URL!
    static let frames = 40
    static let interrupted = NSError(domain: AVFoundationErrorDomain, code: -11847,
                                     userInfo: [NSLocalizedDescriptionKey: "操作已中断"])

    override class func setUp() {
        super.setUp()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-fp3-\(UUID().uuidString).mov")
        let done = DispatchSemaphore(value: 0)
        Task.detached { try? await writeTestMovie(to: url, frames: frames, fps: 30); done.signal() }
        done.wait()
        videoURL = url
    }

    override class func tearDown() {
        try? FileManager.default.removeItem(at: videoURL)
        super.tearDown()
    }

    private func directory() -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-fp3-ckpt-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: d) }
        return d
    }

    private func run(_ s: ResumableStages, _ store: CheckpointStore) async throws -> MatchAnalysis {
        try await AnalysisPipeline(stages: s, checkpoints: store).analyze(Self.videoURL)
    }

    private func reference() async throws -> [[ShuttleCandidate]] {
        let s = ResumableStages()
        _ = try await run(s, NoCheckpoints())
        return try XCTUnwrap(s.linkInput)
    }

    private func events(_ s: ResumableStages, _ name: String) -> [[String: JSONValue]] {
        s.events.filter { $0.name == name }.map(\.fields)
    }

    /// The app's in-process retry loop (MatchPipeline.analyzeLocally): same store, rerun while retryable.
    func testRetryableDetectErrorStopsTheRunAndTheRetryResumesFromTheSavedFrame() async throws {
        let expected = try await reference()
        let store = FileCheckpointStore(directory: directory(), fingerprint: "fp")
        let s = ResumableStages()
        s.failAt = (30, Self.interrupted)

        let first = try await run(s, store)
        XCTAssertEqual(first.outcome, .stageFailed(stage: "detect", message: first.stageFailures[0].message, retryable: true))
        XCTAssertEqual(s.timeline, ["detect"], "no stage may run after a retryable detect failure")
        XCTAssertEqual(first.diagnostics["stopped_after"]?.string, "detect")
        XCTAssertNil(first.diagnostics["stages"]?["calibrate"])
        XCTAssertNil(first.diagnostics["stages"]?["players"])
        XCTAssertEqual(store.loadDetectPartial()?.nextFrame, 24, "checkpoints at 8, 16, 24; failed at 30")
        XCTAssertEqual(events(s, "detect_partial_load").first?["verdict"]?.string, "missing")
        XCTAssertEqual(events(s, "detect_partial_save").map { $0["next_frame"]?.double }, [8, 16, 24])
        XCTAssertEqual(events(s, "detect_partial_save").map { $0["verdict"]?.string }, ["created", "appended", "appended"])
        XCTAssertEqual(first.diagnostics["stages"]?["detect"]?["partial_last_frame"]?.double, 24)

        // The decoder is back; the retry reuses the store and the SAME stages (as the app does).
        s.failAt = nil
        var res = first
        var attempts = 1
        while res.outcome.isRetryable && attempts < 6 {
            res = try await run(s, store)
            attempts += 1
        }
        XCTAssertEqual(attempts, 2, "one retry is enough")
        XCTAssertEqual(s.offered.last??.nextFrame, 24, "the retry resumes from the saved frame, not 0")
        XCTAssertEqual(s.timeline.prefix(3), ["detect", "detect", "link"],
                       "straight from the failed detect to the retry's detect")
        XCTAssertEqual(Array(s.detectedFrames.suffix(Self.frames - 24)), Array(24..<Self.frames))
        XCTAssertEqual(s.linkInput, expected, "identical candidates")
        XCTAssertEqual(res.diagnostics["stages"]?["detect"]?["resume_offered_frame"]?.double, 24)
        XCTAssertEqual(res.diagnostics["stages"]?["detect"]?["partial_load"]?.string, "ok")
        XCTAssertEqual(events(s, "detect_partial_load").last?["next_frame"]?.double, 24)
        XCTAssertFalse(res.outcome.isRetryable)
        XCTAssertFalse(FileManager.default.fileExists(atPath: store.directory.path), "cleared after a clean run")
    }

    /// The device's 17:30:32 run was a NEW analyzeLocally (a cancel had ended the first): a new
    /// store object on the same directory with a fingerprint recomputed from the same file.
    func testFreshStoreOnTheSameDirectoryResumesToo() async throws {
        let expected = try await reference()
        let dir = directory()
        let attrs = try FileManager.default.attributesOfItem(atPath: Self.videoURL.path)
        func fingerprint() -> String {
            checkpointFingerprint(videoSize: (attrs[.size] as? NSNumber)?.intValue ?? 0,
                                  modified: attrs[.modificationDate] as? Date, build: "1.0-1", options: PipelineOptions())
        }
        let s1 = ResumableStages(); s1.failAt = (30, Self.interrupted)
        _ = try await run(s1, FileCheckpointStore(directory: dir, fingerprint: fingerprint()))

        let s2 = ResumableStages()
        let res = try await run(s2, FileCheckpointStore(directory: dir, fingerprint: fingerprint()))
        XCTAssertEqual(s2.offered.first??.nextFrame, 24)
        XCTAssertEqual(s2.detectedFrames, Array(24..<Self.frames))
        XCTAssertEqual(s2.linkInput, expected)
        XCTAssertEqual(res.outcome.isRetryable, false)
    }

    func testLoadVerdictsExplainAMissedResume() async throws {
        let dir = directory()
        let s1 = ResumableStages(); s1.failAt = (30, Self.interrupted)
        _ = try await run(s1, FileCheckpointStore(directory: dir, fingerprint: "fp-a"))

        let other = FileCheckpointStore(directory: dir, fingerprint: "fp-b")
        let (none, mismatch) = other.loadDetectPartialReporting()
        XCTAssertNil(none)
        XCTAssertEqual(mismatch.verdict, "fingerprint_mismatch")
        XCTAssertEqual(mismatch.nextFrame, 24)
        XCTAssertNotNil(mismatch.bytes)

        let url = FileCheckpointStore(directory: dir, fingerprint: "fp-a").detectPartialURL
        try Data("not a checkpoint".utf8).write(to: url)
        XCTAssertEqual(FileCheckpointStore(directory: dir, fingerprint: "fp-a").loadDetectPartialReporting().1.verdict, "unreadable")
        try FileManager.default.removeItem(at: url)
        XCTAssertEqual(FileCheckpointStore(directory: dir, fingerprint: "fp-a").loadDetectPartialReporting().1.verdict, "missing")
    }

    func testAFailedSaveIsReportedNotSwallowed() async throws {
        let parent = directory()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let blocked = parent.appendingPathComponent("ckpt")
        try Data([1]).write(to: blocked)                 // a FILE where the checkpoint directory should be
        let s = ResumableStages(); s.failAt = (30, Self.interrupted)
        let res = try await run(s, FileCheckpointStore(directory: blocked, fingerprint: "fp"))
        let saves = events(s, "detect_partial_save")
        XCTAssertEqual(saves.count, 3)
        XCTAssertEqual(Set(saves.compactMap { $0["verdict"]?.string }), ["failed"])
        XCTAssertNotNil(saves.first?["error"]?.string)
        XCTAssertEqual(res.diagnostics["stages"]?["detect"]?["partial_last_save"]?.string, "failed")
    }

    func testNonRetryableDetectErrorStillDegradesThroughTheLaterStages() async throws {
        let s = ResumableStages(); s.failAt = (30, DetectorLikeError.logic)
        let res = try await run(s, FileCheckpointStore(directory: directory(), fingerprint: "fp"))
        XCTAssertEqual(res.outcome, .stageFailed(stage: "detect", message: "DetectorLikeError: logic", retryable: false))
        XCTAssertNil(res.diagnostics["stopped_after"])
        XCTAssertEqual(s.laterCalls, ["calibration_frames", "calibrate_auto", "players"],
                       "a permanent failure still gets calibration diagnostics")
    }
}
