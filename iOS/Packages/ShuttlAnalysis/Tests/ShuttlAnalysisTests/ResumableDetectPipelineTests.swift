import XCTest
import ShuttlCore
@testable import ShuttlAnalysis

final class ResumableDetectPipelineTests: XCTestCase {
    nonisolated(unsafe) static var videoURL: URL!
    static let frames = 40

    override class func setUp() {
        super.setUp()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-resume-\(UUID().uuidString).mov")
        let done = DispatchSemaphore(value: 0)
        Task.detached { try? await writeTestMovie(to: url, frames: frames, fps: 30); done.signal() }
        done.wait()
        videoURL = url
    }

    override class func tearDown() {
        try? FileManager.default.removeItem(at: videoURL)
        super.tearDown()
    }

    private func store() -> FileCheckpointStore {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-resume-ckpt-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: d) }
        return FileCheckpointStore(directory: d, fingerprint: "fp")
    }

    private func run(_ s: ResumableStages, _ store: CheckpointStore) async throws -> MatchAnalysis {
        try await AnalysisPipeline(stages: s, checkpoints: store).analyze(Self.videoURL)
    }

    private func kill(_ s: ResumableStages, _ store: CheckpointStore, file: StaticString = #filePath, line: UInt = #line) async {
        do { _ = try await run(s, store); XCTFail("expected the run to be killed", file: file, line: line) } catch is CancellationError {
        } catch { XCTFail("unexpected \(error)", file: file, line: line) }
    }

    private func reference() async throws -> [[ShuttleCandidate]] {
        let s = ResumableStages()
        _ = try await run(s, NoCheckpoints())
        XCTAssertEqual(s.detectedFrames.count, Self.frames)
        return try XCTUnwrap(s.linkInput)
    }

    private func exists(_ url: URL) -> Bool { FileManager.default.fileExists(atPath: url.path) }

    func testKilledDetectResumesFromLastCheckpointWithIdenticalCandidates() async throws {
        let expected = try await reference()
        let st = store()
        let killed = ResumableStages(); killed.cancelAtFrame = 30
        await kill(killed, st)
        XCTAssertTrue(exists(st.detectPartialURL))
        XCTAssertFalse(exists(st.directory.appendingPathComponent("candidates.json")))
        let saved = try XCTUnwrap(st.loadDetectPartial())
        XCTAssertEqual(saved.nextFrame, 24, "checkpoints at 8, 16, 24; killed at 30")
        XCTAssertEqual(saved.candidates, Array(expected.prefix(24)))
        XCTAssertEqual(saved.background, ResumableStages.background)
        XCTAssertEqual(saved.detectorName, "fakenet")

        let resumed = ResumableStages()
        let res = try await run(resumed, st)
        XCTAssertEqual(resumed.offered.first??.nextFrame, 24)
        XCTAssertEqual(resumed.detectedFrames, Array(24..<Self.frames), "only the remaining frames are detected")
        XCTAssertEqual(resumed.linkInput, expected, "identical candidates")
        XCTAssertEqual(res.diagnostics["stages"]?["detect"]?["resume_offered_frame"]?.double, 24)
        XCTAssertEqual(res.diagnostics["stages"]?["detect"]?["ok"]?.bool, true)
        XCTAssertFalse(exists(st.directory), "a run without failures clears everything")
    }

    func testDetectorNameMismatchGivesACleanFullRun() async throws {
        let expected = try await reference()
        let st = store()
        let a = ResumableStages(name: "netA"); a.cancelAtFrame = 30
        await kill(a, st)
        let b = ResumableStages(name: "netB"); b.cancelAtFrame = 20
        await kill(b, st)
        XCTAssertEqual(b.offered.first??.detectorName, "netA", "offered as-is; the detector decides")
        XCTAssertEqual(b.detectedFrames, Array(0..<20), "rejected → started from frame 0")
        let rewritten = try XCTUnwrap(st.loadDetectPartial())
        XCTAssertEqual(rewritten.detectorName, "netB", "file rewritten, not appended to netA's")
        XCTAssertEqual(rewritten.nextFrame, 16)
        XCTAssertEqual(rewritten.candidates, Array(expected.prefix(16)))

        let full = ResumableStages(name: "netC")
        _ = try await run(full, st)
        XCTAssertEqual(full.detectedFrames, Array(0..<Self.frames))
        XCTAssertEqual(full.linkInput, expected)
    }

    func testCorruptOrTornPartialFileNeverBreaksTheRun() async throws {
        let expected = try await reference()
        let st = store()
        try FileManager.default.createDirectory(at: st.directory, withIntermediateDirectories: true)
        try Data("garbage, not a checkpoint".utf8).write(to: st.detectPartialURL)
        XCTAssertNil(st.loadDetectPartial())
        let s1 = ResumableStages(); s1.cancelAtFrame = 30
        await kill(s1, st)
        XCTAssertEqual(s1.detectedFrames, Array(0..<30), "corrupt file → full run")
        XCTAssertEqual(try XCTUnwrap(st.loadDetectPartial()).nextFrame, 24, "and a fresh file replaced it")

        // Torn last record: resume from the one before.
        let data = try Data(contentsOf: st.detectPartialURL)
        try data.dropLast(5).write(to: st.detectPartialURL)
        let s2 = ResumableStages()
        _ = try await run(s2, st)
        XCTAssertEqual(s2.detectedFrames, Array(16..<Self.frames))
        XCTAssertEqual(s2.linkInput, expected)

        // Random bytes after a valid header.
        let st2 = store()
        let k = ResumableStages(); k.cancelAtFrame = 12
        await kill(k, st2)
        var bytes = [UInt8](try Data(contentsOf: st2.detectPartialURL))
        for i in stride(from: 20, to: bytes.count, by: 3) { bytes[i] ^= 0x5A }
        try Data(bytes).write(to: st2.detectPartialURL)
        let s3 = ResumableStages()
        _ = try await run(s3, st2)
        XCTAssertEqual(s3.linkInput, expected)
    }

    func testPartialIsClearedOnceCandidatesAreSavedEvenIfALaterStageFails() async throws {
        let st = store()
        let killed = ResumableStages(); killed.cancelAtFrame = 30
        await kill(killed, st)
        let failing = ResumableStages(); failing.failHits = true
        let res = try await run(failing, st)
        XCTAssertEqual(res.outcome.isRetryable, false)
        XCTAssertTrue(exists(st.directory.appendingPathComponent("candidates.json")), "kept for the retry")
        XCTAssertFalse(exists(st.detectPartialURL), "obsolete once candidates exist")
        let retry = ResumableStages()
        _ = try await run(retry, st)
        XCTAssertEqual(retry.detectedFrames, [], "detect restored from candidates")
    }

    /// A full disk: every save of `dropKey` silently does nothing (FileCheckpointStore.save is best effort).
    private struct DroppingStore: CheckpointStore, DetectPartialStore {
        let inner: FileCheckpointStore
        let dropKey: String
        func load<T: Decodable>(_ type: T.Type, key: String) -> T? { inner.load(type, key: key) }
        func save<T: Encodable>(_ value: T, key: String) { if key != dropKey { inner.save(value, key: key) } }
        func clear() { inner.clear() }
        func loadDetectPartial() -> DetectCheckpoint? { inner.loadDetectPartial() }
        func saveDetectPartial(_ c: DetectCheckpoint) { inner.saveDetectPartial(c) }
        func clearDetectPartial() { inner.clearDetectPartial() }
    }

    func testPartialSurvivesWhenCandidatesCouldNotBeSaved() async throws {
        let expected = try await reference()
        let st = store()
        let full = DroppingStore(inner: st, dropKey: "candidates")
        let killed = ResumableStages(); killed.cancelAtFrame = 30
        await kill(killed, full)
        let failing = ResumableStages(); failing.failHits = true
        _ = try await run(failing, full)
        XCTAssertFalse(exists(st.directory.appendingPathComponent("candidates.json")), "the disk was full")
        let kept = try XCTUnwrap(st.loadDetectPartial(), "detect progress must not be thrown away")
        XCTAssertGreaterThanOrEqual(kept.nextFrame, 24)

        // The failed run's track checkpoint would let link be skipped; drop it so the resumed candidates reach link.
        try FileManager.default.removeItem(at: st.directory.appendingPathComponent("track.json"))
        let retry = ResumableStages()
        _ = try await run(retry, st)
        XCTAssertEqual(retry.offered.first??.nextFrame, kept.nextFrame, "resumes instead of starting from frame 0")
        XCTAssertEqual(retry.detectedFrames, Array(kept.nextFrame..<Self.frames))
        XCTAssertEqual(retry.linkInput, expected)
    }

    func testStagesWithoutResumeSupportUseTheOldDetect() async throws {
        let st = store()
        let s = ScriptedStages(); s.hitsError = DetectorLikeError.logic
        _ = try await AnalysisPipeline(stages: s, checkpoints: st).analyze(Self.videoURL)
        XCTAssertTrue(s.calls.contains("detect"))
        XCTAssertFalse(exists(st.detectPartialURL))
    }
}
