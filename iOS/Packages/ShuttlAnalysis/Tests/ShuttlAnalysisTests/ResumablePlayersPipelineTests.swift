import AVFoundation
import XCTest
import ShuttlCore
@testable import ShuttlAnalysis

/// P2-2: players (~400 s on a 10-min clip on the phone) has its own partial checkpoint, like
/// detect: fingerprinted, append-only, players_partial_load / save / clear events, and a
/// retryable players failure stops the run after players so the retry resumes it.
final class ResumablePlayersPipelineTests: XCTestCase {
    nonisolated(unsafe) static var videoURL: URL!
    static let frames = 40                                  // 30 fps → step 2 → 20 sampled frames
    static let interrupted = NSError(domain: AVFoundationErrorDomain, code: -11847,
                                     userInfo: [NSLocalizedDescriptionKey: "操作已中断"])

    override class func setUp() {
        super.setUp()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-p2-\(UUID().uuidString).mov")
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
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-p2-ckpt-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: d) }
        return d
    }

    private func run(_ s: ResumableStages, _ store: CheckpointStore) async throws -> MatchAnalysis {
        try await AnalysisPipeline(stages: s, checkpoints: store).analyze(Self.videoURL)
    }

    private func events(_ s: ResumableStages, _ name: String) -> [[String: JSONValue]] {
        s.events.filter { $0.name == name }.map(\.fields)
    }

    private func reference() async throws -> [PlayerObservation] {
        let s = ResumableStages()
        let res = try await run(s, NoCheckpoints())
        XCTAssertEqual(s.playersSampled, Array(stride(from: 0, to: Self.frames, by: 2)))
        return res.players
    }

    /// The app's in-process retry: same store, same stages, rerun while retryable.
    func testRetryablePlayersErrorStopsAfterPlayersAndTheRetryResumes() async throws {
        let expected = try await reference()
        let store = FileCheckpointStore(directory: directory(), fingerprint: "fp")
        let s = ResumableStages()
        s.playersFailAt = (30, Self.interrupted)

        let first = try await run(s, store)
        XCTAssertEqual(first.outcome, .stageFailed(stage: "players", message: first.stageFailures[0].message, retryable: true))
        XCTAssertEqual(first.diagnostics["stopped_after"]?.string, "players")
        XCTAssertEqual(s.laterCalls.filter { $0 != "calibration_frames" && $0 != "calibrate_auto" }, ["link", "players"],
                       "nothing after a retryable players failure")
        XCTAssertNil(first.diagnostics["stages"]?["hits"])
        XCTAssertEqual(events(s, "players_partial_load").map { $0["verdict"]?.string }, ["missing"])
        XCTAssertEqual(events(s, "players_partial_save").map { $0["next_frame"]?.double }, [8, 16, 24])
        XCTAssertEqual(events(s, "players_partial_save").map { $0["verdict"]?.string }, ["created", "appended", "appended"])
        XCTAssertEqual(first.diagnostics["stages"]?["players"]?["partial_last_frame"]?.double, 24)
        XCTAssertEqual(store.loadPlayersPartial()?.nextFrame, 24)
        XCTAssertTrue(events(s, "players_partial_clear").isEmpty)

        s.playersFailAt = nil
        var res = first, attempts = 1
        while res.outcome.isRetryable && attempts < 6 {
            res = try await run(s, store)
            attempts += 1
        }
        XCTAssertEqual(attempts, 2, "one retry is enough")
        XCTAssertEqual(s.playersOffered.last??.nextFrame, 24, "the retry resumes players from the saved frame")
        XCTAssertEqual(s.detectedFrames.count, Self.frames, "detect came from its checkpoint, not run again")
        XCTAssertEqual(Array(s.playersSampled.suffix(8)), Array(stride(from: 24, to: Self.frames, by: 2)))
        XCTAssertEqual(res.players, expected, "identical observations")
        XCTAssertEqual(res.diagnostics["stages"]?["players"]?["resume_offered_frame"]?.double, 24)
        XCTAssertEqual(res.diagnostics["stages"]?["players"]?["partial_load"]?.string, "ok")
        XCTAssertNil(res.diagnostics["stopped_after"])
        XCTAssertFalse(res.outcome.isRetryable)
        XCTAssertFalse(FileManager.default.fileExists(atPath: store.directory.path), "cleared after a clean run")
    }

    /// Killed (cancelled: lock, SIGKILL-like) during players at several points, each time
    /// resumed by a NEW store on the same directory (a relaunch): identical observations.
    func testKilledPlayersResumesFromEveryPointInAFreshProcess() async throws {
        let expected = try await reference()
        for cut in [10, 18, 26, 36] {
            let dir = directory()
            let s1 = ResumableStages()
            s1.playersCancelAtFrame = cut
            do { _ = try await run(s1, FileCheckpointStore(directory: dir, fingerprint: "fp")); XCTFail("killed") }
            catch is CancellationError {}
            let saved = (cut / 8) * 8                                  // last save at or below the cut
            let store = FileCheckpointStore(directory: dir, fingerprint: "fp")
            PlayersPartialFile.cache.forget(store.playersPartialURL)   // a new process knows nothing
            XCTAssertEqual(store.loadPlayersPartial()?.nextFrame, saved, "@\(cut)")
            let s2 = ResumableStages()
            let res = try await run(s2, store)
            XCTAssertEqual(s2.detectedFrames, [], "@\(cut): detect from its checkpoint")
            XCTAssertEqual(s2.playersOffered.first??.nextFrame, saved, "@\(cut)")
            XCTAssertEqual(s2.playersSampled, Array(stride(from: saved, to: Self.frames, by: 2)), "@\(cut)")
            XCTAssertEqual(res.players, expected, "@\(cut): identical observations")
        }
    }

    /// The store: append-only (a save writes exactly one new record), torn tails are dropped,
    /// load verdicts explain a missed resume, other fingerprints / contexts are never reused.
    func testPlayersPartialStoreAppendsAndReports() throws {
        let store = FileCheckpointStore(directory: directory(), fingerprint: "fp")
        let url = store.playersPartialURL
        func cp(_ k: Int, ctx: String = "c1") -> PlayersCheckpoint {
            PlayersCheckpoint(nextFrame: 2 * k, context: ctx, frames: (0..<k).map { ResumableStages.playerRecord(2 * $0) })
        }
        func size() -> Int { (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? -1 }
        XCTAssertEqual(store.loadPlayersPartialReporting().1.verdict, "missing")
        XCTAssertFalse(store.hasPlayersPartial)

        XCTAssertEqual(store.savePlayersPartialReporting(cp(4)).verdict, "created")
        var last = size()
        for k in [8, 12, 20] {
            let r = store.savePlayersPartialReporting(cp(k))
            XCTAssertEqual(r.verdict, "appended")
            let delta = PlayersPartialFile.deltaRecord(start: k == 8 ? 4 : (k == 12 ? 8 : 12),
                                                       frames: cp(k).frames[(k == 8 ? 4 : (k == 12 ? 8 : 12))...],
                                                       nextFrame: 2 * k)
            XCTAssertEqual(size() - last, delta.count, "k=\(k): only the new frames are written")
            XCTAssertEqual(r.bytes, size())
            last = size()
        }
        XCTAssertEqual(store.savePlayersPartialReporting(cp(20)).verdict, "unchanged")
        XCTAssertEqual(store.loadPlayersPartial(), cp(20))

        // A torn append (killed mid-write): the next process drops the tail and appends.
        let h = try FileHandle(forWritingTo: url); try h.seekToEnd(); try h.write(contentsOf: Data([2, 9, 9, 9])); try h.close()
        PlayersPartialFile.cache.forget(url)
        XCTAssertEqual(store.loadPlayersPartial(), cp(20), "a torn tail never hides what came before")
        XCTAssertEqual(store.savePlayersPartialReporting(cp(24)).verdict, "appended")
        PlayersPartialFile.cache.forget(url)
        XCTAssertEqual(store.loadPlayersPartial(), cp(24))

        // A file changed behind the cache's back (same process) is re-parsed, not trusted.
        let h2 = try FileHandle(forWritingTo: url); try h2.seekToEnd(); try h2.write(contentsOf: Data([7])); try h2.close()
        XCTAssertEqual(store.savePlayersPartialReporting(cp(28)).verdict, "appended")
        PlayersPartialFile.cache.forget(url)
        XCTAssertEqual(store.loadPlayersPartial(), cp(28))

        // Another context (re-calibrated camera) or a restarted pass: rewritten, not appended.
        XCTAssertEqual(store.savePlayersPartialReporting(cp(4, ctx: "c2")).verdict, "created")
        XCTAssertEqual(store.loadPlayersPartial(), cp(4, ctx: "c2"))
        var other = cp(8, ctx: "c2"); other.frames[0] = Data([1, 2, 3])        // a different pass's first frame
        XCTAssertEqual(store.savePlayersPartialReporting(other).verdict, "created", "a different prefix")

        // Another run's fingerprint.
        let otherStore = FileCheckpointStore(directory: store.directory, fingerprint: "fp2")
        XCTAssertEqual(otherStore.loadPlayersPartialReporting().1.verdict, "fingerprint_mismatch")
        XCTAssertEqual(otherStore.savePlayersPartialReporting(cp(4)).verdict, "created")
        XCTAssertEqual(store.loadPlayersPartialReporting().1.verdict, "fingerprint_mismatch")

        // Header only / garbage.
        let header = PlayersPartialFile.header(fingerprint: "fp", context: "c1")
        try Data(header).write(to: url); PlayersPartialFile.cache.forget(url)
        XCTAssertEqual(store.loadPlayersPartialReporting().1.verdict, "empty")
        try Data("not a partial".utf8).write(to: url); PlayersPartialFile.cache.forget(url)
        XCTAssertEqual(store.loadPlayersPartialReporting().1.verdict, "unreadable")
        XCTAssertEqual(store.savePlayersPartialReporting(cp(4)).verdict, "created")
        store.clearPlayersPartial()
        XCTAssertFalse(store.hasPlayersPartial)
    }

    func testCorruptPartialNeverBreaksTheRun() async throws {
        let expected = try await reference()
        let store = FileCheckpointStore(directory: directory(), fingerprint: "fp")
        let s = ResumableStages(); s.playersFailAt = (30, Self.interrupted)
        _ = try await run(s, store)
        var bytes = try [UInt8](Data(contentsOf: store.playersPartialURL))
        bytes[bytes.count - 3] ^= 0xff                                  // damage the last record
        try Data(bytes).write(to: store.playersPartialURL)
        PlayersPartialFile.cache.forget(store.playersPartialURL)
        let s2 = ResumableStages()
        let res = try await run(s2, store)
        XCTAssertEqual(s2.playersOffered.first??.nextFrame, 16, "the damaged record is dropped, the ones before kept")
        XCTAssertEqual(res.players, expected)
        XCTAssertFalse(res.outcome.isRetryable)
    }

    /// A non-retryable players failure still degrades through the later stages (no stopped_after),
    /// and keeps the partial for a later manual retry.
    func testNonRetryablePlayersErrorDegradesAsBefore() async throws {
        let store = FileCheckpointStore(directory: directory(), fingerprint: "fp")
        let s = ResumableStages(); s.playersFailAt = (30, DetectorLikeError.logic)
        let res = try await run(s, store)
        XCTAssertNil(res.diagnostics["stopped_after"])
        XCTAssertTrue(s.laterCalls.contains("hits"), "later stages still run")
        XCTAssertEqual(res.players, [])
        XCTAssertEqual(store.loadPlayersPartial()?.nextFrame, 24)
    }

    /// P2-4: the watchdog's error is retryable by type and by its recorded message.
    func testStageStalledErrorIsRetryable() {
        let e = StageStalledError(stage: "players", seconds: 61.2, lastFrame: 38)
        XCTAssertTrue(AnalysisErrorClassifier.isRetryable(e))
        let f = StageFailure(stage: "players", error: e)
        XCTAssertTrue(f.retryable)
        XCTAssertTrue(AnalysisErrorClassifier.isRetryable(message: f.message, domain: f.domain), f.message)
        XCTAssertEqual(e.description, "players stalled: no progress for 61 s after frame 38")
    }

    /// For R4's last attempt: stopOnRetryableFailure false = the pre-P2 behaviour (later stages
    /// run without players, no stopped_after), the outcome stays retryable and the partial is
    /// kept, so a later normal run still resumes.
    func testStopCanBeTurnedOffForALastAttempt() async throws {
        let expected = try await reference()
        let store = FileCheckpointStore(directory: directory(), fingerprint: "fp")
        let s = ResumableStages(); s.playersFailAt = (30, Self.interrupted)
        let last = try await AnalysisPipeline(stages: s, checkpoints: store, stopOnRetryableFailure: false).analyze(Self.videoURL)
        XCTAssertNil(last.diagnostics["stopped_after"])
        XCTAssertTrue(s.laterCalls.contains("hits"), "later stages run on what there is")
        XCTAssertNotNil(last.diagnostics["stages"]?["rallies"])
        XCTAssertEqual(last.players, [])
        XCTAssertEqual(last.outcome, .stageFailed(stage: "players", message: last.stageFailures[0].message, retryable: true))
        XCTAssertEqual(store.loadPlayersPartial()?.nextFrame, 24, "partial kept")

        s.playersFailAt = nil
        let res = try await run(s, store)
        XCTAssertEqual(s.playersOffered.last??.nextFrame, 24)
        XCTAssertEqual(res.players, expected)

        let d = ResumableStages(); d.failAt = (30, Self.interrupted)
        let detectLast = try await AnalysisPipeline(stages: d, checkpoints: FileCheckpointStore(directory: directory(), fingerprint: "fp"),
                                                    stopOnRetryableFailure: false).analyze(Self.videoURL)
        XCTAssertNil(detectLast.diagnostics["stopped_after"], "the detect stop is off too")
        XCTAssertNotNil(detectLast.diagnostics["stages"]?["players"])
    }

    /// With NoCheckpoints nothing is loaded or saved, and the stage still runs to the end.
    func testNoStoreMeansNoPartialEvents() async throws {
        let s = ResumableStages()
        let res = try await run(s, NoCheckpoints())
        XCTAssertEqual(res.players.count, Self.frames / 2)
        XCTAssertTrue(events(s, "players_partial_load").isEmpty)
        XCTAssertTrue(events(s, "players_partial_save").isEmpty)
    }
}
