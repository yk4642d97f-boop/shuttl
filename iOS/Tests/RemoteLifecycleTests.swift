import XCTest
import SwiftData
@testable import Shuttl

actor LifecycleGate {
    private var entered = false
    private var waiter: CheckedContinuation<Void, Never>?
    func wait() async { entered = true; await withCheckedContinuation { waiter = $0 } }
    func hasEntered() -> Bool { entered }
    func release() { waiter?.resume(); waiter = nil }
}

@MainActor
final class RemoteLifecycleTests: XCTestCase {
    var directory: URL!
    var queue: RemoteCommandQueue!
    var backend: MockBackend!
    var account: RemoteAccount!
    var container: ModelContainer!
    var context: ModelContext { container.mainContext }
    var video: URL!

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("remote-lifecycle-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        queue = RemoteCommandQueue(url: directory.appendingPathComponent("pending.json"))
        backend = MockBackend()
        account = await backend.remoteAccount()
        container = try ModelContainer(for: Match.self, Rally.self, ShotPoint.self,
                                       configurations: ModelConfiguration(isStoredInMemoryOnly: true))
        video = directory.appendingPathComponent("recording.mov")
        try Data(repeating: 7, count: 2048).write(to: video)
    }
    override func tearDown() async throws { try? FileManager.default.removeItem(at: directory) }
    func pipeline(queue: RemoteCommandQueue? = nil) -> MatchPipeline {
        MatchPipeline(context: context, backend: backend, resumesAutomatically: false, remoteCommands: queue ?? self.queue)
    }
    func match(step: PipelineStep = .poll) throws -> Match {
        let match = Match(status: .analyzing, localVideoPath: video.path, cloudJobID: "old-job", pipelineStep: step)
        match.cloudAccount = account
        context.insert(match); try context.save()
        return match
    }
    func waitForGate(_ gate: LifecycleGate) async throws {
        for _ in 0..<500 {
            if await gate.hasEntered() { return }
            try await Task.sleep(for: .milliseconds(2))
        }
        XCTFail("operation never reached gate")
        throw APIError.timeout
    }

    func testOfflineDeleteSurvivesRestartAndOnlyOriginalAccountFlushes() async throws {
        let id = UUID()
        try await queue.enqueue(account: account, matchID: id, kind: .delete)
        await backend.set(commandHook: { throw URLError(.notConnectedToInternet) })
        await queue.flush(using: backend)
        let reloaded = RemoteCommandQueue(url: directory.appendingPathComponent("pending.json"))
        let pending = try await reloaded.pending(account: account)
        XCTAssertEqual(pending.map(\.failures), [1])
        await backend.set(account: RemoteAccount(baseURL: account.baseURL, userID: "different-user"))
        await reloaded.flush(using: backend)
        let first = await backend.remoteDeletes
        XCTAssertEqual(first, [id])
        await backend.set(account: RemoteAccount(baseURL: "http://another-server", userID: account.userID))
        await reloaded.flush(using: backend)
        let wrongServer = await backend.remoteDeletes
        XCTAssertEqual(wrongServer, [id])
        await backend.set(account: account)
        await backend.set(commandHook: {})
        await reloaded.flush(using: backend)
        let final = try await reloaded.pending(account: account)
        XCTAssertTrue(final.isEmpty)
        let disk = RemoteCommandQueue(url: directory.appendingPathComponent("pending.json"))
        let saved = await disk.count()
        XCTAssertEqual(saved, 0)
    }

    func testDeleteSupersedesCancelEvenWhileCancelIsInFlight() async throws {
        let id = UUID(), gate = LifecycleGate()
        try await queue.enqueue(account: account, matchID: id, kind: .cancel, jobID: "j1")
        await backend.set(commandHook: { await gate.wait() })
        let flush = Task { await queue.flush(using: backend) }
        try await waitForGate(gate)
        try await queue.enqueue(account: account, matchID: id, kind: .delete)
        try await queue.enqueue(account: account, matchID: id, kind: .cancel, jobID: "j2")
        await gate.release(); await flush.value
        let pending = try await queue.pending(account: account)
        XCTAssertEqual(pending.map(\.kind), [.delete])
        await backend.set(commandHook: {})
        await queue.flush(using: backend)
        let deletes = await backend.remoteDeletes, cancels = await backend.remoteCancels
        XCTAssertEqual(deletes, [id]); XCTAssertEqual(cancels, ["j1"])
    }

    func testConcurrentFlushDoesNotDuplicateRequest() async throws {
        let gate = LifecycleGate(), id = UUID()
        try await queue.enqueue(account: account, matchID: id, kind: .delete)
        await backend.set(commandHook: { await gate.wait() })
        let flush = Task { await queue.flush(using: backend) }
        try await waitForGate(gate)
        await queue.flush(using: backend)
        await gate.release(); await flush.value
        let deletes = await backend.remoteDeletes
        XCTAssertEqual(deletes, [id])
    }

    func testCorruptQueueIsPreservedAndReported() async throws {
        let file = directory.appendingPathComponent("pending.json")
        let corrupt = Data("incomplete file".utf8)
        try corrupt.write(to: file)
        let broken = RemoteCommandQueue(url: file)
        do { try await broken.enqueue(account: account, matchID: UUID(), kind: .delete); XCTFail("must not overwrite") }
        catch {}
        await broken.flush(using: backend)
        let error = await broken.storageErrorMessage()
        XCTAssertNotNil(error)
        XCTAssertEqual(try Data(contentsOf: file), corrupt)
    }

    func testQueueWriteFailureKeepsLocalRecordingAndRecord() async throws {
        let parent = directory.appendingPathComponent("not-a-directory")
        try Data([1]).write(to: parent)
        let broken = RemoteCommandQueue(url: parent.appendingPathComponent("pending.json"))
        let p = pipeline(queue: broken), m = try match()
        do { try await p.discard(m); XCTFail("must keep local data") } catch {}
        XCTAssertTrue(FileManager.default.fileExists(atPath: video.path))
        XCTAssertEqual(try context.fetchCount(FetchDescriptor<Match>()), 1)
        let deletes = await backend.remoteDeletes
        XCTAssertTrue(deletes.isEmpty)
    }

    func testCreateInFlightDeletionUsesPersistedClientID() async throws {
        let gate = LifecycleGate()
        await backend.set(createHook: { id in
            await gate.wait() // Deliberately emulate a transport that finishes despite Task cancellation.
            return MockBackend.remote(id, status: "uploading")
        })
        let p = pipeline()
        let m = try p.ingestRecording(fileURL: video, durationSeconds: 12, courtType: .doubles)
        let clientID = m.id
        try await waitForGate(gate)
        XCTAssertEqual(m.cloudAccount, account)
        try await p.discard(m)
        context.delete(m); try context.save()
        await p.syncRemoteCommands()
        await gate.release()
        try await Task.sleep(for: .milliseconds(20))
        let deletes = await backend.remoteDeletes, calls = await backend.calls
        XCTAssertEqual(deletes, [clientID])
        XCTAssertEqual(calls, ["create"], "late create completion must not upload or analyze")
        XCTAssertFalse(FileManager.default.fileExists(atPath: video.path))
    }

    func testPendingCancelPreventsNewAnalysisUntilAcknowledged() async throws {
        let p = pipeline(), m = try match()
        await backend.set(commandHook: { throw URLError(.notConnectedToInternet) })
        try await p.cancelRemotely(m)
        XCTAssertEqual(m.status, .failed)
        p.retry(m); await p.waitUntilIdle()
        let blockedCalls = await backend.calls
        XCTAssertFalse(blockedCalls.contains("analyze"))
        XCTAssertTrue(m.failureReason?.contains("同步") == true)
        await backend.set(commandHook: {})
        p.retry(m); await p.waitUntilIdle()
        XCTAssertEqual(m.status, .done)
        let cancels = await backend.remoteCancels
        XCTAssertTrue(cancels.allSatisfy { $0 == "old-job" })
        let completedCalls = await backend.calls
        XCTAssertEqual(completedCalls.filter { $0 == "analyze" }.count, 1)
    }

    func testLegacy404NeverReuploadsUnderNewLogin() async throws {
        let p = pipeline(), m = try match(step: .upload)
        m.cloudAccount = nil
        await backend.set(getMatch: { _ in throw APIError.serverError(404, "missing") })
        p.start(m); await p.waitUntilIdle()
        XCTAssertEqual(m.status, .failed)
        XCTAssertTrue(m.failureReason?.contains("原账号") == true)
        let calls = await backend.calls
        XCTAssertEqual(calls, ["getMatch"])
        XCTAssertTrue(FileManager.default.fileExists(atPath: video.path))
    }

    func testKnownOtherAccountCannotStartUpload() async throws {
        let p = pipeline(), m = try match(step: .upload)
        await backend.set(account: RemoteAccount(baseURL: account.baseURL, userID: "another"))
        p.start(m); await p.waitUntilIdle()
        let calls = await backend.calls
        XCTAssertTrue(calls.isEmpty)
        XCTAssertTrue(m.failureReason?.contains("原账号") == true)
    }

    func testDeleteQueuesForOriginalAccountWhileLoggedOut() async throws {
        let p = pipeline(), m = try match()
        await backend.set(account: nil)
        try await p.discard(m)
        let pending = try await queue.pending(account: account)
        XCTAssertEqual(pending.map(\.matchID), [m.id])
        XCTAssertFalse(FileManager.default.fileExists(atPath: video.path))
    }

    func testLocalOnlyDeletionDoesNotEnqueueOrCallBackend() async throws {
        let p = MatchPipeline(context: context, backend: backend, analyzer: FakeAnalyzer(),
                              resumesAutomatically: false, remoteCommands: queue)
        let m = Match(status: .failed, localVideoPath: video.path, pipelineStep: .local)
        context.insert(m); try context.save()
        try await p.discard(m)
        let pending = await queue.count(), calls = await backend.calls, deletes = await backend.remoteDeletes
        XCTAssertEqual(pending, 0); XCTAssertTrue(calls.isEmpty); XCTAssertTrue(deletes.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: video.path))
    }

    func testDeleteBeforeFirstCreateNeedsNoAccountOrRemoteCommand() async throws {
        let p = pipeline()
        let m = try p.ingestRecording(fileURL: video, durationSeconds: 12, courtType: .singles)
        try await p.discard(m)
        let pending = await queue.count(), calls = await backend.calls
        XCTAssertEqual(pending, 0); XCTAssertTrue(calls.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: video.path))
    }

    func testOldJobCannotBeCancelledWhileNewTriggerIsPending() throws {
        let p = pipeline(), m = try match(step: .analyze)
        XCTAssertFalse(p.canCancelRemotely(m))
        m.pipelineStep = .poll
        XCTAssertTrue(p.canCancelRemotely(m))
    }

    func testPreLifecycleStoreMigratesWithoutLosingVideoOrRallies() throws {
        let store = directory.appendingPathComponent("legacy.store"), id = UUID()
        do {
            let legacy = try ModelContainer(for: LifecycleBeforeCloudSchema.Match.self,
                LifecycleBeforeCloudSchema.Rally.self, LifecycleBeforeCloudSchema.ShotPoint.self,
                configurations: ModelConfiguration(url: store))
            let m = LifecycleBeforeCloudSchema.Match(id: id, status: .done, localVideoPath: video.path,
                                                       cloudJobID: "legacy-job", courtType: .doubles)
            let rally = LifecycleBeforeCloudSchema.Rally(startFrame: 10, endFrame: 30, winner: .near)
            rally.shots = [LifecycleBeforeCloudSchema.ShotPoint(frameIndex: 20, xPx: 100, yPx: 200, shotType: .smash)]
            m.rallies = [rally]
            legacy.mainContext.insert(m); try legacy.mainContext.save()
        }
        let migrated = try ModelContainer(for: Match.self, Rally.self, ShotPoint.self,
                                          configurations: ModelConfiguration(url: store))
        let rows = try migrated.mainContext.fetch(FetchDescriptor<Match>())
        let m = try XCTUnwrap(rows.first)
        XCTAssertEqual(rows.count, 1); XCTAssertEqual(m.id, id)
        XCTAssertEqual(m.localVideoPath, video.path); XCTAssertEqual(m.cloudJobID, "legacy-job")
        XCTAssertEqual(m.courtType, .doubles); XCTAssertNil(m.cloudAccount); XCTAssertNil(m.cloudCreateRequested)
        XCTAssertEqual(m.rallies.first?.shots.first?.shotType, .smash)
        XCTAssertEqual(m.rallies.first?.winner, .near)
    }

    func testAccountFieldsSurviveSwiftDataStoreReopen() throws {
        let store = directory.appendingPathComponent("matches.store")
        let id = UUID()
        do {
            let disk = try ModelContainer(for: Match.self, Rally.self, ShotPoint.self,
                                          configurations: ModelConfiguration(url: store))
            let m = Match(id: id); m.cloudAccount = account
            disk.mainContext.insert(m); try disk.mainContext.save()
        }
        let reopened = try ModelContainer(for: Match.self, Rally.self, ShotPoint.self,
                                         configurations: ModelConfiguration(url: store))
        let rows = try reopened.mainContext.fetch(FetchDescriptor<Match>())
        XCTAssertEqual(rows.map(\.id), [id]); XCTAssertEqual(rows.first?.cloudAccount, account)
    }
}
