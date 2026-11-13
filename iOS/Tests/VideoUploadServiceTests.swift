import XCTest
@testable import Shuttl

/// Models system-owned tasks surviving a coordinator relaunch, with ordered callbacks.
final class RetainedUploadTransport: UploadTransport, @unchecked Sendable {
    private let lock = NSLock()
    private var sink: UploadEventSink?
    private var live: [Int: UploadTaskSnapshot] = [:]
    private var nextID = 1
    private var responses: [Int] = []
    private var action: (@Sendable () -> Void)?
    private var made: [URLRequest] = []
    private var cancelled: [Int] = []
    private var resumed: [Int] = []
    func connect(_ sink: @escaping UploadEventSink) { lock.withLock { self.sink = sink } }
    func seed(_ snapshot: UploadTaskSnapshot) { lock.withLock { live[snapshot.identifier] = snapshot } }
    func replyWith(_ statuses: [Int]) { lock.withLock { responses = statuses } }
    func beforeReply(_ action: @escaping @Sendable () -> Void) { lock.withLock { self.action = action } }
    var requests: [URLRequest] { lock.withLock { made } }
    var cancelledIDs: [Int] { lock.withLock { cancelled } }
    var resumedIDs: [Int] { lock.withLock { resumed } }
    func allTasks() async -> [UploadTaskSnapshot] { lock.withLock { Array(live.values) } }
    func makeUpload(request: URLRequest, file: URL, generation: UUID) async throws -> Int {
        lock.withLock {
            let id = nextID; nextID += 1; made.append(request)
            live[id] = UploadTaskSnapshot(identifier: id, generation: generation, suspended: true, fraction: 0)
            return id
        }
    }
    func resume(_ identifier: Int) async {
        let reply: Int? = lock.withLock {
            resumed.append(identifier)
            return responses.isEmpty ? nil : responses.removeFirst()
        }
        if let reply {
            let action = lock.withLock { self.action }; action?()
            await complete(identifier, status: reply)
        }
    }
    func cancel(_ identifier: Int) async { lock.withLock { cancelled.append(identifier); live[identifier] = nil } }
    func complete(_ identifier: Int, status: Int) async {
        let generation = lock.withLock { live.removeValue(forKey: identifier)?.generation }
        await send(.completed(generation, status, nil))
    }
    func send(_ event: UploadEvent) async {
        let handler = lock.withLock { sink }; await handler?(event)
    }
}

@MainActor
final class VideoUploadServiceTests: XCTestCase {
    private var directory: URL!
    private var video: URL!
    private var journal: URL { directory.appendingPathComponent("state/receipts.json") }
    private let key = "stable-match-key"
    private let signedURL = URL(string: "https://uploads.example.invalid/video?token=DO-NOT-PERSIST")!
    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        video = directory.appendingPathComponent("fixture.mov")
        try Data(repeating: 7, count: 1024).write(to: video)
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: directory) }
    private func service(_ transport: RetainedUploadTransport) -> VideoUploadService {
        VideoUploadService(journalURL: journal, makeTransport: { sink in transport.connect(sink); return transport }, retryDelay: { _ in })
    }
    private func saved() throws -> [String: VideoUploadService.Record] {
        try JSONDecoder().decode([String: VideoUploadService.Record].self, from: Data(contentsOf: journal))
    }
    private func seed(state: VideoUploadService.Record.State = .running) throws -> UUID {
        let id = UUID()
        let stored = VideoUploadService.Record(generation: id, file: try .init(video), state: state)
        try FileManager.default.createDirectory(at: journal.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONEncoder().encode([key: stored]).write(to: journal)
        return id
    }
    private func eventually(_ condition: () -> Bool) async throws {
        for _ in 0..<100 {
            if condition() { return }; try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("Upload did not reach expected state"); throw URLError(.timedOut)
    }

    func testCompletedReceiptSurvivesRelaunchWithoutNewURLOrTransfer() async throws {
        let transport = RetainedUploadTransport(); transport.replyWith([204])
        let first = service(transport), url = signedURL
        try await first.upload(key: key, fileURL: video, target: { url }, progress: { _ in })
        XCTAssertEqual(try saved()[key]?.state, .succeeded)
        let nextTransport = RetainedUploadTransport(), restarted = service(nextTransport)
        try await restarted.upload(key: key, fileURL: video, target: { XCTFail("refetched completed upload"); return url }, progress: { _ in })
        XCTAssertTrue(nextTransport.requests.isEmpty)
        XCTAssertFalse(try String(contentsOf: journal, encoding: .utf8).contains("DO-NOT-PERSIST"))
    }
    func testRelaunchReattachesSuspendedTaskAndPersistsCompletion() async throws {
        let generation = try seed(), transport = RetainedUploadTransport()
        transport.seed(.init(identifier: 42, generation: generation, suspended: true, fraction: 0.4)); transport.replyWith([200])
        let uploader = service(transport), url = signedURL
        try await uploader.upload(key: key, fileURL: video, target: { XCTFail("reuploaded instead of reattaching"); return url }, progress: { _ in })
        XCTAssertTrue(transport.requests.isEmpty); XCTAssertEqual(transport.resumedIDs, [42])
        XCTAssertEqual(try saved()[key]?.state, .succeeded)
    }
    func testMissingSystemTaskRetriesInsteadOfInventingSuccess() async throws {
        _ = try seed()
        let transport = RetainedUploadTransport(); transport.replyWith([204])
        let uploader = service(transport), url = signedURL
        try await uploader.upload(key: key, fileURL: video, target: { url }, progress: { _ in })
        XCTAssertEqual(transport.requests.count, 1); XCTAssertEqual(try saved()[key]?.state, .succeeded)
    }
    func testConcurrentCallersShareOneTransfer() async throws {
        let transport = RetainedUploadTransport(), uploader = service(transport), url = signedURL
        let first = Task { try await uploader.upload(key: key, fileURL: video, target: { url }, progress: { _ in }) }
        let second = Task { try await uploader.upload(key: key, fileURL: video, target: { url }, progress: { _ in }) }
        try await eventually { transport.requests.count == 1 }
        await transport.complete(1, status: 204)
        try await first.value; try await second.value
        XCTAssertEqual(transport.requests.count, 1)
    }
    func testExpiredCredentialGetsFreshURL() async throws {
        let transport = RetainedUploadTransport(); transport.replyWith([403, 204])
        let uploader = service(transport), calls = Counter()
        try await uploader.upload(key: key, fileURL: video, target: {
            URL(string: "https://uploads.example.invalid/v?token=\(calls.next())")!
        }, progress: { _ in })
        XCTAssertEqual(transport.requests.count, 2)
        XCTAssertNotEqual(transport.requests[0].url, transport.requests[1].url)
    }
    func testHTTPFailureDoesNotBecomeSuccessAndRetriesAreBounded() async throws {
        let transport = RetainedUploadTransport(); transport.replyWith([503, 503, 503])
        let uploader = service(transport), url = signedURL
        do { try await uploader.upload(key: key, fileURL: video, target: { url }, progress: { _ in }); XCTFail("503 accepted") }
        catch { XCTAssertTrue(error is APIError) }
        XCTAssertEqual(transport.requests.count, 3); XCTAssertEqual(try saved()[key]?.state, .failed)
    }
    func testCancellationFencesLateCompletion() async throws {
        let transport = RetainedUploadTransport(), uploader = service(transport), url = signedURL
        let operation = Task { try await uploader.upload(key: key, fileURL: video, target: { url }, progress: { _ in }) }
        try await eventually { transport.resumedIDs.count == 1 }
        let generation = try XCTUnwrap(saved()[key]?.generation)
        operation.cancel()
        do { try await operation.value; XCTFail("cancelled upload succeeded") } catch is CancellationError {} catch { XCTFail("\(error)") }
        XCTAssertEqual(transport.cancelledIDs, [1])
        await transport.send(.completed(generation, 204, nil))
        XCTAssertEqual(try saved()[key]?.state, .cancelled)
    }
    func testChangedSourceCannotReuseReceipt() async throws {
        _ = try seed(state: .succeeded); try Data(repeating: 2, count: 2048).write(to: video)
        let transport = RetainedUploadTransport(), uploader = service(transport), url = signedURL
        do { try await uploader.upload(key: key, fileURL: video, target: { url }, progress: { _ in }); XCTFail("changed source accepted") }
        catch { XCTAssertTrue(error is APIError) }
        XCTAssertTrue(transport.requests.isEmpty)
    }
    func testServerMissingFileCanInvalidateReceipt() async throws {
        _ = try seed(state: .succeeded)
        let transport = RetainedUploadTransport(); transport.replyWith([204])
        let uploader = service(transport), url = signedURL
        try await uploader.forget(key: key)
        try await uploader.upload(key: key, fileURL: video, target: { url }, progress: { _ in })
        XCTAssertEqual(transport.requests.count, 1)
    }
    func testReceiptWriteFailureDoesNotHangOrReportSuccess() async throws {
        let transport = RetainedUploadTransport(); transport.replyWith([204])
        let parent = journal.deletingLastPathComponent()
        transport.beforeReply {
            try? FileManager.default.removeItem(at: parent); try? Data([1]).write(to: parent)
        }
        let uploader = service(transport), url = signedURL
        do { try await uploader.upload(key: key, fileURL: video, target: { url }, progress: { _ in }); XCTFail("lost receipt accepted") }
        catch { XCTAssertTrue(error is APIError) }
    }
    func testBackgroundAcknowledgementFollowsDurableReceipt() async throws {
        let generation = try seed(), transport = RetainedUploadTransport(), uploader = service(transport)
        transport.seed(.init(identifier: 42, generation: generation, suspended: false, fraction: 0.9))
        let acknowledged = expectation(description: "OS completion handler")
        await uploader.handleBackgroundEvents {
            XCTAssertEqual(try? self.saved()[self.key]?.state, .succeeded); acknowledged.fulfill()
        }
        await transport.complete(42, status: 204); await transport.send(.finishedEvents)
        await fulfillment(of: [acknowledged], timeout: 1)
    }
    func testCorruptJournalNeverAdoptsForeignTask() async throws {
        try FileManager.default.createDirectory(at: journal.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("corrupt".utf8).write(to: journal)
        let transport = RetainedUploadTransport(); transport.replyWith([204])
        transport.seed(.init(identifier: 42, generation: UUID(), suspended: false, fraction: 0.5))
        let uploader = service(transport), url = signedURL
        try await uploader.upload(key: key, fileURL: video, target: { url }, progress: { _ in })
        XCTAssertEqual(transport.cancelledIDs, [42]); XCTAssertEqual(transport.requests.count, 1)
    }

    func testSignOutCancelsRecoveredTasksWithoutResumingSuspendedUpload() async throws {
        let generation = try seed(), transport = RetainedUploadTransport()
        transport.seed(.init(identifier: 42, generation: generation, suspended: true, fraction: 0))
        let uploader = service(transport)
        await uploader.cancelAll()
        XCTAssertTrue(transport.resumedIDs.isEmpty)
        XCTAssertEqual(transport.cancelledIDs, [42])
        XCTAssertEqual(try saved()[key]?.state, .cancelled)
    }

    func testDiscardingAnotherMatchDoesNotStrandRecoveredSuspendedTask() async throws {
        let generation = try seed(), transport = RetainedUploadTransport()
        transport.seed(.init(identifier: 42, generation: generation, suspended: true, fraction: 0.4))
        let uploader = service(transport), url = signedURL
        try await uploader.forget(key: "another-match")
        XCTAssertTrue(transport.resumedIDs.isEmpty)
        transport.replyWith([204])
        try await uploader.upload(key: key, fileURL: video, target: { XCTFail("lost recovered task"); return url }, progress: { _ in })
        XCTAssertEqual(transport.resumedIDs, [42])
        XCTAssertTrue(transport.requests.isEmpty)
    }

    func testSystemCancellationRetriesButUserCancellationDoesNot() async throws {
        let transport = RetainedUploadTransport(), uploader = service(transport), url = signedURL
        let operation = Task { try await uploader.upload(key: key, fileURL: video, target: { url }, progress: { _ in }) }
        try await eventually { transport.resumedIDs.count == 1 }
        let generation = try XCTUnwrap(saved()[key]?.generation)
        transport.replyWith([204])
        await transport.send(.completed(generation, nil, .cancelled))
        try await operation.value
        XCTAssertEqual(transport.requests.count, 2)
        XCTAssertEqual(try saved()[key]?.state, .succeeded)
    }
}
