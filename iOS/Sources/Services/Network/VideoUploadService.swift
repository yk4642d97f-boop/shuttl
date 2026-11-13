import Foundation
import CryptoKit

/// One stable session lets iOS retain file uploads while the app is suspended or relaunched.
/// Receipts are written before waking the pipeline. No signed URL/token is stored in the journal.
actor VideoUploadService {
    static let backgroundIdentifier = "com.example.shuttl.video-upload.v1"
    static let shared = VideoUploadService()
    typealias Progress = @Sendable (Double) -> Void
    typealias Target = @Sendable () async throws -> URL

    struct FileIdentity: Codable, Equatable, Sendable {
        let path: String
        let size: Int64
        let modified: Date

        init(_ url: URL) throws {
            let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
            guard let size = attributes[.size] as? NSNumber, size.int64Value > 0,
                  let modified = attributes[.modificationDate] as? Date else {
                throw APIError.uploadFailed("本地视频不存在或为空")
            }
            self.path = VideoStore.storedPath(for: url)
            self.size = size.int64Value
            self.modified = modified
        }
    }

    struct Record: Codable, Sendable {
        enum State: String, Codable { case running, succeeded, failed, cancelled }
        let generation: UUID
        let file: FileIdentity
        var state: State
        var failure: UploadFailure?
    }

    private struct Operation {
        let token: UUID
        let file: FileIdentity
        let task: Task<Void, Error>
    }

    private let journalURL: URL
    private let makeTransport: @Sendable (@escaping UploadEventSink) -> any UploadTransport
    private let retryDelay: @Sendable (Int) async throws -> Void
    private var transportInstance: (any UploadTransport)?
    private var records: [String: Record] = [:]
    private var loadError: Error?
    private var restoration: Task<[UploadTaskSnapshot], Never>?
    private var restored = false
    private var tasks: [UUID: Int] = [:]
    private var operations: [String: Operation] = [:]
    private var completions: [UUID: CheckedContinuation<Void, Error>] = [:]
    private var observers: [String: [UUID: Progress]] = [:]
    private var fractions: [String: Double] = [:]
    private var backgroundCompletion: (@MainActor @Sendable () -> Void)?
    private var finishedBackgroundEvents = false

    init(journalURL: URL? = nil,
         makeTransport: @escaping @Sendable (@escaping UploadEventSink) -> any UploadTransport = { sink in
             URLSessionUploadTransport(configuration: VideoUploadService.configuration(), sink: sink)
         },
         retryDelay: @escaping @Sendable (Int) async throws -> Void = { attempt in
             try await Task.sleep(for: .seconds(attempt == 0 ? 1 : 3))
         }) {
        self.journalURL = journalURL ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Uploads/receipts-v1.json")
        self.makeTransport = makeTransport
        self.retryDelay = retryDelay
        do {
            if FileManager.default.fileExists(atPath: self.journalURL.path) {
                records = try JSONDecoder().decode([String: Record].self, from: Data(contentsOf: self.journalURL))
            }
        } catch { loadError = error }
    }

    nonisolated static func configuration() -> URLSessionConfiguration {
        let config = URLSessionConfiguration.background(withIdentifier: backgroundIdentifier)
        config.sessionSendsLaunchEvents = true
        config.isDiscretionary = false
        config.waitsForConnectivity = true
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 24 * 60 * 60
        config.httpMaximumConnectionsPerHost = 2
        return config
    }

    /// Scope is the API origin/path plus server match UUID, not the expiring upload URL.
    nonisolated static func key(scope: String, matchID: UUID) -> String {
        SHA256.hash(data: Data("\(scope)|\(matchID.uuidString.lowercased())".utf8))
            .map { String(format: "%02x", $0) }.joined()
    }

    private func transport() -> any UploadTransport {
        if let transportInstance { return transportInstance }
        let instance = makeTransport { [weak self] event in await self?.receive(event) }
        transportInstance = instance
        return instance
    }

    private func restore(resumeSuspended: Bool = true) async throws {
        guard !restored else { return }
        let transport = transport()
        let pending = restoration ?? Task { await transport.allTasks() }
        restoration = pending
        let live = await pending.value
        guard !restored else { return }
        // A corrupt journal is never treated as a successful upload. Cancel unknown work,
        // retain the damaged file for diagnosis, then allow a safe fresh upload.
        if loadError != nil {
            try FileManager.default.moveItem(at: journalURL, to: journalURL.appendingPathExtension("corrupt-\(UUID())"))
            records = [:]
            loadError = nil
        }
        var cancelIDs: [Int] = []
        var resumeIDs: [Int] = []
        for task in live {
            guard let generation = task.generation,
                  let (key, record) = records.first(where: { $0.value.generation == generation }),
                  record.state == .running else {
                cancelIDs.append(task.identifier)
                continue
            }
            tasks[generation] = task.identifier
            fractions[key] = task.fraction
            if resumeSuspended && task.suspended { resumeIDs.append(task.identifier) }
        }
        restored = true
        for identifier in cancelIDs { await transport.cancel(identifier) }
        for identifier in resumeIDs { await transport.resume(identifier) }
    }

    /// Coalesces callers and first reconnects to an existing system task or completion receipt.
    /// A new signed target is fetched only when a new attempt is actually needed.
    func upload(key: String, fileURL: URL, target: @escaping Target, progress: @escaping Progress) async throws {
        try Task.checkCancellation()
        try await restore()
        try Task.checkCancellation()
        let file = try FileIdentity(fileURL)
        if let record = records[key], record.file != file {
            throw APIError.uploadFailed("录像已发生变化，请创建新比赛后上传")
        }
        if records[key]?.state == .succeeded { progress(1); return }
        let operation: Operation
        if let current = operations[key] {
            guard current.file == file else { throw APIError.uploadFailed("该比赛正在上传另一段录像") }
            operation = current
        } else {
            let token = UUID()
            let task = Task { try await self.run(key: key, file: file, fileURL: fileURL, target: target) }
            operation = Operation(token: token, file: file, task: task)
            operations[key] = operation
        }
        let observerID = UUID()
        observers[key, default: [:]][observerID] = progress
        progress(fractions[key] ?? 0)
        defer {
            observers[key]?[observerID] = nil
            if operations[key]?.token == operation.token { operations[key] = nil }
        }
        try await withTaskCancellationHandler {
            try await operation.task.value
            try Task.checkCancellation()
        } onCancel: {
            Task { await self.cancel(key: key, token: operation.token) }
        }
    }

    private func run(key: String, file: FileIdentity, fileURL: URL, target: @escaping Target) async throws {
        for attempt in 0..<3 {
            do {
                try Task.checkCancellation()
                if let record = records[key], record.state == .succeeded { return }
                if let record = records[key], record.state == .running, let identifier = tasks[record.generation] {
                    // A prior discard/logout reconciliation may deliberately leave tasks suspended.
                    await transport().resume(identifier)
                    try await wait(for: record.generation)
                    return
                }
                let url = try await target()
                try Task.checkCancellation()
                guard try FileIdentity(fileURL) == file else {
                    throw APIError.uploadFailed("录像已发生变化，请创建新比赛后上传")
                }
                var request = URLRequest(url: url)
                request.httpMethod = "PUT"
                request.setValue("video/quicktime", forHTTPHeaderField: "Content-Type")
                request.setValue(String(file.size), forHTTPHeaderField: "Content-Length")
                let generation = UUID()
                let taskID = try await transport().makeUpload(request: request, file: fileURL, generation: generation)
                do {
                    try Task.checkCancellation()
                    var updated = records
                    updated[key] = Record(generation: generation, file: file, state: .running)
                    try persist(updated) // The system task is still suspended at this point.
                } catch {
                    await transport().cancel(taskID)
                    throw error
                }
                tasks[generation] = taskID
                fractions[key] = 0
                await transport().resume(taskID)
                try await wait(for: generation)
                return
            } catch is CancellationError { throw CancellationError() }
            catch {
                guard attempt < 2, Self.isTransient(error) else { throw Self.surface(error) }
                try await retryDelay(attempt)
            }
        }
    }

    private func wait(for generation: UUID) async throws {
        try Task.checkCancellation()
        // The completion callback can arrive before the pipeline subscribes.
        guard let record = records.values.first(where: { $0.generation == generation }) else {
            throw CancellationError()
        }
        if record.state == .succeeded { return }
        if record.state != .running { throw record.failure ?? UploadFailure.cancelled }
        try await withCheckedThrowingContinuation { completions[generation] = $0 }
    }

    private func persist(_ updated: [String: Record]) throws {
        let directory = journalURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var resource = directory
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try resource.setResourceValues(values)
        try JSONEncoder().encode(updated).write(to: journalURL, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        records = updated
    }

    private func cancel(key: String, token: UUID) async {
        guard let operation = operations[key], operation.token == token else { return }
        operation.task.cancel()
        await cancelAttempt(key: key)
    }

    private func cancelAttempt(key: String) async {
        guard var record = records[key], record.state == .running else { return }
        record.state = .cancelled
        record.failure = .cancelled
        var updated = records
        updated[key] = record
        do { try persist(updated) } catch { records = updated }
        completions.removeValue(forKey: record.generation)?.resume(throwing: CancellationError())
        if let identifier = tasks.removeValue(forKey: record.generation) { await transport().cancel(identifier) }
    }

    /// Called only after the server explicitly reports missing input, or the match is discarded.
    func forget(key: String) async throws {
        try await restore(resumeSuspended: false)
        operations[key]?.task.cancel()
        await cancelAttempt(key: key)
        var updated = records
        updated[key] = nil
        try persist(updated)
        fractions[key] = nil
    }

    func cancelAll() async {
        for operation in operations.values { operation.task.cancel() }
        try? await restore(resumeSuspended: false)
        for key in Array(records.keys) { await cancelAttempt(key: key) }
    }

    func handleBackgroundEvents(completion: @escaping @MainActor @Sendable () -> Void) async {
        backgroundCompletion = completion
        do { try await restore() } catch { /* Still release iOS's finite event delivery time. */ }
        await finishBackgroundEventsIfReady()
    }

    private func finishBackgroundEventsIfReady() async {
        guard finishedBackgroundEvents, let completion = backgroundCompletion else { return }
        backgroundCompletion = nil
        finishedBackgroundEvents = false
        await completion()
    }

    private func receive(_ event: UploadEvent) async {
        switch event {
        case .finishedEvents:
            finishedBackgroundEvents = true
            await finishBackgroundEventsIfReady()
        case let .progress(generation, fraction):
            guard let generation, let key = records.first(where: { $0.value.generation == generation && $0.value.state == .running })?.key else { return }
            fractions[key] = fraction
            for callback in observers[key]?.values ?? [:].values { callback(fraction) }
        case let .completed(generation, status, failure):
            guard let generation, let key = records.first(where: { $0.value.generation == generation && $0.value.state == .running })?.key else { return }
            tasks[generation] = nil
            let verdict = failure ?? (status.map { (200...299).contains($0) ? nil : UploadFailure.http($0) } ?? .invalidResponse)
            var updated = records
            updated[key]?.state = verdict == nil ? .succeeded : .failed
            updated[key]?.failure = verdict
            do {
                try persist(updated)
                if let verdict {
                    completions.removeValue(forKey: generation)?.resume(throwing: verdict)
                } else {
                    fractions[key] = 1
                    for callback in observers[key]?.values ?? [:].values { callback(1) }
                    completions.removeValue(forKey: generation)?.resume()
                }
            } catch {
                updated[key]?.state = .failed
                updated[key]?.failure = .journal
                records = updated
                completions.removeValue(forKey: generation)?.resume(throwing: UploadFailure.journal)
            }
        }
    }

    nonisolated static func isTransient(_ error: Error) -> Bool {
        guard let failure = error as? UploadFailure else { return false }
        switch failure {
        case .http(let code): return code >= 500 || [401, 403, 408, 429].contains(code)
        case .network, .cancelled: return true // An OS interruption; explicit user cancellation throws CancellationError.
        default: return false
        }
    }

    nonisolated private static func surface(_ error: Error) -> Error {
        guard let failure = error as? UploadFailure else { return error }
        switch failure {
        case .cancelled: return APIError.uploadFailed("上传被系统中断，请重试")
        case .network(let code):
            return code == URLError.timedOut.rawValue ? APIError.timeout : APIError.networkError(URLError(URLError.Code(rawValue: code)))
        case .http(413): return APIError.uploadFailed("视频文件过大，服务器拒收")
        case .http(401), .http(403): return APIError.uploadFailed("上传凭证已失效，请重试")
        case .http(409): return APIError.uploadFailed("该比赛已上传其他录像，请创建新比赛")
        case .http(let code): return APIError.uploadFailed("上传服务器返回 \(code)，请重试")
        case .invalidResponse: return APIError.uploadFailed("服务器未返回有效响应")
        case .journal: return APIError.uploadFailed("无法保存上传状态，请腾出手机存储空间后重试")
        }
    }
}
