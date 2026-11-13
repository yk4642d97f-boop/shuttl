import Foundation

/// Small boundary around the system-owned transfer, also used to exercise relaunch races in tests.
enum UploadFailure: Error, Codable, Equatable, Sendable {
    case http(Int), network(Int), cancelled, invalidResponse, journal
}

struct UploadTaskSnapshot: Sendable {
    let identifier: Int
    let generation: UUID?
    let suspended: Bool
    let fraction: Double
}

enum UploadEvent: Sendable {
    case progress(UUID?, Double)
    case completed(UUID?, Int?, UploadFailure?)
    case finishedEvents
}

typealias UploadEventSink = @Sendable (UploadEvent) async -> Void

protocol UploadTransport: Sendable {
    func allTasks() async -> [UploadTaskSnapshot]
    /// Creates a suspended task. The caller persists its generation before resume.
    func makeUpload(request: URLRequest, file: URL, generation: UUID) async throws -> Int
    func resume(_ identifier: Int) async
    func cancel(_ identifier: Int) async
}

/// URLSession is thread-safe. Session is assigned once during init; delegate-event chaining is locked.
/// Chaining preserves didComplete -> didFinishEvents order across the hop to Swift concurrency.
final class URLSessionUploadTransport: NSObject, UploadTransport, URLSessionTaskDelegate, @unchecked Sendable {
    private var session: URLSession!
    private let sink: UploadEventSink
    private let lock = NSLock()
    private var tail: Task<Void, Never>?

    init(configuration: URLSessionConfiguration, sink: @escaping UploadEventSink) {
        self.sink = sink
        super.init()
        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1
        queue.name = "shuttl.upload.events"
        session = URLSession(configuration: configuration, delegate: self, delegateQueue: queue)
    }

    func allTasks() async -> [UploadTaskSnapshot] {
        await session.allTasks.map { task in
            UploadTaskSnapshot(identifier: task.taskIdentifier, generation: task.taskDescription.flatMap(UUID.init(uuidString:)),
                               suspended: task.state == .suspended,
                               fraction: task.countOfBytesExpectedToSend > 0 ? min(1, Double(task.countOfBytesSent) / Double(task.countOfBytesExpectedToSend)) : 0)
        }
    }

    func makeUpload(request: URLRequest, file: URL, generation: UUID) async throws -> Int {
        let task = session.uploadTask(with: request, fromFile: file)
        task.taskDescription = generation.uuidString
        return task.taskIdentifier
    }

    func resume(_ identifier: Int) async {
        await session.allTasks.first { $0.taskIdentifier == identifier }?.resume()
    }

    func cancel(_ identifier: Int) async {
        await session.allTasks.first { $0.taskIdentifier == identifier }?.cancel()
    }

    func invalidate() { session.invalidateAndCancel() }

    private func emit(_ event: UploadEvent) {
        lock.withLock {
            let previous = tail
            let sink = sink
            tail = Task {
                await previous?.value
                await sink(event)
            }
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didSendBodyData bytesSent: Int64,
                    totalBytesSent: Int64, totalBytesExpectedToSend: Int64) {
        guard totalBytesExpectedToSend > 0 else { return }
        emit(.progress(task.taskDescription.flatMap(UUID.init(uuidString:)),
                       min(1, max(0, Double(totalBytesSent) / Double(totalBytesExpectedToSend)))))
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        let failure: UploadFailure? = error.map {
            let code = ($0 as NSError).code
            return code == URLError.cancelled.rawValue ? .cancelled : .network(code)
        }
        emit(.completed(task.taskDescription.flatMap(UUID.init(uuidString:)), (task.response as? HTTPURLResponse)?.statusCode, failure))
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) { emit(.finishedEvents) }
}
