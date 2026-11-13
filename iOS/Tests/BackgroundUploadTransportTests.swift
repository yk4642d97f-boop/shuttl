import XCTest
import Network
import CryptoKit
@testable import Shuttl

/// Loopback-only HTTP target for the real iOS background URLSession adapter.
private final class UploadHTTPServer: @unchecked Sendable {
    private let queue = DispatchQueue(label: "shuttl.test.upload.http")
    private let listener: NWListener
    private let lock = NSLock()
    private var body: Data?
    private var count = 0
    private var connections: [NWConnection] = []
    var received: Data? { lock.withLock { body } }
    var requestCount: Int { lock.withLock { count } }

    init(ready: @escaping @Sendable (UInt16) -> Void) throws {
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: .ipv4(.loopback), port: .any)
        listener = try NWListener(using: parameters)
        listener.stateUpdateHandler = { [weak self] state in
            if case .ready = state, let port = self?.listener.port { ready(port.rawValue) }
        }
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { return }
            self.connections.append(connection)
            connection.start(queue: self.queue)
            self.read(connection, buffer: Data(), length: nil)
        }
        listener.start(queue: queue)
    }
    func stop() {
        listener.cancel()
        queue.async { for connection in self.connections { connection.cancel() } }
    }
    private func read(_ connection: NWConnection, buffer: Data, length: Int?) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { [weak self] bytes, _, ended, error in
            guard let self, error == nil else { return }
            var data = buffer; if let bytes { data.append(bytes) }
            var wanted = length
            if wanted == nil, let separator = data.range(of: Data("\r\n\r\n".utf8)) {
                let header = String(decoding: data[..<separator.lowerBound], as: UTF8.self)
                guard header.hasPrefix("PUT /video "),
                      let size = header.components(separatedBy: "\r\n").first(where: { $0.lowercased().hasPrefix("content-length:") })?.split(separator: ":").last,
                      let value = Int(size.trimmingCharacters(in: .whitespaces)), value <= 2 * 1024 * 1024 else { connection.cancel(); return }
                wanted = value
                data = Data(data[separator.upperBound...])
                if header.lowercased().contains("expect: 100-continue") {
                    connection.send(content: Data("HTTP/1.1 100 Continue\r\n\r\n".utf8), completion: .contentProcessed { _ in })
                }
            }
            if let wanted, data.count >= wanted {
                self.lock.withLock { self.body = data.prefix(wanted); self.count += 1 }
                connection.send(content: Data("HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".utf8),
                                completion: .contentProcessed { _ in connection.cancel() })
            } else if !ended { self.read(connection, buffer: data, length: wanted) }
        }
    }
}

private final class UploadTransportHolder: @unchecked Sendable {
    private let lock = NSLock()
    private var value: URLSessionUploadTransport?
    func set(_ value: URLSessionUploadTransport) { lock.withLock { self.value = value } }
    func stop() { lock.withLock { value?.invalidate() } }
}

private final class UploadTestURL: @unchecked Sendable {
    private let lock = NSLock()
    private var value: URL?
    func set(_ value: URL) { lock.withLock { self.value = value } }
    func get() -> URL? { lock.withLock { value } }
}

@MainActor
final class BackgroundUploadTransportTests: XCTestCase {
    func testRealBackgroundSessionStreamsFileAndStoresReceipt() async throws {
        let ready = expectation(description: "loopback HTTP ready"), target = UploadTestURL()
        let server = try UploadHTTPServer { port in
            target.set(URL(string: "http://127.0.0.1:\(port)/video")!); ready.fulfill()
        }
        defer { server.stop() }
        await fulfillment(of: [ready], timeout: 5)
        let url = try XCTUnwrap(target.get())
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("fixture.mov"), journal = directory.appendingPathComponent("state/receipts.json")
        let bytes = Data((0..<(256 * 1024)).map { UInt8($0 % 251) })
        try bytes.write(to: file)
        let holder = UploadTransportHolder(), identifier = "com.example.shuttl.test.upload.\(UUID())"
        defer { holder.stop() }
        let uploader = VideoUploadService(journalURL: journal, makeTransport: { sink in
            let config = URLSessionConfiguration.background(withIdentifier: identifier)
            config.isDiscretionary = false
            config.timeoutIntervalForResource = 20
            let transport = URLSessionUploadTransport(configuration: config, sink: sink)
            holder.set(transport); return transport
        }, retryDelay: { _ in })
        let finished = expectation(description: "background upload completed")
        let operation = Task {
            do { try await uploader.upload(key: "loopback", fileURL: file, target: { url }, progress: { _ in }) }
            catch { XCTFail("Background upload failed: \(error)") }
            finished.fulfill()
        }
        await fulfillment(of: [finished], timeout: 30)
        operation.cancel()
        await uploader.cancelAll()
        await operation.value
        XCTAssertEqual(server.requestCount, 1)
        XCTAssertEqual(server.received.map { Data(SHA256.hash(data: $0)) }, Data(SHA256.hash(data: bytes)))
        let records = try JSONDecoder().decode([String: VideoUploadService.Record].self, from: Data(contentsOf: journal))
        XCTAssertEqual(records["loopback"]?.state, .succeeded)
    }
}
