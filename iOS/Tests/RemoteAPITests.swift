import XCTest
import ShuttlCore
@testable import Shuttl

private final class LifecycleCredentials: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [String: String] = [:]
    func read(_ key: String) -> String? { lock.lock(); defer { lock.unlock() }; return values[key] }
    func write(_ key: String, _ value: String) { lock.lock(); defer { lock.unlock() }; values[key] = value }
    func delete(_ key: String) { lock.lock(); defer { lock.unlock() }; values[key] = nil }
}
private final class LifecycleHTTPHandler: @unchecked Sendable {
    private let lock = NSLock()
    private var handler: (@Sendable (URLRequest) throws -> (Int, String))?
    func set(_ handler: @escaping @Sendable (URLRequest) throws -> (Int, String)) {
        lock.lock(); defer { lock.unlock() }; self.handler = handler
    }
    func handle(_ request: URLRequest) throws -> (Int, String) {
        lock.lock(); let handler = handler; lock.unlock()
        guard let handler else { throw URLError(.badServerResponse) }
        return try handler(request)
    }
}
private final class LifecycleURLProtocol: URLProtocol, @unchecked Sendable {
    static let handler = LifecycleHTTPHandler()
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        do {
            let (status, body) = try Self.handler.handle(request)
            let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: "HTTP/1.1", headerFields: ["Content-Type": "application/json"])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Data(body.utf8))
            client?.urlProtocolDidFinishLoading(self)
        } catch { client?.urlProtocol(self, didFailWithError: error) }
    }
    override func stopLoading() {}
}

final class RemoteAPITests: XCTestCase {
    func testNativeCloudArchiveSurvivesRefreshAndKeepsSnakeCaseKeys() async throws {
        let file = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/cloud_archive.json")
        let fixture = try String(contentsOf: file, encoding: .utf8)
        let oldToken = token!, calls = Counter()
        LifecycleURLProtocol.handler.set { request in
            let n = calls.next()
            if request.url!.path == "/auth/refresh" {
                return (200, "{\"access_token\":\"\(oldToken)\",\"refresh_token\":\"\(oldToken)\"}")
            }
            XCTAssertTrue(request.url!.path.hasSuffix("/analysis/archive"))
            return n == 1 ? (401, "{}") : (200, fixture)
        }
        let fetched = try await api.getAnalysisArchive(matchID: UUID(), account: account)
        let archive = try XCTUnwrap(fetched)
        XCTAssertEqual(archive.meta.nFrames, 100)
        XCTAssertEqual(archive.camera?.imageSize.width, 1920)
        XCTAssertEqual(archive.players.first?.trackID, 7)
        XCTAssertEqual(archive.players.first?.footWorld?.x, -4)
        XCTAssertTrue(archive.track!.xy[0].x.isNaN)
        XCTAssertEqual(archive.rallies[0].shots[0].shotType, .kill)
        XCTAssertEqual(archive.rallies[0].shots[0].segment!.frames.count, 31)
        XCTAssertEqual(archive.diagnostics["cloud_round_trip"], .bool(true))
    }

    func testCloudRequestPreservesCameraCornersAndStableRetryKey() async throws {
        let file = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/cloud_archive.json")
        let archive = try ShuttlJSON.decoder().decode(MatchAnalysis.self, from: Data(contentsOf: file))
        let id = UUID()
        let inputs = MatchInputs(manualCorners: ["nb_ld": [0.1, 0.9]], cameraHint: archive.camera, cloudReanalysisID: id)
        LifecycleURLProtocol.handler.set { request in
            var data = request.httpBody ?? Data()
            if let stream = request.httpBodyStream {
                stream.open(); defer { stream.close() }
                var bytes = [UInt8](repeating: 0, count: 4096)
                while stream.hasBytesAvailable {
                    let n = stream.read(&bytes, maxLength: bytes.count)
                    if n <= 0 { break }; data.append(contentsOf: bytes.prefix(n))
                }
            }
            let body = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let camera = try XCTUnwrap(body["camera"] as? [String: Any])
            XCTAssertNotNil(camera["K"]); XCTAssertNil(camera["k"])
            XCTAssertNotNil(camera["image_size"])
            XCTAssertEqual(body["force"] as? Bool, true)
            XCTAssertEqual(body["corners_normalized"] as? Bool, true)
            XCTAssertEqual((body["corners"] as? [String: [Double]])?["nb_ld"], [0.1, 0.9])
            XCTAssertEqual(request.value(forHTTPHeaderField: "Idempotency-Key"), id.uuidString.lowercased())
            return (200, "{\"job_id\":\"job-1\",\"status\":\"queued\"}")
        }
        _ = try await api.triggerAnalysis(matchID: UUID(), doubles: false, account: account, inputs: inputs)
    }

    func testCloudPolicyRequiresExplicitHTTPSConfiguration() {
        XCTAssertTrue(CloudConfiguration.acceptsCloud(enabled: true, endpoint: "https://service.example.com:8443"))
        XCTAssertFalse(CloudConfiguration.acceptsCloud(enabled: true, endpoint: "http://192.168.1.1:8000"))
        XCTAssertFalse(CloudConfiguration.acceptsCloud(enabled: false, endpoint: "https://example.com"))
        XCTAssertFalse(CloudConfiguration.acceptsCloud(enabled: true, endpoint: "https://user:secret@example.com"))
    }

    private let origin = "http://lifecycle.test"
    private var credentials: LifecycleCredentials!
    private var session: URLSession!
    private var api: APIClient!
    private var account: RemoteAccount!
    private var token: String!
    private static func token(_ id: UUID) -> String {
        let data = Data("{\"sub\":\"\(id.uuidString)\"}".utf8)
        let payload = data.base64EncodedString().replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_").replacingOccurrences(of: "=", with: "")
        return "header.\(payload).signature"
    }
    override func setUp() {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [LifecycleURLProtocol.self]
        session = URLSession(configuration: config)
        let store = LifecycleCredentials()
        credentials = store
        token = Self.token(UUID())
        store.write("access_token", token); store.write("refresh_token", token)
        api = APIClient(baseURL: origin, session: session, tokenRead: { store.read($0) },
                        tokenWrite: { store.write($0, $1) }, tokenDelete: { store.delete($0) })
        account = APIClient.account(token: token, baseURL: origin)
    }
    override func tearDown() { session.invalidateAndCancel() }

    func testDifferentAccountIsRejectedBeforeAnyHTTP() async throws {
        let calls = Counter()
        LifecycleURLProtocol.handler.set { _ in _ = calls.next(); return (202, "{\"status\":\"deleted\"}") }
        let oldAccount = RemoteAccount(baseURL: origin, userID: UUID().uuidString.lowercased())
        do { try await api.deleteMatch(matchID: UUID(), account: oldAccount); XCTFail("wrong account sent") }
        catch { let isAuth = await MatchPipeline.isAuthFailure(error); XCTAssertTrue(isAuth) }
        XCTAssertEqual(calls.next(), 1, "counter should have no HTTP calls")
    }

    func testAccountChangeDuring401DoesNotRefreshNewLogin() async throws {
        let store = credentials!, next = Self.token(UUID()), oldToken = token!, calls = Counter()
        LifecycleURLProtocol.handler.set { request in
            XCTAssertEqual(calls.next(), 1)
            XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer \(oldToken)")
            store.write("access_token", next); store.write("refresh_token", next)
            return (401, "{}")
        }
        do { try await api.deleteMatch(matchID: UUID(), account: account); XCTFail("must retain intent") }
        catch { let isAuth = await MatchPipeline.isAuthFailure(error); XCTAssertTrue(isAuth) }
        XCTAssertEqual(store.read("access_token"), next)
        XCTAssertEqual(calls.next(), 2)
    }

    func testRefreshCompletionCannotOverwriteDifferentLogin() async throws {
        let store = credentials!, next = Self.token(UUID()), oldToken = token!, calls = Counter()
        LifecycleURLProtocol.handler.set { request in
            _ = calls.next()
            if request.url!.path == "/auth/refresh" {
                store.write("access_token", next); store.write("refresh_token", next)
                return (200, "{\"access_token\":\"\(oldToken)\",\"refresh_token\":\"\(oldToken)\"}")
            }
            return (401, "{}")
        }
        do { try await api.deleteMatch(matchID: UUID(), account: account); XCTFail("must not retry as new user") }
        catch { let isAuth = await MatchPipeline.isAuthFailure(error); XCTAssertTrue(isAuth) }
        XCTAssertEqual(store.read("access_token"), next)
        XCTAssertEqual(store.read("refresh_token"), next)
        XCTAssertEqual(calls.next(), 3)
    }

    func testSameAccountRefreshRetriesDeletion() async throws {
        let oldToken = token!, calls = Counter()
        LifecycleURLProtocol.handler.set { request in
            let n = calls.next()
            if request.url!.path == "/auth/refresh" {
                return (200, "{\"access_token\":\"\(oldToken)\",\"refresh_token\":\"\(oldToken)\"}")
            }
            return n == 1 ? (401, "{}") : (202, "{\"status\":\"deleted\"}")
        }
        try await api.deleteMatch(matchID: UUID(), account: account)
        XCTAssertEqual(calls.next(), 4)
    }

    func testCancelCarriesExactJobIDAndTreats409AsObsolete() async throws {
        let id = UUID(), job = UUID().uuidString
        LifecycleURLProtocol.handler.set { request in
            XCTAssertEqual(request.httpMethod, "POST")
            XCTAssertEqual(request.url!.path, "/matches/\(id.uuidString.lowercased())/cancel")
            let data: Data
            if let body = request.httpBody { data = body }
            else if let stream = request.httpBodyStream {
                stream.open(); defer { stream.close() }
                var bytes = Data(), buffer = [UInt8](repeating: 0, count: 1024)
                while stream.hasBytesAvailable {
                    let n = stream.read(&buffer, maxLength: buffer.count)
                    if n <= 0 { break }; bytes.append(buffer, count: n)
                }
                data = bytes
            } else { data = Data() }
            let body = try JSONSerialization.jsonObject(with: data) as? [String: String]
            XCTAssertEqual(body?["job_id"], job)
            return (409, "{\"detail\":\"stale job\"}")
        }
        try await api.cancelAnalysis(matchID: id, jobID: job, account: account)
    }

    func testAccountParserRejectsInvalidSubject() {
        XCTAssertNil(APIClient.account(token: "broken", baseURL: origin))
        let payload = Data("{\"sub\":\"not-a-uuid\"}".utf8).base64EncodedString()
        XCTAssertNil(APIClient.account(token: "h.\(payload).s", baseURL: origin))
        XCTAssertEqual(account.userID, account.userID.lowercased())
    }
}
