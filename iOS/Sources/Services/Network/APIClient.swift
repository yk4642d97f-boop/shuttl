import Foundation
import ShuttlCore

// MARK: - Response Models
// Shapes mirror backend/app/schemas/*.py. Keys arrive in snake_case and are
// converted by the decoder, so `upload_url` → `uploadUrl`, `job_id` → `jobId`.

struct TokenResponse: Codable, Sendable {
    let accessToken: String
    let refreshToken: String
    let tokenType: String?
}

struct MatchResponse: Codable, Sendable {
    let id: UUID
    let status: String
    let durationSeconds: Double
    let videoOssKey: String?
    let celeryJobId: String?
    let createdAt: Date
}

struct UploadURLResponse: Codable, Sendable {
    let uploadUrl: URL
    let ossKey: String
}

struct JobResponse: Codable, Sendable {
    let jobId: String
    let status: String
}

/// Celery task state as reported by `GET /matches/{id}/job/{job_id}`.
struct JobStatusResponse: Codable, Sendable {
    let jobId: String
    /// PENDING | STARTED | PROGRESS | SUCCESS | FAILURE | RETRY
    let state: String
    let info: String?
}

struct TrajectoryPointDTO: Codable, Sendable {
    let frame: Int
    let xPx: Float
    let yPx: Float
    let xMeters: Float
    let yMeters: Float
    let speedKmh: Float
    let shotType: String
}

struct TrajectoryResponse: Codable, Sendable {
    let matchId: UUID
    let fps: Double
    let points: [TrajectoryPointDTO]
}

/// Subset of the v2 analysis document (`GET /matches/{id}/analysis`, see
/// backend services/tracking/serialize.py). Positions are normalised 0–1 of the upright frame.
struct AnalysisDocumentDTO: Decodable, Sendable {
    struct Video: Decodable, Sendable {
        let fps: Double?
        let width: Int?
        let height: Int?
    }
    struct Flight: Decodable, Sendable {
        let speed0Kmh: Double?
    }
    struct Shot: Decodable, Sendable {
        let frame: Int
        let xyNorm: [Double?]?
        let shotType: String?
        let legacyShotType: String?
        let hitterPos: [Double?]?
        let flight: Flight?
    }
    struct RallyDTO: Decodable, Sendable {
        let startFrame: Int
        let endFrame: Int
        let winner: String?
        let shots: [Shot]
    }

    let video: Video?
    let rallies: [RallyDTO]
}

struct StatisticsResponse: Codable, Sendable {
    let matchId: UUID
    let totalRallies: Int
    let totalShots: Int
    let maxSpeedKmh: Float
    let avgSpeedKmh: Float
    let shotTypeCounts: [String: Int]
}

// MARK: - API Error

enum APIError: Error, LocalizedError, Sendable {
    case networkError(Error)
    case serverError(Int, String)
    case unauthorized
    case decodingError(Error)
    case uploadFailed(String)
    case timeout

    var errorDescription: String? {
        switch self {
        case .networkError(let err):
            return "网络错误：\(err.localizedDescription)"
        case .serverError(let code, let msg):
            return "服务器错误 (\(code))：\(msg)"
        case .unauthorized:
            return "身份验证失败，请重新登录"
        case .decodingError(let err):
            return "数据解析错误：\(err.localizedDescription)"
        case .uploadFailed(let reason):
            return "视频上传失败：\(reason)"
        case .timeout:
            return "请求超时，请检查网络连接"
        }
    }
}

// MARK: - APIClient

actor APIClient {
    static let shared = APIClient()

    /// Read from Info.plist `ShuttlAPIBaseURL`; the simulator reaches the
    /// host Mac's uvicorn on 127.0.0.1.
    private let baseURL: String
    private let session: URLSession
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder

    private let tokenRead: @Sendable (String) -> String?
    private let tokenWrite: @Sendable (String, String) -> Void
    private let tokenDelete: @Sendable (String) -> Void

    init(baseURL overrideURL: String? = nil, session testSession: URLSession? = nil,
         tokenRead: @escaping @Sendable (String) -> String? = { keychainLoad(key: $0) },
         tokenWrite: @escaping @Sendable (String, String) -> Void = { keychainSave(key: $0, value: $1) },
         tokenDelete: @escaping @Sendable (String) -> Void = { keychainDelete(key: $0) }) {
        self.tokenRead = tokenRead
        self.tokenWrite = tokenWrite
        self.tokenDelete = tokenDelete
        self.baseURL = overrideURL ?? CloudConfiguration.baseURL

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 300
        self.session = testSession ?? URLSession(configuration: config)

        let dec = JSONDecoder()
        dec.keyDecodingStrategy = .convertFromSnakeCase
        dec.dateDecodingStrategy = .custom(APIClient.decodeISO8601)
        self.decoder = dec

        let enc = JSONEncoder()
        enc.keyEncodingStrategy = .convertToSnakeCase
        self.encoder = enc
    }

    /// FastAPI emits microsecond timestamps (`2026-09-26T03:16:34.412579Z`),
    /// which `.iso8601` rejects.
    @Sendable private static func decodeISO8601(_ decoder: Decoder) throws -> Date {
        let raw = try decoder.singleValueContainer().decode(String.self)
        let withFraction = ISO8601DateFormatter()
        withFraction.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = withFraction.date(from: raw) { return date }

        let plain = ISO8601DateFormatter()
        plain.formatOptions = [.withInternetDateTime]
        let stripped = raw.replacingOccurrences(of: #"\.\d+"#, with: "", options: .regularExpression)
        if let date = plain.date(from: stripped) { return date }

        throw DecodingError.dataCorrupted(.init(
            codingPath: decoder.codingPath,
            debugDescription: "无法解析日期：\(raw)"
        ))
    }

    // MARK: - Token storage (Keychain helpers live in AuthService.swift)

    private let accessTokenKey = "access_token"
    private let refreshTokenKey = "refresh_token"

    private func store(_ tokens: TokenResponse) {
        tokenWrite(accessTokenKey, tokens.accessToken)
        tokenWrite(refreshTokenKey, tokens.refreshToken)
    }

    func logout() async {
        tokenDelete(accessTokenKey)
        tokenDelete(refreshTokenKey)
        await VideoUploadService.shared.cancelAll()
    }

    // MARK: - Request plumbing

    private func makeRequest(_ path: String, method: String) -> URLRequest {
        var request = URLRequest(url: URL(string: baseURL + path)!)
        request.httpMethod = method
        return request
    }

    private func jsonRequest<Body: Encodable>(_ path: String, method: String, body: Body) throws -> URLRequest {
        var request = makeRequest(path, method: method)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try encoder.encode(body)
        return request
    }

    private func perform<T: Decodable>(
        _ request: URLRequest,
        authenticated: Bool = true,
        retryOn401: Bool = true,
        expectedAccount: RemoteAccount? = nil,
        nativeContract: Bool = false
    ) async throws -> T {
        let account = expectedAccount ?? (authenticated ? remoteAccount() : nil)
        if let account, remoteAccount() != account { throw APIError.unauthorized }
        var req = request
        if authenticated {
            guard let token = tokenRead(accessTokenKey) else { throw APIError.unauthorized }
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: req)
        } catch let urlError as URLError where urlError.code == .timedOut {
            throw APIError.timeout
        } catch let urlError as URLError where urlError.code == .cancelled {
            throw CancellationError()
        } catch {
            throw APIError.networkError(error)
        }

        if let account, remoteAccount() != account { throw APIError.unauthorized }
        guard let http = response as? HTTPURLResponse else {
            throw APIError.networkError(URLError(.badServerResponse))
        }

        switch http.statusCode {
        case 200...299:
            do {
                return try (nativeContract ? ShuttlJSON.decoder() : decoder).decode(T.self, from: data)
            } catch {
                throw APIError.decodingError(error)
            }
        case 401 where authenticated && retryOn401:
            try await refreshTokens()
            return try await perform(request, authenticated: true, retryOn401: false, expectedAccount: account, nativeContract: nativeContract)
        case 401:
            if authenticated { await logout() }
            throw APIError.serverError(401, detailMessage(from: data) ?? "身份验证失败")
        default:
            let message = detailMessage(from: data)
                ?? HTTPURLResponse.localizedString(forStatusCode: http.statusCode)
            throw APIError.serverError(http.statusCode, message)
        }
    }

    /// FastAPI errors look like `{"detail": "..."}` or `{"detail": [{...}]}`.
    private func detailMessage(from data: Data) -> String? {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        if let detail = obj["detail"] as? String { return detail }
        if let items = obj["detail"] as? [[String: Any]], let first = items.first {
            return first["msg"] as? String
        }
        return nil
    }

    private func refreshTokens() async throws {
        guard let refreshToken = tokenRead(refreshTokenKey) else { throw APIError.unauthorized }
        let request = try jsonRequest("/auth/refresh", method: "POST", body: ["refresh_token": refreshToken])
        let tokens: TokenResponse
        do {
            tokens = try await perform(request, authenticated: false)
        } catch {
            if tokenRead(refreshTokenKey) == refreshToken { await logout() }
            throw APIError.unauthorized
        }
        guard tokenRead(refreshTokenKey) == refreshToken else { throw APIError.unauthorized }
        store(tokens)
    }

    // MARK: - Auth

    /// `/auth/login` is an OAuth2 password form, not JSON.
    func login(email: String, password: String) async throws -> TokenResponse {
        var request = makeRequest("/auth/login", method: "POST")
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        var form = URLComponents()
        form.queryItems = [
            URLQueryItem(name: "username", value: email),
            URLQueryItem(name: "password", value: password),
        ]
        // URLComponents leaves "+" unescaped, which form decoding reads as a space.
        let encoded = (form.percentEncodedQuery ?? "").replacingOccurrences(of: "+", with: "%2B")
        request.httpBody = Data(encoded.utf8)
        let tokens: TokenResponse = try await perform(request, authenticated: false)
        store(tokens)
        return tokens
    }

    func register(email: String, password: String, nickname: String) async throws -> TokenResponse {
        struct Body: Encodable { let email: String; let password: String; let nickname: String }
        let request = try jsonRequest(
            "/auth/register", method: "POST",
            body: Body(email: email, password: password, nickname: nickname)
        )
        let tokens: TokenResponse = try await perform(request, authenticated: false)
        store(tokens)
        return tokens
    }

    // MARK: - Matches

    func createMatch(id: UUID, durationSeconds: Double, account: RemoteAccount? = nil) async throws -> MatchResponse {
        struct Body: Encodable { let id: UUID; let durationSeconds: Double }
        let request = try jsonRequest("/matches", method: "POST", body: Body(id: id, durationSeconds: durationSeconds))
        let response: MatchResponse = try await perform(request, expectedAccount: account)
        guard response.id == id else { throw APIError.serverError(409, "服务器版本过旧，无法安全创建比赛") }
        return response
    }

    /// Decode only for local routing. Authorization always remains on the server.
    func remoteAccount() -> RemoteAccount? {
        guard let token = tokenRead(accessTokenKey) ?? tokenRead(refreshTokenKey) else { return nil }
        return Self.account(token: token, baseURL: baseURL)
    }

    nonisolated static func account(token: String, baseURL: String) -> RemoteAccount? {
        let parts = token.split(separator: ".")
        guard parts.count == 3 else { return nil }
        var payload = String(parts[1]).replacingOccurrences(of: "-", with: "+").replacingOccurrences(of: "_", with: "/")
        payload += String(repeating: "=", count: (4 - payload.count % 4) % 4)
        guard let data = Data(base64Encoded: payload),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let sub = json["sub"] as? String, let id = UUID(uuidString: sub) else { return nil }
        return RemoteAccount(baseURL: baseURL, userID: id.uuidString.lowercased())
    }

    func deleteMatch(matchID: UUID, account: RemoteAccount) async throws {
        struct Acknowledgement: Decodable { let status: String }
        let result: Acknowledgement = try await perform(
            makeRequest("/matches/\(matchID.uuidString.lowercased())", method: "DELETE"), expectedAccount: account)
        guard result.status == "deleted" else { throw APIError.serverError(502, "服务器尚未确认删除") }
    }

    func cancelAnalysis(matchID: UUID, jobID: String, account: RemoteAccount) async throws {
        struct Body: Encodable { let jobId: String }
        let request = try jsonRequest("/matches/\(matchID.uuidString.lowercased())/cancel", method: "POST", body: Body(jobId: jobID))
        do { let _: JobStatusResponse = try await perform(request, expectedAccount: account) }
        catch APIError.serverError(let code, _) where code == 404 || code == 409 {
            // Deleted match or a newer job: this exact old job no longer needs cancellation.
        }
    }

    func getMatch(matchID: UUID, account: RemoteAccount? = nil) async throws -> MatchResponse {
        try await perform(makeRequest("/matches/\(matchID.uuidString.lowercased())", method: "GET"), expectedAccount: account)
    }

    func getUploadURL(matchID: UUID, account: RemoteAccount? = nil) async throws -> UploadURLResponse {
        try await perform(makeRequest("/matches/\(matchID.uuidString.lowercased())/upload", method: "POST"), expectedAccount: account)
    }

    /// Preserve the calibrated camera's K/R keys; generic snake-case encoding changes them.
    func triggerAnalysis(matchID: UUID, doubles: Bool? = nil, account: RemoteAccount? = nil,
                         inputs: MatchInputs = MatchInputs()) async throws -> JobResponse {
        struct Body: Encodable {
            let doubles: Bool?
            let corners: [String: [Double]]?
            let cornersNormalized = true
            let camera: CameraModel?
            let force: Bool
            enum CodingKeys: String, CodingKey {
                case doubles, corners, camera, force, cornersNormalized = "corners_normalized"
            }
        }
        var request = makeRequest("/matches/\(matchID.uuidString.lowercased())/analyze", method: "POST")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try ShuttlJSON.encoder().encode(Body(
            doubles: doubles, corners: inputs.manualCorners?.mapValues { [$0.x, $0.y] },
            camera: inputs.cameraHint, force: inputs.cloudReanalysisID != nil))
        if let id = inputs.cloudReanalysisID {
            request.setValue(id.uuidString.lowercased(), forHTTPHeaderField: "Idempotency-Key")
        }
        return try await perform(request, expectedAccount: account)
    }

    func getAnalysisArchive(matchID: UUID, account: RemoteAccount? = nil) async throws -> MatchAnalysis? {
        do {
            return try await perform(
                makeRequest("/matches/\(matchID.uuidString.lowercased())/analysis/archive", method: "GET"),
                expectedAccount: account, nativeContract: true)
        } catch APIError.serverError(let code, _) where code == 404 {
            return nil // Older saved jobs keep their legacy report fallback.
        }
    }

    /// Upload targets may come back relative ("/matches/…/video?token=…"); resolve them against the API host.
    func absoluteURL(_ url: URL) -> URL {
        if url.scheme != nil { return url }
        return URL(string: url.relativeString, relativeTo: URL(string: baseURL))?.absoluteURL ?? url
    }

    /// Stable upload namespace; never includes access tokens or a signed storage URL.
    func uploadScope() -> String { baseURL }

    func pollJob(matchID: UUID, jobID: String, account: RemoteAccount? = nil) async throws -> JobStatusResponse {
        try await perform(makeRequest("/matches/\(matchID.uuidString.lowercased())/job/\(jobID)", method: "GET"), expectedAccount: account)
    }

    func getTrajectory(matchID: UUID, account: RemoteAccount? = nil) async throws -> TrajectoryResponse {
        try await perform(makeRequest("/matches/\(matchID.uuidString.lowercased())/trajectory", method: "GET"), expectedAccount: account)
    }

    func getAnalysis(matchID: UUID, account: RemoteAccount? = nil) async throws -> AnalysisDocumentDTO {
        try await perform(makeRequest("/matches/\(matchID.uuidString.lowercased())/analysis", method: "GET"), expectedAccount: account)
    }

    func getStatistics(matchID: UUID) async throws -> StatisticsResponse {
        try await perform(makeRequest("/matches/\(matchID.uuidString.lowercased())/statistics", method: "GET"))
    }
}
