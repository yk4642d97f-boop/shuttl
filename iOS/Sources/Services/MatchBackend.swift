import Foundation
import ShuttlCore

/// The server calls the match pipeline needs. `LiveMatchBackend` talks to the
/// real API; tests and demo mode substitute their own.
protocol MatchBackend: Sendable {
    var usesRemoteStorage: Bool { get }
    func remoteAccount() async -> RemoteAccount?
    func scoped(to account: RemoteAccount) -> any MatchBackend
    func deleteRemote(matchID: UUID, account: RemoteAccount) async throws
    func cancelRemote(matchID: UUID, jobID: String, account: RemoteAccount) async throws
    func createMatch(id: UUID, durationSeconds: Double) async throws -> MatchResponse
    func getMatch(id: UUID) async throws -> MatchResponse
    func getUploadURL(matchID: UUID) async throws -> UploadURLResponse
    func uploadVideo(matchID: UUID, fileURL: URL, progress: @escaping @Sendable (Double) -> Void) async throws
    func forgetUpload(matchID: UUID) async throws
    func triggerAnalysis(matchID: UUID, doubles: Bool?) async throws -> JobResponse
    func pollJob(matchID: UUID, jobID: String) async throws -> JobStatusResponse
    func getTrajectory(matchID: UUID) async throws -> TrajectoryResponse
    func getAnalysis(matchID: UUID) async throws -> AnalysisDocumentDTO
    func getAnalysisArchive(matchID: UUID) async throws -> MatchAnalysis?
    func triggerAnalysis(matchID: UUID, doubles: Bool?, inputs: MatchInputs) async throws -> JobResponse
}

extension MatchBackend {
    func getAnalysisArchive(matchID: UUID) async throws -> MatchAnalysis? { nil }
    func triggerAnalysis(matchID: UUID, doubles: Bool?, inputs: MatchInputs) async throws -> JobResponse {
        try await triggerAnalysis(matchID: matchID, doubles: doubles)
    }
    var usesRemoteStorage: Bool { true }
    func remoteAccount() async -> RemoteAccount? { nil }
    func scoped(to account: RemoteAccount) -> any MatchBackend { self }
    func deleteRemote(matchID: UUID, account: RemoteAccount) async throws { throw APIError.unauthorized }
    func cancelRemote(matchID: UUID, jobID: String, account: RemoteAccount) async throws { throw APIError.unauthorized }
    func forgetUpload(matchID: UUID) async throws {}
}

struct LiveMatchBackend: MatchBackend {
    var account: RemoteAccount? = nil
    func remoteAccount() async -> RemoteAccount? { await api.remoteAccount() }
    func scoped(to account: RemoteAccount) -> any MatchBackend { LiveMatchBackend(account: account) }
    func deleteRemote(matchID: UUID, account: RemoteAccount) async throws {
        try await api.deleteMatch(matchID: matchID, account: account)
    }
    func cancelRemote(matchID: UUID, jobID: String, account: RemoteAccount) async throws {
        try await api.cancelAnalysis(matchID: matchID, jobID: jobID, account: account)
    }
    private var api: APIClient { .shared }

    func getAnalysisArchive(matchID: UUID) async throws -> MatchAnalysis? {
        try await api.getAnalysisArchive(matchID: matchID, account: account)
    }

    func triggerAnalysis(matchID: UUID, doubles: Bool?, inputs: MatchInputs) async throws -> JobResponse {
        try await api.triggerAnalysis(matchID: matchID, doubles: doubles, account: account, inputs: inputs)
    }

    func createMatch(id: UUID, durationSeconds: Double) async throws -> MatchResponse {
        try await api.createMatch(id: id, durationSeconds: durationSeconds, account: account)
    }

    func getMatch(id: UUID) async throws -> MatchResponse {
        try await api.getMatch(matchID: id, account: account)
    }

    func getUploadURL(matchID: UUID) async throws -> UploadURLResponse {
        let response = try await api.getUploadURL(matchID: matchID, account: account)
        let absolute = await api.absoluteURL(response.uploadUrl)
        return UploadURLResponse(uploadUrl: absolute, ossKey: response.ossKey)
    }

    func uploadVideo(matchID: UUID, fileURL: URL, progress: @escaping @Sendable (Double) -> Void) async throws {
        let scope = await api.uploadScope()
        let key = VideoUploadService.key(scope: account?.baseURL ?? scope, matchID: matchID)
        try await VideoUploadService.shared.upload(key: key, fileURL: fileURL, target: {
            try await self.getUploadURL(matchID: matchID).uploadUrl
        }, progress: progress)
    }

    func forgetUpload(matchID: UUID) async throws {
        let scope = await api.uploadScope()
        let key = VideoUploadService.key(scope: account?.baseURL ?? scope, matchID: matchID)
        try await VideoUploadService.shared.forget(key: key)
    }

    func triggerAnalysis(matchID: UUID, doubles: Bool?) async throws -> JobResponse {
        try await api.triggerAnalysis(matchID: matchID, doubles: doubles, account: account)
    }

    func pollJob(matchID: UUID, jobID: String) async throws -> JobStatusResponse {
        try await api.pollJob(matchID: matchID, jobID: jobID, account: account)
    }

    func getTrajectory(matchID: UUID) async throws -> TrajectoryResponse {
        try await api.getTrajectory(matchID: matchID, account: account)
    }

    func getAnalysis(matchID: UUID) async throws -> AnalysisDocumentDTO {
        try await api.getAnalysis(matchID: matchID, account: account)
    }
}

/// Demo mode (and anything else that must stay offline): every call fails fast
/// without touching the network.
struct OfflineMatchBackend: MatchBackend {
    var usesRemoteStorage: Bool { false }
    struct Offline: LocalizedError {
        var errorDescription: String? { "演示模式不连接分析服务器" }
    }

    func createMatch(id: UUID, durationSeconds: Double) async throws -> MatchResponse { throw Offline() }
    func getMatch(id: UUID) async throws -> MatchResponse { throw Offline() }
    func getUploadURL(matchID: UUID) async throws -> UploadURLResponse { throw Offline() }
    func uploadVideo(matchID: UUID, fileURL: URL, progress: @escaping @Sendable (Double) -> Void) async throws {
        throw Offline()
    }
    func triggerAnalysis(matchID: UUID, doubles: Bool?) async throws -> JobResponse { throw Offline() }
    func pollJob(matchID: UUID, jobID: String) async throws -> JobStatusResponse { throw Offline() }
    func getTrajectory(matchID: UUID) async throws -> TrajectoryResponse { throw Offline() }
    func getAnalysis(matchID: UUID) async throws -> AnalysisDocumentDTO { throw Offline() }
}
