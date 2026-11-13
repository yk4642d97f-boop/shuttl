import Foundation

/// Local routing identity, never an authorization credential. The server still verifies every request.
struct RemoteAccount: Codable, Equatable, Hashable, Sendable {
    let baseURL: String
    let userID: String
}

struct RemoteCommand: Codable, Equatable, Sendable {
    enum Kind: String, Codable { case delete, cancel }
    let id: UUID
    let account: RemoteAccount
    let matchID: UUID
    let kind: Kind
    let jobID: String?
    var failures: Int = 0
}

/// Save user intent before removing local records. HTTP retries are scoped to the original account.
actor RemoteCommandQueue {
    static let shared = RemoteCommandQueue()
    private let url: URL
    private var commands: [RemoteCommand] = []
    private var loadError: Error?
    private var flushing = false

    init(url: URL? = nil) {
        self.url = url ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("RemoteActions/pending-v1.json")
        do {
            if FileManager.default.fileExists(atPath: self.url.path) {
                commands = try JSONDecoder().decode([RemoteCommand].self, from: Data(contentsOf: self.url))
            }
        } catch { loadError = error } // Never silently discard a user's deletion request.
    }

    func enqueue(account: RemoteAccount, matchID: UUID, kind: RemoteCommand.Kind, jobID: String? = nil) throws {
        if let loadError { throw loadError }
        if kind == .cancel && jobID == nil { throw APIError.uploadFailed("尚未取得云端任务编号，请稍后取消") }
        let matching = commands.filter { $0.account == account && $0.matchID == matchID }
        if matching.contains(where: { $0.kind == .delete || ($0.kind == kind && $0.jobID == jobID) }) { return }
        var updated = commands
        if kind == .delete { updated.removeAll { $0.account == account && $0.matchID == matchID } }
        updated.append(RemoteCommand(id: UUID(), account: account, matchID: matchID, kind: kind, jobID: jobID))
        try save(updated)
    }

    func count() -> Int { commands.count }
    func storageErrorMessage() -> String? {
        loadError == nil ? nil : "云端操作记录无法读取，请保留应用数据后重试"
    }

    func pending(account: RemoteAccount, matchID: UUID? = nil) throws -> [RemoteCommand] {
        if let loadError { throw loadError }
        return commands.filter { $0.account == account && (matchID == nil || $0.matchID == matchID) }
    }

    /// One bounded pass. Foreground/network recovery or explicit retry invokes another pass.
    func flush(using backend: any MatchBackend) async {
        guard !flushing, loadError == nil, let account = await backend.remoteAccount() else { return }
        // Check again after the actor hop, so concurrent callers cannot start two passes.
        guard !flushing else { return }
        flushing = true
        defer { flushing = false }
        let batch = commands.filter { $0.account == account }
        for command in batch {
            guard await backend.remoteAccount() == account else { break }
            guard commands.contains(where: { $0.id == command.id }) else { continue }
            do {
                switch command.kind {
                case .delete: try await backend.deleteRemote(matchID: command.matchID, account: account)
                case .cancel:
                    guard let jobID = command.jobID else { continue }
                    try await backend.cancelRemote(matchID: command.matchID, jobID: jobID, account: account)
                }
                // A concurrent delete may have superseded this cancellation; remove only this ID.
                try save(commands.filter { $0.id != command.id })
            } catch {
                var updated = commands
                if let index = updated.firstIndex(where: { $0.id == command.id }) { updated[index].failures += 1 }
                try? save(updated)
                break // Avoid hammering an offline server or an expired login.
            }
        }
    }

    private func save(_ updated: [RemoteCommand]) throws {
        let directory = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try JSONEncoder().encode(updated).write(to: url, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        commands = updated
    }
}
