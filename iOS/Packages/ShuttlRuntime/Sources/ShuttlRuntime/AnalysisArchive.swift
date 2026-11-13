import Foundation
import ShuttlAnalysis
import ShuttlCore

/// Saves a finished `MatchAnalysis` for the app's library without the memory blow-up
/// of JSON-encoding every player pose, and deletes everything a match left on disk.
///
/// On disk (same conventions as the app's AnalysisStore):
///   Documents/Analyses/<id>.json          MatchAnalysis JSON with `players: []`
///   Documents/Analyses/<id>.players.bin   players, slim binary (PlayersCodec, no keypoints)
///   Application Support/AnalysisCheckpoints/<id>/   resumable stage checkpoints
/// The JSON stays a valid MatchAnalysis, so an older loader still opens it (just
/// without players). The side file is tied to its JSON by a token in
/// diagnostics["archive"], so a stale side file from an earlier save is never mixed in.
public struct AnalysisArchive: Sendable {
    public let analysesDirectory: URL
    public let checkpointsDirectory: URL

    public init(analysesDirectory: URL, checkpointsDirectory: URL) {
        self.analysesDirectory = analysesDirectory
        self.checkpointsDirectory = checkpointsDirectory
    }

    /// Documents/Analyses + Application Support/AnalysisCheckpoints.
    public static func standard(documents: URL? = nil, applicationSupport: URL? = nil) -> AnalysisArchive {
        let fm = FileManager.default
        let docs = documents ?? fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let support = applicationSupport ?? fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return AnalysisArchive(analysesDirectory: docs.appendingPathComponent("Analyses", isDirectory: true),
                               checkpointsDirectory: support.appendingPathComponent("AnalysisCheckpoints", isDirectory: true))
    }

    static let diagnosticsKey = "archive"
    static let format = "shuttl-archive-1"

    private static func name(_ id: UUID) -> String { id.uuidString.lowercased() }

    public func analysisURL(matchID: UUID) -> URL {
        analysesDirectory.appendingPathComponent("\(Self.name(matchID)).json")
    }

    public func playersURL(matchID: UUID) -> URL { Self.playersURL(forAnalysis: analysisURL(matchID: matchID)) }

    /// `<name>.json` → `<name>.players.bin`.
    public static func playersURL(forAnalysis url: URL) -> URL {
        url.deletingPathExtension().appendingPathExtension("players.bin")
    }

    public func checkpointDirectory(matchID: UUID) -> URL {
        checkpointsDirectory.appendingPathComponent(Self.name(matchID), isDirectory: true)
    }

    /// Checkpoints keyed to this exact video file, build and options
    /// (see `checkpointFingerprint`): corrected corner taps never reuse a stale camera.
    public func checkpointStore(matchID: UUID, video: URL, build: String, options: PipelineOptions) -> FileCheckpointStore {
        let attrs = try? FileManager.default.attributesOfItem(atPath: video.path)
        let size = (attrs?[.size] as? NSNumber)?.intValue ?? 0
        let modified = attrs?[.modificationDate] as? Date
        let fp = checkpointFingerprint(videoSize: size, modified: modified, build: build, options: options)
        return FileCheckpointStore(directory: checkpointDirectory(matchID: matchID), fingerprint: fp)
    }

    /// "<CFBundleShortVersionString>-<CFBundleVersion>" of `bundle`.
    public static func build(of bundle: Bundle = .main) -> String {
        let v = bundle.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0"
        let b = bundle.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "0"
        return "\(v)-\(b)"
    }

    // MARK: Save / load

    /// Writes the players side file, then the JSON (atomic). Returns the JSON URL.
    /// Keypoints are dropped unless `keepKeypoints` (nothing in the library reads them).
    @discardableResult
    public func save(_ analysis: MatchAnalysis, matchID: UUID, keepKeypoints: Bool = false) throws -> URL {
        let fm = FileManager.default
        try fm.createDirectory(at: analysesDirectory, withIntermediateDirectories: true)
        let json = analysisURL(matchID: matchID)
        let side = playersURL(matchID: matchID)
        let token = UUID().uuidString
        try PlayersCodec.write(analysis.players, to: side, fingerprint: token, includeKeypoints: keepKeypoints)

        let data: Data = try autoreleasepool {
            var shell = analysis                  // copy-on-write: only `players` / diagnostics diverge
            shell.players = []
            shell.diagnostics[Self.diagnosticsKey] = .object([
                "format": .string(Self.format), "players_file": .string(side.lastPathComponent),
                "players_token": .string(token), "players_count": .number(Double(analysis.players.count)),
                "keypoints": .bool(keepKeypoints),
            ])
            return try ShuttlJSON.encoder().encode(shell)
        }
        try data.write(to: json, options: .atomic)
        return json
    }

    public func load(matchID: UUID) throws -> MatchAnalysis { try Self.load(analysisAt: analysisURL(matchID: matchID)) }

    /// Loads an archived analysis (players from the side file) or a legacy full JSON.
    public static func load(analysisAt url: URL) throws -> MatchAnalysis {
        var a = try autoreleasepool { try ShuttlJSON.load(MatchAnalysis.self, from: url) }
        guard let info = a.diagnostics[diagnosticsKey] else { return a }      // legacy: players inline
        a.diagnostics[diagnosticsKey] = nil
        let side = url.deletingLastPathComponent()
            .appendingPathComponent(info["players_file"]?.string ?? playersURL(forAnalysis: url).lastPathComponent)
        if let token = info["players_token"]?.string, FileManager.default.fileExists(atPath: side.path),
           let players = try? PlayersCodec.read(from: side, expectedFingerprint: token) {
            a.players = players
        }
        return a
    }

    // MARK: Delete

    /// Removes the match's analysis JSON, players side file, leftover temp files, its
    /// checkpoint directory and (when given) the video. Missing files are fine.
    /// Returns what was actually removed.
    @discardableResult
    public func deleteAll(matchID: UUID, video: URL? = nil) -> [URL] {
        let fm = FileManager.default
        var targets = [analysisURL(matchID: matchID), playersURL(matchID: matchID), checkpointDirectory(matchID: matchID)]
        if let video { targets.append(video) }
        // Interrupted atomic writes leave ".<name>.<uuid>.tmp" next to the side file.
        let prefix = ".\(Self.name(matchID))."
        if let names = try? fm.contentsOfDirectory(atPath: analysesDirectory.path) {
            targets += names.filter { $0.hasPrefix(prefix) && $0.hasSuffix(".tmp") }
                .map { analysesDirectory.appendingPathComponent($0) }
        }
        return targets.filter { url in
            guard fm.fileExists(atPath: url.path) else { return false }
            return (try? fm.removeItem(at: url)) != nil
        }
    }

    /// Only the checkpoints (e.g. once the app accepts a result as final despite a failed stage).
    public func removeCheckpoints(matchID: UUID) {
        try? FileManager.default.removeItem(at: checkpointDirectory(matchID: matchID))
    }
}
