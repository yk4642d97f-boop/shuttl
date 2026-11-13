import Foundation

/// A directory of timestamped JSON files that keeps only the newest `maxFiles`.
/// Only files it named (`<UTC timestamp>-<kind>.json`) are counted or pruned; anything
/// else in the directory is left alone.
public struct DiagnosticsFileStore: Sendable {
    public let directory: URL
    public let maxFiles: Int
    private let now: @Sendable () -> Date

    /// Documents/Diagnostics/MetricKit (the orchestrator's Documents/Diagnostics/stage-metrics.jsonl
    /// is one level up and never touched).
    public static func metricKit(documents: URL? = nil, maxFiles: Int = 20) -> DiagnosticsFileStore {
        let docs = documents ?? FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return DiagnosticsFileStore(directory: docs.appendingPathComponent("Diagnostics/MetricKit", isDirectory: true),
                                    maxFiles: maxFiles)
    }

    public init(directory: URL, maxFiles: Int = 20, now: @escaping @Sendable () -> Date = { Date() }) {
        self.directory = directory
        self.maxFiles = max(1, maxFiles)
        self.now = now
    }

    /// Writes `data` as a new file, then prunes. Returns the file written.
    @discardableResult
    public func write(_ data: Data, kind: String) throws -> URL {
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let safeKind = String(kind.map { $0.isLetter || $0.isNumber || $0 == "-" ? $0 : "_" })
        let stamp = Self.stamp(now())
        var url = directory.appendingPathComponent("\(stamp)-\(safeKind).json")
        var n = 1
        while fm.fileExists(atPath: url.path) {
            n += 1
            url = directory.appendingPathComponent("\(stamp)-\(safeKind)-\(n).json")
        }
        try data.write(to: url, options: .atomic)
        prune()
        return url
    }

    /// Own files, oldest first.
    public func files() -> [URL] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        return names.filter(Self.isOwn).sorted().map { directory.appendingPathComponent($0) }
    }

    public func prune() {
        let own = files()
        guard own.count > maxFiles else { return }
        for url in own.prefix(own.count - maxFiles) { try? FileManager.default.removeItem(at: url) }
    }

    /// "20260928T081530.123Z" — sorts chronologically as text.
    static func stamp(_ d: Date) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "yyyyMMdd'T'HHmmss.SSS'Z'"
        return f.string(from: d)
    }

    static func isOwn(_ name: String) -> Bool {
        name.hasSuffix(".json") && name.count > 21 && name.prefix(20).allSatisfy { $0.isNumber || "TZ.".contains($0) }
            && name.dropFirst(20).first == "-"
    }
}
