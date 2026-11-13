import Foundation

/// Match recordings live under Documents/Matches. `Match.localVideoPath` stores
/// the path relative to Documents because the absolute container path
/// (/var/mobile/Containers/Data/Application/<UUID>/…) changes whenever the app is
/// reinstalled from Xcode, which would orphan every stored absolute path.
enum VideoStore {
    static let folderName = "Matches"

    static var documentsDirectory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    /// Documents/Matches, created on demand.
    static func matchesDirectory() -> URL {
        let dir = documentsDirectory.appendingPathComponent(folderName, isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// A fresh file URL for a new take.
    static func newRecordingURL(now: Date = .now) -> URL {
        let stamp = Int(now.timeIntervalSince1970 * 1000)
        return matchesDirectory().appendingPathComponent("match_\(stamp).mov")
    }

    /// The value to persist for a file: relative to Documents when it lives there,
    /// otherwise the absolute path unchanged.
    static func storedPath(for url: URL, documents: URL = documentsDirectory) -> String {
        // Compare both as given and with symlinks resolved (/private/var vs /var).
        let candidates = [
            (url.standardizedFileURL.path, documents.standardizedFileURL.path),
            (url.resolvingSymlinksInPath().path, documents.resolvingSymlinksInPath().path),
        ]
        for (file, root) in candidates {
            let prefix = root.hasSuffix("/") ? root : root + "/"
            if file.hasPrefix(prefix), file.count > prefix.count { return String(file.dropFirst(prefix.count)) }
        }
        return url.standardizedFileURL.path
    }

    /// Resolves a stored path back to a file URL.
    /// - Relative paths are resolved against the current Documents directory.
    /// - Legacy absolute paths inside some app container's `/Documents/` are
    ///   re-rooted onto the current container (the old UUID may be gone).
    /// - Any other absolute path is returned as is.
    static func resolve(_ stored: String?, documents: URL = documentsDirectory) -> URL? {
        guard let stored, !stored.isEmpty else { return nil }
        guard stored.hasPrefix("/") else {
            return documents.appendingPathComponent(stored, isDirectory: false)
        }
        let absolute = URL(filePath: stored)
        if FileManager.default.fileExists(atPath: absolute.path) { return absolute }
        if let range = stored.range(of: "/Documents/", options: .backwards) {
            let relative = String(stored[range.upperBound...])
            if !relative.isEmpty { return documents.appendingPathComponent(relative, isDirectory: false) }
        }
        return absolute
    }
}
