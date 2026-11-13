import AVFoundation
import Foundation
import os

/// Test footage pushed onto the phone with `devicectl device copy to` lands in
/// Documents/DebugInbox. Each finished file is moved into the match library and
/// analysed like a fresh recording, so long device runs don't need a person
/// filming. The folder only exists when a developer creates it (it is not
/// "Documents/Inbox", which iOS reserves for documents opened from other apps).
/// A file whose name contains "doubles" is filed as doubles, otherwise singles.
enum DebugInbox {
    static let folderName = "DebugInbox"
    static let videoExtensions: Set<String> = ["mov", "mp4", "m4v"]

    static func directory(documents: URL = VideoStore.documentsDirectory) -> URL {
        documents.appendingPathComponent(folderName, isDirectory: true)
    }

    /// Videos waiting in the inbox, oldest first. Empty when the folder doesn't exist.
    static func pendingVideos(documents: URL = VideoStore.documentsDirectory) -> [URL] {
        let keys: [URLResourceKey] = [.contentModificationDateKey, .isRegularFileKey]
        let files = (try? FileManager.default.contentsOfDirectory(
            at: directory(documents: documents), includingPropertiesForKeys: keys)) ?? []
        return files
            .filter { videoExtensions.contains($0.pathExtension.lowercased()) }
            .filter { (try? $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true }
            .sorted { modified($0) < modified($1) }
    }

    static func courtType(for url: URL) -> CourtType {
        url.lastPathComponent.lowercased().contains("doubles") ? .doubles : .singles
    }

    /// Moves every settled inbox video into Documents/Matches and hands it to the pipeline.
    /// A file that is still growing (a copy in progress) is left for the next pass.
    @MainActor private static var draining = false

    @MainActor
    static func drain(into pipeline: MatchPipeline, documents: URL = VideoStore.documentsDirectory,
                      settle: Duration = .seconds(1)) async {
        guard !draining else { return }
        draining = true
        defer { draining = false }
        let pending = pendingVideos(documents: documents)
        #if DEBUG
        if !pending.isEmpty {
            pipeline.admissionLog.record(event: "debug_inbox", [
                "phase": .string("begin"),
                "count": .number(Double(pending.count))
            ])
        }
        #endif
        var ingested = 0
        for url in pending {
            let bytes = size(url)
            let fileName = url.lastPathComponent

            #if DEBUG
            let trigger = pending.count == 1 ? "single" : "batch"
            #endif

            guard await isSettled(url, after: settle) else {
                #if DEBUG
                pipeline.admissionLog.record(event: "debug_inbox", [
                    "trigger": .string(trigger),
                    "file": .string(fileName),
                    "bytes": .number(Double(bytes)),
                    "outcome": .string("unsettled")
                ])
                #endif
                continue
            }

            let asset = AVURLAsset(url: url)
            guard let duration = try? await asset.load(.duration), duration.seconds.isFinite,
                  duration.seconds > 0 else {
                #if DEBUG
                pipeline.admissionLog.record(event: "debug_inbox", [
                    "trigger": .string(trigger),
                    "file": .string(fileName),
                    "bytes": .number(Double(bytes)),
                    "outcome": .string("no_duration")
                ])
                #endif
                continue
            }

            let destination = VideoStore.newRecordingURL()
                .deletingPathExtension().appendingPathExtension(url.pathExtension.lowercased())
            do {
                try FileManager.default.moveItem(at: url, to: destination)
                try pipeline.ingestRecording(fileURL: destination, durationSeconds: duration.seconds,
                                             courtType: courtType(for: url))
                ShuttlDebugLog.note("debug inbox: ingested \(url.lastPathComponent) (\(Int(duration.seconds)) s)")
                #if DEBUG
                pipeline.admissionLog.record(event: "debug_inbox", [
                    "trigger": .string(trigger),
                    "file": .string(fileName),
                    "bytes": .number(Double(bytes)),
                    "outcome": .string("ingested")
                ])
                #endif
                ingested += 1
            } catch {
                ShuttlDebugLog.note("debug inbox: skipped \(url.lastPathComponent): \(error)")
                #if DEBUG
                pipeline.admissionLog.record(event: "debug_inbox", [
                    "trigger": .string(trigger),
                    "file": .string(fileName),
                    "bytes": .number(Double(bytes)),
                    "outcome": .string("error"),
                    "error": .string(String(describing: error))
                ])
                #endif
            }
        }
        #if DEBUG
        if !pending.isEmpty {
            pipeline.admissionLog.record(event: "debug_inbox", [
                "phase": .string("end"),
                "ingested": .number(Double(ingested)),
                "skipped": .number(Double(pending.count - ingested))
            ])
        }
        #endif
    }

    private static func modified(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
    }

    private static func size(_ url: URL) -> Int {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? -1
    }

    private static func isSettled(_ url: URL, after delay: Duration) async -> Bool {
        let before = size(url)
        guard before > 0 else { return false }
        try? await Task.sleep(for: delay)
        return size(url) == before
    }

    #if DEBUG
    /// DEBUG-only: periodic re-poll for up to 10 minutes while app is active and inbox is non-empty.
    @MainActor private static var pollTask: Task<Void, Never>?

    @MainActor
    static func startPolling(pipeline: MatchPipeline, documents: URL = VideoStore.documentsDirectory) {
        guard pollTask == nil else { return }
        pollTask = Task {
            let deadline = Date().addingTimeInterval(10 * 60)
            while Date() < deadline, !Task.isCancelled {
                try? await Task.sleep(for: .seconds(5))
                guard !pendingVideos(documents: documents).isEmpty else { break }
                await drain(into: pipeline, documents: documents)
            }
            pollTask = nil
        }
    }

    @MainActor
    static func stopPolling() {
        pollTask?.cancel()
        pollTask = nil
    }
    #endif
}

/// Minimal logger until the app adopts ShuttlRuntime's ShuttlLog.
enum ShuttlDebugLog {
    private static let logger = Logger(subsystem: "com.example.shuttl", category: "pipeline")
    static func note(_ message: String) { logger.notice("\(message, privacy: .public)") }
}
