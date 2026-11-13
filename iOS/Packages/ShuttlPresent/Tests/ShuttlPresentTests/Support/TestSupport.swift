import Foundation
import XCTest
import ShuttlCore

/// …/iOS/Packages/ShuttlPresent/Tests/ShuttlPresentTests/Support/TestSupport.swift → repo root.
let repoRoot: URL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent()

let testSplit = repoRoot.appendingPathComponent("backend/ml/data/raw/shuttlecock_trajectory/TrackNetV2/Test")

/// The 29 real Test clips (sorted), or skip when not gated on / not present.
func realClips() throws -> [URL] {
    guard ProcessInfo.processInfo.environment["SHUTTL_PRESENT_REAL"] == "1" else {
        throw XCTSkip("set SHUTTL_PRESENT_REAL=1 to run the real-clip CourtLock tests")
    }
    guard FileManager.default.fileExists(atPath: testSplit.path) else { throw XCTSkip("Test split missing") }
    var out: [URL] = []
    let matches = try FileManager.default.contentsOfDirectory(atPath: testSplit.path).filter { $0.hasPrefix("match") }.sorted()
    for m in matches {
        let dir = testSplit.appendingPathComponent(m).appendingPathComponent("video")
        let vids = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        out += vids.filter { $0.hasSuffix(".mp4") }.sorted().map { dir.appendingPathComponent($0) }
    }
    return out
}

func clipTag(_ url: URL) -> String {
    url.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent + "/" + url.deletingPathExtension().lastPathComponent
}

/// Upright RGB frames every `step` frames from `start`, at most `count`.
func decodeFrames(_ url: URL, start: Int = 0, step: Int, count: Int) async throws -> (meta: VideoMeta, frames: [(Int, ImageU8)]) {
    let src = try await VideoSource.open(url)
    var out: [(Int, ImageU8)] = []
    let stop = min(src.meta.nFrames, start + step * count)
    for (i, pb) in try src.frames(start: start, stop: stop, step: step) {
        out.append((i, try ImageU8.rgb(from: pb)))
        if out.count >= count { break }
    }
    return (src.meta, out)
}

/// Append a line to the package log (real-clip numbers, perf numbers).
func logLine(_ s: String) {
    print(s)
    let url = repoRoot.appendingPathComponent(".agent-logs/ShuttlPresent.log")
    guard let h = try? FileHandle(forWritingTo: url) else { return }
    defer { try? h.close() }
    _ = try? h.seekToEnd()
    try? h.write(contentsOf: Data((s + "\n").utf8))
}

func ms(_ block: () throws -> Void) rethrows -> Double {
    let t0 = DispatchTime.now().uptimeNanoseconds
    try block()
    return Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
}
