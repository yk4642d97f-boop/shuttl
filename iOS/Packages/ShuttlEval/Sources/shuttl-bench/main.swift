// shuttl-bench — the Swift twin of backend/scripts/benchmark.py (macOS only).
// See ShuttlEval/BenchCLI.swift for usage.
#if os(macOS)
import Foundation
import ShuttlEval

/// backend/ derived from this file: iOS/Packages/ShuttlEval/Sources/shuttl-bench/main.swift.
let defaultBackend = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("backend")

var argv = Array(CommandLine.arguments.dropFirst())
var backend = defaultBackend
if let i = argv.firstIndex(of: "--backend"), i + 1 < argv.count {
    backend = URL(fileURLWithPath: argv[i + 1])
    argv.removeSubrange(i...(i + 1))
}

// Integration point: set `liveRunner` to the on-device pipeline (an `AnalysisRunner`)
// to benchmark it directly on the Test clips.
let cli = BenchCLI(backend: backend, liveRunner: nil)
do {
    let (_, diff) = try await cli.run(argv)
    exit(diff.map { $0.ok ? 0 : 1 } ?? 0)
} catch let e as BenchCLIError {
    FileHandle.standardError.write(Data("\(e)\n".utf8))
    exit(2)
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
#else
print("shuttl-bench is macOS-only")
#endif
