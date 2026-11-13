import XCTest
import ShuttlCore
@testable import ShuttlLive

final class ConcurrencyTests: XCTestCase {

    /// Two tasks through one gate never overlap, and both finish.
    func testGateSerialises() async throws {
        let gate = AsyncGate()
        let box = OverlapBox()
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<4 {
                group.addTask {
                    try await withGate(gate) {
                        box.enter()
                        try await Task.sleep(nanoseconds: 20_000_000)
                        box.leave()
                    }
                }
            }
            try await group.waitForAll()
        }
        XCTAssertEqual(box.maxConcurrent, 1)
        XCTAssertEqual(box.completed, 4)
    }

    /// A failing body still releases the gate.
    func testGateReleasesOnError() async throws {
        struct Boom: Error {}
        let gate = AsyncGate()
        do { _ = try await withGate(gate) { throw Boom() } as Int; XCTFail() } catch is Boom {}
        let v = try await withGate(gate) { 7 }
        XCTAssertEqual(v, 7)
    }

    /// Cancelling the caller reaches the synchronous detached loop.
    func testDetachedWorkSeesCancellation() async throws {
        let started = expectation(description: "started")
        let task = Task {
            try await detachedCancellable { flag -> Int in
                started.fulfill()
                var spins = 0
                while !flag.isSet {
                    spins += 1
                    if spins > 50_000_000 { return -1 }
                    usleep(100)
                }
                throw CancellationError()
            }
        }
        await fulfillment(of: [started], timeout: 5)
        task.cancel()
        do { _ = try await task.value; XCTFail("expected cancellation") } catch is CancellationError {}
    }

    /// MLComputePlan on the compiled detector (the phone ships .mlmodelc).
    func testComputePlanOnCompiledDetector() async throws {
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-live-models")
        let compiled = (try? FileManager.default.contentsOfDirectory(at: cache, includingPropertiesForKeys: nil))?
            .first { $0.lastPathComponent.hasPrefix("TrackNetV3") && $0.pathExtension == "mlmodelc" }
        let url = try XCTUnwrap(compiled, "run LiveEndToEndTests once to compile the model into \(cache.path)")
        let report = await ComputePlanReport.summarize(compiledModel: url)
        print("[compute plan] \(report)")
        XCTAssertNil(report["error"], "\(report)")
        XCTAssertNotNil(report["cost_share"])
    }

    /// One JSON line per stage with the resource fields.
    func testMetricsWritesJSONLines() async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-metrics-\(UUID().uuidString)/m.jsonl")
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let m = StageMetrics(url: url)
        _ = try await m.measure("detect", ["x": .number(1)]) { 42 }
        struct Boom: Error {}
        _ = try? await m.measure("players") { () async throws -> Int in throw Boom() }
        let lines = try String(contentsOf: url, encoding: .utf8).split(separator: "\n")
        XCTAssertEqual(lines.count, 2)
        let first = try ShuttlJSON.decoder().decode(JSONValue.self, from: Data(lines[0].utf8))
        XCTAssertEqual(first["stage"]?.string, "detect")
        XCTAssertEqual(first["outcome"]?.string, "ok")
        XCTAssertNotNil(first["seconds"]?.double)
        XCTAssertGreaterThan(first["footprint_peak_mb"]?.double ?? 0, 1)
        XCTAssertNotNil(first["thermal_end"]?.string)
        let second = try ShuttlJSON.decoder().decode(JSONValue.self, from: Data(lines[1].utf8))
        XCTAssertEqual(second["outcome"]?.string, "error")
        XCTAssertEqual(StageMetrics(url: nil).url, nil)
    }
}

final class OverlapBox: @unchecked Sendable {
    private let lock = NSLock()
    private var current = 0
    private(set) var maxConcurrent = 0
    private(set) var completed = 0
    func enter() { lock.withLock { current += 1; maxConcurrent = max(maxConcurrent, current) } }
    func leave() { lock.withLock { current -= 1; completed += 1 } }
}
