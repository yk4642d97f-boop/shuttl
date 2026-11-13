import XCTest
import ShuttlAnalysis
import ShuttlCore
import ShuttlRuntime

final class FakeClock: @unchecked Sendable {
    private let lock = NSLock()
    private var t = Date(timeIntervalSince1970: 1_790_000_000)
    func tick() -> Date { lock.withLock { t = t.addingTimeInterval(1); return t } }
    func same() -> Date { lock.withLock { t } }
}

final class DiagnosticsFileStoreTests: XCTestCase {
    private func docs() -> URL {
        let d = FileManager.default.temporaryDirectory.appendingPathComponent("shuttl-diag-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: d) }
        return d
    }

    func testKeepsNewest20AndNeverTouchesOtherFiles() throws {
        let documents = docs(), fm = FileManager.default
        let diagnostics = documents.appendingPathComponent("Diagnostics")
        try fm.createDirectory(at: diagnostics.appendingPathComponent("MetricKit"), withIntermediateDirectories: true)
        let stageMetrics = diagnostics.appendingPathComponent("stage-metrics.jsonl")
        try Data("{\"event\":\"stage\"}\n".utf8).write(to: stageMetrics)
        let foreign = diagnostics.appendingPathComponent("MetricKit/notes.json")
        try Data("{}".utf8).write(to: foreign)

        let clock = FakeClock()
        let base = DiagnosticsFileStore.metricKit(documents: documents)
        XCTAssertEqual(base.directory.path, diagnostics.appendingPathComponent("MetricKit").path)
        let store = DiagnosticsFileStore(directory: base.directory, maxFiles: base.maxFiles) { clock.tick() }
        var written: [URL] = []
        for i in 0..<25 { written.append(try store.write(Data("{\"i\":\(i)}".utf8), kind: i % 2 == 0 ? "metrics" : "diagnostics")) }
        let kept = store.files()
        XCTAssertEqual(kept.count, 20)
        XCTAssertEqual(kept, Array(written.suffix(20)), "the 5 oldest were pruned")
        XCTAssertTrue(written.prefix(5).allSatisfy { !fm.fileExists(atPath: $0.path) })
        XCTAssertEqual(try String(contentsOf: stageMetrics, encoding: .utf8), "{\"event\":\"stage\"}\n", "stage-metrics.jsonl untouched")
        XCTAssertTrue(fm.fileExists(atPath: foreign.path), "files the store didn't name are left alone")
        XCTAssertTrue(kept.last!.lastPathComponent.hasSuffix("-metrics.json"))
        XCTAssertEqual(try Data(contentsOf: kept.last!), Data("{\"i\":24}".utf8))
    }

    func testSameMillisecondAndOddKindsGetDistinctSafeNames() throws {
        let clock = FakeClock()
        let store = DiagnosticsFileStore(directory: docs().appendingPathComponent("M"), maxFiles: 5) { clock.same() }
        let a = try store.write(Data("1".utf8), kind: "metrics")
        let b = try store.write(Data("2".utf8), kind: "metrics")
        let c = try store.write(Data("3".utf8), kind: "../evil kind")
        XCTAssertNotEqual(a, b)
        XCTAssertEqual(c.deletingLastPathComponent().standardizedFileURL.path, store.directory.standardizedFileURL.path, "no path escape")
        XCTAssertTrue(c.lastPathComponent.hasSuffix("-___evil_kind.json"))
        XCTAssertEqual(store.files().count, 3)
    }

    func testLogSubsystem() {
        XCTAssertEqual(ShuttlLog.subsystem, "com.example.shuttl")
        ShuttlLog.pipeline.debug("test line")                     // compiles + doesn't crash
    }
}

final class ResourceSamplerTests: XCTestCase {
    final class Levels: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [ThermalLevel]
        init(_ v: [ThermalLevel]) { values = v }
        func next() -> ThermalLevel { lock.withLock { values.count > 1 ? values.removeFirst() : values[0] } }
    }

    func testRecordsPeakFootprintAndWorstThermalPerStage() throws {
        let levels = Levels([.fair, .serious])
        let s = ResourceSampler(interval: .milliseconds(5), logs: false) { levels.next() }
        let before = ResourceSnapshot.now()
        XCTAssertGreaterThan(before.physFootprintMB, 1)
        s.stageDidStart("detect")
        let bytes = 96 << 20
        let p = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 16_384)
        memset(p, 0x5A, bytes)                                      // touch every page
        Thread.sleep(forTimeInterval: 0.15)
        p.deallocate()
        s.stageDidFinish("detect", seconds: 0.15)
        let d = s.stageDiagnostics("detect")
        let peak = try XCTUnwrap(d["peak_phys_footprint_mb"]?.double)
        XCTAssertGreaterThan(peak, before.physFootprintMB + 60, "the 96 MB inside the stage is seen")
        XCTAssertEqual(d["thermal_state"]?.string, "serious", "worst of start / end")
        XCTAssertEqual(s.stageDiagnostics("never-ran"), [:])
        s.stageDidFinish("never-ran", seconds: 0)                  // no-op
    }

    func testRepeatedStagesAndPipelineMergeKeys() {
        let s = ResourceSampler(interval: .milliseconds(5), logs: true)
        for _ in 0..<50 { s.stageDidStart("shots"); s.stageDidFinish("shots", seconds: 0) }
        let d = s.stageDiagnostics("shots")
        XCTAssertNotNil(d["peak_phys_footprint_mb"]?.double)
        XCTAssertEqual(Set(d.keys), ["peak_phys_footprint_mb", "thermal_state"], "never shadows the pipeline's own keys")
        XCTAssertNotNil(ThermalLevel.allCases.first { $0.name == d["thermal_state"]?.string })
    }
}
