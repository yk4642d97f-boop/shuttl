import Darwin
import Foundation
import XCTest
import ShuttlRuntime

/// Thread-safe log of what fake jobs did.
final class Recorder: @unchecked Sendable {
    private let lock = NSLock()
    private var _events: [String] = []
    private var _running = 0
    private var _maxRunning = 0
    var events: [String] { lock.withLock { _events } }
    var maxConcurrent: Int { lock.withLock { _maxRunning } }
    func log(_ s: String) { lock.withLock { _events.append(s) } }
    func enter() { lock.withLock { _running += 1; _maxRunning = max(_maxRunning, _running) } }
    func leave() { lock.withLock { _running -= 1 } }
}

/// A switch a fake job waits on (cancellation-aware, like the real pipeline).
final class Gate: @unchecked Sendable {
    private let lock = NSLock()
    private var open = false
    func release() { lock.withLock { open = true } }
    func wait() async throws {
        while !lock.withLock({ open }) { try await Task.sleep(for: .milliseconds(2)) }
    }
}

/// Polls `condition` until true or `timeout`.
func eventually(_ what: String, timeout: Double = 5, file: StaticString = #filePath, line: UInt = #line,
                _ condition: @Sendable () async -> Bool) async {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
        if await condition() { return }
        try? await Task.sleep(for: .milliseconds(2))
    }
    XCTFail("timed out waiting for: \(what)", file: file, line: line)
}

/// Records calls instead of driving a queue.
actor FakeQueueControl: AnalysisQueueControl {
    private(set) var calls: [String] = []
    private var stopWaiters: [CheckedContinuation<Void, Never>] = []
    var stopsImmediately = true

    func pause(_ reason: AnalysisPauseReason) { calls.append("pause:\(reason.rawValue)") }
    func resume(_ reason: AnalysisPauseReason) { calls.append("resume:\(reason.rawValue)") }
    func hold(_ reason: AnalysisPauseReason) { calls.append("hold:\(reason.rawValue)") }
    func release(_ reason: AnalysisPauseReason) { calls.append("release:\(reason.rawValue)") }
    func setStopsImmediately(_ v: Bool) { stopsImmediately = v }

    func waitForRunningJobToStop() async {
        calls.append("wait")
        guard !stopsImmediately else { return }
        await withCheckedContinuation { stopWaiters.append($0) }
    }

    /// The job has stopped: pending and later waits return (like the real queue, where
    /// nothing running means `waitForRunningJobToStop` returns at once).
    func jobStopped() {
        stopsImmediately = true
        let w = stopWaiters
        stopWaiters = []
        w.forEach { $0.resume() }
    }
}

final class FakeConditions: DeviceConditions, @unchecked Sendable {
    private let lock = NSLock()
    private var _level: ThermalLevel = .nominal
    private var _low = false
    var thermalLevel: ThermalLevel {
        get { lock.withLock { _level } }
        set { lock.withLock { _level = newValue } }
    }
    var isLowPowerModeEnabled: Bool {
        get { lock.withLock { _low } }
        set { lock.withLock { _low = newValue } }
    }
}

/// Peak phys_footprint growth (MB) while `body` runs: 1 ms sampling on a background
/// thread, cross-checked with the kernel's lifetime peak when that rose.
enum Footprint {
    private static func rusage() -> rusage_info_v4 {
        var info = rusage_info_v4()
        _ = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0) }
        }
        return info
    }

    static func measure(_ body: () throws -> Void) rethrows -> Double {
        final class Box: @unchecked Sendable {
            let lock = NSLock(); var running = true; var peak: UInt64 = 0
        }
        let box = Box()
        let base = rusage().ri_phys_footprint
        let lifeBefore = rusage().ri_lifetime_max_phys_footprint
        box.peak = base
        let done = DispatchSemaphore(value: 0)
        Thread.detachNewThread {
            while box.lock.withLock({ box.running }) {
                let now = Footprint.rusage().ri_phys_footprint
                box.lock.withLock { box.peak = max(box.peak, now) }
                usleep(1000)
            }
            done.signal()
        }
        try body()
        box.lock.withLock { box.running = false }
        done.wait()
        var peak = box.lock.withLock { box.peak }
        let lifeAfter = rusage().ri_lifetime_max_phys_footprint
        if lifeAfter > lifeBefore { peak = max(peak, lifeAfter) }
        return Double(peak > base ? peak - base : 0) / 1_048_576
    }
}
