import Darwin
import Foundation
import ShuttlAnalysis
import ShuttlCore

/// Memory / thermal snapshot of this process.
public struct ResourceSnapshot: Sendable, Equatable {
    /// phys_footprint — what jetsam judges — in MB.
    public var physFootprintMB: Double
    /// Lifetime peak phys_footprint in MB.
    public var peakPhysFootprintMB: Double
    public var thermal: ThermalLevel
    public var lowPowerMode: Bool

    public static func now() -> ResourceSnapshot {
        let (fp, peak) = ResourceSampler.footprintBytes()
        return ResourceSnapshot(physFootprintMB: Double(fp) / 1_048_576, peakPhysFootprintMB: Double(peak) / 1_048_576,
                                thermal: ThermalLevel(ProcessInfo.processInfo.thermalState),
                                lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled)
    }
}

/// `StageObserver` that records per stage the peak phys_footprint (sampled every
/// `interval` on a utility queue, cross-checked with the kernel's lifetime peak) and the
/// worst thermal state seen, as diagnostics["stages"][name]["peak_phys_footprint_mb" /
/// "thermal_state"], and logs one line per stage to `ShuttlLog.pipeline`.
///
///     AnalysisPipeline(stages: stages, checkpoints: store, observer: ResourceSampler())
public final class ResourceSampler: StageObserver, @unchecked Sendable {
    private struct Running {
        let timer: DispatchSourceTimer
        let box: PeakBox
        let lifetimeBefore: UInt64
        let thermalStart: ThermalLevel
    }

    private final class PeakBox: @unchecked Sendable {
        private let lock = NSLock()
        private var peak: UInt64 = 0
        func add(_ v: UInt64) { lock.withLock { peak = max(peak, v) } }
        var value: UInt64 { lock.withLock { peak } }
    }

    private let lock = NSLock()
    private var running: [String: Running] = [:]
    private var results: [String: [String: JSONValue]] = [:]
    private let interval: DispatchTimeInterval
    private let thermal: @Sendable () -> ThermalLevel
    private let queue = DispatchQueue(label: "com.example.shuttl.resource-sampler", qos: .utility)
    public let logs: Bool

    public init(interval: DispatchTimeInterval = .milliseconds(50), logs: Bool = true,
                thermal: @escaping @Sendable () -> ThermalLevel = { ThermalLevel(ProcessInfo.processInfo.thermalState) }) {
        self.interval = interval
        self.logs = logs
        self.thermal = thermal
    }

    public func stageDidStart(_ name: String) {
        let (fp, lifetime) = Self.footprintBytes()
        let box = PeakBox()
        box.add(fp)
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + interval, repeating: interval)
        timer.setEventHandler { box.add(Self.footprintBytes().current) }
        let r = Running(timer: timer, box: box, lifetimeBefore: lifetime, thermalStart: thermal())
        let old = lock.withLock { () -> Running? in
            let o = running[name]
            running[name] = r
            return o
        }
        old?.timer.cancel()
        timer.resume()
    }

    public func stageDidFinish(_ name: String, seconds: Double) {
        guard let r = lock.withLock({ running.removeValue(forKey: name) }) else { return }
        r.timer.cancel()
        let (fp, lifetime) = Self.footprintBytes()
        r.box.add(fp)
        var peak = r.box.value
        if lifetime > r.lifetimeBefore { peak = max(peak, lifetime) }     // the exact peak when the kernel's rose
        let worst = max(r.thermalStart, thermal())
        let mb = (Double(peak) / 1_048_576 * 10).rounded() / 10
        lock.withLock {
            results[name] = ["peak_phys_footprint_mb": .number(mb), "thermal_state": .string(worst.name)]
        }
        if logs {
            ShuttlLog.pipeline.info("stage \(name, privacy: .public) \(seconds, format: .fixed(precision: 2), privacy: .public) s peak \(mb, format: .fixed(precision: 1), privacy: .public) MB thermal \(worst.name, privacy: .public)")
        }
    }

    public func stageDiagnostics(_ name: String) -> [String: JSONValue] {
        lock.withLock { results[name] ?? [:] }
    }

    /// (phys_footprint, lifetime peak) in bytes; (0, 0) if unavailable.
    public static func footprintBytes() -> (current: UInt64, lifetimePeak: UInt64) {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(taskPort, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return (0, 0) }
        return (info.phys_footprint, UInt64(max(info.ledger_phys_footprint_peak, 0)))
    }

    /// Fetched once: each task_self_trap() adds a send-right reference to the port.
    private static let taskPort: mach_port_t = task_self_trap()
}
