import CoreML
import Darwin
import Foundation
import ShuttlCore

/// One JSON line per stage: wall time, memory (phys_footprint now and lifetime
/// peak, what jetsam judges), thermal state and Low Power Mode. On the phone it goes
/// to Documents/Diagnostics/stage-metrics.jsonl (pull with devicectl); on macOS only
/// when SHUTTL_STAGE_METRICS names a file.
final class StageMetrics: @unchecked Sendable {
    let url: URL?
    private let lock = NSLock()
    private static let maxBytes = 4_000_000

    init(url: URL?) {
        self.url = url
        if let url {
            try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        }
    }

    static var defaultURL: URL? {
        #if os(iOS)
        return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
            .appendingPathComponent("Diagnostics/stage-metrics.jsonl")
        #else
        return ProcessInfo.processInfo.environment["SHUTTL_STAGE_METRICS"].map { URL(fileURLWithPath: $0) }
        #endif
    }

    func record(_ event: String, _ fields: [String: JSONValue]) {
        guard let url else { return }
        var obj = fields
        obj["event"] = .string(event)
        obj["ts"] = .string(ISO8601DateFormatter().string(from: Date()))
        guard var line = try? ShuttlJSON.encoder().encode(JSONValue.object(obj)) else { return }
        line.append(0x0A)
        lock.withLock {
            let fm = FileManager.default
            if let size = (try? fm.attributesOfItem(atPath: url.path)[.size] as? Int), size > Self.maxBytes {
                let old = url.appendingPathExtension("old")
                try? fm.removeItem(at: old)
                try? fm.moveItem(at: url, to: old)
            }
            if !fm.fileExists(atPath: url.path) { fm.createFile(atPath: url.path, contents: nil) }
            guard let h = try? FileHandle(forWritingTo: url) else { return }
            defer { try? h.close() }
            _ = try? h.seekToEnd()
            try? h.write(contentsOf: line)
        }
    }

    /// Times `body` and records it as stage `name` (ok, error or cancelled).
    func measure<T>(_ name: String, _ extra: [String: JSONValue] = [:], _ body: () async throws -> T) async throws -> T {
        guard url != nil else { return try await body() }
        let start = Snapshot.now()
        do {
            let value = try await body()
            record("stage", fields(name, start, outcome: "ok", extra))
            return value
        } catch is CancellationError {
            record("stage", fields(name, start, outcome: "cancelled", extra))
            throw CancellationError()
        } catch {
            var e = extra
            e["error"] = .string(String(describing: error))
            record("stage", fields(name, start, outcome: "error", e))
            throw error
        }
    }

    private func fields(_ name: String, _ start: Snapshot, outcome: String, _ extra: [String: JSONValue]) -> [String: JSONValue] {
        let end = Snapshot.now()
        var f = extra
        f["stage"] = .string(name)
        f["outcome"] = .string(outcome)
        f["seconds"] = .number(((end.uptime - start.uptime) * 1000).rounded() / 1000)
        f["footprint_mb_start"] = .number(start.footprintMB)
        f["footprint_mb_end"] = .number(end.footprintMB)
        f["footprint_peak_mb"] = .number(end.peakMB)
        f["thermal_start"] = .string(start.thermal)
        f["thermal_end"] = .string(end.thermal)
        f["low_power"] = .bool(end.lowPower)
        return f
    }
}

struct Snapshot {
    var uptime: Double
    var footprintMB: Double
    var peakMB: Double
    var thermal: String
    var lowPower: Bool

    static func now() -> Snapshot {
        let (fp, peak) = memory()
        return Snapshot(uptime: ProcessInfo.processInfo.systemUptime, footprintMB: fp, peakMB: peak,
                        thermal: thermalName(ProcessInfo.processInfo.thermalState),
                        lowPower: ProcessInfo.processInfo.isLowPowerModeEnabled)
    }

    /// (phys_footprint, lifetime peak) in MB — the numbers jetsam judges; NaN if unavailable.
    static func memory() -> (Double, Double) {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(task_self_trap(), task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return (.nan, .nan) }
        let mb = 1024.0 * 1024.0
        return ((Double(info.phys_footprint) / mb * 10).rounded() / 10,
                (Double(info.ledger_phys_footprint_peak) / mb * 10).rounded() / 10)
    }

    static func thermalName(_ s: ProcessInfo.ThermalState) -> String {
        switch s {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}

/// Which compute unit Core ML plans each op of a compiled model for, weighted by
/// its estimated cost (iOS 17.4+). Tells whether the detector really runs on the ANE.
enum ComputePlanReport {
    static func summarize(compiledModel url: URL) async -> [String: JSONValue] {
        guard url.pathExtension == "mlmodelc" else { return ["skipped": .string("not compiled")] }
        guard #available(iOS 17.4, macOS 14.4, *) else { return ["skipped": .string("os too old")] }
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            let plan = try await MLComputePlan.load(contentsOf: url, configuration: config)
            guard case .program(let program) = plan.modelStructure,
                  let main = program.functions["main"] else { return ["skipped": .string("not an ML program")] }
            var cost: [String: Double] = [:], ops: [String: Int] = [:]
            func visit(_ block: MLModelStructure.Program.Block) {
                for op in block.operations {
                    let device = plan.deviceUsage(for: op).map { name($0.preferred) } ?? "none"
                    ops[device, default: 0] += 1
                    cost[device, default: 0] += plan.estimatedCost(of: op)?.weight ?? 0
                    op.blocks.forEach(visit)
                }
            }
            visit(main.block)
            return ["ops": .object(ops.mapValues { .number(Double($0)) }),
                    "cost_share": .object(cost.mapValues { .number(($0 * 1000).rounded() / 1000) })]
        } catch {
            return ["error": .string(String(describing: error))]
        }
    }

    @available(iOS 17.4, macOS 14.4, *)
    private static func name(_ d: MLComputeDevice) -> String {
        switch d {
        case .cpu: return "cpu"
        case .gpu: return "gpu"
        case .neuralEngine: return "ane"
        @unknown default: return "other"
        }
    }
}
