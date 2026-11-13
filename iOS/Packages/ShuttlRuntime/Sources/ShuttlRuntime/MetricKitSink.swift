import Foundation
#if canImport(MetricKit) && os(iOS)
import MetricKit

/// Receives MetricKit payloads (daily metrics; crash / hang / CPU / disk-write diagnostics,
/// delivered on the next launch) and writes each as JSON to Documents/Diagnostics/MetricKit,
/// keeping the newest 20. Keep one instance alive for the app's lifetime:
///
///     let metricKit = MetricKitSink()     // e.g. a stored property of the App
///     metricKit.start()
public final class MetricKitSink: NSObject, MXMetricManagerSubscriber, @unchecked Sendable {
    public let store: DiagnosticsFileStore

    public init(store: DiagnosticsFileStore = .metricKit()) {
        self.store = store
        super.init()
    }

    public func start() { MXMetricManager.shared.add(self) }
    public func stop() { MXMetricManager.shared.remove(self) }

    public func didReceive(_ payloads: [MXMetricPayload]) {
        for p in payloads { save(p.jsonRepresentation(), kind: "metrics") }
    }

    public func didReceive(_ payloads: [MXDiagnosticPayload]) {
        for p in payloads { save(p.jsonRepresentation(), kind: "diagnostics") }
    }

    private func save(_ data: Data, kind: String) {
        do {
            let url = try store.write(data, kind: kind)
            ShuttlLog.pipeline.info("MetricKit \(kind, privacy: .public) → \(url.lastPathComponent, privacy: .public)")
        } catch {
            ShuttlLog.pipeline.error("MetricKit \(kind, privacy: .public) not saved: \(String(describing: error), privacy: .public)")
        }
    }
}
#endif
