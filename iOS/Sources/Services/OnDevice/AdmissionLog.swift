import Foundation
import ShuttlCore

/// Why an on-device analysis is not allowed to start yet. The pipeline admits one analysis
/// at a time, never while a take is recorded or the camera runs, and not while the phone is hot.
enum AdmissionBlock: String, Sendable, Equatable {
    /// Another on-device analysis holds the slot.
    case busy
    /// A take is being recorded.
    case recording
    /// The record screen's camera is running: its encoder / preview shares the hardware codec
    /// with the analysis decode (the phone showed a 15-min stall and AVError -11847 that way).
    case camera
    /// The thermal state holds new starts (serious; after critical, until fair).
    case thermal

    /// `PipelineActivity.localStage` shown while waiting for this reason. A thermal wait with
    /// the record camera running says so ("cooling_camera"): the preview is what keeps it hot.
    func localStage(cameraLive: Bool) -> String {
        switch self {
        case .busy: return "queued"
        case .recording: return "recording"
        case .camera: return "camera"
        case .thermal: return cameraLive ? "cooling_camera" : "cooling"
        }
    }
}

/// Appends admission waits and pauses to Documents/Diagnostics/stage-metrics.jsonl, the same JSON-lines
/// file (and format) LiveStages writes per stage, so one pull explains a late start or a gap:
///   {"event":"admission_wait","phase":"begin","reason":"thermal","thermal":"serious","camera_live":true,"ts":…}
///   {"event":"admission_wait","phase":"end","reason":"thermal","seconds":412.3,"thermal":"fair","ts":…}
///   {"event":"analysis_paused","reason":"camera","stage":"detect","frame":504,"frames":3600,"ts":…}
///   {"event":"analysis_resumed","reason":"camera","seconds":91.2,"ts":…}
struct AdmissionLog: Sendable {
    let url: URL?

    static var `default`: AdmissionLog {
        #if os(iOS)
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        return AdmissionLog(url: docs?.appendingPathComponent("Diagnostics/stage-metrics.jsonl"))
        #else
        return AdmissionLog(url: nil)
        #endif
    }

    func record(_ fields: [String: JSONValue]) {
        record(event: "admission_wait", fields)
    }

    func record(event: String, _ fields: [String: JSONValue]) {
        guard let url else { return }
        var obj = fields
        obj["event"] = .string(event)
        obj["ts"] = .string(ISO8601DateFormatter().string(from: Date()))
        guard var line = try? ShuttlJSON.encoder().encode(JSONValue.object(obj)) else { return }
        line.append(0x0A)
        let fm = FileManager.default
        try? fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !fm.fileExists(atPath: url.path) { fm.createFile(atPath: url.path, contents: nil) }
        guard let h = try? FileHandle(forWritingTo: url) else { return }
        defer { try? h.close() }
        _ = try? h.seekToEnd()
        try? h.write(contentsOf: line)
    }

    static func seconds(_ d: Duration) -> Double {
        Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
    }
}
