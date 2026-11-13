import Foundation
import ShuttlCore

/// One failed stage, as recorded in diagnostics["stage_failures"].
public struct StageFailure: Sendable, Equatable {
    public static let diagnosticsKey = "stage_failures"

    public var stage: String
    /// "<Swift type>: <description>" — the same text as diagnostics["stages"][name]["error"].
    public var message: String
    /// NSError bridging of the error (Swift errors bridge to "<Module>.<Type>" / case index).
    public var domain: String
    public var code: Int
    /// Looks like an interruption or a transient resource problem (backgrounded GPU,
    /// media services reset, reader interrupted, memory pressure) — worth retrying.
    public var retryable: Bool
    /// How many times this exact failure happened (per-rally stages repeat).
    public var occurrences: Int

    public init(stage: String, message: String, domain: String, code: Int, retryable: Bool, occurrences: Int = 1) {
        self.stage = stage; self.message = message; self.domain = domain; self.code = code
        self.retryable = retryable; self.occurrences = occurrences
    }

    public init(stage: String, error: Error) {
        let ns = error as NSError
        self.init(stage: stage, message: "\(type(of: error)): \(error)", domain: ns.domain, code: ns.code,
                  retryable: AnalysisErrorClassifier.isRetryable(error))
    }

    public var json: JSONValue {
        .object(["stage": .string(stage), "error": .string(message), "domain": .string(domain),
                 "code": .number(Double(code)), "retryable": .bool(retryable),
                 "occurrences": .number(Double(occurrences))])
    }

    public init?(json: JSONValue) {
        guard let stage = json["stage"]?.string else { return nil }
        let message = json["error"]?.string ?? ""
        self.init(stage: stage, message: message, domain: json["domain"]?.string ?? "",
                  code: Int(json["code"]?.double ?? 0),
                  retryable: json["retryable"]?.bool
                      ?? AnalysisErrorClassifier.isRetryable(message: message, domain: json["domain"]?.string ?? ""),
                  occurrences: Int(json["occurrences"]?.double ?? 1))
    }
}

/// Decides whether an analysis error is worth an automatic retry.
///
/// Retryable = the OS interrupted or starved the work: CoreML / Vision / AVFoundation
/// / Metal errors, or messages about background execution, interruption, memory or
/// resources. Everything else (bad model, corrupt / missing video, logic errors) is not:
/// retrying would fail the same way.
public enum AnalysisErrorClassifier {
    public static let retryableDomains: Set<String> = [
        "AVFoundationErrorDomain", "com.apple.CoreML", "com.apple.Vision", "VNErrorDomain",
        "MTLCommandBufferErrorDomain", "com.apple.coremedia", "CMSampleBufferErrorDomain",
    ]
    static let typeMarkers = ["MLModelError", "AVError", "VNError", "MTLCommandBufferError", "StageStalledError"]
    static let keywords = [
        "from background", "in background", "in the background", "background execution", "backgroundexecution",
        "backgrounded", "interrupted", "interruption", "insufficient permission", "not permitted",
        "media services", "mediaserverd", "cannot complete action", "resource busy", "resource unavailable",
        "resources exhausted", "insufficient resources", "out of memory", "memory pressure", "timed out",
        "gpu", "neural engine", "error computing nn outputs",
        // ShuttlDetector DetectorError.backgroundChanged: the in-memory plan was damaged, but the
        // detect_partial checkpoint still holds the good bytes, so a retry resumes correctly.
        "background plan segments",
    ]
    /// Messages that mean "this will fail again" even when a keyword above also matches.
    static let permanentMarkers = [
        "modelunavailable", "model unavailable", "badmodel", "unsupported detector model",
        "novideotrack", "no video track", "does not exist", "no such file",
    ]

    public static func isRetryable(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        // VideoIO's watchdog: another client (the record-screen camera) held the hardware decoder.
        // Matched by case, not by its wording.
        if case ShuttlCoreError.decodeStalled? = error as? ShuttlCoreError { return true }
        // A stage watchdog (players: Vision perform() blocked); the retry resumes from the partial.
        if error is StageStalledError { return true }
        let ns = error as NSError
        let message = "\(type(of: error)): \(error) \(ns.localizedDescription)"
        if permanentMarkers.contains(where: message.lowercased().contains) { return false }
        var domains = [ns.domain]
        if let u = ns.userInfo[NSUnderlyingErrorKey] as? NSError { domains.append(u.domain) }
        return domains.contains(where: retryableDomains.contains) || isRetryable(message: message, domain: "")
    }

    /// Text-only verdict (diagnostics written by an older build carry only the message).
    public static func isRetryable(message: String, domain: String) -> Bool {
        let m = message.lowercased()
        if permanentMarkers.contains(where: m.contains) { return false }
        if retryableDomains.contains(domain) { return true }
        if typeMarkers.contains(where: message.contains) { return true }
        if retryableDomains.contains(where: message.contains) { return true }
        return keywords.contains(where: m.contains)
    }
}
