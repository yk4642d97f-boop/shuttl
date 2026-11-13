import CoreML
import Foundation
import ShuttlCore

/// Errors from the on-device learned hit detector.
public enum HitNetError: Error, CustomStringConvertible {
    case modelUnavailable(String)
    case badModel(String)
    case prediction(String)
    case badInput(String)

    public var description: String {
        switch self {
        case .modelUnavailable(let s): return "hit model unavailable: \(s)"
        case .badModel(let s): return "hit model contract mismatch: \(s)"
        case .prediction(let s): return "hit model prediction failed: \(s)"
        case .badInput(let s): return "hit model bad input: \(s)"
        }
    }
}

/// Anything that turns a (T,51) feature matrix into (T,2) logits (the CoreML model; fakes in tests).
public protocol HitLogitsProvider: Sendable {
    /// `features` is (T,51) row-major. Returns 2*T floats, row-major.
    func logits(features: [Float], frames T: Int) throws -> [Float]
}

/// The bundled `HitNetDetector.mlpackage`: the 3-checkpoint bidirectional dilated TCN ensemble
/// whose logits are averaged inside the graph (research/hitnet/REPORT.md, "3-TCN logit-average
/// ensemble"; research/hitnet/coreml/README.md).
///
/// Input `features` (1,T,51) float32 — one row per video frame from `HitFeatures.derive`, T
/// dynamic in 1...4096. Output `logits` (1,T,2) float32 — raw pre-softmax `[hit, hitter-is-far]`.
/// One `predict()` per rally; the net is non-causal, so it sees the whole sequence at once.
///
/// fp32 only. fp16 was catastrophic for the sibling BST port (12–54 % argmax flips) and this
/// decoder thresholds probabilities, so the fp16 spelling must not be shipped.
public final class HitNetModel: HitLogitsProvider, @unchecked Sendable {
    public static let resourceName = "HitNetDetector"
    /// The largest T the exported graph accepts (RangeDim(1, 4096)).
    public static let maxFrames = 4096

    public let model: MLModel
    public let computeUnits: MLComputeUnits
    public let metadata: [String: String]
    private let lock = NSLock()
    public private(set) var predictions = 0

    /// The decoder the model was measured with (metadata "decoder").
    public static let defaultParams = HitDecoder.Params(thr: 0.4, gapS: 0.3)

    /// Loads an already compiled `.mlmodelc` and checks its contract (name, feature count).
    public init(compiledModelURL: URL, computeUnits: MLComputeUnits = HitNetModel.defaultComputeUnits) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        let m: MLModel
        do { m = try MLModel(contentsOf: compiledModelURL, configuration: cfg) } catch {
            throw HitNetError.modelUnavailable("\(compiledModelURL.lastPathComponent): \(error.localizedDescription)")
        }
        model = m
        self.computeUnits = computeUnits
        let desc = m.modelDescription
        metadata = desc.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        guard let inp = desc.inputDescriptionsByName["features"] else {
            throw HitNetError.badModel("no 'features' input (inputs: \(desc.inputDescriptionsByName.keys.sorted()))")
        }
        // (1,T,51): the last axis is fixed at 51 and a model with a different width would be
        // silently wrong on every frame, so pin it; T is a range and is checked per call.
        let shape = inp.multiArrayConstraint?.shape.map(\.intValue) ?? []
        guard shape.count == 3, shape[2] == HitFeatures.count else {
            throw HitNetError.badModel("input features shape \(shape), expected (1,T,\(HitFeatures.count))")
        }
        guard inp.multiArrayConstraint?.dataType == .float32 else {
            throw HitNetError.badModel("input features is not float32")
        }
        guard desc.outputDescriptionsByName["logits"] != nil else {
            throw HitNetError.badModel("no 'logits' output")
        }
        if let n = metadata["n_in"], n != String(HitFeatures.count) {
            throw HitNetError.badModel("metadata n_in \(n) differs from \(HitFeatures.count)")
        }
    }

    /// CPU only: the export measured CPU and CPU_AND_NE as identical on these small 1-D
    /// convolutions (no ANE win), so the simpler unit is the shipped one.
    public static let defaultComputeUnits: MLComputeUnits = .cpuOnly

    /// Compiles `modelURL` (.mlpackage) at runtime, or loads it when it is already an `.mlmodelc`.
    /// `cacheDirectory` keeps the compiled model keyed by name, size and date.
    public static func load(modelURL: URL, computeUnits: MLComputeUnits = defaultComputeUnits,
                            cacheDirectory: URL? = nil) async throws -> HitNetModel {
        let fm = FileManager.default
        guard fm.fileExists(atPath: modelURL.path) else {
            throw HitNetError.modelUnavailable("\(modelURL.path) does not exist")
        }
        if modelURL.pathExtension == "mlmodelc" {
            return try HitNetModel(compiledModelURL: modelURL, computeUnits: computeUnits)
        }
        if let dir = cacheDirectory, let m = loadCached(modelURL: modelURL, cacheDirectory: dir, computeUnits: computeUnits) {
            return m
        }
        let compiled: URL
        do { compiled = try await MLModel.compileModel(at: modelURL) } catch {
            throw HitNetError.modelUnavailable("compile \(modelURL.lastPathComponent): \(error.localizedDescription)")
        }
        var url = compiled
        if let dir = cacheDirectory {
            let c = dir.appendingPathComponent("\(cacheName(modelURL)).mlmodelc")
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
            try? fm.removeItem(at: c)
            if (try? fm.moveItem(at: compiled, to: c)) != nil { url = c }
        }
        return try HitNetModel(compiledModelURL: url, computeUnits: computeUnits)
    }

    /// `resourceName-size-mtime.mlmodelc`: the key that makes a stale compile impossible.
    static func cacheName(_ url: URL) -> String {
        "\(url.deletingPathExtension().lastPathComponent)-\(directorySize(url))-\(cacheStamp(url))"
    }

    /// The bundled model: the app bundle's `.mlmodelc` if it was linked there, else the package
    /// resource (ShuttlHitModel; Xcode compiles the `.mlpackage` on first load).
    public static func bundledModelURL(in bundle: Bundle = .main) -> URL? {
        bundle.url(forResource: resourceName, withExtension: "mlmodelc")
            ?? bundle.url(forResource: resourceName, withExtension: "mlpackage")
            ?? HitModelResource.modelURL
    }

    /// Loads `modelURL` through `cacheDirectory`, compiling an `.mlpackage` into it first.
    public static func loadCached(modelURL: URL, cacheDirectory: URL,
                                  computeUnits: MLComputeUnits = defaultComputeUnits) -> HitNetModel? {
        let fm = FileManager.default
        guard fm.fileExists(atPath: modelURL.path) else { return nil }
        if modelURL.pathExtension == "mlmodelc" {
            return try? HitNetModel(compiledModelURL: modelURL, computeUnits: computeUnits)
        }
        let c = cacheDirectory.appendingPathComponent("\(cacheName(modelURL)).mlmodelc")
        if fm.fileExists(atPath: c.path), let m = try? HitNetModel(compiledModelURL: c, computeUnits: computeUnits) {
            return m
        }
        return nil
    }

    /// Compiles an `.mlpackage` into `cacheDirectory` and loads it from there; nil on any failure.
    public static func compileAndLoad(modelURL: URL, cacheDirectory: URL,
                                      computeUnits: MLComputeUnits = defaultComputeUnits) async -> HitNetModel? {
        if let cached = loadCached(modelURL: modelURL, cacheDirectory: cacheDirectory, computeUnits: computeUnits) {
            return cached
        }
        guard let compiled = try? await MLModel.compileModel(at: modelURL) else { return nil }
        let fm = FileManager.default
        let c = cacheDirectory.appendingPathComponent("\(cacheName(modelURL)).mlmodelc")
        try? fm.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        try? fm.removeItem(at: c)
        guard (try? fm.moveItem(at: compiled, to: c)) != nil else {
            return try? HitNetModel(compiledModelURL: compiled, computeUnits: computeUnits)
        }
        return try? HitNetModel(compiledModelURL: c, computeUnits: computeUnits)
    }

    /// The bundled model, compiled if needed; nil when it is missing or does not load.
    public static func loadBundled(in bundle: Bundle = .main, cacheDirectory: URL? = nil) async -> HitNetModel? {
        guard let url = bundledModelURL(in: bundle) else { return nil }
        return try? await load(modelURL: url, cacheDirectory: cacheDirectory)
    }

    // MARK: - Inference

    /// (T,51) row-major -> (T,2) row-major, one whole-sequence call.
    public func logits(features: [Float], frames T: Int) throws -> [Float] {
        guard T > 0, features.count == T * HitFeatures.count else {
            throw HitNetError.badInput("features \(features.count) floats for T = \(T)")
        }
        guard T <= Self.maxFrames else {
            throw HitNetError.badInput("T = \(T) exceeds the graph's \(Self.maxFrames)-frame limit")
        }
        let a = try Self.multiArray(features, [1, T, HitFeatures.count])
        let provider: MLDictionaryFeatureProvider
        do { provider = try MLDictionaryFeatureProvider(dictionary: ["features": MLFeatureValue(multiArray: a)]) }
        catch { throw HitNetError.prediction("inputs: \(error.localizedDescription)") }
        let out: MLFeatureProvider
        do {
            out = try lock.withLock {
                predictions += 1
                return try model.prediction(from: provider)
            }
        } catch { throw HitNetError.prediction(error.localizedDescription) }
        guard let lg = out.featureValue(for: "logits")?.multiArrayValue else {
            throw HitNetError.prediction("missing logits")
        }
        // The output is a flexibly-shaped (1,T,2); read it as a flat fp32 buffer.
        let need = 2 * T
        guard lg.count >= need else {
            throw HitNetError.prediction("logits has \(lg.count) values, expected \(need)")
        }
        var buf = [Float](repeating: 0, count: need)
        switch lg.dataType {
        case .float32:
            lg.withUnsafeBufferPointer(ofType: Float.self) { p in
                buf.withUnsafeMutableBufferPointer { o in o.baseAddress!.update(from: p.baseAddress!, count: need) }
            }
        case .double:
            lg.withUnsafeBufferPointer(ofType: Double.self) { p in
                for i in 0..<need { buf[i] = Float(p[i]) }
            }
        default:
            for i in 0..<need { buf[i] = lg[i].floatValue }
        }
        return buf
    }

    static func multiArray(_ v: [Float], _ shape: [Int]) throws -> MLMultiArray {
        let a: MLMultiArray
        do { a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32) }
        catch { throw HitNetError.badInput("MLMultiArray \(shape): \(error.localizedDescription)") }
        a.withUnsafeMutableBufferPointer(ofType: Float.self) { p, strides in
            // a freshly allocated array is contiguous row-major, which is the model's layout
            var rowMajor = [Int](repeating: 1, count: shape.count)
            for i in stride(from: shape.count - 2, through: 0, by: -1) { rowMajor[i] = rowMajor[i + 1] * shape[i + 1] }
            precondition(strides == rowMajor && v.count == p.count, "unexpected MLMultiArray layout")
            v.withUnsafeBufferPointer { s in p.baseAddress!.update(from: s.baseAddress!, count: v.count) }
        }
        return a
    }

    static func cacheStamp(_ url: URL) -> Int {
        let manifest = url.appendingPathComponent("Manifest.json")
        let attrs = try? FileManager.default.attributesOfItem(atPath: FileManager.default.fileExists(atPath: manifest.path)
                                                             ? manifest.path : url.path)
        return Int((attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)
    }

    static func directorySize(_ url: URL) -> Int {
        var total = 0
        if let e = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) {
            for case let f as URL in e { total += (try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0 }
        }
        return total
    }
}

/// The bundled `HitNetDetector.mlpackage` (see `HitNetModel`).
public enum HitModelResource {
    /// Compiled by Xcode into `HitNetDetector.mlmodelc`, else the package itself (SwiftPM keeps
    /// `.mlpackage` verbatim; CoreML compiles it on first load).
    public static var modelURL: URL? {
        Bundle.module.url(forResource: "HitNetDetector", withExtension: "mlmodelc")
            ?? Bundle.module.url(forResource: "HitNetDetector", withExtension: "mlpackage")
    }
}
