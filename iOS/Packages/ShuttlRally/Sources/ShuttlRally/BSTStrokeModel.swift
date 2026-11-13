import CoreML
import Foundation
import ShuttlCore
import ShuttlStrokeModel

/// Errors from the on-device stroke model.
public enum StrokeModelError: Error, CustomStringConvertible {
    case modelUnavailable(String)
    case badModel(String)
    case prediction(String)

    public var description: String {
        switch self {
        case .modelUnavailable(let s): return "stroke model unavailable: \(s)"
        case .badModel(let s): return "stroke model contract mismatch: \(s)"
        case .prediction(let s): return "stroke model prediction failed: \(s)"
        }
    }
}

/// Anything that turns a clip into the 25 class logits (the CoreML model; fakes in tests).
public protocol StrokeLogitsProvider: Sendable {
    func logits(_ clip: StrokeClip) throws -> [Float]
}

/// CoreML inference for `BSTStroke.mlpackage`: the averaged-logit ensemble of the 10 released
/// BST-CG-AP seq30 checkpoints (research/bst/coreml/REPORT.md).
///
/// Inputs, batch 1, zero beyond `valid`: `joints` (1,30,2,17,2), `pos` (1,30,2,2), `shuttle`
/// (1,30,2), `valid` (1,30). Output `logits` (1,25), raw mean logits in `StrokeClasses.names` order.
/// Bones are computed inside the graph. fp32 (see the report for the fp16 measurement).
public final class BSTStrokeModel: StrokeLogitsProvider, @unchecked Sendable {
    public static let resourceName = "BSTStroke"
    public let model: MLModel
    public let computeUnits: MLComputeUnits
    public let metadata: [String: String]
    private let lock = NSLock()
    public private(set) var predictions = 0

    static let shapes: [String: [Int]] = ["joints": [1, 30, 2, 17, 2], "pos": [1, 30, 2, 2],
                                          "shuttle": [1, 30, 2], "valid": [1, 30]]

    /// Loads an already compiled `.mlmodelc` and checks its contract (names, shapes, class list).
    public init(compiledModelURL: URL, computeUnits: MLComputeUnits = BSTStrokeModel.defaultComputeUnits) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        let m: MLModel
        do { m = try MLModel(contentsOf: compiledModelURL, configuration: cfg) } catch {
            throw StrokeModelError.modelUnavailable("\(compiledModelURL.lastPathComponent): \(error.localizedDescription)")
        }
        model = m
        self.computeUnits = computeUnits
        let desc = m.modelDescription
        metadata = desc.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        for (name, shape) in Self.shapes {
            let got = desc.inputDescriptionsByName[name]?.multiArrayConstraint?.shape.map(\.intValue)
            guard got == shape else {
                throw StrokeModelError.badModel("input \(name) shape \(got.map { "\($0)" } ?? "missing"), expected \(shape)")
            }
        }
        let out = desc.outputDescriptionsByName["logits"]?.multiArrayConstraint?.shape.map(\.intValue)
        guard out == [1, StrokeClasses.count] else {
            throw StrokeModelError.badModel("output logits shape \(out.map { "\($0)" } ?? "missing"), expected [1, 25]")
        }
        // A model with a different class order would be silently wrong on every shot.
        if let js = metadata["classes_json"], let d = js.data(using: .utf8),
           let names = try? JSONDecoder().decode([String].self, from: d), names != StrokeClasses.names {
            throw StrokeModelError.badModel("class list \(names) differs from StrokeClasses.names")
        }
    }

    /// CPU + Neural Engine: the unit measured with 100% argmax agreement on the 749 match-039
    /// clips (REPORT.md "Parity"); the simulator runs it on the CPU.
    public static let defaultComputeUnits: MLComputeUnits = .cpuAndNeuralEngine

    /// Compiles `modelURL` (.mlpackage) at runtime, or loads it when it is already an `.mlmodelc`.
    /// `cacheDirectory` keeps the compiled model keyed by name, size and date. The compile itself
    /// is CoreML's synchronous `MLModel.compileModel`, which is a second or two for the 71 MB
    /// ensemble, so this is `async` for the caller's sake rather than by necessity.
    public static func load(modelURL: URL, computeUnits: MLComputeUnits = defaultComputeUnits,
                            cacheDirectory: URL? = nil) async throws -> BSTStrokeModel {
        let fm = FileManager.default
        guard fm.fileExists(atPath: modelURL.path) else {
            throw StrokeModelError.modelUnavailable("\(modelURL.path) does not exist")
        }
        if modelURL.pathExtension == "mlmodelc" {
            return try BSTStrokeModel(compiledModelURL: modelURL, computeUnits: computeUnits)
        }
        if let dir = cacheDirectory, let m = loadCached(modelURL: modelURL, cacheDirectory: dir, computeUnits: computeUnits) {
            return m
        }
        let compiled: URL
        do { compiled = try await MLModel.compileModel(at: modelURL) } catch {
            throw StrokeModelError.modelUnavailable("compile \(modelURL.lastPathComponent): \(error.localizedDescription)")
        }
        var url = compiled
        if let dir = cacheDirectory {
            let c = dir.appendingPathComponent(
                "\(modelURL.deletingPathExtension().lastPathComponent)-\(directorySize(modelURL))-\(cacheStamp(modelURL)).mlmodelc")
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
            try? fm.removeItem(at: c)
            if (try? fm.moveItem(at: compiled, to: c)) != nil { url = c }
        }
        return try BSTStrokeModel(compiledModelURL: url, computeUnits: computeUnits)
    }

    /// The bundled model: the app bundle's `.mlmodelc` if it was linked there, else the package
    /// resource (ShuttlStrokeModel; Xcode compiles the `.mlpackage` on first load).
    public static func bundledModelURL(in bundle: Bundle = .main) -> URL? {
        bundle.url(forResource: resourceName, withExtension: "mlmodelc")
            ?? bundle.url(forResource: resourceName, withExtension: "mlpackage")
            ?? StrokeModelResource.modelURL
    }

    /// Loads `modelURL` through `cacheDirectory`, compiling an `.mlpackage` into it first
    /// (keyed by name, size and date). Returns nil when the model is missing, does not compile,
    /// or does not match the contract.
    public static func loadCached(modelURL: URL, cacheDirectory: URL,
                                  computeUnits: MLComputeUnits = defaultComputeUnits) -> BSTStrokeModel? {
        let fm = FileManager.default
        guard fm.fileExists(atPath: modelURL.path) else { return nil }
        if modelURL.pathExtension == "mlmodelc" {
            return try? BSTStrokeModel(compiledModelURL: modelURL, computeUnits: computeUnits)
        }
        let c = cacheDirectory.appendingPathComponent(
            "\(modelURL.deletingPathExtension().lastPathComponent)-\(directorySize(modelURL))-\(cacheStamp(modelURL)).mlmodelc")
        if fm.fileExists(atPath: c.path), let m = try? BSTStrokeModel(compiledModelURL: c, computeUnits: computeUnits) {
            return m
        }
        return nil
    }

    /// Compiles an `.mlpackage` into `cacheDirectory` and loads it from there; nil on any failure.
    public static func compileAndLoad(modelURL: URL, cacheDirectory: URL,
                                      computeUnits: MLComputeUnits = defaultComputeUnits) async -> BSTStrokeModel? {
        if let cached = loadCached(modelURL: modelURL, cacheDirectory: cacheDirectory, computeUnits: computeUnits) {
            return cached
        }
        guard let compiled = try? await MLModel.compileModel(at: modelURL) else { return nil }
        let fm = FileManager.default
        let c = cacheDirectory.appendingPathComponent(
            "\(modelURL.deletingPathExtension().lastPathComponent)-\(directorySize(modelURL))-\(cacheStamp(modelURL)).mlmodelc")
        try? fm.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        try? fm.removeItem(at: c)
        guard (try? fm.moveItem(at: compiled, to: c)) != nil else {
            return try? BSTStrokeModel(compiledModelURL: compiled, computeUnits: computeUnits)
        }
        return try? BSTStrokeModel(compiledModelURL: c, computeUnits: computeUnits)
    }

    /// Cache key component: the .mlpackage's own modification time.
    private static func cacheStamp(_ url: URL) -> Int {
        let manifest = url.appendingPathComponent("Manifest.json")
        let attrs = try? FileManager.default.attributesOfItem(atPath: FileManager.default.fileExists(atPath: manifest.path)
                                                             ? manifest.path : url.path)
        return Int((attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)
    }

    /// The bundled model, compiled if needed; nil when it is missing or does not load.
    public static func loadBundled(in bundle: Bundle = .main, cacheDirectory: URL? = nil) async -> BSTStrokeModel? {
        guard let url = bundledModelURL(in: bundle) else { return nil }
        return try? await load(modelURL: url, cacheDirectory: cacheDirectory)
    }

    public func logits(_ clip: StrokeClip) throws -> [Float] {
        let T = StrokeClip.seqLen
        precondition(clip.joints.count == T * 68 && clip.pos.count == T * 4 && clip.shuttle.count == T * 2,
                     "clip arrays must be padded to 30 frames")
        let provider: MLDictionaryFeatureProvider
        do {
            provider = try MLDictionaryFeatureProvider(dictionary: [
                "joints": MLFeatureValue(multiArray: try Self.array(clip.joints, Self.shapes["joints"]!)),
                "pos": MLFeatureValue(multiArray: try Self.array(clip.pos, Self.shapes["pos"]!)),
                "shuttle": MLFeatureValue(multiArray: try Self.array(clip.shuttle, Self.shapes["shuttle"]!)),
                "valid": MLFeatureValue(multiArray: try Self.array(clip.valid, Self.shapes["valid"]!)),
            ])
        } catch { throw StrokeModelError.prediction("inputs: \(error.localizedDescription)") }
        let out: MLFeatureProvider
        do {
            out = try lock.withLock {
                predictions += 1
                return try model.prediction(from: provider)
            }
        } catch { throw StrokeModelError.prediction(error.localizedDescription) }
        guard let a = out.featureValue(for: "logits")?.multiArrayValue, a.count == StrokeClasses.count else {
            throw StrokeModelError.prediction("missing logits")
        }
        return (0..<a.count).map { a[$0].floatValue }
    }

    static func array(_ v: [Float], _ shape: [Int]) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        var rowMajor = [Int](repeating: 1, count: shape.count)
        for i in stride(from: shape.count - 2, through: 0, by: -1) { rowMajor[i] = rowMajor[i + 1] * shape[i + 1] }
        a.withUnsafeMutableBufferPointer(ofType: Float.self) { p, strides in
            // freshly allocated arrays are contiguous row-major, which is the model's layout
            precondition(strides == rowMajor && v.count == p.count, "unexpected MLMultiArray layout")
            v.withUnsafeBufferPointer { s in p.baseAddress!.update(from: s.baseAddress!, count: v.count) }
        }
        return a
    }

    static func directorySize(_ url: URL) -> Int {
        var total = 0
        if let e = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) {
            for case let f as URL in e { total += (try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0 }
        }
        return total
    }
}

/// The 25 merged ShuttleSet classes in the model's output order (repo get_merged_stroke_types),
/// and their meaning in the app's vocabulary.
public enum StrokeClasses {
    /// Side-free stroke types, index 1...12 of the class list per side.
    public static let types = ["放小球", "擋小球", "殺球", "挑球", "長球", "平球", "切球", "推球",
                               "撲球", "勾球", "發短球", "發長球"]
    public static let names: [String] = ["未知球種"] + types.map { "Top_" + $0 } + types.map { "Bottom_" + $0 }
    public static var count: Int { names.count }

    /// net shot, return net, smash, lob, clear, drive, drop, push, rush, cross-court net shot,
    /// short service, long service.
    static let shotTypes: [ShotType] = [.net, .block, .smash, .lift, .clear, .drive, .drop, .push,
                                        .kill, .crossNet, .shortServe, .longServe]

    /// Class index -> (stroke type, striker side). nil for 0 "未知球種" (the model abstains).
    /// Top is the far player, Bottom the near one.
    public static func decode(_ index: Int) -> (type: ShotType, striker: CourtSide)? {
        guard index >= 1 && index < count else { return nil }
        let k = (index - 1) % 12
        return (shotTypes[k], index <= 12 ? .far : .near)
    }
}
