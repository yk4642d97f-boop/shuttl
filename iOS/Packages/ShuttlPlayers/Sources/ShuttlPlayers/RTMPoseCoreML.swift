import CoreML
import Foundation
import ShuttlCore

/// Errors from the on-device RTMPose model.
public enum RTMPoseError: Error, CustomStringConvertible {
    case modelUnavailable(String)
    case badModel(String)
    case prediction(String)

    public var description: String {
        switch self {
        case .modelUnavailable(let s): return "RTMPose model unavailable: \(s)"
        case .badModel(let s): return "RTMPose model shape mismatch: \(s)"
        case .prediction(let s): return "RTMPose prediction failed: \(s)"
        }
    }
}

/// CoreML inference for the shipped `RTMPoseM.mlpackage` port (REPORT.md section 5).
///
/// Input `image` (1,3,256,192) NCHW float32, **RGB raw 0..255** — the ImageNet normalisation
/// is folded into the graph, so the caller must pass raw bytes (`RTMPoseGeometry.modelInput`
/// does exactly that). Outputs `simcc_x` (1,17,384) and `simcc_y` (1,17,512) raw logits.
///
/// The model is configured with `MLComputeUnits.cpuAndNeuralEngine`: the compute unit that
/// gave a literal 0-flip, 100% argmax agreement over 120 real crops (REPORT 4.1 / 7.2). GPU /
/// ALL route a few ops through Metal and introduce 2 adjacent-bin ties worth 0.5 px each.
public final class RTMPoseCoreML: @unchecked Sendable {
    public let model: MLModel
    public let computeUnits: MLComputeUnits
    public let metadata: [String: String]
    let inputName: String
    let simccXName: String
    let simccYName: String
    private let input: MLMultiArray
    private let provider: MLFeatureProvider
    private let options = MLPredictionOptions()
    public private(set) var predictions = 0
    public private(set) var predictionSeconds = 0.0

    /// Loads an already compiled `.mlmodelc`.
    public init(compiledModelURL: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        let m: MLModel
        do { m = try MLModel(contentsOf: compiledModelURL, configuration: cfg) } catch {
            throw RTMPoseError.modelUnavailable("\(compiledModelURL.lastPathComponent): \(error.localizedDescription)")
        }
        model = m
        self.computeUnits = computeUnits
        let desc = m.modelDescription
        metadata = desc.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        let inputs = desc.inputDescriptionsByName.filter { $0.value.type == .multiArray }
        let outputs = desc.outputDescriptionsByName.filter { $0.value.type == .multiArray }
        guard let inDesc = inputs["image"] ?? inputs.first?.value,
              let xDesc = outputs["simcc_x"], let yDesc = outputs["simcc_y"] else {
            throw RTMPoseError.badModel("expected one multi-array input plus simcc_x / simcc_y")
        }
        inputName = inDesc.name
        simccXName = xDesc.name
        simccYName = yDesc.name
        let inShape = inDesc.multiArrayConstraint?.shape.map(\.intValue) ?? []
        let xShape = xDesc.multiArrayConstraint?.shape.map(\.intValue) ?? []
        let yShape = yDesc.multiArrayConstraint?.shape.map(\.intValue) ?? []
        let W = RTMPoseGeometry.modelWidth, H = RTMPoseGeometry.modelHeight
        guard inShape == [1, 3, H, W], xShape == [1, 17, W * 2], yShape == [1, 17, H * 2] else {
            throw RTMPoseError.badModel("shapes \(inShape) -> \(xShape) / \(yShape), "
                + "expected (1,3,\(H),\(W)) -> (1,17,\(W * 2)) / (1,17,\(H * 2))")
        }
        // The metadata name the input contract; a pre-normalised model must never be fed raw.
        if let contract = metadata["input_contract"], !contract.lowercased().contains("raw") {
            throw RTMPoseError.badModel("input contract \"\(contract)\" is not the raw 0..255 model")
        }
        input = try MLMultiArray(shape: inShape.map { NSNumber(value: $0) }, dataType: .float32)
        provider = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: input)])
    }

    /// Compiles `modelURL` (.mlpackage / .mlmodel) at runtime, or loads it when it is already
    /// an `.mlmodelc`. `cacheDirectory` keeps the compiled model keyed by name, size and date.
    public static func load(modelURL: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
                            cacheDirectory: URL? = nil) async throws -> RTMPoseCoreML {
        let fm = FileManager.default
        guard fm.fileExists(atPath: modelURL.path) else {
            throw RTMPoseError.modelUnavailable("\(modelURL.path) does not exist")
        }
        if modelURL.pathExtension == "mlmodelc" {
            return try RTMPoseCoreML(compiledModelURL: modelURL, computeUnits: computeUnits)
        }
        var cached: URL? = nil
        if let dir = cacheDirectory {
            let manifest = modelURL.appendingPathComponent("Manifest.json")
            let attrs = try? fm.attributesOfItem(atPath: fm.fileExists(atPath: manifest.path) ? manifest.path : modelURL.path)
            let stamp = Int((attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)
            let size = directorySize(modelURL)
            let c = dir.appendingPathComponent("\(modelURL.deletingPathExtension().lastPathComponent)-\(size)-\(stamp).mlmodelc")
            cached = c
            if fm.fileExists(atPath: c.path),
               let m = try? RTMPoseCoreML(compiledModelURL: c, computeUnits: computeUnits) {
                return m
            }
        }
        let compiled: URL
        do { compiled = try await MLModel.compileModel(at: modelURL) } catch {
            throw RTMPoseError.modelUnavailable("compile \(modelURL.lastPathComponent): \(error.localizedDescription)")
        }
        var url = compiled
        if let c = cached {
            try? fm.createDirectory(at: c.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? fm.removeItem(at: c)
            if (try? fm.moveItem(at: compiled, to: c)) != nil { url = c }
        }
        return try RTMPoseCoreML(compiledModelURL: url, computeUnits: computeUnits)
    }

    /// The bundled model, preferring the compiled `.mlmodelc` Xcode produced from the
    /// `.mlpackage` source, then the package itself.
    public static func bundledModelURL(in bundle: Bundle = .main) -> URL? {
        bundle.url(forResource: "RTMPoseM", withExtension: "mlmodelc")
            ?? bundle.url(forResource: "RTMPoseM", withExtension: "mlpackage")
            ?? bundle.url(forResource: "RTMPoseM", withExtension: "mlmodel")
    }

    /// Synchronous load for a path that is already compiled (`.mlmodelc`), or a bundle that
    /// holds one. Returns nil when only an uncompiled `.mlpackage` is present — use
    /// `load(modelURL:...)` there, which compiles.
    public static func loadCompiled(url: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine) -> RTMPoseCoreML? {
        guard url.pathExtension == "mlmodelc" else { return nil }
        return try? RTMPoseCoreML(compiledModelURL: url, computeUnits: computeUnits)
    }

    /// One crop's SimCC logits. `nchw` must be 3*192*256 floats of raw 0..255 RGB.
    public func predict(_ nchw: [Float]) throws -> (x: [Float], y: [Float]) {
        let plane = RTMPoseGeometry.modelWidth * RTMPoseGeometry.modelHeight
        precondition(nchw.count == 3 * plane, "input must be 3*192*256 floats")
        input.withUnsafeMutableBufferPointer(ofType: Float.self) { ptr, _ in
            nchw.withUnsafeBufferPointer { src in
                guard let d = ptr.baseAddress, let s = src.baseAddress else { return }
                d.update(from: s, count: 3 * plane)
            }
        }
        let t = CFAbsoluteTimeGetCurrent()
        let out: MLFeatureProvider
        do { out = try model.prediction(from: provider, options: options) } catch {
            throw RTMPoseError.prediction(error.localizedDescription)
        }
        predictionSeconds += CFAbsoluteTimeGetCurrent() - t
        predictions += 1
        guard let xa = out.featureValue(for: simccXName)?.multiArrayValue,
              let ya = out.featureValue(for: simccYName)?.multiArrayValue else {
            throw RTMPoseError.prediction("missing simcc_x / simcc_y in output")
        }
        return (floatArray(xa), floatArray(ya))
    }

    /// Convenience: crop already warped and packed (see `RTMPoseGeometry.modelInput`).
    public func keypoints(fromNCHW nchw: [Float], center: SIMD2<Double>, scale: SIMD2<Double>) throws -> [Keypoint] {
        let (sx, sy) = try predict(nchw)
        return RTMPoseGeometry.decode(simccX: sx, simccY: sy, center: center, scale: scale)
    }

    func floatArray(_ a: MLMultiArray) -> [Float] {
        var out = [Float](repeating: 0, count: a.count)
        if a.dataType == .float32 {
            a.withUnsafeBufferPointer(ofType: Float.self) { p in
                if let b = p.baseAddress { out.withUnsafeMutableBufferPointer { o in o.baseAddress!.update(from: b, count: a.count) } }
            }
        } else {
            for i in 0..<a.count { out[i] = a[i].floatValue }
        }
        return out
    }

    /// Recursive byte size of a model directory (cache key).
    static func directorySize(_ url: URL) -> Int {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: url.path, isDirectory: &isDir) else { return 0 }
        if !isDir.boolValue { return (try? fm.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0 }
        var total = 0
        if let e = fm.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) {
            for case let f as URL in e {
                total += (try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            }
        }
        return total
    }
}
