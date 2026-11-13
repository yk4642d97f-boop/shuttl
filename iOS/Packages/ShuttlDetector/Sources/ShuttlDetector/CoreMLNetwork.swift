import Accelerate
import CoreML
import Foundation
import ShuttlCore

/// A heatmap net exported by backend/ml/export_coreml.py (contract: input "frames" float32
/// (1, C, 288, 512) plain 0-1 RGB, background first then oldest → newest; output "heatmap"
/// (1, S, 288, 512) probabilities — ImageNet normalisation and sigmoid are inside the graph).
/// Replaces the torch module of detector.py.
public final class CoreMLHeatmapNetwork: HeatmapNetwork, @unchecked Sendable {
    public let spec: NetSpec
    public let model: MLModel
    public let metadata: [String: String]
    public let computeUnits: MLComputeUnits
    let inputName: String
    let outputName: String
    private let input: MLMultiArray
    private let provider: MLFeatureProvider
    private let options = MLPredictionOptions()
    private var outputBacking: MLMultiArray?
    public private(set) var predictions = 0
    public private(set) var predictionSeconds = 0.0

    /// Loads an already compiled `.mlmodelc`. `spec` defaults to the detector.py preset named by
    /// the model's `model` metadata (checked against the tensor shapes).
    public init(compiledModelURL: URL, computeUnits: MLComputeUnits = .all, spec requested: NetSpec? = nil) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        let m: MLModel
        do { m = try MLModel(contentsOf: compiledModelURL, configuration: cfg) } catch {
            throw DetectorError.modelUnavailable("\(compiledModelURL.lastPathComponent): \(error.localizedDescription)")
        }
        model = m
        self.computeUnits = computeUnits
        let desc = m.modelDescription
        metadata = desc.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        let inputs = desc.inputDescriptionsByName.filter { $0.value.type == .multiArray }
        let outputs = desc.outputDescriptionsByName.filter { $0.value.type == .multiArray }
        guard let inDesc = inputs["frames"] ?? inputs.first?.value, let outDesc = outputs["heatmap"] ?? outputs.first?.value else {
            throw DetectorError.badModel("expected one multi-array input and output")
        }
        inputName = inDesc.name
        outputName = outDesc.name
        let inShape = inDesc.multiArrayConstraint?.shape.map(\.intValue) ?? []
        let outShape = outDesc.multiArrayConstraint?.shape.map(\.intValue) ?? []
        guard inShape.count == 4, outShape.count == 4, inShape[2] == detectorNetHeight, inShape[3] == detectorNetWidth,
              outShape[2] == detectorNetHeight, outShape[3] == detectorNetWidth else {
            throw DetectorError.badModel("shapes \(inShape) → \(outShape), expected (1,C,288,512) → (1,S,288,512)")
        }
        var s: NetSpec
        if let requested { s = requested } else if let name = metadata["model"], let preset = NetSpec.specs[name] { s = preset } else {
            switch (inShape[1], outShape[1]) {
            case (27, 8): s = .tracknetv3
            case (9, 3): s = .wasb
            default: throw DetectorError.badModel("unknown net (metadata model=\(metadata["model"] ?? "nil"))")
            }
        }
        if s.inChannels != inShape[1] || s.seqLen != outShape[1] {
            // Same family, other window length (spec_from_state_dict reads it from the weights).
            let bg = inShape[1] == 3 * (outShape[1] + 1)
            guard bg || inShape[1] == 3 * outShape[1] else {
                throw DetectorError.badModel("channels \(inShape[1]) do not fit \(outShape[1]) heatmaps")
            }
            s.seqLen = outShape[1]; s.usesBackground = bg
        }
        if let norm = metadata["normalize"], norm != "0-1" { throw DetectorError.badModel("input normalisation \(norm)") }
        if let act = metadata["output_activation"], act != "sigmoid" { throw DetectorError.badModel("output activation \(act)") }
        self.spec = s
        input = try MLMultiArray(shape: inShape.map { NSNumber(value: $0) }, dataType: .float32)
        provider = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(multiArray: input)])
        outputBacking = try? MLMultiArray(shape: outShape.map { NSNumber(value: $0) }, dataType: .float32)
    }

    /// Compiles `modelURL` (.mlpackage / .mlmodel) at runtime with `MLModel.compileModel(at:)`,
    /// or loads it directly when it is already an .mlmodelc. With `cacheDirectory`, the compiled
    /// model is kept there (keyed by name, size and modification date) and reused next time.
    public static func load(modelURL: URL, computeUnits: MLComputeUnits = .all, spec: NetSpec? = nil,
                            cacheDirectory: URL? = nil) async throws -> CoreMLHeatmapNetwork {
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw DetectorError.modelUnavailable("\(modelURL.path) does not exist")
        }
        if modelURL.pathExtension == "mlmodelc" {
            return try CoreMLHeatmapNetwork(compiledModelURL: modelURL, computeUnits: computeUnits, spec: spec)
        }
        let fm = FileManager.default
        var cached: URL? = nil
        if let dir = cacheDirectory {
            let manifest = modelURL.appendingPathComponent("Manifest.json")
            let attrs = try? fm.attributesOfItem(atPath: fm.fileExists(atPath: manifest.path) ? manifest.path : modelURL.path)
            let stamp = Int((attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)
            let size = directorySize(modelURL)
            cached = dir.appendingPathComponent("\(modelURL.deletingPathExtension().lastPathComponent)-\(size)-\(stamp).mlmodelc")
            if let c = cached, fm.fileExists(atPath: c.path),
               let net = try? CoreMLHeatmapNetwork(compiledModelURL: c, computeUnits: computeUnits, spec: spec) {
                return net
            }
        }
        let compiled: URL
        do { compiled = try await MLModel.compileModel(at: modelURL) } catch {
            throw DetectorError.modelUnavailable("compile \(modelURL.lastPathComponent): \(error.localizedDescription)")
        }
        var url = compiled
        if let c = cached {
            try? fm.createDirectory(at: c.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? fm.removeItem(at: c)
            if (try? fm.moveItem(at: compiled, to: c)) != nil { url = c }
        }
        return try CoreMLHeatmapNetwork(compiledModelURL: url, computeUnits: computeUnits, spec: spec)
    }

    public func predict(_ windows: [NetWindow], into output: UnsafeMutablePointer<Float>) throws {
        let plane = detectorNetWidth * detectorNetHeight
        let S = spec.seqLen
        for (b, w) in windows.enumerated() {
            precondition(w.frames.count == S && (w.background != nil) == spec.usesBackground, "window does not match the net")
            input.withUnsafeMutableBufferPointer(ofType: Float.self) { ptr, _ in w.packPlanar(into: ptr.baseAddress!) }
            let t = CFAbsoluteTimeGetCurrent()
            let heat = try runModel()
            predictionSeconds += CFAbsoluteTimeGetCurrent() - t
            predictions += 1
            try copyHeatmaps(heat, into: output + b * S * plane)
        }
    }

    private func runModel() throws -> MLMultiArray {
        if let backing = outputBacking {
            options.outputBackings = [outputName: backing]
            do {
                let out = try model.prediction(from: provider, options: options)
                if let arr = out.featureValue(for: outputName)?.multiArrayValue { return arr }
            } catch {
                outputBacking = nil            // backing refused by this model/compute unit: allocate per call
            }
            options.outputBackings = [:]
        }
        let out: MLFeatureProvider
        do { out = try model.prediction(from: provider, options: options) } catch {
            throw DetectorError.predictionFailed(error.localizedDescription)
        }
        guard let arr = out.featureValue(for: outputName)?.multiArrayValue else {
            throw DetectorError.predictionFailed("no \(outputName) output")
        }
        return arr
    }

    /// (1, S, H, W) → S contiguous planes, honouring strides and float16 outputs.
    private func copyHeatmaps(_ heat: MLMultiArray, into dst: UnsafeMutablePointer<Float>) throws {
        let shape = heat.shape.map(\.intValue), st = heat.strides.map(\.intValue)
        guard shape.count == 4 else { throw DetectorError.predictionFailed("heatmap rank \(shape.count)") }
        let S = shape[1], H = shape[2], W = shape[3]
        switch heat.dataType {
        case .float32:
            heat.withUnsafeBufferPointer(ofType: Float.self) { src in
                let sp = src.baseAddress!
                for s in 0..<S {
                    let o = dst + s * H * W
                    if st[3] == 1 && st[2] == W {
                        o.update(from: sp + s * st[1], count: H * W)
                    } else {
                        for y in 0..<H {
                            let row = sp + s * st[1] + y * st[2]
                            for x in 0..<W { o[y * W + x] = row[x * st[3]] }
                        }
                    }
                }
            }
        case .float16:
            guard st[3] == 1 else { throw DetectorError.predictionFailed("strided float16 heatmap") }
            try heat.withUnsafeBytes { raw in
                for s in 0..<S {
                    var sb = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: raw.baseAddress! + 2 * s * st[1]),
                                           height: vImagePixelCount(H), width: vImagePixelCount(W), rowBytes: 2 * st[2])
                    var db = vImage_Buffer(data: dst + s * H * W, height: vImagePixelCount(H), width: vImagePixelCount(W),
                                           rowBytes: 4 * W)
                    if vImageConvert_Planar16FtoPlanarF(&sb, &db, vImage_Flags(kvImageNoFlags)) != kvImageNoError {
                        throw DetectorError.predictionFailed("float16 conversion")
                    }
                }
            }
        default:
            throw DetectorError.predictionFailed("heatmap data type \(heat.dataType.rawValue)")
        }
    }
}

private func directorySize(_ url: URL) -> Int {
    guard let en = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
    var total = 0
    for case let f as URL in en { total += (try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0 }
    return total
}

// MARK: - build_detector

/// Which net to use (build_detector ``kind``). `auto` = first of `autoOrder` whose model exists.
public enum DetectorKind: String, Sendable, CaseIterable {
    case auto, tracknetv3, wasb
}

public extension HeatmapDetector {
    /// detector.py AUTO_ORDER (server default first).
    static let autoOrder: [DetectorKind] = [.tracknetv3, .wasb]

    /// Model file names searched per kind inside a models directory (fine-tuned first, as WEIGHT_FILES).
    static let modelFiles: [DetectorKind: [String]] = [
        .tracknetv3: ["TrackNetV3_finetuned", "TrackNetV3"],
        .wasb: ["WASB_finetuned", "WASB", "ShuttleTracker"],
    ]

    /// First existing model for `kind` in `directory` (.mlmodelc preferred over .mlpackage).
    static func findModel(_ kind: DetectorKind, in directory: URL) -> URL? {
        for name in modelFiles[kind] ?? [] {
            for ext in ["mlmodelc", "mlpackage", "mlmodel"] {
                let u = directory.appendingPathComponent("\(name).\(ext)")
                if FileManager.default.fileExists(atPath: u.path) { return u }
            }
        }
        return nil
    }

    /// Detector for one model URL with build_detector's per-model defaults (DETECTOR_DEFAULTS)
    /// unless `config` is given.
    static func load(modelURL: URL, config: DetectorConfig? = nil, computeUnits: MLComputeUnits = .all,
                     cacheDirectory: URL? = nil) async throws -> HeatmapDetector {
        let net = try await CoreMLHeatmapNetwork.load(modelURL: modelURL, computeUnits: computeUnits, cacheDirectory: cacheDirectory)
        return HeatmapDetector(network: net, config: config ?? .defaults(for: net.spec.name))
    }

    /// build_detector: `kind` from a directory of exported models (auto: AUTO_ORDER).
    static func build(kind: DetectorKind = .auto, modelsDirectory: URL, config: DetectorConfig? = nil,
                      computeUnits: MLComputeUnits = .all, cacheDirectory: URL? = nil) async throws -> HeatmapDetector {
        let order = kind == .auto ? autoOrder : [kind]
        var lastError: Error? = nil
        for k in order {
            guard let url = findModel(k, in: modelsDirectory) else { continue }
            do { return try await load(modelURL: url, config: config, computeUnits: computeUnits, cacheDirectory: cacheDirectory) } catch {
                if kind != .auto { throw error }
                lastError = error
            }
        }
        throw lastError ?? DetectorError.modelUnavailable("no \(kind.rawValue) model in \(modelsDirectory.path)")
    }
}
