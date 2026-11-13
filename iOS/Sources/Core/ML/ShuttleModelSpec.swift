import Foundation
import CoreML

// MARK: - ShuttleModelSpec
// Everything the engine needs to know about ShuttleTracker.mlpackage, read at runtime from the
// model's creator-defined metadata (written by backend/ml/export_coreml.py) and its input /
// output descriptions. Nothing about S, C or the canvas size is hard-coded; missing keys fall
// back to the original contract (512×288, stretch, no peak offset, 0–1 RGB, sigmoid output).

struct ShuttleModelSpec: Sendable, Equatable {
    enum Resize: String, Sendable { case letterbox, stretch }
    enum Normalization: String, Sendable { case unit, imagenet }

    var name = "unknown"
    var inputName = "frames"
    var outputName = "heatmap"
    /// Frames per window == heatmaps per prediction.
    var seqLen = 3
    /// Input channels = 3 · (seqLen + background).
    var inChannels = 9
    var usesBackground = false
    var inputIsBGR = false
    var normalization = Normalization.unit
    var inputWidth = 512
    var inputHeight = 288
    var resize = Resize.stretch
    /// Learned (content − peak) offset in net px: full = (peak + offset − pad) / s.
    var centroidOffset = 0.0
    var scoreThreshold: Float = 0.5
    /// false when the export left the sigmoid out (decoder then applies it).
    var outputIsProbability = true
    var newestFirst = false

    init() {}

    /// Metadata keys: model, seq_len, in_channels, uses_background, input_order, normalize,
    /// input_width, input_height, resize, centroid_offset, score_threshold, output_activation,
    /// frame_order.
    init(metadata m: [String: String]) {
        func int(_ k: String) -> Int? { m[k].flatMap { Int($0.trimmingCharacters(in: .whitespaces)) } }
        func dbl(_ k: String) -> Double? { m[k].flatMap { Double($0.trimmingCharacters(in: .whitespaces)) } }
        name = m["model"] ?? name
        usesBackground = ["1", "true", "yes"].contains(m["uses_background"]?.lowercased() ?? "0")
        seqLen = max(1, int("seq_len") ?? seqLen)
        inChannels = int("in_channels") ?? 3 * (seqLen + (usesBackground ? 1 : 0))
        inputIsBGR = m["input_order"]?.lowercased() == "bgr"
        normalization = (m["normalize"]?.lowercased()).map { $0.contains("imagenet") ? .imagenet : .unit } ?? .unit
        inputWidth = max(8, int("input_width") ?? inputWidth)
        inputHeight = max(8, int("input_height") ?? inputHeight)
        switch m["resize"]?.lowercased() {
        case .some(let r) where r.contains("aspect") || r.contains("letterbox") || r.contains("pad"):
            resize = .letterbox
        default:
            resize = .stretch
        }
        centroidOffset = dbl("centroid_offset") ?? 0
        scoreThreshold = Float(dbl("score_threshold") ?? Double(scoreThreshold))
        if let act = m["output_activation"]?.lowercased() {
            outputIsProbability = !(act == "none" || act == "logits" || act == "linear")
        }
        newestFirst = m["frame_order"]?.lowercased().hasPrefix("newest") ?? false
    }

    /// Spec of a loaded model; the actual tensor shapes win over metadata when they disagree.
    init(model: MLModel) {
        let desc = model.modelDescription
        let meta = desc.metadata[.creatorDefinedKey] as? [String: String] ?? [:]
        self.init(metadata: meta)
        let inputs = desc.inputDescriptionsByName.filter { $0.value.type == .multiArray }
        if let input = inputs[inputName] ?? inputs.first?.value {
            inputName = input.name
            let shape = input.multiArrayConstraint?.shape.map(\.intValue) ?? []
            if shape.count == 4 {
                inChannels = shape[1]
                inputHeight = shape[2]
                inputWidth = shape[3]
            }
        }
        let outputs = desc.outputDescriptionsByName.filter { $0.value.type == .multiArray }
        if let output = outputs[outputName] ?? outputs.first?.value {
            outputName = output.name
            let shape = output.multiArrayConstraint?.shape.map(\.intValue) ?? []
            if shape.count == 4, shape[1] > 0 { seqLen = shape[1] }
        }
    }

    /// Channels expected from the frame layout; a mismatch means the metadata is wrong.
    var isConsistent: Bool { inChannels == 3 * (seqLen + (usesBackground ? 1 : 0)) }

    func geometry(width: Int, height: Int) -> FrameGeometry {
        FrameGeometry(width: width, height: height, netWidth: inputWidth, netHeight: inputHeight,
                      mode: resize == .letterbox ? .letterbox : .stretch, peakOffset: centroidOffset)
    }
}
