import Foundation
import ShuttlCore

/// Network description (detector.py ``NetSpec``).
public struct NetSpec: Sendable, Equatable, Hashable {
    public var name: String
    /// Frames in == heatmaps out.
    public var seqLen: Int
    /// Median background prepended as an extra RGB frame.
    public var usesBackground: Bool
    /// ImageNet mean/std input (else x/255). CoreML exports bake this into the graph.
    public var imagenetNorm: Bool
    /// Output needs a sigmoid. CoreML exports bake this into the graph.
    public var logits: Bool
    /// WASB centre-affine convention.
    public var cornerGeometry: Bool
    /// Learned (target − content) offset in net px. Both repos draw targets at int(coord*s)
    /// (floor → −0.5). WASB content is index-affine (warpAffine) → −0.5. TrackNetV3 content is
    /// pixel-centre resized (PIL) while targets are index-scaled at s=0.4 (1280→512) → −0.5·0.4.
    public var peakOffset: Double

    public init(name: String, seqLen: Int, usesBackground: Bool, imagenetNorm: Bool, logits: Bool,
                cornerGeometry: Bool, peakOffset: Double = -0.5) {
        self.name = name; self.seqLen = seqLen; self.usesBackground = usesBackground
        self.imagenetNorm = imagenetNorm; self.logits = logits; self.cornerGeometry = cornerGeometry
        self.peakOffset = peakOffset
    }

    public var inChannels: Int { 3 * (seqLen + (usesBackground ? 1 : 0)) }

    /// detector.py SPECS.
    public static let tracknetv3 = NetSpec(name: "tracknetv3", seqLen: 8, usesBackground: true, imagenetNorm: false,
                                           logits: false, cornerGeometry: false, peakOffset: -0.2)
    public static let wasb = NetSpec(name: "wasb", seqLen: 3, usesBackground: false, imagenetNorm: true,
                                     logits: true, cornerGeometry: true, peakOffset: -0.5)
    public static let specs: [String: NetSpec] = ["tracknetv3": tracknetv3, "wasb": wasb]
}

/// Positional weights for overlapping windows (detector.py ``ensemble_weights``).
public enum EnsembleMode: String, Sendable, Codable {
    /// TrackNetV3 triangular weights.
    case weight
    case average
}

public func ensembleWeights(_ seqLen: Int, _ mode: EnsembleMode) -> [Float] {
    var w = [Float](repeating: 1, count: seqLen)
    if mode == .average { return w }
    for i in 0..<((seqLen + 1) / 2) {
        w[i] = Float(i + 1)
        w[seqLen - i - 1] = Float(i + 1)
    }
    return w
}

/// Median-background settings (estimate_backgrounds keyword defaults).
public struct BackgroundConfig: Sendable, Equatable {
    public var sampleHz = 4.0
    public var maxSamples = 64
    public var maxSegmentSeconds = 120.0
    public var cutFraction = 0.45
    public init(sampleHz: Double = 4.0, maxSamples: Int = 64, maxSegmentSeconds: Double = 120.0, cutFraction: Double = 0.45) {
        self.sampleHz = sampleHz; self.maxSamples = maxSamples
        self.maxSegmentSeconds = maxSegmentSeconds; self.cutFraction = cutFraction
    }
}

/// HeatmapDetector options (constructor keywords of detector.py ``HeatmapDetector``; defaults = Python's).
public struct DetectorConfig: Sendable, Equatable {
    /// Frames between window starts (nil: seq_len, i.e. non-overlapping).
    public var stride: Int? = nil
    /// Windows per network call (bounds memory; results do not depend on it).
    public var batchSize = 8
    public var k = 3
    public var minScore = 0.1
    public var ensemble = EnsembleMode.weight
    /// Resize filter (nil: INTER_AREA for TrackNetV3, INTER_LINEAR for WASB — each repo's training preprocessing).
    public var interpolation: ResizeInterpolation? = nil
    public var refine = PeakRefinement.centroid
    /// Override `NetSpec.peakOffset`.
    public var peakOffset: Double? = nil
    /// Test-time augmentation: average with the prediction on the horizontally mirrored input
    /// (detector.py `tta_flip`; doubles the network cost).
    public var ttaFlip = false
    /// Decode + preprocess on a separate thread (bounded queue of 2·seq_len frames).
    public var prefetch = true
    public var background = BackgroundConfig()

    public init() {}

    /// build_detector's per-model defaults (DETECTOR_DEFAULTS): each repo's own
    /// non-overlapping inference mode.
    public static func defaults(for kind: String) -> DetectorConfig {
        var c = DetectorConfig()
        switch kind {
        case "wasb": c.stride = 3; c.batchSize = 8
        case "tracknetv3": c.stride = 8; c.batchSize = 4
        default: break
        }
        return c
    }
}

public enum DetectorError: Error, CustomStringConvertible {
    case nonConsecutiveFrames(Int)
    /// A background segment's pixels changed during detection (memory damage): the result
    /// would silently differ, so the pass stops; the saved checkpoint still holds the good plan.
    case backgroundChanged([Int])
    case modelUnavailable(String)
    case badModel(String)
    case predictionFailed(String)

    public var description: String {
        switch self {
        case .nonConsecutiveFrames(let i): return "frame source must yield consecutive indices (got \(i))"
        case .backgroundChanged(let segs): return "background plan segments \(segs) changed during detection (memory damage)"
        case .modelUnavailable(let s): return "detector model unavailable: \(s)"
        case .badModel(let s): return "unsupported detector model: \(s)"
        case .predictionFailed(let s): return "detector prediction failed: \(s)"
        }
    }
}
