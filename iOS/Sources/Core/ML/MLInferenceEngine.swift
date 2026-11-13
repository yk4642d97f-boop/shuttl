import Foundation
import CoreML
import CoreMedia
import CoreGraphics
import Accelerate

// MARK: - MLInferenceEngine
// On-device shuttle detector around ShuttleTracker.mlmodelc (WASB HRNet today; any model
// following the export_coreml.py metadata contract).
//
//   camera frame ─► FramePreprocessor (GPU resize/letterbox, vImage → planar float) ─► ring of
//   S canvases (+ approximate-median background) ─► one MLMultiArray (reused) ─► MLModel
//   .prediction (compute units .all → ANE) ─► S heatmaps ─► vDSP peak decoding (top-2, NMS,
//   sub-pixel) ─► FrameGeometry back to full-res px ─► FrameDetections (real PTS).
//
// Budget: 60 fps capture = 16.7 ms/frame. Frames are thinned to `maxInputRate` (30 fps, the
// rate the model was trained on). If a prediction does not fit the frame interval, the engine
// predicts once every `stride` frames (≤ S) and decodes the newest `stride` heatmaps of that
// window, so every accepted frame still gets detections — latency grows by ≤ S−1 frames
// instead of frames being lost. Anything beyond that is dropped upstream (the frame feed keeps
// only the newest buffers). The actor runs on its own serial queue so the synchronous CoreML
// call never blocks Swift's cooperative pool. Without a model every call is a cheap no-op.

actor MLInferenceEngine {
    struct Options: Sendable {
        var computeUnits: MLComputeUnits = .all
        /// Accepted frames per second (the network's temporal window was trained at 30 fps).
        var maxInputRate: Double = 30
        /// Peaks per frame handed to the tracker.
        var candidatesPerFrame = 2
        /// Peak floor for candidates (the tracker gates weak ones; below the model's 0.5 on purpose).
        var minScore: Float = 0.3
        /// Half-max centroid, as the backend. On held-out real footage (TrackNetV2 Test 1_05_02,
        /// 180 frames) it matches the 3×3 log-parabola in median (1.35 vs 1.34 px) and beats it
        /// in the tail (p90 2.49 vs 2.93 px) because WASB's sigmoid blobs are flat-topped.
        var refinement: HeatmapDecoder.Refinement = .centroid
        /// Fixed prediction stride (tests); nil = adaptive.
        var fixedStride: Int?
        /// A longer gap between accepted frames restarts the temporal window.
        var maxFrameGap: Double = 0.5
    }

    struct Stats: Sendable {
        var accepted = 0
        var skipped = 0
        var predictions = 0
        var stride = 1
        /// Exponential means, seconds.
        var preprocessSeconds = 0.0
        var predictionSeconds = 0.0
    }

    static let shared = MLInferenceEngine()

    private let queue = DispatchSerialQueue(label: "com.shuttl.inference", qos: .userInitiated)
    nonisolated var unownedExecutor: UnownedSerialExecutor { queue.asUnownedSerialExecutor() }

    private let explicitModelURL: URL?
    let options: Options

    private enum LoadState { case notLoaded, ready, unavailable }
    private var loadState = LoadState.notLoaded
    private var model: MLModel?
    private(set) var spec: ShuttleModelSpec?
    private var preprocessor: FramePreprocessor?
    private var input: MLMultiArray?
    private var provider: MLFeatureProvider?
    private let predictionOptions = MLPredictionOptions()
    private var decoder = HeatmapDecoder()
    private var heatmapScratch: [Float] = []

    // Ring of preprocessed canvases, oldest → newest by `ringSeq`.
    private struct Slot {
        var timestamp: CMTime
        var frameIndex: Int
        var geometry: FrameGeometry
    }
    private var ringStorage: FloatStorage?
    private var slots: [Slot?] = []
    private var ringCount = 0          // frames written since the window (re)started
    private var sincePrediction = 0    // frames written since the last prediction
    private var background = ApproximateMedianBackground()
    private var lastAccepted: Double?
    private var frameCounter = 0
    private var meanInterval = 1.0 / 30
    private(set) var stats = Stats()

    init(modelURL: URL? = nil, options: Options = Options()) {
        explicitModelURL = modelURL
        self.options = options
    }

    // MARK: Model

    /// Loads the model once (first call pays the ~1 s CoreML / ANE compile). Returns availability.
    @discardableResult
    func load() -> Bool {
        switch loadState {
        case .ready: return true
        case .unavailable: return false
        case .notLoaded: break
        }
        loadState = .unavailable
        guard let url = explicitModelURL ?? Self.bundledModelURL() else {
            print("[MLInferenceEngine] ShuttleTracker model not in bundle — live detection disabled.")
            return false
        }
        do {
            let compiled = url.pathExtension == "mlmodelc" ? url : try MLModel.compileModel(at: url)
            let config = MLModelConfiguration()
            config.computeUnits = options.computeUnits
            let m = try MLModel(contentsOf: compiled, configuration: config)
            let s = ShuttleModelSpec(model: m)
            guard s.isConsistent, s.inputWidth > 0, s.inputHeight > 0,
                  let pre = FramePreprocessor(netWidth: s.inputWidth, netHeight: s.inputHeight) else {
                print("[MLInferenceEngine] Model metadata inconsistent (\(s)) — live detection disabled.")
                return false
            }
            let arr = try MLMultiArray(shape: [1, NSNumber(value: s.inChannels), NSNumber(value: s.inputHeight),
                                               NSNumber(value: s.inputWidth)], dataType: .float32)
            provider = try MLDictionaryFeatureProvider(dictionary: [s.inputName: MLFeatureValue(multiArray: arr)])
            model = m
            spec = s
            preprocessor = pre
            input = arr
            let plane = s.inputWidth * s.inputHeight
            ringStorage = FloatStorage(count: s.seqLen * 3 * plane)
            slots = Array(repeating: nil, count: s.seqLen)
            heatmapScratch = [Float](repeating: 0, count: plane)
            decoder = HeatmapDecoder(minScore: options.minScore, maxPeaks: options.candidatesPerFrame,
                                     nmsRadius: 6, refinement: options.refinement, centroidRadius: 4)
            loadState = .ready
            return true
        } catch {
            print("[MLInferenceEngine] Failed to load model — \(error)")
            return false
        }
    }

    var isAvailable: Bool { loadState == .ready }

    static func bundledModelURL(in bundle: Bundle = .main) -> URL? {
        bundle.url(forResource: "ShuttleTracker", withExtension: "mlmodelc")
            ?? bundle.url(forResource: "ShuttleTracker", withExtension: "mlpackage")
    }

    // MARK: Frames

    /// Restarts the temporal window (seek, new rally, camera switch).
    func reset() {
        ringCount = 0
        sincePrediction = 0
        lastAccepted = nil
        background.reset()
        for i in slots.indices { slots[i] = nil }
    }

    /// Feeds one camera frame. Returns detections for 0…stride frames (usually the one just
    /// fed); empty while the window fills, for thinned frames, or when no model is present.
    func process(_ frame: CameraFrame) -> [FrameDetections] {
        guard load(), let spec, let pre = preprocessor, let ring = ringStorage?.pointer else { return [] }
        let t = frame.seconds
        if let t, let last = lastAccepted {
            let dt = t - last
            if dt < 0 || dt > options.maxFrameGap {
                reset()
            } else if dt < 0.75 / options.maxInputRate {
                stats.skipped += 1
                return []
            } else {
                meanInterval += 0.1 * (dt - meanInterval)
            }
        }
        lastAccepted = t ?? lastAccepted

        // 1. Preprocess into the next ring slot.
        let (w, h) = frame.uprightSize
        let geometry = spec.geometry(width: w, height: h)
        if let prev = slots.compactMap({ $0 }).last, prev.geometry != geometry { reset() }
        let plane = spec.inputWidth * spec.inputHeight
        let slotIndex = ringCount % spec.seqLen
        let start = CFAbsoluteTimeGetCurrent()
        guard pre.render(frame, geometry: geometry, into: ring + slotIndex * 3 * plane,
                         bgr: spec.inputIsBGR, imagenet: spec.normalization == .imagenet) else { return [] }
        if spec.usesBackground { background.update(with: ring + slotIndex * 3 * plane, count: 3 * plane) }
        slots[slotIndex] = Slot(timestamp: frame.timestamp, frameIndex: frameCounter, geometry: geometry)
        frameCounter += 1
        ringCount += 1
        sincePrediction += 1
        stats.accepted += 1
        let prep = CFAbsoluteTimeGetCurrent() - start
        stats.preprocessSeconds += 0.1 * (prep - stats.preprocessSeconds)

        // 2. Predict when the window is full and the stride is due.
        guard ringCount >= spec.seqLen, sincePrediction >= currentStride(spec) else { return [] }
        return predictWindow(spec: spec, plane: plane, ring: ring)
    }

    private func currentStride(_ spec: ShuttleModelSpec) -> Int {
        if let s = options.fixedStride { return min(max(s, 1), spec.seqLen) }
        let budget = max(meanInterval - stats.preprocessSeconds, 1e-3)
        let need = Int((stats.predictionSeconds / budget).rounded(.up))
        return min(max(need, 1), spec.seqLen)
    }

    /// Ring slot holding window position `p` (0 = oldest of the last `seqLen` frames).
    private func slot(ofPosition p: Int, seqLen s: Int) -> Int {
        (ringCount - s + p) % s
    }

    private func predictWindow(spec: ShuttleModelSpec, plane: Int, ring: UnsafeMutablePointer<Float>) -> [FrameDetections] {
        guard let model, let input, let provider else { return [] }
        let s = spec.seqLen
        // Window positions 0..<s, oldest first; ring slot of position p.
        let start = CFAbsoluteTimeGetCurrent()
        input.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
            guard let dst = buf.baseAddress else { return }
            var c = 0
            if spec.usesBackground {
                background.values.withUnsafeBufferPointer { bg in
                    if let b = bg.baseAddress { dst.update(from: b, count: 3 * plane) }
                }
                c = 3
            }
            for p in 0..<s {
                let src = ring + slot(ofPosition: spec.newestFirst ? s - 1 - p : p, seqLen: s) * 3 * plane
                (dst + (c + 3 * p) * plane).update(from: src, count: 3 * plane)
            }
        }
        let output: MLFeatureProvider
        do {
            output = try model.prediction(from: provider, options: predictionOptions)
        } catch {
            print("[MLInferenceEngine] Prediction failed — \(error)")
            return []
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        stats.predictionSeconds = stats.predictions == 0 ? elapsed : stats.predictionSeconds + 0.1 * (elapsed - stats.predictionSeconds)
        stats.predictions += 1
        guard let heat = output.featureValue(for: spec.outputName)?.multiArrayValue else { return [] }

        // Decode the newest `n` frames (those not covered by the previous prediction).
        let n = min(sincePrediction, s)
        sincePrediction = 0
        stats.stride = currentStride(spec)
        var results: [FrameDetections] = []
        for p in (s - n)..<s {
            guard let entry = slots[slot(ofPosition: p, seqLen: s)] else { continue }
            let channel = spec.newestFirst ? s - 1 - p : p
            guard copyHeatmap(heat, channel: channel, spec: spec) else { continue }
            let peaks = heatmapScratch.withUnsafeMutableBufferPointer { buf -> [HeatmapPeak] in
                if !spec.outputIsProbability { HeatmapDecoder.applySigmoid(buf) }
                return decoder.decode(buf, width: spec.inputWidth, height: spec.inputHeight)
            }
            let g = entry.geometry
            let dets = peaks.enumerated().compactMap { rank, pk -> ShuttleDetection? in
                let px = g.peakToFull(x: pk.x, y: pk.y)
                guard px.x >= -0.5, px.y >= -0.5, px.x <= Double(g.width) - 0.5, px.y <= Double(g.height) - 0.5 else {
                    return nil   // peak in the letterbox
                }
                return ShuttleDetection(pixel: px, position: g.normalized(px), confidence: pk.score,
                                        frameIndex: entry.frameIndex, timestamp: entry.timestamp, rank: rank)
            }
            results.append(FrameDetections(detections: dets, timestamp: entry.timestamp, frameIndex: entry.frameIndex,
                                           imageSize: g.fullSize, latency: elapsed + stats.preprocessSeconds))
        }
        return results
    }

    /// Copies heatmap `channel` of a (1, S, H, W) or (S, H, W) output into `heatmapScratch`
    /// (float32, contiguous), honouring strides and float16 outputs.
    private func copyHeatmap(_ heat: MLMultiArray, channel: Int, spec: ShuttleModelSpec) -> Bool {
        let shape = heat.shape.map(\.intValue), strides = heat.strides.map(\.intValue)
        guard shape.count >= 3 else { return false }
        let hDim = shape.count - 2, wDim = shape.count - 1, cDim = shape.count - 3
        let h = shape[hDim], w = shape[wDim]
        guard h == spec.inputHeight, w == spec.inputWidth, channel < shape[cDim] else { return false }
        let rowStride = strides[hDim], colStride = strides[wDim]
        let base = channel * strides[cDim]
        switch heat.dataType {
        case .float32:
            return heat.withUnsafeBufferPointer(ofType: Float.self) { src in
                guard let sp = src.baseAddress else { return false }
                heatmapScratch.withUnsafeMutableBufferPointer { dst in
                    guard let dp = dst.baseAddress else { return }
                    if colStride == 1 && rowStride == w {
                        dp.update(from: sp + base, count: w * h)
                    } else {
                        var zero: Float = 0
                        for y in 0..<h {
                            vDSP_vsadd(sp + base + y * rowStride, vDSP_Stride(colStride), &zero,
                                       dp + y * w, 1, vDSP_Length(w))
                        }
                    }
                }
                return true
            }
        case .float16:
            guard colStride == 1 else { return false }
            return heat.withUnsafeBytes { raw in
                guard let sp = raw.baseAddress else { return false }
                return heatmapScratch.withUnsafeMutableBufferPointer { dst in
                    guard let dp = dst.baseAddress else { return false }
                    var s = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: sp + 2 * base), height: vImagePixelCount(h),
                                          width: vImagePixelCount(w), rowBytes: rowStride * 2)
                    var d = vImage_Buffer(data: dp, height: vImagePixelCount(h), width: vImagePixelCount(w), rowBytes: w * 4)
                    return vImageConvert_Planar16FtoPlanarF(&s, &d, vImage_Flags(kvImageNoFlags)) == kvImageNoError
                }
            }
        default:
            return false
        }
    }
}
