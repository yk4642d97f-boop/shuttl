import CoreML
import CoreVideo
import Foundation
import ShuttlCore
import Vision

/// On-device pose from the CoreML RTMPose-m port, behind the same `PoseDetector` seam as the
/// Apple Vision detector. This is the migration path: the *pose* comes from RTMPose (whose
/// keypoints the stroke classifier was trained on — 0.8730 vs Apple Vision's 0.5305 on 318
/// identical clips), while Apple Vision stays in place as the box provider and as a whole-path
/// fallback (`VisionPoseDetector` is untouched and still the default when no model is bundled).
///
/// Boxes come from Vision's person rectangles (`VNDetectHumanRectanglesRequest`) through the
/// existing `VisionPoseDetector` machinery, so the far-court ROI crop passes work unchanged:
/// `VisionPoseDetector` is run with `posesEnabled = false`, i.e. as a box provider only, and its
/// merged, full-frame `FrameDetections` give the boxes and scores. Each box is then warped to
/// 192x256 and run through the model (REPORT.md section 5), yielding 17 COCO keypoints in
/// full-frame px with per-keypoint confidence.
public final class RTMPoseDetector: PoseDetector {
    public struct Options: Sendable {
        /// Far-half crop passes in full-res px; pass `TrackPlayers.defaultFarROIs(...)`.
        public var farROIs: [PixelRect] = []
        /// Persons whose Vision box is below this score are dropped before the model runs.
        public var minBoxScore: Double = 0.3
        /// Also keep a box even when Vision returns none (uses the full frame). Off by default:
        /// a full-frame RTMPose pass is meaningless (the person would be tiny and centred).
        public var fullFrameFallback = false
        /// Compute unit for the model. `.cpuAndNeuralEngine` is the verified 0-flip unit.
        public var computeUnits: MLComputeUnits = .cpuAndNeuralEngine

        public init(farROIs: [PixelRect] = [], minBoxScore: Double = 0.3,
                    fullFrameFallback: Bool = false,
                    computeUnits: MLComputeUnits = .cpuAndNeuralEngine) {
            self.farROIs = farROIs
            self.minBoxScore = minBoxScore
            self.fullFrameFallback = fullFrameFallback
            self.computeUnits = computeUnits
        }

        public init(farROI: PixelRect?, minBoxScore: Double = 0.3) {
            self.init(farROIs: farROI.map { [$0] } ?? [], minBoxScore: minBoxScore)
        }
    }

    public var options: Options
    public let model: RTMPoseCoreML
    /// Vision in box-only mode: rectangles, plus the far-crop ROI passes.
    private let boxes: VisionPoseDetector
    public private(set) var timing = Timing()

    public struct Timing: Sendable {
        public var boxSeconds = 0.0, models = 0
    }

    /// Loads the bundled model (or `modelURL`) and builds the Vision box provider.
    public init(modelURL: URL? = nil, options: Options = Options(), cacheDirectory: URL? = nil) async throws {
        guard let url = modelURL ?? RTMPoseCoreML.bundledModelURL() else {
            throw RTMPoseError.modelUnavailable("RTMPoseM is not in the bundle")
        }
        self.options = options
        self.model = try await RTMPoseCoreML.load(modelURL: url, computeUnits: options.computeUnits,
                                                  cacheDirectory: cacheDirectory)
        self.boxes = VisionPoseDetector(options: .init(farROIs: options.farROIs, boxSource: .humanRectangles))
        self.boxes.options.posesEnabled = false
    }

    /// Synchronous construction from an already-loaded model (tests, or a caller that loaded
    /// it once and wants several detectors).
    public init(model: RTMPoseCoreML, options: Options = Options()) {
        self.options = options
        self.model = model
        self.boxes = VisionPoseDetector(options: .init(farROIs: options.farROIs, boxSource: .humanRectangles))
        self.boxes.options.posesEnabled = false
    }

    public func detect(frames: [CVPixelBuffer], frameIndices: [Int]) throws -> [FrameDetections] {
        try frames.map { try detect(frame: $0) }
    }

    /// One upright frame: Vision boxes, then one RTMPose pass per box.
    public func detect(frame pb: CVPixelBuffer) throws -> FrameDetections {
        let W = CVPixelBufferGetWidth(pb), H = CVPixelBufferGetHeight(pb)
        let size = ImageSize(width: W, height: H)
        let t = CFAbsoluteTimeGetCurrent()
        let obs = try boxes.observe(frame: pb)
        // Vision's rectangle-only detections: no keypoints, so `jointBox` never runs.
        let d = boxes.detections(obs)
        timing.boxSeconds += CFAbsoluteTimeGetCurrent() - t
        var keep: [Int] = []
        for i in 0..<d.n where d.scores[i] >= options.minBoxScore { keep.append(i) }
        if keep.isEmpty, options.fullFrameFallback {
            return try detect(frame: pb, boxes: [BBox(x1: 0, y1: 0, x2: Double(W - 1), y2: Double(H - 1))],
                              scores: [1.0], imageSize: size)
        }
        return try detect(frame: pb, boxes: keep.map { d.boxes[$0] }, scores: keep.map { d.scores[$0] },
                          imageSize: size)
    }

    /// The seam this whole change exists for: **an image and a person box in, COCO-17
    /// keypoints in frame px out**. Public so a caller with its own person detector can use the
    /// model directly.
    public func detect(frame pb: CVPixelBuffer, boxes personBoxes: [BBox], scores: [Double]? = nil,
                       imageSize: ImageSize) throws -> FrameDetections {
        var outBoxes: [BBox] = [], outScores: [Double] = [], outKP: [[Keypoint]] = []
        try PosePixelView.withBGRA(pb) { px in
            for (i, b) in personBoxes.enumerated() {
                let (center, scale) = RTMPoseGeometry.centerScale(box: b, imageSize: imageSize)
                let matrix = RTMPoseGeometry.warpMatrix(center: center, scale: scale)
                let nchw = RTMPoseGeometry.modelInput(src: px, matrix: matrix)
                let kp = try model.keypoints(fromNCHW: nchw, center: center, scale: scale)
                timing.models += 1
                outBoxes.append(b)
                outScores.append(scores?[safe: i] ?? 1.0)
                outKP.append(kp)
            }
        }
        return FrameDetections(boxes: outBoxes, scores: outScores, keypoints: outKP)
    }
}

private extension Array {
    subscript(safe i: Int) -> Element? { indices.contains(i) ? self[i] : nil }
}
