import CoreGraphics
import CoreVideo
import Foundation
import ShuttlCore
import Vision

/// On-device pose detector: Apple Vision `VNDetectHumanBodyPoseRequest` (system framework,
/// no third-party / AGPL weights) behind `PoseDetector`, replacing players.py's YOLOv8-pose.
///
/// Mapping to what players.py expects from YOLO:
///  • keypoints: Vision's joints in COCO-17 order (neck/root dropped), normalised
///    bottom-left → full-res top-left px; `conf` = joint confidence. A joint Vision does
///    not report gets conf 0 at the joint centroid (YOLO also returns coordinates for
///    invisible keypoints; every consumer gates on conf first);
///  • box: pose observations have no box. `.joints` derives one from the confident joint
///    extremes with head / sole / side margins fitted to YOLO boxes on the TrackNetV2 Test
///    clips (ShuttlPlayersBench); `.humanRectangles` uses the best-overlapping full-body
///    `VNDetectHumanRectanglesRequest` box (falls back to the joint box);
///  • score: observation confidence;
///  • far-half crop passes (players._far_roi / _merge_roi): the pose request runs again with
///    `regionOfInterest` = each crop of `farROIs` — Vision scales the region to its network
///    input, i.e. the upscaled crop of the Python pass — and `mergeROI` merges each crop in
///    turn. One crop = players.py; two crops (`FarCropLayout`) magnify small far players more.
public final class VisionPoseDetector: PoseDetector {
    public enum BoxSource: Sendable { case joints, humanRectangles }

    public struct Options: Sendable {
        /// Far-half crop passes (full-res px), merged in order after the full-frame pass
        /// (see `FarCrops.rects`); empty = full frame only.
        public var farROIs: [PixelRect] = []
        /// Single-crop convenience (players.py `far_roi`): the first crop, or nil.
        public var farROI: PixelRect? {
            get { farROIs.first }
            set { farROIs = newValue.map { [$0] } ?? [] }
        }
        /// Run all crop requests in one `perform` call (Vision may share preprocessing).
        public var batchCrops = false
        public var boxSource: BoxSource = .joints
        /// Run the pose requests at all. Set false to use this detector as a *box provider only*
        /// (person rectangles, including the ROI passes) for a different pose model — see
        /// `RTMPoseDetector`, which takes Vision's boxes and gets keypoints from CoreML RTMPose.
        /// With poses off, the pose requests are skipped and `detections` yields boxes only.
        public var posesEnabled = true
        /// Also run VNDetectHumanRectanglesRequest when boxSource == .joints (benchmarking).
        public var alwaysRequestRectangles = false
        /// A joint counts towards the box / joint count from this Vision confidence.
        public var jointConfidence: Double = 0.1
        /// Observations with fewer confident joints are dropped (no usable box).
        public var minJoints: Int = 4
        /// Joint-box margins as fractions of the joint span: medians of the YOLO-box margins
        /// over 3.8 k matched persons of 12 TrackNetV2 Test clips (ShuttlPlayersBench);
        /// joint box vs YOLO box median IoU 0.85 (Vision's own human rectangles: 0.87).
        public var padTop: Double = 0.082
        public var padBottom: Double = 0.164
        public var padSide: Double = 0.091
        /// ROI pass: a person whose legs are outside the ROI (no confident ankle) is cut by the
        /// crop. YOLO's crop boxes then touch the crop border and `mergeROI` drops them; joint
        /// boxes stop at the last visible joint, so the adapter drops them explicitly.
        public var roiRequireAnkle = true
        /// Vision's ankle joints sit higher on the leg than COCO's (median −2.8 % of the box
        /// height vs YOLOv8-pose, IQR −4.4…−1.3 %, 3.8 k persons). players.py projects ankles to
        /// ANKLE_H = 8 cm, calibrated on COCO ankles, so they are moved down by this fraction of
        /// the person box height (0 = raw Vision joints). Halves the foot disagreement with
        /// YOLO (near median 0.143 → 0.054 m, far 0.153 → 0.076 m).
        public var ankleDrop: Double = 0.028
        public init(farROI: PixelRect? = nil, boxSource: BoxSource = .joints) {
            self.farROIs = farROI.map { [$0] } ?? []; self.boxSource = boxSource
        }
        public init(farROIs: [PixelRect], boxSource: BoxSource = .joints) {
            self.farROIs = farROIs; self.boxSource = boxSource
        }
    }

    /// Raw Vision output of one frame, before conversion.
    public struct Observations {
        public struct Crop {
            public var rect: PixelRect
            public var poses: [VNHumanBodyPoseObservation]
            public var rects: [VNHumanObservation]
        }
        public var imageSize: ImageSize
        public var poses: [VNHumanBodyPoseObservation]
        public var rects: [VNHumanObservation]
        public var crops: [Crop]
    }

    /// Seconds spent inside Vision, per request kind (for benchmarking). `roiPose` sums the
    /// crop passes; `crops[i]` is crop i alone (not split when `batchCrops`).
    public struct Timing: Sendable {
        public var pose = 0.0, roiPose = 0.0, rects = 0.0, frames = 0
        public var crops: [Double] = []
    }

    public var options: Options
    public private(set) var timing = Timing()
    private let poseRequest = VNDetectHumanBodyPoseRequest()
    private var cropPoseRequests: [VNDetectHumanBodyPoseRequest] = []
    private let rectRequest = VisionPoseDetector.makeRectRequest()
    private var cropRectRequests: [VNDetectHumanRectanglesRequest] = []

    public init(options: Options = Options()) { self.options = options }

    static func makeRectRequest() -> VNDetectHumanRectanglesRequest {
        let r = VNDetectHumanRectanglesRequest()
        r.upperBodyOnly = false
        return r
    }

    static let cocoJoints: [VNHumanBodyPoseObservation.JointName] = [
        .nose, .leftEye, .rightEye, .leftEar, .rightEar, .leftShoulder, .rightShoulder,
        .leftElbow, .rightElbow, .leftWrist, .rightWrist, .leftHip, .rightHip,
        .leftKnee, .rightKnee, .leftAnkle, .rightAnkle,
    ]

    public func detect(frames: [CVPixelBuffer], frameIndices: [Int]) throws -> [FrameDetections] {
        try frames.map { try detect(frame: $0) }
    }

    /// Detections of one upright frame, full-res px (full pass merged with the ROI pass).
    public func detect(frame pb: CVPixelBuffer) throws -> FrameDetections {
        detections(try observe(frame: pb))
    }

    /// Run the Vision requests on one frame.
    public func observe(frame pb: CVPixelBuffer) throws -> Observations {
        let W = CVPixelBufferGetWidth(pb), H = CVPixelBufferGetHeight(pb)
        let handler = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up, options: [:])
        let wantRects = options.boxSource == .humanRectangles || options.alwaysRequestRectangles
        var t = CFAbsoluteTimeGetCurrent()
        // Box-provider mode (RTMPoseDetector): no pose request, rectangles only.
        if options.posesEnabled { try handler.perform([poseRequest]) }
        timing.pose += CFAbsoluteTimeGetCurrent() - t
        var obs = Observations(imageSize: ImageSize(width: W, height: H),
                               poses: options.posesEnabled ? (poseRequest.results ?? []) : [],
                               rects: [], crops: [])
        if wantRects {
            t = CFAbsoluteTimeGetCurrent()
            try handler.perform([rectRequest])
            obs.rects = rectRequest.results ?? []
            timing.rects += CFAbsoluteTimeGetCurrent() - t
        }
        let rois = options.farROIs.filter { $0.width > 0 && $0.height > 0 }
        while cropPoseRequests.count < rois.count {
            cropPoseRequests.append(VNDetectHumanBodyPoseRequest())
            cropRectRequests.append(Self.makeRectRequest())
        }
        if timing.crops.count < rois.count { timing.crops += Array(repeating: 0, count: rois.count - timing.crops.count) }
        for (i, roi) in rois.enumerated() {
            // normalised, origin bottom-left; results come back normalised to the region
            cropPoseRequests[i].regionOfInterest = CGRect(
                x: Double(roi.x1) / Double(W), y: 1 - Double(roi.y2) / Double(H),
                width: Double(roi.width) / Double(W), height: Double(roi.height) / Double(H))
            cropRectRequests[i].regionOfInterest = cropPoseRequests[i].regionOfInterest
        }
        if options.posesEnabled, options.batchCrops, !rois.isEmpty {
            t = CFAbsoluteTimeGetCurrent()
            try handler.perform(Array(cropPoseRequests.prefix(rois.count)))
            timing.roiPose += CFAbsoluteTimeGetCurrent() - t
        }
        for (i, roi) in rois.enumerated() {
            if options.posesEnabled, !options.batchCrops {
                t = CFAbsoluteTimeGetCurrent()
                try handler.perform([cropPoseRequests[i]])
                let dt = CFAbsoluteTimeGetCurrent() - t
                timing.roiPose += dt; timing.crops[i] += dt
            }
            var crop = Observations.Crop(rect: roi,
                                         poses: options.posesEnabled ? (cropPoseRequests[i].results ?? []) : [],
                                         rects: [])
            if wantRects {
                t = CFAbsoluteTimeGetCurrent()
                try handler.perform([cropRectRequests[i]])
                crop.rects = cropRectRequests[i].results ?? []
                timing.rects += CFAbsoluteTimeGetCurrent() - t
            }
            obs.crops.append(crop)
        }
        timing.frames += 1
        return obs
    }

    /// Convert observations (optionally with another box source) and merge the crop passes
    /// in order: each crop replaces the detections it overlaps (players._merge_roi rule,
    /// applied once per crop). `crops` selects which crop passes to merge (default: all;
    /// `useROI: false` = full frame only).
    public func detections(_ o: Observations, boxSource: BoxSource? = nil, useROI: Bool = true,
                           ankleDrop: Double? = nil, crops: [Int]? = nil) -> FrameDetections {
        let src = boxSource ?? options.boxSource
        let drop = ankleDrop ?? options.ankleDrop
        var out = convert(poses: o.poses, rects: o.rects, width: Double(o.imageSize.width),
                          height: Double(o.imageSize.height), boxSource: src, ankleDrop: drop)
        guard useROI else { return out }
        for i in crops ?? Array(o.crops.indices) {
            let c = o.crops[i]
            var crop = convert(poses: c.poses, rects: c.rects, width: Double(c.rect.width),
                               height: Double(c.rect.height), boxSource: src, ankleDrop: drop)
            // Box-provider mode yields no keypoints, so the ankle rule cannot apply (there is
            // no PoseDetector contract that fills them in here — see RTMPoseDetector).
            if options.roiRequireAnkle, options.posesEnabled, let first = crop.keypoints.first, first.count > KP.rAnk {
                let keep = (0..<crop.n).filter { j in
                    crop.keypoints[j][KP.lAnk].conf >= options.jointConfidence
                        || crop.keypoints[j][KP.rAnk].conf >= options.jointConfidence
                }
                crop = FrameDetections(boxes: keep.map { crop.boxes[$0] }, scores: keep.map { crop.scores[$0] },
                                       keypoints: keep.map { crop.keypoints[$0] })
            }
            out = mergeROI(full: out, crop: crop, roi: c.rect, imageSize: o.imageSize)
        }
        return out
    }

    /// Vision observations normalised to a (width × height) region → detections in region px.
    func convert(poses: [VNHumanBodyPoseObservation], rects: [VNHumanObservation], width w: Double,
                 height h: Double, boxSource: BoxSource, ankleDrop: Double) -> FrameDetections {
        func px(_ p: CGPoint) -> SIMD2<Double> { SIMD2(Double(p.x) * w, (1 - Double(p.y)) * h) }
        let rectBoxes: [BBox] = rects.map { r in
            let b = r.boundingBox
            return BBox(x1: Double(b.minX) * w, y1: (1 - Double(b.maxY)) * h, x2: Double(b.maxX) * w, y2: (1 - Double(b.minY)) * h)
        }
        // Box-provider mode: no pose observations, so every detection is a person rectangle.
        // The box is clamped to the image the detector saw, as in the pose branch.
        if !options.posesEnabled {
            var bxs: [BBox] = [], scs: [Double] = [], kps: [[Keypoint]] = []
            for (i, r) in rects.enumerated() {
                let o = rectBoxes[i]
                let box = BBox(x1: min(max(o.x1, 0), w), y1: min(max(o.y1, 0), h),
                               x2: min(max(o.x2, 0), w), y2: min(max(o.y2, 0), h))
                bxs.append(box); scs.append(Double(r.confidence)); kps.append([])
            }
            return FrameDetections(boxes: bxs, scores: scs, keypoints: kps)
        }
        var boxes: [BBox] = [], scores: [Double] = [], kps: [[Keypoint]] = []
        for obs in poses {
            guard let pts = try? obs.recognizedPoints(.all) else { continue }
            var kp = [Keypoint](repeating: Keypoint(x: 0, y: 0, conf: 0), count: 17)
            var sum = SIMD2<Double>(0, 0), cnt = 0, confident = 0
            var have = [Bool](repeating: false, count: 17)
            for (i, name) in Self.cocoJoints.enumerated() {
                guard let p = pts[name], p.confidence > 0 else { continue }
                let q = px(p.location)
                kp[i] = Keypoint(x: q.x, y: q.y, conf: Double(p.confidence))
                have[i] = true
                sum += q; cnt += 1
                if Double(p.confidence) >= options.jointConfidence { confident += 1 }
            }
            guard confident >= options.minJoints, cnt > 0 else { continue }
            let c = sum / Double(cnt)
            for i in 0..<17 where !have[i] { kp[i] = Keypoint(x: c.x, y: c.y, conf: 0) }
            guard var box = jointBox(kp) else { continue }
            if ankleDrop != 0 {
                let dy = ankleDrop * box.height
                for a in [KP.lAnk, KP.rAnk] where have[a] { kp[a].y += dy }
            }
            // like YOLO, never report a box beyond the image the detector saw
            box = BBox(x1: min(max(box.x1, 0), w), y1: min(max(box.y1, 0), h), x2: min(max(box.x2, 0), w), y2: min(max(box.y2, 0), h))
            if boxSource == .humanRectangles, !rectBoxes.isEmpty {
                let ious = iou1n(box, rectBoxes)
                let best = ious.indices.max { ious[$0] < ious[$1] }!
                if ious[best] > 0.3 { box = rectBoxes[best] }
            }
            boxes.append(box); scores.append(Double(obs.confidence)); kps.append(kp)
        }
        return FrameDetections(boxes: boxes, scores: scores, keypoints: kps)
    }

    /// YOLO-like person box from the confident joints: side margins beyond the joint
    /// extremes, head top above the highest joint, shoe sole below the lowest.
    public func jointBox(_ kp: [Keypoint]) -> BBox? {
        let good = kp.filter { $0.conf >= options.jointConfidence }
        guard good.count >= 2 else { return nil }
        let xs = good.map(\.x), ys = good.map(\.y)
        let x1 = xs.min()!, x2 = xs.max()!, y1 = ys.min()!, y2 = ys.max()!
        let span = max(y2 - y1, 1)
        return BBox(x1: x1 - options.padSide * span, y1: y1 - options.padTop * span,
                    x2: x2 + options.padSide * span, y2: y2 + options.padBottom * span)
    }
}
