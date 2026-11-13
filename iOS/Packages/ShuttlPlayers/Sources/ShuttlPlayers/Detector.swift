import CoreVideo
import Foundation
import ShuttlCore

/// Raw output of one pose-detector pass over one frame, in FULL-RESOLUTION upright
/// pixels — what players.py gets from YOLOv8-pose as `(boxes, scores, kps)`.
/// Everything downstream of this struct is a faithful port of players.py.
public struct FrameDetections: Sendable, Equatable {
    public var boxes: [BBox]                 // (N,) xyxy px
    public var scores: [Double]              // (N,)
    public var keypoints: [[Keypoint]]       // (N, 17) COCO order, conf = visibility

    public init(boxes: [BBox], scores: [Double], keypoints: [[Keypoint]]) {
        precondition(boxes.count == scores.count && scores.count == keypoints.count)
        self.boxes = boxes; self.scores = scores; self.keypoints = keypoints
    }

    public static let empty = FrameDetections(boxes: [], scores: [], keypoints: [])

    public var n: Int { scores.count }
}

/// Pose detector behind which YOLO (Python), Apple Vision (on device) or a test double
/// sit: a batch of upright 32BGRA frames → one `FrameDetections` per frame
/// (players.py `Detector = Callable[[frames, frame_idx], list[FrameDets]]`).
public protocol PoseDetector {
    func detect(frames: [CVPixelBuffer], frameIndices: [Int]) throws -> [FrameDetections]
}

/// Detector from a closure (the Python `detector=` hook; tests feed fixture detections).
public struct ClosureDetector: PoseDetector {
    private let body: ([CVPixelBuffer], [Int]) throws -> [FrameDetections]
    public init(_ body: @escaping ([CVPixelBuffer], [Int]) throws -> [FrameDetections]) { self.body = body }
    public func detect(frames: [CVPixelBuffer], frameIndices: [Int]) throws -> [FrameDetections] {
        try body(frames, frameIndices)
    }
}

/// Integer pixel rectangle `[x1, x2) × [y1, y2)` (players.py ROI tuple).
public struct PixelRect: Sendable, Equatable {
    public var x1, y1, x2, y2: Int
    public init(x1: Int, y1: Int, x2: Int, y2: Int) { self.x1 = x1; self.y1 = y1; self.x2 = x2; self.y2 = y2 }
    public var width: Int { x2 - x1 }
    public var height: Int { y2 - y1 }
}

// MARK: - far-half ROI crop (players.py _far_roi / _merge_roi)

public enum FarROI {
    /// Image rect covering the far half (+margin, up to 2.6 m) and the apparent height
    /// (px) of a 1.75 m player on the far baseline; nil when unusable.
    public static func compute(camera: CameraModel, size: ImageSize, courtMargin: Double = PlayersConfig.default.courtMargin)
        -> (rect: PixelRect, baselineHeightPx: Double)? {
        let xs = [-0.3, CourtModel.halfLength + courtMargin]
        let ys = [-CourtModel.doublesHalfWidth - courtMargin, CourtModel.doublesHalfWidth + courtMargin]
        var pts: [SIMD3<Double>] = []
        for x in xs { for y in ys { for z in [0.0, 2.6] { pts.append([x, y, z]) } } }
        let uv = camera.project(pts)
        guard uv.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else { return nil }
        let W = Double(size.width), H = Double(size.height)
        // np.clip(uv.min(0), 0, [W, H]).astype(int)
        let x1 = Int(min(max(uv.map(\.x).min()!, 0), W)), y1 = Int(min(max(uv.map(\.y).min()!, 0), H))
        let x2 = Int(min(max(uv.map(\.x).max()!, 0), W)), y2 = Int(min(max(uv.map(\.y).max()!, 0), H))
        if x2 - x1 < 32 || y2 - y1 < 32 { return nil }
        let base = camera.project([SIMD3(CourtModel.halfLength, 0, 0), SIMD3(CourtModel.halfLength, 0, 1.75)])
        return (PixelRect(x1: x1, y1: y1, x2: x2, y2: y2), abs(base[0].y - base[1].y))
    }

    /// YOLO `imgsz` of the crop pass (2× the crop's long side, ≤ 1280, multiple of 32).
    public static func yoloImgsz(_ r: PixelRect) -> Int {
        let side = max(r.width, r.height)
        return Int(min(1280, 32 * (2 * Double(side) / 32).rounded(.up)))
    }
}

/// Shift crop detections to frame px; they replace overlapping full-frame boxes
/// (higher effective resolution → better keypoints for small far players).
/// A crop detection touching an inner crop border (one that is not also the image
/// border) is a person cut by the crop and is discarded.
public func mergeROI(full: FrameDetections, crop: FrameDetections, roi: PixelRect,
                     imageSize: ImageSize, edgePx: Double = 3.0) -> FrameDetections {
    guard crop.n > 0 else { return full }
    let x1 = Double(roi.x1), y1 = Double(roi.y1), x2 = Double(roi.x2), y2 = Double(roi.y2)
    var cb: [BBox] = [], cs: [Double] = [], ck: [[Keypoint]] = []
    for i in 0..<crop.n {
        let o = crop.boxes[i]
        let b = BBox(x1: o.x1 + x1, y1: o.y1 + y1, x2: o.x2 + x1, y2: o.y2 + y1)
        var cut = false
        if roi.x1 > 0, b.x1 <= x1 + edgePx { cut = true }
        if roi.y1 > 0, b.y1 <= y1 + edgePx { cut = true }
        if roi.x2 < imageSize.width, b.x2 >= x2 - edgePx { cut = true }
        if roi.y2 < imageSize.height, b.y2 >= y2 - edgePx { cut = true }
        if cut { continue }
        cb.append(b); cs.append(crop.scores[i])
        ck.append(crop.keypoints[i].map { Keypoint(x: $0.x + x1, y: $0.y + y1, conf: $0.conf) })
    }
    if cb.isEmpty { return full }
    var keep: [Int] = []
    for i in 0..<full.n where !(iou1n(full.boxes[i], cb).max()! > 0.4) { keep.append(i) }
    return FrameDetections(boxes: keep.map { full.boxes[$0] } + cb,
                           scores: keep.map { full.scores[$0] } + cs,
                           keypoints: keep.map { full.keypoints[$0] } + ck)
}

/// IoU of one box against many (players.py `_iou_1n`).
public func iou1n(_ a: BBox, _ boxes: [BBox]) -> [Double] {
    boxes.map { iou(a, $0) }
}

@inline(__always)
func iou(_ a: BBox, _ b: BBox) -> Double {
    let ix = max(min(a.x2, b.x2) - max(a.x1, b.x1), 0)
    let iy = max(min(a.y2, b.y2) - max(a.y1, b.y1), 0)
    let inter = ix * iy
    let ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / max(ua, 1e-9)
}
