import CoreVideo
import Foundation
import ShuttlCore
import ShuttlDetector
import ShuttlLinker
import ShuttlPlayers
import Vision

/// Everything extracted for one rally span (local frame k = video frame start + k).
struct RallyOut: Codable {
    var spec: RallySpec
    var error: String?
    var t: [Double] = []
    var candidates: [Float] = []          // (n, K, 3) x, y, score; NaN padded
    var shuttle: [Float] = []             // (n, 2) linked track px, NaN where none
    var visible: [Bool] = []
    var interpolated: [Bool] = []
    var shuttleConf: [Float] = []
    var personStart: [Int32] = []         // (n + 1) offsets into person arrays
    var personKp: [Float] = []            // (P, 19, 3)
    var personBox: [Float] = []           // (P, 4)
    var personScore: [Float] = []
    var personSource: [Int8] = []         // 0 full frame, i = far crop i
    var playerPerson: [Int32] = []        // (n, 2) [far, near] → person index in this rally, −1
    var playerTrack: [Int32] = []         // (n, 2) tracker id, −1
    var playerBox: [Float] = []           // (n, 2, 4)
    var playerKp: [Float] = []            // (n, 2, 17, 3) tracker keypoints (ankles dropped)
    var playerFoot: [Float] = []          // (n, 2, 2) metres on the floor
    var court: CourtResult?
    var cameraSource: Int8 = 0            // 0 none, 1 this rally's calibration, 2 the match calibration
    var cameraDiffPx: Double = .nan       // median keypoint px, rally vs match camera
    var timing: [String: Double] = [:]

    var n: Int { t.count }
}

/// Frames `start..<stop` of a video re-indexed from 0 (the detector wants consecutive indices from 0).
struct SpanSource: DetectorFrameSource {
    let base: VideoFrameSource
    init(_ v: VideoSource, _ start: Int, _ stop: Int) { base = VideoFrameSource(v, start: start, stop: stop) }

    func forEachFrame(step: Int, _ body: (Int, PixelView) throws -> Bool) throws {
        try base.forEachFrame(step: step) { i, view in try body(i - base.start, view) }
    }
}

enum Joints {
    static let vision: [VNHumanBodyPoseObservation.JointName] = [
        .nose, .leftEye, .rightEye, .leftEar, .rightEar, .leftShoulder, .rightShoulder,
        .leftElbow, .rightElbow, .leftWrist, .rightWrist, .leftHip, .rightHip,
        .leftKnee, .rightKnee, .leftAnkle, .rightAnkle, .neck, .root,
    ]
    static let names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
                        "left_knee", "right_knee", "left_ankle", "right_ankle", "neck", "root"]
}

final class RallyExtractor {
    let video: VideoSource
    let detector: HeatmapDetector
    let matchCamera: CameraModel?
    let poseStep: Int
    let maxCandidates = 3

    init(video: VideoSource, detector: HeatmapDetector, matchCamera: CameraModel?, poseStep: Int) {
        self.video = video; self.detector = detector; self.matchCamera = matchCamera; self.poseStep = poseStep
    }

    func run(_ spec: RallySpec) throws -> RallyOut {
        var out = RallyOut(spec: spec)
        let m = video.meta, n = spec.stop - spec.start
        let t0 = m.timestamps[spec.start]
        let ts = (spec.start..<spec.stop).map { m.timestamps[$0] }
        let local = VideoMeta(fps: m.fps, width: m.width, height: m.height, nFrames: n, timestamps: ts.map { $0 - t0 })
        out.t = ts

        var clock = CFAbsoluteTimeGetCurrent()
        func lap(_ k: String) { let now = CFAbsoluteTimeGetCurrent(); out.timing[k] = now - clock; clock = now }

        let calIdx = stride(from: 0, to: 12, by: 1).map { spec.start + $0 * (n - 1) / 11 }
        let court = Court.calibrate(try Court.frames(video, Array(Set(calIdx)).sorted()), indices: calIdx, size: m.size)
        out.court = court
        var camera: CameraModel? = nil
        if let c = court.camera { camera = c; out.cameraSource = 1 } else if let c = matchCamera { camera = c; out.cameraSource = 2 }
        if let a = court.camera, let b = matchCamera { out.cameraDiffPx = Court.difference(a, b).median }
        lap("calibrate")

        let cands = try detector.detectFrames(SpanSource(video, spec.start, spec.stop), width: m.width, height: m.height,
                                              fps: m.fps, nFrames: n)
        lap("detect")
        let K = maxCandidates
        out.candidates = [Float](repeating: .nan, count: n * K * 3)
        for (k, cs) in cands.enumerated() {
            for (j, c) in cs.prefix(K).enumerated() {
                let o = (k * K + j) * 3
                out.candidates[o] = Float(c.x); out.candidates[o + 1] = Float(c.y); out.candidates[o + 2] = Float(c.score)
            }
        }
        let track = try linkCandidates(cands, meta: local)
        out.shuttle = track.xy.flatMap { [Float($0.x), Float($0.y)] }
        out.visible = track.visible
        out.interpolated = track.interpolated
        out.shuttleConf = track.conf.map(Float.init)
        lap("link")

        try poses(&out, local: local, camera: camera, start: spec.start)
        return out
    }

    private func poses(_ out: inout RallyOut, local: VideoMeta, camera: CameraModel?, start: Int) throws {
        let n = local.nFrames
        let rois = PlayerTracker.defaultFarROIs(camera: camera, meta: local)
        let vision = VisionPoseDetector(options: .init(farROIs: rois))
        var frames: [PlayerCandidates] = []
        var perFrame = [[(kp: [Float], box: BBox, score: Double, src: Int8)]](repeating: [], count: n)
        var clock = CFAbsoluteTimeGetCurrent()
        let seq = try video.frames(start: start, stop: start + n, step: poseStep)
        for (fi, pb) in seq {
            let k = fi - start
            let obs = try vision.observe(frame: pb)
            let dets = vision.detections(obs)
            let raw = Self.rawPeople(obs)
            perFrame[k] = (0..<dets.n).map { i in
                let r = Self.match(dets.keypoints[i], raw)
                return (r?.kp ?? Self.flat(dets.keypoints[i]), dets.boxes[i], dets.scores[i], r?.src ?? -1)
            }
            frames.append(try BGRAPixels.with(pb) { px in
                Candidates.build(frame: k, detections: dets, camera: camera, imageSize: local.size, image: px)
            })
        }
        if let e = seq.error { throw e }
        out.timing["pose"] = CFAbsoluteTimeGetCurrent() - clock
        clock = CFAbsoluteTimeGetCurrent()
        let tracks = PlayerTracker.assembleTracks(frames, meta: local, camera: camera, doubles: false)
        out.timing["track"] = CFAbsoluteTimeGetCurrent() - clock

        out.personStart = [0]
        for people in perFrame {
            for p in people {
                out.personKp += p.kp
                out.personBox += [p.box.x1, p.box.y1, p.box.x2, p.box.y2].map(Float.init)
                out.personScore.append(Float(p.score))
                out.personSource.append(p.src)
            }
            out.personStart.append(Int32(out.personScore.count))
        }
        out.playerPerson = [Int32](repeating: -1, count: n * 2)
        out.playerTrack = [Int32](repeating: -1, count: n * 2)
        out.playerBox = [Float](repeating: .nan, count: n * 8)
        out.playerKp = [Float](repeating: .nan, count: n * 2 * 51)
        out.playerFoot = [Float](repeating: .nan, count: n * 4)
        for o in tracks.observations where o.frame >= 0 && o.frame < n && o.side != .unknown {
            let s = o.side == .far ? 0 : 1, slot = o.frame * 2 + s
            if out.playerTrack[slot] >= 0 && out.playerTrack[slot] < o.trackID { continue }
            out.playerTrack[slot] = Int32(o.trackID)
            let b = [o.bbox.x1, o.bbox.y1, o.bbox.x2, o.bbox.y2]
            for j in 0..<4 { out.playerBox[slot * 4 + j] = Float(b[j]) }
            if let kp = o.keypoints {
                for (j, p) in kp.prefix(17).enumerated() {
                    out.playerKp[slot * 51 + j * 3] = Float(p.x)
                    out.playerKp[slot * 51 + j * 3 + 1] = Float(p.y)
                    out.playerKp[slot * 51 + j * 3 + 2] = Float(p.conf)
                }
            }
            if let f = o.footWorld { out.playerFoot[slot * 2] = Float(f.x); out.playerFoot[slot * 2 + 1] = Float(f.y) }
            let people = perFrame[o.frame]
            let base = Int(out.personStart[o.frame])
            if let best = people.indices.max(by: { iouBox(people[$0].box, o.bbox) < iouBox(people[$1].box, o.bbox) }),
               iouBox(people[best].box, o.bbox) > 0.5 {
                out.playerPerson[slot] = Int32(base + best)
            } else {
                out.playerPerson[slot] = -1
            }
        }
    }

    /// Every Vision pose of the frame (full pass and crop passes) as 19 joints in full-frame px.
    static func rawPeople(_ o: VisionPoseDetector.Observations) -> [(kp: [Float], src: Int8)] {
        func convert(_ poses: [VNHumanBodyPoseObservation], _ x0: Double, _ y0: Double, _ w: Double, _ h: Double, _ src: Int8)
            -> [(kp: [Float], src: Int8)] {
            poses.compactMap { obs in
                guard let pts = try? obs.recognizedPoints(.all) else { return nil }
                var kp = [Float](repeating: 0, count: 19 * 3)
                for (i, name) in Joints.vision.enumerated() {
                    guard let p = pts[name], p.confidence > 0 else { continue }
                    kp[i * 3] = Float(x0 + Double(p.location.x) * w)
                    kp[i * 3 + 1] = Float(y0 + (1 - Double(p.location.y)) * h)
                    kp[i * 3 + 2] = p.confidence
                }
                return (kp, src)
            }
        }
        var out = convert(o.poses, 0, 0, Double(o.imageSize.width), Double(o.imageSize.height), 0)
        for (i, c) in o.crops.enumerated() {
            out += convert(c.poses, Double(c.rect.x1), Double(c.rect.y1), Double(c.rect.width), Double(c.rect.height), Int8(i + 1))
        }
        return out
    }

    /// The raw pose a converted detection came from: same confident non-ankle joints (to float precision).
    static func match(_ det: [Keypoint], _ raw: [(kp: [Float], src: Int8)]) -> (kp: [Float], src: Int8)? {
        var best: (err: Double, i: Int)? = nil
        for (i, r) in raw.enumerated() {
            var err = 0.0, used = 0
            for j in 0..<15 where det[j].conf > 0 {
                err += abs(Double(r.kp[j * 3]) - det[j].x) + abs(Double(r.kp[j * 3 + 1]) - det[j].y)
                err += r.kp[j * 3 + 2] > 0 ? 0 : 1e6
                used += 1
            }
            guard used > 0 else { continue }
            err /= Double(used)
            if best == nil || err < best!.err { best = (err, i) }
        }
        guard let b = best, b.err < 0.5 else { return nil }
        return raw[b.i]
    }

    static func flat(_ kp: [Keypoint]) -> [Float] {
        kp.flatMap { [Float($0.x), Float($0.y), Float($0.conf)] } + [Float](repeating: 0, count: 6)
    }
}

func iouBox(_ a: BBox, _ b: BBox) -> Double {
    let ix = max(min(a.x2, b.x2) - max(a.x1, b.x1), 0), iy = max(min(a.y2, b.y2) - max(a.y1, b.y1), 0)
    let inter = ix * iy
    return inter / max(a.width * a.height + b.width * b.height - inter, 1e-9)
}
