// ShuttlPlayersBench — Apple Vision vs YOLOv8s-pose on held-out TrackNetV2 Test clips.
//
// Input: /tmp/shuttl_players_yolo_ref/*.json from
//   backend/.venv/bin/python -m scripts.swift_fixtures.players --skip-synthetic --compare 12
// (camera, raw YOLO detections per sampled frame, Python's final tracks).
// For every clip:
//   1. tracker parity: Swift tracker on the recorded YOLO detections == Python tracks;
//   2. Vision on the same sampled frames (AVFoundation decode): variants full frame,
//      + far ROI pass, + human-rectangle boxes; Swift tracker on each;
//   3. metrics against the YOLO tracks (reference): per-side detection / track recall,
//      foot disagreement (m), identity switches; box / keypoint agreement; timing.
// Usage: swift run -c release ShuttlPlayersBench [--ref DIR] [--repo DIR] [--clips N] [--out FILE]
//        [--layouts 1 [--time-every 10] [--dump DIR]]   far-crop layout study (Layouts.swift)
// On the shared Mac run builds and benches through the queue, e.g.
//   /tmp/heavy.sh players-layouts swift run -c release --scratch-path /tmp/sc_players ShuttlPlayersBench --layouts 1
import CoreVideo
import Foundation
import ShuttlCore
import ShuttlPlayers

// MARK: - arguments / JSON

var args = [String: String]()
var it = CommandLine.arguments.dropFirst().makeIterator()
while let a = it.next() { if a.hasPrefix("--") { args[String(a.dropFirst(2))] = it.next() ?? "" } }
let refDir = URL(fileURLWithPath: args["ref"] ?? "/tmp/shuttl_players_yolo_ref")
let repo = URL(fileURLWithPath: args["repo"] ?? URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().path)
let outURL = URL(fileURLWithPath: args["out"] ?? "/tmp/shuttl_players_bench.json")
let maxClips = Int(args["clips"] ?? "") ?? Int.max

// --layouts 1: far-crop layout study (none / single / lateral / depth) on the broadcast clips
// and on phone-like frames emulated from them (Layouts.swift). Default mode: detector study.
if args["layouts"] != nil {
    try await runLayouts(refDir: refDir, repo: repo, maxClips: maxClips,
                         timeEvery: Int(args["time-every"] ?? "") ?? 10,
                         out: URL(fileURLWithPath: args["out"] ?? "/tmp/shuttl_players_layouts.json"),
                         dumpDir: args["dump"].map { URL(fileURLWithPath: $0) })
    exit(0)
}

func d(_ x: Any?) -> [String: Any] { x as? [String: Any] ?? [:] }
func l(_ x: Any?) -> [Any] { x as? [Any] ?? [] }
func n(_ x: Any?) -> Double { (x as? NSNumber)?.doubleValue ?? .nan }
func ns(_ x: Any?) -> [Double] { l(x).map(n) }
func now() -> Double { CFAbsoluteTimeGetCurrent() }
func median(_ a: [Double]) -> Double { a.isEmpty ? .nan : { let s = a.sorted(); return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2 }() }
func pct(_ a: [Double], _ q: Double) -> Double {
    guard !a.isEmpty else { return .nan }
    let s = a.sorted(), p = q / 100 * Double(s.count - 1), i = Int(p), f = p - Double(i)
    return i + 1 < s.count ? s[i] + (s[i + 1] - s[i]) * f : s[i]
}
func dist(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double { ((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)).squareRoot() }

func detections(_ raw: Any?) -> FrameDetections {
    let r = d(raw)
    let boxes = l(r["boxes"]).map(ns).map { BBox(x1: $0[0], y1: $0[1], x2: $0[2], y2: $0[3]) }
    let kps = l(r["kps"]).map(ns).map { v in (0..<17).map { Keypoint(x: v[$0 * 3], y: v[$0 * 3 + 1], conf: v[$0 * 3 + 2]) } }
    return FrameDetections(boxes: boxes, scores: ns(r["scores"]), keypoints: kps)
}

// MARK: - metrics

struct SideMetrics: Codable {
    var refObs = 0, refSampled = 0
    var detHits = 0            // sampled frames: a gated candidate within 0.5 m of the reference foot
    var trackHits = 0          // all frames: a track of this side within 1.0 m of the reference foot
    var trackPresent = 0       // all frames: any track of this side
    var footErr: [Double] = [] // all frames with both
    var switches = 0           // correct (< 0.5 m) → wrong (> 1 m) episodes of one track id
    var wrongFrames = 0        // frames a track sits > 1 m from the reference player
    var ids = Set<Int>()
    mutating func add(_ o: SideMetrics) {
        refObs += o.refObs; refSampled += o.refSampled; detHits += o.detHits; trackHits += o.trackHits
        trackPresent += o.trackPresent; footErr += o.footErr; switches += o.switches; wrongFrames += o.wrongFrames
    }
    var summary: [String: Double] {
        ["det_recall": Double(detHits) / Double(max(refSampled, 1)), "track_recall": Double(trackHits) / Double(max(refObs, 1)),
         "track_present": Double(trackPresent) / Double(max(refObs, 1)),
         "foot_err_median_m": median(footErr), "foot_err_p90_m": pct(footErr, 90), "foot_err_p95_m": pct(footErr, 95),
         "id_switches": Double(switches), "wrong_frac": Double(wrongFrames) / Double(max(refObs, 1)),
         "n_ref_obs": Double(refObs)]
    }
}

/// Vision tracks vs reference (YOLO) tracks for one side.
func sideMetrics(side: CourtSide, ref: [PlayerObservation], cands: [PlayerCandidates], tracks: [PlayerObservation]) -> SideMetrics {
    var m = SideMetrics()
    let refSide = ref.filter { $0.side == side && $0.footWorld != nil }
    let refByFrame = Dictionary(grouping: refSide, by: \.frame)
    let trkSide = tracks.filter { $0.side == side && $0.footWorld != nil }
    let trkByFrame = Dictionary(grouping: trkSide, by: \.frame)
    m.ids = Set(trkSide.map(\.trackID))
    m.refObs = refSide.count
    for c in cands {
        for r in refByFrame[c.frame] ?? [] {
            m.refSampled += 1
            if c.pos.contains(where: { dist($0, r.footWorld!) < 0.5 }) { m.detHits += 1 }
        }
    }
    for r in refSide {
        let t = trkByFrame[r.frame] ?? []
        if !t.isEmpty { m.trackPresent += 1 }
        if let e = t.map({ dist($0.footWorld!, r.footWorld!) }).min() {
            m.footErr.append(e)
            if e < 1.0 { m.trackHits += 1 }
        }
    }
    for id in m.ids {                                          // identity switches
        var state = 0                                          // 0 unknown, 1 on the player, 2 elsewhere
        for o in trkSide.filter({ $0.trackID == id }).sorted(by: { $0.frame < $1.frame }) {
            guard let e = (refByFrame[o.frame] ?? []).map({ dist(o.footWorld!, $0.footWorld!) }).min() else { continue }
            if e > 1.0 { m.wrongFrames += 1 }
            if e < 0.5 { state = 1 } else if e > 1.0 { if state == 1 { m.switches += 1 }; state = 2 }
        }
    }
    return m
}

// MARK: - box / keypoint agreement

struct BoxStats: Codable {
    var iouJoint: [Double] = [], iouRect: [Double] = []
    var topErr: [Double] = [], bottomErr: [Double] = []           // (vision - yolo) / yolo height, joint box
    var fitTop: [Double] = [], fitBottom: [Double] = [], fitSide: [Double] = []
    var ankleErr: [Double] = [], hipErr: [Double] = [], shoulderErr: [Double] = []   // / yolo height
    var ankleDX: [Double] = [], ankleDY: [Double] = []                              // signed, / yolo height
    var ankleSwaps = 0, anklePairs = 0
    var yoloBothAnkles = 0, visionBothAnkles = 0, matched = 0
    var visionAnkleConf: [Double] = []
    var near = 0, far = 0, yoloNear = 0, yoloFar = 0
}

func boxAgreement(_ s: inout BoxStats, yolo: FrameDetections, vision: FrameDetections, rects: FrameDetections,
                  camera: CameraModel?, detector: VisionPoseDetector) {
    let minH = 0.025 * 720.0
    var usedV = Set<Int>()
    for i in 0..<yolo.n where yolo.boxes[i].height >= minH {
        let yb = yolo.boxes[i]
        let footY = SIMD2((yb.x1 + yb.x2) / 2, yb.y2)
        let isNear = camera.map { $0.pixelToFloor(footY).x < 0 } ?? (yb.y2 > 360)
        if isNear { s.yoloNear += 1 } else { s.yoloFar += 1 }
        let ious = vision.boxes.map { iou1nPublic(yb, $0) }
        guard let j = ious.indices.filter({ !usedV.contains($0) }).max(by: { ious[$0] < ious[$1] }), ious[j] > 0.3 else { continue }
        usedV.insert(j)
        s.matched += 1
        if isNear { s.near += 1 } else { s.far += 1 }
        let vb = vision.boxes[j], h = yb.height
        s.iouJoint.append(ious[j])
        if j < rects.n { s.iouRect.append(iou1nPublic(yb, rects.boxes[j])) }
        s.topErr.append((vb.y1 - yb.y1) / h); s.bottomErr.append((vb.y2 - yb.y2) / h)
        let good = vision.keypoints[j].filter { $0.conf >= detector.options.jointConfidence }
        if good.count >= 2 {
            let xs = good.map(\.x), ys = good.map(\.y), span = max(ys.max()! - ys.min()!, 1)
            s.fitTop.append((ys.min()! - yb.y1) / span); s.fitBottom.append((yb.y2 - ys.max()!) / span)
            s.fitSide.append(((xs.min()! - yb.x1) + (yb.x2 - xs.max()!)) / 2 / span)
        }
        let yk = yolo.keypoints[i], vk = vision.keypoints[j]
        func err(_ a: Int) -> Double? {
            guard yk[a].conf >= 0.35, vk[a].conf >= 0.1 else { return nil }
            return dist(SIMD2(yk[a].x, yk[a].y), SIMD2(vk[a].x, vk[a].y)) / h
        }
        for a in [15, 16] {
            if let e = err(a) {
                s.ankleErr.append(e)
                s.ankleDX.append((vk[a].x - yk[a].x) / h); s.ankleDY.append((vk[a].y - yk[a].y) / h)
            }
        }
        for a in [11, 12] { if let e = err(a) { s.hipErr.append(e) } }
        for a in [5, 6] { if let e = err(a) { s.shoulderErr.append(e) } }
        if yk[15].conf >= 0.35, yk[16].conf >= 0.35 { s.yoloBothAnkles += 1 }
        if vk[15].conf >= 0.35, vk[16].conf >= 0.35 { s.visionBothAnkles += 1 }
        for a in [15, 16] where vk[a].conf > 0 { s.visionAnkleConf.append(vk[a].conf) }
        if yk[15].conf >= 0.35, yk[16].conf >= 0.35, vk[15].conf >= 0.1, vk[16].conf >= 0.1 {
            let p = { (k: [Keypoint], a: Int) in SIMD2(k[a].x, k[a].y) }
            let same = dist(p(yk, 15), p(vk, 15)) + dist(p(yk, 16), p(vk, 16))
            let swap = dist(p(yk, 15), p(vk, 16)) + dist(p(yk, 16), p(vk, 15))
            s.anklePairs += 1
            if swap < 0.7 * same { s.ankleSwaps += 1 }
        }
    }
}

func iou1nPublic(_ a: BBox, _ b: BBox) -> Double { iou1n(a, [b])[0] }

// MARK: - main

let variants = ["vision_full", "vision_roi", "vision_roi_rawankle", "vision_rect_roi"]
var total: [String: [CourtSide: SideMetrics]] = [:]
var boxes = BoxStats()
var clipRows: [[String: Any]] = []
var parityFail = 0, parityClips = 0, parityObs = 0, parityMaxFoot = 0.0
var tPose = 0.0, tROI = 0.0, tRect = 0.0, tBuild = 0.0, tAssembleYolo = 0.0, tAssembleVision = 0.0, tDecode = 0.0
var nFrames = 0, nFramesROI = 0, yoloSecondsPython = 0.0
var allYolo: [(meta: VideoMeta, cands: [PlayerCandidates], camera: CameraModel?)] = []

let files = try FileManager.default.contentsOfDirectory(at: refDir, includingPropertiesForKeys: nil)
    .filter { $0.pathExtension == "json" }.sorted { $0.lastPathComponent < $1.lastPathComponent }.prefix(maxClips)
do {
        for f in files {
            let ref = d(try JSONSerialization.jsonObject(with: Data(contentsOf: f)))
            let name = ref["name"] as? String ?? f.lastPathComponent
            let md = d(ref["meta"])
            let meta = VideoMeta(fps: n(md["fps"]), width: Int(n(md["width"])), height: Int(n(md["height"])),
                                 nFrames: Int(n(md["n_frames"])), timestamps: ns(md["timestamps"]))
            let camera = try (ref["camera"] as? [String: Any]).map {
                try ShuttlJSON.decoder().decode(CameraModel.self, from: JSONSerialization.data(withJSONObject: $0))
            }
            let step = Int(n(ref["step"]))
            let raws = l(ref["raw"]).map(d)
            let yoloByFrame = Dictionary(uniqueKeysWithValues: raws.map { (Int(n($0["frame"])), detections($0)) })
            let refObs: [PlayerObservation] = l(ref["expect"]).map(d).map { e in
                let fw = ns(e["foot_world"])
                return PlayerObservation(frame: Int(n(e["frame"])), trackID: Int(n(e["track_id"])),
                                         bbox: BBox(x1: 0, y1: 0, x2: 0, y2: 0), score: n(e["score"]),
                                         side: CourtSide(rawValue: e["side"] as? String ?? "") ?? .unknown,
                                         footWorld: fw.count == 2 && fw[0].isFinite ? SIMD2(fw[0], fw[1]) : nil)
            }
            yoloSecondsPython += n(ref["detector_seconds"])

            // 1. tracker parity on the YOLO detections (singles: appearance unused)
            let yoloCands = raws.map { r in
                Candidates.build(frame: Int(n(r["frame"])), detections: detections(r), camera: camera, imageSize: meta.size)
            }
            allYolo.append((meta, yoloCands, camera))
            var t0 = now()
            let yTr = PlayerTracker.assembleTracks(yoloCands, meta: meta, camera: camera)
            tAssembleYolo += now() - t0
            var ok = yTr.observations.count == refObs.count
            for (a, b) in zip(yTr.observations, refObs) {
                if a.frame != b.frame || a.trackID != b.trackID || a.side != b.side { ok = false; break }
                if let fa = a.footWorld, let fb = b.footWorld { parityMaxFoot = max(parityMaxFoot, abs(fa.x - fb.x), abs(fa.y - fb.y)) }
            }
            let dbl = ref["doubles_auto"] as? Bool ?? false
            if !dbl { parityClips += 1; parityObs += refObs.count; if !ok { parityFail += 1 } }

            // 2. Vision on the same frames
            let url = repo.appendingPathComponent(ref["relative_path"] as? String ?? "")
            let src = try await VideoSource.open(url)
            let ruleROI = PlayerTracker.defaultFarROI(camera: camera, meta: meta, farCrop: nil)
            let forcedROI = PlayerTracker.defaultFarROI(camera: camera, meta: meta, farCrop: true)
            var opts = VisionPoseDetector.Options(farROI: forcedROI)
            opts.alwaysRequestRectangles = true
            let det = VisionPoseDetector(options: opts)
            var cands: [String: [PlayerCandidates]] = [:]
            var tDec0 = now()
            for (fi, pb) in try src.frames(step: step) {
                tDecode += now() - tDec0
                let obs = try det.observe(frame: pb)
                let dFull = det.detections(obs, boxSource: .joints, useROI: false)
                let dROI = det.detections(obs, boxSource: .joints, useROI: true)
                let dRect = det.detections(obs, boxSource: .humanRectangles, useROI: true)
                let dRaw = det.detections(obs, boxSource: .joints, useROI: true, ankleDrop: 0)
                let dFullRaw = det.detections(obs, boxSource: .joints, useROI: false, ankleDrop: 0)
                let rectsFull = det.detections(obs, boxSource: .humanRectangles, useROI: false)
                t0 = now()
                try BGRAPixels.with(pb) { px in
                    cands["vision_full", default: []].append(
                        Candidates.build(frame: fi, detections: dFull, camera: camera, imageSize: meta.size, image: px))
                    tBuild += now() - t0
                    cands["vision_roi", default: []].append(
                        Candidates.build(frame: fi, detections: dROI, camera: camera, imageSize: meta.size, image: px))
                    cands["vision_roi_rawankle", default: []].append(
                        Candidates.build(frame: fi, detections: dRaw, camera: camera, imageSize: meta.size, image: px))
                    cands["vision_rect_roi", default: []].append(
                        Candidates.build(frame: fi, detections: dRect, camera: camera, imageSize: meta.size, image: px))
                }
                if let y = yoloByFrame[fi] {
                    boxAgreement(&boxes, yolo: y, vision: dFullRaw, rects: rectsFull, camera: camera, detector: det)
                }
                nFrames += 1
                tDec0 = now()
            }
            tPose += det.timing.pose; tROI += det.timing.roiPose; tRect += det.timing.rects
            nFramesROI += forcedROI == nil ? 0 : det.timing.frames
            var row: [String: Any] = ["clip": name, "sampled_frames": raws.count, "parity_ok": ok,
                                      "python_far_roi": ref["far_roi"] is NSNull ? false : ref["far_roi"] != nil,
                                      "swift_rule_roi": ruleROI != nil,
                                      "far_baseline_height_px": n(ref["far_baseline_height_px"])]
            for v in variants {
                t0 = now()
                let tr = PlayerTracker.assembleTracks(cands[v]!, meta: meta, camera: camera)
                if v == "vision_full" { tAssembleVision += now() - t0 }
                for side in [CourtSide.near, .far] {
                    let m = sideMetrics(side: side, ref: refObs, cands: cands[v]!, tracks: tr.observations)
                    total[v, default: [:]][side, default: SideMetrics()].add(m)
                    row["\(v)_\(side.rawValue)"] = m.summary.merging(["ids": Double(m.ids.count)]) { a, _ in a }
                }
            }
            clipRows.append(row)
            let vf = d(row["vision_full_far"]), vr = d(row["vision_roi_far"]), vn = d(row["vision_full_near"])
            print(String(format: "%-18@ frames %3d parity %@ | near det %.3f trk %.3f err %.3f | far det %.3f trk %.3f err %.3f | far+roi det %.3f trk %.3f err %.3f sw %.0f",
                         name as NSString, raws.count, ok ? "ok" : "FAIL",
                         n(vn["det_recall"]), n(vn["track_recall"]), n(vn["foot_err_median_m"]),
                         n(vf["det_recall"]), n(vf["track_recall"]), n(vf["foot_err_median_m"]),
                         n(vr["det_recall"]), n(vr["track_recall"]), n(vr["foot_err_median_m"]), n(vr["id_switches"])))
        }
} catch { print("bench error:", error) }

// MARK: - long-match scaling: all clips' YOLO candidates concatenated, repeated

var scaling: [[String: Double]] = []
for reps in [1, 4] where !allYolo.isEmpty {
    var cands: [PlayerCandidates] = [], ts: [Double] = []
    var off = 0
    for _ in 0..<reps {
        for (meta, cs, _) in allYolo {
            let t0 = ts.last.map { $0 + 1 / 30.0 } ?? 0
            ts += meta.timestamps.map { $0 + t0 }
            cands += cs.map { var c = $0; c.frame += off; return c }
            off += meta.nFrames
        }
    }
    let meta = VideoMeta(fps: 30, width: 1280, height: 720, nFrames: ts.count, timestamps: ts)
    let t0 = now()
    let tr = PlayerTracker.assembleTracks(cands, meta: meta, camera: allYolo[0].camera)
    let dt = now() - t0
    scaling.append(["sampled_frames": Double(cands.count), "video_s": ts.last ?? 0, "assemble_s": dt,
                    "ms_per_sampled_frame": 1000 * dt / Double(cands.count), "observations": Double(tr.observations.count)])
    print(String(format: "[scaling] %d sampled frames (%.0f s of video): assemble %.3f s = %.3f ms/sampled frame",
                 cands.count, ts.last ?? 0, dt, 1000 * dt / Double(cands.count)))
}

// MARK: - summary

func ms(_ s: Double, _ k: Int) -> Double { 1000 * s / Double(max(k, 1)) }
var summary: [String: Any] = [:]
for v in variants {
    for side in [CourtSide.near, .far] { summary["\(v)_\(side.rawValue)"] = total[v]?[side]?.summary ?? [:] }
}
summary["tracker_parity_on_yolo"] = ["clips": parityClips, "failed": parityFail, "observations": parityObs,
                                     "max_foot_diff_m": parityMaxFoot]
summary["boxes"] = [
    "matched": boxes.matched, "yolo_near": boxes.yoloNear, "yolo_far": boxes.yoloFar,
    "matched_near": boxes.near, "matched_far": boxes.far,
    "iou_joint_median": median(boxes.iouJoint), "iou_rect_median": median(boxes.iouRect),
    "top_err_median": median(boxes.topErr), "bottom_err_median": median(boxes.bottomErr),
    "bottom_abs_err_median": median(boxes.bottomErr.map(abs)),
    "fit_pad_top": median(boxes.fitTop), "fit_pad_bottom": median(boxes.fitBottom), "fit_pad_side": median(boxes.fitSide),
    "ankle_err_median_h": median(boxes.ankleErr), "hip_err_median_h": median(boxes.hipErr),
    "ankle_dx_median_h": median(boxes.ankleDX), "ankle_dy_median_h": median(boxes.ankleDY),
    "ankle_dy_p25_h": pct(boxes.ankleDY, 25), "ankle_dy_p75_h": pct(boxes.ankleDY, 75),
    "shoulder_err_median_h": median(boxes.shoulderErr),
    "ankle_lr_swap_frac": Double(boxes.ankleSwaps) / Double(max(boxes.anklePairs, 1)),
    "both_ankles_yolo": Double(boxes.yoloBothAnkles) / Double(max(boxes.matched, 1)),
    "both_ankles_vision": Double(boxes.visionBothAnkles) / Double(max(boxes.matched, 1)),
    "vision_ankle_conf_median": median(boxes.visionAnkleConf),
]
summary["timing_ms_per_sampled_frame"] = [
    "vision_pose_full": ms(tPose, nFrames), "vision_pose_roi": ms(tROI, nFramesROI), "vision_rects": ms(tRect, nFrames),
    "candidates_with_appearance": ms(tBuild, nFrames), "assemble_yolo": ms(tAssembleYolo, nFrames),
    "assemble_vision": ms(tAssembleVision, nFrames), "decode_1280x720": ms(tDecode, nFrames),
    "python_yolo_mps_incl_decode": ms(yoloSecondsPython, nFrames),
]
summary["frames"] = nFrames
summary["scaling"] = scaling
summary["clips"] = clipRows
/// JSONSerialization throws (NSException) on NaN / inf: write them as null.
func clean(_ x: Any) -> Any {
    switch x {
    case let v as Double: return v.isFinite ? v : NSNull()
    case let v as [String: Any]: return v.mapValues(clean)
    case let v as [String: Double]: return v.mapValues { $0.isFinite ? $0 as Any : NSNull() }
    case let v as [Any]: return v.map(clean)
    default: return x
    }
}
summary = clean(summary) as! [String: Any]
if let data = try? JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys]) {
    try? data.write(to: outURL)
}
@MainActor func show(_ k: String) {
    if let data = try? JSONSerialization.data(withJSONObject: summary[k] ?? [:], options: [.sortedKeys]),
       let s = String(data: data, encoding: .utf8) { print(k + ":", s) }
}
print("\n== \(files.count) clips, \(nFrames) sampled frames ==")
for v in variants { show("\(v)_near"); show("\(v)_far") }
show("tracker_parity_on_yolo"); show("boxes"); show("timing_ms_per_sampled_frame")
print("wrote", outURL.path)
