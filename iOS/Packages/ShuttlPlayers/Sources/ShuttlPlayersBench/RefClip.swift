import Foundation
import ShuttlCore
import ShuttlPlayers

/// One Vision-vs-YOLO reference clip (/tmp/shuttl_players_yolo_ref/*.json, exported by
/// backend/scripts/swift_fixtures/players.py --compare): camera, sampled frames and the
/// Python YOLOv8s-pose tracks, which are the reference for recall.
struct RefClip {
    var name: String
    var meta: VideoMeta
    var camera: CameraModel
    var step: Int
    var sampled: [Int]
    var obs: [PlayerObservation]
    var video: URL

    init(_ url: URL, repo: URL) throws {
        let r = d(try JSONSerialization.jsonObject(with: Data(contentsOf: url)))
        name = r["name"] as? String ?? url.deletingPathExtension().lastPathComponent
        let md = d(r["meta"])
        meta = VideoMeta(fps: n(md["fps"]), width: Int(n(md["width"])), height: Int(n(md["height"])),
                         nFrames: Int(n(md["n_frames"])), timestamps: ns(md["timestamps"]))
        guard let cd = r["camera"] as? [String: Any] else { throw NSError(domain: "ref: no camera", code: 1) }
        camera = try ShuttlJSON.decoder().decode(CameraModel.self, from: JSONSerialization.data(withJSONObject: cd))
        step = Int(n(r["step"]))
        sampled = l(r["raw"]).map { Int(n(d($0)["frame"])) }
        obs = l(r["expect"]).map(d).map { e in
            let fw = ns(e["foot_world"]), b = ns(e["bbox"])
            return PlayerObservation(frame: Int(n(e["frame"])), trackID: Int(n(e["track_id"])),
                                     bbox: b.count == 4 ? BBox(x1: b[0], y1: b[1], x2: b[2], y2: b[3]) : BBox(x1: 0, y1: 0, x2: 0, y2: 0),
                                     score: n(e["score"]), side: CourtSide(rawValue: e["side"] as? String ?? "") ?? .unknown,
                                     footWorld: fw.count == 2 && fw[0].isFinite ? SIMD2(fw[0], fw[1]) : nil)
        }
        video = repo.appendingPathComponent(r["relative_path"] as? String ?? "")
    }

    static func files(_ dir: URL) throws -> [URL] {
        try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }
}

/// Phone setups of the layout study (heights / distances of tools/venue/calib_sweep.py).
let phoneSetups = [
    PhoneSetup(name: "phone h1.2 d0.5", height: 1.2, distance: 0.5),
    PhoneSetup(name: "phone h1.4 d1.5", height: 1.4, distance: 1.5),
    PhoneSetup(name: "phone h1.6 d3.0", height: 1.6, distance: 3.0),
    PhoneSetup(name: "phone 0.5x h1.4 d3.0", height: 1.4, distance: 3.0, hfovDeg: 106),
]

/// Crop passes of one combined detector: every layout's crops concatenated; `options` maps
/// a layout to its crop indices (the detections of that layout alone).
struct LayoutPlan {
    var rois: [PixelRect] = []
    var options: [(FarCropLayout, [Int])] = []

    init(_ e: Emulation) {
        for layout in FarCropLayout.allCases {
            let r = e.rects[layout] ?? []
            options.append((layout, Array(rois.count..<(rois.count + r.count))))
            rois += r
        }
    }
}
