import CoreVideo
import Foundation
import ShuttlCore
import ShuttlPlayers

/// Accumulated result of one (setup, layout) cell of the study.
struct Cell {
    var near = SideMetrics(), far = SideMetrics()
    var ms: [Double] = []                  // production-config detect() wall time per timed frame
}

/// Layout study: for every reference clip and setup (native broadcast + emulated phones),
/// Vision on every sampled frame (15 Hz), each layout's detections merged from one combined
/// set of passes, the Swift tracker per layout, metrics vs the YOLO tracks; every
/// `timeEvery`-th frame each layout also runs alone in its production config, timed.
@MainActor
func runLayouts(refDir: URL, repo: URL, maxClips: Int, timeEvery: Int, out: URL, dumpDir: URL? = nil) async throws {
    let clips = try RefClip.files(refDir).prefix(maxClips).map { try RefClip($0, repo: repo) }
    var cells: [String: [String: Cell]] = [:]
    var farPx: [String: [Double]] = [:], kOf: [String: [String: [Double]]] = [:]
    let timed: [(String, FarCropLayout, Bool)] = [("none", .none, false), ("single", .single, false),
                                                 ("lateral", .lateral, false), ("lateral_batched", .lateral, true),
                                                 ("depth", .depth, false)]
    let t0 = CFAbsoluteTimeGetCurrent()
    for clip in clips {
        let refFar = Dictionary(grouping: clip.obs.filter { $0.side == .far }, by: \.frame)
        let emus = [Emulation.native(clip.camera)] + phoneSetups.compactMap { Emulation(phone: $0, broadcast: clip.camera) }
        let plans = emus.map(LayoutPlan.init)
        let dets = plans.map { VisionPoseDetector(options: .init(farROIs: $0.rois)) }
        let prod = emus.map { e in timed.map { t -> VisionPoseDetector in
            var o = VisionPoseDetector.Options(farROIs: e.rects[t.1] ?? []); o.batchCrops = t.2
            return VisionPoseDetector(options: o)
        } }
        var cands = emus.map { _ in [FarCropLayout: [PlayerCandidates]]() }
        let warper = Warper()
        let want = Set(clip.sampled)
        var k = 0
        let src = try await VideoSource.open(clip.video)
        for (fi, pb) in try src.frames(step: clip.step) where want.contains(fi) {
            for (ei, e) in emus.enumerated() {
                let frame = e.setup == "broadcast" ? pb : try warper.warp(pb, s: e.s, t: e.t, width: e.size.width, height: e.size.height)
                if let r = refFar[fi]?.first, r.bbox.height > 0 { farPx[e.setup, default: []].append(r.bbox.height * e.s) }
                if let dumpDir, k == 0 {
                    let rs = FarCropLayout.allCases.flatMap { l in (e.rects[l] ?? []).map { ($0, layoutColour[l]!) } }
                    dumpFrame(frame, rects: rs, camera: e.camera,
                              to: dumpDir.appendingPathComponent("\(clip.name)_\(e.setup.replacingOccurrences(of: " ", with: "_")).png"))
                }
                let obs = try dets[ei].observe(frame: frame)
                try BGRAPixels.with(frame) { px in
                    for (layout, idx) in plans[ei].options {
                        let d = dets[ei].detections(obs, crops: idx)
                        cands[ei][layout, default: []].append(
                            Candidates.build(frame: fi, detections: d, camera: e.camera, imageSize: e.size, image: px))
                    }
                }
                if k % timeEvery == 0 {
                    for j in 0..<timed.count {
                        let ti = (j + k / timeEvery) % timed.count      // rotate the order: load drift hits all
                        let t = CFAbsoluteTimeGetCurrent()
                        _ = try prod[ei][ti].detect(frame: frame)
                        cells[e.setup, default: [:]][timed[ti].0, default: Cell()].ms.append(1000 * (CFAbsoluteTimeGetCurrent() - t))
                    }
                }
            }
            k += 1
        }
        for (ei, e) in emus.enumerated() {
            let meta = VideoMeta(fps: clip.meta.fps, width: e.size.width, height: e.size.height,
                                 nFrames: clip.meta.nFrames, timestamps: clip.meta.timestamps)
            for (layout, _) in plans[ei].options {
                let c = cands[ei][layout] ?? []
                let tr = PlayerTracker.assembleTracks(c, meta: meta, camera: e.camera)
                var cell = cells[e.setup, default: [:]][layout.rawValue, default: Cell()]
                cell.near.add(sideMetrics(side: .near, ref: clip.obs, cands: c, tracks: tr.observations))
                cell.far.add(sideMetrics(side: .far, ref: clip.obs, cands: c, tracks: tr.observations))
                cells[e.setup]![layout.rawValue] = cell
                // k = sqrt(crop area) / far-player px of the largest crop (full frame for .none):
                // Vision's recall follows the region's area per player, not its long side
                let side = (e.rects[layout] ?? []).map { (Double($0.width) * Double($0.height)).squareRoot() }.max()
                    ?? (Double(e.size.width) * Double(e.size.height)).squareRoot()
                kOf[e.setup, default: [:]][layout.rawValue, default: []].append(side / Emulation.farPx(e.camera))
            }
        }
        print(String(format: "[layouts] %@ done: %d frames, %.0f s elapsed", clip.name, k, CFAbsoluteTimeGetCurrent() - t0))
    }
    writeLayoutReport(cells: cells, farPx: farPx, kOf: kOf, setups: ["broadcast"] + phoneSetups.map(\.name),
                      timedNames: timed.map(\.0), out: out)
}
