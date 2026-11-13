import Foundation
import ShuttlCore

// sim.py truth → pipeline data types (truth_track / truth_hits / truth_players /
// truth_segments / truth_rallies) and the benchmark's view of a sample (SimTruth).

public extension SimScenario {
    static let playerIDs = ["near": 1, "far": 2, "near2": 3, "far2": 4]

    /// Perfect 2-D track (visible frames only).
    func truthTrack() -> Track2D {
        Track2D(t: timestamps, xy: (0..<n).map { visible[$0] ? shuttlePx[$0] : SIMD2(.nan, .nan) },
                conf: visible.map { $0 ? 1 : 0 }, visible: visible, interpolated: [Bool](repeating: false, count: n))
    }

    func truthHits() -> [HitEvent] {
        hits.map { h in
            HitEvent(frame: h.frame, t: h.t, xyPx: camera.project(h.pos), hitter: CourtSide(rawValue: h.hitter) ?? .unknown,
                     confidence: 1.0, playerTrackID: Self.playerIDs[h.player])
        }
    }

    /// Per-frame PlayerObservation (every `every` frames); players fully outside the image skipped.
    func truthPlayers(every: Int = 1) -> [PlayerObservation] {
        let w = Double(imageSize.width), h = Double(imageSize.height)
        var out: [PlayerObservation] = []
        for i in stride(from: 0, to: n, by: max(every, 1)) {
            for p in players {
                let b = p.bbox[i]
                if !(b.x1.isFinite && b.y1.isFinite && b.x2.isFinite && b.y2.isFinite) || b.x2 < 0 || b.x1 >= w || b.y2 < 0 || b.y1 >= h { continue }
                out.append(PlayerObservation(frame: i, trackID: Self.playerIDs[p.key]!, bbox: b, score: 1.0,
                                             side: CourtSide(rawValue: p.side) ?? .unknown, footWorld: p.foot[i]))
            }
        }
        return out
    }

    /// True flights (hit → next contact / floor / net) as ShotSegment3D.
    func truthSegments() -> [ShotSegment3D] {
        shots.map { sh in
            let fr = (sh.frame...max(sh.frame, sh.endFrame)).filter { $0 <= sh.endFrame && timestamps[$0] >= sh.t0 && timestamps[$0] <= sh.endT }
            let pos = SimPhysics.simulatePositions(p0: sh.p0, v0: sh.v0, times: fr.map { timestamps[$0] - sh.t0 }, L: sh.dragLength)
            let lands = sh.outcome == "in" || sh.outcome == "out"
            return ShotSegment3D(startFrame: sh.frame, endFrame: sh.endFrame, t0: sh.t0, p0: sh.p0, v0: sh.v0,
                                 dragLength: sh.dragLength, rmsePx: 0, frames: fr, positions: pos,
                                 landingXY: lands ? sh.landingXY : nil, landingT: lands ? sh.landingT : nil,
                                 netCrossingHeight: sh.netCrossingHeight.isFinite ? sh.netCrossingHeight : nil, confidence: 1.0)
        }
    }

    /// RallyV2 list mirroring the ground truth (shot types filled).
    func truthRallies(withShots: Bool = true) -> [RallyV2] {
        let th = truthHits(), segs = truthSegments()
        return rallies.map { r in
            var shotsOut: [ShotEventV2] = []
            if withShots {
                for k in r.firstShot..<(r.firstShot + r.nShots) {
                    let sh = shots[k]
                    shotsOut.append(ShotEventV2(hit: th[k], segment: segs[k], shotType: ShotType(rawValue: sh.shotType) ?? .unknown,
                                                shotTypeConfidence: 1.0, hitterPosWorld: player(sh.player)?.foot[sh.frame],
                                                hitHeightM: sh.p0.z))
                }
            }
            return RallyV2(startFrame: r.startFrame, endFrame: r.endFrame, shots: shotsOut,
                           endReason: RallyEndReason(rawValue: r.endReason) ?? .unknown, landingXY: r.landingXY,
                           inOutMarginM: r.margin, winner: CourtSide(rawValue: r.winner) ?? .unknown, confidence: 1.0)
        }
    }

    /// The benchmark's truth of this sample (what bench bundles record; see eval.py `sim_truth`).
    func benchTruth(stage: String) -> SimTruth {
        let (trkPx, trkVis) = imageTruth(occluded: true)
        let (_, detVis) = imageTruth(occluded: false)
        let shotRecs = shots.map { sh -> EvalSegment in
            let lands = sh.outcome == "in" || sh.outcome == "out"
            return EvalSegment(startFrame: sh.frame, p0: sh.p0, v0: sh.v0,
                               netCrossingHeight: sh.netCrossingHeight.isFinite ? sh.netCrossingHeight : nil,
                               landingXY: lands ? sh.landingXY : nil)
        }
        var t = SimTruth(seed: seed, n: n, fps: fps, width: imageSize.width, height: imageSize.height, doubles: doubles,
                         nShots: shots.count, nRallies: rallies.count, trkPx: trkPx, trkVis: trkVis, detVis: detVis,
                         hits: hits.map { EvalHit(frame: $0.frame, hitter: $0.hitter, shotType: $0.shotType) },
                         shots: shotRecs,
                         rallies: rallies.map { EvalRally(startFrame: $0.startFrame, endFrame: $0.endFrame, endReason: $0.endReason,
                                                          winner: $0.winner, landingXY: $0.landingXY, margin: $0.margin) },
                         camera: camera)
        if stage == "oracle" {
            t.finalCalls = rallies.map { r in
                let sh = shots[r.firstShot + r.nShots - 1]
                return SimTruth.FinalCall(frame: sh.frame, isServe: sh.index == r.firstShot, inOut: sh.inOut,
                                          margin: sh.margin.isFinite ? sh.margin : nil)
            }
            t.rallyTypes = rallies.map { r in (r.firstShot..<(r.firstShot + r.nShots)).map { shots[$0].shotType } }
        }
        return t
    }
}

/// benchmark.py's synthetic sample generation (`make_sim` + the candidate corruption)
/// on native Swift streams — scenarios statistically like the Python benchmark's.
public enum SimBench {
    public static func makeSim(seed: Int, options: BenchOptions, source: any SimRandomSource) throws -> SimScenario {
        let rng = source.stream([seed, 7])
        let style = options.args["style"]?.string ?? "random"
        let cam = try SimCameras.randomCamera(rng, imageSize: options.imageSize, style: style)
        let doubles = options.args["doubles"]?.bool ?? (rng.random() < 0.3)
        let fps = options.args["fps"]?.double ?? 60
        let rallies = options.args["rallies"]?.int ?? 1
        if rallies > 1 { return try Sim.simulateMatch(seed: seed, nRallies: rallies, camera: cam, fps: fps, doubles: doubles, source: source) }
        return try Sim.simulateRally(seed: seed, camera: cam, fps: fps, doubles: doubles, source: source)
    }

    /// `--n` samples from `--seed`: truth + corrupt_detections candidates (stream [seed, 11]).
    public static func samples(options: BenchOptions, source: any SimRandomSource = NativeRandomSource()) throws -> [SimSample] {
        let n = options.args["n"]?.int ?? 10, seed0 = options.args["seed"]?.int ?? 0
        return try (0..<n).map { k in
            let seed = seed0 + k
            let sim = try makeSim(seed: seed, options: options, source: source)
            let rng = source.stream([seed, 11])
            let c = SimNoise.corruptDetections(sim, rng: rng, sigmaPx: options.args["sigma"]?.double ?? 1.5,
                                               missRate: options.args["miss_rate"]?.double ?? 0.1,
                                               falsePositiveRate: options.args["fp_rate"]?.double ?? 0.03)
            return SimSample(seed: seed, truth: sim.benchTruth(stage: options.stage), candidates: c, camera: sim.camera)
        }
    }
}
