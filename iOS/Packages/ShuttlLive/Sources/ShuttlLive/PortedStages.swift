import Foundation
import ShuttlAnalysis
import ShuttlCore
import ShuttlHits
import ShuttlLinker
import ShuttlRally
import ShuttlReconstruct

/// The pure-geometry stages (no video access, no ML models besides the optional
/// learned hit re-scorer): link → hits → reconstruct → rallies → shots → score.
/// Each forwards to its package with Python defaults.
public enum PortedStages {
    public static func link(_ candidates: [[ShuttleCandidate]], meta: VideoMeta) throws -> Track2D {
        try linkCandidates(candidates, meta: meta)
    }

    /// Uses the process-wide learned re-scorer if one is installed (HitModelRegistry), like Python.
    public static func hits(track: Track2D, meta: VideoMeta, camera: CameraModel?, players: [PlayerObservation]?) -> [HitEvent] {
        detectHits(track: track, meta: meta, camera: camera, players: players)
    }

    /// Keeps each segment's landing sigma (rally.py reads it for close in/out calls).
    public static func reconstruct(track: Track2D, hits: [HitEvent], camera: CameraModel, meta: VideoMeta,
                                   players: [PlayerObservation]?) -> ReconstructionResult {
        let segs = Reconstruct3D.reconstructSegments(track: track, hits: hits, camera: camera, meta: meta, players: players)
        // rally._landing_sigma: the attribute, else r3.landing_sigma(seg) from the fit covariance.
        let sigmas = segs.map { s -> Double? in
            if let v = s.landingSigmaM, v.isFinite { return v }
            return Reconstruct3D.landingSigma(s)
        }
        return ReconstructionResult(segments: segs.map(\.segment), landingSigmaM: sigmas)
    }

    public static func rallies(track: Track2D, hits: [HitEvent], reconstruction: ReconstructionResult,
                               meta: VideoMeta, camera: CameraModel?, players: [PlayerObservation]?) -> [RallyV2] {
        segmentRallies(track: track, meta: meta, hits: hits, segments: reconstruction.segments,
                       camera: camera, players: players, landingSigmaM: reconstruction.landingSigmaM)
    }

    public static func shots(_ rally: RallyV2, camera: CameraModel?, players: [PlayerObservation]?,
                             meta: VideoMeta? = nil, track: Track2D? = nil, learned: Bool = true) -> RallyV2 {
        classifyShots(rally, camera: camera, players: players, meta: meta, track: track, learned: learned)
    }

    public static func score(_ rallies: [RallyV2], players: [PlayerObservation]? = nil) -> [JSONValue] {
        scoreMatch(rallies, doubles: detectDoubles(players) ?? false)
    }
}
