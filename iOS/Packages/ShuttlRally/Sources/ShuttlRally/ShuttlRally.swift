import Foundation
import ShuttlCore

/// On-device port of backend/app/services/tracking/rally.py, shots.py and
/// serialize.py (summarize + per-rally/shot JSON shape).
///
/// Public entry points (Python name → Swift):
///   rally.segment_rallies   → `segmentRallies(track:meta:hits:segments:camera:players:config:landingSigmaM:)`
///   rally.score_match       → `scoreMatch(_:firstServer:gamesToWin:minConfidence:)`
///   shots.classify_shots    → `classifyShots(_:camera:players:handedness:)` (details in `ShotClassifier`)
///   serialize.summarize     → `summarize(_:)`; dict shapes in `RallySerialization`
///
/// `landingSigmaM` carries reconstruct3d's per-segment `landing_sigma_m` extra
/// (ShotSegment3D has no field for it); nil → `RallyConfig.defaultLandingSigmaM`.
public enum ShuttlRally {
    /// Schema version of the analysis JSON (serialize.SCHEMA_VERSION).
    public static let schemaVersion = RallySerialization.schemaVersion
}
