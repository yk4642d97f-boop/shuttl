import Foundation
import ShuttlCore

/// Options for one analysis run (port of pipeline_v2.PipelineOptions, minus the
/// server-only weights_dir / device / pose_model knobs).
public struct PipelineOptions: Sendable, Equatable, Codable {
    /// nil = auto-detect singles / doubles.
    public var doubles: Bool?
    /// User-tapped court keypoints by CourtModel name (e.g. "nb_ld").
    public var manualCorners: [String: SIMD2<Double>]?
    /// Manual corners are given as 0…1 of the upright frame.
    public var cornersNormalized: Bool
    /// Camera from a previous session / the device, used when auto-calibration fails.
    public var cameraHint: CameraModel?
    /// The lens's real focal length in px at the video size (device intrinsics). Automatic
    /// calibration rejects a camera more than 15 % off it; it never picks a different one.
    public var focalHintPx: Double?
    /// nil = every ~1/PLAYER_HZ s (fps-aware).
    public var playerStep: Int?
    public var calibrationFrames: Int

    public init(doubles: Bool? = nil, manualCorners: [String: SIMD2<Double>]? = nil, cornersNormalized: Bool = true,
                cameraHint: CameraModel? = nil, playerStep: Int? = nil, calibrationFrames: Int = 24,
                focalHintPx: Double? = nil) {
        self.doubles = doubles; self.manualCorners = manualCorners; self.cornersNormalized = cornersNormalized
        self.cameraHint = cameraHint; self.playerStep = playerStep; self.calibrationFrames = calibrationFrames
        self.focalHintPx = focalHintPx
    }
}

/// Pose sampling rate. step = 2 was tuned on 30 fps footage; a fixed frame step would
/// double the pose work at 60 fps for no tracking gain.
public let playerHz = 15.0

/// pipeline_v2.player_step.
public func playerStep(fps: Double?, override: Int? = nil) -> Int {
    if let override { return max(1, override) }
    let f = (fps ?? 30).isFinite && (fps ?? 30) > 0 ? (fps ?? 30) : 30
    return max(1, Num.iround(f / playerHz))
}
