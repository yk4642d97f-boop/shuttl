/// Camera calibration (intrinsics + pose) from badminton court lines — public API.
/// Port of backend/app/services/tracking/court_calibration.py.
///
///   calibrateFromPoints(points, size)          → CameraModel   (user-tapped keypoints; throws)
///   calibrateFromFrames(frames, size, hint)    → CameraModel?  (automatic; nil = not confident)
///   refineCamera(camera, frames)               → CameraModel   (dense line refinement; never worse)
///
/// Camera model: square pixels, principal point at the image centre, optional k1.
/// World frame / keypoint names: ShuttlCore.CourtModel. The near half (x < 0) is always
/// the camera side. Frames are ImageU8 (RGB or BGR — the algorithm is channel-order
/// invariant: it only uses per-pixel min/max over channels and per-channel medians).
import Foundation
import simd
import ShuttlCore

public enum CourtCalibration {

    /// Camera from >= 4 named floor keypoints (`CourtModel.keypointNames`, e.g. the four
    /// corners "nb_ld", "nb_rd", "fb_ld", "fb_rd"). Unknown names are ignored.
    /// Homography DLT → focal from Zhang's constraints (principal point at centre) → pose
    /// → Levenberg–Marquardt on reprojection error (weak focal prior). Mirrored left/right
    /// labels are corrected; a camera behind the far baseline is rotated 180° so the
    /// camera side is always x < 0.
    public static func calibrateFromPoints(_ points: [String: SIMD2<Double>], size: ImageSize) throws -> CameraModel {
        try ManualCalibration.calibrate(points, size: size)
    }

    /// Dense refinement of `camera` against detected court-line pixels (`refine_camera`).
    ///
    /// Minimises point-to-line distances between model line samples (every 10 cm) and
    /// sub-pixel ridge centres found along the image normal, over focal, rotation,
    /// translation (+ k1 if it helps), with a soft-L1 loss. Never returns a worse camera:
    /// falls back to the input if the fit degrades (coverage drops, fewer lines, or a
    /// model line lands on bare floor). `frames`: sampled video frames (temporal median of
    /// up to 7) or a single median image.
    public static func refineCamera(_ camera: CameraModel, frames: [ImageU8], searchPx: Double? = nil,
                                    config: CourtCalibrationConfig = .default) -> CameraModel {
        refineCamera(camera, frames: frames, fitK1: .auto, searchPx: searchPx, config: config)
    }

    static func refineCamera(_ camera: CameraModel, frames: [ImageU8], fitK1: K1Mode, searchPx: Double? = nil,
                             config: CourtCalibrationConfig = .default) -> CameraModel {
        guard let img = FrameOps.toImage(frames, size: camera.imageSize, maxFrames: config.medianMaxFrames) else { return camera }
        // line paint model: best `paintFit` under the given camera (RGB only); a non-white
        // model's refinement is kept only if its paint fit did not drop, else white refines
        if img.channels == 3 {
            let fit = AutoCalibration.paintFit(img, camera)
            var best = LineColour.allCases[0]
            for c in LineColour.allCases where fit[c]! > fit[best]! { best = c }
            if best != .white {
                let feat = Features(img, config: config, colour: best)
                let (cam, moved) = refineCameraChecked(camera, features: feat, fitK1: fitK1, searchPx: searchPx, config: config)
                if moved, AutoCalibration.paintFit(img, cam)[best]! >= fit[best]! { return cam }
            }
        }
        let feat = Features(img, config: config)
        return refineCamera(camera, features: feat, fitK1: fitK1, searchPx: searchPx, config: config)
    }

    static func refineCamera(_ camera: CameraModel, features feat: Features, fitK1: K1Mode = .auto,
                             searchPx: Double? = nil, config: CourtCalibrationConfig = .default) -> CameraModel {
        refineCameraChecked(camera, features: feat, fitK1: fitK1, searchPx: searchPx, config: config).0
    }

    /// `_refine_colour`: (camera, whether the refinement was kept — Python `cam is not camera`).
    static func refineCameraChecked(_ camera: CameraModel, features feat: Features, fitK1: K1Mode = .auto,
                                    searchPx: Double? = nil, config: CourtCalibrationConfig = .default) -> (CameraModel, Bool) {
        let mt = Matcher(feat: feat, config: config)
        let cx = mget(camera.K, 0, 2), cy = mget(camera.K, 1, 2)
        let search = searchPx ?? max(8.0, 0.012 * feat.D)
        let before = mt.fitStats(camParams(camera, withK1: (camera.dist.first ?? 0) != 0), cx: cx, cy: cy,
                                 .fine, search: max(3.0, 0.003 * feat.D))
        guard let (cam0, stats) = mt.refine(camera, search0: search, fitK1: fitK1) else { return (camera, false) }
        if stats.coverage < before.coverage - 0.02 || stats.nT < 2 || stats.nL < 2 || stats.nAbsent > before.nAbsent {
            return (camera, false)
        }
        var cam = cam0
        cam.confidence = max(camera.confidence, mt.confidence(stats, margin: 1.0))
        return (cam, true)
    }

    /// `homography_court_to_image`: 3x3 H mapping floor (X, Y, 1) in metres → pixels.
    public static func homographyCourtToImage(_ camera: CameraModel) -> simd_double3x3 {
        camera.floorHomography()
    }

    /// `image_to_court`: pixels → world points on the plane Z = z (NaN where the ray misses).
    public static func imageToCourt(_ camera: CameraModel, _ uv: [SIMD2<Double>], z: Double = 0) -> [SIMD3<Double>] {
        camera.pixelToFloor(uv, z: z)
    }
}
