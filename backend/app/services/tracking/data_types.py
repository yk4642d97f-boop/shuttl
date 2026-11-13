from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ShuttleDetection:
    """Single-frame shuttlecock detection from TrackNetV4 heatmap."""
    frame: int
    x_px: float
    y_px: float
    confidence: float
    x_meters: float = 0.0
    y_meters: float = 0.0
    speed_kmh: float = 0.0


@dataclass
class PlayerDetection:
    """Single-frame player bounding box from YOLOv8."""
    frame: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    confidence: float
    class_id: int  # 0 = player_near, 1 = player_far


@dataclass
class Detection:
    """Generic detection for ByteTrack input."""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    score: float
    class_id: int = 0


@dataclass
class Track:
    """Active track maintained by BYTETracker."""
    track_id: int
    bbox: np.ndarray          # [x1, y1, x2, y2]
    score: float
    state: str                # 'tracked' | 'lost' | 'removed'
    frame_id: int
    kalman_mean: Optional[np.ndarray] = None
    kalman_cov: Optional[np.ndarray] = None


@dataclass
class CourtKeypoints:
    """Four corners of the court detected in image pixel space.
    Order: top-left, top-right, bottom-right, bottom-left."""
    corners: list[tuple[float, float]]  # 4 points
    homography: Optional[np.ndarray] = None   # 3×3 matrix, set after cv2.findHomography


@dataclass
class ShotEvent:
    """A classified shot within a rally — output of ShotAnalyzer."""
    frame_idx: int
    x_px: float
    y_px: float
    x_meters: float = 0.0
    y_meters: float = 0.0
    speed_kmh: float = 0.0
    shot_type: str = "unknown"  # matches ShotType enum values on both sides


@dataclass
class RallyData:
    start_frame: int
    end_frame: int
    winner: str = "unknown"        # 'near' | 'far' | 'unknown'
    shuttle_detections: list[ShuttleDetection] = field(default_factory=list)
    player_tracks: list[Track] = field(default_factory=list)
    shot_events: list[ShotEvent] = field(default_factory=list)


@dataclass
class AnalysisResult:
    rallies: list[RallyData]
    fps: float
    total_frames: int
    court_homography: Optional[np.ndarray]   # 3×3, None if court detection failed
    processing_time_seconds: float = 0.0
    video: Optional["VideoMeta"] = None
    camera: Optional[dict] = None            # CameraModel.to_dict()
    track: Optional["Track2D"] = None
    diagnostics: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# v2 tracking contracts. Pixel coords are FULL-RESOLUTION pixels of the upright
# frame; world coords follow court_model.py (origin = court centre, +x → far
# baseline, +z up). Time is seconds from video start (use timestamps, not
# frame/fps — phone video is often variable frame rate).
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VideoMeta:
    fps: float                   # nominal
    width: int                   # upright frame size
    height: int
    n_frames: int
    timestamps: np.ndarray       # (n_frames,) seconds
    rotation_deg: int = 0


@dataclass
class ShuttleCandidate:
    x: float                     # px
    y: float                     # px
    score: float                 # detector confidence 0..1


@dataclass
class Track2D:
    """One shuttle position per frame after linking/cleaning."""
    t: np.ndarray                # (N,) seconds
    xy: np.ndarray               # (N,2) px; NaN where not visible and not filled
    conf: np.ndarray             # (N,) 0..1 (0 for filled frames)
    visible: np.ndarray          # (N,) bool — real detection used
    interpolated: np.ndarray     # (N,) bool — gap-filled
    # (N,2) px: the adopted detection itself (unsmoothed), NaN where none was adopted.
    # Kink/hit analysis should prefer it: smoothing can bend a path where the data doesn't.
    raw_xy: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        return len(self.t)


@dataclass
class PlayerObservation:
    frame: int
    track_id: int
    bbox: np.ndarray             # [x1,y1,x2,y2] px
    score: float
    side: str = "unknown"        # 'near' | 'far'
    foot_world: Optional[np.ndarray] = None   # (x,y) m on floor
    keypoints: Optional[np.ndarray] = None    # (17,3) COCO x,y,conf px


@dataclass
class HitEvent:
    frame: int
    t: float
    xy_px: tuple[float, float]
    hitter: str = "unknown"      # 'near' | 'far' | 'unknown'
    confidence: float = 0.0
    player_track_id: Optional[int] = None
    # how the contact was found: "kink" (a visible direction change, the normal case) or an
    # inferred kind such as "gap_contact" (first sighting after a hole in the clock).
    # Serialised only when it isn't "kink", so existing JSON / fixtures stay byte-identical.
    source: str = "kink"


@dataclass
class ShotSegment3D:
    """Flight between two consecutive hits (or hit → floor/net/out of view)."""
    start_frame: int
    end_frame: int
    t0: float                    # time of p0/v0
    p0: np.ndarray               # (3,) m
    v0: np.ndarray               # (3,) m/s
    drag_length: float
    rmse_px: float               # reprojection error of the fit
    frames: np.ndarray           # (M,) frame indices covered
    positions: np.ndarray        # (M,3) fitted positions at those frames
    landing_xy: Optional[tuple[float, float]] = None   # floor contact, if reached
    landing_t: Optional[float] = None
    net_crossing_height: Optional[float] = None        # z at x=0, if crossed
    confidence: float = 0.0
    net_crossing_y: Optional[float] = None             # y at x=0, if crossed (m)
    end_reason: str = "unknown"                        # 'hit' | 'floor' | 'net' | 'unknown'
    hitter_feet_xy: Optional[tuple[float, float]] = None   # hitter's feet on the floor (m) at contact

    @property
    def speed0_kmh(self) -> float:
        return float(np.linalg.norm(self.v0) * 3.6)


@dataclass
class ShotEventV2:
    hit: HitEvent
    segment: Optional[ShotSegment3D]
    shot_type: str = "unknown"   # see ShotType values (+ 'lift', 'push', 'kill', 'block')
    shot_type_confidence: float = 0.0
    hitter_pos_world: Optional[tuple[float, float]] = None
    hit_height_m: Optional[float] = None
    stroke_side: str = "unknown" # 'forehand' | 'backhand' | 'overhead' | 'unknown'


@dataclass
class RallyV2:
    start_frame: int
    end_frame: int
    shots: list[ShotEventV2] = field(default_factory=list)
    end_reason: str = "unknown"  # 'in' | 'out' | 'net' | 'unknown'
    landing_xy: Optional[tuple[float, float]] = None
    in_out_margin_m: Optional[float] = None
    winner: str = "unknown"      # 'near' | 'far' | 'unknown'
    confidence: float = 0.0
    server: str = "unknown"      # 'near' | 'far' | 'unknown'
    serve_missed: bool = False   # shuttle already in flight before the first detected hit
    segmentation_confidence: float = 0.0
    outcome_confidence: float = 0.0
    outcome_source: str = "unknown"   # how end_reason/winner were decided
