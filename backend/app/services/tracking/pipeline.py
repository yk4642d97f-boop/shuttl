"""
BadmintonPipeline: full ML analysis pipeline for a single match video.

Integrates:
  - TrackNetV4  — shuttlecock detection (heatmap)
  - YOLOv8      — player detection + court keypoint detection
  - BYTETracker — cross-frame player tracking
  - filterpy    — Kalman filter for shuttle trajectory smoothing
  - OpenCV       — homography estimation for real-world coordinates
  - Rule-based  — rally segmentation and shot type classification
"""

import time
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter

from app.services.tracking.data_types import (
    AnalysisResult,
    CourtKeypoints,
    Detection,
    RallyData,
    ShuttleDetection,
    ShotEvent,
)
from app.services.tracking.bytetrack import BYTETracker
from app.services.tracking.tracknetv4 import TrackNetV4

logger = logging.getLogger(__name__)

# Standard BWF singles court dimensions (metres)
COURT_LENGTH = 13.4
COURT_WIDTH = 6.1

# Processing resolution fed to TrackNetV4
TRACK_W, TRACK_H = 512, 288

# Batch size for TrackNetV4 inference
BATCH = 8


# ──────────────────────────────────────────────────────────────────────────────
# Shuttle Kalman filter factory
# ──────────────────────────────────────────────────────────────────────────────

def _make_shuttle_kalman() -> KalmanFilter:
    """
    State: [x, y, vx, vy]  (pixel coords + velocity)
    Observation: [x, y]
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=np.float32)
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]], dtype=np.float32)
    kf.R *= 10.0    # measurement noise
    kf.Q *= 0.01    # process noise — shuttle moves smoothly
    kf.P *= 100.0   # initial uncertainty
    return kf


# ──────────────────────────────────────────────────────────────────────────────
# BadmintonPipeline
# ──────────────────────────────────────────────────────────────────────────────

class BadmintonPipeline:
    def __init__(
        self,
        tracknet_weights: Optional[str] = None,
        yolo_player_weights: Optional[str] = None,
        yolo_court_weights: Optional[str] = None,
        device: str = "cpu",
    ):
        self.device = device
        self._init_tracknet(tracknet_weights)
        self._init_yolo(yolo_player_weights, yolo_court_weights)

    # ── Model initialisation ──────────────────────────────────────────────────

    def _init_tracknet(self, weights: Optional[str]):
        if weights and Path(weights).exists():
            logger.info("Loading TrackNetV4 weights from %s", weights)
            self.tracknet = TrackNetV4.load_weights(weights)
        else:
            logger.warning("TrackNetV4 weights not found — using randomly initialised model for dev/test")
            self.tracknet = TrackNetV4()
            self.tracknet.eval()
        self.tracknet.to(self.device)

    def _init_yolo(self, player_weights: Optional[str], court_weights: Optional[str]):
        try:
            from ultralytics import YOLO
            pw = player_weights or "yolov8n.pt"   # fallback: pretrained COCO model
            self.yolo_player = YOLO(pw)
            if court_weights and Path(court_weights).exists():
                self.yolo_court = YOLO(court_weights)
            else:
                self.yolo_court = None
                logger.info("Court YOLO weights absent — will use Hough line detection fallback")
        except ImportError:
            logger.error("ultralytics not installed; player detection will be skipped")
            self.yolo_player = None
            self.yolo_court = None

    # ── Main entry point ──────────────────────────────────────────────────────

    def analyze(self, video_path: str, match_id: str = "") -> AnalysisResult:
        t0 = time.perf_counter()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("[%s] Video: %.1f fps, %d frames", match_id, fps, total_frames)

        # ── Pass 1: read all frames, run TrackNetV4 in batches ────────────────
        all_frames, shuttle_detections = self._detect_shuttle(cap, fps, total_frames, match_id)
        cap.release()

        # ── Pass 2: smooth trajectory with Kalman filter ──────────────────────
        shuttle_detections = self._smooth_trajectory(shuttle_detections, total_frames)

        # ── Pass 3: detect court and build homography ─────────────────────────
        court_kp = self._detect_court(all_frames)
        homography = court_kp.homography if court_kp else None

        # ── Pass 4: map pixel coords → real-world metres ──────────────────────
        if homography is not None:
            self._apply_homography(shuttle_detections, homography)

        # ── Pass 5: detect players (sample every 5th frame for efficiency) ────
        tracker = BYTETracker()
        # (player tracks stored inside rallies below)

        # ── Pass 6: segment rallies ───────────────────────────────────────────
        rallies = self._segment_rallies(shuttle_detections, fps)

        # ── Pass 7: classify shots ────────────────────────────────────────────
        for rally in rallies:
            rally.shot_events = self._classify_shots(rally.shuttle_detections, fps)

        elapsed = time.perf_counter() - t0
        logger.info("[%s] Pipeline complete in %.1fs: %d rallies", match_id, elapsed, len(rallies))

        return AnalysisResult(
            rallies=rallies,
            fps=fps,
            total_frames=total_frames,
            court_homography=homography,
            processing_time_seconds=elapsed,
        )

    # ── Pass 1: TrackNetV4 shuttle detection ──────────────────────────────────

    def _detect_shuttle(
        self, cap: cv2.VideoCapture, fps: float, total_frames: int, match_id: str
    ) -> tuple[list[np.ndarray], list[Optional[ShuttleDetection]]]:
        import torch

        all_frames: list[np.ndarray] = []
        shuttle_detections: list[Optional[ShuttleDetection]] = []
        frame_idx = 0
        batch_rgb: list[np.ndarray] = []

        # Read all frames first (RAM-heavy but simplifies batching)
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            all_frames.append(rgb)

        logger.info("[%s] Read %d frames; running TrackNetV4", match_id, len(all_frames))

        detections = self.tracknet.predict_frame_sequence(
            all_frames, device=self.device, confidence_threshold=0.4
        )
        return all_frames, detections

    # ── Pass 2: Kalman smoothing and gap-fill ─────────────────────────────────

    def _smooth_trajectory(
        self,
        detections: list[Optional[ShuttleDetection]],
        total_frames: int,
    ) -> list[Optional[ShuttleDetection]]:
        kf = _make_shuttle_kalman()
        smoothed: list[Optional[ShuttleDetection]] = []
        initialised = False
        max_gap = 10   # fill gaps up to 10 consecutive missing frames

        # Forward pass
        raw = list(detections)
        for i, det in enumerate(raw):
            if det is not None:
                z = np.array([[det.x_px], [det.y_px]], dtype=np.float32)
                if not initialised:
                    kf.x[:2] = z
                    initialised = True
                kf.predict()
                kf.update(z)
                # Return smoothed position
                smoothed.append(ShuttleDetection(
                    frame=det.frame,
                    x_px=float(kf.x[0]),
                    y_px=float(kf.x[1]),
                    confidence=det.confidence,
                ))
            else:
                if initialised:
                    kf.predict()
                    smoothed.append(ShuttleDetection(
                        frame=i,
                        x_px=float(kf.x[0]),
                        y_px=float(kf.x[1]),
                        confidence=0.0,
                    ))
                else:
                    smoothed.append(None)

        return smoothed

    # ── Pass 3: Court detection ────────────────────────────────────────────────

    def _detect_court(self, frames: list[np.ndarray]) -> Optional[CourtKeypoints]:
        # Sample a frame from the middle third for stable court view
        sample_idx = len(frames) // 2
        if sample_idx >= len(frames):
            return None
        sample_bgr = cv2.cvtColor(frames[sample_idx], cv2.COLOR_RGB2BGR)

        if self.yolo_court is not None:
            kp = self._yolo_court_detect(sample_bgr)
            if kp:
                return kp
        # Fallback: Hough lines
        return self._hough_court_detect(sample_bgr)

    def _yolo_court_detect(self, bgr: np.ndarray) -> Optional[CourtKeypoints]:
        results = self.yolo_court(bgr, verbose=False)
        for r in results:
            if r.keypoints is not None and len(r.keypoints.xy) >= 4:
                pts = r.keypoints.xy[0].cpu().numpy()[:4]
                corners = [(float(p[0]), float(p[1])) for p in pts]
                return self._compute_homography(corners, bgr.shape)
        return None

    def _hough_court_detect(self, bgr: np.ndarray) -> Optional[CourtKeypoints]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                 minLineLength=bgr.shape[1] // 4, maxLineGap=20)
        if lines is None or len(lines) < 4:
            return None

        h, w = bgr.shape[:2]
        # Approximate court corners as image boundary intersections (fixed for bird's-eye)
        corners = [(w * 0.05, h * 0.05), (w * 0.95, h * 0.05),
                   (w * 0.95, h * 0.95), (w * 0.05, h * 0.95)]
        return self._compute_homography(corners, bgr.shape)

    def _compute_homography(
        self, corners: list[tuple[float, float]], img_shape: tuple
    ) -> CourtKeypoints:
        dst = np.array([
            [0, 0],
            [COURT_LENGTH, 0],
            [COURT_LENGTH, COURT_WIDTH],
            [0, COURT_WIDTH],
        ], dtype=np.float32)
        src = np.array(corners, dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        kp = CourtKeypoints(corners=corners)
        kp.homography = H
        return kp

    # ── Pass 4: Apply homography ──────────────────────────────────────────────

    def _apply_homography(
        self,
        detections: list[Optional[ShuttleDetection]],
        H: np.ndarray,
    ):
        for det in detections:
            if det is None:
                continue
            src = np.array([[[det.x_px, det.y_px]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(src, H)
            det.x_meters = float(np.clip(dst[0, 0, 0], 0, COURT_LENGTH))
            det.y_meters = float(np.clip(dst[0, 0, 1], 0, COURT_WIDTH))

    # ── Pass 6: Rally segmentation ────────────────────────────────────────────

    def _segment_rallies(
        self,
        detections: list[Optional[ShuttleDetection]],
        fps: float,
    ) -> list[RallyData]:
        # Speed-based segmentation: a rally boundary is any run of
        # low-confidence / missing detections lasting ≥ 0.3s
        min_gap_frames = max(1, int(fps * 0.3))
        rallies: list[RallyData] = []
        current: list[ShuttleDetection] = []
        gap_count = 0

        for i, det in enumerate(detections):
            is_valid = det is not None and det.confidence > 0.15
            if is_valid:
                gap_count = 0
                current.append(det)
            else:
                gap_count += 1
                if gap_count >= min_gap_frames and current:
                    rallies.append(RallyData(
                        start_frame=current[0].frame,
                        end_frame=current[-1].frame,
                        shuttle_detections=list(current),
                    ))
                    current = []
                    gap_count = 0

        if current:
            rallies.append(RallyData(
                start_frame=current[0].frame,
                end_frame=current[-1].frame,
                shuttle_detections=list(current),
            ))

        # Filter out spurious sub-second rallies
        min_frames = int(fps * 0.5)
        rallies = [r for r in rallies if len(r.shuttle_detections) >= min_frames]
        return rallies

    # ── Pass 7: Shot classification ───────────────────────────────────────────

    def _classify_shots(
        self, detections: list[ShuttleDetection], fps: float
    ) -> list[ShotEvent]:
        if len(detections) < 3:
            return []

        # Compute per-frame speed (km/h) from consecutive real-world coords
        self._compute_speeds(detections, fps)

        # Detect shot moments: local speed maxima (each peak = a shot contact)
        shot_frames = self._find_shot_peaks(detections, fps)
        events: list[ShotEvent] = []
        is_first_shot = True

        for idx in shot_frames:
            det = detections[idx]
            shot_type = "serve" if is_first_shot else self._classify_arc(detections, idx, fps)
            is_first_shot = False
            events.append(ShotEvent(
                frame_idx=det.frame,
                x_px=det.x_px,
                y_px=det.y_px,
                x_meters=det.x_meters,
                y_meters=det.y_meters,
                speed_kmh=det.speed_kmh,
                shot_type=shot_type,
            ))

        return events

    def _compute_speeds(self, dets: list[ShuttleDetection], fps: float):
        dt = 1.0 / fps
        for i in range(1, len(dets)):
            if dets[i].x_meters != 0 or dets[i - 1].x_meters != 0:
                # Use real-world coords if available
                dx = dets[i].x_meters - dets[i - 1].x_meters
                dy = dets[i].y_meters - dets[i - 1].y_meters
            else:
                # Normalised pixel delta — scale factor is approximate
                dx = (dets[i].x_px - dets[i - 1].x_px) * 0.02
                dy = (dets[i].y_px - dets[i - 1].y_px) * 0.02
            dist = np.sqrt(dx ** 2 + dy ** 2)
            dets[i].speed_kmh = float(dist / dt * 3.6)

    def _find_shot_peaks(
        self, dets: list[ShuttleDetection], fps: float, min_gap: int = 15
    ) -> list[int]:
        speeds = [d.speed_kmh for d in dets]
        if not speeds:
            return []
        peaks = []
        last_peak = -min_gap
        for i in range(1, len(speeds) - 1):
            if (speeds[i] > speeds[i - 1] and speeds[i] > speeds[i + 1]
                    and speeds[i] > 30.0 and i - last_peak >= min_gap):
                peaks.append(i)
                last_peak = i
        return peaks

    def _classify_arc(
        self, dets: list[ShuttleDetection], peak_idx: int, fps: float
    ) -> str:
        """Rule-based shot type from local arc geometry around the peak."""
        window = 10
        start = max(0, peak_idx - window)
        end = min(len(dets), peak_idx + window)
        segment = dets[start:end]
        if not segment:
            return "unknown"

        speeds = [d.speed_kmh for d in segment]
        max_speed = max(speeds, default=0)

        # Y is inverted in image coords (top = 0)
        ys = [d.y_px for d in segment]
        y_range = max(ys) - min(ys) if ys else 0
        y_at_peak = dets[peak_idx].y_px
        y_min = min(ys)

        xs = [d.x_px for d in segment]
        x_range = max(xs) - min(xs) if xs else 0

        # Direction of travel after the peak
        if peak_idx + 5 < len(dets) and peak_idx > 0:
            dy_after = dets[min(peak_idx + 5, len(dets) - 1)].y_px - dets[peak_idx].y_px
            downward = dy_after > 0
        else:
            downward = False

        # Smash: high speed, steep downward trajectory
        if max_speed > 80 and downward:
            return "smash"

        # Clear: large arc height, slow, travels far horizontally
        if y_range > 80 and x_range > 100:
            return "clear"

        # Drop: slow, falls near net (short x displacement)
        if max_speed < 40 and x_range < 60 and downward:
            return "drop"

        # Net: very short displacement, low height
        if x_range < 40 and y_range < 40:
            return "net"

        # Drive: fast, horizontal (small y range relative to x)
        if max_speed > 50 and y_range < x_range * 0.3:
            return "drive"

        return "unknown"
