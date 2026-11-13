"""
Inline BYTE algorithm implementation for player tracking in badminton videos.

Reference: "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
ECCV 2022. We only need to track 2 players max, so the implementation is
deliberately lean — no C++ extension deps, pure NumPy.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from app.services.tracking.data_types import Detection, Track


# ──────────────────────────────────────────────
# Kalman filter for bounding box state
# ──────────────────────────────────────────────
# State: [cx, cy, w, h, vx, vy, vw, vh]  (center x/y, width, height, velocities)

_F = np.eye(8, dtype=np.float32)       # state transition
for i in range(4):
    _F[i, i + 4] = 1.0

_H = np.eye(4, 8, dtype=np.float32)    # observation matrix

_Q_scale = 1.0    # process noise
_R_scale = 1.0    # measurement noise

_Q = np.eye(8, dtype=np.float32) * _Q_scale
_Q[4:, 4:] *= 10.0  # higher noise on velocity

_R = np.eye(4, dtype=np.float32) * _R_scale


def _xyxy_to_cxcywh(bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=np.float32)


def _cxcywh_to_xyxy(state: np.ndarray) -> np.ndarray:
    cx, cy, w, h = state[:4]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


class TrackKalman:
    def __init__(self, bbox: np.ndarray):
        self.mean = np.zeros(8, dtype=np.float32)
        self.mean[:4] = _xyxy_to_cxcywh(bbox)
        self.cov = np.eye(8, dtype=np.float32) * 10.0

    def predict(self) -> np.ndarray:
        self.mean = _F @ self.mean
        self.cov = _F @ self.cov @ _F.T + _Q
        return _cxcywh_to_xyxy(self.mean)

    def update(self, bbox: np.ndarray):
        z = _xyxy_to_cxcywh(bbox)
        S = _H @ self.cov @ _H.T + _R
        K = self.cov @ _H.T @ np.linalg.inv(S)
        self.mean = self.mean + K @ (z - _H @ self.mean)
        self.cov = (np.eye(8) - K @ _H) @ self.cov

    @property
    def bbox(self) -> np.ndarray:
        return _cxcywh_to_xyxy(self.mean)


# ──────────────────────────────────────────────
# IoU helpers
# ──────────────────────────────────────────────

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _iou_matrix(tracks: list, detections: list) -> np.ndarray:
    mat = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, t in enumerate(tracks):
        for j, d in enumerate(detections):
            mat[i, j] = _iou(t.kalman.bbox, d.bbox)
    return mat


# ──────────────────────────────────────────────
# Greedy IoU matching (linear_sum_assignment replacement for small N)
# ──────────────────────────────────────────────

def _greedy_match(cost_matrix: np.ndarray, threshold: float):
    """Returns (matched_track_idx, matched_det_idx), unmatched_track_idx, unmatched_det_idx."""
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    matched, used_t, used_d = [], set(), set()
    # Sort by descending IoU
    pairs = sorted(
        [(cost_matrix[i, j], i, j) for i in range(cost_matrix.shape[0]) for j in range(cost_matrix.shape[1])],
        reverse=True,
    )
    for iou_val, ti, di in pairs:
        if iou_val < threshold:
            break
        if ti not in used_t and di not in used_d:
            matched.append((ti, di))
            used_t.add(ti); used_d.add(di)
    unmatched_t = [i for i in range(cost_matrix.shape[0]) if i not in used_t]
    unmatched_d = [j for j in range(cost_matrix.shape[1]) if j not in used_d]
    return matched, unmatched_t, unmatched_d


# ──────────────────────────────────────────────
# Internal track state
# ──────────────────────────────────────────────

@dataclass
class _InternalTrack:
    track_id: int
    kalman: TrackKalman
    score: float
    state: str = "tracked"   # tracked | lost | removed
    lost_count: int = 0
    frame_id: int = 0

    def to_track(self) -> Track:
        return Track(
            track_id=self.track_id,
            bbox=self.kalman.bbox.copy(),
            score=self.score,
            state=self.state,
            frame_id=self.frame_id,
        )


# ──────────────────────────────────────────────
# BYTETracker
# ──────────────────────────────────────────────

class BYTETracker:
    """
    Two-stage association:
      Stage 1: match high-confidence detections (score >= high_thresh) to tracked tracks
      Stage 2: match low-confidence detections to unmatched tracks
    Lost tracks survive for max_lost frames before removal.
    """

    def __init__(
        self,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        match_thresh_high: float = 0.5,
        match_thresh_low: float = 0.3,
        max_lost: int = 30,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh_high = match_thresh_high
        self.match_thresh_low = match_thresh_low
        self.max_lost = max_lost

        self._tracks: list[_InternalTrack] = []
        self._next_id = 1
        self._frame_id = 0

    def update(self, detections: list[Detection]) -> list[Track]:
        self._frame_id += 1

        # Split detections by confidence
        high_dets = [d for d in detections if d.score >= self.high_thresh]
        low_dets  = [d for d in detections if self.low_thresh <= d.score < self.high_thresh]

        # Predict all tracked tracks
        active = [t for t in self._tracks if t.state == "tracked"]
        lost   = [t for t in self._tracks if t.state == "lost"]
        for t in active + lost:
            t.kalman.predict()

        # ── Stage 1: match high-conf dets to active tracks ──────────────────
        matched1, unmatched_t1, unmatched_d_high = _greedy_match(
            _iou_matrix(active, high_dets), self.match_thresh_high
        )
        for ti, di in matched1:
            active[ti].kalman.update(high_dets[di].bbox)
            active[ti].score = high_dets[di].score
            active[ti].state = "tracked"
            active[ti].lost_count = 0
            active[ti].frame_id = self._frame_id

        # ── Stage 2: match low-conf dets to unmatched active tracks ─────────
        remaining_active = [active[i] for i in unmatched_t1]
        matched2, unmatched_t2, _ = _greedy_match(
            _iou_matrix(remaining_active, low_dets), self.match_thresh_low
        )
        for ti, di in matched2:
            remaining_active[ti].kalman.update(low_dets[di].bbox)
            remaining_active[ti].score = low_dets[di].score
            remaining_active[ti].state = "tracked"
            remaining_active[ti].lost_count = 0
            remaining_active[ti].frame_id = self._frame_id

        # ── Mark unmatched active tracks as lost ─────────────────────────────
        still_unmatched = [remaining_active[i] for i in unmatched_t2]
        for t in still_unmatched:
            t.state = "lost"
            t.lost_count += 1

        # ── Try to recover lost tracks with high-conf unmatched dets ─────────
        unmatched_high_dets = [high_dets[j] for j in unmatched_d_high]
        matched3, _, unmatched_d_new = _greedy_match(
            _iou_matrix(lost, unmatched_high_dets), self.match_thresh_high
        )
        for ti, di in matched3:
            lost[ti].kalman.update(unmatched_high_dets[di].bbox)
            lost[ti].state = "tracked"
            lost[ti].lost_count = 0
            lost[ti].frame_id = self._frame_id

        # ── Spawn new tracks from still-unmatched high-conf detections ────────
        for j in unmatched_d_new:
            det = unmatched_high_dets[j]
            new_t = _InternalTrack(
                track_id=self._next_id,
                kalman=TrackKalman(det.bbox),
                score=det.score,
                frame_id=self._frame_id,
            )
            self._tracks.append(new_t)
            self._next_id += 1

        # ── Remove tracks that have been lost too long ────────────────────────
        for t in self._tracks:
            if t.state == "lost" and t.lost_count > self.max_lost:
                t.state = "removed"
        self._tracks = [t for t in self._tracks if t.state != "removed"]

        return [t.to_track() for t in self._tracks if t.state == "tracked"]

    def reset(self):
        self._tracks.clear()
        self._next_id = 1
        self._frame_id = 0
