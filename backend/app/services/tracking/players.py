"""
Player tracking + pose (tracking v2).

    frames (streamed, every `step`)  ─ YOLOv8-pose ─▶ person candidates (bbox, 17 COCO kps)
      → foot pixel (ankle midpoint if both ankles confident, else bbox bottom-centre)
      → camera.pixel_to_floor → floor (x, y) m
      → gate + player-likeness cost: court ± 1.5 m, net-post zone (umpire / service
        judge), implausible apparent height, seated pose, a spot occupied ≥ 70 % of
        ±6 s while outside the lines (officials, spectators)
      → per court half, a global Viterbi over all sampled frames picks one candidate
        (or "missing") per frame; a player never changes half
      → doubles: the two paths of a half are pooled, cut into unambiguous tracklets
        (constant-velocity Kalman association) and re-stitched into 2 identities by
        an exact DP on junction likelihood + shirt/shorts colour, so partners keep
        their ids when they cross or rotate
      → interpolate skipped frames, RTS-smooth foot_world (constant-velocity Kalman).

Track ids: singles near=1, far=2. Doubles near 1 (starts on y<0) / 3, far 2 (y<0) / 4.
Without a camera the court is unknown: the two (four) largest, most central persons
are kept, near/far split by foot image-row, foot_world stays None.

Model: ultralytics YOLOv8-pose, weights AGPL-3.0 (commercial closed-source use needs an
Ultralytics licence). yolo11 needs ultralytics ≥ 8.3 (installed 8.2.103). Measured on
Apple M4 / MPS, batch 4, imgsz 1280, ~5 other jobs running (load ≈ 5.5), in fps:
s 8.1 at 720p, 8.0 at 1080p; m 4.3 / 4.2; l 2.2 / 1.6. Tracking on 3 real broadcast
clips is identical for s, m and l (same 2 ids, ≥ 99 % coverage, foot jitter within
0.5 cm). s keypoints stay within 1.2–1.8 % of box height of l (median) vs 0.8–1.5 %
for m, so s is the default; pass model_name="yolov8m-pose.pt" for slightly better
keypoints at half the speed.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from app.services.tracking import court_model as cm
from app.services.tracking import video_io
from app.services.tracking.camera import CameraModel
from app.services.tracking.data_types import PlayerObservation, VideoMeta

logger = logging.getLogger(__name__)

# ── configuration ─────────────────────────────────────────────────────────────

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ml", "weights")
DEFAULT_MODEL = "yolov8s-pose.pt"     # chosen by measurement, see module docstring
KP_CONF = 0.35                        # keypoint visibility threshold
L_ANK, R_ANK, L_HIP, R_HIP, L_KNEE, R_KNEE = 15, 16, 11, 12, 13, 14

COURT_MARGIN = 1.5                    # m outside the doubles lines still eligible
SIDE_TOL = 0.4                        # m a foot may stray over the net line
V_MAX = 7.5                           # m/s, player speed gate (sprint + foot noise)
LOOKBACK_S = 1.0                      # max gap bridged with a motion model
C_MISS = 2.5                          # cost per sampled frame without the player
C_RESTART = 6.0                       # re-acquire after a long gap (no motion prior)

FrameDets = tuple[np.ndarray, np.ndarray, np.ndarray]   # boxes (N,4), scores (N,), kps (N,17,3)
Detector = Callable[[list[np.ndarray], list[int]], list[FrameDets]]
Progress = Callable[[float], None]


@dataclass
class _Cands:
    """Person candidates of one sampled frame after gating."""
    frame: int
    boxes: np.ndarray        # (N,4)
    scores: np.ndarray       # (N,)
    kps: np.ndarray          # (N,17,3)
    foot_px: np.ndarray      # (N,2)
    pos: np.ndarray          # (N,2) floor metres (camera) or normalised px (no camera)
    unary: np.ndarray        # (N,) player-likeness cost (lower = more player-like)
    unit: np.ndarray         # (N,) metres per unit of `pos` (1 with camera)
    noise: np.ndarray        # (N,) expected foot-position noise, metres
    app: np.ndarray | None = None   # (N,APP_DIM) colour descriptor, None = no image

    @property
    def n(self) -> int:
        return len(self.scores)


# ── pose detector ─────────────────────────────────────────────────────────────

_MODEL_CACHE: dict[tuple[str, str], object] = {}


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:           # torch missing/broken → CPU
        pass
    return "cpu"


def load_pose_model(name: str = DEFAULT_MODEL, device: str | None = None):
    """Load (and cache) a YOLO pose model from ml/weights (auto-download if absent)."""
    dev = pick_device(device)
    key = (name, dev)
    if key not in _MODEL_CACHE:
        from ultralytics import YOLO
        path = os.path.join(WEIGHTS_DIR, name)
        _MODEL_CACHE[key] = YOLO(path if os.path.exists(path) else name)
    return _MODEL_CACHE[key]


def make_yolo_detector(model_name: str = DEFAULT_MODEL, device: str | None = None,
                       imgsz: int = 1280, conf: float = 0.15,
                       far_roi: tuple[int, int, int, int] | None = None,
                       far_imgsz: int | None = None) -> Detector:
    """Batch detector: list of BGR frames → [(boxes, scores, kps)] in full-res px.

    `far_roi` (x1,y1,x2,y2) adds a second pass on an upscaled crop of the far half
    (small players); crop detections replace full-frame ones they overlap.
    `far_imgsz` defaults to 2× the crop's long side (≤ 1280, multiple of 32)."""
    dev = pick_device(device)
    model = load_pose_model(model_name, dev)
    if far_roi is not None and far_imgsz is None:
        side = max(far_roi[2] - far_roi[0], far_roi[3] - far_roi[1])
        far_imgsz = int(min(1280, 32 * math.ceil(2 * side / 32)))

    def _run(imgs: list[np.ndarray], size: int) -> list[FrameDets]:
        res = model.predict(imgs, imgsz=size, device=dev, conf=conf, verbose=False,
                            half=False, max_det=30)
        out: list[FrameDets] = []
        for r in res:
            if r.boxes is None or len(r.boxes) == 0:
                out.append((np.zeros((0, 4)), np.zeros(0), np.zeros((0, 17, 3))))
                continue
            b = r.boxes.xyxy.cpu().numpy().astype(np.float64)
            s = r.boxes.conf.cpu().numpy().astype(np.float64)
            k = (r.keypoints.data.cpu().numpy().astype(np.float64)
                 if r.keypoints is not None else np.zeros((len(s), 17, 3)))
            out.append((b, s, k))
        return out

    def detect(frames: list[np.ndarray], frame_idx: list[int]) -> list[FrameDets]:
        full = _run(frames, imgsz)
        if far_roi is None:
            return full
        x1, y1, x2, y2 = far_roi
        crops = _run([f[y1:y2, x1:x2] for f in frames], far_imgsz)
        size = (frames[0].shape[1], frames[0].shape[0]) if frames else (0, 0)
        return [_merge_roi(a, c, far_roi, size) for a, c in zip(full, crops)]

    return detect


def _merge_roi(full: FrameDets, crop: FrameDets, roi: tuple[int, int, int, int],
               image_size: tuple[int, int], edge_px: float = 3.0) -> FrameDets:
    """Shift crop detections to frame px; they replace overlapping full-frame boxes
    (higher effective resolution → better keypoints for small far players).

    A crop detection touching an inner crop border (one that is not also the image
    border) is a person cut by the crop, e.g. a near player at the net whose head
    enters the far-half ROI: its box and foot point are wrong, so it is discarded
    and the full-frame detection kept."""
    cb, cs, ck = crop
    if len(cs) == 0:
        return full
    x1, y1, x2, y2 = roi
    W, H = image_size
    cb = cb + np.array([x1, y1, x1, y1], float)
    cut = np.zeros(len(cs), bool)
    if x1 > 0:
        cut |= cb[:, 0] <= x1 + edge_px
    if y1 > 0:
        cut |= cb[:, 1] <= y1 + edge_px
    if x2 < W:
        cut |= cb[:, 2] >= x2 - edge_px
    if y2 < H:
        cut |= cb[:, 3] >= y2 - edge_px
    if cut.all():
        return full
    cb, cs, ck = cb[~cut], cs[~cut], ck[~cut].copy()
    ck[..., 0] += x1
    ck[..., 1] += y1
    fb, fs, fk = full
    keep = np.ones(len(fs), bool)
    for i in range(len(fs)):
        if _iou_1n(fb[i], cb).max() > 0.4:
            keep[i] = False
    return (np.concatenate([fb[keep], cb]), np.concatenate([fs[keep], cs]),
            np.concatenate([fk[keep], ck]))


def _iou_1n(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(ua, 1e-9)


# ── geometry: foot point, floor position, player-likeness ─────────────────────

ANKLE_H = 0.08                        # m, ankle joint above the floor
PERSON_H = 1.8                        # m, scale prior when there is no camera


def foot_pixels(boxes: np.ndarray, kps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N,2) foot pixel and (N,) bool 'from ankles' per candidate.

    Both ankles confident → ankle midpoint (a point ~8 cm above the floor);
    else bbox bottom with u = mean of confident ankles/hips (robust to a raised arm)."""
    n = len(boxes)
    out = np.zeros((n, 2))
    ank = np.zeros(n, bool)
    for i in range(n):
        k = kps[i]
        if k[L_ANK, 2] >= KP_CONF and k[R_ANK, 2] >= KP_CONF:
            out[i] = (k[L_ANK, :2] + k[R_ANK, :2]) / 2
            ank[i] = True
            continue
        xs = [k[j, 0] for j in (L_ANK, R_ANK, L_HIP, R_HIP) if k[j, 2] >= KP_CONF]
        u = float(np.mean(xs)) if xs else (boxes[i, 0] + boxes[i, 2]) / 2
        out[i] = (min(max(u, boxes[i, 0]), boxes[i, 2]), boxes[i, 3])
    return out, ank


def floor_positions(camera: CameraModel, foot_px: np.ndarray,
                    from_ankle: np.ndarray) -> np.ndarray:
    """Foot pixels → (N,2) floor metres (ankle rays cut the z=ANKLE_H plane)."""
    pos = np.full((len(foot_px), 2), np.nan)
    for mask, z in ((from_ankle, ANKLE_H), (~from_ankle, 0.0)):
        if mask.any():
            pos[mask] = camera.pixel_to_floor(foot_px[mask], z=z)[:, :2]
    return pos


def outside_distance(xy: np.ndarray) -> np.ndarray:
    """Distance (m) outside the doubles court rectangle, 0 inside."""
    dx = np.clip(np.abs(xy[:, 0]) - cm.HALF_LENGTH, 0, None)
    dy = np.clip(np.abs(xy[:, 1]) - cm.DOUBLES_HALF_WIDTH, 0, None)
    return np.hypot(dx, dy)


def person_height_m(camera: CameraModel, xy: np.ndarray, top_v: np.ndarray) -> np.ndarray:
    """Apparent standing height (m) of a person whose feet are at floor `xy` and whose
    bbox top is at image row `top_v` (vertical line through the foot point)."""
    n = len(xy)
    p0 = camera.project(np.column_stack([xy, np.zeros(n)]))
    p2 = camera.project(np.column_stack([xy, np.full(n, 2.0)]))
    dv = p0[:, 1] - p2[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(dv > 1e-6, 2.0 * (p0[:, 1] - top_v) / dv, np.nan)


def metres_per_px(camera: CameraModel, foot_px: np.ndarray) -> np.ndarray:
    """Local floor scale at each foot pixel (geometric mean of du, dv steps)."""
    n = len(foot_px)
    q = camera.pixel_to_floor(np.concatenate([foot_px, foot_px + [1.0, 0.0], foot_px + [0.0, 1.0]]))
    a, bu, bv = q[:n, :2], q[n:2 * n, :2], q[2 * n:, :2]
    s = np.sqrt(np.linalg.norm(bu - a, axis=1) * np.linalg.norm(bv - a, axis=1))
    return np.where(np.isfinite(s), s, 0.05)


L_SH, R_SH = 5, 6
W_SIT = 2.0                           # cost of a fully seated pose (never a rally player)


def sitting_score(kps: np.ndarray) -> np.ndarray:
    """(N,) 0 = standing/lunging … 1 = seated, camera-independent.

    Ratio of the thigh's vertical image extent to the torso's: ≈0.9 standing, ≈0.3
    seated (thighs horizontal, viewed from an elevated camera). The more vertical
    thigh is used, so a lunge (rear thigh upright) never looks seated."""
    out = np.zeros(len(kps))
    for i, k in enumerate(kps):
        c = k[:, 2] >= KP_CONF
        if not (c[L_SH] and c[R_SH] and c[L_HIP] and c[R_HIP]):
            continue
        torso = (k[L_HIP, 1] + k[R_HIP, 1] - k[L_SH, 1] - k[R_SH, 1]) / 2
        r = [abs(k[kn, 1] - k[hp, 1]) / torso for hp, kn in ((L_HIP, L_KNEE), (R_HIP, R_KNEE))
             if c[kn]]
        if torso > 2.0 and r:
            out[i] = float(np.clip((0.5 - max(r)) / 0.25, 0.0, 1.0))
    return out


# ── appearance (colour histograms of shirt and shorts) ────────────────────────

APP_H, APP_S, APP_V = 8, 2, 3         # hue × saturation bins + achromatic value bins
APP_REGION = APP_H * APP_S + APP_V
APP_DIM = 2 * APP_REGION


def _region_hist(img: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    H, W = img.shape[:2]
    a, b = int(max(0, min(x1, x2))), int(min(W, max(x1, x2) + 1))
    c, d = int(max(0, min(y1, y2))), int(min(H, max(y1, y2) + 1))
    if b - a < 2 or d - c < 2:
        return np.zeros(APP_REGION, np.float32)
    import cv2
    hsv = cv2.cvtColor(np.ascontiguousarray(img[c:d, a:b]), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = hsv[:, 0].astype(int), hsv[:, 1].astype(int), hsv[:, 2].astype(int)
    chroma = (s >= 50) & (v >= 40)
    idx = np.where(chroma, (h * APP_H // 180) * APP_S + np.minimum(s // 150, APP_S - 1),
                   APP_H * APP_S + np.digitize(v, (80, 170)))
    hist = np.bincount(idx, minlength=APP_REGION).astype(np.float32)
    return hist / max(hist.sum(), 1.0)


def appearance_descriptors(img: np.ndarray, boxes: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """(N,APP_DIM) colour descriptor per person: shirt (shoulders→hips) and shorts
    (hips→knees) HSV histograms; bbox fractions when keypoints are unreliable."""
    out = np.zeros((len(boxes), APP_DIM), np.float32)
    for i, (bx, k) in enumerate(zip(boxes, kps)):
        x1, y1, x2, y2 = bx
        w, h = x2 - x1, y2 - y1
        c = k[:, 2] >= KP_CONF
        if c[[L_SH, R_SH, L_HIP, R_HIP]].all():
            xs = k[[L_SH, R_SH, L_HIP, R_HIP], 0]
            m, hw = xs.mean(), max(0.35 * (xs.max() - xs.min()), 0.08 * w)
            ys, yh = (k[L_SH, 1] + k[R_SH, 1]) / 2, (k[L_HIP, 1] + k[R_HIP, 1]) / 2
            top = (m - hw, ys + 0.15 * (yh - ys), m + hw, yh)
            yk = (k[L_KNEE, 1] + k[R_KNEE, 1]) / 2 if c[[L_KNEE, R_KNEE]].all() else yh + 0.25 * h
            bot = (m - hw, yh, m + hw, yh + 0.6 * (yk - yh))
        else:
            top = (x1 + 0.3 * w, y1 + 0.22 * h, x2 - 0.3 * w, y1 + 0.48 * h)
            bot = (x1 + 0.3 * w, y1 + 0.5 * h, x2 - 0.3 * w, y1 + 0.64 * h)
        out[i, :APP_REGION] = _region_hist(img, *top)
        out[i, APP_REGION:] = _region_hist(img, *bot)
    return out


def appearance_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Mean Hellinger distance of the two region histograms (0 same … 1 disjoint);
    0 when either descriptor is missing/empty (no evidence either way)."""
    if a is None or b is None or len(a) != APP_DIM or len(b) != APP_DIM:
        return 0.0
    d = 0.0
    for sl in (slice(0, APP_REGION), slice(APP_REGION, APP_DIM)):
        pa, pb = a[sl], b[sl]
        if pa.sum() < 0.5 or pb.sum() < 0.5:
            return 0.0
        d += math.sqrt(max(0.0, 1.0 - float(np.sum(np.sqrt(pa * pb)))))
    return d / 2


def unary_cost(xy: np.ndarray, scores: np.ndarray, height_m: np.ndarray | None,
               truncated: np.ndarray) -> np.ndarray:
    """Player-likeness cost (camera mode); +inf = rejected.

    Off-court distance dominates (spectators, line judges, people walking behind);
    the net-post zone rejects umpire and service judge; implausible apparent height
    catches mis-projected feet (people in front of the camera, stands)."""
    c = -np.log(np.clip(scores, 0.05, 1.0))
    d = outside_distance(xy)
    c = c + (d / 0.6) ** 2
    post = (np.abs(xy[:, 0]) < 1.2) & (np.abs(xy[:, 1]) > cm.DOUBLES_HALF_WIDTH + 0.25)
    c = c + 6.0 * post
    if height_m is not None:
        h = np.nan_to_num(height_m, nan=0.0)
        c = c + np.where(h < 0.9, ((0.9 - h) / 0.2) ** 2, 0.0)
        c = c + np.where(h > 2.9, ((h - 2.9) / 0.3) ** 2, 0.0)
    c = c + 2.0 * truncated
    c[~np.isfinite(xy).all(axis=1) | (d > COURT_MARGIN)] = np.inf
    return c


def build_candidates(frame: int, dets: FrameDets, camera: CameraModel | None,
                     image_size: tuple[int, int], image: np.ndarray | None = None) -> _Cands:
    """Gate raw detections of one frame and attach floor position + unary cost
    (+ colour descriptors of the survivors when the frame `image` is given)."""
    boxes, scores, kps = (np.asarray(dets[0], float).reshape(-1, 4),
                          np.asarray(dets[1], float).reshape(-1),
                          np.asarray(dets[2], float).reshape(-1, 17, 3))
    W, H = image_size
    h_px = boxes[:, 3] - boxes[:, 1]
    keep = (h_px >= max(12.0, 0.025 * H)) & (scores > 0.0)
    boxes, scores, kps, h_px = boxes[keep], scores[keep], kps[keep], h_px[keep]
    foot, from_ankle = foot_pixels(boxes, kps)
    truncated = boxes[:, 3] >= H - 2
    sit = W_SIT * sitting_score(kps)
    if camera is not None:
        pos = floor_positions(camera, foot, from_ankle)
        height = person_height_m(camera, pos, boxes[:, 1]) if len(pos) else np.zeros(0)
        unary = unary_cost(pos, scores, height, truncated) + sit
        noise = 0.06 + 4.0 * metres_per_px(camera, foot) if len(pos) else np.zeros(0)
        unit = np.ones(len(pos))
    else:
        pos = foot.copy()
        unit = PERSON_H / np.maximum(h_px, 1.0)
        noise = 0.1 + 4.0 * unit
        cu = np.abs(foot[:, 0] - W / 2) / (W / 2)
        unary = -np.log(np.clip(scores, 0.05, 1.0)) + 2.0 * cu ** 2 + sit
        unary = unary + np.where(foot[:, 1] < 0.2 * H, 3.0, 0.0) + 2.0 * truncated
    ok = np.isfinite(unary)
    app = appearance_descriptors(image, boxes[ok], kps[ok]) if image is not None else None
    return _Cands(frame, boxes[ok], scores[ok], kps[ok], foot[ok], pos[ok], unary[ok],
                  unit[ok], noise[ok], app)


# ── static people (seated officials, spectators) ──────────────────────────────

STATIC_WIN_S = 6.0                    # ± window over which a spot must stay occupied
STATIC_R_M = 0.3                      # "same spot" radius (m; px × unit without camera)
STATIC_MAX_REF = 24                   # reference frames sampled from the window
W_STATIC = 3.0                        # cost of a person that never moves (off court)
W_STATIC_NOCAM = 1.0                  # same without a camera (court unknown)


def static_fraction(frames: Sequence[_Cands], times: np.ndarray) -> list[np.ndarray]:
    """Per candidate: fraction of sampled frames within ±STATIC_WIN_S that have a
    detection on the same spot. ≈1 for umpire, line judges, seated spectators;
    a player who waits 3 s to serve scores ≈0.25 over the 12 s window."""
    K = len(frames)
    out: list[np.ndarray] = []
    lo = hi = 0
    for k in range(K):
        c = frames[k]
        while times[k] - times[lo] > STATIC_WIN_S:
            lo += 1
        while hi < K and times[hi] - times[k] <= STATIC_WIN_S:
            hi += 1
        refs = [r for r in range(lo, hi) if r != k]
        if c.n == 0 or len(refs) < 8:
            out.append(np.zeros(c.n))
            continue
        if len(refs) > STATIC_MAX_REF:
            refs = [refs[i] for i in np.linspace(0, len(refs) - 1, STATIC_MAX_REF).round().astype(int)]
        pts = [(i, frames[r].pos) for i, r in enumerate(refs) if frames[r].n]
        if not pts:
            out.append(np.zeros(c.n))
            continue
        P_ = np.concatenate([p for _, p in pts])
        F_ = np.concatenate([np.full(len(p), i) for i, p in pts])
        d = np.linalg.norm(c.pos[:, None, :] - P_[None], axis=2) * c.unit[:, None]
        rows, cols = np.nonzero(d < STATIC_R_M)
        hit = np.zeros((c.n, len(refs)), bool)
        hit[rows, F_[cols]] = True
        out.append(hit.mean(axis=1))
    return out


def add_static_penalty(frames: Sequence[_Cands], times: np.ndarray,
                       has_camera: bool) -> list[_Cands]:
    """Copies with unary += weight × ramp(static fraction, 0.7 → 0.95).

    A player may legitimately stand still for a long time (receiving stance,
    between points), but never outside the lines, where every official sits.
    With a camera the penalty therefore ramps in with the distance outside the
    court (0 → 0.3 m); without one there is no court, so a lighter weight is used
    everywhere (seated officials are caught by the sitting cue as well)."""
    out = []
    for c, s in zip(frames, static_fraction(frames, times)):
        ramp = np.clip((s - 0.7) / 0.25, 0.0, 1.0)
        if has_camera:
            w = W_STATIC * np.clip(outside_distance(c.pos) / 0.3, 0.0, 1.0) if c.n else 0.0
        else:
            w = W_STATIC_NOCAM
        out.append(replace(c, unary=c.unary + w * ramp))
    return out


# ── identity: global Viterbi per court half ───────────────────────────────────

def _trans_cost(a: _Cands, ia: np.ndarray, b: _Cands, ib: np.ndarray, dt: float) -> np.ndarray:
    """(len(ia), len(ib)) motion cost; inf where the implied speed is impossible."""
    pa, pb = a.pos[ia], b.pos[ib]
    ua, ub = a.unit[ia], b.unit[ib]
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2) * (ua[:, None] + ub[None, :]) / 2
    noise = a.noise[ia][:, None] + b.noise[ib][None, :]
    allow = V_MAX * dt + 2.0 * noise
    sig = 3.0 * dt + noise
    c = 0.5 * (d / sig) ** 2
    c[d > allow] = np.inf
    return c


def viterbi_path(frames: Sequence[_Cands], allowed: Sequence[np.ndarray], times: np.ndarray,
                 extra: Sequence[np.ndarray] | None = None) -> list[tuple[int, int]]:
    """Globally optimal single-target track through per-frame candidates.

    State per sampled frame k: one allowed candidate j or 'missing'. Cost = unary
    (+ `extra`) + motion (bridging gaps ≤ LOOKBACK_S with a speed gate) + C_MISS per
    missing frame; after a gap > LOOKBACK_S the track may re-acquire anywhere at
    C_RESTART. Returns [(k, j)] in time order (empty = never present)."""
    K = len(frames)
    if K == 0:
        return []
    best: list[np.ndarray] = []
    back: list[list[tuple[int, int] | None]] = []
    ends = np.full(K, np.inf)            # min cost of a path whose last detection is at k
    pref_min, pref_arg = np.inf, None    # min_k' (ends[k'] - C_MISS*k') for k' out of lookback
    p = 0                                # next k' to fold into the prefix minimum
    for k in range(K):
        idx = allowed[k]
        n = len(idx)
        cst = np.full(n, np.inf)
        bk: list[tuple[int, int] | None] = [None] * n
        if n:
            un = frames[k].unary[idx] + (extra[k][idx] if extra is not None else 0.0)
            cst = C_MISS * k + un                       # start here (all earlier missing)
            while p < k and times[k] - times[p] > LOOKBACK_S:
                v = ends[p] - C_MISS * p
                if v < pref_min:
                    pref_min, pref_arg = v, p
                p += 1
            if pref_arg is not None:                    # re-acquire after a long gap
                rc = pref_min + C_MISS * (k - 1) + C_RESTART + un
                jb = int(np.argmin(best[pref_arg]))
                better = rc < cst
                cst = np.where(better, rc, cst)
                for j in np.nonzero(better)[0]:
                    bk[j] = (pref_arg, jb)
            un_min = float(un.min())
            for kp in range(k - 1, p - 1, -1):          # motion-gated bridges, newest first
                ia = allowed[kp]
                if len(ia) == 0 or not np.isfinite(ends[kp]):
                    continue
                if ends[kp] + C_MISS * (k - kp - 1) + un_min >= cst.max():
                    continue                            # cannot improve any state
                tc = _trans_cost(frames[kp], ia, frames[k], idx, float(times[k] - times[kp]))
                tot = best[kp][:, None] + tc + C_MISS * (k - kp - 1)
                arg = np.argmin(tot, axis=0)
                val = tot[arg, np.arange(n)] + un
                better = val < cst
                cst = np.where(better, val, cst)
                for j in np.nonzero(better)[0]:
                    bk[j] = (kp, int(arg[j]))
            ends[k] = float(cst.min())
        best.append(cst)
        back.append(bk)
    final = ends + C_MISS * (K - 1 - np.arange(K))
    k = int(np.argmin(final))
    if not np.isfinite(final[k]) or final[k] >= C_MISS * K:
        return []
    j = int(np.argmin(best[k]))
    path: list[tuple[int, int]] = []
    node: tuple[int, int] | None = (k, j)
    while node is not None:
        kk, jj = node
        path.append((kk, int(allowed[kk][jj])))
        node = back[kk][jj]
    return path[::-1]


# ── smoothing ─────────────────────────────────────────────────────────────────

ACC_SIGMA = 5.0                       # m/s², white-acceleration process noise
MAX_INTERP_S = 0.5                    # longer gaps split a track into segments


def rts_smooth(t: np.ndarray, z: np.ndarray, r: np.ndarray,
               acc_sigma: float = ACC_SIGMA) -> np.ndarray:
    """Rauch–Tung–Striebel smoother, constant-velocity model, per axis.

    t (N,) seconds; z (N,D) measurements with NaN rows = no measurement (the
    smoother interpolates there); r (N,) measurement std (m). Returns (N,D)."""
    n, dim = z.shape
    q = acc_sigma ** 2
    xs = np.zeros((n, 2, dim))            # filtered [pos, vel] per axis
    Ps = np.zeros((n, 2, 2))              # covariance (shared by the axes)
    xp_all = np.zeros((n, 2, dim))
    Pp_all = np.zeros((n, 2, 2))
    first = int(np.argmax(np.isfinite(z[:, 0])))
    x = np.vstack([z[first], np.zeros(dim)])
    P = np.diag([r[first] ** 2 + 1e-4, 4.0])
    for i in range(n):
        dt = float(t[i] - t[i - 1]) if i else 0.0
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        x = F @ x
        P = F @ P @ F.T + Q
        xp_all[i], Pp_all[i] = x, P
        if np.isfinite(z[i, 0]):
            S = P[0, 0] + r[i] ** 2
            Kg = P[:, 0] / S
            x = x + np.outer(Kg, z[i] - x[0])
            P = P - np.outer(Kg, P[0, :])
        xs[i], Ps[i] = x, P
    out = xs.copy()
    for i in range(n - 2, -1, -1):
        dt = float(t[i + 1] - t[i])
        F = np.array([[1.0, dt], [0.0, 1.0]])
        pp = Pp_all[i + 1]
        det = pp[0, 0] * pp[1, 1] - pp[0, 1] * pp[1, 0]
        inv = np.array([[pp[1, 1], -pp[0, 1]], [-pp[1, 0], pp[0, 0]]]) / det
        C = Ps[i] @ F.T @ inv
        out[i] = xs[i] + C @ (out[i + 1] - xp_all[i + 1])
    return out[:, 0, :]


# ── main entry ────────────────────────────────────────────────────────────────

class PlayerTrackList(list):
    """list[PlayerObservation] that also carries timing (for movement_stats)."""
    timestamps: np.ndarray | None = None
    fps: float = 30.0
    doubles: bool = False
    diagnostics: dict | None = None


def _far_roi(camera: CameraModel, size: tuple[int, int]) -> tuple[tuple[int, int, int, int], float] | None:
    """Image rect covering the far half (+margin, up to 2.6 m) and the apparent
    height (px) of a 1.75 m player on the far baseline."""
    W, H = size
    xs, ys = (-0.3, cm.HALF_LENGTH + COURT_MARGIN), (-cm.DOUBLES_HALF_WIDTH - COURT_MARGIN,
                                                    cm.DOUBLES_HALF_WIDTH + COURT_MARGIN)
    pts = np.array([[x, y, z] for x in xs for y in ys for z in (0.0, 2.6)])
    uv = camera.project(pts)
    if not np.isfinite(uv).all():
        return None
    x1, y1 = np.clip(uv.min(0), 0, [W, H]).astype(int)
    x2, y2 = np.clip(uv.max(0), 0, [W, H]).astype(int)
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    base = camera.project(np.array([[cm.HALF_LENGTH, 0, 0], [cm.HALF_LENGTH, 0, 1.75]]))
    return (x1, y1, x2, y2), float(abs(base[0, 1] - base[1, 1]))


def track_players(path: str, meta: VideoMeta, camera: CameraModel | None, step: int = 2,
                  device: str | None = None, doubles: bool | None = None,
                  progress: Progress | None = None, *, detector: Detector | None = None,
                  model_name: str = DEFAULT_MODEL, imgsz: int = 1280,
                  far_crop: bool | None = None, batch: int = 8) -> list[PlayerObservation]:
    """Detect, filter, identify and smooth the on-court players of a video.

    Returns one PlayerObservation per frame per player while the player is tracked
    (skipped frames interpolated, gaps ≤ 0.5 s bridged), sorted by (frame, track_id).
    The returned list is a PlayerTrackList carrying `timestamps` for movement_stats.
    `detector` (frames, frame_idx) → [(boxes, scores, kps)] overrides YOLO (tests)."""
    step = max(1, int(step))
    size = (meta.width, meta.height)
    if detector is None:
        roi = None
        if camera is not None and far_crop is not False:
            fr = _far_roi(camera, size)
            if fr is not None and (far_crop or fr[1] < 0.11 * meta.height):
                roi = fr[0]
        detector = make_yolo_detector(model_name, device, imgsz=imgsz, far_roi=roi)
    n_total = meta.n_frames or 0
    frames: list[_Cands] = []
    buf: list[np.ndarray] = []
    buf_idx: list[int] = []

    def flush() -> None:
        if not buf:
            return
        for fi, img, d in zip(buf_idx, buf, detector(buf, buf_idx)):
            frames.append(build_candidates(fi, d, camera, size, image=img))
        buf.clear()
        buf_idx.clear()
        if progress is not None and n_total:
            progress(min(1.0, (frames[-1].frame + 1) / n_total))

    for fi, img in video_io.iter_frames(path, step=step):
        buf.append(img)
        buf_idx.append(fi)
        if len(buf) >= batch:
            flush()
    flush()
    return assemble_tracks(frames, meta, camera, doubles)


def _time_fn(meta: VideoMeta) -> Callable[[int], float]:
    ts = np.asarray(meta.timestamps if meta.timestamps is not None else [], float)
    fps = float(meta.fps or 30.0)

    def t_of(i: int) -> float:
        if i < len(ts):
            return float(ts[i])
        return (float(ts[-1]) + (i - len(ts) + 1) / fps) if len(ts) else i / fps
    return t_of


def detect_doubles(frames: Sequence[_Cands], camera: CameraModel | None) -> bool:
    """Doubles if ≥ 2 confident on-court persons per half (≥ 4 without camera)
    in more than 30 % of sampled frames."""
    if not frames:
        return False
    hits = 0
    for c in frames:
        good = c.unary < 1.2
        if camera is not None:
            good &= outside_distance(c.pos) < 0.3
            hits += int(np.sum(good & (c.pos[:, 0] < 0)) >= 2 and np.sum(good & (c.pos[:, 0] > 0)) >= 2)
        else:
            hits += int(np.sum(good) >= 4)
    return hits / len(frames) > 0.3


def _exclusion(c: _Cands, j: int) -> np.ndarray:
    """Candidates that are the chosen one or a duplicate detection of it."""
    iou = _iou_1n(c.boxes[j], c.boxes)
    d = np.linalg.norm(c.pos - c.pos[j], axis=1) * c.unit
    return (iou > 0.5) | (d < 0.35)


def _extract_paths(frames: Sequence[_Cands], times: np.ndarray, base_allowed: list[np.ndarray],
                   extra: list[np.ndarray] | None, n_paths: int,
                   taken: list[np.ndarray]) -> list[dict[int, int]]:
    """Sequentially extract up to n_paths disjoint Viterbi paths (k → j)."""
    out: list[dict[int, int]] = []
    for _ in range(n_paths):
        allowed = [a[~taken[k][a]] for k, a in enumerate(base_allowed)]
        path = dict(viterbi_path(frames, allowed, times, extra))
        if not path:
            break
        out.append(path)
        for k, j in path.items():
            taken[k] |= _exclusion(frames[k], j)
    return out


def _median_y_start(frames: Sequence[_Cands], times: np.ndarray, path: dict[int, int],
                    camera: CameraModel | None) -> float:
    """Median lateral position over the first 2 s of a path, as world y
    (without a camera: −u, since image-right is world −y for a baseline camera)."""
    ks = sorted(path)
    t0 = times[ks[0]]
    vals = [(frames[k].pos[path[k], 1] if camera is not None else -frames[k].foot_px[path[k], 0])
            for k in ks if times[k] - t0 <= 2.0]
    return float(np.median(vals))


def assemble_tracks(frames: list[_Cands], meta: VideoMeta, camera: CameraModel | None,
                    doubles: bool | None = None) -> PlayerTrackList:
    """Identity assignment + densify + smoothing on already-detected candidates."""
    t_of = _time_fn(meta)
    out = PlayerTrackList()
    out.fps = float(meta.fps or 30.0)
    out.timestamps = np.asarray(meta.timestamps, float) if meta.timestamps is not None else None
    if not frames:
        return out
    frames = sorted(frames, key=lambda c: c.frame)
    times = np.array([t_of(c.frame) for c in frames])
    frames = add_static_penalty(frames, times, camera is not None)
    dbl = detect_doubles(frames, camera) if doubles is None else bool(doubles)
    per_side = 2 if dbl else 1
    out.doubles = dbl
    K = len(frames)
    taken = [np.zeros(c.n, bool) for c in frames]
    slots: list[tuple[str, dict[int, int]]] = []
    if camera is not None:
        for side, sgn in (("near", -1.0), ("far", 1.0)):
            allowed = [np.nonzero(sgn * c.pos[:, 0] > -SIDE_TOL)[0] for c in frames]
            extra = [3.0 * np.clip(-sgn * c.pos[:, 0] / SIDE_TOL + 1.0, 0, 1) *
                     (sgn * c.pos[:, 0] < 0) for c in frames]
            side_taken = [t.copy() for t in taken]
            for p in _extract_paths(frames, times, allowed, extra, per_side, side_taken):
                slots.append((side, p))
        _resolve_cross_side(frames, slots)
    else:
        allowed = [np.arange(c.n) for c in frames]
        paths = _extract_paths(frames, times, allowed, None, 2 * per_side, taken)
        v_med = [float(np.median([frames[k].foot_px[j, 1] for k, j in p.items()])) for p in paths]
        order = np.argsort(v_med)[::-1]                 # largest image row = nearest
        for rank, pi in enumerate(order):
            slots.append(("near" if rank < max(1, len(paths) // 2) else "far", paths[pi]))
    if per_side == 2:                                   # partner identity: motion + appearance
        split: list[tuple[str, dict[int, int]]] = []
        for side in ("near", "far"):
            sp = [p for s, p in slots if s == side and p]
            split += [(side, p) for p in (split_partners(frames, times, sp) if len(sp) == 2 else sp)]
        slots = split
    slots = [(s, p) for s, p in slots if p]
    # ids: near 1 (/3), far 2 (/4); doubles: lower id = smaller world y at the start
    ids: list[int] = []
    for side in ("near", "far"):
        sp = [(i, p) for i, (s, p) in enumerate(slots) if s == side]
        sp.sort(key=lambda ip: _median_y_start(frames, times, ip[1], camera))
        base = 1 if side == "near" else 2
        for rank, (i, _) in enumerate(sp):
            ids.append((i, base + 2 * rank))
    diag = {"doubles": dbl, "sampled_frames": K,
            "coverage": {}}
    for i, tid in sorted(ids, key=lambda x: x[1]):
        side, path = slots[i]
        obs = _densify(frames, times, path, tid, side, camera, meta, t_of)
        diag["coverage"][tid] = round(len(path) / K, 3)
        out.extend(obs)
    out.sort(key=lambda o: (o.frame, o.track_id))
    out.diagnostics = diag
    return out


def _resolve_cross_side(frames: Sequence[_Cands], slots: list[tuple[str, dict[int, int]]]) -> None:
    """A candidate claimed by paths of both halves stays with the half its foot is in."""
    for k in range(len(frames)):
        claims: dict[int, list[int]] = {}
        for si, (_, p) in enumerate(slots):
            if k in p:
                claims.setdefault(p[k], []).append(si)
        for j, sis in claims.items():
            if len(sis) < 2:
                continue
            true_side = cm.side_of(float(frames[k].pos[j, 0]))
            keep = [si for si in sis if slots[si][0] == true_side][:1] or sis[:1]
            for si in sis:
                if si not in keep:
                    del slots[si][1][k]


# ── doubles: partner identity by motion + appearance continuity ───────────────
#
# The per-half Viterbi paths reliably say WHICH detections are the two players,
# but greedy path extraction is free to hop onto the partner whenever that avoids
# a missed frame. So identity is solved separately on the pooled detections:
#   1. tracklets: frame-to-frame association on constant-velocity Kalman predictions;
#      a tracklet is cut wherever the alternative (swapped) association is almost
#      as likely (partners close / merged into one box / long gap);
#   2. stitching: an exact DP labels tracklets with 2 identities (same-identity
#      tracklets may not overlap in time) minimising the Gaussian NLL of each
#      junction (forward + reversed-time Kalman predictions) + shirt/shorts colour
#      distance.

Q_LINK = 225.0                        # m²/s³ white-acceleration density (≈15 m/s² over 1 s)
V0_LINK = 0.01                        # m², reference variance of the Gaussian NLL costs
KF_TAIL = 16                          # samples filtered at a tracklet end
AMBIG = 2.0                           # cost margin below which an association is ambiguous
C_NEW = 4.0                           # start a tracklet instead of extending one …
C_COAST = 2.0                         # … while that tracklet goes unobserved
TRACKLET_GAP_S = 1.0                  # tracklets coast at most this long
W_APP = 8.0                           # cost per unit of appearance distance
C_DROP = 3.0                          # per sample, discard a tracklet fitting neither identity
STITCH_BEAM = 400                     # DP states kept per tracklet (exact below this)


@dataclass
class _KF:
    """Constant-velocity Kalman state at one end of a tracklet (metres; isotropic)."""
    x: np.ndarray                     # (2,) position
    v: np.ndarray                     # (2,) velocity (points backwards in time at a start)
    p00: float
    p01: float
    p11: float
    t: float
    unit: float

    def mean(self, g: float) -> np.ndarray:
        return self.x + self.v * g

    def var(self, g: float) -> float:
        return self.p00 + 2 * g * self.p01 + g * g * self.p11 + Q_LINK * g ** 3 / 3


def _kf_end(frames: Sequence[_Cands], times: np.ndarray, pts: Sequence[tuple[int, int]],
            at_end: bool) -> _KF:
    """Filter the last (at_end) or, in reversed time, the first KF_TAIL samples.

    A causal filter keeps the end estimate close to the latest measurements even
    through a sharp change of direction, where a window line-fit is biased."""
    seq = list(pts[-KF_TAIL:]) if at_end else list(pts[:KF_TAIL][::-1])
    unit = float(np.mean([frames[k].unit[j] for k, j in seq]))
    k0, j0 = seq[0]
    x = frames[k0].pos[j0] * unit
    v = np.zeros(2)
    p00, p01, p11 = float(frames[k0].noise[j0]) ** 2, 0.0, 9.0
    tp = float(times[k0])
    for k, j in seq[1:]:
        g = abs(float(times[k]) - tp)
        tp = float(times[k])
        x = x + v * g
        p00, p01, p11 = (p00 + 2 * g * p01 + g * g * p11 + Q_LINK * g ** 3 / 3,
                         p01 + g * p11 + Q_LINK * g * g / 2, p11 + Q_LINK * g)
        s_ = p00 + float(frames[k].noise[j]) ** 2
        k0_, k1_ = p00 / s_, p01 / s_
        inn = frames[k].pos[j] * unit - x
        x, v = x + k0_ * inn, v + k1_ * inn
        p00, p01, p11 = p00 - k0_ * p00, p01 - k0_ * p01, p11 - k1_ * p01
    return _KF(x, v, p00, p01, p11, tp, unit)


def _nll(e2: float, var: float) -> float:
    """Isotropic 2-D Gaussian negative log-likelihood (constants dropped)."""
    return 0.5 * e2 / var + math.log(var / V0_LINK)


def _assoc_cost(st: _KF, c: _Cands, j: int, tk: float) -> float:
    """Cost of detection j (frame time tk) continuing a tracklet with end state st."""
    g = tk - st.t
    z = c.pos[j] * c.unit[j]
    r = float(c.noise[j])
    if float(np.linalg.norm(z - st.x)) > V_MAX * g + 2.0 * (math.sqrt(st.p00) + r):
        return np.inf
    return _nll(float(np.sum((z - st.mean(g)) ** 2)), st.var(g) + r * r)


def _matchings(n_t: int, n_d: int) -> list[tuple[tuple[int, int], ...]]:
    """All partial injective matchings {(track, det)} between n_t tracks and n_d dets."""
    out: list[tuple[tuple[int, int], ...]] = []

    def rec(d: int, used: frozenset, acc: tuple) -> None:
        if d == n_d:
            out.append(acc)
            return
        rec(d + 1, used, acc)
        for t in range(n_t):
            if t not in used:
                rec(d + 1, used | {t}, acc + ((t, d),))
    rec(0, frozenset(), ())
    return out


def build_tracklets(frames: Sequence[_Cands], times: np.ndarray,
                    pooled: Sequence[Sequence[int]]) -> list[list[tuple[int, int]]]:
    """Cut the pooled detections of one half into unambiguous tracklets [(k, j)]."""
    done: list[list[tuple[int, int]]] = []
    active: list[list[tuple[int, int]]] = []
    for k, js in enumerate(pooled):
        still = []
        for tr in active:
            (done if times[k] - times[tr[-1][0]] > TRACKLET_GAP_S else still).append(tr)
        active = still
        if not js:
            continue
        states = [_kf_end(frames, times, tr, True) for tr in active]
        cost = np.array([[_assoc_cost(st, frames[k], j, float(times[k])) for j in js]
                         for st in states]).reshape(len(active), len(js))
        scored = []
        for m in _matchings(len(active), len(js)):
            c = sum(cost[t, d] for t, d in m)
            c += C_NEW * (len(js) - len(m)) + C_COAST * (len(active) - len(m))
            if np.isfinite(c):
                scored.append((c, m))
        scored.sort(key=lambda x: x[0])
        best_c, best = scored[0]
        pairs = dict(best)                               # track → det
        # identity conflict: a near-equal alternative pairs a track or det differently
        unsure_t: set[int] = set()
        for c, m in scored[1:]:
            if c - best_c >= AMBIG:
                break
            for t, d in m:
                if pairs.get(t, d) != d or any(t2 != t and d2 == d for t2, d2 in best):
                    unsure_t.add(t)
                    unsure_t.update(t2 for t2, d2 in best if d2 == d)
        new_d = [d for d in range(len(js)) if d not in pairs.values()]
        for t in sorted(unsure_t):
            if t in pairs:
                new_d.append(pairs.pop(t))
        for t, d in pairs.items():
            active[t].append((k, js[d]))
        if new_d:           # keep same-identity tracklets disjoint in time: close the rest
            keep = [tr for t, tr in enumerate(active) if t in pairs]
            done.extend(tr for t, tr in enumerate(active) if t not in pairs)
            active = keep + [[(k, js[d])] for d in sorted(new_d)]
    done.extend(active)
    return sorted(done, key=lambda tr: (tr[0][0], tr[-1][0]))


def _mean_app(frames: Sequence[_Cands], pts: Sequence[tuple[int, int]]) -> np.ndarray | None:
    a = [frames[k].app[j] for k, j in pts if frames[k].app is not None and frames[k].app.shape[1]]
    return np.mean(a, axis=0) if a else None


def _link_cost(frames: Sequence[_Cands], times: np.ndarray, a: list[tuple[int, int]],
               b: list[tuple[int, int]]) -> float:
    """Cost of tracklet b continuing tracklet a (a ends before b starts): mean of
    the forward (a's end filter → b's start) and backward (b's reversed-time start
    filter → a's end) Gaussian NLLs, + speed gate + appearance distance."""
    sa, sb = _kf_end(frames, times, a, True), _kf_end(frames, times, b, False)
    g = max(sb.t - sa.t, 1e-3)
    c = 0.5 * (_nll(float(np.sum((sa.mean(g) - sb.x) ** 2)), sa.var(g) + sb.p00)
               + _nll(float(np.sum((sb.mean(g) - sa.x) ** 2)), sb.var(g) + sa.p00))
    if float(np.linalg.norm(sb.x - sa.x)) > V_MAX * g + 2.0 * (math.sqrt(sa.p00) + math.sqrt(sb.p00)):
        c += 50.0
    return c + W_APP * appearance_distance(_mean_app(frames, a[-10:]), _mean_app(frames, b[:10]))


def stitch_tracklets(frames: Sequence[_Cands], times: np.ndarray,
                     trk: Sequence[list[tuple[int, int]]]) -> list[list[int]]:
    """Exact DP: label tracklets (sorted by start) with ≤ 2 identities.

    State = the last tracklet of each identity (unordered, −1 = identity unused);
    a tracklet joins an identity whose last tracklet ended before it starts, or is
    dropped at C_DROP per sample. Returns tracklet indices per identity."""
    n = len(trk)
    start = [tr[0][0] for tr in trk]
    end = [tr[-1][0] for tr in trk]
    cache: dict[tuple[int, int], float] = {}

    def link(a: int, b: int) -> float:
        if a < 0:
            return 0.0
        if (a, b) not in cache:
            cache[(a, b)] = _link_cost(frames, times, trk[a], trk[b])
        return cache[(a, b)]

    layer: dict[tuple[int, int], float] = {(-1, -1): 0.0}
    back: list[dict[tuple[int, int], tuple[tuple[int, int], int | None]]] = []
    for c in range(n):
        nxt: dict[tuple[int, int], float] = {}
        bk: dict[tuple[int, int], tuple[tuple[int, int], int | None]] = {}

        def push(key: tuple[int, int], val: float, prev: tuple[int, int], act: int | None) -> None:
            key = tuple(sorted(key))                     # identities are interchangeable
            if val < nxt.get(key, np.inf):
                nxt[key], bk[key] = val, (prev, act)
        for (l1, l2), v in layer.items():
            for last, other in ((l1, l2), (l2, l1)):
                if last < 0 or end[last] < start[c]:
                    push((c, other), v + link(last, c), (l1, l2), last)
                if l1 == l2:                             # symmetric state: one branch suffices
                    break
            push((l1, l2), v + C_DROP * len(trk[c]), (l1, l2), None)
        if len(nxt) > STITCH_BEAM:                       # beam: O(n·B) on long matches
            keep = sorted(nxt, key=nxt.get)[:STITCH_BEAM]
            nxt = {q: nxt[q] for q in keep}
            bk = {q: bk[q] for q in keep}
        layer = nxt
        back.append(bk)
    key = min(layer, key=layer.get)
    acts: list[int | None] = [None] * n
    for c in range(n - 1, -1, -1):
        prev, act = back[c][key]
        acts[c] = -2 if act is None else act             # −2 = dropped, −1 = new identity
        key = prev
    ids: list[list[int]] = []
    owner: dict[int, int] = {}
    for c, act in enumerate(acts):
        if act == -2:
            continue
        if act == -1:
            ids.append([c])
            owner[c] = len(ids) - 1
        else:
            owner[c] = owner[act]
            ids[owner[c]].append(c)
    return ids


def split_partners(frames: Sequence[_Cands], times: np.ndarray,
                   paths: Sequence[dict[int, int]]) -> list[dict[int, int]]:
    """Re-assign the pooled detections of two same-half paths to two identities."""
    pooled: list[list[int]] = [[] for _ in frames]
    for p in paths:
        for k, j in p.items():
            if j not in pooled[k]:
                pooled[k].append(j)
    trk = build_tracklets(frames, times, pooled)
    if not trk:
        return []
    out = [{k: j for i in ident for k, j in trk[i]} for ident in stitch_tracklets(frames, times, trk)]
    return sorted(out, key=len, reverse=True)


def _densify(frames: Sequence[_Cands], times: np.ndarray, path: dict[int, int], tid: int,
             side: str, camera: CameraModel | None, meta: VideoMeta,
             t_of: Callable[[int], float]) -> list[PlayerObservation]:
    """Per-frame observations for one identity: interpolate bbox/keypoints between
    samples, RTS-smooth foot_world; gaps > MAX_INTERP_S split the track."""
    ks = sorted(path)
    segs: list[list[int]] = [[ks[0]]] if ks else []
    for a, b in zip(ks, ks[1:]):
        if times[b] - times[a] > MAX_INTERP_S:
            segs.append([b])
        else:
            segs[-1].append(b)
    res: list[PlayerObservation] = []
    for seg in segs:
        F = np.array([frames[k].frame for k in seg])
        B = np.array([frames[k].boxes[path[k]] for k in seg])
        S = np.array([frames[k].scores[path[k]] for k in seg])
        KP = np.array([frames[k].kps[path[k]] for k in seg])
        D = np.arange(F[0], F[-1] + 1)
        if len(F) > 1:
            i = np.clip(np.searchsorted(F, D, side="right") - 1, 0, len(F) - 2)
            w = ((D - F[i]) / (F[i + 1] - F[i]))[:, None]
            Bd = B[i] * (1 - w) + B[i + 1] * w
            Sd = np.where(w[:, 0] == 0, S[i], np.minimum(S[i], S[i + 1]))
            KPd = KP[i] * (1 - w[:, :, None]) + KP[i + 1] * w[:, :, None]
            KPd[..., 2] = np.where(w == 0, KP[i][..., 2], np.minimum(KP[i][..., 2], KP[i + 1][..., 2]))
        else:
            Bd, Sd, KPd = B, S, KP
        foot = None
        if camera is not None:
            td = np.array([t_of(int(f)) for f in D])
            z = np.full((len(D), 2), np.nan)
            r = np.full(len(D), 0.2)
            loc = F - F[0]
            z[loc] = [frames[k].pos[path[k]] for k in seg]
            r[loc] = [frames[k].noise[path[k]] for k in seg]
            foot = rts_smooth(td, z, r)
        for n, f in enumerate(D):
            res.append(PlayerObservation(
                frame=int(f), track_id=tid, bbox=Bd[n].astype(np.float64), score=float(Sd[n]),
                side=side, foot_world=None if foot is None else foot[n].copy(),
                keypoints=KPd[n].astype(np.float32)))
    return res


# ── movement statistics ───────────────────────────────────────────────────────

HEAT_ROWS, HEAT_COLS = 12, 10         # depth from net × court width
SPEED_WIN_S = 0.3
SPEED_CAP = 9.0                       # m/s; anything faster is a tracking artefact


def movement_stats(obs: Sequence[PlayerObservation], side: str,
                   timestamps: np.ndarray | None = None, fps: float | None = None) -> dict:
    """Footwork stats for the player(s) on `side` ('near' | 'far').

    distance_m (10 Hz resampled path length), avg_speed_ms (distance / tracked time),
    max_speed_ms (95th pct of 0.3 s-window speed, capped), heatmap (HEAT_ROWS ×
    HEAT_COLS fraction of tracked time; row 0 = at the net, last row = baseline;
    column 0 = world y −3.05 m, i.e. the near player's right), court_coverage
    (fraction of half-court cells occupied ≥ 0.2 s), tracked_s, and `players`
    with the same numbers per track id (doubles)."""
    ts = timestamps if timestamps is not None else getattr(obs, "timestamps", None)
    fps = fps or getattr(obs, "fps", None) or 30.0

    def t_of(f: int) -> float:
        if ts is not None and f < len(ts):
            return float(ts[f])
        return f / fps

    groups: dict[int, list[PlayerObservation]] = {}
    for o in obs:
        if o.side == side and o.foot_world is not None and np.all(np.isfinite(o.foot_world[:2])):
            groups.setdefault(o.track_id, []).append(o)
    per = {}
    for tid, g in sorted(groups.items()):
        g.sort(key=lambda o: o.frame)
        t = np.array([t_of(o.frame) for o in g])
        xy = np.array([o.foot_world[:2] for o in g], float)
        per[tid] = _stats_one(t, xy)
    if not per:
        return {"distance_m": 0.0, "avg_speed_ms": 0.0, "max_speed_ms": 0.0,
                "heatmap": np.zeros((HEAT_ROWS, HEAT_COLS)).tolist(), "court_coverage": 0.0,
                "tracked_s": 0.0, "players": {}}
    tot_t = sum(p["tracked_s"] for p in per.values())
    heat = sum(np.asarray(p["heatmap"]) * p["tracked_s"] for p in per.values())
    heat = heat / max(tot_t, 1e-9)
    dwell = sum(np.asarray(p["_dwell"]) for p in per.values())
    dist = sum(p["distance_m"] for p in per.values())
    out = {"distance_m": round(dist, 2),
           "avg_speed_ms": round(dist / tot_t, 3) if tot_t > 0 else 0.0,
           "max_speed_ms": max(p["max_speed_ms"] for p in per.values()),
           "heatmap": np.round(heat, 5).tolist(),
           "court_coverage": round(float(np.mean(dwell >= 0.2)), 3),
           "tracked_s": round(tot_t, 2)}
    for p in per.values():
        p.pop("_dwell")
    out["players"] = per
    return out


def _stats_one(t: np.ndarray, xy: np.ndarray) -> dict:
    # split into continuous runs (gap > MAX_INTERP_S → new run)
    cuts = np.nonzero(np.diff(t) > MAX_INTERP_S)[0] + 1
    dist, tracked, speeds = 0.0, 0.0, []
    dwell = np.zeros((HEAT_ROWS, HEAT_COLS))
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(t)]):
        tt, pp = t[a:b], xy[a:b]
        if len(tt) < 2:
            continue
        dt = np.diff(tt)
        tracked += float(tt[-1] - tt[0])
        # path length at ~10 Hz (suppresses residual jitter, keeps footwork)
        grid = np.arange(tt[0], tt[-1] + 1e-9, 0.1)
        rs = np.column_stack([np.interp(grid, tt, pp[:, i]) for i in range(2)])
        dist += float(np.linalg.norm(np.diff(rs, axis=0), axis=1).sum())
        # windowed speed
        h = SPEED_WIN_S / 2
        lo = np.interp(tt - h, tt, np.arange(len(tt)))
        hi = np.interp(tt + h, tt, np.arange(len(tt)))
        for i in range(len(tt)):
            ia, ib = int(np.floor(lo[i])), int(np.ceil(hi[i]))
            if tt[ib] - tt[ia] >= 0.5 * SPEED_WIN_S:
                v = np.linalg.norm(pp[ib] - pp[ia]) / (tt[ib] - tt[ia])
                if v <= SPEED_CAP:
                    speeds.append(v)
        # dwell time per half-court cell
        w = np.r_[dt, dt[-1]] if len(dt) else np.zeros(1)
        r_i = np.clip((np.abs(pp[:, 0]) / cm.HALF_LENGTH * HEAT_ROWS).astype(int), 0, HEAT_ROWS - 1)
        c_i = np.clip(((pp[:, 1] + cm.DOUBLES_HALF_WIDTH) / cm.DOUBLES_WIDTH * HEAT_COLS).astype(int),
                      0, HEAT_COLS - 1)
        np.add.at(dwell, (r_i, c_i), w)
    heat = dwell / dwell.sum() if dwell.sum() > 0 else dwell
    return {"distance_m": round(dist, 2),
            "avg_speed_ms": round(dist / tracked, 3) if tracked > 0 else 0.0,
            "max_speed_ms": round(float(np.percentile(speeds, 95)), 3) if speeds else 0.0,
            "heatmap": np.round(heat, 5).tolist(),
            "court_coverage": round(float(np.mean(dwell >= 0.2)), 3),
            "tracked_s": round(tracked, 2), "_dwell": dwell}
