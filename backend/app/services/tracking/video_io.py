"""Streaming video access. Never hold a whole match in RAM.

Frames are delivered UPRIGHT (the container's rotation tag is applied), exactly like
the phone's ShuttlCore VideoSource: `probe` reports the upright width/height and every
reader returns upright pixels. `meta.timestamps` is the REAL presentation clock
(seconds from the first frame); only duplicate / reversed times are re-timed, so a
hole in the clock (paused / edited / dropped-frame file) stays visible to the stages
via `pts_gaps`.
"""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np

from app.services.tracking.data_types import VideoMeta

# ── PTS gaps ────────────────────────────────────────────────────────────────
# A frame step dt[i] = t[i] - t[i-1] is a GAP when
#       dt[i] > max(PTS_GAP_MIN_S, PTS_GAP_FACTOR * median positive step).
# 0.30 s = the linker's longest bridged miss burst (LinkerConfig.max_link_gap_s):
# a shorter hole is indistinguishable from missed detections and every stage
# already handles it; a longer one is time the camera never saw. The relative
# term only matters below 1 / (0.30 / 4) = 13.3 fps. Measured on real footage
# (29 TrackNetV2 Test clips, 3 phone 1080p60 files, a real iPhone .mov,
# a 59.6 fps VFR screen recording): max step / median ≤ 2.02 and max step ≤
# 34 ms — far from 0.30 s. The synthetic VFR corpus clip (steps up to 0.267 s)
# is not a gap either. Same rule, same constants in ShuttlCore.ptsGaps (Swift).
PTS_GAP_MIN_S = 0.30
PTS_GAP_FACTOR = 4.0


def pts_gaps(timestamps) -> np.ndarray:
    """First frame index after every presentation-time gap (sorted, int64; the frame
    BEFORE the gap is index - 1). Same convention as linker scene_cuts. Needs ≥ 2
    finite positive steps to know the normal step; non-finite times never form a gap."""
    t = np.asarray(timestamps if timestamps is not None else [], np.float64).ravel()
    if len(t) < 3:
        return np.zeros(0, np.int64)
    d = np.diff(t)
    pos = d[np.isfinite(d) & (d > 0)]
    if len(pos) < 2:
        return np.zeros(0, np.int64)
    thr = max(PTS_GAP_MIN_S, PTS_GAP_FACTOR * float(np.median(pos)))
    return (np.flatnonzero(np.isfinite(d) & (d > thr)) + 1).astype(np.int64)


def pts_gap_info(timestamps) -> list[dict]:
    """JSON-safe gap list for diagnostics["video"]["pts_gaps"] (identical keys on the
    phone): frame = first frame after the gap, t_before / t_after = times (s) of the
    frames either side, dt_s = t_after - t_before."""
    t = np.asarray(timestamps if timestamps is not None else [], np.float64).ravel()
    return [{"frame": int(i), "t_before": float(t[i - 1]), "t_after": float(t[i]),
             "dt_s": float(t[i] - t[i - 1])} for i in pts_gaps(t)]


def retime_duplicates(ts: np.ndarray, step0: float) -> np.ndarray:
    """Duplicate PTS = repeated frames. A reversed or stalled clock repeats the previous
    time (running max); each run of equal times keeps its first frame's time and spreads
    its repeats evenly up to the next distinct time. A trailing run (or a whole-clip run)
    has no next time and advances by the clip's median positive step (else `step0`).
    Strictly increasing afterwards (a step below the float resolution becomes one ulp)."""
    t = np.maximum.accumulate(ts)
    n = len(t)
    d = np.diff(t)
    pos = d[d > 0]
    step_end = float(np.median(pos)) if len(pos) else step0
    out = t.copy()
    starts = np.flatnonzero(np.r_[True, d > 0])            # first frame of each run
    ends = np.r_[starts[1:], n]                           # one past its last frame
    for a, b in zip(starts, ends):
        if b - a < 2:
            continue
        step = (t[b] - t[a]) / (b - a) if b < n else step_end
        out[a + 1:b] = t[a] + np.arange(1, b - a) * step
    bad = np.flatnonzero(np.diff(out) <= 0)
    if len(bad):
        for i in range(int(bad[0]) + 1, n):
            if not out[i] > out[i - 1]:
                out[i] = np.nextafter(out[i - 1], np.inf)
    return out


# ── rotation ────────────────────────────────────────────────────────────────
_ROT_CODE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _rotation_deg(cap: cv2.VideoCapture) -> int:
    """Clockwise quarter turns (0/90/180/270) that make the stored frame upright —
    the same convention as ShuttlCore's rotationDegrees(preferredTransform)."""
    r = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    r = int(round(r)) if r and np.isfinite(r) else 0
    return ((r % 360) + 360) % 360 // 90 * 90


class _Capture:
    """cv2.VideoCapture that always delivers upright frames. OpenCV's own
    auto-rotation (CAP_PROP_ORIENTATION_AUTO, default 0 in the 4.11 FFmpeg build)
    is switched OFF and the rotation is applied here explicitly, so the result does
    not depend on the build's default or on whether it honours the flag."""

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        self.rot = 0
        if self.cap.isOpened():
            self.rot = _rotation_deg(self.cap)
            self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
            self._auto = bool(self.cap.get(cv2.CAP_PROP_ORIENTATION_AUTO))
            self._stored: tuple[int, int] | None = None

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def get(self, prop):
        return self.cap.get(prop)

    def set(self, prop, value):
        return self.cap.set(prop, value)

    def grab(self) -> bool:
        return self.cap.grab()

    def _upright(self, ok: bool, f):
        if not ok or f is None or not self.rot:
            return ok, f
        if self._auto:          # the build refused to switch auto-rotation off
            return ok, f
        return ok, cv2.rotate(f, _ROT_CODE[self.rot])

    def retrieve(self):
        return self._upright(*self.cap.retrieve())

    def read(self):
        return self._upright(*self.cap.read())

    def release(self):
        self.cap.release()


def open_capture(path: str) -> _Capture:
    """The one way to open a video for pixels (upright frames)."""
    return _Capture(path)


# ── public API ──────────────────────────────────────────────────────────────

def probe(path: str) -> VideoMeta:
    """Read metadata and per-frame timestamps (one decode pass, frames discarded)."""
    cap = open_capture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rot = cap.rot
    ts: list[float] = []
    w = h = 0
    while cap.grab():
        ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        if not w:
            ok, f = cap.retrieve()
            if ok:
                h, w = f.shape[:2]      # upright (rotation applied)
    cap.release()
    t = np.asarray(ts, dtype=np.float64)
    fin_fps = float(fps) if np.isfinite(fps) and fps > 0 else 30.0
    if len(t) and np.all(np.isfinite(t)):
        t = t - t[0]                    # seconds from the first frame (like the phone)
    elif len(t):
        t = np.arange(len(t)) / fin_fps
    # Keep the REAL clock (a hole in it is information: pts_gaps). Duplicate / reversed
    # times (broken muxers; containers reporting 0 for every POS_MSEC) are re-timed
    # evenly between their distinct neighbours — an all-zero clock becomes i/fps.
    if len(t) > 1 and not np.all(np.diff(t) > 0):
        t = retime_duplicates(t, 1.0 / fin_fps)
    return VideoMeta(fps=float(fps), width=w, height=h, n_frames=len(t),
                     timestamps=t, rotation_deg=rot)


def iter_frames(path: str, start: int = 0, stop: int | None = None,
                step: int = 1, rgb: bool = False) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, frame) — BGR by default, upright."""
    cap = open_capture(path)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    i = start
    try:
        while stop is None or i < stop:
            if (i - start) % step:
                if not cap.grab():
                    break
            else:
                ok, f = cap.read()
                if not ok:
                    break
                yield i, (cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if rgb else f)
            i += 1
    finally:
        cap.release()


# Below this spacing, decoding straight through beats keyframe seeks.
_SEEK_MIN_GAP = 120


def sample_frames(path: str, n: int, n_total: int | None = None) -> list[tuple[int, np.ndarray]]:
    """n evenly spaced upright BGR frames (for calibration / background estimation)."""
    cap = open_capture(path)
    try:
        total = n_total or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = sorted(set(np.linspace(0, max(total - 1, 0),
                                     num=min(n, max(total, 1))).astype(int).tolist()))
        if not idx:
            return []
        if len(idx) < 2 or (idx[-1] - idx[0]) / (len(idx) - 1) < _SEEK_MIN_GAP:
            want = set(idx)
            return [(i, f) for i, f in iter_frames(path, stop=idx[-1] + 1) if i in want]
        out = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if ok:
                out.append((i, f))
        return out
    finally:
        cap.release()
