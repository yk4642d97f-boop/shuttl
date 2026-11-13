"""Cloud HitNet adapter; the current learned model is validated for singles only."""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
import torch

from app.services.tracking import hitnet_core as core
from app.services.tracking.data_types import HitEvent, PlayerObservation, Track2D, VideoMeta
from app.services.tracking.video_io import pts_gaps


def player_index(players: list[PlayerObservation] | None):
    index = {}
    multiple = False
    for observation in players or []:
        if observation.side not in ("far", "near"):
            continue
        key = (observation.frame, observation.side)
        previous = index.get(key)
        if previous is not None and previous.track_id != observation.track_id:
            multiple = True
        if previous is None or observation.track_id < previous.track_id:
            index[key] = observation
    return index, multiple


def raw_matrix(track: Track2D, meta: VideoMeta, players: list[PlayerObservation] | None) -> np.ndarray:
    if meta.width <= 0 or meta.height <= 0:
        raise ValueError("Positive upright video dimensions are required")
    n = track.n
    if (np.shape(track.xy) != (n, 2) or
            any(np.shape(a) != (n,) for a in (track.conf, track.visible, track.interpolated))):
        raise ValueError("Track arrays must have one aligned entry per frame")
    raw = np.full((n, 16), np.nan, np.float32)
    xy = np.asarray(track.xy, np.float64)
    visible = np.asarray(track.visible, bool) & np.isfinite(xy).all(1)
    raw[visible, :2] = xy[visible] / [meta.width, meta.height]
    raw[visible, 2] = np.asarray(track.conf)[visible]
    raw[:, 3] = track.interpolated
    index, _ = player_index(players)
    for (frame, side), observation in index.items():
        if not 0 <= frame < n:
            continue
        player = 0 if side == "far" else 1
        box = np.asarray(observation.bbox, np.float64)
        if box.shape == (4,) and np.isfinite(box).all():
            raw[frame, 4 + player * 4:8 + player * 4] = [
                (box[0] + box[2]) / 2 / meta.width, (box[1] + box[3]) / 2 / meta.height,
                (box[2] - box[0]) / meta.width, (box[3] - box[1]) / meta.height]
        if observation.foot_world is not None:
            foot = np.asarray(observation.foot_world, np.float64)
            if foot.shape == (2,) and np.isfinite(foot).all():
                raw[frame, 12 + player * 2:14 + player * 2] = foot
    return raw


class HitNet:
    def __init__(self, weights: str | Path, device: str = "cpu"):
        self.path = Path(weights).resolve()
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        # Only load the portable export, not an arbitrary Python training checkpoint.
        checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
        if checkpoint.get("arch") != "tcn" or checkpoint.get("n_in") != 51:
            raise ValueError("Unsupported HitNet architecture or feature contract")
        self.decoder = checkpoint.get("decoder", {})
        if self.decoder != {"thr": .4, "gap_s": .3, "lam": 0.0}:
            raise ValueError("Unexpected HitNet decoder contract")
        self.device = torch.device(device)
        self.net = core.HitTCN(51, int(checkpoint["hidden"]))
        self.net.load_state_dict(checkpoint["state"], strict=True)
        self.net.to(self.device).eval()

    @torch.inference_mode()
    def logits(self, raw: np.ndarray, fps: float) -> np.ndarray:
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("FPS must be positive and finite")
        if raw.ndim != 2 or raw.shape[1] != 16:
            raise ValueError("HitNet expects (frames,16) raw features")
        n = len(raw)
        if n == 0:
            return np.empty((0, 2), np.float32)
        x = torch.from_numpy(core.derive(raw, fps))
        window, stride = 384, 192
        if n <= window:
            result = self.net(x[None].to(self.device))[0].float().cpu().numpy()
            if not np.isfinite(result).all():
                raise ValueError("Non-finite HitNet model output")
            return result
        starts = list(range(0, n - window + 1, stride))
        if starts[-1] != n - window:
            starts.append(n - window)
        result = np.empty((n, 2), np.float32)
        margin = (window - stride) // 2
        for i, start in enumerate(starts):
            logits = self.net(x[start:start + window][None].to(self.device))[0].float().cpu().numpy()
            lo = 0 if i == 0 else margin
            hi = window if i == len(starts) - 1 else window - margin
            result[start + lo:start + hi] = logits[lo:hi]
        if not np.isfinite(result).all():
            raise ValueError("Non-finite HitNet model output")
        return result

    def detect(self, track: Track2D, meta: VideoMeta, players: list[PlayerObservation] | None,
               *, doubles: bool = False) -> list[HitEvent]:
        index, multiple = player_index(players)
        if doubles or multiple:
            raise ValueError("This HitNet checkpoint has only been validated with two players")
        if not track.n:
            return []
        if len(meta.timestamps) != track.n or meta.n_frames != track.n:
            raise ValueError("Video timestamps and track frames are not aligned")
        timestamps = np.asarray(meta.timestamps, np.float64)
        if not np.array_equal(np.asarray(track.t), timestamps):
            raise ValueError("Track and video timestamps differ")
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError("Timestamps must be finite and strictly increasing")
        raw = raw_matrix(track, meta, players)
        boundaries = [0, *pts_gaps(timestamps).tolist(), track.n]
        hits = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            part = raw[start:stop]
            if not np.isfinite(part[:, :2]).all(1).any():
                continue  # no shuttle observation to support a learned hit
            logits = self.logits(part, meta.fps)
            probability = core.sigmoid(logits[:, 0])
            top = core.sigmoid(logits[:, 1])
            for row in core.decode_nms(probability, meta.fps, self.decoder["thr"], self.decoder["gap_s"]):
                frame = start + row
                side = "far" if top[row] > .5 else "near"
                observation = index.get((frame, side))
                hits.append(HitEvent(frame=frame, t=float(timestamps[frame]),
                                     xy_px=tuple(float(v) for v in track.xy[frame]), hitter=side,
                                     confidence=float(probability[row]),
                                     player_track_id=observation.track_id if observation else None,
                                     source="hitnet"))
        return hits


@lru_cache(maxsize=2)
def _cached(path: str, size: int, mtime_ns: int, device: str) -> HitNet:
    return HitNet(path, device)


def load(weights: str | Path, device: str = "cpu") -> HitNet:
    path = Path(weights).resolve()
    stat = path.stat()
    return _cached(str(path), stat.st_size, stat.st_mtime_ns, device)
