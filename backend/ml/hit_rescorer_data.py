"""Reproducible candidate rows and held-out evaluators for hit rescoring.

This module deliberately imports the runtime candidate builder and its private
feature vector.  The exported rows therefore have the same 13 columns as the
on-device scorer, rather than a parallel feature implementation.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from app.services.tracking import hits, linker, sim
from app.services.tracking.data_types import Track2D, VideoMeta


FEATURES = tuple(hits.HIT_FEATURES)
DEFAULT_TEST_ROOT = Path(__file__).resolve().parent / "data/raw/shuttlecock_trajectory"


@dataclass
class CandidateRows:
    """Flat candidate table plus provenance used for group-aware splitting."""

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    shot_types: np.ndarray
    sides: np.ndarray

    @classmethod
    def empty(cls) -> "CandidateRows":
        return cls(np.empty((0, len(FEATURES)), float), np.empty(0, np.int8),
                   np.empty(0, object), np.empty(0, object), np.empty(0, object))

    def __len__(self) -> int:
        return len(self.y)

    def take(self, mask: np.ndarray) -> "CandidateRows":
        mask = np.asarray(mask, bool)
        return CandidateRows(self.X[mask], self.y[mask], self.groups[mask],
                             self.shot_types[mask], self.sides[mask])

    def concat(self, other: "CandidateRows") -> "CandidateRows":
        if not len(self):
            return other
        if not len(other):
            return self
        return CandidateRows(np.concatenate([self.X, other.X]),
                             np.concatenate([self.y, other.y]),
                             np.concatenate([self.groups, other.groups]),
                             np.concatenate([self.shot_types, other.shot_types]),
                             np.concatenate([self.sides, other.sides]))


def _rows_from_candidates(cands: list[Any], labels: list[tuple[float, str, str]],
                          group: str, fps: float, tolerance_frames: float = 4.0,
                          ) -> CandidateRows:
    """Label runtime candidates by one-to-one nearest hit matching."""
    if not cands:
        return CandidateRows.empty()
    pred = np.array([float(c.tau) * fps for c in cands], float)
    truth = np.array([float(x[0]) for x in labels], float)
    pairs: list[tuple[float, int, int]] = []
    if len(truth):
        for i, p in enumerate(pred):
            for j, t in enumerate(truth):
                d = abs(p - t)
                if d <= tolerance_frames:
                    pairs.append((d, i, j))
    used_c, used_t = set(), set()
    matched: dict[int, int] = {}
    for _, i, j in sorted(pairs):
        if i not in used_c and j not in used_t:
            used_c.add(i)
            used_t.add(j)
            matched[i] = j
    X = np.stack([np.asarray(c.feats, float) for c in cands])
    X = np.nan_to_num(X, nan=0.0, posinf=20.0, neginf=-20.0)
    y = np.zeros(len(cands), np.int8)
    shot = np.full(len(cands), "negative", object)
    side = np.full(len(cands), "unknown", object)
    for i, j in matched.items():
        y[i] = 1
        shot[i], side[i] = str(labels[j][1]), str(labels[j][2])
    groups = np.full(len(cands), group, object)
    return CandidateRows(X, y, groups, shot, side)


def _sim_label_rows(r: sim.SimRally) -> list[tuple[float, str, str]]:
    return [(float(h["frame"]), str(h.get("shot_type", "unknown")), str(h["hitter"]))
            for h in r.hits]


def make_sim_sample(seed: int, image_size: tuple[int, int] = (1280, 720),
                    fps: float = 60.0, style: str = "random", rallies: int = 1,
                    sigma_px: float = 1.5, miss_rate: float = 0.1,
                    fp_rate: float = 0.03) -> tuple[Any, Track2D, list[Any], list[Any]]:
    """Create one deterministic sim sample and run detector corruption/linking."""
    rng = np.random.default_rng([int(seed), 7])
    camera = sim.random_camera(rng, image_size=image_size, style=style)
    doubles = bool(rng.random() < 0.3)
    if rallies > 1:
        rally = sim.simulate_match(seed, rallies, camera=camera, fps=fps, doubles=doubles)
    else:
        rally = sim.simulate_rally(seed, camera=camera, fps=fps, doubles=doubles)
    det_rng = np.random.default_rng([int(seed), 11])
    detections = sim.corrupt_detections(rally, det_rng, sigma_px, miss_rate, True, fp_rate)
    track = linker.link_candidates(detections, rally.meta)
    players = sim.truth_players(rally, every=2)
    return rally, track, players, detections


def collect_sim_rows(seed_start: int, n: int, image_size: tuple[int, int] = (1280, 720),
                     fps_choices: tuple[float, ...] = (30.0, 60.0, 120.0),
                     style_cycle: tuple[str, ...] = ("random", "behind", "broadcast",
                                                      "elevated_side"),
                     rallies: int = 1, sigma_px: float = 1.5,
                     miss_rate: float = 0.1, fp_rate: float = 0.03,
                     tolerance_frames: float = 4.0,
                     ) -> tuple[CandidateRows, list[dict[str, Any]]]:
    """Run the actual sim -> detector corruption -> linker -> hit candidate path.

    The seed stream is explicit.  Simulation rows are intentionally generated
    through the same linker and player-evidence calls used by the benchmark.
    """
    out = CandidateRows.empty()
    info: list[dict[str, Any]] = []
    for k in range(int(n)):
        seed = int(seed_start + k)
        fps = float(fps_choices[k % len(fps_choices)])
        style = style_cycle[k % len(style_cycle)]
        r, track, players, _ = make_sim_sample(seed, image_size, fps, style, rallies,
                                                sigma_px, miss_rate, fp_rate)
        candidates = hits.detect_candidates(hits._prefer_raw(track), r.meta, r.camera, players)
        rows = _rows_from_candidates(candidates, _sim_label_rows(r), f"sim:{seed}", fps,
                                     tolerance_frames)
        out = out.concat(rows)
        info.append({"seed": seed, "fps": fps, "style": style, "doubles": r.doubles,
                     "frames": r.n, "shots": len(r.shots), "candidates": len(candidates),
                     "positives": int(rows.y.sum())})
    return out, info


_SHOT_TYPE = {
    "\u653e\u5c0f\u7403": "net", "\u64cb\u5c0f\u7403": "block", "\u63a8\u7403": "push",
    "\u6311\u7403": "lift", "\u9577\u7403": "clear", "\u767c\u9577\u7403": "serve",
    "\u767c\u77ed\u7403": "serve", "\u6bba\u7403": "smash", "\u9ede\u6263": "smash",
    "\u5207\u7403": "drop", "\u904e\u5ea6\u5207\u7403": "drop", "\u52fe\u7403": "net",
    "\u5e73\u7403": "drive", "\u5c0f\u5e73\u7403": "drive", "\u64b2\u7403": "kill",
    "\u5f8c\u5834\u62bd\u5e73\u7403": "drive", "\u9632\u5b88\u56de\u62bd": "drive",
    "\u9632\u5b88\u56de\u6311": "lift", "\u672a\u77e5\u7403\u7a2e": "unknown",
}


def _finite_xy(row: dict[str, str], x: str, y: str) -> np.ndarray | None:
    try:
        p = np.array([float(row[x]), float(row[y])], float)
    except (KeyError, TypeError, ValueError):
        return None
    return p if np.all(np.isfinite(p)) else None


def _read_set_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("frame_num") and r.get("rally")]
    rows.sort(key=lambda r: (int(float(r["rally"])), float(r["frame_num"])))
    return rows


def _rally_track(rows: list[dict[str, str]], rng: np.random.Generator, fps: float = 30.0,
                  noise_px: float = 1.2, miss_rate: float = 0.06,
                  lead_frames: int = 12, tail_frames: int = 12,
                  ) -> tuple[Track2D, list[tuple[float, str, str]]]:
    """Create a noisy 2-D training track from one ShuttleSet rally.

    ShuttleSet supplies contact frame and contact/landing coordinates but not a
    packaged trajectory.  Between annotated contacts a curved ballistic-looking
    2-D segment is used; its endpoints remain the published annotation points.
    """
    rows = sorted(rows, key=lambda r: float(r["frame_num"]))
    source_frames = np.array([int(round(float(r["frame_num"]))) for r in rows], int)
    first = int(source_frames[0])
    local_frames = source_frames - first + lead_frames
    last = int(local_frames[-1] + tail_frames)
    n = max(last + 1, lead_frames + 20)
    points: list[np.ndarray] = []
    for i, row in enumerate(rows):
        p = _finite_xy(row, "hit_x", "hit_y")
        if p is None and i > 0:
            p = _finite_xy(rows[i - 1], "landing_x", "landing_y")
        if p is None and i + 1 < len(rows):
            p = _finite_xy(rows[i + 1], "hit_x", "hit_y")
        if p is None:
            p = np.array([640.0, 500.0], float) + rng.normal(0, 15, 2)
        points.append(p)
    xy = np.full((n, 2), np.nan, float)
    # Lead-in gives the first contact enough left-side evidence when possible.
    if len(points) > 1:
        d = points[1] - points[0]
    else:
        d = np.array([120.0, -70.0])
    dn = max(float(np.linalg.norm(d)), 1.0)
    pre = points[0] - d / dn * max(25.0, 0.35 * dn)
    for f in range(0, local_frames[0] + 1):
        u = f / max(local_frames[0], 1)
        xy[f] = (1.0 - u) * pre + u * points[0]
    for i in range(len(points) - 1):
        a, b = int(local_frames[i]), int(local_frames[i + 1])
        if b <= a:
            continue
        p0, p1 = points[i], points[i + 1]
        delta = p1 - p0
        dn = float(np.linalg.norm(delta))
        if dn < 4.0:
            perp = np.array([0.0, 1.0])
        else:
            perp = np.array([-delta[1], delta[0]]) / dn
        curve = float(rng.normal(0.0, max(5.0, 0.16 * dn)))
        ff = np.arange(a, b + 1)
        u = (ff - a) / max(b - a, 1)
        xy[ff] = ((1.0 - u)[:, None] * p0 + u[:, None] * p1
                  + (curve * np.sin(np.pi * u))[:, None] * perp)
    a = int(local_frames[-1])
    for f in range(a, n):
        u = (f - a) / max(n - 1 - a, 1)
        xy[f] = points[-1] + u * np.array([18.0, 8.0])
    xy += rng.normal(0.0, noise_px, xy.shape)
    visible = np.ones(n, bool)
    if miss_rate > 0:
        visible &= rng.random(n) >= miss_rate
        for start in np.flatnonzero(rng.random(n) < miss_rate / 18.0):
            visible[start:min(n, start + int(rng.integers(2, 6)))] = False
    # Keep enough observations around contacts; missingness still affects other frames.
    for f in local_frames:
        visible[max(0, f - 2):min(n, f + 3)] = True
    xy[~visible] = np.nan
    t = np.arange(n, dtype=float) / fps
    track = Track2D(t=t, xy=xy, conf=visible.astype(float), visible=visible,
                    interpolated=np.zeros(n, bool))
    labels = [(float(f), _SHOT_TYPE.get(str(r.get("type", "")), "unknown"),
               str(r.get("player", "unknown"))) for f, r in zip(local_frames, rows)]
    return track, labels


def shuttleset_matches(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted({p.parent for p in root.glob("*/*.csv") if p.is_file()}, key=lambda p: p.name)


def shuttleset_split(matches: list[Path], validation_fraction: float = 0.2,
                     index: int | None = None) -> str:
    """Deterministic match-level split; no set from a match is split across sides."""
    if index is None:
        index = matches.index(matches[0]) if matches else 0
    n_val = max(1, int(math.ceil(len(matches) * validation_fraction))) if matches else 0
    # Every fifth sorted match is validation, with the tail padded if needed.
    val_indices = set(range(0, len(matches), max(2, int(round(1 / validation_fraction)))))
    if len(val_indices) < n_val:
        val_indices.update(range(len(matches) - n_val, len(matches)))
    return "val" if index in val_indices else "train"


def collect_shuttleset_rows(root: str | Path, split: str = "train", seed: int = 0,
                            max_rallies: int | None = None, fps: float = 30.0,
                            noise_px: float = 1.2, miss_rate: float = 0.06,
                            tolerance_frames: float = 4.0,
                            ) -> tuple[CandidateRows, list[dict[str, Any]]]:
    """Build rows from ShuttleSet annotations, split by complete match."""
    matches = shuttleset_matches(root)
    out = CandidateRows.empty()
    info: list[dict[str, Any]] = []
    count = 0
    for mi, match_dir in enumerate(matches):
        if shuttleset_split(matches, index=mi) != split:
            continue
        for set_path in sorted(match_dir.glob("*.csv")):
            rows = _read_set_rows(set_path)
            by_rally: dict[str, list[dict[str, str]]] = {}
            for row in rows:
                by_rally.setdefault(str(row["rally"]), []).append(row)
            for rally_name, rally_rows in sorted(by_rally.items(), key=lambda kv: int(kv[0])):
                if max_rallies is not None and count >= max_rallies:
                    return out, info
                local_seed = (seed * 1_000_003 + mi * 10_007 + count * 97
                              + sum(ord(c) for c in match_dir.name)) & 0xFFFFFFFF
                track, labels = _rally_track(rally_rows, np.random.default_rng(local_seed),
                                              fps=fps, noise_px=noise_px,
                                              miss_rate=miss_rate)
                meta = VideoMeta(fps, 1280, 720, track.n, track.t)
                candidates = hits.detect_candidates(track, meta)
                rows_out = _rows_from_candidates(candidates, labels,
                                                 f"shuttleset:{match_dir.name}", fps,
                                                 tolerance_frames)
                out = out.concat(rows_out)
                info.append({"match": match_dir.name, "set": set_path.stem,
                             "rally": rally_name, "strokes": len(labels),
                             "candidates": len(candidates), "positives": int(rows_out.y.sum())})
                count += 1
    return out, info


def _probe(path: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")
    n = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    t = np.arange(n, dtype=float) / max(fps, 1.0)
    return VideoMeta(fps, w, h, n, t)


def load_tracknet_csv(path: str | Path, n: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.loadtxt(str(path), delimiter=",", skiprows=1, ndmin=2)
    xy = np.full((n, 2), np.nan, float)
    visible = np.zeros(n, bool)
    if rows.size == 0:
        return xy, visible
    frames = rows[:, 0].astype(int)
    valid = (frames >= 0) & (frames < n) & (rows[:, 1] > 0)
    xy[frames[valid]] = rows[valid, 2:4]
    visible[frames[valid]] = True
    return xy, visible


def real_test_clips(root: str | Path = DEFAULT_TEST_ROOT) -> list[dict[str, Any]]:
    root = Path(root)
    out = []
    for video in sorted((root / "TrackNetV2/Test").glob("match*/video/*.mp4")):
        match, clip = video.parent.parent.name, video.stem
        corrected = root / "corrected" / match / "corrected_csv" / f"{clip}_ball.csv"
        original = video.parent.parent / "csv" / f"{clip}_ball.csv"
        labels = corrected if corrected.exists() else original
        if labels.exists():
            out.append({"match": match, "clip": clip, "video": video, "labels": labels,
                        "corrected": corrected.exists()})
    return out


def reversal_labels(xy: np.ndarray, visible: np.ndarray, fps: float,
                    half_s: float = 0.16, gap_s: float = 0.03,
                    min_speed_px_s: float = 80.0, cos_max: float = 0.1,
                    nms_s: float = 0.20) -> list[int]:
    """Independent pseudo-labels: a direction reversal in the labelled track."""
    xy = np.asarray(xy, float)
    visible = np.asarray(visible, bool)
    ts = np.arange(len(xy), dtype=float) / float(fps)
    idx = np.flatnonzero(visible & np.isfinite(xy).all(axis=1))
    scored: list[tuple[int, float]] = []
    for k in range(len(xy)):
        left = idx[(ts[idx] >= ts[k] - half_s) & (ts[idx] <= ts[k] - gap_s)]
        right = idx[(ts[idx] >= ts[k] + gap_s) & (ts[idx] <= ts[k] + half_s)]
        if len(left) < 3 or len(right) < 3:
            continue
        vl = np.array([np.polyfit(ts[left] - ts[k], xy[left, j], 1)[0] for j in (0, 1)])
        vr = np.array([np.polyfit(ts[right] - ts[k], xy[right, j], 1)[0] for j in (0, 1)])
        sl, sr = float(np.linalg.norm(vl)), float(np.linalg.norm(vr))
        cos = float(np.dot(vl, vr) / (sl * sr + 1e-9))
        if min(sl, sr) < min_speed_px_s or cos > cos_max:
            continue
        scored.append((k, (1.0 - cos) * math.sqrt(sl * sr)))
    kept: list[tuple[int, float]] = []
    radius = max(1, int(round(nms_s * fps)))
    for row in sorted(scored, key=lambda x: -x[1]):
        if all(abs(row[0] - old[0]) >= radius for old in kept):
            kept.append(row)
    return sorted(int(x[0]) for x in kept)


def event_pairs(pred: Iterable[int], truth: Iterable[int], tolerance_frames: int = 3,
                ) -> list[tuple[int, int]]:
    p, t = list(map(int, pred)), list(map(int, truth))
    pairs = [(abs(a - b), i, j) for i, a in enumerate(p) for j, b in enumerate(t)
             if abs(a - b) <= tolerance_frames]
    used_p, used_t, out = set(), set(), []
    for _, i, j in sorted(pairs):
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
            out.append((i, j))
    return out


def event_scores(pred: Iterable[int], truth: Iterable[int], tolerance_frames: int = 3,
                 ) -> dict[str, float]:
    p, t = list(pred), list(truth)
    pairs = event_pairs(p, t, tolerance_frames)
    tp, fp, fn = len(pairs), len(p) - len(pairs), len(t) - len(pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "n_pred": len(p), "n_true": len(t),
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)
            if precision + recall else 0.0}
