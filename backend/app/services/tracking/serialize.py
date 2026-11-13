"""MatchAnalysis → JSON-safe dict (schema version 2), stored per match and
served by GET /matches/{id}/analysis. Pixel positions are exported normalised
to 0–1 of the upright frame so clients never need the video size; world
positions use the court_model frame (metres, origin at court centre)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from app.services.tracking import court_model
from app.services.tracking.data_types import RallyV2, ShotEventV2, ShotSegment3D

SCHEMA_VERSION = 2
_BASE_TYPES = {"smash", "clear", "drop", "net", "drive", "serve", "unknown"}
# Extra fine-grained types collapse onto the closest legacy type for old clients.
LEGACY_TYPE = {"lift": "clear", "push": "drive", "kill": "smash", "block": "net",
               "cross_net": "net", "short_serve": "serve", "long_serve": "serve"}


def legacy_shot_type(t: str) -> str:
    return t if t in _BASE_TYPES else LEGACY_TYPE.get(t, "unknown")


def _f(x: Any, nd: int = 3) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if not math.isfinite(x) else round(x, nd)


def _xy(p: Any, nd: int = 3) -> list[float | None] | None:
    return None if p is None else [_f(v, nd) for v in p]


def _segment(seg: ShotSegment3D | None, w: int, h: int, camera) -> dict | None:
    if seg is None:
        return None
    speed = np.linalg.norm(seg.v0)
    apex = float(np.nanmax(seg.positions[:, 2])) if len(seg.positions) else None
    path_px = None
    if camera is not None and len(seg.positions):
        uv = camera.project(seg.positions)
        path_px = [[_f(u / w, 4), _f(v / h, 4)] for u, v in uv]
    return {
        "start_frame": seg.start_frame, "end_frame": seg.end_frame,
        "speed0_kmh": _f(speed * 3.6, 1),
        "p0": _xy(seg.p0), "v0": _xy(seg.v0),
        "apex_m": _f(apex, 2),
        "net_clearance_m": _f(None if seg.net_crossing_height is None
                              else seg.net_crossing_height - court_model.NET_HEIGHT_CENTRE, 2),
        "net_crossing_y": _f(seg.net_crossing_y, 2), "end_reason": seg.end_reason,
        "landing_xy": _xy(seg.landing_xy), "landing_t": _f(seg.landing_t),
        "rmse_px": _f(seg.rmse_px, 2), "confidence": _f(seg.confidence, 2),
        "frames": [int(f) for f in seg.frames],
        "world": [_xy(p) for p in seg.positions],
        "path_norm": path_px,
    }


def _shot(s: ShotEventV2, w: int, h: int, camera) -> dict:
    return {
        "frame": s.hit.frame, "t": _f(s.hit.t),
        "hitter": s.hit.hitter, "hit_confidence": _f(s.hit.confidence, 2),
        "xy_norm": [_f(s.hit.xy_px[0] / w, 4), _f(s.hit.xy_px[1] / h, 4)],
        "shot_type": s.shot_type, "legacy_shot_type": legacy_shot_type(s.shot_type),
        "shot_type_confidence": _f(s.shot_type_confidence, 2),
        "hitter_pos": _xy(s.hitter_pos_world), "hit_height_m": _f(s.hit_height_m, 2),
        "stroke_side": s.stroke_side,
        "flight": _segment(s.segment, w, h, camera),
        **({"hit_source": s.hit.source} if getattr(s.hit, "source", "kink") != "kink" else {}),
    }


def _rally(r: RallyV2, w: int, h: int, camera) -> dict:
    return {
        "start_frame": r.start_frame, "end_frame": r.end_frame,
        "end_reason": r.end_reason, "winner": r.winner, "server": r.server,
        "landing_xy": _xy(r.landing_xy), "in_out_margin_m": _f(r.in_out_margin_m),
        "confidence": _f(r.confidence, 2),
        "outcome_confidence": _f(r.outcome_confidence, 2), "outcome_source": r.outcome_source,
        "shots": [_shot(s, w, h, camera) for s in r.shots],
    }


def _track_runs(track, w: int, h: int) -> list[dict]:
    """Contiguous runs of known positions — compact enough for an hour of video."""
    if track is None:
        return []
    ok = np.isfinite(track.xy).all(axis=1)
    runs, i, n = [], 0, track.n
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < n and ok[j]:
            j += 1
        seg = track.xy[i:j]
        runs.append({
            "start_frame": int(i), "t0": _f(track.t[i]),
            "xy": np.round(seg / [w, h], 4).tolist(),
            "interp": [int(k) for k in np.flatnonzero(track.interpolated[i:j])],
        })
        i = j
    return runs


def summarize(rallies: list[RallyV2]) -> dict:
    shots = [s for r in rallies for s in r.shots]
    speeds = [s.segment.speed0_kmh for s in shots if s.segment is not None]
    smash = [s.segment.speed0_kmh for s in shots
             if s.segment is not None and s.shot_type in ("smash", "kill")]
    counts: dict[str, int] = {}
    for s in shots:
        counts[s.shot_type] = counts.get(s.shot_type, 0) + 1
    ends: dict[str, int] = {}
    for r in rallies:
        ends[r.end_reason] = ends.get(r.end_reason, 0) + 1
    return {
        "total_rallies": len(rallies), "total_shots": len(shots),
        "max_speed_kmh": _f(max(speeds, default=0.0), 1),
        "avg_speed_kmh": _f(float(np.mean(speeds)) if speeds else 0.0, 1),
        "max_smash_kmh": _f(max(smash, default=0.0), 1),
        "avg_smash_kmh": _f(float(np.mean(smash)) if smash else 0.0, 1),
        "shot_type_counts": counts, "end_reason_counts": ends,
        "avg_rally_shots": _f(len(shots) / len(rallies), 1) if rallies else 0.0,
    }


def to_dict(res) -> dict:
    """Serialise a pipeline_v2.MatchAnalysis."""
    w, h = res.meta.width or 1, res.meta.height or 1
    cam = res.camera
    out = {
        "schema_version": SCHEMA_VERSION,
        "video": {"fps": _f(res.meta.fps, 3), "width": res.meta.width,
                  "height": res.meta.height, "frames": res.meta.n_frames},
        "camera": cam.to_dict() if cam is not None else None,
        "calibrated": cam is not None,
        "summary": summarize(res.rallies),
        "rallies": [_rally(r, w, h, cam) for r in res.rallies],
        "score": res.score,
        "track": _track_runs(res.track, w, h),
        "diagnostics": res.diagnostics,
        "processing_time_seconds": _f(res.processing_time_seconds, 2),
    }
    try:
        from app.services.tracking import players as players_mod
        if res.players and hasattr(players_mod, "movement_stats"):
            out["movement"] = {side: _jsonable(players_mod.movement_stats(res.players, side))
                               for side in ("near", "far")}
    except Exception:
        out["movement"] = None
    return _jsonable(out)


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return _f(x, 6)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def mobile_archive(res) -> dict:
    """Lossless ShuttlCore.MatchAnalysis contract for the existing iPhone reports.

    Preserve pixel coordinates, VFR timestamps and confidence/missing-value masks.
    Keypoints are transient inference data; the phone archive also omits them.
    Walk fields without asdict's expensive deep copy of full-video numpy arrays.
    """
    from dataclasses import fields, is_dataclass

    def value(x):
        if is_dataclass(x):
            return {f.name: value(getattr(x, f.name)) for f in fields(x)
                    if f.name != "keypoints"}
        if isinstance(x, np.ndarray):
            return value(x.tolist())
        if isinstance(x, dict):
            return {k: value(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [value(v) for v in x]
        if isinstance(x, (np.bool_, bool)):
            return bool(x)
        if isinstance(x, (np.integer, int)):
            return int(x)
        if isinstance(x, (np.floating, float)):
            return float(x) if math.isfinite(x) else None
        return x

    return value(res)
