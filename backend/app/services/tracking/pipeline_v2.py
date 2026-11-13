"""
Tracking pipeline v2 — orchestrates the stage modules for one match video.

  probe → detect shuttle candidates → link into a 2-D track
        → calibrate camera from court lines (or user-tapped corners)
        → track players + pose → detect hits → fit 3-D flights (drag ODE)
        → segment rallies (in/out/net, winner) → classify shots → score

Every stage is optional at runtime: a failure or missing module is logged in
`diagnostics` and downstream stages degrade (e.g. no camera → no metric
speeds / landing points, never fabricated ones).
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.services.tracking import video_io
from app.services.tracking.camera import CameraModel
from app.services.tracking.data_types import (
    HitEvent, PlayerObservation, RallyV2, ShotSegment3D, Track2D, VideoMeta,
)

logger = logging.getLogger(__name__)
Progress = Callable[[float, str], None]


@dataclass
class PipelineOptions:
    weights_dir: str | None = None
    device: str | None = None
    doubles: bool | None = None                 # None = auto-detect
    manual_corners: dict[str, tuple[float, float]] | None = None   # user-tapped
    corners_normalized: bool = True             # manual corners given as 0–1 of frame
    camera_hint: dict | None = None             # CameraModel.to_dict() from device
    player_step: int | None = None              # None = every ~1/PLAYER_HZ s (fps-aware)
    pose_model: str | None = None               # e.g. "yolov8m-pose.pt"; None = players default
    pose_imgsz: int | None = None
    calibration_frames: int = 24
    # Explicit opt-in until continuous-phone-video validation is available.
    hitnet_weights: str | None = None
    bst_manifest: str | None = None
    rtmpose_weights: str | None = None


# Pose sampling rate. step=2 was tuned on 30 fps footage; a fixed frame step would
# double the pose work at 60 fps (iPhone) for no tracking gain — on a 60 s 1080p60
# clip, 15 Hz vs 30 Hz: 102 s vs 216 s, foot position median diff 1.3 cm (p95 5.9 cm).
PLAYER_HZ = 15.0


def player_step(fps: float | None, override: int | None = None) -> int:
    if override is not None:
        return max(1, int(override))
    return max(1, int(round((fps or 30.0) / PLAYER_HZ)))


@dataclass
class MatchAnalysis:
    meta: VideoMeta
    track: Track2D | None = None
    camera: CameraModel | None = None
    players: list[PlayerObservation] = field(default_factory=list)
    hits: list[HitEvent] = field(default_factory=list)
    segments: list[ShotSegment3D] = field(default_factory=list)
    rallies: list[RallyV2] = field(default_factory=list)
    score: list[dict] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    processing_time_seconds: float = 0.0


class _Stages:
    """Records timing and failures per stage without aborting the run."""

    def __init__(self, diagnostics: dict[str, Any], progress: Progress | None):
        self.diag = diagnostics
        self.progress = progress or (lambda p, s: None)

    def run(self, name: str, pct: float, fn: Callable[[], Any], default: Any = None) -> Any:
        self.progress(pct, name)
        t0 = time.perf_counter()
        try:
            out = fn()
            self.diag.setdefault("stages", {})[name] = {
                "ok": True, "seconds": round(time.perf_counter() - t0, 3)}
            return out
        except Exception as exc:          # stage isolation is the point here
            logger.exception("stage %s failed", name)
            self.diag.setdefault("stages", {})[name] = {
                "ok": False, "error": f"{type(exc).__name__}: {exc}",
                "seconds": round(time.perf_counter() - t0, 3)}
            return default


def analyze_video(path: str, opts: PipelineOptions | None = None,
                  progress: Progress | None = None) -> MatchAnalysis:
    from app.services.tracking import (
        court_calibration, detector, hits, linker, players, rally, reconstruct3d, shots,
    )

    opts = opts or PipelineOptions()
    t_start = time.perf_counter()
    meta = video_io.probe(path)
    res = MatchAnalysis(meta=meta)
    st = _Stages(res.diagnostics, progress)
    # pts_gaps: holes in the presentation clock (video_io.pts_gap_info; same key and
    # fields on the phone). No stage links, fits or measures speed across one.
    res.diagnostics["video"] = {"fps": meta.fps, "size": [meta.width, meta.height],
                                "frames": meta.n_frames,
                                "pts_gaps": video_io.pts_gap_info(meta.timestamps)}
    size = (meta.width, meta.height)

    def _detect():
        det = detector.build_detector(opts.weights_dir, opts.device)
        res.diagnostics["detector"] = type(det).__name__
        # detector progress occupies 10 → 50 %
        return det.detect_video(path, meta, progress=lambda f: st.progress(10 + 40 * f, "detect"))

    candidates = st.run("detect", 10, _detect, default=None)
    if candidates is not None:
        res.track = st.run("link", 50, lambda: linker.link_candidates(candidates, meta))

    auto: dict[str, Any] = {}
    res.camera = st.run("calibrate", 55, lambda: _calibrate(path, meta, size, opts, court_calibration, auto))
    # Why the automatic path accepted or rejected the court: the calibrator's own evidence,
    # same key and fields as the phone (ShuttlAnalysis). Absent when corners were tapped.
    if auto:
        res.diagnostics["calibration_auto"] = auto
    if res.camera is not None:
        res.diagnostics["calibration"] = {"confidence": res.camera.confidence,
                                          "reproj_rmse_px": res.camera.reproj_rmse_px}

    pose_kw = {k: v for k, v in (("model_name", opts.pose_model), ("imgsz", opts.pose_imgsz))
               if v is not None}
    res.players = st.run("players", 60, lambda: players.track_players(
        path, meta, res.camera, step=player_step(meta.fps, opts.player_step), device=opts.device,
        doubles=opts.doubles, **pose_kw), default=[]) or []

    if res.track is None:
        res.processing_time_seconds = time.perf_counter() - t_start
        return res

    res.hits = _detect_hits(res, opts, st, hits)
    if res.camera is not None:
        res.segments = st.run("reconstruct3d", 85, lambda: reconstruct3d.reconstruct_segments(
            res.track, res.hits, res.camera, meta, res.players or None), default=[]) or []

    res.rallies = st.run("rallies", 92, lambda: rally.segment_rallies(
        res.track, res.hits, res.segments, meta, res.camera,
        players=res.players or None), default=[]) or []
    res.rallies = [st.run("shots", 95, lambda r=r: shots.classify_shots(
        r, res.camera, res.players or None), default=r) for r in res.rallies]
    if opts.bst_manifest:
        _refine_strokes(path, res, opts, st)
    if hasattr(rally, "score_match"):
        res.score = st.run("score", 98, lambda: rally.score_match(
            res.rallies, doubles=bool(rally.detect_doubles(res.players or None))), default=[]) or []

    res.processing_time_seconds = time.perf_counter() - t_start
    return res


def _refine_strokes(path: str, res: MatchAnalysis, opts: PipelineOptions, st: _Stages):
    diag = res.diagnostics.setdefault("stroke_model", {"requested": "BST_CG_AP_seq30", "used": "rules"})
    if res.diagnostics.get("hit_model", {}).get("used") != "hitnet_single_v2":
        diag["fallback_reason"] = "validated_hitnet_required"
        return
    if not opts.rtmpose_weights or res.camera is None:
        diag["fallback_reason"] = "rtmpose_and_calibration_required"
        return
    def refine():
        from app.services.tracking import bst, rtmpose
        pose = rtmpose.load(opts.rtmpose_weights, opts.device or "cpu")
        pose_players = pose.refine(path, res.meta, res.players, res.hits)
        res.diagnostics["rtmpose"] = {"sha256": pose.sha256, "providers": pose.providers,
                                      "observations": len(pose_players)}
        model = bst.load(opts.bst_manifest, opts.device or "cpu")
        return model.refine(res.rallies, res.track, res.meta, pose_players)
    result = st.run("bst", 96, refine)
    if result is None:
        diag["fallback_reason"] = "bst_stage_failed"
    else:
        diag.update(result)


def _detect_hits(res: MatchAnalysis, opts: PipelineOptions, st: _Stages, rules):
    """Use the configured singles candidate; report every rules fallback explicitly."""
    diag = res.diagnostics.setdefault("hit_model", {"requested": "rules", "used": "rules"})
    if opts.hitnet_weights:
        from app.services.tracking import hitnet
        diag["requested"] = "hitnet_single_v2"
        _, multiple = hitnet.player_index(res.players)
        if opts.doubles or multiple:
            diag["fallback_reason"] = "doubles_not_validated"
        else:
            def learned():
                model = hitnet.load(opts.hitnet_weights, opts.device or "cpu")
                diag["sha256"] = model.sha256
                diag["validation_scope"] = "annotated singles rallies; continuous video pending"
                return model.detect(res.track, res.meta, res.players)
            detected = st.run("hitnet", 80, learned, default=None)
            if detected is not None:
                diag["used"] = "hitnet_single_v2"
                return detected
            diag["fallback_reason"] = "hitnet_stage_failed"
    return st.run("hits", 80, lambda: rules.detect_hits(
        res.track, res.meta, res.camera, res.players or None), default=[]) or []


def _calibrate(path: str, meta: VideoMeta, size: tuple[int, int], opts: PipelineOptions,
               court_calibration, auto: dict[str, Any] | None = None) -> CameraModel | None:
    """Prefer user-tapped corners (then refine on lines), else fully automatic.

    On the automatic path `auto` receives calibrate_from_frames' diagnostics, JSON-safe
    and minus the per-candidate objects (source, n_hyp, n_lines, n_valid, coverage,
    rmse, n_t, n_l, …, margin, confidence, reason when rejected), so a rejected court
    is explainable from the analysis alone."""
    auto = auto if auto is not None else {}
    frames = [f for _, f in video_io.sample_frames(path, opts.calibration_frames, meta.n_frames)]
    if opts.manual_corners:
        scale = size if opts.corners_normalized else (1, 1)
        pts = {k: (float(u) * scale[0], float(v) * scale[1])
               for k, (u, v) in opts.manual_corners.items()}
        cam = court_calibration.calibrate_from_points(pts, size)
        if frames and hasattr(court_calibration, "refine_camera"):
            try:
                cam = court_calibration.refine_camera(cam, frames)
            except Exception:
                logger.warning("refinement from manual corners failed; keeping DLT result")
        return cam
    diag: dict[str, Any] = {}
    fn = court_calibration.calibrate_from_frames
    if "diagnostics" in inspect.signature(fn).parameters:
        cam = fn(frames, size, doubles_hint=opts.doubles, diagnostics=diag)
    else:                                   # a calibrator without the diagnostics hook
        cam = fn(frames, size, doubles_hint=opts.doubles)
    from app.services.tracking.serialize import _jsonable
    auto.update(_jsonable({k: v for k, v in diag.items() if k != "candidates"}))
    if cam is None and opts.camera_hint:
        cam = CameraModel.from_dict(opts.camera_hint)
    return cam
