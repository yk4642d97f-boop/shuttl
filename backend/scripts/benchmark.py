#!/usr/bin/env python
"""
Shuttl tracking benchmark: runs seeded synthetic rallies (sim.py) or the real
labelled TrackNetV2 Test split through whatever pipeline stages exist, prints a
metrics table and writes JSON to backend/ml/benchmarks/<timestamp>_<stage>.json.

Run from backend/:
  .venv/bin/python scripts/benchmark.py --stage oracle --n 10
  .venv/bin/python scripts/benchmark.py --stage e2e --n 10 --rallies 3
  .venv/bin/python scripts/benchmark.py --stage video --n 2 --image-size 1280x720
  .venv/bin/python scripts/benchmark.py --stage real --clips 3 --max-frames 300 --detector wasb
  .venv/bin/python scripts/benchmark.py --stage real_oracle

Stages
  oracle       truth fed into each stage in isolation: true camera + corrupted
               detections → linker; true track (+ --track-noise px) → hits;
               true track + hits → 3-D; true track/hits/segments → rallies;
               true rallies (types blanked) → shots.
  e2e          corrupted detections → linker → hits → 3-D → rallies → shots, true camera.
  video        render_video → detector.detect_video + court_calibration → e2e chain
               with the calibrated camera (true camera if calibration is missing/fails).
  real         TrackNetV2 Test clips (corrected labels preferred): detector →
               detection P/R/F1 (top-1, score ≥ 0.5) at the TrackNetV3 rule
               (4 px @ 512×288 with int truncation, 'tn4') and a fixed 10 px, px error
               stats; linker on the full candidate lists → same + jitter/jumps.
               Candidates cached in --cache-dir (npz per clip+detector). Also runs
               real_oracle unless --no-real-oracle.
  real_oracle  real GT track corrupted with sim.corrupt_px (same noise model as
               corrupt_detections) → linker: the linker on real motion statistics.
Synthetic detection/track truth = sim.image_truth(): the shuttle wherever it is in
the image (in play, plus the dead shuttle held/lying between rallies, like the real
labels); tracks are also scored on frames where it is hidden behind a player.
Metrics are pooled over all samples (frame-level counts summed, error stats over
all frames). Missing modules/functions are reported as SKIPPED; exceptions are
recorded per sample and do not stop the run. Players are always the simulator's
truth (PlayerObservation per --player-step frames) in synthetic modes.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import json
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.tracking import court_model as cm  # noqa: E402
from app.services.tracking import metrics as M  # noqa: E402
from app.services.tracking import sim as S  # noqa: E402
from app.services.tracking.data_types import ShuttleCandidate, Track2D, VideoMeta  # noqa: E402

REAL_ROOT = ROOT / "ml/data/raw/shuttlecock_trajectory"
OUT_DIR = ROOT / "ml/benchmarks"
GAP = 100_000                     # frame offset between pooled samples

STAGE_API = {
    "linker": ("linker", "link_candidates"),
    "hits": ("hits", "detect_hits"),
    "reconstruct3d": ("reconstruct3d", "reconstruct_segments"),
    "rally": ("rally", "segment_rallies"),
    "shots": ("shots", "classify_shots"),
    "calibration": ("court_calibration", "calibrate_from_frames"),
    "detector": ("detector", "build_detector"),
}


# ── module discovery ─────────────────────────────────────────────────────────

def resolve_stages() -> tuple[dict[str, Callable], dict[str, str]]:
    """{stage: callable} for available stages, {stage: reason} for missing ones."""
    fns, missing = {}, {}
    for stage, (mod, attr) in STAGE_API.items():
        try:
            m = importlib.import_module(f"app.services.tracking.{mod}")
        except Exception as e:                      # ImportError or a broken sibling
            missing[stage] = f"import {mod}: {type(e).__name__}: {e}"[:200]
            continue
        if not hasattr(m, attr):
            missing[stage] = f"{mod}.{attr} not defined"
            continue
        fns[stage] = getattr(m, attr)
    return fns, missing


def optional_attr(mod: str, attr: str) -> Callable | None:
    try:
        return getattr(importlib.import_module(f"app.services.tracking.{mod}"), attr, None)
    except Exception:
        return None


# ── pooling ──────────────────────────────────────────────────────────────────

class Pool:
    """Collects per-sample payloads (frame indices shifted by a per-sample
    offset) and computes pooled metrics per stage."""

    def __init__(self) -> None:
        self.frames: dict[str, list] = defaultdict(list)     # (pred, true, vis, interp, mask)
        self.events: dict[str, list] = defaultdict(list)     # (pred dicts, true dicts)
        self.lists: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        self.scalars: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        self.errors: dict[str, list[str]] = defaultdict(list)
        self.skipped: dict[str, str] = {}
        self.seconds: dict[str, float] = defaultdict(float)
        self.kind: dict[str, str] = {}

    def add_frames(self, stage: str, pred: np.ndarray, true: np.ndarray, vis: np.ndarray,
                   interp: np.ndarray | None = None, mask: np.ndarray | None = None) -> None:
        n = len(true)
        self.kind[stage] = "frames"
        self.frames[stage].append((np.asarray(pred, float).reshape(-1, 2), np.asarray(true, float),
                                   np.asarray(vis, bool),
                                   np.zeros(n, bool) if interp is None else np.asarray(interp, bool),
                                   np.ones(n, bool) if mask is None else np.asarray(mask, bool)))

    def add_events(self, stage: str, kind: str, pred: list[dict], true: list[dict], off: int) -> None:
        self.kind[stage] = kind
        self.events[stage].append((_shift(pred, off), _shift(true, off)))

    def add_list(self, stage: str, key: str, values: list) -> None:
        self.lists[stage][key].extend(values)

    def add_scalar(self, stage: str, key: str, value: float) -> None:
        self.scalars[stage][key].append(float(value))

    def error(self, stage: str, sample: str, exc: BaseException) -> None:
        tb = traceback.format_exception_only(type(exc), exc)[-1].strip()
        self.errors[stage].append(f"{sample}: {tb}"[:400])

    def results(self, tol_px: float, tol_frames: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        stages = list(dict.fromkeys(list(self.kind) + list(self.skipped) + list(self.errors)
                                    + list(self.lists) + list(self.scalars)))
        for st in stages:
            res: dict[str, Any] = {}
            kind = self.kind.get(st)
            if kind == "frames" and self.frames[st]:
                res.update(_pooled_frames(self.frames[st], tol_px, track="linker" in st))
            elif kind == "hits":
                P = [e for p, _ in self.events[st] for e in p]
                T = [e for _, t in self.events[st] for e in t]
                res.update(M.hit_metrics(P, T, tol_frames))
            elif kind == "segments":
                P = [e for p, _ in self.events[st] for e in p]
                T = [e for _, t in self.events[st] for e in t]
                res.update(M.segment_metrics(P, T, tol_frames + 1))
            elif kind == "rallies":
                P = [e for p, _ in self.events[st] for e in p]
                T = [e for _, t in self.events[st] for e in t]
                res.update(M.rally_metrics(P, T))
            L = self.lists.get(st, {})
            if L.get("type_pred"):
                sm = M.shot_type_metrics(L["type_pred"], L["type_true"])
                res.update({k if k != "n" else "n_types": v for k, v in sm.items()})
                if L.get("type_cov"):
                    res["type_coverage"] = float(np.mean(L["type_cov"]))
            if L.get("call_pred"):
                io = M.in_out_metrics(L["call_pred"], L["call_true"], L["call_margin"])
                res.update({f"inout_{k}": v for k, v in io.items()})
            if L.get("land_pred"):
                res.update(M.landing_metrics(L["land_pred"], L["land_true"]))
            for k, v in self.scalars.get(st, {}).items():
                v = np.asarray(v, float)
                res[f"{k}_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
                res[f"{k}_median"] = float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")
            if st in self.skipped and not res:
                res["skipped"] = self.skipped[st]
            if self.errors.get(st):
                res["n_errors"] = len(self.errors[st])
                res["errors"] = self.errors[st][:5]
            if self.seconds.get(st):
                res["seconds"] = round(self.seconds[st], 3)
            out[st] = res
        return out


def _shift(items: list[dict], off: int) -> list[dict]:
    out = []
    for d in items:
        d = dict(d)
        for k in ("frame", "start_frame", "end_frame"):
            if k in d and d[k] is not None:
                d[k] = int(d[k]) + off
        out.append(d)
    return out


def _pooled_frames(chunks: list, tol_px: float, track: bool) -> dict[str, Any]:
    sep = (np.full((1, 2), np.nan), np.full((1, 2), np.nan), np.zeros(1, bool),
           np.zeros(1, bool), np.zeros(1, bool))
    parts = [c for ch in chunks for c in (ch, sep)]
    pred, true, vis, interp, mask = (np.concatenate([p[i] for p in parts]) for i in range(5))
    tr = Track2D(t=np.arange(len(pred), dtype=float), xy=pred, conf=np.isfinite(pred[:, 0]).astype(float),
                 visible=np.isfinite(pred[:, 0]) & ~interp, interpolated=interp)
    res = M.track_metrics(tr, true, vis, tol_px, mask=mask) if track else \
        M.detection_metrics(pred, true, vis, tol_px, mask=mask)
    return res


# ── converters to plain (poolable) dicts ─────────────────────────────────────

def _hit_d(h: Any) -> dict:
    return {"frame": int(M._get(h, "frame")), "hitter": str(M._get(h, "hitter", "unknown"))}


def _seg_d(s: Any) -> dict:
    lx = M._get(s, "landing_xy")
    return {"start_frame": int(M._get(s, "start_frame", M._get(s, "frame", -1))),
            "p0": np.asarray(M._get(s, "p0"), float), "v0": np.asarray(M._get(s, "v0"), float),
            "net_crossing_height": M._get(s, "net_crossing_height"),
            "landing_xy": None if lx is None else tuple(float(v) for v in lx)}


def _true_shot_d(sh: S.SimShot) -> dict:
    d = _seg_d(sh)
    d["start_frame"] = sh.frame
    if sh.outcome not in ("in", "out"):
        d["landing_xy"] = None                   # a returned shot never lands
    if not np.isfinite(sh.net_crossing_height):
        d["net_crossing_height"] = None
    return d


def _rally_d(r: Any) -> dict:
    m = M._get(r, "margin", M._get(r, "in_out_margin_m"))
    lx = M._get(r, "landing_xy")
    return {"start_frame": int(M._get(r, "start_frame")), "end_frame": int(M._get(r, "end_frame")),
            "end_reason": str(M._get(r, "end_reason", "unknown")),
            "winner": str(M._get(r, "winner", "unknown")),
            "landing_xy": None if lx is None else tuple(float(v) for v in lx),
            "margin": None if m is None else float(m)}


# ── stage runner ─────────────────────────────────────────────────────────────

class Runner:
    def __init__(self, pool: Pool, fns: dict[str, Callable], missing: dict[str, str]) -> None:
        self.pool, self.fns, self.missing = pool, fns, missing
        self.cache: dict[str, Any] = {}

    def call(self, stage: str, sample: str, *a, **kw) -> Any:
        """fn(*a, **kw) for an available stage; None (+ SKIPPED/error record) otherwise."""
        fn = self.fns.get(stage)
        if fn is None:
            self.pool.skipped[stage] = self.missing.get(stage, "not available")
            return None
        t = time.perf_counter()
        try:
            return fn(*a, **kw)
        except Exception as e:
            self.pool.error(stage, sample, e)
            return None
        finally:
            self.pool.seconds[stage] += time.perf_counter() - t

    def skip(self, stage: str, why: str) -> None:
        self.pool.skipped.setdefault(stage, why)


def noisy_truth_track(sim: S.SimRally, sigma: float, rng: np.random.Generator) -> Track2D:
    tr = S.truth_track(sim)
    if sigma > 0:
        tr.xy = tr.xy + rng.normal(0, sigma, tr.xy.shape)
    return tr


def blank_types(rally: Any) -> Any:
    shots = [replace(s, shot_type="unknown", shot_type_confidence=0.0, hitter_pos_world=None,
                     hit_height_m=None) for s in rally.shots]
    return replace(rally, shots=shots)


def eval_chain(run: Runner, pool: Pool, name: str, sim: S.SimRally, off: int, meta: VideoMeta,
               cands: list | None, camera, players, prefix: str) -> None:
    """linker → hits → 3-D → rallies → shots on one sample, each evaluated vs truth."""
    det_px, det_vis = sim.image_truth(occluded=False)
    trk_px, trk_vis = sim.image_truth(occluded=True)
    P = lambda s: f"{prefix}{s}"                                      # noqa: E731
    if cands is None:
        for st in ("linker", "hits", "reconstruct3d", "rally", "shots"):
            run.skip(P(st), "no detections")
        return
    pool.add_frames(P("detections"), M.candidates_to_xy(cands)[0], det_px, det_vis)
    track = run.call("linker", name, cands, meta)
    if track is None:
        for st in ("hits", "reconstruct3d", "rally", "shots"):
            run.skip(P(st), "no track (linker missing/failed)")
        return
    pool.add_frames(P("linker"), track.xy, trk_px, trk_vis, track.interpolated)
    hits = run.call("hits", name, track, meta, camera, players)
    if hits is None:
        for st in ("reconstruct3d", "rally", "shots"):
            run.skip(P(st), "no hits")
        return
    pool.add_events(P("hits"), "hits", [_hit_d(h) for h in hits], [_hit_d(h) for h in sim.hits], off)
    segs = []
    if camera is not None:
        segs = run.call("reconstruct3d", name, track, hits, camera, meta, players)
        if segs is not None:
            pool.add_events(P("reconstruct3d"), "segments", [_seg_d(s) for s in segs],
                            [_true_shot_d(s) for s in sim.shots], off)
        segs = segs or []
    rallies = run.call("rally", name, track, hits, segs, meta, camera, players=players)
    if rallies is None:
        run.skip(P("shots"), "no rallies")
        return
    pool.add_events(P("rally"), "rallies", [_rally_d(r) for r in rallies],
                    [_rally_d(r) for r in sim.rallies], off)
    typed = [run.call("shots", name, r, camera, players) for r in rallies]
    if any(t is None for t in typed):
        return
    pairs = M._match_events(np.array([s.hit.frame for r in typed for s in r.shots], int),
                            np.array([h["frame"] for h in sim.hits], int), 3)
    pred_shots = [s for r in typed for s in r.shots]
    pool.add_list(P("shots"), "type_pred", [pred_shots[i].shot_type for i, _ in pairs])
    pool.add_list(P("shots"), "type_true", [sim.hits[j]["shot_type"] for _, j in pairs])
    pool.add_list(P("shots"), "type_cov", [len(pairs) / max(len(sim.hits), 1)])


def eval_oracle(run: Runner, pool: Pool, name: str, sim: S.SimRally, off: int,
                args: argparse.Namespace, rng: np.random.Generator) -> None:
    meta, cam = sim.meta, sim.camera
    players = S.truth_players(sim, every=args.player_step)
    cands = S.corrupt_detections(sim, rng, args.sigma, args.miss_rate, True, args.fp_rate)
    det_px, det_vis = sim.image_truth(occluded=False)
    trk_px, trk_vis = sim.image_truth(occluded=True)
    pool.add_frames("detections", M.candidates_to_xy(cands)[0], det_px, det_vis)
    track = run.call("linker", name, cands, meta)
    if track is not None:
        pool.add_frames("linker", track.xy, trk_px, trk_vis, track.interpolated)
    ttrack = noisy_truth_track(sim, args.track_noise, rng)
    thits = S.truth_hits(sim)
    hits = run.call("hits", name, ttrack, meta, cam, players)
    if hits is not None:
        pool.add_events("hits", "hits", [_hit_d(h) for h in hits], [_hit_d(h) for h in sim.hits], off)
    segs = run.call("reconstruct3d", name, ttrack, thits, cam, meta, players)
    if segs is not None:
        pool.add_events("reconstruct3d", "segments", [_seg_d(s) for s in segs],
                        [_true_shot_d(s) for s in sim.shots], off)
        call_fn = optional_attr("reconstruct3d", "call_in_out")
        by_frame = {int(s.start_frame): s for s in segs}
        for r in sim.rallies:                       # the call on each rally's final shot
            sh = sim.shots[r.first_shot + r.n_shots - 1]
            seg = min(by_frame.items(), key=lambda kv: abs(kv[0] - sh.frame), default=(None, None))[1]
            if seg is not None and abs(seg.start_frame - sh.frame) > 4:
                seg = None
            pool.add_list("reconstruct3d", "call_pred", [_call(seg, sim.doubles, sh.index == r.first_shot, call_fn)])
            pool.add_list("reconstruct3d", "call_true", [sh.in_out])
            pool.add_list("reconstruct3d", "call_margin", [None if not np.isfinite(sh.margin) else sh.margin])
    tsegs = S.truth_segments(sim)
    rallies = run.call("rally", name, ttrack, thits, tsegs, meta, cam, players=players)
    if rallies is not None:
        pool.add_events("rally", "rallies", [_rally_d(r) for r in rallies],
                        [_rally_d(r) for r in sim.rallies], off)
    for r in S.truth_rallies(sim):
        typed = run.call("shots", name, blank_types(r), cam, players)
        if typed is None:
            break
        pool.add_list("shots", "type_pred", [s.shot_type for s in typed.shots])
        pool.add_list("shots", "type_true", [s.shot_type for s in r.shots])


def _call(seg: Any, doubles: bool, is_serve: bool, call_fn: Callable | None) -> str:
    if seg is None:
        return "unknown"
    z = seg.net_crossing_height
    if z is not None and np.isfinite(z) and z < cm.NET_HEIGHT_CENTRE:
        return "net"
    if call_fn is not None:
        try:
            c = call_fn(seg, doubles, is_serve=is_serve)["call"]
            return c if c in ("in", "out") else "unknown"
        except Exception:
            pass
    if seg.landing_xy is None:
        return "unknown"
    return "in" if cm.in_out_margin(*seg.landing_xy, doubles=doubles) >= 0 else "out"


def make_sim(args: argparse.Namespace, seed: int) -> S.SimRally:
    rng = np.random.default_rng([seed, 7])
    cam = S.random_camera(rng, args.image_size, args.style)
    doubles = args.doubles if args.doubles is not None else bool(rng.random() < 0.3)
    if args.rallies > 1:
        return S.simulate_match(seed, args.rallies, camera=cam, fps=args.fps, doubles=doubles)
    return S.simulate_rally(seed, camera=cam, fps=args.fps, doubles=doubles)


def run_synthetic(args: argparse.Namespace, pool: Pool, fns: dict, missing: dict) -> dict:
    run = Runner(pool, fns, missing)
    info = {"samples": []}
    for k in range(args.n):
        seed = args.seed + k
        t = time.perf_counter()
        sim = make_sim(args, seed)
        pool.seconds["simulate"] += time.perf_counter() - t
        name, off = f"seed{seed}", k * GAP
        rng = np.random.default_rng([seed, 11])
        info["samples"].append({"seed": seed, "frames": sim.n, "shots": len(sim.shots),
                                "rallies": len(sim.rallies), "doubles": sim.doubles})
        if args.stage == "oracle":
            eval_oracle(run, pool, name, sim, off, args, rng)
        elif args.stage == "e2e":
            cands = S.corrupt_detections(sim, rng, args.sigma, args.miss_rate, True, args.fp_rate)
            eval_chain(run, pool, name, sim, off, sim.meta, cands, sim.camera,
                       S.truth_players(sim, every=args.player_step), "")
        elif args.stage == "video":
            eval_video(run, pool, name, sim, off, args)
    return info


def eval_video(run: Runner, pool: Pool, name: str, sim: S.SimRally, off: int,
               args: argparse.Namespace) -> None:
    from app.services.tracking import video_io
    path = f"/tmp/shuttl_bench_{name}.mp4"
    t = time.perf_counter()
    try:
        S.render_video(sim, path)
    except Exception as e:
        pool.error("render", name, e)
        return
    pool.seconds["render"] += time.perf_counter() - t
    try:
        meta = video_io.probe(path)
        pool.add_scalar("render", "frame_count_ok", float(meta.n_frames == sim.n))
        cands = None
        if "detector" in run.fns:
            if "det" not in run.cache:
                run.cache["det"] = run.call("detector", name, weights_dir=args.weights_dir,
                                            device=args.device, kind=args.detector)
            det = run.cache["det"]
            if det is not None:
                t = time.perf_counter()
                try:
                    cands = det.detect_video(path, meta)
                except Exception as e:
                    pool.error("detector", name, e)
                pool.seconds["detector"] += time.perf_counter() - t
        else:
            run.skip("detector", missing_reason(run, "detector"))
        camera = sim.camera
        if "calibration" in run.fns:
            frames = [f for _, f in video_io.sample_frames(path, 8)] if hasattr(video_io, "sample_frames") else []
            cal = run.call("calibration", name, frames, sim.image_size)
            if cal is not None:
                names, P3 = cm.keypoints_array()
                err = np.linalg.norm(cal.project(P3) - sim.camera.project(P3), axis=1)
                pool.add_scalar("calibration", "keypoint_err_px", float(np.nanmean(err)))
                pool.add_scalar("calibration", "success", 1.0)
                camera = cal
            else:
                pool.add_scalar("calibration", "success", 0.0)
        else:
            run.skip("calibration", missing_reason(run, "calibration"))
        players = S.truth_players(sim, every=args.player_step)
        eval_chain(run, pool, name, sim, off, sim.meta if cands is None else meta, cands, camera, players, "")
    finally:
        if not args.keep_videos:
            Path(path).unlink(missing_ok=True)


def missing_reason(run: Runner, st: str) -> str:
    return run.missing.get(st, "not available")


# ── real labelled data (TrackNetV2 Test split) ───────────────────────────────

def real_clips(limit: int | None = None) -> list[dict]:
    """[{match, clip, video, labels, corrected}] sorted by match/clip."""
    out = []
    for vid in sorted((REAL_ROOT / "TrackNetV2/Test").glob("match*/video/*.mp4")):
        match, clip = vid.parent.parent.name, vid.stem
        corr = REAL_ROOT / "corrected" / match / "corrected_csv" / f"{clip}_ball.csv"
        orig = vid.parent.parent / "csv" / f"{clip}_ball.csv"
        lab = corr if corr.exists() else orig
        if lab.exists():
            out.append({"match": match, "clip": clip, "video": str(vid), "labels": str(lab),
                        "corrected": corr.exists()})
    return out[:limit] if limit else out


def load_label_csv(path: str, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Frame,Visibility,X,Y csv → ((n,2) px with NaN when invisible, (n,) visible)."""
    rows = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    xy, vis = np.full((n_frames, 2), np.nan), np.zeros(n_frames, bool)
    f = rows[:, 0].astype(int)
    ok = (f >= 0) & (f < n_frames)
    f, r = f[ok], rows[ok]
    v = r[:, 1] > 0
    vis[f[v]] = True
    xy[f[v]] = r[v, 2:4]
    return xy, vis


def drop_mask(match: str, clip: str, n: int) -> np.ndarray | None:
    """TrackNetV3 corrected/drop_frame.json evaluation range [start, end) or None."""
    p = REAL_ROOT / "corrected" / "drop_frame.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    key = f"{match.replace('match', '')}_{clip}"
    st, en = d.get("start", {}), d.get("end", {})
    if key not in st or key not in en:
        return None
    m = np.zeros(n, bool)
    m[int(st[key]):int(en[key])] = True
    return m


def _cands_to_arr(cands: list[list[ShuttleCandidate]], k: int = 3) -> np.ndarray:
    arr = np.full((len(cands), k, 3), np.nan, np.float32)
    for i, cs in enumerate(cands):
        for j, c in enumerate(cs[:k]):
            arr[i, j] = (c.x, c.y, c.score)
    return arr


def _arr_to_cands(arr: np.ndarray) -> list[list[ShuttleCandidate]]:
    return [[ShuttleCandidate(float(x), float(y), float(s)) for x, y, s in row if np.isfinite(s)]
            for row in arr]


def detector_name(kind: str, weights_dir: str | None = None) -> str:
    """What build_detector(kind=...) resolves to (cache key): 'auto' → env
    SHUTTL_DETECTOR or the first AUTO_ORDER entry whose weights exist, else 'motion'."""
    import os
    if kind != "auto":
        return kind
    env = os.environ.get("SHUTTL_DETECTOR", "auto").strip().lower() or "auto"
    if env != "auto":
        return env
    order, find = optional_attr("detector", "AUTO_ORDER"), optional_attr("detector", "find_weights")
    for k in order or ():
        try:
            if find is not None and find(k, weights_dir) is not None:
                return k
        except Exception:
            continue
    return "motion"


def detect_cached(clip: dict, meta: VideoMeta, args: argparse.Namespace, get_det: Callable,
                  pool: Pool) -> list[list[ShuttleCandidate]] | None:
    name = detector_name(args.detector, args.weights_dir)
    cache = Path(args.cache_dir) / f"{clip['match']}_{clip['clip']}_{name}.npz"
    n = meta.n_frames
    if cache.exists() and not args.no_cache:
        z = np.load(cache)
        if int(z["n_done"]) >= n:
            return _arr_to_cands(z["cands"][:n])
    det = get_det()
    if det is None:
        return None
    t = time.perf_counter()
    try:
        if hasattr(det, "detect_frames"):
            from app.services.tracking.detector import video_source
            src = video_source(clip["video"])

            def limited(step: int, _src=src, _n=n):
                for i, f in _src(step):
                    if i >= _n:
                        break
                    yield i, f
            cands = det.detect_frames(limited, meta.width, meta.height, meta.fps, n)
        else:
            cands = det.detect_video(clip["video"], meta)[:n]
    except Exception as e:
        pool.error("real_detector", clip["clip"], e)
        return None
    finally:
        pool.seconds["real_detector"] += time.perf_counter() - t
    pool.add_scalar("real_detector", "fps", n / max(time.perf_counter() - t, 1e-6))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, cands=_cands_to_arr(cands), n_done=n)
    return cands


def run_real(args: argparse.Namespace, pool: Pool, fns: dict, missing: dict, detector_on: bool,
             oracle_on: bool) -> dict:
    from app.services.tracking import video_io
    run = Runner(pool, fns, missing)
    clips = real_clips(args.clips)
    det_holder: dict[str, Any] = {}

    def get_det():
        if "det" not in det_holder:
            det_holder["det"] = run.call("detector", "build", weights_dir=args.weights_dir,
                                         device=args.device, kind=args.detector)
        return det_holder["det"]

    info = {"clips": []}
    per_clip = []
    for ci, clip in enumerate(clips):
        meta = video_io.probe(clip["video"])
        n = min(meta.n_frames, args.max_frames) if args.max_frames else meta.n_frames
        meta = VideoMeta(fps=meta.fps, width=meta.width, height=meta.height, n_frames=n,
                         timestamps=meta.timestamps[:n], rotation_deg=meta.rotation_deg)
        true_xy, vis = load_label_csv(clip["labels"], n)
        dm = drop_mask(clip["match"], clip["clip"], n)
        mask = np.ones(n, bool) if (dm is None or args.all_frames) else dm
        info["clips"].append({**clip, "frames": n, "visible": int(vis.sum()),
                              "eval_frames": int(mask.sum())})
        size = (meta.width, meta.height)
        row = {"clip": f"{clip['match']}/{clip['clip']}"}
        if detector_on:
            if "detector" not in fns:
                run.skip("real_detector", missing_reason(run, "detector"))
            else:
                cands = detect_cached(clip, meta, args, get_det, pool)
                if cands is not None:
                    top, _ = M.candidates_to_xy(cands, args.score_thresh)
                    pool.add_frames("real_detector", top, true_xy, vis, mask=mask)
                    r = M.detection_metrics(top, true_xy, vis, 4.0, mask=mask, tracknet_int=True,
                                            image_size=size)
                    row["det_f1_tn4"] = r["f1"]
                    pool.add_list("real_detector", "_tn4", [(top, true_xy, vis, mask, size)])
                    track = run.call("linker", clip["clip"], cands, meta)
                    if track is not None:
                        pool.add_frames("real_linker", track.xy, true_xy, vis, track.interpolated, mask)
                        pool.add_list("real_linker", "_tn4", [(track.xy, true_xy, vis, mask, size)])
                        row["link_f1_tn4"] = M.detection_metrics(track.xy, true_xy, vis, 4.0, mask=mask,
                                                                 tracknet_int=True, image_size=size)["f1"]
        if oracle_on:
            rng = np.random.default_rng([args.seed, ci])
            cands = S.corrupt_px(true_xy, vis, size, rng, args.sigma, args.miss_rate, True, args.fp_rate)
            pool.add_frames("real_oracle_detections", M.candidates_to_xy(cands)[0], true_xy, vis, mask=mask)
            track = run.call("linker", clip["clip"], cands, meta)
            if track is not None:
                pool.add_frames("real_oracle_linker", track.xy, true_xy, vis, track.interpolated, mask)
                pool.add_list("real_oracle_linker", "_tn4", [(track.xy, true_xy, vis, mask, size)])
        per_clip.append(row)
    info["per_clip"] = per_clip
    return info


def add_tn4(pool: Pool, results: dict) -> None:
    """TrackNetV3-protocol F1 (4 px @ 512×288, int truncation) pooled over clips."""
    for st, L in pool.lists.items():
        chunks = L.pop("_tn4", None)
        if not chunks:
            continue
        tp = fp1 = fp2 = fn = tn = 0
        for pred, true, vis, mask, size in chunks:
            r = M.detection_metrics(pred, true, vis, 4.0, mask=mask, tracknet_int=True, image_size=size)
            tp, fp1, fp2, fn, tn = tp + r["tp"], fp1 + r["fp1"], fp2 + r["fp2"], fn + r["fn"], tn + r["tn"]
        p, rc = M._ratio(tp, tp + fp1 + fp2), M._ratio(tp, tp + fn)
        results.setdefault(st, {}).update({"tn4_precision": p, "tn4_recall": rc, "tn4_f1": M._f1(p, rc),
                                           "tn4_accuracy": M._ratio(tp + tn, tp + fp1 + fp2 + fn + tn)})


# ── main ─────────────────────────────────────────────────────────────────────

def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    return o


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="oracle",
                    choices=["oracle", "e2e", "video", "real", "real_oracle"])
    g = ap.add_argument_group("synthetic")
    g.add_argument("--n", type=int, default=10, help="number of seeded synthetic samples")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--rallies", type=int, default=1, help="rallies per sample (>1 → simulate_match)")
    g.add_argument("--fps", type=float, default=60.0)
    g.add_argument("--image-size", default="1920x1080")
    g.add_argument("--style", default="random", choices=["random", "behind", "elevated_side", "broadcast"])
    g.add_argument("--doubles", default=None, type=lambda s: s.lower() in ("1", "true", "yes"),
                   help="force singles/doubles (default: 30%% doubles)")
    g.add_argument("--player-step", type=int, default=2, help="truth players every k frames")
    g.add_argument("--track-noise", type=float, default=1.0, help="px noise on the oracle truth track")
    g.add_argument("--keep-videos", action="store_true", help="video mode: keep /tmp renders")
    g = ap.add_argument_group("noise model (corrupt_detections / corrupt_px)")
    g.add_argument("--sigma", type=float, default=1.5)
    g.add_argument("--miss-rate", type=float, default=0.1)
    g.add_argument("--fp-rate", type=float, default=0.03)
    g = ap.add_argument_group("detector / real data")
    g.add_argument("--clips", type=int, default=None, help="real: first N clips (default all 29)")
    g.add_argument("--max-frames", type=int, default=None, help="real: first M frames per clip")
    g.add_argument("--detector", default="auto", help="build_detector kind: auto|wasb|tracknetv3|motion")
    g.add_argument("--device", default=None)
    g.add_argument("--weights-dir", default=None)
    g.add_argument("--score-thresh", type=float, default=0.5, help="top-1 detection threshold")
    g.add_argument("--cache-dir", default="/tmp/shuttl_bench_cache")
    g.add_argument("--no-cache", action="store_true")
    g.add_argument("--no-real-oracle", action="store_true", help="real: skip the real_oracle sub-mode")
    g.add_argument("--all-frames", action="store_true",
                   help="real: evaluate every frame instead of the drop_frame.json range")
    g = ap.add_argument_group("evaluation / output")
    g.add_argument("--tol-px", type=float, default=None,
                   help="pixel tolerance (default: 4 px @ 512x288 scaled to the image width)")
    g.add_argument("--tol-frames", type=int, default=3)
    g.add_argument("--out", default=str(OUT_DIR))
    g.add_argument("--no-json", action="store_true")
    a = ap.parse_args(argv)
    w, h = (int(v) for v in a.image_size.lower().split("x"))
    a.image_size = (w, h)
    return a


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    t0 = time.perf_counter()
    fns, missing = resolve_stages()
    pool = Pool()
    if args.stage in ("real", "real_oracle"):
        tol = args.tol_px if args.tol_px is not None else 10.0
        info = run_real(args, pool, fns, missing, detector_on=args.stage == "real",
                        oracle_on=args.stage == "real_oracle" or not args.no_real_oracle)
    else:
        tol = args.tol_px if args.tol_px is not None else M.scaled_tolerance(4.0, args.image_size)
        info = run_synthetic(args, pool, fns, missing)
    results = pool.results(tol, args.tol_frames)
    if args.stage in ("real", "real_oracle"):
        add_tn4(pool, results)
    if args.stage in ("real", "video"):
        info["detector"] = detector_name(args.detector, args.weights_dir)
    report = {"stage": args.stage, "created": _dt.datetime.now().isoformat(timespec="seconds"),
              "args": {k: v for k, v in vars(args).items()}, "tol_px": tol,
              "available": sorted(fns), "missing": missing, "info": info, "results": results,
              "wall_seconds": round(time.perf_counter() - t0, 2)}
    print(f"== benchmark --stage {args.stage}  (tol {tol:.1f} px, ±{args.tol_frames} frames, "
          f"{report['wall_seconds']} s)")
    keys = M._TABLE_KEYS + ("tn4_f1", "type_coverage", "inout_accuracy", "inout_accuracy_close",
                            "keypoint_err_px_mean", "success_mean", "frame_count_ok_mean", "fps_mean",
                            "n_errors")
    print(M.format_table(results, keys))
    for st, r in results.items():
        for e in r.get("errors", [])[:2]:
            print(f"  [{st}] {e}")
    if args.stage == "real":
        for row in info.get("per_clip", []):
            print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in row.items()))
    if not args.no_json:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{_dt.datetime.now():%Y%m%d_%H%M%S}_{args.stage}.json"
        path.write_text(json.dumps(_jsonable(report), indent=1))
        print(f"wrote {path}")
    return report


if __name__ == "__main__":
    main()
