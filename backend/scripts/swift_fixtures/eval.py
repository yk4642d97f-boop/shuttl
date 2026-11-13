"""Reference fixtures for the Swift ShuttlEval package (metrics.py port + shuttl-bench).

Outputs
  metrics.json        every metrics.py function on hand-built and randomised inputs
                      (NaNs, masks, ties, tracknet_int, >8192-frame pooling) with the
                      exact Python result (dict order and int/float kept), plus numpy
                      probes (sum/mean/median/percentile/norm/interp/linspace/geomspace)
                      the Swift NumPy helpers must reproduce bit for bit.
  bench_<name>.json   a "bench bundle": benchmark.py run in-process with every stage call
                      recorded (per sample: truth, detector / corrupted candidates, each
                      stage's output or exception, pool scalars) + Python's report and
                      printed table. `shuttl-bench --predictions bench_<name>.json` must
                      reproduce the report exactly (stage timings are carried over).
  analysis/<clip>.json  (full real bundles only) MatchAnalysis JSON (meta + linker track)
                      per clip + <clip>.candidates.json — the --analysis-dir input path.

Compact bundles go to Tests/ShuttlEvalTests/Fixtures; the full ones (the baselines:
e2e --n 12, oracle --n 10, real on all 29 clips with tracknetv3 / wasb) to --full-dir.

Run from backend/ (detector candidates must already be cached in /tmp/shuttl_bench_cache,
i.e. `scripts/benchmark.py --stage real` was run once per detector):
  .venv/bin/python -m scripts.swift_fixtures.eval [--only metrics,bench] [--full] [--full-dir DIR]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from app.services.tracking import metrics as M
from app.services.tracking import sim as S
from app.services.tracking import video_io
from app.services.tracking.data_types import HitEvent, ShuttleCandidate, Track2D
from scripts import benchmark as B
from scripts.swift_fixtures.common import SWIFT_PACKAGES, to_jsonable

OUT = SWIFT_PACKAGES / "ShuttlEval/Tests/ShuttlEvalTests/Fixtures"
FULL_DIR = Path("/tmp/shuttl_eval_bench")


def dump_ordered(obj: Any, path: Path) -> Path:
    """Like common.dump but keeps dict insertion order (table / key order is checked)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, allow_nan=False, separators=(",", ":"))
    return path


# ── metrics.json ─────────────────────────────────────────────────────────────

def _rows(a: np.ndarray) -> list:
    return [[None if not np.isfinite(v) else float(v) for v in r] for r in np.asarray(a, float)]


def metric_cases() -> dict:
    rng = np.random.default_rng(20260927)
    cases: list[dict] = []

    def add(fn: str, args: dict, out: Any) -> None:
        cases.append({"fn": fn, "args": args, "out": out})

    # --- detection / track: test_metrics._truth + edits
    t = np.arange(50, dtype=float)
    xy = np.column_stack([100 + 5 * t, 300 - 0.2 * (t - 25) ** 2])
    vis = np.ones(50, bool)
    vis[[3, 4, 30]] = False
    xy[~vis] = np.nan
    p = xy.copy()
    p[10] += [20.0, 0.0]; p[11] = np.nan; p[3] = [5.0, 5.0]; p[12] += [3.0, 4.0]
    m = np.zeros(50, bool); m[20:40] = True

    def det(pred, true, vis=None, tol=10.0, mask=None, tn=False, size=None):
        r = M.detection_metrics(pred, true, vis, tol, mask=mask, tracknet_int=tn, image_size=size)
        add("detection", {"pred": _rows(pred), "true": _rows(true),
                          "vis": None if vis is None else [bool(v) for v in vis], "tol_px": tol,
                          "mask": None if mask is None else [bool(v) for v in mask],
                          "tracknet_int": tn, "image_size": None if size is None else list(size)}, r)

    det(xy, xy, vis)
    det(p, xy, vis, 10.0)
    det(p, xy, vis, 4.0)
    det(p, xy, vis, 10.0, m)
    det(np.array([[649.9, 360.0]]), np.array([[640.0, 360.0]]), None, 4.0, tn=True, size=(1280, 720))
    det(np.array([[652.6, 360.0]]), np.array([[640.0, 360.0]]), None, 4.0, tn=True, size=(1280, 720))
    for n, size in ((300, (1280, 720)), (8300, (1920, 1080))):     # > 8192: pooled-sum chunking
        tr = np.column_stack([rng.uniform(0, size[0], n), rng.uniform(0, size[1], n)])
        tv = rng.random(n) < 0.85
        tr[rng.random(n) < 0.05] = np.nan
        pr = tr + rng.normal(0, 3.0, tr.shape)
        pr[rng.random(n) < 0.08] = np.nan
        far = rng.random(n) < 0.04
        pr[far] += rng.uniform(-200, 200, (int(far.sum()), 2))
        ghost = ~tv & (rng.random(n) < 0.3)
        pr[ghost] = rng.uniform(0, 500, (int(ghost.sum()), 2))
        mk = rng.random(n) < 0.9
        det(pr, tr, tv, 10.0, mk)
        if n == 300:
            det(pr, tr, tv, 4.0, mk, tn=True, size=size)
            det(pr, tr, None, 15.0)

    def trk(pred, true, vis=None, tol=10.0, mask=None, interp=None):
        if interp is not None:
            obj = Track2D(t=np.arange(len(pred), dtype=float), xy=np.asarray(pred, float),
                          conf=np.ones(len(pred)), visible=np.isfinite(np.asarray(pred)[:, 0]),
                          interpolated=np.asarray(interp, bool))
        else:
            obj = np.asarray(pred, float)
        r = M.track_metrics(obj, true, vis, tol, mask=mask)
        add("track", {"pred": _rows(pred), "true": _rows(true),
                      "vis": None if vis is None else [bool(v) for v in vis], "tol_px": tol,
                      "mask": None if mask is None else [bool(v) for v in mask],
                      "interpolated": None if interp is None else [bool(v) for v in interp]}, r)

    trk(xy, xy, vis, interp=np.zeros(50, bool))
    p2 = xy.copy(); p2[20:23] += [200.0, 0.0]; p2[40] += [0.0, 1.0]
    trk(p2, xy, vis, 10.0)
    filled = xy.copy(); filled[[3, 4]] = [[115, 200], [120, 200]]
    trk(filled, xy, vis, interp=np.isin(np.arange(50), [3, 4]))
    n = 400
    tr = np.cumsum(rng.normal(0, 6, (n, 2)), axis=0) + [640, 360]
    tv = rng.random(n) < 0.9
    tr[~tv] = np.nan
    pr = tr + rng.normal(0, 1.5, tr.shape)
    gap = np.zeros(n, bool); gap[rng.integers(0, n, 40)] = True
    pr[gap] = np.nan
    ip = rng.random(n) < 0.1
    pr[ip & ~tv] = rng.uniform(0, 600, (int((ip & ~tv).sum()), 2))
    pr[ip & tv] = tr[ip & tv] + rng.normal(0, 8, (int((ip & tv).sum()), 2))
    pr[150:160] += 120.0
    trk(pr, tr, tv, 10.0, rng.random(n) < 0.95, interp=ip)
    trk(pr, tr, tv, 15.0)
    trk(pr[:2], tr[:2], tv[:2], 10.0)

    # --- candidates_to_xy (ties: first maximum; NaN scores)
    C = ShuttleCandidate
    cand = [[C(1, 2, 0.9), C(5, 5, 0.95)], [], [C(3, 3, 0.3)], [C(7, 7, 0.8), C(9, 9, 0.8)],
            [C(1, 1, float("nan")), C(2, 2, 0.7)], [C(4, 4, 0.5)]]
    for thr in (0.5, 0.3, 0.9):
        xy_, sc = M.candidates_to_xy(cand, thr)
        add("candidates_to_xy", {"cands": [[[c.x, c.y, c.score] for c in cs] for cs in cand], "thresh": thr},
            {"xy": xy_, "score": sc})
    r = M.detection_metrics(cand[:3], np.array([[5, 5], [np.nan, np.nan], [3, 3]], float))
    add("detection_cands", {"cands": [[[c.x, c.y, c.score] for c in cs] for cs in cand[:3]],
                            "true": [[5.0, 5.0], [None, None], [3.0, 3.0]]}, r)

    # --- hits
    true_h = [{"frame": 10, "hitter": "near"}, {"frame": 50, "hitter": "far"}, {"frame": 90, "hitter": "near"}]
    perfect = [HitEvent(frame=h["frame"], t=h["frame"] / 30, xy_px=(0, 0), hitter=h["hitter"]) for h in true_h]

    def hits(pred, true, tol=3):
        pd = [{"frame": int(M._get(h, "frame")), "hitter": M._get(h, "hitter", "unknown")} for h in pred]
        add("hits", {"pred": pd, "true": true, "tol_frames": tol}, M.hit_metrics(pred, true, tol))

    hits(perfect, true_h)
    hits([{"frame": 12, "hitter": "near"}, {"frame": 11, "hitter": "far"},
          {"frame": 49, "hitter": "unknown"}, {"frame": 70, "hitter": "far"}], true_h)
    hits([], true_h); hits([], []); hits([{"frame": 3}], [])
    tf = np.sort(rng.choice(5000, 120, replace=False))
    th = [{"frame": int(f), "hitter": ["near", "far"][k % 2]} for k, f in enumerate(tf)]
    ph = []
    for k, f in enumerate(tf):
        if rng.random() < 0.85:
            ph.append({"frame": int(f + rng.integers(-4, 5)),
                       "hitter": ["near", "far", "unknown"][int(rng.choice(3, p=[0.45, 0.45, 0.1]))]})
    ph += [{"frame": int(f), "hitter": "near"} for f in rng.integers(0, 5000, 15)]
    ph.sort(key=lambda d: d["frame"])
    for tol in (3, 1, 0):
        hits(ph, th, tol)

    # --- segments / landing
    shots = [{"start_frame": 10, "p0": [-5.0, 0, 2.8], "v0": [80.0, 0, -10], "net_crossing_height": 2.0,
              "landing_xy": [3.0, 1.0]},
             {"start_frame": 60, "p0": [4.0, 1, 1.0], "v0": [-20.0, 0, 5], "net_crossing_height": 1.8,
              "landing_xy": None}]

    def seg(pred, true, tol=4):
        conv = lambda L: [dict(d, p0=np.asarray(d["p0"], float), v0=np.asarray(d["v0"], float),
                               landing_xy=None if d.get("landing_xy") is None else tuple(d["landing_xy"]))
                          for d in L]
        add("segments", {"pred": pred, "true": true, "tol_frames": tol}, M.segment_metrics(conv(pred), conv(true), tol))

    seg(shots, shots)
    seg([dict(shots[0], v0=[88.0, 0, -11.0], landing_xy=[3.3, 1.4], start_frame=12)], shots)
    segs_t, segs_p = [], []
    for k in range(60):
        p0 = [float(rng.uniform(-6, 6)), float(rng.uniform(-3, 3)), float(rng.uniform(0.3, 3.1))]
        v0 = [float(v) for v in rng.normal(0, 25, 3)]
        land = None if rng.random() < 0.3 else [float(rng.uniform(-6.7, 6.7)), float(rng.uniform(-3, 3))]
        znet = None if rng.random() < 0.2 else float(rng.uniform(1.5, 3.0))
        segs_t.append({"start_frame": int(k * 40), "p0": p0, "v0": v0, "net_crossing_height": znet, "landing_xy": land})
        if rng.random() < 0.85:
            segs_p.append({"start_frame": int(k * 40 + rng.integers(-5, 6)),
                           "p0": [float(a + rng.normal(0, 0.2)) for a in p0],
                           "v0": [float(a * rng.uniform(0.8, 1.2)) for a in v0],
                           "net_crossing_height": None if rng.random() < 0.2 else float(rng.uniform(1.5, 3.0)),
                           "landing_xy": None if rng.random() < 0.2 else
                           [float(rng.uniform(-6.7, 6.7)), float(rng.uniform(-3, 3))]})
    segs_t[5]["v0"] = [0.0, 0.0, 0.0]                  # ts_kmh = 0 → max(…, 1e-6)
    seg(segs_p, segs_t); seg(segs_p, segs_t, 2)

    def land(pred, true):
        add("landing", {"pred": pred, "true": true},
            M.landing_metrics([None if v is None else np.array(v, float) for v in pred],
                              [None if v is None else np.array(v, float) for v in true]))

    land([[0, 0], None], [[0, 1], [2, 2]])
    land([[0.1, 0.2], [None, 1.0], [3.0, 4.0]], [[0.0, 0.0], [1.0, 1.0], [2.5, 3.5]])
    land([[float(a), float(b)] for a, b in rng.normal(0, 3, (40, 2))],
         [[float(a), float(b)] for a, b in rng.normal(0, 3, (40, 2))])

    # --- in/out, shot types, confusion
    def io(pred, true, margins, close=0.2):
        add("in_out", {"pred": pred, "true": true, "margins": margins, "close_m": close},
            M.in_out_metrics(pred, true, margins, close))

    io(["in", "out", "net", "unsure"], ["in", "in", "net", "out"], [0.05, 0.1, None, -0.5])
    io(["in", "out"], ["in", "out"], None)
    io(["unknown", "x"], ["in", "out"], [0.1, 0.3])
    io([], [], [])
    lab = ["in", "out", "net", "unknown", "unsure"]
    pc = [lab[int(i)] for i in rng.integers(0, 5, 80)]
    tc = [["in", "out", "net"][int(i)] for i in rng.integers(0, 3, 80)]
    mg = [None if rng.random() < 0.2 else float(rng.normal(0, 0.3)) for _ in range(80)]
    io(pc, tc, mg); io(pc, tc, mg, 0.1)

    def st(pred, true, labels=None):
        add("shot_type", {"pred": pred, "true": true, "labels": labels}, M.shot_type_metrics(pred, true, labels))

    st(["smash", "clear", "drop"], ["smash", "clear", "net"])
    st(["a", "b"], ["a", "b"]); st([], [])
    kinds = list(S.SHOT_TYPES) + ["unknown", "Zed", "lift_2"]
    st([kinds[int(i)] for i in rng.integers(0, len(kinds), 90)], [kinds[int(i)] for i in rng.integers(0, 10, 90)])
    st(["smash", "clear"], ["smash", "net"], ["smash", "net", "clear", "drop"])
    for pr_, tr_, lb in ((["b", "a", "B", "_x"], ["a", "c", "Ä", "b"], None), (["x"], ["y"], ["y"])):
        add("confusion", {"pred": pr_, "true": tr_, "labels": lb}, M.confusion_matrix(pr_, tr_, lb))

    # --- rallies
    true_r = [{"start_frame": 0, "end_frame": 100, "end_reason": "in", "winner": "near",
               "landing_xy": [1.0, 2.0], "margin": 0.1},
              {"start_frame": 300, "end_frame": 400, "end_reason": "net", "winner": "far",
               "landing_xy": None, "margin": None}]

    def ral(pred, true, thr=0.5):
        conv = lambda L: [dict(d, landing_xy=None if d.get("landing_xy") is None else tuple(d["landing_xy"])) for d in L]
        add("rally", {"pred": pred, "true": true, "iou_thresh": thr}, M.rally_metrics(conv(pred), conv(true), thr))

    ral(true_r, true_r)
    ral([dict(true_r[0], start_frame=10, end_reason="out", winner="far"),
         {"start_frame": 150, "end_frame": 200, "end_reason": "in", "winner": "near"}], true_r)
    ral([], true_r); ral([], [])
    ral([{"start_frame": 0, "end_frame": 100}, {"start_frame": 0, "end_frame": 100},
         {"start_frame": 50, "end_frame": 150}], [{"start_frame": 0, "end_frame": 100, "in_out_margin_m": 0.05,
                                                   "end_reason": "in"}, {"start_frame": 50, "end_frame": 150,
                                                                         "margin": None, "in_out_margin_m": 0.01}])
    tr_r, pr_r = [], []
    f = 0
    for k in range(30):
        a = f + int(rng.integers(50, 400)); b = a + int(rng.integers(100, 900)); f = b
        tr_r.append({"start_frame": a, "end_frame": b, "end_reason": ["in", "out", "net"][k % 3],
                     "winner": ["near", "far"][int(rng.integers(2))],
                     "landing_xy": None if k % 3 == 2 else [float(rng.uniform(-7, 7)), float(rng.uniform(-3, 3))],
                     "margin": None if k % 3 == 2 else float(rng.normal(0, 0.3))})
        if rng.random() < 0.85:
            pr_r.append({"start_frame": a + int(rng.integers(-60, 60)), "end_frame": b + int(rng.integers(-90, 60)),
                         "end_reason": ["in", "out", "net", "unknown"][int(rng.integers(4))],
                         "winner": ["near", "far", "unknown"][int(rng.integers(3))],
                         "landing_xy": None if rng.random() < 0.3 else [float(rng.uniform(-7, 7)), float(rng.uniform(-3, 3))]})
        if rng.random() < 0.15:
            pr_r.append({"start_frame": b - 30, "end_frame": b + 40, "end_reason": "in", "winner": "near"})
    for thr in (0.5, 0.3, 0.8):
        ral(pr_r, tr_r, thr)

    # --- table / tolerance
    res = {"linker": {"f1": 0.9, "jumps": 2, "flag": True, "x": float("nan"), "n": np.int64(3)},
           "hits": {"skipped": "no module"}, "raw": 3.5, "e": {}, "c": {"precision": 1 / 3, "err_p95": 2.0005}}
    add("format_table", {"results": {k: v for k, v in res.items()}},
        {"text": M.format_table(res), "text_keys": M.format_table(res, ("n", "f1", "flag", "precision"))})
    add("scaled_tolerance", {"cases": [[4.0, 1280, 720], [4.0, 1920, 1080], [2.5, 1000, 500]]},
        {"values": [M.scaled_tolerance(a, (w, h)) for a, w, h in [[4.0, 1280, 720], [4.0, 1920, 1080], [2.5, 1000, 500]]]})
    obj = {"stage": "real", "path": "/Users/x/华北五省/a.mp4", "ints": [1, -2, 0], "floats": [0.1, 1e-05, 1e16, 2.5e-07,
           123456789.125, -0.0, 1.0], "none": None, "t": True, "f": False, "empty_l": [], "empty_d": {},
           "nested": {"b": [{"x": 1}, []], "a": "q\"uote\\ \n\t\u0001 é 😀"}}
    js = {"obj": obj, "dumps": json.dumps(obj), "dumps_indent1": json.dumps(obj, indent=1)}
    return {"cases": cases, "numpy": numpy_probes(rng), "json": js}


def numpy_probes(rng: np.random.Generator) -> dict:
    out = {"sum": [], "percentile": [], "nan": [], "norm": [], "interp": None, "grids": []}
    for n in (1, 2, 5, 8, 9, 16, 17, 127, 128, 129, 1000, 8193, 17000):
        x = rng.normal(size=n) * rng.uniform(1, 1e3)
        out["sum"].append({"x": x, "sum": float(np.sum(x)), "mean": float(np.mean(x)),
                           "median": float(np.median(x))})
    for n in (1, 2, 3, 4, 7, 20, 21, 100, 101):
        x = rng.normal(size=n)
        out["percentile"].append({"x": x, "q": [0, 5, 25, 50, 75, 90, 95, 99, 100],
                                  "p": [float(np.percentile(x, q)) for q in (0, 5, 25, 50, 75, 90, 95, 99, 100)]})
    for n in (5, 12, 700):
        x = rng.normal(size=n); x[rng.random(n) < 0.2] = np.nan
        out["nan"].append({"x": x, "nanmean": float(np.nanmean(x)), "nanmedian": float(np.nanmedian(x))})
    V = rng.normal(size=(400, 3)) * rng.uniform(0.01, 100, (400, 1))
    out["norm"] = {"v": V, "norm3": [float(np.linalg.norm(v)) for v in V],
                   "norm2": [float(np.linalg.norm(v[:2])) for v in V],
                   "row2": np.linalg.norm(V[:, :2], axis=1), "row3": np.linalg.norm(V, axis=1),
                   "hypot": [float(np.hypot(a, b)) for a, b, _ in V]}
    xp = np.sort(rng.uniform(0, 10, 40)); fp = rng.normal(size=40)
    probe = np.concatenate([rng.uniform(-1, 11, 300), xp[:5], [xp[-1], xp[0]]])
    out["interp"] = {"xp": xp, "fp": fp, "x": probe, "y": np.interp(probe, xp, fp)}
    for a, b, k in ((-60, 45, 106), (-60, 45, 211), (8.0, 80.0, 30), (-35.0, 82.0, 30), (-60, 40, 101), (0.0, 1.0, 13)):
        out["grids"].append({"kind": "linspace_rad", "a": a, "b": b, "n": k, "v": np.radians(np.linspace(a, b, k))})
    for rel in (0.04, 0.003):
        out["grids"].append({"kind": "linspace", "a": -rel, "b": rel, "n": 13, "v": np.linspace(-rel, rel, 13)})
    out["grids"].append({"kind": "geomspace", "a": 2.5, "b": 120.0, "n": 36, "v": np.geomspace(2.5, 120.0, 36)})
    out["grids"].append({"kind": "arange", "a": 0.0, "b": 6.0, "n": 0, "step": 1 / 120, "v": np.arange(0.0, 6.0, 1 / 120)})
    out["grids"].append({"kind": "arange", "a": 0.0, "b": 3.0, "n": 0, "step": 1 / 120, "v": np.arange(0.0, 3.0, 1 / 120)})
    th = rng.uniform(-3, 3, 50)
    out["grids"].append({"kind": "degrees", "a": 0, "b": 0, "n": 50, "x": th, "v": np.degrees(th)})
    out["grids"].append({"kind": "radians", "a": 0, "b": 0, "n": 50, "x": th * 50, "v": np.radians(th * 50)})
    return out


# ── bench bundles ────────────────────────────────────────────────────────────

def _cands(cands) -> list:
    return [[[float(c.x), float(c.y), float(c.score)] for c in cs] for cs in cands]


def _track(tr: Track2D) -> dict:
    return {"xy": np.asarray(tr.xy, float), "visible": np.asarray(tr.visible, bool),
            "interpolated": np.asarray(tr.interpolated, bool)}


def _seg(s) -> dict:
    d = to_jsonable(s)
    d["frames"], d["positions"] = [], []           # not used by scoring (large)
    return d


def _rally(r) -> dict:
    from dataclasses import replace
    d = to_jsonable(replace(r, shots=[]))
    d["shots"] = [{"hit": to_jsonable(s.hit), "segment": None, "shot_type": s.shot_type} for s in r.shots]
    return d


def sim_truth(sim: S.SimRally, stage: str) -> dict:
    det_px, det_vis = sim.image_truth(occluded=False)
    trk_px, trk_vis = sim.image_truth(occluded=True)
    assert np.all(det_vis <= trk_vis) and np.array_equal(det_px[det_vis], trk_px[det_vis])
    w, h = sim.image_size
    out = {"seed": sim.seed, "n": sim.n, "fps": sim.fps, "width": int(w), "height": int(h),
           "camera": sim.camera.to_dict(),
           "doubles": bool(sim.doubles), "n_shots": len(sim.shots), "n_rallies": len(sim.rallies),
           "trk_px": trk_px, "trk_vis": trk_vis, "det_vis": det_vis,
           "hits": [{"frame": int(x["frame"]), "hitter": x["hitter"], "shot_type": x["shot_type"]} for x in sim.hits],
           "shots": [B._true_shot_d(sh) for sh in sim.shots],
           "rallies": [B._rally_d(r) for r in sim.rallies]}
    if stage == "oracle":
        fc = []
        for r in sim.rallies:
            sh = sim.shots[r.first_shot + r.n_shots - 1]
            fc.append({"frame": int(sh.frame), "is_serve": bool(sh.index == r.first_shot), "in_out": sh.in_out,
                       "margin": None if not np.isfinite(sh.margin) else float(sh.margin)})
        out["final_calls"] = fc
        out["rally_types"] = [[s.shot_type for s in r.shots] for r in S.truth_rallies(sim)]
    return out


class Recorder:
    """Patches benchmark.py / sim.py entry points to record one benchmark run."""

    def __init__(self) -> None:
        self.samples: list[dict] = []
        self.cur: dict | None = None
        self.pool = None
        self.depth = 0
        self.stage = ""
        self.call_fn = B.optional_attr("reconstruct3d", "call_in_out")
        self._labels = None
        self._probe: dict[str, Any] = {}
        self._saved: list[tuple[Any, str, Any]] = []

    def _patch(self, obj, name, fn) -> None:
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, fn)

    def restore(self) -> None:
        for obj, name, fn in reversed(self._saved):
            setattr(obj, name, fn)
        self._saved.clear()

    def new(self, d: dict) -> None:
        d.setdefault("stages", {})
        d.setdefault("scalars", [])
        self.samples.append(d)
        self.cur = d

    def install(self) -> None:
        rec = self
        orig = {k: getattr(B, k) for k in ("make_sim", "detect_cached", "load_label_csv", "drop_mask")}
        orig_cd, orig_cp, orig_probe = S.corrupt_detections, S.corrupt_px, video_io.probe
        orig_call, orig_scalar = B.Runner.call, B.Pool.add_scalar

        def make_sim(args, seed):
            sim = orig["make_sim"](args, seed)
            rec.new({"kind": "sim", "seed": int(seed), "truth": sim_truth(sim, args.stage)})
            rec.cur["_sim"] = sim
            return sim

        def corrupt_detections(*a, **kw):
            rec.depth += 1
            try:
                out = orig_cd(*a, **kw)
            finally:
                rec.depth -= 1
            if rec.cur is not None:
                rec.cur["candidates"] = _cands(out)
            return out

        def corrupt_px(*a, **kw):
            rec.depth += 1
            try:
                out = orig_cp(*a, **kw)
            finally:
                rec.depth -= 1
            if rec.depth == 0 and rec.cur is not None:        # real_oracle (not inside corrupt_detections)
                rec.cur["oracle_candidates"] = _cands(out)
            return out

        def probe(path):
            m = orig_probe(path)
            rec._probe[str(path)] = m
            return m

        def load_label_csv(path, n):
            out = orig["load_label_csv"](path, n)
            rec._labels = (path, n, out)
            return out

        def drop_mask(match, clip, n):
            dm = orig["drop_mask"](match, clip, n)
            path, n_, (xy, vis) = rec._labels
            vid = str(B.REAL_ROOT / "TrackNetV2/Test" / match / "video" / f"{clip}.mp4")
            meta = rec._probe.get(vid)
            rec.new({"kind": "real", "match": match, "clip": clip, "n": int(n), "labels": str(path),
                     "fps": None if meta is None else float(meta.fps),
                     "width": None if meta is None else int(meta.width),
                     "height": None if meta is None else int(meta.height),
                     "truth_xy": xy, "truth_vis": vis, "drop_mask": None if dm is None else dm})
            return dm

        def detect_cached(clip, meta, args, get_det, pool):
            before = len(pool.errors.get("real_detector", []))
            out = orig["detect_cached"](clip, meta, args, get_det, pool)
            errs = pool.errors.get("real_detector", [])
            rec.cur["candidates"] = None if out is None else _cands(out)
            rec.cur["detector_error"] = errs[before] if len(errs) > before else None
            rec.cur["_cands_obj"] = out
            return out

        def add_scalar(self_, stage, key, value):
            if rec.cur is not None:
                rec.cur["scalars"].append([stage, key, float(value)])
            return orig_scalar(self_, stage, key, value)

        def call(self_, stage, sample, *a, **kw):
            rec.pool = self_.pool
            before = len(self_.pool.errors.get(stage, []))
            present = stage in self_.fns
            out = orig_call(self_, stage, sample, *a, **kw)
            errs = self_.pool.errors.get(stage, [])
            entry = {"present": present, "error": errs[before] if len(errs) > before else None}
            if out is None:
                entry["out"] = None
            elif stage == "linker":
                entry["out"] = _track(out)
                if rec.cur.get("kind") == "real" and a and a[0] is rec.cur.get("_cands_obj"):
                    rec.cur["_track_full"] = out
            elif stage == "hits":
                entry["out"] = [to_jsonable(h) for h in out]
            elif stage == "reconstruct3d":
                entry["out"] = [_seg(s) for s in out]
                if rec.stage == "oracle" and rec.call_fn is not None:
                    dbl = rec.cur["_sim"].doubles
                    for d, s in zip(entry["out"], out):
                        d["calls"] = {k: _safe_call(rec.call_fn, s, dbl, v) for k, v in (("serve", True), ("rally", False))}
            elif stage in ("rally", "shots"):
                entry["out"] = [_rally(r) for r in out] if stage == "rally" else _rally(out)
            else:
                entry["out"] = "<object>"
            if rec.cur is not None:
                rec.cur["stages"].setdefault(stage, []).append(entry)
            return out

        self._patch(B, "make_sim", make_sim)
        self._patch(B, "detect_cached", detect_cached)
        self._patch(B, "load_label_csv", load_label_csv)
        self._patch(B, "drop_mask", drop_mask)
        self._patch(S, "corrupt_detections", corrupt_detections)
        self._patch(S, "corrupt_px", corrupt_px)
        self._patch(video_io, "probe", probe)
        self._patch(B.Runner, "call", call)
        self._patch(B.Pool, "add_scalar", add_scalar)
        orig_results = B.Pool.results

        def results(self_, *a, **kw):
            rec.pool = self_
            return orig_results(self_, *a, **kw)
        self._patch(B.Pool, "results", results)


def _safe_call(fn, seg, doubles: bool, is_serve: bool):
    """Raw call_in_out(...)['call'] or None when it raises (benchmark `_call` then falls back)."""
    try:
        return str(fn(seg, doubles, is_serve=is_serve)["call"])
    except Exception:
        return None


def record_bench(argv: list[str]) -> tuple[dict, list[dict]]:
    """Run benchmark.main(argv) with recording → (bundle, raw samples)."""
    rec = Recorder()
    args = B.parse_args(argv)
    rec.stage = args.stage
    if args.stage in ("real",):
        det = B.detector_name(args.detector, args.weights_dir)
        for c in B.real_clips(args.clips):
            p = Path(args.cache_dir) / f"{c['match']}_{c['clip']}_{det}.npz"
            if not p.exists():
                sys.exit(f"missing detector cache {p}: run scripts/benchmark.py --stage real --detector {det} first")
    rec.install()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            report = B.main(argv)
    finally:
        rec.restore()
    text = buf.getvalue()
    sys.stdout.write(text)
    samples = []
    for s in rec.samples:
        samples.append({k: v for k, v in s.items() if not k.startswith("_")})
    bundle = {"format": "shuttl-bench-bundle/1", "argv": argv, "stage": args.stage,
              "has_call_in_out": rec.call_fn is not None,
              "pool_seconds": {k: float(v) for k, v in rec.pool.seconds.items()},
              "report": B._jsonable(report), "stdout": text, "samples": samples}
    return bundle, rec.samples


def export_analysis_dir(raw: list[dict], out_dir: Path) -> int:
    """MatchAnalysis JSON (meta + detector-linker track) + candidates per real clip."""
    n = 0
    for s in raw:
        tr = s.get("_track_full")
        if s.get("kind") != "real" or tr is None:
            continue
        stem = f"{s['match']}_{s['clip']}"
        meta = {"fps": s["fps"], "width": s["width"], "height": s["height"], "n_frames": s["n"],
                "timestamps": np.asarray(tr.t, float)[: s["n"]], "rotation_deg": 0}
        ma = {"meta": meta, "track": {"t": tr.t, "xy": tr.xy, "conf": tr.conf, "visible": tr.visible,
                                      "interpolated": tr.interpolated},
              "camera": None, "players": [], "hits": [], "segments": [], "rallies": [], "score": [],
              "diagnostics": {"source": "scripts.swift_fixtures.eval (Python linker)"},
              "processing_time_seconds": 0.0}
        dump_ordered(ma, out_dir / f"{stem}.json")
        dump_ordered({"candidates": s["candidates"]}, out_dir / f"{stem}.candidates.json")
        n += 1
    return n


# ── sim.py replay fixtures ───────────────────────────────────────────────────

class RecGen:
    """numpy Generator proxy recording every draw sim.py makes (method, args, result)."""

    def __init__(self, gen: np.random.Generator, log: list) -> None:
        self._g, self._log = gen, log

    def _rec(self, m, a, r, n=None, **extra):
        d = {"m": m, "a": [float(v) for v in a], "r": r}
        if n is not None:
            d["n"] = int(n)
        d.update(extra)
        self._log.append(d)

    @staticmethod
    def _n(size):
        return None if size is None else int(np.prod(size))

    def random(self, size=None):
        r = self._g.random(size)
        self._rec("random", [], np.asarray(r).ravel().tolist() if size is not None else float(r), self._n(size))
        return r

    def uniform(self, low=0.0, high=1.0, size=None):
        r = self._g.uniform(low, high, size)
        self._rec("uniform", [low, high], np.asarray(r).ravel().tolist() if size is not None else float(r), self._n(size))
        return r

    def normal(self, loc=0.0, scale=1.0, size=None):
        r = self._g.normal(loc, scale, size)
        self._rec("normal", [loc, scale], np.asarray(r).ravel().tolist() if size is not None else float(r), self._n(size))
        return r

    def triangular(self, left, mode, right):
        r = self._g.triangular(left, mode, right)
        self._rec("triangular", [left, mode, right], float(r))
        return r

    def geometric(self, p):
        r = self._g.geometric(p)
        self._rec("geometric", [p], int(r))
        return r

    def integers(self, low, high=None):
        lo, hi = (0, low) if high is None else (low, high)
        r = self._g.integers(lo, hi)
        self._rec("integers", [lo, hi], int(r))
        return int(r)

    def choice(self, a, p=None):
        n = int(a) if np.isscalar(a) and not isinstance(a, str) else len(a)
        idx = int(self._g.choice(n, p=p))
        extra = {}
        if p is not None:
            pa = np.asarray(p, float)
            if len(pa) <= 16:
                extra["p"] = pa.tolist()
            else:
                extra["ps"] = [float(np.sum(np.arange(len(pa)) * pa)), float(np.sum(pa * pa))]
        self._rec("choice", [n], idx, **extra)
        if np.isscalar(a) and not isinstance(a, str):
            return np.int64(idx)
        return np.asarray(a)[idx]

    def __getattr__(self, name):
        raise AttributeError(f"RecGen: sim.py called unrecorded Generator.{name}")


@contextlib.contextmanager
def recording_rng(streams: list):
    """Patch np.random.default_rng so every stream sim.py creates is recorded."""
    orig = np.random.default_rng

    def fake(seed=None):
        if isinstance(seed, (RecGen, np.random.Generator)):
            return seed
        key = json.dumps(seed if not isinstance(seed, np.ndarray) else seed.tolist())
        log: list = []
        streams.append({"key": key, "draws": log})
        return RecGen(orig(seed), log)

    np.random.default_rng = fake
    try:
        yield
    finally:
        np.random.default_rng = orig


def scenario_dict(sim: S.SimRally) -> dict:
    return {
        "n": sim.n, "fps": sim.fps, "seed": sim.seed, "doubles": bool(sim.doubles), "drag_length": sim.drag_length,
        "camera": sim.camera.to_dict(), "shuttle_world": sim.shuttle_world, "shuttle_px": sim.shuttle_px,
        "visible": sim.visible, "in_play": sim.in_play, "occluded": sim.occluded,
        "hits": [{"frame": int(h["frame"]), "t": h["t"], "hitter": h["hitter"], "player": h["player"],
                  "pos": h["pos"], "shot_type": h["shot_type"], "shot": int(h["shot"])} for h in sim.hits],
        "shots": [to_jsonable(sh) for sh in sim.shots],
        "rallies": [to_jsonable(r) for r in sim.rallies],
        "players": [{"key": k, "side": p["side"], "height": p["height"], "racket_side": p["racket_side"],
                     "foot": p["foot"], "bbox": p["bbox"], "racket": p["racket"], "depth": p["depth"]}
                    for k, p in sim.players.items()],
        "resting": [{"from": int(a), "to": int(b), "p": p} for a, b, p in sim.resting],
    }


SIM_CASES = [   # name, kind, kwargs, camera (style, seed, size) or None (sim draws its own)
    ("rally_a", "rally", {"seed": 8, "fps": 30.0}, ("broadcast", 1, (1280, 720))),
    ("rally_b", "rally", {"seed": 3, "fps": 30.0, "n_shots": 5, "doubles": True}, ("behind", 2, (1920, 1080))),
    ("rally_c", "rally", {"seed": 11, "fps": 30.0}, None),
    ("match_d", "match", {"seed": 7, "n_rallies": 2, "fps": 30.0, "n_shots": 3}, ("elevated_side", 4, (1280, 720))),
    ("rally_e", "rally", {"seed": 104, "fps": 60.0, "n_shots": 1}, ("broadcast", 3, (1280, 720))),
]


def sim_cases() -> dict:
    out = {"scenarios": [], "corrupt": [], "cameras": [], "libm": libm_probes()}
    sims = {}
    for name, kind, kw, cam in SIM_CASES:
        camera = None if cam is None else S.random_camera(np.random.default_rng(cam[1]), cam[2], cam[0])
        streams: list = []
        with recording_rng(streams):
            sim = S.simulate_rally(camera=camera, **kw) if kind == "rally" else S.simulate_match(camera=camera, **kw)
        sims[name] = sim
        out["scenarios"].append({"name": name, "kind": kind, "args": kw,
                                 "camera": None if camera is None else camera.to_dict(),
                                 "streams": streams, "scenario": scenario_dict(sim)})
    for name, of, fn, kw in (
            ("det_a", "rally_a", "detections", {}),
            ("det_d", "match_d", "detections", {"sigma_px": 2.0, "miss_rate": 0.2, "false_positive_rate": 0.08,
                                                "dead_shuttle_rate": 0.7}),
            ("clean_e", "rally_e", "detections", {"sigma_px": 0.0, "miss_rate": 0.0, "false_positive_rate": 0.0,
                                                  "dead_shuttle_rate": 0.0}),
            ("px_a", "rally_a", "px", {"burst_miss": False, "miss_rate": 0.15})):
        sim = sims[of]
        streams = []
        with recording_rng(streams):
            if fn == "detections":
                c = S.corrupt_detections(sim, [sim.seed, 11], **kw)
            else:
                c = S.corrupt_px(sim.shuttle_px, sim.visible, sim.image_size, [sim.seed, 12], **kw)
        out["corrupt"].append({"name": name, "of": of, "fn": fn, "args": kw, "streams": streams, "candidates": _cands(c)})
    for style in ("behind", "elevated_side", "broadcast", "random"):
        for k in range(3):
            streams = []
            with recording_rng(streams):
                cam = S.random_camera([k, 5], (1280, 720), style)
            out["cameras"].append({"style": style, "size": [1280, 720], "streams": streams, "camera": cam.to_dict()})
    return out


def libm_probes() -> dict:
    rng = np.random.default_rng(99)
    x = np.concatenate([rng.uniform(-4, 4, 300), rng.uniform(-60, 60, 100)])
    xp = rng.uniform(1e-3, 50, 200)
    return {"x": x, "cos": np.cos(x), "sin": np.sin(x), "exp": np.exp(-np.abs(x)), "xp": xp,
            "log10": np.log10(xp), "pow10": np.power(10.0, x[:200] / 20), "cube": xp ** 3, "sqrt_pow": xp ** 0.5,
            "sqrt": np.sqrt(xp)}


COMPACT = {
    "e2e": ["--stage", "e2e", "--n", "3", "--no-json"],
    "oracle": ["--stage", "oracle", "--n", "3", "--no-json"],
    "real": ["--stage", "real", "--clips", "3", "--max-frames", "300", "--detector", "tracknetv3", "--no-json"],
}
FULL = {                                            # the 2026-09-27 baselines
    "e2e": ["--stage", "e2e", "--n", "12", "--no-json"],
    "oracle": ["--stage", "oracle", "--n", "10", "--no-json"],
    "real_tracknetv3": ["--stage", "real", "--detector", "tracknetv3", "--no-json"],
    "real_wasb": ["--stage", "real", "--detector", "wasb", "--no-real-oracle", "--no-json"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="metrics,sim,bench")
    ap.add_argument("--full", action="store_true", help="also write the full baseline bundles to --full-dir")
    ap.add_argument("--full-dir", default=str(FULL_DIR))
    a = ap.parse_args()
    only = set(a.only.split(","))
    if "metrics" in only:
        print("wrote", dump_ordered(metric_cases(), OUT / "metrics.json"))
    if "sim" in only:
        print("wrote", dump_ordered(sim_cases(), OUT / "sim_replay.json"))
    if "bench" in only:
        for name, argv in COMPACT.items():
            bundle, raw = record_bench(argv)
            print("wrote", dump_ordered(bundle, OUT / f"bench_{name}.json"))
            if name == "real":
                print("analysis files:", export_analysis_dir(raw, OUT / "analysis_real"))
    if a.full:
        full = Path(a.full_dir)
        for name, argv in FULL.items():
            bundle, raw = record_bench(argv)
            print("wrote", dump_ordered(bundle, full / f"bench_{name}.json"))
            if name == "real_tracknetv3":
                print("analysis files:", export_analysis_dir(raw, full / "analysis_tracknetv3"))


if __name__ == "__main__":
    main()
