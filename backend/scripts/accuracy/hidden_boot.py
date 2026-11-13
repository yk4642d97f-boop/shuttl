#!/usr/bin/env python
"""Paired bootstrap (per match) of a hits.py option through the whole chain, on two
pinned upstream caches (A = option off, B = default): hits F1 at +-3 frames and at
+-0.5 s (hidden contacts are timed at the out-of-view gap midpoint, median 0.22 s
off), hitter side, rally F1 / end / winner, side of the final shots, landing p95.

  .venv/bin/python -m scripts.accuracy.hidden_boot --style behind --seed 1000 --n 40 \
      --hash-a 252cd2187a --hash-b 7f43851d1f
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from app.services.tracking import metrics as M
from app.services.tracking import rally as R
from scripts.accuracy.common import add_common_args, load_samples, sim_args
from scripts.accuracy.rally_errors import match


def per_match(d: dict) -> dict:
    tf = [{"frame": h["frame"], "hitter": h["hitter"]} for h in d["true_hits"]]
    ph = [{"frame": h.frame, "hitter": h.hitter} for h in d["hits"]]
    fps = float(d["meta"].fps or 60.0)
    h3, h30 = M.hit_metrics(ph, tf, 3), M.hit_metrics(ph, tf, int(round(0.5 * fps)))
    pred = R.segment_rallies(d["track"], d["hits"], d["segs"], d["meta"], d["camera"], players=d["players"])
    rm = M.rally_metrics(pred, d["true_rallies"])
    shots = [s for r in pred for s in r.shots]
    pairs = M._match_events(np.array([s.hit.frame for s in shots], int), np.array([t["frame"] for t in tf], int), 3)
    side_ok = sum(shots[i].hit.hitter == tf[j]["hitter"] for i, j in pairs)
    errs = []                                   # landing error of IoU-matched rallies (rally_metrics pairing)
    for i, j in match(pred, d["true_rallies"]):
        p, t = pred[i].landing_xy, d["true_rallies"][j].landing_xy
        if p is not None and t is not None:
            errs.append(float(np.hypot(p[0] - t[0], p[1] - t[1])))
    nm = rm["tp"]
    return {"h3": (h3["tp"], h3["n_pred"], h3["n_true"], h3["hitter_acc"] * h3["tp"] if h3["tp"] else 0.0),
            "h30": (h30["tp"], h30["n_pred"], h30["n_true"]),
            "r": (rm["tp"], rm["n_pred"], rm["n_true"], (rm["winner_acc"] or 0) * nm if nm else 0.0,
                  (rm["end_reason_acc"] or 0) * nm if nm else 0.0, nm),
            "side": (side_ok, len(pairs)), "land": errs}


def metrics(ms: list[dict]) -> dict:
    s = lambda k, i: sum(m[k][i] for m in ms)          # noqa: E731
    f1 = lambda tp, a, b: 2 * tp / max(a + b, 1)        # noqa: E731
    land = np.concatenate([m["land"] for m in ms]) if any(m["land"] for m in ms) else np.zeros(0)
    return {"hits_f1_3f": f1(s("h3", 0), s("h3", 1), s("h3", 2)), "hits_side": s("h3", 3) / max(s("h3", 0), 1),
            "hits_f1_05s": f1(s("h30", 0), s("h30", 1), s("h30", 2)),
            "rally_f1": f1(s("r", 0), s("r", 1), s("r", 2)), "winner": s("r", 3) / max(s("r", 5), 1),
            "end": s("r", 4) / max(s("r", 5), 1), "side_shots": s("side", 0) / max(s("side", 1), 1),
            "land_p95": float(np.percentile(land, 95)) if len(land) else float("nan")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--hash-a", required=True, help="cache with the option OFF")
    ap.add_argument("--hash-b", required=True, help="cache with the default")
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    seeds = list(range(args.seed, args.seed + args.n))
    runs = {}
    for name, h in (("off", args.hash_a), ("on", args.hash_b)):
        os.environ["SHUTTL_ACC_HASH"] = h
        runs[name] = [per_match(d) for d in load_samples(a, seeds, 1)]
    A, B = metrics(runs["off"]), metrics(runs["on"])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(seeds), (args.boot, len(seeds)))
    diffs = {k: [] for k in A}
    for row in idx:
        ma, mb = metrics([runs["off"][i] for i in row]), metrics([runs["on"][i] for i in row])
        for k in A:
            diffs[k].append(mb[k] - ma[k])
    print(f"{args.style} s{args.seed}..{seeds[-1]}  off={args.hash_a} on={args.hash_b}  (paired bootstrap, {args.boot} resamples)")
    for k in A:
        lo, hi = np.nanpercentile(diffs[k], [2.5, 97.5])
        tag = "  CI>0" if lo > 0 else ("  CI<0" if hi < 0 else "")
        print(f"  {k:12s} off {A[k]:.3f}  on {B[k]:.3f}  diff {B[k] - A[k]:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]{tag}")


if __name__ == "__main__":
    main()
