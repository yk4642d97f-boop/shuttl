#!/usr/bin/env python
"""Does the label-conditioned 3-D fit betray a wrong hitter label? Per matched
hit with a segment: label ok?, rmse_px, confidence, signed launch x (towards
the labelled hitter's half = +), launch vx towards the other half (+), whether
the fit started within 1 m of the net.

  .venv/bin/python -m scripts.accuracy.seg_side --style behind --seed 0 --n 40
"""
from __future__ import annotations

import argparse

import numpy as np

from app.services.tracking import rally as R
from scripts.accuracy.common import add_common_args, load_samples, sim_args


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    cfg = R.RallyConfig()
    rows = []
    for d in samples:
        mo = R._Motion(d["track"], float(d["meta"].height), cfg)
        hs = R._clean_hits(d["hits"], mo.n)
        segs = R._match_segments(hs, d["segs"], mo.t, cfg)
        tf = np.array([h["frame"] for h in d["true_hits"]])
        for h, s in zip(hs, segs):
            j = int(np.argmin(np.abs(tf - h.frame)))
            if abs(tf[j] - h.frame) > 3 or s is None or h.hitter not in R.SIDES:
                continue
            th = d["true_hits"][j]["hitter"]
            sg = -1.0 if h.hitter == "near" else 1.0          # labelled hitter's half sign
            rows.append((h.hitter == th, float(s.rmse_px), float(s.confidence),
                         sg * float(s.p0[0]), -sg * float(s.v0[0]), float(s.p0[2]),
                         float(np.linalg.norm(s.v0)), getattr(s, "n_obs", 0)))
    A = np.array(rows, float)
    ok = A[:, 0] > 0
    names = ["rmse_px", "conf", "x_own", "vx_across", "z0", "speed", "n_obs"]
    print(f"n={len(A)} label ok={ok.sum()} wrong={(~ok).sum()}")
    for k, nm in enumerate(names, start=1):
        a_, b_ = A[ok, k], A[~ok, k]
        print(f"{nm:10s} ok p10/50/90 {np.percentile(a_, [10, 50, 90]).round(2)}  "
              f"wrong p10/50/90 {np.percentile(b_, [10, 50, 90]).round(2)}")


if __name__ == "__main__":
    main()
