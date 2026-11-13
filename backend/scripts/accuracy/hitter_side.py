#!/usr/bin/env python
"""Hitter-side cues on matched hits (pred within 3 frames of a true hit):
hits.py label vs the 3-D fit (launch x side, launch direction sign(vx)).

  .venv/bin/python -m scripts.accuracy.hitter_side --style behind
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from app.services.tracking import rally as R
from scripts.accuracy.common import add_common_args, load_samples, sim_args


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    c = Counter()
    cfg = R.RallyConfig()
    for d in samples:
        mo = R._Motion(d["track"], float(d["meta"].height), cfg)
        hs = R._clean_hits(d["hits"], mo.n)
        segs = R._match_segments(hs, d["segs"], mo.t, cfg)
        tf = np.array([h["frame"] for h in d["true_hits"]])
        for h, s in zip(hs, segs):
            j = np.argmin(np.abs(tf - h.frame))
            if abs(tf[j] - h.frame) > 3:
                continue
            th = d["true_hits"][j]["hitter"]
            c["n"] += 1
            c["label_ok"] += h.hitter == th
            if s is None:
                c["noseg"] += 1
                c["noseg_label_ok"] += h.hitter == th
                continue
            c["seg"] += 1
            x, vx = float(s.p0[0]), float(s.v0[0])
            vside = "near" if vx > 0 else "far"
            xside = "near" if x < 0 else "far"
            c["seg_label_ok"] += h.hitter == th
            c["vx_ok"] += vside == th
            c["x_ok"] += xside == th
            spd = abs(vx)
            if spd > 3:
                c["vx>3"] += 1
                c["vx>3_ok"] += vside == th
                c["vx>3_label_ok"] += h.hitter == th
            if h.hitter != th:
                c[f"label_wrong: vx_ok={vside == th} |vx|>3={spd > 3} |x|<1.5={abs(x) < 1.5}"] += 1
    for k, v in sorted(c.items()):
        print(v, k)


if __name__ == "__main__":
    main()
