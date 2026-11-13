#!/usr/bin/env python
"""Features of the last shot of every predicted rally: real final contact vs
spurious (no true hit within 3 frames). Used to design the trailing-hit test.

  .venv/bin/python -m scripts.accuracy.trailing_hits --style behind
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
    for d in samples:
        pred = R.segment_rallies(d["track"], d["hits"], d["segs"], d["meta"], d["camera"],
                                 players=d["players"])
        tf = np.array([h["frame"] for h in d["true_hits"]])
        for r in pred:
            if len(r.shots) < 2:
                continue
            s, p = r.shots[-1], r.shots[-2]
            real = bool(np.any(np.abs(tf - s.hit.frame) <= 3))
            if s.segment is None:
                c[(real, "noseg")] += 1
                continue
            x0, vx = float(s.segment.p0[0]), float(s.segment.v0[0])
            away = x0 * vx > 0                       # moving away from the net
            own = r.landing_xy is not None and np.sign(r.landing_xy[0]) == np.sign(x0)
            pvx = float(p.segment.v0[0]) if p.segment is not None else np.nan
            same_dir = np.sign(pvx) == np.sign(vx)
            c[(real, f"away={away} own={own} same_dir={same_dir}")] += 1
    for k, v in sorted(c.items(), key=str):
        print(v, k)


if __name__ == "__main__":
    main()
