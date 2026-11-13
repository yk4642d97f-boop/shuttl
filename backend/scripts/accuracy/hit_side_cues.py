#!/usr/bin/env python
"""Per detected hit: which hits.py side cue is right / wrong (player reach LLR,
camera-ray LLR, final label) and the true hitter's feet vs the shuttle.

  .venv/bin/python -m scripts.accuracy.hit_side_cues --style behind --n 20
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from app.services.tracking import hits as H
from scripts.accuracy.common import add_common_args, load_samples, sim_args


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    c = Counter()
    cfg = H.HitConfig()
    for d in samples:
        tr = H._prefer_raw(d["track"])
        cands = H.detect_candidates(tr, d["meta"], d["camera"], d["players"], cfg)
        pidx = H._PlayerIndex(d["players"], d["camera"])
        tf = np.array([h["frame"] for h in d["true_hits"]])
        byf = {c_.frame: c_ for c_ in cands}
        for h in d["hits"]:
            j = int(np.argmin(np.abs(tf - h.frame)))
            if abs(tf[j] - h.frame) > 3:
                continue
            th = d["true_hits"][j]["hitter"]
            cd = byf.get(h.frame)
            if cd is None:
                continue
            ray = H._ray_side_llr(d["camera"], cd.xy) if np.all(np.isfinite(cd.xy)) else 0.0
            ply = cd.side_llr - ray
            sgn = 1 if th == "near" else -1
            key = (f"label_ok={h.hitter == th} ply={'ok' if ply * sgn > 0.5 else ('bad' if ply * sgn < -0.5 else 'weak')}"
                   f" ray={'ok' if ray * sgn > 0.5 else ('bad' if ray * sgn < -0.5 else 'weak')}")
            c[key] += 1
    for k, v in sorted(c.items()):
        print(f"{v:5d} {k}")


if __name__ == "__main__":
    main()
