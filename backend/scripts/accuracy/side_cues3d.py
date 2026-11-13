#!/usr/bin/env python
"""Accuracy of candidate hitter-side cues on matched hits (dev seeds):
  label     hits.py hitter
  vx_own    sign of the hit's own 3-D launch vx (vx > 0 = towards far = near hitter)
  x_own     side of its own 3-D launch point
  x_in      side of the incoming (previous) fit's position at this contact time
  vx_in     incoming fit's vx at this contact (flying towards far -> far player hits)
Each cue is reported overall and on the hits whose label is wrong, with a
reliability split by |value|.

  .venv/bin/python -m scripts.accuracy.side_cues3d --style behind --seed 0 --n 40
"""
from __future__ import annotations

import argparse
from collections import defaultdict

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
    stats = defaultdict(lambda: [0, 0, 0, 0])          # n, ok, n_on_wrong, ok_on_wrong
    for d in samples:
        mo = R._Motion(d["track"], float(d["meta"].height), cfg)
        hs = R._clean_hits(d["hits"], mo.n)
        segs = R._match_segments(hs, d["segs"], mo.t, cfg)
        tf = np.array([h["frame"] for h in d["true_hits"]])
        for i, (h, s) in enumerate(zip(hs, segs)):
            j = int(np.argmin(np.abs(tf - h.frame)))
            if abs(tf[j] - h.frame) > 3:
                continue
            th = d["true_hits"][j]["hitter"]
            lab_ok = h.hitter == th
            cues = {"label": (h.hitter, 9.0)}
            if s is not None:
                vx, x = float(s.v0[0]), float(s.p0[0])
                cues["vx_own"] = ("near" if vx > 0 else "far", abs(vx))
                cues["x_own"] = ("near" if x < 0 else "far", abs(x))
            p = segs[i - 1] if i > 0 else None
            if p is not None and h.t - float(p.t0) < 2.5 and float(p.t0) < h.t:
                st = R._state_at(p, h.t)
                t_cov = float(mo.t[min(max(int(p.end_frame), 0), mo.n - 1)]) + 0.25
                if h.t <= t_cov:
                    cues["x_in"] = ("near" if st[0] < 0 else "far", abs(st[0]))
                    cues["vx_in"] = ("near" if st[3] < 0 else "far", abs(st[3]))
            for k, (side, mag) in cues.items():
                for lo, hi in ((0, 1e9), (0, 1), (1, 3), (3, 1e9)):
                    if k == "label" and (lo, hi) != (0, 1e9):
                        continue
                    if not lo <= mag < hi:
                        continue
                    key = f"{k:7s} |v| in [{lo},{hi})"
                    st_ = stats[key]
                    st_[0] += 1
                    st_[1] += side == th
                    if not lab_ok:
                        st_[2] += 1
                        st_[3] += side == th
    for k in sorted(stats):
        n, ok, nw, okw = stats[k]
        print(f"{k:28s} n={n:5d} acc={ok / max(n, 1):.3f} | on label-wrong n={nw:4d} acc={okw / max(nw, 1):.3f}")


if __name__ == "__main__":
    main()
