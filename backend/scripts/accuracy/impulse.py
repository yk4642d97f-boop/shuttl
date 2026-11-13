#!/usr/bin/env python
"""3-D impulse at the last contact of each predicted rally: |v_new - v_prev(t)|
where v_prev(t) is the previous fitted flight propagated (drag ODE) to the
contact time. Real racket contacts reverse / redirect the shuttle; a spurious
contact inside one flight shows a small change.

  .venv/bin/python -m scripts.accuracy.impulse --style behind
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from app.services.tracking import rally as R
from app.services.tracking.physics import DRAG_LENGTH
from scripts.accuracy.common import add_common_args, load_samples, sim_args


def state_at(seg, t, h=1 / 240):
    s = [float(v) for v in (*seg.p0, *seg.v0)]
    L = float(seg.drag_length or DRAG_LENGTH)
    tt, stop = 0.0, t - float(seg.t0)
    while tt < stop - 1e-12:
        step = min(h, stop - tt)
        s = R._rk4(s, step, L)
        tt += step
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    rows = []
    for d in samples:
        pred = R.segment_rallies(d["track"], d["hits"], d["segs"], d["meta"], d["camera"],
                                 players=d["players"])
        tf = np.array([h["frame"] for h in d["true_hits"]])
        for r in pred:
            for k in range(1, len(r.shots)):
                s, p = r.shots[k], r.shots[k - 1]
                if s.segment is None or p.segment is None:
                    continue
                real = bool(np.any(np.abs(tf - s.hit.frame) <= 3))
                st = state_at(p.segment, float(s.segment.t0))
                dv = math.dist(st[3:], [float(v) for v in s.segment.v0])
                dp = math.dist(st[:3], [float(v) for v in s.segment.p0])
                rev = st[3] * float(s.segment.v0[0]) < 0
                rows.append((real, k == len(r.shots) - 1, dv, dp, rev, st[2]))
    rows = np.array(rows, float)
    for last in (1, 0):
        for real in (1, 0):
            sel = (rows[:, 0] == real) & (rows[:, 1] == last)
            dv = rows[sel, 2]
            print(f"last={last} real={real} n={sel.sum():4d} dv pct10/50/90 "
                  f"{np.percentile(dv, [10, 50, 90]).round(1) if sel.any() else '-'} "
                  f"rev_x={rows[sel, 4].mean():.2f} "
                  f"dv<4:{(dv < 4).sum()} dv<6:{(dv < 6).sum()} dv<8:{(dv < 8).sum()}")


if __name__ == "__main__":
    main()
