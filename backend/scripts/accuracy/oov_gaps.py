#!/usr/bin/env python
"""Out-of-view gaps in the hits.py observation stream (after its preprocessing):
every gap > 0.3 s whose last sighting before AND first sighting after are near
the image border. Does a true contact lie inside (a hidden contact)? Features:
gap length, which border (top/side/bottom) at exit / entry, image vy at exit /
entry, and whether the flight after entry is followed by a rest (rally end).

  .venv/bin/python -m scripts.accuracy.oov_gaps --style behind --seed 0 --n 40
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from app.services.tracking import hits as H
from scripts.accuracy.common import add_common_args, load_samples, sim_args
from scripts.accuracy.exit_cases import prep


def border(p, W, Hh, cfg):
    bx, by = cfg.border_band * W, cfg.border_band * Hh
    if p[1] < by:
        return "top"
    if p[1] > Hh - by:
        return "bottom"
    if p[0] < bx or p[0] > W - bx:
        return "side"
    return "in"


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    cfg = H.HitConfig()
    c = Counter()
    for d in samples:
        track = H._prefer_raw(d["track"])
        meta = d["meta"]
        W, Hh = meta.width, meta.height
        oi0, oi, t, xy, keep2, sigma = prep(track, meta)
        t, xy = t[keep2], xy[keep2]
        tt = np.array([h["t"] for h in d["true_hits"]])
        last_k = {r.first_shot + r.n_shots - 1 for r in d["true_rallies"]}
        for i in np.flatnonzero(np.diff(t) > 0.3):
            ea, eb = border(xy[i], W, Hh, cfg), border(xy[i + 1], W, Hh, cfg)
            if ea == "in" and eb == "in":
                continue
            inside = np.flatnonzero((tt > t[i]) & (tt < t[i + 1]))
            n_in = len(inside)
            fin = any(k in last_k for k in inside)
            vy_a = (xy[i, 1] - xy[i - 1, 1]) / (t[i] - t[i - 1]) if i > 0 else 0.0
            vy_b = (xy[i + 2, 1] - xy[i + 1, 1]) / (t[i + 2] - t[i + 1]) if i + 2 < len(t) else 0.0
            g = t[i + 1] - t[i]
            gb = "<0.6" if g < 0.6 else ("<1.0" if g < 1.0 else ("<2.0" if g < 2.0 else ("<4" if g < 4 else ">=4")))
            key = (f"exit={ea}/{'up' if vy_a < 0 else 'down'} entry={eb}/{'down' if vy_b > 0 else 'up'} gap{gb}")
            c[(key, f"contacts_inside={min(n_in, 2)}")] += 1
            if fin:
                c[(key, "FINAL contact inside")] += 1
    rows = {}
    for (k, what), v in c.items():
        rows.setdefault(k, {})[what] = v
    for k in sorted(rows, key=lambda k: -sum(rows[k].values())):
        r = rows[k]
        print(f"{k:44s} none {r.get('contacts_inside=0', 0):4d} one {r.get('contacts_inside=1', 0):4d} "
              f"two+ {r.get('contacts_inside=2', 0):3d} | final inside {r.get('FINAL contact inside', 0)}")


if __name__ == "__main__":
    main()
