#!/usr/bin/env python
"""Top-exit -> top-entry out-of-view gaps (dev): contact inside or not by gap
length, and how well the hidden contact time is placed by (a) the gap midpoint,
(b) the time the two extrapolated image arcs (quadratics through the last /
first k sightings) come closest.

  .venv/bin/python -m scripts.accuracy.hidden_tau --style behind --seed 0 --n 40
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from app.services.tracking import hits as H
from scripts.accuracy.common import add_common_args, load_samples, sim_args
from scripts.accuracy.exit_cases import prep
from scripts.accuracy.oov_gaps import border


def meet_time(ta, pa, tb, pb, t0, t1, k=6):
    """Closest approach in [t0, t1] of quadratic fits to (ta, pa) and (tb, pb)."""
    A = np.polyfit(ta[-k:] - t0, pa[-k:], 2)
    B = np.polyfit(tb[:k] - t0, pb[:k], 2)
    g = np.linspace(0.0, t1 - t0, 200)
    da = np.stack([np.polyval(A[:, j], g) for j in range(2)], 1)
    db = np.stack([np.polyval(B[:, j], g) for j in range(2)], 1)
    return t0 + g[int(np.argmin(np.linalg.norm(da - db, axis=1)))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    cfg = H.HitConfig()
    c = Counter()
    err_mid, err_meet = [], []
    for d in samples:
        track = H._prefer_raw(d["track"])
        W, Hh = d["meta"].width, d["meta"].height
        oi0, oi, t, xy, keep2, sigma = prep(track, d["meta"])
        t, xy = t[keep2], xy[keep2]
        tt = np.array([h["t"] for h in d["true_hits"]])
        for i in np.flatnonzero(np.diff(t) > 0.3):
            if i < 6 or i + 7 > len(t):
                continue
            if border(xy[i], W, Hh, cfg) != "top" or border(xy[i + 1], W, Hh, cfg) != "top":
                continue
            if not (xy[i, 1] < xy[i - 2, 1] and xy[i + 2, 1] > xy[i + 1, 1]):   # leaves upwards, comes down
                continue
            g = t[i + 1] - t[i]
            inside = tt[(tt > t[i]) & (tt < t[i + 1])]
            b = f"{int(g / 0.25) * 0.25:.2f}"
            c[(b, len(inside) > 0)] += 1
            if len(inside) == 1 and g >= 1.5:
                err_mid.append(0.5 * (t[i] + t[i + 1]) - inside[0])
                ok_a, ok_b = slice(i - 5, i + 1), slice(i + 1, i + 7)
                err_meet.append(meet_time(t[ok_a], xy[ok_a], t[ok_b], xy[ok_b], t[i], t[i + 1]) - inside[0])
    print("gap bin (s): no contact / contact inside")
    for b in sorted({k[0] for k in c}, key=float):
        print(f"  {b}: {c[(b, False)]:4d} / {c[(b, True)]:4d}")
    for nm, e in (("midpoint", err_mid), ("arc meeting", err_meet)):
        e = np.abs(np.array(e))
        print(f"{nm:12s} |tau - true| median {np.median(e):.3f} s  p90 {np.percentile(e, 90):.3f} s  within 3 frames {np.mean(e <= 0.05):.2f}  n {len(e)}")


if __name__ == "__main__":
    main()
