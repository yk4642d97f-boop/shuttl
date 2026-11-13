#!/usr/bin/env python
"""Prototype hitter-side cue: horizontal distance from each side's feet to the
camera sight ray through the contact pixel, over contact heights 0.3-3.2 m.
Accuracy of the cue alone and combined with hits.py's label.

  .venv/bin/python -m scripts.accuracy.feet_ray --style behind --n 20
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from app.services.tracking import hits as H
from scripts.accuracy.common import add_common_args, load_samples, sim_args


def feet_ray_dist(camera, uv, feet):
    o, d = camera.ray(np.asarray(uv, float)[None])
    d = d[0]
    if abs(d[2]) < 1e-6:
        return np.full(len(feet), np.inf)
    lam = np.linspace(0.3, 60, 1200)
    P = o[None] + lam[:, None] * d[None]
    ok = (P[:, 2] >= 0.3) & (P[:, 2] <= 3.2)
    P = P[ok]
    if not len(P):
        return np.full(len(feet), np.inf)
    return np.array([np.min(np.hypot(P[:, 0] - f[0], P[:, 1] - f[1])) for f in feet])


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    c = Counter()
    ds = []
    for d in samples:
        pidx = H._PlayerIndex(d["players"], d["camera"])
        tf = np.array([h["frame"] for h in d["true_hits"]])
        for h in d["hits"]:
            j = int(np.argmin(np.abs(tf - h.frame)))
            if abs(tf[j] - h.frame) > 3:
                continue
            th = d["true_hits"][j]["hitter"]
            obs = pidx.around(h.frame, 2)
            best = {"near": np.inf, "far": np.inf}
            for o, s in obs:
                if s in best and o.foot_world is not None:
                    dd = feet_ray_dist(d["camera"], h.xy_px, [o.foot_world])[0]
                    best[s] = min(best[s], dd)
            dn, df = best["near"], best["far"]
            cue = "near" if dn < df else "far"
            margin = abs(dn - df)
            c["n"] += 1
            c["label_ok"] += h.hitter == th
            c["cue_ok"] += cue == th
            # cue only when decisive (one side within reach, other clearly not)
            dec = min(dn, df) < 1.6 and margin > 0.8
            c["dec"] += dec
            c["dec_ok"] += dec and cue == th
            comb = cue if dec else h.hitter
            c["comb_ok"] += comb == th
            ds.append((dn if th == "near" else df, df if th == "near" else dn))
    for k, v in sorted(c.items()):
        print(v, k)
    ds = np.array(ds)
    print("true-hitter feet-ray dist pct50/90/99", np.percentile(ds[:, 0], [50, 90, 99]).round(2))
    print("other side  feet-ray dist pct1/10/50", np.percentile(ds[:, 1], [1, 10, 50]).round(2))


if __name__ == "__main__":
    main()
