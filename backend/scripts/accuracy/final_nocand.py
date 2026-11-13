#!/usr/bin/env python
"""Why does hits.py produce NO candidate at a rally's true final contact?
Per such contact (dev seeds): observed detections in (-0.4, 0) and (0, +0.4) s
around it, time to the true landing, the scan score lam_v at the contact
(0 if the window is not scannable), whether a stronger peak within nms_s
suppressed it, and the true final shot type.

  .venv/bin/python -m scripts.accuracy.final_nocand --style behind --seed 0 --n 40
"""
from __future__ import annotations

import argparse
import math
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
    cfg = H.HitConfig()
    c = Counter()
    for d in samples:
        track = H._prefer_raw(d["track"])
        meta = d["meta"]
        W, Hh = meta.width, meta.height
        diag = math.hypot(W, Hh)
        oi = H._observed(track)
        track_t = np.asarray(track.t, float)
        xy = np.asarray(track.xy, float)[oi]
        keep, t = H._content_clock(oi, xy, track_t)
        oi, t, xy = oi[keep], t[keep], xy[keep]
        sigma = H._estimate_sigma(t, xy)
        keep = H._reject_outliers(t, xy, sigma, diag)
        oi, t, xy = oi[keep], t[keep], xy[keep]
        grid = track_t[(track_t >= t[0]) & (track_t <= t[-1])]
        sc = H._scan(t, xy, grid, cfg, sigma)
        cands = H.detect_candidates(track, meta, d["camera"], d["players"], cfg)
        cf = np.array([x.frame for x in cands], int) if cands else np.zeros(0, int)
        for r in d["true_rallies"]:
            k = r.first_shot + r.n_shots - 1
            th = d["true_hits"][k]
            f, tc = int(th["frame"]), float(th["t"])
            if len(cf) and np.min(np.abs(cf - f)) <= 3:
                continue
            sh = d["true_shots"][k]
            nb = int(np.sum((t > tc - 0.4) & (t < tc)))
            na = int(np.sum((t > tc) & (t < tc + 0.4)))
            t_land = float(sh.landing_t) if sh.landing_t is not None and np.isfinite(sh.landing_t) else float("nan")
            g = int(np.argmin(np.abs(grid - tc))) if len(grid) else -1
            lv = float(sc["lam_v"][g]) if g >= 0 and sc["ok"][g] else 0.0
            ok = bool(sc["ok"][g]) if g >= 0 else False
            # strongest scan score within nms_s
            near = (np.abs(grid - tc) < cfg.nms_s) & sc["ok"]
            mx = float(np.max(sc["lam_v"][near])) if near.any() else 0.0
            why = ("no_obs_after" if na < cfg.min_side else
                   "no_obs_before" if nb < cfg.min_side else
                   "not_scannable" if not ok else
                   "below_thr" if mx < cfg.lam_min else
                   "suppressed/refined_away")
            c[why] += 1
            c[f"type={sh.shot_type} ({why})"] += 1
            c[f"land_after<{0.4 if t_land - tc < 0.4 else 9}s ({why})"] += 1
    for k, v in sorted(c.items()):
        print(f"{v:5d} {k}")


if __name__ == "__main__":
    main()
