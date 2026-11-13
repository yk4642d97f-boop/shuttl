#!/usr/bin/env python
"""Anatomy of true contacts hits.py has no candidate for (dev seeds): the
observed detections around the contact after the same preprocessing as
hits.detect_candidates (observed -> content clock -> outlier rejection), which
survive, where they are relative to the image border, image velocity before /
after, the unobserved gap after the contact, and where the shuttle is seen next.

  .venv/bin/python -m scripts.accuracy.exit_cases --style behind --seed 0 --n 40 [--all] [--dump 20]
"""
from __future__ import annotations

import argparse
import math
from collections import Counter

import numpy as np

from app.services.tracking import hits as H
from scripts.accuracy.common import add_common_args, load_samples, sim_args


def prep(track, meta):
    """(oi, t, xy, kept_mask_over_observed, sigma) exactly as detect_candidates."""
    W, Hh = meta.width, meta.height
    diag = math.hypot(W, Hh)
    oi0 = H._observed(track)
    track_t = np.asarray(track.t, float)
    xy0 = np.asarray(track.xy, float)[oi0]
    keep, t = H._content_clock(oi0, xy0, track_t)
    oi, t, xy = oi0[keep], t[keep], xy0[keep]
    sigma = H._estimate_sigma(t, xy)
    keep2 = H._reject_outliers(t, xy, sigma, diag)
    return oi0, oi, t, xy, keep2, sigma


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--all", action="store_true", help="every true contact, not only a rally's last")
    ap.add_argument("--dump", type=int, default=0)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    cfg = H.HitConfig()
    c = Counter()
    dumped = 0
    for d in samples:
        track = H._prefer_raw(d["track"])
        meta = d["meta"]
        W, Hh = meta.width, meta.height
        oi0, oi, t, xy, keep2, sigma = prep(track, meta)
        cands = H.detect_candidates(track, meta, d["camera"], d["players"], cfg)
        cf = np.array([x.frame for x in cands], int) if cands else np.zeros(0, int)
        last_k = {r.first_shot + r.n_shots - 1 for r in d["true_rallies"]}
        for k, th in enumerate(d["true_hits"]):
            if not args.all and k not in last_k:
                continue
            f, tc = int(th["frame"]), float(th["t"])
            has = len(cf) and np.min(np.abs(cf - f)) <= 3
            c["contacts"] += 1
            if has:
                continue
            na = int(np.sum((t > tc) & (t < tc + 0.4)))
            nb = int(np.sum((t > tc - 0.4) & (t < tc)))
            if na >= cfg.min_side and nb >= cfg.min_side:
                c["nocand: enough obs both sides"] += 1
                continue
            if na < cfg.min_side:
                aft = np.flatnonzero(t > tc)
                a0 = aft[:na] if na else []
                # observations right after the contact that were REJECTED as outliers
                raw_after = np.flatnonzero((np.asarray(track.t)[oi0] > tc) & (np.asarray(track.t)[oi0] < tc + 0.4))
                n_raw = len(raw_after)
                nxt = aft[na] if na < len(aft) else None
                gap = float(t[nxt] - tc) if nxt is not None else float("inf")
                pb = xy[np.flatnonzero(t < tc)[-1]] if nb else None
                pa = xy[a0[-1]] if na else None
                vb = None
                ib = np.flatnonzero(t < tc)
                if len(ib) >= 2:
                    vb = (xy[ib[-1]] - xy[ib[-2]]) / (t[ib[-1]] - t[ib[-2]])
                va = None
                if na >= 2:
                    va = (xy[a0[1]] - xy[a0[0]]) / (t[a0[1]] - t[a0[0]])
                elif na == 1 and pb is not None:
                    va = (pa - pb) / (t[a0[0]] - t[ib[-1]])
                top = pa is not None and pa[1] < 0.08 * Hh
                edge = pa is not None and (pa[0] < 0.06 * W or pa[0] > 0.94 * W or pa[1] > 0.94 * Hh)
                key = (f"no_obs_after: kept_after={na} raw_after={n_raw} "
                       f"last_after={'top' if top else ('edge' if edge else ('none' if pa is None else 'interior'))} "
                       f"vy_before={'down' if vb is not None and vb[1] > 0 else ('up' if vb is not None else '?')} "
                       f"vy_after={'up' if va is not None and va[1] < 0 else ('down' if va is not None else '?')}")
                c[key] += 1
                if dumped < args.dump:
                    dumped += 1
                    print(d["seed"], f, round(tc, 3), "before", None if pb is None else pb.round(), "vb", None if vb is None else vb.round(),
                          "after kept", [(round(float(t[i] - tc), 3), xy[i].round().tolist()) for i in a0],
                          "raw after", n_raw, "next seen after", round(gap, 2), "at", None if nxt is None else xy[nxt].round().tolist(),
                          "type", d["true_shots"][k].shot_type, "hitter", th["hitter"])
            else:
                c["no_obs_before"] += 1
    for kk, v in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"{v:5d} {kk}")


if __name__ == "__main__":
    main()
