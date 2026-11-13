#!/usr/bin/env python
"""Long hit gaps (> max_hit_gap_s) inside vs between true rallies: which cues
separate them? Per consecutive (merged) hit pair with gap > 2.6 s:
  same      both hits inside one true rally (a split here is WRONG)
  rest      rally._Motion rest anywhere strictly between the hits
  unseen    longest unobserved stretch (s)
  slowfrac  fraction of observed speeds below min_flight_speed_hps
  arr       median image speed arriving at b (last 0.3 s), None = unseen
  unseen_ok rally._unseen_flight verdict, cont_ok _continuous_flight verdict

  .venv/bin/python -m scripts.accuracy.gap_rest --style behind --seed 0 --n 40
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from app.services.tracking import rally as R
from scripts.accuracy.common import add_common_args, load_samples, sim_args
from scripts.accuracy.side_trans import prepared


def rally_of(frame, true):
    for j, t in enumerate(true):
        if t.start_frame - 3 <= frame <= t.end_frame + 3:
            return j
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--dump", type=int, default=0)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    cfg = R.RallyConfig()
    c = Counter()
    rows = []
    for d in samples:
        mo, hs, segs, conts, groups, truth = prepared(d, cfg)
        for i in range(1, len(hs)):
            h0, h1 = hs[i - 1], hs[i]
            gap = h1.t - h0.t
            if gap <= cfg.max_hit_gap_s:
                continue
            r0, r1 = rally_of(h0.frame, d["true_rallies"]), rally_of(h1.frame, d["true_rallies"])
            same = r0 is not None and r0 == r1
            f0, f1 = h0.frame, h1.frame
            rest = bool(mo.rest[f0 + 1:f1].any())
            ok = mo.fin[f0:f1 + 1]
            tt = np.r_[mo.t[f0], mo.t[f0:f1 + 1][ok], mo.t[f1]]
            unseen = float(np.max(np.diff(tt)))
            v = mo.speed[f0:f1 + 1]
            v = v[np.isfinite(v)]
            slow = float(np.mean(v < cfg.min_flight_speed_hps)) if len(v) else float("nan")
            arr = R._median_speed(mo, h1.t - cfg.launch_window_s, h1.t - 1e-3)
            uok = R._unseen_flight(mo, h0, h1, cfg)
            cok = R._continuous_flight(mo, f0, f1, cfg)
            gb = "<4.5" if gap < 4.5 else ("<6" if gap < 6 else ">=6")
            key = f"same={same} rest={rest} gap{gb}"
            c[key] += 1
            if same:
                c[f"same unseen_ok={uok} cont_ok={cok}"] += 1
                rows.append((d["seed"], f0, f1, round(gap, 2), rest, round(unseen, 2),
                             round(slow, 2), None if arr is None else round(arr, 2), uok, cok))
    for k, v in sorted(c.items()):
        print(f"{v:5d} {k}")
    print("seed f0 f1 gap rest unseen slowfrac arr unseen_ok cont_ok")
    for r in rows[:args.dump]:
        print(r)


if __name__ == "__main__":
    main()
