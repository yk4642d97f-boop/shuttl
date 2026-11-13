#!/usr/bin/env python
"""Evidence behind every matched rally whose end_reason is wrong: last hitter
(pred vs truth), the final fit's end_reason / net crossing / landing and p0.

  .venv/bin/python -m scripts.accuracy.net_calls --style behind
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from app.services.tracking import rally as R
from scripts.accuracy.common import add_common_args, load_samples, sim_args
from scripts.accuracy.rally_errors import match


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--dump", type=int, default=25)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    c = Counter()
    rows = []
    for d in samples:
        pred = R.segment_rallies(d["track"], d["hits"], d["segs"], d["meta"], d["camera"],
                                 players=d["players"])
        true = d["true_rallies"]
        for i, j in match(pred, true):
            r, t = pred[i], true[j]
            if r.end_reason == t.end_reason or r.end_reason == "unknown":
                continue
            tl = d["true_shots"][t.first_shot + t.n_shots - 1]
            s = r.shots[-1]
            seg = s.segment
            ok_last = abs(s.hit.frame - tl.frame) <= 3
            key = (f"{t.end_reason}->{r.end_reason} last_ok={ok_last} "
                   f"hitter_ok={s.hit.hitter == tl.hitter}")
            if seg is not None:
                key += f" fit_end={seg.end_reason}"
            c[key] += 1
            rows.append(dict(seed=d["seed"], key=key, hit=s.hit.frame, tlast=tl.frame,
                             hitter=s.hit.hitter, thitter=tl.hitter,
                             net_h=None if seg is None or seg.net_crossing_height is None else round(seg.net_crossing_height, 3),
                             tnet_h=round(tl.net_crossing_height, 3) if np.isfinite(tl.net_crossing_height) else None,
                             land=None if seg is None or seg.landing_xy is None else tuple(round(v, 2) for v in seg.landing_xy),
                             tland=t.landing_xy and tuple(round(v, 2) for v in t.landing_xy),
                             p0=None if seg is None else tuple(round(float(v), 2) for v in seg.p0),
                             conf=None if seg is None else round(seg.confidence, 2),
                             sig=None if seg is None else round(getattr(seg, "landing_sigma_m", -1) or -1, 3)))
    for k, v in c.most_common():
        print(v, k)
    for r in rows[:args.dump]:
        print("  ", r)


if __name__ == "__main__":
    main()
