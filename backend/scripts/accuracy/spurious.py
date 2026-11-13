#!/usr/bin/env python
"""Anatomy of false rallies between true rallies and of trailing extra contacts
after a true rally's final contact (dev seeds). sim.py has no feeds/knock-ups:
between rallies the dead shuttle only lies still / is held, so these come from
spurious hits. Per spurious hit: confidence, 3-D fit (speed, start height, n obs,
rmse), image speed after it, time since the true rally end, whether the dead
shuttle's rest span overlaps it, and what the truth track shows (visible? in play?).

  .venv/bin/python -m scripts.accuracy.spurious --style behind --seed 0 --n 40
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from app.services.tracking import rally as R
from scripts.accuracy.common import add_common_args, load_samples, sim_args


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--dump", type=int, default=0)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    cfg = R.RallyConfig()
    c = Counter()
    dumped = 0
    for d in samples:
        mo = R._Motion(d["track"], float(d["meta"].height), cfg)
        pred = R.segment_rallies(d["track"], d["hits"], d["segs"], d["meta"], d["camera"], players=d["players"])
        tf = np.array([h["frame"] for h in d["true_hits"]])
        tr = d["true_rallies"]
        for r in pred:
            for k, s in enumerate(r.shots):
                f = s.hit.frame
                if len(tf) and np.min(np.abs(tf - f)) <= 3:
                    continue
                inside = [j for j, t in enumerate(tr) if t.start_frame - 3 <= f <= t.end_frame + 3]
                if inside:
                    t = tr[inside[0]]
                    last_f = d["true_shots"][t.first_shot + t.n_shots - 1].frame
                    kind = "trailing" if f > last_f + 3 else "mid_rally"
                else:
                    kind = "between"
                seg = s.segment
                sp = None if seg is None else float(np.linalg.norm(seg.v0)) * 3.6
                pk = mo.peak_speed(f, cfg.flight_window_s)
                hold = R._hold_before(mo, f, s.hit.t, cfg)
                conf = s.hit.confidence
                key = (kind, "fit" if seg is not None else "nofit",
                       "conf<0.5" if conf < 0.5 else ("conf<0.8" if conf < 0.8 else "conf>=0.8"),
                       {1.0: "held", -1.0: "moving", 0.0: "unseen"}[hold])
                c[key] += 1
                c[(kind, "ALL")] += 1
                if args.dump and dumped < args.dump and kind != "mid_rally":
                    dumped += 1
                    prev_end = max([t.end_frame for t in tr if t.end_frame < f], default=None)
                    print(d["seed"], kind, "f", f, "conf", round(conf, 2), "hitter", s.hit.hitter, "k", k, "/", len(r.shots),
                          "fit km/h", None if sp is None else round(sp), "z0", None if seg is None else round(float(seg.p0[2]), 2),
                          "p0x", None if seg is None else round(float(seg.p0[0]), 1),
                          "img peak", None if pk is None else round(pk, 2), "hold", hold,
                          "since true end", None if prev_end is None else round(float(mo.t[f] - mo.t[prev_end]), 2),
                          "rally", r.start_frame, r.end_frame)
    for k in sorted(c, key=lambda k: (k[0], -c[k])):
        print(f"{c[k]:5d} {k}")


if __name__ == "__main__":
    main()
