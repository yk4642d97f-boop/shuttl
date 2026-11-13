#!/usr/bin/env python
"""One-hit rallies: real (a serve that is the whole point: ace / fault / return missed)
vs spurious (dead-time hit: pick-up / hand motion). Cues: 3-D fit present, hit
confidence, observed flight time after it, and the image displacement between
where the shuttle rested before the hit and where it rests next (px / image height).

  .venv/bin/python -m scripts.accuracy.onehit_cue --seed 0
"""
from __future__ import annotations

import argparse

import numpy as np

from app.services.tracking import rally as R
from scripts.accuracy.common import load_samples, sim_args


def rest_pos(mo, f0, f1):
    k = np.flatnonzero(mo.rest[max(f0, 0):min(f1, mo.n)])
    return None if not len(k) else mo.xy[max(f0, 0) + k[-1]]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args(argv)
    cfg = R.RallyConfig()
    rows = []
    for style in ("behind", "random"):
        for d in load_samples(sim_args(style, "1920x1080", 60.0, 5), list(range(args.seed, args.seed + args.n)), 4):
            mo = R._Motion(d["track"], float(d["meta"].height), cfg)
            pred = R.segment_rallies(d["track"], d["hits"], d["segs"], d["meta"], d["camera"], players=d["players"], config=R.RallyConfig(trail_max_conf=0.0, onehit_min_flight_s=0.0))
            tf = np.array([h["frame"] for h in d["true_hits"]])
            for r in pred:
                if len(r.shots) != 1:
                    continue
                s = r.shots[0]
                f = s.hit.frame
                real = bool(len(tf) and np.min(np.abs(tf - f)) <= 3)
                fps = d["meta"].fps or 60.0
                pb = rest_pos(mo, mo.frame_at(s.hit.t - 1.0), f)
                fr = mo.first_rest_after(f, min(mo.n, mo.frame_at(s.hit.t + 4.0) + 1))
                pa = mo.xy[fr] if fr is not None else None
                disp = None if pb is None or pa is None else float(np.linalg.norm(pa - pb)) / mo.h
                end = fr if fr is not None else mo.frame_at(s.hit.t + 4.0)
                fl = float((mo.fin[f + 1:end] & ~mo.rest[f + 1:end]).sum()) / fps
                rows.append((real, s.segment is not None, s.hit.confidence, fl, disp, style))
    for nm, rule in (("nofit & flight<0.3", lambda r: not r[1] and r[3] < 0.3),
                     ("nofit & flight<0.4", lambda r: not r[1] and r[3] < 0.4),
                     ("nofit & flight<0.6", lambda r: not r[1] and r[3] < 0.6),
                     ("nofit", lambda r: not r[1]),
                     ("flight<0.3 & conf<0.95", lambda r: r[3] < 0.3 and r[2] < 0.95)):
        rr = sum(1 for r in rows if r[0] and rule(r)); ss = sum(1 for r in rows if not r[0] and rule(r))
        print(f"  rule {nm:24s}: drops spurious {ss}/{sum(not r[0] for r in rows)}, real {rr}/{sum(r[0] for r in rows)}")
    for real in (True, False):
        A = [r for r in rows if r[0] == real]
        print(("REAL one-shot rallies" if real else "SPURIOUS one-hit groups"), len(A))
        print("   with 3-D fit", sum(r[1] for r in A), " conf<0.8", sum(r[2] < 0.8 for r in A),
              " flight<0.3 s", sum(r[3] < 0.3 for r in A),
              " disp known", sum(r[4] is not None for r in A),
              " disp<0.05 H", sum(r[4] is not None and r[4] < 0.05 for r in A),
              " disp<0.1 H", sum(r[4] is not None and r[4] < 0.1 for r in A))
        print("   (fit, conf, flight_s, disp):", [(r[1], round(r[2], 2), round(r[3], 2), None if r[4] is None else round(r[4], 3)) for r in A][:24])


if __name__ == "__main__":
    main()
