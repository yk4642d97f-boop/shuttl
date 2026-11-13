#!/usr/bin/env python
"""
Why does rally.py split / extend rallies? For every rally boundary rally.py
places (a cue between consecutive cleaned hits), report whether it falls inside
a true rally (wrong split) or between true rallies (correct), with the cue,
the hit gap and how many true hits the detector missed in between. Also
reports trailing predicted hits after the true final contact (with their 3-D
start height and the time relative to the previous flight's floor contact).

  .venv/bin/python -m scripts.accuracy.split_cues --style behind --seed 0 --n 40
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from app.services.tracking import rally as R
from scripts.accuracy.common import add_common_args, load_samples, sim_args


def rally_of(frame: int, true) -> int | None:
    for j, t in enumerate(true):
        if t.start_frame - 3 <= frame <= t.end_frame + 3:
            return j
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--dump", type=int, default=8)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    cfg = R.RallyConfig()
    stats = Counter()
    ex: dict[str, list] = {}
    for d in samples:
        mo = R._Motion(d["track"], float(d["meta"].height), cfg)
        hs = R._clean_hits(d["hits"], mo.n)
        segs = R._match_segments(hs, d["segs"], mo.t, cfg)
        hs, segs, conts = R._merge_same_flight(hs, segs, mo.t, cfg)
        groups, cues = R._group_hits(hs, segs, mo, cfg, conts)
        true = d["true_rallies"]
        thf = np.array([h["frame"] for h in d["true_hits"]])
        for gi, cue in enumerate(cues):
            a_h, b_h = hs[groups[gi][-1]], hs[groups[gi + 1][0]]
            ra, rb = rally_of(a_h.frame, true), rally_of(b_h.frame, true)
            same = ra is not None and ra == rb
            miss = int(((thf > a_h.frame + 3) & (thf < b_h.frame - 3)).sum())
            key = f"{'WRONG' if same else 'ok'}:{cue}"
            stats[key] += 1
            seg = segs[groups[gi][-1]]
            info = dict(seed=d["seed"], a=a_h.frame, b=b_h.frame, gap=round(b_h.t - a_h.t, 2),
                        missed_true=miss, seg=seg is not None,
                        seg_conf=None if seg is None else round(seg.confidence, 2),
                        seg_land_t=None if seg is None or seg.landing_t is None else round(seg.landing_t - a_h.t, 2),
                        seg_end=None if seg is None else int(seg.end_frame),
                        seg_endr=None if seg is None else seg.end_reason)
            if same:
                ex.setdefault(key, []).append(info)
        # trailing extra hits after the true last hit of each true rally
        for t in true:
            last = d["true_shots"][t.first_shot + t.n_shots - 1].frame
            for i, h in enumerate(hs):
                if last + 3 < h.frame <= t.end_frame + 30:
                    s = segs[i]
                    prev = segs[i - 1] if i > 0 else None
                    ft = None
                    if prev is not None:
                        ft = R._floor_time(prev, h.t + 1.0)
                    stats["extra_after_last"] += 1
                    ex.setdefault("extra_after_last", []).append(dict(
                        seed=d["seed"], true_last=last, true_end=t.end_frame, hit=h.frame,
                        conf=round(h.confidence, 2), hitter=h.hitter,
                        z0=None if s is None else round(float(s.p0[2]), 2),
                        prev_floor_minus_hit=None if ft is None else round(ft - h.t, 3),
                        t_end_reason=t.end_reason))
    for k, v in sorted(stats.items()):
        print(f"{v:5d}  {k}")
    for k, lst in ex.items():
        print("---", k)
        for r in lst[:args.dump]:
            print("   ", r)


if __name__ == "__main__":
    main()
