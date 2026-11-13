#!/usr/bin/env python
"""Matched rallies: does the 2-D track keep flying after the final fit ends /
lands? Compares rallies whose true last contact was detected vs missed.

  .venv/bin/python -m scripts.accuracy.last_missed --style behind
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
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    cfg = R.RallyConfig()
    rows = []
    for d in samples:
        mo = R._Motion(d["track"], float(d["meta"].height), cfg)
        pred = R.segment_rallies(d["track"], d["hits"], d["segs"], d["meta"], d["camera"],
                                 players=d["players"])
        true = d["true_rallies"]
        for i, j in match(pred, true):
            r, t = pred[i], true[j]
            tl = d["true_shots"][t.first_shot + t.n_shots - 1]
            s = r.shots[-1]
            missed = s.hit.frame < tl.frame - 3
            seg = s.segment
            if seg is None:
                rows.append((missed, "noseg", None, None, r.winner == t.winner))
                continue
            f_end = int(seg.end_frame)
            t_end = seg.landing_t if seg.landing_t is not None else float(mo.t[min(f_end, mo.n - 1)])
            f_e = mo.frame_at(t_end)
            # observed fast flight after the fit's end, before the next rest
            fr = mo.first_rest_after(f_e, min(mo.n, f_e + int(4 * d["meta"].fps)))
            f_stop = fr if fr is not None else min(mo.n - 1, f_e + int(4 * d["meta"].fps))
            fly = R._extend_flight(mo, f_e, f_e, f_stop, cfg)
            after = float(mo.t[fly] - t_end)
            rows.append((missed, seg.end_reason, round(after, 2),
                         getattr(r, "outcome_source", "?"), r.winner == t.winner))
    c = Counter()
    for m, er, af, src, ok in rows:
        c[(m, er, None if af is None else af > 0.35, ok)] += 1
    for k, v in sorted(c.items(), key=str):
        print(v, "missed=%s fit_end=%s flies_after>0.35=%s winner_ok=%s" % k)


if __name__ == "__main__":
    main()
