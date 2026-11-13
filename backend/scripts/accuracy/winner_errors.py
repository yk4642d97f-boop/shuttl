#!/usr/bin/env python
"""Why is a matched rally's winner wrong? Buckets per matched rally:
  last_missed     the true final contact was not the predicted last shot
  side_wrong      predicted last hitter side != true last hitter
  source          outcome_source (3d / next_serve / missed_final / 2d / none)
  end_ok          end_reason right?

  .venv/bin/python -m scripts.accuracy.winner_errors --style behind --seed 0 --n 40
"""
from __future__ import annotations

import argparse
from collections import Counter

from app.services.tracking import rally as R
from scripts.accuracy.common import add_common_args, load_samples, sim_args
from scripts.accuracy.rally_errors import match


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--dump", type=int, default=0)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    c = Counter()
    dumped = 0
    for d in samples:
        pred = R.segment_rallies(d["track"], d["hits"], d["segs"], d["meta"], d["camera"],
                                 players=d["players"])
        for i, j in match(pred, d["true_rallies"]):
            r, t = pred[i], d["true_rallies"][j]
            k_last = t.first_shot + t.n_shots - 1
            tl = d["true_shots"][k_last]
            th = d["true_hits"][k_last]["hitter"]
            last = r.shots[-1] if r.shots else None
            missed = last is None or abs(last.hit.frame - tl.frame) > 3
            side_ok = (not missed) and last.hit.hitter == th
            src = getattr(r, "outcome_source", "?")
            ok = r.winner == t.winner
            c["n"] += 1
            c["win_ok"] += ok
            key = (f"win_ok={ok} missed_last={missed} side_ok={side_ok if not missed else '-'} "
                   f"src={src} end_ok={r.end_reason == t.end_reason}")
            c[key] += 1
            if not ok and dumped < args.dump:
                dumped += 1
                print(d["seed"], j, "true", t.end_reason, t.winner, "pred", r.end_reason, r.winner, src,
                      round(getattr(r, "outcome_confidence", 0), 2), "missed", missed,
                      "last", None if last is None else (last.hit.frame, last.hit.hitter), "true last", tl.frame, th)
    for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"{v:5d} {k}")


if __name__ == "__main__":
    main()
