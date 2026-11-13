#!/usr/bin/env python
"""Hidden final contact cue (dev seeds): after a rally's last detected hit, where
does the shuttle come to rest (first 2-D rest, back-projected to the floor), and
how deep is that into the LAST HITTER'S OWN half? A shuttle cannot come to rest
deep in its hitter's own half (a net fault drops back next to the net) unless
somebody on the other side hit it back. Tabulated against the truth: was the
true final contact missed (the predicted last shot is an earlier true contact)?

  .venv/bin/python -m scripts.accuracy.own_half_rest --style behind --seed 0 --n 40
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
    ap.add_argument("--dump", type=int, default=0)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    cfg = R.RallyConfig()
    c = Counter()
    rows = []
    for d in samples:
        mo = R._Motion(d["track"], float(d["meta"].height), cfg)
        pred = R.segment_rallies(d["track"], d["hits"], d["segs"], d["meta"], d["camera"], players=d["players"])
        mp = dict(match(pred, d["true_rallies"]))
        for i, r in enumerate(pred):
            if not r.shots or i not in mp:
                continue
            t = d["true_rallies"][mp[i]]
            k_last = t.first_shot + t.n_shots - 1
            tl = d["true_shots"][k_last]
            last = r.shots[-1].hit
            if last.hitter not in R.SIDES:
                continue
            missed = last.frame < tl.frame - 3
            limit = pred[i + 1].shots[0].hit.frame if i + 1 < len(pred) and pred[i + 1].shots else mo.n
            fr = mo.first_rest_after(last.frame, min(limit, mo.frame_at(last.t + 6.0) + 1))
            if fr is None:
                c[(("MISSED" if missed else "ok"), "no rest")] += 1
                continue
            p = d["camera"].pixel_to_floor(mo.xy[fr][None, :])[0]
            if not np.all(np.isfinite(p[:2])):
                c[(("MISSED" if missed else "ok"), "rest not on floor")] += 1
                continue
            own = (-1.0 if last.hitter == "near" else 1.0) * float(p[0])   # + = into the own half, m from the net
            dt = float(mo.t[fr] - last.t)
            b = "own>2.0" if own > 2.0 else ("own 1-2" if own > 1.0 else ("own 0-1" if own > 0 else "other half"))
            c[(("MISSED" if missed else "ok"), b)] += 1
            lab_ok = last.hitter == d["true_hits"][min(range(len(d["true_hits"])), key=lambda j: abs(d["true_hits"][j]["frame"] - last.frame))]["hitter"]
            in_court = abs(float(p[0])) <= 6.7 + 0.5 and abs(float(p[1])) <= 3.05 + 0.5
            rows.append((missed, own, dt, t.end_reason, lab_ok, in_court))
            if args.dump and missed and own > 1.0:
                print(d["seed"], mp[i], "last", last.frame, last.hitter, "true last", tl.frame, d["true_hits"][k_last]["hitter"],
                      "rest", fr, "own", round(own, 2), "dt", round(dt, 2), "true", t.end_reason, t.winner, "pred", r.end_reason, r.winner)
    for kk, v in sorted(c.items(), key=str):
        print(f"{v:5d} {kk}")
    for thr in (1.0, 1.5, 2.0, 2.5):
        sel = [r for r in rows if r[1] > thr and r[5] and 0.25 <= r[2] <= 3.0]
        tp = sum(1 for r in sel if r[0])
        fp = len(sel) - tp
        print(f"rule own > {thr} m: flags {tp + fp}, of which missed final {tp} (precision {tp / max(tp + fp, 1):.2f});"
              f" missed finals with a floor rest {sum(1 for r in rows if r[0])}")
    fpr = [r for r in rows if not r[0] and r[1] > 1.0 and r[5] and 0.25 <= r[2] <= 3.0]
    print("false flags in court, dt 0.25-3 s (own depth, dt, true end, label_ok):", [(round(r[1], 2), round(r[2], 2), r[3], r[4]) for r in fpr])


if __name__ == "__main__":
    main()
