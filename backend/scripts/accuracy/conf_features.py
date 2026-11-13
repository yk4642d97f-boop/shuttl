#!/usr/bin/env python
"""Which evidence predicts a correct winner for 3-D outcome calls? Per matched
rally: winner correct, last-hit confidence, fit confidence, |p0.x| (launch
distance from the net), launch direction agreeing with the hitter label,
|margin| / sigma. Prints accuracy by feature bin.

  .venv/bin/python -m scripts.accuracy.conf_features --style behind
"""
from __future__ import annotations

import argparse

import numpy as np

from app.services.tracking import rally as R
from scripts.accuracy.common import add_common_args, load_samples, sim_args
from scripts.accuracy.rally_errors import match


def rows_for(samples, cfg=None):
    out = []
    for d in samples:
        pred = R.segment_rallies(d["track"], d["hits"], d["segs"], d["meta"], d["camera"],
                                 players=d["players"], config=cfg)
        for i, j in match(pred, d["true_rallies"]):
            r, t = pred[i], d["true_rallies"][j]
            if r.winner not in R.SIDES:
                continue
            s = r.shots[-1]
            seg = s.segment
            row = dict(ok=r.winner == t.winner, end_ok=r.end_reason == t.end_reason,
                       src=getattr(r, "outcome_source", "?"), oconf=r.outcome_confidence,
                       hconf=s.hit.confidence, reason=r.end_reason, nshots=len(r.shots))
            if seg is not None:
                row.update(fconf=seg.confidence, ax=abs(float(seg.p0[0])),
                           dir_ok=(float(seg.v0[0]) > 0) == (s.hit.hitter == "near"),
                           vx=abs(float(seg.v0[0])))
            out.append(row)
    return out


def table(rows, key, bins):
    vals = [(r[key], r["ok"]) for r in rows if key in r and r[key] is not None]
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [ok for v, ok in vals if lo <= v < hi]
        if sel:
            print(f"   {key} [{lo},{hi}) n={len(sel):3d} acc={np.mean(sel):.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    rows = rows_for(samples)
    for src in sorted({r["src"] for r in rows}):
        rs = [r for r in rows if r["src"] == src]
        print(src, len(rs), f"acc={np.mean([r['ok'] for r in rs]):.3f}")
        if src != "3d":
            table(rs, "oconf", [0, .5, .6, .7, .8, 1.01])
            continue
        table(rs, "hconf", [0, .5, .7, .9, .97, 1.01])
        table(rs, "fconf", [0, .2, .4, .6, 1.01])
        table(rs, "ax", [0, 1, 2, 3, 4, 20])
        table(rs, "vx", [0, 3, 6, 10, 100])
        table(rs, "oconf", [0, .5, .6, .7, .8, 1.01])
        for reason in ("in", "out", "net"):
            s = [r["ok"] for r in rs if r["reason"] == reason]
            print(f"   reason={reason} n={len(s)} acc={np.mean(s) if s else float('nan'):.3f}")
        s = [r["ok"] for r in rs if r.get("dir_ok") is False]
        print(f"   dir disagrees n={len(s)} acc={np.mean(s) if s else float('nan'):.3f}")


if __name__ == "__main__":
    main()
