#!/usr/bin/env python
"""Before/after rally metrics (metrics.rally_metrics, pooled like benchmark.py)
and rally-stage latency on the same cached upstream: pristine rally.py from
.ui-backup vs the current one. Also the outcome_confidence reliability table.

  .venv/bin/python -m scripts.accuracy.heldout --style behind --seed 1000 --n 40
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from app.services.tracking import metrics as M
from app.services.tracking import rally as R
from scripts.accuracy.ab import load_old
from scripts.accuracy.common import add_common_args, load_samples, sim_args
from scripts.accuracy.rally_errors import match, reliability

GAP = 100_000


def _d(r, off):
    return {"start_frame": r.start_frame + off, "end_frame": r.end_frame + off,
            "end_reason": r.end_reason, "winner": r.winner, "landing_xy": r.landing_xy,
            "margin": getattr(r, "in_out_margin_m", getattr(r, "margin", None))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    old = load_old()
    res = {}
    for name, mod in (("before", old), ("after", R)):
        P, T, rel, secs = [], [], [], 0.0
        for k, d in enumerate(samples):
            t0 = time.perf_counter()
            pred = mod.segment_rallies(d["track"], d["hits"], d["segs"], d["meta"], d["camera"],
                                       players=d["players"])
            secs += time.perf_counter() - t0
            P += [_d(r, k * GAP) for r in pred]
            T += [{"start_frame": t.start_frame + k * GAP, "end_frame": t.end_frame + k * GAP,
                   "end_reason": t.end_reason, "winner": t.winner, "landing_xy": t.landing_xy,
                   "margin": t.margin} for t in d["true_rallies"]]
            for i, j in match(pred, d["true_rallies"]):
                r, t = pred[i], d["true_rallies"][j]
                rel.append((getattr(r, "outcome_confidence", 0.0), r.winner == t.winner,
                            r.end_reason == t.end_reason, r.winner in R.SIDES))
        m = M.rally_metrics(P, T)
        res[name] = {k: m[k] for k in ("n_pred", "n_true", "tp", "precision", "recall", "f1",
                                       "end_reason_acc", "winner_acc", "landing_err_m_median",
                                       "landing_err_m_p95")}
        res[name]["seconds"] = secs
        res[name]["reliability"] = reliability(rel)
        hi = [x for x in rel if x[0] >= 0.8]
        res[name]["conf>=0.8"] = {"n": len(hi), "winner_acc": float(np.mean([x[1] for x in hi])) if hi else None}
    print(f"{args.style} {args.image_size} seeds {args.seed}..{args.seed + args.n - 1} r{args.rallies}")
    for name in ("before", "after"):
        r = res[name]
        print(f"  {name:6s} F1 {r['f1']:.3f} (P {r['precision']:.3f} R {r['recall']:.3f} pred {r['n_pred']}/{r['n_true']})"
              f" end {r['end_reason_acc']:.3f} win {r['winner_acc']:.3f} land med {r['landing_err_m_median']:.3f}"
              f" p95 {r['landing_err_m_p95']:.2f} | conf>=0.8 n={r['conf>=0.8']['n']} win={r['conf>=0.8']['winner_acc']}"
              f" | rally stage {r['seconds']:.2f} s")
    print("  reliability (after):")
    for row in res["after"]["reliability"]:
        print("    ", row)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
