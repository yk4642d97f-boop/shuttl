#!/usr/bin/env python
"""DELIVERY_R3 shot-threshold proposal on the benchmark chain (cached upstream -> rally ->
shots.classify_shots): shot-type accuracy on shots matched to true hits (+-3 frames,
stage_ab definition), per-class recall, paired bootstrap per match vs the current
thresholds. The variants patch shots._scores_3d with R3's parameterised copy
(flightref.app_thresholds.scores; parity with the app's own function is checked first).

  .venv/bin/python -m scripts.accuracy.shot_thresholds_ab --style behind --seed 1000 --n 40
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np

from app.services.tracking import metrics as M
from app.services.tracking import rally as R
from app.services.tracking import shots as SH
from scripts.accuracy.common import ROOT, add_common_args, load_samples, sim_args

sys.path.insert(0, str(ROOT.parent / "deliverables/DELIVERY_R3/src"))
from flightref.app_thresholds import CURRENT, PROPOSED, scores  # noqa: E402

ORIG = SH._scores_3d


def run(samples, rallies, params: dict | None):
    """Per match (correct, n) and a Counter of (true, pred) over matched shots."""
    if params is None:
        SH._scores_3d = ORIG
    else:
        P = {**{k: v for k, (v, _) in CURRENT.items()}, **params}
        SH._scores_3d = lambda f, oh: scores(SH, f, oh, P)
    per, conf = [], Counter()
    try:
        for d, pred in zip(samples, rallies):
            typed = [SH.classify_shots(r, d["camera"], d["players"]) for r in pred]
            ts = [s for r in typed for s in r.shots]
            th = d["true_hits"]
            pairs = M._match_events(np.array([s.hit.frame for s in ts], int),
                                    np.array([h["frame"] for h in th], int), 3)
            ok = 0
            for i, j in pairs:
                conf[(th[j]["shot_type"], ts[i].shot_type)] += 1
                ok += ts[i].shot_type == th[j]["shot_type"]
            per.append((ok, len(pairs)))
    finally:
        SH._scores_3d = ORIG
    return np.array(per, float), conf


def recall(conf: Counter) -> dict:
    tot = Counter()
    for (t, _), n in conf.items():
        tot[t] += n
    return {t: conf[(t, t)] / n for t, n in sorted(tot.items())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), 1)
    rallies = [R.segment_rallies(d["track"], d["hits"], d["segs"], d["meta"], d["camera"], players=d["players"])
               for d in samples]
    cur = {k: v for k, (v, _) in CURRENT.items()}
    base, cb = run(samples, rallies, None)
    par, _ = run(samples, rallies, {})
    assert (base == par).all(), "parity: R3's parameterised scorer != shots._scores_3d"
    variants = [(f"only {k}={v}", {k: v}) for k, v in PROPOSED.items()] + [("all four", dict(PROPOSED))]
    rb = recall(cb)
    print(f"{args.style} s{args.seed}..{args.seed + args.n - 1}: shots matched {int(base[:, 1].sum())}, "
          f"current acc {base[:, 0].sum() / base[:, 1].sum():.4f} (current = {cur})")
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(samples), (args.boot, len(samples)))
    for name, chg in variants:
        v, cv = run(samples, rallies, chg)
        acc_b, acc_v = base[:, 0].sum() / base[:, 1].sum(), v[:, 0].sum() / v[:, 1].sum()
        d = v[idx, 0].sum(1) / v[idx, 1].sum(1) - base[idx, 0].sum(1) / base[idx, 1].sum(1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        rv = recall(cv)
        worst = min(((rv.get(k, 0) - rb[k], k) for k in rb), default=(0, "-"))
        print(f"  {name:22s} acc {acc_b:.4f} -> {acc_v:.4f} ({acc_v - acc_b:+.4f}) 95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"{'  CI>0' if lo > 0 else ''}  worst class {worst[1]} {worst[0]:+.3f}")
        print("      recall " + " ".join(f"{k} {rb[k]:.3f}->{rv.get(k, 0):.3f}" for k in rb if abs(rv.get(k, 0) - rb[k]) > 1e-9))


if __name__ == "__main__":
    main()
