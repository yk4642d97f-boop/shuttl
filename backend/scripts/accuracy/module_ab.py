#!/usr/bin/env python
"""Rally-stage before/after of two rally.py FILES on one cached upstream (common.py):
F1 / end / winner / side / shots / reliability for each, plus a paired
bootstrap (per match) of the winner-accuracy and rally-F1 differences.

  .venv/bin/python -m scripts.accuracy.module_ab --style behind --seed 1000 --n 40 \
      --before ../.ui-backup/accuracy2-20260928/phase3/backend/app/services/tracking/rally.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys

import numpy as np

from app.services.tracking import rally as R
from scripts.accuracy.common import add_common_args, load_samples, sim_args, stage_hash
from scripts.accuracy.rally_errors import match
from scripts.accuracy.stage_ab import evaluate, fmt


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def per_match(samples, mod, cfg=None):
    """Per match: (tp, n_pred, n_true, winner_ok, matched)."""
    out = []
    for d in samples:
        pred = mod.segment_rallies(d["track"], d["hits"], d["segs"], d["meta"], d["camera"],
                                   players=d["players"], config=cfg)
        m = match(pred, d["true_rallies"])
        win = sum(pred[i].winner == d["true_rallies"][j].winner for i, j in m)
        out.append((len(m), len(pred), len(d["true_rallies"]), win, len(m)))
    return np.array(out, float)


def paired_boot(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0):
    """95 % CI of (b - a) for rally F1 and winner accuracy, resampling matches."""
    rng = np.random.default_rng(seed)
    k = len(a)
    idx = rng.integers(0, k, (n, k))
    def f1(x):
        return 2 * x[..., 0].sum(-1) / (x[..., 1].sum(-1) + x[..., 2].sum(-1))
    def win(x):
        return x[..., 3].sum(-1) / np.maximum(x[..., 4].sum(-1), 1)
    res = {}
    for name, fn in (("f1", f1), ("win", win)):
        d = fn(b[idx]) - fn(a[idx])
        res[name] = (float(fn(b) - fn(a)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", default=None, help="default: the current rally.py")
    ap.add_argument("--no-shots", action="store_true")
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    mb = load(args.before, "rally_before")
    ma = load(args.after, "rally_after") if args.after else R
    print(f"{args.style} s{args.seed}..{args.seed + args.n - 1} upstream {stage_hash()}")
    for name, mod in (("before", mb), ("after", ma)):
        r = evaluate(samples, rally_module=mod, with_shots=not args.no_shots)
        print(f"  {name:6s} {fmt(r)}")
    ci = paired_boot(per_match(samples, mb), per_match(samples, ma))
    print("  paired bootstrap (after - before, 95 % CI): " +
          "  ".join(f"{k} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]" for k, v in ci.items()))


if __name__ == "__main__":
    main()
