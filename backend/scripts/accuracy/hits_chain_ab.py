#!/usr/bin/env python
"""Whole-chain A/B of a hits.py option (hits -> reconstruct3d -> rally -> shots,
current rally.py): upstream caches for SHUTTL_HITS_OVERRIDE='' (default) and
the given override, then hits F1 / side and the stage_ab rally metrics for both.

  .venv/bin/python -m scripts.accuracy.hits_chain_ab --style behind --seed 0 --n 40 --override hidden_contacts=False
"""
from __future__ import annotations

import argparse
import os

from app.services.tracking import metrics as M
from scripts.accuracy.common import add_common_args, load_samples, sim_args
from scripts.accuracy.stage_ab import evaluate, fmt

GAP = 100_000


def hits_f1(samples):
    P, T = [], []
    for k, d in enumerate(samples):
        P += [{"frame": h.frame + k * GAP, "hitter": h.hitter} for h in d["hits"]]
        T += [{"frame": h["frame"] + k * GAP, "hitter": h["hitter"]} for h in d["true_hits"]]
    return M.hit_metrics(P, T, 3)


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--override", default="hidden_contacts=False")
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    seeds = list(range(args.seed, args.seed + args.n))
    for name, ov in ((args.override, args.override), ("default", "")):
        os.environ["SHUTTL_HITS_OVERRIDE"] = ov
        s = load_samples(a, seeds, args.workers)
        h = hits_f1(s)
        r = evaluate(s)
        print(f"{args.style} s{args.seed}.. [{name}] hits F1 {h['f1']:.3f} P {h['precision']:.3f} R {h['recall']:.3f} side {h['hitter_acc']:.3f}")
        print(f"      rally {fmt(r)}")
        for row in r["reliability"]:
            print("        ", row)


if __name__ == "__main__":
    main()
