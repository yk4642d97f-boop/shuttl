#!/usr/bin/env python
"""Final before/after on ONE upstream (current tree unless SHUTTL_ACC_HASH pins
a cache): the rally.py at the start of this work (.ui-backup/accuracy2-20260928,
= Astra's final) vs the current one, rally + shots + side + reliability.

  .venv/bin/python -m scripts.accuracy.final_ab --style behind --seed 1000 --n 40
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from app.services.tracking import rally as R
from scripts.accuracy.common import ROOT, add_common_args, load_samples, sim_args, stage_hash
from scripts.accuracy.stage_ab import evaluate, fmt

BEFORE = ROOT.parent / ".ui-backup/accuracy2-20260928/backend/app/services/tracking/rally.py"


def load_before():
    spec = importlib.util.spec_from_file_location("rally_before", BEFORE)
    m = importlib.util.module_from_spec(spec)
    sys.modules["rally_before"] = m
    spec.loader.exec_module(m)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    print(f"{args.style} s{args.seed}..{args.seed + args.n - 1} upstream {stage_hash()}")
    for name, mod in (("before", load_before()), ("after", R)):
        r = evaluate(samples, rally_module=mod)
        print(f"  {name:6s} {fmt(r)}")
        for row in r["reliability"]:
            print("        ", row)


if __name__ == "__main__":
    main()
