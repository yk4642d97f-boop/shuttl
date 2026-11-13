#!/usr/bin/env python
"""A/B: pristine rally.py (backup) vs current, same cached upstream. Prints
per-rally differences in winner / end_reason.

  .venv/bin/python -m scripts.accuracy.ab --style random --image-size 1280x720 --rallies 1 --n 20
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

from app.services.tracking import rally as R
from scripts.accuracy.common import ROOT, add_common_args, load_samples, sim_args
from scripts.accuracy.rally_errors import match

BACKUP = ROOT.parent / ".ui-backup/handoff-20260928/backend/app/services/tracking/rally.py"


def load_old():
    spec = importlib.util.spec_from_file_location("rally_old", BACKUP)
    m = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["rally_old"] = m
    spec.loader.exec_module(m)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    old = load_old()
    for d in samples:
        res = {}
        for name, mod in (("old", old), ("new", R)):
            pred = mod.segment_rallies(d["track"], d["hits"], d["segs"], d["meta"], d["camera"],
                                       players=d["players"])
            res[name] = {j: pred[i] for i, j in match(pred, d["true_rallies"])}
        for j, t in enumerate(d["true_rallies"]):
            o, n = res["old"].get(j), res["new"].get(j)
            fo = None if o is None else (o.end_reason, o.winner, getattr(o, "outcome_source", ""), round(o.outcome_confidence, 2), len(o.shots))
            fn = None if n is None else (n.end_reason, n.winner, getattr(n, "outcome_source", ""), round(n.outcome_confidence, 2), len(n.shots))
            if fo != fn:
                print(f"seed {d['seed']} rally {j} true=({t.end_reason},{t.winner},{t.n_shots}) old={fo} new={fn}")


if __name__ == "__main__":
    main()
