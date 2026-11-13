#!/usr/bin/env python
"""Per-rally differences between two RallyConfigs on the same cached upstream:
matched true rally -> (end_reason, winner, source, outcome_conf, last hitter,
true last hitter) under A and B, printed where they differ.

  .venv/bin/python -m scripts.accuracy.cfg_diff --style random --seed 0 --n 40 --ab joint_sides=False
"""
from __future__ import annotations

import argparse
from collections import Counter

from app.services.tracking import rally as R
from scripts.accuracy.common import add_common_args, load_samples, sim_args
from scripts.accuracy.rally_errors import match


def parse_cfg(items: list[str]) -> R.RallyConfig:
    kw = {}
    for kv in items:
        k, v = kv.split("=")
        kw[k] = (v == "True") if v in ("True", "False") else type(getattr(R.RallyConfig(), k))(v)
    return R.RallyConfig(**kw)


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--ab", nargs="*", default=[])
    ap.add_argument("--bb", nargs="*", default=[])
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    ca, cb = parse_cfg(args.ab), parse_cfg(args.bb)
    tally = Counter()
    for d in samples:
        res = {}
        for name, cfg in (("A", ca), ("B", cb)):
            pred = R.segment_rallies(d["track"], d["hits"], d["segs"], d["meta"], d["camera"],
                                     players=d["players"], config=cfg)
            res[name] = {j: pred[i] for i, j in match(pred, d["true_rallies"])}
        for j, t in enumerate(d["true_rallies"]):
            th = d["true_hits"][t.first_shot + t.n_shots - 1]["hitter"]
            row = {}
            for name in ("A", "B"):
                r = res[name].get(j)
                row[name] = None if r is None else (r.end_reason, r.winner, r.outcome_source,
                                                    round(r.outcome_confidence, 2),
                                                    r.shots[-1].hit.hitter if r.shots else None)
            if row["A"] != row["B"]:
                okA = row["A"] is not None and row["A"][1] == t.winner
                okB = row["B"] is not None and row["B"][1] == t.winner
                tally[f"win A={okA} B={okB}"] += 1
                print(f"seed {d['seed']} r{j} true=({t.end_reason},{t.winner},last {th}) A={row['A']} B={row['B']}")
    print(tally)


if __name__ == "__main__":
    main()
