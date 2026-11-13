#!/usr/bin/env python
"""Rally / shots stage CPU time (time.process_time, best of 3, before/after
interleaved) on cached upstream. Before = .ui-backup pristine rally.py.

  .venv/bin/python -m scripts.accuracy.latency --style behind --seed 1000 --n 40
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from app.services.tracking import rally as R
from app.services.tracking import shots as SH
from scripts.accuracy.ab import load_old
from scripts.accuracy.common import add_common_args, load_samples, sim_args


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args(argv)
    samples = load_samples(sim_args(args.style, args.image_size, args.fps, args.rallies),
                           list(range(args.seed, args.seed + args.n)), args.workers)
    import os
    if os.environ.get("SHUTTL_OLD_RALLY"):
        from scripts.accuracy import ab
        ab.BACKUP = Path(os.environ["SHUTTL_OLD_RALLY"])
    old = load_old()
    best = {"before": 1e9, "after": 1e9, "shots": 1e9}
    for _ in range(3):
        for name, mod in (("before", old), ("after", R)):
            t = time.process_time()
            for d in samples:
                mod.segment_rallies(d["track"], d["hits"], d["segs"], d["meta"], d["camera"],
                                    players=d["players"])
            best[name] = min(best[name], time.process_time() - t)
        rs = [(d, R.segment_rallies(d["track"], d["hits"], d["segs"], d["meta"], d["camera"],
                                    players=d["players"])) for d in samples]
        t = time.process_time()
        for d, rr in rs:
            for r in rr:
                SH.classify_shots(r, d["camera"], d["players"])
        best["shots"] = min(best["shots"], time.process_time() - t)
    frames = sum(d["n"] for d in samples)
    print(f"{args.style}: {frames} frames | rally before {best['before']:.3f} s, after {best['after']:.3f} s "
          f"({100 * (best['after'] / best['before'] - 1):+.1f} %) | shots {best['shots']:.3f} s (shots.py unchanged)")


if __name__ == "__main__":
    main()
