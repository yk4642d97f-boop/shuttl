#!/usr/bin/env python
"""Every rally that differs between two rally.py files on the same inputs: cached sim upstream
(both styles, given seeds), the tests/test_rally generator (30/60/120 fps) and the 45-min
endurance runs. Prints the differing rallies with the before landing and outcome source.

  .venv/bin/python -m scripts.accuracy.rally_diff --before ../.ui-backup/accuracy2-20260928/phase3b/backend/app/services/tracking/rally.py
"""
from __future__ import annotations

import argparse

from app.services.tracking import rally as R
from scripts.accuracy.module_ab import load


def key(r):
    return (r.start_frame, r.end_frame, len(r.shots), r.winner, r.end_reason, r.landing_xy,
            round(float(r.confidence), 12), getattr(r, "outcome_source", None))


def diff(name, pa, pb, out):
    sa, sb = [key(r) for r in pa], [key(r) for r in pb]
    if sa == sb:
        return 0
    n = 0
    for x in sa:
        if x not in sb:
            y = next((k for k in sb if k[0] == x[0]), None)
            out.append(f"  {name}: start {x[0]}: before {x[3]}/{x[4]} land {x[5]} src {x[7]}  ->  after "
                       f"{None if y is None else (y[3], y[4], y[5], y[7])}")
            n += 1
    return max(n, 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1000, 2000])
    a = ap.parse_args(argv)
    B = load(a.before, "rally_before")
    out, tot = [], {}
    from scripts.accuracy.common import load_samples, sim_args
    for seed in a.seeds:
        for st in ("behind", "random"):
            n = 0
            for d in load_samples(sim_args(st, "1920x1080", 60.0, 5), list(range(seed, seed + 40)), 1):
                args = (d["track"], d["hits"], d["segs"], d["meta"], d["camera"])
                n += diff(f"{st} s{d['seed']}", B.segment_rallies(*args, players=d["players"]),
                          R.segment_rallies(*args, players=d["players"]), out)
            tot[f"sim {st} s{seed}-{seed + 39}"] = n
    from scripts.accuracy.seg_quality_ab import inputs
    for fps in (30.0, 60.0, 120.0):
        n = 0
        for s in range(20):
            c, hits, src, segs = inputs(s, fps)
            n += diff(f"testgen fps{fps:g} s{s}", B.segment_rallies(c.track, hits, segs, c.meta, c.camera),
                      R.segment_rallies(c.track, hits, segs, c.meta, c.camera), out)
        tot[f"test generator {fps:g} fps"] = n
    from scripts.accuracy.codec_ab import load as load_run
    for nm in ("h264", "hevc"):
        d = load_run(f"/tmp/accC/codec/{nm}.json")
        args = (d["track"], d["hits"], d["segs"], d["meta"], d["camera"])
        tot[f"45-min {nm}"] = diff(f"45min {nm}", B.segment_rallies(*args), R.segment_rallies(*args), out)
    print("\n".join(out))
    for k, v in tot.items():
        print(f"{k:28s} changed rallies: {v}")


if __name__ == "__main__":
    main()
