#!/usr/bin/env python
"""Upper bound for a pipeline-level iteration: decode sides in the rally stage,
re-run reconstruct3d with the decoded labels (so every fit is made with the
corrected side prior), then segment rallies again. Costs one extra
reconstruct3d pass; measures what an orchestration change would buy.

  .venv/bin/python -m scripts.accuracy.refit_oracle --style behind --seed 0 --n 40
"""
from __future__ import annotations

import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from app.services.tracking import rally as R
from scripts.accuracy.common import CACHE, add_common_args, load_samples, sim_args, stage_hash
from scripts.accuracy.stage_ab import evaluate, fmt


def decoded_labels(d: dict, cfg: R.RallyConfig) -> dict[int, str]:
    mo = R._Motion(d["track"], float(d["meta"].height), cfg)
    hs = R._clean_hits(d["hits"], mo.n)
    hs = [h if h.t is not None and np.isfinite(h.t) else replace(h, t=float(mo.t[h.frame])) for h in hs]
    segs = R._match_segments(hs, list(d["segs"] or []), mo.t, cfg)
    hs, segs, conts = R._merge_same_flight(hs, segs, mo.t, cfg)
    groups, _ = R._group_hits(hs, segs, mo, cfg, conts)
    out, _ = R._decode_sides(hs, segs, conts, groups, mo.t, cfg)
    return {h.frame: h.hitter for h in out}


def _refit(job):
    d, tag = job
    from app.services.tracking import reconstruct3d
    hits2, _ = R.refine_hitters(d["track"], d["hits"], d["segs"], d["meta"], d["camera"], d["players"])
    segs2 = reconstruct3d.reconstruct_segments(d["track"], hits2, d["camera"], d["meta"], d["players"]) or []
    return d["seed"], hits2, segs2


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    base = evaluate(samples, with_shots=False)
    print("one pass ", fmt(base))
    import hashlib
    rs = int(hashlib.sha1(Path("app/services/tracking/rally.py").read_bytes()).hexdigest()[:6], 16)
    ck = CACHE / f"refit_{args.style}_s{args.seed}_n{args.n}_{stage_hash()}_{rs:x}.pkl"
    if ck.exists():
        res = pickle.load(open(ck, "rb"))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(_refit, [(d, "") for d in samples]))
        pickle.dump(res, open(ck, "wb"))
    by = {s: (h, g) for s, h, g in res}
    s2 = [dict(d, hits=by[d["seed"]][0], segs=by[d["seed"]][1]) for d in samples]
    new = evaluate(s2, with_shots=False)
    print("refitted ", fmt(new))
    old = evaluate(s2, with_shots=False, config=R.RallyConfig(joint_sides=False))
    print("refit, no joint in 2nd rally pass ", fmt(old))


if __name__ == "__main__":
    main()
