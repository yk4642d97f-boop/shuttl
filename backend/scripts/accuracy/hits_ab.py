#!/usr/bin/env python
"""hits.py A/B on the cached linker tracks (no 3-D): event F1 and hitter-side
accuracy (benchmark.py definitions, pooled) for HitConfig overrides.

  .venv/bin/python -m scripts.accuracy.hits_ab --style behind --seed 0 --n 40 --ab pair_y_weight=0
"""
from __future__ import annotations

import argparse
import time

from app.services.tracking import hits as H
from app.services.tracking import metrics as M
from scripts.accuracy.common import add_common_args, load_samples, sim_args

GAP = 100_000


def run(samples, cfg: H.HitConfig, module=H) -> dict:
    P, T = [], []
    secs = 0.0
    for k, d in enumerate(samples):
        t0 = time.process_time()
        hits = module.detect_hits(d["track"], d["meta"], d["camera"], d["players"], cfg)
        secs += time.process_time() - t0
        P += [{"frame": h.frame + k * GAP, "hitter": h.hitter} for h in hits]
        T += [{"frame": h["frame"] + k * GAP, "hitter": h["hitter"]} for h in d["true_hits"]]
    m = M.hit_metrics(P, T, 3)
    m["secs"] = secs
    return m


def parse(items):
    kw = {}
    for kv in items:
        k, v = kv.split("=")
        cur = getattr(H.HitConfig(), k)
        kw[k] = (v == "True") if isinstance(cur, bool) else type(cur)(v) if cur is not None else float(v)
    return kw


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--ab", nargs="*", default=[])
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    for name, cfg in (("base", H.HitConfig()), (" ".join(args.ab), H.HitConfig(**parse(args.ab)))):
        m = run(samples, cfg)
        print(f"{args.style} [{name}] F1 {m['f1']:.3f} P {m['precision']:.3f} R {m['recall']:.3f} "
              f"side {m['hitter_acc']:.3f} (known {m['hitter_acc_known']:.3f}) tp {m['tp']} | {m['secs']:.1f}s")


if __name__ == "__main__":
    main()
