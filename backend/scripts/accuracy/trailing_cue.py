#!/usr/bin/env python
"""Rule candidates for dropping a spurious TRAILING contact (the last shot of a
multi-shot rally, after the true final contact): hit confidence x observed
flight time after it (until rest). Counts real last contacts vs trailing
spurious ones per rule, on dev seeds (both styles).

  .venv/bin/python -m scripts.accuracy.trailing_cue --seed 0
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from app.services.tracking import rally as R
from scripts.accuracy.common import load_samples, sim_args


def rows(seed: int, n: int):
    cfg = R.RallyConfig()
    out = []
    for style in ("behind", "random"):
        for d in load_samples(sim_args(style, "1920x1080", 60.0, 5), list(range(seed, seed + n)), 4):
            mo = R._Motion(d["track"], float(d["meta"].height), cfg)
            pred = R.segment_rallies(d["track"], d["hits"], d["segs"], d["meta"], d["camera"], players=d["players"], config=R.RallyConfig(trail_max_conf=0.0, onehit_min_flight_s=0.0))
            tf = np.array([h["frame"] for h in d["true_hits"]])
            tr = d["true_rallies"]
            for r in pred:
                if len(r.shots) < 2:
                    continue
                s = r.shots[-1]
                f = s.hit.frame
                real = bool(len(tf) and np.min(np.abs(tf - f)) <= 3)
                if not real:
                    ins = [t for t in tr if t.start_frame - 3 <= f <= t.end_frame + 3]
                    if not ins or f <= d["true_shots"][ins[0].first_shot + ins[0].n_shots - 1].frame + 3:
                        continue
                fr = mo.first_rest_after(f, min(mo.n, mo.frame_at(s.hit.t + 4.0) + 1))
                end = fr if fr is not None else mo.frame_at(s.hit.t + 4.0)
                mv = mo.fin[f + 1:end] & ~mo.rest[f + 1:end]
                disp = float(np.linalg.norm(mo.xy[fr] - np.asarray(s.hit.xy_px))) / mo.h if fr is not None else float("nan")
                tk = int(np.argmin(np.abs(tf - f))) if len(tf) else -1
                net = real and tk >= 0 and any(t.first_shot + t.n_shots - 1 == tk and t.end_reason == "net" for t in tr)
                out.append((real, s.hit.confidence, float(mv.sum()) / (d["meta"].fps or 60.0), style, disp, net))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args(argv)
    R_ = rows(args.seed, args.n)
    nr = sum(r[0] for r in R_)
    ns = len(R_) - nr
    print(f"real last contacts {nr}, trailing spurious {ns}")
    for kind, sel in (("real net-fault last", [r for r in R_ if r[0] and r[5]]), ("real other last", [r for r in R_ if r[0] and not r[5]]),
                      ("trailing spurious", [r for r in R_ if not r[0]])):
        dd = np.array([r[4] for r in sel]); dd = dd[np.isfinite(dd)]
        fl = np.array([r[2] for r in sel])
        print(f"  {kind:20s} n={len(sel):3d} contact->rest disp/H p10/50/90 {np.percentile(dd, [10, 50, 90]).round(3) if len(dd) else None}"
              f"  flight p10/50 {np.percentile(fl, [10, 50]).round(2) if len(fl) else None}")
    for dthr in (0.03, 0.05, 0.08, 0.12):
        drop = [r for r in R_ if r[1] < 0.8 and r[2] < 0.6 and np.isfinite(r[4]) and r[4] < dthr]
        dr = sum(r[0] for r in drop)
        print(f"  conf<0.8 & flight<0.6 & disp<{dthr}: drops spurious {len(drop) - dr}/{ns}, real {dr}/{nr}")
    for cthr in (0.6, 0.7, 0.8, 0.9):
        for fthr in (0.3, 0.45, 0.6, 9.0):
            drop = [r for r in R_ if r[1] < cthr and r[2] < fthr]
            dr = sum(r[0] for r in drop)
            print(f"  conf < {cthr} and flight < {fthr:4.2f} s: drops spurious {len(drop) - dr:3d}/{ns}, real {dr:3d}/{nr}")


if __name__ == "__main__":
    main()
