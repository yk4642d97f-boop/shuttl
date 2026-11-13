#!/usr/bin/env python
"""Missed final contacts: for every true rally's LAST contact, is it among
hits.py's output, among its scored candidates (before decoding), and if a
candidate exists, why was it not decoded (score, weak, seg, neighbours)?
Also: extra detected hits after the true last contact (before the next serve).

  .venv/bin/python -m scripts.accuracy.final_contact --style behind --seed 0 --n 40
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from app.services.tracking import hits as H
from scripts.accuracy.common import add_common_args, load_samples, sim_args
from scripts.accuracy.side_errors import decode


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--dump", type=int, default=0)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    c = Counter()
    rows = []
    for d in samples:
        cands, idx, pn, dec = decode(d)
        cf = np.array([x.frame for x in cands], int) if cands else np.zeros(0, int)
        hf = np.array([h.frame for h in d["hits"]], int)
        chosen = set(idx)
        for r in d["true_rallies"]:
            k = r.first_shot + r.n_shots - 1
            th = d["true_hits"][k]
            f = int(th["frame"])
            c["rallies"] += 1
            det = len(hf) and np.min(np.abs(hf - f)) <= 3
            if det:
                c["last_detected"] += 1
            else:
                j = int(np.argmin(np.abs(cf - f))) if len(cf) else -1
                if j >= 0 and abs(cf[j] - f) <= 3:
                    x = cands[j]
                    score = float(dec.hit[j])
                    why = "gated" if j in chosen else "viterbi"
                    c[f"missed: cand {why} weak={x.weak} start={x.is_start}"] += 1
                    c["missed_cand_score<-4" if score < -4 else ("missed_cand_score<0" if score < 0 else "missed_cand_score>=0")] += 1
                    rows.append((d["seed"], f, round(score, 2), round(x.kin, 2), round(x.reach, 2),
                                 round(x.lam_v, 1), x.static_after, x.settles, x.outside, x.weak,
                                 round(float(np.linalg.norm(x.v_out)) / np.hypot(d["meta"].width, d["meta"].height), 3)))
                else:
                    c["missed: no candidate"] += 1
            # extra detections after the true last contact, before the rally end
            extra = [g for g in hf if f + 3 < g <= r.end_frame]
            if extra:
                c["rallies_with_extra_hits_after_last"] += 1
    for k, v in sorted(c.items()):
        print(f"{v:5d} {k}")
    print("seed frame score kin reach lam_v static_after settles outside weak s_out")
    for r in rows[:args.dump]:
        print(r)


if __name__ == "__main__":
    main()
