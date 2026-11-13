#!/usr/bin/env python
"""Tune decoder (threshold, NMS gap, alternation weight) on VAL 041 ONLY.

  tune.py --ckpt ckpt/a.pt [...] [--windowed]   -> prints grid, writes out/tuned_<tag>.json
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch

import common as C
import infer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--tag", default="tuned")
    ap.add_argument("--windowed", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cpu")
    models = infer.load_models(a.ckpt, dev)
    m = C.load_match("041")
    pr = infer.predict_windowed if a.windowed else C.predict_rally
    logits = [pr(models, r["raw"], r["fps"], dev) for r in m["rallies"]]
    rows = []
    thrs = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    gaps = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
    lams = [0.0, 0.5, 1.0, 2.0, 4.0]
    for lam, thr, gap in itertools.product(lams, thrs, gaps):
        P = dict(thr=thr, gap_s=gap, lam=lam)
        dets = [C.detections_from_logits(lg, r["fps"], r["frames"], P) for lg, r in zip(logits, m["rallies"])]
        res, srv, rly = C.eval_dets(m["rallies"], dets)
        rows.append(dict(P, f1_2=C.f1_of(res, 2), f1_1=C.f1_of(res, 1), f1_4=C.f1_of(res, 4)))
    for lam in lams:
        sub = [r for r in rows if r["lam"] == lam]
        b = max(sub, key=lambda r: r["f1_2"])
        print("lam %.1f best:" % lam, {k: round(v, 4) for k, v in b.items()})
    best_nolam = max([r for r in rows if r["lam"] == 0], key=lambda r: r["f1_2"])
    best_all = max(rows, key=lambda r: r["f1_2"])
    json.dump(dict(best_nolam=best_nolam, best_all=best_all, rows=rows, ckpts=a.ckpt, windowed=a.windowed),
              open(os.path.join(C.OUT, a.tag + ".json"), "w"), indent=1)
    print("BEST no-alt:", best_nolam)
    print("BEST overall:", best_all)


if __name__ == "__main__":
    main()
