#!/usr/bin/env python
"""Run the learned hit detector and write research/hits-style JSON into hitnet/out/.

  infer.py --ckpt ckpt/a.pt [ckpt/b.pt ...] --matches 039 040 041 042 043 044
           [--thr 0.4] [--gap 0.45] [--lam 0] [--out out] [--windowed] [--device cpu]

Output frames are the model's frames as-is (score them with `score.py --shift 0`).
Several checkpoints = logit-average ensemble.
"""
import argparse
import json
import os

import numpy as np
import torch

import common as C


def load_models(paths, dev):
    ms = []
    for p in paths:
        ck = torch.load(p, map_location="cpu")
        m = C.make_model(ck["arch"], ck["n_in"], ck["hidden"])
        m.load_state_dict(ck["state"])
        ms.append(m.to(dev).eval())
    return ms


@torch.no_grad()
def predict_windowed(models, raw, fps, dev, win=384, stride=192):
    """Overlapping-window inference (centre of each window only); removes any dependence on
    the rally's absolute start/end position beyond a 384-frame window."""
    T = len(raw)
    X = torch.from_numpy(C.derive(raw, fps))
    if T <= win:
        return C.predict_rally(models, raw, fps, dev)
    out = np.zeros((T, 2), np.float32)
    starts = list(range(0, T - win + 1, stride))
    if starts[-1] != T - win:
        starts.append(T - win)
    margin = (win - stride) // 2
    for k, a in enumerate(starts):
        xb = X[a:a + win][None].to(dev)
        lg = np.mean([m(xb)[0].float().cpu().numpy() for m in models], 0)
        lo = 0 if k == 0 else margin
        hi = win if k == len(starts) - 1 else win - margin
        out[a + lo:a + hi] = lg[lo:hi]
    return out


def run_match(models, mid, params, dev, windowed=False, logit_cache=None, w=(384,), st=(192,)):
    m = C.load_match(mid)
    dets, logits = [], []
    for r in m["rallies"]:
        lg = (predict_windowed(models, r["raw"], r["fps"], dev, w[0], st[0]) if windowed
              else C.predict_rally(models, r["raw"], r["fps"], dev))
        logits.append(lg)
        dets.append(C.detections_from_logits(lg, r["fps"], r["frames"], params))
    return m, dets, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--matches", nargs="+", required=True)
    ap.add_argument("--thr", type=float, default=0.4)
    ap.add_argument("--gap", type=float, default=0.45)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--out", default=C.OUT)
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--win", type=int, default=384)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    models = load_models(a.ckpt, dev)
    params = dict(thr=a.thr, gap_s=a.gap, lam=a.lam)
    os.makedirs(a.out, exist_ok=True)
    thr_logit = float(np.log(a.thr / (1 - a.thr)))
    for mid in a.matches:
        m, dets, _ = run_match(models, mid, params, dev, a.windowed, w=(a.win,), st=(a.stride,))
        C.write_match_json(os.path.join(a.out, mid + ".hits.json"), mid, m["label_dir"],
                           m["rallies"], dets, thr_logit)
        print(mid, sum(len(d) for d in dets), "detections")


if __name__ == "__main__":
    main()
