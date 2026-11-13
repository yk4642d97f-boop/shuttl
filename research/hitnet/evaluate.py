#!/usr/bin/env python
"""Final held-out comparison: learned detector (shift 0) vs rule baseline (shift 2) scored
by the SAME research/hits/score.py pairing on the SAME matches.  Decoder params come from
tune.py (val 041 only); nothing here tunes on test.

  evaluate.py --ckpt ckpt/a.pt [...] --thr .. --gap .. --lam .. [--windowed] [--tag name]
"""
import argparse
import collections
import json
import os
import time

import numpy as np
import torch

import common as C
import infer

S = C.scorelib


def load_dets_json(path, shift):
    m = json.load(open(path))
    by = collections.defaultdict(list)
    for r in m["rallies"]:
        by[(r["set"], r["rally"])] = sorted(
            [dict(h, frame=h["frame"] + shift) for h in r["hits"]], key=lambda x: x["frame"])
    return m, by


def tally(files, shift, tols=(1, 2, 4)):
    """Official pairing over the CSV labels. Returns per-match + pooled counts, serve/rally recall,
    offset histogram (+-4)."""
    per = {}
    pool = {t: [0, 0, 0] for t in tols}
    srv = [0, 0]
    rly = [0, 0]
    off = collections.Counter()
    for mid, path in files:
        m, by = load_dets_json(path, shift)
        gt = collections.defaultdict(list)
        for s, rl, fr, ty in S.load_labels(C.LABELS_ROOT, m["labelDir"]):
            gt[(s, rl)].append(fr)
        c = {t: [0, 0, 0] for t in tols}
        ms = [0, 0]
        mr = [0, 0]
        for key in set(gt) | set(by):
            gf = sorted(gt.get(key, []))
            d = by.get(key, [])
            kept, miss, free = S.pair(gf, d, max(tols))
            for t in tols:
                k = [x for x in kept if abs(x[2]) <= t]
                c[t][0] += len(k); c[t][1] += len(d) - len(k); c[t][2] += len(gf) - len(k)
            k2 = {x[0] for x in kept if abs(x[2]) <= 2}
            for x in kept:
                off[x[2]] += abs(x[2]) <= 4
            for gi in range(len(gf)):
                tgt = ms if gi == 0 else mr
                tgt[1] += 1
                tgt[0] += gi in k2
        per[mid] = dict(c=c, srv=ms, rly=mr)
        for t in tols:
            for i in range(3):
                pool[t][i] += c[t][i]
        srv[0] += ms[0]; srv[1] += ms[1]; rly[0] += mr[0]; rly[1] += mr[1]
    return per, pool, srv, rly, off


def fmt(c):
    p, r, f = S.prf(*c)
    return "%.3f/%.3f/%.3f" % (f, p, r)


def report(name, ids, learned_dir, rule_dir):
    L = tally([(i, os.path.join(learned_dir, i + ".hits.json")) for i in ids], 0)
    R = tally([(i, os.path.join(rule_dir, i + ".hits.json")) for i in ids], 2)
    lines = ["### %s  (matches: %s)" % (name, " ".join(ids)),
             "F1/P/R, official pairing; learned shift 0, rule shift +2", "",
             "| match | det | tol | rule F1/P/R | learned F1/P/R |", "|---|---|---|---|---|"]
    for i in ids + ["POOLED"]:
        for t in (1, 2, 4):
            rc = R[1][t] if i == "POOLED" else R[0][i]["c"][t]
            lc = L[1][t] if i == "POOLED" else L[0][i]["c"][t]
            lines.append("| %s | | ±%d | %s | %s |" % (i, t, fmt(rc), fmt(lc)))
    lines += ["", "serve recall @±2 (first labelled hit of rally): rule %d/%d = %.3f, learned %d/%d = %.3f"
              % (R[2][0], R[2][1], R[2][0] / R[2][1], L[2][0], L[2][1], L[2][0] / L[2][1]),
              "non-serve recall @±2: rule %d/%d = %.3f, learned %d/%d = %.3f"
              % (R[3][0], R[3][1], R[3][0] / R[3][1], L[3][0], L[3][1], L[3][0] / L[3][1]),
              "learned counts pooled @±2 (tp,fp,fn): %s ; rule: %s" % (L[1][2], R[1][2]),
              "learned matched-offset histogram (±4): " + " ".join("%+d:%d" % kv for kv in sorted(L[4].items())),
              "rule (shift +2) matched-offset histogram (±4): " + " ".join("%+d:%d" % kv for kv in sorted(R[4].items())),
              ""]
    return "\n".join(lines), L, R


def timing(models_paths, params, windowed):
    out = {}
    m = C.load_match("041")
    frames = sum(len(r["raw"]) for r in m["rallies"])
    minutes = frames / m["fps"] / 60.0
    for nth in (1, 4):
        torch.set_num_threads(nth)
        dev = torch.device("cpu")
        models = infer.load_models(models_paths, dev)
        t0 = time.time()
        feats = [C.derive(r["raw"], r["fps"]) for r in m["rallies"]]
        t1 = time.time()
        lgs = [infer.predict_windowed(models, r["raw"], r["fps"], dev) if windowed
               else C.predict_rally(models, r["raw"], r["fps"], dev) for r in m["rallies"]]
        t2 = time.time()
        for lg, r in zip(lgs, m["rallies"]):
            C.detections_from_logits(lg, r["fps"], r["frames"], params)
        t3 = time.time()
        out[nth] = dict(feat=(t1 - t0) / minutes, model_incl_feat=(t2 - t1) / minutes,
                        decode=(t3 - t2) / minutes)
    out["minutes_of_video"] = minutes
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--gap", type=float, required=True)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--tag", default="final")
    ap.add_argument("--timing", action="store_true")
    a = ap.parse_args()
    ids_all = C.scan_ids()
    test = [i for i in C.TEST_IDS if i in ids_all]
    odir = os.path.join(C.OUT, a.tag)
    os.makedirs(odir, exist_ok=True)
    dev = torch.device("cpu")
    models = infer.load_models(a.ckpt, dev)
    P = dict(thr=a.thr, gap_s=a.gap, lam=a.lam)
    thr_logit = float(np.log(a.thr / (1 - a.thr)))
    for mid in test + C.VAL_IDS:
        m, dets, _ = infer.run_match(models, mid, P, dev, a.windowed)
        C.write_match_json(os.path.join(odir, mid + ".hits.json"), mid, m["label_dir"], m["rallies"], dets, thr_logit)
    rule_dir = os.path.join(C.RESEARCH, "hits", "out")
    txt = ["# %s: ckpts %s params %s windowed=%s" % (a.tag, a.ckpt, P, a.windowed), ""]
    t, _, _ = report("TEST", test, odir, rule_dir)
    txt.append(t)
    t, _, _ = report("VAL (tuned on this match; optimistic)", C.VAL_IDS, odir, rule_dir)
    txt.append(t)
    if a.timing:
        tm = timing(a.ckpt, P, a.windowed)
        txt.append("timing (s of CPU per minute of video, 041, %.1f min): %s" % (tm["minutes_of_video"], json.dumps(tm)))
    s = "\n".join(txt)
    print(s)
    open(os.path.join(odir, "evaluation.md"), "w").write(s)


if __name__ == "__main__":
    main()
