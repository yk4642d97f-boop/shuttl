#!/usr/bin/env python
"""Gap-contact rule through the full hits decoder on sim tracks with synthetic PTS gaps
(gap_contact_cal): per label, gap_contacts on vs off: recall of the true contact (+-3 frames, in
the gapped index), extra hits within 0.3 s after a mid-flight gap (false contacts), event F1 over
the whole gapped sample, and hitter side of recovered contacts. Players are remapped to the gapped frames.

  .venv/bin/python -m scripts.accuracy.gap_contact_eval --seed 0 --n 40 --per 250
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace

import numpy as np

from app.services.tracking import hits as H
from app.services.tracking import metrics as M
from scripts.accuracy.common import load_samples, sim_args
from scripts.accuracy.gap_contact_cal import cases, cut


def remap_players(players, keep):
    idx = {int(f): i for i, f in enumerate(keep)}
    return [replace(p, frame=idx[p.frame]) for p in players or [] if p.frame in idx]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--per", type=int, default=250)
    ap.add_argument("--ab", nargs="*", default=[], help="HitConfig overrides for the 'on' arm")
    a = ap.parse_args(argv)
    H.load_learned_hit_model(None)
    kw = {}
    for kv in a.ab:
        k, v = kv.split("=")
        cur = getattr(H.HitConfig(), k)
        kw[k] = (v == "True") if isinstance(cur, bool) else type(cur)(v)
    on, off = H.HitConfig(**kw), H.HitConfig(gap_contacts=False)
    rng = np.random.default_rng(11)
    allc = []
    for st in ("behind", "random"):
        for d in load_samples(sim_args(st, "1920x1080", 60.0, 5), list(range(a.seed, a.seed + a.n)), 1):
            allc += [(d, c) for c in cases(d, np.random.default_rng([d["seed"], 3]))]
    by = defaultdict(list)
    for d, c in allc:
        by[c[0]].append((d, c))
    res = defaultdict(lambda: defaultdict(float))
    for lab, lst in by.items():
        pick = rng.choice(len(lst), min(a.per, len(lst)), replace=False)
        for k in pick:
            d, (lab_, x0, x1, cf) = lst[k]
            tr, mt, keep = cut(d["track"], d["meta"], x0, x1)
            pl = remap_players(d["players"], keep)
            tf = [int(np.searchsorted(keep, h["frame"])) for h in d["true_hits"] if h["frame"] in set(keep.tolist())
                  or (x0 < h["frame"] < x1)]
            ts = [h["hitter"] for h in d["true_hits"] if h["frame"] in set(keep.tolist()) or (x0 < h["frame"] < x1)]
            T = [{"frame": f, "hitter": s} for f, s in zip(tf, ts)]
            g = x0 + 1                                   # first frame after the gap (gapped index)
            for arm, cfg in (("on", on), ("off", off)):
                hs = H.detect_hits(tr, mt, d["camera"], pl, cfg)
                P = [{"frame": h.frame, "hitter": h.hitter} for h in hs]
                m = M.hit_metrics(P, T, 3)
                r = res[(lab, arm)]
                r["n"] += 1; r["tp"] += m["tp"]; r["pred"] += m["n_pred"]; r["true"] += m["n_true"]
                if cf is not None:
                    c_new = int(np.searchsorted(keep, cf)) if cf in set(keep.tolist()) else g
                    near = [h for h in hs if abs(h.frame - c_new) <= 3]
                    r["recall"] += bool(near)
                    true_side = next(h["hitter"] for h in d["true_hits"] if h["frame"] == cf)
                    r["side_ok"] += bool(near) and near[0].hitter == true_side
                else:
                    r["extra"] += any(0 <= h.frame - g <= int(0.3 * 60) for h in hs)
    for (lab, arm), r in sorted(res.items()):
        n = r["n"]
        f1 = 2 * r["tp"] / max(r["pred"] + r["true"], 1)
        extra = f"recall {r['recall'] / n:.3f} side {r['side_ok'] / max(r['recall'], 1):.3f}" if lab != "mid_flight" \
            else f"false hit after the gap {r['extra'] / n:.3f}"
        print(f"{lab:15s} {arm:3s} n {int(n):4d}  hits F1 {f1:.4f}  {extra}")


if __name__ == "__main__":
    main()
