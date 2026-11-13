#!/usr/bin/env python
"""Hitter-side error structure of hits.py on the cached sim e2e chain.

Per matched hit (pred within 3 frames of a true hit): true / predicted side,
posterior, unary cue parts (player reach LLR, camera-ray LLR), rally segment
(hits.py `seg`) and whether the neighbours in the decoded chain are real
(matched) contacts. Errors are classed per decoded segment:
  whole_flip  every matched hit of the segment is wrong (phase error)
  partial     some wrong: after a missed / spurious neighbour (phase slip) or isolated

  .venv/bin/python -m scripts.accuracy.side_errors --style behind --seed 0 --n 40
"""
from __future__ import annotations

import argparse
import math
from collections import Counter

import numpy as np

from app.services.tracking import hits as H
from scripts.accuracy.common import add_common_args, load_samples, sim_args


def decode(d: dict, cfg: H.HitConfig | None = None):
    """Candidates, accepted chain indices and P(near) exactly as detect_hits."""
    cfg = cfg or H.HitConfig()
    track = H._prefer_raw(d["track"]) if cfg.use_raw_xy else d["track"]
    cands = H.detect_candidates(track, d["meta"], d["camera"], d["players"], cfg)
    if not cands:
        return cands, [], np.zeros(0), None
    meta, camera, players = d["meta"], d["camera"], d["players"]
    diag = math.hypot(meta.width or 1280, meta.height or 720)
    pair_w = cfg.pair_y_weight * (0.5 if (camera is not None or players) else 1.0)
    dec = H._Decoder(cands, cfg, diag, pair_w)
    chain = [(k, s) for k, s in dec.viterbi() if dec.hit[k] >= cfg.min_hit_logodds]
    idx = [k for k, _ in chain]
    return cands, idx, dec.posteriors(idx), dec


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
        th = d["true_hits"]
        tf = np.array([h["frame"] for h in th])
        pf = np.array([cands[k].frame for k in idx])
        from app.services.tracking.metrics import _match_events
        pairs = dict(_match_events(pf, tf, 3))                 # chain pos -> true idx
        segs: dict[int, list[int]] = {}
        for p, k in enumerate(idx):
            segs.setdefault(cands[k].seg, []).append(p)
        for sg, ps in segs.items():
            m = [p for p in ps if p in pairs]
            if not m:
                continue
            wrong = []
            for p in m:
                k, j = idx[p], pairs[p]
                side = "near" if pn[p] >= 0.6 else ("far" if pn[p] <= 0.4 else "unknown")
                wrong.append(side != th[j]["hitter"])
            nw = sum(wrong)
            c["hits"] += len(m)
            c["wrong"] += nw
            c["segs"] += 1
            if nw == 0:
                continue
            if nw == len(m):
                c["whole_flip_segs"] += 1
                c["whole_flip_hits"] += nw
                cls = "whole_flip"
            else:
                c["partial_segs"] += 1
                c["partial_hits"] += nw
                cls = "partial"
            for q, p in enumerate(m):
                if not wrong[q]:
                    continue
                k, j = idx[p], pairs[p]
                cd = cands[k]
                ray = H._ray_side_llr(d["camera"], cd.xy) if np.all(np.isfinite(cd.xy)) else 0.0
                ply = cd.side_llr - ray
                sgn = 1 if th[j]["hitter"] == "near" else -1
                pos = ps.index(p)
                prev_ok = pos == 0 or (ps[pos - 1] in pairs and pairs[ps[pos - 1]] == j - 1)
                next_ok = pos == len(ps) - 1 or (ps[pos + 1] in pairs and pairs[ps[pos + 1]] == j + 1)
                c[f"{cls}: ply={'ok' if ply * sgn > 0.5 else ('bad' if ply * sgn < -0.5 else 'weak')}"] += 1
                c[f"{cls}: ray={'ok' if ray * sgn > 0.5 else ('bad' if ray * sgn < -0.5 else 'weak')}"] += 1
                if cls == "partial":
                    c[f"partial: prev_ok={prev_ok} next_ok={next_ok}"] += 1
                rows.append((d["seed"], sg, cls, th[j]["hitter"], round(float(pn[p]), 2),
                             round(ply, 2), round(ray, 2), cd.frame, len(m), nw, prev_ok, next_ok,
                             round(cd.tau, 2)))
    for k, v in sorted(c.items()):
        print(f"{v:6d} {k}")
    print(f"side acc over matched chain hits: {1 - c['wrong'] / max(c['hits'], 1):.3f}")
    for r in rows[:args.dump]:
        print(r)


if __name__ == "__main__":
    main()
