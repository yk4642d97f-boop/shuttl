#!/usr/bin/env python
"""Statistics for a rally-level hitter-side HMM (dev seeds only):
  * transitions between consecutive contacts of one rally group (after the
    same-flight merge): P(same true side | gap) for real/real pairs, and how
    often a pair involves a spurious contact;
  * per-cue calibration of the 3-D evidence, conditioned on agreement with
    the hits.py label (the fits were made WITH that label as a prior).

  .venv/bin/python -m scripts.accuracy.side_trans --style behind --seed 0 --n 40
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np

from app.services.tracking import metrics as M
from app.services.tracking import rally as R
from scripts.accuracy.common import add_common_args, load_samples, sim_args


def prepared(d: dict, cfg: R.RallyConfig):
    """hits / own segs / continuation segs / groups exactly as segment_rallies
    builds them before _fill_hitters, plus the true side of each hit (None =
    spurious)."""
    mo = R._Motion(d["track"], float(d["meta"].height), cfg)
    hs = R._clean_hits(d["hits"], mo.n)
    hs = [h if h.t is not None and np.isfinite(h.t) else R.replace(h, t=float(mo.t[h.frame])) for h in hs]
    segs = R._match_segments(hs, list(d["segs"] or []), mo.t, cfg)
    hs, segs, conts = R._merge_same_flight(hs, segs, mo.t, cfg)
    groups, cues = R._group_hits(hs, segs, mo, cfg, conts)
    th = d["true_hits"]
    tf = np.array([h["frame"] for h in th], int)
    pf = np.array([h.frame for h in hs], int)
    truth: list[str | None] = [None] * len(hs)
    for i, j in M._match_events(pf, tf, 3):
        truth[i] = th[j]["hitter"]
    return mo, hs, segs, conts, groups, truth


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    args = ap.parse_args(argv)
    a = sim_args(args.style, args.image_size, args.fps, args.rallies)
    samples = load_samples(a, list(range(args.seed, args.seed + args.n)), args.workers)
    cfg = R.RallyConfig()
    tr = defaultdict(lambda: [0, 0])
    kinds = Counter()
    cal = defaultdict(lambda: [0, 0])      # key -> [n, label_wrong]
    for d in samples:
        mo, hs, segs, conts, groups, truth = prepared(d, cfg)
        for g in groups:
            for p, q in zip(g[:-1], g[1:]):
                gap = hs[q].t - hs[p].t
                if truth[p] is None or truth[q] is None:
                    kinds["pair_with_spurious"] += 1
                    continue
                kinds["real_pair"] += 1
                b = min(int(gap / 0.3), 8)
                tr[b][0] += 1
                tr[b][1] += truth[p] == truth[q]
        for i, h in enumerate(hs):
            if truth[i] is None or h.hitter not in R.SIDES:
                continue
            wrong = h.hitter != truth[i]
            sg = 1.0 if h.hitter == "near" else -1.0            # label's launch direction (+x)
            s = segs[i]
            if s is not None:
                vx = float(s.v0[0]) * sg                           # + = agrees with label
                b = "agree" if vx > 0 else "conflict"
                m = abs(vx)
                mb = "<1" if m < 1 else ("<3" if m < 3 else (">=3"))
                cal[f"vx_own {b} {mb}"][0] += 1
                cal[f"vx_own {b} {mb}"][1] += wrong
                x = -float(s.p0[0]) * sg                           # + = on the label's half
                b = "agree" if x > 0 else "conflict"
                m = abs(x)
                mb = "<0.5" if m < 0.5 else ("<1.5" if m < 1.5 else ">=1.5")
                cal[f"x_own {b} {mb}"][0] += 1
                cal[f"x_own {b} {mb}"][1] += wrong
            if i > 0:
                p = conts[i - 1] if conts[i - 1] is not None else segs[i - 1]
                if p is not None and float(p.t0) < h.t:
                    t_cov = float(mo.t[min(max(int(p.end_frame), 0), mo.n - 1)]) + 0.25
                    if h.t <= t_cov:
                        st = R._state_at(p, h.t)
                        x = -st[0] * sg
                        b = "agree" if x > 0 else "conflict"
                        m = abs(x)
                        mb = "<0.5" if m < 0.5 else ("<1.5" if m < 1.5 else ">=1.5")
                        cal[f"x_in {b} {mb}"][0] += 1
                        cal[f"x_in {b} {mb}"][1] += wrong
    print(kinds)
    for b in sorted(tr):
        n, same = tr[b]
        print(f"gap [{0.3 * b:.1f},{0.3 * (b + 1):.1f}) n={n:5d} P(same side)={same / max(n, 1):.3f}")
    for k in sorted(cal):
        n, w = cal[k]
        print(f"{k:28s} n={n:5d} P(label wrong)={w / max(n, 1):.3f}")


if __name__ == "__main__":
    main()
