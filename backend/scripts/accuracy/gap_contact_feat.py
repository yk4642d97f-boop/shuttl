#!/usr/bin/env python
"""Features of the flight across a PTS gap for the gap-contact rule, on gap_contact_cal cases (dev)
and on real_drop.mov: lam_v = cubic vs cubic + velocity kink at the first observation after the gap
(hits._fit_batch / _lambda on K observations each side), and the pre-gap fit extrapolated to that
observation vs the post-gap start (position error / image height, speed ratio, direction cosine).

  .venv/bin/python -m scripts.accuracy.gap_contact_feat --seed 0 --n 40
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from app.services.tracking import hits as H
from app.services.tracking.video_io import pts_gaps
from scripts.accuracy.gap_contact_cal import cases, cut


def gap_features(track, meta, K: int = 6):
    """Per PTS gap: dict of features (None where < 3 observations on a side)."""
    tt = np.asarray(track.t, float)
    tr = H._prefer_raw(track)
    oi = H._observed(tr)
    xy = np.asarray(tr.xy, float)[oi]
    keep, t = H._content_clock(oi, xy, tt)
    oi, t, xy = oi[keep], t[keep], xy[keep]
    sigma = H._estimate_sigma(t, xy)
    hgt = float(meta.height or 720)
    out = []
    for g in pts_gaps(tt):
        m = int(np.searchsorted(oi, g))                # first kept observation at/after the gap frame
        f = {"gap_frame": int(g), "gap_s": float(tt[g] - tt[g - 1]), "m_frame": int(oi[m]) if m < len(oi) else None,
             "first_after_s": float(t[m] - tt[g]) if m < len(oi) else None}
        nl, nr = min(K, m), min(K, len(t) - m)
        f["nl"], f["nr"] = nl, nr
        if nl >= 3 and nr >= 3:
            lo, hi = m - nl, m + nr
            tau = np.array([0.5 * (t[m - 1] + t[m])])
            fit = H._fit_batch(t, xy, tau, np.array([lo]), np.array([hi]), np.array([m]), 0.15)
            lam, lam_v, s2 = H._lambda(fit, sigma)
            f["lam_v"], f["lam"] = float(lam_v[0]), float(lam[0])
            A = np.polyfit(t[lo:m] - t[m - 1], xy[lo:m], 2)            # pre-gap, extrapolated
            Bq = np.polyfit(t[m:hi] - t[m], xy[m:hi], 2)                # post-gap start
            dt = t[m] - t[m - 1]
            p_pred = np.array([np.polyval(A[:, j], dt) for j in range(2)])
            v_pred = np.array([np.polyval(np.polyder(A[:, j]), dt) for j in range(2)])
            v_end = np.array([np.polyval(np.polyder(A[:, j]), 0.0) for j in range(2)])
            v_post = Bq[1]
            f["pos_err_h"] = float(np.linalg.norm(p_pred - Bq[2]) / hgt)
            f["speed_ratio"] = float(np.linalg.norm(v_post) / max(np.linalg.norm(v_end), 1e-6))
            f["cos_pred"] = float(v_pred @ v_post / (np.linalg.norm(v_pred) * np.linalg.norm(v_post) + 1e-9))
            f["cos_end"] = float(v_end @ v_post / (np.linalg.norm(v_end) * np.linalg.norm(v_post) + 1e-9))
            f["v_end_h"] = float(np.linalg.norm(v_end) / hgt)
        out.append(f)
    return out


def main(argv=None):
    from scripts.accuracy.common import load_samples, sim_args
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--dump", default="/tmp/accC/gap_feats.json")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(7)
    rows = []
    for st in ("behind", "random"):
        for d in load_samples(sim_args(st, "1920x1080", 60.0, 5), list(range(a.seed, a.seed + a.n)), 1):
            for lab, x0, x1, cf in cases(d, rng):
                tr, mt, keep = cut(d["track"], d["meta"], x0, x1)
                for f in gap_features(tr, mt):
                    f.update(label=lab, style=st, seed=d["seed"], a=x0, b=x1,
                             contact_new=None if cf is None else int(np.searchsorted(keep, cf)))
                    rows.append(f)
    import json
    json.dump(rows, open(a.dump, "w"))
    print("rows", len(rows), {k: sum(r["label"] == k for r in rows) for k in ("contact_at_end", "contact_inside", "mid_flight")})


if __name__ == "__main__":
    main()
