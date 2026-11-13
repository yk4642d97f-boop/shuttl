#!/usr/bin/env python
"""Calibration set for the gap-contact rule (dev seeds only): PTS gaps cut into cached sim tracks
(frames removed, real clock kept, so dt jumps by the gap). Positives: the gap ends AT a true contact
(F4 pattern: the contact is the first frame after the gap) or hides it inside; negatives: the gap
lies inside a flight (>= 0.25 s from any contact on both sides). Returns gapped tracks + labels.

  .venv/bin/python -m scripts.accuracy.gap_contact_cal --seed 0 --n 40
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from app.services.tracking.data_types import Track2D, VideoMeta


def cut(track: Track2D, meta: VideoMeta, a: int, b: int):
    """Remove frames a+1 .. b-1 (keep a and b): a PTS gap between a and b."""
    n = len(track.t)
    keep = np.r_[0:a + 1, b:n]
    tr = Track2D(t=np.asarray(track.t)[keep], xy=np.asarray(track.xy)[keep], conf=np.asarray(track.conf)[keep],
                 visible=np.asarray(track.visible)[keep], interpolated=np.asarray(track.interpolated)[keep])
    if getattr(track, "raw_xy", None) is not None:
        tr.raw_xy = np.asarray(track.raw_xy)[keep]
    mt = replace(meta, n_frames=len(keep), timestamps=np.asarray(meta.timestamps)[keep])
    return tr, mt, keep


def cases(d: dict, rng, gap_s=(0.35, 0.5, 0.7)):
    """(label, a, b, contact_frame_or_None) per case in one sample."""
    fps = float(d["meta"].fps or 60.0)
    tf = sorted(int(h["frame"]) for h in d["true_hits"])
    n = len(d["track"].t)
    out = []
    for k, f in enumerate(tf):
        L = int(round(float(rng.choice(gap_s)) * fps))
        if f - L - 1 > 0 and (k == 0 or tf[k - 1] < f - L - int(0.2 * fps)):
            out.append(("contact_at_end", f - L - 1, f, f))          # contact = first frame after the gap
        a = f - int(rng.integers(2, max(3, L // 2)))
        if a > 0 and a + L < n and (k == 0 or tf[k - 1] < a) and (k + 1 == len(tf) or tf[k + 1] > a + L + int(0.2 * fps)):
            out.append(("contact_inside", a, a + L, f))
    for f0, f1 in zip(tf[:-1], tf[1:]):
        L = int(round(float(rng.choice(gap_s)) * fps))
        lo, hi = f0 + int(0.25 * fps), f1 - int(0.25 * fps) - L
        if hi > lo:
            a = int(rng.integers(lo, hi + 1))
            out.append(("mid_flight", a, a + L, None))
    return out


def main(argv=None):
    from scripts.accuracy.common import load_samples, sim_args
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(7)
    tot = {}
    for st in ("behind", "random"):
        for d in load_samples(sim_args(st, "1920x1080", 60.0, 5), list(range(a.seed, a.seed + a.n)), 1):
            for c in cases(d, rng):
                tot[c[0]] = tot.get(c[0], 0) + 1
    print(tot)


if __name__ == "__main__":
    main()
