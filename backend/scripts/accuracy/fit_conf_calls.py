#!/usr/bin/env python
"""Outcome call vs final-flight fit confidence: per predicted rally the current call (3-D
unless no fit), the 2-D rest-point call (rally._outcome_2d on the same last hit / rest frame)
and the fit confidence. Sim (truth): winner accuracy per bin, 3-D vs 2-D. 45-min real runs
(no truth): H.264 vs HEVC agreement per bin for both calls.

  SHUTTL_ACC_HASH=7f43851d1f .venv/bin/python -m scripts.accuracy.fit_conf_calls
"""
from __future__ import annotations

import numpy as np

from app.services.tracking import rally as R
from scripts.accuracy.codec_ab import load, pair, rallies
from scripts.accuracy.common import load_samples, sim_args
from scripts.accuracy.rally_errors import match

BINS = (0.0, 0.05, 0.1, 0.2, 0.3, 1.01)


def calls(d: dict, pred: list) -> list[dict]:
    cfg = R.RallyConfig()
    mo = R._Motion(d["track"], float(d["meta"].height), cfg)
    out = []
    for r in pred:
        seg = next((s.segment for s in reversed(r.shots) if s.segment is not None), None)
        c = {"fit": np.nan if seg is None else float(seg.confidence or 0.0), "w3": r.winner, "w2": "unknown",
             "src": r.outcome_source}
        if r.shots and d["camera"] is not None:
            last = r.shots[-1].hit
            f_rest = mo.first_rest_after(last.frame, min(mo.n, mo.frame_at(last.t + 4.0) + 1))
            o = R._outcome_2d(last, f_rest, mo, d["camera"], None, cfg)
            c["w2"] = o.winner
        out.append(c)
    return out


def binned(vals, fits):
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        sel = [v for v, f in zip(vals, fits) if lo <= f < hi]
        rows.append(f"[{lo:.2f},{hi:.2f}) {np.mean(sel) if sel else float('nan'):.3f} n{len(sel)}")
    return "  ".join(rows)


def main():
    for st in ("behind", "random"):
        acc3, acc2, fits = [], [], []
        for seed in (0, 1000):
            for d in load_samples(sim_args(st, "1920x1080", 60.0, 5), list(range(seed, seed + 40)), 1):
                pred = R.segment_rallies(d["track"], d["hits"], d["segs"], d["meta"], d["camera"], players=d["players"])
                cs = calls(d, pred)
                for i, j in match(pred, d["true_rallies"]):
                    tw = d["true_rallies"][j].winner
                    if np.isfinite(cs[i]["fit"]):
                        fits.append(cs[i]["fit"]); acc3.append(cs[i]["w3"] == tw); acc2.append(cs[i]["w2"] == tw)
        print(f"sim {st} (dev+held-out) winner acc by final-fit conf\n   3-D/current: {binned(acc3, fits)}\n   2-D rest   : {binned(acc2, fits)}")
    A, B = load("/tmp/accC/codec/h264.json"), load("/tmp/accC/codec/hevc.json")
    ra, rb = rallies(A), rallies(B)
    ca, cb = calls(A, ra), calls(B, rb)
    ag3, ag2, fits = [], [], []
    for i, j in pair(ra, rb, A["meta"].fps):
        f = min(ca[i]["fit"], cb[j]["fit"])
        if np.isfinite(f):
            fits.append(f); ag3.append(ca[i]["w3"] == cb[j]["w3"]); ag2.append(ca[i]["w2"] == cb[j]["w2"])
    print(f"45-min H.264 vs HEVC winner agreement by min(final-fit conf)\n   3-D/current: {binned(ag3, fits)}\n   2-D rest   : {binned(ag2, fits)}")
    k2 = sum(c["w2"] in R.SIDES for c in ca)
    print(f"   2-D call available (known winner) on {k2}/{len(ca)} H.264 rallies")


if __name__ == "__main__":
    main()
