#!/usr/bin/env python
"""Next-serve rule vs the geometric call, on sim with truth (cached upstream): for every rally whose
next group gives a usable next-serve verdict, the geometric call before the rule (winner, confidence),
the implied winner, whether the serve was seen (held) or inferred (serve_missed), the final output and
the truth. Reports, for AGREE / CONFLICT (known geometric winner != implied), who is right and how the
output's confidence compares with its accuracy (calibration), by geometric-confidence band.

  SHUTTL_ACC_HASH=34fb5ed7d6 .venv/bin/python -m scripts.accuracy.ns_conflict --seeds 0 1000 2000
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from app.services.tracking import rally as R
from scripts.accuracy.common import load_samples, sim_args
from scripts.accuracy.rally_errors import match


def rows_for(d: dict, mod=R) -> list[dict]:
    rows, orig = [], mod._infer_from_next_serve

    def spy(cands, cfg, first_server="near"):
        st, pre = mod._Score(first_server), []
        for i, c in enumerate(cands):
            nxt = cands[i + 1] if i + 1 < len(cands) else None
            ev = None
            if nxt is not None and nxt.hits:
                h0 = nxt.hits[0]
                if (nxt.server in R.SIDES and (h0.confidence or 0.0) >= 0.3 and nxt.seg_conf >= 0.5
                        and mod._serve_launch_ok(nxt)
                        and (not cfg.next_serve_needs_hold or nxt.serve_missed or nxt.held_before)):
                    ev = {"implied": mod._implied_winner(st, nxt.server), "seen": bool(nxt.held_before and not nxt.serve_missed)}
            pre.append((c.outcome.winner, c.outcome.confidence, ev))
            if c.outcome.winner in R.SIDES:
                st.point(c.outcome.winner)
        orig(cands, cfg, first_server)
        for c, (w, q, ev) in zip(cands, pre):
            rows.append({"start": c.start_frame, "geo_w": w, "geo_q": q, "ev": ev, "out_w": c.outcome.winner,
                         "out_q": c.outcome.confidence, "src": c.outcome.source})

    mod._infer_from_next_serve = spy
    try:
        pred = mod.segment_rallies(d["track"], d["hits"], d["segs"], d["meta"], d["camera"], players=d["players"])
    finally:
        mod._infer_from_next_serve = orig
    by_start = {r["start"]: r for r in rows}
    out = []
    for i, j in match(pred, d["true_rallies"]):
        r = by_start.get(pred[i].start_frame)
        if r is not None and r["ev"] is not None and r["ev"]["implied"] in R.SIDES:
            out.append({**r, "truth": d["true_rallies"][j].winner})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1000, 2000])
    a = ap.parse_args(argv)
    agg = defaultdict(lambda: [0, 0, 0, 0.0])      # n, geo right, ns right, sum of output confidence (right: +)
    for seed in a.seeds:
        for st in ("behind", "random"):
            for d in load_samples(sim_args(st, "1920x1080", 60.0, 5), list(range(seed, seed + 40)), 1):
                for r in rows_for(d):
                    if r["geo_w"] not in R.SIDES:
                        rel = "geo unknown"
                    else:
                        rel = "AGREE" if r["geo_w"] == r["ev"]["implied"] else "CONFLICT"
                    band = "q<.55" if r["geo_q"] < 0.55 else ("q.55-.8" if r["geo_q"] < 0.8 else "q>=.8")
                    k = (rel, "seen serve" if r["ev"]["seen"] else "inferred serve", band if rel != "geo unknown" else "-")
                    g = agg[k]
                    g[0] += 1; g[1] += r["geo_w"] == r["truth"]; g[2] += r["ev"]["implied"] == r["truth"]
                    g[3] += r["out_q"]
                    agg[k + ("out right",)][0] += r["out_w"] == r["truth"]
    print("relation        serve            geo band   n   geo right  next-serve right  output right  mean output conf")
    for k in sorted(x for x in agg if len(x) == 3):
        n, gr, nr, sq = agg[k]
        orr = agg[k + ("out right",)][0]
        print(f"{k[0]:15s} {k[1]:16s} {k[2]:8s} {n:4d}   {gr / n:.3f}      {nr / n:.3f}            {orr / n:.3f}         {sq / n:.3f}")


if __name__ == "__main__":
    main()
